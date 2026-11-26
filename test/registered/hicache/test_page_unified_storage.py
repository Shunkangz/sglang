"""Restore page_unified KV from L2 under CP and share L3 objects across instances.

Run with four CUDA GPUs. SGLANG_TEST_PAGE_UNIFIED_MODEL can point at a local
Qwen3-0.6B checkpoint. These smoke cases keep attention-TP and PP layer ranges
fixed to exercise CP writer election. For TP/PP reshaping accuracy, see
test_page_unified_qwen3_30b_accuracy.py.
GLM-5.2's indexer sidecar and DeepSeek-V4's composite pools are not supported by
the unified storage key scheme yet.
"""

import os
import random
import shutil
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path

import requests

from sglang.srt.utils.device import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_test_utils import compare_kl_divergence
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=450, stage="extra-b", runner_config="4-gpu-h100")


class TestPageUnifiedStorage(CustomTestCase):
    model = "Qwen/Qwen3-0.6B"
    page_size = 128
    prompt_tokens = 1024

    @contextmanager
    def _server(self, work_dir, *, pp_size, cp_size, label, use_l3=True):
        base_url = f"http://127.0.0.1:{find_available_port(23337)}"
        args = [
            "--tp-size",
            str(cp_size),
            "--pp-size",
            str(pp_size),
            "--attention-backend",
            "fa4" if is_sm100_supported() else "fa3",
            "--mem-fraction-static",
            "0.6",
            "--max-total-tokens",
            "8192",
            "--max-running-requests",
            "4",
            "--chunked-prefill-size",
            "2048",
            "--cuda-graph-backend-decode=disabled",
            "--cuda-graph-backend-prefill=disabled",
            "--page-size",
            str(self.page_size),
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "2",
            "--hicache-write-policy",
            "write_through",
            "--hicache-io-backend",
            "kernel",
            "--hicache-mem-layout",
            "page_unified",
            "--enable-cache-report",
        ]
        if use_l3:
            args += [
                "--hicache-storage-key-scheme",
                "unified",
                "--hicache-storage-backend",
                "file",
                "--hicache-storage-prefetch-policy",
                "wait_complete",
            ]
        if cp_size > 1:
            args += [
                "--attn-cp-size",
                str(cp_size),
                "--enable-prefill-cp",
                "--cp-strategy",
                "zigzag",
            ]

        log_path = work_dir / f"{label}.log"
        print(f"{label}: TP={cp_size}, CP={cp_size}, PP={pp_size}; {log_path}")
        with log_path.open("w") as log_file:
            process = popen_launch_server(
                os.environ.get("SGLANG_TEST_PAGE_UNIFIED_MODEL", self.model),
                base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=args,
                env={
                    "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": str(work_dir / "l3"),
                    "SGLANG_ENABLE_RANK_CONSENSUS_CHECKER": "1",
                },
                return_stdout_stderr=(log_file, log_file),
            )
            try:
                yield base_url
            finally:
                terminate_and_kill_process_tree(process, wait_timeout=60)

    def _generate(self, base_url, input_ids, *, cache_salt=None, continuation=()):
        response = requests.post(
            f"{base_url}/generate",
            json={
                "input_ids": [*input_ids, *continuation],
                "cache_salt": cache_salt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1 if continuation else 8,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "logprob_start_len": len(input_ids) - 1,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def _wait_for_backup(self, work_dir, pp_size):
        # .bin files are published by atomic rename. Wait for every prompt page
        # in every PP stage before terminating the only instance holding L1/L2.
        expected_pages = self.prompt_tokens // self.page_size * pp_size
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            files = list((work_dir / "l3").rglob("*.bin"))
            if len(files) >= expected_pages:
                self.assertTrue(all(path.stat().st_size > 0 for path in files))
                return
            time.sleep(0.1)
        self.fail(f"Only {len(files)} L3 objects published; expected {expected_pages}")

    def _assert_cross_instance_reuse(self, *, pp_size, writer_cp, reader_cp):
        work_dir = Path(tempfile.mkdtemp(prefix="page_unified_e2e_"))
        (work_dir / "l3").mkdir()
        succeeded = False
        rng = random.Random(39728)
        input_ids = [rng.randint(100, 30000) for _ in range(self.prompt_tokens)]
        try:
            with self._server(
                work_dir, pp_size=pp_size, cp_size=writer_cp, label="writer"
            ) as base_url:
                initial = self._generate(base_url, input_ids)["meta_info"]
                self.assertEqual(initial["cached_tokens"], 0)
                continuation = [
                    token for _, token, _ in initial["output_token_logprobs"]
                ]
                self.assertEqual(len(continuation), 8)
                self._wait_for_backup(work_dir, pp_size)

            # This is a fresh process with fresh GPU and host pools. Its first
            # request can only reuse data through the shared file backend.
            with self._server(
                work_dir, pp_size=pp_size, cp_size=reader_cp, label="reader"
            ) as base_url:
                # Score the same continuation under both topologies: a BF16
                # tie can change a greedy token and all tokens that follow it.
                cached = self._generate(base_url, input_ids, continuation=continuation)[
                    "meta_info"
                ]
                details = cached["cached_tokens_details"]
                self.assertEqual(details["device"], 0)
                self.assertEqual(details["host"], 0)
                self.assertEqual(details["storage_backend"], "HiCacheFile")
                self.assertGreaterEqual(
                    details["storage"], self.prompt_tokens - self.page_size
                )
                self.assertEqual(cached["cached_tokens"], details["storage"])

                # Build a reference prefix on the reader and reuse it from L1.
                cold = self._generate(
                    base_url, input_ids, cache_salt="fresh-reference"
                )["meta_info"]
                self.assertEqual(cold["cached_tokens"], 0)
                resident = self._generate(
                    base_url,
                    input_ids,
                    cache_salt="fresh-reference",
                    continuation=continuation,
                )["meta_info"]
                self.assertEqual(resident["cached_tokens"], details["storage"])
                self.assertEqual(
                    resident["cached_tokens_details"]["device"], details["storage"]
                )
                # The first entry is the prompt's last token, which has no
                # preceding logprob at logprob_start_len.
                cached_probs = cached["input_token_logprobs"][1:]
                resident_probs = resident["input_token_logprobs"][1:]
                self.assertEqual([entry[1] for entry in cached_probs], continuation)
                self.assertEqual([entry[1] for entry in resident_probs], continuation)
                compare_kl_divergence(
                    [[entry[0] for entry in resident_probs]],
                    [[entry[0] for entry in cached_probs]],
                    {self.model: {"kl_div": 0.005}},
                    self.model,
                    f"page_unified PP={pp_size}, CP={writer_cp}->{reader_cp}",
                )
                print(
                    f"PP={pp_size}, CP={writer_cp}->{reader_cp}: "
                    f"{details['storage']} L3 tokens, continuation logprobs match"
                )
            succeeded = True
        finally:
            if succeeded:
                shutil.rmtree(work_dir)
            else:
                print(f"Retained server logs and L3 objects at {work_dir}")

    def test_tp_to_cp(self):
        self._assert_cross_instance_reuse(pp_size=1, writer_cp=1, reader_cp=2)

    def test_cp_to_tp_with_pp(self):
        self._assert_cross_instance_reuse(pp_size=2, writer_cp=2, reader_cp=1)

    def test_cp_restores_prefix_from_host(self):
        rng = random.Random(39728)
        prefix = [rng.randint(100, 30000) for _ in range(self.prompt_tokens)]
        continuation = prefix + [rng.randint(100, 30000) for _ in range(193)]
        with tempfile.TemporaryDirectory(prefix="page_unified_l2_") as work_dir:
            with self._server(
                Path(work_dir), pp_size=1, cp_size=2, label="l2", use_l3=False
            ) as base_url:
                # Use independent prefixes with identical contents so both
                # continuations compute the same suffix under zigzag CP.
                self._generate(base_url, prefix, cache_salt="resident")
                resident = self._generate(
                    base_url, continuation, cache_salt="resident"
                )["meta_info"]
                self.assertEqual(
                    resident["cached_tokens_details"]["device"], len(prefix)
                )
                seed = self._generate(base_url, prefix, cache_salt="offloaded")
                self.assertEqual(seed["meta_info"]["cached_tokens"], 0)

                # Two unrelated 4096-token prompts fill the 8192-token device
                # pool, while both target prefixes fit within the 2x host pool.
                for _ in range(2):
                    filler = [rng.randint(100, 30000) for _ in range(4096)]
                    self._generate(base_url, filler)

                restored = self._generate(
                    base_url, continuation, cache_salt="offloaded"
                )["meta_info"]
                details = restored["cached_tokens_details"]
                self.assertEqual(details["device"], 0)
                self.assertEqual(details["host"], len(prefix))
                self.assertEqual(restored["cached_tokens"], details["host"])
                self.assertEqual(
                    restored["output_token_logprobs"],
                    resident["output_token_logprobs"],
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
