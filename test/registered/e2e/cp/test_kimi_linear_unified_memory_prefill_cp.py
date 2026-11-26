"""Prefix reuse with replicated MLA KV under zigzag context-parallel prefill."""

import random
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_test_utils import _flush_cache, _generate
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=100, stage="extra-a", runner_config="2-gpu-large")


class TestKimiLinearUnifiedMemoryPrefillCP(DefaultServerBase):
    model = "yujiepan/kimi-linear-tiny-random"
    other_args = [
        "--trust-remote-code",
        "--skip-tokenizer-init",
        "--tp-size",
        "2",
        "--enable-prefill-cp",
        "--attn-cp-size",
        "2",
        "--cp-strategy",
        "zigzag",
        "--enable-unified-memory",
        "--attention-backend",
        "fa3",
        "--linear-attn-backend",
        "triton",
        "--page-size",
        "64",
        "--max-running-requests",
        "8",
        "--max-total-tokens",
        "8192",
        "--max-mamba-cache-size",
        "32",
        "--context-length",
        "2048",
        "--chunked-prefill-size",
        "1024",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--cuda-graph-backend-decode",
        "disabled",
        "--enable-cache-report",
    ]

    def test_prefix_hit_matches_cold_prefill(self):
        rng = random.Random(680)
        first = [rng.randint(101, 15000) for _ in range(513)]
        prefix = first[:257]
        prompt = prefix + [15021] * 256

        def generate(ids):
            return _generate(self.base_url, ids, max_new_tokens=4, return_logprob=True)[
                "meta_info"
            ]

        # The second branch checkpoints the shared prefix at a page boundary.
        # The third has a fresh suffix long enough for both CP ranks to compute.
        generate(first)
        generate(prefix + [15019] * 256)
        warm = generate(prompt)
        self.assertEqual(warm["cached_tokens_details"]["device"], 256)

        _flush_cache(self.base_url)
        cold = generate(prompt)
        self.assertEqual(cold["cached_tokens"], 0)
        cold_tokens = cold["output_token_logprobs"]
        warm_tokens = warm["output_token_logprobs"]
        self.assertEqual(len(cold_tokens), 4)
        for (cold_lp, cold_id, _), (warm_lp, warm_id, _) in zip(
            cold_tokens, warm_tokens, strict=True
        ):
            self.assertEqual(cold_id, warm_id)
            self.assertAlmostEqual(cold_lp, warm_lp, delta=0.05)


if __name__ == "__main__":
    unittest.main()
