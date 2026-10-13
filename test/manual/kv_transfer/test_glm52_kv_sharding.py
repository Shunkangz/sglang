r"""Compare GLM-5.2 prefill or PD generation with and without KV sharding.

This is a manual HTTP client, not a pytest test or a server launcher. It flushes
both servers' caches before each pass. Use identical model, tokenizer, attention
backends, parallelism, and chunk size for the two passes.

Example prefill-only workflow (adjust TP and memory settings for your hardware)::

    export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
    export SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0
    export SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC=1
    python -m sglang.launch_server --model-path zai-org/GLM-5.2-FP8 \
        --tp 8 --disaggregation-mode prefill --port 30000 \
        --disaggregation-bootstrap-port 8998 --attention-backend dsa \
        --dsa-prefill-backend flashmla_sparse --dsa-topk-backend flashinfer \
        --kv-cache-dtype bfloat16 \
        --page-size 64 --chunked-prefill-size 2048
    python test/manual/kv_transfer/test_glm52_kv_sharding.py \
        --url http://127.0.0.1:30000 --bootstrap-port 8998 \
        --output baseline.json

The dense-attention threshold is disabled for both runs so their sparse
attention kernels match. Dense MHA and MLA use different projection orders,
which can change NVFP4 rounding independently of KV sharding. Deterministic
FlashInfer top-k avoids the legacy CUDA top-k kernel's varying output order for
contexts over 2048 tokens; that ordering can also change attention reductions.
Use the same top-k settings on the decode server.

Restart the prefill server with the same arguments, adding
``--enable-kv-cache-sharding`` and changing its bootstrap port to 8999. Then run::

    python test/manual/kv_transfer/test_glm52_kv_sharding.py \
        --url http://127.0.0.1:30000 --bootstrap-port 8999 \
        --output sharded.json --reference baseline.json --logprob-atol 0.05

For full PD verification, also launch a compatible decode server on separate
GPUs, using the same checkpoint, KV dtype, page size, and attention TP size.
Add ``--decode-url http://127.0.0.1:30001`` to BOTH client passes. This compares
up to 16 generated tokens instead of one prefill token. Change the prefill bootstrap
port between passes (or restart decode): decode caches prefill topology by
bootstrap address. ``--bootstrap-host`` must be reachable from decode.

The prompts cover sparse selection beyond index_topk=2048, multiple prefill
chunks, prefix reuse, a branched prefix, and concurrent mixed-length requests
submitted as one batch.
Baseline collection repeats once to check numerical repeatability before
comparing sharding. Use --skip-baseline-repeat only when repeatability is already
known. Generated token IDs must match exactly. Output token logprobs must be finite
and differ by at most --logprob-atol; the maximum difference is reported.
Input/top logprobs are saved for inspection, but input rows are not compared
because prefix-cache hits change which rows are returned. Use --model-path or
--tokenizer-path for local checkpoints, including GLM-5.2-NVFP4.
"""

import argparse
import concurrent.futures
import json
import math
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


def build_prompts(tokenizer):
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    paragraph = encode(
        "A warehouse has 37 red boxes and 19 blue boxes. Each box contains 12 pencils. "
        "After selling 144 pencils, the remaining pencils are split equally between "
        "8 classrooms. Work through the arithmetic carefully.\n"
    )
    tokens = (paragraph * math.ceil(6000 / len(paragraph)))[:6000]
    suffix = encode("\nThe final number of pencils per classroom is")
    return {
        "short": tokens[:257] + suffix,
        "beyond_topk": tokens[:2305] + suffix,
        "chunked": tokens[:4353] + suffix,
        "cached_repeat": tokens[:4353] + suffix,
        "cached_branch": tokens[:4352] + encode("\nExplain the number of red boxes."),
        "batch_short": encode("Write the result of 12 times 13:"),
        "batch_cached": tokens[:4417] + suffix,
        "batch_long": encode("An independent exercise.\n") + tokens[:3137] + suffix,
    }


def compare_results(results, reference, logprob_atol):
    if results.keys() != reference.keys():
        raise AssertionError("Reference and result prompt names differ")
    max_delta = 0.0
    compared_tokens = 0
    for name, data in results.items():
        actual = data["meta_info"]["output_token_logprobs"]
        expected = reference[name]["meta_info"]["output_token_logprobs"]
        if not actual or [x[1] for x in actual] != [x[1] for x in expected]:
            raise AssertionError(
                f"{name}: generated token IDs differ: {actual} != {expected}"
            )
        for actual_lp, expected_lp in zip(actual, expected):
            delta = abs(actual_lp[0] - expected_lp[0])
            if not math.isfinite(delta) or delta > logprob_atol:
                raise AssertionError(
                    f"{name}: output logprob delta {delta} exceeds {logprob_atol}: "
                    f"{actual_lp} != {expected_lp}"
                )
            max_delta = max(max_delta, delta)
            compared_tokens += 1
    return {
        "passed": len(results),
        "compared_output_tokens": compared_tokens,
        "max_output_logprob_delta": max_delta,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-path", default="zai-org/GLM-5.2-FP8")
    parser.add_argument("--tokenizer-path", help="Defaults to --model-path")
    parser.add_argument("--url", "--prefill-url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--decode-url", help="Enable full PD generation against this server"
    )
    parser.add_argument(
        "--bootstrap-host",
        help="Prefill host reachable from decode; defaults to --url's host",
    )
    parser.add_argument("--bootstrap-port", type=int, default=8998)
    parser.add_argument("--output", "--results", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--skip-baseline-repeat",
        action="store_true",
        help="Skip the second baseline pass that checks numerical repeatability",
    )
    parser.add_argument("--logprob-atol", type=float, default=0.05)
    parser.add_argument(
        "--max-new-tokens", type=int, help="Defaults to 16 with decode, otherwise 1"
    )
    parser.add_argument("--request-timeout", type=float, default=900)
    args = parser.parse_args()
    if not math.isfinite(args.logprob_atol) or args.logprob_atol < 0:
        parser.error("--logprob-atol must be finite and nonnegative")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if not args.decode_url and args.max_new_tokens not in (None, 1):
        parser.error("Prefill-only verification requires --max-new-tokens 1")
    reference = json.loads(args.reference.read_text()) if args.reference else None

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.model_path, trust_remote_code=True
    )
    prompts = build_prompts(tokenizer)
    for url in (args.url, args.decode_url):
        if url:
            requests.post(
                url.rstrip("/") + "/flush_cache", timeout=60
            ).raise_for_status()
    room_base = time.time_ns() % (1 << 50)
    bootstrap_host = (
        (args.bootstrap_host or urlparse(args.url).hostname)
        if args.decode_url
        else "2.2.2.2"
    )

    def generate(group):
        names, ids = zip(*group)
        batched = len(group) > 1
        rooms = [room_base + list(prompts).index(name) for name in names]
        payload = {
            "input_ids": list(ids) if batched else ids[0],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": args.max_new_tokens or (16 if args.decode_url else 1),
            },
            "return_logprob": True,
            "logprob_start_len": (
                [len(tokens) - 1 for tokens in ids] if batched else len(ids[0]) - 1
            ),
            "top_logprobs_num": 10,
            "bootstrap_host": bootstrap_host,
            "bootstrap_port": args.bootstrap_port,
            "bootstrap_room": rooms if batched else rooms[0],
        }
        prefill_meta_info = None
        if args.decode_url:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pair:
                decode = pair.submit(
                    requests.post,
                    args.decode_url.rstrip("/") + "/generate",
                    json=payload,
                    timeout=args.request_timeout,
                )
                prefill = requests.post(
                    args.url.rstrip("/") + "/generate",
                    json=payload,
                    timeout=args.request_timeout,
                )
                prefill.raise_for_status()
                prefill_data = prefill.json()
                prefill_meta_info = [
                    value["meta_info"]
                    for value in (prefill_data if batched else [prefill_data])
                ]
                response = decode.result()
        else:
            response = requests.post(
                args.url.rstrip("/") + "/generate",
                json=payload,
                timeout=args.request_timeout,
            )
        response.raise_for_status()
        data = response.json()
        responses = data if batched else [data]
        if len(responses) != len(group):
            raise AssertionError("Server returned a different batch size")
        for index, (name, value) in enumerate(zip(names, responses)):
            if prefill_meta_info is not None:
                value["prefill_meta_info"] = prefill_meta_info[index]
            print(name, json.dumps(value), flush=True)
        return list(zip(names, responses))

    items = list(prompts.items())

    def run_requests():
        results = dict(pair for item in items[:5] for pair in generate([item]))
        # Submit mixed lengths atomically so HTTP arrival timing does not
        # change NVFP4 matrix shapes between the compared runs.
        results.update(generate(items[5:]))
        return results

    results = run_requests()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    cache_meta_key = "prefill_meta_info" if args.decode_url else "meta_info"
    if results["cached_repeat"][cache_meta_key]["cached_tokens"] <= 0:
        raise AssertionError("cached_repeat did not exercise prefix-cache reuse")
    if reference is None and not args.skip_baseline_repeat:
        for url in (args.url, args.decode_url):
            if url:
                requests.post(
                    url.rstrip("/") + "/flush_cache", timeout=60
                ).raise_for_status()
        room_base += 1 << 20
        repeat = run_requests()
        args.output.with_suffix(".repeat.json").write_text(
            json.dumps(repeat, indent=2) + "\n"
        )
        try:
            repeated = compare_results(repeat, results, args.logprob_atol)
        except AssertionError as error:
            raise AssertionError(
                "The unsharded baseline is not repeatable; use deterministic top-k "
                "and consistent request batching before comparing sharding. "
                + str(error)
            ) from error
        print(json.dumps({"baseline_repeatability": repeated}), flush=True)
    if reference is not None:
        print(
            json.dumps(compare_results(results, reference, args.logprob_atol)),
            flush=True,
        )


if __name__ == "__main__":
    main()
