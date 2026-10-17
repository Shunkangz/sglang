import json
import sys

f = lambda v, s=1e3: "  n/a" if v is None else f"{v * s:5.0f}"
for l in open(sys.argv[1] if len(sys.argv) > 1 else "out/sweep.jsonl"):
    r = json.loads(l)
    if "fail" in r:
        print("FAIL", r)
        continue
    tag = (
        r["b"]
        + ("" if r["scope"] == "global" else "/local")
        + (" lead" if r.get("lead") else "")
        + ("" if r["prio"] else " prio0")
        + (" graph" if r.get("graph") else "")
        + (f" green{r['green_b']}" if r.get("green_b") else "")
    )
    knob = (
        f"chunk={r['b_chunk_kib'] / 1024:g}M"
        if r["b"] in ("nccl-gated", "ce", "ce-gated", "nccl-ce", "nccl-ce-gated")
        else (
            f"item={r['b_item_kib']:g}K ctas={r['b_ctas']}"
            if r["b"].startswith("sm")
            else (f"maxCTAs={r['b_nccl_ctas']}" if r["b_nccl_ctas"] else "")
        )
    )
    print(
        f"{tag:26s} {knob:20s} A={r['a_ops']}/{r['a_mib']:g}M fwd={r['fwd_ms']:6.2f} A mean/p99={f(r['a_mean_ms'])}/{f(r['a_p99_ms'])}us gemm={f(r['gemm_mean_ms'])}us B@fwd={f(r['b_during_fwd_GBps'], 1)} Btot={r['b_ms']:6.1f}ms ({r['b_ingress_GBps']:3.0f}GB/s) bad={r['bad']}"
    )
