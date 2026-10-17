#!/usr/bin/env python3
"""A's collective time vs a rate-limited, ungated B (profile_rate.sh traces), with gated references."""

import json
import os
import re

import analyze_nsys as A
import numpy as np

GROUPS = [
    (
        "SM put, CTA-capped (steady)",
        re.compile(r"^(2[0-3]_sm_cta\d+|07_sm_naive)$"),
        "#2a78d6",
    ),
    ("CE push 1 MiB, paced (bursty)", re.compile(r"^2[4-8]_ce1M_pace\d+$"), "#eb6834"),
    ("NCCL all-gather, CTA-capped", re.compile(r"^(29|30)_nccl_cta\d+$"), "#1baf7a"),
    (
        "gated movers (B paused during A)",
        re.compile(
            r"^(05_ce_gated_4M|11_nccl_ce_gated_4M|04_nccl_ce_gated_16M|08_sm_preempt_256K)$"
        ),
        "#52514e",
    ),
]


def row(name):
    D = A.load(name)
    P = A.analyze(D)
    s = A.summarize(D, P)
    res = D["res"]
    gemm = float(np.mean(np.concatenate([P[d]["gemm_us"] for d in P])))
    return dict(
        trace=name,
        b_rate=res.get("b_during_fwd_GBps") or res.get("b_ingress_GBps") or 0.0,
        a_after=s["a_after_last_arrival_mean_us"],
        a_after_p99=s["a_after_last_arrival_p99_us"],
        a_rank=s["a_mean_us"],
        skew=s["a_arrival_skew_mean_us"],
        gemm=gemm,
        fwd=s["fwd_ms"],
        b_busy_A=s["overlap_any_pct"],
        nvl_tx_A=s.get("gpu0_nvlink_tx_user_data_during_A"),
        nvl_tx_G=s.get("gpu0_nvlink_tx_user_data_during_GEMM"),
    )


def main():
    names = sorted(
        os.path.basename(p)[:-7] for p in os.listdir(A.NS) if p.endswith(".sqlite")
    )
    names = [
        n for n in names if n in ("01_none",) or any(g.match(n) for _, g, _ in GROUPS)
    ]
    rows = [row(n) for n in names]
    base = next(r for r in rows if r["trace"] == "01_none")
    print(
        "| trace | B rate during fwd GB/s | A after last arrival us (p99) | vs no-B | A per rank us | skew us | GEMM kernel us | fwd ms (16 layers) | GPU0 NVLink TX % during A / GEMM |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['trace']} | {r['b_rate']:.0f} | {r['a_after']:.0f} ({r['a_after_p99']:.0f}) | {100 * (r['a_after'] / base['a_after'] - 1):+.1f}% | "
            f"{r['a_rank']:.0f} | {r['skew']:.0f} | {r['gemm']:.0f} | {r['fwd']:.2f} | "
            f"{r['nvl_tx_A'] or 0:.0f} / {r['nvl_tx_G'] or 0:.0f} |"
        )
    json.dump(rows, open(os.path.join(A.H, "out", "rate_sweep.json"), "w"), indent=1)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor=A.SURFACE)
    ax.set_facecolor(A.SURFACE)
    ax.axhline(base["a_after"], color=A.INK2, lw=1, ls=":")
    ax.text(
        300,
        base["a_after"] - 18,
        f"no B: {base['a_after']:.0f} us",
        fontsize=8,
        color=A.INK2,
    )
    for label, pat, color in GROUPS:
        pts = sorted(
            (r["b_rate"], r["a_after"], r["trace"])
            for r in rows
            if pat.match(r["trace"])
        )
        if not pts:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        gated = label.startswith("gated")
        ax.plot(
            xs,
            ys,
            "o" if gated else "o-",
            color=color,
            lw=2,
            ms=8,
            label=label,
            markeredgecolor=A.SURFACE,
            markeredgewidth=2,
        )
    ax.set_xlabel(
        "B goodput while the forward runs (GB/s per GPU)", fontsize=9, color=A.INK2
    )
    ax.set_ylabel("Comm A time after last rank arrives (us)", fontsize=9, color=A.INK2)
    ax.set_title(
        "Can B run through A? A's collective time vs B's rate (64 MiB all-reduce, 8x B200)",
        loc="left",
        fontsize=10.5,
        color=A.INK,
    )
    ax.grid(color=A.GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(labelsize=8, colors=A.INK2)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=False, labelcolor=A.INK2, loc="upper left")
    fig.tight_layout()
    path = os.path.join(A.H, "out", "rate_vs_A.png")
    fig.savefig(path, dpi=140, facecolor=A.SURFACE)
    print("wrote", path)


if __name__ == "__main__":
    main()
