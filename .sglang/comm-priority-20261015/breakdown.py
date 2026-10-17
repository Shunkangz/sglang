"""Per-layer critical-path breakdown on the A stream from the nsys traces (mean over GPUs and layers)."""

import glob
import json
import os
import sys

import analyze_nsys as A
import numpy as np

rows = []
for p in sorted(glob.glob(os.path.join(A.NS, "*.sqlite"))):
    name = os.path.basename(p)[:-7]
    D = A.load(name)
    P = A.analyze(D)
    lead = D["res"].get("lead", 0)
    per = 2 if lead else 1
    g_dur, mid_gap, g2a, a_dur, a2g = [], [], [], [], []
    for d, p_ in P.items():
        Aops, G = p_["A"], p_["G"]
        for l in range(len(Aops)):
            gs = G[l * per : (l + 1) * per]
            if len(gs) < per:
                continue
            g_dur.append(sum(e - s for s, e, _ in gs) / 1e3)
            if lead:
                mid_gap.append((gs[1][0] - gs[0][1]) / 1e3)
            g2a.append((Aops[l][0] - gs[-1][1]) / 1e3)
            a_dur.append((Aops[l][1] - Aops[l][0]) / 1e3)
            if (l + 1) * per < len(G):
                a2g.append((G[(l + 1) * per][0] - Aops[l][1]) / 1e3)
    layer = (
        np.mean(g_dur)
        + (np.mean(mid_gap) if mid_gap else 0)
        + np.mean(g2a)
        + np.mean(a_dur)
        + np.mean(a2g)
    )
    r = dict(
        trace=name,
        gemm=np.mean(g_dur),
        gap_in_gemm=np.mean(mid_gap) if mid_gap else 0.0,
        gemm_to_A=np.mean(g2a),
        A=np.mean(a_dur),
        A_to_gemm=np.mean(a2g),
        layer=layer,
    )
    rows.append(r)
print(
    "| trace | GEMM us | gap between GEMM halves | GEMM end -> A start | A (per rank) | A end -> next GEMM | layer total us |"
)
print("|---|---|---|---|---|---|---|")
for r in rows:
    print(
        f"| {r['trace']} | {r['gemm']:.0f} | {r['gap_in_gemm']:.1f} | {r['gemm_to_A']:.1f} | {r['A']:.0f} | {r['A_to_gemm']:.1f} | {r['layer']:.0f} |"
    )
json.dump(rows, open(os.path.join(A.H, "out", "layer_breakdown.json"), "w"), indent=1)
