#!/usr/bin/env python3
"""Render the result files into the markdown tables used by DESIGN_comm_priority_preemption.md."""

import json
import os
import statistics as st
import sys
from collections import defaultdict

H = os.path.dirname(os.path.abspath(__file__))


def rows(path):
    out = []
    for line in open(os.path.join(H, path)):
        r = json.loads(line)
        if "fail" not in r:
            out.append(r)
    return out


def tag(r):
    t = r["b"]
    if r["b"] in ("nccl-gated", "ce", "ce-gated", "nccl-ce", "nccl-ce-gated"):
        t += f" {r['b_chunk_kib'] / 1024:g}M"
    if r["b"].startswith("sm"):
        t += f" {r['b_item_kib']:g}K x{r['b_ctas']}"
    if r["b"] == "nccl" and r["b_nccl_ctas"]:
        t += f" maxCTAs={r['b_nccl_ctas']}"
    if r["scope"] == "local" and r["b"] not in ("none", "nccl", "ce", "sm", "nccl-ce"):
        t += " local"
    if r.get("lead"):
        t += " lookahead"
    if not r["prio"]:
        t += " equal-prio"
    if r.get("green_b"):
        t += f" green{r['green_b']}"
    if r.get("graph"):
        t += " graph"
    return t


SM_USE = {
    "none": "-",
    "none-gated": "-",
    "nccl": "NCCL CTAs",
    "nccl-gated": "NCCL CTAs",
    "nccl-ce": "0",
    "nccl-ce-gated": "0",
    "ce": "0",
    "ce-gated": "0",
    "sm": "16-32 CTAs",
    "sm-preempt": "16-32 CTAs (0 paused)",
}


def fmt(v, f="{:.0f}"):
    return "n/a" if v is None else f.format(v)


def med(vs):
    vs = [v for v in vs if v is not None]
    return st.median(vs) if vs else None


def headline(path="out/repeat.jsonl"):
    g = defaultdict(list)
    for r in rows(path):
        g[(tag(r), r["a_ops"], r["a_mib"])].append(r)
    base = None
    for (t, ops, mib), rs in g.items():
        if t == "none-gated" and ops == "ar":
            base = rs
    bf = med([r["fwd_ms"] for r in base]) if base else None
    print(
        "| B mode | B SMs | fwd ms (vs none-gated) | A mean / p99 us | GEMM + gate close us | B goodput during fwd GB/s | B total ms (GB/s) | reps |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for (t, ops, mib), rs in g.items():
        f = med([r["fwd_ms"] for r in rs])
        a = med([r["a_mean_ms"] and r["a_mean_ms"] * 1e3 for r in rs])
        a99 = med([r["a_p99_ms"] and r["a_p99_ms"] * 1e3 for r in rs])
        gm = med([r["gemm_mean_ms"] and r["gemm_mean_ms"] * 1e3 for r in rs])
        bfw = med([r["b_during_fwd_GBps"] for r in rs])
        bms = med([r["b_ms"] for r in rs])
        bgb = med([r["b_ingress_GBps"] for r in rs])
        d = f" ({100 * (f / bf - 1):+.1f}%)" if bf and t != "none-gated" else ""
        bt = "-" if rs[0]["b"].startswith("none") else f"{bms:.1f} ({bgb:.0f})"
        print(
            f"| {t} | {SM_USE.get(rs[0]['b'], '?')} | {f:.2f}{d} | {fmt(a)} / {fmt(a99)} | {fmt(gm)} | {fmt(bfw)} | {bt} | {len(rs)} |"
        )


def matrix(path="out/sweep.jsonl"):
    print(
        "| B mode | A | fwd ms | A mean / p99 us | GEMM + gate close us | B goodput during fwd GB/s | B total GB/s | SM pause exits / launches |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in rows(path):
        a = r["a_mean_ms"] and r["a_mean_ms"] * 1e3
        a99 = r["a_p99_ms"] and r["a_p99_ms"] * 1e3
        gm = r["gemm_mean_ms"] and r["gemm_mean_ms"] * 1e3
        ex = (
            f"{r['sm_pause_exits']:.0f} / {r['sm_launches']:.0f}"
            if r["b"] == "sm-preempt"
            else "-"
        )
        bt = "-" if r["b"].startswith("none") else f"{r['b_ingress_GBps']:.0f}"
        print(
            f"| {tag(r)} | {r['a_ops']} {r['a_mib']:g}M | {r['fwd_ms']:.2f} | {fmt(a)} / {fmt(a99)} | {fmt(gm)} | "
            f"{fmt(r['b_during_fwd_GBps'])} | {bt} | {ex} |"
        )


def nsys(path="out/nsys_summary.jsonl"):
    print(
        "| trace | A per rank us | arrival skew us | A after last arrival us (p99) | B busy during A (this GPU / any GPU) | "
        "B tail into A us mean / p99 / max | B ops starting inside A | resume us (median) | GPU0 NVLink RX/TX % during A | GPU0 GPC MHz GEMM / A |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for line in open(os.path.join(H, path)):
        s = json.loads(line)
        g = lambda k, f="{:.0f}": fmt(s.get(k), f)
        print(
            f"| {s['mode']} | {g('a_mean_us')} | {g('a_arrival_skew_mean_us')} | {g('a_after_last_arrival_mean_us')} "
            f"({g('a_after_last_arrival_p99_us')}) | {g('overlap_local_pct', '{:.1f}')}% / {g('overlap_any_pct', '{:.1f}')}% | "
            f"{g('pause_tail_mean_us', '{:.1f}')} / {g('pause_tail_p99_us', '{:.1f}')} / {g('pause_tail_max_us', '{:.1f}')} | "
            f"{s['b_starts_inside_A']} | {g('resume_median_us', '{:.1f}')} | "
            f"{g('gpu0_nvlink_rx_user_data_during_A', '{:.0f}')} / {g('gpu0_nvlink_tx_user_data_during_A', '{:.0f}')} | "
            f"{g('gpu0_gpc_clock_mhz_during_GEMM')} / {g('gpu0_gpc_clock_mhz_during_A')} |"
        )


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "headline"):
        headline()
        print()
    if what in ("all", "matrix"):
        matrix()
        print()
    if what in ("all", "nsys") and os.path.exists(
        os.path.join(H, "out/nsys_summary.jsonl")
    ):
        nsys()
