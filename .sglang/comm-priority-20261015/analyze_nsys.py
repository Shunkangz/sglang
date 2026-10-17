#!/usr/bin/env python3
"""Turn the nsys exports from profile.sh into pause/overlap metrics and timeline figures.

Per device, for the measured (last) forward:
  A ops      = NCCL kernels on the A stream (last L of them)
  B ops      = kernels + memcpys on that device's B streams
  overlap    = |A interval  intersect  union(B ops on this device)| / |A|
  overlap_in = same, against B ops of ALL devices (push traffic into this GPU comes from peers)
  pause_tail = for B ops already running when A starts: max(end) - A.start   (what A has to share)
  starts_in  = B ops that START inside an A interval (a correct gate admits none, up to the close race)
  resume     = first B op start after A.end - A.end
NVLink throughput on GPU 0 comes from the gb10x GPU-metrics set.
"""

import glob
import json
import os
import re
import sqlite3
import sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
NS = os.path.join(H, "out", "nsys")


def union(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def covered(u, s, e):
    t = 0
    for a, b in u:
        if b <= s:
            continue
        if a >= e:
            break
        t += min(b, e) - max(a, s)
    return t


def tables(con):
    return {
        r[0] for r in con.execute("select name from sqlite_master where type='table'")
    }


def load(name):
    log = open(os.path.join(NS, name + ".log")).read()
    streams = {}
    for m in re.finditer(r"STREAMS dev=(\d+) A=(\d+) B=([\d,]+)", log):
        streams[int(m[1])] = (int(m[2]), [int(x) for x in m[3].split(",")])
    rm = re.search(r"^RESULT (.*)$", log, re.M)
    res = json.loads(rm[1]) if rm else {}
    con = sqlite3.connect(os.path.join(NS, name + ".sqlite"))
    tb = tables(con)
    # CUPTI stream ids differ from cudaStreamGetId; the NVTX stream names map them
    streams = {}
    for nm, dev, sid in con.execute(
        "select name, deviceId, streamId from TARGET_INFO_NVTX_CUDA_STREAM"
    ):
        a, b = streams.get(dev, (None, []))
        if nm.startswith("A-main"):
            a = sid
        elif nm.startswith("B-kv"):
            b = b + [sid]
        streams[dev] = (a, b)
    strings = dict(con.execute("select id, value from StringIds"))
    K = con.execute(
        "select start, end, deviceId, streamId, shortName from CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchall()
    K = [(s, e, d, st, strings.get(n, "?")) for s, e, d, st, n in K]
    M = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tb:
        # payload copies only: the 128 B D2H progress mirror of sm-preempt is bookkeeping, not B traffic
        kinds = {
            name: i
            for i, name in con.execute("select id, name from ENUM_CUDA_MEMCPY_OPER")
        }
        payload = tuple(
            i
            for name, i in kinds.items()
            if "PTOP" in name.upper() or "DTOD" in name.upper()
        )
        M = [
            m
            for m in con.execute(
                "select start, end, deviceId, streamId, bytes, copyKind, srcDeviceId, dstDeviceId "
                "from CUPTI_ACTIVITY_KIND_MEMCPY"
            ).fetchall()
            if m[5] in payload and m[4] >= 4096
        ]
    metrics = {}
    if "GPU_METRICS" in tb and "TARGET_INFO_GPU_METRICS" in tb:
        names = {
            n: mid
            for mid, n in con.execute(
                "select distinct metricId, metricName from TARGET_INFO_GPU_METRICS"
            )
        }

        def series(n):
            rows = con.execute(
                "select timestamp, value from GPU_METRICS where metricId=? order by timestamp",
                (names[n],),
            ).fetchall()
            return np.array(rows, dtype=float)

        # writes travel as requests, reads as responses: user data = both
        for key, parts in (
            (
                "NVLink RX user data %",
                [
                    "NVLink RX Requests User Data [Throughput %]",
                    "NVLink RX Responses User Data [Throughput %]",
                ],
            ),
            (
                "NVLink TX user data %",
                [
                    "NVLink TX Requests User Data [Throughput %]",
                    "NVLink TX Responses User Data [Throughput %]",
                ],
            ),
            ("GPC clock MHz", ["GPC Clock Frequency [MHz]"]),
            ("DRAM read %", ["DRAM Read Bandwidth [Throughput %]"]),
            ("SMs active %", ["SMs Active [Throughput %]"]),
        ):
            if not all(p in names for p in parts):
                continue
            arrs = [series(p) for p in parts]
            n = min(len(a) for a in arrs)
            if n:
                v = sum(a[:n, 1] for a in arrs)
                if key == "GPC clock MHz" and np.nanmedian(v) > 1e6:
                    v = v / 1e6  # reported in Hz despite the [MHz] label
                metrics[key] = np.stack([arrs[0][:n, 0], v], axis=1)
    return dict(name=name, streams=streams, res=res, K=K, M=M, metrics=metrics)


def analyze(D):
    L = D["res"].get("layers", 16)
    lead = D["res"].get("lead", 0)
    per_dev = {}
    allB = []
    devs = sorted(D["streams"])
    Bops = {}
    for d in devs:
        sA, sBs = D["streams"][d]
        b = [(s, e) for s, e, dd, st, n in D["K"] if dd == d and st in sBs]
        b += [(s, e) for s, e, dd, st, *_ in D["M"] if dd == d and st in sBs and e > s]
        Bops[d] = sorted(b)
        allB += b
    allU = union(allB)
    for d in devs:
        sA, sBs = D["streams"][d]
        ak = sorted(
            [
                (s, e, n)
                for s, e, dd, st, n in D["K"]
                if dd == d and st == sA and "nccl" in n.lower()
            ]
        )
        gk = sorted(
            [
                (s, e, n)
                for s, e, dd, st, n in D["K"]
                if dd == d and st == sA and "nccl" not in n.lower()
            ]
        )
        A = ak[-L:]
        G = gk[-L * (2 if lead else 1) :]
        if not A:
            continue
        fw0, fw1 = min(G[0][0], A[0][0]) if G else A[0][0], A[-1][1]
        U = union(Bops[d])
        ov, ovin, tail, starts_in, resume = [], [], [], 0, []
        for s, e, _ in A:
            dur = e - s
            ov.append(covered(U, s, e) / dur)
            ovin.append(covered(allU, s, e) / dur)
            running = [be for bs, be in Bops[d] if bs < s < be]
            tail.append((max(running) - s) if running else 0)
            starts_in += sum(1 for bs, be in Bops[d] if s < bs < e)
            nxt = [bs for bs, be in Bops[d] if bs >= e]
            if nxt:
                resume.append(min(nxt) - e)
        busy = covered(U, fw0, fw1) / (fw1 - fw0)
        # same-stream overlap: A's kernel starting before the preceding GEMM ended means programmatic
        # dependent launch let it in early; gap = A start - preceding GEMM end
        gaps = []
        for s_, e_, _ in A:
            prev = [ge for gs, ge, _ in G if gs < s_]
            if prev:
                gaps.append((s_ - max(prev)) / 1e3)
        per_dev[d] = dict(
            a_us=np.array([e - s for s, e, _ in A]) / 1e3,
            gemm_us=np.array([e - s for s, e, _ in G]) / 1e3,
            overlap=np.array(ov),
            overlap_in=np.array(ovin),
            tail_us=np.array(tail) / 1e3,
            starts_in=starts_in,
            resume_us=np.array(resume) / 1e3,
            b_busy=busy,
            fwd_ms=(fw1 - fw0) / 1e6,
            gemm_to_a_gap_us=np.array(gaps),
            window=(fw0, fw1),
            A=A,
            G=G,
        )
    return per_dev


def summarize(D, P):
    cat = lambda k: np.concatenate([P[d][k] for d in P]) if P else np.array([0.0])
    tail = cat("tail_us")
    resume = cat("resume_us")
    out = dict(
        mode=D["name"],
        a_mean_us=float(cat("a_us").mean()),
        a_p99_us=float(np.percentile(cat("a_us"), 99)),
        gemm_mean_us=float(cat("gemm_us").mean()) if len(cat("gemm_us")) else 0.0,
        overlap_local_pct=float(100 * cat("overlap").mean()),
        overlap_any_pct=float(100 * cat("overlap_in").mean()),
        pause_tail_mean_us=float(tail.mean()),
        pause_tail_p99_us=float(np.percentile(tail, 99)),
        pause_tail_max_us=float(tail.max()),
        b_starts_inside_A=int(sum(P[d]["starts_in"] for d in P)),
        resume_median_us=float(np.median(resume)) if len(resume) else None,
        b_busy_pct=float(100 * np.mean([P[d]["b_busy"] for d in P])),
        fwd_ms=float(max(P[d]["fwd_ms"] for d in P)),
        gemm_to_a_gap_median_us=float(np.median(cat("gemm_to_a_gap_us"))),
        a_starts_before_gemm_end=int((cat("gemm_to_a_gap_us") < 0).sum()),
    )
    # A = arrival skew (wait for the last rank) + collective time once every rank is in
    devs = sorted(P)
    nl = min(len(P[d]["A"]) for d in devs)
    skew, pure, gem_spread = [], [], []
    for l in range(nl):
        st = [P[d]["A"][l][0] for d in devs]
        en = [P[d]["A"][l][1] for d in devs]
        skew.append((max(st) - min(st)) / 1e3)
        pure.append((float(np.median(en)) - max(st)) / 1e3)
    out["a_arrival_skew_mean_us"] = float(np.mean(skew))
    out["a_after_last_arrival_mean_us"] = float(np.mean(pure))
    out["a_after_last_arrival_p99_us"] = float(np.percentile(pure, 99))
    g = np.array(
        [
            [P[d]["gemm_us"][i] for d in devs]
            for i in range(min(len(P[d]["gemm_us"]) for d in devs))
        ]
    )
    if g.size:
        out["gemm_cross_rank_spread_us"] = float(np.mean(g.max(1) - g.min(1)))
    mt = D["metrics"]
    if 0 in P and mt:
        A = P[0]["A"]
        G = P[0]["G"]
        for n, arr in mt.items():
            t, v = arr[:, 0], arr[:, 1]
            ina = np.zeros_like(t, dtype=bool)
            ing = np.zeros_like(t, dtype=bool)
            for s, e, _ in A:
                ina |= (t >= s) & (t <= e)
            for s, e, _ in G:
                ing |= (t >= s) & (t <= e)
            key = re.sub(r"[^a-z0-9]+", "_", n.lower()).strip("_")
            out["gpu0_" + key + "_during_A"] = (
                float(v[ina].mean()) if ina.any() else None
            )
            out["gpu0_" + key + "_during_GEMM"] = (
                float(v[ing].mean()) if ing.any() else None
            )
    return out


# --------------------------------------------------------------------------------------------
# Timeline figure (light theme; reference categorical palette: slot 1 blue = A, slot 2 orange = B)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
C_GEMM = "#b9b8b3"
C_A = "#2a78d6"
C_B = "#eb6834"
C_BIN = "#f4b294"


def plot(runs, path, layer0=4, nlayers=3, title=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nruns = len(runs)
    has_metrics = any(D["metrics"] for D, P in runs)
    rows_per = 2 if has_metrics else 1
    fig = plt.figure(
        figsize=(13, 2.05 * nruns * (1.45 if has_metrics else 1)), facecolor=SURFACE
    )
    hr = []
    for _ in runs:
        hr += [1.0, 0.7] if has_metrics else [1.0]
    gs = fig.add_gridspec(
        len(hr),
        1,
        height_ratios=hr,
        hspace=0.55,
        top=0.97 if title else 0.99,
        bottom=0.03,
    )
    for i, (D, P) in enumerate(runs):
        if 0 not in P:
            continue
        p = P[0]
        A, G = p["A"], p["G"]
        per = 2 if D["res"].get("lead", 0) else 1
        t0 = G[layer0 * per][0] if len(G) > layer0 * per else A[0][0]
        t1 = A[min(layer0 + nlayers, len(A)) - 1][1] + 20_000
        ax = fig.add_subplot(gs[i * rows_per])
        ax.set_facecolor(SURFACE)
        lanes = [
            ("GEMM (A stream)", 3),
            ("Comm A: NCCL", 2),
            ("Comm B: this GPU", 1),
            ("Comm B: any GPU", 0),
        ]
        sA, sBs = D["streams"][0]

        def bars(iv, y, color):
            xs = [
                ((max(s, t0) - t0) / 1e3, (min(e, t1) - max(s, t0)) / 1e3)
                for s, e in iv
                if e > t0 and s < t1
            ]
            if xs:
                ax.broken_barh(
                    xs,
                    (y - 0.32, 0.64),
                    facecolors=color,
                    edgecolor=SURFACE,
                    linewidth=0.6,
                )

        bars([(s, e) for s, e, _ in G], 3, C_GEMM)
        bars([(s, e) for s, e, _ in A], 2, C_A)
        b0 = [(s, e) for s, e, dd, st, nn in D["K"] if dd == 0 and st in sBs]
        b0 += [(s, e) for s, e, dd, st, *_ in D["M"] if dd == 0 and st in sBs and e > s]
        bars(b0, 1, C_B)
        ball = []
        for d, (sa, sb) in D["streams"].items():
            ball += [(s, e) for s, e, dd, st, nn in D["K"] if dd == d and st in sb]
            ball += [
                (s, e) for s, e, dd, st, *_ in D["M"] if dd == d and st in sb and e > s
            ]
        bars(union(ball), 0, C_BIN)
        for s, e, _ in A:
            if e > t0 and s < t1:
                ax.axvspan((s - t0) / 1e3, (e - t0) / 1e3, color=C_A, alpha=0.07, lw=0)
        ax.set_yticks(
            [y for _, y in lanes], [l for l, _ in lanes], fontsize=8, color=INK2
        )
        ax.set_xlim(0, (t1 - t0) / 1e3)
        ax.set_ylim(-0.6, 3.6)
        ax.tick_params(axis="x", labelsize=8, colors=INK2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        s = summarize(D, P)
        ax.set_title(
            f"{D['name']}   A mean {s['a_mean_us']:.0f} us, B active during A {s['overlap_any_pct']:.0f}%, "
            f"pause tail p99 {s['pause_tail_p99_us']:.1f} us, fwd {s['fwd_ms']:.2f} ms",
            loc="left",
            fontsize=9.5,
            color=INK,
        )
        if has_metrics:
            ax2 = fig.add_subplot(gs[i * rows_per + 1], sharex=ax)
            ax2.set_facecolor(SURFACE)
            mt = D["metrics"]
            shown = 0
            styles = [("-", C_B), ("--", INK2)]
            for mname, arr in mt.items():
                if not mname.startswith("NVLink"):
                    continue
                sel = (arr[:, 0] >= t0) & (arr[:, 0] <= t1)
                ls, col = styles[shown]
                ax2.plot(
                    (arr[sel, 0] - t0) / 1e3,
                    arr[sel, 1],
                    ls,
                    color=col,
                    lw=1.4,
                    label=mname,
                )
                shown += 1
            for s_, e_, _ in A:
                if e_ > t0 and s_ < t1:
                    ax2.axvspan(
                        (s_ - t0) / 1e3, (e_ - t0) / 1e3, color=C_A, alpha=0.07, lw=0
                    )
            ax2.set_ylabel("GPU0 NVLink\n(% of peak)", fontsize=7.5, color=INK2)
            ax2.tick_params(labelsize=7.5, colors=INK2)
            for sp in ax2.spines.values():
                sp.set_visible(False)
            ax2.grid(color=GRID, lw=0.6)
            ax2.set_ylim(0, 100)
            if shown:
                ax2.legend(
                    fontsize=7,
                    frameon=False,
                    loc="upper right",
                    ncol=2,
                    labelcolor=INK2,
                )
            if i == nruns - 1:
                ax2.set_xlabel(
                    "time (us) from the start of a layer, GPU 0",
                    fontsize=8.5,
                    color=INK2,
                )
        elif i == nruns - 1:
            ax.set_xlabel(
                "time (us) from the start of a layer, GPU 0", fontsize=8.5, color=INK2
            )
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK, y=0.995)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", path)


def plot_zoom(runs, path, layer=6, before_us=15, after_us=45):
    """Microsecond view around one A start on GPU 0: what B is still doing when A begins."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(runs),
        1,
        figsize=(11, 1.55 * len(runs) + 0.6),
        facecolor=SURFACE,
        sharex=True,
    )
    for ax, (D, P) in zip(np.atleast_1d(axes), runs):
        p = P[0]
        a0 = p["A"][layer][0]
        t0, t1 = a0 - before_us * 1e3, a0 + after_us * 1e3
        sA, sBs = D["streams"][0]
        ax.set_facecolor(SURFACE)

        def bars(iv, y, color):
            xs = [
                ((max(s, t0) - a0) / 1e3, (min(e, t1) - max(s, t0)) / 1e3)
                for s, e in iv
                if e > t0 and s < t1
            ]
            if xs:
                ax.broken_barh(
                    xs,
                    (y - 0.3, 0.6),
                    facecolors=color,
                    edgecolor=SURFACE,
                    linewidth=0.8,
                )

        bars([(s, e) for s, e, _ in p["G"]], 3, C_GEMM)
        bars([(s, e) for s, e, _ in p["A"]], 2, C_A)
        b0 = [(s, e) for s, e, dd, st, nn in D["K"] if dd == 0 and st in sBs]
        b0 += [(s, e) for s, e, dd, st, *_ in D["M"] if dd == 0 and st in sBs and e > s]
        bars(b0, 1, C_B)
        ball = []
        for d, (sa, sb) in D["streams"].items():
            ball += [(s, e) for s, e, dd, st, nn in D["K"] if dd == d and st in sb]
            ball += [
                (s, e) for s, e, dd, st, *_ in D["M"] if dd == d and st in sb and e > s
            ]
        bars(union(ball), 0, C_BIN)
        ax.axvline(0, color=C_A, lw=1.2)
        run_b = [e for s, e in b0 if s < a0 < e]
        tail = (max(run_b) - a0) / 1e3 if run_b else 0.0
        ax.set_yticks(
            [3, 2, 1, 0],
            ["GEMM", "Comm A", "B: GPU 0", "B: any GPU"],
            fontsize=8,
            color=INK2,
        )
        ax.set_ylim(-0.6, 3.6)
        ax.set_xlim(-before_us, after_us)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8, colors=INK2)
        ax.set_title(
            f"{D['name']}   B on GPU 0 still running {tail:.1f} us after A started (layer {layer})",
            loc="left",
            fontsize=9,
            color=INK,
        )
    np.atleast_1d(axes)[-1].set_xlabel(
        "time (us) relative to the start of Comm A on GPU 0", fontsize=8.5, color=INK2
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", path)


def main():
    names = sorted(
        os.path.basename(p)[:-7] for p in glob.glob(os.path.join(NS, "*.sqlite"))
    )
    if len(sys.argv) > 1:
        names = [n for n in names if any(a in n for a in sys.argv[1:])]
    runs, rows = [], []
    for n in names:
        D = load(n)
        P = analyze(D)
        runs.append((D, P))
        s = summarize(D, P)
        rows.append(s)
        print(json.dumps(s))
    with open(os.path.join(H, "out", "nsys_summary.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for D, P in runs:
        plot([(D, P)], os.path.join(H, "out", f"timeline_{D['name']}.png"))
    order = [
        "10_none_gated",
        "02_nccl_naive",
        "03_nccl_gated_16M",
        "11_nccl_ce_gated_4M",
        "05_ce_gated_4M",
        "08_sm_preempt_256K",
    ]
    by = {r[0]["name"]: r for r in runs}
    key = [by[n] for n in order if n in by]
    zoom = [
        by[n]
        for n in [
            "02_nccl_naive",
            "03_nccl_gated_16M",
            "11_nccl_ce_gated_4M",
            "05_ce_gated_4M",
            "08_sm_preempt_256K",
            "06_ce_gated_4M_lead",
        ]
        if n in by
    ]
    if zoom:
        plot_zoom(zoom, os.path.join(H, "out", "pause_zoom.png"))
    if key:
        plot(
            key,
            os.path.join(H, "out", "timeline_key_modes.png"),
            title="GPU 0, three layers: GEMM, then Comm A (64 MiB all-reduce); Comm B is the background KV all-gather",
        )


if __name__ == "__main__":
    main()
