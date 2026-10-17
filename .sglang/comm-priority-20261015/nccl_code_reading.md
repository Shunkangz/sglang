# Why background traffic B slows NCCL AllReduce A on 8x B200, read from the NCCL 2.30.7 code

All citations are `nccl@2.30.7:src/...`. I compared the 2.32.3 checkout (`/home/scratch.shunkangz_gpu_1/LLM-Inference/nccl`) against it. `device/all_reduce.h`, `op128.h`, `prims_ll128.h` and `primitives.h` are identical. `prims_simple.h` only gains PAT-NVLS code and `loadSsize`, and the NVLS chunk clamp is unchanged at 2.32.3 `src/enqueue/enqueue.cc:2304-2322`. Nothing below changed in 2.32.3.

## 1. Short answer

- **A is not limited by NCCL's FIFO slots or by NVLink bandwidth.** The code fixes how much A has in flight: 8 × 16 B per thread per round, one step per role per CTA, one 32 KiB NVLS step or one 128 KiB Ring slice at a time.
- **Every step waits on work that crosses GPUs.** Simple steps end with a `fence.acq_rel.sys` followed by a flag store. NVLS steps also run `multimem.ld_reduce` data loads and credit polls whose result depends on all 8 GPUs.
- **So A is a closed loop.** Throughput ≈ (fixed amount in flight) / (latency per step), by Little's law. Anything that lengthens those latencies slows A in proportion, even with idle link bandwidth.
- **B does not need A's bandwidth.** It only needs to delay A's fences, flags and multimem round trips at the endpoint GPUs.
- **The data point to the system-scope fence and release (inferred).** CUDA documents that a fence can wait on another kernel's in-flight NVLink writes, and domains do not isolate `.sys` fences. LL128 has no fence on the P2P path and loses +3.1%, against about +11% for both Simple paths. Your release round trip rises 24% under only 44 GB/s of SM-issued B, but only 2% under 436 GB/s of copy-engine B.
- **At high B, bandwidth also matters.** Ring's own traffic (~457 GB/s per direction) plus B approaches the endpoint's SM-write ceiling. NVLS adds a max-of-8 coupling on every multimem operation, which makes it the most sensitive.
- **Your window test did not test the window.** Three of the four knobs left the in-flight data unchanged. The one real change (NVLS with 32 CTAs) shows A alone already sits at a shared-resource limit.

## 2. What bounds A alone

### 2.1 Machinery shared by every Simple path

- **Copy loop.** `COLL_UNROLL` = 8 on sm_100 (`include/device.h:535-538`). A hunk is 8 × 32 lanes × 16 B = 4 KiB per warp (`device/common_kernel.h:42`). Each thread issues 8 loads of 16 B per source, then 8 stores of 16 B that use those registers (`common_kernel.h:80-162`).
  - At most 128 B per thread per source is outstanding before the warp stalls. This is inferred from in-order issue plus register dependence; `asm volatile` only fixes compile order.
  - Stores are weak `st.global` and are posted (`op128.h:306-319`). Nothing waits for them until the step's fence.
- **Step protocol.** Each slice or step runs: `waitPeer` (a spin with no backoff, `prims_simple.h:100-115`), then `subBarrier`, then `reduceCopy`, then `barrier`, then `postPeer` (`prims_simple.h:218-301`).
  - `postPeer` executes `fence.acq_rel.sys` and then `st.relaxed.sys.global` of the step counter (`prims_simple.h:164-173`, `op128.h:393-399`).
  - The fence runs only for the send-side post thread, and only when data was stored.
- **Sync warp.** One extra warp exists "to overlap the threadfence and the copy" (`prims_simple.h:584-586`). The workers meet it again at the next `barrier`, so the step period is at least fence time plus post time.
- **Credit store may wait behind the fence (inferred, SASS not checked).** PostRecv and PostSend are adjacent lanes of the sync warp (`prims_simple.h:611-626`). They share the one store at line 171, which follows the divergent fence at line 169. The credit returned to the upstream rank probably also waits for that fence.
- **Slots.** `NCCL_STEPS` = 8 (`include/device.h:26`). A sender blocks while `head + 8 < step + StepPerSlice` (`prims_simple.h:107`).
- **Polls.** Except NVLS, polls are `ld.volatile` (relaxed, system scope) with no acquire. The code's own comment says "faster than acquire but not as correct" (`prims_simple.h:84-98`).

### 2.2 NVLS Simple (the A you ran: unregistered buffers, `regUsed=0`)

Each GPU runs 24 CTAs (`NVLS_NCHANNELS_SM100_NVL`, `transport/nvls.cc:155-157`) of 640 threads (`graph/tuning.cc:251-252`). The warps split into three roles (`device/all_reduce.h:395-408`):

| Role | Warps (workers + sync) | Data path | Crosses NVLink? | Per-step synchronization |
|---|---|---|---|---|
| Scatter | 8 (7 + 1) | `sendbuff` → this GPU's own unicast (UC) staging slot (`ld.volatile`/`st.global`) | No | Polls 8 heads' credits locally. The sync warp runs `fence.acq_rel.sys` and stores 8 local tails. |
| Reduce (this GPU's head) | 5 (4 + 1) | `multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2` on the multicast (MC) alias of the 8 GPUs' scatter slots, then `multimem.st.global.v4.f32` to the MC gather slot | **Yes: all of A's data traffic** | Polls tail and head with `multimem.ld_reduce.acquire.sys.global.min.u64` (the minimum over 8 GPUs). Then bar.sync, `fence.acq_rel.sys`, and a plain `st.relaxed.sys` to the MC credit address. |
| Gather | 7 (7 + 0) | Local UC slot → `recvbuff` | No | Waits for all 8 heads' tails, each written by a different GPU. Posts its head without a fence, and has no overlap warp. |

Sources for the table:
- Staging buffers: `nvls.cc:344-375,447-460`.
- Credit layout: `nvls.cc:567-602`.
- Credit poll: `prims_simple.h:84-98,489-495,540-546`.
- Multimem instructions: `reduce_kernel.h:1162-1179,1264,1287`, `op128.h:436-440`.
- Corrected from one of the readings: the reduce role has 4 worker warps, not 5 (`prims_simple.h:586`).

**Geometry.**
- Chunk: the host computes it per channel (`enqueue.cc:719-737`), giving 32 KiB per head per step (`enqueue.cc:2258-2275`). Section 3 has the clamp.
- Each channel holds 2.80 MB, so it runs **11 steps** of 8 heads × 32 KiB (`all_reduce.h:410-456`).
- The 8 FIFO slots already cover 8 of those 11 steps, and each role works on one slot at a time. The slots do not bind.

**In flight on NVLink.** At most 4 warps × 4 KiB = 16 KiB of `multimem.ld_reduce` per CTA, or 384 KiB per GPU. This corrects the 20 KiB / 480 KiB figure in one reading.

**Local critical path.** A 32 KiB head chunk is 8 hunks over 7 workers, and the rotation resets on every call. Warp 0 therefore does 2 hunks per head, which is **16 dependent load→store rounds per step**; the other warps do 8 (`common_kernel.h:48-61,184-199,253-257`, `prims_simple.h:428-464`). One reading said 9-10 rounds; that was wrong.

**Bytes per GPU (inferred accounting).** About 72 MiB per direction: 64 MiB of switch reads of the local replica plus 8 MiB of `multimem.st` on TX, and the mirror image on RX. That is ≈ 268 GB/s, about 30% of 900 GB/s, which matches your nsys 25-30%. HBM traffic is ≈ 384 MiB per GPU ≈ 1.4 TB/s, about 18% of peak.

**Throughput model: T ≈ (S + f) · P.**
- S = 11 steps and f ≈ 2 steps of fill, so P ≈ **22 µs per step**.
- With 32 CTAs, S = 8 and T = 269 µs, so P ≈ 27 µs: **+24% per step for +33% concurrency**.
- At idle latencies (inferred: local latency 0.6-0.8 µs, multimem round trip 2-3 µs), scatter and gather would take ≈ 10-13 µs and reduce ≈ 6-10 µs, predicting 150-190 µs in total. The real kernel is 1.5× slower, and its step time grows with A's own CTA count. A alone therefore sits at the knee of a shared, load-dependent resource. Candidates: the multimem/NVLS engine rate, per-GPU tracking of outstanding requests, or fence cost that grows with total outstanding writes.
- By Little's law, 384 KiB in flight ÷ ~30 GB/s of reduced output per GPU ≈ 13 µs per round, far above any plausible multimem round trip. The reduce warps spend most of their time waiting in the synchronization chain.
- One reading claimed "the Little's-law multimem bound is what bounds A". That was wrong: it gives a bound 4-7× above the measurement. **The code does not determine which of the three stages is critical.**

### 2.3 Ring Simple

**Configuration.**
- 32 CTAs (inferred default: 8 rings × 90 → duplicated to 16 → doubled to 32; `graph/search.cc:1036-1063`, `graph/connect.cc:460,503-505`).
- 544 threads each: 16 worker warps + 1 sync warp (`tuning.cc:244-247`, `enqueue.cc:2106-2107`).
- P2P write mode: read mode is a default only on cc 8.0 (`graph/paths.cc:416-417`). The FIFO and tail live in the receiver's memory and the head in the sender's (`transport/p2p.cc:549-568,598-616`). All cross-GPU traffic is posted writes, and both polls are local.
- Eager and unregistered, so the `direct*` primitives stage through the FIFO (`prims_simple.h:501-516,548-563`).

**Geometry.**
- 2 MiB per channel. The device shrinks the chunk to channelBytes/8 = **256 KiB** (`all_reduce.h:34-38`), sent as **2 slices of 128 KiB** (`prims_simple.h:183-184`; SlicePerChunk = StepPerSlice = 2, `collectives.h:19-20`).
- 15 primitive calls per channel make 30 slice operations. **28 of them send**, so there are 28 fence+tail handshakes per channel (`all_reduce.h:42-81`). One reading said 30 handshakes; corrected.

**Dependency.**
- Send slice k needs receive slice k−2 (`all_reduce.h:47,55`), so each hop has **one chunk** in flight.
- Window per GPU = C × chunk = nBytes/nranks = **8 MiB, independent of C and of `NCCL_BUFFSIZE`**.
- The 4-slice credit window never binds in steady state.

**Per slice, per thread.** Two passes. Each pass issues 8 or 16 × 16 B local `ld.volatile` and 8 × 16 B remote `st.global.v2.b64`, plus 8 local stores on copy slices. The whole slice's remote stores can still be outstanding when the fence starts; this corrects the 128 B per-thread figure in one reading.

**Model: T ≈ 30 · P.**
- P ≈ 8.6 µs. If the path is latency-bound, P ≈ L_hop/2, so L_hop ≈ 17 µs.
- L_hop = P_s (CTA work, ≈ 2-4 dependent local latencies) + F (the fence drains this slice's posted stores, queued behind the other 31 channels' ≈ 4 MiB) + W (one-way tail store) + D (poll and bar.sync).
- Check: 8 MiB / 17 µs ≈ 490 GB/s, against a measured egress of 2·(7/8)·64 MiB / 257 µs = **457 GB/s per direction**.
- The data cannot yet separate this latency-bound case from a P_s-bound case. Experiment 4 does.
- Ring is **not** far from link-bound: 457 GB/s is ~51% of 900 GB/s and ~67% of the 679 GB/s your SM-put B reached. The 25-30% nsys figure was measured on the NVLS A.

### 2.4 Ring LL128

**Configuration.** 32 CTAs × 640 threads (20 warps) (`tuning.cc:255-257`).

**Wire format.** A 128 B line carries 15 data u64 plus a flag u64 (`include/device.h:110-112`). The flag lane is `tid%8==7` (`prims_ll128.h:373`).
- Sender: 4 × `st.volatile.global.v2.u64` per lane, i.e. 2 KiB on the wire per warp per slice, written into the peer's memory (`prims_ll128.h:282-287`, `op128.h:17-19`).
- Receiver: polls its local buffer with 4 × `ld.volatile.v2.u64` per lane plus `__any_sync`, then reloads the slice (`prims_ll128.h:190-207`).

**No fence and no tail on NVLink P2P.** `postSend` fences only if `sendConnTailPtr` is set. That requires `connFifo`, which P2P sets only in copy-engine memcpy mode (`prims_ll128.h:87-96,356-364`, `p2p.cc:559-572`).

**Per step.** Thread 0 checks a local head, there are 2 bar.syncs over 640 threads, and one fence-free head store goes to the sender (`prims_ll128.h:70-86,306-333,340-345`).

**Model: T ≈ 15 × (7t + 2 bar) + fill.**
- Each warp handles ceil(137/20) = 7 slices per 256 KiB step, so t ≈ 2.5 µs per 2 KiB warp-slice (fitted).
- About 1-1.5 µs of t is the poll and reload local latency. The rest is unexplained: repeated polls or store backpressure.
- The cross-rank dependency is one one-way store latency. It is hidden while it stays under (n−1)·t ≈ 15 µs.
- Wire traffic is 125 MB per direction ≈ 464 GB/s.

### 2.5 Summary

| Path | Unit of work | Units per CTA | Cross-GPU waits per unit | Fitted period | Own link use per direction |
|---|---|---|---|---|---|
| NVLS Simple | 8 heads × 32 KiB step | 11 | min-poll (8 GPUs), 2 × ld_reduce (8 GPUs), fence after multimem.st, MC flag; gather waits on 8 tails | ~22 µs | ~268 GB/s (30%) |
| Ring Simple | 128 KiB slice | 30 | fence (drains remote stores), remote tail, local poll | ~8.6 µs (L_hop ~17 µs) | ~457 GB/s (51%) |
| Ring LL128 | 2 KiB warp-slice | 7 per step × 15 | one-way store only; no fence | ~2.5 µs | ~464 GB/s |

A side note on algorithm choice: the tuner gives NVLS and Ring Simple the same bus bandwidth (720 → 411.4 GB/s). NVLS wins only on latency, 25 µs against 8.4 + 14 × 3.4 = 56 µs (`tuning.cc:150,160-161,307-324,409,424-425`). The model predicts 188 µs for NVLS and 219 µs for Ring; you measured 282 and 257. It has no term for the step pipeline.

## 3. The window test: what each knob actually did

| Knob | Effective change in code | In-flight data changed? |
|---|---|---|
| `NCCL_NVLS_CHUNKSIZE=524288` | The chunk is computed per channel with nChannels=1 and nBytes = 2.80 MB. It starts at min(512 KiB, cap), clamps to 64 KiB because nBytes < 64·8·chunk, then to **32 KiB** because nBytes < 8·8·64 KiB = 4 MiB (`enqueue.cc:719-737,2222-2225,2258-2275`). Only the slot stride (128 → 512 KiB) and the staging buffer (384 MiB → 1.5 GiB per GPU, `nvls.cc:427-430`) change. | **No** (no-op on device) |
| `NCCL_NVLS_NCHANNELS=32` | Read into `config.nvlsCTAs`, clamped to [1, 64] (`init.cc:2169-2175`, `nvls.cc:222,235`). 32 CTAs, 2 MiB per channel, still a 32 KiB chunk, 8 steps. Multimem in flight goes from 384 to 512 KiB per GPU. | **Yes (the only real change)** |
| `NCCL_BUFFSIZE=16 MiB` (Ring) | The host chunk grows to 8 MiB, but the device resets it to 256 KiB (`all_reduce.h:38`). The slice floor `stepSize·2/32` becomes exactly 128 KiB, equal to the slice already in use (`prims_simple.h:183-184`). FIFO slots grow; at most 1/16 of each is used. | **No** |
| Ring `NCCL_MIN/MAX_NCHANNELS=32` | The default is already 32 (inferred). Even if it were not, the window is C × chunk = nBytes/nranks = 8 MiB for any C. | **No** |

**What this changes in your reading.** "More in-flight data neither speeds A up nor reduces its sensitivity" was tested only once, by `NVLS_NCHANNELS=32`.
- That test moved 33% more data in flight for 5% less time.
- Per-step time rose about 24%.
- High-B sensitivity rose slightly (+45.8% → +50.4%).

That is the signature of a shared queue near saturation, not of a per-CTA window. A fixed-latency window limit would have given about 33% more throughput. To actually change the per-step payload, use fewer, larger channels: `NCCL_NVLS_NCHANNELS≤16` gives 4 MiB per channel and a 64 KiB chunk (`enqueue.cc:2273-2274`); `NCCL_MAX_NCHANNELS=8` gives a 1 MiB Ring chunk with 512 KiB slices.

## 4. The interference mechanism

### 4.1 What in A's loop touches something B also uses (read in code)

| Operation on A's critical path | Paths | Resource B shares |
|---|---|---|
| `fence.acq_rel.sys` after the slice's stores (`prims_simple.h:168-170`) | Ring Simple (28 per channel); NVLS reduce and scatter (every step) | The GPU's tracking of outstanding SM-issued writes and their acknowledgements |
| Remote `st.relaxed.sys` flag (tail to the next GPU, head to the previous GPU, or a broadcast through an MC address) | Ring Simple, NVLS | Endpoint NVLink egress and ingress queues |
| Posted remote `st.global.v2.b64` / `st.volatile.v2.u64` data | Ring Simple, LL128 | Endpoint egress and ingress (backpressure at high B) |
| `multimem.ld_reduce` data loads and `.acquire.min` credit polls | NVLS | Every GPU's port: the result needs all 8 replicas |
| Local `ld.volatile` polls, local copies, bar.sync | All | Only SM, L2 and HBM (indirect) |

LL128 on P2P has only the posted stores and one fence-free head store per step. It never waits on a fence (`prims_ll128.h:87-96`).

### 4.2 Documented semantics

- **PTX fences.** `membar`/`fence.sc` guarantee that "prior memory accesses requested by this thread … are performed at the specified level". The release pattern is a release or acq_rel fence followed by a strong write. Its effect reaches other threads through causality order, and `bar.sync` synchronizes with `bar.sync`. NCCL's single post thread's `fence.acq_rel.sys` therefore has to make every worker warp's remote stores visible at system scope before the flag store. Source: https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar and §8.8 (Release and Acquire Patterns).
  - Caveat: the "performed" wording is stated for `membar`/`fence.sc`; applying it to `fence.acq_rel` is an extrapolation.
- **PTX multimem.** `multimem.ld_reduce` loads from all locations behind the multimem address and reduces them; `multimem.st` stores to all of them (same document, multimem section).
- **CUDA Programming Guide, Memory Synchronization Domains** (https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/memory-sync-domains.html):
  - "Some CUDA applications may see degraded performance due to memory fence/flush operations waiting on more transactions than those necessitated by the CUDA memory consistency model."
  - The flush "may unnecessarily wait, fully or partially, on slower nvlink or PCIe writes from the communication kernel."
  - Domains help because "a fence will only order writes matching the fence's domain", but "ordering or synchronization between distinct domains on the same GPU requires system-scope fencing".
- **NCCL's domain does not protect A (inferred from the guide).** NCCL launches in the Remote domain (`enqueue.cc:1750,1800-1805`). Its fences are `.sys` (`op128.h:393-399`), which is exactly the scope domains do not isolate. The domain tag shields other kernels' GPU-scope fences from NCCL's traffic, not NCCL's own fences from B.

### 4.3 Inferred mechanisms, strongest first

1. **System-scope release waits on SM-issued remote writes, including B's.** The guide says this can happen, and the code puts such a fence on every Simple step.
   - Your probe isolates it. The release round trip rises **+24% at only 44 GB/s** of SM-put B, but only **+2% at 436 GB/s** of copy-engine B. The release's cost tracks SM-issued remote writes, not bytes on the link.
   - This would explain the low-B pattern: Ring Simple +11.3% and NVLS +11.8% (both fence every step) against LL128 +3.1% (no fence).
   - It may also explain A's self-interference: 32 NVLS CTAs mean more outstanding multimem stores per fence.
   - Not documented: whether an in-kernel fence on SM x waits for writes from SM y. The guide's example is a flush at kernel completion.
2. **Queueing at the endpoint's request/response path.**
   - Remote loads rise +11% under 436 GB/s of copy-engine B: bytes alone cost a little.
   - They rise +12% at 44 GB/s and +52% at 679 GB/s of SM-put B. SM-issued 16 B writes cost far more per byte, perhaps through request rate or the shared SM→hub queues (inferred).
   - At high B, Ring's ~457-464 GB/s plus 246-266 GB/s of B is ≈ 700-720 GB/s per direction, above the 679 GB/s SM-put ceiling you observed. This puts Ring in a backpressure regime, so the two Ring protocols converge (+29.9% and +33.0%). LL128 carries 16/15 more bytes.
3. **NVLS takes the maximum over 8 GPUs.**
   - Every `ld_reduce` and credit min-poll needs all 8 replicas.
   - Gather waits on 8 tails written by 8 different GPUs.
   - The fence after `multimem.st` must cover 8 replicas (code plus PTX; the acknowledgement mechanism is inferred).
   - NVLS therefore pays the worst endpoint several times per step, and is the most sensitive at high B (+45.6%) despite the lowest own link use.
4. **Not excluded:** B's CTAs sharing SMs with A's CTAs, and B's incoming writes inflating L2/HBM latency for NVLS's 16-round local scatter/gather chains.

### 4.4 Reading the probes

| Probe result | Interpretation |
|---|---|
| SM-put B at 44 GB/s: release round trip +24%, remote load +12% | Release is about 2× more sensitive than a plain remote load, at about 5% link load. Points at the fence/release path (mechanism 1). |
| Copy-engine B at 436 GB/s: round trip +2%, remote load +11% | Copy-engine writes do not enter what the SM release waits on. Raw bytes cost only modest read latency. |
| B only among GPUs 2-7 at 674 GB/s: +0% / +0% | The NVSwitch core is not the contended resource; the endpoints are. For an 8-rank A every GPU is an endpoint, and NVLS touches all 8 GPUs in every multimem operation. |
| Interpolated at B ≈ 75 GB/s: round trip ≈ +28%, load ≈ +14% (A: Simple ≈ +11%, LL128 +3%) | Consistent with fence and flag latency being about 40% of a Simple step (inferred). |
| Interpolated at B ≈ 300 GB/s: round trip ≈ +54%, load ≈ +28% (A: NVLS +46%, Ring Simple +30%) | NVLS stacks several round trips per step, each a max over 8 GPUs. Ring also enters the bandwidth regime. |

## 5. The slots argument

1. **The slots are not what bounds A at 64 MiB in any path.**
   - NVLS: 8 slots against 11 steps, and each role uses one slot at a time.
   - Ring Simple: a 4-slice credit window, but the receive-to-send dependency limits the lead to 2 slices.
   - LL128: 8 steps against 15 calls; thread 0 only spins when all 8 are outstanding (`prims_ll128.h:73`).
   - The binding concurrency is structural: Unroll × 16 B per thread, one step per role, and a fence plus flag per step. No environment variable changes it.
2. **Even if A were slot-bound, that would make B matter, not prevent it.** A window-limited flow has throughput = window / round-trip time. It cannot use idle bandwidth, because it has no more requests to issue. It slows by exactly the factor its loop's latency grows.
3. **Free bandwidth only protects a flow that has spare requests queued.** A is a closed loop, and B lengthens the latencies inside it: the fence/release, the flag delivery, and the multimem round trips over 8 GPUs. Your probes show those latencies do grow under SM-put B at link loads far below saturation.

## 6. Cheap experiments, ranked

1. **Run A under copy-engine B vs SM-put B at matched rates (75 and 300 GB/s), for all three paths.**
   - If mechanism 1 holds: the low-B slowdown of the Simple paths collapses to a few percent, and LL128 stays flat.
   - Any remaining high-B Ring slowdown under copy-engine B measures the bytes effect (mechanism 2).
2. **Fence-only probe.** One thread on GPU0 times a lone `fence.acq_rel.sys` (and `.gpu`) with no remote stores of its own. Run it under: SM-put B issued from GPU0, B that only targets GPU0, copy-engine B, and B launched in the Remote vs the Default domain.
   - Prediction: SM-put B issued from GPU0 inflates it, copy-engine B does not, and the domain makes no difference at `.sys` scope.
3. **Confirm the effective configuration from logs (free).** Run with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,COLL` and check:
   - "N coll channels … M nvls channels" (`init.cc:1646`), expecting 32 and 24;
   - "NVLS tuning: nChannels 24 chunkSize 131072" (`nvls.cc:242`);
   - the Ring `channel{Lo..Hi}` line;
   - that no buffer registration happened.
   A TRACE build shows `chunkBytes` = 32768 for NVLS (`enqueue.cc:833-841`).
4. **Actually change the per-step payload.** Use `NCCL_NVLS_NCHANNELS=16` or `8` (64 KiB chunks), `NCCL_MAX_NCHANNELS=8` for Ring (512 KiB slices), or a 256 MiB message. Fit T = F + N_steps·S alone and under B.
   - If the per-step fence and flag cost is the sensitive part, the relative slowdown at low B should fall as bytes per step rise.
   - This also separates "Ring is latency-bound" from "Ring is CTA-throughput-bound".
5. **Localize B.**
   - 2-rank A on GPUs 0-1 with B on GPUs 2-7: expect ~0%.
   - 8-rank NVLS A with B only between GPUs 6 and 7, at the same per-GPU rate: the max-of-8 coupling predicts most of the full-B slowdown.
6. **`NCCL_P2P_READ_ENABLE=1` for Ring Simple.** The fence then covers only local stores, and the data crosses GPUs as remote loads.
   - If the fence (mechanism 1) dominates, the low-B slowdown drops.
   - If link queueing (mechanism 2) dominates, it rises, tracking the +12% / +52% remote-load inflation.
7. **Registered NVLS buffers** (`ncclMemAlloc` + `ncclCommRegister`, so `regUsed=1`). This gives 17 multimem worker warps and no staging copies (`all_reduce.h:396-399`). It separates the staging chain from the multimem and fence path.
8. **No-GPU checks.**
   - `cuobjdump -sass` on the AllReduce kernels, to see whether the shared credit store issues after `MEMBAR`/`FENCE.SYS` (the same-warp credit coupling in 2.1).
   - Log `%smid` of B's CTAs, to rule out A and B sharing SMs.
