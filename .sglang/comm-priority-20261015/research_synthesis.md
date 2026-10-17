# Comm A over Comm B on 8x B200: making a background KV all-gather yield to TP collectives

**Scope:** 8x B200 (SM 10.0, NVLink5 through NVSwitch), driver 615.x, CUDA 13.x, NCCL 2.30.x. Comm A is a high-priority all-reduce, all-gather or reduce-scatter of 1-128 MB, often inside a CUDA graph. Comm B is a best-effort all-gather that collects KV-cache pages, from tens of MB to GBs.

**Evidence tags:**
- **[D]** documented by NVIDIA (docs or release notes)
- **[S]** observed in source code
- **[F]** forum post, blog, paper or third-party measurement
- **[I]** inferred
- **[U]** no source found
- **[V✓]** the verifier confirmed it
- **[V✎]** the verifier corrected it; the corrected version is used here

## Bottom line

- **No CUDA, driver or NCCL mechanism can preempt, pause or cancel an in-flight collective, kernel or copy-engine (CE) copy on demand. No priority or QoS mechanism exists for NVLink or HBM.**
  - Stream, launch and graph-node priorities only reorder blocks that are still pending.
  - Those priorities apply to compute kernels only, not to copies.
  - NCCL has no priority between communicators or operations.
  - NCCL's only stop primitives (Revoke and Abort) are terminal.
- **"Stop B now, resume later" must therefore be built from three parts:**
  1. B is split into idempotent chunks.
  2. A gate on the GPU is checked before each chunk. A's stream or graph closes it before A and reopens it after.
  3. Resume picks up from a chunk cursor or completion bitmap.
- **Pause latency is the gate wake-up time plus the B bytes already past the gate, divided by the bandwidth they share.** That is about 1.4 µs per MiB in flight at roughly 750 GB/s [I]. To hide even that, close the gate one op before A ("lookahead close").
- **Recommended:**
  - **D1 (gated CE pull):** a one-sided pull of B over copy engines, rank-local, with `cuStreamWaitValue32` before each chunk and the close/open writes captured in A's graph.
  - **D2 (yielding SM kernel):** a custom copy kernel that polls a flag and can use CLC. Use it when KV pages are small or scattered and issuing copies from the host becomes the bottleneck.
  - **Hygiene for both:** set the priorities, raise the work-queue counts, warm up every kernel, keep A off copy engines, and give B its own communicator or none at all.
- **Keep B out of two-sided NCCL collectives.** NCCL 2.30's default capture-time ordering, combined with a gated two-sided B, can produce a cross-rank wait cycle (deadlock) [I] (§3, D3).

---

## 1. Mechanism catalog

**Verifier corrections applied in this catalog:**
1. **Priority wording.** "May preempt" appears in CUDA 8.0 through 12.2.2. 12.3.0 is the first release that says priorities "do not preempt already-running work".
2. **Priority and copies.** The docs name only H2D/D2H copies, but D2D and peer copies fall under "only compute kernels are affected". Whether the driver's internal SM copy kernels follow stream priority is undocumented.
3. **Graph-priority versions.** Kernel-node priority arrived in 11.7. `cudaLaunchAttributePriority` was graph-nodes-only in 11.8 and became valid for streams and launches in 12.0.
4. **Timeslice.** The 2048 µs default comes from one NVIDIA moderator's forum answer, not from official docs.
5. **CLC.** It is not the only documented way to yield (short non-persistent blocks also yield), and its latency is unbounded.
6. **NCCL and Blackwell all-gathers.**
   - From 2.29.2, large symmetric all-gathers on Blackwell run on copy engines.
   - Operations on the same communicator, or on split-shared communicators, are serialized.
   - Priority matters only when A needs SMs that B holds.
7. **Memop gate.** That the device evaluates the condition at execution time is implied, not stated. A close racing a wait can let up to ~2 chunks through. Lazy module loading adds a deadlock hazard.
8. **Where memop nodes may live.** They are allowed in child graphs, but not in conditional bodies or device-launched graphs.
9. **Events.** They can still gate at enqueue time if the host keeps a bounded lookahead.
10. **Zero-SM memops.** This is inferred. NVSHMEM falls back to a 1-thread kernel.
11. **Batched memcpy under capture.** Not capturable; this is observed, not documented. The call returns error 900, and the copy only goes missing if the caller ignores that error.
12. **`NCCL_SYM_CE_THRESHOLD`.** It counts elements, not bytes. All-reduce and reduce-scatter never take the CE path.
13. **Single P2P logical CE.** Observed in driver source. That CUDA routes every peer copy to it is inferred, and one logical CE does not mean one FIFO.
14. **Green contexts.**
    - Kernels can run on more SMs than provisioned.
    - The restrictions may not apply to cross-device copies.
    - Whether other contexts are kept off the carved SMs is disputed.
15. **NVIDIA green-context blog numbers.** They are wall-clock latency of the critical kernel against a compute-only background load.
16. **Split granularity.** `IGNORE_SM_COSCHEDULING` exists since 12.5. Multiples of 2 SMs come from `coscheduledSmCount=2`.
17. **NCCL in a green context.** `cudaSetDevice` explains NCCL issue #1736. NCCL's internal streams live in the primary context.
18. **Memory isolation.** The L2/HBM contention result was measured on H100. An NVLink scheduling hint does exist (not a QoS).
19. **Zero-CTA collectives.**
    - Release notes and user docs disagree on whether AllGather is included.
    - Ineligible calls silently fall back to SM kernels.
    - Self-copies stay on SMs before CUDA 13.1.
20. **NCCL ordering.** `NCCL_GRAPH_STREAM_ORDERING=1` is the default since 2.30.
21. **Host peer pointers.** They can be NULL, and the source pages need a lease.
22. **nccl-tests numbers.** The table's flags are unverified, it is AllReduce only, and an 8 MB chunk runs at about 60% efficiency.
23. **Prior art.**
    - DFCCL preempts only stuck collectives.
    - ExpertPlex's numbers are compute tiles.
    - DualPath says nothing about NVLink.
    - XSched and GPreempt reach about 32-40 µs, but only with driver changes.
24. **Device-API barrier timeout.** A barrier `timeoutCycles` parameter is not in the docs.

### 1a. CUDA scheduling and preemption

| Mechanism (sources) | API (+min version) | Stops in-flight B? | Pause latency / granularity | SM cost | Frees NVLink? | Graph-capturable | Verdict |
|---|---|---|---|---|---|---|---|
| Stream priority [D][V✎] ([rt-stream](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html), [PG-host](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html), [12.2.2](https://docs.nvidia.com/cuda/archive/12.2.2/cuda-runtime-api/group__CUDART__STREAM.html), [12.3.0](https://docs.nvidia.com/cuda/archive/12.3.0/cuda-runtime-api/group__CUDART__STREAM.html)) | `cudaStreamCreateWithPriority`, `cudaDeviceGetStreamPriorityRange`, `CU_STREAM_ATTRIBUTE_PRIORITY`. Long-standing API. "Do not preempt" wording from 12.3.0. | No | Waits for blocks to retire, and only when A needs SMs that B holds. NCCL B CTAs stay resident for the whole collective [S]. H100 example: 0.9 ms wait ([GC guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html)). 148-SM Blackwell: 0.140 ms at high priority vs 3.727 ms at equal priority ([GC blog](https://developer.nvidia.com/blog/control-how-your-gpu-shares-work-with-green-contexts/)) [F]. | 0 | No. "only compute kernels ... are affected"; no HBM or NVLink arbitration [I]. | Captured kernel nodes copy the capture stream's priority. At replay the graph uses the launch stream's priority unless instantiated with UseNodePriority [D]. | Needed (A highest, B lowest), but it cannot stop B and does nothing for a CE-based B. |
| Per-launch / graph-node priority [D][V✓] ([rt-graph](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html), [env-vars](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html), [11.8 types](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-runtime-api/group__CUDART__TYPES.html), [12.0 types](https://docs.nvidia.com/cuda/archive/12.0.0/cuda-runtime-api/group__CUDART__TYPES.html)) | Node priority and `cudaGraphInstantiateFlagUseNodePriority`: 11.7. `cudaLaunchAttributePriority`: graph nodes in 11.8; streams and launches from 12.0. `CUDA_GRAPHS_USE_NODE_PRIORITY=1`. | No | Same as stream priority | 0 | No | Kernel nodes only. With UseNodePriority, exec-update cannot change priority. The 12.4 notes advertise "updatable graph node priorities" ([12.4 RN](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/index.html)); these conflict, so test. | Keeps A's kernel nodes high inside the forward graph. It is a hint only. |
| Instruction-level preemption / TSG time-slicing [F][V✎] ([forum 371720](https://forums.developer.nvidia.com/t/371720), [forum 78183](https://forums.developer.nvidia.com/t/how-high-priority-stream-preemption/78183), [Pascal WP](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf), [RTAS'24](https://www.cs.unc.edu/~jbakita/rtas24.pdf)) | Query-only `cudaDevAttrComputePreemptionSupported`. Admin `nvidia-smi compute-policy --set-timeslice=...` (values undocumented, [forum](https://forums.developer.nvidia.com/t/is-there-any-document-on-set-timeslice-on-nvidia-smi/156566)). Pascal+. | Yes, but only between contexts or processes without MPS | Timeslice expiry. 2048 µs default per an NVIDIA moderator (V100, r580), plus a context save. | Whole GPU; A and B never run concurrently | Only while B is switched out [I] | n/a | Reject. Use only as a reference baseline (`nsys --gpuctxsw=true`). |
| Host-initiated cancel/suspend [D, absence][I] ([drv-types](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html)) | None. The CUDA 13.4 `CU_LAUNCH_ATTRIBUTE_*` list has no preempt, cancel or suspend. | n/a | n/a | n/a | n/a | n/a | Any stop has to be cooperative. |
| Cluster Launch Control (CLC) [D][V✎] ([CLC](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html), [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) | PTX `clusterlaunchcontrol.try_cancel/query_cancel`, plus libcu++ wrappers. CC 10.0, PTX ISA 8.6, sm_100+; CUDA 12.8+ in practice [I]. | Partly. try_cancel fails "if ... a higher-priority kernel being scheduled". If the CTA then exits, the scheduler "can" run the higher-priority kernel and afterwards "will continue scheduling the remaining thread blocks". | One work item per CTA plus the try_cancel round trip. No bound is documented. | B's CTAs, only while working | Partly: CTAs that exit stop issuing; CTAs on SMs A does not need keep running [I] | Yes (an ordinary launch) | Best hardware-assisted yield for a custom SM-based B. It fires only under SM pressure, so a software flag is still needed to stop NVLink traffic. NCCL does not use CLC [S]. |
| Short non-persistent blocks [D][F][V✎] (comparison table on the CLC page; [forum 243831](https://forums.developer.nvidia.com/t/how-to-verify-that-high-priority-stream-is-served/243831)) | Ordinary launch | Partly, as blocks retire | One block | B's blocks | Partly | Yes | The simplest way to yield. Costs launch and block overhead. |
| PDL [D] ([PDL](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)) | Programmatic stream serialization. CC 9.0+; graph edge data from 12.3. | No | n/a | 0 | No | Yes | Does not apply: same-stream only. |
| Hardware work queues [D] ([env-vars](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)) | `CUDA_DEVICE_MAX_CONNECTIONS` (1-32, default 8). `CUDA_DEVICE_MAX_COPY_CONNECTIONS` (CC ≥8.0, default 8, which the 12.9 guide describes as "per async copy engine"; documented from 12.6). `CUDA_SCALE_LAUNCH_QUEUES`. | No | If A and B alias a queue, A waits for all of B. With a memop gate, aliasing can deadlock. | 0 | No | n/a | Mandatory: queue count ≥ active streams, never 1. Stream memops run on the compute channel ([nccl#2238](https://github.com/NVIDIA/nccl/issues/2238)), so the compute queue count matters even for a CE-based B. |
| Memory sync domains [D] ([MSD](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/memory-sync-domains.html)) | `cudaLaunchAttributeMemSyncDomain(Map)`, CUDA 12.0, CC 9.0+. NCCL tags its kernels Remote (`NCCL_MEM_SYNC_DOMAIN`, 2.16). | No | n/a | 0 | No | Yes | For an SM-based B, use a different domain from A so A's fences do not wait on B's NVLink writes. Unmeasured. |
| NVLink-utilization scheduling hint [D][S] ([rt-types](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html)) | `CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING`, CUDA 13.0, CC 10.0. NCCL's `nvlinkCentricSched` (undocumented in NCCL, default 0). | No | n/a | 0 | No: "a hint only ... no functional or performance guarantee" | Yes | Optional experiment on A's side. Not a priority. |

### 1b. Spatial isolation

| Mechanism | API (+min version) | Stops in-flight B? | Pause latency / granularity | SM cost | Frees NVLink? | Graph-capturable | Verdict |
|---|---|---|---|---|---|---|---|
| Green contexts, driver API [D][V✎] ([GC drv](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html), [12.5](https://docs.nvidia.com/cuda/archive/12.5.0/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html), [13.0.3](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html), [13.1.0](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html), [GC guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html)) | `cuGreenCtxCreate`, `cuDevSmResourceSplitByCount`: CUDA 12.4 (concurrency text present in the 12.4.0 docs). `cuGreenCtxStreamCreate` and `SPLIT_IGNORE_SM_COSCHEDULING` first appear in 12.5. `cuDevSmResourceSplit` and backfill: 13.1. | No | No pause. A finds reserved SMs free: 0.007 ms with an 8-SM green context vs 0.140 ms at high priority, against a compute-only load [F]. | Static partition. Default on CC ≥9.0: at least 8 SMs, in multiples of 8 (148 = 18x8 + 4). Multiples of 2 via `coscheduledSmCount=2` (13.1). | No. "For cross-device copies, green context restrictions may not be applied." No HBM/L2 partitioning ([arXiv 2501.16909](https://arxiv.org/pdf/2501.16909), H100). | Yes: a captured node runs in the capture stream's context [D]. A forum report of cross-green-context graph hangs on R610 is unchecked on 615 ([forum 385732](https://forums.developer.nvidia.com/t/cuda-graph-hangs-with-cross-green-context-edges-on-r610-driver-works-on-r580/385732)). | Use to bound an SM-based B. It is not a pause. It may run on more SMs than provisioned (under MPS active-thread-percentage, or with CDP on CC 9.x). Whether the primary context avoids the carved SMs is disputed ([forum 377440](https://forums.developer.nvidia.com/t/green-context-sm-provisioning-vs-the-primary-context/377440) vs [feature survey](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html)). |
| Execution contexts (runtime) and workqueue config [D] ([EC 13.1](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-runtime-api/group__CUDART__EXECUTION__CONTEXT.html), [WQ struct](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-runtime-api/structcudaDevWorkqueueConfigResource.html), [PyTorch GreenContext](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.green_contexts.GreenContext.html)) | `cudaGreenCtxCreate`, `cudaExecutionCtxStreamCreate`; `CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED` / `cudaDevWorkqueueConfigScopeGreenCtxBalanced`. All CUDA 13.1. PyTorch wrapper is beta. | No | Removes A waiting behind queued B work, best effort only | n/a | No | Undocumented for graph launches | Put B's streams in a BALANCED green context so B's memop waits never share a queue with A. Concurrency is "not guaranteed". |
| MPS: active-thread %, static partition, client priority [D] ([when-to-use](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html), [MPS env](https://docs.nvidia.com/deploy/mps/latest/appendix-environment-variables.html), [MPS arch](https://docs.nvidia.com/deploy/mps/architecture.html)) | `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`; static partitions in 8-SM chunks on Hopper+; `CUDA_MPS_CLIENT_PRIORITY=0/1`; `CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING`. Introduction versions [U]. | No | Priority is "only a performance hint" | Active-thread % is not a reservation; static partitions are exclusive | No | n/a | Poor fit. Green contexts give the same effect inside one process. |
| NCCL CTA caps [D] ([types](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html), [flags](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/flags.html), [2.28.3](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-28-3.html)) | `maxCTAs/minCTAs` (2.17); default max 16 on Blackwell (2.28.3); `CTAPolicy=EFFICIENCY` (2.27); per-call `ncclCollConfig_t` (2.31, not available in 2.30.x) | No | n/a | At most maxCTAs for B | Indirectly: fewer CTAs inject less | Yes | A static rate limit for an SM-based NCCL B. Issue #2421: AlltoAll ignores the per-call maxCTAs. |

### 1c. Stream-ordered gating and graph control flow

| Mechanism | API (+min version) | Stops in-flight B? | Pause latency / granularity | SM cost | Frees NVLink? | Graph-capturable | Verdict |
|---|---|---|---|---|---|---|---|
| `cuStreamWaitValue32/64` [D][V✎] ([MEMOP](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html), [11.7](https://docs.nvidia.com/cuda/archive/11.7.0/cuda-driver-api/group__CUDA__MEMOP.html)) | Driver API only. v1 in CUDA 8.0 (needed `NVreg_EnableStreamMemOPs=1`). v2 on by default from 11.7. Only v2 semantics from 12.0. Flags GEQ (cyclic) / EQ / AND / NOR / FLUSH. | No: blocks only the next op | Remaining in-flight op plus re-poll time. Re-poll is undocumented; GA100 PBDMA retries after MAN·2^EXP nvclk or switches channel ([dev_pbdma](https://github.com/NVIDIA/open-gpu-doc/blob/master/manuals/ampere/ga100/dev_pbdma.ref.txt)) [I]. A close racing a wait can let ~2 chunks through. | 0 SMs [I, strong]. NVSHMEM falls back to a 1-thread kernel without FLUSH_REMOTE_WRITES ([nvshmem sync.cpp](https://github.com/NVIDIA/nvshmem/blob/devel/src/host/comm/sync.cpp)). | Indirectly: stops new issue | Yes, as `CU_GRAPH_NODE_TYPE_BATCH_MEM_OP` (11.7). Allowed in child graphs; not in conditional bodies or device graphs; no runtime node type. | **Primary gate.** "Improper use ... may deadlock". Keep the gate word in plain local device memory. |
| `cuStreamWriteValue32/64` [D] | Same versions. `NO_MEMORY_BARRIER` is "not supported in the v2 API", so every write is fenced. | No | Runs in A's stream order | 0 [I] | No | Yes; NCCL records these during capture ([ce_coll.cc](https://github.com/NVIDIA/nccl/blob/master/src/ce_coll.cc)) [S] | Close and open the gate from A's stream or graph. Use absolute values so graph replay is idempotent. Writes to peer addresses: works single-node in NVSHMEM; fails on MNNVL and EGM [S]; unverified on B200. |
| `cuStreamBatchMemOp` (ATOMIC_REDUCTION, BARRIER, FLUSH) [D] ([13.1 types](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/group__CUDA__TYPES.html), [13.0 types](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__TYPES.html)) | Fewer than 256 ops per call. BARRIER (v2, 11.7). ATOMIC_REDUCTION OR/AND/ADD on u32/u64 from 13.1.0; check `CU_DEVICE_ATTRIBUTE_ATOMIC_REDUCTION_SUPPORTED`. | No | Same as the single ops | 0 [I] | No | Yes; capture of ATOMIC_REDUCTION is unverified | Reference-counted gate (ADD +1 / ADD 0xFFFFFFFF). Wait plus progress write in one batch. |
| `cuGraphExecBatchMemOpNodeSetParams` [D] ([drv-graph](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html)) | 11.7 | n/a | Applies to future launches | n/a | n/a | Yes | Epoch-numbered gates without re-capturing. |
| Conditional nodes [D][V✎] ([graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html), [blog](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes)) | IF/WHILE 12.3 (enhanced in 12.4); IF-ELSE/SWITCH 12.8. Condition set only on the device (`cudaGraphSetConditional`). | No | Per iteration, decided by a kernel | ≥1-thread kernel per decision | Only by choosing whether a memcpy runs | Native. Body may hold only kernel, empty, memcpy, memset, child and conditional nodes (no memops). Graphs with conditionals cannot be child graphs or cloned. | A secondary skip-gate. Peer memcpy in a body is allowed by the doc text; test it. |
| Device graph launch [D] ([blog](https://developer.nvidia.com/blog/enabling-dynamic-control-flow-in-cuda-graphs-with-device-graph-launch)) | 12.0; device-side node update 12.4 | No | ">2x lower than host launch" | The launching kernel | No | Device graphs hold only kernel, memcpy, memset and child nodes; limit 120 fire-and-forget launches | Optional relaunch path for an SM-based B. Cannot hold a zero-SM wait. |
| Node enable / device-updatable kernel nodes [D] | `cudaGraphNodeSetEnabled` (host, future launches); device-side kernel-node enable (version [U]) | No | Next launch | n/a | n/a | Yes | Skip only, not a block. |
| Events [D][V✎] ([rt-event](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)) | `cudaStreamWaitEvent` binds "at the time of the API call" | No | Bounded by host lookahead | 0 | No | Yes | Use for real data dependencies and for host-driven gating (D5). |
| `cudaLaunchHostFunc` [D] ([drv-exec](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html), [VCCL](https://arxiv.org/html/2510.00991v2)) | CUDA 10.0 | No | Host round trip | Host thread | No | Host nodes | Reject. The docs forbid exactly this synchronization pattern, CUDA calls are disallowed inside it, and VCCL hit deadlocks with it. |
| External semaphores [D] ([extres](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html)) | CUDA 10.0+ [I] | No | Undocumented | 0 [I] | No | Yes | Reject: needs an exporter (Vulkan, D3D12, NvSciSync). |
| Host-mapped gate word [D] | `cuMemHostRegister` + `cuMemHostGetDevicePointer` | No | Polling over PCIe/C2C is slower | 0 | n/a | n/a | Lets the CPU close the gate early with a plain store. |

### 1d. Copy engines

| Mechanism | API (+min version) | Stops in-flight B? | Pause latency / granularity | SM cost | Frees NVLink? | Graph-capturable | Verdict |
|---|---|---|---|---|---|---|---|
| Peer memcpy on CE [D] ([multi-GPU](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html), [nvbandwidth](https://github.com/NVIDIA/nvbandwidth/blob/main/testcases.cpp)) | `cudaMemcpyAsync`, `cuMemcpyPeerAsync`; P2P since CUDA 4.0 | No: no cancel exists | Whole copy | ~0. Same-device D2D runs as an SM kernel ([forum 371720](https://forums.developer.nvidia.com/t/millisecond-scale-d2d-memcpy-admission-latency-in-secondary-cuda-context-while-primary-context-kernel-is-running/371720)). | No: runs at engine/link speed | Yes, as memcpy nodes | B's data path. Confirm in nsys that copies show as Memcpy PtoP, not kernels. |
| `cudaMemcpyBatchAsync` [D][F][V✎] ([rt-mem](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html), [12.8 RN](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html), [13.0 RN](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html), [rmm#2518](https://github.com/rapidsai/rmm/issues/2518)) | 12.8; 13.0 removed `failIdx`; must not use the legacy stream | No | Whole batch; copies inside are unordered and concurrent | Hint only | No | No. Returns `cudaErrorStreamCaptureUnsupported`; the copy is missing only if that error is ignored (CUDA 13.3, r595) [F][S] | Spreads host issue cost across copies when B is issued eagerly. |
| `PreferOverlapWithCompute` flag [D] ([13.1 RN](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html)) | Enum since 12.8; honoured on non-Tegra from 13.1 | n/a | n/a | Keeps SMs free (hint): "Final engine selection depends on the platform and transfer" | No | Only through the batch and attribute APIs | Set it on B. Without it, NCCL leaves self-copies on SMs [S]. |
| `cuMemcpyWithAttributesAsync` [D] ([drv-mem](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html), [13.2 blog](https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features)) | 13.2 | No | One copy | Hint | No | Undocumented; test | Per-chunk B copy with the CE hint. |
| P2P logical-CE mapping [S][V✎] ([kernel_ce_gb100.c](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/615.78.08/src/nvidia/src/kernel/gpu/ce/arch/blackwell/kernel_ce_gb100.c)) | Driver 615.78.08: on NVSwitch, "only assign 1 LCE" for all P2P traffic (with auto-config on) | Unknown | n/a | 0 | No user control | n/a | A's CE traffic and B's CE traffic share one logical CE, fed by several channels that are time-sliced, with no CUDA priority. **Keep A on SMs.** |
| Driver (RM) channel-group controls [S] ([ctrla06c.h](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/615.78.08/src/common/sdk/nvidia/inc/ctrl/ctrla06c.h), [US9483423](https://patents.google.com/patent/US9483423)) | `SET_TIMESLICE`, `SET_INTERLEAVE_LEVEL` (privileged), `PREEMPT` | Per channel group | n/a | n/a | n/a | n/a | Not reachable from CUDA. The patent's copy-priority design is itself "block at submission", which supports the gate approach. |
| NVLink/NVSwitch QoS [I, absence] ([FM guide](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html)) | None found | n/a | n/a | n/a | n/a | n/a | Admission control at the source is the only lever. |

### 1e. NCCL

| Mechanism | API (+min version) | Stops in-flight B? | Pause latency / granularity | SM cost | Frees NVLink? | Graph-capturable | Verdict |
|---|---|---|---|---|---|---|---|
| Priority between communicators or ops [D, absence][V✓] ([types](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html), [roadmap#2272](https://github.com/NVIDIA/nccl/issues/2272)) | None in `ncclConfig_t` or `ncclCollConfig_t` (checked through 2.32.3) | n/a | n/a | n/a | n/a | n/a | Priority has to be built outside NCCL. |
| `trafficClass`, IB SL/TC, GIN TC [D] ([2.26.2](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-26-2.html), [env](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)) | 2.26.2; `ginTrafficClass` 2.30.3; `NCCL_GIN_IB_TC` 2.30.7; AUTO defaults in 2.32 | No | Packet arbitration | 0 | No: "network operations" only | Set at config time | Irrelevant on a single NVLink node. |
| NCCL kernel launch path [S][V✎] ([enqueue.cc 2.30.7](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/enqueue.cc), [master](https://github.com/NVIDIA/nccl/blob/master/src/enqueue/enqueue.cc)) | `grid={nChannels}`, `cuLaunchKernelEx` on the first user stream, no PRIORITY attribute. Same-communicator and split-shared ops are serialized through `sharedRes->deviceStream`. | No | Whole collective | Up to maxCTAs (16 by default on Blackwell) | No | Yes | A and B must never share a communicator or `splitShare`. |
| NCCL CE (zero-SM) collectives [D][S][V✎] ([bufferreg](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html), [2.29.2](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-29-2.html), [2.30.7](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-30-7.html), [ce_coll 2.30.7](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/ce_coll.cc), [blog](https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/)) | `CTAPolicy=NCCL_CTA_POLICY_ZERO` (2.28.3; release notes list AlltoAll/Gather/Scatter, while the docs and blog include AllGather, a conflict) with symmetric windows; driver ≥12.5. Automatic CE AllGather on Blackwell from 2.29.2 when `count > NCCL_SYM_CE_THRESHOLD` (8Mi **elements**), recv window symmetric, SM ≥100 and all-direct NVLink [S]. Hierarchical variant 2.30.6/2.30.7. Fixes: graph hang 2.30.3; 2-rank hang 2.31.2. | No: call granularity; a submitted batch is not cancellable | Whole call. Internal sync every 8 ops above 512 MiB [S]. `NCCL_CE_CHUNK_SIZE` is 8 MiB on master [S]. | 0 CTAs | No | Yes (falls back to per-op `cudaMemcpyAsync` under capture) | Option for D3. Hazard: A's own all-gather may land on CE and contend with B. Silently falls back to SM kernels when ineligible. |
| Symmetric windows and host peer pointers [D][V✎] ([deviceapi](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html), [device_setup](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html)) | `ncclCommWindowRegister` (2.27; called by all ranks; VMM or `ncclMemAlloc` buffers only). `ncclGetPeerDevicePointer` and siblings (2.29.2). | n/a | Chosen by the user | 0 with CE | n/a | Yes (pointers are static) | Foundation for D1 and D2. Returns NULL if the peer is not in the LSA team. Pointer lifetime is the shorter of window and communicator. |
| Device API (LSA) [D][V✎] ([2.28.3](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-28-3.html), [2.29.7](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-29-7.html), [device_memory](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_memory.html)) | 2.28.3, experimental, no ABI guarantee. LSA backward-compatible since 2.29. `ncclLsaCopy` 2.29.7. `ncclLsaCopyTma` sm100. Device-API timeouts 2.30.3; a barrier `timeoutCycles` parameter is **not in the docs**. | Cooperative (user code) | One tile plus a poll | User-chosen grid | Indirectly | Yes | Building block for D2. Recompile for each NCCL release. |
| One-sided `ncclPutSignal`/`ncclWaitSignal` [D] ([p2p api](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html), [2.30.3](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-30-3.html)) | 2.29.2 (CUDA ≥12.5); capture 2.30.3; batched CE 2.30.7; multiple contexts and signals 2.31. 2.30.7 known issue: every rank needs a warm-up call. | No: op granularity | One put chunk | 0 on NVLink | Indirectly | Yes since 2.30.3 | D3b. It is still an NCCL op, so the ordering rules may apply (undocumented). |
| Ordering controls [D][V✎] ([communicators](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html), [env](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html), [cudagraph](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html)) | `NCCL_LAUNCH_ORDER_IMPLICIT` (2.26.2; per-communicator in 2.31). `NCCL_GRAPH_STREAM_ORDERING` (2.30, **default 1**). `graphUsageMode`. `ncclCollConfig_t.launchCompletionEvent` (2.32). | No | n/a | n/a | n/a | Yes | Hazard. With ordering on, a captured A may queue behind B's NCCL op "across graphs and communicators that share a GPU". Turning it off requires "at most one NCCL operation ... at a time". |
| `ncclCommRevoke` [D] ([comms](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html), [2.28.7](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-28-7.html)) | 2.28.7 | Yes, but terminal | Next abort poll (~10k spins [S]). Afterwards, new collectives return `ncclInvalidUsage`. | n/a | Yes, once stopped | No | Emergency kill only. Re-init cost was ~0.7 s on 8xH100 ([2.22 blog](https://developer.nvidia.com/blog/memory-efficiency-faster-initialization-and-cost-estimation-with-nvidia-collective-communications-library-2-22/)). |
| `ncclCommAbort` / `ncclCommShrink(ABORT)` [D] | Abort is long-standing; Shrink 2.27 | Yes, terminal | n/a | n/a | n/a | No | Fault tolerance only. |
| `ncclCommSuspend`/`Resume` [D] | 2.29.7 | No: "no outstanding NCCL operations" | n/a | n/a | n/a | No | Memory offload; the name is misleading. |
| Chunked NCCL calls [F][V✎] ([nccl-tests#333](https://github.com/NVIDIA/nccl-tests/issues/333)) | Any version | Between calls | AllReduce on 8xB200: 8/16/32 MB take 30/47/84 µs (≈60/76/85% of the 816 GB/s asymptote); ~3 µs fixed cost. The test flags are unverified. | Per call | Duty cycle | Yes | Cost model for D3. 8 MB chunks are not cheap. |
| NVLS resources [D][V✎] ([2.32.3](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2323/release-notes/rel_2-32-3.html), [2.31.2](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2323/release-notes/rel_2-31-2.html)) | NVLink multicast for small AllGather (2.31.2). Slot-exhaustion fix and per-communicator `nvlsHostMode`: the verifier says 2.32.3, the researcher said 2.31 (**conflict**). | n/a | n/a | n/a | n/a | n/a | Check whether B's communicator takes multicast slots that A's NVLS all-reduce needs. |

### 1f. Prior art

| System | Mechanism | Stops in-flight B? | Pause latency / granularity | SM cost | Frees NVLink? | Graph-capturable | Verdict |
|---|---|---|---|---|---|---|---|
| P3, TicTac, ByteScheduler, PACE, Lina [F][V✎] ([Lina](https://arxiv.org/html/2210.17223v2), [PACE](https://i.cs.hku.hk/~cwu/papers/yxbao-infocom20.pdf), [ByteScheduler](https://i.cs.hku.hk/~cwu/papers/yhpeng-sosp19.pdf)) | Host priority queue over chunks; ByteScheduler's "credit" (bytes in flight); Lina 30 MB micro-ops | No. Lina: preempting an all-reduce "is not possible in current ... NCCL" | Chunk (MB to tens of MB), decided by the host | NCCL default | Admission only | No | Confirms the chunk-plus-credit model. The host decides too slowly for a captured A. |
| ChunkFlow [F][V✓] ([arXiv 2605.11335](https://arxiv.org/pdf/2605.11335)) | Chunked CE prefetch; host checks a flag, then `cudaStreamWaitEvent` on the collective's event | Partly: "no in-flight DMA is ever aborted" | One chunk; 16 MB best on PCIe; under 2% overhead (2x H100 PCIe, host-to-device) | 0 | Admission | Host-side flag | Closest template. D1 moves the flag check onto the GPU. |
| DFCCL [F][V✎] ([arXiv 2303.06324](https://arxiv.org/html/2303.06324v2)) | Persistent daemon kernel; aborts a primitive, saves chunk ID, resumes | Yes, but only **stuck** collectives | Per primitive; context save 0.05 µs, load 0.45 µs (RTX 3090) | Persistent CTAs | Yes | Inferred | Shows resumable abort of a collective is feasible; not priority-driven. |
| ExpertPlex [F][V✎] ([arXiv 2607.18002](https://arxiv.org/html/2607.18002v1)) | Tile-boundary preemption inside a persistent kernel | Yes (compute) | Tile intervals 2.2-25.3 µs (H800, **compute** tiles) | Persistent | n/a | n/a | Supports polling a flag per work item. |
| REEF, Hummingbird, Valve, XSched, GPreempt, Tally [F][V✎] ([REEF](https://usenix.org/system/files/osdi22-han.pdf), [Hummingbird](https://arxiv.org/html/2601.04071v2), [Valve](https://arxiv.org/html/2604.07874), [XSched](https://www.usenix.org/conference/osdi25/presentation/shen-weihang), [GPreempt](https://www.usenix.org/conference/atc25/presentation/fan), [Tally](https://arxiv.org/abs/2410.07381)) | Host splitting or queue reset (REEF-N 71-288 µs; Hummingbird 139 µs avg); channel disable (<1 ms with a driver patch); debug-interrupt or timeslice tricks (XSched P99 32 µs on GV100, GPreempt <40 µs on A100) | Yes, process-wide, or by killing idempotent kernels | 32 µs to ms | n/a | Not shown for CE DMA | No | Reject. Needs driver changes or undocumented ioctls, and works at context granularity. |
| ConServe [F] ([arXiv 2410.01228](https://arxiv.org/pdf/2410.01228)) | Layer safepoints; host flag (7 µs) plus NVLink broadcast (13 µs) to keep TP ranks in step | Partly | Layer | Small | No | Partly | Lesson: a partial-rank pause of a two-sided collective deadlocks, so B must be one-sided. Keep the flag in device memory. |
| T3 [F] ([arXiv 2401.16677](https://arxiv.org/html/2401.16677v1)) | Hardware memory-controller arbitration favouring compute (simulation only) | Yes (simulated) | Per memory request | 0 | HBM | n/a | Shows the knob that is missing. 1.4x all-reduce slowdown from GEMM co-run (simulated). |
| DualPath and IB virtual lanes [F][V✎] ([arXiv 2602.21548](https://arxiv.org/pdf/2602.21548)) | ~99% weighted round-robin for collectives on IB/RoCE | Partly | Per packet | 0 | No (NIC/PCIe only) | n/a | Says "existing GPUs do not support PCIe QoS" and that software shaping is impractical. Says nothing about NVLink. |
| CE offload: DeepEP V2.5, SymmetricMemory, Triton-distributed, ConCCL [F][D] ([DeepEP](https://github.com/deepseek-ai/DeepEP), [Triton-dist](https://triton-distributed.readthedocs.io/en/latest/_sources/getting-started/tutorials/07-overlapping-allgather-gemm.rst.txt), [ConCCL](https://arxiv.org/abs/2412.14335)) | All-gather on copy engines (`num_sms=0`); set_signal/wait_eq on the copy stream | No | Per copy | 0 | No | Memcpy nodes | Production precedent for D1's data path. That wait_eq maps to `cuStreamWaitValue32` is [I]. |
| NVSHMEM, MSCCL++, TileLink, FlashOverlap [F] ([NVSHMEM signal](https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html), [arXiv 2504.09014](https://arxiv.org/html/2504.09014v4)) | One-sided put/get per chunk with a signal | Partly | Tile or chunk | Varies | Indirectly | Device-initiated kernels: yes | Protocol template for D1 and D2 completion tracking. |
| Llumnix, PipeFill [F] ([Llumnix](https://arxiv.org/pdf/2406.03243), [PipeFill](https://arxiv.org/html/2410.07192v1)) | Append-only KV makes copies idempotent; size background work to known gaps | No | Stage or gap | n/a | n/a | Planned ahead | Supports idempotent resume and a planned per-gap budget for B. |

---

## 2. What CUDA cannot do (hard limits)

1. **Priority cannot preempt.** "Priorities provide a hint ... but do not preempt already-running work" ([rt-stream](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html), wording since 12.3.0). "The GPU does not reassess work queues during task execution" ([PG-host](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html)). [D]
2. **Priority does not apply to copies or memop nodes.** "Only compute kernels launched in priority streams are affected". Priority "may not be respected for memory transfers" ([async-exec](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)). In graphs, "priorities are only available on kernel nodes" ([rt-graph](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html)). So no copy-engine traffic can carry a priority. [D]
3. **No host-triggered cancel, suspend or preempt for a kernel or a copy.** The CUDA 13.4 launch-attribute list has none ([drv-types](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html)). "The programmer has no direct control over preemption" ([forum 78183](https://forums.developer.nvidia.com/t/how-high-priority-stream-preemption/78183)). [D, absence][F]
4. **Hardware preemption works only between contexts.** It switches at timeslice granularity (2048 µs default per an NVIDIA moderator, not in official docs, [forum 371720](https://forums.developer.nvidia.com/t/371720)). Without MPS, work from different contexts "cannot execute concurrently" ([MPS arch](https://docs.nvidia.com/deploy/mps/architecture.html)). [F][D]
5. **No NVLink, NVSwitch or HBM QoS.** No priority, virtual channel or rate control exists in the CUDA, NCCL or Fabric Manager docs [I, absence]. `NVLINK_UTIL_CENTRIC_SCHEDULING` is "a hint only" ([rt-types](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html)). NCCL `trafficClass` covers "network operations" only ([types](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html)). Green contexts "only partition SMs and not DRAM channels" ([arXiv 2501.16909](https://arxiv.org/pdf/2501.16909), H100).
6. **Green contexts guarantee neither concurrency nor isolation of copies.** "It is not guaranteed that the kernels launched in them will run concurrently or have forward progress guarantees." "For cross-device copies, green context restrictions may not be applied" ([GC drv](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html)). [D]
7. **The stream-to-work-queue mapping is opaque.** "The user has no direct control over the specific work queues". Aliasing creates "a false dependency" ([env-vars](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)). [D]
8. **CUDA does not see the ordering that stream memops create.** "Improper use of these APIs may deadlock the application. Synchronization ordering established through these APIs is not visible to CUDA" ([MEMOP](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html)). Memop nodes are barred from conditional bodies and device-launched graphs ([graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)). [D]
9. **A closed gate plus lazy loading can hang.** The first launch of a kernel needs device-wide synchronization, which a pending `cuStreamWaitValue32` can block ([forum 353972](https://forums.developer.nvidia.com/t/deadlock-when-using-custreamwaitvalue32-custreamwritevalue32-for-async-cross-stream-ordering/353972), [lazy-loading](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html)). [F][D]
10. **Events and host functions cannot act as a dynamic gate.** `cudaStreamWaitEvent` binds "at the time of the API call" ([rt-event](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)). A host function "must not make any CUDA API calls" and "must not perform any synchronization that may depend on outstanding CUDA work" ([drv-exec](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html)). [D]
11. **The copy engine cannot be chosen, and the P2P engine is shared.** Engine choice is a hint: "Final engine selection depends on the platform and transfer" ([13.1 RN](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html)). On NVSwitch the driver maps all P2P traffic to one logical CE ([kernel_ce_gb100.c](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/615.78.08/src/nvidia/src/kernel/gpu/ce/arch/blackwell/kernel_ce_gb100.c)) [S]. A CE-based A and a CE-based B therefore cannot be given separate engines.
12. **Batched memcpy cannot be captured.** It returns `cudaErrorStreamCaptureUnsupported` ([rmm#2518](https://github.com/rapidsai/rmm/issues/2518), CUDA 13.3/r595; NCCL comment in [ce_coll.cc](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/ce_coll.cc)). [F][S]
13. **CLC gives no latency bound.** The scheduler "can" start the higher-priority kernel ([CLC](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)). [D]
14. **Launch-completion ordering is best effort, and priority can invert it.** A kernel "may begin before all blocks of A have begun ... if B is a higher priority than A. Exercise caution if such an ordering inversion could lead to deadlock" ([rt-types](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html)). NCCL's implicit cross-communicator ordering relies on this. [D]
15. **MPS client priority is only a hint.** "CUDA priority levels are not guarantees of execution order" ([when-to-use](https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html)). [D]
16. **NCCL cannot pause an in-flight op or rank one communicator above another.** It has no priority field, Revoke and Abort are terminal, and Suspend needs no outstanding ops ([comms](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html)). Two communicators running at once carry the risk "CUDA gives no forward progress guarantees" ([nccl#574](https://github.com/NVIDIA/nccl/issues/574)). [D][F]
17. **Research systems that reach tens of µs need modified or undocumented driver paths.** XSched uses a debug-interrupt ioctl; GPreempt and Valve patch the driver. None is shown to stop CE DMA. [F]

---

## 3. Candidate designs

### Ranking summary

| Rank | Design | Expected pause bound | B throughput cost | SM cost of B | CUDA-graph fit | Main risk |
|---|---|---|---|---|---|---|
| 1 | **D1: gated CE pull** | T_wake + k·c/BW ≈ 1.4 µs per MiB in flight [I], plus up to one extra chunk from the race. ~0 exposed with lookahead close. | Per-chunk and host-issue overhead; gate-closed duty cycle | 0 [I] | A side: memop nodes in A's graph. B side: eager. | Work-queue aliasing or lazy-loading deadlock; shared P2P CE if A uses CE |
| 2 | **D2: yielding SM pull kernel** (flag + optional CLC) | Poll time + one tile per CTA (few µs [I]) | SMs and HBM while running; CTA count limits bandwidth | N CTAs running, 0 when paused | Ordinary kernels; yes | Experimental NCCL device API ABI; must never wait on a peer inside the loop |
| 3 | **D3: chunked NCCL zero-SM AllGather or `ncclPutSignal` + gate** | One NCCL call (~20-85 µs at 16-32 MB, AllReduce proxy [I]) | ~3 µs per call plus CE barriers; 76-85% efficiency at 16-32 MB (AllReduce proxy) | 0 | Yes | Ranks coupled per chunk; NCCL ordering can serialize A behind B and create a deadlock cycle [I] |
| 4 | **D4: static isolation** (priority + green context + CTA caps + queue config) | No pause; A's dispatch delay ~7 µs (blog, compute-only load) | Static cap on B's peak | Partition | Yes | NVLink and HBM contention untouched |
| 5 | **D5: host-driven chunk scheduling** (ChunkFlow/Lina style) | Host lookahead; ≥ tens to hundreds of µs | Host-bound | 0 (CE) | Poor: the host cannot see A inside a graph | Too coarse for TP collectives |

**Recommended stack:** D1, or D2 when pages are small or scattered, combined with the D4 hygiene items and a lookahead close. D3 is a fallback only if B must stay an NCCL collective. D5 serves as the baseline.

### D1 (rank 1): gated, rank-local copy-engine pull

**Control path.**
- One 32-bit gate word per GPU lives in plain local device memory (`cuMemAlloc`). Never put it at a multicast, EGM or fabric address: NVSHMEM reports memop failures there ([nvshmem sync.cpp](https://github.com/NVIDIA/nvshmem/blob/main/src/host/comm/sync.cpp)) [S].
- A's stream, or A's captured forward graph, closes and opens the gate with driver memops. Inside a runtime capture these are recorded as batch-memop nodes, as NCCL does [S].
- If several A producers exist (TP all-reduce, EP, custom all-reduce), use a reference count with `CU_STREAM_MEM_OP_ATOMIC_REDUCTION` ADD (CUDA ≥13.1; check the device attribute). Otherwise use absolute values, which make graph replay idempotent ([MEMOP](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html)).
- B sees the gate through a `cuStreamWaitValue32(gate, 0, EQ)` placed before every chunk on its own copy stream(s).

```text
// A side (recorded once into the forward graph; driver memops on the capturing stream)
cuStreamWriteValue32(s_a, gate, 1, 0)   // CLOSE: placed before the op that precedes A (lookahead) or right before A
<A: NCCL AR/AG/RS or custom AR kernel>
cuStreamWriteValue32(s_a, gate, 0, 0)   // OPEN
// multi-producer variant (CUDA >= 13.1): ATOMIC_REDUCTION ADD +1 / ADD 0xFFFFFFFF; B waits EQ 0

// B side (eager host loop on 1..k low-priority non-blocking streams; bounded pre-enqueue window W)
for chunk in plan:                       // chunk = up to c bytes, possibly many pages
    cuStreamWaitValue32(s_b, gate, 0, CU_STREAM_WAIT_VALUE_EQ)
    cudaMemcpyBatchAsync(dst[], src_peer[], sz[], n, {PreferOverlapWithCompute, srcAccessOrderStream}, ..., s_b)
    cudaEventRecord(ev[chunk], s_b)      // CUDA-visible dependency for KV consumers
    (optional) cuStreamWriteValue32(s_b, progress, chunk+1, 0)
```

**Data path.**
- One-sided pull of KV pages from peer windows. Peer pointers come from `ncclGetPeerDevicePointer` (NCCL ≥2.29.2), or from the user's own `cuMem` IPC mappings.
- Copies run on copy engines with `PreferOverlapWithCompute` (honoured from CUDA 13.1).
- `cudaMemcpyBatchAsync` (12.8+, cannot be captured) spreads the host issue cost.
- If B itself must be captured, use per-op `cudaMemcpyAsync` nodes with batch-memop wait nodes in **B's own graph**, not as sibling branches of A's graph. NVIDIA's 2022 graphs talk marks in/out dependencies through memops as "NOT PERMITTED" ([CUDA Graphs Update 2022](https://hihat.opencommons.org/images/3/34/CUDA_Graphs_Update_2022.pdf)) [F]. A's graph only writes, so it gains no inbound dependency [I].
- Check in nsys that peer copies on mapped or imported addresses show up as Memcpy PtoP, not as an SM copy kernel [open].

**Multi-rank consistency.**
- **Pause is rank-local.** No cross-rank protocol runs during the transfer, so a paused rank can never strand a peer inside a collective. This is the main advantage over D3.
- **Residual traffic:** after rank r closes its gate, peers that are still open keep pulling from r's HBM and NVLink egress. A is a TP collective, so every rank closes within the A-start skew δ. Bound on residual traffic at r: its own in-flight bytes plus, for each peer, that peer's in-flight bytes plus δ times its rate [I].
- **Optional global close:** rank r also writes the peers' gates. NCCL writes remote flags with a 4-byte `cudaMemcpyAsync` ([ce_coll.cc](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/ce_coll.cc)) [S]. `cuStreamWriteValue32` to a peer P2P address works single-node in NVSHMEM but is unverified on B200. Add it only if measured skew makes it necessary.
- **Source-page lease:** pages must not be evicted or overwritten while peers are pulling them. The consumer releases the lease after completion with a flag write [I].

**Resume.** Automatic. Waits queued behind a closed gate release once it reopens, and chunks not yet issued are untouched. No relaunch, no lost data, and no host action. The host only refills a bounded pre-enqueue window (`CUDA_SCALE_LAUNCH_QUEUES` limits how many can be queued).

**Expected pause latency.**
- L_pause ≈ T_wake + k·c/BW_shared, where k = copies in flight and c = chunk size [I].
- At ~750 GB/s: 7 × 256 KiB ≈ 2.4 µs; 7 × 1 MiB ≈ 10 µs; 7 × 4 MiB ≈ 39 µs; 1 × 1 MiB ≈ 1.4 µs [I].
- A close that races a wait being evaluated can let one more chunk per stream through ([V✎] on the memop claim).
- T_wake is unmeasured. The GA100 host re-polls after MAN·2^EXP nvclk or switches channel [I].
- **Lookahead close:** put the close before the op that precedes A, with lead time ≥ L_pause, so the drain overlaps compute and A sees ~0. The lead granularity is one op boundary, and B gives up that op's window.
- For large A (e.g. 128 MB AllReduce, roughly 300 µs interpolated from [nccl-tests#333](https://github.com/NVIDIA/nccl-tests/issues/333) [I]), closing right at A may be enough. Small A (1-4 MB, ~10-20 µs) needs lookahead.

**B throughput cost.**
- (a) B runs only while the gate is open; with 2 A collectives per layer, B loses (A + lead) twice per layer.
- (b) A satisfied wait plus a copy per chunk has a device-side cost (unmeasured).
- (c) **Host issue rate.** DualPath measured ~5-7 µs per `cudaMemcpyAsync` (H2D) ([arXiv 2602.21548](https://arxiv.org/pdf/2602.21548)). At 7 calls × 1 MiB per round, one host thread sustains only ~150-200 GB/s [I]. Use the batch API, larger chunks with lookahead, or D2 for scattered pages.
- (d) Per-peer concurrency needed to saturate NVLink5 is unmeasured.

**CUDA-graph compatibility.**
- A side: full. Memop nodes are allowed at top level and in child graphs; they do not carry priority.
- Check whether memop nodes break PDL edges in A's graph [open].
- B side: eager is simplest; a separate B graph also works.

**Failure modes and mitigations.**
1. **Work-queue aliasing deadlock.** A blocked wait on B's stream stalls A's reopen write, or A's kernels, when they share a hardware queue. Memops run on the **compute** channel ([nccl#2238](https://github.com/NVIDIA/nccl/issues/2238)). Mitigation: `CUDA_DEVICE_MAX_CONNECTIONS=32`, `CUDA_DEVICE_MAX_COPY_CONNECTIONS=32`, few B streams, and ideally B's streams in a BALANCED-workqueue green context (13.1) [D][I]. Never run with 1 connection.
2. **Lazy-loading hang** while the gate is closed. Set `CUDA_MODULE_LOADING=EAGER` and warm up every A shape and algorithm (NCCL picks kernels per size) and every B kernel with the gate open. In one forum thread EAGER alone did not fix the hang ([forum 353972](https://forums.developer.nvidia.com/t/deadlock-when-using-custreamwaitvalue32-custreamwritevalue32-for-async-cross-stream-ordering/353972)).
3. **A on copy engines.** NCCL ≥2.29.2 switches large symmetric-window all-gathers on Blackwell to CE, and they would share the single P2P logical CE with B [S]. Keep A's buffers out of symmetric windows, or raise `NCCL_SYM_CE_THRESHOLD` (undocumented env, counted in elements). All-reduce and reduce-scatter are unaffected.
4. **Copy silently runs as an SM kernel** (D2D classification, or a pre-13.1 driver). Verify in nsys.
5. **Memory ordering.** Every gate write is fenced (v2 has no NO_MEMORY_BARRIER). Data dependencies for KV consumers go through events, as the memop docs recommend. Do not rely on the progress word alone.
6. **Graph replay drift.** Use absolute gate values, or update epoch values with `cuGraphExecBatchMemOpNodeSetParams`.
7. **Starvation of B** under back-to-back A collectives. Add a minimum-progress policy: skip the lookahead when B's deadline nears. DualPath reserves ~1%; T3 uses a starvation timer [F].

### D2 (rank 2): yielding SM pull kernel (flag poll + optional CLC)

**Control path.** Same gate as D1, closed and opened by A's stream or graph memops. B's CTAs read it with `ld.acquire.gpu` before each work item [I].

**Data path.**
- A custom kernel pulls peer pages through LSA pointers: NCCL Device API (2.28.3+, experimental, recompile per release) with `ncclLsaCopy` / `ncclLsaCopyTma` (2.29.7, sm100), or raw `cuMem` IPC peer pointers.
- The page table is a device-resident descriptor, so scattered pages need no host enqueue. This is D2's main advantage over D1.
- Launch at the lowest priority, optionally in a green context with 8-16 SMs and a separate memory-sync domain. Use 2-16 CTAs; "More CTAs are needed to drive peak bandwidth" ([NCCL tuning blog](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication)).

```text
kv_pull<<<n_cta, ..., s_b_lowprio>>>(plan, gate, done_bitmap):
  for item in my_items():                       // CLC variant: item = current block; try_cancel next
      if ld_acquire(gate) != 0: return          // leave the rest; bitmap records progress
      if done_bitmap[item]: continue            // idempotent resume
      copy_page(plan.src_peer[item], plan.dst[item])   // ld/st or TMA over NVLink
      st_release(done_bitmap[item], 1)
// stream s_b: [kv_pull][WaitValue32(gate==0)][kv_pull][WaitValue32]...  (refilled by host)
```

**Multi-rank consistency.**
- Pull with **no per-chunk cross-rank barrier**, so the pause is rank-local, as in D1.
- Completion goes to the consumer's bitmap. Final notification to the owner is a flag write that releases the lease, or a device-API barrier with a timeout. Device-API timeouts exist from 2.30.3, but the barrier timeout parameter is not documented, so verify it.
- The kernel must never spin on a peer inside the item loop. That is the forward-progress hazard NCCL warns about ([nccl#574](https://github.com/NVIDIA/nccl/issues/574)).

**Resume.**
- Pre-enqueued `[wait gate==0][kv_pull]` pairs. Each relaunch skips items already done, so resume costs zero SMs while paused.
- With CLC (CC 10.0), CTAs exit when try_cancel fails because A's higher-priority kernel is pending, and hardware schedules the remaining blocks after A ([CLC](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)).
- Blocks that launch while the gate is closed must leave their item undone in the bitmap, not drop it [I].
- CLC only fires under SM pressure. If B is confined to a green context, A never contends for B's SMs, so CLC and green-context confinement are alternatives, not complements [I].

**Expected pause latency.** Flag visibility plus one item per CTA. Per-CTA NVLink throughput on B200 is unmeasured, so the latency for 64-256 KB tiles is unknown; "few µs" is [I]. The ExpertPlex analog bounds preemption by one tile (2.2-25.3 µs compute tiles) [F].

**B throughput cost.** SMs and HBM bandwidth while running; a poll per item (negligible [I]); CTA count against achieved bandwidth.

**CUDA-graph compatibility.** B is an ordinary kernel (capturable). The A side is the same as D1. Node priority survives capture only with UseNodePriority, or when the launch stream already has it.

**Failure modes.**
- Persistent CTAs that never exit defeat priority. Always exit on a closed gate.
- Cluster launches inside small green contexts: avoid `IGNORE_SM_COSCHEDULING` if B uses clusters [I].
- NCCL device-API ABI churn.
- A's memory fences waiting on B's NVLink writes: use a separate memory-sync domain.
- The D1 lazy-loading and work-queue rules apply because of the memop waits.

### D3 (rank 3): chunked NCCL zero-SM AllGather (3a) or `ncclPutSignal` (3b) behind a gate

**Control path.** The gate memop sits before each NCCL call on B's stream. B has its own communicator: `splitShare=0`, `CTAPolicy=NCCL_CTA_POLICY_ZERO`, symmetric windows, driver ≥12.5, NCCL ≥2.30.3 for the CE graph-hang fix.

**Data path.**
- **3a:** `ncclAllGather` in chunks of 16-32 MB. Ready/push/complete barriers use CE plus memops [S]. KV pages are not contiguous equal per-rank segments, so 3a needs SM pack/unpack kernels [I].
- **3b:** one-sided `ncclPutSignal` per page run per peer (2.29.2; capturable from 2.30.3; warm-up known issue in 2.30.7; `sigIdx` and `ctx` must be 0 before 2.31).

**Multi-rank consistency.**
- **3a is two-sided.** Every rank must issue the same chunk sequence in the same order. A rank whose gate opens early enters chunk k+1 and blocks on the CE barrier memop. That costs zero SMs, but it occupies a compute work queue.
- **Deadlock cycle [I]:**
  - NCCL may serialize A behind B's in-flight op: same or split-shared communicator, `NCCL_LAUNCH_ORDER_IMPLICIT=1`, or the default `NCCL_GRAPH_STREAM_ORDERING=1` "across graphs and communicators that share a GPU" ([env](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)).
  - In that case rank q's A waits on B chunk k+1 on q.
  - That chunk waits on rank r's B chunk k+1.
  - r's chunk waits on r's gate.
  - r's gate waits on r's A.
  - r's A waits on q's A. The loop closes.
  - Disabling graph stream ordering requires "at most one NCCL operation ... on a given GPU at a time", which rules out running A and B concurrently.
- **3b is one-sided,** so pausing is rank-local. Whether the NCCL ordering rules apply to it is undocumented [open].

**Resume.** Automatic for calls not yet issued.

**Pause latency.** One call: 16/32 MB AllReduce takes 47/84 µs on 8xB200 ([nccl-tests#333](https://github.com/NVIDIA/nccl-tests/issues/333), flags unverified). An all-gather moves about half the bytes [I], so roughly 20-45 µs [I]. NCCL's internal CE sub-chunking is not exposed.

**B throughput cost.** ~3 µs fixed per call, plus two barrier rounds per call; ~60% efficiency at 8 MB.

**Graph fit.** Yes. NCCL falls back to per-op memcpy under capture [S].

**Failure modes.** The deadlock cycle above; NVLS multicast slot pressure from a second communicator (fixed and given `nvlsHostMode` in 2.32.3 per the verifier); silent fallback to SM kernels when ineligible.

**Verdict:** use only if B must be an NCCL collective, and prefer 3b over 3a.

### D4 (rank 4, complement and baseline): static isolation

**Control path.** None (static configuration).

**What it involves.**
- A on the greatest-priority stream; per-node priority kept through graphs with `cudaGraphInstantiateFlagUseNodePriority` or `CUDA_GRAPHS_USE_NODE_PRIORITY=1`.
- B on the least-priority stream.
- An SM-based B confined to an 8-16 SM green context with BALANCED workqueues (CUDA 13.1), or capped with `maxCTAs` / `EFFICIENCY`.
- `CUDA_DEVICE_MAX_CONNECTIONS=32` and a separate memory-sync domain for B.

**Effect.** A's dispatch latency falls to ~7 µs (blog, compute-only load). B is **not stopped**: NVLink and HBM contention remains (up to 1.3x on H100 for a copy kernel on disjoint SMs, [arXiv 2501.16909](https://arxiv.org/pdf/2501.16909)), and CE copies are not bound by green contexts at all.

**Multi-rank and resume.** Not applicable.

**Failure modes.**
- NCCL inside a green context: the current context changes ([nccl#1736](https://github.com/NVIDIA/nccl/issues/1736)), and NCCL's internal streams live in the primary context.
- A forum report of graph hangs across green contexts on R610; check on 615.
- Disputed whether the primary context stays off the carved SMs.

### D5 (rank 5, baseline): host-driven chunk scheduling

**How it works.** The CPU issues B chunks with `cudaStreamWaitEvent` on A's latest event, or checks a host flag first (ChunkFlow, [arXiv 2605.11335](https://arxiv.org/pdf/2605.11335)).

**Limits.**
- Pause bound = host lookahead × chunk size.
- It does not work when A sits inside a captured forward graph: the host only knows when the whole graph launches.
- Host-side splitting reaches ~139 µs on average (Hummingbird); a host flag read costs 7 µs (ConServe) [F].
- Keep it only as the reference implementation.

**Rejected outright.** Stream priority alone; separate-process time-slicing (~2 ms); MPS; `ncclCommRevoke`/`Abort`; `ncclCommSuspend`; host functions; external semaphores; driver-ioctl preemption (XSched, GPreempt, Valve).

---

## 4. Things that must be measured, and the benchmark

### 4.0 Register of unverified or conflicting claims (check first)

| # | Claim | Status |
|---|---|---|
| U1 | Stream memops use zero SMs (front-end semaphore acquire) | [I], strong |
| U2 | Gate wake-up / re-poll latency on B200 | Unknown |
| U3 | `cuStreamWriteValue32` to peer P2P/VMM addresses on B200 | Untested; fails on MNNVL and EGM per NVSHMEM [S] |
| U4 | CUDA routes every peer copy to the single P2P logical CE | [I] from driver source |
| U5 | NCCL in a green-context stream is confined to its SMs | [I] |
| U6 | Whether the primary context avoids carved SMs | Conflicting (Programming Guide vs forum measurement) |
| U7 | `NCCL_GRAPH_STREAM_ORDERING=1` serializes a captured A behind B on distinct, non-shared communicators | Docs suggest yes; source suggests per-sharedRes only |
| U8 | CE AllGather in 2.28.3 | Release notes vs docs/blog conflict; settled from 2.29.2/2.30.7 |
| U9 | `nvlsHostMode` version | 2.31 (researcher) vs 2.32.3 (verifier) |
| U10 | Device-API barrier `timeoutCycles` parameter | Not in the docs |
| U11 | `cuMemcpyWithAttributesAsync` capturability | Undocumented |
| U12 | ATOMIC_REDUCTION memop capturability | Undocumented |
| U13 | Peer memcpy nodes inside conditional bodies | The doc text implies allowed; untested |
| U14 | nccl-tests #333 flags (symmetric, graphs) | Unverified; AllReduce only |
| U15 | 2048 µs default timeslice | One moderator post |
| U16 | Updatable node priority (12.4 notes) vs exec-update restriction under UseNodePriority | Tension |
| U17 | Memop nodes in A's graph keep PDL edges and launch fast paths | Unknown |
| U18 | Triton-distributed wait_eq maps to `cuStreamWaitValue32` | [I] |
| U19 | CUDA 13.x minor version on the target | Many features need ≥13.1 (ATOMIC_REDUCTION, CE hint honoured, execution contexts, workqueues); ≥13.2 for `cuMemcpyWithAttributesAsync` |

### 4.1 Consolidated measurements

**Capability probes (P0)**
- **M1.** `cudaDeviceGetStreamPriorityRange`, `asyncEngineCount`, `MemSyncDomainCount`; memop attributes (`CAN_USE_STREAM_WAIT_VALUE_NOR_V2`, `CAN_USE_64_BIT_STREAM_MEM_OPS`, `CAN_FLUSH_REMOTE_WRITES`, `ATOMIC_REDUCTION_SUPPORTED`); green-context SM resource (smCount, minimum partition, alignment).
- **M2.** Capture checks: memop nodes in A's graph, ATOMIC_REDUCTION, `cuMemcpyWithAttributesAsync`, peer memcpy in a conditional body. Also whether captured NCCL kernel nodes keep high priority with and without UseNodePriority.
- **M3.** Engine classification: does a peer `cudaMemcpyAsync` or batch copy on raw P2P, VMM-imported or NCCL-window addresses appear as Memcpy PtoP or as a kernel, with and without `PreferOverlapWithCompute`?

**Gate mechanics (P1)**
- **M4.** T_close: time from the close write until B's next op is held. T_open: time from the open write until the next B copy starts. Measure both blocked and already-past cases, local and remote close (4-byte CE write vs memop to a peer).
- **M5.** Cost of a satisfied wait plus a copy per chunk; whether batching wait + write in one `cuStreamBatchMemOp` helps.
- **M6.** Aliasing and deadlock: `CUDA_DEVICE_MAX_CONNECTIONS` 1/8/32 and `COPY_CONNECTIONS` 8/32 with many streams, NCCL internal streams included. Does a blocked B wait stall A's kernels or copies, or unrelated H2D/D2H?
- **M7.** Lazy-loading hang reproduction: closed gate plus first launch of a new NCCL kernel; LAZY vs EAGER, with and without warm-up.

**Copy-engine behaviour (P1)**
- **M8.** Bandwidth vs chunk size (64 KiB-256 MiB), push vs pull, with 1/2/4/7 concurrent peers or streams. Find the smallest chunk that reaches ≥90% of peak.
- **M9.** Host issue cost per call: `cudaMemcpyAsync`, `cudaMemcpyBatchAsync`, `cuStreamBatchMemOp`. Can one thread sustain the target chunk rate?
- **M10.** Delay of a CE-based A (all-gather forced onto CE) behind B's in-flight chunk on the shared logical CE.
- **M11.** NVLink and HBM arbitration between SM-issued A traffic and CE-issued B traffic.

**Interference (P2)**
- **M12.** A's latency inflation for each B variant × A type and size, eager and captured.
- **M13.** Pause latency against chunk size and in-flight depth k, compared with the model T_wake + k·c/BW.
- **M14.** B goodput and duty cycle under the gate (at-A vs lookahead), and cost per pause/resume cycle.

**SM-based B (P2)**
- **M15.** D2 poll overhead, yield latency per tile size, exit-and-relaunch vs resident CTAs.
- **M16.** CLC on B200: how quickly try_cancel starts failing after A's launch; whether all or only some B CTAs exit; whether B CTAs re-steal on SMs A does not use.
- **M17.** Green context vs `maxCTAs` caps: A's dispatch latency and B's bandwidth at 8/16/32 SMs.

**NCCL interactions (P3)**
- **M18.** `NCCL_GRAPH_STREAM_ORDERING` 0/1 and `NCCL_LAUNCH_ORDER_IMPLICIT` 0/1 with captured A and NCCL-based B. Is A serialized? Try to reproduce the D3 deadlock cycle under injected skew.
- **M19.** `ncclPutSignal` ordering behaviour; NVLS multicast slot pressure with a second communicator; whether `nvlinkCentricSched` on A's communicator changes A's NVLink share.
- **M20.** Cost of Revoke plus re-split on 8xB200, to confirm it is unusable as a pause.

**Multi-rank (P3)**
- **M21.** Skew in A start times across ranks. Residual pulls into a closed rank: is a local gate enough, or is a global close needed?
- **M22.** Starvation: B progress at realistic A frequency (2 A per layer).

**Other**
- **M23.** Same-context preemption ever observed? Use %globaltimer/%smid gaps in long B CTAs, plus `nsys --gpuctxsw=true`.
- **M24.** Green-context graph hang on driver 615 (the R610 report).
- **M25.** Overhead of memop nodes in A's graph: extra latency per node and effect on PDL edges.

### 4.2 Benchmark design

**Harness.**
- A single C++/CUDA binary, `comm_prio_bench`, running 8 ranks, one process per GPU.
- Uses the driver API for memops, NCCL 2.30.x (≥2.30.3) and CUDA ≥13.1.

**Each iteration emulates one layer.**
- `s_compute` runs a busy kernel: configurable duration (e.g. 50-300 µs) and SM fill (idle, 50%, or saturating).
- `s_a` runs A. Eager mode, or captured with the compute kernel into one "forward graph" of L layers × 2 A collectives.
- `s_b` runs B continuously.

**Instrumentation.**
- NVTX ranges per phase on the host. Optional NCCL NVTX ranges [I].
- 1-thread marker kernels that record `%globaltimer` right before the gate close and open and around A. These pin the close and open moments on the GPU timeline, since memops may not appear in nsys [open].
- B per-chunk events for timing.
- For D2, per-CTA `%globaltimer`/`%smid` exit stamps.

**Metrics.**
- L_pause = end of the last B transfer after the close marker − close marker.
- L_resume = start of the first B transfer after the open marker − open marker.
- I_A = A duration with B − A duration alone.
- B goodput (bytes/s) and B duty cycle.
- A's NVLink share from GPU metrics.

**Proposed targets (design goals, not sourced facts):**
- I_A ≤ 5% across 1-128 MB with gate plus lookahead.
- L_pause ≤ 10 µs at the chosen chunk size.
- B goodput ≥ 80% of open-time peak.

**Matrix (primary sweep in bold; the rest is secondary):**

| Dimension | Values |
|---|---|
| A type | **NCCL AllReduce (NVLS and ring)**, **ReduceScatter**, **AllGather (SM)**, AllGather forced to CE (symmetric window above threshold), custom all-reduce kernel |
| A size | **1, 4, 16, 64, 128 MB** |
| A mode | **eager**, **captured with UseNodePriority**, captured without |
| B variant | **none**; B0 NCCL AllGather (SM, default CTAs); B1 B0 with `maxCTAs=2`/EFFICIENCY; B2 monolithic NCCL CE AllGather (ZERO); **B3 = D1**, chunk ∈ {**256K, 1M, 4M**, 16M} × k ∈ {**1, 7**}; **B4 = D2**, tile ∈ {64K, **256K**, 1M} × CTAs ∈ {2, **8**, 16} × {±CLC} × {±green context 16 SMs}; B5 = D3a/D3b, chunk ∈ {4, 16, 32} MB; B6 = D5 host-gated |
| Gate placement | none, **at A**, **lookahead (one op earlier)** |
| Compute background | idle, **saturating** |
| Environment | `CUDA_DEVICE_MAX_CONNECTIONS` ∈ {1, 8, **32**}, `COPY_CONNECTIONS` ∈ {8, **32**}, `CUDA_MODULE_LOADING` ∈ {LAZY, **EAGER + warm-up**}, priorities {equal, **split**}, `NCCL_GRAPH_STREAM_ORDERING` ∈ {0, **1**}, `NCCL_LAUNCH_ORDER_IMPLICIT` ∈ {**0**, 1} |
| Skew injection | delay one rank's A by {0, 10, 50} µs |
| Reference | B in a separate process without MPS (time-slicing), recorded with `--gpuctxsw=true` |

**Phases.**
- P0: probes M1-M3, single process.
- P1: gate and CE microbenchmarks M4-M11, on 2 GPUs then 8.
- P2: interference M12-M17, 8 GPUs, primary sweep.
- P3: hazards and NCCL interactions M18-M24.
- P4: forward-like loop (L layers × 2 A, B continuous), reporting per-layer time and B goodput for D1, D2 and D4 against the baselines.

**nsys capture.** Inside `zsk-sglang`, GPU metrics come from one rank only so samplers do not collide [I]. Verify every flag against `nsys profile --help` and the [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).

```bash
docker exec -it zsk-sglang bash -lc '
export CUDA_DEVICE_MAX_CONNECTIONS=32 CUDA_DEVICE_MAX_COPY_CONNECTIONS=32 CUDA_MODULE_LOADING=EAGER
OUT=/home/scratch.shunkangz_gpu_1/comm-prio-bench/nsys; mkdir -p $OUT
cat > /tmp/nsys_rank.sh <<EOF
#!/bin/bash
R=\${OMPI_COMM_WORLD_RANK:-\$RANK}
GM=""; [ "\$R" = "0" ] && GM="--gpu-metrics-devices=0 --gpu-metrics-frequency=100000"   # verify set incl. NVLink rx/tx: --gpu-metrics-set=help
exec nsys profile -t cuda,nvtx,osrt --cuda-graph-trace=node \$GM \
     --capture-range=cudaProfilerApi --capture-range-end=stop \
     -o $OUT/\${TAG}_r\$R "\$@"
EOF
chmod +x /tmp/nsys_rank.sh
TAG=ar16M_d1_c1M_k7_look mpirun -np 8 --allow-run-as-root -x TAG \
   /tmp/nsys_rank.sh ./comm_prio_bench --A allreduce --A-bytes 16M --A-mode graph \
   --B gated-ce --chunk 1M --streams 7 --gate lookahead --compute saturate --iters 200'
```

**Notes on the nsys setup.**
- `--cuda-graph-trace=node` shows each kernel inside the graph.
- NVLink rx/tx metrics are `nvlrx__bytes` / `nvltx__bytes` (`.avg.pct_of_peak_sustained_elapsed`). Caveat from the guide: inactive links "may appear as fully utilized", and GPU Metrics cannot run alongside DCGM profiling.
- For NCCL CE ranges (ce-coll, ce-sync, ce-batch), nsys ≥2025.6.1 with NCCL ≥2.29 has an NCCL tracing option. The option name is unverified.
- Analysis: `nsys export --type sqlite`, then join the kernel, memcpy (copyKind PtoP) and NVTX tables to compute L_pause, L_resume, I_A and goodput per iteration. Table names are [I].

**What the timelines should show.**
1. **Ungated** (B3 with gate=none): B's PtoP memcpy rows overlap A's NCCL kernel on every rank, A's kernel is stretched (I_A > 0), and NVLink receive stays saturated through A.
2. **Gated at A**: the B memcpy row shows a gap starting at most one chunk after the close marker and ending at the open marker. A is shortened.
3. **Gated with lookahead**: the gap starts during the preceding compute kernel, and A runs with B's NVLink traffic already at ~0.
4. **D2**: B kernel CTAs end within one tile after close, and the `[wait][kv_pull]` relaunch appears after open.
5. **D3a under skew** with ordering on: watch for the stall or deadlock cycle.

---

## 5. Sources

**CUDA documentation**
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
- https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html
- https://docs.nvidia.com/cuda/archive/12.2.0/cuda-runtime-api/group__CUDART__STREAM.html
- https://docs.nvidia.com/cuda/archive/12.2.2/cuda-runtime-api/group__CUDART__STREAM.html
- https://docs.nvidia.com/cuda/archive/12.3.0/cuda-runtime-api/group__CUDART__STREAM.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html
- https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html
- https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TYPES.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__GRAPH.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/unioncudaLaunchAttributeValue.html
- https://docs.nvidia.com/cuda/archive/11.6.1/cuda-driver-api/unionCUkernelNodeAttrValue__v1.html
- https://docs.nvidia.com/cuda/archive/11.7.0/cuda-driver-api/unionCUkernelNodeAttrValue__v1.html
- https://docs.nvidia.com/cuda/archive/11.8.0/cuda-runtime-api/group__CUDART__TYPES.html
- https://docs.nvidia.com/cuda/archive/12.0.0/cuda-runtime-api/group__CUDART__TYPES.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/memory-sync-domains.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html
- https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html
- https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEMOP.html
- https://docs.nvidia.com/cuda/archive/11.7.0/cuda-driver-api/group__CUDA__MEMOP.html
- https://docs.nvidia.com/cuda/archive/11.7.0/cuda-driver-api/group__CUDA__GRAPH.html
- https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html
- https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GRAPH.html
- https://docs.nvidia.com/cuda/cuda-driver-api/unionCUstreamBatchMemOpParams__v1.html
- https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__BATCH__MEM__OP__NODE__PARAMS__v2.html
- https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/structCUstreamBatchMemOpParams__v1_1_1CUstreamMemOpAtomicReductionParams__st.html
- https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__TYPES.html
- https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/group__CUDA__TYPES.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaConditionalNodeParams.html
- https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__CONDITIONAL__NODE__PARAMS.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
- https://docs.nvidia.com/cuda/archive/13.1.2/cuda-runtime-api/group__CUDART__TYPES.html
- https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html
- https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html
- https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html
- https://docs.nvidia.com/cuda/archive/12.5.0/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html
- https://docs.nvidia.com/cuda/archive/13.0.3/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html
- https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html
- https://docs.nvidia.com/cuda/archive/13.1.0/cuda-runtime-api/group__CUDART__EXECUTION__CONTEXT.html
- https://docs.nvidia.com/cuda/archive/13.3.0/cuda-runtime-api/structcudaDevWorkqueueConfigResource.html
- https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html
- https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html
- https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/index.html
- https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html
- https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html
- https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html
- https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf
- https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html
- https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.graph.GraphCompleteOptions.html
- https://nvidia.github.io/cuda-python/cuda-core/1.2.0/generated/cuda.core.utils.copy_batch.html
- https://nvidia.github.io/cuda-python/coverage/z_6928e1fc216e3a2e__copy_ops_pyx.html
- https://github.com/NVIDIA/cuda-python/pull/2593

**NVIDIA blogs and MPS documentation**
- https://developer.nvidia.com/blog/control-how-your-gpu-shares-work-with-green-contexts/
- https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes
- https://developer.nvidia.com/blog/enabling-dynamic-control-flow-in-cuda-graphs-with-device-graph-launch
- https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features
- https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/
- https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/
- https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication
- https://developer.nvidia.com/blog/memory-efficiency-faster-initialization-and-cost-estimation-with-nvidia-collective-communications-library-2-22/
- https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html
- https://docs.nvidia.com/deploy/mps/when-to-use-mps.html
- https://docs.nvidia.com/deploy/mps/latest/appendix-environment-variables.html
- https://docs.nvidia.com/deploy/mps/appendix-environment-variables.html
- https://docs.nvidia.com/deploy/mps/latest/appendix-tools-and-interface-reference.html
- https://docs.nvidia.com/deploy/mps/appendix-tools-and-interface-reference.html
- https://docs.nvidia.com/deploy/mps/latest/mpsv3-interface.html
- https://docs.nvidia.com/deploy/mps/architecture.html
- https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf
- https://docs.pytorch.org/docs/2.14/generated/torch.cuda.green_contexts.GreenContext.html
- https://docs.pytorch.org/docs/main/torch_nccl_environment_variables.html

**NVIDIA developer forums**
- https://forums.developer.nvidia.com/t/how-high-priority-stream-preemption/78183
- https://forums.developer.nvidia.com/t/questions-of-cuda-stream-priority/250343
- https://forums.developer.nvidia.com/t/how-to-verify-that-high-priority-stream-is-served/243831
- https://forums.developer.nvidia.com/t/43857
- https://forums.developer.nvidia.com/t/45701
- https://forums.developer.nvidia.com/t/371720
- https://forums.developer.nvidia.com/t/millisecond-scale-d2d-memcpy-admission-latency-in-secondary-cuda-context-while-primary-context-kernel-is-running/371720
- https://forums.developer.nvidia.com/t/the-configuration-of-gpu-time-slice-on-kubernetes/288938
- https://forums.developer.nvidia.com/t/is-there-any-document-on-set-timeslice-on-nvidia-smi/156566
- https://forums.developer.nvidia.com/t/memcpy-priority-of-cuda-streams/115189
- https://forums.developer.nvidia.com/t/cuda-graph-hangs-with-cross-green-context-edges-on-r610-driver-works-on-r580/385732
- https://forums.developer.nvidia.com/t/green-context-sm-provisioning-vs-the-primary-context/377440
- https://forums.developer.nvidia.com/t/deadlock-when-using-custreamwaitvalue32-custreamwritevalue32-for-async-cross-stream-ordering/353972
- https://forums.developer.nvidia.com/t/cuda-streams-custreamwaitvalue32-and-deadlocks-in-inter-kernel-communication/343726

**NCCL documentation, release notes and source**
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/flags.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_memory.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_reducecopy.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html
- https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-26-2.html
- https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-27-3.html
- https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-28-3.html
- https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-28-7.html
- https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-29-2.html
- https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-29-7.html
- https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-30-3.html
- https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-30-7.html
- https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-31-2.html
- https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2303/user-guide/docs/api/comms.html
- https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2323/release-notes/rel_2-28-3.html
- https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2323/release-notes/rel_2-30-7.html
- https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2323/release-notes/rel_2-31-2.html
- https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2323/release-notes/rel_2-32-3.html
- https://github.com/NVIDIA/nccl/blob/master/src/enqueue/enqueue.cc
- https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/enqueue.cc
- https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in
- https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/nccl.h.in
- https://github.com/NVIDIA/nccl/blob/master/src/ce_coll.cc
- https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/ce_coll.cc
- https://github.com/NVIDIA/nccl/blob/master/src/misc/cudawrap.cc
- https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/init.cc
- https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/device/primitives.h
- https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/include/strongstream.h
- https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/include/nccl_device.h
- https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/include/nccl_device/impl/lsa_barrier__funcs.h
- https://github.com/NVIDIA/nccl/releases
- https://github.com/NVIDIA/nccl/releases/tag/v2.31.2-1
- https://github.com/NVIDIA/nccl/discussions/1879

**NCCL and nccl-tests issues**
- https://github.com/NVIDIA/nccl/issues/574
- https://github.com/NVIDIA/nccl/issues/208
- https://github.com/NVIDIA/nccl/issues/592
- https://github.com/NVIDIA/nccl/issues/1736
- https://github.com/NVIDIA/nccl/issues/2090
- https://github.com/NVIDIA/nccl/issues/2138
- https://github.com/NVIDIA/nccl/issues/2238
- https://github.com/NVIDIA/nccl/issues/2272
- https://github.com/nvidia/nccl/issues/2288
- https://github.com/NVIDIA/nccl/issues/2421
- https://github.com/NVIDIA/nccl-tests/issues/333
- https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md

**NVSHMEM, driver source and hardware documents**
- https://github.com/NVIDIA/nvshmem/blob/devel/src/host/comm/sync.cpp
- https://github.com/NVIDIA/nvshmem/blob/devel/src/host/coll/cpu_coll.h
- https://github.com/NVIDIA/nvshmem/blob/main/src/host/comm/sync.cpp
- https://github.com/NVIDIA/nvshmem/blob/main/src/host/init/init.cu
- https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html
- https://github.com/NVIDIA/open-gpu-doc/blob/master/manuals/ampere/ga100/dev_pbdma.ref.txt
- https://raw.githubusercontent.com/NVIDIA/open-gpu-doc/master/manuals/ampere/ga100/dev_pbdma.ref.txt
- https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/host/clc76f.h
- https://github.com/NVIDIA/open-gpu-kernel-modules/blob/615.78.08/src/nvidia/src/kernel/gpu/ce/arch/blackwell/kernel_ce_gb100.c
- https://github.com/NVIDIA/open-gpu-kernel-modules/blob/615.78.08/src/common/sdk/nvidia/inc/ctrl/ctrla06c.h
- https://github.com/NVIDIA/nvbandwidth/blob/main/testcases.cpp
- https://openfabrics.org/downloads/ofv/ofv_presentation_GPU.pdf
- https://hihat.opencommons.org/images/3/34/CUDA_Graphs_Update_2022.pdf
- https://patents.google.com/patent/US9483423

**Other projects**
- https://github.com/rapidsai/rmm/issues/2518
- https://github.com/deepseek-ai/DeepEP
- https://dev-discuss.pytorch.org/t/pytorch-symmetricmemory-harnessing-nvlink-programmability-with-ease/2798
- https://triton-distributed.readthedocs.io/en/latest/_sources/getting-started/tutorials/07-overlapping-allgather-gemm.rst.txt
- https://github.com/kvcache-ai/Mooncake/issues/2568
- https://github.com/kvcache-ai/Mooncake/pull/4114
- https://github.com/kvcache-ai/Mooncake/issues/2132
- https://github.com/kvcache-ai/Mooncake/pull/4423
- https://github.com/sgl-project/sglang/issues/31117
- https://github.com/bytedance/byteps/tree/bytescheduler/bytescheduler

**Papers**
- https://arxiv.org/pdf/2110.00459
- https://arxiv.org/pdf/2510.00991
- https://arxiv.org/html/2510.00991v2
- https://arxiv.org/pdf/2602.21548
- https://arxiv.org/html/2602.21548
- https://arxiv.org/html/2604.26889
- https://arxiv.org/abs/2605.08524
- https://arxiv.org/pdf/2501.16909
- https://arxiv.org/abs/1905.03960
- https://arxiv.org/pdf/1905.03960
- https://ar5iv.labs.arxiv.org/html/2112.13509
- https://arxiv.org/pdf/1803.03288
- https://arxiv.org/html/2210.17223v2
- https://arxiv.org/pdf/2605.11335
- https://arxiv.org/pdf/2303.06324
- https://arxiv.org/html/2303.06324v2
- https://arxiv.org/html/2607.18002v1
- https://arxiv.org/html/2601.04071v2
- https://arxiv.org/abs/2604.07874
- https://arxiv.org/html/2604.07874
- https://arxiv.org/pdf/2410.01228
- https://arxiv.org/abs/2606.09200
- https://arxiv.org/html/2408.12757v2
- https://arxiv.org/html/2502.19811v1
- https://arxiv.org/html/2503.20313v3
- https://arxiv.org/html/2606.05951
- https://arxiv.org/html/2504.09014v4
- https://arxiv.org/pdf/2504.09014
- https://arxiv.org/html/2504.19519v2
- https://arxiv.org/html/2401.16677v1
- https://arxiv.org/html/2401.16677
- https://arxiv.org/pdf/2409.15241
- https://arxiv.org/html/2410.07192v1
- https://arxiv.org/pdf/2406.03243
- https://arxiv.org/abs/2412.14335
- https://arxiv.org/abs/2410.07381
- https://www.cs.unc.edu/~jbakita/rtas24.pdf
- https://sigops.org/s/conferences/sosp/2019/slides/peng.pdf
- https://i.cs.hku.hk/~cwu/papers/yhpeng-sosp19.pdf
- https://i.cs.hku.hk/~cwu/papers/yxbao-infocom20.pdf
- https://proceedings.mlsys.org/paper_files/paper/2019/hash/94cb28874a503f34b3c4a41bddcea2bd-Abstract.html
- https://www.usenix.org/conference/atc23/presentation/li-jiamin
- https://usenix.org/system/files/osdi22-han.pdf
- https://www.usenix.org/conference/osdi22/presentation/han
- https://www.usenix.org/conference/osdi25/presentation/shen-weihang
- https://www.usenix.org/conference/atc25/presentation/fan
- https://ppopp24.sigplan.org/details/PPoPP-2024-papers/2/Liger-Interleaving-Intra-and-Inter-Operator-Parallelism-for-Distributed-Large-Model
- https://odr.chalmers.se/items/4c25d40c-6fe9-4426-9fcf-c7857413db27/full
- https://eo.proxy.cloudns.pro/https://researchr.org/publication/ChengLDWWSLZD0025 (proxy URL, as given by the researcher)
- https://unpaywall.org/10.1145%2F3651890.3672239
