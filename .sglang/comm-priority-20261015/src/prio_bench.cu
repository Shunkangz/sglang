// prio_bench.cu -- priority / preemption benchmark for two concurrent GPU collectives.
//
//   Comm A (high priority): the model-forward collective (all-reduce / all-gather / reduce-scatter),
//                           issued on the main stream between GEMMs.
//   Comm B (best effort):   a background KV-cache all-gather issued on a low-priority stream.
//
// One process, one A-thread + one B-thread per GPU, NCCL comms per role.
// B data paths:
//   nccl           one ncclAllGather of the whole shard            (naive; not pausable)
//   nccl-gated     chunked ncclAllGather, each chunk behind a stream-memop gate
//   ce / ce-gated  copy-engine pulls (cudaMemcpyBatchAsync), chunked, optionally gated
//   nccl-ce[-gated] NCCL 2.30 zero-SM (CTAPolicy ZERO) copy-engine all-gather, chunked
//   sm / sm-preempt custom put kernel; the -preempt variant polls the pause word and exits,
//                   then a gate-wait + relaunch chain resumes it from a device-side cursor
//
// Pause protocol (all GPU-side, no host in the loop):
//   ctrl[g].pause[r] : one 32-bit slot per source rank r, in GPU g's memory.
//   A side, before its collective: cuStreamBatchMemOp WRITE pause[r]=1 into every GPU of the B group
//                                  (scope=global) or only its own GPU (scope=local).
//   A side, after its collective:  same writes with 0.
//   B side, before each chunk:     cuStreamBatchMemOp WAIT pause[s]==0 for every slot s (or own slot).
//   sm-preempt kernel:             polls the slots between work items; exits when any slot is set.
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>
#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(1);                                                                             \
    }                                                                                      \
  } while (0)
#define CU(x)                                                          \
  do {                                                                 \
    CUresult r_ = (x);                                                 \
    if (r_ != CUDA_SUCCESS) {                                          \
      const char* s_;                                                  \
      cuGetErrorString(r_, &s_);                                       \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, s_); \
      exit(1);                                                         \
    }                                                                  \
  } while (0)
#define NC(x)                                                                              \
  do {                                                                                     \
    ncclResult_t r_ = (x);                                                                 \
    if (r_ != ncclSuccess) {                                                               \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, ncclGetErrorString(r_)); \
      exit(1);                                                                             \
    }                                                                                      \
  } while (0)
#define CB(x)                                                                      \
  do {                                                                             \
    cublasStatus_t s_ = (x);                                                       \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                             \
      fprintf(stderr, "%s:%d %s -> cublas %d\n", __FILE__, __LINE__, #x, (int)s_); \
      exit(1);                                                                     \
    }                                                                              \
  } while (0)

static constexpr int kMaxRanks = 16;

enum BType { B_NONE, B_NCCL, B_CE, B_SM, B_NCCL_CE };

struct Opts {
  int ngpus = 8;
  int fwd = 1;
  BType b = B_NONE;
  bool gated = false;
  std::string bname = "none";
  int prio = 1;
  int global_scope = 1;
  int layers = 32;
  int iters = 4;
  int gm = 8192, gn = 7168, gk = 7168;
  size_t a_bytes = 64ull << 20;
  std::string a_ops = "ar";
  size_t b_bytes = 1ull << 30;   // per-rank shard
  size_t b_chunk = 4ull << 20;   // per-rank bytes per gated op (nccl / ce)
  size_t b_item = 256ull << 10;  // sm kernel work item
  int b_ctas = 16;               // sm kernel CTAs
  int b_nccl_ctas = 0;           // >0: maxCTAs for the B NCCL comm
  int ce_batch = 1;
  int ce_overlap_flag = 0;
  // Pull (peer shard -> local recv) hangs at 8 GPUs with the default CUDA_DEVICE_MAX_CONNECTIONS=8
  // while the stream is parked on a stream-wait (work-queue aliasing); 32 connections fix it.
  int ce_push = 1;
  int ce_streams = 4;  // CE B streams per GPU; peers are dealt round-robin (B200 has 4 async engines)
  int b_go = 1;        // 0: B stream does not park on the go flag (debug)
  int validate = 1;
  int queue_depth = 8;  // sm-preempt: outstanding (wait,kernel) pairs
  int gemm_sms = 0;     // >0: cublasSetSmCountTarget, leaves SMs free for an SM-based B
  int lead = 0;         // 1: split the GEMM in two and close the gate between the halves
  int green_b = 0;      // >0: run B's stream in a green context with this many SMs
  int graph = 0;        // 1: capture the whole forward (GEMMs, gate writes, A) into one CUDA graph
  int b_pace_us = 0;    // CE mover: idle this long on each B stream after every chunk (rate limit)
} opt;

// ---------------------------------------------------------------------------------------------
// Device side
// ---------------------------------------------------------------------------------------------
struct alignas(128) Ctrl {
  unsigned int pause[kMaxRanks];
  unsigned int go;
  unsigned int pad[31 - kMaxRanks];
  unsigned long long progress[kMaxRanks];  // per B stream: cumulative bytes this rank has sent (egress)
};

struct alignas(128) Work {
  unsigned long long next;
  unsigned long long done;
  unsigned long long t_start;
  unsigned long long t_end;
  unsigned int exits_paused;  // CTA exits caused by the pause word
  unsigned int launches;      // kernel launches that did at least one item
  unsigned long long t_first_pause_seen;
};

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ unsigned pat(int r, size_t i) {
  return (unsigned)(r + 1) * 0x9E3779B1u ^ (unsigned)i ^ ((unsigned)(i >> 32) * 0x85EBCA6Bu);
}

// Rate limiter for the CE mover: one thread idles on the B stream between chunks.
__global__ void pace_kernel(unsigned long long ns) {
  unsigned long long t0 = gtimer();
  while (gtimer() - t0 < ns) __nanosleep(2000);
}

__global__ void fill_pattern(unsigned* p, size_t n, int r) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = pat(r, i);
}

__global__ void fill_bf16(__nv_bfloat16* p, size_t n, unsigned seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    unsigned h = (unsigned)i * 2654435761u ^ seed;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    p[i] = __float2bfloat16(((h & 0xffff) / 65535.0f - 0.5f) * 0.1f);
  }
}

// recv layout "chunk-major with chunk C": [chunk c][src rank][C bytes]. C == S gives rank-major.
__global__ void check_recv(const unsigned* recv, size_t S, size_t C, int N, int self, int skip_self,
                           unsigned long long* bad) {
  size_t total = (size_t)N * S / 4;
  unsigned long long local_bad = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    size_t o = i * 4;
    size_t c = o / ((size_t)N * C);
    size_t w = o % ((size_t)N * C);
    int src = (int)(w / C);
    size_t off = c * C + w % C;
    if (skip_self && src == self) continue;
    if (recv[i] != pat(src, off / 4)) local_bad++;
  }
  if (local_bad) atomicAdd(bad, local_bad);
}

// Push this rank's shard into every peer's receive buffer (rank-major layout).
// Work item i -> (peer offset 1 + i % (N-1), item index i / (N-1)).
// preemptible: before claiming an item, CTA leader polls the pause slots; any non-zero -> exit.
// Items are never abandoned half-way: a claimed item is always completed, so the cursor
// (Work::next) is exact and the relaunched kernel resumes where the last one stopped.
__global__ void __launch_bounds__(512)
    kv_put_kernel(const int4* __restrict__ src, int4* const* dst_rank_base, int N, int rank, size_t shard_bytes,
                  size_t item_bytes, unsigned long long total_items, const unsigned int* pause, int slot_lo,
                  int slot_hi, int preemptible, Work* w) {
  __shared__ unsigned long long s_item;
  __shared__ int s_stop;
  bool did_work = false;
  if (threadIdx.x == 0 && blockIdx.x == 0) atomicCAS(&w->t_start, 0ull, gtimer());
  for (;;) {
    if (threadIdx.x == 0) {
      int paused = 0;
      if (preemptible) {
        for (int s = slot_lo; s < slot_hi; ++s) paused |= ((volatile const unsigned int*)pause)[s];
      }
      unsigned long long it = ~0ull;
      if (!paused) it = atomicAdd(&w->next, 1ull);
      s_stop = paused ? 1 : (it >= total_items ? 2 : 0);
      s_item = it;
      if (paused) {
        atomicAdd(&w->exits_paused, 1u);
        atomicCAS(&w->t_first_pause_seen, 0ull, gtimer());
      }
    }
    __syncthreads();
    if (s_stop) break;
    did_work = true;
    unsigned long long it = s_item;
    int p = 1 + (int)(it % (unsigned long long)(N - 1));
    unsigned long long c = it / (unsigned long long)(N - 1);
    int peer = (rank + p) % N;
    size_t off = (size_t)c * item_bytes;
    size_t bytes = min(item_bytes, shard_bytes - off);
    const int4* s = src + off / 16;
    int4* d = dst_rank_base[peer] + off / 16;
    size_t n16 = bytes / 16;
    constexpr int U = 4;
    size_t i = threadIdx.x;
    for (; i + (U - 1) * blockDim.x < n16; i += U * blockDim.x) {
      int4 v[U];
#pragma unroll
      for (int u = 0; u < U; ++u) v[u] = __ldg(s + i + u * blockDim.x);
#pragma unroll
      for (int u = 0; u < U; ++u) d[i + u * blockDim.x] = v[u];
    }
    for (; i < n16; i += blockDim.x) d[i] = __ldg(s + i);
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned long long dn = atomicAdd(&w->done, 1ull) + 1;
      if (dn == total_items) {
        __threadfence_system();
        w->t_end = gtimer();
      }
    }
  }
  if (threadIdx.x == 0) {
    __threadfence_system();
    if (did_work && blockIdx.x == 0) atomicAdd(&w->launches, 1u);
  }
}

// ---------------------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------------------
struct Gpu {
  int rank = 0;
  cudaStream_t sA = nullptr, sB = nullptr, sCtl = nullptr;
  std::vector<cudaStream_t> sBx;  // CE B streams; sBx[0] == sB
  CUcontext ctxB = nullptr;       // green context of B (nullptr: primary context)
  int green_sms = 0;
  std::vector<unsigned long long> idBx;
  cudaEvent_t evFork;
  std::vector<cudaEvent_t> evJoin;
  unsigned long long idA = 0, idB = 0;
  ncclComm_t commA = nullptr, commB = nullptr;
  cublasHandle_t blas = nullptr;
  __nv_bfloat16 *gA = nullptr, *gB = nullptr, *gC = nullptr;
  void *aBuf = nullptr, *aOut = nullptr;
  void *bSend = nullptr, *bRecv = nullptr;
  ncclWindow_t bWin = nullptr;
  Ctrl* ctrl = nullptr;
  Work* work = nullptr;
  Work* work_host = nullptr;  // pinned + mapped mirror for the B thread's top-up loop
  int4** dst_base = nullptr;  // device array [N] of peer recv + rank*S
  unsigned long long* bad = nullptr;
  std::vector<cudaEvent_t> evG0, evA0, evA1;
  cudaEvent_t evFwd0, evFwd1, evB0, evB1;
  std::atomic<int> b_ready{0};
  // results per iteration
  std::vector<std::vector<float>> a_ms, g_ms;
  std::vector<float> fwd_ms, b_ms;
  std::vector<unsigned> exits_paused, launches;
  std::vector<double> b_fwd_bytes;     // B egress bytes completed when the forward finished
  unsigned long long* snap = nullptr;  // device: [0..15] progress copy, [16] sm done items
};

static Gpu gpus[kMaxRanks];
static ncclUniqueId idA, idB;

static size_t parse_size(const char* s) {
  char* end;
  double v = strtod(s, &end);
  switch (*end) {
    case 'k':
    case 'K':
      v *= 1024;
      break;
    case 'm':
    case 'M':
      v *= 1024 * 1024;
      break;
    case 'g':
    case 'G':
      v *= 1024.0 * 1024 * 1024;
      break;
    default:
      break;
  }
  return (size_t)v;
}

static void parse(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto nxt = [&]() -> const char* {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", k.c_str());
        exit(1);
      }
      return argv[++i];
    };
    if (k == "--ngpus")
      opt.ngpus = atoi(nxt());
    else if (k == "--fwd")
      opt.fwd = atoi(nxt());
    else if (k == "--b") {
      opt.bname = nxt();
      std::string b = opt.bname;
      opt.gated = false;
      if (b.size() > 6 && b.substr(b.size() - 6) == "-gated") {
        opt.gated = true;
        b = b.substr(0, b.size() - 6);
      }
      if (b == "sm-preempt") {
        opt.gated = true;
        b = "sm";
      }
      if (b == "none")
        opt.b = B_NONE;
      else if (b == "nccl")
        opt.b = B_NCCL;
      else if (b == "ce")
        opt.b = B_CE;
      else if (b == "sm")
        opt.b = B_SM;
      else if (b == "nccl-ce")
        opt.b = B_NCCL_CE;
      else {
        fprintf(stderr, "bad --b %s\n", opt.bname.c_str());
        exit(1);
      }
    } else if (k == "--prio")
      opt.prio = atoi(nxt());
    else if (k == "--scope")
      opt.global_scope = std::string(nxt()) == "global";
    else if (k == "--layers")
      opt.layers = atoi(nxt());
    else if (k == "--iters")
      opt.iters = atoi(nxt());
    else if (k == "--gemm") {
      opt.gm = atoi(nxt());
      opt.gn = atoi(nxt());
      opt.gk = atoi(nxt());
    } else if (k == "--a-bytes")
      opt.a_bytes = parse_size(nxt());
    else if (k == "--a-ops")
      opt.a_ops = nxt();
    else if (k == "--b-bytes")
      opt.b_bytes = parse_size(nxt());
    else if (k == "--b-chunk")
      opt.b_chunk = parse_size(nxt());
    else if (k == "--b-item")
      opt.b_item = parse_size(nxt());
    else if (k == "--b-ctas")
      opt.b_ctas = atoi(nxt());
    else if (k == "--b-nccl-ctas")
      opt.b_nccl_ctas = atoi(nxt());
    else if (k == "--ce-batch")
      opt.ce_batch = atoi(nxt());
    else if (k == "--ce-overlap-flag")
      opt.ce_overlap_flag = atoi(nxt());
    else if (k == "--ce-push")
      opt.ce_push = atoi(nxt());
    else if (k == "--ce-streams")
      opt.ce_streams = atoi(nxt());
    else if (k == "--b-go")
      opt.b_go = atoi(nxt());
    else if (k == "--validate")
      opt.validate = atoi(nxt());
    else if (k == "--queue-depth")
      opt.queue_depth = atoi(nxt());
    else if (k == "--gemm-sms")
      opt.gemm_sms = atoi(nxt());
    else if (k == "--lead")
      opt.lead = atoi(nxt());
    else if (k == "--green-b")
      opt.green_b = atoi(nxt());
    else if (k == "--graph")
      opt.graph = atoi(nxt());
    else if (k == "--b-pace-us")
      opt.b_pace_us = atoi(nxt());
    else {
      fprintf(stderr, "unknown arg %s\n", k.c_str());
      exit(1);
    }
  }
  if (opt.green_b > 0 && opt.b != B_SM && opt.b != B_NONE) {
    fprintf(stderr, "--green-b supports only --b sm/sm-preempt/none\n");
    exit(1);
  }
  if (opt.b_bytes % opt.b_chunk) {
    fprintf(stderr, "b-bytes must be a multiple of b-chunk\n");
    exit(1);
  }
  if (opt.b_item % 16 || opt.b_chunk % 16) {
    fprintf(stderr, "sizes must be 16B multiples\n");
    exit(1);
  }
}

// A side: set/clear this rank's pause slot (zero SMs: executed by the GPU front end).
static void pause_write(Gpu& G, unsigned v) {
  CUstreamBatchMemOpParams ops[kMaxRanks];
  int n = 0;
  for (int g = 0; g < opt.ngpus; ++g) {
    if (!opt.global_scope && g != G.rank) continue;
    memset(&ops[n], 0, sizeof(ops[n]));
    ops[n].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    ops[n].writeValue.address = (CUdeviceptr)&gpus[g].ctrl->pause[G.rank];
    ops[n].writeValue.value = v;
    ops[n].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
    ++n;
  }
  CU(cuStreamBatchMemOp((CUstream)G.sA, n, ops, 0));
}

// B side: block the B stream until no rank (scope=global) / this rank (scope=local) is in A.
static void gate_wait(Gpu& G, cudaStream_t s) {
  CUstreamBatchMemOpParams ops[kMaxRanks];
  int n = 0;
  for (int src = 0; src < opt.ngpus; ++src) {
    if (!opt.global_scope && src != G.rank) continue;
    memset(&ops[n], 0, sizeof(ops[n]));
    ops[n].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    ops[n].waitValue.address = (CUdeviceptr)&G.ctrl->pause[src];
    ops[n].waitValue.value = 0;
    ops[n].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    ++n;
  }
  CU(cuStreamBatchMemOp((CUstream)s, n, ops, 0));
}

static void write_progress(Gpu& G, cudaStream_t s, int slot, unsigned long long bytes) {
  CU(cuStreamWriteValue64((CUstream)s, (CUdeviceptr)&G.ctrl->progress[slot], bytes, 0));
}

static void wait_go(Gpu& G, cudaStream_t s, unsigned v) {
  CU(cuStreamWaitValue32((CUstream)s, (CUdeviceptr)&G.ctrl->go, v, CU_STREAM_WAIT_VALUE_GEQ));
}

static std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> out;
  size_t b = 0;
  for (;;) {
    size_t e = s.find(c, b);
    out.push_back(s.substr(b, e - b));
    if (e == std::string::npos) break;
    b = e + 1;
  }
  return out;
}

static void issue_A(Gpu& G, const std::string& op) {
  int N = opt.ngpus;
  size_t elems = opt.a_bytes / 2;
  if (op == "ar") {
    NC(ncclAllReduce(G.aBuf, G.aBuf, elems, ncclBfloat16, ncclSum, G.commA, G.sA));
  } else if (op == "ag") {
    NC(ncclAllGather(G.aBuf, G.aOut, elems / N, ncclBfloat16, G.commA, G.sA));
  } else if (op == "rs") {
    NC(ncclReduceScatter(G.aBuf, G.aOut, elems / N, ncclBfloat16, ncclSum, G.commA, G.sA));
  } else {
    fprintf(stderr, "bad A op %s\n", op.c_str());
    exit(1);
  }
}

static void enqueue_forward(Gpu& G, const std::vector<std::string>& ops, bool timed) {
  int L = opt.layers;
  const float alpha = 1.f, beta = 0.f;
  for (int l = 0; l < L; ++l) {
    char rng[32];
    snprintf(rng, sizeof rng, "L%d GEMM", l);
    nvtxRangePushA(rng);
    if (timed) CK(cudaEventRecord(G.evG0[l], G.sA));
    if (opt.lead) {
      // lookahead close: the gate closes one op (half a GEMM) before A, so B drains under compute
      int h = opt.gn / 2;
      CB(cublasGemmEx(G.blas, CUBLAS_OP_N, CUBLAS_OP_N, opt.gm, h, opt.gk, &alpha, G.gA, CUDA_R_16BF, opt.gm, G.gB,
                      CUDA_R_16BF, opt.gk, &beta, G.gC, CUDA_R_16BF, opt.gm, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      if (opt.gated) pause_write(G, 1);
      CB(cublasGemmEx(G.blas, CUBLAS_OP_N, CUBLAS_OP_N, opt.gm, opt.gn - h, opt.gk, &alpha, G.gA, CUDA_R_16BF, opt.gm,
                      G.gB + (size_t)opt.gk * h, CUDA_R_16BF, opt.gk, &beta, G.gC + (size_t)opt.gm * h, CUDA_R_16BF,
                      opt.gm, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    } else {
      CB(cublasGemmEx(G.blas, CUBLAS_OP_N, CUBLAS_OP_N, opt.gm, opt.gn, opt.gk, &alpha, G.gA, CUDA_R_16BF, opt.gm, G.gB,
                      CUDA_R_16BF, opt.gk, &beta, G.gC, CUDA_R_16BF, opt.gm, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    nvtxRangePop();
    snprintf(rng, sizeof rng, "L%d A:%s", l, ops[l % ops.size()].c_str());
    nvtxRangePushA(rng);
    if (opt.gated && !opt.lead) pause_write(G, 1);
    if (timed) CK(cudaEventRecord(G.evA0[l], G.sA));
    issue_A(G, ops[l % ops.size()]);
    if (timed) CK(cudaEventRecord(G.evA1[l], G.sA));
    if (opt.gated) pause_write(G, 0);
    nvtxRangePop();
  }
}

static void a_thread(int r, std::barrier<>* bar_all, std::barrier<>* bar_a) {
  Gpu& G = gpus[r];
  CK(cudaSetDevice(r));
  char nm[32];
  snprintf(nm, sizeof nm, "A-gpu%d", r);
  pthread_setname_np(pthread_self(), nm);
  std::vector<std::string> ops = split(opt.a_ops, ',');
  int L = opt.layers;
  cudaGraphExec_t exec = nullptr;
  if (opt.graph && opt.fwd) {
    // gate writes are captured as batch-mem-op nodes; kernel nodes keep the capture stream's priority
    cudaGraph_t graph;
    CK(cudaStreamBeginCapture(G.sA, cudaStreamCaptureModeThreadLocal));
    enqueue_forward(G, ops, false);
    CK(cudaStreamEndCapture(G.sA, &graph));
    CK(cudaGraphInstantiateWithFlags(&exec, graph, cudaGraphInstantiateFlagUseNodePriority));
    size_t nn = 0;
    CK(cudaGraphGetNodes(graph, nullptr, &nn));
    if (r == 0) printf("captured forward graph: %zu nodes\n", nn);
    CK(cudaGraphUpload(exec, G.sA));
    CK(cudaStreamSynchronize(G.sA));
  }
  for (int it = 0; it < opt.iters; ++it) {
    // nsys --capture-range=cudaProfilerApi records only the last iteration; CUPTI traces only work
    // launched after the start, so it must precede every thread's enqueue (barrier [1])
    if (r == 0 && it == opt.iters - 1) CK(cudaProfilerStart());
    bar_all->arrive_and_wait();  // [1] previous iteration fully drained, buffers reset
    wait_go(G, G.sA, it + 1);
    CK(cudaEventRecord(G.evFwd0, G.sA));
    if (opt.fwd) {
      if (exec)
        CK(cudaGraphLaunch(exec, G.sA));
      else
        enqueue_forward(G, ops, true);
    }
    CK(cudaEventRecord(G.evFwd1, G.sA));
    if (opt.b != B_NONE) {
      CK(cudaMemcpyAsync(G.snap, G.ctrl->progress, sizeof(G.ctrl->progress), cudaMemcpyDeviceToDevice, G.sA));
      CK(cudaMemcpyAsync(G.snap + kMaxRanks, &G.work->done, 8, cudaMemcpyDeviceToDevice, G.sA));
    }
    // wait for the local B thread to have enqueued its head, then start every GPU together
    while (G.b_ready.load() < it + 1) std::this_thread::yield();
    bar_a->arrive_and_wait();
    if (r == 0) {
      // one front end releases every GPU (P2P writes), so host thread wake-up skew never reaches the GPUs
      CUstreamBatchMemOpParams ops[kMaxRanks];
      for (int g = 0; g < opt.ngpus; ++g) {
        memset(&ops[g], 0, sizeof(ops[g]));
        ops[g].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
        ops[g].writeValue.address = (CUdeviceptr)&gpus[g].ctrl->go;
        ops[g].writeValue.value = it + 1;
        ops[g].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
      }
      CU(cuStreamBatchMemOp((CUstream)G.sCtl, opt.ngpus, ops, 0));
    }
    CK(cudaStreamSynchronize(G.sA));
    bar_all->arrive_and_wait();  // [2] both streams done on every GPU
    if (r == 0 && it == opt.iters - 1) CK(cudaProfilerStop());
    // collect
    std::vector<float> a(L), g(L);
    if (opt.fwd && !exec) {
      for (int l = 0; l < L; ++l) {
        CK(cudaEventElapsedTime(&a[l], G.evA0[l], G.evA1[l]));
        CK(cudaEventElapsedTime(&g[l], G.evG0[l], G.evA0[l]));
      }
    }
    float f = 0;
    CK(cudaEventElapsedTime(&f, G.evFwd0, G.evFwd1));
    G.a_ms.push_back(a);
    G.g_ms.push_back(g);
    G.fwd_ms.push_back(f);
    double fb = 0;
    if (opt.b != B_NONE) {
      unsigned long long h[kMaxRanks + 1];
      CK(cudaMemcpy(h, G.snap, sizeof h, cudaMemcpyDeviceToHost));
      if (opt.b == B_SM)
        fb = (double)h[kMaxRanks] * opt.b_item;
      else
        for (int k = 0; k < kMaxRanks; ++k) fb += (double)h[k];
    }
    G.b_fwd_bytes.push_back(fb);
    bar_all->arrive_and_wait();  // [3] results collected
  }
}

static void reset_b(Gpu& G) {
  if (opt.b == B_NONE) return;
  CK(cudaMemsetAsync(G.bRecv, 0, (size_t)opt.ngpus * opt.b_bytes, G.sB));
  CK(cudaMemsetAsync(G.work, 0, sizeof(Work), G.sB));
  CK(cudaMemsetAsync(G.ctrl->progress, 0, sizeof(G.ctrl->progress), G.sB));
  CK(cudaStreamSynchronize(G.sB));
}

static void b_thread(int r, std::barrier<>* bar_all) {
  Gpu& G = gpus[r];
  CK(cudaSetDevice(r));
  char nm[32];
  snprintf(nm, sizeof nm, "B-gpu%d", r);
  pthread_setname_np(pthread_self(), nm);
  if (G.ctxB) CU(cuCtxSetCurrent(G.ctxB));
  int N = opt.ngpus;
  size_t S = opt.b_bytes;
  for (int it = 0; it < opt.iters; ++it) {
    reset_b(G);
    bar_all->arrive_and_wait();  // [1]
    if (opt.b_go) wait_go(G, G.sB, it + 1);
    CK(cudaEventRecord(G.evB0, G.sB));
    int head = 0;
    auto mark = [&](int n) {
      head += n;
      if (head >= 64 && G.b_ready.load() < it + 1) G.b_ready.store(it + 1);
    };
    if (opt.b == B_NCCL || opt.b == B_NCCL_CE) {
      size_t C = opt.gated ? opt.b_chunk : (opt.b == B_NCCL_CE ? opt.b_chunk : S);
      for (size_t c = 0; c < S / C; ++c) {
        if (opt.gated) gate_wait(G, G.sB);
        NC(ncclAllGather((char*)G.bSend + c * C, (char*)G.bRecv + c * N * C, C, ncclUint8, G.commB, G.sB));
        write_progress(G, G.sB, 0, (c + 1) * C * (N - 1));
        mark(2);
      }
    } else if (opt.b == B_CE) {
      size_t C = opt.b_chunk;
      std::vector<void*> dsts;
      std::vector<const void*> srcs;
      std::vector<size_t> sizes;
      cudaMemcpyAttributes attr{};
      attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
      attr.flags = opt.ce_overlap_flag ? cudaMemcpyFlagPreferOverlapWithCompute : 0;
      size_t attrIdx = 0;
      int K = (int)G.sBx.size();
      std::vector<unsigned long long> sent(K, 0);
      CK(cudaEventRecord(G.evFork, G.sB));
      for (int k = 1; k < K; ++k) CK(cudaStreamWaitEvent(G.sBx[k], G.evFork, 0));
      for (size_t c = 0; c < S / C; ++c) {
        for (int k = 0; k < K; ++k) {
          cudaStream_t sk = G.sBx[k];
          if (opt.gated) gate_wait(G, sk);
          dsts.clear();
          srcs.clear();
          sizes.clear();
          for (int p = 1 + k; p < N; p += K) {
            int peer = (r + p) % N;
            if (opt.ce_push) {
              dsts.push_back((char*)gpus[peer].bRecv + (size_t)r * S + c * C);
              srcs.push_back((const char*)G.bSend + c * C);
            } else {
              dsts.push_back((char*)G.bRecv + (size_t)peer * S + c * C);
              srcs.push_back((const char*)gpus[peer].bSend + c * C);
            }
            sizes.push_back(C);
          }
          if (dsts.empty()) continue;
          if (opt.ce_batch) {
            CK(cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), dsts.size(), &attr, &attrIdx, 1, sk));
          } else {
            for (size_t q = 0; q < dsts.size(); ++q)
              CK(cudaMemcpyAsync(dsts[q], srcs[q], sizes[q], cudaMemcpyDeviceToDevice, sk));
          }
          sent[k] += C * dsts.size();
          write_progress(G, sk, k, sent[k]);
          if (opt.b_pace_us) pace_kernel<<<1, 1, 0, sk>>>(opt.b_pace_us * 1000ull);
          mark(2);
        }
      }
      for (int k = 1; k < K; ++k) {
        CK(cudaEventRecord(G.evJoin[k], G.sBx[k]));
        CK(cudaStreamWaitEvent(G.sB, G.evJoin[k], 0));
      }
    } else if (opt.b == B_SM) {
      unsigned long long total_items = (unsigned long long)((S + opt.b_item - 1) / opt.b_item) * (N - 1);
      int slot_lo = opt.global_scope ? 0 : r, slot_hi = opt.global_scope ? N : r + 1;
      if (!opt.gated) {
        kv_put_kernel<<<opt.b_ctas, 512, 0, G.sB>>>((const int4*)G.bSend, G.dst_base, N, r, S, opt.b_item, total_items,
                                                    G.ctrl->pause, slot_lo, slot_hi, 0, G.work);
        CK(cudaGetLastError());
        G.b_ready.store(it + 1);
      } else {
        // Chain of (gate wait, kernel) pairs, topped up by this thread with a bounded queue.
        // The host is not in the pause/resume path: a pause makes the running kernel exit, the
        // next gate wait parks the stream until A clears the slots, and the next kernel resumes
        // from the device-side cursor.
        std::vector<cudaEvent_t> evs(opt.queue_depth);
        for (auto& e : evs) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        for (long k = 0;; ++k) {
          cudaEvent_t& e = evs[k % opt.queue_depth];
          if (k >= opt.queue_depth) {
            while (cudaEventQuery(e) == cudaErrorNotReady) std::this_thread::sleep_for(std::chrono::microseconds(20));
            if (__atomic_load_n(&G.work_host->done, __ATOMIC_ACQUIRE) >= total_items) break;
          }
          gate_wait(G, G.sB);
          kv_put_kernel<<<opt.b_ctas, 512, 0, G.sB>>>((const int4*)G.bSend, G.dst_base, N, r, S, opt.b_item,
                                                      total_items, G.ctrl->pause, slot_lo, slot_hi, 1, G.work);
          CK(cudaGetLastError());
          // mirror the cursor to host memory so the top-up loop can stop
          CK(cudaMemcpyAsync(G.work_host, G.work, sizeof(Work), cudaMemcpyDeviceToHost, G.sB));
          CK(cudaEventRecord(e, G.sB));
          if (k + 1 == opt.queue_depth || G.b_ready.load() < it + 1) G.b_ready.store(it + 1);
        }
        for (auto& e : evs) CK(cudaEventDestroy(e));
      }
    }
    CK(cudaEventRecord(G.evB1, G.sB));
    if (G.b_ready.load() < it + 1) G.b_ready.store(it + 1);
    CK(cudaStreamSynchronize(G.sB));
    bar_all->arrive_and_wait();  // [2]
    float b = 0;
    if (opt.b != B_NONE) {
      if (opt.b == B_SM) {
        Work w;
        CK(cudaMemcpy(&w, G.work, sizeof w, cudaMemcpyDeviceToHost));
        b = (w.t_end - w.t_start) / 1e6f;
        G.exits_paused.push_back(w.exits_paused);
        G.launches.push_back(w.launches);
      } else {
        CK(cudaEventElapsedTime(&b, G.evB0, G.evB1));
      }
    }
    G.b_ms.push_back(b);
    bar_all->arrive_and_wait();  // [3]
  }
}

static void setup(int r) {
  Gpu& G = gpus[r];
  G.rank = r;
  int N = opt.ngpus;
  CK(cudaSetDevice(r));
  for (int p = 0; p < N; ++p)
    if (p != r) {
      cudaError_t e = cudaDeviceEnablePeerAccess(p, 0);
      if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CK(e);
      cudaGetLastError();
    }
  int lo, hi;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  CK(cudaStreamCreateWithPriority(&G.sA, cudaStreamNonBlocking, opt.prio ? hi : lo));
  if (opt.green_b > 0) {
    CUdevice dev;
    CU(cuDeviceGet(&dev, r));
    CUdevResource sm, part, rest;
    CU(cuDeviceGetDevResource(dev, &sm, CU_DEV_RESOURCE_TYPE_SM));
    unsigned n = 1;
    CU(cuDevSmResourceSplitByCount(&part, &n, &sm, &rest, 0, opt.green_b));
    CUdevResourceDesc desc;
    CU(cuDevResourceGenerateDesc(&desc, &part, 1));
    CUgreenCtx g;
    CU(cuGreenCtxCreate(&g, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
    CU(cuCtxFromGreenCtx(&G.ctxB, g));
    CUstream cs;
    CU(cuGreenCtxStreamCreate(&cs, g, CU_STREAM_NON_BLOCKING, lo));
    G.sB = (cudaStream_t)cs;
    G.green_sms = part.sm.smCount;
    if (r == 0) printf("green context for B: %u SMs (rest %u)\n", part.sm.smCount, rest.sm.smCount);
  } else {
    CK(cudaStreamCreateWithPriority(&G.sB, cudaStreamNonBlocking, lo));
  }
  CK(cudaStreamCreateWithPriority(&G.sCtl, cudaStreamNonBlocking, hi));
  CK(cudaStreamGetId(G.sA, &G.idA));
  CK(cudaStreamGetId(G.sB, &G.idB));
  char nm[32];
  G.sBx.assign(1, G.sB);
  G.idBx.assign(1, G.idB);
  if (opt.b == B_CE) {
    for (int k = 1; k < opt.ce_streams; ++k) {
      cudaStream_t s;
      unsigned long long id;
      CK(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, lo));
      CK(cudaStreamGetId(s, &id));
      snprintf(nm, sizeof nm, "B-kv%d gpu%d", k, r);
      nvtxNameCudaStreamA(s, nm);
      G.sBx.push_back(s);
      G.idBx.push_back(id);
    }
  }
  if (G.ctxB) CU(cuCtxSetCurrent(G.ctxB));
  CK(cudaEventCreateWithFlags(&G.evFork, cudaEventDisableTiming));
  G.evJoin.resize(G.sBx.size());
  for (auto& e : G.evJoin) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  if (G.ctxB) CK(cudaSetDevice(r));
  snprintf(nm, sizeof nm, "A-main gpu%d", r);
  nvtxNameCudaStreamA(G.sA, nm);
  snprintf(nm, sizeof nm, "B-kv gpu%d", r);
  nvtxNameCudaStreamA(G.sB, nm);

  CK(cudaMalloc(&G.ctrl, sizeof(Ctrl)));
  CK(cudaMemset(G.ctrl, 0, sizeof(Ctrl)));
  CK(cudaMalloc(&G.work, sizeof(Work)));
  CK(cudaMemset(G.work, 0, sizeof(Work)));
  CK(cudaHostAlloc(&G.work_host, sizeof(Work), cudaHostAllocPortable));
  memset(G.work_host, 0, sizeof(Work));
  CK(cudaMalloc(&G.bad, 8));
  CK(cudaMalloc(&G.snap, 8 * (kMaxRanks + 1)));

  if (opt.fwd) {
    size_t na = (size_t)opt.gm * opt.gk, nb = (size_t)opt.gk * opt.gn, nc = (size_t)opt.gm * opt.gn;
    CK(cudaMalloc(&G.gA, na * 2));
    CK(cudaMalloc(&G.gB, nb * 2));
    CK(cudaMalloc(&G.gC, nc * 2));
    fill_bf16<<<1024, 256>>>(G.gA, na, 1 + r);
    fill_bf16<<<1024, 256>>>(G.gB, nb, 77 + r);
    CK(cudaMalloc(&G.aBuf, opt.a_bytes));
    CK(cudaMalloc(&G.aOut, opt.a_bytes));
    fill_bf16<<<1024, 256>>>((__nv_bfloat16*)G.aBuf, opt.a_bytes / 2, 5 + r);
    CB(cublasCreate(&G.blas));
    CB(cublasSetStream(G.blas, G.sA));
    if (opt.gemm_sms > 0) CB(cublasSetSmCountTarget(G.blas, opt.gemm_sms));
    if (opt.graph) {
      // capture must not allocate: give cuBLAS a fixed workspace
      void* ws;
      CK(cudaMalloc(&ws, 64ull << 20));
      CB(cublasSetWorkspace(G.blas, ws, 64ull << 20));
    }
  }

  ncclConfig_t cfgA = NCCL_CONFIG_INITIALIZER;
  cfgA.commName = "commA";
  NC(ncclCommInitRankConfig(&G.commA, N, idA, r, &cfgA));
  if (opt.b == B_NCCL || opt.b == B_NCCL_CE) {
    ncclConfig_t cfgB = NCCL_CONFIG_INITIALIZER;
    cfgB.commName = "commB";
    if (opt.b_nccl_ctas > 0) cfgB.maxCTAs = opt.b_nccl_ctas;
    if (opt.b == B_NCCL_CE) cfgB.CTAPolicy = NCCL_CTA_POLICY_ZERO;
    NC(ncclCommInitRankConfig(&G.commB, N, idB, r, &cfgB));
  }

  if (opt.b != B_NONE) {
    size_t recv_bytes = (size_t)N * opt.b_bytes;
    CK(cudaMalloc(&G.bSend, opt.b_bytes));
    fill_pattern<<<2048, 256>>>((unsigned*)G.bSend, opt.b_bytes / 4, r);
    if (opt.b == B_NCCL_CE) {
      NC(ncclMemAlloc(&G.bRecv, recv_bytes));
      NC(ncclCommWindowRegister(G.commB, G.bRecv, recv_bytes, &G.bWin, NCCL_WIN_COLL_SYMMETRIC));
      if (!G.bWin) {
        fprintf(stderr, "rank %d: symmetric window is NULL (CE path unavailable)\n", r);
        exit(1);
      }
    } else {
      CK(cudaMalloc(&G.bRecv, recv_bytes));
    }
  }
  int L = opt.layers;
  G.evG0.resize(L);
  G.evA0.resize(L);
  G.evA1.resize(L);
  for (int l = 0; l < L; ++l) {
    CK(cudaEventCreate(&G.evG0[l]));
    CK(cudaEventCreate(&G.evA0[l]));
    CK(cudaEventCreate(&G.evA1[l]));
  }
  CK(cudaEventCreate(&G.evFwd0));
  CK(cudaEventCreate(&G.evFwd1));
  if (G.ctxB) CU(cuCtxSetCurrent(G.ctxB));
  CK(cudaEventCreate(&G.evB0));
  CK(cudaEventCreate(&G.evB1));
  if (G.ctxB) CK(cudaSetDevice(r));
  CK(cudaDeviceSynchronize());
}

static void setup_peers(int r) {
  Gpu& G = gpus[r];
  CK(cudaSetDevice(r));
  if (opt.b == B_NONE) return;
  std::vector<int4*> base(opt.ngpus);
  for (int p = 0; p < opt.ngpus; ++p) base[p] = (int4*)((char*)gpus[p].bRecv + (size_t)r * opt.b_bytes);
  CK(cudaMalloc(&G.dst_base, sizeof(int4*) * opt.ngpus));
  CK(cudaMemcpy(G.dst_base, base.data(), sizeof(int4*) * opt.ngpus, cudaMemcpyHostToDevice));
}

// Every stream of the timed loop parks on the go flag while the host enqueues. NCCL's lazy
// connection setup and cuBLAS's first-call setup synchronize with the device, so they must have
// run once before the loop, or the enqueue deadlocks against the parked streams.
static void warmup(int r) {
  Gpu& G = gpus[r];
  CK(cudaSetDevice(r));
  const float alpha = 1.f, beta = 0.f;
  if (opt.fwd) {
    CB(cublasGemmEx(G.blas, CUBLAS_OP_N, CUBLAS_OP_N, opt.gm, opt.gn, opt.gk, &alpha, G.gA, CUDA_R_16BF, opt.gm, G.gB,
                    CUDA_R_16BF, opt.gk, &beta, G.gC, CUDA_R_16BF, opt.gm, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    if (opt.lead) {
      // the half-width GEMMs pick other cuBLAS kernels; load them before the streams park
      int h = opt.gn / 2;
      CB(cublasGemmEx(G.blas, CUBLAS_OP_N, CUBLAS_OP_N, opt.gm, h, opt.gk, &alpha, G.gA, CUDA_R_16BF, opt.gm, G.gB,
                      CUDA_R_16BF, opt.gk, &beta, G.gC, CUDA_R_16BF, opt.gm, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      CB(cublasGemmEx(G.blas, CUBLAS_OP_N, CUBLAS_OP_N, opt.gm, opt.gn - h, opt.gk, &alpha, G.gA, CUDA_R_16BF, opt.gm,
                      G.gB + (size_t)opt.gk * h, CUDA_R_16BF, opt.gk, &beta, G.gC + (size_t)opt.gm * h, CUDA_R_16BF,
                      opt.gm, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    for (const auto& op : split(opt.a_ops, ',')) issue_A(G, op);
  }
  if (opt.b == B_NCCL || opt.b == B_NCCL_CE) {
    size_t C = opt.b_chunk;
    NC(ncclAllGather(G.bSend, G.bRecv, C, ncclUint8, G.commB, G.sB));
    if (opt.b == B_NCCL && !opt.gated) NC(ncclAllGather(G.bSend, G.bRecv, opt.b_bytes, ncclUint8, G.commB, G.sB));
  }
  if (G.ctxB) CU(cuCtxSetCurrent(G.ctxB));
  if (opt.b == B_SM) {
    // lazy module loading synchronizes on a kernel's first launch
    kv_put_kernel<<<1, 512, 0, G.sB>>>((const int4*)G.bSend, G.dst_base, opt.ngpus, r, opt.b_bytes, opt.b_item, 0,
                                       G.ctrl->pause, 0, 0, 1, G.work);
    CK(cudaGetLastError());
  }
  if (opt.b == B_CE && opt.b_pace_us) pace_kernel<<<1, 1, 0, G.sB>>>(0);
  if (opt.b == B_CE) {
    int peer = (r + 1) % opt.ngpus;
    void* dst = G.bRecv;
    const void* src = gpus[peer].bSend;
    size_t sz = opt.b_chunk, idx = 0;
    cudaMemcpyAttributes attr{};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    CK(cudaMemcpyBatchAsync(&dst, &src, &sz, 1, &attr, &idx, 1, G.sB));
    CK(cudaMemcpyAsync(dst, src, sz, cudaMemcpyDeviceToDevice, G.sB));
  }
  CK(cudaStreamSynchronize(G.sB));
  if (G.ctxB) CK(cudaSetDevice(r));
  CK(cudaStreamSynchronize(G.sA));
  CK(cudaDeviceSynchronize());
}

static unsigned long long validate(int r) {
  Gpu& G = gpus[r];
  CK(cudaSetDevice(r));
  CK(cudaMemset(G.bad, 0, 8));
  size_t C = opt.b_bytes;
  if (opt.b == B_NCCL && opt.gated) C = opt.b_chunk;
  if (opt.b == B_NCCL_CE) C = opt.b_chunk;
  int skip_self = (opt.b == B_CE || opt.b == B_SM);
  check_recv<<<2048, 256>>>((const unsigned*)G.bRecv, opt.b_bytes, C, opt.ngpus, r, skip_self, G.bad);
  unsigned long long h;
  CK(cudaMemcpy(&h, G.bad, 8, cudaMemcpyDeviceToHost));
  return h;
}

static double pct(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  size_t i = std::min(v.size() - 1, (size_t)(q * (v.size() - 1) + 0.5));
  return v[i];
}

int main(int argc, char** argv) {
  parse(argc, argv);
  int N = opt.ngpus;
  CU(cuInit(0));
  NC(ncclGetUniqueId(&idA));
  NC(ncclGetUniqueId(&idB));
  {
    std::vector<std::thread> ts;
    for (int r = 0; r < N; ++r) ts.emplace_back(setup, r);
    for (auto& t : ts) t.join();
  }
  for (int r = 0; r < N; ++r) setup_peers(r);
  {
    std::vector<std::thread> ts;
    for (int r = 0; r < N; ++r) ts.emplace_back(warmup, r);
    for (auto& t : ts) t.join();
  }
  for (int r = 0; r < N; ++r) {
    printf("STREAMS dev=%d A=%llu B=", r, gpus[r].idA);
    for (size_t k = 0; k < gpus[r].idBx.size(); ++k) printf("%s%llu", k ? "," : "", gpus[r].idBx[k]);
    printf("\n");
  }
  fflush(stdout);

  std::barrier<> bar_all(2 * N), bar_a(N);
  std::vector<std::thread> ts;
  for (int r = 0; r < N; ++r) {
    ts.emplace_back(a_thread, r, &bar_all, &bar_a);
    ts.emplace_back(b_thread, r, &bar_all);
  }
  for (auto& t : ts) t.join();

  unsigned long long bad = 0;
  if (opt.validate && opt.b != B_NONE)
    for (int r = 0; r < N; ++r) bad += validate(r);

  // aggregate (skip iteration 0 = warmup)
  std::vector<double> a_all, g_all, fwd, bms, exits, launches, bfwd, bfwd_mean;
  int L = opt.layers;
  for (int it = 1; it < opt.iters; ++it) {
    double f = 0, b = 0;
    std::vector<double> it_rates;
    for (int r = 0; r < N; ++r) {
      f = std::max(f, (double)gpus[r].fwd_ms[it]);
      b = std::max(b, (double)gpus[r].b_ms[it]);
      // B goodput while the forward runs: bytes this rank had sent when the forward ended, over the
      // part of the forward B was alive for (B starts with the forward; it may finish first)
      double win = std::min((double)gpus[r].fwd_ms[it], (double)gpus[r].b_ms[it]);
      if (win > 0) it_rates.push_back(gpus[r].b_fwd_bytes[it] / (win * 1e-3) / 1e9);
      if (!gpus[r].exits_paused.empty()) {
        exits.push_back(gpus[r].exits_paused[it]);
        launches.push_back(gpus[r].launches[it]);
      }
    }
    fwd.push_back(f);
    bms.push_back(b);
    if (!it_rates.empty()) {
      bfwd.push_back(*std::min_element(it_rates.begin(), it_rates.end()));
      bfwd_mean.push_back(std::accumulate(it_rates.begin(), it_rates.end(), 0.0) / it_rates.size());
    }
    if (opt.fwd && !opt.graph)
      for (int l = 0; l < L; ++l) {
        double a = 0, g = 0;
        for (int r = 0; r < N; ++r) {
          a += gpus[r].a_ms[it][l];
          g += gpus[r].g_ms[it][l];
        }
        a_all.push_back(a / N);
        g_all.push_back(g / N);
      }
  }
  auto mean = [](const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return v.empty() ? 0.0 : s / v.size();
  };
  double b_ms = mean(bms);
  double b_gbps = b_ms > 0 ? (double)(N - 1) * opt.b_bytes / (b_ms * 1e-3) / 1e9 : 0;
  auto num = [](double v, const char* fmt) {
    char b[64];
    snprintf(b, sizeof b, fmt, v);
    return std::string(b);
  };
  const std::string null = "null";
  bool a_ok = opt.fwd && !opt.graph;
  // a single un-chunked NCCL call reports progress only on completion
  bool bfwd_ok = opt.b != B_NONE && !(opt.b == B_NCCL && !opt.gated);
  std::string o = "{";
  auto kv = [&](const char* k, const std::string& v) {
    o += (o.size() > 1 ? "," : "");
    o += "\"" + std::string(k) + "\":" + v;
  };
  auto ks = [&](const char* k, const std::string& v) { kv(k, "\"" + v + "\""); };
  ks("b", opt.bname);
  kv("fwd", std::to_string(opt.fwd));
  kv("prio", std::to_string(opt.prio));
  ks("scope", opt.global_scope ? "global" : "local");
  kv("layers", std::to_string(L));
  ks("a_ops", opt.a_ops);
  kv("a_mib", num(opt.a_bytes / 1048576.0, "%.1f"));
  kv("graph", std::to_string(opt.graph));
  kv("gemm_sms", std::to_string(opt.gemm_sms));
  kv("lead", std::to_string(opt.lead));
  kv("green_b", std::to_string(gpus[0].green_sms));
  kv("b_mib", num(opt.b_bytes / 1048576.0, "%.1f"));
  kv("b_chunk_kib", num(opt.b_chunk / 1024.0, "%.1f"));
  kv("b_item_kib", num(opt.b_item / 1024.0, "%.1f"));
  kv("b_ctas", std::to_string(opt.b_ctas));
  kv("b_nccl_ctas", std::to_string(opt.b_nccl_ctas));
  kv("ce_batch", std::to_string(opt.ce_batch));
  kv("ce_streams", std::to_string(opt.ce_streams));
  kv("b_pace_us", std::to_string(opt.b_pace_us));
  kv("fwd_ms", num(mean(fwd), "%.3f"));
  kv("a_mean_ms", a_ok ? num(mean(a_all), "%.4f") : null);
  kv("a_p50_ms", a_ok ? num(pct(a_all, .5), "%.4f") : null);
  kv("a_p90_ms", a_ok ? num(pct(a_all, .9), "%.4f") : null);
  kv("a_p99_ms", a_ok ? num(pct(a_all, .99), "%.4f") : null);
  kv("a_max_ms", a_ok ? num(pct(a_all, 1.0), "%.4f") : null);
  kv("gemm_mean_ms", a_ok ? num(mean(g_all), "%.4f") : null);
  kv("b_ms", num(b_ms, "%.3f"));
  kv("b_ingress_GBps", num(b_gbps, "%.1f"));
  kv("b_during_fwd_GBps", bfwd_ok ? num(mean(bfwd), "%.1f") : null);
  kv("b_during_fwd_mean_rank_GBps", bfwd_ok ? num(mean(bfwd_mean), "%.1f") : null);
  kv("sm_pause_exits", num(mean(exits), "%.1f"));
  kv("sm_launches", num(mean(launches), "%.1f"));
  kv("bad", std::to_string(bad));
  o += "}";
  printf("RESULT %s\n", o.c_str());
  fflush(stdout);
  // Skip NCCL teardown: comms with in-flight-free state are fine to drop at exit.
  return bad ? 2 : 0;
}
