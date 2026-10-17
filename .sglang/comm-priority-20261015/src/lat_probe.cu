// lat_probe.cu -- NVLink latency between GPU 0 and GPU 1 while all 8 GPUs push background traffic.
//
// Background: every GPU runs an all-to-all put kernel with K CTAs (steady, ~35 GB/s per CTA), or
// copy-engine pushes of 4 MiB to every peer (full CE rate).
// Probes (GPU 0 <-> GPU 1, one thread each, highest-priority streams):
//   pingpong: st.release.sys a flag into the peer, spin on the local flag for the echo -> round trip
//   chase:    dependent ld.volatile loads from an array in GPU 1's memory -> remote load latency
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
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

static constexpr int N = 8;
static constexpr size_t kShard = 256ull << 20;  // per-peer destination window
static constexpr size_t kItem = 256ull << 10;

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void st_release(unsigned* p, unsigned v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// dst[j], j < npeer: this GPU's window on the j-th target peer
__global__ void bg_put(const int4* __restrict__ src, int4* const* dst, int npeer, const volatile int* stop,
                       unsigned long long* moved) {
  __shared__ int s_stop;
  const size_t n16 = kItem / 16;
  for (unsigned long long it = blockIdx.x;; it += gridDim.x) {
    if (threadIdx.x == 0) s_stop = *stop;
    __syncthreads();
    if (s_stop) break;
    int j = (int)(it % npeer);
    size_t off16 = ((it / npeer) * kItem % kShard) / 16;
    const int4* s = src + off16;
    int4* d = dst[j] + off16;
    for (size_t i = threadIdx.x; i < n16; i += 4 * blockDim.x) {
      int4 v[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) v[u] = s[i + u * blockDim.x];
#pragma unroll
      for (int u = 0; u < 4; ++u) d[i + u * blockDim.x] = v[u];
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(moved, (unsigned long long)kItem);
  }
}

__global__ void ping(const unsigned* my_flag, unsigned* peer_flag, int iters, unsigned long long* out_ns) {
  unsigned long long t0 = gtimer();
  for (unsigned i = 1; i <= (unsigned)iters; ++i) {
    st_release(peer_flag, i);
    while (ld_acquire(my_flag) != i) {
    }
  }
  out_ns[0] = gtimer() - t0;
}

__global__ void pong(const unsigned* my_flag, unsigned* peer_flag, int iters) {
  for (unsigned i = 1; i <= (unsigned)iters; ++i) {
    while (ld_acquire(my_flag) != i) {
    }
    st_release(peer_flag, i);
  }
}

__global__ void chase(const unsigned* peer_arr, int iters, unsigned long long* out_ns, unsigned* sink) {
  unsigned idx = 0;
  unsigned long long t0 = gtimer();
  for (int i = 0; i < iters; ++i) idx = *((const volatile unsigned*)peer_arr + idx);
  out_ns[0] = gtimer() - t0;
  *sink = idx;
}

// Fence cost seen by one thread on GPU 0. mode 0: fence.acq_rel.sys alone; 1: fence.acq_rel.gpu alone;
// 2: remote st.global to GPU 1 then fence.acq_rel.sys (a release); 3: local st.global then fence.acq_rel.sys.
__global__ void fence_probe(int mode, int iters, unsigned* remote, unsigned* local, unsigned long long* out_ns) {
  unsigned long long t0 = gtimer();
  for (int i = 0; i < iters; ++i) {
    if (mode == 2) remote[0] = i;
    if (mode == 3) local[0] = i;
    if (mode == 1)
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
    else
      asm volatile("fence.acq_rel.sys;" ::: "memory");
  }
  out_ns[0] = gtimer() - t0;
}

struct Dev {
  cudaStream_t sBg, sProbe, sCtl;
  int4 *src, *dst;
  int4** dst_tab;      // all 7 peers
  int4** dst_tab_far;  // peers among GPUs 2-7 only (bystander mode)
  int* stop;
  unsigned long long* moved;
  unsigned* flag;
  unsigned long long* out;
  unsigned* sink;
};

int main(int argc, char** argv) {
  int reps = argc > 1 ? atoi(argv[1]) : 3;
  const int iters = 2000;
  Dev d[N];
  int lo, hi;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  for (int g = 0; g < N; ++g) {
    CK(cudaSetDevice(g));
    for (int p = 0; p < N; ++p)
      if (p != g) {
        cudaError_t e = cudaDeviceEnablePeerAccess(p, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CK(e);
        cudaGetLastError();
      }
    CK(cudaStreamCreateWithPriority(&d[g].sBg, cudaStreamNonBlocking, lo));
    CK(cudaStreamCreateWithPriority(&d[g].sProbe, cudaStreamNonBlocking, hi));
    CK(cudaStreamCreateWithPriority(&d[g].sCtl, cudaStreamNonBlocking, hi));
    CK(cudaMalloc(&d[g].src, kShard));
    CK(cudaMemset(d[g].src, 1, kShard));
    CK(cudaMalloc(&d[g].dst, kShard * N));
    CK(cudaMalloc(&d[g].dst_tab, sizeof(int4*) * N));
    CK(cudaMalloc(&d[g].dst_tab_far, sizeof(int4*) * N));
    CK(cudaMalloc(&d[g].stop, 4));
    CK(cudaMalloc(&d[g].moved, 8));
    CK(cudaMalloc(&d[g].flag, 4));
    CK(cudaMalloc(&d[g].out, 16));
    CK(cudaMalloc(&d[g].sink, 4));
  }
  for (int g = 0; g < N; ++g) {
    CK(cudaSetDevice(g));
    std::vector<int4*> tab, far;
    for (int k = 1; k < N; ++k) {
      int p = (g + k) % N;
      int4* w = (int4*)((char*)d[p].dst + (size_t)g * kShard);
      tab.push_back(w);
      if (p >= 2) far.push_back(w);
    }
    CK(cudaMemcpy(d[g].dst_tab, tab.data(), sizeof(int4*) * tab.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d[g].dst_tab_far, far.data(), sizeof(int4*) * far.size(), cudaMemcpyHostToDevice));
  }
  // pointer-chase ring in GPU 1's memory (64 MiB footprint, random cycle)
  const size_t nchase = 16u << 20;
  unsigned* chase_arr;
  CK(cudaSetDevice(1));
  CK(cudaMalloc(&chase_arr, nchase * 4));
  {
    std::vector<unsigned> perm(nchase), next(nchase);
    std::iota(perm.begin(), perm.end(), 0u);
    std::shuffle(perm.begin(), perm.end(), std::mt19937(7));
    for (size_t i = 0; i < nchase; ++i) next[perm[i]] = perm[(i + 1) % nchase];
    CK(cudaMemcpy(chase_arr, next.data(), nchase * 4, cudaMemcpyHostToDevice));
  }
  // warm every kernel once (lazy loading must not happen under load)
  for (int g = 0; g < N; ++g) {
    CK(cudaSetDevice(g));
    int one = 1;
    CK(cudaMemcpy(d[g].stop, &one, 4, cudaMemcpyHostToDevice));
    bg_put<<<1, 512, 0, d[g].sBg>>>(d[g].src, d[g].dst_tab, N - 1, d[g].stop, d[g].moved);
    CK(cudaDeviceSynchronize());
  }
  CK(cudaSetDevice(0));
  chase<<<1, 1, 0, d[0].sProbe>>>(chase_arr, 10, d[0].out, d[0].sink);
  for (int m = 0; m < 4; ++m) fence_probe<<<1, 1, 0, d[0].sProbe>>>(m, 2, d[1].flag, d[0].flag, d[0].out);
  CK(cudaDeviceSynchronize());

  // far: only GPUs 2-7 send, only to each other (GPUs 0 and 1 carry no background traffic)
  // into0: GPUs 1-7 push to all peers (GPU 0 receives, issues nothing); from0: only GPU 0 pushes
  if (argc > 2 && std::string(argv[2]) == "domain") {
    // Does a memory-sync domain split decouple a fence from another kernel's remote writes?
    // NCCL launches its kernels in the Remote domain; the background here is SM put from every GPU.
    int ndom = 0;
    CK(cudaDeviceGetAttribute(&ndom, cudaDevAttrMemSyncDomainCount, 0));
    printf("memSyncDomainCount=%d\n", ndom);
    auto launch_bg = [&](int g, int ctas, cudaLaunchMemSyncDomain dom) {
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = dim3(ctas);
      cfg.blockDim = dim3(512);
      cfg.stream = d[g].sBg;
      cudaLaunchAttribute at;
      at.id = cudaLaunchAttributeMemSyncDomain;
      at.val.memSyncDomain = dom;
      cfg.attrs = &at;
      cfg.numAttrs = 1;
      CK(cudaLaunchKernelEx(&cfg, bg_put, (const int4*)d[g].src, (int4* const*)d[g].dst_tab, N - 1,
                            (const volatile int*)d[g].stop, d[g].moved));
    };
    auto fence_in = [&](int mode, cudaLaunchMemSyncDomain dom) {
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = dim3(1);
      cfg.blockDim = dim3(1);
      cfg.stream = d[0].sProbe;
      cudaLaunchAttribute at;
      at.id = cudaLaunchAttributeMemSyncDomain;
      at.val.memSyncDomain = dom;
      cfg.attrs = &at;
      cfg.numAttrs = 1;
      CK(cudaSetDevice(0));
      CK(cudaLaunchKernelEx(&cfg, fence_probe, mode, iters, d[1].flag, d[0].flag, d[0].out));
      CK(cudaStreamSynchronize(d[0].sProbe));
      unsigned long long f;
      CK(cudaMemcpy(&f, d[0].out, 8, cudaMemcpyDeviceToHost));
      return f / (double)iters / 1e3;
    };
    const char* dn[2] = {"default", "remote"};
    cudaLaunchMemSyncDomain doms[2] = {cudaLaunchMemSyncDomainDefault, cudaLaunchMemSyncDomainRemote};
    for (int ctas : {0, 2, 8}) {
      for (int bd = 0; bd < 2; ++bd) {
        if (ctas == 0 && bd == 1) continue;
        for (int pd = 0; pd < 2; ++pd) {
          std::vector<double> fs, rs;
          for (int r = 0; r < reps; ++r) {
            for (int g = 0; g < N; ++g) {
              CK(cudaSetDevice(g));
              CK(cudaMemset(d[g].stop, 0, 4));
              CK(cudaDeviceSynchronize());
              if (ctas) launch_bg(g, ctas, doms[bd]);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            fs.push_back(fence_in(0, doms[pd]));
            rs.push_back(fence_in(2, doms[pd]));
            int one = 1;
            for (int g = 0; g < N; ++g) {
              CK(cudaSetDevice(g));
              CK(cudaMemcpyAsync(d[g].stop, &one, 4, cudaMemcpyHostToDevice, d[g].sCtl));
              CK(cudaDeviceSynchronize());
            }
          }
          std::sort(fs.begin(), fs.end());
          std::sort(rs.begin(), rs.end());
          printf(
              "DOMAIN {\"bg_ctas\":%d,\"bg_domain\":\"%s\",\"fence_domain\":\"%s\",\"fence_sys_us\":%.3f,"
              "\"remote_store_fence_sys_us\":%.3f}\n",
              ctas, dn[bd], dn[pd], fs[fs.size() / 2], rs[rs.size() / 2]);
          fflush(stdout);
        }
      }
    }
    return 0;
  }
  struct Bg {
    std::string name;
    int ctas;
    bool ce;
    bool far;
    int only = -1;
    bool skip0 = false;
  };
  std::vector<Bg> bgs = {{"none", 0, false, false},
                         {"sm x1", 1, false, false},
                         {"sm x2", 2, false, false},
                         {"sm x4", 4, false, false},
                         {"sm x8", 8, false, false},
                         {"sm x16", 16, false, false},
                         {"ce 4MiB", 0, true, false},
                         {"bystander sm x2", 2, false, true},
                         {"bystander sm x8", 8, false, true},
                         {"bystander sm x16", 16, false, true},
                         {"into GPU0 sm x4", 4, false, false, -1, true},
                         {"from GPU0 only sm x4", 4, false, false, 0, false},
                         {"from GPU0 only sm x16", 16, false, false, 0, false}};
  for (auto& b : bgs) {
    std::vector<double> rtt, lat, rate;
    std::vector<double> fence_us[4];
    for (int r = 0; r < reps; ++r) {
      // reset
      for (int g = 0; g < N; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaMemset(d[g].stop, 0, 4));
        CK(cudaMemset(d[g].moved, 0, 8));
        CK(cudaMemset(d[g].flag, 0, 4));
        CK(cudaDeviceSynchronize());
      }
      std::vector<cudaEvent_t> e0(N), e1(N);
      size_t ce_bytes = 0;
      auto t_start = std::chrono::steady_clock::now();
      for (int g = 0; g < N; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaEventCreate(&e0[g]));
        CK(cudaEventCreate(&e1[g]));
        CK(cudaEventRecord(e0[g], d[g].sBg));
        if (b.ctas && !(b.far && g < 2) && !(b.skip0 && g == 0) && !(b.only >= 0 && g != b.only))
          bg_put<<<b.ctas, 512, 0, d[g].sBg>>>(d[g].src, b.far ? d[g].dst_tab_far : d[g].dst_tab,
                                               b.far ? (g >= 2 ? N - 3 : 0) : N - 1, d[g].stop, d[g].moved);
        if (b.ce) {
          // ~90 ms of copy-engine push to every peer (must outlast the probes)
          for (int rnd = 0; rnd < 1200; ++rnd)
            for (int p = 1; p < N; ++p) {
              int peer = (g + p) % N;
              CK(cudaMemcpyAsync((char*)d[peer].dst + (size_t)g * kShard + (rnd % 32) * (4u << 20), d[g].src, 4u << 20,
                                 cudaMemcpyDeviceToDevice, d[g].sBg));
            }
          ce_bytes = 1200ull * (N - 1) * (4u << 20);
        }
        CK(cudaEventRecord(e1[g], d[g].sBg));
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      // probes
      CK(cudaSetDevice(1));
      pong<<<1, 1, 0, d[1].sProbe>>>(d[1].flag, d[0].flag, iters);
      CK(cudaSetDevice(0));
      ping<<<1, 1, 0, d[0].sProbe>>>(d[0].flag, d[1].flag, iters, d[0].out);
      CK(cudaStreamSynchronize(d[0].sProbe));
      CK(cudaSetDevice(1));
      CK(cudaStreamSynchronize(d[1].sProbe));
      CK(cudaSetDevice(0));
      chase<<<1, 1, 0, d[0].sProbe>>>(chase_arr, iters, d[0].out + 1, d[0].sink);
      CK(cudaStreamSynchronize(d[0].sProbe));
      unsigned long long h[2];
      CK(cudaMemcpy(h, d[0].out, 16, cudaMemcpyDeviceToHost));
      double fns[4];
      for (int m = 0; m < 4; ++m) {
        fence_probe<<<1, 1, 0, d[0].sProbe>>>(m, iters, d[1].flag, d[0].flag, d[0].out);
        CK(cudaStreamSynchronize(d[0].sProbe));
        unsigned long long f;
        CK(cudaMemcpy(&f, d[0].out, 8, cudaMemcpyDeviceToHost));
        fns[m] = f / (double)iters / 1e3;
      }
      for (int m = 0; m < 4; ++m) fence_us[m].push_back(fns[m]);
      auto t_probe = std::chrono::steady_clock::now();
      // stop the background and measure its rate on GPU 0
      unsigned long long moved0 = 0;
      if (b.ctas) {
        int rg = b.far ? 2 : (b.skip0 ? 1 : 0);
        CK(cudaSetDevice(rg));
        CK(cudaMemcpyAsync(&moved0, d[rg].moved, 8, cudaMemcpyDeviceToHost, d[rg].sCtl));
        CK(cudaStreamSynchronize(d[rg].sCtl));
        int one = 1;
        for (int g = 0; g < N; ++g) {
          CK(cudaSetDevice(g));
          CK(cudaMemcpyAsync(d[g].stop, &one, 4, cudaMemcpyHostToDevice, d[g].sCtl));
          CK(cudaStreamSynchronize(d[g].sCtl));
        }
      }
      for (int g = 0; g < N; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaStreamSynchronize(d[g].sBg));
      }
      double secs = std::chrono::duration<double>(t_probe - t_start).count();
      double gbps = 0;
      if (b.ctas) gbps = moved0 / secs / 1e9;
      if (b.ce) {
        float ms;
        CK(cudaSetDevice(0));
        CK(cudaEventElapsedTime(&ms, e0[0], e1[0]));
        gbps = ce_bytes / (ms * 1e-3) / 1e9;
        if (ms * 1e-3 < secs) fprintf(stderr, "warning: CE background ended before the probes (%.1f ms)\n", ms);
      }
      rtt.push_back(h[0] / (double)iters / 1e3);
      lat.push_back(h[1] / (double)iters / 1e3);
      rate.push_back(gbps);
      for (int g = 0; g < N; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaEventDestroy(e0[g]));
        CK(cudaEventDestroy(e1[g]));
      }
    }
    auto med = [](std::vector<double> v) {
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    printf(
        "RESULT {\"bg\":\"%s\",\"bg_GBps_per_gpu\":%.1f,\"flag_rtt_us\":%.3f,\"remote_load_us\":%.3f,"
        "\"fence_sys_us\":%.3f,\"fence_gpu_us\":%.3f,\"remote_store_fence_sys_us\":%.3f,\"local_store_fence_sys_us\":%."
        "3f}\n",
        b.name.c_str(), med(rate), med(rtt), med(lat), med(fence_us[0]), med(fence_us[1]), med(fence_us[2]),
        med(fence_us[3]));
    fflush(stdout);
  }
  return 0;
}
