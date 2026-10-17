// Capability probe for the comm-priority benchmark.
// Checks: stream priority range, copy engines, stream mem-op support, and whether
// cuStreamWriteValue32 / cuStreamWaitValue32 work on a P2P-mapped peer address.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CK(x)                                                                    \
  do {                                                                           \
    cudaError_t e = (x);                                                         \
    if (e != cudaSuccess) {                                                      \
      printf("%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e)); \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)
#define CU(x)                                                \
  do {                                                       \
    CUresult r = (x);                                        \
    if (r != CUDA_SUCCESS) {                                 \
      const char* s;                                         \
      cuGetErrorString(r, &s);                               \
      printf("%s:%d %s -> %s\n", __FILE__, __LINE__, #x, s); \
    } else                                                   \
      printf("  ok: %s\n", #x);                              \
  } while (0)

int main() {
  int drv;
  cudaDriverGetVersion(&drv);
  int rt;
  cudaRuntimeGetVersion(&rt);
  printf("driver API %d runtime %d\n", drv, rt);
  int lo, hi;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  printf("stream priority range: least=%d greatest=%d\n", lo, hi);
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  printf("%s SMs=%d asyncEngineCount=%d concurrentKernels=%d computePreemption=%d\n", p.name, p.multiProcessorCount,
         p.asyncEngineCount, p.concurrentKernels, 0);
  int v;
  cudaDeviceGetAttribute(&v, cudaDevAttrComputePreemptionSupported, 0);
  printf("computePreemptionSupported=%d\n", v);
  CUdevice d0;
  cuInit(0);
  cuDeviceGet(&d0, 0);
  cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR, d0);
  printf("CAN_USE_STREAM_WAIT_VALUE_NOR=%d\n", v);
  cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, d0);
  printf("CAN_USE_64_BIT_STREAM_MEM_OPS=%d\n", v);
  cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, d0);
  printf("CAN_FLUSH_REMOTE_WRITES=%d\n", v);

  // Peer mem-op test: GPU1's stream writes a flag that lives on GPU0; GPU0's stream waits on it.
  int canp;
  CK(cudaDeviceCanAccessPeer(&canp, 1, 0));
  printf("canAccessPeer(1->0)=%d\n", canp);
  CK(cudaSetDevice(0));
  uint32_t* flag0;
  CK(cudaMalloc(&flag0, 4));
  CK(cudaMemset(flag0, 0, 4));
  uint32_t* out0;
  CK(cudaMalloc(&out0, 4));
  CK(cudaMemset(out0, 0, 4));
  cudaStream_t s0;
  CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
  CK(cudaSetDevice(1));
  CK(cudaDeviceEnablePeerAccess(0, 0));
  cudaStream_t s1;
  CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));

  CK(cudaSetDevice(0));
  printf("enqueue wait on GPU0 local flag (EQ 7) then write out0=1\n");
  CU(cuStreamWaitValue32((CUstream)s0, (CUdeviceptr)flag0, 7, CU_STREAM_WAIT_VALUE_EQ));
  CU(cuStreamWriteValue32((CUstream)s0, (CUdeviceptr)out0, 1, 0));
  CK(cudaSetDevice(1));
  printf("GPU1 stream writes 7 into GPU0's flag (peer address)\n");
  CU(cuStreamWriteValue32((CUstream)s1, (CUdeviceptr)flag0, 7, 0));
  CK(cudaStreamSynchronize(s1));
  CK(cudaSetDevice(0));
  CK(cudaStreamSynchronize(s0));
  uint32_t h = 0;
  CK(cudaMemcpy(&h, out0, 4, cudaMemcpyDeviceToHost));
  printf("peer write released local wait: out0=%u (expect 1)\n", h);

  // GPU0 waits on a flag that lives on GPU1 (wait on a peer address)
  CK(cudaSetDevice(1));
  uint32_t* flag1;
  CK(cudaMalloc(&flag1, 4));
  CK(cudaMemset(flag1, 0, 4));
  CK(cudaSetDevice(0));
  CK(cudaDeviceEnablePeerAccess(1, 0));
  printf("GPU0 stream waits on a flag in GPU1 memory\n");
  CU(cuStreamWaitValue32((CUstream)s0, (CUdeviceptr)flag1, 3, CU_STREAM_WAIT_VALUE_GEQ));
  CU(cuStreamWriteValue32((CUstream)s0, (CUdeviceptr)out0, 2, 0));
  CK(cudaSetDevice(1));
  CU(cuStreamWriteValue32((CUstream)s1, (CUdeviceptr)flag1, 3, 0));
  CK(cudaStreamSynchronize(s1));
  CK(cudaSetDevice(0));
  CK(cudaStreamSynchronize(s0));
  CK(cudaMemcpy(&h, out0, 4, cudaMemcpyDeviceToHost));
  printf("wait on peer address: out0=%u (expect 2)\n", h);
  printf("done\n");
  return 0;
}
