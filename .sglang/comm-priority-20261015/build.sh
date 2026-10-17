#!/bin/bash
# Build inside zsk-sglang (CUDA 13.0, NCCL 2.30.7 from the pip wheel torch uses).
set -e
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NCCL=/opt/sglang/lib/python3.12/site-packages/nvidia/nccl
mkdir -p $H/out
nvcc -O3 -std=c++20 -arch=sm_100a -lineinfo -I$NCCL/include -o $H/out/${OUTBIN:-prio_bench} $H/src/prio_bench.cu \
  -L$NCCL/lib -l:libnccl.so.2 -lcublas -lcuda -ldl -Xlinker -rpath=$NCCL/lib
nvcc -O3 -std=c++20 -arch=sm_100a -o $H/out/probe $H/src/probe.cu -lcuda
nvcc -O3 -std=c++20 -arch=sm_100a -lineinfo -o $H/out/lat_probe $H/src/lat_probe.cu -lcuda
