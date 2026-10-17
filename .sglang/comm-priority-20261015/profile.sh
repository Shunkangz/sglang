#!/bin/bash
# nsys timelines for the key modes (one warm-up + one measured forward each), with NVLink metrics on GPU 0.
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd $H
source ./guard.sh
export CUDA_DEVICE_MAX_CONNECTIONS=32
BASE="--layers 16 --iters 3 --b-bytes 1G"
declare -A CFG=(
  [01_none]="--b none"
  [02_nccl_naive]="--b nccl"
  [03_nccl_gated_16M]="--b nccl-gated --b-chunk 16M"
  [04_nccl_ce_gated_16M]="--b nccl-ce-gated --b-chunk 16M"
  [05_ce_gated_4M]="--b ce-gated --b-chunk 4M"
  [06_ce_gated_4M_lead]="--b ce-gated --b-chunk 4M --lead 1"
  [07_sm_naive]="--b sm"
  [08_sm_preempt_256K]="--b sm-preempt --b-item 256K"
  [09_ce_naive_4M]="--b ce --b-chunk 4M"
  [10_none_gated]="--b none-gated"
  [11_nccl_ce_gated_4M]="--b nccl-ce-gated --b-chunk 4M"
  [12_sm_preempt_256K_local]="--b sm-preempt --b-item 256K --scope local"
)
for name in $(echo "${!CFG[@]}" | tr ' ' '\n' | sort); do
  [ -n "$ONLY" ] && [[ "$name" != *$ONLY* ]] && continue
  echo "== $name: ${CFG[$name]}"
  for try in 1 2 3; do
  wait_idle
  guarded timeout -s KILL 300 nsys profile -o out/nsys/$name --force-overwrite=true -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
    --gpu-metrics-devices=0 --gpu-metrics-set=gb10x --gpu-metrics-frequency=100000 \
    ./out/prio_bench $BASE ${CFG[$name]} > out/nsys/$name.log 2>&1
  rc=$?; [ $rc != 99 ] && break; echo "   contaminated, retrying $name"
  done
  grep -E "^RESULT|->|error" out/nsys/$name.log | tail -1 | cut -c1-200
  nsys export --type sqlite --force-overwrite true -o out/nsys/$name.sqlite out/nsys/$name.nsys-rep > /dev/null 2>&1 && echo "   exported"
done
echo PROFILE_DONE
