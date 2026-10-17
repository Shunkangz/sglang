#!/bin/bash
# Is the no-B per-rank A inflation a function of run length (32 vs 16 layers), i.e. thermal/power skew?
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd $H; source ./guard.sh; export CUDA_DEVICE_MAX_CONNECTIONS=32
OUT=out/baseline_check.jsonl; rm -f $OUT
for rep in 1 2; do
  for c in "--layers 16 --b none" "--layers 32 --b none" "--layers 32 --b none-gated" "--layers 64 --b none"; do
    wait_idle
    timeout -s KILL 240 ./out/prio_bench --iters 6 --b-bytes 2G $c > /tmp/prio_run.log 2>&1
    grep -E "^RESULT" /tmp/prio_run.log | sed 's/^RESULT //' >> $OUT && echo "ok $c"
  done
done
wait_idle
nsys profile -o out/nsys/13_none_L32 --force-overwrite=true -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
  --gpu-metrics-devices=0 --gpu-metrics-set=gb10x --gpu-metrics-frequency=100000 \
  ./out/prio_bench --layers 32 --iters 3 --b none > out/nsys/13_none_L32.log 2>&1
nsys export --type sqlite --force-overwrite true -o out/nsys/13_none_L32.sqlite out/nsys/13_none_L32.nsys-rep > /dev/null 2>&1
echo CHECK_DONE
