#!/bin/bash
# Window-limited test: if A is bounded by bytes in flight / round trip, more in flight per step should make A
# faster alone and less sensitive to B. Same setup as algo_sens.sh; one run each (algo_sens reps agreed within 1 us).
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd $H; source ./guard.sh; export CUDA_DEVICE_MAX_CONNECTIONS=32
OUT=out/window_test.jsonl; rm -f $OUT
BASE="--layers 64 --iters 5 --gemm 256 256 256 --b-bytes 1G"
for cfg in "NCCL_ALGO=NVLS NCCL_PROTO=Simple NCCL_NVLS_CHUNKSIZE=524288" \
           "NCCL_ALGO=NVLS NCCL_PROTO=Simple NCCL_NVLS_NCHANNELS=32" \
           "NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_BUFFSIZE=16777216" \
           "NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=32 NCCL_MAX_NCHANNELS=32"; do
  for b in "--b none" "--b sm --b-ctas 2" "--b sm --b-ctas 8"; do
    wait_idle
    env $cfg timeout -s KILL 240 ./out/prio_bench $BASE $b > /tmp/prio_run.log 2>&1
    line=$(grep -E "^RESULT" /tmp/prio_run.log | tail -1)
    [[ "$line" == RESULT* ]] && echo "{\"algo\":\"$cfg\",${line#RESULT \{}" >> $OUT && echo "ok $cfg $b" || echo "FAIL $cfg $b :: $(tail -2 /tmp/prio_run.log)"
  done
done
wait_idle
./out/lat_probe 5 > out/lat_probe.txt 2>&1
echo WINDOW_DONE
