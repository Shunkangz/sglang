#!/bin/bash
# Is A's sensitivity to an ungated B a property of the collective algorithm? Back-to-back 64 MiB all-reduces
# (tiny GEMM, so ranks stay aligned and the GEMM->A baseline anomaly is out of the picture), B = steady SM put.
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd $H; source ./guard.sh; export CUDA_DEVICE_MAX_CONNECTIONS=32
OUT=out/algo_sens.jsonl; rm -f $OUT
BASE="--layers 64 --iters 5 --gemm 256 256 256 --b-bytes 1G"
for algo in "NCCL_ALGO=NVLS NCCL_PROTO=Simple" "NCCL_ALGO=Ring NCCL_PROTO=Simple" "NCCL_ALGO=Ring NCCL_PROTO=LL128"; do
  for rep in 1 2; do
    for b in "--b none" "--b sm --b-ctas 2" "--b sm --b-ctas 8"; do
      wait_idle
      env $algo timeout -s KILL 240 ./out/prio_bench $BASE $b > /tmp/prio_run.log 2>&1
      line=$(grep -E "^RESULT" /tmp/prio_run.log | tail -1)
      [[ "$line" == RESULT* ]] && echo "{\"algo\":\"$algo\",${line#RESULT \{}" >> $OUT && echo "ok $algo $b" || echo "FAIL $algo $b :: $(tail -2 /tmp/prio_run.log)"
    done
  done
done
echo ALGO_DONE
