#!/bin/bash
# Main measurement matrix (no profiler). Runs inside zsk-sglang; appends RESULT json to out/sweep.jsonl.
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd $H
source ./guard.sh
export CUDA_DEVICE_MAX_CONNECTIONS=32
OUT=${OUT:-out/sweep.jsonl}
BASE="--layers 32 --iters 6 --b-bytes 2G"
CONFIGS=(
  "--b none"
  "--b none --lead 1"
  "--b none-gated"
  "--b none-gated --scope local"
  "--b nccl"
  "--b nccl --prio 0"
  "--b nccl --b-nccl-ctas 4"
  "--b nccl-gated --b-chunk 4M"
  "--b nccl-gated --b-chunk 16M"
  "--b nccl-gated --b-chunk 64M"
  "--b nccl-gated --b-chunk 16M --scope local"
  "--b nccl-ce --b-chunk 64M"
  "--b nccl-ce-gated --b-chunk 4M"
  "--b nccl-ce-gated --b-chunk 16M"
  "--b ce --b-chunk 4M"
  "--b ce-gated --b-chunk 1M"
  "--b ce-gated --b-chunk 4M"
  "--b ce-gated --b-chunk 16M"
  "--b ce-gated --b-chunk 4M --scope local"
  "--b ce-gated --b-chunk 4M --lead 1"
  "--b sm"
  "--b sm-preempt --b-item 64K"
  "--b sm-preempt --b-item 256K"
  "--b sm-preempt --b-item 1M"
  "--b sm-preempt --b-item 256K --scope local"
  "--b sm-preempt --b-item 256K --b-ctas 32"
  "--b sm-preempt --b-item 256K --prio 0"
  "--b sm-preempt --b-item 256K --green-b 16"
  "--b none --a-ops ar,ag,rs"
  "--b nccl --a-ops ar,ag,rs"
  "--b nccl-gated --b-chunk 16M --a-ops ar,ag,rs"
  "--b ce-gated --b-chunk 4M --a-ops ar,ag,rs"
  "--b sm-preempt --b-item 256K --a-ops ar,ag,rs"
  "--b none --a-bytes 8M"
  "--b nccl --a-bytes 8M"
  "--b nccl-gated --b-chunk 16M --a-bytes 8M"
  "--b ce-gated --b-chunk 4M --a-bytes 8M"
  "--b ce-gated --b-chunk 1M --a-bytes 8M"
  "--b sm-preempt --b-item 256K --a-bytes 8M"
)
for c in "${CONFIGS[@]}"; do
  for try in 1 2 3; do
    wait_idle
    guarded bash -c "timeout -s KILL 180 ./out/prio_bench $BASE $c > /tmp/prio_run.log 2>&1"; rc=$?
    [ $rc != 99 ] && break
    echo "   contaminated by a foreign GPU process, retrying: $c"
  done
  line=$(grep -E "^RESULT|->|error" /tmp/prio_run.log | tail -1)
  if [[ "$line" == RESULT* ]]; then echo "${line#RESULT }" >> $OUT; echo "ok   $c"; else echo "FAIL $c :: $line"; echo "{\"fail\":\"$c\"}" >> $OUT; fi
done
echo SWEEP_DONE
