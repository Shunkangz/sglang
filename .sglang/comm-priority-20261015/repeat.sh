#!/bin/bash
# Headline table: key configs x 3 repetitions, interleaved (round-robin) so drift hits every config alike.
# Then CUDA-graph variants (the forward, gate writes included, captured into one graph).
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd $H
source ./guard.sh
export CUDA_DEVICE_MAX_CONNECTIONS=32
BIN=${BIN:-./out/prio_bench}
OUT=${OUT:-out/repeat.jsonl}
BASE="--layers 32 --iters 6 --b-bytes 2G"
CONFIGS=(
  "--b none"
  "--b none-gated"
  "--b nccl"
  "--b nccl-gated --b-chunk 16M"
  "--b nccl-ce-gated --b-chunk 4M"
  "--b ce-gated --b-chunk 4M"
  "--b ce-gated --b-chunk 4M --lead 1"
  "--b sm-preempt --b-item 256K"
)
GRAPH=(
  "--b none --graph 1"
  "--b nccl --graph 1"
  "--b ce-gated --b-chunk 4M --graph 1"
  "--b nccl-ce-gated --b-chunk 4M --graph 1"
  "--b sm-preempt --b-item 256K --graph 1"
  "--b nccl-gated --b-chunk 16M --graph 1"
)
run() {
  for try in 1 2 3; do
    wait_idle
    guarded bash -c "timeout -s KILL 240 $BIN $BASE $1 > /tmp/prio_run.log 2>&1"; rc=$?
    [ $rc != 99 ] && break
    echo "   contaminated by a foreign GPU process, retrying: $1"
  done
  line=$(grep -E "^RESULT|->|error|captured" /tmp/prio_run.log | tr '\n' ' ')
  if [[ "$line" == *RESULT* ]]; then echo "${line##*RESULT }" >> $OUT; echo "ok   $1 ${line%%RESULT*}"; else echo "FAIL $1 :: $line"; echo "{\"fail\":\"$1\"}" >> $OUT; fi
}
for rep in 1 2 3; do for c in "${CONFIGS[@]}"; do run "$c"; done; done
[ -z "$NOGRAPH" ] && for c in "${GRAPH[@]}"; do run "$c"; done
echo REPEAT_DONE
