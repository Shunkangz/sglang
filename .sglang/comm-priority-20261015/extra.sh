#!/bin/bash
# Controls and ceilings the design review asked for: B-only bandwidth per mover, matched gated controls
# (lookahead, 8 MiB A, AR/AG/RS rotation), M1 with local scope, and the capability probe log.
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd $H; source ./guard.sh; export CUDA_DEVICE_MAX_CONNECTIONS=32
OUT=out/extra.jsonl; rm -f $OUT
./out/probe > out/probe.txt 2>&1
CONFIGS=(
  "--fwd 0 --b nccl"
  "--fwd 0 --b nccl-ce --b-chunk 64M"
  "--fwd 0 --b ce --b-chunk 4M --ce-streams 1"
  "--fwd 0 --b ce --b-chunk 4M --ce-streams 4"
  "--fwd 0 --b ce --b-chunk 4M --ce-streams 7"
  "--fwd 0 --b ce --b-chunk 16M --ce-streams 4"
  "--fwd 0 --b sm --b-ctas 16"
  "--fwd 0 --b sm --b-ctas 32"
  "--b none-gated --lead 1"
  "--b nccl-ce-gated --b-chunk 4M --scope local"
  "--b nccl-ce-gated --b-chunk 16M --scope local"
  "--b ce-gated --b-chunk 4M --scope local"
  "--b none-gated --a-bytes 8M"
  "--b none-gated --a-ops ar,ag,rs"
)
for c in "${CONFIGS[@]}"; do
  for try in 1 2 3; do
    wait_idle
    guarded bash -c "timeout -s KILL 240 ./out/prio_bench --layers 32 --iters 6 --b-bytes 2G $c > /tmp/prio_run.log 2>&1"; rc=$?
    [ $rc != 99 ] && break
    echo "   contaminated, retrying: $c"
  done
  line=$(grep -E "^RESULT" /tmp/prio_run.log | tail -1)
  [[ "$line" == RESULT* ]] && echo "${line#RESULT }" >> $OUT && echo "ok   $c" || echo "FAIL $c :: $(tail -2 /tmp/prio_run.log)"
done
echo EXTRA_DONE
