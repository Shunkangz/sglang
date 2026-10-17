#!/bin/bash
# The sweep host (umbriel-b200-048) rebooted before profiling; this reruns the nsys phase on the new
# host, preceded by a one-rep cross-host check of the headline configs so the traces are comparable.
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd $H
source ./guard.sh
export CUDA_DEVICE_MAX_CONNECTIONS=32
hostname > out/profile_host.txt
BASE="--layers 32 --iters 6 --b-bytes 2G"
for c in "--b none-gated" "--b nccl" "--b nccl-ce-gated --b-chunk 4M" "--b ce-gated --b-chunk 4M" "--b sm-preempt --b-item 256K"; do
  wait_idle
  timeout -s KILL 240 ./out/prio_bench $BASE $c > /tmp/prio_run.log 2>&1
  line=$(grep -E "^RESULT" /tmp/prio_run.log | tail -1)
  [[ "$line" == RESULT* ]] && echo "${line#RESULT }" >> out/crosshost.jsonl && echo "ok   $c" || echo "FAIL $c"
done
echo CROSSHOST_DONE
rm -f out/nsys/*
./profile.sh
echo RESUME_DONE
