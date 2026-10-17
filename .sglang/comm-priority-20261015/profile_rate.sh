#!/bin/bash
# Rate-limited, ungated B: does A stay unaffected when B runs through A at a bounded rate?
# Instantaneous cap (SM put with few CTAs, NCCL with few CTAs) vs average cap (paced CE chunks).
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd $H; source ./guard.sh; export CUDA_DEVICE_MAX_CONNECTIONS=32
BASE="--layers 16 --iters 3 --b-bytes 1G"
declare -A CFG=(
  [20_sm_cta1]="--b sm --b-ctas 1"
  [21_sm_cta2]="--b sm --b-ctas 2"
  [22_sm_cta4]="--b sm --b-ctas 4"
  [23_sm_cta8]="--b sm --b-ctas 8"
  [24_ce1M_pace0]="--b ce --b-chunk 1M --ce-streams 1 --b-pace-us 0"
  [25_ce1M_pace25]="--b ce --b-chunk 1M --ce-streams 1 --b-pace-us 25"
  [26_ce1M_pace50]="--b ce --b-chunk 1M --ce-streams 1 --b-pace-us 50"
  [27_ce1M_pace100]="--b ce --b-chunk 1M --ce-streams 1 --b-pace-us 100"
  [28_ce1M_pace200]="--b ce --b-chunk 1M --ce-streams 1 --b-pace-us 200"
  [29_nccl_cta2]="--b nccl --b-nccl-ctas 2"
  [30_nccl_cta4]="--b nccl --b-nccl-ctas 4"
)
for name in $(echo "${!CFG[@]}" | tr ' ' '\n' | sort); do
  echo "== $name: ${CFG[$name]}"
  for try in 1 2 3; do
    wait_idle
    guarded timeout -s KILL 400 nsys profile -o out/nsys/$name --force-overwrite=true -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
      --gpu-metrics-devices=0 --gpu-metrics-set=gb10x --gpu-metrics-frequency=100000 \
      ./out/prio_bench $BASE ${CFG[$name]} > out/nsys/$name.log 2>&1
    rc=$?; [ $rc != 99 ] && break; echo "   contaminated, retrying $name"
  done
  grep -E "^RESULT" out/nsys/$name.log | grep -o '"fwd_ms[^,]*\|"b_ingress[^,]*\|"b_during_fwd_GBps[^,]*\|"bad[^}]*' | tr '\n' ' '; echo
  nsys export --type sqlite --force-overwrite true -o out/nsys/$name.sqlite out/nsys/$name.nsys-rep > /dev/null 2>&1 && echo "   exported"
done
echo RATE_DONE
