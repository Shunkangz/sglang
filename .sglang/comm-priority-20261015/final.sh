#!/bin/bash
# Final measurement pass with the reviewed binary: repetitions first (headline table), then the
# full matrix, then nsys timelines. Every run waits for idle GPUs (guard.sh).
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd $H
mkdir -p out/old && mv -f out/sweep.jsonl out/repeat.jsonl out/old/ 2>/dev/null
OUT=out/repeat.jsonl ./repeat.sh
OUT=out/sweep.jsonl ./sweep.sh
rm -f out/nsys/*
./profile.sh
echo FINAL_DONE
