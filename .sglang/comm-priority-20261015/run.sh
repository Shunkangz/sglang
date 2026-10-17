#!/bin/bash
# Run a command inside zsk-sglang as the host uid (writable mount) from the harness dir.
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec docker exec -u 101527:30 -e HOME=/home/scratch.shunkangz_gpu_1/.dockerhome/zsk \
  -e XDG_CACHE_HOME=/home/scratch.shunkangz_gpu_1/.dockerhome/zsk/.cache -w $H "${DOCKER_ENV[@]}" zsk-sglang bash -c "$*"
