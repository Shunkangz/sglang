| B alone (2 GiB shard per rank) | ingress GB/s per GPU | B time ms |
|---|---|---|
| NCCL all-gather (SM kernels), one call | 668 | 22.5 |
| NCCL zero-SM CE all-gather, 64 MiB calls | 709 | 21.2 |
| CE push, 4 MiB chunks, 1 stream(s) | 310 | 48.5 |
| CE push, 4 MiB chunks, 4 stream(s) | 317 | 47.4 |
| CE push, 4 MiB chunks, 7 stream(s) | 310 | 48.4 |
| CE push, 16 MiB chunks, 4 stream(s) | 421 | 35.7 |
| SM put kernel, 16 CTAs | 611 | 24.6 |
| SM put kernel, 32 CTAs | 702 | 21.4 |
