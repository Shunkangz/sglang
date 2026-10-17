| B mode | A | fwd ms | A mean / p99 us | GEMM + gate close us | B goodput during fwd GB/s | B total GB/s | SM pause exits / launches |
|---|---|---|---|---|---|---|---|
| none-gated lookahead | ar 64M | 28.17 | 318 / 365 | 544 | n/a | - | - |
| nccl-ce-gated 4M local | ar 64M | 27.53 | 295 / 317 | 558 | 216 | 268 | - |
| nccl-ce-gated 16M local | ar 64M | 28.71 | 304 / 320 | 585 | 393 | 418 | - |
| ce-gated 4M local | ar 64M | 27.48 | 291 / 313 | 560 | 208 | 264 | - |
| none-gated | ar 8M | 21.89 | 102 / 112 | 563 | n/a | - | - |
| none-gated | ar,ag,rs 64M | 26.36 | 234 / 382 | 571 | n/a | - | - |
