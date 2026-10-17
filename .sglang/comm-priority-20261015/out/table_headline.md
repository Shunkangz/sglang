| B mode | B SMs | fwd ms (vs none-gated) | A mean / p99 us | GEMM + gate close us | B goodput during fwd GB/s | B total ms (GB/s) | reps |
|---|---|---|---|---|---|---|---|
| none | - | 28.44 (+1.6%) | 334 / 410 | 552 | n/a | - | 3 |
| none-gated | - | 28.00 | 300 / 326 | 555 | n/a | - | 3 |
| nccl | NCCL CTAs | 38.88 (+38.8%) | 562 / 796 | 645 | n/a | 31.5 (477) | 3 |
| nccl-gated 16M | NCCL CTAs | 32.50 (+16.1%) | 349 / 398 | 648 | 116 | 53.5 (281) | 3 |
| nccl-ce-gated 4M | 0 | 28.12 (+0.4%) | 288 / 307 | 572 | 212 | 56.6 (266) | 3 |
| ce-gated 4M | 0 | 28.33 (+1.2%) | 286 / 292 | 580 | 202 | 57.9 (260) | 3 |
| ce-gated 4M lookahead | 0 | 27.43 (-2.1%) | 288 / 296 | 550 | 102 | 66.6 (226) | 3 |
| sm-preempt 256K x16 | 16 CTAs (0 paused) | 27.96 (-0.1%) | 286 / 292 | 570 | 143 | 47.2 (318) | 3 |
| none graph | - | 28.75 (+2.7%) | n/a / n/a | n/a | n/a | - | 1 |
| nccl graph | NCCL CTAs | 38.52 (+37.6%) | n/a / n/a | n/a | n/a | 30.2 (498) | 1 |
| ce-gated 4M graph | 0 | 27.92 (-0.3%) | n/a / n/a | n/a | 204 | 57.5 (262) | 1 |
| nccl-ce-gated 4M graph | 0 | 27.75 (-0.9%) | n/a / n/a | n/a | 213 | 56.5 (266) | 1 |
| sm-preempt 256K x16 graph | 16 CTAs (0 paused) | 27.68 (-1.1%) | n/a / n/a | n/a | 143 | 46.9 (320) | 1 |
| nccl-gated 16M graph | NCCL CTAs | 32.08 (+14.6%) | n/a / n/a | n/a | 117 | 53.2 (282) | 1 |
