| B mode | A | fwd ms | A mean / p99 us | GEMM + gate close us | B goodput during fwd GB/s | B total GB/s | SM pause exits / launches |
|---|---|---|---|---|---|---|---|
| none | ar 64M | 27.78 | 317 / 398 | 548 | n/a | - | - |
| none lookahead | ar 64M | 30.40 | 392 / 503 | 554 | n/a | - | - |
| none-gated | ar 64M | 27.73 | 293 / 305 | 556 | n/a | - | - |
| none-gated local | ar 64M | 28.41 | 321 / 356 | 559 | n/a | - | - |
| nccl | ar 64M | 38.42 | 553 / 799 | 640 | n/a | 490 | - |
| nccl equal-prio | ar 64M | 38.89 | 564 / 784 | 644 | n/a | 484 | - |
| nccl maxCTAs=4 | ar 64M | 27.59 | 318 / 324 | 540 | n/a | 88 | - |
| nccl-gated 4M | ar 64M | 30.18 | 302 / 340 | 622 | 31 | 167 | - |
| nccl-gated 16M | ar 64M | 32.48 | 350 / 392 | 647 | 116 | 281 | - |
| nccl-gated 64M | ar 64M | 44.25 | 767 / 812 | 583 | 329 | 339 | - |
| nccl-gated 16M local | ar 64M | 30.33 | 390 / 404 | 550 | 124 | 293 | - |
| nccl-ce 64M | ar 64M | 33.82 | 474 / 886 | 574 | 565 | 564 | - |
| nccl-ce-gated 4M | ar 64M | 28.16 | 288 / 310 | 573 | 214 | 267 | - |
| nccl-ce-gated 16M | ar 64M | 29.32 | 285 / 291 | 612 | 384 | 415 | - |
| ce 4M | ar 64M | 32.93 | 465 / 518 | 556 | 232 | 268 | - |
| ce-gated 1M | ar 64M | 27.59 | 283 / 288 | 561 | 96 | 140 | - |
| ce-gated 4M | ar 64M | 28.29 | 285 / 290 | 580 | 203 | 261 | - |
| ce-gated 16M | ar 64M | 29.59 | 296 / 318 | 609 | 263 | 318 | - |
| ce-gated 4M local | ar 64M | 27.32 | 288 / 300 | 558 | 208 | 264 | - |
| ce-gated 4M lookahead | ar 64M | 27.43 | 288 / 296 | 550 | 102 | 225 | - |
| sm 256K x16 | ar 64M | 34.01 | 500 / 572 | 558 | 254 | 269 | - |
| sm-preempt 64K x16 | ar 64M | 27.71 | 284 / 288 | 564 | 122 | 247 | 512 / 33 |
| sm-preempt 256K x16 | ar 64M | 27.98 | 286 / 292 | 570 | 143 | 316 | 512 / 33 |
| sm-preempt 1024K x16 | ar 64M | 28.16 | 291 / 297 | 571 | 151 | 342 | 512 / 33 |
| sm-preempt 256K x16 local | ar 64M | 27.34 | 293 / 322 | 554 | 144 | 318 | 512 / 33 |
| sm-preempt 256K x32 | ar 64M | 28.32 | 287 / 294 | 580 | 176 | 353 | 1024 / 33 |
| sm-preempt 256K x16 equal-prio | ar 64M | 27.92 | 285 / 294 | 569 | 143 | 319 | 512 / 33 |
| sm-preempt 256K x16 green16 | ar 64M | 27.54 | 283 / 287 | 559 | 6 | 286 | 512 / 33 |
| none | ar,ag,rs 64M | 25.03 | 226 / 381 | 553 | n/a | - | - |
| nccl | ar,ag,rs 64M | 34.28 | 390 / 793 | 673 | n/a | 467 | - |
| nccl-gated 16M | ar,ag,rs 64M | 29.74 | 266 / 400 | 645 | 126 | 295 | - |
| ce-gated 4M | ar,ag,rs 64M | 26.33 | 221 / 394 | 582 | 209 | 259 | - |
| sm-preempt 256K x16 | ar,ag,rs 64M | 25.86 | 219 / 360 | 571 | 147 | 318 | 512 / 33 |
| none | ar 8M | 21.38 | 116 / 135 | 549 | n/a | - | - |
| nccl | ar 8M | 28.39 | 184 / 254 | 695 | n/a | 470 | - |
| nccl-gated 16M | ar 8M | 26.58 | 167 / 246 | 645 | 140 | 304 | - |
| ce-gated 4M | ar 8M | 22.13 | 94 / 103 | 578 | 257 | 288 | - |
| ce-gated 1M | ar 8M | 21.27 | 90 / 93 | 556 | 124 | 138 | - |
| sm-preempt 256K x16 | ar 8M | 21.76 | 94 / 100 | 567 | 181 | 360 | 512 / 33 |
