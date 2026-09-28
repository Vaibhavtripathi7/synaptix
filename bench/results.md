Server on 4 cores, load generator on 4 others, 64 connections, 15s per run after a warm-up run.

| processes | path | req/s | p50 ms | p99 ms | success |
|---|---|---:|---:|---:|---:|
| 1 | framework floor | 3966 | 16.3 | 34.2 | 1.000 |
| 1 | authenticated DB read | 250 | 254.4 | 326.7 | 1.000 |
| 1 | cached answer | 282 | 223.9 | 303.2 | 1.000 |
| 4 | framework floor | 7316 | 7.9 | 20.1 | 1.000 |
| 4 | authenticated DB read | 589 | 106.3 | 169.7 | 1.000 |
| 4 | cached answer | 610 | 100.2 | 164.7 | 1.000 |
