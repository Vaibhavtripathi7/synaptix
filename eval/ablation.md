16 questions over 2 documents (`eval/golden.jsonl`); query expansion off; `gemini-embedding-001`.

| variant | hit@1 | hit@8 | MRR | input tokens / question | p50 latency |
|---|---:|---:|---:|---:|---:|
| dense only | 0.94 | 1.00 | 0.97 | 0 | 0.03 s |
| BM25 only | 0.56 | 0.94 | 0.67 | 0 | 0.04 s |
| hybrid (RRF) | 0.88 | 1.00 | 0.93 | 0 | 0.06 s |
