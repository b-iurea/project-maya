42 MoE layers, 832 decode tokens per layer

| window rows | distinct experts / layer (mean) | x one token | expert reads saved vs w separate tokens |
|---:|---:|---:|---:|
| 1 | 8.00 | 1.00x | 0% |
| 2 | 13.91 | 1.74x | 13% |
| 3 | 19.18 | 2.40x | 20% |
| 4 | 24.07 | 3.01x | 25% |
