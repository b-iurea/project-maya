# Complete expert pool on Linux UMA (Strix Halo, 2026-10-29)

`uma-complete.patch` (throwaway, env-gated `STRATA_GLM_UMA_COMPLETE=1`): on Linux unified memory the pool budget may exceed the experts' bytes (still MemAvailable - headroom - reserve), and `complete_pool_slots` is no longer skipped for `unified_memory`. The prompt then borrows only the slots beyond every expert + 3 spares.

2K, 3 runs, baseline config + the patch:
- `plain-and-mtp-h16/`: headroom 16 GB. Plain: complete pool taken (87.13 GB, 313 slots/layer). MTP: NOT taken (needs ~89.5 GB with the draft layer, budget 85.66) - old pool.
- `mtp-h11/`: MTP with `MAYA_GLM_RAM_HEADROOM_GB=11`: complete pool taken (89.50 GB, 312 slots/layer).

Rerun: `BENCH_LENS=2048 BENCH_MTP=<0|1> BENCH_ENV='{"STRATA_GLM_UMA_COMPLETE":"1"[,"MAYA_GLM_RAM_HEADROOM_GB":"11"]}' python3 bench_ab.py <maya-root> <config.json> <out> 3`
