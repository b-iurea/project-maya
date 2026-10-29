# Is moe_wait a host round trip? (2026-10-29)

No. `moe_route_kernel` sets `wait_seq` only when a routed expert is on disk alone (`mask`, src/kernels/cuda/glm_fast.cu:4192); otherwise `moe_wait_kernel` returns at once. The profile's 10.2 ms/token of `moe_wait` equals the engine's own `disk ... 10.13 ms/tok`: SSD refills of experts the prompt's pool loan (6.1 GB) evicted.

A/B at 2K, 3 runs, baseline config + `MAYA_GLM_PREFILL_MB=1024 MAYA_GLM_PRESTAGE=0` (pool stays full): `plain/` and `mtp/`.
Rerun: `BENCH_LENS=2048 BENCH_MTP=<0|1> BENCH_ENV='{"MAYA_GLM_PREFILL_MB":"1024","MAYA_GLM_PRESTAGE":"0"}' python3 bench_ab.py <maya-root> <config.json> <out> 3`
