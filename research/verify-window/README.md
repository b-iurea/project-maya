# GLM MTP verify-window cost on Strix Halo (2026-10-29)

Config: the baseline (Maya-S-v2, int8 KV, greedy, 2K prompts, 3 × 256 tokens), commit cf6472c.

- `union.md`: distinct experts per MoE layer over w consecutive decode tokens, computed from `route-log-2k.txt.gz` (`STRATA_GLM_ROUTE_LOG`, plain decode, 832 tokens × 42 layers).
- `fixed<n>-*`: MTP with `STRATA_GLM_MTP_FIXED=<n>` (n drafts per round, so a verify window of n+1 rows).

Rerun: `BENCH_LENS=2048 BENCH_MTP=1 BENCH_ENV='{"STRATA_GLM_MTP_FIXED":"2"}' python3 bench_ab.py <root> <cfg> <out> 3`
