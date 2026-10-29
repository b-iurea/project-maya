# GLM decode profile on Strix Halo (2026-10-29)

rocprofv3 kernel trace of 128-token greedy decodes on the baseline config (Maya-S-v2, int8 KV, commit cf6472c), at 2K and 64K prompts, with MTP off and on. Each decode is bracketed by CLOCK_MONOTONIC marks. The profiler adds about 9%: 64.4 ms/token here vs 59 ms without it.

Rerun: `rocprofv3 --kernel-trace --output-format csv -d <dir> -o run -- python3 profile_decode.py <maya-root> <config.json> <out> <0|1> 128`, then `python3 analyze_trace.py <dir>/run_kernel_trace.csv <marks.json>`. Raw traces (~90 MB each) are not committed.

Full tables: `breakdown-mtp0.md` (plain) and `breakdown-mtp1.md` (MTP).
