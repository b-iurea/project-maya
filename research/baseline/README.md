# Strix Halo GLM baseline (2026-10-29)

Reference benchmark for the wayfinder map "Make GLM-5.3-Flash fast on Strix Halo 128 GB".

- **Setup:** Maya-S-v2 IQ2_XXS, int8 KV, `--max-context 1048576`, greedy decoding, 256-token answers.
- **Prompts:** one document per length (2K / 32K / 64K) with 3 different questions. Run 0 measures prefill; runs 1–2 reuse the saved document state.
- **Engine and system:** commit cf6472c, ROCm 7.2.4, kernel 7.0.0-38. Full system state is in `system.json`.

Rerun: `python3 research/baseline/bench_baseline.py <maya-root> <config.json> <out-dir> 3` (about 20 min; nothing else may hold the GPU).

## Results

See `summary.md`. Per-run data, including the engine's `glm stat` line and MTP acceptance, is in `results.json`. Filtered engine logs are in `engine-mtp*.lines.txt`.

## Findings

- **Experts are reloaded from the SSD on Linux.** Every prompt borrows 6142 MB of the expert pool. Afterwards decode reads the evicted experts back from the SSD: 0.68–1.10 disk reads/token, costing 4.4–6.7 ms/token, which is about 8% of plain decode. The pool holds every expert (`all experts fit`), but `MAYA_GLM_COMPLETE_POOL` does not apply to Linux APUs (docs/SETTINGS.md:54).
- **Long context barely slows decode** at this length: plain decode drops only ~3% from 2K to 64K.
- **MTP gains ~+26%**, with acceptance between 68% and 83%.
- **System state:** the CPU governor is `powersave`, there is no platform_profile, IOMMU is on, and mclk tops out at 1066 MHz.
