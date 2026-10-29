# Gufo vs Maya prefill on Strix Halo (gfx1151)

Research only; nothing was run on the GPU. Gufo read from a shallow clone of `gufo-org/gufo` (default branch, October 2026 docs).
Maya read at commit `52e9131`. Gufo paths below are relative to its repo root
(https://github.com/gufo-org/gufo/blob/main/<path>).

## Answer

**1. What gufo reports.** README "Models and benchmarks" table and `docs/models/*/BENCHMARKS.md`. Gufo has no GLM support. All figures are Strix Halo 128 GB, pp2048 unless noted, one warmed sample per point.

| Model (gufo doc) | Quant | Total / active | Gufo pp (tok/s) | Baseline engine pp |
|---|---|---|---|---|
| DeepSeek V4 Flash (`docs/models/deepseek-v4-flash/BENCHMARKS.md`) | antirez Flash 0731 IQ2_XXS experts (Q2_K down, Q8 attention/shared) | 284B / ~13B (see note) | 484.6 (d0), 399.8 (d4K), 368 (d8K), 370 (d32K), 321 (d64K), 280 (d128K) | antirez/ds4: 248 (d0), 177 (d64K) |
| Qwen3.8 Flash-Next (`docs/models/qwen3.8-flash-next/BENCHMARKS.md`) | UD-Q4_K_XL | 512 experts, 10 used, ff 640, n_embd 2560 (`src/models/qwen38_flash_next/config.hpp`); about 3B active (my estimate) | 1801 (d0), 1707 (d16K), 1603 (d64K) | llama.cpp: 490 / 222 |
| Qwen3.8 27B dense (`docs/models/qwen3.8-27b/BENCHMARKS.md`) | UD-Q4_K_XL | 27B / 27B | 649 | llama.cpp (not quoted here) |

Note on DS4 active params: 43 layers, n_embd 4096, expert ff 2048, 256 experts (`src/models/deepseek_v4_flash/runtime/model_data_internal.h`). The 13B figure is the model's published number, not stated in the gufo repo; the geometry (6 routed experts x 3 x 4096 x 2048 x 43 layers = 6.5B routed) is consistent with it.

**Like-for-like with Maya? Only partly.** The only gufo model close to GLM-5.3-Flash is DS4 Flash. It has the same expert shape as GLM (E=4096, FF=2048; Maya already notes this at `src/prefill/moe_fused_rdna4.cu:7`) and the same 2-bit IQ class of quant, but fewer active routed experts (6 vs GLM's top-8) and fewer active parameters.

Normalization (tok/s x active params x 2 = effective TFLOP/s):

| Run | tok/s | Active | TFLOP/s |
|---|---|---|---|
| Maya GLM-5.3-Flash Maya-S (2K-64K) | 275-298 | 18B | 9.9-10.7 |
| Gufo DS4 d0 | 485 | 13B | 12.6 |
| Gufo DS4 d64K | 321 | 13B | 8.3 |
| Gufo Flash-Next | 1800 | about 3B | about 10.8 |
| Gufo Qwen27B dense Q4 | 649 | 27B | 35.0 |

Routed-expert work per token only (8 x 42 layers for GLM vs 6 x 43 for DS4) gives a ratio of about 1.3. Findings:
- At short context, gufo DS4 normalized to GLM size is about 485/1.3-1.4 = 350-370 tok/s vs Maya 275-298. That is a real gap, roughly 1.2-1.3x, not the raw 1.7x.
- At 64K, gufo DS4 degrades (321, about 235-250 GLM-equivalent) while Maya stays flat at about 290 (GLM has 34 linear-attention KDA layers and only 11 full MLA layers, `docs/GLM5-FLASH.md:26-30`). Maya is ahead at long context.
- Gufo's MoE paths all land at about 10-13 TFLOP/s effective. Only its dense W8A8 GEMM reaches about 35. So even gufo's MoE prefill is far below its dense-GEMM ceiling; the headroom is in expert-bucket efficiency, not in a secret kernel.
- Maya has its own DS4 measurement on the same 8060S (`docs/SETTINGS.md`, section "DeepSeek: routed prompt experts on AMD", `MAYA_DS4_PREFILL_FUSED`): 234 -> 257 tok/s at 2K and 430 -> 468 at 8K (GSQ-RCO IQ1_M/IQ2_XXS/IQ2_XS V4.1 GGUF, different quant mix and a different measurement from gufo's pp2048-at-depth). Gufo's own baseline antirez/ds4 is 248 at 2K, the same as Maya's pre-fusion 234. So Maya's DS4 at 2K is about 1.9x below gufo's, but at 8K the two are close (Maya 468 whole-prompt vs gufo 368 for a 2048-token slice at 8K depth; not the same metric). The 2K gap is the part worth chasing. Quant mixes differ, so treat this as indicative only.

**Net verdict.** The "much higher prefill" claim is mostly a model-size and architecture effect. The residual, real gap is about 1.2-1.3x at 2K-8K and sits in short-chunk MoE efficiency and dense/attention overhead. There is no single missing kernel class: Maya already does int8 WMMA with on-the-fly IQ decode, as gufo's DS4 gate/up does.

## 2. Why gufo's prefill is fast (source)

- Routed IQ2 gate/up uses llama.cpp's MMQ port (MIT) with int8 WMMA and Q8_1 activations: `src/models/deepseek_v4_flash/kernels/rocm/mmq/mmq.hip.hpp`, `mma.hip.hpp` (`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`, RDNA3 branch), entry `ds4_mmq_iq2_xxs_moe_pair_token_bound_swiglu` in `mmq/ds4_mmq.hip.cpp`. It fuses gate and up as a pair and fuses SwiGLU, weighting and the F16 mid mirror into the epilogue (call site `kernels/rocm/detail/ds4_rocm_moe_launch.hip.hpp`, about lines 760-860).
- Routed-bucket geometry tuning, the best-documented lever. `ds4_mmq_set_routed_max_expert_rows` bounds the column-tile grid by the real largest expert bucket: at a 4096-token chunk and 256 experts about 96% of the naive grid is empty blocks that still reserve 31.5 KiB LDS and cap residency at 2 workgroups per CU (`mmq/ds4_mmq.h`). `ds4_mmq_routed_tile_cols_for_counts` picks the column-tile width from the real count distribution with a cost model sum_e ceil(c_e/w)*w. Pinned-width sweep, tok/s at pp256/512/1024/2048: width 16 -> 261/354/427/472; 48 -> 241/347/436/499; 80 -> 220/321/420/493. The default 80 loses 14% at pp256 and 10% at pp512; 48 wins +3.4% at 1K and +1.1% at 2K (comment in `ds4_rocm_moe_launch.hip.hpp`).
- Q2_K down projection runs on F16 WMMA over hot experts only (count >= threshold), with a staged dequantizer: the raw 84-byte Q2_K slab is staged once in LDS as uint32 words and dequantized from LDS, mid tiles are double-buffered, there is a wide-N variant (2 or 4 row fragments) and a tile map that packs 128-row groups then 64-row tails (`kernels/rocm/detail/ds4_rocm_q2_down.hip.hpp`, `ds4_rocm_moe_launch.hip.hpp` about lines 290-380). Cold experts with few rows use scalar/small-tile kernels.
- Packed IQ2 weights are reused across short token tiles, and 8-row gate/up tiles use constant pair indices (9-10% faster tile kernel; `docs/models/deepseek-v4-flash/EXPERIMENTS.md`).
- Chunking: prompts over 4096 tokens run in 4096-token chunks; an 8K capacity regressed the 8K prompt (201.8 vs 210.2 tok/s), so it is policy, not a tunable (`runtime/rocm_graph.cpp:42`).
- Dense Q8 projections: persistent F16 copy of the Q8 weights (`hip_q8_f16_ptr`, `kernels/rocm/detail/ds4_rocm_runtime.hip.hpp:478`) so there is no per-chunk dequant; hipBLASLt for F16 GEMMs with a tuned plan DB (`src/core/hip/hipblaslt_plan_database.cpp`, `tune_hipblaslt` per `docs/PERFORMANCE.md`); one hand-written rocWMMA kernel for the attention-output-B shape (m=4096, k=8192, 256x128 macro tile; 15.82 -> 11.80 ms vs hipBLASLt MT96x96x32, which re-reads panels 5.7 GB vs 3.2 GB; `ds4_rocm_matmul.hip.hpp:1-40`). Other shapes stay on Tensile, which wins every opA=T shape by 1.4-2.6x in their harness.
- Attention (DS4-specific compressed/indexed): single-pass 32-head rocWMMA kernel with Q staged transposed so the score GEMM has both operands row-major (-31% on the indexed shape, -21% on the mixed window), exact partial top-k for the indexer (`ds4_rocm_attention_launch.hip.hpp`). Flash-Next sparse-attention experiments (LDS-bound, 65-78% active, 12-19% bank conflicts) are in `docs/models/qwen3.8-flash-next/EXPERIMENTS.md`.
- Dense Qwen W8A8 path: Q8_1 activation quantization fused with residual and RMSNorm in one pass (`src/models/qwen/hip/kernels/prefill_quant_gemm.hip`; 221 -> 137 MB traffic per norm at batch 2048), K-quant weights dequantized in-tile into int8 WMMA at 128x128 / 128x64 tiles, SwiGLU fused with the next quantization. This is how a 27B dense model reaches about 35 TFLOP/s.
- Process (`docs/PERFORMANCE.md`): kernel-resource check that fails CI on scratch/spill, roofline calibration (`tools/bench/gfx1151_peak`), "size the working set like the model does" (the 32 MB MALL hides repeated-shape benchmarks), and the finding that `__builtin_nontemporal_load` costs more than half the bandwidth on streaming kernels.

## 3. Maya's prefill path today (read, not run)

- Routed experts: `src/prefill/moe_fused_rdna4.cu` (adapted from Niko1221/Strata, MIT). `v_wmma_i32_16x16x16_iu8`, IQ weights decoded to int8 in LDS (double-buffered), int8 activations with per-32 scales, 64-row populated-tile enumeration (`make_tiles_kernel`), 256 threads, 128-row work items, gate/up + clipped SwiGLU + H quantization fused, down writes FP32. Default on gfx1151 (`MAYA_GLM_PREFILL_FUSED`, +13/+7.5/+5% at 4K/8K/16K, `docs/SETTINGS.md:81`). Fallback `src/prefill/moe_mmq.cu` (llama.cpp MMQ).
- Dense projections: `src/prefill/gemm.cu` dequantizes weights to an F16 scratch for each GEMM (row-sliced if too large), then hipBLAS/hipBLASLt (`Gemm::native`, lines 395-412), with tuned solution IDs in `tools/hip/gfx1151-glm-hipblaslt-100500.txt` (84 F16 shapes).
- Attention: MLA `mla_attn_wmma2_kernel` in `src/kernels/cuda/glm_batch.cu:1781`, default on gfx1151, measured only +4% on prompts (`docs/SETTINGS.md:142`).
- Sub-batching: `MAYA_GLM_PREFILL_SUB=1024`, `MAYA_GLM_PREFILL_MB=4096` (`docs/AMD_MAYA.md`, "Prompt speed"): 116 -> 195 (tuned hipBLASLt) -> 217 tok/s in that table; Windows Gorgon Halo 271-278 at 3K/8K.

Structural differences that matter: (a) Maya's expert kernel is already the gufo class, so gains are in bucket geometry and chunk size, not a rewrite; (b) Maya dequantizes dense weights per GEMM while gufo keeps a persistent F16 copy and quantizes activations once per norm; (c) I found no routed-count-aware tile-width or grid bounding in Maya (fixed `kTileRows = 64`, `NW_ROWS = 128`); (d) Maya's chunk size follows a memory budget (`MAYA_GLM_PREFILL_MB`), gufo's a measured policy (4096).

## 4. Transferable techniques

Licensing: gufo is MIT (`LICENSE`, "Copyright (c) 2026 gufo contributors"). Maya's `LICENSE` is MIT (Peasant Smith and Project Maya contributors; Niko1221 and Strata contributors). Compatible, provided the copyright and permission notice travel with any copied code. Gufo's DS4 half derives from antirez/ds4 (`src/models/deepseek_v4_flash/LICENSE.ds4`, `UPSTREAM.md`) and its MMQ files are vendored llama.cpp (MIT); keep those notices if copying. Gufo's Qwen/MiniMax licenses cover model weights/code under those names, not the DS4 kernels. Check file headers and `THIRD_PARTY_NOTICES.md` in gufo before copying any file verbatim; reimplementing the ideas needs no attribution.

Gain estimates are my judgment from the evidence above, not measurements; none were run.

| # | Technique | Where in Maya | Est. gain | Confidence | License |
|---|---|---|---|---|---|
| 1 | Routed-count-aware tile geometry: read per-expert counts, bound the tile grid by the real max bucket, choose tile width from a cost model sum_e ceil(c_e/w)*w. Gufo's sweep: 10-14% at 256-512 token chunks, 1-3% at 1-2K, about 0 at 4K. | `src/prefill/moe_fused_rdna4.cu` (`make_tiles_kernel`, `kTileRows`, `NW_ROWS`; add a 32/48-row variant for sub-128-row buckets). GLM mean bucket at a 1024-token chunk is about 28 rows over 288 experts, so a 64-row tile is mostly padding. | +3-8% at short prompts and small last chunks, 0-2% at 4K chunks | Medium (proven in gufo DS4; Maya's tile enumeration already skips empty tiles, so part is banked) | Idea free; code MIT, keep notice |
| 2 | Larger prefill chunk with a measured policy (gufo: 4096; 8192 regressed). Each chunk re-reads the whole expert set; bigger chunks fill buckets. | Chunk sizing in `src/prefill/prefill.cpp` (chunk buffers near lines 264/512) and `src/core/glm_prefill.cu`. | 0-10% at 2K-8K | Low-Medium; needs a 2048/4096/8192 sweep on the box. Cheap. | n/a |
| 3 | Persistent F16 copy of hot dense projections (attention q/kv/out, shared expert, dense layers 0-2) so `Gemm::native` stops dequantizing each GEMM. | `src/prefill/gemm.cu` (`Gemm::native`). Needs a memory budget, available only after the roughly 80 GiB expert pool. | +2-5% | Low-Medium; depends on the `dequant_f16` time share, which I did not profile | Idea free |
| 4 | Fuse activation quantization with RMSNorm/residual (and SwiGLU with the next quant) so FP32 normed rows and staging copies never hit memory. | `src/prefill/kernels.cu`, `quant_rows_nat_kernel` in `moe_fused_rdna4.cu`. | +1-3% | Low-Medium (gufo gives traffic numbers, not end-to-end) | MIT, notice if copied |
| 5 | Custom 256x128 rocWMMA macro tile for the few large-panel GEMMs where hipBLASLt re-reads dominate; keep Tensile elsewhere. Use gufo's methodology: three rotating operand copies to defeat the 32 MB MALL (`rocm/tools/f16_gemm_bench.cpp`). | `src/prefill/gemm.cu` and the table `tools/hip/gfx1151-glm-hipblaslt-100500.txt` (candidates with large K/N, e.g. 4096x8192x4096, 16384x1536x16384). | +1-3% overall (up to -25% on that one GEMM) | Low; the win was for DS4's m=4096, k=8192 shape; Maya's tuner already picks per-shape Tensile solutions | MIT, notice if copied |
| 6 | Measurement discipline: working sets larger than the MALL, kernel-resource CI check for spills (`tools/ci/check-kernel-resources.py`), roofline calibration (`tools/bench/gfx1151_peak.hip`). | `tools/hip/`, CI. | Indirect | High that it avoids false wins | MIT |
| 7 | Pair-fused gate/up with fused SwiGLU/weight/mirror in the epilogue. | Already in Maya's fused GU kernel. Not a gap. | 0 | High | n/a |
| 8 | MLA attention operand layout (transpose Q so the score GEMM is row-major on both sides; ablate QK/PV/softmax shares). | `mla_attn_wmma2_kernel`, `src/kernels/cuda/glm_batch.cu:1781`. Gufo gained 21-31% in its own DS4 kernel. | +1-4% overall (wmma2 gave only +4% in total, so attention is not dominant at 2K-8K; at long context KDA dominates and gufo has nothing for it) | Low | Idea only |

Not transferable: DS4 compressed/indexed attention, Q2_K F16 hot-list down kernels (GLM down is IQ2_S/IQ3_XXS and Maya already has an int8 down), DSpark/MTP (decode side), the RDMA variant.

## Caveats and unknowns

- Gufo's numbers are one warmed sample per cell, with a different harness, a "pp2048 at depth d" definition and a different quant mix from Maya's 2K-64K whole-prompt runs. The normalization (active params x 2 x tok/s) ignores attention FLOPs and memory traffic, which differ a lot between DS4, GLM and Qwen.
- Maya's per-stage time breakdown (experts / dense GEMM / KDA / MLA / quant / staging) was not available without GPU profiling. Re-rank items 1-5 once a rocprofv3 trace exists. If experts are well under about 50% of prefill time, items 1 and 7 have less room than listed.
- On this APU the full expert pool sits in unified memory, so the staging ring in `src/prefill/prefill.cpp` should not matter; confirm no staging copies remain in the prefill hot path.
