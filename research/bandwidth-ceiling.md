# WF-2: GPU memory-bandwidth ceiling on gfx1151 (Strix Halo) and the techniques that reach it

Date: 2026-10-29. Machine: Ryzen AI Max+ 395 / Radeon 8060S (gfx1151, 20 WGP = 40 CU), 128 GB, ROCm 7.2.4, kernel 7.0.0-38,
cmdline `amdgpu.gttsize=126976 ttm.pages_limit=32505856`. Benchmark was run on this machine (source and raw output below).

## Answer

1. **Achievable GPU streaming-read bandwidth on this machine is ~250 GB/s** (measured: 246-252 GB/s sustained, 4 GiB working set, 20 reps,
   best configs peak ~257-260 GB/s on 64 MiB). Maya's ~165 GB/s expert decode is therefore **~66% of the measured ceiling**; the ceiling is
   ~1.5x what the kernels reach. The "240 GB/s peak" quoted in docs/AMD_EXPERT_KERNELS.md:145-150 is itself slightly *below* what a trivial
   read kernel hits, so the 204-216 GB/s (85-90%) target in that doc is conservative: 85-90% of 250 is 212-225 GB/s.
2. **Theoretical peak is uncertain, 256 vs ~273 GB/s.** 256-bit LPDDR5X-8000 = 256 GB/s is the spec/community figure. But `pp_dpm_mclk`
   on this box tops out at 1066 MHz (x8 = 8533 MT/s -> 273 GB/s) and we measure 250 GB/s, i.e. 98% of 256 but 92% of 273. Could not read
   DIMM speed (dmidecode needs root). Community reports of "~215-230 GB/s measured" (secondary sources, older stacks) are lower than what
   a well-formed kernel gets here. Treat 250 GB/s as the empirical ceiling regardless of which nominal peak is right.
3. **Allocation type does not matter for steady-state bandwidth.** hipMalloc (coarse), hipExtMalloc fine-grained, hipMallocManaged,
   hipHostMalloc (default and non-coherent), and malloc+hipHostRegister (GTT/zero-copy) all give the same ~246-252 GB/s at saturation. On
   UMA all of these resolve to the same system DRAM (note VRAM carve-out is only 512 MB; everything else is GTT). hipHostRegister'd
   pageable memory is slightly slower at low occupancy only (133 vs 175 GB/s at 20 blocks u128), converging by 80 blocks.
4. **Load width and occupancy are the real levers**, not allocation mode:
   - 128-bit (`uint4`, `global_load_b128`) loads saturate DRAM with only ~2 workgroups of 256 threads per WGP (40 blocks): 247 GB/s.
   - 64-bit loads need ~4 blocks/WGP (80 blocks): 239-250 GB/s.
   - 32-bit loads need >= 8 blocks/WGP (160+ blocks) and top out lower: ~233 GB/s (6% below the wide loads).
   - With too little parallelism (1 block/WGP) b128 reaches only 175 GB/s, b64 89, b32 44: bandwidth is latency x outstanding-bytes limited.
   Implication for Maya: expert kernels must keep >= ~16 wave32 (512 threads) per WGP resident, each with 16 B in flight per lane, and
   the per-row dequant ALU must not stall the issue of the next b128 loads (software-pipeline loads ahead of dequant/dot).
5. **MALL / Infinity Cache (32 MB) is a separate regime**: a 16 MiB buffer re-read repeatedly gives ~850 GB/s (default loads), 3.4x DRAM.
   Chips and Cheese measured "just under 1 TB/s" from Infinity Cache. Only matters for data that is re-read within one kernel
   sequence and fits in 32 MB (e.g. shared expert / attention weights across tokens in a batch, KV hot window, activations), not for
   streaming routed-expert weights which are far larger than 32 MB. Non-temporal loads (`__builtin_nontemporal_load`) bypass this: the
   16 MiB case drops to ~245-270 GB/s, i.e. NT loads do not benefit from MALL. NT loads give no gain on DRAM-resident data (<=1%
   noise) so use plain loads.
6. **Other community evidence** (secondary): decode on 8060S is described as "bandwidth wall dominated"; Vulkan (RADV) often beats
   llama.cpp HIP on tg; HIP gaps cited include packed-integer codegen (3.2x slower than RADV/ACO on gfx1151) and graph/stream dispatch
   overhead. See sources. This suggests Maya's remaining gap to 250 GB/s is likely kernel codegen / ALU / occupancy for quantized
   dequant, not memory-system limits.

## Measurement detail (run on this machine)

Kernel: grid-stride read-only reduction (XOR of loaded words to keep loads live), 256 threads/block (wave32), blocks = 20 WGP x mult,
20 timed launches after warm-up, hipEvent timing; GB = 1e9 bytes. Source: scratchpad `bw.hip` (not committed; ~50 lines, easy to recreate:
template over `unsigned/uint2/uint4`, optional `__builtin_nontemporal_load` via an ext_vector_type). No running server was touched.

hipMalloc, 4 GiB (GB/s), columns = element width of loads:

| blocks | u32 | u64 | u128 | u128 non-temporal |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 43.6 | 89.2 | 175.1 | 175.9 |
| 40 | 91.1 | 181.0 | 246.9 | 247.4 |
| 80 | 165.0 | 239.0 | 246.3 | 252.0 |
| 160 | 231.5 | 250.3 | 247.7 | 247.2 |
| 320 | 232.8 | 250.1 | 248.8 | 248.8 |
| 640 | 233.1 | 249.9 | 249.7 | 250.5 |

Allocation comparison at 640 blocks, u128 (GB/s): hipMalloc 249.7; hipExtMalloc fine-grained 250.0; hipMallocManaged 248.8;
hipHostMalloc default 249.6; hipHostMalloc non-coherent 249.5; malloc+hipHostRegister 249.3. All within noise.

Cache regime (hipMalloc, 640 blocks, u128, plain loads): 16 MiB 858 GB/s (MALL-resident); 64 MiB 256 GB/s (spills to DRAM);
4 GiB 250 GB/s. 16 MiB with non-temporal loads: 271 GB/s.

Caveats: single run per config (variation ~1-3%); no XNACK-on test; a trivial reduction has minimal ALU, so real quantized kernels
(IQ2_S/IQ3_XXS dequant) will land below this ceiling; other processes on the machine (idle at test time as far as checked) could perturb it;
CPU contention for DRAM was not tested (Chips and Cheese reports the GPU can "squeeze out" the CPU under heavy GPU bandwidth load).

## Comparison with Maya

- Maya expert decode ~165 GB/s effective (docs/AMD_EXPERT_KERNELS.md:116-150) vs 250 GB/s measured = 66%. Closing to 85-90% means
  ~212-225 GB/s, i.e. ~1.3-1.36x faster expert phase.
- The doc's own 20% latency reduction target (gate/up <=183 us) corresponds to ~200 GB/s, reachable in principle given the ceiling.
- Dense/shared (non-expert) weights smaller than 32 MB per layer-step may enjoy MALL hits above 250 GB/s only if re-read in a short window.

## Technique checklist (ordered by expected payoff; items 1-3 are measured here, rest are hypotheses to test)

1. Use 128-bit loads for weight streaming (measured: 6% above 32-bit at saturation; 4x at low occupancy).
2. Ensure occupancy: >= 2 blocks of 256 threads per WGP (>= 16 waves) with high VGPR budget awareness; check VGPR count of expert
   kernels (`--save-temps`/`-Rpass-analysis=kernel-resource-usage`). Low occupancy from heavy dequant register use is the likeliest cause
   of the 165 GB/s plateau.
3. Do not spend effort on allocation mode (coarse vs fine vs host) for bandwidth; keep hipMalloc/GTT as is. (Page-fault/migration
   behaviour for hipMallocManaged and first-touch cost not measured; only steady state.)
4. Prefetch / software-pipeline: issue next b128 loads before dequant of current (hypothesis).
5. Keep rows 128 B / 256 B aligned so each wave's load covers full 128 B cache lines (gfx1151 cacheline is 128 B per rocminfo).
6. Be careful with packed-integer dequant codegen on gfx1151 (LLVM issue cited by hipEngine post); inspect ISA for v_perm/dp4a lowering.
7. Fuse gate/up/down to keep activations in LDS/L2 and exploit the 32 MB MALL for the activation/intermediate traffic.

## Sources

Primary / first-party
- Local measurement, this machine (above). `rocminfo`, `/sys/class/drm/card*/device/pp_dpm_mclk` (400/800/1066 MHz), `rocm-smi --showmeminfo`
  (VRAM 512 MB, GTT 124 GiB).
- Chips and Cheese, "Strix Halo's Memory Subsystem: Tackling iGPU Challenges": https://chipsandcheese.com/p/strix-halos-memory-subsystem-tackling
  (32 MB Infinity Cache, "just under 1 TB/s" from it, 2 MB L2, 256 KB L1 per shader array, 256 GB/s theoretical, GPU can squeeze out CPU).
  Independent first-party microbenchmark article, not AMD itself.
- Chips and Cheese, Infinity Cache evaluation: https://chipsandcheese.com/p/evaluating-the-infinity-cache-in
- ROCm/HIP docs, GPU memory and allocation APIs: https://rocm.docs.amd.com/en/latest/conceptual/gpu-memory.html and
  https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/host_memory.html
  (fetched page excerpts did not include a gfx1151-specific bandwidth statement; AMD publishes no measured GPU bandwidth for this part
  that I could find. The AMD product spec gives only the LPDDR5X-8000 256-bit interface.)
- Repo: docs/AMD_EXPERT_KERNELS.md:116-150 (Maya 165 GB/s; 240 GB/s "supplied peak").

Secondary / community (numbers not independently verified; treat as context)
- Level1Techs Strix Halo LLM benchmark thread (claims ~215 GB/s max GPU measured): https://forum.level1techs.com/t/strix-halo-ryzen-ai-max-395-llm-benchmark-results/
- Framework community GPU performance tests: https://community.frame.work/t/amd-strix-halo-ryzen-ai-max-395-gpu-llm-performance-tests/72521
- hipEngine post (decode tok/s vs llama.cpp HIP/Vulkan; cites LLVM packed-int codegen gap and dispatch overhead on gfx1151):
  https://community.frame.work/t/hipengine-fast-native-qwen-3-6-inference-for-rdna3-including-strix-halo/82803
- visorcraft strix-halo-llm-perf ("bandwidth wall dominates", no GB/s numbers): https://github.com/visorcraft/strix-halo-llm-perf
- Phoronix ROCm 7.0 on Strix Halo: https://www.phoronix.com/review/amd-rocm-7-strix-halo/4
- llama.cpp ROCm performance discussion: https://github.com/ggml-org/llama.cpp/discussions/15021 ; Vulkan: https://github.com/ggml-org/llama.cpp/discussions/10879
- Chips and Cheese RDNA3 microbenchmarks (cache bandwidth methodology): https://chipsandcheese.com/p/microbenchmarking-amds-rdna-3-graphics-architecture

Not found: an AMD-published rocm-bandwidth-test result for gfx1151; rocm-bandwidth-test is not installed here. Load-width guidance for
RDNA3 from a first-party AMD document was not located; the load-width conclusions above rest on the local measurement.
