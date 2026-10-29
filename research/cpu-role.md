# WF-5: Can the Zen 5 CPU speed up GLM decode on Strix Halo without stealing GPU bandwidth?

## Answer

Mostly no for bandwidth-bound work, yes for latency-bound and non-bandwidth work. Nothing found measures concurrent CPU+GPU bandwidth on Strix Halo, so the contention side is inference, not data.

- All 256 GB/s (LPDDR5X-8000, 256-bit) is one pool shared by CPU, GPU and NPU. The GPU alone measures ~215 GB/s; the CPU complex tops out around 124 GB/s for reads (both CCDs). The two sum to more than 256, so running both flat out must contend. There is no published split.
- Running MoE experts on the CPU (ktransformers / llama.cpp `--n-cpu-moe` style) only pays when the GPU cannot hold the experts (discrete VRAM). Here the GPU can address the whole 128 GB. The CPU reads the same bytes over a narrower path (~half the GPU's bandwidth), so a CPU expert slice adds less throughput than it takes from the GPU. Expect no win at best. No Strix Halo A/B of expert offload vs full-GPU residency was found, so measure before ruling it out.
- Roles that help (little bandwidth): host-side orchestration (launch issue, routing readback, sampling, tokenization, scheduling) kept off the critical path; speculative/MTP drafting on the GPU (Maya already gets +26-50% from it); latency- or compute-bound work such as grammar masks and tiny n-gram/lookup drafters.
- Roles that hurt: CPU expert slices on shared memory, and any CPU thread pool that spins or streams memory while GPU decode runs. Maya's CPU lane exists for discrete GPUs with experts in host RAM; on a Strix Halo with all experts GPU-resident it has nothing to compute (see below), so the right setting is lane off or minimal threads, leaving cores for the decode-issuing thread.

## Measurements found

| Claim | Source |
|---|---|
| 256-bit LPDDR5X-8000, "up to 256GB/s shared between all of the components" | Chips and Cheese, https://chipsandcheese.com/p/amds-chiplet-apu-an-overview-of-strix |
| Both CCDs: reads 124 GB/s; read-modify-add test pulls >175 GB/s | same |
| Per CCD: 32 B/cycle read and 32 B/cycle write links to the IO die at ~2000 MHz: reads cap at 64 GB/s, writes measured ~43 GB/s; single-CCD combined just over 103 GB/s (theory 128) | same |
| Read-only CPU traffic "still limited by cross-die bandwidth and falls far short of 256 GB/s" | Chips and Cheese, https://chipsandcheese.com/p/strix-halos-memory-subsystem-tackling |
| Contention (qualitative only): CPU latency baseline ~140 ns; ~200 ns in games; under heavy GPU load plus CPU read threads it passed 300 ns. GB/s split is in charts only; I could not read chart values | same |
| GPU: ~215 GB/s max GPU memory bandwidth of 256 theoretical | Framework community thread, https://community.frame.work/t/amd-strix-halo-ryzen-ai-max-395-gpu-llm-performance-tests/72521 (community post, not first-party) |
| Same thread, Mistral Nemo BF16 12.25B: CPU (16 threads) tg 4.97 t/s vs GPU ~9.1 t/s. My derivation (~24.5 GB weights x t/s): CPU ~122 GB/s, GPU ~223 GB/s effective, consistent with the Chips and Cheese read figure | same |
| Memory is mapped, not partitioned; GTT = system RAM mappable into GPU address space | AMD ROCm docs, https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html |

Inference (mine, not measured): CPU peak (~124) + GPU peak (~215) = ~340 GB/s of demand against a 256 GB/s bus, so both saturated forces a split. Decode is bandwidth-bound on both, so the GPU loses what the CPU gains, minus the CPU's lower efficiency.

## What other engines do on this chip

- llama.cpp: guidance for Strix Halo is to give the GPU the whole model (large GTT allocation) and skip `--n-cpu-moe` / `--cpu-moe`, which target discrete-VRAM hosts. Flag descriptions: https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide and https://llmhardware.io/guides/moe-hybrid-gpu-cpu-offloading-guide . The "not needed on Strix Halo" statement comes from a secondary guide, https://hal0-web.pages.dev/docs/hardware/strix-halo/ . Typical Strix Halo runs use `-ngl 99` with 6-8 threads and do not sweep thread count, so there is no published thread-count-vs-decode data.
- ktransformers: no Strix Halo results found. Its design (CPU expert kernels, GPU attention) assumes limited VRAM and separate DRAM; that premise does not hold on UMA.
- llama.cpp MTP/speculative decoding on Strix Halo has community benchmarks (not verified here). Maya's own numbers below are first-party.

## What Maya's CPU lane does today (repo grep)

- Purpose: computes routed experts that are neither in VRAM nor staged, from the pinned host-RAM tier, in parallel with the GPU and PCIe staging copies, chosen per expert by measured CPU-vs-PCIe cost (`src/mm3/mm3_model.cu` ~674-800; GLM engine `src/core/glm_fast_path.cu`; `docs/SETTINGS.md` rows `MAYA_GLM_CPU_LANE`, `MAYA_GLM_CPU_LANE_MAX`, `MAYA_GLM_CPU_CAL`, `MAYA_GLM_CPU_RETIME`, and the "CPUs the engine runs on" paragraph ~line 35).
- Default threads: one per physical core, reduced for NUMA/SMT hosts (`glm_fast_path.cu` ~2975-2992); `MAYA_GLM_CPU_LANE=0` disables. Idle workers spin after each batch (SETTINGS.md ~line 35).
- Not gated on AMD/UMA: `unified_memory` is set from `cudaDevAttrIntegrated` on Linux (`glm_fast_path.cu:636-638`); I found no code that disables the lane for it. It is on by default but idles when nothing misses.
- On UMA the engine treats the RAM tier as the GPU's memory (`full_unified_pool`, `warm_pool_slots`, `minimal_ram_tier(unified...)`, `glm_fast_path.cu` ~1509, 3714, 4995). `MAYA_GLM_COMPLETE_POOL` (SETTINGS.md ~line 54): where the GPU budget holds every expert (Ryzen AI Max, Maya-S, 8065S on Windows) "the decode never misses and the CPU lane has nothing to take"; decode +2.6-4% (+7% after an 8K prompt). Linux APUs "keep their unified-memory policy".
- Helper path: `MAYA_GLM_EXPERT_HELPER` lets an APU compute other GPUs' misses inside the pinned RAM tier "at the APU's memory speed instead of the part's CPU lane" (`docs/AMD_MAYA.md` ~140-165, SETTINGS.md ~line 49), zero-copy on integrated helpers. On UMA the iGPU, not the CPU, is the expert engine.
- Existing non-CPU wins on this chip: `MAYA_GLM_MTP_DECODE` (on for gfx115x) 256-token answers +44-50% on short prompts, +26-27% after 3-8K prompts; `MAYA_GLM_MV_RDNA` +3.1%; `MAYA_GLM_GU22_TUNED` 1.12x reading pinned host memory on the 8060S.
- CPU-kernel tuning (`MAYA_GLM_CPU_1T`, AVX-512 VBMI, Zen 4+) targets discrete-GPU hosts (Ryzen 7 9700X + RTX 3090: +6-8%), not the APU.

## Gaps / suggested measurements

1. Microbench on the target box: GPU-only stream kernel GB/s, CPU-only STREAM (1 and 2 CCDs), then both concurrently. This is the missing number.
2. A/B on GLM-5.3-Flash with all experts GPU-resident: `MAYA_GLM_CPU_LANE=0` vs default vs 2/4 threads; check the `glm fast: ... CPU lane` log line and the experts actually taken.
3. Check thread placement: the decode issue thread, service threads and spinning lane workers should not compete for cores.
4. Candidates: sampling/readback off the critical path; a CPU n-gram or prompt-lookup drafter (tiny bandwidth); no CPU expert slice unless the model exceeds the GPU-addressable GTT.
