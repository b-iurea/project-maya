# WF-3: Linux system settings that change Strix Halo LLM decode speed

Researched 2026-10-29. Target: Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), 128 GB LPDDR5X, Linux, ROCm 7.2.4.

## Answer

Decode is memory-bandwidth-bound (about 256 GB/s theoretical, about 215 GB/s measured per community spec sheets). Almost no system setting moves decode by more than a few percent. Most settings decide whether the model fits and runs stably, or move prefill.

1. **Capacity settings (gate everything).** BIOS UMA carve-out at the minimum (0.5 GB), with `ttm.pages_limit` (and `amdgpu.gttsize`) raised. AMD recommends this. Without it the GTT pool defaults to about 50% of RAM and a large model does not fit. This is a yes/no effect, not a speed effect.
2. **IOMMU off is the only setting with a measured speed delta.** `amd_iommu=off` gives about +2.3% on decode (tg128, 37.17 vs 36.32 t/s) and +5 to 12% on prefill, against both translated and `iommu=pt` modes (single-author benchmark, kyuz0 issue #66). Trade-off: NPU stops working and DMA isolation is lost. `iommu=pt` is no faster than the default.
3. **Stability gates.** Kernel at least 6.18.4 (or Ubuntu HWE 6.17.0-19 / OEM 6.14.0-1018) is required for ROCm on gfx1151. `linux-firmware-20251125` is a known-bad firmware. An unstable system hurts more than any tuning gains.
4. **Power/TDP: no measured decode data found.** No source sweeps 55/85/120 W with decode numbers. Reasoning (not measurement): decode is bandwidth-bound, so TDP matters mainly for prefill and for sustained thermal throttling. Needs a local A/B (ryzenadj or BIOS limit, then `--bench`).
5. **TuneD `accelerator-performance`, CPU governor, THP: plausible and low-cost, but no measured decode deltas.** Maya's RAM tier already uses `MADV_HUGEPAGE` (see `docs/SETTINGS.md`), so the `madvise` THP default is correct for Maya.

Expected ceiling for system tuning on this machine: single-digit percent on decode. The big wins are in the engine (see `docs/AMD_MAYA.md`: MTP decode raised 18.4-19.7 to 28.4-29.6 tok/s on Windows Gorgon Halo).

## This machine (read-only, nothing changed)

| Item | Value |
|---|---|
| CPU / GPU | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, gfx1151 |
| Kernel | 7.0.0-38-generic (Ubuntu) |
| Cmdline | `amdgpu.gttsize=126976 ttm.pages_limit=32505856 amdgpu.noretry=0 amdgpu.vm_update_mode=0 amdgpu.sg_display=0` (no `amd_iommu=` setting) |
| IOMMU | Enabled, default translated mode (`/sys/kernel/iommu_groups` populated) |
| Dedicated VRAM (BIOS UMA) | 512 MiB (`mem_info_vram_total` = 536870912), already at AMD's recommended minimum |
| GTT total | 133143986176 B (124 GiB) |
| `ttm.pages_limit` | 32505856 pages (124 GiB); `page_pool_size` 16081316 |
| THP | `always [madvise] never` |
| CPU governor | powersave (cpu0) |
| GPU DPM level | `manual` (`power_dpm_force_performance_level`) |
| platform_profile | not present (`/sys/firmware/acpi/platform_profile` missing) |
| tuned | not installed |
| ROCm / amdgpu packages | amdgpu-core `7.2.07020400` series (ROCm 7.2.4), linux-firmware 20260319 |
| RAM | 122 GiB visible |

Observation: capacity settings already follow best practice. Untested levers here: `amd_iommu=off`, TuneD / CPU governor, TDP.

## What Maya's docs already say

| Doc | Content |
|---|---|
| `docs/AMD_MAYA.md` (Windows section) | Give the GPU most memory via BIOS iGPU size or Variable Graphics Memory (160 of 192 GB; 96 GB on 128 GB). No Linux equivalent guidance. |
| `docs/AMD_MAYA.md` ~L181, L191 | Engine does not trust HIP-reported GTT (112 GiB on a 128 GB box). Measured 128 GB Strix Halo, ROCm 7.2.2, "manually tuned settings" (unspecified): about 210 tok/s prompt, 17.8 tok/s answers. |
| `docs/AMD_MAYA.md` v1.1.10 | MTP decode on gfx115x: 28.4-29.6 tok/s (Windows, Gorgon Halo). |
| `docs/SETTINGS.md` ~L1244-1300 | RAM tier uses `MADV_HUGEPAGE`; `defrag=defer` and `vm.nr_hugepages` notes. |
| `docs/SETUP.md` | Nothing on GTT/TTM/IOMMU/power. |

Gap: Maya docs have no Linux Strix Halo kernel-parameter guidance (cmdline, IOMMU, kernel/firmware minimums, TDP), and the "manually tuned settings" behind the 17.8 tok/s figure are not recorded.

## Per-setting table

| Setting | What it does | Effect on bandwidth / decode | Measured delta | Source |
|---|---|---|---|---|
| BIOS UMA carve-out | Firmware-reserved VRAM | Small (0.5 GB) recommended. Large reservation permanently removes RAM; GTT is dynamic. AMD says CPU and GPU share physical memory, so no discrete-style VRAM/RAM speed gap (a Framework forum user claims VRAM is faster than GTT; anecdotal). | None measured | [AMD ROCm Strix Halo optimization](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html); [AMD Ryzen install](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-ryzen.html); [Framework thread](https://community.frame.work/t/igpu-vram-how-much-can-be-assigned/73081) |
| `ttm.pages_limit` | Max pinned system pages for GPU (4 KB pages; 32505856 = 124 GiB) | Capacity only. Default GTT is about 50% of RAM. `amd-ttm --set N` writes `/etc/modprobe.d/ttm.conf`. | None (capacity) | [AMD ROCm Strix Halo optimization](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html); [kyuz0 toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) |
| `amdgpu.gttsize` | GTT size in MB (126976 = 124 GB) | Capacity only | None (capacity) | [strix-halo-toolboxes.com](https://strix-halo-toolboxes.com) |
| `amd_iommu=off` | Disable IOMMU translation | Lower DMA/translation overhead. Loses NPU and DMA protection. | tg128 +2.3%, pp512 +5.2%, pp16384 +6.1%, pp65536 +4.6%, multi-stream up to +10-12%. Qwen3-Coder-Next Q8, kernel 7.0.4, ROCm 7.2.3, Framework Desktop, single author. | [kyuz0 issue #66](https://github.com/kyuz0/amd-strix-halo-toolboxes/issues/66); [strix-halo-toolboxes.com](https://strix-halo-toolboxes.com) |
| `iommu=pt` | Passthrough | No better than default translated (identical within noise). `amd_iommu=pt` is not a valid value; use `iommu=pt`. One forum user reported HIP crashes with IOMMU off and suspected pt was needed; unconfirmed. | tg128 equal to translated (36.32) | [kyuz0 issue #66](https://github.com/kyuz0/amd-strix-halo-toolboxes/issues/66); [Framework thread](https://community.frame.work/t/llama-cpp-vllm-toolboxes-for-llm-inference-on-strix-halo/74916?page=2) |
| Power/TDP (45-120 W, ryzenadj, BIOS) | cTDP range 45-120 W; Framework Desktop allows more | Decode bandwidth-bound so small effect expected; prefill compute-bound so larger expected. Reasoning only. | No measured sweep found | [Framework forum](https://community.frame.work/t/amd-strix-halo-ryzen-ai-max-395-gpu-llm-performance-tests/72521) (context only); gap, test locally |
| ACPI platform profile | `platform_profile` sysfs | Absent on this machine, so no knob here | N/A | local sysfs |
| TuneD `accelerator-performance` | Disables high-latency CPU C-states | Recommended by toolbox author; mostly CPU side | None quantified | [strix-halo-toolboxes.com](https://strix-halo-toolboxes.com) |
| CPU governor / GPU DPM | powersave vs performance; DPM level | Not studied for decode; this machine is powersave / manual | None | local sysfs; gap |
| Transparent huge pages | 2 MB pages for host allocations | Helps pinned host (RAM-tier) TLB behavior. Maya's RAM tier already uses `MADV_HUGEPAGE` on the `madvise` default. | None for decode | `docs/SETTINGS.md` L1244-1300 |
| Kernel version | KFD fixes (CWSR, queue/memory limits) | Stability. Minimum: Ubuntu HWE 6.17.0-19.19~24.04.2, OEM 6.14.0-1018, others 6.18.4. ROCm 7.2.0 unsupported below 6.18.4. | Not speed; crashes without | [AMD ROCm Strix Halo optimization](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html) |
| linux-firmware | GPU/microcode | `20251125` breaks ROCm on Strix Halo (recalled); `20251111` recommended. This machine has 20260319 (newer than anything the sources list). | Not speed; instability | [kyuz0 firmware doc](https://github.com/kyuz0/amd-strix-halo-toolboxes/blob/main/docs/troubleshooting-firmware.md) |
| Kernel 6.19.11 | Reported gfx1151 hard lockups (user suspects TLB fence rework) | Stability, unconfirmed, anecdotal (Fedora bug report seen via search, not opened) | N/A | search result only; unverified |
| ROCm version | Runtime and kernels | 10.0/10.1 stable on supported kernels; 7.2.0 needs kernel 6.18.4+ | Not benchmarked | [AMD ROCm Strix Halo optimization](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html) |

## Caveats

- The IOMMU numbers come from one issue author on one model. The decode gain (2.3%) is small enough that noise could hide it on other builds. Verify locally with Maya `--bench`.
- Several search results (Level1Techs, Framework forum, blogs) report absolute tok/s without stating power or IOMMU state, so they were not used for deltas.
- No primary source (AMD docs, kernel docs) addresses power-mode effects on LLM decode for this chip.
- The kyuz0 repo and issue are community, not AMD; AMD's own page covers only memory and kernel minimums.

## Suggested local A/B tests (for the plan, not done here)

1. `amd_iommu=off` vs current, Maya `--bench`, decode and prefill (reboot required).
2. Power limit sweep (BIOS or ryzenadj) with sustained decode, watching clocks and temperatures.
3. TuneD `accelerator-performance` and CPU governor `performance` vs current.
