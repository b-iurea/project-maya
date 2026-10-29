"""Strix Halo GLM baseline: plain vs effective decode at 2K / 32K / 64K prompts, greedy, 3 runs each.

Drives the engine directly the way `maya.sh --bench` does (GEN lines over stdin), one engine start per MTP mode.
usage: python3 bench_baseline.py <maya-root> <config.json> <out-dir> [runs]
"""
import json, os, platform, re, subprocess, sys, time
from pathlib import Path

ROOT, CFG, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
RUNS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
LENS = [int(x) for x in os.environ.get("BENCH_LENS", "2048,32768,65536").split(",")]
EXTRA_ENV = json.loads(os.environ.get("BENCH_ENV", "{}"))
MODES = os.environ.get("BENCH_MTP", "0,1").split(",")
NEW = 256
# One document per length, a different question per run: the engine's saved conversation slot reuses the document's
# prefill, so a long prompt is read once (run 0's prefill is the measurement; later runs show the reuse).
QUESTIONS = ["\n\nSummarize the text above in detail, section by section.",
             "\n\nList every configuration setting the text above mentions and explain each one.",
             "\n\nWrite a critical review of the text above: what is unclear, missing or contradictory?"]
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
import maya as M                     # noqa: E402  (its bench helpers: chat format, GEN line, DONE parsing)
import serve.server as SV            # noqa: E402
import strata_tokenizer as ST        # noqa: E402


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:           # noqa: BLE001
        return f"? ({e})"


def system_state():
    return {
        "date": time.strftime("%Y-%m-%d %H:%M"), "kernel": platform.release(),
        "cmdline": sh("cat /proc/cmdline"),
        "rocm": sh("ls -d /opt/rocm-* | tail -1"),
        "gpu": sh("rocminfo | grep -m1 -o 'gfx1[0-9]*'"),
        "governor": sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "platform_profile": sh("cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo none"),
        "mclk_levels": sh("cat /sys/class/drm/card*/device/pp_dpm_mclk 2>/dev/null | tr '\\n' ' '"),
        "vram_carveout_mb": sh("cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null | head -1"),
        "thp": sh("cat /sys/kernel/mm/transparent_hugepage/enabled"),
        "mem": sh("free -g | sed -n 2p"),
        "git": sh(f"git -C {ROOT} rev-parse --short HEAD"),
    }


cfg = json.loads(CFG.read_text())
M.select_build_backend(cfg.get("backend", "cuda"))
if cfg.get("backend") == "hip":
    cfg["exe"] = str(M.bench_exe(cfg))
args = cfg["args"]
pack = Path(args[args.index("--glm-pack") + 1])
tp = Path(cfg.get("tokenizer") or pack / "tokenizer")
tok = ST.Tokenizer.from_dir(tp)
fmt = M.bench_chat(M.config_family(cfg), tp)
text = "\n\n".join(p.read_text(encoding="utf-8", errors="replace")
                   for p in [ROOT / "README.md", ROOT / "CHANGELOG.md"] + sorted((ROOT / "docs").glob("*.md")) +
                   sorted((ROOT / "bench" / "results").glob("*.md")) if p.exists())
doc = tok.encode(text)
while len(doc) < max(LENS) * 2:     # repeat the corpus if it is short
    doc = doc + doc
q_max = max(len(tok.encode(q)) for q in QUESTIONS)


def prompt(n, run):
    off = LENS.index(n) * 1237 % (len(doc) - n)
    body = tok.decode(doc[off:off + n - q_max - 16])
    return tok.encode(fmt(body + QUESTIONS[run % len(QUESTIONS)]), parse_special=True)


results, state = [], system_state()
(OUT / "system.json").write_text(json.dumps(state, indent=1))
for mtp in MODES:
    env = SV.child_env(cfg) | EXTRA_ENV | {"MAYA_GLM_MTP_DECODE": mtp, "STRATA_GLM_MTP_DECODE": mtp}
    log_path = OUT / f"engine-mtp{mtp}.log"
    t0 = time.time()
    with open(log_path, "w") as log:
        p = subprocess.Popen([cfg["exe"], "--serve"] + SV.engine_args(cfg), stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=log, text=True, env=env,
                             cwd=cfg.get("cwd") or str(ROOT), bufsize=1)
        multi = False
        while True:
            line = p.stdout.readline()
            if not line:
                sys.exit(f"engine stopped while loading, see {log_path}")
            if line.startswith("READY"):
                multi = any(w.startswith("multi=") for w in line.split())
                break
        print(f"mtp={mtp} loaded in {time.time() - t0:.0f} s", flush=True)

        def gen(ids, n_new):
            p.stdin.write(M.bench_gen_line(n_new, ids, multi))
            p.stdin.flush()
            while True:
                line = p.stdout.readline()
                if not line:
                    sys.exit(f"engine stopped, see {log_path}")
                f = M.bench_reply(line, multi)
                if f is not None:
                    return f

        gen(tok.encode(fmt("Say hello in five languages."), parse_special=True), 64)   # warm-up
        for n in LENS:
            for run in range(RUNS):
                ids = prompt(n, run)
                mark = log_path.stat().st_size
                def cpu_s():
                    st = open(f"/proc/{p.pid}/stat").read().rsplit(")", 1)[1].split()
                    return (int(st[11]) + int(st[12])) / os.sysconf("SC_CLK_TCK")
                c0 = cpu_s()
                f = gen(ids, NEW)
                cpu_used = cpu_s() - c0
                if f[0] != "DONE":
                    print("ERR", f, flush=True)
                    continue
                time.sleep(0.5)
                with open(log_path, errors="replace") as lf:
                    lf.seek(mark)
                    tail = lf.read()
                stat = re.findall(r"glm stat: .*", tail)
                acc = re.findall(r"glm spec: (\d+) of (\d+) drafts accepted", tail)
                ntok, pre_ms, dec_ms = int(f[1]), float(f[3]), float(f[4])
                r = {"mtp": int(mtp), "prompt": len(ids), "target": n, "run": run, "tokens": ntok,
                     "prefill_tps": len(ids) / pre_ms * 1000 if pre_ms else None,
                     "decode_tps": ntok / dec_ms * 1000 if dec_ms else None,
                     "glm_stat": stat[-1] if stat else None,
                     "accept": f"{acc[-1][0]}/{acc[-1][1]}" if acc else None,
                     "cpu_cores_busy": cpu_used / (dec_ms / 1000 + pre_ms / 1000) if dec_ms else None}
                results.append(r)
                print(json.dumps(r), flush=True)
                (OUT / "results.json").write_text(json.dumps(results, indent=1))
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        try:
            p.wait(timeout=180)
        except subprocess.TimeoutExpired:
            p.kill()

# summary table
rows = ["| decode | prompt | runs | decode tok/s (mean, min-max) | prefill tok/s (run 0) | tokens (min) | CPU cores busy |",
        "|---|---:|---:|---|---:|---:|---:|"]
for mtp in sorted({int(m) for m in MODES}):
    for n in LENS:
        rs = [r for r in results if r["mtp"] == mtp and r["target"] == n and r["decode_tps"]]
        if not rs:
            continue
        d = [r["decode_tps"] for r in rs]
        pf = [r["prefill_tps"] for r in rs if r["run"] == 0]   # later runs reuse the document's prefill
        rows.append(f"| {'effective (MTP on)' if mtp else 'plain (MTP off)'} | {n // 1024}K | {len(rs)} | "
                    f"{sum(d) / len(d):.1f} ({min(d):.1f}-{max(d):.1f}) | {sum(pf) / len(pf):.0f} | "
                    f"{min(r['tokens'] for r in rs)} | {sum(r['cpu_cores_busy'] for r in rs) / len(rs):.2f} |")
(OUT / "summary.md").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
