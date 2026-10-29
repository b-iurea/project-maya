"""Drive one engine for a decode profile: per prompt length, prefill once (GEN 1), then decode N tokens on the same
prompt (the saved slot reuses it), bracketing each decode with CLOCK_MONOTONIC marks for the kernel trace.

usage: python3 profile_decode.py <maya-root> <config.json> <out-dir> <mtp 0|1> [tokens]
Run it under rocprofv3; marks go to <out-dir>/marks-mtp<m>.json.
"""
import json, sys, time
from pathlib import Path

ROOT, CFG, OUT, MTP = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
NEW = int(sys.argv[5]) if len(sys.argv) > 5 else 128
LENS = [2048, 65536]
sys.path.insert(0, str(ROOT))
import maya as M                     # noqa: E402
import serve.server as SV            # noqa: E402
import strata_tokenizer as ST        # noqa: E402
import subprocess                    # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
cfg = json.loads(CFG.read_text())
M.select_build_backend(cfg.get("backend", "cuda"))
cfg["exe"] = str(M.bench_exe(cfg))
args = cfg["args"]
tp = Path(cfg.get("tokenizer") or Path(args[args.index("--glm-pack") + 1]) / "tokenizer")
tok = ST.Tokenizer.from_dir(tp)
fmt = M.bench_chat(M.config_family(cfg), tp)
text = "\n\n".join(p.read_text(encoding="utf-8", errors="replace")
                   for p in [ROOT / "README.md", ROOT / "CHANGELOG.md"] + sorted((ROOT / "docs").glob("*.md")))
doc = tok.encode(text)
Q = "\n\nSummarize the text above in detail, section by section."
env = SV.child_env(cfg) | {"MAYA_GLM_MTP_DECODE": MTP, "STRATA_GLM_MTP_DECODE": MTP, "MAYA_GLM_TIMING": "1"}
log_path = OUT / f"engine-mtp{MTP}.log"
marks = []
with open(log_path, "w") as log:
    p = subprocess.Popen([cfg["exe"], "--serve"] + SV.engine_args(cfg), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=log, text=True, env=env, cwd=cfg.get("cwd") or str(ROOT), bufsize=1)
    while not (line := p.stdout.readline()).startswith("READY"):
        if not line:
            sys.exit(f"engine stopped while loading, see {log_path}")
    multi = any(w.startswith("multi=") for w in line.split())

    def gen(ids, n):
        p.stdin.write(M.bench_gen_line(n, ids, multi))
        p.stdin.flush()
        while True:
            line = p.stdout.readline()
            if not line:
                sys.exit(f"engine stopped, see {log_path}")
            if (f := M.bench_reply(line, multi)) is not None:
                return f

    gen(tok.encode(fmt("Say hello in five languages."), parse_special=True), 32)    # warm-up
    for n in LENS:
        ids = tok.encode(fmt(tok.decode(doc[:n - 32]) + Q), parse_special=True)
        gen(ids, 1)                                       # prefill; the slot keeps it
        t0 = time.monotonic_ns()
        f = gen(ids, NEW)                                 # decode on the reused prompt
        t1 = time.monotonic_ns()
        marks.append({"len": n, "t0": t0, "t1": t1, "tokens": int(f[1]), "decode_ms": float(f[4]),
                      "prefill_ms": float(f[3])})
        print(json.dumps(marks[-1]), flush=True)
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=180)
(OUT / f"marks-mtp{MTP}.json").write_text(json.dumps(marks, indent=1))
