"""Per-token GPU time breakdown of decode windows from a rocprofv3 kernel trace.

usage: python3 analyze_trace.py <kernel_trace.csv> <marks.json> [--top N]
"""
import csv, json, re, sys
from collections import defaultdict

trace, marks = sys.argv[1], json.load(open(sys.argv[2]))
TOP = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25

CATS = [  # first match wins
    ("experts", r"iq2|iq3|iq1|gate_up|expert|moe|down<|ffn_exp"),
    ("attention", r"kda|dsa|mla|attn|attention|flash|hc_|_hc|softmax|rope|kv"),
    ("router", r"route|router|topk|top_k|select|gating"),
    ("dense gemv", r"mv_|gemv|q6_k|bf16|q8|matvec|dense|lm_head|logit"),
    ("norm/elementwise", r"norm|rms|add|mul|silu|swiglu|copy|scale|residual|fill|cast|convert"),
    ("sampling/mtp", r"sample|argmax|mtp|draft|verify|accept"),
]


def cat(name):
    n = name.lower()
    for c, rx in CATS:
        if re.search(rx, n):
            return c
    return "other"


rows = []
with open(trace) as f:
    for r in csv.DictReader(f):
        rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
rows.sort()

for m in marks:
    win = [(s, e, n) for s, e, n in rows if s >= m["t0"] and e <= m["t1"]]
    if not win:
        print(f"len {m['len']}: no kernels in window (clock mismatch?)")
        continue
    tok = m["tokens"]
    wall = (m["t1"] - m["t0"]) / 1e6
    busy, last_end = 0, None
    for s, e, _ in win:                       # union of busy intervals (kernels on one stream mostly)
        if last_end is None or s >= last_end:
            busy += e - s
            last_end = e
        elif e > last_end:
            busy += e - last_end
            last_end = e
    by_cat, by_name, cnt = defaultdict(int), defaultdict(int), defaultdict(int)
    for s, e, n in win:
        by_cat[cat(n)] += e - s
        short = re.sub(r"\((?!anonymous).*", "", n.replace("(anonymous namespace)::", "")).replace("strata::kernels::", "")[:110]
        by_name[short] += e - s
        cnt[short] += 1
    span = (win[-1][1] - win[0][0]) / 1e6
    print(f"\n### {m['len'] // 1024}K prompt: {tok} tokens, engine decode {m['decode_ms'] / tok:.2f} ms/token, "
          f"GPU span {span / tok:.2f} ms/token, GPU busy {busy / 1e6 / tok:.2f} ms/token "
          f"(idle {100 * (1 - busy / 1e6 / span):.1f}% of span), {len(win) / tok:.0f} kernels/token\n")
    print("| category | ms/token | share of busy |\n|---|---:|---:|")
    for c, v in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"| {c} | {v / 1e6 / tok:.2f} | {100 * v / busy:.1f}% |")
    print(f"\n| top kernels | ms/token | calls/token | category |\n|---|---:|---:|---|")
    for n, v in sorted(by_name.items(), key=lambda x: -x[1])[:TOP]:
        print(f"| `{n}` | {v / 1e6 / tok:.3f} | {cnt[n] / tok:.1f} | {cat(n)} |")
