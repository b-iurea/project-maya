"""Distinct experts per MoE layer touched by windows of w consecutive decode tokens, from a STRATA_GLM_ROUTE_LOG file.

Each line: "layer e0 .. e7 fetch miss cpu | near..."; one line per (token, layer), layers in order per token.
usage: python3 union.py <route-log>
"""
import sys
from collections import defaultdict

seq = defaultdict(list)          # layer -> [expert set per token, in order]
for line in open(sys.argv[1]):
    head = line.split("|")[0].split()
    if len(head) < 9:
        continue
    layer, experts = int(head[0]), frozenset(map(int, head[1:9]))
    seq[layer].append(experts)

k = 8
n_tok = min(len(v) for v in seq.values())
print(f"{len(seq)} MoE layers, {n_tok} decode tokens per layer\n")
print("| window rows | distinct experts / layer (mean) | x one token | expert reads saved vs w separate tokens |")
print("|---:|---:|---:|---:|")
for w in (1, 2, 3, 4):
    tot, cnt = 0, 0
    for layer, sets in seq.items():
        for i in range(0, len(sets) - w + 1):
            u = frozenset().union(*sets[i:i + w])
            tot += len(u)
            cnt += 1
    mean = tot / cnt
    print(f"| {w} | {mean:.2f} | {mean / k:.2f}x | {100 * (1 - mean / (k * w)):.0f}% |")
