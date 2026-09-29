#!/usr/bin/env python3
"""M-D.2 -- per-sentence supports on RAGTruth responses with one verifier: (a) the sentence as the claim (the
comparator's unit), (b) the sentence's atomic facts (the frozen decomposition), minimum kept, and (c) the
shuffled-context control for both. Uses the mars package's verifier contracts, so the numbers are the package's.
Output: one row per response with per-sentence lists. Queued design in PREREG.md (M-D).
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from mars.verifiers import PairVerifier, PromptVerifier  # noqa: E402


def load_props(patterns: list[str]) -> dict[str, dict[int, list[str]]]:
    by: dict[str, dict[int, list[str]]] = defaultdict(dict)
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                key, sent = r["doc_key"], int(r["task_id"].rsplit(":", 1)[1])
                by[key][sent] = [p for p in r.get("props", []) if p and p.strip().upper() != "NONE"]
    return by


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--props", nargs="+", required=True)
    ap.add_argument("--verifier", choices=["prompt", "pair"], required=True)
    ap.add_argument("--path", required=True, help="prompt: checkpoint dir; pair: glob of checkpoint dirs")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--shuffle-seed", type=int, default=20260917)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pairs = [json.loads(l) for l in open(a.pairs, encoding="utf-8")]
    if a.limit:
        pairs = pairs[:a.limit]
    props = load_props(a.props)
    ver = PromptVerifier(a.path, batch=a.batch) if a.verifier == "prompt" else PairVerifier.from_glob(a.path, batch=a.batch)
    n = len(pairs)
    # units: (pair index, sentence index, fact index or -1 for the whole sentence, text)
    units = []
    for i, p in enumerate(pairs):
        for j, (s, e) in enumerate(p["sentences"]):
            units.append((i, j, -1, p["output"][s:e].strip()))
            for k, f in enumerate(props.get(p["pair_id"], {}).get(j, [])):
                units.append((i, j, k, f))
    ctx = [pairs[i]["context"] for i, _, _, _ in units]
    sup = ver.support(ctx, [u[3] for u in units])
    rng = random.Random(a.shuffle_seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        groups[p["task_type"]].append(i)
    perm = list(range(n))
    for idx in groups.values():
        for _ in range(200):
            sh = idx[:]; rng.shuffle(sh)
            if all(x != y for x, y in zip(idx, sh)):
                break
        for x, y in zip(idx, sh):
            perm[x] = y
    sup_shuf = ver.support([pairs[perm[i]]["context"] for i, _, _, _ in units], [u[3] for u in units])
    rows = [{"pair_id": p["pair_id"], "task_type": p["task_type"], "model": p["model"], "n_sentences": len(p["sentences"]),
             "sentence_support": [None] * len(p["sentences"]), "fact_supports": [[] for _ in p["sentences"]],
             "sentence_support_shuf": [None] * len(p["sentences"]), "fact_supports_shuf": [[] for _ in p["sentences"]]} for p in pairs]
    for (i, j, k, _), v, vs in zip(units, sup, sup_shuf):
        if k < 0:
            rows[i]["sentence_support"][j] = v; rows[i]["sentence_support_shuf"][j] = vs
        else:
            rows[i]["fact_supports"][j].append(v); rows[i]["fact_supports_shuf"][j].append(vs)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[md] {a.verifier}: {n} responses, {len(units)} units scored -> {out}", flush=True)


if __name__ == "__main__":
    main()
