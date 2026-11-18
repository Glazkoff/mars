#!/usr/bin/env python3
"""M-B.2 -- score LLM-AggreFact claims with one verifier in two readings: the whole claim (A1) and the claim
decomposed into atomic facts (A2; minimum and mean over facts), plus the shuffled-document control on the test
split. One output row per benchmark row. Registered under M-B in PREREG.md.

Verifiers:
  factcg   b10_verifier_swap.FactCG, its own input contract (document cut to fit 2,048 tokens, claim intact),
           support = softmax[:,1]
  marsc    the deployed MARS-C coverage verifier (three seeds averaged) read in the precision role: premise =
           document in overlapping 380-word windows (stride 190), hypothesis = claim or fact; the head scores
           omission, so support = 1 - sigmoid(logit), maximum over windows, mean over seeds.
Facts come from the frozen Gemma decomposition (props shards keyed by doc_key "<split>:<row>"); a claim whose
sentences all decomposed to NONE keeps its whole-claim score (fallback=1).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def windows(text: str, window: int = 380, stride: int = 190) -> list[str]:
    w = text.split()
    if len(w) <= window:
        return [" ".join(w)]
    starts = list(range(0, len(w) - window + 1, stride))
    if starts[-1] + window < len(w):
        starts.append(len(w) - window)
    return [" ".join(w[s:s + window]) for s in starts]


class MarscWindowed:
    """The deployed coverage verifier, premise windowed, support = max over windows of (1 - P(omitted))."""
    name = "marsc"

    def __init__(self, ckpt_glob: str, batch: int, dev: str, window: int = 380, stride: int = 190):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        dirs = [d for d in sorted(glob.glob(ckpt_glob)) if Path(d, "config.json").exists()]
        assert dirs, f"no checkpoint matched {ckpt_glob!r}"
        self.tok = AutoTokenizer.from_pretrained(dirs[0])
        self.models = [AutoModelForSequenceClassification.from_pretrained(d).to(dev).eval() for d in dirs]
        assert all(getattr(m.config, "num_labels", 1) == 1 for m in self.models), "expected the 1-logit MARS-C head"
        self.batch, self.dev, self.window, self.stride = batch, dev, window, stride
        print(f"[mb] marsc: {len(dirs)} checkpoints", flush=True)

    def score(self, docs: list[str], hyps: list[str]) -> list[float]:
        items = []  # (pair index, window text, hypothesis)
        for i, (d, h) in enumerate(zip(docs, hyps)):
            for w in windows(d, self.window, self.stride):
                items.append((i, w, h))
        order = sorted(range(len(items)), key=lambda k: len(items[k][1]) + len(items[k][2]))
        best = np.full(len(docs), -1.0)
        with self.torch.inference_mode():
            for s in range(0, len(order), self.batch):
                idx = order[s:s + self.batch]
                enc = self.tok([items[k][1] for k in idx], [items[k][2] for k in idx], truncation="longest_first",
                               max_length=512, padding=True, return_tensors="pt").to(self.dev)
                sup = np.zeros(len(idx))
                for m in self.models:
                    lg = m(**enc).logits.float().squeeze(-1)
                    sup += (1.0 - self.torch.sigmoid(lg)).cpu().numpy()
                sup /= len(self.models)
                for k, v in zip(idx, sup):
                    i = items[k][0]
                    if v > best[i]:
                        best[i] = v
        assert (best >= 0).all()
        return best.tolist()


def scored_sorted(scorer, docs: list[str], hyps: list[str], chunk: int = 4096) -> list[float]:
    """Length-sorted scoring (padding to the longest in a batch is what makes FactCG slow), order restored."""
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]) + len(hyps[i]))
    out = [0.0] * len(docs); t0 = time.time()
    for s in range(0, len(order), chunk):
        idx = order[s:s + chunk]
        vals = scorer.score([docs[i] for i in idx], [hyps[i] for i in idx])
        for i, v in zip(idx, vals):
            out[i] = float(v)
        print(f"[mb]   {min(s + chunk, len(order))}/{len(order)} pairs ({time.time() - t0:.0f}s)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggrefact-dir", required=True)
    ap.add_argument("--props", nargs="+", required=True, help="decomposition shards (globs)")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--verifier", choices=["factcg", "marsc"], required=True)
    ap.add_argument("--snapshot", default="", help="FactCG snapshot dir")
    ap.add_argument("--ckpts", default="", help="MARS-C checkpoint glob (.../model)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--shuffle-seed", type=int, default=20260917)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/k: score rows with index %% k == i (the shuffle control is drawn over the whole split)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sh_i, sh_k = (int(x) for x in a.shard.split("/"))
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet(Path(a.aggrefact_dir) / f"{a.split}-00000-of-00001.parquet")
    if a.limit:
        df = df.iloc[:a.limit]
    docs, claims, labels, dsets = df["doc"].astype(str).tolist(), df["claim"].astype(str).tolist(), df["label"].astype(int).tolist(), df["dataset"].astype(str).tolist()
    n = len(df)
    facts: dict[tuple[int, int], list[str]] = {}
    for pat in a.props:
        for f in sorted(glob.glob(pat)):
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                sp, row, sent = r["task_id"].split(":")
                if sp != a.split:
                    continue
                facts[(int(row), int(sent))] = [p for p in r.get("props", []) if p and p.strip().upper() != "NONE"]
    by_row: dict[int, list[tuple[int, list[str]]]] = defaultdict(list)
    for (row, sent), ps in facts.items():
        by_row[row].append((sent, ps))
    per_row = [[p for _, ps in sorted(by_row.get(i, [])) for p in ps] for i in range(n)]
    n_fact_rows = sum(1 for fs in per_row if fs)
    print(f"[mb] {a.split}: {n} rows, {n_fact_rows} with facts ({sum(len(f) for f in per_row)} facts), {n - n_fact_rows} fall back to the whole claim", flush=True)

    if a.verifier == "factcg":
        swap = _load(REPO / "research" / "marsc_strengthen_20260918" / "code" / "b10_verifier_swap.py")
        scorer = swap.FactCG(a.snapshot, a.batch, dev)
    else:
        scorer = MarscWindowed(a.ckpts, a.batch, dev)

    keep = [i for i in range(n) if i % sh_k == sh_i]
    print(f"[mb] shard {sh_i}/{sh_k}: {len(keep)} rows", flush=True)
    print("[mb] A1 whole claim", flush=True)
    whole_k = scored_sorted(scorer, [docs[i] for i in keep], [claims[i] for i in keep])
    whole = [None] * n
    for i, v in zip(keep, whole_k):
        whole[i] = v
    print("[mb] A2 decomposed", flush=True)
    pair_docs, pair_facts, pair_row = [], [], []
    for i in keep:
        for p in per_row[i]:
            pair_docs.append(docs[i]); pair_facts.append(p); pair_row.append(i)
    fact_scores = scored_sorted(scorer, pair_docs, pair_facts) if pair_docs else []
    agg_min, agg_mean = [None] * n, [None] * n
    acc: dict[int, list[float]] = defaultdict(list)
    for i, v in zip(pair_row, fact_scores):
        acc[i].append(v)
    for i in range(n):
        if acc.get(i):
            agg_min[i] = float(min(acc[i])); agg_mean[i] = float(np.mean(acc[i]))
    shuf_whole = shuf_min = None
    if a.split == "test":
        print("[mb] shuffled-document control (derangement within dataset)", flush=True)
        rng = random.Random(a.shuffle_seed)
        perm = list(range(n))
        for ds in sorted(set(dsets)):
            idx = [i for i in range(n) if dsets[i] == ds]
            if len(idx) < 2:
                continue
            for _ in range(100):
                sh = idx[:]; rng.shuffle(sh)
                if all(x != y for x, y in zip(idx, sh)):
                    break
            for x, y in zip(idx, sh):
                perm[x] = y
        sdocs = [docs[perm[i]] for i in range(n)]
        sw = scored_sorted(scorer, [sdocs[i] for i in keep], [claims[i] for i in keep])
        shuf_whole = [None] * n
        for i, v in zip(keep, sw):
            shuf_whole[i] = v
        sfs = scored_sorted(scorer, [sdocs[i] for i in pair_row], pair_facts) if pair_docs else []
        acc2: dict[int, list[float]] = defaultdict(list)
        for i, v in zip(pair_row, sfs):
            acc2[i].append(v)
        shuf_min = [float(min(acc2[i])) if acc2.get(i) else None for i in range(n)]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for i in keep:
            r = {"split": a.split, "row": i, "dataset": dsets[i], "label": labels[i], "n_facts": len(per_row[i]),
                 "fallback": int(not per_row[i]), "whole": whole[i],
                 "decomp_min": agg_min[i] if agg_min[i] is not None else whole[i],
                 "decomp_mean": agg_mean[i] if agg_mean[i] is not None else whole[i]}
            if shuf_whole is not None:
                r["whole_shuf"] = shuf_whole[i]
                r["decomp_min_shuf"] = shuf_min[i] if shuf_min[i] is not None else shuf_whole[i]
            fh.write(json.dumps(r) + "\n")
    print(f"[mb] written {out}", flush=True)


if __name__ == "__main__":
    main()
