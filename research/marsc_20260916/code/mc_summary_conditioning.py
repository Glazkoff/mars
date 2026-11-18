#!/usr/bin/env python3
"""MARS-C R03a -- within-document, same-fact summary conditioning (FULL_REVIEW_2026-09-17 failure 4).

The review's objection: fact coverage has strong summary-blind predictability, and MARS-C trains on those
labels, so a system could rank the facts that are GENERALLY omitted without ever checking whether THIS
summary conveyed them. Neither the supplied-ACU AUROC nor recall@k separates the two, because both vary
the fact as well as the summary.

This test holds the fact fixed. RoSE labels every human ACU of a document against every summary of that same
document, so for a fact f there are naturally crossed cells: a summary s+ that CONVEYS f and a summary s- that
OMITS it. The verifier sees the identical hypothesis in both; only the premise changes.

  within-fact discrimination  =  P( z(f | s-) > z(f | s+) )   over all crossed (f, s+, s-) triples,
                                 ties counted as 1/2

A summary-blind scorer scores z(f | s-) = z(f | s+) for every triple and is pinned at exactly 0.500 -- it cannot
score above chance here no matter how good its prior over facts is. Reported per verifier and per seed, with a
document-level bootstrap and a document-level sign test, plus the same quantity restricted to the facts that
actually reach a system's top-10 emission (the population the paper's recall claim is about).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402


def clip(t: str, n: int) -> str:
    return " ".join(t.split()[:n])


def crossed_cells(docs: list[dict], excl=("gold",)) -> tuple[list[dict], list[tuple]]:
    """-> (cells to score, triples). A cell is (doc_index, summary_index, fact_index)."""
    cells, triples = [], []
    for di, d in enumerate(docs):
        summ = [(si, s) for si, s in enumerate(d["summaries"]) if s["system"] not in excl]
        for fi in range(len(d["facts"])):
            pos = [si for si, s in summ if s["labels"][fi] == 1]          # conveys the fact
            neg = [si for si, s in summ if s["labels"][fi] == 0]          # omits the fact
            if not pos or not neg:
                continue
            for sp in pos:
                for sn in neg:
                    triples.append((di, fi, sp, sn))
            for si in set(pos) | set(neg):
                cells.append((di, si, fi))
    return sorted(set(cells)), triples


def score_cells(model_dir: str, docs: list[dict], cells: list[tuple], batch: int, dev: str) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev).eval()
    two_class = getattr(model.config, "num_labels", 1) == 2                # b21 judge head is {omitted, covered}
    out = {}
    with torch.inference_mode():
        for s in range(0, len(cells), batch):
            ch = cells[s:s + batch]
            prem = [clip(docs[di]["summaries"][si]["candidate"], 380) for di, si, _ in ch]
            hyp = [clip(docs[di]["facts"][fi], 96) for di, _, fi in ch]
            enc = tok(prem, hyp, truncation="longest_first", max_length=512, padding=True, return_tensors="pt").to(dev)
            lg = model(**enc).logits.float()
            v = torch.softmax(lg, -1)[:, 0] if two_class else torch.sigmoid(lg.squeeze(-1))
            for c, x in zip(ch, v.tolist()):
                out[c] = x
    del model
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def discriminate(z: dict, triples: list[tuple]) -> tuple[list[float], dict[int, list[float]]]:
    """Per-triple win indicator (1 omitted-summary scored higher, 0.5 tie, 0 otherwise), and the same by document."""
    wins, by_doc = [], defaultdict(list)
    for di, fi, sp, sn in triples:
        a, b = z.get((di, sn, fi)), z.get((di, sp, fi))
        if a is None or b is None:
            continue
        w = 1.0 if a > b else (0.5 if a == b else 0.0)
        wins.append(w); by_doc[di].append(w)
    return wins, by_doc


def sign_test(diffs: list[float]) -> dict | None:
    pos = sum(1 for d in diffs if d > 0); neg = sum(1 for d in diffs if d < 0); n = pos + neg
    if n == 0:
        return None
    lo = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2.0 ** n)
    return {"n_docs": n, "n_pos": pos, "n_neg": neg, "p_value": float(min(1.0, 2.0 * tail))}


def summarise(wins, by_doc, rng, n_boot) -> dict:
    if not wins:
        return {"n_triples": 0}
    doc_means = {d: float(np.mean(v)) for d, v in by_doc.items()}
    keys = sorted(doc_means)
    draws = [float(np.mean([doc_means[k] for k in rng.choice(keys, size=len(keys), replace=True)])) for _ in range(n_boot)]
    return {"n_triples": len(wins), "n_docs": len(keys),
            "discrimination_triple": float(np.mean(wins)),
            "discrimination_doc_macro": float(np.mean([doc_means[k] for k in keys])),
            "ci_doc_bootstrap": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "sign_test_vs_half": sign_test([doc_means[k] - 0.5 for k in keys])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="e0 docs_<split>.jsonl")
    ap.add_argument("--verifiers", nargs="+", required=True, help="name=<glob of checkpoint dirs>")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    docs = common.load_jsonl(Path(a.docs))
    if a.limit_docs:
        docs = docs[:a.limit_docs]
    cells, triples = crossed_cells(docs)
    rng = np.random.default_rng(a.seed)
    rep = {"registered": "research/marsc_20260916/PREREG.md R03a (summary-conditioning identification)",
           "docs": len(docs), "cells": len(cells), "triples": len(triples),
           "docs_with_triples": len({t[0] for t in triples}),
           "facts_with_triples": len({(t[0], t[1]) for t in triples}),
           "chance_level": 0.5, "verifiers": {}}
    print(f"[r03a] {len(docs)} docs -> {len(cells)} crossed cells, {len(triples)} same-fact triples "
          f"over {rep['docs_with_triples']} docs", flush=True)
    for spec in a.verifiers:
        name, pat = spec.split("=", 1)
        ck = [d for d in sorted(glob.glob(pat)) if Path(d, "config.json").exists()]
        assert ck, f"verifier {name!r}: no checkpoint matched {pat!r}"
        per_seed, z_mean = {}, defaultdict(list)
        for d in ck:
            seed = m.group(1) if (m := re.search(r"seed(\d+)", d)) else Path(d).name
            z = score_cells(d, docs, cells, a.batch, dev)
            for c, v in z.items():
                z_mean[c].append(v)
            w, bd = discriminate(z, triples)
            per_seed[seed] = summarise(w, bd, np.random.default_rng(a.seed), a.n_boot)
            print(f"[r03a] {name} seed {seed}: discrimination {per_seed[seed]['discrimination_triple']:.4f}", flush=True)
        zc = {c: float(np.mean(v)) for c, v in z_mean.items()}
        w, bd = discriminate(zc, triples)
        cons = summarise(w, bd, rng, a.n_boot)
        rep["verifiers"][name] = {"checkpoints": ck, "per_seed": per_seed, "seed_averaged": cons}
        ci = cons["ci_doc_bootstrap"]
        print(f"[r03a] {name:22s} seed-averaged discrimination {cons['discrimination_triple']:.4f} "
              f"(doc macro {cons['discrimination_doc_macro']:.4f} [{ci[0]:.4f},{ci[1]:.4f}], "
              f"sign p={cons['sign_test_vs_half']['p_value']:.2e}) vs 0.5 chance", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "summary_conditioning.json").write_text(json.dumps(rep, indent=1))
    print(f"[r03a] written {out / 'summary_conditioning.json'}", flush=True)


if __name__ == "__main__":
    main()
