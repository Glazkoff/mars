#!/usr/bin/env python3
"""MARS-C R04 -- the label-source effect on the verifier's own endpoint, with document-paired intervals.

Reads the calibrated supplied-ACU scores of the matched twins, seed-averages each arm, and reports pooled and
document-macro AUROC against the human coverage labels plus a document bootstrap of every pairwise difference.
Because the evaluation target is the human label, a human-trained model has a home-field advantage in
calibration; AUROC is rank-based and carries no such advantage, so it is the quantity we read, with log-loss
reported beside it and read with that caveat.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if len(set(y.tolist())) < 2:
        return None
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s), float)
    sr = s[order]; i = 0
    ranks = np.arange(1, len(s) + 1, dtype=float)
    while i < len(s):
        j = i
        while j + 1 < len(s) and sr[j + 1] == sr[i]:
            j += 1
        r[order[i:j + 1]] = ranks[i:j + 1].mean(); i = j + 1
    npos = float((y == 1).sum()); nneg = float((y == 0).sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acu", required=True)
    ap.add_argument("--scores-glob", required=True)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [r for r in common.load_jsonl(Path(a.acu)) if r.get("units")]
    truth = {r["pair_id"]: [1 - int(x) for x in r["unit_labels"]] for r in rows}   # 1 = omitted
    doc_of = {r["pair_id"]: common.doc_key(r["source"]) for r in rows}
    acc = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(a.scores_glob)):
        if Path(f).stem.endswith("_logits"):        # raw-logit companions of the calibrated files
            continue
        tag = re.sub(r"^acu_", "", Path(f).stem)
        tag = re.sub(r"_seed\d+$", "", tag)
        _, sc = common.read_scores(Path(f))
        for pid, v in sc.items():
            if pid in truth and len(v) == len(truth[pid]):
                acc[tag][pid].append(np.asarray(v, float))
    arms = {t: {p: np.nanmean(np.vstack(v), axis=0) for p, v in d.items()} for t, d in acc.items()}
    names = sorted(arms)
    assert a.primary in arms, f"{a.primary!r} not in {names}"
    pids = sorted(set.intersection(*[set(arms[t]) for t in names]))
    docs = sorted({doc_of[p] for p in pids})
    by_doc = defaultdict(list)
    for p in pids:
        by_doc[doc_of[p]].append(p)

    def pooled(t, ps):
        y = np.concatenate([truth[p] for p in ps]); s = np.concatenate([arms[t][p] for p in ps])
        return auroc(y, s)

    rng = np.random.default_rng(a.seed)
    boot_docs = [rng.choice(docs, size=len(docs), replace=True) for _ in range(a.n_boot)]
    rep = {"n_pairs": len(pids), "n_docs": len(docs), "primary": a.primary, "arms": {}, "paired_vs_primary": {}}
    for t in names:
        rep["arms"][t] = {"auroc_pooled": pooled(t, pids),
                          "auroc_doc_macro": float(np.nanmean([x for x in (pooled(t, by_doc[d]) for d in docs) if x is not None]))}
        print(f"[r04-auroc] {t:34s} pooled {rep['arms'][t]['auroc_pooled']:.4f} macro {rep['arms'][t]['auroc_doc_macro']:.4f}", flush=True)
    for t in names:
        if t == a.primary:
            continue
        d = []
        for pick in boot_docs:
            sub = [p for dd in pick for p in by_doc[dd]]
            x, y = pooled(a.primary, sub), pooled(t, sub)
            if x is not None and y is not None:
                d.append(x - y)
        rep["paired_vs_primary"][t] = {"mean": float(np.mean(d)), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "n_boot": len(d)}
        r = rep["paired_vs_primary"][t]
        print(f"[r04-auroc] {a.primary} - {t:30s} {r['mean']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "matched_auroc.json").write_text(json.dumps(rep, indent=1))
    print("[r04-auroc] written", out / "matched_auroc.json", flush=True)


if __name__ == "__main__":
    main()
