#!/usr/bin/env python3
"""M-D.3 -- sentence-level localization on RAGTruth: a sentence is flagged when its support (whole sentence) or the
minimum over its facts (decomposed) falls below the threshold; the gold sentences are those overlapping an
annotated hallucination span. Endpoints: sentence precision / recall / F1 (pooled, response-clustered bootstrap),
response-level balanced accuracy of "any sentence flagged" against "any span"; per task type. The registered
contrast: F1(decomposed) - F1(whole) with a 2,000-draw response bootstrap. Thresholds come from M-B's RAGTruth row
(dev-tuned, per arm) and are passed in, never tuned here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def flags(row: dict, key_sent: str, key_facts: str, t_whole: float, t_dec: float) -> tuple[np.ndarray, np.ndarray]:
    whole = np.array([(s is not None and s < t_whole) for s in row[key_sent]], dtype=bool)
    dec = np.array([(min(fs) < t_dec) if fs else (s is not None and s < t_whole) for fs, s in zip(row[key_facts], row[key_sent])], dtype=bool)
    return whole, dec


def prf(pred: np.ndarray, gold: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.sum(pred & gold)); fp = float(np.sum(pred & ~gold)); fn = float(np.sum(~pred & gold))
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def bacc(pred: np.ndarray, gold: np.ndarray) -> float:
    pos, neg = gold, ~gold
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return float(0.5 * (pred[pos].mean() + (~pred[neg]).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--t-whole", type=float, required=True)
    ap.add_argument("--t-decomp", type=float, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pairs = {json.loads(l)["pair_id"]: json.loads(l) for l in open(a.pairs, encoding="utf-8")}
    rows = [json.loads(l) for l in open(a.scores, encoding="utf-8")]
    per = []  # per response: gold sentence mask, whole flags, decomposed flags, control flags, task type
    for r in rows:
        p = pairs[r["pair_id"]]
        gold = np.array([any(s < sp["end"] and e > sp["start"] for sp in p["spans"]) for s, e in p["sentences"]], dtype=bool)
        w, d = flags(r, "sentence_support", "fact_supports", a.t_whole, a.t_decomp)
        ws, ds = flags(r, "sentence_support_shuf", "fact_supports_shuf", a.t_whole, a.t_decomp)
        per.append({"task": r["task_type"], "gold": gold, "whole": w, "dec": d, "whole_shuf": ws, "dec_shuf": ds})

    def summarise(sel: list[dict]) -> dict:
        cat = {k: np.concatenate([x[k] for x in sel]) for k in ("gold", "whole", "dec", "whole_shuf", "dec_shuf")}
        res = {"n_responses": len(sel), "n_sentences": int(len(cat["gold"])), "gold_rate": float(cat["gold"].mean())}
        for arm in ("whole", "dec", "whole_shuf", "dec_shuf"):
            p, r, f = prf(cat[arm], cat["gold"])
            res[arm] = {"precision": p, "recall": r, "f1": f,
                        "response_bacc": bacc(np.array([x[arm].any() for x in sel]), np.array([x["gold"].any() for x in sel]))}
        return res
    rng = np.random.default_rng(a.seed)
    rep = {"thresholds": {"whole": a.t_whole, "decomp": a.t_decomp}, "all": summarise(per), "by_task": {}, "contrast": {}}
    for task in sorted({x["task"] for x in per}):
        rep["by_task"][task] = summarise([x for x in per if x["task"] == task])
    draws = []
    for _ in range(a.n_boot):
        idx = rng.integers(0, len(per), size=len(per)); sel = [per[i] for i in idx]
        g = np.concatenate([x["gold"] for x in sel]); w = np.concatenate([x["whole"] for x in sel]); d = np.concatenate([x["dec"] for x in sel])
        draws.append(prf(d, g)[2] - prf(w, g)[2])
    obs = rep["all"]["dec"]["f1"] - rep["all"]["whole"]["f1"]
    ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    rep["contrast"] = {"f1_decomp_minus_whole": {"observed": obs, "ci95": ci}, "bar_pass": bool(ci[0] > 0)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(rep, indent=1))
    A = rep["all"]
    print(f"[md-eval] sentences {A['n_sentences']} gold rate {A['gold_rate']:.3f} | whole F1 {A['whole']['f1']:.3f} (P {A['whole']['precision']:.3f} R {A['whole']['recall']:.3f}) | "
          f"decomposed F1 {A['dec']['f1']:.3f} (P {A['dec']['precision']:.3f} R {A['dec']['recall']:.3f}) | controls F1 {A['whole_shuf']['f1']:.3f}/{A['dec_shuf']['f1']:.3f}")
    print(f"[md-eval] contrast F1 decomposed - whole {obs:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] -> {'PASS' if ci[0] > 0 else 'FAIL'}; response BAcc whole {A['whole']['response_bacc']:.3f} decomposed {A['dec']['response_bacc']:.3f}")
    for task, T in rep["by_task"].items():
        print(f"  {task:10s} n={T['n_responses']} whole F1 {T['whole']['f1']:.3f} decomposed F1 {T['dec']['f1']:.3f}")


if __name__ == "__main__":
    main()
