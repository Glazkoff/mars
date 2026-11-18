#!/usr/bin/env python3
"""M-B.3 -- LLM-AggreFact evaluation of the whole-claim and decomposed readings: per-dataset balanced accuracy
with the threshold tuned per dataset and per arm on dev (fixed 0.5 beside), macro over the 11 datasets, the
shuffled-document control at the real arm's threshold, and the registered contrast decomp_min - whole with a
2,000-draw bootstrap that resamples claims within each dataset and recomputes the macro per draw. The bar
(PREREG.md, M-B): observed macro gain >= +1.0 point with the interval excluding zero.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = ("whole", "decomp_min", "decomp_mean")


def bacc(y: np.ndarray, s: np.ndarray, t: float) -> float:
    pred = (s > t).astype(int)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return float(0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean()))


def best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    cands = np.unique(np.concatenate([np.round(s, 4), [0.5]]))
    best_t, best_b = 0.5, -1.0
    for t in cands:
        b = bacc(y, s, t)
        if b > best_b + 1e-12:
            best_t, best_b = float(t), b
    return best_t


def load(pattern: str) -> dict[str, list[dict]]:
    """One or several shard files (a glob); rows are ordered by their benchmark row index."""
    import glob
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows.extend(json.loads(line) for line in open(f, encoding="utf-8"))
    rows.sort(key=lambda r: r["row"])
    by = defaultdict(list)
    for r in rows:
        by[r["dataset"]].append(r)
    return by


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--verifier", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--bar-points", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev, test = load(a.dev), load(a.test)
    datasets = sorted(test)
    rep = {"verifier": a.verifier, "datasets": {}, "macro": {}, "controls": {}, "contrast": {}, "bar": {}}
    thr: dict[str, dict[str, float]] = {}
    arrays: dict[str, dict] = {}
    for ds in datasets:
        yd = np.array([r["label"] for r in dev[ds]]); yt = np.array([r["label"] for r in test[ds]])
        thr[ds] = {arm: best_threshold(yd, np.array([r[arm] for r in dev[ds]])) for arm in ARMS}
        arrays[ds] = {"y": yt, **{arm: np.array([r[arm] for r in test[ds]]) for arm in ARMS},
                      "whole_shuf": np.array([r.get("whole_shuf", np.nan) for r in test[ds]]),
                      "decomp_min_shuf": np.array([r.get("decomp_min_shuf", np.nan) for r in test[ds]])}
        d = {"n_dev": int(len(yd)), "n_test": int(len(yt)), "fallback_rate_test": float(np.mean([r["fallback"] for r in test[ds]])),
             "facts_per_claim_test": float(np.mean([r["n_facts"] for r in test[ds]])), "threshold": thr[ds]}
        for arm in ARMS:
            d[f"bacc_{arm}"] = bacc(yt, arrays[ds][arm], thr[ds][arm])
            d[f"bacc_{arm}_fixed05"] = bacc(yt, arrays[ds][arm], 0.5)
        d["bacc_whole_shuf"] = bacc(yt, arrays[ds]["whole_shuf"], thr[ds]["whole"])
        d["bacc_decomp_min_shuf"] = bacc(yt, arrays[ds]["decomp_min_shuf"], thr[ds]["decomp_min"])
        rep["datasets"][ds] = d
    for key in [f"bacc_{arm}" for arm in ARMS] + [f"bacc_{arm}_fixed05" for arm in ARMS] + ["bacc_whole_shuf", "bacc_decomp_min_shuf"]:
        rep["macro"][key] = float(np.nanmean([rep["datasets"][ds][key] for ds in datasets]))
    obs = rep["macro"]["bacc_decomp_min"] - rep["macro"]["bacc_whole"]
    obs_mean = rep["macro"]["bacc_decomp_mean"] - rep["macro"]["bacc_whole"]
    rng = np.random.default_rng(a.seed)
    draws_min, draws_mean = [], []
    for _ in range(a.n_boot):
        m_w, m_min, m_mean = [], [], []
        for ds in datasets:
            A = arrays[ds]; n = len(A["y"]); idx = rng.integers(0, n, size=n)
            y = A["y"][idx]
            m_w.append(bacc(y, A["whole"][idx], thr[ds]["whole"]))
            m_min.append(bacc(y, A["decomp_min"][idx], thr[ds]["decomp_min"]))
            m_mean.append(bacc(y, A["decomp_mean"][idx], thr[ds]["decomp_mean"]))
        draws_min.append(np.nanmean(m_min) - np.nanmean(m_w)); draws_mean.append(np.nanmean(m_mean) - np.nanmean(m_w))
    ci_min = [float(np.percentile(draws_min, 2.5)), float(np.percentile(draws_min, 97.5))]
    ci_mean = [float(np.percentile(draws_mean, 2.5)), float(np.percentile(draws_mean, 97.5))]
    rep["contrast"] = {"decomp_min_minus_whole_macro": {"observed_points": 100 * obs, "ci95_points": [100 * ci_min[0], 100 * ci_min[1]]},
                       "decomp_mean_minus_whole_macro": {"observed_points": 100 * obs_mean, "ci95_points": [100 * ci_mean[0], 100 * ci_mean[1]]},
                       "per_dataset_decomp_min_minus_whole_points": {ds: 100 * (rep["datasets"][ds]["bacc_decomp_min"] - rep["datasets"][ds]["bacc_whole"]) for ds in datasets}}
    worst = min(rep["contrast"]["per_dataset_decomp_min_minus_whole_points"].values())
    rep["bar"] = {"registered": "M-B: macro BAcc(decomp_min) - BAcc(whole) >= +1.0 point, claim bootstrap interval excluding zero",
                  "observed_points": 100 * obs, "ci95_points": [100 * ci_min[0], 100 * ci_min[1]],
                  "pass": bool(100 * obs >= a.bar_points and ci_min[0] > 0),
                  "secondary_no_dataset_worse_than_1_point": bool(worst >= -1.0), "worst_dataset_points": worst}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"[mb-eval] {a.verifier}: macro whole {100*rep['macro']['bacc_whole']:.2f} | decomp_min {100*rep['macro']['bacc_decomp_min']:.2f} | "
          f"decomp_mean {100*rep['macro']['bacc_decomp_mean']:.2f} | controls whole {100*rep['macro']['bacc_whole_shuf']:.2f} decomp {100*rep['macro']['bacc_decomp_min_shuf']:.2f}")
    print(f"[mb-eval] contrast decomp_min - whole: {100*obs:+.2f} [{100*ci_min[0]:+.2f},{100*ci_min[1]:+.2f}] -> {'PASS' if rep['bar']['pass'] else 'FAIL'}; worst dataset {worst:+.2f}")
    for ds in datasets:
        d = rep["datasets"][ds]
        print(f"  {ds:18s} n={d['n_test']:5d} facts/claim {d['facts_per_claim_test']:.2f} whole {100*d['bacc_whole']:.2f} min {100*d['bacc_decomp_min']:.2f} mean {100*d['bacc_decomp_mean']:.2f} | ctrl {100*d['bacc_whole_shuf']:.2f}/{100*d['bacc_decomp_min_shuf']:.2f}")


if __name__ == "__main__":
    main()
