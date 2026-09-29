#!/usr/bin/env python3
"""MARS-C R05 -- equivalence / non-inferiority for the label-efficiency curve (FULL_REVIEW_2026-09-17 failure 5).

The paper said 85 labelled documents "suffice" and that the gain is "the kind of supervision, not the amount".
The evidence was four overlapping confidence intervals, which is an absence of a detected difference, not
equivalence: with these sample sizes the intervals would overlap for a range of real losses too.

This script states the missing test. For each reduced budget it takes the document-level paired difference
against the full budget, forms a document bootstrap, and reports

  * the two-sided interval (what the paper had),
  * TOST at a DECLARED margin: equivalence is claimed only if the whole (1-2*alpha) interval lies inside
    (-delta, +delta). The declared margin is 0.05 recall@10, the same size as the registered headline effect --
    a supervision reduction is "equivalent" only if any loss it causes is smaller than the effect the paper
    claims to detect,
  * the ACHIEVED non-inferiority margin: the smallest delta for which the one-sided upper bound clears, i.e.
    the strongest true statement the data support.

Labelled post-hoc: the curve was registered, this analysis of it was not.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pair", required=True, help="mc_e2_eval --dump-per-pair output")
    ap.add_argument("--k", default="10")
    ap.add_argument("--reference", default="frac100+div1")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.per_pair, encoding="utf-8")]
    rows = [r for r in rows if str(r["k"]) == a.k and r["recall"] is not None]
    by = defaultdict(lambda: defaultdict(list))                    # system -> doc -> recalls
    for r in rows:
        by[r["system"]][r["doc"]].append(float(r["recall"]))
    systems = sorted(by)
    assert a.reference in by, f"reference {a.reference!r} not among {systems}"
    docs = sorted(set(by[a.reference]) & set.intersection(*[set(by[s]) for s in systems]))
    rng = np.random.default_rng(a.seed)
    ref = {d: float(np.mean(by[a.reference][d])) for d in docs}
    rep = {"registered": "post-hoc analysis of the registered T07 label-efficiency curve",
           "k": a.k, "reference": a.reference, "n_docs": len(docs), "declared_margin": a.margin,
           "alpha": a.alpha, "n_boot": a.n_boot,
           "note": "TOST on document-level paired differences; equivalence requires the whole "
                   f"{100 * (1 - 2 * a.alpha):.0f}% interval inside +/-{a.margin}",
           "systems": {}}
    idx = np.arange(len(docs))
    for s in systems:
        cur = {d: float(np.mean(by[s][d])) for d in docs}
        diff = np.array([ref[d] - cur[d] for d in docs])           # positive = the reduced budget LOSES recall
        draws = np.array([diff[rng.choice(idx, size=len(idx), replace=True)].mean() for _ in range(a.n_boot)])
        lo2, hi2 = np.percentile(draws, [100 * a.alpha / 2, 100 * (1 - a.alpha / 2)])
        lo1, hi1 = np.percentile(draws, [100 * a.alpha, 100 * (1 - a.alpha)])
        achieved = float(max(abs(hi1), abs(lo1)))
        rep["systems"][s] = {
            "n_docs": len(docs), "mean_loss_vs_reference": float(diff.mean()),
            "ci95_two_sided": [float(lo2), float(hi2)],
            "tost_interval": [float(lo1), float(hi1)],
            "equivalent_at_declared_margin": bool(hi1 < a.margin and lo1 > -a.margin),
            "non_inferior_at_declared_margin": bool(hi1 < a.margin),
            "smallest_margin_supported": achieved,
            "p_two_sided_vs_zero": float(2 * min((draws <= 0).mean(), (draws >= 0).mean()))}
        r = rep["systems"][s]
        print(f"[r05] {s:18s} loss vs {a.reference} {r['mean_loss_vs_reference']:+.4f} "
              f"95% [{lo2:+.4f},{hi2:+.4f}] | TOST [{lo1:+.4f},{hi1:+.4f}] "
              f"equivalent@{a.margin}={r['equivalent_at_declared_margin']} "
              f"non-inferior@{a.margin}={r['non_inferior_at_declared_margin']} "
              f"smallest margin supported {achieved:.4f}", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "label_efficiency_equivalence.json").write_text(json.dumps(rep, indent=1))
    print(f"[r05] written {out / 'label_efficiency_equivalence.json'}", flush=True)


if __name__ == "__main__":
    main()
