#!/usr/bin/env python3
"""M-C -- apply the registered bars to the three-seed unified verifier.

Inputs: per-seed result.json (coverage axis: AUROC and crossed accuracy on RoSE validation, for the unified model,
zero-shot FactCG and the deployed verifier on identical pairs), per-seed mb_eval JSON (AggreFact test macro BAcc
with dev-tuned thresholds, whole-claim and decomposed), and the M-B A1 reference (FactCG whole-claim macro BAcc).
Bars (PREREG.md, M-C), on the three-seed mean: macro BAcc(whole) >= A1 - 1.0 point; pooled AUROC >= deployed
- 0.01; crossed accuracy (document macro) >= 0.90.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, help="per-seed result.json")
    ap.add_argument("--mb-evals", nargs="+", required=True, help="per-seed mb_eval JSON (verifier = the unified model)")
    ap.add_argument("--a1", required=True, help="mb_eval_factcg.json of M-B")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = [json.load(open(p)) for p in a.results]; ev = [json.load(open(p)) for p in a.mb_evals]; a1 = json.load(open(a.a1))
    a1_macro = 100 * a1["macro"]["bacc_whole"]
    per_seed = []
    for r, e in zip(res, ev):
        per_seed.append({"seed": r["seed"], "macro_whole": 100 * e["macro"]["bacc_whole"], "macro_decomp_min": 100 * e["macro"]["bacc_decomp_min"],
                         "auroc": r["unified"]["auroc_pooled"], "crossed": r["unified"]["crossed"]["discrimination_doc_macro"],
                         "auroc_deployed": r.get("deployed_marsc", {}).get("auroc_pooled"), "crossed_deployed": r.get("deployed_marsc", {}).get("crossed", {}).get("discrimination_doc_macro"),
                         "auroc_zeroshot_factcg": r["zero_shot_factcg"]["auroc_pooled"], "crossed_zeroshot_factcg": r["zero_shot_factcg"]["crossed"]["discrimination_doc_macro"]})
    mean = {k: float(np.mean([s[k] for s in per_seed if s[k] is not None])) for k in per_seed[0] if k != "seed" and any(s[k] is not None for s in per_seed)}
    bars = {"aggrefact": {"threshold": a1_macro - 1.0, "value": mean["macro_whole"], "pass": mean["macro_whole"] >= a1_macro - 1.0},
            "auroc": {"threshold": (mean.get("auroc_deployed") or 0) - 0.01, "value": mean["auroc"], "pass": mean["auroc"] >= (mean.get("auroc_deployed") or 0) - 0.01},
            "crossed": {"threshold": 0.90, "value": mean["crossed"], "pass": mean["crossed"] >= 0.90}}
    rep = {"a1_factcg_whole_macro": a1_macro, "per_seed": per_seed, "three_seed_mean": mean, "bars": bars, "pass_all": all(b["pass"] for b in bars.values())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"[mc-bars] A1 {a1_macro:.2f} | unified macro whole {mean['macro_whole']:.2f} decomp {mean['macro_decomp_min']:.2f} | AUROC {mean['auroc']:.4f} (deployed {mean.get('auroc_deployed')}, FactCG zero-shot {mean['auroc_zeroshot_factcg']:.4f}) | crossed {mean['crossed']:.4f} (deployed {mean.get('crossed_deployed')})")
    print("[mc-bars] " + ", ".join(f"{k}: {'PASS' if v['pass'] else 'FAIL'} ({v['value']:.4f} vs {v['threshold']:.4f})" for k, v in bars.items()) + f" -> {'PASS' if rep['pass_all'] else 'FAIL'}")


if __name__ == "__main__":
    main()
