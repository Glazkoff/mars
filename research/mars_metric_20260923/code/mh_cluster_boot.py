#!/usr/bin/env python3
"""M-H (b) -- LLM-AggreFact cluster-resampling sensitivity (review finding R1-W4, 2026-09-29).

`mb_eval.py` resamples claims within each dataset. Several LLM-AggreFact datasets carry many claims per source
document (RAGTruth about 36, the two TofuEval sets about 50), so claims are not independent draws. This script
reproduces mb_eval.py exactly (dev-tuned thresholds per dataset and arm, macro balanced accuracy over the eleven
datasets, observed contrast decomp_min - whole) and adds a bootstrap that resamples source documents (SHA-1 of the
`doc` field) within each dataset, keeping every claim of a drawn document. The point estimates are unchanged by
construction; only the interval is added. Descriptive; no bar.

  python mh_cluster_boot.py --aggrefact-dir <parquet dir> --dev "scores_<v>_dev_shard*.jsonl" \
      --test "scores_<v>_test_shard*.jsonl" --verifier <v> --out mh_aggrefact_cluster_bootstrap_<v>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mb_eval import ARMS, bacc, best_threshold, load  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggrefact-dir", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--verifier", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_parquet(Path(a.aggrefact_dir) / "test-00000-of-00001.parquet")
    doc_hash = df["doc"].astype(str).map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:16]).tolist()
    dev, test = load(a.dev), load(a.test)
    datasets = sorted(test)
    thr, arrays, clusters = {}, {}, {}
    rep = {"registered": "research/mars_metric_20260923/PREREG.md, Amendment M-H (b)", "verifier": a.verifier,
           "n_boot": a.n_boot, "seed": a.seed, "datasets": {}, "macro": {}, "contrast": {}}
    for ds in datasets:
        yd = np.array([r["label"] for r in dev[ds]]); yt = np.array([r["label"] for r in test[ds]])
        thr[ds] = {arm: best_threshold(yd, np.array([r[arm] for r in dev[ds]])) for arm in ARMS}
        arrays[ds] = {"y": yt, **{arm: np.array([r[arm] for r in test[ds]]) for arm in ARMS}}
        rows = [r["row"] for r in test[ds]]
        assert all(df["dataset"].iloc[i] == ds for i in rows), f"row/dataset mismatch in {ds}"
        cl = np.array([doc_hash[i] for i in rows])
        _, inv = np.unique(cl, return_inverse=True)
        clusters[ds] = inv
        rep["datasets"][ds] = {"n_test_claims": int(len(yt)), "n_source_documents": int(inv.max() + 1),
                               "claims_per_document": float(len(yt) / (inv.max() + 1)),
                               "bacc_whole": bacc(yt, arrays[ds]["whole"], thr[ds]["whole"]),
                               "bacc_decomp_min": bacc(yt, arrays[ds]["decomp_min"], thr[ds]["decomp_min"])}
    for key in ("whole", "decomp_min", "decomp_mean"):
        rep["macro"][f"bacc_{key}"] = float(np.nanmean([bacc(arrays[ds]["y"], arrays[ds][key], thr[ds][key]) for ds in datasets]))
    obs = rep["macro"]["bacc_decomp_min"] - rep["macro"]["bacc_whole"]
    rng = np.random.default_rng(a.seed)
    draws_claim, draws_doc = [], []
    for _ in range(a.n_boot):
        mw_c, mm_c, mw_d, mm_d = [], [], [], []
        for ds in datasets:
            A = arrays[ds]; n = len(A["y"])
            idx = rng.integers(0, n, size=n)                                   # claim resampling (mb_eval.py)
            mw_c.append(bacc(A["y"][idx], A["whole"][idx], thr[ds]["whole"])); mm_c.append(bacc(A["y"][idx], A["decomp_min"][idx], thr[ds]["decomp_min"]))
            inv = clusters[ds]; nd = inv.max() + 1
            cnt = np.bincount(rng.integers(0, nd, size=nd), minlength=nd)      # document resampling
            w = cnt[inv]; idx2 = np.repeat(np.arange(n), w)
            mw_d.append(bacc(A["y"][idx2], A["whole"][idx2], thr[ds]["whole"])); mm_d.append(bacc(A["y"][idx2], A["decomp_min"][idx2], thr[ds]["decomp_min"]))
        draws_claim.append(np.nanmean(mm_c) - np.nanmean(mw_c)); draws_doc.append(np.nanmean(mm_d) - np.nanmean(mw_d))
    pct = lambda d: [float(100 * np.percentile(d, 2.5)), float(100 * np.percentile(d, 97.5))]
    rep["contrast"] = {"decomp_min_minus_whole_macro_points": 100 * obs,
                       "ci95_claim_resampling_points": pct(draws_claim),
                       "ci95_document_resampling_points": pct(draws_doc),
                       "n_documents_total": int(sum(rep["datasets"][ds]["n_source_documents"] for ds in datasets)),
                       "n_claims_total": int(sum(rep["datasets"][ds]["n_test_claims"] for ds in datasets))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"[mh-cluster] {a.verifier}: contrast {100 * obs:+.2f} points; claim CI {pct(draws_claim)}; document CI {pct(draws_doc)}")
    for ds in datasets:
        d = rep["datasets"][ds]
        print(f"  {ds:18s} claims {d['n_test_claims']:5d} docs {d['n_source_documents']:5d} claims/doc {d['claims_per_document']:6.2f}")


if __name__ == "__main__":
    main()
