#!/usr/bin/env python3
"""M-H (a) -- cohort composition of the common-population metric table (review finding R1-W3, 2026-09-29).

Reads the meta-evaluation pairs and score files exactly as `ma_meta.py --common-cohort` does (same Column class,
same tau rule, same column list) and reports, per (dataset, dimension), the counts at every filtering stage:
pairs carrying the human dimension; pairs every listed column scores (the caption's n_pairs); documents on which
every column has a defined within-document tau (n_docs_all_tau); and the pairs inside those documents, which is
what the reported mean tau is actually computed over. For UniSumEval it adds the domain and source-length
composition of the full pool, of the all-metric cohort and of the shared-pair populations of the contrasts quoted
in the text. Descriptive; no bar.

  python mh_cohort.py --pairs pairs_ma.jsonl --scores "ma/scores/scores_*.jsonl" "mg/scores/scores_*.jsonl" \
      --meta-common meta_common.json --out mh_cohort_composition.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ma_meta import SKIP, TARGETS, Column, common_mask, scored  # noqa: E402

CONTRASTS = [  # (dataset, dimension, mars column, comparator column) as quoted in Section VI-G
    ("unisumeval", "completeness", "marsR_marsc", "a3cu_R"),
    ("unisumeval", "completeness", "marsR_factcg", "geval_completeness"),
    ("unisumeval", "completeness", "marsR_free", "geval_completeness"),
    ("unisumeval", "completeness", "marsR_free", "a3cu_R"),
    ("unisumeval", "faithfulness", "marsP_factcg", "factcg_whole"),
    ("unisumeval", "faithfulness", "marsP_factcg", "alignscore"),
    ("rose", "acu_recall", "marsR_marsc", "a2cu_R"),
    ("summeval", "relevance", "marsR_marsc", "unieval_relevance"),
]


def composition(sub: list[dict], docs_keep: set[str] | None, q: np.ndarray) -> dict:
    """Domain counts by pair and by document; source-length statistics once per DOCUMENT (a source appears
    once per system among the pairs, so pair-weighted length statistics would repeat it up to nine times)."""
    rows = [p for p in sub if docs_keep is None or p["doc_id"] in docs_keep]
    first = {}
    for p in rows:
        first.setdefault(p["doc_id"], p)
    dom_pairs = Counter(p["domain"] for p in rows)
    dom_docs = Counter(p["domain"] for p in first.values())
    lens = np.array([len(p["source"].split()) for p in first.values()])
    quart = Counter(int(np.searchsorted(q, L, side="right")) for L in lens) if len(lens) else Counter()
    return {"n_pairs": len(rows), "n_docs": len(first), "domain_pairs": dict(sorted(dom_pairs.items())),
            "domain_docs": dict(sorted(dom_docs.items())),
            "source_words_median_per_document": float(np.median(lens)) if len(lens) else None,
            "source_words_quartile_share_per_document": {f"Q{i + 1}": round(quart.get(i, 0) / max(1, len(first)), 3) for i in range(4)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--meta-common", required=True, help="meta_common.json of record; its column list defines the cohort")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pairs = [json.loads(l) for l in open(a.pairs, encoding="utf-8")]
    for p in pairs:
        p["scores"] = {k: float(v) for k, v in (p.get("machine") or {}).items() if v is not None}
    by_id = {p["pair_id"]: p for p in pairs}
    for pat in a.scores:
        for f in sorted(glob.glob(pat)):
            for line in open(f, encoding="utf-8"):
                r = json.loads(line); p = by_id.get(r["pair_id"])
                if p is None:
                    continue
                for k, v in r.items():
                    if k not in SKIP and v is not None:
                        p["scores"][k] = float(v)
    meta = json.loads(Path(a.meta_common).read_text())
    rep: dict = {"registered": "research/mars_metric_20260923/PREREG.md, Amendment M-H (a)", "datasets": {}, "contrasts": {}}
    for ds, dims in TARGETS.items():
        sub_all = [p for p in pairs if p["dataset"] == ds]
        docs = sorted({p["doc_id"] for p in sub_all}); doc_index = {d: i for i, d in enumerate(docs)}
        systems = sorted({p["system"] for p in sub_all}); sys_index = {s: i for i, s in enumerate(systems)}
        first_of = {}
        for p in sub_all:
            first_of.setdefault(p["doc_id"], p)
        full_lens = np.array([len(p["source"].split()) for p in first_of.values()])   # one length per document
        q = np.quantile(full_lens, [0.25, 0.5, 0.75])
        rep["datasets"][ds] = {"n_pairs_total": len(sub_all), "n_docs_total": len(docs),
                               "source_words_quartiles_per_document": [float(x) for x in q], "dims": {}}
        for dim in dims:
            cc = meta["datasets"][ds]["common_cohort"][dim]["columns"]
            cc = [c for c in cc if any(scored(p, c) for p in sub_all)]
            with_human = [p for p in sub_all if p["human"].get(dim) is not None]
            sub = [p for p in with_human if all(scored(p, c) for c in cc)]
            keep = common_mask(sub, cc, dim, doc_index, sys_index)
            keep_docs = {docs[i] for i in np.flatnonzero(keep)}
            contributing = [p for p in sub if p["doc_id"] in keep_docs]
            # documents lost to each single column's undefined tau, on the score-complete pairs
            lost = {}
            for c in cc:
                col = Column(sub, c, dim, doc_index, sys_index)
                lost[c] = int(np.sum(np.isnan(col.tau_doc[[doc_index[d] for d in sorted({p['doc_id'] for p in sub})]])))
            e: dict = {"columns": cc,
                 "stage_1_pairs_with_human_dimension": len(with_human),
                 "stage_1_docs": len({p["doc_id"] for p in with_human}),
                 "stage_2_pairs_every_column_scored": len(sub),
                 "stage_2_docs": len({p["doc_id"] for p in sub}),
                 "stage_3_docs_every_tau_defined": int(keep.sum()),
                 "stage_3_pairs_contributing": len(contributing),
                 "docs_with_undefined_tau_per_column_on_stage_2": dict(sorted(lost.items(), key=lambda kv: -kv[1])),
                 "check_meta_common": {"n_pairs": meta["datasets"][ds]["common_cohort"][dim]["n_pairs"],
                                       "n_docs_all_tau": meta["datasets"][ds]["common_cohort"][dim]["n_docs_all_tau"]}}
            if ds == "unisumeval":
                e["composition_full_pool"] = composition(with_human, None, q)
                e["composition_stage_2"] = composition(sub, None, q)
                e["composition_stage_3_cohort"] = composition(sub, keep_docs, q)
            rep["datasets"][ds]["dims"][dim] = e
            print(f"[mh-cohort] {ds}/{dim}: human {len(with_human)} -> scored {len(sub)} pairs -> {int(keep.sum())} docs / {len(contributing)} pairs", flush=True)
        for cds, dim, mc, oc in CONTRASTS:
            if cds != ds:
                continue
            sp = [p for p in sub_all if p["human"].get(dim) is not None and scored(p, mc) and scored(p, oc)]
            km = common_mask(sp, [mc, oc], dim, doc_index, sys_index)
            kd = {docs[i] for i in np.flatnonzero(km)}
            contributing = [p for p in sp if p["doc_id"] in kd]
            e: dict = {"pairs_both_scored": len(sp), "docs_both_scored": len({p["doc_id"] for p in sp}),
                 "docs_both_tau_defined": int(km.sum()), "pairs_contributing": len(contributing)}
            if ds == "unisumeval":
                e["composition"] = composition(sp, kd, q)
            rep["contrasts"][f"{ds}/{dim}/{mc}-{oc}"] = e
            print(f"[mh-cohort] contrast {mc}-{oc} on {ds}/{dim}: {len(sp)} pairs -> {int(km.sum())} docs / {len(contributing)} pairs", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"[mh-cohort] written {a.out}")


if __name__ == "__main__":
    main()
