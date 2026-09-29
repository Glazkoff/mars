#!/usr/bin/env python3
"""MARS-C strengthen B0 -- the WITHIN-FAMILY control gaps (secondary estimands of B2).

mc_e2_eval.py contrasts every system against ONE primary, and its `paired_recall.mean` is the BOOTSTRAP mean,
not the observed contrast (the project trap that produced a retracted claim). The secondary estimand of B2 is a
different contrast: for each verifier FAMILY, real summary minus its own shuffled-summary and no-summary
controls, on both recall@10 and omission precision.

This file computes exactly that from the evaluator's own frozen `--dump-per-pair` rows, so no scoring, no
matching rule and no emission rule is re-implemented here. The bootstrap resamples SOURCE DOCUMENTS (the `doc`
column the evaluator writes, i.e. common.doc_key) and the SAME document draw is applied to every system in a
replicate, so the gaps are paired across families as well as within them.

Reported per family: observed recall@k and omission precision in each mode, the observed gaps, their
document-clustered 95% intervals, and the registered signature test.

Registered expectation (declared before execution, PREREG.md Wave B / B2 secondary estimands):
a family that genuinely conditions on the candidate shows a POSITIVE omission-precision gap and a NEGATIVE
recall gap; MARS-C's own measured signature is +0.065 precision / -0.055 recall. A family with a gap of zero on
both is candidate-blind on this endpoint too.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EXPECTED = {
    "registered": "research/marsc_strengthen_20260918/PREREG.md -- Wave B / B0 controls, B2 secondary estimands",
    "declared_before_any_number_in_this_file_was_read": True,
    "estimands": "per verifier family: recall@k and omission precision under real / shuffled / empty candidate, "
                 "and the real-minus-control gaps",
    "conditioning_signature": "positive omission-precision gap AND negative recall gap",
    "mars_c_measured_signature_in_the_published_run": {"omission_precision_gap": 0.065, "recall_gap": -0.055},
    "reading_rule": "recall@k never charges for demoting a fact the summary DID convey, so a NEGATIVE recall gap "
                    "is evidence of conditioning, not of weakness; the precision gap is the endpoint that charges "
                    "for false alarms.",
    "no_bar": "B0 is infrastructure and B2's secondary estimands carry no pass/fail bar; the sign and interval "
              "are the result.",
}


def stats(rec: np.ndarray, hit: np.ndarray, fa: np.ndarray, ne: np.ndarray, sel: np.ndarray) -> tuple[float, float]:
    r = rec[sel]
    r = r[np.isfinite(r)]
    recall = float(r.mean()) if r.size else float("nan")
    h = float((hit[sel] * ne[sel]).sum())
    f = float((fa[sel] * ne[sel]).sum())
    return recall, (h / (h + f) if (h + f) > 0 else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pair", required=True, help="mc_e2_eval.py --dump-per-pair rows")
    ap.add_argument("--k", default="10")
    ap.add_argument("--modes", default="shuffled,empty", help="control modes to contrast against --reference")
    ap.add_argument("--reference", default="real")
    ap.add_argument("--sep", default="@", help="separator in the system name: FAMILY<sep>MODE")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.per_pair, encoding="utf-8")]
    rows = [r for r in rows if str(r["k"]) == str(a.k)]
    assert rows, f"no rows at k={a.k} in {a.per_pair}"
    systems = sorted({r["system"] for r in rows})
    pairs = sorted({r["pair_id"] for r in rows})
    pidx = {p: i for i, p in enumerate(pairs)}
    doc_of = {r["pair_id"]: r["doc"] for r in rows}
    by_doc: dict[str, list[int]] = defaultdict(list)
    for p in pairs:
        by_doc[doc_of[p]].append(pidx[p])
    docs = sorted(by_doc)
    doc_arrays = [np.array(by_doc[d], dtype=np.int64) for d in docs]

    n = len(pairs)
    arr = {s: {k: np.full(n, np.nan) for k in ("recall", "hit", "false_alert", "n_emitted")} for s in systems}
    for r in rows:
        i = pidx[r["pair_id"]]
        t = arr[r["system"]]
        t["recall"][i] = np.nan if r["recall"] is None else float(r["recall"])
        t["hit"][i] = float(r["hit"])
        t["false_alert"][i] = float(r["false_alert"])
        t["n_emitted"][i] = float(r["n_emitted"])
    for s in systems:
        for k in ("hit", "false_alert", "n_emitted"):
            arr[s][k] = np.nan_to_num(arr[s][k], nan=0.0)

    allsel = np.arange(n)
    obs = {s: stats(arr[s]["recall"], arr[s]["hit"], arr[s]["false_alert"], arr[s]["n_emitted"], allsel)
           for s in systems}

    rng = np.random.default_rng(a.seed)
    draws = {s: {"recall": [], "omission_precision": []} for s in systems}
    for _ in range(a.n_boot):
        pick = rng.integers(0, len(docs), size=len(docs))
        sel = np.concatenate([doc_arrays[j] for j in pick])
        for s in systems:
            rc, op = stats(arr[s]["recall"], arr[s]["hit"], arr[s]["false_alert"], arr[s]["n_emitted"], sel)
            draws[s]["recall"].append(rc)
            draws[s]["omission_precision"].append(op)
    for s in systems:
        for m in draws[s]:
            draws[s][m] = np.asarray(draws[s][m], dtype=float)

    fams: dict[str, dict[str, str]] = defaultdict(dict)
    for s in systems:
        fam, _, mode = s.rpartition(a.sep)
        fams[fam or s][mode or a.reference] = s

    rep = {"expectations": EXPECTED, "per_pair_file": str(a.per_pair), "k": a.k, "n_pairs": n,
           "n_docs": len(docs), "n_boot": a.n_boot, "bootstrap_seed": a.seed,
           "reference_mode": a.reference, "control_modes": a.modes.split(","),
           "systems_seen": systems, "families": {}}

    for fam, modes in sorted(fams.items()):
        entry = {"modes_present": sorted(modes), "observed": {}, "gaps": {}}
        for mname, s in sorted(modes.items()):
            entry["observed"][mname] = {"system": s, "recall": obs[s][0], "omission_precision": obs[s][1]}
        ref = modes.get(a.reference)
        if ref is None:
            entry["note"] = f"no {a.reference!r} arm for this family; gaps not computed"
            rep["families"][fam] = entry
            continue
        for mname in a.modes.split(","):
            s = modes.get(mname)
            if s is None:
                continue
            g = {}
            for metric in ("recall", "omission_precision"):
                o = (obs[ref][0] - obs[s][0]) if metric == "recall" else (obs[ref][1] - obs[s][1])
                d = draws[ref][metric] - draws[s][metric]
                d = d[np.isfinite(d)]
                g[metric] = {"observed_gap": float(o),
                             "ci95_document_clustered": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                             "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0),
                             "bootstrap_mean_NOT_the_observed_gap": float(d.mean()), "n_draws": int(d.size)}
            g["conditioning_signature"] = bool(g["omission_precision"]["observed_gap"] > 0
                                               and g["recall"]["observed_gap"] < 0)
            entry["gaps"][mname] = g
            print(f"[b0gap] {fam:28s} real-{mname:8s} recall {g['recall']['observed_gap']:+.4f} "
                  f"[{g['recall']['ci95_document_clustered'][0]:+.4f},{g['recall']['ci95_document_clustered'][1]:+.4f}] "
                  f"| prec {g['omission_precision']['observed_gap']:+.4f} "
                  f"[{g['omission_precision']['ci95_document_clustered'][0]:+.4f},"
                  f"{g['omission_precision']['ci95_document_clustered'][1]:+.4f}] "
                  f"| signature {'YES' if g['conditioning_signature'] else 'no'}", flush=True)
        rep["families"][fam] = entry

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "control_gaps.json").write_text(json.dumps(rep, indent=1))
    print(f"[b0gap] written {out / 'control_gaps.json'}", flush=True)


if __name__ == "__main__":
    main()
