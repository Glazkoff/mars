#!/usr/bin/env python3
"""MARS-C D1b -- the three-role scorer: agreement-conditioned rate BESIDE the truly adjudicated rate.

The defect this exists to make visible.
Every human-precision number this project has published is AGREEMENT-CONDITIONED. `mc_human_analyze.py` builds
`systems_both_agree` from the items the two annotators labelled the same way; `mc_human_rank.py` calls that mode
`both_agree` and the manuscript calls it "adjudicated". They are not the same estimand. Agreement-conditioning
DROPS the items the two annotators resolved differently; adjudication RESOLVES them with a third, blinded
judgement. Dropping disagreements keeps the easy items, so the agreement-conditioned rate is biased upward by an
amount nobody has measured -- and the bias is exactly the quantity a reviewer will ask about.

What this script reports, per system and per budget:
  * `per_annotator`     -- each annotator's own rate (the unconditioned, single-rater view);
  * `agreement_only`    -- the published estimand, reproduced exactly, with its dropped-item count stated;
  * `adjudicated`       -- the agreed verdict where the annotators agree, the ADJUDICATOR's verdict where they
                           do not, over the whole co-labelled set;
  * `adjudicated_bounds`-- the same with every still-unresolved disagreement forced negative, then positive.
                           The claim must survive the lower bound;
  * `selection_gap`     -- agreement_only minus adjudicated. This number is the finding.

Intervals are DOCUMENT-CLUSTERED (the A3 repair): the bootstrap resamples `doc_key` clusters from the frozen
`human_sample_docmap.jsonl`, never items inside status strata. Status strata are reweighted to the evaluator's
own natural status mix at the budget, exactly as `mc_human_rank.py` does, with the weights held fixed.

This script reads labels. It must not be run until labels exist, and it computes nothing else.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

STATUSES = ("unknown", "hit", "false_alert")


def load_jsonl(p: Path):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def kappa(x, y):
    if not x:
        return None
    po = sum(a == b for a, b in zip(x, y)) / len(x)
    cats = set(x) | set(y)
    pe = sum((x.count(c) / len(x)) * (y.count(c) / len(y)) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else None


def weighted_rate(items: list[dict], w: dict[str, float]) -> float | None:
    """Status-stratified rate reweighted to the evaluator's natural status mix. None if a weighted stratum is empty."""
    by = defaultdict(list)
    for it in items:
        by[it["status"]].append(it["v"])
    tot, acc = 0.0, 0.0
    for st, wt in w.items():
        if wt <= 0:
            continue
        if not by.get(st):
            return None
        acc += wt * (sum(by[st]) / len(by[st]))
        tot += wt
    return acc / tot if tot > 0 else None


def cluster_boot(items: list[dict], w: dict[str, float], rng, n_boot: int) -> dict:
    """Percentile bootstrap over SOURCE-DOCUMENT clusters. Degenerate replicates are counted, not hidden."""
    point = weighted_rate(items, w)
    by_doc = defaultdict(list)
    for it in items:
        by_doc[it["doc_key"]].append(it)
    docs = sorted(by_doc)
    if point is None or len(docs) < 2:
        return {"point": point, "ci": None, "n_items": len(items), "n_docs": len(docs),
                "n_boot_valid": 0, "note": "too few document clusters for a percentile interval"}
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(docs), len(docs))
        rep = [it for j in pick for it in by_doc[docs[j]]]
        v = weighted_rate(rep, w)
        if v is not None:
            vals.append(v)
    ci = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) >= 100 else None
    return {"point": float(point), "ci": ci, "n_items": len(items), "n_docs": len(docs),
            "n_boot_valid": len(vals),
            "note": None if ci else "fewer than 100 non-degenerate replicates; no interval is reported"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", nargs="+", required=True, help="label app export.jsonl (carries `role` from app v2)")
    ap.add_argument("--key", nargs="+", required=True, help="blinded sample key file(s)")
    ap.add_argument("--docmap", nargs="+", required=True, help="human_sample_docmap.jsonl written by d1_freeze.py")
    ap.add_argument("--e2-json", required=True, help="evaluator json supplying the natural status mix per budget")
    ap.add_argument("--annotators", nargs=2, default=None, help="override the two primary annotators by name")
    ap.add_argument("--adjudicator", default=None, help="override the adjudicator by name")
    ap.add_argument("--budgets", type=int, nargs="+", default=[10])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    key = {r["item_id"]: r for f in a.key for r in load_jsonl(Path(f))}
    dmap = {r["item_id"]: r for f in a.docmap for r in load_jsonl(Path(f))}
    lab: dict[str, dict[str, dict]] = defaultdict(dict)
    roles: dict[str, str] = {}
    for f in a.export:
        for r in load_jsonl(Path(f)):
            lab[r["annotator"]][r["item_id"]] = r
            roles[r["annotator"]] = r.get("role", "annotator")

    ann = list(a.annotators) if a.annotators else sorted(n for n in lab if roles.get(n) != "adjudicator")[:2]
    adj = a.adjudicator if a.adjudicator else next((n for n in sorted(lab) if roles.get(n) == "adjudicator"), None)
    if len(ann) < 2:
        raise SystemExit(f"[d1b] need two annotators, found {ann} (roles {roles})")
    verdict = lambda r: (r.get("q1") == "A" and r.get("q2") == "B")

    e2 = json.load(open(a.e2_json))["k"]
    rep = {"registered": "research/marsc_strengthen_20260918/PREREG.md Wave D / D1b",
           "estimands": {
               "agreement_only": "the published estimand: items the two annotators labelled the same way; "
                                 "disagreements DROPPED",
               "adjudicated": "agreed verdict where they agree, the adjudicator's verdict where they do not; "
                              "no item dropped",
               "selection_gap": "agreement_only - adjudicated"},
           "roles": {"annotators": ann, "adjudicator": adj, "declared": roles},
           "n_labelled": {n: len(lab[n]) for n in sorted(lab)},
           "adjudicator_present": adj is not None, "budgets": {}}

    if adj is None:
        rep["WARNING"] = ("No adjudicator in the export. Only the agreement-conditioned estimand is computable, "
                          "and it must not be called 'adjudicated' in the manuscript.")

    # --- degeneracy gate: the A3b failure (one rater answered identically on every item) ------------------
    deg = {}
    for n in ann:
        sig = Counter((lab[n][i].get("q1"), lab[n][i].get("q2")) for i in lab[n])
        deg[n] = {"n": len(lab[n]), "distinct_answer_patterns": len(sig),
                  "modal_share": (max(sig.values()) / max(1, len(lab[n]))) if sig else None}
    rep["rater_degeneracy"] = deg
    if any(d["distinct_answer_patterns"] == 1 for d in deg.values()):
        rep["WARNING_DEGENERATE"] = ("A rater gave the identical answer on every item. An agreement filter against a "
                                     "constant rater is not a two-annotator result and kappa is exactly 0.")

    both = sorted(i for i in lab[ann[0]] if i in lab[ann[1]] and i in key and i in dmap)
    vx = {i: verdict(lab[ann[0]][i]) for i in both}
    vy = {i: verdict(lab[ann[1]][i]) for i in both}
    dis = [i for i in both if vx[i] != vy[i]]
    resolved = {i: verdict(lab[adj][i]) for i in dis if adj and i in lab.get(adj, {})}
    rep["agreement"] = {
        "n_colabelled": len(both), "n_agree": len(both) - len(dis), "n_disagree": len(dis),
        "n_adjudicated": len(resolved), "n_unresolved": len(dis) - len(resolved),
        "verdict_kappa": kappa([str(vx[i]) for i in both], [str(vy[i]) for i in both]),
        "q1_kappa": kappa([lab[ann[0]][i].get("q1") for i in both], [lab[ann[1]][i].get("q1") for i in both]),
        "q2_kappa": kappa([lab[ann[0]][i].get("q2") for i in both], [lab[ann[1]][i].get("q2") for i in both]),
    }

    for kb in a.budgets:
        w_all = e2[str(kb)]["systems"]
        out_b = {"weights_from": f"k={kb}", "systems": {}}
        for sysname in sorted({key[i]["system"] for i in both}):
            if sysname not in w_all:
                continue
            w = {st: w_all[sysname][f"{st}_rate"] for st in STATUSES}
            sel = [i for i in both if key[i]["system"] == sysname
                   and (dmap[i].get("rank") is not None and dmap[i]["rank"] <= kb)]
            base = lambda i, v: {"status": dmap[i]["status"], "doc_key": dmap[i]["doc_key"], "v": bool(v)}
            entry = {"weights": {k: round(v, 4) for k, v in w.items()}, "n_items_in_budget": len(sel),
                     "n_by_status": dict(Counter(dmap[i]["status"] for i in sel)),
                     "n_documents": len({dmap[i]["doc_key"] for i in sel}),
                     "per_annotator": {n: cluster_boot([base(i, verdict(lab[n][i])) for i in sel if i in lab[n]],
                                                       w, rng, a.n_boot) for n in ann}}
            agree_items = [base(i, vx[i]) for i in sel if vx[i] == vy[i]]
            entry["agreement_only"] = {**cluster_boot(agree_items, w, rng, a.n_boot),
                                       "n_dropped_disagreements": sum(1 for i in sel if vx[i] != vy[i])}
            if adj is not None:
                adj_items = [base(i, vx[i] if vx[i] == vy[i] else resolved[i])
                             for i in sel if vx[i] == vy[i] or i in resolved]
                entry["adjudicated"] = {**cluster_boot(adj_items, w, rng, a.n_boot),
                                        "n_unresolved_excluded": sum(1 for i in sel
                                                                     if vx[i] != vy[i] and i not in resolved)}
                for nm, fill in (("adjudicated_lower", False), ("adjudicated_upper", True)):
                    fitems = [base(i, vx[i] if vx[i] == vy[i] else resolved.get(i, fill)) for i in sel]
                    entry[nm] = cluster_boot(fitems, w, rng, a.n_boot)
                ao, ad = entry["agreement_only"]["point"], entry["adjudicated"]["point"]
                entry["selection_gap"] = None if (ao is None or ad is None) else round(ao - ad, 6)
            out_b["systems"][sysname] = entry
        rep["budgets"][str(kb)] = out_b

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "d1b_adjudicated_precision.json").write_text(json.dumps(rep, indent=1))
    g = rep["agreement"]
    print(f"[d1b] roles: annotators {ann}, adjudicator {adj}", flush=True)
    print(f"[d1b] co-labelled {g['n_colabelled']}: agree {g['n_agree']}, disagree {g['n_disagree']} "
          f"(adjudicated {g['n_adjudicated']}, unresolved {g['n_unresolved']}), verdict kappa {g['verdict_kappa']}",
          flush=True)
    for kb in a.budgets:
        print(f"--- budget k<={kb}", flush=True)
        for s, e in rep["budgets"][str(kb)]["systems"].items():
            f = lambda d: ("n/a" if not d or d.get("point") is None else
                           f"{d['point']:.3f}" + (f" [{d['ci'][0]:.3f},{d['ci'][1]:.3f}]" if d.get("ci") else " [--]"))
            print(f"[d1b k<={kb}] {s:30s} n={e['n_items_in_budget']:3d} docs={e['n_documents']:3d} | "
                  f"agreement-only {f(e.get('agreement_only'))} (dropped {e['agreement_only'].get('n_dropped_disagreements')}) | "
                  f"adjudicated {f(e.get('adjudicated'))} | lower {f(e.get('adjudicated_lower'))} | "
                  f"gap {e.get('selection_gap')}", flush=True)
    print(f"[d1b] written {out / 'd1b_adjudicated_precision.json'}", flush=True)


if __name__ == "__main__":
    main()
