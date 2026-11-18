#!/usr/bin/env python3
"""MARS-C B5 -- a CONSERVATIVE MACHINE-PROXY LOWER BOUND on the corrected emitter's useful yield.

Registered in research/marsc_strengthen_20260918/PREREG.md, Wave B / B5.

Why this file exists
--------------------
The paper's only semantic precision evidence annotates an emission the pipeline NO LONGER PRODUCES. The measured
drift (r02_human_rank/emitted_overlap.json) is per-system corrected-top-10 overlap 0.5907 / 0.6637 / 0.6322
(MARS-2 1.0), and only 51 / 53 / 48 / 70 of the 100 annotated items survive into the corrected top-10.
Re-analysing the old labels on the surviving subset is a SELECTION ON RANK, not a measurement, so it is not done
here and must not be done anywhere.

What IS defensible with no new human labels, and exactly why
-----------------------------------------------------------
On the 63,343 human-ACU cells of r04_matched/e0_judge/relabel_report.json the coverage judge's POSITIVE omission
verdict is near-exact (precision_on_judge_omitted = 0.9900) while its recall is only 0.5535, and its SUPPORT
answer is the measured failure mode (raw agreement 0.7163, Cohen kappa 0.4727). Therefore:

    "the judge answers OMITTED"  ==>  a CONSERVATIVE LOWER BOUND on useful yield, and NOTHING ELSE.

It is NOT precision. It is not an estimate of precision. It may not be printed as precision, nor compared against
a precision target, nor used as the numerator of any ratio whose denominator is an emission count called
"precision". Every field this script writes that carries the bound is named `lb_*` and the report carries the
caveat inline so that the number cannot travel without it.

The judgments already exist for the corrected emitter: r07_crossed/{tasks.jsonl, judgments/gemma_coverage_shard*}
(40,586 top-10 tasks over 1,388 pairs, 0 unjudged slots). This script only re-aggregates them, correctly.

What is different from mc_crossed_precision.py (and why this is a separate file)
-------------------------------------------------------------------------------
1. `mc_crossed_precision.py`'s resampling variable is named `by_doc` but is keyed by `pair_id`; it never imports
   `common` and never calls `doc_key`. RoSE contributes 8-12 summaries of ONE source document, so its intervals
   are anti-conservative by up to sqrt(n_pairs / n_docs). Here the bootstrap clusters on `common.doc_key(source)`,
   matching the main evaluator.
2. That file estimates a RATE among emitted units. The B5 estimand is a COUNT per document -- the lower bound on
   the proposal's per-document useful yield U_d -- reported with the rate beside it.
3. The observed paired contrast is computed directly and reported separately from the bootstrap mean (the
   `paired_recall.mean` trap that produced a retracted claim in this project).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402

CAVEAT = ("judge-CONFIRMED omissions only. This is a CONSERVATIVE LOWER BOUND on useful yield and is NOT "
          "precision: the coverage judge's positive omission verdict is near-exact (precision_on_judge_omitted "
          "0.9900 on 63,343 human-ACU cells) but its recall is only 0.5535, so the true useful yield is "
          "strictly larger by an unknown amount. Never report this quantity as a precision, never compare it "
          "against a precision target, never report it without the recall shortfall beside it.")

TRANSPORT = ("TRANSPORTABILITY CAVEAT. precision_on_judge_omitted was measured on HUMAN-ACU cells (reference "
             "atomic content units judged against the candidate). The units bounded here are INVENTORY units "
             "(gemma+ent / distilled / MARS-2 extractions from the source). The bound is conservative only "
             "under the assumption that the judge's positive-verdict precision transports from the ACU "
             "population to the inventory population. That assumption is stated, not tested; no human labels "
             "on the corrected emission exist to test it. This is exactly the gap Wave D / D1 would close.")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ci(d: np.ndarray) -> list:
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def _stat(obs: float, boot: np.ndarray) -> dict:
    """Observed point estimate first; the bootstrap mean is reported SEPARATELY and never substituted for it."""
    return {"point_observed": float(obs), "ci95": _ci(boot), "boot_mean_not_the_estimate": float(boot.mean())}


def _contrast(obs: float, boot: np.ndarray) -> dict:
    lo, hi = _ci(boot)
    frac_neg = float((boot <= 0).mean())
    p = float(min(1.0, 2 * min(frac_neg, 1 - frac_neg)))
    return {"observed_contrast": float(obs), "ci95": [lo, hi],
            "boot_mean_not_the_observed_contrast": float(boot.mean()),
            "boot_two_sided_p_approx": p, "excludes_zero": bool(lo > 0 or hi < 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="r07_crossed/tasks.jsonl (carries source, pair_id, rank, status)")
    ap.add_argument("--judgments", nargs="+", required=True, help="glob(s) of coverage judgment shards")
    ap.add_argument("--systems", nargs="+", required=True, help="system names to bound (must appear in tasks.rank)")
    ap.add_argument("--primary", required=True, help="system every paired contrast is taken against")
    ap.add_argument("--max-rank", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.5, help="judge p_omitted threshold for CONFIRMED")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--judge-validation", default="", help="r04_matched/e0_judge/relabel_report.json")
    ap.add_argument("--emission-drift", default="", help="r02_human_rank/emitted_overlap.json")
    ap.add_argument("--human-json", default="", help="r02_human_rank/human_precision_ranked.json -- used ONLY "
                                                     "for the lower-bound sanity flag, never as a comparator")
    ap.add_argument("--limit", type=int, default=0, help="debug only: read at most N task rows")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tasks_p = Path(a.tasks)
    # ---- judgments. r07_crossed shards are written by `mc_judge_precision.py run`, whose key is `p_b`.
    # `scripts/plan2026/b1_judge_labels.py` writes `p_omitted` instead; the two formats are NOT interchangeable,
    # so the key actually present is detected and recorded rather than assumed.
    ans, key_seen, files = {}, set(), []
    for g in a.judgments:
        for f in sorted(glob.glob(g)):
            files.append(f)
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                if r.get("question") != "coverage":
                    continue
                if "p_b" in r:
                    key_seen.add("p_b"); ans[r["task_id"]] = float(r["p_b"])
                elif "p_omitted" in r:
                    key_seen.add("p_omitted"); ans[r["task_id"]] = float(r["p_omitted"])
                else:
                    raise SystemExit(f"[b5] judgment row in {f} has neither p_b nor p_omitted: {sorted(r)}")
    if not files:
        raise SystemExit(f"[b5] no judgment file matched {a.judgments}")
    if len(key_seen) != 1:
        raise SystemExit(f"[b5] MIXED judgment formats {sorted(key_seen)} -- refusing to aggregate")
    prob_key = key_seen.pop()
    print(f"[b5] {len(files)} judgment shards, {len(ans)} coverage answers, probability key '{prob_key}'", flush=True)

    # ---- tasks -> per (system, pair) emitted slots, with the document cluster of each pair
    systems = list(dict.fromkeys(a.systems))
    if a.primary not in systems:
        raise SystemExit(f"[b5] --primary {a.primary} is not among --systems {systems}")
    doc_of_pair, emitted, confirmed, texts = {}, defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(set))
    n_tasks = n_missing = 0
    seen_systems = defaultdict(int)
    with open(tasks_p, encoding="utf-8") as fh:
        for line in fh:
            t = json.loads(line); n_tasks += 1
            pid = t["pair_id"]
            if pid not in doc_of_pair:
                doc_of_pair[pid] = common.doc_key(t["source"])
            for sysname, rank in t["rank"].items():
                seen_systems[sysname] += 1
                if sysname not in systems or rank >= a.max_rank:
                    continue
                p = ans.get(t["task_id"])
                if p is None:
                    n_missing += 1
                    continue
                emitted[sysname][pid] += 1
                if p >= a.threshold:
                    confirmed[sysname][pid] += 1
                    texts[sysname][doc_of_pair[pid]].add(t["unit_text"].strip().lower())
            if a.limit and n_tasks >= a.limit:
                break
    missing_sys = [s for s in systems if s not in seen_systems]
    if missing_sys:
        raise SystemExit(f"[b5] systems absent from tasks.rank: {missing_sys}; present: {sorted(seen_systems)}")
    if n_missing and not a.limit:
        print(f"[b5] WARNING {n_missing} emitted (system, unit) slots carry no judgment", flush=True)

    # ---- the common pair set: every contrast is paired, so it is taken on pairs every system emitted into
    pairs_all = sorted(set.intersection(*[set(emitted[s]) for s in systems]))
    docs_all = sorted({doc_of_pair[p] for p in pairs_all})
    pairs_by_doc = defaultdict(list)
    for p in pairs_all:
        pairs_by_doc[doc_of_pair[p]].append(p)
    print(f"[b5] {n_tasks} task rows; {len(pairs_all)} pairs carry every system at rank<{a.max_rank}; "
          f"{len(docs_all)} source documents (mean {len(pairs_all) / max(1, len(docs_all)):.2f} summaries/doc)", flush=True)

    # ---- per-document vectors (one entry per document, so documents weigh equally)
    D = len(docs_all)
    vec = {}
    for s in systems:
        per_list = np.zeros(D); per_rate = np.zeros(D); per_doc_total = np.zeros(D)
        per_distinct = np.zeros(D); any_hit = np.zeros(D)
        for i, d in enumerate(docs_all):
            ps = pairs_by_doc[d]
            c = np.array([confirmed[s][p] for p in ps], dtype=float)
            e = np.array([emitted[s][p] for p in ps], dtype=float)
            per_list[i] = c.mean()
            per_rate[i] = (c.sum() / e.sum()) if e.sum() else 0.0
            per_doc_total[i] = c.sum()
            per_distinct[i] = len(texts[s][d])
            any_hit[i] = float((c > 0).any())
        vec[s] = {"lb_confirmed_per_emitted_list": per_list, "lb_confirmed_rate_per_emitted_slot": per_rate,
                  "lb_confirmed_per_document_all_summaries": per_doc_total,
                  "lb_distinct_confirmed_units_per_document": per_distinct,
                  "share_documents_with_at_least_one_confirmed": any_hit}
    metrics = list(vec[systems[0]])

    rng = np.random.default_rng(a.seed)
    boot_idx = [rng.choice(np.arange(D), size=D, replace=True) for _ in range(a.n_boot)]

    rep = {
        "registered": "research/marsc_strengthen_20260918/PREREG.md Wave B / B5",
        "ESTIMAND_IS_A_LOWER_BOUND": True,
        "MUST_NOT_BE_REPORTED_AS_PRECISION": CAVEAT,
        "TRANSPORTABILITY_CAVEAT": TRANSPORT,
        "design": {
            "estimand": "count of judge-CONFIRMED omissions among the corrected top-k emission, per document",
            "confirmed_rule": f"coverage judge {prob_key} >= {a.threshold}",
            "max_rank": a.max_rank, "n_boot": a.n_boot, "seed": a.seed,
            "bootstrap_cluster": "common.doc_key(source) = sha1(source[:200])[:16]",
            "why_not_pair_id": "RoSE contributes 8-12 summaries of one source document; clustering on pair_id "
                               "(as mc_crossed_precision.py does) is anti-conservative by up to sqrt(n_pairs/n_docs)",
            "selection_on_rank_refused": "the 100 pre-drift human items are NOT re-analysed on the surviving "
                                         "subset; that would be a selection on rank, not a measurement",
        },
        "inputs": {"tasks": {"path": str(tasks_p), "sha256": _sha256(tasks_p)},
                   "judgment_shards": [{"path": f, "sha256": _sha256(Path(f))} for f in files],
                   "probability_key": prob_key},
        "counts": {"task_rows": n_tasks, "coverage_answers": len(ans), "unjudged_emitted_slots": n_missing,
                   "pairs_common": len(pairs_all), "documents_common": len(docs_all),
                   "summaries_per_document_mean": len(pairs_all) / max(1, len(docs_all)),
                   "anticonservatism_factor_if_clustered_on_pair_id":
                       float(np.sqrt(len(pairs_all) / max(1, len(docs_all))))},
        "metric_glossary": {
            "lb_confirmed_per_emitted_list": "LOWER BOUND: judge-confirmed omissions in one top-k list, "
                                             "averaged within document then across documents",
            "lb_confirmed_rate_per_emitted_slot": "LOWER BOUND: confirmed / emitted slots, per document",
            "lb_confirmed_per_document_all_summaries": "LOWER BOUND: confirmed omissions summed over every "
                                                       "summary of the document",
            "lb_distinct_confirmed_units_per_document": "LOWER BOUND: distinct confirmed unit texts per document "
                                                        "(deduplicated across that document's summaries)",
            "share_documents_with_at_least_one_confirmed": "share of documents where the emission surfaces at "
                                                           "least one judge-confirmed omission",
        },
        "systems": {}, "paired_contrasts_vs_primary": {}, "primary": a.primary,
    }

    for s in systems:
        entry = {"n_emitted_units_on_common_pairs": int(sum(emitted[s][p] for p in pairs_all)),
                 "n_confirmed_units_on_common_pairs": int(sum(confirmed[s][p] for p in pairs_all)),
                 "n_pairs_any": len(emitted[s]), "n_pairs_common": len(pairs_all)}
        for m in metrics:
            v = vec[s][m]
            boot = np.array([v[b].mean() for b in boot_idx])
            entry[m] = _stat(v.mean(), boot)
            entry[m]["per_document_median"] = float(np.median(v))
            entry[m]["per_document_iqr"] = [float(np.percentile(v, 25)), float(np.percentile(v, 75))]
        rep["systems"][s] = entry
        e = entry["lb_confirmed_per_emitted_list"]; r = entry["lb_confirmed_rate_per_emitted_slot"]
        print(f"[b5] {s:32s} LOWER BOUND {e['point_observed']:.4f} confirmed omissions per top-{a.max_rank} list "
              f"[{e['ci95'][0]:.4f},{e['ci95'][1]:.4f}]  (rate {r['point_observed']:.4f} per emitted slot)", flush=True)

    for s in systems:
        if s == a.primary:
            continue
        out = {}
        for m in metrics:
            dv = vec[a.primary][m] - vec[s][m]
            boot = np.array([dv[b].mean() for b in boot_idx])
            out[m] = _contrast(dv.mean(), boot)
        rep["paired_contrasts_vs_primary"][f"{a.primary}_minus_{s}"] = out
        c = out["lb_confirmed_per_emitted_list"]
        print(f"[b5 contrast] {a.primary} - {s:28s} {c['observed_contrast']:+.4f} "
              f"[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] p~{c['boot_two_sided_p_approx']:.3f}", flush=True)

    # ---- the numbers that must be printed BESIDE the bound, read from their artefacts, not hard-coded
    if a.judge_validation and Path(a.judge_validation).exists():
        jv = json.load(open(a.judge_validation))
        rep["judge_validation_printed_beside_the_bound"] = {
            "source": a.judge_validation, "sha256": _sha256(Path(a.judge_validation)),
            **jv.get("judge_vs_human_cells", {}),
            "reading": "the bound omits the human-omitted facts the judge missed; with recall_on_human_omitted "
                       "= r, a crude first-order back-of-envelope on the same population would be bound / r, "
                       "which is NOT reported as an estimate here because r was measured on a different "
                       "(human-ACU) population."}
        jc = jv.get("judge_vs_human_cells", {})
        if "recall_on_human_omitted" in jc:
            print(f"[b5] recall shortfall printed beside the bound: judge recall on human-omitted "
                  f"{jc['recall_on_human_omitted']:.4f}, positive precision {jc.get('precision_on_judge_omitted', float('nan')):.4f} "
                  f"(n={jc.get('n')} human-ACU cells, kappa {jc.get('cohen_kappa', float('nan')):.4f})", flush=True)
    else:
        rep["judge_validation_printed_beside_the_bound"] = {"status": "ABSENT -- bound reported without its "
                                                                      "shortfall, which is not acceptable for publication"}
    # ---- sanity flag on the lower-bound reading itself.
    # A LOWER BOUND must not sit above a human estimate of the same quantity. The only human estimate available
    # is r02's agreement-conditioned rate on the PRE-DRIFT emission, and it is doubly compromised: (i) the
    # emission drifted (only 51/53/48/70 of the 100 annotated items survive into the corrected top-10), and
    # (ii) PREREG A3b records that annotator 2 answered `A` on all 400 items, so "both_agree" is a filter
    # against a degenerate second rater. So this is NOT a comparator and NOT a validation -- it is a flag. If
    # the judged rate sits above the human interval, the transportability assumption behind the word "bound"
    # is under strain and the manuscript must say so rather than print the bound unqualified.
    if a.human_json and Path(a.human_json).exists():
        hj = json.load(open(a.human_json))
        hs = hj.get("budgets", {}).get(str(a.max_rank), {}).get("systems", {})
        flags = {}
        for s in systems:
            ba = (hs.get(s) or {}).get("both_agree") or {}
            if ba.get("point") is None:
                flags[s] = {"status": "no human cell at this budget"}
                continue
            jr = rep["systems"][s]["lb_confirmed_rate_per_emitted_slot"]["point_observed"]
            hi = (ba.get("ci") or [None, None])[1]
            flags[s] = {"judge_confirmed_rate_corrected_emission": jr,
                        "human_agreement_conditioned_rate_PRE_DRIFT": ba["point"], "human_ci95": ba.get("ci"),
                        "n_annotated_items_in_budget": (hs.get(s) or {}).get("n_items_in_budget"),
                        "judged_rate_above_human_ci_upper": bool(hi is not None and jr > hi)}
            if flags[s]["judged_rate_above_human_ci_upper"]:
                print(f"[b5 FLAG] {s}: judged rate {jr:.4f} exceeds the upper end of the pre-drift human "
                      f"interval {ba.get('ci')} -- the LOWER-BOUND reading is under strain for this cell", flush=True)
        rep["lower_bound_sanity_flag"] = {
            "source": a.human_json, "sha256": _sha256(Path(a.human_json)),
            "NOT_A_COMPARATOR": "the human cells are on the PRE-DRIFT emission and are agreement-conditioned "
                                "against a degenerate second annotator (PREREG A3b: annotator 2 answered `A` "
                                "on all 400 items, kappa exactly 0.0). Neither side identifies the other.",
            "reading": "a flag only. Where `judged_rate_above_human_ci_upper` is true, the transportability "
                       "assumption behind the word BOUND is under strain for that cell and the manuscript must "
                       "report the bound with that strain stated, or not at all.",
            "per_system": flags,
            "any_flagged": bool(any(v.get("judged_rate_above_human_ci_upper") for v in flags.values()))}

    if a.emission_drift and Path(a.emission_drift).exists():
        rep["emission_drift_that_motivates_this_block"] = {
            "source": a.emission_drift, "sha256": _sha256(Path(a.emission_drift)),
            **json.load(open(a.emission_drift))}

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "b5_yield_lower_bound.json").write_text(json.dumps(rep, indent=1))
    print(f"[b5] written {out / 'b5_yield_lower_bound.json'}", flush=True)


if __name__ == "__main__":
    main()
