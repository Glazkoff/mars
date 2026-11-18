#!/usr/bin/env python3
"""MARS-C strengthening campaign, Wave A blocks A2 / A3 / A3b / A8 (research/marsc_strengthen_20260918/PREREG.md).

Four repairs, one module. None of them carries a bar: a repair cannot be allowed to pass or fail, the
corrected number is the number.

A2  `mc_crossed_precision.py` builds its resampling unit on `pair_id`. Its variable is *named* `by_doc`
    (L63/73/75/77-79) but is keyed by pair, and the file never imports `common`, so `doc_key` is never called.
    RoSE contributes 8-12 summaries of ONE source document as nominally independent units (1388 pairs over
    ~151 documents), so every interval in `r07_crossed/analysis/crossed_precision.json` is anti-conservative by
    up to sqrt(n_pairs / n_docs) on width. This module re-estimates the identical estimand with the resampling
    unit set to `common.doc_key(source)` -- the unit the main evaluator `mc_e2_eval.py` already uses -- and
    prints the pair-level interval beside it so the inflation factor is visible. The OBSERVED point is
    recomputed from the data and reported separately from the bootstrap mean (the `paired_recall.mean`
    bootstrap-mean-as-observed-contrast confusion already retracted one claim in this project).

A3  `mc_human_rank.py` resamples annotated items within status strata (`boot_ci` L84-97) and reports Wilson
    intervals on item counts (`wilson` L64-70). Both treat 8-12 summaries of one document as independent. This
    module re-estimates with the document as the resampling unit, reports the distinct-cluster count beside
    every interval, and -- because at rank <= 10 a system carries only ~37-53 items over ~36-42 documents -- it
    detects a degenerate percentile bootstrap explicitly and always prints a Wilson fallback computed on the
    CLUSTER count as well as on the item count. Nothing is hidden: item-level and cluster-level intervals are
    emitted side by side for every cell.

A3b A disclosure. The 400-item blinded human study is reported as a two-annotator study with an agreement
    filter. This mode counts each annotator's answers on Q1 (support) and Q2 (coverage) separately, builds the
    full 2x2 contingency tables, and computes Cohen's kappa for Q1, Q2 and the composite verdict, so that a
    reader can recompute every number from the printed counts. If one rater is constant on a question, its
    kappa is exactly 0 by construction and the agreement filter on that question carries no information.

A8  The confidence-interval site audit. Scans the MARS-C code, the MARS-2/MARS-3 gate code and
    scripts/plan2026/ for interval-emitting sites, joins each to a hand-verified classification of its
    resampling unit (recorded here by file + anchor, not by line number, so the table survives edits), and
    reports any site the table does not cover as UNCLASSIFIED rather than silently dropping it.

Every mode writes JSON under --out and prints the numbers it wrote.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402  (doc_key / load_jsonl -- the same helpers the main evaluator uses)

STATUSES = ("unknown", "hit", "false_alert")


# --------------------------------------------------------------------------------------------- helpers
def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def wilson(k: int, n: int, z: float = 1.959963985) -> list[float] | None:
    """Verbatim copy of mc_human_rank.wilson (L64-70); reproduced here so this module is self-contained."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def boot_p(draws: np.ndarray) -> float:
    """Two-sided bootstrap p-value against 0 with the (k+1)/(B+1) correction (the b0_recipe convention)."""
    b = len(draws)
    if b == 0:
        return float("nan")
    lo = (float((draws <= 0).sum()) + 1) / (b + 1)
    hi = (float((draws >= 0).sum()) + 1) / (b + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def ci_of(draws: np.ndarray) -> list[float]:
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def degeneracy(draws: np.ndarray) -> dict:
    """A percentile bootstrap on ~37 clusters can pile on a boundary. Say so rather than printing [1.0, 1.0]."""
    lo, hi = ci_of(draws)
    uniq = int(len(np.unique(np.round(draws, 12))))
    return {"ci_width": float(hi - lo), "n_distinct_draw_values": uniq,
            "frac_draws_at_0": float((draws <= 1e-12).mean()), "frac_draws_at_1": float((draws >= 1 - 1e-12).mean()),
            "degenerate": bool(hi - lo < 1e-9 or uniq < 5)}


def cluster_index(keys: list, cluster_of: dict) -> tuple[list, dict]:
    """(sorted cluster ids, cluster id -> np.ndarray of positions in `keys`)."""
    pos = defaultdict(list)
    for i, k in enumerate(keys):
        pos[cluster_of[k]].append(i)
    cl = sorted(pos)
    return cl, {c: np.asarray(pos[c], dtype=int) for c in cl}


def cluster_draws(cl: list, idx_by: dict, rng, n_boot: int) -> list[np.ndarray]:
    """n_boot resampled position vectors, resampling CLUSTERS with replacement (the mc_e2_eval.py convention:
    `pick = rng.choice(docs, ...); sub = [p for dd in pick for p in by_doc[dd]]`)."""
    arr = np.asarray(cl, dtype=object)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(arr), size=len(arr), replace=True)
        out.append(np.concatenate([idx_by[cl[i]] for i in pick]))
    return out


# ============================================================================================== A2
def run_a2(a) -> None:
    """The R07 crossed inventory x verifier design, re-estimated with the document as the resampling unit."""
    want_sys = set()
    cells = {}
    for spec in a.cells:
        key, sysname = spec.split("=", 1)
        inv, ver = key.split(":", 1)
        cells[(inv, ver)] = sysname
        want_sys.add(sysname)
    want_sys |= set(a.extra_systems)
    systems = sorted(want_sys)

    ans = {}
    n_judg_files = 0
    for g in a.judgments:
        for f in sorted(glob.glob(g)):
            n_judg_files += 1
            for r in load_jsonl(Path(f)):
                if r.get("question") == "coverage":
                    ans[r["task_id"]] = float(r["p_b"])
    assert n_judg_files > 0, f"no judgment file matched {a.judgments}"

    # Stream tasks.jsonl: it is ~137 MB because every row carries the full source and candidate.
    by_sys_pair = defaultdict(lambda: defaultdict(list))   # system -> pair_id -> [judged omitted]
    doc_of_pair, docfield_of_pair = {}, {}
    n_tasks = n_missing = 0
    with open(a.tasks, encoding="utf-8") as fh:
        for line in fh:
            t = json.loads(line)
            n_tasks += 1
            pid = t["pair_id"]
            if pid not in doc_of_pair:
                doc_of_pair[pid] = common.doc_key(t["source"])
                parts = pid.split(":")
                docfield_of_pair[pid] = parts[2] if len(parts) > 2 else ""
            for sysname, rank in t["rank"].items():
                if sysname not in want_sys or rank >= a.max_rank:
                    continue
                p = ans.get(t["task_id"])
                if p is None:
                    n_missing += 1
                    continue
                by_sys_pair[sysname][pid].append(p >= 0.5)
    print(f"[a2] {n_tasks} tasks, {len(ans)} coverage answers from {n_judg_files} judgment files, "
          f"{n_missing} unjudged (system, unit) slots", flush=True)

    pairs_all = sorted(set.intersection(*[set(by_sys_pair[s]) for s in systems])) if systems else []
    assert pairs_all, "no pair carries every requested system"
    per_pair = {s: np.array([float(np.mean(by_sys_pair[s][p])) for p in pairs_all]) for s in systems}
    docs, idx_by_doc = cluster_index(pairs_all, doc_of_pair)

    # Independent cross-check of the clustering: for RoSE the third pair_id field is the source-document id,
    # so doc_key(source) must partition the pairs exactly the same way. If it does not, say so loudly.
    _dfs, _ = cluster_index(pairs_all, docfield_of_pair)
    cross = defaultdict(set)
    for p in pairs_all:
        cross[docfield_of_pair[p]].add(doc_of_pair[p])
    split_docfields = sorted(k for k, v in cross.items() if len(v) > 1)
    sizes = np.array([len(idx_by_doc[d]) for d in docs], dtype=float)
    design_effect = math.sqrt(len(pairs_all) / len(docs))
    print(f"[a2] {len(pairs_all)} pairs over {len(docs)} documents "
          f"(mean {sizes.mean():.2f}, min {int(sizes.min())}, max {int(sizes.max())}, "
          f"{int((sizes > 1).sum())} multi-summary); naive sqrt(n_pairs/n_docs) = {design_effect:.2f}x", flush=True)
    print(f"[a2] pair_id document-field partition: {len(_dfs)} ids, {len(docs)} doc_key clusters, "
          f"{len(split_docfields)} document-fields split across doc_keys", flush=True)

    rng = np.random.default_rng(a.seed)
    doc_idx = cluster_draws(docs, idx_by_doc, rng, a.n_boot)
    rng_pair = np.random.default_rng(a.seed)
    n_p = len(pairs_all)
    pair_idx = [rng_pair.choice(n_p, size=n_p, replace=True) for _ in range(a.n_boot)]

    rep = {"registered": "research/marsc_strengthen_20260918/PREREG.md A2",
           "repairs": "research/marsc_20260916/code/mc_crossed_precision.py L63/73/75/77-79 "
                      "(variable named by_doc, keyed by pair_id; common.doc_key never called)",
           "bar": "none -- this is a repair; the corrected interval is the interval",
           "estimand": "share of a system's top-k emitted units the validated coverage judge answers omitted",
           "note": "judge-estimated omission rate among emitted units, NOT a human confirmed-omission rate",
           "resampling_unit": "common.doc_key(source[:200]) -- the unit mc_e2_eval.py already uses",
           "max_rank": a.max_rank, "n_boot": a.n_boot, "seed": a.seed,
           "n_tasks": n_tasks, "n_coverage_answers": len(ans), "unjudged_slots": n_missing,
           "n_pairs": len(pairs_all), "n_docs": len(docs),
           "pairs_per_doc": {"mean": float(sizes.mean()), "min": int(sizes.min()), "max": int(sizes.max()),
                             "n_multi_summary_docs": int((sizes > 1).sum())},
           "naive_design_effect_sqrt_npairs_over_ndocs": design_effect,
           "pair_id_docfield_crosscheck": {"n_docfields": len(_dfs), "n_doc_keys": len(docs),
                                           "docfields_split_across_doc_keys": split_docfields[:20],
                                           "partition_identical": bool(len(_dfs) == len(docs) and not split_docfields)},
           "systems": {}, "crossed": {}}

    for s in systems:
        obs = float(per_pair[s].mean())
        d_doc = np.array([per_pair[s][b].mean() for b in doc_idx])
        d_pair = np.array([per_pair[s][b].mean() for b in pair_idx])
        ci_d, ci_p = ci_of(d_doc), ci_of(d_pair)
        rep["systems"][s] = {
            "judged_omission_rate": obs,
            "ci_doc_clustered": ci_d, "ci_pair_level_as_published": ci_p,
            "boot_mean_doc_clustered": float(d_doc.mean()),
            "width_inflation_doc_over_pair": float((ci_d[1] - ci_d[0]) / (ci_p[1] - ci_p[0])) if ci_p[1] > ci_p[0] else None,
            "n_units": int(sum(len(v) for v in by_sys_pair[s].values())), "n_pairs": len(pairs_all), "n_docs": len(docs)}
        r = rep["systems"][s]
        print(f"[a2] {s:32s} judged-omitted {obs:.4f} doc [{ci_d[0]:.4f},{ci_d[1]:.4f}] "
              f"pair [{ci_p[0]:.4f},{ci_p[1]:.4f}] x{r['width_inflation_doc_over_pair']:.2f} "
              f"over {r['n_units']} emitted units", flush=True)

    invs = sorted({k[0] for k in cells})
    vers = sorted({k[1] for k in cells})
    if len(invs) == 2 and len(vers) == 2 and len(cells) == 4:
        i0, i1 = invs
        v0, v1 = vers

        def eff(fn):
            obs = float(fn({k: per_pair[v] for k, v in cells.items()}))
            d_doc = np.array([fn({k: per_pair[v][b] for k, v in cells.items()}) for b in doc_idx])
            d_pair = np.array([fn({k: per_pair[v][b] for k, v in cells.items()}) for b in pair_idx])
            ci_d, ci_p = ci_of(d_doc), ci_of(d_pair)
            return {"point_observed": obs,
                    "ci_doc_clustered": ci_d, "ci_pair_level_as_published": ci_p,
                    "boot_mean_doc_clustered": float(d_doc.mean()),
                    "p_two_sided_doc_clustered": boot_p(d_doc),
                    "crosses_zero_doc_clustered": bool(ci_d[0] <= 0.0 <= ci_d[1]),
                    "crosses_zero_pair_level": bool(ci_p[0] <= 0.0 <= ci_p[1]),
                    "width_inflation_doc_over_pair": float((ci_d[1] - ci_d[0]) / (ci_p[1] - ci_p[0])) if ci_p[1] > ci_p[0] else None}

        rep["crossed"] = {
            "inventories": invs, "verifiers": vers,
            f"inventory_effect_{i1}_minus_{i0}": eff(lambda c: np.mean([(c[(i1, v)] - c[(i0, v)]).mean() for v in vers])),
            f"verifier_effect_{v1}_minus_{v0}": eff(lambda c: np.mean([(c[(i, v1)] - c[(i, v0)]).mean() for i in invs])),
            "interaction": eff(lambda c: ((c[(i1, v1)] - c[(i0, v1)]).mean() - (c[(i1, v0)] - c[(i0, v0)]).mean()))}
        for k, v in rep["crossed"].items():
            if isinstance(v, dict) and "point_observed" in v:
                print(f"[a2 crossed] {k:44s} {v['point_observed']:+.4f} "
                      f"doc [{v['ci_doc_clustered'][0]:+.4f},{v['ci_doc_clustered'][1]:+.4f}] p={v['p_two_sided_doc_clustered']:.4f} "
                      f"| pair [{v['ci_pair_level_as_published'][0]:+.4f},{v['ci_pair_level_as_published'][1]:+.4f}] "
                      f"| crosses zero: doc={v['crosses_zero_doc_clustered']} pair={v['crosses_zero_pair_level']}", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "a2_crossed_precision_docclustered.json").write_text(json.dumps(rep, indent=1))
    print(f"[a2] written {out / 'a2_crossed_precision_docclustered.json'}", flush=True)


# ============================================================================================== A3
def build_rank_index(emitted_files: list[Path]) -> dict:
    """(system, pair_id) -> ordered emitted list. Copied from mc_human_rank.build_rank_index (L37-45)."""
    idx: dict[tuple[str, str], list[dict]] = {}
    for f in emitted_files:
        for r in load_jsonl(f):
            key = (r["system"], r["pair_id"])
            if r.get("emitted") and key not in idx:
                idx[key] = r["emitted"]
    return idx


def recover_rank(item: dict, idx: dict) -> dict | None:
    """Copied from mc_human_rank.recover_rank (L48-61). The only change is a defensive float() on both sides of
    the tie-break comparison. Verified on the artefacts this block reads (g6/human_sample_key_all.jsonl carries
    `score` as a JSON float, g3/emitted_validation.jsonl likewise), so the coercion is a no-op and the recovered
    ranks -- and therefore the point estimates -- are identical to the published ones."""
    em = idx.get((item["system"], item["pair_id"]))
    if not em:
        return None
    cands = [(i, e) for i, e in enumerate(em) if e["text"] == item["fact"]]
    if len(cands) > 1 and item.get("score") is not None:
        try:
            want = float(item["score"])
        except (TypeError, ValueError):
            want = None
        if want is not None:
            tight = [(i, e) for i, e in cands if e.get("score") is not None and abs(float(e["score"]) - want) < 1e-9]
            if tight:
                cands = tight
    if not cands:
        return None
    i, e = cands[0]
    return {"rank": i + 1, "status_dump": e["status"], "ambiguous": len(cands) > 1, "n_emitted": len(em)}


def weighted_precision(per_status: dict[str, list[bool]], w: dict[str, float]) -> float | None:
    """Copied from mc_human_rank.weighted_precision (L73-81). Unchanged: the POINT estimate is not repaired,
    only its interval."""
    have = [s for s in w if per_status.get(s)]
    if not have:
        return None
    z = sum(w[s] for s in have)
    if z <= 0:
        return None
    return float(sum((w[s] / z) * (sum(per_status[s]) / len(per_status[s])) for s in have))


def item_boot_ci(per_status: dict[str, list[bool]], w: dict[str, float], rng, n_boot: int) -> dict:
    """The PUBLISHED estimator (mc_human_rank.boot_ci L84-97): resample items inside each status stratum."""
    have = [s for s in w if per_status.get(s)]
    if not have:
        return {"point": None, "ci": None, "n_boot": 0}
    draws = []
    for _ in range(n_boot):
        rs = {s: list(rng.choice(per_status[s], size=len(per_status[s]), replace=True)) for s in have}
        v = weighted_precision(rs, w)
        if v is not None:
            draws.append(v)
    return {"point": weighted_precision(per_status, w),
            "ci": ci_of(np.asarray(draws)) if draws else None, "n_boot": len(draws)}


def doc_boot_ci(rows: list[tuple[str, str, bool]], w: dict[str, float], rng, n_boot: int) -> dict:
    """The REPAIR: resample source documents with replacement, then re-stratify inside the drawn documents.

    `rows` is [(doc_key, status, verdict)] over exactly the items that enter this cell. Strata that a draw does
    not populate drop out and `weighted_precision` renormalises over the strata that survive -- the same rule
    the published estimator uses when a stratum carries no annotations.
    """
    if not rows:
        return {"point": None, "ci": None, "n_boot": 0, "n_items": 0, "n_docs": 0}
    by_doc = defaultdict(list)
    for d, st, v in rows:
        by_doc[d].append((st, v))
    docs = sorted(by_doc)
    obs = defaultdict(list)
    for d, st, v in rows:
        obs[st].append(v)
    draws = []
    n_empty = 0
    for _ in range(n_boot):
        pick = rng.choice(len(docs), size=len(docs), replace=True)
        ps = defaultdict(list)
        for i in pick:
            for st, v in by_doc[docs[i]]:
                ps[st].append(v)
        val = weighted_precision(ps, w)
        if val is None:
            n_empty += 1
        else:
            draws.append(val)
    arr = np.asarray(draws)
    point = weighted_precision(obs, w)
    out = {"point": point, "ci": ci_of(arr) if len(arr) else None, "n_boot": int(len(arr)),
           "n_draws_undefined": n_empty, "n_items": len(rows), "n_docs": len(docs),
           "items_per_doc_max": int(max(len(v) for v in by_doc.values())),
           "n_by_status_items": {s: len(v) for s, v in sorted(obs.items())},
           "n_by_status_docs": {s: len({d for d, st, _ in rows if st == s}) for s in sorted(obs)}}
    if len(arr):
        out.update(degeneracy(arr))
    # Fallbacks that never degenerate: Wilson on the item count (as published) and Wilson on the CLUSTER count,
    # i.e. the binomial interval you would get if each source document contributed one independent observation.
    if point is not None:
        out["wilson_on_items"] = wilson(int(round(point * len(rows))), len(rows))
        out["wilson_on_clusters"] = wilson(int(round(point * len(docs))), len(docs))
    return out


def run_a3(a) -> None:
    rng = np.random.default_rng(a.seed)
    idx = build_rank_index([Path(f) for f in a.emitted])

    pair_source = {}
    for f in a.pair_source:
        for r in load_jsonl(Path(f)):
            if "source" in r:
                pair_source.setdefault(r["pair_id"], r["source"])
    assert pair_source, f"no pair_id -> source mapping loaded from {a.pair_source}"
    doc_of_pair = {p: common.doc_key(s) for p, s in pair_source.items()}

    key = {}
    for f in a.key:
        for r in load_jsonl(Path(f)):
            key[r["item_id"]] = r
    lab: dict[str, dict[str, dict]] = defaultdict(dict)
    for f in a.export:
        for r in load_jsonl(Path(f)):
            lab[r["annotator"]][r["item_id"]] = r
    annotators = sorted(lab)
    e2 = json.load(open(a.e2_json))["k"]

    joined, unjoined, no_doc = {}, [], []
    for iid, it in key.items():
        rk = recover_rank(it, idx)
        if rk is None:
            unjoined.append(iid)
            continue
        if it["pair_id"] not in doc_of_pair:
            no_doc.append(iid)
            continue
        joined[iid] = {**it, **rk, "doc": doc_of_pair[it["pair_id"]]}
    status_check = Counter((joined[i]["status"] == joined[i]["status_dump"]) for i in joined)

    rep = {"registered": "research/marsc_strengthen_20260918/PREREG.md A3",
           "repairs": "research/marsc_20260916/code/mc_human_rank.py boot_ci L84-97 (item-level stratified "
                      "bootstrap) and wilson L64-70 (item-count Wilson)",
           "bar": "none -- this is a repair; the corrected interval is the interval",
           "resampling_unit": "common.doc_key(source[:200])",
           "point_estimator_unchanged": "weighted_precision (mc_human_rank.py L73-81) copied verbatim",
           "emitted": list(a.emitted), "key": list(a.key), "export": list(a.export), "e2_json": a.e2_json,
           "n_boot": a.n_boot, "seed": a.seed,
           "n_key_items": len(key), "n_joined": len(joined), "n_unjoined": len(unjoined),
           "n_dropped_no_source_for_pair": len(no_doc), "dropped_no_source_item_ids": no_doc[:50],
           "unjoined_item_ids": unjoined[:50],
           "n_ambiguous_text_match": sum(1 for i in joined if joined[i]["ambiguous"]),
           "status_agrees_with_dump": {str(k): v for k, v in status_check.items()},
           "rank_histogram": dict(sorted(Counter(joined[i]["rank"] for i in joined).items())),
           "n_distinct_docs_all_joined": len({joined[i]["doc"] for i in joined}),
           "annotators": {n: len(lab[n]) for n in annotators},
           "budgets": {}}
    print(f"[a3] joined {len(joined)}/{len(key)} sampled items ({len(unjoined)} unjoined, {len(no_doc)} without a "
          f"source), over {rep['n_distinct_docs_all_joined']} distinct documents; "
          f"ambiguous text matches {rep['n_ambiguous_text_match']}, status agrees {rep['status_agrees_with_dump']}",
          flush=True)

    def verdict(r):
        return r["q1"] == "A" and r["q2"] == "B"

    for kb in a.budgets:
        w_all = e2[str(kb)]["systems"]
        out_b = {"weights_from": f"k={kb}", "systems": {}}
        for sysname in sorted({joined[i]["system"] for i in joined}):
            if sysname not in w_all:
                continue
            w = {"hit": w_all[sysname]["hit_rate"], "false_alert": w_all[sysname]["false_alert_rate"],
                 "unknown": w_all[sysname]["unknown_rate"]}
            items = [i for i in joined if joined[i]["system"] == sysname and joined[i]["rank"] <= kb]
            entry = {"weights": {k: round(v, 4) for k, v in w.items()},
                     "n_items_in_budget": len(items),
                     "n_docs_in_budget": len({joined[i]["doc"] for i in items}),
                     "n_by_status": dict(Counter(joined[i]["status"] for i in items))}
            for n in annotators:
                ps = defaultdict(list)
                rows = []
                for i in items:
                    if i in lab[n]:
                        v = verdict(lab[n][i])
                        ps[joined[i]["status"]].append(v)
                        rows.append((joined[i]["doc"], joined[i]["status"], v))
                entry[n] = {"doc_clustered": doc_boot_ci(rows, w, rng, a.n_boot),
                            "item_level_as_published": item_boot_ci(ps, w, rng, a.n_boot),
                            "raw_by_status": {s: {"n": len(v), "n_docs": len({d for d, st, _ in rows if st == s}),
                                                  "rate": sum(v) / len(v),
                                                  "wilson_on_items": wilson(sum(v), len(v)),
                                                  "wilson_on_clusters": wilson(
                                                      int(round((sum(v) / len(v)) * len({d for d, st, _ in rows if st == s}))),
                                                      len({d for d, st, _ in rows if st == s}))}
                                              for s, v in sorted(ps.items())}}
            if len(annotators) >= 2:
                x, y = annotators[0], annotators[1]
                for mode in ("both_agree", "either", "both_positive"):
                    ps = defaultdict(list)
                    rows = []
                    for i in items:
                        if i in lab[x] and i in lab[y]:
                            vx, vy = verdict(lab[x][i]), verdict(lab[y][i])
                            if mode == "both_agree":
                                if vx != vy:
                                    continue
                                v = vx
                            elif mode == "either":
                                v = vx or vy
                            else:
                                v = vx and vy
                            ps[joined[i]["status"]].append(v)
                            rows.append((joined[i]["doc"], joined[i]["status"], v))
                    entry[mode] = {"doc_clustered": doc_boot_ci(rows, w, rng, a.n_boot),
                                   "item_level_as_published": item_boot_ci(ps, w, rng, a.n_boot),
                                   "n_items": sum(len(v) for v in ps.values()),
                                   "n_docs": len({d for d, _, _ in rows})}
            out_b["systems"][sysname] = entry
        rep["budgets"][str(kb)] = out_b

    for kb in a.budgets:
        print(f"--- budget k<={kb}", flush=True)
        for s, e in rep["budgets"][str(kb)]["systems"].items():
            print(f"[a3 k<={kb}] {s:32s} n_items={e['n_items_in_budget']:3d} n_docs={e['n_docs_in_budget']:3d} "
                  f"{e['n_by_status']}", flush=True)
            for n in annotators + ["both_agree", "either", "both_positive"]:
                d = e.get(n)
                if not isinstance(d, dict) or not isinstance(d.get("doc_clustered"), dict):
                    continue
                dc, il = d["doc_clustered"], d["item_level_as_published"]
                if dc.get("point") is None:
                    continue
                fmt = lambda c: f"[{c[0]:.3f},{c[1]:.3f}]" if c else "[n/a]"
                flag = " DEGENERATE" if dc.get("degenerate") else ""
                print(f"           {n:16s} {dc['point']:.3f} doc{fmt(dc.get('ci'))}"
                      f" item{fmt(il.get('ci'))}"
                      f" wilson_clusters{fmt(dc.get('wilson_on_clusters'))}"
                      f" (items {dc['n_items']}, docs {dc['n_docs']}){flag}", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "a3_human_precision_docclustered.json").write_text(json.dumps(rep, indent=1))
    print(f"[a3] written {out / 'a3_human_precision_docclustered.json'}", flush=True)


# ============================================================================================== A3b
def kappa_full(x: list[str], y: list[str]) -> dict:
    """Cohen's kappa with every intermediate quantity printed, so a reader can recompute it from the counts.

    Identical arithmetic to mc_human_analyze.kappa (L14-19); this version also returns p_o, p_e, the marginals
    and the contingency table, and flags the case where a rater is constant (then p_e == p_o and kappa is 0 by
    construction, carrying no information about agreement)."""
    if not x:
        return {"n": 0, "kappa": None}
    cats = sorted(set(x) | set(y))
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    mx = {c: x.count(c) / n for c in cats}
    my = {c: y.count(c) / n for c in cats}
    pe = sum(mx[c] * my[c] for c in cats)
    tab = Counter(zip(x, y))
    return {"n": n, "categories": cats,
            "contingency": {f"{i}|{j}": tab.get((i, j), 0) for i in cats for j in cats},
            "marginal_x": {c: x.count(c) for c in cats}, "marginal_y": {c: y.count(c) for c in cats},
            "p_observed": float(po), "p_expected": float(pe),
            "kappa": float((po - pe) / (1 - pe)) if pe < 1 else None,
            "x_is_constant": len(set(x)) == 1, "y_is_constant": len(set(y)) == 1,
            "kappa_uninformative_constant_rater": bool(len(set(x)) == 1 or len(set(y)) == 1)}


def run_a3b(a) -> None:
    key = {}
    for f in a.key:
        for r in load_jsonl(Path(f)):
            key[r["item_id"]] = r
    lab: dict[str, dict[str, dict]] = defaultdict(dict)
    for f in a.export:
        for r in load_jsonl(Path(f)):
            lab[r["annotator"]][r["item_id"]] = r
    annotators = sorted(lab)

    rep = {"registered": "research/marsc_strengthen_20260918/PREREG.md A3b",
           "purpose": "disclosure -- what the 400-item two-annotator human study actually contains",
           "key_files": list(a.key), "export_files": list(a.export),
           "n_key_items": len(key), "annotators": {}, "agreement": {}, "by_system": {}, "by_status": {}}

    def verdict(r):
        return r["q1"] == "A" and r["q2"] == "B"

    for n in annotators:
        rows = list(lab[n].values())
        ts = sorted(float(r["ts"]) for r in rows if r.get("ts") not in (None, ""))
        gaps = np.diff(np.asarray(ts)) if len(ts) > 1 else np.asarray([])
        rep["annotators"][n] = {
            "n_items": len(rows),
            "q1_counts": dict(Counter(r["q1"] for r in rows)),
            "q2_counts": dict(Counter(r["q2"] for r in rows)),
            "q1_q2_joint_counts": {f"{i}|{j}": c for (i, j), c in sorted(Counter((r["q1"], r["q2"]) for r in rows).items())},
            "verdict_counts": dict(Counter(verdict(r) for r in rows).most_common()),
            "q1_constant": len({r["q1"] for r in rows}) == 1,
            "q2_constant": len({r["q2"] for r in rows}) == 1,
            "n_with_comment": sum(1 for r in rows if (r.get("comment") or "").strip()),
            "timing": {"n_timestamps": len(ts),
                       "span_seconds": float(ts[-1] - ts[0]) if len(ts) > 1 else None,
                       "median_gap_seconds": float(np.median(gaps)) if len(gaps) else None,
                       "frac_gaps_under_3s": float((gaps < 3.0).mean()) if len(gaps) else None,
                       "frac_gaps_under_1s": float((gaps < 1.0).mean()) if len(gaps) else None}}
        d = rep["annotators"][n]
        print(f"[a3b] {n}: n={d['n_items']} Q1 {d['q1_counts']} (constant={d['q1_constant']}) "
              f"Q2 {d['q2_counts']} (constant={d['q2_constant']}) verdict {d['verdict_counts']} "
              f"median gap {d['timing']['median_gap_seconds']}s", flush=True)

    if len(annotators) >= 2:
        x, y = annotators[0], annotators[1]
        both = sorted(set(lab[x]) & set(lab[y]))
        q1x = [lab[x][i]["q1"] for i in both]
        q1y = [lab[y][i]["q1"] for i in both]
        q2x = [lab[x][i]["q2"] for i in both]
        q2y = [lab[y][i]["q2"] for i in both]
        vx = [str(verdict(lab[x][i])) for i in both]
        vy = [str(verdict(lab[y][i])) for i in both]
        rep["agreement"] = {"annotator_x": x, "annotator_y": y, "n_co_labelled": len(both),
                            "q1_support": kappa_full(q1x, q1y),
                            "q2_coverage": kappa_full(q2x, q2y),
                            "composite_verdict_q1A_and_q2B": kappa_full(vx, vy)}
        for lbl in ("q1_support", "q2_coverage", "composite_verdict_q1A_and_q2B"):
            g = rep["agreement"][lbl]
            kp = "None" if g["kappa"] is None else f"{g['kappa']:.4f}"
            print(f"[a3b] {lbl:32s} n={g['n']} raw={g['p_observed']:.4f} p_e={g['p_expected']:.4f} kappa={kp} "
                  f"constant-rater={g['kappa_uninformative_constant_rater']} {g['contingency']}", flush=True)

        # Does the degenerate answer pattern depend on the system or the evaluator status? If a rater is
        # constant everywhere, the agreement filter on that question is uninformative in every subgroup too.
        for field, dest in (("system", "by_system"), ("status", "by_status")):
            groups = defaultdict(list)
            for i in both:
                if i in key:
                    groups[key[i][field]].append(i)
            for gname, ids in sorted(groups.items()):
                rep[dest][gname] = {
                    "n": len(ids),
                    f"{x}_q1": dict(Counter(lab[x][i]["q1"] for i in ids)),
                    f"{y}_q1": dict(Counter(lab[y][i]["q1"] for i in ids)),
                    f"{x}_q2": dict(Counter(lab[x][i]["q2"] for i in ids)),
                    f"{y}_q2": dict(Counter(lab[y][i]["q2"] for i in ids)),
                    "q1_kappa": kappa_full([lab[x][i]["q1"] for i in ids], [lab[y][i]["q1"] for i in ids])["kappa"],
                    "q2_kappa": kappa_full([lab[x][i]["q2"] for i in ids], [lab[y][i]["q2"] for i in ids])["kappa"]}

        a2c = rep["annotators"][y]
        rep["disclosure"] = {
            "statement": (f"On Q1 (support), {y} answered a single value on all {a2c['n_items']} items "
                          f"({a2c['q1_counts']}), so Cohen's kappa on Q1 is 0 by construction and the "
                          f"two-annotator agreement filter carries no information on that question. Q2 "
                          f"(coverage) is non-degenerate for both raters."
                          if a2c["q1_constant"] else
                          "Neither annotator is constant on Q1; the agreement filter is informative on both questions."),
            "q1_kappa": rep["agreement"]["q1_support"]["kappa"],
            "q2_kappa": rep["agreement"]["q2_coverage"]["kappa"],
            "q2_raw_agreement": rep["agreement"]["q2_coverage"]["p_observed"],
            "composite_verdict_kappa": rep["agreement"]["composite_verdict_q1A_and_q2B"]["kappa"],
            "remedy": "no analysis can repair a constant rater; only new annotation can."}
        print(f"[a3b] DISCLOSURE: {rep['disclosure']['statement']}", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "a3b_annotator_audit.json").write_text(json.dumps(rep, indent=1))
    print(f"[a3b] written {out / 'a3b_annotator_audit.json'}", flush=True)


# ============================================================================================== A8
# Hand-verified classification of every interval-emitting site. Keyed by (relative path, anchor substring) so
# the table survives line-number drift; the scan records the line each anchor is found at today. `unit` is what
# the estimator actually resamples, read off the code, NOT what the variable is named.
A8_TABLE = [
    # --- document-clustered and correct
    ("research/marsc_20260916/code/mc_e2_eval.py", 'rk["paired_recall"][n] = {"mean"', "document",
     "rng.choice(docs) then sub = [p for dd in pick for p in by_doc[dd]]; docs = common.doc_key(source)", True),
    ("research/marsc_20260916/code/mc_e2_eval.py", 'rk.setdefault("paired_omission_precision"', "document",
     "same document draw as paired_recall", True),
    ("research/marsc_20260916/code/mc_analyze.py", 'rep["paired"][other] = {"lexical_stress"', "document",
     "rng.choice(docs); by_doc keyed by doc_of[p]", True),
    ("research/marsc_20260916/code/mc_matched_auroc.py", 'rep["paired_vs_primary"][t] = {"mean"', "document",
     "boot_docs = rng.choice(docs, ...); sub expands by_doc", True),
    ("research/marsc_20260916/code/mc_judge_precision.py", 'rep["paired"][s_] = {"other_minus_primary"', "document",
     "rng.choice(docs); pair_doc = doc_of[tid]", True),
    ("research/marsc_20260916/code/mc_summary_conditioning.py", '"ci_doc_bootstrap"', "document",
     "rng.choice(keys) where keys = sorted(doc_means)", True),
    ("research/marsc_20260916/code/mc_equivalence.py", "lo2, hi2 = np.percentile(draws,", "document",
     "diff is one value per document; rng.choice over the document index (site NOT named in the paper's audit table)", True),
    ("research/mars2_gates_20260916/code/analyze.py", 'rep["paired_vs_primary"][other] = {k:', "document",
     "rng.choice(docs); by_doc expansion", True),
    ("research/mars3_20260916/code/stage0_diagnostics.py", '"delta_r2_ci"', "document",
     "rng.choice(docs) with idx_by[d]; g is the document group", True),
    ("scripts/plan2026/b3_endpoint.py", 'rep["paired"][f"{m} - {other}"]', "document",
     "rng.choice(docs); by_doc keyed by doc_of[pair_id]", True),
    ("scripts/plan2026/b0_recipe_conditional_value.py", "def rmse_group_bootstrap", "document (group)",
     "resamples np.unique(g); g = row['group'], documented as the source document in b4_certify.py:10", True),
    ("scripts/plan2026/b4_certify.py", "def split_half", "document (group)",
     "splits np.unique(g) = source documents; a split-half spread, not a bootstrap CI on a contrast", True),
    ("scripts/plan2026/b4_certify.py", "ci, ci_au, p = cv.paired_group_bootstrap", "document (group)",
     "delegates to cv.paired_group_bootstrap, which resamples np.unique(g)", True),
    ("scripts/plan2026/a4_reliability_grouping.py", 'res[tag] = {"delta_logloss"', "document x system",
     "cv.paired_group_bootstrap on g2 = doc|system -- FINER than the document, so still anti-conservative "
     "if the same document appears under several systems", False),
    ("scripts/plan2026/a4_reliability_grouping.py", 'res["leave_one_system_out"] = {"delta_logloss"', "document (group)",
     "cv.paired_group_bootstrap on g = the document group", True),
    ("scripts/plan2026/a1_partial_input_reference.py", '"frac_draws_positive"', "document (group)",
     "percentiles over draws merged from the refit mode, whose every draw resamples groups "
     "('Full-procedure group bootstrap ... resampling groups')", True),
    ("scripts/plan2026/b6_rerank.py", 'rep["paired"] = {k: {"mean"', "document",
     "resamples srcs = source[:200] keys (site NOT named in the paper's audit table)", True),
    ("scripts/conditional_value.py", "def paired_group_bootstrap", "document (group)",
     "the shared helper every group-bootstrap site above delegates to; resamples np.unique(g)", True),
    ("scripts/conditional_value.py", "ci_ll, ci_au, p_ll = paired_group_bootstrap", "document (group)",
     "the recipe-level certify call; delegates to paired_group_bootstrap on gg", True),
    ("research/mars2_gates_20260916/code/analyze.py", "ci, _, p = cv.paired_group_bootstrap", "document (group)",
     "the certify block's conditional-value CI; cv.paired_group_bootstrap on g", True),
    ("research/mars2_gates_20260916/code/analyze.py", "ci2, _, p2 = cv.paired_group_bootstrap", "document (group)",
     "the over-reference CI in the same certify block; cv.paired_group_bootstrap on g", True),
    ("scripts/plan2026/a1_partial_input_reference.py", "ci_b, _, p_b = cv.paired_group_bootstrap", "document (group)",
     "per-cell conditional-value CI; cv.paired_group_bootstrap on gg", True),
    ("scripts/plan2026/a1_partial_input_reference.py", "ci_p, ci_au, p_p = cv.paired_group_bootstrap", "document (group)",
     "per-cell primary-rule CI; cv.paired_group_bootstrap on gg", True),
    ("scripts/plan2026/b0_recipe_conditional_value.py", "ci, ci_au, p = cv.paired_group_bootstrap", "document (group)",
     "the binary-endpoint CI beside the RMSE group bootstrap; cv.paired_group_bootstrap on gg", True),
    ("scripts/plan2026/b4_certify.py", "ci2, _, p2 = cv.paired_group_bootstrap", "document (group)",
     "the counters_plus_learned CI; cv.paired_group_bootstrap on g", True),
    # --- pair-level: the A2 defect
    ("research/marsc_20260916/code/mc_crossed_precision.py", '"ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],', "pair",
     "DEFECT: variable named by_doc but keyed by t['pair_id']; the file never imports common and never calls "
     "doc_key. REPAIRED by research/marsc_strengthen_20260918/code/a2a3_clustered.py mode a2", False),
    ("research/marsc_20260916/code/mc_crossed_precision.py", 'return {"point": float(obs), "ci":', "pair",
     "the crossed 2x2 effects reuse the same pair-level boot_idx. REPAIRED by a2a3_clustered.py mode a2", False),
    # --- item-level
    ("research/marsc_20260916/code/mc_human_rank.py", "def boot_ci", "annotated item (within status stratum)",
     "DEFECT: rng.choice over the items of each status; 8-12 summaries of one document enter independently. "
     "REPAIRED by a2a3_clustered.py mode a3", False),
    ("research/marsc_20260916/code/mc_human_rank.py", "def wilson", "annotated item",
     "DEFECT: binomial interval on the item count, which is not the independent unit. a2a3_clustered.py mode a3 "
     "reports a Wilson interval on the CLUSTER count beside it", False),
    ("research/marsc_20260916/code/mc_human_rank.py", '"wilson": wilson(sum(v), len(v))', "annotated item",
     "DEFECT: the per-status call site of the item-count Wilson interval. REPAIRED by a2a3_clustered.py mode a3, "
     "which emits wilson_on_items and wilson_on_clusters side by side", False),
    ("scripts/plan2026/a5_metric_selection.py", '"ci95": [float(np.percentile(boots, 2.5))', "evaluation cell",
     "DEFECT (third unclustered site, NOT named in the paper's audit table): rng.choice over the ~10 "
     "leave-one-cell-out evaluation cells. The cell, not the document, is the exchangeable unit for a "
     "selection experiment, but the interval is a 10-point bootstrap and must be labelled as such", False),
    # --- np.percentile used for something that is not an interval (listed so the scan is auditable)
    ("scripts/plan2026/b7_failure_analysis.py", "sw = np.percentile(", "n/a",
     "tertile binning of source/candidate length, not a confidence interval", None),
    ("scripts/plan2026/b4_certify.py", '"p95": float(np.percentile(vals, 95))', "n/a",
     "descriptive cost percentile, not a confidence interval", None),
    ("scripts/plan2026/b4_certify.py", "bins = np.digitize(cw, np.percentile(", "n/a",
     "tertile binning, not a confidence interval", None),
]

SCAN_PAT = re.compile(r"np\.percentile|def wilson|wilson\(|paired_group_bootstrap|proportion_confint|"
                      r"\"ci\"|'ci'|ci95|ci_doc_bootstrap|tost_interval")
# Lines that only PRINT an interval already classified elsewhere are not themselves estimator sites.
NOISE_PAT = re.compile(r"^\s*(print\(|f[\"'])|flush=True")


def run_a8(a) -> None:
    roots = [Path(x) for x in a.roots]
    found = []
    for root in roots:
        base = REPO / root
        files = sorted(base.rglob("*.py")) if base.is_dir() else ([base] if base.suffix == ".py" else [])
        for f in files:
            if "__pycache__" in f.parts:
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if SCAN_PAT.search(line) and not NOISE_PAT.search(line):
                    found.append({"file": str(f.relative_to(REPO)), "line": i, "text": line.strip()[:200]})

    table = []
    matched_scan = set()
    for path, anchor, unit, why, ok in A8_TABLE:
        p = REPO / path
        hits = []
        if p.exists():
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if anchor in line:
                    hits.append(i)
        entry = {"file": path, "anchor": anchor, "line": hits[0] if hits else None, "n_anchor_hits": len(hits),
                 "resampling_unit": unit, "clustering": ("document-clustered" if ok is True else
                                                         ("not-a-confidence-interval" if ok is None else
                                                          "NOT document-clustered")),
                 "correct_for_a_document_clustered_design": ok, "evidence": why}
        if not hits:
            entry["ANCHOR_NOT_FOUND"] = True
        table.append(entry)
        for h in hits:
            matched_scan.add((path, h))

    # Any scanned site whose line is not covered by a table anchor within +/- `--slack` lines is UNCLASSIFIED.
    unclassified = []
    for s in found:
        near = any(s["file"] == f and abs(s["line"] - l) <= a.slack for f, l in matched_scan)
        if not near:
            unclassified.append(s)

    rep = {"registered": "research/marsc_strengthen_20260918/PREREG.md A8",
           "purpose": "complete the confidence-interval site audit: every interval site, its resampling unit, "
                      "and whether that unit is the source document",
           "roots_scanned": [str(r) for r in roots],
           "scan_pattern": SCAN_PAT.pattern, "anchor_slack_lines": a.slack,
           "n_scan_hits": len(found), "n_classified_sites": len(table),
           "n_unclassified_scan_hits": len(unclassified),
           "counts_by_clustering": dict(Counter(e["clustering"] for e in table)),
           "sites": table,
           "unclassified_scan_hits": unclassified,
           "sites_not_named_in_the_papers_audit_table": [e["file"] + ":" + str(e["line"]) for e in table
                                                         if "NOT named in the paper's audit table" in e["evidence"]]}
    for e in table:
        mark = {"document-clustered": "OK ", "NOT document-clustered": "BAD", "not-a-confidence-interval": "---"}[e["clustering"]]
        print(f"[a8] {mark} {e['file']}:{e['line']}  unit={e['resampling_unit']}", flush=True)
    print(f"[a8] {len(table)} classified sites, {dict(Counter(e['clustering'] for e in table))}", flush=True)
    if unclassified:
        print(f"[a8] {len(unclassified)} UNCLASSIFIED scan hits (report them, do not drop them):", flush=True)
        for s in unclassified:
            print(f"[a8]   ? {s['file']}:{s['line']}  {s['text']}", flush=True)
    else:
        print("[a8] no unclassified scan hit", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "a8_ci_site_audit.json").write_text(json.dumps(rep, indent=1))
    print(f"[a8] written {out / 'a8_ci_site_audit.json'}", flush=True)


# ============================================================================================== cli
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    p2 = sub.add_parser("a2", help="R07 crossed precision with a document-clustered bootstrap")
    p2.add_argument("--tasks", required=True)
    p2.add_argument("--judgments", nargs="+", required=True)
    p2.add_argument("--cells", nargs="+", required=True)
    p2.add_argument("--extra-systems", nargs="*", default=[])
    p2.add_argument("--max-rank", type=int, default=10)
    p2.add_argument("--n-boot", type=int, default=5000)
    p2.add_argument("--seed", type=int, default=20260917)
    p2.add_argument("--out", required=True)
    p2.set_defaults(fn=run_a2)

    p3 = sub.add_parser("a3", help="human precision study with a document-clustered bootstrap")
    p3.add_argument("--emitted", nargs="+", required=True)
    p3.add_argument("--key", nargs="+", required=True)
    p3.add_argument("--export", nargs="+", required=True)
    p3.add_argument("--e2-json", required=True)
    p3.add_argument("--pair-source", nargs="+", required=True,
                    help="jsonl file(s) carrying pair_id + source, used only to build the document clusters")
    p3.add_argument("--budgets", type=int, nargs="+", default=[10, 20])
    p3.add_argument("--n-boot", type=int, default=10000)
    p3.add_argument("--seed", type=int, default=20260917)
    p3.add_argument("--out", required=True)
    p3.set_defaults(fn=run_a3)

    pb = sub.add_parser("a3b", help="annotator answer distributions and kappas, with raw counts")
    pb.add_argument("--key", nargs="+", required=True)
    pb.add_argument("--export", nargs="+", required=True)
    pb.add_argument("--out", required=True)
    pb.set_defaults(fn=run_a3b)

    p8 = sub.add_parser("a8", help="confidence-interval site audit")
    p8.add_argument("--roots", nargs="+",
                    default=["research/marsc_20260916/code", "research/mars2_gates_20260916/code",
                             "research/mars3_20260916/code", "scripts/plan2026", "scripts/conditional_value.py"])
    p8.add_argument("--slack", type=int, default=15,
                help="a scan hit within this many lines of a classified anchor is taken to belong to "
                     "that estimator (the anchor names the function, the hit is inside its body)")
    p8.add_argument("--out", required=True)
    p8.set_defaults(fn=run_a8)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
