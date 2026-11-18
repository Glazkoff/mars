#!/usr/bin/env python3
"""B18 / B21 analysis -- every system on one unit set, human ACU labels (acu) or B1 labels (m2).

Systems
  mars2:<arm>|masked|q / |ll      b18_score_mars2 outputs (supplied units; seed-averaged)      [acu]
  mars2:<arm>|nomask|q            diagnostic variant                                            [acu]
  mars2:<arm>|aligned|q           1-q of the dumped extracted units inside the unit's source sentence [acu]
  mars2:<arm>|q / |ll             1-q and -loglik from the B2 dumps (b3 alignment)              [m2]
  <baseline files>                unit_scores from b21_zeroshot / b21_finetune_xenc (seed files averaged)
  judge:gemma                     supplied-target judge p_omitted (b1_judge_labels format)     [acu]
  open_judge:<model>              generated omissions re-matched to units by token F1 >= 0.5
  counter:*                       oriented counters (omission direction)
Endpoint: per-pair macro AUROC, P@k, R@5, pooled AUROC (all / per resource / paraphrased-only),
paired document bootstrap vs the primary. Certification: counter max, split-half, permutation null,
conditional value counters -> counters + system per cell, one BH family (cells x systems).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import common

cv = common.cv()
b3 = common._load("b3_endpoint", common.REPO / "scripts" / "plan2026")
b4 = common._load("b4_certify", common.REPO / "scripts" / "plan2026")
K_LIST = (1, 3, 5, 10)
ARM_RE = re.compile(r"/b2/(?:[a-z]+-[a-z0-9_]+_)?([a-z_]+)_seed(\d+)/")


def seed_average(per_seed: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {pid: np.nanmean(np.vstack(v), axis=0) for pid, v in per_seed.items()}


def load_mars2_acu(paths: list[str], rows: dict[str, dict]) -> dict[str, dict[str, np.ndarray]]:
    acc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for p in paths:
        for rec in common.load_jsonl(Path(p)):
            m = ARM_RE.search(rec["ckpt"] + "/")
            arm = m.group(1) if m else Path(rec["ckpt"]).parent.name
            if rec["pair_id"] not in rows or rec["n_units"] != len(rows[rec["pair_id"]]["units"]):
                continue
            for var, d in rec["variants"].items():
                acc[f"mars2:{arm}|{var}|q"][rec["pair_id"]].append(1 - np.asarray(d["q"], float))
                acc[f"mars2:{arm}|{var}|ll"][rec["pair_id"]].append(-np.asarray(d["loglik"], float))
    return {k: seed_average(v) for k, v in acc.items()}


def load_mars2_dumps(specs: list[str], rows: dict[str, dict], unit_set: str) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for spec in specs:
        name, paths = spec.split("=", 1)
        dumps = [b3.load_dump(Path(p)) for p in paths.split(",")]
        q_acc, ll_acc, al_acc = defaultdict(list), defaultdict(list), defaultdict(list)
        for pid, r in rows.items():
            for d in dumps:
                units = d.get(pid)
                if not units:
                    continue
                if unit_set == "m2":
                    idx = b3.align_units(units, r["units"])
                    if any(j is None for j in idx):
                        continue
                    q_acc[pid].append(np.array([1 - float(units[j]["support"]) for j in idx]))
                    if all("loglik" in units[j] for j in idx):
                        ll_acc[pid].append(np.array([-float(units[j]["loglik"]) for j in idx]))
                else:   # supplied units: max omission over the dumped units inside the unit's source sentence
                    v = []
                    for u in r["units"]:
                        s, e = common.best_sentence_span(r["source"], u["text"])
                        inside = [1 - float(x["support"]) for x in units if s <= b3._u(x).get("start", -1) and b3._u(x).get("end", 1e9) <= e]
                        if not inside:
                            ct = set(common.tokens(u["text"], content_only=True))
                            inside = [1 - float(x["support"]) for x in units if ct & set(common.tokens(b3._u(x).get("text", "")))]
                        v.append(max(inside) if inside else 0.0)   # no inventory unit -> undetectable (was 0.5 before 2026-09-16 16:30)
                    al_acc[pid].append(np.array(v))
        if unit_set == "m2":
            out[f"mars2:{name}|q"] = seed_average(q_acc)
            if ll_acc:
                out[f"mars2:{name}|ll"] = seed_average(ll_acc)
        else:
            out[f"mars2:{name}|aligned|q"] = seed_average(al_acc)
    return out


def load_aligned_pipeline(spec: str, rows: dict[str, dict]) -> tuple[str, dict[str, np.ndarray]]:
    """name=<units file>:<scores glob> -> pipeline-aligned:<name> on the acu set: for each ACU the max omission score over
    inventory units inside its best-matching sentence that share a content token; 0.0 when the inventory has none."""
    name, rest = spec.split("=", 1); units_file, sglob = rest.rsplit(":", 1)
    inv = {r["pair_id"]: r["units"] for r in common.load_jsonl(Path(units_file)) if r.get("units")}
    acc: dict[str, list] = defaultdict(list)
    for f in sorted(glob.glob(sglob)):
        _, sc = common.read_scores(Path(f))
        for pid, v in sc.items():
            if pid in inv and len(v) == len(inv[pid]):
                acc[pid].append(v)
    scores = {pid: np.nanmean(np.vstack(v), axis=0) for pid, v in acc.items()}
    out = {}
    for pid, r in rows.items():
        if pid not in scores:
            continue
        units, sc = inv[pid], scores[pid]; v = []
        for u in r["units"]:
            s, e = common.best_sentence_span(r["source"], u["text"])
            ct = set(common.tokens(u["text"], content_only=True)) or set(common.tokens(u["text"]))
            cand = [sc[j] for j, x in enumerate(units) if x.get("start", -1) < e and x.get("end", 1e9) > s and ct & set(common.tokens(x["text"])) and np.isfinite(sc[j])]
            v.append(max(cand) if cand else 0.0)
        out[pid] = np.array(v)
    return f"pipeline-aligned:{name}", out


def load_baselines(paths: list[str]) -> dict[str, dict[str, np.ndarray]]:
    acc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for p in paths:
        name, sc = common.read_scores(Path(p))
        for pid, v in sc.items():
            acc[name][pid].append(v)
    return {k: seed_average(v) for k, v in acc.items()}


def load_judge(paths: list[str], rows: dict[str, dict]) -> dict[str, np.ndarray]:
    pj = defaultdict(dict)
    for p in paths:
        for x in common.load_jsonl(Path(p)):
            pj[x["pair_id"]][x["unit_index"]] = x["p_omitted"]
    out = {}
    for pid, r in rows.items():
        if pid in pj and len(pj[pid]) == len(r["units"]):
            out[pid] = np.array([pj[pid][i] for i in range(len(r["units"]))])
    return out


def load_open_judge(path: str, rows: dict[str, dict], thr: float = 0.5) -> dict[str, np.ndarray]:
    out = {}
    for x in common.load_jsonl(Path(path)):
        r = rows.get(x["pair_id"])
        if r is None:
            continue
        items = [it.get("item", "") for it in x.get("items", [])]
        out[x["pair_id"]] = np.array([1.0 if any(common.token_f1(it, u["text"]) >= thr for it in items) else 0.0 for u in r["units"]])
    return out


def counter_systems(rows: dict[str, dict], unit_set: str):
    """Oriented counters as systems + the raw counter matrix per pair for certification."""
    raw = {pid: [common.counters_for(unit_set, r, i) for i in range(len(r["units"]))] for pid, r in rows.items()}
    names = sorted(next(iter(raw.values()))[0])
    systems = {}
    for cname, flip in (("ca_token_recall", True), ("ca_rougeL_recall", True), ("ca_unit_in_cand", True),
                        ("acu_relpos" if unit_set == "acu" else "sbl_first_mention_relpos", False)):
        if cname in names:
            systems[f"counter:{'1-' if flip else ''}{cname}"] = {pid: np.array([(1 - c[cname]) if flip else c[cname] for c in cs]) for pid, cs in raw.items()}
    return systems, raw, names


def per_pair_metrics(v: np.ndarray, truth: list) -> dict | None:
    m = np.array([t is not None for t in truth])
    y = np.array([t if t is not None else 0 for t in truth])[m]; s = np.asarray(v, float)[m]
    if len(y) < 2 or y.sum() in (0, len(y)) or not np.isfinite(s).all():
        return None
    r = {"auroc": cv.auroc(y, s)}
    for k in K_LIST:
        p, rr = b3.prec_rec_at_k(s, y.tolist(), k); r[f"P@{k}"] = p; r[f"R@{k}"] = rr
    return r


def summarise(pm: dict[str, dict], pids: list[str]) -> dict:
    vals = [pm[p] for p in pids if pm.get(p)]
    if not vals:
        return {"n_pairs_scored": 0}
    out = {"n_pairs_scored": len(vals), "auroc_macro": float(np.mean([v["auroc"] for v in vals]))}
    for k in K_LIST:
        out[f"P@{k}"] = float(np.mean([v[f"P@{k}"] for v in vals]))
    out["R@5"] = float(np.nanmean([v["R@5"] for v in vals]))
    return out


def pooled_auroc(sys_scores: dict[str, np.ndarray], rows: dict[str, dict], pids: list[str], keep=None) -> float | None:
    y, s = [], []
    for pid in pids:
        t = common.truth_of(rows[pid]); v = sys_scores[pid]
        for i, ti in enumerate(t):
            if ti is None or not np.isfinite(v[i]) or (keep is not None and not keep[pid][i]):
                continue
            y.append(ti); s.append(v[i])
    y = np.array(y); s = np.array(s)
    return float(cv.auroc(y, s)) if len(y) and 0 < y.sum() < len(y) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit-set", choices=["acu", "m2"], required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--units", default="", help="rows file override")
    ap.add_argument("--mars2-glob", default="", help="b18_score_mars2 outputs (acu)")
    ap.add_argument("--dumps", nargs="*", default=[], help="name=path[,path] B2 dumps (m2: q/ll; acu: aligned)")
    ap.add_argument("--scores-glob", nargs="*", default=[], help="unit_scores files (baselines)")
    ap.add_argument("--judge-glob", default="")
    ap.add_argument("--aligned-pipeline", nargs="*", default=[], help="name=<units file>:<scores glob> (acu set): inventory-based pipeline mapped onto ACUs")
    ap.add_argument("--open-judge", nargs="*", default=[], help="name=path b3_open_judge outputs")
    ap.add_argument("--primary", default="")
    ap.add_argument("--reference-system", default="", help="also certify every system over counters + this system (e.g. the cross-encoder)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--min-cell-rows", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows_l = common.load_jsonl(Path(a.units)) if a.units else common.unit_rows(a.unit_set, a.split)
    rows = {r["pair_id"]: r for r in rows_l if r.get("units")}
    systems: dict[str, dict[str, np.ndarray]] = {}
    if a.mars2_glob:
        systems.update(load_mars2_acu(sorted(glob.glob(a.mars2_glob)), rows))
    systems.update(load_mars2_dumps(a.dumps, rows, a.unit_set))
    files = [f for g in a.scores_glob for f in sorted(glob.glob(g))]
    systems.update(load_baselines(files))
    for spec in a.aligned_pipeline:
        n, s = load_aligned_pipeline(spec, rows)
        if s:
            systems[n] = s
    if a.judge_glob:
        j = load_judge(sorted(glob.glob(a.judge_glob)), rows)
        if j:
            systems["judge:gemma"] = j
    for spec in a.open_judge:
        name, path = spec.split("=", 1)
        systems[f"open_judge:{name}"] = load_open_judge(path, rows)
    csys, raw_counters, cnames = counter_systems(rows, a.unit_set)
    systems.update(csys)
    systems = {k: v for k, v in systems.items() if v}
    primary = a.primary or next((k for k in systems if k.startswith("mars2:direct_enabled") and k.endswith("|q") and "nomask" not in k), next(iter(systems)))

    common_pids = sorted(pid for pid in rows if all(pid in s and len(s[pid]) == len(rows[pid]["units"]) for s in systems.values()))
    docs = sorted({common.doc_key(rows[p]["source"]) for p in common_pids}); doc_of = {p: common.doc_key(rows[p]["source"]) for p in common_pids}
    by_doc = defaultdict(list)
    for p in common_pids:
        by_doc[doc_of[p]].append(p)
    print(f"[analyze:{a.unit_set}] systems={len(systems)} pairs common={len(common_pids)}/{len(rows)} docs={len(docs)} primary={primary}", flush=True)
    para = {pid: [c["ca_unit_in_cand"] == 0.0 and c["ca_token_recall"] < 0.5 for c in raw_counters[pid]] for pid in common_pids}
    pm = {name: {p: per_pair_metrics(s[p], common.truth_of(rows[p])) for p in common_pids} for name, s in systems.items()}
    rep = {"registered": "research/mars2_gates_20260916/PREREG.md", "unit_set": a.unit_set, "split": a.split,
           "n_pairs_all": len(rows), "n_pairs_common": len(common_pids), "n_docs": len(docs), "primary": primary, "systems": {}}
    resources = sorted({rows[p]["resource"] for p in common_pids})
    for name, s in systems.items():
        e = summarise(pm[name], common_pids)
        e["auroc_pooled"] = pooled_auroc(s, rows, common_pids)
        e["auroc_pooled_paraphrased_only"] = pooled_auroc(s, rows, common_pids, para)
        e["auroc_pooled_by_resource"] = {res: pooled_auroc(s, rows, [p for p in common_pids if rows[p]["resource"] == res]) for res in resources}
        rep["systems"][name] = e
        print(f"[analyze:{a.unit_set}] {name:44s} macro {e.get('auroc_macro')} pooled {e['auroc_pooled']} para {e['auroc_pooled_paraphrased_only']} P@5 {e.get('P@5')}", flush=True)
    # paired document bootstrap: primary - other (macro AUROC, P@5) from precomputed per-pair metrics
    rng = np.random.default_rng(common.SEED)
    rep["paired_vs_primary"] = {}
    prim = pm[primary]
    for other in systems:
        if other == primary:
            continue
        diffs = defaultdict(list)
        for _ in range(a.n_boot):
            pick = rng.choice(docs, size=len(docs), replace=True)
            sub = [p for d in pick for p in by_doc[d] if prim.get(p) and pm[other].get(p)]
            if not sub:
                continue
            for key in ("auroc", "P@5"):
                diffs[key].append(np.mean([prim[p][key] for p in sub]) - np.mean([pm[other][p][key] for p in sub]))
        rep["paired_vs_primary"][other] = {k: {"mean": float(np.mean(v)), "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for k, v in diffs.items()}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    common.dump_json(out / f"{a.unit_set}_endpoint.json", rep)

    # certification per resource cell
    cert = {"registered": rep["registered"], "unit_set": a.unit_set, "counters": cnames, "cells": {}}
    fam = []
    noncounter = [n for n in systems if not n.startswith("counter:")]
    for res in resources:
        pids = [p for p in common_pids if rows[p]["resource"] == res]
        X, y, g, S = [], [], [], defaultdict(list)
        for p in pids:
            t = common.truth_of(rows[p])
            for i, ti in enumerate(t):
                if ti is None or any(not np.isfinite(systems[n][p][i]) for n in noncounter):
                    continue
                X.append([raw_counters[p][i][k] for k in cnames]); y.append(ti); g.append(doc_of[p])
                for n in noncounter:
                    S[n].append(float(systems[n][p][i]))
        if len(y) < a.min_cell_rows or len(set(y)) < 2 or len(set(g)) < 8:
            cert["cells"][res] = {"skipped": f"{len(y)} usable rows, {len(set(g))} docs"}; continue
        Xc = np.array(X, float); y = np.array(y); g = np.array(g)
        obs, best, sign = b4.oriented_max(Xc, y, cnames, np.arange(len(y)))
        cr = {"n": int(len(y)), "n_docs": int(len(set(g))), "n_pos": int(y.sum()),
              "counter_max": {"auroc": obs, "counter": best, "sign": sign},
              "split_half": b4.split_half(Xc, y, g, cnames, rng), "permutation_null": b4.perm_null(Xc, y, g, cnames, rng, obs), "systems": {}}
        pa = cv.oof_predictions(Xc, y, g, rng)
        ref = a.reference_system if a.reference_system in S else ""
        if ref:
            rv = np.asarray(S[ref]); pref = cv.oof_predictions(np.column_stack([Xc, rv]), y, g, rng)
            cr["reference"] = {"system": ref, "auroc_counters_plus_reference": cv.auroc(y[np.isfinite(pref)], pref[np.isfinite(pref)])}
        for n in noncounter:
            v = np.asarray(S[n]); pb = cv.oof_predictions(np.column_stack([Xc, v]), y, g, rng)
            fin = np.isfinite(pa) & np.isfinite(pb)
            ci, _, p = cv.paired_group_bootstrap(y[fin], pa[fin], pb[fin], g[fin], rng)
            cr["systems"][n] = {"marginal_auroc": cv.auroc(y, v), "delta_logloss": cv.logloss(y[fin], pa[fin]) - cv.logloss(y[fin], pb[fin]),
                                "ci": ci, "delta_auroc": cv.auroc(y[fin], pb[fin]) - cv.auroc(y[fin], pa[fin]), "p": p}
            fam.append((res, n, p or 1.0))
            if ref and n != ref:
                pb2 = cv.oof_predictions(np.column_stack([Xc, rv, v]), y, g, rng)
                f2 = np.isfinite(pref) & np.isfinite(pb2)
                ci2, _, p2 = cv.paired_group_bootstrap(y[f2], pref[f2], pb2[f2], g[f2], rng)
                cr["systems"][n]["over_reference"] = {"delta_logloss": cv.logloss(y[f2], pref[f2]) - cv.logloss(y[f2], pb2[f2]), "ci": ci2, "p": p2,
                                                      "adds_information": bool(p2 is not None and p2 < 0.05 and ci2 and ci2[0] > 0)}
            extra = cr["systems"][n].get("over_reference"); print(f"[certify:{a.unit_set}] {res}/{n}: marginal {cr['systems'][n]['marginal_auroc']:.3f} counter-max {obs:.3f} dLL {cr['systems'][n]['delta_logloss']:+.4f} {ci}" + (f" | over {ref}: {extra['delta_logloss']:+.4f} {extra['ci']}" if extra else ""), flush=True)
        cert["cells"][res] = cr
        common.dump_json(out / f"{a.unit_set}_certify.json", cert)
    keep = cv.bh([p for *_, p in fam])
    for (res, n, _), k in zip(fam, keep):
        e = cert["cells"][res]["systems"][n]; e["bh_reject"] = bool(k); e["adds_information"] = bool(k and e["delta_logloss"] > 0)
    cert["multiplicity"] = {"family_size": len(fam), "n_reject": int(sum(1 for (res, n, _), k in zip(fam, keep) if k and cert["cells"][res]["systems"][n]["delta_logloss"] > 0))}
    cert["bars"] = {n: {"cells_positive_over_counters": [res for res in cert["cells"] if cert["cells"][res].get("systems", {}).get(n, {}).get("adds_information")]} for n in noncounter}
    # pre-registered bars for the primary
    cmax_pooled = max((pooled_auroc(csys[c], rows, common_pids) or 0.0) for c in csys) if csys else None
    nli = [n for n in systems if n.startswith("nli:")]
    best_nli = max(nli, key=lambda n: rep["systems"][n].get("auroc_pooled") or 0.0) if nli else None
    prim_pooled = rep["systems"][primary].get("auroc_pooled")
    cert["prereg_bars"] = {
        "H1_primary_adds_information_all_cells": {"cells": cert["bars"].get(primary, {}).get("cells_positive_over_counters", []), "n_cells": len([c for c in cert["cells"] if "systems" in cert["cells"][c]])},
        "H2_primary_pooled_minus_counter_max": None if prim_pooled is None or cmax_pooled is None else prim_pooled - cmax_pooled,
        "H3_primary_minus_best_zero_shot_nli": None if not best_nli or prim_pooled is None else
            {"best_nli": best_nli, "delta_pooled": prim_pooled - (rep["systems"][best_nli].get("auroc_pooled") or 0.0),
             "macro_paired": rep["paired_vs_primary"].get(best_nli, {}).get("auroc")}}
    common.dump_json(out / f"{a.unit_set}_certify.json", cert)
    print(f"[certify:{a.unit_set}] bars {json.dumps(cert['prereg_bars'])}", flush=True)
    print(f"[analyze:{a.unit_set}] written {out}", flush=True)


if __name__ == "__main__":
    main()
