#!/usr/bin/env python3
"""MARS-C A1 -- the calibration contract (registered in ../PREREG.md, Wave A / A1).

Defect. `mc_train.py` fits a per-seed Platt map on its calibration documents and applies it to every score file
it writes; `mc_score_units.py` emits a bare sigmoid. The contract is therefore mixed *within* validation: the
primary `humanfact` cell on gemma+ent is Platt-shrunk while `humanfact.distilled` and every `judgefact` cell
are raw. A1a declares ONE contract -- raw sigmoid everywhere -- and re-derives every validation cell under it.

Route. The Platt map is p_cal = sigmoid(slope * z + intercept) with {slope, intercept} stored in each arm's
`<trainout>/result.json` under "calibration". It is strictly monotone for slope > 0, so it is exactly
invertible: z = (logit(p_cal) - intercept) / slope. Inversion is CPU-only and needs no checkpoint. It is
VERIFIED before use against the raw-logit companions `acu_<tag>_seed<S>_logits.jsonl` that mc_train wrote
beside the calibrated validation-ACU files; `verify` exits non-zero if the reproduction is outside tolerance,
so no downstream number can be produced on an unverified inversion.

Subcommands
  invert   one calibrated unit_scores file + its arm's result.json -> the raw-sigmoid file (same basename)
  verify   reproduce the stored raw logits from the calibrated probabilities; PASS/FAIL per file (A1a gate)
  topk     how many pairs change their top-k SET between two score rosters (A1b invariance measurement)
  auroc    matched label-source AUROC under three seed-aggregations: calibrated prob, raw prob, raw LOGIT (A1d)
  collect  side-by-side of the same system names across several mc_e2_eval outputs, with observed deltas

Every output is JSON. Nothing here reads a held-out split.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402

EPS = 1e-15


# --------------------------------------------------------------------------- helpers
def calib_of(result_json: Path) -> dict:
    d = json.loads(Path(result_json).read_text())
    c = d.get("calibration")
    assert c is not None, f"{result_json}: no 'calibration' key"
    assert c.get("slope") is not None and c.get("intercept") is not None, f"{result_json}: incomplete calibration {c}"
    assert float(c["slope"]) > 0.0, (f"{result_json}: slope {c['slope']} is not positive -- the Platt map is not "
                                     f"monotone increasing and the rank-invariance argument does not hold")
    return {"n_cells": c.get("n_cells"), "omitted_rate": c.get("omitted_rate"),
            "slope": float(c["slope"]), "intercept": float(c["intercept"]), "result_json": str(result_json)}


def _logit(p: np.ndarray) -> tuple[np.ndarray, int]:
    """logit with saturation accounting: returns (z, n_clipped)."""
    p = np.asarray(p, dtype=float)
    fin = np.isfinite(p)
    n_clip = int(np.sum(fin & ((p <= 0.0) | (p >= 1.0))))
    q = np.clip(p, EPS, 1.0 - EPS)
    z = np.full_like(p, np.nan)
    z[fin] = np.log(q[fin] / (1.0 - q[fin]))
    return z, n_clip


def invert_rows(path: Path, slope: float, intercept: float) -> tuple[dict[str, list], dict]:
    """p_cal -> p_raw = sigmoid((logit(p_cal) - intercept) / slope), per pair. Nulls stay null."""
    per: dict[str, list] = {}
    stats = {"n_pairs": 0, "n_units": 0, "n_null": 0, "n_clipped": 0,
             "raw_logit_min": math.inf, "raw_logit_max": -math.inf}
    sysname, unit_set = None, None
    for r in common.load_jsonl(path):
        sysname = r["system"]; unit_set = r["unit_set"]
        v = np.array([np.nan if x is None else float(x) for x in r["scores"]], dtype=float)
        z_cal, n_clip = _logit(v)
        z_raw = (z_cal - intercept) / slope
        p_raw = 1.0 / (1.0 + np.exp(-z_raw))
        per[r["pair_id"]] = p_raw.tolist()
        stats["n_pairs"] += 1; stats["n_units"] += int(v.size)
        stats["n_null"] += int(np.sum(~np.isfinite(v))); stats["n_clipped"] += n_clip
        fin = np.isfinite(z_raw)
        if fin.any():
            stats["raw_logit_min"] = min(stats["raw_logit_min"], float(np.min(z_raw[fin])))
            stats["raw_logit_max"] = max(stats["raw_logit_max"], float(np.max(z_raw[fin])))
    if stats["raw_logit_min"] == math.inf:
        stats["raw_logit_min"] = stats["raw_logit_max"] = None
    return per, {**stats, "system": sysname, "unit_set": unit_set}


def cmd_invert(a) -> None:
    c = calib_of(Path(a.result_json))
    per, stats = invert_rows(Path(a.inp), c["slope"], c["intercept"])
    out = Path(a.out)
    common.write_scores(out, stats["system"], stats["unit_set"], per)
    rec = {"in": str(a.inp), "out": str(out), "calibration": c, **stats}
    rng_s = ("none" if stats["raw_logit_min"] is None
             else f"[{stats['raw_logit_min']:.3f},{stats['raw_logit_max']:.3f}]")
    print(f"[a1:invert] {Path(a.inp).name} slope {c['slope']:.4f} icpt {c['intercept']:+.4f} -> {out.name}: "
          f"{stats['n_pairs']} pairs, {stats['n_units']} units, {stats['n_null']} null, "
          f"{stats['n_clipped']} saturated, raw logit range {rng_s}", flush=True)
    assert stats["n_pairs"] > 0, f"{a.inp}: no rows read"
    if a.record:
        Path(a.record).parent.mkdir(parents=True, exist_ok=True)
        Path(a.record).write_text(json.dumps(rec, indent=1))


# --------------------------------------------------------------------------- A1a gate: verify the inversion
def cmd_verify(a) -> None:
    """Each --triple is TAG:CALIBRATED:RAW_LOGITS:RESULT_JSON. The inversion must reproduce the stored raw
    logits to --tol; otherwise this exits 5 and nothing downstream may run."""
    rep = {"registered": "PREREG.md Wave A / A1a (inversion gate)", "tol": a.tol, "files": [], "pass": True}
    for spec in a.triple:
        tag, cal, logits, rj = spec.split(":", 3)
        c = calib_of(Path(rj))
        _, cal_sc = common.read_scores(Path(cal))
        _, raw_sc = common.read_scores(Path(logits))
        pids = sorted(set(cal_sc) & set(raw_sc))
        errs, n_units, n_clip, n_len_mismatch = [], 0, 0, 0
        for p in pids:
            vc, vr = cal_sc[p], raw_sc[p]
            if len(vc) != len(vr):
                n_len_mismatch += 1
                continue
            z_cal, nc = _logit(vc)
            n_clip += nc
            z_hat = (z_cal - c["intercept"]) / c["slope"]
            m = np.isfinite(z_hat) & np.isfinite(vr)
            errs.append(np.abs(z_hat[m] - np.asarray(vr)[m]))
            n_units += int(m.sum())
        e = np.concatenate(errs) if errs else np.array([np.inf])
        ok = bool(n_units > 0 and n_len_mismatch == 0 and np.max(e) <= a.tol)
        row = {"tag": tag, "calibrated": cal, "raw_logits": logits, "calibration": c,
               "n_pairs": len(pids), "n_units": n_units, "n_pairs_len_mismatch": n_len_mismatch,
               "n_saturated_probs": n_clip,
               "max_abs_err": float(np.max(e)), "mean_abs_err": float(np.mean(e)),
               "p999_abs_err": float(np.percentile(e, 99.9)), "pass": ok}
        rep["files"].append(row); rep["pass"] = rep["pass"] and ok
        print(f"[a1:verify] {tag:34s} n={n_units:7d} max|err| {row['max_abs_err']:.3e} "
              f"mean {row['mean_abs_err']:.3e} saturated {n_clip} -> {'PASS' if ok else 'FAIL'}", flush=True)
    rep["n_files"] = len(rep["files"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"[a1:verify] overall {'PASS' if rep['pass'] else 'FAIL'} -> {a.out}", flush=True)
    if not rep["pass"]:
        print("[a1:verify] REFUSING to continue: the analytic inversion does not reproduce the stored raw "
              "logits. Route (i) of A1a is unusable; the arm must be rescored from its checkpoint.", flush=True)
        sys.exit(5)


# --------------------------------------------------------------------------- A1b: top-k set invariance
def _order_like_evaluator(sc: np.ndarray, starts: list[int], k: int) -> list[int]:
    """The exact tie-break mc_e2_eval.per_pair uses: descending score, then ascending unit start."""
    idx = range(len(sc))
    return sorted(idx, key=lambda j: (-sc[j] if np.isfinite(sc[j]) else 1.0, starts[j]))[:k]


def _mean_of(files: list[str]) -> dict[str, np.ndarray]:
    acc: dict[str, list[np.ndarray]] = {}
    for f in files:
        _, sc = common.read_scores(Path(f))
        for p, v in sc.items():
            acc.setdefault(p, []).append(np.asarray(v, dtype=float))
    return {p: np.nanmean(np.vstack(v), axis=0) for p, v in acc.items() if len({len(x) for x in v}) == 1}


def cmd_topk(a) -> None:
    name, path = a.units.split("=", 1)
    starts = {}
    for r in common.load_jsonl(Path(path)):
        if r.get("units"):
            n = len(r["source"])
            starts[r["pair_id"]] = [int(u.get("start", n // 2)) for u in r["units"]]
    rep = {"registered": "PREREG.md Wave A / A1b", "units": {"name": name, "path": path, "n_pairs": len(starts)},
           "k": a.k, "comparisons": {}}
    for spec in a.compare:
        label, rest = spec.split("=", 1)
        cal_s, raw_s = rest.split("|", 1)
        cal_f = [x for x in cal_s.split(",") if x]
        raw_f = [x for x in raw_s.split(",") if x]
        for f in cal_f + raw_f:
            assert Path(f).exists(), f"comparison {label!r}: missing score file {f}"
        A, B = _mean_of(cal_f), _mean_of(raw_f)
        pids = sorted(set(A) & set(B) & set(starts))
        n_set, n_order, n_eligible, ex = 0, 0, 0, []
        for p in pids:
            if len(A[p]) != len(starts[p]) or len(B[p]) != len(starts[p]):
                continue
            n_eligible += 1
            oa = _order_like_evaluator(A[p], starts[p], a.k)
            ob = _order_like_evaluator(B[p], starts[p], a.k)
            if set(oa) != set(ob):
                n_set += 1
                if len(ex) < 5:
                    ex.append({"pair_id": p, "only_calibrated": sorted(set(oa) - set(ob)),
                               "only_raw": sorted(set(ob) - set(oa))})
            if oa != ob:
                n_order += 1
        rep["comparisons"][label] = {
            "n_calibrated_files": len(cal_f), "n_raw_files": len(raw_f),
            "calibrated_files": [Path(f).name for f in cal_f], "raw_files": [Path(f).name for f in raw_f],
            "n_pairs_compared": n_eligible,
            "n_pairs_topk_set_changed": n_set,
            "n_pairs_topk_order_changed": n_order,
            "share_topk_set_changed": (n_set / n_eligible) if n_eligible else None,
            "examples": ex}
        print(f"[a1:topk] {label:34s} n={n_eligible:5d}  set-changed {n_set:5d} "
              f"({(100.0 * n_set / n_eligible) if n_eligible else float('nan'):.2f}%)  "
              f"order-changed {n_order:5d}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"[a1:topk] written {a.out}", flush=True)


# --------------------------------------------------------------------------- A1d: matched AUROC from raw logits
def auroc(y, s):
    """Tie-aware Mann-Whitney AUROC, identical to mc_matched_auroc.auroc."""
    y = np.asarray(y); s = np.asarray(s)
    if len(set(y.tolist())) < 2:
        return None
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s), float)
    sr = s[order]; i = 0
    ranks = np.arange(1, len(s) + 1, dtype=float)
    while i < len(s):
        j = i
        while j + 1 < len(s) and sr[j + 1] == sr[i]:
            j += 1
        r[order[i:j + 1]] = ranks[i:j + 1].mean(); i = j + 1
    npos = float((y == 1).sum()); nneg = float((y == 0).sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def cmd_auroc(a) -> None:
    """Each --arm is TAG:cal1,cal2,cal3:log1,log2,log3 -- the calibrated probability files and their raw-logit
    companions, in matching seed order. Three seed-aggregations are compared on identical data."""
    rows = [r for r in common.load_jsonl(Path(a.acu)) if r.get("units")]
    truth = {r["pair_id"]: common.truth_of(r) for r in rows}          # 1 = OMITTED
    doc_of = {r["pair_id"]: common.doc_key(r["source"]) for r in rows}
    arms: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    meta: dict[str, dict] = {}
    for spec in a.arm:
        tag, cal_s, log_s = spec.split(":", 2)
        cal_f = [x for x in cal_s.split(",") if x]
        log_f = [x for x in log_s.split(",") if x]
        assert len(cal_f) == len(log_f) and cal_f, f"arm {tag!r}: {len(cal_f)} calibrated vs {len(log_f)} logit files"
        seeds_c = [re.search(r"seed(\d+)", Path(f).name).group(1) for f in cal_f]
        seeds_l = [re.search(r"seed(\d+)", Path(f).name).group(1) for f in log_f]
        assert seeds_c == seeds_l, f"arm {tag!r}: seed order differs {seeds_c} vs {seeds_l}"
        assert len(set(seeds_c)) == len(seeds_c), f"arm {tag!r}: repeated seed in {seeds_c}"
        for f in cal_f + log_f:
            assert Path(f).exists(), f"arm {tag!r}: missing {f}"
        cal_per, log_per = defaultdict(list), defaultdict(list)
        per_seed_auroc = {}
        for s, fc, fl in zip(seeds_c, cal_f, log_f):
            _, sc = common.read_scores(Path(fc)); _, sl = common.read_scores(Path(fl))
            for pid in sc:
                if pid in truth and len(sc[pid]) == len(truth[pid]) and pid in sl and len(sl[pid]) == len(truth[pid]):
                    cal_per[pid].append(np.asarray(sc[pid], float))
                    log_per[pid].append(np.asarray(sl[pid], float))
            ps = sorted(set(sl) & set(truth))
            ps = [p for p in ps if len(sl[p]) == len(truth[p])]
            per_seed_auroc[s] = auroc(np.concatenate([truth[p] for p in ps]),
                                      np.concatenate([np.asarray(sl[p], float) for p in ps]))
        keep = [p for p in cal_per if len(cal_per[p]) == len(cal_f) and len(log_per[p]) == len(cal_f)]
        arms[tag] = {
            "cal_prob": {p: np.nanmean(np.vstack(cal_per[p]), axis=0) for p in keep},
            "raw_logit": {p: np.nanmean(np.vstack(log_per[p]), axis=0) for p in keep},
            "raw_prob": {p: np.nanmean(1.0 / (1.0 + np.exp(-np.vstack(log_per[p]))), axis=0) for p in keep},
        }
        meta[tag] = {"seeds": seeds_c, "calibrated_files": [Path(f).name for f in cal_f],
                     "logit_files": [Path(f).name for f in log_f], "n_pairs": len(keep),
                     "per_seed_auroc_pooled": per_seed_auroc}
        print(f"[a1:auroc] arm {tag:34s} seeds {seeds_c} pairs {len(keep)} "
              f"per-seed AUROC {[None if v is None else round(v, 4) for v in per_seed_auroc.values()]}", flush=True)
    names = sorted(arms)
    assert a.primary in arms, f"{a.primary!r} not in {names}"
    pids = sorted(set.intersection(*[set(arms[t]["cal_prob"]) for t in names]) & set(truth))
    docs = sorted({doc_of[p] for p in pids})
    by_doc = defaultdict(list)
    for p in pids:
        by_doc[doc_of[p]].append(p)

    def pooled(mode, t, ps):
        return auroc(np.concatenate([truth[p] for p in ps]), np.concatenate([arms[t][mode][p] for p in ps]))

    rng = np.random.default_rng(a.seed)
    boot = [rng.choice(docs, size=len(docs), replace=True) for _ in range(a.n_boot)]
    rep = {"registered": "PREREG.md Wave A / A1d", "acu": str(a.acu), "n_pairs": len(pids), "n_docs": len(docs),
           "primary": a.primary, "n_boot": a.n_boot, "boot_seed": a.seed, "arm_meta": meta,
           "note": "'observed' is the contrast computed once on the full sample; 'bootstrap_mean' is the mean of "
                   "the resampled contrasts and is NOT the observed value (mc_e2_eval/mc_matched_auroc print the "
                   "latter under the key 'mean').",
           "modes": {}}
    for mode in ("cal_prob", "raw_prob", "raw_logit"):
        m = {"arms": {}, "paired_vs_primary": {}}
        for t in names:
            m["arms"][t] = {
                "auroc_pooled": pooled(mode, t, pids),
                "auroc_doc_macro": float(np.nanmean([x for x in (pooled(mode, t, by_doc[d]) for d in docs) if x is not None]))}
            print(f"[a1:auroc {mode:9s}] {t:34s} pooled {m['arms'][t]['auroc_pooled']:.4f} "
                  f"macro {m['arms'][t]['auroc_doc_macro']:.4f}", flush=True)
        for t in names:
            if t == a.primary:
                continue
            obs = m["arms"][a.primary]["auroc_pooled"] - m["arms"][t]["auroc_pooled"]
            d = []
            for pick in boot:
                sub = [p for dd in pick for p in by_doc[dd]]
                x, y = pooled(mode, a.primary, sub), pooled(mode, t, sub)
                if x is not None and y is not None:
                    d.append(x - y)
            m["paired_vs_primary"][t] = {
                "observed": float(obs), "bootstrap_mean": float(np.mean(d)),
                "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "n_boot": len(d)}
            r = m["paired_vs_primary"][t]
            print(f"[a1:auroc {mode:9s}] {a.primary} - {t:30s} observed {obs:+.4f} "
                  f"[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] (boot mean {r['bootstrap_mean']:+.4f})", flush=True)
        rep["modes"][mode] = m
    # headline: how much the aggregation contract moves the matched label-source effect
    rep["contract_shift"] = {}
    for t in names:
        if t == a.primary:
            continue
        old = rep["modes"]["cal_prob"]["paired_vs_primary"][t]["observed"]
        new = rep["modes"]["raw_logit"]["paired_vs_primary"][t]["observed"]
        rep["contract_shift"][t] = {"calibrated_prob_mean": old, "raw_logit_mean": new, "delta": new - old}
        print(f"[a1:auroc shift] {t:34s} calibrated {old:+.4f} -> raw-logit {new:+.4f} (delta {new - old:+.4f})", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"[a1:auroc] written {a.out}", flush=True)


# --------------------------------------------------------------------------- collect: side-by-side of eval runs
def cmd_collect(a) -> None:
    """--eval LABEL=<dir or e2_emitted.json> (repeatable); --baseline LABEL names the reference column."""
    runs = {}
    for spec in a.eval:
        label, p = spec.split("=", 1)
        f = Path(p)
        if f.is_dir():
            f = f / "e2_emitted.json"
        assert f.exists(), f"eval {label!r}: missing {f}"
        runs[label] = {"path": str(f), "report": json.loads(f.read_text())}
    assert a.baseline in runs, f"baseline {a.baseline!r} not among {sorted(runs)}"
    kl = str(a.k)
    rep = {"registered": a.registered, "k": kl, "baseline": a.baseline,
           "runs": {lab: {"path": r["path"], "n_pairs": r["report"].get("n_pairs"),
                          "n_docs": r["report"].get("n_docs"),
                          "strict_seeds": r["report"].get("strict_seeds"),
                          "expect_seeds": r["report"].get("expect_seeds")} for lab, r in runs.items()},
           "cells": {}}
    base = runs[a.baseline]["report"]["k"][kl]["systems"]
    allnames = sorted(set().union(*[set(r["report"]["k"][kl]["systems"]) for r in runs.values()]))
    for n in allnames:
        row = {}
        for lab, r in runs.items():
            s = r["report"]["k"][kl]["systems"].get(n)
            if s is None:
                continue
            row[lab] = {"recall": s.get("recall"), "omission_precision": s.get("omission_precision"),
                        "recall_stress": s.get("recall_stress"), "hit_rate": s.get("hit_rate"),
                        "false_alert_rate": s.get("false_alert_rate"), "n_pairs": s.get("n_pairs")}
        if a.baseline in row:
            for lab in row:
                if lab == a.baseline:
                    continue
                for key in ("recall", "omission_precision"):
                    x, y = row[lab].get(key), row[a.baseline].get(key)
                    row[lab][f"delta_{key}_vs_{a.baseline}"] = (None if x is None or y is None else x - y)
        rep["cells"][n] = row
    labs = [a.baseline] + [l for l in sorted(runs) if l != a.baseline]
    hdr = "  ".join(f"{l:>12s}" for l in labs)
    print(f"[a1:collect] k={kl}  {'system':40s} {hdr}   delta(recall)", flush=True)
    for n in sorted(allnames, key=lambda x: -(rep["cells"][x].get(a.baseline, {}).get("recall") or 0)):
        row = rep["cells"][n]
        vals = "  ".join((f"{row[l]['recall']:12.4f}" if l in row and row[l]["recall"] is not None else " " * 12) for l in labs)
        dl = [row[l].get(f"delta_recall_vs_{a.baseline}") for l in labs[1:] if l in row]
        ds = " ".join(f"{d:+.4f}" for d in dl if d is not None)
        print(f"[a1:collect] {n:40s} {vals}   {ds}", flush=True)
    # spread of the cells named by --spread-over (A1c: the label-efficiency curve)
    if a.spread_over:
        rep["spread"] = {}
        for lab in labs:
            vs = [rep["cells"][n][lab]["recall"] for n in a.spread_over
                  if lab in rep["cells"].get(n, {}) and rep["cells"][n][lab]["recall"] is not None]
            if len(vs) == len(a.spread_over):
                rep["spread"][lab] = {"points": dict(zip(a.spread_over, vs)), "min": min(vs), "max": max(vs),
                                      "spread": max(vs) - min(vs), "mean": float(np.mean(vs))}
                print(f"[a1:collect spread] {lab:14s} " +
                      " ".join(f"{n}={v:.4f}" for n, v in zip(a.spread_over, vs)) +
                      f"  spread {max(vs) - min(vs):.4f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"[a1:collect] written {a.out}", flush=True)


# --------------------------------------------------------------------------- A1e: leadunion bound
def cmd_leadcheck(a) -> None:
    """A1e: the leadunion aggregation sorts primarily on per-seed sentence-leader COUNTS, which no strictly
    monotone per-seed map can change, so its cell is calibration-invariant up to the tie-break. This reports
    (i) how far leadunion sits from the consensus headline inside each contract and (ii) how far the leadunion
    cell itself moves between the two contracts -- the bound on the paper's calibration exposure."""
    runs, paths = {}, {}
    for spec in a.eval:
        label, p = spec.split("=", 1)
        f = Path(p)
        if f.is_dir():
            f = f / "e2_emitted.json"
        assert f.exists(), f"eval {label!r}: missing {f}"
        runs[label] = json.loads(f.read_text()); paths[label] = str(f)
    kl = str(a.k)
    rep = {"registered": "PREREG.md Wave A / A1e", "split": a.split, "k": kl,
           "consensus": a.consensus, "lead": a.lead, "single": a.single, "runs": paths,
           "within_contract": {}, "across_contract": {}}
    for lab, rep_j in runs.items():
        sysd = rep_j["k"][kl]["systems"]
        row = {}
        for role, nm in (("consensus", a.consensus), ("lead", a.lead), ("single", a.single)):
            if nm and nm in sysd:
                row[role] = {"system": nm, "recall": sysd[nm]["recall"],
                             "omission_precision": sysd[nm].get("omission_precision")}
        if "consensus" in row and "lead" in row:
            row["abs_diff_lead_minus_consensus_recall"] = abs(row["lead"]["recall"] - row["consensus"]["recall"])
            row["lead_minus_consensus_recall"] = row["lead"]["recall"] - row["consensus"]["recall"]
        rep["within_contract"][lab] = row
        print(f"[a1:lead] {lab:12s} consensus {row.get('consensus', {}).get('recall')} "
              f"lead {row.get('lead', {}).get('recall')} "
              f"|diff| {row.get('abs_diff_lead_minus_consensus_recall')}", flush=True)
    labs = sorted(runs)
    if len(labs) == 2:
        x, y = labs
        for role in ("consensus", "lead", "single"):
            rx = rep["within_contract"][x].get(role)
            ry = rep["within_contract"][y].get(role)
            if rx and ry:
                rep["across_contract"][role] = {x: rx["recall"], y: ry["recall"],
                                                "abs_delta": abs(rx["recall"] - ry["recall"])}
                print(f"[a1:lead] across-contract {role:10s} {x} {rx['recall']:.6f} vs {y} {ry['recall']:.6f} "
                      f"|delta| {abs(rx['recall'] - ry['recall']):.6f}", flush=True)
    if a.external:
        rep["external_reference"] = {}
        for spec in a.external:
            nm, con, lea = spec.split(":", 2)
            rep["external_reference"][nm] = {"consensus": float(con), "lead": float(lea),
                                             "abs_diff": abs(float(con) - float(lea))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"[a1:lead] written {a.out}", flush=True)


# --------------------------------------------------------------------------- cli
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("invert", help="invert one arm's Platt map on one calibrated unit_scores file")
    i.add_argument("--in", dest="inp", required=True)
    i.add_argument("--result-json", required=True, help="<trainout>/result.json of the SAME arm and seed")
    i.add_argument("--out", required=True)
    i.add_argument("--record", default="", help="optional json describing the inversion")

    v = sub.add_parser("verify", help="A1a gate: the inversion must reproduce the stored raw logits")
    v.add_argument("--triple", nargs="+", required=True, help="TAG:CALIBRATED:RAW_LOGITS:RESULT_JSON")
    v.add_argument("--tol", type=float, default=1e-6)
    v.add_argument("--out", required=True)

    t = sub.add_parser("topk", help="A1b: pairs whose top-k SET changes between two score rosters")
    t.add_argument("--units", required=True, help="name=path of the inventory the scores refer to")
    t.add_argument("--k", type=int, default=10)
    t.add_argument("--compare", nargs="+", required=True, help="LABEL=cal1,cal2,cal3|raw1,raw2,raw3")
    t.add_argument("--out", required=True)

    u = sub.add_parser("auroc", help="A1d: matched AUROC under calibrated-prob / raw-prob / raw-logit averaging")
    u.add_argument("--acu", required=True)
    u.add_argument("--arm", nargs="+", required=True, help="TAG:cal1,cal2,cal3:log1,log2,log3")
    u.add_argument("--primary", required=True)
    u.add_argument("--n-boot", type=int, default=2000)
    u.add_argument("--seed", type=int, default=20260917)
    u.add_argument("--out", required=True)

    c = sub.add_parser("collect", help="side-by-side of the same system names across mc_e2_eval outputs")
    c.add_argument("--eval", nargs="+", required=True, help="LABEL=<eval dir or e2_emitted.json>")
    c.add_argument("--baseline", required=True)
    c.add_argument("--k", type=int, default=10)
    c.add_argument("--spread-over", nargs="*", default=[], help="system names whose min/max spread to report")
    c.add_argument("--registered", default="PREREG.md Wave A / A1")
    c.add_argument("--out", required=True)

    l = sub.add_parser("leadcheck", help="A1e: leadunion vs consensus, within and across the two contracts")
    l.add_argument("--eval", nargs="+", required=True, help="LABEL=<eval dir or e2_emitted.json>")
    l.add_argument("--k", type=int, default=10)
    l.add_argument("--split", default="validation")
    l.add_argument("--consensus", default="humanfact+div1")
    l.add_argument("--lead", default="humanfact.lead+div1")
    l.add_argument("--single", default="humanfact.single+div1")
    l.add_argument("--external", nargs="*", default=[],
                   help="POOL:CONSENSUS_RECALL:LEAD_RECALL frozen numbers from the external pools, for the bound")
    l.add_argument("--out", required=True)

    a = ap.parse_args()
    {"invert": cmd_invert, "verify": cmd_verify, "topk": cmd_topk, "auroc": cmd_auroc,
     "collect": cmd_collect, "leadcheck": cmd_leadcheck}[a.cmd](a)


if __name__ == "__main__":
    main()
