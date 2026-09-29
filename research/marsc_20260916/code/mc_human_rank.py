#!/usr/bin/env python3
"""MARS-C R02 -- rank-recovered human precision (FULL_REVIEW_2026-09-17 decision-critical failure 2).

Defect. G3 asked the evaluator for k=5,10,20; `mc_e2_eval.py` dumps the emitted units of ONE budget, max(k)=20.
`mc_human_sample.py` then drew the blinded sample from that top-20 dump and kept only (system, pair_id, status,
fact, score) -- rank was discarded -- while `mc_human_analyze.py` reweighted the strata by the k=10 status mix.
The published 0.85/0.82/0.74/0.64 therefore weighted top-20 annotations by top-10 weights and identify neither
budget's precision.

Repair. The dump is ordered by the emission rule, so position in `emitted` IS the rank. This script joins the
sample key back onto the canonical dump on (system, pair_id, fact text, score), recovers each sampled item's
rank, and estimates precision separately at the budget the paper reports (rank <= 10) and at the sampled budget
(rank <= 20), with stratified bootstrap intervals. Weights are the evaluator's own status shares at that budget,
held fixed (they are estimated on every emitted unit of ~1.4k pairs, so their sampling error is second order
next to the 20-60 annotated items per stratum; this is stated, not assumed away).

Reported per system: per-annotator, adjudicated (items both annotators labelled the same way) and the two
one-sided bounds (either / both annotators call it a confirmed omission).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

STATUSES = ("unknown", "hit", "false_alert")


def load_jsonl(p: Path):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def build_rank_index(emitted_files: list[Path]) -> dict:
    """(system, pair_id) -> ordered emitted list. Later files win only if they add unseen keys."""
    idx: dict[tuple[str, str], list[dict]] = {}
    for f in emitted_files:
        for r in load_jsonl(f):
            key = (r["system"], r["pair_id"])
            if r.get("emitted") and key not in idx:
                idx[key] = r["emitted"]
    return idx


def recover_rank(item: dict, idx: dict) -> dict | None:
    """Rank of the sampled unit inside its system's emitted list (1 = top). None if the item cannot be joined."""
    em = idx.get((item["system"], item["pair_id"]))
    if not em:
        return None
    cands = [(i, e) for i, e in enumerate(em) if e["text"] == item["fact"]]
    if len(cands) > 1 and item.get("score") is not None:
        tight = [(i, e) for i, e in cands if e.get("score") is not None and abs(e["score"] - item["score"]) < 1e-9]
        if tight:
            cands = tight
    if not cands:
        return None
    i, e = cands[0]
    return {"rank": i + 1, "status_dump": e["status"], "ambiguous": len(cands) > 1, "n_emitted": len(em)}


def wilson(k: int, n: int, z: float = 1.959963985) -> list[float] | None:
    if n == 0:
        return None
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def weighted_precision(per_status: dict[str, list[bool]], w: dict[str, float]) -> float | None:
    """Sum_status w_status * confirmed-omission rate, renormalised over the strata that carry annotations."""
    have = [s for s in w if per_status.get(s)]
    if not have:
        return None
    z = sum(w[s] for s in have)
    if z <= 0:
        return None
    return float(sum((w[s] / z) * (sum(per_status[s]) / len(per_status[s])) for s in have))


def boot_ci(per_status: dict[str, list[bool]], w: dict[str, float], rng, n_boot: int) -> dict:
    """Stratified nonparametric bootstrap: resample annotated items within each status, reweight, percentile CI."""
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
            "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))] if draws else None,
            "n_boot": len(draws)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emitted", nargs="+", required=True, help="canonical emitted dump(s) the sample was drawn from")
    ap.add_argument("--key", nargs="+", required=True, help="blinded sample key file(s)")
    ap.add_argument("--export", nargs="+", required=True, help="annotator export file(s)")
    ap.add_argument("--e2-json", required=True, help="evaluator json supplying the per-budget status mix (weights)")
    ap.add_argument("--budgets", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    idx = build_rank_index([Path(f) for f in a.emitted])
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

    # ---- rank recovery
    joined, unjoined = {}, []
    for iid, it in key.items():
        rk = recover_rank(it, idx)
        if rk is None:
            unjoined.append(iid)
        else:
            joined[iid] = {**it, **rk}
    status_check = Counter((joined[i]["status"] == joined[i]["status_dump"]) for i in joined)
    rep = {"registered": "research/marsc_20260916/PREREG.md R02 (repair of G6 estimand)",
           "n_key_items": len(key), "n_joined": len(joined), "n_unjoined": len(unjoined),
           "unjoined_item_ids": unjoined[:50],
           "n_ambiguous_text_match": sum(1 for i in joined if joined[i]["ambiguous"]),
           "status_agrees_with_dump": {str(k): v for k, v in status_check.items()},
           "rank_histogram": dict(sorted(Counter(joined[i]["rank"] for i in joined).items())),
           "rank_le_10_by_system_status": {}, "budgets": {}, "annotators": {n: len(lab[n]) for n in annotators}}
    per_sys_status = defaultdict(Counter)
    for i, it in joined.items():
        if it["rank"] <= 10:
            per_sys_status[it["system"]][it["status"]] += 1
    rep["rank_le_10_by_system_status"] = {s: dict(c) for s, c in per_sys_status.items()}

    verdict = lambda r: r["q1"] == "A" and r["q2"] == "B"
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
                     "n_by_status": dict(Counter(joined[i]["status"] for i in items))}
            for n in annotators:
                ps = defaultdict(list)
                for i in items:
                    if i in lab[n]:
                        ps[joined[i]["status"]].append(verdict(lab[n][i]))
                entry[n] = {**boot_ci(ps, w, rng, a.n_boot),
                            "raw_by_status": {s: {"n": len(v), "rate": sum(v) / len(v), "wilson": wilson(sum(v), len(v))}
                                              for s, v in sorted(ps.items())}}
            if len(annotators) >= 2:
                x, y = annotators[0], annotators[1]
                for mode, keep in (("both_agree", lambda vx, vy: vx == vy),
                                   ("either", lambda vx, vy: True), ("both_positive", lambda vx, vy: True)):
                    ps = defaultdict(list)
                    for i in items:
                        if i in lab[x] and i in lab[y]:
                            vx, vy = verdict(lab[x][i]), verdict(lab[y][i])
                            if mode == "both_agree":
                                if vx == vy:
                                    ps[joined[i]["status"]].append(vx)
                            elif mode == "either":
                                ps[joined[i]["status"]].append(vx or vy)
                            else:
                                ps[joined[i]["status"]].append(vx and vy)
                    entry[mode] = {**boot_ci(ps, w, rng, a.n_boot),
                                   "n_items": sum(len(v) for v in ps.values())}
            out_b["systems"][sysname] = entry
        rep["budgets"][str(kb)] = out_b

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "human_precision_ranked.json").write_text(json.dumps(rep, indent=1))
    print(f"[r02] joined {len(joined)}/{len(key)} sampled items to the canonical dump "
          f"(ambiguous text matches: {rep['n_ambiguous_text_match']}, status agrees: {rep['status_agrees_with_dump']})", flush=True)
    print(f"[r02] rank histogram: {rep['rank_histogram']}", flush=True)
    for kb in a.budgets:
        print(f"--- budget k<={kb}", flush=True)
        for s, e in rep["budgets"][str(kb)]["systems"].items():
            parts = []
            for n in annotators + ["both_agree", "either", "both_positive"]:
                d = e.get(n)
                if isinstance(d, dict) and d.get("point") is not None:
                    ci = d["ci"]; parts.append(f"{n} {d['point']:.3f} [{ci[0]:.3f},{ci[1]:.3f}]" if ci else f"{n} {d['point']:.3f}")
            print(f"[r02 k<={kb}] {s:30s} n={e['n_items_in_budget']:3d} {e['n_by_status']} | " + " | ".join(parts), flush=True)
    print(f"[r02] written {out / 'human_precision_ranked.json'}", flush=True)


if __name__ == "__main__":
    main()
