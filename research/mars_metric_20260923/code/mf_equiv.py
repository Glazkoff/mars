#!/usr/bin/env python3
"""M-F -- equivalence check: does the released `mars/` package (PromptVerifier, PairVerifier) reproduce the
ad-hoc research-script scores (ma_score.py's FactCG/MarscWindowed) that the paper's M-A numbers come from, on
200 random pairs, to within 1e-3? Same verifier checkpoints, same cached decomposition (props), same
fallback/aggregation rules ma_score.py uses. Registered bar in PREREG.md (M-F): the check passes, or the
package is fixed until it does.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from mars.verifiers import PairVerifier, PromptVerifier  # noqa: E402


def load_props(patterns: list[str]) -> dict[str, list[str]]:
    by_key: dict[str, list[tuple[int, list[str]]]] = defaultdict(list)
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                sent = int(r["task_id"].rsplit(":", 1)[1])
                by_key[r["doc_key"]].append((sent, [p for p in r.get("props", []) if p and p.strip().upper() != "NONE"]))
    return {k: [p for _, ps in sorted(v) for p in ps] for k, v in by_key.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--props", nargs="*", default=[])
    ap.add_argument("--score-p", required=True, help="scores_marsP_factcg.jsonl")
    ap.add_argument("--score-r", required=True, help="scores_marsR_marsc.jsonl")
    ap.add_argument("--factcg-snapshot", required=True)
    ap.add_argument("--marsc-ckpts", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pairs = {json.loads(line)["pair_id"]: json.loads(line) for line in open(a.pairs, encoding="utf-8")}
    stored_p = {json.loads(line)["pair_id"]: json.loads(line) for line in open(a.score_p, encoding="utf-8")}
    stored_r = {json.loads(line)["pair_id"]: json.loads(line) for line in open(a.score_r, encoding="utf-8")}
    common = sorted(set(pairs) & set(stored_p) & set(stored_r))
    rng = random.Random(a.seed)
    sample = rng.sample(common, min(a.n, len(common)))
    props = load_props(a.props)

    p_ver = PromptVerifier(a.factcg_snapshot, batch=32)
    r_ver = PairVerifier.from_glob(a.marsc_ckpts, batch=128)

    rows = []
    prem_p, hyp_p, own_p = [], [], []
    prem_r, hyp_r, own_r = [], [], []
    for i, pid in enumerate(sample):
        p = pairs[pid]
        cand_facts = props.get(p["cand_key"], []) or [p["candidate"]]
        for f in cand_facts:
            prem_p.append(p["source"]); hyp_p.append(f); own_p.append(i)
        ref_facts_sets = [p["acus"]] if p.get("acus") else ([props.get(k, []) for k in p["ref_keys"]] if p["ref_keys"] else [])
        ref_facts_sets = [fs for fs in ref_facts_sets if fs]
        for k, fs in enumerate(ref_facts_sets):
            for f in fs:
                prem_r.append(p["candidate"]); hyp_r.append(f); own_r.append((i, k))
        rows.append({"pair_id": pid, "n_ref_sets": len(ref_facts_sets)})

    sup_p = p_ver.support(prem_p, hyp_p)
    acc_p: dict[int, list[float]] = defaultdict(list)
    for o, s in zip(own_p, sup_p):
        acc_p[o].append(s)

    sup_r = r_ver.support(prem_r, hyp_r)
    acc_r: dict[tuple[int, int], list[float]] = defaultdict(list)
    for o, s in zip(own_r, sup_r):
        acc_r[o].append(s)

    n_ok_p = n_ok_r = n_checked_p = n_checked_r = 0
    max_diff_p = max_diff_r = 0.0
    for i, pid in enumerate(sample):
        r = rows[i]
        vals = acc_p.get(i, [])
        computed_p = sum(vals) / len(vals) if vals else None
        stored_pv = stored_p[pid].get("marsP_factcg")
        if computed_p is not None and stored_pv is not None:
            diff = abs(computed_p - stored_pv)
            max_diff_p = max(max_diff_p, diff); n_checked_p += 1; n_ok_p += int(diff < a.tol)
        r["computed_marsP_factcg"] = computed_p; r["stored_marsP_factcg"] = stored_pv

        best = None
        for k in range(r["n_ref_sets"]):
            items = acc_r.get((i, k), [])
            if items:
                v = sum(items) / len(items)
                if best is None or v > best:
                    best = v
        stored_rv = stored_r[pid].get("marsR_marsc")
        if best is not None and stored_rv is not None:
            diff = abs(best - stored_rv)
            max_diff_r = max(max_diff_r, diff); n_checked_r += 1; n_ok_r += int(diff < a.tol)
        r["computed_marsR_marsc"] = best; r["stored_marsR_marsc"] = stored_rv

    report = {
        "n_sampled": len(sample), "tol": a.tol,
        "marsP_factcg": {"n_checked": n_checked_p, "n_within_tol": n_ok_p, "max_abs_diff": max_diff_p,
                          "pass": n_checked_p > 0 and n_ok_p == n_checked_p},
        "marsR_marsc": {"n_checked": n_checked_r, "n_within_tol": n_ok_r, "max_abs_diff": max_diff_r,
                         "pass": n_checked_r > 0 and n_ok_r == n_checked_r},
        "rows": rows,
    }
    report["pass"] = report["marsP_factcg"]["pass"] and report["marsR_marsc"]["pass"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"[mf-equiv] P: {n_ok_p}/{n_checked_p} within {a.tol}, max diff {max_diff_p:.6f}; "
          f"R: {n_ok_r}/{n_checked_r} within {a.tol}, max diff {max_diff_r:.6f}; "
          f"overall {'PASS' if report['pass'] else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
