#!/usr/bin/env python3
"""B3 -- OBSERVED paired contrasts with document-clustered intervals, recomputed from the evaluator's per-pair dump.

Why this file exists at all.  `mc_e2_eval.py` writes `paired_recall[<system>]["mean"]`, and that field is the
BOOTSTRAP mean of the resampled difference, not the observed contrast.  Reading it as the observed difference
has already produced one retracted claim in this project (NUMBERS_PROVENANCE_2026-09-17, corrections table:
"+0.121 (bootstrap mean)" -> "+0.122 observed").  B3 is a repair-plus-extension whose entire value is the sign
and the interval, so the observed contrast has to be computed explicitly and printed next to the bootstrap
mean, with the gap between them visible.

What it does.  It re-derives `summarise()` exactly from `--dump-per-pair` rows (which carry the document key),
cross-checks every re-derived point estimate against the evaluator's own `e2_emitted.json` to 1e-9, and then
runs its own document-clustered paired bootstrap for:
  * recall@k              : mean over pairs that have >= 1 human-omitted ACU
  * omission precision    : sum(hit * n_emitted) / sum((hit + false_alert) * n_emitted), unit-weighted
reporting, per contrast, the OBSERVED difference, the bootstrap mean, the percentile CI, the basic
(reverse-percentile) CI re-centred on the observed value, and the one-sided 95% lower bound.

Nothing here re-reads a split, loads a model or re-scores anything; it is pure re-analysis of artefacts the
evaluator already wrote.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

K_LABEL = "10"


def load_dump(path: Path) -> tuple[dict[str, dict[str, dict]], dict[str, str]]:
    """-> ({system: {pair_id: row}}, {pair_id: doc_key}) for the single registered k label."""
    per: dict[str, dict[str, dict]] = defaultdict(dict)
    doc_of: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if str(r.get("k")) != K_LABEL:
                continue
            per[r["system"]][r["pair_id"]] = r
            doc_of[r["pair_id"]] = r["doc"]
    return dict(per), doc_of


def summarise(rows: list[dict]) -> dict:
    """Byte-for-byte the evaluator's `summarise` for the two quantities B3 reports."""
    rec = [r["recall"] for r in rows if r["recall"] is not None]
    h = sum(r["hit"] * r["n_emitted"] for r in rows)
    f = sum(r["false_alert"] * r["n_emitted"] for r in rows)
    return {"recall": float(np.mean(rec)) if rec else None,
            "omission_precision": (h / (h + f)) if (h + f) > 0 else None,
            "n_pairs": len(rows), "n_pairs_recall_eligible": len(rec)}


def _doc_blocks(rows: list[dict], pids: list[str], doc_of: dict[str, str], docs: list[str]):
    """Per-document partial sums. Every quantity the evaluator averages is a ratio of additive per-pair terms,
    so resampling documents with replacement is exactly a resample of these per-document sums."""
    di = {d: i for i, d in enumerate(docs)}
    rec_sum = np.zeros(len(docs)); rec_n = np.zeros(len(docs))
    num = np.zeros(len(docs)); den = np.zeros(len(docs)); npair = np.zeros(len(docs))
    for p in pids:
        r = rows[p]; i = di[doc_of[p]]
        npair[i] += 1
        if r["recall"] is not None:
            rec_sum[i] += r["recall"]; rec_n[i] += 1
        num[i] += r["hit"] * r["n_emitted"]
        den[i] += (r["hit"] + r["false_alert"]) * r["n_emitted"]
    return {"rec_sum": rec_sum, "rec_n": rec_n, "num": num, "den": den, "n_pairs": npair}


def _stats(b: dict, pick: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """recall and omission precision, either on the full sample (pick=None) or on each bootstrap row of `pick`."""
    if pick is None:
        rs, rn, nu, de = b["rec_sum"].sum(), b["rec_n"].sum(), b["num"].sum(), b["den"].sum()
    else:
        rs, rn = b["rec_sum"][pick].sum(1), b["rec_n"][pick].sum(1)
        nu, de = b["num"][pick].sum(1), b["den"][pick].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"recall": np.where(np.asarray(rn) > 0, np.asarray(rs) / np.where(np.asarray(rn) > 0, rn, 1), np.nan),
                "omission_precision": np.where(np.asarray(de) > 0, np.asarray(nu) / np.where(np.asarray(de) > 0, de, 1), np.nan)}


def contrast(prim: dict[str, dict], other: dict[str, dict], pids: list[str], doc_of: dict[str, str],
             n_boot: int, seed: int) -> dict:
    """Observed difference prim - other, plus a document-clustered paired bootstrap around it.

    The clustering variable is the evaluator's own `common.doc_key(source)`, carried through in the per-pair
    dump, not `pair_id`: RoSE contributes 8-12 summaries of one source document, so a pair-level resample is
    anti-conservative by up to sqrt(n_pairs / n_docs).
    """
    docs = sorted({doc_of[p] for p in pids})
    bp, bo = _doc_blocks(prim, pids, doc_of, docs), _doc_blocks(other, pids, doc_of, docs)
    op, oo = _stats(bp), _stats(bo)
    obs = {q: (float(op[q] - oo[q]) if np.isfinite(op[q]) and np.isfinite(oo[q]) else None)
           for q in ("recall", "omission_precision")}
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(docs), size=(int(n_boot), len(docs)))
    sp, so = _stats(bp, pick), _stats(bo, pick)
    out = {"n_pairs": len(pids), "n_docs": len(docs), "observed": obs,
           "prim": summarise([prim[p] for p in pids]), "other": summarise([other[p] for p in pids]),
           "n_boot": int(n_boot), "bootstrap_seed": int(seed)}
    for q in ("recall", "omission_precision"):
        d = (sp[q] - so[q]); d = d[np.isfinite(d)]
        if d.size == 0 or obs[q] is None:
            out[q] = None
            continue
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        out[q] = {"observed": float(obs[q]), "bootstrap_mean": float(d.mean()),
                  "ci_percentile": [lo, hi],
                  # basic / reverse-percentile interval, re-centred on the observed statistic; reported as a
                  # sensitivity because the percentile interval is the project's house estimator
                  "ci_basic": [float(2 * obs[q] - hi), float(2 * obs[q] - lo)],
                  "one_sided_lower_95": float(np.percentile(d, 5.0)),
                  "excludes_zero": bool(lo > 0.0 or hi < 0.0),
                  "n_boot_used": int(d.size),
                  "bootstrap_mean_minus_observed": float(d.mean() - obs[q])}
    return out


def crosscheck(evaldir: Path, systems: dict[str, dict], pids: list[str], tol: float) -> list[str]:
    """Every re-derived point estimate must reproduce the evaluator's own JSON, or we do not trust either."""
    notes: list[str] = []
    rep = json.loads((evaldir / "e2_emitted.json").read_text())
    ref = rep["k"][K_LABEL]["systems"]
    for name, rows in systems.items():
        mine = summarise([rows[p] for p in pids])
        for q in ("recall", "omission_precision"):
            a, b = mine[q], ref.get(name, {}).get(q)
            if a is None or b is None:
                notes.append(f"{name}.{q}: one side is null (mine={a}, evaluator={b})")
            elif abs(a - b) > tol:
                notes.append(f"{name}.{q}: MISMATCH mine={a!r} evaluator={b!r} delta={a - b:.3e}")
    return notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", nargs="+", required=True,
                    help="LABEL=<eval dir>:<per-pair dump jsonl>  (the dir must hold e2_emitted.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=20260918)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--skip-missing", action="store_true",
                    help="tolerate an eval that did not run (a wall-clock casualty) instead of failing the job")
    a = ap.parse_args()
    report: dict[str, dict] = {"registered": "research/marsc_strengthen_20260918/PREREG.md B3",
                               "k": K_LABEL, "n_boot": int(a.n_boot), "boot_seed": int(a.boot_seed),
                               "note": ("`observed` is the difference of the two point estimates; `bootstrap_mean` is the "
                                        "mean of the resampled differences. They are not the same quantity and the "
                                        "manuscript must quote `observed`."),
                               "evals": {}, "crosscheck": {}, "missing": []}
    for spec in a.eval:
        label, rest = spec.split("=", 1)
        d, dump = rest.rsplit(":", 1)
        evaldir, dumpf = Path(d), Path(dump)
        if not (evaldir / "e2_emitted.json").exists() or not dumpf.exists():
            msg = f"{label}: missing {evaldir/'e2_emitted.json'} or {dumpf}"
            if a.skip_missing:
                print(f"[b3] SKIP {msg}", flush=True); report["missing"].append(msg); continue
            raise SystemExit(f"[b3] {msg}")
        systems, doc_of = load_dump(dumpf)
        rep = json.loads((evaldir / "e2_emitted.json").read_text())
        prim = rep["k"][K_LABEL]["primary"]
        pids = sorted(set.intersection(*[set(v) for v in systems.values()]))
        notes = crosscheck(evaldir, systems, pids, a.tol)
        report["crosscheck"][label] = notes or "ok"
        for n in notes:
            print(f"[b3:crosscheck] {label}: {n}", flush=True)
        block = {"eval_dir": str(evaldir), "primary": prim, "n_pairs": len(pids),
                 "systems": {n: summarise([systems[n][p] for p in pids]) for n in sorted(systems)},
                 "contrasts": {}}
        for n in sorted(systems):
            if n == prim:
                continue
            c = contrast(systems[prim], systems[n], pids, doc_of, a.n_boot, a.boot_seed)
            ev = rep["k"][K_LABEL]
            c["evaluator_paired_recall"] = ev.get("paired_recall", {}).get(n)
            c["evaluator_paired_omission_precision"] = ev.get("paired_omission_precision", {}).get(n)
            c["evaluator_sign_test"] = ev.get("sign_test", {}).get(n)
            c["evaluator_perm_test"] = ev.get("perm_test", {}).get(n)
            block["contrasts"][n] = c
        report["evals"][label] = block
        print(f"\n===== {label}  primary={prim}  n_pairs={len(pids)}  n_docs={len(set(doc_of.values()))}", flush=True)
        for n, s in sorted(block["systems"].items(), key=lambda kv: -(kv[1]["recall"] or 0)):
            op = "   None" if s["omission_precision"] is None else f"{s['omission_precision']:.4f}"
            print(f"[b3 {label}] {n:34s} recall {s['recall']:.4f}  omission-prec {op}  n={s['n_pairs']}", flush=True)
        for n, c in block["contrasts"].items():
            for q, tag in (("recall", "rec "), ("omission_precision", "prec")):
                x = c[q]
                if x is None:
                    continue
                print(f"[b3 {label} vs {n:30s}] {tag} observed {x['observed']:+.4f} "
                      f"(boot mean {x['bootstrap_mean']:+.4f}, delta {x['bootstrap_mean_minus_observed']:+.1e}) "
                      f"pct CI [{x['ci_percentile'][0]:+.4f},{x['ci_percentile'][1]:+.4f}] "
                      f"basic CI [{x['ci_basic'][0]:+.4f},{x['ci_basic'][1]:+.4f}] "
                      f"one-sided-lo {x['one_sided_lower_95']:+.4f} "
                      f"{'EXCLUDES 0' if x['excludes_zero'] else 'straddles 0'}", flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    bad = {k: v for k, v in report["crosscheck"].items() if v != "ok"}
    print(f"\n[b3] written {out}; crosscheck failures: {sorted(bad) or 'none'}", flush=True)
    if bad:
        raise SystemExit("[b3] REFUSING to exit clean: a re-derived point estimate disagrees with the evaluator")


if __name__ == "__main__":
    main()
