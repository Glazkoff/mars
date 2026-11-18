#!/usr/bin/env python3
"""MARS-C strengthen C2 / H-SURF -- the registered BAR: does the MARS-C advantage on the crossed-conditioning
endpoint survive residualising out the surface cues?

Registered in research/marsc_strengthen_20260918/PREREG.md (Wave C / C2).

The endpoint is the B2 / R03a estimand, unchanged: within a document, for the SAME fact f,

    crossed accuracy = P( z(f | a summary that OMITS f) > z(f | a summary that CONVEYS f) ),  ties = 1/2

Any summary-blind scorer is pinned at exactly 0.500.  The cells, the win indicator, the document-macro
statistic and the document-clustered bootstrap come from research/marsc_20260916/code/mc_summary_conditioning.py
(imported, never modified).  This file adds one thing: the same endpoint recomputed on the RESIDUALISED score.

Why residualising matters HERE and what it can and cannot remove.  Inside a same-fact triple the fact is held
fixed, so two of the three registered surface regressors -- unit character position in the source and unit token
length -- are IDENTICAL in the two cells of the triple and cannot move the win indicator at all.  The only
regressor that varies within a triple is the fact-summary lexical overlap.  So the crossed endpoint's
residualisation is exactly the test a reviewer wants: it asks whether the family's apparent summary
conditioning is nothing but lexical overlap between the fact and the candidate.  To make that explicit the job
also scores a SURFACE-ONLY arm (the fitted prediction used as the score), whose crossed accuracy is the share
of the endpoint the surface model can reach on its own.

Identification contract: the surface model is fitted on the VALIDATION cells and applied UNCHANGED to the
sealed TEST cells.  Per checkpoint seed, so no seed's scale leaks into another's residual.

BAR, declared before any number in this file was read:
  the primary family's margin over every comparator family, on the RESIDUALISED score, must have a
  document-clustered one-sided 95% lower bound above 0 on validation AND on TEST.  If a family's entire
  advantage is surface-predictable, that is a finding and it is reported.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


msc = _load(REPO / "research" / "marsc_20260916" / "code" / "mc_summary_conditioning.py")
c2s = _load(Path(__file__).resolve().parent / "c2_surface_control.py")

BARS = {
    "registered": "research/marsc_strengthen_20260918/PREREG.md -- Wave C / C2, hypothesis H-SURF",
    "declared_before_any_number_in_this_file_was_read": True,
    "estimand": "within-document, same-fact crossed accuracy P(z(f|omitting summary) > z(f|conveying summary)),"
                " ties 1/2; document-macro statistic, document-clustered bootstrap",
    "chance_level_for_any_summary_blind_scorer": 0.5,
    "bar": "PASSES if the primary family's margin over EVERY comparator family, computed on the RESIDUALISED "
           "score, has a document-clustered one-sided 95% lower bound above 0 on validation AND on TEST.",
    "fail": "If a family's entire advantage is surface-predictable, that is a finding and it is reported; the "
            "supervision claim then narrows to whatever survives.",
    "identification": "surface model fitted on the VALIDATION cells per checkpoint seed, applied UNCHANGED to "
                      "the sealed TEST cells; no moment of the evaluation split enters the transform",
    "within_triple_note": "rel_start and tok_len are constant within a same-fact triple and cannot move the win "
                          "indicator; the residualisation therefore removes fact-summary LEXICAL OVERLAP, which "
                          "is precisely the surface channel that could fake conditioning on this endpoint.",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def cell_features(docs: list[dict], cells: list[tuple]) -> np.ndarray:
    """One row per crossed cell, same three registered regressors as c2_surface_control, fact-level."""
    pos_len: dict[tuple, tuple[float, float, set]] = {}
    for di, _, fi in cells:
        if (di, fi) in pos_len:
            continue
        src = docs[di]["source"]
        fact = docs[di]["facts"][fi]
        s, _ = common.best_sentence_span(src, fact)
        toks = common.tokens(fact)
        pos_len[(di, fi)] = (min(1.0, max(0.0, s / max(1, len(src)))), float(len(toks)),
                             set(common.tokens(fact, content_only=True)) or set(toks))
    cand_toks: dict[tuple, set] = {}
    for di, si, _ in cells:
        if (di, si) in cand_toks:
            continue
        c = docs[di]["summaries"][si]["candidate"]
        cand_toks[(di, si)] = set(common.tokens(c, content_only=True)) or set(common.tokens(c))
    X = np.zeros((len(cells), 3), dtype=np.float64)
    for i, (di, si, fi) in enumerate(cells):
        rel, tl, fb = pos_len[(di, fi)]
        cb = cand_toks[(di, si)]
        un = fb | cb
        X[i] = (rel, tl, (len(fb & cb) / len(un)) if un else 0.0)
    return X


def checkpoints(pattern: str, expect: int) -> list[str]:
    ck = [d for d in sorted(glob.glob(pattern)) if Path(d, "config.json").exists()]
    assert ck, f"no checkpoint with a config.json matched {pattern!r}"
    if expect:
        assert len(ck) == expect, f"expected {expect} checkpoints, matched {len(ck)}: {ck}"
    return ck


def seed_of(d: str) -> str:
    m = re.search(r"seed(\d+)", d)
    return m.group(1) if m else Path(d).name


def paired_margin(a_means: dict, b_means: dict, seed: int, n_boot: int) -> dict:
    keys = sorted(set(a_means) & set(b_means))
    if not keys:
        return {"n_docs": 0}
    d = np.array([a_means[k] - b_means[k] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    draws = d[rng.integers(0, len(keys), size=(n_boot, len(keys)))].mean(axis=1)
    return {"n_docs": len(keys), "observed_margin_doc_macro": float(d.mean()),
            "ci95_two_sided": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "lower_bound_95_one_sided": float(np.percentile(draws, 5.0)),
            "p_one_sided_bootstrap": float(np.mean(draws <= 0.0)),
            "n_docs_a_better": int(np.sum(d > 0)), "n_docs_b_better": int(np.sum(d < 0)),
            "n_docs_tied": int(np.sum(d == 0))}


def verdict_vs_chance(stat: dict) -> str:
    lo, hi = stat["ci_doc_bootstrap"]
    if lo <= 0.5 <= hi:
        return "CANDIDATE-BLIND"
    return "CONDITIONS-ON-CANDIDATE" if lo > 0.5 else "ANTI-CONDITIONED"


def score_split(a, docs_file: str, split: str, fam_specs: list[str], dev: str, out: Path) -> dict:
    """-> {"cells", "triples", "X", "z": {family: {seed: {cell: float}}}, "meta"}."""
    docs = common.load_jsonl(Path(docs_file))
    if a.limit_docs:
        docs = docs[:a.limit_docs]
    cells, triples = msc.crossed_cells(docs)
    print(f"[c2x] {split}: {len(docs)} docs -> {len(cells)} crossed cells, {len(triples)} same-fact triples "
          f"over {len({t[0] for t in triples})} docs", flush=True)
    X = cell_features(docs, cells)
    z: dict[str, dict[str, dict]] = {}
    for spec in fam_specs:
        name, pat = spec.split("=", 1)
        ck = checkpoints(pat, a.expect_seeds)
        z[name] = {}
        for d in ck:
            s = seed_of(d)
            z[name][s] = msc.score_cells(d, docs, cells, a.batch, dev)
            print(f"[c2x] {split} {name} seed {s}: scored {len(z[name][s])} cells from {d}", flush=True)
            dump = out / f"z_{split}_{name}_seed{s}.jsonl"
            with open(dump, "w", encoding="utf-8") as fh:
                for i, c in enumerate(cells):
                    fh.write(json.dumps({"i": i, "di": c[0], "si": c[1], "fi": c[2],
                                         "z": float(z[name][s][c])}) + "\n")
    meta = {"docs_file": docs_file, "docs_sha256": sha256(Path(docs_file)), "n_docs": len(docs),
            "n_cells": len(cells), "n_triples": len(triples),
            "docs_with_triples": len({t[0] for t in triples}),
            "facts_with_triples": len({(t[0], t[1]) for t in triples}),
            "doc_keys": {int(di): common.doc_key(docs[di]["source"]) for di in sorted({t[0] for t in triples})}}
    return {"cells": cells, "triples": triples, "X": X, "z": z, "meta": meta}


def endpoint(zc: dict, triples, seed: int, n_boot: int) -> tuple[dict, dict]:
    w, bd = msc.discriminate(zc, triples)
    stat = msc.summarise(w, bd, np.random.default_rng(seed), n_boot)
    return stat, {d: float(np.mean(v)) for d, v in bd.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-validation", required=True)
    ap.add_argument("--docs-test", default="", help="omit to run validation only")
    ap.add_argument("--families", nargs="+", required=True, help="NAME=<glob of checkpoint dirs>")
    ap.add_argument("--primary", default="humanfact")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260917,
                    help="20260917 reproduces the published R03a intervals for the raw arms exactly")
    ap.add_argument("--expect-seeds", type=int, default=3)
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fam_names = [s.split("=", 1)[0] for s in a.families]
    assert a.primary in fam_names, f"--primary {a.primary!r} not among {fam_names}"

    splits = {"validation": score_split(a, a.docs_validation, "validation", a.families, dev, out)}
    if a.docs_test:
        splits["test"] = score_split(a, a.docs_test, "test", a.families, dev, out)

    # ---- surface model: FIT ON VALIDATION ONLY, per family per seed
    models: dict[str, dict[str, dict]] = {}
    Xv, cv = splits["validation"]["X"], splits["validation"]["cells"]
    surface = {}
    for fam in fam_names:
        models[fam] = {}
        surface[fam] = {}
        for s, zs in splits["validation"]["z"][fam].items():
            y = np.array([zs[c] for c in cv], dtype=float)
            ok = np.isfinite(y)
            models[fam][s] = c2s.fit_ols(Xv[ok], y[ok], c2s.FULL_COLS)
            blind = c2s.fit_ols(Xv[ok], y[ok], c2s.BLIND_COLS)
            jac = c2s.fit_ols(Xv[ok], y[ok], c2s.JAC_COLS)
            surface[fam][s] = {"full": models[fam][s], "blind": blind, "jac": jac}
            print(f"[c2x] surface fit {fam} seed {s}: R2 full {models[fam][s]['r2']:.4f} "
                  f"blind {blind['r2']:.4f} jac {jac['r2']:.4f} | std beta "
                  + " ".join(f"{n}={b:+.4f}" for n, b in zip(c2s.FEATURES, models[fam][s]["coef_standardised"])),
                  flush=True)

    rep = {"bars": BARS, "feature_spec": c2s.FEATURE_SPEC, "primary_family": a.primary,
           "families": fam_names, "bootstrap_seed": a.seed, "n_boot": a.n_boot,
           "surface_models_fitted_on": "validation", "surface_models": surface, "splits": {}}

    for split, S in splits.items():
        cells, triples, X = S["cells"], S["triples"], S["X"]
        entry = {**S["meta"], "arms": {}}
        doc_means: dict[str, dict[str, dict]] = {"raw": {}, "res": {}, "surface_only": {}}
        for fam in fam_names:
            seeds = sorted(S["z"][fam])
            acc_raw = defaultdict(list)
            acc_res = defaultdict(list)
            acc_srf = defaultdict(list)
            per_seed = {}
            for s in seeds:
                zs = S["z"][fam][s]
                y = np.array([zs[c] for c in cells], dtype=float)
                pred = c2s.predict(models[fam][s], X)
                res = y - pred
                for i, c in enumerate(cells):
                    acc_raw[c].append(float(y[i]))
                    acc_res[c].append(float(res[i]))
                    acc_srf[c].append(float(pred[i]))
                st_raw, _ = endpoint({c: float(y[i]) for i, c in enumerate(cells)}, triples, a.seed, a.n_boot)
                st_res, _ = endpoint({c: float(res[i]) for i, c in enumerate(cells)}, triples, a.seed, a.n_boot)
                per_seed[s] = {"raw": st_raw, "res": st_res}
                print(f"[c2x] {split} {fam} seed {s}: crossed accuracy raw {st_raw['discrimination_triple']:.4f} "
                      f"-> residualised {st_res['discrimination_triple']:.4f}", flush=True)
            arm = {"seeds": seeds, "per_seed": per_seed}
            for tag, acc in (("raw", acc_raw), ("res", acc_res), ("surface_only", acc_srf)):
                zc = {c: float(np.mean(v)) for c, v in acc.items()}
                stat, dm = endpoint(zc, triples, a.seed, a.n_boot)
                stat["verdict_vs_chance"] = verdict_vs_chance(stat)
                arm[tag] = stat
                doc_means[tag][fam] = dm
                ci = stat["ci_doc_bootstrap"]
                print(f"[c2x] {split} {fam:12s} {tag:12s} crossed {stat['discrimination_triple']:.4f} "
                      f"(doc macro {stat['discrimination_doc_macro']:.4f} [{ci[0]:.4f},{ci[1]:.4f}]) "
                      f"-> {stat['verdict_vs_chance']}", flush=True)
            arm["retained_share_of_advantage_over_chance"] = (
                float((arm["res"]["discrimination_doc_macro"] - 0.5) /
                      (arm["raw"]["discrimination_doc_macro"] - 0.5))
                if abs(arm["raw"]["discrimination_doc_macro"] - 0.5) > 1e-9 else None)
            entry["arms"][fam] = arm
        entry["margins"] = {}
        for tag in ("raw", "res", "surface_only"):
            entry["margins"][tag] = {}
            for fam in fam_names:
                if fam == a.primary:
                    continue
                mg = paired_margin(doc_means[tag][a.primary], doc_means[tag][fam], a.seed, a.n_boot)
                mg["clears_bar_on_this_split"] = bool(mg.get("lower_bound_95_one_sided", -1.0) > 0.0)
                entry["margins"][tag][fam] = mg
                print(f"[c2x] {split} margin[{tag}] {a.primary} - {fam:12s} "
                      f"{mg['observed_margin_doc_macro']:+.4f} one-sided 95% LB "
                      f"{mg['lower_bound_95_one_sided']:+.4f} (p={mg['p_one_sided_bootstrap']:.4f}) -> "
                      f"{'CLEARS' if mg['clears_bar_on_this_split'] else 'DOES NOT CLEAR'}", flush=True)
        failing = [f for f, m in entry["margins"]["res"].items() if not m["clears_bar_on_this_split"]]
        entry["split_verdict"] = {
            "comparators": [f for f in fam_names if f != a.primary],
            "not_cleared_on_residualised_score": failing,
            "result": "PASS-ON-THIS-SPLIT" if entry["margins"]["res"] and not failing
                      else ("FAIL-ON-THIS-SPLIT" if entry["margins"]["res"] else "NO-COMPARATOR")}
        rep["splits"][split] = entry
        print(f"[c2x] {split}: H-SURF {entry['split_verdict']['result']} "
              f"(not cleared: {failing or 'none'})", flush=True)

    got = sorted(rep["splits"])
    failing_any = sorted({f for s in rep["splits"].values()
                          for f in s["split_verdict"]["not_cleared_on_residualised_score"]})
    has_comp = any(s["margins"]["res"] for s in rep["splits"].values())
    rep["hsurf_verdict"] = {
        "splits_run": got,
        "both_splits_present": got == ["test", "validation"],
        "families_not_cleared": failing_any,
        "result": ("PASS" if has_comp and not failing_any and got == ["test", "validation"]
                   else ("FAIL" if has_comp and failing_any
                         else "INCOMPLETE -- the bar requires validation AND the sealed TEST split")),
        "candidate_blind_after_residualisation": sorted(
            {f"{sp}:{fam}" for sp, s in rep["splits"].items() for fam, arm in s["arms"].items()
             if arm["res"]["verdict_vs_chance"] == "CANDIDATE-BLIND"})}
    (out / "c2_crossed_residual.json").write_text(json.dumps(rep, indent=1))
    print(f"[c2x] H-SURF across {got}: {rep['hsurf_verdict']['result']} "
          f"(not cleared: {failing_any or 'none'}; candidate-blind after residualisation: "
          f"{rep['hsurf_verdict']['candidate_blind_after_residualisation'] or 'none'})", flush=True)
    print(f"[c2x] written {out / 'c2_crossed_residual.json'}", flush=True)


if __name__ == "__main__":
    main()
