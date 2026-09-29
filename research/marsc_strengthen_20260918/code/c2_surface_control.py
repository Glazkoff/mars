#!/usr/bin/env python3
"""MARS-C strengthen C2 / H-SURF -- the surface-cue control on the unit-level emission endpoint.

Registered in research/marsc_strengthen_20260918/PREREG.md (Wave C / C2).

The objection this answers.  A skeptical reviewer will ask whether the verifier has learned that LONG units, or
units LATE in the source document, or units LEXICALLY DISTANT from the summary, are simply more likely to be
omitted -- rather than learning coverage.  The paper currently has no answer.

What this file computes.
  1. A per-arm surface model: OLS of the per-unit omission score on three registered regressors
       rel_start     unit character start position / len(source)          (candidate-BLIND)
       tok_len       unit token count, common.tokens                      (candidate-BLIND)
       cand_jaccard  content-token Jaccard(unit text, candidate summary)  (candidate-DEPENDENT)
     Reported per arm: R^2, standardised and raw-scale coefficients, and the same for two nested models
     (blind-only = rel_start+tok_len, jac-only = cand_jaccard).  That table alone is a publishable diagnostic.
  2. The RESIDUALISED score, score - surface_prediction, written in the unit_scores format so that it plugs
     into mc_diversity.py -> mc_e2_eval.py UNCHANGED.  Two variants:
       +res       residual of the FULL three-regressor model (this is the registered variant the bar is on)
       +resblind  residual of the candidate-BLIND model only (removes position/length priors, leaves whatever
                  conditioning on the actual summary the verifier has)

Identification contract, declared before execution and enforced by the two modes of this script:
  the surface model is FITTED ON VALIDATION ONLY and applied UNCHANGED to TEST and to UniSumEval.  `--mode fit`
  refuses a units file whose `split` field is not the declared fit split; `--mode apply` refuses a model whose
  seed keys do not cover every score file it is asked to transform.  Standardisation moments are part of the
  model and travel with it, so no statistic of the evaluation split enters the transform.

Score files in, score files out: {"pair_id", "system", "unit_set", "scores": [...]}, higher = more omitted.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402

FEATURES = ("rel_start", "tok_len", "cand_jaccard")
BLIND_COLS = (0, 1)          # rel_start, tok_len   -- computable without reading the candidate
JAC_COLS = (2,)              # cand_jaccard         -- the only regressor that reads the candidate
FULL_COLS = (0, 1, 2)

FEATURE_SPEC = {
    "rel_start": "int(unit['start']) / max(1, len(source)), clipped to [0,1]; a unit with no 'start' key is 0.0",
    "tok_len": "len(common.tokens(unit['text']))  -- all tokens, stopwords included",
    "cand_jaccard": "|A & B| / |A | B| with A = set(common.tokens(unit['text'], content_only=True)) or the "
                    "all-token set when empty, B = the same construction on the candidate summary",
    "estimator": "ordinary least squares with intercept on z-scored regressors; the z-scoring moments are "
                 "fitted on the FIT split and stored in the model",
    "registered": "research/marsc_strengthen_20260918/PREREG.md -- Wave C / C2, hypothesis H-SURF",
}

BARS = {
    "registered": "research/marsc_strengthen_20260918/PREREG.md -- Wave C / C2, hypothesis H-SURF",
    "declared_before_any_number_in_this_file_was_read": True,
    "bar": "The MARS-C advantage on the B2 crossed-conditioning endpoint must survive residualisation with its "
           "document-clustered interval still excluding zero.  That bar is adjudicated by "
           "c2_crossed_residual.py; THIS file supplies the surface model, the diagnostic R^2 table and the "
           "residualised unit-level emission arms.",
    "reporting_rule": "If a family's entire advantage is surface-predictable, that is a finding and it is "
                      "reported.  Every residualised arm is reported beside its raw arm and beside the B0 "
                      "shuffled-summary / no-summary controls; no arm is reported on recall@k alone.",
    "identification": "Fitted on VALIDATION only; applied UNCHANGED to TEST and UniSumEval.",
}


# ----------------------------------------------------------------------------- features
def unit_features(source: str, candidate: str, units: list[dict]) -> np.ndarray:
    n = max(1, len(source))
    cb = set(common.tokens(candidate, content_only=True)) or set(common.tokens(candidate))
    X = np.zeros((len(units), len(FEATURES)), dtype=np.float64)
    for i, u in enumerate(units):
        st = u.get("start")
        X[i, 0] = min(1.0, max(0.0, (0 if st is None else int(st)) / n))
        txt = u.get("text") or ""
        toks = common.tokens(txt)
        X[i, 1] = float(len(toks))
        ub = set(common.tokens(txt, content_only=True)) or set(toks)
        un = ub | cb
        X[i, 2] = (len(ub & cb) / len(un)) if un else 0.0
    return X


def load_features(units_file: Path) -> tuple[dict[str, np.ndarray], dict[str, int], str]:
    """-> ({pair_id: X}, {pair_id: n_units}, declared split of the units file)."""
    feats, sizes, split = {}, {}, ""
    for r in common.load_jsonl(units_file):
        u = r.get("units")
        if not u:
            continue
        split = split or str(r.get("split", ""))
        feats[r["pair_id"]] = unit_features(r["source"], r["candidate"], u)
        sizes[r["pair_id"]] = len(u)
    return feats, sizes, split


# ----------------------------------------------------------------------------- OLS
def _design(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, cols) -> np.ndarray:
    Z = (X - mu) / sd
    return np.hstack([np.ones((len(Z), 1)), Z[:, list(cols)]])


def fit_ols(X: np.ndarray, y: np.ndarray, cols) -> dict:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    A = _design(X, mu, sd, cols)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"cols": [int(c) for c in cols], "feature_names": [FEATURES[c] for c in cols],
            "mean": [float(v) for v in mu], "sd": [float(v) for v in sd],
            "intercept": float(beta[0]), "coef_standardised": [float(b) for b in beta[1:]],
            "coef_raw_scale": [float(b / sd[c]) for b, c in zip(beta[1:], cols)],
            "r2": float(1.0 - sse / sst) if sst > 0 else 0.0,
            "n": int(len(y)), "y_mean": float(y.mean()), "y_sd": float(y.std())}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    mu = np.asarray(model["mean"], dtype=float)
    sd = np.asarray(model["sd"], dtype=float)
    A = _design(X, mu, sd, model["cols"])
    beta = np.asarray([model["intercept"]] + list(model["coef_standardised"]), dtype=float)
    return A @ beta


# ----------------------------------------------------------------------------- io
def seed_key(path: Path) -> str:
    m = re.search(r"seed([A-Za-z0-9]+)", path.name)
    return m.group(1) if m else path.stem


def manifest(spec: str) -> list[Path]:
    """Explicit comma manifest ONLY -- a glob is refused (standing rule 1 of the campaign)."""
    assert "*" not in spec and "?" not in spec, f"c2 refuses a glob score source: {spec!r}"
    files = [Path(x.strip()) for x in spec.split(",") if x.strip()]
    assert files, f"empty score manifest {spec!r}"
    missing = [str(f) for f in files if not f.exists()]
    assert not missing, f"score manifest lists files that do not exist: {missing}"
    keys = [seed_key(f) for f in files]
    assert len(set(keys)) == len(keys), f"score manifest repeats a seed key {keys}: {spec!r}"
    return files


def stack(feats: dict[str, np.ndarray], sizes: dict[str, int], path: Path):
    """-> (X, y, n_pairs_used, n_pairs_skipped) for one score file, aligned unit-by-unit to the inventory."""
    _, sc = common.read_scores(path)
    xs, ys, used, skipped = [], [], 0, 0
    for pid, v in sc.items():
        if pid not in feats or len(v) != sizes[pid]:
            skipped += 1
            continue
        xs.append(feats[pid])
        ys.append(np.asarray(v, dtype=float))
        used += 1
    assert xs, f"{path}: no pair aligned to the inventory"
    return np.vstack(xs), np.concatenate(ys), used, skipped


# ----------------------------------------------------------------------------- modes
def do_fit(a) -> None:
    name, upath = a.units.split("=", 1)
    feats, sizes, split = load_features(Path(upath))
    assert not a.require_split or split == a.require_split, \
        f"--mode fit refuses {upath}: declared split {split!r} != required {a.require_split!r}"
    files = manifest(a.scores)
    rep = {"arm": a.arm, "unit_set": name, "fit_units_file": str(upath), "fit_split": split,
           "feature_spec": FEATURE_SPEC, "bars": BARS, "per_key": {}, "diagnostic": {}}
    for f in files:
        X, y, used, skipped = stack(feats, sizes, f)
        ok = np.isfinite(y)
        Xf, yf = X[ok], y[ok]
        full = fit_ols(Xf, yf, FULL_COLS)
        blind = fit_ols(Xf, yf, BLIND_COLS)
        jac = fit_ols(Xf, yf, JAC_COLS)
        k = seed_key(f)
        rep["per_key"][k] = {"source_file": str(f), "n_pairs": used, "n_pairs_skipped": skipped,
                            "n_units": int(len(y)), "n_units_finite": int(ok.sum()),
                            "full": full, "blind": blind, "jac": jac}
        print(f"[c2:fit] {a.arm} seed {k}: n={full['n']} R2 full {full['r2']:.4f} "
              f"blind {blind['r2']:.4f} jac {jac['r2']:.4f} | std beta "
              + " ".join(f"{nm}={b:+.4f}" for nm, b in zip(full["feature_names"], full["coef_standardised"])),
              flush=True)
    keys = sorted(rep["per_key"])
    for mk in ("full", "blind", "jac"):
        rep["diagnostic"][mk] = {
            "r2_mean": float(np.mean([rep["per_key"][k][mk]["r2"] for k in keys])),
            "r2_per_seed": {k: rep["per_key"][k][mk]["r2"] for k in keys},
            "coef_standardised_mean": [float(v) for v in np.mean(
                [rep["per_key"][k][mk]["coef_standardised"] for k in keys], axis=0)],
            "feature_names": rep["per_key"][keys[0]][mk]["feature_names"]}
    Path(a.model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.model_out).write_text(json.dumps(rep, indent=1))
    print(f"[c2:fit] {a.arm}: surface R2 (seed mean) full {rep['diagnostic']['full']['r2_mean']:.4f} "
          f"blind {rep['diagnostic']['blind']['r2_mean']:.4f} jac {rep['diagnostic']['jac']['r2_mean']:.4f} "
          f"-> {a.model_out}", flush=True)


def do_apply(a) -> None:
    name, upath = a.units.split("=", 1)
    feats, sizes, split = load_features(Path(upath))
    model = json.loads(Path(a.model).read_text())
    assert model["arm"] == a.arm or a.allow_arm_mismatch, \
        f"model was fitted for arm {model['arm']!r}, asked to apply to {a.arm!r} (pass --allow-arm-mismatch " \
        f"to residualise a B0 control with its own family's validation model, which is the declared rule)"
    files = manifest(a.scores)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    written = {"arm": a.arm, "variant": a.variant, "model": str(a.model), "model_arm": model["arm"],
               "apply_units_file": str(upath), "apply_split": split, "unit_set": name,
               "model_fit_split": model["fit_split"], "files": {}}
    mk = {"res": "full", "resblind": "blind"}[a.variant]
    for f in files:
        k = seed_key(f)
        assert k in model["per_key"], f"model {a.model} has no fit for seed key {k!r} (has {sorted(model['per_key'])})"
        m = model["per_key"][k][mk]
        sysname, sc = common.read_scores(f)
        per, used, skipped = {}, 0, 0
        for pid, v in sc.items():
            if pid not in feats or len(v) != sizes[pid]:
                skipped += 1
                continue
            r = np.asarray(v, dtype=float) - predict(m, feats[pid])
            per[pid] = [None if not np.isfinite(x) else float(x) for x in r]
            used += 1
        dst = out / f.name
        common.write_scores(dst, f"{sysname}+{a.variant}", name, per)
        written["files"][str(dst)] = {"from": str(f), "seed_key": k, "n_pairs": used,
                                      "n_pairs_skipped": skipped, "r2_on_fit_split": m["r2"]}
        print(f"[c2:apply] {a.arm} {a.variant} seed {k}: {used} pairs (skipped {skipped}) -> {dst}", flush=True)
    if a.manifest_out:
        Path(a.manifest_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.manifest_out).write_text(json.dumps(written, indent=1))


def do_table(a) -> None:
    """Collect every surface_<arm>.json into the one diagnostic table the paper prints."""
    rows = {}
    for p in sorted(a.models):
        m = json.loads(Path(p).read_text())
        rows[m["arm"]] = {"unit_set": m["unit_set"], "fit_split": m["fit_split"],
                          "fit_units_file": m["fit_units_file"],
                          "n_units": int(np.mean([v["n_units_finite"] for v in m["per_key"].values()])),
                          "seeds": sorted(m["per_key"]),
                          **{f"r2_{k}": m["diagnostic"][k]["r2_mean"] for k in ("full", "blind", "jac")},
                          "coef_standardised_full": dict(zip(m["diagnostic"]["full"]["feature_names"],
                                                             m["diagnostic"]["full"]["coef_standardised_mean"])),
                          "r2_full_per_seed": m["diagnostic"]["full"]["r2_per_seed"]}
    rep = {"bars": BARS, "feature_spec": FEATURE_SPEC, "arms": rows,
           "reading": "r2_full is the share of the per-unit omission score this arm's three surface regressors "
                      "explain on the FIT split. r2_blind uses only the two regressors computable without "
                      "reading the candidate summary; the gap r2_full - r2_blind is the share attributable to "
                      "unit-summary lexical overlap."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"[c2:table] {len(rows)} arms -> {a.out}", flush=True)
    for arm, r in sorted(rows.items(), key=lambda kv: -kv[1]["r2_full"]):
        print(f"[c2:table] {arm:32s} R2 full {r['r2_full']:.4f} blind {r['r2_blind']:.4f} jac {r['r2_jac']:.4f} | "
              + " ".join(f"{k}={v:+.4f}" for k, v in r["coef_standardised_full"].items()), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["fit", "apply", "table"])
    ap.add_argument("--arm", default="")
    ap.add_argument("--units", default="", help="unit_set_name=<inventory jsonl>")
    ap.add_argument("--scores", default="", help="EXPLICIT comma manifest of per-seed unit_scores files (no globs)")
    ap.add_argument("--model-out", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--manifest-out", default="")
    ap.add_argument("--variant", default="res", choices=["res", "resblind"])
    ap.add_argument("--require-split", default="", help="--mode fit refuses a units file declaring another split")
    ap.add_argument("--allow-arm-mismatch", action="store_true")
    ap.add_argument("--models", nargs="*", default=[], help="--mode table: the surface_<arm>.json files")
    a = ap.parse_args()
    if a.mode == "fit":
        assert a.arm and a.units and a.scores and a.model_out, "fit needs --arm --units --scores --model-out"
        do_fit(a)
    elif a.mode == "apply":
        assert a.arm and a.units and a.scores and a.model and a.out, "apply needs --arm --units --scores --model --out"
        do_apply(a)
    else:
        assert a.models and a.out, "table needs --models and --out"
        do_table(a)


if __name__ == "__main__":
    main()
