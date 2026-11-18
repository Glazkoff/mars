#!/usr/bin/env python3
"""MARS-C strengthening C3 -- rank fusion of two frozen verifier score files (PREREG Wave C / C3).

THE FUSION RULE IS DECLARED HERE, IN CODE, BEFORE ANY FUSED NUMBER IS READ.

  For a pair p with n units and a constituent verifier f with per-unit omission scores s_f[p][.]:
      rank_f[p][j]  = descending rank of s_f[p][j] among the pair's FINITE scores, ties averaged
                      (rank 1 = the unit f calls most likely omitted)
      nrank_f[p][j] = (rank_f[p][j] - 1) / (n_finite - 1)          in [0, 1], 0 = most omitted
      fused[p][j]   = 1 - mean_f( nrank_f[p][j] )                  in [0, 1], HIGHER = more omitted
  A unit whose score is non-finite in ANY constituent is written as null (propagated), so the fused
  emitter can never be credited for a unit one of its constituents could not score.

Why per-pair normalised ranks and not a score average: the constituents are on incomparable scales
(a cross-encoder sigmoid, a SummaC entailment margin, a cosine), the emission rule downstream is
rank-only, and a per-pair normalisation makes the fusion invariant to any strictly monotone
recalibration of either constituent -- which is exactly the A1 calibration exposure this campaign is
closing elsewhere. Ties are averaged because the off-the-shelf sentence-level verifiers assign one
score to every unit of a source sentence and a first-index tie-break would silently order them.

Seed pairing: constituents are given as explicit comma manifests. If both carry the same number of
files they are paired in sorted order (seed i with seed i); a single-file constituent is broadcast
across the other's seeds. Any other shape is refused.

Output: `<out>/<name>_seed<S>.jsonl` in the `common.write_scores` unit_scores format, so it plugs into
mc_diversity.py -> mc_e2_eval.py unchanged, plus `<out>/<name>_fusion_rule.json` recording the rule
text and the sha256 of every input BEFORE the fused scores are evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402

RULE_ID = "mean_per_pair_normalised_rank_v1"
RULE_TEXT = (
    "fused[p][j] = 1 - mean over constituents f of ((rank_desc_f(p, j) - 1) / (n_finite(p) - 1)), "
    "ranks over the pair's finite scores with ties averaged, higher = more likely omitted; a unit "
    "non-finite in any constituent is written null."
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def avg_rank_desc(s: np.ndarray) -> np.ndarray:
    """Descending ranks (1 = largest) with ties averaged; non-finite entries get NaN."""
    out = np.full(s.shape, np.nan, dtype=float)
    ok = np.isfinite(s)
    v = s[ok]
    if v.size == 0:
        return out
    order = np.argsort(-v, kind="mergesort")
    r = np.empty(v.size, dtype=float)
    r[order] = np.arange(1, v.size + 1, dtype=float)
    sv = v[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    out[ok] = r
    return out


def nrank(s: np.ndarray) -> np.ndarray:
    r = avg_rank_desc(s)
    n = int(np.isfinite(s).sum())
    if n <= 1:
        return np.where(np.isfinite(r), 0.0, np.nan)
    return (r - 1.0) / (n - 1.0)


def manifest(spec: str) -> tuple[str, list[Path]]:
    name, rest = spec.split("=", 1)
    files = [Path(x.strip()) for x in rest.split(",") if x.strip()]
    assert files, f"constituent {name!r}: empty manifest"
    assert len(set(files)) == len(files), f"constituent {name!r}: manifest repeats a path"
    missing = [str(f) for f in files if not f.exists()]
    assert not missing, f"constituent {name!r}: missing {missing}"
    return name, sorted(files)


def seed_of(p: Path) -> str:
    m = re.search(r"seed([A-Za-z0-9]+)", p.name)
    return m.group(1) if m else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--constituent", action="append", required=True,
                    help="NAME=comma manifest of unit_scores files (repeat; >=2 required)")
    ap.add_argument("--name", required=True, help="fused system name, e.g. fuse.mc_summac.distilled")
    ap.add_argument("--unit-set", required=True, help="unit_set tag written into every row (gemma+ent | distilled)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="20260906 20260907 20260908",
                    help="seed tokens used to name the outputs when a constituent carries no seed token")
    a = ap.parse_args()
    cons = [manifest(c) for c in a.constituent]
    assert len(cons) >= 2, "fusion needs at least two constituents"
    ns = sorted({len(f) for _, f in cons} - {1})
    assert len(ns) <= 1, f"constituents carry incompatible seed counts: {[(n, len(f)) for n, f in cons]}"
    n_out = ns[0] if ns else 1
    # the seed labels of the output come from the first multi-file constituent, else from --seeds
    labels: list[str] = []
    for _, files in cons:
        if len(files) == n_out and n_out > 1:
            labels = [seed_of(f) for f in files]
            break
    if not labels:
        labels = a.seeds.split()[:n_out] or [str(i) for i in range(n_out)]
    assert len(labels) == n_out and all(labels), f"cannot name {n_out} output seeds, got {labels}"

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rule = {
        "declared_before_numbers": True,
        "rule_id": RULE_ID,
        "rule": RULE_TEXT,
        "fused_system": a.name,
        "unit_set": a.unit_set,
        "n_output_seeds": n_out,
        "output_seed_labels": labels,
        "constituents": [
            {"name": n, "n_files": len(f), "broadcast": len(f) == 1 and n_out > 1,
             "files": [str(x) for x in f], "sha256": {x.name: _sha256(x) for x in f},
             "seed_tokens": [seed_of(x) for x in f]}
            for n, f in cons
        ],
    }
    (out / f"{a.name}_fusion_rule.json").write_text(json.dumps(rule, indent=1))
    print(f"[fuse] rule declared -> {out / (a.name + '_fusion_rule.json')}", flush=True)

    loaded = [[common.read_scores(f)[1] for f in files] for _, files in cons]
    stats = {"n_pairs_written": [], "n_units_null_propagated": [], "n_pairs_dropped_shape": []}
    for i, lab in enumerate(labels):
        per_con = [lc[i if len(lc) > 1 else 0] for lc in loaded]
        pids = set(per_con[0])
        for d in per_con[1:]:
            pids &= set(d)
        per: dict[str, list[float]] = {}
        n_null = 0
        n_drop = 0
        for pid in sorted(pids):
            vs = [d[pid] for d in per_con]
            if len({len(v) for v in vs}) != 1:
                n_drop += 1
                continue
            nr = np.vstack([nrank(np.asarray(v, dtype=float)) for v in vs])
            fused = 1.0 - np.mean(nr, axis=0)          # NaN propagates through mean
            n_null += int(np.sum(~np.isfinite(fused)))
            per[pid] = fused.tolist()
        path = out / f"{a.name}_seed{lab}.jsonl"
        common.write_scores(path, a.name, a.unit_set, per)
        stats["n_pairs_written"].append(len(per))
        stats["n_units_null_propagated"].append(n_null)
        stats["n_pairs_dropped_shape"].append(n_drop)
        print(f"[fuse] {path.name}: {len(per)} pairs, {n_null} null units, {n_drop} pairs dropped on unit-count mismatch",
              flush=True)
    rule["output_stats"] = stats
    rule["outputs"] = [f"{a.name}_seed{l}.jsonl" for l in labels]
    (out / f"{a.name}_fusion_rule.json").write_text(json.dumps(rule, indent=1))
    assert min(stats["n_pairs_written"]) > 0, "fusion produced no pairs"
    print(f"[fuse] done {a.name}", flush=True)


if __name__ == "__main__":
    main()
