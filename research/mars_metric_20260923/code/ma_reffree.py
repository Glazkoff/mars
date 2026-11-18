#!/usr/bin/env python3
"""M-A (iii) -- the reference-free MARS-R: salience-weighted support of the SOURCE inventory by the candidate.

    MARS-R_free(c) = sum_i w_i (1 - z_i) / sum_i w_i

where z_i is the deployed coverage verifier's omission probability of source unit i given the candidate (the
omission finder's own score files, three seeds averaged) and w_i the E2 importance tagger's probability that the
unit is one a reference would carry (three seeds averaged). Also written: the unweighted mean support and, when
the verifier's shuffled-candidate / no-candidate score files exist, the same reading under those controls.
Pair ids are matched to the meta-evaluation index (unisum:<uid>:<model>, rose:<pair id>). Secondary, no bar.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_scores(pattern: str) -> dict[str, np.ndarray]:
    """Seed files -> per pair the mean score vector (files must agree on unit order)."""
    acc: dict[str, list[np.ndarray]] = defaultdict(list)
    files = sorted(glob.glob(pattern))
    for f in files:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line); acc[r["pair_id"]].append(np.asarray(r["scores"], dtype=float))
    out = {}
    for k, v in acc.items():
        if len({len(x) for x in v}) != 1:
            raise SystemExit(f"[reffree] unit count differs across seeds for {k}")
        out[k] = np.mean(v, axis=0)
    print(f"[reffree] {pattern}: {len(files)} files, {len(out)} pairs", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="pairs_ma.jsonl")
    ap.add_argument("--set", action="append", required=True,
                    help="<prefix>=<omission glob>|<importance glob>[|<shuffled glob>|<empty glob>]; prefix unisum or rose")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pairs = [json.loads(l) for l in open(a.pairs, encoding="utf-8")]
    rows = []
    for spec in a.set:
        prefix, rest = spec.split("=", 1); parts = rest.split("|")
        z = load_scores(parts[0]); w = load_scores(parts[1])
        zs = load_scores(parts[2]) if len(parts) > 2 and parts[2] else {}
        ze = load_scores(parts[3]) if len(parts) > 3 and parts[3] else {}
        n_ok = 0
        for p in pairs:
            if not p["pair_id"].startswith(prefix + ":"):
                continue
            key = p["pair_id"][len(prefix) + 1:]
            if key not in z or key not in w:
                continue
            zz, ww = z[key], w[key]
            if len(zz) != len(ww):
                raise SystemExit(f"[reffree] {key}: {len(zz)} verifier units vs {len(ww)} importance units")
            sup = 1.0 - zz
            r = {"pair_id": p["pair_id"], "dataset": p["dataset"], "marsR_free": float(np.sum(ww * sup) / max(np.sum(ww), 1e-9)),
                 "marsR_free_unweighted": float(sup.mean()), "n_units": int(len(zz))}
            top = ww >= np.quantile(ww, 0.8) if len(ww) >= 5 else np.ones(len(ww), dtype=bool)
            r["marsR_free_top20"] = float(sup[top].mean())
            if key in zs:
                r["marsR_free_shufcand"] = float(np.sum(ww * (1.0 - zs[key])) / max(np.sum(ww), 1e-9))
            if key in ze:
                r["marsR_free_nocand"] = float(np.sum(ww * (1.0 - ze[key])) / max(np.sum(ww), 1e-9))
            rows.append(r); n_ok += 1
        print(f"[reffree] {prefix}: {n_ok} pairs scored", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[reffree] written {a.out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
