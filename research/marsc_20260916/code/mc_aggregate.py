#!/usr/bin/env python3
"""MARS-C T08 -- seed-aggregation variants for the emission rule (registered amendment, Tier 2).

The regime of record applies the >=1-unit-per-sentence rule per verifier seed and averages the three outputs
(`consensus`); the cheap alternative applies it once to the seed-averaged probabilities (`single-model`, mc_variants.py
seedavg). G7 found the single-model regime to carry the LARGER external margin at a third of the cost, so the
aggregation step itself is a free parameter worth measuring rather than assuming. Two further aggregations, both
producing one score file in the unit_scores format so the rule and the evaluator apply unchanged:

  rankavg   mean of the per-seed RANKS instead of the per-seed probabilities. Scale-free: a seed that is globally
            over-confident cannot dominate the mean. Emitted score = -(mean rank), so higher still means "more omitted".
  leadunion count of seeds in which a unit is the best-scored unit of its source sentence, tie-broken by the mean
            probability. This is the aggregation the rule actually consumes, applied before averaging rather than after.

Neither reads any label. Both are pure re-rankings of score files that already exist on disk.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402


def _load_seeds(globpat: str) -> dict[str, list[np.ndarray]]:
    acc: dict[str, list[np.ndarray]] = {}
    for f in sorted(glob.glob(globpat)):
        _, sc = common.read_scores(Path(f))
        for p, v in sc.items():
            acc.setdefault(p, []).append(np.asarray(v, dtype=float))
    return {p: v for p, v in acc.items() if len({len(x) for x in v}) == 1}


def _sentence_of(units_path: Path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for r in common.load_jsonl(units_path):
        if not r.get("units"):
            continue
        spans = common.sentence_spans(r["source"])
        idx = []
        for u in r["units"]:
            st = int(u.get("start", 0))
            idx.append(next((i for i, (a, b) in enumerate(spans) if a <= st < b), -1))
        out[r["pair_id"]] = idx
    return out


def rankavg(a) -> None:
    name, path = a.units.split("=", 1)
    rows = {r["pair_id"]: r for r in common.load_jsonl(Path(path)) if r.get("units")}
    seeds = _load_seeds(a.scores)
    per = {}
    for p, vs in seeds.items():
        if p not in rows or len(vs[0]) != len(rows[p]["units"]):
            continue
        ranks = []
        for v in vs:
            filled = np.nan_to_num(v, nan=-np.inf)
            order = np.argsort(-filled, kind="stable")
            r = np.empty(len(filled), dtype=float)
            r[order] = np.arange(len(filled), dtype=float)
            ranks.append(r)
        per[p] = (-np.mean(np.vstack(ranks), axis=0)).tolist()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    common.write_scores(out / f"{name}_{a.tag}_seedrank.jsonl", a.tag, name, per)
    print(f"[rankavg] {name}/{a.tag}: {len(per)} pairs from {len(next(iter(seeds.values()), []))} seeds", flush=True)


def leadunion(a) -> None:
    name, path = a.units.split("=", 1)
    rows = {r["pair_id"]: r for r in common.load_jsonl(Path(path)) if r.get("units")}
    sent_of = _sentence_of(Path(path))
    seeds = _load_seeds(a.scores)
    per = {}
    for p, vs in seeds.items():
        si = sent_of.get(p)
        if p not in rows or si is None or len(si) != len(vs[0]):
            continue
        lead = np.zeros(len(si), dtype=float)
        for v in vs:
            filled = np.nan_to_num(v, nan=-np.inf)
            best: dict[int, int] = {}
            for i in range(len(filled)):
                s = si[i]
                if s not in best or filled[i] > filled[best[s]]:
                    best[s] = i
            for i in best.values():
                lead[i] += 1.0
        mean_p = np.nan_to_num(np.mean(np.vstack(vs), axis=0), nan=0.0)
        # leaders sort above non-leaders; the mean probability only breaks ties inside a lead count
        per[p] = (lead + np.clip(mean_p, 0.0, 1.0) * 0.999).tolist()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    common.write_scores(out / f"{name}_{a.tag}_seedlead.jsonl", a.tag, name, per)
    print(f"[leadunion] {name}/{a.tag}: {len(per)} pairs", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("rankavg", "leadunion"):
        q = sub.add_parser(cmd)
        q.add_argument("--units", required=True, help="name=path")
        q.add_argument("--scores", required=True, help="glob of the per-seed score files")
        q.add_argument("--tag", required=True)
        q.add_argument("--out", required=True)
    a = ap.parse_args()
    {"rankavg": rankavg, "leadunion": leadunion}[a.cmd](a)


if __name__ == "__main__":
    main()
