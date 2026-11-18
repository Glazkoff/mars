#!/usr/bin/env python3
"""MARS-C G3 margin variants (registered in PLAN_MAIN_CONTRIBUTION.md): union inventory, verifier ensemble, MARS-2 dumps as units.

  union      concatenate two inventories per pair and the matching per-seed score files (seed files paired in sorted order)
  ensemble   mean of two verifiers' seed-averaged omission probabilities over one inventory -> one score file (seed 'ens')
  dumps2units  mars_v2 dump rows -> a units file (source/candidate taken from a units file of the same pairs) + one score file
All outputs use the unit_scores format, so the emission rule and the evaluator apply unchanged.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402


def _seed(f: str) -> str:
    m = re.search(r"seed(\d+)", f)
    return m.group(1) if m else "0"


def _avg(globpat: str) -> dict[str, np.ndarray]:
    acc: dict[str, list[np.ndarray]] = {}
    for f in sorted(glob.glob(globpat)):
        _, sc = common.read_scores(Path(f))
        for p, v in sc.items():
            acc.setdefault(p, []).append(v)
    return {p: np.nanmean(np.vstack(v), axis=0) for p, v in acc.items() if len({len(x) for x in v}) == 1}


def union(a) -> None:
    (na, pa), (nb, pb) = (x.split("=", 1) for x in a.units)
    ra = {r["pair_id"]: r for r in common.load_jsonl(Path(pa)) if r.get("units")}
    rb = {r["pair_id"]: r for r in common.load_jsonl(Path(pb)) if r.get("units")}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pids = [p for p in ra if p in rb]
    with open(out / f"units_{a.name}_{a.split}.jsonl", "w", encoding="utf-8") as fh:
        for p in pids:
            r = dict(ra[p]); r["units"] = list(ra[p]["units"]) + list(rb[p]["units"]); fh.write(json.dumps(r) + "\n")
    print(f"[union] {a.name}: {len(pids)} pairs, mean units/pair {np.mean([len(ra[p]['units']) + len(rb[p]['units']) for p in pids]):.1f}", flush=True)
    for ga, gb in zip(a.scores_a, a.scores_b):
        fa, fb = sorted(glob.glob(ga)), sorted(glob.glob(gb))
        assert len(fa) == len(fb) and fa, f"seed files differ: {ga} vs {gb}"
        for x, y in zip(fa, fb):
            _, sa = common.read_scores(Path(x)); _, sb = common.read_scores(Path(y)); per = {}
            for p in pids:
                if p in sa and p in sb and len(sa[p]) == len(ra[p]["units"]) and len(sb[p]) == len(rb[p]["units"]):
                    per[p] = np.concatenate([sa[p], sb[p]]).tolist()
            tag = re.sub(r"_seed\d+\.jsonl$", "", Path(x).name).split("_", 1)[-1]
            common.write_scores(out / f"{a.name}_{tag}_seed{_seed(x)}.jsonl", f"{tag}[union]", a.name, per)
            print(f"[union] scores {tag} seed {_seed(x)}: {len(per)} pairs", flush=True)


def ensemble(a) -> None:
    name, path = a.units.split("=", 1)
    rows = {r["pair_id"]: r for r in common.load_jsonl(Path(path)) if r.get("units")}
    A, B = _avg(a.scores_a), _avg(a.scores_b); per = {}
    for p, r in rows.items():
        if p in A and p in B and len(A[p]) == len(B[p]) == len(r["units"]):
            per[p] = (0.5 * (A[p] + B[p])).tolist()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    common.write_scores(out / f"{name}_{a.tag}_seedens.jsonl", a.tag, name, per)
    print(f"[ensemble] {name}/{a.tag}: {len(per)} pairs", flush=True)


def seedavg(a) -> None:
    """Average a verifier's per-seed omission probabilities into ONE score file (the single-model regime of the rule)."""
    name, path = a.units.split("=", 1)
    rows = {r["pair_id"]: r for r in common.load_jsonl(Path(path)) if r.get("units")}
    A = _avg(a.scores); per = {p: A[p].tolist() for p, r in rows.items() if p in A and len(A[p]) == len(r["units"])}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    common.write_scores(out / f"{name}_{a.tag}_seedavg.jsonl", a.tag, name, per)
    print(f"[seedavg] {name}/{a.tag}: {len(per)} pairs", flush=True)


def dumps2units(a) -> None:
    b3 = common._load("b3_endpoint", REPO / "scripts" / "plan2026")
    ctx = {r["pair_id"]: r for r in common.load_jsonl(Path(a.source_units))}
    acc: dict[str, list[list[float]]] = {}; units_of: dict[str, list[dict]] = {}
    for f in sorted(glob.glob(a.dumps)):
        for pid, per in b3.load_dump(Path(f)).items():
            if pid in ctx and per:
                units_of[pid] = [b3._u(x) for x in per]; acc.setdefault(pid, []).append([1 - float(x["support"]) for x in per])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); n = 0; per = {}
    with open(out / f"units_{a.name}_{a.split}.jsonl", "w", encoding="utf-8") as fh:
        for pid, v in acc.items():
            if len({len(x) for x in v}) != 1:
                continue
            r = ctx[pid]
            fh.write(json.dumps({"pair_id": pid, "resource": r["resource"], "system": r["system"], "split": a.split, "source": r["source"], "candidate": r["candidate"],
                                 "units": [{"kind": u.get("kind", "unit"), "text": u.get("text", ""), "start": u.get("start", 0), "end": u.get("end", 0)} for u in units_of[pid]]}) + "\n")
            per[pid] = np.mean(np.array(v), axis=0).tolist(); n += 1
    common.write_scores(out / f"{a.name}_mars2_seed0.jsonl", "mars2", a.name, per)
    print(f"[dumps2units] {a.name}: {n} pairs", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("union"); u.add_argument("--units", nargs=2, required=True, help="A=path B=path"); u.add_argument("--scores-a", nargs="+", required=True); u.add_argument("--scores-b", nargs="+", required=True)
    u.add_argument("--name", default="union"); u.add_argument("--split", default="validation"); u.add_argument("--out", required=True)
    e = sub.add_parser("ensemble"); e.add_argument("--units", required=True); e.add_argument("--scores-a", required=True); e.add_argument("--scores-b", required=True); e.add_argument("--tag", default="ens"); e.add_argument("--out", required=True)
    v = sub.add_parser("seedavg"); v.add_argument("--units", required=True); v.add_argument("--scores", required=True); v.add_argument("--tag", required=True); v.add_argument("--out", required=True)
    d = sub.add_parser("dumps2units"); d.add_argument("--dumps", required=True); d.add_argument("--source-units", required=True); d.add_argument("--name", default="mars2"); d.add_argument("--split", default="validation"); d.add_argument("--out", required=True)
    a = ap.parse_args(); {"union": union, "ensemble": ensemble, "seedavg": seedavg, "dumps2units": dumps2units}[a.cmd](a)


if __name__ == "__main__":
    main()
