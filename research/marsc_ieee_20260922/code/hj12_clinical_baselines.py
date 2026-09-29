#!/usr/bin/env python3
"""H-J12 (audit response, 2026-09-28) -- chance and position baselines for the clinical top-10 (review finding F15).

The paper compared the clinical recall@10 of 0.205 with a "one in 38" per-fact rate, which is not the chance level
of a ten-item emission. This writes two summary-blind emitters over the IDENTICAL clinical Gemma+ent inventory, in
the unit_scores format the frozen div1 rule and evaluator read, so they pass through the same emission rule, the
same alignment rule and the same consultation-clustered bootstrap as the verifier:

  random   uniform scores, three seed-locked draws per pair (seeds 1..3), evaluated as one three-seed system
  leadpos  1 - start / len(source): the lead-position salience prior (earlier transcript content first)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True, help="name=path of the units file")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    name, path = a.units.split("=", 1)
    rows = common.load_jsonl(Path(path))
    lead = {}
    rnd = {s: {} for s in (1, 2, 3)}
    for r in rows:
        n = max(1, len(r["source"]))
        lead[r["pair_id"]] = [1.0 - min(1.0, max(0.0, float(u.get("start", n // 2)) / n)) for u in r["units"]]
        h = int(hashlib.sha1(r["pair_id"].encode()).hexdigest()[:8], 16)
        for s in rnd:
            rnd[s][r["pair_id"]] = np.random.default_rng([h, s]).random(len(r["units"])).tolist()
    out = Path(a.out)
    common.write_scores(out / f"{name}_leadpos_seed0.jsonl", "ctl:leadpos", name, lead)
    for s, per in rnd.items():
        common.write_scores(out / f"{name}_random_seed{s}.jsonl", "ctl:random", name, per)
    print(f"[hj12-clin] {len(rows)} pairs -> {out}", flush=True)


if __name__ == "__main__":
    main()
