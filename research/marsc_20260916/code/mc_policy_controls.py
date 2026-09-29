#!/usr/bin/env python3
"""MARS-C E2 policy controls (registered amendment): summary-blind heuristics standing in for the learned importance policy.

  leadpos  importance = 1 - start / len(source): the lead-position prior (a threshold keeps the first part of the document)
  random   seed-locked uniform importance; the caller sets its threshold to 1 - (learned policy's keep rate), so the same
           share of units survives as under the learned policy

Both are written in the unit_scores format, so `mc_importance.py combine` applies them exactly like the learned policy.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402


def keep_rate(glob_pat: str, tau: float) -> float:
    """Share of units the learned policy keeps at tau (importance averaged over its seed files)."""
    acc: dict[str, list[np.ndarray]] = {}
    for f in sorted(glob.glob(glob_pat)):
        _, sc = common.read_scores(Path(f))
        for p, v in sc.items():
            acc.setdefault(p, []).append(v)
    if not acc:
        return float("nan")
    allv = np.concatenate([np.nanmean(np.vstack(v), axis=0) for v in acc.values()])
    return float(np.mean(allv >= tau))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", required=True, help="name=path of a units file")
    ap.add_argument("--learned-glob", nargs="*", default=[], help="name=glob of the learned importance files (matched keep rate)")
    ap.add_argument("--tau-learned", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    learned = dict(x.split("=", 1) for x in a.learned_glob)
    man = {}
    for spec in a.units:
        name, path = spec.split("=", 1)
        lead: dict[str, list[float]] = {}; rnd: dict[str, list[float]] = {}
        for r in common.load_jsonl(Path(path)):
            n = max(1, len(r["source"]))
            lead[r["pair_id"]] = [1.0 - min(1.0, max(0.0, float(u.get("start", n // 2)) / n)) for u in r["units"]]
            rng = np.random.default_rng(int(hashlib.sha1(r["pair_id"].encode()).hexdigest()[:8], 16))
            rnd[r["pair_id"]] = rng.random(len(r["units"])).tolist()
        common.write_scores(out / f"{name}_leadpos.jsonl", "ctl:leadpos", name, lead)
        common.write_scores(out / f"{name}_random.jsonl", "ctl:random", name, rnd)
        kr = keep_rate(learned[name], a.tau_learned) if name in learned else float("nan")
        man[name] = {"pairs": len(lead), "learned_keep_rate": kr, "tau_random": round(1.0 - kr, 3) if np.isfinite(kr) else None}
        print(f"[ctl] {name}: {len(lead)} pairs; learned keep rate at tau {a.tau_learned:g} = {kr:.3f} -> random tau {man[name]['tau_random']}", flush=True)
    json.dump(man, open(out / "controls_manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
