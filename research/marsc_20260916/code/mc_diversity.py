#!/usr/bin/env python3
"""MARS-C E2 diversity step (registered amendment): at most one emitted unit per source sentence.

Motivation (controls job 4837): a random filter at the learned policy's keep rate beat the verifier's own ranking
(0.327 vs 0.292 recall@10), i.e. the top of the omission ranking is redundant — several units of one sentence. Rule, applied
identically to every emitter: within a pair, the best-scored kept unit of each source sentence keeps its score; other kept
units of that sentence are pushed by -5 (below every sentence leader, above units the policy removed at -10).
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


def sentence_index(source: str, start: int) -> int:
    for i, (a, b) in enumerate(common.sentence_spans(source)):
        if a <= start < b:
            return i
    return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True, help="name=path of the units file the scores refer to")
    ap.add_argument("--scores-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=1, help="kept units per source sentence before the -5 push (registered default 1)")
    a = ap.parse_args()
    name, path = a.units.split("=", 1)
    sent_of: dict[str, list[int]] = {}
    for r in common.load_jsonl(Path(path)):
        n = len(r["source"])
        sent_of[r["pair_id"]] = [sentence_index(r["source"], int(u.get("start", n // 2))) for u in r["units"]]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for f in sorted(glob.glob(a.scores_glob)):
        sysname, sc = common.read_scores(Path(f)); per = {}
        for p, v in sc.items():
            si = sent_of.get(p)
            if si is None or len(si) != len(v):
                continue
            v = np.array(v, dtype=float); new = v.copy(); count: dict[int, int] = {}
            order = np.argsort(-np.nan_to_num(v, nan=-np.inf))
            for i in order:
                if not np.isfinite(v[i]) or v[i] <= -9.0:          # removed by the policy: leave at -10
                    continue
                s = si[i]
                if count.get(s, 0) >= a.cap:
                    new[i] = v[i] - 5.0
                else:
                    count[s] = count.get(s, 0) + 1
            per[p] = new.tolist()
        common.write_scores(out / Path(f).name.replace(".jsonl", f"_div{a.cap}.jsonl"), f"{sysname}+div{a.cap}", name, per)
        print(f"[div] {Path(f).name}: {len(per)} pairs", flush=True)


if __name__ == "__main__":
    main()
