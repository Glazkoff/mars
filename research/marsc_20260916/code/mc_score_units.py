#!/usr/bin/env python3
"""Score units files with saved MARS-C cross-encoder verifiers (hypothesis = the fact text alone, as trained).

--candidate-mode selects what plays the premise, which is how the R03 summary-conditioning controls are run:
  real      the pair's own summary (the deployed system)
  shuffled  the summary of a DIFFERENT pair, drawn by a seeded derangement inside the same (resource, system)
            stratum so length and style are held fixed -- a system that keeps its recall here is ranking facts
            by a summary-blind "generally omitted" prior rather than by what THIS summary left out
  empty     no summary at all (the fact-only floor)
"""
from __future__ import annotations

import argparse
import glob
import random
import re
from pathlib import Path

import numpy as np
import torch

import sys
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402


def premises(rows: list[dict], a) -> list[str]:
    """One premise per row under the registered candidate mode; `shuffled` is a seeded derangement within stratum."""
    real = [" ".join(r["candidate"].split()[:380]) for r in rows]
    if a.candidate_mode == "real":
        return real
    if a.candidate_mode == "empty":
        return [a.empty_text] * len(rows)
    strata: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        strata.setdefault((r.get("resource", ""), r.get("system", "")), []).append(i)
    out = list(real); rng = random.Random(a.shuffle_seed); leftover = []
    for key, idxs in sorted(strata.items()):
        if len(idxs) < 2:
            leftover += idxs                     # nothing to swap with inside the stratum; handled pool-wide below
            continue
        shift = 1 + rng.randrange(len(idxs) - 1)  # a rotation by 1..n-1 is a derangement of the stratum
        for t, dst in enumerate(idxs):
            out[dst] = real[idxs[(t + shift) % len(idxs)]]
    if len(leftover) >= 2:
        rng.shuffle(leftover)
        for t, dst in enumerate(leftover):
            out[dst] = real[leftover[(t + 1) % len(leftover)]]
    n_same = sum(1 for i in range(len(rows)) if out[i] == real[i])
    print(f"[score] candidate-mode=shuffled: {len(rows)} rows, {len(strata)} strata, "
          f"{n_same} rows whose replacement summary is textually identical to their own", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", required=True, help="name=path")
    ap.add_argument("--models", required=True, help="glob of saved model dirs")
    ap.add_argument("--tag", default="humanfact")
    ap.add_argument("--scores-out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--candidate-mode", choices=["real", "shuffled", "empty"], default="real")
    ap.add_argument("--shuffle-seed", type=int, default=20260917)
    ap.add_argument("--empty-text", default="", help="premise used by --candidate-mode empty")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = True
    for d in sorted(glob.glob(a.models)):
        seed = re.search(r"seed(\d+)", d).group(1) if re.search(r"seed(\d+)", d) else "0"
        tok = AutoTokenizer.from_pretrained(d); model = AutoModelForSequenceClassification.from_pretrained(d).to(dev).eval()
        for spec in a.units:
            name, path = spec.split("=", 1); rows = [r for r in common.load_jsonl(Path(path)) if r.get("units")]
            prem = premises(rows, a)
            flat = [(i, j, u["text"]) for i, r in enumerate(rows) for j, u in enumerate(r["units"])]; vals = []
            with torch.inference_mode():
                for s in range(0, len(flat), a.batch):
                    ch = flat[s:s + a.batch]
                    enc = tok([prem[i] for i, _, _ in ch], [f for _, _, f in ch], truncation="longest_first", max_length=512, padding=True, return_tensors="pt").to(dev)
                    vals.extend(torch.sigmoid(model(**enc).logits.squeeze(-1).float()).tolist())
            per = {r["pair_id"]: [np.nan] * len(r["units"]) for r in rows}
            for (i, j, _), v in zip(flat, vals):
                per[rows[i]["pair_id"]][j] = v
            out = Path(a.scores_out) / f"{name}_{a.tag}_seed{seed}.jsonl"; common.write_scores(out, f"{a.tag}", name, per)
            print(f"[score] {name} seed {seed}: {len(flat)} units -> {out.name}", flush=True)
        del model
    print("[score] done", flush=True)


if __name__ == "__main__":
    main()
