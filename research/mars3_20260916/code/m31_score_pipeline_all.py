#!/usr/bin/env python3
"""Score any units file (pairs with units) with the saved in-domain cross-encoder checkpoints -> unit_scores.
The target-free pipeline baseline for inventory-free comparisons: units come from an inventory (spaCy, LLM
propositions, distilled) and are verified against the candidate. System name xenc[<name>].

--candidate-mode mirrors mc_score_units.py so the judge-label verifier can be run through the same R03
summary-conditioning controls (real / shuffled / empty premise)."""
from __future__ import annotations

import argparse
import glob
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

import m3common as m3
from m3common import gates


def clip(t: str, n: int) -> str:
    return " ".join(t.split()[:n])


def hyp(source: str, u: dict) -> str:
    ctx = gates.render_claim(source, u) if u.get("start") is not None else u["text"]
    return clip(f"[{u['text']}] {ctx}", 96)


def premises(rows, a):
    """One premise per row under the registered candidate mode (see mc_score_units.premises)."""
    real = [clip(r["candidate"], 380) for r in rows]
    if a.candidate_mode == "real":
        return real
    if a.candidate_mode == "empty":
        return [a.empty_text] * len(rows)
    strata = {}
    for i, r in enumerate(rows):
        strata.setdefault((r.get("resource", ""), r.get("system", "")), []).append(i)
    out = list(real); rng = random.Random(a.shuffle_seed); leftover = []
    for _, idxs in sorted(strata.items()):
        if len(idxs) < 2:
            leftover += idxs
            continue
        shift = 1 + rng.randrange(len(idxs) - 1)
        for t, dst in enumerate(idxs):
            out[dst] = real[idxs[(t + shift) % len(idxs)]]
    if len(leftover) >= 2:
        rng.shuffle(leftover)
        for t, dst in enumerate(leftover):
            out[dst] = real[leftover[(t + 1) % len(leftover)]]
    print(f"[pipe] candidate-mode=shuffled: {len(rows)} rows, {len(strata)} strata, "
          f"{sum(1 for i in range(len(rows)) if out[i] == real[i])} textually unchanged", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", required=True, help="name=path units files")
    ap.add_argument("--ckpts", default=str(gates.OUT_ROOT / "b21" / "xenc_seed*"))
    ap.add_argument("--scores-out", default=str(m3.M3_OUT / "scores"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--candidate-mode", choices=["real", "shuffled", "empty"], default="real")
    ap.add_argument("--shuffle-seed", type=int, default=20260917)
    ap.add_argument("--empty-text", default="")
    ap.add_argument("--tag", default="xenc", help="file/system tag; use e.g. xencshuf for a control arm")
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = True
    ck = [d for d in sorted(glob.glob(a.ckpts)) if Path(d, "config.json").exists()]
    print(f"[pipe] checkpoints: {ck}", flush=True)
    sets = {}
    for spec in a.units:
        name, path = spec.split("=", 1)
        rows = [r for r in gates.load_jsonl(Path(path)) if r.get("units")]
        sets[name] = rows[:a.limit] if a.limit else rows
    for d in ck:
        seed = re.search(r"seed(\d+)", d).group(1) if re.search(r"seed(\d+)", d) else "0"
        tok = AutoTokenizer.from_pretrained(d); model = AutoModelForSequenceClassification.from_pretrained(d).to(dev).eval()
        for name, rows in sets.items():
            t0 = time.time(); flat = [(i, j, hyp(r["source"], u)) for i, r in enumerate(rows) for j, u in enumerate(r["units"])]
            prem = premises(rows, a); vals = []
            with torch.inference_mode():
                for s in range(0, len(flat), a.batch):
                    ch = flat[s:s + a.batch]
                    enc = tok([prem[i] for i, _, _ in ch], [h for _, _, h in ch], truncation="longest_first", max_length=512, padding=True, return_tensors="pt").to(dev)
                    vals.extend(torch.softmax(model(**enc).logits.float(), -1)[:, 0].tolist())
            per = {r["pair_id"]: [np.nan] * len(r["units"]) for r in rows}
            for (i, j, _), v in zip(flat, vals):
                per[rows[i]["pair_id"]][j] = v
            out = Path(a.scores_out) / f"{name}_{a.tag}_seed{seed}.jsonl"
            gates.write_scores(out, f"{a.tag}[{name}]", name, per)
            print(f"[pipe] {name} seed {seed}: {len(flat)} units, {(time.time() - t0) / max(1, len(rows)):.3f} s/pair -> {out.name}", flush=True)
        del model; torch.cuda.empty_cache() if dev.startswith("cuda") else None
    print("[pipe] done", flush=True)


if __name__ == "__main__":
    main()
