#!/usr/bin/env python3
"""B18 -- score supplied units (human-labelled ACUs) with a MARS-2 checkpoint.

A supplied unit has no span in the source, so the masked window is built from the source
sentence with the highest token F1 to the unit (the region the unit came from):
  masked   that sentence replaced by <extra_id_0>, target "<extra_id_0> {unit} <extra_id_1>"   (PRIMARY)
  nomask   same window, nothing masked                                                          (diagnostic)
Outputs per pair: q (support head) and loglik (recovery readout) per unit per variant.
Resumable by pair_id. One checkpoint per job; seeds are averaged at analysis time.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import common

sys.path.insert(0, str(common.REPO))
from mars_v2.model import load_model, local_window  # noqa: E402
from mars_v2.units import Unit  # noqa: E402


def windows(source: str, claim: str, radius: int = 600):
    s, e = common.best_sentence_span(source, claim)
    u = Unit("acu", claim, s, e)
    masked, target = local_window(source, u, radius)
    lo, hi = max(0, s - radius), min(len(source), e + radius)
    return masked, source[lo:hi], target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--units", required=True, help="acu_units_<split>.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="masked,nomask")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = common.load_jsonl(Path(a.units))
    if a.limit:
        rows = rows[:a.limit]
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["pair_id"] for l in open(outp)} if outp.exists() else set()
    todo = [r for r in rows if r["pair_id"] not in done]
    print(f"[b18-mars2] ckpt={a.ckpt} pairs {len(todo)}/{len(rows)} to score", flush=True)
    model = load_model(a.ckpt).eval()
    variants = a.variants.split(",")
    t0 = time.time()
    with open(outp, "a", encoding="utf-8") as fh, torch.inference_mode():
        for k, r in enumerate(todo):
            rec = {"pair_id": r["pair_id"], "ckpt": a.ckpt, "n_units": len(r["units"]), "variants": {}}
            w = [windows(r["source"], u["text"]) for u in r["units"]]
            for v in variants:
                q_all, ll_all = [], []
                for lo in range(0, len(w), a.batch):
                    ch = w[lo:lo + a.batch]
                    wins = [x[0] if v == "masked" else x[1] for x in ch]
                    tgts = [x[2] for x in ch]
                    sup = model.support_forward([r["candidate"]] * len(ch), wins, tgts)
                    q_all.extend(model.support_q(sup).float().tolist()); ll_all.extend(sup.loglik.float().tolist())
                rec["variants"][v] = {"q": q_all, "loglik": ll_all}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if (k + 1) % 50 == 0:
                print(f"[b18-mars2] {k + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    print("[b18-mars2] done", flush=True)


if __name__ == "__main__":
    main()
