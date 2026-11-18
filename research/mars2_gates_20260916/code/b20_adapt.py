#!/usr/bin/env python3
"""B20 -- continue training a cross-resource fold checkpoint on label-free pseudo rows of the held-out resource.

Same recipe as B2 (support BCE on the pseudo labels + the recovery anchor on positives for
recovery arms), same optimiser family, no B1 label of the held-out resource is read. The adapted
checkpoint is saved where b2_score.sbatch expects it:
  /home/user/models/plan2026/b2/<tag><arm>_seed<seed>/epoch0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

import common

sys.path.insert(0, str(common.REPO))
from mars_v2.model import load_model, save_model  # noqa: E402
from mars_v2.train import load_quadruples, support_examples  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="fold checkpoint dir (…/epoch1)")
    ap.add_argument("--rows", required=True, help="pseudo rows from b20_build_pseudo.py")
    ap.add_argument("--out", required=True, help="…/b2/<tag><arm>_seed<seed> (epoch0 is created inside)")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--units-per-row", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--aux-coef", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--log-every", type=int, default=25)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    base_cfg = {}
    p = Path(a.ckpt) / "training_config.json"
    if p.exists():
        base_cfg = json.load(open(p))
    arm = str(base_cfg.get("arm", "recovery"))
    rows = load_quadruples(a.rows)
    model = load_model(a.ckpt); model.train()
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=0.01)
    bce = torch.nn.BCEWithLogitsLoss(); rng = random.Random(a.seed)
    print(f"[b20-adapt] ckpt={a.ckpt} arm={arm} lf={model.likelihood_features} rows={len(rows)} steps={a.steps} device={dev}", flush=True)
    t0 = time.time(); losses = []
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _m in range(a.accum):
            ex = []
            while len(ex) < a.micro:
                ex.extend(support_examples(rng.choice(rows), step // 200, a.units_per_row))
            ex = ex[:a.micro]
            cands, wins, tgts, labels = zip(*ex)
            sup = model.support_forward(list(cands), list(wins), list(tgts))
            logits = model.support_head(model.support_features(sup)).squeeze(-1)
            y = torch.tensor(labels, device=logits.device, dtype=logits.dtype)
            loss = bce(logits, y)
            pos = y > 0.5
            if arm == "recovery" and pos.any():
                loss = loss - a.aux_coef * sup.loglik[pos].mean()
            (loss / a.accum).backward(); losses.append(float(loss))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % a.log_every == 0:
            print(f"[b20-adapt] step {step}/{a.steps} loss {np.mean(losses[-a.log_every * a.accum:]):.4f} ({time.time() - t0:.0f}s)", flush=True)
    out = Path(a.out) / "epoch0"; save_model(model, str(out))
    json.dump({**base_cfg, "arm": arm, "likelihood_features": model.likelihood_features, "adapted_from": a.ckpt,
               "adapt_rows": a.rows, "adapt_rows_sha256": hashlib.sha256(Path(a.rows).read_bytes()).hexdigest(),
               "adapt_steps": a.steps, "adapt_lr": a.lr, "adapt_seed": a.seed, "train_loss_last": float(np.mean(losses[-200:])),
               "seconds": time.time() - t0}, open(out / "training_config.json", "w"), indent=1)
    print(f"[b20-adapt] saved {out}", flush=True)


if __name__ == "__main__":
    main()
