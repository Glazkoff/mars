#!/usr/bin/env python3
"""M31 -- MARS-3 omission tagger: one encoder pass over [candidate ; source], a per-token omission logit.

Training: unit-level BCE (mean logit over the unit's source tokens vs the B1 label, 1 = omitted) plus a token-level
auxiliary BCE on tokens covered by labelled units. Optional phase 1 on label-free extractive pseudo rows
(--pretrain-rows; the recovery idea as pre-training). --source-only drops the candidate (summary-blind control).
Scoring is inventory-free: any span or supplied unit is scored by pooling its aligned source tokens.
Outputs unit_scores (gates format) for the m2 set, the human ACU set and any extra units file.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import m3common as m3
from m3common import gates


class Tagger(nn.Module):
    def __init__(self, backbone: str):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(backbone)
        self.head = nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, **enc):
        h = self.enc(**enc).last_hidden_state
        return self.head(h).squeeze(-1)          # (B, T) omission logits


def encode(tok, cands, sources, max_len, source_only):
    enc = tok(["" if source_only else c for c in cands], sources, truncation="only_second", max_length=max_len,
              padding=True, return_offsets_mapping=True, return_tensors="pt")
    offs = enc.pop("offset_mapping").tolist()
    is_src = [[sid == 1 for sid in enc.sequence_ids(i)] for i in range(len(sources))]
    return enc, offs, is_src


def unit_mask(source, u, offs, is_src):
    if u.get("kind") == "prop_llm":        # LLM proposition: sentence span + its own content words
        return m3.prop_token_mask(source, u["text"], int(u["start"]), int(u["end"]), offs, is_src)
    if u.get("start") is None:
        return m3.acu_token_mask(source, u["text"], offs, is_src)
    return m3.span_token_mask(int(u["start"]), int(u["end"]), offs, is_src)


def unit_masks(row, offs, is_src):
    return [unit_mask(row["source"], u, offs, is_src) for u in row["units"]]


def batch_loss(model, tok, rows, dev, max_len, source_only, aux_w=0.5):
    enc, offs, is_src = encode(tok, [r["candidate"] for r in rows], [r["source"] for r in rows], max_len, source_only)
    enc = {k: v.to(dev) for k, v in enc.items()}
    logits = model(**enc)
    losses, tok_logits, tok_targets = [], [], []
    bce = nn.functional.binary_cross_entropy_with_logits
    for i, r in enumerate(rows):
        masks = unit_masks(r, offs[i], is_src[i])
        for m, lab in zip(masks, r["unit_labels"]):
            if lab is None or not m.any():
                continue
            idx = torch.as_tensor(np.flatnonzero(m), device=dev)
            y = torch.tensor(1.0 - float(lab), device=dev)
            losses.append(bce(logits[i, idx].mean(), y))
            tok_logits.append(logits[i, idx]); tok_targets.append(y.expand(len(idx)))
    if not losses:
        return None
    loss = torch.stack(losses).mean()
    if aux_w > 0:
        loss = loss + aux_w * bce(torch.cat(tok_logits), torch.cat(tok_targets))
    return loss


def train_phase(model, tok, rows, dev, a, epochs, lr, tag):
    if not rows:
        return
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = a.steps or int(np.ceil(len(rows) * epochs / a.batch))
    from transformers import get_linear_schedule_with_warmup
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    rng = random.Random(a.seed); order = list(range(len(rows))); rng.shuffle(order)
    model.train(); t0 = time.time(); losses = []
    for step in range(steps):
        lo = (step * a.batch) % len(order)
        batch = [rows[j] for j in order[lo:lo + a.batch]] or [rows[j] for j in order[:a.batch]]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            loss = batch_loss(model, tok, batch, dev, a.max_len, a.source_only)
        if loss is None:
            continue
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True); losses.append(float(loss.detach()))
        if (step + 1) % 25 == 0:
            print(f"[m31:{tag}] step {step + 1}/{steps} loss {np.mean(losses[-25:]):.4f} ({time.time() - t0:.0f}s)", flush=True)
    return float(np.mean(losses[-50:])) if losses else None


@torch.inference_mode()
def token_probs(model, tok, rows, dev, a):
    """Per-pair token omission probabilities with offsets (source tokens only meaningful)."""
    out = []
    for lo in range(0, len(rows), a.eval_batch):
        ch = rows[lo:lo + a.eval_batch]
        enc, offs, is_src = encode(tok, [r["candidate"] for r in ch], [r["source"] for r in ch], a.max_len, a.source_only)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            p = torch.sigmoid(model(**enc).float()).cpu().numpy()
        for i in range(len(ch)):
            out.append((p[i], offs[i], is_src[i]))
    return out


def score_units(model, tok, rows, dev, a, unit_set, sysname, path):
    per = {}
    for r, (p, offs, is_src) in zip(rows, token_probs(model, tok, rows, dev, a)):
        sc = []
        for u in r["units"]:
            m = m3.acu_token_mask(r["source"], u["text"], offs, is_src) if unit_set == "acu" else unit_mask(r["source"], u, offs, is_src)
            sc.append(m3.safe_mean(p[m]) if m.any() else np.nan)
        per[r["pair_id"]] = sc
    gates.write_scores(path, sysname, unit_set, per)
    print(f"[m31] wrote {path.name}: {len(per)} pairs", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="answerdotai/ModernBERT-large")
    ap.add_argument("--train", default=str(gates.B1_DIR / "training_rows.jsonl"))
    ap.add_argument("--pretrain-rows", default="", help="label-free extractive pseudo rows (phase 1)")
    ap.add_argument("--pretrain-epochs", type=float, default=1.0)
    ap.add_argument("--exclude-resource", default="")
    ap.add_argument("--label-frac", type=float, default=1.0)
    ap.add_argument("--source-only", action="store_true")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--steps", type=int, default=0, help="override (smoke)")
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--tag", required=True, help="system name suffix: m3:<tag>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--eval-m2", default=str(gates.B1_DIR / "labels_validation.jsonl"))
    ap.add_argument("--eval-acu", default=str(gates.OUT_ROOT / "b18" / "acu_units_validation.jsonl"))
    ap.add_argument("--eval-units", nargs="*", default=[], help="name=path extra units files (pairs with units)")
    ap.add_argument("--scores-out", default=str(m3.M3_OUT / "scores"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer
    bb = m3.snap(a.backbone); tok = AutoTokenizer.from_pretrained(bb)
    model = Tagger(bb).to(dev)
    rows = [r for r in gates.load_jsonl(Path(a.train)) if r.get("split", "train") == "train" and r.get("units")]
    if a.exclude_resource:
        rows = [r for r in rows if r.get("resource") != a.exclude_resource]
    rows = [r for r in rows if m3.frac_keep(r["pair_id"], r["source"], a.label_frac, a.seed)]
    if a.limit:
        rows = rows[:a.limit]
    print(f"[m31:{a.tag}] backbone={bb} train rows={len(rows)} source_only={a.source_only} frac={a.label_frac} excl={a.exclude_resource!r} device={dev}", flush=True)
    t0 = time.time(); rep = {"tag": a.tag, "config": vars(a), "train_rows": len(rows)}
    if a.pretrain_rows:
        pre = [r for r in gates.load_jsonl(Path(a.pretrain_rows)) if r.get("units")]
        if a.exclude_resource:
            pre = [r for r in pre if r.get("resource") != a.exclude_resource]
        if a.limit:
            pre = pre[:a.limit]
        rep["pretrain_rows"] = len(pre); rep["pretrain_loss"] = train_phase(model, tok, pre, dev, a, a.pretrain_epochs, a.lr, a.tag + "/pre")
    rep["train_loss"] = train_phase(model, tok, rows, dev, a, a.epochs, a.lr, a.tag)
    rep["train_seconds"] = time.time() - t0
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.save_model:
        model.enc.save_pretrained(out / "backbone"); tok.save_pretrained(out / "backbone"); torch.save(model.head.state_dict(), out / "head.pt")
    model.eval(); sysname = f"m3:{a.tag}"; so = Path(a.scores_out); so.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    m2 = [r for r in gates.load_jsonl(Path(a.eval_m2)) if r.get("units")]
    if a.limit:
        m2 = m2[:max(2, a.limit // 5)]
    score_units(model, tok, m2, dev, a, "m2", sysname, so / f"m2_{a.tag}_seed{a.seed}.jsonl")
    rep["seconds_per_pair_m2"] = (time.time() - t1) / max(1, len(m2))
    if Path(a.eval_acu).exists():
        acu = gates.load_jsonl(Path(a.eval_acu))
        if a.limit:
            acu = acu[:max(2, a.limit // 5)]
        score_units(model, tok, acu, dev, a, "acu", sysname, so / f"acu_{a.tag}_seed{a.seed}.jsonl")
    for spec in a.eval_units:
        name, path = spec.split("=", 1)
        ur = [r for r in gates.load_jsonl(Path(path)) if r.get("units")]
        if a.limit:
            ur = ur[:max(2, a.limit // 5)]
        score_units(model, tok, ur, dev, a, name, sysname, so / f"{name}_{a.tag}_seed{a.seed}.jsonl")
    json.dump(rep, open(out / "result.json", "w"), indent=1)
    print(f"[m31:{a.tag}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
