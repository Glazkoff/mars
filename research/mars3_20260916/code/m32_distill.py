#!/usr/bin/env python3
"""M32 -- distil the Gemma sentence->propositions decomposition into LongT5-base, predict for held-out sentences,
and report agreement with Gemma (item matching by token F1 >= 0.6). Output in decompose format."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import m3common as m3
from m3common import gates

SEP = " | "


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--props", nargs="+", required=True, help="Gemma decompose outputs (all shards)")
    ap.add_argument("--backbone", default="google/long-t5-tglobal-base")
    ap.add_argument("--train-splits", default="train")
    ap.add_argument("--predict-splits", default="validation,test")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--out", default=str(m3.M3_OUT / "m32"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = [r for f in a.props for r in gates.load_jsonl(Path(f))]
    tr_s = set(a.train_splits.split(",")); pr_s = set(a.predict_splits.split(","))
    train = [r for r in rows if set(r["splits"]) & tr_s and not (set(r["splits"]) & pr_s)]   # never train on a held-out sentence
    pred = [r for r in rows if set(r["splits"]) & pr_s]
    if a.limit:
        train, pred = train[:a.limit], pred[:a.limit]
    if not train:
        raise SystemExit("no training sentences: every decomposed sentence belongs to a predict split")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
    bb = m3.snap(a.backbone); tok = AutoTokenizer.from_pretrained(bb)
    if tok.pad_token_id is None:
        tok.pad_token = "<pad>"
    model = AutoModelForSeq2SeqLM.from_pretrained(bb).to(dev)
    steps = a.steps or int(np.ceil(len(train) * a.epochs / a.batch))
    print(f"[m32-distill] train sentences {len(train)} predict {len(pred)} steps {steps} device {dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01); sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    rng = random.Random(a.seed); order = list(range(len(train))); rng.shuffle(order); model.train(); t0 = time.time(); losses = []
    amp = dev.startswith("cuda")
    for step in range(steps):
        lo = (step * a.batch) % max(1, len(order)); batch = [train[j] for j in order[lo:lo + a.batch]] or [train[j] for j in order[:a.batch]]
        enc = tok([f"propositions: {r['sentence']}" for r in batch], truncation=True, max_length=256, padding=True, return_tensors="pt").to(dev)
        lab = tok(text_target=[SEP.join(r["props"]) if r["props"] else "NONE" for r in batch], truncation=True, max_length=192, padding=True, return_tensors="pt")["input_ids"].to(dev)
        lab[lab == tok.pad_token_id] = -100
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            loss = model(**enc, labels=lab).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True); losses.append(float(loss.detach()))
        if (step + 1) % 50 == 0:
            print(f"[m32-distill] step {step + 1}/{steps} loss {np.mean(losses[-50:]):.4f} ({time.time() - t0:.0f}s)", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); model.eval()
    agree = {"tp": 0, "n_pred": 0, "n_gold": 0}
    with open(out / "distilled_props.jsonl", "w", encoding="utf-8") as fh, torch.inference_mode():
        for lo in range(0, len(pred), 32):
            ch = pred[lo:lo + 32]
            enc = tok([f"propositions: {r['sentence']}" for r in ch], truncation=True, max_length=256, padding=True, return_tensors="pt").to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                ids = model.generate(**enc, num_beams=4, max_new_tokens=192, no_repeat_ngram_size=3)
            for r, t in zip(ch, tok.batch_decode(ids, skip_special_tokens=True)):
                props = [x.strip() for x in t.split("|") if x.strip() and x.strip().upper() != "NONE"]
                fh.write(json.dumps({**{k: v for k, v in r.items() if k != "props"}, "props": props, "gemma_props": r["props"]}) + "\n")
                agree["n_pred"] += len(props); agree["n_gold"] += len(r["props"])
                agree["tp"] += sum(1 for p in props if any(gates.token_f1(p, g) >= 0.6 for g in r["props"]))
    P = agree["tp"] / max(1, agree["n_pred"]); R = agree["tp"] / max(1, agree["n_gold"])
    json.dump({"config": vars(a), "train_sentences": len(train), "predicted_sentences": len(pred), "agreement_vs_gemma": {"precision": P, "recall": R, **agree},
               "train_loss_last50": float(np.mean(losses[-50:])) if losses else None, "seconds": time.time() - t0}, open(out / "distill_result.json", "w"), indent=1)
    model.save_pretrained(out / "segmenter"); tok.save_pretrained(out / "segmenter")
    print(f"[m32-distill] agreement vs Gemma: P {P:.3f} R {R:.3f}; done", flush=True)


if __name__ == "__main__":
    main()
