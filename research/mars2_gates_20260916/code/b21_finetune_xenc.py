#!/usr/bin/env python3
"""B21 -- supervised cross-encoder baseline on the SAME labels as MARS-2 (B1 training rows).

DeBERTa-large-MNLI, classification head re-initialised for {omitted, covered}; premise = candidate,
hypothesis = "[unit text] rendered claim" (proposition -> subject predicate object, entity -> its
source sentence). One epoch, AdamW, linear warm-up. Scores the m2 validation units (same hypothesis
form) and the human-labelled ACUs (hypothesis = ACU text). Omission score = P(omitted).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import common


def clip(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def hyp_m2(source: str, u: dict) -> str:
    return clip(f"[{u['text']}] {common.render_claim(source, u)}", 96)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--train", default=str(common.B1_DIR / "training_rows.jsonl"))
    ap.add_argument("--exclude-resource", default="")
    ap.add_argument("--tag", default="", help="system-name / file suffix, e.g. xdom-rose_samsum (cross-resource fold)")
    ap.add_argument("--model", default="microsoft/deberta-large-mnli")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=0, help="override the step count (smoke)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--out", required=True, help="checkpoint dir")
    ap.add_argument("--eval-m2", default=str(common.B1_DIR / "labels_validation.jsonl"))
    ap.add_argument("--eval-acu", default=str(common.OUT_ROOT / "b18" / "acu_units_validation.jsonl"))
    ap.add_argument("--scores-out", default=str(common.OUT_ROOT / "b21" / "scores"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2, ignore_mismatched_sizes=True).to(dev)

    rows = [r for r in common.load_jsonl(Path(a.train)) if r.get("split", "train") == "train" and r.get("units")]
    if a.exclude_resource:
        rows = [r for r in rows if r.get("resource") != a.exclude_resource]
    ex = [(clip(r["candidate"], 380), hyp_m2(r["source"], u), int(lab)) for r in rows
          for u, lab in zip(r["units"], r["unit_labels"]) if lab is not None]
    if a.limit:
        ex = ex[:a.limit]
    rng = random.Random(a.seed); rng.shuffle(ex)
    steps = a.steps or int(np.ceil(len(ex) * a.epochs / a.batch))
    print(f"[b21-xenc] seed={a.seed} rows={len(rows)} examples={len(ex)} steps={steps} device={dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    model.train(); t0 = time.time(); losses = []
    use_amp = False   # DeBERTa-v1 attention does masked_fill(finfo(dtype).min) and overflows under bf16 autocast; fp32 + TF32 instead
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    for step in range(steps):
        lo = (step * a.batch) % len(ex)
        batch = ex[lo:lo + a.batch] or ex[:a.batch]
        enc = tok([c for c, _, _ in batch], [h for _, h, _ in batch], truncation="longest_first", max_length=a.max_len,
                  padding=True, return_tensors="pt").to(dev)
        y = torch.tensor([lab for _, _, lab in batch], device=dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True); losses.append(float(loss))
        if (step + 1) % 100 == 0:
            print(f"[b21-xenc] step {step + 1}/{steps} loss {np.mean(losses[-100:]):.4f} ({time.time() - t0:.0f}s)", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    json.dump({**vars(a), "examples": len(ex), "steps": steps, "train_loss_last100": float(np.mean(losses[-100:])),
               "seconds": time.time() - t0}, open(out / "training_config.json", "w"), indent=1)

    model.eval()

    @torch.inference_mode()
    def p_omitted(prem: list[str], hyp: list[str]) -> list[float]:
        vals = []
        for s in range(0, len(prem), 64):
            enc = tok([clip(x, 380) for x in prem[s:s + 64]], hyp[s:s + 64], truncation="longest_first", max_length=a.max_len, padding=True,
                      return_tensors="pt").to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                pr = torch.softmax(model(**enc).logits.float(), -1)[:, 0]
            vals.extend(pr.tolist())
        return vals

    sysname = f"xenc{'[' + a.tag + ']' if a.tag else ''}:{a.model.split('/')[-1]}"
    for unit_set, path in (("m2", a.eval_m2), ("acu", a.eval_acu)):
        if not path or not Path(path).exists():
            print(f"[b21-xenc] skip {unit_set}: {path} missing", flush=True); continue
        erows = [r for r in common.load_jsonl(Path(path)) if r.get("units")]
        if a.limit:
            erows = erows[:max(2, a.limit // 10)]
        flat = [(i, j, clip(u["text"], 96) if unit_set == "acu" else hyp_m2(r["source"], u)) for i, r in enumerate(erows) for j, u in enumerate(r["units"])]
        vals = p_omitted([erows[i]["candidate"] for i, _, _ in flat], [h for _, _, h in flat])
        per = {r["pair_id"]: [np.nan] * len(r["units"]) for r in erows}
        for (i, j, _), v in zip(flat, vals):
            per[erows[i]["pair_id"]][j] = v
        tag = hashlib.sha1(f"{a.seed}".encode()).hexdigest()[:6]
        common.write_scores(Path(a.scores_out) / f"{unit_set}_xenc{'_' + a.tag if a.tag else ''}_seed{a.seed}_{tag}.jsonl", sysname, unit_set, per)
        print(f"[b21-xenc] scored {unit_set}: {len(flat)} units", flush=True)
    print("[b21-xenc] done", flush=True)


if __name__ == "__main__":
    main()
