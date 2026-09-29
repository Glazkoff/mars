#!/usr/bin/env python3
"""M31 -- recovery writer: a seq2seq model that WRITES the omitted content of a summary.

LongT5-base, input "summary: {candidate} source: {source}", target = the omitted units' texts (B1 label 0) joined by
" | " in source order, or NONE. Generated items are matched to units / human ACUs by token F1 (as the open-ended judges
were), giving a soft score (max F1) and a binary one (F1 >= 0.5). This is the recovery idea in its original sense and
the task open-ended LLM judges fail at (0.51).
"""
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


def make_target(r: dict) -> str:
    om = [u["text"] for u, lab in zip(r["units"], r["unit_labels"]) if lab is not None and int(lab) == 0]
    return SEP.join(om) if om else "NONE"


def parse_items(text: str) -> list[str]:
    return [x.strip() for x in text.split("|") if x.strip() and x.strip().upper() != "NONE"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="google/long-t5-tglobal-base")
    ap.add_argument("--train", default=str(gates.B1_DIR / "training_rows.jsonl"))
    ap.add_argument("--exclude-resource", default="")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-src", type=int, default=2048)
    ap.add_argument("--max-tgt", type=int, default=256)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--tag", default="writer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-m2", default=str(gates.B1_DIR / "labels_validation.jsonl"))
    ap.add_argument("--eval-acu", default=str(gates.OUT_ROOT / "b18" / "acu_units_validation.jsonl"))
    ap.add_argument("--scores-out", default=str(m3.M3_OUT / "scores"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
    bb = m3.snap(a.backbone); tok = AutoTokenizer.from_pretrained(bb)
    if tok.pad_token_id is None:
        tok.pad_token = "<pad>"
    model = AutoModelForSeq2SeqLM.from_pretrained(bb).to(dev)
    rows = [r for r in gates.load_jsonl(Path(a.train)) if r.get("split", "train") == "train" and r.get("units")]
    if a.exclude_resource:
        rows = [r for r in rows if r.get("resource") != a.exclude_resource]
    if a.limit:
        rows = rows[:a.limit]
    steps = a.steps or int(np.ceil(len(rows) * a.epochs / (a.batch * a.accum)))
    print(f"[writer] rows={len(rows)} steps={steps} device={dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    rng = random.Random(a.seed); order = list(range(len(rows))); rng.shuffle(order)
    model.train(); t0 = time.time(); losses = []; k = 0
    amp = dev.startswith("cuda")
    for step in range(steps):
        for _ in range(a.accum):
            lo = (k * a.batch) % len(order); k += 1
            batch = [rows[j] for j in order[lo:lo + a.batch]] or [rows[j] for j in order[:a.batch]]
            enc = tok([f"summary: {r['candidate']} source: {r['source']}" for r in batch], truncation=True, max_length=a.max_src,
                      padding=True, return_tensors="pt").to(dev)
            lab = tok(text_target=[make_target(r) for r in batch], truncation=True, max_length=a.max_tgt, padding=True, return_tensors="pt")["input_ids"].to(dev)
            lab[lab == tok.pad_token_id] = -100
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                loss = model(**enc, labels=lab).loss
            (loss / a.accum).backward(); losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % 25 == 0:
            print(f"[writer] step {step + 1}/{steps} loss {np.mean(losses[-50:]):.4f} ({time.time() - t0:.0f}s)", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model.eval()

    @torch.inference_mode()
    def generate(pairs: list[dict]) -> dict[str, list[str]]:
        gen = {}
        for lo in range(0, len(pairs), 8):
            ch = pairs[lo:lo + 8]
            enc = tok([f"summary: {r['candidate']} source: {r['source']}" for r in ch], truncation=True, max_length=a.max_src, padding=True, return_tensors="pt").to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                ids = model.generate(**enc, num_beams=a.beams, max_new_tokens=a.max_tgt, no_repeat_ngram_size=3)
            for r, t in zip(ch, tok.batch_decode(ids, skip_special_tokens=True)):
                gen[r["pair_id"]] = parse_items(t)
        return gen

    m2 = [r for r in gates.load_jsonl(Path(a.eval_m2)) if r.get("units")]
    acu = gates.load_jsonl(Path(a.eval_acu)) if Path(a.eval_acu).exists() else []
    if a.limit:
        m2, acu = m2[:max(2, a.limit // 5)], acu[:max(2, a.limit // 5)]
    pairs = {r["pair_id"]: r for r in m2 + acu}
    t1 = time.time(); gen = generate(list(pairs.values())); sec = (time.time() - t1) / max(1, len(pairs))
    so = Path(a.scores_out); so.mkdir(parents=True, exist_ok=True)
    with open(out / f"generations_seed{a.seed}.jsonl", "w") as fh:
        for pid, items in gen.items():
            fh.write(json.dumps({"pair_id": pid, "items": items}) + "\n")
    for unit_set, rs in (("m2", m2), ("acu", acu)):
        soft, hard = {}, {}
        for r in rs:
            items = gen.get(r["pair_id"], [])
            s = [max((gates.token_f1(it, u["text"]) for it in items), default=0.0) for u in r["units"]]
            soft[r["pair_id"]] = s; hard[r["pair_id"]] = [1.0 if x >= 0.5 else 0.0 for x in s]
        if rs:
            gates.write_scores(so / f"{unit_set}_{a.tag}_seed{a.seed}.jsonl", f"m3:{a.tag}", unit_set, soft)
            gates.write_scores(so / f"{unit_set}_{a.tag}bin_seed{a.seed}.jsonl", f"m3:{a.tag}|bin", unit_set, hard)
    n_items = float(np.mean([len(v) for v in gen.values()])) if gen else 0.0
    json.dump({"tag": a.tag, "config": vars(a), "train_rows": len(rows), "train_loss_last50": float(np.mean(losses[-50:])) if losses else None,
               "seconds_per_pair_generate": sec, "mean_items_per_pair": n_items, "train_seconds": time.time() - t0}, open(out / "result.json", "w"), indent=1)
    print(f"[writer] done; {n_items:.1f} items/pair, {sec:.2f} s/pair", flush=True)


if __name__ == "__main__":
    main()
