#!/usr/bin/env python3
"""H-J11 (gap fill, 2026-09-27) -- the DISTILLED fact inventory for the UniSumEval pool.

The RoSE distilled inventory (research/mars3_20260916, M32) is the frozen LongT5-base segmenter's propositions with
no entity units. UniSumEval was only ever decomposed by Gemma. This script applies the SAME frozen segmenter with the
SAME decoding (prompt "propositions: <sentence>", 256-token input, 4 beams, 192 new tokens, no_repeat_ngram_size 3,
"|"-separated output, NONE dropped; m32_distill.py) to the frozen UniSumEval sentence tasks, then writes a
propositions-only inventory beside the Gemma+ent one, under a new file name so the frozen inventory is untouched.

  segment    --tasks decompose_tasks_unisum.jsonl --segmenter $M3/m32/segmenter --out props_distilled_unisum.jsonl
  inventory  --acu acu_units_unisum.jsonl --props props_distilled_unisum.jsonl --out units_distilled_unisum.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def segment(a) -> None:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"; amp = dev == "cuda"
    tok = AutoTokenizer.from_pretrained(a.segmenter)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.segmenter).to(dev).eval()
    tasks = load(a.tasks); t0 = time.time(); n_props = 0
    with open(a.out, "w", encoding="utf-8") as fh, torch.inference_mode():
        for lo in range(0, len(tasks), a.batch):
            ch = tasks[lo:lo + a.batch]
            enc = tok([f"propositions: {r['sentence']}" for r in ch], truncation=True, max_length=256, padding=True,
                      return_tensors="pt").to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                ids = model.generate(**enc, num_beams=4, max_new_tokens=192, no_repeat_ngram_size=3)
            for r, t in zip(ch, tok.batch_decode(ids, skip_special_tokens=True)):
                props = [x.strip() for x in t.split("|") if x.strip() and x.strip().upper() != "NONE"]
                n_props += len(props)
                fh.write(json.dumps({**r, "props": props}) + "\n")
            if (lo // a.batch) % 50 == 0:
                print(f"[hj11-seg] {lo + len(ch)}/{len(tasks)} sentences ({time.time() - t0:.0f}s)", flush=True)
    print(f"[hj11-seg] {len(tasks)} sentences -> {n_props} propositions in {time.time() - t0:.0f}s -> {a.out}", flush=True)


def inventory(a) -> None:
    props = defaultdict(list)
    for r in load(a.props):
        for p in r.get("props", []):
            props[r["doc_key"]].append({"kind": "prop_llm", "text": p, "start": r["start"], "end": r["end"], "label": "", "args": {}})
    n_units, docs = [], set()
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in load(a.acu):
            k = hashlib.sha1(r["source"].encode()).hexdigest()[:12]; docs.add(k)
            units = props.get(k, []); n_units.append(len(units))
            fh.write(json.dumps({"pair_id": r["pair_id"], "resource": r["resource"], "system": r["system"], "split": "unisum",
                                 "source": r["source"], "candidate": r["candidate"], "units": units}) + "\n")
    empty = sum(1 for k in docs if not props.get(k))
    print(f"[hj11-inv] {len(n_units)} pairs, {len(docs)} documents ({empty} without propositions), "
          f"{sum(n_units) / max(1, len(n_units)):.1f} units/pair -> {a.out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("segment"); s.add_argument("--tasks", required=True); s.add_argument("--segmenter", required=True)
    s.add_argument("--out", required=True); s.add_argument("--batch", type=int, default=32)
    i = sub.add_parser("inventory"); i.add_argument("--acu", required=True); i.add_argument("--props", required=True)
    i.add_argument("--out", required=True)
    a = ap.parse_args(); {"segment": segment, "inventory": inventory}[a.cmd](a)


if __name__ == "__main__":
    main()
