#!/usr/bin/env python3
"""M32 -- Gemma-4-31B decomposes every source sentence into atomic propositions (the LLM unit inventory).

--build-tasks writes one task per sentence of every unique source in the B1 splits; shards then generate
(greedy, HF bf16, resumable by task_id). Output rows: {task_id, doc_key, splits, start, end, sentence, props}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import m3common as m3
from m3common import gates

judge = gates._load("b1_judge_labels", gates.REPO / "scripts" / "plan2026")   # Gemma loader + apply_chat (thinking off)
PROMPT = ("Decompose the sentence into its atomic facts. Rules: one fact per line; each fact is a short standalone sentence "
          "with an explicit subject; keep names, numbers and dates exactly as written; do not add or infer information; "
          "if the sentence states no fact, output NONE.\n\nSentence: {sentence}\n\nFacts:")
_LEAD = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def build_tasks(out: Path, min_words: int = 4) -> None:
    docs = {}
    for split in ("train", "validation", "test"):
        for r in gates.load_jsonl(gates.B1_DIR / f"pairs_{split}.jsonl"):
            k = hashlib.sha1(r["source"].encode()).hexdigest()[:12]
            d = docs.setdefault(k, {"source": r["source"], "splits": set(), "resources": set()})
            d["splits"].add(split); d["resources"].add(r["resource"])
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for k, d in docs.items():
            for i, (s, e) in enumerate(gates.sentence_spans(d["source"])):
                sent = d["source"][s:e].strip()
                if len(sent.split()) < min_words:
                    continue
                fh.write(json.dumps({"task_id": f"{k}:{i}", "doc_key": k, "splits": sorted(d["splits"]), "resources": sorted(d["resources"]),
                                     "start": s, "end": e, "sentence": sent}) + "\n"); n += 1
    print(f"[m32-tasks] docs {len(docs)} sentence tasks {n} -> {out}", flush=True)


def parse(text: str) -> list[str]:
    props = []
    for line in text.split("\n"):
        line = _LEAD.sub("", line).strip()
        if not line or line.upper().startswith("NONE") or line.lower().startswith("facts"):
            continue
        if len(line.split()) >= 2:
            props.append(line)
    return props[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(m3.M3_OUT / "m32" / "decompose_tasks.jsonl"))
    ap.add_argument("--build-tasks", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--model-snapshot", default="")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    tasks_p = Path(a.tasks); tasks_p.parent.mkdir(parents=True, exist_ok=True)
    if a.build_tasks:
        build_tasks(tasks_p); return
    import torch
    i, k = (int(x) for x in a.shard.split("/"))
    tasks = [t for j, t in enumerate(gates.load_jsonl(tasks_p)) if j % k == i]
    if a.limit:
        tasks = tasks[:a.limit]
    outp = Path(a.out or (m3.M3_OUT / "m32" / f"props_shard{i}.jsonl")); outp.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if outp.exists():
        for l in open(outp, encoding="utf-8"):
            try:
                done.add(json.loads(l)["task_id"])
            except Exception:
                pass
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"[m32-decompose] shard {i}/{k}: {len(tasks)} tasks, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return
    tok, model = judge.load_model(a.model_snapshot or m3.snap("google/gemma-4-31B-it"), a.device)
    chat = bool(getattr(tok, "chat_template", None)); t0 = time.time(); stats = defaultdict(int)
    with open(outp, "a", encoding="utf-8") as fh, torch.inference_mode():
        for s in range(0, len(todo), a.batch_size):
            ch = todo[s:s + a.batch_size]
            texts = [(judge.apply_chat(tok, PROMPT.format(sentence=t["sentence"])) if chat else PROMPT.format(sentence=t["sentence"])) for t in ch]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(a.device)
            ids = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            outs = tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for t, o in zip(ch, outs):
                props = parse(o); stats["props"] += len(props); stats["sent"] += 1
                fh.write(json.dumps({**t, "props": props}) + "\n")
            if (s // a.batch_size) % 20 == 0:
                fh.flush(); print(f"[m32-decompose] {s + len(ch)}/{len(todo)} ({time.time() - t0:.0f}s) {stats['props'] / max(1, stats['sent']):.2f} props/sentence", flush=True)
    print("[m32-decompose] done", flush=True)


if __name__ == "__main__":
    main()
