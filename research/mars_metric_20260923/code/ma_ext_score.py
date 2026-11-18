#!/usr/bin/env python3
"""M-A.3 -- the coverage and multi-dimension comparators a reviewer asked for, over the same meta-evaluation pairs
(pairs_ma.jsonl), written as scores_<metric>.jsonl with one row per pair so that ma_meta.py folds them in beside
every other column. Registered as Amendment M-A.3 in PREREG.md before any of it ran.

  unieval    UniEval (Zhong et al., 2022), MingZhong/unieval-sum: coherence, consistency, fluency (source / sentence
             based) and relevance (maximum over references); unieval_overall = mean of the four
  a3cu       A3CU recall and F against the reference (maximum over references)       -- ~/.venv-acu worker
  a2cu       A2CU recall: reference ACUs generated, each matched to the candidate   -- ~/.venv-acu worker
  questeval  QuestEval source-based score, and its precision and recall directions  -- ~/.venv-questeval worker
  qaeval     QAEval EM and F1 against the reference (maximum over references)       -- ~/.venv-qaeval-gpu worker
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKERS = HERE / "workers"


def refs_of(p: dict) -> list[str]:
    return [r for r in (p.get("references") or []) if r and r.strip()]


def sentences(t: str) -> list[str]:
    s = [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", t.strip()) if x.strip()]
    return s or [t]


class UniEval:
    YES, NO = 2163, 465  # T5 vocabulary ids of "Yes" / "No", as in the UniEval release

    def __init__(self, path: str, batch: int, dev: str):
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        self.torch = torch; self.dev = dev; self.batch = batch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = T5ForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16).to(dev).eval()
        assert self.tok.convert_tokens_to_ids("▁Yes") == self.YES and self.tok.convert_tokens_to_ids("▁No") == self.NO

    def score(self, inputs: list[str]) -> list[float]:
        torch = self.torch
        order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]))
        out = [0.0] * len(inputs)
        with torch.inference_mode():
            for s in range(0, len(order), self.batch):
                idx = order[s:s + self.batch]
                enc = self.tok([inputs[i] for i in idx], max_length=1024, truncation=True, padding=True, return_tensors="pt").to(self.dev)
                dec = torch.zeros((len(idx), 1), dtype=torch.long, device=self.dev)
                logits = self.m(**enc, decoder_input_ids=dec).logits[:, 0, :].float()
                p = torch.softmax(logits, -1)
                y, n = p[:, self.YES], p[:, self.NO]
                for i, v in zip(idx, (y / (y + n)).tolist()):
                    out[i] = v
        return out


def run_unieval(pairs, a) -> list[dict]:
    ue = UniEval(a.unieval, a.batch, a.device)
    rows = [{"pair_id": p["pair_id"], "dataset": p["dataset"]} for p in pairs]
    coh = ue.score([f"question: Is this a coherent summary to the document? </s> summary: {p['candidate']} </s> document: {p['source']}" for p in pairs])
    ins, own = [], []
    for i, p in enumerate(pairs):
        for s in sentences(p["candidate"]):
            ins.append(f"question: Is this claim consistent with the document? </s> claim: {s} </s> document: {p['source']}"); own.append(i)
    con = ue.score(ins)
    ins2, own2 = [], []
    for i, p in enumerate(pairs):
        for s in sentences(p["candidate"]):
            ins2.append(f"question: Is this a fluent paragraph? </s> paragraph: {s}"); own2.append(i)
    flu = ue.score(ins2)
    ins3, own3 = [], []
    for i, p in enumerate(pairs):
        for r in refs_of(p):
            ins3.append(f"question: Is this summary relevant to the reference? </s> summary: {p['candidate']} </s> reference: {r}"); own3.append(i)
    rel = ue.score(ins3)

    def agg(vals, own, fn):
        acc: dict[int, list[float]] = {}
        for o, v in zip(own, vals):
            acc.setdefault(o, []).append(v)
        return {o: fn(v) for o, v in acc.items()}
    con_a, flu_a, rel_a = agg(con, own, lambda v: sum(v) / len(v)), agg(flu, own2, lambda v: sum(v) / len(v)), agg(rel, own3, max)
    for i, r in enumerate(rows):
        r["unieval_coherence"] = coh[i]; r["unieval_consistency"] = con_a[i]; r["unieval_fluency"] = flu_a[i]
        dims = [coh[i], con_a[i], flu_a[i]]
        if i in rel_a:
            r["unieval_relevance"] = rel_a[i]; dims.append(rel_a[i])
        r["unieval_overall"] = sum(dims) / len(dims)
    return rows


def run_worker(pairs, a, python: str, script: str, extra: list[str]) -> list[dict]:
    tmp = Path(a.out).with_suffix(".work"); tmp.mkdir(parents=True, exist_ok=True)
    inp, outp = tmp / "in.jsonl", tmp / "out.jsonl"
    with open(inp, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps({"id": p["pair_id"], "doc": p["source"], "summary": p["candidate"], "refs": refs_of(p)}) + "\n")
    cmd = [python, str(WORKERS / script), "--in", str(inp), "--out", str(outp)] + extra
    print("[ma-ext] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ})
    got = {}
    for line in open(outp, encoding="utf-8"):
        r = json.loads(line); got[r.pop("id")] = r
    return [{"pair_id": p["pair_id"], "dataset": p["dataset"], **got.get(p["pair_id"], {})} for p in pairs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--metric", required=True, choices=["unieval", "a3cu", "a2cu", "questeval", "qaeval"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--unieval", default="MingZhong/unieval-sum")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--py-acu", default=os.path.expanduser("~/.venv-acu/bin/python"))
    ap.add_argument("--py-qe", default=os.path.expanduser("~/.venv-questeval/bin/python"))
    ap.add_argument("--py-qa", default=os.path.expanduser("~/.venv-qaeval-gpu/bin/python"))
    ap.add_argument("--qaeval-models", default="/home/user/data/qaeval_models")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: score the i-th of n contiguous shards")
    a = ap.parse_args()
    pairs = [json.loads(l) for l in open(a.pairs, encoding="utf-8")]
    i, n = (int(x) for x in a.shard.split("/"))
    lo, hi = len(pairs) * i // n, len(pairs) * (i + 1) // n
    pairs = pairs[lo:hi]
    if a.limit:
        pairs = pairs[:a.limit]
    t0 = time.time()
    if a.metric == "unieval":
        rows = run_unieval(pairs, a)
    elif a.metric in ("a3cu", "a2cu"):
        rows = run_worker(pairs, a, a.py_acu, "acu_worker.py", ["--which", a.metric, "--acu-cache", str(Path(a.out).with_suffix(".acus.json"))])
    elif a.metric == "questeval":
        rows = run_worker(pairs, a, a.py_qe, "questeval_pr_worker.py", ["--log-dir", str(Path(a.out).with_suffix(".qelogs"))])
    else:
        rows = run_worker(pairs, a, a.py_qa, "qaeval_worker.py", ["--gen", f"{a.qaeval_models}/generation/model.tar.gz",
                          "--ans", f"{a.qaeval_models}/answering/model", "--cuda", "0" if a.device.startswith("cuda") else "-1",
                          "--workdir", a.qaeval_models])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    scored = sum(1 for r in rows if len(r) > 2)
    print(f"[ma-ext] {a.metric}: {scored}/{len(rows)} pairs scored in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
