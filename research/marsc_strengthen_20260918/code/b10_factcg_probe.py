#!/usr/bin/env python3
"""B10 probe -- FactCG-DeBERTa-L under BOTH candidate input contracts, side by side.

This file is a demonstration, not a gate. The gate lives in the production path
(`b10_verifier_swap.py --validate-aggrefact`, with an absolute floor), where it runs immediately before the
inventory is scored and can actually refuse.

What it demonstrates is a trap worth a paragraph in the paper. `yaxili96/FactCG-DeBERTa-v3-Large` ships
`num_labels: null`, no `id2label`, and no statement of its input format. The obvious reading -- it is a
DeBERTa cross-encoder, so encode `(premise, claim)` as a sentence pair, exactly as every other cross-encoder
in this project is driven -- produces a near-constant output and chance accuracy. Its own repository instead
formats ONE string from an instruction template (`factcg/utils.INSTRUCTION_TEMPLATE`). Same weights, same
data, two contracts, and the difference is the whole result:

    pair encoding          ~0.51 BAcc  (chance; reads as "this model is weak")
    instruction template   ~0.81 BAcc  (against 0.756 published)

A verifier run under the wrong contract does not announce itself -- it looks like a weak baseline, and a
weak baseline is a publishable-looking number. That is the same failure mode the audit paper documents in
other people's work, which is why every verifier added to this project is now re-measured against its
published benchmark before it is allowed to contribute a single score.

Run:  python b10_factcg_probe.py --snapshot <dir> [--n-aggrefact 2000]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

# Quoted from factcg/utils.py at derenlei/FactCG@main.
INSTRUCTION_TEMPLATE = ("{text_a}\n\nChoose your answer: based on the paragraph above can we conclude that "
                        "\"{text_b}\"?\n\nOPTIONS:\n- Yes\n- No\nI think the answer is ")

PUBLISHED_BACC = 0.756


def load(snapshot: str, device: str):
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    cfg = AutoConfig.from_pretrained(snapshot, num_labels=2, finetuning_task="text-classification",
                                     local_files_only=True)
    cfg.problem_type = "single_label_classification"
    tok = AutoTokenizer.from_pretrained(snapshot, use_fast=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(snapshot, config=cfg, local_files_only=True)
    return tok, model.eval().to(device)


@torch.no_grad()
def p_label1(tok, model, docs, claims, device, mode: str, batch: int = 32) -> np.ndarray:
    """P(label=1) under `mode`: 'pair' = sentence-pair encoding, 'template' = the repo's own single string."""
    out = []
    for s in range(0, len(docs), batch):
        d, c = docs[s:s + batch], claims[s:s + batch]
        if mode == "pair":
            enc = tok(d, c, return_tensors="pt", padding=True, truncation=True, max_length=512)
        else:
            txt = [INSTRUCTION_TEMPLATE.format(text_a=x, text_b=y) for x, y in zip(d, c)]
            enc = tok(txt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        out.extend(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy().tolist())
    return np.array(out)


def bacc(y: np.ndarray, pred: np.ndarray) -> float:
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    return float(0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--n-aggrefact", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    tok, model = load(a.snapshot, a.device)
    from datasets import load_dataset
    ds = load_dataset("lytang/LLM-AggreFact", split="test")
    rng = np.random.default_rng(20260921)
    idx = rng.choice(len(ds), size=min(2000, len(ds)), replace=False)[:a.n_aggrefact]
    sub = ds.select(idx)
    y = np.array(sub["label"], dtype=int)
    docs, claims = list(sub["doc"]), list(sub["claim"])
    print(f"[probe] LLM-AggreFact n={len(y)}, published BAcc {PUBLISHED_BACC:.3f}", flush=True)

    for mode in ("pair", "template"):
        pv = p_label1(tok, model, docs, claims, a.device, mode)
        b1 = bacc(y, (pv > 0.5).astype(int))
        b0 = bacc(y, (pv <= 0.5).astype(int))
        print(f"[probe] {mode:9s}: P(label=1) mean {pv.mean():.4f} sd {pv.std():.4f} | "
              f"BAcc label1=support {b1:.4f} | label0=support {b0:.4f} | "
              f"best {max(b1, b0):.4f} vs published {PUBLISHED_BACC:.3f}", flush=True)

    print("[probe] the contract that lands near the published figure is the correct one; the other is a "
          "model being driven wrongly, which presents as a weak baseline rather than as an error.")


if __name__ == "__main__":
    main()
