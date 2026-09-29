#!/usr/bin/env python3
"""M-C -- one verifier for both axes: continued fine-tuning of FactCG-DeBERTa-v3-Large on MiniCheck's C2D and D2C
synthetic sets (MIT; label 1 = supported) mixed one-to-one per batch with the natural coverage cells of e0
(premise = summary, hypothesis = source fact, label 1 = conveyed), on FactCG's own input contract (the prompt
template of b10_verifier_swap, class 1 = support). Registered under M-C in PREREG.md: lr 1e-5, 1,500 steps,
batch 16, no early stopping, evaluation at the end only.

After training the script evaluates the coverage axis on RoSE validation for the trained model, for the untouched
FactCG (zero-shot) and for the deployed MARS-C verifier on the identical pairs: pooled AUROC on the human ACU cells
and the same-fact crossed accuracy (mc_summary_conditioning's estimator, document macro with its bootstrap). The
AggreFact axis is scored afterwards by mb_score.py --verifier factcg --snapshot <this model>.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
sys.path.insert(0, str(REPO / "research" / "mars_metric_20260923" / "code"))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


swap = _load(REPO / "research" / "marsc_strengthen_20260918" / "code" / "b10_verifier_swap.py")
msc = _load(REPO / "research" / "marsc_20260916" / "code" / "mc_summary_conditioning.py")
TEMPLATE = swap.FACTCG_TEMPLATE


def load_cells(docs_path: Path, exclude=("gold",)) -> list[tuple[str, str, int]]:
    cells = []
    for line in open(docs_path, encoding="utf-8"):
        d = json.loads(line)
        for s in d["summaries"]:
            if s["system"] in exclude:
                continue
            for fi, lab in enumerate(s["labels"]):
                if lab is None:
                    continue
                cells.append((s["candidate"], d["facts"][fi], int(lab)))
    return cells


def load_synth(paths: list[str]) -> list[tuple[str, str, int]]:
    out = []
    for p in paths:
        d = pd.read_parquet(p)
        out.extend((str(r.doc), str(r.claim), int(r.label)) for r in d.itertuples())
    return out


class Fit:
    """Render the template with the premise cut to the token budget the claim and scaffold leave (b10's rule)."""

    def __init__(self, tok, max_len: int):
        self.tok, self.max_len = tok, max_len

    def __call__(self, prem: str, claim: str) -> str:
        tail = TEMPLATE.format(text_a="", text_b=claim)
        n_tail = len(self.tok(tail, add_special_tokens=False)["input_ids"])
        budget = self.max_len - n_tail - 4
        ids = self.tok(prem, add_special_tokens=False)["input_ids"]
        if budget > 0 and len(ids) > budget:
            prem = self.tok.decode(ids[:budget])
        return TEMPLATE.format(text_a=prem, text_b=claim)


def support_scores(model, tok, fit: Fit, prems: list[str], claims: list[str], batch: int, dev, max_len: int) -> np.ndarray:
    import torch
    order = sorted(range(len(prems)), key=lambda i: len(prems[i]) + len(claims[i]))
    out = np.zeros(len(prems))
    with torch.inference_mode():
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            enc = tok([fit(prems[i], claims[i]) for i in idx], max_length=max_len, truncation=True, padding=True, return_tensors="pt").to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                lg = model(**enc).logits.float()
            p = torch.softmax(lg, -1)[:, 1].cpu().numpy()
            for i, v in zip(idx, p):
                out[i] = v
    return out


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def coverage_eval(name: str, score_fn, acu_rows: list[dict], docs_val: list[dict], n_boot: int, seed: int) -> dict:
    """score_fn(prems, hyps) -> support in [0,1]. AUROC over ACU cells (label 1 = conveyed) and crossed accuracy."""
    prems, hyps, ys = [], [], []
    for r in acu_rows:
        for u, lab in zip(r["units"], r["unit_labels"]):
            if lab is None:
                continue
            prems.append(r["candidate"]); hyps.append(u["text"]); ys.append(int(lab))
    sup = score_fn(prems, hyps)
    res = {"acu_cells": len(ys), "auroc_pooled": auroc(np.array(ys), sup)}
    cells, triples = msc.crossed_cells(docs_val)
    z = dict(zip(cells, 1.0 - score_fn([docs_val[di]["summaries"][si]["candidate"] for di, si, _ in cells],
                                        [docs_val[di]["facts"][fi] for di, _, fi in cells])))
    wins, by_doc = msc.discriminate(z, triples)
    res["crossed"] = msc.summarise(wins, by_doc, np.random.default_rng(seed), n_boot)
    print(f"[mc] {name}: AUROC {res['auroc_pooled']:.4f} over {len(ys)} cells; crossed doc-macro "
          f"{res['crossed']['discrimination_doc_macro']:.4f} {res['crossed']['ci_doc_bootstrap']}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="FactCG snapshot")
    ap.add_argument("--synth", nargs="+", required=True, help="C2D and D2C parquet files")
    ap.add_argument("--cells-docs", required=True, help="e0/docs_train.jsonl")
    ap.add_argument("--eval-acu", required=True, help="acu_units_validation.jsonl")
    ap.add_argument("--eval-docs", required=True, help="e0/docs_validation.jsonl")
    ap.add_argument("--deployed-ckpts", default="", help="MARS-C checkpoint glob for the reference AUROC / crossed accuracy")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = AutoConfig.from_pretrained(a.init, num_labels=2, finetuning_task="text-classification", local_files_only=True)
    cfg.problem_type = "single_label_classification"
    tok = AutoTokenizer.from_pretrained(a.init, use_fast=True, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(a.init, config=cfg, local_files_only=True).to(dev)
    fit = Fit(tok, a.max_len)
    cells = load_cells(Path(a.cells_docs)); synth = load_synth(a.synth)
    print(f"[mc] cells {len(cells)} (conveyed rate {np.mean([c[2] for c in cells]):.3f}); synthetic {len(synth)} (supported rate {np.mean([c[2] for c in synth]):.3f})", flush=True)
    result = {"seed": a.seed, "init": a.init, "steps": a.steps, "lr": a.lr, "batch": a.batch, "max_len": a.max_len,
              "n_cells": len(cells), "n_synth": len(synth)}
    acu_rows = [json.loads(l) for l in open(a.eval_acu, encoding="utf-8")]
    docs_val = [json.loads(l) for l in open(a.eval_docs, encoding="utf-8")]

    def fn_model(prems, hyps):
        model.eval()
        return support_scores(model, tok, fit, prems, hyps, a.eval_batch, dev, a.max_len)
    result["zero_shot_factcg"] = coverage_eval("FactCG zero-shot", fn_model, acu_rows, docs_val, a.n_boot, a.seed)
    if a.deployed_ckpts:
        from mb_score import MarscWindowed
        dep = MarscWindowed(a.deployed_ckpts, 128, dev)
        result["deployed_marsc"] = coverage_eval("deployed MARS-C", lambda p, h: np.array(dep.score(p, h)), acu_rows, docs_val, a.n_boot, a.seed)
        del dep; torch.cuda.empty_cache()

    if not a.skip_train:
        model.gradient_checkpointing_enable(); model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        sch = get_linear_schedule_with_warmup(opt, a.warmup, a.steps)
        rng = random.Random(a.seed); half = a.batch // 2
        t0 = time.time(); losses = []
        for step in range(1, a.steps + 1):
            b = rng.sample(cells, half) + rng.sample(synth, a.batch - half)
            enc = tok([fit(p, c) for p, c, _ in b], max_length=a.max_len, truncation=True, padding=True, return_tensors="pt").to(dev)
            y = torch.tensor([lab for _, _, lab in b], device=dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss))
            if step % 50 == 0:
                print(f"[mc] step {step}/{a.steps} loss {np.mean(losses[-50:]):.4f} ({time.time() - t0:.0f}s)", flush=True)
        result["train_seconds"] = time.time() - t0; result["loss_last100"] = float(np.mean(losses[-100:]))
        model.gradient_checkpointing_disable()
        (out / "model").mkdir(exist_ok=True); model.save_pretrained(out / "model"); tok.save_pretrained(out / "model")
        result["unified"] = coverage_eval("unified verifier", fn_model, acu_rows, docs_val, a.n_boot, a.seed)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(f"[mc] written {out / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
