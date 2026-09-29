#!/usr/bin/env python3
"""MARS-C E1 -- objective isolation on naturally crossed human fact coverage (docs/FINAL_PROPOSAL_2026-09-16.md).

Arms (identical student, human labels, update count; A-D see the SAME sampled cells in the same order, seed-locked):
  R  random-cell BCE (natural cells only, matched cell count)      -- tests the tetrad SAMPLER (R vs A)
  A  BCE with the tetrad sampler                                    -- ordinary control
  B  A + same-fact pairwise ranking   0.5*mean(sp(-r1), sp(-r2))
  C  A + generic row/column contrast  0.5*mean(sp(-r1), sp(-r2), sp(-c1), sp(-c2))
  D  A + conditional coverage loss    0.25*sp(-(r1+r2))            -- the proposed objective
with omission logits z (fact x summary), labels [[0,1],[1,0]], r1=z12-z11, r2=z21-z22, c1=z21-z11, c2=z12-z22.
Representations: `tagger` (ModernBERT-large token tagger, fact = pooled aligned source tokens; the M32b student) or
`crossenc` (DeBERTa-large-MNLI on (summary, fact) pairs; the registered alternative). Calibration: one logistic
slope/intercept fitted on the calibration documents at natural prevalence, identical for every arm.
Outputs: calibrated unit_scores + raw logits on the validation ACUs (gates format), result.json with cost accounting.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars3_20260916" / "code")); sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import m3common as m3  # noqa: E402
from m3common import gates  # noqa: E402
from m31_tagger import Tagger  # noqa: E402

sp = torch.nn.functional.softplus


def load_role(data_dir: Path, role: str, resources: set[str] | None) -> list[dict]:
    docs = gates.load_jsonl(data_dir / f"docs_{role}.jsonl")
    return [d for d in docs if not resources or d["resource"] in resources]


class Sampler:
    """Seed-locked draws. Natural cell: document -> non-reference summary -> k labelled facts. Tetrad: document with
    crossed coverage -> eligible summary pair -> one fact from each opposing set; collided pairs are skipped."""

    def __init__(self, docs: list[dict], seed: int, excl=("gold",)):
        self.rng = random.Random(seed)
        self.docs = [d for d in docs if any(s["system"] not in excl for s in d["summaries"])]
        self.excl = set(excl)
        self.tet_docs = []
        for d in self.docs:
            coll = {tuple(p) for p in d.get("collided_pairs", [])}
            groups = {}
            for f, g, i, j in d.get("tetrads", []):
                if (f, g) in coll or (g, f) in coll:
                    continue
                groups.setdefault((i, j), ([], []))
                X, Y = groups[(i, j)]
                if f not in X: X.append(f)
                if g not in Y: Y.append(g)
            d["_groups"] = [(k, v) for k, v in groups.items() if v[0] and v[1]]
            if d["_groups"]:
                self.tet_docs.append(d)

    def natural(self, k: int):
        d = self.rng.choice(self.docs)
        S = [s for s in d["summaries"] if s["system"] not in self.excl]; s = self.rng.choice(S)
        facts = [f for f, lab in enumerate(s["labels"]) if lab is not None]
        picks = self.rng.sample(facts, min(k, len(facts)))
        return d, s, [(f, 1 - int(s["labels"][f])) for f in picks]        # label 1 = omitted

    def tetrad(self):
        d = self.rng.choice(self.tet_docs)
        (i, j), (X, Y) = self.rng.choice(d["_groups"])
        return d, d["summaries"][i], d["summaries"][j], self.rng.choice(X), self.rng.choice(Y)


class CrossEnc(torch.nn.Module):
    def __init__(self, backbone: str):
        super().__init__()
        from transformers import AutoModelForSequenceClassification
        self.m = AutoModelForSequenceClassification.from_pretrained(backbone, num_labels=1, ignore_mismatched_sizes=True)

    def forward(self, **enc):
        return self.m(**enc).logits.squeeze(-1)


def span_cache(d: dict) -> list[tuple[int, int]]:
    if "_spans" not in d:
        d["_spans"] = [gates.best_sentence_span(d["source"], f) for f in d["facts"]]
    return d["_spans"]


def forward_cells(model, tok, repr_, encs, dev, max_len):
    """encs: list of (source, candidate, doc, [fact_idx...]) -> list of logit tensors (one per enc, len = n facts), token count."""
    if repr_ == "tagger":
        enc = tok([c for _, c, _, _ in encs], [s for s, _, _, _ in encs], truncation="only_second", max_length=max_len, padding=True, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping").tolist(); is_src = [[sid == 1 for sid in enc.sequence_ids(i)] for i in range(len(encs))]
        ntok = int(enc["attention_mask"].sum())
        logits = model(**{k: v.to(dev) for k, v in enc.items()})
        out = []
        for i, (src, _, d, facts) in enumerate(encs):
            spans = span_cache(d); per = []
            for f in facts:
                m = m3.prop_token_mask(src, d["facts"][f], spans[f][0], spans[f][1], offs[i], is_src[i])
                idx = np.flatnonzero(m)
                per.append(logits[i, torch.as_tensor(idx, device=dev)].mean() if len(idx) else logits[i, is_src[i].index(True) if True in is_src[i] else 0] * 0.0)
            out.append(torch.stack(per) if per else torch.zeros(0, device=dev))
        return out, ntok
    pairs = [(c, d["facts"][f]) for _, c, d, facts in encs for f in facts]
    if not pairs:
        return [torch.zeros(0, device=dev) for _ in encs], 0
    enc = tok([" ".join(c.split()[:380]) for c, _ in pairs], [f for _, f in pairs], truncation="longest_first", max_length=max_len, padding=True, return_tensors="pt")
    ntok = int(enc["attention_mask"].sum()); z = model(**{k: v.to(dev) for k, v in enc.items()})
    out, k = [], 0
    for _, _, _, facts in encs:
        out.append(z[k:k + len(facts)]); k += len(facts)
    return out, ntok


def aux_loss(arm: str, z11, z12, z21, z22):
    r1, r2, c1, c2 = z12 - z11, z21 - z22, z21 - z11, z12 - z22
    if arm == "B":
        return 0.5 * torch.stack([sp(-r1), sp(-r2)]).mean()
    if arm == "C":
        return 0.5 * torch.stack([sp(-r1), sp(-r2), sp(-c1), sp(-c2)]).mean()
    if arm == "D":
        return 0.25 * sp(-(r1 + r2))
    return torch.zeros((), device=z11.device)


def strict_four(z11, z12, z21, z22) -> bool:
    return bool(z12 > z11 and z21 > z22 and z21 > z11 and z12 > z22)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list("RABCD"), required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--repr", choices=["tagger", "crossenc"], default="tagger")
    ap.add_argument("--backbone", default="")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--resources", default="", help="comma list; empty = all (screen uses one resource)")
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--tetrads-per-step", type=int, default=2)
    ap.add_argument("--natural-per-step", type=int, default=4)
    ap.add_argument("--facts-per-natural", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--max-len", type=int, default=0)
    ap.add_argument("--eval-acu", default=str(gates.OUT_ROOT / "b18" / "acu_units_validation.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--scores-out", default="")
    ap.add_argument("--eval-units", nargs="*", default=[], help="name=path extra units files (pairs with units), scored as supplied facts -> <name>_<tag>_seed.jsonl")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="gradient checkpointing on the encoder (memory only; the published runs did not use it)")
    ap.add_argument("--eval-pairs", type=int, default=8,
                    help="pairs per forward chunk in the post-training scoring (batching only; 8 = the published runs; "
                         "a 0.9B encoder needs 2 on a 140 GB device)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--train-limit", type=int, default=0, help="truncate the TRAIN/calib documents only and leave the evaluation set at full size "
                                                              "(--limit truncates both, which confounds a label-efficiency curve with a shrinking eval set)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    backbone = m3.snap(a.backbone or ("answerdotai/ModernBERT-large" if a.repr == "tagger" else "microsoft/deberta-large-mnli"))
    lr = a.lr or (3e-5 if a.repr == "tagger" else 2e-5); max_len = a.max_len or (3072 if a.repr == "tagger" else 512)
    tag = a.tag or f"mc-{a.arm}-lam{a.lam:g}-{a.repr}"
    data = Path(a.data_dir); res = set(a.resources.split(",")) if a.resources else None
    train_docs = load_role(data, "train", res); calib_docs = load_role(data, "calib", res)
    _tl = a.train_limit or a.limit
    if _tl:
        train_docs, calib_docs = train_docs[:_tl], calib_docs[:max(2, _tl // 4)]
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained(backbone)
    model = (Tagger(backbone) if a.repr == "tagger" else CrossEnc(backbone)).to(dev)
    if a.grad_ckpt:
        inner = getattr(model, "m", None)
        assert inner is not None and hasattr(inner, "gradient_checkpointing_enable"), "--grad-ckpt needs the cross-encoder"
        inner.gradient_checkpointing_enable()
        print("[mc] gradient checkpointing ON (activations recomputed in the backward pass; same gradient)", flush=True)
    amp = dev.startswith("cuda") and a.repr == "tagger"        # DeBERTa-v1 overflows under bf16 autocast
    torch.backends.cuda.matmul.allow_tf32 = True
    S = Sampler(train_docs, a.seed)
    print(f"[mc:{tag}] repr={a.repr} backbone={backbone} train docs={len(S.docs)} tetrad docs={len(S.tet_docs)} steps={a.steps} device={dev}", flush=True)
    if not S.tet_docs and a.arm != "R":
        raise SystemExit("no tetrad documents in the training role for these resources")
    T, N, K = a.tetrads_per_step, a.natural_per_step, a.facts_per_natural
    kR = max(1, round((4 * T + N * K) / (2 * T + N)))            # R matches the cell count of A with natural cells only
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01); sch = get_linear_schedule_with_warmup(opt, int(0.06 * a.steps), a.steps)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    model.train(); t0 = time.time(); log = {"bce": [], "aux": [], "gnorm": [], "four": []}; cost = {"encodings": 0, "tokens": 0, "cells": 0}
    for step in range(1, a.steps + 1):
        encs, cells, tets = [], [], []             # cells: (enc_idx, pos, label); tets: (enc_i, enc_j, pos_f_i, pos_g_i, pos_f_j, pos_g_j)
        if a.arm == "R":
            for _ in range(2 * T + N):
                d, s, cs = S.natural(kR); encs.append((d["source"], s["candidate"], d, [f for f, _ in cs])); cells += [(len(encs) - 1, p, lab) for p, (_, lab) in enumerate(cs)]
        else:
            for _ in range(T):
                d, si, sj, f, g = S.tetrad()
                ei = len(encs); encs.append((d["source"], si["candidate"], d, [f, g])); ej = len(encs); encs.append((d["source"], sj["candidate"], d, [f, g]))
                cells += [(ei, 0, 0), (ei, 1, 1), (ej, 0, 1), (ej, 1, 0)]; tets.append((ei, ej))
            for _ in range(N):
                d, s, cs = S.natural(K); encs.append((d["source"], s["candidate"], d, [f for f, _ in cs])); cells += [(len(encs) - 1, p, lab) for p, (_, lab) in enumerate(cs)]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            Z, ntok = forward_cells(model, tok, a.repr, encs, dev, max_len)
            zc = torch.stack([Z[e][p] for e, p, _ in cells if p < len(Z[e])]).float(); yc = torch.tensor([lab for e, p, lab in cells if p < len(Z[e])], device=dev, dtype=torch.float32)
            loss_bce = bce(zc, yc); loss_aux = torch.zeros((), device=dev); four = []
            if tets and a.arm in "BCD":
                loss_aux = torch.stack([aux_loss(a.arm, Z[ei][0].float(), Z[ej][0].float(), Z[ei][1].float(), Z[ej][1].float()) for ei, ej in tets]).mean()
            for ei, ej in tets:
                four.append(strict_four(Z[ei][0], Z[ej][0], Z[ei][1], Z[ej][1]))
            loss = loss_bce + a.lam * loss_aux
        loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        cost["encodings"] += len(encs); cost["tokens"] += ntok; cost["cells"] += len(cells)
        log["bce"].append(float(loss_bce)); log["aux"].append(float(loss_aux)); log["gnorm"].append(gn); log["four"] += four
        if step % 25 == 0:
            print(f"[mc:{tag}] step {step}/{a.steps} bce {np.mean(log['bce'][-25:]):.4f} aux {np.mean(log['aux'][-25:]):.4f} gnorm {np.mean(log['gnorm'][-25:]):.2f} four-cell(train) {np.mean(log['four'][-50:]) if log['four'] else float('nan'):.2f} ({time.time() - t0:.0f}s)", flush=True)
    train_sec = time.time() - t0; model.eval()

    @torch.inference_mode()
    def score_pairs(rows, facts_key="units", text_key="text"):
        """rows: pairs with a fact list -> raw logits per fact (facts scored by their aligned tokens / as claims)."""
        out = {}
        for lo in range(0, len(rows), a.eval_pairs):
            ch = rows[lo:lo + a.eval_pairs]
            encs = []
            for r in ch:
                d = {"source": r["source"], "facts": [u[text_key] for u in r[facts_key]]}
                encs.append((r["source"], r["candidate"], d, list(range(len(d["facts"])))))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                Z, _ = forward_cells(model, tok, a.repr, encs, dev, max_len)
            for r, z in zip(ch, Z):
                out[r["pair_id"]] = z.float().cpu().numpy().tolist()
        return out

    # calibration cells: every labelled fact of every non-reference summary of the calibration documents
    cal_rows = [{"pair_id": s["pair_id"], "source": d["source"], "candidate": s["candidate"], "units": [{"text": d["facts"][f], "lab": s["labels"][f]} for f in range(len(d["facts"])) if s["labels"][f] is not None]}
                for d in calib_docs for s in d["summaries"] if s["system"] != "gold"]
    cal_rows = [r for r in cal_rows if r["units"]]
    zc = score_pairs(cal_rows); zs, ys = [], []
    for r in cal_rows:
        for u, z in zip(r["units"], zc[r["pair_id"]]):
            zs.append(z); ys.append(1 - int(u["lab"]))
    slope, intercept = 1.0, 0.0
    if len(set(ys)) == 2:
        from sklearn.linear_model import LogisticRegression
        lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(np.array(zs).reshape(-1, 1), np.array(ys)); slope, intercept = float(lr_.coef_[0][0]), float(lr_.intercept_[0])
    calib = {"n_cells": len(ys), "omitted_rate": float(np.mean(ys)) if ys else None, "slope": slope, "intercept": intercept}
    # validation ACUs
    acu = gates.load_jsonl(Path(a.eval_acu))
    if a.limit:
        acu = acu[:max(4, a.limit)]
    t1 = time.time(); zv = score_pairs(acu); eval_sec = (time.time() - t1) / max(1, len(acu))
    sysname = f"mc:{a.arm}|lam{a.lam:g}|{a.repr}"; so = Path(a.scores_out or (Path(a.out).parent.parent / "scores")); so.mkdir(parents=True, exist_ok=True)
    gates.write_scores(so / f"acu_{tag}_seed{a.seed}.jsonl", sysname, "acu", {p: [1 / (1 + math.exp(-(slope * z + intercept))) for z in zz] for p, zz in zv.items()})
    gates.write_scores(so / f"acu_{tag}_seed{a.seed}_logits.jsonl", sysname + "|logit", "acu", zv)
    for spec in a.eval_units:
        name, path = spec.split("=", 1)
        ur = [r for r in gates.load_jsonl(Path(path)) if r.get("units")]
        if a.limit:
            ur = ur[:max(4, a.limit)]
        zu = score_pairs(ur)
        gates.write_scores(so / f"{name}_{tag}_seed{a.seed}.jsonl", sysname, name, {p: [1 / (1 + math.exp(-(slope * z + intercept))) for z in zz] for p, zz in zu.items()})
        print(f"[mc:{tag}] scored inventory {name}: {len(ur)} pairs", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.save_model:
        if a.repr == "tagger":
            model.enc.save_pretrained(out / "backbone"); tok.save_pretrained(out / "backbone"); torch.save(model.head.state_dict(), out / "head.pt")
        else:
            model.m.save_pretrained(out / "model"); tok.save_pretrained(out / "model")
    json.dump({"tag": tag, "system": sysname, "config": vars(a), "backbone": backbone, "lr": lr, "max_len": max_len, "train_docs": len(S.docs), "tetrad_docs": len(S.tet_docs),
               "cells_per_step_R": kR if a.arm == "R" else None, "train_seconds": train_sec, "cost": cost, "train_bce_last50": float(np.mean(log["bce"][-50:])),
               "train_aux_last50": float(np.mean(log["aux"][-50:])), "train_four_cell_last200": float(np.mean(log["four"][-200:])) if log["four"] else None,
               "grad_norm_mean": float(np.mean(log["gnorm"])), "calibration": calib, "eval_seconds_per_pair": eval_sec}, open(out / "result.json", "w"), indent=1)
    print(f"[mc:{tag}] done: train {train_sec:.0f}s, {cost['encodings']} encodings, {cost['tokens']} tokens; calib slope {slope:.3f} icpt {intercept:.3f}; eval {eval_sec:.3f} s/pair", flush=True)


if __name__ == "__main__":
    main()
