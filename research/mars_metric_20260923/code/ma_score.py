#!/usr/bin/env python3
"""M-A.2 -- one metric at a time over the meta-evaluation pairs (pairs_ma.jsonl), written as scores_<metric>.jsonl
with one row per pair. Registered under M-A in PREREG.md.

Metrics under test
  marsP_factcg     candidate facts (frozen decomposition) verified against the source by FactCG; mean and min over
                   facts; control: source replaced by another document's source (derangement within dataset)
  marsP_marsc      the same with the deployed MARS-C verifier in the precision role (source windowed, max over
                   windows, support = 1 - P(omitted), three seeds averaged)
  marsR_marsc      reference facts (frozen decomposition; RoSE: the human ACUs) verified against the candidate by
                   the MARS-C verifier (candidate windowed); mean over facts, maximum over references; controls:
                   candidate replaced by another document's candidate of the same system, and no candidate
  marsR_factcg     the same coverage reading with FactCG as the verifier
Comparators
  rouge            ROUGE-1/2/L F with stemming, maximum over references
  bertscore        BERTScore F (roberta-large), maximum over references
  bartscore        BART-large-CNN mean token log-likelihood: source->candidate and max over references->candidate
  alignscore, summac_zs   the scorer zoo, source->candidate whole
  factcg_whole     FactCG, source->candidate whole
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
sys.path.insert(0, str(REPO / "research" / "mars_metric_20260923" / "code"))
from mb_score import MarscWindowed, scored_sorted  # noqa: E402


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def load_props(patterns: list[str]) -> dict[str, list[str]]:
    by_key: dict[str, list[tuple[int, list[str]]]] = defaultdict(list)
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                sent = int(r["task_id"].rsplit(":", 1)[1])
                by_key[r["doc_key"]].append((sent, [p for p in r.get("props", []) if p and p.strip().upper() != "NONE"]))
    return {k: [p for _, ps in sorted(v) for p in ps] for k, v in by_key.items()}


def derangement(pairs: list[dict], rng: random.Random, key) -> list[int]:
    """Index of the pair whose candidate/source replaces pair i, within groups given by key(pair); a pair whose
    group has one member keeps itself (reported)."""
    perm = list(range(len(pairs)))
    groups: dict = defaultdict(list)
    for i, p in enumerate(pairs):
        groups[key(p)].append(i)
    for idx in groups.values():
        if len(idx) < 2:
            continue
        for _ in range(200):
            sh = idx[:]; rng.shuffle(sh)
            if all(x != y for x, y in zip(idx, sh)):
                break
        for x, y in zip(idx, sh):
            perm[x] = y
    return perm


def aggregate(scores: list[float], owner: list[int], n: int, fn) -> list[float | None]:
    acc: dict[int, list[float]] = defaultdict(list)
    for o, v in zip(owner, scores):
        acc[o].append(v)
    return [fn(acc[i]) if acc.get(i) else None for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--props", nargs="*", default=[])
    ap.add_argument("--metric", required=True)
    ap.add_argument("--snapshot", default="", help="FactCG snapshot")
    ap.add_argument("--ckpts", default="", help="MARS-C checkpoint glob")
    ap.add_argument("--bart", default="facebook/bart-large-cnn")
    ap.add_argument("--bertscore-model", default="roberta-large")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--shuffle-seed", type=int, default=20260917)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pairs = [json.loads(l) for l in open(a.pairs, encoding="utf-8")]
    if a.limit:
        pairs = pairs[:a.limit]
    n = len(pairs)
    props = load_props(a.props) if a.props else {}
    rows: list[dict] = [{"pair_id": p["pair_id"], "dataset": p["dataset"]} for p in pairs]
    t0 = time.time()
    m = a.metric

    if m in ("rouge",):
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        for r, p in zip(rows, pairs):
            refs = p["references"] or ([" ".join(p["acus"])] if p.get("acus") else [])
            if not refs:
                continue
            best = {k: max(sc.score(ref, p["candidate"])[k].fmeasure for ref in refs) for k in ("rouge1", "rouge2", "rougeL")}
            r.update({"rouge1": best["rouge1"], "rouge2": best["rouge2"], "rougeL": best["rougeL"]})
    elif m == "bertscore":
        import bert_score
        cands, refs, owners = [], [], []
        for i, p in enumerate(pairs):
            rs = p["references"] or ([" ".join(p["acus"])] if p.get("acus") else [])
            if rs:
                cands.append(p["candidate"]); refs.append(rs); owners.append(i)
        P, R, F = bert_score.score(cands, refs, model_type=a.bertscore_model, num_layers=17 if "/" in a.bertscore_model else None,
                                   lang="en", batch_size=a.batch, verbose=False)  # 17 = the package default for roberta-large
        for o, pp, rr, ff in zip(owners, P.tolist(), R.tolist(), F.tolist()):
            rows[o].update({"bertscore_P": pp, "bertscore_R": rr, "bertscore_F": ff})
    elif m == "bartscore":
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(a.bart); mdl = AutoModelForSeq2SeqLM.from_pretrained(a.bart).to(dev).eval()

        def ll(srcs: list[str], tgts: list[str]) -> list[float]:
            out = []
            with torch.inference_mode():
                for s in range(0, len(srcs), a.batch):
                    enc = tok(srcs[s:s + a.batch], truncation=True, max_length=1024, padding=True, return_tensors="pt").to(dev)
                    lab = tok(text_target=tgts[s:s + a.batch], truncation=True, max_length=1024, padding=True, return_tensors="pt")["input_ids"].to(dev)
                    mask = lab != tok.pad_token_id
                    labels = lab.masked_fill(~mask, -100)
                    logits = mdl(**enc, labels=labels).logits
                    lp = torch.log_softmax(logits.float(), -1).gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                    out.extend(((lp * mask).sum(-1) / mask.sum(-1).clamp(min=1)).tolist())
            return out
        rows_src = ll([p["source"] for p in pairs], [p["candidate"] for p in pairs])
        for r, v in zip(rows, rows_src):
            r["bartscore_src2cand"] = v
        rs, rt, ro = [], [], []
        for i, p in enumerate(pairs):
            for ref in (p["references"] or ([" ".join(p["acus"])] if p.get("acus") else [])):
                rs.append(ref); rt.append(p["candidate"]); ro.append(i)
        vals = ll(rs, rt) if rs else []
        agg = aggregate(vals, ro, n, max)
        for r, v in zip(rows, agg):
            if v is not None:
                r["bartscore_ref2cand"] = v
    elif m in ("alignscore", "summac_zs", "factcg_whole"):
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if m == "factcg_whole":
            swap = _load(REPO / "research" / "marsc_strengthen_20260918" / "code" / "b10_verifier_swap.py")
            scorer = swap.FactCG(a.snapshot, a.batch, dev)
        else:
            ssc = _load(REPO / "scripts" / "strengthen_score_cells.py")
            scorer = ssc.build_scorer(m, a.batch, dev)
            if m == "alignscore":
                # AlignScore pairs a ~350-word premise chunk with one claim sentence under truncation="only_first";
                # a "sentence" the splitter cannot break (over ~400 words) cannot be fitted and raises. Cap each
                # claim piece at 200 words (the zoo class is untouched); the count of capped pieces is logged.
                orig_sent, capped = scorer.sent, {"n": 0}

                def capped_sent(t, cap=200):
                    out = []
                    for h in (orig_sent(t) or [""]):
                        ids = scorer.tok(h, add_special_tokens=False)["input_ids"]
                        if len(ids) <= cap:
                            out.append(h)
                        else:  # token-dense pieces (URLs, unspaced strings) are what the word count misses
                            capped["n"] += 1
                            out.extend(scorer.tok.decode(ids[i:i + cap]) for i in range(0, len(ids), cap))
                    return out
                scorer.sent = capped_sent
        vals = scored_sorted(scorer, [p["source"] for p in pairs], [p["candidate"] for p in pairs])
        for r, v in zip(rows, vals):
            r[m] = v
        if m == "alignscore":
            print(f"[ma] alignscore: {capped['n']} over-long claim pieces capped at 200 tokens", flush=True)
    elif m in ("marsP_factcg", "marsP_marsc", "marsR_marsc", "marsR_factcg"):
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if m.endswith("factcg"):
            swap = _load(REPO / "research" / "marsc_strengthen_20260918" / "code" / "b10_verifier_swap.py")
            scorer = swap.FactCG(a.snapshot, a.batch, dev)
        else:
            scorer = MarscWindowed(a.ckpts, a.batch, dev)
        rng = random.Random(a.shuffle_seed)
        if m.startswith("marsP"):
            prem, hyp, own = [], [], []
            for i, p in enumerate(pairs):
                fs = props.get(p["cand_key"], []) or [p["candidate"]]
                rows[i]["n_facts"] = len(props.get(p["cand_key"], [])); rows[i]["fallback"] = int(not props.get(p["cand_key"]))
                for f in fs:
                    prem.append(p["source"]); hyp.append(f); own.append(i)
            vals = scored_sorted(scorer, prem, hyp)
            for r, mean_v, min_v in zip(rows, aggregate(vals, own, n, np.mean), aggregate(vals, own, n, min)):
                r[m] = float(mean_v); r[m + "_min"] = float(min_v)
            perm = derangement(pairs, rng, key=lambda p: p["dataset"])
            vals = scored_sorted(scorer, [pairs[perm[o]]["source"] for o in own], hyp)
            for r, v in zip(rows, aggregate(vals, own, n, np.mean)):
                r[m + "_shufsrc"] = float(v)
        else:
            prem, hyp, own_pair, own_ref = [], [], [], []
            for i, p in enumerate(pairs):
                ref_facts = [p["acus"]] if p.get("acus") else ([props.get(k, []) for k in p["ref_keys"]] if p["ref_keys"] else [])
                ref_facts = [fs for fs in ref_facts if fs]
                rows[i]["n_refs"] = len(ref_facts)
                for k, fs in enumerate(ref_facts):
                    for f in fs:
                        prem.append(p["candidate"]); hyp.append(f); own_pair.append(i); own_ref.append(k)
            def per_ref_max(vals):
                acc: dict[tuple[int, int], list[float]] = defaultdict(list)
                for i, k, v in zip(own_pair, own_ref, vals):
                    acc[(i, k)].append(v)
                best: dict[int, float] = {}
                for (i, k), vs in acc.items():
                    best[i] = max(best.get(i, -1.0), float(np.mean(vs)))
                return [best.get(i) for i in range(n)]
            vals = scored_sorted(scorer, prem, hyp)
            for r, v in zip(rows, per_ref_max(vals)):
                if v is not None:
                    r[m] = v
            perm = derangement(pairs, rng, key=lambda p: (p["dataset"], p["system"]))
            vals = scored_sorted(scorer, [pairs[perm[i]]["candidate"] for i in own_pair], hyp)
            for r, v in zip(rows, per_ref_max(vals)):
                if v is not None:
                    r[m + "_shufcand"] = v
            vals = scored_sorted(scorer, ["" for _ in own_pair], hyp)
            for r, v in zip(rows, per_ref_max(vals)):
                if v is not None:
                    r[m + "_nocand"] = v
    else:
        raise SystemExit(f"[ma] unknown metric {m!r}")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    scored = sum(1 for r in rows if len(r) > 2)
    print(f"[ma] {m}: {scored}/{n} pairs scored in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
