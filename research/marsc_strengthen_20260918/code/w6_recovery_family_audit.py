#!/usr/bin/env python3
"""W6 -- the recovery family audits itself: Shannon Score and BLANC-help on SummEval under the counter suite.

Registered in MARS_WIN_HYPOTHESES_2026-09-09.md Sec. W6, and pulled forward by
MARS_PUBLICATION_PLAN_2026-09-20.md Sec. 3.2 (Paper A, "< 1 GPU-h"). MARS belongs to the *recovery* family:
metrics that score a summary by how well it helps reconstruct the source. Shannon Score (Egan et al., AAAI
2022) and BLANC-help (Vasilyev et al., 2020) are that family's published, human-validated members. The audit
paper's whole argument is that published validations of this kind can be reproduced by summary-blind counters;
it would be a double standard to apply that only to our own scorer. So the suite is turned on the family MARS
belongs to.

REGISTERED PREDICTION (W6 Sec. 2, quoted): length + entity count reproduce >= 1/2 of Shannon's summary-level
correlation with relevance, and the increment over length is small. KILL: the increment is large, i.e. the
family carries semantic signal beyond rate -- which would be good for MARS-2 and is reported either way.

ENDPOINTS (W6 Sec. 5)
---------------------
  * summary-level Kendall tau, per human axis: within a document, rank the 16 system summaries by the metric
    and by the human axis, take tau, average over documents. This is the SummEval summary-level protocol.
  * system-level Kendall tau: 16 points, metric system-mean vs human system-mean.
  * conditional increment: nested group-aware CV, groups = documents, target = the human axis. Model A is the
    counters alone, model B counters + metric, model M the metric alone. Reported as held-out R^2 and as the
    held-out Spearman of the predictions, with a paired bootstrap over DOCUMENTS (not rows), matching
    scripts/conditional_value.py. That file itself cannot be reused: it is a binary-label instrument
    (logistic regression, log loss, AUROC) and SummEval's axes are Likert means, so the same design is
    re-expressed for a continuous target with ridge and R^2.
  * reproduction fraction, reported TWO ways because the obvious one is not apples-to-apples:
      `reproduction_fraction`          tau(counter-model OOF prediction) / tau(RAW metric). This is how W6
                                       Sec. 2 phrases the registered prediction, and it is the number that
                                       prediction is judged on -- but it flatters the counters, because the
                                       counter side is a ridge fitted to this human axis while the metric
                                       side is a fixed function that never sees the axis.
      `reproduction_fraction_matched`  tau(counter-model OOF) / tau(metric-model OOF), both fitted under the
                                       identical nested CV. This is the fair comparison. Lead with it if the
                                       two disagree: reporting only the flattering one would be the same
                                       defect this paper audits other people for.

SHANNON SCORE, AND WHY IT IS NOT THE PACKAGED LOOP
---------------------------------------------------
Shannon Score = (ll_help - ll_base) / (ll_full - ll_base). `blanc.shannon.Shannon.go` computes the three
terms PER SENTENCE with `num_upstream = 0`: the document is split with nltk, and each sentence is scored
three times -- given nothing, given the summary, and given itself -- with no preceding-document context in
any of them. The per-token log-likelihoods are summed over the whole document and the ratio is taken once.
`ShannonScorer.score` reproduces exactly that decomposition.

What is NOT reproduced is the package's per-token Python loop, which is written against transformers 3
(`logits, past = out`) -- a shape transformers 5 does not return -- and costs one forward pass per token.
For a CAUSAL language model the teacher-forced single pass gives identical per-token log-probabilities,
because GPT-2 attends only leftwards; so the arithmetic is batched and the value is unchanged.

A REJECTED EARLIER VERSION, recorded because its output was briefly reported: scoring the document as one
continuous sequence, LL(D|nothing) / LL(D|S) / LL(D|D). That is a DIFFERENT METRIC, not a batched form of
this one. It lets each sentence condition on the document text preceding it, which the package explicitly
does not at num_upstream=0, and its cheating term conditions the whole document on the whole document
instead of each sentence on itself. Numbers from it are not Shannon Score and were withdrawn.

BLANC-help is the official package, unmodified.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402

AXES = ["relevance", "coherence", "fluency", "consistency"]


# ----------------------------------------------------------------------------- blanc 0.3 / transformers 5
def import_blanc_help():
    """blanc/__init__.py imports Shannon, which imports model classes transformers 5 removed (TransfoXL,
    OpenAIGPT, XLM, Reformer) and an optimiser HF moved back to torch. None are on the BlancHelp path, and
    setattr does not satisfy `from transformers import X` on a lazy module, so a stub module is registered
    under the name instead. Shannon is computed in this file, not taken from that module."""
    import torch
    import transformers
    if not hasattr(transformers, "AdamW"):
        transformers.AdamW = torch.optim.AdamW
    stub = types.ModuleType("blanc.shannon")
    stub.Shannon = type("Shannon", (), {})          # never instantiated; presence is all __init__ needs
    sys.modules.setdefault("blanc.shannon", stub)
    from blanc import BlancHelp
    return BlancHelp


# ----------------------------------------------------------------------------- Shannon Score
class ShannonScorer:
    def __init__(self, snapshot: str, device: str, doc_tokens: int, batch: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True).eval().to(device)
        self.device, self.doc_tokens, self.batch = device, doc_tokens, batch
        self.bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
        self.max_ctx = getattr(self.model.config, "n_positions", 1024)

    def _ll(self, ctxs: list[list[int]], docs: list[list[int]]) -> np.ndarray:
        """Sum log p(doc tokens | [BOS] + ctx), teacher forced, one pass per row."""
        torch = self.torch
        out = np.zeros(len(docs), dtype=float)
        for s in range(0, len(docs), self.batch):
            cs, ds = ctxs[s:s + self.batch], docs[s:s + self.batch]
            seqs = [[self.bos] + c + d for c, d in zip(cs, ds)]
            n_ctx = [1 + len(c) for c in cs]
            width = max(len(x) for x in seqs)
            ids = torch.full((len(seqs), width), self.tok.eos_token_id, dtype=torch.long)
            att = torch.zeros((len(seqs), width), dtype=torch.long)
            for i, x in enumerate(seqs):
                ids[i, :len(x)] = torch.tensor(x)
                att[i, :len(x)] = 1
            ids, att = ids.to(self.device), att.to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids=ids, attention_mask=att).logits.float()
            logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
            tgt = ids[:, 1:]
            tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)      # log p of each next token
            for i, x in enumerate(seqs):
                # doc tokens occupy positions n_ctx[i] .. len(x)-1, predicted at shifted index -1
                lo, hi = n_ctx[i] - 1, len(x) - 1
                out[s + i] = tok_lp[i, lo:hi].sum().item()
        return out

    def score(self, sources: list[str], summaries: list[str]) -> dict[str, np.ndarray]:
        """Shannon Score under the PUBLISHED per-sentence contract (blanc.shannon.Shannon.go).

        The document is split into sentences and each sentence is scored INDEPENDENTLY under three prompts,
        with `num_upstream = 0` (the package default), i.e. no preceding-document context:

            base  measure(sent, [])            help  measure(sent, summ)        full  measure(sent, sent)

        and the score is (sum ll_help - sum ll_base) / (sum ll_full - sum ll_base) over all of a document's
        sentence tokens. `measure` prepends the model's eos to the prompt, which `_ll` reproduces.

        An earlier version of this file scored the document as ONE continuous sequence. That is a different
        metric, not a batched form of this one: it lets every sentence condition on the document text before
        it (the package explicitly does not, at num_upstream=0), and its "cheating" term conditions the whole
        document on the whole document rather than each sentence on itself. It is not the published quantity
        and nothing computed from it may be labelled Shannon Score.
        """
        from nltk import sent_tokenize
        ctx_base, ctx_help, ctx_full, tgt, owner = [], [], [], [], []
        n_sent_capped = 0
        for k, (src, summ) in enumerate(zip(sources, summaries)):
            s_tok = self.tok.encode(summ)
            # the package's own cap for the help condition ...
            cap = self.max_ctx - 1 - len(s_tok)
            # ... plus one the package omits: the full condition is [eos] + sent + sent, so a sentence longer
            # than (max_ctx - 1) / 2 would overflow the window. Recorded rather than silently truncated.
            cap = min(cap, (self.max_ctx - 1) // 2)
            for sent in sent_tokenize(src):
                st = self.tok.encode(sent)
                if len(st) > cap:
                    st, n_sent_capped = st[:cap], n_sent_capped + 1
                if not st:
                    continue
                tgt.append(st)
                owner.append(k)
                ctx_base.append([])
                ctx_help.append(s_tok)
                ctx_full.append(st)
        if n_sent_capped:
            print(f"[w6] {n_sent_capped} sentences capped to the GPT-2 window", flush=True)

        ll_b = self._ll(ctx_base, tgt)
        ll_h = self._ll(ctx_help, tgt)
        ll_f = self._ll(ctx_full, tgt)

        n = len(sources)
        agg = {name: np.zeros(n) for name in ("base", "help", "full")}
        for i, k in enumerate(owner):
            agg["base"][k] += ll_b[i]
            agg["help"][k] += ll_h[i]
            agg["full"][k] += ll_f[i]
        denom = agg["full"] - agg["base"]
        ok = np.abs(denom) > 1e-9
        score = np.where(ok, (agg["help"] - agg["base"]) / np.where(ok, denom, 1.0), np.nan)
        return {"shannon_score": score, "ll_null": agg["base"], "ll_summ": agg["help"],
                "ll_doc": agg["full"], "n_sentences": len(tgt)}


# ----------------------------------------------------------------------------- counters (W6 Sec. 4)
_PUNCT = ".,;:!?\"'()[]{}"


def _norm(tokens) -> list[str]:
    """Lowercase and strip edge punctuation. Applied to BOTH sides of the novelty counter: stripping only
    the summary side makes any summary word whose source occurrence happens to be punctuation-adjacent
    ("Sterling." in the source, "Sterling" in the summary) count as novel, which inflates the counter
    systematically rather than randomly."""
    return [t for t in (x.lower().strip(_PUNCT) for x in tokens) if t]


def counters(source: str, summary: str, nlp) -> dict[str, float]:
    sw, src_w = summary.split(), source.lower().split()
    src_set = set(_norm(src_w))
    novel = [w for w in _norm(sw) if w not in src_set]
    ents = len(nlp(summary).ents) if nlp is not None else float("nan")
    return {
        "cnt_words": float(len(sw)),
        "cnt_chars": float(len(summary)),
        "cnt_sents": float(len(common.sentences(summary)) or 1),
        "cnt_entities": float(ents),
        "cnt_novelty": float(len(novel)) / max(1, len(sw)),
        "cnt_rougeL_recall": float(common.rougeL_recall(summary, source)),
        "cnt_token_recall": float(common.token_recall(summary, source)),
        "cnt_compression": float(len(sw)) / max(1, len(src_w)),
    }


# ----------------------------------------------------------------------------- endpoints
def kendall(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import kendalltau
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or len(set(x[ok].tolist())) < 2 or len(set(y[ok].tolist())) < 2:
        return np.nan
    return float(kendalltau(x[ok], y[ok]).statistic)


def summary_level_tau(vals: np.ndarray, human: np.ndarray, doc_ids: np.ndarray) -> float:
    """Within each document, tau across the systems; averaged over documents (SummEval protocol)."""
    taus = [kendall(vals[doc_ids == d], human[doc_ids == d]) for d in np.unique(doc_ids)]
    taus = [t for t in taus if np.isfinite(t)]
    return float(np.mean(taus)) if taus else np.nan


def system_level_tau(vals: np.ndarray, human: np.ndarray, sys_ids: np.ndarray) -> float:
    us = np.unique(sys_ids)
    mv = np.array([np.nanmean(vals[sys_ids == s]) for s in us])
    mh = np.array([np.nanmean(human[sys_ids == s]) for s in us])
    return kendall(mv, mh)


def nested_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    """Out-of-fold predictions from a ridge with an inner group-aware alpha search."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    oof = np.full(len(y), np.nan)
    outer = GroupKFold(n_splits=5)
    for tr, te in outer.split(X, y, groups):
        best, best_mse = alphas[0], np.inf
        inner = GroupKFold(n_splits=5)
        for a in alphas:
            errs = []
            for itr, ite in inner.split(X[tr], y[tr], groups[tr]):
                m = make_pipeline(StandardScaler(), Ridge(alpha=a, random_state=seed))
                m.fit(X[tr][itr], y[tr][itr])
                errs.append(float(np.mean((m.predict(X[tr][ite]) - y[tr][ite]) ** 2)))
            if np.mean(errs) < best_mse:
                best, best_mse = a, float(np.mean(errs))
        m = make_pipeline(StandardScaler(), Ridge(alpha=best, random_state=seed))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def r2(y: np.ndarray, p: np.ndarray) -> float:
    ss = float(np.sum((y - p) ** 2))
    tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss / tot if tot > 0 else np.nan


def boot_ci(stat_fn, groups: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """Cluster bootstrap over `groups`; `stat_fn(idx, gid)` gets the resampled row indices AND a synthetic
    cluster label per row in which EVERY DRAW is its own cluster.

    The synthetic label is the whole point. A statistic that groups internally -- `summary_level_tau` does,
    by document -- must not be handed the ORIGINAL document ids, because a document drawn twice then has its
    rows concatenated under one id and `np.unique` collapses the two draws into a single cluster of double
    size. That is not a cluster bootstrap. Measured on this design over 2000 replicates it shrinks the tau
    interval by ~24 % (width 0.052 against the correct 0.069) while leaving the point estimate unbiased, so
    it silently reports intervals that are too confident. Statistics that do not group internally (the R^2
    increment) are unaffected either way and simply ignore `gid`.
    """
    vals = boot_vals(stat_fn, groups, n_boot, seed)
    if not len(vals):
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def boot_vals(stat_fn, groups: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """The replicate values themselves, so a caller can take both an interval and a one-sided p."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        parts = [idx_by_g[g] for g in pick]
        idx = np.concatenate(parts)
        gid = np.concatenate([np.full(len(p), k, dtype=int) for k, p in enumerate(parts)])
        v = stat_fn(idx, gid)
        if np.isfinite(v):
            vals.append(v)
    return np.asarray(vals, dtype=float)


def bh(pvals: list[float]) -> list[bool]:
    m = len(pvals)
    order = np.argsort(pvals)
    keep = np.zeros(m, dtype=bool)
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= 0.05 * rank / m:
            keep[order[:rank]] = True
    return keep.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpt2-snapshot", required=True)
    ap.add_argument("--doc-tokens", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--blanc-batch", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--limit-docs", type=int, default=0, help="pilot only")
    ap.add_argument("--skip-blanc", action="store_true")
    a = ap.parse_args()

    import torch
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset("mteb/summeval", split="test")
    if a.limit_docs:
        ds = ds.select(range(min(a.limit_docs, len(ds))))
    rows = []
    for di, r in enumerate(ds):
        for si, summ in enumerate(r["machine_summaries"]):
            row = {"doc_id": di, "sys_id": si, "source": r["text"], "summary": summ}
            for ax in AXES:
                row[ax] = float(r[ax][si])
            rows.append(row)
    print(f"[w6] SummEval: {len(ds)} documents x 16 systems = {len(rows)} summaries", flush=True)

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception as exc:                                  # entity count is one counter of eight
        print(f"[w6] WARNING spaCy unavailable ({exc}); cnt_entities will be dropped", flush=True)
        nlp = None

    t0 = time.time()
    cnt = [counters(r["source"], r["summary"], nlp) for r in rows]
    cnames = [k for k in cnt[0] if np.isfinite(cnt[0][k])]
    print(f"[w6] counters {cnames} in {time.time() - t0:.0f}s", flush=True)

    metrics: dict[str, np.ndarray] = {}
    t0 = time.time()
    sh = ShannonScorer(a.gpt2_snapshot, dev, a.doc_tokens, a.batch)
    res = sh.score([r["source"] for r in rows], [r["summary"] for r in rows])
    metrics["shannon_score"] = res["shannon_score"]
    print(f"[w6] shannon in {time.time() - t0:.0f}s; mean {np.nanmean(res['shannon_score']):.4f}", flush=True)
    del sh
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()

    if not a.skip_blanc:
        t0 = time.time()
        BlancHelp = import_blanc_help()
        bl = BlancHelp(device=dev, inference_batch_size=a.blanc_batch)
        vals = bl.eval_pairs([r["source"] for r in rows], [r["summary"] for r in rows])
        metrics["blanc_help"] = np.array(vals, dtype=float)
        print(f"[w6] blanc_help in {time.time() - t0:.0f}s; mean {np.nanmean(metrics['blanc_help']):.4f}",
              flush=True)

    doc_ids = np.array([r["doc_id"] for r in rows])
    sys_ids = np.array([r["sys_id"] for r in rows])
    Xc = np.array([[c[k] for k in cnames] for c in cnt], dtype=float)

    report = {"n_docs": int(len(np.unique(doc_ids))), "n_rows": len(rows), "counters": cnames,
              "doc_tokens": a.doc_tokens, "seed": a.seed, "n_boot": a.n_boot,
              "gpt2_snapshot": a.gpt2_snapshot, "cells": []}

    for mname, mvals in metrics.items():
        for ax in AXES:
            human = np.array([r[ax] for r in rows], dtype=float)
            ok = np.isfinite(mvals) & np.isfinite(human) & np.isfinite(Xc).all(axis=1)
            mv, hv, gv = mvals[ok], human[ok], doc_ids[ok]
            sv, Xk = sys_ids[ok], Xc[ok]

            tau_metric = summary_level_tau(mv, hv, gv)
            tau_sys = system_level_tau(mv, hv, sv)

            oof_A = nested_cv(Xk, hv, gv, a.seed)                       # counters
            oof_B = nested_cv(np.column_stack([Xk, mv]), hv, gv, a.seed)  # counters + metric
            oof_M = nested_cv(mv.reshape(-1, 1), hv, gv, a.seed)          # metric alone

            tau_counter = summary_level_tau(oof_A, hv, gv)
            tau_metric_model = summary_level_tau(oof_M, hv, gv)
            # Two reproduction fractions, because the obvious one is not apples-to-apples.
            #   `repro`         counters-MODEL vs the RAW metric. This is the quantity W6 Sec. 2 is phrased
            #                   as ("length + entity count reproduce >= 1/2 of Shannon's correlation"), but
            #                   it flatters the counters: the counter side is a ridge FITTED to this human
            #                   axis, the metric side is a fixed function that never sees the axis.
            #   `repro_matched` counters-MODEL vs metric-MODEL, both fitted out-of-fold under the identical
            #                   nested CV. This is the fair comparison and the one to lead with if the two
            #                   disagree; the asymmetry above is exactly the kind of thing this paper
            #                   audits other people for.
            repro = (tau_counter / tau_metric) if (np.isfinite(tau_metric) and abs(tau_metric) > 1e-9) else np.nan
            repro_matched = (tau_counter / tau_metric_model) \
                if (np.isfinite(tau_metric_model) and abs(tau_metric_model) > 1e-9) else np.nan

            r2A, r2B, r2M = r2(hv, oof_A), r2(hv, oof_B), r2(hv, oof_M)
            inc = r2B - r2A

            incv = boot_vals(lambda idx, gid: r2(hv[idx], oof_B[idx]) - r2(hv[idx], oof_A[idx]),
                             gv, a.n_boot, a.seed)
            lo, hi = (float(np.percentile(incv, 2.5)), float(np.percentile(incv, 97.5))) \
                if len(incv) else (np.nan, np.nan)
            # one-sided bootstrap p for "the metric adds nothing", carried into a BH correction over the
            # eight (metric x axis) cells -- without it eight uncorrected intervals are eight chances to
            # find an increment.
            p_inc = float((incv <= 0).mean()) if len(incv) else np.nan
            # gid, not gv[idx]: see boot_ci's docstring -- gv[idx] collapses repeated draws.
            tlo, thi = boot_ci(lambda idx, gid: summary_level_tau(mv[idx], hv[idx], gid),
                               gv, a.n_boot, a.seed)
            # the registered prediction is stated on the reproduction fraction, so it gets an interval too
            rlo, rhi = boot_ci(
                lambda idx, gid: (summary_level_tau(oof_A[idx], hv[idx], gid)
                                  / summary_level_tau(mv[idx], hv[idx], gid))
                if abs(summary_level_tau(mv[idx], hv[idx], gid) or 0.0) > 1e-9 else np.nan,
                gv, a.n_boot, a.seed)

            cell = {"metric": mname, "axis": ax, "n": int(ok.sum()),
                    "tau_summary_level": tau_metric, "tau_summary_level_ci": [tlo, thi],
                    "tau_system_level": tau_sys,
                    "tau_counter_only": tau_counter, "tau_metric_model": tau_metric_model,
                    "reproduction_fraction": repro, "reproduction_fraction_matched": repro_matched,
                    "r2_counters": r2A, "r2_counters_plus_metric": r2B, "r2_metric_only": r2M,
                    "conditional_increment_r2": inc, "conditional_increment_ci": [lo, hi],
                    "conditional_increment_p_one_sided": p_inc,
                    "reproduction_fraction_ci": [rlo, rhi],
                    "increment_clears_zero": bool(np.isfinite(lo) and lo > 0)}
            report["cells"].append(cell)
            print(f"[w6] {mname:14s} {ax:12s} tau {tau_metric:+.4f} [{tlo:+.4f},{thi:+.4f}] | "
                  f"sys {tau_sys:+.4f} | counters {tau_counter:+.4f} metric-model {tau_metric_model:+.4f} "
                  f"(repro {repro:.2f} [{rlo:.2f},{rhi:.2f}] matched {repro_matched:.2f}) | "
                  f"dR2 {inc:+.4f} [{lo:+.4f},{hi:+.4f}] p={p_inc:.4f}", flush=True)

    incs = [c["conditional_increment_ci"][0] for c in report["cells"]]
    report["n_cells_increment_clears_zero"] = int(sum(1 for lo in incs if np.isfinite(lo) and lo > 0))
    # BH over the eight cells on the one-sided increment p, so the "which metrics add signal" count is
    # multiplicity-controlled rather than eight independent looks at alpha = 0.05.
    pv = [c["conditional_increment_p_one_sided"] for c in report["cells"]]
    pv = [1.0 if not np.isfinite(x) else x for x in pv]
    keep = bh(pv)
    for c, k in zip(report["cells"], keep):
        c["increment_bh_significant"] = bool(k)
    report["n_cells_increment_bh_significant"] = int(sum(keep))
    rep = [c["reproduction_fraction"] for c in report["cells"]
           if c["metric"] == "shannon_score" and c["axis"] == "relevance"]
    repm = [c["reproduction_fraction_matched"] for c in report["cells"]
            if c["metric"] == "shannon_score" and c["axis"] == "relevance"]
    report["registered_prediction"] = {
        "statement": "counters reproduce >= 1/2 of Shannon's summary-level tau with relevance",
        "shannon_relevance_reproduction_fraction": rep[0] if rep else None,
        "shannon_relevance_reproduction_fraction_matched": repm[0] if repm else None,
        "holds": bool(rep and np.isfinite(rep[0]) and rep[0] >= 0.5),
        "holds_matched": bool(repm and np.isfinite(repm[0]) and repm[0] >= 0.5)}

    (out / "w6_report.json").write_text(json.dumps(report, indent=1, default=float))
    with open(out / "w6_rows.jsonl", "w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            rec = {"doc_id": r["doc_id"], "sys_id": r["sys_id"],
                   **{ax: r[ax] for ax in AXES},
                   **{k: float(cnt[i][k]) for k in cnames},
                   **{m: (None if not np.isfinite(v[i]) else float(v[i])) for m, v in metrics.items()}}
            fh.write(json.dumps(rec) + "\n")
    print(f"[w6] wrote {out}/w6_report.json and w6_rows.jsonl", flush=True)
    print(f"[w6] registered prediction holds: {report['registered_prediction']['holds']}", flush=True)


if __name__ == "__main__":
    main()
