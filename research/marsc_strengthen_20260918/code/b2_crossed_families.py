#!/usr/bin/env python3
"""MARS-C strengthen B2 / H-COND -- the same-fact crossed-accuracy endpoint, computed IDENTICALLY for every
verifier family that can be dropped into the frozen inventory and emission rule.

Registered in research/marsc_strengthen_20260918/PREREG.md (Wave B / B0, B2).

The estimand (the R03a estimator, unchanged): within a document, for the SAME fact f,

    crossed accuracy = P( z(f | a summary that OMITS f) > z(f | a summary that CONVEYS f) ),  ties = 1/2

Any summary-blind scorer is pinned at exactly 0.500 -- it cannot be won by ignoring the candidate, which is why
this and not recall@k is the endpoint H-COND is barred on. The cells, the win indicator, the document-macro
statistic and the document-clustered bootstrap are reused verbatim from
research/marsc_20260916/code/mc_summary_conditioning.py (imported, never modified); this file only widens the
set of scorers that can produce `z` and adds the paired MARS-C-minus-family margin the bar is stated on.

Families (--families NAME=KIND:ARG):
  ckpt:<glob>    saved cross-encoder dirs; score = softmax(logits,-1)[:,0] for a 2-class head (b21 xenc),
                 sigmoid(logit) for the 1-logit head (MARS-C humanfact). Seed roster + seed average.
  zs:<metric>    scripts/strengthen_score_cells.py::build_scorer metric (nli, summac_zs, minicheck,
                 alignscore, factcc, bartscore); premise = candidate summary, claim = fact, omission = 1 - support.
  lex:<fn>       common.token_recall / common.rougeL_recall counter, omission = 1 - recall.
  emb:<model>    sentence-transformers max cosine to a candidate sentence, omission = 1 - maxcos.
The lex and emb families are the implementation's own sanity check: they read the candidate only through a
bag of tokens / one embedding, so they are expected NEAR but not AT 0.500, and a bug that dropped the candidate
entirely would pin every family at exactly 0.500.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402

CHUNK = 256


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


msc = _load(REPO / "research" / "marsc_20260916" / "code" / "mc_summary_conditioning.py")


def _ckey(c) -> str:
    return f"{c[0]}|{c[1]}|{c[2]}"


def cached(cache_dir: str, key: str, cells, fn):
    """Per-arm on-disk memo of the cell -> omission-score map.

    The crossed endpoint is cheap on a GPU and expensive on CPU, and this node's GPUs are contended, so an arm
    that has already been scored must never be scored twice: a job that is cut short by its walltime resumes
    from what it wrote. The cache is keyed by the arm and validated against the exact cell set, so a stale or
    partial file is discarded rather than silently reused.
    """
    if not cache_dir:
        return fn()
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._+-]", "_", key)[:180]
    f = d / f"{safe}.json"
    want = [_ckey(c) for c in cells]
    if f.exists():
        try:
            blob = json.loads(f.read_text())
            if blob.get("cells") == want:
                print(f"[b2] cache HIT  {f.name} ({len(want)} cells)", flush=True)
                return {c: float(v) for c, v in zip(cells, blob["z"])}
            print(f"[b2] cache STALE {f.name}: cell set differs, recomputing", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[b2] cache UNREADABLE {f.name} ({e}), recomputing", flush=True)
    t0 = time.time()
    z = fn()
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"key": key, "cells": want, "z": [float(z[c]) for c in cells]}))
    tmp.replace(f)
    print(f"[b2] cache WRITE {f.name} ({len(want)} cells, {time.time() - t0:.0f}s)", flush=True)
    return z

BARS = {
    "registered": "research/marsc_strengthen_20260918/PREREG.md -- Wave B / B2, hypothesis H-COND",
    "declared_before_any_number_in_this_file_was_read": True,
    "primary_estimand": "within-document, same-fact crossed accuracy P(z(f|omitting summary) > z(f|conveying summary)),"
                        " ties 1/2; document-macro statistic, document-clustered bootstrap",
    "chance_level_for_any_summary_blind_scorer": 0.5,
    "pass": "H-COND PASSES if MARS-C's crossed accuracy exceeds EVERY off-the-shelf family's by a margin whose "
            "document-clustered one-sided 95% lower bound is above 0, on validation AND on the sealed TEST split.",
    "fail": "H-COND FAILS if any off-the-shelf family matches MARS-C within that margin. Then the paper says so and "
            "the supervision claim narrows to the fixed-inventory verifier effect, which this endpoint does not touch.",
    "candidate_blind": "A family whose document-macro crossed accuracy has a two-sided 95% document-clustered "
                       "interval covering 0.500 is reported as CANDIDATE-BLIND, and any recall@k advantage it holds "
                       "is evidence FOR the methodological claim, not against MARS-C.",
    "primary_family": "humanfact (MARS-C natural-coverage cross-encoder, 3 seeds)",
    "off_the_shelf_families": "every family whose kind is zs, lex or emb",
}


# ----------------------------------------------------------------------------- scorers
def score_cells_ckpt(model_dir: str, docs, cells, batch: int, dev: str, window: int = 0, stride: int = 0) -> dict:
    """msc.score_cells with LENGTH-BUCKETED batching.

    Identical premise/hypothesis construction, identical head convention (softmax(logits,-1)[:,0] for the
    2-class b21 judge head, sigmoid(logit) for the 1-logit MARS-C head) -- the only change is that cells are
    batched in length order, so a batch is not padded to the longest sequence in the whole pool. Padding is
    masked out, so this is a speed change, not a scoring change; the MARS-C rows are checked against the
    published R03a values (validation triple 0.9181 / doc-macro 0.9173) as the guard.

    `window`/`stride` (H-J10a, 2026-09-22): the deployed verifier reads the first 380 words of a candidate.
    When `window` > 0 a candidate longer than `window` words is scored once per overlapping window of
    `window` words (`stride` words apart, the last window ending at the final word) and the cell's omission
    score is the MINIMUM over windows: a fact counts as conveyed if any window conveys it. With the default
    `window == 0` this function is byte-for-byte the published path.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev).eval()
    two_class = getattr(model.config, "num_labels", 1) == 2
    if window:
        # one sub-cell per window; keyed (cell, window index) and reduced by min below
        assert stride > 0, "--window needs --stride"
        prem, hypo = {}, {}
        for c in cells:
            for wi, w in enumerate(windows(docs[c[0]]["summaries"][c[1]]["candidate"], window, stride)):
                prem[(c, wi)] = w
                hypo[(c, wi)] = msc.clip(docs[c[0]]["facts"][c[2]], 96)
    else:
        prem = {c: msc.clip(docs[c[0]]["summaries"][c[1]]["candidate"], 380) for c in cells}
        hypo = {c: msc.clip(docs[c[0]]["facts"][c[2]], 96) for c in cells}
    order = sorted(prem, key=lambda c: (len(prem[c].split()) + len(hypo[c].split()), c))
    out = {}
    with torch.inference_mode():
        for s in range(0, len(order), batch):
            ch = order[s:s + batch]
            enc = tok([prem[c] for c in ch], [hypo[c] for c in ch], truncation="longest_first",
                      max_length=512, padding=True, return_tensors="pt").to(dev)
            lg = model(**enc).logits.float()
            v = torch.softmax(lg, -1)[:, 0] if two_class else torch.sigmoid(lg.squeeze(-1))
            for c, x in zip(ch, v.tolist()):
                out[c] = x
    if window:
        red: dict = {}
        for (c, _wi), x in out.items():
            red[c] = min(red.get(c, 1.0), x)
        out = red
        print(f"[b2]   windowed ({window} words, stride {stride}): {len(prem)} window cells for {len(cells)} cells",
              flush=True)
    del model
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def windows(text: str, window: int, stride: int) -> list[str]:
    """Overlapping `window`-word windows `stride` words apart; the last one ends at the final word."""
    w = text.split()
    if len(w) <= window:
        return [" ".join(w)]
    starts = list(range(0, len(w) - window + 1, stride))
    if starts[-1] + window < len(w):
        starts.append(len(w) - window)
    return [" ".join(w[s:s + window]) for s in starts]


def score_cells_zs(metric: str, docs, cells, batch: int, dev: str) -> dict:
    """Omission score from a canonical entailment metric: premise = candidate summary, claim = the fact."""
    ssc = common._load("strengthen_score_cells")
    prem = [docs[di]["summaries"][si]["candidate"] for di, si, _ in cells]
    hypo = [msc.clip(docs[di]["facts"][fi], 96) for di, _, fi in cells]
    sc = ssc.build_scorer(metric, batch, dev)
    vals: list[float] = []
    for s in range(0, len(cells), CHUNK):
        vals.extend(sc.score(prem[s:s + CHUNK], hypo[s:s + CHUNK]))
        if (s // CHUNK) % 10 == 0:
            print(f"[b2]   {metric} {min(s + CHUNK, len(cells))}/{len(cells)}", flush=True)
    assert len(vals) == len(cells), f"{metric}: {len(vals)} values for {len(cells)} cells"
    del sc
    return {c: 1.0 - float(v) for c, v in zip(cells, vals)}


def score_cells_lex(fn: str, docs, cells) -> dict:
    f = {"token_recall": common.token_recall, "rougeL_recall": common.rougeL_recall}[fn]
    return {c: 1.0 - float(f(msc.clip(docs[c[0]]["facts"][c[2]], 96), docs[c[0]]["summaries"][c[1]]["candidate"]))
            for c in cells}


def score_cells_emb(model_id: str, docs, cells, dev: str) -> dict:
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(model_id, device=dev)
    sums = sorted({(di, si) for di, si, _ in cells})
    facts = sorted({(di, fi) for di, _, fi in cells})
    cand_emb = {}
    for di, si in sums:
        s = common.sentences(docs[di]["summaries"][si]["candidate"]) or [docs[di]["summaries"][si]["candidate"]]
        cand_emb[(di, si)] = st.encode(s, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    fe = st.encode([msc.clip(docs[di]["facts"][fi], 96) for di, fi in facts], batch_size=128,
                   convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    fact_emb = {k: fe[i] for i, k in enumerate(facts)}
    del st
    return {(di, si, fi): 1.0 - float((cand_emb[(di, si)] @ fact_emb[(di, fi)]).max()) for di, si, fi in cells}


# ----------------------------------------------------------------------------- statistics
def doc_means_of(z: dict, triples) -> dict[int, float]:
    _, by_doc = msc.discriminate(z, triples)
    return {d: float(np.mean(v)) for d, v in by_doc.items()}


def paired_margin(a_means: dict, b_means: dict, seed: int, n_boot: int) -> dict:
    """Document-clustered PAIRED bootstrap of (family a - family b) on the document-macro crossed accuracy."""
    keys = sorted(set(a_means) & set(b_means))
    if not keys:
        return {"n_docs": 0}
    d = np.array([a_means[k] - b_means[k] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    draws = d[rng.integers(0, len(keys), size=(n_boot, len(keys)))].mean(axis=1)
    return {"n_docs": len(keys), "observed_margin_doc_macro": float(d.mean()),
            "ci95_two_sided": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "lower_bound_95_one_sided": float(np.percentile(draws, 5.0)),
            "p_one_sided_bootstrap": float(np.mean(draws <= 0.0)),
            "n_docs_a_better": int(np.sum(d > 0)), "n_docs_b_better": int(np.sum(d < 0)),
            "n_docs_tied": int(np.sum(d == 0))}


def verdict_blind(stat: dict) -> str:
    lo, hi = stat["ci_doc_bootstrap"]
    if lo <= 0.5 <= hi:
        return "CANDIDATE-BLIND"
    return "CONDITIONS-ON-CANDIDATE" if lo > 0.5 else "ANTI-CONDITIONED"


# ----------------------------------------------------------------------------- driver
def run_split(a) -> dict:
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    docs = common.load_jsonl(Path(a.docs))
    if a.limit_docs:
        docs = docs[:a.limit_docs]
    cells, triples = msc.crossed_cells(docs)
    print(f"[b2] {Path(a.docs).name}: {len(docs)} docs -> {len(cells)} crossed cells, {len(triples)} same-fact "
          f"triples over {len({t[0] for t in triples})} docs", flush=True)

    rep = {"bars": BARS, "split_docs_file": str(a.docs), "docs": len(docs), "cells": len(cells),
           "triples": len(triples), "docs_with_triples": len({t[0] for t in triples}),
           "facts_with_triples": len({(t[0], t[1]) for t in triples}), "chance_level": 0.5,
           "bootstrap_seed": a.seed, "n_boot": a.n_boot, "families": {}}
    if a.window:
        rep["ckpt_candidate_windows"] = {"window_words": a.window, "stride_words": a.stride,
                                         "reduction": "min omission score over windows (ckpt families only; "
                                                      "zs/lex/emb families read the candidate as before)"}

    dm: dict[str, dict[int, float]] = {}
    for spec in a.families:
        name, rest = spec.split("=", 1)
        kind, arg = rest.split(":", 1)
        entry = {"kind": kind, "spec": arg}
        if kind == "ckpt":
            ck = [d for d in sorted(glob.glob(arg)) if Path(d, "config.json").exists()]
            assert ck, f"family {name!r}: no checkpoint matched {arg!r}"
            if a.expect_seeds:
                assert len(ck) == a.expect_seeds, f"family {name!r}: expected {a.expect_seeds} checkpoints, got {ck}"
            entry["checkpoints"] = ck
            per_seed, acc = {}, defaultdict(list)
            for d in ck:
                seed = m.group(1) if (m := re.search(r"seed(\d+)", d)) else Path(d).name
                wkey = f".w{a.window}s{a.stride}" if a.window else ""
                z = cached(a.cache, f"{name}.seed{seed}{wkey}", cells,
                           lambda d=d: score_cells_ckpt(d, docs, cells, a.batch, dev, a.window, a.stride))
                for c, v in z.items():
                    acc[c].append(v)
                w, bd = msc.discriminate(z, triples)
                per_seed[seed] = msc.summarise(w, bd, np.random.default_rng(a.seed), a.n_boot)
                print(f"[b2] {name} seed {seed}: crossed accuracy {per_seed[seed]['discrimination_triple']:.4f}", flush=True)
            entry["per_seed"] = per_seed
            z = {c: float(np.mean(v)) for c, v in acc.items()}
        elif kind == "zs":
            z = cached(a.cache, f"{name}.zs.{arg}", cells,
                       lambda: score_cells_zs(arg, docs, cells, a.batch, dev))
        elif kind == "lex":
            z = cached(a.cache, f"{name}.lex.{arg}", cells,
                       lambda: score_cells_lex(arg, docs, cells))
        elif kind == "emb":
            z = cached(a.cache, f"{name}.emb.{arg}", cells,
                       lambda: score_cells_emb(arg, docs, cells, dev))
        else:
            raise SystemExit(f"unknown family kind {kind!r} in {spec!r}")
        w, bd = msc.discriminate(z, triples)
        stat = msc.summarise(w, bd, np.random.default_rng(a.seed), a.n_boot)
        entry["seed_averaged"] = stat
        entry["verdict_vs_chance"] = verdict_blind(stat)
        entry["off_the_shelf"] = kind in ("zs", "lex", "emb")
        rep["families"][name] = entry
        dm[name] = {d: float(np.mean(v)) for d, v in bd.items()}
        ci = stat["ci_doc_bootstrap"]
        print(f"[b2] {name:22s} crossed accuracy {stat['discrimination_triple']:.4f} (doc macro "
              f"{stat['discrimination_doc_macro']:.4f} [{ci[0]:.4f},{ci[1]:.4f}]) -> {entry['verdict_vs_chance']}", flush=True)
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

    prim = a.primary
    assert prim in dm, f"--primary {prim!r} is not among the families {sorted(dm)}"
    rep["primary_family"] = prim
    rep["margins_vs_primary"] = {}
    for name in rep["families"]:
        if name == prim:
            continue
        mg = paired_margin(dm[prim], dm[name], a.seed, a.n_boot)
        mg["family_is_off_the_shelf"] = rep["families"][name]["off_the_shelf"]
        mg["clears_bar_on_this_split"] = bool(mg.get("lower_bound_95_one_sided", -1.0) > 0.0)
        rep["margins_vs_primary"][name] = mg
        print(f"[b2] margin {prim} - {name:22s} {mg['observed_margin_doc_macro']:+.4f} "
              f"one-sided 95% LB {mg['lower_bound_95_one_sided']:+.4f} "
              f"(p={mg['p_one_sided_bootstrap']:.4f}) -> {'CLEARS' if mg['clears_bar_on_this_split'] else 'DOES NOT CLEAR'}", flush=True)

    off = [n for n, m in rep["margins_vs_primary"].items() if m["family_is_off_the_shelf"]]
    failing = [n for n in off if not rep["margins_vs_primary"][n]["clears_bar_on_this_split"]]
    rep["hcond_split_verdict"] = {
        "off_the_shelf_families_compared": off,
        "families_not_cleared": failing,
        "split_result": "PASS-ON-THIS-SPLIT" if off and not failing else ("FAIL-ON-THIS-SPLIT" if off else "NO-COMPARATOR"),
        "note": "The registered bar requires PASS on validation AND on the sealed TEST split; use --combine.",
    }
    print(f"[b2] H-COND on this split: {rep['hcond_split_verdict']['split_result']} "
          f"(not cleared: {failing or 'none'})", flush=True)
    return rep


# The registered H-COND bar is a conjunction over BOTH splits and over the registered off-the-shelf family set
# (PREREG Wave B / B2). combine() used to judge a family on whichever splits happened to be passed, so a lone
# validation file could print PASS (audit 2026-09-27, medium item 6). It now refuses to pass unless every required
# split and every required family is present; a missing piece yields INCOMPLETE, never PASS.
REQUIRED_SPLITS = ("validation", "test")
REQUIRED_OFF_THE_SHELF = ("nli", "summac_zs", "lex_token_recall", "lex_rougeL_recall", "emb_minilm")


def split_of(path: str) -> str:
    """crossed_<split>/crossed_families.json -> <split>; also accepts a bare <split> directory."""
    name = Path(path).parent.name or Path(path).name
    return name[len("crossed_"):] if name.startswith("crossed_") else name


def combine(paths: list[str], required_splits=REQUIRED_SPLITS, required_families=REQUIRED_OFF_THE_SHELF) -> dict:
    per = {}
    for p in paths:
        key = split_of(p)
        if key in per:
            raise SystemExit(f"[b2] combine: two inputs map to split {key!r}; refusing to overwrite ({p})")
        per[key] = json.loads(Path(p).read_text())
    names = sorted({n for r in per.values() for n in r.get("margins_vs_primary", {})})
    rows = {}
    for n in names:
        ms = {k: r["margins_vs_primary"][n] for k, r in per.items() if n in r.get("margins_vs_primary", {})}
        rows[n] = {"per_split": ms,
                   "off_the_shelf": any(m["family_is_off_the_shelf"] for m in ms.values()),
                   "splits_present": sorted(ms),
                   "clears_every_split": bool(ms) and set(required_splits) <= set(ms)
                                         and all(m["clears_bar_on_this_split"] for m in ms.values())}
    missing_splits = sorted(set(required_splits) - set(per))
    missing_families = sorted({f for f in required_families for k in required_splits
                               if k in per and f not in per[k].get("margins_vs_primary", {})})
    off = [n for n, v in rows.items() if v["off_the_shelf"]]
    failing = [n for n in off if not rows[n]["clears_every_split"]]
    if missing_splits or missing_families:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS" if off and not failing else ("FAIL" if off else "NO-COMPARATOR")
    return {"bars": BARS, "splits": sorted(per), "inputs": paths, "margins": rows,
            "required_splits": list(required_splits), "required_families": list(required_families),
            "missing_splits": missing_splits, "missing_families": missing_families,
            "hcond_verdict": verdict,
            "families_not_cleared": failing,
            "candidate_blind_families": sorted({n for r in per.values() for n, f in r["families"].items()
                                                if f["verdict_vs_chance"] == "CANDIDATE-BLIND"})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="", help="e0 docs_<split>.jsonl")
    ap.add_argument("--families", nargs="*", default=[], help="NAME=ckpt:<glob>|zs:<metric>|lex:<fn>|emb:<model>")
    ap.add_argument("--primary", default="humanfact")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260917,
                    help="20260917 reproduces the published R03a intervals for the MARS-C families exactly")
    ap.add_argument("--expect-seeds", type=int, default=3)
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--window", type=int, default=0,
                    help="ckpt families: score a candidate longer than this many words once per overlapping "
                         "window and take the minimum omission score (0 = the published 380-word clip)")
    ap.add_argument("--stride", type=int, default=0, help="window stride in words (with --window)")
    ap.add_argument("--cache", default="", help="per-arm on-disk score cache; lets a walltime-truncated "
                                              "job resume instead of rescoring what it already has")
    ap.add_argument("--combine", nargs="*", default=[], help="crossed_families.json paths -> one H-COND verdict")
    ap.add_argument("--require-splits", default=",".join(REQUIRED_SPLITS))
    ap.add_argument("--require-families", default=",".join(REQUIRED_OFF_THE_SHELF))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.combine:
        rep = combine(a.combine, tuple(x for x in a.require_splits.split(",") if x),
                      tuple(x for x in a.require_families.split(",") if x))
        (out / "hcond_verdict.json").write_text(json.dumps(rep, indent=1))
        print(f"[b2] H-COND across {rep['splits']}: {rep['hcond_verdict']} "
              f"(not cleared: {rep['families_not_cleared'] or 'none'}; "
              f"candidate-blind: {rep['candidate_blind_families'] or 'none'})", flush=True)
        return
    assert a.docs and a.families, "--docs and --families are required unless --combine is given"
    rep = run_split(a)
    (out / "crossed_families.json").write_text(json.dumps(rep, indent=1))
    print(f"[b2] written {out / 'crossed_families.json'}", flush=True)


if __name__ == "__main__":
    main()
