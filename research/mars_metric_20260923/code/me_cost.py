#!/usr/bin/env python3
"""M-E -- end-to-end cost of the MARS package per (source, candidate) and of the sentence-level comparators, on
the H-J5 conventions: 200 UniSumEval pairs drawn uniformly at random (seed 20260922, never the first N), batched
throughput over the whole set, batch-1 latency as the median over 60 single pairs after 10 warm-ups with CUDA
synchronised around every call, peak allocated memory after a reset at each setting, resident weight bytes of
every loaded model. Queued design in PREREG.md (M-E); a measurement, no bar.

Arms
  mars_seq2seq   distilled LongT5 inventory + FactCG-contract P verifier + MARS-C pair-contract R verifier (3 seeds)
  mars_llm       the same with the Gemma-4-31B inventory
  mars_sentences the same with sentences as facts (no inventory model)
  factcg_sent    FactCG alone, sentence-level claims, P only (the comparator's own unit)
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from mars.inventory import LLMDecomposer, SentenceDecomposer, Seq2SeqDecomposer  # noqa: E402
from mars.score import MarsScorer  # noqa: E402
from mars.verifiers import PairVerifier, PromptVerifier  # noqa: E402


def weight_bytes(*objs) -> int:
    total = 0
    for o in objs:
        for attr in ("_model", "_models"):
            m = getattr(o, attr, None)
            models = m if isinstance(m, list) else ([m] if m is not None else [])
            for mm in models:
                total += sum(p.numel() * p.element_size() for p in mm.parameters())
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--arm", choices=["mars_seq2seq", "mars_llm", "mars_sentences", "factcg_sent"], required=True)
    ap.add_argument("--factcg", required=True)
    ap.add_argument("--marsc", required=True, help="glob of MARS-C seed checkpoints")
    ap.add_argument("--segmenter", default="")
    ap.add_argument("--llm", default="")
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--n-single", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    rows = [json.loads(l) for l in open(a.pairs, encoding="utf-8")]
    rows = [r for r in rows if r["dataset"] == "unisumeval" and r["references"]]
    rng = random.Random(a.seed); sel = rng.sample(rows, a.n_pairs)
    cands = [r["candidate"] for r in sel]; srcs = [r["source"] for r in sel]; refs = [r["references"] for r in sel]
    p_ver = PromptVerifier(a.factcg, batch=32)
    if a.arm == "factcg_sent":
        scorer = MarsScorer(p_ver, None, SentenceDecomposer())
        objs = (p_ver,)
    else:
        r_ver = PairVerifier.from_glob(a.marsc, batch=128)
        dec = {"mars_seq2seq": lambda: Seq2SeqDecomposer(a.segmenter, batch=32), "mars_llm": lambda: LLMDecomposer(a.llm, batch=12),
               "mars_sentences": lambda: SentenceDecomposer()}[a.arm]()
        scorer = MarsScorer(p_ver, r_ver, dec)
        objs = (p_ver, r_ver, dec)

    def run(c, s, r):
        return scorer.score(c, sources=s, references=(r if a.arm != "factcg_sent" else None))
    # warm-up (loads every model)
    run(cands[:4], srcs[:4], refs[:4]); torch.cuda.synchronize()
    weights = weight_bytes(*objs); after_load = torch.cuda.memory_allocated()
    # batched: the whole set in one call
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.perf_counter()
    res = run(cands, srcs, refs); torch.cuda.synchronize(); el = time.perf_counter() - t0
    peak_batched = torch.cuda.max_memory_allocated()
    # batch-1: single pairs
    single_idx = rng.sample(range(len(sel)), a.n_single)
    for i in single_idx[:10]:
        run([cands[i]], [srcs[i]], [refs[i]])
    torch.cuda.reset_peak_memory_stats(); lat = []
    for i in single_idx:
        torch.cuda.synchronize(); t1 = time.perf_counter(); run([cands[i]], [srcs[i]], [refs[i]]); torch.cuda.synchronize(); lat.append(time.perf_counter() - t1)
    peak_single = torch.cuda.max_memory_allocated()
    rep = {"arm": a.arm, "n_pairs": len(sel), "seconds_batched_total": el, "seconds_per_pair_batched": el / len(sel),
           "latency_batch1_median_ms": 1e3 * statistics.median(lat), "latency_batch1_p90_ms": 1e3 * sorted(lat)[int(0.9 * len(lat)) - 1],
           "peak_alloc_batched_bytes": peak_batched, "peak_alloc_batch1_bytes": peak_single, "weight_bytes_in_memory": weights,
           "alloc_after_load_bytes": after_load, "facts_per_candidate": statistics.mean(x.n_candidate_facts for x in res),
           "reference_facts_per_pair": statistics.mean(x.n_reference_facts for x in res),
           "mean_P": statistics.mean(x.P for x in res if x.P is not None) if any(x.P is not None for x in res) else None,
           "mean_R": statistics.mean(x.R for x in res if x.R is not None) if any(x.R is not None for x in res) else None,
           "device": torch.cuda.get_device_name(0)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"[me] {a.arm}: {rep['seconds_per_pair_batched']*1e3:.0f} ms/pair batched, {rep['latency_batch1_median_ms']:.0f} ms batch-1 median, "
          f"peak {peak_batched/2**30:.2f} / {peak_single/2**30:.2f} GiB, weights {weights/2**30:.2f} GiB, facts/cand {rep['facts_per_candidate']:.1f}", flush=True)


if __name__ == "__main__":
    main()
