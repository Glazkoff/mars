#!/usr/bin/env python3
"""M-E -- render tables/mars_cost_e2e.tex from the per-arm cost_*.json files (three repeats each, median
reported). No bar; a measurement table for the package's end-to-end cost, distinct from Table marsc_cost /
marsc_deploy which measure the pair verifier alone.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path

ARMS = [
    ("mars_seq2seq", "MARS (distilled inventory + FactCG P + MARS-C R)"),
    ("mars_llm", "MARS (Gemma-4-31B inventory + FactCG P + MARS-C R)"),
    ("mars_sentences", "MARS (sentences as facts, no inventory model)"),
    ("factcg_sent", "FactCG alone, sentence-level claims"),
]


def load(indir: str, arm: str) -> dict | None:
    files = sorted(glob.glob(f"{indir}/cost_{arm}_rep*.json"))
    if not files:
        return None
    rows = [json.load(open(f)) for f in files]
    keys = ["seconds_per_pair_batched", "latency_batch1_median_ms", "peak_alloc_batched_bytes",
            "peak_alloc_batch1_bytes", "weight_bytes_in_memory", "facts_per_candidate"]
    return {"n_reps": len(rows), **{k: st.median(r[k] for r in rows) for k in keys}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines = []
    missing = []
    for arm, label in ARMS:
        m = load(a.indir, arm)
        if m is None:
            missing.append(arm)
            continue
        lines.append(
            f"{label} & ${m['seconds_per_pair_batched']:.2f}$ & ${m['latency_batch1_median_ms']:.0f}$ & "
            f"${m['peak_alloc_batched_bytes']/2**30:.1f}$ & ${m['weight_bytes_in_memory']/2**30:.1f}$ & "
            f"${m['facts_per_candidate']:.1f}$ \\\\"
        )
    tex = r"""% M-E end-to-end package cost, rendered by research/mars_metric_20260923/code/me_table.py.
\begin{table}[t]
\centering\small
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Configuration} & \textbf{s/pair} & \textbf{batch-1 ms} & \textbf{peak GiB} & \textbf{weights GiB} & \textbf{facts/cand} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption{\textbf{End-to-end package cost}, median of three repeats on $200$ UniSumEval pairs, one H200
(batched throughput at batch $32$; batch-$1$ latency the median of $60$ single calls after ten warm-ups).
Distinct from Table~\ref{tab:marsc_cost}/\ref{tab:marsc_deploy}, which measure the pair verifier alone: these
rows include the inventory stage (decomposition), both verifiers where applicable, and peak memory of the
whole call graph rather than the verifier's own footprint. No comparator's registered fourth arm
(Bespoke-MiniCheck-7B at sentence level) ran; this table reports the three MARS configurations and the one
comparator that did.}
\label{tab:marsc_cost_e2e}
\end{table}
"""
    Path(a.out).write_text(tex)
    print(f"wrote {a.out}" + (f"; missing arms: {missing}" if missing else ""))


if __name__ == "__main__":
    main()
