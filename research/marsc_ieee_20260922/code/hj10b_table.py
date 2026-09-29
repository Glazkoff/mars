#!/usr/bin/env python3
"""H-J10b -- render tables/marsc_omb_e2e.tex from the OmissionBench end-to-end evaluation (hj10b/eval_cmp):
recall@10 of the removed fact over the errored notes, ACU-aligned omission precision over every note, the mean
number of the ten emissions on a CLEAN twin that align to a fact the note states (the benchmark's false-alarm
axis at ten alerts), and the OBSERVED consultation-clustered contrasts MARS-C minus row. Nothing is computed
here beyond averaging the per-note dump for the clean-twin column.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROWS = [
    ("humanfact+div1", r"\textbf{MARS-C human-fact (ours)}", "ours"),
    ("humanfact.shuffled+div1", r"\quad another consultation's note", "ours"),
    ("humanfact.empty+div1", r"\quad no note", "ours"),
    ("judgefact+div1", "MARS-C judge-label twin", "ours"),
    ("nli_judge_gemma31_real+div1", "Gemma-4-31B zero-shot judge, real note", "judge"),
    ("nli_judge_gemma31_shuffled+div1", r"\quad another consultation's note", "judge"),
    ("nli_judge_gemma31_empty+div1", r"\quad no note", "judge"),
    ("nli_factcg_real+div1", "FactCG-DeBERTa-v3-L, real note", "factcg"),
    ("nli_factcg_shuffled+div1", r"\quad another consultation's note", "factcg"),
    ("nli_factcg_empty+div1", r"\quad no note", "factcg"),
    ("nli_minicheck7b_real+div1", "Bespoke-MiniCheck-7B, real note", "mc"),
    ("nli_minicheck7b_shuffled+div1", r"\quad another consultation's note", "mc"),
    ("nli_minicheck7b_empty+div1", r"\quad no note", "mc"),
]


def ci(c):
    lo, hi = c["ci_percentile"]
    return f"${c['observed']:+.3f}$ $[{lo:+.3f},{hi:+.3f}]$"


def clean_false_alerts(per_pair: Path) -> dict[str, tuple[float, list[float]]]:
    """Mean count of aligned-to-stated-fact emissions among the ten, over the clean twins, with a
    consultation-clustered percentile bootstrap (clean twins are one per consultation, so this is a plain
    bootstrap over twins)."""
    by_sys = defaultdict(list)
    for line in open(per_pair, encoding="utf-8"):
        r = json.loads(line)
        if r["pair_id"].endswith("|clean"):
            by_sys[r["system"]].append(r["false_alert"] * r["n_emitted"])
    out = {}
    rng = np.random.default_rng(20260918)
    for s, v in by_sys.items():
        v = np.array(v, dtype=float)
        draws = [v[rng.integers(0, len(v), size=len(v))].mean() for _ in range(2000)]
        out[s] = (float(v.mean()), [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, help="hj10b/eval_cmp (or hj10b/eval)")
    ap.add_argument("--eval-name", default="omb_hj10b_cmp", help="key under b3_contrasts.json['evals']")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.eval_dir)
    e2 = json.load(open(d / "eval_omb" / "e2_emitted.json"))
    k = e2["k"]["10"]["systems"]
    con = json.load(open(d / "b3_contrasts.json"))["evals"][a.eval_name]["contrasts"]
    fa = clean_false_alerts(d / "per_pair_omb_k10.jsonl")
    lines, prev = [], None
    for key, label, blk in ROWS:
        if key not in k:
            continue
        if blk != prev and prev is not None:
            lines.append(r"\midrule")
        prev = blk
        s = k[key]
        f = fa.get(key)
        fac = f"${f[0]:.2f}$ $[{f[1][0]:.2f},{f[1][1]:.2f}]$" if f else r"\pend"
        rc, pc = (ci(con[key]["recall"]), ci(con[key]["omission_precision"])) if key in con else (r"\pend", r"\pend")
        lines.append(f"{label} & ${s['recall']:.3f}$ & ${s['omission_precision']:.3f}$ & {fac} & {rc} & {pc} \\\\")
    tex = r"""% H-J10b: OmissionBench end to end, rendered by research/marsc_ieee_20260922/code/hj10b_table.py from hj10b/eval_cmp.
\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccll@{}}
\toprule
& & & \textbf{clean twin:} & \multicolumn{2}{c}{\textbf{MARS-C minus this row (95\% CI)}} \\
\cmidrule(lr){5-6}
\textbf{Emitter (Gemma+ent inventory, rule)} & \textbf{rec@10} & \textbf{om.\,prec} & false alerts / 10 & recall@10 & omission precision \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption{\textbf{The pipeline end to end on the clinical pool.} OmissionBench's """ + f"{e2['n_pairs']}" + r""" notes over
""" + f"{e2['n_docs']}" + r""" consultations: the inventory is extracted from the transcript by the frozen recipe, the
verifier reads the note, and ten facts are emitted. Recall@$10$ is over the errored notes (one removed fact
each); omission precision over every note's aligned emissions; the fourth column counts, on the clean twins,
how many of the ten emissions align to a fact the note states, with a bootstrap interval over twins.
Contrasts are observed consultation-paired differences with consultation-clustered percentile intervals.
Registered bars: the package effect and the precision penalty for ignoring the note
(\S\ref{sec:marsc_results}); the comparator block carries no bar.}
\label{tab:marsc_omb_e2e}
\end{table}
"""
    Path(a.out).write_text(tex)
    print(f"wrote {a.out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
