#!/usr/bin/env python3
"""H-J3b -- render tables/marsc_unisum_comparators.tex from the evaluator output and the observed contrasts.

Reads hj3/eval_unisum/eval_unisum/e2_emitted.json (recall@10 and omission precision per arm) and
hj3/eval_unisum/b3_contrasts.json (OBSERVED primary-minus-arm contrasts with document-clustered percentile
intervals). Nothing is computed here beyond formatting; every number is the artifact's.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWS = [  # (arm key, label, block)
    ("humanfact+div1", r"\textbf{MARS-C human-fact (ours)}", "ours"),
    ("judgefact+div1", "MARS-C judge-label twin", "ours"),
    ("nli_factcg_real+div1", "FactCG-DeBERTa-v3-L, real summary", "factcg"),
    ("nli_factcg_shuffled+div1", r"\quad another document's summary", "factcg"),
    ("nli_factcg_empty+div1", r"\quad no summary", "factcg"),
    ("nli_minicheck7b_real+div1", "Bespoke-MiniCheck-7B, real summary", "mc"),
    ("nli_minicheck7b_shuffled+div1", r"\quad another document's summary", "mc"),
    ("nli_minicheck7b_empty+div1", r"\quad no summary", "mc"),
    ("nli_judge_gemma31_real+div1", "Gemma-4-31B zero-shot judge, real summary", "judge"),
    ("nli_judge_gemma31_shuffled+div1", r"\quad another document's summary", "judge"),
    ("nli_judge_gemma31_empty+div1", r"\quad no summary", "judge"),
]


def ci(c):
    lo, hi = c["ci_percentile"]
    s = f"${c['observed']:+.3f}$ $[{lo:+.3f},{hi:+.3f}]$"
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2", required=True)
    ap.add_argument("--b3", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    e2 = json.load(open(a.e2)); b3 = json.load(open(a.b3))
    k = e2["k"]["10"]["systems"]; con = b3["evals"]["unisum_hj3"]["contrasts"]
    lines = []; prev = None
    for key, label, blk in ROWS:
        if blk != prev and prev is not None:
            lines.append(r"\midrule")
        prev = blk
        s = k[key]
        if key in con:
            c = con[key]; rc, pc = ci(c["recall"]), ci(c["omission_precision"])
        else:
            rc, pc = r"\pend", r"\pend"
        lines.append(f"{label} & ${s['recall']:.3f}$ & ${s['omission_precision']:.3f}$ & {rc} & {pc} \\\\")
    body = "\n".join(lines)
    n_pairs = e2["n_pairs"]; n_docs = e2["n_docs"]
    tex = r"""% H-J3b: the three post-registration comparators on UniSumEval, with both summary-blind controls.
% Rendered by research/marsc_ieee_20260922/code/hj3_unisum_table.py from hj3/eval_unisum/ artifacts.
\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lccll@{}}
\toprule
& & & \multicolumn{2}{c}{\textbf{MARS-C minus this row (95\% CI)}} \\
\cmidrule(lr){4-5}
\textbf{Emitter (Gemma+ent inventory, rule)} & \textbf{rec@10} & \textbf{om.\,prec} & recall@10 & omission precision \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\caption{\textbf{The three later comparators on the external pool, with their summary-blind controls.}
UniSumEval, """ + f"{n_pairs:,}" + r""" summaries over """ + f"{n_docs}" + r""" documents, the frozen Gemma+ent inventory and emission rule at
$k{=}10$, scored after the pool had been read (post-hoc secondary). Contrasts are observed
document-paired differences with document-clustered percentile intervals. The held-out-split ordering does
not transfer: each comparator leads MARS-C on both emitted endpoints here, while its own summary-blind
arms fall below it on precision by $0.08$--$0.10$ and do not win on recall. The package effect over the
judge-label twin holds on the same rows.}
\label{tab:marsc_unisum_comparators}
\end{table}
"""
    Path(a.out).write_text(tex)
    print(f"wrote {a.out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
