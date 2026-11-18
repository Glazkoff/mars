#!/usr/bin/env python3
"""H-J9 -- render tables/marsc_backbone.tex from the hj9 evaluator outputs and observed contrasts, both pools.

Rows: the natural-coverage package on the FactCG encoder (human and judge-twin packages, with their
summary-blind controls), the deployed MARS-C verifier, its matched judge twin, and zero-shot FactCG,
MiniCheck-7B and the judge. Columns per pool: recall@10, omission precision, and the OBSERVED contrast
(new human-package arm minus row) with document-clustered percentile intervals from b3_contrast.py.
Nothing is computed here beyond formatting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWS = [  # (arm key, label, block)
    ("humanfact+div1", r"\textbf{MARS-C, pipeline of record (DeBERTa-large-MNLI)}", "rec"),
    ("matched.judge+div1", "judge-label twin, DeBERTa-large-MNLI", "rec"),
    ("hj9factcghuman+div1", "human-fact package, FactCG encoder (bar failed)", "factcg"),
    ("hj9factcghuman.shuffled+div1", r"\quad another document's summary", "factcg"),
    ("hj9factcghuman.empty+div1", r"\quad no summary", "factcg"),
    ("hj9factcgjudge+div1", "judge-label twin, FactCG encoder", "factcg"),
    ("hj9xlargehuman+div1", "human-fact package, DeBERTa-xlarge-MNLI (size control)", "xl"),
    ("hj9xlargehuman.shuffled+div1", r"\quad another document's summary", "xl"),
    ("hj9xlargehuman.empty+div1", r"\quad no summary", "xl"),
    ("hj9xlargejudge+div1", "judge-label twin, DeBERTa-xlarge-MNLI", "xl"),
    ("nli_factcg_real+div1", "FactCG-DeBERTa-v3-L, zero-shot", "zs"),
    ("nli_minicheck7b_real+div1", "Bespoke-MiniCheck-7B, zero-shot", "zs"),
    ("nli_judge_gemma31_real+div1", "Gemma-4-31B zero-shot judge", "zs"),
]


def cell(c):
    lo, hi = c["ci_percentile"]
    return f"${c['observed']:+.3f}$ $[{lo:+.3f},{hi:+.3f}]$"


def load(pool_dir: Path, pool: str):
    e2 = json.load(open(pool_dir / f"eval_{pool}" / "e2_emitted.json"))["k"]["10"]["systems"]
    b3 = json.load(open(pool_dir / "b3_contrasts.json"))["evals"][f"{pool}_hj9joint"]["contrasts"]
    return e2, b3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-dir", required=True, help="hj9_joint/ (one evaluator call per pool, primary = pipeline of record)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.joint_dir)
    data = {pool: load(root / f"eval_{pool}", pool) for pool in ("test", "unisum")}
    lines = []
    for pool, title in (("test", "RoSE held-out split (2{,}688 pairs, 285 documents)"),
                        ("unisum", "UniSumEval (1{,}813 pairs, 222 documents)")):
        e2, b3 = data[pool]
        if pool == "unisum":
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{5}{@{}l}{\emph{" + title + r"}}\\")
        prev = None
        for key, label, blk in ROWS:
            if key not in e2:
                continue
            if blk != prev and prev is not None:
                lines.append(r"\cmidrule(lr){1-5}")
            prev = blk
            s_ = e2[key]; con = b3.get(key)
            rc, pc = (cell(con["recall"]), cell(con["omission_precision"])) if con else (r"\pend", r"\pend")
            lines.append(f"{label} & ${s_['recall']:.3f}$ & ${s_['omission_precision']:.3f}$ & {rc} & {pc} \\\\")
    body = "\n".join(lines)
    tex = r"""% H-J9 / H-J9b backbone x label-source design, rendered by research/marsc_ieee_20260922/code/hj9_table.py
% from hj9_joint/ (every arm in one evaluator call per pool; primary = the pipeline of record).
\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lccll@{}}
\toprule
& & & \multicolumn{2}{c}{\textbf{pipeline of record minus this row (95\% CI)}} \\
\cmidrule(lr){4-5}
\textbf{Emitter (Gemma+ent inventory, rule)} & \textbf{rec@10} & \textbf{om.\,prec} & recall@10 & omission precision \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\caption{\textbf{The natural-coverage package on two other encoders.} The deployed recipe, unchanged, with
the verifier initialised from FactCG's DeBERTa-v3-large and, as a size control at the deployed objective,
from DeBERTa-xlarge-MNLI, each trained on the human coverage labels and, as its twin, on the judge's labels
for the identical cells; three seeds, the frozen inventory and rule, both pools, both endpoints, with the
summary-blind controls of each human arm. Contrasts are observed document-paired differences of the
pipeline of record minus each row, every arm in one evaluator call per pool. The FactCG block's registered
bar, a lead over zero-shot FactCG on both pools, failed (\S\ref{sec:marsc_results}); the size control
carried no bar. Post-hoc secondary: both pools had been read before.}
\label{tab:marsc_backbone}
\end{table}
"""
    Path(a.out).write_text(tex)
    print(f"wrote {a.out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
