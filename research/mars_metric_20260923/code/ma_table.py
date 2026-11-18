#!/usr/bin/env python3
"""M-A -- render tables/mars_meta.tex from meta_evaluation.json: summary-level Kendall tau of every metric with
every human dimension on the three sets, and a second small table with the candidate-blind controls. Intervals
live in the JSON and in the text's contrasts; the cells carry the observed tau (the convention of meta-evaluation
tables). Nothing is computed here beyond formatting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWS = [  # (column, label, block)
    ("marsR_marsc", r"\textbf{MARS-R}, coverage verifier (ours)", "mars"),
    ("marsR_factcg", r"\textbf{MARS-R}, FactCG verifier", "mars"),
    ("marsR_free", r"\textbf{MARS-R}, reference-free (source facts)", "mars"),
    ("marsP_factcg", r"\textbf{MARS-P}, FactCG verifier", "mars"),
    ("rouge1", "ROUGE-1", "ref"), ("rouge2", "ROUGE-2", "ref"), ("rougeL", "ROUGE-L", "ref"),
    ("bertscore_P", "BERTScore P", "ref"), ("bertscore_R", "BERTScore R", "ref"), ("bertscore_F", "BERTScore F", "ref"),
    ("bartscore_ref2cand", r"BARTScore ref$\to$cand", "ref"),
    ("bartscore_src2cand", r"BARTScore src$\to$cand", "src"), ("alignscore", "AlignScore", "src"), ("factcg_whole", "FactCG, whole summary", "src"),
    ("a3cu_R", "A3CU recall", "unit"), ("a3cu_F", "A3CU F", "unit"), ("a2cu_R", "A2CU recall", "unit"),
    ("qaeval_F1", "QAEval F1", "unit"), ("questeval", "QuestEval", "unit"), ("questeval_R", "QuestEval recall", "unit"),
    ("unieval_relevance", "UniEval relevance", "unit"), ("unieval_consistency", "UniEval consistency", "unit"),
    ("unieval_coherence", "UniEval coherence", "unit"), ("unieval_overall", "UniEval overall", "unit"),
    ("geval_completeness", "G-Eval (completeness prompt)", "llm"), ("gevalplus_completeness", "G-Eval+ (completeness prompt)", "llm"),
    ("geval_faithfulness", "G-Eval (faithfulness prompt)", "llm"), ("gevalplus_faithfulness", "G-Eval+ (faithfulness prompt)", "llm"),
]
COLS = [("unisumeval", "completeness", "compl."), ("unisumeval", "faithfulness", "faith."), ("unisumeval", "conciseness", "conc."),
        ("summeval", "relevance", "relev."), ("summeval", "consistency", "consist."), ("summeval", "coherence", "coher."), ("summeval", "fluency", "fluen."),
        ("rose", "acu_recall", "ACU recall")]
CONTROLS = [("marsR_marsc", "marsR_marsc_shufcand", "marsR_marsc_nocand", "MARS-R (coverage verifier)"),
            ("marsR_factcg", "marsR_factcg_shufcand", "marsR_factcg_nocand", "MARS-R (FactCG verifier)"),
            ("marsR_free", "marsR_free_shufcand", "marsR_free_nocand", "MARS-R (reference-free)"),
            ("marsP_factcg", "marsP_factcg_shufsrc", None, "MARS-P (FactCG verifier)")]


def cell(rep, ds, dim, col, best: float | None = None) -> str:
    e = rep["datasets"].get(ds, {}).get("dims", {}).get(dim, {}).get(col)
    if not e:
        return r"\pend"
    t = e["summary_tau"]["observed"]
    if t != t:
        return r"\pend"
    s = f"{t:.3f}"
    return r"$\mathbf{" + s + "}$" if best is not None and abs(t - best) < 1e-9 else f"${s}$"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-controls", required=True)
    a = ap.parse_args()
    rep = json.load(open(a.meta))
    best = {}
    for ds, dim, _ in COLS:
        vals = [rep["datasets"].get(ds, {}).get("dims", {}).get(dim, {}).get(c, {}).get("summary_tau", {}).get("observed") for c, _, _ in ROWS]
        vals = [v for v in vals if v is not None and v == v]
        best[(ds, dim)] = max(vals) if vals else None
    lines, prev = [], None
    for col, label, blk in ROWS:
        if blk != prev and prev is not None:
            lines.append(r"\midrule")
        prev = blk
        lines.append(label + " & " + " & ".join(cell(rep, ds, dim, col, best[(ds, dim)]) for ds, dim, _ in COLS) + r" \\")
    n = {ds: rep["datasets"][ds] for ds in ("unisumeval", "summeval", "rose") if ds in rep["datasets"]}
    head = " & ".join(f"{lab}" for _, _, lab in COLS)
    tex = r"""% M-A meta-evaluation, rendered by research/mars_metric_20260923/code/ma_table.py from ma/meta/meta_evaluation.json.
\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lccc cccc c@{}}
\toprule
& \multicolumn{3}{c}{\textbf{UniSumEval}} & \multicolumn{4}{c}{\textbf{SummEval}} & \textbf{RoSE} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-8}\cmidrule(lr){9-9}
\textbf{Metric} & """ + head + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption{\textbf{Agreement with human judgments, summary level.} Kendall $\tau$ between each metric and each
human dimension within a document across its systems, averaged over documents (UniSumEval: """ + f"{n['unisumeval']['n_docs']}" + r"""
documents $\times$ up to """ + f"{n['unisumeval']['n_systems']}" + r""" systems; SummEval: """ + f"{n['summeval']['n_docs']}" + r""" $\times$ 16; RoSE test: """ + f"{n['rose']['n_docs']}" + r""" documents, human
ACU recall). Reference-based metrics use the reference (maximum over references where several exist); MARS-R uses the
reference's atomic facts, on RoSE the human ACUs themselves; MARS-P and the source-side comparators read the source.
The fourth block holds the learned unit- and question-based evaluators (A3CU and A2CU: reference ACUs; QAEval:
questions from the reference; QuestEval: questions from the source; UniEval: a boolean-QA evaluator). The G-Eval
rows are the GPT-4 judge scores UniSumEval ships. Bold marks the best metric per column. Intervals and the
pre-specified contrasts are in the text (document-clustered bootstrap, $2{,}000$ draws).}
\label{tab:mars_meta}
\end{table*}
"""
    Path(a.out).write_text(tex)
    # controls table, transposed: one row per (dataset, dimension), one column per MARS reading
    clines, prevds = [], None
    for ds, dim, lab in COLS:
        if prevds is not None and ds != prevds:
            clines.append(r"\midrule")
        prevds = ds
        dsn = {"unisumeval": "UniSumEval", "summeval": "SummEval", "rose": "RoSE"}[ds]
        row = [f"{dsn} {lab}"]
        for real, shuf, nocand, _ in CONTROLS:
            r_ = cell(rep, ds, dim, real)
            if r_ == r"\pend":
                row.append(r"\pend"); continue
            s_ = cell(rep, ds, dim, shuf)
            row.append(f"{r_} / {s_}")
        clines.append(" & ".join(row) + r" \\")
    head2 = " & ".join(r"\textbf{" + lab + "}" for _, _, _, lab in CONTROLS)
    ctex = r"""% M-A controls, rendered by research/mars_metric_20260923/code/ma_table.py.
\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{human dimension} & """ + head2 + r""" \\
\midrule
""" + "\n".join(clines) + r"""
\bottomrule
\end{tabular}
\caption{\textbf{What the metric reads.} Summary-level $\tau$ of each MARS score with the real candidate / with the
candidate replaced by another document's candidate of the same system; for MARS-P the control replaces the source by
another document's source. A score computed with no candidate at all is constant within a document, so it cannot
rank the systems of a document and is not shown. A shuffled-candidate score that still correlates measures what
the systems produce in general, not this summary; the difference to the real value is what the metric reads from
the candidate.}
\label{tab:mars_meta_controls}
\end{table*}
"""
    Path(a.out_controls).write_text(ctex)
    print(f"wrote {a.out} and {a.out_controls}")


if __name__ == "__main__":
    main()
