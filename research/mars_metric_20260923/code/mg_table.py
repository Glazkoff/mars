#!/usr/bin/env python3
"""M-G -- write the metric table (Table 3 of the IEEE Access paper) from an ma_meta.py report, same rows and
columns as the published available-case table, so the common-cohort reading can replace it one for one.

  python mg_table.py --meta meta_common.json --out tables/mars_meta.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

COLS = [("unisumeval", "completeness"), ("unisumeval", "faithfulness"), ("unisumeval", "conciseness"),
        ("summeval", "relevance"), ("summeval", "consistency"), ("summeval", "coherence"), ("summeval", "fluency"),
        ("rose", "acu_recall")]
BLOCKS = [
    [(r"\textbf{MARS-R}, coverage verifier (ours)", "marsR_marsc"), (r"\quad RoSE with units extracted from the reference", "marsR_marsc_autogemma"),
     (r"\textbf{MARS-R}, FactCG verifier", "marsR_factcg"),
     (r"\textbf{MARS-R}, reference-free (source facts)", "marsR_free"), (r"\textbf{MARS-P}, FactCG verifier", "marsP_factcg")],
    [("ROUGE-1", "rouge1"), ("ROUGE-2", "rouge2"), ("ROUGE-L", "rougeL"), ("BERTScore P", "bertscore_P"),
     ("BERTScore R", "bertscore_R"), ("BERTScore F", "bertscore_F"), (r"BARTScore ref$\to$cand", "bartscore_ref2cand")],
    [(r"BARTScore src$\to$cand", "bartscore_src2cand"), ("AlignScore", "alignscore"), ("FactCG, whole summary", "factcg_whole")],
    [("A3CU recall", "a3cu_R"), ("A3CU F", "a3cu_F"), ("A2CU recall", "a2cu_R"), ("QAEval F1", "qaeval_F1"),
     ("QuestEval", "questeval"), ("QuestEval recall", "questeval_R"), ("UniEval relevance", "unieval_relevance"),
     ("UniEval consistency", "unieval_consistency"), ("UniEval coherence", "unieval_coherence"), ("UniEval overall", "unieval_overall")],
    [("G-Eval (completeness prompt)", "geval_completeness"), ("G-Eval+ (completeness prompt)", "gevalplus_completeness"),
     ("G-Eval (faithfulness prompt)", "geval_faithfulness"), ("G-Eval+ (faithfulness prompt)", "gevalplus_faithfulness")],
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = json.loads(Path(a.meta).read_text())
    val = {}
    for ds, dim in COLS:
        e = r["datasets"][ds]["dims"][dim]
        for _, rows in enumerate(BLOCKS):
            for _, col in rows:
                if col in e:
                    val[(col, ds, dim)] = e[col]["summary_tau"]["observed"]
    best = {c: max((v for (col, *k), v in val.items() if tuple(k) == c), default=None) for c in COLS}
    fmt = lambda x: "$-0.000$" if x is not None and -0.0005 < x < 0 else f"${x:.3f}$"
    lines = []
    for bi, rows in enumerate(BLOCKS):
        if bi:
            lines.append(r"\midrule")
        for name, col in rows:
            cells = []
            for c in COLS:
                v = val.get((col,) + c)
                if v is None:
                    cells.append(r"\pend")
                else:
                    s = fmt(v)
                    cells.append(r"$\mathbf{" + s.strip("$") + "}$" if best[c] is not None and abs(v - best[c]) < 1e-12 else s)
            lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    cc = {ds: r["datasets"][ds]["common_cohort"] for ds in ("unisumeval", "summeval", "rose")}
    n = lambda d: {k: (f"{v:,}".replace(",", "{,}") if isinstance(v, int) else v) for k, v in d.items()}
    u = n(cc["unisumeval"]["completeness"]); s = n(cc["summeval"]["relevance"]); ro = n(cc["rose"]["acu_recall"])
    body = "\n".join(lines)
    tex = rf"""\begin{{table*}}[t]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{@{{}}lccc cccc c@{{}}}}
\toprule
& \multicolumn{{3}}{{c}}{{\textbf{{UniSumEval}}}} & \multicolumn{{4}}{{c}}{{\textbf{{SummEval}}}} & \textbf{{RoSE}} \\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-8}}\cmidrule(lr){{9-9}}
\textbf{{Metric}} & compl. & faith. & conc. & relev. & consist. & coher. & fluen. & ACU recall \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{\textbf{{Agreement with human judgments, summary level, on one common population per column.}} Kendall
$\tau$ between each metric and each human dimension within a document across its systems, averaged over documents.
Every column uses the same pairs and documents for every metric: the pairs every listed metric scores, then the
documents on which every metric has a defined $\tau$ (UniSumEval completeness ${u['n_pairs']}$ pairs over
${u['n_docs_all_tau']}$ documents, the reference-based metrics requiring a reference; SummEval relevance
${s['n_pairs']}$ pairs over ${s['n_docs_all_tau']}$ documents; RoSE test ${ro['n_pairs']}$ pairs over ${ro['n_docs_all_tau']}$
documents); per-column counts and the available-case values, where each metric keeps its own pairs, are in
Table~\ref{{tab:mars_meta_available}}. Reference-based metrics use the reference (maximum over references where several
exist); reference-based MARS-R uses the reference's atomic facts, on RoSE the human ACUs themselves (an oracle-unit
reading, in-domain for the MARS-C verifier, which was trained on RoSE training labels; the indented row replaces them with
facts the frozen decomposer extracts from the reference, as on the other two datasets), and the reference-free row the
source's salient facts; MARS-P and the source-side comparators read the source. The fourth block holds the learned unit-
and question-based evaluators; the G-Eval rows are the GPT-4 judge scores UniSumEval ships. Bold marks the best metric per
column. Contrasts are in the text (document-clustered bootstrap over the same documents, $2{{,}}000$ draws).}}
\label{{tab:mars_meta}}
\end{{table*}}
"""
    Path(a.out).write_text(tex)
    print(f"[mg-table] {a.out}")


if __name__ == "__main__":
    main()
