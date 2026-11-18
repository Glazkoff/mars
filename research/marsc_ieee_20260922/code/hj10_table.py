#!/usr/bin/env python3
"""H-J10a -- render tables/marsc_omb.tex from hj10/ artifacts: crossed accuracy on OmissionBench for every
family under the as-deployed and the windowed candidate readings, the benchmark's own paired discrimination
against the clean twin (windowed cells), and the windowed MARS-C-minus-row margin with its one-sided lower
bound. Nothing is computed here beyond formatting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWS = [
    ("humanfact", r"\textbf{MARS-C human-fact (ours)}", "reg"),
    ("judgefact", "MARS-C judge-label twin", "reg"),
    ("nli", "windowed RoBERTa-large-MNLI", "reg"),
    ("summac_zs", "SummaC-ZS", "reg"),
    ("lex_token_recall", "lexical token-recall", "reg"),
    ("lex_rougeL_recall", "lexical ROUGE-L recall", "reg"),
    ("emb_minilm", "MiniLM cosine", "reg"),
    ("judge_gemma31", "Gemma-4-31B zero-shot judge", "ext"),
    ("minicheck7b", "Bespoke-MiniCheck-7B", "ext"),
    ("factcg", "FactCG-DeBERTa-v3-L", "ext"),
    ("hj9factcghuman", "human-fact package on the FactCG encoder (H-J9)", "hj9"),
    ("hj9factcgjudge", "judge-label twin on the FactCG encoder (H-J9)", "hj9"),
]
BLOCK = {"reg": r"\emph{the registered comparator family}",
         "ext": r"\emph{added after registration: the 2026 LLM-AggreFact leaders and a zero-shot 31B judge}",
         "hj9": r"\emph{the natural-coverage package on the stronger encoder}"}


def acc(fam):
    s = fam["seed_averaged"]; lo, hi = s["ci_doc_bootstrap"]
    return f"${s['discrimination_doc_macro']:.3f}$ $[{lo:.3f},{hi:.3f}]$"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hj10-dir", required=True)
    ap.add_argument("--set", default="complete")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.hj10_dir)
    dep = json.load(open(root / f"hcond_{a.set}_ext" / "crossed_families.json"))
    win = json.load(open(root / f"hcond_{a.set}_w_ext" / "crossed_families.json"))
    twin = json.load(open(root / f"twin_{a.set}_w.json"))
    lines, prev = [], None
    for key, label, blk in ROWS:
        if key not in win["families"]:
            continue
        if blk != prev:
            if prev is not None:
                lines.append(r"\midrule")
            lines.append(r"\multicolumn{5}{@{}l}{" + BLOCK[blk] + r"}\\")
            prev = blk
        d = acc(dep["families"][key]) if key in dep["families"] else r"\pend"
        w = acc(win["families"][key])
        t = twin["families"].get(key)
        tw = (f"${t['paired_discrimination']:.3f}$ $[{t['ci95_consultation_bootstrap'][0]:.3f},"
              f"{t['ci95_consultation_bootstrap'][1]:.3f}]$") if t else r"\pend"
        m = win["margins_vs_primary"].get(key)
        mg = (f"${m['observed_margin_doc_macro']:+.3f}\\ ({m['lower_bound_95_one_sided']:+.3f})$") if m else r"\pend"
        lines.append(f"{label} & {d} & {w} & {tw} & {mg} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\emph{any summary-blind scorer} & $0.500$ & $0.500$ & $0.500$ & \pend \\")
    n_tr, n_doc = win["triples"], win["docs_with_triples"]
    tex = r"""% H-J10a: OmissionBench, rendered by research/marsc_ieee_20260922/code/hj10_table.py from hj10/ artifacts.
\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccc@{}}
\toprule
& \multicolumn{2}{c}{\textbf{crossed accuracy (95\% CI)}} & \textbf{vs.\ clean twin} & \textbf{MARS-C minus row} \\
\cmidrule(lr){2-3}
\textbf{verifier family} & as deployed & windowed & windowed (95\% CI) & windowed (LB$_{95}$) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption{\textbf{The same-fact crossed endpoint on a clinical pool of another genre.} OmissionBench's
""" + f"{twin['n_pairs']}" + r""" complete-omission pairs over """ + f"{twin['n_consultations']}" + r""" synthetic consultations: the fact is the statement the
benchmark removed, the clean twin states it, the errored note omits it (""" + f"{n_tr}" + r""" same-fact triples over
""" + f"{n_doc}" + r""" consultations). \emph{As deployed} reads the first $380$ words of a note, which half of these notes
exceed; \emph{windowed} scores every overlapping $380$-word window and keeps the most conveying one. The
third column is the benchmark's own statistic, the errored note against its clean twin, given the removed
fact. Registered bar on the windowed reading (\S\ref{sec:marsc_conditioning}); the two lower blocks were
added after registration and the registered verdict is not restated over them.}
\label{tab:marsc_omb}
\end{table}
"""
    Path(a.out).write_text(tex)
    print(f"wrote {a.out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
