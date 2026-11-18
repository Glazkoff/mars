#!/usr/bin/env python3
"""H-J8 (extension) -- F-marsc-domains.pdf: UniSumEval recall@10 and omission precision per domain, for the
pipeline of record, its judge-label twin and the three post-registration comparators, from the frozen
evaluator's per-pair dump (hj3/eval_unisum/per_pair_unisum_k10.jsonl; seed-averaged recall per pair, hit /
false-alert counts among the ten emissions). Intervals are document-clustered percentile bootstraps
(2,000 draws, seed 20260918). No number is recomputed by a different estimator: the per-domain means of the
systems reproduce the pool totals of the comparator table (consensus regime of record: MARS-C 0.347,
judge-label twin 0.298, FactCG 0.384, MiniCheck-7B 0.384, judge 0.370), which the script asserts. The
appendix's per-domain list is the seed-average regime and is a different, also published, reading.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
C_OURS, C_JUDGE, C_2026A, C_2026B, C_2026C = "#1f4e8c", "#c0504d", "#2e8b57", "#8fbc8f", "#e0a53a"
SYSTEMS = [("humanfact+div1", "MARS-C human-fact (ours)", C_OURS, "*", 46),
           ("judgefact+div1", "MARS-C judge-label twin", C_JUDGE, "o", 20),
           ("nli_factcg_real+div1", "FactCG-DeBERTa-v3-L", C_2026A, "s", 18),
           ("nli_minicheck7b_real+div1", "Bespoke-MiniCheck-7B", C_2026B, "s", 18),
           ("nli_judge_gemma31_real+div1", "Gemma-4-31B judge", C_2026C, "s", 18)]
NAMES = {"unisum_CNNDM": "CNN/DM", "unisum_wikihow": "WikiHow", "unisum_MultiWOZ": "MultiWOZ",
         "unisum_dialogsum": "DialogSum", "unisum_Pubmed": "PubMed", "unisum_MeetingBank": "MeetingBank",
         "unisum_mediasum": "MediaSum", "unisum_GovReport": "GovReport", "unisum_SQuALITY": "SQuALITY"}
TOTALS = {"humanfact+div1": 0.347, "judgefact+div1": 0.298, "nli_factcg_real+div1": 0.384,
          "nli_minicheck7b_real+div1": 0.384, "nli_judge_gemma31_real+div1": 0.370}


def boot(by_doc: dict, stat, rng, n=2000):
    keys = sorted(by_doc)
    obs = stat([by_doc[k] for k in keys])
    draws = [stat([by_doc[keys[i]] for i in rng.integers(0, len(keys), size=len(keys))]) for _ in range(n)]
    return obs, np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def mean_recall(groups):
    v = [x["recall"] for g in groups for x in g if x["recall"] is not None]  # pairs with no omitted ACU carry no recall
    return float(np.mean(v))


def precision(groups):
    h = sum(x["hit"] * x["n_emitted"] for g in groups for x in g)
    f = sum(x["false_alert"] * x["n_emitted"] for g in groups for x in g)
    return h / (h + f) if h + f else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pair", required=True)
    ap.add_argument("--out", required=True, help="Figures directory")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.per_pair, encoding="utf-8")]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # system -> domain -> doc -> rows
    for r in rows:
        if r["system"] in dict((s[0], 1) for s in SYSTEMS):
            data[r["system"]][NAMES[r["resource"]]][r["doc"]].append(r)
    stats = {}
    for sysk, *_ in SYSTEMS:
        for dom in NAMES.values():
            rng = np.random.default_rng(20260918)
            rec = boot(data[sysk][dom], mean_recall, rng)
            rng = np.random.default_rng(20260918)
            prec = boot(data[sysk][dom], precision, rng)
            stats[(sysk, dom)] = (rec, prec, sum(len(v) for v in data[sysk][dom].values()))
    for sysk, tot in TOTALS.items():
        got = float(np.mean([x["recall"] for dom in NAMES.values() for g in data[sysk][dom].values()
                             for x in g if x["recall"] is not None]))
        assert abs(got - tot) < 0.0015, (sysk, got, tot)
    order = sorted(NAMES.values(), key=lambda d: -stats[("humanfact+div1", d)][0][0])
    n_pairs = {d: stats[("humanfact+div1", d)][2] for d in order}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"wspace": 0.28})
    x = np.arange(len(order))
    offs = np.linspace(-0.3, 0.3, len(SYSTEMS))
    for ax, which, ylabel in ((axes[0], 0, "recall@10 of human-marked omissions"),
                              (axes[1], 1, "ACU-aligned omission precision")):
        for (sysk, label, col, mk, sz), off in zip(SYSTEMS, offs):
            ys = [stats[(sysk, d)][which][0] for d in order]
            lo = [stats[(sysk, d)][which][0] - stats[(sysk, d)][which][1] for d in order]
            hi = [stats[(sysk, d)][which][2] - stats[(sysk, d)][which][0] for d in order]
            ax.errorbar(x + off, ys, yerr=[lo, hi], fmt="none", ecolor=col, elinewidth=0.6, capsize=0, alpha=0.8)
            ax.scatter(x + off, ys, s=sz, color=col, marker=mk, edgecolor="k", linewidth=0.3, zorder=5,
                       label=label if which == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d} ({n_pairs[d]})" for d in order], rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        ax.set_axisbelow(True)
        for xi in x[:-1]:
            ax.axvline(xi + 0.5, color="#dddddd", lw=0.4)
    axes[0].set_ylim(-0.02, 1.0)
    axes[1].set_ylim(0.2, 1.0)
    h, l = axes[0].get_legend_handles_labels()
    axes[1].legend(h, l, loc="lower right", frameon=False, ncol=1, handletextpad=0.3, borderpad=0.2)
    axes[0].set_title("recall@10, by domain (pairs in brackets)", loc="left")
    axes[1].set_title("omission precision, same emissions", loc="left")
    out = Path(a.out)
    fig.savefig(out / "F-marsc-domains.pdf", bbox_inches="tight")
    fig.savefig(out / "F-marsc-domains.png", bbox_inches="tight", dpi=110)
    summ = {d: {s[1]: {"recall": round(stats[(s[0], d)][0][0], 3), "precision": round(stats[(s[0], d)][1][0], 3)}
                for s in SYSTEMS} for d in order}
    (out / "F-marsc-domains.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({d: {k: v["recall"] for k, v in summ[d].items()} for d in order}, indent=0)[:1500])


if __name__ == "__main__":
    main()
