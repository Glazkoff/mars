#!/usr/bin/env python3
"""H-J8 -- the MARS-C figures for the IEEE Access manuscript, drawn from numbers already in the paper's tables.

Every number below is copied from articles/iclr2027/tables/marsc_{main,conditioning,blindcontrols}.tex,
research/marsc_strengthen_20260918/RESULTS_PUBPLAN_20260921.md Sec. 4-5 (the b9/b10 rows) and the C4
attribution artifacts (passed in as --c4 JSON). No figure computes a new result.

  F-marsc-pipeline.pdf    Fig. 1  the pipeline, the two supervision packages, the two endpoints, the controls
  F-marsc-endpoints.pdf   Fig. 2  recall@10 against omission precision, every verifier family, with the
                                  summary-blind controls drawn as arrows (held-out RoSE test split)
  F-marsc-conditioning.pdf Fig. 3 the crossed candidate-conditioning endpoint (floor 0.500) and the
                                  overlap-residualised reading
  F-marsc-attribution.pdf Fig. 4  where the misses live: inventory / verifier / rule, per pool
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
C_OURS, C_JUDGE, C_OTS, C_2026, C_GREY = "#1f4e8c", "#c0504d", "#7f7f7f", "#2e8b57", "#b0b0b0"


# ----------------------------------------------------------------------------- Fig. 2
# (name, inventory, real recall, real precision, shuffled recall, shuffled precision, empty recall, empty precision, class)
ENDPOINTS = [
    ("MARS-C human-fact (ours)", "Gemma+ent", 0.377, 0.720, 0.440, 0.666, 0.458, 0.655, "ours"),
    ("MARS-C judge-label twin", "Gemma+ent", 0.255, 0.692, None, None, None, None, "judge"),
    ("Gemma-4-31B judge", "Gemma+ent", 0.338, 0.720, 0.374, 0.661, 0.417, 0.664, "2026"),
    ("Bespoke-MiniCheck-7B", "Gemma+ent", 0.348, 0.717, 0.417, 0.651, 0.390, 0.656, "2026"),
    ("FactCG-DeBERTa-v3-L", "Gemma+ent", 0.377, 0.711, 0.419, 0.636, 0.420, 0.652, "2026"),
    ("windowed RoBERTa-MNLI", "Gemma+ent", 0.359, 0.718, 0.353, 0.645, 0.339, 0.650, "ots"),
    ("SummaC-ZS", "Gemma+ent", 0.367, 0.680, 0.344, 0.639, 0.387, 0.648, "ots"),
    ("MiniCheck (Flan-T5)", "Gemma+ent", 0.357, 0.709, 0.412, 0.661, 0.394, 0.650, "ots"),
    ("AlignScore", "Gemma+ent", 0.354, 0.711, 0.350, 0.650, 0.368, 0.657, "ots"),
    ("lexical token-recall", "Gemma+ent", 0.321, 0.725, 0.459, 0.658, 0.477, 0.647, "ots"),
    ("lexical ROUGE-L", "Gemma+ent", 0.320, 0.724, 0.373, 0.654, 0.477, 0.647, "ots"),
    ("MiniLM cosine", "Gemma+ent", 0.319, 0.720, 0.384, 0.654, 0.413, 0.663, "ots"),
]


def fig_endpoints(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    colors = {"ours": C_OURS, "judge": C_JUDGE, "2026": C_2026, "ots": C_OTS}
    for name, inv, r, p, rs, ps, re_, pe, cls in ENDPOINTS:
        c = colors[cls]
        for (r2, p2, style) in ((rs, ps, "--"), (re_, pe, ":")):
            if r2 is None:
                continue
            ax.annotate("", xy=(r2, p2), xytext=(r, p),
                        arrowprops=dict(arrowstyle="->", color=c, lw=0.7, ls=style, alpha=0.75, shrinkA=2, shrinkB=1))
        ax.scatter([r], [p], s=34 if cls == "ours" else 18, color=c, zorder=5,
                   marker="*" if cls == "ours" else ("s" if cls == "2026" else "o"), edgecolor="k", linewidth=0.3)
    labels = {"MARS-C human-fact (ours)": (0.380, 0.7245, "left"), "MARS-C judge-label twin": (0.258, 0.6905, "left"),
              "Gemma-4-31B judge": (0.336, 0.7235, "right"), "Bespoke-MiniCheck-7B": (0.346, 0.7135, "right"),
              "FactCG-DeBERTa-v3-L": (0.380, 0.7085, "left"), "windowed RoBERTa-MNLI": (0.361, 0.7165, "left"),
              "SummaC-ZS": (0.369, 0.678, "left"), "lexical token-recall / ROUGE-L": (0.318, 0.7275, "right"),
              "MiniLM cosine": (0.317, 0.7175, "right"), "AlignScore": (0.352, 0.7075, "right"), "MiniCheck (Flan-T5)": (0.355, 0.7035, "right")}
    for name, (x, y, ha) in labels.items():
        ax.text(x, y, name, fontsize=6, ha=ha, va="center")
    ax.set_xlabel("recall@10 of human-marked omissions (RoSE held-out split)")
    ax.set_ylabel("omission precision among ACU-aligned emissions")
    ax.set_xlim(0.24, 0.49); ax.set_ylim(0.63, 0.738)
    ax.grid(True, lw=0.3, alpha=0.4)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="*", color=C_OURS, ls="", ms=8, label="MARS-C (ours)"),
               Line2D([], [], marker="o", color=C_JUDGE, ls="", ms=4, label="judge-label twin"),
               Line2D([], [], marker="s", color=C_2026, ls="", ms=4, label="2026 comparators"),
               Line2D([], [], marker="o", color=C_OTS, ls="", ms=4, label="off-the-shelf verifiers"),
               Line2D([], [], color="k", ls="--", lw=0.7, label="→ another document's summary"),
               Line2D([], [], color="k", ls=":", lw=0.7, label="→ no summary")]
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.9, borderpad=0.4, labelspacing=0.25)
    ax.text(0.245, 0.636, "each arrow: the same family rescored with another\ndocument's summary (dashed) or with none (dotted):\nrecall rises, precision falls", fontsize=6.2, va="bottom", color="#333333")
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "F-marsc-endpoints.pdf", bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 3
COND = [  # (name, TEST doc-macro, lo, hi, block, UniSumEval doc-macro, lo, hi)
    ("MARS-C human-fact (ours)", 0.914, 0.898, 0.929, "reg", 0.864, 0.853, 0.875),
    ("windowed RoBERTa-MNLI", 0.900, 0.883, 0.915, "reg", 0.797, 0.782, 0.811),
    ("MARS-C judge-label twin", 0.893, 0.875, 0.909, "reg", 0.869, 0.860, 0.879),
    ("SummaC-ZS", 0.855, 0.834, 0.875, "reg", 0.732, 0.715, 0.750),
    ("lexical token-recall", 0.828, 0.809, 0.846, "reg", 0.847, 0.837, 0.857),
    ("lexical ROUGE-L recall", 0.798, 0.776, 0.820, "reg", 0.816, 0.805, 0.827),
    ("MiniLM cosine", 0.785, 0.759, 0.809, "reg", 0.821, 0.810, 0.833),
    ("Gemma-4-31B judge", 0.903, 0.886, 0.917, "post", 0.859, 0.847, 0.870),
    ("Bespoke-MiniCheck-7B", 0.900, 0.885, 0.915, "post", 0.839, 0.825, 0.852),
    ("FactCG-DeBERTa-v3-L", 0.888, 0.870, 0.904, "post", 0.833, 0.819, 0.846),
]


def fig_conditioning(out: Path) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [2.4, 1.0], "wspace": 0.42})
    names = [c[0] for c in COND]; vals = [c[1] for c in COND]
    y = list(range(len(COND)))[::-1]
    for yi, (name, v, lo, hi, blk, u, ulo, uhi) in zip(y, COND):
        col = C_OURS if "ours" in name else (C_JUDGE if "twin" in name else (C_2026 if blk == "post" else C_OTS))
        ax.barh(yi, v - 0.5, left=0.5, color=col, height=0.62, alpha=0.9)
        ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="none", ecolor="k", elinewidth=0.7, capsize=2)
        ax.errorbar(u, yi - 0.02, xerr=[[u - ulo], [uhi - u]], fmt="D", ms=3.2, mfc="white", mec="k", mew=0.7,
                    ecolor="k", elinewidth=0.5, capsize=1.5, zorder=6)
        ax.text(0.505, yi, name, va="center", ha="left", fontsize=6.5, color="white" if v > 0.62 else "k")
    ax.axvline(0.5, color="k", lw=1.0)
    ax.text(0.507, -0.45, "any summary-blind scorer: exactly 0.500", fontsize=6.5, ha="left", va="center", color="k")
    ax.axhline(2.5, color="k", lw=0.5, ls="--")
    ax.text(0.955, 2.62, "registered\nfamily", fontsize=6, ha="left", va="bottom", style="italic", clip_on=False)
    ax.text(0.955, 2.38, "added after the\nsplit was read", fontsize=6, ha="left", va="top", style="italic", clip_on=False)
    ax.set_yticks([]); ax.set_xlim(0.5, 0.95); ax.set_ylim(-0.6, len(COND) - 0.3)
    ax.set_xlabel("same-fact crossed accuracy (bars: RoSE held-out split; diamonds: UniSumEval)", loc="left")
    ax.grid(True, axis="x", lw=0.3, alpha=0.4)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="D", ms=3.2, mfc="white", mec="k", ls="", label="UniSumEval, external pool (95% CI)")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=6, borderaxespad=0.1)
    # residualised reading
    xs = [0, 1]; ours = [0.914, 0.881]; twin = [0.893, 0.667]
    ours_u = [0.864, 0.726]; twin_u = [0.869, 0.796]
    ax2.plot(xs, ours, "-o", color=C_OURS, ms=5, label="ours, RoSE held-out")
    ax2.plot(xs, twin, "-o", color=C_JUDGE, ms=5, label="twin, RoSE held-out")
    ax2.plot(xs, ours_u, "--D", color=C_OURS, ms=4, mfc="white", label="ours, UniSumEval")
    ax2.plot(xs, twin_u, "--D", color=C_JUDGE, ms=4, mfc="white", label="twin, UniSumEval")
    ax2.text(1.04, 0.726, "0.726", fontsize=6, color=C_OURS, va="center")
    ax2.text(1.04, 0.796, "0.796", fontsize=6, color=C_JUDGE, va="center")
    ax2.axhline(0.5, color="k", lw=1.0)
    ax2.text(0.02, 0.51, "summary-blind floor", fontsize=6, va="bottom")
    ax2.set_xticks(xs); ax2.set_xticklabels(["raw score", "residualised on\nfact–summary overlap"])
    ax2.set_ylim(0.45, 0.95); ax2.set_xlim(-0.3, 1.45)
    ax2.set_ylabel("crossed accuracy")
    for xi, v in zip(xs, ours):
        ax2.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.5, color=C_OURS)
    ax2.text(-0.06, 0.893, "0.893", ha="right", va="center", fontsize=6.5, color=C_JUDGE)
    ax2.text(1.0, 0.667 - 0.03, "0.667", ha="center", fontsize=6.5, color=C_JUDGE)
    ax2.legend(loc="center left", bbox_to_anchor=(-0.04, 0.27), frameon=False, fontsize=5.6, labelspacing=0.2)
    ax2.grid(True, axis="y", lw=0.3, alpha=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "F-marsc-conditioning.pdf", bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 4
def fig_attribution(c4: dict, out: Path) -> None:
    pools = [("validation", "RoSE validation"), ("test", "RoSE held-out test"), ("unisum", "UniSumEval")]
    systems = [("humanfact+div1", "human-fact · Gemma+ent (record)"), ("judgefact+div1", "judge-label · Gemma+ent"),
               ("humanfact.distilled+div1", "human-fact · distilled"), ("judgefact.distilled+div1", "judge-label · distilled"),
               ("tagger.spacy+div1", "spaCy tagger (MARS-2 units)")]
    ypos = {s: len(systems) - 1 - i for i, (s, _) in enumerate(systems)}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True, gridspec_kw={"wspace": 0.08})
    cols = {"hit": C_OURS, "inventory": "#d9a441", "verifier": "#c0504d", "rule": "#7f7f7f"}
    for ax, (pool, title) in zip(axes, pools):
        blk = c4[pool]["systems"]
        for s, lab in systems:
            if s not in blk:
                continue
            o = blk[s]["observed"]; left = 0.0; yi = ypos[s]
            for key in ("hit", "inventory", "verifier", "rule"):
                w = o[f"{key}_of_omitted"]
                ax.barh(yi, w, left=left, color=cols[key], height=0.62)
                if w > 0.09:
                    ax.text(left + w / 2, yi, f"{w:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if key in ("hit", "verifier") else "k")
                left += w
        ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(systems) - 0.4)
        ax.set_title(f"{title}, n = {c4[pool]['n_pairs']:,} pairs", fontsize=7)
        ax.tick_params(axis="x", labelsize=6.5)
    axes[0].set_yticks([ypos[s] for s, _ in systems]); axes[0].set_yticklabels([lab for _, lab in systems], fontsize=6.5)
    axes[1].set_xlabel("share of the human-marked omitted ACUs of each pool", fontsize=7, labelpad=2)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=cols[k], label=l) for k, l in
                        (("hit", "recovered in the final top-10"), ("inventory", "miss: no inventory unit overlaps the fact"),
                         ("verifier", "miss: unit ranked outside the budget"), ("rule", "miss: demoted by the emission rule"))],
               loc="lower center", ncol=2, frameon=False, fontsize=6.3, bbox_to_anchor=(0.5, -0.2))
    fig.savefig(out / "F-marsc-attribution.pdf", bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 1
def box(ax, x, y, w, h, text, fc="#f2f2f2", ec="k", fs: float = 7, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, text="", ls="-", color="k"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.8, ls=ls, color=color))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.03, text, ha="center", va="bottom", fontsize=6, color=color)


def fig_pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.35)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.3); ax.axis("off")
    box(ax, 0.15, 1.85, 1.35, 0.85, "source\ndocument", fc="#ffffff")
    box(ax, 0.15, 0.45, 1.35, 0.85, "candidate\nsummary", fc="#ffffff")
    box(ax, 2.0, 1.85, 1.9, 0.85, "fact inventory\nGemma+ent (94 units/pair)\nor distilled (40)", fc="#eef3fa", fs=6.5)
    arrow(ax, 1.5, 2.27, 2.0, 2.27)
    box(ax, 4.4, 1.55, 2.05, 1.45, "verifier\nDeBERTa-large cross-encoder\n(summary, fact) → P(omitted)\n0.4B parameters", fc="#eef3fa", fs=6.5)
    arrow(ax, 3.9, 2.27, 4.4, 2.5, "each fact")
    arrow(ax, 1.5, 0.87, 4.4, 1.9, "premise")
    box(ax, 6.95, 1.85, 1.55, 0.85, "emission rule\none unit per sentence,\ntop-k", fc="#eef3fa", fs=6.5)
    arrow(ax, 6.45, 2.27, 6.95, 2.27)
    box(ax, 8.95, 1.85, 0.95, 0.85, "k facts\nlikely\nomitted", fc="#fff5e6", fs=6.5, bold=True)
    arrow(ax, 8.5, 2.27, 8.95, 2.27)
    # supervision packages
    box(ax, 4.4, 0.15, 0.95, 0.95, "human-fact\npackage\n16,332 cells\nRoSE ACUs", fc="#dde8f5", fs=5.8)
    box(ax, 5.5, 0.15, 0.95, 0.95, "judge-label\npackage\n85,600 cells\nLLM units", fc="#f5dede", fs=5.8)
    arrow(ax, 4.87, 1.1, 5.1, 1.55); arrow(ax, 5.97, 1.1, 5.75, 1.55)
    ax.text(5.42, 0.02, "supervision (under test)", ha="center", va="bottom", fontsize=6, style="italic")
    # controls and endpoints
    box(ax, 2.0, 0.15, 1.9, 0.95, "summary-blind controls\nanother document's summary /\nno summary at all", fc="#f7f7f7", fs=5.8)
    arrow(ax, 2.95, 1.1, 2.95, 1.85, ls="--", color="#555555")
    box(ax, 6.95, 0.15, 2.95, 1.2, "endpoints\nrecall@10 of human-marked omissions;\nomission precision among aligned emissions;\nsame-fact crossed accuracy (summary-blind floor 0.500)",
        fc="#f7f7f7", fs=5.8)
    arrow(ax, 9.42, 1.85, 9.42, 1.35, ls="--", color="#555555")
    fig.tight_layout(pad=0.1)
    fig.savefig(out / "F-marsc-pipeline.pdf", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c4", required=True, help="JSON with the C4 attribution block (from fetch_a3_c4.py)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    c4 = json.load(open(a.c4))["c4"]
    fig_pipeline(out); fig_endpoints(out); fig_conditioning(out); fig_attribution(c4, out)
    for f in ("F-marsc-pipeline.pdf", "F-marsc-endpoints.pdf", "F-marsc-conditioning.pdf", "F-marsc-attribution.pdf"):
        print("wrote", out / f, (out / f).stat().st_size, "bytes")


if __name__ == "__main__":
    main()
