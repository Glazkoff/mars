#!/usr/bin/env python3
"""The IEEE Access versions of the four hj8 figures, written to articles/ieee_access_marsc/figs/. Same numbers as
hj8_figures.py (imported, not copied); only the drawing changes: labels that overlapped are moved, the
judge-label package is named as such, the block annotations read "pre-specified", and adjacent panels no longer
share a tick label. The C4 attribution input is the three c4_attribution_<pool>.json files.

  python research/marsc_ieee_20260922/code/hj8_figures_ieee.py --c4-dir <dir> --out articles/ieee_access_marsc/figs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hj8_figures as H  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Style of the Figure 1 visual abstract: IBM Plex Sans, navy ink, teal for MARS-C, coral for the judge-label
# package, slate for the rest. The fonts are vendored (SIL OFL) and embedded as TrueType, not Type 3.
FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"
for _ttf in sorted(FONT_DIR.glob("IBMPlexSans-*.ttf")):
    font_manager.fontManager.addfont(str(_ttf))
INK, MUTED, RULE, GRID, PANEL = "#12304F", "#5E7185", "#9AA8B6", "#DCE3EA", "#EEF2F6"
C_OURS, C_JUDGE, C_OTS, C_2026 = "#1F8A8A", "#E0634D", "#8A99A8", "#12304F"
plt.rcParams.update({
    "font.family": "IBM Plex Sans", "font.size": 7.5, "axes.titlesize": 7.5, "axes.titleweight": "semibold",
    "axes.labelsize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.edgecolor": RULE,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "grid.color": GRID, "grid.linewidth": 0.5,
    "grid.alpha": 1.0, "legend.labelcolor": INK, "pdf.fonttype": 42, "ps.fonttype": 42})
PKG = "MARS-C judge-label package"


def fig_endpoints(out: Path) -> None:
    rows = [(PKG if r[0] == "MARS-C judge-label twin" else r[0],) + tuple(r[1:]) for r in H.ENDPOINTS]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    colors = {"ours": C_OURS, "judge": C_JUDGE, "2026": C_2026, "ots": C_OTS}
    for name, inv, r, p, rs, ps, re_, pe, cls in rows:
        c = colors[cls]
        for (r2, p2, style) in ((rs, ps, "--"), (re_, pe, ":")):
            if r2 is None:
                continue
            ax.annotate("", xy=(r2, p2), xytext=(r, p),
                        arrowprops=dict(arrowstyle="->", color=c, lw=0.6, ls=style, alpha=0.6, shrinkA=2, shrinkB=1))
        ax.scatter([r], [p], s=60 if cls == "ours" else 22, color=c, zorder=5,
                   marker="*" if cls == "ours" else ("s" if cls == "2026" else "o"), edgecolor="white", linewidth=0.5)
    # (label, anchor x, anchor y, text x, text y, ha): a thin leader line joins label and point
    labels = [("MARS-C human-fact (ours)", 0.377, 0.720, 0.392, 0.7310, "left"),
              (PKG, 0.255, 0.692, 0.258, 0.6845, "left"),
              ("Gemma-4-31B judge", 0.338, 0.720, 0.345, 0.7345, "left"),
              ("Bespoke-MiniCheck-7B", 0.348, 0.717, 0.300, 0.7050, "right"),
              ("FactCG-DeBERTa-v3-L", 0.377, 0.711, 0.392, 0.7075, "left"),
              ("windowed RoBERTa-MNLI", 0.359, 0.718, 0.392, 0.7195, "left"),
              ("SummaC-ZS", 0.367, 0.680, 0.372, 0.6780, "left"),
              ("lexical token-recall / ROUGE-L", 0.3205, 0.7245, 0.300, 0.7330, "right"),
              ("MiniLM cosine", 0.319, 0.720, 0.300, 0.7190, "right"),
              ("AlignScore", 0.354, 0.711, 0.300, 0.6995, "right"),
              ("MiniCheck (Flan-T5)", 0.357, 0.709, 0.300, 0.6940, "right")]
    for name, x0, y0, x, y, ha in labels:
        ax.annotate(name, xy=(x0, y0), xytext=(x, y), fontsize=6, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color=RULE, lw=0.4, shrinkA=0, shrinkB=3))
    ax.set_xlabel("recall@10 of human-marked omissions (RoSE held-out split)")
    ax.set_ylabel("omission precision among ACU-aligned emissions")
    ax.set_xlim(0.24, 0.49); ax.set_ylim(0.63, 0.742)
    ax.grid(True)
    handles = [Line2D([], [], marker="*", color=C_OURS, ls="", ms=8, label="MARS-C (ours)"),
               Line2D([], [], marker="o", color=C_JUDGE, ls="", ms=4, label="judge-label package"),
               Line2D([], [], marker="s", color=C_2026, ls="", ms=4, label="2026 comparators"),
               Line2D([], [], marker="o", color=C_OTS, ls="", ms=4, label="off-the-shelf verifiers"),
               Line2D([], [], color=MUTED, ls="--", lw=0.7, label="→ another document's summary"),
               Line2D([], [], color=MUTED, ls=":", lw=0.7, label="→ no summary")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, labelspacing=0.3, columnspacing=1.2)
    ax.text(0.245, 0.636, "each arrow: the same family rescored with another\ndocument's summary (dashed) or with none (dotted):\n"
            "recall rises in most families, precision falls in all", fontsize=6.2, va="bottom", color=MUTED)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "F-marsc-endpoints.pdf", bbox_inches="tight"); plt.close(fig)


# H-J12 (2026-09-28): the "twin" row of H.COND is the judge-label PACKAGE (B21); the true label-only twin (R04,
# identical human-ACU cells relabelled by the judge) was scored afterwards and is appended to the post block.
# the judge's TEST value is 0.9025 in every result file (the table of record had rounded it to 0.903)
COND_HJ12 = [(("judge-label package" if "twin" in c[0] else c[0]),) + ((0.902,) if "31B" in c[0] else c[1:2]) + tuple(c[2:])
             for c in H.COND] + [
    ("label-only judge twin", 0.908, 0.891, 0.922, "post", 0.870, 0.860, 0.880),
    # H-J12(c), jobs 7880-7882: the judge-label twin trained on Gemma+ent units of the same documents
    ("unit-matched judge twin", 0.906, 0.891, 0.920, "post", 0.875, 0.865, 0.884)]
# right panel, seed-averaged scores throughout (raw -> residualised on fact-summary overlap), job 7879
RESID = {"ours": ((0.914, 0.891), (0.864, 0.726)), "package": ((0.894, 0.688), (0.869, 0.796)),
         "twin": ((0.908, 0.739), (0.870, 0.819))}


def fig_conditioning(out: Path) -> None:
    COND = COND_HJ12
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.95), gridspec_kw={"width_ratios": [2.4, 1.0], "wspace": 0.28})
    y = list(range(len(COND)))[::-1]
    for yi, (name, v, lo, hi, blk, u, ulo, uhi) in zip(y, COND):
        col = C_OURS if "ours" in name else (C_JUDGE if ("twin" in name or "package" in name) else (C_2026 if blk == "post" else C_OTS))
        ax.barh(yi, v - 0.5, left=0.5, color=col, height=0.62)
        ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, elinewidth=0.7, capsize=2)
        ax.errorbar(u, yi - 0.02, xerr=[[u - ulo], [uhi - u]], fmt="D", ms=3.2, mfc="white", mec=INK, mew=0.7,
                    ecolor=INK, elinewidth=0.5, capsize=1.5, zorder=6)
        ax.text(0.505, yi, name, va="center", ha="left", fontsize=6.5, color="white" if v > 0.62 else INK)
    ax.axvline(0.5, color=INK, lw=1.0)
    div = sum(1 for c in COND if c[4] == "post") - 0.5
    ax.axhline(div, color=RULE, lw=0.6, ls="--")
    ax.text(1.027, div + 0.06, "↑ pre-specified", fontsize=6, ha="right", va="bottom", style="italic")
    ax.text(1.027, div - 0.06, "↓ scored afterwards", fontsize=6, ha="right", va="top", style="italic")
    ax.text(0.507, -0.95, "any summary-blind scorer: exactly 0.500", fontsize=6.5, ha="left", va="center")
    ax.set_yticks([]); ax.set_xlim(0.5, 1.03); ax.set_ylim(-1.2, len(COND) - 0.3)
    ax.set_xlabel("same-fact crossed accuracy (bars: RoSE held-out split; diamonds: UniSumEval)", loc="left")
    ax.grid(True, axis="x")
    ax.legend(handles=[Line2D([], [], marker="D", ms=3.2, mfc="white", mec=INK, ls="", label="UniSumEval, external pool (95% CI)")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=6, borderaxespad=0.1)
    xs = [0, 1]
    style = {"ours": (C_OURS, "ours"), "package": (C_JUDGE, "package"), "twin": (C_2026, "label-only twin")}
    for k, ((r0, r1), (u0, u1)) in RESID.items():
        col, lab = style[k]
        ax2.plot(xs, [r0, r1], "-o", color=col, ms=4.5, label=f"{lab}, RoSE held-out")
        ax2.plot(xs, [u0, u1], "--D", color=col, ms=3.5, mfc="white", label=f"{lab}, UniSumEval")
        ax2.text(1.07, r1, f"{r1:.3f}", fontsize=5.6, color=col, va="center")
        ax2.text(1.07, u1, f"{u1:.3f}", fontsize=5.6, color=col, va="center")
    ax2.axhline(0.5, color=INK, lw=1.0)
    ax2.text(0.5, 0.49, "summary-blind floor", fontsize=6, va="top", ha="center")
    ax2.set_xticks(xs); ax2.set_xticklabels(["raw score", "residualised on\nfact–summary overlap"])
    ax2.set_ylim(0.42, 0.95); ax2.set_xlim(-0.45, 1.4)
    ax2.set_ylabel("crossed accuracy")
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 0.12), frameon=False, fontsize=5.6, labelspacing=0.2, ncol=1)
    ax2.grid(True, axis="y")
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "F-marsc-conditioning.pdf", bbox_inches="tight"); plt.close(fig)


def fig_attribution(c4: dict, out: Path) -> None:
    pools = [("validation", "RoSE validation"), ("test", "RoSE held-out test"), ("unisum", "UniSumEval")]
    systems = [("humanfact+div1", "human-fact · Gemma+ent (record)"), ("judgefact+div1", "judge-label · Gemma+ent"),
               ("humanfact.distilled+div1", "human-fact · distilled"), ("judgefact.distilled+div1", "judge-label · distilled"),
               ("tagger.spacy+div1", "spaCy units (MARS-2)")]
    ypos = {s: len(systems) - 1 - i for i, (s, _) in enumerate(systems)}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True, gridspec_kw={"wspace": 0.12})
    cols = {"hit": C_OURS, "inventory": "#C9D3DD", "verifier": C_JUDGE, "rule": MUTED}
    for j, (ax, (pool, title)) in enumerate(zip(axes, pools)):
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
                            color=INK if key == "inventory" else "white")
                left += w
        ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(systems) - 0.4)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1" if j == len(pools) - 1 else ""])
        ax.set_title(f"{title}, n = {c4[pool]['n_pairs']:,} pairs", fontsize=7)
        ax.tick_params(axis="x", labelsize=6.5)
    axes[0].set_yticks([ypos[s] for s, _ in systems]); axes[0].set_yticklabels([lab for _, lab in systems], fontsize=6.5)
    axes[1].set_xlabel("share of all human-marked omitted ACUs of the pool (pooled over pairs)", fontsize=7, labelpad=2)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=cols[k], label=l) for k, l in
                        (("hit", "recovered in the final top-10"), ("inventory", "miss: no inventory unit overlaps the fact"),
                         ("verifier", "miss: unit ranked outside the budget"), ("rule", "miss: demoted by the emission rule"))],
               loc="lower center", ncol=2, frameon=False, fontsize=6.3, bbox_to_anchor=(0.5, -0.2))
    fig.savefig(out / "F-marsc-attribution.pdf", bbox_inches="tight"); plt.close(fig)


def fig_pipeline(out: Path) -> None:
    box, arrow = H.box, H.arrow
    fig, ax = plt.subplots(figsize=(7.0, 2.35)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.3); ax.axis("off")
    box(ax, 0.15, 1.85, 1.35, 0.85, "source\ndocument", fc="#ffffff")
    box(ax, 0.15, 0.45, 1.35, 0.85, "candidate\nsummary", fc="#ffffff")
    box(ax, 2.0, 1.85, 1.9, 0.85, "fact inventory\nGemma+ent (94 units/pair)\nor distilled (40)", fc="#eef3fa", fs=6.5)
    arrow(ax, 1.5, 2.27, 2.0, 2.27)
    box(ax, 4.4, 1.55, 2.05, 1.45, "verifier\nDeBERTa-large cross-encoder\n(summary, fact) → P(omitted)\n0.4B parameters", fc="#eef3fa", fs=6.5)
    arrow(ax, 3.9, 2.4, 4.4, 2.4)
    ax.text(4.15, 2.75, "each fact", ha="center", va="bottom", fontsize=5.8)
    arrow(ax, 1.5, 1.2, 4.4, 1.75)
    ax.text(3.55, 1.72, "premise", ha="center", va="bottom", fontsize=6)
    box(ax, 6.95, 1.85, 1.55, 0.85, "emission rule\none unit per sentence,\ntop-k", fc="#eef3fa", fs=6.5)
    arrow(ax, 6.45, 2.27, 6.95, 2.27)
    box(ax, 8.95, 1.85, 0.95, 0.85, "k facts\nlikely\nomitted", fc="#fff5e6", fs=6.5, bold=True)
    arrow(ax, 8.5, 2.27, 8.95, 2.27)
    box(ax, 4.4, 0.15, 0.95, 0.95, "human-fact\npackage\n16,332 cells\nRoSE ACUs", fc="#dde8f5", fs=5.8)
    box(ax, 5.5, 0.15, 0.95, 0.95, "judge-label\npackage\n85,600 cells\nLLM units", fc="#f5dede", fs=5.8)
    arrow(ax, 4.87, 1.1, 5.1, 1.55); arrow(ax, 5.97, 1.1, 5.75, 1.55)
    ax.text(5.42, 0.02, "supervision (under test)", ha="center", va="bottom", fontsize=6, style="italic")
    box(ax, 2.0, 0.15, 1.9, 0.8, "summary-blind controls\nanother document's summary /\nno summary at all", fc="#f7f7f7", fs=5.8)
    arrow(ax, 3.9, 0.62, 4.42, 1.6, ls="--", color="#555555")
    box(ax, 6.95, 0.15, 2.95, 1.2, "endpoints\nrecall@10 of human-marked omissions;\nomission precision among aligned emissions;\n"
        "same-fact crossed accuracy\n(summary-blind floor 0.500)", fc="#f7f7f7", fs=5.8)
    arrow(ax, 9.42, 1.85, 9.42, 1.35, ls="--", color="#555555")
    fig.tight_layout(pad=0.1)
    fig.savefig(out / "F-marsc-pipeline.pdf", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c4-dir", default="")
    ap.add_argument("--only", default="", help="conditioning: redraw that figure alone (no c4 inputs needed)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.only == "conditioning":
        fig_conditioning(out); print("wrote F-marsc-conditioning.pdf"); return
    c4 = {}
    for pool in ("validation", "test", "unisum"):
        d = json.load(open(Path(a.c4_dir) / f"c4_attribution_{pool}.json"))
        c4[pool] = {"systems": d["systems"], "n_pairs": d["n_pairs_common"]}
    # Figure 1 (F-marsc-pipeline.pdf) is the drawn visual abstract and is not regenerated here
    fig_endpoints(out); fig_conditioning(out); fig_attribution(c4, out)
    print("wrote", sorted(p.name for p in out.glob("*.pdf")))


if __name__ == "__main__":
    main()
