#!/usr/bin/env python3
"""A5 -- measure the seedrank x div1 incompatibility from the actual score files (PREREG Wave A / A5, item 4c).

`mc_diversity.py` implements "at most one emitted unit per source sentence" by pushing every non-leading kept unit
of a sentence by a FIXED -5.0, documented as "below every sentence leader, above units the policy removed at -10".
That constant is calibrated for a PROBABILITY-scale score in [0, 1], where -5.0 is below every undemoted unit by
construction. The `seedrank` aggregation instead scores a unit by the negated mean rank across seeds, so its scores
run to -(max units per pair). A -5.0 push then moves a unit five RANK SLOTS, not below the sentence leaders, and the
diversity rule silently degenerates.

This job measures that directly, on the SAME pool and the SAME emission step, for four arms:
a seedrank arm and its probability-scale seedavg counterpart, twice (gemma+ent humanfact, distilled xenc).
Reported per arm: score range before and after div1, how many units the div1 pass actually moved, how many DEMOTED
units are still inside the top-10 afterwards, the top-10 overlap between the pre- and post-div orders, and -- the
quantity the rule exists to control -- the number of DISTINCT SOURCE SENTENCES among the top-10.

Under a working rule the top-10 holds ~10 distinct sentences and a demoted unit essentially never survives in it.
No bar: this is a diagnostic, and whichever way it falls is the number reported.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402


def sentence_index(source: str, start: int) -> int:
    """Byte-identical to mc_diversity.sentence_index, so the audit measures the rule that actually ran."""
    for i, (a, b) in enumerate(common.sentence_spans(source)):
        if a <= start < b:
            return i
    return -1


def topk(v: np.ndarray, starts: list[int], k: int) -> list[int]:
    """The evaluator's own ordering: descending score, NaN last, ties broken by source position."""
    return sorted(range(len(v)), key=lambda j: (-v[j] if np.isfinite(v[j]) else 1.0, starts[j]))[:k]


def audit(label: str, units_path: Path, pre: Path, post: Path, k: int) -> dict:
    sent_of: dict[str, list[int]] = {}; start_of: dict[str, list[int]] = {}
    for r in common.load_jsonl(units_path):
        n = len(r["source"])
        st = [int(u.get("start", n // 2)) for u in r["units"]]
        start_of[r["pair_id"]] = st
        sent_of[r["pair_id"]] = [sentence_index(r["source"], s) for s in st]
    name_pre, sc_pre = common.read_scores(pre)
    name_post, sc_post = common.read_scores(post)
    pids = sorted(p for p in sc_pre if p in sc_post and p in sent_of
                  and len(sc_pre[p]) == len(sc_post[p]) == len(sent_of[p]))

    allv_pre = np.concatenate([sc_pre[p] for p in pids]) if pids else np.array([])
    allv_post = np.concatenate([sc_post[p] for p in pids]) if pids else np.array([])
    fin_pre = allv_pre[np.isfinite(allv_pre)]; fin_post = allv_post[np.isfinite(allv_post)]

    n_moved = 0; n_units = 0; deltas: dict[str, int] = {}
    moved_in_topk = 0; moved_total_in_pre_topk = 0
    ov = []; dist_pre = []; dist_post = []; n_max_units = 0
    for p in pids:
        a_, b_ = sc_pre[p], sc_post[p]; st = start_of[p]; si = sent_of[p]
        n_units += len(a_); n_max_units = max(n_max_units, len(a_))
        d = np.where(np.isfinite(a_) & np.isfinite(b_), b_ - a_, 0.0)
        mv = np.abs(d) > 1e-9
        n_moved += int(mv.sum())
        for x in np.unique(np.round(d[mv], 6)):
            deltas[f"{float(x):+.6g}"] = deltas.get(f"{float(x):+.6g}", 0) + int(np.sum(np.round(d[mv], 6) == x))
        tpre, tpost = topk(a_, st, k), topk(b_, st, k)
        ov.append(len(set(tpre) & set(tpost)) / max(1, len(tpre)))
        dist_pre.append(len({si[j] for j in tpre}))
        dist_post.append(len({si[j] for j in tpost}))
        moved_in_topk += sum(1 for j in tpost if mv[j])
        moved_total_in_pre_topk += sum(1 for j in tpre if mv[j])

    return {
        "label": label, "system_pre": name_pre, "system_post": name_post,
        "pre_file": str(pre), "post_file": str(post), "units_file": str(units_path),
        "n_pairs": len(pids), "n_units": n_units, "max_units_per_pair": n_max_units,
        "score_range_pre": [float(fin_pre.min()), float(fin_pre.max())] if fin_pre.size else None,
        "score_range_post": [float(fin_post.min()), float(fin_post.max())] if fin_post.size else None,
        "range_changed_by_div1": (None if not (fin_pre.size and fin_post.size)
                                  else bool(abs(float(fin_pre.min()) - float(fin_post.min())) > 1e-9
                                            or abs(float(fin_pre.max()) - float(fin_post.max())) > 1e-9)),
        "units_moved_by_div1": n_moved, "units_moved_share": n_moved / max(1, n_units),
        "observed_deltas": deltas,
        f"demoted_units_still_in_top{k}": moved_in_topk,
        f"demoted_units_per_pair_in_top{k}": moved_in_topk / max(1, len(pids)),
        f"pre_div_top{k}_units_that_get_demoted": moved_total_in_pre_topk,
        f"top{k}_overlap_pre_vs_post_mean": float(np.mean(ov)) if ov else None,
        f"distinct_sentences_in_top{k}_pre_mean": float(np.mean(dist_pre)) if dist_pre else None,
        f"distinct_sentences_in_top{k}_post_mean": float(np.mean(dist_post)) if dist_post else None,
        "k": k,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="LABEL=UNITS_FILE:PRE_SCORES:POST_DIV_SCORES (repeatable)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = []
    for spec in a.arm:
        label, rest = spec.split("=", 1)
        units_path, pre, post = rest.split(":")
        for f in (units_path, pre, post):
            assert Path(f).exists(), f"arm {label!r}: missing {f}"
        arms.append(audit(label, Path(units_path), Path(pre), Path(post), a.k))
        r = arms[-1]
        print(f"[a5] {label:34s} pairs={r['n_pairs']} maxunits={r['max_units_per_pair']} "
              f"range {r['score_range_pre']} -> {r['score_range_post']} changed={r['range_changed_by_div1']}", flush=True)
        print(f"[a5] {label:34s} moved={r['units_moved_by_div1']} ({r['units_moved_share']:.4f}) "
              f"demoted-still-in-top{a.k}={r[f'demoted_units_still_in_top{a.k}']} "
              f"({r[f'demoted_units_per_pair_in_top{a.k}']:.3f}/pair) "
              f"top{a.k}-overlap={r[f'top{a.k}_overlap_pre_vs_post_mean']:.4f} "
              f"distinct-sentences {r[f'distinct_sentences_in_top{a.k}_pre_mean']:.3f} -> "
              f"{r[f'distinct_sentences_in_top{a.k}_post_mean']:.3f}", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "a5_seedrank_audit.json").write_text(json.dumps(
        {"registered": "research/marsc_strengthen_20260918/PREREG.md Wave A / A5 (provenance item 4c)",
         "diversity_push": -5.0, "arms": arms}, indent=1))
    print(f"[a5] written {out / 'a5_seedrank_audit.json'}", flush=True)


if __name__ == "__main__":
    main()
