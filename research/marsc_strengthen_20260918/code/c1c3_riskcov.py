#!/usr/bin/env python3
"""MARS-C strengthening C1 (H-SEL) + C3 (H-COMP) -- risk-coverage curves and complementarity.

Registered in research/marsc_strengthen_20260918/PREREG.md, Wave C / C1 and C3. Every bar below is
declared in this file, in code, and is written into the output JSON BEFORE any number it is applied to.

INPUT is the frozen emitted dump of `mc_e2_eval.py --k 10 --dump-emitted` (one row per system x pair
carrying the k=10 emitted units with their score and their evaluator-assigned status hit /
false_alert / unknown). Nothing is re-matched here: the ACU alignment, the emission rule and the
top-k budget are exactly the audited evaluator's, so a C1 curve cannot drift from the recall@10 table
it sits beside.

C1 -- H-SEL, the risk-coverage curve
  A deployed omission finder may abstain; the paper's ten-slot budget forces ten alerts. Sweep a
  single global score threshold per verifier family. At threshold tau the family fills only the slots
  it scores >= tau, so

      coverage(tau)  = (# emitted slots with score >= tau) / (# emitted slots available)
      precision(tau) = hits / (hits + false_alerts) among the FILLED slots

  which is the evaluator's `omission_precision` restricted to the filled slots and equals it exactly
  at coverage 1. Because a family's curve is produced by walking its own score order, the curve is
  invariant to any strictly monotone recalibration of that family -- the A1 calibration exposure
  cannot touch it. Slots the evaluator marked `unknown` (the emitted unit aligns to no reference ACU)
  occupy a slot and so enter the coverage denominator, but enter neither side of the precision ratio;
  the strict variant that charges every unknown as a false alert is reported beside it as
  `precision_strict`. `hit_yield_per_pair` is the un-normalised numerator of recall (it double-counts
  two emitted units that hit the same ACU, so it is NOT recall and is never called that).

  BAR (declared before execution): H-SEL passes for a primary arm against a comparator if the
  document-clustered 95% interval of the paired precision difference is strictly above zero at EVERY
  coverage grid point in [0.20, 1.00]. The bar is a conjunction over grid points, so no multiplicity
  correction is applied or needed. Partial dominance is reported as the sub-interval over which the
  interval excludes zero, and a crossing is reported as a crossing.

C3 -- H-COMP, complementarity
  BAR (declared before execution): H-COMP passes for a fusion if, at the DECLARED PRIMARY COVERAGE
  (1.00 -- the paper's own ten-slot operating point), the document-clustered 95% interval of the
  paired omission-precision difference against EACH of its two constituents is strictly above zero.
  The whole [0.20, 1.00] grid is reported beside it; passing at some coverages and not at 1.00 is
  reported as exactly that and does not pass the bar.

Bootstrap: documents (common.doc_key(source)) are resampled with replacement, 2000 replicates,
rng seed 20260918. ONE document resample per replicate is shared by every system, so every reported
difference is paired. Within a replicate a family's coverage is recomputed inside the resample, so
"at equal coverage" means equal coverage in the same resampled population.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402

BOOT_SEED = 20260918
BAR_C1 = ("H-SEL passes against a comparator iff the document-clustered 95% interval of the paired "
          "precision difference excludes zero from above at EVERY coverage grid point in [0.20, 1.00].")
BAR_C3 = ("H-COMP passes iff, at coverage 1.00, the document-clustered 95% interval of the paired "
          "omission-precision difference against EACH constituent excludes zero from above.")
C3_PRIMARY_COVERAGE = 1.00
P_STAR = [0.70, 0.72, 0.75, 0.78]   # declared; brackets the frozen r06b_fast TEST omission precisions


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dumps(files: list[Path], keep_k: str) -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if str(r.get("k")) != keep_k or r.get("emitted") is None:
                    continue
                out[r["system"]][r["pair_id"]] = r["emitted"]
    return out


class Curve:
    """Per-system slot arrays, pre-sorted by descending score (the family's own deployment order)."""

    def __init__(self, name: str, per_pair: dict[str, list[dict]], pid_index: dict[str, int]):
        pair_ix, score, hit, fa, unk = [], [], [], [], []
        for pid, em in per_pair.items():
            pi = pid_index[pid]
            for u in em:
                s = u.get("score")
                pair_ix.append(pi)
                score.append(-np.inf if s is None else float(s))
                hit.append(1.0 if u["status"] == "hit" else 0.0)
                fa.append(1.0 if u["status"] == "false_alert" else 0.0)
                unk.append(1.0 if u["status"] == "unknown" else 0.0)
        s = np.asarray(score, dtype=float)
        order = np.argsort(-s, kind="mergesort")
        self.name = name
        self.pair_ix = np.asarray(pair_ix, dtype=np.int64)[order]
        self.score = s[order]
        self.hit = np.asarray(hit, dtype=float)[order]
        self.fa = np.asarray(fa, dtype=float)[order]
        self.unk = np.asarray(unk, dtype=float)[order]
        self.n_slots = int(s.size)

    def at(self, w: np.ndarray, grid: np.ndarray, n_pairs_w: float) -> dict[str, np.ndarray]:
        """Weighted (bootstrap) curve. w = per-slot multiplicity from the document resample."""
        cw = np.cumsum(w)
        tot = cw[-1]
        if tot <= 0:
            nan = np.full(grid.size, np.nan)
            return {"precision": nan, "precision_strict": nan, "hit_yield_per_pair": nan, "unknown_share": nan}
        ch = np.cumsum(w * self.hit)
        cf = np.cumsum(w * self.fa)
        cu = np.cumsum(w * self.unk)
        # the last slot whose cumulative weight reaches c * tot (searchsorted on a non-decreasing cumsum)
        idx = np.searchsorted(cw, grid * tot, side="left")
        idx = np.clip(idx, 0, cw.size - 1)
        h, f, u, n = ch[idx], cf[idx], cu[idx], cw[idx]
        den = h + f
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.where(den > 0, h / np.maximum(den, 1e-12), np.nan)
            strict = np.where(n > 0, h / np.maximum(n, 1e-12), np.nan)
            unkn = np.where(n > 0, u / np.maximum(n, 1e-12), np.nan)
        return {"precision": prec, "precision_strict": strict,
                "hit_yield_per_pair": h / max(n_pairs_w, 1e-12), "unknown_share": unkn}


def ci(a: np.ndarray) -> list[float]:
    ok = a[np.isfinite(a)]
    if ok.size < 20:
        return [float("nan"), float("nan")]
    return [float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))]


def interval_of_dominance(grid: np.ndarray, lo: np.ndarray, floor: float) -> dict:
    """Longest contiguous run of grid points >= floor whose paired-difference lower bound is > 0."""
    mask = np.isfinite(lo) & (lo > 0.0) & (grid >= floor - 1e-9)
    best, cur = (None, None), None
    for i, m in enumerate(mask):
        if m and cur is None:
            cur = i
        if (not m or i == mask.size - 1) and cur is not None:
            end = i if m else i - 1
            if best[0] is None or (end - cur) > (best[1] - best[0]):
                best = (cur, end)
            cur = None
    in_range = grid >= floor - 1e-9
    return {"all_points_pass": bool(np.all(mask[in_range])),
            "n_points_pass": int(np.sum(mask[in_range])), "n_points": int(np.sum(in_range)),
            "longest_dominant_interval": (None if best[0] is None else [float(grid[best[0]]), float(grid[best[1]])]),
            "coverages_not_dominated": [float(grid[i]) for i in range(grid.size) if in_range[i] and not mask[i]][:40]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emitted", nargs="+", required=True, help="emitted dump(s) from mc_e2_eval --dump-emitted")
    ap.add_argument("--acu", required=True, help="ACU truth file of the same split (for document clusters)")
    ap.add_argument("--k-label", default="10")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--grid-step", type=float, default=0.05)
    ap.add_argument("--bar-floor", type=float, default=0.20, help="lower end of the declared dominance interval")
    ap.add_argument("--primary", action="append", default=[], help="MARS-C arm(s) the C1 bar is declared for (repeat)")
    ap.add_argument("--comparator", action="append", default=[], help="system(s) the C1 bar is declared against (repeat)")
    ap.add_argument("--control", action="append", default=[],
                    help="system(s) compared and reported but EXCLUDED from the C1 bar conjunction: the B0 "
                         "summary-blind controls (which the bar is not about) and sibling MARS-C arms (which "
                         "are not off-the-shelf comparators)")
    ap.add_argument("--fusion", action="append", default=[],
                    help="FUSED=CONST_A+CONST_B; the C3 bar is declared against both constituents (repeat)")
    ap.add_argument("--eval-json", nargs="*", default=[], help="e2_emitted.json file(s) to copy the recall@10 table from")
    ap.add_argument("--label", default="", help="free-text label for this run (split, arm, dry-run status)")
    ap.add_argument("--dry-run-note", default="", help="recorded verbatim when this run is not a paper number")
    a = ap.parse_args()

    grid = np.round(np.arange(a.grid_step, 1.0 + 1e-9, a.grid_step), 6)
    dumps = load_dumps([Path(f) for f in a.emitted], a.k_label)
    assert dumps, f"no rows with k={a.k_label!r} in {a.emitted}"
    acu = {r["pair_id"]: r for r in common.load_jsonl(Path(a.acu))}

    pid_sets = {n: set(d) for n, d in dumps.items()}
    common_pids = sorted(set.intersection(*pid_sets.values()) & set(acu))
    assert common_pids, "no pair is present in every system and in the ACU file"
    coverage_note = {n: {"n_pairs": len(s), "n_pairs_dropped_to_common": len(s) - len(set(s) & set(common_pids))}
                     for n, s in pid_sets.items()}
    pid_index = {p: i for i, p in enumerate(common_pids)}
    doc_of = np.array([common.doc_key(acu[p]["source"]) for p in common_pids])
    docs = sorted(set(doc_of.tolist()))
    doc_ix = {d: i for i, d in enumerate(docs)}
    pairs_of_doc: list[list[int]] = [[] for _ in docs]
    for i, d in enumerate(doc_of):
        pairs_of_doc[doc_ix[d]].append(i)

    curves = {n: Curve(n, {p: d[p] for p in common_pids}, pid_index) for n, d in dumps.items()}
    names = sorted(curves)
    missing = [x for x in a.primary + a.comparator + a.control if x not in curves]
    fus = []
    for spec in a.fusion:
        f, rest = spec.split("=", 1)
        cs = [x for x in rest.split("+") if x]
        assert len(cs) == 2, f"--fusion {spec!r}: expected FUSED=A+B"
        fus.append((f, cs))
        missing += [x for x in [f] + cs if x not in curves]
    assert not missing, f"named systems absent from the dump: {sorted(set(missing))}; present: {names}"

    rep = {
        "registered": "research/marsc_strengthen_20260918/PREREG.md Wave C / C1 (H-SEL) and C3 (H-COMP)",
        "declared_before_numbers": {
            "bar_c1_H_SEL": BAR_C1,
            "bar_c3_H_COMP": BAR_C3,
            "c3_primary_coverage": C3_PRIMARY_COVERAGE,
            "dominance_interval": [a.bar_floor, 1.0],
            "coverage_definition": "filled slots / available emitted slots at k=10, a single global score threshold per family",
            "precision_definition": "hits / (hits + false_alerts) among filled slots; equals the evaluator's omission_precision at coverage 1",
            "unknown_slots": "counted in the coverage denominator, excluded from the precision ratio; precision_strict charges them as false alerts",
            "bootstrap": f"document-clustered (common.doc_key), {a.n_boot} replicates, rng seed {BOOT_SEED}, one shared resample per replicate so every difference is paired",
            "multiplicity": "the C1 bar is a conjunction over grid points and the C3 bar a conjunction over two constituents; conjunctions need no multiplicity correction",
            "who_enters_the_c1_bar": "only systems passed as --comparator (the off-the-shelf verifier families). Systems passed as --control -- the B0 shuffled/no-summary arms and the sibling MARS-C judge-label arms -- are computed and reported on the same axes but are excluded from the bar conjunction, because the registered bar is about off-the-shelf comparators.",
            "hit_yield_is_not_recall": "hit_yield_per_pair double-counts two emitted units hitting one ACU; it is never reported as recall",
            "ties": "slots tied on score are ordered by the evaluator's emission order; a real threshold cannot split a tie, so a grid point falling inside a tie run is an interpolation of that run",
            "equal_precision_addendum_carries_NO_bar": (
                "Beside the equal-coverage bars above, the same curves are read at EQUAL PRECISION: for each "
                "target p* the largest grid coverage a family can fill while holding precision >= p*, and the "
                "hit yield it reaches there. This is a reported quantity, not a bar, and no claim in this "
                "campaign turns on it. The p* grid is fixed at (0.70, 0.72, 0.75, 0.78) because those values "
                "bracket the omission precisions already frozen and published in the r06b_fast TEST table "
                "(SummaC-ZS distilled 0.7074, MARS-C gemma+ent 0.7202, MARS-C distilled 0.7517); it is not "
                "chosen after reading anything this job computes."),
        },
        "label": a.label, "dry_run_note": a.dry_run_note,
        "k_label": a.k_label, "n_boot": int(a.n_boot), "grid": grid.tolist(),
        "n_pairs": len(common_pids), "n_docs": len(docs),
        "acu": {"path": str(a.acu), "sha256": _sha256(Path(a.acu))},
        "emitted_dumps": [{"path": str(f), "sha256": _sha256(Path(f))} for f in a.emitted],
        "systems": names, "system_pair_coverage": coverage_note,
        "roles": {"primary": a.primary, "comparator": a.comparator, "control": a.control,
                  "fusion": {f: c for f, c in fus}},
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "c1c3_declaration.json").write_text(json.dumps(rep, indent=1))
    print(f"[c1c3] declaration written before any number -> {out / 'c1c3_declaration.json'}", flush=True)

    ones = np.ones(len(common_pids), dtype=float)
    obs = {n: curves[n].at(ones[curves[n].pair_ix], grid, float(len(common_pids))) for n in names}

    rng = np.random.default_rng(BOOT_SEED)
    boot = {n: {m: np.empty((a.n_boot, grid.size), dtype=float) for m in
                ("precision", "precision_strict", "hit_yield_per_pair", "unknown_share")} for n in names}
    n_docs = len(docs)
    for b in range(a.n_boot):
        pick = rng.integers(0, n_docs, size=n_docs)
        mult = np.zeros(len(common_pids), dtype=float)
        cnt = np.bincount(pick, minlength=n_docs)
        for d, c in enumerate(cnt):
            if c:
                mult[pairs_of_doc[d]] += c
        n_pairs_w = float(mult.sum())
        for n in names:
            r = curves[n].at(mult[curves[n].pair_ix], grid, n_pairs_w)
            for m, v in r.items():
                boot[n][m][b] = v
        if (b + 1) % 250 == 0:
            print(f"[c1c3] bootstrap {b + 1}/{a.n_boot}", flush=True)

    c1 = {"curves": {}, "dominance": {}}
    for n in names:
        c1["curves"][n] = {
            "n_slots": curves[n].n_slots,
            "slots_per_pair": round(curves[n].n_slots / max(1, len(common_pids)), 4),
            **{m: {"observed": [None if not np.isfinite(x) else float(x) for x in obs[n][m]],
                   "ci": [ci(boot[n][m][:, j]) for j in range(grid.size)]}
               for m in ("precision", "precision_strict", "hit_yield_per_pair", "unknown_share")},
        }
    for prim in a.primary:
        block = {}
        for comp in [x for x in a.comparator + a.control if x != prim]:
            d_obs = obs[prim]["precision"] - obs[comp]["precision"]
            d_boot = boot[prim]["precision"] - boot[comp]["precision"]
            lo = np.array([ci(d_boot[:, j])[0] for j in range(grid.size)])
            hi = np.array([ci(d_boot[:, j])[1] for j in range(grid.size)])
            block[comp] = {
                "observed_diff": [None if not np.isfinite(x) else float(x) for x in d_obs],
                "ci_low": [None if not np.isfinite(x) else float(x) for x in lo],
                "ci_high": [None if not np.isfinite(x) else float(x) for x in hi],
                "is_control": comp in a.control, "enters_bar": comp not in a.control,
                **interval_of_dominance(grid, lo, a.bar_floor),
            }
        barred = [c for c in block if not block[c]["is_control"]]
        c1["dominance"][prim] = {
            "against": block,
            "bar": BAR_C1,
            "bar_verdict": ("PASS" if barred and all(block[c]["all_points_pass"] for c in barred)
                            else "FAIL" if barred else "NOT-EVALUATED-no-comparator"),
            "comparators_dominated": sorted(c for c in barred if block[c]["all_points_pass"]),
            "comparators_not_dominated": sorted(c for c in barred if not block[c]["all_points_pass"]),
        }
        print(f"[c1] primary {prim}: verdict {c1['dominance'][prim]['bar_verdict']} "
              f"({len(c1['dominance'][prim]['comparators_dominated'])}/{len(barred)} comparators dominated)", flush=True)

    jc = int(np.argmin(np.abs(grid - C3_PRIMARY_COVERAGE)))
    c3 = {"primary_coverage": float(grid[jc]), "bar": BAR_C3, "fusions": {}}
    for f, cs in fus:
        vs = {}
        for c in cs:
            d_obs = obs[f]["precision"] - obs[c]["precision"]
            d_boot = boot[f]["precision"] - boot[c]["precision"]
            lo = np.array([ci(d_boot[:, j])[0] for j in range(grid.size)])
            hi = np.array([ci(d_boot[:, j])[1] for j in range(grid.size)])
            vs[c] = {
                "observed_diff_at_primary_coverage": None if not np.isfinite(d_obs[jc]) else float(d_obs[jc]),
                "ci_at_primary_coverage": [None if not np.isfinite(lo[jc]) else float(lo[jc]),
                                           None if not np.isfinite(hi[jc]) else float(hi[jc])],
                "passes_at_primary_coverage": bool(np.isfinite(lo[jc]) and lo[jc] > 0.0),
                "observed_diff": [None if not np.isfinite(x) else float(x) for x in d_obs],
                "ci_low": [None if not np.isfinite(x) else float(x) for x in lo],
                "ci_high": [None if not np.isfinite(x) else float(x) for x in hi],
                **interval_of_dominance(grid, lo, a.bar_floor),
            }
        c3["fusions"][f] = {
            "constituents": cs, "vs": vs,
            "bar_verdict": "PASS" if all(vs[c]["passes_at_primary_coverage"] for c in cs) else "FAIL",
            "observed_precision_at_primary_coverage": {
                x: (None if not np.isfinite(obs[x]["precision"][jc]) else float(obs[x]["precision"][jc]))
                for x in [f] + cs},
        }
        print(f"[c3] fusion {f} vs {cs}: verdict {c3['fusions'][f]['bar_verdict']} " +
              " ".join(f"| {c} {vs[c]['observed_diff_at_primary_coverage']:+.4f} "
                       f"[{vs[c]['ci_at_primary_coverage'][0]:+.4f},{vs[c]['ci_at_primary_coverage'][1]:+.4f}]"
                       for c in cs), flush=True)

    headline = {}
    for p in a.eval_json:
        try:
            e = json.load(open(p))
        except Exception as ex:  # noqa: BLE001
            headline[p] = {"error": str(ex)}
            continue
        kk = e.get("k", {}).get(a.k_label) or {}
        sysd = kk.get("systems") or {}
        prim = kk.get("primary")
        pr = sysd.get(prim) or {}
        obs_contrast = {
            n: {"d_recall_observed": (None if (s.get("recall") is None or pr.get("recall") is None)
                                      else float(pr["recall"] - s["recall"])),
                "d_omission_precision_observed": (None if (s.get("omission_precision") is None or
                                                           pr.get("omission_precision") is None)
                                                  else float(pr["omission_precision"] - s["omission_precision"]))}
            for n, s in sysd.items() if n != prim}
        headline[p] = {"n_pairs": e.get("n_pairs"), "n_docs": e.get("n_docs"), "primary": prim,
                       "systems": {n: {"recall": s.get("recall"), "omission_precision": s.get("omission_precision"),
                                       "unknown_rate": s.get("unknown_rate")} for n, s in sysd.items()},
                       "observed_contrast_vs_primary": obs_contrast,
                       "WARNING_paired_blocks_are_bootstrap_means": (
                           "the evaluator's paired_recall.mean / paired_omission_precision.mean are BOOTSTRAP means, "
                           "not observed contrasts; use observed_contrast_vs_primary for the observed difference and "
                           "the paired block only for its interval"),
                       "paired_recall_bootstrap": kk.get("paired_recall"),
                       "paired_omission_precision_bootstrap": kk.get("paired_omission_precision")}

    rep["c1_risk_coverage"] = c1
    rep["c3_complementarity"] = c3
    rep["recall_at_k_table"] = headline
    (out / "c1c3_result.json").write_text(json.dumps(rep, indent=1))
    print(f"[c1c3] written {out / 'c1c3_result.json'} (primary result complete)", flush=True)

    # ----- reported addendum, NO bar: the same curves read at equal PRECISION instead of equal coverage.
    # Written second and guarded, so a defect here can never cost the primary result above.
    try:
        def op_point(prec: np.ndarray, yld: np.ndarray, p_star: float) -> tuple[float, float]:
            ok = np.isfinite(prec) & (prec >= p_star)
            if not np.any(ok):
                return 0.0, 0.0
            j = int(np.max(np.nonzero(ok)[0]))
            return float(grid[j]), float(yld[j] if np.isfinite(yld[j]) else np.nan)

        eq = {"targets": P_STAR, "bar": "none -- reported quantity", "by_target": {}}
        for ps in P_STAR:
            cov_obs, yld_obs = {}, {}
            cov_boot = {}
            for n in names:
                c_, y_ = op_point(obs[n]["precision"], obs[n]["hit_yield_per_pair"], ps)
                cov_obs[n], yld_obs[n] = c_, y_
                cb = np.empty(a.n_boot, dtype=float)
                for b in range(a.n_boot):
                    cb[b] = op_point(boot[n]["precision"][b], boot[n]["hit_yield_per_pair"][b], ps)[0]
                cov_boot[n] = cb
            blk = {n: {"max_coverage_holding_precision": cov_obs[n], "ci": ci(cov_boot[n]),
                       "hit_yield_per_pair_there": (None if not np.isfinite(yld_obs[n]) else yld_obs[n])}
                   for n in names}
            for prim in a.primary:
                for comp in names:
                    if comp == prim:
                        continue
                    d = cov_boot[prim] - cov_boot[comp]
                    blk[comp][f"d_coverage_vs_{prim}"] = {
                        "observed": cov_obs[prim] - cov_obs[comp], "ci": ci(d)}
            eq["by_target"][f"{ps:.2f}"] = blk
            best = sorted(names, key=lambda n: -cov_obs[n])[:4]
            print("[c1 equal-precision] p*>=%.2f  " % ps +
                  "  ".join(f"{n} cov {cov_obs[n]:.2f}" for n in best), flush=True)
        rep["c1_equal_precision_operating_point"] = eq
    except Exception as ex:  # noqa: BLE001
        rep["c1_equal_precision_operating_point"] = {"error": repr(ex)}
        print(f"[c1c3] WARNING equal-precision addendum failed: {ex!r} "
              f"(the primary result above is unaffected)", flush=True)
    (out / "c1c3_result.json").write_text(json.dumps(rep, indent=1))
    print(f"[c1c3] written {out / 'c1c3_result.json'}", flush=True)

    for n in sorted(names, key=lambda x: -(obs[x]["precision"][-1] if np.isfinite(obs[x]["precision"][-1]) else -9)):
        row = " ".join(f"{c:.2f}:{obs[n]['precision'][j]:.3f}" if np.isfinite(obs[n]["precision"][j]) else f"{c:.2f}:--"
                       for j, c in enumerate(grid) if abs(c * 100 % 20) < 1e-6)
        print(f"[c1 curve] {n:36s} {row}", flush=True)


if __name__ == "__main__":
    main()
