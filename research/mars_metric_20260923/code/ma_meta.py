#!/usr/bin/env python3
"""M-A.3 -- meta-evaluation: every metric column against every human dimension, per dataset.

Statistics (PREREG.md, M-A): summary-level Kendall tau-b (within document across systems, documents with at least
three scored systems and non-constant values, averaged over documents), pooled Spearman rho, system-level Pearson
r over system means. Document-clustered percentile bootstrap, 2,000 draws, the SAME document draws for every
column so that paired differences between a MARS column and a comparator are differences of matched draws.
The per-document tau is computed once and resampled; rho and r are recomputed per draw with weighted bincounts.
Bars (i) and (ii) of M-A are applied here and written into the report; nothing else decides them.

M-G (audit response, 2026-09-28; research/mars_metric_20260923/PREREG.md, Amendment M-G). The default run above lets
every column drop its own missing pairs and its own undefined-tau documents, so two columns of one contrast can
average over different documents (review finding F01). Options added, none of which changes the default output:
  --common-cohort COLS   one population per (dataset, dimension): pairs where every listed column that exists on the
                         dataset is scored, then documents where every one of them has a defined tau
  --pairwise-common      every MARS-vs-comparator contrast recomputed on the pairs and documents both columns share
  --exclude-docs FILE    doc_ids to drop before anything else (the SummEval sources seen in MARS-C training)
  --ni-margin D          report, beside the registered "not_below" reading, whether the 95% lower bound clears -D
  Holm                   step-down adjustment of bootstrap two-sided p over the declared primary family (the BARS
                         contrasts), reported beside the registered verdicts
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

TARGETS = {"unisumeval": ["completeness", "faithfulness", "conciseness"],
           "summeval": ["relevance", "consistency", "coherence", "fluency"],
           "rose": ["acu_recall"]}
BARS = [  # (dataset, dimension, mars column, comparator columns, direction)
    ("unisumeval", "completeness", "marsR_marsc", ["rougeL", "bertscore_F"], "exceeds"),
    ("summeval", "relevance", "marsR_marsc", ["rougeL", "bertscore_F"], "exceeds"),
    ("unisumeval", "faithfulness", "marsP_factcg", ["alignscore", "factcg_whole"], "not_below"),
    ("summeval", "consistency", "marsP_factcg", ["alignscore", "factcg_whole"], "not_below"),
]
SKIP = {"pair_id", "dataset", "n_facts", "fallback", "n_refs"}
TOL = 1e-4  # within-document spread below this is "constant" (GPU batching jitter reaches 3e-5)


class Column:
    """One (metric column, human dimension) on one dataset: arrays over the pairs that carry both values."""

    def __init__(self, sub: list[dict], col: str, dim: str, doc_index: dict[str, int], sys_index: dict[str, int],
                 keep_docs: np.ndarray | None = None):
        d, s, h, m = [], [], [], []
        for p in sub:
            hv = p["human"].get(dim); mv = p["scores"].get(col)
            if hv is None or mv is None or (isinstance(mv, float) and np.isnan(mv)):
                continue
            d.append(doc_index[p["doc_id"]]); s.append(sys_index[p["system"]]); h.append(float(hv)); m.append(float(mv))
        self.doc, self.sys, self.h, self.m = (np.array(x) for x in (d, s, h, m))
        self.n = len(self.h)
        # per-document tau-b, once
        self.tau_doc = np.full(len(doc_index), np.nan)
        for di in np.unique(self.doc):
            sel = self.doc == di
            # a column constant within the document up to float noise (the no-candidate control) has no ranking;
            # ranking its 1e-9 jitter would follow the fixed system order of the datasets, a pure artefact
            if sel.sum() >= 3 and np.std(self.h[sel]) > 0 and np.std(self.m[sel]) > TOL:
                t = stats.kendalltau(self.h[sel], self.m[sel]).statistic
                self.tau_doc[di] = t
        if keep_docs is not None:
            self.tau_doc[~keep_docs] = np.nan
        self.n_sys = len(sys_index)

    def stats(self, counts: np.ndarray) -> tuple[float, float, float]:
        """counts[doc] = multiplicity in this draw (all ones for the observed statistic)."""
        w = counts[self.doc]
        tau = float(np.nanmean(np.repeat(self.tau_doc, counts.astype(int))))
        rep = np.repeat(np.arange(self.n), w.astype(int))
        if len(rep) > 2 and np.std(self.h[rep]) > 0 and np.std(self.m[rep]) > TOL:
            rho = float(np.corrcoef(stats.rankdata(self.h[rep]), stats.rankdata(self.m[rep]))[0, 1])
        else:
            rho = float("nan")
        cnt = np.bincount(self.sys, weights=w, minlength=self.n_sys)
        ok = cnt > 0
        hs = np.bincount(self.sys, weights=w * self.h, minlength=self.n_sys)[ok] / cnt[ok]
        ms = np.bincount(self.sys, weights=w * self.m, minlength=self.n_sys)[ok] / cnt[ok]
        r = float(np.corrcoef(hs, ms)[0, 1]) if len(hs) > 2 and np.std(hs) > 0 and np.std(ms) > TOL else float("nan")
        return tau, rho, r


CONTROL_SUFFIXES = ("_shufcand", "_nocand", "_shufsrc")


def scored(p: dict, col: str) -> bool:
    v = p["scores"].get(col)
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def boot_p(d: np.ndarray) -> float:
    d = d[~np.isnan(d)]
    if not len(d):
        return float("nan")
    return float(min(1.0, 2.0 * min(np.mean(d <= 0), np.mean(d >= 0))))


def holm(ps: dict[str, float]) -> dict[str, float]:
    items = sorted((v, k) for k, v in ps.items() if not np.isnan(v))
    out, run, m = {}, 0.0, len(items)
    for i, (v, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v)); out[k] = run
    return out


def common_mask(sub, cols, dim, doc_index, sys_index) -> np.ndarray:
    keep = np.ones(len(doc_index), dtype=bool)
    for c in cols:
        keep &= ~np.isnan(Column(sub, c, dim, doc_index, sys_index).tau_doc)
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--scores", nargs="+", required=True, help="scores_*.jsonl (globs)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--min-pairs", type=int, default=30)
    ap.add_argument("--common-cohort", default="", help="comma list of columns defining one shared population")
    ap.add_argument("--pairwise-common", action="store_true")
    ap.add_argument("--exclude-docs", default="", help="file with one doc_id per line, dropped before anything else")
    ap.add_argument("--ni-margin", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pairs = [json.loads(l) for l in open(a.pairs, encoding="utf-8")]
    for p in pairs:
        p["scores"] = {k: float(v) for k, v in (p.get("machine") or {}).items() if v is not None}
    by_id = {p["pair_id"]: p for p in pairs}
    for pat in a.scores:
        for f in sorted(glob.glob(pat)):
            for line in open(f, encoding="utf-8"):
                r = json.loads(line); p = by_id.get(r["pair_id"])
                if p is None:
                    continue
                for k, v in r.items():
                    if k not in SKIP and v is not None:
                        p["scores"][k] = float(v)
    excluded = set()
    if a.exclude_docs:
        excluded = {x.strip() for x in open(a.exclude_docs, encoding="utf-8") if x.strip()}
        pairs = [p for p in pairs if p["doc_id"] not in excluded]
    cols = sorted({k for p in pairs for k in p["scores"]})
    common = [c for c in a.common_cohort.split(",") if c]
    rng = np.random.default_rng(a.seed)
    rep: dict = {"columns": cols, "n_boot": a.n_boot, "datasets": {}, "bars": [],
                 "mode": {"common_cohort": common, "pairwise_common": a.pairwise_common,
                          "excluded_docs": sorted(excluded), "ni_margin": a.ni_margin}}
    for ds, dims in TARGETS.items():
        sub_all = [p for p in pairs if p["dataset"] == ds]
        if not sub_all:
            continue
        sub = sub_all
        docs = sorted({p["doc_id"] for p in sub}); doc_index = {d: i for i, d in enumerate(docs)}
        systems = sorted({p["system"] for p in sub}); sys_index = {s: i for i, s in enumerate(systems)}
        draws = [np.bincount(rng.integers(0, len(docs), size=len(docs)), minlength=len(docs)) for _ in range(a.n_boot)]
        ones = np.ones(len(docs))
        rep["datasets"][ds] = {"n_pairs": len(sub), "n_docs": len(docs), "n_systems": len(systems), "dims": {}, "contrasts": {}}
        for dim in dims:
            entry, boot = {}, {}
            keep = None
            if common:
                cc = [c for c in common if any(scored(p, c) for p in sub_all)]
                sub = [p for p in sub_all if p["human"].get(dim) is not None and all(scored(p, c) for c in cc)]
                keep = common_mask(sub, cc, dim, doc_index, sys_index)
                rep["datasets"][ds].setdefault("common_cohort", {})[dim] = {
                    "columns": cc, "n_pairs": len(sub), "n_docs_all_tau": int(keep.sum())}
            for col in cols:
                c = Column(sub, col, dim, doc_index, sys_index, keep)
                if c.n < a.min_pairs:
                    continue
                obs = c.stats(ones)
                arr = np.array([c.stats(w) for w in draws])  # (n_boot, 3)
                boot[col] = arr
                entry[col] = {"n_pairs": int(c.n), "docs_with_tau": int(np.sum(~np.isnan(c.tau_doc)))}
                for j, name in enumerate(("summary_tau", "pooled_rho", "system_r")):
                    entry[col][name] = {"observed": obs[j], "ci95": [float(np.nanpercentile(arr[:, j], 2.5)), float(np.nanpercentile(arr[:, j], 97.5))]}
            rep["datasets"][ds]["dims"][dim] = entry
            con = {}
            for mc in [c for c in entry if c.startswith("mars")]:
                for oc in [c for c in entry if c != mc]:
                    d = boot[mc] - boot[oc]
                    con[f"{mc}-{oc}"] = {name: {"observed": entry[mc][name]["observed"] - entry[oc][name]["observed"],
                                                "ci95": [float(np.nanpercentile(d[:, j], 2.5)), float(np.nanpercentile(d[:, j], 97.5))]}
                                         for j, name in enumerate(("summary_tau", "pooled_rho", "system_r"))}
            rep["datasets"][ds]["contrasts"][dim] = con
            if a.pairwise_common:
                pw = {}
                main_cols = [c for c in entry if not c.endswith(CONTROL_SUFFIXES) and c != "n_units"]
                for mc in [c for c in main_cols if c.startswith("mars")]:
                    for oc in [c for c in main_cols if c != mc]:
                        sp = [p for p in sub if scored(p, mc) and scored(p, oc)]
                        km = common_mask(sp, [mc, oc], dim, doc_index, sys_index)
                        A = Column(sp, mc, dim, doc_index, sys_index, km); B = Column(sp, oc, dim, doc_index, sys_index, km)
                        if A.n < a.min_pairs:
                            continue
                        oa, ob = A.stats(ones)[0], B.stats(ones)[0]
                        d = np.array([A.stats(w)[0] - B.stats(w)[0] for w in draws])
                        pw[f"{mc}-{oc}"] = {"n_pairs": int(A.n), "n_docs": int(km.sum()), "mars_tau": oa, "other_tau": ob,
                                            "summary_tau": {"observed": oa - ob,
                                                            "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
                                                            "p_boot_two_sided": boot_p(d)}}
                rep["datasets"][ds].setdefault("contrasts_pairwise_common", {})[dim] = pw
            for k, v in con.items():
                mc, oc = k.split("-", 1)
                if mc in boot and oc in boot:
                    v["summary_tau"]["p_boot_two_sided"] = boot_p(boot[mc][:, 0] - boot[oc][:, 0])
            print(f"[ma-meta] {ds}/{dim}: " + ", ".join(f"{c} {v['summary_tau']['observed']:+.3f}" for c, v in sorted(entry.items(), key=lambda kv: -np.nan_to_num(kv[1]['summary_tau']['observed'], nan=-9))[:10]), flush=True)
    for ds, dim, mc, comps, direction in BARS:
        con = rep["datasets"].get(ds, {}).get("contrasts", {}).get(dim, {})
        res = {"dataset": ds, "dimension": dim, "mars": mc, "direction": direction, "statistic": "summary_tau", "contrasts": {}, "pass": None}
        verdicts = []
        for c in comps:
            k = f"{mc}-{c}"
            if k not in con:
                res["contrasts"][c] = "missing"; continue
            o, ci = con[k]["summary_tau"]["observed"], con[k]["summary_tau"]["ci95"]
            res["contrasts"][c] = {"observed": o, "ci95": ci, "p_boot_two_sided": con[k]["summary_tau"].get("p_boot_two_sided")}
            if a.ni_margin is not None and direction == "not_below":
                res["contrasts"][c]["clears_ni_margin"] = bool(ci[0] > -a.ni_margin)
            pwc = rep["datasets"].get(ds, {}).get("contrasts_pairwise_common", {}).get(dim, {}).get(k)
            if pwc:
                res["contrasts"][c]["pairwise_common"] = pwc
            verdicts.append((ci[0] > 0) if direction == "exceeds" else (ci[1] >= 0 or o >= 0))
        res["pass"] = (all(verdicts) if len(verdicts) == len(comps) else None)
        rep["bars"].append(res)
        print(f"[ma-meta] bar {ds}/{dim} {mc} {direction}: " + json.dumps(res["contrasts"]) + f" -> {res['pass']}", flush=True)
    fam = {f"{b['dataset']}/{b['dimension']}/{b['mars']}-{c}": v["p_boot_two_sided"]
           for b in rep["bars"] for c, v in b["contrasts"].items() if isinstance(v, dict) and v.get("p_boot_two_sided") is not None}
    rep["holm_primary_family"] = {"family": sorted(fam), "p_raw": fam, "p_holm": holm(fam)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"[ma-meta] written {a.out}", flush=True)


if __name__ == "__main__":
    main()
