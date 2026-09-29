#!/usr/bin/env python3
"""MARS-3 Stage 0 -- diagnostics on existing artifacts (PLAN.md M30). No model is run.

D1 inventory ceiling : do human-omitted ACUs have an aligned extracted unit? (validation + test, per resource)
D2 loss anatomy      : MARS-2 vs cross-encoder AUROC by lexical-recall stratum, unit kind, unit length, resource
D3 cost arithmetic   : passes and tokens per pair for per-unit verification vs one-pass tagging
D4 coverage pilot    : summary-level mean(1 - omission) vs human ACU recall (RoSE) and human entity coverage (770);
                       out-of-fold R2 of ridge(counters + score) - ridge(counters), document-grouped bootstrap
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
GATES = REPO / "research" / "mars2_gates_20260916" / "code"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(GATES))
import common  # noqa: E402  (gates helpers: acu_rows, tokens, best_sentence_span, read_scores, cv)

cv = common.cv()
b3 = common._load("b3_endpoint", REPO / "scripts" / "plan2026")
BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.01)]


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    return float(cv.auroc(y, s)) if len(y) > 1 and 0 < y.sum() < len(y) else None


def seed_avg(files, rows):
    acc = defaultdict(list)
    for f in files:
        _, sc = common.read_scores(Path(f))
        for pid, v in sc.items():
            if pid in rows and len(v) == len(rows[pid]["units"]):
                acc[pid].append(v)
    return {p: np.nanmean(np.vstack(v), axis=0) for p, v in acc.items()}


def mars2_q(dump_paths, rows):
    dumps = [b3.load_dump(Path(p)) for p in dump_paths]
    out = {}
    for pid, r in rows.items():
        per = []
        for d in dumps:
            units = d.get(pid)
            if not units:
                continue
            idx = b3.align_units(units, r["units"])
            if any(j is None for j in idx):
                continue
            per.append([1 - float(units[j]["support"]) for j in idx])
        if per:
            out[pid] = np.mean(np.array(per), axis=0)
    return out


# ------------------------------------------------------------------------------------------ D1
def d1_inventory(splits):
    from mars_v2.units import get_extractor
    extractor = get_extractor("entity+proposition")
    rep = {}
    for split in splits:
        pairs = common.load_jsonl(common.B1_DIR / f"pairs_{split}.jsonl")
        units_by_src = {}
        for r in pairs:
            if r.get("units"):
                units_by_src[r["source"]] = r["units"]
        stat = defaultdict(lambda: Counter())
        kinds = Counter(); n_aligned = []
        for r in pairs:
            acus = r.get("acu_units")
            if not acus:
                continue
            src = r["source"]
            if src not in units_by_src:
                units_by_src[src] = [u.to_dict() for u in extractor(src)[:40]]
            units = units_by_src[src]
            for a in acus:
                s, e = common.best_sentence_span(src, a["text"])
                ct = set(common.tokens(a["text"], content_only=True))
                al = [u for u in units if s <= u["start"] and u["end"] <= e and ct & set(common.tokens(u["text"]))]
                res = r["resource"]; omitted = a["label"] == 0
                stat[res]["acus"] += 1; stat[res]["aligned"] += bool(al)
                if omitted:
                    stat[res]["omitted"] += 1; stat[res]["omitted_aligned"] += bool(al)
                for u in al:
                    kinds[u["kind"]] += 1
                n_aligned.append(len(al))
        cells = {}
        for res, c in stat.items():
            cells[res] = {"acus": c["acus"], "ceiling_all": c["aligned"] / max(1, c["acus"]),
                          "human_omitted": c["omitted"], "ceiling_omitted": c["omitted_aligned"] / max(1, c["omitted"])}
        tot = sum(c["omitted_aligned"] for c in stat.values()) / max(1, sum(c["omitted"] for c in stat.values()))
        rep[split] = {"cells": cells, "ceiling_omitted_pooled": tot, "aligned_unit_kinds": dict(kinds),
                      "mean_aligned_units_per_acu": float(np.mean(n_aligned)) if n_aligned else None}
        print(f"[d1] {split}: human-omitted ACUs with an aligned extracted unit = {tot:.3f}; per cell " +
              ", ".join(f"{k} {v['ceiling_omitted']:.3f}" for k, v in cells.items()), flush=True)
    return rep


# ------------------------------------------------------------------------------------------ D2
def d2_loss_anatomy(rows, systems):
    """systems: name -> pid -> per-unit omission score. Strata over the m2 units."""
    strata = defaultdict(lambda: defaultdict(lambda: ([], [])))   # stratum -> system -> (y, s)
    common_p = [p for p in rows if all(p in s for s in systems.values())]
    for pid in common_p:
        r = rows[pid]
        for i, u in enumerate(r["units"]):
            lab = r["unit_labels"][i]
            if lab is None:
                continue
            y = 1 - int(lab)
            tr = common.token_recall(u["text"], r["candidate"])
            keys = [f"recall[{lo},{hi})" for lo, hi in BINS if lo <= tr < hi] + [f"kind={u.get('kind')}",
                    f"resource={r['resource']}", f"words={'1' if len(u['text'].split()) == 1 else '2-3' if len(u['text'].split()) <= 3 else '4-7' if len(u['text'].split()) <= 7 else '8+'}"]
            for k in keys + ["ALL"]:
                for n, s in systems.items():
                    v = s[pid][i]
                    if np.isfinite(v):
                        strata[k][n][0].append(y); strata[k][n][1].append(v)
    rep = {"n_pairs": len(common_p), "strata": {}}
    for k, per in strata.items():
        rep["strata"][k] = {"n": len(next(iter(per.values()))[0]), "omitted_rate": float(np.mean(next(iter(per.values()))[0])),
                            "auroc": {n: auroc(y, s) for n, (y, s) in per.items()}}
    # disagreement: cross-encoder vs MARS-2 at 0.5
    names = list(systems)
    if len(names) >= 2:
        a, b = names[0], names[1]; dis = Counter()
        for pid in common_p:
            r = rows[pid]
            for i, lab in enumerate(r["unit_labels"]):
                if lab is None:
                    continue
                y = 1 - int(lab); pa, pb = systems[a][pid][i] > 0.5, systems[b][pid][i] > 0.5
                if pa != pb:
                    dis[f"{a}_right" if pa == bool(y) else f"{b}_right"] += 1
                else:
                    dis["agree_right" if pa == bool(y) else "agree_wrong"] += 1
        rep["disagreement_at_0.5"] = {"systems": [a, b], **dis}
    for k in sorted(rep["strata"]):
        print(f"[d2] {k:24s} n {rep['strata'][k]['n']:6d} omitted {rep['strata'][k]['omitted_rate']:.2f} " +
              " ".join(f"{n.split(':')[0]}={v if v is None else round(v, 3)}" for n, v in rep["strata"][k]["auroc"].items()), flush=True)
    return rep


# ------------------------------------------------------------------------------------------ D3
def d3_cost(rows_m2):
    n_units = [len(r["units"]) for r in rows_m2.values()]
    src_words = [len(r["source"].split()) for r in rows_m2.values()]
    cand_words = [len(r["candidate"].split()) for r in rows_m2.values()]
    mu, ms, mc = float(np.mean(n_units)), float(np.mean(src_words)), float(np.mean(cand_words))
    tok = 1.35   # words -> tokens
    per_unit_pass = (mc + 60) * tok             # cross-encoder: candidate + claim (~60 words)
    mars2_pass = (mc + 240) * tok               # MARS-2: candidate + 1,200-char window, plus a decoder pass
    one_pass = (mc + ms) * tok
    rep = {"mean_units_per_pair": mu, "mean_source_words": ms, "mean_candidate_words": mc,
           "tokens_per_pair": {"cross_encoder": mu * per_unit_pass, "mars2_encoder_only": mu * mars2_pass, "one_pass_tagger": one_pass},
           "ratio_cross_encoder_over_tagger": (mu * per_unit_pass) / one_pass}
    print(f"[d3] units/pair {mu:.1f}, source words {ms:.0f}; tokens/pair cross-encoder {rep['tokens_per_pair']['cross_encoder']:.0f} "
          f"vs one-pass {one_pass:.0f} (x{rep['ratio_cross_encoder_over_tagger']:.1f})", flush=True)
    return rep


# ------------------------------------------------------------------------------------------ D4
def oof_r2(X, y, g, alpha_grid=(0.1, 1.0, 10.0, 100.0)):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        best, best_err = None, np.inf   # inner grouped CV on alpha
        for al in alpha_grid:
            err = 0.0
            for itr, ite in GroupKFold(n_splits=3).split(Xtr, y[tr], g[tr]):
                m = Ridge(alpha=al).fit(Xtr[itr], y[tr][itr]); err += ((m.predict(Xtr[ite]) - y[tr][ite]) ** 2).sum()
            if err < best_err:
                best, best_err = al, err
        pred[te] = Ridge(alpha=best).fit(Xtr, y[tr]).predict(Xte)
    return pred


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / max(1e-12, ((y - y.mean()) ** 2).sum())


def d4_coverage(rows_m2, unit_systems, acu_rows, acu_systems, n_boot=500, seed=20260916):
    """Two targets: RoSE human ACU recall (pairs), 770 human entity coverage (pairs)."""
    rng = np.random.default_rng(seed); rep = {}
    acu_recall = {r["pair_id"]: float(np.mean(r["unit_labels"])) for r in acu_rows}
    pairs = common.load_jsonl(common.B1_DIR / "pairs_validation.jsonl")
    ent_cov = {}
    for r in pairs:
        h = r.get("human_entities") or []
        if len(h) >= 3:
            ent_cov[r["pair_id"]] = float(np.mean([x["covered"] for x in h]))

    def block(name, target, rows, systems, extra_counters):
        pids = [p for p in rows if p in target and all(p in s for s in systems.values())]
        if len(pids) < 40:
            return {"skipped": f"{len(pids)} pairs"}
        y = np.array([target[p] for p in pids]); g = np.array([common.doc_key(rows[p]["source"]) for p in pids])
        C = np.array([[len(rows[p]["candidate"].split()), len(rows[p]["candidate"]), len(rows[p]["units"]),
                       len(rows[p]["candidate"].split()) / max(1, len(rows[p]["source"].split())),
                       float(np.mean([common.token_recall(u["text"], rows[p]["candidate"]) for u in rows[p]["units"]]))] + extra_counters(p) for p in pids], float)
        pc = oof_r2(C, y, g); base = r2(y, pc)
        out = {"n_pairs": len(pids), "n_docs": int(len(set(g))), "r2_counters": base, "systems": {}}
        docs = np.unique(g); idx_by = {d: np.flatnonzero(g == d) for d in docs}
        for n, s in systems.items():
            v = np.array([float(np.nanmean(1 - s[p])) for p in pids])       # coverage = mean(1 - omission)
            ps = oof_r2(np.column_stack([C, v]), y, g)
            from scipy.stats import spearmanr
            deltas = []
            for _ in range(n_boot):
                pick = rng.choice(docs, size=len(docs), replace=True); ii = np.concatenate([idx_by[d] for d in pick])
                deltas.append(r2(y[ii], ps[ii]) - r2(y[ii], pc[ii]))
            out["systems"][n] = {"spearman": float(spearmanr(v, y).correlation), "r2_counters_plus_score": r2(y, ps),
                                 "delta_r2": r2(y, ps) - base, "delta_r2_ci": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]}
            print(f"[d4] {name}/{n}: spearman {out['systems'][n]['spearman']:.3f} R2 counters {base:.3f} -> +score {r2(y, ps):.3f} "
                  f"dR2 {out['systems'][n]['delta_r2']:+.3f} {out['systems'][n]['delta_r2_ci']}", flush=True)
        return out

    rose_m2 = {p: r for p, r in rows_m2.items() if r["resource"].startswith("rose_")}
    rep["rose_acu_recall__extracted_units"] = block("rose/extracted", acu_recall, rose_m2, unit_systems, lambda p: [])
    rep["rose_acu_recall__supplied_acus_oracle_inventory"] = block("rose/acu-oracle", acu_recall, {r["pair_id"]: r for r in acu_rows}, acu_systems, lambda p: [])
    nat = {p: r for p, r in rows_m2.items() if r["resource"] == "natural770"}
    rep["natural770_entity_coverage__extracted_units"] = block("770/extracted", ent_cov, nat, unit_systems, lambda p: [])
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", default=str(common.OUT_ROOT))
    ap.add_argument("--dumps-root", default="/home/user/results/plan2026/b2/scores")
    ap.add_argument("--out", default="/home/user/results/mars3/stage0")
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); skip = set(a.skip.split(","))
    rep = {"registered": "research/mars3_20260916/PLAN.md M30"}
    rows_m2 = {r["pair_id"]: r for r in common.m2_rows("validation")}
    acu_rows = common.acu_rows("validation")
    xenc_m2 = seed_avg([f for f in glob.glob(f"{a.gates}/b21/scores/m2_xenc_seed*.jsonl") if "xdom" not in f], rows_m2)
    m2q = {}
    for arm in ("direct_enabled", "full_recovery"):
        m2q[f"mars2:{arm}"] = mars2_q(sorted(glob.glob(f"{a.dumps_root}/{arm}_seed*/units_validation.jsonl")), rows_m2)
    unit_systems = {"xenc:deberta-large": xenc_m2, **m2q,
                    "counter:1-token_recall": {p: np.array([1 - common.token_recall(u["text"], r["candidate"]) for u in r["units"]]) for p, r in rows_m2.items()}}
    if "d1" not in skip:
        rep["d1_inventory_ceiling"] = d1_inventory(["validation", "test"]); common.dump_json(out / "stage0.json", rep)
    if "d2" not in skip:
        rep["d2_loss_anatomy_m2"] = d2_loss_anatomy(rows_m2, {k: v for k, v in unit_systems.items() if k.startswith(("xenc", "mars2:direct", "counter"))})
        common.dump_json(out / "stage0.json", rep)
    if "d3" not in skip:
        rep["d3_cost"] = d3_cost(rows_m2); common.dump_json(out / "stage0.json", rep)
    if "d4" not in skip:
        acu_map = {r["pair_id"]: r for r in acu_rows}
        acu_systems = {"xenc:deberta-large": seed_avg([f for f in glob.glob(f"{a.gates}/b21/scores/acu_xenc_seed*.jsonl") if "xdom" not in f], acu_map)}
        mq = defaultdict(list)
        for f in glob.glob(f"{a.gates}/b18/mars2/direct_enabled_seed*_validation.jsonl"):
            for rec in common.load_jsonl(Path(f)):
                if rec["pair_id"] in acu_map and rec["n_units"] == len(acu_map[rec["pair_id"]]["units"]):
                    mq[rec["pair_id"]].append(1 - np.asarray(rec["variants"]["masked"]["q"], float))
        acu_systems["mars2:direct_enabled|masked"] = {p: np.mean(np.vstack(v), axis=0) for p, v in mq.items()}
        acu_systems["counter:1-token_recall"] = {p: np.array([1 - common.token_recall(u["text"], r["candidate"]) for u in r["units"]]) for p, r in acu_map.items()}
        rep["d4_coverage_pilot"] = d4_coverage(rows_m2, unit_systems, acu_rows, acu_systems); common.dump_json(out / "stage0.json", rep)
    print("[stage0] written", out / "stage0.json", flush=True)


if __name__ == "__main__":
    main()
