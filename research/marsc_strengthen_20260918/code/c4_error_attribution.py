#!/usr/bin/env python3
"""MARS-C strengthen C4 / H-ATTR -- where does the end-to-end error actually live?

Registered in research/marsc_strengthen_20260918/PREREG.md (Wave C / C4).  No bar: this is the "where is the
error" decomposition reviewers ask for, and it is the machine-computable half of the governing proposal's
Block 1 bottleneck audit.  It is decision-relevant: if INVENTORY misses dominate, the proposal's matched-
supervision training programme is aimed at the wrong stage, and the paper should say so.

For every human-marked OMITTED ACU that a system fails to surface in its top-k, the miss is classified into
exactly one of three mutually exclusive, exhaustive causes:

  (i)   INVENTORY MISS  no inventory unit overlaps the ACU at all, so no emitter over this inventory could
                        ever have surfaced it.  "Overlaps" is the SAME span+token rule mc_e2_eval.py uses to
                        credit a match, so the attribution is consistent with the endpoint:
                          unit.start < acu_sentence_end  and  unit.end > acu_sentence_start
                          and  |content_tokens(acu) & tokens(unit)| >= 1
                        (this is also exactly m32_inventory.py's `weak` rule, so the complement of the
                        INVENTORY MISS rate reproduces the published inventory ceiling).
  (ii)  VERIFIER MISS   a unit overlaps, but no overlapping unit reaches the budget under the RAW score either.
                        The verifier ranked the right region too low.
  (iii) RULE MISS       a unit overlaps AND an overlapping unit is inside the budget under the RAW score, but
                        after the div1 one-unit-per-sentence rule applies its fixed -5.0 demotion no
                        overlapping unit is left inside the budget.  The emission rule, not the verifier,
                        lost this fact.

Exhaustive and mutually exclusive by construction: an ACU whose overlapping unit survives into the div1 top-k
is a HIT, so a MISS with an overlapping unit in the raw top-k must have lost it to the rule.

Also reported: the inventory CEILING (share of human-omitted ACUs any emitter over this inventory could in
principle reach), per resource and per pool, for reconciliation against
/home/user/results/mars3/m32/ceilings.json; and per-pair macro recall@k, which must reproduce the recall
mc_e2_eval.py prints for the same system from the same manifests -- that equality is this file's self-check.

Every share carries a DOCUMENT-CLUSTERED bootstrap interval (clusters = common.doc_key(source), the evaluator's
own clustering variable, rng seed 20260916 as in mc_e2_eval.py).

System spec (';' separated, so a path may contain ':'):
    NAME;<inventory jsonl>;<raw score manifest, comma separated>;<div1 score manifest, comma separated>
Both manifests must be explicit comma lists of distinct numeric seeds -- a glob is refused (standing rule 1).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402

CAUSES = ("inventory", "verifier", "rule")
REGISTERED = "research/marsc_strengthen_20260918/PREREG.md -- Wave C / C4, hypothesis H-ATTR"


def manifest(spec: str, expect_seeds: int) -> list[Path]:
    assert "*" not in spec and "?" not in spec, f"c4 refuses a glob score source: {spec!r}"
    files = [Path(x.strip()) for x in spec.split(",") if x.strip()]
    assert files, f"empty score manifest {spec!r}"
    missing = [str(f) for f in files if not f.exists()]
    assert not missing, f"score manifest lists files that do not exist: {missing}"
    toks = [(m.group(1) if (m := re.search(r"seed([A-Za-z0-9]+)", f.name)) else "") for f in files]
    if len(files) > 1:
        bad = [t for t in toks if not t.isdigit()]
        assert not bad, f"multi-file roster carries pseudo-seed token(s) {bad}: {spec!r}"
        assert len(set(toks)) == len(files), f"roster repeats a seed {toks}: {spec!r}"
    if expect_seeds:
        assert len(files) == expect_seeds, f"expected {expect_seeds} seed files, got {len(files)}: {spec!r}"
    return files


def seed_average(files: list[Path], sizes: dict[str, int], pairs: set[str]) -> dict[str, np.ndarray]:
    """mc_e2_eval.load_emitter's averaging, verbatim: keep only pairs whose score vector matches the
    inventory length and that are present in the ACU file; average across the seed roster with nanmean."""
    acc = defaultdict(list)
    for f in files:
        _, sc = common.read_scores(f)
        for pid, v in sc.items():
            if pid in pairs and pid in sizes and len(v) == sizes[pid]:
                acc[pid].append(v)
    return {p: np.nanmean(np.vstack(v), axis=0) for p, v in acc.items() if len(v) == len(files)}


def topk(units: list[dict], sc: np.ndarray, k: int) -> list[int]:
    """mc_e2_eval.per_pair's ordering, verbatim."""
    return sorted(range(len(units)),
                  key=lambda j: (-sc[j] if np.isfinite(sc[j]) else 1.0, units[j].get("start", 0)))[:k]


def acu_targets(acu_row: dict) -> list[tuple[tuple[int, int], set]]:
    """mc_e2_eval.py's ACU side of the match rule: the best-matching source sentence span and the ACU's
    content tokens (falling back to all tokens when the ACU is entirely stopwords), cached per pair."""
    src = acu_row["source"]
    return [(common.best_sentence_span(src, u["text"]),
             set(common.tokens(u["text"], content_only=True)) or set(common.tokens(u["text"])))
            for u in acu_row["units"]]


def unit_keys(units: list[dict]) -> list[tuple[int, int, set]]:
    """mc_e2_eval.py's emitted-unit side: the span and the unit's ALL-token set (stopwords included)."""
    return [(u.get("start", -1), u.get("end", -1), set(common.tokens(u["text"]))) for u in units]


def overlap_map(targets: list[tuple[tuple[int, int], set]], ut: list[tuple[int, int, set]]) -> list[list[int]]:
    """For each ACU index, the inventory units mc_e2_eval.py would credit as a match."""
    return [[j for j, (us, ue, tk) in enumerate(ut) if us < e and ue > s and (ct & tk)]
            for (s, e), ct in targets]


def boot_shares(rows: list[tuple], docs: list[str], by_doc: dict, n_boot: int, seed: int) -> dict:
    """rows = (doc, resource, cause) with cause in {'hit'} | CAUSES.  Document-clustered percentile bootstrap
    of every share, both as a share of ALL omitted ACUs and as a share of the MISSES."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(docs))
    keys = ["hit", *CAUSES]
    draws = {f"{k}_of_omitted": [] for k in keys}
    draws.update({f"{k}_of_misses": [] for k in CAUSES})
    draws["ceiling"] = []
    for _ in range(n_boot):
        pick = idx[rng.integers(0, len(docs), size=len(docs))]
        c = Counter()
        for i in pick:
            c.update(by_doc[docs[i]])
        tot = sum(c[k] for k in keys)
        if not tot:
            continue
        miss = tot - c["hit"]
        for k in keys:
            draws[f"{k}_of_omitted"].append(c[k] / tot)
        for k in CAUSES:
            draws[f"{k}_of_misses"].append((c[k] / miss) if miss else None)
        draws["ceiling"].append((tot - c["inventory"]) / tot)
    out = {}
    for k, v in draws.items():
        vv = [x for x in v if x is not None]
        out[k] = {"ci95": [float(np.percentile(vv, 2.5)), float(np.percentile(vv, 97.5))],
                  "boot_mean": float(np.mean(vv))} if vv else None
    return out


def summarise(rows: list[tuple], n_boot: int, seed: int) -> dict:
    """rows = (doc, resource, cause).  OBSERVED shares first (never the bootstrap mean), then the interval."""
    c = Counter(x[2] for x in rows)
    tot = len(rows)
    miss = tot - c["hit"]
    by_doc = defaultdict(Counter)
    for d, _, cause in rows:
        by_doc[d][cause] += 1
    docs = sorted(by_doc)
    rep = {"n_omitted_acus": tot, "n_docs": len(docs), "n_hit": c["hit"], "n_miss": miss,
           "observed": {"hit_of_omitted": (c["hit"] / tot) if tot else None,
                        **{f"{k}_of_omitted": (c[k] / tot) if tot else None for k in CAUSES},
                        **{f"{k}_of_misses": (c[k] / miss) if miss else None for k in CAUSES},
                        "ceiling": ((tot - c["inventory"]) / tot) if tot else None},
           "counts": {k: c[k] for k in ("hit", *CAUSES)},
           "document_clustered_bootstrap": boot_shares(rows, docs, by_doc, n_boot, seed) if tot else None}
    byres = defaultdict(list)
    for d, res, cause in rows:
        byres[res].append((d, res, cause))
    rep["by_resource"] = {}
    for res, rr in sorted(byres.items()):
        cc = Counter(x[2] for x in rr)
        t = len(rr)
        m = t - cc["hit"]
        rep["by_resource"][res] = {"n_omitted_acus": t, "counts": {k: cc[k] for k in ("hit", *CAUSES)},
                                   "hit_of_omitted": cc["hit"] / t,
                                   **{f"{k}_of_omitted": cc[k] / t for k in CAUSES},
                                   **{f"{k}_of_misses": (cc[k] / m) if m else None for k in CAUSES},
                                   "ceiling": (t - cc["inventory"]) / t}
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acu", required=True)
    ap.add_argument("--pool", required=True, help="label for this pool, e.g. validation / test / unisum")
    ap.add_argument("--systems", nargs="+", required=True,
                    help="NAME;<units jsonl>;<raw manifest>;<div1 manifest>")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260916, help="mc_e2_eval.py's bootstrap seed")
    ap.add_argument("--expect-seeds", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    acu = {r["pair_id"]: r for r in common.load_jsonl(Path(a.acu))}
    print(f"[c4] {a.pool}: {len(acu)} ACU pairs from {a.acu}", flush=True)

    specs = []
    for s in a.systems:
        parts = s.split(";")
        assert len(parts) == 4, f"system spec needs NAME;UNITS;RAW;DIV, got {s!r}"
        name, upath, rawspec, divspec = parts
        specs.append({"name": name, "units_file": upath,
                      "raw": manifest(rawspec, a.expect_seeds), "div": manifest(divspec, a.expect_seeds)})

    inv_cache: dict[str, tuple[dict, dict]] = {}
    for sp in specs:
        u = sp["units_file"]
        if u not in inv_cache:
            rows = {r["pair_id"]: r["units"] for r in common.load_jsonl(Path(u)) if r.get("units")}
            inv_cache[u] = (rows, {p: len(v) for p, v in rows.items()})
            print(f"[c4] inventory {u}: {len(rows)} pairs, {sum(len(v) for v in rows.values())} units", flush=True)

    pairset = set(acu)
    for sp in specs:
        units, sizes = inv_cache[sp["units_file"]]
        sp["raw_sc"] = seed_average(sp["raw"], sizes, pairset)
        sp["div_sc"] = seed_average(sp["div"], sizes, pairset)
        sp["pairs"] = set(sp["raw_sc"]) & set(sp["div_sc"])
        print(f"[c4] {sp['name']}: {len(sp['raw_sc'])} raw / {len(sp['div_sc'])} div1 pairs "
              f"-> {len(sp['pairs'])} usable", flush=True)
    common_p = sorted(set.intersection(*[sp["pairs"] for sp in specs]))
    assert common_p, "no pair is common to every system"
    print(f"[c4] {a.pool}: {len(common_p)} pairs common to all {len(specs)} systems", flush=True)

    # overlap maps, computed once per (inventory, pair) and shared across every system over that inventory
    tgt_cache: dict[str, list] = {}

    def targets(p: str):
        if p not in tgt_cache:
            tgt_cache[p] = acu_targets(acu[p])
        return tgt_cache[p]

    omap: dict[tuple[str, str], list[list[int]]] = {}

    def ovmap(u: str, p: str):
        key = (u, p)
        if key not in omap:
            omap[key] = overlap_map(targets(p), unit_keys(inv_cache[u][0][p]))
        return omap[key]

    for sp in specs:
        for p in common_p:
            ovmap(sp["units_file"], p)

    rep = {"registered": REGISTERED, "pool": a.pool, "acu_file": a.acu, "k": a.k,
           "n_pairs_common": len(common_p), "bootstrap_seed": a.seed, "n_boot": a.n_boot,
           "cause_definitions": {
               "inventory": "no inventory unit overlaps the ACU under mc_e2_eval.py's span+content-token rule",
               "verifier": "an overlapping unit exists but none reaches the top-k under the RAW seed-averaged "
                           "score",
               "rule": "an overlapping unit is inside the RAW top-k but the div1 -5.0 demotion pushed every "
                       "overlapping unit out of the final top-k",
               "hit": "an overlapping unit is inside the FINAL (div1) top-k -- mc_e2_eval.py credits the match"},
           "systems": {}}

    # inventory ceilings, per inventory, on the pairs evaluated here and on the whole ACU file
    rep["inventory_ceiling"] = {}
    for u, (units, _) in inv_cache.items():
        for scope, pids in (("evaluated_pairs", common_p),
                            ("all_acu_pairs", [p for p in acu if p in units])):
            tot = om = 0
            byres = defaultdict(lambda: [0, 0])
            for p in pids:
                ov = ovmap(u, p)
                truth = common.truth_of(acu[p])
                res = acu[p].get("resource", "?")
                for i, t in enumerate(truth):
                    if t == 1:
                        tot += 1
                        byres[res][0] += 1
                        if ov[i]:
                            om += 1
                            byres[res][1] += 1
            rep["inventory_ceiling"].setdefault(u, {})[scope] = {
                "n_pairs": len(pids), "human_omitted_acus": tot,
                "ceiling_weak_pooled": (om / tot) if tot else None,
                "by_resource": {r: {"human_omitted": v[0], "ceiling_weak": v[1] / v[0]} for r, v in sorted(byres.items())}}
            print(f"[c4] ceiling {Path(u).name} [{scope}]: {om}/{tot} = "
                  f"{(om / tot) if tot else float('nan'):.4f}", flush=True)

    for sp in specs:
        units, _ = inv_cache[sp["units_file"]]
        rows, pair_recall = [], []
        for p in common_p:
            r = acu[p]
            truth = common.truth_of(r)
            om_idx = [i for i, t in enumerate(truth) if t == 1]
            if not om_idx:
                continue
            ov = omap[(sp["units_file"], p)]
            tk_div = set(topk(units[p], sp["div_sc"][p], a.k))
            tk_raw = set(topk(units[p], sp["raw_sc"][p], a.k))
            d = common.doc_key(r["source"])
            res = r.get("resource", "?")
            nhit = 0
            for i in om_idx:
                cand = ov[i]
                if any(j in tk_div for j in cand):
                    cause = "hit"
                    nhit += 1
                elif not cand:
                    cause = "inventory"
                elif any(j in tk_raw for j in cand):
                    cause = "rule"
                else:
                    cause = "verifier"
                rows.append((d, res, cause))
            pair_recall.append(nhit / len(om_idx))
        s = summarise(rows, a.n_boot, a.seed)
        s["macro_recall_at_k"] = float(np.mean(pair_recall)) if pair_recall else None
        s["n_pairs_with_omitted_acu"] = len(pair_recall)
        s["units_file"] = sp["units_file"]
        s["raw_manifest"] = [str(x) for x in sp["raw"]]
        s["div_manifest"] = [str(x) for x in sp["div"]]
        rep["systems"][sp["name"]] = s
        o = s["observed"]
        b = s["document_clustered_bootstrap"] or {}

        def f(x):
            return "n/a" if x is None else f"{x:.4f}"

        def ci(key):
            v = b.get(key)
            return "" if not v else f" [{v['ci95'][0]:.4f},{v['ci95'][1]:.4f}]"

        print(f"[c4] {a.pool} {sp['name']:34s} macro recall@{a.k} {f(s['macro_recall_at_k'])} | "
              f"pooled hit {f(o['hit_of_omitted'])} | of ALL omitted: "
              f"inv {f(o['inventory_of_omitted'])} ver {f(o['verifier_of_omitted'])} "
              f"rule {f(o['rule_of_omitted'])} | of MISSES: "
              f"inv {f(o['inventory_of_misses'])}{ci('inventory_of_misses')} "
              f"ver {f(o['verifier_of_misses'])}{ci('verifier_of_misses')} "
              f"rule {f(o['rule_of_misses'])}{ci('rule_of_misses')} | "
              f"ceiling {f(o['ceiling'])}{ci('ceiling')}", flush=True)

    dom = {n: max(CAUSES, key=lambda c: s["observed"][f"{c}_of_misses"] or 0.0) for n, s in rep["systems"].items()}
    rep["dominant_miss_cause"] = dom
    rep["decision_note"] = (
        "If INVENTORY dominates, the bottleneck is decomposition coverage, not the verifier, and a matched-"
        "supervision training programme aimed at the verifier is aimed at the wrong stage. If VERIFIER "
        "dominates, verifier training is the right target. If RULE dominates, the emission rule is throwing "
        "away facts the verifier already ranked correctly.")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"c4_attribution_{a.pool}.json").write_text(json.dumps(rep, indent=1))
    print(f"[c4] dominant miss cause: {dom}", flush=True)
    print(f"[c4] written {out / f'c4_attribution_{a.pool}.json'}", flush=True)


if __name__ == "__main__":
    main()
