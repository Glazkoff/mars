#!/usr/bin/env python3
"""H-J10a -- OmissionBench's own metric from b2's score cache: paired discrimination against the clean twin.

The benchmark scores a judge on whether the errored note scores strictly below its own clean twin (ties count
half; chance 0.500). In our omission-score convention that is z(fact | errored note) > z(fact | clean twin)
for the pair's own removed fact, i.e. the subset of b2's same-fact triples whose conveying note is the clean
twin. This script reads the per-arm cell scores b2 wrote (`<cache>/<family>.seed<S>[.w..].json`, averaged
over seeds, or the single file of a zero-shot / lexical / embedding family) and reports, per family, the
pair-pooled paired discrimination (the benchmark's definition) and the consultation-macro version, both with
consultation-clustered percentile bootstrap intervals, and the breakdown by residual level, severity, stratum
and by whether the first removed site lies beyond the deployed 380-word clip. Nothing is rescored here.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_family(cache: Path, name: str) -> dict[str, float] | None:
    files = sorted(glob.glob(str(cache / f"{name}.seed*.json")))
    files = [f for f in files if not f.endswith(".receipt.json")]
    if not files:
        files = [f for f in sorted(glob.glob(str(cache / f"{name}.*.json"))) if not f.endswith(".receipt.json")]
    if not files:
        return None
    acc: dict[str, list[float]] = defaultdict(list)
    for f in files:
        b = json.loads(Path(f).read_text())
        for c, z in zip(b["cells"], b["z"]):
            acc[c].append(float(z))
    return {c: float(np.mean(v)) for c, v in acc.items()}


def boot_ci(vals_by_doc: dict[str, list[float]], rng, n_boot: int, macro: bool) -> tuple[float, list[float]]:
    keys = sorted(vals_by_doc)
    per = [np.array(vals_by_doc[k], dtype=float) for k in keys]
    obs = float(np.mean([p.mean() for p in per])) if macro else float(np.concatenate(per).mean())
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(keys), size=len(keys))
        if macro:
            draws.append(np.mean([per[i].mean() for i in idx]))
        else:
            draws.append(np.concatenate([per[i] for i in idx]).mean())
    return obs, [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--primary", default="humanfact")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--clip", type=int, default=380)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    docs = [json.loads(l) for l in open(a.docs, encoding="utf-8")]
    pairs = []  # (doc index, doc key, errored si, clean si, fi, strata)
    for di, d in enumerate(docs):
        clean = [si for si, s in enumerate(d["summaries"]) if s["system"] == "clean"]
        assert len(clean) == 1, f"doc {d['doc_key']} has {len(clean)} clean twins"
        for si, s in enumerate(d["summaries"]):
            if s["system"] != "errored":
                continue
            pairs.append({"di": di, "key": d["doc_key"], "si": si, "ci": clean[0], "fi": s["fact_index"],
                          "pair_id": s["pair_id"], "residual_level": str(s.get("residual_level")),
                          "severity": s.get("severity"), "stratum": s.get("domain"),
                          "beyond_clip": "beyond" if s.get("first_diff_word", 0) >= a.clip else "within"})
    rep = {"docs": a.docs, "cache": a.cache, "n_pairs": len(pairs), "n_consultations": len(docs),
           "metric": "paired discrimination: z(fact|errored) > z(fact|clean twin), ties 1/2, chance 0.500",
           "families": {}, "missing": []}
    per_pair_win: dict[str, dict[str, float]] = {}
    rng = np.random.default_rng(a.seed)
    for name in a.families:
        z = load_family(Path(a.cache), name)
        if z is None:
            rep["missing"].append(name)
            print(f"[twin] {name}: no cache file, skipped", flush=True)
            continue
        wins = {}
        for p in pairs:
            ze, zc = z[f"{p['di']}|{p['si']}|{p['fi']}"], z[f"{p['di']}|{p['ci']}|{p['fi']}"]
            wins[p["pair_id"]] = 1.0 if ze > zc else (0.5 if ze == zc else 0.0)
        per_pair_win[name] = wins
        by_doc = defaultdict(list)
        for p in pairs:
            by_doc[p["key"]].append(wins[p["pair_id"]])
        pooled, ci_pooled = boot_ci(by_doc, np.random.default_rng(a.seed), a.n_boot, macro=False)
        macro, ci_macro = boot_ci(by_doc, np.random.default_rng(a.seed), a.n_boot, macro=True)
        ent = {"paired_discrimination": pooled, "ci95_consultation_bootstrap": ci_pooled,
               "consultation_macro": macro, "ci95_macro": ci_macro,
               "n_ties": int(sum(1 for v in wins.values() if v == 0.5)), "by": {}}
        for facet in ("residual_level", "severity", "stratum", "beyond_clip"):
            groups = defaultdict(lambda: defaultdict(list))
            for p in pairs:
                groups[p[facet]][p["key"]].append(wins[p["pair_id"]])
            ent["by"][facet] = {}
            for g, bd in sorted(groups.items()):
                o, ci = boot_ci(bd, np.random.default_rng(a.seed), a.n_boot, macro=False)
                ent["by"][facet][g] = {"n_pairs": int(sum(len(v) for v in bd.values())), "paired_discrimination": o,
                                       "ci95": ci}
        rep["families"][name] = ent
        print(f"[twin] {name:22s} paired discrimination {pooled:.4f} [{ci_pooled[0]:.4f},{ci_pooled[1]:.4f}] "
              f"(macro {macro:.4f}; ties {ent['n_ties']})", flush=True)
    # paired margins of the primary over each family, consultation-clustered
    rep["margins_vs_primary"] = {}
    if a.primary in per_pair_win:
        for name, wins in per_pair_win.items():
            if name == a.primary:
                continue
            by_doc = defaultdict(list)
            for p in pairs:
                by_doc[p["key"]].append(per_pair_win[a.primary][p["pair_id"]] - wins[p["pair_id"]])
            obs, ci = boot_ci(by_doc, np.random.default_rng(a.seed), a.n_boot, macro=False)
            keys = sorted(by_doc); per = [np.array(by_doc[k]) for k in keys]
            draws = [np.concatenate([per[i] for i in rng.integers(0, len(keys), size=len(keys))]).mean()
                     for _ in range(a.n_boot)]
            rep["margins_vs_primary"][name] = {"observed": obs, "ci95_two_sided": ci,
                                               "lower_bound_95_one_sided": float(np.percentile(draws, 5.0))}
            print(f"[twin] margin {a.primary} - {name:22s} {obs:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]", flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"[twin] written {a.out}", flush=True)


if __name__ == "__main__":
    main()
