#!/usr/bin/env python3
"""MARS-C E1 analysis -- every arm on the validation human ACUs, with the prospective screen bars.

Per system (seed files averaged): pooled AUROC, document-macro AUROC, lexical-stress AUROC (ACU not verbatim in the
summary and content-token recall < .5), calibrated log-loss, per-resource pooled AUROC, strict four-cell success and
sign(C) rate on validation tetrads (from raw logits; <= 50 tetrads per document, fixed seed).
Paired document bootstrap of the primary (D) against every control on lexical-stress and overall AUROC.
Screen: D - strongest control >= +.02 lexical-stress with CI lower bound > 0, and overall degradation <= .005.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402

cv = common.cv()
_WS = re.compile(r"\s+")


def norm(t):
    return _WS.sub(" ", re.sub(r"[^\w\s]", " ", t.lower())).strip()


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def load_group(files, rows):
    acc = defaultdict(list); names = set()
    for f in files:
        name, sc = common.read_scores(Path(f)); names.add(name)
        for pid, v in sc.items():
            if pid in rows and len(v) == len(rows[pid]["units"]):
                acc[pid].append(v)
    return (names.pop() if names else None), {p: np.nanmean(np.vstack(v), axis=0) for p, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--acu", default=str(common.OUT_ROOT / "b18" / "acu_units_validation.jsonl"))
    ap.add_argument("--scores-glob", required=True, help="calibrated unit_scores, e.g. .../scores/acu_mc-*_seed*.jsonl (logits files excluded automatically)")
    ap.add_argument("--primary", default="D")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = {r["pair_id"]: r for r in common.load_jsonl(Path(a.acu))}
    files = [f for f in sorted(glob.glob(a.scores_glob)) if not f.endswith("_logits.jsonl")]
    by_sys = defaultdict(list)
    for f in files:
        by_sys[re.sub(r"_seed\d+\.jsonl$", "", Path(f).name)].append(f)
    systems, logits = {}, {}
    for key, fs in by_sys.items():
        name, sc = load_group(fs, rows)
        name = key[4:] if key.startswith("acu_") else key        # name by file stem: fold/variant runs share the in-file system name
        if sc:
            systems[name] = sc
            _, lg = load_group([f.replace(".jsonl", "_logits.jsonl") for f in fs if Path(f.replace(".jsonl", "_logits.jsonl")).exists()], rows)
            logits[name] = lg
    if not systems:
        raise SystemExit("no systems")
    common_p = sorted(p for p in rows if all(p in s for s in systems.values()))
    doc_of = {p: common.doc_key(rows[p]["source"]) for p in common_p}; docs = sorted(set(doc_of.values()))
    stress = {p: [c["ca_unit_in_cand"] == 0.0 and c["ca_token_recall"] < 0.5 for c in (common.counters_for("acu", rows[p], i) for i in range(len(rows[p]["units"])))] for p in common_p}
    # validation tetrads from the E0 docs (fact text -> ACU index by normalised text)
    vdocs = common.load_jsonl(Path(a.data_dir) / "docs_validation.jsonl"); rng = random.Random(20260916); tetrads = []
    for d in vdocs:
        pid_of = {s["pair_id"]: s for s in d["summaries"]}
        T = d.get("tetrads", [])
        for f, g, i, j in rng.sample(T, min(50, len(T))):
            si, sj = d["summaries"][i]["pair_id"], d["summaries"][j]["pair_id"]
            if si in rows and sj in rows:
                def idx(pid, ftxt):
                    n = norm(ftxt)
                    for k, u in enumerate(rows[pid]["units"]):
                        if norm(u["text"]) == n:
                            return k
                    return None
                fi, gi, fj, gj = idx(si, d["facts"][f]), idx(si, d["facts"][g]), idx(sj, d["facts"][f]), idx(sj, d["facts"][g])
                if None not in (fi, gi, fj, gj):
                    tetrads.append((si, sj, fi, gi, fj, gj))

    def metrics(name, pids):
        s = systems[name]; y, v, ys, vs, macro = [], [], [], [], []
        for p in pids:
            t = common.truth_of(rows[p]); sc = s[p]
            yy = np.array([tt for tt in t if tt is not None]); ss = np.array([sc[i] for i, tt in enumerate(t) if tt is not None])
            y += yy.tolist(); v += ss.tolist()
            ys += [tt for i, tt in enumerate(t) if tt is not None and stress[p][i]]; vs += [sc[i] for i, tt in enumerate(t) if tt is not None and stress[p][i]]
            if len(yy) > 1 and 0 < yy.sum() < len(yy):
                macro.append(cv.auroc(yy, ss))
        y, v, ys, vs = map(np.array, (y, v, ys, vs))
        return {"auroc_pooled": float(cv.auroc(y, v)) if len(y) else None, "auroc_macro": float(np.mean(macro)) if macro else None,
                "auroc_lexical_stress": float(cv.auroc(ys, vs)) if len(ys) and 0 < ys.sum() < len(ys) else None, "n_stress": int(len(ys)), "logloss": logloss(y, v) if len(y) else None}

    rep = {"registered": "research/marsc_20260916/PREREG.md E1", "n_pairs": len(common_p), "n_docs": len(docs), "n_validation_tetrads": len(tetrads), "systems": {}}
    resources = sorted({rows[p]["resource"] for p in common_p})
    for name in systems:
        e = metrics(name, common_p); e["by_resource"] = {r: metrics(name, [p for p in common_p if rows[p]["resource"] == r])["auroc_pooled"] for r in resources}
        lg = logits.get(name, {})
        if lg and tetrads:
            four = []; csign = []
            for si, sj, fi, gi, fj, gj in tetrads:
                if si in lg and sj in lg:
                    z11, z21, z12, z22 = lg[si][fi], lg[si][gi], lg[sj][fj], lg[sj][gj]
                    four.append(z12 > z11 and z21 > z22 and z21 > z11 and z12 > z22); csign.append((z12 + z21 - z11 - z22) > 0)
            e["four_cell_strict"] = float(np.mean(four)) if four else None; e["C_positive_rate"] = float(np.mean(csign)) if csign else None; e["n_tetrads_scored"] = len(four)
        rep["systems"][name] = e
        print(f"[e1] {name:28s} pooled {e['auroc_pooled']:.3f} macro {e['auroc_macro']:.3f} stress {e['auroc_lexical_stress']:.3f} (n {e['n_stress']}) logloss {e['logloss']:.3f} four-cell {e.get('four_cell_strict')} C>0 {e.get('C_positive_rate')}", flush=True)
    # paired document bootstrap: primary vs every other system, lexical-stress and overall pooled AUROC
    prim = next((n for n in systems if n.startswith(f"mc-{a.primary}-") or n.startswith(f"mc:{a.primary}|")), None); rep["primary"] = prim; rep["paired"] = {}
    if prim:
        rng = np.random.default_rng(20260916); by_doc = defaultdict(list)
        for p in common_p:
            by_doc[doc_of[p]].append(p)
        for other in systems:
            if other == prim:
                continue
            ds, do = [], []
            for _ in range(a.n_boot):
                pick = rng.choice(docs, size=len(docs), replace=True); sub = [p for d in pick for p in by_doc[d]]
                mp, mo = metrics(prim, sub), metrics(other, sub)
                if mp["auroc_lexical_stress"] is not None and mo["auroc_lexical_stress"] is not None:
                    ds.append(mp["auroc_lexical_stress"] - mo["auroc_lexical_stress"])
                do.append(mp["auroc_pooled"] - mo["auroc_pooled"])
            rep["paired"][other] = {"lexical_stress": {"mean": float(np.mean(ds)), "ci": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]} if ds else None,
                                    "pooled": {"mean": float(np.mean(do)), "ci": [float(np.percentile(do, 2.5)), float(np.percentile(do, 97.5))]}}
        controls = [n for n in systems if n != prim and (n.startswith("mc:") or n.startswith("mc-"))]
        if controls:
            best = max(controls, key=lambda n: rep["systems"][n]["auroc_lexical_stress"] or 0)
            pr = rep["paired"][best]
            rep["screen"] = {"strongest_control": best, "delta_lexical_stress": rep["systems"][prim]["auroc_lexical_stress"] - rep["systems"][best]["auroc_lexical_stress"],
                             "ci_lexical_stress": pr["lexical_stress"]["ci"] if pr["lexical_stress"] else None,
                             "delta_pooled": rep["systems"][prim]["auroc_pooled"] - rep["systems"][best]["auroc_pooled"],
                             "generic_contrast_ties": any((n.startswith("mc:C|") or n.startswith("mc-C-")) and rep["paired"][n]["lexical_stress"] and rep["paired"][n]["lexical_stress"]["ci"][0] <= 0 for n in controls)}
            s = rep["screen"]; s["PASS"] = bool(s["delta_lexical_stress"] >= 0.02 and s["ci_lexical_stress"] and s["ci_lexical_stress"][0] > 0 and s["delta_pooled"] >= -0.005)
            s["HEADROOM"] = bool(s["delta_lexical_stress"] >= 0.0)
            print(f"[e1] screen: {json.dumps(s)}", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / "e1_screen.json").write_text(json.dumps(rep, indent=1))
    print("[e1] written", out / "e1_screen.json", flush=True)


if __name__ == "__main__":
    main()
