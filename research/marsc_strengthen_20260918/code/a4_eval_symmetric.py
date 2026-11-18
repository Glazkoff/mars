#!/usr/bin/env python3
"""MARS-C E2 evaluation -- emitted top-k missing facts, matched to HUMAN omissions (docs/FINAL_PROPOSAL E2).

Every system is an EMITTER: an inventory of source units (frozen) + an omission score per unit; it emits its top-k units.
An emitted unit is matched to the human ACUs whose best-matching source sentence overlaps the unit's span and which share
a content token with it. Per pair: recall@k = share of human-OMITTED ACUs matched by >= 1 emitted unit (pairs with >= 1
omitted ACU); emitted-unit status = hit (matches an omitted ACU) / false alert (matches only covered ACUs) / unknown (no
ACU). Recall on the lexical-stress ACUs separately. Document-paired bootstrap between systems. No gold ACU enters any emitter.

A4 FORK of research/marsc_20260916/code/mc_e2_eval.py (campaign marsc_strengthen_20260918, PREREG "Wave A / A4").
The ONLY behavioural difference is the opt-in --match-symmetric flag. Upstream, the ACU side of the token match is
built content-only (`common.tokens(text, content_only=True)` with an all-token fallback) while the emitted-unit side
is built with ALL tokens including stopwords. The shared-token floor is unaffected by that (a content-only ACU set
cannot intersect a stopword), but the Jaccard floor is not: the emitted unit's stopwords enter the UNION only, so the
measured Jaccard is deflated and the floor bites harder than registered. This is why the Jaccard-floor sensitivity
rows were withdrawn from the paper. --match-symmetric builds BOTH sides content-only, with the same fallback, and
changes nothing else. Everything outside the flagged expression is byte-identical to the upstream file.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code")); sys.path.insert(0, str(REPO / "scripts" / "plan2026"))
import common  # noqa: E402

b3 = common._load("b3_endpoint", REPO / "scripts" / "plan2026")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_files(pattern: str) -> list[str]:
    """A score source is either an explicit comma-separated manifest of paths or a single glob.

    An explicit manifest is the registered form: every path must exist, so a silently missing or a silently
    EXTRA file (the `*_seedavg_div1.jsonl` that a `seed*_div1.jsonl` glob used to sweep into a nominally
    three-seed consensus) cannot change what the estimator averages.
    """
    if "," in pattern:
        files = [x.strip() for x in pattern.split(",") if x.strip()]
        missing = [f for f in files if not Path(f).exists()]
        assert not missing, f"explicit score manifest lists files that do not exist: {missing}"
        assert len(set(files)) == len(files), f"explicit score manifest repeats a path: {pattern}"
        return files
    return sorted(glob.glob(pattern))


def check_seeds(name: str, files: list[str], strict: bool) -> dict:
    """Parse the seed token of every score file and refuse ambiguous multi-file rosters under --strict-seeds.

    Rule: a multi-file emitter is a *seed ensemble*, so every file must carry a distinct NUMERIC seed token.
    Pseudo-seed tokens (`seedavg`, `seedens`, `seedrank`, `seedlead`) name an aggregate of other files and must
    stand alone; mixing one into a seed roster double-counts the seeds it already contains.
    """
    toks = [(m.group(1) if (m := re.search(r"seed([A-Za-z0-9]+)", Path(f).name)) else "") for f in files]
    numeric = [t for t in toks if t.isdigit()]
    info = {"n_files": len(files), "seed_tokens": toks, "distinct_numeric_seeds": sorted(set(numeric))}
    if strict and len(files) > 1:
        bad = [t for t in toks if not t.isdigit()]
        assert not bad, (f"emitter {name!r}: a multi-file seed roster may only contain numeric seed files, got "
                         f"pseudo-seed token(s) {bad} in {[Path(f).name for f in files]}")
        assert len(set(numeric)) == len(files), (f"emitter {name!r}: {len(files)} files but "
                                                 f"{len(set(numeric))} distinct seeds {sorted(set(numeric))}")
    return info


def load_emitter(spec: str, pairs: dict[str, dict], strict_seeds: bool = False, expect_seeds: int = 0):
    """name=<units file>:<scores glob|comma manifest>  |  name=dumps:<glob>  -> name, {pid: (units, scores)}, provenance."""
    name, rest = spec.split("=", 1)
    if rest.startswith("dumps:"):
        files = resolve_files(rest[6:])
        prov = {"kind": "dumps", "files": files, **check_seeds(name, files, strict_seeds)}
        acc = defaultdict(list); units_of = {}
        for f in files:
            for pid, per in b3.load_dump(Path(f)).items():
                if pid in pairs and per:
                    units_of[pid] = [b3._u(x) for x in per]; acc[pid].append([1 - float(x["support"]) for x in per])
        return name, {p: (units_of[p], np.mean(np.array(v), axis=0)) for p, v in acc.items() if len({len(x) for x in v}) == 1}, prov
    units_file, sglob = rest.rsplit(":", 1)
    inv = {r["pair_id"]: r["units"] for r in common.load_jsonl(Path(units_file)) if r.get("units")}
    files = resolve_files(sglob)
    assert files, f"emitter {name!r}: no score file matched {sglob!r}"
    if expect_seeds and len(files) > 1:
        assert len(files) == expect_seeds, (f"emitter {name!r}: expected {expect_seeds} seed files, matched {len(files)}: "
                                            f"{[Path(f).name for f in files]}")
    prov = {"kind": "scores", "units_file": units_file, "units_sha256": _sha256(Path(units_file)),
            "files": files, "sha256": {Path(f).name: _sha256(Path(f)) for f in files},
            **check_seeds(name, files, strict_seeds)}
    acc = defaultdict(list)
    for f in files:
        _, sc = common.read_scores(Path(f))
        for pid, v in sc.items():
            if pid in inv and pid in pairs and len(v) == len(inv[pid]):
                acc[pid].append(v)
    n_per_pair = {len(v) for v in acc.values()}
    prov["scores_per_pair"] = sorted(n_per_pair)
    if strict_seeds and len(files) > 1:
        assert n_per_pair == {len(files)}, (f"emitter {name!r}: {len(files)} score files but pairs carry "
                                            f"{sorted(n_per_pair)} scores each -- a seed is missing on some pairs")
    return name, {p: (inv[p], np.nanmean(np.vstack(v), axis=0)) for p, v in acc.items()}, prov


def acu_spans(r: dict):
    return [common.best_sentence_span(r["source"], u["text"]) for u in r["units"]]


def sign_test(diffs):
    """Two-sided exact sign test over document-level paired differences (ties dropped)."""
    pos = sum(1 for d in diffs if d > 0); neg = sum(1 for d in diffs if d < 0); n = pos + neg
    if n == 0:
        return None
    lo = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2.0 ** n)
    return {"n_docs": n, "n_pos": pos, "n_neg": neg, "n_tie": len(diffs) - n, "p_value": float(min(1.0, 2.0 * tail))}


def perm_test(diffs, rng, n_perm):
    """Document-level paired permutation (sign-flip) test on the mean difference."""
    d = np.asarray([x for x in diffs], dtype=float)
    if d.size == 0:
        return None
    obs = abs(float(d.mean()))
    flips = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, d.size))
    null = np.abs((flips * d).mean(axis=1))
    return {"n_docs": int(d.size), "observed_mean": float(d.mean()), "n_perm": int(n_perm),
            "p_value": float((int(np.sum(null >= obs - 1e-12)) + 1) / (n_perm + 1))}


def adaptive_budgets(spec, pairs, pids, kmin, kmax):
    """'sent:RATIO' -> {pair_id: k}. The budget scales with the number of SOURCE sentences, so long
    documents are not judged at the same top-10 as short ones (registered T03)."""
    kind, _, val = spec.partition(":")
    assert kind == "sent" and val, f"unsupported adaptive-k spec {spec!r}"
    ratio = float(val)
    out = {}
    for p in pids:
        n_sent = max(1, len(common.sentence_spans(pairs[p]["source"])))
        out[p] = int(min(kmax, max(kmin, round(ratio * n_sent))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acu", default=str(common.OUT_ROOT / "b18" / "acu_units_validation.jsonl"))
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--primary", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-emitted", default="", help="write the emitted units (max k) of every system per pair to this jsonl")
    ap.add_argument("--adaptive-k", default="", help="length-adaptive budget, spec 'sent:RATIO' -> k = clip(round(RATIO * n_source_sentences), --ak-min, --ak-max); may be repeated comma-separated")
    ap.add_argument("--ak-min", type=int, default=3)
    ap.add_argument("--ak-max", type=int, default=50)
    ap.add_argument("--robust-tests", action="store_true", help="add a document-level paired permutation test and a per-document sign test next to the bootstrap CI")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--dump-per-pair", default="", help="write per-pair recall of every system to this jsonl (for offline re-analysis)")
    ap.add_argument("--strict-seeds", action="store_true",
                    help="every multi-file emitter must be a roster of distinct NUMERIC seeds present on every pair "
                         "(refuses a pseudo-seed such as *_seedavg_* being swept into a seed average)")
    ap.add_argument("--expect-seeds", type=int, default=0, help="if set, every multi-file emitter must match exactly this many score files")
    ap.add_argument("--match-min-tokens", type=int, default=1,
                    help="content tokens an emitted unit must share with an ACU to be credited a match (R11 matching "
                         "sensitivity; the registered rule is 1)")
    ap.add_argument("--match-min-jaccard", type=float, default=0.0,
                    help="additionally require this content-token Jaccard between the emitted unit and the ACU")
    ap.add_argument("--match-symmetric", action="store_true",
                    help="A4: build the EMITTED-UNIT token set content-only as well (same `or all-tokens` fallback the "
                         "ACU side already uses), making the shared-token/Jaccard filter symmetric. Off by default so "
                         "the registered rule is reproduced exactly.")
    a = ap.parse_args()
    pairs = {r["pair_id"]: r for r in common.load_jsonl(Path(a.acu))}
    emitters = {}; provenance = {}
    for spec in a.systems:
        name, em, prov = load_emitter(spec, pairs, strict_seeds=a.strict_seeds, expect_seeds=a.expect_seeds)
        if em:
            emitters[name] = em
        provenance[name] = {**prov, "n_pairs": len(em)}
        print(f"[e2] {name}: {len(em)} pairs from {prov['n_files']} score file(s) "
              f"seeds={prov.get('distinct_numeric_seeds') or prov.get('seed_tokens')}", flush=True)
        for f in prov["files"]:
            print(f"[e2:manifest] {name} <- {f}", flush=True)
    common_p = sorted(p for p in pairs if all(p in e for e in emitters.values()))
    spans = {p: acu_spans(pairs[p]) for p in common_p}
    ctoks = {p: [set(common.tokens(u["text"], content_only=True)) or set(common.tokens(u["text"])) for u in pairs[p]["units"]] for p in common_p}
    stress = {p: [common.counters_for("acu", pairs[p], i)["ca_unit_in_cand"] == 0.0 and common.counters_for("acu", pairs[p], i)["ca_token_recall"] < 0.5 for i in range(len(pairs[p]["units"]))] for p in common_p}
    doc_of = {p: common.doc_key(pairs[p]["source"]) for p in common_p}; docs = sorted(set(doc_of.values()))

    def unit_toks(text: str) -> set:
        """A4: the emitted-unit side of the filter. Registered rule = all tokens; --match-symmetric = content-only
        with the same all-token fallback the ACU side uses, so an all-stopword unit is not silently emptied."""
        if a.match_symmetric:
            return set(common.tokens(text, content_only=True)) or set(common.tokens(text))
        return set(common.tokens(text))

    def match_ok(acu_toks: set, unit_toks: set) -> bool:
        """The registered rule is >= 1 shared content token; --match-min-tokens/--match-min-jaccard tighten it."""
        sh = acu_toks & unit_toks
        if len(sh) < a.match_min_tokens:
            return False
        if a.match_min_jaccard > 0.0:
            un = acu_toks | unit_toks
            if not un or len(sh) / len(un) < a.match_min_jaccard:
                return False
        return True

    def per_pair(name, k, keep=False):
        out = {}
        for p in common_p:
            units, sc = emitters[name][p]; r = pairs[p]; truth = common.truth_of(r)
            order = sorted(range(len(units)), key=lambda j: (-sc[j] if np.isfinite(sc[j]) else 1.0, units[j].get("start", 0)))[:(k if isinstance(k, int) else int(k.get(p, 10)))]
            matched = set(); status = []
            for j in order:
                u = units[j]; us, ue = u.get("start", -1), u.get("end", -1); ut = unit_toks(u["text"])
                hits = [i for i, (s, e) in enumerate(spans[p]) if us < e and ue > s and match_ok(ctoks[p][i], ut)]
                matched.update(hits)
                if not hits: status.append("unknown")
                elif any(truth[i] == 1 for i in hits): status.append("hit")
                else: status.append("false_alert")
            om = [i for i, t in enumerate(truth) if t == 1]; om_s = [i for i in om if stress[p][i]]
            emitted = [{"j": int(j), "text": units[j]["text"], "start": units[j].get("start"), "end": units[j].get("end"), "status": st, "score": (float(sc[j]) if np.isfinite(sc[j]) else None)} for j, st in zip(order, status)] if keep else None
            out[p] = {"emitted": emitted, "recall": (len([i for i in om if i in matched]) / len(om)) if om else None,
                      "recall_stress": (len([i for i in om_s if i in matched]) / len(om_s)) if om_s else None,
                      "hit": status.count("hit") / max(1, len(status)), "false_alert": status.count("false_alert") / max(1, len(status)),
                      "unknown": status.count("unknown") / max(1, len(status)), "n_emitted": len(status)}
        return out

    def summarise(pp, pids):
        vals = [pp[p] for p in pids]
        m = lambda key: float(np.mean([v[key] for v in vals if v[key] is not None])) if any(v[key] is not None for v in vals) else None
        # Omission precision: of the emitted units that align to SOME reference ACU, the share aligning to one the
        # annotators marked omitted. Recall@k alone is maximised by never demoting a fact the summary conveys --
        # a summary-blind emitter matches more ACUs and is rewarded for it -- so this is the endpoint on which
        # conditioning on the actual summary can pay (R03).
        h = sum(v["hit"] * v["n_emitted"] for v in vals); f = sum(v["false_alert"] * v["n_emitted"] for v in vals)
        return {"recall": m("recall"), "recall_stress": m("recall_stress"), "hit_rate": m("hit"),
                "false_alert_rate": m("false_alert"), "unknown_rate": m("unknown"),
                "acu_matched_rate": m("hit") + m("false_alert") if m("hit") is not None and m("false_alert") is not None else None,
                "omission_precision": (h / (h + f)) if (h + f) > 0 else None, "n_pairs": len(vals)}

    rep = {"registered": "research/marsc_strengthen_20260918/PREREG.md Wave A / A4 (fork of marsc_20260916 E2 pilot)", "n_pairs": len(common_p), "n_docs": len(docs),
           "strict_seeds": bool(a.strict_seeds), "expect_seeds": int(a.expect_seeds),
           "match_min_tokens": int(a.match_min_tokens), "match_min_jaccard": float(a.match_min_jaccard),
           "match_symmetric": bool(a.match_symmetric),
           "emitter_manifest": provenance, "k": {}}
    rng = np.random.default_rng(20260916); by_doc = defaultdict(list)
    for p in common_p:
        by_doc[doc_of[p]].append(p)
    kspecs = [(str(k), k) for k in a.k]
    for spec in [x for x in a.adaptive_k.split(",") if x.strip()]:
        kmap = adaptive_budgets(spec.strip(), pairs, common_p, a.ak_min, a.ak_max)
        label = f"adaptive:{spec.strip()}"
        kspecs.append((label, kmap))
        print(f"[e2] {label}: k median {int(np.median(list(kmap.values())))} min {min(kmap.values())} max {max(kmap.values())}", flush=True)
    per_pair_rows = []
    for klabel, k in kspecs:
        keep = bool(a.dump_emitted) and klabel == str(max(a.k))
        pp = {n: per_pair(n, k, keep) for n in emitters}
        if keep:
            Path(a.dump_emitted).parent.mkdir(parents=True, exist_ok=True)
            with open(a.dump_emitted, "w", encoding="utf-8") as fh:
                for n in emitters:
                    for p in common_p:
                        fh.write(json.dumps({"system": n, "pair_id": p, "resource": pairs[p]["resource"], "k": klabel, "emitted": pp[n][p]["emitted"]}) + "\n")
            print(f"[e2] emitted units written to {a.dump_emitted}", flush=True)
        rk = {"systems": {n: summarise(pp[n], common_p) for n in emitters}, "paired_recall": {}}
        prim = a.primary if a.primary in emitters else max(emitters, key=lambda n: rk["systems"][n]["recall"] or 0)
        rk["primary"] = prim
        for n in emitters:
            if n == prim: continue
            d = []; dp = []
            for _ in range(a.n_boot):
                pick = rng.choice(docs, size=len(docs), replace=True); sub = [p for dd in pick for p in by_doc[dd]]
                sp, so = summarise(pp[prim], sub), summarise(pp[n], sub)
                if sp["recall"] is not None and so["recall"] is not None: d.append(sp["recall"] - so["recall"])
                if sp["omission_precision"] is not None and so["omission_precision"] is not None:
                    dp.append(sp["omission_precision"] - so["omission_precision"])
            rk["paired_recall"][n] = {"mean": float(np.mean(d)), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]} if d else None
            rk.setdefault("paired_omission_precision", {})[n] = {"mean": float(np.mean(dp)), "ci": [float(np.percentile(dp, 2.5)), float(np.percentile(dp, 97.5))]} if dp else None
            if a.robust_tests:
                dd = []
                for dk in docs:
                    ps = [p for p in by_doc[dk] if pp[prim][p]["recall"] is not None and pp[n][p]["recall"] is not None]
                    if ps:
                        dd.append(float(np.mean([pp[prim][p]["recall"] for p in ps]) - np.mean([pp[n][p]["recall"] for p in ps])))
                rk.setdefault("sign_test", {})[n] = sign_test(dd)
                rk.setdefault("perm_test", {})[n] = perm_test(dd, np.random.default_rng(20260917), a.n_perm)
        if a.dump_per_pair:
            for n in emitters:
                for p in common_p:
                    per_pair_rows.append({"k": klabel, "system": n, "pair_id": p, "resource": pairs[p]["resource"], "doc": doc_of[p],
                                          "recall": pp[n][p]["recall"], "recall_stress": pp[n][p]["recall_stress"],
                                          "hit": pp[n][p]["hit"], "false_alert": pp[n][p]["false_alert"], "unknown": pp[n][p]["unknown"], "n_emitted": pp[n][p]["n_emitted"]})
        resources = sorted({pairs[p]["resource"] for p in common_p})
        rk["by_resource"] = {n: {res: summarise(pp[n], [p for p in common_p if pairs[p]["resource"] == res]) for res in resources} for n in emitters}
        rep["k"][klabel] = rk
        for n in sorted(emitters, key=lambda n: -(rk["systems"][n]["recall"] or 0)):
            print(f"[e2 k={klabel} by-resource] {n:34s} " + " ".join(f"{res} {rk['by_resource'][n][res]['recall']:.3f} (n={rk['by_resource'][n][res]['n_pairs']})" for res in resources), flush=True)
        for n, s in sorted(rk["systems"].items(), key=lambda kv: -(kv[1]["recall"] or 0)):
            pr = rk["paired_recall"].get(n)
            if a.robust_tests and rk.get("sign_test", {}).get(n):
                st, pt = rk["sign_test"][n], rk["perm_test"][n]
                print(f"[e2 robust k={klabel}] {n:34s} sign {st['n_pos']}+/{st['n_neg']}- p={st['p_value']:.4f} | perm mean {pt['observed_mean']:+.4f} p={pt['p_value']:.4f}", flush=True)
            op = rk.get("paired_omission_precision", {}).get(n)
            print(f"[e2 k={klabel}] {n:34s} recall {s['recall']:.3f} omission-prec {s['omission_precision'] if s['omission_precision'] is None else round(s['omission_precision'], 3)} acu-matched {s['acu_matched_rate'] if s['acu_matched_rate'] is None else round(s['acu_matched_rate'], 3)} hit {s['hit_rate']:.3f} false-alert {s['false_alert_rate']:.3f} unknown {s['unknown_rate']:.3f}" + (f" | rec {pr['mean']:+.3f} [{pr['ci'][0]:+.3f},{pr['ci'][1]:+.3f}]" if pr else " | primary") + (f" prec {op['mean']:+.3f} [{op['ci'][0]:+.3f},{op['ci'][1]:+.3f}]" if op else ""), flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / "e2_emitted.json").write_text(json.dumps(rep, indent=1))
    if a.dump_per_pair:
        Path(a.dump_per_pair).parent.mkdir(parents=True, exist_ok=True)
        with open(a.dump_per_pair, "w", encoding="utf-8") as fh:
            for row in per_pair_rows:
                fh.write(json.dumps(row) + "\n")
        print(f"[e2] per-pair recalls written to {a.dump_per_pair} ({len(per_pair_rows)} rows)", flush=True)
    print("[e2] written", out / "e2_emitted.json", flush=True)


if __name__ == "__main__":
    main()
