#!/usr/bin/env python3
"""B4 -- the pre-registered ordering-reproduction estimator for the unseen third extractor.

The bar, declared before execution: cross-extractor transfer is evidence for deployment-distribution learning
only if the MARS-C ordering on the DEVELOPMENT extractors reproduces on the UNSEEN one, with a
document-clustered interval excluding zero.

This reads the per-pair dump of mc_e2_eval.py (`--dump-per-pair`), which carries, for every (system, pair):
recall@k, the emitted-unit hit / false-alert shares and the number of emitted units, plus the pair's document
key. From those the two registered endpoints are reconstructed exactly as mc_e2_eval.summarise defines them:

    recall              mean over pairs that have at least one human-omitted ACU
    omission_precision  sum(hit * n_emitted) / (sum(hit * n_emitted) + sum(false_alert * n_emitted))

and any named contrast between two systems, or difference of two contrasts, is resampled by DOCUMENT with
replacement -- the same clustering mc_e2_eval uses and the clustering `mc_crossed_precision.py` was found to
be missing.

Two reporting rules are enforced here because both have already cost this project a retracted number:
  * `observed` is the contrast computed once on the real data. `bootstrap_mean` is the mean of the resampled
    contrasts. They are printed under different names and the observed value is the one that is reported.
  * a contrast is only called reproduced when the 95 % percentile interval excludes zero AND the observed sign
    matches the declared direction.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: Path, klabel: str):
    rows = defaultdict(dict)          # system -> pair_id -> row
    doc_of: dict[str, str] = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if str(r.get("k")) != klabel:
            continue
        rows[r["system"]][r["pair_id"]] = r
        doc_of[r["pair_id"]] = r["doc"]
    return rows, doc_of


def metric(rs: dict, pids, name: str):
    if name == "recall":
        v = [rs[p]["recall"] for p in pids if rs[p].get("recall") is not None]
        return float(np.mean(v)) if v else None
    if name == "omission_precision":
        h = sum(rs[p]["hit"] * rs[p]["n_emitted"] for p in pids)
        f = sum(rs[p]["false_alert"] * rs[p]["n_emitted"] for p in pids)
        return (h / (h + f)) if (h + f) > 0 else None
    raise SystemExit(f"unknown metric {name!r}")


def boot(rows, doc_of, terms, metric_name, n_boot, seed):
    """terms = [(+1, sysA), (-1, sysB), ...]; resample documents, recompute every term on the same resample."""
    sysnames = [s for _, s in terms]
    common = sorted(set.intersection(*[set(rows[s]) for s in sysnames]))
    by_doc = defaultdict(list)
    for p in common:
        by_doc[doc_of[p]].append(p)
    docs = sorted(by_doc)

    def value(pids):
        out = 0.0
        for sign, s in terms:
            m = metric(rows[s], pids, metric_name)
            if m is None:
                return None
            out += sign * m
        return out

    observed = value(common)
    rng = np.random.default_rng(seed); draws = []
    for _ in range(n_boot):
        pick = rng.choice(docs, size=len(docs), replace=True)
        sub = [p for d in pick for p in by_doc[d]]
        v = value(sub)
        if v is not None:
            draws.append(v)
    d = np.asarray(draws, dtype=float)
    return {"metric": metric_name, "observed": observed,
            "bootstrap_mean": float(d.mean()) if d.size else None,
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if d.size else None,
            "one_sided_lower_95": float(np.percentile(d, 5.0)) if d.size else None,
            "n_boot": int(d.size), "n_docs": len(docs), "n_pairs": len(common),
            "terms": [{"sign": int(s), "system": n, "value": metric(rows[n], common, metric_name)} for s, n in terms]}


def parse_spec(spec: str):
    parts = [x.strip() for x in spec.split("|")]
    if len(parts) == 3:
        label, a, b = parts
        return label, [(+1, a), (-1, b)]
    if len(parts) == 5:
        label, a, b, c, d = parts
        return label, [(+1, a), (-1, b), (-1, c), (+1, d)]
    raise SystemExit(f"bad spec {spec!r}: use 'LABEL|SYS_A|SYS_B' or 'LABEL|SYS_A|SYS_B|SYS_C|SYS_D' for (A-B)-(C-D)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pair", required=True, help="mc_e2_eval.py --dump-per-pair output")
    ap.add_argument("--k", default="10")
    ap.add_argument("--contrast", nargs="*", default=[], help="'LABEL|SYS_A|SYS_B' (A minus B)")
    ap.add_argument("--did", nargs="*", default=[], help="'LABEL|A|B|C|D' for (A-B)-(C-D)")
    ap.add_argument("--metrics", nargs="+", default=["recall", "omission_precision"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--direction", default="positive", choices=["positive", "negative", "either"],
                    help="declared sign of a contrast for the reproduction verdict")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows, doc_of = load(Path(a.per_pair), a.k)
    print(f"[b4-order] {len(rows)} systems at k={a.k}: {sorted(rows)}", flush=True)
    rep = {"k": a.k, "n_boot": a.n_boot, "seed": a.seed, "direction": a.direction,
           "reporting_rule": "`observed` is the contrast on the real data; `bootstrap_mean` is the mean of the "
                             "document-clustered resamples and is NOT the observed contrast",
           "levels": {}, "contrasts": {}, "did": {}}

    allp = sorted(doc_of)
    for s in sorted(rows):
        pids = sorted(rows[s])
        rep["levels"][s] = {m: metric(rows[s], pids, m) for m in a.metrics} | {"n_pairs": len(pids)}
        print(f"[b4-order level] {s:42s} " + "  ".join(
            f"{m} {rep['levels'][s][m]:.4f}" if rep["levels"][s][m] is not None else f"{m} None" for m in a.metrics), flush=True)

    for kind, specs in (("contrasts", a.contrast), ("did", a.did)):
        for spec in specs:
            label, terms = parse_spec(spec)
            missing = [n for _, n in terms if n not in rows]
            if missing:
                raise SystemExit(f"[b4-order] ABORT: contrast {label!r} names systems not in the dump: {missing}")
            rep[kind][label] = {}
            for m in a.metrics:
                r = boot(rows, doc_of, terms, m, a.n_boot, a.seed)
                lo, hi = r["ci95"]
                excl = (lo > 0) or (hi < 0)
                ok = excl and ((a.direction == "positive" and r["observed"] > 0)
                               or (a.direction == "negative" and r["observed"] < 0)
                               or (a.direction == "either"))
                r["ci95_excludes_zero"] = bool(excl); r["reproduces_declared_direction"] = bool(ok)
                rep[kind][label][m] = r
                print(f"[b4-order {kind}] {label:38s} {m:20s} observed {r['observed']:+.4f} "
                      f"ci95 [{lo:+.4f},{hi:+.4f}] lower95 {r['one_sided_lower_95']:+.4f} "
                      f"boot-mean {r['bootstrap_mean']:+.4f} docs {r['n_docs']} pairs {r['n_pairs']} "
                      f"-> {'EXCLUDES ZERO' if excl else 'spans zero'}{'' if ok else '  [direction not met]'}", flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(f"[b4-order] written {out}", flush=True)


if __name__ == "__main__":
    main()
