#!/usr/bin/env python3
"""M-A.4 (gap fill, 2026-09-27) -- the Gemma+ent SOURCE inventory for the 100 SummEval documents, so the
reference-free MARS-R can be read on SummEval as it already is on UniSumEval and RoSE. The rules are those of
mc_unisum.py (build + inventory): doc_key = sha1(source)[:12]; one M32 task per sentence of at least four words
(common.sentence_spans); Gemma propositions plus spaCy entity units. Pair ids are "<doc_id>:<system>" so that
ma_reffree.py --set summeval=... matches pairs_ma.jsonl ("summeval:<doc_id>:<system>").

  tasks      --pairs pairs_ma.jsonl --out <dir>   -> decompose_tasks_summeval.jsonl, pairs_summeval.jsonl, overlap.json
  inventory  --pairs <dir>/pairs_summeval.jsonl --props <shards> --out <dir>/units_gemma+ent_summeval.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code")); sys.path.insert(0, str(REPO))
import common  # noqa: E402


def tasks(a) -> None:
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    docs, n = {}, 0
    train_keys = set()
    for f in a.train_sources:
        for r in common.load_jsonl(Path(f)):
            if r.get("source"):
                train_keys.add(hashlib.sha1(r["source"][:200].encode()).hexdigest())
    overlap = set()
    with open(out / "pairs_summeval.jsonl", "w", encoding="utf-8") as fh:
        for r in common.load_jsonl(Path(a.pairs)):
            if r["dataset"] != "summeval":
                continue
            src = r["source"]; k = hashlib.sha1(src.encode()).hexdigest()[:12]; docs[k] = src
            if hashlib.sha1(src[:200].encode()).hexdigest() in train_keys:
                overlap.add(r["doc_id"])
            key = r["pair_id"].split(":", 1)[1]
            fh.write(json.dumps({"pair_id": key, "resource": "summeval", "system": r["system"], "split": "summeval",
                                 "source": src, "candidate": r["candidate"]}) + "\n"); n += 1
    m = 0
    with open(out / "decompose_tasks_summeval.jsonl", "w", encoding="utf-8") as fh:
        for k, src in docs.items():
            for i, (s, e) in enumerate(common.sentence_spans(src)):
                sent = src[s:e].strip()
                if len(sent.split()) < 4:
                    continue
                fh.write(json.dumps({"task_id": f"{k}:{i}", "doc_key": k, "splits": ["summeval"], "resources": ["summeval"],
                                     "start": s, "end": e, "sentence": sent}) + "\n"); m += 1
    rep = {"pairs": n, "documents": len(docs), "sentence_tasks": m,
           "documents_overlapping_marsc_training_sources": sorted(overlap)}
    json.dump(rep, open(out / "build_manifest.json", "w"), indent=1)
    print(f"[ma4] {json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in rep.items()})}", flush=True)


def inventory(a) -> None:
    from mars_v2.units import get_extractor
    extractor = get_extractor("entity")
    props = defaultdict(list)
    for f in a.props:
        for r in common.load_jsonl(Path(f)):
            for p in r.get("props", []):
                props[r["doc_key"]].append({"kind": "prop_llm", "text": p, "start": r["start"], "end": r["end"], "label": "", "args": {}})
    cache, n_units = {}, []
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in common.load_jsonl(Path(a.pairs)):
            k = hashlib.sha1(r["source"].encode()).hexdigest()[:12]
            if k not in cache:
                cache[k] = list(props.get(k, [])) + [u.to_dict() for u in extractor(r["source"])]
            units = cache[k]; n_units.append(len(units))
            fh.write(json.dumps({**r, "units": units}) + "\n")
    with_props = sum(1 for k in cache if props.get(k))
    print(f"[ma4] {len(n_units)} pairs, {len(cache)} documents ({with_props} with propositions), "
          f"{sum(n_units) / max(1, len(n_units)):.1f} units/pair -> {a.out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tasks"); t.add_argument("--pairs", required=True); t.add_argument("--out", required=True)
    t.add_argument("--train-sources", nargs="*", default=[])
    i = sub.add_parser("inventory"); i.add_argument("--pairs", required=True); i.add_argument("--props", nargs="+", required=True)
    i.add_argument("--out", required=True)
    a = ap.parse_args(); {"tasks": tasks, "inventory": inventory}[a.cmd](a)


if __name__ == "__main__":
    main()
