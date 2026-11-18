#!/usr/bin/env python3
"""MARS-C G7 -- external confirmation on UniSumEval (PLAN_MAIN_CONTRIBUTION.md, block G7; untouched pool found by G5).

UniSumEval (DISL-Lab) supplies, for 225 documents in nine domains and nine summarizers, human-validated key facts of the
source and a per-key-fact coverage label for every summary. That is exactly the reference universe of the emitted-top-k
protocol. Stages:
  build      -> acu_units_unisum.jsonl (rows: pair_id = uid, units = key facts, unit_labels 1 = covered) and the sentence
                decomposition tasks in the M32 format (documents overlapping our source pool are excluded)
  inventory  -> units_gemma+ent_unisum.jsonl from the Gemma decomposition shards + spaCy entity units (the frozen inventory)
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


def build(a) -> None:
    ours = set()
    for f in a.our_sources:
        for r in common.load_jsonl(Path(f)):
            if r.get("source"):
                ours.add(hashlib.sha1(r["source"][:200].encode()).hexdigest())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n = {"rows": 0, "skipped_no_labels": 0, "excluded_overlap": 0}; docs = {}
    with open(out / "acu_units_unisum.jsonl", "w", encoding="utf-8") as fh:
        for r in common.load_jsonl(Path(a.keyfact)):
            kf, lab = r.get("keyfact") or [], r.get("keyfact_label")
            if not kf or not lab or len(kf) != len(lab) or not (r.get("summary") or "").strip():
                n["skipped_no_labels"] += 1; continue
            src = r["input_context"]
            if hashlib.sha1(src[:200].encode()).hexdigest() in ours:
                n["excluded_overlap"] += 1; continue
            k = hashlib.sha1(src.encode()).hexdigest()[:12]; docs[k] = src
            fh.write(json.dumps({"pair_id": f"{r['uid']}:{r['model']}", "resource": f"unisum_{r['source']}", "system": r["model"], "split": "unisum", "domain": r["domain"], "length": r["length"],
                                 "input_type": r["input_type"], "doc_id": r["doc_id"], "source": src, "candidate": r["summary"],
                                 "units": [{"kind": "acu", "text": t} for t in kf], "unit_labels": [1 if bool(x) else 0 for x in lab]}) + "\n"); n["rows"] += 1
    m = 0
    with open(out / "decompose_tasks_unisum.jsonl", "w", encoding="utf-8") as fh:
        for k, src in docs.items():
            for i, (s, e) in enumerate(common.sentence_spans(src)):
                sent = src[s:e].strip()
                if len(sent.split()) < 4:
                    continue
                fh.write(json.dumps({"task_id": f"{k}:{i}", "doc_key": k, "splits": ["unisum"], "resources": ["unisum"], "start": s, "end": e, "sentence": sent}) + "\n"); m += 1
    n["documents"] = len(docs); n["sentence_tasks"] = m
    pids = [json.loads(l)["pair_id"] for l in open(out / "acu_units_unisum.jsonl", encoding="utf-8")]
    n["unique_pair_ids"] = len(set(pids))
    if len(set(pids)) != len(pids):
        raise SystemExit(f"[g7-build] ABORT: pair_id not unique ({len(set(pids))} of {len(pids)})")
    json.dump(n, open(out / "build_manifest.json", "w"), indent=1); print(f"[g7-build] {json.dumps(n)}", flush=True)


def inventory(a) -> None:
    from mars_v2.units import get_extractor
    extractor = get_extractor("entity")
    props = defaultdict(list)
    for f in a.props:
        for r in common.load_jsonl(Path(f)):
            for p in r.get("props", []):
                props[r["doc_key"]].append({"kind": "prop_llm", "text": p, "start": r["start"], "end": r["end"], "label": "", "args": {}})
    cache = {}; n_units = []; out = Path(a.out)
    with open(out / "units_gemma+ent_unisum.jsonl", "w", encoding="utf-8") as fh:
        for r in common.load_jsonl(Path(a.acu)):
            k = hashlib.sha1(r["source"].encode()).hexdigest()[:12]
            if k not in cache:
                cache[k] = list(props.get(k, [])) + [u.to_dict() for u in extractor(r["source"])]
            units = cache[k]; n_units.append(len(units))
            fh.write(json.dumps({"pair_id": r["pair_id"], "resource": r["resource"], "system": r["system"], "split": "unisum", "source": r["source"], "candidate": r["candidate"], "units": units}) + "\n")
    docs_with_props = sum(1 for k in cache if props.get(k)); print(f"[g7-inventory] {len(n_units)} pairs, {len(cache)} documents ({docs_with_props} with LLM propositions), {sum(n_units) / max(1, len(n_units)):.1f} units/pair", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--keyfact", required=True); b.add_argument("--our-sources", nargs="+", required=True); b.add_argument("--out", required=True)
    i = sub.add_parser("inventory"); i.add_argument("--acu", required=True); i.add_argument("--props", nargs="+", required=True); i.add_argument("--out", required=True)
    a = ap.parse_args(); {"build": build, "inventory": inventory}[a.cmd](a)


if __name__ == "__main__":
    main()
