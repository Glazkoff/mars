#!/usr/bin/env python3
"""M-A.1 -- candidates and references of the three meta-evaluation sets as sentence-decomposition tasks (M32
format) plus the pair index the scorers read.

Tasks (doc_key -> text):
  unisum:cand:<uid>:<model>      UniSumEval summary, its own `sentences` list (the unit of its faithfulness labels)
  unisum:ref:<doc_id>            UniSumEval reference (one per document)
  summeval:cand:<id>:<m>         SummEval machine summary m (0..15)
  summeval:ref:<id>:<k>          SummEval human reference k (0..10)
  rose:cand:<pair_id>            RoSE test candidate summary (its references are the human ACUs, not decomposed)
The pair index (pairs_ma.jsonl) carries, per candidate: dataset, doc_id, system, source, candidate, reference(s),
the human dimensions, and the doc_keys of its candidate and reference decompositions. Registered under M-A.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402


def sent_tasks(fh, key: str, text: str, presplit: list[str] | None = None) -> int:
    n = 0
    if presplit is not None:
        pos = 0
        for j, s in enumerate(presplit):
            s = str(s).strip()
            if not s:
                continue
            st = text.find(s, pos); st = st if st >= 0 else pos; en = st + len(s); pos = en
            fh.write(json.dumps({"task_id": f"{key}:{j}", "doc_key": key, "splits": ["ma"], "resources": [key.split(":")[0]],
                                 "start": st, "end": en, "sentence": s}) + "\n"); n += 1
        return n
    for j, (s, e) in enumerate(common.sentence_spans(text)):
        sent = text[s:e].strip()
        if not sent:
            continue
        fh.write(json.dumps({"task_id": f"{key}:{j}", "doc_key": key, "splits": ["ma"], "resources": [key.split(":")[0]],
                             "start": s, "end": e, "sentence": sent}) + "\n"); n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unisum-dir", required=True)
    ap.add_argument("--summeval", required=True)
    ap.add_argument("--rose-acu", required=True)
    ap.add_argument("--rose-agg", nargs="*", default=[], help="RoSE *.acus.aggregated.jsonl files: reference summaries by source")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n = {"tasks": 0, "pairs": 0}
    seen_keys = set()
    with open(out / "tasks_ma.jsonl", "w", encoding="utf-8") as ft, open(out / "pairs_ma.jsonl", "w", encoding="utf-8") as fp:
        # ---- UniSumEval
        kf = [json.loads(l) for l in open(Path(a.unisum_dir) / "unisumeval_keyfact.jsonl", encoding="utf-8")]
        ff = {(r["uid"], r["model"]): r for r in (json.loads(l) for l in open(Path(a.unisum_dir) / "unisumeval_faithfulness.jsonl", encoding="utf-8"))}
        for r in kf:
            f = ff.get((r["uid"], r["model"]))
            if f is None or str(r.get("summary_success_state", "success")) != "success":
                continue
            ck = f"unisum:cand:{r['uid']}:{r['model']}"; rk = f"unisum:ref:{r['doc_id']}"
            sents = json.loads(r["sentences"]) if isinstance(r["sentences"], str) else list(r["sentences"])
            n["tasks"] += sent_tasks(ft, ck, r["summary"], presplit=sents)
            has_ref = bool(r.get("reference"))
            if not has_ref:
                n["unisum_without_reference"] = n.get("unisum_without_reference", 0) + 1
            elif rk not in seen_keys:
                seen_keys.add(rk); n["tasks"] += sent_tasks(ft, rk, r["reference"])
            geval_c = eval(r["machine_evaluation_results_completeness"]) if isinstance(r.get("machine_evaluation_results_completeness"), str) else (r.get("machine_evaluation_results_completeness") or {})
            geval_f = eval(f["machine_evaluation_results_faithfulness"]) if isinstance(f.get("machine_evaluation_results_faithfulness"), str) else (f.get("machine_evaluation_results_faithfulness") or {})
            fp.write(json.dumps({"pair_id": f"unisum:{r['uid']}:{r['model']}", "dataset": "unisumeval", "doc_id": r["doc_id"], "domain": r["domain"],
                                 "system": r["model"], "source": r["input_context"], "candidate": r["summary"], "references": [r["reference"]] if has_ref else [],
                                 "cand_key": ck, "ref_keys": [rk] if has_ref else [],
                                 "human": {"completeness": float(r["completeness"]), "conciseness": float(r["conciseness"]),
                                           "faithfulness": float(f["faithfulness_score"])},
                                 "machine": {"geval_completeness": geval_c.get("G-Eval"), "gevalplus_completeness": geval_c.get("G-Eval+"),
                                             "geval_faithfulness": geval_f.get("G-Eval"), "gevalplus_faithfulness": geval_f.get("G-Eval+")}}) + "\n"); n["pairs"] += 1
        # ---- SummEval
        df = pd.read_parquet(a.summeval)
        for _, row in df.iterrows():
            did = row["id"]; refs = [str(x) for x in list(row["human_summaries"])]; mach = [str(x) for x in list(row["machine_summaries"])]
            rkeys = []
            for k, ref in enumerate(refs):
                rk = f"summeval:ref:{did}:{k}"; rkeys.append(rk); n["tasks"] += sent_tasks(ft, rk, ref)
            for m, cand in enumerate(mach):
                ck = f"summeval:cand:{did}:{m}"; n["tasks"] += sent_tasks(ft, ck, cand)
                fp.write(json.dumps({"pair_id": f"summeval:{did}:{m}", "dataset": "summeval", "doc_id": did, "domain": "news", "system": f"M{m}",
                                     "source": row["text"], "candidate": cand, "references": refs, "cand_key": ck, "ref_keys": rkeys,
                                     "human": {d: float(list(row[d])[m]) for d in ("relevance", "consistency", "coherence", "fluency")}}) + "\n"); n["pairs"] += 1
        # ---- RoSE test (facts = human ACUs; the reference summary, when the aggregated release is given, is the
        # reference text for ROUGE / BERTScore / BARTScore)
        ref_by_source: dict[str, str] = {}
        for f in a.rose_agg:
            for line in open(f, encoding="utf-8"):
                rr = json.loads(line)
                if rr.get("reference"):
                    ref_by_source[" ".join(str(rr["source"]).split())] = str(rr["reference"])
        n["rose_with_reference"] = 0
        for r in (json.loads(l) for l in open(a.rose_acu, encoding="utf-8")):
            ref = ref_by_source.get(" ".join(str(r["source"]).split()))
            n["rose_with_reference"] += int(ref is not None)
            lab = [x for x in r["unit_labels"] if x is not None]
            ck = f"rose:cand:{r['pair_id']}"; n["tasks"] += sent_tasks(ft, ck, r["candidate"])
            fp.write(json.dumps({"pair_id": f"rose:{r['pair_id']}", "dataset": "rose", "doc_id": hashlib.sha1(r["source"].encode()).hexdigest()[:12],
                                 "domain": r.get("resource", ""), "system": r["system"], "source": r["source"], "candidate": r["candidate"],
                                 "references": [ref] if ref else [], "cand_key": ck, "ref_keys": [], "acus": [u["text"] for u in r["units"]], "acu_labels": r["unit_labels"],
                                 "human": {"acu_recall": (sum(lab) / len(lab)) if lab else None}}) + "\n"); n["pairs"] += 1
    (out / "tasks_ma_manifest.json").write_text(json.dumps(n, indent=1))
    print(f"[ma-tasks] {json.dumps(n)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
