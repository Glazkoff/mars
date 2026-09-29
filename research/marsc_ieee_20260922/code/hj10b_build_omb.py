#!/usr/bin/env python3
"""H-J10b -- OmissionBench end to end: the emitted-top-k protocol on a clinical dialogue-to-note pool.

Stages (mirroring mc_unisum.py, the UniSumEval builder, so the frozen inventory recipe and evaluator run unchanged):
  build      -> acu_units_omb.jsonl: one row per NOTE (every evaluation omission pair's errored note, and each
                consultation's clean twin), source = transcript, candidate = note, units = the consultation's
                reference facts (the benchmark's must_contain fact sheet, plus every statement removed in one of
                its omission pairs), unit_labels 1 = the note states the fact, 0 = the note is the one it was
                removed from (its near-duplicates in the sheet, content-token F1 >= 0.6, are labelled 0 with it);
                the clean twin carries all 1s. Plus the sentence-decomposition tasks in the M32 format.
  inventory  -> units_gemma+ent_omb.jsonl from the Gemma decomposition shards + spaCy entity units.
A census is written by `build` before any scorer runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
import sys  # noqa: E402

sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code")); sys.path.insert(0, str(REPO))
import common  # noqa: E402

DUP_F1 = 0.6


def fact_text(f) -> str:
    return f if isinstance(f, str) else f["fact"]


def f1(a: str, b: str) -> float:
    ta, tb = set(common.tokens(a, content_only=True)), set(common.tokens(b, content_only=True))
    if not ta or not tb:
        return 0.0
    i = len(ta & tb)
    return 0.0 if i == 0 else 2 * i / (len(ta) + len(tb))


def load_sheets(d: Path) -> dict:
    sheets = {}
    for s in ("primock", "aci", "trapblind"):
        for r in json.load(open(d / f"{s}.json", encoding="utf-8"))["records"]:
            sheets[(s, r["id"])] = [fact_text(f) for f in r["fact_sheet"]["must_contain"]]
    for sc in json.load(open(d / "authored.json", encoding="utf-8"))["scenarios"]:
        sheets[("authored", sc["id"])] = [fact_text(f) for f in sc["fact_sheet"]["must_contain"]]
    return sheets


def build(a) -> None:
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pairs = [p for p in json.load(open(a.pairs, encoding="utf-8"))["pairs"] if p["eval_set"] and p["type"] == "omit"]
    sheets = load_sheets(Path(a.sheets))
    by_key: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_key[p["key"]].append(p)
    n = Counter(); rows = []; docs = {}; acus_per_row = []
    for key in sorted(by_key):
        ps = sorted(by_key[key], key=lambda p: p["pair_id"])
        stratum, cid = ps[0]["stratum"], ps[0]["id"]
        src = (Path(a.transcripts) / stratum / f"{cid}.txt").read_text(encoding="utf-8")
        facts = list(dict.fromkeys(sheets[(stratum, cid)]))          # the sheet, de-duplicated by text
        removed_idx: dict[str, list[int]] = {}                       # pair_id -> indices labelled 0 in its errored note
        for p in ps:
            dup = [i for i, f in enumerate(facts) if f1(f, p["fact"]) >= DUP_F1 or f == p["fact"]]
            if p["fact"] in facts:
                n["removed_fact_in_sheet_verbatim"] += 1
            elif dup:
                n["removed_fact_near_duplicate_in_sheet"] += 1
            else:
                n["removed_fact_added_to_reference"] += 1
            if p["fact"] not in facts:
                facts.append(p["fact"])
            idx = sorted(set(dup) | {facts.index(p["fact"])})
            removed_idx[p["pair_id"]] = idx
            n["near_duplicates_labelled_with_removed"] += len(idx) - 1
        k = hashlib.sha1(src.encode()).hexdigest()[:12]; docs[k] = src
        units = [{"kind": "acu", "text": f} for f in facts]
        rows.append({"pair_id": f"omb:{key}|clean", "resource": f"omb_{stratum}", "system": "clean", "split": "omb",
                     "domain": stratum, "consultation": cid, "doc_id": k, "source": src, "candidate": ps[0]["clean"],
                     "units": units, "unit_labels": [1] * len(facts)})
        for p in ps:
            labels = [1] * len(facts)
            for i in removed_idx[p["pair_id"]]:
                labels[i] = 0
            rows.append({"pair_id": f"omb:{p['pair_id']}", "resource": f"omb_{stratum}", "system": "errored", "split": "omb",
                         "domain": stratum, "consultation": cid, "doc_id": k, "source": src, "candidate": p["errored"],
                         "units": units, "unit_labels": labels, "removed_fact": p["fact"], "class": p["class"],
                         "residual_level": p["residual_level"], "severity": p["severity"], "cell": p.get("cell")})
        acus_per_row.append(len(facts))
    with open(out / "acu_units_omb.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    m = 0
    with open(out / "decompose_tasks_omb.jsonl", "w", encoding="utf-8") as fh:
        for k, src in docs.items():
            for i, (s, e) in enumerate(common.sentence_spans(src)):
                sent = src[s:e].strip()
                if len(sent.split()) < 4:
                    continue
                fh.write(json.dumps({"task_id": f"{k}:{i}", "doc_key": k, "splits": ["omb"], "resources": ["omb"],
                                     "start": s, "end": e, "sentence": sent}, ensure_ascii=False) + "\n"); m += 1
    pids = [r["pair_id"] for r in rows]
    if len(set(pids)) != len(pids):
        raise SystemExit("[hj10b-build] ABORT: pair_id not unique")
    census = {"rows": len(rows), "errored_notes": sum(r["system"] == "errored" for r in rows),
              "clean_twins": sum(r["system"] == "clean" for r in rows), "consultations": len(docs),
              "reference_facts_per_consultation_median": st.median(acus_per_row),
              "reference_facts_per_consultation_min_max": [min(acus_per_row), max(acus_per_row)],
              "omitted_facts_per_errored_note_median": st.median(sum(1 for x in r["unit_labels"] if x == 0) for r in rows if r["system"] == "errored"),
              "sentence_tasks": m, "transcript_words_median": st.median(len(s.split()) for s in docs.values()),
              "note_words_median": st.median(len(r["candidate"].split()) for r in rows),
              "label_semantics": "unit_labels 1 = the note states the fact; 0 = the note is the one the fact was removed from",
              "duplicate_rule": f"a sheet fact with content-token F1 >= {DUP_F1} against the removed statement is labelled 0 with it",
              **{k: v for k, v in n.items()},
              "sha256_acu": hashlib.sha256((out / "acu_units_omb.jsonl").read_bytes()).hexdigest()}
    (out / "build_manifest_omb.json").write_text(json.dumps(census, indent=1))
    print(f"[hj10b-build] {json.dumps(census)}", flush=True)


def inventory(a) -> None:
    from mars_v2.units import get_extractor
    extractor = get_extractor("entity")
    props = defaultdict(list)
    for f in a.props:
        for r in common.load_jsonl(Path(f)):
            for p in r.get("props", []):
                props[r["doc_key"]].append({"kind": "prop_llm", "text": p, "start": r["start"], "end": r["end"], "label": "", "args": {}})
    cache = {}; n_units = []; out = Path(a.out)
    with open(out / "units_gemma+ent_omb.jsonl", "w", encoding="utf-8") as fh:
        for r in common.load_jsonl(Path(a.acu)):
            k = hashlib.sha1(r["source"].encode()).hexdigest()[:12]
            if k not in cache:
                cache[k] = list(props.get(k, [])) + [u.to_dict() for u in extractor(r["source"])]
            units = cache[k]; n_units.append(len(units))
            fh.write(json.dumps({"pair_id": r["pair_id"], "resource": r["resource"], "system": r["system"], "split": "omb",
                                 "source": r["source"], "candidate": r["candidate"], "units": units}, ensure_ascii=False) + "\n")
    docs_with_props = sum(1 for k in cache if props.get(k))
    if docs_with_props != len(cache):
        raise SystemExit(f"[hj10b-inventory] ABORT: {len(cache) - docs_with_props} consultations have no LLM propositions")
    print(f"[hj10b-inventory] {len(n_units)} notes, {len(cache)} consultations, {sum(n_units) / max(1, len(n_units)):.1f} units/note", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--pairs", required=True); b.add_argument("--sheets", required=True)
    b.add_argument("--transcripts", required=True); b.add_argument("--out", required=True)
    i = sub.add_parser("inventory"); i.add_argument("--acu", required=True); i.add_argument("--props", nargs="+", required=True); i.add_argument("--out", required=True)
    a = ap.parse_args(); {"build": build, "inventory": inventory}[a.cmd](a)


if __name__ == "__main__":
    main()
