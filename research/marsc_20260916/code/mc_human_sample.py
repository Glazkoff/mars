#!/usr/bin/env python3
"""MARS-C G6 -- blinded human confirmation sample of EMITTED facts (PLAN_MAIN_CONTRIBUTION.md, block G6).

From the evaluator's emitted-unit dump: per system, N units stratified by evaluator status (unknown / hit / false alert),
shuffled, system hidden. Annotators answer two questions per item; the key stays in a separate file.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402

INSTRUCTIONS = """# Omission confirmation — annotator instructions

Each item shows a SOURCE document, a SUMMARY of it, and one FACT that a system claims the summary left out.
Answer two questions per item (write A or B in the answer columns of the CSV):

Q1 SUPPORTED — Is the FACT stated in the SOURCE, or does it follow directly from it?
   A = yes, supported by the source.  B = no (absent, contradicted, or invented).

Q2 CONVEYED — Does the SUMMARY convey the FACT? Equivalent names, abbreviations, pronouns and faithful paraphrases count
   as conveyed; do not require the wording verbatim; do not judge whether the fact is important.
   A = conveyed / covered.  B = omitted / not conveyed.

A fact that is supported (Q1 = A) and not conveyed (Q2 = B) is a confirmed omission. Do not guess which system produced
an item; items are shuffled and systems are hidden. Work independently of the other annotator.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emitted", required=True); ap.add_argument("--acu", required=True)
    ap.add_argument("--systems", nargs="+", required=True); ap.add_argument("--per-system", type=int, default=100)
    ap.add_argument("--strata", default="unknown=0.6,hit=0.2,false_alert=0.2"); ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--out", required=True); ap.add_argument("--id-offset", type=int, default=0, help="first item number minus one (extension batches)")
    a = ap.parse_args()
    rng = random.Random(a.seed); pairs = {r["pair_id"]: r for r in common.load_jsonl(Path(a.acu))}
    strata = {k: float(v) for k, v in (x.split("=") for x in a.strata.split(","))}
    pool = defaultdict(lambda: defaultdict(list))
    for r in common.load_jsonl(Path(a.emitted)):
        if r["system"] in a.systems and r["pair_id"] in pairs and r.get("emitted"):
            for e in r["emitted"]:
                pool[r["system"]][e["status"]].append((r["pair_id"], e))
    items = []
    for s_ in a.systems:
        for st, frac in strata.items():
            cand = pool[s_][st]; rng.shuffle(cand); n = round(a.per_system * frac)
            for pid, e in cand[:n]:
                items.append({"system": s_, "pair_id": pid, "status": st, "fact": e["text"], "score": e.get("score")})
    rng.shuffle(items)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "human_sample_blind.jsonl", "w", encoding="utf-8") as fb, open(out / "human_sample_key.jsonl", "w", encoding="utf-8") as fk, \
         open(out / "human_sample_blind.csv", "w", encoding="utf-8", newline="") as fc:
        w = csv.writer(fc); w.writerow(["item_id", "source", "summary", "fact", "Q1_supported_A_or_B", "Q2_conveyed_A_or_B"])
        for i, it in enumerate(items):
            iid = f"H{a.id_offset + i + 1:03d}"; pr = pairs[it["pair_id"]]
            fb.write(json.dumps({"item_id": iid, "source": pr["source"], "summary": pr["candidate"], "fact": it["fact"]}) + "\n")
            fk.write(json.dumps({"item_id": iid, **it}) + "\n"); w.writerow([iid, pr["source"], pr["candidate"], it["fact"], "", ""])
    (out / "INSTRUCTIONS.md").write_text(INSTRUCTIONS)
    per = defaultdict(int)
    for it in items:
        per[(it["system"], it["status"])] += 1
    print(f"[g6] {len(items)} blinded items -> {out}; strata {dict(per)}", flush=True)


if __name__ == "__main__":
    main()
