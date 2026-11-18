#!/usr/bin/env python3
"""MARS-C G2 x G6 -- validate the LLM judges against the HUMAN labels of the blinded sample (not against lexical status).

Joins the G2 judge tasks (task_id = sha1(pair_id + fact)) with the annotators' answers: per family, accuracy of the coverage
answer against Q2 and of the true-omission verdict against (Q1 = A and Q2 = B), on items where both annotators agree and on all.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--families", nargs="+", required=True); ap.add_argument("--export", required=True); ap.add_argument("--key", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = {json.loads(l)["item_id"]: json.loads(l) for l in open(a.key, encoding="utf-8")}
    lab = defaultdict(dict)
    for l in open(a.export, encoding="utf-8"):
        r = json.loads(l); lab[r["annotator"]][r["item_id"]] = r
    names = sorted(lab)
    tid_of = {iid: hashlib.sha1((k["pair_id"] + "\x1f" + k["fact"].strip().lower()).encode()).hexdigest()[:16] for iid, k in key.items()}
    fam = {}
    for spec in a.families:
        n, pat = spec.split("=", 1); fam[n] = {"coverage": {}, "support": {}}
        for f in sorted(glob.glob(pat)):
            for l in open(f, encoding="utf-8"):
                r = json.loads(l); fam[n][r["question"]][r["task_id"]] = r["p_b"]
    rep = {"families": {}}
    for n, q in fam.items():
        acc_q2 = defaultdict(list); acc_v = defaultdict(list)
        for iid, tid in tid_of.items():
            if tid not in q["coverage"] or tid not in q["support"]:
                continue
            j_om = q["coverage"][tid] >= 0.5; j_sup = q["support"][tid] < 0.5; j_v = j_om and j_sup
            for ann in names:
                r = lab[ann].get(iid)
                if r:
                    acc_q2[ann].append(j_om == (r["q2"] == "B")); acc_v[ann].append(j_v == (r["q1"] == "A" and r["q2"] == "B"))
            if len(names) >= 2 and all(iid in lab[x] for x in names[:2]):
                r1, r2 = lab[names[0]][iid], lab[names[1]][iid]
                if r1["q2"] == r2["q2"]:
                    acc_q2["both_agree"].append(j_om == (r1["q2"] == "B"))
                v1, v2 = (r1["q1"] == "A" and r1["q2"] == "B"), (r2["q1"] == "A" and r2["q2"] == "B")
                if v1 == v2:
                    acc_v["both_agree"].append(j_v == v1)
        rep["families"][n] = {"coverage_vs_human": {k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in acc_q2.items()},
                              "verdict_vs_human": {k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in acc_v.items()}}
        print(f"[g2xg6] {n}: coverage vs human " + ", ".join(f"{k} {d['accuracy']:.2f} (n={d['n']})" for k, d in rep['families'][n]['coverage_vs_human'].items()) + " | verdict vs human " + ", ".join(f"{k} {d['accuracy']:.2f}" for k, d in rep['families'][n]['verdict_vs_human'].items()), flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / "judge_vs_human.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
