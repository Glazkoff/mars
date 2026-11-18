#!/usr/bin/env python3
"""B18 -- materialise the human-labelled ACU unit set and the supplied-target judge tasks for it.

Writes <out>/acu_units_<split>.jsonl (unit-set rows, label 1 = present) and
<out>/judge_tasks_acu_<split>.jsonl in the b1_judge_labels.py task format, plus manifest.json.
No model, no label is read beyond RoSE's human ACU labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import common


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--out", default=str(common.OUT_ROOT / "b18"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    man = {"registered": "research/mars2_gates_20260916/PREREG.md B18", "splits": {}, "sha256": {}}
    for split in a.splits.split(","):
        rows = common.acu_rows(split)
        cnt = Counter(); n_units = 0; docs = set(); lab = Counter()
        with open(out / f"acu_units_{split}.jsonl", "w", encoding="utf-8") as fu, \
                open(out / f"judge_tasks_acu_{split}.jsonl", "w", encoding="utf-8") as ft:
            for r in rows:
                fu.write(json.dumps(r) + "\n"); cnt[r["resource"]] += 1; docs.add(common.doc_key(r["source"]))
                for i, u in enumerate(r["units"]):
                    n_units += 1; lab[r["unit_labels"][i]] += 1
                    ft.write(json.dumps({"task_id": f"{r['pair_id']}#a{i}", "pair_id": r["pair_id"], "unit_index": i,
                                         "split": split, "source": r["source"], "candidate": r["candidate"],
                                         "unit_text": u["text"], "unit_kind": "acu"}) + "\n")
        man["splits"][split] = {"pairs": len(rows), "acus": n_units, "docs": len(docs), "by_resource": dict(cnt),
                                "labels_present_1": dict(lab)}
        for f in (f"acu_units_{split}.jsonl", f"judge_tasks_acu_{split}.jsonl"):
            man["sha256"][f] = hashlib.sha256((out / f).read_bytes()).hexdigest()
        print(f"[b18] {split}: {man['splits'][split]}", flush=True)
    common.dump_json(out / "manifest.json", man)


if __name__ == "__main__":
    main()
