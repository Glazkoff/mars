#!/usr/bin/env python3
"""M-G.2 (audit response, 2026-09-28) -- RoSE MARS-R without the gold ACUs (review finding F03).

On RoSE the published MARS-R reads the human ACUs directly, while AutoACU extracts its own units from the
reference, so the RoSE column mixes verifier quality with privileged units. This builds two automatic-unit arms
on the identical 2,688 RoSE test pairs:

  gemma  the reference summary decomposed by the frozen M32 Gemma-4-31B sentence decomposer -- the extractor
         MARS-R already uses for UniSumEval and SummEval references (ref_keys -> props)
  a2cu   the ACUs A2CU generated from the same reference (cached by the A2CU scorer, keyed by reference text),
         so MARS-R and A2CU read the same automatic units and differ only in the matcher

Writes pairs_rose_<arm>.jsonl (the pairs_ma schema, gold ACUs removed or replaced) and the decomposition task
file for the reference summaries. Human acu_recall is untouched: it is the target, not an input.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ma_tasks", HERE / "ma_tasks.py")
ma_tasks = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ma_tasks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="pairs_ma.jsonl")
    ap.add_argument("--a2cu-acus", required=True, help="scores_a2cu.acus.json (reference text -> ACUs)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    acus = json.loads(Path(a.a2cu_acus).read_text())
    rose = [json.loads(line) for line in open(a.pairs, encoding="utf-8")]
    rose = [p for p in rose if p["dataset"] == "rose"]
    n = {"rose_pairs": len(rose), "with_reference": 0, "references": 0, "tasks": 0, "a2cu_hit": 0}
    seen = set()
    with open(out / "tasks_rose_ref.jsonl", "w", encoding="utf-8") as ft, \
         open(out / "pairs_rose_gemma.jsonl", "w", encoding="utf-8") as fg, \
         open(out / "pairs_rose_a2cu.jsonl", "w", encoding="utf-8") as fa:
        for p in rose:
            if not p["references"]:
                continue
            n["with_reference"] += 1
            ref = p["references"][0]; rk = f"rose:ref:{p['doc_id']}"
            if rk not in seen:
                seen.add(rk); n["references"] += 1
                n["tasks"] += ma_tasks.sent_tasks(ft, rk, ref)
            g = dict(p); g.pop("acus", None); g.pop("acu_labels", None); g["ref_keys"] = [rk]
            fg.write(json.dumps(g) + "\n")
            units = acus.get(ref)
            if units:
                n["a2cu_hit"] += 1
                q = dict(p); q.pop("acu_labels", None); q["acus"] = list(units); q["ref_keys"] = []
                fa.write(json.dumps(q) + "\n")
    (out / "manifest.json").write_text(json.dumps(n, indent=1))
    print(f"[mg-rose] {json.dumps(n)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
