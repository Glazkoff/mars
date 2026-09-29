#!/usr/bin/env python3
"""M-B.1 -- LLM-AggreFact claims as sentence-decomposition tasks in the M32 format (task_id, doc_key, start, end,
sentence), one task per claim sentence, dev and test. The decomposer (m32_decompose.py, frozen prompt) writes
`props` per task; mb_score.py regroups them by doc_key = "<split>:<row>". No minimum sentence length: a claim can
be short, and a fact-free sentence returns NONE. Registered under M-B in PREREG.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggrefact-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n_tasks, n_rows = 0, 0
    with open(out / "tasks_mb.jsonl", "w", encoding="utf-8") as fh:
        for split in ("dev", "test"):
            df = pd.read_parquet(Path(a.aggrefact_dir) / f"{split}-00000-of-00001.parquet")
            for i, claim in enumerate(df["claim"].tolist()):
                claim = str(claim); n_rows += 1
                spans = common.sentence_spans(claim)
                for j, (s, e) in enumerate(spans):
                    sent = claim[s:e].strip()
                    if not sent:
                        continue
                    fh.write(json.dumps({"task_id": f"{split}:{i}:{j}", "doc_key": f"{split}:{i}", "dataset": str(df["dataset"].iloc[i]),
                                         "splits": [split], "resources": ["llm_aggrefact"], "start": s, "end": e, "sentence": sent}) + "\n")
                    n_tasks += 1
    (out / "tasks_mb_manifest.json").write_text(json.dumps({"rows": n_rows, "tasks": n_tasks}, indent=1))
    print(f"[mb-tasks] {n_rows} claims -> {n_tasks} sentence tasks -> {out / 'tasks_mb.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
