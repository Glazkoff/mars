#!/usr/bin/env python3
"""M-D.1 -- RAGTruth-processed test responses as sentence-decomposition tasks (M32 format) plus the pair index
with the annotated hallucination spans (character offsets into the response). Queued design in PREREG.md; this
builder prepares data and reads no result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from mars.inventory import sentence_spans  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ragtruth-test", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(a.ragtruth_test)
    n = {"rows": 0, "tasks": 0, "rows_with_spans": 0, "spans": 0, "skipped_quality": 0}
    with open(out / "tasks_md.jsonl", "w", encoding="utf-8") as ft, open(out / "pairs_md.jsonl", "w", encoding="utf-8") as fp:
        for i, r in enumerate(d.itertuples()):
            if str(r.quality) != "good":
                n["skipped_quality"] += 1
                continue
            output = str(r.output)
            try:
                spans = json.loads(r.hallucination_labels) if isinstance(r.hallucination_labels, str) else list(r.hallucination_labels or [])
            except json.JSONDecodeError:
                spans = []
            spans = [{"start": int(s["start"]), "end": int(s["end"]), "label_type": s.get("label_type")} for s in spans]
            sents = [(s, e) for s, e in sentence_spans(output) if output[s:e].strip()]
            for j, (s, e) in enumerate(sents):
                ft.write(json.dumps({"task_id": f"rt:{i}:{j}", "doc_key": f"rt:{i}", "splits": ["md"], "resources": ["ragtruth"],
                                     "start": s, "end": e, "sentence": output[s:e].strip()}) + "\n"); n["tasks"] += 1
            fp.write(json.dumps({"pair_id": f"rt:{i}", "row": i, "id": str(r.id), "task_type": str(r.task_type), "model": str(r.model),
                                 "context": str(r.context), "query": str(r.query), "output": output, "sentences": sents,
                                 "spans": spans}, ensure_ascii=False) + "\n")
            n["rows"] += 1; n["rows_with_spans"] += int(bool(spans)); n["spans"] += len(spans)
    (out / "tasks_md_manifest.json").write_text(json.dumps(n, indent=1))
    print(f"[md-tasks] {json.dumps(n)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
