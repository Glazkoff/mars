#!/usr/bin/env python3
"""A6 -- evidence the B1 judge-label census from inside the bounded artifact set (PREREG Wave A / A6, closes C06).

The manuscript asserts a judge-label training budget of "85,600 cells over 2,625 pairs" and cites
`~/data/plan2026/b1/training_rows.jsonl` together with
`~/results/mars2_gates/b21/xenc_seed20260906/training_config.json`. The audit records that assertion as NOT
EVIDENCED: the config's `examples` field is a number the training run printed about itself, and nothing in the
bounded artifact set recomputes it from the rows.

This script recomputes it directly, using the EXACT selection `b21_finetune_xenc.py` applies:

    rows = [r for r in load_jsonl(train) if r.get("split", "train") == "train" and r.get("units")]
    cells = [(r, u, lab) for r in rows for u, lab in zip(r["units"], r["unit_labels"]) if lab is not None]

and reports rows, distinct pair_id, distinct source DOCUMENTS (`common.doc_key`), cells, the label-source and
resource breakdowns, and the sha256 of every file read. If the recomputed number differs from 85,600 / 2,625,
the recomputed number is the number -- that is the point of the exercise. No bar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(common.B1_DIR / "training_rows.jsonl"))
    ap.add_argument("--configs", nargs="*", default=[], help="training_config.json files whose `examples` field is the claim under audit")
    ap.add_argument("--claim-cells", type=int, default=85600)
    ap.add_argument("--claim-pairs", type=int, default=2625)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    train = Path(a.train)
    all_rows = list(common.load_jsonl(train))
    # The trainer's own filter, copied verbatim from b21_finetune_xenc.py.
    rows = [r for r in all_rows if r.get("split", "train") == "train" and r.get("units")]

    cells = 0; cells_by_source = Counter(); cells_by_resource = Counter(); cells_by_label = Counter()
    units_total = 0; none_labels = 0
    for r in rows:
        labs = r.get("unit_labels") or []
        units_total += len(r["units"])
        for u, lab in zip(r["units"], labs):
            if lab is None:
                none_labels += 1
                continue
            cells += 1
            cells_by_resource[r.get("resource")] += 1
            cells_by_label[int(lab)] += 1
        src = r.get("unit_label_source") or []
        for u, lab, s in zip(r["units"], labs, src):
            if lab is not None:
                cells_by_source[s] += 1

    pairs = {r["pair_id"] for r in rows}
    docs = {common.doc_key(r["source"]) for r in rows}
    eligible = [r for r in rows if r.get("training_eligible")]
    cells_eligible = sum(1 for r in eligible for lab in (r.get("unit_labels") or []) if lab is not None)

    rep = {
        "registered": "research/marsc_strengthen_20260918/PREREG.md Wave A / A6 (closes provenance C06)",
        "selection": "b21_finetune_xenc.py: split=='train' (default 'train') and non-empty units; cell = (unit, label) with label is not None",
        "file": str(train), "file_sha256": sha256(train), "file_bytes": train.stat().st_size,
        "rows_in_file": len(all_rows), "rows_selected": len(rows),
        "n_pairs_distinct": len(pairs), "n_documents_distinct": len(docs),
        "units_in_selected_rows": units_total, "labels_none": none_labels,
        "cells": cells,
        "cells_by_label_value": {str(k): v for k, v in sorted(cells_by_label.items())},
        "cells_by_label_source": dict(cells_by_source.most_common()),
        "cells_by_resource": dict(cells_by_resource.most_common()),
        "rows_training_eligible": len(eligible), "cells_training_eligible": cells_eligible,
        "claim": {"cells": a.claim_cells, "pairs": a.claim_pairs},
        "matches_claim": {"cells": cells == a.claim_cells, "pairs": len(pairs) == a.claim_pairs,
                          "rows": len(rows) == a.claim_pairs},
        "configs": {},
    }
    for c in a.configs:
        p = Path(c)
        if not p.exists():
            rep["configs"][c] = {"exists": False}
            continue
        cfg = json.loads(p.read_text())
        rep["configs"][c] = {"exists": True, "sha256": sha256(p), "examples": cfg.get("examples"),
                             "train": cfg.get("train"), "seed": cfg.get("seed"), "steps": cfg.get("steps"),
                             "batch": cfg.get("batch"), "epochs": cfg.get("epochs")}

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "a6_b1_census.json").write_text(json.dumps(rep, indent=1))
    print(f"[a6] rows_in_file={rep['rows_in_file']} rows_selected={rep['rows_selected']} "
          f"pairs={rep['n_pairs_distinct']} documents={rep['n_documents_distinct']}", flush=True)
    print(f"[a6] cells={rep['cells']} (claim {a.claim_cells}: {rep['matches_claim']['cells']}) "
          f"pairs vs claim {a.claim_pairs}: {rep['matches_claim']['pairs']}", flush=True)
    print(f"[a6] label sources {rep['cells_by_label_source']}", flush=True)
    print(f"[a6] resources {rep['cells_by_resource']}", flush=True)
    for c, v in rep["configs"].items():
        print(f"[a6] config {Path(c).parent.name}: examples={v.get('examples')} exists={v['exists']}", flush=True)
    print(f"[a6] written {out / 'a6_b1_census.json'}", flush=True)


if __name__ == "__main__":
    main()
