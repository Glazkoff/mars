#!/usr/bin/env python3
"""B20 -- label-free adaptation rows for a held-out resource.

From the TRAIN-split SOURCES of one resource only (no system candidates, no judge labels):
pseudo-candidate = an ordered random subset of source sentences (dialogue turns for SAMSum),
units = the B1 extractor's entity+proposition units (cap 40), label 1 iff the unit's span lies
inside a kept sentence. Rows follow training_rows.jsonl so mars_v2.train's support_examples applies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import common

sys.path.insert(0, str(common.REPO))
from mars_v2.units import get_extractor  # noqa: E402

MAX_UNITS = 40


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, help="resource name, e.g. rose_samsum")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-cands", type=int, default=4)
    ap.add_argument("--frac-lo", type=float, default=0.3)
    ap.add_argument("--frac-hi", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=common.SEED)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    srcs = sorted({r["source"] for r in common.load_jsonl(common.B1_DIR / f"pairs_{a.split}.jsonl") if r["resource"] == a.domain})
    if a.limit:
        srcs = srcs[:a.limit]
    extractor = get_extractor("entity+proposition")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = n_pos = n_units = 0
    with open(out, "w", encoding="utf-8") as fh:
        for src in srcs:
            units = [u.to_dict() for u in extractor(src)[:MAX_UNITS]]
            spans = common.sentence_spans(src)
            if not units or len(spans) < 2:
                continue
            did = hashlib.sha1(src.encode()).hexdigest()[:12]
            for k in range(a.n_cands):
                for _attempt in range(3):
                    frac = rng.uniform(a.frac_lo, a.frac_hi)
                    keep = sorted(rng.sample(range(len(spans)), max(1, round(frac * len(spans)))))
                    kept = [spans[i] for i in keep]
                    labels = [1 if any(s <= u["start"] and u["end"] <= e for s, e in kept) else 0 for u in units]
                    if 0 < sum(labels) < len(labels):
                        break
                else:
                    continue
                cand = " ".join(src[s:e].strip() for s, e in kept)
                fh.write(json.dumps({"pair_id": f"pseudo:{a.domain}:{did}:{k}", "cls": "natural", "resource": a.domain,
                                     "system": "pseudo_extractive", "split": "train", "source": src, "candidate": cand,
                                     "units": units, "unit_labels": labels, "consistency_label": None,
                                     "training_eligible": True}) + "\n")
                n_rows += 1; n_pos += sum(labels); n_units += len(labels)
    man = {"domain": a.domain, "sources": len(srcs), "rows": n_rows, "units": n_units,
           "covered_fraction": n_pos / max(1, n_units), "seed": a.seed, "n_cands": a.n_cands,
           "frac": [a.frac_lo, a.frac_hi], "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    common.dump_json(out.with_suffix(".manifest.json"), man)
    print(f"[b20-build] {json.dumps(man)}", flush=True)


if __name__ == "__main__":
    main()
