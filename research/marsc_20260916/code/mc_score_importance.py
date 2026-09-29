#!/usr/bin/env python3
"""MARS-C R10 -- score a units file with the FROZEN source-only importance tagger checkpoints.

m31_tagger.py trains and scores in one pass and has no inference entry point; the E2 runs saved the encoder and
the head (`--save-model`), so the policy can be applied to a split it never saw without retraining anything.
Used by R10 to carry the importance policy, frozen exactly as E2 left it (weights and the TUNE-chosen threshold),
onto the held-out test inventories after the estimator repair reversed the decision that dropped it.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars3_20260916" / "code"))
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
from m3common import gates  # noqa: E402
from m31_tagger import Tagger, score_units  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="glob of dirs holding backbone/ and head.pt")
    ap.add_argument("--units", nargs="+", required=True, help="name=path")
    ap.add_argument("--tag", default="importance_all")
    ap.add_argument("--scores-out", required=True)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument("--source-only", action="store_true", default=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    a.limit = 0
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    from transformers import AutoTokenizer
    dirs = [d for d in sorted(glob.glob(a.ckpts)) if Path(d, "head.pt").exists()]
    assert dirs, f"no checkpoint with head.pt matched {a.ckpts!r}"
    so = Path(a.scores_out); so.mkdir(parents=True, exist_ok=True)
    for d in dirs:
        seed = m.group(1) if (m := re.search(r"seed(\d+)", d)) else "0"
        bb = str(Path(d) / "backbone")
        tok = AutoTokenizer.from_pretrained(bb)
        model = Tagger(bb).to(dev)
        model.head.load_state_dict(torch.load(Path(d) / "head.pt", map_location=dev))
        model.eval()
        print(f"[imp-score] {d} -> seed {seed}", flush=True)
        for spec in a.units:
            name, path = spec.split("=", 1)
            rows = [r for r in gates.load_jsonl(Path(path)) if r.get("units")]
            with torch.inference_mode():
                score_units(model, tok, rows, dev, a, name, f"m3:{a.tag}", so / f"{name}_{a.tag}_seed{seed}.jsonl")
        del model
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
    print("[imp-score] done", flush=True)


if __name__ == "__main__":
    main()
