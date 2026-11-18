#!/usr/bin/env python3
"""MARS-C strengthen B0 -- the standing control harness for the OFF-THE-SHELF verifier families.

The cross-encoder families already have their three candidate modes (mc_score_units.py --candidate-mode for the
MARS-C humanfact head, m31_score_pipeline_all.py --candidate-mode for the b21 judge-label head). The zero-shot
and counter families did not, so every off-the-shelf row in the paper was a real-summary row with no control.
This file supplies the missing half, with three properties that make the grid comparable:

  1. The premise under --candidate-mode shuffled/empty is produced by mc_score_units.premises, imported, not
     re-implemented -- same seeded derangement within (resource, system), same row filter (`r.get("units")`),
     same row order. The shuffled premise a zero-shot family sees is therefore the SAME wrong summary the
     cross-encoder families saw, so the two control columns are paired, not merely analogous.
  2. The claim rendering is the family-native one already used for the published real-summary rows
     (common.render_claim for an extracted-unit inventory), so real / shuffled / empty differ ONLY in the premise.
  3. Output is the repo unit_scores format (common.write_scores), so mc_diversity.py -> mc_e2_eval.py consume it
     unchanged and no evaluator code is touched.

Omission score = 1 - support, higher = more likely omitted, as in research/mars2_gates_20260916/code/b21_zeroshot.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402

CHUNK = 256


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


msu = _load(REPO / "research" / "marsc_20260916" / "code" / "mc_score_units.py")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", required=True, help="name=path inventory files")
    ap.add_argument("--systems", default="lex,minilm,nli,summac_zs")
    ap.add_argument("--candidate-mode", choices=["real", "shuffled", "empty"], default="real")
    ap.add_argument("--shuffle-seed", type=int, default=20260917,
                    help="must match the cross-encoder control arms (mc_score_units.py default) or the "
                         "shuffled columns are not paired")
    ap.add_argument("--empty-text", default="")
    ap.add_argument("--scores-out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="pilot only: first N pairs")
    ap.add_argument("--claim-words", type=int, default=96)
    a = ap.parse_args()
    out = Path(a.scores_out)
    out.mkdir(parents=True, exist_ok=True)
    mode = a.candidate_mode

    for spec in a.units:
        uname, path = spec.split("=", 1)
        rows = [r for r in common.load_jsonl(Path(path)) if r.get("units")]
        if a.limit:
            rows = rows[:a.limit]
        prem = msu.premises(rows, a)                       # identical derangement to the cross-encoder controls
        flat = [(i, j, " ".join(common.render_claim(r["source"], u).split()[:a.claim_words]))
                for i, r in enumerate(rows) for j, u in enumerate(r["units"])]
        n_changed = sum(1 for i in range(len(rows)) if prem[i] != " ".join(rows[i]["candidate"].split()[:380]))
        print(f"[b0zs] {uname} mode={mode}: {len(rows)} pairs, {len(flat)} units, "
              f"{n_changed} pairs whose premise differs from their own summary", flush=True)

        def emit(sysname: str, omission: list[float]) -> None:
            per = {r["pair_id"]: [np.nan] * len(r["units"]) for r in rows}
            for (i, j, _), v in zip(flat, omission):
                per[rows[i]["pair_id"]][j] = v
            fn = f"{uname}_{sysname.replace(':', '_')}_{mode}.jsonl"
            common.write_scores(out / fn, f"{sysname}.{mode}", uname, per)
            print(f"[b0zs] wrote {fn}", flush=True)

        for sysname in [s for s in a.systems.split(",") if s]:
            t0 = time.time()
            if sysname == "lex":
                emit("lex:token_recall", [1 - common.token_recall(c, prem[i]) for i, _, c in flat])
                emit("lex:rougeL_recall", [1 - common.rougeL_recall(c, prem[i]) for i, _, c in flat])
            elif sysname == "minilm":
                import torch
                from sentence_transformers import SentenceTransformer
                dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
                st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
                cand_emb = [st.encode(common.sentences(p) or [p], convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=False) for p in prem]
                claim_emb = st.encode([c for _, _, c in flat], batch_size=128, convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=False)
                emit("emb:minilm_maxcos",
                     [1 - float((cand_emb[i] @ claim_emb[k]).max()) for k, (i, _, _) in enumerate(flat)])
                del st
            else:
                import torch
                dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
                ssc = common._load("strengthen_score_cells")
                sc = ssc.build_scorer(sysname, a.batch, dev)
                vals: list[float] = []
                for s in range(0, len(flat), CHUNK):
                    ch = flat[s:s + CHUNK]
                    vals.extend(sc.score([prem[i] for i, _, _ in ch], [c for _, _, c in ch]))
                    if (s // CHUNK) % 20 == 0:
                        done = min(s + CHUNK, len(flat))
                        el = time.time() - t0
                        print(f"[b0zs]   {sysname} {done}/{len(flat)} ({el:.0f}s, {done / max(el, 1e-9):.1f} units/s)",
                              flush=True)
                assert len(vals) == len(flat), f"{sysname}: {len(vals)} values for {len(flat)} units"
                emit(f"nli:{sysname}", [1 - float(v) for v in vals])
                del sc
                if dev.startswith("cuda"):
                    torch.cuda.empty_cache()
            el = time.time() - t0
            print(f"[b0zs] {uname} {sysname} mode={mode} done in {el:.0f}s "
                  f"({len(flat) / max(el, 1e-9):.1f} units/s)", flush=True)
    print("[b0zs] done", flush=True)


if __name__ == "__main__":
    main()
