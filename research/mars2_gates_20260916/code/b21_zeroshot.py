#!/usr/bin/env python3
"""B21 -- zero-shot baselines on the same units as MARS-2.

Claim per unit: the ACU text (acu set) or the rendered claim of an extracted unit (m2 set:
proposition -> "subject predicate object", entity -> containing source sentence). Document for
every entailment scorer = the candidate summary. Omission score = 1 - support.
Systems: lex (token recall, ROUGE-L recall), minilm (max cosine to a candidate sentence),
minicheck (Flan-T5-L), alignscore (large), nli (RoBERTa-L-MNLI window), summac_zs (DeBERTa-L-MNLI).
Scorers are the repo's canonical ones (scripts/strengthen_score_cells.py).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import common

ssc = common._load("strengthen_score_cells")
CHUNK = 256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit-set", choices=["acu", "m2"], required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--units", default="", help="override rows file (acu_units_<split>.jsonl or labels_<split>.jsonl)")
    ap.add_argument("--systems", default="lex,minilm,minicheck,alignscore,nli,summac_zs")
    ap.add_argument("--out", default=str(common.OUT_ROOT / "b21" / "scores"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--units-name", default="", help="name written into the score files instead of --unit-set "
                                                     "(so several inventories can share one --out directory)")
    a = ap.parse_args()
    uname = a.units_name or a.unit_set
    rows = common.load_jsonl(Path(a.units)) if a.units else common.unit_rows(a.unit_set, a.split)
    if a.unit_set == "m2":
        rows = [r for r in rows if r.get("units")]
    if a.limit:
        rows = rows[:a.limit]
    flat = []      # (row index, unit index, claim)
    for i, r in enumerate(rows):
        for j, u in enumerate(r["units"]):
            claim = u["text"] if a.unit_set == "acu" else common.render_claim(r["source"], u)
            flat.append((i, j, " ".join(claim.split()[:96])))   # cap: a run-on "sentence" must not exceed a 512-token pair
    print(f"[b21-zs] unit_set={a.unit_set} pairs={len(rows)} units={len(flat)} systems={a.systems}", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    def emit(name: str, omission: list[float]) -> None:
        per = {r["pair_id"]: [np.nan] * len(r["units"]) for r in rows}
        for (i, j, _), v in zip(flat, omission):
            per[rows[i]["pair_id"]][j] = v
        common.write_scores(out / f"{uname}_{name.replace(':', '_')}.jsonl", name, uname, per)
        print(f"[b21-zs] wrote {name}", flush=True)

    for sysname in a.systems.split(","):
        t0 = time.time()
        if sysname == "lex":
            emit("lex:token_recall", [1 - common.token_recall(c, rows[i]["candidate"]) for i, _, c in flat])
            emit("lex:rougeL_recall", [1 - common.rougeL_recall(c, rows[i]["candidate"]) for i, _, c in flat])
        elif sysname == "minilm":
            import torch
            from sentence_transformers import SentenceTransformer
            dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
            st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
            cand_sents = [common.sentences(r["candidate"]) or [r["candidate"]] for r in rows]
            cand_emb = [st.encode(s, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False) for s in cand_sents]
            claim_emb = st.encode([c for _, _, c in flat], batch_size=128, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=False)
            emit("emb:minilm_maxcos", [1 - float((cand_emb[i] @ claim_emb[k]).max()) for k, (i, _, _) in enumerate(flat)])
        else:
            import torch
            dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
            sc = ssc.build_scorer(sysname, a.batch, dev)
            vals: list[float] = []
            for s in range(0, len(flat), CHUNK):
                ch = flat[s:s + CHUNK]
                vals.extend(sc.score([rows[i]["candidate"] for i, _, _ in ch], [c for _, _, c in ch]))
                if (s // CHUNK) % 10 == 0:
                    print(f"[b21-zs]   {sysname} {min(s + CHUNK, len(flat))}/{len(flat)} ({time.time() - t0:.0f}s)", flush=True)
            emit(f"nli:{sysname}", [1 - float(v) for v in vals])
            del sc
        print(f"[b21-zs] {sysname} done in {time.time() - t0:.0f}s", flush=True)
    print("[b21-zs] done", flush=True)


if __name__ == "__main__":
    main()
