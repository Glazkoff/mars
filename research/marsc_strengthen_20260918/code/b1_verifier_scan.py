#!/usr/bin/env python3
"""B1 -- MiniCheck and AlignScore as MARS-C verifiers, shardable (PREREG.md Wave B / B1).

The R06 chain never evidenced these two: `r06_offshelf` (job 5001) was cancelled inside MiniCheck at
2816/123576 units and left an empty val/; `r06b_fast` (job 5035) ran only lex, minilm, nli, summac_zs.
This script completes the family under the IDENTICAL contract as `b21_zeroshot.py`:

  * claim per unit  -- acu: the ACU text; m2: `common.render_claim(source, unit)`, both capped at 96 words
  * premise         -- the candidate summary (`--candidate-mode real`), a deranged other summary
                       (`shuffled`) or nothing (`empty`); the derangement is byte-for-byte the rule in
                       `mc_score_units.premises` (seeded rotation within the resource x system stratum)
  * omission score  -- 1 - support, written with `common.write_scores`, so the output plugs into
                       `mc_diversity.py` -> `mc_e2_eval.py` unchanged
  * system names    -- `nli:minicheck` / `nli:alignscore` (+ `_shuf` / `_empty` suffix for the B0 controls),
                       matching the `nli:<name>` convention b21_zeroshot writes for every scorer arm

Two things `b21_zeroshot.py` cannot do, and why this file exists:

1. SHARDING. `b21_zeroshot.py` has only `--limit` (head truncation). `--shard i/n` here cuts the row list
   into n contiguous, unit-count-balanced blocks and writes `parts/<name>_<sys>.part<i>of<n>.jsonl`;
   `--merge` stitches them and REFUSES unless the merged file has exactly one record per scored pair with
   the unit count of the inventory file.

2. MiniCheck THROUGHPUT. `minicheck.inference.Inferencer.inference_example_batch` loops one (doc, claim)
   at a time and only batches the chunks of a single document, so its `batch_size` is dead for
   short premises; on top of that `device_map="auto"` puts an accelerate hook on every submodule. That is
   the whole of the measured 1.36 units/s that made job 5001 impossible under a 12 h limit.
   `FastMiniCheckFlanT5` reproduces the official flan-t5-large path exactly -- same checkpoint, same
   `sent_tokenize_with_newlines`, same 500-word chunking, same `'predict: ' + doc + eos + claim` input,
   same `logits[:, [3, 209]]` softmax, same max over chunks -- and only batches ACROSS examples.
   `--verify-canonical N` runs the official `strengthen_score_cells.MiniCheckFlanT5` beside it on the same
   N units and records the max absolute deviation. No result is taken from the fast path in a job that has
   not printed that deviation.

Deviation register (the only one):
   For `--candidate-mode empty` the premise is the empty string, whose chunk list is empty, and the
   official path would raise in `torch.cat([])`. The fast path substitutes a single empty chunk. This is
   unreachable for `real` and `shuffled`.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402

ssc = common._load("strengthen_score_cells")

CLAIM_WORD_CAP = 96           # b21_zeroshot.py
MINICHECK_CHUNK_WORDS = 500   # MiniCheck._score_inferencer default for flan-t5-large
MINICHECK_MAX_LEN = 2048      # Inferencer.max_model_len default for flan-t5-large
MINICHECK_CKPT = "lytang/MiniCheck-Flan-T5-Large"
SUPPORT_TOKENS = [3, 209]     # "no support" / "support", as hard-wired in minicheck.inference


# ----------------------------------------------------------------------------- premise modes
def premises(rows: list[dict], mode: str, shuffle_seed: int) -> list[str]:
    """One premise per row. `shuffled` is `mc_score_units.premises`' seeded within-stratum derangement.

    Unlike `mc_score_units.py` the candidate is NOT clipped to 380 words: `b21_zeroshot.py` feeds the full
    summary to every off-the-shelf verifier, and these rows must stay comparable to the R06b rows.
    """
    real = [r["candidate"] for r in rows]
    if mode == "real":
        return real
    if mode == "empty":
        return ["" for _ in rows]
    strata: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        strata.setdefault((r.get("resource", ""), r.get("system", "")), []).append(i)
    out = list(real)
    rng = random.Random(shuffle_seed)
    leftover: list[int] = []
    for _key, idxs in sorted(strata.items()):
        if len(idxs) < 2:
            leftover += idxs
            continue
        shift = 1 + rng.randrange(len(idxs) - 1)
        for t, dst in enumerate(idxs):
            out[dst] = real[idxs[(t + shift) % len(idxs)]]
    if len(leftover) >= 2:
        rng.shuffle(leftover)
        for t, dst in enumerate(leftover):
            out[dst] = real[leftover[(t + 1) % len(leftover)]]
    n_same = sum(1 for i in range(len(rows)) if out[i] == real[i])
    print(f"[b1] candidate-mode=shuffled: {len(rows)} rows, {len(strata)} strata, "
          f"{n_same} rows whose replacement summary is textually identical to their own", flush=True)
    return out


# ----------------------------------------------------------------------------- sharding
def shard_bounds(sizes: list[int], n: int) -> list[int]:
    """n+1 contiguous row boundaries splitting the row list into ~equal total unit counts."""
    tot = sum(sizes)
    bounds = [0]
    cum = 0
    for idx, s in enumerate(sizes):
        cum += s
        if len(bounds) < n and tot and cum >= len(bounds) * tot / n:
            bounds.append(idx + 1)
    while len(bounds) < n + 1:
        bounds.append(len(sizes))
    return bounds


# ----------------------------------------------------------------------------- fast MiniCheck
class FastMiniCheckFlanT5:
    """Batched re-implementation of the official flan-t5-large MiniCheck path. See the module docstring."""

    def __init__(self, batch_size: int, device: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from minicheck.inference import sent_tokenize_with_newlines
        self.torch = torch
        self.device = device
        self.bs = batch_size
        self.sent = sent_tokenize_with_newlines
        self.tok = AutoTokenizer.from_pretrained(MINICHECK_CKPT)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MINICHECK_CKPT).to(device).eval()
        self.eos = self.tok.eos_token
        self.idx = torch.tensor(SUPPORT_TOKENS, device=device)

    def chunks(self, doc: str) -> list[str]:
        """`Inferencer.inference_per_example`'s flan-t5 word-count chunker, verbatim in behaviour."""
        sents = self.sent(doc) or ['']
        out, cur, cw = [], [], 0
        for s in sents:
            w = len(s.split())
            if cw + w > MINICHECK_CHUNK_WORDS:
                out.append(' '.join(cur))
                cur, cw = [s], w
            else:
                cur.append(s)
                cw += w
        if cur:
            out.append(' '.join(cur))
        out = [c.replace(" \n ", "\n").strip() for c in out]
        out = [c for c in out if c != '']
        return out or ['']        # deviation register: reachable only for a blank premise

    def score(self, docs: list[str], sums: list[str]) -> list[float]:
        torch = self.torch
        flat: list[str] = []
        shape: list[int] = []
        for d, c in zip(docs, sums):
            ch = self.chunks(d)
            shape.append(len(ch))
            flat.extend('predict: ' + self.eos.join([x, c]) for x in ch)
        probs: list[float] = []
        with torch.no_grad():
            for a in range(0, len(flat), self.bs):
                mb = flat[a:a + self.bs]
                e = self.tok(mb, max_length=MINICHECK_MAX_LEN, truncation=True,
                             padding=True, return_tensors="pt").to(self.device)
                dii = torch.zeros((e["input_ids"].size(0), 1), dtype=torch.long, device=self.device)
                logits = self.model(input_ids=e["input_ids"], attention_mask=e["attention_mask"],
                                    decoder_input_ids=dii).logits.squeeze(1)
                lab = logits[:, self.idx].float()
                probs.extend(torch.nn.functional.softmax(lab, dim=-1)[:, 1].tolist())
        out, k = [], 0
        for n in shape:
            out.append(max(probs[k:k + n]))
            k += n
        return out


def build(name: str, batch: int, device: str):
    """Scorers with a `.score(docs, claims) -> support probability` contract."""
    if name == "minicheck":
        return FastMiniCheckFlanT5(batch, device)
    if name == "minicheck_canonical":
        return ssc.MiniCheckFlanT5(batch, device)
    if name == "alignscore":
        return ssc.AlignScoreCanonical(batch, device)
    if name == "nli":
        return ssc.WindowNLI("FacebookAI/roberta-large-mnli", batch, 400, 300, device)
    if name == "summac_zs":
        return ssc.SummaCZS("microsoft/deberta-large-mnli", batch, device)
    raise SystemExit(f"[b1] unsupported system {name!r}")


# The names b21_zeroshot.py writes, so an arm of this file is indistinguishable downstream from an R06b arm.
SYSNAMES = {"minicheck": ["nli:minicheck"], "alignscore": ["nli:alignscore"],
            "nli": ["nli:nli"], "summac_zs": ["nli:summac_zs"],
            "lex": ["lex:token_recall", "lex:rougeL_recall"], "minilm": ["emb:minilm_maxcos"]}
GPU_FREE = {"lex"}
MODESUF = {"real": "", "shuffled": "_shuf", "empty": "_empty"}


# ----------------------------------------------------------------------------- row loading
def load_rows(a) -> list[dict]:
    rows = common.load_jsonl(Path(a.units)) if a.units else common.unit_rows(a.unit_set, a.split)
    if a.unit_set == "m2":
        rows = [r for r in rows if r.get("units")]
    return rows


def flatten(rows: list[dict], prem: list[str], unit_set: str) -> list[tuple[int, int, str]]:
    flat = []
    for i, r in enumerate(rows):
        for j, u in enumerate(r["units"]):
            claim = u["text"] if unit_set == "acu" else common.render_claim(r["source"], u)
            flat.append((i, j, " ".join(claim.split()[:CLAIM_WORD_CAP])))
    del prem
    return flat


# ----------------------------------------------------------------------------- merge
def do_merge(a) -> None:
    rows = load_rows(a)
    want = {r["pair_id"]: len(r["units"]) for r in rows}
    parts_dir = Path(a.parts or (Path(a.out) / "parts"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"units": a.units, "units_name": a.units_name, "n_pairs_expected": len(want), "merged": {}}
    todo = [(n, m) for sysid in a.systems.split(",") for n in SYSNAMES[sysid] for m in a.modes.split(",")]
    for name, mode in todo:
            stem = f"{a.units_name}_{(name + MODESUF[mode]).replace(':', '_')}"
            parts = sorted(parts_dir.glob(f"{stem}.part*of*.jsonl"))
            if not parts:
                print(f"[b1:merge] no parts for {stem}", flush=True)
                continue
            seen: dict[str, dict] = {}
            for p in parts:
                for r in common.load_jsonl(p):
                    assert r["pair_id"] not in seen, f"[b1:merge] duplicate pair {r['pair_id']} in {p}"
                    seen[r["pair_id"]] = r
            bad_len = [pid for pid, r in seen.items() if len(r["scores"]) != want.get(pid, -1)]
            missing = sorted(set(want) - set(seen))
            assert not bad_len, f"[b1:merge] {stem}: {len(bad_len)} pairs with the wrong unit count, e.g. {bad_len[:5]}"
            assert not missing, f"[b1:merge] {stem}: {len(missing)} pairs missing, e.g. {missing[:5]}"
            dst = out / f"{stem}.jsonl"
            with open(dst, "w", encoding="utf-8") as fh:
                for r in rows:                      # inventory order, not part order
                    fh.write(json.dumps(seen[r["pair_id"]]) + "\n")
            n_units = sum(len(seen[r["pair_id"]]["scores"]) for r in rows)
            n_null = sum(1 for r in rows for x in seen[r["pair_id"]]["scores"] if x is None)
            print(f"[b1:merge] {dst.name}: {len(seen)} pairs, {n_units} units, {n_null} null, "
                  f"{len(parts)} parts", flush=True)
            report["merged"][stem] = {"n_parts": len(parts), "n_pairs": len(seen), "n_units": n_units,
                                      "n_null": n_null, "parts": [p.name for p in parts]}
    common.dump_json(out / f"b1_merge_{a.units_name}.json", report)


# ----------------------------------------------------------------------------- verify
def do_verify(a) -> None:
    import torch
    rows = load_rows(a)
    prem = premises(rows, "real", a.shuffle_seed)
    allflat = flatten(rows, prem, a.unit_set)
    stride = max(1, len(allflat) // max(1, a.verify_canonical))
    flat = allflat[::stride][:a.verify_canonical]        # spread over the pool, not the first few pairs
    docs = [prem[i] for i, _, _ in flat]
    claims = [c for _, _, c in flat]
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rep = {"n_units": len(flat), "n_units_pool": len(allflat), "stride": stride, "device": dev, "batch": a.batch,
           "units": a.units, "units_name": a.units_name, "split": a.split}
    fast = build("minicheck", a.batch, dev)
    t0 = time.time()
    vf = fast.score(docs, claims)
    rep["fast_seconds"] = time.time() - t0
    rep["fast_units_per_s"] = len(flat) / max(1e-9, rep["fast_seconds"])
    del fast
    torch.cuda.empty_cache() if dev.startswith("cuda") else None
    canon = build("minicheck_canonical", a.batch, dev)
    t0 = time.time()
    vc = canon.score(docs, claims)
    rep["canonical_seconds"] = time.time() - t0
    rep["canonical_units_per_s"] = len(flat) / max(1e-9, rep["canonical_seconds"])
    del canon
    torch.cuda.empty_cache() if dev.startswith("cuda") else None
    d = np.abs(np.asarray(vf) - np.asarray(vc))
    rep["max_abs_diff"] = float(d.max())
    rep["mean_abs_diff"] = float(d.mean())
    rep["n_over_1e-3"] = int((d > 1e-3).sum())
    rep["spearman_like_rank_disagreements"] = int((np.argsort(vf) != np.argsort(vc)).sum())
    rep["speedup"] = rep["canonical_seconds"] / max(1e-9, rep["fast_seconds"])
    rep["equivalent"] = bool(rep["max_abs_diff"] <= a.verify_tol)
    print(f"[b1:verify] n={len(flat)} max|d|={rep['max_abs_diff']:.3e} mean|d|={rep['mean_abs_diff']:.3e} "
          f"fast={rep['fast_units_per_s']:.1f} u/s canonical={rep['canonical_units_per_s']:.2f} u/s "
          f"speedup={rep['speedup']:.1f}x equivalent={rep['equivalent']}", flush=True)
    als = build("alignscore", a.batch, dev)
    t0 = time.time()
    va = als.score(docs, claims)
    rep["alignscore_seconds"] = time.time() - t0
    rep["alignscore_units_per_s"] = len(flat) / max(1e-9, rep["alignscore_seconds"])
    rep["alignscore_mean_support"] = float(np.mean(va))
    rep["alignscore_min"] = float(np.min(va))
    rep["alignscore_max"] = float(np.max(va))
    print(f"[b1:verify] alignscore {rep['alignscore_units_per_s']:.1f} u/s "
          f"support mean {rep['alignscore_mean_support']:.4f} "
          f"range [{rep['alignscore_min']:.4f}, {rep['alignscore_max']:.4f}]", flush=True)
    rep["minicheck_mean_support"] = float(np.mean(vc))
    out = Path(a.out)
    common.dump_json(out / "b1_pilot.json", rep)
    print(f"[b1:verify] written {out / 'b1_pilot.json'}", flush=True)
    if not rep["equivalent"]:
        raise SystemExit(f"[b1:verify] REFUSED: max|d|={rep['max_abs_diff']:.3e} > tol {a.verify_tol}")


# ----------------------------------------------------------------------------- scan
def do_scan(a) -> None:
    import torch
    rows_all = load_rows(a)
    sizes_all = [len(r["units"]) for r in rows_all]
    i_sh, n_sh = (1, 1) if not a.shard else tuple(int(x) for x in a.shard.split("/"))
    assert 1 <= i_sh <= n_sh, f"[b1] bad --shard {a.shard!r}"
    bounds = shard_bounds(sizes_all, n_sh)
    lo, hi = bounds[i_sh - 1], bounds[i_sh]
    prem_all = premises(rows_all, a.candidate_mode, a.shuffle_seed)   # computed on the FULL pool, then sliced
    rows, prem = rows_all[lo:hi], prem_all[lo:hi]
    if a.limit:
        rows, prem = rows[:a.limit], prem[:a.limit]
    flat = flatten(rows, prem, a.unit_set)
    print(f"[b1] units={a.units_name} split={a.split} mode={a.candidate_mode} shard={i_sh}/{n_sh} "
          f"row_offset={lo} rows={len(rows)} units={len(flat)} of {sum(sizes_all)} systems={a.systems}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    docs = [prem[i] for i, _, _ in flat]
    claims = [c for _, _, c in flat]
    manifest = {"units": a.units, "units_name": a.units_name, "split": a.split, "unit_set": a.unit_set,
                "candidate_mode": a.candidate_mode, "shuffle_seed": a.shuffle_seed,
                "shard": f"{i_sh}/{n_sh}", "row_offset": lo, "row_end": hi, "n_rows": len(rows),
                "n_units": len(flat), "n_units_pool": sum(sizes_all), "batch": a.batch, "device": dev,
                "claim_word_cap": CLAIM_WORD_CAP, "systems": a.systems, "timings": {}}

    def emit(name: str, omission: list[float]) -> str:
        per = {r["pair_id"]: [np.nan] * len(r["units"]) for r in rows}
        for (i, j, _), v in zip(flat, omission):
            per[rows[i]["pair_id"]][j] = v
        full = name + MODESUF[a.candidate_mode]
        stem = f"{a.units_name}_{full.replace(':', '_')}"
        # ALWAYS carry the part suffix, n=1 included: --merge finds parts by `<stem>.part*of*.jsonl`, so an
        # unsuffixed single-shard file would be silently invisible to it and the merge would report "no parts".
        fn = out / f"{stem}.part{i_sh}of{n_sh}.jsonl"
        common.write_scores(fn, full, a.units_name, per)
        print(f"[b1] wrote {fn.name}", flush=True)
        return fn.name

    for sysname in a.systems.split(","):
        assert sysname in SYSNAMES, f"[b1] unsupported system {sysname!r}"
        t0 = time.time()
        written = []
        if sysname == "lex":
            written.append(emit("lex:token_recall", [1 - common.token_recall(c, docs[k]) for k, (_, _, c) in enumerate(flat)]))
            written.append(emit("lex:rougeL_recall", [1 - common.rougeL_recall(c, docs[k]) for k, (_, _, c) in enumerate(flat)]))
            el = time.time() - t0
        elif sysname == "minilm":
            from sentence_transformers import SentenceTransformer
            st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
            cand_sents = [common.sentences(p) or [p] for p in prem]
            cand_emb = [st.encode(x, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                        for x in cand_sents]
            claim_emb = st.encode(claims, batch_size=128, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=False)
            written.append(emit("emb:minilm_maxcos",
                                [1 - float((cand_emb[i] @ claim_emb[k]).max()) for k, (i, _, _) in enumerate(flat)]))
            el = time.time() - t0
            del st
        else:
            sc = build(sysname, a.batch, dev)
            t_load = time.time() - t0
            t0 = time.time()
            vals: list[float] = []
            for s in range(0, len(flat), a.chunk):
                vals.extend(sc.score(docs[s:s + a.chunk], claims[s:s + a.chunk]))
                if (s // a.chunk) % 10 == 0:
                    el = time.time() - t0
                    done = min(s + a.chunk, len(flat))
                    print(f"[b1]   {sysname} {done}/{len(flat)} ({el:.0f}s, {done / max(1e-9, el):.1f} u/s)", flush=True)
            el = time.time() - t0
            written.append(emit(SYSNAMES[sysname][0], [1.0 - float(v) for v in vals]))
            manifest["timings"].setdefault(sysname, {})["load_s"] = t_load
            del sc
        manifest["timings"].setdefault(sysname, {}).update(
            {"score_s": el, "units_per_s": len(flat) / max(1e-9, el), "out": written})
        print(f"[b1] {sysname} done in {el:.0f}s ({len(flat) / max(1e-9, el):.1f} u/s)", flush=True)
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
    common.dump_json(out / f"b1_manifest_{a.units_name}_{a.candidate_mode}_{i_sh}of{n_sh}.json", manifest)
    print("[b1] done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit-set", choices=["acu", "m2"], default="m2")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--units", default="", help="units jsonl (overrides --unit-set/--split lookup)")
    ap.add_argument("--units-name", required=True, help="inventory name written into the score files")
    ap.add_argument("--systems", default="minicheck,alignscore")
    ap.add_argument("--candidate-mode", choices=["real", "shuffled", "empty"], default="real")
    ap.add_argument("--shuffle-seed", type=int, default=20260917)
    ap.add_argument("--shard", default="", help="i/n over unit-count-balanced contiguous row blocks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="", help="--merge: directory holding the part files (default <out>/parts)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=512, help="units handed to a scorer per call")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="head truncation INSIDE the shard (pilots only)")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--modes", default="real", help="--merge: comma list of candidate modes to stitch")
    ap.add_argument("--verify-canonical", type=int, default=0,
                    help="run the official MiniCheck path beside the fast one on this many units and refuse "
                         "if they differ by more than --verify-tol")
    ap.add_argument("--verify-tol", type=float, default=1e-3)
    a = ap.parse_args()
    if a.merge:
        do_merge(a)
    elif a.verify_canonical:
        do_verify(a)
    else:
        do_scan(a)


if __name__ == "__main__":
    main()
