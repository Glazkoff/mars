#!/usr/bin/env python3
"""B6 -- end-to-end cost by pipeline stage, and the reuse-amortisation curve.

PREREG (marsc_strengthen_20260918) Wave B / B6.  No bar: a cost measurement cannot pass or fail, the measured
number is the number.

WHAT THE PAPER CURRENTLY HAS, AND WHY IT IS HALF-DONE
-----------------------------------------------------
`research/marsc_20260916/code/mc_cost.py` (block G4, job 4843) produced the only cost artifact in the project.
Read it before reading this file; the list below is the answer to the brief's question "say what it does not
cover", established by reading that source, not inferred:

 1. It times ONE verifier checkpoint per label source.  The deployed emitter seed-averages THREE, so the
    measured `s_per_pair` is one third of the inference cost of the pipeline of record.
 2. It times only `AutoModelForSequenceClassification` at `max_length=512`, i.e. exactly the MARS-C verifier
    family.  It cannot time the off-the-shelf comparators (SummaC-ZS's sentence x sentence NLI grid,
    MiniCheck's Flan-T5 generator, the windowed RoBERTa-MNLI scorer, the MiniLM bi-encoder, the lexical
    counters), and cannot time any generative judge.  The paper's comparison of MARS-C against off-the-shelf
    verifiers is therefore a quality comparison with no cost axis.
 3. Model load is excluded (`from_pretrained` happens before `t0`) -- correct for a throughput number, but it
    means the figure is not comparable to the inventory figure, which is allocation wall-clock (below).
 4. The inventory cost is not measured at all.  It is HARVESTED: `mc_g4_cost.sbatch` runs
    `sacct -j 4704 -X -o Elapsed` and sums the *allocation* elapsed time of the eight decomposition array
    tasks.  That includes model load, tokenizer build, task-file IO, checkpoint flushes and any scheduler
    slack, and it is the number the paper's headline 6.637 s/document is built from.  This block re-times the
    same decomposition as ACTIVE GENERATION on the current node and reports both, labelled.
 5. The spaCy entity half of the `gemma+ent` inventory is never priced.  `gemma+ent` = LLM propositions PLUS a
    spaCy entity pass over the same source (`m32_inventory.py`), and only the LLM half appears in the G4 table.
 6. The emission rule is not priced.  `mc_diversity.py` re-segments every source into sentences and maps every
    unit onto a sentence -- per-pair CPU work that a deployment pays on every summary.
 7. The seed aggregation is not priced.
 8. `units_per_pair` is taken from the first 200 pairs of the validation inventory, which carry 165.5 Gemma+ent
    units per pair against a split-wide 94.  Every per-pair second in the G4 table is therefore inflated by
    ~1.76x relative to the workload the paper evaluates.  This block measures seconds per UNIT (a rate that
    does not depend on the sample's unit density) and prices the curve at SPLIT-WIDE unit counts.
 9. `pairs_per_document_in_sample` is likewise 11.76 in that 200-pair window; split-wide RoSE is lower.  The
    amortisation denominator is re-derived here from the full inventory files.
10. There is no evidence-handling stage in the G4 table because there is none in the pipeline: the MARS-C
    verifier's premise is the candidate summary alone, the source is not an input at inference
    (`marsc_method.tex`).  That is a structural cost property and is reported as a measured zero, not omitted.
    The off-the-shelf comparators are scored the same way (document = candidate summary), so the same holds
    for them; what they do pay, and MARS-C does not, is claim RENDERING (`common.render_claim` walks the
    source to turn an extracted unit into a sentence), which is timed here as its own stage.

STAGES TIMED HERE
-----------------
  decomposition   source -> units.  Gemma-4-31B (the inventory of record), the distilled LongT5 segmenter,
                  and the spaCy entity pass.  Per DOCUMENT; paid once per source and amortised over its
                  summaries.
  evidence        premise construction.  Zero for MARS-C by construction (asserted, not assumed: the premise
                  is the candidate clipped to 380 words).  Claim rendering timed for the comparator family.
  verification    cross-encoder forward passes, per UNIT, per seed.  Per (summary, unit) cell.
  aggregation     the seed mean over three score vectors.  Per pair.
  emission        the one-unit-per-sentence rule (sentence re-segmentation + unit->sentence map + the ranked
                  demotion walk) and the top-k selection.  Per pair.

THE CURVE
---------
    cost per summary (s) = decomposition_seconds_per_document / s  +  per-summary seconds
with s = summaries per source from 1 to 12.  s = 1 is the deployment price of one new summary of one new
source -- the number the 2026-09-18 correction added; s = 11.8 is RoSE's own reuse factor, which is a property
of the BENCHMARK and not of any deployment, and is marked as such on the curve.

Declared comparator constants (given by the campaign brief as already measured; cited, never re-derived here)
are carried in DECLARED with their provenance string so the curve never silently mixes a measured second with
a quoted one -- every arm in the output carries `measured` or `declared` per component.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO), str(REPO / "research" / "mars2_gates_20260916" / "code"),
           str(REPO / "research" / "mars3_20260916" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common  # noqa: E402

# --------------------------------------------------------------------------------------------------------
# Constants quoted from the campaign brief's measured-cost table.  NOT re-derived here.  Anything that enters
# the curve from this dict is labelled `declared` in the output.
# --------------------------------------------------------------------------------------------------------
DECLARED = {
    "llm_cell_judge_s_per_cell_low": (0.168, "campaign brief, forced-choice coverage judge, 0.168-0.203 GPU-s/cell"),
    "llm_cell_judge_s_per_cell_high": (0.203, "campaign brief, forced-choice coverage judge, 0.168-0.203 GPU-s/cell"),
    "generative_judge_s_per_pair_gemma": (4.2, "campaign brief, generative 400-token judge, Gemma, GPU-s/pair"),
    "generative_judge_s_per_pair_qwen": (2.1, "campaign brief, generative 400-token judge, Qwen3.8-27B, GPU-s/pair"),
    "mars2_recovery_s_per_pair": (0.82, "appendix_marsc.tex cost paragraph, MARS-2 recovery, historical measurement"),
    "mars2_tagger_s_per_pair": (0.013, "appendix_marsc.tex cost paragraph, released one-pass tagger, historical"),
    "g4_decomposition_s_per_document_allocation": (6.637462235649547, "results/marsc/g4/cost_table.json, sacct Elapsed of 8 array tasks / 1986 documents (ALLOCATION time)"),
    "g4_distill_s_per_document_allocation": (0.2366565961732125, "results/marsc/g4/cost_table.json, sacct Elapsed of the distil job / 1986 documents (ALLOCATION time, TRAIN+PREDICT)"),
    "minicheck_units_per_s_declared": (1.36, "campaign brief, MiniCheck ~1.36 units/s on one H200"),
}

N_SEEDS_OF_RECORD = 3          # the deployed emitter seed-averages three checkpoints
CANDIDATE_WORD_CLIP = 380      # mc_score_units.py premise contract
MAX_LENGTH = 512               # mc_score_units.py / mc_cost.py tokenizer contract
DIV_CAP = 1                    # mc_diversity.py registered default
TOPK = 10                      # the emission budget of record


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=False))


# ---------------------------------------------------------------------------------------------------------
# population: the split-wide unit and document counts the curve is priced at
# ---------------------------------------------------------------------------------------------------------
def population(inventories: dict[str, str]) -> dict:
    out = {}
    for name, path in inventories.items():
        p = Path(path)
        if not p.exists():
            out[name] = {"error": f"missing {path}"}
            continue
        rows = [r for r in common.load_jsonl(p) if r.get("units")]
        units = sum(len(r["units"]) for r in rows)
        docs = {common.doc_key(r["source"]) for r in rows}
        out[name] = {
            "file": str(p), "pairs_with_units": len(rows), "units_total": units,
            "units_per_pair": units / max(1, len(rows)),
            "documents": len(docs), "pairs_per_document": len(rows) / max(1, len(docs)),
            "units_per_pair_first200": sum(len(r["units"]) for r in rows[:200]) / max(1, len(rows[:200])),
            "pairs_per_document_first200": len(rows[:200]) / max(1, len({common.doc_key(r["source"]) for r in rows[:200]})),
        }
        print(f"[b6] population {name}: {out[name]['pairs_with_units']} pairs, {out[name]['units_per_pair']:.2f} units/pair "
              f"(first-200 window {out[name]['units_per_pair_first200']:.2f}), "
              f"{out[name]['pairs_per_document']:.3f} pairs/document (first-200 window {out[name]['pairs_per_document_first200']:.3f})", flush=True)
    return out


# ---------------------------------------------------------------------------------------------------------
# verification + aggregation
# ---------------------------------------------------------------------------------------------------------
def time_verification(inv_rows: dict[str, list[dict]], ckpts: dict[str, list[str]], batch: int, dev: str) -> tuple[dict, dict]:
    """Time each MARS-C verifier checkpoint over each inventory sample; keep the humanfact score vectors so
    the seed aggregation can be timed on real arrays rather than on synthetic ones."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = True
    res: dict = {}
    keep: dict[str, list[np.ndarray]] = {}
    for label_source, dirs in ckpts.items():
        for d in dirs:
            m = re.search(r"seed(\d+)", str(d))          # the mc_score_units.py seed convention
            seed = m.group(1) if m else Path(d).name
            if not Path(d, "config.json").exists():
                res[f"{label_source}|seed{seed}"] = {"error": f"no config.json under {d}"}
                print(f"[b6] SKIP verifier {d}: no config.json", flush=True)
                continue
            t_load = time.time()
            tok = AutoTokenizer.from_pretrained(d)
            model = AutoModelForSequenceClassification.from_pretrained(d).to(dev).eval()
            load_s = time.time() - t_load
            for inv, rows in inv_rows.items():
                prem = [" ".join(r["candidate"].split()[:CANDIDATE_WORD_CLIP]) for r in rows]
                flat = [(i, u["text"]) for i, r in enumerate(rows) for u in r["units"]]
                vals: list[float] = []
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                with torch.inference_mode():
                    for s in range(0, len(flat), batch):
                        ch = flat[s:s + batch]
                        enc = tok([prem[i] for i, _ in ch], [f for _, f in ch], truncation="longest_first",
                                  max_length=MAX_LENGTH, padding=True, return_tensors="pt").to(dev)
                        lg = model(**enc).logits
                        vals.extend(torch.sigmoid(lg[:, 0]).float().cpu().numpy().tolist() if lg.shape[-1] == 1
                                    else torch.softmax(lg, -1)[:, 0].float().cpu().numpy().tolist())
                if dev == "cuda":
                    torch.cuda.synchronize()
                dt = time.time() - t0
                key = f"{label_source}|{inv}|seed{seed}"
                res[key] = {"pairs": len(rows), "units": len(flat), "units_per_pair": len(flat) / max(1, len(rows)),
                            "seconds": dt, "s_per_unit": dt / max(1, len(flat)), "s_per_pair_in_sample": dt / max(1, len(rows)),
                            "model_load_seconds": load_s, "checkpoint": d}
                print(f"[b6] verify {key}: {1000 * res[key]['s_per_unit']:.3f} ms/unit over {len(flat)} units "
                      f"(load {load_s:.1f}s)", flush=True)
                if label_source == "humanfact":
                    keep.setdefault(inv, []).append(np.asarray(vals, dtype=float))
            del model
            if dev == "cuda":
                torch.cuda.empty_cache()
    agg: dict = {}
    for inv, vecs in keep.items():
        if len(vecs) < 2:
            agg[inv] = {"error": f"only {len(vecs)} seed vector(s) kept; aggregation not timed"}
            continue
        reps = 20
        t0 = time.time()
        for _ in range(reps):
            _ = np.mean(np.stack(vecs, axis=0), axis=0)
        dt = (time.time() - t0) / reps
        n_units = len(vecs[0]); n_pairs = len(inv_rows[inv])
        agg[inv] = {"seeds": len(vecs), "units": n_units, "pairs": n_pairs, "repeats": reps,
                    "seconds_per_repeat": dt, "s_per_unit": dt / max(1, n_units), "s_per_pair_in_sample": dt / max(1, n_pairs)}
        print(f"[b6] aggregate {inv}: {1e6 * agg[inv]['s_per_unit']:.3f} us/unit over {len(vecs)} seeds", flush=True)
    return res, agg


# ---------------------------------------------------------------------------------------------------------
# emission (the mc_diversity.py rule, logic copied rather than imported so nothing shared is touched)
# ---------------------------------------------------------------------------------------------------------
def _sentence_index(source: str, start: int, spans: list[tuple[int, int]]) -> int:
    for i, (a, b) in enumerate(spans):
        if a <= start < b:
            return i
    return -1


def time_emission(rows: list[dict], rng_seed: int = 20260918) -> dict:
    """Per-pair cost of the emission rule: re-segment the source, map units onto sentences, walk the ranking
    applying the -5.0 demotion, then take the top-k.  Scores are drawn, not computed: the rule's cost does not
    depend on what produced the scores, and drawing them keeps this stage free of a GPU."""
    rng = np.random.default_rng(rng_seed)
    scores = [rng.random(len(r["units"])) for r in rows]
    t0 = time.time()
    n_units = 0
    for r, v in zip(rows, scores):
        spans = common.sentence_spans(r["source"])
        n = len(r["source"])
        si = [_sentence_index(r["source"], int(u.get("start", n // 2)), spans) for u in r["units"]]
        new = v.copy(); count: dict[int, int] = {}
        order = np.argsort(-np.nan_to_num(v, nan=-np.inf))
        for i in order:
            if not np.isfinite(v[i]) or v[i] <= -9.0:
                continue
            s = si[i]
            if count.get(s, 0) >= DIV_CAP:
                new[i] = v[i] - 5.0
            else:
                count[s] = count.get(s, 0) + 1
        _ = np.argsort(-new)[:TOPK]
        n_units += len(v)
    dt = time.time() - t0
    out = {"pairs": len(rows), "units": n_units, "seconds": dt, "s_per_pair": dt / max(1, len(rows)),
           "s_per_unit": dt / max(1, n_units), "rule": "mc_diversity.py cap=1 then top-10", "device": "cpu"}
    print(f"[b6] emission: {1000 * out['s_per_pair']:.3f} ms/pair over {len(rows)} pairs ({n_units} units)", flush=True)
    return out


def time_claim_rendering(rows: list[dict]) -> dict:
    """The premise/claim construction the off-the-shelf comparators pay and MARS-C does not: `render_claim`
    walks the source to turn an extracted unit into a standalone sentence.  Also times MARS-C's own premise
    construction (a 380-word clip of the candidate) so the 'evidence handling' row is a measurement."""
    t0 = time.time(); n = 0
    for r in rows:
        for u in r["units"]:
            _ = common.render_claim(r["source"], u); n += 1
    dt_render = time.time() - t0
    t0 = time.time()
    for r in rows:
        _ = " ".join(r["candidate"].split()[:CANDIDATE_WORD_CLIP])
    dt_prem = time.time() - t0
    out = {"claim_rendering": {"units": n, "seconds": dt_render, "s_per_unit": dt_render / max(1, n),
                               "who_pays": "off-the-shelf comparators over m2 unit sets (b21_zeroshot render_claim)"},
           "marsc_premise_construction": {"pairs": len(rows), "seconds": dt_prem, "s_per_pair": dt_prem / max(1, len(rows)),
                                          "who_pays": "MARS-C (candidate clipped to 380 words; the source is NOT an input)"},
           "source_retrieval_seconds": 0.0,
           "note": ("no stage of MARS-C reads the source at inference time -- the verifier's premise is the candidate "
                    "summary alone (marsc_method.tex).  The off-the-shelf comparators are scored the same way "
                    "(document = candidate summary), so neither family pays an evidence-retrieval cost; the asymmetry "
                    "is claim rendering, which only the comparator family pays.")}
    print(f"[b6] evidence: claim rendering {1e6 * out['claim_rendering']['s_per_unit']:.1f} us/unit; "
          f"MARS-C premise {1e6 * out['marsc_premise_construction']['s_per_pair']:.1f} us/pair; source retrieval 0 by construction", flush=True)
    return out


# ---------------------------------------------------------------------------------------------------------
# off-the-shelf comparator verification
# ---------------------------------------------------------------------------------------------------------
def time_offshelf(rows: list[dict], systems: list[str], batch: int, dev: str, cap: int, slow_cap: int) -> dict:
    """Per-unit cost of each off-the-shelf verifier under b21_zeroshot's contract (document = candidate
    summary, claim = rendered unit), logic copied here so the cap can be applied per system and a failing
    dependency records a failure instead of killing the job."""
    ssc = common._load("strengthen_score_cells")
    flat = [(i, " ".join(common.render_claim(r["source"], u).split()[:96]))
            for i, r in enumerate(rows) for u in r["units"]]
    res: dict = {}
    for name in systems:
        limit = slow_cap if name in ("minicheck", "alignscore", "bartscore") else cap
        sub = flat[:limit]
        try:
            if name.startswith("lex"):
                t0 = time.time()
                _ = [1 - common.token_recall(c, rows[i]["candidate"]) for i, c in sub]
                dt1 = time.time() - t0
                t0 = time.time()
                _ = [1 - common.rougeL_recall(c, rows[i]["candidate"]) for i, c in sub]
                dt2 = time.time() - t0
                res["lex:token_recall"] = {"units": len(sub), "seconds": dt1, "s_per_unit": dt1 / max(1, len(sub)), "device": "cpu"}
                res["lex:rougeL_recall"] = {"units": len(sub), "seconds": dt2, "s_per_unit": dt2 / max(1, len(sub)), "device": "cpu"}
            elif name == "minilm":
                import torch  # noqa: F401
                from sentence_transformers import SentenceTransformer
                t_load = time.time()
                st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
                load_s = time.time() - t_load
                idx = sorted({i for i, _ in sub})
                t0 = time.time()
                cand_emb = {i: st.encode(common.sentences(rows[i]["candidate"]) or [rows[i]["candidate"]],
                                         convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False) for i in idx}
                claim_emb = st.encode([c for _, c in sub], batch_size=128, convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=False)
                _ = [1 - float((cand_emb[i] @ claim_emb[k]).max()) for k, (i, _) in enumerate(sub)]
                dt = time.time() - t0
                res["emb:minilm_maxcos"] = {"units": len(sub), "seconds": dt, "s_per_unit": dt / max(1, len(sub)),
                                            "model_load_seconds": load_s, "device": dev}
                del st
            else:
                t_load = time.time()
                sc = ssc.build_scorer(name, batch, dev)
                load_s = time.time() - t_load
                t0 = time.time()
                vals: list[float] = []
                for s in range(0, len(sub), 256):
                    ch = sub[s:s + 256]
                    vals.extend(sc.score([rows[i]["candidate"] for i, _ in ch], [c for _, c in ch]))
                dt = time.time() - t0
                res[f"nli:{name}"] = {"units": len(sub), "seconds": dt, "s_per_unit": dt / max(1, len(sub)),
                                      "units_per_s": len(sub) / max(1e-9, dt), "model_load_seconds": load_s, "device": dev}
                del sc
            for k in [k for k in res if k.endswith(name) or name in k]:
                if "s_per_unit" in res[k]:
                    print(f"[b6] offshelf {k}: {1000 * res[k]['s_per_unit']:.3f} ms/unit over {res[k]['units']} units", flush=True)
        except Exception as exc:                                   # a missing dependency is a RESULT, not a crash
            res[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[b6] offshelf {name} FAILED: {type(exc).__name__}: {exc}", flush=True)
        try:
            import torch
            if dev == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
    return res


# ---------------------------------------------------------------------------------------------------------
# decomposition
# ---------------------------------------------------------------------------------------------------------
def pick_documents(inv_path: Path, n_docs: int) -> list[dict]:
    seen: dict[str, dict] = {}
    for r in common.load_jsonl(inv_path):
        k = common.doc_key(r["source"])
        if k not in seen:
            seen[k] = {"doc_key": k, "source": r["source"]}
        if len(seen) >= n_docs:
            break
    return list(seen.values())


def sentence_tasks(docs: list[dict], min_words: int = 4) -> list[dict]:
    """The m32_decompose.build_tasks contract: one task per source sentence of >= 4 words."""
    out = []
    for d in docs:
        for i, (s, e) in enumerate(common.sentence_spans(d["source"])):
            sent = d["source"][s:e].strip()
            if len(sent.split()) < min_words:
                continue
            out.append({"doc_key": d["doc_key"], "start": s, "end": e, "sentence": sent})
    return out


def time_spacy_entities(docs: list[dict]) -> dict:
    try:
        from mars_v2.units import get_extractor
        ex = get_extractor("entity")
        _ = ex(docs[0]["source"][:2000])                            # warm-up: model load is not the rate
        t0 = time.time(); n = 0
        for d in docs:
            n += len(ex(d["source"]))
        dt = time.time() - t0
        out = {"documents": len(docs), "units": n, "seconds": dt, "s_per_document": dt / max(1, len(docs)),
               "units_per_document": n / max(1, len(docs)), "device": "cpu",
               "role": "the '+ent' half of the gemma+ent inventory, never priced in the G4 table"}
        print(f"[b6] spacy entities: {out['s_per_document']:.4f} s/document over {len(docs)} documents", flush=True)
        return out
    except Exception as exc:
        print(f"[b6] spacy entities FAILED: {type(exc).__name__}: {exc}", flush=True)
        return {"error": f"{type(exc).__name__}: {exc}"}


def time_distilled(tasks: list[dict], n_docs: int, seg_dir: str, batch: int, dev: str) -> dict:
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        t_load = time.time()
        tok = AutoTokenizer.from_pretrained(seg_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(seg_dir).to(dev).eval()
        load_s = time.time() - t_load
        amp = dev.startswith("cuda")
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time(); n_props = 0
        with torch.inference_mode():
            for lo in range(0, len(tasks), batch):
                ch = tasks[lo:lo + batch]
                enc = tok([f"propositions: {r['sentence']}" for r in ch], truncation=True, max_length=256,
                          padding=True, return_tensors="pt").to(dev)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                    ids = model.generate(**enc, num_beams=4, max_new_tokens=192, no_repeat_ngram_size=3)
                for t in tok.batch_decode(ids, skip_special_tokens=True):
                    n_props += len([x for x in t.split("|") if x.strip() and x.strip().upper() != "NONE"])
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
        out = {"documents": n_docs, "sentences": len(tasks), "propositions": n_props, "seconds": dt,
               "s_per_document": dt / max(1, n_docs), "s_per_sentence": dt / max(1, len(tasks)),
               "model_load_seconds": load_s, "device": dev, "decoding": "num_beams=4, max_new_tokens=192, no_repeat_ngram_size=3",
               "measurement": "ACTIVE GENERATION on the current node, model load excluded"}
        print(f"[b6] distilled segmenter: {out['s_per_document']:.4f} s/document, {out['s_per_sentence']:.4f} s/sentence "
              f"(load {load_s:.1f}s)", flush=True)
        return out
    except Exception as exc:
        print(f"[b6] distilled segmenter FAILED: {type(exc).__name__}: {exc}", flush=True)
        return {"error": f"{type(exc).__name__}: {exc}"}


def time_gemma(tasks: list[dict], n_docs: int, snapshot: str, batch: int, max_new_tokens: int, dev: str) -> dict:
    try:
        import torch
        import m32_decompose as m32d
        judge = m32d.judge
        t_load = time.time()
        tok, model = judge.load_model(snapshot, dev)
        load_s = time.time() - t_load
        probe = judge.apply_chat(tok, m32d.PROMPT.format(sentence="The company reported revenue of 1.2 billion dollars in 2023."))
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time(); n_props = 0; n_blank = 0
        with torch.inference_mode():
            for s in range(0, len(tasks), batch):
                ch = tasks[s:s + batch]
                texts = [judge.apply_chat(tok, m32d.PROMPT.format(sentence=t["sentence"])) for t in ch]
                enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(dev)
                ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
                for o in tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
                    n_props += len(m32d.parse(o)); n_blank += int(not o.strip())
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
        out = {"documents": n_docs, "sentences": len(tasks), "propositions": n_props, "blank_outputs": n_blank,
               "seconds": dt, "s_per_document": dt / max(1, n_docs), "s_per_sentence": dt / max(1, len(tasks)),
               "model_load_seconds": load_s, "snapshot": snapshot, "device": dev,
               "decoding": f"greedy, batch={batch}, max_new_tokens={max_new_tokens}, chat template, thinking disabled",
               "chat_prompt_tail": probe[-40:],
               "measurement": "ACTIVE GENERATION on the current node, model load excluded",
               "compare_to": DECLARED["g4_decomposition_s_per_document_allocation"][1]}
        print(f"[b6] gemma decomposition: {out['s_per_document']:.3f} s/document ACTIVE vs "
              f"{DECLARED['g4_decomposition_s_per_document_allocation'][0]:.3f} s/document recorded (allocation); "
              f"load {load_s:.1f}s", flush=True)
        return out
    except Exception as exc:
        print(f"[b6] gemma decomposition FAILED: {type(exc).__name__}: {exc}", flush=True)
        return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------------------------------------
# the curve
# ---------------------------------------------------------------------------------------------------------
def _get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def build_curve(st: dict, grid: list[int], spd: float | None = None) -> dict:
    """cost per summary (s summaries per source) = decomposition_per_document / s + per-summary seconds.

    Every component carries measured|declared.  A component that could not be measured falls back to the
    recorded G4 number and is labelled `declared`; nothing is silently dropped."""
    pop = st.get("population", {})
    stages = st.get("stages", {})
    ver = _get(stages, "verification", default={}) or {}
    off = _get(stages, "offshelf", default={}) or {}
    dec = _get(stages, "decomposition", default={}) or {}
    emit = _get(stages, "emission", default={}) or {}
    agg = _get(stages, "aggregation", default={}) or {}
    ev = _get(stages, "evidence", default={}) or {}

    def upp(inv: str) -> tuple[float, str]:
        v = _get(pop, inv, "units_per_pair")
        if v is not None:
            return float(v), "measured split-wide"
        return (165.54 if inv == "gemma+ent" else 69.92), "declared (G4 first-200 window)"

    def ppd(inv: str) -> tuple[float, str]:
        v = _get(pop, inv, "pairs_per_document")
        return (float(v), "measured split-wide") if v is not None else (11.7647, "declared (G4 first-200 window)")

    def emit_s() -> tuple[float, str]:
        v = emit.get("s_per_pair")
        return (float(v), "measured") if v is not None else (0.0, "not measured -> 0 (understates MARS-C)")

    def agg_s(inv: str) -> tuple[float, str]:
        v = _get(agg, inv, "s_per_unit")
        return (float(v), "measured") if v is not None else (0.0, "not measured -> 0")

    def prem_s() -> tuple[float, str]:
        v = _get(ev, "marsc_premise_construction", "s_per_pair")
        return (float(v), "measured") if v is not None else (0.0, "not measured -> 0")

    def render_s() -> tuple[float, str]:
        v = _get(ev, "claim_rendering", "s_per_unit")
        return (float(v), "measured") if v is not None else (0.0, "not measured -> 0")

    def decomp(inv: str) -> tuple[float, str, dict]:
        """Seconds per DOCUMENT for the inventory build, active where we measured it."""
        comp: dict = {}

        def per_doc(stage: dict, fallback_key: str) -> tuple[float, str]:
            """Seconds per document for a generator stage.  Seconds per SENTENCE is the invariant rate; the
            timing sample's own documents are not the corpus's, so when the corpus sentences-per-document is
            supplied the rate is re-priced at it and the sample rate is reported beside it."""
            if spd and "s_per_sentence" in stage:
                return (stage["s_per_sentence"] * spd,
                        f"measured active, {stage['s_per_sentence']:.5f} s/sentence x {spd:.4f} corpus sentences/document "
                        f"(sample's own rate was {stage.get('s_per_document', float('nan')):.4f} s/document over "
                        f"{stage.get('sentences', 0) / max(1, stage.get('documents', 1)):.2f} sentences/document)")
            if "s_per_document" in stage:
                return stage["s_per_document"], "measured active, at the timing sample's own document length"
            return DECLARED[fallback_key][0], "declared allocation"

        if inv == "gemma+ent":
            comp["gemma_llm"] = per_doc(dec.get("gemma", {}), "g4_decomposition_s_per_document_allocation")
            sp = dec.get("spacy_entity", {})
            comp["spacy_entity"] = ((sp["s_per_document"], "measured at the timing sample's own document length")
                                    if "s_per_document" in sp
                                    else (0.0, "not measured -> 0 (understates gemma+ent)"))
        else:
            comp["distilled_segmenter"] = per_doc(dec.get("distilled", {}), "g4_distill_s_per_document_allocation")
        total = sum(v for v, _ in comp.values())
        prov = "; ".join(f"{k}={v:.4f}s ({p})" for k, (v, p) in comp.items())
        return total, prov, {k: {"s_per_document": v, "provenance": p} for k, (v, p) in comp.items()}

    def verif_rate(label_source: str, inv: str) -> tuple[float, str, int]:
        keys = [k for k in ver if k.startswith(f"{label_source}|{inv}|") and "s_per_unit" in ver[k]]
        if keys:
            vals = [ver[k]["s_per_unit"] for k in keys]
            return float(np.mean(vals)), f"measured, mean over {len(keys)} checkpoint(s)", len(keys)
        return (0.00083, "declared (campaign brief, 0.00083 GPU-s/unit)", 0)

    arms: dict = {}

    def add(name: str, inv: str | None, per_summary: float, per_doc: float, components: dict, endpoint_note: str = "") -> None:
        arms[name] = {
            "inventory": inv,
            "decomposition_s_per_document": per_doc,
            "per_summary_seconds": per_summary,
            "components": components,
            "note": endpoint_note,
            "cost_per_summary": {str(s): per_doc / s + per_summary for s in grid},
        }

    for inv in ("gemma+ent", "distilled"):
        u, u_prov = upp(inv)
        d_tot, d_prov, d_comp = decomp(inv)
        e_s, e_prov = emit_s()
        a_s, a_prov = agg_s(inv)
        p_s, p_prov = prem_s()
        for n_seeds in (N_SEEDS_OF_RECORD, 1):
            r, r_prov, n_ck = verif_rate("humanfact", inv)
            per = n_seeds * r * u + a_s * u * (1 if n_seeds > 1 else 0) + e_s + p_s
            add(f"MARS-C humanfact / {inv} / {n_seeds}-seed", inv, per, d_tot,
                {"units_per_pair": {"value": u, "provenance": u_prov},
                 "verification_s_per_unit": {"value": r, "provenance": r_prov, "checkpoints_timed": n_ck},
                 "verification_passes": {"value": n_seeds, "provenance": "pipeline of record seed-averages 3"},
                 "aggregation_s_per_unit": {"value": a_s if n_seeds > 1 else 0.0, "provenance": a_prov},
                 "emission_s_per_pair": {"value": e_s, "provenance": e_prov},
                 "premise_s_per_pair": {"value": p_s, "provenance": p_prov},
                 "decomposition": d_comp, "decomposition_provenance": d_prov},
                "the pipeline of record is the 3-seed arm")
        # off-the-shelf comparators over the same frozen inventory and emission rule
        rs, rs_prov = render_s()
        for k, v in sorted(off.items()):
            if "s_per_unit" not in v:
                continue
            per = v["s_per_unit"] * u + rs * u + e_s
            add(f"off-the-shelf {k} / {inv}", inv, per, d_tot,
                {"units_per_pair": {"value": u, "provenance": u_prov},
                 "verification_s_per_unit": {"value": v["s_per_unit"], "provenance": f"measured over {v['units']} units"},
                 "verification_passes": {"value": 1, "provenance": "single model, no seed ensemble"},
                 "claim_rendering_s_per_unit": {"value": rs, "provenance": rs_prov},
                 "emission_s_per_pair": {"value": e_s, "provenance": e_prov},
                 "decomposition": d_comp, "decomposition_provenance": d_prov},
                "timed on the gemma+ent sample; priced at this inventory's split-wide unit count")
        # the LLM cell judge over the same inventory
        for tag, key in (("low", "llm_cell_judge_s_per_cell_low"), ("high", "llm_cell_judge_s_per_cell_high")):
            c, prov = DECLARED[key]
            add(f"LLM cell judge ({tag}) / {inv}", inv, c * u + e_s, d_tot,
                {"units_per_pair": {"value": u, "provenance": u_prov},
                 "verification_s_per_cell": {"value": c, "provenance": prov},
                 "emission_s_per_pair": {"value": e_s, "provenance": e_prov},
                 "decomposition": d_comp, "decomposition_provenance": d_prov},
                "judges every inventory cell; needs the same inventory, so amortises the same way")

    for key, label in (("generative_judge_s_per_pair_gemma", "generative 400-token judge (Gemma)"),
                       ("generative_judge_s_per_pair_qwen", "generative 400-token judge (Qwen3.8-27B)"),
                       ("mars2_recovery_s_per_pair", "MARS-2 recovery (historical)"),
                       ("mars2_tagger_s_per_pair", "MARS-2 released one-pass tagger (historical)")):
        v, prov = DECLARED[key]
        add(label, None, v, 0.0,
            {"per_pair_seconds": {"value": v, "provenance": prov}},
            "inventory-free: flat in summaries per source")

    markers = {
        "one_new_summary_per_new_source": 1,
        "rose_summaries_per_source_split_wide": _get(pop, "gemma+ent", "pairs_per_document"),
        "rose_summaries_per_source_g4_window": 11.7647,
    }
    return {"grid": grid, "arms": arms, "markers": markers,
            "definition": "cost_per_summary(s) = decomposition_s_per_document / s + per_summary_seconds",
            "built_utc": now()}


def g4_audit(st: dict, a, spd: float | None) -> dict:
    """Audit the two published decomposition seconds-per-document against what this block measured.

    Both published figures divide a Slurm ALLOCATION elapsed time by 1,986 documents.  For the Gemma
    decomposition that denominator is right (the array did decompose every document).  For the distilled
    segmenter it is NOT: `m32_distill.py` trains on the train-split sentences and predicts only the
    validation+test sentences, so the elapsed time covers a one-off TRAINING cost and the per-document
    divisor counts documents the segmenter never predicted for."""
    out: dict = {}
    dec = _get(st, "stages", "decomposition", default={}) or {}
    for name, key, stage in (("gemma", "g4_decomposition_s_per_document_allocation", dec.get("gemma", {})),
                             ("distilled", "g4_distill_s_per_document_allocation", dec.get("distilled", {}))):
        pub, prov = DECLARED[key]
        row = {"published_s_per_document": pub, "published_provenance": prov}
        if "s_per_sentence" in stage:
            row["measured_s_per_sentence_active"] = stage["s_per_sentence"]
            if spd:
                row["measured_s_per_document_at_corpus_length"] = stage["s_per_sentence"] * spd
                row["ratio_measured_over_published"] = stage["s_per_sentence"] * spd / max(1e-9, pub)
        out[name] = row
    if a.distill_result and Path(a.distill_result).exists():
        d = json.loads(Path(a.distill_result).read_text())
        pred = d.get("predicted_sentences"); train = d.get("train_sentences"); secs = d.get("seconds")
        out["distilled"]["distil_job_record"] = {
            "file": a.distill_result, "train_sentences": train, "predicted_sentences": pred, "job_seconds": secs,
            "finding": ("the published 0.2367 s/document divides this job's whole elapsed time -- which covers "
                        f"TRAINING on {train} sentences as well as PREDICTION on {pred} -- by 1,986 documents, "
                        "while the segmenter predicted only for the validation+test documents. It therefore "
                        "charges a one-off training cost to inference and divides by a document count the "
                        "segmenter never ran on, and it UNDERSTATES the per-document inference cost of the "
                        "cheap arm the paper promotes."),
        }
    return out


def print_table(curve: dict) -> None:
    grid = curve["grid"]
    head = "s(summaries/source) " + " ".join(f"{s:>9d}" for s in grid)
    print("[b6] " + "-" * len(head), flush=True)
    print("[b6] " + head, flush=True)
    for name, a in curve["arms"].items():
        row = " ".join(f"{a['cost_per_summary'][str(s)]:>9.3f}" for s in grid)
        print(f"[b6] {name:<46s} {row}", flush=True)
    print("[b6] " + "-" * len(head), flush=True)


# ---------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["timings", "curve"], required=True)
    ap.add_argument("--which", default="all", help="comma list of cpu,gpu (timings only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timings-name", default="stage_times.json")
    ap.add_argument("--merge", nargs="*", default=[], help="curve: extra stage_times json files to merge (later wins)")
    ap.add_argument("--curve-name", default="amortisation_curve.json")
    ap.add_argument("--inv-gemma", default="/home/user/results/mars3/m32/units_gemma+ent_validation.jsonl")
    ap.add_argument("--inv-distilled", default="/home/user/results/mars3/m32/units_distilled_validation.jsonl")
    ap.add_argument("--ckpt-human", default="/home/user/results/marsc/e1/mc-A-lam1-crossenc-inv_seed{seed}/model")
    ap.add_argument("--ckpt-judge", default="/home/user/results/mars2_gates/b21/xenc_seed{seed}")
    ap.add_argument("--seeds", default="20260906,20260907,20260908")
    ap.add_argument("--segmenter", default="/home/user/results/mars3/m32/segmenter")
    ap.add_argument("--gemma-snapshot", default="")
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--n-docs", type=int, default=50)
    ap.add_argument("--n-docs-gemma", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--gemma-batch", type=int, default=12)
    ap.add_argument("--gemma-max-new-tokens", type=int, default=160)
    ap.add_argument("--offshelf", default="lex,minilm,nli,summac_zs,minicheck,alignscore")
    ap.add_argument("--offshelf-unit-cap", type=int, default=6000)
    ap.add_argument("--slow-unit-cap", type=int, default=900)
    ap.add_argument("--grid", default="1,2,3,4,5,6,7,8,9,10,11,12")
    ap.add_argument("--tasks", default="", help="curve: the frozen decompose_tasks.jsonl. When given, the "
                                                "decomposition stages are priced at their measured seconds-per-SENTENCE "
                                                "times the CORPUS sentences-per-document, instead of at the timing "
                                                "sample's own document length")
    ap.add_argument("--distill-result", default="", help="curve: results/mars3/m32/distill_result.json, for the "
                                                          "audit of the published distilled seconds-per-document")
    ap.add_argument("--skip-gemma", action="store_true")
    ap.add_argument("--offshelf-only", action="store_true",
                    help="time only the off-the-shelf verifiers (no MARS-C verification, no decomposition)")
    ap.add_argument("--offshelf-inventories", default="gemma+ent",
                    help="comma list of inventories to time the off-the-shelf verifiers on; the first keeps the "
                         "stage name 'offshelf', others are written as 'offshelf_<name>'")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    grid = [int(x) for x in a.grid.split(",") if x.strip()]

    if a.stage == "curve":
        st: dict = {}
        files = [out / a.timings_name] + [Path(m) for m in a.merge]
        used = []
        for f in files:
            if not f.exists():
                print(f"[b6] curve: {f} absent, skipped", flush=True)
                continue
            d = json.loads(f.read_text()); used.append(str(f))
            st.setdefault("population", {}).update(d.get("population", {}))
            for k, v in (d.get("stages") or {}).items():
                if isinstance(v, dict):
                    st.setdefault("stages", {}).setdefault(k, {}).update(v)
                else:
                    st.setdefault("stages", {})[k] = v
        if not used:
            raise SystemExit(f"[b6] ABORT: no stage timing file found among {[str(f) for f in files]}")
        spd = None
        norm: dict = {}
        if a.tasks and Path(a.tasks).exists():
            docs: set[str] = set(); n_sent = 0
            for r in common.load_jsonl(Path(a.tasks)):
                docs.add(r.get("doc_key", "")); n_sent += 1
            spd = n_sent / max(1, len(docs))
            norm = {"tasks_file": a.tasks, "sentences": n_sent, "documents": len(docs),
                    "corpus_sentences_per_document": spd,
                    "why": ("seconds per sentence is the invariant decomposition rate; the timing sample's documents "
                            "are longer than the corpus average, so a per-document figure read off the sample is not "
                            "the corpus per-document figure")}
            print(f"[b6] corpus normalisation: {n_sent} sentences over {len(docs)} documents "
                  f"= {spd:.4f} sentences/document", flush=True)
        curve = build_curve(st, grid, spd)
        curve["decomposition_normalisation"] = norm
        curve["g4_denominator_audit"] = g4_audit(st, a, spd)
        curve["inputs"] = used
        curve["declared_constants"] = {k: {"value": v, "provenance": p} for k, (v, p) in DECLARED.items()}
        curve["what_mc_cost_py_does_not_cover"] = [ln.strip() for ln in (__doc__ or "").split("WHAT THE PAPER CURRENTLY HAS")[1]
                                                   .split("STAGES TIMED HERE")[0].splitlines() if ln.strip()]
        save(out / a.curve_name, curve)
        print_table(curve)
        print(f"[b6] curve written -> {out / a.curve_name}", flush=True)
        return

    which = {w.strip() for w in a.which.split(",") if w.strip()}
    do_cpu = "cpu" in which or "all" in which
    do_gpu = "gpu" in which or "all" in which
    dev = "cpu"
    if do_gpu:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cpu":
            print("[b6] WARNING: --which includes gpu but no CUDA device is visible; GPU stages will be timed on CPU "
                  "and are NOT comparable to the recorded figures", flush=True)

    st = {"block": "B6", "written_utc": now(), "which": sorted(which),
          "host": platform.node(), "python": sys.version.split()[0],
          "device": dev, "n_pairs": a.n_pairs, "n_docs": a.n_docs, "n_docs_gemma": a.n_docs_gemma,
          "env": {k: os.environ.get(k, "") for k in ("SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES", "HF_HOME")},
          "population": {}, "stages": {}}
    if do_gpu:
        import torch
        st["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    target = out / a.timings_name

    inventories = {"gemma+ent": a.inv_gemma, "distilled": a.inv_distilled}
    st["population"] = population(inventories)
    save(target, st)

    inv_rows = {}
    for name, path in inventories.items():
        if Path(path).exists():
            inv_rows[name] = [r for r in common.load_jsonl(Path(path)) if r.get("units")][:a.n_pairs]

    if do_cpu:
        st["stages"]["emission"] = time_emission(inv_rows.get("gemma+ent", []) or inv_rows.get("distilled", []))
        save(target, st)
        st["stages"]["evidence"] = time_claim_rendering(inv_rows.get("gemma+ent", []) or inv_rows.get("distilled", []))
        save(target, st)
        docs = pick_documents(Path(a.inv_gemma), a.n_docs) if Path(a.inv_gemma).exists() else []
        if docs:
            st["stages"].setdefault("decomposition", {})["spacy_entity"] = time_spacy_entities(docs)
            save(target, st)

    if do_gpu:
        seeds = [s.strip() for s in a.seeds.split(",") if s.strip()]
        ckpts = {"humanfact": [a.ckpt_human.format(seed=s) for s in seeds],
                 "judgefact": [a.ckpt_judge.format(seed=s) for s in seeds]}
        if not a.offshelf_only:
            ver, agg = time_verification(inv_rows, ckpts, a.batch, dev)
            st["stages"]["verification"] = ver; st["stages"]["aggregation"] = agg
            save(target, st)

        systems = [s.strip() for s in a.offshelf.split(",") if s.strip()]
        for k, inv_name in enumerate(x.strip() for x in a.offshelf_inventories.split(",") if x.strip()):
            if systems and inv_rows.get(inv_name):
                key = "offshelf" if k == 0 and inv_name == "gemma+ent" else f"offshelf_{inv_name}"
                st["stages"][key] = time_offshelf(inv_rows[inv_name], systems, 32, dev,
                                                  a.offshelf_unit_cap, a.slow_unit_cap)
                save(target, st)
        if a.offshelf_only:
            st["finished_utc"] = now(); save(target, st)
            return

        docs = pick_documents(Path(a.inv_gemma), a.n_docs) if Path(a.inv_gemma).exists() else []
        if docs:
            tasks = sentence_tasks(docs)
            print(f"[b6] decomposition sample: {len(docs)} documents, {len(tasks)} sentence tasks "
                  f"({len(tasks) / max(1, len(docs)):.2f} sentences/document)", flush=True)
            st["stages"].setdefault("decomposition", {})["distilled"] = time_distilled(tasks, len(docs), a.segmenter, 32, dev)
            save(target, st)
            if not a.skip_gemma:
                gdocs = docs[:a.n_docs_gemma]
                gtasks = sentence_tasks(gdocs)
                snap = a.gemma_snapshot
                if not snap:
                    import m3common as m3
                    snap = m3.snap("google/gemma-4-31B-it")
                st["stages"]["decomposition"]["gemma"] = time_gemma(gtasks, len(gdocs), snap, a.gemma_batch,
                                                                   a.gemma_max_new_tokens, dev)
                save(target, st)

    st["finished_utc"] = now()
    save(target, st)
    print(f"[b6] timings written -> {target}", flush=True)


if __name__ == "__main__":
    main()
