#!/usr/bin/env python3
"""M0 -- census of every human-judgment set and benchmark the MARS metric program uses. NOTHING IS SCORED.

Fetches the sets that are not on the cluster yet (SummEdits full release, FRANK annotations, RAGTruth-processed,
MiniCheck C2D/D2C) into --data-root, reads the ones already there, and writes census.json (counts, human
dimensions with their value ranges, text lengths, licence strings, SHA-256 of every input file). Registered as
M0 in research/mars_metric_20260923/PREREG.md; runs on the login node (network + CPU). A set that fails to fetch
or parse is recorded under "errors" rather than aborting the census.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def words(t) -> int:
    return len(str(t).split())


def q(v):
    v = [float(x) for x in v if x is not None]
    return {"n": len(v), "min": min(v), "median": st.median(v), "max": max(v)} if v else {"n": 0}


def as_dict(v) -> dict:
    """UniSumEval ships its machine scores as a dict in some rows and as its repr string in others."""
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip().startswith("{"):
        try:
            return eval(v)  # noqa: S307 - trusted release file, Python-repr dicts
        except Exception:  # noqa: BLE001
            return {}
    return {}


def fetch_hub(repo: str, files: list[str], dest: Path) -> list[Path]:
    from huggingface_hub import hf_hub_download
    return [Path(hf_hub_download(repo, f, repo_type="dataset", local_dir=str(dest))) for f in files]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/user/data/mars_metric")
    ap.add_argument("--out", default="/home/user/results/mars_metric/m0")
    ap.add_argument("--unisum-dir", default="/home/user/results/marsc/g5/UniSumEval/data")
    ap.add_argument("--summeval", default="/home/user/hf_cache/hub/datasets--mteb--summeval/snapshots/bfc121155064afa2d81b5505682ffc0d96f4334c/data/test-00000-of-00001-35901af5f6649399.parquet")
    ap.add_argument("--aggrefact-dir", default="/home/user/hf_cache/hub/datasets--lytang--LLM-AggreFact/snapshots/981dfd0bd8e58e7238a9ab92b2e6ea44bce918e4/data")
    ap.add_argument("--rose-acu", default="/home/user/results/mars2_gates/b18/acu_units_test.jsonl")
    ap.add_argument("--halueval", default="/home/user/data/public/halueval_summarization_data.json")
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    root = Path(a.data_root); root.mkdir(parents=True, exist_ok=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    census: dict = {"sets": {}, "files": {}, "errors": {}}

    def record(name: str, path: Path, licence: str, **stats):
        census["sets"][name] = {"path": str(path), "licence": licence, **stats}
        census["files"][str(path)] = sha(path)
        print(f"[m0] {name}: {json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}, default=str)[:300]}", flush=True)

    def guarded(name: str, fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            census["errors"][name] = f"{type(e).__name__}: {e}"
            print(f"[m0] {name}: {type(e).__name__}: {e}", flush=True)

    # ---------------- UniSumEval (on cluster)
    def unisum():
        kf = [json.loads(l) for l in open(Path(a.unisum_dir) / "unisumeval_keyfact.jsonl", encoding="utf-8")]
        ff = [json.loads(l) for l in open(Path(a.unisum_dir) / "unisumeval_faithfulness.jsonl", encoding="utf-8")]
        fmap = {(r["uid"], r["model"]): r for r in ff}
        rows = [r for r in kf if (r["uid"], r["model"]) in fmap and str(r.get("summary_success_state", "success")) == "success"]
        record("unisumeval", Path(a.unisum_dir) / "unisumeval_keyfact.jsonl", "DISLab UniSumEval release (see its repository)",
               rows=len(rows), docs=len({r["uid"] for r in rows}), systems=sorted({r["model"] for r in rows}),
               domains=dict(Counter(r["domain"] for r in rows)),
               completeness=q(r.get("completeness") for r in rows), conciseness=q(r.get("conciseness") for r in rows),
               faithfulness=q(fmap[(r["uid"], r["model"])].get("faithfulness_score") for r in rows),
               keyfacts_per_doc=q(len(json.loads(r["keyfact"])) if isinstance(r["keyfact"], str) else len(r["keyfact"]) for r in rows),
               summary_words=q(words(r["summary"]) for r in rows), source_words=q(words(r["input_context"]) for r in rows),
               geval_columns=sorted({k for r in rows for k in as_dict(r.get("machine_evaluation_results_completeness")).keys()}))
        census["files"][str(Path(a.unisum_dir) / "unisumeval_faithfulness.jsonl")] = sha(Path(a.unisum_dir) / "unisumeval_faithfulness.jsonl")
    guarded("unisumeval", unisum)

    # ---------------- SummEval (mteb copy on cluster)
    def summeval():
        df = pd.read_parquet(a.summeval)
        n_mach = [len(list(x)) for x in df["machine_summaries"]]; n_ref = [len(list(x)) for x in df["human_summaries"]]
        record("summeval", Path(a.summeval), "mteb/summeval (SummEval, Fabbri et al. 2021)",
               docs=int(len(df)), machine_summaries_per_doc=q(n_mach), references_per_doc=q(n_ref),
               dims={d: q(float(v) for x in df[d] for v in list(x)) for d in ("relevance", "consistency", "coherence", "fluency")},
               source_words=q(words(t) for t in df["text"]))
    guarded("summeval", summeval)

    # ---------------- RoSE test (human ACU recall per pair)
    def rose():
        rows = [json.loads(l) for l in open(a.rose_acu, encoding="utf-8")]
        rec = []
        for r in rows:
            lab = [x for x in r["unit_labels"] if x is not None]
            if lab:
                rec.append(sum(lab) / len(lab))
        record("rose_test", Path(a.rose_acu), "Salesforce/rose", pairs=len(rows), docs=len({r["source"] for r in rows}),
               systems=len({r["system"] for r in rows}), acu_recall=q(rec), acus_per_pair=q(len(r["units"]) for r in rows))
    guarded("rose_test", rose)

    # ---------------- LLM-AggreFact (on cluster)
    def aggrefact():
        for split in ("dev", "test"):
            p = Path(a.aggrefact_dir) / f"{split}-00000-of-00001.parquet"; d = pd.read_parquet(p)
            record(f"llm_aggrefact_{split}", p, "lytang/LLM-AggreFact (per-dataset licences in its card)", rows=int(len(d)),
                   datasets=d.groupby("dataset").size().to_dict(), label_rate=float(d["label"].mean()),
                   doc_words=q(words(t) for t in d["doc"]), claim_words=q(words(t) for t in d["claim"]))
    guarded("llm_aggrefact", aggrefact)

    # ---------------- HaluEval summarization (local)
    def halueval():
        h = json.load(open(a.halueval, encoding="utf-8")) if open(a.halueval).read(1) == "[" else [json.loads(l) for l in open(a.halueval, encoding="utf-8")]
        record("halueval_summarization", Path(a.halueval), "HaluEval (MIT, Li et al. 2023)", pairs=len(h),
               doc_words=q(words(x["document"]) for x in h), summary_words=q(words(x["right_summary"]) for x in h))
    guarded("halueval_summarization", halueval)

    # ---------------- fetched sets
    if not a.no_fetch:
        def summedits():
            (p,) = fetch_hub("Salesforce/summedits", ["summedits.json"], root / "summedits")
            se = json.load(open(p, encoding="utf-8"))
            recs = se if isinstance(se, list) else [x for v in se.values() for x in (v if isinstance(v, list) else [])]
            record("summedits", p, "CC BY 4.0 (Salesforce/summedits)", rows=len(recs),
                   domains=dict(Counter(str(x.get("domain", x.get("dataset", "?"))) for x in recs)),
                   labels=dict(Counter(str(x.get("label")) for x in recs)), fields=sorted(recs[0].keys()) if recs else [])
        guarded("summedits", summedits)

        def frank():
            fr = root / "frank"; fr.mkdir(exist_ok=True)
            for f in ("human_annotations.json", "benchmark_data.json"):
                urllib.request.urlretrieve(f"https://raw.githubusercontent.com/artidoro/frank/master/data/{f}", fr / f)
            ha = json.load(open(fr / "human_annotations.json", encoding="utf-8"))
            record("frank", fr / "human_annotations.json", "FRANK (artidoro/frank, MIT) built on CNN/DM and XSum", rows=len(ha),
                   fields=sorted(ha[0].keys()) if ha else [], models=dict(Counter(x.get("model_name", "?") for x in ha)))
            census["files"][str(fr / "benchmark_data.json")] = sha(fr / "benchmark_data.json")
        guarded("frank", frank)

        def ragtruth():
            for p in fetch_hub("wandb/RAGTruth-processed", ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"], root / "ragtruth"):
                d = pd.read_parquet(p); ex = d.iloc[0]
                record(f"ragtruth_{p.stem.split('-')[0]}", p, "RAGTruth (wandb/RAGTruth-processed; original: ParticleMedia/RAGTruth, MIT)",
                       rows=int(len(d)), columns=list(d.columns),
                       span_like_columns=[c for c in d.columns if "span" in c.lower() or "label" in c.lower()],
                       example={c: str(ex[c])[:160] for c in d.columns})
        guarded("ragtruth", ragtruth)

        def minicheck():
            for p in fetch_hub("lytang/C2D-and-D2C-MiniCheck", ["data/c2d-00000-of-00001-5a473df7966a0bbe.parquet", "data/d2c-00000-of-00001-fb38b2c3dc66f609.parquet"], root / "minicheck"):
                d = pd.read_parquet(p); ex = d.iloc[0]
                record(f"minicheck_{p.stem.split('-')[0]}", p, "MIT (lytang/C2D-and-D2C-MiniCheck)", rows=int(len(d)), columns=list(d.columns),
                       example={c: str(ex[c])[:120] for c in d.columns})
        guarded("minicheck", minicheck)

    (out / "census.json").write_text(json.dumps(census, indent=1, default=str))
    print(f"[m0] written {out / 'census.json'} ({len(census['sets'])} sets, {len(census['files'])} files hashed, "
          f"{len(census['errors'])} errors)", flush=True)


if __name__ == "__main__":
    main()
