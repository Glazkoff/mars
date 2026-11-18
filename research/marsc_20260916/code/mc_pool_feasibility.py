#!/usr/bin/env python3
"""MARS-C G5 -- feasibility census of untouched confirmation pools (PLAN_MAIN_CONTRIBUTION.md, block G5).

Clones the public repositories, walks their data files, counts records and fields, flags per-fact coverage annotations,
reads licence files, and measures overlap between their source documents and our source pool (sha1 of the first 200
characters, then 8-gram shingle Jaccard >= 0.5). Reports go / no-go per pool; scores nothing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402

POOLS = {"UniSumEval": "https://github.com/DISL-Lab/UniSumEval-v1.0", "QAPyramid": "https://github.com/ZhangShiyue/QAPyramid"}
TEXT_KEYS = ("source", "document", "article", "transcript", "input", "text", "dialogue", "doc")
FACT_KEYS = ("key_fact", "keyfact", "key-fact", "scu", "acu", "fact", "qa", "question", "pyramid", "completeness", "coverage", "label")


def shingles(text: str, n: int = 8) -> set:
    t = common.tokens(text)
    return {" ".join(t[i:i + n]) for i in range(max(0, len(t) - n + 1))}


def iter_records(p: Path):
    try:
        if p.suffix == ".jsonl":
            for l in open(p, encoding="utf-8", errors="ignore"):
                l = l.strip()
                if l:
                    yield json.loads(l)
        elif p.suffix == ".json":
            d = json.load(open(p, encoding="utf-8", errors="ignore"))
            if isinstance(d, list):
                yield from (x for x in d if isinstance(x, dict))
            elif isinstance(d, dict):
                vals = [v for v in d.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
                yield from (x for v in vals for x in v) if vals else [d]
        elif p.suffix in (".csv", ".tsv"):
            with open(p, encoding="utf-8", errors="ignore", newline="") as fh:
                yield from csv.DictReader(fh, delimiter="\t" if p.suffix == ".tsv" else ",")
    except Exception as e:  # noqa: BLE001
        yield {"__error__": str(e)[:200]}


def find_text(rec: dict) -> str | None:
    for k, v in rec.items():
        if isinstance(v, str) and len(v) > 200 and any(t in k.lower() for t in TEXT_KEYS):
            return v
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--our-sources", nargs="+", required=True, help="jsonl files whose rows carry a 'source' field")
    ap.add_argument("--out", required=True); ap.add_argument("--max-records", type=int, default=20000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ours_hash = set(); ours_sh = []
    for f in a.our_sources:
        for r in common.load_jsonl(Path(f)):
            s = r.get("source", "")
            if s:
                h = hashlib.sha1(s[:200].encode()).hexdigest()
                if h not in ours_hash:
                    ours_hash.add(h); ours_sh.append(shingles(s))
    rep = {"our_sources": len(ours_hash), "pools": {}}
    for name, url in POOLS.items():
        d = out / name; info = {"url": url, "clone": None, "licence": None, "files": [], "records": 0, "text_records": 0, "overlap_sha1": 0, "overlap_jaccard": 0, "fact_fields": Counter(), "errors": 0}
        if not d.exists():
            r = subprocess.run(["git", "clone", "--depth", "1", url, str(d)], capture_output=True, text=True, timeout=600)
            info["clone"] = "ok" if r.returncode == 0 else f"failed: {r.stderr[-300:]}"
        else:
            info["clone"] = "cached"
        if d.exists():
            for lic in list(d.glob("LICENSE*")) + list(d.glob("LICENCE*")) + list(d.glob("*/LICENSE*")):
                info["licence"] = lic.read_text(errors="ignore")[:300].replace("\n", " "); break
            n = 0; seen = set()
            for p in sorted(d.rglob("*")):
                if p.suffix not in (".json", ".jsonl", ".csv", ".tsv") or not p.is_file() or p.stat().st_size > 200_000_000:
                    continue
                cnt = 0; keys = Counter()
                for rec in iter_records(p):
                    if "__error__" in rec:
                        info["errors"] += 1; break
                    cnt += 1; n += 1
                    for k in rec:
                        keys[str(k)[:40]] += 1
                        if any(t in str(k).lower() for t in FACT_KEYS):
                            info["fact_fields"][str(k)[:40]] += 1
                    t = find_text(rec)
                    if t:
                        info["text_records"] += 1; h = hashlib.sha1(t[:200].encode()).hexdigest()
                        if h in seen:
                            continue
                        seen.add(h)
                        if h in ours_hash:
                            info["overlap_sha1"] += 1
                        elif len(seen) <= 3000:
                            sh = shingles(t)
                            if sh and any(len(sh & o) / max(1, len(sh | o)) >= 0.5 for o in ours_sh):
                                info["overlap_jaccard"] += 1
                    if n >= a.max_records:
                        break
                info["files"].append({"path": str(p.relative_to(d)), "records": cnt, "keys": [k for k, _ in keys.most_common(12)], "size_kb": p.stat().st_size // 1024})
                if n >= a.max_records:
                    break
            info["records"] = n; info["unique_texts"] = len(seen)
        info["fact_fields"] = dict(info["fact_fields"].most_common(15)); rep["pools"][name] = info
        print(f"[g5] {name}: clone {info['clone']}; files {len(info['files'])}, records {info['records']}, text records {info['text_records']}, unique texts {info.get('unique_texts')}, overlap sha1 {info['overlap_sha1']} / jaccard {info['overlap_jaccard']}; fact-like fields {list(info['fact_fields'])[:8]}; licence: {(info['licence'] or 'none found')[:80]}", flush=True)
    (out / "feasibility.json").write_text(json.dumps(rep, indent=1, default=str))
    md = ["# Untouched-pool feasibility census\n", f"Our source pool: {rep['our_sources']} unique documents.\n"]
    for name, info in rep["pools"].items():
        md.append(f"## {name}\n- clone: {info['clone']}\n- licence: {(info['licence'] or 'none found')[:200]}\n- records: {info['records']} in {len(info['files'])} data files; records with a long text field: {info['text_records']} ({info.get('unique_texts')} unique)\n- overlap with our sources: {info['overlap_sha1']} exact, {info['overlap_jaccard']} near-duplicate (8-gram Jaccard ≥ 0.5, first 3000 texts)\n- fact-like fields: {info['fact_fields']}\n- files: " + "; ".join(f"{f['path']} ({f['records']} rec; keys {f['keys'][:6]})" for f in info["files"][:12]) + "\n")
    (out / "FEASIBILITY.md").write_text("\n".join(md)); print(f"[g5] written {out}", flush=True)
    if not re.search(r"ok|cached", " ".join(str(v["clone"]) for v in rep["pools"].values())):
        raise SystemExit("[g5] no pool could be cloned (network?)")


if __name__ == "__main__":
    main()
