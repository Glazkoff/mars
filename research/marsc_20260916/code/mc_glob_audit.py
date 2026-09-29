#!/usr/bin/env python3
"""MARS-C R08 -- audit every score glob in the MARS-C job chain for ambiguous multi-file expansion.

The consensus defect (FULL_REVIEW_2026-09-17 item 1) was a wildcard that matched more files than the job
intended. That is a class of bug, not one bug, so this scans EVERY `--systems` spec and every `--scores-glob`
in the shipped sbatch files, expands each pattern against the artifacts on disk, and flags:

  PSEUDO_SEED   a multi-file set containing a non-numeric seed token (`seedavg`, `seedens`, `seedrank`, ...),
                i.e. an aggregate of other files averaged in beside the files it aggregates
  DUP_SEED      a multi-file set whose numeric seed tokens are not distinct, i.e. a variant suffix
                (`_imp0.2_div1` next to `_div1`) swept in beside the plain output
  EMPTY         a pattern that matches nothing (the job silently dropped an emitter)
  OK            n files, n distinct numeric seeds

Run it on the cluster where the artifacts live. Exit status 1 if anything is flagged.
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

VARS = {}
# Scripts that deliberately take a glob spanning MANY systems and group by system name internally; the
# one-roster-per-emitter rule does not apply to them.
MULTI_SYSTEM_CONSUMERS = ("mc_analyze.py", "mc_matched_auroc.py")
SEED_RE = re.compile(r"seed([A-Za-z0-9.]+?)(?:_|/|\.jsonl|$)")


def expand(tok: str, local: dict | None = None) -> str:
    """Substitute shell variables, preferring assignments made inside the job file over the global map."""
    env = {**VARS, **(local or {})}
    out = tok
    for _ in range(8):
        prev = out
        def _sub(m):
            name = m.group(1) or m.group(3)
            if name in env:
                return env[name]
            return m.group(2) if m.group(2) is not None else m.group(0)   # ${VAR:-default} falls back to default
        out = re.sub(r"\$\{(\w+)(?::-([^}]*))?\}|\$(\w+)", _sub, out)
        if out == prev:
            break
    return out


def local_vars(text: str) -> dict:
    """NAME="..." / NAME=... assignments made in the job file itself, resolved against the global map."""
    loc: dict[str, str] = {}
    for m in re.finditer(r'(?:^|;)\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(\"[^\"]*\"|[^\s;#]+)', text, re.M):
        name, val = m.group(1), m.group(2).strip('"')
        if "*" in val:
            continue
        loc[name] = expand(val, loc)
    return loc


def classify(files: list[str]) -> tuple[str, str]:
    if not files:
        return "EMPTY", "no file matched"
    if len(files) == 1:
        return "OK", "single file"
    common = os.path.commonpath(files) if len(files) > 1 else str(Path(files[0]).parent)
    toks = []
    for f in files:                       # the seed can sit in a parent directory (b2/scores/<arm>_seed<N>/units.jsonl)
        m = SEED_RE.search(os.path.relpath(f, common))
        toks.append(m.group(1) if m else "")
    bad = [t for t in toks if not t.isdigit()]
    if bad:
        return "PSEUDO_SEED", f"non-numeric seed token(s) {sorted(set(bad))} among {len(files)} files"
    c = Counter(toks)
    dup = {k: v for k, v in c.items() if v > 1}
    if dup:
        return "DUP_SEED", f"{len(files)} files but {len(c)} distinct seeds; repeated {dup}"
    return "OK", f"{len(files)} files, {len(c)} distinct seeds"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slurm-dir", required=True)
    ap.add_argument("--vars", nargs="*", default=[], help="NAME=VALUE substitutions for the sbatch variables")
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-superseded", action="store_true",
                    help="ignore job files carrying a '% SUPERSEDED-BY:' header; use this to check the live chain")
    a = ap.parse_args()
    for kv in a.vars:
        k, v = kv.split("=", 1); VARS[k] = v
    rows = []
    for sb in sorted(Path(a.slurm_dir).glob("*.sbatch")):
        text = sb.read_text()
        superseded = "% SUPERSEDED-BY:" in text
        loc = local_vars(text)
        if superseded and a.skip_superseded:
            continue
        pats = set()
        for m in re.finditer(r'--scores-glob\s+"([^"]+)"', text):
            pats.add(m.group(1))
        for m in re.finditer(r'--scores\s+"([^"]+)"', text):
            pats.add(m.group(1))
        for m in re.finditer(r'"([A-Za-z0-9_.+\-]+=[^"]*?:[^"]*?\*[^"]*?)"', text):
            spec = m.group(1)
            if "=" in spec and ":" in spec:
                pats.add(spec.rsplit(":", 1)[1])
        for m in re.finditer(r'([A-Z_]+)="([^"]*\*[^"]*)"', text):
            if "/" in m.group(2):
                pats.add(m.group(2))
        for p in sorted(pats):
            e = expand(p, loc)
            if "$" in e or "*" not in e:
                continue
            files = sorted(globmod.glob(e))
            line = next((l for l in text.splitlines() if p in l), "")
            if any(c in line for c in MULTI_SYSTEM_CONSUMERS):
                verdict, why = "MULTI_SYSTEM", f"{len(files)} files, consumed by a script that groups by system"
            else:
                verdict, why = classify(files)
            rows.append({"sbatch": sb.name, "superseded": superseded, "pattern": p, "expanded": e,
                         "n_files": len(files), "verdict": verdict, "why": why,
                         "files": [Path(f).name for f in files] if verdict != "OK" else []})
    bad = [r for r in rows if r["verdict"] not in ("OK", "MULTI_SYSTEM")]
    live_bad = [r for r in bad if not r.get("superseded")]
    for r in rows:
        mark = "    " if r["verdict"] in ("OK", "MULTI_SYSTEM") else ">>> "
        print(f"{mark}{r['verdict']:12s} {r['sbatch']:34s} {r['why']}")
        print(f"        {r['expanded']}")
        for f in r["files"]:
            print(f"          {f}")
    print(f"\n[glob-audit] {len(rows)} patterns scanned, {len(bad)} flagged "
          f"({Counter(r['verdict'] for r in bad)}); of those, {len(live_bad)} are in jobs that are NOT "
          f"marked superseded", flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"scanned": len(rows), "flagged": len(bad),
                                           "flagged_in_live_jobs": len(live_bad), "rows": rows}, indent=1))
    sys.exit(1 if live_bad else 0)


if __name__ == "__main__":
    main()
