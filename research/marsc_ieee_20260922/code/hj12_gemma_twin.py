#!/usr/bin/env python3
"""H-J12 (audit response, 2026-09-28) -- a judge-label verifier trained on the deployed inventory's UNIT TYPE.

The audit's point: the judge-label package (B21) was trained on spaCy entity + dependency-proposition units,
while MARS-C is evaluated on Gemma+ent facts, so the package-vs-package recall gap (+0.122) mixes the label
source with a unit-type mismatch; the label-only twin (R04, human ACU cells) gives +0.023. This builds the
third twin: the E0 train/calib documents, summaries and roles unchanged, the fact units replaced by Gemma+ent
facts of the same source (the frozen m32 decomposition, `train_rows_facts.jsonl`), labelled by the same
Gemma-4-31B coverage judge and prompt the other judge arms used.

Count matching: per document, as many Gemma+ent facts as the document has human ACUs, drawn uniformly without
replacement with a fixed seed, so the labelled-cell budget matches R04 (63,343 cells) to within rounding.

  tasks  -> judge tasks (b1_judge_labels.py format) + the sampled-unit manifest
  apply  -> docs_<role>.jsonl twin (facts = sampled Gemma+ent units, labels = judge), tetrads re-derived by the E0
            rule, census copied; a report with the cell count and the judge covered rate
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location(
    "mc_r04_relabel", HERE.parents[2] / "marsc_20260916" / "code" / "mc_r04_relabel.py")
r04 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(r04)


def load_jsonl(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def units_by_source(path: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for r in load_jsonl(path):
        texts = [u["text"] for u in r["units"]]
        prev = out.setdefault(r["source"], texts)
        if prev != texts:
            raise SystemExit(f"unit lists differ across pairs of one source ({r['pair_id']})")
    return out


def sample_units(d: dict, pool: list[str], seed: int) -> list[str]:
    # deduplicate texts, keep order, then draw n = number of human ACUs of the document
    seen, uniq = set(), []
    for t in pool:
        if t not in seen:
            seen.add(t); uniq.append(t)
    rng = random.Random(f"{seed}:{d['doc_key']}")
    n = min(len(d["facts"]), len(uniq))
    return [uniq[i] for i in sorted(rng.sample(range(len(uniq)), n))]


def cmd_tasks(a) -> None:
    pool = units_by_source(a.units)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    excl = set(a.exclude.split(","))
    n, per_role, manifest = 0, Counter(), {}
    with open(out / "judge_tasks.jsonl", "w", encoding="utf-8") as fh:
        for role in a.roles.split(","):
            for d in load_jsonl(Path(a.docs_dir) / f"docs_{role}.jsonl"):
                if d["source"] not in pool:
                    raise SystemExit(f"no Gemma+ent units for doc {d['doc_key']} ({role})")
                units = sample_units(d, pool[d["source"]], a.seed)
                manifest[d["doc_key"]] = units
                for s in d["summaries"]:
                    if s["system"] in excl:
                        continue
                    for ui, u in enumerate(units):
                        fh.write(json.dumps({"task_id": f"{s['pair_id']}#g{ui}", "pair_id": s["pair_id"],
                                             "unit_index": ui, "split": role, "source": d["source"],
                                             "candidate": s["candidate"], "unit_text": u,
                                             "unit_kind": "prop_llm", "doc_key": d["doc_key"]}) + "\n")
                        n += 1; per_role[role] += 1
    (out / "sampled_units.json").write_text(json.dumps({"seed": a.seed, "units_file": a.units,
                                                        "units_sha256": sha256(a.units), "docs": manifest}))
    print(f"[hj12g-tasks] {n} judge tasks over {len(manifest)} docs ({dict(per_role)}) -> {out}", flush=True)


def cmd_apply(a) -> None:
    man = json.loads((Path(a.tasks_dir) / "sampled_units.json").read_text())["docs"]
    jl = {}
    for f in sorted(glob.glob(a.judge)):
        for r in load_jsonl(f):
            jl[r["task_id"]] = r
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    excl = set(a.exclude.split(","))
    rep = {"registered": "research/marsc_ieee_20260922/PREREG.md H-J12 (unit-matched judge twin)",
           "judge_glob": a.judge, "threshold_p_omitted": a.threshold, "roles": {}}
    miss = cov = tot = 0
    for role in a.roles.split(","):
        docs = load_jsonl(Path(a.docs_dir) / f"docs_{role}.jsonl")
        cells = 0
        for d in docs:
            units = man[d["doc_key"]]
            d["facts"] = list(units)
            for s in d["summaries"]:
                if s["system"] in excl:
                    s["labels"] = [None] * len(units)
                    continue
                lab = []
                for ui in range(len(units)):
                    j = jl.get(f"{s['pair_id']}#g{ui}")
                    if j is None:
                        miss += 1; lab.append(None); continue
                    v = 0 if float(j["p_omitted"]) >= a.threshold else 1      # 1 = covered, as in E0
                    lab.append(v); cov += v; tot += 1; cells += 1
                s["labels"] = lab
            all_t = r04.tetrads_of(d, a.ratio, excl)
            d["tetrads"] = all_t[:2000]; d["n_tetrads"] = len(all_t)
            d["n_eligible_pairs"] = r04.eligible_pairs(d, a.ratio, excl)
            d["collided_pairs"] = []; d["conflicts"] = 0
        with open(out / f"docs_{role}.jsonl", "w", encoding="utf-8") as fh:
            for d in docs:
                fh.write(json.dumps(d) + "\n")
        rep["roles"][role] = {"docs": len(docs), "labelled_cells": cells,
                              "docs_with_tetrads": sum(1 for d in docs if d["n_tetrads"]),
                              "tetrads": sum(d["n_tetrads"] for d in docs)}
        print(f"[hj12g-apply] {role}: {len(docs)} docs, {cells} cells, "
              f"{rep['roles'][role]['docs_with_tetrads']} docs with tetrads", flush=True)
    rep["judge_covered_rate"] = cov / max(1, tot)
    rep["unjudged_cells_dropped"] = miss
    (out / "relabel_report.json").write_text(json.dumps(rep, indent=1))
    print(f"[hj12g-apply] judge covered rate {rep['judge_covered_rate']:.3f}; unjudged {miss}", flush=True)


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tasks")
    t.add_argument("--docs-dir", required=True); t.add_argument("--units", required=True)
    t.add_argument("--roles", default="train,calib"); t.add_argument("--exclude", default="gold")
    t.add_argument("--seed", type=int, default=20260928); t.add_argument("--out", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--docs-dir", required=True); p.add_argument("--tasks-dir", required=True)
    p.add_argument("--judge", required=True); p.add_argument("--roles", default="train,calib")
    p.add_argument("--exclude", default="gold"); p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--ratio", type=float, default=1.25); p.add_argument("--out", required=True)
    a = ap.parse_args(); {"tasks": cmd_tasks, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    main()
