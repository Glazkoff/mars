#!/usr/bin/env python3
"""MARS-C R04 -- matched-unit, matched-count label-source control (FULL_REVIEW_2026-09-17 failure 3).

The paper compares a human-fact verifier with a judge-label verifier, but the two differ in FOUR things at once:
label source, fact-unit construction (human ACUs vs LLM-extracted propositions), unit distribution, and the
number of labelled cells (about 19k sampled draws vs 119,676 rows). That identifies a *supervision package*,
not a label source.

This control holds everything but the label fixed. It takes the identical documents, the identical non-reference
summaries and the identical human ACUs of the E0 TRAIN and CALIB roles, asks the same Gemma-4-31B-it coverage
judge the paper's pipeline used to label exactly those cells, and writes a judge-labelled twin of the E0 data.
Training the identical recipe, sampler, seeds and step budget on the twin isolates the label source.

  tasks   docs_<role>.jsonl -> judge tasks in the b1_judge_labels.py format (one per labelled (summary, ACU) cell)
  apply   judge shards -> docs_<role>.jsonl twin with judge labels, tetrads re-derived from those labels,
          plus the judge-vs-human cell agreement that the paper should report either way
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def load_jsonl(p: Path):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def tetrads_of(d: dict, ratio: float, excl: set[str]) -> list[list[int]]:
    """(fact_f, fact_g, summary_i, summary_j) with f covered by i and omitted by j, g the opposite.

    Byte-for-byte the E0 rule (mc_data.tetrads_of) so the twin's crossed structure is derived the same way."""
    out = []
    S = [s for s in d["summaries"] if s["system"] not in excl]
    for a in range(len(S)):
        for b in range(a + 1, len(S)):
            si, sj = S[a], S[b]
            if max(si["words"], sj["words"]) > ratio * max(1, min(si["words"], sj["words"])):
                continue
            X = [f for f in range(len(d["facts"])) if si["labels"][f] == 1 and sj["labels"][f] == 0]
            Y = [f for f in range(len(d["facts"])) if si["labels"][f] == 0 and sj["labels"][f] == 1]
            ia, ib = d["summaries"].index(si), d["summaries"].index(sj)
            out.extend([f, g, ia, ib] for f in X for g in Y)
    return out


def eligible_pairs(d: dict, ratio: float, excl: set[str]) -> int:
    S = [s for s in d["summaries"] if s["system"] not in excl]
    n = 0
    for a in range(len(S)):
        for b in range(a + 1, len(S)):
            si, sj = S[a], S[b]
            if max(si["words"], sj["words"]) > ratio * max(1, min(si["words"], sj["words"])):
                continue
            if any(si["labels"][f] == 1 and sj["labels"][f] == 0 for f in range(len(d["facts"]))) and \
               any(si["labels"][f] == 0 and sj["labels"][f] == 1 for f in range(len(d["facts"]))):
                n += 1
    return n


def cmd_tasks(a) -> None:
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    n = 0; per_role = Counter()
    with open(out, "w", encoding="utf-8") as fh:
        for role in a.roles.split(","):
            for d in load_jsonl(Path(a.docs_dir) / f"docs_{role}.jsonl"):
                for s in d["summaries"]:
                    if s["system"] in set(a.exclude.split(",")):
                        continue
                    for fi, lab in enumerate(s["labels"]):
                        if lab is None:
                            continue
                        fh.write(json.dumps({"task_id": f"{s['pair_id']}#f{fi}", "pair_id": s["pair_id"],
                                             "unit_index": fi, "split": role, "source": d["source"],
                                             "candidate": s["candidate"], "unit_text": d["facts"][fi],
                                             "unit_kind": "acu", "doc_key": d["doc_key"]}) + "\n")
                        n += 1; per_role[role] += 1
    print(f"[r04-tasks] {n} judge tasks -> {out} ({dict(per_role)})", flush=True)


def cmd_apply(a) -> None:
    jl = {}
    for f in sorted(glob.glob(a.judge)):
        for r in load_jsonl(Path(f)):
            jl[r["task_id"]] = r
    print(f"[r04-apply] {len(jl)} judge answers from {len(glob.glob(a.judge))} shard file(s)", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    excl = set(a.exclude.split(","))
    agree = Counter(); conf = Counter(); miss = 0; low_mass = 0
    rep = {"registered": "research/marsc_20260916/PREREG.md R04 (matched-unit label-source control)",
           "judge_glob": a.judge, "threshold_p_omitted": a.threshold, "roles": {}}
    for role in a.roles.split(","):
        docs = load_jsonl(Path(a.docs_dir) / f"docs_{role}.jsonl")
        n_cells = n_relabelled = 0
        for d in docs:
            for s in d["summaries"]:
                if s["system"] in excl:
                    continue
                new = list(s["labels"])
                for fi, lab in enumerate(s["labels"]):
                    if lab is None:
                        continue
                    j = jl.get(f"{s['pair_id']}#f{fi}")
                    n_cells += 1
                    if j is None:
                        miss += 1; new[fi] = None            # unjudged cell drops out of BOTH twins downstream
                        continue
                    if j.get("choice_mass", 1.0) < a.min_choice_mass:
                        low_mass += 1
                    jlab = 0 if float(j["p_omitted"]) >= a.threshold else 1     # 1 = covered, as in E0
                    conf[(int(lab), jlab)] += 1
                    agree["same" if jlab == int(lab) else "diff"] += 1
                    if jlab != int(lab):
                        n_relabelled += 1
                    new[fi] = jlab
                s["labels"] = new
            d["tetrads"] = tetrads_of(d, a.ratio, excl)[:2000]
            d["n_tetrads"] = len(tetrads_of(d, a.ratio, excl))
            d["n_eligible_pairs"] = eligible_pairs(d, a.ratio, excl)
            d["collided_pairs"] = []          # the E0 collision census is a tagger-representation diagnostic; not re-run
        with open(out / f"docs_{role}.jsonl", "w", encoding="utf-8") as fh:
            for d in docs:
                fh.write(json.dumps(d) + "\n")
        rep["roles"][role] = {"docs": len(docs), "labelled_cells": n_cells, "cells_flipped_by_judge": n_relabelled,
                              "docs_with_tetrads": sum(1 for d in docs if d["n_tetrads"]),
                              "tetrads": sum(d["n_tetrads"] for d in docs)}
        print(f"[r04-apply] {role}: {len(docs)} docs, {n_cells} cells, {n_relabelled} flipped, "
              f"{rep['roles'][role]['docs_with_tetrads']} docs with tetrads", flush=True)
    tot = sum(agree.values())
    h1j1, h1j0, h0j1, h0j0 = conf[(1, 1)], conf[(1, 0)], conf[(0, 1)], conf[(0, 0)]
    po = (h1j1 + h0j0) / max(1, tot)
    ph = (h1j1 + h1j0) / max(1, tot); pj = (h1j1 + h0j1) / max(1, tot)
    pe = ph * pj + (1 - ph) * (1 - pj)
    rep["judge_vs_human_cells"] = {
        "n": tot, "raw_agreement": po, "cohen_kappa": ((po - pe) / (1 - pe)) if pe < 1 else None,
        "human_covered_judge_covered": h1j1, "human_covered_judge_omitted": h1j0,
        "human_omitted_judge_covered": h0j1, "human_omitted_judge_omitted": h0j0,
        "human_covered_rate": ph, "judge_covered_rate": pj,
        "recall_on_human_omitted": h0j0 / max(1, h0j0 + h0j1),
        "precision_on_judge_omitted": h0j0 / max(1, h0j0 + h1j0),
        "unjudged_cells_dropped": miss, "cells_below_min_choice_mass": low_mass}
    (out / "relabel_report.json").write_text(json.dumps(rep, indent=1))
    k = rep["judge_vs_human_cells"]["cohen_kappa"]
    print(f"[r04-apply] judge vs human on {tot} identical cells: raw {po:.4f}, kappa {k:.4f}; "
          f"human covered {ph:.3f} vs judge covered {pj:.3f}; unjudged dropped {miss}", flush=True)
    print(f"[r04-apply] written {out / 'relabel_report.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tasks")
    t.add_argument("--docs-dir", required=True); t.add_argument("--roles", default="train,calib")
    t.add_argument("--exclude", default="gold"); t.add_argument("--out", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--docs-dir", required=True); p.add_argument("--judge", required=True)
    p.add_argument("--roles", default="train,calib"); p.add_argument("--exclude", default="gold")
    p.add_argument("--threshold", type=float, default=0.5); p.add_argument("--ratio", type=float, default=1.25)
    p.add_argument("--min-choice-mass", type=float, default=0.5); p.add_argument("--out", required=True)
    a = ap.parse_args(); {"tasks": cmd_tasks, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    main()
