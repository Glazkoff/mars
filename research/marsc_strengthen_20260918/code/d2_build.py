#!/usr/bin/env python3
"""MARS-C D2 -- build and FREEZE the fixed-budget revision study (PREREG Wave D / D2, governing proposal Block 3).

The study. One revision model, one prompt, six arms that differ ONLY in the list of facts handed to the reviser:

  none       no feedback at all (the floor: what the model does with the document alone)
  importance source-importance facts -- the frozen E2 importance tagger, source-only, never sees the summary
  llmjudge   the judge-label-supervised verifier pipeline (the paper's strongest LLM-feedback comparator)
  mars2      the MARS-2 support-based coverage pipeline (the NLI-style coverage comparator)
  marsc      MARS-C, the method under test
  oracle     human-marked omissions -- the ceiling. Not attainable by any detector; it bounds what feedback can buy.

Matched by construction, and asserted here rather than asserted in prose:
  * source access      -- every arm receives the identical source text, truncated identically;
  * revision calls     -- exactly one per (pair, arm); candidate count exactly one (greedy, no best-of-n);
  * feedback budget    -- at most K facts, K identical across arms; per-fact length capped identically;
  * final word budget  -- W is a function of the ORIGINAL summary only, so it is arm-invariant BY CONSTRUCTION,
                         and this script asserts that the value stored on every arm of a pair is identical.
                         The governing proposal is explicit that MARS-C must not win by making summaries longer.

Emission rule. The detector arms are read at the SAME budget and under the SAME rule as the main table: the
corrected k=10 dump for the three dump-backed arms, and for the importance arm a local copy of the registered
div1 rule (one unit per source sentence, later units pushed by -5.0) followed by the evaluator's own ordering
key (-score, start). The copy is deliberate: research/marsc_20260916/ is shared with live sessions and may not
be edited, and a drifting shared file must not silently change a frozen artifact.

This script computes NO outcome. It selects pairs, assembles feedback, and freezes. Every outcome of the
revision study (recovered important omissions, preservation of previously correct content, unsupported new
claims, redundancy, usefulness) requires two blinded reviewers plus adjudication; the only machine analogue is
the support judge, which is the question this project measured to FAIL (kappa 0.4727; it calls 36% of supported
facts unsupported). No outcome may be read from a machine here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402

ARMS = ["none", "importance", "llmjudge", "mars2", "marsc", "oracle"]
DUMP_ARM_SYSTEM = {"marsc": "humanfact+div1", "llmjudge": "judgefact+div1", "mars2": "mars2"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(p: Path):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def sentence_index(source: str, start: int) -> int:
    for i, (a, b) in enumerate(common.sentence_spans(source)):
        if a <= start < b:
            return i
    return -1


def div_push(scores: np.ndarray, sent_of: list[int], cap: int = 1) -> np.ndarray:
    """Local copy of the registered div1 rule (mc_diversity.py): keep `cap` units per source sentence, push the rest -5."""
    v = np.array(scores, dtype=float)
    new = v.copy()
    count: dict[int, int] = {}
    for i in np.argsort(-np.nan_to_num(v, nan=-np.inf)):
        if not np.isfinite(v[i]) or v[i] <= -9.0:
            continue
        s = sent_of[i]
        if count.get(s, 0) >= cap:
            new[i] = v[i] - 5.0
        else:
            count[s] = count.get(s, 0) + 1
    return new


def top_k(units: list[dict], sc: np.ndarray, k: int) -> list[dict]:
    """The evaluator's own ordering key, copied verbatim from mc_e2_eval.per_pair."""
    order = sorted(range(len(units)),
                   key=lambda j: (-sc[j] if np.isfinite(sc[j]) else 1.0, units[j].get("start", 0)))[:k]
    return [units[j] for j in order]


def clip_words(text: str, n: int) -> str:
    w = text.split()
    return text if len(w) <= n else " ".join(w[:n]) + " ..."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acu", required=True, help="labelled ACU file: source, candidate, units, unit_labels")
    ap.add_argument("--emitted", required=True, help="the corrected k=10 emitted dump (marsc / llmjudge / mars2 arms)")
    ap.add_argument("--units-gemma", required=True, help="units_gemma+ent_<split>.jsonl (importance arm inventory)")
    ap.add_argument("--importance-scores", required=True,
                    help="COMMA-SEPARATED explicit seed manifest of importance score files -- never a glob")
    ap.add_argument("--k", type=int, default=10, help="feedback budget: facts handed to the reviser, identical per arm")
    ap.add_argument("--max-fact-words", type=int, default=60)
    ap.add_argument("--max-source-words", type=int, default=2500)
    ap.add_argument("--budget-factor", type=float, default=1.5, help="W = max(floor, round(factor * original words))")
    ap.add_argument("--budget-floor", type=int, default=60)
    ap.add_argument("--docs-per-resource", type=int, default=50)
    ap.add_argument("--summaries-per-doc", type=int, default=2)
    ap.add_argument("--min-omitted", type=int, default=2, help="pairs need this many human-marked omissions to be scorable")
    ap.add_argument("--pilot-docs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    imp_files = [x for x in a.importance_scores.split(",") if x]
    assert len(imp_files) == 3, f"importance arm expects an explicit three-seed manifest, got {imp_files}"
    seeds = sorted({(re.search(r"seed(\d+)", f).group(1) if re.search(r"seed(\d+)", f) else f) for f in imp_files})
    assert len(seeds) == 3, f"importance manifest does not carry three DISTINCT seeds: {imp_files} -> {seeds}"
    for f in imp_files:
        assert Path(f).exists(), f"missing importance score file {f}"

    pairs = {r["pair_id"]: r for r in load_jsonl(Path(a.acu))}

    # --- dump-backed arms --------------------------------------------------------------------------------
    dump: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for r in load_jsonl(Path(a.emitted)):
        if r.get("emitted"):
            dump[r["system"]][r["pair_id"]] = r["emitted"]
    for sysname in DUMP_ARM_SYSTEM.values():
        assert dump.get(sysname), f"emitted dump has no rows for system {sysname!r}"

    # --- importance arm: seed-mean -> div1 -> evaluator ordering ------------------------------------------
    inv = {r["pair_id"]: r["units"] for r in load_jsonl(Path(a.units_gemma)) if r.get("units")}
    sent_of: dict[str, list[int]] = {}
    for r in load_jsonl(Path(a.units_gemma)):
        if r.get("units"):
            n = len(r["source"])
            sent_of[r["pair_id"]] = [sentence_index(r["source"], int(u.get("start", n // 2))) for u in r["units"]]
    acc: dict[str, list[np.ndarray]] = defaultdict(list)
    for f in imp_files:
        _, sc = common.read_scores(Path(f))
        for pid, v in sc.items():
            if pid in inv and len(v) == len(inv[pid]):
                acc[pid].append(v)
    imp_top: dict[str, list[dict]] = {}
    for pid, vs in acc.items():
        if len(vs) != 3:
            continue
        pushed = div_push(np.nanmean(np.vstack(vs), axis=0), sent_of[pid])
        imp_top[pid] = top_k(inv[pid], pushed, a.k)

    # --- eligibility and the frozen document-clustered pair sample ---------------------------------------
    elig = []
    for pid, r in pairs.items():
        truth = common.truth_of(r)
        n_om = sum(1 for t in truth if t == 1)
        if n_om < a.min_omitted:
            continue
        if any(pid not in dump[s] for s in DUMP_ARM_SYSTEM.values()) or pid not in imp_top:
            continue
        elig.append({"pair_id": pid, "doc_key": common.doc_key(r["source"]), "resource": r["resource"],
                     "summarizer": r["system"], "n_omitted": n_om})
    by_doc = defaultdict(list)
    for e in elig:
        by_doc[e["doc_key"]].append(e)
    doc_res = {d: v[0]["resource"] for d, v in by_doc.items()}
    per_res = defaultdict(list)
    for d in sorted(by_doc):
        per_res[doc_res[d]].append(d)
    for res in per_res:
        rng.shuffle(per_res[res])
        del per_res[res][a.docs_per_resource:]
    # round-robin across resources so any prefix of the review order is domain-balanced
    doc_order: list[str] = []
    cur = {res: 0 for res in per_res}
    while len(doc_order) < sum(len(v) for v in per_res.values()):
        for res in sorted(per_res):
            if cur[res] < len(per_res[res]):
                doc_order.append(per_res[res][cur[res]])
                cur[res] += 1
    chosen = []
    for d in doc_order:
        cands = sorted(by_doc[d], key=lambda e: e["pair_id"])
        rng.shuffle(cands)
        seen_sys, take = set(), []
        for e in cands:                                    # prefer distinct summarizer systems within a document
            if e["summarizer"] not in seen_sys:
                seen_sys.add(e["summarizer"])
                take.append(e)
            if len(take) == a.summaries_per_doc:
                break
        chosen.extend(take)

    # --- assemble the arms --------------------------------------------------------------------------------
    rows, budget_of = [], {}
    for e in chosen:
        pid = e["pair_id"]
        r = pairs[pid]
        src = " ".join(r["source"].split()[:a.max_source_words])
        orig_words = len(r["candidate"].split())
        W = max(a.budget_floor, int(round(a.budget_factor * orig_words)))
        budget_of[pid] = W
        truth = common.truth_of(r)
        facts_of = {"none": []}
        for arm, sysname in DUMP_ARM_SYSTEM.items():
            facts_of[arm] = [x["text"] for x in dump[sysname][pid][:a.k]]
        facts_of["importance"] = [u["text"] for u in imp_top[pid][:a.k]]
        facts_of["oracle"] = [r["units"][i]["text"] for i, t in enumerate(truth) if t == 1][:a.k]
        for arm in ARMS:
            seen, facts = set(), []
            for t in facts_of[arm]:
                t = clip_words(" ".join(t.split()), a.max_fact_words)
                if t and t.lower() not in seen:
                    seen.add(t.lower())
                    facts.append(t)
            rows.append({"row_id": f"{pid}|{arm}", "pair_id": pid, "arm": arm, "doc_key": e["doc_key"],
                         "resource": e["resource"], "summarizer": e["summarizer"], "n_omitted": e["n_omitted"],
                         "source": src, "summary": r["candidate"], "facts": facts,
                         "n_facts": len(facts), "word_budget": W, "orig_words": orig_words})

    # --- the matching assertions, made programmatically ---------------------------------------------------
    fail = []
    per_pair = defaultdict(dict)
    for row in rows:
        per_pair[row["pair_id"]][row["arm"]] = row
    for pid, d in per_pair.items():
        if set(d) != set(ARMS):
            fail.append(f"{pid}: arms present {sorted(d)} != {ARMS}")
            continue
        if len({d[x]["word_budget"] for x in ARMS}) != 1:
            fail.append(f"{pid}: word budget is NOT arm-invariant: {[d[x]['word_budget'] for x in ARMS]}")
        if len({d[x]["source"] for x in ARMS}) != 1:
            fail.append(f"{pid}: source text differs between arms")
        if len({d[x]["summary"] for x in ARMS}) != 1:
            fail.append(f"{pid}: original summary differs between arms")
        if d["none"]["n_facts"] != 0:
            fail.append(f"{pid}: the no-feedback arm carries {d['none']['n_facts']} facts")
        for x in ARMS:
            if d[x]["n_facts"] > a.k:
                fail.append(f"{pid}/{x}: {d[x]['n_facts']} facts exceeds the budget K={a.k}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "d2_arms.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    pilot = doc_order[:a.pilot_docs]
    (out / "d2_review_order.json").write_text(json.dumps(
        {"seed": a.seed, "doc_order": [d for d in doc_order if d in {e["doc_key"] for e in chosen}],
         "pilot_docs": pilot,
         "pilot_pair_ids": [e["pair_id"] for e in chosen if e["doc_key"] in set(pilot)]}, indent=1))

    freeze = {
        "block": "D2",
        "registered": "research/marsc_strengthen_20260918/PREREG.md, Wave D / D2 (governing proposal Block 3)",
        "arms": ARMS, "dump_arm_systems": DUMP_ARM_SYSTEM,
        "budget": {"K_facts": a.k, "max_fact_words": a.max_fact_words, "max_source_words": a.max_source_words,
                   "word_budget_rule": f"W = max({a.budget_floor}, round({a.budget_factor} * original_words)) "
                                       f"-- a function of the ORIGINAL summary only, hence arm-invariant",
                   "revision_calls_per_pair_per_arm": 1, "candidates_per_call": 1},
        "inputs": {"acu": a.acu, "acu_sha256": sha256(Path(a.acu)),
                   "emitted": a.emitted, "emitted_sha256": sha256(Path(a.emitted)),
                   "units_gemma": a.units_gemma, "units_gemma_sha256": sha256(Path(a.units_gemma)),
                   "importance_manifest": imp_files,
                   "importance_sha256": {Path(f).name: sha256(Path(f)) for f in imp_files}},
        "selection": {"seed": a.seed, "docs_per_resource": a.docs_per_resource,
                      "summaries_per_doc": a.summaries_per_doc, "min_omitted": a.min_omitted,
                      "n_eligible_pairs": len(elig), "n_eligible_docs": len(by_doc),
                      "n_selected_pairs": len(chosen), "n_selected_docs": len({e['doc_key'] for e in chosen}),
                      "resources": dict(Counter(e["resource"] for e in chosen)),
                      "summarizers": dict(Counter(e["summarizer"] for e in chosen))},
        "n_rows": len(rows),
        "facts_per_arm": {arm: dict(Counter(r["n_facts"] for r in rows if r["arm"] == arm)) for arm in ARMS},
        "mean_facts_per_arm": {arm: round(float(np.mean([r["n_facts"] for r in rows if r["arm"] == arm])), 3)
                               for arm in ARMS},
        "word_budget": {"min": min(budget_of.values()) if budget_of else None,
                        "max": max(budget_of.values()) if budget_of else None,
                        "mean": round(float(np.mean(list(budget_of.values()))), 2) if budget_of else None},
        "outcome_policy": ("NO outcome is computed here or by any job in this block. Every outcome needs two "
                           "blinded reviewers plus adjudication; the only machine analogue is the support judge, "
                           "which is the measured failure mode (kappa 0.4727, calls 36% of supported facts "
                           "unsupported) and is therefore not an admissible substitute."),
        "known_gap": ("No off-the-shelf verifier (SummaC-ZS, MiniCheck, AlignScore) can be an arm: every "
                      "off-the-shelf score file in r06_offshelf/ and r06b_fast/ -- including the directory "
                      "misleadingly named val/ -- carries 2829 rows, the TEST inventory, with ZERO validation "
                      "pair_ids. If Wave B1 lands validation off-shelf scores, an arm may be added ONLY by "
                      "re-running this build and the generation unchanged, at the same seed, prompt and model."),
        "checks_failed": fail,
        "status": "FROZEN" if not fail else "REFUSED",
    }
    (out / "d2_freeze.json").write_text(json.dumps(freeze, indent=1))

    for line in fail:
        print(f"[d2] FAIL {line}", flush=True)
    s = freeze["selection"]
    print(f"[d2] eligible {s['n_eligible_pairs']} pairs / {s['n_eligible_docs']} docs -> selected "
          f"{s['n_selected_pairs']} pairs / {s['n_selected_docs']} docs {s['resources']}", flush=True)
    print(f"[d2] {len(rows)} arm rows ({len(ARMS)} arms); mean facts per arm {freeze['mean_facts_per_arm']}", flush=True)
    print(f"[d2] word budget min/mean/max {freeze['word_budget']}", flush=True)
    print(f"[d2] status {freeze['status']} -> {out}", flush=True)
    if fail:
        sys.exit(5)


if __name__ == "__main__":
    main()
