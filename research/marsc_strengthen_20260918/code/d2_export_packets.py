#!/usr/bin/env python3
"""MARS-C D2 -- export the BLINDED revision-review packets (PREREG Wave D / D2). Computes no outcome.

Each generated revision becomes one blinded item: the reviewer sees the source document, the original summary
and ONE revised summary, and never learns which arm produced it or that other arms exist. Judging one revision
at a time (rather than six side by side) removes cross-arm anchoring, which a side-by-side panel cannot.

Item ids use the `R#####` namespace so they can never collide with the `H###` omission-confirmation ids that
already live in the deployed label app's database.

The packet is exported in the schema the adjudicator-capable label app consumes under `--study d2`:
`{item_id, source, summary, revised}` and nothing else. The arm identity lives only in the key file, which is
never served.

Review order. `d2_review_order.json` froze a domain-balanced document order and a 30-document pilot prefix
BEFORE any revision existed. This script exports the full packet and the pilot packet separately, so annotating
only the pilot is a pre-registered subsample rather than a post-hoc choice.

No outcome is computed here. The five review questions are frozen in INSTRUCTIONS.md; scoring them requires two
blinded reviewers plus the adjudicator role (research/marsc_strengthen_20260918/labelapp/), and the analysis
must report the agreement-conditioned rate and the adjudicated rate separately (d1b_adjudicate.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

BLIND_KEYS = {"item_id", "source", "summary", "revised"}

INSTRUCTIONS = """# Summary revision review — reviewer instructions

Each item shows a SOURCE document, the ORIGINAL summary of it, and one REVISED summary. Different revisions of
the same document were produced by different systems; you are not told which, and the items are shuffled. Judge
each revision on its own merits and work independently of the other reviewer.

Answer five questions per item.

Q1 NEW IMPORTANT CONTENT — Does the revised summary state at least one important fact from the document that
   the original summary did not?  A = yes.  B = no.

Q2 PRESERVATION — Does the revised summary keep every important fact the original summary already had?
   A = yes, all kept.  B = no, something important was lost.

Q3 SUPPORT — Does the revised summary contain any claim the document does not support (invented, contradicted,
   or a distortion)?  A = no unsupported claim.  B = yes, at least one unsupported claim.

Q4 REDUNDANCY — Does the revised summary repeat the same information more than once?
   A = no redundancy.  B = yes, redundant.

Q5 USEFULNESS — Overall, is the revised summary more useful than the original for a reader who will not read
   the document?  A = more useful.  B = not more useful.

"Important" means: a reader who only sees the summary would be materially misled or under-informed without it.
Length is not a virtue. A longer revision that adds unimportant detail is not more useful. Judge Q3 strictly:
a claim that is merely plausible but absent from the document counts as unsupported.
"""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(p: Path):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="d2_arms.jsonl")
    ap.add_argument("--revisions", nargs="+", required=True, help="revisions.jsonl shard(s)")
    ap.add_argument("--review-order", required=True, help="d2_review_order.json")
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms_rows = {r["row_id"]: r for r in load_jsonl(Path(a.arms))}
    revs: dict[str, dict] = {}
    for f in a.revisions:
        for r in load_jsonl(Path(f)):
            revs[r["row_id"]] = r
    order = json.load(open(a.review_order))
    pilot_docs = set(order.get("pilot_docs") or [])

    by_pair = defaultdict(list)
    for rid, r in revs.items():
        by_pair[r["pair_id"]].append(r)

    fail = []
    n_expected_arms = len({r["arm"] for r in arms_rows.values()})
    items, key = [], []
    n = 0
    doc_rank = {d: i for i, d in enumerate(order.get("doc_order") or [])}
    for pid in sorted(by_pair, key=lambda p: (doc_rank.get(by_pair[p][0]["doc_key"], 1 << 30), p)):
        group = sorted(by_pair[pid], key=lambda r: r["arm"])
        if len(group) != n_expected_arms:
            fail.append(f"{pid}: {len(group)} revisions, expected {n_expected_arms} (one per arm)")
            continue
        empties = [r["arm"] for r in group if r.get("empty")]
        if empties:
            fail.append(f"{pid}: empty revision on arm(s) {empties}")
        rng = random.Random(f"{a.seed}:{pid}")
        variants = list(group)
        rng.shuffle(variants)
        src = arms_rows[group[0]["row_id"]]["source"]
        summ = arms_rows[group[0]["row_id"]]["summary"]
        for vi, r in enumerate(variants):
            n += 1
            iid = f"R{n:05d}"
            items.append({"item_id": iid, "source": src, "summary": summ, "revised": r["revision"]})
            key.append({"item_id": iid, "pair_id": pid, "arm": r["arm"], "variant": chr(ord("A") + vi),
                        "row_id": r["row_id"], "doc_key": r["doc_key"], "resource": r["resource"],
                        "summarizer": r["summarizer"], "n_words": r["n_words"],
                        "word_budget": r["word_budget"], "over_budget": r["over_budget"],
                        "n_facts_used": r["n_facts_used"], "in_pilot": r["doc_key"] in pilot_docs})

    for it in items:
        if set(it) != BLIND_KEYS:
            fail.append(f"{it.get('item_id')}: blind keys {sorted(set(it))} != {sorted(BLIND_KEYS)}")
            break

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    key_by_id = {r["item_id"]: r for r in key}
    with open(out / "d2_review_blind.jsonl", "w", encoding="utf-8") as fb, \
         open(out / "d2_review_key.jsonl", "w", encoding="utf-8") as fk, \
         open(out / "d2_review_pilot_blind.jsonl", "w", encoding="utf-8") as fp:
        for it in items:
            fb.write(json.dumps(it) + "\n")
            if key_by_id[it["item_id"]]["in_pilot"]:
                fp.write(json.dumps(it) + "\n")
        for r in key:
            fk.write(json.dumps(r) + "\n")
    (out / "INSTRUCTIONS.md").write_text(INSTRUCTIONS)

    per_arm = Counter(r["arm"] for r in key)
    words = defaultdict(list)
    for r in key:
        words[r["arm"]].append(r["n_words"])
    freeze = {
        "block": "D2", "stage": "review packets",
        "registered": "research/marsc_strengthen_20260918/PREREG.md, Wave D / D2",
        "n_items": len(items), "n_pairs": len({r["pair_id"] for r in key}),
        "n_documents": len({r["doc_key"] for r in key}),
        "n_pilot_items": sum(1 for r in key if r["in_pilot"]),
        "items_per_arm": dict(per_arm),
        "words_per_arm": {k: {"mean": round(sum(v) / len(v), 2), "min": min(v), "max": max(v)}
                          for k, v in sorted(words.items())},
        "over_budget_per_arm": {k: sum(1 for r in key if r["arm"] == k and r["over_budget"])
                                for k in sorted(per_arm)},
        "blind_schema": sorted(BLIND_KEYS),
        "id_namespace": "R##### (disjoint from the H### omission-confirmation ids in the deployed database)",
        "review_order_sha256": sha256(Path(a.review_order)),
        "arms_sha256": sha256(Path(a.arms)),
        "revisions_sha256": {Path(f).name: sha256(Path(f)) for f in a.revisions},
        "artifacts": {"d2_review_blind.jsonl": sha256(out / "d2_review_blind.jsonl"),
                      "d2_review_pilot_blind.jsonl": sha256(out / "d2_review_pilot_blind.jsonl"),
                      "d2_review_key.jsonl": sha256(out / "d2_review_key.jsonl")},
        "questions": ["q1 new important content", "q2 preservation", "q3 support", "q4 redundancy", "q5 usefulness"],
        "outcome_policy": ("NOT SCORED. Two blinded reviewers plus an adjudicator are required; the "
                           "agreement-conditioned rate and the adjudicated rate must be reported separately "
                           "(d1b_adjudicate.py). The support judge is not an admissible machine substitute for "
                           "q3: it was measured at kappa 0.4727 and calls 36% of supported facts unsupported."),
        "checks_failed": fail,
        "status": "FROZEN" if not fail else "REFUSED",
    }
    (out / "d2_packets_freeze.json").write_text(json.dumps(freeze, indent=1))
    for line in fail[:20]:
        print(f"[d2] FAIL {line}", flush=True)
    print(f"[d2] {len(items)} blinded items over {freeze['n_pairs']} pairs / {freeze['n_documents']} documents; "
          f"pilot {freeze['n_pilot_items']} items", flush=True)
    print(f"[d2] words per arm {freeze['words_per_arm']}", flush=True)
    print(f"[d2] over-budget per arm {freeze['over_budget_per_arm']}", flush=True)
    print(f"[d2] status {freeze['status']} -> {out}", flush=True)
    if fail:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
