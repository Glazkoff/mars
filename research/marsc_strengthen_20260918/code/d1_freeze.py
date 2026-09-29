#!/usr/bin/env python3
"""MARS-C D1 -- freeze the CURRENT-EMITTER blinded precision sample (PREREG Wave D, D1).

Defect this closes. The paper's only semantic precision evidence annotates a PRE-REPAIR emission
(`g3/emitted_validation.jsonl`, 2026-09-16 20:11). The corrected emission is
`r07_crossed/emitted_validation_k10.jsonl` (2026-09-17 15:41:22); measured top-10 overlap between the two is
0.59/0.66/0.63, so re-analysing the old labels is a selection on rank, not a measurement of the current system.

What this script does, and does NOT do. It VERIFIES and FREEZES a sample that `mc_human_sample.py` (the
registered sampling instrument, used unmodified) has already drawn. It reads no label, computes no outcome and
touches no annotator. Specifically it

  1. asserts the blinded file matches the label app's contract EXACTLY -- every line is an object with keys
     {item_id, source, summary, fact} and nothing else, no system identity anywhere;
  2. asserts no item_id collides with ANY previously issued id (g6 300 + g6 extension 100 = H001..H400 live in
     the deployed label app's sqlite database, so a collision would silently merge two studies);
  3. asserts the id format `H%03d` did not overflow past H999 (`mc_human_sample.py` formats ids as
     f"H{offset+i+1:03d}", which is not an error at 1000 -- it silently produces a 4-digit id);
  4. writes `human_sample_docmap.jsonl`, the companion the published analysis never had: item_id -> pair_id,
     doc_key, resource, summarizer system, evaluator status, rank inside the emitted list, score. `doc_key` is
     `common.doc_key(source)`, so the D1 analysis can cluster on the SOURCE DOCUMENT (the A3 repair) instead of
     resampling items inside status strata;
  5. writes `d1_freeze.json`: sha256 of every artifact, the exact argv that drew the sample, the per
     system x status cell counts, the document census, and the declared review priority order.

The review priority order is frozen HERE, before any label exists, so that a partial annotation run is a
pre-registered prefix of the sample rather than a post-hoc subset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402

BLIND_KEYS = {"item_id", "source", "summary", "fact"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(p: Path):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def emitted_rank_index(emitted_file: Path) -> dict:
    """(system, pair_id) -> ordered emitted list. The dump is written in emission order, so position IS rank."""
    idx: dict[tuple[str, str], list[dict]] = {}
    for r in load_jsonl(emitted_file):
        if r.get("emitted"):
            idx[(r["system"], r["pair_id"])] = r["emitted"]
    return idx


def recover_rank(item: dict, idx: dict) -> dict:
    """Rank of a sampled unit inside its system's emitted list (1 = top).

    Follows mc_human_rank.recover_rank -- match on text, then tighten on score -- and adds ONE disambiguator it
    lacks. A pair can carry two emitted units with identical text AND identical score (two inventory units of
    the same proposition in different source sentences); `mc_human_rank` then takes the first occurrence
    arbitrarily, which can attach the wrong evaluator status and the wrong rank to the sampled item. Measured on
    the D1 batch: 57/500 items have an ambiguous text match; 4 survive the score tie-break, and without this
    third disambiguator 2 of them attach a status that contradicts the sampler's. The key legitimately carries
    the sampled status, so it is used as the final tie-break. This resolves the ambiguity rather than hiding
    it; every count is reported in d1_freeze.json under `rank_recovery`.
    """
    em = idx.get((item["system"], item["pair_id"])) or []
    cands = [(i, e) for i, e in enumerate(em) if e["text"] == item["fact"]]
    ambiguous = len(cands) > 1
    if len(cands) > 1 and item.get("score") is not None:
        tight = [(i, e) for i, e in cands if e.get("score") is not None and abs(e["score"] - item["score"]) < 1e-9]
        if tight:
            cands = tight
    tie_broken_on_status = False
    if len(cands) > 1 and item.get("status") is not None:
        same = [(i, e) for i, e in cands if e.get("status") == item["status"]]
        if same and len(same) < len(cands):
            cands, tie_broken_on_status = same, True
    if not cands:
        return {"rank": None, "status_dump": None, "ambiguous": False, "tie_broken_on_status": False}
    i, e = cands[0]
    return {"rank": i + 1, "status_dump": e.get("status"), "ambiguous": ambiguous,
            "tie_broken_on_status": tie_broken_on_status}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind", required=True, help="human_sample_blind.jsonl written by mc_human_sample.py")
    ap.add_argument("--key", required=True, help="human_sample_key.jsonl written by mc_human_sample.py")
    ap.add_argument("--emitted", required=True, help="the corrected emitted dump the sample was drawn from")
    ap.add_argument("--acu", required=True, help="labelled ACU file supplying source/candidate per pair_id")
    ap.add_argument("--prior-keys", nargs="*", default=[], help="every previously ISSUED key file (id-collision gate)")
    ap.add_argument("--sample-argv", default="", help="the exact mc_human_sample.py command line, recorded verbatim")
    ap.add_argument("--pilot-docs", type=int, default=30, help="documents in the declared pilot prefix")
    ap.add_argument("--order-seed", type=int, default=20260918)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    blind_p, key_p = Path(a.blind), Path(a.key)
    blind = load_jsonl(blind_p)
    key = {r["item_id"]: r for r in load_jsonl(key_p)}
    pairs = {r["pair_id"]: r for r in load_jsonl(Path(a.acu))}

    fail: list[str] = []

    # --- 1. schema contract of the file the label app consumes -------------------------------------------
    for i, r in enumerate(blind):
        if set(r) != BLIND_KEYS:
            fail.append(f"blind line {i + 1}: keys {sorted(set(r))} != {sorted(BLIND_KEYS)}")
            break
        if not all(isinstance(r[k], str) and r[k] for k in BLIND_KEYS):
            fail.append(f"blind line {i + 1}: a field is empty or not a string")
            break
    # The blind file carries source/summary/fact free text, so a substring scan for system names would fire on
    # ordinary English ("the system", "distilled"). The binding guarantee is structural: the key-set assertion
    # above admits exactly {item_id, source, summary, fact}, so no system field can reach the annotator, and the
    # key file (which does carry system identity) is asserted below to be a different file that is never served.
    if blind_p.resolve() == key_p.resolve():
        fail.append("blind and key are the same file -- the sample is not blinded")

    # --- 2/3. id contract -------------------------------------------------------------------------------
    ids = [r["item_id"] for r in blind]
    if len(set(ids)) != len(ids):
        fail.append(f"blind file has duplicate item_ids ({len(ids) - len(set(ids))} duplicates)")
    if set(ids) != set(key):
        fail.append("blind and key item_id sets differ")
    bad_fmt = [i for i in ids if not (len(i) == 4 and i[0] == "H" and i[1:].isdigit())]
    if bad_fmt:
        fail.append(f"{len(bad_fmt)} item_ids are not H%03d (id-offset overflowed past H999): {bad_fmt[:5]}")
    prior: dict[str, str] = {}
    for f in a.prior_keys:
        for r in load_jsonl(Path(f)):
            prior[r["item_id"]] = f
    clash = sorted(set(ids) & set(prior))
    if clash:
        fail.append(f"{len(clash)} item_ids collide with an already-issued batch: {clash[:10]} "
                    f"(first from {prior[clash[0]]})")

    # --- 4. the document map the published analysis never had -------------------------------------------
    idx = emitted_rank_index(Path(a.emitted))
    rows, unjoined = [], []
    for iid in sorted(key):
        it = key[iid]
        pr = pairs.get(it["pair_id"])
        if pr is None:
            unjoined.append(iid)
            continue
        rk = recover_rank(it, idx)
        rows.append({"item_id": iid, "pair_id": it["pair_id"], "doc_key": common.doc_key(pr["source"]),
                     "resource": pr["resource"], "summarizer": pr["system"], "system": it["system"],
                     "status": it["status"], "score": it.get("score"), **rk})
    if unjoined:
        fail.append(f"{len(unjoined)} sampled items have no pair in the ACU file: {unjoined[:5]}")
    n_norank = sum(1 for r in rows if r["rank"] is None)
    n_ambig = sum(1 for r in rows if r["ambiguous"])
    n_tiebreak = sum(1 for r in rows if r["tie_broken_on_status"])
    mismatch = [r for r in rows if r["status_dump"] is not None and r["status_dump"] != r["status"]]
    hard = [r for r in mismatch if not r["ambiguous"]]
    if n_norank:
        fail.append(f"{n_norank} sampled items could not be joined back to the emitted dump (rank unrecoverable)")
    if hard:
        # An UNAMBIGUOUS status mismatch means the sample and the dump disagree about the same unit -- i.e. the
        # sample was not drawn from this dump. That is a real defect and must stop the freeze.
        fail.append(f"{len(hard)} sampled items disagree with the dump on evaluator status WITHOUT a text "
                    f"ambiguity to explain it: {[r['item_id'] for r in hard][:10]}")

    # --- 5. frozen review priority order ----------------------------------------------------------------
    by_doc = defaultdict(list)
    for r in rows:
        by_doc[r["doc_key"]].append(r["item_id"])
    doc_resource = {r["doc_key"]: r["resource"] for r in rows}
    rng = random.Random(a.order_seed)
    per_res = defaultdict(list)
    for d in sorted(by_doc):
        per_res[doc_resource[d]].append(d)
    for res in per_res:
        rng.shuffle(per_res[res])
    # round-robin across resources so any prefix is domain-balanced
    order: list[str] = []
    cursors = {res: 0 for res in per_res}
    while len(order) < len(by_doc):
        for res in sorted(per_res):
            c = cursors[res]
            if c < len(per_res[res]):
                order.append(per_res[res][c])
                cursors[res] = c + 1
    pilot_docs = order[:a.pilot_docs]
    pilot_items = [i for d in pilot_docs for i in sorted(by_doc[d])]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "human_sample_docmap.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (out / "review_order.json").write_text(json.dumps(
        {"order_seed": a.order_seed, "doc_order": order, "pilot_docs": pilot_docs,
         "pilot_item_ids": pilot_items, "n_docs": len(by_doc)}, indent=1))

    cells = Counter((r["system"], r["status"]) for r in rows)
    freeze = {
        "block": "D1",
        "registered": "research/marsc_strengthen_20260918/PREREG.md, Wave D / D1",
        "purpose": "blinded precision sample of the CORRECTED emitter; frozen before any label exists",
        "sample_argv": a.sample_argv,
        "inputs": {"emitted": a.emitted, "emitted_sha256": sha256(Path(a.emitted)),
                   "acu": a.acu, "acu_sha256": sha256(Path(a.acu)),
                   "prior_keys": {f: sha256(Path(f)) for f in a.prior_keys}},
        "artifacts": {"human_sample_blind.jsonl": sha256(blind_p), "human_sample_key.jsonl": sha256(key_p)},
        "n_items": len(blind), "item_id_min": min(ids) if ids else None, "item_id_max": max(ids) if ids else None,
        "n_prior_ids_checked": len(prior),
        "cells_system_x_status": {f"{s}|{st}": n for (s, st), n in sorted(cells.items())},
        "n_documents": len(by_doc), "n_pairs": len({r["pair_id"] for r in rows}),
        "resources": dict(Counter(r["resource"] for r in rows)),
        "summarizers": dict(Counter(r["summarizer"] for r in rows)),
        "rank_histogram": dict(sorted(Counter(r["rank"] for r in rows).items(), key=lambda kv: (kv[0] is None, kv[0]))),
        "rank_recovery": {
            "n_ambiguous_text_match": n_ambig, "n_tie_broken_on_status": n_tiebreak,
            "n_status_mismatch_after_tiebreak": len(mismatch),
            "n_status_mismatch_unambiguous": len(hard),
            "status_mismatch_item_ids": [r["item_id"] for r in mismatch][:20],
            "note": "An ambiguous match means the pair carries two emitted units with identical text and score. "
                    "Blinding is unaffected (the annotator sees neither rank nor status); only a rank-restricted "
                    "re-analysis is. mc_human_rank.py records this as a diagnostic and does not resolve it."},
        "items_per_document": {"min": min(map(len, by_doc.values())), "max": max(map(len, by_doc.values())),
                               "mean": sum(map(len, by_doc.values())) / max(1, len(by_doc))},
        "review_priority": {"order_seed": a.order_seed, "pilot_docs": a.pilot_docs,
                            "n_pilot_items": len(pilot_items),
                            "rule": "documents in the frozen round-robin order; any prefix is a pre-registered, "
                                    "domain-balanced subsample"},
        "analysis_contract": {
            "clustering": "bootstrap resamples doc_key clusters (A3 repair), never items inside status strata",
            "roles": "TWO blinded annotators PLUS an adjudicator; the agreement-conditioned rate is reported "
                     "beside the adjudicated rate, and the two are never conflated",
            "scorer": "research/marsc_strengthen_20260918/code/d1b_adjudicate.py"},
        "checks_failed": fail,
        "status": "FROZEN" if not fail else "REFUSED",
    }
    (out / "d1_freeze.json").write_text(json.dumps(freeze, indent=1))

    for line in fail:
        print(f"[d1] FAIL {line}", flush=True)
    print(f"[d1] {len(blind)} items, ids {freeze['item_id_min']}..{freeze['item_id_max']}, "
          f"{len(by_doc)} documents, {freeze['n_pairs']} pairs; checked against {len(prior)} previously issued ids",
          flush=True)
    for k, v in sorted(cells.items()):
        print(f"[d1] cell {k[0]:30s} {k[1]:12s} {v}", flush=True)
    print(f"[d1] pilot prefix: {a.pilot_docs} documents / {len(pilot_items)} items (order seed {a.order_seed})",
          flush=True)
    print(f"[d1] status {freeze['status']} -> {out}", flush=True)
    if fail:
        sys.exit(5)


if __name__ == "__main__":
    main()
