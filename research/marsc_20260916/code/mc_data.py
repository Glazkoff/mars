#!/usr/bin/env python3
"""MARS-C E0 -- the data gate (docs/FINAL_PROPOSAL_2026-09-16.md, E0).

From the B1 pairs with HUMAN ACU labels (RoSE), build per-document fact tables: canonical facts = exact normalised ACU
text within a source; per natural summary the human coverage label of every fact (1 = covered). Then
  * exact + near-duplicate source clusters over TRAIN, VALIDATION and TEST (word 8-gram shingles, Jaccard >= 0.5);
    any cluster touching a held-out split stays out of training;
  * fixed-seed (20260916) cluster allocation of the remaining TRAIN clusters: 80 % model training, 10 % tuning,
    10 % calibration, stratified by resource;
  * tetrad eligibility: two natural summaries (not the reference, length ratio <= 1.25) with opposite coverage of two
    facts; census of eligible documents / summary pairs / crossed fact pairs per resource and role;
  * token-mask collision census (--tokenizer): opposing facts whose pooled source-token masks are identical cannot be
    told apart by a token tagger; rate over a document-balanced tetrad sample (bar: <= 5 %).
Writes docs_<role>.jsonl (train/tune/calib/validation), census.json, split_manifest.json. No model is trained.
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

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars3_20260916" / "code")); sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import m3common as m3  # noqa: E402
from m3common import gates  # noqa: E402

SEED = 20260916
_WS = re.compile(r"\s+")


def norm(t: str) -> str:
    return _WS.sub(" ", re.sub(r"[^\w\s]", " ", t.lower())).strip()


def shingles(text: str, n: int = 8) -> set[str]:
    w = norm(text).split()
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


class DSU:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.p[max(a, b)] = min(a, b)


def load_docs(b1: Path) -> dict[str, dict]:
    docs: dict[str, dict] = {}
    for split in ("train", "validation", "test"):
        for r in gates.load_jsonl(b1 / f"pairs_{split}.jsonl"):
            acus = r.get("acu_units")
            if not acus:
                continue
            k = hashlib.sha1(r["source"].encode()).hexdigest()[:12]
            d = docs.setdefault(k, {"doc_key": k, "resource": r["resource"], "source": r["source"], "splits": set(), "facts": [], "fact_index": {}, "summaries": [], "conflicts": 0})
            d["splits"].add(split)
            labels = {}
            for a in acus:
                key = norm(a["text"])
                if key not in d["fact_index"]:
                    d["fact_index"][key] = len(d["facts"]); d["facts"].append(a["text"])
                j = d["fact_index"][key]
                if j in labels and labels[j] != int(a["label"]):
                    d["conflicts"] += 1; labels[j] = None
                elif j not in labels:
                    labels[j] = int(a["label"])
            d["summaries"].append({"pair_id": r["pair_id"], "system": r["system"], "split": split, "candidate": r["candidate"],
                                   "words": len(r["candidate"].split()), "labels": labels})
    for d in docs.values():
        n = len(d["facts"])
        for s in d["summaries"]:
            s["labels"] = [s["labels"].get(j) for j in range(n)]
        d["splits"] = sorted(d["splits"]); del d["fact_index"]
    return docs


def cluster(docs: dict[str, dict], jaccard: float) -> dict[str, int]:
    keys = list(docs); idx = {k: i for i, k in enumerate(keys)}; dsu = DSU(len(keys))
    sh = {k: shingles(docs[k]["source"]) for k in keys}
    inv = defaultdict(list)
    for k in keys:
        for s in sh[k]:
            inv[s].append(k)
    seen = set()
    for s, ks in inv.items():
        if len(ks) < 2:
            continue
        for a in ks:
            for b in ks:
                if a < b and (a, b) not in seen:
                    seen.add((a, b))
                    inter = len(sh[a] & sh[b]); uni = len(sh[a] | sh[b])
                    if uni and inter / uni >= jaccard:
                        dsu.union(idx[a], idx[b])
    return {k: dsu.find(idx[k]) for k in keys}


def tetrads_of(d: dict, ratio: float, exclude_systems: set[str]) -> list[tuple[int, int, int, int]]:
    """(fact_f, fact_g, summary_i, summary_j) with f covered by i & omitted by j, g the opposite."""
    out = []
    S = [s for s in d["summaries"] if s["system"] not in exclude_systems]
    for a in range(len(S)):
        for b in range(a + 1, len(S)):
            si, sj = S[a], S[b]
            if max(si["words"], sj["words"]) > ratio * max(1, min(si["words"], sj["words"])):
                continue
            X = [f for f in range(len(d["facts"])) if si["labels"][f] == 1 and sj["labels"][f] == 0]
            Y = [f for f in range(len(d["facts"])) if si["labels"][f] == 0 and sj["labels"][f] == 1]
            ia, ib = d["summaries"].index(si), d["summaries"].index(sj)
            out.extend((f, g, ia, ib) for f in X for g in Y)
    return out


def collision_census(docs_role: list[dict], tok, max_len: int, per_doc: int, rng: random.Random) -> dict:
    """Share of sampled tetrads whose opposing facts pool onto identical token masks (tagger representation)."""
    n = coll = empty = 0; by_res = defaultdict(lambda: [0, 0])
    for d in docs_role:
        T = d["_tetrads"]
        if not T:
            continue
        cand = next((s["candidate"] for s in d["summaries"] if s["system"] != "gold"), d["summaries"][0]["candidate"])
        enc = tok(cand, d["source"], truncation="only_second", max_length=max_len, return_offsets_mapping=True)
        offs = enc["offset_mapping"]; is_src = [sid == 1 for sid in enc.sequence_ids()]
        masks = {}
        def mask(f):
            if f not in masks:
                masks[f] = m3.acu_token_mask(d["source"], d["facts"][f], offs, is_src).tobytes()
            return masks[f]
        sample = rng.sample(T, min(per_doc, len(T)))
        d["collided_pairs"] = sorted({(f, g) for f, g, _, _ in sample if mask(f) == mask(g)})
        for f, g, _, _ in sample:
            n += 1; by_res[d["resource"]][1] += 1
            if not any(mask(f)) or not any(mask(g)):
                empty += 1
            if mask(f) == mask(g):
                coll += 1; by_res[d["resource"]][0] += 1
    return {"sampled_tetrads": n, "collided": coll, "rate": coll / max(1, n), "empty_or_truncated_masks": empty,
            "by_resource": {r: {"collided": c, "sampled": t, "rate": c / max(1, t)} for r, (c, t) in by_res.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1", default=str(gates.B1_DIR))
    ap.add_argument("--out", required=True)
    ap.add_argument("--jaccard", type=float, default=0.5)
    ap.add_argument("--ratio", type=float, default=1.25)
    ap.add_argument("--exclude-systems", default="gold", help="reference summaries define the facts; never a training candidate")
    ap.add_argument("--tokenizer", default="answerdotai/ModernBERT-large")
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--collision-per-doc", type=int, default=20)
    ap.add_argument("--no-collisions", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    excl = set(a.exclude_systems.split(",")) if a.exclude_systems else set()
    docs = load_docs(Path(a.b1))
    cl = cluster(docs, a.jaccard)
    members = defaultdict(list)
    for k, c in cl.items():
        members[c].append(k)
    role_of_cluster = {}
    for c, ks in members.items():
        splits = {s for k in ks for s in docs[k]["splits"]}
        if "test" in splits:
            role_of_cluster[c] = "test"
        elif "validation" in splits:
            role_of_cluster[c] = "validation"
        else:
            h = int(hashlib.sha256(f"{SEED}|{docs[ks[0]]['resource']}|{c}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            role_of_cluster[c] = "train" if h < 0.8 else "tune" if h < 0.9 else "calib"
    rng = random.Random(SEED)
    for k, d in docs.items():
        d["cluster"] = cl[k]; d["role"] = role_of_cluster[cl[k]]
        d["_tetrads"] = tetrads_of(d, a.ratio, excl)
        d["n_tetrads"] = len(d["_tetrads"])
        d["n_eligible_pairs"] = len({(i, j) for _, _, i, j in d["_tetrads"]})
    census = {"registered": "research/marsc_20260916/PREREG.md E0", "seed": SEED, "jaccard": a.jaccard, "length_ratio": a.ratio,
              "excluded_systems": sorted(excl), "documents": len(docs), "clusters": len(members),
              "near_duplicate_clusters_gt1": sum(1 for ks in members.values() if len(ks) > 1),
              "clusters_touching_heldout_with_train_docs": sum(1 for c, ks in members.items() if role_of_cluster[c] in ("validation", "test") and any("train" in docs[k]["splits"] for k in ks)),
              "label_conflicts": sum(d["conflicts"] for d in docs.values()), "roles": {}}
    for role in ("train", "tune", "calib", "validation", "test"):
        dr = [d for d in docs.values() if d["role"] == role]
        by = defaultdict(Counter)
        for d in dr:
            by[d["resource"]]["docs"] += 1; by[d["resource"]]["summaries"] += len(d["summaries"]); by[d["resource"]]["facts"] += len(d["facts"])
            if d["n_tetrads"]:
                by[d["resource"]]["docs_with_tetrads"] += 1; by[d["resource"]]["tetrads"] += d["n_tetrads"]; by[d["resource"]]["eligible_summary_pairs"] += d["n_eligible_pairs"]
        census["roles"][role] = {"docs": len(dr), "docs_with_tetrads": sum(1 for d in dr if d["n_tetrads"]), "by_resource": {r: dict(c) for r, c in by.items()}}
    if not a.no_collisions:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(m3.snap(a.tokenizer))
        census["collisions_train"] = collision_census([d for d in docs.values() if d["role"] == "train"], tok, a.max_len, a.collision_per_doc, rng)
        census["collisions_validation"] = collision_census([d for d in docs.values() if d["role"] == "validation"], tok, a.max_len, a.collision_per_doc, rng)
    tr = census["roles"]["train"]
    per_res_ok = all(v.get("docs_with_tetrads", 0) >= 50 for v in tr["by_resource"].values())
    gate = {"eligible_training_docs_ge_200": tr["docs_with_tetrads"] >= 200, "resources_ge_3": sum(1 for v in tr["by_resource"].values() if v.get("docs_with_tetrads", 0) > 0) >= 3,
            "each_resource_ge_50": per_res_ok, "collision_rate_le_0.05": (census.get("collisions_train", {}).get("rate", 0.0) <= 0.05)}
    gate["PASS"] = all(gate.values()); census["gate"] = gate
    for role in ("train", "tune", "calib", "validation", "test"):
        with open(out / f"docs_{role}.jsonl", "w", encoding="utf-8") as fh:
            for d in docs.values():
                if d["role"] != role:
                    continue
                rec = {k: v for k, v in d.items() if not k.startswith("_")}
                rec["tetrads"] = d["_tetrads"][:2000]      # enumerated list capped; the sampler re-derives from labels anyway
                fh.write(json.dumps(rec) + "\n")
    manifest = {"seed": SEED, "roles": {role: sorted(d["doc_key"] for d in docs.values() if d["role"] == role) for role in ("train", "tune", "calib", "validation", "test")}}
    json.dump(manifest, open(out / "split_manifest.json", "w"))
    json.dump(census, open(out / "census.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in census.items() if k != "roles"}, indent=1))
    for role, v in census["roles"].items():
        print(f"[e0] {role:11s} docs {v['docs']:4d} with tetrads {v['docs_with_tetrads']:4d} " + " ".join(f"{r}:{c.get('docs_with_tetrads', 0)}/{c['docs']}" for r, c in v["by_resource"].items()), flush=True)


if __name__ == "__main__":
    main()
