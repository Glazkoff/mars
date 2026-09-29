#!/usr/bin/env python3
"""MARS-C E2 -- the frozen source-only IMPORTANCE policy (docs/FINAL_PROPOSAL, "Importance is separate from coverage").

Training signal (TRAIN role only): an inventory unit is reference-worthy if it aligns (same source sentence + shared
content token) to any HUMAN reference ACU of its document -- the reference defines what a summary should keep, never what
a given summary omitted. Emits training rows for the source-only tagger (unit_labels = 1 - important, so the tagger's
"omission" logit is importance) plus the tune-role units file used to choose the threshold. Gold ACUs never enter the
deployed emitter: the learned policy is applied identically to every system at evaluation.
  build    -> imp_rows_train.jsonl, units_<inventory>_tune.jsonl
  combine  -> filtered emitter scores: omission score kept where importance >= tau, pushed below every kept unit otherwise
  choose   -> tau maximising emitted recall@k on the TUNE documents (human omissions), reported per candidate tau
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402


def aligned_to_reference(source: str, unit: dict, facts: list[str], spans: list[tuple[int, int]]) -> bool:
    us, ue = unit.get("start", -1), unit.get("end", -1); ut = set(common.tokens(unit["text"]))
    for f, (s, e) in zip(facts, spans):
        if us < e and ue > s and (set(common.tokens(f, content_only=True)) or set(common.tokens(f))) & ut:
            return True
    return False


def build(a) -> None:
    e0 = Path(a.data_dir)
    role_of = {}
    for role in ("train", "tune", "calib"):
        for d in common.load_jsonl(e0 / f"docs_{role}.jsonl"):
            role_of[d["doc_key"]] = (role, d)
    inv_rows = common.load_jsonl(Path(a.inventory_rows))          # M32b rows: pair_id, source, candidate, units (gemma+ent)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n = {"train": 0, "tune": 0}; pos = 0; tot = 0; seen_docs = set()
    with open(out / "imp_rows_train.jsonl", "w", encoding="utf-8") as ftr, open(out / f"units_{a.inventory_name}_tune.jsonl", "w", encoding="utf-8") as ftu, open(out / "acu_units_tune.jsonl", "w", encoding="utf-8") as facu:
        for r in inv_rows:
            k = hashlib.sha1(r["source"].encode()).hexdigest()[:12]
            if k not in role_of:
                continue
            role, d = role_of[k]
            if role == "train":
                if k in seen_docs:
                    continue                                     # one importance row per document (source-only)
                seen_docs.add(k)
                spans = [common.best_sentence_span(d["source"], f) for f in d["facts"]]
                labels = [1 - int(aligned_to_reference(d["source"], u, d["facts"], spans)) for u in r["units"]]   # 0 = important
                pos += labels.count(0); tot += len(labels)
                ftr.write(json.dumps({"pair_id": f"imp:{k}", "cls": "natural", "resource": r["resource"], "system": "importance", "split": "train",
                                      "source": r["source"], "candidate": "", "units": r["units"], "unit_labels": labels, "consistency_label": None, "training_eligible": True}) + "\n")
                n["train"] += 1
            elif role == "tune":
                acus = next((s for s in d["summaries"] if s["pair_id"] == r["pair_id"]), None)
                if acus is None:
                    continue
                ftu.write(json.dumps({"pair_id": r["pair_id"], "resource": r["resource"], "system": r["system"], "split": "tune", "source": r["source"],
                                      "candidate": r["candidate"], "units": r["units"]}) + "\n")
                facts = [(f, lab) for f, lab in zip(d["facts"], acus["labels"]) if lab is not None]
                facu.write(json.dumps({"pair_id": r["pair_id"], "resource": r["resource"], "system": r["system"], "split": "tune", "source": r["source"],
                                       "candidate": r["candidate"], "units": [{"kind": "acu", "text": f} for f, _ in facts], "unit_labels": [int(lab) for _, lab in facts]}) + "\n")
                n["tune"] += 1
    man = {"train_docs": n["train"], "tune_pairs": n["tune"], "important_fraction": pos / max(1, tot), "units": tot}
    json.dump(man, open(out / "importance_manifest.json", "w"), indent=1); print(f"[imp] {json.dumps(man)}", flush=True)


def combine(a) -> None:
    """Filtered emitter: keep omission scores of units with importance >= tau; push the rest below every kept unit."""
    import glob
    acc = {}
    for f in sorted(glob.glob(a.importance_glob)):
        _, sc = common.read_scores(Path(f))
        for p, v in sc.items():
            acc.setdefault(p, []).append(v)
    imp = {p: np.nanmean(np.vstack(v), axis=0) for p, v in acc.items()}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for f in sorted(glob.glob(a.omission_glob)):
        name, sc = common.read_scores(Path(f)); per = {}
        for p, v in sc.items():
            if p not in imp or len(imp[p]) != len(v):
                continue
            keep = imp[p] >= a.tau
            per[p] = [float(v[i]) if keep[i] else float(v[i]) - 10.0 for i in range(len(v))]
        common.write_scores(out / Path(f).name.replace(".jsonl", f"_imp{a.tau:g}.jsonl"), f"{name}+imp{a.tau:g}", "units", per)
    print(f"[imp] combined tau={a.tau}: {len(imp)} pairs with importance", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--data-dir", required=True); b.add_argument("--inventory-rows", required=True); b.add_argument("--inventory-name", default="gemma+ent"); b.add_argument("--out", required=True)
    c = sub.add_parser("combine"); c.add_argument("--omission-glob", required=True); c.add_argument("--importance-glob", required=True); c.add_argument("--tau", type=float, required=True); c.add_argument("--out", required=True)
    a = ap.parse_args()
    {"build": build, "combine": combine}[a.cmd](a)


if __name__ == "__main__":
    main()
