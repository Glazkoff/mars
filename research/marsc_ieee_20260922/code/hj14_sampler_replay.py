#!/usr/bin/env python3
"""H-J14 -- which training cells the R04 twins drew (registered in research/marsc_ieee_20260922/PREREG.md, Amendment H-J14).

CPU only, no model. The `Sampler` of mc_train.py is imported unchanged and the draws of both arms are replayed for
both label sets and the three seeds of record. A cell is (doc_key, pair_id, fact index); its target is 1 = omitted.

    python hj14_sampler_replay.py --human ~/results/marsc/e0 --judge ~/results/marsc/r04_matched/e0_judge \
        --models ~/results/marsc/r04_matched/models --out ~/results/marsc_strengthen/hj14
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "marsc_20260916" / "code"))
import mc_train as mc  # noqa: E402

SEEDS = (20260906, 20260907, 20260908)
STEPS, T, N, K = 600, 2, 4, 6                       # the defaults of mc_train.py, which every R04 run used
K_R = max(1, round((4 * T + N * K) / (2 * T + N)))  # arm R matches the cell count of arm A with natural cells only


def replay(data_dir: Path, arm: str, seed: int) -> tuple[list, list, dict]:
    """The encodings and cells of one run, in draw order. cells: (doc_key, pair_id, fact, target)."""
    docs = mc.load_role(data_dir, "train", None)
    s = mc.Sampler(docs, seed)
    encs, cells = [], []
    for _ in range(STEPS):
        if arm == "R":
            for _ in range(2 * T + N):
                d, su, cs = s.natural(K_R)
                encs.append((d["doc_key"], su["pair_id"], tuple(f for f, _ in cs)))
                cells += [(d["doc_key"], su["pair_id"], f, lab) for f, lab in cs]
        else:
            for _ in range(T):
                d, si, sj, f, g = s.tetrad()
                encs.append((d["doc_key"], si["pair_id"], (f, g)))
                encs.append((d["doc_key"], sj["pair_id"], (f, g)))
                cells += [(d["doc_key"], si["pair_id"], f, 0), (d["doc_key"], si["pair_id"], g, 1),
                          (d["doc_key"], sj["pair_id"], f, 1), (d["doc_key"], sj["pair_id"], g, 0)]
            for _ in range(N):
                d, su, cs = s.natural(K)
                encs.append((d["doc_key"], su["pair_id"], tuple(f for f, _ in cs)))
                cells += [(d["doc_key"], su["pair_id"], f, lab) for f, lab in cs]
    pool = {"train_docs": len(s.docs), "tetrad_docs": len(s.tet_docs),
            "tetrad_groups": sum(len(d["_groups"]) for d in s.tet_docs)}
    return encs, cells, pool


def compare(h_cells: list, j_cells: list) -> dict:
    hk = [c[:3] for c in h_cells]
    jk = [c[:3] for c in j_cells]
    same_pos = sum(1 for a, b in zip(hk, jk) if a == b)
    identical = len(hk) == len(jk) and same_pos == len(hk)
    ch, cj = Counter(hk), Counter(jk)
    common = sum((ch & cj).values())
    out = {"cells_human": len(hk), "cells_judge": len(jk), "identical_sequence": identical,
           "same_cell_at_same_position": same_pos,
           "cells_in_common_multiset": common,
           "share_of_human_cells_also_drawn_by_judge": common / max(1, len(hk)),
           "distinct_cells_human": len(ch), "distinct_cells_judge": len(cj),
           "distinct_cells_in_common": len(set(ch) & set(cj))}
    if identical:
        diff = sum(1 for a, b in zip(h_cells, j_cells) if a[3] != b[3])
        out["targets_that_differ"] = diff
        out["share_of_targets_that_differ"] = diff / max(1, len(hk))
        out["omitted_rate_human"] = sum(c[3] for c in h_cells) / max(1, len(hk))
        out["omitted_rate_judge"] = sum(c[3] for c in j_cells) / max(1, len(jk))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--models", default="", help="directory of the twelve R04 checkpoints (result.json receipts)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"registered": "research/marsc_ieee_20260922/PREREG.md, Amendment H-J14",
           "python": sys.version.split()[0], "steps": STEPS, "tetrads_per_step": T, "natural_per_step": N,
           "facts_per_natural": K, "facts_per_natural_arm_R": K_R, "human_data": str(a.human),
           "judge_data": str(a.judge), "runs": {}}
    for arm in ("A", "R"):
        for seed in SEEDS:
            he, hc, hp = replay(Path(a.human), arm, seed)
            je, jc, jp = replay(Path(a.judge), arm, seed)
            r = compare(hc, jc)
            r.update({"encodings_human": len(he), "encodings_judge": len(je),
                      "identical_encodings": he == je, "pool_human": hp, "pool_judge": jp})
            rep["runs"][f"arm{arm}_seed{seed}"] = r
            print(f"[hj14] arm {arm} seed {seed}: identical={r['identical_sequence']} "
                  f"cells {r['cells_human']}/{r['cells_judge']} common {r['share_of_human_cells_also_drawn_by_judge']:.3f} "
                  f"tetrad docs {hp['tetrad_docs']}/{jp['tetrad_docs']}", flush=True)
    rep["arm_R_identical_for_every_seed"] = all(rep["runs"][f"armR_seed{s}"]["identical_sequence"] for s in SEEDS)
    rep["arm_A_identical_for_any_seed"] = any(rep["runs"][f"armA_seed{s}"]["identical_sequence"] for s in SEEDS)
    (out / "sampler_replay.json").write_text(json.dumps(rep, indent=1))
    if a.models:
        rec = {}
        for f in sorted(glob.glob(str(Path(a.models) / "mc-*_seed*" / "result.json"))):
            r = json.load(open(f))
            rec[Path(f).parent.name] = {"cost": r.get("cost"), "train_docs": r.get("train_docs"),
                                        "tetrad_docs": r.get("tetrad_docs")}
        (out / "training_receipts.json").write_text(json.dumps(rec, indent=1))
    print(f"[hj14] arm R identical for every seed: {rep['arm_R_identical_for_every_seed']}; "
          f"arm A identical for any seed: {rep['arm_A_identical_for_any_seed']} -> {out}", flush=True)


if __name__ == "__main__":
    main()
