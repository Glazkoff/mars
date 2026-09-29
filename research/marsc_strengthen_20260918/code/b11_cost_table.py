#!/usr/bin/env python3
"""B11 -- the cost/latency table MARS_PUBLICATION_PLAN_2026-09-20.md Sec. 3.2 asks for (Paper B).

"Access/PeerJ reviewers weigh practicality; this is where a small verifier wins outright." So the table has
to put MARS-C's own verifier, the off-the-shelf families already measured in B6/B1, and the three arms added
this week (the Gemma-4-31B judge, FactCG-DeBERTa-L, Bespoke-MiniCheck-7B) on ONE per-unit axis.

Every row was measured on this node rather than quoted from a paper, but the rows do NOT all come from a
machine-readable artefact, and the difference is marked in each row's `source` field:
  * MARS-C's verifier is read from `b6b8/b6/stage_times_gpu.json` (s_per_unit) at run time;
  * the arms added this week are read from the receipt each scoring job writes beside its scores
    (`*.receipt.json`, `units_per_s`), so those rows cannot exist without the run that produced them;
  * the seven B1/B6 off-the-shelf timings in `B1_MS_PER_UNIT` below are TRANSCRIBED from the B1/B6 campaign
    write-up (RESULTS.md Sec. 3.8-3.9). They are measurements from this node, but this file re-types them
    rather than reading them, so they carry no provenance of their own and a change upstream will not
    propagate. Treat them as a citation, not as an artefact, and re-derive them before publication.

MARS-C's deployed cost is its verifier x the number of seeds in the arm of record (3), because the emitter
of record averages three seeds; the single-seed figure is reported beside it rather than instead of it. The
MARS-C rows are filtered to the `humanfact` label source and reported PER INVENTORY, because b6 measures
both label sources over both inventories and averaging the lot silently prices MARS-C using the twin it is
being compared against.

USD is derived, not measured: --usd-per-gpu-hour times the measured GPU seconds. The default is 0, which
prints the compute time only -- pass the number your accounting actually uses rather than inheriting a guess.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Measured in B1/B6 on this node and reported in RESULTS.md Sec. 3.8-3.9, in ms/unit.
B1_MS_PER_UNIT = {
    "lex:token_recall": 0.018,
    "emb:minilm_maxcos": 0.164,
    "lex:rougeL_recall": 0.166,
    "nli:nli (windowed RoBERTa-MNLI)": 1.014,
    "nli:alignscore": 1.135,
    "nli:summac_zs": 3.379,
    "nli:minicheck (Flan-T5-Large)": 30.013,
}


# Directory-name markers for runs that have been WITHDRAWN (superseded by a rerun after a bug fix). Their
# receipts are kept for the record but must never enter the table: rglob would otherwise pick them up and
# emit two rows for the same (system, mode, split, inventory) with different costs.
WITHDRAWN_MARKERS = ("withdrawn", "stale", "_preboot_fix", "_wholedoc")


def receipts(root: Path) -> list[dict]:
    out, skipped = [], 0
    for p in sorted(root.rglob("*.receipt.json")):
        if any(m in part for part in p.parts for m in WITHDRAWN_MARKERS):
            skipped += 1
            continue
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if r.get("units_per_s"):
            r["_path"] = str(p)
            out.append(r)
    if skipped:
        print(f"[b11] skipped {skipped} receipts under withdrawn/stale directories")
    # a duplicate (system, mode, split, unit_set) means two live runs claim the same row -- refuse rather
    # than silently print both, which is how the withdrawn double-BOS rows first got in
    seen = {}
    for r in out:
        uf = r.get("units_file") or ""
        split = "test" if "_test" in uf else ("validation" if "_validation" in uf else "?")
        k = (r.get("system"), r.get("candidate_mode"), split, r.get("unit_set"))
        if k in seen:
            raise SystemExit(f"[b11] ABORT: two live receipts for {k}:\n  {seen[k]}\n  {r['_path']}")
        seen[k] = r["_path"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--marsc-strengthen", default="/home/user/results/marsc_strengthen")
    ap.add_argument("--stage-times", default="/home/user/results/marsc_strengthen/b6b8/b6/stage_times_gpu.json")
    ap.add_argument("--seeds", type=int, default=3, help="seeds in the MARS-C arm of record")
    ap.add_argument("--label-source", default="humanfact",
                    help="which b6 label source the MARS-C rows are priced from; 'judgefact' is the "
                         "judge-supervised twin and must never be blended into a MARS-C row")
    ap.add_argument("--usd-per-gpu-hour", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows: list[dict] = []

    # ---- MARS-C's own verifier, from the B6 stage times.
    # b6_cost keys every measurement "<label_source>|<inventory>|seed<n>", and runs BOTH label sources over
    # BOTH inventories: 2 x 2 x 3 = 12 entries. Averaging all twelve -- which an earlier version of this file
    # did -- blends MARS-C's humanfact verifier with the JUDGE-SUPERVISED judgefact twin it is being compared
    # against, and blends the two inventories, which have genuinely different per-unit costs. It produced
    # 1.071 ms/unit for a row labelled "MARS-C verifier" where the project's own RESULTS.md Sec. 3.8 reports
    # 1.028 -- two values for one quantity in one submission. b6's canonical accessor `verif_rate` filters
    # on "<label_source>|<inventory>|", so this does the same and emits ONE ROW PER INVENTORY.
    try:
        st = json.loads(Path(a.stage_times).read_text())
        ver = st.get("stages", {}).get("verification", {})
        for inv in ("gemma+ent", "distilled"):
            pref = f"{a.label_source}|{inv}|"
            per_unit = [v["s_per_unit"] for k, v in ver.items()
                        if k.startswith(pref) and isinstance(v, dict) and "s_per_unit" in v]
            if not per_unit:
                continue
            one = sum(per_unit) / len(per_unit)
            rows.append({"system": f"MARS-C verifier / {inv} (1 seed)", "ms_per_unit": one * 1000.0,
                         "unit_set": inv, "source": f"b6/stage_times_gpu.json [{pref}*]",
                         "n_measurements": len(per_unit)})
            rows.append({"system": f"MARS-C emitter of record / {inv} ({a.seeds} seeds)",
                         "ms_per_unit": one * 1000.0 * a.seeds, "unit_set": inv,
                         "source": f"b6/stage_times_gpu.json [{pref}*] x {a.seeds} seeds",
                         "n_measurements": len(per_unit)})
    except Exception as exc:
        print(f"[b11] WARNING no stage times ({exc})")

    for name, ms in B1_MS_PER_UNIT.items():
        rows.append({"system": name, "ms_per_unit": ms, "source": "B1/B6 measured (RESULTS.md 3.8-3.9)"})

    # ---- the arms added this week, from their own receipts
    for r in receipts(Path(a.marsc_strengthen)):
        uf = r.get("units_file") or ""
        split = "test" if "_test" in uf else ("validation" if "_validation" in uf else "?")
        rows.append({"system": r.get("system", r.get("model", "?")),
                     "ms_per_unit": 1000.0 / r["units_per_s"],
                     "units_per_s": r["units_per_s"], "n_units": r.get("n_units"),
                     "candidate_mode": r.get("candidate_mode"), "unit_set": r.get("unit_set"),
                     "split": split, "source": r["_path"].rsplit("/", 1)[-1]})

    for r in rows:
        ms = r["ms_per_unit"]
        r["s_per_1k_units"] = ms
        if a.usd_per_gpu_hour:
            r["usd_per_1k_units"] = ms / 1000.0 / 3600.0 * 1000.0 * a.usd_per_gpu_hour

    rows.sort(key=lambda r: r["ms_per_unit"])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"usd_per_gpu_hour": a.usd_per_gpu_hour, "rows": rows}, indent=1))

    w = max(len(str(r["system"])) for r in rows)
    print(f"{'system'.ljust(w)}  ms/unit   s/1k units  mode      split       unit_set    n_units")
    for r in rows:
        print(f"{str(r['system']).ljust(w)}  {r['ms_per_unit']:8.3f}  {r['s_per_1k_units']:9.2f}  "
              f"{str(r.get('candidate_mode') or '-'):8s}  {str(r.get('split') or '-'):10s}  "
              f"{str(r.get('unit_set') or '-'):10s}  {r.get('n_units') or '-'}")
    print(f"[b11] wrote {out}")


if __name__ == "__main__":
    main()
