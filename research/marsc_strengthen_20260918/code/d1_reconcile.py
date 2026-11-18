#!/usr/bin/env python3
"""Study D1 -- reconcile the annotators' completed answer sheets with the transcribed export.

The annotators worked offline on CSV sheets (columns item_id, source, summary, fact, Q1_supported_A_or_B,
Q2_conveyed_A_or_B); a script transcribed their answers into the label platform, and every D1 estimate is computed
from the platform export (export.jsonl: 500 records per annotator and 58 adjudications). This program compares the
sheets with the export answer by answer and writes the record the manuscript needs: how many answers were checked,
which ones differ, which items are missing or duplicated, and the hashes of every file compared.

    python d1_reconcile.py --export d1/export.jsonl --key d1/human_sample_key.jsonl \
        --sheet "Annotator 1=sheets/annotator1.csv" --sheet "Annotator 2=sheets/annotator2.csv" \
        --sheet "Adjudicator=sheets/adjudicator.csv" --out d1/sheets/reconciliation.json

Exit status 0 when every answer of every sheet equals the export and nothing is missing, 1 otherwise. The sheets are
read, never written; nothing is sent anywhere. No estimate is recomputed here: if the report is clean the published
table stands as it is, and if it is not, correct the export from the sheets and rerun d1b_adjudicate.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

Q1_COLUMNS = ("Q1_supported_A_or_B", "Q1", "q1")
Q2_COLUMNS = ("Q2_conveyed_A_or_B", "Q2", "q2")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confirmed(q1: str, q2: str) -> bool:
    """A confirmed omission: supported by the source (Q1 = A) and not conveyed by the summary (Q2 = B)."""
    return q1 == "A" and q2 == "B"


def read_export(path: Path) -> dict[str, dict[str, tuple[str, str]]]:
    out: dict[str, dict[str, tuple[str, str]]] = {}
    dup = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        who, item = r["annotator"], r["item_id"]
        if item in out.setdefault(who, {}):
            dup[(who, item)] += 1
        out[who][item] = (str(r.get("q1", "")).strip().upper(), str(r.get("q2", "")).strip().upper())
    if dup:
        raise SystemExit(f"the export holds more than one record for {len(dup)} (annotator, item) pairs: {list(dup)[:5]}")
    return out


def read_sheet(path: Path) -> tuple[dict[str, tuple[str, str]], list[str], list[str]]:
    csv.field_size_limit(10 ** 9)
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rd = csv.DictReader(fh)
        cols = rd.fieldnames or []
        c1 = next((c for c in Q1_COLUMNS if c in cols), None)
        c2 = next((c for c in Q2_COLUMNS if c in cols), None)
        if "item_id" not in cols or c1 is None or c2 is None:
            raise SystemExit(f"{path}: expected the columns item_id, {Q1_COLUMNS[0]}, {Q2_COLUMNS[0]}; found {cols}")
        rows: dict[str, tuple[str, str]] = {}
        duplicates, invalid = [], []
        for r in rd:
            item = (r["item_id"] or "").strip()
            if not item:
                continue
            a = ((r[c1] or "").strip().upper(), (r[c2] or "").strip().upper())
            if item in rows:
                duplicates.append(item)
            if a[0] not in ("A", "B") or a[1] not in ("A", "B"):
                invalid.append(item)
            rows[item] = a
    return rows, duplicates, invalid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True, type=Path)
    ap.add_argument("--key", type=Path, help="human_sample_key.jsonl: the 500 item ids of the frozen sample")
    ap.add_argument("--sheet", action="append", default=[], metavar="NAME=CSV",
                    help='a completed sheet, NAME as in the export ("Annotator 1", "Annotator 2", "Adjudicator")')
    ap.add_argument("--adjudicator", default="Adjudicator")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if not a.sheet:
        ap.error("give at least one --sheet NAME=CSV")

    export = read_export(a.export)
    sample = None
    if a.key:
        sample = {json.loads(line)["item_id"] for line in a.key.read_text(encoding="utf-8").splitlines() if line.strip()}
    rep = {"export": {"path": str(a.export), "sha256": sha256(a.export),
                      "records": {k: len(v) for k, v in sorted(export.items())}},
           "sample_items": None if sample is None else len(sample), "sheets": {}}
    sheets: dict[str, dict[str, tuple[str, str]]] = {}
    clean = True
    for spec in a.sheet:
        name, _, p = spec.partition("=")
        path = Path(p)
        if name not in export:
            raise SystemExit(f"{name!r} is not an annotator of the export; it has {sorted(export)}")
        rows, duplicates, invalid = read_sheet(path)
        sheets[name] = rows
        exp = export[name]
        mismatches = [{"item_id": i, "question": q, "sheet": rows[i][k], "export": exp[i][k]}
                      for i in sorted(set(rows) & set(exp)) for k, q in enumerate(("Q1", "Q2")) if rows[i][k] != exp[i][k]]
        in_sheet_only = sorted(set(rows) - set(exp))
        in_export_only = sorted(set(exp) - set(rows))
        outside = sorted(set(rows) - sample) if sample is not None else []
        r = {"path": str(path), "sha256": sha256(path), "rows": len(rows), "export_records": len(exp),
             "items_compared": len(set(rows) & set(exp)), "answers_compared": 2 * len(set(rows) & set(exp)),
             "answers_that_differ": len(mismatches), "mismatches": mismatches,
             "duplicated_item_ids": duplicates, "rows_without_a_valid_answer": invalid,
             "items_only_in_the_sheet": in_sheet_only, "items_only_in_the_export": in_export_only,
             "items_outside_the_frozen_sample": outside}
        r["clean"] = not (mismatches or duplicates or invalid or in_sheet_only or in_export_only or outside)
        clean = clean and r["clean"]
        rep["sheets"][name] = r
        print(f"[d1-reconcile] {name}: {r['rows']} rows, {r['answers_compared']} answers compared, "
              f"{len(mismatches)} differ, {len(in_export_only)} only in the export, {len(in_sheet_only)} only in "
              f"the sheet, {len(duplicates)} duplicated, {len(invalid)} invalid -> {'clean' if r['clean'] else 'NOT CLEAN'}")

    primary = [n for n in sheets if n != a.adjudicator]
    if len(primary) == 2 and a.adjudicator in sheets:
        x, y = (sheets[n] for n in primary)
        both = sorted(set(x) & set(y))
        disagree = {i for i in both if confirmed(*x[i]) != confirmed(*y[i])}
        adj = set(sheets[a.adjudicator])
        m = {"items_both_annotators_answered": len(both), "verdict_disagreements_in_the_sheets": len(disagree),
             "adjudicated_in_the_sheet": len(adj), "disagreements_not_adjudicated": sorted(disagree - adj),
             "adjudicated_without_a_disagreement": sorted(adj - disagree),
             "verdict_matrix": {"both_confirmed": sum(1 for i in both if confirmed(*x[i]) and confirmed(*y[i])),
                                "both_rejected": sum(1 for i in both if not confirmed(*x[i]) and not confirmed(*y[i])),
                                f"{primary[0]}_only": sum(1 for i in both if confirmed(*x[i]) and not confirmed(*y[i])),
                                f"{primary[1]}_only": sum(1 for i in both if not confirmed(*x[i]) and confirmed(*y[i]))}}
        m["clean"] = not (m["disagreements_not_adjudicated"] or m["adjudicated_without_a_disagreement"])
        clean = clean and m["clean"]
        rep["adjudication_membership"] = m
        print(f"[d1-reconcile] adjudication: {len(disagree)} verdict disagreements in the sheets, {len(adj)} adjudicated "
              f"-> {'clean' if m['clean'] else 'NOT CLEAN'}; verdict matrix {m['verdict_matrix']}")
    rep["all_sheets_given"] = sorted(sheets) == sorted(export)
    rep["clean"] = clean
    rep["verdict"] = ("the sheets equal the export" if clean and rep["all_sheets_given"] else
                      "clean for the sheets given, but not every annotator's sheet was compared" if clean else
                      "the sheets and the export differ: see the mismatches")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
        print(f"[d1-reconcile] report -> {a.out}")
    print(f"[d1-reconcile] {rep['verdict']}")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
