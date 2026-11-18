"""The D1 sheet-to-export reconciliation (research/marsc_strengthen_20260918/code/d1_reconcile.py)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "marsc_strengthen_20260918" / "code"))
import d1_reconcile  # noqa: E402

ANSWERS = {"Annotator 1": {"H1": ("A", "B"), "H2": ("A", "A"), "H3": ("A", "B")},
           "Annotator 2": {"H1": ("A", "B"), "H2": ("A", "B"), "H3": ("A", "B")},
           "Adjudicator": {"H2": ("A", "B")}}


def _write(tmp: Path, answers=ANSWERS) -> tuple[Path, list[str]]:
    with open(tmp / "export.jsonl", "w") as fh:
        for who, items in ANSWERS.items():
            for item, (q1, q2) in items.items():
                fh.write(json.dumps({"annotator": who, "item_id": item, "q1": q1, "q2": q2}) + "\n")
    specs = []
    for who, items in answers.items():
        p = tmp / f"{who.replace(' ', '_')}.csv"
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["item_id", "source", "summary", "fact", "Q1_supported_A_or_B", "Q2_conveyed_A_or_B"])
            for item, (q1, q2) in items.items():
                w.writerow([item, "s", "c", "f", q1, f" {q2.lower()} "])
        specs += ["--sheet", f"{who}={p}"]
    return tmp / "export.jsonl", specs


def test_identical_sheets_are_clean(tmp_path: Path):
    export, specs = _write(tmp_path)
    out = tmp_path / "report.json"
    assert d1_reconcile.main(["--export", str(export), *specs, "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["clean"] and rep["all_sheets_given"]
    assert rep["sheets"]["Annotator 1"]["answers_compared"] == 6
    assert rep["adjudication_membership"]["verdict_disagreements_in_the_sheets"] == 1
    assert rep["adjudication_membership"]["verdict_matrix"]["both_confirmed"] == 2


def test_a_changed_answer_and_a_missing_row_are_reported(tmp_path: Path):
    changed = {k: dict(v) for k, v in ANSWERS.items()}
    changed["Annotator 2"]["H3"] = ("B", "B")
    del changed["Annotator 1"]["H1"]
    export, specs = _write(tmp_path, changed)
    out = tmp_path / "report.json"
    assert d1_reconcile.main(["--export", str(export), *specs, "--out", str(out)]) == 1
    rep = json.loads(out.read_text())
    assert rep["sheets"]["Annotator 2"]["mismatches"] == [
        {"item_id": "H3", "question": "Q1", "sheet": "B", "export": "A"}]
    assert rep["sheets"]["Annotator 1"]["items_only_in_the_export"] == ["H1"]
    assert not rep["clean"]
