"""combine() of the H-COND crossed-accuracy bar must never PASS on partial input (audit 2026-09-27, medium 6)."""
import importlib.util
import json
from pathlib import Path

import pytest

B2 = Path(__file__).resolve().parents[1] / "research/marsc_strengthen_20260918/code/b2_crossed_families.py"
spec = importlib.util.spec_from_file_location("b2_crossed_families", B2)
b2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b2)

OFF = b2.REQUIRED_OFF_THE_SHELF


def split_file(tmp: Path, split: str, families=OFF, clears=True) -> str:
    d = tmp / f"crossed_{split}"; d.mkdir()
    rep = {"margins_vs_primary": {f: {"family_is_off_the_shelf": True, "clears_bar_on_this_split": clears}
                                  for f in families},
           "families": {f: {"verdict_vs_chance": "CONDITIONS-ON-CANDIDATE"} for f in ("humanfact", *families)}}
    (d / "crossed_families.json").write_text(json.dumps(rep))
    return str(d / "crossed_families.json")


def test_both_splits_full_family_passes(tmp_path):
    r = b2.combine([split_file(tmp_path, "validation"), split_file(tmp_path, "test")])
    assert r["hcond_verdict"] == "PASS"


def test_single_split_is_incomplete_not_pass(tmp_path):
    r = b2.combine([split_file(tmp_path, "validation")])
    assert r["hcond_verdict"] == "INCOMPLETE" and r["missing_splits"] == ["test"]


def test_missing_family_is_incomplete(tmp_path):
    r = b2.combine([split_file(tmp_path, "validation", OFF[:-1]), split_file(tmp_path, "test")])
    assert r["hcond_verdict"] == "INCOMPLETE" and r["missing_families"] == [OFF[-1]]


def test_one_failing_split_fails(tmp_path):
    r = b2.combine([split_file(tmp_path, "validation"), split_file(tmp_path, "test", clears=False)])
    assert r["hcond_verdict"] == "FAIL"


def test_duplicate_split_refused(tmp_path):
    a = split_file(tmp_path, "validation")
    other = tmp_path / "x"; other.mkdir(); b = other / "crossed_validation"; b.mkdir()
    (b / "crossed_families.json").write_text(Path(a).read_text())
    with pytest.raises(SystemExit):
        b2.combine([a, str(b / "crossed_families.json")])
