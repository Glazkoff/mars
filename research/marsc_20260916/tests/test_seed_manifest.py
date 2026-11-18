#!/usr/bin/env python3
"""Regression test for the MARS-C consensus-aggregation defect (FULL_REVIEW_2026-09-17 item 1).

The shipped G1/G7 jobs fed `*_seed*_div1.jsonl` to the evaluator. That glob matched the three per-seed files
AND the `*_seedavg_div1.jsonl` pseudo-seed written by mc_variants.py seedavg, so the advertised "three-seed
consensus" averaged four files, one of which already contained the other three. These tests pin the two
guards that make that impossible: --strict-seeds (a multi-file roster must be distinct numeric seeds) and
the explicit comma-separated manifest form.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
sys.path.insert(0, str(HERE.parent.parent / "mars2_gates_20260916" / "code"))

import mc_e2_eval as E  # noqa: E402


def _touch(d: Path, names) -> list[str]:
    for n in names:
        (d / n).write_text("{}\n")
    return [str(d / n) for n in names]


SEEDS = ["gemma+ent_humanfact_seed20260906_div1.jsonl",
         "gemma+ent_humanfact_seed20260907_div1.jsonl",
         "gemma+ent_humanfact_seed20260908_div1.jsonl"]
PSEUDO = "gemma+ent_humanfact_seedavg_div1.jsonl"


def test_glob_still_sweeps_in_the_pseudo_seed(tmp_path):
    """The historical failure mode: the glob matches four files, not three."""
    _touch(tmp_path, SEEDS + [PSEUDO])
    files = E.resolve_files(str(tmp_path / "gemma+ent_humanfact_seed*_div1.jsonl"))
    assert len(files) == 4, "the defect this test guards has changed shape"


def test_strict_seeds_rejects_the_pseudo_seed(tmp_path):
    files = _touch(tmp_path, SEEDS + [PSEUDO])
    with pytest.raises(AssertionError, match="pseudo-seed token"):
        E.check_seeds("humanfact+div1", files, strict=True)


def test_strict_seeds_accepts_three_distinct_seeds(tmp_path):
    files = _touch(tmp_path, SEEDS)
    info = E.check_seeds("humanfact+div1", files, strict=True)
    assert info["n_files"] == 3
    assert info["distinct_numeric_seeds"] == ["20260906", "20260907", "20260908"]


def test_strict_seeds_rejects_a_repeated_seed(tmp_path):
    d2 = tmp_path / "dup"; d2.mkdir()
    files = _touch(tmp_path, SEEDS) + _touch(d2, SEEDS[:1])
    with pytest.raises(AssertionError, match="distinct seeds"):
        E.check_seeds("humanfact+div1", files, strict=True)


def test_single_pseudo_seed_file_is_allowed(tmp_path):
    """The single-model regime is one seed-averaged file and must stay legal."""
    files = _touch(tmp_path, [PSEUDO])
    assert E.check_seeds("humanfact.single+div1", files, strict=True)["n_files"] == 1


def test_explicit_manifest_resolves_exactly_the_listed_files(tmp_path):
    files = _touch(tmp_path, SEEDS + [PSEUDO])
    got = E.resolve_files(",".join(files[:3]))
    assert got == files[:3]
    assert E.check_seeds("humanfact+div1", got, strict=True)["n_files"] == 3


def test_explicit_manifest_refuses_a_missing_path(tmp_path):
    files = _touch(tmp_path, SEEDS)
    with pytest.raises(AssertionError, match="do not exist"):
        E.resolve_files(",".join(files + [str(tmp_path / "gemma+ent_humanfact_seed20260909_div1.jsonl")]))


def test_explicit_manifest_refuses_a_repeated_path(tmp_path):
    files = _touch(tmp_path, SEEDS)
    with pytest.raises(AssertionError, match="repeats a path"):
        E.resolve_files(",".join(files + files[:1]))
