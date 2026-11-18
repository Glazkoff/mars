"""Model-free tests of the MARS package: aggregation, multi-reference maximum, evidence spans, controls, edge cases.
A deterministic stub verifier stands in for the cross-encoders."""
from __future__ import annotations

import pytest

from mars.inventory import Fact, SentenceDecomposer, parse_facts, sentence_spans
from mars.score import MarsScorer, _derangement
from mars.verifiers import LexicalVerifier, Verifier, windows


class StubVerifier(Verifier):
    """support = 1 if the hypothesis appears verbatim in the premise, else 0.2."""
    name = "stub"

    def support(self, premises, hypotheses):
        return [1.0 if h.strip() and h.strip() in p else 0.2 for p, h in zip(premises, hypotheses)]


def test_sentence_spans_cover_text():
    t = "Alice went home. Bob stayed! Carol asked why?\nDan left."
    spans = sentence_spans(t)
    assert [t[s:e].strip() for s, e in spans] == ["Alice went home.", "Bob stayed!", "Carol asked why?", "Dan left."]
    assert sentence_spans("") == []


def test_parse_facts_drops_none_and_bullets():
    assert parse_facts("- A is B.\n2) C is D.\nNONE\n\n") == ["A is B.", "C is D."]


def test_windows_cover_the_end():
    text = " ".join(f"w{i}" for i in range(1000))
    ws = windows(text, 380, 190)
    assert ws[0].split()[0] == "w0" and ws[-1].split()[-1] == "w999" and len(ws) == 5
    assert windows("a b c", 380, 190) == ["a b c"]


def test_precision_and_evidence():
    src = "Alice went home. Bob stayed at the office."
    cand = "Alice went home. Bob flew to Paris."
    sc = MarsScorer(StubVerifier(), StubVerifier(), SentenceDecomposer(), threshold=0.5)
    r = sc.score([cand], sources=[src])[0]
    assert r.n_candidate_facts == 2 and abs(r.P - 0.6) < 1e-9 and abs(r.P_min - 0.2) < 1e-9
    assert [f.text for f, _ in r.unsupported] == ["Bob flew to Paris."]
    f, _ = r.unsupported[0]
    assert cand[f.start:f.end].strip() == "Bob flew to Paris."
    assert r.R is None and r.F is None


def test_recall_takes_the_best_reference_and_lists_omissions():
    cand = "Alice went home."
    refs = [["Alice went home. Bob stayed at the office.", "Alice went home."]]
    sc = MarsScorer(StubVerifier(), StubVerifier(), SentenceDecomposer())
    r = sc.score([cand], references=refs)[0]
    assert abs(r.R - 1.0) < 1e-9 and r.omitted == [] and r.n_reference_facts == 1
    r2 = sc.score([cand], references=[[refs[0][0]]])[0]
    assert abs(r2.R - 0.6) < 1e-9 and [f.text for f, _ in r2.omitted] == ["Bob stayed at the office."]


def test_f_is_harmonic_mean_and_controls_run():
    src = "Alice went home. Bob stayed at the office."
    cands = ["Alice went home. Bob flew to Paris.", "Bob stayed at the office."]
    srcs = [src, src]
    sc = MarsScorer(StubVerifier(), StubVerifier(), SentenceDecomposer())
    res = sc.score(cands, sources=srcs, references=[[src], [src]], controls=True)
    for r in res:
        assert r.P is not None and r.R is not None and abs(r.F - 2 * r.P * r.R / (r.P + r.R)) < 1e-9
        assert set(r.controls) == {"P_shuffled_source", "R_shuffled_candidate", "R_no_candidate"}
    # no candidate at all: every reference fact is unsupported under the stub
    assert all(abs(r.controls["R_no_candidate"] - 0.2) < 1e-9 for r in res)


def test_reference_free_coverage_with_salience():
    cand = "Alice went home."
    facts = [[Fact("Alice went home.", 0, 16), Fact("Bob stayed at the office.", 17, 42)]]
    sc = MarsScorer(StubVerifier(), StubVerifier(), SentenceDecomposer())
    r = sc.score([cand], sources=["x"], salient_source_facts=facts, salience=[[3.0, 1.0]])[0]
    assert abs(r.R - (3.0 * 1.0 + 1.0 * 0.2) / 4.0) < 1e-9


def test_derangement_moves_everyone():
    perm = _derangement(7, 1)
    assert sorted(perm) == list(range(7)) and all(i != j for i, j in enumerate(perm))
    with pytest.raises(ValueError):
        _derangement(1, 1)


def test_single_item_call_never_returns_identity_shuffled_controls():
    """A one-item call has nothing to swap in: the shuffled controls must be absent, not the real inputs relabelled."""
    src = "Alice went home. Bob flew to Paris. Carol stayed at the office."
    sc = MarsScorer(StubVerifier(), StubVerifier(), SentenceDecomposer())
    with pytest.warns(UserWarning, match="at least two items"):
        r = sc.score(["Alice went home."], sources=[src], references=[[src]], controls=True)[0]
    assert r.P is not None and r.R is not None
    assert "P_shuffled_source" not in r.controls and "R_shuffled_candidate" not in r.controls
    assert "R_no_candidate" in r.controls


def test_lexical_stand_in_is_bounded():
    v = LexicalVerifier()
    assert v.support(["the quick brown fox"], ["quick fox"]) == [1.0]
    assert v.support([""], ["quick fox"]) == [0.0]


def test_empty_candidate_yields_no_scores():
    sc = MarsScorer(StubVerifier(), StubVerifier(), SentenceDecomposer())
    r = sc.score([""], sources=["Alice went home."])[0]
    assert r.P is None and r.n_candidate_facts == 0
