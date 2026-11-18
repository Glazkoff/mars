"""The shuffled controls of `mars.score` are cross-document controls (full review of 2026-09-29, R3-W1): a row
shuffle is not enough when several candidates summarise one source."""
import warnings

import pytest

from mars.score import MarsScorer, _cross_document_partners
from mars.verifiers import LexicalVerifier


class Recording(LexicalVerifier):
    """Lexical support that remembers every premise it was handed, call by call."""

    def __init__(self):
        self.calls = []

    def support(self, premises, hypotheses):
        self.calls.append(list(premises))
        return super().support(premises, hypotheses)


SRC_A = "The council approved the harbour budget on Monday. The mayor opposed the measure."
SRC_B = "Engineers reopened the northern bridge after repairs. Traffic resumed within hours."
SRC_C = "The museum acquired three medieval manuscripts. Curators plan an autumn exhibition."


def test_two_candidates_of_one_source_get_no_shuffled_control():
    p, r = Recording(), Recording()
    with pytest.warns(UserWarning, match="no item of another document"):
        out = MarsScorer(p, r).score(["The council approved the budget.", "The mayor opposed the measure."],
                                     sources=[SRC_A, SRC_A], references=[["The council approved the budget."]] * 2,
                                     controls=True)
    for res in out:
        assert "P_shuffled_source" not in res.controls and "R_shuffled_candidate" not in res.controls
        assert res.control_info["P_shuffled_source"] == {"available": False, "reason": "no other document in the batch"}
        assert "R_no_candidate" in res.controls
    assert len(p.calls) == 1 and set(p.calls[0]) == {SRC_A}              # the real source never served as a control
    assert len(r.calls) == 2 and set(r.calls[1]) == {""}                 # R and R_no_candidate; no shuffled call ran


def test_repeated_sources_are_paired_across_documents():
    cands = ["The council approved the budget.", "The mayor opposed the measure.",
             "Engineers reopened the bridge.", "Traffic resumed within hours.",
             "The museum acquired manuscripts.", "Curators plan an exhibition."]
    sources = [SRC_A, SRC_A, SRC_B, SRC_B, SRC_C, SRC_C]
    p, r = Recording(), Recording()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = MarsScorer(p, r).score(cands, sources=sources, references=[[c] for c in cands], controls=True)
    for i, res in enumerate(out):
        for key in ("P_shuffled_source", "R_shuffled_candidate"):
            info = res.control_info[key]
            assert info["available"] and sources[info["partner"]] != sources[i]
            assert key in res.controls
    shuffled_sources = p.calls[1]                                        # one premise per candidate fact, in row order
    assert all(a != b for a, b in zip(p.calls[0], shuffled_sources))
    partners = [res.control_info["R_shuffled_candidate"]["partner"] for res in out]
    assert sorted(partners) == list(range(len(cands)))                   # every candidate is used once


def test_doc_ids_override_the_text_and_groups_stratify():
    docs = ["d1", "d1", "d2", "d2", "d3", "d3"]
    groups = ["bart", "gpt", "bart", "gpt", "bart", "gpt"]
    partner, reason = _cross_document_partners(docs, groups, seed=7)
    for i, j in enumerate(partner):
        assert reason[i] is None and docs[j] != docs[i] and groups[j] == groups[i]
    v = LexicalVerifier()
    out = MarsScorer(v, v).score(["a claim here"] * 6, sources=["one shared source text"] * 6, controls=True,
                                 doc_ids=docs, groups=groups)
    assert all(r.control_info["P_shuffled_source"]["available"] for r in out)


def test_group_without_a_second_document_is_reported():
    partner, reason = _cross_document_partners(["d1", "d2", "d1"], ["bart", "bart", "gpt"], seed=1)
    assert partner[2] is None and reason[2] == "no other document in group 'gpt'"
    assert partner[0] == 1 and partner[1] == 0


def test_dominant_document_still_gets_cross_document_partners():
    docs = ["d1", "d1", "d1", "d2"]
    partner, reason = _cross_document_partners(docs, None, seed=3)
    assert all(r is None for r in reason)
    assert all(docs[j] != docs[i] for i, j in enumerate(partner))


def test_one_document_batch_and_single_item():
    v = LexicalVerifier()
    with pytest.warns(UserWarning):
        out = MarsScorer(v, v).score(["The council approved the budget."], sources=[SRC_A],
                                     references=[["The council approved the budget."]], controls=True)
    assert set(out[0].controls) == {"R_no_candidate"}
    assert not out[0].control_info["R_shuffled_candidate"]["available"]
