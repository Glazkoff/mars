"""Package API contracts named in the 2026-09-28 review (F09): no silent verifier substitution, and the convenience
function routes scoring keywords to the scoring call."""
import pytest

import mars
from mars.score import MarsScorer
from mars.verifiers import LexicalVerifier


def test_default_refuses_silent_r_fallback():
    with pytest.raises(ValueError, match="r_glob"):
        MarsScorer.default()


def test_convenience_routes_controls(monkeypatch):
    lex = LexicalVerifier()
    monkeypatch.setattr(MarsScorer, "default", classmethod(lambda cls, **kw: cls(lex, lex)))
    out = mars.score(["The cat sat on the mat.", "Dogs bark loudly."],
                     sources=["The cat sat on the mat. It was warm.", "Dogs bark loudly at night."],
                     references=[["The cat sat."], ["Dogs bark."]], controls=True)
    assert "R_shuffled_candidate" in out[0]["controls"] and "P_shuffled_source" in out[0]["controls"]
