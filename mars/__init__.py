"""MARS: fact-level precision and recall for summaries and RAG outputs, with built-in candidate-blind controls.

    from mars import MarsScorer
    scorer = MarsScorer.default(r_glob="ckpts/marsc_seed*")  # FactCG-contract verifier for P, MARS-C for R
    out = scorer.score(candidates, sources=sources)    # MARS-P against the source; reference-free MARS-R needs
                                                       # salient_source_facts= (the package ships no salience model)
    out = scorer.score(candidates, references=refs)    # reference-based MARS-R over the reference's facts
    out[0].P, out[0].R, out[0].F, out[0].unsupported, out[0].omitted

Two axes, never collapsed by the package itself: P (faithfulness, the hallucination axis) and R (coverage, the
omission axis). F is an application-side convenience. Every score can be recomputed with the candidate replaced or
removed (`controls=True`), because recall of omissions can be won by ignoring the candidate.
"""
from .score import Fact, MarsResult, MarsScorer, score  # noqa: F401

__version__ = "0.1.0"
