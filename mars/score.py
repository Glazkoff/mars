"""The two-axis score.

MARS-P(c | S)   = mean over the candidate's facts f of  q(f | S)          faithfulness, the hallucination axis
MARS-R(c | R)   = max over references r of  mean over r's facts g of  q(g | c)   coverage, the omission axis
MARS-R(c | S, w) = sum_i w_i q(g_i | c) / sum_i w_i  over salient source facts   reference-free coverage
MARS-F          = harmonic mean of P and R, application-side only

q is a verifier's support score in [0, 1] (a sigmoid/softmax output, not a calibrated probability). Evidence: the candidate facts with q below the threshold (unsupported) and
the reference facts with q below the threshold (omitted), each with the span of the sentence it came from.
Controls (`controls=True`): P with the source replaced by the source of ANOTHER DOCUMENT; R with the candidate
replaced by the candidate of another document, and with no candidate. Documents are told apart by `doc_ids`; without
them the source text names the document, then the reference list, then the salient facts, so two candidates of one
source are never each other's control. `groups` (for example the system that wrote each candidate) restricts the
candidate control to partners of the same group. An item with no valid partner gets no shuffled control and
`control_info` says why. A score that survives its controls is measuring the candidate.

Scope. This package is the scoring core. The article's configuration additionally fixes the fact decomposer (the
Gemma-4-31B sentence decomposition or its distilled segmenter; the default here is plain sentences), the MARS-C
checkpoints for R, and, for omission detection, the salience model and the consensus top-k emission rule, which
live in the research code and are not reproduced by `score()`.
"""
from __future__ import annotations

import random
import warnings
from dataclasses import dataclass, field

from .inventory import Decomposer, Fact, SentenceDecomposer
from .verifiers import Verifier


@dataclass
class MarsResult:
    P: float | None = None
    R: float | None = None
    F: float | None = None
    P_min: float | None = None
    n_candidate_facts: int = 0
    n_reference_facts: int = 0
    unsupported: list[tuple[Fact, float]] = field(default_factory=list)
    omitted: list[tuple[Fact, float]] = field(default_factory=list)
    controls: dict = field(default_factory=dict)
    control_info: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"P": self.P, "R": self.R, "F": self.F, "P_min": self.P_min,
                "n_candidate_facts": self.n_candidate_facts, "n_reference_facts": self.n_reference_facts,
                "unsupported": [{"fact": f.text, "start": f.start, "end": f.end, "support": s} for f, s in self.unsupported],
                "omitted": [{"fact": f.text, "start": f.start, "end": f.end, "support": s} for f, s in self.omitted],
                "controls": self.controls, "control_info": self.control_info}


def _f(p, r):
    return (2 * p * r / (p + r)) if (p is not None and r is not None and p + r > 0) else None


def _document_keys(n: int, doc_ids, sources, references, salient_source_facts) -> list | None:
    """The document each item belongs to. Explicit `doc_ids` win; otherwise the input that defines the document
    names it: the source text, else the reference list, else the salient source facts."""
    if doc_ids is not None:
        if len(doc_ids) != n:
            raise ValueError(f"doc_ids has {len(doc_ids)} entries for {n} candidates")
        return list(doc_ids)
    if sources is not None:
        return list(sources)
    if references is not None:
        return [tuple(r) for r in references]
    if salient_source_facts is not None:
        return [tuple(f.text for f in facts) for facts in salient_source_facts]
    return None


def _cross_document_partners(docs: list, strata: list | None, seed: int) -> tuple[list[int | None], list[str | None]]:
    """For every item, the index of an item of another document (and of the same stratum when strata are given), or
    None with the reason. Within a stratum the items are laid out document by document in a seeded order and shifted
    by the size of the largest document, which pairs every item with another document's item and uses each item
    once; when one document holds more than half of a stratum no such pairing exists and partners are drawn from
    the other documents with reuse."""
    rng = random.Random(seed)
    n = len(docs)
    partner: list[int | None] = [None] * n
    reason: list[str | None] = [None] * n
    by_stratum: dict = {}
    for i in range(n):
        by_stratum.setdefault(strata[i] if strata is not None else None, []).append(i)
    for key, idx in by_stratum.items():
        blocks: dict = {}
        for i in idx:
            blocks.setdefault(docs[i], []).append(i)
        if len(blocks) < 2:
            why = "no other document in the batch" if strata is None else f"no other document in group {key!r}"
            for i in idx:
                reason[i] = why
            continue
        groups = list(blocks.values()); rng.shuffle(groups)
        for g in groups:
            rng.shuffle(g)
        order = [i for g in groups for i in g]
        m = max(len(g) for g in groups)
        if 2 * m <= len(order):
            for pos, i in enumerate(order):
                partner[i] = order[(pos + m) % len(order)]
        else:
            for i in idx:
                partner[i] = rng.choice([j for j in idx if docs[j] != docs[i]])
    return partner, reason


class MarsScorer:
    def __init__(self, p_verifier: Verifier | None, r_verifier: Verifier | None, decomposer: Decomposer | None = None,
                 threshold: float = 0.5, seed: int = 20260917):
        self.p_verifier, self.r_verifier = p_verifier, r_verifier
        self.decomposer = decomposer or SentenceDecomposer()
        self.threshold, self.seed = threshold, seed

    @classmethod
    def default(cls, p_path: str = "yaxili96/FactCG-DeBERTa-v3-Large", r_glob: str | None = None,
                decomposer: Decomposer | None = None, r_from_p: bool = False, **kw):
        """A FactCG-contract verifier for P and the MARS-C coverage verifier (a glob of its seed checkpoints) for R.

        MARS-R with the MARS-C verifier needs `r_glob`. Scoring R with the P verifier instead (the "MARS-R, FactCG
        verifier" row of the article) must be asked for with `r_from_p=True`; it is never substituted silently.
        The decomposer defaults to sentences, which is lighter than the article's fact decomposition."""
        from .verifiers import PairVerifier, PromptVerifier
        if not r_glob and not r_from_p:
            raise ValueError("MarsScorer.default(): pass r_glob=<MARS-C checkpoint glob> for the coverage verifier, "
                             "or r_from_p=True to score R with the P verifier (FactCG)")
        p = PromptVerifier(p_path)
        r = PairVerifier.from_glob(r_glob) if r_glob else p
        return cls(p, r, decomposer, **kw)

    # ------------------------------------------------------------------ helpers
    def _p_axis(self, cand_facts: list[list[Fact]], sources: list[str], results: list[MarsResult], key: str | None,
                only: set[int] | None = None):
        prem, hyp, own, facts_flat = [], [], [], []
        for i, facts in enumerate(cand_facts):
            if only is not None and i not in only:
                continue
            for f in facts:
                prem.append(sources[i]); hyp.append(f.text); own.append(i); facts_flat.append(f)
        sup = self.p_verifier.support(prem, hyp) if prem else []
        acc: dict[int, list[tuple[Fact, float]]] = {}
        for o, s, f in zip(own, sup, facts_flat):
            acc.setdefault(o, []).append((f, s))
        for i, r in enumerate(results):
            items = acc.get(i, [])
            if not items:
                continue
            vals = [s for _, s in items]
            if key is None:
                r.P = sum(vals) / len(vals); r.P_min = min(vals); r.n_candidate_facts = len(items)
                r.unsupported = [(f, s) for f, s in items if s < self.threshold]
            else:
                r.controls[key] = sum(vals) / len(vals)

    def _r_axis(self, ref_facts: list[list[list[Fact]]], candidates: list[str], results: list[MarsResult], key: str | None,
                weights: list[list[list[float]]] | None = None, only: set[int] | None = None):
        prem, hyp = [], []
        for i, refs in enumerate(ref_facts):
            if only is not None and i not in only:
                continue
            for facts in refs:
                for f in facts:
                    prem.append(candidates[i]); hyp.append(f.text)
        sup = self.r_verifier.support(prem, hyp) if prem else []
        acc: dict[tuple[int, int], list[tuple[Fact, float]]] = {}
        it = iter(sup)
        for i, refs in enumerate(ref_facts):
            if only is not None and i not in only:
                continue
            for k, facts in enumerate(refs):
                for f in facts:
                    acc.setdefault((i, k), []).append((f, next(it)))
        for i, r in enumerate(results):
            best, best_items = None, []
            for k in range(len(ref_facts[i])):
                items = acc.get((i, k), [])
                if not items:
                    continue
                if weights is not None:
                    w = weights[i][k]; tot = sum(w)
                    val = sum(wi * s for wi, (_, s) in zip(w, items)) / tot if tot > 0 else None
                else:
                    val = sum(s for _, s in items) / len(items)
                if val is not None and (best is None or val > best):
                    best, best_items = val, items
            if best is None:
                continue
            if key is None:
                r.R = best; r.n_reference_facts = len(best_items)
                r.omitted = [(f, s) for f, s in best_items if s < self.threshold]
            else:
                r.controls[key] = best

    @staticmethod
    def _note_partners(results: list[MarsResult], key: str, partner: list[int | None], reason: list[str | None]) -> set[int]:
        for i, r in enumerate(results):
            r.control_info[key] = ({"available": True, "partner": partner[i]} if partner[i] is not None
                                   else {"available": False, "reason": reason[i]})
        return {i for i, j in enumerate(partner) if j is not None}

    # ------------------------------------------------------------------ public
    def score(self, candidates: list[str], sources: list[str] | None = None, references: list[list[str]] | None = None,
              salient_source_facts: list[list[Fact]] | None = None, salience: list[list[float]] | None = None,
              controls: bool = False, doc_ids: list | None = None, groups: list | None = None) -> list[MarsResult]:
        """candidates: texts to score. sources: one per candidate (P, and reference-free R when salient facts are
        given). references: a list of reference texts per candidate (reference-based R, maximum over references).
        salient_source_facts / salience: precomputed salient facts of each source and their weights.
        doc_ids: the document of each candidate, for the shuffled controls (default: the source text, else the
        reference list, else the salient facts). groups: a stratum per candidate, such as its system; the
        shuffled-candidate control then swaps in a candidate of the same group and another document."""
        n = len(candidates)
        results = [MarsResult() for _ in range(n)]
        if groups is not None and len(groups) != n:
            raise ValueError(f"groups has {len(groups)} entries for {n} candidates")
        src_partner = cand_partner = src_why = cand_why = None
        if controls:
            docs = _document_keys(n, doc_ids, sources, references, salient_source_facts)
            if docs is not None:
                src_partner, src_why = _cross_document_partners(docs, None, self.seed)
                cand_partner, cand_why = _cross_document_partners(docs, groups, self.seed)
                missing = sum(j is None for j in cand_partner)
                if missing:
                    warnings.warn(f"controls=True: {missing} of {n} items have no item of another document to swap in "
                                  "and get no shuffled control (see control_info); R_no_candidate is still reported",
                                  stacklevel=2)
        if sources is not None and self.p_verifier is not None:
            cand_facts = self.decomposer.facts_batch(candidates)
            self._p_axis(cand_facts, sources, results, None)
            if src_partner is not None:
                only = self._note_partners(results, "P_shuffled_source", src_partner, src_why)
                self._p_axis(cand_facts, [sources[j] if j is not None else "" for j in src_partner], results,
                             "P_shuffled_source", only)
        ref_facts: list[list[list[Fact]]] | None = None
        weights = None
        if references is not None and self.r_verifier is not None:
            flat = [r for refs in references for r in refs]
            dec = self.decomposer.facts_batch(flat) if flat else []
            ref_facts, k = [], 0
            for refs in references:
                ref_facts.append(dec[k:k + len(refs)]); k += len(refs)
        elif salient_source_facts is not None and self.r_verifier is not None:
            ref_facts = [[facts] for facts in salient_source_facts]
            weights = [[w] for w in salience] if salience is not None else None
        if ref_facts is not None:
            self._r_axis(ref_facts, candidates, results, None, weights)
            if cand_partner is not None:
                only = self._note_partners(results, "R_shuffled_candidate", cand_partner, cand_why)
                self._r_axis(ref_facts, [candidates[j] if j is not None else "" for j in cand_partner], results,
                             "R_shuffled_candidate", weights, only)
            if controls:
                self._r_axis(ref_facts, ["" for _ in candidates], results, "R_no_candidate", weights)
        for r in results:
            r.F = _f(r.P, r.R)
        return results


_SCORE_KW = ("salient_source_facts", "salience", "controls", "doc_ids", "groups")


def score(candidates: list[str], sources: list[str] | None = None, references: list[list[str]] | None = None, **kw) -> list[dict]:
    """One-call convenience: MarsScorer.default(...) then .score(...); returns plain dicts. Keyword arguments of
    `MarsScorer.score` (salient_source_facts, salience, controls, doc_ids, groups) go to the scoring call, the rest
    to `default()`."""
    run = {k: kw.pop(k) for k in _SCORE_KW if k in kw}
    return [r.as_dict() for r in MarsScorer.default(**kw).score(candidates, sources=sources, references=references, **run)]
