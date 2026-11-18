"""python -m mars score --candidates c.jsonl --sources s.jsonl [--references r.jsonl] --out scores.jsonl

Input files are JSON lines with a "text" field (references: a "texts" list per line), aligned by line. Verifier and
decomposer choices are flags; with no model flags the sentence decomposer and the lexical stand-in are used, which
is a smoke test, not MARS.
"""
from __future__ import annotations

import argparse
import json
import sys

from .inventory import LLMDecomposer, SentenceDecomposer, Seq2SeqDecomposer
from .score import MarsScorer
from .verifiers import LexicalVerifier, PairVerifier, PromptVerifier


def _read(path: str, key: str):
    return [json.loads(line)[key] for line in open(path, encoding="utf-8") if line.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mars")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--candidates", required=True); s.add_argument("--sources"); s.add_argument("--references")
    s.add_argument("--p-verifier", default="", help="FactCG-contract checkpoint (prompt template, class 1 = support)")
    s.add_argument("--r-verifier", default="", help="glob of MARS-C pair-contract checkpoints (seeds averaged); default = the P verifier")
    s.add_argument("--decomposer", default="sentences", help="sentences | llm:<path> | seq2seq:<path>")
    s.add_argument("--threshold", type=float, default=0.5)
    s.add_argument("--controls", action="store_true")
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "score":
        if a.decomposer == "sentences":
            dec = SentenceDecomposer()
        elif a.decomposer.startswith("llm:"):
            dec = LLMDecomposer(a.decomposer[4:])
        elif a.decomposer.startswith("seq2seq:"):
            dec = Seq2SeqDecomposer(a.decomposer[8:])
        else:
            raise SystemExit(f"unknown decomposer {a.decomposer!r}")
        p = PromptVerifier(a.p_verifier) if a.p_verifier else LexicalVerifier()
        r = PairVerifier.from_glob(a.r_verifier) if a.r_verifier else p
        if not a.p_verifier:
            print("[mars] no verifier given: using the lexical stand-in (a smoke test, not MARS)", file=sys.stderr)
        if a.references and not a.r_verifier:
            print("[mars] no --r-verifier: MARS-R is scored with the P verifier, not the MARS-C coverage verifier",
                  file=sys.stderr)
        scorer = MarsScorer(p, r, dec, threshold=a.threshold)
        cands = _read(a.candidates, "text")
        srcs = _read(a.sources, "text") if a.sources else None
        refs = _read(a.references, "texts") if a.references else None
        res = scorer.score(cands, sources=srcs, references=refs, controls=a.controls)
        with open(a.out, "w", encoding="utf-8") as fh:
            for x in res:
                fh.write(json.dumps(x.as_dict()) + "\n")
        print(f"[mars] {len(res)} candidates scored -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
