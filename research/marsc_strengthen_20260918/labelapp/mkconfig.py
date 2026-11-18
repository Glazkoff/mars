#!/usr/bin/env python3
"""Write config.json for the adjudicator-capable label app and print the personal links.

  python3 mkconfig.py --annotators "Ann A" "Ann B" --adjudicator "Adj C" [--study d1|d2] [--port 8788]
                      [--host https://label2.example.org]

The adjudicator must be a THIRD person: the app refuses at startup if an adjudicator name is also an annotator.
`--study d2` swaps in the five revision-review questions and the packet's panels; `--study d1` (default) keeps
the two-question omission-confirmation study byte-compatible with the deployed 400-item database.
"""
from __future__ import annotations

import argparse
import json
import secrets

D1_QUESTIONS = [
    {"id": "q1", "label": "Q1. Is the fact stated in the source, or does it follow directly from it?",
     "options": [["A", "A — yes, supported by the source"], ["B", "B — no (absent, contradicted, or invented)"]]},
    {"id": "q2", "label": "Q2. Does the summary convey the fact? (paraphrases, pronouns and equivalent names count; "
                          "wording need not match; ignore importance)",
     "options": [["A", "A — conveyed / covered"], ["B", "B — omitted / not conveyed"]]},
]
D2_QUESTIONS = [
    {"id": "q1", "label": "Q1. Does the revised summary state at least one important fact from the document that the "
                          "original summary did not?",
     "options": [["A", "A — yes"], ["B", "B — no"]]},
    {"id": "q2", "label": "Q2. Does the revised summary keep every important fact the original summary already had?",
     "options": [["A", "A — yes, all kept"], ["B", "B — no, something important was lost"]]},
    {"id": "q3", "label": "Q3. Does the revised summary contain any claim that the document does not support?",
     "options": [["A", "A — no unsupported claim"], ["B", "B — yes, at least one unsupported claim"]]},
    {"id": "q4", "label": "Q4. Does the revised summary repeat the same information more than once?",
     "options": [["A", "A — no redundancy"], ["B", "B — yes, redundant"]]},
    {"id": "q5", "label": "Q5. Overall, is the revised summary more useful than the original for a reader who will "
                          "not read the document?",
     "options": [["A", "A — more useful"], ["B", "B — not more useful"]]},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotators", nargs="+", default=["Annotator 1", "Annotator 2"])
    ap.add_argument("--adjudicator", nargs="*", default=["Adjudicator"])
    ap.add_argument("--study", choices=["d1", "d2"], default="d1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--items", default="")
    ap.add_argument("--db", default="labels.db")
    ap.add_argument("--host", default="https://label.example.org")
    a = ap.parse_args()

    if a.study == "d2":
        questions, positive = D2_QUESTIONS, {"q1": "A", "q3": "A"}
        panels = [["source", "Source"], ["summary", "Original summary"], ["revised", "Revised summary"]]
        items = a.items or "d2_review_blind.jsonl"
        highlight = "revised"
    else:
        questions, positive = D1_QUESTIONS, {"q1": "A", "q2": "B"}
        panels = [["source", "Source"], ["summary", "Summary"],
                  ["fact", "Fact claimed to be missing from the summary"]]
        items = a.items or "human_sample_blind.jsonl"
        highlight = "fact"

    dup = set(a.annotators) & set(a.adjudicator)
    if dup:
        raise SystemExit(f"adjudicator must be a third person, not {sorted(dup)}")

    cfg = {"port": a.port, "items": items, "db": a.db, "study": a.study,
           "admin_token": secrets.token_urlsafe(24),
           "annotators": {secrets.token_urlsafe(18): n for n in a.annotators},
           "adjudicators": {secrets.token_urlsafe(18): n for n in a.adjudicator},
           "primary_annotators": list(a.annotators[:2]),
           "questions": questions, "verdict_positive": positive, "panels": panels, "highlight_on": highlight}
    json.dump(cfg, open("config.json", "w"), indent=1)
    for t, n in cfg["annotators"].items():
        print(f"{n} (annotator): {a.host}/a/{t}/")
    for t, n in cfg["adjudicators"].items():
        print(f"{n} (adjudicator): {a.host}/j/{t}/")
    print(f"admin: {a.host}/admin/{cfg['admin_token']}/")


if __name__ == "__main__":
    main()
