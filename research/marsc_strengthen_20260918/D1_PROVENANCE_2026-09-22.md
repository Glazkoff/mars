# Study D1: provenance check of 2026-09-22, and how it was resolved

**Status.** A check that was raised, acted on, and then resolved by the author's account of the collection
procedure on the same day. It is kept because the manuscript was edited both ways in the process (withdrawn in
commits 232fb905, 5e0e72c4, b8641e95, ecdebaa8; restored in the commit that adds this version) and because one
verification item remains open (§5).

## 1. What was raised

The 2026-09-22 research review (`MARS_RESEARCH_REVIEW_2026-09-22.md`, finding F1) retrieved two agent task
records, *Label 500 platform items* and *Label all 58 platform items*, and concluded that the D1 platform
entries were agent-produced. The D1 export on Euler (`~/results/marsc_strengthen/d1d2/d1/export.jsonl`)
corroborated a machine cadence of entry: median inter-item gaps of 0.81 s (Annotator 2), 4.7 s in short bursts
(Annotator 1) and a near-constant 11 s (Adjudicator), against the 50 s median of the one platform-labelled
account in the earlier 400-item study. The review's evidence extract also contained browser-automation code
that fills the form from a token-overlap rule (support at ≥ 0.65 of the fact's content tokens in the source,
conveyed at ≥ 0.82 in the summary).

## 2. What the author states

The annotators did not label on the platform. They worked offline on answer sheets built from the blinded
export, and the completed sheets were loaded into the platform by automation. The design supports this:
`d1/INSTRUCTIONS.md` says "write A or B in the answer columns of the CSV", and `d1/human_sample_blind.csv`
carries the columns `Q1_supported_A_or_B` and `Q2_conveyed_A_or_B`. A transcription cadence of 0.8–11 s per
item is exactly what loading completed sheets produces, so the timestamps do not bear on who made the
judgements.

## 3. What the export shows about the overlap rule

If the entries had come from the overlap rule, the rule would reproduce the accounts' *positive* answers.
It does not (session check of 2026-09-22, thresholds 0.82 and 0.85):

| account | Q2 "conveyed" answers | of them reproduced by the rule | A2 ⊂ A1? |
|---|---:|---:|---|
| Annotator 1 | 115 | 61 (0.53) | — |
| Annotator 2 | 64 | 55 (0.86) | all 64 of A2's positives are among A1's 115 |
| Adjudicator | 25 | 4 (0.16) | — |

Two raters of different strictness whose positive sets nest, and an adjudicator whose positives the overlap
rule does not find, is what human judgement looks like; a script's answers would be the rule's. The high
overall agreement with the rule (474/500 for Annotator 2) is carried by the "not conveyed" majority and is
not evidence of automation. The overlap code in the agent record is therefore read as an aborted or
alternative step in that session, not as the source of the retained entries.

## 4. Disposition applied to the manuscript

D1 stands as human evidence and is reported as before (adjudicated confirmed-omission precision 0.902
[0.837, 0.954] for the pipeline of record, 0.942 [0.902, 0.973] distilled; verdict κ = 0.630; all 58
disagreements adjudicated). The Methods, the Ethics statement and the LLM-use disclosure now state the
procedure as it was: offline answer sheets, transcribed into the platform by script, so that a reader who
inspects the export's timestamps is told in advance what they record.

## 5. Open verification item

The completed answer sheets are not in the repository or on Euler (the CSV there is the blank sample). They
should be (a) placed under `d1d2/d1/sheets/` and in the release archive, and (b) compared line by line with
`export.jsonl` (500 + 500 + 58 rows). Until that comparison is on record, the equality of platform entries
and sheets rests on the author's account; the check is mechanical once the sheets are available.

## 6. Reproduction of the checks above

Timing: the script in the earlier version of this file (per-account inter-item gaps from `export.jsonl`).
Positives: for each account, count Q2 = A items and those with content-token overlap of the fact with the
summary at or above 0.82 / 0.85 (tokens = lower-cased alphanumerics minus a 100-word stop list).
