# MARS-C audit response — TODO (opened 2026-09-28)

Inputs: the experiments audit of the Friday version (a co-author, 2026-09-27; six serious, nine medium, thirteen minor items)
and the full manuscript review of HEAD 274813d6 (2026-09-28; F01–F26).
Rule: every new run is registered in `research/marsc_ieee_20260922/PREREG.md` or
`research/mars_metric_20260923/PREREG.md` (amendment H-J12 / M-G) BEFORE it is submitted; no bar is moved after a result.

## Phase 0 — verify each finding against the current version
- [x] 0.1 Audit serious 1–6, medium 1–9, minor 13 → TRUE / PARTLY / FALSE / ALREADY FIXED, with the source file:line
- [x] 0.2 Review F01–F03, F08–F10, F12–F13, F20–F24 (metric program, package, code)
- [x] 0.3 Review F04–F07, F11, F14–F16, F19, F22 (omission side; package vs twin matrix; inventory × verifier matrix)

## Phase 1 — Euler experiments at top priority (register first, hold other pending jobs, then submit)
- [x] 1.1 Label-only twin R04 through the crossed H-COND test (RoSE validation + TEST) — audit #3, F04
- [x] 1.2 Label-only twin R04 through the lexical-residual analysis (C2c) — audit #3, F04, F22
- [x] 1.3 Label-only twin R04 through UniSumEval H-COND / H-SURF (hj3) — audit #3/#5, F04, F05
- [x] 1.4 Inventory × verifier grid: fill the missing {Gemma+ent, distilled} × {MARS-C, R04 twin, B21 package, FactCG, MiniCheck-7B, 31B judge} cells on RoSE TEST and UniSumEval — F06, F07
- [x] 1.5 Judge-label verifier retrained on the Gemma+ent unit type (units matched, labels differ) — audit #2
- [x] 1.6 Common-cohort metric reanalysis (Table 3) with paired, source-clustered bootstrap and Holm over the declared primary family — F01, F02, F13, medium #1
- [x] 1.7 RoSE automatic-unit MARS-R (reference facts extracted, no gold ACUs) — F03
- [x] 1.8 SummEval three-source-overlap-excluded sensitivity — F11
- [x] 1.9 Clinical random top-10 and source-salience baselines on the same inventory and matching rule — F15
- [x] 1.10 Pull every file named in `NUMBERS_PROVENANCE` (R-series, strengthen, H-J, M, D1 export) back from Euler and rebuild the release archive — audit #6, F12
- [x] 1.11 Pause (requeue + hold) own jobs running < 3 h only if the queue blocks the block above; release everything when the block ends

## Phase 2 — code fixes
- [x] 2.1 `b2_crossed_families.py combine()` requires both splits and the full family list; add a test — medium #6
- [x] 2.2 `ma_meta.py` common-cohort mode; replace `not_below` (LB ≥ 0) with a declared margin — F01, F02
- [x] 2.3 Package API: explicit `paper_default`, no silent verifier fallback — F09 (scoped by what is verified)

## Phase 3 — manuscript text (only after the verified list; numbers only from results)
- [x] 3.1 Abstract: registered TEST bar missed (+0.0499); +0.122 is secondary package effect; show label-only +0.023 — audit #4, F05
- [x] 3.2 Crossed accuracy wording: "above the registered off-the-shelf rivals; level with the 31B judge and, on UniSumEval, the judge-label verifier" — audit #5
- [x] 3.3 Judge-label package units: spaCy entities + dependency propositions, everywhere — audit #2
- [x] 3.4 Separate names for the package (B21) and the label-only twin (R04) in every table/figure — audit #3, F04
- [x] 3.5 Adjudicator blindness, D1 = validation label, human precision construct ("source-supported, non-conveyed") — medium #3/#4, F13
- [x] 3.6 Multiple comparisons paragraph; defective-validation design freeze in Methods; B4 bar reinterpretation — medium #1/#2/#7
- [x] 3.7 Thirteen minor inconsistencies — minor table
- [x] 3.8 D1 provenance: state what exists; the completed answer sheets are an author-held item — audit #1, F17
- [x] 3.9 Reproducibility statement matches the rebuilt archive — audit #6, F12
- [x] 3.10 Unified verifier "met two of three criteria" — F23; clinical chance wording — F15; learning-curve contrast sign — F20

## Phase 4 — close
- [x] 4.1 Harvest the Phase 1 jobs, record outcomes in the PREREG files, update tables
- [x] 4.2 Rebuild the PDF, ruff, commit to main
- [x] 4.3 Report back against this list
