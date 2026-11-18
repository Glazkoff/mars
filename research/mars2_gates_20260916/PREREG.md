# Pre-registration — MARS-2 main-contribution gates B18–B22 (2026-09-16)

Committed before any of these runs read a label. Estimator, folds, bootstrap and BH are the audit paper's
(`scripts/conditional_value.py`, imported); unit-level metrics are B3's; counter certification is B4's.
Analysis code: `code/analyze.py`. Outputs: `~/results/mars2_gates/`.

Common definitions. "MARS-2 score" = 1 − q from the support head (q-only; the salience weight is a
coverage aggregator, not a ranking signal — post-hoc finding of 2026-09-16, now registered). Arms:
`full_recovery` and `direct_enabled`, 5 seeds averaged. "Counter reference" = candidate length (words,
chars), unit length, token recall of the unit in the candidate, ROUGE-L recall, verbatim presence, and
unit position (ACU index / summary-blind first-mention position). Cells = resources. BH q = 0.05, one
family per analysis (cells × non-counter systems). Paraphrased-only subset = units not verbatim in the
candidate and with token recall < 0.5.

## B18 — agreement with HUMAN unit labels (removes the judge-label risk)

Data: RoSE atomic content units with human presence labels, validation split: 1,412 pairs, 9,760 ACUs,
154 documents, cells rose_cnndm / rose_samsum / rose_xsum. TEST (2,688 pairs, 20,652 ACUs) reserved for B19.
Ground truth: omitted = 1 − human presence label. No judge label is used as truth.

Scoring a supplied unit (no span in the source): mask the source sentence with the highest token F1 to the
unit, target = the unit text (`code/b18_score_mars2.py`, variant `masked`, PRIMARY). Diagnostics: `nomask`
(same window, nothing masked), `aligned` (max 1 − q over the dumped extracted units inside that sentence),
readout `ll` (−log-likelihood).

Systems compared on the same units: lexical (token recall, ROUGE-L recall), MiniLM max cosine,
MiniCheck-Flan-T5-L, AlignScore-L, RoBERTa-L-MNLI window NLI, SummaC-ZS (DeBERTa-L-MNLI), the B21 supervised
cross-encoder, the supplied-target Gemma-4-31B judge (B1 prompt verbatim), the open-ended judges (generated
omissions re-matched to ACUs by token F1 ≥ 0.5), and the counters.

Endpoints: pooled ACU AUROC (all, per cell, paraphrased-only), per-pair macro AUROC, P@1/3/5, R@5; paired
document bootstrap (1,000 draws) of primary − other; certification per cell (counter max, split-half,
permutation null, conditional value counters → counters + system).

Bars (primary = `mars2:direct_enabled|masked|q`; `full_recovery` reported alongside):
- H1 the primary adds information over the counter reference on 3/3 cells (BH-positive).
- H2 primary pooled AUROC ≥ counter-family maximum + 0.03.
- H3 non-inferiority to the best zero-shot entailment baseline: pooled difference ≥ −0.03 and the paired
  macro-AUROC CI lower bound ≥ −0.03. Superiority is not required; the supervised cross-encoder is
  reported without a bar (its bar lives in B21).
Failure interpretation: H1/H2 fail → MARS-2's per-unit verdicts do not track humans beyond lexical
overlap; the method claim is withdrawn. H3 fails → MARS-2 is a target-free enumerator, not a verifier;
the paper pairs it with an entailment verifier (the amortised pipeline) and says so.

## B19 — sealed TEST (removes the validation-only risk) — NOT run until the gate decision

One shot, after the LODO folds, B18 and B20 have been read. Systems and endpoints frozen to the lists
above. Runs: `B2_SPLIT=test` scoring of the arms; B3/B4 on TEST (2,829 pairs, 4 cells); B18 on TEST ACUs;
B22. Decision rule for going to TEST: B18 H1–H2 met on validation AND (LODO transfers on ≥ 3/4 resources
OR B20 closes the held-out gap). Otherwise TEST is scored only for the in-domain claim, stated as such.

## B20 — label-free domain adaptation (addresses the cross-domain null)

Held-out resource D = rose_samsum (fold `xdom-rose_samsum_*`, 3 seeds, arms full_recovery and
direct_enabled). Adaptation rows are built from D's TRAIN-split SOURCES only (`code/b20_build_pseudo.py`):
4 extractive pseudo-candidates per source (ordered random 30–60 % of sentences / dialogue turns), units =
B1 extractor, label 1 iff the unit lies in a kept sentence. No system candidate and no judge label of D is
read. Continue training each fold checkpoint 600 optimiser steps (effective batch 16, lr 5e-5, B2 losses;
`code/b20_adapt.py`) → `adapt-rose_samsum_*`. Control: identical procedure with rows built from rose_xsum
sources → `adaptctl-rose_samsum_*`. Then B2 scoring + B3/B4 on validation.
Bars on the held-out rose_samsum B4 cell, q-only: (1) adapted arm BH-positive over the counter reference
(fold baseline: ΔLL +0.007, not BH+); (2) marginal AUROC ≥ counter max + 0.03 (0.875 → ≥ 0.905);
(3) the control adaptation meets neither (1) nor (2). Failure interpretation: (1)/(2) fail → the domain gap
is not closable label-free; MARS-2 is scoped as in-domain-trained. (3) fails → the gain is extra synthetic
training, not domain knowledge; report both.

## B21 — strong baselines on the same units (removes the weak-baseline risk)

Unit sets: m2 (B1 validation labels, 11,148 units, 4 cells) and the B18 ACUs. Zero-shot: lexical, MiniLM,
MiniCheck, AlignScore, RoBERTa-MNLI, SummaC-ZS with claim = ACU text or rendered unit claim (proposition →
"subject predicate object", entity → containing sentence); document = candidate. Supervised: DeBERTa-large
cross-encoder fine-tuned on the SAME B1 training labels (1 epoch, 3 seeds; `code/b21_finetune_xenc.py`),
hypothesis "[unit] rendered claim".
Bars (m2 set, macro AUROC, paired document bootstrap): (1) MARS-2 best arm q-only beats every zero-shot
baseline (CI excludes 0); (2) versus the supervised cross-encoder MARS-2 is competitive if the difference is
≥ −0.02. If the cross-encoder wins by more than 0.02, the method claim is re-scoped to target-free
enumeration at low cost, and B22 must carry the utility argument.

## B22 — utility on TEST (runs with B19)

(a) Coverage-guided best-of-N at matched length on every TEST source with ≥ 4 candidates (b6_rerank);
(b) error localisation: P@5 of the top-5 omitted ACUs against the human-omitted ACUs on TEST. Bar: (a)
paired gain over the length baseline with CI excluding 0; (b) P@5 ≥ 0.80.

Amendments must be dated and appended below.

## Amendments

**2026-09-16 13:40 MSK — B4 counter family extended (before any TEST run).** B21's zero-shot table showed that content-token
recall of the unit in the candidate reaches 0.88–0.91 pooled AUROC on the B1 units (verbatim presence: 0.79–0.88). The
audit's own rule is that the counter reference must contain the strongest cheap baseline, so `ca_token_recall` and
`ca_rougeL_recall` join `scripts/plan2026/b4_certify.py`'s family and `unit_fuzzy_absent` joins the B3 counter systems.
Every B3/B4 output produced under the old family is kept as `b3b4_*.counters_v1`; the reruns (jobs 4648–4651 and every
later B3/B4) use the extended family. B18/B21 already used the extended family from registration.

**2026-09-16 13:40 MSK — B18 outcome on validation (recorded, not amended).** Primary `mars2:direct_enabled|masked|q`:
pooled 0.880 vs counter max 0.881 (token recall) → H2 failed; adds information on 1/3 cells (rose_cnndm) → H1 failed;
best zero-shot entailment (RoBERTa-L-MNLI window) 0.951, difference −0.071 → H3 failed. `full_recovery` 0.720. On the
paraphrased-only subset: MARS-2 0.740, lexical 0.692, entailment models 0.92–0.95. Per the registered failure
interpretation, MARS-2 is a target-free enumerator, not a per-unit verifier; the method claim on supplied units is
withdrawn, and the paper pairs MARS-2 with an entailment verifier (amortised pipeline). The readout `ll` of the direct
arm is sign-inverted (0.20) because that arm never trains the decoder; reported as a diagnostic only.

**2026-09-16 15:00 MSK — B20 and B21 outcomes on validation (recorded); B21b registered.**
B20 FAILED all three bars on the held-out SAMSum cell (extended counter family, counter max 0.904 = token recall):
fold full_recovery 0.847 (ΔLL +0.011), adapted on SAMSum sources 0.899 (+0.007, not BH+), adapted on the XSum CONTROL
0.931 (+0.014, BH+); direct_enabled 0.882 → 0.871 (adapt) / 0.884 (control). The control beats the target-domain
adaptation, so the gain is generic extra synthetic recovery training, not domain knowledge. Label-free adaptation is dropped.
B21: bar (1) PASSED — MARS-2 q-only (macro 0.945 / 0.934) beats every zero-shot baseline (best entailment 0.65, token
recall 0.858) with CIs excluding 0, and adds information over the extended counter family on 4/4 cells (+0.033…+0.078 nats).
Bar (2) FAILED — the DeBERTa-large cross-encoder trained on the same labels reaches macro 0.984 (pooled 0.985, P@5 0.896,
paraphrased-only 0.949 vs MARS-2 0.816), −0.039 [−0.048, −0.030] for MARS-2, beyond the −0.02 bar; on the human ACUs the
cross-encoder reads 0.951, equal to zero-shot RoBERTa-MNLI. Per the registered rule the method claim is re-scoped: the
recovery/likelihood architecture is not the best per-unit verifier. Since both systems score the SAME extracted units, the
target-free enumeration is the shared unit extractor, not MARS-2; the constructive contribution becomes the audited
unit-level pipeline (extract units → trained cross-encoder), which passes on judge and human labels where open-ended judges read 0.51.
**B21b (registered now, before running):** cross-encoder cross-resource folds, each B1 resource held out, 3 seeds
(`slurm/b21_xenc_lodo.sbatch`). Bars on the held-out cell: adds information over the extended counter family on ≥ 3/4
resources (BH+), and marginal AUROC ≥ the MARS-2 fold's on that cell. Also registered: conditional value of every system
over counters + the in-domain cross-encoder (`analyze.py --reference-system`), to test whether MARS-2's recovery signal
carries any residual information (bar: BH+ on ≥ 2/4 cells; otherwise MARS-2 leaves the main text).
