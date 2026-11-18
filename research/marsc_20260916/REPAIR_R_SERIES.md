# MARS-C repair programme (R01--R09), opened 2026-09-17

**Status of this document.** This is a *repair* record, not a pre-registration of a new hypothesis. It was
written after an external read of the submission (`article/iclr2027/FULL_REVIEW_2026-09-17.md`,
`EXPERIMENT_AUDIT.md`, `PAPER_CLAIM_AUDIT.md`) found two estimator defects and four claim-scope problems. Every
block below states what was wrong, what the repair computes, and what bar (if any) it is held to. Blocks that
introduce a *new* quantity (R03, R04, R05, R06, R07) state their bar before the result is read; blocks that
re-compute an existing quantity (R01, R02, R09) have no bar, because a repair cannot be allowed to pass or
fail -- the corrected number is the number.

No model was retrained for R01, R02, R08 or R09: the frozen per-seed score files are the same bytes as in G1,
G3 and G7, and their sha256 is recorded in the `emitter_manifest` block of every output JSON.

---

## R01 -- corrected consensus estimator (repair, no bar)

**Defect.** The manuscript defines consensus as the diversity rule applied independently to three verifier
seeds, the three outputs averaged. `mc_variants.py seedavg` also writes `*_seedavg.jsonl` and `mc_diversity.py`
turns it into `*_seedavg_div1.jsonl` in the same directory, so the job's `*_seed*_div1.jsonl` glob matched
**four** files: three per-seed outputs plus one file that is already a function of all three. Confirmed in
`results/g1_test/SHA256SUMS` and reproduced by `tests/test_seed_manifest.py::test_glob_still_sweeps_in_the_pseudo_seed`.

**Repair.** `mc_e2_eval.py` now accepts an explicit comma-separated manifest instead of a glob, records every
matched path with its sha256 in the output JSON, and offers `--strict-seeds` (a multi-file emitter must be a
roster of distinct *numeric* seed tokens, present on every pair) and `--expect-seeds N`. `mc_r01_consensus_fix.sbatch`
re-runs the sealed TEST split, UniSumEval, the k-curve, the adaptive budget, the MARS-2 comparison and the
seed-aggregation alternatives from explicit three-file manifests under `--strict-seeds --expect-seeds 3`.

## R01b -- single-model regime and label-efficiency curve (repair, no bar)

These cells were never affected (one seed-averaged file each), and are re-run only so that every interval in
the paper comes out of the hardened evaluator with a recorded manifest. Expected to reproduce exactly; they did.

## R02 -- rank-recovered human precision (repair, no bar)

**Defect.** G3 asked for k=5/10/20; the evaluator dumps one budget, max(k)=20. The blinded sample was drawn
from that top-20 dump with rank discarded, and the analysis then weighted the strata by the **k=10** status
mix. The published 0.85/0.82/0.74/0.64 identify neither budget.

**Repair.** The dump is rank-ordered, so position in `emitted` is the rank. `mc_human_rank.py` re-joins the
sample key to the canonical dump on (system, pair_id, fact text, score), recovers each item's rank, and
estimates precision separately at rank<=10 and rank<=20 with stratified bootstrap intervals, per annotator,
adjudicated, and under both one-sided annotator rules. Join quality and the rank histogram are reported.

## R03 -- summary-conditioning identification (new; bars stated before reading)

**Objection answered.** Fact coverage has strong summary-blind predictability (the paper's own audit), and
MARS-C trains on those labels, so the ranking could reflect a prior over which facts are *generally* omitted.

* **R03a** within-document, same-fact crossed cells: `P(z(f | summary that omits f) > z(f | summary that
  conveys f))`. The fact, the document and the inventory are identical across the two scorings; only the
  premise changes. A summary-blind scorer is pinned at exactly 0.500.
  **Bar: > 0.60 with the document-bootstrap lower bound above 0.5 on validation AND on the sealed test split.**
* **R03b** summary-shuffled emission: rescore the frozen TEST inventories with another pair's summary (seeded
  derangement within resource x system), run the identical rule, evaluate recall@10 against the true labels.
* **R03c** fact-only floor: the same with no summary at all.
  **Bar for R03b/c: the real-summary recall@10 must exceed both controls with the paired interval above zero.**

Failing either bar means the mechanism claim must be withdrawn, whatever the recall numbers say.

## R04 -- matched-unit, matched-count label-source control (new; bar stated before reading)

**Objection answered.** The deployed human-fact and judge-label verifiers differ in label source *and* unit
construction *and* unit distribution *and* label budget, so the paper identifies a supervision package.

**Design.** The same Gemma-4-31B-it coverage judge, with the prompt the paper's judge pipeline used, labels the
*identical* (summary, human-ACU) cells of the *identical* E0 train and calib documents -- 63,343 cells. The
identical recipe, backbone, sampler, seeds and step budget then trains on the human-labelled and the
judge-labelled twin. Arm A is the deployed objective; arm R (random natural cells) is run as well because its
sampler cannot be influenced by the label-derived crossed structure.

**Bar: none on the outcome.** The result is reported whichever way it falls, beside the deployed-package gap;
the paper's causal wording is licensed only by the sign and interval this returns. Judge-vs-human agreement on
the shared cells is reported either way.

## R05 -- equivalence for the label-efficiency curve (new; margin declared before reading)

**Objection answered.** Four overlapping intervals are an absence of a detected difference, not equivalence.

**Declared margin: 0.05 recall@10**, chosen as the size of the registered headline effect -- a supervision
reduction counts as equivalent only if any loss it causes is smaller than the effect the paper claims to
detect. TOST on document-level paired differences, 20,000 bootstrap draws. The achieved (smallest supportable)
margin is reported alongside.

## R06 -- off-the-shelf omission baselines on the sealed test split (new; no bar)

**Objection answered.** Direct omission baselines are missing. A windowed RoBERTa-large-MNLI entailment check
(the standard NLI verifier), SummaC-ZS, MiniCheck, AlignScore, sentence-embedding similarity and two lexical
counters are each run as the *verifier* inside the identical frozen inventory and emission rule.

## R07 -- crossed inventory x verifier precision (new; no bar)

**Objection answered.** The claim that the precision cost sits "in the entity units, not in the verifier" needs
a 2x2, and the human sample covers three of its four cells. The missing cell (Gemma+ent inventory, judge-label
verifier) was never even evaluated on validation. R07 evaluates all four cells at k=10 from explicit three-seed
manifests and completes the square with the coverage judge that the same human sample validated (0.96 agreement
on adjudicated items). The judged rate is a proxy and is reported as one, beside the three human cells.

## R08 -- glob audit of the whole job chain (new; no bar)

`mc_glob_audit.py` expands every score pattern in every shipped sbatch against the artifacts on disk and flags
multi-file sets that contain a pseudo-seed, repeat a seed, or match nothing. Run over the 121 patterns of the released chain it flags 50: 33 PSEUDO_SEED (the defect the review found), 11
DUP_SEED (a second instance the review did not, see R09) and 6 EMPTY. Every one is inside a job carrying a
`% SUPERSEDED-BY:` header; with `--skip-superseded` the live chain flags nothing and the tool exits zero.

## R09 -- corrected validation comparisons (repair, no bar)

**Defect found by R08, not by the review.** On validation the three diversity outputs sit in the same directory
as the three diversity outputs of the *rejected* learned-importance policy (`*_seed<N>_imp0.2_div1.jsonl`), and
`*_seed*_div1.jsonl` matches both. Every validation consensus cell in E2/G3 therefore averaged six files. The
G3 union glob separately matched nothing, so the union variant was silently dropped from its own comparison.
R09 re-runs the whole validation comparison from explicit three-seed manifests and rebuilds the union outputs.

---

# Outcomes, recorded after the blocks ran

## R01 / R01b -- corrected consensus (repair, no bar)

The sealed-test headline moved by $0.001$ (0.378 -> 0.3765) and its interval tightened to [+0.024,+0.080]; the
UniSumEval consensus gap is unchanged at +0.048; the single-model cells reproduced exactly. No ordering or sign
changed. **Correction (2026-09-17, post-repair review):** the registered +0.05 margin is *not* cleared --
the observed contrast is 0.376536-0.326605 = 0.049931, missing it by 7e-05. The +0.051 printed here and in
the first revision is `paired_recall.mean`, which `mc_e2_eval.py` computes as the mean of the bootstrap
replicates rather than the observed difference. Direction and interval above zero are unaffected.

## R02 / R02b -- rank-recovered human precision (repair, no bar)

All 400 items rejoined (399 with the stored status intact); 239 have rank <= 10, 56--63 per system. Adjudicated
precision at rank <= 10: ours distilled 0.870, judge-label distilled 0.825, MARS-2 0.783, ours Gemma+ent 0.640.
The sample predates the repair, so it annotates the pre-repair emission: 48--70 of each system's 100 items are
still in the corrected top ten, and on that subset the rates are 0.908 / 0.864 / 0.782 / 0.738. Ordering stable
across all readings; level is not, and the restriction is a selection on rank. Both are reported.

## R03 -- summary-conditioning identification (BAR FAILED on R03b/c)

* **R03a PASSED its 0.60 bar by a wide margin.** Same-fact crossed cells: human-fact 0.9163 [0.8982,0.9287] on
  test and 0.9181 [0.8973,0.9354] on validation; judge-label 0.8991 and 0.8957. Chance is exactly 0.500.
* **R03b/c FAILED their bar, and the failure is informative.** Recall@10 *rose* under the controls: 0.377 real,
  0.440 with another document's summary, 0.458 with none. Diagnosis: recall@k never charges for demoting a fact
  the summary conveyed, and reference ACUs over-represent conveyed facts, so a summary-blind emitter aligns to
  more ACUs (0.226 of its emissions against 0.173) and is rewarded. On omission precision --- the endpoint that
  does charge --- conditioning pays: 0.720 against 0.666 and 0.656, +0.055 [+0.035,+0.076] and +0.065
  [+0.041,+0.090] document-paired.

  Consequence, stated in the paper: recall@k against human-marked omissions cannot on its own certify that a
  detector finds what a *particular* summary is missing. Omission precision was added to every reported cell
  (R12). The comparative claims are unaffected: both packages are scored on the same endpoint with the same
  real summaries.

## R04 -- matched-unit, matched-count label-source control (no bar; reported as it fell)

The judge answered all 63,343 shared cells. It agrees with the annotators on 0.7163 of them (kappa 0.4727),
calling 64.9% covered where the annotators call 37.2%. Trained on the twin under the identical recipe, seeds
and step budget: supplied-ACU AUROC 0.9740 (human labels) against 0.9613 (judge labels), document-paired
+0.0128 [+0.0082,+0.0180]; arm R 0.9751 against 0.9575; the arm itself moves the primary by -0.0011
[-0.0030,+0.0005]. The deployed package gap on the same endpoint is +0.023, so roughly half is the label source
and half is everything else the package changes. That is weaker than the causal claim the first version made,
and it is what the paper now says.

## R05 -- equivalence (margin declared before the test)

Equivalence holds at the declared 0.05 margin for 85, 213 and 426 documents against 853; the 85-document TOST
interval is [-0.034,+0.006] and the data support equivalence down to a margin of 0.034.

## R07 -- crossed inventory x verifier precision (no bar)

Census of 40,586 top-10 emissions across the four cells. Inventory main effect -0.0285 [-0.0355,-0.0217],
verifier main effect +0.0075 [+0.0051,+0.0099], interaction +0.0066 [+0.0019,+0.0116]. The paper's attribution
(the precision cost is in the inventory, not the verifier) is supported; its magnitude (0.1--0.2) is not --- the
judge separates the inventories by 0.032 where the annotators separate them by 0.23. Reported as: attribution
supported, magnitude uncertain between 0.03 and 0.23.

## R08 -- glob audit

121 patterns scanned, 50 flagged (33 pseudo-seed, 11 repeated-seed, 6 empty), all inside superseded jobs. With
`--skip-superseded` the live chain flags nothing.

## R09 / R09b -- corrected validation (repair, no bar)

Validation consensus cells move from 0.490/0.473/0.442 to 0.400/0.369/0.311; the human-versus-judge margin
roughly doubles, to +0.087 [+0.041,+0.132]. The G3 union variant, which its own glob had silently dropped,
reads 0.403 [-0.035,+0.027] against the rule and still clears no bar.

## R10 -- the component the repair un-dropped (post-hoc secondary read of a spent split)

The learned importance policy adds +0.060 [+0.015,+0.101] on validation and +0.081 [+0.050,+0.112] on the
held-out split, against the -0.029 the contaminated comparison showed. But it pays the fact-only arm's price
for the same gain: -0.045 [-0.075,-0.018] omission precision for +0.081 recall, against -0.065 for +0.081. It
is a summary-blind reference-likeness prior, so the registered decision stands --- for a better reason than the
one it was made for. Not promoted.

## R11 -- matching-rule sensitivity

At fixed inventory the verifier margin survives every informative rule (TEST +0.122/+0.070/+0.041/+0.028;
UniSumEval +0.049/+0.080/+0.093/+0.053, rising under tightening). The cross-inventory headline does not:
+0.050 at one shared token, -0.008 at two, -0.008 at three, +0.051 under a Jaccard floor of 0.20. A Jaccard
floor of 0.34 leaves recall at 0.05 and separates nothing.

## R12 -- the omission-precision endpoint

Added to the evaluator because of R03 and recomputed for every headline cell. At fixed inventory on Gemma+ent
the human-fact verifier beats the judge-label one on BOTH endpoints (+0.121 recall, +0.029 precision); across
inventories the headline trades +0.050 recall for -0.035 precision.

## R04 scope note

The emitted-endpoint arm of R04 was run on the Gemma+ent inventory, the pipeline of record's
(`mc_r04d_fast_recall.sbatch`). The full 24-pass scoring of R04c, which would have added the distilled
inventory and arm R on the emitted endpoint, was cancelled for compute: it duplicated no claim, and the
label-source result is carried by the supplied-ACU AUROC (all four arms, `analysis_auroc/matched_auroc.json`)
and by the Gemma+ent emitted comparison. The trained twins are in `r04_matched/models/` and the run can be
completed from them without retraining.
