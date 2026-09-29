# MARS-C strengthening campaign — registered 2026-09-18

**Status of this document.** A pre-registration written *before* any of its numbered jobs was executed.
Every block states its defect or hypothesis, what it computes, and — where a bar exists — the bar, declared
before the result is read. Blocks that *repair* an existing quantity carry **no bar**: a repair cannot be
allowed to pass or fail, the corrected number is the number.

**Author of this registration.** Claude Opus 5 session `0608f529`, acting on the user instruction to strengthen
the MARS ICLR 2027 submission. Governing design document:
`/path/to/mars-research-strategy-20260918/refine-logs/FINAL_PROPOSAL.md`.
Defect list: the three "Still open" sections of `article/iclr2027/NUMBERS_PROVENANCE_2026-09-17.md`.

**Deadline context.** ICLR 2027 abstract 2026-09-18 23:59 AoE; full paper 2026-09-25 23:59 AoE. Wave A and
Wave B are sized to land inside that window. Wave D is explicitly *not*: it is registered now so that its
artifacts are frozen before any outcome is read, and executed in the rebuttal period or not at all.

---

## 0. The strategic decision this campaign encodes

Reconnaissance established three facts that change what the paper should claim.

1. **The paper's own control refutes its own endpoint.** Rescoring the frozen inventory with *another
   document's* summary raises recall@10 from 0.377 to 0.440, and removing the summary entirely raises it to
   0.458. Recall@k against human-marked omissions never charges for demoting a fact the summary *did* convey,
   and reference ACUs over-represent conveyed facts, so a summary-blind emitter is *rewarded*. Any claim whose
   primary endpoint is recall@k is unsafe — including MARS-C's own.
2. **A cheap off-the-shelf baseline already beats the pipeline of record on that endpoint.** In the completed
   R06b table, SummaC-ZS over the distilled inventory scores 0.3848 recall@10 against MARS-C's 0.3765, with the
   precision gap straddling zero. A paper that leads on recall@10 loses to a 2021 baseline.
3. **MARS-C wins decisively on the endpoints that charge for false alarms and that require reading the
   candidate.** Omission precision: 0.720 real summary vs 0.666 shuffled vs 0.655 none. Same-fact crossed
   accuracy: 0.914 [0.898, 0.929] against a ceiling of exactly 0.500 for any summary-blind scorer.

The strongest *defensible* position is therefore not "MARS-C has the highest recall@10". It is:

> **Recall@k against human-marked omissions cannot certify an omission detector, because it rewards ignoring
> the candidate; under endpoints that do charge for that, natural coverage supervision produces the only
> verifier family that demonstrably reads the summary.**

This is a methodological contribution with a constructive detector attached, and every artifact needed to back
it already exists. Wave B is designed to test the second half of that sentence honestly — including the
possibility that a baseline conditions just as well, in which case we report that.

**Standing rule for this campaign.** No block may be reported on recall@k alone. Every new emitter arm is run
together with its shuffled-summary and no-summary controls in the same job (Block B0), and reports omission
precision beside recall.

---

## Wave A — entry gate and defect closure (no bars; repairs)

These close named open items. None introduces a new scientific claim.

### A1 — the calibration contract (closes provenance item C01/R1)

**Defect as recorded.** "Validation uses fitted per-seed calibration, test/external uses raw sigmoid."

**Defect as actually established by reconnaissance.** Narrower and differently shaped. `mc_train.py` fits a
per-seed Platt map and applies it to everything it writes; `mc_score_units.py` emits a bare sigmoid. The
mismatch is therefore not split cleanly by pool — *within* validation the contract is mixed, with the primary
`humanfact` cell on gemma+ent calibrated while `humanfact.distilled` and `judgefact` are raw.

**What is invariant, and why this matters.** The emission rule is rank-only: `mc_diversity.py` sorts by score
and applies a fixed demotion, and `mc_e2_eval.py` orders by descending score. A *strictly monotone* per-seed
map therefore cannot change a single-seed top-k. Measured: 0/1466 pairs change at k=10, all three seeds.
The entire exposure lives in **seed averaging**, where a per-seed monotone map is applied before the mean and
the mean of monotone transforms is not monotone in the mean. Measured exposure: 154/1466 pairs change at k=10
under consensus, 510/1466 under the single-model regime.

**A1a.** Re-derive every validation cell under one declared contract: **raw sigmoid everywhere** (contract
(a)). Chosen because the judge-label `xenc` arm has no fitted Platt map and no calib-role ACU file to fit one
on, and the MARS-2 dump emitter (score = 1 − support) has no calibratable logit at all, so contract (b) is not
available for the full system set. Frozen here, before any output is read.

**A1b.** Publish the invariance result as a positive robustness statement with the measured numbers above.

**A1c.** Put the label-efficiency curve (0.409/0.402/0.395/0.398) on the same contract. Its four points were
each fitted at a *different* calib-set size (n_cells 2772/5385/7225/7225, Platt slopes 0.56–0.84), so the
reported spread of 0.014 is the same order as the calibration perturbation itself. This underwrites the R05
equivalence result and is the most fragile number in the paper.

**A1d.** Recompute the R04 matched label-source AUROC from **raw logits**. Previously unflagged instance of the
same defect: the two arms' Platt maps were fitted at different prevalences with opposite-signed intercepts
(human omitted_rate 0.6094, intercepts +0.159/+0.355/+0.365; judge 0.3219, intercepts −0.444/−0.351/−0.170),
and `mc_matched_auroc.py` seed-averages *calibrated* probabilities before computing AUROC.

**A1e (zero cost, added robustness).** Run the `leadunion` aggregation on validation. Its primary sort key is
per-seed sentence-leader counts, which is exactly calibration-invariant, and it already reproduces the
consensus headline to 3e-4 on both external pools. Extending it to validation bounds the paper's entire
calibration exposure at ~3e-4 across three splits.

### A2 — R07 document-clustered bootstrap (closes provenance item R5/PCA-003)

`mc_crossed_precision.py`'s resampling variable is named `by_doc` but is **keyed by `pair_id`**; the file never
imports `common` and never calls `doc_key`. RoSE contributes 8–12 summaries of one source document as
nominally independent units (1388 pairs over 151 documents, mean 9.19, all 151 multi-summary), so every
interval in that file is anti-conservative by up to sqrt(1388/151) ≈ 3.03×.

**Repair.** A new estimator that clusters on `common.doc_key(source)`, matching the main evaluator. No bar.
**Declared before execution:** the interaction effect is currently +0.0066 [+0.0019, +0.0116] — 0.0019 from
zero — and may cross zero under correct clustering. If it does, the paper reports that it does. No manuscript
sentence written before this runs may presume either outcome.

### A3 — document-clustered intervals for the human precision study (closes R4/C03)

`mc_human_rank.py` resamples items within status strata. **Repair:** resample source-document clusters.
**Declared before execution:** at rank ≤ 10 the agreement-conditioned items are 47/40/44/53 backed by only
37/36/37/42 distinct documents. The intervals will get materially **wider**, and a percentile bootstrap on ~37
clusters can be degenerate. That is the honest outcome, not a failure.

**A3b — a disclosure this campaign is obliged to make.** Reconnaissance found that in the 400-item human
study, annotator 2 answered `A` on **all 400 items**, giving kappa exactly 0.0. Every "two-annotator"
statement in the paper rests on an agreement filter against a degenerate second rater. This must be stated in
the manuscript in plain language. No analysis can repair it; only new annotation can.

### A4 — symmetric matching rule and the restored Jaccard sensitivity (closes provenance item 4b)

`mc_e2_eval.py` builds ACU tokens content-only but emitted-unit tokens with all tokens including stopwords,
inflating the Jaccard union. This is why the Jaccard-floor rows were withdrawn. **Repair:** add an opt-in
symmetric mode and re-run the sensitivity. The bias is *lenient to the emitter*, so a symmetric rule will
likely lower every recall number. Reported as a sensitivity, not a headline. The ≥1/≥2/≥3 shared-token rows
(+0.122/+0.070/+0.041) are unaffected and stand.

### A5 — withdraw the rank-average (seedrank) ablation (closes provenance item 4c)

Settled by reconnaissance, no job needed: `seedrank` scores span [−438, 0], so `mc_diversity.py`'s fixed −5.0
demotion moves a unit five rank slots instead of below the sentence leaders — measured proof, `div1` did not
change the file's range at all. The published "humanfact.rank+div1 = 0.2787 vs consensus 0.3765" is measuring a
**broken rule**, not a worse aggregation. The row is withdrawn and the incompatibility disclosed.

### A6 — evidence the B1 census (closes C06) and A7 — rebuild the supplement (closes R2/item 3)

The 85,600-label / 2,625-pair census is recomputed from the training rows inside the bounded artifact set.
`marsc_update.zip` is rebuilt from the post-repair tree with a regenerated `ARCHIVE_VERIFICATION.json` and a
re-run anonymity gate. A7 **must run last**, after every Wave A/B number is final.

### A8 — complete the confidence-interval audit

Reconnaissance found a third unclustered site the paper's audit table does not name:
`scripts/plan2026/a5_metric_selection.py:44` resamples at item level. The audit table is corrected to name all
sites and their clustering level.

### A9 — `mc_glob_audit.py` as a mandatory pre-submission gate

Run before every submission in this campaign; its clean exit is cited. This is the standing guard against the
defect that invalidated an entire results generation (121 patterns scanned, 50 flagged: 33 pseudo-seed, 11
duplicate-seed, 6 empty).

---

## Wave B — new evidence, machine-only, no new human labels

### B0 — the standing control harness (infrastructure, no bar)

Every arm produced anywhere in Wave B is scored three times: real summary, **shuffled** summary (seeded
derangement within resource × system), and **no** summary. Recall@10 and omission precision are reported for
all three. Rationale: the summary-blind control is the project's strongest identification instrument and it is
the control that came out *against* the paper. Attaching it by default means no result in this campaign can be
published that a shuffled-summary emitter would also achieve.

### B1 — complete the off-the-shelf verifier family (closes provenance item 4a)

**What was actually established.** The R06 chain does **not** evidence six verifiers. `r06_offshelf` was
cancelled mid-MiniCheck at 48:59 leaving 5 distilled score files and an empty `val/`; the completed `r06b_fast`
ran only `lex, minilm, nli, summac_zs`. **MiniCheck and AlignScore were never run as MARS-C verifiers.**

**Why they matter.** They are precisely the two metrics the paper's own audit half singles out as retaining
conditional value on 8/10 cells, against 0/10 for windowed NLI.

**Design.** Both run as the verifier inside the identical frozen inventory and emission rule, CPU-sharded
across the 768 idle cores (MiniCheck measured at ~1.36 units/s on one H200: 423,936 units is ~85 GPU-h
unsharded, which is why the original job could not finish under a 12 h limit). Pilot with `--limit 200` to size
AlignScore, which is unmeasured, before committing the array.

**No bar, and a declared risk.** SummaC-ZS already beats the pipeline of record on recall@10. MiniCheck or
AlignScore may too. The reporting rule and multiplicity control are frozen here, before the results are read:
**all** completed verifier rows enter the main table, ordered by recall@10, with omission precision and the B0
controls beside them, whatever the ordering turns out to be.

### B2 — candidate-conditioning head-to-head (the campaign's primary new hypothesis)

**Hypothesis H-COND.** Among verifier families that can be dropped into the identical inventory and emission
rule, natural-coverage-supervised MARS-C is the one that conditions on the candidate summary; strong
off-the-shelf alternatives that match or beat it on recall@k do so substantially by exploiting candidate-blind
omission priors.

**Why this is the right hypothesis.** It is the reviewer's first question, it is the one MARS-C's supervision
choice actually predicts, and — unlike a recall@k margin — a summary-blind system is *pinned* on its endpoint,
so it cannot be won by ignoring the summary.

**Primary estimand.** The R03a same-fact crossed accuracy, computed identically for every verifier family:
within a document, for the same fact, P(z(fact | a summary that omits it) > z(fact | a summary that conveys
it)). Any summary-blind scorer is pinned at exactly 0.500. Document-clustered bootstrap, validation and the
sealed TEST split.

**Secondary estimands.** Δrecall@10 and Δomission-precision between the real summary and each of the two B0
controls, per verifier family. A family that conditions shows a *positive* precision gap and a *negative*
recall gap (the paper's own finding for MARS-C: +0.065 precision, −0.055 recall).

**Bars, declared before execution.**
- **H-COND passes** if MARS-C's crossed accuracy exceeds every off-the-shelf family's by a margin whose
  document-clustered one-sided 95% lower bound is above 0, on validation **and** on TEST.
- **H-COND fails** if any off-the-shelf family matches MARS-C within that margin. If it fails, the paper says
  so and the supervision claim narrows to the fixed-inventory verifier effect, which is unaffected.
- A family whose crossed accuracy is not distinguishable from 0.500 is reported as **candidate-blind**, and any
  recall@k advantage it holds is reported as evidence for the methodological claim, not against MARS-C.

### B3 — external matched-label transfer (closes provenance item 5)

The only clean causal contrast the paper has — label source, human vs judge, everything else matched (AUROC
0.9740 vs 0.9613 on validation) — currently exists on validation only. The 12 R04 checkpoints already exist;
this extends them to TEST and UniSumEval by **scoring only, no retraining**. Requires `MC_TEST_REREAD=yes`.
**No bar** — the value is the sign and interval, whichever way they fall.

### B4 — the literal unseen third extractor

`Qwen/Qwen3.8-27B` decomposes source documents into atomic propositions under the decomposition prompt and
decoding contract **copied verbatim** from the declared interface, with model revision and prompt sha256
frozen before execution. None of its units, scores, labels or output statistics may inform training, mining,
calibration or model selection; the threshold learned on the development extractors is applied unchanged.

**Trap, disclosed and guarded.** Qwen3.8-27B opens a `<think>` block before every answer; this collapsed choice
mass to 1e-7 and returned 88% empty lists in an earlier campaign, producing a widely-repeated but **wrong**
result. `enable_thinking=False` is mandatory and is asserted in the job.

**Bar.** Cross-extractor transfer is evidence for deployment-distribution learning only if the MARS-C ordering
on the *development* extractors reproduces on the unseen one with a document-clustered interval excluding zero.
If it does not, the transfer claim is withdrawn, not rescued.

### B5 — a conservative machine-proxy lower bound on corrected-emitter yield

The only semantic precision evidence in the paper annotates an emission the pipeline no longer produces
(measured drift: corrected-top-10 overlap 0.59/0.66/0.63; only 51/53/48/70 of 100 annotated items survive).
Re-analysing the old labels is a selection on rank, not a measurement.

**What is defensible without new humans.** On 63,343 human-ACU cells the judge's *positive* omission verdict is
near-exact: precision_on_judge_omitted = 0.9900, with recall_on_human_omitted = 0.5535. The judge's *support*
answer is the measured failure mode (it calls 36% of emitted facts unsupported where annotators almost always
say yes; raw agreement 0.7163, kappa 0.4727). So a judge-confirmed omission gives a **conservative lower
bound** on useful yield and nothing more. The corrected-emitter judgments already exist
(`r07_crossed/emitted_validation_k10.jsonl`, 40,586 top-10 tasks, 0 unjudged slots).

**Reported strictly as a lower bound**, never as precision, with the recall shortfall printed beside it.

### B6 — cost and the amortisation curve · B7 — QAPyramid overlap · B8 — vLLM pilot

B6 re-times the pipeline at one new summary per new source and completes the reuse-amortisation curve, giving
the paper an efficiency fallback if quality reaches only non-inferiority.
B7 recomputes QAPyramid's overlap with RoSE: the recorded "zero overlap" is a **measurement artefact**
(`mc_pool_feasibility.py`'s `find_text()` found 0 text records in QAPyramid's dict-keyed JSON), and QAPyramid
is CNN/DM-based while `rose_cnndm` is in the MARS training pool, so overlap is plausibly large.
B8 pilots vLLM, which is installed but has never run on this box (every LLM log records `backend=hf`).

---

## Wave C — additional hypotheses, registered to strengthen the contribution

These are this campaign's own additions, beyond the FINAL_PROPOSAL. Each is cheap, each reuses frozen scores,
and each is registered with its bar before execution.

### C1 — H-SEL: selective prediction. Does MARS-C dominate on the risk–coverage curve?

A deployed omission finder may abstain. **Estimand:** omission precision as a function of coverage (share of
the ten slots actually filled), swept over the score threshold, per verifier family, with the B0 controls.
**Bar:** H-SEL passes if MARS-C's curve dominates every comparator's over the coverage interval [0.2, 1.0] with
a document-clustered band excluding crossing. A dominance result is a deployment claim recall@k cannot make;
it also directly answers "is the recall gain bought with precision?".

### C2 — H-SURF: does the gain survive controlling for surface cues?

**Estimand:** regress each verifier's unit score on unit character position in the source, unit token length,
and unit-summary lexical overlap; re-evaluate every arm on the **residualised** score. **Bar:** the MARS-C
advantage on the B2 crossed endpoint must survive residualisation with its interval still excluding zero.
This is the identification control a skeptical reviewer will ask for and the paper currently has no answer to.

### C3 — H-COMP: is MARS-C's signal complementary to the strongest baseline?

**Estimand:** rank-fuse MARS-C with the strongest off-the-shelf family and evaluate the fused emitter on both
endpoints plus the B0 controls. **Bar:** H-COMP passes if the fusion exceeds both constituents on omission
precision at equal coverage with a document-clustered interval excluding zero. This claim survives even if
MARS-C alone only ties on recall — "natural coverage supervision contributes signal no off-the-shelf verifier
carries" is a durable contribution.

### C4 — H-ATTR: where does the end-to-end error actually live?

**Estimand:** decompose each system's miss of a human-marked omitted ACU into (i) no inventory unit overlaps it
(inventory miss), (ii) a unit overlaps but is ranked outside the budget (verifier miss), (iii) a unit is ranked
in but the emission rule demoted it (rule miss). Per system, per pool. **No bar** — this is the "where is the
error" figure reviewers ask for, and it is the machine-computable half of the FINAL_PROPOSAL's Block 1
bottleneck audit. It also tells the project whether training is worth doing at all before any human is asked
to label anything.

### C5 — H-XEXT: the cross-extractor generalisation matrix

Training-extractor × evaluation-extractor, including the B4 unseen extractor, on held-out documents.
**Bar:** transfer to the unseen extractor must retain a majority of the within-extractor advantage with a
document-clustered interval excluding zero; otherwise the effect is extractor memorisation and is reported as
such.

---

## Wave D — registered now, human-gated, not executed this week

Registered before any outcome is read so that no artifact can be chosen after seeing results.

- **D1** current-emitter blinded precision study, drawn from the corrected dump, with a genuine adjudicator
  role. **This role does not exist**: the label app has no adjudicator and every analysis script takes
  `names[0]`/`names[1]`; "adjudicated" has meant *agreement-conditioned* (disagreements dropped) throughout
  this project. It must be built before D1 can mean what the FINAL_PROPOSAL says it means.
- **D2** the fixed-budget revision study (FINAL_PROPOSAL Block 3): generation arms and blinded review packets
  are built and frozen; every outcome needs two blinded reviewers plus adjudication.
- **D3** the matched-supervision training rows (FINAL_PROPOSAL's claimed dominant contribution). Registered
  with its pre-weakening stated honestly: `mc_train`'s own objective isolation already returned *objective
  REFUTED* (D−A = −0.002), the supplied-ACU task is near ceiling (AUROC 0.974), and R04 shows label source
  alone is worth +0.023 recall. The hypothesis under test is **population shift**, not objective. No new loss
  may be added to rescue it.
- **D4** the one-sided distribution-free bounded-mean bound on the per-document loss L_d required by the
  FINAL_PROPOSAL's confirmatory contract. The method (Hoeffding vs empirical-Bernstein vs Waudby-Smith-Ramdas
  betting) must be frozen in a dated amendment **before** any confirmation label is read. Sizing note:
  at the human study's 37–42 clusters, Hoeffding costs ~0.22 half-width, which may make the proposed 0.90
  precision target unreachable as specified.

**Honest statement of what this campaign cannot deliver.** Nothing in Wave D — and therefore nothing in the
FINAL_PROPOSAL's end-to-end semantic claim, its useful-yield estimand U_d, or its 0.90-precision confirmatory
contract — is obtainable without new blinded human annotation with real adjudication. If that effort is not
available, the paper must state that its end-to-end semantic precision is unestablished. It currently does.

---

## Standing operational rules for every job in this campaign

1. Never feed a glob to `mc_e2_eval.py`. Explicit comma manifest, `--strict-seeds --expect-seeds 3`.
2. `paired_recall.mean` is the **bootstrap** mean. Always report and bar on the **observed** contrast.
3. `--dump-emitted` fires only at the largest integer `k` and never for an adaptive budget. To dump at k=10,
   pass `--k 10` alone.
4. Every `sbatch` sets an explicit `--time`. CPU work goes to `--partition=infer --gres=none`.
5. Never cancel, requeue or preempt a job this session did not submit.
6. Anything projected past ~12 h is chunked at submission with offsets recorded.
7. `enable_thinking=False` for every Qwen call.
8. Use the plain judge prompt set; `--prompt-set anchored` is **rejected** (0.8307 vs 0.9524 coverage accuracy
   against blinded human labels).
9. Deploy before submitting; record the sha256 of every edited file on both sides.
10. Prereg status fields lie — A1/A2/A3/B0 all still read "REGISTERED-NOT-RUN" after executing. Check the
    Euler result JSON, never the status field.
