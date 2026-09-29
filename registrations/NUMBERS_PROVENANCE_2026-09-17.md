# Where every MARS-C number in the revised paper comes from

Every quantity below was produced on 2026-09-17 by the repaired evaluator
(`mc_e2_eval.py --strict-seeds --expect-seeds 3`), which records in each output JSON an `emitter_manifest`
block naming the exact score files it averaged and their SHA-256. Paths are relative to `~/results/marsc/` on
the Euler cluster; the same tree ships in `marsc_update.zip`. No model was retrained for R01/R01b/R02/R09/R11/R12.

| Quantity in the paper | Value | Artifact |
|---|---|---|
| TEST recall@10, human-fact consensus | 0.376536 | `r01_consensus/eval_test/e2_emitted.json` k=10 `humanfact+div1` |
| TEST recall@10, judge-label distilled consensus | 0.326605 | same, `judgefact.distilled+div1` |
| TEST paired gap, consensus | **observed +0.049931**; bootstrap mean +0.0506, CI [+0.0237,+0.0796] | same, `paired_recall` (the `mean` field is the bootstrap mean, not the observed contrast; the registered +0.05 margin is MISSED by 7e-05) |
| TEST sign / permutation | 149+/79−, p=4.1e−6; perm p=0.0005 | same, `sign_test`, `perm_test` |
| TEST paired gap, single-model | +0.044 [+0.015,+0.068] | `r01_consensus/eval_test_single/e2_emitted.json` |
| TEST verifier effect at fixed inventory | +0.1208 [+0.1003,+0.1443] | `r01_consensus/eval_test/…` `judgefact+div1` vs primary |
| TEST emission rule at fixed verifier | 0.266248 → 0.376536 | same, `humanfact` and `humanfact+div1` |
| TEST omission precision, per system | 0.720 / 0.692 / 0.752 / 0.755 | `r12_precision/eval_test/e2_emitted.json`, `omission_precision` |
| TEST omission precision, MARS-2 and single cells | 0.704 / 0.705 / 0.760 / 0.692 | `r12_precision/eval_test_fill/e2_emitted.json` |
| MARS-2 on TEST | 0.194539 (rule), 0.170644 (unruled) | `r01_consensus/eval_test_mars2/e2_emitted.json` |
| Validation recall@10 (three cells) | 0.399885 / 0.369247 / 0.311035 | `r09_validation/eval_validation/e2_emitted.json` |
| Validation, importance policy | 0.457 (+imp+div1) vs 0.400 (+div1) | `r09_validation/eval_policy/e2_emitted.json` |
| UniSumEval consensus pair | 0.347054 / 0.298347, +0.0483 [+0.0312,+0.0689] | `r01_consensus/eval_unisum/e2_emitted.json` |
| UniSumEval single-model pair | 0.388846 / 0.316954, +0.0722 [+0.0531,+0.0919] | `r01_consensus/eval_unisum_single/e2_emitted.json` |
| UniSumEval per-domain | see Appendix | `r01_consensus/eval_unisum/e2_emitted.json` `by_resource` |
| Adaptive budget, both pools | 0.368/0.301 and 0.427/0.360 | `r01_consensus/eval_{test,unisum}_adaptive/e2_emitted.json` |
| k-curve, both pools | k=1..50 | `r01_consensus/eval_{test,unisum}_kcurve/e2_emitted.json` |
| Label-efficiency curve | 0.409/0.402/0.395/0.398 | `r01_consensus/eval_curve/e2_emitted.json` |
| Equivalence at a 0.05 margin | 85-doc TOST [−0.034,+0.006] | `r05_equivalence/k10/label_efficiency_equivalence.json` |
| Human precision, rank ≤ 10, adjudicated | 0.640 / 0.870 / 0.825 / 0.783 | `r02_human_rank/human_precision_ranked.json` budget "10" — **these are the values the main table now prints** |
| Human precision, corrected emitter | 0.738 / 0.908 / 0.864 / 0.782 | `r02_human_rank/corrected/human_precision_ranked.json` — selection on rank; **appendix only**, removed from the main table |
| Annotated-vs-corrected emission overlap | 0.59–0.66 units; 48–70 of 100 items | `r02_human_rank/emitted_overlap.json` |
| Same-fact crossed cells, TEST | 0.9163 / 0.8991 triple-micro; **0.9140 / 0.8936 document-macro**, which is the estimator the printed CIs belong to | `r03_sumblind/crossed_test/summary_conditioning.json` |
| Same-fact crossed cells, validation | 0.9181 / 0.8957, 9,523 triples | `r03_sumblind/crossed_validation/summary_conditioning.json` |
| Summary-shuffled / fact-only recall | 0.440 / 0.458 | `r03_sumblind/eval_test/e2_emitted.json`, `r12_precision/eval_test/…` |
| Summary-shuffled / fact-only omission precision | 0.666 / 0.655 | `r12_precision/eval_test/e2_emitted.json` |
| Judge-vs-human agreement on shared cells | 0.7163, κ 0.4727, 63,343 cells | `r04_matched/e0_judge/relabel_report.json` |
| Matched label-source AUROC | 0.9740 vs 0.9613, +0.0128 [+0.0082,+0.0180] | `r04_matched/analysis_auroc/matched_auroc.json` |
| Matched control, arm R | 0.9751 vs 0.9575 | same |
| Crossed inventory × verifier (judged) | −0.0285 / +0.0075 / +0.0066 | `r07_crossed/analysis/crossed_precision.json` |
| Importance policy on TEST (secondary) | 0.457 vs 0.377; prec 0.677 vs 0.720 | `r10_policy_test/eval_test/e2_emitted.json` |
| Matching-rule sensitivity | ≥1/≥2/≥3 tokens, Jaccard 0.20/0.34 | `r11_matching/eval_{test,unisum}_{t1,t2,t3,j20,j34}/e2_emitted.json` |
| Glob audit | 121 patterns, 50 flagged (33 pseudo-seed, 11 repeated-seed, 6 empty) | `r08_globaudit/glob_audit.json` |
| Judge labels behind the judge-label verifier | 85,600 cells over 2,625 pairs | `~/data/plan2026/b1/training_rows.jsonl`; `~/results/mars2_gates/b21/xenc_seed20260906/training_config.json` |
| LLM labelling cost | 5,777 GPU-seconds = **1.6** GPU-hours | Slurm accounting, job 4119 (8 array tasks). The previously printed 5.8 h did not follow from 5,777 s; if the true total is per-array-task the whole figure must be re-derived from Slurm. |
| Inventory sizes | 94 / 106 / 307 and 40 / 44 units per pair | `~/results/mars3/m32/units_*.jsonl`, counted over the whole split |
| Inventory ceilings | 0.8607 val / 0.8966 test; spaCy 0.4611 | `~/results/mars3/m32/ceilings.json` |
| Verifier cost | 0.82–0.91 ms per unit | `g4/cost_table.json` |
| H-R1 registered readout | 0.2257 EN / 0.0799 RU | `article/iclr2027/hr1_results/analysis_{en,ru}.json`, `bars.rho` |
| Natural-benchmark interval | 0.4615 [0.2500,0.6923] | `article/iclr2027/strengthen_results/w16_natural_counter_reference.json` |

## Known provenance gap

The 1,268-row dump behind the A1 (natural PolyTope omissions) paragraph of Appendix~A1 is not in the release
and was not relocated while preparing this revision; that paragraph is flagged in place as unverified by the
packaged artifacts.


## Corrections applied 2026-09-17 (post-repair adversarial review, round 1)

| Item | Was | Now |
|---|---|---|
| Registered +0.05 margin | "cleared" | observed gap +0.049931 — margin missed by 7e-05; direction and CI stand |
| Seed-average regime per-pair cost | 0.70 / 0.08 s | 0.98 / 0.21 s — that regime also scores all three seeds |
| MARS-2 cost ratio | "ten times below" | 3.9x |
| LLM labelling | 5.8 GPU-hours | 1.6 GPU-hours from the cited 5,777 GPU-seconds |
| Main-table human precision | corrected-emitter 0.74/0.91/0.86/0.78 | rank<=10 adjudicated 0.64/0.87/0.83/0.78 |
| Crossed-cell point vs CI | 0.916 paired with a doc-macro CI | estimators named; 0.914 [0.898,0.929] is the matched pair |
| "85 documents" | 85 | 106 labelled documents (85 train + 21 calibration) vs 959; validation-only |
| "significant at every k" | every k under all tests | positive at every k, decisive at k=10 |

### Open, NOT closed (need cluster work before submission)

1. Per-seed calibration is applied on validation but not at test scoring; seed-averaged cells inherit a
   cross-seed monotonicity mismatch. Regenerate aggregated cells from existing checkpoints.
2. The crossed inventory x verifier bootstrap resamples summary pairs, not source documents. Intervals are
   provisional; the census main effects are unaffected.
3. `marsc_update.zip` was cut pre-repair (source commit 1751710c) and contains none of the R-series
   artifacts this document cites. Rebuild from the post-repair tree, re-run the anonymity gate, regenerate
   `ARCHIVE_VERIFICATION.json`.
4. Not locally verifiable in this checkout, flagged by review and unresolved: whether the R06 chain really
   evidences six off-the-shelf verifiers (MiniCheck and AlignScore may be absent from the fast chain); whether
   the Jaccard control content-filters emitted-unit tokens as well as ACU tokens; and the rank-average
   ablation's compatibility with the fixed -5 demotion.


## Corrections applied 2026-09-18 (external review: FULL_REVIEW / AS_IS_TO_BE / PAPER_CLAIM_AUDIT)

| Item | Was | Now | Source |
|---|---|---|---|
| Fixed-inventory verifier effect | +0.121 (bootstrap mean) | **+0.122** observed (0.3765356 - 0.2548314), CI [+0.100,+0.144] | PCA-002; R01 TEST |
| Rule effect at the same cell | +0.111 | **+0.110** observed (0.110288) | item 07 |
| UniSumEval consensus gap | +0.048 [+0.031,+0.066] | **+0.049** observed (0.0487074), CI **[+0.031,+0.069]** | PCA-005; r01_consensus/eval_unisum |
| UniSumEval regime contrast | 0.042 [-0.054,-0.031] | 0.042 observed, CI **[-0.053,-0.029]** | PCA-005 |
| "eight of nine domains" | unqualified | scoped to the **seed-average** regime; consensus instead loses on MultiWOZ by 0.003 | PCA-006 |
| UniSumEval denominator | 1,822 summaries | 1,822 over 224 documents, of which **1,813 over 222** are recall-eligible | PCA-007 |
| No-summary aligned precision | 0.656 | **0.655** (0.6554608) | PCA-004; R12 |
| LLM labelling cost | 1.6 GPU-hours / 5,777 GPU-seconds | **5.78 allocated GPU-hours / 20,797 GPU-seconds**, eight one-GPU array tasks | PCA-001; slurm job 4119 |
| MARS-2 before/after | unruled `full_recovery` 0.171/0.705 vs ruled `direct_enabled` 0.195/0.704 | one arm: **`direct_enabled` 0.168/0.729 -> 0.195/0.704** | PCA-011; r12_precision/eval_test_fill |
| k-curve claim | "every budget from one to fifty" | **every tested budget, k in {1,3,5,10,20,50}** | PCA-012 |
| Registered +0.05 margin | "the ordering and the margin transferred" | ordering transferred; the observed gain **missed** the margin | item 08 |
| Strict-match robustness | "+0.122 down to +0.028" | **+0.122 / +0.070 / +0.041** at one, two, three shared tokens; the +0.028 end was the withdrawn Jaccard branch | item 21 |
| R07 intervals | [-0.036,-0.022] and [+0.005,+0.010] with attribution language | **descriptive points -0.029 and +0.008, no intervals** (bootstrap resamples summary pairs, not documents) | PCA-003 |
| Human precision CI | [0.81,0.93] | **[0.80,0.93]** (0.80474) | item 32 |
| Human rates | "adjudicated" | **agreement-conditioned** throughout (disagreements dropped, not resolved) | item 12 |
| Aligned precision | unqualified | named **ACU-aligned omission precision**; only 0.173 of emissions align | item 15 |
| Deployment cost | 0.21 s / 3.9x, amortised | amortised figures kept **and** priced at one summary per source: ~0.43 s distilled, ~7.05 s Gemma+ent, ~1.9x | item 25 |
| R04 emitted result | mapped to `eval_test` | **`r04_matched/eval_test_d/`** | item 03 |

### Still open after this pass (unchanged by editing)

1. Per-seed calibration is applied on validation but not at test scoring; seed-aggregated cells inherit it.
   Needs the cluster rescore. (AS IS -> TO BE item 05.)
2. R07 document-clustered bootstrap not run; intervals withdrawn rather than corrected. (item 11)
3. `marsc_update.zip` still predates the repair and must be rebuilt from the final tree. (item 03)
4. Human validation targets the pre-repair emitter; a current-emitter blinded study is not run. (items 13, 14)
5. External matched-label (label-source) transfer not run. (item 23)
6. PCA-008/009/010 historical appendix families remain `insufficient_source`.


## Corrections applied 2026-09-18 (second pass: RECHECK.md R1-R8, PAPER_CLAIM_AUDIT C01-C06)

| Finding | Was | Now |
|---|---|---|
| R3 | abstract implied current-emitter human validation | abstract states a blinded study of **historical** outputs on agreement-filtered annotations; semantic precision of the corrected emitter **unestablished** |
| C05 | "+0.023 ... with -0.016 of aligned precision --- real" | recall and AUROC gains positive; aligned precision **-0.016 [-0.033,+0.000]**, not separated from zero |
| R6 / C04-4 | rule effect +0.111 (components + repair tables) | **+0.110** observed |
| R6 / C04-1 | repair table UniSumEval +0.048 [+0.031,+0.066] | **+0.049 [+0.031,+0.069]** observed |
| R6 / C04-2 | human distilled [0.81,0.93]; judge distilled 0.83 [0.71,0.93] | **[0.80,0.93]**; **0.82 [0.71,0.92]** (0.8249395) |
| C04-3 | budget curve printed bootstrap means | observed contrasts: TEST +.019/+.021/+.023/+.050/+.038/+.039; UniSum +.016/+.033/+.040/+.049/+.069/+.083 |
| R5 | "cost sits where we claimed"; "established by the census"; 0.03-0.23 as a bound | descriptive only; 0.032 and 0.23 named as **two different measurements**, not interval endpoints |
| R7 | appendix said the policy gate "stands for a better reason" | matches the main text: **original gate verdict invalidated**, precision rationale post hoc |
| R7 | appendix + Discussion said "the fault is the endpoint's" | narrowed to summary-sensitivity **on supplied ACUs**, not certification of generated emissions |
| R7 | controls asserted pair bootstrap is "narrower"; appendix inferred conservatism from 63% distinct clusters | both directional claims removed |
| R7 | CARE described as conferring a coverage guarantee by composition | qualified: risk guarantee conditional on CARE's own calibration and exchangeability assumptions |
| R7 | no-reuse cost said "inventory plus one verifier pass" | **inventory plus all three verifier passes**, matching the 0.43/7.05 s figures |
| R7 / item 26 | "single-model" in results, appendix, repair table | **seed-average** throughout |
| R7 / item 34 | "scored once" in method and controls | evaluated once frozen, then rescored by disclosed diagnostics |
| R7 / item 27 | "about ten facts each" | **median** of ten facts each |
| C03 | method said all intervals are paired document bootstraps | carves out the human study as **status-stratified item bootstraps** over its historical sample |
| (found in this pass) | Discussion said training buys precision over NLI | NLI is **indistinguishable on both endpoints**; the Discussion now matches the results section |

Main text still ends on page 9 (`page:mainend=9`); every addition above was paid for by compressing
secondary audit prose, not by dropping a claim.

### Still open (unchanged by this pass, and not closable by editing)

1. **C01 / R1 calibration.** Validation uses fitted per-seed calibration, test/external uses raw sigmoid.
   Every cell formed by averaging probabilities across seeds is provisional until the existing checkpoints are
   rescored under one contract. No inspected artifact shows a sign reversal; the exact values are what is
   uncertified.
2. **R2 supplement.** `marsc_update.zip` is still the pre-repair archive (sha256 5af597b6..., 460 entries).
3. **R4 / C03 human uncertainty.** Intervals still resample items within status strata, not source-document
   clusters. At rank <=10 only 47/40/44/53 agreement-conditioned items identify the four rates.
4. **R5 R07 estimator.** Still groups on `pair_id`; only the inference and intervals were withdrawn.
5. **R8.** No external matched-label transfer, no final-emitter human study, main table still omits the
   NLI/SummaC rows, MARS-C still begins on page 5.
6. **C06.** The 85,600-label / 2,625-pair B1 census is not evidenced inside the bounded artifact set.


## Corrections applied 2026-09-18 (third pass: the `marsc_strengthen_20260918` campaign)

Registered in `research/marsc_strengthen_20260918/PREREG.md` **before** execution; full results in
`research/marsc_strengthen_20260918/RESULTS.md`. Artifacts under `/home/user/results/marsc_strengthen/`
on Euler, each directory carrying `SHA256SUMS` and `complete.json`.

| Item | Was | Now | Block |
|---|---|---|---|
| R07 crossed **interaction** | +0.0066, pair-level [+0.0019,+0.0116] | **WITHDRAWN** — document-clustered [−0.0024,+0.0161], p=0.149 | A2 |
| R07 inventory main effect | −0.029, no interval (withdrawn) | **−0.029 [−0.045,−0.013]**, doc-clustered, p=0.0008 | A2 |
| R07 verifier main effect | +0.008, no interval (withdrawn) | **+0.007 [+0.003,+0.012]**, doc-clustered, p=0.0024 | A2 |
| Human precision intervals | item-level within status strata | doc-clustered; median widening only **1.031×**, zero degenerate draws (136 documents) | A3 |
| "two annotators", Q1 | implied two independent raters | Annotator 2 answered `A` on **all 400** Q1 items → Q1 κ **exactly 0**; median inter-item gap **0.94 s** vs Annotator 1's **49.6 s**. Q2 is sound (κ 0.7249) | A3b |
| Union / ensemble inventory arms | validation 0.4025 vs primary 0.3999 | under ONE calibration contract **0.3894 vs 0.4029** — the parity was a mixed-contract artefact | A1a |
| Label-efficiency curve | 0.4085/0.4023/0.3953/0.3982, spread 0.0132 | raw contract **0.4097/0.4027/0.3923/0.4011**, spread **0.0174** | A1c |
| Matched label-source AUROC | +0.0128 [+0.0082,+0.0180] (calibrated) | **+0.0128 [+0.0082,+0.0181]** from raw logits — survives the repair | A1d |
| Calibration exposure | "every seed-aggregated cell provisional" | bounded: single-seed top-k **exactly** invariant (0/1466); exposure only in seed averaging (154/1466 consensus, 510/1466 single-model); `leadunion` reproduces the headline to **<3e−4 on all three splits** | A1b, A1e |
| Rank-average (seedrank) ablation | "scores go negative, the −5 demotion no longer dominates" | **rule never fired**: `mc_diversity.py` skips units ≤ −9.0 and seedrank spans [−438,0]. Row withdrawn | A5 |
| Jaccard matching sensitivity | withdrawn (asymmetric token filter) | **restored and stronger** under a symmetric filter: TEST j20 **+0.0451 [+0.0295,+0.0610]**, TEST j34 **+0.0151 [+0.0059,+0.0235]** (crossed zero before). Shared-token rows exactly invariant (252 scalars, zero differ) | A4 |
| B1 census (C06) | "not evidenced inside the bounded artifact set" | **CLOSED** — recomputed 85,600 cells / 2,625 pairs exactly, `labels_none = 0`; newly citable: **1,411 distinct documents**, and 954 cells (1.11%) are human-labelled, not judge | A6 |
| Amortisation denominator | 11.8 summaries/source | **7.716** split-wide (the 11.8 was the timing window's) | B6 |
| Distilled cost at one summary/source | ~0.43 s, "advantage ~1.9×" | **0.716 s**, advantage **~1.15×** — the published figure charged one-off training to inference | B6 |
| Gemma decomposition cost | 6.637 s/document | **stands** — 6.760 s re-timed as active generation (ratio 1.018) | B6 |
| Verifier cost parity | unstated | MARS-C **3.08 ms/unit** (3 seeds) vs SummaC-ZS **3.38** — the fixed-inventory swap is cost-neutral | B6 |
| R06 "six off-the-shelf verifiers" | implied complete | **MiniCheck and AlignScore had never been run**; and every published off-the-shelf row is **TEST-only** (`r06b_fast` wrote TEST into a dir named `val/`) | B1 |
| MiniCheck throughput | 1.36 units/s (→ 85 GPU-h) | **41.9 units/s**; AlignScore **2035 units/s**. Whole block ≈**2 GPU-h** | B1 |
| QAPyramid overlap | "zero overlap with our sources" | **inverted** — 499/499 documents join `rose_cnndm`; **100 sit in our sealed TEST role**, 282 in TRAIN. No transfer arm may be planned on it | B7 |
| ComposoAI/OmissionBench | unverified (quoted from a proposal) | **real and disjoint** — 161 transcripts, 0 exact and 0 near-duplicate matches against our 1,498 documents | B7 |
| CI-site audit | 8 sites named | **34 classified**: 24 doc-clustered, 7 not, 3 not intervals. Three unnamed: `mc_equivalence.py:63` and `b6_rerank.py:65` (both correct), `a5_metric_selection.py:48` (defective) | A8 |

### New results this campaign adds (not corrections)

| Quantity | Value | Artifact |
|---|---|---|
| **Same-fact crossed accuracy by verifier family** (validation, doc-macro, chance 0.500; 9,523 triples / 569 facts / 138 docs) | MARS-C **0.9170** [0.8971,0.9351]; NLI 0.8911; judge-label twin 0.8872; SummaC-ZS 0.8658; lex token 0.8304; lex ROUGE-L 0.8056; MiniLM 0.7885 | `b2/crossed_validation/crossed_families.json` |
| **H-COND paired margins** (doc-clustered, one-sided LB95) | vs NLI +0.0259 (LB +0.0137); vs judge twin +0.0298 (+0.0146); **vs SummaC-ZS +0.0513 (+0.0379)**; vs lex token +0.0866; vs ROUGE-L +0.1114; vs MiniLM +0.1286 — **all clear the bar on validation; TEST pending** | `b2/hcond_verdict.json` |
| Selective prediction: real-vs-no-summary omission-precision gap | **+0.2480 [+0.1597,+0.3349] at 20% coverage** (vs +0.065 at the full ten-slot budget), 17/17 grid points for both controls | `c1c3/` |
| MARS-C vs SummaC-ZS on the risk–coverage curve | dominates **16/17** (distilled) and **17/17** (gemma+ent) grid points | `c1c3/` |
| MARS-C judge-label arm, sealed TEST | omission precision **1.000 [1.000,1.000] at 10% and 20% coverage** against its shuffled control at ~0.68 | `c1c3/` |
| Judge-confirmed omissions per 10-slot list (**lower bound**, validation) | humanfact 7.60 [7.14,8.05]; vs mars2 **+0.7514 [+0.5119,+0.9837]**; vs judgefact **+0.2751 [+0.1402,+0.4233]**. Distinct per document 17.32 vs 11.07 | `b5b7/` |
| Unseen third extractor (Qwen3.8-27B) inventory | 110.3 units/pair, human-omitted ACU ceiling **0.906** weak / 0.291 strict vs development gemma+ent 106.2 / 0.897 / 0.289 | `b4/inv/` |
| vLLM throughput | **25.9× faster** than HF on the same model and prompt (0.521 vs 13.506 GPU-s/document) | `b6b8/` |
| Error attribution (spaCy, validation, top-10) | **53.9% inventory misses**, 23.8% verifier, 2.6% rule, 19.7% hits → 67.1% [0.615,0.733] of misses are inventory | `c2c4/` |
| Surface-cue control | omission score is 35.0% surface-predictable but only **1.4% from candidate-blind cues**; 30.3% from unit–summary Jaccard at a negative coefficient | `c2c4/` |

### Registered bars that FAILED (recorded as registered)

1. **H-SEL** (C1): MARS-C does **not** dominate the risk–coverage curve — 0/10 and 1/10 comparators. Cheap
   lexical and embedding counters are better selective predictors at low coverage (0.984/0.980/0.979 at 20%
   coverage against MARS-C.distilled's 0.919).
2. **H-COMP** (C3): all eight rank fusions fail; the primary is **−0.0094 [−0.0194,−0.0003] below** MARS-C
   alone. The pre-declared null-control fusion behaved as a null (+0.0065 [−0.0017,+0.0158]), so the
   refutation is not a fusion-rule artefact. Salvageable secondary: the fusion matches SummaC-ZS's recall@10
   exactly (0.385) at **+0.035** omission precision.
3. **A2** interaction: withdrawn, as the registration said it might have to be.

### Still open after this pass

1. **The human study's support question needs re-collection** (A3b). `marsc_update.zip` still predates the
   repair and must be rebuilt **after** the campaign's numbers are final (A7, deliberately deferred).
2. H-COND's sealed-TEST half, B1's TEST rows, B3's transfer eval, B4's unseen-extractor scoring, the B0
   grids, C2's residualised crossed endpoint and D2's revision generation were still on the Euler queue when
   this block was written.
3. PCA-008/009/010 and the A1 1,268-row dump remain `insufficient_source`; neither is closable by compute.
4. The FINAL_PROPOSAL's 0.90-precision confirmatory contract is **unreachable at the current human-study
   scale**: a one-sided 95% Hoeffding bound on its bounded per-document loss costs 0.2012 half-width at 37
   clusters and 0.1888 at 42, so it needs ~120–200 clusters or a revised target.
