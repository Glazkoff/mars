# MARS-C IEEE Access strengthening — registration of 2026-09-22

**Status.** Written before any output of the blocks below was read. Governing plan:
`MARS_IEEE_ACCESS_PLAN_2026-09-22.md` (hypotheses H-J3 and H-J6). Standing rules inherited from
`research/marsc_strengthen_20260918/PREREG.md`: every new emitter arm is scored with its shuffled-summary
and no-summary controls in the same job; recall@10 is never reported without omission precision beside it;
intervals resample source documents; the `observed` contrast is quoted, never the bootstrap mean; a
registered verdict is never restated over a comparator family it was not registered on; a bar that misses
is closed, not iterated.

**Pool status, declared.** UniSumEval has already been read by this project for recall@10 and omission
precision (G7, B3, R11, T02, T04, T08, C4). Nothing in this registration was tuned on it and no setting is
changed for it, but it is not a fresh pool: every block below is **post-hoc secondary** evidence and is
reported as such. The RoSE held-out split is not re-read by any block here.

**Settings frozen.** The frozen Gemma+ent UniSumEval inventory (`g7_unisum/units_gemma+ent_unisum.jsonl`,
559,048 units), the frozen `div1` emission rule with cap 1, k = 10, the evaluator `mc_e2_eval.py` with
`--strict-seeds --expect-seeds 3`, the registered matching rule (sentence overlap plus one shared content
token), the deployed MARS-C arms (`g7_unisum/val/gemma+ent_{humanfact,xenc}_seed*_div1.jsonl`), the
repaired scorers of 2026-09-21 (`b9_judge_units.py --score-mode exact`; `b10_verifier_swap.py` with
single-BOS tokenisation and right-side truncation), the crossed-cell estimator `mc_summary_conditioning.py`
imported unmodified by `b2_crossed_families.py`, bootstrap seed 20260917 and 2,000 draws. Candidate text is
clipped to 380 words for every cross-encoder family, as on RoSE.

---

## H-J3a — the crossed candidate-conditioning endpoint on UniSumEval

**Question.** Does the same-fact crossed accuracy, on which any summary-blind scorer is pinned at exactly
0.500, reproduce on the external pool: MARS-C above every registered comparator, and the 31B judge
indistinguishable from it?

**Cells.** UniSumEval labels every human-validated key fact of a document against every summary of it
(1,822 summaries over 224 documents; identical fact lists across a document's systems, verified 2026-09-22).
`docs_unisum.jsonl` is built in the `e0` schema (facts = key facts; summaries = per-system candidates with
per-fact labels 1 = covered, 0 = omitted) by `code/hj3_make_docs_unisum.py`; the crossed-cell census
computed before scoring is ≈ 17,200 cells and ≈ 26,500 same-fact triples over 224 documents.

**Families.** The registered family exactly as on RoSE (`humanfact` three seeds, `judgefact` three seeds,
`nli`, `summac_zs`, `lex_token_recall`, `lex_rougeL_recall`, `emb_minilm`) and, in a separately labelled
block, the three post-registration comparators (`factcg`, `minicheck7b`, `judge_gemma31`) scored through
`b12_hcond_precompute.py` into `b2`'s cache. Primary = `humanfact`.

**Estimand.** Document-macro crossed accuracy; document-clustered paired margin MARS-C minus family with
its one-sided 95 % lower bound.

**Bars.**
- **PASS** if MARS-C's lower-bound margin is above zero against every registered off-the-shelf comparator
  (`nli`, `summac_zs`, the two lexical counters, `emb_minilm`) on UniSumEval. The RoSE conjunction stays
  the registered verdict; this block adds a third pool to the sentence, it does not re-decide the verdict.
- **FAIL** if any registered comparator's lower bound is at or below zero; the Discussion then scopes the
  crossed-design sentence to "on the held-out split".
- The three post-registration comparators are reported with their margins and no bar; the prediction,
  written here, is that the judge again does not separate (RoSE validation margin +0.007, LB −0.008).
- A family whose two-sided interval covers 0.500 is reported candidate-blind, as on RoSE.

## H-J3b — the emitted endpoint on UniSumEval for the three 2026 comparators

**Question.** Do the recall@10 leads over the 31B judge and Bespoke-MiniCheck-7B, and the tie with FactCG,
measured on the RoSE held-out split at fixed inventory, reproduce on the external pool, with omission
precision beside them and the two summary-blind controls attached to every new arm?

**Arms.** Deployed `humanfact+div1` (primary) and `judgefact+div1`; `judge_gemma31`, `factcg`, `minicheck7b`,
each in real, shuffled (seeded derangement within resource × system, seed 20260917) and empty candidate
mode; every arm through the frozen rule and evaluator; observed contrasts and document-clustered intervals
from `b3_contrast.py` over the per-pair dump.

**Bars.**
- **PASS** if the observed recall@10 contrast primary minus judge and primary minus MiniCheck-7B both have
  document-clustered 95 % intervals excluding zero in MARS-C's favour, and omission precision against each
  of the three has an interval that includes zero or favours MARS-C.
- **FAIL** otherwise, reported per comparator; a comparator that leads on either endpoint is stated as a
  lead. The FactCG contrast is reported with no bar (it was level on RoSE).
- Gameability: for every new arm, the shuffled and empty controls are reported on both endpoints; the
  prediction is that the precision penalty for ignoring the candidate holds in every cell, as it did for
  every system on both pools so far, and that the recall effect is weaker here than on RoSE.

## H-J6 — error attribution on the deployed inventories (harvest, no bar)

C4's registered attribution (`c2c4/c4_attribution/c4_attribution_{validation,test,unisum}.json`, job of
2026-09-18) already covers the Gemma+ent and distilled inventories on all three pools. No new computation:
the figure in the manuscript is drawn from those files, and the text quotes the observed shares with their
document-clustered intervals (held-out split, pipeline of record: inventory 0.153 [0.123, 0.190] of misses,
verifier 0.763 [0.726, 0.795], rule 0.084 [0.067, 0.104]; ceiling 0.897).

## Not registered here

No human study (H-J1, H-J2, H-J4) is registered by this file; their designs are in the plan and each needs
its own registration once raters are confirmed. D1 stands as human evidence; the provenance check of
2026-09-22 and its resolution are in `research/marsc_strengthen_20260918/D1_PROVENANCE_2026-09-22.md`, and
no block here substitutes for a human study.

## Artifacts

All outputs under `/home/user/results/marsc_strengthen/hj3/` on Euler: `docs/`, `hcond_unisum/` (cache,
`crossed_families.json`, `b2_stderr.log`), `scores_unisum/<system>/`, `eval_unisum/` (evaluator output,
per-pair dump, `b3_contrasts.json`), each with `started.json` / `complete.json` receipts and `SHA256SUMS`.
Job scripts: `slurm/hj3_hcond_unisum.sbatch`, `slurm/hj3_score_unisum.sbatch` (one submission per system),
`slurm/hj3_eval_unisum.sbatch`.

---

## Amendment 1 (2026-09-22, after H-J3a was read, before H-J3c ran)

**H-J3a outcome, recorded:** PASS. On UniSumEval's crossed cells the human-fact verifier reads 0.864
[0.853, 0.875] document-macro; every registered off-the-shelf comparator is cleared with its one-sided 95 %
lower bound above zero (margins +0.017 to +0.132); FactCG (+0.031, LB +0.024) and Bespoke-MiniCheck-7B
(+0.025, LB +0.019) are cleared; the 31B judge reads +0.006 with LB +0.001 and a two-sided interval
[−0.0004, +0.011] that includes zero, which is not read as separation. The judge-label twin reads 0.869
[0.860, 0.879], −0.005 [−0.011, +0.001] relative to the human-fact verifier: the label-source separation on
this endpoint does not reproduce externally.

**H-J3c (added).** The H-SURF estimator applied to UniSumEval: the overlap surface model fitted on the RoSE
validation cells per checkpoint seed (C2's contract, `c2_crossed_residual.py` unmodified) and applied
unchanged to the UniSumEval crossed cells, for the human-fact verifier and the judge-label twin. **No bar;
post-hoc secondary.** Prediction, written before the run: the judge-label twin loses more of its crossed
accuracy under residualisation than the human-fact verifier, as on RoSE (0.893 → 0.667 against 0.914 → 0.881);
if it does not, that is reported as the external boundary of the H-SURF mechanism claim. Output:
`hj3/surf_unisum/`.

## Amendment 2 (2026-09-22, H-J3c outcome, recorded after the run)

**H-J3c outcome: the prediction failed.** With the overlap surface model fitted on RoSE validation and applied
unchanged to the UniSumEval crossed cells, the surface-only arm reads 0.776 [0.761, 0.791]; the human-fact
verifier falls from 0.864 to 0.726 [0.707, 0.743] (retaining 0.62 of its advantage over chance) and the
judge-label twin from 0.869 to 0.796 [0.782, 0.809] (retaining 0.80); residualised margin −0.070 [−0.091,
−0.049] in the twin's favour. On RoSE the same estimator gives 0.917 → 0.886 against 0.887 → 0.663. The
H-SURF mechanism reading is therefore scoped in the manuscript to the held-out split, in the results, the
contribution list, the Discussion and Figure 3. Output: `hj3/surf_unisum/c2_crossed_residual.json`.

## Amendment 3 (2026-09-22, before H-J5 ran)

**H-J5 (added): deployment cost, a measurement with no bar.** For MARS-C (one verifier seed; the emitter of
record is three), FactCG-DeBERTa-v3-L, windowed RoBERTa-large-MNLI, AlignScore (HF, mars venv) and
Bespoke-MiniCheck-7B and the Gemma-4-31B judge (vLLM venv, one process each): resident weight bytes, peak
GPU allocation at the batched and the batch-1 setting (HF), nvidia-smi used memory after engine start
(vLLM, whose pool is set by `gpu_memory_utilization` and is not a property of the model), batch-1 request
latency per unit (300 units drawn uniformly, 20 warm-up calls discarded, CUDA synchronised) and batched
throughput over every unit of 200 pairs drawn **uniformly at random** (seed 20260922) from the RoSE
validation Gemma+ent inventory, never the first 200. Premise and claim are the systems' own contracts (380-word
candidate; unit text for MARS-C, rendered 96-word claim for the comparators). Reported as measured; the only
sentence it may change is the Discussion's deployment statement, which today rests on weight size alone.
Output: `hj5/deploy_cost_*.json`. Script: `code/hj5_deploy_cost.py`, job `slurm/hj5_deploy_cost.sbatch`.

## Amendment 4 (2026-09-22, H-J5 outcome, recorded after the run)

Measured on one H200, 200 uniformly drawn validation pairs (20,218 units), 300 latency units: MARS-C one
seed 1.51 GiB fp32 weights, peak 3.62 / 1.57 GiB (batch 64 / 1), batch-1 13.6 ms (p95 13.9), 1,279 units/s;
FactCG 1.62 GiB, 6.42 / 3.22 GiB, 13.6 ms, 783 u/s; windowed RoBERTa-MNLI 2.51 / 2.21 GiB, 4.5 ms (p95 32.8),
1,857 u/s; AlignScore 2.48 / 2.22 GiB, 4.4 ms (p95 35.0), 1,450 u/s; Bespoke-MiniCheck-7B 14.41 GiB bf16 on
disk, 7.5 ms, 1,199 u/s; Gemma-4-31B judge 58.25 GiB, 24.1 ms (p95 27.6), 303 u/s. Table
`tables/marsc_deploy.tex`; the Discussion's deployment sentences now rest on these numbers. Receipts:
`hj5/deploy_cost_*.json`.

## Amendment 5 (2026-09-22, H-J3b outcome, recorded after the run)

**H-J3b outcome: FAIL, reported.** On UniSumEval at fixed inventory and rule (1,813 summaries, 222
documents; jobs 6154/6155/6158 scoring, 6159 evaluation) FactCG reads 0.384 recall@10 / 0.570 omission
precision, Bespoke-MiniCheck-7B 0.384 / 0.582 and the judge 0.370 / 0.585 against MARS-C's 0.347 / 0.544;
observed document-clustered contrasts MARS-C minus comparator: recall −0.037 [−0.052, −0.022], −0.037
[−0.054, −0.021], −0.023 [−0.039, −0.008]; precision −0.025 [−0.043, −0.006], −0.037 [−0.057, −0.019],
−0.041 [−0.061, −0.022]. Controls: every comparator's real-summary arm beats its shuffled and empty arms on
precision by 0.08–0.10; no summary-blind arm wins on recall on this pool. Package effect over the judge-label
twin on the same rows: +0.049 [+0.031, +0.068] recall, +0.022 [+0.003, +0.041] precision. The manuscript
scopes the emitted-endpoint lead to the held-out split (abstract, contribution 1, Discussion, identification
table) and adds `tables/marsc_unisum_comparators.tex`. Artifacts: `hj3/eval_unisum/`.

---

## Amendment 6 (2026-09-22, before H-J9 ran) — H-J9: does the natural-coverage package transfer to a stronger verifier backbone?

**Why now.** H-J3b showed the three 2026 comparators ahead of MARS-C on both emitted endpoints on the external
pool, and the length diagnostic ruled out the 380-word premise clip as the cause (99 % of UniSumEval
candidates are shorter than the clip; the deficit sits on summaries under 200 words). The remaining
explanation is verifier competence: MARS-C's encoder is DeBERTa-large-MNLI, FactCG's is a DeBERTa-v3-large
trained on synthetic factual-consistency graphs. If the package is what the paper says it is, training it on
the stronger encoder should carry the natural-coverage signal there.

**Design.** The deployed recipe unchanged (`mc_train.py --arm A --lam 1 --repr crossenc --steps 600`, lr
2e-5, seeds 20260906/07/08, `--save-model`), with `--backbone` pointed at the FactCG-DeBERTa-v3-Large snapshot
(its 2-class head is replaced by the recipe's fresh 1-logit head; the encoder weights are FactCG's). Two label
sources on the identical cells: human (`e0`, the deployed package) and the Gemma-4-31B judge twin
(`r04_matched/e0_judge`), so the design is backbone × label source with the DeBERTa-large-MNLI cells already
in the record (deployed human-fact verifier; `r04judge` twin). Every new arm is scored on the RoSE held-out
split and on UniSumEval over the frozen Gemma+ent inventory in real, shuffled and empty candidate modes, passed
through the frozen `div1` rule and `mc_e2_eval.py` with explicit three-seed manifests, and put on the crossed
H-COND endpoint on both pools with the cached comparator families. The DeBERTa-xlarge-MNLI size control is
deferred (scoring cost), and no other hyper-parameter is touched.

**Primary bar, H-J9a (package transfer).** PASS if the human-package verifier on the FactCG encoder leads
zero-shot FactCG at fixed inventory and rule on recall@10 with a document-clustered interval excluding zero
on **both** pools, with omission precision not lower on both (interval including zero or favouring the
trained verifier). FAIL otherwise, reported per pool; if the trained verifier is below zero-shot FactCG, the
package does not transfer to this encoder at this recipe and the paper says so.

**Secondary, no bar.** (i) Label source at the new backbone: human minus judge twin on both endpoints, both
pools. (ii) The new verifier against the deployed MARS-C and against MiniCheck-7B and the judge. (iii) H-COND
crossed accuracy of both new arms on both pools, with margins against the cached comparator families.
(iv) The summary-blind controls of every new arm on both endpoints.

**Predictions, written before the run.** The trained FactCG encoder leads zero-shot FactCG on both pools on
recall@10 (the package adds coverage supervision the synthetic pretraining lacks); the judge-twin gap at the
new backbone is small on RoSE and null externally, as at the old one; the precision contrast against zero-shot
FactCG is the uncertain one. If the primary bar passes, the manuscript reports the new verifier as a
registered extension beside the pipeline of record, not as a replacement chosen after the fact.

**Artifacts.** `hj9/models/`, `hj9/scores_{test,unisum}/`, `hj9/val_{test,unisum}/`, `hj9/eval_{test,unisum}/`,
`hj9/hcond_{test,unisum}/`; jobs `slurm/hj9_train_score.sbatch` (one per label source),
`slurm/hj9_eval.sbatch`, `slurm/hj9_hcond.sbatch`. The RoSE held-out split is re-read (post-hoc secondary;
`MC_TEST_REREAD=yes` declared here), UniSumEval likewise.

## Amendment 7 (2026-09-22, before H-J10a ran) — H-J10a: the same-fact crossed endpoint on a third pool of another genre

**Pool.** `ComposoAI/OmissionBench` (CC BY 4.0; synthetic clinical consultations built from PriMock57 and
ACI-Bench plus authored and trap-blind strata; no real patient). B7 measured it disjoint from every RoSE
document (0 exact, 0 near-duplicate matches). Each evaluation omission pair is a consultation transcript, a
verified clean note, and an errored note from which exactly one stated fact was removed, with the removed
statement given. Fetched to Euler `~/data/omissionbench/` (pairs, transcripts, fact sheets) on 2026-09-22;
nothing from it has been scored.

**Census, written before scoring** (`hj10_make_docs_omb.py`, sha256 in `hj10/docs/census_omb.json`):
primary set = the 150 `complete`-residual omission pairs over 95 consultations (150 facts, 426 crossed
cells, 276 same-fact triples); secondary set = all 293 evaluation omission pairs over 112 consultations
(248 facts, 982 cells, 793 triples; residual levels complete 150 / partial-weak 86 / partial-strong 57;
severity critical 151 / supporting 103 / peripheral 39; strata aci 126 / primock 94 / authored 52 /
trapblind 21). Note length: median 370 words (primary) and 444 (all), maximum 1,077; 49 % / 53 % of notes
exceed the deployed verifier's 380-word candidate clip, and in 20 % / 16 % of pairs the first removed site
lies beyond word 380, where the as-deployed reading cannot see the removal at all.

**Design.** The docs files put the removed statements as the facts of a consultation, the clean twin as a
note that states every one of them, and each errored note as the note its own fact was removed from (every
other fact present, since a pair changes one fact and nothing else). `b2_crossed_families.py` then forms
the same-fact crossed triples and estimates crossed accuracy exactly as on RoSE and UniSumEval (document =
consultation; 2,000-draw consultation-clustered bootstrap, seed 20260917). Two candidate readings, both
declared here: *as deployed* (the published path, first 380 words) and *windowed* (a candidate longer than
380 words is scored once per overlapping 380-word window at stride 190 and the cell's omission score is the
minimum over windows; a new `--window/--stride` option of b2 whose default leaves the published path
byte-identical; applied to the checkpoint families only, since the zero-shot NLI scorer already windows and
FactCG, MiniCheck-7B, the judge, SummaC-ZS, the lexical and the embedding families read the whole note).
Families: the registered seven (humanfact, judgefact, nli, summac_zs, lex_token_recall, lex_rougeL_recall,
emb_minilm), the three post-registration comparators (FactCG, MiniCheck-7B, Gemma-4-31B judge) and, if their
checkpoints exist when the job starts, the two H-J9 FactCG-encoder arms — each in a separate block, as in
Table `tab:marsc_conditioning`. `hj10_twin_metric.py` also reports the benchmark's own statistic (paired
discrimination of the errored note against its clean twin, ties half, chance 0.500) from the same cell
scores, pooled over pairs and by residual level, severity, stratum and clip position.

**Primary bar, H-J10a.** On the primary set, the *windowed* human-fact verifier: PASS if its
consultation-macro crossed accuracy has a bootstrap interval excluding 0.500 and it leads every registered
off-the-shelf family (nli, summac_zs, lex_token_recall, lex_rougeL_recall, emb_minilm) with a one-sided 95 %
lower bound above zero — the H-COND bar as applied on UniSumEval in H-J3a. FAIL otherwise, reported per
family. The windowed reading is primary because half the notes exceed the clip; the as-deployed reading is
reported beside it as the number the pipeline of record gives on this pool.

**Secondary, no bar.** (i) The as-deployed reading of every family. (ii) The three comparators and the
H-J9 arms, in their own block, never restating the registered verdict over them. (iii) The judge-label twin
(package effect) on this pool. (iv) The all-pairs set and its residual-level, severity and stratum
breakdowns. (v) The benchmark's own paired discrimination per family, beside its published judge rows
(best single-prompt gpt-5.4 design 0.634; naive coverage recipe 0.809; a two-stage fact-extraction pipeline
0.786 on a 151-pair subset), stated with the caveat that our number is given the removed fact.

**Predictions, written before the run.** The windowed human-fact verifier is well above 0.500 and leads the
lexical, embedding and SummaC families clearly; the NLI margin is the uncertain one (it was +0.068 on
UniSumEval); the as-deployed reading is lower than the windowed one by roughly the share of removals beyond
the clip; discrimination falls from complete to partial-weak to partial-strong residual, since a surviving
restatement is, for a fact-level verifier, a conveyed fact; the judge-label twin is within a few points of
the human-fact verifier, as on UniSumEval; the H-J9 arms are reported without a prediction.

**Artifacts.** `hj10/docs/`, `hj10/hcond_{complete,all}[_ext][_w]/crossed_families.json`,
`hj10/twin_{complete,all}[_w].json`; job `slurm/hj10_hcond_omb.sbatch`; code `hj10_make_docs_omb.py`,
`hj10_twin_metric.py`, the `--window/--stride` option in `b2_crossed_families.py`. No RoSE split is read.

## Amendment 8 (2026-09-22, H-J10a outcome, recorded after the run)

**Verdict: FAIL** on the registered bar (job 6244; `hj10/hcond_complete_w_ext/crossed_families.json`,
`hj10/twin_complete_w.json`; SHA256SUMS in `hj10/`). The windowed human-fact verifier on the primary set
reads a consultation-macro crossed accuracy of 0.937 [0.893, 0.974] (as deployed 0.853 [0.810, 0.892]; the
30 pairs whose removal lies beyond the 380-word clip are pinned at 0.500 as deployed and read 0.933 windowed,
so the clip accounts for the whole difference). It clears the two registered entailment families by wide
margins (windowed RoBERTa-MNLI 0.665, margin +0.272, LB +0.208; SummaC-ZS 0.664, +0.273, LB +0.214) and does
**not** clear the three registered surface families: lexical token-recall 0.944 (margin −0.007, LB −0.048),
ROUGE-L recall 0.946 (−0.009, LB −0.044), MiniLM cosine 0.940 (−0.003, LB −0.048). The bar required every
registered off-the-shelf family to be cleared; three are level with MARS-C, so it fails and is reported as
failed. Post-registration block (no bar): FactCG 0.986 and the Gemma-4-31B judge 0.978 are above MARS-C
(−0.049, LB −0.086; −0.041, LB −0.079); Bespoke-MiniCheck-7B 0.867 is below it (+0.070, LB +0.014). The
judge-label twin reads 0.918 (+0.019 [−0.020, +0.057]); the H-J9 arms read 0.783 (human package on the FactCG
encoder) and 0.919 (judge twin on that encoder).

**Predictions scored.** Above 0.500 and clearing the entailment scorers: right. Clearing the lexical,
embedding and SummaC families: right for SummaC, wrong for the lexical and embedding counters. The clip
costing roughly the beyond-clip share: right (0.853 → 0.937; 20 % of pairs at 0.5 vs 0.93). Residual-level
ordering complete > partial-weak > partial-strong: right (0.947 / 0.919 / 0.746 against the clean twin on
the all-pairs set; every family falls on partial-strong, lexical 0.711, FactCG 0.807, judge 0.842). Judge
twin within a few points: right (0.918 vs 0.937 primary; 0.908 vs 0.899 all pairs).

**Why the surface families are level (post-hoc, descriptive; `hj10/overlap_census_three_pools.json`).**
The two notes of a triple are one text minus one deleted span, so any counter of the fact's wording
separates them: the lexical gap between the conveying and the omitting candidate is positive in 0.906 of
OmissionBench triples against 0.731 on RoSE and 0.786 on UniSumEval, whose conveying and omitting summaries
are written independently. The facts themselves are no more verbatim here (median token recall of a
conveyed fact 0.63, against 0.80 on RoSE and 0.64 on UniSumEval). On a single-edit pool the crossed endpoint
certifies that a scorer reads the candidate and stops discriminating among scorers that do.

**Consequence for the paper.** The crossed-endpoint lead over the registered comparator family is scoped to
the two summarisation pools; the manuscript reports the clinical pool as a boundary of the endpoint, in a
new table (`tab:marsc_omb`), a results paragraph, an appendix record, the abstract and the limits. The
as-deployed clip is reported as a deployment limitation for long candidates, with the windowed reading
as the remedy (an inference-time option; no retraining, no threshold).

**Secondary set.** All 293 pairs, windowed: crossed 0.867 [0.826, 0.905]; against the clean twin 0.899
[0.866, 0.933]; by severity 0.887 critical / 0.908 supporting / 0.923 peripheral. Not registered for a bar.

## Amendment 9 (2026-09-22, before H-J9b ran) — H-J9b: the size control for the backbone transfer

**Why.** Amendment 6 deferred the DeBERTa-xlarge-MNLI control. H-J9a pits the deployed DeBERTa-large-MNLI
encoder (0.4B, MNLI objective) against FactCG's DeBERTa-v3-large (0.4B, synthetic factual-consistency graph
objective on top of MNLI-style data). A third encoder of the *same* objective family and about twice the
parameters, `microsoft/deberta-xlarge-mnli` (0.9B, 48 layers), separates what the H-J9a contrast cannot:
whether a change at the FactCG encoder, in either direction, is the pretraining objective or the size.

**Design.** Identical to amendment 6 with `HJ9_BACKBONE=xlarge` and a separate results root
(`hj9_xlarge/`; `HJ9_ROOT`, a one-line addition to the three job files that leaves the FactCG runs untouched):
the deployed recipe unchanged, human and judge-twin packages, three seeds, both pools, three candidate modes,
the frozen rule and evaluator, H-COND with the cached comparator families. Walltime raised to 12 h for the
scoring cost of the larger encoder; nothing else differs.

**No bar.** This block is a control on H-J9a's reading, not a test of the method. The contrasts of interest,
reported per pool on recall@10, omission precision and H-COND with document-clustered intervals, are (i)
xlarge-human minus large-human (the deployed verifier): the size effect at the same objective; (ii)
FactCG-human minus xlarge-human: the objective effect at matched size class, once (i) is known; (iii)
human minus judge twin at the xlarge encoder: the package effect at a third backbone. Predictions: (i) is
small and non-negative on the held-out split and near zero externally; if H-J9a finds the FactCG-encoder
arm *below* the deployed verifier (as H-J10a's clinical rows suggest, 0.783 against 0.937), (ii) will be
negative and the loss is attributable to the objective, not the size. Whatever the numbers, the pipeline of
record is not replaced after the fact.

**Artifacts.** `hj9_xlarge/{models,scores_*,val_*,eval_*,hcond_*}/`; the same three job files with
`HJ9_BACKBONE=xlarge HJ9_ROOT=$MCS/hj9_xlarge`. The RoSE held-out split is re-read (post-hoc secondary;
`MC_TEST_REREAD=yes`), UniSumEval likewise.

**Run note (2026-09-22 23:10, before any xlarge number was read).** The first submission (jobs 6245/6246)
trained all 600 steps and then ran out of GPU memory in the post-training validation scoring, which
`mc_train.py` batches as chunks of 8 pairs with every fact of each pair in one forward pass. A
`--eval-pairs` option (default 8, so the published runs are unchanged) was added and the chain resubmitted
with `HJ9_EVAL_PAIRS=2`; this changes batching only, not any score.

**Run note 2 (2026-09-22 23:45, before any xlarge number was read).** The resubmitted judge-twin job (6250)
trained and scored its first seed and then ran out of GPU memory in the *training* of the second, at a step
whose natural-cell batch was large (the judge package labels about five times as many cells per document
as the human one, so its per-step sequence count varies more). A `--grad-ckpt` option (gradient
checkpointing: activations recomputed in the backward pass, mathematically the same gradient, default off)
was added to `mc_train.py` and the xlarge judge twin resubmitted with it; the xlarge human arm (job 6249)
trains without it. Recorded because the two xlarge arms therefore differ in this one execution detail.

## Amendment 10 (2026-09-22, before H-J10b ran) — H-J10b: OmissionBench end to end, the emitted-top-k protocol

**Why.** H-J10a scored the verifier given the removed fact. The benchmark's own protocol gives a judge the
transcript and one note and asks whether something is missing; our pipeline does the same when it runs end to
end (inventory from the transcript, verifier against the note, ten emissions). This block runs the frozen
recipe unchanged on the clinical pool and reports the paper's two emitted endpoints with their controls, the
package contrast, and the three comparators in the identical inventory and rule.

**Census, written before scoring** (`hj10b_build_omb.py build`, sha256 `6b9622d6…d44146`): 405 notes over 112
consultations — the 293 evaluation omission pairs' errored notes and each consultation's clean twin; reference
facts per consultation median 37.5 (17–58): the benchmark's `must_contain` sheet plus every statement removed in
one of the consultation's pairs (141 removed statements are in the sheet verbatim, 38 have a near-duplicate
there at content-token F1 ≥ 0.6 and are labelled with it, 114 are added); one omitted fact per errored note
(median; 43 near-duplicate labels in total), none in a clean twin; 9,835 sentence-decomposition tasks;
transcripts median 1,390 words, notes median 444. Labels: 1 = the note states the fact, 0 = the note is the
one it was removed from; clean twins carry all 1s and therefore contribute only false alerts.

**Design.** `mc_unisum.py`'s stages re-implemented for this pool (`hj10b_build_omb.py`), then the frozen
recipe: Gemma-4-31B sentence decomposition (`m32_decompose.py`, batch 12, one shard) + spaCy entity units;
the deployed human-fact verifier (three seeds) in real, shuffled (seeded derangement within stratum × system)
and no-summary modes; the judge-label twin (b21 xenc, three seeds, real); the `div1` rule; `mc_e2_eval.py`
with explicit three-seed manifests, k = 10, 2,000 bootstrap draws, permutation and sign tests; observed
document-clustered contrasts (`b3_contrast.py`, document = consultation). Then the three comparators
(Gemma-4-31B judge, FactCG, Bespoke-MiniCheck-7B) over the identical inventory in the three modes, exactly as
H-J3b, in a second evaluation. Candidate notes are read as deployed (380-word clip) because the emitted
protocol is the pipeline of record; the windowed reading is not part of this block.

**Bars.** (i) *Package effect, third pool*: PASS if the human-fact pipeline leads the judge-label twin on
recall@10 with a consultation-clustered interval excluding zero and omission precision not lower (interval
including zero or favouring the human-fact pipeline). (ii) *Precision penalty for ignoring the note*: PASS if
the real-note pipeline's ACU-aligned omission precision exceeds both summary-blind controls with intervals
excluding zero. FAIL otherwise, per bar. The comparator block has no bar.

**Predictions, written before the run.** Absolute recall@10 of the removed fact is moderate (0.3–0.5:
~350 candidate units per transcript compete for ten slots); bar (i) passes with a smaller margin than on
RoSE; bar (ii) passes; on recall@10 the summary-blind controls do not beat the real-note arm here (as on
UniSumEval) because every errored note omits exactly one reference fact; the comparators lead as on UniSumEval,
FactCG and the judge most, given H-J10a. The clean-twin rows are reported as false-alert rates at ten
emissions, the benchmark's second axis, for every system.

**Artifacts.** `hj10b/{acu_units_omb.jsonl,decompose_tasks_omb.jsonl,props/,units_gemma+ent_omb.jsonl,
scores/,val/,eval/,eval_cmp/}`; jobs `slurm/hj10b_pipeline.sbatch`, `slurm/hj10b_score_comparators.sbatch`
(one per system), `slurm/hj10b_eval_comparators.sbatch`. No RoSE split is read.

## Amendment 11 (2026-09-23, H-J9a outcome, recorded after the run)

**Verdict: FAIL** (jobs 6240–6243; `hj9/eval_{test,unisum}/`, `hj9/hcond_{test,unisum}/`, SHA256SUMS in each).
The human-fact package trained with the deployed recipe on FactCG's DeBERTa-v3-large reads recall@10 0.353 /
omission precision 0.704 on the RoSE held-out split and 0.299 / 0.499 on UniSumEval. Against zero-shot FactCG
(0.377 / 0.711; 0.384 / 0.570) the observed document-clustered contrasts are −0.024 [−0.052, +0.003] recall
and −0.007 [−0.029, +0.015] precision on the split, −0.085 [−0.105, −0.067] and −0.070 [−0.090, −0.051]
externally: the bar (lead on recall@10 on both pools, precision not lower) fails on both pools, and on the
external pool the trained encoder is behind. Against the pipeline of record (0.377 / 0.720; 0.347 / 0.544)
it is behind on both pools as well (−0.024 [−0.042, −0.004] and −0.048 [−0.062, −0.034] recall; −0.016
[−0.034, +0.000] and −0.045 [−0.062, −0.030] precision). Controls behave as on the deployed verifier: the
shuffled and empty arms win recall on both pools and lose precision on the split (+0.031, +0.023) and, for the
shuffled arm, externally (+0.022).

**Secondary.** (i) Label source at the new encoder: human minus judge twin +0.003 [−0.019, +0.024] recall on
the split and −0.043 [−0.056, −0.031] externally (precision −0.022 [−0.040, −0.006] and −0.044 [−0.059,
−0.031]): level on the split, judge-twin ahead on the external pool — the package ordering established at
the deployed encoder does not hold at this one. (ii) Both FactCG-encoder arms are below zero-shot FactCG on
both pools on recall@10 (judge twin 0.350 and 0.342). (iii) H-COND: human arm 0.921 [0.908, 0.932] on the
split (deployed 0.914; margin deployed minus new −0.007, one-sided LB −0.016) and 0.848 [0.835, 0.860]
externally (deployed 0.864; +0.016, LB +0.012); judge twin 0.895 [0.879, 0.911] and 0.865 [0.854, 0.876]
(+0.019, LB +0.007; −0.001, LB −0.006). Margins recomputed from the caches with `--primary humanfact`
(`hj9/hcond_{test,unisum}_primary_humanfact/`), so they follow the conditioning table's convention.

**Predictions scored.** "The trained FactCG encoder leads zero-shot FactCG on both pools": wrong on both.
"The judge-twin gap at the new backbone is small on RoSE and null externally": right on RoSE, wrong
externally (the twin leads). "The precision contrast against zero-shot FactCG is the uncertain one": moot.

**Reading, for the paper.** The deployed recipe — 600 steps at 2e-5 on 16,332 natural cells — does not
preserve what FactCG's encoder was pretrained to do; the verifier of record's standing comes from the
package and its MNLI initialisation together, and the supervision (package) claim is made for the encoder
deployed, not for encoders in general. No schedule was tuned on the held-out split to rescue this. The
manuscript reports the block as a failed bar (Table `tab:marsc_backbone`, a results paragraph, two rows in
the conditioning table, the appendix record, the abstract, the contributions and the limits). The size control
(amendment 9) is pending and will be recorded as its own amendment.

## Amendment 12 (2026-09-23, H-J10b outcome for the registered bars, recorded after the run; comparator block pending)

**Bar (i), package effect on the third pool: FAIL.** Job 6253 (`hj10b/eval/`, SHA256SUMS in `hj10b/`). End
to end on the 293 errored notes the human-fact pipeline recalls the removed fact within ten emissions at
0.205 and the judge-label twin at 0.184: +0.020 [−0.031, +0.074] recall@10 (consultation-clustered), interval
including zero; omission precision 0.053 against 0.058 (−0.005 [−0.021, +0.010], not lower). The lead is in
the predicted direction and not detected.

**Bar (ii), precision penalty for ignoring the note: PASS.** Real note 0.053 against 0.031 with another
consultation's note (+0.022 [+0.010, +0.036]) and 0.029 with no note (+0.025 [+0.012, +0.038]); both intervals
exclude zero. On recall@10 the real-note arm is ahead of both controls (+0.034 [−0.024, +0.092]; +0.055
[+0.000, +0.111]), so the recall-by-ignoring-the-candidate effect of the held-out split does not appear here,
as predicted.

**Absolute level and what the precision endpoint means here.** 38 reference facts per note (median) and one
omitted fact per errored note put the base rate of a hit among aligned emissions at 0.026; the pipeline's
0.053 is twice that and the summary-blind arms sit at the base rate. On the clean twins the pipeline's ten
emissions include 2.82 that align to a fact the note states (judge twin 2.27; shuffled 4.18; no note 3.94),
so ignoring the note roughly halves the useful alerts and adds false ones. The predicted recall band
(0.3–0.5) was wrong: 0.205 overall, 0.217 on complete removals, 0.186 partial-weak, 0.202 partial-strong;
by severity 0.195 critical / 0.199 supporting / 0.256 peripheral; by stratum aci 0.202, authored 0.192,
primock 0.207, trapblind 0.238. Of the ten emissions 0.71 align to no reference fact at all (the sheet is a
`must_contain` list, not a census of the transcript), which is why the inventory-side attribution of
Figure 4 is the right lens here: the transcript yields 137.6 units per note and the removed statement has
to be among the ten kept.

**Reading, for the paper.** The precision penalty for ignoring the candidate now holds in every cell of
three pools and two genres; the package effect over the judge-label twin is established on the two
summarisation pools and not detected on the clinical one; and the frozen pipeline surfaces a fifth of the
single removed statements among ten alerts, with about three of the ten aligning to stated facts on a note
with nothing missing. The three comparators (jobs 6254–6257) will be added in a second evaluation over the
identical inventory, with no bar, and recorded as amendment 13.

## Amendment 13 (2026-09-23, H-J10b comparator block, recorded after the run; no bar)

Jobs 6254–6257 (`hj10b/eval_cmp/`, SHA256SUMS). The three comparators over the identical OmissionBench
inventory and rule, three candidate modes each, evaluated beside the MARS-C arms in one evaluator call
(pipeline of record as primary; contrasts MARS-C minus row, consultation-clustered): Bespoke-MiniCheck-7B
0.268 recall@10 / 0.080 omission precision (−0.063 [−0.117, −0.007]; −0.026 [−0.043, −0.011]), the
Gemma-4-31B judge 0.241 / 0.079 (−0.036 [−0.098, +0.029]; −0.026 [−0.047, −0.006]), FactCG 0.188 / 0.044
(+0.017 [−0.033, +0.069]; +0.010 [−0.002, +0.022]). MiniCheck-7B leads on both endpoints, the judge on
precision with recall not separated, FactCG is level. Every comparator's summary-blind arms fall below its
real-note arm on both endpoints (recall 0.09–0.19 against 0.19–0.27; precision 0.023–0.032 against
0.044–0.080), and MARS-C's contrasts against each of those arms exclude zero on precision (+0.021 to
+0.031). Clean-twin alerts (of ten) aligning to a fact the note states: MARS-C 2.82 [2.51, 3.14], judge twin
2.27, Gemma judge 2.17 [1.81, 2.53], MiniCheck-7B 2.36 [2.01, 2.69], FactCG 2.97 [2.59, 3.38]; the
summary-blind arms 3.3–4.5. Prediction "the comparators lead as on UniSumEval, FactCG and the judge most":
half right (MiniCheck-7B and the judge lead; FactCG does not). Table `tab:marsc_omb_e2e`.

## Amendment 14 (2026-09-23, H-J9b outcome, recorded after the run; the block carried no bar)

Jobs 6249, 6258 (gradient checkpointing, judge twin only), 6259, 6260 (`hj9_xlarge/`). DeBERTa-xlarge-MNLI
(0.9B, 48 layers) with the deployed recipe, human package: recall@10 / omission precision 0.386 / 0.731 on
the held-out split and 0.351 / 0.546 on UniSumEval. Contrasts (xlarge human arm minus row, document-
clustered): pipeline of record +0.010 [−0.006, +0.025] and +0.011 [−0.001, +0.022] on the split, +0.003
[−0.007, +0.015] and +0.002 [−0.012, +0.015] externally; zero-shot FactCG +0.009 [−0.016, +0.034] and
+0.020 [+0.001, +0.039] on the split, −0.033 [−0.051, −0.017] and −0.023 [−0.041, −0.005] externally;
MiniCheck-7B and the judge externally −0.034 [−0.053, −0.015] / −0.035 and −0.020 [−0.038, −0.002] /
−0.039. Judge twin at xlarge 0.353 / 0.721 and 0.347 / 0.549: human minus twin +0.033 [+0.015, +0.051] and
+0.010 [−0.003, +0.024] on the split, +0.004 [−0.009, +0.017] and −0.002 [−0.015, +0.010] externally.
Controls: shuffled and empty arms win recall on the split (0.461, 0.484) and lose precision (+0.055
[+0.034, +0.077], +0.057 [+0.033, +0.082]); externally they lose both (recall −0.019 [−0.036, −0.003] and
−0.019 [−0.043, +0.006]; precision +0.059 [+0.042, +0.078], +0.054 [+0.033, +0.077]). H-COND: xlarge human
0.911 [0.895, 0.925] and 0.865 [0.853, 0.876], xlarge judge 0.903 [0.888, 0.917] and 0.868 [0.856, 0.879];
pipeline of record minus xlarge human +0.003 (one-sided LB −0.004) and −0.001 (−0.004), minus xlarge judge
+0.011 (+0.001) and −0.004 (−0.008) (`hj9_xlarge/hcond_{test,unisum}_primary_humanfact/`).

**The three contrasts of amendment 9.** (i) Size at the same objective: +0.010 and +0.003 recall, +0.011
and +0.002 precision, crossed accuracy within 0.003 — nothing measurable. (ii) Objective at matched size
class: the FactCG-encoder human arm sits 0.033 (split) and 0.052 (external) below the xlarge one on
recall@10, so the H-J9a loss is the initialisation, not the parameter count. (iii) Package at a third
encoder: the deployed encoder's pattern (lead on the split, null externally), unlike FactCG's (level,
reversed). Predictions (i)–(iii) right. The external deficit against the 2026 comparators is unchanged at
twice the size (−0.033 against zero-shot FactCG, the deployed verifier's −0.037), so it is neither the
clip (H-J3b diagnostic) nor the size; what the comparators bring is their pretraining, which the recipe does
not keep when started from it (amendment 11).

**Joint re-evaluation (job 6380, `hj9_joint/`).** Every arm of amendments 6 and 9 with the pipeline of
record, the matched judge twin and the three comparators in one evaluator call per pool, primary = pipeline
of record, so that Table `tab:marsc_backbone` quotes "pipeline of record minus row" from a single
bootstrap. No new scoring; the same div1 files.

## Amendment H-J11 (2026-09-27, registered before any of it ran): filling the empty table cells

Measurement block, no bar, every cell labelled secondary where the split was read before. The purpose is to fill
table cells that were empty only because a block had been specified for a narrower scope. Nothing is retrained,
no setting is tuned, and every run reuses the frozen inventories, checkpoints, rule and evaluator.

- **UniSumEval rows of Table 1** (jobs 7657, 7658, 7659): the frozen LongT5 segmenter over the frozen UniSumEval
  sentence tasks (props-only inventory, new file), human-fact and judge-label verifiers scored as G1 scored the RoSE
  distilled inventory; MARS-2 `direct_enabled` dumps for the same three seeds as the TEST rows; one evaluator call with
  primary `humanfact+div1` on the frozen Gemma+ent inventory, which must reproduce 0.347.
- **Validation controls of Table 1** (B0 `validation × gemma+ent`, jobs 7650–7653): the standing control harness on
  the one cell it had not run; `humanfact.gemma+ent@real` must reproduce 0.400.
- **Validation crossed accuracy for the H-J9/H-J9b arms** (job 7654): `b2_crossed_families.py` on RoSE validation
  with the b12 validation cache copied, `--primary humanfact`.
- **Distilled off-the-shelf timings** (job 7656): `b6_cost.py` with a new `--offshelf-only
  --offshelf-inventories gemma+ent,distilled` option; Gemma+ent re-timed in the same run as a consistency check.
- **Deployment weights of RoBERTa-MNLI and AlignScore** (job 7655): the H-J5 probe fixed to the models' real
  attribute names (`.m`, `.enc` plus the AlignScore head tensors) and the weight dtype recorded (these two load in
  fp16 on CUDA, not fp32 as the table said).
- The distilled summary-blind controls of Table 5 and the distilled FactCG / Bespoke-MiniCheck-7B timings already
  exist (R12.1 `r12_precision/eval_test`; b10 receipts in `b11/cost_table.json`) and are only transcribed.

Not fillable and left as is: human precision for rows outside study D1 (a new annotation round), G-Eval on SummEval
and RoSE (the benchmark does not ship those GPT-4 scores), and cells that are undefined by construction.
To run ahead of the queue, the user's other pending jobs were held (`scontrol top` is not permitted); the list is
`$MCS/hj11/held_jobs.txt` and job 7663 releases them once no H-J11 GPU job is pending.

**Outcome, H-J11 (2026-09-28).** All jobs completed. UniSumEval recall@10 (primary `humanfact+div1` reproduces
0.347): human-fact distilled consensus 0.425, judge-label distilled consensus 0.392 and seed-avg 0.401, MARS-2
0.248 released ranking and 0.268 with the rule. B0 validation x Gemma+ent: real 0.401 (published 0.400),
shuffled 0.430, no summary 0.484. Validation crossed margins (MARS-C minus arm, LB95): FactCG-encoder human +0.011
(-0.002), its judge twin +0.024 (+0.009), xlarge human +0.014 (+0.005), xlarge judge +0.016 (+0.006); every existing
validation margin reproduced exactly. RoBERTa-MNLI and AlignScore weights 0.66 GiB each (fp16, 355,362,819
parameters each). Off-the-shelf timings, median of three runs of one harness: SummaC-ZS 7.87 ms/unit on Gemma+ent,
not the 3.38 transcribed from the earlier B1 harness; the table now carries the one-harness values for all seven.
UniSumEval controls for Table 1 were taken from the existing b3 matched rebuild (shuffled 0.356, no summary 0.341).
The release helper's fail-safe treated squeue's error for completed job ids as "pending", so the holds were
released by hand once every H-J11 job had finished.

## Amendment H-J12 (2026-09-28, registered before any of it ran): package versus label-only twin, audit response

Response to the 2026-09-27 experiments audit (serious items 2, 3, 5) and the 2026-09-28 manuscript review (F04, F05,
F07, F22). Measurement block, no new bar; every TEST and UniSumEval reading here is post-hoc secondary, because
both were read before. Nothing already trained is retrained and no setting is tuned.

The audit showed that every crossed-accuracy and residual number the paper attributes to "the label-only twin"
was produced by the judge-label **package** (B21, `$G/b21/xenc_seed*`: spaCy entity + dependency-proposition units,
85,600 labels), while the true twin (R04, `mc-A-lam1-crossenc-r04judge`, the identical 63,343 human-ACU cells
relabelled by the Gemma-4-31B judge) was never scored on the crossed endpoint. This block scores it.

- **(a) Twin on the crossed endpoint** (`slurm/hj12_twin_crossed.sbatch`). `b2_crossed_families.py`, unmodified,
  on RoSE validation, RoSE TEST and UniSumEval, families: deployed MARS-C, B21 package, R04 human twin
  (`r04human`, the same recipe retrained on the human labels of the identical cells), R04 judge twin, and every
  existing comparator, served from the frozen caches of record (copied, never written). Two passes: `--primary
  humanfact` (the paper's margins) and `--primary twinhuman` (the exact label-only contrast).
- **(b) Twin through the lexical residual.** `c2_crossed_residual.py`, unmodified, same four families, surface
  model fitted on RoSE validation per seed and applied unchanged to TEST and to UniSumEval; both primaries.
- **(c) Unit-matched judge twin** (`slurm/hj12_gemma_twin.sbatch`, `code/hj12_gemma_twin.py`). The E0 train/calib
  documents, summaries and roles unchanged; per document as many Gemma+ent facts (the frozen m32 decomposition of the
  same source, `train_rows_facts.jsonl`) as the document has human ACUs, drawn uniformly with seed 20260928; every
  (summary, fact) cell labelled by the same Gemma-4-31B judge and prompt (`b1_judge_labels.py`); arm A, λ=1, three
  seeds, the R04c recipe. It is scored on the RoSE TEST Gemma+ent and distilled inventories with the frozen
  emission rule and evaluator in one call with the deployed systems and both R04 twins, and on the crossed endpoint.
  It separates the unit-type mismatch from the label source: package (spaCy units, judge) vs twin (human-ACU units,
  judge) vs unit-matched twin (Gemma+ent units, judge), all against MARS-C (human-ACU units, human labels).
- **Reading, fixed now.** The paper attributes to the label source only what the R04 contrast shows; if the
  unit-matched twin closes most of the +0.122 package gap, the headline is restated as a package/unit effect.

To run ahead of the queue, the user's other pending jobs are held (`scontrol top` is not permitted); the list is
`$MCS/hj12/held_jobs.txt` and is released by hand when no H-J12 job is pending.
- **(d) Inventory × verifier grid, empty cells (F06, F07)** (`slurm/hj12_grid.sbatch`, `hj3_score_unisum.sbatch` with
  `HJ3_INV=distilled`). The R04 human and judge twins on the distilled inventory of RoSE TEST and of UniSumEval; the
  Gemma-4-31B judge, FactCG and Bespoke-MiniCheck-7B on distilled UniSumEval (real candidate only; the controls of
  those three are not needed for the grid). One evaluator call per pool with every verifier on the distilled
  inventory, primary the deployed human-fact verifier; the RoSE TEST distilled comparator cells that already exist
  (`b9b10_eval/test`) are read, not recomputed. The Gemma+ent column already exists on both pools.
- **(e) Clinical chance baselines (F15)** (`slurm/hj12_clinical.sbatch`, `code/hj12_clinical_baselines.py`). Uniform
  random scores (three seed-locked draws) and the lead-position prior over the identical clinical Gemma+ent inventory,
  through the same div1 rule (cap 1), alignment rule and evaluator as the verifier. The paper's "one in 38" comparison
  is replaced by these.

**Outcome, H-J12 (a) and (b) (2026-09-28, job 7879).** Every existing family reproduced from the cache of record
(MARS-C validation 0.917, TEST 0.914, UniSumEval 0.864; package 0.887 / 0.893 / 0.869). The R04 human twin is
identical to the deployed verifier (margin 0.0000 on all three pools), so the twin contrast is also MARS-C minus the
R04 judge twin. Crossed accuracy of the label-only judge twin (document macro, seed-averaged scores): validation
0.905, TEST 0.908, UniSumEval 0.870. MARS-C minus twin: validation +0.012 [−0.002, +0.027] (one-sided LB +0.001),
TEST +0.006 [−0.002, +0.014] (LB −0.001), UniSumEval −0.006 [−0.011, −0.001]. MARS-C minus package: +0.030,
+0.021, −0.005 (as published). Lexical residual, seed-averaged scores throughout (surface model fitted on RoSE
validation per seed): TEST MARS-C 0.914 → 0.891, package 0.894 → 0.688, twin 0.908 → 0.739; residual margin over the
twin +0.153 (LB +0.128), over the package +0.203 (LB +0.176). UniSumEval: MARS-C 0.864 → 0.726, package
0.869 → 0.796, twin 0.870 → 0.819; residual margin over the twin −0.093 (LB −0.109). Reading: on the crossed
endpoint the label source alone is not separable on RoSE and reverses on UniSumEval; most of the published
crossed-accuracy gap to the package is not the label. The residual advantage over the true twin holds on RoSE TEST
(+0.153) and reverses on UniSumEval, as it did for the package.

**Outcome, H-J12 (e) (2026-09-28, job 7893).** Clinical pool, 405 notes, identical inventory, div1 rule (cap 1),
alignment rule and consultation-clustered bootstrap. recall@10 / omission precision: MARS-C 0.205 / 0.053 (reproduces
the published row); uniform random 0.097 / 0.025; lead position 0.061 / 0.015; shuffled note 0.171 / 0.031; no note
0.150 / 0.029. MARS-C minus random: recall +0.107 [+0.048, +0.167], precision +0.028 [+0.013, +0.045]; minus lead
position: +0.143 [+0.088, +0.202], +0.039 [+0.024, +0.055]. The clinical detector is above the chance and position
baselines on both endpoints; against its own summary-blind controls only precision separates (as published).
Run note, H-J12 (d): the MiniCheck-7B pass on distilled UniSumEval (job 7896) stopped at its guard because 3 prompts
exceed the 8,192-token window; it is rerun unchanged with a 16,384-token window (job 7900), as the H-J3a crossed run
already used a 32,768-token window for this model.

**Outcome, H-J12 (d) (2026-09-28, jobs 7894, 7895, 7897, 7900, 7898).** Distilled inventory, recall@10 / omission
precision. RoSE TEST (2,688 pairs): MARS-C 0.369 / 0.752, R04 human twin identical, R04 judge twin 0.375 / 0.758,
package 0.327 / 0.755, FactCG 0.384 / 0.738, MiniCheck-7B 0.353 / 0.738, 31B judge 0.346 / 0.745. UniSumEval (1,822
pairs): MARS-C 0.426 / 0.585, judge twin 0.423 / 0.592, package 0.393 / 0.583, FactCG 0.441 / 0.586, MiniCheck-7B
0.440 / 0.599, 31B judge 0.428 / 0.606. MARS-C minus twin (recall): −0.006 [−0.024, +0.011] and +0.003 [−0.008, +0.014];
MARS-C minus FactCG: −0.015 [−0.039, +0.007] and −0.015 [−0.027, −0.002]. The package gap holds in every cell; the
label-only gap is null on the distilled inventory on both pools; the strongest zero-shot verifiers are level with or
above MARS-C. Added to the paper as Table `tab:marsc_grid`.

**Outcome, H-J12 (c) (2026-09-28, jobs 7880-7882).** 63,303 judge tasks (R04: 63,343), judge covered rate 0.331 (the
annotators' rate on ACU cells is 0.372, the judge's on ACU cells 0.649). RoSE TEST, recall@10 / omission precision,
one evaluator call: MARS-C 0.377 / 0.720; R04 human twin identical; R04 judge twin (ACU units) 0.353 / 0.736; unit-matched
judge twin (Gemma+ent units) 0.333 / 0.714; package (spaCy units) 0.255 / 0.692. MARS-C minus unit-matched twin: recall
+0.044 [+0.024, +0.063], precision +0.007 [−0.014, +0.026]; distilled inventory 0.350 / 0.748, recall +0.020 [+0.004,
+0.036]. Crossed accuracy of the unit-matched twin: validation 0.909, TEST 0.906 [0.891, 0.920], UniSumEval 0.875
[0.865, 0.884]; MARS-C minus it +0.008 (LB −0.003), +0.008 (LB +0.000), −0.010 [−0.016, −0.005]. Reading, per the rule
fixed above: training a judge-label verifier on the deployed fact type closes about two thirds of the +0.122 package
gap on RoSE TEST (to +0.044); the spaCy unit type is the largest single component of the package effect. The judge
does better on human ACU units (0.353) than on Gemma+ent units (0.333), so the human ACU inventory contributes beyond
the labels. On the crossed endpoint no judge-labelled variant separates from MARS-C on RoSE, and all are above it on
UniSumEval.
Queue note, H-J12 (2026-09-28): at the user's instruction the mage-campaign gpu-broker was stopped while the block ran,
the user's other pending jobs were held and two jobs under three hours (7441, 7451; later 7441 and 7498 after the
broker had restarted them) were requeue-held. When the last GPU job finished, the 29 holds this block placed were
released and the broker was restarted with its original command (`hj12/broker_cmd.txt`).

## Amendment H-J14 (2026-09-29, registered before the replay ran): which training cells the R04 twins drew

**Question.** The paper calls the R04 judge twin "label-only". The full review of 2026-09-29 (R1-W1) points out that
the twin's crossed tetrads are recomputed from its own labels, so the two label sets need not train on the same
cells. B3 (registered 2026-09-18, outcome recorded in `marsc_strengthen_20260918/RESULTS.md`) already scored both
arms of the twins on RoSE TEST and UniSumEval: arm A (the deployed objective, two crossed tetrads and four natural
draws per step) and arm R (natural draws only). This block establishes, for each arm, whether the human-label and
the judge-label run drew the same training cells.

**What runs.** CPU only, no model, no evaluation data. `code/hj14_sampler_replay.py` imports the `Sampler` of
`marsc_20260916/code/mc_train.py` unchanged, loads the training role of the human data (`marsc/e0`) and of the judge
twin (`marsc/r04_matched/e0_judge`), and replays the 600 steps of each arm for the three seeds of record
(20260906, 20260907, 20260908) with the training defaults (two tetrads, four natural draws, six facts per natural
draw; arm R draws eight natural items of four facts). For every (arm, seed) it writes the number of encodings and
drawn cells under each label set, whether the two sequences of (document, summary, fact) cells are identical
position by position, the share of drawn cells the two runs have in common, the share of common cells whose target
differs, and the tetrad pool of each label set. The training receipts (`result.json` of the twelve checkpoints:
encodings, tokens, cells after truncation) are copied beside it as the record of the runs themselves.

**No bar.** The block is descriptive. Its reading is fixed here: if arm R draws identical cells under both label
sets, the B3 contrast of arm R (RoSE TEST +0.051 [+0.032, +0.072] recall@10, UniSumEval +0.008 [-0.006, +0.022]) is
the fixed-cell label-source contrast and is reported as such; the arm A contrast (+0.023 [+0.006, +0.042], -0.002
[-0.015, +0.011]) is reported as the label-source contrast of the deployed procedure, whose crossed samples follow
the labels. If arm R does not draw identical cells, no fixed-cell contrast is claimed and the text says only what
is matched.

**Output.** `~/results/marsc_strengthen/hj14/sampler_replay.json`, `training_receipts.json`, SHA256SUMS.

**Outcome, H-J14 (2026-09-29, job 8033, CPU, 2 s).** The replay reproduces the training receipts exactly (drawn
cells after truncation: 16,332 / 16,297 / 16,296 for the human arm A runs, 16,287 / 16,391 / 16,335 for the judge
arm A runs, 17,689 / 17,683 / 17,758 for both arm R runs), so it describes the runs themselves. **Arm R draws the
identical sequence of cells under both label sets in all three seeds**; 0.325, 0.323 and 0.332 of the targets
differ between the two label sets on those cells. The training receipts agree: encodings, tokens and cells of
every arm R pair are equal to the unit (986,474 / 981,134 / 980,319 tokens). **Arm A does not**: the two runs share
0.336, 0.339 and 0.328 of their drawn cells (63, 81 and 70 cells at the same position), because the tetrad pool
follows the labels (401 tetrad documents and 3,434 summary-pair groups under the human labels, 346 and 2,260 under
the judge's; the judge twin also carries no collision list). Reading, per the rule fixed above: the B3 arm R
contrast is the fixed-cell label-source contrast (RoSE TEST recall@10 0.4025 against 0.3511, +0.051 [+0.032, +0.072];
omission precision +0.003 [-0.011, +0.018]; UniSumEval 0.3588 against 0.3509, +0.008 [-0.006, +0.022]); the arm A
contrast (+0.023 [+0.006, +0.042]; -0.002 [-0.015, +0.011]) is the label-source contrast of the deployed procedure.
The manuscript no longer says that the twin "differs in nothing but the labels" or that the labels were changed
"on identical cells"; it reports both arms.

Queue note, H-J14 (2026-09-29): at the user's instruction the running `audit-` array tasks (7948_39, 7948_40,
7949_23, 7949_24) were requeued and held and the pending `audit-` jobs (7948, 7949, 7954) were held at 16:00 MSK
while job 8033 ran; all holds were released about a minute later, the arrays kept their task limit of 2 and the
next tasks started at once. The `audit-` jobs are CPU-only; no GPU job was touched and the gpu-broker was left
running.
