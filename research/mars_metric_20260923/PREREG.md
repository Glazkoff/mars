# Pre-registration: MARS as a metric and a detector (program of 2026-09-23)

Plan: `MARS_METRIC_PLAN_2026-09-23.md`. House rules (`ORACLE.md` §3–4, the H-J register): bars before data;
candidate-blind and shuffled-candidate controls in every scored arm; document-clustered percentile intervals,
2,000 draws; the observed contrast is quoted; a registered verdict is never restated over a family it was not
registered on; a failed bar is closed, not iterated. Results root on Euler: `/home/user/results/mars_metric/`.

## M0 — census (no bar; written before any scorer runs)

One schema for every human-judgment set and benchmark: `pair_id, dataset, doc_id, source, candidate,
reference (optional), human: {dimension: value}, splits`. Sets: UniSumEval (faithfulness per sentence →
`faithfulness_score`; key-fact labels → `completeness`; `conciseness`; also the shipped G-Eval / G-Eval+ machine
scores as comparators), SummEval (expert means for relevance, consistency, coherence, fluency; 16 machine
summaries and 11 references per document), RoSE test (human normalized ACU recall per pair, from the existing
`acu_units_test.jsonl`), HaluEval-summarization (pairs), SummEdits (full release), FRANK (annotations),
LLM-AggreFact dev and test (as shipped), RAGTruth-processed (span fields verified and counted). Output:
`m0/census.json` with counts, licence strings, and SHA-256 of every input file. Nothing is scored.

## M-A — metric validity (H4.1, H2.1)

**Targets.** UniSumEval: `completeness`, `faithfulness_score`, `conciseness`. SummEval: `relevance`,
`consistency`, `coherence`, `fluency` (expert means). RoSE test: human ACU recall.

**Metrics under test.** MARS-P (candidate facts by the frozen decomposition; FactCG as verifier, document as
premise with its own 2,048-token contract; mean over facts). MARS-R reference-based (reference facts by the frozen
decomposition; deployed MARS-C verifier, three seeds averaged, candidate as premise in 380-word windows with the
maximum over windows; mean over facts). MARS-R reference-free (the MARS-C source inventory; E2 importance-tagger
salience as weights; same verifier; weighted mean). Each with two controls: the candidate replaced by another
document's candidate of the same system (seeded derangement within dataset and system) and no candidate.

**Comparators.** ROUGE-1, ROUGE-2, ROUGE-L (F, vs reference), BERTScore-F (roberta-large, layer default of the
package, vs reference), BARTScore (source→candidate and reference→candidate), AlignScore (source→candidate),
SummaC-ZS, FactCG-whole (source→candidate), Bespoke-MiniCheck-7B (source→candidate), and the G-Eval / G-Eval+
columns UniSumEval ships.

**Endpoints.** Summary-level Kendall τ-b (within document across systems, averaged over documents with at
least three systems), pooled Spearman ρ, system-level Pearson r; document-clustered bootstrap for every
statistic and for every paired difference between a MARS score and a comparator.

**Bars.** (i) PASS if MARS-R (reference-based) exceeds both ROUGE-L and BERTScore-F on summary-level τ with
`completeness` on UniSumEval and with `relevance` on SummEval, each paired difference's interval excluding
zero (four contrasts). (ii) PASS if MARS-P is not below AlignScore and FactCG-whole on summary-level τ with
`faithfulness_score` on UniSumEval and `consistency` on SummEval (interval including zero or favouring MARS-P,
four contrasts). (iii) secondary, no bar: reference-free MARS-R against `completeness` versus ROUGE-L.
Controls: every MARS arm must exceed its two controls on its target dimension (reported; a control that wins
is reported as such, as the omission program did).

**Predictions.** (i) passes on UniSumEval; on SummEval the relevance contrast against BERTScore-F is the
uncertain one. (ii) passes; MARS-P's τ with faithfulness lands within 0.03 of AlignScore's. (iii) the
reference-free score is below the reference-based one but above ROUGE-L on UniSumEval.

**Artifacts.** `ma/decomp/`, `ma/scores/`, `ma/meta/meta_evaluation.json`; jobs `slurm/ma_decompose.sbatch`,
`slurm/ma_score.sbatch`, `slurm/ma_meta.sbatch`. Registered 2026-09-23 before any of it ran.

## M-B — decomposition improves detection (H4.2)

**Benchmark.** LLM-AggreFact, split `test` (29,320 rows, 11 datasets), thresholds tuned per dataset and per arm
on `dev` (30,420 rows) to maximise balanced accuracy; fixed 0.5 reported beside. Macro = unweighted mean over the
11 datasets, the leaderboard's convention.

**Arms.** A1 FactCG whole-claim (`b10_verifier_swap.FactCG`, document truncated to fit 2,048 tokens with the
claim intact). A2 MARS-P(FactCG): the claim decomposed by the frozen recipe (Gemma-4-31B, `m32_decompose.py`
prompt, sentence by sentence); each fact scored against the document with the same contract; claim score =
minimum over facts (mean reported as secondary); a claim decomposed to NONE keeps its whole-claim score.
Secondary arms: A3/A4 the deployed MARS-C verifier in the precision role, whole and decomposed, with the
document read in 380-word windows and the maximum kept; A5/A6 Bespoke-MiniCheck-7B whole and decomposed, if
the budget allows. Controls: the shuffled-document control for A1 and A2 (document replaced by another row's
document within dataset), whose balanced accuracy should sit at 0.5.

**Bar.** PASS if macro balanced accuracy of A2 minus A1 ≥ +1.0 point with a 2,000-draw bootstrap interval
(claims resampled within each dataset, macro recomputed per draw) excluding zero. Secondary, no bar: no
dataset lower by more than 1.0 point; the RAG subsets (RAGTruth, ExpertQA, Lfqa, ClaimVerify) individually.

**Predictions.** Passes; the gain is carried by RAGTruth, ExpertQA and Lfqa; AggreFact-XSum shows no gain;
the deployed coverage verifier in the precision role is below FactCG by more than 5 points macro.

**Artifacts.** `mb/tasks/`, `mb/decomp/props_shard*.jsonl`, `mb/scores/`, `mb/eval/mb_eval.json`; jobs
`slurm/mb_decompose.sbatch` (array of 2), `slurm/mb_score.sbatch`, `slurm/mb_eval.sbatch`. Registered
2026-09-23 before any of it ran. The test split is read once per arm.

## M-C — one verifier for both axes

**Design.** Initialise from FactCG-DeBERTa-v3-Large; training set = MiniCheck C2D + D2C (MIT; `lytang/
C2D-and-D2C-MiniCheck`) mixed 1:1 per batch with the 16,332 natural coverage cells of `e0` (premise = summary,
hypothesis = source fact, label = conveyed); one binary head (support); learning rate 1e-5, 1,500 steps, batch
16, three seeds (20260906/07/08); selection by the mean of RoSE-validation ACU AUROC and LLM-AggreFact-dev macro
BAcc at the end of training only (no early stopping on test). ANLI is not used (non-commercial licence).

**Bars.** PASS if all three hold for the three-seed average: LLM-AggreFact test macro BAcc ≥ FactCG-whole
(A1 of M-B) − 1.0; pooled AUROC on RoSE validation ACUs ≥ the deployed verifier − 0.01; crossed accuracy on
RoSE validation (b2, document macro) ≥ 0.90. FAIL otherwise, reported per bar; the package then ships two
verifiers.

**Predictions.** The two coverage bars pass; the AggreFact bar is uncertain (within ±2 points of FactCG).

**Artifacts.** `mc/models/`, `mc/eval/`; jobs `slurm/mc_train.sbatch`, `slurm/mc_eval.sbatch`. Registered
2026-09-23; to be submitted after M-B's A1 number exists (the bar references it) and before any M-C number is
read.

## Queued, not yet registered

M-D span localization on RAGTruth; M-E end-to-end cost; M-F package release. Opened only after one of M-A, M-B,
M-C is harvested (at most three open).

## Run log (job ids on Euler; appended as submitted, before any result was read)

- 2026-09-23 14:5x: M0 census written (`m0/census.json`: 12 sets, 0 errors). M-B tasks 62,405 sentences from
  59,740 claims; M-A tasks 20,983 sentences, 6,116 pairs (1,828 UniSumEval with 204 lacking a reference, 1,600
  SummEval, 2,688 RoSE).
- Decompositions: 6659 (M-B, two shards), 6660 (M-A, two shards).
- M-B scoring 6673 (FactCG, dev/test), 6674 (deployed MARS-C in the precision role, dev/test); evaluation 6675.
- M-A scoring array 6676 (marsR_marsc, marsP_factcg, marsP_marsc, marsR_factcg, rouge, bertscore, bartscore,
  alignscore, factcg_whole); meta-evaluation 6677 (afterany, so a failed comparator does not block the report).
- M-C training 6687 (three seeds, afterok on 6675 so that A1 exists first), scoring 6688, bars 6689.
- The RoSE reference summaries (from the aggregated RoSE release on the cluster) were added to the pair index
  before the scoring array 6676 started, so the RoSE comparators (ROUGE, BERTScore, BARTScore ref→cand) use
  the real reference; MARS-R on RoSE uses the human ACUs as its facts. Recorded before any M-A result was read.

## Queued designs (written now, NOT opened; each opens only when one of M-A/M-B/M-C is harvested)

### M-D — span-level localization on RAGTruth (H4.2, the RAG use)

**Data.** RAGTruth-processed test: 2,700 responses (900 each of Summary, QA, Data2txt; six generators), 943
with at least one annotated hallucination span (character offsets into the response, with a label type). The
train split is not used. The context is the `context` field (retrieved passages / source), the response the
`output`.

**Design.** MARS-P over the response with the frozen decomposition (Gemma; the distilled segmenter as the cheap
arm) and FactCG as the verifier (and the unified verifier if M-C passed): every fact carries the span of the
sentence it came from; a sentence is flagged when any of its facts has support below the dev-tuned threshold of
M-B's RAGTruth row. Comparators in the identical units: FactCG at sentence level (whole sentence as the claim),
MiniCheck-7B at sentence level, and a candidate-blind control (context replaced by another response's context).
**Endpoints.** Response level: balanced accuracy of "any span" against the annotation. Sentence level: precision
and recall of flagged sentences against sentences overlapping an annotated span, with a response-clustered
bootstrap, per task type. **Bar.** Sentence-level F1 of MARS-P(FactCG, decomposed) exceeds FactCG at sentence
level with a response-bootstrap interval excluding zero on the full test set; secondary, per task type.
**Prediction.** Passes on Summary and QA, uncertain on Data2txt (table-derived responses whose facts the
decomposer may not carve at the annotation's grain).

**Closed without opening (2026-09-24).** M-D's mechanism is the same one M-B already tested directly:
decomposed-min (here, decomposed-flag) scoring against whole/sentence-level scoring, and RAGTruth is one of
M-B's eleven datasets (Amendment M-B.1: decomposed-min macro BAcc on RAGTruth is $-1.28$ points for FactCG,
$+1.00$ for the MARS-C verifier, i.e. mixed and not a reliable gain). Opening a fourth decomposition-for-detection
test after H-J9, H-J10a, M-C and M-B have already converged on the same negative finding would read as searching
for a pass rather than completing the registered protocol. Per ORACLE \S3, closed, not opened.

### M-E — end-to-end cost of the package

**Design.** Wall-clock and peak memory per (source, candidate) for the package's `score` with (a) the distilled
segmenter + FactCG (P) + MARS-C (R), (b) the Gemma inventory, (c) FactCG alone at sentence level, (d)
Bespoke-MiniCheck-7B at sentence level, on 200 UniSumEval pairs at batch 1 and at batch 32, one H200, three
repeats, the H-J5 harness conventions (resident weights, peak allocated memory, median latency). No bar;
the table in the paper. **Prediction.** (a) is under 0.5 s per pair at batch 32 and under 4 GiB peak.

**Harvested (2026-09-24, job 6866, array of 4).** Ran with an unregistered fourth arm, `mars_sentences`
(sentences as facts, no inventory model), in addition to the three registered ones; the registered arm (d),
Bespoke-MiniCheck-7B at sentence level, was never implemented in `me_cost.py` and did not run. No bar to
apply -- as designed, this is a measurement table, `tables/mars_cost_e2e.tex` (`me_table.py`) -- but the
prediction on (a) is wrong: $1.34$\,s per pair at batch $32$ against the predicted $<0.5$\,s, and $7.1$\,GiB
of resident weights against the predicted $<4$\,GiB. (b), the Gemma inventory, is far more expensive: $6.06$\,s
per pair, $64.4$\,GiB. The sentence-level fallback arm (unregistered) is close to the FactCG-alone comparator
in latency ($0.51$ against $0.41$\,s) and reads out the package's floor cost without a learned inventory. Peak
batched-memory figures are activation-dominated at batch $32$ and are not a deployment estimate; Table
marsc\_deploy's batch-$1$ verifier-only figures remain the deployment reference.

### M-F — the package release

**Design.** `mars/` (committed 2026-09-23 with model-free tests): `MarsScorer.score` returns P, R, F, evidence
and the three controls. Equivalence check registered here: on 200 random M-A pairs the package's MARS-P and
MARS-R reproduce the M-A score files to within 1e-3 (same verifiers, same decomposition inputs). Bar: the check
passes; otherwise the package is fixed until it does and the fix is recorded. Weights: the coverage verifier
(three seeds) and, if M-C passed, the unified verifier; the distilled segmenter; a model card naming the training
data and licences (RoSE labels, MiniCheck C2D/D2C MIT, FactCG initialisation).

**Equivalence check: PASS (2026-09-24, job 7000).** 200 random pairs, decomposition held fixed via the cached
`props`, same FactCG snapshot and MARS-C checkpoint glob `ma_score.py` used. `mars.verifiers.PromptVerifier`
against `marsP_factcg`: 200/200 within $10^{-3}$, max absolute difference $2.0\times10^{-10}$.
`mars.verifiers.PairVerifier` against `marsR_marsc`: 184/184 within $10^{-3}$ (184, not 200: some sampled pairs
have no reference facts and are skipped by both the package and the research script alike), max absolute
difference $6.5\times10^{-8}$. Both differences are at floating-point-noise scale, not a near miss: the
released package's two verifier classes are a faithful reimplementation of the FactCG and MARS-C contracts the
paper's own numbers were computed with, not merely numerically close to them. `mf/equiv/mf_equiv.json`.
Remaining M-F items (weights, model card, README) are packaging, not a registered bar.
- AlignScore (comparator of bar (ii)) failed on the compute node: a candidate whose sentence splitter returns one
  piece longer than the 512-token pair budget cannot be truncated on the premise side alone. The scorer now caps
  each claim piece at 200 tokens of its own tokenizer for AlignScore only (a word cap did not suffice: token-dense pieces) (the zoo implementation is untouched) and logs the count of
  capped pieces; rerun as job array 6676 task 7 (resubmitted). Recorded before any M-A result was read.
- 16:45: three GPUs idle behind dependencies. M-C training (6687) released from its afterok on the M-B evaluation and
  started; the amendment stands in substance: no M-C number is read against the A1 bar until A1 exists, and the
  coverage-side numbers the training job prints do not touch that bar. M-B scoring re-plumbed as 2 splits x 4
  row shards per verifier (same rows, same seeded shuffle drawn over the whole split), the evaluation reading the
  shard files; the pending scoring arrays 6673/6674 cancelled and resubmitted sharded, mb_eval re-pointed.
- 16:55, M-C, recorded before any M-C AggreFact number was read: the coverage-cell pool the training script draws
  from is every labelled (summary, human-ACU) cell of the 853 e0 training documents across all systems, 56,118
  cells, not "the 16,332 natural coverage cells" the design named; 16,332 is the number of cells the deployed
  600-step recipe touched, not the pool. The unified run draws 8 cells per step for 1,500 steps (12,000 draws
  with replacement) from the 56,118, mixed with 8 synthetic pairs from the 14,395 of C2D + D2C. The bars are
  unchanged. Coverage side of the first two seeds (RoSE validation, same cells for every arm): unified AUROC
  0.9697 / 0.9697, crossed 0.8965 / 0.9007; deployed 0.9740, 0.9170; zero-shot FactCG 0.9125, 0.8595.

## Amendment M-C.1 (2026-09-23 17:00, coverage-side outcome, recorded after the run and before any M-C AggreFact number was read)

Three seeds (jobs 6687_0–2; `mc/seed*/result.json`). On RoSE validation, identical cells for every arm: unified
verifier pooled AUROC 0.9697 / 0.9697 / 0.9695 (mean 0.9696), crossed accuracy (document macro) 0.8965 /
0.9007 / 0.8999 (mean 0.8990); deployed verifier 0.9740 and 0.9170; zero-shot FactCG 0.9125 and 0.8595.
**AUROC bar (≥ 0.9640): PASS. Crossed bar (≥ 0.90): FAIL by 0.001 on the registered point threshold.** The
block requires all three bars, so its verdict is FAIL regardless of the AggreFact bar, which will be applied
and reported when the scoring finishes; the unified verifier is reported as a near miss (it recovers 0.057 AUROC
and 0.040 crossed accuracy over the encoder it starts from, and sits 0.004 AUROC and 0.018 crossed below the
deployed coverage verifier). The package's default remains two verifiers. Not iterated.

**AggreFact bar, added 2026-09-23 22:21 (job 6689, closes the block).** Macro BAcc, whole-claim, three-seed mean:
unified verifier 76.25, against a threshold of 75.52 (A1 FactCG-whole 76.52 minus the registered 1.0-point
tolerance) — **PASS**. So the unified verifier is competitive on whole-claim AggreFact detection and passes two
of its three bars (AggreFact, AUROC); it is the single crossed-discrimination bar, decided already above, that
fails, by 0.001. `pass_all: false` in `mc/eval/mc_bars.json`. Verdict unchanged: **FAIL**, block closed, not
iterated — this AggreFact number completes the record rather than reopening the question.

## Amendment M-A.1 (2026-09-23 17:45, M-A outcome, recorded after the run)

Jobs 6660 (decomposition), 6676 (scoring; AlignScore rerun 6786 with the token cap), 6857 (meta-evaluation with
the constant-column tolerance; the earlier 6853 run differed only in the no-candidate control columns, which it
ranked on float noise). `ma/meta/meta_evaluation.json`, SHA256SUMS in `ma/meta/`. 1,828 UniSumEval, 1,600
SummEval and 2,688 RoSE pairs; summary-level Kendall τ, document-clustered 2,000-draw intervals.

**Bar (i) — FAIL as registered (three of four contrasts pass).** MARS-R (reference-based, coverage verifier)
vs ROUGE-L / BERTScore-F on UniSumEval completeness: +0.268 [+0.211, +0.320] and +0.323 [+0.267, +0.377]
(PASS); on SummEval relevance: +0.104 [+0.044, +0.164] (PASS) and +0.049 [−0.003, +0.105] (FAIL, the
contrast predicted as uncertain). Observed τ: UniSumEval completeness MARS-R 0.361 (FactCG verifier 0.417),
ROUGE-L 0.093, BERTScore-F 0.038, BERTScore-R 0.236, G-Eval 0.410, G-Eval+ 0.478; SummEval relevance MARS-R
0.290, BERTScore-R 0.298 (−0.007 [−0.048, +0.037]), BERTScore-F 0.242, ROUGE-L 0.186; RoSE ACU recall MARS-R
0.578, BERTScore-R 0.425 (+0.153 [+0.121, +0.186]), ROUGE-1 0.371, ROUGE-L 0.306. System-level Pearson r of
MARS-R with completeness 0.96 (ROUGE-L −0.09, BERTScore-F 0.27), with RoSE ACU recall 0.98.

**Bar (ii) — FAIL as registered (two of four contrasts pass).** MARS-P (FactCG verifier, mean over facts) vs
AlignScore / FactCG-whole on UniSumEval faithfulness: −0.051 [−0.101, −0.005] and −0.068 [−0.105, −0.033]
(FAIL); on SummEval consistency: +0.117 [+0.076, +0.161] and +0.001 [−0.033, +0.038] (PASS). Observed τ:
SummEval consistency MARS-P 0.390, FactCG-whole 0.389, AlignScore 0.272, BARTScore 0.206; UniSumEval
faithfulness MARS-P 0.123, FactCG-whole 0.192, AlignScore 0.174, G-Eval+ 0.565. The minimum over facts is not
better than the mean (0.390 / 0.092). Prediction "MARS-P within 0.03 of AlignScore": wrong on UniSumEval.

**Secondary (iii)** not run yet (the reference-free MARS-R needs the importance-tagger weights over the source
inventories; queued behind M-B). **Controls.** MARS-P's shuffled-source control is at zero on every dimension
(−0.04 to +0.04): MARS-P reads the source. MARS-R with the FactCG verifier: shuffled-candidate control at zero
everywhere (−0.03 to +0.04). MARS-R with the coverage verifier: shuffled-candidate control 0.222 on UniSumEval
completeness (real 0.361; real minus shuffled +0.139 [+0.080, +0.195]), 0.060 on SummEval relevance, 0.031 on
RoSE: the deployed verifier's scores carry a system-level prior that survives replacing the candidate; the
FactCG-verifier reading does not. A no-candidate score is constant within a document and has no ranking.

**Findings beyond the bars.** (a) On the coverage axis MARS-R is the best non-LLM metric on all three sets and,
with the FactCG verifier, level with the GPT-4 judge on UniSumEval completeness (0.417 vs G-Eval 0.410; G-Eval+
0.478, −0.061 [−0.102, −0.021]). The verifier choice splits by pool: FactCG's verifier is better on UniSumEval
(+0.056 [+0.024, +0.092]), the deployed one on RoSE (+0.065 [+0.037, +0.093]), level on SummEval. (b) The two
axes anti-correlate across systems: MARS-P's system-level r with completeness is −0.68 on UniSumEval and its
summary-level τ −0.098; systems that say more omit less and support less. (c) Decomposing the candidate into
facts does not help the summary-level faithfulness correlation (bar (ii) on UniSumEval); the detection setting
is M-B's question.

**Consequence for the paper.** The metric role is claimed for the coverage axis with the comparators and the
one narrow miss stated (relevance vs BERTScore-F); the faithfulness axis is reported as level with the best
small verifier on SummEval and below it on UniSumEval, with the whole-summary reading offered as the package's
faithfulness default for summaries if M-B confirms the pattern. Both bars are reported as failed. Opened at this
harvest: M-E (cost).

## Amendment M-A.2 (2026-09-23 20:01, secondary (iii), no bar)

Job 6877 (E2 importance tagger over the UniSumEval/RoSE source inventories, three seeds), `ma_reffree.py` run
by hand against `ma/tasks/pairs_ma.jsonl`, job 6932 (`ma_meta` rerun folding `scores_marsR_free.jsonl` into
`ma/meta/meta_evaluation.json`). marsR_free(c) = Σ w_i(1−z_i)/Σw_i over source units: z_i the deployed coverage
verifier's omission probability of unit i given the candidate, w_i the E2 tagger's importance weight — no
reference summary is read at all. 1,813 UniSumEval and 2,688 RoSE pairs scored (4,501 rows).

**UniSumEval completeness: marsR_free reads *above* both reference-based readings** — 0.443 weighted, 0.473
unweighted, against 0.361 for the deployed (coverage) verifier and 0.417 for the FactCG verifier. Controls:
shuffled-candidate 0.117 (real − shuffled = +0.326, a materially cleaner separation than the deployed verifier's
reference-based reading, whose shuffled-candidate control sits at 0.222 — Amendment M-A.1), no-candidate −0.029
(clean). On faithfulness and conciseness marsR_free reads near zero (0.016, 0.038), as expected for a coverage
score not built to track those dimensions.

**RoSE ACU recall: marsR_free reads well below both reference-based readings** — 0.227 against 0.578 (deployed
verifier) and 0.514 (FactCG verifier); controls are clean (shuffled-candidate 0.009, no-candidate 0.008) but the
signal itself is weak. RoSE's ACUs are fine-grained, reference-authored units; approximating "what a reference
would carry" from source-side importance alone loses more on this set than on UniSumEval's coarser per-key-fact
completeness.

**Reading.** The reference-free coverage score is not a drop-in replacement for the reference-based one — it
splits by pool, better on one set and substantially worse on the other, same as the verifier-choice split in
Amendment M-A.1. But on UniSumEval it does not merely survive the removal of the reference, it reads *cleaner*
than the reference-based version (larger real-vs-shuffled margin, near-zero no-candidate). Since a live RAG
pipeline has no human reference to score against, this is the reading that would actually run in that setting;
it is offered as a secondary, unregistered capability, not a claim that supersedes bar (i).

## Amendment M-B.1 (2026-09-23 20:30, M-B outcome, recorded after the run)

Jobs 6799/6800 (`mb_score`, sharded 4-way over dev and test, two verifier arms), 6943 (`mb_eval`; the first
attempt, job 6675, hit an NFS visibility race — both arms read as "score files missing" against files that had
in fact just been written by 6800's last shard, and exited 0 with an empty report; resubmitted once the files
were independently confirmed present). LLM-AggreFact test, 29,320 rows over 11 datasets, per-dataset dev-tuned
thresholds, 2,000-draw claim bootstrap. `mb/eval/mb_eval_factcg.json`, `mb/eval/mb_eval_marsc.json`, SHA256SUMS
in `mb/eval/`.

**Bar — FAIL for both verifier arms, and not merely short of +1.0: significantly negative.** A2 (FactCG,
decomposed-min) − A1 (FactCG, whole): macro BAcc **−1.21 points**, CI95 [−1.93, −0.50]. A4 (deployed MARS-C
verifier, decomposed-min) − A3 (same verifier, whole): **−1.42 points**, CI95 [−2.31, −0.50]. Both intervals
exclude zero on the wrong side of the registered bar. The registered prediction ("passes; gain carried by
RAGTruth/ExpertQA/Lfqa") is wrong: decomposing the claim and scoring by the minimum over facts does not help
detection on this benchmark for either verifier — it hurts, on average, by a point or more.

**Per-dataset (worse than 1.0 point, secondary check — FAIL for both).** FactCG arm: worst is AggreFact-CNN
(−4.45), also TofuEval-MediaS (−3.11), Wice (−2.18), RAGTruth (−1.28), Lfqa (−1.27); two datasets go the other
way, FactCheck-GPT (+0.43) and Reveal (+0.40), ExpertQA flat (+0.02). MARS-C arm: worst is AggreFact-CNN
(−6.17), also Lfqa (−4.16), FactCheck-GPT (−3.71), Wice (−2.26), ClaimVerify (−1.77); AggreFact-XSum goes the
other way and clears the per-dataset bar on its own (+2.45), as does RAGTruth (+1.00, exactly at threshold).
The pattern is not the one predicted (RAGTruth/ExpertQA/Lfqa carrying a gain) — if anything the sign is inverted
on Lfqa and the gain, where it exists, shows up on different datasets than registered (AggreFact-XSum, RAGTruth
for the MARS-C arm only).

**Reading.** This is the direct, most heavily-instrumented test of whether the package's core mechanism —
decompose the candidate, verify each fact, keep the worst — helps hallucination/omission detection over scoring
the whole claim. It does not, for either verifier, and the failure is concentrated on summarization-style
datasets (AggreFact-CNN/XSum, TofuEval) rather than the longer QA-style claims where per-fact windowing was
expected to matter (RAGTruth, ExpertQA, Lfqa) — the opposite of the registered prediction. Read together with
H-J9 (transfer fail), H-J10a (crossed-endpoint saturation) and M-C (crossed bar fails by 0.001), this closes the
detection-role question across four independent tests: decomposition-based verification does not clear a
registered bar anywhere in this program. Per ORACLE §3, closed, not iterated. The paper's central claim moves to
the metric role (M-A, coverage axis); the detection work is reported as a rigorous negative result with a
documented mechanism (fragmentation hurts short claims, may help longer multi-fact ones — MARS-C's AggreFact-XSum
and RAGTruth cells are the only two contrasts in the whole M-B/M-C/H-J program that clear +1.0 individually).

## Amendment M-A.3 (2026-09-24, registered before any of it ran): additional comparators

A reviewer of the manuscript asked for the coverage metrics the M-A comparator list did not include. Added over the
identical `pairs_ma.jsonl` (1,828 UniSumEval, 1,600 SummEval, 2,688 RoSE), each written as its own score file and
folded by the unchanged `ma_meta.py` (same bootstrap draws, same statistics, contrasts of every MARS column against
each new column): **UniEval** (`MingZhong/unieval-sum`; coherence, consistency, fluency, relevance and their mean),
**A3CU** and **A2CU** (AutoACU, the successors of Lite$^3$Pyramid by the RoSE authors; reference-based recall, and F
for A3CU, maximum over references), **QuestEval** (source-based, its precision and recall directions reported
beside its own score; weighter off), **QAEval** (EM and F1 against the reference, maximum over references; the
released generation and answering models, CPU). Code `code/ma_ext_score.py` with workers in `code/workers/`; job
`slurm/ma_ext_score.sbatch` (31 array tasks), then `ma_meta.sbatch` (afterany) and `ma_table.py`.

No bar (bars (i) and (ii) stand as decided in M-A.1). **Predictions**, recorded now: MARS-R (either verifier) is above
QuestEval, QAEval and UniEval-relevance on UniSumEval completeness and on RoSE ACU recall; A3CU is the strongest of the
new comparators on RoSE, since it was trained on RoSE's own ACU annotations (its training split, CNN/DM), and is
closest to MARS-R there; on SummEval relevance UniEval-relevance may be level with or above MARS-R, since UniEval was
trained on pseudo-labels derived from CNN/DM references. A new comparator that beats MARS-R on a target is reported
as such in the table and the text.

**Run log, M-A.3 (2026-09-24).** Jobs 7027 (UniEval, A3CU, A2CU, QuestEval x4) and 7039 (QAEval x4), meta 7044.
QAEval's released stack (torch 1.6) has no H200 kernels; it runs in `~/.venv-qaeval-gpu` (the same qaeval 0.1.0,
transformers 3.0.2, allennlp stack with torch 2.4), and on 20 pairs its F1 is identical to the released CPU stack
(20/20, maximum difference 0.0; `ma/qaeval_equiv/`). The released generation archive predates BART's
`final_logits_bias` buffer; the worker fills that one missing key with zeros, its value in bart-large.
**A2CU defect, found before its numbers were used:** autoacu reads texts through line-based files, and five
references contain `\r` / `\x0b`-class characters that the reader treats as line ends, so the generated ACUs were
shifted against their references (2,637 of 2,688 RoSE scores exactly 0). The first A2CU output and its ACU cache
are moved to `ma/superseded_a2cu/`; the worker now collapses all whitespace and asserts one generated ACU list per
reference; A2CU is rerun as job 7046 with the meta-evaluation 7047. The other four comparators pass texts in memory
and are unaffected.
**Meta-evaluation tolerance (2026-09-24).** The reference-free MARS-R no-candidate control is constant within a
document up to GPU batching jitter of at most 3e-5, which exceeded the 1e-6 tolerance `ma_meta.py` uses to call a
column constant, so its within-document tau ranked that jitter. The tolerance is raised to 1e-4 (every real metric
column varies within a document by orders of magnitude more) and the meta-evaluation rerun; no other column changes.

## Amendment M-A.4 (2026-09-27, registered before it ran): reference-free MARS-R on SummEval

Secondary, no bar, as M-A.2. The 100 SummEval sources get the Gemma+ent inventory by the frozen M32 prompt and the
`mc_unisum.py` rules (`code/ma_summeval_inventory.py`), then the deployed coverage verifier under the three candidate
modes, the E2 importance tagger and `ma_reffree.py`, into `ma/scores/scores_marsR_free_summeval.jsonl` (job 7660),
folded by `ma_meta.py` (job 7661). The build manifest records any SummEval document whose source overlaps the
MARS-C training sources. Prediction: on SummEval relevance the reference-free reading is below the reference-based
one (0.290), as on RoSE, because SummEval's relevance judgements are made against the source's key content broadly
rather than per key fact.

**Outcome, M-A.4 (2026-09-27, jobs 7660/7661).** Reference-free MARS-R on SummEval: relevance 0.240 (shuffled
candidate 0.005), below the reference-based 0.290 as predicted; consistency 0.248, coherence 0.145, fluency 0.160.
Three of the 100 SummEval sources overlap the MARS-C training sources (build manifest). No other cell of the
meta-evaluation changed.

## Amendment M-G (2026-09-28, registered before any of it ran): the metric table on common populations

Response to the 2026-09-28 manuscript review (findings F01, F02, F03, F11, F13) and the 2026-09-27 experiments
audit (medium item 1). Secondary; no bar is added and no earlier verdict is re-decided. The published M-A numbers
remain the record of what the registered estimator gave; what follows decides how they may be read.

- **Common cohort (F01, F13).** `ma_meta.py --common-cohort` over the Table 3 columns: per (dataset, dimension), the
  pairs on which every column that exists on that dataset is scored, then the documents on which every one of them
  has a defined within-document tau; the same 2,000 document draws (seed 20260918). `--pairwise-common` recomputes
  every MARS-vs-comparator contrast on the pairs and documents the two columns share. Counts of pairs and
  documents are reported per column and per contrast. The weighted versus unweighted reference-free MARS-R reading
  is compared on the common cohort only.
- **Decision rule, fixed now.** A ranking or superiority sentence in the paper survives only if it holds on the
  common cohort (point estimate) and its pairwise-common contrast interval excludes zero; "level with" is used only
  when the pairwise-common 95% interval lies inside ±0.05 tau; otherwise the paper reports the estimates without an
  ordering word. The not-below reading of M-A (ii) stays as registered; beside it the report gives whether the 95%
  lower bound clears −0.05 (`--ni-margin 0.05`), a margin chosen now, after M-A was read, and labelled so.
- **Multiplicity (audit M1).** The declared primary family is the eight contrasts of M-A bars (i) and (ii). Holm's
  step-down adjustment of the bootstrap two-sided p over that family is reported beside each verdict; every other
  contrast in the paper is exploratory and is labelled as such.
- **SummEval overlap (F11).** Every reading repeated without the three SummEval sources that occur among MARS-C
  training sources (`ma/summeval_source/build_manifest.json`).
- **RoSE without gold ACUs (F03).** MARS-R (MARS-C verifier and FactCG) on the 2,688 RoSE test pairs with units
  extracted automatically from the reference: (a) the frozen M32 Gemma-4-31B sentence decomposition (the extractor
  MARS-R uses on UniSumEval and SummEval), (b) the ACUs A2CU itself generated from the same references (cached by the
  A2CU run), so MARS-R and A2CU read identical units. The published gold-ACU row becomes an oracle-unit diagnostic.
  Prediction: both automatic arms are below the gold-ACU 0.578; arm (b) against A2CU isolates the matcher.

Jobs: `slurm/mg_metric.sbatch` (`MG_STAGE=rose` GPU, then `MG_STAGE=meta` CPU). Outputs under `mars_metric/mg/`.

**Outcome, M-G meta (2026-09-28, job 7899).** The default run reproduces the published table (UniSumEval completeness
MARS-R 0.361, reference-free 0.443). Common cohort (UniSumEval 911 pairs, 73-78 documents per dimension; SummEval
1,600 / 96-100; RoSE 2,688 / 223), UniSumEval completeness: G-Eval+ 0.479, reference-free unweighted 0.430, G-Eval
0.399, MARS-R FactCG 0.390, reference-free weighted 0.367, MARS-R MARS-C 0.308, A3CU 0.262, A2CU 0.234. MARS-R FactCG
minus G-Eval −0.009 [−0.081, +0.069]; reference-free minus G-Eval −0.032 [−0.111, +0.054]; MARS-R MARS-C minus A3CU
+0.045 [−0.004, +0.099]; weighted minus unweighted reference-free −0.063 [−0.110, −0.019]. RoSE MARS-R minus A2CU
+0.007 [−0.031, +0.046]; SummEval relevance MARS-R minus UniEval −0.053 [−0.104, −0.001]. Bars on the common cohort: (i)
UniSumEval passes, SummEval fails (BERTScore-F +0.049 [−0.003, +0.105]); (ii) SummEval passes, and on UniSumEval both
intervals include zero (AlignScore −0.015 [−0.090, +0.059], FactCG-whole −0.037 [−0.087, +0.010]), so the not-below
reading would pass on the common population while the registered available-case verdict (FAIL) stands; neither
clears the 0.05 margin. Holm over the eight (available-case): SummEval relevance vs BERTScore-F p=0.138, SummEval
consistency vs FactCG-whole 0.978, UniSumEval faithfulness vs AlignScore 0.078, all others ≤0.004. Without the three
overlapping SummEval sources every SummEval tau moves by at most 0.010 and no verdict changes. The paper's Table 3
now carries the common-cohort values (`code/mg_table.py`), the available-case table moves to the appendix, and every
ranking sentence follows the decision rule above.

**Outcome, M-G RoSE without gold ACUs (2026-09-28, jobs 7883/7884).** 286 references, 702 sentence tasks, every one of
the 2,688 pairs covered by both arms. Common cohort (2,688 pairs, 223 documents): MARS-R (MARS-C verifier) gold ACUs
0.587, Gemma units 0.528, A2CU units 0.537; FactCG reader 0.529 / 0.481 / 0.482; A2CU 0.580, A3CU 0.499. Automatic arms
minus A2CU: −0.052 [−0.085, −0.015] (Gemma), −0.043 [−0.071, −0.015] (A2CU units); minus A3CU +0.029 [−0.002, +0.060],
+0.038 [+0.010, +0.067]. Gold minus Gemma units +0.059 [+0.033, +0.086]. Prediction confirmed: both automatic arms are
below the gold-ACU reading; "level with A2CU on RoSE" holds only with oracle units. Shuffled-candidate controls
0.014-0.019. Table 3 carries the Gemma-unit row.
Run note, M-G: configuration (5), every MARS contrast on the pairs both columns share over the full data, was added to
`mg_metric.sbatch` after job 7884 had been queued, so Slurm's copy did not contain it; it runs as job 7905 with the
identical `ma_meta.py` call.

**Outcome, M-G (5) shared-pair contrasts (2026-09-29, job 7905), and the decision rule applied.** UniSumEval
completeness: MARS-R FactCG minus G-Eval −0.003 [−0.049, +0.042] (1,440 pairs / 178 documents) → level; MARS-R MARS-C
minus A3CU +0.080 [+0.039, +0.122] → above (common-cohort point +0.045); reference-free minus G-Eval +0.038 [−0.016,
+0.096] with a negative common-cohort point → no ordering; weighted minus unweighted reference-free −0.051 [−0.078,
−0.024]. RoSE MARS-R minus A2CU +0.004 [−0.033, +0.043] (gold units, level); automatic units −0.054 / −0.045, below.
SummEval relevance unchanged (below UniEval, not separable from BERTScore-F). UniSumEval faithfulness: MARS-P below
whole-summary FactCG −0.068 [−0.105, −0.033] and AlignScore −0.051 [−0.101, −0.005]. The paper's sentences follow
these readings.

## Amendment M-H (2026-09-29, registered before any of it ran): reporting analyses for the 2026-09-29 review

Response to the full manuscript review of 2026-09-29 (findings R1-W3, R1-W4, R1-W6). Reporting analyses of
existing score files; no bar, no new model, no new scoring, no verdict re-decided. Every value is descriptive.

- **(a) Cohort composition of Table 4 (R1-W3).** `code/mh_cohort.py` reads `ma/tasks/pairs_ma.jsonl` and the
  `ma/scores` and `mg/scores` files exactly as `ma_meta.py --common-cohort` does (same `Column`, same τ rule,
  same column list) and reports, per (dataset, dimension) column of the common-population table, the counts at
  each stage: pairs carrying the human dimension; pairs every listed column scores (`n_pairs` of the caption);
  documents on which every column has a defined τ (`n_docs_all_tau`); and the pairs inside those documents,
  which is the number that actually contributes to the reported mean τ. For UniSumEval it adds the domain and
  source-length (word-count quartile) composition of the full pool, of the all-metric cohort and of the
  shared-pair populations of the contrasts quoted in the text (MARS-R/MARS-C vs A3CU, MARS-R/FactCG vs G-Eval,
  reference-free vs G-Eval, MARS-P vs FactCG-whole). Output `results_snapshot/mh_cohort_composition.json`.
- **(b) LLM-AggreFact cluster-resampling sensitivity (R1-W4).** `code/mh_cluster_boot.py` reproduces
  `mb_eval.py` (thresholds tuned per dataset and arm on dev, macro balanced accuracy, observed contrast
  decomp_min − whole) and adds a 2,000-draw bootstrap that resamples source documents (SHA-1 of the `doc` field)
  within each dataset instead of claims, for both verifiers, beside the claim-resampled interval of record and
  the claims-per-document count of every dataset. The point estimates are unchanged by construction; only the
  interval is added. Output `results_snapshot/mh_aggrefact_cluster_bootstrap.json`.
- **(c) Emission-rule controls (R1-W6).** A correction of provenance, not a run: the random-filter (0.311) and
  lead-position (0.281) validation values of the components table were read from
  `marsc/r09_validation/eval_validation/e2_emitted.json` (`humanfact+random`, `humanfact+leadpos`), the same
  `--strict-seeds --expect-seeds 3` evaluator call that gives the rule's 0.400; the caption saying they predate
  the seed-manifest repair is wrong and is corrected. Because those two arms apply the filter without the
  emission rule, `research/marsc_ieee_20260922/slurm/hj13_rule_controls.sbatch` (CPU) additionally applies the
  frozen rule (`mc_diversity.py`, cap 1) on top of each filter from the identical three-seed files and evaluates
  filter, filter+rule, rule and raw ranking in one evaluator call (validation, k=10, 500 document draws, primary
  `humanfact+div1`). No bar; the like-for-like reading replaces the mixed one in the table.
