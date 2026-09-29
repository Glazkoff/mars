# MARS-C strengthening campaign — results, 2026-09-18

Registered in `PREREG.md` (same directory) before execution. Artifacts under
`/home/user/results/marsc_strengthen/` on Euler; every job writes `SHA256SUMS` and a `complete.json`.
This file records what the executed blocks returned, **including the bars that failed**.

Status: **all waves complete, queue empty** (final check 2026-09-20). Wave A complete. Wave B complete
(B0-B8, including B1 TEST, B2, B3 eval and B4 scoring/eval). Wave C complete (C1-C4). Wave D: artifacts
frozen and 1,602 revisions generated; its blinded packets are staged but **no D2 outcome is computed** --
every D2 outcome needs two blinded reviewers plus adjudication.

Registered bars: **H-COND, H-SURF, B4 transfer and B1 gameability PASS**; **H-SEL and H-COMP FAIL** and are
reported as such; the R07 crossed interaction is **withdrawn**. The one endpoint that did not generalise to the
external pool is the *label-source* contrast (B3), and the manuscript is scoped accordingly.

---

## 1. Corrections this campaign FORCES on the manuscript

Each is a number or claim in the current paper that the campaign showed to be wrong. All are repairs, not
scientific losses, except where noted.

| # | What the paper says | What the campaign measured | Source |
|---|---|---|---|
| 1 | R07 crossed **interaction** +0.0066, pair-level CI [+0.0019,+0.0116] | Withdrawn. Document-clustered CI **[−0.0024,+0.0161]**, bootstrap p=0.149 — crosses zero | A2 |
| 2 | The **union** inventory arm beats the primary (0.4025 vs 0.3999) | Artefact of a mixed calibration contract. Under one contract the union arm **loses**: 0.3894 vs 0.4029 | A1a |
| 3 | Label-efficiency curve 0.4085/0.4023/0.3953/0.3982, spread 0.0132 | Raw contract: 0.4097/0.4027/0.3923/0.4011, spread **0.0174 (+32%)** — the spread widens | A1c |
| 4 | Rank-average ablation "humanfact.rank+div1 = 0.2787 vs 0.3765" | **Withdraw the row.** `mc_diversity.py` skips units scoring ≤ −9.0; on the seedrank scale ([−438,0] TEST, [−2235,0] UniSum) essentially every unit is already below −9, so the rule never fired. It measures a broken rule, not a worse aggregation | A5 |
| 5 | Distilled pipeline ≈0.43 s at one summary per source; "advantage falls to about 1.9×" | Distilled inventory cost is **understated 2.45×** (the published figure charges one-off *training* to inference and divides by a document count the segmenter never ran on). Corrected: **0.716 s**, and the advantage over MARS-2 becomes **≈1.15×** | B6 |
| 6 | Amortisation over **11.8** summaries per source | Sample artefact of the first-200-pair timing window. Split-wide RoSE validation is **7.716** summaries/source (1466 pairs / 190 documents). Units/pair likewise: split-wide 94.31 gemma+ent and 40.63 distilled vs the window's 165.54 / 69.92 | B6 |
| 7 | The off-the-shelf baseline rows (implicitly a full comparison) | Every published off-the-shelf row is **TEST-only**. `r06b_fast` wrote TEST results into a directory named `val/`; `r06_offshelf`'s real `val/` is empty. Validation numbers for that family did not exist until this campaign produced them | B1 |
| 8 | "a blinded **two-annotator** study", rates reported as **adjudicated** | Not supportable as written. See §2 — this is the campaign's most serious finding | A3b |
| 9 | QAPyramid recorded as having **zero overlap** with our sources | Inverted. Overlap is essentially **total**: 499/499 documents join our `rose_cnndm` example_ids, 498/498 match on sha1 of the first 200 chars, 497/498 near-duplicate at 8-gram Jaccard ≥0.5. **100 of its documents sit in our sealed TEST role**; 282 in TRAIN. No transfer arm may be planned on it | B7 |
| 10 | Gemma decomposition ≈6.637 s/document | **Stands.** Re-timed as active generation: 6.760 s/document, ratio 1.018 | B6 |

---

## 2. The most serious finding: the human study is effectively single-annotator

Registered as A3b, verified directly from `g6/labels_export_all.jsonl` (400 items × 2 annotators).

**Q1 (is the fact supported by the source?)** — Annotator 2 answered `A` on **all 400 items**. Contingency
A|A 342, A|B 0, B|A 58, B|B 0, so p_observed = p_expected = 0.8550 and Cohen's **kappa is exactly 0.0000 by
construction**. There is no second independent support judgement in this study.

**Q2 (does the summary convey it?)** — healthy for both raters: raw agreement 0.8800, kappa **0.7249**.

**Annotation timing — stronger and more checkable than the kappa.**

| | Annotator 1 | Annotator 2 |
|---|---|---|
| items | 400 | 400 |
| span | 54,086 s | 49,223 s |
| **median inter-item gap** | **49.6 s** | **0.94 s** |
| gaps under 3 s | 0.0% | 99.2% |
| gaps under 1 s | 0.0% | 85.7% |
| gaps > 60 s | 161 | 2 |
| total time in gaps ≤60 s | 7,599 s | **401 s** |

Annotator 1's trace is what human annotation looks like: median ~50 s per item, high variance (18.9, 28.0,
226.7 …). Annotator 2 entered 400 items in about **401 seconds of working time**, in near-constant ~1.0 s
steps (1.13, 1.03, 1.02, 0.99, 0.99, 1.01, 1.02, 1.01 …), split into three bursts by two long pauses.
`ts` is written per item at INSERT time by the label app, so this is the entry cadence, not an export artefact.

**Consequence.** The paper cannot describe this as two independent blinded annotators, and cannot describe its
rates as adjudicated. Separately, and independently of the timing: **"adjudicated" has never meant adjudicated
in this project.** The label app has annotator and admin roles only — no adjudicator — and
`mc_human_analyze.py` / `mc_human_rank.py` both take `names[0]`/`names[1]` and silently ignore a third rater.
Disagreements were **dropped**, not resolved. Dropping disagreements keeps the easy items, so every published
human-precision figure is biased upward by an amount this project has never measured.

**What has been built in response (D1).** A 500-item blinded sample (ids H401–H900, 147 source documents,
418 pairs) drawn from the **corrected** post-repair emission, frozen at sha256 `2b9acc51…81a2`, with a
collision gate against every previously issued id. Plus a forked label app with a real adjudicator role — the
adjudicator sees only items where the two annotators disagree, without seeing who said what — and
`d1b_adjudicate.py`, a scorer that reports the agreement-conditioned rate and the true adjudicated rate side by
side and prints their difference as `selection_gap`. The fork is **not** deployed to the live server; that is
the user's call.

**This requires a decision that is not mine to make:** whether to re-run the human study with annotators whose
traces can be verified, disclose the limitation and keep the single-annotator reading, or drop the human
precision claim. All three are defensible; the current wording is not.

---

## 3. Results that strengthen the paper

### 3.1 The calibration defect is closed, and converts into a robustness claim (A1)

- **Inversion exact.** `z_raw = (logit(p_cal) − intercept)/slope` reproduces the stored raw logits to
  max |err| **6.2e−14** across all 27 arm×seed validation files (263,520 cells), zero saturated probabilities.
  No rescoring from checkpoints was needed.
- **Internal null control.** All 11 validation cells that were already raw move by **exactly 0.0000** in both
  recall and omission precision, proving the two contract runs differ only in the score source.
- **Single-seed top-k is exactly calibration-invariant**: 0/1466 pairs change their top-10 set *or order*, all
  three seeds, with and without the emission rule. The exposure lives only in seed averaging: 154/1466 (10.5%)
  under consensus, 510/1466 (34.8%) single-model.
- **New:** without the rule both aggregations sit at 518/1466 (35.3%). The div1 rule **is itself a
  calibration-robustness device** for the consensus regime — it cuts exposure from 518 to 154.
- **A1e bounds the whole exposure.** `leadunion`, whose primary sort key is per-seed sentence-leader *counts*
  and is therefore exactly calibration-invariant, reproduces the consensus headline to **<3e−4 on all three
  splits** (validation 7.8e−5, TEST 2.7e−4, UniSumEval 2.5e−4).
- **A1d: the label-source result survives.** Raw-logit averaging gives human 0.9740 vs judge 0.9611, observed
  **+0.0128 [+0.0082, +0.0181]** — a +0.0001 shift from the published value, despite opposite-signed Platt
  intercepts and a 0.6094 vs 0.3219 prevalence mismatch.

### 3.2 The withdrawn Jaccard sensitivity comes back stronger (A4)

The registered direction was **backwards**, and in MARS-C's favour. The emitted unit's stopwords could never
enter the intersection but always inflated the union, so the asymmetric rule was *strict*, not lenient.
Under `--match-symmetric` every Jaccard-floor recall **rises** (validation j20 0.1811→0.2228; TEST j20
0.1855→0.2197; UniSum j34 0.1239→0.1620), and the human-vs-judge margin **widens on every pool** with the
document-clustered CI excluding zero everywhere:

| pool / floor | asymmetric (published, withdrawn) | symmetric |
|---|---|---|
| validation j20 | +0.0331 [+0.0103,+0.0540] | **+0.0582 [+0.0330,+0.0827]** |
| validation j34 | — | **+0.0249 [+0.0101,+0.0418]** |
| TEST j20 | +0.0277 | **+0.0451 [+0.0295,+0.0610]** |
| TEST j34 | +0.0062 **[−0.0011,+0.0138]** (crossed zero) | **+0.0151 [+0.0059,+0.0235]** (no longer crosses) |
| UniSum j20 | — | **+0.0572 [+0.0434,+0.0733]** |

And the shared-token rows are **exactly invariant**: 252 scalars compared at full float precision across
3 pools × floors ≥1/≥2/≥3 × every system, **zero differ**. The published +0.122/+0.070/+0.041 is now
*evidenced* as filter-invariant rather than asserted.

### 3.3 The human intervals barely move under document clustering (A3)

The declared degeneracy risk did **not** materialise: across all 80 (config × budget × system × estimator)
cells, **zero degenerate bootstraps**, and median interval widening from item-level to document-clustered is
only **1.031×** (range 0.907–1.500×). The reason is that the 400-item sample was drawn thinly across 136
distinct documents (~1.3 items per document inside a system-budget cell), so the item bootstrap was already
close to independent. A positive robustness result.

By contrast the R07 crossed effects widen **1.90–2.29×**, and the five per-system judged-omission rates
**2.42–2.67×** (naive bound sqrt(1388/151) = 3.03×).

### 3.4 The judge-confirmed yield lower bound favours natural coverage supervision (B5)

Judge-confirmed omissions per corrected top-10 list, validation, 151 document clusters, 5,000 bootstrap
replicates. **A lower bound, never precision** — printed beside it: judge recall on human-omitted 0.5535,
positive precision 0.9900, kappa 0.4727, n = 63,343 cells.

| system | judge-confirmed per 10-slot list |
|---|---|
| judgefact.distilled+div1 | 7.6962 [7.2028, 8.1631] |
| humanfact.distilled+div1 | 7.6526 [7.1677, 8.1142] |
| **humanfact+div1 (primary)** | **7.6043 [7.1369, 8.0494]** |
| judgefact+div1 | 7.3292 [6.8722, 7.7693] |
| mars2 | 6.8529 [6.4050, 7.2815] |

Paired contrasts vs the primary: **vs mars2 +0.7514 [+0.5119, +0.9837]**; **vs judgefact+div1 (same
inventory, judge-label verifier) +0.2751 [+0.1402, +0.4233]**. Both distilled arms are null.

The sharper margin is **distinct** confirmed units per document, deduplicated across that document's ~9
summaries: humanfact+div1 **17.32 [15.80, 18.89]** vs mars2 **11.07 [10.30, 11.82]** — about **56% more
distinct confirmed omissions per document**.

### 3.5 Selective prediction is the campaign's strongest new result (C1)

**The registered bar failed, and something better appeared on the same axes.**

**H-SEL fails as registered.** MARS-C does not dominate the risk–coverage curve: `mc.humanfact` dominates
0 of 10 off-the-shelf comparators over coverage [0.20,1.00]; `mc.humanfact.distilled` dominates 1 of 10. The
mechanism is specific and must be reported — **cheap lexical and embedding counters are better selective
predictors at low coverage**. At 20% coverage omission precision is 0.984 (lex token-recall), 0.980 (lex
ROUGE-L), 0.979 (MiniLM cosine) against MARS-C.distilled's 0.919 (paired doc-clustered −0.0648
[−0.1302,−0.0066] vs lex token-recall).

**But three positive results sit on the same curves:**

1. **MARS-C's distilled arm dominates SummaC-ZS across the coverage range** — and SummaC-ZS is the family that
   beats it on recall@10 (0.3848 vs 0.3765). `mc.humanfact.distilled` vs `zs.distilled.nli_summac_zs` passes
   **16/17** grid points (cov 0.20: +0.1413 [+0.0670,+0.2113]; cov 1.00: +0.0443); vs `zs.nli_summac_zs`
   **17/17** (all points pass; cov 0.20 +0.1496 [+0.0600,+0.2355], cov 1.00 +0.0706).
   **Verified correction (2026-09-18, re-read from the artifact):** this holds for the *distilled* arm. The
   **Gemma+ent pipeline of record is weaker** — 13/17 against distilled SummaC-ZS and 15/17 against Gemma+ent
   SummaC-ZS, and its full-coverage contrast against distilled SummaC-ZS **straddles zero** (+0.0128
   [−0.0194,+0.0452]). An earlier summary of this block said "MARS-C dominates SummaC-ZS across essentially the
   whole coverage range" without that distinction; the distinction matters, because the pipeline of record is
   the Gemma+ent arm.
2. **Selective prediction amplifies candidate-conditioning by 3–4×.** The paper currently reports +0.065
   omission precision for the real summary over the no-summary control at the full ten-slot budget. On the
   risk–coverage curve the same contrast is **+0.2480 [+0.1597,+0.3349] at 20% coverage** (vs empty) and
   **+0.1997 [+0.1334,+0.2673]** (vs shuffled), decaying monotonically to the familiar +0.0647/+0.0546 at full
   coverage. Dominance holds at **17/17** grid points for **both** controls and **both** inventories.
3. **The sharpest deployment number in the campaign.** The MARS-C judge-label arm has omission precision
   **1.000 [1.000,1.000] at 10% and 20% coverage**, 1.000 [0.9914,1.000] at 30% and 0.994 [0.976,1.000] at 50%
   **on the sealed TEST split** — while its own shuffled control sits at 0.697/0.668/0.657/0.659 and its
   no-summary control at 0.673/0.709/0.703/0.682. A ≈+0.33 gap no summary-blind emitter can close, on 2,688
   pairs over 285 documents. This arm is currently the paper's **worst** cell on recall@10 (0.255).

The curve is calibration-invariant by construction (it walks each family's own score order), so §3.1's
exposure cannot touch any number in this block.

### 3.6 Complementarity fails, and the failure is trustworthy (C3)

**H-COMP fails on all eight fusions.** The primary fusion (MARS-C.distilled ⊕ SummaC-ZS.distilled, mean
per-pair normalised rank) reaches precision 0.7423: +0.0349 [+0.0179,+0.0516] over SummaC-ZS but
**−0.0094 [−0.0194,−0.0003] below MARS-C** — significantly worse than its own stronger constituent.

The refutation is credible because its pre-declared controls behaved correctly: the **null-control** fusion
(MARS-C.distilled ⊕ lex token-recall) gives +0.0065 [−0.0017,+0.0158] over MARS-C, indistinguishable from
zero. Fusing does not automatically raise precision, so this is a real refutation, not a fusion-rule artefact.
Conversely the **B0 control fusions gained more over their constituents than the real fusion did**, which is
what kills the claim.

**Salvageable as a secondary:** `fuse.mc_summac.distilled` matches SummaC-ZS's recall@10 exactly (0.385 vs
0.385) while carrying **+0.035 omission precision**. "MARS-C's signal is complementary" is not supported;
"a rank fusion buys SummaC-ZS's recall at MARS-C's precision" is, with intervals.

### 3.7 The unseen third extractor is live and its inventory matches the development one (B4)

Qwen3.8-27B, revision `1d4bf0f2…`, frozen prompt sha256 and decoding contract recorded **before** execution,
`enable_thinking=False` asserted. Both decomposition shards completed: 7,342 held-out sentences → 21,742
propositions, 2.96/sentence, blank-decode 0.4%, empty-parse **7.0%** — *better* than the development
extractor's own 10.43% on the same sentences (the frozen prompt instructs `NONE` for a factless sentence, so
a nonzero empty rate is correct behaviour, not failure).

Inventory quality is equivalent to the development extractor, which is the precondition that makes the
transfer test interpretable:

| inventory | units/pair | ceiling (weak) | ceiling (strict) |
|---|---|---|---|
| **qwenprops+ent (unseen)** | 110.3 | **0.906** | 0.291 |
| gemma+ent (development) | 106.2 | 0.897 | 0.289 |
| distilled (development) | 43.7 | 0.848 | 0.254 |

### 3.8 Cost: the verifier swap is free, and vLLM unlocks the generation half (B6, B8)

First per-unit cost measurements for the off-the-shelf family under the b21 contract: lex token-recall 0.018
ms/unit, MiniLM 0.164, lex ROUGE-L 0.166, windowed RoBERTa-MNLI 1.014, AlignScore 1.135, **SummaC-ZS 3.379**,
MiniCheck 30.013. MARS-C's own verifier is 1.028 ms/unit × 3 seeds = **3.08 ms/unit**. So the fixed-inventory
verifier swap is **cost-neutral** against the strongest comparator — which strengthens the identification
story, because the comparison cannot be dismissed as a compute difference.

**vLLM works and is 25.9× faster** (0.521 GPU-s/document vs 13.506 for the same model and prompt under HF;
2166 generated tok/s vs 96.6). The first attempt failed in FlashInfer's JIT build with
`FileNotFoundError: 'ninja'` — ninja *is* installed at `/home/user/.venv-vllm/bin/ninja`, just not on the
job's PATH. A naive reading would have recorded "vLLM does not load this architecture", which is wrong.

### 3.9 MiniCheck and AlignScore are affordable after all (B1)

The 85 GPU-h estimate was an artefact. `minicheck`'s `inference_example_batch` loops one (doc, claim) pair at a
time, so `batch_size` is dead for short premises, and `device_map='auto'` adds an accelerate hook per
submodule. On a clean GPU the official path measures **41.9 units/s**, not 1.36 — the original figure was
measured after four other models had been loaded and deleted in the same process. A batched fast path was
validated against the official path on 200 spread units: **max |Δ| 7.9e−7, zero rank disagreements**, with the
job refusing to proceed if the deviation exceeds tolerance. AlignScore throughput measured for the first time:
**666 units/s** pilot, **2035 units/s** production; the `yzha/AlignScore` snapshot is 4.6 GB (real weights).
Total B1 compute: **≈2 GPU-h**, not 85.

### 3.10 Where the error actually lives (C4, partial)

On the weakest inventory (spaCy, validation, top-10): **53.9%** of all human-omitted ACUs are **inventory**
misses, 23.8% verifier misses, 2.6% rule misses, 19.7% hits — i.e. **67.1% [0.615, 0.733] of all misses** come
from the decomposition never producing a unit that overlaps the fact, against 29.7% [0.237, 0.352] for the
verifier and 3.2% [0.020, 0.047] for the emission rule. The gemma+ent ceiling is 0.8607 rather than spaCy's
0.4611, so the balance will shift — but the instrument works and the question is now decidable. **This is
decision-relevant for the governing proposal:** if inventory misses dominate on the deployed inventory too,
its matched-supervision training programme is aimed at the wrong stage.

### 3.11 Surface cues do not explain the signal (C2, partial)

On the spaCy tagger validation arm (45,317 units, 3 seeds), the omission score is 35.0% predictable from the
three registered surface cues — but only **1.4% from the two candidate-blind cues** (position in source, token
length), with 30.3% coming from unit–summary lexical Jaccard alone, at a **negative** coefficient (−0.268
standardised). The reviewer objection splits cleanly: the "long units / late units are just more often
omitted" half is **dead** (R² 0.014); the remaining predictability is *candidate-dependent*, i.e. it is
conditioning, not a shortcut.

A design note the registration missed: inside a same-fact crossed triple the fact is held fixed, so position
and token length are **identical** in the two cells and cannot move the win indicator at all. Only
fact–summary lexical overlap varies within a triple, so residualisation on that endpoint removes exactly the
one surface channel that could fake conditioning — a cleaner test than a generic surface control.

### 3.12 Smaller closures

- **A6.** The B1 census reproduces **exactly**: 85,600 cells over 2,625 pairs, `labels_none = 0`, and all three
  `training_config.json` files independently report `examples=85600`. Newly citable: those 2,625 pairs rest on
  only **1,411 distinct source documents** (mean 1.86 summaries/document), and the label set is not purely
  judge-derived — **84,646 judge-labelled cells (98.89%) plus 954 human-labelled (1.11%)**.
- **A8.** 34 interval sites classified: 24 document-clustered, 7 not, 3 that are not intervals at all. Three
  sites the paper's audit table does not name: `mc_equivalence.py:63` and `b6_rerank.py:65` (both correct —
  credits the table omits) and `a5_metric_selection.py:48` (the third defective site).
- **B7.** `ComposoAI/OmissionBench` is real and **disjoint** from our pool by measurement: HTTP 200, 161 source
  transcripts, **0 exact and 0 near-duplicate matches** against our 1,498 documents, median best-Jaccard
  0.0000. A genuine untouched candidate pool — subject to domain fit, since it is clinical dialogue-to-note.
- **B7.** Of QAPyramid's 50 documents carrying the 10-system human coverage evaluation, **12 fall in our
  sealed TEST role** — an independent human coverage label source from a different annotation protocol. Small,
  but genuinely independent.

---

## 4. Bars that failed, recorded as registered

| Hypothesis | Verdict |
|---|---|
| **H-SEL** (C1): MARS-C dominates the risk–coverage curve over [0.20,1.00] | **FAIL** — 0/10 and 1/10 comparators dominated. Cheap lexical counters are better selective predictors at low coverage |
| **H-COMP** (C3): fusion exceeds both constituents on omission precision | **FAIL** — all eight fusions; the primary is −0.0094 [−0.0194,−0.0003] *below* MARS-C. Refuted by its own pre-declared null control |
| **A2** interaction effect | **WITHDRAWN** — crosses zero under document clustering, exactly as the registration said it might |
| **H-COND** (B2) | **Still running.** Early 3-document CPU smoke put SummaC-ZS at 0.8942 crossed accuracy against MARS-C on the same documents (margin +0.0028, one-sided 95% LB −0.0126). n=3 is meaningless, but the registered fallback must be honoured if it holds: the paper may **not** pre-write "the only verifier family that demonstrably reads the summary" |

---

## 5. New operational traps discovered (for the project trap list)

1. **`sbatch --export` splits its value on commas.** `--export=ALL,B1_SYSTEMS=minicheck,alignscore` sets
   `B1_SYSTEMS=minicheck` and exports an unrelated variable named `alignscore` — proved by probe job 5305
   (`SYSTEMS=[minicheck] ALIGN=[UNSET]`). Silent. Use a different separator and translate in-script.
2. **A missing `--output` directory fails the job at 00:00:00 with no log.** Cost the whole B4 chain one cycle;
   `mkdir -p` the log directory in the submitting shell, not in the job body.
3. **`train` and `infer` are two partitions over the same eight H200s**, with `infer` at PriorityTier=100 and
   `train` at 1. A GPU job on `train` waits behind every pending `infer` job. All existing MARS-C GPU jobs use
   `infer`. The campaign brief's `train` header cost ~7 hours of queue.
4. **Walltime, not priority, is the binding constraint.** Identical jobs at `--time=00:45:00` were scheduled
   ~19 h out; the same jobs at `--time=00:30:00` backfilled within seconds.
5. **`mc_score_units.py` cannot score the judge-label verifier.** It is hard-wired to
   `torch.sigmoid(logits.squeeze(-1))`, correct for the 1-logit human-fact head; the b21 `xenc` checkpoints
   have a 2-logit head scored `softmax(logits,−1)[:,0]`. On an (N,2) tensor `squeeze(-1)` is a no-op and
   `.tolist()` yields lists, which would be written into the score file as non-scalar values.
6. **`mc_unisum.py inventory` hard-codes its output filename and stamps `"split": "unisum"`** into every row.
   Using it for a TEST-split inventory would mislabel 2,829 rows as an external pool.
7. **`b1/pairs_test.jsonl` carries `natural770`**, a B1 *training* resource that `acu_units_test.jsonl` does
   not label. Any inventory built from it without restriction silently pulls training documents into a
   held-out artifact. All 27 zero-proposition documents in B4 were natural770; restricted to the 286 evaluated
   documents, coverage is exactly 1.0000.
8. **`mc_human_rank.recover_rank`'s (system, pair_id, text, score) join is ambiguous** — on the 500-item D1
   batch, 57/500 items match more than one emitted unit by text, 4 survive the score tie-break, and 2 attach a
   status contradicting the sampler's own record. The shared script takes the first occurrence arbitrarily.

---

## 6. What remains blocked, honestly

Nothing in the governing proposal's end-to-end semantic claim — its useful-yield estimand `U_d`, its
0.90-precision confirmatory contract, the Block 1 bottleneck audit's support/usefulness labels, or the Block 3
revision outcomes — is obtainable without new blinded human annotation **with real adjudication**. The only
machine analogue for support is the judge, and support is precisely the question measured to fail (kappa
0.4727; calls 36% of emitted facts unsupported where annotators almost always say yes).

**D4 sizing, computed rather than quoted.** For a one-sided 95% Hoeffding bound on the proposal's bounded
per-document loss (range 1.0 at p0 = 0.90, N ≤ 10), the half-width is **0.2012 at 37 clusters** and **0.1888 at
42**. The target `E[L_d] ≤ 0` sits exactly at the point estimate when true precision is 0.90, so **the
proposal's 0.90-precision confirmatory contract is unreachable at the human study's current scale** unless
observed precision is far above 0.90 or the study grows to roughly **120–200 document clusters** (half-width
0.112–0.087). That target must be revised before any confirmation label is read.

Hoeffding, empirical-Bernstein, Maurer–Pontil, Waudby-Smith–Ramdas betting and Clopper-Pearson exist
**nowhere** in this codebase (a repo-wide scan; the four apparent "betting" hits are regex false positives).
Only Wilson and a scipy sign test exist, and both are unclustered. D4 is 100% new code.

---

## 7. B2 / H-COND — the campaign's primary new hypothesis PASSES on validation

Bars were written into `b2/hcond_verdict.json` before any number in that file was read
(`declared_before_any_number_in_this_file_was_read: true`).

**Primary estimand.** Within-document, same-fact crossed accuracy
`P(z(f | a summary that omits f) > z(f | a summary that conveys f))`, ties counted 1/2, document-macro,
document-clustered bootstrap. **Any summary-blind scorer is pinned at exactly 0.500.**
Validation: 5,464 cells → 9,523 triples over 569 facts and 138 documents.

| verifier family | crossed accuracy (doc-macro) | 95% CI | off-the-shelf |
|---|---|---|---|
| **MARS-C human-fact (ours)** | **0.9170** | [0.8971, 0.9351] | — |
| windowed RoBERTa-large-MNLI | 0.8911 | [0.8676, 0.9135] | yes |
| MARS-C judge-label twin | 0.8872 | [0.8589, 0.9147] | — |
| SummaC-ZS | 0.8658 | [0.8403, 0.8900] | yes |
| lexical token-recall | 0.8304 | [0.8039, 0.8570] | yes |
| lexical ROUGE-L recall | 0.8056 | [0.7751, 0.8361] | yes |
| MiniLM cosine | 0.7885 | [0.7572, 0.8183] | yes |

**Paired margins against MARS-C, document-clustered, one-sided 95% lower bound — every one above zero:**

| comparator | observed margin | one-sided LB95 |
|---|---|---|
| windowed NLI | +0.0259 | +0.0137 |
| MARS-C judge-label twin | +0.0298 | +0.0146 |
| **SummaC-ZS** | **+0.0513** | **+0.0379** |
| lexical token-recall | +0.0866 | +0.0680 |
| lexical ROUGE-L | +0.1114 | +0.0883 |
| MiniLM cosine | +0.1286 | +0.1073 |

**Why this matters more than the recall@10 table.** SummaC-ZS *beats* MARS-C on recall@10 (0.3848 vs 0.3765).
On the endpoint that a summary-blind system cannot win, MARS-C beats it by +0.0513 [LB +0.0379]. Together with
§3.5 — MARS-C dominates SummaC-ZS on omission precision at 16/17 and 17/17 coverage grid points — the exchange
is now fully characterised: **the recall@10 loss buys conditioning and precision at every operating point.**

**Registration correction, recorded rather than quietly fixed.** The PREREG stated that lexical and embedding
counters "SHOULD come out near 0.500 and that is the sanity check". That was **wrong**: a token-overlap or
embedding counter reads the candidate, so it moves when the candidate changes, and they land at 0.79–0.83.
The object genuinely pinned at 0.500 is a scorer that never sees the candidate — the B0 `empty` control.
No family in this table is candidate-blind.

### FINAL VERDICT: **H-COND PASSES** on validation *and* sealed TEST

`b2/hcond_verdict.json`: `hcond_verdict: "PASS"`, `families_not_cleared: []`, `candidate_blind_families: []`.
Sealed TEST: 286 documents, 11,554 cells, **20,000 triples** over 1,160 facts / 241 documents with triples.

| verifier family | crossed accuracy (doc-macro) | 95% CI | off-the-shelf |
|---|---|---|---|
| **MARS-C human-fact (ours)** | **0.9140** | [0.8982, 0.9287] | - |
| windowed RoBERTa-large-MNLI | 0.9002 | [0.8832, 0.9148] | yes |
| MARS-C judge-label twin | 0.8931 | [0.8745, 0.9094] | - |
| SummaC-ZS | 0.8549 | [0.8336, 0.8751] | yes |
| lexical token-recall | 0.8281 | [0.8088, 0.8460] | yes |
| lexical ROUGE-L recall | 0.7982 | [0.7762, 0.8200] | yes |
| MiniLM cosine | 0.7848 | [0.7594, 0.8088] | yes |

Every comparator clears on **both** splits with the document-clustered one-sided 95% lower bound above zero:

| comparator | TEST margin (LB95) | validation margin (LB95) |
|---|---|---|
| windowed NLI | +0.0138 (+0.0041) | +0.0259 (+0.0137) |
| MARS-C judge-label twin | +0.0209 (+0.0114) | +0.0298 (+0.0146) |
| **SummaC-ZS** | **+0.0591 (+0.0449)** | +0.0513 (+0.0379) |
| lexical token-recall | +0.0859 (+0.0717) | +0.0866 (+0.0680) |
| lexical ROUGE-L | +0.1158 (+0.0988) | +0.1114 (+0.0883) |
| MiniLM cosine | +0.1291 (+0.1108) | +0.1286 (+0.1073) |

**The SummaC-ZS margin is LARGER on TEST than on validation** (+0.0591 against +0.0513) - and SummaC-ZS is the
one off-the-shelf arm that beats MARS-C on recall@10 (0.3854 against 0.3765). The two endpoints point in
opposite directions on the same pair of systems, which is precisely the campaign's thesis: recall@k is not
measuring what an omission detector is for.

---

## 8. B1 TEST — the gameability result generalises to every verifier family

`b1/test/eval_test/e2_emitted.json`, 46 emitters over 2,688 pairs / 285 documents, `--strict-seeds
--expect-seeds 3`. MiniCheck and AlignScore run as MARS-C verifiers **for the first time in this project**, and
every off-the-shelf family carries its B0 shuffled and no-summary controls, per the campaign's standing rule.

### The headline: recall@10 is won by ignoring the summary, in 13 of 14 families

| verifier family (distilled / Gemma+ent) | rec@10 real | shuffled | no summary | om.prec real | shuffled | none |
|---|---|---|---|---|---|---|
| distilled · MiniLM cosine | 0.3284 | 0.3966 | **0.4198** | **0.7541** | 0.6824 | 0.6920 |
| distilled · lex ROUGE-L | 0.3225 | 0.3896 | **0.4613** | **0.7488** | 0.6842 | 0.6652 |
| distilled · lex token-recall | 0.3258 | 0.4397 | **0.4613** | **0.7457** | 0.6798 | 0.6652 |
| distilled · windowed NLI | 0.3712 | 0.3656 | **0.3773** | **0.7386** | 0.6750 | 0.6972 |
| distilled · MiniCheck | 0.3761 | **0.4280** | 0.4110 | **0.7365** | 0.6921 | 0.6902 |
| distilled · AlignScore | 0.3608 | 0.3660 | **0.3816** | **0.7304** | 0.6846 | 0.6891 |
| Gemma+ent · lex token-recall | 0.3206 | 0.4589 | **0.4775** | **0.7248** | 0.6579 | 0.6465 |
| Gemma+ent · lex ROUGE-L | 0.3199 | 0.3733 | **0.4775** | **0.7236** | 0.6539 | 0.6465 |
| Gemma+ent · MiniLM cosine | 0.3188 | 0.3839 | **0.4127** | **0.7201** | 0.6539 | 0.6632 |
| Gemma+ent · windowed NLI | **0.3589** | 0.3529 | 0.3393 | **0.7181** | 0.6448 | 0.6500 |
| Gemma+ent · AlignScore | 0.3540 | 0.3498 | **0.3681** | **0.7105** | 0.6496 | 0.6568 |
| Gemma+ent · MiniCheck | 0.3574 | **0.4120** | 0.3940 | **0.7093** | 0.6614 | 0.6502 |
| distilled · SummaC-ZS | 0.3854 | 0.3696 | **0.3920** | **0.7083** | 0.6836 | 0.6848 |
| Gemma+ent · SummaC-ZS | 0.3673 | 0.3437 | **0.3867** | **0.6803** | 0.6389 | 0.6485 |

- **13 / 14 families** score *higher* recall@10 under a summary-blind control than with the real summary.
  The single exception is Gemma+ent windowed NLI (0.3589 real against 0.3529 and 0.3393).
- **14 / 14 families** lose omission precision under **both** controls, without exception.

The paper currently owns this finding for MARS-C alone (0.377 real → 0.440 shuffled → 0.458 none). It is
now established across lexical, embedding, entailment, SummaC, MiniCheck and AlignScore verifiers over two
inventories on the sealed split. That converts a self-criticism into a **field-level methodological result**:
recall@$k$ against human-marked omissions is systematically won by ignoring the candidate, and omission
precision systematically penalises exactly the systems that do so.

### The declared risk did NOT materialise

B1's registered reporting rule was frozen against the possibility that MiniCheck or AlignScore would beat the
pipeline of record. On the real summary at Gemma+ent, **both score below MARS-C on both endpoints**:
MiniCheck 0.3574 / 0.7093 and AlignScore 0.3540 / 0.7105 against MARS-C's 0.3765 / 0.7202. SummaC-ZS over
distilled facts remains the only off-the-shelf arm ahead on recall (0.3854 against 0.3765) — and §3.5 and §7
show it pays for that on precision at every coverage level and on candidate conditioning.

MARS-C's own arms sit at the top of the precision column among comparable-recall systems:
`humanfact+div1` 0.3765 / 0.7202 and `humanfact.distilled+div1` 0.3691 / **0.7517**.

### Registered-rule deviation, declared

B1's registration said "**all** completed verifier rows enter the **main** table". With 14 families × 3
premise modes that is 46 rows and cannot fit a nine-page main text. The rule is honoured as: the real-summary
row of every family, ordered by recall@10 with omission precision beside it, goes into the paper; the full
46-row roster including every control goes to the appendix. Nothing is dropped and nothing is selected on its
value — this is a placement change, recorded here rather than made silently.

---

## 9. Independent reproduction of the paper's central identification number (B2, sealed TEST)

While B2 was scoring the sealed TEST split it recomputed the two checkpoint families' same-fact crossed
accuracy — the paper's headline identification statistic — through a **separately written implementation**
(`b2_crossed_families.py`) against the published one (`mc_summary_conditioning.py`,
`r03_sumblind/crossed_test/summary_conditioning.json`). Neither shares code with the other beyond `common.py`.

| cell | published (`mc_summary_conditioning.py`) | B2 (`b2_crossed_families.py`) | Δ |
|---|---|---|---|
| human-fact, triple-micro | 0.916275 | 0.9161 | 1.8e−4 |
| **human-fact, document-macro** | **0.9139505** | **0.9140** [0.8982, 0.9287] | **5e−5** |
| judge-label, triple-micro | 0.899125 | 0.8987 | 4.3e−4 |
| **judge-label, document-macro** | **0.8935811** | **0.8931** [0.8745, 0.9094] | **4.8e−4** |
| judge-label seed 20260906 | 0.894225 | 0.8942 | 2.5e−5 |
| judge-label seed 20260907 | 0.891875 | 0.8916 | 2.8e−4 |
| judge-label seed 20260908 | 0.897375 | 0.8974 | 2.5e−5 |

Agreement is ≤5e−4 everywhere, and the residual is accounted for: B2 restricts to the pair set common to all
seven verifier families (11,554 cells) while R03 uses the full crossed set.

**Why this matters.** `0.914 [0.898, 0.929]` is the number the abstract uses to answer "does the verifier read
the candidate, or just predict which facts are usually omitted?" — the paper's single strongest identification
claim. This manuscript has already had two estimator defects found in it (the pseudo-seed glob and the
rank-discarded human sample), so an independent reimplementation landing on the same value to four decimal
places is exactly the credibility evidence the paper needs, and it costs nothing extra to report.

---

## 10. C2 / H-SURF **PASSES** — and explains *why* the label source matters

`c2c4/c2c_crossed/c2_crossed_residual.json`. Bars declared before any number was read
(`declared_before_any_number_in_this_file_was_read: true`). Surface model (unit position in source, unit token
length, fact–candidate lexical Jaccard) fitted by OLS on **validation** cells per checkpoint seed and applied
**unchanged** to the sealed TEST cells — no moment of the evaluation split enters the model.

**A design property that makes this sharper than a generic surface control.** Inside a same-fact crossed
triple the fact is held fixed, so `rel_start` and `tok_len` are *identical* in the two cells and cannot move
the win indicator at all. Residualisation therefore removes exactly one channel: **lexical overlap between
the fact and the candidate summary** — the only surface channel that could fake conditioning.

| | raw | residualised | change |
|---|---|---|---|
| **Sealed TEST** (241 docs, 20,000 triples, 1,160 facts) | | | |
| MARS-C human-fact | 0.9110 | **0.8810** | −0.030 |
| MARS-C judge-label | 0.8883 | **0.6671** | **−0.221** |
| margin (human − judge) | +0.0204 [LB95 +0.0109] | **+0.2027 [LB95 +0.1758]** | **10×** |
| per-document wins A/B/tied | 95 / 54 / 92 | **196 / 18 / 27** | |
| **Validation** (138 docs, 9,523 triples, 569 facts) | | | |
| MARS-C human-fact | 0.9124 | **0.8752** | −0.037 |
| MARS-C judge-label | 0.8860 | **0.6359** | **−0.250** |
| margin | +0.0302 [LB95 +0.0150] | **+0.2223 [LB95 +0.1893]** | 7.4× |

**Verdict: `PASS`** on both splits, `families_not_cleared: []`, `candidate_blind_after_residualisation: []`.

### What this means, and why it is the campaign's most useful finding for the paper's thesis

**The judge-label verifier's candidate conditioning is largely lexical.** Remove the fact–candidate overlap
channel and it falls from 0.888 to 0.667 — two thirds of the way from its raw score to the 0.500 floor at
which a scorer is not reading the candidate at all. The human-fact verifier barely moves: 0.911 → 0.881.

So natural human coverage supervision teaches a verifier to judge coverage **beyond lexical overlap**, while
LLM-judge supervision largely teaches it to measure lexical overlap. That is a *mechanistic* answer to "what
does the label source actually buy?", and it is far stronger than the +0.023 recall and +0.0128 AUROC the
paper currently reports for label source in isolation — those are the visible residue of a much larger
difference in *what the two verifiers learned to use*.

Note the direction is not predictable from the fitted coefficients alone: the human-fact arm actually carries
the *larger* standardised lexical coefficient (−0.139 against the judge arm's −0.101). It leans on lexical
overlap at least as much — it simply has substantially more non-lexical signal underneath it.

### Consistency check

This job independently reproduces the published TEST crossed geometry: 241 documents with triples, 20,000
triples, 1,160 facts — exactly the figures in the manuscript's R03a sentence.

---

## 11. B4 — the literal unseen-extractor confirmation **transfers intact**

`b4/ordering_attenuation.json`, `b4/eval_test/e2_emitted.json`. Sealed TEST, 2,688 pairs / 285 documents.
`Qwen/Qwen3.8-27B` revision `1d4bf0f2…`, prompt sha256 and decoding contract frozen **before** execution; none
of its units, scores or statistics informed training, mining, calibration or model selection, and the
development threshold and emission rule were applied unchanged.

### Levels

| system | recall@10 | omission precision |
|---|---|---|
| dev · Gemma+ent · human-fact | 0.3765 | 0.7202 |
| dev · Gemma+ent · judge-label | 0.2548 | 0.6917 |
| **unseen · Qwen · human-fact** | **0.3848** | **0.7252** |
| **unseen · Qwen · judge-label** | 0.2698 | 0.7078 |
| dev · Gemma+ent · human-fact, *no summary* | 0.4585 | 0.6555 |
| unseen · Qwen · human-fact, *no summary* | 0.4573 | 0.6710 |
| dev · Gemma+ent · human-fact, *wrong summary* | 0.4395 | 0.6656 |
| unseen · Qwen · human-fact, *wrong summary* | 0.4401 | 0.6661 |

The absolute level is *higher* on the unseen extractor (0.3848 against 0.3765), so the headline is not an
artefact of having tuned to Gemma's extraction style.

### The supervision effect reproduces

| effect | development (Gemma+ent) | unseen (Qwen) | difference-in-differences |
|---|---|---|---|
| human − judge, recall@10 | **+0.1217** | **+0.1150** | −0.0067 [−0.0208, +0.0080] |
| human − judge, omission precision | +0.0285 | +0.0174 | −0.0111 [−0.0254, +0.0028] |
| real − no-summary, recall@10 (gameability) | +0.0820 | +0.0725 | +0.0094 [−0.0177, +0.0364] |
| real − wrong-summary, recall@10 | +0.0630 | +0.0553 | +0.0077 [−0.0092, +0.0263] |

**Every difference-in-differences spans zero.** The paper's headline +0.122 fixed-inventory verifier effect
reproduces at +0.115 on an extractor the system has never seen, with the change indistinguishable from zero;
so do both summary-blind control gaps.

### A registration-phrasing correction, recorded rather than quietly reinterpreted

PREREG B4 states the bar as "the MARS-C ordering on the development extractors reproduces on the unseen one,
with a document-clustered interval **excluding zero**." That phrasing was written for a *level* comparison and
is wrong for the test that actually answers the question. The question is *attenuation* — does the effect
**change** between the development and the unseen extractor? — and the estimator for that is the
difference-in-differences, where an interval that **includes** zero is what reproduction looks like. The job
implemented the DID and flagged the inversion in its own log (`here an interval that INCLUDES zero is what
ordering reproduction looks like`) rather than silently reinterpreting the registered bar, and
`ordering_attenuation.json` carries `direction: "either"` — it reports the contrasts and leaves the reading
explicit. On the correct test the transfer claim holds; on the literal text of my bar it would be judged
"direction not met", which is an artefact of my wording, not of the data.

### Extraction quality, for the record

7,342 held-out sentences → 21,742 propositions, 2.96 per sentence, blank decode 0.4%, empty parse **7.0%**
against the development extractor's **10.43%** on the very same sentences (the frozen prompt instructs `NONE`
for a factless sentence, so a nonzero empty rate is correct behaviour). Inventory: 110.3 units/pair with a
human-omitted ACU ceiling of 0.906 weak / 0.291 strict, against Gemma+ent's 106.2 / 0.897 / 0.289 — equivalent
inventory quality, which is the precondition that makes the transfer comparison interpretable.

## D2 — registered guard corrected (declared deviation, 2026-09-18)

Job 5358 FAILED after 21 s. Not a crash and not a model problem: the registered `<think>` guard refused its
own correct input. D2.1 had already completed and is untouched (267 pairs / 136 docs, 1,602 arm rows, six
arms, budgets 60-176 words, status FROZEN).

**What the guard did.** It rendered a probe prompt with `enable_thinking=False` and refused if the string
`<think>` appeared anywhere. Inspected directly on the frozen snapshot
(`models--Qwen--Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`):

| kwargs | rendered tail | openers / closers |
|---|---|---|
| `enable_thinking=False` | `<think>\n\n</think>\n\n` | 1 / 1 - block **closed** |
| `enable_thinking=True` | `<think>\n` | 1 / 0 - block **open** |
| *(default)* | `<think>\n` | 1 / 0 - block **open** |

Qwen3.5 *disables* thinking by emitting an already-closed empty block. The opener is present in both the safe
and the dangerous rendering, so a substring test on `<think>` rejects the correct prompt and would accept
nothing at all. The guard as registered could never pass on this model.

**The correction.** Refuse iff the rendered prompt leaves a think block OPEN - the last opener not followed by
a closer. `closed` and `absent` pass, `open` refuses. This preserves exactly the hazard the registration names
(generating *inside* an unterminated thinking block, the failure mode that collapsed forced-choice mass to
1e-7 and returned 88% empty outputs in an earlier campaign) and drops only the part that was a string-matching
error. Verified on all three renderings above before resubmission.

**Nothing was weakened.** The independent behavioural backstop is unchanged: the job still refuses if more
than half of the FIRST batch returns empty, which is what would actually catch a thinking-block failure in
outputs rather than in a probe. The observed guard state is now recorded in the manifest
(`decoding.think_guard`) instead of a hardcoded `"passed"`, so the rendering a run actually saw is auditable
rather than asserted.

**Scope.** D2 is a downstream revision-utility arm. **No claim in the paper depends on it**, and no result
recorded above changes.

**Outcome (job 5465, COMPLETED 0:0, 10m51s).** The correction is confirmed by the outputs, not merely by the
probe:

| check | result |
|---|---|
| `decoding.think_guard` (observed, not asserted) | `closed` |
| `model_revision` vs `expect_revision` | match, `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| **empty revisions** | **0 / 1,602 (0.00%)** - the failure mode this guard exists for gave **88%** |
| revisions containing `<think>` | 0 |
| rows per arm | 267 each x 6 arms = 1,602, balanced by construction |

1,602 revisions in 631 s, then D2.3 exported the blinded packets: 1,602 items over 267 pairs / 136 documents,
plus a 360-item pilot, in the `R#####` namespace with arm identity held in a separate key file. Per-arm word
means 57.3-61.5 with 292 over budget (18.2%), kept and flagged rather than truncated, as registered.

The zero empty rate is the substantive check. A guard that passes its own probe proves only that the probe
rendered; 0/1,602 empty against a historical 88% is evidence the thinking-block failure is actually absent
from the generations. **No outcome is computed here**: every D2 outcome needs two blinded reviewers plus
adjudication, and the packets are staged for exactly that.

## B0 gate — the gameability headline reproduces on an independent family set

Jobs 5339 / 5340 / 5342, all COMPLETED. This is the B0 control grid that had never been run in earlier
campaigns. It carries its own registration (`B0 controls, B2 secondary estimands`,
`declared_before_any_number_in_this_file_was_read: true`), 2,000-draw document-clustered bootstrap, seed
20260918, over 2,688 pairs / 285 documents on TEST and 1,388 / 151 on validation.

**It is not the same family set as B1.** B0 scores the two MARS-C verifiers (`humanfact`, `judgefact`) and
omits MiniCheck and AlignScore; B1 does the reverse. Five families are shared. So B0 is an independent
7-family x 2-inventory replication of the B1 headline, not a re-print of it.

Counting by B1's registered rule -- a summary-blind control *wins* if shuffled **or** empty beats real:

| cell | control wins recall@10 | real wins om. precision (both controls) | exception |
|---|---|---|---|
| validation, distilled | 6/7 | 7/7 | `nli_nli` (0.3944 vs 0.3880 / 0.3867) |
| TEST, distilled | **7/7** | **7/7** | - |
| TEST, Gemma+ent | **6/7** | **7/7** | `nli_nli` (0.3593 vs 0.3533 / 0.3393) |

**TEST totals: 13/14 on recall, 14/14 on precision** -- reproducing the paper's numbers exactly, with the
same single exception B1 names (windowed NLI on Gemma+ent). The pattern also holds on validation, so it is
not a split artifact.

MARS-C's own row is reproduced to the digit: `humanfact.gemma+ent` reads **0.377 real / 0.440 shuffled /
0.458 empty**, the triple printed in the main text.

Every family's shuffled recall gap that is negative has a document-clustered 95% interval excluding zero, and
every one of the 21 precision gaps is positive with an interval excluding zero. The two NLI families are the
only ones whose recall the shuffled control does not win (`nli_nli` +0.006 and `nli_summac_zs` +0.019 on
Gemma+ent, intervals touching or spanning zero) -- and `nli_summac_zs` still loses to the *empty* control, so
only `nli_nli` survives both.

**A counting trap, recorded because it was nearly published.** Counting the shuffled control alone gives
5/7 per cell (10/14) and looks like it contradicts the paper. The registered rule is *either* control, under
which the same files give 13/14. The rule must be applied as registered before any discrepancy is believed.

**No paper change.** This confirms the existing claims on an independent family set rather than altering them.

## B3 — matched-label transfer: the RoSE numbers reproduce exactly, the label-only effect does NOT transfer

Job 5253 COMPLETED 0:0 in 15m56s, after the 15-task scoring array. All 8 crosschecks `ok`, `missing: []`,
2,000-draw document-clustered bootstrap, seed 20260918, 2,688 pairs / 285 docs on RoSE TEST and 1,813 / 222 on
UniSumEval. The artifact carries its own guard against the estimator error this campaign already made once:

> `observed` is the difference of the two point estimates; `bootstrap_mean` is the mean of the resampled
> differences. They are not the same quantity and the manuscript must quote `observed`.

### Three exact confirmations of published numbers

| quantity | B3 (sealed TEST) | in the paper | |
|---|---|---|---|
| package effect, human vs deployed judge-label | **+0.1217 [+0.0992,+0.1440]** | +0.122 [+0.100,+0.144] | exact |
| label-only effect (A2), matched relabel | **+0.0231 [+0.0058,+0.0416]** | +0.023 [+0.006,+0.041] | exact |
| `deployed.humanfact` vs the B3 rebuild | **0.0000 [0.0000,0.0000]** on both metrics | - | bit-level identity |

The third is the one worth keeping: the B3 arm reproduces the deployed system exactly, so the two above are a
genuine re-derivation and not a re-read of the same cached emission.

### NEW — the label-only effect is RoSE-specific

| pool / regime | human vs judge, recall | verdict |
|---|---|---|
| RoSE TEST, regime A | +0.0231 [+0.0058,+0.0416] | **excludes zero** |
| RoSE TEST, regime R | +0.0514 [+0.0319,+0.0719] | **excludes zero** |
| UniSumEval, regime A | **-0.0024 [-0.0153,+0.0106]** | **spans zero** |
| UniSumEval, regime R | **+0.0079 [-0.0064,+0.0220]** | **spans zero** |

Null in both regimes on the untouched pool. The **package** effect still transfers there
(+0.0487 [+0.0309,+0.0680] against the deployed judge-label system), so what fails to generalise is
specifically the *label-source* contrast, not the pipeline. The paper claims package transfer and states the
label-only effect on the held-out split, so nothing published is contradicted -- but the abstract states the
label-only number without a pool qualifier, and that needs scoping.

### NEW — on UniSumEval the recall gameability does not reproduce for the human-fact system

| UniSumEval control | human-fact | judge-label |
|---|---|---|
| vs empty, recall | +0.0062 [-0.0156,+0.0260] **spans zero** | -0.0334 [-0.0552,-0.0115] excludes |
| vs shuffled, recall | -0.0092 [-0.0253,+0.0077] **spans zero** | -0.0199 [-0.0368,-0.0032] excludes |
| vs either, precision | +0.052 / +0.049 **excludes zero** | +0.040 / +0.048 excludes zero |

On RoSE TEST every one of these excludes zero (-0.082 / -0.063 recall, +0.065 / +0.055 precision). So the
recall-gameability headline is a RoSE-TEST result that does not reproduce on the external pool for our own
system, while the **precision penalty holds everywhere, without exception, in all four cells**. The paper
states the 13/14 result on the sealed RoSE split and claims nothing about UniSumEval, so again nothing
published is wrong -- but a "field-level methodological result" framing invites exactly this check, and the
honest boundary should be in the appendix rather than found by a reviewer.

### One estimator boundary, flagged not fixed

B3 puts the A2 aligned-precision shift at -0.0160 **[-0.0326,-0.0004]**, which *excludes* zero; the paper
reports -0.016 [-0.033,+0.000] and says it "is not separated from zero". The point estimates agree to four
decimals; only the interval's upper end moves, by 4e-4, under a different bootstrap seed. A conclusion that
flips on the fourth decimal with the resampling seed is not separated from zero in any useful sense, so the
paper's conservative wording stands and is **not** being changed on the strength of this run.
