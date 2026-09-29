# MARS-3 program — making the omission scorer the main contribution (2026-09-16)

**Problem.** Target-free omission detection: given (source, candidate summary), say what the summary leaves out,
at the level of content units, without a reference. Open-ended LLM judges fail at it (0.51 AUROC, two families).
**Thesis.** A single-pass, candidate-conditioned *omission tagger* over the source, pre-trained with the recovery
objective and fine-tuned on distilled coverage labels, localises omissions as well as a per-unit entailment verifier
at a fraction of its cost, yields the first coverage metric that survives the counter audit, and transfers across domains.

## Where MARS-2 stands (validation, 2026-09-16)
| Fact | Number | Consequence |
|---|---|---|
| MARS-2 q-only vs counters (extended family) | +0.03…+0.08 nats, BH+ 4/4 cells | signal is real |
| MARS-2 vs DeBERTa-large cross-encoder, same labels, same units | 0.945 vs 0.984 macro; human ACUs 0.880 vs 0.951 | architecture is not the best verifier |
| recovery objective vs likelihood feature | direct_enabled ≥ full_recovery everywhere | the decoder's training objective adds nothing |
| both systems rank the same spaCy units | — | "target-free" is the shared extractor, not the model |
| label-free domain adaptation | control ≥ target domain | extra synthetic training helps generically; no domain effect |
| paraphrased units | MARS-2 0.816 vs cross-encoder 0.949 | the loss is semantic matching, not context |

So the contribution cannot be "a better per-unit verifier". It has to be one of: a different formulation (one pass over the
source instead of 40 verifier passes), a better unit inventory, a coverage *metric* that passes the audit, or content the
verifier cannot produce (what is missing, in words). The program below tests all four, gated, in order of leverage.

## Claim map
| Claim | Minimum convincing evidence | Blocks |
|---|---|---|
| C1 One-pass omission tagging matches per-unit verification | m2 macro AUROC ≥ cross-encoder − 0.01; human ACUs ≥ 0.94; ≤ 1/10 verifier cost per pair | M31, M30-D3 |
| C2 The first coverage metric that survives the counter audit | summary-level score adds information over length/compression/lexical-recall counters on ≥ 3/4 human-labelled cells where A3CU / QuestEval / FineSurE / judges do not (A2 showed ≤ 1–4/10) | M33, M30-D4 |
| C3 Transfer without target-domain labels | LODO ≥ 3/4 resources BH+ (registered family) | M31-LODO |
| C4 Recovery pre-training is what makes it work with few labels | label-efficiency curve: recovery-pretrained tagger ≥ scratch tagger at 10 % / 25 % of labels by ≥ 0.02 | M31-R |
| C5 Useful to people | omission highlights judged correct (P@5 ≥ 0.80 vs human ACUs); best-of-N coverage gain at matched length | M34, B22 |
| Anti-claims to rule out | summary-blind shortcut (source-only tagger must fail: W16 trap); inventory ceiling (M30-D1); cross-encoder wins on cost-matched budget | M31 controls |

## Blocks

### M30 Stage 0 — diagnostics on existing artifacts (CPU, today; `code/stage0_diagnostics.py`)
- D1 Inventory ceiling: fraction of human-omitted ACUs that have an aligned extracted unit (validation + test, per resource).
  If < 0.85 the extractor caps every extractor-based system and M32 becomes MUST.
- D2 Loss anatomy: MARS-2 vs cross-encoder AUROC by token-recall stratum, unit kind, unit length, resource; disagreement cases.
- D3 Cost: measured seconds per pair for the cross-encoder (n_units passes) and MARS-2; projected one-pass cost.
- D4 Coverage-metric pilot: summary-level mean(1 − omission) vs human ACU recall (RoSE) and human entity coverage (770);
  ΔR² of ridge(counters + score) over ridge(counters), document-grouped bootstrap. Counters: candidate words/chars, unit count,
  compression, mean lexical recall.

### M31 Stage 1 — MARS-3 tagger (GPU, ~2 GPU-days)
Model: long-context encoder (ModernBERT-large 8k, cached; LongT5 encoder as the second backbone) over
`[candidate] [SEP] [source]`; per-token omission logit; unit score = mean logit over the unit span; BCE on unit labels
(the B1 judge labels, same as B2/B21); optional token-level auxiliary from projected labels. One forward pass per pair.
Arms (3 seeds each): (a) tagger; (b) tagger + recovery pre-training on extractive pseudo rows from ALL train sources
(label-free, `b20_build_pseudo.py`, then fine-tune); (c) source-only tagger (control: must fail, summary-blind trap);
(d) tagger top-K → cross-encoder verify (amortised, K = 8); (e) cross-encoder with a cost-matched budget (baseline).
Endpoints: m2 units (judge labels), human ACUs via span pooling over the best-matching sentence, extended B4
certification, LODO (4 folds, arm a and b), label-efficiency curve at 10/25/50/100 % of labels (arms a vs b), cost.
Bars: C1, C3, C4 as above; (c) must read ≤ counter max.

### M32 Stage 2 — unit inventory (conditional on D1 < 0.85)
Distilled proposition segmenter: Gemma-4-31B decomposes source sentences into atomic propositions on the train split;
a small seq2seq (LongT5-base) is distilled; recall of human ACUs vs the spaCy inventory; plug into M31 and the
cross-encoder pipeline; R@5 and P@5 on human ACUs. Bar: human-omitted ACU recall ceiling ≥ 0.90.

### M33 Stage 3 — the audited coverage metric (C2)
Summary-level score = 1 − mean omission over units (uniform; salience-weighted as an ablation). Cells with human
coverage labels: RoSE CNN/DM, XSum, SAMSum (ACU recall), the 770 resource (entity coverage), plus UniSumEval
completeness / SummEval relevance if accessible. Reference = counters + TF-IDF partial input (A1 machinery); comparison
metrics = A3CU, QuestEval-recall, FineSurE completeness, Gemma judge (A2 workers). Bar: BH+ on ≥ 3/4 cells; the
comparison metrics reproduce A2's ≤ 1–4/10 pattern on the same cells.

### M34 Stage 4 — humans and utility
Omission highlights (top-5 tagged spans) judged by two annotators on 300 sources (the B011 sample, re-purposed);
best-of-N at matched length on TEST (B22). Needs the user's annotators.

### M35 Stage 5 — sealed TEST, one shot (B19 protocol, frozen systems).

## Run order and gates
| Milestone | Runs | Gate | Cost |
|---|---|---|---|
| M30 | stage0 (CPU) + cost timing | D1 decides M32; D4 decides whether C2 is plausible before building | 1 CPU-h + 10 GPU-min |
| M31 | 5 arms × 3 seeds + 8 LODO + label curve (8) ≈ 40 runs × 15–30 min | C1 met on validation → proceed; C1 missed by > 0.02 → the paper's method is the audited pipeline (extract → verify) and M31 becomes an efficiency appendix | ~2 GPU-days |
| M32 | segmenter distillation (Gemma 20 GPU-h) + LongT5 training | only if D1 < 0.85 | 1 GPU-day |
| M33 | CPU analyses + A2 workers on new cells | C2 | 1 CPU-day + 6 GPU-h |
| M34 | annotation + B22 | C5 | annotators |
| M35 | TEST | — | 2 GPU-h |

Timeline: M30 today; M31 code 2 days, runs 1 day; M33 in parallel (mostly CPU); M32 the following week if needed;
draft with M31–M33 by mid-October; ARR December 2026 or ICML 2027 (dates to verify), TACL as fallback.

## What stays from MARS-2
The name and the recovery idea (as pre-training, C4), the B1 labels and splits, the B3/B4/B18/B21 evaluation machinery,
the finding that open-ended judges are omission-blind, and the mechanism result (likelihood feature, salience) as
motivation. The per-unit recovery scorer itself becomes an ablation row.

## Stage 0 results (2026-09-16 15:50 MSK, `~/results/mars3/stage0/stage0.json`) — and what they change

| Diagnostic | Result | Consequence |
|---|---|---|
| D1 inventory ceiling | only **0.45** of human-omitted ACUs have an aligned extracted unit (CNN/DM 0.44 / 0.40, SAMSum 0.67, XSum 0.30 / 0.40; validation / test) | the spaCy inventory, not the verifier, is the binding ceiling: every extractor-based system (MARS-2, cross-encoder) is blind to 55 % of what humans mark as missing. **M32 is MUST**, and the tagger's inventory-free evaluation is the headline endpoint |
| D2 loss anatomy | MARS-2 loses to the cross-encoder mostly on lexically absent units (recall < 0.25: 0.826 vs 0.953), on verbatim-present units it wrongly flags (recall = 1: 0.884 vs 0.986) and on propositions (0.909 vs 0.967); counters are useless exactly where the models matter (0.51 / 0.50 at the extremes) | the loss is semantic matching; the tagger must use a strong bidirectional encoder |
| D3 cost | 30.8 units/pair, 400 source words: per-unit verification ≈ 4,200 tokens/pair vs one pass ≈ 600 (×7; more for long sources) | C1's cost claim is ×7, not ×40 |
| D4 coverage pilot | summary-level mean coverage over extracted units correlates only 0.20–0.25 with human ACU recall and adds nothing over counters; with the oracle ACU inventory lexical recall alone reaches ρ 0.70 and the models add ΔR² ≤ +0.02 (n.s.) | **C2 (audited summary-level coverage metric) is not plausible with the current inventory and weak even with an oracle one → demoted to an appendix check after M32**; the contribution lives at the unit level (localisation), not in a summary score |

Revised order: **M32 (inventory) and M31 (inventory-free tagger) run first and in parallel**; both are evaluated on the
same human ACUs, where the pipeline's ceiling is 0.45 recall. Primary MARS-3 endpoints become recall-oriented:
R@5 / R@10 of human-omitted ACUs per pair and pooled AUROC at sentence granularity (ACU → best-matching sentence),
alongside the m2 endpoints. A generative "recovery writer" arm (LongT5 decoder produces the omitted content as text,
matched to ACUs by token F1) joins M31 as the second inventory-free formulation — it is the recovery idea in its
original sense, and it is exactly what open-ended judges fail at (0.51).
