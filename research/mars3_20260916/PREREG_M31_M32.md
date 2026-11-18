# Pre-registration — M31 (MARS-3 tagger and writer) and M32 (unit inventory), 2026-09-16 16:40 MSK

Registered before any run. Data, labels, splits, counters and the analysis code are those of `research/mars2_gates_20260916`
(B1 labels, human ACUs, extended counter family, `analyze.py`). Bars per `PLAN.md`; failure interpretations fixed here.

## M31 arms (3 seeds each unless noted; `code/m31_tagger.py`, `code/m31_writer.py`)
- `tagger`          ModernBERT-large, one pass over [candidate ; source], unit = mean token logit, BCE on B1 labels + 0.5 token aux; 2 epochs, lr 3e-5, batch 8, max 3,072 tokens.
- `tagger_pre`      as `tagger`, preceded by one epoch on label-free extractive pseudo rows from ALL train sources (B20 builder, 4 per source).
- `tagger_srconly`  candidate removed (summary-blind control).
- `writer`          LongT5-base writes the omitted units (" | "-joined); items matched to units / ACUs by token F1 (soft = max F1; binary ≥ 0.5).
- LODO: `tagger[xdom-D]`, `tagger_pre[xdom-D]` for D ∈ {rose_cnndm, rose_xsum, rose_samsum, natural770}.
- Label curve: `tagger[frac{0.1,0.25,0.5}]`, `tagger_pre[frac…]` (document-level subsample, same documents across arms).
- Pipeline baselines on the same inventories: `xenc[<inventory>]` (the three saved in-domain cross-encoders, `code/m31_score_pipeline_all.py`).

## Endpoints
1. m2 set (B1 labels, 362 pairs): macro / pooled AUROC, P@k, extended-counter certification (analyze.py m2).
2. Human ACUs (1,412 pairs): inventory-free scoring by aligned source tokens (tagger) or generated items (writer);
   pipeline systems enter only through their inventory (`pipeline-aligned:*`, undetectable ACU = 0). Endpoints: pooled /
   macro AUROC, **R@5 / R@10 of human-omitted ACUs** (primary for the inventory-free claim), paraphrased-only AUROC, certification.
3. Cost: measured seconds per pair (tagger, writer, per-unit cross-encoder) on one H200.

## Bars
- C1 (tagger vs per-unit verifier): m2 macro AUROC ≥ cross-encoder − 0.01 (i.e. ≥ 0.974) AND human-ACU pooled AUROC ≥ 0.94;
  cost ≤ 1/5 of the cross-encoder per pair. Partial: macro ≥ 0.95 with R@10 on human-omitted ACUs above the best
  pipeline-aligned system by a paired CI excluding 0 → the inventory-free claim stands, the parity claim does not.
- C3 (transfer): LODO held-out cell BH+ over the extended counter family on ≥ 3/4 resources.
- C4 (recovery pre-training): `tagger_pre` − `tagger` ≥ +0.02 macro at 10 % and 25 % labels (paired CI excluding 0).
- Control: `tagger_srconly` pooled AUROC on m2 ≤ counter max + 0.02 (otherwise a summary-blind shortcut exists in the data).
- Writer: R@10 of human-omitted ACUs ≥ the open-ended judges (0.20) by ≥ 0.15 and binary P@5 ≥ 0.60; otherwise it is an appendix.
- M32: inventory ceiling (weak rule) for human-omitted ACUs ≥ 0.85 for `gemma` or `gemma+ent` (spaCy: 0.45); distilled ≥ 0.80
  with ≥ 0.85 P/R agreement with Gemma; the cross-encoder on the new inventory must raise R@10 on human-omitted ACUs by ≥ 0.15 over spaCy.

## Failure interpretations
C1 fails and partial fails → MARS-3 is not a contribution; the paper's method is the audited pipeline with the M32 inventory.
C1 partial only → the contribution is inventory-free recall at lower cost, verification stays with the cross-encoder (two-stage).
C4 fails → recovery pre-training is dropped; "MARS" survives as the task framing and the audit, not as a training idea.
M32 fails (ceiling < 0.85) → LLM decomposition does not fix the inventory; report the ceiling as a finding about ACU-based evaluation.

Amendments (dated) below.

**Amendment 2026-09-16 15:25 UTC — M32b fact-level units (registered before running).** The writer trained on spaCy units
reproduces those fragments (11.7 items/pair such as "Europe", "their driving licence number") and matches human ACUs at
0.57 pooled AUROC: the target inventory, not the model, was wrong. M32b builds fact-level training rows: the LLM inventory
(`gemma+ent` or `gemma`, whichever validation ceiling ≥ 0.85 is higher; the job exits if neither does) of the SAME 2,625
B1 training pairs, labelled by the three in-domain cross-encoders as teacher (label = p_omitted < 0.5). Arms
`tagger_facts` (proposition tokens = the sentence's tokens whose word is a content token of the proposition) and
`writer_facts` (targets = omitted propositions), 3 seeds each, evaluated exactly like the other arms.
Bars: `tagger_facts` human-ACU R@10 ≥ `tagger` + 0.05 and pooled AUROC ≥ `tagger`; `writer_facts` R@10 of human-omitted ACUs
≥ open-ended judges + 0.15 (the original writer bar, now applied to this arm; the spaCy-target writer is reported as the
ablation that motivated it). Validation and test labels are never read by M32b.
Also recorded: B21b PASSED — the cross-encoder transfers on 4/4 held-out resources (0.965–0.987, BH+; SAMSum 0.965 vs
MARS-2's 0.882), and MARS-2's in-domain increments survive the extended counter family (+0.04…+0.08 nats, 4/4 BH+).

**Outcome 2026-09-16 13:50 UTC (validation).** C1 extracted units: tagger macro 0.979 / pooled 0.982 / paraphrased 0.942 vs
cross-encoder 0.984 / 0.985 / 0.949 (paired −0.004 [−0.008, −0.001]) → parity bar (≥ 0.974) MET; cost 0.013 s/pair vs
0.10–0.17 for the per-unit cross-encoder → cost bar MET. C1 human ACUs: 0.798 pooled (bar 0.94) → NOT MET; recovery
pre-training 0.795; all folds 0.789–0.796; label curve 0.733 → 0.798. C3: held-out folds 0.971–0.978 macro, BH+ over the
extended counters on 4/4 → MET. C4: pre-training +0.013 at 10 % and +0.003 at 25 % labels → NOT MET (bar +0.02 at both).
Control: source-only tagger 0.892 macro / 0.907 pooled on the judge labels, above counter max + 0.02 (0.878) → a learned
summary-blind prior exists in the B1 labels; the tagger's margin over it is +0.087 and is the part of its score that is
about the summary. Writer (spaCy targets): 0.77 on extracted units, 0.58 on ACUs → appendix ablation. Inventory-free
partial claim: NOT MET (tagger R@10 on human-omitted ACUs 0.810 vs token recall 0.847 and the cross-encoder 0.873 on
supplied ACUs; pipeline-aligned systems pending job 4712).
