# MARS-C: closing the distance to a main-contribution claim (registered 2026-09-16 17:30 UTC)

**Thesis.** A deployable omission finder — LLM fact inventory → verifier trained on natural human facts → one unit per source
sentence → top-k — recovers 0.49 of the facts human annotators marked omitted (validation, top-10), 2.6× the released MARS and
2× the best pre-existing judge-label pipeline; under the identical protocol it beats the strongest judge-label pipeline by
+0.047 [+0.007, +0.081]. What a main-contribution claim still lacks is (i) a TEST confirmation, (ii) evidence that the recall is
not bought with precision (76 % of emitted facts match no reference ACU), (iii) a cost account, (iv) an untouched confirmation
pool, and (v) prior-art positioning (not compute; see LITERATURE notes in the redesign report).

## Claim map
| Claim | Minimum convincing evidence | Block |
|---|---|---|
| C-A The pipeline finds human omissions far better than every pre-existing system, at the registered margin | TEST, one shot, frozen systems, +0.05 over the strongest judge-label pipeline under the identical rule with CI > 0 | G1 |
| C-B The emitted facts are mostly real omissions (recall is not bought with precision) | judge-estimated true-omission rate of emitted facts ≥ 0.5 and non-inferior to the strongest judge-label pipeline (two judge families, validated on the human-known units); blinded human sample prepared | G2, G6 |
| C-C The margin is robust / the frozen pipeline is the right one | inventory union and verifier ensemble tried BEFORE TEST; cap and k sensitivity; MARS-2 under the identical rule | G3 |
| C-D The gain is not paid for in cost | measured seconds per pair for every component; verifier gain is at identical cost; inventory cost amortised over systems per document | G4 |
| C-E The result transfers to an untouched pool | feasibility census of UniSumEval / QAPyramid (docs, key-fact labels, overlap with our sources, licence) | G5 |

## Blocks, bars, order
- **G3 — margin variants (CPU, first; may change the frozen pipeline).** Union inventory (gemma+ent ∪ distilled, human-fact
  verifier), verifier ensemble (mean of human-fact and judge-label probabilities) per inventory, cap 2 (report), k ∈ {5, 10, 20}
  (report), MARS-2 under the rule (fairness). Bars: union or ensemble ≥ frozen + 0.02 recall@10 with CI > 0 → promoted to the
  frozen pipeline for G1; otherwise the frozen pipeline stays. Decided before any TEST scoring.
- **G2 — judge-estimated precision of emitted facts (GPU).** Top-10 emitted units on validation of: frozen pipeline, distilled
  human-fact + rule, strongest judge-label pipeline (distilled + rule), MARS-3 tagger + rule, MARS-2 (released ranking). Two
  questions per unit — supported by the source? conveyed by the summary? — forced A/B choice, two families (Gemma-4-31B-it,
  Qwen3.8-27B). True omission = supported ∧ not conveyed. Endpoints: true-omission rate per emitter (pair-weighted), resolved
  share of the "unknown" units, judge validity = coverage accuracy on the human-known emitted units (hit / false alert),
  family agreement (κ). Bars: frozen pipeline rate ≥ 0.50 (primary family); non-inferiority — the CI of (strongest judge-label
  pipeline − frozen) must not exceed +0.03; validity ≥ 0.85 and κ ≥ 0.6, else the judge estimate is reported as unvalidated.
- **G4 — cost (GPU timing + receipts).** Seconds per pair on one H200 for: Gemma decomposition (per document, from the M32
  array), distilled segmenter, human-fact and judge-label verifiers per inventory, MARS-3 tagger, the rule. Per-pair inventory
  cost amortised over the systems per document in the resource. Report only; the verifier comparison is at identical cost.
- **G5 — untouched pool feasibility (CPU, network).** Clone UniSumEval-v1.0 and QAPyramid; census records, fields, key-fact
  coverage labels, licences; overlap with our source pool (sha1, 8-gram Jaccard ≥ 0.5). Output: go / no-go for a confirmation
  run; no scoring.
- **G6 — human confirmation sample (CPU).** 300 emitted units, blinded and shuffled: 100 each from the frozen pipeline, the
  strongest judge-label pipeline and MARS-2, stratified 60/20/20 unknown / hit / false alert; key kept separately; instructions
  file. Deliverable for two annotators (user action).
- **G1 — TEST, one shot (GPU; gated on the user's explicit `B2_SPLIT=test`).** Frozen systems: the pipeline of record after G3,
  distilled human-fact + rule, judge-label + rule over gemma+ent and distilled, unruled references. Human-fact verifier = E1b
  arm A, 3 seeds; judge-label verifier = B21 cross-encoder, 3 seeds. k ∈ {5, 10}, 500 document bootstraps, primary = pipeline
  of record. Bar: +0.05 over the strongest judge-label pipeline under the identical rule with CI > 0 (secondary: CI > 0). The
  MARS-3 tagger has no saved weights and is absent; MARS-2 TEST dumps do not exist (B19). Nothing changes after this record.

## Run order
G3 → (G2 judge array, G6 sample, G4 cost, G5 census in parallel) → G2 analysis → G1 on the explicit word.
Budget: G3 3 min CPU; G2 ≈ 6 GPU × 1.5 h; G4 20 min GPU; G5 10 min CPU; G1 ≈ 40 min GPU.
