# Pre-registration — MARS-C E0 / E1 (2026-09-16 18:10 UTC)

Source documents: `docs/FINAL_PROPOSAL_2026-09-16.md`, `docs/REPORT_2026-09-16.md` (external redesign report; census 4730 in
`docs/census_train_4730.json`). This file fixes what is implemented and what decides. E2 (deployable pipeline, emitted
facts, importance policy, untouched confirmation pool) is NOT part of this registration; it needs the frozen M32 inventory
and is registered separately when E1 is read.

## E0 — data gate (`code/mc_data.py`, CPU)
Human TRAIN ACU labels only (B1 `pairs_train.jsonl`, RoSE). Canonical fact = exact normalised ACU text within a source;
label conflicts within a summary drop that fact for that summary and are counted. Reference summaries (`gold`) are never
training candidates. Near-duplicate clusters: word 8-gram shingles, Jaccard ≥ 0.5, over TRAIN + VALIDATION + TEST sources;
a cluster touching a held-out split stays out of training. Remaining clusters: sha256("20260916|resource|cluster") →
80 % train / 10 % tune / 10 % calibration. Tetrad = two non-reference summaries with length ratio ≤ 1.25 and opposite
coverage of two facts. Collision census: opposing facts whose pooled source-token masks (ModernBERT tokenizer, 3,072
tokens, the M32b student representation) are identical, over ≤ 20 sampled tetrads per training document.
Gate: ≥ 200 training documents with tetrads, ≥ 3 resources, ≥ 50 per resource, collision rate ≤ 5 %. If collisions
exceed 5 % the tagger representation is closed for the tetrad headline and E1 runs with the registered alternative
(`--repr crossenc`, DeBERTa-large-MNLI on (summary, fact)); the one-pass claim is then abandoned. Collided tetrads are
removed from training in every arm; nothing is removed from evaluation.

## E1 — objective isolation (`code/mc_train.py`, 3 seeds, identical student and update count)
Arms: R random-cell BCE (natural cells, matched cell count; tests the sampler), A BCE with the tetrad sampler, B + same-fact
pairwise, C + generic row/column contrast, D + conditional coverage loss `0.25·softplus(−(r1+r2))`. λ = 1 primary;
λ ∈ {0.25} additional for B/C/D with equal allowance; λ = 0 is arm A. Steps 600, lr 3e-5 (tagger) / 2e-5 (crossenc),
warm-up 6 %, batch = 2 tetrads (4 encodings) + 4 natural summaries × ≤ 6 facts. Arms A–D consume the same seed-locked
sample stream, so they see identical cells in identical order. Calibration: one logistic slope/intercept on the
calibration documents at natural prevalence, applied identically to every arm.
Order: development screen first (1 seed, λ = 1, five arms, resource rose_cnndm — the largest crossed pool), then the full
matrix only if D shows headroom (lexical-stress AUROC ≥ the strongest control's on the screen; override `MC_FORCE=1`).
Endpoints on the existing VALIDATION ACUs (historically exposed, exploratory): pooled and document-macro AUROC, calibrated
log-loss, lexical-stress AUROC (ACU absent verbatim from the summary and content-token recall < 0.5, frozen rule), strict
four-cell success and sign(C) rate on validation tetrads, cost (encodings, tokens, seconds).
Screen bar: D − strongest equal-data control ≥ +0.02 lexical-stress AUROC with the document-paired 95 % CI above 0, and
overall pooled AUROC not lower by more than 0.005. If C (generic contrast) ties D, the benefit is attributed to balanced
training and the objective novelty is dropped. One seed rejects obvious failure; advancement needs all three seeds.
Mandatory system comparators, taken from the existing analyses: the in-domain cross-encoder (0.951 pooled on the same
ACUs), MARS-2 direct+feature (0.880), `tagger_facts` / `writer_facts` when they finish.

## Failure interpretations
E0 fails → the natural crossed-coverage hypothesis stops; no edited-summary substitute under the same claim.
E1 D fails → BCE on balanced batches is enough (or generic contrast ties); the conditional objective is not a contribution.
Sampler effect only (A ≫ R, D ≈ A) → report balanced natural sampling as the finding, not the loss.

Amendments below, dated.

**Amendment 2026-09-16 13:45 UTC — screen result and representation switch.** Development screen (rose_cnndm, one seed,
λ = 1, tagger student): pooled human-ACU AUROC A 0.819, D 0.814, C 0.810, B 0.809, R 0.797; lexical-stress A 0.729, D 0.727,
B 0.712, C 0.707, R 0.696; strict four-cell 0.50–0.56 for every arm. D − A = −0.002 [−0.037, +0.025] → no headroom; the
generic contrast ties D; the tetrad sampler beats random cells (A − R +0.022 pooled, +0.033 lexical-stress, CI [0.002, 0.061]).
Per the registration, on this representation the conditional objective is not a contribution and the sampler effect is the
finding. Independently, M31 showed that token-pooled scoring of supplied facts saturates near 0.80 for a tagger trained on
100 % of the judge labels while the cross-encoder reads 0.951 on the same ACUs: the tagger representation, not the
objective, bounds the screen. The registration names one alternative representation (explicit (summary, fact) cross-encoder);
its trigger was the collision gate, which passed. Amendment: run the identical five-arm screen with `--repr crossenc`
(DeBERTa-large-MNLI, same seed-locked cells, same bars) before closing E1; no further representation search after it.
The tagger full matrix (jobs 4788–4790) is cancelled as never-satisfiable behind the failed headroom gate.

**Amendment 2026-09-16 13:40 UTC — cross-encoder screen result; E1b registered.** Cross-encoder representation, same
screen (rose_cnndm training documents, one seed, λ = 1): pooled human-ACU AUROC R 0.969, A 0.969, B 0.970, C 0.971,
D 0.970; lexical-stress 0.961 / 0.955 / 0.959 / 0.962 / 0.960; strict four-cell 0.80–0.82; log-loss 0.226–0.231. The five
objectives are indistinguishable; the sampler effect seen on the tagger vanishes. E1 closes on both representations:
**the conditional coverage loss is not a contribution** (failure interpretation "BCE on balanced batches is enough").
What the screen does show is the report's higher-confidence engineering hypothesis: a verifier trained on complete
human-labelled facts (250 CNN/DM training documents, 600 updates) reads 0.970 on human ACUs where the same architecture
trained on 119k judge labels of spaCy units reads 0.951, and 0.96 vs 0.94 on the lexical-stress subset.
**E1b (registered now, before running):** arm A (plain BCE) and arm R (random cells) on the cross-encoder representation,
3 seeds, all three resources; plus arm A trained on two resources with the third held out (3 configurations × 3 seeds).
Bars: (1) pooled human-ACU AUROC of A ≥ 0.951 with the document-paired CI above the judge-label cross-encoder on the same
ACUs; (2) held-out resource pooled AUROC ≥ 0.94 on ≥ 2 of 3 resources; (3) R vs A within ±0.01 (sampling is irrelevant on
this representation) — if not, report the sampler. Outcome use: the human-fact-trained verifier becomes the pipeline's
verifier in E2; no method-novelty claim attaches to it. Budget: 15 runs × ~3 min.

**Amendment 2026-09-16 14:10 UTC — E2 pilot on validation (exploratory, registered before running).** The human-fact-trained
cross-encoder (E1 arm A, 3 seeds, all resources) scores the frozen `gemma+ent` inventory of every validation pair
(`--eval-units`); the pipeline is mapped onto the human ACUs exactly like the judge-label pipelines (undetectable ACU = 0).
Endpoint: R@10 of human-omitted ACUs per pair, next to the judge-label pipelines on the same inventories, the M31 taggers
and token recall; paired document bootstrap. Bar for going to E2 proper on TEST: pipeline-aligned human-fact verifier
R@10 ≥ the best judge-label pipeline + 0.05 with CI above 0. Gold ACUs never enter the pipeline; the inventory is frozen.

**Outcome 2026-09-16 14:55 UTC — E1b and the E2 pilot (validation, exploratory).**
E1b PASSED: human-fact verifier (arm A, 3 seeds, all resources) 0.974 pooled / 0.968 macro / 0.964 lexical-stress on human
ACUs; arm R 0.975 / 0.971 / 0.968 (bar 3: sampling irrelevant); held-out resource pooled 0.972 (CNN/DM) / 0.964 (SAMSum) /
0.983 (XSum) (bar 2, 3/3). The R@10-over-ACUs endpoint registered for the inventory-free claim is degenerate on RoSE
(≈ 7 ACUs per pair) and is replaced by the emitted-top-k protocol (`code/mc_e2_eval.py`): each system emits its top-k
inventory units, matched to human ACUs by sentence overlap + shared content token; recall of human-omitted ACUs, hit /
false-alert / unknown shares of the emitted units. E2 pilot (1,412 pairs, no importance policy, rank by omission confidence):

| emitter (inventory · verifier) | recall@5 | recall@10 | false alerts | unknown |
|---|---:|---:|---:|---:|
| gemma+ent · human-fact cross-encoder (E1b) | 0.187 | 0.280 | 0.037 | 0.856 |
| distilled · judge-label cross-encoder | 0.154 | 0.244 | 0.039 | 0.864 |
| gemma · judge-label cross-encoder | 0.135 | 0.217 | 0.050 | 0.839 |
| gemma+ent · judge-label cross-encoder | 0.131 | 0.214 | 0.032 | 0.885 |
| spaCy · judge-label cross-encoder | 0.129 | 0.203 | 0.030 | 0.897 |
| gemma+ent · fact-level tagger | 0.111 | 0.202 | 0.027 | 0.899 |
| spaCy · MARS-3 tagger | 0.117 | 0.194 | 0.027 | 0.903 |
| spaCy · MARS-2 | 0.110 | 0.189 | 0.032 | 0.902 |

Same inventory, verifier swapped: +0.066 recall@10 [+0.038, +0.091] for the human-fact verifier. Against the strongest
judge-label pipeline (distilled inventory): +0.035 [−0.009, +0.080] → the E2-pilot bar (+0.05, CI > 0) is NOT MET.
86–90 % of emitted units match no reference ACU (unknown), i.e. the ranking is dominated by facts outside the reference
universe: the proposal's frozen source-only importance policy is the missing component, not the verifier. E2 proper on
TEST is not opened. Next registered step (not run): an importance filter trained on TRAIN/development evidence only,
applied identically to every emitter, then the same protocol.

**Amendment 2026-09-16 15:20 UTC — E2 with the frozen importance policy (registered before running; `slurm/mc_e2imp.sbatch`).**
Importance = a source-only tagger (ModernBERT-large, `--source-only`, 3 seeds, 3 epochs) trained on TRAIN-role documents to
predict whether a gemma+ent inventory unit aligns to a human REFERENCE ACU of its document (reference-worthiness; never a
summary's omission label). The threshold τ is chosen on the TUNE-role documents by emitted recall@10 of human omissions for
the human-fact emitter over τ ∈ {0.2, 0.35, 0.5, 0.65, 0.8}, then applied identically to every emitter on validation.
Bar (unchanged): filtered human-fact emitter recall@10 ≥ the strongest judge-label pipeline + 0.05 with CI above 0; the
filter must also lower the unknown share of emitted units. Gold ACUs of validation never enter any emitter.

**Outcome 2026-09-16 14:55 UTC — E2 with the frozen importance policy (job 4835, validation, 1,412 pairs / 1,388 with omissions).**
Importance tagger: 853 TRAIN-role documents, 81,237 gemma+ent units of which 10.8 % align to a human reference ACU; 3 seeds,
161 s training each, 0.0006 s/pair to score. τ chosen on TUNE (202 pairs): recall@10 0.428 (τ 0.2) / 0.382 (0.35) / 0.369 (0.5)
/ 0.365 (0.65) / 0.354 (0.8 = no filter) → **τ = 0.2**. Validation, every emitter filtered identically:

| emitter (inventory · verifier) | recall@5 | recall@10 | hit | false alerts | unknown | Δ recall@10 vs primary [95 % CI] |
|---|---:|---:|---:|---:|---:|---:|
| gemma+ent · human-fact cross-encoder **+ importance** (primary) | 0.272 | **0.396** | 0.163 | 0.069 | 0.769 | — |
| gemma+ent · judge-label cross-encoder + importance | 0.225 | 0.360 | 0.150 | 0.066 | 0.784 | +0.035 [+0.019, +0.053] |
| gemma+ent · fact-level tagger + importance | 0.212 | 0.346 | 0.146 | 0.060 | 0.794 | +0.049 [+0.035, +0.066] |
| gemma+ent · human-fact cross-encoder, no filter (pilot) | 0.187 | 0.280 | 0.106 | 0.037 | 0.856 | +0.114 [+0.064, +0.160] |
| distilled · judge-label cross-encoder, no filter (pilot's strongest judge pipeline) | 0.154 | 0.244 | 0.097 | 0.039 | 0.864 | +0.149 [+0.100, +0.199] |
| gemma+ent · judge-label cross-encoder, no filter | 0.131 | 0.214 | 0.083 | 0.032 | 0.885 | +0.182 [+0.130, +0.227] |
| spaCy · MARS-3 tagger, no filter | 0.117 | 0.194 | 0.071 | 0.027 | 0.903 | +0.203 [+0.157, +0.250] |
| spaCy · MARS-2, no filter | 0.110 | 0.189 | 0.066 | 0.032 | 0.902 | +0.205 [+0.156, +0.254] |

Reading against the registered bar. (i) The filter lowers the unknown share (0.856 → 0.769) — met. (ii) Against the strongest
judge-label pipeline **as it existed when the bar was written** (distilled, unfiltered, 0.244): +0.149 [+0.100, +0.199] — met.
(iii) Against the strongest judge-label pipeline **under the identical policy** (gemma+ent judge-label + importance, 0.360):
+0.035 [+0.019, +0.053] — the CI is above 0 but the +0.05 margin is not reached (recall@5: +0.049 [+0.028, +0.072]).
Reading (iii) is the honest comparator because the registration promised the policy "applied identically to every emitter";
the distilled-inventory judge pipeline was not filtered in this job (no importance scores for that inventory yet), so (iii) is
provisional until the completeness run below. Component attribution on the same inventory and verifier: importance policy
+0.114 (the dominant component), human-fact verifier over judge-label verifier +0.035 (both CIs above 0). The filter raises the
false-alert share (0.037 → 0.069): emitted units that match a covered ACU. Decision: TEST is not opened on this result alone;
the bar under reading (iii) is decided by the completeness run. Gold ACUs of validation entered no emitter.

**Amendment 2026-09-16 15:10 UTC — completeness run (`slurm/mc_e2imp_all.sbatch`, registered before running).** Same policy,
τ fixed at 0.2 (the TUNE choice; not re-chosen), importance taggers retrained with the same rows / seeds / config and saved,
scored on all four validation inventories (gemma+ent, distilled, gemma, spaCy). Emitters filtered identically: human-fact
cross-encoder over gemma+ent (primary; replication check: recall@10 within ±0.02 of 0.396) and over distilled (new omission
scores from the saved E1b models); judge-label cross-encoder over gemma+ent / distilled / gemma / spaCy; fact-level tagger over
gemma+ent; MARS-3 tagger over spaCy; unfiltered references. Bar: primary recall@10 ≥ the strongest filtered judge-label pipeline
+ 0.05 with CI above 0 (reading iii). If the margin is not reached, the registered conclusion is: the deployable pipeline's
gain over every pre-existing pipeline comes from the reference-worthiness policy plus the natural human-fact verifier, with the
verifier's own contribution +0.03–0.05 (CI above 0) rather than ≥ +0.05; no method-novelty claim attaches to the verifier.

**Outcome 2026-09-16 15:30 UTC — completeness run (job 4836, validation).** Replication check met: human-fact + importance over
gemma+ent 0.413 vs 0.396 in 4835 (+0.017; retrained policy, same seeds → ±0.02 is the policy's seed noise). Every emitter under
the identical policy (τ = 0.2), recall@10 of human-omitted ACUs / unknown share / false-alert share:

| emitter (inventory · verifier) | recall@5 | recall@10 | unknown | false alerts | Δ vs human-fact+imp (gemma+ent) [95 % CI] |
|---|---:|---:|---:|---:|---:|
| distilled · human-fact + importance | 0.306 | **0.448** | 0.747 | 0.082 | −0.035 [−0.077, +0.007] |
| distilled · judge-label + importance | 0.287 | 0.441 | 0.750 | 0.081 | −0.026 [−0.070, +0.018] |
| gemma+ent · human-fact + importance (4835 primary) | 0.280 | 0.413 | 0.750 | 0.074 | — |
| gemma · judge-label + importance | 0.264 | 0.402 | 0.721 | 0.088 | +0.011 [−0.016, +0.046] |
| gemma+ent · judge-label + importance | 0.247 | 0.386 | 0.763 | 0.073 | +0.028 [+0.012, +0.048] |
| gemma+ent · fact-level tagger + importance | 0.231 | 0.378 | 0.771 | 0.066 | +0.035 [+0.020, +0.050] |
| spaCy · MARS-3 tagger + importance | 0.184 | 0.294 | 0.815 | 0.078 | +0.118 [+0.078, +0.154] |
| spaCy · judge-label + importance | 0.198 | 0.293 | 0.814 | 0.081 | +0.121 [+0.077, +0.159] |
| distilled · human-fact, no filter | 0.188 | 0.292 | 0.844 | 0.042 | +0.120 [+0.071, +0.179] |
| gemma+ent · human-fact, no filter | 0.187 | 0.280 | 0.856 | 0.037 | +0.132 [+0.080, +0.187] |
| distilled · judge-label, no filter | 0.154 | 0.244 | 0.864 | 0.039 | +0.170 [+0.111, +0.226] |
| spaCy · MARS-2, no filter | 0.110 | 0.189 | 0.902 | 0.032 | +0.226 [+0.179, +0.267] |

Bar under reading (iii) — the strongest judge-label pipeline under the identical policy is distilled · judge-label + importance
0.441; the human-fact verifier over the same inventory reads 0.448 (≈ +0.007; the paired CI is computed in the controls job
below) → **NOT MET**. The registered conclusion applies: the deployable pipeline's gain over every pre-existing pipeline comes
from the reference-worthiness policy, which lifts every emitter by +0.10 to +0.20 recall@10 (judge-label distilled 0.244 → 0.441,
human-fact 0.280 → 0.413, MARS-3 tagger 0.194 → 0.294) and lowers the unknown share from 0.84–0.90 to 0.72–0.82; the natural
human-fact verifier's own contribution is +0.028 [+0.012, +0.048] on gemma+ent and ≈ 0 on distilled; inventory order
distilled ≥ gemma+ent > gemma ≫ spaCy. No method-novelty claim attaches to the verifier. Cost of the filter: the share of
emitted-and-known units that are true omissions falls slightly (human-fact gemma+ent 0.74 → 0.70; judge-label distilled 0.71 →
0.68). The policy is now the component that carries the result, so its own novelty must be isolated before any claim.

**Amendment 2026-09-16 15:45 UTC — policy controls (`slurm/mc_e2ctl.sbatch`, `code/mc_policy_controls.py`, registered before
running; CPU only, every omission score exists).** Two summary-blind stand-ins for the learned policy, applied identically to the
human-fact and judge-label emitters over the distilled and gemma+ent inventories: (a) **lead position**, importance = 1 − start /
len(source), threshold chosen on TUNE by the same grid and criterion as the learned policy; (b) **random filter** at the learned
policy's keep rate (threshold = 1 − keep rate at τ 0.2, per inventory, seed-locked). Primary: human-fact + learned policy over
distilled (0.448); paired document bootstrap against every control and against judge-label + learned policy over distilled.
Bars: learned − lead-position ≥ +0.03 recall@10 with CI above 0 on both inventories (else the policy is a lead-position prior
and is reported as such); learned − random CI above 0 (else filtering per se, not what is kept, explains the gain). Also
reported: the paired Δ human-fact vs judge-label under the learned policy on distilled (reading iii, exact).

**Outcome 2026-09-16 16:00 UTC — policy controls (job 4837, validation; lead-position τ chosen on TUNE = 0.2, i.e. only the
last fifth of the document is dropped; random thresholds 0.795 / 0.778 = the learned policy's keep rates 0.205 / 0.222).**
Recall@10 of human-omitted ACUs, paired Δ against the primary (human-fact + learned policy over distilled, 0.448):

| emitter | recall@10 | unknown | Δ primary − this [95 % CI] |
|---|---:|---:|---:|
| distilled · judge-label + learned policy | 0.441 | 0.750 | +0.007 [−0.005, +0.019] (reading iii, exact: verifier ≈ 0) |
| gemma+ent · human-fact + learned policy | 0.413 | 0.750 | +0.034 [−0.004, +0.078] |
| distilled · human-fact + **random** filter, matched keep rate | 0.327 | 0.824 | **+0.121 [+0.083, +0.161]** |
| distilled · judge-label + random filter | 0.307 | 0.829 | +0.141 [+0.096, +0.185] |
| distilled · human-fact + **lead position** | 0.300 | 0.842 | **+0.150 [+0.105, +0.194]** |
| distilled · human-fact, no filter | 0.292 | 0.844 | +0.156 [+0.109, +0.206] |
| gemma+ent · human-fact + lead position | 0.281 | 0.849 | +0.164 [+0.109, +0.227] |
| distilled · judge-label + lead position | 0.255 | 0.859 | +0.194 [+0.148, +0.241] |

Both control bars are met on the distilled inventory: learned − lead position +0.150 (bar ≥ +0.03, CI above 0) and learned −
random +0.121 (CI above 0). On gemma+ent the same differences are +0.132 and +0.102 by point estimate; their exact paired CIs
(primary = gemma+ent human-fact + learned policy) are computed in the next job. The lead-position prior adds nothing over no
filter (0.300 vs 0.292); the learned reference-worthiness policy is therefore neither a position prior nor "filtering per se".
Unexpected: the random filter at the learned keep rate beats the verifier's own full ranking (0.327 vs 0.292; 0.311 vs 0.280 on
gemma+ent) — the top of the omission ranking is redundant (several units of one source sentence), which motivates the step below.

**Amendment 2026-09-16 16:10 UTC — diversity step (`code/mc_diversity.py`, `slurm/mc_e2div.sbatch`, registered before running;
CPU).** Rule, applied identically to every emitter: within a pair, the best-scored kept unit of each source sentence keeps its
score; other kept units of that sentence are pushed by −5 (below every sentence leader, above policy-removed units at −10). No
tuning (cap fixed at one unit per sentence). Primary: human-fact + learned policy + diversity over distilled. Bars: diversity −
no diversity ≥ +0.03 recall@10 with CI above 0 on the primary; reported for the judge-label emitter and the unfiltered emitters as
well. The same job computes the exact gemma+ent control CIs. Cost: none (re-ranking).

**Outcome 2026-09-16 16:25 UTC — exact gemma+ent controls and the diversity step (job 4838, validation).**
Gemma+ent controls, primary = human-fact + learned policy (0.411): − lead position (0.280) +0.130 [+0.081, +0.176]; − random at
matched keep rate (0.309) +0.102 [+0.065, +0.142]; − judge-label + learned policy (0.384) +0.026 [+0.012, +0.046]; both control
bars met on gemma+ent as well. Diversity step (≤ 1 kept unit per source sentence), recall@10:

| emitter | no diversity | + diversity | Δ vs primary (distilled · human-fact + policy + diversity, 0.455) |
|---|---:|---:|---:|
| gemma+ent · human-fact, **no policy** | 0.280 | **0.490** | −0.035 [−0.075, +0.005] |
| distilled · human-fact, **no policy** | 0.292 | **0.473** | −0.018 [−0.037, −0.002] |
| gemma+ent · human-fact + learned policy | 0.413 | 0.461 | −0.008 [−0.049, +0.031] |
| distilled · human-fact + learned policy (primary) | 0.448 | 0.455 | +0.005 [−0.009, +0.018] (no diversity vs primary) |
| distilled · judge-label, no policy | 0.244 | 0.442 | +0.012 [−0.005, +0.029] |
| distilled · judge-label + learned policy | 0.441 | 0.440 | +0.015 [+0.001, +0.029] |
| gemma+ent · judge-label + learned policy | 0.386 | 0.427 | +0.027 [−0.011, +0.070] |

The registered diversity bar (≥ +0.03 over the policy-filtered primary) is **NOT MET** (+0.005). The unregistered but decisive
finding: **the emission rule alone does what the learned policy did, and more** — human-fact 0.280 → 0.490 (gemma+ent) and
0.292 → 0.473 (distilled), judge-label 0.244 → 0.442; on top of the rule the learned policy adds nothing (−0.008 / +0.005 by
point estimate, and the no-policy distilled emitter is significantly above the policy + diversity primary, −0.018 [−0.037,
−0.002]). Interpretation: the policy's gain in 4835–4837 was mostly de-duplication of the top of the omission ranking (several
units of one sentence) — the reference-worthiness signal itself is worth little once redundancy is removed. This is a
simplification result: the deployable pipeline needs no learned importance component. Everything above is on validation; the
learned policy stays in the record as run.

**Amendment 2026-09-16 16:35 UTC — full comparison under the identical emission rule (`slurm/mc_e2div_full.sbatch`, registered
before running; CPU, re-ranking only).** Every unit-scores emitter gets the same ≤ 1-unit-per-sentence rule and no learned policy:
human-fact over gemma+ent (primary) and distilled; judge-label over gemma+ent, distilled, gemma, spaCy; MARS-3 tagger over spaCy;
fact-level tagger over gemma+ent; plus the unruled references and the policy variants. MARS-2 (dump format) is reported unruled.
Bars: (1) the registered E2 bar under the identical rule — primary ≥ strongest judge-label + rule + 0.05 with CI above 0; (2) the
verifier contribution under the rule — human-fact vs judge-label on the same inventory, CI reported. No tuning of the cap.

**Outcome 2026-09-16 16:45 UTC — full comparison under the identical emission rule (job 4839, validation; ≤ 1 kept unit per
source sentence, no learned policy).** Recall@10 of human-omitted ACUs, paired Δ against the primary (gemma+ent · human-fact + rule):

| emitter | recall@5 | recall@10 | unknown | Δ primary − this [95 % CI] |
|---|---:|---:|---:|---:|
| **gemma+ent · human-fact + rule (primary)** | 0.364 | **0.490** | 0.761 | — |
| distilled · human-fact + rule | 0.345 | 0.473 | 0.752 | +0.018 [−0.016, +0.056] |
| gemma+ent · human-fact + learned policy + rule | 0.353 | 0.461 | 0.741 | +0.029 [+0.009, +0.057] |
| distilled · human-fact + learned policy, no rule | 0.306 | 0.448 | 0.747 | +0.042 [−0.002, +0.087] |
| **distilled · judge-label + rule** (strongest judge-label pipeline) | 0.319 | 0.442 | 0.761 | **+0.047 [+0.007, +0.081]** |
| gemma · judge-label + rule | 0.189 | 0.334 | 0.808 | +0.154 [+0.105, +0.207] |
| gemma+ent · human-fact, no rule | 0.187 | 0.280 | 0.856 | +0.210 [+0.160, +0.257] |
| gemma+ent · judge-label + rule | 0.148 | 0.276 | 0.874 | +0.215 [+0.174, +0.258] |
| gemma+ent · fact-level tagger + rule | 0.150 | 0.268 | 0.881 | +0.221 [+0.172, +0.271] |
| spaCy · MARS-3 tagger + rule | 0.161 | 0.261 | 0.873 | +0.232 [+0.180, +0.282] |
| spaCy · judge-label + rule | 0.142 | 0.259 | 0.873 | +0.230 [+0.182, +0.287] |
| distilled · judge-label, no rule (best pre-existing pipeline) | 0.154 | 0.244 | 0.864 | +0.250 [+0.190, +0.309] |
| spaCy · MARS-3 tagger, no rule | 0.117 | 0.194 | 0.903 | +0.294 [+0.241, +0.346] |
| spaCy · MARS-2 (dump format, no rule) | 0.110 | 0.189 | 0.902 | +0.297 [+0.240, +0.349] |

Bar (1), the registered E2 bar under the identical rule: +0.047 [+0.007, +0.081] over the strongest judge-label pipeline — the
CI is above 0, the +0.05 margin is missed by 0.003 (recall@5: +0.044 [+0.015, +0.079]) → **NOT MET on the margin, positive
with CI above 0.** Bar (2), the verifier under the rule: on gemma+ent the human-fact verifier beats the judge-label verifier by
+0.215 [+0.174, +0.258] (the judge-label cross-encoder, trained on spaCy units, ranks entity units of the gemma+ent inventory
as omitted — unknown share 0.874 vs 0.761); on distilled propositions the gap is +0.031 (0.473 vs 0.442; the two Δs against
the primary overlap). The learned importance policy is unnecessary under the rule (+0.029 [+0.009, +0.057] worse). Frozen
deployable pipeline as of this record: gemma+ent inventory → human-fact cross-encoder (E1b, 3 seeds) → ≤ 1 unit per sentence →
top-k. Result on validation: 0.490 vs 0.189 (MARS-2), 0.194 (MARS-3 tagger), 0.244 (best pre-existing pipeline). TEST remains
closed until the user's decision; the pipeline above is what E2 proper would score, unchanged.

**Per-resource breakdown (job 4840, same systems, recall@10 of human-omitted ACUs; CNN/DM n = 540 pairs, SAMSum 464, XSum 384).**
Frozen pipeline (gemma+ent · human-fact + rule) 0.447 / 0.670 / 0.366; distilled · human-fact + rule 0.486 / 0.552 / 0.372;
distilled · judge-label + rule 0.458 / 0.552 / 0.307; gemma+ent · judge-label + rule 0.161 / 0.594 / 0.114; MARS-3 tagger
(spaCy) 0.103 / 0.479 / 0.033; MARS-2 (spaCy) 0.078 / 0.476 / 0.056. The gemma+ent inventory's advantage is on SAMSum
dialogues (+0.12 over distilled); on CNN/DM the distilled propositions are the better inventory for both verifiers (0.486 /
0.458 vs 0.447), and the strongest judge-label pipeline is ahead of the frozen pipeline there by 0.011. XSum stays the hardest
resource for every emitter (≤ 0.37). MARS-2's 0.189 overall is carried by SAMSum alone (0.48; CNN/DM 0.08, XSum 0.06).

**Amendment 2026-09-16 17:30 UTC — program to close the distance to a main-contribution claim (`PLAN_MAIN_CONTRIBUTION.md`,
registered before running).** Blocks G1–G6 with their bars are fixed in that file: G3 margin variants (union inventory, verifier
ensemble; promotion bar +0.02 with CI > 0, decided before TEST), G2 judge-estimated precision of emitted facts (two families,
validated on human-known units; bars: frozen ≥ 0.50, non-inferiority within 0.03, validity ≥ 0.85, κ ≥ 0.6), G4 cost account,
G5 untouched-pool feasibility census, G6 blinded human sample, G1 TEST one shot (gated on the user's explicit decision; systems,
k, bootstrap and bar fixed above). Code: `code/mc_variants.py`, `code/mc_judge_precision.py`, `code/mc_human_sample.py`,
`code/mc_cost.py`, `code/mc_pool_feasibility.py`; `code/mc_e2_eval.py --dump-emitted`, `code/mc_diversity.py --cap`;
`slurm/mc_g3.sbatch`, `mc_g2_judge.sbatch`, `mc_g2_analyze.sbatch`, `mc_g4_cost.sbatch`, `mc_g5_pool.sbatch`, `mc_g1_test.sbatch`.

**Outcome 2026-09-16 18:40 UTC — G3 margin variants (jobs 4842, 4853) and the definition of the rule.** Recall@10, validation:
union inventory (gemma+ent ∪ distilled, human-fact) 0.403 (−0.086 vs frozen); verifier ensemble 0.294 / 0.307; cap 2 0.331
(−0.157); k = 20: frozen 0.607 vs strongest judge-label + rule 0.580 (+0.026 [−0.019, +0.073]); MARS-2 under the rule 0.218
(from 0.189). **Nothing is promoted; the frozen pipeline stays.** The diagnostic of the ensemble collapse exposed how the rule
had been applied everywhere since 4838: per verifier SEED, then the three post-rule score vectors are averaged — a
**consensus regime** in which units that lead their sentence in all three seeds rank first. Applying the same rule once to the
seed-averaged scores (**single-model regime**) gives human-fact 0.362 (single seed 0.363) and judge-label distilled 0.303; the
consensus regime gives 0.490 and 0.442. Both regimes are legitimate deployable systems (3× vs 1× verifier cost), both were
applied identically to every system within a regime, and the human-fact verifier leads in both (+0.048 consensus, +0.059
single-model; paired CIs of the single-model gap to be read from G1/G7). The regime of record for TEST and for the external
confirmation is the consensus regime as evaluated so far; the single-model regime is reported next to it in both. Every prior
number in this record labelled "+ rule" is the consensus regime.

**Amendment 2026-09-16 18:45 UTC — G7 external confirmation on UniSumEval (registered before running; `code/mc_unisum.py`,
`slurm/mc_g7a/b/c_*.sbatch`).** G5 found UniSumEval (DISL-Lab): 1,828 labelled summaries of 225 documents in nine domains
(CNN/DM, WikiHow, SQuALITY, PubMed, GovReport, MediaSum, MeetingBank, DialogSum, MultiWOZ; short and long inputs, median 683
words, p90 5,104), nine summarizers (GPT-4, Claude 2.1, GPT-3.5, Mixtral, Mistral, Llama-2, phi-2, BART, T5), ten human-validated
key facts per document with a per-summary coverage label each (median 5.5 omitted per summary; 92 % of summaries have ≥ 1).
One document overlaps our source pool and is excluded. Nothing in this pool touched any development step. Protocol: the frozen
inventory recipe (Gemma-4-31B sentence decomposition + spaCy entities), the frozen verifiers (E1b arm A ×3, B21 cross-encoder
×3), the rule in both regimes, emitted-top-k recall of human-omitted key facts, 500 document bootstraps, per-domain breakdown.
Bars: (1) frozen pipeline vs judge-label + rule (same inventory, consensus regime): CI above 0 (secondary: ≥ +0.05); (2) the
same in the single-model regime; (3) both verifiers + rule above their unruled references. No tuning of anything on this pool.

**Outcome 2026-09-16 20:20 UTC — G2 judge-estimated precision of emitted facts (jobs 4845/4846; 69,596 emitted units, two
families, two questions).** Judge validity **NOT MET**: coverage accuracy on the human-known emitted units (hit vs false alert,
n = 9,866) is 0.685 (Gemma-4-31B) and 0.705 (Qwen3.8-27B), against the registered 0.85; family agreement on the true-omission
verdict κ = 0.457 (bar 0.6); Gemma calls 36 % of emitted units unsupported by the source, Qwen 10 %. Per registration the
estimate is reported as unvalidated and the precision claim rests on the human sample (G6). For the record, judged true-omission
rates (Gemma / Qwen), pair-weighted: distilled · human-fact + rule 0.679 / 0.764; distilled · judge-label + rule 0.677 / 0.762;
**frozen pipeline (gemma+ent · human-fact + rule) 0.615 / 0.745**; MARS-2 0.219 / 0.680; MARS-3 tagger + rule 0.214 / 0.608.
Frozen ≥ 0.50 (met by both families); non-inferiority vs the strongest judge-label pipeline NOT met under Gemma (judge distilled
− frozen +0.063 [+0.041, +0.085]), but the gap is the inventory, not the verifier: on the same distilled inventory the two
verifiers read the same judged precision (0.679 vs 0.677) while the human-fact verifier has the higher recall (0.473 vs 0.442).
Entity units of the gemma+ent inventory are the ones the judges call non-omissions more often. Caveat on the validity number:
the "known" status comes from lexical matching to reference ACUs and is itself noisy, so 0.69–0.71 bounds the judge from below
only loosely; the blinded human sample is the arbiter. Consequence for the paper: no judge-based precision figure; the
distilled inventory is the candidate for the precision-sensitive variant (same verifier, +0.06 judged precision, −0.017 recall).

**Outcome 2026-09-16 19:00 UTC — G1 TEST, one shot (job 4858, submitted by the user with `B2_SPLIT=test` at 17:22 UTC; sealed split,
2,688 pairs with ≥ 1 human-omitted ACU: CNN/DM 1,200, SAMSum 816, XSum 672; `~/results/marsc/g1_test/SHA256SUMS`).** Frozen
systems exactly as registered; recall@10 of human-omitted ACUs, paired Δ against the primary:

| system (TEST) | recall@5 | recall@10 | CNN/DM | SAMSum | XSum | Δ primary − this [95 % CI] |
|---|---:|---:|---:|---:|---:|---:|
| **frozen pipeline: gemma+ent · human-fact · rule (consensus)** | 0.211 | **0.378** | 0.280 | 0.634 | 0.294 | — |
| distilled · human-fact · rule (consensus) | 0.219 | 0.371 | 0.292 | 0.497 | 0.386 | +0.009 [−0.020, +0.034] |
| gemma+ent · human-fact · rule (single-model) | 0.202 | 0.351 | 0.248 | 0.626 | 0.256 | +0.028 [+0.010, +0.044] |
| distilled · human-fact · rule (single-model) | 0.205 | 0.338 | 0.258 | 0.490 | 0.330 | +0.040 [+0.010, +0.069] |
| **distilled · judge-label · rule (consensus) — strongest judge-label pipeline** | 0.189 | 0.327 | 0.236 | 0.495 | 0.322 | **+0.051 [+0.022, +0.077]** |
| distilled · judge-label · rule (single-model) | 0.179 | 0.308 | 0.219 | 0.480 | 0.294 | +0.070 [+0.041, +0.101] |
| gemma+ent · human-fact, no rule | 0.170 | 0.266 | 0.153 | 0.538 | 0.195 | +0.111 [+0.088, +0.137] |
| distilled · judge-label, no rule | 0.157 | 0.260 | 0.160 | 0.450 | 0.249 | +0.118 [+0.085, +0.149] |
| gemma+ent · judge-label · rule (consensus) | 0.135 | 0.256 | 0.148 | 0.579 | 0.119 | +0.122 [+0.101, +0.143] |

**Bar MET**: +0.051 over the strongest judge-label pipeline under the identical rule with the CI above 0 (registered ≥ +0.05,
CI > 0); at recall@5 the gap is +0.022 [−0.000, +0.045]. The single-model regime: human-fact 0.351 vs judge-label 0.308
(+0.043 by point estimate; the paired CI of this contrast is computed in the re-evaluation below, same files, no new scoring).
Absolute recall is lower than on validation for every system (frozen 0.378 vs 0.490; judge 0.327 vs 0.442): TEST pairs come
from more systems per document and harder documents, and the ordering is unchanged. Per resource, the human-fact verifier's
margin is largest on SAMSum (0.634 vs 0.495) and present on CNN/DM (0.280 vs 0.236); on XSum the distilled inventory is the
better carrier (0.386 human-fact vs 0.322 judge-label). MARS-3 tagger and MARS-2 are absent (no saved weights / no TEST dumps).
Single-model regime, paired on the same TEST files (re-evaluation only, primary = gemma+ent · human-fact · rule single): vs
distilled · judge-label · rule single **+0.044 [+0.015, +0.068]** (recall@5 +0.023 [+0.000, +0.045]); vs distilled · human-fact
single +0.012 [−0.011, +0.037]; vs gemma+ent · judge-label single +0.114 [+0.094, +0.136]. So the verifier's advantage holds
with a CI above 0 in both regimes on the sealed split; the +0.05 margin is reached in the consensus regime (+0.051) and not in
the single-model regime (+0.044).

**Note 2026-09-16 21:30 UTC — G7 first pass (jobs 4854–4856, 4866) is bug-affected and is NOT the registered result.** The build
keyed pairs by UniSumEval's `uid`, which is per DOCUMENT, so the nine summaries of a document collapsed onto one (the last in
file order): every score file and the evaluation cover 224 pairs (one summary per document), although all 1,822 pairs were
scored. Outputs moved to `*_collapsed_224`. For the record, that 224-pair subset read recall@10 human-fact single-model 0.387,
human-fact no rule 0.354, human-fact consensus 0.340, judge-label single 0.312, judge-label consensus 0.289, judge-label no rule
0.275 (consensus contrast +0.051 [+0.023, +0.078]); the single-model regime was ahead of consensus there, unlike on RoSE.
Fix: `pair_id = uid:model` with a uniqueness assertion; rebuild (CPU), rescore all 1,822 pairs (6 GPU tasks), evaluate (CPU),
same registered protocol and bars, unchanged. Jobs recorded in the ledger as g7a2 / g7e / g7f.

**Interim 2026-09-16 21:50 UTC — G6 human confirmation, first reading (Annotator 2 complete 300/300, Annotator 1 at 223/300;
labels 17:49–21:41 UTC on label.example.org).** Agreement on the 223 co-labelled items: Q2 (conveyed) raw 0.85, κ 0.66; Q1
(supported) raw 0.88 with κ ≈ 0 because Q1 is almost always A; confirmed-omission verdict raw 0.77, κ 0.52. Confirmed-omission
rate (Q1 = A ∧ Q2 = B) by system and evaluator status, Annotator 1 / Annotator 2: judge-label distilled + rule — all 0.63 / 0.68,
unknown 0.77 / 0.78, hit 0.47 / 0.70, false alert 0.31 / 0.35; **frozen pipeline (gemma+ent · human-fact + rule) — all 0.51 /
0.61, unknown 0.59 / 0.73, hit 0.64 / 0.70, false alert 0.13 / 0.15**; MARS-2 — all 0.51 / 0.63, unknown 0.68 / 0.77, hit 0.53 /
0.65, false alert 0.00 / 0.20. Readings: (i) most "unknown" emitted facts are real omissions, so the reference ACUs are
incomplete and the recall endpoint undercounts everyone; (ii) the lexical hit / false-alert status agrees with the human Q2 only
0.71–0.73 of the time — the G2 judge "validity" of 0.69–0.71 was measured against that noisy status, so the judges are re-scored
against the humans directly (`code/mc_judge_vs_human.py`); (iii) precision favours the distilled inventory: the entity-bearing
gemma+ent inventory pays ≈ 0.1 in confirmed-omission rate for its +0.02 recall (validation 0.490 vs 0.473; TEST 0.378 vs 0.371,
CI incl. 0). The natural-mix precision (strata weighted by each system's own hit / false-alert / unknown shares) is computed in
the same script.

**Amendment 2026-09-16 21:55 UTC — G6b sample extension (registered before drawing).** 100 further blinded items from
**distilled · human-fact + rule** (the candidate pipeline of record on precision grounds), same strata 60/20/20, seed 20260917,
item ids H301–H400, appended to the live app for both annotators. Decision rule for the pipeline of record in the paper: if the
distilled · human-fact pipeline's natural-mix confirmed-omission rate is ≥ the judge-label distilled pipeline's − 0.03 (both
annotators, and on items where they agree), the distilled inventory becomes the pipeline of record (recall 0.371 on TEST, +0.044
over the same-rule judge-label pipeline by the single-model contrast, 0.08 s/pair); otherwise the gemma+ent pipeline stays with
its precision cost stated.

**Outcome 2026-09-16 22:05 UTC — G6 natural-mix precision and the judges re-scored against the humans.** Natural-mix
confirmed-omission rate of the emitted top-10 (strata weighted by each system's own hit / false-alert / unknown shares;
Annotator 1 / Annotator 2 / items where both agree): **judge-label distilled + rule 0.687 / 0.735 / 0.806**; **frozen pipeline
(gemma+ent · human-fact + rule) 0.560 / 0.682 / 0.666**; MARS-2 0.650 / 0.741 / 0.773. So the frozen pipeline buys its recall
(0.490 vs 0.442 validation; 0.378 vs 0.327 TEST) with ≈ 0.1 lower precision than the judge-label pipeline over distilled facts,
and MARS-2's emitted facts are mostly real omissions that are simply not the reference ones. Judges against the humans
(`mc_judge_vs_human.py`, 300 items): coverage answer vs Q2 — Gemma 0.90 / 0.86 (0.95 on the 189 agreed items), Qwen 0.88 /
0.84 (0.93) → **the coverage judgment is valid at the registered 0.85** on agreed items and for Gemma on Annotator 1; the
true-omission verdict is only 0.71–0.79 accurate because the support question is too strict (Gemma calls 36 % of emitted facts
unsupported where the humans answer A almost always). Consequence: G2's coverage-based "omitted" rates may be used with the
human-validated accuracy stated; its support-based rates may not. The G6b extension (distilled · human-fact + rule, 100 items,
H301–H400) is live for both annotators and decides the pipeline of record by the rule registered above.

**Outcome 2026-09-16 22:45 UTC — G7 external confirmation on UniSumEval, corrected run (jobs 4869–4871; 1,822 pairs, 224
documents, nine domains, 500 document bootstraps).** Recall@10 of human-omitted key facts, frozen inventory recipe, frozen verifiers:

| system | recall@5 | recall@10 | unknown | Δ vs human-fact consensus [95 % CI] |
|---|---:|---:|---:|---:|
| human-fact · rule, **single-model** | 0.283 | **0.389** | 0.649 | −0.042 [−0.054, −0.031] |
| human-fact · rule, consensus (primary) | 0.258 | 0.347 | 0.661 | — |
| human-fact, no rule | 0.251 | 0.338 | 0.633 | +0.009 [−0.003, +0.021] |
| judge-label · rule, single-model | 0.227 | 0.317 | 0.704 | +0.031 [+0.012, +0.051] |
| judge-label · rule, consensus | 0.219 | 0.299 | 0.708 | **+0.048 [+0.030, +0.069]** |
| judge-label, no rule | 0.203 | 0.274 | 0.691 | +0.073 [+0.054, +0.093] |

Single-model contrast, paired: human-fact 0.389 vs judge-label 0.317 → **+0.072 [+0.054, +0.089]**; human-fact single vs its
unruled reference +0.051 [+0.039, +0.062]. Bars: (1) consensus regime CI above 0 — MET (+0.048; the secondary +0.05 missed by
0.002); (2) single-model regime — MET with margin (+0.072); (3) above the unruled references — MET for the single-model rule
and for the judge-label verifier; the consensus rule adds nothing over no rule on this pool (+0.009, CI incl. 0). Per domain
(k = 10, human-fact single vs judge-label single): CNN/DM 0.501 vs 0.334, WikiHow 0.839 vs 0.834, MultiWOZ 0.804 vs 0.764,
DialogSum 0.658 vs 0.648, PubMed 0.287 vs 0.081, MeetingBank 0.282 vs 0.200, MediaSum 0.216 vs 0.119, GovReport 0.063 vs
0.042, SQuALITY 0.053 vs 0.055 — ahead in seven of nine, tied on the two where both are near the ceiling (WikiHow) or near
zero (SQuALITY). Long inputs (GovReport, SQuALITY; hundreds of units per pair) are below 0.07 for every system at top-10.
**Regime finding:** on this pool the single-model rule beats the consensus rule by 0.042, the reverse of RoSE (validation
0.362 vs 0.490; TEST 0.351 vs 0.378). The single-model regime is therefore the robust default (cheaper, wins externally, CI
above 0 on TEST as well); the consensus regime is reported as a RoSE-specific gain. This changes the presentation, not the
verdict: the human-fact verifier is ahead in both regimes on TEST and on the untouched pool.

**Amendment 2026-09-16 23:17 UTC — T01–T08, the decision-space broadening block (registered BEFORE submission; jobs listed in the tracker).**
The ICLR 2027 submission was reframed on 2026-09-17 from the audit thesis to the MARS-C method thesis, so the manuscript now
has to survive a method-paper reading rather than an audit reading. The registered primary claim is unchanged and is NOT
re-opened by anything below: frozen pipeline (gemma+ent inventory · human-fact cross-encoder · ≤ 1 unit per source sentence)
beats the strongest judge-label pipeline under the identical rule by +0.051 [+0.022, +0.077] recall@10 on the sealed TEST
split, replicated on UniSumEval. Eight additions, each registered here with its bar before it is run:

| id | what | split | status of the endpoint | bar / reporting rule |
|---|---|---|---|---|
| T01 | full ablation grid: inventory × verifier × rule × seed-regime (16 cells) | TEST (frozen outputs) | post-hoc SECONDARY | report-only; the verifier contrast at fixed inventory and fixed rule is the quantity of interest |
| T02 | recall@k curve, k ∈ {1,3,5,10,20,50} | TEST + UniSumEval (frozen outputs) | post-hoc SECONDARY | report-only; k=10 remains the registered primary and is not re-chosen |
| T03 | length-adaptive budget k = clip(ratio · n_source_sentences, 3, 50), ratio ∈ {0.25, 0.50} | UniSumEval + TEST (frozen outputs) | post-hoc SECONDARY | applied identically to every system; reported as an additional budget, never substituted for the flat-k primary |
| T04 | document-level paired permutation test (10,000 sign-flips) + exact per-document sign test | TEST + UniSumEval (frozen outputs) | post-hoc SECONDARY | report-only; the bootstrap CI stays the inferential procedure of record |
| T05 | MARS-2 (`full_recovery`, `direct_enabled`, 3 seeds) as an emitter | TEST — NEW SCORING | ADDING A BASELINE | no bar; adds our own predecessor to the headline table. Registered as a baseline addition, not a re-read of our own system |
| T06 | precision judge re-validated with four hand-written boundary-anchored examples | G2 tasks, validated against the blinded G6 human labels | confirmatory, pre-registered bar | judged precision becomes reportable only if accuracy on human-known units ≥ 0.85 (the original bar); plain prompts scored 0.685 / 0.705 |
| T07 | verifier label-efficiency curve, 10/25/50/100 % of the 853 human-fact TRAIN documents × 3 seeds | VALIDATION only | exploratory | report-only; TEST is not touched |
| T08 | seed-aggregation variants (rank-average, lead-union) vs consensus and single-model | TEST + UniSumEval (frozen outputs) | post-hoc SECONDARY | promotion bar +0.02 with CI above 0 against the consensus regime; below that the regime of record is unchanged |

Integrity constraints carried by this block. (1) The sealed TEST split was opened once, on 2026-09-16 (G1/MC8), and is no longer
pristine for tuning MARS-C; T01–T04 and T08 are therefore re-reads of the frozen score files with no tuning and no re-fitting,
and every one of them is reported as a post-hoc secondary endpoint. (2) T05 is the only entry that runs new scoring on TEST; it
scores a BASELINE we already owned and whose weights predate the split being opened, and it cannot change our own numbers.
(3) T03 changes the emission budget, which is a property of the protocol, not of a system — it is applied to every system in the
comparison or to none. (4) T06's bar is the one already registered for G2 and is not relaxed; if the anchored prompts miss 0.85
the judge stays unusable and the precision claim continues to rest on the human sample alone. (5) No result below licenses a
change to the registered primary claim, the pipeline of record, or the regime of record except through T08's stated bar.

**Outcome 2026-09-17 07:00 UTC — G6 human confirmation, complete (both annotators 300/300; extension H301–H400 not yet started).**
Agreement on 300 co-labelled items: Q2 (conveyed) raw 0.85, κ 0.67; verdict raw 0.76, κ 0.52; Q1 raw 0.86 (κ degenerate, almost
always A). Natural-mix confirmed-omission rate of the emitted top-10 (Annotator 1 / Annotator 2 / items where both agree):
**judge-label distilled + rule 0.660 / 0.735 / 0.778; frozen pipeline (gemma+ent · human-fact + rule) 0.557 / 0.682 / 0.666;
MARS-2 0.608 / 0.741 / 0.737.** By status, frozen pipeline: unknown 0.60 / 0.73, hit 0.55 / 0.70, false alert 0.15 / 0.15.
The G6 bar ("frozen pipeline at least as precise as the judge-label pipeline and MARS-2") is **NOT MET** for the gemma+ent
inventory: ≈ 0.11 below the judge-label pipeline on agreed items. Judges against the humans on all 300 items: coverage — Gemma
0.91 / 0.86 (0.95 on the 255 agreed), Qwen 0.90 / 0.84 (0.94) → the plain coverage judgment is validated; verdict 0.70–0.79
(support question too strict). The parallel session's T06 confirmed this and rejected anchored prompts (0.95 → 0.83). Decision
on the pipeline of record waits for the 100-item extension on distilled · human-fact + rule (registered rule: within 0.03 of the
judge-label pipeline's precision → distilled becomes the pipeline of record; TEST 0.371, +0.044 single-model, 0.08 s/pair).

**Outcome 2026-09-17 10:40 UTC — G6b complete (both annotators 400/400; H301–H400, distilled · human-fact + rule, labelled).**
Final export `g6/labels_export_all.jsonl` (800 rows); analysis `g6/analysis_all/` (`mc_human_analyze.py --e2-json
g3/eval_validation/e2_emitted.json`, `mc_judge_vs_human.py`, both via srun). Annotator 2 revised 11 Q2 answers on H001–H300
before starting the extension (7 A→B, 4 B→A, all on the three original systems); the final export supersedes the 300-item reading
of 07:00 UTC. Agreement on the 400 co-labelled items: Q1 raw 0.86 (κ degenerate), Q2 raw 0.88, κ 0.72; verdict raw 0.78, κ 0.53.
Natural-mix confirmed-omission rate (Annotator 1 / Annotator 2 / items where both agree): **distilled · human-fact + rule
0.688 / 0.763 / 0.848; judge-label distilled + rule 0.660 / 0.806 / 0.819; MARS-2 0.608 / 0.753 / 0.742; frozen gemma+ent ·
human-fact + rule 0.557 / 0.657 / 0.643.** By status, distilled · human-fact: unknown 0.77 / 0.87, hit 0.55 / 0.55, false alert
0.25 / 0.25. Registered rule (distilled ≥ judge-label distilled − 0.03 on both annotators AND on agreed items): Annotator 1
+0.028 MET, agreed items +0.029 MET, Annotator 2 −0.043 NOT MET (0.013 beyond the tolerance). The rule is a conjunction →
**NOT MET: the gemma+ent pipeline stays the pipeline of record, with its precision cost stated (0.10–0.18 below the judge-label
pipeline: 0.103 / 0.149 / 0.176).** The distilled · human-fact pipeline is reported as the precision-matched configuration: the
most precise of the four systems on agreed items and for Annotator 1, within 0.05 of the judge-label pipeline for Annotator 2,
TEST 0.371 (consensus) / 0.338 (single), 0.21 / 0.08 s per pair. Judges against the final labels (all 400 items carry G2 tasks):
coverage — Gemma 0.90 / 0.90 (0.96 on the 352 agreed), Qwen 0.92 / 0.88 (0.95); verdict (coverage ∧ support) — Gemma 0.71 /
0.77 (0.81), Qwen 0.78 / 0.85 (0.90): the coverage answer stays validated, the verdict does not. Evaluator lexical status vs Q2
on hit / false-alert items: 0.71 / 0.72. The G6 bar for the frozen pipeline remains NOT MET; no change to the registered primary
claim or the regime of record. All "in progress" statements in the manuscript are replaced by these numbers.

**Outcome 2026-09-17 12:20 UTC — pre-submission audit of the manuscript (facts, language, scientific consistency); corrections applied.**
Every number in the MARS-C sections was re-checked against the result files. Verified against their sources with no change:
all recall@10 cells of Table 1 (`g1_test`, `t01_grid`, `g3`, `g7_unisum`, `t05_mars2`), the paired contrasts and intervals,
the per-domain UniSumEval and per-resource TEST breakdowns, the human-confirmation rates, agreements and by-status rates
(`g6/analysis_all/human_confirmation.json`), the judge validations (`g6/analysis_all/judge_vs_human.json`), the cost table
(`g4/cost_table.json`, 0.85 ms/unit, 165 and 70 units/pair) and the derived ratios (a quarter of MARS-2's cost, ten times
below 0.82, twice its recall). Four corrections:
(1) **Judge-label supervision count was unsourced.** The method section claimed 119,808 coverage labels; no artefact produces
that number. The B1 assemble manifest records 119,801 judge tasks issued, 119,676 judgements returned, 125 dropped for low
choice mass (label sources: judge 118,353, human anchors 1,323). Corrected to **119,676**, the count of coverage labels the
judge returned. The wrong figure entered at commit ab9fb629 (the manuscript reframe) and was never in any result file.
(2) **Regime of record contradicted this registration.** The Discussion and Results called the cheaper single-model rule
"the claim of record" / "the robust default". The amendment of 2026-09-16 fixed the **consensus** regime as the regime of
record for TEST and for the external confirmation, and constraint (5) allowed a change only through T08's bar, which promoted
nothing. Corrected: the paper now keeps consensus as the registered regime of record, reports single-model beside it, states
that single-model is positive but short of the +0.05 margin on TEST (+0.044) and stronger on UniSumEval (+0.072), and calls
the choice a cost decision rather than a claim.
(3) **Abstract, contributions and conclusion mixed regimes across pools** — a consensus TEST number beside a single-model
UniSumEval number, unlabelled. All three now give the UniSumEval consensus contrast (+0.048 [+0.030,+0.069]) as the matched
figure and name the single-model regime where its number appears.
(4) **MARS-2 baseline rows took the better of its two decoding modes per cell** (TEST: full_recovery 0.171 unruled / 0.192
ruled; direct_enabled 0.168 / 0.195; the table reports 0.171 and 0.195). The choice favours the baseline, i.e. it understates
our margin, but it was undisclosed; the Results now state that MARS-2 is reported at its better decoding mode in each case.
Also fixed: Table 2's human-confirmation row still carried the superseded 300-item numbers (now 400 items, 0.85/0.82/0.74/0.64,
kappa 0.72); an appendix sentence named the internal block id "T06"; one British/American spelling split; one comma splice
introduced while trimming. Language pass over the MARS-C sections found no doubled words, no unbalanced math, no broken
cross-references ('??' count 0). Build: errors 0, undefined 0, main text ends p.9. Anonymity gate re-run: PASS.
The supplementary packet `submission-artifacts/20260917/marsc_update.zip` was rebuilt from the corrected export.

**Outcome 2026-09-17 13:10 UTC — second audit pass, appendix-focused; six corrections.**
Mechanical sweep over all nine appendix source files plus the main text: no duplicate labels, no unresolved
cross-reference ('??' count 0), every `\cite` key present in the bibliography (117 entries, 109 cited), balanced
math in every file, no doubled words. Cell arithmetic re-derived and correct: 33 headline cells = 24
perturbation-built (11 cross-task + 10 SummEdits + 2 BUMP + VQA~v2) + 9 AggreFact strata, 32 of them text;
the 21 published summarization cells = 10 SummEdits + 9 AggreFact + 2 BUMP and split 5/1/2/13 across the four
bands; the BH family of 50 = 10 cells x 5 metrics with 40 flagged-cell pairs. E0 claims re-verified against the
census artifact (853 training-role documents, 408 with crossed coverage, 0 near-duplicate clusters, 0 label
conflicts, 2.0% token-mask collisions = 100/5,022). Corrections:
(1) **Four appendix tables had no textual pointer** — the inventory audit, the fifteen factorial contrasts and the
two creativity pilots floated with captions that no sentence referenced. Each now has a `Table~\ref{}` in the
paragraph that describes it. (`tab:four_domain` and `tab:rivals` remain unreferenced by design: they are legacy
aliases on the matched-deletion table, which is referenced.)
(2) **Tercile range mismatch between main text and appendix.** The Discussion quotes 0.646--0.904 within every
counter tercile across the flagged SummEdits domains and AggreFact strata; the appendix documented only
0.646--0.886, its AggreFact-only scope, so the two read as contradictory. Both figures are correct for their own
scope (the wider range was verified at commit aed8fb61); the appendix now states both and names the section that
quotes the wider one.
(3) **Regression from the 12:20 UTC pass.** That pass rewrote the pairwise-gap sentence as "of 100 pairwise metric
gaps 49 survive". The appendix statistic is "49 of the 10x10 pairwise gaps have a 95% paired interval excluding
zero", i.e. resolved, and "survive" collides with the neighbouring matching statistics (92/100 and 94/100).
Corrected to "49 of 100 pairwise metric gaps are resolved".
(4) **Held-out AUROCs were worded as pooled.** 0.972/0.964/0.983 are the AUROCs *on the held-out resource*
(`analysis_e1b_lodo`: 0.9721/0.9638/0.9831); the pooled values under those ablations are 0.974/0.974/0.974.
Method and appendix now say "on the held-out resource".
(5) **UniSumEval scored pool was not distinguished from the built pool.** The build yields 1,822 summaries of 224
documents; recall is computed over the 1,813 pairs of 222 documents that carry at least one omitted key fact
(`eval_unisum_single`: n_pairs 1,813, n_docs 222). The appendix now makes the same distinction it already made for
RoSE (1,388 of 1,466 validation, 2,688 of 2,829 test).
(6) **Two artifacts behind appendix claims were absent from the release.** `e0/census.json` (+ `split_manifest.json`)
and `g5/feasibility.json` (+ `FEASIBILITY.md`) were on the cluster only, so the E0 and pool-feasibility numbers were
not checkable from the packet. Both are now in `results/` and in the rebuilt `marsc_update.zip`.
Build: errors 0, undefined 0, main text ends p.9. Anonymity gate: PASS.

**Outcome 2026-09-17 14:05 UTC — third audit pass, main-text-focused; five corrections.**
Read pages 1--10 in full and re-derived every count. Mechanical sweep of the main text clean: no
article errors, no doubled words, no mixed spellings, no double spaces, no broken cross-reference.
Verified with no change: the abstract's 0.78--1.00 counter range (min 0.782 target presence on
Multi-News, max 1.00 type-Jaccard), the cell arithmetic (33 = 24 + 9; 21 published = 5+1+2+13;
BH 50 = 10 cells x 5 metrics, decision subset 40), the matched-deletion group counts
(670/542/508/700) against Table~7, the surface-controlled counts (150/84/16/96) against
Appendix~A2.1, and the MARS-C numbers re-checked in the 12:20 pass. Two apparent contradictions
resolved as correct-but-confusing and reworded rather than changed: the introduction's
"two of three instruments" against S4.1's "three audits ... each found a counter above every
metric" (the audits are three, the instruments they cover are two, so the introduction now names
the two instruments), and "the two flagged cells with fewer than 30 groups" against "the three
cells below 30 groups" (flagged subset of 8 vs all ten cells; now "three of the ten"). Corrections:
(1) **One bibliography entry misspelled an author.** `factcc` had "Krysci\'nski"; the other three
entries for the same author have "Kry\'sci\'nski". The paper rendered both spellings. Fixed; the
name now renders identically in all five places.
(2--4) **Three S4.4 claim groups cited appendix sections that did not contain their evidence.**
The object-nearest coverage metrics (zero-shot 31B judge 10/10 both families, median +0.18 nats;
A3CU 4/10, +0.006; FineSurE completeness 1--2/10; QuestEval recall 1/10, +0.005), the twenty-one
-cell extension (AlignScore 17/21, MiniCheck 14/21, NLI 0/21) and the repair-transfer results
(20 frozen-repair splits per cell; 200/200 permuted-label paths) each pointed at a section that
never mentions them; A3CU, FineSurE and QuestEval appeared nowhere in the appendix at all. All
three are TRUE and were verified here against artifacts: tracker rows A006/A008, the C3 21-cell
record, `panel5_results/p14_sensitivity/repair/` (20 draws x 10 cells, held-out counter maxima
median 0.598--0.656) and `panel5_results/p14_sensitivity/adaptive_null/` (200 draws x 10 cells,
every draw below 0.60, per-cell maxima 0.568--0.600). A new appendix subsection
(`app:objectnearest`) now documents all three, and the three main-text citations point to it.
(5) **The twenty-one-cell claim carried neither caveat its own record requires.** The C3 record
states that the small-group caveat travels with those counts (seven of the eleven new cells have
fewer than 30 groups, 17--28) and a 2026-09-14 methodology review (W9, Major) noted that the
family declaration moves a count on identical data: the 105-test family leaves AlignScore with
nine rejections on the canonical ten against the fifty-test family's eight. The main text now
carries the group-size and family qualifiers, and the appendix states the 9-versus-8 sensitivity
explicitly. W9's request for an inspectable artifact stands as a release gap: the per-cell JSON
for the 21-cell run is still not in the tree, and the appendix now says the counts are conditional
on the declared family.
Build: errors 0, undefined 0, main text ends p.9 (47 pages total). Anonymity gate: PASS.

**Outcome 2026-09-17 14:45 UTC — the 21-cell per-cell file is shipped (review W9 closed).**
The 2026-09-14 methodology review (W9, Major) asked for the per-cell JSON behind "AlignScore 17/21,
MiniCheck 14/21 and NLI 0/21", which no released directory carried. The file existed on the cluster
(`~/results/strengthen/c3_21/conditional_value.json`, produced 2026-09-12 from the per-row scores in
`~/results/strengthen/rows_21`) and had simply never been brought into the tree. It is now released as
`article/iclr2027/panel5_results/p13_conditional_value_21cell/conditional_value.json`, in the schema of
the ten-cell run beside it, with a `RECEIPT.json`. It was placed under `panel5_results/` rather than
`strengthen_results/` because only the former is in the anonymity export's include list; the file is
confirmed present in the export (gate PASS, 1990 files), where the anonymizer redacts the absolute path
in its `source` field, so the released copy hashes differently and the receipt says so.
Verified on the shipped file: 21 cells; BH family 105, 46 rejections; AlignScore 17/21 (median
+0.0569), MiniCheck 14/21 (+0.0614), SummaC-ZS 10/21 (+0.0164), FactCC 6/21 (+0.0032), NLI 0/21
(+0.0004) — every count matches the manuscript. Guard: all 50 canonical (cell, metric) increments
reproduce the ten-cell run at a maximum absolute difference of exactly 0, so the eleven new cells enter
on the same footing and not through a re-fit. Small-group caveat confirmed exactly: 7 of the 11 new
cells sit below 30 groups (17--28). **New finding from the shipped file:** the family declaration moves
counts in BOTH directions on the same ten canonical cells, not one as the review reported — AlignScore
9 (family 105) against 8 (family 50), and SummaC-ZS 7 against 8. The appendix previously stated only
the AlignScore direction and now states both.
