# MARS / MARS-C — supplementary archive (anonymized)

This archive accompanies the IEEE Access submission on MARS-C (omission finding) and MARS (the metric). It
carries the registrations, the analysis code and job scripts, the `mars` package, and the compact result files
the paper's numbers were read from. Identifying names, affiliations, host names and home paths have been
replaced (`ANONYMIZED`, `cluster-A` / `cluster-B`, `/home/user/...`); nothing else in the text was edited.

## Layout

| Path | What it is |
|---|---|
| `research/marsc_20260916/` | MARS-C main campaign: registration (`PREREG.md`), repair register (`REPAIR_R_SERIES.md`), code, job scripts, in-tree compact results (G/T series, human study G6), label app |
| `research/marsc_strengthen_20260918/` | strengthening campaign: `PREREG.md` (written before execution), `RESULTS.md` (every block, including the bars that failed), `D1_PROVENANCE_2026-09-22.md`, code, job chains |
| `research/marsc_ieee_20260922/` | H-J series for this journal version: `PREREG.md` with dated amendments H-J3 ... H-J12 and their outcome paragraphs, code, job scripts |
| `research/mars_metric_20260923/` | metric program M0, M-A ... M-G: `PREREG.md` with outcome amendments, code, job scripts, `results_snapshot/` |
| `research/marsc_audit_20260928/TODO.md` | the audit-response task list of 2026-09-28: what was re-run and what is still open |
| `research/mars2_gates_20260916/`, `research/mars3_20260916/` | supporting code the MARS-C scripts import (judge-label package B21, inventories, scorers) |
| `mars/`, `tests/test_mars_api.py`, `tests/test_b2_combine.py` | the MARS package (`mars.score`, CLI) and the tests added in the audit response |
| `registrations/NUMBERS_PROVENANCE_2026-09-17.md` | number-by-number provenance of the MARS-C conference-version figures, with the corrections applied since |
| `results/marsc/`, `results/marsc_strengthen/`, `results/mars_metric/` | cluster result snapshot: the cluster's `~/results/<program>/` tree, same relative paths |
| `results/SNAPSHOT_INFO.json` | when the snapshot was taken, the selection rules, and which pending outputs were absent |
| `results/RESULTS_MANIFEST.tsv` | SHA-256 and size of every shipped result file, and whether anonymisation rewrote it |

What `results/` contains: every `*.json` of the cited blocks up to 5 MB (evaluator outputs `e2_emitted.json`,
crossed-endpoint reports `crossed_families.json`, contrasts, verdicts, manifests, cost tables), every
`started.json` / `complete.json` receipt and `SHA256SUMS`, per-pair evaluation dumps under 2 MB, and the D1
human-study files. Not shipped: checkpoints, score dumps, emitted lists, caches, raw judge shards, logs and
third-party dataset copies (see `known_gaps` in `ARCHIVE_VERIFICATION.json`, distributed beside this archive).
Every evaluator output lists, in its `emitter_manifest`, the score files it averaged with their SHA-256.
Registrations name result paths as `~/results/<program>/...` or relative to a program root; in this archive the
same file is at `results/<program>/...`. The cluster-side `SHA256SUMS` hash the original bytes; files whose paths
were anonymised differ from them (marked `yes` in `RESULTS_MANIFEST.tsv`).

## Where the paper's tables come from

Program roots: `R = results/marsc`, `S = results/marsc_strengthen`, `M = results/mars_metric`.
Evaluator outputs are read at `k["10"].systems[<system>]` (`recall`, `omission_precision`); crossed reports at
`families.<family>.seed_averaged.discrimination_doc_macro` with `ci_doc_bootstrap`. Table numbers follow the manuscript of 2026-09-29 (Table 1 endpoints, Table 3 crossed endpoint, Table 4 metric table). Rows marked *verified* were checked
cell by cell against the shipped JSON when this archive was built; the rest are taken from the registrations'
artifact paragraphs and table-rendering scripts and were not re-checked cell by cell.

### Table 1 — two endpoints on every frozen system (`tab:marsc_main`) — verified

| Column | File | Systems |
|---|---|---|
| Val. rec@10, consensus rows and MARS-2 | `R/r09_validation/eval_validation/e2_emitted.json` | `mars2` 0.189, `mars2+div1` 0.218, `judgefact+div1` 0.276, `judgefact.distilled+div1` 0.311, `humanfact+div1` 0.400, `humanfact.distilled+div1` 0.369 |
| Val. rec@10, seed-average rows | human-fact: `S/a1/eval_val_cal/e2_emitted.json` `humanfact.single+div1` 0.362; judge-label distilled: `research/marsc_20260916/results/g3/b/eval_validation/e2_emitted.json` `judgefact.distilled.seedavg+div1` 0.303 (a G3 run of 2026-09-16; the same file's human-fact seed-average, 0.362, is reproduced by the post-repair A1 run) | |
| Val. rec@10, wrong / no summary | `S/b0/validation_gemma+ent/eval/e2_emitted.json` | `humanfact.gemma+ent@shuffled` 0.430, `@empty` 0.484 (H-J11) |
| TEST rec@10 and om. prec | `R/r12_precision/eval_test/e2_emitted.json` | all judge-label and human-fact rows, and the wrong/no-summary controls (0.440/0.666, 0.458/0.655) |
| TEST, MARS-2 rows | `R/r12_precision/eval_test_fill/e2_emitted.json` | `mars2.direct_enabled` 0.168/0.729, `mars2.direct_enabled+div1` 0.195/0.704 |
| UniSum rec@10 | `S/hj11/unisum_eval/eval/e2_emitted.json` (H-J11) | MARS-2 0.248/0.268, distilled judge-label 0.392 (consensus) / 0.401 (seed-avg), distilled human-fact 0.425, Gemma+ent human-fact 0.347 |
| UniSum rec@10, Gemma+ent judge-label and seed-avg human-fact | `R/r01_consensus/eval_unisum/e2_emitted.json` | `judgefact+div1` 0.298, `humanfact.single+div1` 0.389 |
| UniSum, wrong / no summary | `S/b3/eval_unisum_ctl/e2_emitted.json` | `b3.A.human.shuffled+div1` 0.356, `b3.A.human.empty+div1` 0.341 |
| human prec. (val.) | `S/d1d2/d1/analysis/d1b_adjudicated_precision.json` (study D1) | per system, `budgets["10"]`, adjudicated estimand |

### Table 3 — same-fact crossed endpoint (`tab:marsc_conditioning`) — verified

| Block | TEST / validation | UniSumEval |
|---|---|---|
| comparator family, 31B judge, MiniCheck-7B, FactCG, label-only twin (`twinjudge`) | `S/hj12/twin/crossed_test/primary_humanfact/crossed_families.json`, `.../crossed_validation/primary_humanfact/...` (H-J12a; earlier families first in `S/b2/crossed_*/` and `S/b12/{test,validation}/`) | `S/hj12/twin/crossed_unisum/primary_humanfact/crossed_families.json` (earlier families: `S/hj3/hcond_unisum_ext/`) |
| encoder rows (FactCG encoder, DeBERTa-xlarge) | `S/hj9/hcond_test_primary_humanfact/`, `S/hj9_xlarge/hcond_test_primary_humanfact/`; validation margins `S/hj11/hcond_validation/crossed_families.json` | `S/hj9/hcond_unisum_primary_humanfact/`, `S/hj9_xlarge/hcond_unisum_primary_humanfact/` |
| exact label-only contrast (`--primary twinhuman`) | `S/hj12/twin/crossed_{validation,test}/primary_twinhuman/` | `S/hj12/twin/crossed_unisum/primary_twinhuman/` |

The Gemma-4-31B judge row prints 0.903 on TEST; the JSON value is 0.90248 (0.902 at three decimals).

### Table 4 — agreement with human judgments, common population (`tab:mars_meta`) — verified

`M/mg/meta/meta_common.json` (M-G, `ma_meta.py --common-cohort`, rendered by
`research/mars_metric_20260923/code/mg_table.py`): `datasets.<set>.dims.<dimension>.<metric>.summary_tau.observed`,
with the per-column pair and document counts beside it. The available-case table (`tab:mars_meta_available`) and
the candidate-blind controls (`tab:mars_meta_controls`) come from `M/ma/meta/meta_evaluation.json` (M-A, rendered by
`code/ma_table.py`).

### H-J12 and M-G additions (audit response of 2026-09-28)

| Reading | File | Status |
|---|---|---|
| H-J12(a) label-only twin on the crossed endpoint | `S/hj12/twin/crossed_{validation,test,unisum}/primary_{humanfact,twinhuman}/crossed_families.json` | shipped |
| H-J12(b) twin through the lexical residual | `S/hj12/twin/residual_{test,unisum}/primary_{humanfact,twinhuman}/c2_crossed_residual.json` | shipped |
| H-J12(c) unit-matched judge twin | `S/hj12/gemma_twin/` (`sampled_units.json`, receipts, `eval_test/`, `eval_test_distilled/`, `crossed_{validation,test,unisum}/`) | shipped |
| H-J12(d) inventory x verifier grid, distilled column | `S/hj12/unisum_distilled_scores/{factcg,minicheck7b}/` receipts, `S/hj12/grid/eval/eval_test/e2_emitted.json`, `S/hj12/grid/eval/eval_unisum/e2_emitted.json`, `S/hj12/grid/eval/b3_contrasts.json` | shipped |
| H-J12(e) clinical random / lead-position baselines | `S/hj12/clinical/eval/b3_contrasts.json`, `.../eval_omb/e2_emitted.json`, `.../per_pair_omb_k10.jsonl` | shipped |
| M-G common-cohort metric table | `M/mg/meta/meta_common.json` | shipped |
| M-G pairwise-common (shared-pair) contrasts, Holm | `M/mg/meta/meta_pairwise.json` (job 7905) | shipped |
| M-G RoSE automatic-unit MARS-R | `M/mg/rose/` (receipts and manifest; the per-pair score files, mg/scores/scores_marsR_*_auto*.jsonl on the cluster, stay there like every score dump, and their values enter `M/mg/meta/meta_common.json`) | shipped |
| M-H (2026-09-29) cohort composition of Table 4 and the document-clustered LLM-AggreFact intervals | `research/mars_metric_20260923/results_snapshot/mh_cohort_composition.json`, `mh_aggrefact_cluster_bootstrap_{factcg,marsc}.json` (`code/mh_cohort.py`, `code/mh_cluster_boot.py`) | shipped |
| H-J13 (2026-09-29) emission-rule controls with the rule applied on top of each filter | `S/hj13/eval_validation/e2_emitted.json` (`research/marsc_ieee_20260922/slurm/hj13_rule_controls.sbatch`) | shipped |
| M-G screen | `M/mg/screen/mg_screen.json` | shipped |
| B3 label-source twins on the emitted endpoints, both arms (arm A: the deployed objective; arm R: natural cells only, identical cells under both label sets) | `S/b3/eval_{test,unisum}_{A,R}/e2_emitted.json`, `S/b3/analysis/b3_contrasts.json` | shipped |
| H-J14 (2026-09-29) replay of the training sampler of the twins, and the training receipts of the twelve checkpoints | `research/marsc_ieee_20260922/results_snapshot/hj14/sampler_replay.json`, `training_receipts.json` (`code/hj14_sampler_replay.py`) | in the repository; pending in the archive, which was built before the block ran |

### Other tables (from the registrations; not re-checked cell by cell)

| Table | Files |
|---|---|
| clinical crossed endpoint (`tab:marsc_omb`) | `S/hj10/hcond_complete_ext/`, `S/hj10/hcond_complete_w_ext/crossed_families.json`, `S/hj10/twin_complete_w.json` (`code/hj10_table.py`) |
| clinical end to end (`tab:marsc_omb_e2e`) | `S/hj10b/eval/`, `S/hj10b/eval_cmp/` (H-J10b), `S/hj12/clinical/eval/` (chance baselines) |
| UniSumEval comparators (`tab:marsc_unisum_comparators`) | `S/hj3/eval_unisum/` (H-J3b; `b3_contrasts.json`) |
| backbone transfer (`tab:marsc_backbone`) | `S/hj9/`, `S/hj9_xlarge/`, `S/hj9_joint/`, `S/hj11/hcond_validation/` |
| summary-blind controls (`tab:marsc_blindcontrols`) | `S/b0/{validation,test}_{gemma+ent,distilled}/eval/e2_emitted.json`, `S/b9b10_eval/` |
| risk-coverage (`tab:marsc_riskcoverage`) | `S/c1c3/` |
| deployment and cost (`tab:marsc_deploy`, `tab:marsc_cost`) | `S/hj5/deploy_cost_*.json`, `S/hj11/deploy/`, `S/hj11/cost/`, `S/b11/cost_table.json`, `S/b6b8/`, `R/g4/cost_table.json` |
| end-to-end package cost (`tab:marsc_cost_e2e`) | `M/me/` (M-E) |
| components (`tab:marsc_components`) | R series (`R/r01_consensus/`, `R/r04_matched/`, `R/r07_crossed/`) and `S/b3/`, `S/a2a3/` — unknown per cell; see `REPAIR_R_SERIES.md` and `RESULTS.md` |
| study D1 (`tab:d1`) | `S/d1d2/d1/analysis/d1b_adjudicated_precision.json`; items in `S/d1d2/d1/export.jsonl` with the key `human_sample_key.jsonl` |

## Study D1 files

`results/marsc_strengthen/d1d2/d1/` holds the instructions, the blinded sheet (`human_sample_blind.csv`, answer
columns empty), the key, the freeze record, the platform export and the analysis. The export names annotators only
as `Annotator 1`, `Annotator 2` and `Adjudicator`; `PRIVACY_CHECK.json` records the check. The completed offline
answer sheets are held by the authors (see `D1_PROVENANCE_2026-09-22.md`).
