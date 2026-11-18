# MARS: fact-level coverage and faithfulness scoring, with a reference-free omission finder

Code, analysis plans and result files for the manuscript

> **MARS: A Fact-Level Coverage Metric and Reference-Free Omission Detector Trained on Natural Human Coverage Labels**
> Nikita Glazkov, Olga Volkova, Konstantin Pchelin, Ivan Nasonov, Mikhail Mozikov, Daniil Sukhorukov,
> Iaroslav Bespalov, Ilya Makarov and Dmitry V. Dylov, 2026.

MARS decomposes a text into atomic facts, scores each fact with a cross-encoder verifier and returns two mean
fact-level support scores with per-fact evidence:

- **MARS-R** (coverage): support of the reference's, or the source's salient, facts by the candidate;
- **MARS-P** (faithfulness): support of the candidate's own facts by the source.

Its coverage verifier, **MARS-C**, is trained on the coverage labels annotators gave to reference facts, and the
omission finder built on it emits the source facts a summary most likely dropped, with no reference at test time.

This repository is generated automatically from the development repository. Host names, private paths, user
names and e-mail addresses in the code, plans and result files are replaced by placeholders (`cluster-A`,
`cluster-B`, `/home/user`, `ANONYMIZED`); nothing else in those files is edited. `MANIFEST.txt` lists the SHA-256 of
every file.

## What is here

| Path | Content |
|---|---|
| `mars/` | the MARS package: `mars.score`, the verifier contracts, the decomposers, a command-line entry point, tests |
| `research/marsc_20260916/` | the omission finder: analysis plan, repair register, training and evaluation code, job scripts, compact results |
| `research/marsc_strengthen_20260918/` | candidate-conditioning, summary-blind and matched-label controls, the blinded human study D1: plan, results record, code |
| `research/marsc_ieee_20260922/` | later blocks of the journal version (other encoders, the clinical pool, twins, sampler replay) |
| `research/mars_metric_20260923/` | MARS as an evaluation metric: plan, meta-evaluation code, result snapshot |
| `research/mars2_gates_20260916/`, `research/mars3_20260916/` | supporting code the scripts import: the counter suite, fact inventories, scorers |
| `results/` | the result files the manuscript's numbers are read from (evaluator outputs, contrasts, receipts, study D1 keys and labels) |
| `RESULTS_MAP.md` | which file holds which table |
| `registrations/` | number-by-number provenance of the earlier conference-version figures |

The analysis plans are internal dated documents, not entries in an external registry. Every block of experiments
was written into its plan, with endpoints and criteria, before its data were read; outcomes, including the criteria
that were missed, are recorded in the same files.

## What is not here

- **Trained checkpoints.** The three MARS-C seeds, the unified verifier and the distilled segmenter are not
  distributed. Their recipe, training manifests and hashes are in `research/marsc_20260916/` and `results/`;
  retraining a MARS-C seed takes about a minute of GPU time after the labels are prepared.
- **Blinded annotation sheets.** They contain the full source documents of RoSE (CNN/DailyMail, XSum, SAMSum).
  The item keys (`human_sample_key.jsonl`) and the labels (`export.jsonl`) are shipped; a sheet is rebuilt by
  joining a key with RoSE on `pair_id`. The annotators appear only as `Annotator 1`, `Annotator 2` and
  `Adjudicator`.
- **Per-item score dumps** and third-party dataset copies.

## Using the package

```bash
pip install -e ".[models]"
```

```python
from mars import MarsScorer

scorer = MarsScorer.default(r_glob="ckpts/marsc_seed*")   # FactCG contract for P, MARS-C seeds for R
out = scorer.score(candidates, sources=sources, references=references, controls=True)
out[0].P, out[0].R, out[0].unsupported, out[0].omitted, out[0].controls
```

`MarsScorer.default(r_from_p=True)` scores MARS-R with the FactCG verifier instead, which needs no MARS-C
checkpoint. `python -m mars score --help` describes the command-line entry point. The ranked top-k omission
finder of the manuscript is research code (`research/marsc_20260916/code/`: `mc_score_units.py`,
`mc_diversity.py`, `mc_e2_eval.py`), not part of `mars.score`.

```bash
pip install -e ".[test]" && python -m pytest mars/tests tests/test_mars_api.py
```

## Data and licences

The code and the files produced by the authors are released under the MIT licence (`LICENSE`). RoSE, UniSumEval,
SummEval, OmissionBench and LLM-AggreFact are public datasets used under their own licences and are not
redistributed here; result files quote identifiers, scores and short fact strings derived from them.

## Citation

See `CITATION.cff`.
