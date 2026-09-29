# Third-party material and the works to cite

The MIT licence in `LICENSE` covers the code and the files produced by the authors of MARS. It does **not** cover
the third-party texts listed here, which stay under the licences of their sources. The licence texts are in
`licenses/`.

## Texts reproduced in this repository

Three blinded annotation sheets reproduce source documents and system summaries of the **RoSE** benchmark, as
RoSE packages them and with no edit to their wording, so that the human studies can be re-read item by item:

| File | Items | Source documents |
|---|---:|---|
| `results/marsc_strengthen/d1d2/d1/human_sample_blind.csv` (study D1) | 500 | CNN/DailyMail 174, XSum 130, SAMSum 196 |
| `research/marsc_20260916/results/g6/human_sample_blind.jsonl` (pilot) | 300 | CNN/DailyMail 130, XSum 74, SAMSum 96 |
| `research/marsc_20260916/results/g6b/human_sample_blind.jsonl` (pilot) | 100 | CNN/DailyMail 55, XSum 28, SAMSum 17 |

In each item the fields `source` and `summary` are third-party text; the field `fact` is a short statement produced
by the MARS fact inventory from the source. The `pair_id` of the key files (`human_sample_key*.jsonl`) names the
RoSE record of every item, for example `rose:cnndm:<document id>:<system>`.

| Material | Licence | Terms that apply here |
|---|---|---|
| RoSE: annotations, system summaries and the documents as packaged by RoSE | BSD 3-Clause, Copyright (c) 2023, Salesforce.com, Inc. (`licenses/RoSE-BSD-3-Clause.txt`) | the copyright notice, the conditions and the disclaimer are retained; the names of the copyright holder and contributors are not used to endorse this work |
| CNN/DailyMail documents | Apache License 2.0 as distributed (`licenses/Apache-2.0.txt`) | the licence is included; the documents are unmodified. The articles were published by CNN and the Daily Mail |
| XSum documents | the XSum repository is MIT-licensed, Copyright (c) 2018 Shashi Narayan (`licenses/XSum-MIT.txt`); the dataset card on the Hugging Face hub states no licence for the articles, which were published by the BBC | the notice is retained; the documents are unmodified |
| SAMSum dialogues | CC BY-NC-ND 4.0 (`licenses/SAMSum-CC-BY-NC-ND-4.0.txt`) | attribution given; **non-commercial use only; no modified versions may be distributed**; the dialogues are unmodified |

## Datasets used but not reproduced

Result files quote identifiers, scores and aggregate statistics of the following datasets, and no text of theirs:
UniSumEval, SummEval, OmissionBench (CC BY 4.0) and LLM-AggreFact (CC BY-ND 4.0). Obtain them from their
publishers.

## Works to cite

If you use the material above, cite the original works as well as MARS.

- **RoSE.** Y. Liu, A. R. Fabbri, P. Liu, Y. Zhao, L. Nan, R. Han, S. Han, S. Joty, C.-S. Wu, C. Xiong and D. Radev,
  "Revisiting the Gold Standard: Grounding Summarization Evaluation with Robust Human Evaluation," in *Proceedings
  of ACL*, 2023. `https://github.com/Yale-LILY/ROSE`
- **CNN/DailyMail.** K. M. Hermann, T. Kočiský, E. Grefenstette, L. Espeholt, W. Kay, M. Suleyman and P. Blunsom,
  "Teaching Machines to Read and Comprehend," in *Advances in Neural Information Processing Systems*, 2015; and
  A. See, P. J. Liu and C. D. Manning, "Get To The Point: Summarization with Pointer-Generator Networks," in
  *Proceedings of ACL*, 2017, pp. 1073–1083, doi:10.18653/v1/P17-1099.
  `https://huggingface.co/datasets/abisee/cnn_dailymail`
- **XSum.** S. Narayan, S. B. Cohen and M. Lapata, "Don't Give Me the Details, Just the Summary! Topic-Aware
  Convolutional Neural Networks for Extreme Summarization," in *Proceedings of EMNLP*, 2018, pp. 1797–1807,
  doi:10.18653/v1/D18-1206. `https://github.com/EdinburghNLP/XSum`
- **SAMSum.** B. Gliwa, I. Mochol, M. Biesek and A. Wawer, "SAMSum Corpus: A Human-annotated Dialogue Dataset for
  Abstractive Summarization," in *Proceedings of the 2nd Workshop on New Frontiers in Summarization*, 2019,
  pp. 70–79, doi:10.18653/v1/D19-5409. `https://huggingface.co/datasets/knkarthick/samsum`
- **UniSumEval.** Y. Lee, T. Yun, J. Cai, H. Su and H. Song, "UniSumEval: Towards Unified, Fine-Grained,
  Multi-Dimensional Summarization Evaluation for LLMs," in *Findings of EMNLP*, 2024.
- **SummEval.** A. R. Fabbri, W. Kryściński, B. McCann, C. Xiong, R. Socher and D. Radev, "SummEval: Re-evaluating
  Summarization Evaluation," *Transactions of the Association for Computational Linguistics*, vol. 9,
  pp. 391–409, 2021.
- **OmissionBench.** Composo AI, "OmissionBench: single-error note pairs for omission detection in AI clinical
  notes, version factorial-v1," 2026. `https://huggingface.co/datasets/ComposoAI/OmissionBench`
- **LLM-AggreFact.** L. Tang, P. Laban and G. Durrett, "MiniCheck: Efficient Fact-Checking of LLMs on Grounding
  Documents," in *Proceedings of EMNLP*, 2024, pp. 8818–8847; and L. Tang, T. Goyal, A. R. Fabbri, P. Laban, J. Xu,
  S. Yavuz, W. Kryściński, J. F. Rousseau and G. Durrett, "Understanding Factual Errors in Summarization: Errors,
  Summarizers, Datasets, Error Detectors," in *Proceedings of ACL*, 2023, pp. 11626–11644.
  `https://huggingface.co/datasets/lytang/LLM-AggreFact`
