# Untouched-pool feasibility census

Our source pool: 1291 unique documents.

## UniSumEval
- clone: ok
- licence: none found
- records: 4052 in 3 data files; records with a long text field: 3942 (220 unique)
- overlap with our sources: 1 exact, 0 near-duplicate (8-gram Jaccard ≥ 0.5, first 3000 texts)
- fact-like fields: {'fact_verification_label': 2025, 'fact_verification_error_type': 2025, 'pred_keyfact': 2025, 'keyfact_validation_label': 2025, 'keyfact': 2025, 'keyfact_label': 2025, 'completeness': 2025, 'sentence_label': 2025, 'machine_evaluation_results_completeness': 2025}
- files: data/unisumeval_faithfulness.jsonl (2025 rec; keys ['uid', 'doc_id', 'source', 'domain', 'length', 'input_type']); data/unisumeval_keyfact.jsonl (2025 rec; keys ['uid', 'doc_id', 'source', 'domain', 'length', 'input_type']); results/evaluators_benchmark_faithfulness.csv (2 rec; keys ['', 'source', 'pearson_corr', 'pearson_corr_pvalue', 'spearman_corr', 'spearmancorr_pvalue'])

## QAPyramid
- clone: ok
- licence: MIT License  Copyright (c) 2024 ShiyueZhang  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal 
- records: 2 in 2 data files; records with a long text field: 0 (0 unique)
- overlap with our sources: 0 exact, 0 near-duplicate (8-gram Jaccard ≥ 0.5, first 3000 texts)
- fact-like fields: {}
- files: raw_data/QA_generation.json (1 rec; keys ['e8fe5735b1f8b533062b0a8e7072b4d181f51d3f', 'e809facdb21e9df97dc3073fe922b021add14f0c', '01b0585316ca58505df1d4ace2d888d0e50e7600', '037b49be7f2d2f7b0b9eb48e7bc7e9f97ad2971d', '0593bf53f38aa56a4c6093e77558f9fb86f61d0a', '042aac3be278658a55b253138a43a2ad1d631ba2']); raw_data/QA_presence.json (1 rec; keys ['bart', 'pegasus', 'brio', 'brio-ext', 'matchsum', 'mixtral-8x22b-instruct-v0.1'])
