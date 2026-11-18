#!/usr/bin/env python3
"""Prepare a Bespoke-MiniCheck-7B snapshot that vLLM 0.29 can serve with its NATIVE InternLM2 implementation.

The published checkpoint is an InternLM2 model (`model_type: internlm2`) that carries `auto_map` entries for
bundled modelling and tokenizer code. Those entries force `trust_remote_code=True`, and the bundled
`InternLM2ForCausalLM.forward()` is missing the `intermediate_tensors` argument vLLM 0.29 passes, so the
engine dies during CUDA-graph profiling (job 5660). Without the flag vLLM refuses to parse the config at all
(job 5665).

vLLM implements InternLM2ForCausalLM natively, and the repository ships a serialized fast tokenizer
(`tokenizer.json`). So this builds a directory of symlinks to the real weights with the config's auto_map
reduced to its AutoConfig entry (transformers 5 has no native `internlm2` config, so removing it entirely
makes the checkpoint unparseable) and the tokenizer's auto_map removed. No weight is copied or altered.

Two checks make the substitution auditable rather than assumed:
  1. the sanitized tokenizer must ROUND-TRIP real text (encode then decode back to the input). The gate is
     deliberately NOT agreement with the checkpoint's own remote-code tokenizer: on this box that one fails
     to load its sentencepiece model and silently falls back to byte level, turning
     "Donald Sterling's ..." into 55 ids that decode to "D o n a l d  S t e r ...", so AGREEMENT with it
     would be evidence of a bug rather than of correctness;
  2. downstream, `b10_verifier_swap.py --validate-aggrefact` re-measures LLM-AggreFact BAcc against the
     published 77.4, with an absolute floor, before any inventory is scored.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SAMPLE = [
    "Donald Sterling's racist remarks cost him an NBA team last year.",
    "The company reported revenue of 1.2 billion dollars in 2023, up 14% year over year.",
    "Determine whether the provided claim is consistent with the corresponding document.",
    "As many as 108 students from the University of Nairobi were admitted to hospital on Sunday.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the real HF snapshot directory")
    ap.add_argument("--dst", required=True, help="sanitized directory to create")
    ap.add_argument("--skip-tokenizer-check", action="store_true")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)

    for f in sorted(src.iterdir()):
        if f.name in {"config.json", "tokenizer_config.json"}:
            continue
        link = dst / f.name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(f.resolve(), link)

    # config.json: KEEP the AutoConfig entry and drop the model entries.
    #   * transformers 5 has no native `internlm2` config, so without AutoConfig nothing can parse the file
    #     at all and vLLM dies in config validation (job 5674);
    #   * with AutoModelForCausalLM present, vLLM loads the checkpoint's bundled forward(), which is missing
    #     `intermediate_tensors` (job 5660).
    # Keeping exactly the config mapping lets the config parse through remote code while the MODEL resolves
    # through `architectures` to vLLM's native InternLM2ForCausalLM.
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    am = cfg.get("auto_map", {})
    kept = {k: v for k, v in am.items() if k == "AutoConfig"}
    cfg["auto_map"] = kept
    cfg.setdefault("architectures", ["InternLM2ForCausalLM"])
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[prep] config.json: auto_map {sorted(am)} -> {sorted(kept)}")

    # tokenizer_config.json: drop the remote tokenizer entirely and load the serialized tokenizer.json. The
    # bundled InternLM2Tokenizer fails to load its sentencepiece model here and silently degrades to byte
    # level (see the round-trip check below).
    tcfg = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
    tcfg.pop("auto_map", None)
    tcfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    (dst / "tokenizer_config.json").write_text(json.dumps(tcfg, indent=2))
    print("[prep] tokenizer_config.json: auto_map removed, tokenizer_class=PreTrainedTokenizerFast")

    print(f"[prep] model_type={cfg.get('model_type')} architectures={cfg.get('architectures')}")

    if not a.skip_tokenizer_check:
        # The gate is ROUND-TRIP correctness, not agreement with the remote-code tokenizer. On this box the
        # checkpoint's own `InternLM2TokenizerFast` fails to load its sentencepiece model and silently falls
        # back to byte level: "Donald Sterling's ..." becomes 55 ids that decode to "D o n a l d  S t e r ...".
        # The serialized tokenizer.json gives 14 ids and decodes back exactly. Agreement between the two would
        # therefore be evidence of a BUG, so each is judged on whether it reproduces its input.
        from transformers import AutoTokenizer
        new = AutoTokenizer.from_pretrained(str(dst), trust_remote_code=True, local_files_only=True)
        bad = 0
        for s in SAMPLE:
            ids = new.encode(s)
            back = new.decode(ids, skip_special_tokens=True)
            ok = back.strip() == s.strip()
            print(f"[prep] roundtrip {'ok ' if ok else 'BAD'} n_ids={len(ids):3d} {s[:46]!r}")
            bad += 0 if ok else 1
        if bad:
            raise SystemExit(f"[prep] ABORT: sanitized tokenizer fails to round-trip {bad}/{len(SAMPLE)}")
        print(f"[prep] sanitized tokenizer round-trips {len(SAMPLE)}/{len(SAMPLE)} samples")

    print(f"[prep] ready -> {dst}")


if __name__ == "__main__":
    main()
