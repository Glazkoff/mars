#!/usr/bin/env python3
"""B12 -- score B2's crossed H-COND cells with the three arms added this week, into B2's own score cache.

MARS_PUBLICATION_PLAN_2026-09-20.md §3.2 asks for the judge row "reported on recall@10, omission precision
**and H-COND**". The first two come out of `b9b10_eval.sbatch`. H-COND is a different endpoint on a different
keying: `mc_summary_conditioning.crossed_cells` indexes (document, summary, FACT) over the RoSE human ACUs,
not over the extracted inventory, so the B9/B10 score files cannot be reprojected onto it — the cells have to
be scored directly.

WHY THIS WRITES A CACHE FILE INSTEAD OF ADDING A FAMILY KIND TO b2_crossed_families.py
--------------------------------------------------------------------------------------
Two of the three new arms (the Gemma-4-31B judge and Bespoke-MiniCheck-7B) need vLLM and therefore
`.venv-vllm`; `b2_crossed_families.py` runs under `.venv-mars`. Rather than make B2 importable from two
environments, this script writes exactly the file B2's own `cached()` helper reads:

    <cache>/<safe(key)>.json  =  {"key": key, "cells": ["di|si|fi", ...], "z": [omission, ...]}

with `key = "<family>.zs.<arg>"` and the cells in `sorted(set(crossed_cells(docs)))` order. B2 then reports a
cache HIT and never calls its own scorer, so **B2 is not modified at all** and the estimator, the triples,
the document-macro statistic, the bootstrap, the chance verdict and the paired margin against MARS-C are the
frozen ones. Declare the family to B2 as `NAME=zs:<arg>`; the cache satisfies it.

The premise/claim contract is copied from `b2_crossed_families.score_cells_zs`: premise is the candidate
summary UNCLIPPED, claim is `msc.clip(fact, 96)`, omission = 1 - support. Any deviation would make the new
rows non-substitutable with the `nli` / `summac_zs` rows already in the table.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


msc = _load(REPO / "research" / "marsc_20260916" / "code" / "mc_summary_conditioning.py")
swap = _load(REPO / "research" / "marsc_strengthen_20260918" / "code" / "b10_verifier_swap.py")
CHUNK = 256


def build(a):
    """The same scorer objects the inventory arms used, so H-COND and recall@10 describe one system."""
    if a.system == "factcg":
        import torch
        dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        return swap.FactCG(a.snapshot, a.batch, dev)
    if a.system == "minicheck7b":
        return swap.MiniCheck7B(a.snapshot, a.max_model_len, a.gpu_memory_utilization, 1,
                                Path(a.minicheck_utils), a.trust_remote_code)
    if a.system == "judge":
        return JudgeScorer(a)
    raise SystemExit(f"[b12] unknown system {a.system!r}")


class JudgeScorer:
    """b9_judge_units' contract, exposed as .score(docs, claims) -> support, so B12 can treat the judge as
    one more substitutable verifier."""

    def __init__(self, a):
        b9 = _load(REPO / "research" / "marsc_strengthen_20260918" / "code" / "b9_judge_units.py")
        self.b9 = b9
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(a.snapshot, local_files_only=True, use_fast=True)
        probe = b9.apply_chat(self.tok, b9.PROMPT.format(summary="A short summary.", fact="A short fact."))
        if "<think>" in probe and not probe.endswith(b9.CLOSED_THINK):
            raise SystemExit("[b12] ABORT: enable_thinking=False did not reach the chat template")
        self.yes = b9.word_token_ids(self.tok, b9.YES)
        self.no = b9.word_token_ids(self.tok, b9.NO)
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=a.snapshot, tokenizer=a.snapshot, dtype="bfloat16",
                       gpu_memory_utilization=a.gpu_memory_utilization, max_model_len=a.max_model_len,
                       tensor_parallel_size=1, disable_log_stats=True, enable_prefix_caching=True,
                       trust_remote_code=False)
        # EXACT readout, as in b9: `logprob_token_ids` returns the logprob of each yes/no id whatever its
        # rank. The top-k read that was used here first maps a confident "no" to support exactly 0, which on
        # the crossed endpoint means z(f|s-) == z(f|s+) == 0 for both cells of a triple -- scored as a TIE,
        # i.e. half a win, on 60-99 % of units. That is not a measurement of the judge's conditioning.
        self.sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1,
                                 logprob_token_ids=sorted(self.yes | self.no))
        self.max_model_len = a.max_model_len
        self.n_undecided = 0

    def score(self, docs: list[str], claims: list[str]) -> list[float]:
        prompts = [self.b9.apply_chat(self.tok, self.b9.PROMPT.format(summary=d, fact=c))
                   for d, c in zip(docs, claims)]
        # b9 refuses rather than truncate; b12 feeds UNCLIPPED candidate summaries (score_cells_zs does not
        # clip the premise), so the same guard is needed here and not merely inherited.
        lens = [len(self.tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        over = sum(1 for n in lens if n >= self.max_model_len)
        print(f"[b12] judge prompt tokens: max {max(lens)} mean {sum(lens) / len(lens):.0f}, "
              f"{over} at/over {self.max_model_len}", flush=True)
        if over:
            raise SystemExit(f"[b12] ABORT: {over} prompts reach the context window; re-run with a larger "
                             f"--max-model-len (>= {max(lens) + 16}) rather than truncating the premise")
        # hand vLLM token ids, not the rendered string: the chat template may already carry a BOS and
        # re-tokenising the string would duplicate it (measured on the MiniCheck checkpoint)
        pids = [self.tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        outs = self.llm.generate([{"prompt_token_ids": x} for x in pids], self.sp)
        assert len(outs) == len(prompts), f"vLLM returned {len(outs)} outputs for {len(prompts)} cells"
        out, undecided = [], 0
        for o in outs:
            lp = o.outputs[0].logprobs
            s = self.b9.support_from_logprobs(lp[0], self.yes, self.no) if lp else None
            if s is None:
                # An imputed 0.5 would be a fabricated crossed-accuracy cell, and b9's own contract says a
                # unit with neither word in the top-k is "never silently mapped to 0.5". There is no NaN
                # channel into b2's cache, so refuse instead of inventing a value.
                undecided += 1
                s = 0.5
            out.append(s)
        self.n_undecided += undecided
        if undecided:
            raise SystemExit(f"[b12] ABORT: {undecided} cells returned no logprob for any yes/no id; b2's "
                             f"cache has no null channel, so these would enter H-COND as a fabricated 0.5")
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="e0 docs_<split>.jsonl")
    ap.add_argument("--system", required=True, choices=["factcg", "minicheck7b", "judge"])
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--family", required=True, help="the family NAME as it will be declared to b2")
    ap.add_argument("--arg", default="", help="the <arg> of NAME=zs:<arg>; defaults to --system")
    ap.add_argument("--cache", required=True, help="b2 --cache directory")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--minicheck-utils",
                    default="/home/user/.venv-mars/lib/python3.12/site-packages/minicheck/utils.py")
    a = ap.parse_args()
    arg = a.arg or a.system

    docs = [json.loads(l) for l in open(a.docs, encoding="utf-8")]
    cells, triples = msc.crossed_cells(docs)
    print(f"[b12] {len(docs)} docs -> {len(cells)} crossed cells, {len(triples)} same-fact triples", flush=True)

    prem = [docs[di]["summaries"][si]["candidate"] for di, si, _ in cells]
    hypo = [msc.clip(docs[di]["facts"][fi], 96) for di, _, fi in cells]

    sc = build(a)
    vals: list[float] = []
    t0 = time.time()
    for s in range(0, len(cells), CHUNK):
        vals.extend(sc.score(prem[s:s + CHUNK], hypo[s:s + CHUNK]))
        if (s // CHUNK) % 5 == 0:
            print(f"[b12]   {a.system} {min(s + CHUNK, len(cells))}/{len(cells)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    assert len(vals) == len(cells), f"{a.system}: {len(vals)} values for {len(cells)} cells"
    z = [1.0 - float(v) for v in vals]

    key = f"{a.family}.zs.{arg}"
    safe = re.sub(r"[^A-Za-z0-9._+-]", "_", key)[:180]
    d = Path(a.cache)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{safe}.json"
    f.write_text(json.dumps({"key": key, "cells": [f"{c[0]}|{c[1]}|{c[2]}" for c in cells], "z": z}))
    # a receipt beside the cache entry, so an H-COND row can be traced to the run that produced it the way
    # every B9/B10 score file can
    (d / f"{safe}.receipt.json").write_text(json.dumps({
        "family": a.family, "arg": arg, "system": a.system, "snapshot": a.snapshot,
        "docs": a.docs, "n_docs": len(docs), "n_cells": len(cells), "n_triples": len(triples),
        "n_undecided": getattr(sc, "n_undecided", 0),
        "n_floored_support_zero": int(sum(1 for v in z if v == 1.0)),
        "seconds": time.time() - t0, "cells_per_s": len(cells) / max(time.time() - t0, 1e-9),
    }, indent=1))
    print(f"[b12] wrote {f} ({len(cells)} cells, {time.time() - t0:.0f}s)", flush=True)
    print(f"[b12] declare to b2 as:  {a.family}=zs:{arg}", flush=True)


if __name__ == "__main__":
    main()
