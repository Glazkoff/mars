#!/usr/bin/env python3
"""B9 -- the zero-shot LLM judge as an OMISSION FINDER on the emitted endpoint.

MARS_PUBLICATION_PLAN_2026-09-20.md Sec. 3.2, first row: the paper already reports a 0.4B verifier beating a
31B zero-shot judge on *supplied-ACU* coverage AUROC (0.974 vs 0.955). That is the clean endpoint, but it is
not the endpoint the detector is sold on. Every reviewer asks the same question next: does the small verifier
still win when the judge is asked to do the actual job -- rank the inventory and emit the k facts it believes
were dropped? This file supplies that row.

WHAT IS HELD FIXED SO THE JUDGE ROW IS SUBSTITUTABLE
-----------------------------------------------------
The judge is a VERIFIER SWAP and nothing else. It is handed the same frozen inventory, the same premise under
the same registered candidate mode, and the same claim rendering as every other off-the-shelf family in
`b0_zs_score_units.py`:

  * rows        -- `r.get("units")` rows of the frozen inventory file, file order
  * premise     -- `mc_score_units.premises`, IMPORTED not re-implemented, so the `shuffled` derangement is
                   byte-for-byte the one the cross-encoder families saw (seeded rotation within
                   (resource, system)); `real` / `shuffled` / `empty` differ ONLY in the premise
  * claim       -- `common.render_claim(source, unit)`, capped at --claim-words, the family-native rendering
  * output      -- omission = 1 - support, via `common.write_scores`, system name `nli:judge_<tag>` (+ the
                   `_shuf` / `_empty` mode suffix b21_zeroshot writes), so `mc_diversity.py` ->
                   `mc_e2_eval.py` consume this arm without a line of evaluator code being touched

So recall@10, omission precision and H-COND for the judge are computed by the SAME evaluators, from the same
inventory, as the numbers already in the manuscript. Nothing downstream knows this arm came from an LLM.

SUPPORT IS READ, NOT GENERATED -- AND IT IS READ EXACTLY
--------------------------------------------------------
A generated "yes"/"no" gives one bit and no ranking, and an emission rule needs an ORDER over the inventory.
So the judge is scored, not sampled: support = P(yes) / (P(yes) + P(no)) over the casings of the two words,
greedy at temperature 0, so the arm is deterministic.

HOW that probability is obtained matters more than it looks. `--score-mode topk` (the original, kept only so
the withdrawn runs reproduce) reads the top-k first-token distribution and sums the mass of whichever
variants appear in it. When the model is confident, "yes" is not in the top 20 at all, its mass reads as
exactly zero, and the unit joins a tied block at the MAXIMUM omission score -- on sealed TEST, 69 % of units
under a real candidate and 99.96 % under a shuffled one. Those blocks are where the top-10 emission is drawn
from, so recall@10 was measuring the emission rule's tie-break, not the judge. Depth is not the cure: the
flooring falls only 62 % -> 58 % between top-20 and top-1000, because the probability genuinely is
negligible. Truncating it to zero is what destroys the ORDER.

`--score-mode exact` (the default) instead asks vLLM for the logprobs of the yes/no ids BY NAME via
`logprob_token_ids`, whatever their rank, so a probability of 1e-9 stays 1e-9 and the ranking is total, at
one request per unit and full throughput. A unit for which no readout returns is written null and counted in
`n_undecided`, never silently mapped to 0.5; the size of the tied block is recorded in every receipt as
`n_floored_support_zero`.

PROMPT ORDER IS A THROUGHPUT DECISION, NOT A SCORING ONE
--------------------------------------------------------
The summary is placed BEFORE the fact in the user turn so that every unit of a pair shares the longest
possible prompt prefix. With vLLM's automatic prefix caching this turns ~44 full prefills per pair into one
prefill plus ~44 short suffixes. It changes no score: the model still conditions on both fields, and the same
order is used in every candidate mode and for every model.

THE <think> TRAP, GUARDED (prereg standing rule 7)
--------------------------------------------------
`apply_chat_template` swallows unknown kwargs, so a TypeError fallback is not proof that
`enable_thinking=False` took effect -- the rendered prefix is. For a model whose template emits a think block
this job REFUSES unless the rendered prompt ends in a closed, empty one. A reasoning block would push the
answer past a 1-token window and return support for nothing at all.

Sealed TEST: scoring `--units test=...` re-reads the held-out split, so the sbatch gates on MC_TEST_REREAD=yes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402

CLOSED_THINK = "<think>\n\n</think>\n\n"

# The judgement contract. Frozen here and hashed into the receipt; changing it changes the arm.
PROMPT = (
    "You are checking whether a summary preserved a specific fact from the source document.\n\n"
    "Summary:\n{summary}\n\n"
    "Fact from the source document:\n{fact}\n\n"
    "Does the summary state or clearly imply this fact? Answer with one word, yes or no."
)

YES = ["yes", "Yes", "YES", " yes", " Yes", " YES"]
NO = ["no", "No", "NO", " no", " No", " NO"]


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


msu = _load(REPO / "research" / "marsc_20260916" / "code" / "mc_score_units.py")


def apply_chat(tok, prompt: str) -> str:
    """Chat-format with thinking disabled. Same contract as d2_revise.apply_chat / m32_decompose.judge."""
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def support_from_logprobs(entry, yes_ids: set[int], no_ids: set[int]) -> float | None:
    """P(yes) / (P(yes) + P(no)) over the first-token distribution; None if neither word is in the top-k."""
    py = pn = 0.0
    for tid, lp in entry.items():
        p = math.exp(lp.logprob)
        if tid in yes_ids:
            py += p
        elif tid in no_ids:
            pn += p
    tot = py + pn
    return None if tot <= 0.0 else py / tot


def word_token_ids(tok, words: list[str]) -> set[int]:
    """First-token ids for each surface form, both as a bare string and after a leading space."""
    ids: set[int] = set()
    for w in words:
        for s in (w, w.lstrip()):
            enc = tok.encode(s, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
    return ids


def exact_support(llm, sp_exact, prompt_ids: list[list[int]], yes_ids: set[int], no_ids: set[int]):
    """support = sum P(yes variant) / (sum P(yes) + sum P(no)), read EXACTLY rather than from a top-k list.

    Why this exists. The top-k reader sums the mass of whichever yes/no variants appear in the first-token
    distribution vLLM returns. When the model is confident, "yes" is not in the top 20 at all, its mass
    reads as exactly zero, and the unit joins a tied block at the MAXIMUM omission score. On sealed TEST
    that block was 69 % of units under a real candidate and 99.96 % under a shuffled one -- and the top-10
    emission is drawn from exactly that block, so recall@10 was ordering by the emission rule's tie-break
    rather than by the judge. Depth is not the cure: flooring falls only 62 % -> 58 % between top-20 and
    top-1000, because the probability genuinely is negligible. TRUNCATING IT TO ZERO is what destroys the
    order.

    `logprob_token_ids` asks vLLM for the logprobs of a named set of ids regardless of their rank -- the
    parameter exists for precisely this ("scoring tasks where you want to compare probabilities of specific
    label tokens"). One request per unit, exact values however small, total order. An earlier attempt forced
    each word as a continuation and read `prompt_logprobs`; it was equally exact but 25x slower (8.6 vs
    220 units/s), because prompt_logprobs re-scores the whole prompt.
    """
    import math as _m
    outs = llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids], sp_exact)
    assert len(outs) == len(prompt_ids), f"vLLM returned {len(outs)} outputs for {len(prompt_ids)} units"
    sup: list[float | None] = []
    missing = 0
    for o in outs:
        lps = o.outputs[0].logprobs
        entry = lps[0] if lps else None
        if not entry:
            missing += 1
            sup.append(None)
            continue
        py = sum(_m.exp(entry[t].logprob) for t in yes_ids if t in entry)
        pn = sum(_m.exp(entry[t].logprob) for t in no_ids if t in entry)
        tot = py + pn
        sup.append(None if tot <= 0.0 else py / tot)
    return sup, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", required=True, help="name=path inventory files")
    ap.add_argument("--model-id", required=True, help="hub id, recorded in the receipt")
    ap.add_argument("--model-snapshot", required=True, help="local snapshot directory actually loaded")
    ap.add_argument("--tag", required=True, help="short system tag, e.g. gemma31 -> nli:judge_gemma31")
    ap.add_argument("--candidate-mode", choices=["real", "shuffled", "empty"], default="real")
    ap.add_argument("--shuffle-seed", type=int, default=20260917,
                    help="must match the cross-encoder control arms or the shuffled columns are not paired")
    ap.add_argument("--empty-text", default="")
    ap.add_argument("--scores-out", required=True)
    ap.add_argument("--claim-words", type=int, default=96)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--logprobs", type=int, default=20)
    ap.add_argument("--score-mode", choices=["exact", "topk"], default="exact",
                    help="exact: ask vLLM for the logprobs of the yes/no ids by name, whatever their rank "
                         "(total order, no truncation-to-zero). topk: the original first-token top-k read, "
                         "kept only so the withdrawn runs can be reproduced.")
    ap.add_argument("--shard", default="", help="i/n -- score only shard i of n, split over PAIRS")
    ap.add_argument("--limit", type=int, default=0, help="pilot only: first N pairs")
    a = ap.parse_args()

    out = Path(a.scores_out)
    out.mkdir(parents=True, exist_ok=True)
    mode = a.candidate_mode
    suffix = {"real": "", "shuffled": "_shuf", "empty": "_empty"}[mode]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_snapshot, local_files_only=True, use_fast=True)
    if not getattr(tok, "chat_template", None):
        raise SystemExit("[b9] ABORT: tokenizer has no chat template; the decoding contract is a chat contract")

    probe = apply_chat(tok, PROMPT.format(summary="A short summary.", fact="A short fact."))
    if "<think>" in probe and not probe.endswith(CLOSED_THINK):
        raise SystemExit("[b9] ABORT: enable_thinking=False did not reach the chat template -- the rendered "
                         f"prompt does not end in a closed empty think block. tail={probe[-120:]!r}")
    print(f"[b9] chat template ok (think-block guard {'passed' if '<think>' in probe else 'n/a'})", flush=True)

    yes_ids = word_token_ids(tok, YES)
    no_ids = word_token_ids(tok, NO)
    if not yes_ids or not no_ids:
        raise SystemExit("[b9] ABORT: could not resolve yes/no token ids for this tokenizer")
    print(f"[b9] yes ids {sorted(yes_ids)} no ids {sorted(no_ids)}", flush=True)

    if yes_ids & no_ids:
        raise SystemExit(f"[b9] ABORT: yes and no share token ids {sorted(yes_ids & no_ids)}")
    print(f"[b9] score-mode={a.score_mode}", flush=True)

    from vllm import LLM, SamplingParams
    import vllm
    t0 = time.time()
    llm = LLM(model=a.model_snapshot, tokenizer=a.model_snapshot, dtype="bfloat16",
              gpu_memory_utilization=a.gpu_memory_utilization, max_model_len=a.max_model_len,
              tensor_parallel_size=a.tensor_parallel_size, disable_log_stats=True,
              enable_prefix_caching=True, trust_remote_code=False)
    print(f"[b9] vLLM {vllm.__version__} engine up in {time.time() - t0:.1f}s", flush=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, logprobs=a.logprobs)
    sp_exact = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1,
                              logprob_token_ids=sorted(yes_ids | no_ids))

    for spec in a.units:
        uname, path = spec.split("=", 1)
        rows = [r for r in common.load_jsonl(Path(path)) if r.get("units")]
        if a.limit:
            rows = rows[:a.limit]
        # Premises are built before SHARDING so the derangement sees the whole shard-set, exactly as the
        # cross-encoder controls saw it. Note it is built AFTER --limit, matching b0_zs_score_units.py:
        # under --limit the derangement is over the truncated pool and is therefore NOT paired with the
        # full-inventory control arms. --limit is a pilot switch only; no reported number uses it.
        prem = msu.premises(rows, a)
        if a.shard:
            i, n = (int(x) for x in a.shard.split("/"))
            keep = [k for k in range(len(rows)) if k % n == i]
            rows = [rows[k] for k in keep]
            prem = [prem[k] for k in keep]
            print(f"[b9] shard {i}/{n}: {len(rows)} pairs", flush=True)

        flat = [(i, j, " ".join(common.render_claim(r["source"], u).split()[:a.claim_words]))
                for i, r in enumerate(rows) for j, u in enumerate(r["units"])]
        print(f"[b9] {uname} mode={mode}: {len(rows)} pairs, {len(flat)} units, model={a.model_id}", flush=True)

        prompts = [apply_chat(tok, PROMPT.format(summary=prem[i], fact=c)) for i, _, c in flat]

        # A prompt over the context window is a hard vLLM error, and truncating one silently would change the
        # claim the judge is answering about. So measure first and refuse with a number, not a stack trace.
        lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        over = sum(1 for n in lens if n >= a.max_model_len)
        print(f"[b9] prompt tokens: max {max(lens)} mean {sum(lens) / len(lens):.0f}, "
              f"{over} at/over max_model_len={a.max_model_len}", flush=True)
        if over:
            raise SystemExit(f"[b9] ABORT: {over} prompts reach the context window; re-run with "
                             f"B9_MAX_LEN >= {max(lens) + 16} rather than truncating the claim")
        t1 = time.time()
        n_missing = 0
        if a.score_mode == "exact":
            # tokenise here, add_special_tokens=False, and hand vLLM ids -- the rendered template may already
            # carry a BOS and re-tokenising the string would duplicate it
            pids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
            support, n_missing = exact_support(llm, sp_exact, pids, yes_ids, no_ids)
        else:
            outs = llm.generate(prompts, sp)
            # the contract check b0_zs_score_units.py and b10_verifier_swap.py both make: a short or
            # reordered result list would misalign every score after the gap when zipped below, silently
            assert len(outs) == len(flat), f"vLLM returned {len(outs)} outputs for {len(flat)} units"
            support = [support_from_logprobs(o.outputs[0].logprobs[0], yes_ids, no_ids)
                       if o.outputs[0].logprobs else None for o in outs]
        el = time.time() - t1
        print(f"[b9] {len(flat)} judgements in {el:.0f}s ({len(flat) / max(el, 1e-9):.1f} units/s)", flush=True)

        n_undecided = sum(1 for s in support if s is None)
        n_floored = sum(1 for s in support if s == 0.0)
        # the tied-block size is the number that decides whether recall@k describes the judge or the
        # emission rule's tie-break, so it is printed and put in the receipt rather than left implicit
        print(f"[b9] support==0 exactly: {n_floored}/{len(support)} "
              f"({n_floored / max(len(support), 1):.2%}); logprob readouts missing: {n_missing}",
              flush=True)
        decided = [s for s in support if s is not None]
        print(f"[b9] undecided {n_undecided}/{len(support)}; support mean "
              f"{(sum(decided) / max(len(decided), 1)):.4f}", flush=True)

        per = {r["pair_id"]: [np.nan] * len(r["units"]) for r in rows}
        for (i, j, _), s in zip(flat, support):
            per[rows[i]["pair_id"]][j] = np.nan if s is None else 1.0 - s
        sysname = f"nli:judge_{a.tag}"
        part = f".part{a.shard.replace('/', 'of')}" if a.shard else ""
        fn = f"{uname}_{sysname.replace(':', '_')}_{mode}{part}.jsonl"
        common.write_scores(out / fn, f"{sysname}{suffix}", uname, per)
        print(f"[b9] wrote {fn}", flush=True)

        receipt = {"model_id": a.model_id, "model_snapshot": a.model_snapshot, "tag": a.tag,
                   "candidate_mode": mode, "shuffle_seed": a.shuffle_seed, "unit_set": uname,
                   "units_file": path, "n_pairs": len(rows), "n_units": len(flat),
                   "n_undecided": n_undecided, "n_floored_support_zero": n_floored,
                   "n_forced_logprobs_missing": n_missing, "score_mode": a.score_mode,
                   "yes_ids": sorted(yes_ids), "no_ids": sorted(no_ids), "shard": a.shard or None,
                   "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                   "claim_words": a.claim_words, "max_model_len": a.max_model_len,
                   "vllm": vllm.__version__, "seconds": el, "units_per_s": len(flat) / max(el, 1e-9),
                   "system": f"{sysname}{suffix}"}
        (out / (fn.replace(".jsonl", ".receipt.json"))).write_text(json.dumps(receipt, indent=1))

    print("[b9] done", flush=True)


if __name__ == "__main__":
    main()
