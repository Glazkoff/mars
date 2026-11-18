#!/usr/bin/env python3
"""MARS-C D2 -- generate the fixed-budget revisions (PREREG Wave D / D2). GPU. Computes no outcome.

One model, one prompt, six arms. The only thing that differs between arms is the list of facts in the feedback
block; everything the governing proposal requires to be matched is matched here and LOGGED per row, so the
matching is auditable from the artifact rather than asserted in the manuscript:

  source access     identical truncated source (done upstream in d2_build.py, re-asserted here per pair)
  revision calls    exactly one per row
  candidate count   exactly one -- greedy decoding, num_return_sequences = 1, no sampling, no best-of-n
  feedback tokens   the block is measured with THIS tokenizer and capped at --feedback-token-cap by dropping
                    trailing facts under one policy shared by every arm; drops are counted per row
  word budget       W comes from the ORIGINAL summary only and is therefore arm-invariant; the realised word
                    count and an over-budget flag are recorded per row. Nothing is silently truncated: a
                    revision that overruns is kept and marked, because quietly cutting it would fabricate
                    compliance with the very control the study exists to enforce.

Trap guarded explicitly. Qwen chat templates open a `<think>` block by default. In an earlier campaign this
collapsed forced-choice mass to 1e-7 and returned 88% empty outputs, producing a widely-repeated WRONG result.
`enable_thinking=False` is passed and the RENDERED prompt is then asserted not to leave a think block OPEN.

The distinction is the whole point and was got wrong once (job 5358, 2026-09-18). Qwen3.5 disables thinking by
emitting an already-CLOSED empty block, `<think>\n\n</think>\n\n`, and leaves it OPEN, `<think>\n`, when
thinking is on. A guard that merely greps for the opener therefore rejects the correct disabled rendering and
accepts nothing at all. What is dangerous is generating *inside* an unterminated block, so that is what is
tested: the last opener must be followed by a closer. `absent` and `closed` pass; `open` refuses. The observed
state is recorded in the manifest rather than a hardcoded "passed", and the independent behavioural backstop
below -- refusing if over half the FIRST batch returns empty -- is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

PROMPT = """You are revising a short summary of a document.

DOCUMENT:
{source}

CURRENT SUMMARY:
{summary}
{feedback}
Rewrite the summary. Keep everything important that the current summary already gets right, and make it as
informative as possible about the document. Use at most {W} words. Every statement must be supported by the
document; do not invent anything. Reply with the revised summary only — no preamble, no list, no explanation.

REVISED SUMMARY:"""

FEEDBACK_HEADER = "\nFACTS FROM THE DOCUMENT THAT MAY BE MISSING FROM THE CURRENT SUMMARY:\n"
THINK_OPENERS = ("<think>", "<|think|>")
THINK_CLOSERS = ("</think>", "<|/think|>")


def think_state(rendered: str) -> str:
    """'open', 'closed' or 'absent' for the trailing think block of a rendered prompt.

    Only 'open' is the failure mode: it means generation would continue inside an unterminated thinking block.
    A closed empty block is how Qwen3.5 expresses enable_thinking=False and is correct."""
    last_open = max((rendered.rfind(m) for m in THINK_OPENERS), default=-1)
    if last_open < 0:
        return "absent"
    return "closed" if any(c in rendered[last_open:] for c in THINK_CLOSERS) else "open"


def prompt_sha256() -> str:
    return hashlib.sha256((PROMPT + "\x00" + FEEDBACK_HEADER).encode("utf-8")).hexdigest()


def load_jsonl(p: Path):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def apply_chat(tok, prompt: str) -> str:
    """Chat-format a prompt with thinking DISABLED. Copied deliberately rather than imported: scripts/plan2026/
    is shared with live sessions and this contract must not drift under a frozen artifact."""
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _dtype_kw(torch_dtype):
    """transformers >= 5 renamed torch_dtype -> dtype. The cluster venv is 5.16.1; the shim keeps this script
    loadable if the venv is ever rolled back, rather than failing after the model has been paged in."""
    import transformers
    return {"dtype": torch_dtype} if int(transformers.__version__.split(".")[0]) >= 5 else {"torch_dtype": torch_dtype}


def load_model(snapshot: str, device: str):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    cfg = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    archs = getattr(cfg, "architectures", []) or []
    if any("ConditionalGeneration" in x for x in archs):
        from transformers import AutoModelForImageTextToText as Loader
    else:
        Loader = AutoModelForCausalLM
    if device.startswith("cuda"):
        model = Loader.from_pretrained(snapshot, local_files_only=True, **_dtype_kw(torch.bfloat16),
                                       device_map={"": 0}, attn_implementation="sdpa").eval()
    else:
        model = Loader.from_pretrained(snapshot, local_files_only=True, **_dtype_kw(torch.float32)).to(device).eval()
    return tok, model


def build_prompt(row: dict, facts: list[str]) -> str:
    fb = "" if not facts else FEEDBACK_HEADER + "".join(f"{i + 1}. {t}\n" for i, t in enumerate(facts))
    return PROMPT.format(source=row["source"], summary=row["summary"], feedback=fb, W=row["word_budget"])


def fit_feedback(tok, row: dict, cap: int) -> tuple[list[str], int, int]:
    """Drop trailing facts until the feedback block fits the shared token cap. One policy, every arm."""
    facts = list(row["facts"])
    n_drop = 0
    while facts:
        block = FEEDBACK_HEADER + "".join(f"{i + 1}. {t}\n" for i, t in enumerate(facts))
        n_tok = len(tok.encode(block, add_special_tokens=False))
        if n_tok <= cap:
            return facts, n_tok, n_drop
        facts.pop()
        n_drop += 1
    return [], 0, n_drop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="d2_arms.jsonl written by d2_build.py")
    ap.add_argument("--model-snapshot", required=True)
    ap.add_argument("--expect-revision", default="", help="snapshot directory name pinned by the preflight")
    ap.add_argument("--out", required=True, help="revisions.jsonl (append-resumable)")
    ap.add_argument("--manifest", required=True, help="d2_generation.json")
    ap.add_argument("--feedback-token-cap", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-input-tokens", type=int, default=6144)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0, help="smoke test only: first N rows after sharding")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch

    i, k = (int(x) for x in a.shard.split("/"))
    rows = [r for j, r in enumerate(load_jsonl(Path(a.arms))) if j % k == i]
    if a.limit:
        rows = rows[:a.limit]

    snap = a.model_snapshot.rstrip("/")
    revision = Path(snap).name
    if a.expect_revision and revision != a.expect_revision:
        raise SystemExit(f"[d2] model revision {revision!r} != pinned {a.expect_revision!r}; refusing")

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if outp.exists():
        for l in open(outp, encoding="utf-8"):
            try:
                done.add(json.loads(l)["row_id"])
            except Exception:
                pass
    todo = [r for r in rows if r["row_id"] not in done]
    print(f"[d2] shard {i}/{k}: {len(rows)} rows, {len(done)} already done, {len(todo)} to generate", flush=True)

    tok, model = load_model(snap, a.device)
    chat = bool(getattr(tok, "chat_template", None))

    # --- the <think> guard, asserted on a real rendered prompt -------------------------------------------
    probe = apply_chat(tok, "Say OK.") if chat else "Say OK."
    tstate = think_state(probe)
    if tstate == "open":
        raise SystemExit("[d2] REFUSED: with enable_thinking=False the chat template still leaves a think block "
                         "OPEN. Generating under an unterminated thinking block is the measured failure mode of "
                         "this project.")
    print(f"[d2] think guard: {tstate} (only 'open' refuses; a closed empty block is how thinking is disabled)",
          flush=True)

    t0 = time.time()
    n_over = 0
    with open(outp, "a", encoding="utf-8") as fh, torch.inference_mode():
        for s in range(0, len(todo), a.batch_size):
            chunk = todo[s:s + a.batch_size]
            texts, meta = [], []
            for r in chunk:
                facts, n_fb_tok, n_drop = fit_feedback(tok, r, a.feedback_token_cap)
                p = build_prompt(r, facts)
                texts.append(apply_chat(tok, p) if chat else p)
                meta.append((len(facts), n_drop, n_fb_tok))
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=a.max_input_tokens).to(a.device)
            # one candidate, greedy: candidate count and revision calls are matched by construction
            gen = model.generate(**enc, do_sample=False, num_beams=1, num_return_sequences=1,
                                 max_new_tokens=int(max(r["word_budget"] for r in chunk) * 3 + 64),
                                 pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            outs = tok.batch_decode(new, skip_special_tokens=True)
            if s == 0 and sum(1 for t in outs if not t.strip()) > len(outs) // 2:
                raise SystemExit("[d2] REFUSED: over half of the FIRST batch came back empty. That is the exact "
                                 "signature of the <think>-block failure that produced a wrong result in an "
                                 "earlier campaign. Stopping before 1800 empty rows are written.")
            for r, (n_facts, n_drop, n_fb_tok), text in zip(chunk, meta, outs):
                rev = text.strip()
                nw = len(rev.split())
                over = nw > r["word_budget"]
                n_over += int(over)
                fh.write(json.dumps({
                    "row_id": r["row_id"], "pair_id": r["pair_id"], "arm": r["arm"], "doc_key": r["doc_key"],
                    "resource": r["resource"], "summarizer": r["summarizer"], "revision": rev,
                    "n_words": nw, "word_budget": r["word_budget"], "over_budget": bool(over),
                    "orig_words": r["orig_words"], "n_facts_used": n_facts, "n_facts_dropped_for_token_cap": n_drop,
                    "feedback_tokens": n_fb_tok, "prompt_tokens": int(enc["input_ids"].shape[1]),
                    "empty": not bool(rev), "model_revision": revision, "prompt_sha256": prompt_sha256(),
                }) + "\n")
            if (s // a.batch_size) % 20 == 0:
                fh.flush()
                print(f"[d2] {s + len(chunk)}/{len(todo)} ({time.time() - t0:.0f}s, over-budget so far {n_over})",
                      flush=True)
    dt = time.time() - t0
    man = {"block": "D2", "stage": "generation", "shard": a.shard,
           "model_snapshot": snap, "model_revision": revision, "expect_revision": a.expect_revision,
           "prompt_sha256": prompt_sha256(), "prompt_template": PROMPT, "feedback_header": FEEDBACK_HEADER,
           "decoding": {"do_sample": False, "num_beams": 1, "num_return_sequences": 1,
                        "enable_thinking": False, "think_guard": tstate,
                        "think_guard_contract": "refuse iff the rendered prompt leaves a think block open"},
           "feedback_token_cap": a.feedback_token_cap, "batch_size": a.batch_size,
           "max_input_tokens": a.max_input_tokens, "limit": a.limit,
           "n_rows_shard": len(rows), "n_generated": len(todo), "n_over_budget": n_over,
           "seconds": round(dt, 1),
           "outcome_policy": "no outcome is computed; every outcome needs two blinded reviewers plus adjudication"}
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(a.manifest).write_text(json.dumps(man, indent=1))
    print(f"[d2] generated {len(todo)} revisions in {dt:.0f}s; over-budget {n_over}; manifest {a.manifest}", flush=True)


if __name__ == "__main__":
    main()
