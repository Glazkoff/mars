#!/usr/bin/env python3
"""B8 -- the vLLM pilot: does vLLM 0.29.0 load and serve this project's generation models, and how much
throughput would it buy?

PREREG (marsc_strengthen_20260918) Wave B / B8.  No bar.  The brief's instruction is explicit: report the
outcome either way -- "vLLM does not load this architecture" is a campaign-shaping result.

WHY THIS EXISTS
---------------
vLLM 0.29.0 is installed at /home/user/.venv-vllm but has never run on this box: every LLM log in the
project records backend=hf, and the Slurm node comment records "vLLM off".  Whether it loads
Qwen3_5ForConditionalGeneration (Qwen/Qwen3.8-27B) and Gemma4ForConditionalGeneration (google/gemma-4-31B-it)
is unverified.  A 5-20x throughput win changes the feasibility of every generation-dependent block left in
the campaign -- fresh summaries, the revision study, bulk judging.

WHAT IS HELD FIXED SO THE TWO BACKENDS ARE COMPARABLE
-----------------------------------------------------
The workload is the frozen decomposition contract of `research/mars3_20260916/code/m32_decompose.py`,
IMPORTED not retyped: the same PROMPT literal, the same per-sentence task construction (one task per source
sentence of >= 4 words), the same parser, greedy decoding, the same max_new_tokens.  Both backends are handed
the SAME fully rendered prompt strings, produced once by the HF chat template with enable_thinking=False, so
no template difference can enter the comparison.  vLLM is given the rendered strings through
`LLM.generate(prompts=...)`, never `LLM.chat(...)`, for exactly that reason.

THE <think> TRAP, GUARDED (prereg standing rule 7)
--------------------------------------------------
Qwen3's chat template emits an OPEN `<think>` block unless enable_thinking=False reaches it, which pushes the
answer past the generation window and returns blank text -- the failure that produced a widely repeated but
wrong result in an earlier campaign.  `apply_chat_template` swallows unknown kwargs, so a TypeError fallback
is not proof the switch took effect; the rendered prefix is.  This job refuses to generate unless the rendered
prompt ends in a CLOSED, empty think block, and records the blank-output rate for both backends.

WHAT THIS PILOT IS NOT
----------------------
The generations are a throughput artifact.  They are written only as a small sample for inspection and a
per-backend proposition-count statistic; no unit, score or output statistic of this job may enter any
inventory, training set, calibration or model selection.  Documents are drawn from the RoSE VALIDATION split,
so nothing here touches the sealed TEST split and this job needs no MC_TEST_REREAD.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO), str(REPO / "research" / "mars2_gates_20260916" / "code"),
           str(REPO / "research" / "mars3_20260916" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import m32_decompose as m32d            # noqa: E402  the frozen decomposition interface
import common as gates                  # noqa: E402

PROMPT = m32d.PROMPT                    # imported, never retyped
parse = m32d.parse                      # imported, never retyped
judge = m32d.judge                      # b1_judge_labels: load_model + apply_chat (enable_thinking=False)
CLOSED_THINK = "<think>\n\n</think>\n\n"

# The HF reference this pilot is measured against.  Quoted from the campaign brief / results/marsc/g4.
HF_REFERENCE = {"s_per_document": 6.637462235649547, "model": "google/gemma-4-31B-it", "backend": "hf",
                "provenance": ("results/marsc/g4/cost_table.json -- sacct Elapsed of the eight decomposition "
                               "array tasks divided by 1986 documents; ALLOCATION time, model load included")}


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))


def load_tasks(tasks_path: Path, splits: str, n_docs: int) -> tuple[list[dict], list[str]]:
    """The first n_docs distinct documents of the frozen task file whose splits intersect `splits`, in file
    order, with every sentence task of those documents.  Deterministic and independent of the backend."""
    want = {s.strip() for s in splits.split(",") if s.strip()}
    keys: list[str] = []
    seen: set[str] = set()
    rows = list(gates.load_jsonl(tasks_path))
    for r in rows:
        if want and not (want & set(r.get("splits", []))):
            continue
        k = r["doc_key"]
        if k not in seen:
            if len(seen) >= n_docs:
                continue
            seen.add(k); keys.append(k)
    tasks = [r for r in rows if r["doc_key"] in seen]
    return tasks, keys


def render(tasks: list[dict], snapshot: str) -> tuple[list[str], dict]:
    """Render every task prompt ONCE with the HF chat template, asserting the closed think block."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=True)
    if not getattr(tok, "chat_template", None):
        raise SystemExit("[b8] ABORT: tokenizer has no chat template; the frozen decoding contract is a chat contract")
    probe = judge.apply_chat(tok, PROMPT.format(sentence="The company reported revenue of 1.2 billion dollars in 2023."))
    if not probe.endswith(CLOSED_THINK):
        raise SystemExit("[b8] ABORT: enable_thinking=False did not reach the chat template -- the rendered prompt "
                         f"does not end in a closed empty think block. tail={probe[-120:]!r}")
    print("[b8] closed-think assertion PASSED (enable_thinking=False reached the template)", flush=True)
    prompts = [judge.apply_chat(tok, PROMPT.format(sentence=t["sentence"])) for t in tasks]
    meta = {"prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
            "rendered_tail": probe[-len(CLOSED_THINK):], "closed_think_asserted": True,
            "n_prompts": len(prompts),
            "prompt_chars_mean": sum(len(p) for p in prompts) / max(1, len(prompts))}
    return prompts, meta


def run_vllm(prompts: list[str], snapshot: str, a) -> dict:
    t_init = time.time()
    try:
        from vllm import LLM, SamplingParams
        import vllm
        llm = LLM(model=snapshot, tokenizer=snapshot, dtype="bfloat16",
                  gpu_memory_utilization=a.gpu_memory_utilization, max_model_len=a.max_model_len,
                  tensor_parallel_size=1, enforce_eager=a.enforce_eager, disable_log_stats=True,
                  trust_remote_code=False)
    except Exception as exc:
        import traceback
        return {"loaded": False, "engine_init_seconds": time.time() - t_init,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:],
                "verdict": "vLLM could not load this model on this node"}
    init_s = time.time() - t_init
    print(f"[b8] vLLM {vllm.__version__} engine up in {init_s:.1f}s", flush=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=a.max_new_tokens, n=1)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    texts = [o.outputs[0].text for o in outs]
    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    return {"loaded": True, "vllm_version": vllm.__version__, "engine_init_seconds": init_s,
            "seconds": dt, "sentences": len(prompts), "generated_tokens": gen_tokens,
            "tokens_per_s": gen_tokens / max(1e-9, dt), "sentences_per_s": len(prompts) / max(1e-9, dt),
            "texts": texts,
            "config": {"gpu_memory_utilization": a.gpu_memory_utilization, "max_model_len": a.max_model_len,
                       "enforce_eager": a.enforce_eager, "tensor_parallel_size": 1, "dtype": "bfloat16"}}


def run_hf(prompts: list[str], snapshot: str, a) -> dict:
    import torch
    t_init = time.time()
    try:
        tok, model = judge.load_model(snapshot, a.device)
    except Exception as exc:
        import traceback
        return {"loaded": False, "engine_init_seconds": time.time() - t_init,
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]}
    init_s = time.time() - t_init
    if getattr(tok, "padding_side", None) != "left":
        raise SystemExit(f"[b8] ABORT: padding_side is {tok.padding_side!r}, the frozen contract is left padding")
    print(f"[b8] HF model up in {init_s:.1f}s", flush=True)
    texts: list[str] = []
    gen_tokens = 0
    if a.device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, len(prompts), a.batch_size):
            ch = prompts[s:s + a.batch_size]
            enc = tok(ch, return_tensors="pt", padding=True, truncation=True, max_length=a.max_model_len).to(a.device)
            ids = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            new = ids[:, enc["input_ids"].shape[1]:]
            gen_tokens += int((new != tok.pad_token_id).sum())
            texts.extend(tok.batch_decode(new, skip_special_tokens=True))
            if (s // a.batch_size) % 5 == 0:
                print(f"[b8] hf {min(s + len(ch), len(prompts))}/{len(prompts)} ({time.time() - t0:.0f}s)", flush=True)
    if a.device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.time() - t0
    return {"loaded": True, "engine_init_seconds": init_s, "seconds": dt, "sentences": len(prompts),
            "generated_tokens": gen_tokens, "tokens_per_s": gen_tokens / max(1e-9, dt),
            "sentences_per_s": len(prompts) / max(1e-9, dt), "texts": texts,
            "config": {"batch_size": a.batch_size, "dtype": "bfloat16", "device": a.device}}


def summarise(res: dict, tasks: list[dict], n_docs: int, sample_out: Path | None) -> dict:
    texts = res.pop("texts", [])
    if not res.get("loaded"):
        return res
    props = [parse(t) for t in texts]
    res["documents"] = n_docs
    res["sentences_per_document"] = len(tasks) / max(1, n_docs)
    res["s_per_document"] = res["seconds"] / max(1, n_docs)
    res["documents_per_s"] = n_docs / max(1e-9, res["seconds"])
    res["propositions"] = sum(len(p) for p in props)
    res["propositions_per_sentence"] = res["propositions"] / max(1, len(texts))
    res["blank_raw"] = sum(1 for t in texts if not t.strip())
    res["blank_raw_rate"] = res["blank_raw"] / max(1, len(texts))
    res["empty_parse"] = sum(1 for p in props if not p)
    res["empty_parse_rate"] = res["empty_parse"] / max(1, len(props))
    res["think_trap_signature"] = res["blank_raw_rate"] > 0.10
    if sample_out is not None:
        with open(sample_out, "w", encoding="utf-8") as fh:
            for t, p, o in list(zip(tasks, props, texts))[:40]:
                fh.write(json.dumps({"doc_key": t["doc_key"], "sentence": t["sentence"],
                                     "raw": o.strip()[:600], "props": p}) + "\n")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "hf"], required=True)
    ap.add_argument("--tasks", default="/home/user/results/mars3/m32/decompose_tasks.jsonl")
    ap.add_argument("--splits", default="validation")
    ap.add_argument("--n-docs", type=int, default=50)
    ap.add_argument("--model-id", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--model-snapshot", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--enforce-eager", action="store_true")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    name = a.name or a.backend
    snapshot = a.model_snapshot
    if not snapshot:
        import m3common as m3
        snapshot = m3.snap(a.model_id)
    if not Path(snapshot, "config.json").exists():
        raise SystemExit(f"[b8] ABORT: no config.json under {snapshot}")
    cfg = json.loads(Path(snapshot, "config.json").read_text())

    tasks, keys = load_tasks(Path(a.tasks), a.splits, a.n_docs)
    print(f"[b8] backend={a.backend} model={a.model_id} arch={cfg.get('architectures')} "
          f"docs={len(keys)} sentences={len(tasks)} ({len(tasks) / max(1, len(keys)):.2f} sentences/doc)", flush=True)
    prompts, meta = render(tasks, snapshot)

    rec = {"block": "B8", "backend": a.backend, "name": name, "written_utc": now(),
           "host": platform.node(), "python": sys.version.split()[0],
           "env": {k: os.environ.get(k, "") for k in ("SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES", "HF_HOME", "VIRTUAL_ENV")},
           "model": {"repo_id": a.model_id, "snapshot": snapshot,
                     "resolved_revision": Path(snapshot.rstrip("/")).name,
                     "architectures": cfg.get("architectures"), "model_type": cfg.get("model_type")},
           "workload": {"tasks_file": str(a.tasks), "splits": a.splits, "documents": len(keys),
                        "doc_keys": keys, "sentences": len(tasks),
                        "source": "research/mars3_20260916/code/m32_decompose.py PROMPT + build_tasks contract (imported)"},
           "decoding": {"greedy": True, "max_new_tokens": a.max_new_tokens, "enable_thinking": False, **meta},
           "hf_reference": HF_REFERENCE,
           "isolation_rule": "throughput pilot only; no output of this job enters any inventory, training set, "
                             "calibration or model selection"}
    save(out / f"b8_{name}_started.json", rec)

    t_all = time.time()
    res = run_vllm(prompts, snapshot, a) if a.backend == "vllm" else run_hf(prompts, snapshot, a)
    res = summarise(res, tasks, len(keys), out / f"b8_{name}_sample.jsonl" if res.get("loaded") else None)
    rec["result"] = res
    rec["wall_seconds_including_init"] = time.time() - t_all
    if res.get("loaded"):
        rec["vs_hf_reference"] = {
            "reference_s_per_document": HF_REFERENCE["s_per_document"],
            "reference_model": HF_REFERENCE["model"],
            "measured_s_per_document": res["s_per_document"],
            "speedup_vs_reference": HF_REFERENCE["s_per_document"] / max(1e-9, res["s_per_document"]),
            "caveat": ("the recorded reference is Gemma-4-31B under HF with ALLOCATION time (model load included) "
                       "while this figure is active generation with a different model; the controlled comparison "
                       "is the same-model hf arm of this job, not this ratio"),
        }
        print(f"[b8] {name}: {res['s_per_document']:.3f} s/document, {res['documents_per_s']:.4f} docs/s, "
              f"{res['tokens_per_s']:.1f} generated tok/s, blank-raw {res['blank_raw_rate']:.4f}, "
              f"empty-parse {res['empty_parse_rate']:.4f}", flush=True)
        if res["think_trap_signature"]:
            print("[b8] WARNING: blank-output rate above 0.10 -- the <think>-trap signature. "
                  "The throughput number stands; the GENERATIONS are not usable.", flush=True)
    else:
        print(f"[b8] {name}: NOT LOADED -- {res.get('error')}", flush=True)
    save(out / f"b8_{name}.json", rec)
    print(f"[b8] written -> {out / f'b8_{name}.json'}", flush=True)


if __name__ == "__main__":
    main()
