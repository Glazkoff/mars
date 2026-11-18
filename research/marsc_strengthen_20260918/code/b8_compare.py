#!/usr/bin/env python3
"""B8 -- combine the per-backend pilot records into one verdict.

The only comparison that identifies a backend effect is the SAME-MODEL, SAME-PROMPT, SAME-WORKLOAD pair
produced by b8_vllm_pilot.py.  The recorded 6.637 GPU-s/document is a different model measured as allocation
time, so it enters the verdict only as a labelled context line, never as the denominator of the speedup.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vllm", required=True)
    ap.add_argument("--hf", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    def load(p: str) -> dict:
        q = Path(p)
        return json.loads(q.read_text()) if q.exists() else {"missing": p}

    v, h = load(a.vllm), load(a.hf)
    vr = (v.get("result") or {}); hr = (h.get("result") or {})
    rep = {"block": "B8", "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "inputs": {"vllm": a.vllm, "hf": a.hf},
           "vllm_loaded": bool(vr.get("loaded")), "hf_loaded": bool(hr.get("loaded")),
           "model": (v.get("model") or h.get("model")),
           "prompt_sha256": ((v.get("decoding") or {}).get("prompt_sha256")
                             or (h.get("decoding") or {}).get("prompt_sha256")),
           "vllm": {k: vr.get(k) for k in ("vllm_version", "engine_init_seconds", "seconds", "sentences",
                                           "documents", "s_per_document", "documents_per_s", "tokens_per_s",
                                           "blank_raw_rate", "empty_parse_rate", "propositions_per_sentence",
                                           "think_trap_signature", "error", "verdict")},
           "hf": {k: hr.get(k) for k in ("engine_init_seconds", "seconds", "sentences", "documents",
                                         "s_per_document", "documents_per_s", "tokens_per_s",
                                         "blank_raw_rate", "empty_parse_rate", "propositions_per_sentence",
                                         "think_trap_signature", "error")},
           "recorded_context": (v.get("hf_reference") or h.get("hf_reference"))}

    if vr.get("loaded") and hr.get("loaded"):
        sp = hr["s_per_document"] / max(1e-9, vr["s_per_document"])
        tp = vr["tokens_per_s"] / max(1e-9, hr["tokens_per_s"])
        same = (v.get("workload", {}).get("sentences") == h.get("workload", {}).get("sentences")
                and v.get("model", {}).get("snapshot") == h.get("model", {}).get("snapshot"))
        rep["controlled_speedup"] = {
            "same_model_same_prompt_same_workload": bool(same),
            "identical_document_set": (v.get("workload", {}).get("doc_keys") == h.get("workload", {}).get("doc_keys")),
            "hf_documents": hr.get("documents"), "vllm_documents": vr.get("documents"),
            "speedup_s_per_document": sp, "speedup_generated_tokens_per_s": tp,
            "note": ("document sets may differ in size by design (the HF arm is deliberately smaller); the rate "
                     "comparison is per document and per generated token, both of which are size-invariant"),
        }
        rep["verdict"] = (f"vLLM loads {rep['model'].get('architectures')} and is "
                          f"{sp:.2f}x faster per document than HF on the same model and workload")
    elif vr.get("loaded"):
        rep["verdict"] = ("vLLM loads and runs; the same-model HF arm did not complete, so only the labelled "
                          "context ratio against the recorded Gemma figure is available")
    elif "missing" in v:
        rep["verdict"] = "the vLLM arm produced no record at all (job did not reach it)"
    else:
        # Be precise about WHICH failure this was. "vLLM does not load this architecture" is a
        # campaign-shaping claim and must not be asserted when the traceback says something else --
        # job 5346 loaded the weights and captured CUDA graphs and then died in FlashInfer's JIT.
        tb = (vr.get("traceback") or "") + " " + (vr.get("error") or "")
        arch = rep["model"].get("architectures")
        if "not supported" in tb.lower() or "unsupported" in tb.lower() or "ValueError" in tb and "architecture" in tb.lower():
            rep["verdict"] = f"vLLM does NOT support {arch} on this node: {vr.get('error')}"
        elif "ninja" in tb or "nvcc" in tb or "jit" in tb.lower():
            rep["verdict"] = (f"vLLM initialisation failed on an ENVIRONMENT dependency, not on {arch}: "
                              f"{vr.get('error')} -- see traceback (JIT toolchain missing from PATH)")
        else:
            rep["verdict"] = (f"vLLM engine initialisation failed for {arch} on this node "
                              f"(architecture support NOT disproved): {vr.get('error')}")
    attempts = []
    for f in sorted(out.glob("b8_*.json")):
        if f.name in ("b8_verdict.json",) or f.name.endswith("_started.json"):
            continue
        try:
            d = json.loads(f.read_text()); r = d.get("result") or {}
        except Exception:
            continue
        attempts.append({"file": f.name, "backend": d.get("backend"), "name": d.get("name"),
                         "loaded": bool(r.get("loaded")), "error": r.get("error"),
                         "engine_init_seconds": r.get("engine_init_seconds"),
                         "s_per_document": r.get("s_per_document")})
    rep["all_attempts_in_this_directory"] = attempts
    (out / "b8_verdict.json").write_text(json.dumps(rep, indent=1))
    print(f"[b8] VERDICT: {rep['verdict']}", flush=True)
    print(f"[b8] written -> {out / 'b8_verdict.json'}", flush=True)


if __name__ == "__main__":
    main()
