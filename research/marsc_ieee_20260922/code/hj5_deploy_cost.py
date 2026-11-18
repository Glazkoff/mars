#!/usr/bin/env python3
"""H-J5 -- deployment cost the paper does not yet measure: resident weights, peak GPU memory, batch-1 request
latency per (summary, fact) unit, and batched throughput, for every verifier the paper compares.

Registered in research/marsc_ieee_20260922/PREREG.md (amendment 3): a measurement, no bar.

Contract. 200 pairs are drawn UNIFORMLY at random (seed 20260922) from the RoSE validation Gemma+ent inventory
(the project's cost-probe rule: never the first N of a file), every unit of those pairs is scored once at the
system's batched setting for throughput, and 300 units drawn uniformly from the same set are scored one at a
time for latency (20 warm-up calls discarded), with CUDA synchronised around every call. Premise and claim are
the systems' own: the candidate summary clipped to 380 words, the unit text for MARS-C (as trained) and the
rendered claim clipped to 96 words for the comparators (as in b9/b10). Peak memory is
torch.cuda.max_memory_allocated() after a reset, measured separately at the batched and the batch-1 setting;
for the vLLM systems, which pre-allocate a KV-cache pool, peak allocation is not a property of the model, so
the weight bytes on disk and nvidia-smi's used memory after engine start are recorded instead.

Two backends because two venvs: --backend hf (mars venv: marsc, factcg, nli, alignscore) and --backend vllm
(vllm venv: minicheck7b, judge), one process per vLLM model so nothing shares a GPU pool.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sample_units(inventory: Path, n_pairs: int, n_latency: int, seed: int, claim_words: int):
    rows = [r for r in common.load_jsonl(inventory) if r.get("units")]
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(rows)), n_pairs))
    rows = [rows[i] for i in idx]
    prem = [" ".join(r["candidate"].split()[:380]) for r in rows]
    units = []
    for i, r in enumerate(rows):
        for u in r["units"]:
            claim = " ".join(common.render_claim(r["source"], u).split()[:claim_words])
            units.append({"premise": prem[i], "unit_text": u["text"], "claim": claim, "pair_id": r["pair_id"]})
    lat = rng.sample(units, min(n_latency, len(units)))
    return rows, units, lat


def nvidia_smi_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        dev = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]) if out and len(out) > 1 else 0
        return int(out[min(dev, len(out) - 1)])
    except Exception:  # noqa: BLE001
        return None


def dir_bytes(p: str, suffixes=(".safetensors", ".bin", ".pt")) -> int:
    tot = 0
    for root, _, files in os.walk(p):
        for f in files:
            if f.endswith(suffixes):
                tot += os.path.getsize(os.path.join(root, f))
    return tot


def time_calls(fn, items, warmup: int, sync) -> dict:
    """Batch-1 latency: one call per item, synchronised; returns ms statistics."""
    for it in items[:warmup]:
        fn(it); sync()
    lat = []
    for it in items[warmup:]:
        sync(); t0 = time.perf_counter(); fn(it); sync(); lat.append((time.perf_counter() - t0) * 1e3)
    lat_sorted = sorted(lat)
    return {"n": len(lat), "median_ms": statistics.median(lat), "mean_ms": statistics.fmean(lat),
            "p95_ms": lat_sorted[int(0.95 * (len(lat) - 1))], "min_ms": lat_sorted[0], "max_ms": lat_sorted[-1]}


# ----------------------------------------------------------------------------- HF backend
def run_hf(a, units, lat_units) -> dict:
    import torch
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    res = {}
    handle = None  # the loaded model of the current system, released at the end of each iteration
    for system in a.systems:
        del handle; handle = None
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        if system == "marsc":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(a.marsc_ckpt)
            model = AutoModelForSequenceClassification.from_pretrained(a.marsc_ckpt).to(dev).eval()
            handle = model
            weights = sum(p.numel() * p.element_size() for p in model.parameters())
            n_params = sum(p.numel() for p in model.parameters())

            def score(batch):
                enc = tok([u["premise"] for u in batch], [u["unit_text"] for u in batch], truncation="longest_first",
                          max_length=512, padding=True, return_tensors="pt").to(dev)
                with torch.inference_mode():
                    return torch.sigmoid(model(**enc).logits.squeeze(-1).float()).tolist()
            batch = a.batch
        elif system == "factcg":
            swap = _load(REPO / "research" / "marsc_strengthen_20260918" / "code" / "b10_verifier_swap.py")
            sc = swap.FactCG(a.factcg_snapshot, a.batch, dev)
            handle = sc
            weights = sum(p.numel() * p.element_size() for p in sc.model.parameters())
            n_params = sum(p.numel() for p in sc.model.parameters())

            def score(batch):
                with contextlib.redirect_stdout(io.StringIO()):
                    return sc.score([u["premise"] for u in batch], [u["claim"] for u in batch])
            batch = a.batch
        elif system in ("nli", "alignscore"):
            ssc = common._load("strengthen_score_cells")
            sc = ssc.build_scorer(system, a.batch, dev)
            handle = sc
            weights, n_params = None, None
            # WindowNLI keeps its model in .m and AlignScoreCanonical in .enc plus the head tensors .w/.b
            # (scripts/strengthen_score_cells.py); the earlier probe looked only for model/nli/scorer.
            for attr in ("model", "nli", "scorer", "m", "enc"):
                m = getattr(sc, attr, None)
                if m is not None and hasattr(m, "parameters"):
                    weights = sum(p.numel() * p.element_size() for p in m.parameters())
                    n_params = sum(p.numel() for p in m.parameters()); break
            for attr in ("w", "b"):
                t = getattr(sc, attr, None)
                if weights is not None and isinstance(t, torch.Tensor):
                    weights += t.numel() * t.element_size(); n_params += t.numel()
            wdt = next((str(p.dtype) for a in ("m", "enc", "model") for p in getattr(getattr(sc, a, None), "parameters", lambda: [])()), None)

            def score(batch):
                with contextlib.redirect_stdout(io.StringIO()):
                    return sc.score([u["premise"] for u in batch], [u["claim"] for u in batch])
            batch = a.batch
        else:
            raise SystemExit(f"[hj5] unknown hf system {system!r}")
        after_load = torch.cuda.memory_allocated() - base
        # batched throughput + peak
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter(); n = 0
        for s in range(0, len(units), batch):
            score(units[s:s + batch]); n += min(batch, len(units) - s)
        torch.cuda.synchronize(); el = time.perf_counter() - t0
        peak_batched = torch.cuda.max_memory_allocated()
        # batch-1 latency + peak
        torch.cuda.reset_peak_memory_stats()
        lat = time_calls(lambda u: score([u]), lat_units, a.warmup, torch.cuda.synchronize)
        peak_b1 = torch.cuda.max_memory_allocated()
        res[system] = {"backend": "hf", "n_units_throughput": n, "batch": batch, "seconds": el,
                       "units_per_s": n / el, "ms_per_unit_batched": 1e3 * el / n,
                       "latency_batch1": lat, "n_params": n_params, "weight_bytes_in_memory": weights,
                       "memory_after_load_bytes": after_load, "peak_alloc_batched_bytes": peak_batched,
                       "peak_alloc_batch1_bytes": peak_b1,
                       "weight_dtype": (wdt if system in ("nli", "alignscore") else "torch.float32")}
        print(f"[hj5] {system}: {n} units in {el:.1f}s ({n / el:.0f} u/s, {1e3 * el / n:.3f} ms/u batched); "
              f"batch-1 median {lat['median_ms']:.1f} ms p95 {lat['p95_ms']:.1f}; peak batched "
              f"{peak_batched / 2**30:.2f} GiB, batch-1 {peak_b1 / 2**30:.2f} GiB, weights "
              f"{(weights or 0) / 2**30:.2f} GiB", flush=True)
        handle = None
        torch.cuda.empty_cache()
    return res


# ----------------------------------------------------------------------------- vLLM backend
def run_vllm(a, units, lat_units) -> dict:
    system = a.systems[0]
    assert len(a.systems) == 1, "one vLLM model per process"
    used0 = nvidia_smi_used_mib()
    t0 = time.perf_counter()
    if system == "minicheck7b":
        swap = _load(REPO / "research" / "marsc_strengthen_20260918" / "code" / "b10_verifier_swap.py")
        sc = swap.MiniCheck7B(a.minicheck_snapshot, a.max_model_len, a.gpu_util, 1, Path(a.minicheck_utils), True)
        weight_bytes = dir_bytes(a.minicheck_snapshot)

        def score(batch):
            with contextlib.redirect_stdout(io.StringIO()):
                return sc.score([u["premise"] for u in batch], [u["claim"] for u in batch])
    elif system == "judge":
        b9 = _load(REPO / "research" / "marsc_strengthen_20260918" / "code" / "b9_judge_units.py")
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        tok = AutoTokenizer.from_pretrained(a.judge_snapshot, local_files_only=True, use_fast=True)
        yes_ids, no_ids = b9.word_token_ids(tok, b9.YES), b9.word_token_ids(tok, b9.NO)
        llm = LLM(model=a.judge_snapshot, tokenizer=a.judge_snapshot, dtype="bfloat16",
                  gpu_memory_utilization=a.gpu_util, max_model_len=a.max_model_len, tensor_parallel_size=1,
                  disable_log_stats=True, enable_prefix_caching=True, trust_remote_code=False)
        sp_exact = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, logprob_token_ids=sorted(yes_ids | no_ids))
        weight_bytes = dir_bytes(a.judge_snapshot)

        def score(batch):
            pids = [tok(b9.apply_chat(tok, b9.PROMPT.format(summary=u["premise"], fact=u["claim"])),
                        add_special_tokens=False)["input_ids"] for u in batch]
            with contextlib.redirect_stdout(io.StringIO()):
                sup, _ = b9.exact_support(llm, sp_exact, pids, yes_ids, no_ids)
            return sup
    else:
        raise SystemExit(f"[hj5] unknown vllm system {system!r}")
    load_s = time.perf_counter() - t0
    used1 = nvidia_smi_used_mib()
    # batched throughput (the whole sample in one generate call, as the scoring jobs do)
    t0 = time.perf_counter(); score(units); el = time.perf_counter() - t0
    lat = time_calls(lambda u: score([u]), lat_units, a.warmup, lambda: None)
    res = {system: {"backend": "vllm", "n_units_throughput": len(units), "batch": "engine-scheduled", "seconds": el,
                    "units_per_s": len(units) / el, "ms_per_unit_batched": 1e3 * el / len(units),
                    "latency_batch1": lat, "weight_bytes_on_disk": weight_bytes,
                    "nvidia_smi_used_mib_before": used0, "nvidia_smi_used_mib_after_engine": used1,
                    "gpu_memory_utilization": a.gpu_util, "max_model_len": a.max_model_len,
                    "engine_start_s": load_s,
                    "memory_note": "vLLM pre-allocates a KV-cache pool; used memory reflects gpu_memory_utilization, not the model"}}
    print(f"[hj5] {system}: {len(units)} units in {el:.1f}s ({len(units) / el:.0f} u/s, {1e3 * el / len(units):.3f} ms/u); "
          f"batch-1 median {lat['median_ms']:.1f} ms p95 {lat['p95_ms']:.1f}; weights on disk {weight_bytes / 2**30:.2f} GiB; "
          f"nvidia-smi used {used1} MiB after engine start", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "vllm"], required=True)
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--n-latency", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--claim-words", type=int, default=96)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--marsc-ckpt", default="")
    ap.add_argument("--factcg-snapshot", default="")
    ap.add_argument("--minicheck-snapshot", default="")
    ap.add_argument("--minicheck-utils", default="")
    ap.add_argument("--judge-snapshot", default="")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, units, lat_units = sample_units(Path(a.inventory), a.n_pairs, a.n_latency, a.seed, a.claim_words)
    print(f"[hj5] backend={a.backend} systems={a.systems}: {len(rows)} pairs, {len(units)} units, "
          f"{len(lat_units)} latency units (seed {a.seed})", flush=True)
    res = run_hf(a, units, lat_units) if a.backend == "hf" else run_vllm(a, units, lat_units)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = "_".join(a.systems)
    blob = {"registered": "research/marsc_ieee_20260922/PREREG.md amendment 3 (H-J5, measurement, no bar)",
            "inventory": a.inventory, "n_pairs": len(rows), "n_units": len(units), "n_latency_units": len(lat_units),
            "seed": a.seed, "claim_words": a.claim_words, "warmup": a.warmup, "systems": res}
    (out / f"deploy_cost_{tag}.json").write_text(json.dumps(blob, indent=1))
    print(f"[hj5] written {out / f'deploy_cost_{tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
