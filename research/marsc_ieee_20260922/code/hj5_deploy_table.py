#!/usr/bin/env python3
"""H-J5 -- render the deployment-cost table (articles/iclr2027/tables/marsc_deploy.tex) from the hj5 receipts.

Reads deploy_cost_<systems>.json files written by hj5_deploy_cost.py and emits one LaTeX table with, per
system: resident weights, peak GPU allocation at the batched and batch-1 settings (HF) or nvidia-smi used
memory after engine start (vLLM), batch-1 request latency (median and p95, ms per unit) and batched throughput
(units per second and ms per unit). Nothing is computed beyond unit conversion; every number is the receipt's.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

GIB = 2 ** 30
LABEL = {
    "marsc": ("MARS-C verifier, one seed (ours)", "$0.4$B enc., fp32"),
    "factcg": ("FactCG-DeBERTa-v3-L", "$0.4$B enc., fp32"),
    "nli": ("windowed RoBERTa-large-MNLI", "$0.4$B enc., fp32"),
    "alignscore": ("AlignScore", "$<1$B enc., fp32"),
    "minicheck7b": ("Bespoke-MiniCheck-7B", "$7$B dec., bf16, vLLM"),
    "judge": ("Gemma-4-31B zero-shot judge", "$31$B dec., bf16, vLLM"),
}
ORDER = ["marsc", "nli", "factcg", "alignscore", "minicheck7b", "judge"]


def fmt_gib(b):
    return "\\pend" if not b else f"${b / GIB:.2f}$"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hj5-dir", required=True, help="directory holding deploy_cost_*.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    systems: dict[str, dict] = {}
    meta = None
    for f in sorted(glob.glob(str(Path(a.hj5_dir) / "deploy_cost_*.json"))):
        j = json.load(open(f))
        meta = meta or {k: j[k] for k in ("n_pairs", "n_units", "n_latency_units", "seed", "warmup")}
        systems.update(j["systems"])
    if not systems or meta is None:
        raise SystemExit("no receipts found")
    rows = []
    for s in ORDER:
        if s not in systems:
            continue
        r = systems[s]; lab, kind = LABEL[s]; lat = r["latency_batch1"]
        if r["backend"] == "hf":
            weights = fmt_gib(r.get("weight_bytes_in_memory"))
            mem = f"{fmt_gib(r['peak_alloc_batched_bytes'])} / {fmt_gib(r['peak_alloc_batch1_bytes'])}"
        else:
            weights = fmt_gib(r.get("weight_bytes_on_disk"))
            used = r.get("nvidia_smi_used_mib_after_engine")
            mem = ("\\pend" if used is None else f"${used / 1024:.1f}$\\textsuperscript{{p}}") + " / \\pend"
        rows.append(f"{lab} & {kind} & {weights} & {mem} & ${lat['median_ms']:.1f}$ (${lat['p95_ms']:.1f}$) & "
                    f"${r['units_per_s']:.0f}$ (${r['ms_per_unit_batched']:.3f}$) \\\\")
    body = "\n".join(rows)
    tex = r"""% H-J5 deployment cost, rendered by research/marsc_ieee_20260922/code/hj5_deploy_table.py from the hj5 receipts.
\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}llccrr@{}}
\toprule
& & \textbf{weights} & \textbf{peak GPU memory} & \textbf{batch-1 latency} & \textbf{batched throughput} \\
\textbf{Scorer} & \textbf{form} & GiB & batched / batch-1, GiB & ms per unit, median (p95) & units/s (ms per unit) \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\caption{\textbf{Deployment cost, measured on one H200 over the same uniformly drawn sample for every
scorer.} """ + f"{meta['n_pairs']}" + r""" pairs drawn uniformly at random (seed """ + f"{meta['seed']}" + r""") from the RoSE
validation Gemma+ent inventory, """ + f"{meta['n_units']:,}" + r""" (summary, fact) units for throughput and """ + f"{meta['n_latency_units']}" + r""" of them
scored one at a time for latency after """ + f"{meta['warmup']}" + r""" warm-up calls, CUDA synchronised around every call. Peak GPU
memory is the allocator's high-water mark for the HF cross-encoders; the vLLM rows report nvidia-smi's used
memory after engine start (\textsuperscript{p}), which is the pre-allocated pool set by the engine's
memory-utilisation target rather than a property of the model, and their weights are the checkpoint's bytes on
disk. The emitter of record runs three MARS-C seeds, so its batched cost is three times the first row's;
batch-1 latency is per verifier call and does not add across seeds run in parallel. Per-unit throughput here
comes from a different window than Table~\ref{tab:marsc_cost} and is not substituted for it.}
\label{tab:marsc_deploy}
\end{table}
"""
    Path(a.out).write_text(tex)
    print(f"wrote {a.out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
