#!/usr/bin/env python3
"""MARS-C G4 -- cost account of the deployable pipeline (PLAN_MAIN_CONTRIBUTION.md, block G4).

Measures seconds per pair on the current GPU for the verifiers over each inventory (first N pairs), and combines them with
the inventory-generation cost harvested from the M32 receipts (passed in as elapsed seconds and document counts). Inventory
cost is per DOCUMENT and is amortised over the systems per document in the evaluation resource.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402


def time_verifier(model_dir: str, rows: list[dict], batch: int, dev: str) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir); model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev).eval()
    flat = [(i, u["text"]) for i, r in enumerate(rows) for u in r["units"]]
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, len(flat), batch):
            ch = flat[s:s + batch]
            enc = tok([" ".join(rows[i]["candidate"].split()[:380]) for i, _ in ch], [f for _, f in ch], truncation="longest_first", max_length=512, padding=True, return_tensors="pt").to(dev)
            model(**enc)
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0; del model
    return {"pairs": len(rows), "units": len(flat), "units_per_pair": len(flat) / len(rows), "seconds": dt, "s_per_pair": dt / len(rows), "s_per_unit": dt / max(1, len(flat))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", required=True, help="name=path")
    ap.add_argument("--verifiers", nargs="+", required=True, help="name=model_dir")
    ap.add_argument("--n-pairs", type=int, default=200); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--inventory-cost", nargs="*", default=[], help="name=elapsed_seconds:documents (GPU seconds for the whole inventory build)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.backends.cuda.matmul.allow_tf32 = True
    rep = {"device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "n_pairs_timed": a.n_pairs, "verifier": {}, "inventory": {}}
    for spec in a.units:
        name, path = spec.split("=", 1); rows = [r for r in common.load_jsonl(Path(path)) if r.get("units")][:a.n_pairs]
        docs = {common.doc_key(r["source"]) for r in rows}
        rep["inventory"].setdefault(name, {})["pairs_per_document_in_sample"] = len(rows) / max(1, len(docs))
        for vspec in a.verifiers:
            vname, vdir = vspec.split("=", 1); vdir = sorted(glob.glob(vdir))[0] if glob.glob(vdir) else vdir
            rep["verifier"][f"{vname}|{name}"] = time_verifier(vdir, rows, a.batch, dev)
            r = rep["verifier"][f"{vname}|{name}"]; print(f"[g4] {vname} over {name}: {r['units_per_pair']:.1f} units/pair, {r['s_per_pair']:.3f} s/pair, {1000 * r['s_per_unit']:.2f} ms/unit", flush=True)
    for spec in a.inventory_cost:
        name, rest = spec.split("=", 1); secs, docs = (float(x) for x in rest.split(":"))
        rep["inventory"].setdefault(name, {}).update({"gpu_seconds_total": secs, "documents": docs, "s_per_document": secs / docs})
        ppd = rep["inventory"][name].get("pairs_per_document_in_sample")
        if ppd:
            rep["inventory"][name]["s_per_pair_amortised"] = secs / docs / ppd
        print(f"[g4] inventory {name}: {secs / docs:.2f} s/document" + (f", {secs / docs / ppd:.3f} s/pair amortised over {ppd:.1f} pairs/document" if ppd else ""), flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / "cost_table.json").write_text(json.dumps(rep, indent=1))
    print(f"[g4] written {out / 'cost_table.json'}", flush=True)


if __name__ == "__main__":
    main()
