#!/usr/bin/env python3
"""M32b -- fact-level training rows: the LLM inventory of the B1 TRAIN pairs, labelled by the in-domain cross-encoder
(teacher; 0.95 AUROC vs human ACUs). Same pairs as the B1 training rows, so tagger/writer arms trained on these rows
are data-matched with the spaCy-unit arms. Output rows follow training_rows.jsonl (unit_labels 1 = covered) plus
`unit_soft` (teacher p_omitted). Never touches validation/test labels.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import m3common as m3
from m3common import gates

inv = gates._load("m32_inventory", Path(__file__).resolve().parent)
pipe = gates._load("m31_score_pipeline_all", Path(__file__).resolve().parent)


def choose_variant(ceilings: Path, minimum: float) -> str | None:
    d = json.load(open(ceilings)) if ceilings.exists() else {}
    best = None
    for v in ("gemma+ent", "gemma"):
        c = d.get(f"{v}/validation", {}).get("ceiling_weak_pooled")
        if c is not None and c >= minimum and (best is None or c > best[1]):
            best = (v, c)
    return best[0] if best else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="auto", help="gemma | gemma+ent | auto (best variant whose validation ceiling >= --min-ceiling)")
    ap.add_argument("--min-ceiling", type=float, default=0.85)
    ap.add_argument("--props", nargs="+", required=True)
    ap.add_argument("--pairs-from", default=str(gates.B1_DIR / "training_rows.jsonl"))
    ap.add_argument("--ckpts", default=str(gates.OUT_ROOT / "b21" / "xenc_seed*"))
    ap.add_argument("--out", default=str(m3.M3_OUT / "m32" / "train_rows_facts.jsonl"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    variant = a.variant if a.variant != "auto" else choose_variant(m3.M3_OUT / "m32" / "ceilings.json", a.min_ceiling)
    if variant is None:
        raise SystemExit(f"GATE: no LLM inventory reaches the {a.min_ceiling} validation ceiling; M32b not run")
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    keep = {r["pair_id"] for r in gates.load_jsonl(Path(a.pairs_from)) if r.get("split", "train") == "train"}
    pairs = [r for r in gates.load_jsonl(gates.B1_DIR / "pairs_train.jsonl") if r["pair_id"] in keep]
    if a.limit:
        pairs = pairs[:a.limit]
    props = inv.props_by_doc(a.props)
    extractor = None
    if variant == "gemma+ent":
        from mars_v2.units import get_extractor
        extractor = get_extractor("entity")
    cache = {}

    def inventory(src):
        k = hashlib.sha1(src.encode()).hexdigest()[:12]
        if k not in cache:
            units = list(props.get(k, []))
            if extractor is not None:
                units += [u.to_dict() for u in extractor(src)]
            cache[k] = units
        return cache[k]

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = True
    ck = [d for d in sorted(glob.glob(a.ckpts)) if Path(d, "config.json").exists()]
    print(f"[m32b] variant={variant} pairs={len(pairs)} teachers={len(ck)} device={dev}", flush=True)
    rows = [{"pair_id": r["pair_id"], "cls": "natural", "resource": r["resource"], "system": r["system"], "split": "train",
             "source": r["source"], "candidate": r["candidate"], "units": inventory(r["source"])} for r in pairs]
    rows = [r for r in rows if r["units"]]
    flat = [(i, j, pipe.hyp(r["source"], u)) for i, r in enumerate(rows) for j, u in enumerate(r["units"])]
    soft = np.zeros(len(flat)); t0 = time.time()
    for d in ck:
        tok = AutoTokenizer.from_pretrained(d); model = AutoModelForSequenceClassification.from_pretrained(d).to(dev).eval()
        with torch.inference_mode():
            for s in range(0, len(flat), a.batch):
                ch = flat[s:s + a.batch]
                enc = tok([pipe.clip(rows[i]["candidate"], 380) for i, _, _ in ch], [h for _, _, h in ch], truncation="longest_first", max_length=512, padding=True, return_tensors="pt").to(dev)
                soft[s:s + len(ch)] += torch.softmax(model(**enc).logits.float(), -1)[:, 0].cpu().numpy()
                if (s // a.batch) % 200 == 0:
                    print(f"[m32b]   {Path(d).name} {s + len(ch)}/{len(flat)} ({time.time() - t0:.0f}s)", flush=True)
        del model; torch.cuda.empty_cache() if dev.startswith("cuda") else None
    soft /= max(1, len(ck))
    per = {}
    for (i, j, _), v in zip(flat, soft):
        per.setdefault(i, {})[j] = float(v)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); n_units = n_om = 0
    with open(out, "w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            sv = [per[i].get(j, np.nan) for j in range(len(r["units"]))]
            labels = [None if not np.isfinite(v) else int(v < 0.5) for v in sv]       # 1 = covered
            n_units += len(labels); n_om += sum(1 for x in labels if x == 0)
            fh.write(json.dumps({**r, "unit_labels": labels, "unit_soft": sv, "consistency_label": None, "training_eligible": True,
                                 "label_source": f"xenc-teacher:{variant}"}) + "\n")
    man = {"variant": variant, "pairs": len(rows), "units": n_units, "omitted_fraction": n_om / max(1, n_units), "teachers": ck,
           "seconds": time.time() - t0, "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    json.dump(man, open(out.with_suffix(".manifest.json"), "w"), indent=1)
    print(f"[m32b] {json.dumps({k: v for k, v in man.items() if k != 'teachers'})}", flush=True)


if __name__ == "__main__":
    main()
