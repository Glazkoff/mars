#!/usr/bin/env python3
"""Shared helpers for the MARS-2 main-contribution gates (B18-B22), registered in ../PREREG.md.

Unit sets
  acu  RoSE atomic content units with HUMAN presence labels (pairs_<split>.jsonl: acu_units)
  m2   MARS-2 extracted entity+proposition units with B1 labels (labels_<split>.jsonl)
Every system writes one line per pair into a `unit_scores` file:
  {"pair_id", "system", "unit_set", "scores": [omission score per unit index, higher = more likely omitted]}
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
B1_DIR = Path(os.environ.get("B1_DIR", "/home/user/data/plan2026/b1"))
OUT_ROOT = Path(os.environ.get("MARS2_GATES_OUT", "/home/user/results/mars2_gates"))
SEED = 20260916
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])|\n+")
_TOK = re.compile(r"[A-Za-z0-9]+(?:'[a-z]+)?")
STOP = set("a an the of to in on at for and or but is are was were be been being it its this that these those with by from as "
           "he she they we you i his her their our your him them us me my not no s".split())


def _load(name: str, base: Path = REPO / "scripts"):
    spec = importlib.util.spec_from_file_location(name, base / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def cv():
    return _load("conditional_value")


# ----------------------------------------------------------------------------- text
def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of sentences; newlines also split (dialogue turns)."""
    spans, start = [], 0
    for m in _SENT.finditer(text):
        if m.start() > start and text[start:m.start()].strip():
            spans.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans or [(0, len(text))]


def sentences(text: str) -> list[str]:
    return [text[a:b].strip() for a, b in sentence_spans(text)]


def tokens(text: str, content_only: bool = False) -> list[str]:
    t = [x.lower() for x in _TOK.findall(text)]
    return [x for x in t if x not in STOP] if content_only else t


def token_recall(unit: str, cand: str) -> float:
    u = tokens(unit, content_only=True) or tokens(unit)
    if not u:
        return 0.0
    c = set(tokens(cand))
    return sum(1 for x in u if x in c) / len(u)


def token_f1(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    common = sum((defaultdict(int, {t: min(ta.count(t), tb.count(t)) for t in set(ta)})).values())
    if common == 0:
        return 0.0
    p, r = common / len(tb), common / len(ta)
    return 2 * p * r / (p + r)


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rougeL_recall(unit: str, cand: str) -> float:
    u, c = tokens(unit), tokens(cand)
    return lcs_len(u, c) / len(u) if u else 0.0


def best_sentence_span(source: str, claim: str) -> tuple[int, int]:
    """Source sentence with the highest token F1 to the claim (the region a supplied unit came from)."""
    spans = sentence_spans(source)
    scores = [token_f1(source[a:b], claim) for a, b in spans]
    return spans[int(np.argmax(scores))]


def containing_span(source: str, start: int, end: int) -> tuple[int, int]:
    for a, b in sentence_spans(source):
        if a <= start < b:
            return a, max(b, end)
    return max(0, start - 200), min(len(source), end + 200)


def render_claim(source: str, u: dict) -> str:
    """Claim text for an extracted unit: proposition -> 'subject predicate object'; entity -> containing sentence."""
    args = u.get("args") or {}
    if u.get("kind") == "proposition" and args.get("subject") and args.get("predicate"):
        return f"{args['subject']} {args['predicate']} {u['text']}".strip()
    a, b = containing_span(source, int(u.get("start", 0)), int(u.get("end", 0)))
    return source[a:b].strip()


def doc_key(source: str) -> str:
    return hashlib.sha1(source[:200].encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- data
def load_jsonl(p: Path):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def acu_rows(split: str, b1_dir: Path = B1_DIR) -> list[dict]:
    """RoSE pairs with human ACU presence labels -> unit-set rows (label 1 = present)."""
    out = []
    for r in load_jsonl(b1_dir / f"pairs_{split}.jsonl"):
        acus = r.get("acu_units")
        if not acus:
            continue
        out.append({"pair_id": r["pair_id"], "resource": r["resource"], "system": r["system"], "split": split,
                    "source": r["source"], "candidate": r["candidate"],
                    "units": [{"kind": "acu", "text": a["text"]} for a in acus],
                    "unit_labels": [int(a["label"]) for a in acus]})
    return out


def m2_rows(split: str, b1_dir: Path = B1_DIR) -> list[dict]:
    rows = load_jsonl(b1_dir / f"labels_{split}.jsonl")
    return [r for r in rows if r.get("units")]


def unit_rows(unit_set: str, split: str) -> list[dict]:
    return acu_rows(split) if unit_set == "acu" else m2_rows(split)


def truth_of(r: dict) -> list[int | None]:
    """1 = omitted, None = unlabelled."""
    return [None if v is None else 1 - int(v) for v in r["unit_labels"]]


# ----------------------------------------------------------------------------- unit_scores io
def write_scores(path: Path, system: str, unit_set: str, per_pair: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for pid, sc in per_pair.items():
            fh.write(json.dumps({"pair_id": pid, "system": system, "unit_set": unit_set,
                                 "scores": [None if (x is None or not np.isfinite(x)) else float(x) for x in sc]}) + "\n")


def read_scores(path: Path) -> tuple[str, dict[str, np.ndarray]]:
    name, out = None, {}
    for r in load_jsonl(path):
        name = r["system"]
        out[r["pair_id"]] = np.array([np.nan if x is None else x for x in r["scores"]], dtype=float)
    return name, out


def counters_for(unit_set: str, r: dict, i: int) -> dict:
    """Counter reference for one unit (the B4 protocol, adapted to supplied ACUs)."""
    u = r["units"][i]; cand = r["candidate"]
    c = {"ca_cand_words": float(len(cand.split())), "ca_cand_chars": float(len(cand)),
         "ca_unit_words": float(len(u["text"].split())),
         "ca_token_recall": token_recall(u["text"], cand), "ca_rougeL_recall": rougeL_recall(u["text"], cand),
         "ca_unit_in_cand": 1.0 if re.search(r"\b" + re.escape(u["text"]) + r"\b", cand, flags=re.I) else 0.0}
    if unit_set == "acu":
        c["acu_relpos"] = i / max(1, len(r["units"]) - 1)
    else:
        w16 = _load("w16_natural_counter_reference")
        c.update(w16.summary_blind_features(r["source"], u["text"]))
        c["kind_entity"] = 1.0 if u.get("kind") == "entity" else 0.0
    return c


def dump_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))
