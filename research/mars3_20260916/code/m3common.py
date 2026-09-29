#!/usr/bin/env python3
"""Shared helpers for MARS-3 (M31 tagger / writer, M32 inventory). Re-uses the gates helpers for data and io."""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common as gates  # noqa: E402

M3_OUT = Path(os.environ.get("MARS3_OUT", "/home/user/results/mars3"))
_WORD = re.compile(r"[A-Za-z0-9]+(?:'[a-z]+)?")


def snap(model_id: str) -> str:
    """Resolve a cached HF snapshot dir (offline clusters), else return the id."""
    hub = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub"
    d = hub / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if d.exists():
        snaps = sorted(d.iterdir(), key=lambda p: p.stat().st_mtime)
        if snaps:
            return str(snaps[-1])
    return model_id


def word_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0).lower()) for m in _WORD.finditer(text)]


def acu_token_mask(source: str, acu: str, offsets: list[tuple[int, int]], is_source: list[bool]) -> np.ndarray:
    """Boolean mask over tokens: source tokens inside the ACU's best-matching sentence whose word is an ACU
    content token; falls back to every token of that sentence (inventory-free scoring of a supplied unit)."""
    s, e = gates.best_sentence_span(source, acu)
    content = set(gates.tokens(acu, content_only=True)) or set(gates.tokens(acu))
    words = [(a, b, w) for a, b, w in word_spans(source) if a < e and b > s]
    hits = np.zeros(len(offsets), dtype=bool); in_sent = np.zeros(len(offsets), dtype=bool)
    for i, ((o0, o1), src) in enumerate(zip(offsets, is_source)):
        if not src or o1 <= o0 or o1 <= s or o0 >= e:
            continue
        in_sent[i] = True
        for a, b, w in words:
            if o0 < b and o1 > a and w in content:
                hits[i] = True; break
    return hits if hits.any() else in_sent


def span_token_mask(start: int, end: int, offsets, is_source) -> np.ndarray:
    return np.array([bool(src and o1 > o0 and o0 < end and o1 > start) for (o0, o1), src in zip(offsets, is_source)])


def frac_keep(pair_id: str, source: str, frac: float, seed: int) -> bool:
    """Document-level subsampling of the training set (label-efficiency curve), stable across arms."""
    if frac >= 1.0:
        return True
    h = int(hashlib.sha256(f"{seed}|{source[:200]}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < frac


def safe_mean(x: np.ndarray, default: float = 0.5) -> float:
    return float(np.mean(x)) if len(x) else default


def prop_token_mask(source: str, text: str, start: int, end: int, offsets, is_source) -> np.ndarray:
    """Tokens of the sentence [start, end) whose word is a content token of the proposition; falls back to the whole
    sentence. Lets several propositions of one sentence receive different (partly overlapping) token sets."""
    content = set(gates.tokens(text, content_only=True)) or set(gates.tokens(text))
    words = [(a, b, w) for a, b, w in word_spans(source) if a < end and b > start]
    hits = np.zeros(len(offsets), dtype=bool); in_sent = np.zeros(len(offsets), dtype=bool)
    for i, ((o0, o1), src) in enumerate(zip(offsets, is_source)):
        if not src or o1 <= o0 or o1 <= start or o0 >= end:
            continue
        in_sent[i] = True
        for a, b, w in words:
            if o0 < b and o1 > a and w in content:
                hits[i] = True; break
    return hits if hits.any() else in_sent
