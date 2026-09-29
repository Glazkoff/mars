"""Inventories: a text -> atomic facts with character spans.

SentenceDecomposer  sentences as facts; no model. The fallback every installation has.
LLMDecomposer       the frozen MARS recipe: an instruction model decomposes each sentence into atomic facts (one per
                    line; names, numbers and dates verbatim; NONE for a fact-free sentence). Default prompt = the
                    one the paper's inventories were built with. Any causal LM with a chat template works; the
                    paper used Gemma-4-31B-it.
Seq2SeqDecomposer   a distilled sentence->facts model (LongT5-base in the paper), the cheap inventory.
Every fact carries the span of the sentence it came from, so evidence can be highlighted in the text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")
_LEAD = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
PROMPT = ("Decompose the sentence into its atomic facts. Rules: one fact per line; each fact is a short standalone sentence "
          "with an explicit subject; keep names, numbers and dates exactly as written; do not add or infer information; "
          "if the sentence states no fact, output NONE.\n\nSentence: {sentence}\n\nFacts:")


@dataclass(frozen=True)
class Fact:
    text: str
    start: int
    end: int


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in _SENT.finditer(text):
        if m.start() > start and text[start:m.start()].strip():
            spans.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans or ([(0, len(text))] if text else [])


def parse_facts(output: str) -> list[str]:
    facts = []
    for line in output.splitlines():
        t = _LEAD.sub("", line).strip()
        if not t or t.upper() == "NONE":
            continue
        facts.append(t)
    return facts


class Decomposer:
    def facts(self, text: str) -> list[Fact]:  # pragma: no cover - interface
        raise NotImplementedError

    def facts_batch(self, texts: list[str]) -> list[list[Fact]]:
        return [self.facts(t) for t in texts]


class SentenceDecomposer(Decomposer):
    name = "sentences"

    def facts(self, text: str) -> list[Fact]:
        return [Fact(text[s:e].strip(), s, e) for s, e in sentence_spans(text) if text[s:e].strip()]


@dataclass
class LLMDecomposer(Decomposer):
    path: str
    batch: int = 12
    max_new_tokens: int = 160
    device: str | None = None
    prompt: str = PROMPT
    name: str = "llm"

    def __post_init__(self):
        self._tok = None; self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.path); self._tok.padding_side = "left"
        self._model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.bfloat16 if self.device == "cuda" else None).eval().to(self.device)

    def _render(self, sentence: str) -> str:
        p = self.prompt.format(sentence=sentence)
        if getattr(self._tok, "chat_template", None):
            return self._tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        return p

    def facts_batch(self, texts: list[str]) -> list[list[Fact]]:
        import torch
        self._load()
        jobs = [(ti, s, e, texts[ti][s:e].strip()) for ti, t in enumerate(texts) for s, e in sentence_spans(t) if t[s:e].strip()]
        out: list[list[Fact]] = [[] for _ in texts]
        with torch.inference_mode():
            for b in range(0, len(jobs), self.batch):
                ch = jobs[b:b + self.batch]
                enc = self._tok([self._render(j[3]) for j in ch], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.device)
                ids = self._model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self._tok.pad_token_id)
                dec = self._tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for (ti, s, e, _), o in zip(ch, dec):
                    out[ti].extend(Fact(f, s, e) for f in parse_facts(o))
        return out

    def facts(self, text: str) -> list[Fact]:
        return self.facts_batch([text])[0]


@dataclass
class Seq2SeqDecomposer(Decomposer):
    """The distilled sentence->facts model of the paper (LongT5-base, `m32_distill.py`): input
    `propositions: <sentence>`, output facts joined by ` | ` (NONE for none), decoded with four beams."""
    path: str
    batch: int = 32
    max_new_tokens: int = 192
    num_beams: int = 4
    prefix: str = "propositions: "
    device: str | None = None
    name: str = "seq2seq"

    def __post_init__(self):
        self._tok = None; self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.path)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.path).eval().to(self.device)

    def facts_batch(self, texts: list[str]) -> list[list[Fact]]:
        import torch
        self._load()
        jobs = [(ti, s, e, texts[ti][s:e].strip()) for ti, t in enumerate(texts) for s, e in sentence_spans(t) if t[s:e].strip()]
        out: list[list[Fact]] = [[] for _ in texts]
        with torch.inference_mode():
            for b in range(0, len(jobs), self.batch):
                ch = jobs[b:b + self.batch]
                enc = self._tok([self.prefix + j[3] for j in ch], return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
                ids = self._model.generate(**enc, num_beams=self.num_beams, max_new_tokens=self.max_new_tokens, no_repeat_ngram_size=3)
                for (ti, s, e, _), o in zip(ch, self._tok.batch_decode(ids, skip_special_tokens=True)):
                    out[ti].extend(Fact(f.strip(), s, e) for f in o.split("|") if f.strip() and f.strip().upper() != "NONE")
        return out

    def facts(self, text: str) -> list[Fact]:
        return self.facts_batch([text])[0]
