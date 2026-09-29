"""Verifiers: P(hypothesis is supported by premise) under two input contracts.

PromptVerifier  the FactCG contract: one sequence rendered from a prompt template, a two-class head whose class 1 is
                "supported"; the premise is cut to the token budget the hypothesis and scaffold leave (never the
                hypothesis). This is the contract FactCG-DeBERTa-v3-Large was trained on and the one the MARS
                unified verifier keeps.
PairVerifier    the MARS-C contract: (premise, hypothesis) as a sentence pair, the premise read in overlapping
                word windows with the maximum support over windows; a one-logit head scores OMISSION (support is
                one minus its sigmoid), a two-class head scores support in class 1. Several checkpoints (seeds)
                are averaged.
Both expose `support(premises, hypotheses) -> list[float]` and batch in length order.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FACTCG_TEMPLATE = ("{text_a}\n\nChoose your answer: based on the paragraph above can we conclude that "
                   "\"{text_b}\"?\n\nOPTIONS:\n- Yes\n- No\nI think the answer is ")


def windows(text: str, window: int, stride: int) -> list[str]:
    w = text.split()
    if len(w) <= window:
        return [" ".join(w)]
    starts = list(range(0, len(w) - window + 1, stride))
    if starts[-1] + window < len(w):
        starts.append(len(w) - window)
    return [" ".join(w[s:s + window]) for s in starts]


class Verifier:
    name = "verifier"

    def support(self, premises: list[str], hypotheses: list[str]) -> list[float]:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class PromptVerifier(Verifier):
    path: str
    batch: int = 32
    max_len: int = 2048
    device: str | None = None
    template: str = FACTCG_TEMPLATE
    name: str = "prompt"
    _tok: object = field(default=None, repr=False)
    _model: object = field(default=None, repr=False)

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = AutoConfig.from_pretrained(self.path, num_labels=2, finetuning_task="text-classification")
        cfg.problem_type = "single_label_classification"
        self._tok = AutoTokenizer.from_pretrained(self.path, use_fast=True)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForSequenceClassification.from_pretrained(self.path, config=cfg).eval().to(self.device)

    def render(self, premise: str, hypothesis: str) -> str:
        tail = self.template.format(text_a="", text_b=hypothesis)
        n_tail = len(self._tok(tail, add_special_tokens=False)["input_ids"])
        budget = self.max_len - n_tail - 4
        ids = self._tok(premise, add_special_tokens=False)["input_ids"]
        if budget > 0 and len(ids) > budget:
            premise = self._tok.decode(ids[:budget])
        return self.template.format(text_a=premise, text_b=hypothesis)

    def support(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        import torch
        self._load()
        order = sorted(range(len(premises)), key=lambda i: len(premises[i]) + len(hypotheses[i]))
        out = [0.0] * len(premises)
        with torch.inference_mode():
            for s in range(0, len(order), self.batch):
                idx = order[s:s + self.batch]
                enc = self._tok([self.render(premises[i], hypotheses[i]) for i in idx], max_length=self.max_len,
                                truncation=True, padding=True, return_tensors="pt").to(self.device)
                p = torch.softmax(self._model(**enc).logits.float(), -1)[:, 1].cpu().numpy()
                for i, v in zip(idx, p):
                    out[i] = float(v)
        return out


@dataclass
class PairVerifier(Verifier):
    paths: list[str]
    batch: int = 128
    max_len: int = 512
    window: int = 380
    stride: int = 190
    device: str | None = None
    name: str = "pair"
    _tok: object = field(default=None, repr=False)
    _models: list = field(default_factory=list, repr=False)
    _two_class: bool = False

    @classmethod
    def from_glob(cls, pattern: str, **kw) -> "PairVerifier":
        dirs = [d for d in sorted(glob.glob(pattern)) if Path(d, "config.json").exists()]
        if not dirs:
            raise FileNotFoundError(f"no checkpoint matched {pattern!r}")
        return cls(paths=dirs, **kw)

    def _load(self):
        if self._models:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.paths[0])
        self._models = [AutoModelForSequenceClassification.from_pretrained(p).eval().to(self.device) for p in self.paths]
        self._two_class = getattr(self._models[0].config, "num_labels", 1) == 2

    def support(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        import torch
        self._load()
        items = [(i, w, h) for i, (p, h) in enumerate(zip(premises, hypotheses)) for w in windows(p, self.window, self.stride)]
        order = sorted(range(len(items)), key=lambda k: len(items[k][1]) + len(items[k][2]))
        best = np.full(len(premises), -1.0)
        with torch.inference_mode():
            for s in range(0, len(order), self.batch):
                idx = order[s:s + self.batch]
                enc = self._tok([items[k][1] for k in idx], [items[k][2] for k in idx], truncation="longest_first",
                                max_length=self.max_len, padding=True, return_tensors="pt").to(self.device)
                sup = np.zeros(len(idx))
                for m in self._models:
                    lg = m(**enc).logits.float()
                    sup += (torch.softmax(lg, -1)[:, 1] if self._two_class else 1.0 - torch.sigmoid(lg.squeeze(-1))).cpu().numpy()
                sup /= len(self._models)
                for k, v in zip(idx, sup):
                    i = items[k][0]
                    if v > best[i]:
                        best[i] = float(v)
        return best.tolist()


class LexicalVerifier(Verifier):
    """Content-token recall of the hypothesis in the premise. A stand-in for tests and a sanity baseline; it is
    NOT a MARS verifier and the package never uses it by default."""
    name = "lexical"

    @staticmethod
    def _toks(t: str) -> set[str]:
        return {w.strip(".,;:!?\"'()[]").lower() for w in t.split() if len(w) > 3}

    def support(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        out = []
        for p, h in zip(premises, hypotheses):
            ht = self._toks(h); pt = self._toks(p)
            out.append(len(ht & pt) / len(ht) if ht else 0.0)
        return out
