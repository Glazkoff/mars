#!/usr/bin/env python3
"""B10 -- the 2025/26 LLM-AggreFact leaders as MARS-C verifiers: Bespoke-MiniCheck-7B and FactCG-DeBERTa-L.

MARS_PUBLICATION_PLAN_2026-09-20.md Sec. 3.2, row 2. The paper's substitutable-verifier table stops at the
2022-24 generation (RoBERTa-MNLI, SummaC-ZS, MiniCheck-Flan-T5, AlignScore, FactCC). A reviewer holding the
2026 LLM-AggreFact leaderboard will ask why the two current leaders are missing. These are them: Bespoke
MiniCheck-7B (77.4 BAcc) and FactCG-DeBERTa-v3-Large (75.6 BAcc).

SAME CONTRACT AS EVERY OTHER OFF-THE-SHELF ARM
----------------------------------------------
Rows, premise and claim are produced exactly as in `b0_zs_score_units.py`:
  * rows     -- `r.get("units")` rows of the frozen inventory, file order
  * premise  -- `mc_score_units.premises`, IMPORTED, so the `shuffled` derangement is the paired one
  * claim    -- `common.render_claim(source, unit)` capped at --claim-words
  * output   -- omission = 1 - support via `common.write_scores`, system `nli:<name>` (+ `_shuf` / `_empty`)
so `mc_diversity.py` -> `mc_e2_eval.py` consume these arms unchanged.

EACH MODEL IS DRIVEN BY ITS OWN PUBLISHED INPUT CONTRACT, NOT BY A GUESS
------------------------------------------------------------------------
Both of these are easy to run *wrongly* in a way that looks like a weak result rather than a bug:

* **FactCG** ships `num_labels: null`, no `id2label`, and -- the trap -- is NOT a sentence-pair classifier.
  `factcg/inference.py` feeds it ONE string built from `factcg/utils.INSTRUCTION_TEMPLATE` and reads
  `softmax(logits)[:, 1]`. Encoded as a plain `(premise, claim)` pair it returns a near-constant 0.90 and
  scores 0.508 BAcc on LLM-AggreFact -- chance, and indistinguishable from "this model is useless" unless
  you check. Under its own template the same weights give 0.811 BAcc on a 2,000-item sample (measured here,
  2026-09-21) against 0.756 published. The template below is quoted from the repo, and `--validate-aggrefact`
  re-measures it before any inventory is scored.

* **Bespoke-MiniCheck-7B** is a generative checker: its support probability is the first-token probability of
  "Yes" under the official system/user prompts. Those prompts are LOADED from the installed `minicheck`
  package (a three-line constants module, exec'd without importing the package, so the vLLM venv stays clean)
  rather than retyped, and the scoring reproduces `LLMCheck.get_support_prob`.

Sealed TEST: scoring `--units test=...` re-reads the held-out split, so the sbatch gates on MC_TEST_REREAD=yes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402

# Quoted from factcg/utils.py at derenlei/FactCG@main -- the single-string input contract of the DeBERTa head.
FACTCG_TEMPLATE = ("{text_a}\n\nChoose your answer: based on the paragraph above can we conclude that "
                   "\"{text_b}\"?\n\nOPTIONS:\n- Yes\n- No\nI think the answer is ")

MODESUF = {"real": "", "shuffled": "_shuf", "empty": "_empty"}


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


msu = _load(REPO / "research" / "marsc_20260916" / "code" / "mc_score_units.py")


def minicheck_prompts(utils_path: Path) -> tuple[str, str]:
    """SYSTEM_PROMPT / USER_PROMPT out of the installed minicheck package without importing it."""
    ns: dict = {}
    exec(compile(utils_path.read_text(encoding="utf-8"), str(utils_path), "exec"), ns)
    return ns["SYSTEM_PROMPT"], ns["USER_PROMPT"]


# ----------------------------------------------------------------------------- FactCG (HF cross-encoder)
class FactCG:
    name = "factcg"

    def __init__(self, snapshot: str, batch: int, device: str):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        cfg = AutoConfig.from_pretrained(snapshot, num_labels=2, finetuning_task="text-classification",
                                         local_files_only=True)
        cfg.problem_type = "single_label_classification"
        self.tok = AutoTokenizer.from_pretrained(snapshot, use_fast=True, local_files_only=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(
            snapshot, config=cfg, local_files_only=True).eval().to(device)
        self.batch, self.device, self.max_len = batch, device, 2048
        self.n_truncated = 0

    def _fit(self, doc: str, claim: str) -> str:
        """Render the template with the DOCUMENT truncated to fit, never the formatted string.

        The repo's own call is `tok(text_list, max_length=2048, truncation='only_first')` on a SINGLE
        sequence, and HF truncates a single sequence from the right. In this template the document comes
        first and everything the classifier actually has to read -- "can we conclude that <claim>", the
        OPTIONS block and the "I think the answer is " cue -- comes last, so right-truncation silently
        deletes the claim and leaves the model scoring a bare document. Inventory premises are ~700 tokens
        and never hit it, but LLM-AggreFact documents reach 11k tokens, which is precisely where the
        benchmark gate runs. So the document is cut to the budget the claim and scaffold leave.
        """
        tail = FACTCG_TEMPLATE.format(text_a="", text_b=claim)
        n_tail = len(self.tok(tail, add_special_tokens=False)["input_ids"])
        budget = self.max_len - n_tail - 4                      # 4 = special tokens + slack
        d_ids = self.tok(doc, add_special_tokens=False)["input_ids"]
        if budget > 0 and len(d_ids) > budget:
            doc = self.tok.decode(d_ids[:budget])
            self.n_truncated += 1
        return FACTCG_TEMPLATE.format(text_a=doc, text_b=claim)

    def score(self, docs: list[str], claims: list[str]) -> list[float]:
        out: list[float] = []
        with self.torch.no_grad():
            for s in range(0, len(docs), self.batch):
                txt = [self._fit(d, c)
                       for d, c in zip(docs[s:s + self.batch], claims[s:s + self.batch])]
                enc = self.tok(txt, max_length=self.max_len, truncation=True, padding=True,
                               return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                out.extend(self.torch.softmax(logits.float(), -1)[:, 1].cpu().numpy().tolist())
        if self.n_truncated:
            print(f"[b10] factcg: {self.n_truncated} documents truncated to fit the claim in 2048 tokens",
                  flush=True)
        return out


AGGREFACT_POOL = 2000          # the seeded pool both branches draw from, so n only ever takes a prefix
AGGREFACT_SEED = 20260921


def load_aggrefact(n: int, cache: str | None) -> tuple[list[str], list[str], np.ndarray]:
    """The benchmark sample, from a pre-dumped jsonl when `datasets` is not importable.

    The vLLM venv is deliberately minimal and has no `datasets`, so the MiniCheck-7B arm would otherwise skip
    the one check that catches a wrong input contract. `--aggrefact-jsonl` lets the same seeded sample be
    dumped once from the mars venv and re-used verbatim.

    BOTH branches draw a pool of AGGREFACT_POOL under one seed and then take the first `n`. They have to:
    `rng.choice(N, size=n)` is NOT a prefix of `rng.choice(N, size=2000)` from the same seed, so taking
    `[:n]` of the dumped file while the dataset branch re-draws at size n would gate the two verifiers on
    different items -- which is exactly what the docstring promises does not happen.
    """
    if cache and Path(cache).exists():
        rows = [json.loads(l) for l in open(cache, encoding="utf-8")][:n]
        return ([r["doc"] for r in rows], [r["claim"] for r in rows],
                np.array([int(r["label"]) for r in rows], dtype=int))
    from datasets import load_dataset
    ds = load_dataset("lytang/LLM-AggreFact", split="test")
    rng = np.random.default_rng(AGGREFACT_SEED)
    idx = rng.choice(len(ds), size=min(AGGREFACT_POOL, len(ds)), replace=False)[:n]
    sub = ds.select(idx)
    return list(sub["doc"]), list(sub["claim"]), np.array(sub["label"], dtype=int)


def validate_aggrefact(scorer, n: int, cache: str | None = None, floor: float = 0.0,
                       published: float | None = None) -> dict:
    """Re-measure the published benchmark before the model is allowed to score the inventory.

    TWO gates, because the orientation check alone is not one. `b < inv` is algebraically `BAcc < 0.5`, so
    it fires only on a model that is worse than chance. The exact failure this function exists to catch --
    FactCG driven with the wrong input contract -- scored 0.508, which sails through. So an ABSOLUTE floor
    is required as well: a model that cannot get within reach of its published number is being run wrong,
    whatever its orientation.
    """
    docs, claims, y = load_aggrefact(n, cache)
    sup = np.array(scorer.score(docs, claims))
    pred = (sup > 0.5).astype(int)
    pos, neg = y == 1, y == 0
    b = float(0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean()))
    inv = float(0.5 * (((1 - pred)[pos] == 1).mean() + ((1 - pred)[neg] == 0).mean()))
    print(f"[b10] LLM-AggreFact n={len(y)}: BAcc {b:.4f} (inverted convention would give {inv:.4f}); "
          f"support mean {sup.mean():.4f}"
          + (f"; published {published:.3f}" if published else ""), flush=True)
    if b < inv:
        raise SystemExit("[b10] ABORT: the inverted convention scores better -- support/omission is flipped")
    if floor and b < floor:
        raise SystemExit(f"[b10] ABORT: BAcc {b:.4f} is below the floor {floor:.4f}. Chance on this "
                         f"benchmark is ~0.50 and the published figure for this model is "
                         f"{published if published else 'in the 0.75-0.78 range'}; a score in between means "
                         f"the model is being driven with the wrong input contract, not that it is weak.")
    return {"n": int(len(y)), "bacc": b, "bacc_inverted": inv, "support_mean": float(sup.mean()),
            "floor": floor, "published": published}


# ----------------------------------------------------------------------------- Bespoke-MiniCheck-7B (vLLM)
class MiniCheck7B:
    name = "minicheck7b"

    @staticmethod
    def _patch_vllm_internlm2() -> bool:
        """vLLM 0.29's own InternLM2ForCausalLM.forward() declares

            intermediate_tensors: IntermediateTensors | None      # no default

        while every other model in that package -- including the InternLM2Model it wraps, one class up in the
        same file -- defaults it to None. The model runner calls forward() without it during CUDA-graph
        memory profiling, so the engine dies with `missing 1 required positional argument`
        (jobs 5660, 5674, 5677). It is an upstream signature bug, not a property of this checkpoint.

        Rather than editing the shared venv, the default is restored in-process. That only reaches the model
        if the engine runs in this process, so the caller must set VLLM_ENABLE_V1_MULTIPROCESSING=0; a patch
        applied here would not survive the fork into a separate EngineCore.
        """
        import inspect
        from vllm.model_executor.models import internlm2 as _il2
        cls = _il2.InternLM2ForCausalLM
        p = inspect.signature(cls.forward).parameters.get("intermediate_tensors")
        if p is None or p.default is not inspect.Parameter.empty:
            return False
        orig = cls.forward

        def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
            return orig(self, input_ids, positions, intermediate_tensors, inputs_embeds)

        cls.forward = forward
        return True

    def __init__(self, snapshot: str, max_model_len: int, gpu_util: float, tp: int, minicheck_utils: Path,
                 trust_remote_code: bool = False):
        patched = self._patch_vllm_internlm2()
        print(f"[b10] vllm internlm2 forward default patched: {patched}", flush=True)
        from vllm import LLM, SamplingParams
        self.system_prompt, self.user_prompt = minicheck_prompts(minicheck_utils)
        # Bespoke-MiniCheck-7B is an InternLM2 checkpoint. The official package passes trust_remote_code=True,
        # which makes vLLM load the checkpoint's bundled modelling code; under vLLM 0.29 that class's
        # forward() is missing `intermediate_tensors` and the engine dies during CUDA-graph profiling
        # (job 5660). vLLM implements InternLM2ForCausalLM natively, so the default here is to use that.
        self.llm = LLM(model=snapshot, tokenizer=snapshot, dtype="bfloat16", seed=2024,
                       trust_remote_code=trust_remote_code, tensor_parallel_size=tp,
                       max_model_len=max_model_len, gpu_memory_utilization=gpu_util,
                       enable_prefix_caching=True, disable_log_stats=True)
        self.tok = self.llm.get_tokenizer()
        self.sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=5)
        self.max_model_len = max_model_len

    def render(self, doc: str, claim: str) -> str:
        user = self.user_prompt.replace("[DOCUMENT]", doc).replace("[CLAIM]", claim)
        msg = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)

    def score(self, docs: list[str], claims: list[str]) -> list[float]:
        prompts = [self.render(d, c) for d, c in zip(docs, claims)]
        # Tokenise HERE, with add_special_tokens=False, and hand vLLM the ids rather than the string.
        # `render` returns a chat template that already begins with <s>; letting vLLM re-tokenise that
        # string applies the tokenizer's default add_special_tokens=True and prepends a SECOND <s>
        # (measured on this checkpoint: [1, 1, 92543, ...] against [1, 92543, ...]). Every score produced
        # before this fix carried the duplicate, so those runs were withdrawn and rerun.
        ids = [self.tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        lens = [len(x) for x in ids]
        over = sum(1 for n in lens if n >= self.max_model_len)
        print(f"[b10] minicheck7b prompt tokens: max {max(lens)} mean {sum(lens) / len(lens):.0f}, "
              f"{over} at/over {self.max_model_len}", flush=True)
        if over:
            raise SystemExit(f"[b10] ABORT: {over} prompts reach the context window; the official path would "
                             f"chunk the document, which this inventory's 380-word premises never need")
        outs = self.llm.generate([{"prompt_token_ids": x} for x in ids], self.sp)
        # LLMCheck.get_support_prob: sum exp(logprob) over first-token candidates that decode to 'yes'.
        # A unit where no 'yes' variant is in the top-k gets support exactly 0 and therefore the MAXIMUM
        # omission score, which puts it in a tied block at the head of the emission ranking where the order
        # is decided by the emission rule, not by the verifier. That is a property worth knowing about any
        # generative checker used as a ranker, so the size of the block is counted and reported rather than
        # left implicit.
        sup: list[float] = []
        floored = 0
        for o in outs:
            p = 0.0
            for tp_ in o.outputs[0].logprobs[0].values():
                if (tp_.decoded_token or "").strip().lower() == "yes":
                    p += math.exp(tp_.logprob)
            if p == 0.0:
                floored += 1
            sup.append(p)
        self.last_floored = floored
        print(f"[b10] minicheck7b support==0 (no 'yes' in top-{self.sp.logprobs}): "
              f"{floored}/{len(sup)} ({floored / max(len(sup), 1):.2%})", flush=True)
        return sup


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", required=True, help="name=path inventory files")
    ap.add_argument("--system", required=True, choices=["factcg", "minicheck7b"])
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--candidate-mode", choices=["real", "shuffled", "empty"], default="real")
    ap.add_argument("--shuffle-seed", type=int, default=20260917)
    ap.add_argument("--empty-text", default="")
    ap.add_argument("--scores-out", required=True)
    ap.add_argument("--claim-words", type=int, default=96)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--minicheck-utils",
                    default="/home/user/.venv-mars/lib/python3.12/site-packages/minicheck/utils.py")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="load the checkpoint's bundled modelling code instead of vLLM's native "
                         "implementation; see MiniCheck7B.__init__ for why this is off by default")
    ap.add_argument("--validate-aggrefact", type=int, default=0,
                    help="re-measure LLM-AggreFact BAcc on N items before scoring (0 = skip)")
    ap.add_argument("--aggrefact-floor", type=float, default=0.70,
                    help="refuse to score the inventory if the re-measured BAcc falls below this. Chance is "
                         "~0.50 and both models publish 0.75-0.78, so anything below 0.70 means a broken "
                         "input contract rather than a weak model")
    ap.add_argument("--aggrefact-jsonl", default=None,
                    help="pre-dumped benchmark sample (doc/claim/label per line); used when `datasets` is "
                         "not importable, as in the vLLM venv")
    ap.add_argument("--dump-aggrefact", default=None,
                    help="write the seeded benchmark sample to this jsonl and exit")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.dump_aggrefact:
        docs, claims, y = load_aggrefact(a.validate_aggrefact or AGGREFACT_POOL, None)
        dp = Path(a.dump_aggrefact); dp.parent.mkdir(parents=True, exist_ok=True)
        with open(dp, "w", encoding="utf-8") as fh:
            for d, c, l in zip(docs, claims, y):
                fh.write(json.dumps({"doc": d, "claim": c, "label": int(l)}) + "\n")
        print(f"[b10] dumped {len(docs)} benchmark rows -> {dp}")
        return

    out = Path(a.scores_out)
    out.mkdir(parents=True, exist_ok=True)
    mode = a.candidate_mode

    if a.system == "factcg":
        import torch
        dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        scorer = FactCG(a.snapshot, a.batch, dev)
    else:
        scorer = MiniCheck7B(a.snapshot, a.max_model_len, a.gpu_memory_utilization,
                             a.tensor_parallel_size, Path(a.minicheck_utils), a.trust_remote_code)

    _PUBLISHED = {"factcg": 0.756, "minicheck7b": 0.774}
    bench = validate_aggrefact(scorer, a.validate_aggrefact, a.aggrefact_jsonl,
                               a.aggrefact_floor, _PUBLISHED.get(a.system)) \
        if a.validate_aggrefact else None

    for spec in a.units:
        uname, path = spec.split("=", 1)
        rows = [r for r in common.load_jsonl(Path(path)) if r.get("units")]
        if a.limit:
            rows = rows[:a.limit]
        prem = msu.premises(rows, a)
        flat = [(i, j, " ".join(common.render_claim(r["source"], u).split()[:a.claim_words]))
                for i, r in enumerate(rows) for j, u in enumerate(r["units"])]
        print(f"[b10] {uname} mode={mode} system={a.system}: {len(rows)} pairs, {len(flat)} units", flush=True)

        t0 = time.time()
        sup = scorer.score([prem[i] for i, _, _ in flat], [c for _, _, c in flat])
        el = time.time() - t0
        assert len(sup) == len(flat), f"{len(sup)} scores for {len(flat)} units"
        print(f"[b10] {a.system} {len(flat)} units in {el:.0f}s ({len(flat) / max(el, 1e-9):.1f} units/s)",
              flush=True)

        per = {r["pair_id"]: [np.nan] * len(r["units"]) for r in rows}
        for (i, j, _), v in zip(flat, sup):
            per[rows[i]["pair_id"]][j] = 1.0 - float(v)
        sysname = f"nli:{a.system}"
        fn = f"{uname}_{sysname.replace(':', '_')}_{mode}.jsonl"
        common.write_scores(out / fn, f"{sysname}{MODESUF[mode]}", uname, per)
        print(f"[b10] wrote {fn}", flush=True)

        receipt = {"system": f"{sysname}{MODESUF[mode]}", "model": a.system, "snapshot": a.snapshot,
                   "candidate_mode": mode, "shuffle_seed": a.shuffle_seed, "unit_set": uname,
                   "units_file": path, "n_pairs": len(rows), "n_units": len(flat),
                   "claim_words": a.claim_words, "seconds": el,
                   "units_per_s": len(flat) / max(el, 1e-9), "aggrefact_validation": bench,
                   "factcg_template_sha256": hashlib.sha256(FACTCG_TEMPLATE.encode()).hexdigest()
                   if a.system == "factcg" else None}
        (out / fn.replace(".jsonl", ".receipt.json")).write_text(json.dumps(receipt, indent=1))

    print("[b10] done", flush=True)


if __name__ == "__main__":
    main()
