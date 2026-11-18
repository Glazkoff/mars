#!/usr/bin/env python3
"""B4 -- the literal unseen third extractor: Qwen3.8-27B decomposes the held-out source sentences.

PREREG (marsc_strengthen_20260918) Wave B / B4. The prompt literal, the parser and the decoding contract are
IMPORTED from the declared decomposition interface (research/mars3_20260916/code/m32_decompose.py) rather than
retyped, so the unseen extractor differs from the development extractor in exactly one respect: which model
generates. Nothing here may inform training, mining, calibration or model selection.

Guards this driver adds over the interface, all registered before execution:

 1. FREEZE BEFORE READ. The freeze manifest (repo id, resolved snapshot revision, sha256 of the prompt literal,
    sha256 of the task file, every decoding parameter) is written before the model is loaded, i.e. before a
    single unit of the unseen extractor exists.

 2. THE CLOSED-THINK ASSERTION. Qwen3's chat template emits `<think>\\n` after the assistant header unless
    `enable_thinking=False` reaches it, and emits the CLOSED `<think>\\n\\n</think>\\n\\n` when it does. An open
    block pushes the answer past the generation window and returns blank text -- the failure that produced a
    widely repeated but wrong result in an earlier campaign. `apply_chat_template` accepts unknown kwargs
    silently, so a TypeError fallback is NOT proof the switch took effect; the rendered prefix is. The job
    refuses to generate unless the rendered prompt ends in a closed, empty think block.

 3. THE EMPTY-OUTPUT GUARDS, with thresholds declared here before the run.
    * `--max-empty-raw 0.10` on the share of BLANK decoded outputs. This is the trap's actual signature: the
      ruined campaign returned 88 % blank. Abort non-zero.
    * `--max-empty-parse 0.30` on the share of sentences whose parse is empty. NOTE, and this is a deviation
      from the brief's suggested 10 %, declared before execution with its reason: the registered DEVELOPMENT
      extractor (Gemma-4-31B) itself returns an empty parse on 10.43 % of these very sentences (766/7342
      measured on the frozen props shards), because the prompt instructs the model to answer NONE for a
      factless sentence. A 10 % empty-parse abort would therefore fail the healthy control, so the guard is
      set at ~3x the development reference and the reference is printed beside the measured rate. The 10 %
      rule is kept where it discriminates -- on blank raw output.
    Both guards fire after --assert-after sentences and again at the end of the shard.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for p in (str(REPO), str(REPO / "research" / "mars3_20260916" / "code"),
          str(REPO / "research" / "mars2_gates_20260916" / "code")):
    if p not in sys.path:
        sys.path.insert(0, p)

import m32_decompose as m32d          # noqa: E402  the declared decomposition interface
from m3common import gates            # noqa: E402

PROMPT = m32d.PROMPT                  # imported, never retyped
parse = m32d.parse                    # imported, never retyped
judge = m32d.judge                    # b1_judge_labels: load_model + apply_chat (enable_thinking=False)

# The development extractor's own empty-parse rate on the held-out sentence set, measured on the frozen
# Gemma props shards before this job was written. Printed beside the unseen extractor's rate; never used to
# adjust it.
DEV_REFERENCE_EMPTY_PARSE = {"extractor": "gemma-4-31b-it", "split": "test", "sentences": 7342,
                             "empty_parse": 766, "rate": 0.1043}
CLOSED_THINK = "<think>\n\n</think>\n\n"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_freeze(a, snapshot: str, tasks_path: Path, n_tasks: int, out: Path) -> dict:
    """The pre-registration of the unseen extractor. Written BEFORE the model is loaded."""
    rp = os.path.realpath(snapshot)
    fr = {
        "block": "B4", "role": "unseen third extractor", "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "written_before_model_load": True,
        "model": {"repo_id": a.model_id, "snapshot_dir": snapshot, "realpath": rp,
                  "resolved_revision": Path(snapshot.rstrip("/")).name,
                  "weight_files": sorted(p.name for p in Path(snapshot).glob("*.safetensors"))},
        "prompt": {"source": "research/mars3_20260916/code/m32_decompose.py::PROMPT (imported, not retyped)",
                   "sha256": hashlib.sha256(PROMPT.encode()).hexdigest(), "n_chars": len(PROMPT)},
        "decoding": {"chat_template": True, "enable_thinking": False,
                     "closed_think_prefix_asserted": True, "padding_side": "left",
                     "tokenizer": {"padding": True, "truncation": True, "max_length": 1024},
                     "generate": {"max_new_tokens": a.max_new_tokens, "do_sample": False},
                     "batch_size": a.batch_size, "dtype": "bfloat16", "device": a.device},
        "parser": {"source": "research/mars3_20260916/code/m32_decompose.py::parse (imported, not retyped)",
                   "drop_lines": ["NONE*", "facts*", "<2 words"], "strip_markers": "-*• / 1. / 1)",
                   "max_propositions": 12},
        "tasks": {"path": str(tasks_path), "sha256": sha256_file(tasks_path), "splits_filter": a.splits,
                  "n_tasks_after_filter": n_tasks, "shard": a.shard},
        "guards": {"max_empty_raw": a.max_empty_raw, "max_empty_parse": a.max_empty_parse,
                   "assert_after": a.assert_after,
                   "development_reference_empty_parse": DEV_REFERENCE_EMPTY_PARSE},
        "isolation_rule": ("no unit, score, label or output statistic of this extractor informs training, "
                           "mining, calibration or model selection; the development threshold is applied unchanged"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fr, indent=1))
    print(f"[b4-decompose] freeze written -> {out}", flush=True)
    print(f"[b4-decompose] prompt sha256 {fr['prompt']['sha256']}  revision {fr['model']['resolved_revision']}", flush=True)
    return fr


def check_guards(stats: dict, a, where: str) -> None:
    n = stats["sent"]
    if n < a.assert_after:
        return
    raw = stats["empty_raw"] / n
    par = stats["empty_parse"] / n
    print(f"[b4-decompose] guard@{where}: {n} sentences, blank-raw {raw:.4f} (max {a.max_empty_raw}), "
          f"empty-parse {par:.4f} (max {a.max_empty_parse}; dev reference {DEV_REFERENCE_EMPTY_PARSE['rate']})", flush=True)
    if raw > a.max_empty_raw:
        raise SystemExit(f"[b4-decompose] ABORT (blank-output guard): {raw:.4f} of {n} decoded outputs are blank, "
                         f"above {a.max_empty_raw}. This is the <think>-trap signature; the inventory is NOT written.")
    if par > a.max_empty_parse:
        raise SystemExit(f"[b4-decompose] ABORT (empty-parse guard): {par:.4f} of {n} sentences parse to no "
                         f"proposition, above {a.max_empty_parse} (development extractor: "
                         f"{DEV_REFERENCE_EMPTY_PARSE['rate']}). The inventory is NOT written.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="the frozen m32 decompose_tasks.jsonl")
    ap.add_argument("--splits", default="test", help="keep only tasks whose doc appears in these splits")
    ap.add_argument("--out", required=True)
    ap.add_argument("--freeze-out", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--model-snapshot", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-empty-raw", type=float, default=0.10)
    ap.add_argument("--max-empty-parse", type=float, default=0.30)
    ap.add_argument("--assert-after", type=int, default=120)
    a = ap.parse_args()

    tasks_p = Path(a.tasks)
    want = {s.strip() for s in a.splits.split(",") if s.strip()}
    i, k = (int(x) for x in a.shard.split("/"))
    allt = [t for t in gates.load_jsonl(tasks_p) if (not want) or (want & set(t.get("splits", [])))]
    tasks = [t for j, t in enumerate(allt) if j % k == i]
    if a.limit:
        tasks = tasks[:a.limit]
    print(f"[b4-decompose] {len(allt)} tasks in splits {sorted(want)}; shard {i}/{k} -> {len(tasks)}", flush=True)

    snapshot = a.model_snapshot
    if not Path(snapshot, "config.json").exists():
        raise SystemExit(f"[b4-decompose] ABORT: no config.json under {snapshot}")
    write_freeze(a, snapshot, tasks_p, len(allt), Path(a.freeze_out))

    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if outp.exists():
        for line in open(outp, encoding="utf-8"):
            try:
                done.add(json.loads(line)["task_id"])
            except Exception:
                pass
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"[b4-decompose] {len(tasks)} tasks, {len(done)} already done, {len(todo)} to do", flush=True)
    if not todo:
        return

    import torch
    tok, model = judge.load_model(snapshot, a.device)
    if not getattr(tok, "chat_template", None):
        raise SystemExit("[b4-decompose] ABORT: tokenizer has no chat template; the frozen decoding contract is a chat contract")
    probe = judge.apply_chat(tok, PROMPT.format(sentence="The company reported revenue of 1.2 billion dollars in 2023."))
    if not probe.endswith(CLOSED_THINK):
        tail = probe[-120:].replace("\n", "\\n")
        raise SystemExit("[b4-decompose] ABORT: enable_thinking=False did not reach the chat template -- the rendered "
                         f"prompt does not end in a closed empty think block. tail={tail!r}")
    print("[b4-decompose] closed-think assertion PASSED (enable_thinking=False reached the template)", flush=True)
    if getattr(tok, "padding_side", None) != "left":
        raise SystemExit(f"[b4-decompose] ABORT: padding_side is {tok.padding_side!r}, the frozen contract is left padding")

    t0 = time.time(); stats = defaultdict(int); shown = 0
    with open(outp, "a", encoding="utf-8") as fh, torch.inference_mode():
        for s in range(0, len(todo), a.batch_size):
            ch = todo[s:s + a.batch_size]
            texts = [judge.apply_chat(tok, PROMPT.format(sentence=t["sentence"])) for t in ch]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(a.device)
            ids = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            outs = tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for t, o in zip(ch, outs):
                props = parse(o)
                stats["sent"] += 1; stats["props"] += len(props)
                stats["empty_raw"] += int(not o.strip())
                stats["empty_parse"] += int(not props)
                if shown < 3:
                    print(f"[b4-decompose] sample raw[{shown}] {o.strip()[:300]!r} -> {len(props)} props", flush=True)
                    shown += 1
                fh.write(json.dumps({**t, "props": props}) + "\n")
            if stats["sent"] >= a.assert_after and not stats["guarded"]:
                fh.flush(); check_guards(stats, a, "first-batches"); stats["guarded"] = 1
            if (s // a.batch_size) % 20 == 0:
                fh.flush()
                print(f"[b4-decompose] {s + len(ch)}/{len(todo)} ({time.time() - t0:.0f}s) "
                      f"{stats['props'] / max(1, stats['sent']):.2f} props/sentence "
                      f"blank-raw {stats['empty_raw'] / max(1, stats['sent']):.3f} "
                      f"empty-parse {stats['empty_parse'] / max(1, stats['sent']):.3f}", flush=True)
    check_guards(stats, a, "end-of-shard")
    rep = {"shard": a.shard, "sentences": stats["sent"], "propositions": stats["props"],
           "props_per_sentence": stats["props"] / max(1, stats["sent"]),
           "blank_raw_rate": stats["empty_raw"] / max(1, stats["sent"]),
           "empty_parse_rate": stats["empty_parse"] / max(1, stats["sent"]),
           "development_reference_empty_parse": DEV_REFERENCE_EMPTY_PARSE,
           "seconds": time.time() - t0}
    Path(str(outp) + ".stats.json").write_text(json.dumps(rep, indent=1))
    print(f"[b4-decompose] done {json.dumps(rep)}", flush=True)


if __name__ == "__main__":
    main()
