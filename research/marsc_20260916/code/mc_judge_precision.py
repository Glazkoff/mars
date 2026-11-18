#!/usr/bin/env python3
"""MARS-C G2 -- judge-estimated precision of EMITTED facts (PLAN_MAIN_CONTRIBUTION.md, block G2).

76 % of emitted units match no reference ACU; whether they are real omissions is unmeasured. Two forced-choice questions per
emitted unit, two judge families (Gemma-4-31B-it, Qwen3.8-27B; the B1 judge's readout: P(B) among {A, B}):
  coverage  -- the B1 coverage prompt with a TARGET FACT: A = conveyed by the summary, B = omitted
  support   -- A = stated in or directly following from the source, B = not supported
True omission = supported AND omitted. The judge is validated on the human-known emitted units (hit = a human-omitted ACU,
false alert = a human-covered ACU). Stages: build -> run (sharded, resumable) -> analyze.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402

b1 = common._load("b1_judge_labels", REPO / "scripts" / "plan2026")

PROMPTS = {
    "coverage": ("Determine whether the summary conveys the target fact from the source. Equivalent names, abbreviations, pronouns and "
                 "faithful paraphrases count as coverage. Do not require the wording verbatim. Judge coverage, not whether the fact is important. "
                 "A = conveyed or covered. B = omitted or not conveyed. Answer only A or B.\n\nSOURCE:\n{source}\n\nSUMMARY:\n{summary}\n\nTARGET FACT: {target}\n\nAnswer:"),
    "support": ("Determine whether the target fact is stated in the source or follows directly from it. A = supported by the source. "
                "B = not supported (absent from, contradicted by, or not inferable from the source). Answer only A or B.\n\nSOURCE:\n{source}\n\nTARGET FACT: {target}\n\nAnswer:"),
}


# T06 (registered amendment): the plain prompts scored 0.685 (Gemma) / 0.705 (Qwen) against the human labels of the
# blinded G6 sample, below the registered validity bar of 0.85, with kappa 0.457 -- so judged precision could not be
# used. The two failures visible in the disagreements are (a) treating an entity mention as coverage of a fact ABOUT
# that entity and (b) refusing paraphrase. Both are decision-boundary problems, not knowledge problems, so the repair
# is four worked examples that pin the boundary. The examples are written by hand and contain no corpus text, so no
# evaluation item can leak through them; the answer format and the readout are unchanged.
PROMPTS_ANCHORED = {
    "coverage": ("Determine whether the summary conveys the target fact from the source. Equivalent names, abbreviations, pronouns and "
                 "faithful paraphrases count as coverage. Do not require the wording verbatim. Judge coverage, not whether the fact is important. "
                 "Naming the people or things a fact is about is NOT coverage of the fact itself: the summary must convey what the fact asserts.\n\n"
                 "Worked examples:\n"
                 "1. TARGET FACT: 'The mayor resigned on Tuesday.' SUMMARY: 'The mayor stepped down early in the week.' -> A (faithful paraphrase).\n"
                 "2. TARGET FACT: 'The mayor resigned on Tuesday.' SUMMARY: 'The mayor faced growing criticism.' -> B (the mayor is named, the resignation is not conveyed).\n"
                 "3. TARGET FACT: 'Dr. Okafor led the trial.' SUMMARY: 'She led the trial.' where the summary has already introduced Dr. Okafor -> A (pronoun refers to the same person).\n"
                 "4. TARGET FACT: 'Sales fell 12 percent.' SUMMARY: 'Sales fell sharply.' -> B (the direction is conveyed, the asserted quantity is not).\n\n"
                 "A = conveyed or covered. B = omitted or not conveyed. Answer only A or B.\n\nSOURCE:\n{source}\n\nSUMMARY:\n{summary}\n\nTARGET FACT: {target}\n\nAnswer:"),
    "support": ("Determine whether the target fact is stated in the source or follows directly from it. A single obvious inference from "
                "what the source states counts as support; a plausible guess that the source does not license does not.\n\n"
                "Worked examples:\n"
                "1. TARGET FACT: 'The bridge was closed.' SOURCE states 'officials shut the bridge at noon' -> A (stated in other words).\n"
                "2. TARGET FACT: 'The bridge reopened the next morning.' SOURCE says nothing about reopening -> B (not in the source).\n"
                "3. TARGET FACT: 'The company employs more than 500 people.' SOURCE states 'the company's 640 employees' -> A (one direct inference).\n"
                "4. TARGET FACT: 'The company is the largest employer in the region.' SOURCE gives only the headcount -> B (not licensed by the source).\n\n"
                "A = supported by the source. B = not supported (absent from, contradicted by, or not inferable from the source). Answer only A or B."
                "\n\nSOURCE:\n{source}\n\nTARGET FACT: {target}\n\nAnswer:"),
}


def build(a) -> None:
    pairs = {r["pair_id"]: r for r in common.load_jsonl(Path(a.acu))}
    want = set(a.systems); tasks: dict[str, dict] = {}
    for r in common.load_jsonl(Path(a.emitted)):
        if r["system"] not in want or r["pair_id"] not in pairs or not r.get("emitted"):
            continue
        pr = pairs[r["pair_id"]]
        for rank, e in enumerate(r["emitted"]):
            if a.max_rank and rank >= a.max_rank:      # R07: judge only the budget the paper's precision claim is about
                break
            tid = hashlib.sha1((r["pair_id"] + "\x1f" + e["text"].strip().lower()).encode()).hexdigest()[:16]
            t = tasks.setdefault(tid, {"task_id": tid, "pair_id": r["pair_id"], "resource": r["resource"], "source": pr["source"], "candidate": pr["candidate"],
                                       "unit_text": e["text"], "status": {}, "rank": {}})
            t["status"][r["system"]] = e["status"]; t["rank"][r["system"]] = rank
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "tasks.jsonl", "w", encoding="utf-8") as fh:
        for t in tasks.values():
            fh.write(json.dumps(t) + "\n")
    per_sys = defaultdict(int)
    for t in tasks.values():
        for s_ in t["status"]:
            per_sys[s_] += 1
    print(f"[g2-build] {len(tasks)} unique (pair, fact) tasks from {len(want)} emitters; per emitter {dict(per_sys)}", flush=True)


def run(a) -> None:
    import torch
    i, k = (int(x) for x in a.shard.split("/"))
    tasks = [t for j, t in enumerate(common.load_jsonl(Path(a.tasks))) if j % k == i]
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True); done = set()
    if outp.exists():
        for l in open(outp, encoding="utf-8"):
            try:
                done.add(json.loads(l)["task_id"])
            except Exception:
                pass
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"[g2-judge:{a.question}] shard {i}/{k}: {len(tasks)} tasks, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return
    tok, model = b1.load_model(a.model_snapshot, a.device); chat = bool(getattr(tok, "chat_template", None))
    ids_a = sorted({tok.encode(v, add_special_tokens=False)[0] for v in ("A", " A")}); ids_b = sorted({tok.encode(v, add_special_tokens=False)[0] for v in ("B", " B")})
    t0 = time.time(); prompt = (PROMPTS_ANCHORED if getattr(a, "prompt_set", "plain") == "anchored" else PROMPTS)[a.question]
    with open(outp, "a", encoding="utf-8") as fh, torch.inference_mode():
        for s in range(0, len(todo), a.batch_size):
            chunk = todo[s:s + a.batch_size]; texts = []
            for t in chunk:
                src = " ".join(t["source"].split()[:b1.MAX_SOURCE_WORDS])
                p = prompt.format(source=src, summary=t["candidate"], target=t["unit_text"])
                texts.append(b1.apply_chat(tok, p) if chat else p)
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=6144).to(a.device)
            lp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), dim=-1)
            la = torch.logsumexp(lp[:, ids_a], -1); lb = torch.logsumexp(lp[:, ids_b], -1)
            mass = (la.exp() + lb.exp()).cpu().tolist(); pb = torch.softmax(torch.stack([la, lb], -1), -1)[:, 1].cpu().tolist()
            for t, p_b, m in zip(chunk, pb, mass):
                fh.write(json.dumps({"task_id": t["task_id"], "question": a.question, "p_b": float(p_b), "choice_mass": float(m), "model": a.model_snapshot.rstrip("/").split("/")[-1]}) + "\n")
            if (s // a.batch_size) % 50 == 0:
                fh.flush(); print(f"[g2-judge:{a.question}] {s + len(chunk)}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[g2-judge:{a.question}] done in {time.time() - t0:.0f}s", flush=True)


def _kappa(x: np.ndarray, y: np.ndarray) -> float:
    po = float(np.mean(x == y)); pe = float(np.mean(x) * np.mean(y) + (1 - np.mean(x)) * (1 - np.mean(y)))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def analyze(a) -> None:
    tasks = {t["task_id"]: t for t in common.load_jsonl(Path(a.tasks))}
    fam: dict[str, dict[str, dict[str, float]]] = {}                     # family -> question -> task -> p_b
    for spec in a.families:
        name, pat = spec.split("=", 1); fam[name] = {"coverage": {}, "support": {}}
        for f in sorted(glob.glob(pat)):
            for r in common.load_jsonl(Path(f)):
                fam[name][r["question"]][r["task_id"]] = r["p_b"]
    prim_f = a.primary_family
    systems = sorted({s_ for t in tasks.values() for s_ in t["status"]})
    doc_of = {tid: common.doc_key(t["source"]) for tid, t in tasks.items()}
    rep = {"registered": "PLAN_MAIN_CONTRIBUTION.md G2", "n_tasks": len(tasks), "families": {}, "systems": {}, "paired": {}, "kappa_true_omission": None}

    def verdicts(fname):
        q = fam[fname]; out = {}
        for tid in tasks:
            if tid in q["coverage"] and tid in q["support"]:
                out[tid] = {"omitted": q["coverage"][tid] >= 0.5, "supported": q["support"][tid] < 0.5}
        return out

    V = {f: verdicts(f) for f in fam}
    for f, v in V.items():
        known = [(tid, t["status"]) for tid, t in tasks.items() if tid in v and any(st in ("hit", "false_alert") for st in t["status"].values())]
        acc = [1.0 if (v[tid]["omitted"] == any(st == "hit" for st in status.values())) else 0.0 for tid, status in known]
        rep["families"][f] = {"n_judged": len(v), "coverage_accuracy_on_known": float(np.mean(acc)) if acc else None, "n_known": len(acc),
                              "support_rate": float(np.mean([x["supported"] for x in v.values()])) if v else None}
    if len(V) >= 2:
        f1, f2 = list(V)[:2]; both = [tid for tid in V[f1] if tid in V[f2]]
        x = np.array([V[f1][t]["omitted"] and V[f1][t]["supported"] for t in both]); y = np.array([V[f2][t]["omitted"] and V[f2][t]["supported"] for t in both])
        rep["kappa_true_omission"] = {"families": [f1, f2], "n": len(both), "kappa": _kappa(x, y), "agreement": float(np.mean(x == y)) if both else None}

    def per_pair(system, fname):
        by_pair = defaultdict(list); unk = defaultdict(list)
        for tid, t in tasks.items():
            if system in t["status"] and tid in V[fname]:
                vv = V[fname][tid]; tr = float(vv["omitted"] and vv["supported"]); by_pair[t["pair_id"]].append(tr)
                if t["status"][system] == "unknown":
                    unk[t["pair_id"]].append(tr)
        return {p: float(np.mean(v)) for p, v in by_pair.items()}, {p: float(np.mean(v)) for p, v in unk.items()}

    PP = {(s_, f): per_pair(s_, f) for s_ in systems for f in fam}
    for s_ in systems:
        rep["systems"][s_] = {}
        for f in fam:
            pp, unk = PP[(s_, f)]
            rep["systems"][s_][f] = {"true_omission_rate": float(np.mean(list(pp.values()))) if pp else None, "n_pairs": len(pp),
                                     "unknown_resolved_rate": float(np.mean(list(unk.values()))) if unk else None}
    rng = np.random.default_rng(20260916); pp_prim = PP[(a.primary, prim_f)][0]
    pair_doc = {t["pair_id"]: doc_of[tid] for tid, t in tasks.items()}; docs = sorted(set(pair_doc.values()))
    by_doc = defaultdict(list)
    for p, d in pair_doc.items():
        by_doc[d].append(p)
    for s_ in systems:
        if s_ == a.primary:
            continue
        pp_o = PP[(s_, prim_f)][0]; d = []
        for _ in range(a.n_boot):
            pick = rng.choice(docs, size=len(docs), replace=True); sub = [p for dd in pick for p in by_doc[dd]]
            xp = [pp_prim[p] for p in sub if p in pp_prim]; xo = [pp_o[p] for p in sub if p in pp_o]
            if xp and xo:
                d.append(float(np.mean(xo) - np.mean(xp)))
        rep["paired"][s_] = {"other_minus_primary": float(np.mean(d)), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]} if d else None
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / "g2_precision.json").write_text(json.dumps(rep, indent=1))
    for f, r in rep["families"].items():
        print(f"[g2] family {f}: judged {r['n_judged']}, coverage accuracy on human-known units {r['coverage_accuracy_on_known']} (n={r['n_known']}), support rate {r['support_rate']:.3f}", flush=True)
    if rep["kappa_true_omission"]:
        print(f"[g2] family agreement on true omission: kappa {rep['kappa_true_omission']['kappa']:.3f} (n={rep['kappa_true_omission']['n']})", flush=True)
    for s_ in sorted(systems, key=lambda s_: -(rep["systems"][s_][prim_f]["true_omission_rate"] or 0)):
        r = rep["systems"][s_]; pr = rep["paired"].get(s_)
        print(f"[g2] {s_:34s} true-omission rate " + " ".join(f"{f} {r[f]['true_omission_rate']:.3f}" for f in fam) + f" | unknown resolved {r[prim_f]['unknown_resolved_rate']} | n {r[prim_f]['n_pairs']}" + (f" | this - primary {pr['other_minus_primary']:+.3f} [{pr['ci'][0]:+.3f},{pr['ci'][1]:+.3f}]" if pr else " | primary"), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--max-rank", type=int, default=0,
        help="only emit tasks for the first N emitted units of each system (0 = all); the dump is rank-ordered")
    b.add_argument("--emitted", required=True); b.add_argument("--acu", required=True); b.add_argument("--systems", nargs="+", required=True); b.add_argument("--out", required=True)
    r = sub.add_parser("run"); r.add_argument("--tasks", required=True); r.add_argument("--out", required=True); r.add_argument("--model-snapshot", required=True); r.add_argument("--question", choices=list(PROMPTS), required=True)
    r.add_argument("--shard", default="0/1"); r.add_argument("--batch-size", type=int, default=8); r.add_argument("--device", default="cuda")
    r.add_argument("--prompt-set", choices=["plain", "anchored"], default="plain", help="anchored = the T06 boundary-pinning prompts")
    z = sub.add_parser("analyze"); z.add_argument("--tasks", required=True); z.add_argument("--families", nargs="+", required=True, help="name=glob of judgment files"); z.add_argument("--primary", required=True)
    z.add_argument("--primary-family", default="gemma"); z.add_argument("--n-boot", type=int, default=300); z.add_argument("--out", required=True)
    a = ap.parse_args(); {"build": build, "run": run, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
