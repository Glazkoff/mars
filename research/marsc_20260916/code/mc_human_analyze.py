#!/usr/bin/env python3
"""MARS-C G6 analysis -- join the annotators' export (labelapp export.jsonl) with the blinded sample key.

Per system: confirmed-omission rate of emitted facts (Q1 = A supported AND Q2 = B not conveyed), by evaluator status
(unknown / hit / false alert), per annotator and by agreement; inter-annotator agreement (raw, kappa) on Q1, Q2 and the
confirmed-omission verdict. Also the evaluator's own status against the human verdict on the hit / false-alert items.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def kappa(x, y):
    if not x:
        return None
    po = sum(a == b for a, b in zip(x, y)) / len(x); cats = set(x) | set(y)
    pe = sum((x.count(c) / len(x)) * (y.count(c) / len(y)) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else None


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--export", required=True); ap.add_argument("--key", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--e2-json", default="", help="evaluator json whose k=10 hit/false-alert/unknown shares weight the strata to the natural emitted mix")
    a = ap.parse_args()
    key = {json.loads(l)["item_id"]: json.loads(l) for l in open(a.key, encoding="utf-8")}
    lab = defaultdict(dict)
    for l in open(a.export, encoding="utf-8"):
        r = json.loads(l); lab[r["annotator"]][r["item_id"]] = r
    names = sorted(lab); rep = {"annotators": {n: len(lab[n]) for n in names}, "systems": {}, "agreement": None, "evaluator_status_vs_human": {}}
    verdict = lambda r: r["q1"] == "A" and r["q2"] == "B"
    for n in names:
        per = defaultdict(lambda: defaultdict(list))
        for iid, r in lab[n].items():
            k = key[iid]; per[k["system"]]["all"].append(verdict(r)); per[k["system"]][k["status"]].append(verdict(r))
        for s_, d in per.items():
            rep["systems"].setdefault(s_, {})[n] = {st: {"n": len(v), "confirmed_omission_rate": sum(v) / len(v)} for st, v in d.items()}
    if len(names) >= 2:
        x, y = names[0], names[1]; both = sorted(i for i in lab[x] if i in lab[y])
        q1x, q1y = [lab[x][i]["q1"] for i in both], [lab[y][i]["q1"] for i in both]; q2x, q2y = [lab[x][i]["q2"] for i in both], [lab[y][i]["q2"] for i in both]
        vx, vy = [str(verdict(lab[x][i])) for i in both], [str(verdict(lab[y][i])) for i in both]
        rep["agreement"] = {"n": len(both), "q1_raw": sum(p == q for p, q in zip(q1x, q1y)) / max(1, len(both)), "q1_kappa": kappa(q1x, q1y),
                            "q2_raw": sum(p == q for p, q in zip(q2x, q2y)) / max(1, len(both)), "q2_kappa": kappa(q2x, q2y),
                            "verdict_raw": sum(p == q for p, q in zip(vx, vy)) / max(1, len(both)), "verdict_kappa": kappa(vx, vy)}
        both_agree = defaultdict(list)
        for idx, i in enumerate(both):
            if vx[idx] == vy[idx]:
                both_agree[key[i]["system"]].append(vx[idx] == "True")
        rep["systems_both_agree"] = {s_: {"n": len(v), "confirmed_omission_rate": sum(v) / len(v)} for s_, v in both_agree.items()}
    for n in names:
        acc = []
        for iid, r in lab[n].items():
            st = key[iid]["status"]
            if st in ("hit", "false_alert"):
                acc.append((r["q2"] == "B") == (st == "hit"))
        rep["evaluator_status_vs_human"][n] = {"n": len(acc), "agreement_q2_vs_status": sum(acc) / len(acc) if acc else None}
    if a.e2_json:
        e2 = json.load(open(a.e2_json))["k"]["10"]["systems"]; rep["precision_natural_mix"] = {}
        for s_, per_ann in rep["systems"].items():
            if s_ not in e2:
                continue
            w = {"hit": e2[s_]["hit_rate"], "false_alert": e2[s_]["false_alert_rate"], "unknown": e2[s_]["unknown_rate"]}
            rep["precision_natural_mix"][s_] = {"weights": w}
            for n, d in per_ann.items():
                if all(st in d for st in w):
                    rep["precision_natural_mix"][s_][n] = sum(w[st] * d[st]["confirmed_omission_rate"] for st in w)
            if len(names) >= 2:
                bb = defaultdict(list)
                for idx, i in enumerate(both):
                    if key[i]["system"] == s_ and vx[idx] == vy[idx]:
                        bb[key[i]["status"]].append(vx[idx] == "True")
                if all(st in bb for st in w):
                    rep["precision_natural_mix"][s_]["both_agree"] = sum(w[st] * (sum(bb[st]) / len(bb[st])) for st in w)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / "human_confirmation.json").write_text(json.dumps(rep, indent=1))
    for s_, d in rep["systems"].items():
        print(f"[g6] {s_:28s} " + " | ".join(f"{n}: all {d[n]['all']['confirmed_omission_rate']:.2f} (n={d[n]['all']['n']})" + "".join(f" {st} {d[n][st]['confirmed_omission_rate']:.2f}" for st in ("unknown", "hit", "false_alert") if st in d[n]) for n in d))
    if rep.get("agreement"):
        g = rep["agreement"]; print(f"[g6] agreement n={g['n']}: Q1 raw {g['q1_raw']:.2f} k {g['q1_kappa']}; Q2 raw {g['q2_raw']:.2f} k {g['q2_kappa']:.2f}; verdict raw {g['verdict_raw']:.2f} k {g['verdict_kappa']:.2f}")
    for s_, d in rep.get("precision_natural_mix", {}).items():
        print(f"[g6] natural-mix precision {s_:28s} " + " ".join(f"{k} {v:.3f}" for k, v in d.items() if k != "weights") + f" | weights {dict((k, round(v, 3)) for k, v in d['weights'].items())}")
    for n, d in rep["evaluator_status_vs_human"].items():
        print(f"[g6] evaluator status vs {n}: agreement {d['agreement_q2_vs_status']:.2f} (n={d['n']})")


if __name__ == "__main__":
    main()
