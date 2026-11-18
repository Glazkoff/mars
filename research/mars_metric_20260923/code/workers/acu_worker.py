#!/usr/bin/env python3
"""AutoACU (Liu et al., 2023) in ~/.venv-acu: A3CU (end-to-end ACU recall/F) and A2CU (ACUs generated from the
reference, each checked against the candidate), reference-based, maximum over references. Input rows carry
`refs`; rows without a reference are skipped."""
import argparse
import json


def one_line(t: str) -> str:
    """autoacu passes texts through line-based files; str.split() also breaks on \\r, \\x0b-\\x1f and \\u2028,
    which the file reader treats as line ends, so collapsing all whitespace keeps one text per line."""
    return " ".join(t.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", choices=["a3cu", "a2cu"], required=True)
    ap.add_argument("--acu-cache", default="")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.inp) if l.strip()]
    refs, cands, owner = [], [], []
    for i, r in enumerate(rows):
        for ref in r.get("refs") or []:
            refs.append(ref); cands.append(r["summary"]); owner.append(i)
    best = [dict() for _ in rows]
    if a.which == "a3cu":
        from autoacu import A3CU
        m = A3CU(device=0)
        rec, prec, f1 = m.score(references=refs, candidates=cands, batch_size=32, output_path=None)
        for o, rr, ff in zip(owner, rec, f1):
            b = best[o]
            b["a3cu_R"] = max(b.get("a3cu_R", -1.0), float(rr)); b["a3cu_F"] = max(b.get("a3cu_F", -1.0), float(ff))
    else:
        import os
        import tempfile

        from autoacu import A2CU
        m = A2CU(device=0)
        # autoacu's A2CU regenerates the ACUs of a reference for every pair; SummEval repeats each of its 1,100
        # references 16 times, so the ACUs of each unique reference are generated once (acu_generation) and every
        # (reference, candidate) pair is then matched against them (acu_matching), the package's own two stages.
        uniq = sorted(set(refs))
        acus = {}
        if a.acu_cache and os.path.exists(a.acu_cache):
            acus = json.load(open(a.acu_cache))
        todo = [u for u in uniq if u not in acus]
        with tempfile.TemporaryDirectory() as tmp:
            if todo:
                rp, gp = os.path.join(tmp, "uniq_refs.txt"), os.path.join(tmp, "uniq_acus.jsonl")
                with open(rp, "w") as f:
                    for u in todo:
                        print(one_line(u), file=f)
                gen = m.acu_generation(rp, gp, 16)
                assert len(gen) == len(todo), (len(gen), len(todo))
                for u, g in zip(todo, gen):
                    acus[u] = g
                if a.acu_cache:
                    json.dump(acus, open(a.acu_cache, "w"))
            rp, cp, ap_, op = (os.path.join(tmp, x) for x in ("ref.txt", "cand.txt", "acu.jsonl", "scores.txt"))
            with open(rp, "w") as fr, open(cp, "w") as fc, open(ap_, "w") as fa:
                for r_, c_ in zip(refs, cands):
                    print(one_line(r_), file=fr); print(one_line(c_), file=fc)
                    print(json.dumps({"acus": acus[r_]}), file=fa)
            m.acu_matching(rp, cp, ap_, op, tmp, 64)
            recalls = [float(line.strip()) for line in open(op)]
        assert len(recalls) == len(owner), (len(recalls), len(owner))
        for o, rr in zip(owner, recalls):
            best[o]["a2cu_R"] = max(best[o].get("a2cu_R", -1.0), float(rr))
    with open(a.out, "w") as fh:
        for r, b in zip(rows, best):
            fh.write(json.dumps({"id": r["id"], **b}) + "\n")


if __name__ == "__main__":
    main()
