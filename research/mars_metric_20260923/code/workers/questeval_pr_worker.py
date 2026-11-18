#!/usr/bin/env python3
"""QuestEval (Scialom et al., 2021), source-based, in ~/.venv-questeval (questeval 0.2.4). Writes the metric's own
score (mean of its precision and recall directions) and the two directions separately.
precision: questions generated from the candidate, answered on the source; recall: questions generated from the
source, answered on the candidate."""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-dir", default="")  # unused: questeval 0.2.4 keeps its own cache
    ap.add_argument("--weighter", type=int, default=0)
    a = ap.parse_args()
    from questeval.questeval_metric import QuestEval

    store = []

    class Split(QuestEval):
        def _calculate_score_from_logs(self, hyp_log, compared_logs):
            ps, rs, fs = [], [], []
            for c in compared_logs:
                if c["text"] == "" or hyp_log["text"] == "":
                    p = r = 0.0
                else:
                    p = self._base_score(hyp_log, c); r = self._base_score(c, hyp_log)
                ps.append(p); rs.append(r); fs.append(float(np.average([p, r])))
            store.append((float(np.mean(ps)), float(np.mean(rs))))
            return self.reduction_multi_refs(fs)

    qe = Split(task="summarization", do_weighter=bool(a.weighter))  # logs cached by text hash in the package dir
    rows = [json.loads(l) for l in open(a.inp) if l.strip()]
    res = qe.corpus_questeval(hypothesis=[r["summary"] for r in rows], sources=[r["doc"] for r in rows], batch_size=64)
    assert len(store) == len(rows), (len(store), len(rows))
    with open(a.out, "w") as fh:
        for r, v, (p, rc) in zip(rows, res["ex_level_scores"], store):
            fh.write(json.dumps({"id": r["id"], "questeval": float(v), "questeval_P": p, "questeval_R": rc}) + "\n")


if __name__ == "__main__":
    main()
