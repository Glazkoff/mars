#!/usr/bin/env python3
"""QAEval (Deutsch et al., 2021) in ~/.venv-qaeval (qaeval 0.1.0, transformers 3.0.2, torch 2.4 for H200 kernels): questions generated from the
reference, answered on the candidate; exact match and token F1, maximum over references."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--ans", required=True)
    ap.add_argument("--cuda", type=int, default=-1)
    ap.add_argument("--workdir", required=True, help="holds ./bart-large, which the archive's config names")
    a = ap.parse_args()
    import os

    import torch
    os.chdir(a.workdir)
    # The released generation archive was saved before transformers registered BART's `final_logits_bias`
    # buffer (all zeros in bart-large); fill that one missing key with the module's own zeros, nothing else.
    orig = torch.nn.Module.load_state_dict

    def load_state_dict(self, state_dict, strict=True, **kw):
        own = self.state_dict()
        for k, v in own.items():
            if k.endswith("final_logits_bias") and k not in state_dict:
                state_dict[k] = torch.zeros_like(v)
        return orig(self, state_dict, strict, **kw)
    torch.nn.Module.load_state_dict = load_state_dict
    from qaeval import QAEval
    metric = QAEval(generation_model_path=a.gen, answering_model_dir=a.ans, cuda_device=a.cuda)
    rows = [json.loads(l) for l in open(a.inp) if l.strip()]
    keep = [r for r in rows if r.get("refs")]
    res = metric.score_batch([r["summary"] for r in keep], [r["refs"] for r in keep]) if keep else []
    out = {r["id"]: {"qaeval_EM": float(s["qa-eval"]["em"]), "qaeval_F1": float(s["qa-eval"]["f1"])} for r, s in zip(keep, res)}
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps({"id": r["id"], **out.get(r["id"], {})}) + "\n")


if __name__ == "__main__":
    main()
