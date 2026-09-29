#!/usr/bin/env python3
"""MARS-C B7 -- recompute QAPyramid's overlap with our RoSE pool, and resolve two unverified pool claims.

Registered in research/marsc_strengthen_20260918/PREREG.md, Wave B / B7.

Why this file exists
--------------------
The G5 census recorded "QAPyramid: 0 text records, 0 overlap with our sources". That is a MEASUREMENT ARTEFACT,
not a fact. `mc_pool_feasibility.py`'s `find_text()` requires a long string under a key whose name contains a
text-like token; QAPyramid's `raw_data/*.json` are dict-of-dict keyed by DOCUMENT ID, so `iter_records` yielded
one record whose keys are 40-hex ids and `find_text` returned None for it. The overlap was therefore never
computed. QAPyramid is CNN/DM-based and `rose_cnndm` is in the MARS pool, so overlap is plausibly large.

Nothing in `mc_pool_feasibility.py` is modified; the corrected extraction lives here.

What is computed
----------------
1. DOCUMENT-ID JOIN (decisive, local, no network). RoSE pair ids are `rose:<domain>:<example_id>:<system>` and for
   `cnndm` the `example_id` IS the CNN/DM story id. QAPyramid keys its raw annotations `<cnndm_id>_<sent_idx>`.
   The join is exact and is reported per split role (train / validation / test / calib / tune), because a pool
   that overlaps our TRAIN split is contaminated for a transfer arm while one that overlaps only our sealed TEST
   split is an independent human label source ON documents no model of ours trained on.
2. TEXT OVERLAP as originally registered: sha1 of the first 200 characters (exact), and 8-gram shingle Jaccard
   >= --jaccard-floor (near duplicate), against our source pool. Requires the HF dataset `shiyue/QAPyramid`,
   which is where the source documents live; the repo's `raw_data/` carries only summaries and QA annotations.
3. The 10-system human coverage subset: how many of QAPyramid's 500 examples carry `QA_presence`, and how those
   fall across our split roles.
4. Resolution of two UNVERIFIED claims quoted from a proposal: whether the HF dataset `ComposoAI/OmissionBench`
   exists, and whether arXiv id 2608.31016 resolves. Both are reported as resolved / not resolved with the raw
   status, and no downstream claim may cite either until this file says it resolved.

Every network step is individually guarded: a failure is recorded in the report and the job still writes its
artefact, because step 1 alone already answers the registered question.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "mars2_gates_20260916" / "code"))
import common  # noqa: E402

HEXID = re.compile(r"^[0-9a-f]{40}$")
TEXT_KEYS = ("source", "document", "article", "transcript", "input", "text", "dialogue", "doc", "story", "body")


def sha1_200(t: str) -> str:
    return hashlib.sha1(t[:200].encode()).hexdigest()


def shingles(text: str, n: int = 8) -> set:
    t = common.tokens(text)
    return {" ".join(t[i:i + n]) for i in range(max(0, len(t) - n + 1))}


def deep_texts(obj, min_len: int, out: list, keypath: str = "") -> None:
    """Collect (keypath, string) for every long string anywhere in a nested record.

    `mc_pool_feasibility.find_text` looks only at top-level keys with text-like names. QAPyramid's shape defeats
    that. This walks the whole structure and lets the caller decide which key path is the source document.
    """
    if isinstance(obj, str):
        if len(obj) >= min_len:
            out.append((keypath, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            deep_texts(v, min_len, out, f"{keypath}.{k}" if keypath else str(k))
    elif isinstance(obj, (list, tuple)):
        for v in obj[:200]:
            deep_texts(v, min_len, out, f"{keypath}[]")


def expand_records(d):
    """Turn one parsed .json object into the list of RECORDS it actually contains.

    This is the same failure mode the whole block exists to repair, one level up: a pool ships a dict whose
    values are the records (QAPyramid keys by document id) or a dict carrying a `records` / `scenarios` list
    (OmissionBench's fact sheets). Treating the file as ONE record makes every document after the first
    invisible, which is precisely how "0 text records, 0 overlap" was produced in the first place.
    """
    if isinstance(d, list):
        return [x for x in d if isinstance(x, dict)]
    if not isinstance(d, dict):
        return []
    lists = [v for v in d.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
    if lists:
        return [x for v in lists for x in v]
    vals = [v for v in d.values() if isinstance(v, dict)]
    return vals if len(vals) >= max(2, len(d) // 2) else [d]


def http_json(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "mars-c-b7/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8", "replace"))


def http_text(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "mars-c-b7/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


# --------------------------------------------------------------------------- our pool
def load_our_pool(specs: list[str]) -> dict:
    """`role=path` jsonl. Rows carry `source`; ids come from `summaries[].pair_id` or a top-level `pair_id`."""
    ids_by_role, hash_by_role = defaultdict(set), defaultdict(set)
    texts = {}
    for spec in specs:
        role, path = spec.split("=", 1)
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"[b7] missing our-source file {p}")
        for r in common.load_jsonl(p):
            src = r.get("source") or ""
            if not src:
                continue
            h = sha1_200(src)
            hash_by_role[role].add(h)
            texts.setdefault(h, src)
            pids = [s.get("pair_id") for s in r.get("summaries", []) if isinstance(s, dict)]
            if not pids and r.get("pair_id"):
                pids = [r["pair_id"]]
            for pid in pids:
                if not pid:
                    continue
                parts = str(pid).split(":")
                if len(parts) >= 3 and parts[0] == "rose":
                    ids_by_role[role].add((parts[1], parts[2]))
    return {"ids_by_role": {k: v for k, v in ids_by_role.items()},
            "hash_by_role": {k: v for k, v in hash_by_role.items()}, "texts": texts}


# --------------------------------------------------------------------------- QAPyramid, locally
def qapyramid_repo(out: Path, cached: str) -> tuple[Path | None, str]:
    c = Path(cached) if cached else None
    if c and c.exists():
        return c, "cached (read-only)"
    d = out / "QAPyramid"
    if d.exists():
        return d, "cached in out"
    r = subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ZhangShiyue/QAPyramid", str(d)],
                       capture_output=True, text=True, timeout=900)
    return (d, "cloned") if r.returncode == 0 else (None, f"clone failed: {r.stderr[-300:]}")


def qapyramid_ids(repo: Path) -> dict:
    gen_p, pres_p = repo / "raw_data" / "QA_generation.json", repo / "raw_data" / "QA_presence.json"
    gen = json.load(open(gen_p, encoding="utf-8")) if gen_p.exists() else {}
    pres = json.load(open(pres_p, encoding="utf-8")) if pres_p.exists() else {}
    gen_ids = {k.rsplit("_", 1)[0] for k in gen}
    pres_ids, systems = set(), sorted(pres)
    for s in pres.values():
        if isinstance(s, dict):
            pres_ids |= set(s)
    return {"generation_docs": sorted(gen_ids), "presence_docs": sorted(pres_ids),
            "presence_systems": systems, "n_generation_rows": len(gen),
            "id_shape_ok": all(HEXID.match(x) for x in list(gen_ids)[:50])}


# --------------------------------------------------------------------------- QAPyramid, from the hub
def hub_texts(repo_id: str, out: Path, max_files: int, include: str = "") -> tuple[dict, dict]:
    """Download a hub dataset's data files and recover its source documents, whatever the record shape.

    `include` is an optional regex on the repo-relative file path; it exists because a pool such as
    OmissionBench ships hundreds of per-run judgement manifests alongside the handful of files that actually
    carry the source documents, and `max_files` would otherwise be spent on the manifests.
    """
    info = {"repo_id": repo_id, "status": None, "files": [], "records": 0, "texts": 0, "text_keypath": None}
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except Exception as e:  # noqa: BLE001
        info["status"] = f"huggingface_hub unavailable: {type(e).__name__}: {e}"
        return info, {}
    try:
        files = [f for f in list_repo_files(repo_id, repo_type="dataset")
                 if f.endswith((".json", ".jsonl", ".parquet", ".csv"))]
        info["n_data_files_total"] = len(files)
        if include:
            files = [f for f in files if re.search(include, f)]
        info["include_regex"] = include or None
    except Exception as e:  # noqa: BLE001
        info["status"] = f"list_repo_files failed: {type(e).__name__}: {e}"
        return info, {}
    info["files_listed"] = files[:50]
    recs = []
    for f in files[:max_files]:
        try:
            local = hf_hub_download(repo_id, f, repo_type="dataset", local_dir=str(out / "hf" / repo_id.replace("/", "__")))
        except Exception as e:  # noqa: BLE001
            info["files"].append({"file": f, "error": f"{type(e).__name__}: {e}"}); continue
        got = []
        try:
            if f.endswith(".parquet"):
                import pyarrow.parquet as pq
                got = pq.read_table(local).to_pylist()
            elif f.endswith(".jsonl"):
                got = [json.loads(l) for l in open(local, encoding="utf-8") if l.strip()]
            elif f.endswith(".json"):
                got = expand_records(json.load(open(local, encoding="utf-8")))
            elif f.endswith(".csv"):
                import csv as _csv
                with open(local, encoding="utf-8", newline="") as fh:
                    got = list(_csv.DictReader(fh))
        except Exception as e:  # noqa: BLE001
            info["files"].append({"file": f, "error": f"parse {type(e).__name__}: {e}"}); continue
        info["files"].append({"file": f, "records": len(got)})
        recs.extend(got if isinstance(got, list) else [])
    info["records"] = len(recs)
    if not recs:
        info["status"] = info.get("status") or "no records parsed"
        return info, {}
    # choose the key path that most often holds a long string with a text-like name; fall back to the longest
    votes = Counter()
    for r in recs[:400]:
        found = []
        deep_texts(r, 200, found)
        named = [kp for kp, _ in found if any(t in kp.lower() for t in TEXT_KEYS)]
        votes[named[0] if named else (max(found, key=lambda x: len(x[1]))[0] if found else "")] += 1
    keypath = votes.most_common(1)[0][0] if votes else ""
    info["text_keypath"] = keypath
    info["keypath_votes"] = dict(votes.most_common(8))
    texts, ids = {}, set()
    for r in recs:
        found = []
        deep_texts(r, 200, found)
        pick = next((v for kp, v in found if kp == keypath), None)
        if pick is None and found:
            pick = max(found, key=lambda x: len(x[1]))[1]
        if pick:
            texts[sha1_200(pick)] = pick
        if isinstance(r, dict):
            for v in r.values():
                if isinstance(v, str) and HEXID.match(v):
                    ids.add(v)
    info["texts"] = len(texts)
    info["ids_found"] = sorted(ids)
    info["status"] = "ok"
    return info, texts


def jaccard_overlap(theirs: dict, ours_texts: dict, floor: float) -> dict:
    """Best 8-gram shingle Jaccard of each of their documents against our pool, via an inverted index."""
    our_sh = {h: shingles(t) for h, t in ours_texts.items()}
    inv = defaultdict(list)
    for h, sh in our_sh.items():
        for g in sh:
            inv[g].append(h)
    best, hits = [], 0
    for h, t in theirs.items():
        sh = shingles(t)
        if not sh:
            best.append(0.0); continue
        c = Counter()
        for g in sh:
            for oh in inv.get(g, ()):
                c[oh] += 1
        b = 0.0
        for oh, inter in c.most_common(20):
            j = inter / max(1, len(sh) + len(our_sh[oh]) - inter)
            b = max(b, j)
        best.append(b)
        if b >= floor:
            hits += 1
    best_sorted = sorted(best, reverse=True)
    return {"n_their_docs": len(theirs), "n_our_docs": len(our_sh), "jaccard_floor": floor,
            "n_near_duplicate": hits,
            "share_near_duplicate": hits / max(1, len(theirs)),
            "best_jaccard_top10": best_sorted[:10],
            "best_jaccard_median": float(sorted(best)[len(best) // 2]) if best else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--our-sources", nargs="+", required=True, help="role=path jsonl with a `source` field")
    ap.add_argument("--qapyramid-cached", default="", help="existing read-only clone (g5/QAPyramid)")
    ap.add_argument("--hf-qapyramid", default="shiyue/QAPyramid")
    ap.add_argument("--hf-probe", nargs="*", default=["ComposoAI/OmissionBench"])
    ap.add_argument("--hf-overlap", nargs="*", default=["ComposoAI/OmissionBench"],
                    help="further hub datasets put through the SAME overlap measurement as QAPyramid")
    ap.add_argument("--hf-overlap-include", default=r"fact_sheets/|(^|/)data/|\.parquet$",
                    help="regex on repo-relative paths, so a pool's per-run manifests do not crowd out its data")
    ap.add_argument("--arxiv-probe", nargs="*", default=["2608.31016"])
    ap.add_argument("--jaccard-floor", type=float, default=0.5)
    ap.add_argument("--max-hf-files", type=int, default=12)
    ap.add_argument("--no-network", action="store_true", help="step 1 only (id join); for a dry check")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    ours = load_our_pool(a.our_sources)
    roles = sorted(ours["ids_by_role"])
    all_ids = set().union(*ours["ids_by_role"].values()) if roles else set()
    print(f"[b7] our pool: {sum(len(v) for v in ours['hash_by_role'].values())} source rows over roles {roles}; "
          f"{len(all_ids)} rose (domain, example_id) ids; {len(ours['texts'])} unique source texts", flush=True)

    rep = {"registered": "research/marsc_strengthen_20260918/PREREG.md Wave B / B7",
           "supersedes": "the G5 census line 'QAPyramid: 0 text records, 0 overlap' -- a find_text() artefact, "
                         "not a measurement",
           "our_pool": {"roles": {r: {"n_sources": len(ours["hash_by_role"][r]),
                                      "n_rose_ids": len(ours["ids_by_role"].get(r, ()))} for r in roles},
                        "n_unique_source_texts": len(ours["texts"])},
           "qapyramid": {}, "probes": {}}

    # ---- step 1: the decisive local document-id join
    repo, how = qapyramid_repo(out, a.qapyramid_cached)
    rep["qapyramid"]["repo"] = {"path": str(repo) if repo else None, "how": how}
    if repo is None:
        print(f"[b7] QAPyramid repo unavailable: {how}", flush=True)
    else:
        q = qapyramid_ids(repo)
        cnndm_ours = {e for d, e in all_ids if d == "cnndm"}
        gen, pres = set(q["generation_docs"]), set(q["presence_docs"])
        per_role = {}
        for r in roles:
            ids_r = {e for d, e in ours["ids_by_role"][r] if d == "cnndm"}
            per_role[r] = {"our_cnndm_docs": len(ids_r),
                           "in_qapyramid_generation": len(ids_r & gen),
                           "in_qapyramid_presence_50": len(ids_r & pres)}
        rep["qapyramid"]["document_id_join"] = {
            "method": "RoSE pair_id `rose:cnndm:<example_id>:<system>` example_id == CNN/DM story id == "
                      "QAPyramid raw_data key prefix `<story_id>_<sent_idx>`",
            "qapyramid_generation_docs": len(gen), "qapyramid_presence_docs": len(pres),
            "qapyramid_presence_systems": q["presence_systems"],
            "our_cnndm_docs": len(cnndm_ours),
            "exact_id_overlap_all_roles": len(cnndm_ours & gen),
            "share_of_qapyramid_covered_by_our_pool": len(cnndm_ours & gen) / max(1, len(gen)),
            "share_of_our_cnndm_covered_by_qapyramid": len(cnndm_ours & gen) / max(1, len(cnndm_ours)),
            "per_role": per_role, "id_shape_ok": q["id_shape_ok"]}
        j = rep["qapyramid"]["document_id_join"]
        print(f"[b7] ID JOIN: QAPyramid {len(gen)} docs, ours(cnndm) {len(cnndm_ours)}, exact overlap "
              f"{j['exact_id_overlap_all_roles']} "
              f"({j['share_of_qapyramid_covered_by_our_pool']:.3f} of QAPyramid)", flush=True)
        for r, v in per_role.items():
            print(f"[b7]   role {r:12s} ours {v['our_cnndm_docs']:4d} | in QAPyramid {v['in_qapyramid_generation']:4d} "
                  f"| in the 10-system presence subset {v['in_qapyramid_presence_50']:3d}", flush=True)

    # ---- step 2: the text-level overlap as registered (needs the hub: sources are not in the repo)
    if a.no_network:
        rep["qapyramid"]["text_overlap"] = {"status": "skipped (--no-network)"}
    else:
        try:
            info, texts = hub_texts(a.hf_qapyramid, out, a.max_hf_files)
            rep["qapyramid"]["hub"] = info
            if texts:
                exact = sum(1 for h in texts if any(h in hs for hs in ours["hash_by_role"].values()))
                ov = jaccard_overlap(texts, ours["texts"], a.jaccard_floor)
                ov["exact_sha1_first200_overlap"] = exact
                ov["share_exact"] = exact / max(1, len(texts))
                rep["qapyramid"]["text_overlap"] = ov
                print(f"[b7] TEXT: {len(texts)} QAPyramid source texts; exact sha1(first200) matches {exact}; "
                      f"near-duplicate (Jaccard>={a.jaccard_floor}) {ov['n_near_duplicate']}", flush=True)
            else:
                rep["qapyramid"]["text_overlap"] = {"status": "no source texts recovered from the hub; "
                                                              "the id join above stands on its own"}
        except Exception as e:  # noqa: BLE001
            rep["qapyramid"]["text_overlap"] = {"status": f"failed: {type(e).__name__}: {e}"}
            print(f"[b7] text overlap failed: {type(e).__name__}: {e}", flush=True)

    # ---- step 2b: the SAME overlap measurement on every further pool named on the command line.
    # Domain reasoning ("OmissionBench is clinical, RoSE is news, so overlap must be zero") is exactly the
    # reasoning that produced the QAPyramid artefact this block exists to repair, so it is measured, not assumed.
    rep["other_pools"] = {}
    for repo_id in (a.hf_overlap or []):
        ent = {}
        if a.no_network:
            ent["status"] = "skipped (--no-network)"
        else:
            try:
                info, texts = hub_texts(repo_id, out, a.max_hf_files, a.hf_overlap_include)
                ent["hub"] = info
                if texts:
                    exact = sum(1 for h in texts if any(h in hs for hs in ours["hash_by_role"].values()))
                    ov = jaccard_overlap(texts, ours["texts"], a.jaccard_floor)
                    ov["exact_sha1_first200_overlap"] = exact
                    ov["share_exact"] = exact / max(1, len(texts))
                    ent["text_overlap"] = ov
                    print(f"[b7] {repo_id}: {len(texts)} source texts; exact matches {exact}; "
                          f"near-duplicate {ov['n_near_duplicate']}; best-Jaccard median "
                          f"{ov['best_jaccard_median']:.4f}", flush=True)
                else:
                    ent["text_overlap"] = {"status": "no source texts recovered -- overlap NOT measured, "
                                                     "and must not be reported as zero"}
                    print(f"[b7] {repo_id}: no source texts recovered; overlap NOT measured", flush=True)
            except Exception as e:  # noqa: BLE001
                ent["status"] = f"failed: {type(e).__name__}: {e}"
                print(f"[b7] {repo_id} overlap failed: {type(e).__name__}: {e}", flush=True)
        rep["other_pools"][repo_id] = ent

    # ---- step 3: resolve the unverified claims
    for repo_id in (a.hf_probe or []):
        ent = {"claim_source": "quoted from a proposal, UNVERIFIED before this job"}
        if a.no_network:
            ent["status"] = "skipped (--no-network)"
        else:
            try:
                st, d = http_json(f"https://huggingface.co/api/datasets/{repo_id}")
                ent.update({"resolves": True, "http_status": st,
                            "downloads": d.get("downloads"), "lastModified": d.get("lastModified"),
                            "siblings": [s.get("rfilename") for s in (d.get("siblings") or [])][:40]})
            except urllib.error.HTTPError as e:
                ent.update({"resolves": False, "http_status": e.code, "reason": e.reason})
            except Exception as e:  # noqa: BLE001
                ent.update({"resolves": None, "reason": f"{type(e).__name__}: {e}"})
        rep["probes"][f"hf_dataset:{repo_id}"] = ent
        print(f"[b7] probe hf dataset {repo_id}: resolves={ent.get('resolves')} ({ent.get('http_status', ent.get('reason'))})", flush=True)
    for aid in (a.arxiv_probe or []):
        ent = {"claim_source": "quoted from a proposal, UNVERIFIED before this job"}
        if a.no_network:
            ent["status"] = "skipped (--no-network)"
        else:
            try:
                st, body = http_text(f"http://export.arxiv.org/api/query?id_list={aid}&max_results=1")
                n = body.count("<entry>")
                title = re.search(r"<entry>.*?<title>(.*?)</title>", body, re.S)
                ent.update({"resolves": n > 0, "http_status": st, "n_entries": n,
                            "title": (title.group(1).strip()[:300] if title else None)})
            except Exception as e:  # noqa: BLE001
                ent.update({"resolves": None, "reason": f"{type(e).__name__}: {e}"})
        rep["probes"][f"arxiv:{aid}"] = ent
        print(f"[b7] probe arxiv {aid}: resolves={ent.get('resolves')} title={str(ent.get('title'))[:80]}", flush=True)

    # ---- verdict, written from the measured numbers
    j = rep["qapyramid"].get("document_id_join") or {}
    share = j.get("share_of_qapyramid_covered_by_our_pool")
    if share is None:
        verdict = "UNRESOLVED -- the QAPyramid repository was not available to this job"
    elif share >= 0.5:
        tr = (j.get("per_role") or {}).get("train", {}).get("in_qapyramid_generation", 0)
        te = (j.get("per_role") or {}).get("test", {}).get("in_qapyramid_generation", 0)
        verdict = (f"RULED OUT as an INDEPENDENT pool: {share:.3f} of QAPyramid's documents are the same "
                   f"documents as our RoSE CNN/DM pool ({tr} of them in our TRAIN role). The recorded 'zero "
                   f"overlap' was a find_text() artefact. QAPyramid is a RE-ANNOTATION of documents we already "
                   f"use, not an untouched pool, and no transfer arm may be planned on it. It remains usable as "
                   f"an INDEPENDENT HUMAN LABEL SOURCE on the {te} of its documents that fall in our sealed TEST "
                   f"role, where no model of ours was trained; only the presence subset carries the 10-system "
                   f"human coverage evaluation.")
    else:
        verdict = (f"CANDIDATE: only {share:.3f} of QAPyramid's documents overlap our pool; the non-overlapping "
                   f"remainder is a usable labelled pool subject to the licence and the presence-subset size.")
    rep["verdict"] = verdict
    print(f"[b7] VERDICT: {verdict}", flush=True)
    rep["other_pool_verdicts"] = {}
    for repo_id, ent in rep.get("other_pools", {}).items():
        ov = (ent or {}).get("text_overlap") or {}
        pr = rep["probes"].get(f"hf_dataset:{repo_id}", {})
        if "share_exact" not in ov:
            v = (f"NOT MEASURED ({ov.get('status') or ent.get('status')}). The dataset "
                 f"{'resolves' if pr.get('resolves') else 'does not resolve'} on the hub. Overlap with our pool "
                 f"is UNKNOWN and may not be reported as zero.")
        elif ov["share_exact"] == 0 and ov["n_near_duplicate"] == 0:
            v = (f"DISJOINT from our pool by measurement: 0 of {ov['n_their_docs']} of its documents match ours "
                 f"exactly or as near-duplicates (best-Jaccard median {ov['best_jaccard_median']:.4f}). "
                 f"Usable as an independent pool subject to licence, domain fit and label shape.")
        else:
            v = (f"OVERLAPS our pool: {ov['exact_sha1_first200_overlap']} exact and {ov['n_near_duplicate']} "
                 f"near-duplicate of {ov['n_their_docs']} documents.")
        rep["other_pool_verdicts"][repo_id] = v
        print(f"[b7] VERDICT {repo_id}: {v}", flush=True)

    (out / "b7_pool_overlap.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"[b7] written {out / 'b7_pool_overlap.json'}", flush=True)


if __name__ == "__main__":
    main()
