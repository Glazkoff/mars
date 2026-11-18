#!/usr/bin/env python3
"""Omission confirmation -- blinded annotation web app WITH A REAL ADJUDICATOR ROLE (stdlib: http.server + sqlite3).

This is a fork of `research/marsc_20260916/labelapp/app.py`. The original is deployed and live at
https://label.example.org with the merged 400-item batch in its database; it is NOT edited by this campaign.

Why the fork exists -- the campaign's most important infrastructure finding.
The governing proposal requires "two blinded reviewers AND adjudication". The deployed app has two roles:
annotator and admin. It has no adjudicator. Every analysis script in the project
(`mc_human_analyze.py`, `mc_human_rank.py`) takes `names[0]` / `names[1]` as THE two annotators and silently
ignores a third. So "adjudicated" has meant AGREEMENT-CONDITIONED throughout this project: items the two
annotators labelled differently were DROPPED, not resolved. That is a materially weaker -- and upward-biased --
estimand than the manuscript implies, because dropping disagreements keeps the easy items.

What the adjudicator role is here.
  * The adjudicator's queue is exactly the items on which the two annotators' derived VERDICTS disagree, and
    only once both have answered. Nothing else is shown; there is no way to page to an agreed item.
  * The adjudicator sees the item exactly as an annotator sees it. They are shown neither who said what nor
    that anyone said anything: no names, no counts, no A/B tallies. The page is indistinguishable from an
    annotator page apart from its banner.
  * The adjudicator answers the same questions and their answer is the RESOLVING verdict for that item.
  * The adjudicator cannot be one of the two annotators (asserted at startup against the config).

Storage. The legacy `labels` table (annotator, item_id, q1, q2, comment, ts) is kept byte-compatible so the
existing 400-item database and `mc_human_analyze.py` keep working unchanged. A generic `answers` table carries
studies with more than two questions (the D2 revision review has five). Export emits both, plus `role`.

Access: per-person secret links /a/<token>/ (annotator), /j/<token>/ (adjudicator), /admin/<admin_token>/.
"""
from __future__ import annotations

import hmac
import html
import json
import re
import sqlite3
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
CFG = json.load(open(HERE / "config.json"))
ITEMS = [json.loads(l) for l in open(HERE / CFG.get("items", "human_sample_blind.jsonl"), encoding="utf-8")]
INDEX = {it["item_id"]: i for i, it in enumerate(ITEMS)}
INSTR = (HERE / "INSTRUCTIONS.md").read_text(encoding="utf-8") if (HERE / "INSTRUCTIONS.md").exists() else ""

# --- study definition (config-driven; the defaults reproduce the G6 two-question study exactly) -------------
QUESTIONS = CFG.get("questions") or [
    {"id": "q1", "label": "Q1. Is the fact stated in the source, or does it follow directly from it?",
     "options": [["A", "A — yes, supported by the source"], ["B", "B — no (absent, contradicted, or invented)"]]},
    {"id": "q2", "label": "Q2. Does the summary convey the fact? (paraphrases, pronouns and equivalent names count; "
                          "wording need not match; ignore importance)",
     "options": [["A", "A — conveyed / covered"], ["B", "B — omitted / not conveyed"]]},
]
QIDS = [q["id"] for q in QUESTIONS]
QVALS = {q["id"]: [o[0] for o in q["options"]] for q in QUESTIONS}
# A derived binary verdict; the adjudicator queue is defined by disagreement ON THIS VERDICT, not on raw answers.
POSITIVE = CFG.get("verdict_positive") or {"q1": "A", "q2": "B"}
PANELS = CFG.get("panels") or [["source", "Source"], ["summary", "Summary"],
                               ["fact", "Fact claimed to be missing from the summary"]]
HILITE = CFG.get("highlight_on", "fact")

ANNOTATORS = dict(CFG.get("annotators") or {})
ADJUDICATORS = dict(CFG.get("adjudicators") or {})
_dup = set(ANNOTATORS.values()) & set(ADJUDICATORS.values())
assert not _dup, f"adjudicator must not also be an annotator: {sorted(_dup)}"
assert not (set(ANNOTATORS) & set(ADJUDICATORS)), "a token is assigned to two roles"
ROLE = {**{n: "annotator" for n in ANNOTATORS.values()}, **{n: "adjudicator" for n in ADJUDICATORS.values()}}
PRIMARY = CFG.get("primary_annotators") or sorted(ANNOTATORS.values())[:2]

LOCK = threading.Lock()
DB = sqlite3.connect(HERE / CFG.get("db", "labels.db"), check_same_thread=False)
DB.execute("CREATE TABLE IF NOT EXISTS labels (annotator TEXT, item_id TEXT, q1 TEXT, q2 TEXT, comment TEXT, ts REAL, PRIMARY KEY (annotator, item_id))")
DB.execute("CREATE TABLE IF NOT EXISTS answers (annotator TEXT, item_id TEXT, payload TEXT, comment TEXT, ts REAL, PRIMARY KEY (annotator, item_id))")
DB.commit()
STOP = set("the a an of to in on at for and or is are was were be been by with as that this it its from has have had not but they their them there than then who which what when where will would can could into over after before about also more most some such only".split())

CSS = """
:root{--bg:#f7f7f4;--card:#fff;--ink:#1d1f22;--mut:#6b7078;--line:#dcdfe3;--acc:#245bcc;--ok:#1e8e4a;--warn:#c0392b;--hl:#fff1a8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 Georgia,'Times New Roman',serif}
.wrap{max-width:1100px;margin:0 auto;padding:18px 20px 60px}.top{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap;font-family:system-ui,sans-serif;font-size:14px;color:var(--mut)}
.top a{color:var(--acc);text-decoration:none;margin-left:14px}.bar{height:6px;background:var(--line);border-radius:3px;margin:8px 0 18px;overflow:hidden}.bar i{display:block;height:100%;background:var(--ok)}
.grid{display:grid;grid-template-columns:1.25fr 1fr;gap:18px}@media(max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:16px 18px}.card h3{margin:0 0 8px;font:600 12px/1 system-ui,sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}
.src{max-height:62vh;overflow:auto;white-space:pre-wrap}.fact{font-size:19px;line-height:1.45;padding:12px 14px;background:#f1f5ff;border-left:4px solid var(--acc);border-radius:4px}
mark{background:var(--hl);padding:0 1px}.q{margin:14px 0 6px;font-family:system-ui,sans-serif}.q b{display:block;margin-bottom:6px}
.opt{display:flex;gap:10px;flex-wrap:wrap}.opt label{border:1px solid var(--line);border-radius:6px;padding:8px 12px;cursor:pointer;background:#fff;font-family:system-ui,sans-serif;font-size:15px}
.opt input{margin-right:8px}.opt label:has(input:checked){border-color:var(--acc);background:#eaf0ff}
textarea{width:100%;min-height:54px;border:1px solid var(--line);border-radius:6px;padding:8px;font:14px system-ui,sans-serif}
button{font:600 15px system-ui,sans-serif;background:var(--acc);color:#fff;border:0;border-radius:6px;padding:10px 18px;cursor:pointer}button.sec{background:#fff;color:var(--ink);border:1px solid var(--line)}
.nav{display:flex;gap:10px;align-items:center;margin-top:14px;flex-wrap:wrap}.kbd{font:12px system-ui,sans-serif;color:var(--mut)}kbd{border:1px solid var(--line);border-radius:3px;padding:0 5px;background:#fff}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(56px,1fr));gap:6px;font:12px system-ui,sans-serif}.tiles a{display:block;text-align:center;padding:6px 0;border:1px solid var(--line);border-radius:4px;color:var(--ink);text-decoration:none;background:#fff}
.tiles a.done{background:#e3f6ea;border-color:#9fd6b3}.tiles a.cur{outline:2px solid var(--acc)}.msg{font-family:system-ui,sans-serif;color:var(--ok)}
table{border-collapse:collapse;font:14px system-ui,sans-serif}td,th{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left}
.banner{font-family:system-ui,sans-serif;font-size:14px;background:#fff6e0;border:1px solid #e8d79a;border-radius:6px;padding:10px 14px;margin-bottom:14px}
"""


def page(title: str, body: str, top: str = "") -> bytes:
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body><div class='wrap'><div class='top'><span><b>Omission confirmation</b></span>"
            f"<span>{top}</span></div>{body}</div></body></html>").encode()


def hl(text: str, key: str) -> str:
    words = {w.lower() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]+", key) if len(w) > 2 and w.lower() not in STOP}
    out = []
    for tok in re.split(r"(\s+)", text):
        core = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", tok).lower()
        e = html.escape(tok)
        out.append(f"<mark>{e}</mark>" if core and core in words else e)
    return "".join(out)


# ----------------------------------------------------------------------------------------------- storage
def record(name: str, item_id: str, ans: dict[str, str], comment: str) -> None:
    with LOCK:
        DB.execute("INSERT OR REPLACE INTO answers (annotator, item_id, payload, comment, ts) VALUES (?,?,?,?,?)",
                   (name, item_id, json.dumps(ans, sort_keys=True), comment, time.time()))
        if "q1" in ans and "q2" in ans:  # keep the legacy table byte-compatible with the deployed study
            DB.execute("INSERT OR REPLACE INTO labels (annotator, item_id, q1, q2, comment, ts) VALUES (?,?,?,?,?,?)",
                       (name, item_id, ans["q1"], ans["q2"], comment, time.time()))
        DB.commit()


def labels_of(name: str) -> dict[str, dict]:
    with LOCK:
        rows = DB.execute("SELECT item_id, payload, comment FROM answers WHERE annotator=?", (name,)).fetchall()
        legacy = DB.execute("SELECT item_id, q1, q2, comment FROM labels WHERE annotator=?", (name,)).fetchall()
    out = {r[0]: {**json.loads(r[1]), "comment": r[2]} for r in rows}
    for item_id, q1, q2, comment in legacy:  # rows written by the deployed app before this fork existed
        out.setdefault(item_id, {"q1": q1, "q2": q2, "comment": comment})
    return out


def verdict(ans: dict) -> bool | None:
    """The derived binary verdict. None when any constituent answer is missing."""
    if any(ans.get(k) is None for k in POSITIVE):
        return None
    return all(ans.get(k) == v for k, v in POSITIVE.items())


def disagreement_ids() -> list[str]:
    """Items both primary annotators have answered and on which their verdicts differ. The adjudicator's queue."""
    if len(PRIMARY) < 2:
        return []
    a, b = (labels_of(PRIMARY[0]), labels_of(PRIMARY[1]))
    out = []
    for it in ITEMS:
        i = it["item_id"]
        if i in a and i in b:
            va, vb = verdict(a[i]), verdict(b[i])
            if va is not None and vb is not None and va != vb:
                out.append(i)
    return out


def person_of(token: str, table: dict) -> str | None:
    for t, name in table.items():
        if hmac.compare_digest(t, token):
            return name
    return None


def is_admin(token: str) -> bool:
    return hmac.compare_digest(CFG["admin_token"], token)


# ----------------------------------------------------------------------------------------------- pages
def question_html(cur: dict) -> str:
    parts = []
    for q in QUESTIONS:
        opts = "".join(
            f"<label><input type='radio' name='{q['id']}' value='{html.escape(v)}'"
            f"{' checked' if cur.get(q['id']) == v else ''}{' required' if i == 0 else ''}>{html.escape(t)}</label>"
            for i, (v, t) in enumerate(q["options"]))
        parts.append(f"<div class='q'><b>{html.escape(q['label'])}</b><div class='opt'>{opts}</div></div>")
    return "".join(parts)


def keymap_js() -> str:
    m = {}
    k = 1
    for q in QUESTIONS:
        for v, _ in q["options"]:
            if k <= 9:
                m[str(k)] = [q["id"], v]
                k += 1
    return json.dumps(m)


def item_body(it: dict, cur: dict, action: str, header: str, nav: str, banner: str = "") -> str:
    key = it.get(HILITE, "")
    left = "".join(f"<div class='card'><h3>{html.escape(lab)}</h3><div class='src'>{hl(str(it.get(f, '')), key)}</div></div>"
                   for f, lab in PANELS[:1] if f in it)
    mid = "".join(f"<div class='card' style='margin-bottom:14px'><h3>{html.escape(lab)}</h3>"
                  f"<div>{hl(str(it.get(f, '')), key)}</div></div>" for f, lab in PANELS[1:-1] if f in it)
    f_last, lab_last = PANELS[-1]
    return f"""{banner}{header}
<div class='grid'>
 <div>{left}</div>
 <div>{mid}
  <div class='card'><h3>{html.escape(lab_last)}</h3><div class='fact'>{html.escape(str(it.get(f_last, '')))}</div>
  <form method='post' action='{action}' id='f'>{question_html(cur)}
   <div class='q'><b>Comment (optional)</b><textarea name='comment'>{html.escape(cur.get('comment') or '')}</textarea></div>
   {nav}
  </form></div>
 </div>
</div>
<script>
const M={keymap_js()};
document.addEventListener('keydown',e=>{{if(e.target.tagName==='TEXTAREA')return;
if(M[e.key]){{const el=document.querySelector(`input[name=${{M[e.key][0]}}][value=${{M[e.key][1]}}]`);if(el)el.checked=true;e.preventDefault();}}
else if(e.key==='Enter'){{document.getElementById('f').requestSubmit();}}}});
</script>"""


def next_unlabeled(done: dict, start: int) -> int:
    n = len(ITEMS)
    for k in range(n):
        i = (start + k) % n
        if ITEMS[i]["item_id"] not in done:
            return i
    return -1


def item_page(token: str, name: str, i: int, saved: bool) -> bytes:
    it = ITEMS[i]
    done = labels_of(name)
    cur = done.get(it["item_id"], {})
    n = len(ITEMS)
    base = f"/a/{token}"
    header = (f"<div class='bar'><i style='width:{100 * len(done) / n:.1f}%'></i></div>"
              f"<div class='top' style='margin-bottom:10px'><span>Item <b>{i + 1}</b> of {n} &middot; {it['item_id']} "
              f"&middot; labelled {len(done)}/{n}{' &middot; <span class=msg>saved</span>' if saved else ''}</span>"
              f"<span><a href='{base}/list'>all items</a><a href='{base}/instructions'>instructions</a></span></div>")
    nav = (f"<div class='nav'><button type='submit'>Save &amp; next</button>"
           f"<a href='{base}/item/{max(0, i - 1)}'><button type='button' class='sec'>&larr; previous</button></a>"
           f"<a href='{base}/item/{min(n - 1, i + 1)}'><button type='button' class='sec'>skip &rarr;</button></a>"
           f"<span class='kbd'>keys: <kbd>1</kbd>..<kbd>{min(9, 2 * len(QUESTIONS))}</kbd> answers &nbsp; <kbd>Enter</kbd> save</span></div>")
    return page(f"Item {i + 1}", item_body(it, cur, f"{base}/item/{i}", header, nav), html.escape(name))


ADJ_BANNER = ("<div class='banner'><b>Adjudication.</b> These are the items the two independent annotators "
              "resolved differently. You are not shown who said what, or what either of them said — judge the "
              "item on its own merits, exactly as they did. Your answer is the resolving verdict for this item.</div>")


def adj_page(token: str, name: str, pos: int, saved: bool) -> bytes:
    queue = disagreement_ids()
    done = labels_of(name)
    base = f"/j/{token}"
    if not queue:
        return page("Adjudication", "<div class='card'><p>Nothing to adjudicate yet. This queue fills only once "
                                    "both annotators have answered an item and their verdicts differ.</p></div>",
                    html.escape(name))
    pos = max(0, min(pos, len(queue) - 1))
    iid = queue[pos]
    it = ITEMS[INDEX[iid]]
    cur = done.get(iid, {})
    ndone = sum(1 for i in queue if i in done)
    header = (f"<div class='bar'><i style='width:{100 * ndone / len(queue):.1f}%'></i></div>"
              f"<div class='top' style='margin-bottom:10px'><span>Disagreement <b>{pos + 1}</b> of {len(queue)} "
              f"&middot; {iid} &middot; resolved {ndone}/{len(queue)}"
              f"{' &middot; <span class=msg>saved</span>' if saved else ''}</span>"
              f"<span><a href='{base}/list'>all disagreements</a><a href='{base}/instructions'>instructions</a></span></div>")
    nav = (f"<div class='nav'><button type='submit'>Resolve &amp; next</button>"
           f"<a href='{base}/item/{max(0, pos - 1)}'><button type='button' class='sec'>&larr; previous</button></a>"
           f"<a href='{base}/item/{min(len(queue) - 1, pos + 1)}'><button type='button' class='sec'>skip &rarr;</button></a>"
           f"<span class='kbd'>keys: <kbd>1</kbd>..<kbd>{min(9, 2 * len(QUESTIONS))}</kbd> answers &nbsp; <kbd>Enter</kbd> save</span></div>")
    return page(f"Adjudication {pos + 1}", item_body(it, cur, f"{base}/item/{pos}", header, nav, ADJ_BANNER),
                html.escape(name) + " &middot; adjudicator")


def adj_list_page(token: str, name: str) -> bytes:
    queue = disagreement_ids()
    done = labels_of(name)
    base = f"/j/{token}"
    tiles = "".join(f"<a class='{'done' if i in done else ''}' href='{base}/item/{k}'>{k + 1}</a>"
                    for k, i in enumerate(queue))
    ndone = sum(1 for i in queue if i in done)
    body = (f"<div class='bar'><i style='width:{100 * ndone / max(1, len(queue)):.1f}%'></i></div>"
            f"<p class='top'>{ndone} of {len(queue)} disagreements resolved. "
            f"<a href='{base}/instructions'>instructions</a></p><div class='tiles'>{tiles}</div>")
    return page("Disagreements", body, html.escape(name) + " &middot; adjudicator")


def list_page(token: str, name: str) -> bytes:
    done = labels_of(name)
    base = f"/a/{token}"
    tiles = "".join(f"<a class='{'done' if it['item_id'] in done else ''}' href='{base}/item/{i}'>{i + 1}</a>"
                    for i, it in enumerate(ITEMS))
    nxt = next_unlabeled(done, 0)
    body = (f"<div class='bar'><i style='width:{100 * len(done) / len(ITEMS):.1f}%'></i></div>"
            f"<p class='top'>{len(done)} of {len(ITEMS)} labelled. "
            + (f"<a href='{base}/item/{nxt}'>continue with item {nxt + 1}</a>" if nxt >= 0
               else "<span class='msg'>All done — thank you.</span>")
            + f" <a href='{base}/instructions'>instructions</a></p><div class='tiles'>{tiles}</div>")
    return page("All items", body, html.escape(name))


def instructions_page(base: str, name: str, extra: str = "") -> bytes:
    body = (f"<p class='top'><a href='{base}/list'>all items</a> <a href='{base}/'>continue</a></p>{extra}"
            f"<div class='card' style='white-space:pre-wrap;font-family:system-ui,sans-serif;font-size:15px'>"
            f"{html.escape(INSTR)}</div>")
    return page("Instructions", body, html.escape(name))


def kappa(x: list[str], y: list[str]) -> float | None:
    if not x:
        return None
    po = sum(a == b for a, b in zip(x, y)) / len(x)
    cats = set(x) | set(y)
    pe = sum((x.count(c) / len(x)) * (y.count(c) / len(y)) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else None


def admin_page(token: str) -> bytes:
    names = list(ANNOTATORS.values()) + list(ADJUDICATORS.values())
    per = {n: labels_of(n) for n in names}
    queue = disagreement_ids()
    rows = "".join(f"<tr><td>{html.escape(n)}</td><td>{ROLE.get(n, '?')}</td><td>{len(per[n])}</td>"
                   f"<td>{(len(queue) if ROLE.get(n) == 'adjudicator' else len(ITEMS)) - len(per[n])}</td></tr>"
                   for n in names)
    agree = ""
    if len(PRIMARY) >= 2:
        a, b = PRIMARY[0], PRIMARY[1]
        both = [i for i in per.get(a, {}) if i in per.get(b, {})]
        if both:
            lines = []
            for q in QIDS:
                x = [per[a][i].get(q) for i in both]
                y = [per[b][i].get(q) for i in both]
                if all(v is not None for v in x + y):
                    kk = kappa(x, y)
                    lines.append(f"{q} agreement {sum(p == r for p, r in zip(x, y)) / len(both):.2f} "
                                 f"(&kappa; {'n/a' if kk is None else f'{kk:.2f}'})")
            n_adj = sum(1 for i in queue for n in ADJUDICATORS.values() if i in per.get(n, {}))
            degenerate = [n for n in PRIMARY
                          if per.get(n) and len({json.dumps({q: per[n][i].get(q) for q in QIDS}, sort_keys=True)
                                                 for i in per[n]}) == 1]
            warn = (f"<p style='color:var(--warn)'><b>Warning:</b> {', '.join(map(html.escape, degenerate))} gave the "
                    f"identical answer on every item labelled so far — an agreement filter against a constant rater "
                    f"is not a two-annotator result.</p>" if degenerate else "")
            agree = (f"<p>Co-labelled items: {len(both)}. " + "; ".join(lines) + ".</p>"
                     f"<p>Verdict disagreements: <b>{len(queue)}</b>; adjudicated <b>{n_adj}</b>; "
                     f"unresolved <b>{len(queue) - n_adj}</b>. "
                     f"An agreement-conditioned rate DROPS all {len(queue)}; the adjudicated rate resolves them.</p>"
                     f"{warn}")
    body = (f"<div class='card'><h3>Progress</h3><table><tr><th>Person</th><th>role</th><th>done</th>"
            f"<th>remaining</th></tr>{rows}</table>{agree}"
            f"<p><a href='/admin/{token}/export.jsonl'>export all labels (jsonl)</a></p></div>")
    return page("Admin", body, "admin")


def export(person: str | None) -> bytes:
    with LOCK:
        q = ("SELECT annotator, item_id, payload, comment, ts FROM answers"
             + (" WHERE annotator=?" if person else "") + " ORDER BY annotator, item_id")
        rows = DB.execute(q, (person,) if person else ()).fetchall()
        lq = ("SELECT annotator, item_id, q1, q2, comment, ts FROM labels"
              + (" WHERE annotator=?" if person else "") + " ORDER BY annotator, item_id")
        legacy = DB.execute(lq, (person,) if person else ()).fetchall()
    seen, out = set(), []
    for n, iid, payload, comment, ts in rows:
        ans = json.loads(payload)
        seen.add((n, iid))
        out.append({"annotator": n, "role": ROLE.get(n, "annotator"), "item_id": iid,
                    **{k: ans.get(k) for k in QIDS}, "answers": ans, "comment": comment, "ts": ts})
    for n, iid, q1, q2, comment, ts in legacy:
        if (n, iid) not in seen:
            out.append({"annotator": n, "role": ROLE.get(n, "annotator"), "item_id": iid, "q1": q1, "q2": q2,
                        "answers": {"q1": q1, "q2": q2}, "comment": comment, "ts": ts})
    out.sort(key=lambda r: (r["annotator"], r["item_id"]))
    return "".join(json.dumps(r) + "\n" for r in out).encode()


def parse_answers(form: dict) -> tuple[dict, bool]:
    ans = {q: (form.get(q, [""])[0]) for q in QIDS}
    ok = all(ans[q] in QVALS[q] for q in QIDS)
    return ans, ok


class H(BaseHTTPRequestHandler):
    server_version = "labelapp/2-adj"

    def log_message(self, fmt, *args):  # quiet, no tokens in logs
        pass

    def send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Robots-Tag", "noindex")
        self.end_headers()
        self.wfile.write(body)

    def redirect(self, loc: str):
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", loc)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        p = urlparse(self.path).path.rstrip("/") or "/"
        saved = "saved" in urlparse(self.path).query
        if p == "/":
            return self.send(200, page("Omission confirmation", "<div class='card'><p>This is a private annotation "
                                                                "task. Open the personal link you received; there is "
                                                                "nothing to see here without it.</p></div>"))
        if p == "/robots.txt":
            return self.send(200, b"User-agent: *\nDisallow: /\n", "text/plain")
        m = re.match(r"^/a/([A-Za-z0-9_-]+)(?:/(.*))?$", p)
        if m:
            name = person_of(m.group(1), ANNOTATORS)
            if not name:
                return self.send(404, page("Not found", "<p>Unknown link.</p>"))
            token, sub = m.group(1), m.group(2) or ""
            if sub == "":
                nxt = next_unlabeled(labels_of(name), 0)
                return self.redirect(f"/a/{token}/item/{nxt if nxt >= 0 else 0}")
            if sub == "list":
                return self.send(200, list_page(token, name))
            if sub == "instructions":
                return self.send(200, instructions_page(f"/a/{token}", name))
            if sub == "export.jsonl":
                return self.send(200, export(name), "application/x-ndjson")
            mi = re.match(r"^item/(\d+)$", sub)
            if mi and int(mi.group(1)) < len(ITEMS):
                return self.send(200, item_page(token, name, int(mi.group(1)), saved))
            return self.send(404, page("Not found", "<p>No such page.</p>"))
        m = re.match(r"^/j/([A-Za-z0-9_-]+)(?:/(.*))?$", p)
        if m:
            name = person_of(m.group(1), ADJUDICATORS)
            if not name:
                return self.send(404, page("Not found", "<p>Unknown link.</p>"))
            token, sub = m.group(1), m.group(2) or ""
            if sub == "":
                queue = disagreement_ids()
                done = labels_of(name)
                nxt = next((k for k, i in enumerate(queue) if i not in done), 0)
                return self.redirect(f"/j/{token}/item/{nxt}")
            if sub == "list":
                return self.send(200, adj_list_page(token, name))
            if sub == "instructions":
                return self.send(200, instructions_page(f"/j/{token}", name, ADJ_BANNER))
            if sub == "export.jsonl":
                return self.send(200, export(name), "application/x-ndjson")
            mi = re.match(r"^item/(\d+)$", sub)
            if mi:
                return self.send(200, adj_page(token, name, int(mi.group(1)), saved))
            return self.send(404, page("Not found", "<p>No such page.</p>"))
        m = re.match(r"^/admin/([A-Za-z0-9_-]+)(?:/(.*))?$", p)
        if m and is_admin(m.group(1)):
            if (m.group(2) or "") == "export.jsonl":
                return self.send(200, export(None), "application/x-ndjson")
            return self.send(200, admin_page(m.group(1)))
        return self.send(404, page("Not found", "<p>No such page.</p>"))

    def do_POST(self):
        p = urlparse(self.path).path
        n = int(self.headers.get("Content-Length") or 0)
        form = parse_qs(self.rfile.read(min(n, 200_000)).decode("utf-8", "replace"))
        comment = form.get("comment", [""])[0][:2000]
        m = re.match(r"^/a/([A-Za-z0-9_-]+)/item/(\d+)$", p)
        if m:
            name = person_of(m.group(1), ANNOTATORS)
            i, token = int(m.group(2)), m.group(1)
            if not name or i >= len(ITEMS):
                return self.send(404, page("Not found", "<p>No such page.</p>"))
            ans, ok = parse_answers(form)
            if not ok:
                return self.redirect(f"/a/{token}/item/{i}")
            record(name, ITEMS[i]["item_id"], ans, comment)
            nxt = next_unlabeled(labels_of(name), i + 1)
            return self.redirect(f"/a/{token}/item/{nxt}?saved=1" if nxt >= 0 else f"/a/{token}/list")
        m = re.match(r"^/j/([A-Za-z0-9_-]+)/item/(\d+)$", p)
        if m:
            name = person_of(m.group(1), ADJUDICATORS)
            pos, token = int(m.group(2)), m.group(1)
            queue = disagreement_ids()
            if not name or pos >= len(queue):
                return self.send(404, page("Not found", "<p>No such page.</p>"))
            ans, ok = parse_answers(form)
            if not ok:
                return self.redirect(f"/j/{token}/item/{pos}")
            record(name, queue[pos], ans, comment)
            done = labels_of(name)
            nxt = next((k for k in range(pos + 1, len(queue)) if queue[k] not in done), None)
            if nxt is None:
                nxt = next((k for k, i in enumerate(queue) if i not in done), None)
            return self.redirect(f"/j/{token}/item/{nxt}?saved=1" if nxt is not None else f"/j/{token}/list")
        return self.send(404, page("Not found", "<p>No such page.</p>"))


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", int(CFG.get("port", 8787))), H)
    print(f"labelapp/2-adj: {len(ITEMS)} items, {len(QUESTIONS)} questions, {len(ANNOTATORS)} annotators "
          f"(primary {PRIMARY}), {len(ADJUDICATORS)} adjudicators, 127.0.0.1:{CFG.get('port', 8787)}", flush=True)
    srv.serve_forever()
