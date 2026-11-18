#!/usr/bin/env python3
"""Omission confirmation — blinded annotation web app (stdlib only: http.server + sqlite3).

Items: human_sample_blind.jsonl (item_id, source, summary, fact) — no system identity anywhere in this app.
Access: per-annotator secret links /a/<token>/ ; admin at /admin/<admin_token>/ (progress, agreement, export).
Storage: labels.db (sqlite). Runs behind Caddy (TLS) on 127.0.0.1:PORT.
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
LOCK = threading.Lock()
DB = sqlite3.connect(HERE / CFG.get("db", "labels.db"), check_same_thread=False)
DB.execute("CREATE TABLE IF NOT EXISTS labels (annotator TEXT, item_id TEXT, q1 TEXT, q2 TEXT, comment TEXT, ts REAL, PRIMARY KEY (annotator, item_id))")
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
"""


def page(title: str, body: str, top: str = "") -> bytes:
    return f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{CSS}</style></head><body><div class='wrap'><div class='top'><span><b>Omission confirmation</b></span><span>{top}</span></div>{body}</div></body></html>".encode()


def hl(text: str, fact: str) -> str:
    words = {w.lower() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]+", fact) if len(w) > 2 and w.lower() not in STOP}
    out = []
    for tok in re.split(r"(\s+)", text):
        core = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", tok).lower()
        e = html.escape(tok)
        out.append(f"<mark>{e}</mark>" if core and core in words else e)
    return "".join(out)


def labels_of(annotator: str) -> dict[str, dict]:
    with LOCK:
        rows = DB.execute("SELECT item_id, q1, q2, comment FROM labels WHERE annotator=?", (annotator,)).fetchall()
    return {r[0]: {"q1": r[1], "q2": r[2], "comment": r[3]} for r in rows}


def annotator_of(token: str) -> str | None:
    for t, name in CFG["annotators"].items():
        if hmac.compare_digest(t, token):
            return name
    return None


def is_admin(token: str) -> bool:
    return hmac.compare_digest(CFG["admin_token"], token)


def next_unlabeled(done: dict, start: int) -> int:
    n = len(ITEMS)
    for k in range(n):
        i = (start + k) % n
        if ITEMS[i]["item_id"] not in done:
            return i
    return -1


def item_page(token: str, name: str, i: int, saved: bool) -> bytes:
    it = ITEMS[i]; done = labels_of(name); cur = done.get(it["item_id"], {})
    prog = len(done); n = len(ITEMS); base = f"/a/{token}"
    q = lambda k, v: "checked" if cur.get(k) == v else ""
    body = f"""
<div class='bar'><i style='width:{100 * prog / n:.1f}%'></i></div>
<div class='top' style='margin-bottom:10px'><span>Item <b>{i + 1}</b> of {n} &middot; {it['item_id']} &middot; labelled {prog}/{n}{" &middot; <span class='msg'>saved</span>" if saved else ""}</span>
<span><a href='{base}/list'>all items</a><a href='{base}/instructions'>instructions</a></span></div>
<div class='grid'>
 <div class='card'><h3>Source</h3><div class='src'>{hl(it['source'], it['fact'])}</div></div>
 <div>
  <div class='card' style='margin-bottom:14px'><h3>Summary</h3><div>{hl(it['summary'], it['fact'])}</div></div>
  <div class='card'><h3>Fact claimed to be missing from the summary</h3><div class='fact'>{html.escape(it['fact'])}</div>
  <form method='post' action='{base}/item/{i}' id='f'>
   <div class='q'><b>Q1. Is the fact stated in the source, or does it follow directly from it?</b>
    <div class='opt'><label><input type='radio' name='q1' value='A' {q('q1', 'A')} required>A — yes, supported by the source</label>
    <label><input type='radio' name='q1' value='B' {q('q1', 'B')}>B — no (absent, contradicted, or invented)</label></div></div>
   <div class='q'><b>Q2. Does the summary convey the fact? (paraphrases, pronouns and equivalent names count; wording need not match; ignore importance)</b>
    <div class='opt'><label><input type='radio' name='q2' value='A' {q('q2', 'A')} required>A — conveyed / covered</label>
    <label><input type='radio' name='q2' value='B' {q('q2', 'B')}>B — omitted / not conveyed</label></div></div>
   <div class='q'><b>Comment (optional)</b><textarea name='comment'>{html.escape(cur.get('comment') or '')}</textarea></div>
   <div class='nav'><button type='submit'>Save &amp; next</button>
    <a href='{base}/item/{max(0, i - 1)}'><button type='button' class='sec'>&larr; previous</button></a>
    <a href='{base}/item/{min(n - 1, i + 1)}'><button type='button' class='sec'>skip &rarr;</button></a>
    <span class='kbd'>keys: <kbd>1</kbd>/<kbd>2</kbd> Q1 A/B &nbsp; <kbd>3</kbd>/<kbd>4</kbd> Q2 A/B &nbsp; <kbd>Enter</kbd> save</span></div>
  </form></div>
 </div>
</div>
<script>
document.addEventListener('keydown',e=>{{if(e.target.tagName==='TEXTAREA')return;const m={{'1':['q1','A'],'2':['q1','B'],'3':['q2','A'],'4':['q2','B']}};
if(m[e.key]){{const el=document.querySelector(`input[name=${{m[e.key][0]}}][value=${{m[e.key][1]}}]`);if(el)el.checked=true;e.preventDefault();}}
else if(e.key==='Enter'){{document.getElementById('f').requestSubmit();}}}});
</script>"""
    return page(f"Item {i + 1}", body, f"{html.escape(name)}")


def list_page(token: str, name: str) -> bytes:
    done = labels_of(name); base = f"/a/{token}"
    tiles = "".join(f"<a class='{'done' if it['item_id'] in done else ''}' href='{base}/item/{i}'>{i + 1}</a>" for i, it in enumerate(ITEMS))
    nxt = next_unlabeled(done, 0)
    body = f"<div class='bar'><i style='width:{100 * len(done) / len(ITEMS):.1f}%'></i></div><p class='top'>{len(done)} of {len(ITEMS)} labelled. " + (f"<a href='{base}/item/{nxt}'>continue with item {nxt + 1}</a>" if nxt >= 0 else "<span class='msg'>All done — thank you.</span>") + f" <a href='{base}/instructions'>instructions</a></p><div class='tiles'>{tiles}</div>"
    return page("All items", body, html.escape(name))


def instructions_page(token: str, name: str) -> bytes:
    base = f"/a/{token}"
    body = f"<p class='top'><a href='{base}/list'>all items</a> <a href='{base}/'>continue</a></p><div class='card' style='white-space:pre-wrap;font-family:system-ui,sans-serif;font-size:15px'>{html.escape(INSTR)}</div>"
    return page("Instructions", body, html.escape(name))


def kappa(x: list[str], y: list[str]) -> float | None:
    if not x:
        return None
    po = sum(a == b for a, b in zip(x, y)) / len(x)
    cats = set(x) | set(y); pe = sum((x.count(c) / len(x)) * (y.count(c) / len(y)) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else None


def admin_page(token: str) -> bytes:
    names = list(CFG["annotators"].values()); per = {n: labels_of(n) for n in names}
    rows = "".join(f"<tr><td>{html.escape(n)}</td><td>{len(per[n])}</td><td>{len(ITEMS) - len(per[n])}</td></tr>" for n in names)
    agree = ""
    if len(names) >= 2:
        a, b = names[0], names[1]; both = [i for i in per[a] if i in per[b]]
        if both:
            x1, y1 = [per[a][i]["q1"] for i in both], [per[b][i]["q1"] for i in both]; x2, y2 = [per[a][i]["q2"] for i in both], [per[b][i]["q2"] for i in both]
            f = lambda v: "n/a" if v is None else f"{v:.2f}"
            agree = f"<p>Co-labelled items: {len(both)}. Q1 agreement {sum(p == q for p, q in zip(x1, y1)) / len(both):.2f} (κ {f(kappa(x1, y1))}); Q2 agreement {sum(p == q for p, q in zip(x2, y2)) / len(both):.2f} (κ {f(kappa(x2, y2))}).</p>"
    body = f"<div class='card'><h3>Progress</h3><table><tr><th>Annotator</th><th>labelled</th><th>remaining</th></tr>{rows}</table>{agree}<p><a href='/admin/{token}/export.jsonl'>export all labels (jsonl)</a></p></div>"
    return page("Admin", body, "admin")


def export(annotator: str | None) -> bytes:
    with LOCK:
        q = "SELECT annotator, item_id, q1, q2, comment, ts FROM labels" + (" WHERE annotator=?" if annotator else "") + " ORDER BY annotator, item_id"
        rows = DB.execute(q, (annotator,) if annotator else ()).fetchall()
    return "".join(json.dumps({"annotator": r[0], "item_id": r[1], "q1": r[2], "q2": r[3], "comment": r[4], "ts": r[5]}) + "\n" for r in rows).encode()


class H(BaseHTTPRequestHandler):
    server_version = "labelapp/1"

    def log_message(self, fmt, *args):  # quiet, no tokens in logs
        pass

    def send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8", extra: dict | None = None):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.send_header("Referrer-Policy", "no-referrer"); self.send_header("X-Robots-Tag", "noindex")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)

    def redirect(self, loc: str):
        self.send_response(HTTPStatus.SEE_OTHER); self.send_header("Location", loc); self.send_header("Content-Length", "0"); self.end_headers()

    def do_GET(self):
        p = urlparse(self.path).path.rstrip("/") or "/"
        if p == "/":
            return self.send(200, page("Omission confirmation", "<div class='card'><p>This is a private annotation task. Open the personal link you received; there is nothing to see here without it.</p></div>"))
        if p == "/robots.txt":
            return self.send(200, b"User-agent: *\nDisallow: /\n", "text/plain")
        m = re.match(r"^/a/([A-Za-z0-9_-]+)(?:/(.*))?$", p)
        if m:
            name = annotator_of(m.group(1))
            if not name:
                return self.send(404, page("Not found", "<p>Unknown link.</p>"))
            token, sub = m.group(1), m.group(2) or ""
            if sub == "":
                nxt = next_unlabeled(labels_of(name), 0); return self.redirect(f"/a/{token}/item/{nxt if nxt >= 0 else 0}")
            if sub == "list":
                return self.send(200, list_page(token, name))
            if sub == "instructions":
                return self.send(200, instructions_page(token, name))
            if sub == "export.jsonl":
                return self.send(200, export(name), "application/x-ndjson")
            mi = re.match(r"^item/(\d+)$", sub)
            if mi and int(mi.group(1)) < len(ITEMS):
                return self.send(200, item_page(token, name, int(mi.group(1)), "saved" in urlparse(self.path).query))
            return self.send(404, page("Not found", "<p>No such page.</p>"))
        m = re.match(r"^/admin/([A-Za-z0-9_-]+)(?:/(.*))?$", p)
        if m and is_admin(m.group(1)):
            if (m.group(2) or "") == "export.jsonl":
                return self.send(200, export(None), "application/x-ndjson")
            return self.send(200, admin_page(m.group(1)))
        return self.send(404, page("Not found", "<p>No such page.</p>"))

    def do_POST(self):
        p = urlparse(self.path).path
        m = re.match(r"^/a/([A-Za-z0-9_-]+)/item/(\d+)$", p)
        name = annotator_of(m.group(1)) if m else None
        if not name or int(m.group(2)) >= len(ITEMS):
            return self.send(404, page("Not found", "<p>No such page.</p>"))
        n = int(self.headers.get("Content-Length") or 0); form = parse_qs(self.rfile.read(min(n, 200_000)).decode("utf-8", "replace"))
        q1, q2 = form.get("q1", [""])[0], form.get("q2", [""])[0]; comment = form.get("comment", [""])[0][:2000]
        i = int(m.group(2)); token = m.group(1)
        if q1 in ("A", "B") and q2 in ("A", "B"):
            with LOCK:
                DB.execute("INSERT OR REPLACE INTO labels (annotator, item_id, q1, q2, comment, ts) VALUES (?,?,?,?,?,?)", (name, ITEMS[i]["item_id"], q1, q2, comment, time.time())); DB.commit()
            nxt = next_unlabeled(labels_of(name), i + 1)
            return self.redirect(f"/a/{token}/item/{nxt}?saved=1" if nxt >= 0 else f"/a/{token}/list")
        return self.redirect(f"/a/{token}/item/{i}")


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", int(CFG.get("port", 8787))), H)
    print(f"labelapp: {len(ITEMS)} items, {len(CFG['annotators'])} annotators, 127.0.0.1:{CFG.get('port', 8787)}", flush=True)
    srv.serve_forever()
