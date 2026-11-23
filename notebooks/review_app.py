"""Tiny local UI for the manual checks of the generated data (no extra packages needed).

    cd notebooks && ../.venv/bin/python review_app.py        # opens http://localhost:8765

Tab «Відмінки»: edit resources/name_declensions.csv row by row; changes are saved to the file immediately.
Tab «Документи»: read generated documents with highlighted entities and leave a verdict / comment per document;
verdicts go to data/synthetic/review_notes.json. After editing the table, re-run generate_data.ipynb.
"""
import csv
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
DECL = HERE / "resources" / "name_declensions.csv"
DATA = HERE.parent / "data" / "synthetic"
NOTES = DATA / "review_notes.json"
SPLITS = ["dev", "train", "test_a", "test_b"]
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
_docs = {}


def read_decl():
    with open(DECL, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def write_decl(fields, rows):
    tmp = DECL.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    tmp.replace(DECL)


def docs(split):
    if split not in _docs:
        with open(DATA / f"{split}.jsonl", encoding="utf-8") as f:
            _docs[split] = [json.loads(l) for l in f]
    return _docs[split]


def notes():
    return json.loads(NOTES.read_text(encoding="utf-8")) if NOTES.exists() else {}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, obj, ctype="application/json"):
        body = obj.encode() if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/":
            return self.send(PAGE, "text/html")
        if u.path == "/api/decl":
            fields, rows = read_decl()
            return self.send({"fields": fields, "rows": rows})
        if u.path == "/api/doc":
            split = q.get("split", "dev"); ds = docs(split)
            if q.get("id"):
                matches = [n for n, row in enumerate(ds) if str(row["id"]) == str(q["id"])]
                if not matches:
                    return self.send({"error": f"Document ID {q['id']} was not found in split {split}",
                                      "i": 0, "n": len(ds), "doc": ds[0],
                                      "note": {}, "reviewed": sum(1 for x in ds if x["id"] in notes()),
                                      "found": False})
                i = matches[0]
            else:
                i = max(0, min(int(q.get("i", 0)), len(ds) - 1))
            d = ds[i]
            return self.send({"i": i, "n": len(ds), "doc": d, "note": notes().get(d["id"], {}),
                              "reviewed": sum(1 for x in ds if x["id"] in notes()), "found": True})
        self.send_error(404)

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        if self.path == "/api/decl":
            fields, rows = read_decl()
            for r in rows:
                if r["слово"] == data["слово"] and r["стать"] == data["стать"] and r["тип"] == data["тип"]:
                    r.update({k: v for k, v in data.items() if k in fields})
            write_decl(fields, rows)
            return self.send({"ok": True})
        if self.path == "/api/note":
            n = notes(); n[data.pop("id")] = data
            NOTES.write_text(json.dumps(n, ensure_ascii=False, indent=1), encoding="utf-8")
            return self.send({"ok": True})
        self.send_error(404)


PAGE = r"""<!doctype html><html lang="uk"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Перевірка даних</title>
<style>
:root{--bg:#1f2124;--panel:#2a2d31;--fg:#dcdcd6;--muted:#9a9a92;--line:#3a3d42;--acc:#8fb3d9;
--per:#3d5a80;--adr:#5b4a7a;--num:#6b5a2e;--inf:#2f6650;--clear:#4a4f57;--bad:#e07a6a;--ok:#8cc084}
:root[data-theme=light]{--bg:#f4f1ea;--panel:#fbf9f4;--fg:#2b2b28;--muted:#77756d;--line:#dcd8cc;--acc:#2f5f8f;
--per:#cfe0f5;--adr:#e4d8f2;--num:#f3e6c2;--inf:#cfeadb;--clear:#e2e2e2;--bad:#b3402f;--ok:#3f7d36}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.7 -apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;background:var(--panel);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;z-index:2}
button,select,input{font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:4px 10px}
button{cursor:pointer}button.on{border-color:var(--acc);color:var(--acc)}
main{max-width:1100px;margin:0 auto;padding:16px}.muted{color:var(--muted);font-size:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:14px}
.forms{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;margin:10px 0}
.forms label{font-size:13px;color:var(--muted)}.forms input{width:100%}
.forms input.changed{border-color:var(--acc)}
.status-ok{color:var(--ok)}.status-todo{color:var(--bad)}
#text{white-space:pre-wrap;max-width:80ch;font-size:18px;line-height:1.85}
.e{border-radius:4px;padding:1px 3px;cursor:help}
.e.ОСОБА{background:var(--per)}.e.АДРЕСА{background:var(--adr)}.e.НОМЕР{background:var(--num)}.e.ІНФОРМАЦІЯ{background:var(--inf)}
.e.clear{background:var(--clear)}.e.unsure{outline:2px dashed var(--bad);outline-offset:1px}
.tag{font-size:11px;color:var(--muted);margin-left:2px;vertical-align:super}
.legend span{margin-right:12px;font-size:14px}
textarea{width:100%;min-height:60px;font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:6px}
</style></head><body>
<header>
 <b>Перевірка даних</b>
 <button id="t-decl" class="on" onclick="tab('decl')">Відмінки</button>
 <button id="t-docs" onclick="tab('docs')">Документи</button>
 <span style="flex:1"></span>
 <button onclick="theme()">☀︎ / ☾</button>
</header>
<main>
<section id="decl">
 <p class="muted">Виправте форму прямо в полі — зберігається одразу (рамка підсвічується). «Все правильно» знімає позначку «перевірити».
 Після правок перезапустіть <code>generate_data.ipynb</code>.</p>
 <p><label><input type="checkbox" id="onlyTodo" checked onchange="renderDecl()"> показувати лише ті, що треба перевірити</label>
 &nbsp;<span id="declCount" class="muted"></span></p>
 <div id="declList"></div>
</section>
<section id="docs" hidden>
 <div class="card">
  <select id="split" onchange="loadDoc(0)"></select>
  <button onclick="loadDoc(cur.i-1)">← назад</button>
  <input id="num" type="number" min="1" style="width:90px" onchange="loadDoc(this.value-1)">
  <input id="docid" type="text" placeholder="document_id" style="width:125px" onchange="loadDocById(this.value)">
  <button onclick="loadDocById($("docid").value)">знайти за ID</button>
  <span id="pos" class="muted"></span>
  <button onclick="loadDoc(cur.i+1)">далі →</button>
  <div class="legend muted" style="margin-top:8px">
   <span><span class="e ОСОБА">ім'я</span></span><span><span class="e clear">справжнє відкрите ім'я</span></span>
   <span><span class="e АДРЕСА">адреса</span></span><span><span class="e НОМЕР">номер</span></span><span><span class="e ІНФОРМАЦІЯ">інформація</span></span>
   <span><span class="e ОСОБА unsure">відмінок невпевнений</span></span><span>? — стать навмання</span>
  </div>
  <div class="muted">Наведіть на виділення — побачите мітку, відмінок, стать. Клавіші: ← → гортати, 1 — ок, 2 — є проблема.</div>
 </div>
 <div class="card" id="text"></div>
 <div class="card">
  <b>Ваш вердикт:</b> <button id="v-ok" onclick="verdict('ок')">1 · ок</button>
  <button id="v-bad" onclick="verdict('проблема')">2 · є проблема</button>
  <span id="saved" class="muted"></span>
  <textarea id="comment" placeholder="Що не так (необов'язково). Зберігається, коли виходите з поля." onblur="verdict()"></textarea>
 </div>
</section>
</main>
<script>
const CASES=["називний","родовий","давальний","знахідний","орудний","місцевий","кличний"];
let decl=[],cur={i:0},note={};
const $=id=>document.getElementById(id);
const esc=s=>s.replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function theme(){const r=document.documentElement;r.dataset.theme=r.dataset.theme==="light"?"":"light";try{localStorage.theme=r.dataset.theme}catch(e){}}
try{document.documentElement.dataset.theme=localStorage.theme||""}catch(e){}
function tab(t){for(const x of["decl","docs"]){$(x).hidden=x!==t;$("t-"+x).classList.toggle("on",x===t)}if(t==="docs"&&!cur.doc)loadDoc(0)}
async function post(url,obj){await fetch(url,{method:"POST",body:JSON.stringify(obj)})}
async function loadDecl(){decl=(await (await fetch("/api/decl")).json()).rows;renderDecl()}
async function loadDocById(id){
 const split=$("split").value||"dev";
 const d=await (await fetch(`/api/doc?split=${encodeURIComponent(split)}&id=${encodeURIComponent(id.trim())}`)).json();
 if(!d.found){$("pos").textContent=d.error;return;}
 cur={i:d.i,doc:d.doc};note=d.note||{};
 $("num").value=d.i+1;$("docid").value=d.doc.id;$("pos").textContent=`з ${d.n} · переглянуто ${d.reviewed}`;
 renderDoc(d);
}
function renderDecl(){
 const only=$("onlyTodo").checked, rows=decl.filter(r=>!only||r["перевірити"]==="так");
 $("declCount").textContent=`лишилось перевірити: ${decl.filter(r=>r["перевірити"]==="так").length} з ${decl.length}`;
 $("declList").innerHTML=rows.map(r=>{const k=decl.indexOf(r);return `<div class="card">
  <b style="font-size:20px">${esc(r["слово"])}</b> <span class="muted">${esc(r["тип"])}, ${r["стать"]==="ч"?"чоловіче":"жіноче"}</span>
  <span class="${r["перевірити"]==="так"?"status-todo":"status-ok"}" style="float:right">${r["перевірити"]==="так"?"треба перевірити":"перевірено"}</span>
  ${r["що змінено"]?`<div class="muted">Що виправлено автоматично: ${esc(r["що змінено"])}</div>`:""}
  <div class="forms">${CASES.map(c=>`<div><label>${c}</label><input value="${esc(r[c])}" onchange="edit(${k},'${c}',this)"></div>`).join("")}</div>
  <input style="width:100%" placeholder="коментар (необов'язково)" value="${esc(r["коментар перевіряльника"]||"")}" onchange="edit(${k},'коментар перевіряльника',this)">
  <div style="margin-top:8px"><button onclick="approve(${k},'')">✓ Все правильно</button>
  <button onclick="approve(${k},'так')">повернути в «треба перевірити»</button></div></div>`}).join("")||"<p>Усе перевірено 🎉</p>";
}
function key(r){return {"слово":r["слово"],"стать":r["стать"],"тип":r["тип"]}}
async function edit(k,field,el){decl[k][field]=el.value.trim();el.classList.add("changed");await post("/api/decl",{...key(decl[k]),[field]:decl[k][field]})}
async function approve(k,v){decl[k]["перевірити"]=v;await post("/api/decl",{...key(decl[k]),"перевірити":v});renderDecl()}
async function loadDoc(i){
 const split=$("split").value||"dev";const d=await (await fetch(`/api/doc?split=${split}&i=${i}`)).json();
 cur={i:d.i,doc:d.doc};note=d.note||{};$("num").value=d.i+1;$("pos").textContent=`з ${d.n} · переглянуто ${d.reviewed}`;
 renderDoc(d);
}
function renderDoc(d){
 const t=d.doc.original;let html="",last=0;
 for(const a of [...d.doc.annotations].sort((x,y)=>x.start-y.start)){
  const typ=a.pseudonym.split("_")[0],clear=a.subtype==="clear_name",unsure=typ==="ОСОБА"&&!clear&&a.case_confident===false;
  const tip=[a.pseudonym,a.subtype,a.case&&("відмінок: "+a.case+(a.case_confident?"":" (невпевнено)")),
   a.gender&&("стать: "+(a.gender==="ч"?"ч":"ж")+(a.gender_from_context?" (з контексту)":" (навмання)"))].filter(Boolean).join(" · ");
  html+=esc(t.slice(last,a.start))+`<span class="e ${typ}${clear?" clear":""}${unsure?" unsure":""}" title="${esc(tip)}">${esc(t.slice(a.start,a.end))}</span>`
   +`<span class="tag">${a.pseudonym.replace("ОСОБА_","О").replace("АДРЕСА_","А").replace("НОМЕР_","Н").replace("ІНФОРМАЦІЯ_","І")}${a.gender&&!a.gender_from_context?"?":""}</span>`;
  last=a.end}
 $("text").innerHTML=html+esc(t.slice(last));$("comment").value=note.comment||"";showVerdict();window.scrollTo(0,0)}
function showVerdict(){$("v-ok").classList.toggle("on",note.verdict==="ок");$("v-bad").classList.toggle("on",note.verdict==="проблема");$("saved").textContent=note.verdict?"збережено":""}
async function verdict(v){if(v)note.verdict=v;note.comment=$("comment").value;if(!note.verdict&&!note.comment)return;
 await post("/api/note",{id:cur.doc.id,split:$("split").value,...note});showVerdict()}
document.addEventListener("keydown",e=>{if($("docs").hidden||["INPUT","TEXTAREA","SELECT"].includes(e.target.tagName))return;
 if(e.key==="ArrowRight")loadDoc(cur.i+1);if(e.key==="ArrowLeft")loadDoc(cur.i-1);if(e.key==="1")verdict("ок");if(e.key==="2")verdict("проблема")});
$("split").innerHTML=__SPLITS__.map(s=>`<option>${s}</option>`).join("");loadDecl();
</script></body></html>""".replace("__SPLITS__", json.dumps(SPLITS))

if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    url = f"http://localhost:{PORT}"
    print(f"Відкрийте {url}  (зупинити: Ctrl+C)")
    webbrowser.open(url)
    srv.serve_forever()
