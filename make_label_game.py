"""Generate an interactive 'same person?' labeling game as a self-contained HTML
file (reports/label_game.html).

Editing a 359-row CSV by hand is the slow, error-prone way to build the grouping
ground truth. This walks the tracks one at a time, suggests the most visually
similar person already created (from the ArcFace track templates), and lets you
decide with a single keypress: Same / New person / pick Other / Ignore. It builds
consistent per-person clusters and exports a data/ground_truth.csv you drop in
place, then score with eval_labeled.py.

Everything is embedded -- crops as base64, nearest-neighbour lists precomputed --
so the page runs offline from file:// with no server. Progress persists to the
browser's localStorage, so you can stop and resume.

    python make_label_game.py      # -> reports/label_game.html, then open it
"""
import base64
import json

import cv2
import numpy as np
import pandas as pd

import config
import util

log = util.get_logger()

THUMB = 128
TOPK = 25          # nearest neighbours embedded per track (enough to find a match)


def _best_crop_per_track(faces: pd.DataFrame) -> dict:
    faces = faces.copy()
    faces["score"] = (faces["det_score"] * faces["blur_var"]
                      / (1.0 + faces["nose_offset"]))
    return {str(tid): g.loc[g["score"].idxmax(), "crop_file"]
            for tid, g in faces.groupby("track_id")}


def _thumb_b64(crop_file) -> str:
    im = cv2.imread(str(config.FACE_DIR / crop_file)) if crop_file else None
    if im is None:
        return ""
    ok, buf = cv2.imencode(".jpg", cv2.resize(im, (THUMB, THUMB)))
    return base64.b64encode(buf).decode() if ok else ""


def _neighbors(track_ids, templates) -> dict:
    """track_id -> [[neighbor_id, sim], ...] top-K by cosine, so the game can
    suggest 'is this the same as ...' from the strongest visual match."""
    sim = templates @ templates.T
    np.fill_diagonal(sim, -1.0)
    out = {}
    for i, tid in enumerate(track_ids):
        order = np.argsort(-sim[i])[:TOPK]
        out[tid] = [[track_ids[j], round(float(sim[i][j]), 3)] for j in order]
    return out


def build_data() -> dict:
    faces = pd.read_csv(config.FACES_CSV)
    ident = pd.read_csv(config.IDENTITIES_CSV).sort_values("first_sec")
    best = _best_crop_per_track(faces)

    tmpl_by_track, neighbors = {}, {}
    if config.TEMPLATE_FILE.exists():
        data = np.load(config.TEMPLATE_FILE)
        tids = [str(t) for t in data["track_ids"]]
        neighbors = _neighbors(tids, data["templates"])
        tmpl_by_track = {t: True for t in tids}

    existing = {}
    gt = config.DATA / "ground_truth.csv"
    if gt.exists():
        df = pd.read_csv(gt, dtype=str).fillna("")
        if "true_id" in df.columns:
            existing = {str(r.track_id): r.true_id.strip()
                        for r in df.itertuples(index=False) if r.true_id.strip()}

    tracks = []
    for _, t in ident.iterrows():
        tid = str(t["track_id"])
        tracks.append({
            "id": tid,
            "thumb": _thumb_b64(best.get(tid, "")),
            "first": round(float(t["first_sec"]), 1),
            "n": int(t["n_faces"]),
            "pred": str(t["face_id"]),
            "prior": existing.get(tid, ""),
            "has_tmpl": tid in tmpl_by_track,
        })
    return {"tracks": tracks, "neighbors": neighbors}


def run() -> None:
    config.ensure_dirs()
    data = build_data()
    html = _TEMPLATE.replace("/*DATA*/", json.dumps(data))
    util.write_text_atomic(config.REPORT_DIR / "label_game.html", html)
    log.info("labeling game: %d tracks -> reports/label_game.html "
             "(open it in a browser)", len(data["tracks"]))


# The whole app: static HTML/CSS/JS with a /*DATA*/ placeholder. Kept out of an
# f-string so JS braces need no escaping.
_TEMPLATE = r"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Same person? — identity labeler</title>
<style>
 :root{color-scheme:light dark}
 body{font:15px/1.4 system-ui,sans-serif;margin:0;background:#111;color:#eee}
 header{padding:10px 16px;background:#1b1b1b;position:sticky;top:0;
   display:flex;gap:16px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #333}
 .bar{flex:1;height:8px;background:#333;border-radius:4px;overflow:hidden;min-width:120px}
 .bar>i{display:block;height:100%;background:#3aa;width:0}
 button{font:inherit;padding:8px 14px;border:0;border-radius:8px;cursor:pointer;
   background:#2a2a2a;color:#eee}
 button:hover{background:#3a3a3a}
 .same{background:#1f7a1f}.new{background:#7a5a1f}.ign{background:#5a1f1f}
 main{max-width:920px;margin:0 auto;padding:18px}
 .stage{display:flex;gap:24px;align-items:center;justify-content:center;
   flex-wrap:wrap;margin:10px 0 18px}
 .card{text-align:center}
 .card img{width:128px;height:128px;border-radius:10px;border:3px solid #444;display:block}
 .card.q img{border-color:#3aa}.card.s img{border-color:#1f7a1f}
 .mini{display:flex;gap:4px;justify-content:center;margin-top:6px}
 .mini img{width:38px;height:38px;border-radius:5px;border:1px solid #444}
 .meta{font:12px monospace;color:#aaa;margin-top:4px}
 .sim{font-weight:bold;color:#3aa}
 .actions{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin:8px 0}
 .keys{color:#888;font-size:12px;text-align:center;margin-top:8px}
 #picker{display:none;flex-wrap:wrap;gap:8px;margin-top:14px;
   max-height:340px;overflow:auto;border-top:1px solid #333;padding-top:12px}
 .person{text-align:center;cursor:pointer;padding:4px;border-radius:8px;background:#1b1b1b}
 .person:hover{background:#2a2a2a}.person img{width:64px;height:64px;border-radius:6px}
 .person b{display:block;font:12px monospace}
 #done{display:none;text-align:center;padding:40px}
 a.dl{display:inline-block;margin-top:12px;background:#1f7a1f;color:#fff;
   padding:10px 18px;border-radius:8px;text-decoration:none}
</style></head><body>
<header>
 <b>Same person?</b>
 <div class=bar><i id=prog></i></div>
 <span id=stat></span>
 <button onclick=exportCsv()>Export CSV (E)</button>
 <button onclick=reset()>Reset</button>
</header>
<main>
 <div id=game>
  <div class=stage id=stage></div>
  <div class=actions id=actions></div>
  <div class=keys>Y = same &nbsp; N = new person &nbsp; O = other &nbsp; I = ignore (non-face) &nbsp; U = undo</div>
  <div id=picker></div>
 </div>
 <div id=done>
  <h2>All tracks labeled 🎉</h2>
  <p id=summary></p>
  <a class=dl id=dllink>Download ground_truth.csv</a>
  <p class=keys>Save it to <code>data/ground_truth.csv</code>, then run
   <code>python eval_labeled.py</code>.</p>
 </div>
</main>
<script>
const DATA = /*DATA*/;
const TR = DATA.tracks, NB = DATA.neighbors;
const KEY = "labelgame_" + TR.length;
const byId = Object.fromEntries(TR.map(t => [t.id, t]));

let st = load();
function load(){
  try{ const s = JSON.parse(localStorage.getItem(KEY)); if(s && s.assign) return s; }catch(e){}
  // Seed from any prior true_id so existing work isn't lost.
  const assign = {}, people = {}; let next = 1;
  const lbl2p = {};
  for(const t of TR){ if(t.prior && t.prior!=="x" && t.prior!=="ignore"){
      if(!(t.prior in lbl2p)){ lbl2p[t.prior] = "P"+(next++); people[lbl2p[t.prior]]={rep:t.id,members:[]}; }
      const p = lbl2p[t.prior]; assign[t.id]=p; people[p].members.push(t.id);
    } else if(t.prior==="x"||t.prior==="ignore"){ assign[t.id]="__ignore"; } }
  // Start at the first undecided track.
  let idx = 0; while(idx<TR.length && TR[idx].id in assign) idx++;
  return {assign, people, next, idx, hist:[]};
}
function save(){ localStorage.setItem(KEY, JSON.stringify(st)); }

function suggest(t){                         // nearest already-assigned person
  for(const [nid,sim] of (NB[t.id]||[])){
    const p = st.assign[nid];
    if(p && p!=="__ignore") return {p, sim, via:nid};
  }
  return null;
}
function newPerson(t){ const p="P"+(st.next++); st.people[p]={rep:t.id,members:[]}; return p; }
function assignTo(t,p){ st.assign[t.id]=p; if(p!=="__ignore") st.people[p].members.push(t.id); }

function decide(kind, pid){
  const t = TR[st.idx];
  st.hist.push({id:t.id, next:st.next});
  if(kind==="ignore") assignTo(t,"__ignore");
  else if(kind==="new") assignTo(t, newPerson(t));
  else assignTo(t, pid);
  st.idx++; save(); render();
}
function undo(){
  if(!st.hist.length) return;
  const h = st.hist.pop(); const t = byId[h.id]; const p = st.assign[h.id];
  if(p && p!=="__ignore"){ const m=st.people[p].members; m.splice(m.indexOf(h.id),1);
    if(!m.length) delete st.people[p]; }
  delete st.assign[h.id]; st.next=h.next;
  st.idx = TR.findIndex(x=>x.id===h.id); save(); render();
}

function card(t, cls, label){
  const mem = (st.people[st.assign[t.id]]||{}).members || [];
  const minis = mem.slice(0,4).filter(id=>id!==t.id)
    .map(id=>`<img src="data:image/jpeg;base64,${byId[id].thumb}">`).join("");
  return `<div class="card ${cls}">
    <img src="data:image/jpeg;base64,${t.thumb}">
    <div class=meta>${label||t.id} · ${t.first}s</div>
    ${minis?`<div class=mini>${minis}</div>`:""}</div>`;
}

function render(){
  document.getElementById("picker").style.display="none";
  const total=TR.length, done=st.idx;
  document.getElementById("prog").style.width=(100*done/total)+"%";
  document.getElementById("stat").textContent=
    `${done}/${total} · ${Object.keys(st.people).length} people`;
  if(st.idx>=total) return finish();
  const t=TR[st.idx], sg=suggest(t);
  const stage=document.getElementById("stage"), act=document.getElementById("actions");
  if(sg){
    const rep=byId[st.people[sg.p].rep];
    stage.innerHTML = card(t,"q","THIS track") +
      `<div style="font-size:28px;color:#3aa">?</div>` +
      card(rep,"s",sg.p) +
      `<div class=meta>match <span class=sim>${sg.sim.toFixed(2)}</span></div>`;
    act.innerHTML =
      `<button class=same onclick="decide('assign','${sg.p}')">✔ Same (Y)</button>
       <button class=new onclick="decide('new')">✦ New person (N)</button>
       <button onclick=openPicker()>⋯ Other (O)</button>
       <button class=ign onclick="decide('ignore')">✕ no face (I)</button>`;
  } else {
    stage.innerHTML = card(t,"q","THIS track") +
      `<div class=meta>no similar person yet — start a new one</div>`;
    act.innerHTML =
      `<button class=new onclick="decide('new')">✦ New person (N)</button>
       ${Object.keys(st.people).length?`<button onclick=openPicker()>⋯ Pick existing (O)</button>`:""}
       <button class=ign onclick="decide('ignore')">✕ no face (I)</button>`;
  }
}

function openPicker(){
  const p=document.getElementById("picker");
  const people=Object.entries(st.people)
    .sort((a,b)=>b[1].members.length-a[1].members.length);
  p.innerHTML = people.map(([pid,info])=>
    `<div class=person onclick="decide('assign','${pid}')">
       <img src="data:image/jpeg;base64,${byId[info.rep].thumb}">
       <b>${pid}</b><span class=meta>${info.members.length}</span></div>`).join("")
    + `<div class=person onclick="decide('new')"><div style="width:64px;height:64px;
        display:flex;align-items:center;justify-content:center;font-size:32px">＋</div>
        <b>new</b></div>`;
  p.style.display="flex";
}

function finish(){
  document.getElementById("game").style.display="none";
  const d=document.getElementById("done"); d.style.display="block";
  const np=Object.keys(st.people).length;
  const ign=Object.values(st.assign).filter(x=>x==="__ignore").length;
  document.getElementById("summary").textContent=
    `${TR.length} tracks → ${np} people, ${ign} ignored.`;
  const rows=[["track_id","predicted_face_id","n_faces","first_sec","review","true_id"]];
  for(const t of TR){ const p=st.assign[t.id];
    rows.push([t.id,t.pred,t.n,t.first,"", p==="__ignore"?"x":(p||"")]); }
  const csv=rows.map(r=>r.join(",")).join("\n");
  const a=document.getElementById("dllink");
  a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));
  a.download="ground_truth.csv";
}
function exportCsv(){ st.idx=TR.length; finish(); }
function reset(){ if(confirm("Discard all labels and start over?")){
  localStorage.removeItem(KEY); st=load(); document.getElementById("game").style.display="";
  document.getElementById("done").style.display="none"; render(); } }

addEventListener("keydown",e=>{
  if(st.idx>=TR.length) return;
  const k=e.key.toLowerCase();
  const sg=suggest(TR[st.idx]);
  if(k==="y"&&sg) decide("assign",sg.p);
  else if(k==="n") decide("new");
  else if(k==="i") decide("ignore");
  else if(k==="o") openPicker();
  else if(k==="u") undo();
  else if(k==="e") exportCsv();
});
render();
</script></body></html>"""


if __name__ == "__main__":
    run()
