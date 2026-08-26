#!/usr/bin/env python3
"""One self-contained page carrying the 240 tiles of `T40_RULE_V15` §3 and their verdict controls.

    PYTHONPATH=src:scripts .venv/bin/python scripts/build_empty_mask_page.py \
        --look runs/pr08-empty-mask-look --out runs/pr08-empty-mask-look/page.html

WHY A PAGE
----------
The verdicts have to end up somewhere a later reader can check them against the tiles that produced
them, and a directory of images plus a person's memory is not that. ``build_review_page.py`` exists
for the same reason and this is its sibling for a different question -- kept separate rather than
bolted on, because that page shows MASKS and this one must not (V15 §3), and one page that
sometimes hides the mask is one page that will eventually show it by accident.

WHAT THE PAGE MUST NOT CONTAIN, AND THIS SCRIPT IS WHERE THAT IS ENFORCED
------------------------------------------------------------------------
No mask, no overlay, no area fraction, no stratum, no episode id, no frame index. All of those live
in ``SAMPLE.json``, which this script reads for the tile ORDER and deliberately does not copy into
the page: the embedded payload is ``{tile, image}`` and nothing else. V15 §3's list of leaks is the
test to run against any change here.

WHAT IT DOES NOT DO
-------------------
Records no verdict of its own, computes no split, evaluates none of V15 §5's outcomes.
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

STYLE = """
:root{
  --ground:#0E1113; --surface:#171B1E; --raised:#1E2429; --line:#2A3238;
  --ink:#E6ECEF; --muted:#8B979E; --faint:#5C666C;
  --arm:#FF7A8A; --edge:#B79CFF; --none:#3FC2D6; --unk:#C9A227; --accent:#3FC2D6;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:15px; line-height:1.5;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1040px;margin:0 auto;padding:18px 20px 40px;display:flex;flex-direction:column;gap:14px}

header{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px 16px}
h1{
  font-size:17px;font-weight:600;margin:0;letter-spacing:-.01em;
}
.rule{
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;
  color:var(--accent);letter-spacing:.08em;text-transform:uppercase;
}
.count{margin-left:auto;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}

.question{
  background:var(--surface);border:1px solid var(--line);border-radius:6px;
  padding:12px 16px;font-size:16px;
}
.question b{color:var(--accent);font-weight:600}
.question i{color:var(--muted);font-style:italic}
.question p{margin:6px 0 0;font-size:13px;color:var(--muted)}

.rail{display:flex;gap:1px;height:16px;border-radius:3px;overflow:hidden;background:var(--surface);border:1px solid var(--line)}
.rail i{flex:1 1 0;background:var(--raised);cursor:pointer;transition:background .12s}
.rail i[data-v="arm"]{background:var(--arm)}
.rail i[data-v="edge"]{background:var(--edge)}
.rail i[data-v="none"]{background:var(--none)}
.rail i[data-v="unclear"]{background:var(--unk)}
.rail i.cur{outline:2px solid var(--ink);outline-offset:-2px;z-index:1}

.stage{
  position:relative;background:#000;border:1px solid var(--line);border-radius:6px;
  overflow:hidden;aspect-ratio:4/3;display:flex;align-items:center;justify-content:center;
}
.stage img{width:100%;height:100%;object-fit:contain;display:block;transition:filter .1s}
.stage.boost img{filter:brightness(2.4) contrast(1.25)}
.stage.zoom{cursor:crosshair}
.stage.zoom img{transform:scale(2.6);transform-origin:var(--ox,50%) var(--oy,50%)}
.badge{
  position:absolute;top:10px;left:10px;font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-size:11px;letter-spacing:.06em;color:var(--ink);background:#000A;
  padding:3px 8px;border-radius:3px;border:1px solid var(--line);
}
.mark{
  position:absolute;top:10px;right:10px;font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-size:11px;letter-spacing:.06em;padding:3px 8px;border-radius:3px;color:#0E1113;font-weight:600;
}

.bar{display:flex;flex-wrap:wrap;gap:8px}
button{
  font:inherit;color:var(--ink);background:var(--raised);border:1px solid var(--line);
  border-radius:5px;padding:9px 14px;cursor:pointer;display:flex;align-items:center;gap:9px;
  transition:border-color .12s,background .12s;
}
button:hover{border-color:var(--faint)}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
kbd{
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;color:var(--muted);
  border:1px solid var(--line);border-radius:3px;padding:1px 5px;background:var(--surface);
}
.v--arm{border-color:#6E2E38}.v--arm:hover{border-color:var(--arm)}
.v--edge{border-color:#4A3E77}.v--edge:hover{border-color:var(--edge)}
.v--none{border-color:#1E5560}.v--none:hover{border-color:var(--none)}
.v--unk{border-color:#5C4A12}.v--unk:hover{border-color:var(--unk)}
.spacer{flex:1}

.out{display:flex;flex-direction:column;gap:8px}
.out summary{cursor:pointer;font-size:13px;color:var(--muted)}
textarea{
  width:100%;height:150px;background:var(--surface);color:var(--ink);border:1px solid var(--line);
  border-radius:5px;padding:10px;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;
  resize:vertical;
}
.status{font-size:12px;color:var(--muted);min-height:18px;font-family:"IBM Plex Mono",ui-monospace,monospace}
.status.ok{color:var(--none)}
.status.dirty{color:var(--unk)}
.note{font-size:12px;color:var(--faint);border-top:1px solid var(--line);padding-top:12px}
.note b{color:var(--muted);font-weight:600}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

APP = r"""
(function(){
  var TILES = JSON.parse(document.getElementById("tiles").textContent);
  var state = {};
  try {
    var stored = JSON.parse(document.getElementById("verdicts").textContent || "{}");
    if (stored && typeof stored === "object") state = stored;
  } catch (e) {}
  try {
    var local = JSON.parse(localStorage.getItem("v16-verdicts") || "null");
    if (local && Object.keys(local).length >= Object.keys(state).length) state = local;
  } catch (e) {}

  var cur = 0, boost = false, zoom = false, hint = false, dirty = false;
  var LABEL = {arm:"DEUTLICHER ARM", edge:"NUR RANDSTUECK", none:"NICHTS", unclear:"UNKLAR"};
  var TINT  = {arm:"var(--arm)", edge:"var(--edge)", none:"var(--none)", unclear:"var(--unk)"};

  document.getElementById("app").innerHTML =
    '<div class="wrap">' +
      '<header>' +
        '<h1>Leere Roboter-Masken</h1>' +
        '<span class="rule">T40_RULE_V16 &sect;4</span>' +
        '<span class="count" id="count"></span>' +
      '</header>' +
      '<div class="question">' +
        'Was ist von <b>dem Roboter</b> in diesem Bild zu sehen?' +
        '<p><b>Deutlicher Arm</b> hei&szlig;t: unverwechselbare Roboterstruktur, mehr als ein Randfetzen. ' +
        '<b>Nur Randst&uuml;ck</b> hei&szlig;t: irgendwas Dunkles am Bildrand &mdash; Fingerspitze, Sliver oder Schatten. ' +
        'Ob es davon welches ist, <b>musst du nicht entscheiden</b>: das ist eine eigene Antwort, keine Ausweichantwort. ' +
        '<b>H</b> umrandet, was sich gegen&uuml;ber der ruhenden Szene ver&auml;ndert hat.</p>' +
      '</div>' +
      '<div class="rail" id="rail"></div>' +
      '<div class="stage" id="stage">' +
        '<img id="shot" alt="">' +
        '<span class="badge" id="badge"></span>' +
        '<span class="mark" id="mark"></span>' +
      '</div>' +
      '<div class="bar">' +
        '<button class="v--arm"  data-v="arm"><kbd>1</kbd> Deutlicher Arm</button>' +
        '<button class="v--edge" data-v="edge"><kbd>2</kbd> Nur Randst&uuml;ck / Schatten</button>' +
        '<button class="v--none" data-v="none"><kbd>3</kbd> Nichts</button>' +
        '<button class="v--unk"  data-v="unclear"><kbd>4</kbd> Unklar</button>' +
      '</div>' +
      '<div class="bar">' +
        '<button id="bHint"><kbd>H</kbd> Ver&auml;nderung umranden</button>' +
        '<button id="bBoost"><kbd>B</kbd> Aufhellen</button>' +
        '<button id="bZoom"><kbd>Z</kbd> Lupe</button>' +
        '<span class="spacer"></span>' +
        '<button id="bPrev"><kbd>&larr;</kbd></button>' +
        '<button id="bNext"><kbd>&rarr;</kbd></button>' +
        '<button id="bNextOpen"><kbd>N</kbd> N&auml;chstes offenes</button>' +
      '</div>' +
      '<div class="status" id="status"></div>' +
      '<div class="bar">' +
        '<button id="bSave">Speichern</button>' +
        '<button id="bCopy">JSON kopieren</button>' +
      '</div>' +
      '<details class="out">' +
        '<summary>JSON zum Kopieren (falls Speichern nicht geht)</summary>' +
        '<textarea id="json" readonly></textarea>' +
      '</details>' +
      '<div class="note">' +
        '<b>Keine Maske wird gezeigt, und das ist Absicht.</b> Wer die Antwort der Pipeline sieht, bevor er seine eigene gibt, ist kein unabh&auml;ngiger Zeuge. ' +
        'Die Umrandung sagt <i>hier hat sich etwas ver&auml;ndert</i> &mdash; nicht <i>hier ist der Roboter</i>. Der Apfel ist aus ihr herausgerechnet, weil er gemessen der gr&ouml;&szlig;te St&ouml;rer war.' +
      '</div>' +
    '</div>';

  var rail = document.getElementById("rail");
  for (var i = 0; i < TILES.length; i++) {
    var seg = document.createElement("i");
    seg.setAttribute("data-i", String(i));
    rail.appendChild(seg);
  }

  var stage=document.getElementById("stage"), shot=document.getElementById("shot"),
      badge=document.getElementById("badge"), mark=document.getElementById("mark"),
      count=document.getElementById("count"), status=document.getElementById("status"),
      json=document.getElementById("json");

  function done(){ var n=0; for (var k in state) if (state[k]) n++; return n; }
  function say(t,c){ status.textContent=t; status.className="status"+(c?" "+c:""); }

  function paint(){
    var t = TILES[cur];
    shot.src = (hint && t.hint) ? t.hint : t.image;
    badge.textContent = "Kachel " + (cur+1) + " / " + TILES.length + (hint ? "  · Veränderung" : "");
    var v = state[t.tile];
    mark.textContent = v ? LABEL[v] : "";
    mark.style.background = v ? TINT[v] : "transparent";
    count.textContent = done() + " / " + TILES.length + " beurteilt";
    var segs = rail.children;
    for (var i=0;i<segs.length;i++){
      var sv = state[TILES[i].tile];
      if (sv) segs[i].setAttribute("data-v", sv); else segs[i].removeAttribute("data-v");
      segs[i].className = (i===cur) ? "cur" : "";
    }
    json.value = JSON.stringify({rule:"T40_RULE_V16", verdicts:state}, null, 1);
  }
  function go(i){ cur = Math.max(0, Math.min(TILES.length-1, i)); paint(); }

  function set(v){
    var key = TILES[cur].tile;
    if (state[key] === v) delete state[key]; else state[key] = v;
    dirty = true;
    try { localStorage.setItem("v16-verdicts", JSON.stringify(state)); } catch (e) {}
    say(done() + " Urteile lokal gesichert, noch nicht gespeichert.", "dirty");
    if (state[key] && cur < TILES.length-1) go(cur+1); else paint();
  }
  function nextOpen(){
    for (var d=1; d<=TILES.length; d++){
      var i=(cur+d)%TILES.length;
      if (!state[TILES[i].tile]) { go(i); return; }
    }
    say("Alle " + TILES.length + " Kacheln sind beurteilt.", "ok");
  }

  document.querySelectorAll("button[data-v]").forEach(function(b){
    b.addEventListener("click", function(){ set(b.getAttribute("data-v")); });
  });
  rail.addEventListener("click", function(ev){
    var s = ev.target.closest("i[data-i]"); if (s) go(parseInt(s.getAttribute("data-i"),10));
  });
  document.getElementById("bPrev").addEventListener("click", function(){ go(cur-1); });
  document.getElementById("bNext").addEventListener("click", function(){ go(cur+1); });
  document.getElementById("bNextOpen").addEventListener("click", nextOpen);
  document.getElementById("bHint").addEventListener("click", function(){ hint=!hint; paint(); });
  document.getElementById("bBoost").addEventListener("click", function(){ boost=!boost; stage.classList.toggle("boost", boost); });
  document.getElementById("bZoom").addEventListener("click", function(){ zoom=!zoom; stage.classList.toggle("zoom", zoom); });
  stage.addEventListener("mousemove", function(ev){
    if (!zoom) return;
    var r = stage.getBoundingClientRect();
    stage.style.setProperty("--ox", ((ev.clientX-r.left)/r.width*100)+"%");
    stage.style.setProperty("--oy", ((ev.clientY-r.top)/r.height*100)+"%");
  });

  document.addEventListener("keydown", function(ev){
    if (ev.metaKey||ev.ctrlKey||ev.altKey) return;
    if (/^(INPUT|TEXTAREA)$/.test(ev.target.tagName)) return;
    var k = ev.key.toLowerCase();
    if (k==="1") set("arm");
    else if (k==="2") set("edge");
    else if (k==="3") set("none");
    else if (k==="4") set("unclear");
    else if (k==="arrowleft") go(cur-1);
    else if (k==="arrowright") go(cur+1);
    else if (k==="h") { hint=!hint; paint(); }
    else if (k==="b") { boost=!boost; stage.classList.toggle("boost", boost); }
    else if (k==="z") { zoom=!zoom; stage.classList.toggle("zoom", zoom); }
    else if (k==="n") nextOpen();
    else return;
    ev.preventDefault();
  });

  document.getElementById("bCopy").addEventListener("click", function(){
    json.select();
    var ok=false; try { ok = document.execCommand("copy"); } catch (e) {}
    if (navigator.clipboard && !ok) {
      navigator.clipboard.writeText(json.value).then(
        function(){ say("JSON in der Zwischenablage. Fuege es Claude ein.","ok"); },
        function(){ say("Kopieren ging nicht. Oeffne das Feld unten."); });
      return;
    }
    say(ok ? "JSON in der Zwischenablage. Fuege es Claude ein." : "Kopieren ging nicht. Oeffne das Feld unten.", ok?"ok":"");
  });

  document.getElementById("bSave").addEventListener("click", function(){
    say("Speichere ...");
    if (!(window.claude && window.claude.use)) { say("Speichern ist hier nicht verfuegbar. Nimm 'JSON kopieren'."); return; }
    window.claude.use("artifact").then(function(artifact){
      if (!artifact) { say("Speichern ist hier nicht verfuegbar. Nimm 'JSON kopieren'."); return; }
      var head = '<!doctype html>\n<html lang="de"><head><meta charset="utf-8">' +
        '<meta name="viewport" content="width=device-width,initial-scale=1">' +
        '<title>Leere Roboter-Masken</title>' +
        '<link rel="preconnect" href="https://fonts.googleapis.com">' +
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>' +
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap">' +
        '<style>' + document.getElementById("appstyle").textContent + '</style></head><body>' +
        '<div id="app"></div>';
      var tail =
        '<script type="application/json" id="tiles">' + document.getElementById("tiles").textContent + '<\/script>' +
        '<script type="application/json" id="verdicts">' + JSON.stringify(state) + '<\/script>' +
        '<script id="appcode">' + document.getElementById("appcode").textContent + '<\/script>' +
        '</body></html>';
      return artifact.publish(head + tail).then(function(){
        dirty = false;
        say("Gespeichert. " + done() + " von " + TILES.length + " Urteilen liegen jetzt in der Seite.", "ok");
      });
    }).catch(function(err){
      var code = (err && err.code) || "";
      say(code === "conflict"
        ? "Jemand anders hat zwischendurch gespeichert. Seite neu laden und nochmal."
        : "Speichern fehlgeschlagen (" + (code||"unbekannt") + "). Nimm 'JSON kopieren'.");
    });
  });

  window.addEventListener("beforeunload", function(ev){ if (dirty) { ev.preventDefault(); ev.returnValue=""; } });

  go(0);
  say(done() ? done()+" Urteile wiederhergestellt." : "Taste 1 / 2 / 3 / 4 urteilt und springt weiter.", done()?"ok":"");
})();
"""

PAGE = """<title>Leere Roboter-Masken</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style id="appstyle">{style}</style>
<div id="app"></div>
<script type="application/json" id="tiles">{tiles}</script>
<script type="application/json" id="verdicts">{{}}</script>
<script id="appcode">{app}</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--look", type=pathlib.Path, default=REPO_ROOT / "runs/pr08-empty-mask-look")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    sample = json.loads((args.look / "SAMPLE.json").read_text())
    payload = []
    missing_hints = 0
    for record in sample["tiles"]:
        raw = (args.look / "frames" / record["file"]).read_bytes()
        entry = {
            "tile": record["tile"],
            "image": "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii"),
        }
        # V16 §4's "what changed here" toggle. Optional on purpose: the page must still build and
        # judge without it, because it is an aid to finding and never part of the verdict.
        hint = args.look / "hints" / f"hint-{record['tile']:03d}.jpg"
        if hint.is_file():
            entry["hint"] = "data:image/jpeg;base64," + base64.b64encode(hint.read_bytes()).decode("ascii")
        else:
            missing_hints += 1
        payload.append(entry)
    if missing_hints:
        print(f"note: {missing_hints} tiles carry no hint view; run render_empty_mask_hints.py")
    payload.sort(key=lambda r: r["tile"])

    # V16 §4 carries V15 §3's prohibition forward in full. The allowed keys are listed positively
    # rather than the forbidden ones negatively, because a new leak is a key nobody thought to ban.
    allowed = {"tile", "image", "hint"}
    for record in payload:
        extra = set(record) - allowed
        if extra:
            raise SystemExit(f"tile {record['tile']} would leak {sorted(extra)}")

    out = args.out or (args.look / "page.html")
    out.write_text(PAGE.format(style=STYLE, app=APP, tiles=json.dumps(payload, separators=(",", ":"))))
    size = out.stat().st_size
    print(f"{len(payload)} tiles -> {out}  ({size / 1e6:.1f} MB)")
    if size > 16_000_000:
        raise SystemExit("page exceeds the 16 MB artifact limit")


if __name__ == "__main__":
    main()
