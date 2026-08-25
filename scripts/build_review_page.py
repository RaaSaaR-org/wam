#!/usr/bin/env python3
"""Build the standalone review page for the two PR-08 looks that need a person.

WHY THIS EXISTS
---------------
Two open items terminate in somebody looking at pixels, and neither can be delegated to a model
without becoming the correlated observer ``runs/pr08-mask-audit/MASK_AUDIT.json`` warns about:

- ``apple_sam2.GATE_QUALIFICATION_BLOCKERS[0]`` and ``[1]`` -- 382 overlaid apple masks;
- ``T40_RULE_V13`` §3.2 -- *"whether those frames were **looked at**, and what they were"*, over the
  frames above the measured tail edge of the robot-mask area distribution.

The evidence for both already exists as contact sheets on the workstation that rendered them. The
reviewer is not always at that workstation, and 43 PNGs in a directory is not a review tool: there
is nowhere to put a verdict, and copying a directory of images around loses which sheet a judgement
belonged to. This builds one self-contained page carrying every tile and every verdict control,
which the reviewer opens anywhere.

WHAT IT DOES NOT DO
-------------------
It **records** verdicts. It discharges nothing, decides nothing and writes no bound. The tiles are
cropped out of the committed sheets and are not re-rendered here, so nothing about the underlying
measurement can change by running this: a tile is the same pixels the sheet already carried, and the
crop geometry is derived from the sheet's own dimensions and tile count rather than assumed.

The two sections ask **different questions** and therefore carry different verdict vocabularies, and
conflating them would be the whole failure. The apple section asks whether the green mask IS the
apple. The tail section asks V13 §2's question -- whether a large magenta mask is a near-camera arm
(legitimate; a bound must not fire on it) or a mask that has grounded on the table or the scene
(the failure the bound exists for).
"""

from __future__ import annotations

import argparse
import base64
import datetime
import html
import importlib.util
import io
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: ``audit_apple_masks.contact_sheet``'s layout constants. The tile SIZE is not a constant there --
#: it is the largest tile in the sheet -- so it is derived per sheet from the image and the count.
SHEET_COLS = 4
SHEET_HEADER = 28
SHEET_GAP = 6

#: What each section asks, and the only verdicts it accepts. Order is the button order.
MASK_VERDICTS = ("apple", "partial", "wrong_object", "no_mask", "undecidable")
TAIL_VERDICTS = ("arm", "table", "mixed", "undecidable")

VERDICT_LABELS = {
    "apple": ("Apfel", "Die Maske IST der Apfel.", "ok"),
    "partial": ("teilweise", "Apfel, aber nur teilweise abgedeckt.", "warn"),
    "wrong_object": ("falsches Objekt", "Die Maske sitzt auf etwas anderem: Teller, Hand, Tisch.", "bad"),
    "no_mask": ("keine Maske", "Es ist ein Apfel im Bild, aber keine Maske darauf.", "bad"),
    "undecidable": ("nicht erkennbar", "Auf diesem Bild nicht entscheidbar.", "neutral"),
    "arm": ("Arm", "Grosse Flaeche, aber es ist der Roboter nah an der Kamera. Legitim.", "ok"),
    "table": ("Tisch / Szene", "Die Maske ist auf Tischdecke, Hintergrund oder Teller gerutscht.", "bad"),
    "mixed": ("beides", "Arm UND Szene in derselben Maske.", "warn"),
}

JPEG_QUALITY = 76


class BuildError(RuntimeError):
    """Refuse loudly; a page built from the wrong sheets is worse than no page."""


# ------------------------------------------------------------------------------------------------
# the tiles
# ------------------------------------------------------------------------------------------------


def load_sheet_index(audit: dict) -> dict[int, str]:
    """``record_mask_audit_verdicts.sheet_index``, reused rather than reimplemented.

    The mapping from a frame to the sheet whose tiles include it is the one thing that must agree
    between this page and the tool that ingests its verdicts. Two implementations of it is exactly
    the drift that would attach a reviewer's judgement to a frame they never saw.
    """
    path = REPO_ROOT / "scripts" / "record_mask_audit_verdicts.py"
    spec = importlib.util.spec_from_file_location("_rmv", path)
    if spec is None or spec.loader is None:
        raise BuildError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sheet_index(audit["frames"])


def tile_geometry(size: tuple[int, int], n_tiles: int) -> tuple[int, int, int]:
    """``(tile_w, tile_h, rows)`` for a sheet, derived from its own pixels.

    ``contact_sheet`` sizes a sheet as ``cols*tw + gap*(cols+1)`` by
    ``header + rows*th + gap*(rows+1)``. Inverting that is exact, and the caption strip -- which
    differs between the apple sheets and the tail sheets -- comes out in ``th`` rather than being
    assumed.
    """
    width, height = size
    rows = (n_tiles + SHEET_COLS - 1) // SHEET_COLS
    tw, rem_w = divmod(width - SHEET_GAP * (SHEET_COLS + 1), SHEET_COLS)
    th, rem_h = divmod(height - SHEET_HEADER - SHEET_GAP * (rows + 1), rows)
    if rem_w or rem_h:
        raise BuildError(
            f"a {width}x{height} sheet with {n_tiles} tiles does not divide into "
            f"{SHEET_COLS} columns and {rows} rows under contact_sheet's own layout "
            f"(gap {SHEET_GAP}, header {SHEET_HEADER}). Refusing rather than cropping "
            "tiles at an offset a reviewer cannot see is wrong."
        )
    return tw, th, rows


def crop_tiles(sheet_path: pathlib.Path, n_tiles: int) -> list[str]:
    """Every tile of one sheet, as base64 JPEG, in the sheet's own order."""
    from PIL import Image

    image = Image.open(sheet_path)
    tw, th, _rows = tile_geometry(image.size, n_tiles)
    out: list[str] = []
    for i in range(n_tiles):
        row, col = divmod(i, SHEET_COLS)
        left = SHEET_GAP + col * (tw + SHEET_GAP)
        top = SHEET_HEADER + SHEET_GAP + row * (th + SHEET_GAP)
        tile = image.crop((left, top, left + tw, top + th)).convert("RGB")
        buf = io.BytesIO()
        tile.save(buf, "JPEG", quality=JPEG_QUALITY, optimize=True)
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


def mask_tiles(audit_dir: pathlib.Path) -> list[dict]:
    """The 382 apple-mask tiles, grouped by the sheet a reviewer scans them on."""
    audit = json.loads((audit_dir / "MASK_AUDIT.json").read_text())
    frames = audit["frames"]
    index = load_sheet_index(audit)
    by_sheet: dict[str, list[dict]] = {}
    for position, frame in enumerate(frames):
        by_sheet.setdefault(index[position], []).append(frame)

    tiles: list[dict] = []
    for sheet, sheet_frames in sorted(by_sheet.items()):
        path = audit_dir / "sheets" / f"{sheet}.png"
        if not path.is_file():
            raise BuildError(f"{path} does not exist, but the frame->sheet mapping derived it")
        for frame, jpeg in zip(sheet_frames, crop_tiles(path, len(sheet_frames))):
            tiles.append(
                {
                    "section": "mask",
                    "sheet": sheet,
                    "key": f"{frame['episode']}:{frame['frame_index']}",
                    "episode": frame["episode"],
                    "frame": frame["frame_index"],
                    "group": frame["stratum"],
                    "note": frame.get("why_sampled", ""),
                    "flags": list(frame.get("flags") or []),
                    "jpeg": jpeg,
                }
            )
    return tiles


def tail_tiles(look_dir: pathlib.Path) -> list[dict]:
    """The 48 area-tail tiles, with both fractions, because the mismatch is part of the evidence."""
    artifact = json.loads((look_dir / "TAIL_SAMPLE.json").read_text())
    by_sheet: dict[str, list[dict]] = {}
    for frame in artifact["frames"]:
        by_sheet.setdefault(frame["sheet"], []).append(frame)

    tiles: list[dict] = []
    for sheet, sheet_frames in sorted(by_sheet.items()):
        path = look_dir / "sheets" / sheet
        if not path.is_file():
            raise BuildError(f"{path} does not exist, but TAIL_SAMPLE.json names it")
        for frame, jpeg in zip(sheet_frames, crop_tiles(path, len(sheet_frames))):
            flags = ["mismatch"] if frame.get("mismatch") else []
            tiles.append(
                {
                    "section": "tail",
                    "sheet": pathlib.Path(sheet).stem,
                    "key": f"{frame['episode']}:{frame['frame_index']}",
                    "episode": frame["episode"],
                    "frame": frame["frame_index"],
                    "group": "tail",
                    "note": (
                        f"gemessen {frame['recorded_fraction']:.4f} · "
                        f"neu {frame['recomputed_fraction']:.4f}"
                    ),
                    "flags": flags,
                    "jpeg": jpeg,
                }
            )
    return tiles


# ------------------------------------------------------------------------------------------------
# the page
# ------------------------------------------------------------------------------------------------


def render_tile(tile: dict, number: int) -> str:
    verdicts = MASK_VERDICTS if tile["section"] == "mask" else TAIL_VERDICTS
    buttons = "".join(
        f'<button type="button" class="v v--{VERDICT_LABELS[v][2]}" data-verdict="{v}" '
        f'title="{html.escape(VERDICT_LABELS[v][1])}">{html.escape(VERDICT_LABELS[v][0])}</button>'
        for v in verdicts
    )
    flags = "".join(
        f'<span class="flag">{html.escape(f)}</span>' for f in tile["flags"]
    )
    return (
        f'<article class="tile" data-section="{tile["section"]}" data-sheet="{tile["sheet"]}" '
        f'data-group="{tile["group"]}" data-key="{html.escape(tile["key"])}" tabindex="0" '
        f'aria-label="{html.escape(tile["key"])}">'
        f'<img src="data:image/jpeg;base64,{tile["jpeg"]}" loading="lazy" decoding="async" '
        f'alt="{html.escape(tile["key"])}">'
        f'<div class="tile__bar"><span class="num">{number}</span>'
        f'<span class="sheet">{html.escape(tile["sheet"])}</span>{flags}</div>'
        f'<div class="tile__verdicts">{buttons}</div>'
        f"</article>"
    )


def render_page(tiles: list[dict], provenance: dict) -> str:
    mask = [t for t in tiles if t["section"] == "mask"]
    tail = [t for t in tiles if t["section"] == "tail"]
    sheets_mask = sorted({t["sheet"] for t in mask})
    sheets_tail = sorted({t["sheet"] for t in tail})

    def sheet_options(sheets: list[str]) -> str:
        return "".join(f'<option value="{s}">{s}</option>' for s in sheets)

    mask_html = "".join(render_tile(t, i + 1) for i, t in enumerate(mask))
    tail_html = "".join(render_tile(t, i + 1) for i, t in enumerate(tail))

    legend_mask = "".join(
        f'<div class="legend__row"><span class="v v--{VERDICT_LABELS[v][2]} v--static">'
        f"{html.escape(VERDICT_LABELS[v][0])}</span>"
        f"<span>{html.escape(VERDICT_LABELS[v][1])}</span></div>"
        for v in MASK_VERDICTS
    )
    legend_tail = "".join(
        f'<div class="legend__row"><span class="v v--{VERDICT_LABELS[v][2]} v--static">'
        f"{html.escape(VERDICT_LABELS[v][0])}</span>"
        f"<span>{html.escape(VERDICT_LABELS[v][1])}</span></div>"
        for v in TAIL_VERDICTS
    )

    state = {
        "schema": "wam.pr08_review_page_state/1",
        "reviewer": "",
        "verdicts": {},
        "notes": {},
        "saved_at": None,
    }

    return TEMPLATE.format(
        state=json.dumps(state),
        provenance=json.dumps(provenance, indent=1),
        n_mask=len(mask),
        n_tail=len(tail),
        mask_tiles=mask_html,
        tail_tiles=tail_html,
        sheets_mask=sheet_options(sheets_mask),
        sheets_tail=sheet_options(sheets_tail),
        legend_mask=legend_mask,
        legend_tail=legend_tail,
        built=provenance["built_utc"],
        commit=provenance["git_commit"][:12],
    )


TEMPLATE = """<title>Maskenprüfung PR-08</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --ground: #eceef1;
  --surface: #ffffff;
  --surface-2: #f5f6f8;
  --ink: #12161b;
  --muted: #5b646f;
  --line: #d3d9df;
  --accent: #12666e;
  --accent-ink: #ffffff;
  --ok: #2c7a4b;
  --warn: #9a6410;
  --bad: #ab2b21;
  --shadow: 0 1px 2px rgba(18, 22, 27, .06), 0 8px 24px rgba(18, 22, 27, .06);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground: #0e1116;
    --surface: #161a20;
    --surface-2: #1c2128;
    --ink: #e4e8ed;
    --muted: #939ea9;
    --line: #2a3038;
    --accent: #56b6bd;
    --accent-ink: #0b0e12;
    --ok: #64b884;
    --warn: #d9a341;
    --bad: #e08279;
    --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px rgba(0, 0, 0, .35);
  }}
}}
:root[data-theme="dark"] {{
  --ground: #0e1116;
  --surface: #161a20;
  --surface-2: #1c2128;
  --ink: #e4e8ed;
  --muted: #939ea9;
  --line: #2a3038;
  --accent: #56b6bd;
  --accent-ink: #0b0e12;
  --ok: #64b884;
  --warn: #d9a341;
  --bad: #e08279;
  --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px rgba(0, 0, 0, .35);
}}

* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
  font-size: 15px;
  line-height: 1.5;
}}
.mono {{ font-family: "IBM Plex Mono", ui-monospace, monospace; }}

header.top {{
  position: sticky; top: 0; z-index: 20;
  display: flex; flex-wrap: wrap; align-items: center; gap: 12px;
  padding: 10px 18px;
  background: var(--surface);
  border-bottom: 1px solid var(--line);
  box-shadow: var(--shadow);
}}
.top h1 {{
  margin: 0; font-size: 16px; font-weight: 600; letter-spacing: -.01em;
}}
.top h1 span {{ color: var(--muted); font-weight: 400; }}
.grow {{ flex: 1 1 auto; }}
.counter {{
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 13px; color: var(--muted); font-variant-numeric: tabular-nums;
}}
.counter b {{ color: var(--ink); }}

input[type="text"], select {{
  font: inherit; color: var(--ink); background: var(--surface-2);
  border: 1px solid var(--line); border-radius: 6px; padding: 6px 9px;
}}
input[type="text"]::placeholder {{ color: var(--muted); }}

button {{
  font: inherit; cursor: pointer; border-radius: 6px;
  border: 1px solid var(--line); background: var(--surface-2); color: var(--ink);
  padding: 6px 11px;
}}
button:hover {{ border-color: var(--accent); }}
button:focus-visible, .tile:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
button.primary {{ background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }}
button.primary[disabled] {{ opacity: .5; cursor: default; }}

nav.tabs {{ display: flex; gap: 4px; padding: 12px 18px 0; }}
nav.tabs button {{ border-bottom-left-radius: 0; border-bottom-right-radius: 0; }}
nav.tabs button[aria-selected="true"] {{
  background: var(--surface); border-color: var(--line); border-bottom-color: var(--surface);
  font-weight: 600;
}}

main {{ padding: 0 18px 96px; }}
section.panel {{
  background: var(--surface); border: 1px solid var(--line); border-radius: 0 8px 8px 8px;
  padding: 18px;
}}
section.panel[hidden] {{ display: none; }}

.ask {{
  display: grid; gap: 10px; grid-template-columns: minmax(0, 1fr) minmax(0, 340px);
  align-items: start; margin-bottom: 16px;
}}
@media (max-width: 780px) {{ .ask {{ grid-template-columns: minmax(0, 1fr); }} }}
.ask h2 {{ margin: 0 0 6px; font-size: 19px; letter-spacing: -.01em; text-wrap: balance; }}
.ask p {{ margin: 0 0 8px; max-width: 62ch; color: var(--muted); }}
.ask strong {{ color: var(--ink); font-weight: 600; }}
.legend {{
  background: var(--surface-2); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px;
  display: grid; gap: 6px; font-size: 13px;
}}
.legend__row {{ display: grid; grid-template-columns: 128px minmax(0, 1fr); gap: 8px; align-items: baseline; }}
.legend__row span:last-child {{ color: var(--muted); }}

.toolbar {{
  display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  padding: 10px 0 14px; border-top: 1px solid var(--line);
}}

.grid {{
  display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
}}

.tile {{
  background: var(--surface-2); border: 1px solid var(--line); border-radius: 8px;
  overflow: hidden; display: flex; flex-direction: column;
}}
.tile[hidden] {{ display: none; }}
.tile img {{ display: block; width: 100%; height: auto; background: #000; }}
.tile__bar {{
  display: flex; gap: 8px; align-items: center; padding: 5px 8px;
  font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px; color: var(--muted);
  border-top: 1px solid var(--line);
}}
.tile__bar .num {{ color: var(--ink); font-variant-numeric: tabular-nums; }}
.tile__bar .sheet {{ flex: 1 1 auto; }}
.flag {{
  background: var(--warn); color: var(--surface); border-radius: 3px; padding: 0 5px;
  font-size: 10px; text-transform: uppercase; letter-spacing: .04em;
}}
.tile__verdicts {{ display: flex; flex-wrap: wrap; gap: 4px; padding: 6px 8px 8px; }}
.v {{
  font-size: 12px; padding: 4px 8px; border-radius: 5px;
  border: 1px solid var(--line); background: var(--surface);
}}
.v--static {{ cursor: default; }}
.v--ok[aria-pressed="true"] {{ background: var(--ok); border-color: var(--ok); color: var(--surface); }}
.v--warn[aria-pressed="true"] {{ background: var(--warn); border-color: var(--warn); color: var(--surface); }}
.v--bad[aria-pressed="true"] {{ background: var(--bad); border-color: var(--bad); color: var(--surface); }}
.v--neutral[aria-pressed="true"] {{ background: var(--muted); border-color: var(--muted); color: var(--surface); }}
.tile[data-verdict] {{ border-color: var(--accent); }}
.tile[data-verdict] .tile__bar {{ color: var(--ink); }}

footer.bar {{
  position: fixed; left: 0; right: 0; bottom: 0; z-index: 30;
  display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
  padding: 10px 18px;
  background: var(--surface); border-top: 1px solid var(--line); box-shadow: var(--shadow);
}}
.status {{ font-size: 13px; color: var(--muted); }}
.status[data-tone="dirty"] {{ color: var(--warn); }}
.status[data-tone="ok"] {{ color: var(--ok); }}
.status[data-tone="bad"] {{ color: var(--bad); }}

details.prov {{ margin-top: 18px; font-size: 13px; color: var(--muted); }}
details.prov pre {{
  overflow-x: auto; background: var(--surface-2); border: 1px solid var(--line);
  border-radius: 8px; padding: 10px; font-size: 12px;
}}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>

<header class="top">
  <h1>Maskenprüfung <span>PR-08</span></h1>
  <input type="text" id="reviewer" placeholder="Wer schaut? (Name)" size="22" autocomplete="name">
  <span class="grow"></span>
  <span class="counter" id="counter"></span>
</header>

<nav class="tabs" role="tablist">
  <button role="tab" id="tab-mask" aria-controls="panel-mask" aria-selected="true">Apfelmasken ({n_mask})</button>
  <button role="tab" id="tab-tail" aria-controls="panel-tail" aria-selected="false">Flächen-Schwanz ({n_tail})</button>
</nav>

<main>
  <section class="panel" id="panel-mask" role="tabpanel" aria-labelledby="tab-mask">
    <div class="ask">
      <div>
        <h2>Ist die grüne Maske der Apfel?</h2>
        <p>Nicht: ist das ein plausibles Objekt. Der Fehler, um den es geht, ist ein
        <strong>sicherer Kasten auf dem falschen Objekt</strong> — der sieht auf jeder automatischen
        Prüfung korrekt aus und nur ein Mensch sieht ihn.</p>
        <p>Das ist die Evidenz für <span class="mono">GATE_QUALIFICATION_BLOCKERS[0]</span> und
        <span class="mono">[1]</span>. Nicht bewertete Kacheln bleiben ungezählt — Abdeckung wird
        als Bruchteil festgehalten, nicht behauptet.</p>
      </div>
      <div class="legend">{legend_mask}</div>
    </div>
    <div class="toolbar">
      <label for="filter-mask">Blatt</label>
      <select id="filter-mask"><option value="">alle</option>{sheets_mask}</select>
      <label><input type="checkbox" id="only-open-mask"> nur unbewertete</label>
      <button type="button" data-bulk="mask" data-verdict="apple">Sichtbare alle „Apfel“</button>
      <button type="button" data-clear="mask">Sichtbare zurücksetzen</button>
    </div>
    <div class="grid" id="grid-mask">{mask_tiles}</div>
  </section>

  <section class="panel" id="panel-tail" role="tabpanel" aria-labelledby="tab-tail" hidden>
    <div class="ask">
      <div>
        <h2>Ist die magenta Maske der Arm oder die Szene?</h2>
        <p>Diese 48 Frames liegen im oberen Ende der Flächenverteilung der Robotermaske.
        <strong>Ein Arm nah an der Kamera ist legitim</strong> und eine Schranke darf ihn nicht
        wegwerfen. <strong>Eine Maske, die auf Tischdecke oder Hintergrund gerutscht ist</strong>,
        ist genau der Fehler, für den die Schranke existiert.</p>
        <p>Das beantwortet <span class="mono">T40_RULE_V13</span> §3.2: ob die Frames oberhalb der
        Schranke angesehen wurden und was sie waren. Kacheln mit
        <span class="flag">mismatch</span> weichen zwischen Cluster-Messung und Neu-Rendern um mehr
        als 0.01 ab; keine davon wechselt die Seite der Schranke.</p>
      </div>
      <div class="legend">{legend_tail}</div>
    </div>
    <div class="toolbar">
      <label for="filter-tail">Blatt</label>
      <select id="filter-tail"><option value="">alle</option>{sheets_tail}</select>
      <label><input type="checkbox" id="only-open-tail"> nur unbewertete</label>
      <button type="button" data-clear="tail">Sichtbare zurücksetzen</button>
    </div>
    <div class="grid" id="grid-tail">{tail_tiles}</div>
  </section>

  <details class="prov">
    <summary>Herkunft — welche Blätter, welcher Commit</summary>
    <pre>{provenance}</pre>
    <p>Gebaut {built} aus Commit <span class="mono">{commit}</span>. Diese Seite entscheidet nichts,
    hebt keine Sperre auf und schreibt keine Schranke.</p>
  </details>
</main>

<footer class="bar">
  <button type="button" class="primary" id="save">Speichern</button>
  <button type="button" id="copy">JSON kopieren</button>
  <span class="status" id="status">Bereit.</span>
  <span class="grow"></span>
  <span class="counter" id="counter2"></span>
</footer>

<script type="application/json" id="wam-state">{state}</script>
<script>
(function () {{
  "use strict";
  var LS_KEY = "wam.pr08.review.v1";
  var stateEl = document.getElementById("wam-state");
  var state = JSON.parse(stateEl.textContent);
  if (!state.verdicts) state.verdicts = {{}};
  if (!state.notes) state.notes = {{}};

  try {{
    var local = JSON.parse(localStorage.getItem(LS_KEY) || "null");
    if (local && local.verdicts && Object.keys(local.verdicts).length >= Object.keys(state.verdicts).length) {{
      state = local;
      if (!state.notes) state.notes = {{}};
    }}
  }} catch (e) {{ /* private window, blocked storage: the page still works */ }}

  var tiles = Array.prototype.slice.call(document.querySelectorAll(".tile"));
  var byKey = {{}};
  tiles.forEach(function (t) {{
    var k = t.getAttribute("data-key");
    (byKey[k] = byKey[k] || []).push(t);
  }});
  var dirty = false;

  function statusSay(text, tone) {{
    var el = document.getElementById("status");
    el.textContent = text;
    el.setAttribute("data-tone", tone || "");
  }}

  function saveLocal() {{
    try {{ localStorage.setItem(LS_KEY, JSON.stringify(state)); }} catch (e) {{ /* fine */ }}
  }}

  function paint(key) {{
    var verdict = state.verdicts[key] || null;
    (byKey[key] || []).forEach(function (tile) {{
      if (verdict) tile.setAttribute("data-verdict", verdict);
      else tile.removeAttribute("data-verdict");
      tile.querySelectorAll(".tile__verdicts .v").forEach(function (b) {{
        b.setAttribute("aria-pressed", b.getAttribute("data-verdict") === verdict ? "true" : "false");
      }});
    }});
  }}

  function count(section) {{
    var total = 0, done = 0;
    tiles.forEach(function (t) {{
      if (t.getAttribute("data-section") !== section) return;
      total++;
      if (state.verdicts[t.getAttribute("data-key")]) done++;
    }});
    return {{ total: total, done: done }};
  }}

  function refreshCounter() {{
    var m = count("mask"), t = count("tail");
    var text = "Apfel " + m.done + "/" + m.total + "  ·  Schwanz " + t.done + "/" + t.total;
    document.getElementById("counter").innerHTML =
      "<b>" + m.done + "</b>/" + m.total + " Apfel · <b>" + t.done + "</b>/" + t.total + " Schwanz";
    document.getElementById("counter2").textContent = text;
  }}

  function setVerdict(key, verdict) {{
    if (verdict) state.verdicts[key] = verdict;
    else delete state.verdicts[key];
    paint(key);
    refreshCounter();
    dirty = true;
    saveLocal();
    statusSay("Nicht gespeichert — " + Object.keys(state.verdicts).length + " Urteile lokal.", "dirty");
    applyFilter("mask"); applyFilter("tail");
  }}

  document.addEventListener("click", function (ev) {{
    var button = ev.target.closest(".tile__verdicts .v");
    if (button) {{
      var tile = button.closest(".tile");
      var key = tile.getAttribute("data-key");
      var next = button.getAttribute("data-verdict");
      setVerdict(key, state.verdicts[key] === next ? null : next);
      return;
    }}
    var bulk = ev.target.closest("[data-bulk]");
    if (bulk) {{
      var section = bulk.getAttribute("data-bulk");
      var verdict = bulk.getAttribute("data-verdict");
      var n = 0;
      tiles.forEach(function (t) {{
        if (t.getAttribute("data-section") !== section || t.hidden) return;
        state.verdicts[t.getAttribute("data-key")] = verdict; n++;
      }});
      Object.keys(byKey).forEach(paint);
      refreshCounter(); dirty = true; saveLocal();
      statusSay(n + " sichtbare Kacheln gesetzt. Nicht gespeichert.", "dirty");
      applyFilter(section);
      return;
    }}
    var clear = ev.target.closest("[data-clear]");
    if (clear) {{
      var sec = clear.getAttribute("data-clear");
      tiles.forEach(function (t) {{
        if (t.getAttribute("data-section") !== sec || t.hidden) return;
        delete state.verdicts[t.getAttribute("data-key")];
      }});
      Object.keys(byKey).forEach(paint);
      refreshCounter(); dirty = true; saveLocal();
      statusSay("Zurückgesetzt. Nicht gespeichert.", "dirty");
      applyFilter(sec);
    }}
  }});

  document.addEventListener("keydown", function (ev) {{
    var tile = document.activeElement && document.activeElement.closest(".tile");
    if (!tile || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    var buttons = tile.querySelectorAll(".tile__verdicts .v");
    var i = "12345".indexOf(ev.key);
    if (i >= 0 && i < buttons.length) {{
      ev.preventDefault();
      buttons[i].click();
    }}
  }});

  function applyFilter(section) {{
    var sel = document.getElementById("filter-" + section);
    var open = document.getElementById("only-open-" + section);
    if (!sel || !open) return;
    var sheet = sel.value;
    tiles.forEach(function (t) {{
      if (t.getAttribute("data-section") !== section) return;
      var hide = false;
      if (sheet && t.getAttribute("data-sheet") !== sheet) hide = true;
      if (open.checked && state.verdicts[t.getAttribute("data-key")]) hide = true;
      t.hidden = hide;
    }});
  }}

  ["mask", "tail"].forEach(function (section) {{
    var sel = document.getElementById("filter-" + section);
    var open = document.getElementById("only-open-" + section);
    if (sel) sel.addEventListener("change", function () {{ applyFilter(section); }});
    if (open) open.addEventListener("change", function () {{ applyFilter(section); }});
  }});

  var tabs = [["tab-mask", "panel-mask"], ["tab-tail", "panel-tail"]];
  tabs.forEach(function (pair) {{
    document.getElementById(pair[0]).addEventListener("click", function () {{
      tabs.forEach(function (other) {{
        var selected = other[0] === pair[0];
        document.getElementById(other[0]).setAttribute("aria-selected", String(selected));
        document.getElementById(other[1]).hidden = !selected;
      }});
    }});
  }});

  var reviewer = document.getElementById("reviewer");
  reviewer.value = state.reviewer || "";
  reviewer.addEventListener("input", function () {{
    state.reviewer = reviewer.value;
    dirty = true; saveLocal();
  }});

  function publishedDocument() {{
    var clone = document.documentElement.cloneNode(true);
    clone.querySelectorAll(".tile").forEach(function (t) {{
      t.removeAttribute("data-verdict");
      t.hidden = false;
      t.removeAttribute("hidden");
      t.querySelectorAll(".tile__verdicts .v").forEach(function (b) {{
        b.setAttribute("aria-pressed", "false");
      }});
    }});
    var input = clone.querySelector("#reviewer");
    if (input) input.setAttribute("value", "");
    var payload = clone.querySelector("#wam-state");
    payload.textContent = JSON.stringify(state);
    return "<!doctype html>\\n" + clone.outerHTML;
  }}

  document.getElementById("copy").addEventListener("click", function () {{
    var text = JSON.stringify(state, null, 1);
    navigator.clipboard.writeText(text).then(function () {{
      statusSay("JSON in der Zwischenablage — an Claude schicken reicht auch.", "ok");
    }}, function () {{
      statusSay("Zwischenablage blockiert. JSON steht in der Konsole.", "bad");
      console.log(text);
    }});
  }});

  document.getElementById("save").addEventListener("click", function () {{
    var button = this;
    button.disabled = true;
    statusSay("Speichere …", "");
    state.saved_at = new Date().toISOString();
    if (typeof claude === "undefined" || !claude.use) {{
      button.disabled = false;
      statusSay("Diese Kopie kann nicht selbst speichern (lokal geoeffnet). Nimm \u201eJSON kopieren\u201c.", "bad");
      return;
    }}
    claude.use("artifact").then(function (artifact) {{
      if (!artifact) {{
        button.disabled = false;
        statusSay("Speichern hier nicht möglich. Nimm „JSON kopieren“.", "bad");
        return;
      }}
      return artifact.publish(publishedDocument()).then(function () {{
        dirty = false;
        statusSay("Gespeichert. Claude kann es jetzt lesen.", "ok");
      }});
    }}).catch(function (err) {{
      button.disabled = false;
      var code = (err && err.code) || "";
      if (code === "conflict") {{
        statusSay("Jemand anderes hat zuerst gespeichert. Seite neu laden, dann erneut.", "bad");
      }} else if (code === "not_granted" || code === "not_writer") {{
        statusSay("Kein Schreibrecht auf dieser Seite. Nimm „JSON kopieren“.", "bad");
      }} else {{
        statusSay("Speichern fehlgeschlagen (" + (code || "unbekannt") + "). Nimm „JSON kopieren“.", "bad");
      }}
    }});
  }});

  window.addEventListener("beforeunload", function (ev) {{
    if (!dirty) return;
    ev.preventDefault();
    ev.returnValue = "";
  }});

  Object.keys(byKey).forEach(paint);
  refreshCounter();
  applyFilter("mask");
  applyFilter("tail");
  if (Object.keys(state.verdicts).length) {{
    statusSay(Object.keys(state.verdicts).length + " Urteile geladen.", "");
  }}
}})();
</script>
"""


# ------------------------------------------------------------------------------------------------


def build(audit_dir: pathlib.Path, look_dir: pathlib.Path, git_commit: str) -> str:
    tiles = mask_tiles(audit_dir) + tail_tiles(look_dir)
    provenance = {
        "built_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
        "produced_by": "scripts/build_review_page.py",
        "git_commit": git_commit,
        "records": "verdicts only. Discharges no blocker, decides no bound, licenses no clip.",
        "mask_sheets": sorted({t["sheet"] for t in tiles if t["section"] == "mask"}),
        "tail_sheets": sorted({t["sheet"] for t in tiles if t["section"] == "tail"}),
        "mask_source": str(audit_dir / "MASK_AUDIT.json"),
        "tail_source": str(look_dir / "TAIL_SAMPLE.json"),
        "correlated_observer": (
            "This page exists so a PERSON looks. A model filling it in is the correlated observer "
            "MASK_AUDIT.json names, and the verdicts would be worth nothing."
        ),
    }
    return render_page(tiles, provenance)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit-dir", type=pathlib.Path, default=REPO_ROOT / "runs" / "pr08-mask-audit")
    ap.add_argument("--look-dir", type=pathlib.Path, default=REPO_ROOT / "runs" / "pr08-area-tail-look")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--git-commit", default="")
    args = ap.parse_args(argv)

    try:
        page = build(args.audit_dir, args.look_dir, args.git_commit or "unknown")
    except BuildError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
