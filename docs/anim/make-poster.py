#!/usr/bin/env python3
"""Emit the receding-horizon poster SVG from the same geometry the Lottie uses.

The poster is the no-JS / GitHub / reduced-motion fallback for
``receding-horizon.json``: it freezes the EXECUTE beat (~f200) — the whole chunk
planned, three steps clipped by the safety layer, the prefix executed, the tail
about to be dropped. Keeping it generated rather than hand-drawn is what stops it
drifting from the animation when the profile or the layout changes.

Authored at 2x the Lottie canvas (660x464 -> 1320x928) by wrapping the body in
scale(2), so every coordinate below is a Lottie coordinate.

    python3 make-poster.py
    # then, from headquarter/assets/brand/linkedin (needs cairo + the brand fonts):
    uv run python ../../../.claude/skills/emai-graphic/render.py \\
        <wam>/docs/anim/receding-horizon-poster.svg --scale 2
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "receding-horizon-poster.svg")

X0, PITCH, SLOT_W, SLOT_H = 77, 32, 26, 62
BASE, LIMIT = 312, 44
STRIP_R = X0 + 16 * PITCH - (PITCH - SLOT_W)
PREFIX = 3
PX_R = X0 + PREFIX * PITCH - (PITCH - SLOT_W)
LANE_X, LANE_Y = 52, 386
PLANNED = [26, 33, 39, 36, 51, 57, 54, 38, 31, 25, 21, 27, 34, 41, 42, 39]
CLIPPED = [i for i, h in enumerate(PLANNED) if h > LIMIT]
FILTERED = [min(h, LIMIT) for h in PLANNED]
sx = lambda i: X0 + SLOT_W / 2 + i * PITCH

ORANGE, ORANGEB, TEAL = "#FF6700", "#FF8534", "#2DD4BF"
INK, MUT, FAINT = "#F5F5F4", "#A8A29E", "#78716C"
CARD, ELEV, NODE = "#1F1F1F", "#292929", "#141414"
BORD, BORD2 = "#3D3D3D", "#2A2A2A"

b = []
a = b.append


def card(cx, cy, w, h, color, title, sub):
    """Mirror emai_lottie_kit.card()'s geometry exactly."""
    hw, hh = w / 2, h / 2
    nyo = -(hh - 11)
    a(f'<rect x="{cx-hw-4}" y="{cy-hh-4}" width="{w+8}" height="{h+8}" rx="15" fill="none" stroke="{color}" stroke-opacity="0.22"/>')
    a(f'<rect x="{cx-hw}" y="{cy-hh}" width="{w}" height="{h}" rx="12" fill="{CARD}" stroke="{color}" stroke-opacity="0.85" stroke-width="2"/>')
    a(f'<rect x="{cx-hw}" y="{cy+nyo-12}" width="{w}" height="24" rx="12" fill="{ELEV}"/>')
    a(f'<rect x="{cx-hw+1.5}" y="{cy-(h-6)/2}" width="5" height="{h-6}" rx="2" fill="{color}"/>')
    a(f'<line x1="{cx-hw+2}" y1="{cy+nyo+12}" x2="{cx+hw-2}" y2="{cy+nyo+12}" stroke="{color}" stroke-opacity="0.30"/>')
    a(f'<rect x="{cx+hw-38}" y="{cy+nyo-6.5}" width="24" height="13" rx="3" fill="{NODE}" stroke="{color}" stroke-opacity="0.7"/>')
    b1 = int(w * 0.46); b2 = int(w * 0.30)
    a(f'<rect x="{cx-hw+24}" y="{cy+5}" width="{b1}" height="6" rx="3" fill="{BORD}"/>')
    a(f'<rect x="{cx-hw+24}" y="{cy+19}" width="{b2}" height="6" rx="3" fill="{BORD2}"/>')
    a(f'<text x="{cx-hw+18}" y="{cy+nyo+5}" class="disp" font-size="13" fill="{color}">{title}</text>')
    a(f'<text x="{cx}" y="{cy+hh+18}" text-anchor="middle" class="sans" font-size="10.5" fill="{MUT}">{sub}</text>')


def conn(p0, p1):
    a(f'<line x1="{p0[0]}" y1="{p0[1]}" x2="{p1[0]}" y2="{p1[1]}" stroke="{ORANGE}" stroke-opacity="0.26" stroke-width="2" stroke-dasharray="9 7"/>')


def reticle(cx, cy, hw, hh, color, arm=15, pad=6):
    x, y = hw + pad, hh + pad
    for sxx, syy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        a(f'<path d="M{cx+sxx*(x-arm)},{cy+syy*y} L{cx+sxx*x},{cy+syy*y} L{cx+sxx*x},{cy+syy*(y-arm)}" '
          f'fill="none" stroke="{color}" stroke-opacity="0.92" stroke-width="2" stroke-linecap="round"/>')


# ---- the track (back) ------------------------------------------------------
conn((192, 132), (380, 132))
conn((468, 175), (468, 250))
conn((STRIP_R, LANE_Y), (LANE_X, LANE_Y))
conn((LANE_X, LANE_Y), (LANE_X, 159))

# ---- the horizon strip -----------------------------------------------------
for i in range(16):
    a(f'<rect x="{sx(i)-SLOT_W/2}" y="{BASE-SLOT_H}" width="{SLOT_W}" height="{SLOT_H}" rx="4" '
      f'fill="{NODE}" fill-opacity="0.55" stroke="{BORD}" stroke-opacity="0.45" stroke-width="1.1"/>')
a(f'<line x1="{X0-5}" y1="{BASE+1}" x2="{STRIP_R+5}" y2="{BASE+1}" stroke="{BORD}" stroke-opacity="0.85" stroke-width="1.6"/>')

for i, h in enumerate(FILTERED):
    col = ORANGE if i < PREFIX else TEAL
    op = "0.30" if i < PREFIX else "0.24"
    sw = "2" if i < PREFIX else "1.7"
    so = "1" if i < PREFIX else "0.85"
    a(f'<rect x="{sx(i)-11}" y="{BASE-h}" width="22" height="{h}" rx="3" fill="{col}" fill-opacity="{op}" '
      f'stroke="{col}" stroke-opacity="{so}" stroke-width="{sw}"/>')

a(f'<line x1="{X0}" y1="{BASE-LIMIT}" x2="{STRIP_R}" y2="{BASE-LIMIT}" stroke="{ORANGE}" '
  f'stroke-opacity="0.52" stroke-width="1.6" stroke-dasharray="5 5"/>')
for i in CLIPPED:
    a(f'<rect x="{sx(i)-(SLOT_W+4)/2}" y="{BASE-LIMIT-1.75}" width="{SLOT_W+4}" height="3.5" rx="1.5" fill="{ORANGE}"/>')

a(f'<line x1="{X0}" y1="{BASE+9}" x2="{PX_R}" y2="{BASE+9}" stroke="{ORANGE}" stroke-opacity="0.78" stroke-width="2.6"/>')
a(f'<line x1="{PX_R+7}" y1="{BASE+9}" x2="{STRIP_R}" y2="{BASE+9}" stroke="{BORD}" stroke-opacity="0.7" stroke-width="2.2"/>')
reticle(sx(1), BASE - 16, 42, 18, ORANGE)

# ---- the safety band -------------------------------------------------------
a(f'<rect x="{X0}" y="192" width="506" height="32" rx="9" fill="{CARD}" stroke="{TEAL}" stroke-opacity="0.72" stroke-width="1.8"/>')
a(f'<rect x="{X0+2}" y="{208-11}" width="5" height="22" rx="2" fill="{TEAL}"/>')
a(f'<text x="{X0+18}" y="212" class="sans" font-size="11" letter-spacing="0.33" fill="{TEAL}">SAFETY LAYER</text>')
a(f'<text x="{STRIP_R-18}" y="212" text-anchor="end" class="sans" font-size="10.5" fill="{MUT}">deterministic &#183; never bypassed</text>')

# ---- the two nodes ---------------------------------------------------------
card(122, 132, 140, 54, TEAL, "Observation", "cameras + joint state")
a(f'<rect x="381.5" y="94" width="5" height="76" rx="2" fill="{ORANGE}"/>')
a(f'<rect x="380" y="89" width="176" height="86" rx="16" fill="{NODE}" stroke="{ORANGE}" stroke-opacity="0.92" stroke-width="2.6" filter="url(#soft)"/>')
a(f'<text x="404" y="105" class="disp" font-size="12.5" fill="{INK}">WAM policy</text>')
a(f'<text x="404" y="130" class="sans" font-size="10" fill="{MUT}">state + vision in</text>')
a(f'<text x="404" y="152" class="sans" font-size="10" fill="{FAINT}">one pass -&gt; one chunk</text>')

# ---- labels ----------------------------------------------------------------
a(f'<text x="{X0+2}" y="240" class="sans" font-size="10" letter-spacing="0.3" fill="{TEAL}">ONE CHUNK &#8212; 16 STEPS, dt_s APART</text>')
a(f'<text x="650" y="{BASE-LIMIT+4}" text-anchor="end" class="sans" font-size="9" letter-spacing="0.27" fill="{ORANGEB}">SAFETY LIMIT</text>')
a(f'<text x="{X0+2}" y="342" class="sans" font-size="10" letter-spacing="0.3" fill="{ORANGEB}">EXECUTED PREFIX</text>')
a(f'<text x="{STRIP_R-2}" y="342" text-anchor="end" class="sans" font-size="10" letter-spacing="0.3" fill="{FAINT}">DISCARDED AT THE NEXT RE-PLAN</text>')
a(f'<text x="330" y="{LANE_Y-10}" text-anchor="middle" class="sans" font-size="10" letter-spacing="0.3" fill="{FAINT}">RE-PLAN</text>')

# ---- header + standalone caption ------------------------------------------
a(f'<text x="330" y="38" text-anchor="middle" class="sans" font-size="12.5" letter-spacing="0.375" fill="{TEAL}">RECEDING HORIZON</text>')
a(f'<text x="330" y="68" text-anchor="middle" class="disp" font-size="18" fill="{INK}">Plan a whole chunk. Execute the first steps.</text>')
a(f'<text x="330" y="434" text-anchor="middle" class="sans" font-size="15" letter-spacing="0.45" fill="{INK}">Only the prefix runs &#8212; the tail is discarded, not queued.</text>')

ARIA = ("The WAM closed loop, one cycle: a fresh observation of camera frames and joint state reaches "
        "the policy, one forward pass fills a whole horizon of sixteen action steps, the deterministic "
        "safety layer clips the three steps that exceed the limits, only the first steps of the chunk "
        "are executed, and the untouched tail is discarded when the loop goes back for a fresh "
        "observation. Poster frame of the receding-horizon animation.")

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1320 928" width="1320" height="928" role="img" aria-label="{ARIA}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0.3" y2="1">
      <stop offset="0" stop-color="#151515"/>
      <stop offset="1" stop-color="#070707"/>
    </linearGradient>
    <radialGradient id="glow" cx="0.5" cy="0" r="0.78">
      <stop offset="0" stop-color="#FF6700" stop-opacity="0.13"/>
      <stop offset="1" stop-color="#FF6700" stop-opacity="0"/>
    </radialGradient>
    <pattern id="grid" width="22" height="22" patternUnits="userSpaceOnUse">
      <circle cx="1.2" cy="1.2" r="1.2" fill="#ffffff" fill-opacity="0.03"/>
    </pattern>
    <pattern id="tape" width="26" height="26" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)">
      <rect width="13" height="26" fill="#FF6700"/>
      <rect x="13" width="13" height="26" fill="#0A0A0A"/>
    </pattern>
    <filter id="soft" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="3" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <style>
      .disp {{ font-family: 'Space Grotesk','Helvetica Neue',Arial,sans-serif; font-weight:700; }}
      .sans {{ font-family: 'Geist','Helvetica Neue',Arial,sans-serif; }}
    </style>
  </defs>

  <rect width="1320" height="928" fill="url(#bg)"/>
  <rect width="1320" height="928" fill="url(#glow)"/>
  <rect width="1320" height="928" fill="url(#grid)"/>

  <g transform="scale(2)">
    {chr(10).join("    " + s for s in b)}
  </g>

  <rect x="0" y="0" width="1320" height="16" fill="url(#tape)"/>
  <rect x="0" y="0" width="1320" height="2" fill="#FF6700"/>
  <rect x="0" y="14" width="1320" height="2" fill="#FF6700"/>
  <rect x="0" y="912" width="1320" height="16" fill="url(#tape)"/>
  <rect x="0" y="912" width="1320" height="2" fill="#FF6700"/>
  <rect x="0" y="926" width="1320" height="2" fill="#FF6700"/>
</svg>
'''
open(OUT, "w").write(svg)
print("wrote", OUT, len(svg), "bytes")
