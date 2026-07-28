#!/usr/bin/env python3
"""
Receding horizon — the WAM closed loop as an EmAI house-style Lottie explainer.

THE ONE IDEA (wam/runtime/executor.py, FR-05): every cycle predicts a WHOLE action
chunk, but only its first ``prefix_steps`` are executed. The unexecuted tail is
discarded, not queued — the next prediction replaces it. The robot only ever sees
prefixes; no stale chunk tail survives a re-plan.

BEATS (one per stage of ClosedLoopExecutor's own docstring: observe -> predict ->
filter -> execute prefix, then the re-plan that closes the loop):

  1 OBSERVE   a fresh state + camera frames reach the policy
  2 PREDICT   one forward pass fills the whole horizon (16 bars = 16 steps)
  3 FILTER    the deterministic safety layer clips what exceeds the limits
  4 EXECUTE   only the prefix runs
  5 RE-PLAN   the tail is discarded and the loop goes back for a fresh observation

The loop wrap IS the re-plan: at frame 0 the horizon is empty again.

Regenerate:
    python3 make-receding-horizon.py
    python3 "$EMAI_LOTTIE_KIT/check_seam.py" receding-horizon.json
"""
import os
import sys

KIT = os.environ.get(
    "EMAI_LOTTIE_KIT",
    os.path.expanduser("~/develop/emai/headquarter/.claude/skills/emai-animation"),
)
if not os.path.isdir(KIT):
    sys.exit(f"emai_lottie_kit not found at {KIT!r} — set EMAI_LOTTIE_KIT")
sys.path.insert(0, KIT)

import emai_lottie_kit as kit  # noqa: E402

kit.H = 464  # taller than the 440 default: the horizon strip needs the room
from emai_lottie_kit import *  # noqa: F401,F403,E402

HERE = os.path.dirname(os.path.abspath(__file__))

# --- geometry ---------------------------------------------------------------
OBS = (122, 132)
OBS_W, OBS_H = 140, 54
POL = (468, 132)
POL_W, POL_H = 176, 86
BAND = (330, 208)
BAND_W, BAND_H = 506, 32

X0, PITCH, SLOT_W, SLOT_H = 77, 32, 26, 62  # the horizon strip
BASE = 312                                   # bars grow upward from this baseline
LIMIT = 44                                   # the safety limit, in bar units
STRIP_R = X0 + 16 * PITCH - (PITCH - SLOT_W) # 583
LANE_X, LANE_Y = 52, 386                     # the re-plan feedback lane

# one plausible motion profile; three steps overshoot the limit and get clipped
HEIGHTS = [26, 33, 39, 36, 51, 57, 54, 38, 31, 25, 21, 27, 34, 41, 42, 39]
CLIPPED = [i for i, h in enumerate(HEIGHTS) if h > LIMIT]   # -> 4, 5, 6
CLIP_T = [144, 148, 152]
PREFIX = 3                                   # ExecutorConfig.prefix_steps
PX_R = X0 + PREFIX * PITCH - (PITCH - SLOT_W)


def sx(i):
    return X0 + SLOT_W / 2 + i * PITCH


def grow_t(i):
    return 60 + 2.4 * i


def fade_t(i):
    """When step i leaves: the discarded tail first (right to left), the prefix last."""
    return 240.0 if i < PREFIX else 226.0 + (15 - i) * 0.8


def abs_layer(nm, shapes, o=None):
    """A shape layer drawn in absolute canvas coordinates (position pinned to 0,0)."""
    ks = {"p": stat([0, 0, 0]), "a": stat([0, 0, 0])}
    if o is not None:
        ks["o"] = o
    return add(shape_layer(nm, shapes, ks=ks))


# ============================================================================
# LAYER ORDER: earlier add() renders IN FRONT. Text and HUD first, surfaces last.
# ============================================================================

# --- header + narration -----------------------------------------------------
eyebrow("RECEDING HORIZON")
add(text_layer("title", "Plan a whole chunk. Execute the first steps.", 18, INK,
               (W / 2, 68), tr_=4, font="d"))

caption_track([
    ("OBSERVE — camera frames and joint state",        (2, 10, 46, 54),      INK),
    ("PREDICT — one pass fills the whole horizon",     (54, 62, 110, 118),   INK),
    ("FILTER — deterministic limits, never bypassed",  (118, 126, 166, 174), INK),
    ("EXECUTE — only the first steps run",             (174, 182, 214, 222), INK),
    ("RE-PLAN — the tail is discarded, not queued",    (220, 228, 264, 272), INK),
])

# --- persistent labels ------------------------------------------------------
add(text_layer("lbl_chunk", "ONE CHUNK — 16 STEPS, dt_s APART", 10, TEAL,
               (X0 + 2, 240), tr_=UT, j=0))
add(text_layer("lbl_limit", "SAFETY LIMIT", 9, ORANGEB, (W - 10, BASE - LIMIT + 4), tr_=UT, j=1))
add(text_layer("lbl_prefix", "EXECUTED PREFIX", 10, ORANGEB, (X0 + 2, 342), tr_=UT, j=0))
add(text_layer("lbl_drop", "DISCARDED AT THE NEXT RE-PLAN", 10, FAINT,
               (STRIP_R - 2, 342), tr_=UT, j=1))
add(text_layer("lbl_lane", "RE-PLAN", 10, FAINT, (330, LANE_Y - 10), tr_=UT))
add(text_layer("lbl_band", "SAFETY LAYER", 11, TEAL, (X0 + 18, BAND[1] + 4), tr_=UT, j=0))
add(text_layer("lbl_band2", "deterministic · never bypassed", 10.5, MUT,
               (STRIP_R - 18, BAND[1] + 4), j=1))
add(text_layer("pol_1", "state + vision in", 10, MUT, (POL[0] - POL_W / 2 + 24, 130), j=0))
add(text_layer("pol_2", "one pass -> one chunk", 10, FAINT,
               (POL[0] - POL_W / 2 + 24, 152), j=0))

# --- reticles (the active element of each beat) -----------------------------
reticle("obs", OBS[0], OBS[1], OBS_W / 2 + 4, OBS_H / 2 + 3, ORANGE, 4, 12, 42, 50)
reticle("pol", POL[0], POL[1], POL_W / 2 + 4, POL_H / 2 + 4, ORANGE, 56, 64, 106, 114)
reticle("clip", sx(5), BASE - 26, 52, 26, ORANGE, 140, 148, 164, 172)
reticle("pfx", sx(1), BASE - 16, 42, 18, ORANGE, 176, 184, 212, 220)

# --- flashes ----------------------------------------------------------------
flash("at_pol", POL[0], POL[1], ORANGE, 40, w=POL_W, h=POL_H, peak=34)
for k in range(PREFIX):
    flash("step%d" % k, sx(k), BASE - HEIGHTS[k] / 2, ORANGE, 182 + 8 * k,
          w=22, h=HEIGHTS[k], peak=55)
flash("at_obs", OBS[0], OBS[1], ORANGE, 262, w=OBS_W, h=OBS_H, peak=40)

# --- clip ticks: where the safety layer cut the step down --------------------
for k, i in enumerate(CLIPPED):
    t = CLIP_T[k]
    abs_layer("tick%d" % i,
              [group([rect(SLOT_W + 4, 3.5, 1.5, p=(sx(i), BASE - LIMIT)), fill(ORANGE)], nm="t")],
              o=env(t + 2, t + 8, fade_t(i), fade_t(i) + 12))

# --- the ceiling: drawn OVER the bars, so you see what they ran into --------
abs_layer("limit", [group([poly([(X0, BASE - LIMIT), (STRIP_R, BASE - LIMIT)]),
                           stroke(ORANGE, 1.6, o=52, dash=DASH(5, 5))], nm="lim")])

# --- the executed prefix (orange, on top of the planned bars) ---------------
for k in range(PREFIX):
    h = HEIGHTS[k]
    add(shape_layer("exec%d" % k,
                    [group([rect(22, h, 3, p=(0, -h / 2)), fill(ORANGE, 30),
                            stroke(ORANGE, 2.0, o=100)], nm="e")],
                    ks={"p": stat([sx(k), BASE, 0]), "a": stat([0, 0, 0]),
                        "o": env(178 + 8 * k, 184 + 8 * k, 240, 252)}))

# --- the planned chunk: 16 bars that grow in, three of them get clipped -----
for i, h in enumerate(HEIGHTS):
    t0, fo = grow_t(i), fade_t(i)
    keys = [(t0, [100, 0], EASE_OUT), (t0 + 12, [100, 100], LINEAR)]
    if i in CLIPPED:
        ct = CLIP_T[CLIPPED.index(i)]
        keys += [(ct, [100, 100], SETTLE), (ct + 10, [100, 100.0 * LIMIT / h], None)]
    else:
        keys[-1] = (t0 + 12, [100, 100], None)
    add(shape_layer("bar%d" % i,
                    [group([rect(22, h, 3, p=(0, -h / 2)), fill(TEAL, 24),
                            stroke(TEAL, 1.7, o=85)], nm="b")],
                    ks={"p": stat([sx(i), BASE, 0]), "a": stat([0, 0, 0]),
                        "s": A(keys),
                        "o": A([(t0, 0, EASE_OUT), (t0 + 4, 100, LINEAR),
                                (fo, 100, EASE_IN), (fo + 14, 0, None)])}))

# --- travelling pulses ------------------------------------------------------
travel("in", (OBS[0] + OBS_W / 2, OBS[1]), (POL[0] - POL_W / 2, POL[1]), ORANGE, 12, 40, size=11)
travel("down", (POL[0], POL[1] + POL_H / 2), (POL[0], BASE - SLOT_H - 4), ORANGE, 122, 144, size=11)
travel("back1", (STRIP_R, LANE_Y), (LANE_X, LANE_Y), ORANGE, 220, 246, size=10)
travel("back2", (LANE_X, LANE_Y), (LANE_X, OBS[1] + OBS_H / 2), ORANGE, 246, 262, size=10)

# --- the horizon strip: slots, baseline, limit line, prefix rule ------------
abs_layer("slots",
          [group([rect(SLOT_W, SLOT_H, 4, p=(sx(i), BASE - SLOT_H / 2)),
                  fill(NODEF, 55), stroke(SLATE, 1.1, o=45)], nm="s%d" % i) for i in range(16)]
          + [group([poly([(X0 - 5, BASE + 1), (STRIP_R + 5, BASE + 1)]),
                    stroke(SLATE, 1.6, o=85)], nm="baseline")])
abs_layer("rule", [
    group([poly([(X0, BASE + 9), (PX_R, BASE + 9)]), stroke(ORANGE, 2.6, o=78)], nm="rp"),
    group([poly([(PX_R + 7, BASE + 9), (STRIP_R, BASE + 9)]), stroke(SLATE, 2.2, o=70)], nm="rd"),
])

# --- the safety band (lit while it filters) --------------------------------
abs_layer("band_lit", [group([rect(BAND_W, BAND_H, 9, p=BAND), fill(ORANGE, 12),
                              stroke(ORANGE, 2.2, o=100)], nm="bl")],
          o=env(124, 132, 158, 168))
abs_layer("band", [group([rect(BAND_W, BAND_H, 9, p=BAND), fill(CARDBG),
                          stroke(TEAL, 1.8, o=72)], nm="b"),
                   group([rect(5, BAND_H - 10, 2, p=(X0 + 4.5, BAND[1])), fill(TEAL)], nm="eb")])

# --- the two nodes ----------------------------------------------------------
card("obs", OBS[0], OBS[1], OBS_W, OBS_H, TEAL, "Observation", sub="cameras + joint state")
anchor(POL[0], POL[1], POL_W, POL_H, header="WAM policy", accent=ORANGE)

# --- the track (marching dashes; the loop is drawn, always) -----------------
connector("in", (OBS[0] + OBS_W / 2, OBS[1]), (POL[0] - POL_W / 2, POL[1]), ORANGE, o=26)
connector("down", (POL[0], POL[1] + POL_H / 2), (POL[0], BASE - SLOT_H - 4), ORANGE, o=26)
connector("back1", (STRIP_R, LANE_Y), (LANE_X, LANE_Y), ORANGE, o=24)
connector("back2", (LANE_X, LANE_Y), (LANE_X, OBS[1] + OBS_H / 2), ORANGE, o=24)

finish(os.path.join(HERE, "receding-horizon.json"))
