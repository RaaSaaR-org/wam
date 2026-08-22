#!/usr/bin/env python3
"""T-040 / ``T40-TODO-01`` — build and score **C40**, the calibration set from
``docs/t040-identity-prompt-judge.md`` §3, for whatever instrument is about to fill
``runs/t040-identity-prompt/sheet.jsonl``.

    .venv/bin/python scripts/build_identity_calibration.py seed-frames --out <CAL>
    #   ... a seed pass: LOOK at the eight frames, write <CAL>/seed_observations.json ...
    .venv/bin/python scripts/build_identity_calibration.py build --out <CAL> \\
        --sheet runs/t040-identity-prompt/
    #   ... the instrument answers <CAL>/items.jsonl, blind, into <CAL>/answers.jsonl ...
    .venv/bin/python scripts/build_identity_calibration.py score --out <CAL>

WHY THIS FILE EXISTS AND WHY IT IS NOT ``build_identity_prompt_sheet.py``.
``build_identity_prompt_sheet.py`` names no judge anywhere, deliberately, and
``test_build_sheet_names_no_judge_anywhere`` fails the build if one creeps in. This script does not
name a judge either — it writes items and it scores answers — but it *is* the harness a judge is
plugged into, and keeping it separate keeps that tripwire meaningful.

THE FAILURE THIS IS DESIGNED AGAINST. T-041's run came back VOID on G0b because its VLM judge
answered the literal string ``"NO"`` to all 80 items — a constant classifier — and its calibration
reported ``10/20``, which reads like partial credit. So this scorer reports **five numbers and
never an aggregate**, because a constant classifier's signature is a zero in one class and an
aggregate hides zeros. ``degenerate`` runs four synthetic answer vectors (constant match, constant
mismatch, constant unsure, a uniform coin) through the same scorer and shows that each fails; if
that command ever prints a PASS, the scorer is broken and nothing it says about a real instrument
means anything.

WHERE THE LABELS COME FROM, AND WHY THEY ARE NOT OPINIONS. §3.1's circularity: "does the committed
prompt describe episode X" is the unknown being measured, so a corpus frame cannot be a calibration
positive on the strength of anyone's belief that it matches. The way out is a tiny seed established
once by looking, then amplified mechanically. Every item below is a seed frame plus a
transformation whose effect on each clause is known **by construction** — a recoloured cloth, a
falsified clause, a region made unjudgeable. The item's label is a fact about the transformation.

THE SEED PASS IS THE ONE HUMAN STEP, AND IN THIS RUN IT WAS NOT HUMAN. ``seed_observations.json``
records ``established_by``; the artifact carries it through to the scores, and
``docs`` §3.1's five minutes of human attention is what it stands in for. This is a stated weakness
and it is not hidden: it is written into the meta, not into a footnote. What carries the
calibration is the mechanical amplification — the seed only anchors which prompt is TRUE of which
frame.

THE PROMPT IS A TEMPLATE WITH FIVE AXIS SLOTS, AND THE COMMITTED STRING IS ONE FILLING OF IT.
That is what makes a prompt-side negative possible at all: falsifying "the cloth is black" means
substituting the *table* slot and leaving the other four bytes-identical, and it means substituting
**every** mention of that axis — the committed prompt names the apple twice and the plate three
times, and a prompt that says "green apple" in one sentence and "red/yellow apple" in the next is
self-contradictory rather than falsified on one clause.

WHAT A PASS ON C40 DOES NOT BUY (docs §3.6, in its own words). C40's items are manufactured: one
clause falsified hard, or nothing falsified at all. The real forty are natural frames where the
honest answer may be "the cloth is black but this one looks charcoal". **Passing C40 is necessary
and never sufficient.**

EXIT STATUS
-----------
0   the command did what it says. For ``score``: all five floors held.
2   fatal: a refusal — nothing usable was written.
3   ``score``: at least one floor failed. The scores are still written; a failed calibration
    honestly recorded is the outcome T-041 got right.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

CALIBRATION_SCHEMA = "wam.identity_prompt_calibration/1"
TODO_ID = "T40-TODO-01-identity-prompt-provenance"
WRITEUP = "docs/t040-identity-prompt-judge.md"

#: The three tokens, and the six axes. Imported at use sites from the sheet harness so there is one
#: vocabulary; repeated here only as a fallback for a test that imports this module alone.
VERDICT_VALUES = ("match", "mismatch", "unsure")
MISMATCH_AXES = ("apple", "table", "background", "lighting", "plate", "other")

#: Seed for choosing the seven non-provenance seed frames. Not one of the sheet's (40001) and not
#: one of the generation seeds (7001-7015): one number, one meaning.
DEFAULT_SEED_FRAME_SEED = 40002
#: Seed for the interleave of the forty C40 items with the forty real rows.
DEFAULT_SHUFFLE_SEED = 40003
#: Attempt 2's interleave seed. A re-run of the same eighty items has to be re-shuffled, or the
#: block each item falls into — and therefore which of the four judges sees it — is inherited from
#: the run whose failure prompted the re-run.
DEFAULT_SHUFFLE_SEED_ATTEMPT2 = 40005
#: docs §3.1. Eight frames, one of which is the clip the caption was written from.
SEED_FRAME_COUNT = 8
PROVENANCE_EPISODE = "episode_000135"

#: docs §3.2 — the class sizes. Changing one of these changes the pass rule's denominators, so
#: they are named once and asserted against the built set.
N_POSITIVE = 15
N_NEGATIVE = 15
N_PROBE = 10

#: docs §3.4, the five floors. Pre-registered here, before any answer existed, and read by both
#: ``score`` and ``degenerate`` so the synthetic check exercises the identical rule.
FLOOR_POSITIVE = 14          # of N_POSITIVE, answered `match`
FLOOR_NEGATIVE_TOKEN = 14    # of N_NEGATIVE, answered `mismatch`
FLOOR_NEGATIVE_AXIS = 13     # of N_NEGATIVE, named the required axis
FLOOR_PROBE = 8              # of N_PROBE, answered `unsure`
CEIL_LEAKAGE = 1             # `unsure` answers among the 30 decidable items

#: Axis credit, pre-registered because "named the required axis" has a permissive reading and a
#: strict one and they are not the same instrument. STRICT-ENOUGH: the required axis must be named
#: AND the answer must name at most two axes in total. The permissive reading (required axis
#: appears anywhere in the list) is scored and reported too, but it is not the floor, because a
#: classifier that answers `mismatch` with all six axes on every item would score 15/15 on it —
#: which is the T-041 shape wearing a different hat. Under the registered rule a uniform guesser
#: draws one of 6 singletons or 15 pairs, 6 of those 21 contain the required axis, so an axis is a
#: 2/7 guess and a negative (token AND axis) is a 2/21 one.
MAX_AXES_FOR_CREDIT = 2

#: ------------------------------------------------------------------------------------------
#: THE TWO PROBE SETS. docs §4 asks for a clause-bearing region made "genuinely unjudgeable".
#: ``composite`` is attempt 1's answer: the corpus's own Dex3 gripper, cut out and pasted over the
#: apple. ``natural`` is attempt 2's: an UNMODIFIED corpus frame in which the robot's own hand is
#: already over the apple. Both are kept, because attempt 1 is a recorded run and a script that can
#: no longer rebuild its items cannot be audited.
PROBE_SET_COMPOSITE = "composite"
PROBE_SET_NATURAL = "natural"
PROBE_SETS = (PROBE_SET_COMPOSITE, PROBE_SET_NATURAL)

#: A frame is ELIGIBLE as a natural probe when the strict apple mask has all but vanished and what
#: is standing in front of it is the robot. Both halves matter: an apple that is merely small is a
#: hard positive, and an apple that is absent from the SCENE (carried off, never placed) is a
#: `mismatch`, not an abstention — the point of the ring test is that the thing hiding the fruit is
#: in the picture and is visibly a hand.
NATURAL_PROBE_MAX_WARM_PX = 700
#: The census threshold is deliberately looser than the eligibility one, so the artifact records
#: how much of the corpus came CLOSE to qualifying and not only what was chosen.
NATURAL_PROBE_CENSUS_PX = 1200
#: Of the ring of pixels immediately around whatever apple is still visible, this fraction must be
#: foreground — i.e. differ from the episode's own median background — for the frame to count as
#: "the hand is over it" rather than "the fruit is elsewhere".
NATURAL_PROBE_MIN_RING_FOREGROUND = 0.50

EXIT_OK = 0
EXIT_FATAL = 2
EXIT_FLOOR_FAILED = 3


class CalibrationError(RuntimeError):
    """A refusal. Every one of these means an item, a key or a score would claim something untrue."""


# --------------------------------------------------------------------------------------------
# the prompt, as a template over five axis slots
# --------------------------------------------------------------------------------------------

#: The committed string is exactly ``render_prompt(COMMITTED_SLOTS)`` and ``check_template`` proves
#: it against ``configs/transfer25/styles.toml`` before a single item is written. A template that
#: does not reproduce the committed prompt byte-for-byte would silently judge the corpus against a
#: paraphrase of arm C's prompt, and every verdict would be about a string nobody committed.
PROMPT_TEMPLATE = (
    "A {apple_long}, on {table_long}. {plate_long}. {lighting_long}. "
    "Contrast between {background_short} and {plate_short} and {apple_short}. "
    "Live-action video, realistic. {plate_short_cap} keeps its own appearance. "
    "Scene geometry, camera framing and the robot are unchanged."
)

SLOT_NAMES = ("apple_long", "table_long", "plate_long", "lighting_long",
              "background_short", "plate_short", "apple_short", "plate_short_cap")

#: Which slots each axis owns. An axis substitution rewrites ALL of its slots, because the prompt
#: names the apple twice and the plate three times and a half-substituted prompt is incoherent
#: rather than falsified on one clause.
AXIS_SLOTS: dict[str, tuple[str, ...]] = {
    "apple": ("apple_long", "apple_short"),
    "table": ("table_long",),
    "background": ("background_short",),
    "lighting": ("lighting_long",),
    "plate": ("plate_long", "plate_short", "plate_short_cap"),
}

COMMITTED_SLOTS: dict[str, str] = {
    "apple_long": "red and yellow apple with a glossy surface and a visible stem",
    "table_long": "a black cloth covering a flat surface",
    "plate_long": "A white, round plate",
    "lighting_long": ("Even, bright, top-down lighting with minimal shadows; the lighting "
                      "highlights the objects without creating harsh contrasts"),
    "background_short": "the black background",
    "plate_short": "the white plate",
    "apple_short": "red/yellow apple",
    "plate_short_cap": "The white plate",
}

#: docs §3.3's prompt-side column. Each entry is a filling of that axis's slots that is FALSE of
#: every frame in this corpus by a wide margin — a green apple, a checked tablecloth. The margin is
#: the point: a prompt-side negative that turns on whether charcoal counts as black would be
#: measuring the thing under measurement, not the instrument.
FALSIFIED_SLOTS: dict[str, dict[str, str]] = {
    "apple": {
        "apple_long": "green apple with a matte, unpolished skin and no stem",
        "apple_short": "green apple",
    },
    "table": {"table_long": "a blue-and-white checked tablecloth covering a flat surface"},
    "background": {"background_short": "a bright, saturated blue background"},
    "lighting": {"lighting_long": ("Dim, low side lighting casting long hard shadows; the lighting "
                                   "throws the objects into harsh contrast")},
    "plate": {
        "plate_long": "A small dark square tray",
        "plate_short": "the dark square tray",
        "plate_short_cap": "The dark square tray",
    },
}


def render_prompt(slots: dict[str, str]) -> str:
    missing = [s for s in SLOT_NAMES if s not in slots]
    if missing:
        raise CalibrationError(f"prompt slots missing: {missing}")
    return PROMPT_TEMPLATE.format(**slots)


def substitute(slots: dict[str, str], axis: str, replacement: dict[str, str]) -> dict[str, str]:
    """A copy of ``slots`` with every slot this axis owns replaced. Refuses a partial rewrite.

    A prompt that says "green apple" in the first sentence and "red/yellow apple" in the fourth is
    not a one-clause falsification; it is an incoherent prompt, and an instrument that answers
    `mismatch` on it has told us nothing about whether it looked at the picture.
    """
    owned = set(AXIS_SLOTS[axis])
    if set(replacement) != owned:
        raise CalibrationError(
            f"axis {axis!r} owns slots {sorted(owned)} but the replacement names "
            f"{sorted(replacement)}. A partially substituted prompt contradicts itself instead of "
            "falsifying one clause."
        )
    out = dict(slots)
    out.update(replacement)
    return out


def check_template(committed_prompt: str) -> None:
    """``render_prompt(COMMITTED_SLOTS)`` must be the committed string, byte for byte."""
    rendered = render_prompt(COMMITTED_SLOTS)
    if rendered != committed_prompt:
        raise CalibrationError(
            "the prompt template does not reproduce the committed [identity_style].prompt.\n"
            f"       template renders: {rendered!r}\n"
            f"       committed:        {committed_prompt!r}\n"
            "       Every calibration item is a substitution into this template, so a template "
            "that is a paraphrase of arm C's prompt would calibrate an instrument against a string "
            "nobody committed."
        )


# --------------------------------------------------------------------------------------------
# seed frames
# --------------------------------------------------------------------------------------------


def select_seed_episodes(all_ids: list[str], exclude: set[str], k: int, seed: int) -> list[str]:
    """``k`` episode ids spanning the corpus, none of them a measured one. Stratified, like the sheet.

    docs §3.1 asks for seed frames drawn "from strata the 40-row sheet did not sample". The sheet
    draws one episode from each of forty contiguous strata, so no stratum is unsampled and that
    sentence cannot be honoured literally; what it protects is honoured instead — **no calibration
    frame is also a measured frame** — by excluding the forty sampled episodes and then stratifying
    what is left. The deviation is recorded in the build meta rather than quietly resolved here.
    """
    pool = sorted(e for e in all_ids if e not in exclude)
    if k > len(pool):
        raise CalibrationError(f"cannot draw {k} seed episodes from a pool of {len(pool)}.")
    rng = random.Random(seed)
    n = len(pool)
    return [pool[rng.randrange(i * n // k, (i + 1) * n // k)] for i in range(k)]


def _load_corpus(manifest: Path) -> dict[str, dict]:
    """The manifest, read with the DRIVER's own loader. Imported late, on purpose.

    ``restyle_transfer25`` is what `97` restyles from, and a second reader here would be a second
    chance to disagree with the driver about what the corpus is. It is imported inside the function
    rather than at module scope so that ``score`` and the scorer's tests never touch the generator
    stack — the scorer is the part that has to keep working when the driver is mid-edit.
    """
    from restyle_transfer25 import DriverError, load_manifest  # noqa: PLC0415
    try:
        return load_manifest(manifest)
    except DriverError as exc:
        raise CalibrationError(str(exc)) from exc


def _extract(video: Path, index: int, out: Path, ffmpeg: str) -> None:
    from build_identity_prompt_sheet import SheetError, extract_frame  # noqa: PLC0415
    try:
        extract_frame(video, index, out, ffmpeg)
    except SheetError as exc:
        raise CalibrationError(str(exc)) from exc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_seed_frames(args: argparse.Namespace) -> int:
    from build_identity_prompt_sheet import frame_index, read_identity_style  # noqa: PLC0415

    identity = read_identity_style(args.styles)
    check_template(identity["prompt"])

    episodes = _load_corpus(args.manifest)
    measured = {json.loads(ln)["episode"]
                for ln in (args.sheet / "sheet.jsonl").read_text().splitlines() if ln.strip()}
    if PROVENANCE_EPISODE not in episodes:
        raise CalibrationError(
            f"{PROVENANCE_EPISODE} is not in {args.manifest}. It is the only frame in the corpus "
            "whose match is asserted by provenance rather than by opinion (the caption was written "
            "from it), so a seed set without it rests entirely on this session's own eyes."
        )
    exclude = set(measured) | {PROVENANCE_EPISODE}
    drawn = select_seed_episodes(list(episodes), exclude, SEED_FRAME_COUNT - 1, args.seed_frame_seed)
    seeds = sorted([PROVENANCE_EPISODE, *drawn])

    frames_dir = args.out / "seed_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for ep in seeds:
        entry = episodes[ep]
        video = args.manifest.parent / str(entry["video"])
        n_frames = int(entry["frames"])
        idx = frame_index(n_frames, args.frame_fraction)
        out = frames_dir / f"{ep}.png"
        _extract(video, idx, out, args.ffmpeg)
        records.append({
            "episode": ep, "video": str(video), "n_frames": n_frames,
            "frame_index": idx, "frame_fraction": args.frame_fraction,
            "frame": str(out.resolve()), "frame_sha256": sha256_file(out),
            "role": "provenance" if ep == PROVENANCE_EPISODE else "stratified",
        })

    meta = {
        "schema": CALIBRATION_SCHEMA,
        "step": "seed-frames",
        "writeup": WRITEUP,
        "todo": TODO_ID,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": str(args.manifest),
        "n_episodes_in_manifest": len(episodes),
        "seed_frame_seed": args.seed_frame_seed,
        "frame_fraction": args.frame_fraction,
        "measured_episodes_excluded": sorted(measured),
        "seed_episodes": seeds,
        "seed_frames": records,
        "committed_prompt": identity["prompt"],
        "note": (
            "docs §3.1 asks for a seed established by a person looking at the frames and writing "
            "down, per clause, what is true. Write that into seed_observations.json next to this "
            "file, then run `build`."
        ),
    }
    (args.out / "seed_frames_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for r in records:
        print(f"{r['episode']}  frame {r['frame_index']}/{r['n_frames']}  {r['role']}",
              file=sys.stderr)
    print(f"wrote {frames_dir} and {args.out / 'seed_frames_meta.json'}", file=sys.stderr)
    return EXIT_OK


# --------------------------------------------------------------------------------------------
# regions
#
# Every mask below is derived from the frame's own statistics rather than from a hard-coded box.
# A hard-coded box is the failure this avoids: the apple sits in a different place in every
# episode, and a mutation that misses its target leaves the pixels unchanged while the key goes on
# calling the item a negative — a mislabelled item, which is worse than no item. `build` asserts
# each mutation actually landed and REFUSES the whole set if one did not.
# --------------------------------------------------------------------------------------------


def _np():
    import numpy as np  # noqa: PLC0415
    return np


def load_rgb(path: Path):
    from PIL import Image  # noqa: PLC0415
    np = _np()
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32)


def save_rgb(arr, path: Path) -> None:
    from PIL import Image  # noqa: PLC0415
    np = _np()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "RGB").save(path)


def luma(a):
    return a.mean(2)


def saturation(a):
    np = _np()
    mx, mn = a.max(2), a.min(2)
    return np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)


def cloth_level(a) -> int:
    """The modal luminance of the frame — the cloth, because the cloth is most of every frame."""
    np = _np()
    hist, _ = np.histogram(luma(a), bins=256, range=(0, 256))
    return int(hist.argmax())


def _grow(seed, allowed, max_iter: int = 4000):
    """The connected component of ``allowed`` reachable from ``seed``, 4-connected.

    Iterated dilation rather than a label pass: no scipy on this workstation, and a Python BFS over
    300 000 pixels is slow enough to matter when it runs for forty items. Converges in about the
    region's diameter in iterations, each one a handful of whole-array boolean ops.
    """
    np = _np()
    cur = seed & allowed
    for _ in range(max_iter):
        nxt = cur.copy()
        nxt[1:, :] |= cur[:-1, :]
        nxt[:-1, :] |= cur[1:, :]
        nxt[:, 1:] |= cur[:, :-1]
        nxt[:, :-1] |= cur[:, 1:]
        nxt &= allowed
        if np.array_equal(nxt, cur):
            return cur
        cur = nxt
    raise CalibrationError("region growing did not converge — refusing to mutate a partial mask.")


def _seed_pixel(mask):
    """One pixel that is actually IN ``mask``, near its centre of gravity.

    ``(median(ys), median(xs))`` is not a mask pixel for anything non-convex — for the gripper it
    lands in the gap between two fingers — and growing from a pixel outside the mask returns the
    empty set, which then dies in ``.min()`` on an empty array three frames away from the cause.
    """
    np = _np()
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        raise CalibrationError("empty mask has no seed pixel.")
    cy, cx = float(np.median(ys)), float(np.median(xs))
    i = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
    seed = np.zeros_like(mask)
    seed[ys[i], xs[i]] = True
    return seed


def _largest_component(allowed, seeds_from, max_components: int = 40):
    """The biggest connected component of ``allowed`` that contains a ``seeds_from`` pixel.

    The gripper, the forearm, the cable and a fold shadow are all "not cloth"; taking whichever one
    happens to be found first would sometimes cut a 200-pixel cable out and call it the occluder,
    and the coverage assertion in ``probe_occlude`` would then refuse every probe with no hint of
    why.
    """
    np = _np()
    remaining = seeds_from.copy()
    best = None
    for _ in range(max_components):
        if not remaining.any():
            break
        comp = _grow(_seed_pixel(remaining), allowed)
        if best is None or comp.sum() > best.sum():
            best = comp
        remaining &= ~comp
    if best is None:
        raise CalibrationError("no connected component found.")
    return best


def _dilate(mask, r: int):
    np = _np()
    cur = mask
    for _ in range(r):
        nxt = cur.copy()
        nxt[1:, :] |= cur[:-1, :]
        nxt[:-1, :] |= cur[1:, :]
        nxt[:, 1:] |= cur[:, :-1]
        nxt[:, :-1] |= cur[:, 1:]
        cur = nxt
    return cur


def close_mask(mask, r: int):
    """Dilate, fill, erode — so a C-shaped silhouette becomes a solid one.

    The Dex3 gripper at mid-clip is a C: two fingers around the apple, open on one side. Hole
    filling does not close an open C, so an occluder cut straight from it has a gap the apple shows
    through, and ``probe_occlude`` refuses every probe built from it. Closing bridges the opening
    first, which is what turns "the fingers are around the apple" into "the hand is over it".
    """
    grown = fill_holes(_dilate(mask, r))
    return ~_dilate(~grown, r)


def fill_holes(mask):
    """``mask`` with interior holes closed, by flooding the complement inward from the border.

    The Dex3 gripper's silhouette has gaps between the fingers, and an occluder pasted through
    those gaps shows the ORIGINAL apple between them — an 'occlusion' probe whose clause is still
    perfectly judgeable, which is the one thing docs §4 says a probe must not be.
    """
    np = _np()
    outside = np.zeros_like(mask)
    outside[0, :] = outside[-1, :] = True
    outside[:, 0] = outside[:, -1] = True
    outside = _grow(outside & ~mask, ~mask)
    return ~outside


def apple_mask(a):
    """The apple: warm, saturated, and one connected blob. Raises if there is no such blob.

    Warm-and-saturated is the whole discriminator and it is enough here: the cloth and the plate
    are neutral to within two counts (see the seed observations) and the robot is black or bare
    metal, so the only saturated warm thing in any of these frames is the fruit.
    """
    np = _np()
    warm = (a[:, :, 0] > 90) & (a[:, :, 0] - a[:, :, 2] > 50) & (saturation(a) > 0.35)
    if warm.sum() < 1500:
        raise CalibrationError(f"no apple-sized warm region found ({int(warm.sum())} px).")
    return fill_holes(_grow(_seed_pixel(warm), warm))


def plate_mask(a):
    """The plate: the largest bright, near-neutral blob that is not the top band."""
    np = _np()
    bright = (luma(a) > 185) & (saturation(a) < 0.16)
    bright[:45, :] = False                     # the pale strip behind the cloth is not the plate
    if bright.sum() < 8000:
        raise CalibrationError(f"no plate-sized bright region found ({int(bright.sum())} px).")
    return fill_holes(_grow(_seed_pixel(bright), bright))


def cloth_mask(a):
    """The cloth: near the modal luminance, near-neutral, and neither the apple nor the plate.

    Adaptive rather than a fixed luminance window, because the eight seed frames sit at modal
    luminance 86–93 and a window that fits one of them clips another.
    """
    np = _np()
    lvl = cloth_level(a)
    # +/-45 rather than a tighter window: the cloth's fold shadows run 25-40 counts below the modal
    # level, and a tighter window leaves them un-tinted — a ragged dark island in an otherwise blue
    # cloth, which is a TAMPERING cue, and the null-perturbation positives exist precisely so that
    # tampering cannot be the thing an instrument discriminates on. The window still floors above
    # the robot: the Dex3 gripper sits 55-70 counts below the cloth.
    m = (np.abs(luma(a) - lvl) < 45) & (saturation(a) < 0.25)
    m &= ~apple_mask(a)
    m &= ~plate_mask(a)
    return m


def top_band_mask(a):
    """The pale strip of the surface behind the cloth, along the top edge of the frame."""
    np = _np()
    lvl = cloth_level(a)
    band = (luma(a) > lvl + 40) & (saturation(a) < 0.22)
    band[40:, :] = False
    if band.sum() < 800:
        raise CalibrationError(f"no pale top band found ({int(band.sum())} px).")
    return band


# --------------------------------------------------------------------------------------------
# mutations — image side
#
# Each returns (mutated array, assertions). `build` checks the assertions and refuses the entire
# set on a single failure: an item whose mutation silently did nothing is labelled `mismatch` in
# the key while showing the judge an untouched frame, and it would count against an instrument
# that answered it correctly.
# --------------------------------------------------------------------------------------------


def mutate_apple_green(a):
    """Swap R and G inside the apple. (227,140,62) becomes (140,227,62): a vivid green apple.

    A channel swap rather than a flat repaint because it carries the fruit's own shading and its
    specular highlight through unchanged — only the hue moves. A flat repaint would also destroy
    'glossy', and the item is supposed to falsify one clause, not three.
    """
    np = _np()
    m = apple_mask(a)
    out = a.copy()
    px = out[m]
    out[m] = np.stack([px[:, 1], px[:, 0], px[:, 2]], axis=1)
    before, after = a[m].mean(0), out[m].mean(0)
    return out, {
        "apple_px": int(m.sum()),
        "mean_rgb_before": [round(float(v), 1) for v in before],
        "mean_rgb_after": [round(float(v), 1) for v in after],
        "green_now_dominant": bool(after[1] > after[0] + 30),
    }


def _tint(a, mask, rgb_gain):
    """Recolour ``mask`` to a hue while holding its luminance, so only the colour clause moves.

    Holding luminance is the point: a tint that also darkens the cloth would falsify the LIGHTING
    clause as well, and docs §3.3 wants each negative to falsify one clause and leave the rest
    standing.
    """
    np = _np()
    gain = np.asarray(rgb_gain, dtype=np.float32)
    gain = gain / gain.mean()
    out = a.copy()
    px = out[mask]
    # Clipped HERE rather than at save time, so the assertions recorded in the key describe the
    # pixels that reach the instrument. An unclipped mean of 412 in a uint8 image is a number about
    # an array nobody will ever see.
    out[mask] = np.clip(px.mean(1, keepdims=True) * gain[None, :], 0, 255)
    return out


def mutate_table_tint(a, rgb_gain, name: str):
    np = _np()
    m = cloth_mask(a)
    if m.sum() < 60000:
        raise CalibrationError(f"cloth mask is only {int(m.sum())} px — refusing to tint it.")
    out = _tint(a, m, rgb_gain)
    return out, {
        "cloth_px": int(m.sum()),
        "tint": name,
        "mean_rgb_before": [round(float(v), 1) for v in a[m].mean(0)],
        "mean_rgb_after": [round(float(v), 1) for v in out[m].mean(0)],
        "mean_luma_shift": round(float(luma(out)[m].mean() - luma(a)[m].mean()), 2),
        "saturation_after": round(float(saturation(out)[m].mean()), 3),
    }


def mutate_background_band(a):
    """Recolour the pale strip behind the cloth to a vivid blue.

    THE WEAKEST ITEM IN THE SET, and it is labelled so in the key. In this top-down framing the
    backdrop IS the cloth, so the only part of the background clause that can be falsified without
    also falsifying the table clause is its parenthetical about the strip. An instrument that
    reads the clause as a whole and shrugs at the strip will miss this one; that is a real property
    of the axis pair and not a defect in the instrument, which is why the axes' non-separability is
    written into the seed record instead of being hidden behind a cleverer mutation.
    """
    m = top_band_mask(a)
    out = _tint(a, m, (0.35, 0.75, 2.2))
    return out, {
        "band_px": int(m.sum()),
        "mean_rgb_before": [round(float(v), 1) for v in a[m].mean(0)],
        "mean_rgb_after": [round(float(v), 1) for v in out[m].mean(0)],
    }


def mutate_lighting_hard(a):
    """A steep left-to-right falloff plus a hard-edged cast shadow, at unchanged mean exposure.

    Mean-preserving on purpose. A plain gamma crush would darken the cloth toward black and quietly
    falsify the table clause too — and worse, it would make the *committed* prompt's 'black cloth'
    truer, which is the one direction a calibration item must never move the thing under
    measurement.
    """
    np = _np()
    h, w, _ = a.shape
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    ramp = 1.0 + 0.50 * x                       # 0.50x at the left edge, 1.50x at the right
    # A hard-edged wedge, one pixel of feather, running down-right: the signature of a low, hard
    # key light, and the thing 'minimal shadows / no harsh contrasts' denies.
    wedge = (y * 1.35 - x * 0.9 + 0.15) > 0
    shade = np.where(wedge, 0.42, 1.0).astype(np.float32)
    field = ramp * shade
    field = field / (field.mean())              # hold the frame's mean exposure
    out = np.clip(a * field[:, :, None], 0, 255)
    lo = float(luma(out)[:, : w // 3].mean())
    hi = float(luma(out)[:, -w // 3:].mean())
    return out, {
        "left_third_luma": round(lo, 1),
        "right_third_luma": round(hi, 1),
        "left_right_ratio": round(hi / max(lo, 1e-6), 2),
        "mean_luma_before": round(float(luma(a).mean()), 1),
        "mean_luma_after": round(float(luma(out).mean()), 1),
    }


def mutate_plate_dark(a):
    """Take the white plate down to a dark charcoal dish, keeping its shape and its shading."""
    np = _np()
    from PIL import Image, ImageFilter  # noqa: PLC0415
    m = plate_mask(a)
    lum = a.mean(2, keepdims=True)
    dark = np.clip((lum - 150.0) * 0.35 + 46.0, 4, 255) * np.asarray(
        [1.0, 0.98, 1.02], dtype=np.float32)[None, None, :]
    # Feathered rather than hard-edged: a binary mask leaves a one-pixel dotted rim of the original
    # white plate, and that rim is a TAMPERING cue rather than a colour cue. The whole point of the
    # null-perturbation positives is that an instrument must not be able to pass by spotting edits.
    alpha = np.asarray(Image.fromarray((m * 255).astype(np.uint8), "L")
                       .filter(ImageFilter.GaussianBlur(2.5)), dtype=np.float32)[:, :, None] / 255.0
    out = a * (1.0 - alpha) + dark * alpha
    return out, {
        "plate_px": int(m.sum()),
        "feather_sigma": 2.5,
        "mean_luma_before": round(float(luma(a)[m].mean()), 1),
        "mean_luma_after": round(float(luma(out)[m].mean()), 1),
    }


# --------------------------------------------------------------------------------------------
# null perturbations — positives that have been through an image processing step
#
# docs §3.2: these equalise "has been processed" across the positive and the negative class. An
# instrument that passes by spotting tampering rather than by reading the clause is the most
# likely way a synthetic negative set gets gamed, and it is invisible in the score if the positives
# are all pristine.
# --------------------------------------------------------------------------------------------


def null_reencode(a):
    from io import BytesIO  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    np = _np()
    buf = BytesIO()
    Image.fromarray(np.clip(np.rint(a), 0, 255).astype(np.uint8), "RGB").save(
        buf, format="JPEG", quality=92)
    buf.seek(0)
    out = np.asarray(Image.open(buf).convert("RGB")).astype(np.float32)
    return out, {"kind": "jpeg-q92 round trip",
                 "max_abs_delta": int(np.abs(out - a).max()),
                 "mean_abs_delta": round(float(np.abs(out - a).mean()), 3)}


def null_exposure(a, factor: float):
    np = _np()
    out = np.clip(a * factor, 0, 255)
    return out, {"kind": f"exposure x{factor}",
                 "mean_luma_before": round(float(luma(a).mean()), 2),
                 "mean_luma_after": round(float(luma(out).mean()), 2)}


# --------------------------------------------------------------------------------------------
# abstention probes
# --------------------------------------------------------------------------------------------


def build_occluder(frame):
    """The Dex3 gripper and forearm, cut out of a mid-clip frame of the corpus itself.

    Cut from the corpus rather than drawn, because a drawn black blob is not what the rubric's
    canonical abstention describes ("the hand occludes the apple") and an instrument that answers
    `mismatch` on an obviously synthetic blob has told us nothing. Holes are filled: see
    ``fill_holes`` for what an occluder with gaps in it does to a probe.
    """
    np = _np()
    lvl = cloth_level(frame)
    not_cloth = np.abs(luma(frame) - lvl) > 26
    not_cloth[:60, :] = False                   # the pale top strip is not the robot
    warm = (frame[:, :, 0] - frame[:, :, 2] > 50) & (saturation(frame) > 0.35)
    bright = (luma(frame) > 185) & (saturation(frame) < 0.16)
    allowed = not_cloth & ~warm & ~bright
    dark_seeds = allowed & (luma(frame) < lvl - 40)
    if dark_seeds.sum() < 2000:
        raise CalibrationError("no robot-sized dark region in the occluder source frame.")
    # The gripper alone, closed into a solid silhouette. The forearm is deliberately left out: it
    # is a long thin bright limb, and scaling ITS bounding box up until it covers a plate produces
    # an occluder that is mostly empty where the target is.
    m = close_mask(_largest_component(dark_seeds, dark_seeds), 26)
    ys, xs = np.nonzero(m)
    box = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
    patch = frame[box[0]:box[1], box[2]:box[3]].copy()
    pm = m[box[0]:box[1], box[2]:box[3]].copy()
    # Holes that filling closed have the SOURCE frame's apple showing through them. Paint them the
    # gripper's own median colour so the occluder is opaque everywhere it claims to be.
    # Near-neutral as well as dark: a few hundred pixels of the fruit's shaded underside sit below
    # the luminance cut without meeting the warm test, and kept as "gripper" they show through the
    # finished occluder as a red streak — the colour the probe exists to hide, in the one place a
    # careful observer would look.
    dark = (luma(patch) < lvl - 40) & (saturation(patch) < 0.28)
    if dark.sum() < 500:
        raise CalibrationError("occluder patch has too few gripper pixels to fill its holes from.")
    fillrgb = np.median(patch[dark], axis=0)
    # EVERYTHING that is not gripper, inside the patch box and out, not just the filled holes. The
    # box is resized with LANCZOS before it is pasted, so any source pixel adjacent to the
    # silhouette bleeds a few pixels across the boundary — and in the source frame the pixels
    # adjacent to the gripper ARE THE APPLE. That leaves a thin orange fringe around the occluder
    # in the finished probe: a hint of exactly the colour the probe exists to hide.
    patch[~dark] = fillrgb
    return patch, pm


def probe_occlude(a, target_mask, occluder, occ_mask, margin: int = 10):
    """Paste the occluder over ``target_mask`` and REFUSE unless every target pixel is covered.

    The refusal is the whole design. docs §4: "a probe that a careful person can still answer is
    not a probe, it is a hard positive, and it will make an honest instrument look broken." Full
    coverage is a checkable fact about the paste, so the item's `unsure` label is a fact about the
    transformation in exactly the way a negative's label is.
    """
    np = _np()
    from PIL import Image  # noqa: PLC0415
    h, w, _ = a.shape
    ys, xs = np.nonzero(target_mask)
    ty0, ty1, tx0, tx1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    need_h, need_w = (ty1 - ty0) + 2 * margin, (tx1 - tx0) + 2 * margin
    oh, ow = occ_mask.shape
    base_scale = max(need_h / oh, need_w / ow, 0.35)
    cy, cx = (ty0 + ty1) // 2, (tx0 + tx1) // 2

    # Bounding-box-to-bounding-box centring is not coverage: the gripper's silhouette is irregular,
    # so a box that contains the apple can still leave a quarter of it showing through a notch. The
    # placement is therefore SEARCHED — offsets first, then a larger occluder — and the item is
    # only accepted at coverage 1.0. Cheap: a few hundred boolean ANDs.
    best: tuple[float, Any, Any] | None = None
    for grow in (1.0, 1.25, 1.6, 2.0):
        scale = base_scale * grow
        sh, sw = max(int(round(oh * scale)), need_h), max(int(round(ow * scale)), need_w)
        patch = np.asarray(Image.fromarray(np.clip(occluder, 0, 255).astype(np.uint8), "RGB")
                           .resize((sw, sh), Image.LANCZOS)).astype(np.float32)
        pm = np.asarray(Image.fromarray((occ_mask * 255).astype(np.uint8), "L")
                        .resize((sw, sh), Image.NEAREST)) > 127
        for dy in range(-60, 61, 3):
            for dx in range(-60, 61, 3):
                y0 = int(np.clip(cy - sh // 2 + dy, 0, max(h - sh, 0)))
                x0 = int(np.clip(cx - sw // 2 + dx, 0, max(w - sw, 0)))
                ph, pw = min(sh, h - y0), min(sw, w - x0)
                placed = np.zeros((h, w), dtype=bool)
                placed[y0:y0 + ph, x0:x0 + pw] = pm[:ph, :pw]
                cov = float((target_mask & placed).sum()) / float(target_mask.sum())
                if best is None or cov > best[0]:
                    best = (cov, (y0, x0, ph, pw, scale), (patch, pm))
                if cov >= 1.0:
                    break
            if best and best[0] >= 1.0:
                break
        if best and best[0] >= 1.0:
            break

    cov, (y0, x0, ph, pw, scale), (patch, pm) = best     # type: ignore[misc]
    out = a.copy()
    region = out[y0:y0 + ph, x0:x0 + pw]
    sub = pm[:ph, :pw]
    region[sub] = patch[:ph, :pw][sub]
    out[y0:y0 + ph, x0:x0 + pw] = region
    placed = np.zeros((h, w), dtype=bool)
    placed[y0:y0 + ph, x0:x0 + pw] = sub

    # The last few per cent. The gripper's own silhouette gets to ~97 % of the apple and the rest is
    # a hairline rim of fruit at the edge — enough to read the colour off, which is the one thing a
    # probe must not leave available. Those pixels are painted the occluder's own median colour, so
    # the occluder bulges slightly rather than the probe being abandoned. The NATURAL coverage is
    # what gets asserted and recorded; the patch is reported separately and is refused if it has to
    # do real work.
    natural = cov
    residue = _dilate(target_mask, 6) & ~placed
    if residue.any():
        out[residue] = np.median(patch[sub], axis=0) if sub.any() else out[placed].mean(0)
        placed = placed | residue
    return out, {
        "target_px": int(target_mask.sum()),
        "coverage": round(float((target_mask & placed).sum()) / float(target_mask.sum()), 4),
        "natural_coverage": round(natural, 4),
        "patched_px": int(residue.sum()),
        "occluder_scale": round(scale, 3),
        "paste_origin": [int(y0), int(x0)],
    }


# --------------------------------------------------------------------------------------------
# natural abstention probes — found in the corpus, not built out of it
#
# ATTEMPT 1's probes composited the gripper over the apple and 8 of 10 were answered `mismatch`.
# Three of those eight named the axis `other` and said in their notes that the region was a
# "smeared, blocky black artifact" that "breaks the realistic live-action clause" — which is a
# defensible answer to a manufactured occlusion and tells us nothing about abstention. So the
# probes are rebuilt out of frames the corpus already contains, where the robot's own hand is over
# the fruit and not one pixel has been touched.
#
# WHETHER SUCH FRAMES EXIST IS A MEASUREMENT, NOT AN ASSUMPTION, and ``probe-scan`` is that
# measurement over all 402 episodes. Its census is written out whatever it finds, because "the
# corpus does not contain ten of these" is itself a result and it is the one thing an eye-picked
# probe set would hide.
# --------------------------------------------------------------------------------------------


def _decode_gray(video: Path, ffmpeg: str):
    import subprocess  # noqa: PLC0415
    np = _np()
    cmd = [ffmpeg, "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    a = np.frombuffer(raw, dtype=np.uint8)
    n = a.size // (480 * 640)
    return a[: n * 480 * 640].reshape(n, 480, 640).astype(np.float32)


def _decode_rgb(video: Path, ffmpeg: str):
    import subprocess  # noqa: PLC0415
    np = _np()
    cmd = [ffmpeg, "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    a = np.frombuffer(raw, dtype=np.uint8)
    n = a.size // (480 * 640 * 3)
    return a[: n * 480 * 640 * 3].reshape(n, 480, 640, 3).astype(np.int16)


def scan_episode_for_hidden_apple(args_tuple):
    """Per-frame apple visibility for one episode, plus the ring test on the frames that qualify.

    Runs in a worker process, so it takes and returns plain data and imports numpy itself.
    """
    ep, video, ffmpeg = args_tuple
    np = _np()
    try:
        f = _decode_rgb(Path(video), ffmpeg)
    except Exception as exc:                                  # noqa: BLE001
        return ep, {"error": f"{type(exc).__name__}: {exc}"}
    r, b = f[:, :, :, 0], f[:, :, :, 2]
    mx, mn = f.max(3), f.min(3)
    sat_ = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0.0)
    # The same discriminator ``apple_mask`` uses, applied to a whole episode at once.
    warm = (r > 90) & (r - b > 50) & (sat_ > 0.35)
    warm_n = warm.reshape(warm.shape[0], -1).sum(1)
    med = float(np.median(warm_n))
    gray = f.mean(3)
    # The episode's own median background, from a thin sample: everything that differs from it is
    # the robot (or the fruit it is carrying), which is what the ring test needs.
    bg = np.median(gray[:: max(len(gray) // 40, 1)], axis=0)
    rows = []
    for i in np.nonzero(warm_n < NATURAL_PROBE_CENSUS_PX)[0]:
        m = warm[i]
        ring = _dilate(m, 8) & ~m
        fgm = np.abs(gray[i] - bg) > 28
        ring_fg = float((ring & fgm).sum()) / float(max(ring.sum(), 1))
        ys, xs = np.nonzero(m)
        rows.append({
            "frame_index": int(i),
            "apple_warm_px": int(warm_n[i]),
            "apple_warm_ratio": round(float(warm_n[i] / max(med, 1.0)), 4),
            "ring_foreground_fraction": round(ring_fg, 4),
            "warm_touches_border": bool(len(xs) and (xs.min() < 3 or xs.max() > 636
                                                     or ys.min() < 3 or ys.max() > 476)),
            "eligible": bool(warm_n[i] <= NATURAL_PROBE_MAX_WARM_PX
                             and ring_fg >= NATURAL_PROBE_MIN_RING_FOREGROUND),
        })
    return ep, {"n_frames": int(f.shape[0]), "median_apple_warm_px": med,
                "min_apple_warm_px": int(warm_n.min()),
                "n_below_census": int(len(rows)),
                "n_eligible": int(sum(1 for r_ in rows if r_["eligible"])),
                "frames": rows}


def farthest_point_pick(frames, k: int):
    """``k`` of ``frames``, chosen to be as visually unlike each other as the pool allows.

    The eligible frames of one occlusion event are consecutive, so taking the k lowest apple counts
    would take k copies of one instant. Farthest-point sampling on the frames themselves at least
    spends the diversity the pool has; it cannot manufacture diversity the corpus does not contain,
    and the build meta says so rather than letting the spread imply independence.
    """
    np = _np()
    if len(frames) < k:
        raise CalibrationError(
            f"only {len(frames)} eligible probe frame(s) for {k} probes. docs §4 will not be "
            "served by repeating one: the pass rule's denominator is a count and a probe class of "
            "a different size is scored against a rule nobody registered.")
    vecs = np.stack([f["thumb"] for f in frames]).astype(np.float32)
    first = int(np.argmin([f["apple_warm_px"] for f in frames]))
    chosen = [first]
    d = np.linalg.norm(vecs - vecs[first], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(vecs - vecs[nxt], axis=1))
    return [frames[i] for i in sorted(chosen, key=lambda i: frames[i]["frame_index"])]


def cmd_probe_scan(args: argparse.Namespace) -> int:
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415
    np = _np()

    episodes = _load_corpus(args.manifest)
    measured = {json.loads(ln)["episode"]
                for ln in (args.sheet / "sheet.jsonl").read_text().splitlines() if ln.strip()}
    # A measured episode cannot supply a calibration item — docs §3.1's rule, and the reason the
    # seed draw excludes the forty as well.
    pool = [(ep, str(args.manifest.parent / str(e["video"])), args.ffmpeg)
            for ep, e in episodes.items() if ep not in measured]

    census: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for n, (ep, rec) in enumerate(ex.map(scan_episode_for_hidden_apple, pool), 1):
            census[ep] = rec
            if n % 50 == 0:
                print(f"  scanned {n}/{len(pool)} episodes", file=sys.stderr)

    errors = {k: v["error"] for k, v in census.items() if "error" in v}
    if errors:
        raise CalibrationError(f"{len(errors)} episode(s) failed to decode: {sorted(errors)[:5]}")

    eligible: list[dict[str, Any]] = []
    for ep, rec in sorted(census.items()):
        for row in rec["frames"]:
            if row["eligible"]:
                eligible.append({"episode": ep, **row})

    frames_dir = args.out / "probe_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for cand in eligible:
        tmp = frames_dir / f"_cand_{cand['episode']}_f{cand['frame_index']:04d}.png"
        _extract(args.manifest.parent / str(episodes[cand["episode"]]["video"]),
                 cand["frame_index"], tmp, args.ffmpeg)
        a = load_rgb(tmp)
        cand["thumb"] = a[::8, ::8].mean(2).reshape(-1).tolist()
        cand["_tmp"] = tmp

    picked = farthest_point_pick(eligible, N_PROBE)
    records = []
    for i, cand in enumerate(picked):
        out = frames_dir / f"probe{i:02d}_{cand['episode']}_f{cand['frame_index']:04d}.png"
        shutil.copyfile(cand["_tmp"], out)
        records.append({
            "index": i, "episode": cand["episode"], "frame_index": cand["frame_index"],
            "frame": str(out.resolve()), "frame_sha256": sha256_file(out),
            "apple_warm_px": cand["apple_warm_px"],
            "apple_warm_ratio": cand["apple_warm_ratio"],
            "episode_median_apple_warm_px": census[cand["episode"]]["median_apple_warm_px"],
            "ring_foreground_fraction": cand["ring_foreground_fraction"],
            "warm_touches_border": cand["warm_touches_border"],
        })
    for cand in eligible:
        cand["_tmp"].unlink(missing_ok=True)
        cand.pop("thumb", None)
        cand.pop("_tmp", None)

    total_frames = sum(v["n_frames"] for v in census.values())
    meta = {
        "schema": CALIBRATION_SCHEMA,
        "step": "probe-scan",
        "writeup": WRITEUP,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": str(args.manifest),
        "rule": {
            "census_apple_warm_px": NATURAL_PROBE_CENSUS_PX,
            "eligible_apple_warm_px": NATURAL_PROBE_MAX_WARM_PX,
            "eligible_min_ring_foreground": NATURAL_PROBE_MIN_RING_FOREGROUND,
            "measured_episodes_excluded": sorted(measured),
        },
        "corpus": {
            "episodes_scanned": len(census),
            "frames_scanned": total_frames,
            "frames_below_census_threshold": sum(v["n_below_census"] for v in census.values()),
            "frames_eligible": sum(v["n_eligible"] for v in census.values()),
            "episodes_with_any_eligible_frame": sorted(
                ep for ep, v in census.items() if v["n_eligible"]),
        },
        "eligible_frames": eligible,
        "picked": records,
        "picked_note": (
            "Chosen by farthest-point sampling over the eligible pool, not by eye, and not by "
            "taking the ten smallest apples — which in a pool drawn from one continuous occlusion "
            "would be ten copies of one instant."
        ),
        "per_episode": {ep: {k: v for k, v in rec.items() if k != "frames"}
                        for ep, rec in sorted(census.items())},
    }
    (args.out / "probe_census.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"scanned {len(census)} episodes / {total_frames} frames", file=sys.stderr)
    print(f"  below census threshold ({NATURAL_PROBE_CENSUS_PX} px): "
          f"{meta['corpus']['frames_below_census_threshold']}", file=sys.stderr)
    print(f"  eligible: {meta['corpus']['frames_eligible']} frames in "
          f"{meta['corpus']['episodes_with_any_eligible_frame']}", file=sys.stderr)
    for r in records:
        print(f"  picked {r['episode']} f{r['frame_index']:4d}  apple {r['apple_warm_px']:5d} px "
              f"({r['apple_warm_ratio']:.3f} of median)  ring_fg "
              f"{r['ring_foreground_fraction']:.2f}", file=sys.stderr)
    print(f"wrote {args.out / 'probe_census.json'} and {frames_dir}", file=sys.stderr)
    print("NOW LOOK AT EVERY PICKED FRAME and write probe_observations.json next to it.",
          file=sys.stderr)
    del np
    return EXIT_OK


# --------------------------------------------------------------------------------------------
# the scorer — five numbers, never an aggregate
# --------------------------------------------------------------------------------------------


def _norm_axes(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [p for p in value.replace(",", " ").split() if p]
    return [str(v).strip().lower() for v in value if str(v).strip()]


def score_answers(key: dict[str, dict], answers: dict[str, dict]) -> dict[str, Any]:
    """Score one instrument's answers against the key. Returns five floors and their evidence.

    NO AGGREGATE IS RETURNED, and that is the lesson T-041 paid for. Its calibration reported
    ``10/20`` for a classifier that had answered the literal string "NO" to every item, and 10/20
    reads like partial credit. A constant classifier's signature is a ZERO IN ONE CLASS, and only
    per-class reporting makes a zero visible.
    """
    missing = sorted(set(key) - set(answers))
    if missing:
        raise CalibrationError(
            f"{len(missing)} calibration item(s) have no answer: {', '.join(missing[:12])}"
            f"{' …' if len(missing) > 12 else ''}\n"
            "       A calibration scored over the items an instrument happened to answer is a "
            "calibration of a self-selected subset, and the items it skipped are the ones it "
            "found hard. Score all of them or none."
        )

    per_item: list[dict[str, Any]] = []
    for item_id, k in sorted(key.items()):
        a = answers[item_id]
        token = str(a.get("verdict") or "").strip().lower()
        axes = _norm_axes(a.get("mismatched_axes"))
        required = k.get("required_axis")
        permitted = set(k.get("permitted_axes") or ([required] if required else []))
        token_ok = token == k["required_verdict"]
        axis_ok = bool(
            required
            and token_ok
            and required in axes
            and len(axes) <= MAX_AXES_FOR_CREDIT
            and set(axes) <= permitted
        )
        per_item.append({
            "item_id": item_id, "class": k["class"], "mutation": k["mutation"],
            "side": k.get("side"), "episode": k["episode"],
            "required_verdict": k["required_verdict"], "required_axis": required,
            "permitted_axes": sorted(permitted),
            "answered_verdict": token, "answered_axes": axes,
            "token_correct": token_ok,
            "axis_correct": axis_ok,
            "axis_correct_permissive": bool(required and token_ok and required in axes),
            "legal_token": token in VERDICT_VALUES,
        })

    def sel(cls: str) -> list[dict]:
        return [r for r in per_item if r["class"] == cls]

    pos, neg, prb = sel("positive"), sel("negative"), sel("probe")
    for cls, rows, want in (("positive", pos, N_POSITIVE), ("negative", neg, N_NEGATIVE),
                            ("probe", prb, N_PROBE)):
        if len(rows) != want:
            raise CalibrationError(
                f"the key holds {len(rows)} {cls} items but the pass rule is written over {want}. "
                "The floors are counts, not rates, so a set of a different size is scored against "
                "a rule nobody registered."
            )

    positives_match = sum(1 for r in pos if r["token_correct"])
    negative_tokens = sum(1 for r in neg if r["token_correct"])
    negative_axes = sum(1 for r in neg if r["axis_correct"])
    negative_axes_permissive = sum(1 for r in neg if r["axis_correct_permissive"])
    probes_unsure = sum(1 for r in prb if r["token_correct"])
    leakage = sum(1 for r in pos + neg if r["answered_verdict"] == "unsure")

    floors = [
        {"name": "positives answered `match`", "value": positives_match,
         "of": N_POSITIVE, "floor": FLOOR_POSITIVE, "direction": ">=",
         "passed": positives_match >= FLOOR_POSITIVE},
        {"name": "negatives answered `mismatch`", "value": negative_tokens,
         "of": N_NEGATIVE, "floor": FLOOR_NEGATIVE_TOKEN, "direction": ">=",
         "passed": negative_tokens >= FLOOR_NEGATIVE_TOKEN},
        {"name": "negatives naming the required axis", "value": negative_axes,
         "of": N_NEGATIVE, "floor": FLOOR_NEGATIVE_AXIS, "direction": ">=",
         "passed": negative_axes >= FLOOR_NEGATIVE_AXIS},
        {"name": "abstention probes answered `unsure`", "value": probes_unsure,
         "of": N_PROBE, "floor": FLOOR_PROBE, "direction": ">=",
         "passed": probes_unsure >= FLOOR_PROBE},
        {"name": "`unsure` leaked into the decidable items", "value": leakage,
         "of": N_POSITIVE + N_NEGATIVE, "floor": CEIL_LEAKAGE, "direction": "<=",
         "passed": leakage <= CEIL_LEAKAGE},
    ]

    by_side: dict[str, dict[str, int]] = {}
    for r in neg:
        b = by_side.setdefault(r["side"] or "?", {"n": 0, "token": 0, "axis": 0})
        b["n"] += 1
        b["token"] += int(r["token_correct"])
        b["axis"] += int(r["axis_correct"])
    by_axis: dict[str, dict[str, int]] = {}
    for r in neg:
        b = by_axis.setdefault(r["required_axis"], {"n": 0, "token": 0, "axis": 0})
        b["n"] += 1
        b["token"] += int(r["token_correct"])
        b["axis"] += int(r["axis_correct"])

    return {
        "floors": floors,
        "passed": all(f["passed"] for f in floors),
        "counts": {
            "positives_match": positives_match,
            "negative_tokens": negative_tokens,
            "negative_axes": negative_axes,
            "negative_axes_permissive": negative_axes_permissive,
            "probes_unsure": probes_unsure,
            "abstention_leakage": leakage,
        },
        "negatives_by_side": by_side,
        "negatives_by_axis": by_axis,
        "illegal_tokens": sorted({r["answered_verdict"] for r in per_item
                                  if not r["legal_token"]}),
        "answer_token_histogram": {
            t: sum(1 for r in per_item if r["answered_verdict"] == t)
            for t in sorted({r["answered_verdict"] for r in per_item})
        },
        "per_item": per_item,
        "rule": {
            "positives_floor": FLOOR_POSITIVE, "negative_token_floor": FLOOR_NEGATIVE_TOKEN,
            "negative_axis_floor": FLOOR_NEGATIVE_AXIS, "probe_floor": FLOOR_PROBE,
            "leakage_ceiling": CEIL_LEAKAGE, "max_axes_for_axis_credit": MAX_AXES_FOR_CREDIT,
            "note": (
                "Axis credit needs the required axis named, at most "
                f"{MAX_AXES_FOR_CREDIT} axes named in total, and every named axis inside the "
                "item's `permitted_axes`. The permissive count (required axis anywhere in the "
                "list) is reported beside it and is NOT the floor: an instrument that answered "
                "`mismatch` with all six axes on every item would score 15/15 on the permissive "
                "reading, which is the T-041 constant classifier in a better hat."
            ),
        },
    }


def _binom_at_least(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def _binom_at_most(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def coin_probabilities() -> dict[str, float]:
    """What a uniform guesser's chance of clearing each floor actually is, under THIS rule.

    docs §3.5 computes these for "uniform over three tokens, uniform over six axes". The registered
    axis rule here is not that one — it admits a second axis if the item's mutation genuinely moved
    it — so the arithmetic is redone rather than quoted. A guesser drawing a nonempty axis set of
    size <= 2 from six axes draws one of 6 + 15 = 21 sets, 6 of which contain the required axis.
    """
    p_axis_given_mismatch = 6 / 21
    return {
        "p_positive_correct": 1 / 3,
        "p_negative_token_correct": 1 / 3,
        "p_negative_token_and_axis_correct": (1 / 3) * p_axis_given_mismatch,
        "p_probe_correct": 1 / 3,
        "P(>=%d of %d positives)" % (FLOOR_POSITIVE, N_POSITIVE):
            _binom_at_least(FLOOR_POSITIVE, N_POSITIVE, 1 / 3),
        "P(>=%d of %d negative tokens)" % (FLOOR_NEGATIVE_TOKEN, N_NEGATIVE):
            _binom_at_least(FLOOR_NEGATIVE_TOKEN, N_NEGATIVE, 1 / 3),
        "P(>=%d of %d negative axes)" % (FLOOR_NEGATIVE_AXIS, N_NEGATIVE):
            _binom_at_least(FLOOR_NEGATIVE_AXIS, N_NEGATIVE, (1 / 3) * p_axis_given_mismatch),
        "P(>=%d of %d probes)" % (FLOOR_PROBE, N_PROBE):
            _binom_at_least(FLOOR_PROBE, N_PROBE, 1 / 3),
        "P(<=%d unsure in %d decidable)" % (CEIL_LEAKAGE, N_POSITIVE + N_NEGATIVE):
            _binom_at_most(CEIL_LEAKAGE, N_POSITIVE + N_NEGATIVE, 1 / 3),
    }


def degenerate_vectors(key: dict[str, dict], seed: int = 40004) -> dict[str, dict[str, dict]]:
    """The four answer vectors that must all FAIL, built without looking at any item's class.

    If any of these passes, the scorer is broken and nothing it says about a real instrument means
    anything. T-041's judge was the second of them, spelled "NO".
    """
    rng = random.Random(seed)
    ids = sorted(key)
    out: dict[str, dict[str, dict]] = {
        "constant_match": {i: {"verdict": "match", "mismatched_axes": []} for i in ids},
        "constant_mismatch_one_axis": {
            i: {"verdict": "mismatch", "mismatched_axes": ["table"]} for i in ids},
        "constant_unsure": {i: {"verdict": "unsure", "mismatched_axes": []} for i in ids},
    }
    coin: dict[str, dict] = {}
    for i in ids:
        t = rng.choice(list(VERDICT_VALUES))
        coin[i] = {"verdict": t,
                   "mismatched_axes": [rng.choice(list(MISMATCH_AXES))] if t == "mismatch" else []}
    out["uniform_coin_flip"] = coin
    # Not one of the four the task names, and included anyway: the constant classifier that games
    # the PERMISSIVE axis reading by naming every axis at once. It is the reason the registered
    # floor is the strict reading.
    out["constant_mismatch_all_axes"] = {
        i: {"verdict": "mismatch", "mismatched_axes": list(MISMATCH_AXES)} for i in ids}
    return out


# --------------------------------------------------------------------------------------------
# the forty items, as a declared plan
#
# Written out rather than generated by a loop so that the whole set can be READ: which seed frame
# each item is built from, which side of docs §3.3 each negative is on, and which axis it is
# required to name. A plan that has to be executed to be understood is a plan nobody audits.
# --------------------------------------------------------------------------------------------

#: (class, seed index, operation, required axis). Seed indices are into the SORTED seed episode
#: list, so the plan is stable as long as the seed draw is.
ITEM_PLAN_DECIDABLE: tuple[tuple[str, int, str, str | None], ...] = (
    # ---- positives: the eight seed frames untouched ----------------------------------------
    ("positive", 0, "none", None),
    ("positive", 1, "none", None),
    ("positive", 2, "none", None),
    ("positive", 3, "none", None),
    ("positive", 4, "none", None),
    ("positive", 5, "none", None),
    ("positive", 6, "none", None),
    ("positive", 7, "none", None),
    # ---- positives: seven null perturbations, so "processed" is not a class cue -------------
    ("positive", 0, "reencode", None),
    ("positive", 1, "exposure:1.03", None),
    ("positive", 2, "exposure:0.97", None),
    ("positive", 3, "frame:+1", None),
    ("positive", 4, "frame:-1", None),
    ("positive", 5, "reencode", None),
    ("positive", 6, "exposure:1.03", None),
    # ---- negatives: three per axis, mixed across the two mutation sides ---------------------
    ("negative", 0, "img:apple_green", "apple"),
    ("negative", 1, "img:apple_green", "apple"),
    ("negative", 2, "prompt:apple", "apple"),
    ("negative", 3, "img:table_blue", "table"),
    ("negative", 4, "img:table_beige", "table"),
    ("negative", 5, "prompt:table", "table"),
    ("negative", 6, "img:background_band", "background"),
    ("negative", 7, "prompt:background", "background"),
    ("negative", 0, "prompt:background", "background"),
    ("negative", 1, "img:lighting_hard", "lighting"),
    ("negative", 2, "prompt:lighting", "lighting"),
    ("negative", 3, "prompt:lighting", "lighting"),
    ("negative", 4, "img:plate_dark", "plate"),
    ("negative", 5, "img:plate_dark", "plate"),
    ("negative", 6, "prompt:plate", "plate"),
)

#: ATTEMPT 1's probes: the corpus's Dex3 gripper, cut out and composited over the apple.
PROBE_PLAN_COMPOSITE: tuple[tuple[str, int, str, str | None], ...] = (
    ("probe", 0, "probe:apple_occluded", None),
    ("probe", 1, "probe:apple_occluded", None),
    ("probe", 2, "probe:apple_occluded", None),
    ("probe", 3, "probe:apple_occluded", None),
    ("probe", 4, "probe:apple_occluded", None),
    ("probe", 5, "probe:apple_occluded", None),
    ("probe", 6, "probe:apple_occluded", None),
    ("probe", 7, "probe:apple_occluded", None),
    ("probe", 0, "probe:apple_occluded_b", None),
    ("probe", 1, "probe:apple_occluded_b", None),
)

#: ATTEMPT 2's probes: unmodified corpus frames, indexed into ``probe_observations.json``'s
#: accepted list. No pixel of these is manufactured, so no answer to them can be about an artefact.
PROBE_PLAN_NATURAL: tuple[tuple[str, int, str, str | None], ...] = tuple(
    ("probe", i, "probe:natural", None) for i in range(N_PROBE))

#: The full plan for the set attempt 1 ran, kept whole so that run can still be rebuilt.
ITEM_PLAN = ITEM_PLAN_DECIDABLE + PROBE_PLAN_COMPOSITE


def item_plan(probe_set: str) -> tuple[tuple[str, int, str, str | None], ...]:
    """The thirty decidable items — identical across attempts — plus the chosen probe set.

    THE DECIDABLE THIRTY ARE THE SAME BYTES IN BOTH ATTEMPTS, and that is the point of splitting
    the plan here rather than writing a second plan: re-running a calibration after a failure is
    only honest if the part that passed is not quietly re-drawn at the same time.
    """
    if probe_set == PROBE_SET_COMPOSITE:
        return ITEM_PLAN_DECIDABLE + PROBE_PLAN_COMPOSITE
    if probe_set == PROBE_SET_NATURAL:
        return ITEM_PLAN_DECIDABLE + PROBE_PLAN_NATURAL
    raise CalibrationError(f"unknown probe set {probe_set!r}; expected one of {PROBE_SETS}.")

#: ALL TEN PROBES OCCLUDE THE APPLE, and the other two recipes docs §4 offers were built, looked
#: at, and dropped — with the reason recorded here rather than quietly replaced, because "we only
#: probe one clause" is a real limit on what the probe floor measures.
#:
#: * "the frame is too dark to judge the cloth" — the rubric's own second example. Built at
#:   exposure 0.15, 0.10 and 0.07 (mean luminance 15.6, 10.4, 7.3 of 255). Looked at, the APPLE IS
#:   STILL PLAINLY ORANGE AND THE PLATE STILL PLAINLY PALE at every one of them: the vision path
#:   these instruments read images through normalises, so a frame that is arithmetically almost
#:   black is not perceptually unjudgeable. docs §4 is explicit that a probe a careful observer can
#:   still answer "is not a probe, it is a hard positive, and it will make an honest instrument
#:   look broken". So it is not in the set.
#: * "crush the exposure of the plate region to noise" / occlude the plate. The corpus's own
#:   gripper covers 97 % of an apple by its natural silhouette and only 54 % of a plate; the rest
#:   would have to be painted in, and a locally crushed or painted-over plate reads far more
#:   naturally as a PLATE MISMATCH ("the dish is dark now") than as an abstention — which would
#:   make the item measure an instrument's willingness to guess rather than its willingness to
#:   abstain.
#:
#: What survives is the rubric's first canonical case, "the hand occludes the apple", cut from the
#: corpus itself in two different gripper poses. The cost is stated in the artifact: the probe
#: floor tests abstention on ONE clause-bearing region.
PROBE_RECIPES_REJECTED = (
    "too-dark frame (still legible after normalisation)",
    "plate occluded / crushed (natural coverage 54%, and reads as a plate mismatch)",
)

REQUIRED_VERDICT = {"positive": "match", "negative": "mismatch", "probe": "unsure"}

#: An image-side tint of the cloth necessarily falsifies the BACKGROUND clause too, because in this
#: top-down framing the backdrop is the cloth (see the seed record's `axis_referents`). Naming both
#: is therefore a correct answer, not a sloppy one, and the key says so per item rather than the
#: scorer guessing.
PERMITTED_EXTRA = {
    "img:table_blue": {"background"},
    "img:table_beige": {"background"},
    # A hard key light bright enough to be unmistakable also washes the lit half of the cloth pale,
    # so "the cloth is not dark grey any more" is a true observation about this item and not a
    # wrong answer. Naming it INSTEAD of lighting still scores nothing: the required axis has to be
    # in the list.
    "img:lighting_hard": {"table", "background"},
}


def _load_natural_probes(out: Path, measured: set[str]) -> list[dict[str, Any]]:
    """``probe_observations.json``'s accepted frames, refused unless each was LOOKED AT.

    The natural probes are the one part of C40 whose label is not a fact about a transformation —
    nothing was transformed. What stands behind `unsure` instead is that somebody opened the frame
    and could not answer the apple clause from it, and wrote down why. That record is the item's
    warrant, so its absence is fatal here rather than a missing field three steps downstream.
    """
    path = out / "probe_observations.json"
    if not path.is_file():
        raise CalibrationError(
            f"{path} does not exist. `probe-scan` finds candidate frames; it does not decide that "
            "they are unjudgeable. docs §4: 'a probe that a careful person can still answer is not "
            "a probe, it is a hard positive, and it will make an honest instrument look broken.'")
    doc = json.loads(path.read_text())
    probes = doc.get("probes") or []
    if len(probes) != N_PROBE:
        raise CalibrationError(
            f"{path} accepts {len(probes)} probe(s) but the pass rule is written over {N_PROBE}. "
            "The floors are counts, not rates.")
    for p in probes:
        if not p.get("looked_at"):
            raise CalibrationError(f"{p.get('episode')} f{p.get('frame_index')} is not recorded as "
                                   "looked at, and nothing else stands behind its `unsure` label.")
        if not str(p.get("undecidable_because") or "").strip():
            raise CalibrationError(
                f"{p.get('episode')} f{p.get('frame_index')} records no reason it cannot be "
                "answered. A probe without one is an assertion, not an observation.")
        if not p.get("other_clauses_true"):
            raise CalibrationError(
                f"{p.get('episode')} f{p.get('frame_index')} does not record that the other four "
                "clauses hold of it. If one of them is false the honest answer is `mismatch`, and "
                "the item would punish an instrument for being right.")
        if p["episode"] in measured:
            raise CalibrationError(
                f"{p['episode']} is one of the forty MEASURED episodes. docs §3.1: no calibration "
                "frame may also be a measured frame.")
        if not Path(p["frame"]).is_file():
            raise CalibrationError(f"{p['frame']} is missing.")
        if sha256_file(Path(p["frame"])) != p["frame_sha256"]:
            raise CalibrationError(
                f"{p['frame']} does not match the sha256 the observation was written against.")
    return probes


def _prompt_slots_for(seed_rec: dict) -> dict[str, str]:
    slots = dict(COMMITTED_SLOTS)
    slots.update(seed_rec["slot_overrides"])
    return slots


def cmd_build(args: argparse.Namespace) -> int:
    np = _np()
    from build_identity_prompt_sheet import frame_index, read_identity_style  # noqa: PLC0415

    identity = read_identity_style(args.styles)
    check_template(identity["prompt"])

    seed_meta = json.loads((args.out / "seed_frames_meta.json").read_text())
    obs = json.loads((args.out / "seed_observations.json").read_text())
    if obs.get("committed_prompt") != identity["prompt"]:
        raise CalibrationError(
            "seed_observations.json was written against a different [identity_style].prompt than "
            "the one committed now. The seed pass decided which clauses are TRUE of these frames; "
            "against a changed prompt those decisions are about a string nobody committed."
        )
    overrides = obs["slot_overrides_shared"]
    seed_records = {r["episode"]: r for r in seed_meta["seed_frames"]}
    seed_ids = seed_meta["seed_episodes"]
    if len(seed_ids) != SEED_FRAME_COUNT:
        raise CalibrationError(f"expected {SEED_FRAME_COUNT} seed frames, found {len(seed_ids)}.")
    observed = {f["episode"]: f for f in obs["frames"]}
    unlooked = [e for e in seed_ids if not observed.get(e, {}).get("looked_at")]
    if unlooked:
        raise CalibrationError(
            f"the seed pass does not record having looked at {unlooked}. Every positive, every "
            "prompt-side negative and every probe is paired with the prompt that pass decided is "
            "TRUE of its frame; without the observation there is nothing behind the label."
        )

    seeds = [{
        "episode": e,
        "record": seed_records[e],
        "slot_overrides": overrides,
        "slots": None,
    } for e in seed_ids]
    for s in seeds:
        s["slots"] = _prompt_slots_for(s)

    episodes = _load_corpus(args.manifest)
    natural_probes: list[dict[str, Any]] = []
    occluder = occ_mask = occluder_b = occ_mask_b = None
    occ_src_ep = occ_frame_path = occ_b_path = None
    if args.probe_set == PROBE_SET_NATURAL:
        natural_probes = _load_natural_probes(args.out, measured=set(
            json.loads(ln)["episode"]
            for ln in (args.sheet / "sheet.jsonl").read_text().splitlines() if ln.strip()))
    else:
        occ_src_ep = PROVENANCE_EPISODE
        occ_frame_path = args.out / "occluder_source.png"
        entry = episodes[occ_src_ep]
        _extract(args.manifest.parent / str(entry["video"]),
                 int(round(args.occluder_fraction * (int(entry["frames"]) - 1))),
                 occ_frame_path, args.ffmpeg)
        occluder, occ_mask = build_occluder(load_rgb(occ_frame_path))
        # A second gripper pose, from a different episode, so that ten probes are not ten copies of
        # one silhouette pasted at ten offsets. An instrument that had learned this one shape would
        # answer `unsure` to it without ever deciding that a clause was unjudgeable.
        occ_b_path = args.out / "occluder_source_b.png"
        entry_b = episodes[args.occluder_b_episode]
        _extract(args.manifest.parent / str(entry_b["video"]),
                 int(round(args.occluder_b_fraction * (int(entry_b["frames"]) - 1))),
                 occ_b_path, args.ffmpeg)
        occluder_b, occ_mask_b = build_occluder(load_rgb(occ_b_path))

    src_dir = args.out / "items_src"
    if src_dir.exists():
        shutil.rmtree(src_dir)
    src_dir.mkdir(parents=True)

    built: list[dict[str, Any]] = []
    for n, (cls, si, op, axis) in enumerate(item_plan(args.probe_set), 1):
        if op == "probe:natural":
            # An unmodified corpus frame. It is not derived from a seed frame, so it carries the
            # seed pass's SHARED wording (the two clauses that pass found false of all eight) plus
            # its own looked-at record of the other clauses — written down in
            # probe_observations.json, checked by `_load_natural_probes`, and asserted below.
            rec = natural_probes[si]
            ep = rec["episode"]
            base = arr = load_rgb(Path(rec["frame"]))
            slots = _prompt_slots_for({"slot_overrides": overrides})
            side = None
            assertions = {k: rec[k] for k in (
                "frame_index", "apple_warm_px", "apple_warm_ratio",
                "episode_median_apple_warm_px", "ring_foreground_fraction",
                "undecidable_because", "looked_at", "other_clauses_true")}
            _assert_mutation_landed(op, base, arr, assertions)
            tag = f"{cls[:3]}{n:02d}_{ep}_f{rec['frame_index']:04d}_natural"
            png = src_dir / f"{tag}.png"
            shutil.copyfile(rec["frame"], png)
            built.append({
                "tag": tag, "class": cls, "episode": ep, "mutation": op, "side": side,
                "required_verdict": REQUIRED_VERDICT[cls], "required_axis": axis,
                "permitted_axes": [],
                "prompt": render_prompt(slots),
                "prompt_is_committed": render_prompt(slots) == identity["prompt"],
                "source_frame": str(Path(rec["frame"]).resolve()),
                "built_frame": str(png.resolve()),
                "assertions": assertions,
            })
            continue

        s = seeds[si]
        ep = s["episode"]
        base = load_rgb(Path(s["record"]["frame"]))
        slots = dict(s["slots"])
        side = None
        assertions: dict[str, Any] = {}
        arr = base

        if op == "none":
            pass
        elif op == "reencode":
            arr, assertions = null_reencode(base)
        elif op.startswith("exposure:"):
            arr, assertions = null_exposure(base, float(op.split(":", 1)[1]))
        elif op.startswith("frame:"):
            delta = int(op.split(":", 1)[1])
            e = episodes[ep]
            idx = frame_index(int(e["frames"]), s["record"]["frame_fraction"]) + delta
            tmp = src_dir / f"_neighbour_{ep}_{delta:+d}.png"
            _extract(args.manifest.parent / str(e["video"]), idx, tmp, args.ffmpeg)
            arr = load_rgb(tmp)
            assertions = {"kind": f"frame index {delta:+d}", "frame_index": idx,
                          "max_abs_delta": int(np.abs(arr - base).max())}
            tmp.unlink()
        elif op == "img:apple_green":
            side = "image"
            arr, assertions = mutate_apple_green(base)
        elif op == "img:table_blue":
            side = "image"
            arr, assertions = mutate_table_tint(base, (0.55, 0.95, 2.0), "blue")
        elif op == "img:table_beige":
            side = "image"
            arr, assertions = mutate_table_tint(base, (1.45, 1.15, 0.60), "beige")
        elif op == "img:background_band":
            side = "image"
            arr, assertions = mutate_background_band(base)
        elif op == "img:lighting_hard":
            side = "image"
            arr, assertions = mutate_lighting_hard(base)
        elif op == "img:plate_dark":
            side = "image"
            arr, assertions = mutate_plate_dark(base)
        elif op.startswith("prompt:"):
            side = "prompt"
            ax = op.split(":", 1)[1]
            slots = substitute(slots, ax, FALSIFIED_SLOTS[ax])
            assertions = {"falsified_slots": sorted(FALSIFIED_SLOTS[ax])}
        elif op == "probe:apple_occluded":
            arr, assertions = probe_occlude(base, apple_mask(base), occluder, occ_mask)
        elif op == "probe:apple_occluded_b":
            arr, assertions = probe_occlude(base, apple_mask(base), occluder_b, occ_mask_b)
        else:
            raise CalibrationError(f"unknown operation {op!r} in the item plan.")

        _assert_mutation_landed(op, base, arr, assertions)

        tag = f"{cls[:3]}{n:02d}_{ep}_{op.replace(':', '-').replace('+', 'p')}"
        png = src_dir / f"{tag}.png"
        if arr is base and op in ("none",):
            shutil.copyfile(s["record"]["frame"], png)
        else:
            save_rgb(arr, png)

        built.append({
            "tag": tag, "class": cls, "episode": ep, "mutation": op, "side": side,
            "required_verdict": REQUIRED_VERDICT[cls],
            "required_axis": axis,
            "permitted_axes": sorted({axis} | PERMITTED_EXTRA.get(op, set())) if axis else [],
            "prompt": render_prompt(slots),
            "prompt_is_committed": render_prompt(slots) == identity["prompt"],
            "source_frame": str(Path(s["record"]["frame"]).resolve()),
            "built_frame": str(png.resolve()),
            "assertions": assertions,
        })

    for cls, want in (("positive", N_POSITIVE), ("negative", N_NEGATIVE), ("probe", N_PROBE)):
        got = sum(1 for b in built if b["class"] == cls)
        if got != want:
            raise CalibrationError(f"plan built {got} {cls} items, the pass rule expects {want}.")

    # ---- the interleave -------------------------------------------------------------------
    real_rows = [json.loads(ln) for ln in (args.sheet / "sheet.jsonl").read_text().splitlines()
                 if ln.strip()]
    if len(real_rows) != 40:
        raise CalibrationError(f"{args.sheet}/sheet.jsonl holds {len(real_rows)} rows, expected 40.")
    if any(str(r.get("verdict") or "").strip() for r in real_rows):
        raise CalibrationError(
            "the real sheet already carries verdicts. Re-asking a filled sheet would replace "
            "answers somebody produced by looking, and the whole point of interleaving is that "
            "the real rows are answered ONCE, in the same session as the calibration."
        )

    pool: list[dict[str, Any]] = []
    for b in built:
        pool.append({"kind": "calibration", "frame": b["built_frame"], "prompt": b["prompt"],
                     "detail": b})
    for r in real_rows:
        pool.append({"kind": "real_row", "frame": r["frame"], "prompt": r["prompt"],
                     "detail": {"episode": r["episode"], "sheet_id": r["sheet_id"],
                                "frame_sha256": r["frame_sha256"]}})

    rng = random.Random(args.shuffle_seed)
    rng.shuffle(pool)

    items_dir = args.out / "items"
    if items_dir.exists():
        shutil.rmtree(items_dir)
    items_dir.mkdir(parents=True)

    items: list[dict[str, str]] = []
    key: dict[str, Any] = {}
    for n, p in enumerate(pool, 1):
        item_id = f"item_{n:04d}"
        staged = items_dir / f"{item_id}.png"
        # Copied under an opaque name because the FILENAME is the leak. A row pointing at
        # `frames/episode_000003.png` tells the instrument it is looking at a real corpus row, and
        # one pointing at `neg07_..._img-table_blue.png` tells it the answer.
        #
        # RE-ENCODED rather than byte-copied, and PNG is lossless so not one pixel moves. The forty
        # real frames came out of ffmpeg and the mutated ones out of Pillow; left as they were, the
        # two populations differ in encoder metadata, chunk layout and file size, and an instrument
        # with a shell could separate the classes without looking at a single picture. The pixels
        # are the item; everything else about the file is an artefact of who wrote it.
        _restage(Path(p["frame"]), staged)
        items.append({"item_id": item_id, "frame": str(staged.resolve()), "prompt": p["prompt"]})
        d = dict(p["detail"])
        d["kind"] = p["kind"]
        d["item_id"] = item_id
        d["staged_frame_sha256"] = sha256_file(staged)
        key[item_id] = d

    (args.out / "items.jsonl").write_text(
        "".join(json.dumps(i, sort_keys=True) + "\n" for i in items), encoding="utf-8")
    (args.out / "key.json").write_text(json.dumps({
        "schema": CALIBRATION_SCHEMA,
        "step": "key",
        "WARNING": (
            "THE INSTRUMENT MUST NEVER SEE THIS FILE. items.jsonl carries the frame and the prompt "
            "and nothing else; the labels live here, in a different file, because T-041's own note "
            "is right that 'the code did not look' is a weaker guarantee than 'the labels were in "
            "a different file'."
        ),
        "shuffle_seed": args.shuffle_seed,
        "n_items": len(items),
        "items": key,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    meta = {
        "schema": CALIBRATION_SCHEMA,
        "step": "build",
        "writeup": WRITEUP,
        "todo": TODO_ID,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "built_date": date.today().isoformat(),
        "committed_prompt": identity["prompt"],
        "partition_content_sha256": identity["partition_content_sha256"],
        "seed_episodes": seed_ids,
        "seed_established_by": obs["established_by"],
        "seed_established_by_note": obs["established_by_note"],
        "slot_overrides_shared": overrides,
        "true_prompt_example": render_prompt(seeds[0]["slots"]),
        "committed_clauses_false_on_seed": sorted(overrides),
        "shuffle_seed": args.shuffle_seed,
        "seed_frame_seed": seed_meta["seed_frame_seed"],
        "n_calibration_items": len(built),
        "n_real_rows": len(real_rows),
        "class_counts": {c: sum(1 for b in built if b["class"] == c)
                         for c in ("positive", "negative", "probe")},
        "negative_side_counts": {
            s: sum(1 for b in built if b["class"] == "negative" and b["side"] == s)
            for s in ("image", "prompt")},
        "negative_axis_counts": {
            a: sum(1 for b in built if b["class"] == "negative" and b["required_axis"] == a)
            for a in ("apple", "table", "background", "lighting", "plate")},
        "probe_set": args.probe_set,
        "occluder_sources": [] if args.probe_set == PROBE_SET_NATURAL else [
            {"episode": occ_src_ep, "fraction": args.occluder_fraction,
             "frame": str(occ_frame_path.resolve())},
            {"episode": args.occluder_b_episode, "fraction": args.occluder_b_fraction,
             "frame": str(occ_b_path.resolve())},
        ],
        "natural_probes": [
            {k: v for k, v in p.items() if k != "clauses"} for p in natural_probes],
        "probe_recipes_rejected": list(PROBE_RECIPES_REJECTED),
        "pass_rule": {
            "positives_floor": FLOOR_POSITIVE, "negative_token_floor": FLOOR_NEGATIVE_TOKEN,
            "negative_axis_floor": FLOOR_NEGATIVE_AXIS, "probe_floor": FLOOR_PROBE,
            "leakage_ceiling": CEIL_LEAKAGE, "max_axes_for_axis_credit": MAX_AXES_FOR_CREDIT,
        },
        "coin_probabilities": coin_probabilities(),
        "known_weaknesses": [
            "THE SEED PASS WAS NOT HUMAN. See seed_observations.json's `established_by_note`.",
            *([
                "ALL TEN NATURAL PROBES COME FROM ONE EPISODE AND ONE OCCLUSION EVENT, because "
                "that is what the corpus contains: of 154 447 frames in the 362 non-measured "
                "episodes, 48 have less than 1 200 px of apple visible and 24 clear the "
                "eligibility rule, and every one of them is in episode_000094 between frames 108 "
                "and 151. The probe floor therefore measures abstention on ONE event sampled ten "
                "times; ten correct answers are ten correlated successes. See probe_census.json "
                "for the measurement and probe_observations.json for what was seen in each frame.",
            ] if args.probe_set == PROBE_SET_NATURAL else []),
            "docs §3.1 asks for seed frames from strata the sheet did not sample. The sheet draws "
            "one episode from each of forty contiguous strata, so no stratum is unsampled; what is "
            "honoured instead is the property that protects — no calibration frame is also a "
            "measured frame.",
            "`table` and `background` are not separable in this scene: the backdrop IS the cloth. "
            "The one image-side background negative can therefore only falsify the clause's "
            "parenthetical about the pale top strip, and it is the weakest item in the set.",
            "The forty real rows carry the committed prompt; every calibration item carries the "
            "seed pass's TRUE prompt or a falsification of it, and those differ on two clauses. An "
            "instrument that noticed the string frequency could in principle tell the two "
            "populations apart. The items are split across instruments in blocks of twenty, which "
            "weakens but does not remove the cue.",
            "docs §3.6: C40's items are manufactured — one clause falsified hard, or nothing "
            "falsified at all. The real forty are natural frames where the honest answer may be "
            "'the cloth is black but this one looks charcoal'. PASSING C40 IS NECESSARY AND NEVER "
            "SUFFICIENT.",
        ],
        "items": [{k: v for k, v in b.items() if k != "prompt"} for b in built],
    }
    (args.out / "build_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"calibration {len(built)} items  "
          f"({meta['class_counts']}, negatives {meta['negative_side_counts']})", file=sys.stderr)
    print(f"interleave  {len(items)} items under shuffle seed {args.shuffle_seed}", file=sys.stderr)
    print(f"wrote       {args.out / 'items.jsonl'}  (frame + prompt only)", file=sys.stderr)
    print(f"            {args.out / 'key.json'}     (NEVER give this to the instrument)",
          file=sys.stderr)
    print(f"            {args.out / 'build_meta.json'}", file=sys.stderr)
    return EXIT_OK


def _restage(src: Path, dst: Path) -> None:
    from PIL import Image  # noqa: PLC0415
    np = _np()
    a = np.asarray(Image.open(src).convert("RGB"))
    Image.fromarray(a, "RGB").save(dst, format="PNG", optimize=False)
    b = np.asarray(Image.open(dst).convert("RGB"))
    if not np.array_equal(a, b):
        raise CalibrationError(f"re-staging {src} changed its pixels — refusing to issue it.")


def _assert_mutation_landed(op: str, before, after, assertions: dict[str, Any]) -> None:
    """Refuse an item whose transformation did not do what its label says it did.

    This is the difference between a calibration set and a pile of plausible pictures. An item
    labelled `mismatch` whose mutation silently missed the apple is an item that PUNISHES a correct
    answer, and nothing downstream can see it: the frame looks fine, the key looks fine, and the
    instrument's score is quietly wrong in the direction of "the instrument is bad".
    """
    np = _np()
    changed = float(np.abs(after - before).mean())
    if op == "none":
        return
    if op.startswith("prompt:"):
        if changed != 0.0:
            raise CalibrationError(f"{op}: a prompt-side negative must not touch the pixels.")
        return
    if op == "probe:natural":
        # There is no mutation to have landed — that is the whole point — so what is checked is
        # that the frame really is one the corpus already had, and that it really is the frame the
        # scan measured.
        if changed != 0.0:
            raise CalibrationError(
                f"{op}: a natural probe must be the corpus's own pixels, unmodified. Compositing "
                "is what attempt 1 did, and three of its ten probes were answered `mismatch: "
                "other` with notes calling the region a blocky artifact.")
        if assertions.get("apple_warm_px", 10 ** 9) > NATURAL_PROBE_MAX_WARM_PX:
            raise CalibrationError(
                f"{op}: {assertions.get('apple_warm_px')} px of apple are still visible, above "
                f"the registered {NATURAL_PROBE_MAX_WARM_PX}. A frame a careful person can still "
                "answer is a hard positive wearing a probe's label.")
        if assertions.get("ring_foreground_fraction", 0.0) < NATURAL_PROBE_MIN_RING_FOREGROUND:
            raise CalibrationError(
                f"{op}: only {assertions.get('ring_foreground_fraction')} of the ring around what "
                "is left of the apple is foreground, so nothing visible is hiding it. An apple "
                "that is simply absent from the scene is a `mismatch`, not an abstention.")
        if not assertions.get("looked_at") or not assertions.get("other_clauses_true"):
            raise CalibrationError(f"{op}: the frame's observation record is incomplete.")
        return
    if changed == 0.0:
        raise CalibrationError(f"{op}: the mutation changed nothing. The item would be labelled "
                               "against an untouched frame.")
    if op == "img:apple_green" and not assertions.get("green_now_dominant"):
        raise CalibrationError(f"{op}: the apple is not green afterwards ({assertions}).")
    if op.startswith("img:table_") and abs(assertions.get("mean_luma_shift", 99)) > 3.0:
        raise CalibrationError(f"{op}: the tint moved the cloth's luminance by "
                               f"{assertions['mean_luma_shift']} — that falsifies lighting too.")
    if op.startswith("img:table_") and assertions.get("saturation_after", 0) < 0.30:
        raise CalibrationError(f"{op}: the tinted cloth is still near-neutral ({assertions}).")
    if op == "img:lighting_hard" and assertions.get("left_right_ratio", 0) < 2.0:
        raise CalibrationError(f"{op}: the lighting is still even ({assertions}).")
    if op == "img:plate_dark" and assertions.get("mean_luma_after", 255) > 80:
        raise CalibrationError(f"{op}: the plate is still bright ({assertions}).")
    if op.startswith("probe:"):
        if assertions.get("coverage", 0) < 1.0:
            raise CalibrationError(
                f"{op}: the occluder covers only {assertions.get('coverage')} of the target "
                "region. docs §4: a probe a careful person can still answer is a hard positive "
                "wearing a probe's label, and it makes an honest instrument look broken.")
        if assertions.get("natural_coverage", 0) < 0.90:
            raise CalibrationError(
                f"{op}: the gripper's own silhouette covered only "
                f"{assertions.get('natural_coverage')} of the target and the rest had to be "
                "painted in. Past a tenth of the region that is no longer a hand over an apple, "
                "it is a rectangle of paint, and an instrument is right to call it something "
                "other than an occlusion.")



# --------------------------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------------------------


def read_answers(paths: list[Path]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in paths:
        for n, ln in enumerate(p.read_text().splitlines(), 1):
            if not ln.strip():
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError as exc:
                raise CalibrationError(f"{p}:{n} is not valid JSON ({exc}).") from exc
            iid = str(row.get("item_id") or "").strip()
            if not iid:
                raise CalibrationError(f"{p}:{n} carries no item_id.")
            if iid in out:
                raise CalibrationError(
                    f"{iid} is answered twice (second time in {p}:{n}). Two answers to one item "
                    "means one of them is being chosen by whoever merges the files.")
            out[iid] = row
    return out


def cmd_score(args: argparse.Namespace) -> int:
    keydoc = json.loads((args.out / "key.json").read_text())
    full = keydoc["items"]
    cal_key = {k: v for k, v in full.items() if v.get("kind") == "calibration"}
    answers = read_answers(sorted((args.out / "answers").glob("*.jsonl")))

    unknown = sorted(set(answers) - set(full))
    if unknown:
        raise CalibrationError(f"{len(unknown)} answer(s) name an item that was never issued: "
                               f"{', '.join(unknown[:8])}")
    unanswered = sorted(set(full) - set(answers))

    scores = score_answers(cal_key, {k: v for k, v in answers.items() if k in cal_key})
    scores.update({
        "schema": CALIBRATION_SCHEMA,
        "step": "score",
        "scored_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shuffle_seed": keydoc.get("shuffle_seed"),
        "n_items_issued": len(full),
        "n_answers": len(answers),
        "n_real_rows_answered": sum(1 for k in answers if full[k].get("kind") == "real_row"),
        "unanswered_items": unanswered,
        "coin_probabilities": coin_probabilities(),
        "scope": (
            "A statement about the INSTRUMENT and never about arm C. Passing means the instrument "
            "can confirm a true prompt, detect a hard one-clause falsification with the right "
            "axis, and abstain when a clause is unjudgeable — on manufactured items. docs §3.6: "
            "necessary and never sufficient."
        ),
    })
    (args.out / "scores.json").write_text(
        json.dumps(scores, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _print_floors(scores)
    if unanswered:
        print(f"\n{len(unanswered)} issued item(s) unanswered: "
              f"{', '.join(unanswered[:8])}", file=sys.stderr)
    print(f"\nwrote {args.out / 'scores.json'}", file=sys.stderr)
    return EXIT_OK if scores["passed"] else EXIT_FLOOR_FAILED


def _print_floors(scores: dict[str, Any]) -> None:
    print("FIVE FLOORS (docs §3.4) — reported as five numbers, never as an aggregate:",
          file=sys.stderr)
    for f in scores["floors"]:
        mark = "PASS" if f["passed"] else "FAIL"
        print(f"  [{mark}] {f['name']:<44s} {f['value']:>3d} / {f['of']:<3d} "
              f"(floor {f['direction']} {f['floor']})", file=sys.stderr)
    print(f"  => {'ALL FIVE HOLD' if scores['passed'] else 'CALIBRATION FAILED'}", file=sys.stderr)


def cmd_degenerate(args: argparse.Namespace) -> int:
    keydoc = json.loads((args.out / "key.json").read_text())
    cal_key = {k: v for k, v in keydoc["items"].items() if v.get("kind") == "calibration"}
    out: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "step": "degenerate",
        "scored_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "why": (
            "T-041's judge answered the literal string 'NO' to all 80 items and its calibration "
            "reported 10/20. If any vector below PASSES, this scorer cannot detect the failure it "
            "was written for and no number it produces about a real instrument means anything."
        ),
        "coin_probabilities": coin_probabilities(),
        "vectors": {},
    }
    for name, vec in degenerate_vectors(cal_key, args.degenerate_seed).items():
        s = score_answers(cal_key, vec)
        out["vectors"][name] = {"passed": s["passed"], "counts": s["counts"],
                                "floors": s["floors"]}
        print(f"\n--- {name} ---", file=sys.stderr)
        _print_floors(s)
    (args.out / "degenerate_scores.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    any_passed = any(v["passed"] for v in out["vectors"].values())
    print(f"\nwrote {args.out / 'degenerate_scores.json'}", file=sys.stderr)
    if any_passed:
        print("\nA DEGENERATE VECTOR PASSED. The scorer is broken.", file=sys.stderr)
    return EXIT_FLOOR_FAILED if any_passed else EXIT_OK


def cmd_fill_sheet(args: argparse.Namespace) -> int:
    """Copy the blinded run's answers for the forty REAL rows into sheet.jsonl. Refuses if C40 failed.

    The refusal is the point, and it is the T-041 failure spelled as code: a filled sheet from an
    uncalibrated instrument is exactly what VOIDed that run, and "we knew the calibration failed but
    filled it anyway to see" is a decision nobody should be able to make by forgetting.
    """
    scores = json.loads((args.out / "scores.json").read_text())
    if not scores.get("passed"):
        failed = [f["name"] for f in scores["floors"] if not f["passed"]]
        raise CalibrationError(
            f"the instrument did not clear {len(failed)} of the five floors ({'; '.join(failed)}). "
            "Its answers on the real forty are not evidence, and writing them into the sheet would "
            "produce a gate-qualified artifact from an instrument that failed its own calibration "
            "— the precise shape of T-041's VOID."
        )
    keydoc = json.loads((args.out / "key.json").read_text())
    full = keydoc["items"]
    answers = read_answers(sorted((args.out / "answers").glob("*.jsonl")))
    by_episode = {v["episode"]: answers[k] for k, v in full.items()
                  if v.get("kind") == "real_row" and k in answers}

    sheet_file = args.sheet / "sheet.jsonl"
    rows = [json.loads(ln) for ln in sheet_file.read_text().splitlines() if ln.strip()]
    missing = [r["episode"] for r in rows if r["episode"] not in by_episode]
    if missing:
        raise CalibrationError(f"{len(missing)} real row(s) have no answer from the blinded run: "
                               f"{', '.join(missing[:12])}")
    # Checked here rather than left to `verdict`, because the refusal there names the SHEET and
    # this one can name the answer file the row came from. Nothing is repaired: an answer that says
    # `match` and then names a broken axis says two different things, and picking one of them would
    # be this script inventing a verdict nobody gave.
    contradictory = [
        (ep, a) for ep, a in by_episode.items()
        if str(a.get("verdict") or "").strip().lower() != "mismatch" and _norm_axes(
            a.get("mismatched_axes"))
    ]
    if contradictory:
        raise CalibrationError(
            f"{len(contradictory)} blinded answer(s) name a mismatched axis while their verdict is "
            f"not `mismatch`: {', '.join(e for e, _ in contradictory[:8])}. The answer says two "
            "different things; it is not this script's place to choose which half to believe."
        )
    for r in rows:
        a = by_episode[r["episode"]]
        r["verdict"] = str(a.get("verdict") or "").strip().lower()
        r["mismatched_axes"] = _norm_axes(a.get("mismatched_axes"))
        r["notes"] = str(a.get("notes") or "").strip()
    sheet_file.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
                          encoding="utf-8")
    counts = {v: sum(1 for r in rows if r["verdict"] == v) for v in VERDICT_VALUES}
    print(f"filled {len(rows)} rows in {sheet_file}: {counts}", file=sys.stderr)
    return EXIT_OK


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    default_out = _REPO_ROOT / "runs" / "t040-identity-prompt" / "calibration"
    default_sheet = _REPO_ROOT / "runs" / "t040-identity-prompt"
    default_styles = _REPO_ROOT / "configs" / "transfer25" / "styles.toml"

    s = sub.add_parser("seed-frames", help="extract the eight seed frames for the seed pass")
    s.add_argument("--manifest", type=Path,
                   default=Path("/home/humanoid/wam-t041/pr08-apple-640x480/manifest.json"))
    s.add_argument("--sheet", type=Path, default=default_sheet)
    s.add_argument("--styles", type=Path, default=default_styles)
    s.add_argument("--out", type=Path, default=default_out)
    s.add_argument("--seed-frame-seed", type=int, default=DEFAULT_SEED_FRAME_SEED)
    s.add_argument("--frame-fraction", type=float, default=0.10)
    s.add_argument("--ffmpeg", default="ffmpeg")

    b = sub.add_parser("build", help="build C40 and interleave it with the forty real rows")
    b.add_argument("--manifest", type=Path,
                   default=Path("/home/humanoid/wam-t041/pr08-apple-640x480/manifest.json"))
    b.add_argument("--sheet", type=Path, default=default_sheet)
    b.add_argument("--styles", type=Path, default=default_styles)
    b.add_argument("--out", type=Path, default=default_out)
    b.add_argument("--shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    b.add_argument("--occluder-fraction", type=float, default=0.45,
                   help="where in the provenance clip the Dex3 gripper is cut from, for the "
                        "occlusion probes")
    b.add_argument("--occluder-b-episode", default="episode_000018")
    b.add_argument("--occluder-b-fraction", type=float, default=0.50)
    b.add_argument("--probe-set", choices=PROBE_SETS, default=PROBE_SET_COMPOSITE,
                   help="composite: attempt 1's gripper pasted over the apple. natural: attempt "
                        "2's unmodified corpus frames, from probe_observations.json.")
    b.add_argument("--ffmpeg", default="ffmpeg")

    p = sub.add_parser("probe-scan",
                       help="census the corpus for frames whose apple is genuinely hidden")
    p.add_argument("--manifest", type=Path,
                   default=Path("/home/humanoid/wam-t041/pr08-apple-640x480/manifest.json"))
    p.add_argument("--sheet", type=Path, default=default_sheet)
    p.add_argument("--out", type=Path, default=default_out)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--ffmpeg", default="ffmpeg")

    c = sub.add_parser("score", help="score answers/*.jsonl against key.json, five floors")
    c.add_argument("--out", type=Path, default=default_out)

    d = sub.add_parser("degenerate", help="run the degenerate answer vectors through the scorer")
    d.add_argument("--out", type=Path, default=default_out)
    d.add_argument("--degenerate-seed", type=int, default=40004)

    f = sub.add_parser("fill-sheet", help="write the real rows' answers into sheet.jsonl")
    f.add_argument("--out", type=Path, default=default_out)
    f.add_argument("--sheet", type=Path, default=default_sheet)

    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dispatch: dict[str, Callable[[argparse.Namespace], int]] = {
        "seed-frames": cmd_seed_frames,
        "probe-scan": cmd_probe_scan,
        "build": cmd_build,
        "score": cmd_score,
        "degenerate": cmd_degenerate,
        "fill-sheet": cmd_fill_sheet,
    }
    try:
        return dispatch[args.cmd](args)
    except CalibrationError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return EXIT_FATAL


if __name__ == "__main__":
    raise SystemExit(main())
