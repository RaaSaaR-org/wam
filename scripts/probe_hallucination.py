#!/usr/bin/env python3
"""T-040 / PR-08 `T40_RULE_V8` — does Cosmos-Transfer2.5 put a manipulator into a robot-free frame?

    python scripts/probe_hallucination.py \
        --manifest <SOURCE/manifest.json> --styles <STYLE_PARTITION> \
        --checkpoint-path <ckpt> --control depth:0.5,seg:0.5 \
        --out <QUARANTINE-DIR> --episodes 2 --style-count 2 --frames 96

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
It is the measurement `docs/preregistration/PR-08-V8-hallucination-probe.md` registers, and nothing
else. It restyles a **hard-capped** number of source frames that were selected **because a
non-learned reference predicate says the robot is absent from them**, runs the committed robot
masker on the source frame and on the generated frame, and records the pair. Source empty +
generated empty means nothing was invented on that frame; source empty + generated grounding
something is a CANDIDATE invention, upper-bounded and never self-certified — see the caveat below.

**IT IS NOT A CORPUS GENERATOR AND IT IS BUILT SO THAT IT CANNOT BECOME ONE.** That is a design
requirement rather than a preference: a narrowly-licensed generation path that could be widened by
an environment variable is exactly the hole `T40_RULE_V1` §1 exists to close. Four mechanisms, none
of which is a comment:

1. :data:`PROBE_MAX_TOTAL_FRAMES` and its two factors are MODULE CONSTANTS. Every count is clamped
   against them and anything above exits non-zero (:func:`enforce_caps`). There is no environment
   variable, no flag and no config file that raises them. Raising them is a `V9`.
2. It writes NOTHING a downstream consumer reads. No ``manifest.json``, no ``work.jsonl``, no
   ``sample_outputs.json``, no ``vision.mp4``, no parquet, no ``meta/`` — and no file whose name
   ends in ``.mp4`` at all, because ``scripts/assemble_restyled_lerobot.py`` files a clip directory
   by ``glob("*.mp4")`` and ``97_transfer25_restyle.sbatch``'s harvest keys on a file called
   ``vision.mp4``. Video output is renamed to ``.mp4.quarantined`` the moment the generator returns
   it; both readers this repository has are content-based or explicit-plugin, so the bytes stay
   inspectable while the two globs that could file them miss.
3. :func:`audit_output_tree` walks what was written and REFUSES, non-zero, if any of those names
   came back. A guarantee that rests on every future edit remembering a rule is weaker than one
   that rests on a walk of the directory.
4. ``--out`` must carry :data:`QUARANTINE_TOKEN` in its name, so the output cannot be dropped into
   a corpus tree by typing a different path.

**IT DOES NOT CALL ``check_mask``.** G0c's refusal is the thing this measures the premise of;
running the gate would refuse every unit before a number existed. The masker is imported, the gate
is not. Nothing here changes ``scripts/robot_composite.py``, derives a robot-mask area bound, or
proposes one.

THE FRAME SELECTION IS THE DIAGNOSIS'S OWN PREDICATE, NOT A NEW ONE
-------------------------------------------------------------------
``scripts/diagnose_robot_mask_empty.py``'s ``robot_dark_mask`` / ``classify``, at the primary
committed setting (dark_offset 45, sat_max 0.25, change_min 25) and at the band
``runs/pr08-robot-mask-empty/DIAGNOSIS.json`` records under ``instrument.band_px`` (800 / 3000).
Those two numbers are CONSTANTS here and not flags: a band typed at submit time is a per-run
decision about which frames count as robot-free, recorded nowhere anybody would look. The predicate
is NOT ground truth — its own docstring names both directions of its error — and V8 §7 carries the
consequences.

THE CANDIDATE COUNT IS AN UPPER BOUND, AND THE SHEETS ARE THE MEASUREMENT
------------------------------------------------------------------------
``DIAGNOSIS.json`` measured the committed masker returning a non-empty mask on 41 % of robot-absent
frames corpus-wide, and the mask is THE APPLE (~6-7 k px), the plate (~40-48 k) or the whole cloth.
So a non-empty mask on a generated frame is not by itself a manipulator. Two consequences, both
registered in V8 §7 before this ran:

* frames whose SOURCE mask is non-empty are excluded from the headline and counted separately —
  on such a frame nothing about invention is provable either way;
* the automated candidate count is an UPPER BOUND on invention. The finding is a person reading the
  contact sheets this writes. ``human_review.looked_at`` is ``false`` in the artifact and stays
  false until that happens, for the reason ``MASK_AUDIT.json`` records: a model checking masks
  produced by a pipeline a model wired up is a correlated observer.

BACKENDS. ``--backend transfer25`` is the default and is what the cluster path runs;
``107_hallucination_probe.sbatch`` never passes ``--backend``, and a test asserts it. ``null`` is
``restyle_transfer25``'s deterministic placeholder and exists so the plumbing is testable without a
GPU, a checkout or a weight.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import robot_composite  # noqa: E402 — the committed masker; the GATE is deliberately not imported
import restyle_transfer25 as rt  # noqa: E402 — the same pinned generator call the corpus path uses
from diagnose_robot_mask_empty import (  # noqa: E402
    CHANGE_MINS,
    DARK_OFFSETS,
    SAT_MAXES,
    apply_setting,
    background_median,
    classify,
    decode,
    frame_fields,
    largest_component,
    read_manifest,
    runs_of,
)

SCHEMA = "wam.transfer25_hallucination_probe/1"
RULE = "T40_RULE_V8"
V8_DOC = "docs/preregistration/PR-08-V8-hallucination-probe.md"

# ------------------------------------------------------------------------------------------------
# THE CAP. Constants, deliberately: nothing reads these from the environment, from a config file or
# from the command line, and every count the caller asks for is refused above them rather than
# clamped silently. 6 x 121 = 726 frames is 1.70x the mean episode (427) that PR-08 §8 item 3's
# ALREADY LICENSED timing run restyles, 0.42 % of one style-instance (171 625 frames) and 0.017 % of
# the whole registered partition (~4.29 M). V8 §3.1 carries that arithmetic.
# ------------------------------------------------------------------------------------------------
PROBE_MAX_CLIPS = 6
PROBE_MAX_FRAMES_PER_CLIP = 121
PROBE_MAX_TOTAL_FRAMES = 726
PROBE_MAX_EPISODES = 3
PROBE_MAX_STYLES = 3

#: Below this, a contiguous robot-absent run is not worth generating: the paired population would be
#: too small to distinguish "invented nothing" from "measured nothing".
PROBE_MIN_FRAMES_PER_CLIP = 48

#: A unit with fewer paired probe frames than this reports outcome ``U`` (V8 §5.3) rather than a
#: quiet zero. Absence of evidence and evidence of absence get different letters.
MIN_PAIRED_PROBE_FRAMES = 16

#: How many manifest episodes may be scanned looking for eligible ones. A scan is CPU-only and
#: cheap, but an unbounded one on a 402-episode corpus is a job that looks hung.
PROBE_MAX_SCAN_EPISODES = 24

#: The classification band, from DIAGNOSIS.json ``instrument.band_px``. NOT a flag; see the module
#: docstring. ``absent`` is strictly below the first, ``present`` strictly above the second, and the
#: gap is its own bucket rather than being forced into either.
ABSENT_BELOW = 800
PRESENT_ABOVE = 3000

#: The committed style set this probe may draw prompts from. There is no flag that changes it:
#: TRAIN_STYLES is what arm B actually runs, EVAL_STYLES is the held-out domain and has no business
#: being touched before the experiment, and the identity style is arm C's question, not this one.
PROBE_STYLE_SET = "train"

#: ``--out`` must contain this, so the output cannot be written into a corpus tree by typing a path.
QUARANTINE_TOKEN = "hallucination-probe"

#: Renamed to this the moment the generator returns a clip. ``.mp4`` is what
#: ``assemble_restyled_lerobot.py`` globs and ``vision.mp4`` is what 97's harvest keys on.
QUARANTINE_SUFFIX = ".mp4.quarantined"

#: Names that would make this tree consumable. :func:`audit_output_tree` refuses any of them.
FORBIDDEN_NAMES = (
    "manifest.json",
    "work.jsonl",
    "sample_outputs.json",
    "vision.mp4",
    "chunk_metadata.json",
    "info.json",
    "episodes.jsonl",
    "tasks.jsonl",
)
FORBIDDEN_SUFFIXES = (".mp4", ".parquet", ".g0c.json")

PAIRING_BOTH_EMPTY = "both_empty"
PAIRING_CANDIDATE = "candidate_invention"
PAIRING_EXCLUDED = "excluded_source_mask_nonempty"

#: One contact sheet per bucket, including the boring ones. A sheet family that showed only the
#: candidates would give a reader no way to judge what a NON-candidate frame looks like under the
#: same masker, which is the comparison that makes a candidate legible.
STRATA = {
    PAIRING_CANDIDATE: "CANDIDATE INVENTION - source mask EMPTY, generated mask NOT",
    PAIRING_BOTH_EMPTY: "both empty - nothing was invented on these frames",
    PAIRING_EXCLUDED: "EXCLUDED - the source mask is non-empty (apple / plate / cloth)",
}


class ProbeError(RuntimeError):
    """A refusal with a message meant for an operator, not a traceback."""


# ------------------------------------------------------------------------------------------------
# the cap
# ------------------------------------------------------------------------------------------------


def enforce_caps(*, episodes: int, styles: int, frames: int) -> dict:
    """Refuse anything above the registered cap. Returns the accepted shape.

    Refusing rather than clamping is the point. A clamp turns an operator asking for 4 000 frames
    into a run of 726 that looks like it did what was asked, and the log line that would have said
    otherwise is the one nobody reads. Every branch here names ``T40_RULE_V8`` so the refusal cites
    the rule rather than an opinion.
    """
    if episodes < 1 or styles < 1 or frames < 1:
        raise ProbeError(
            f"{RULE}: --episodes {episodes}, --style-count {styles} and --frames {frames} must all "
            "be at least 1; a probe of nothing measures nothing."
        )
    if frames > PROBE_MAX_FRAMES_PER_CLIP:
        raise ProbeError(
            f"{RULE} caps a probe clip at {PROBE_MAX_FRAMES_PER_CLIP} frames and {frames} was "
            f"asked for. The cap is a constant in {pathlib.Path(__file__).name} and there is no "
            f"flag, no environment variable and no config file that raises it. Raising it is a V9 "
            f"alongside {V8_DOC}, not an edit and not a submit-line argument."
        )
    if frames < PROBE_MIN_FRAMES_PER_CLIP:
        raise ProbeError(
            f"{RULE}: --frames {frames} is below {PROBE_MIN_FRAMES_PER_CLIP}. A shorter clip cannot "
            "carry enough paired frames to tell 'invented nothing' from 'measured nothing', which "
            f"{V8_DOC} §5.3 calls outcome U."
        )
    if episodes > PROBE_MAX_EPISODES:
        raise ProbeError(
            f"{RULE} caps the probe at {PROBE_MAX_EPISODES} episodes and {episodes} was asked for."
        )
    if styles > PROBE_MAX_STYLES:
        raise ProbeError(
            f"{RULE} caps the probe at {PROBE_MAX_STYLES} committed styles and {styles} was asked "
            "for."
        )
    clips = episodes * styles
    if clips > PROBE_MAX_CLIPS:
        raise ProbeError(
            f"{RULE} caps the probe at {PROBE_MAX_CLIPS} clips; {episodes} episodes x {styles} "
            f"styles is {clips}."
        )
    total = clips * frames
    if total > PROBE_MAX_TOTAL_FRAMES:
        raise ProbeError(
            f"{RULE} caps the probe at {PROBE_MAX_TOTAL_FRAMES} generated frames; {clips} clips x "
            f"{frames} frames is {total}. That cap is 1.70x the ONE episode PR-08 §8 item 3's "
            "licensed timing run restyles, and this probe is not a corpus."
        )
    return {
        "episodes": episodes,
        "styles": styles,
        "clips": clips,
        "frames_per_clip": frames,
        "total_generated_frames": total,
        "cap": {
            "max_episodes": PROBE_MAX_EPISODES,
            "max_styles": PROBE_MAX_STYLES,
            "max_clips": PROBE_MAX_CLIPS,
            "max_frames_per_clip": PROBE_MAX_FRAMES_PER_CLIP,
            "max_total_generated_frames": PROBE_MAX_TOTAL_FRAMES,
            "raisable_from_the_submit_line": False,
            "rule": RULE,
        },
    }


def require_quarantined_out(out: pathlib.Path) -> pathlib.Path:
    """``--out`` has to say what it holds. A probe tree is not a corpus tree and must not look like one."""
    if QUARANTINE_TOKEN not in out.name:
        raise ProbeError(
            f"{RULE}: --out {out} does not carry {QUARANTINE_TOKEN!r} in its directory name. This "
            "output is quarantined by construction and its path has to say so, so that nothing "
            "downstream can pick it up by looking like a corpus directory."
        )
    return out


# ------------------------------------------------------------------------------------------------
# frame selection — the diagnosis's own predicate, not a new one
# ------------------------------------------------------------------------------------------------


def reference_areas(frames: np.ndarray) -> tuple[list[int], list[int]]:
    """Per-frame reference-predicate area and largest component, at the PRIMARY committed setting.

    The primary setting is ``diagnose_robot_mask_empty``'s own first entry of each sweep tuple —
    imported rather than restated, so a change there cannot leave this file measuring a predicate
    the diagnosis does not describe.
    """
    background = background_median(frames)
    setting = {
        "dark_offset": DARK_OFFSETS[0],
        "sat_max": SAT_MAXES[0],
        "change_min": CHANGE_MINS[0],
    }
    areas: list[int] = []
    largest: list[int] = []
    for index in range(frames.shape[0]):
        mask = apply_setting(frame_fields(frames[index], background), **setting)
        areas.append(int(np.count_nonzero(mask)))
        largest.append(largest_component(mask))
    return areas, largest


def longest_absent_run(areas: list[int]) -> tuple[int, int]:
    """``(start, length)`` of the longest maximal run of ``absent`` frames; ``(0, 0)`` if none.

    A CONTIGUOUS run rather than a scatter of absent frames, and that is the whole reason this probe
    is a different measurement from the licensed timing run. Transfer2.5 conditions on a video: if a
    robot-free frame sits next to robot-present frames, a manipulator appearing in it could be
    temporal propagation from its neighbours rather than invention from nothing. A clip every one of
    whose conditioning frames is robot-free removes that confound. Stitching non-adjacent frames
    into a clip would remove it too and would also fabricate motion the corpus never had, so the run
    is taken as it occurs.
    """
    flags = [classify(a, absent_below=ABSENT_BELOW, present_above=PRESENT_ABOVE) == "absent"
             for a in areas]
    runs = runs_of(flags)
    if not runs:
        return (0, 0)
    start, length = max(runs, key=lambda item: item[1])
    return (int(start), int(length))


def select_episode(entry: dict, video: pathlib.Path, frames_per_clip: int) -> dict | None:
    """One episode's selection record, or ``None`` when its longest absent run is too short.

    ``None`` rather than a raise: an episode without a long enough robot-free run is not an error,
    it is an episode this probe does not use, and the scan moves on. The record keeps the numbers
    for the ones that were skipped too, so the artifact says which frames were chosen AND what was
    passed over.
    """
    frames = decode(video)
    areas, largest = reference_areas(frames)
    start, length = longest_absent_run(areas)
    record = {
        "episode": str(entry.get("id")),
        "video": str(video),
        "n_frames": int(frames.shape[0]),
        "longest_absent_run": {"start": start, "length": length},
        "eligible": bool(length >= frames_per_clip),
        "selected_frame_indices": [],
        "selection_reason": (
            "the longest CONTIGUOUS run of frames the reference predicate classifies absent, "
            f"band {ABSENT_BELOW}/{PRESENT_ABOVE} px at dark_offset {DARK_OFFSETS[0]} / sat_max "
            f"{SAT_MAXES[0]} / change_min {CHANGE_MINS[0]}"
        ),
    }
    if not record["eligible"]:
        return record | {"_frames": None}
    indices = list(range(start, start + frames_per_clip))
    record["selected_frame_indices"] = indices
    record["reference_px"] = [areas[i] for i in indices]
    record["reference_largest_component_px"] = [largest[i] for i in indices]
    # Asserted rather than assumed: every selected frame must classify `absent`, not merely "not
    # present". The run was built from that predicate, so this can only fire if the two disagree,
    # which would mean this file and the diagnosis are not running the same instrument.
    for i in indices:
        if classify(areas[i], absent_below=ABSENT_BELOW, present_above=PRESENT_ABOVE) != "absent":
            raise ProbeError(
                f"{record['episode']} frame {i} is inside the selected run and does not classify "
                "absent. The run finder and the classifier disagree, so the selection is not the "
                "one this probe registered."
            )
    return record | {"_frames": frames[start : start + frames_per_clip]}


# ------------------------------------------------------------------------------------------------
# the pairing — the measurement
# ------------------------------------------------------------------------------------------------


def pair_frame(source_mask: np.ndarray, generated_mask: np.ndarray) -> str:
    """Which of the three buckets this frame's (source, generated) mask pair falls in.

    The excluded bucket is not a failure and is not dropped. ``DIAGNOSIS.json`` measured the
    committed masker grounding the APPLE, the plate or the cloth on 41 % of robot-absent frames
    corpus-wide; on such a frame the source mask is non-empty for a reason that has nothing to do
    with a robot, and "the generated frame also grounds something" proves nothing either way.
    """
    if np.count_nonzero(source_mask):
        return PAIRING_EXCLUDED
    return PAIRING_CANDIDATE if np.count_nonzero(generated_mask) else PAIRING_BOTH_EMPTY


def mask_stats(mask: np.ndarray) -> dict:
    """Area, area fraction and bounding box of a mask — enough for a reader to size what grounded."""
    m = np.asarray(mask, dtype=bool)
    px = int(np.count_nonzero(m))
    out: dict[str, Any] = {"px": px, "frame_fraction": round(px / float(m.size), 6)}
    if px:
        rows = np.flatnonzero(m.any(axis=1))
        cols = np.flatnonzero(m.any(axis=0))
        out["bbox_rc"] = [int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])]
        out["largest_component_px"] = largest_component(m)
    return out


def unit_outcome(rows: list[dict]) -> str:
    """``H`` / ``N`` / ``U`` for one unit, by the definitions V8 §5 fixed BEFORE this ran.

    ``U`` is not a quiet ``N``. A unit whose paired population is too small measured the instrument,
    not the generator, and saying so is the only thing that keeps an empty result from being read as
    an absence.
    """
    paired = [r for r in rows if r["pairing"] != PAIRING_EXCLUDED]
    if len(paired) < MIN_PAIRED_PROBE_FRAMES:
        return "U"
    return "H" if any(r["pairing"] == PAIRING_CANDIDATE for r in paired) else "N"


# ------------------------------------------------------------------------------------------------
# the artifact a person looks at
# ------------------------------------------------------------------------------------------------


def _pair_tile(source: np.ndarray, generated: np.ndarray,
               source_mask: np.ndarray, generated_mask: np.ndarray) -> np.ndarray:
    """Source and generated side by side, each with its robot mask outlined in green.

    Same colour convention as ``diagnose_robot_mask_empty sheet --with-robot-mask``: green is the
    COMMITTED robot mask. The reference predicate's magenta is deliberately absent here — these
    frames were selected by it, so drawing it would decorate every tile with the thing that is
    already true of all of them.
    """
    from audit_apple_masks import boundary  # noqa: PLC0415

    pair = []
    for arr, mask in ((source, source_mask), (generated, generated_mask)):
        panel = np.asarray(arr, dtype=np.uint8).copy()
        edge = boundary(np.asarray(mask, dtype=bool), thickness=2)
        if edge.any():
            panel[edge] = (0, 255, 90)
        pair.append(panel)
    return np.concatenate(pair, axis=1)


def _sheet_tile(name: str, source: np.ndarray, generated: np.ndarray,
                selection: dict, row: dict) -> Any:
    """One captioned, half-size source|generated tile.

    Half size because a 640x480 pair is 1280x480 and twelve of them at full size is a sheet nobody
    opens twice. The caption carries both mask areas and the reference-predicate area, so a reader
    can see WHY this frame was in the probe at all without going back to the JSON.
    """
    from PIL import Image  # noqa: PLC0415

    from audit_apple_masks import captioned  # noqa: PLC0415

    i = row["frame_in_clip"]
    tile = _pair_tile(source[i], generated[i], selection["source_masks"][i], row["_gen_mask"])
    small = np.asarray(
        Image.fromarray(tile).resize((tile.shape[1] // 2, tile.shape[0] // 2))
    )
    return captioned(small, [
        f"{name}  clip f{i:04d}  source f{row['source_frame_index']:05d}",
        f"LEFT source mask {row['source_mask']['px']} px   |   "
        f"RIGHT generated mask {row['generated_mask']['px']} px",
        f"reference predicate {row['reference_px']} px (absent, band "
        f"{ABSENT_BELOW}/{PRESENT_ABOVE})",
        "green = COMMITTED robot mask. NOT a verdict: a person decides whether it is a manipulator.",
    ], flagged=(row["pairing"] == PAIRING_CANDIDATE))


def write_sheet(tiles: list, title: str, path: pathlib.Path) -> str | None:
    """A contact sheet, or ``None`` when the stratum is empty. Empty strata are a finding too."""
    from audit_apple_masks import contact_sheet  # noqa: PLC0415

    if not tiles:
        return None
    sheet = contact_sheet(tiles, title, cols=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return str(path)


def evenly(items: list, k: int) -> list:
    """``k`` items spread across ``items``, not its prefix.

    A prefix of a probe clip is the first second of one episode's approach; the spread is what makes
    a twelve-tile sheet describe the clip rather than its opening.
    """
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return list(items)
    step = len(items) / float(k)
    return [items[min(len(items) - 1, int(i * step))] for i in range(k)]


# ------------------------------------------------------------------------------------------------
# nothing downstream may read this tree
# ------------------------------------------------------------------------------------------------


def audit_output_tree(root: pathlib.Path) -> list[str]:
    """Every path under ``root`` that would make this tree consumable. Empty list is the pass.

    Walked rather than asserted. ``assemble_restyled_lerobot.py`` files a clip directory by
    ``glob("*.mp4")`` and 97's harvest keys on ``vision.mp4``; a guarantee that rests on every
    future edit remembering that is weaker than one that rests on looking.
    """
    offenders: list[str] = []
    for path in sorted(pathlib.Path(root).rglob("*")):
        if not path.is_file():
            continue
        if path.name in FORBIDDEN_NAMES or path.name.endswith(FORBIDDEN_SUFFIXES):
            offenders.append(str(path))
    return offenders


NOT_A_CORPUS = """\
NOT A CORPUS. NOT TRAINING DATA. NOT AN EVALUATION SET.

Everything in this directory was produced by scripts/probe_hallucination.py under
docs/preregistration/PR-08-V8-hallucination-probe.md, rule T40_RULE_V8, which licenses ONE thing:
finding out whether Cosmos-Transfer2.5 puts a manipulator into a source frame that does not contain
one. V8 §4 forbids, explicitly:

  * any frame here entering a training set, a LeRobot dataset, an evaluation set, or any artifact a
    downstream consumer reads;
  * any number here being quoted as a PR-08 result — it is not P, F, N or I, not L1 or L2, not a
    GEOM_TOL, not an EST_DRIFT_P95 and not a robot-mask area bound;
  * this run being used to satisfy PR-08 §8 item 3's throughput measurement, which remains the
    timing run's job alone.

T40_RULE_V1 §1's prohibition on generating a corpus is UNCHANGED and binds in full.

The generated video is stored with a .mp4.quarantined suffix on purpose: assemble_restyled_lerobot.py
files a clip directory by glob("*.mp4") and 97_transfer25_restyle.sbatch's harvest keys on a file
called vision.mp4. The bytes are readable by both decoders this repository uses; the two globs that
could file them are not.

The verdict letter in PROBE.json is NOT the finding until a person has looked at the sheets:
human_review.looked_at is false, and the candidate count is an UPPER BOUND, because the committed
masker grounds the apple on 41 % of robot-absent SOURCE frames (runs/pr08-robot-mask-empty/DIAGNOSIS.json).
"""


# ------------------------------------------------------------------------------------------------
# the run
# ------------------------------------------------------------------------------------------------


def load_train_styles(path: pathlib.Path, count: int) -> list[dict]:
    """The first ``count`` committed TRAIN_STYLES, in committed id order.

    In committed order and never sorted by anything else, because a probe that picked its styles
    after looking at which ones came out well is the failure the committed partition exists to
    prevent. There is no flag that selects a different set; :data:`PROBE_STYLE_SET` is a constant.
    """
    styles = rt.load_styles(path, PROBE_STYLE_SET)
    chosen = [styles[k] for k in sorted(styles)][:count]
    if len(chosen) < count:
        raise ProbeError(
            f"{path} carries {len(chosen)} {PROBE_STYLE_SET} styles; {count} were asked for."
        )
    for style in chosen:
        seeds = style.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ProbeError(
                f"style {style.get('id')!r} carries no committed seeds list. This probe coins no "
                "seed: T40_RULE_V3 committed one per style-instance and that is the one it uses."
            )
    return chosen


def generate(sample: dict, out_dir: pathlib.Path, setup: dict, backend: str) -> dict:
    """One generated clip, immediately renamed out of every downstream glob's way.

    The rename is in a ``finally`` because the window between the framework writing ``<name>.mp4``
    and this function returning is the only moment a consumable filename exists on disk, and a crash
    inside it must not leave one behind. :func:`audit_output_tree` is the backstop, not the plan.
    """
    produced = out_dir / "vision.mp4"
    try:
        record = (
            rt._null_backend(sample, out_dir) if backend == "null"
            else rt._transfer25_backend(sample, out_dir, setup)
        )
    finally:
        if produced.is_file():
            produced.replace(out_dir / (str(sample["name"]) + QUARANTINE_SUFFIX))
        stray = out_dir / f"{sample['name']}.mp4"
        if stray.is_file():
            stray.replace(out_dir / (str(sample["name"]) + QUARANTINE_SUFFIX))
    record["clip"] = str(out_dir / (str(sample["name"]) + QUARANTINE_SUFFIX))
    return record


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", required=True, type=pathlib.Path)
    ap.add_argument("--styles", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path,
                    help=f"a run directory whose name carries {QUARANTINE_TOKEN!r}")
    ap.add_argument("--checkpoint-path", required=True)
    ap.add_argument("--control", required=True,
                    help="e.g. 'depth:0.5,seg:0.5'. Required and never defaulted, for the reason "
                         "restyle_transfer25.py refuses to default it: the choice decides how much "
                         "geometry survives, and V8 §7 item 6 records that it moves this answer.")
    ap.add_argument("--resolution", default="640x480")
    ap.add_argument("--episodes", type=int, default=2,
                    help=f"how many source episodes to draw a robot-free run from (cap {PROBE_MAX_EPISODES})")
    ap.add_argument("--style-count", type=int, default=2,
                    help=f"how many committed TRAIN_STYLES to run (cap {PROBE_MAX_STYLES})")
    ap.add_argument("--frames", type=int, default=96,
                    help=f"frames per probe clip (cap {PROBE_MAX_FRAMES_PER_CLIP})")
    ap.add_argument("--tiles", type=int, default=12, help="tiles per contact sheet")
    ap.add_argument("--no-guardrails", action="store_true")
    ap.add_argument("--backend", choices=("transfer25", "null"), default="transfer25")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    try:
        if not args.no_guardrails:
            raise ProbeError(
                "--no-guardrails is mandatory here for the same reason it is in the driver: the "
                "guardrail's RetinaFace postprocessor rewrites frames and blurs the hand, and the "
                "rewritten frames are what land on disk (docs/transfer25-api.md §3). A probe about "
                "whether a manipulator appears must not run through a stage that erases hands."
            )
        shape = enforce_caps(
            episodes=args.episodes, styles=args.style_count, frames=args.frames
        )
        out = require_quarantined_out(args.out)
        bucket, aspect = rt.resolve_resolution(args.resolution)
        controls = rt.parse_controls(args.control)
        styles = load_train_styles(args.styles, args.style_count)
        entries = read_manifest(args.manifest)
        source_root = args.manifest.parent

        out.mkdir(parents=True, exist_ok=True)
        (out / "NOT_A_CORPUS").write_text(NOT_A_CORPUS, encoding="utf-8")
        clips_dir = out / "probe_clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        masker = robot_composite.build_masker()
        masker.preflight()

        # -- select the episodes, then write each one's robot-free run as its own short clip -------
        # The SOURCE masks are taken on the frames decoded back out of that written clip, not on the
        # manifest clip's frames, because the written clip is what the generator sees. Comparing a
        # mask made on one encoding against a mask made on another would put the re-encode inside
        # the measurement.
        selections: list[dict] = []
        scanned = 0
        for entry in entries:
            if len([s for s in selections if s["eligible"]]) >= args.episodes:
                break
            if scanned >= PROBE_MAX_SCAN_EPISODES:
                break
            scanned += 1
            record = select_episode(
                entry, source_root / str(entry["video"]), args.frames
            )
            frames = record.pop("_frames")
            if frames is None:
                selections.append(record)
                print(f"skip {record['episode']}: longest absent run "
                      f"{record['longest_absent_run']['length']} < {args.frames}", flush=True)
                continue
            clip = clips_dir / f"{record['episode']}.probe-source.mp4"
            fps = robot_composite.container_fps(source_root / str(entry["video"])) or 30.0
            robot_composite.encode_clip(frames, clip, fps)
            quarantined = clip.with_name(clip.stem + QUARANTINE_SUFFIX)
            clip.replace(quarantined)
            record["probe_clip"] = str(quarantined)
            record["probe_clip_fps"] = fps
            record["_decoded"] = robot_composite.decode_clip(quarantined)
            selections.append(record)
            print(f"take {record['episode']}: frames "
                  f"{record['selected_frame_indices'][0]}..{record['selected_frame_indices'][-1]} "
                  f"-> {quarantined.name}", flush=True)

        eligible = [s for s in selections if s["eligible"]]
        if len(eligible) < args.episodes:
            raise ProbeError(
                f"{RULE}: {len(eligible)} of {args.episodes} episodes scanned had a contiguous "
                f"robot-free run of at least {args.frames} frames in the first {scanned} manifest "
                "entries. Lower --frames (never above the cap) or accept fewer episodes; do NOT "
                "widen the scan by loosening the band, which is a constant for that reason."
            )

        for selection in eligible:
            decoded = selection["_decoded"]
            selection["source_masks"] = [
                np.asarray(masker.mask(decoded[i]), dtype=bool) for i in range(decoded.shape[0])
            ]
            empty = sum(1 for m in selection["source_masks"] if not m.any())
            selection["source_mask_empty_frames"] = empty
            selection["source_mask_empty_fraction"] = round(empty / float(decoded.shape[0]), 4)
            print(f"{selection['episode']}: source mask empty on {empty}/{decoded.shape[0]} frames",
                  flush=True)

        # -- generate, pair, and write the sheets ---------------------------------------------------
        setup = {
            "output_dir": out / "raw",
            "disable_guardrails": True,
            "checkpoint_path": args.checkpoint_path,
        }
        units: list[dict] = []
        # The serialisable half of the selection records, taken once. The rest of each record holds
        # decoded frames and boolean masks, and neither goes into an artifact.
        public_selection = [
            {k: v for k, v in s.items() if not k.startswith("_") and k != "source_masks"}
            for s in selections
        ]

        def build_payload(complete: bool) -> dict:
            outcomes = {u["outcome"] for u in units}
            covered_episodes = {u["episode"] for u in units}
            covered_styles = {u["style"] for u in units}
            # V8 §5.2 defines outcome N as zero candidates across AT LEAST two distinct episodes and
            # two distinct committed styles. One episode under one prompt finding nothing is not
            # "the generator invents nothing", it is one draw, so it reports U.
            thin = len(covered_episodes) < 2 or len(covered_styles) < 2
            # A partial run has not answered either. It reports U rather than the letter its
            # finished units happen to agree on, because "every unit that got as far as running said
            # N" and "the probe found nothing" are different claims, and only the second is an
            # answer. An H, by contrast, does NOT need the coverage: one confirmed invented
            # manipulator is a finding about the premise on its own.
            if "H" in outcomes:
                overall = "H"
            elif not complete or not units or "U" in outcomes or thin:
                overall = "U"
            else:
                overall = "N"
            return {
                "schema": SCHEMA,
                "rule": RULE,
                "pre_registration": V8_DOC,
                "question": ("Can Cosmos-Transfer2.5 hallucinate a manipulator into a robot-free "
                             "frame?"),
                "produced_by": "scripts/probe_hallucination.py",
                "produced_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "complete": bool(complete),
                "elapsed_seconds": round(time.time() - started, 1),
                "elapsed_seconds_note": (
                    "an operational fact about this probe. NOT admissible for PR-08 §8 item 3, "
                    "which requires one timed EPISODE on an H200 at 640x480 through the corpus "
                    "path; V8 §4 item 3 forbids the substitution and item 3 stays open."
                ),
                "verdict": overall,
                "coverage": {
                    "episodes": sorted(covered_episodes),
                    "styles": sorted(covered_styles),
                    "meets_outcome_N_coverage": not thin,
                    "note": ("V8 §5.2: an N needs at least two distinct episodes and two distinct "
                             "committed styles. Below that the run reports U — one episode under "
                             "one prompt finding nothing is one draw, not an absence."),
                },
                "verdict_definitions": {
                    "H": "at least one candidate-invention frame; V8 §5.1. Licenses nothing.",
                    "N": "zero candidate-invention frames with enough paired frames; V8 §5.2. "
                         "Licenses nothing, and does NOT bound the rate: 726 frames cannot.",
                    "U": "the instrument, not the generator, is what was measured, or the run did "
                         "not finish; V8 §5.3. U is not a quiet N.",
                },
                "human_review": {
                    "looked_at": False,
                    "note": ("The candidate count is an UPPER BOUND. The committed masker grounds "
                             "the APPLE on 41 % of robot-absent SOURCE frames corpus-wide "
                             "(runs/pr08-robot-mask-empty/DIAGNOSIS.json). Whether a candidate "
                             "frame's mask is on a MANIPULATOR is a person's reading of the "
                             "sheets, and this field stays false until that has happened."),
                    "read_first": sorted(
                        u["sheets"][PAIRING_CANDIDATE] for u in units
                        if PAIRING_CANDIDATE in u["sheets"]
                    ),
                },
                "licensed": {
                    "generation_of_a_corpus": False,
                    "training": False,
                    "satisfies_pr08_section8_item3": False,
                    "derives_a_g0c_area_bound": False,
                    "note": ("T40_RULE_V1 §1 is unchanged and binds in full. No number in this "
                             "file is a PR-08 result."),
                },
                "shape": shape,
                "instrument": {
                    "reference_predicate": (
                        "diagnose_robot_mask_empty.robot_dark_mask, primary setting "
                        f"(dark_offset {DARK_OFFSETS[0]}, sat_max {SAT_MAXES[0]}, "
                        f"change_min {CHANGE_MINS[0]}), band {ABSENT_BELOW}/{PRESENT_ABOVE} px "
                        "from DIAGNOSIS.json instrument.band_px. NOT ground truth; V8 §7 item 1."
                    ),
                    "robot_masker": masker.provenance(),
                    "check_mask_called": False,
                    "check_mask_note": ("G0c's refusal is the premise this measures; calling the "
                                        "gate would refuse every unit before a number existed."),
                    "generator": {
                        "backend": args.backend,
                        "checkpoint_path": args.checkpoint_path,
                        "resolution": args.resolution,
                        "bucket": bucket,
                        "aspect": aspect,
                        "controls": [{"key": c.key, "weight": c.weight} for c in controls],
                        "control_maps": ("estimated in-framework; the probe clips carry no depth "
                                         "or segmentation map (docs/transfer25-api.md §8)"),
                    },
                    "style_set": PROBE_STYLE_SET,
                    "styles": [{"id": s["id"], "seed": int(s["seeds"][0])} for s in styles],
                    "source_manifest": str(args.manifest),
                    "style_partition": str(args.styles),
                },
                "selection": public_selection,
                "units": units,
                "totals": {
                    "clips": len(units),
                    "generated_frames": sum(u["generated_frames"] for u in units),
                    "paired_frames": sum(u["paired_frames"] for u in units),
                    "candidate_frames": sum(u["counts"][PAIRING_CANDIDATE] for u in units),
                    "both_empty_frames": sum(u["counts"][PAIRING_BOTH_EMPTY] for u in units),
                    "excluded_frames": sum(u["counts"][PAIRING_EXCLUDED] for u in units),
                },
            }

        def snapshot(complete: bool) -> dict:
            """Write PROBE.json as it stands. Called after EVERY unit, not only at the end.

            The wall is the one failure this job cannot argue with, and no Cosmos-Transfer2.5
            throughput has ever been measured on this project — the one published number came from
            a run that generated zero clips — so the Slurm `--time` is a request rather than an
            estimate. A probe that wrote its artifact last would turn a wall kill into a total loss
            of the units that DID finish. `complete` is what keeps a partial file from being read
            as an answer.
            """
            payload = build_payload(complete)
            (out / "PROBE.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            return payload

        snapshot(False)

        for selection in eligible:
            for style in styles:
                name = f"{selection['episode']}__{style['id']}__probe"
                unit_dir = out / "units" / name
                unit_dir.mkdir(parents=True, exist_ok=True)
                clip = pathlib.Path(selection["probe_clip"])
                work = rt.WorkUnit(
                    unit=name,
                    episode=selection["episode"],
                    frames=len(selection["selected_frame_indices"]),
                    style=str(style["id"]),
                    repeat=0,
                    seed=int(style["seeds"][0]),
                )
                sample = rt.build_sample(
                    work,
                    source_root=clip.parent,
                    # A synthetic manifest entry, so the payload is built by the SAME function the
                    # corpus path uses rather than by a second hand-rolled dict that could drift
                    # from it. There is no depth or segmentation map to name, so none is passed and
                    # Transfer2.5 estimates its own control maps — the api §8 behaviour the driver
                    # documents. That is a difference from a run whose manifest carried maps, and it
                    # is recorded rather than hidden.
                    episode={"video": clip.name},
                    style=style,
                    controls=controls,
                    bucket=bucket,
                )
                started_unit = time.time()
                record = generate(sample, unit_dir, setup, args.backend)
                elapsed = time.time() - started_unit
                generated = robot_composite.decode_clip(pathlib.Path(record["clip"]))
                source = selection["_decoded"]

                rows: list[dict] = []
                detail = ""
                if generated.shape[0] != source.shape[0]:
                    # V8 §5.3: a clip whose frame count does not match its input is outcome U for
                    # this unit, not a silent truncation of the pairing. The frames are paired BY
                    # INDEX — there is no alignment step here and there must not be one.
                    outcome = "U"
                    detail = (f"the generator returned {generated.shape[0]} frames for a "
                              f"{source.shape[0]}-frame input; the frames cannot be paired by "
                              "index")
                else:
                    for i in range(source.shape[0]):
                        src_mask = selection["source_masks"][i]
                        gen_mask = np.asarray(masker.mask(generated[i]), dtype=bool)
                        rows.append({
                            "frame_in_clip": i,
                            "source_frame_index": selection["selected_frame_indices"][i],
                            "reference_px": selection["reference_px"][i],
                            "reference_largest_component_px":
                                selection["reference_largest_component_px"][i],
                            "source_mask": mask_stats(src_mask),
                            "generated_mask": mask_stats(gen_mask),
                            "iou": (round(robot_composite.mask_iou(src_mask, gen_mask), 6)
                                    if (src_mask.any() or gen_mask.any()) else None),
                            "pairing": pair_frame(src_mask, gen_mask),
                            "_gen_mask": gen_mask,
                        })
                    outcome = unit_outcome(rows)

                sheets: dict[str, str] = {}
                for bucket_name, blurb in STRATA.items():
                    picked = evenly([r for r in rows if r["pairing"] == bucket_name], args.tiles)
                    tiles = [
                        _sheet_tile(name, source, generated, selection, row)
                        for row in picked
                    ]
                    written = write_sheet(
                        tiles,
                        f"{name} | {blurb} | robot prompt "
                        f"{robot_composite.ROBOT_TEXT_PROMPT!r}",
                        out / "sheets" / f"{name}__{bucket_name}.png",
                    )
                    if written:
                        sheets[bucket_name] = written

                counts = {k: sum(1 for r in rows if r["pairing"] == k) for k in STRATA}
                for row in rows:
                    row.pop("_gen_mask", None)
                units.append({
                    "unit": name,
                    "episode": selection["episode"],
                    "style": str(style["id"]),
                    "style_prompt": style["prompt"],
                    "seed": int(style["seeds"][0]),
                    "backend": record.get("backend"),
                    "clip": record["clip"],
                    "generated_frames": int(generated.shape[0]),
                    "elapsed_seconds": round(elapsed, 2),
                    "counts": counts,
                    "paired_frames": counts[PAIRING_BOTH_EMPTY] + counts[PAIRING_CANDIDATE],
                    "outcome": outcome,
                    "outcome_detail": detail,
                    "sheets": sheets,
                    "frames": rows,
                    "generator": {k: record[k] for k in
                                  ("checkpoints_loaded", "checkpoint_path_honoured", "digest")
                                  if k in record},
                })
                snapshot(False)
                print(f"{name}: {counts} -> outcome {outcome}", flush=True)

        payload = snapshot(True)

        offenders = audit_output_tree(out)
        if offenders:
            raise ProbeError(
                f"{RULE} §4 item 1: this probe wrote {len(offenders)} file(s) a downstream "
                "consumer could read, which it is forbidden to do:\n       "
                + "\n       ".join(offenders)
                + "\n       assemble_restyled_lerobot.py files a clip directory by glob('*.mp4') "
                "and 97's harvest keys on vision.mp4. Nothing here may be consumable."
            )
        print(f"=== verdict {payload['verdict']} | {payload['totals']}")
        print(f"=== wrote {out / 'PROBE.json'}")
        print("=== THE VERDICT IS NOT THE FINDING UNTIL A PERSON HAS READ THE SHEETS.")
        return 0
    except (ProbeError, rt.DriverError, robot_composite.CompositeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
