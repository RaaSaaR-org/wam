#!/usr/bin/env python
"""Measure ``EST_DRIFT_P95`` — the estimator error budget PR-08 §6 G0b subtracts from ``GEOM_TOL``.

    EST_DRIFT_P95 := the 95th percentile of the object-centroid displacement, in pixels, between
                     the ESTIMATED segmentation and the TRUE segmentation of the same frame.

PR-08 §4 spells the procedure out in five steps, and this script is steps 1-4:

    0. Attach `distance_to_camera` and `semantic_segmentation` in `isaac_binding.py`.  <- landed
       2026-08-21, commit 5ef3535. Until then step 1 could not run at all.
    1. Render N Isaac episodes with ground-truth depth + segmentation.                 <- `capture`
    2. Run the same monocular depth estimator and the same segmenter on the RGB ONLY.  <- `measure`
    3. Record the error distribution: absolute depth error, and object-centroid displacement in
       pixels between the estimated and the true segmentation.                         <- `measure`
    4. The 95th percentile of that centroid displacement is EST_DRIFT_P95.             <- `measure`

WHY IT IS TWO SUBCOMMANDS AND NOT ONE RUN
-----------------------------------------
Step 1 needs Isaac Sim: Linux, an NVIDIA GPU, and a stage that has never been booted in this
project -- every Isaac test to date runs against ``FakeIsaacBinding``. Steps 2-4 need an estimator
and a segmenter and no simulator at all. Fusing them would make the arithmetic untestable anywhere
Isaac cannot boot, and would re-render the corpus every time the estimator changed. So ``capture``
writes ground truth to disk once, and ``measure`` is a pure function of that directory. The capture
is also the artifact that makes the number auditable later: it is what "the true segmentation" was.

WHAT THIS REFUSES TO DO
-----------------------
**It will not invent an estimator.** Neither the monocular depth estimator nor the segmenter is
wired in this repo. The failure names every package it looked for, exactly as
``measure_geom_tol.no_segmenter_message()`` does, and writes nothing. A plausible stand-in here is
worse than in ``measure_geom_tol``: this number is SUBTRACTED from the tolerance, so an
underestimate silently *widens* the gate and every G0b pass inherits the slack.

**It will not use a different segmenter from GEOM_TOL's, and "different" is not judged by the
name.** §4 step 2 says "the *same* segmenter", and §6 subtracts the two numbers. Two segmenters is
two different quantities and the subtraction is not arithmetic. The committed artifact
``configs/transfer25/pr08_geom_tol.json`` carries a ``segmenter`` block — the prompt, both detection
thresholds, the single retry pair, the box-selection rule, the propagation mode and the two
checkpoint pins — and
``cross_check_geom_tol`` compares it field for field against the estimator module's own
``SEGMENTER_CONTRACT``. Any disagreement disqualifies. The method NAME is compared as well and used
to be the whole of the check, which was the weakest possible version of it: the identical adapter at
``box_threshold`` 0.35 and at 0.15 detects on different frames, produces different centroids, and
reports the same ``ESTIMATOR_NAME`` throughout.

WHERE THAT BLOCK LIVES IS DECIDED IN ONE PLACE, NOT TWO. ``committed_segmenter_contract`` and
``contract_disagreements`` are imported from ``measure_geom_tol`` — the module that WRITES the
document — rather than restated here. A reader and a writer that each keep their own idea of where
the block lives is how a cross-check comes to look somewhere empty and pass: it would compare an
absent dict against an absent dict and report a clean check. The same functions therefore answer
"where is it" for the producer's overwrite guard and for this consumer's cross-check.

**It will not let the object be named twice.** ``--object-class`` defaults to the estimator's own
``OBJECT_TEXT_PROMPT``, and an explicitly typed value that names a different object is FATAL rather
than measured. An apple mask scored against a plate's ground truth produces a large, plausible p95,
no crash and no coverage drop — and that p95 is subtracted from ``GEOM_TOL``. An estimator that
declares no prompt cannot be checked this way and the run carries
``estimator_does_not_declare_object_prompt``.

**It will not compare pixels across resolutions.** Same reason, same rule as ``measure_geom_tol``:
GEOM_TOL is measured at the source grid, so a drift measured on a differently-sized Isaac render is
not subtractable from it. Mismatch disqualifies.

**It will not let a partial run become the gate.** ``--limit`` exists to exercise the pipeline in
seconds and is exactly the shape of a silent corruption, so any non-zero value forces
``gate_qualified: false`` and exit 3 -- the same rule, for the same reason, as ``measure_geom_tol``.

**It will not fold a missing object into the error as a zero.** A frame where the true mask has no
object (occluded by the Dex3 hand, or out of frame) has no centroid and therefore no paired
displacement. It is DROPPED and COUNTED. Folding it in as 0 px would pull the p95 down, which
*widens* G0b, which looks conservative and is the opposite.

THE NUMBER IS A BOUND, ITS DIRECTION IS PER-ROUTE, AND THE ARTIFACT SAYS WHICH
------------------------------------------------------------------------------
PR-08 §4's stated weakness, unedited: *"Isaac frames are not real frames, and a monocular
estimator's error on synthetic renders is not its error on RealSense footage -- plausibly
optimistic. So EST_DRIFT_P95 is a lower bound on the real error, it is recorded as such, and a G0b
margin that only clears under a lower bound is not a pass."*

``is_lower_bound`` used to be stamped ``true`` unconditionally. **It is still not a flag** -- no
command line can move it -- but since PR-08-V5 (``T40_RULE_V5``, 2026-08-22) it is looked up from
:data:`GROUND_TRUTH_BINDINGS` by the CLASS NAME of the binding the capture header records, because
§4's ground-truth source is now "a simulator with ground-truth segmentation" rather than Isaac
specifically and the two routes' errors are argued to point in OPPOSITE directions. The Isaac row
is byte-for-byte the old stamp; **anything not in that table falls back to it verbatim**, so the
path that already existed produces the output it always did. Read that table, not this paragraph,
for which route says what -- and read ``error_direction`` beside the boolean, because
``is_lower_bound: false`` on its own would read as "so it is an upper bound", which no route here
has earned.

What is unchanged in kind: the number is a BOUND, its direction is RECORDED, and a G0b margin that
only clears under a bound pointing the wrong way is not a pass.

WHAT THE ESTIMATOR SAW, RECORDED BESIDE THE BUDGET
--------------------------------------------------
The artifact carries an ``estimator_stats`` block, in the same shape and built by the same code as
``measure_geom_tol``'s (``EstimatorStatsProbe``, imported from it): what this run did to the
adapter's counters -- frames with no detection, empty masks, retries fired and recovered -- and the
distribution of the winning detection scores, with the raw values beside it because a capture is a
few hundred frames rather than a corpus. The adapter's counters are cumulative over its import, so
they are snapshotted before the first frame and differenced after the last; an estimator that
exports no ``stats()`` records an ABSENCE WITH A REASON rather than zeros, since the contract this
harness enforces is ``segment(rgb)`` / ``estimate_depth(rgb)`` and nothing more.

It is additive: nothing reads it back, no disqualification reason depends on it, and recording the
evidence the adapter's second gate-qualification blocker asks for is not the same act as accepting
it. That one is a human's.

TWO ARMS, ONE CAPTURE, AND THE CONFOUND BETWEEN THEM
-----------------------------------------------------
``apple_sam2``'s THIRD gate-qualification blocker is the last difference between this adapter and
the segmenter Cosmos-Transfer2.5 actually runs: upstream drives
``SAM2VideoPredictor.init_state(video_path=...)`` and PROPAGATES one mask across the clip, while
``segment(rgb)`` re-detects and re-segments every frame independently. The blocker names its own
discharge condition — *the same capture measured BOTH ways, the two p95s recorded side by side* —
and ``measure --arm both`` is that measurement. ``--arm`` DEFAULTS to ``per_frame``: a run that
does not pass it is the run this script has always performed, in every number and every field, and
no artifact already on disk means anything different.

**The confound that would have made it worthless.** ``SAM2VideoPredictor`` conventionally ingests a
directory of **JPEG** frames and this project's captures are lossless ``rgb.npy``. Had the
propagation arm been driven from a transcode, the difference between the two p95s would have been
the codec plus propagation — reported as propagation, with nothing in the artifact looking wrong.
So both arms are handed the SAME in-memory array, the propagation module ingests arrays rather than
files (bitwise upstream's own ingest, asserted in
``tests/test_apple_sam2_video_propagation.py``), and every run RECORDS a per-frame digest of what
each arm was shown: ``arm_comparison.identical_input_pixels``.

**And the statistic that is the actual point.** The two p95s answer the blocker's limb (a). Its
limb (b) — propagation drifting off the object and STAYING off for a run of frames — does not show
up in a p95 at all, because a per-frame arm scattering the same count of bad frames across a
capture and a propagation arm losing the object for a contiguous stretch produce the same
distribution. They produce different RUNS, so each arm's longest run of consecutive frames below
``LOW_IOU_THRESHOLD`` ground-truth IoU, and the count of such runs, are recorded beside them.

**It discharges nothing.** ``GATE_QUALIFIED`` and ``GATE_QUALIFICATION_BLOCKERS`` are untouched by
any run of this script, ``estimator_not_gate_qualified`` is still stamped, and the artifact says so
in the block itself. Producing the evidence a blocker names is not the act of closing it.

EXIT STATUS
-----------
0   measured with a gate-qualified estimator pair, coverage above ``--min-coverage``.
2   fatal: nothing was measured (no estimator, no capture, no object label, mixed geometry, or
    ``--object-class`` naming a different object from the segmenter's prompt).
3   measured, but the number MUST NOT be used as G0b's budget -- ungated estimator, coverage below
    the floor, a partial run, a segmenter that disagrees with GEOM_TOL's, or a resolution that does.
    The artifact is still written, because "we tried and this is what came out" is a record.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from measure_geom_tol import (  # noqa: E402
    CANDIDATE_SEGMENTERS,
    # WHAT THE ESTIMATOR SAW, recorded beside what this measured, in the same shape and by the same
    # code as the GEOM_TOL half — the two artifacts are read side by side by whoever has to judge
    # the adapter's second gate-qualification blocker, and two spellings of the same evidence is
    # how one of them comes to be read as the other. Snapshot-and-difference, because the adapter's
    # counters are lifetime totals; see EstimatorStatsProbe.
    EstimatorStatsProbe,
    _importable,
    _local_weight_hits,
    centroid_of_mask,
    # ONE implementation of "where the committed segmenter block lives" and "on which fields do
    # two of them disagree", imported from the PRODUCER rather than restated here. Two copies of
    # this pair is how a cross-check comes to look somewhere the writer does not write: the reader
    # would still pass, having compared an empty dict against an empty dict.
    committed_segmenter_contract,
    contract_disagreements,
    distribution,
)

# The camera map the binding itself defaults to, imported rather than restated. `capture`'s
# --camera used to default to "ego", a name no default Isaac stage carries, so the DEFAULT value
# raised `unknown camera 'ego'` after a full Isaac boot. A default that comes from the same dict
# the binding validates against cannot be wrong in that way, and an unknown name is now an
# argparse error on a laptop instead. Torch-free, numpy-only module: importing it costs nothing
# and it does not pull Isaac in (every isaacsim import lives inside IsaacSimBinding.__init__).
from wam.robot.isaac_binding import DEFAULT_CAMERA_PRIMS  # noqa: E402

SCHEMA = "wam.est_drift/1"
WRITEUP = "docs/preregistration/PR-08-photoreal-augmentation.md"

#: The ``--schedule`` choices, restated here so that ``--help`` and an argparse error work on a box
#: with no MuJoCo installed — this module must stay importable and refusable without the optional
#: `sim` extra, which is why every ``wam.robot.mujoco_binding`` import in it is inside ``main``.
#: **Restated, not forked**: the mujoco branch checks this tuple against that module's own
#: ``SCENE_SCHEDULES`` and REFUSES the run if they have drifted, so a schedule added in one place
#: and not the other is a loud failure rather than a choice argparse silently rejects. ``lattice``
#: is first and is the default everywhere.
SCENE_SCHEDULE_NAMES: tuple[str, ...] = ("lattice", "trajectory")
DEFAULT_SCENE_SCHEDULE = "lattice"
DEFAULT_SCENE_STATES = 20

#: ``trajectory_scene_schedule``'s three cycle counts, exposed on the command line as
#: ``T40_RULE_V17`` §2 registers, with one line each saying what the axis does geometrically.
#:
#: **THEY ARE NOT ENVELOPE PARAMETERS AND THIS IS THE REASON THEY MAY BE FLAGS AT ALL.** V5 §4.5
#: registers that any change raising the object's visibility — the placements it visits, the cube
#: distractor, the occluding hands — must be argued in a V-document rather than made in a commit,
#: and ``mujoco_binding`` withholds the centre and the radii from its own signature for exactly
#: that reason. These three count cycles over a path whose centre, radii and arm amplitude are
#: derived constants: every pose any value of them reaches is a pose the default also reaches, so
#: they move WHEN the object is somewhere, never WHERE it can be. V17 §2 states the claim in the
#: open so a reader who disagrees can reject it there instead of finding it in a docstring.
#:
#: The DEFAULTS ARE NEVER RESTATED HERE. They are read off the function's own signature by
#: :func:`trajectory_schedule_params`, because a second copy of ``2.0`` is how the header would
#: come to record a number the schedule did not use.
TRAJECTORY_PARAM_FLAGS: dict[str, str] = {
    "turns": "complete revolutions of the object around the closed ellipse in the table plane.",
    "yaw_turns": "complete revolutions of the object's own yaw.",
    "arm_cycles": "complete sweeps of the shoulder-pitch offset — i.e. how many times the Dex3 "
    "hands cross in front of the object, which is the event a propagated mask is lost at.",
}

#: The COMMITTED gate artifact, beside GEOM_TOL's and for the same reason: §8 item 4 wants both
#: measured *and committed* before generation, and a path under gitignored ``runs/`` cannot be that.
DEFAULT_OUT_REL = "configs/transfer25/pr08_est_drift.json"
DEFAULT_OUT = _REPO_ROOT / DEFAULT_OUT_REL

#: GEOM_TOL's artifact. Read to cross-check the segmenter and the pixel grid, never written.
GEOM_TOL_ARTIFACT = _REPO_ROOT / "configs/transfer25/pr08_geom_tol.json"

#: THE TWO ARMS OF ``apple_sam2``'s THIRD GATE-QUALIFICATION BLOCKER.
#:
#: The blocker names its own discharge condition: *"measuring the same Isaac capture BOTH ways —
#: this adapter per frame, and the video predictor propagating from frame 0 — and recording the two
#: p95s"*. ``per_frame`` is what this script has always done and is the DEFAULT, so a run that
#: passes no ``--arm`` is byte-for-byte the run it was before the flag existed and no artifact
#: already on disk means anything different. ``propagation`` drives a clip-level video predictor,
#: which ``segment(rgb)`` cannot host, so it lives in its own module behind
#: ``--propagation-module``. ``both`` is the comparison the blocker asks for.
ARM_CHOICES: tuple[str, ...] = ("per_frame", "propagation", "both")
DEFAULT_ARM = "per_frame"

#: The propagation arm's module. Named rather than imported, exactly as ``--estimators`` is: this
#: script must stay importable and testable on a box with no ``sam2`` and no weights.
DEFAULT_PROPAGATION_MODULE = "estimators.apple_sam2_video"

#: What counts as "the mask is off the object" for the RUN statistic below. Half the pixels wrong
#: is not a borderline mask; on the trajectory capture the per-frame arm's median GT-IoU is 0.988
#: and its p1 is 0.926, so 0.5 is nowhere near the body of the distribution and a run of frames
#: under it is a qualitatively different event rather than a tail.
LOW_IOU_THRESHOLD = 0.5

#: Fraction of captured frames that must yield BOTH centroids before the p95 is called a
#: measurement. As in ``measure_geom_tol``: a threshold on how much the estimator could see, not on
#: the scene.
DEFAULT_MIN_COVERAGE = 0.90

DEFAULT_HIST_BIN_PX = 0.5

#: The class name whose centroid G0b tracks. PR-08 §6 gates "object and plate"; the budget is
#: derived from the object, exactly as GEOM_TOL is, or the subtraction compares two different
#: things.
DEFAULT_OBJECT_CLASS = "apple"

#: THE ALLOW-LIST THAT KEEPS A LAPTOP CAPTURE OUT OF A GATE, and what each route's error
#: direction is. Until 2026-08-22 this was one hard-coded comparison — ``type(binding).__name__
#: != "IsaacSimBinding"`` — and widening it is the only edit in this file that could let a
#: capture from something that is not ground truth become G0b's budget. So it is a table, in one
#: place, with the reason for each entry beside it, rather than a second comparison somewhere.
#: ``FakeIsaacBinding`` is deliberately ABSENT and must stay absent: it is a plausible integrator
#: whose "ground truth" is a moving square, and every capture anyone has run so far came from it.
#:
#: The MuJoCo entry is `T40_RULE_V5` (``docs/preregistration/PR-08-V5-ground-truth-route.md``,
#: registered 2026-08-22 before any capture was run): PR-08 §4's ground-truth source becomes "a
#: simulator with ground-truth segmentation" rather than Isaac specifically, because
#: ``EST_DRIFT_P95`` is defined purely on segmentation and §4 step 3's depth error is recorded,
#: not gated.
#:
#: ``is_lower_bound`` USED TO BE STAMPED ``True`` UNCONDITIONALLY AND IS NOW PER-ROUTE. That is a
#: three-line change carrying a large meaning, so: the Isaac row is byte-for-byte what the
#: unconditional stamp said, including the reason string whose Humanoid-Everyday half the runbook
#: (§6b, §7 defect 4) argues is stale — correcting THAT is a judgement for whoever owns PR-08 and
#: is deliberately not made here. The MuJoCo row does not claim a lower bound and does not claim
#: an upper one: it records the ARGUED direction and states in the same breath that the argument
#: is not measured. A number whose direction is unknown is worth less than one whose direction is
#: known, and a number whose direction is *asserted* is worth less than one that says it was
#: asserted.
GROUND_TRUTH_BINDINGS: Mapping[str, Mapping[str, Any]] = {
    "IsaacSimBinding": {
        "route": "isaac",
        "is_lower_bound": True,
        "is_lower_bound_reason": (
            "measured on Isaac renders, not RealSense footage (PR-08 §4). The confirmatory "
            "measurement against Humanoid Everyday is blocked on that corpus's licence and is "
            "deliberately off the critical path."
        ),
        "error_direction": "optimistic (lower bound on the real error) — PR-08 §4's own words",
        "error_direction_measured": False,
    },
    "MuJoCoGroundTruthBinding": {
        "route": "mujoco",
        # NOT a lower bound, and NOT claimed as an upper one. See error_direction.
        "is_lower_bound": False,
        "is_lower_bound_reason": (
            "PR-08-V5 (T40_RULE_V5): measured on MuJoCo renders with exact per-pixel geom-id "
            "segmentation. MuJoCo's rasteriser is markedly less photoreal than Isaac's RTX "
            "path, so a detector trained on photographs does WORSE here, the p95 is LARGER, "
            "more is subtracted from GEOM_TOL and G0b is STRICTER. The error therefore lands "
            "against the generator, which is the safe direction and the opposite of the lower "
            "bound PR-08 §4 warns about. That is an ARGUMENT and not a measurement: this run "
            "does not establish an upper bound either, and V5 says so."
        ),
        "error_direction": (
            "conservative (argued): less photoreal frames -> larger p95 -> smaller "
            "GEOM_TOL - EST_DRIFT_P95 -> stricter G0b"
        ),
        "error_direction_measured": False,
    },
}


def ground_truth_route(binding_name: str | None) -> Mapping[str, Any] | None:
    """The :data:`GROUND_TRUTH_BINDINGS` row for a capture's binding, or ``None``.

    ``None`` is not a default route: every caller treats it as "this capture is not ground
    truth", which is what ``capture_is_not_from_isaac_sim`` already says.
    """
    if not binding_name:
        return None
    return GROUND_TRUTH_BINDINGS.get(str(binding_name))

#: Monocular depth estimators this script would know how to drive if one were wired. Names only --
#: nothing here is imported unless it is present, and nothing is ever fetched.
CANDIDATE_DEPTH_ESTIMATORS: tuple[tuple[str, str], ...] = (
    ("depth_anything_v2", "Depth-Anything-V2 — `depth_anything_v2` package plus a vit checkpoint"),
    ("transformers", "HF `pipeline('depth-estimation')` — Depth-Anything / DPT / GLPN checkpoints"),
    ("midas", "MiDaS — `midas` package plus a dpt_* checkpoint"),
    ("zoedepth", "ZoeDepth — metric monocular depth, `zoedepth` package plus a checkpoint"),
)


# -- the loud failures ---------------------------------------------------------------------------


def _missing_message(kind: str, candidates: tuple[tuple[str, str], ...]) -> str:
    """Name every place that was looked in and what would have to change. Never a fallback."""
    present = [(m, why) for m, why in candidates if _importable(m)]
    absent = [(m, why) for m, why in candidates if not _importable(m)]
    lines = [
        f"FATAL: no gate-qualified {kind} is wired, so EST_DRIFT_P95 cannot be measured.",
        "       Nothing was written.",
        "",
        f"       interpreter: {sys.executable}",
        "",
        f"       {kind} packages NOT importable by this interpreter:",
    ]
    lines += [f"         - {m:<20} {why}" for m, why in absent] or ["         (none)"]
    if present:
        lines += ["", "       importable, but this script has no code path for them yet:"]
        lines += [f"         - {m:<20} {why}" for m, why in present]
    hits = _local_weight_hits()
    lines += ["", "       local weights found:"]
    lines += [f"         - {w}" for w in hits] or ["         (none)"]
    lines += [
        "",
        "       PR-08 §4 step 2 says 'the SAME segmenter' as GEOM_TOL's. Wiring one of these is",
        "       therefore one decision for both measurements, not two — see",
        f"       {WRITEUP} §4 and scripts/measure_geom_tol.py.",
    ]
    return "\n".join(lines)


class EstimatorUnavailable(RuntimeError):
    """Raised when an estimator cannot run here. Always fatal, never fallen back from."""


# -- the object mask -----------------------------------------------------------------------------


def label_text(entry: Any) -> str | None:
    """The comparable label string inside one ``idToLabels`` value, or None if the shape is new.

    Replicator's mapping is documented as ``{"class": "apple"}``-shaped and that is UNVERIFIED
    (``isaac_binding.SegmentationFrame``; preflight check N records the real thing). A bare string
    is accepted because some builds emit one. Anything else returns None and is REPORTED rather
    than guessed at: picking a value out of an unrecognised structure is how the rig would end up
    tracking the plate, or the gripper, and still produce a number.
    """
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for key in ("class", "semanticLabel", "label"):
            value = entry.get(key)
            if isinstance(value, str):
                return value
    return None


def object_ids(id_to_labels: dict[int, Any], object_class: str) -> tuple[list[int], list[str]]:
    """Every label id whose text matches ``object_class``, plus every label text seen.

    Returns the full vocabulary too, because the failure "the apple is not in this scene" and the
    failure "the apple is called something else here" look identical from the caller and are fixed
    differently.
    """
    matched: list[int] = []
    seen: list[str] = []
    for ident, entry in id_to_labels.items():
        text = label_text(entry)
        if text is None:
            continue
        seen.append(text)
        if text.strip().lower() == object_class.strip().lower():
            matched.append(int(ident))
    return matched, sorted(set(seen))


def mask_from_ids(ids: np.ndarray, wanted: list[int]) -> np.ndarray:
    """Binary mask of every pixel carrying one of ``wanted``."""
    if not wanted:
        return np.zeros(ids.shape, dtype=bool)
    return np.isin(ids, np.asarray(wanted, dtype=ids.dtype))


def paired_displacements(
    pairs: list[tuple[tuple[float, float] | None, tuple[float, float] | None]],
) -> tuple[np.ndarray, int]:
    """Per-frame estimated-vs-true centroid distance, and how many frames could not be measured.

    Unlike ``measure_geom_tol.displacements``, which walks a clip in TIME, this compares two masks
    of the SAME frame. A frame counts only when both centroids exist: a true mask with no object is
    not an estimator error, and an estimated mask that found nothing where the truth has an object
    is a *detection* failure whose magnitude in pixels is undefined. Both are dropped and counted,
    and ``coverage`` is what makes the drop rate visible rather than absorbed.
    """
    out: list[float] = []
    dropped = 0
    for est, true in pairs:
        if est is None or true is None:
            dropped += 1
            continue
        out.append(float(np.hypot(est[0] - true[0], est[1] - true[1])))
    return np.asarray(out, dtype=float), dropped


def mask_iou(estimated: np.ndarray, true: np.ndarray) -> float | None:
    """Intersection over union of two binary masks of the same frame, or ``None`` if undefined.

    ``None`` for exactly one case — **both masks empty** — and it is not a rounding decision.
    ``0/0`` there would have to be picked, and either pick is a claim: ``0.0`` says the estimator
    was wrong on a frame where the truth agrees there is nothing to find, and ``1.0`` says a
    detector that returns nothing on every frame of an occluded run scores perfectly. The frame is
    counted instead (``n_frames_both_masks_empty``).

    An estimator that found NOTHING where the truth has an object scores ``0.0`` and stays in the
    distribution. That is the difference between this number and ``est_drift_p95_px``: a missed
    detection has no displacement in pixels — the centroid does not exist — so
    :func:`paired_displacements` drops it into ``coverage``, and the tail of the p95 is computed
    over the frames the estimator got approximately right. The IoU of a missed detection is
    defined, it is zero, and dropping it would be reporting the error only where there was one.
    """
    a = np.asarray(estimated)
    b = np.asarray(true)
    if a.dtype != bool:
        a = a > 0
    if b.dtype != bool:
        b = b > 0
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return None
    return float(int(np.logical_and(a, b).sum()) / union)


#: What ``mask_vs_ground_truth_iou`` says about itself, in the artifact, beside the number.
#:
#: Prose in a JSON document is usually a smell; here it is the point. This block is the second limb
#: of ``apple_sam2``'s FIRST gate-qualification blocker — *"a mask-vs-ground-truth IoU distribution
#: from the Isaac capture recorded beside the centroid displacement"* — and a number that satisfies
#: the letter of a discharge condition is exactly the number somebody will read as the discharge.
#: The three sentences below are the three ways that reading is wrong, and they travel with the
#: value rather than living in a commit message nobody will find.
_IOU_DISCHARGES = (
    "NOTHING. This block RECORDS evidence; accepting it is a separate act and a human's. "
    "scripts/estimators/apple_sam2.py's GATE_QUALIFIED and GATE_QUALIFICATION_BLOCKERS are "
    "untouched by the run that wrote this file, estimator_not_gate_qualified is still stamped "
    "into gate_disqualified_reasons above, and the first blocker's discharge condition has a "
    "second limb this does not supply — 'a human looking at a sample of overlaid masks spanning "
    "the corpus (occluded frames, apple-out-of-frame frames, and the grasp)'."
)
_IOU_SIMULATOR_CAVEAT = (
    "THIS IS A SIMULATOR CAPTURE. The masks compared here are a detector's output on MuJoCo "
    "rasteriser frames of configs/sim/g1_scene.xml against that renderer's exact geom-id "
    "segmentation. It establishes NOTHING on its own about the real AppleToPlate corpus, which is "
    "a real apple on a real tablecloth through a D435 — see capture.object_limitations for the "
    "four named ways the rendered object is not that apple (PR-08-V5 §4.4), all of which travel "
    "with this number too."
)
_IOU_OPEN_RULE_QUESTION = (
    "ANSWERED 2026-08-27, and the wording below is kept because a question that simply disappears "
    "between two artifacts is indistinguishable from one somebody dropped. IT ASKED: 'OPEN, and "
    "deliberately not resolved here. The third gate-qualification blocker names \"the same Isaac "
    "capture\" in its own words; PR-08-V5 (T40_RULE_V5) rerouted §4 step 1's ground truth from "
    "Isaac to any simulator with exact per-pixel segmentation, for a different purpose and without "
    "addressing that blocker. Whether a MuJoCo capture may stand where the blocker says Isaac is a "
    "rule question for the project owner. No session may answer it by writing a capture and "
    "pointing at it.' IT WAS ANSWERED THE WAY IT ASKED TO BE — by the project owner, not by a "
    "capture: PR-08-V14 (T40_RULE_V14, docs/preregistration/PR-08-V14-mujoco-stands-in-for-isaac.md), "
    "signed 2026-08-27, licenses a MuJoCo capture to stand in for the named Isaac one FOR "
    "EST_DRIFT_P95 AND THE ARM COMPARISON AND FOR NOTHING ELSE — not for GEOM_TOL, which is "
    "measured on the real corpus, and not for quoting a MuJoCo number as an Isaac one, so every "
    "artifact still records which simulator produced it and this one does. WHAT THAT DOES NOT DO, "
    "in V14 §3.1's own words: it does not discharge the blocker. The blocker names TWO independent "
    "sufficient reasons and V14 closes one; '480 frames of ONE trajectory is not a corpus' stands, "
    "GATE_QUALIFIED stays False, and the tuple is not shortened on V14's strength alone. How the "
    "second reason is being measured is registered as T40_RULE_V17 "
    "(docs/preregistration/PR-08-V17-drift-rate-protocol.md), whose outcomes were fixed before its "
    "first capture was rendered."
)


def iou_distribution(values: Sequence[float], n_both_masks_empty: int) -> dict[str, Any]:
    """The full IoU distribution — dimensionless, so none of these keys carries a ``_px`` suffix.

    Deliberately NOT ``distribution()``: that helper's keys are ``min_px``/``max_px``/
    ``percentiles_px`` and it bins a histogram in pixels. An overlap fraction in a field called
    ``min_px`` is the units error this repository keeps naming, and it would sit two keys away from
    ``centroid_displacement``, which really is in pixels.

    ``p1``/``p5`` are carried beside the percentiles the blocker asks for because IoU runs the
    other way from a displacement: the interesting tail of "how badly can this mask be wrong" is
    the LOW end, and a p95 of an IoU is the *ninety-fifth best* frame.
    """
    block: dict[str, Any] = {
        "meaning": (
            "Per-frame intersection-over-union between the ESTIMATED object mask and the "
            "renderer's EXACT ground-truth mask (geom ids from the same rasteriser that drew the "
            "RGB — no annotation, no threshold, no model in that path). 1.0 is perfect agreement. "
            "Scored on every frame whose ground truth carries the object label, INCLUDING frames "
            "the estimator missed entirely, which score 0.0."
        ),
        "not_to_be_confused_with": (
            "estimator_stats' mask_validity_iou (scripts/estimators/apple_sam2.py, "
            "MASK_VALIDITY_IOU). That one is the adapter's own check of its mask against a warm-"
            "and-saturated COLOUR predicate — a second non-learned opinion about where the fruit "
            "is, used to REFUSE a mask below MASK_VALIDITY_MIN_IOU. It is not ground truth and "
            "cannot be: it is a predicate for one object under one appearance. This block is "
            "against the simulator's own per-pixel truth."
        ),
        "discharges": _IOU_DISCHARGES,
        "simulator_caveat": _IOU_SIMULATOR_CAVEAT,
        "open_rule_question": _IOU_OPEN_RULE_QUESTION,
        "units": "dimensionless overlap fraction in [0, 1]; higher is better",
        "n_frames_both_masks_empty": int(n_both_masks_empty),
    }
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        block.update({"recorded": False, "n": 0, "values": []})
        return block
    pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    q = np.percentile(arr, pcts)
    block.update(
        {
            "recorded": True,
            "n": int(arr.size),
            "min": float(arr.min()),
            "median": float(np.median(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "max": float(arr.max()),
            "percentiles": {f"p{p}": float(v) for p, v in zip(pcts, q)},
            "n_frames_zero_iou": int((arr == 0.0).sum()),
            "n_frames_below_half": int((arr < 0.5).sum()),
            # Raw, because a few hundred frames is small enough to keep and because a
            # distribution nobody can re-derive is a distribution nobody can check.
            "values": [float(v) for v in arr],
        }
    )
    return block


# -- the two arms, and the confound that would void the comparison between them --------------------
#
# ``apple_sam2``'s THIRD gate-qualification blocker, in its own words: upstream drives
# ``SAM2VideoPredictor.init_state(video_path=...)`` and PROPAGATES one mask across the clip, while
# this adapter re-detects and re-segments every frame independently. The bias is TWO-SIDED — (a)
# per-frame jitter inflates our p95, which subtracts more from GEOM_TOL and is safe; (b)
# propagation's characteristic failure, drifting off the object and STAYING off for a run of
# frames, is invisible to a per-frame estimator that recovers on the next frame, which is unsafe.
# The blocker's discharge condition is the same capture measured BOTH ways with the two p95s
# recorded side by side. Everything below is that measurement and NOTHING below is that discharge.
#
# THE CONFOUND, NAMED HERE BECAUSE IT IS THE REASON THIS IS NOT TEN LINES OF CODE.
# ``SAM2VideoPredictor`` conventionally ingests a directory of JPEG files; our captures are
# lossless ``rgb.npy``. An arm that saw a transcode of the other arm's frames would make the
# difference between the two p95s "the codec plus propagation", reported as propagation — void
# rather than weak, and not visibly wrong in any field of the artifact. So the harness hands both
# arms THE SAME in-memory array object, and records a per-frame digest of what each one was shown.
# :class:`PixelWitness` is that record and :func:`identical_input_pixels` is the verdict, which
# lands in the artifact as a checkable fact rather than as a promise in this comment.


class PixelWitness:
    """Per-frame sha256 of the exact pixels one arm was handed.

    Not a debugging aid. "Both arms saw the same pixels" is the premise the entire comparison rests
    on, and a premise that lives only in the control flow is a premise nobody can check six months
    later from the artifact. :meth:`show` returns the array it was given, so the call site reads as
    the handover it is recording and there is no way to segment a frame that was not witnessed.
    """

    def __init__(self, arm: str) -> None:
        self.arm = arm
        self.digests: dict[int, str] = {}

    def show(self, frame_index: int, rgb: np.ndarray) -> np.ndarray:
        self.digests[int(frame_index)] = hashlib.sha256(
            np.ascontiguousarray(rgb).tobytes()
        ).hexdigest()
        return rgb

    def combined(self, indices: Sequence[int]) -> str:
        """One digest over ``indices`` in order — the whole of what this arm saw, in one string."""
        rolling = hashlib.sha256()
        for i in indices:
            rolling.update(bytes.fromhex(self.digests[int(i)]))
        return rolling.hexdigest()


#: What ``identical_input_pixels`` says about itself. Prose in JSON again, and for the same reason
#: the IoU block carries it: this field is the answer to "was the comparison confounded", and a
#: reader who does not know what the confound WAS cannot tell whether ``equal: true`` is reassuring.
_PIXEL_MECHANISM = (
    "Both arms are handed the SAME in-memory uint8 array read from the capture's rgb.npy — no "
    "file is written between them and no image is encoded. This matters because "
    "SAM2VideoPredictor.init_state conventionally ingests a directory of JPEG frames "
    "(sam2.utils.misc.load_video_frames takes a JPEG folder or an MP4); had the propagation arm "
    "been driven that way while the per-frame arm read the raw arrays, the difference between the "
    "two p95s would have been the codec plus propagation, reported as propagation. The "
    "propagation module replaces load_video_frames for the duration of exactly one init_state "
    "call with an ingest that performs upstream's own _load_img_as_tensor arithmetic on the "
    "arrays; tests/test_apple_sam2_video_propagation.py asserts that ingest is BITWISE upstream's "
    "ingest of a lossless file of the same frames, and separately that a JPEG round trip would "
    "NOT have been. The digests below are the per-run evidence that it happened."
)


def identical_input_pixels(per_frame: PixelWitness, propagation: PixelWitness) -> dict[str, Any]:
    """Did the two arms see the same bytes? The recorded answer, not the assumption.

    Compared over the frames BOTH arms were shown, because the two populations legitimately
    differ: the per-frame arm is not asked to segment a frame whose ground truth carries no object
    label, while the propagation arm must be handed every frame of the clip or it is not
    propagating across the capture at all. The frames that decide the comparison are the ones both
    scored, and those are the ones digested here.
    """
    common = sorted(set(per_frame.digests) & set(propagation.digests))
    block: dict[str, Any] = {
        "mechanism": _PIXEL_MECHANISM,
        "n_common_frames": len(common),
        "n_frames_shown_to_per_frame_arm": len(per_frame.digests),
        "n_frames_shown_to_propagation_arm": len(propagation.digests),
    }
    if not common:
        block.update({
            "equal": None,
            "absent_because": (
                "only one arm ran, so there are no frames both arms were shown and nothing to "
                "compare. This is an absence, not an agreement."
            ),
            "frames_that_differ": [],
            "per_frame_arm_sha256": None,
            "propagation_arm_sha256": None,
        })
        return block
    differ = [i for i in common if per_frame.digests[i] != propagation.digests[i]]
    block.update({
        "equal": not differ,
        "absent_because": None,
        "frames_that_differ": differ,
        "per_frame_arm_sha256": per_frame.combined(common),
        "propagation_arm_sha256": propagation.combined(common),
        "digest": (
            "per frame: sha256 of that frame's C-contiguous bytes. Combined: sha256 over those "
            "32-byte digests concatenated in capture frame order — chained rather than taken over "
            "the raw pixels so that a 480-frame capture needs no second copy of itself in memory "
            "to be witnessed. Re-derivable from the capture with six lines and no other input."
        ),
    })
    return block


#: Why the RUN is the statistic and the p95 is not, stated in the block itself.
_LOW_IOU_RUN_MEANING = (
    "The number blocker 3's failure mode (b) is actually about. 'Propagation drifts off the "
    "object and STAYS off for a run of frames' does not show up in a mean, does not reliably show "
    "up in a p95, and cannot show up at all in a per-frame estimator that recovers on the next "
    "frame — a per-frame arm scattering the same count of bad frames across the capture and a "
    "propagation arm losing the object for a contiguous stretch produce the SAME IoU "
    "distribution. They produce different runs. Indices are into the capture's frame order."
)


def low_iou_runs(
    iou_per_frame: Sequence[float | None], threshold: float = LOW_IOU_THRESHOLD
) -> dict[str, Any]:
    """Longest and count of CONSECUTIVE frames whose ground-truth IoU is below ``threshold``.

    A frame nobody could score — no object in the ground truth, or both masks empty — is ``None``
    and BREAKS a run rather than extending it. That errs SHORT, i.e. toward understating failure
    mode (b), which is the unsafe direction, so the count of such frames is recorded beside the
    runs instead of being left for a reader to assume was zero. The alternative (splicing them out
    and joining the two halves) would claim the tracker was off the object during a frame on which
    nothing observed it, which is a claim this harness has no evidence for.
    """
    runs: list[list[int]] = []
    start: int | None = None
    unscored = 0
    for i, value in enumerate(iou_per_frame):
        if value is None:
            unscored += 1
            if start is not None:
                runs.append([start, i - 1])
                start = None
            continue
        if float(value) < threshold:
            if start is None:
                start = i
        elif start is not None:
            runs.append([start, i - 1])
            start = None
    if start is not None:
        runs.append([start, len(iou_per_frame) - 1])
    lengths = [b - a + 1 for a, b in runs]
    return {
        "meaning": _LOW_IOU_RUN_MEANING,
        "threshold": float(threshold),
        "n_frames": len(iou_per_frame),
        "n_unscored_frames": unscored,
        "unscored_frames_break_a_run": True,
        "longest_run": max(lengths) if lengths else 0,
        "n_runs": len(runs),
        "n_frames_in_runs": sum(lengths),
        "runs": runs,
    }


#: Why the measured motion is NOT the registered ``object_is_static_prop`` field, spelled out in
#: the block itself. The two are one word apart and answer different questions, and a reader who
#: conflates them concludes either "V5 §4.4 was quietly amended" or "the trajectory capture is a
#: still life" — both wrong, both plausible from the field names alone.
_COHERENCE_NOT_THE_SAME_CLAIM = (
    "capture.object_limitations.object_is_static_prop is a DIFFERENT claim and stays true here. "
    "PR-08-V5 §4.4 registers it to mean the object is teleported between scene states rather than "
    "dropped or grasped — it carries no contacts and no physics, on either schedule. This block "
    "answers the other question: did the object's mask actually MOVE in the picture, and by how "
    "much per frame. A trajectory capture is still a static prop in V5's sense."
)


def temporal_coherence_block(
    centroids: Sequence[tuple[float, float] | None],
    *,
    object_class: str | None,
    absent_because: str | None = None,
) -> dict[str, Any]:
    """MEASURED interframe motion of the ground-truth object mask. Additive, read-only, no gate.

    **Why this is measured and not derived from the schedule's name.** ``--schedule trajectory``
    is a string in a header. A capture whose prop never moved — a schedule bug, a mesh that failed
    to place, an object parked behind the hands for the whole run — would carry that string just
    as convincingly, and nobody reading the artifact six months later can re-render 480 frames to
    check. ``max_interframe_motion_px`` is the number that makes "temporally coherent" falsifiable:
    it must be small (no jump cuts) **and** non-zero (the object actually moved).

    On the committed lattice this block is the evidence for the opposite conclusion, which is why
    it is computed on both schedules and not only on the new one: measured 2026-08-25 at 480x640,
    neighbouring lattice frames move the mask **55.9 / 65.3 / 290.1 px**, so a video predictor
    propagating from frame 0 is being asked to track across a cut.

    ``(None, reason)`` — ``measured: false`` — for a capture whose object label is unknown, e.g.
    the Isaac route, which declares none. An absence with a reason, never a zero: "the object did
    not move" and "nobody could tell" are different facts and one of them reads as a still life.
    """
    block: dict[str, Any] = {
        "meaning": (
            "Euclidean distance in pixels between the GROUND-TRUTH object centroid on frame i and "
            "on frame i+1, over the capture as written to disk. Small AND non-zero is what makes "
            "this capture propagatable from frame 0; either failure alone makes it not."
        ),
        "read_the_median_for_smoothness_and_the_max_for_events": (
            "The MEDIAN is the schedule's own step and is O(1/n_frames) on the trajectory "
            "schedule, so a longer capture is a smoother one. The MAX is not: it is dominated by "
            "OCCLUSION TRANSITIONS, where the object passes behind a Dex3 hand and the centroid "
            "of the VISIBLE part of its mask jumps although the object itself moved a few pixels. "
            "That is a real event this capture is required to contain — PR-08-V5 §4.5 registers "
            "that the occluding hands may not be moved out of the way to improve a number — and "
            "it is exactly the kind of frame a propagation arm would be measured on. A max well "
            "above the median is therefore evidence of occlusion, NOT of a jump cut; a MEDIAN "
            "that is tens of pixels is the jump cut."
        ),
        "not_the_same_claim_as": _COHERENCE_NOT_THE_SAME_CLAIM,
        "what_this_makes_possible_and_what_it_is_not": (
            "A capture whose median interframe motion is a few pixels can be handed to a video "
            "predictor propagating one mask from frame 0; the committed lattice cannot, because "
            "its neighbours teleport the object. THIS BLOCK IS A PROPERTY OF THE CAPTURE AND NOT "
            "A RESULT. It says the propagation experiment is RUNNABLE on these frames; it does "
            "not say it was run, and it carries no p95 from either arm. Since 2026-08-25 "
            "measure_est_drift can run it — `measure --arm both` drives this adapter per frame "
            "and a SAM2VideoPredictor propagating from frame 0 over the same frames, and writes "
            "an arm_comparison block — but only a measure artifact from such a run carries that "
            "comparison, and running the experiment is still not discharging the blocker that "
            "asked for it."
        ),
        "discharges": _IOU_DISCHARGES,
        "units": "pixels at the capture resolution",
        "object_class": object_class,
        "n_frames": len(centroids),
    }
    if absent_because is not None or object_class is None:
        block.update(
            {
                "measured": False,
                "absent_because": absent_because
                or (
                    "the capture names no object class, so which ground-truth label to follow is "
                    "unknown. The Isaac route declares none; the mujoco binding declares "
                    "object_limitations.object_label."
                ),
                "object_moved_during_capture": None,
                "max_interframe_motion_px": None,
                "median_interframe_motion_px": None,
                "n_frames_with_object": None,
                "n_interframe_steps": None,
                "interframe_motion_px": [],
            }
        )
        return block

    steps: list[float] = []
    unmeasurable = 0
    for a, b in zip(centroids, centroids[1:]):
        if a is None or b is None:
            unmeasurable += 1
            continue
        steps.append(float(np.hypot(b[0] - a[0], b[1] - a[1])))
    arr = np.asarray(steps, dtype=float)
    block.update(
        {
            "measured": True,
            "absent_because": None,
            "n_frames_with_object": int(sum(c is not None for c in centroids)),
            "n_interframe_steps": int(arr.size),
            # A step whose either end had no visible object. Counted rather than treated as zero:
            # an occlusion is not a stationary object.
            "n_interframe_steps_unmeasurable": int(unmeasurable),
            "object_moved_during_capture": bool(arr.size and float(arr.max()) > 0.0),
            "max_interframe_motion_px": float(arr.max()) if arr.size else None,
            "median_interframe_motion_px": float(np.median(arr)) if arr.size else None,
            "mean_interframe_motion_px": float(arr.mean()) if arr.size else None,
            "min_interframe_motion_px": float(arr.min()) if arr.size else None,
            "interframe_motion_px": [float(v) for v in arr],
        }
    )
    return block


def scene_state_per_frame(header: Mapping[str, Any]) -> tuple[list[int] | None, str | None]:
    """Which scene configuration each captured frame belongs to, or why that is unknowable.

    Derived from the capture header's own ``ticks`` and ``steps_per_state`` rather than from
    ``n_frames // n_scene_states``: a run that ended early would make that arithmetic assign
    frames to states it never visited, which is the same class of error as computing
    ``n_scene_states_visited`` from ``--frames``.

    ``(None, reason)`` for a capture that records no ``steps_per_state`` — the Isaac route
    declares no schedule, and a fabricated grouping is worse than an absence with a reason.
    """
    steps_per_state = header.get("steps_per_state")
    ticks = header.get("ticks")
    if not steps_per_state or not isinstance(ticks, Sequence) or not ticks:
        return None, (
            "the capture header records no steps_per_state/ticks, so which frames share a "
            "scene configuration cannot be recovered. Isaac captures declare no schedule."
        )
    try:
        per_state = int(steps_per_state)
        if per_state < 1:
            raise ValueError
        return [max(0, (int(t) - 1)) // per_state for t in ticks], None
    except (TypeError, ValueError):
        return None, f"steps_per_state={steps_per_state!r} is not a positive integer"


def independent_sample_block(
    header: Mapping[str, Any],
    per_frame_px: Sequence[float | None],
) -> dict[str, Any]:
    """HOW MANY INDEPENDENT OBSERVATIONS ARE BEHIND THE p95. Additive, read-only, never a gate.

    **Nothing here is subtracted from anything and no disqualification reason depends on it.**
    ``est_drift_p95_px`` at the top of the artifact is the budget PR-08 §6 names, and it is
    unchanged by this block existing. What this records is the one property of the sample that
    a frame count cannot express.

    ``T40_RULE_V5`` §5 registers a floor of *"≥ 20 distinct scene states and ≥ 200 measured
    frames"*, on the reasoning that *"below ~100 measured frames a p95 is essentially the
    fifth-largest sample"* and *"over one configuration it is a percentile over one
    viewpoint"*. Both halves can be satisfied at once by a capture whose frames are near
    duplicates: the object is a static prop (V5 §4.4) and the arm settles in milliseconds, so
    the frames inside one configuration differ by a fraction of a pixel and the p95 over 240 of
    them is a p95 over the number of *configurations*. Measured on this box 2026-08-23, a
    20-state / 240-frame MuJoCo capture: the displacement spread **inside** each state was
    0.05–0.28 px while the spread **between** states was 0.06–39.9 px, and the one state that
    failed failed on all twelve of its frames.

    So both numbers are recorded side by side — the p95 over frames, and the p95 over one
    representative (the per-state median) per configuration — and the reader is not asked to
    infer the difference from ``n_measured``.
    """
    states, absent = scene_state_per_frame(header)
    if states is None:
        return {"recorded": False, "absent_because": absent}
    if len(states) != len(per_frame_px):
        return {
            "recorded": False,
            "absent_because": (
                f"the header lists {len(states)} ticks for {len(per_frame_px)} frames — the "
                "capture and the measure disagree about how many frames there are, and a "
                "grouping built on that would be fiction"
            ),
        }

    by_state: dict[int, list[float]] = {}
    frames_per_state: dict[int, int] = {}
    for state, value in zip(states, per_frame_px):
        frames_per_state[state] = frames_per_state.get(state, 0) + 1
        if value is not None:
            by_state.setdefault(state, []).append(float(value))

    medians = [float(np.median(v)) for v in by_state.values()]
    spreads = [float(max(v) - min(v)) for v in by_state.values() if len(v) > 1]
    measured = [v for v in per_frame_px if v is not None]
    counts = sorted(frames_per_state.values())
    return {
        "recorded": True,
        "absent_because": None,
        "meaning": (
            "ADDITIVE AND READ-ONLY. est_drift_p95_px is the budget; this block says how many "
            "independent configurations it was computed over. A p95 over frames and a p95 over "
            "configurations differ whenever the frames inside a configuration are duplicates, "
            "which a static-prop capture makes them (T40_RULE_V5 §4.4)."
        ),
        "n_scene_states_with_a_measured_frame": len(by_state),
        "n_scene_states_visited": header.get("n_scene_states_visited"),
        "n_measured_frames": len(measured),
        "frames_per_scene_state": {
            "min": counts[0] if counts else None,
            "max": counts[-1] if counts else None,
            "median": float(np.median(counts)) if counts else None,
        },
        "measured_frames_per_scene_state": {
            str(k): len(v) for k, v in sorted(by_state.items())
        },
        "p95_over_frames_px": float(np.percentile(measured, 95)) if measured else None,
        "p95_over_scene_state_medians_px": (
            float(np.percentile(medians, 95)) if medians else None
        ),
        "within_state_displacement_spread_px": {
            "n_states": len(spreads),
            "min": min(spreads) if spreads else None,
            "median": float(np.median(spreads)) if spreads else None,
            "max": max(spreads) if spreads else None,
        },
        "scene_state_median_displacement_px": {
            str(k): float(np.median(v)) for k, v in sorted(by_state.items())
        },
    }


#: What the comparison block refuses to be read as. The third blocker's discharge condition and
#: this block are the same sentence, which is exactly why the block has to say that satisfying a
#: condition's LETTER is not the act of closing it. That act is a person's.
_ARM_COMPARISON_DISCHARGES = (
    "NOTHING. This block is the measurement apple_sam2's THIRD gate-qualification blocker names — "
    "the same capture measured both ways, the two p95s side by side — and producing the evidence "
    "a blocker asks for is not the same act as accepting it. "
    "scripts/estimators/apple_sam2.py's GATE_QUALIFIED and GATE_QUALIFICATION_BLOCKERS are "
    "untouched by the run that wrote this file, GATE_QUALIFIED is still False, and "
    "estimator_not_gate_qualified is still stamped into gate_disqualified_reasons above. "
    "THE 'ISAAC, NOT MUJOCO' HALF OF THIS PARAGRAPH IS SPENT and its retirement is recorded in "
    "open_rule_question rather than deleted: T40_RULE_V14, signed by the project owner on "
    "2026-08-27, lets a MuJoCo capture stand in for the named Isaac one for this measurement. What "
    "remains is the blocker's SECOND and independently sufficient reason — 480 frames of one "
    "trajectory is not a corpus — which no capture written by this script discharges either, and "
    "which T40_RULE_V17 registers a design for. Blockers 1 and 2 remain untouched by anything "
    "measured here."
)

_ARM_COMPARISON_MEANING = (
    "THE TWO ARMS, OVER ONE CAPTURE. per_frame is this repository's adapter: GroundingDINO plus "
    "SAM 2 re-run independently on every frame, which is what segment(rgb) can host and what "
    "produces est_drift_p95_px above. propagation is Cosmos-Transfer2.5's topology: one detection "
    "on frame 0, one SAM2VideoPredictor seed, and the mask tracked forward across the clip. "
    "Blocker 3 argues the bias between them is TWO-SIDED — (a) per-frame jitter INFLATES our tail, "
    "which subtracts more from GEOM_TOL and is safe; (b) propagation drifting off the object and "
    "STAYING off is invisible to a per-frame estimator that recovers on the next frame, which is "
    "unsafe. The p95s answer (a). low_iou_runs answers (b), and it is the reason this block exists "
    "rather than two numbers."
)


def arm_block(
    arm: str,
    *,
    measured: bool,
    pairs: Sequence[tuple[tuple[float, float] | None, tuple[float, float] | None]],
    iou_per_frame: Sequence[float | None],
    n_both_masks_empty: int,
    hist_bin_px: float,
    absent_because: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One arm's whole result: its p95, its displacement distribution, its IoU, its runs.

    Built by ONE function for both arms on purpose. Two arms whose numbers are computed by two
    code paths differ in their code paths as well as in their segmenters, and the difference
    between the two p95s is then not attributable to the thing under test.
    """
    if not measured:
        return {
            "arm": arm,
            "measured": False,
            "absent_because": absent_because or f"--arm did not include {arm}",
            "est_drift_p95_px": None,
            "n_frames": len(iou_per_frame),
            "n_measured": 0,
            "n_dropped": 0,
            "coverage": None,
            "centroid_displacement": None,
            "mask_vs_ground_truth_iou": None,
            "low_iou_runs": None,
        }
    values, dropped = paired_displacements(list(pairs))
    ious = [float(v) for v in iou_per_frame if v is not None]
    block: dict[str, Any] = {
        "arm": arm,
        "measured": True,
        "absent_because": None,
        "est_drift_p95_px": float(np.percentile(values, 95)) if values.size else None,
        "n_frames": len(pairs),
        "n_measured": int(values.size),
        "n_dropped": int(dropped),
        "coverage": (float(values.size) / len(pairs)) if len(pairs) else 0.0,
        "centroid_displacement": distribution(values, hist_bin_px),
        "mask_vs_ground_truth_iou": iou_distribution(ious, n_both_masks_empty),
        "low_iou_runs": low_iou_runs(list(iou_per_frame)),
        # THE RAW PER-FRAME DISPLACEMENTS, and the reason they are here rather than only summarised.
        # `distribution` records percentiles and a 0.5 px histogram, and NEITHER can be pooled: a
        # p95 over the union of eight captures is not any function of eight p95s, and re-deriving
        # it from histogram bins would quantise the answer to the bin width. V17 §2 pools eight
        # captures, so the values it pools have to survive the artifact. Cheap — one float per
        # measured frame, ~4 kB for 480.
        "displacements_px": [float(v) for v in values],
    }
    if extra:
        block.update(dict(extra))
    return block


def arm_comparison_block(
    per_frame: Mapping[str, Any],
    propagation: Mapping[str, Any],
    *,
    arms: Sequence[str],
    pixels: Mapping[str, Any],
    propagator: "Propagator | None",
) -> dict[str, Any]:
    """The two arms side by side, with the premise of the comparison recorded beside the result."""
    both = bool(per_frame.get("measured")) and bool(propagation.get("measured"))
    pf_p95 = per_frame.get("est_drift_p95_px")
    pr_p95 = propagation.get("est_drift_p95_px")
    delta: dict[str, Any] = {
        "computed": bool(both and pf_p95 is not None and pr_p95 is not None),
        "propagation_minus_per_frame_px": None,
        "reading": None,
    }
    if delta["computed"]:
        difference = float(pr_p95) - float(pf_p95)
        delta["propagation_minus_per_frame_px"] = difference
        delta["reading"] = (
            "POSITIVE means the propagated mask's centroid sits FURTHER from ground truth at the "
            "95th percentile than the per-frame adapter's, i.e. blocker 3's limb (a) — 'our tail "
            "is inflated relative to the generator's' — does not hold on this capture in that "
            "direction. NEGATIVE means it does. Neither reading settles limb (b), which is what "
            "low_iou_runs is for, and neither is a statement about the real corpus."
            if difference > 0
            else
            "NEGATIVE means the propagated mask's centroid sits CLOSER to ground truth at the "
            "95th percentile than the per-frame adapter's, which is the direction blocker 3's "
            "limb (a) predicts: independent re-detection jitters where propagation is temporally "
            "smooth, so EST_DRIFT_P95 is inflated relative to the generator's mask error and "
            "subtracts more from GEOM_TOL. That is the SAFE side of the two-sided bias and it "
            "says nothing about limb (b), which is what low_iou_runs is for."
        )
    return {
        "meaning": _ARM_COMPARISON_MEANING,
        "blocker": (
            "scripts/estimators/apple_sam2.py GATE_QUALIFICATION_BLOCKERS, third entry — 'PER-"
            "FRAME SEGMENTATION IS NOT UPSTREAM'S PROPAGATION'"
        ),
        "arms": list(arms),
        "discharges": _ARM_COMPARISON_DISCHARGES,
        "simulator_caveat": _IOU_SIMULATOR_CAVEAT,
        "open_rule_question": _IOU_OPEN_RULE_QUESTION,
        "identical_input_pixels": dict(pixels),
        "low_iou_threshold": LOW_IOU_THRESHOLD,
        "per_frame": dict(per_frame),
        "propagation": dict(propagation),
        "delta": delta,
        "propagator": (
            None
            if propagator is None
            else {
                "spec": propagator.spec,
                "name": propagator.name,
                "version": propagator.version,
                "contract": propagator.contract,
                "stats": propagator.stats(),
            }
        ),
    }


def depth_error(estimated: np.ndarray, true: np.ndarray, mask: np.ndarray | None) -> dict[str, Any]:
    """Absolute depth error over finite true pixels, optionally restricted to a mask.

    ``distance_to_camera`` reports a ray that hit nothing as ``inf`` and the binding passes that
    through untouched rather than substituting a sentinel, so the non-finite pixels are excluded
    HERE and counted. Including them would make the mean a function of how much sky is in frame.
    """
    finite = np.isfinite(true) & np.isfinite(estimated)
    if mask is not None:
        finite &= mask
    n_total = int(mask.sum()) if mask is not None else int(true.size)
    if not finite.any():
        return {"n": 0, "n_candidate_px": n_total, "n_non_finite_px": n_total}
    err = np.abs(estimated[finite].astype(np.float64) - true[finite].astype(np.float64))
    pcts = [50, 90, 95, 99, 100]
    q = np.percentile(err, pcts)
    return {
        "n": int(err.size),
        "n_candidate_px": n_total,
        "n_non_finite_px": int(n_total - err.size),
        "mean_m": float(err.mean()),
        "median_m": float(np.median(err)),
        "percentiles_m": {f"p{p}": float(v) for p, v in zip(pcts, q)},
    }


# -- capture -------------------------------------------------------------------------------------


def capture_frames(
    binding: Any,
    camera: str,
    n_frames: int,
    out: Path,
    steps_per_frame: int,
    provenance: Mapping[str, Any] | None = None,
    object_class: str | None = None,
) -> dict:
    """Drive an already-constructed binding and write ground truth to ``out``. Returns the header.

    Takes the binding rather than building one so that the caller owns the Isaac boot -- and so
    that this whole path is exercisable against ``FakeIsaacBinding`` on a laptop, which is the only
    reason any of it is testable before an Isaac node exists.

    ``provenance`` is what the CALLER knows and this function cannot see: which USD stage was
    loaded, which prim the camera name resolved to, which grid was requested and where that number
    came from. It is merged into the header verbatim (existing keys win, so no caller can overwrite
    ``is_simulated_binding`` or the measured ``resolution_hw``) and travels into the EST_DRIFT_P95
    artifact, because "the true segmentation" is only auditable if the artifact says what scene it
    was the true segmentation OF.

    Warmup is a real state, not an error: ``render_*`` returns ``None`` until the renderer settles,
    and a frame is written only when ALL THREE channels are present. A partially-written frame would
    be a frame whose depth belongs to one tick and whose segmentation belongs to another.

    ``object_class`` names the ground-truth label whose centroid is tracked frame to frame, which
    is what makes ``temporal_coherence`` a MEASUREMENT of this capture rather than a restatement of
    the schedule it was asked for. It is computed here, while the segmentation is already in
    memory, and not by re-reading the frames later: a coherence block written by a different pass
    over a different directory is a block that can disagree with the frames it names. ``None`` —
    the Isaac route, which declares no label — records an absence with a reason.
    """
    attached = tuple(binding.ground_truth_channels)
    for needed in ("depth", "segmentation"):
        if needed not in attached:
            raise EstimatorUnavailable(
                f"the binding has no {needed!r} channel attached (has {list(attached)}). "
                f"Construct it with ground_truth=('depth', 'segmentation') — PR-08 §4 step 1 "
                f"measures against ground truth and there is none without it."
            )
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    ticks: list[int] = []
    warmups = 0
    true_centroids: list[tuple[float, float] | None] = []
    coherence_absent: str | None = None
    while written < n_frames:
        for _ in range(steps_per_frame):
            binding.step()
        rgb = binding.render_frame(camera)
        depth = binding.render_depth(camera)
        seg = binding.render_segmentation(camera)
        if rgb is None or depth is None or seg is None:
            warmups += 1
            if warmups > n_frames + 1000:
                raise EstimatorUnavailable(
                    f"the renderer never settled: {warmups} consecutive warmup returns with "
                    f"{written} of {n_frames} frames written."
                )
            continue
        d = frames_dir / f"{written:06d}"
        d.mkdir(exist_ok=True)
        np.save(d / "rgb.npy", rgb)
        np.save(d / "depth.npy", depth)
        np.save(d / "seg_ids.npy", seg.ids)
        (d / "seg_labels.json").write_text(
            json.dumps({str(k): v for k, v in seg.id_to_labels.items()}, indent=2, default=str),
            encoding="utf-8",
        )
        ticks.append(int(binding.get_physics_step_count()))
        if object_class is not None:
            wanted, seen = object_ids(dict(seg.id_to_labels), object_class)
            if not wanted and coherence_absent is None:
                # Not a crash and not a silent zero: the label the caller named is not in this
                # scene's vocabulary, which is §4.2's failure — a full run that measures nothing.
                # `measure` reports the same thing again from the frames on disk; this is the
                # earlier, cheaper copy of it.
                coherence_absent = (
                    f"no ground-truth label matched {object_class!r} on frame {written}; the "
                    f"scene's vocabulary there was {seen}"
                )
            true_centroids.append(
                centroid_of_mask(mask_from_ids(seg.ids, wanted), True, 1) if wanted else None
            )
        written += 1

    header = {
        "schema": "wam.est_drift_capture/1",
        "binding": type(binding).__name__,
        "camera": camera,
        "n_frames": written,
        "steps_per_frame": steps_per_frame,
        "warmup_returns": warmups,
        "ground_truth_channels": list(attached),
        "resolution_hw": [int(rgb.shape[0]), int(rgb.shape[1])],
        "ticks": ticks,
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # The name is load-bearing: a capture from the fake binding is a pipeline test and can
        # never be a gate input, and the reader must not have to infer that from the directory.
        # The comparison is against GROUND_TRUTH_BINDINGS rather than against one class name
        # since 2026-08-22 (PR-08-V5); FakeIsaacBinding is not in that table and this still
        # stamps True for it, which is the whole job of this field.
        "is_simulated_binding": ground_truth_route(type(binding).__name__) is None,
        # WHICH ground truth, when it is one. `measure` reads this to decide what it may say
        # about the number's error direction — a route whose direction is argued conservative
        # must not inherit the Isaac route's "lower bound" sentence, and vice versa.
        "ground_truth_route": (ground_truth_route(type(binding).__name__) or {}).get("route"),
        # WHETHER THE OBJECT ACTUALLY MOVED, AND BY HOW MUCH PER FRAME — measured from the masks
        # that were just written, never inferred from the schedule's name. Additive and read-only:
        # nothing downstream gates on it and no disqualification reason depends on it.
        "temporal_coherence": temporal_coherence_block(
            true_centroids, object_class=object_class, absent_because=coherence_absent
        ),
    }
    for key, value in dict(provenance or {}).items():
        header.setdefault(key, value)
    (out / "capture.json").write_text(json.dumps(header, indent=2) + "\n", encoding="utf-8")
    return header


def load_capture(root: Path) -> tuple[dict, list[Path]]:
    header_path = root / "capture.json"
    if not header_path.is_file():
        raise EstimatorUnavailable(f"no capture.json under {root} — run `capture` first.")
    header = json.loads(header_path.read_text(encoding="utf-8"))
    frames = sorted((root / "frames").glob("[0-9]" * 6))
    if not frames:
        raise EstimatorUnavailable(f"{root}/frames is empty — nothing to measure.")
    return header, frames


# -- measure -------------------------------------------------------------------------------------


class Estimators:
    """The pair under test: RGB in, estimated mask and estimated depth out.

    A named object rather than two loose callables because the ARTIFACT has to say which pair
    produced the number. G0b re-runs this comparison against the restyled clips later; a budget
    measured with one estimator and applied with another compares two different quantities, and
    nothing downstream would notice.
    """

    def __init__(self, module: Any, spec: str) -> None:
        self.spec = spec
        self.module = module
        for fn in ("segment", "estimate_depth"):
            if not callable(getattr(module, fn, None)):
                raise EstimatorUnavailable(
                    f"{spec}: estimator modules must define {fn}(rgb) — this one does not."
                )
        # Opt-IN, and absent means false. An estimator is gate-qualified only if its author said so
        # in the module, which is a thing a reviewer can find; defaulting to true would make every
        # stub a gate input.
        self.gate_qualified = bool(getattr(module, "GATE_QUALIFIED", False))
        self.name = str(getattr(module, "ESTIMATOR_NAME", spec))
        self.version = str(getattr(module, "ESTIMATOR_VERSION", "unversioned"))
        # WHAT THE SEGMENTER ACTUALLY IS, not just what it is called. `name` alone was the whole of
        # the §4 step 2 cross-check until 2026-08-22, and two runs can share a name while
        # disagreeing about the prompt, the detection thresholds, the retry and the box rule —
        # every one of which decides which frames get a mask and where its centroid lands. A module
        # that declares nothing here is not assumed to agree with anything: `main` disqualifies it
        # by its own name (`estimator_does_not_declare_segmenter_contract`).
        contract = getattr(module, "SEGMENTER_CONTRACT", None)
        self.segmenter_contract = dict(contract) if isinstance(contract, Mapping) else None
        # The prompt the segmenter grounds on, so `--object-class` can DEFAULT to it instead of
        # being typed a second time next to it. See `resolve_object_class`.
        prompt = getattr(module, "OBJECT_TEXT_PROMPT", None)
        self.object_text_prompt = prompt if isinstance(prompt, str) and prompt.strip() else None

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        return np.asarray(self.module.segment(rgb))

    def estimate_depth(self, rgb: np.ndarray) -> np.ndarray:
        return np.asarray(self.module.estimate_depth(rgb), dtype=np.float32)


def resolve_estimators(spec: str) -> Estimators:
    """Import the estimator pair named by ``--estimators``, or fail loudly. No fallback path.

    ``auto`` is the honest default and currently always fails: neither a segmenter nor a monocular
    depth estimator is wired in this repo, and PR-08 §4 step 2 requires the segmenter to be the
    SAME one GEOM_TOL used, which is itself unwired. Wiring one is therefore a single decision that
    closes both halves of §8 item 4, and the failure message says so.
    """
    if spec == "auto":
        raise EstimatorUnavailable(
            _missing_message("object segmenter", CANDIDATE_SEGMENTERS)
            + "\n\n"
            + _missing_message("monocular depth estimator", CANDIDATE_DEPTH_ESTIMATORS)
        )
    import importlib

    try:
        module = importlib.import_module(spec)
    except ImportError as exc:
        raise EstimatorUnavailable(f"cannot import estimator module {spec!r}: {exc}") from exc
    return Estimators(module, spec)


class Propagator:
    """The PROPAGATION arm: a whole clip in, one mask per frame out.

    A separate object from :class:`Estimators` because it is a separate CONTRACT, not a variant of
    one. ``segment(rgb)`` is per-frame by construction and a video predictor is clip-level; folding
    the second into the first is what makes the difference between the two arms unmeasurable, which
    is the state ``apple_sam2``'s third blocker describes. So this harness drives two contracts and
    the artifact says which number came from which.
    """

    def __init__(self, module: Any, spec: str) -> None:
        self.spec = spec
        self.module = module
        if not callable(getattr(module, "propagate", None)):
            raise EstimatorUnavailable(
                f"{spec}: a propagation module must define propagate(rgbs) -> list of masks — "
                "this one does not."
            )
        self.name = str(getattr(module, "PROPAGATION_NAME", spec))
        self.version = str(getattr(module, "PROPAGATION_VERSION", "unversioned"))
        contract = getattr(module, "PROPAGATION_CONTRACT", None)
        self.contract = dict(contract) if isinstance(contract, Mapping) else None

    def propagate(self, rgbs: Sequence[np.ndarray]) -> list[np.ndarray]:
        return [np.asarray(m) for m in self.module.propagate(rgbs)]

    def stats(self) -> dict[str, Any] | None:
        fn = getattr(self.module, "stats", None)
        if not callable(fn):
            return None
        try:
            return dict(fn())
        except Exception:  # noqa: BLE001 - a stats block is never worth losing a run over
            return None


def resolve_propagator(spec: str) -> Propagator:
    """Import the propagation module named by ``--propagation-module``, or fail loudly."""
    import importlib

    try:
        module = importlib.import_module(spec)
    except ImportError as exc:
        raise EstimatorUnavailable(
            f"cannot import propagation module {spec!r}: {exc}\n"
            "       The propagation arm needs a clip-level video predictor and the weights the "
            "per-frame arm uses. Nothing was written."
        ) from exc
    return Propagator(module, spec)


def _normalise_object(text: str) -> str:
    """The comparable form of an object name: GroundingDINO's phrase and Replicator's label meet.

    The segmenter's prompt is a lowercase period-terminated phrase (``"apple."``) because that is
    GroundingDINO's documented input format; the ground truth's label is Replicator's semantic class
    string (``"apple"``, sometimes ``"Apple"``). They name the same object in two notations, so the
    comparison strips the notation and nothing else. It deliberately does NOT strip words: ``"red
    apple."`` against a scene labelled ``"apple"`` is exactly the disagreement worth stopping on,
    because whichever of the two is wrong, the mask and the truth are then about different things.
    """
    return text.strip().lower().rstrip(".").strip()


def object_class_mismatch_message(requested: str, prompt: str, spec: str) -> str:
    return "\n".join([
        "FATAL: --object-class and the estimator's own text prompt name different objects, so this",
        "       run would compare one object's mask against another object's ground truth. Nothing",
        "       was written.",
        "",
        f"       --object-class      {requested!r}   (the label looked up in Replicator's idToLabels)",
        f"       {spec} OBJECT_TEXT_PROMPT  {prompt!r}   (what the segmenter grounds on)",
        "",
        "       This is refused rather than measured because it does not fail loudly on its own: an",
        "       apple mask scored against a plate's ground truth yields a large, entirely plausible",
        "       p95, no crash, and no drop in coverage — and that p95 is SUBTRACTED from GEOM_TOL in",
        "       PR-08 §6 G0b. It was a written gate-qualification blocker on the adapter until this",
        "       check existed.",
        "",
        "       Fix it at whichever end is wrong, deliberately:",
        "         - drop --object-class entirely; it defaults to the estimator's own prompt, which",
        "           is the spelling that cannot disagree with itself, or",
        "         - set WAM_PR08_OBJECT_PROMPT so the segmenter grounds on the object the capture",
        "           actually labels.",
        "       Changing either one moves both PR-08 §4 numbers, so it is a registered choice and",
        "       not a flag to sweep.",
    ])


def resolve_object_class(requested: str | None, est: "Estimators") -> tuple[str, str, list[str]]:
    """The object to score against, where it came from, and any disqualifying reasons.

    ONE OBJECT, NAMED ONCE. The object segmented and the object scored against used to be two
    independent knobs — ``$WAM_PR08_OBJECT_PROMPT`` in the estimator module, ``--object-class``
    here — and nothing compared them. The estimator cannot see this flag, so the check has to live
    on this side, and the cheapest correct version of it is to stop asking twice: ``--object-class``
    now defaults to the estimator's own prompt. An explicitly typed value that disagrees is FATAL
    rather than disqualifying, because unlike a coverage shortfall there is no number worth writing
    down: every displacement in the run would be a distance between two different objects.

    An estimator that declares no prompt (a stub, a third-party module) is not assumed to agree:
    the run proceeds on the requested class — there is nothing else to proceed on — and carries
    ``estimator_does_not_declare_object_prompt`` so the artifact cannot be read as having checked.

    EVERY PATH RETURNS THE NORMALISED FORM, and that is a fix rather than a tidy-up. The agreement
    path used to return the RAW ``requested`` string while the default path returned the normalised
    one, so two spellings this function had just declared equivalent behaved differently
    downstream: ``object_ids()`` compares ``label.strip().lower() == object_class.strip().lower()``
    and does NOT strip GroundingDINO's trailing period, so ``--object-class "apple."`` — the exact
    spelling the flag's own help text invites, since it defaults to the estimator's
    ``OBJECT_TEXT_PROMPT`` — matched no Replicator label, every frame counted as
    ``frames_without_label``, and the run ended at coverage 0.0 reporting "the apple is not in this
    scene" about a notation difference. ``_normalise_object`` strips the notation and nothing else,
    so this cannot silently widen what matches: ``"red apple."`` still disagrees with ``"apple"``.
    """
    prompt = est.object_text_prompt
    if prompt is None:
        return _normalise_object(requested or DEFAULT_OBJECT_CLASS), "flag_or_default", [
            "estimator_does_not_declare_object_prompt"
        ]
    if requested is None:
        return _normalise_object(prompt), "estimator_prompt", []
    if _normalise_object(requested) != _normalise_object(prompt):
        raise EstimatorUnavailable(object_class_mismatch_message(requested, prompt, est.spec))
    return _normalise_object(requested), "flag_agrees_with_estimator_prompt", []


def _artifact_label(path: Path) -> str:
    """Repo-relative when it is under the repo, absolute otherwise. Never raises.

    ``relative_to`` throws for any path outside the tree, which a test fixture or an operator
    passing ``--out`` elsewhere legitimately is. Losing the whole cross-check block to a
    ValueError while formatting a label for it would be an absurd way to lose a gate record.
    """
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def cross_check_geom_tol(
    resolution_hw: list[int],
    estimator_name: str | None = None,
    segmenter_contract: Mapping | None = None,
) -> tuple[list[str], dict]:
    """Disqualifying reasons from GEOM_TOL's committed artifact, plus what was compared.

    §6 computes ``GEOM_TOL - EST_DRIFT_P95``. That subtraction is arithmetic only if both were
    measured in the same units on the same grid with the same estimator. Nothing else in the
    pipeline checks it, and a mismatch is invisible in the result: two plausible pixel numbers
    subtract to a plausible pixel number.

    **"The same segmenter" is now checked as a segmenter and not as a string.** Until 2026-08-22
    this function compared the pixel grid, the gate flag and the method NAME — and a name is the one
    thing about a segmenter that cannot change when its behaviour does. The same adapter at
    ``box_threshold`` 0.35 and at 0.15 detects on different frames, produces different masks and
    different centroids, and reports the identical ``ESTIMATOR_NAME`` while doing it. So
    ``segmenter_contract`` (the adapter's ``SEGMENTER_CONTRACT``) is compared field for field
    against the block committed beside GEOM_TOL, and any difference disqualifies.

    ``segmenter_contract=None`` means THIS side declared nothing, which is not a way to pass: it is
    the two existing unit-level callers, and ``main`` turns a silent estimator into
    ``estimator_does_not_declare_segmenter_contract`` on its own. A committed document that records
    no contract while this side has one is ``geom_tol_does_not_record_segmenter_params``.
    """
    if not GEOM_TOL_ARTIFACT.is_file():
        return (
            ["geom_tol_not_committed"],
            {
                "geom_tol_artifact": None,
                "note": f"{GEOM_TOL_ARTIFACT.name} does not exist yet",
                # Recorded even with nothing to compare against: whoever commits GEOM_TOL later
                # needs to know which grid this number is in, and re-deriving it from the capture
                # is a step they should not have to take.
                "this_resolution_hw": list(resolution_hw),
                "this_estimator_name": estimator_name,
                "this_segmenter_contract": (
                    dict(segmenter_contract) if segmenter_contract is not None else None
                ),
            },
        )
    doc = json.loads(GEOM_TOL_ARTIFACT.read_text(encoding="utf-8"))
    reasons: list[str] = []
    theirs_contract, contract_at = committed_segmenter_contract(doc)
    theirs_contract = theirs_contract or {}
    # The pre-measurement contract records neither of the two fields the measured artifact leads
    # with, so both are resolved through it as a fallback rather than read from one place. The
    # method name is the same string in both shapes by construction — it is the adapter's
    # ESTIMATOR_NAME — and the grid is the corpus's, which the contract commits to before anyone
    # has rendered a frame at it.
    theirs_hw = doc.get("resolution_hw") or doc.get("frame_hw") or theirs_contract.get(
        "pixel_grid_hw"
    )
    method = doc.get("mask_method") or {}
    theirs_method = method.get("name") if isinstance(method, Mapping) else None
    theirs_method = theirs_method or theirs_contract.get("method_name")

    # ABSENCE IS NOT AGREEMENT. The first version of this function read
    # `if theirs_hw is not None and ... != ...`, so a GEOM_TOL artifact that simply did not
    # record its grid passed the grid check BY SAYING NOTHING -- the default-permissive pattern
    # this repo rejects everywhere else, and the one place it is least affordable, because the
    # whole purpose here is to make `GEOM_TOL - EST_DRIFT_P95` arithmetic. A missing field means
    # the check could not be made, which is a reason to disqualify and not a reason to proceed.
    for field, value in (
        ("resolution_hw", theirs_hw),
        ("gate_qualified", doc.get("gate_qualified")),
        ("mask_method", theirs_method),
    ):
        if value is None:
            reasons.append(f"geom_tol_does_not_record_{field}")

    if theirs_hw is not None and list(theirs_hw) != list(resolution_hw):
        reasons.append("resolution_disagrees_with_geom_tol")
    if not doc.get("gate_qualified", False):
        reasons.append("geom_tol_is_not_gate_qualified")

    # THE JOIN KEY, finally enforced rather than merely copied. PR-08 §4 step 2 says "the same
    # segmenter", and measure_geom_tol.py's module docstring names `mask_method.name` ==
    # `pr08_est_drift.json estimators.name` as the pair that has to match. Until this line existed
    # both artifacts recorded the two names and nothing compared them, so two different segmenters
    # produced two plausible pixel numbers that subtracted cleanly to a plausible wrong tolerance.
    if theirs_method is not None and estimator_name is not None and theirs_method != estimator_name:
        reasons.append("mask_method_disagrees_with_estimator")

    # THE JOIN KEY IS A NAME, AND A NAME IS NOT A SEGMENTER. Everything above compares labels: two
    # runs agreeing on `grounding-dino+sam2+depth-anything-v2` have agreed on a string. §4 step 2
    # asks for the same SEGMENTER, so the parameters that decide which frames get a mask and where
    # its centroid falls are compared too — the prompt, both thresholds, the retry pair, the box
    # rule, the propagation mode and the two checkpoint pins. A cross-check that cannot see the
    # segmenter is not checking it, whatever it prints.
    disagreements: list[dict] = []
    if segmenter_contract is not None:
        if not theirs_contract:
            reasons.append("geom_tol_does_not_record_segmenter_params")
        else:
            disagreements = contract_disagreements(segmenter_contract, theirs_contract)
            if disagreements:
                reasons.append("segmenter_params_disagree_with_geom_tol")

    return reasons, {
        "geom_tol_artifact": _artifact_label(GEOM_TOL_ARTIFACT),
        "geom_tol_resolution_hw": theirs_hw,
        "geom_tol_gate_qualified": doc.get("gate_qualified"),
        "geom_tol_mask_method": doc.get("mask_method"),
        "geom_tol_mask_method_name": theirs_method,
        "geom_tol_segmenter_contract": theirs_contract or None,
        "geom_tol_segmenter_contract_at": contract_at,
        "this_resolution_hw": list(resolution_hw),
        "this_estimator_name": estimator_name,
        "this_segmenter_contract": (
            dict(segmenter_contract) if segmenter_contract is not None else None
        ),
        "segmenter_param_disagreements": disagreements,
        "join_key": (
            "measure_geom_tol mask_method.name == this artifact's estimators.name, AND the "
            "committed segmenter block (top-level `segmenter`, or mask_method.params.segmenter) == "
            "this estimator module's SEGMENTER_CONTRACT, field for field"
        ),
    }


# -- capture's command line, decided before Isaac boots --------------------------------------------
#
# WHAT THE THREE FUNCTIONS BELOW HAVE IN COMMON IS *WHEN* THEY FIRE. Every one of them refuses a
# capture on a laptop, in milliseconds, for a mistake that used to be found either after a full
# Isaac boot (an unknown camera name) or not at all until `measure` disqualified a finished capture
# (the pixel grid). That is the same defect twice: the cheapest possible error discovered in the
# most expensive possible place, on the one path in this repository that needs a GPU and a stage.


def contract_pixel_grid_hw() -> tuple[list[int] | None, str | None]:
    """``[H, W]`` the committed GEOM_TOL contract pins, and where in the document it was found.

    **THE GRID IS NOT A LITERAL IN THIS FILE, AND MUST NOT BECOME ONE.** ``[480, 640]`` written
    here would be a second copy of a pre-commitment that lives in exactly one place on purpose:
    the day the contract's ``pixel_grid_hw`` moved, `capture` would go on rendering the old grid
    and every run would be disqualified by ``resolution_disagrees_with_geom_tol`` with nothing in
    the record naming the stale copy. So the document that decides the grid decides it for the
    capture too, read at run time, through the same ``committed_segmenter_contract`` the
    cross-check reads — one implementation of "where the block lives", for the producer, the
    cross-check and now the renderer.

    ``(None, None)`` when there is no artifact, or it states no grid. That is a refusal for the
    caller and never a licence to pick a number.
    """
    if not GEOM_TOL_ARTIFACT.is_file():
        return None, None
    try:
        doc = json.loads(GEOM_TOL_ARTIFACT.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, None
    if not isinstance(doc, Mapping):
        return None, None
    contract, where = committed_segmenter_contract(doc)
    grid = (contract or {}).get("pixel_grid_hw")
    at = f"{where}.pixel_grid_hw" if where else None
    if grid is None:
        # The measured artifact leads with these two instead; the contract has neither until a
        # measurement lands. Same precedence as cross_check_geom_tol(), for the same reason.
        for key in ("resolution_hw", "frame_hw"):
            if doc.get(key) is not None:
                grid, at = doc[key], key
                break
    if grid is None or len(list(grid)) != 2:
        return None, None
    return [int(v) for v in grid], at


def resolve_render_hw(requested: Sequence[int] | None) -> tuple[tuple[int, int], str]:
    """The grid to render at, and where that number came from. Refuses rather than guessing.

    PR-08 §6 subtracts ``EST_DRIFT_P95`` from ``GEOM_TOL``, which is arithmetic on ONE pixel grid.
    ``cross_check_geom_tol`` already disqualifies a capture measured on another — but it does so
    after the render, and a render is the expensive half of this path. So the same comparison is
    made here, against the same committed number, before a simulator is booted.

    An explicit ``--render-hw`` that disagrees with the contract is FATAL and not a warning: there
    is no version of that run whose p95 is subtractable from ``GEOM_TOL``, so rendering it would
    only produce an artifact that has to be thrown away. Changing the grid is a reviewed edit to
    the committed contract, made before a measurement and never after one.
    """
    grid, where = contract_pixel_grid_hw()
    label = _artifact_label(GEOM_TOL_ARTIFACT)
    if grid is None:
        raise EstimatorUnavailable(
            f"{label} states no pixel grid, so --render-hw has no default and this capture cannot "
            "be shown to be on GEOM_TOL's grid.\n"
            "       That file is PR-08 §4 step 2's pre-commitment (the segmenter, its pins and "
            "`pixel_grid_hw`).\n"
            "       Restore it from git rather than passing a grid by hand: a capture rendered at "
            "a number nobody\n"
            "       committed to is not a gate input at any resolution, and "
            "GEOM_TOL - EST_DRIFT_P95 across two\n"
            "       grids is two plausible pixel numbers subtracting to a meaningless one."
        )
    if requested is None:
        return (grid[0], grid[1]), f"{label} {where}"
    got = [int(v) for v in requested]
    if got != grid:
        raise EstimatorUnavailable(
            f"--render-hw {got[0]} {got[1]} disagrees with the committed grid {grid[0]}x{grid[1]} "
            f"({label} {where}).\n"
            "       This is `resolution_disagrees_with_geom_tol`, refused BEFORE the render "
            "instead of after it:\n"
            "       PR-08 §6 computes GEOM_TOL - EST_DRIFT_P95 and that subtraction is only "
            "arithmetic on one pixel\n"
            "       grid, so the capture this would produce could never be a gate input. Drop the "
            "flag to take the\n"
            "       committed grid, or move the contract in a reviewed commit of its own — before "
            "the measurement."
        )
    return (got[0], got[1]), f"--render-hw (equal to {label} {where})"


def resolve_camera_prims(declared: Sequence[str] | None) -> dict[str, str]:
    """:data:`DEFAULT_CAMERA_PRIMS` extended by each ``NAME=/Prim/Path`` on the command line.

    The default map is one entry — ``persp`` -> the viewport camera every stage has — because that
    is all the bare ``g1.usd`` carries. A scene authored for §4 step 1 (table, plate, apple, an
    ego-like camera) adds prims, and this is how a capture names one without editing the binding.
    The extension exists so that validating ``--camera`` against a dict cannot paint a real scene
    into a corner: the check stays exact, and the operator states what the stage has.
    """
    prims = dict(DEFAULT_CAMERA_PRIMS)
    for item in declared or ():
        name, sep, prim = str(item).partition("=")
        name, prim = name.strip(), prim.strip()
        if not sep or not name or not prim.startswith("/"):
            raise EstimatorUnavailable(
                f"--camera-prim {item!r} is not NAME=/Prim/Path. The prim path is absolute and "
                "must already exist on the stage; Isaac's camera setup raises if it does not."
            )
        prims[name] = prim
    return prims


def resolve_camera(name: str, prims: Mapping[str, str]) -> str:
    """The USD prim ``name`` maps to, or a refusal naming every camera there is.

    A camera name is checked HERE, against the same dict the binding would check it against, and
    not inside the binding — where the check happens after ``SimulationApp`` has started, the
    stage has loaded and the articulation has resolved. A typo costing a full Isaac boot is the
    most expensive way this repository has to spell a typo.
    """
    if name not in prims:
        raise EstimatorUnavailable(
            f"unknown camera {name!r}; known: {sorted(prims)}. Isaac's own default stage carries "
            f"only {sorted(DEFAULT_CAMERA_PRIMS)} (the viewport camera), and the binding would "
            "raise this same error after a full Isaac boot. Name the stage's camera with "
            "--camera-prim NAME=/Prim/Path and then select it with --camera NAME."
        )
    return prims[name]


def resolve_stage(
    asset: str | None, scene: str | None, backend: str = "isaac"
) -> tuple[str | None, str]:
    """The stage to load, and which flag named it. ``asset`` and ``scene`` are ONE knob.

    ``backend`` decides only what "unnamed" means, and the Isaac answer is unchanged: ``None``
    plus the sentence naming Isaac's asset root. The MuJoCo route has a committed scene on disk
    (``configs/sim/g1_scene.xml``) so its unnamed case is a real path and not a resolver call.

    ``wam.robot.isaac_g1.IsaacG1Robot`` already treats ``scene_path`` as an alias for ``asset`` and
    refuses both ("they are the same knob"), and ``configs/robot/isaac_g1.yaml`` documents
    ``sim.scene`` as that alias. This mirrors it rather than inventing a third vocabulary for the
    same USD path.

    ``None`` keeps the binding's own behaviour: resolve Isaac's asset root and load the bare
    ``g1.usd``. That stage has no table, no plate and no apple — PR-08 §4 step 1's object simply is
    not in it — so the resulting capture measures nothing, and the header says which stage it was
    so a reader of the artifact can tell that case from a measurement.
    """
    if asset is not None and scene is not None:
        raise EstimatorUnavailable(
            "pass either --asset or --scene (they are the same knob: a USD path, referenced onto "
            "the stage). wam.robot.isaac_g1.IsaacG1Robot refuses the same pair for the same "
            "reason — two names for one stage is one of them being ignored silently."
        )
    if scene is not None:
        return scene, "--scene"
    if asset is not None:
        return asset, "--asset"
    if backend == "mujoco":
        from wam.robot.mujoco_binding import DEFAULT_SCENE  # noqa: PLC0415

        return str(DEFAULT_SCENE), "wam.robot.mujoco_binding.DEFAULT_SCENE"
    return None, "isaacsim.storage.native.get_assets_root_path() + DEFAULT_ASSET_SUBPATH"


def trajectory_schedule_params(
    factory: Any, requested: Mapping[str, float | None]
) -> tuple[dict[str, float], str]:
    """The three cycle counts a trajectory capture ran with, and where each came from.

    The defaults are READ OFF ``factory``'s own signature and never restated in this module. That
    is the whole point of the function: ``capture-mujoco-trajectory-f480`` was rendered before the
    flags existed and its header records no parameters at all, so the only way to know what it ran
    with is to read the defaults at the commit that produced it — precisely the inference V5 §5's
    field list exists to make unnecessary. A second copy of ``2.0`` living here would let the header
    state a number the schedule did not use, which is worse than recording nothing.

    Refuses a parameter the factory does not have, rather than silently dropping it: that is what
    would happen if ``trajectory_scene_schedule`` lost an axis while this table kept it.
    """
    signature = inspect.signature(factory)
    missing = [name for name in requested if name not in signature.parameters]
    if missing:
        raise EstimatorUnavailable(
            f"{sorted(missing)} is offered on the command line and "
            f"{factory.__module__}.{factory.__name__} has no such parameter. The CLI table and the "
            "schedule have drifted; a flag that is accepted and then dropped would land in the "
            "capture header as a parameter the capture did not use."
        )
    params: dict[str, float] = {}
    overridden: list[str] = []
    for name in requested:
        value = requested[name]
        if value is None:
            default = signature.parameters[name].default
            if default is inspect.Parameter.empty:  # pragma: no cover — signature has defaults
                raise EstimatorUnavailable(
                    f"{name} has no default in {factory.__name__} and none was given."
                )
            params[name] = float(default)
        else:
            params[name] = float(value)
            overridden.append(name)
    source = (
        "defaults (" + ", ".join(f"{k}={v!r}" for k, v in params.items()) + ")"
        if not overridden
        else ", ".join(f"--{name.replace('_', '-')}" for name in sorted(overridden))
    )
    return params, source


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # PR-08 §4 step 1. Every option below that is not --out or --frames exists because of a way
    # this subcommand used to be unable to produce a gate-qualified capture at all: the grid it
    # rendered at (nothing could set it, and the constructor default disagreed with the committed
    # contract, so EVERY Isaac capture was stamped resolution_disagrees_with_geom_tol), the stage
    # it loaded (nothing could name one, so only the bare g1.usd — with no apple in it — could be
    # captured) and the camera it rendered from (the DEFAULT value was a name no Isaac stage has,
    # and the error arrived after a full Isaac boot). All three are checked before the binding is
    # constructed; see resolve_render_hw / resolve_stage / resolve_camera.
    cap = sub.add_parser("capture", help="PR-08 §4 step 1 — render Isaac ground truth to a dir")
    cap.add_argument("--out", type=Path, required=True)
    cap.add_argument(
        "--backend",
        choices=("isaac", "mujoco"),
        default="isaac",
        help="which ground-truth simulator renders step 1 (default: %(default)s). `isaac` is "
        "PR-08 §4's letter and is unchanged in every flag, default and refusal. `mujoco` is "
        "T40_RULE_V5 (docs/preregistration/PR-08-V5-ground-truth-route.md): exact per-pixel "
        "geom-id segmentation out of the committed configs/sim/g1_scene.xml, on CPU, headless. "
        "EST_DRIFT_P95 is defined on segmentation alone, so the gated number is produced in "
        "full; §4 step 3's depth error is recorded, not gated.",
    )
    cap.add_argument(
        "--camera",
        default=None,
        help="camera NAME to render from. Defaults PER BACKEND: 'persp' for isaac (the viewport "
        "camera every stage has, taken from DEFAULT_CAMERA_PRIMS itself) and 'head' for mujoco "
        "(the scene's own D435i stand-in). An isaac camera is validated against "
        "DEFAULT_CAMERA_PRIMS plus every --camera-prim here rather than after an Isaac boot; a "
        "mujoco camera is validated against the compiled MJCF, which costs a sub-second compile.",
    )
    cap.add_argument(
        "--object-mesh",
        default=None,
        help="mujoco backend only: OBJ/STL for the object the budget is measured on. Defaults "
        "to the first hit in wam.robot.mujoco_binding.OBJECT_MESH_SEARCH_PATHS and NOTHING IS "
        "EVER DOWNLOADED; if none is found the run refuses and names every path it looked in. "
        "It does not fall back to the scene's orange cube — see PR-08-V5 §4.",
    )
    cap.add_argument(
        "--scene-states",
        type=int,
        default=None,
        help="mujoco backend only, --schedule lattice only: how many DISTINCT scene "
        "configurations the capture spans (default: 20). PR-08 §4.6: a p95 over N frames of one "
        "pose is a percentile over one viewpoint, so N is counted in configurations and not in "
        "frames. The frames are divided evenly across them and the count lands in the capture "
        "header. `--schedule trajectory` puts one configuration on every frame by construction "
        "and REFUSES this flag rather than accepting a number it would have to ignore.",
    )
    cap.add_argument(
        "--schedule",
        choices=tuple(SCENE_SCHEDULE_NAMES),
        default=None,
        help="mujoco backend only: which scene schedule drives the capture (default: lattice). "
        "`lattice` is the committed sweep of distinct configurations PR-08-V5 registered and is "
        "what every capture under runs/pr08-est-drift/ was made with — passing nothing changes "
        "nothing. `trajectory` renders a SMOOTH, continuous path instead, one configuration per "
        "frame, so the object moves a few pixels between neighbouring frames rather than "
        "teleporting tens of them. That is the capture a mask PROPAGATED from frame 0 could be "
        "measured on; the lattice's jump cuts make that experiment unrunnable, not merely unrun. "
        "IT BUILDS NO PROPAGATION ARM AND DISCHARGES NOTHING. The name lands in the capture "
        "header beside a MEASURED max_interframe_motion_px, because the name is not evidence.",
    )
    for _flag, _what in TRAJECTORY_PARAM_FLAGS.items():
        cap.add_argument(
            f"--{_flag.replace('_', '-')}",
            type=float,
            default=None,
            metavar="CYCLES",
            help=f"mujoco backend only, --schedule trajectory only: {_what} Counted in COMPLETE "
            "cycles over the whole capture, so it is scale-free in --frames and cannot turn a "
            "smooth path into one with a cut on its own. It does NOT move PR-08-V5 §4.5's "
            "envelope: the centre, the radii and the arm amplitude are derived constants with no "
            "flag, the cube distractor stays and the hands still occlude, so every pose reachable "
            "here is one the default already visits — only WHEN it is visited changes. Registered "
            "as T40_RULE_V17 §2. The value and its source land in the capture header, which is "
            "new: until this flag existed the three were recorded nowhere and were recoverable "
            "only by reading the function defaults at the producing commit.",
        )
    cap.add_argument(
        "--camera-prim",
        action="append",
        default=[],
        metavar="NAME=/Prim/Path",
        help="declare a camera the STAGE carries, e.g. ego=/World/Scene/EgoCam. Repeatable. The "
        "prim must already exist on the stage or Isaac's camera setup raises.",
    )
    cap.add_argument(
        "--asset",
        default=None,
        help="USD to reference onto the stage (default: Isaac's asset root + the bare G1 "
        "g1.usd, which has no table, no plate and no apple — see the runbook §4.2).",
    )
    cap.add_argument(
        "--scene",
        default=None,
        help="alias for --asset, the spelling configs/robot/isaac_g1.yaml uses (sim.scene). "
        "Passing both is refused: they are the same knob.",
    )
    cap.add_argument(
        "--render-hw",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="(H, W) of every render product. DEFAULTS TO THE COMMITTED CONTRACT's "
        "segmenter.pixel_grid_hw (configs/transfer25/pr08_geom_tol.json) and is refused when it "
        "disagrees with it — GEOM_TOL - EST_DRIFT_P95 is arithmetic on one grid only.",
    )
    cap.add_argument("--frames", type=int, default=64)
    cap.add_argument("--steps-per-frame", type=int, default=1)
    cap.add_argument(
        "--fake",
        action="store_true",
        help="drive FakeIsaacBinding instead of Isaac Sim. Exercises this path end to end on a "
        "laptop; the capture is stamped is_simulated_binding and can never be a gate input.",
    )

    mea = sub.add_parser("measure", help="PR-08 §4 steps 2-4 — estimate, compare, write the budget")
    mea.add_argument("--capture", type=Path, required=True)
    mea.add_argument(
        "--estimators",
        default="auto",
        help="importable module defining segment(rgb) and estimate_depth(rgb). 'auto' fails loudly "
        "and is the honest default until one is wired.",
    )
    mea.add_argument(
        "--object-class",
        default=None,
        help="the semantic label whose centroid is scored, looked up in the capture's idToLabels. "
        "DEFAULTS to the estimator module's own OBJECT_TEXT_PROMPT so the object cannot be named "
        "twice and disagree with itself; an explicit value naming a different object is fatal. "
        f"Falls back to {DEFAULT_OBJECT_CLASS!r} only for an estimator that declares no prompt, and "
        "that run is disqualified.",
    )
    mea.add_argument(
        "--arm",
        choices=ARM_CHOICES,
        default=DEFAULT_ARM,
        help="which segmenter topology is measured (default: %(default)s). `per_frame` is this "
        "adapter's segment(rgb), unchanged in every number and every field — passing nothing "
        "changes nothing and no artifact already on disk means anything different. `propagation` "
        "seeds a SAM2VideoPredictor from frame 0 and tracks the mask forward, which is "
        "Cosmos-Transfer2.5's own topology. `both` measures the SAME frames both ways and records "
        "the two p95s side by side, which is the discharge condition apple_sam2's third "
        "gate-qualification blocker names. IT DISCHARGES NOTHING: this produces evidence, and "
        "accepting evidence is a separate act and a person's.",
    )
    mea.add_argument(
        "--propagation-module",
        default=None,
        help="importable module defining propagate(rgbs) -> one mask per frame, used by --arm "
        f"propagation/both (default: {DEFAULT_PROPAGATION_MODULE}). Refused with --arm per_frame "
        "rather than accepted and ignored.",
    )
    mea.add_argument("--min-area-px", type=int, default=40)
    mea.add_argument("--largest-component", action="store_true", default=True)
    mea.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    mea.add_argument("--hist-bin-px", type=float, default=DEFAULT_HIST_BIN_PX)
    mea.add_argument("--limit", type=int, default=0)
    mea.add_argument("--out", type=Path, default=DEFAULT_OUT)

    args = ap.parse_args(argv)

    if args.cmd == "capture":
        try:
            # EVERYTHING THIS CAN REFUSE, REFUSED BEFORE THE BINDING IS CONSTRUCTED. On the real
            # path the next statement starts SimulationApp, loads a stage and resolves 43 DOFs.
            #
            # THE ISAAC PATH BELOW IS UNCHANGED. --backend defaults to `isaac`, every branch it
            # takes is the branch it took before, and the two refusals the mujoco branch adds
            # (--fake, --camera-prim) fire only when that backend was asked for by name.
            if args.backend == "mujoco":
                if args.fake:
                    raise EstimatorUnavailable(
                        "--fake is FakeIsaacBinding, which is the Isaac seam's laptop stand-in "
                        "and not a MuJoCo anything. There is no fake needed here: the mujoco "
                        "backend runs on CPU with no install, so the thing --fake exists to "
                        "stand in for is the thing this backend already is."
                    )
                if args.camera_prim:
                    raise EstimatorUnavailable(
                        "--camera-prim names a USD prim path and the mujoco backend loads an "
                        "MJCF, which has no prims. Its cameras are the ones the scene declares "
                        "(`head`, `wrist_left` in configs/sim/g1_scene.xml); select one with "
                        "--camera NAME."
                    )
            elif args.schedule is not None:
                # The isaac path is unchanged in every flag and refusal (PR-08-V5 §0). A schedule
                # it cannot honour is refused rather than accepted and ignored, because "accepted
                # and ignored" is how a capture header comes to carry a scene_schedule field that
                # names a schedule nothing drove.
                raise EstimatorUnavailable(
                    f"--schedule {args.schedule} names a MuJoCo scene schedule and this run is "
                    f"--backend {args.backend}, which drives no schedule at all: an Isaac capture "
                    "steps a loaded USD stage and this harness never places its objects. Pass "
                    "--backend mujoco, or drop the flag."
                )
            camera = args.camera or ("head" if args.backend == "mujoco" else
                                     next(iter(DEFAULT_CAMERA_PRIMS)))
            # `persp` and its prim path are an Isaac concept. On the mujoco backend the camera
            # is validated against the COMPILED MJCF instead — the answer lives there, and the
            # compile costs under a second, so the argument for checking it here (a typo may not
            # cost a GPU boot) does not apply and inventing a prim path for it would be fiction.
            prims = resolve_camera_prims(args.camera_prim)
            camera_prim = resolve_camera(camera, prims) if args.backend == "isaac" else None
            asset, asset_source = resolve_stage(args.asset, args.scene, args.backend)
            render_hw, render_hw_source = resolve_render_hw(args.render_hw)
            provenance = {
                "camera_prim": camera_prim,
                "camera_prims_declared": {
                    k: v for k, v in prims.items() if DEFAULT_CAMERA_PRIMS.get(k) != v
                },
                # The stage is provenance and not decoration: a capture of the bare g1.usd and a
                # capture of an apple-to-plate scene are indistinguishable in the frames if the
                # segmentation happens to be empty in both, and only one of them is a measurement
                # that failed. `asset: null` is the honest record of "Isaac's own default".
                "asset": asset,
                "asset_source": asset_source,
                "render_hw_requested": list(render_hw),
                "render_hw_source": render_hw_source,
                "geom_tol_contract": _artifact_label(GEOM_TOL_ARTIFACT),
            }
            if args.backend == "mujoco":
                from wam.robot.mujoco_binding import (  # noqa: PLC0415
                    SCENE_SCHEDULES,
                    MuJoCoGroundTruthBinding,
                )

                if tuple(SCENE_SCHEDULES) != SCENE_SCHEDULE_NAMES:
                    raise EstimatorUnavailable(
                        f"--schedule offers {list(SCENE_SCHEDULE_NAMES)} and "
                        f"wam.robot.mujoco_binding.SCENE_SCHEDULES defines "
                        f"{list(SCENE_SCHEDULES)}. One of the two grew a schedule the other does "
                        "not have; the CLI restates that table so --help works without MuJoCo "
                        "installed, and a silent disagreement is a schedule nobody can select."
                    )
                schedule_name = args.schedule or DEFAULT_SCENE_SCHEDULE
                requested_params = {
                    name: getattr(args, name) for name in TRAJECTORY_PARAM_FLAGS
                }
                if schedule_name != "trajectory":
                    given = sorted(k for k, v in requested_params.items() if v is not None)
                    if given:
                        raise EstimatorUnavailable(
                            f"{', '.join('--' + g.replace('_', '-') for g in given)} counts cycles "
                            f"along the smooth trajectory path and --schedule {schedule_name} has "
                            "no such path: the lattice is a sweep of DISTINCT configurations that "
                            "teleports the object between neighbours, so there is no revolution to "
                            "count. Accepting the flag and ignoring it would put a parameter in "
                            "the capture header that the capture did not use."
                        )

                if schedule_name == "trajectory":
                    if args.scene_states is not None:
                        raise EstimatorUnavailable(
                            "--schedule trajectory puts ONE configuration on every frame — that "
                            "is what makes the capture continuous — so --scene-states has "
                            "nothing to divide and would have to be ignored. Use --frames to say "
                            "how long the capture is; the path gets SMOOTHER as that grows, "
                            "because the per-frame increment is O(1/frames). For a capture split "
                            "into N distinct configurations, that is --schedule lattice."
                        )
                    n_states = int(args.frames)
                    steps_per_state = max(1, int(args.steps_per_frame))
                else:
                    n_states = (
                        DEFAULT_SCENE_STATES if args.scene_states is None else int(args.scene_states)
                    )
                    if n_states < 1:
                        raise EstimatorUnavailable(
                            f"--scene-states must be >= 1, got {args.scene_states}"
                        )
                    # THE CALLER OWNS THIS DIVISION, not the binding: `frames` and
                    # `steps_per_frame` live here and the binding only ever sees steps. Spelling
                    # it out means the header can record how many configurations were SCHEDULED
                    # beside how many were VISITED, and a run that ends early cannot be read as
                    # if it had covered the sweep.
                    total_steps = max(1, int(args.frames) * int(args.steps_per_frame))
                    steps_per_state = max(1, total_steps // n_states)
                factory = SCENE_SCHEDULES[schedule_name]
                if schedule_name == "trajectory":
                    schedule_params, schedule_params_source = trajectory_schedule_params(
                        factory, requested_params
                    )
                    schedule = factory(n_states, **schedule_params)
                else:
                    schedule_params, schedule_params_source = {}, "n/a (lattice takes none)"
                    schedule = factory(n_states)
                # RECORDED, not inferred. Until this landed, `turns`/`yaw_turns`/`arm_cycles` were
                # written down in no capture header, no artifact and no document, and the only way
                # to learn what a capture ran with was to read the function defaults at the commit
                # that produced it. V5 §5 registers a field list against exactly that, and V17 §7
                # requires this pair for every capture from here.
                provenance["scene_schedule_params"] = schedule_params
                provenance["scene_schedule_params_source"] = schedule_params_source
                provenance["scene_schedule"] = schedule_name
                provenance["scene_schedule_source"] = (
                    "--schedule" if args.schedule else f"default ({DEFAULT_SCENE_SCHEDULE})"
                )
                provenance["scene_schedule_is_temporally_coherent_by_design"] = (
                    schedule_name == "trajectory"
                )
                try:
                    binding = MuJoCoGroundTruthBinding(
                        scene=asset,
                        object_mesh=args.object_mesh,
                        cameras=(camera,),
                        render_hw=render_hw,
                        ground_truth=("depth", "segmentation"),
                        schedule=schedule,
                        steps_per_state=steps_per_state,
                    )
                except (FileNotFoundError, ValueError, RuntimeError) as exc:
                    # Every one of these is a refusal this module wrote deliberately (no mesh,
                    # unknown camera, a grid past the offscreen buffer, no MuJoCo). Re-raised as
                    # the harness's own failure so the operator gets `FATAL: ...` and exit 2
                    # rather than a traceback that reads like a crash.
                    raise EstimatorUnavailable(str(exc)) from exc
                provenance.update(binding.provenance())
            elif args.fake:
                from wam.robot.isaac_binding import FakeIsaacBinding

                binding = FakeIsaacBinding(
                    cameras=(camera,),
                    render_hw=render_hw,
                    ground_truth=("depth", "segmentation"),
                )
            else:
                from wam.robot.isaac_binding import IsaacSimBinding

                binding = IsaacSimBinding(
                    asset=asset,
                    cameras={camera: camera_prim},
                    render_hw=render_hw,
                    ground_truth=("depth", "segmentation"),
                )
            # WHICH LABEL `temporal_coherence` FOLLOWS, asked of the binding rather than typed
            # here. The mujoco binding declares it (`object_limitations.object_label`) and the
            # Isaac seam declares nothing, which is exactly the case that has to record an
            # absence with a reason instead of a zero — so this is a lookup with no fallback and
            # no default object name.
            capture_object_class = None
            limitations = getattr(binding, "limitations", None)
            if callable(limitations):
                capture_object_class = dict(limitations()).get("object_label")
            try:
                header = capture_frames(
                    binding,
                    camera,
                    args.frames,
                    args.out,
                    args.steps_per_frame,
                    provenance,
                    object_class=capture_object_class,
                )
                if args.backend == "mujoco":
                    # PR-08 §4.6's missing field, filled from the binding rather than from
                    # arithmetic over --frames: it is the count of configurations the run
                    # ACTUALLY applied.
                    header["n_scene_states_visited"] = int(binding.scene_states_visited)
                    (args.out / "capture.json").write_text(
                        json.dumps(header, indent=2) + "\n", encoding="utf-8"
                    )
                # The grid was REQUESTED before the boot; this is what came back. A binding that
                # rendered something else has produced a capture `measure` would disqualify, and
                # the operator should learn that from the run that made it rather than from the run
                # that reads it two machines later.
                if list(header.get("resolution_hw") or []) != list(render_hw):
                    raise EstimatorUnavailable(
                        f"asked for {list(render_hw)} and the binding rendered "
                        f"{header.get('resolution_hw')}. The frames under {args.out} are not on "
                        "GEOM_TOL's grid and `measure` would disqualify them with "
                        "resolution_disagrees_with_geom_tol; nothing downstream should read them."
                    )
            finally:
                # A leaked SimulationApp wedges the interpreter (isaac_binding's module docstring),
                # so the app the capture opened is closed on the way out whichever way it goes —
                # including the refusal three lines above, which is raised with Isaac still up.
                binding.close()
        except EstimatorUnavailable as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(header, indent=2))
        print(f"\nwrote {header['n_frames']} frames to {args.out}")
        if header["is_simulated_binding"]:
            print(
                "NOTE: is_simulated_binding=true — this capture is a pipeline test, not ground "
                "truth. `measure` will refuse it as a gate input."
            )
        return 0

    # -- measure ---------------------------------------------------------------------------------
    try:
        header, frames = load_capture(args.capture)
    except EstimatorUnavailable as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    n_found = len(frames)
    if args.limit:
        frames = frames[: args.limit]

    disqualified: list[str] = []
    if args.limit:
        disqualified.append("partial_run_limit")
    # The route the capture came from, or None. Resolved from the binding's CLASS NAME as
    # recorded in the header rather than from the header's own `ground_truth_route` string,
    # because the name is what capture_frames stamped `is_simulated_binding` off and a header
    # hand-edited to claim a route it did not come from must not be able to buy one.
    route = ground_truth_route(header.get("binding"))
    if header.get("is_simulated_binding", True) or route is None:
        # The reason string is KEPT at its 2026-08-21 spelling even though the check it names
        # is now an allow-list (GROUND_TRUTH_BINDINGS). It is quoted in the runbook's §4.7
        # table and in this repo's test fixtures, it is still literally true of everything it
        # fires on, and renaming a committed disqualifier vocabulary buys nothing here.
        disqualified.append("capture_is_not_from_isaac_sim")

    try:
        est = resolve_estimators(args.estimators)
    except EstimatorUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2
    # Opened before the first frame is read, so the counters it differences bracket exactly the
    # frames this run segments. An estimator that exports no stats() records an absence with a
    # reason and nothing else changes: the contract this harness enforces is segment/estimate_depth.
    stats_probe = EstimatorStatsProbe.open(est.module)
    scores_at_run_start = stats_probe.mark()

    if not est.gate_qualified:
        disqualified.append("estimator_not_gate_qualified")
    if est.segmenter_contract is None:
        # Not "probably the same segmenter". §4 step 2's requirement is unverifiable against a
        # module that will not say what it runs, and unverifiable is not satisfied.
        disqualified.append("estimator_does_not_declare_segmenter_contract")

    # Before any frame is read: a mismatch here makes every displacement in the run a distance
    # between two different objects, and there is no partial artifact worth writing from that.
    try:
        object_class, object_class_source, object_reasons = resolve_object_class(
            args.object_class, est
        )
    except EstimatorUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2
    disqualified += object_reasons

    resolution_hw = list(header.get("resolution_hw") or [])
    geom_reasons, geom_compare = cross_check_geom_tol(
        resolution_hw, est.name, est.segmenter_contract
    )
    disqualified += geom_reasons

    # WHICH ARMS RUN. `per_frame` is the default and is this script's whole previous behaviour;
    # everything below that is conditional on `run_propagation` is code a default run never
    # reaches. A propagation module named alongside `--arm per_frame` is REFUSED rather than
    # accepted and ignored — "accepted and ignored" is how an artifact comes to carry the name of
    # a module that never ran.
    run_per_frame = args.arm in ("per_frame", "both")
    run_propagation = args.arm in ("propagation", "both")
    if args.propagation_module and not run_propagation:
        print(
            f"FATAL: --propagation-module {args.propagation_module!r} names the second arm and "
            f"--arm is {args.arm!r}, which does not drive it. Nothing was written.",
            file=sys.stderr,
        )
        return 2
    propagator: Propagator | None = None
    if run_propagation:
        try:
            propagator = resolve_propagator(
                args.propagation_module or DEFAULT_PROPAGATION_MODULE
            )
        except EstimatorUnavailable as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 2
    if not run_per_frame:
        # est_drift_p95_px is the per-frame adapter's number by definition — PR-08 §6 subtracts
        # THAT from GEOM_TOL. A run that never measured it must not leave a propagation p95 in
        # the field, and must say why the field is null.
        disqualified.append("per_frame_arm_not_measured")

    # THE PREMISE OF THE COMPARISON, RECORDED RATHER THAN ASSUMED. See PixelWitness.
    per_frame_pixels = PixelWitness("per_frame")
    propagation_pixels = PixelWitness("propagation")

    pairs: list[tuple[tuple[float, float] | None, tuple[float, float] | None]] = []
    depth_stats: list[dict] = []
    vocab: set[str] = set()
    frames_without_label = 0
    shapes: set[tuple[int, int]] = set()
    # THE OTHER HALF OF "IS THE MASK RIGHT", collected in the same pass as the centroids because
    # both masks are already in hand. A displacement of 0.2 px between two centroids says the two
    # masks are centred on the same place; it says nothing about whether they are the same SHAPE,
    # and a mask covering the whole tabletop shares its centroid with the apple sitting in the
    # middle of it. See iou_distribution for what this is and, at length, for what it is not.
    gt_ious: list[float] = []
    frames_both_masks_empty = 0
    # PER-FRAME-INDEXED, aligned to the capture's own frame order and carrying None where the
    # frame could not be scored at all. `gt_ious` above is the same values with the gaps closed up,
    # which is what a distribution wants and is exactly what a RUN statistic must not be given:
    # closing the gaps would join two runs that were never adjacent.
    iou_per_frame: list[float | None] = []
    # The propagation arm needs the WHOLE clip — including frames whose ground truth carries no
    # object label, which the per-frame arm is not asked to segment. Skipping them would be
    # propagating across a cut, which is the thing the trajectory capture exists to avoid.
    propagation_rgbs: list[np.ndarray] = []
    propagation_true_masks: list[np.ndarray | None] = []

    for frame_index, d in enumerate(frames):
        rgb = np.load(d / "rgb.npy")
        true_depth = np.load(d / "depth.npy")
        true_ids = np.load(d / "seg_ids.npy")
        labels_raw = json.loads((d / "seg_labels.json").read_text(encoding="utf-8"))
        id_to_labels = {int(k): v for k, v in labels_raw.items()}
        shapes.add((int(true_ids.shape[0]), int(true_ids.shape[1])))

        if run_propagation:
            # THE SAME ARRAY OBJECT the per-frame arm is handed below — not a copy, not a re-read,
            # and emphatically not a re-encode. `show` records the digest that proves it.
            propagation_rgbs.append(propagation_pixels.show(frame_index, rgb))

        wanted, seen = object_ids(id_to_labels, object_class)
        vocab.update(seen)
        if not wanted:
            frames_without_label += 1
            pairs.append((None, None))
            iou_per_frame.append(None)
            if run_propagation:
                propagation_true_masks.append(None)
            continue

        true_mask = mask_from_ids(true_ids, wanted)
        if run_propagation:
            # Held only for the second pass. A default run keeps nothing it did not keep before.
            propagation_true_masks.append(true_mask)
        if not run_per_frame:
            # No estimated mask on this arm, so no pair and no IoU. Recorded as the absence it is
            # rather than skipped, so every list stays aligned to the capture's frame order.
            pairs.append((None, None))
            iou_per_frame.append(None)
            continue

        est_mask = est.segment(per_frame_pixels.show(frame_index, rgb))
        if est_mask.shape[:2] != true_mask.shape[:2]:
            print(
                f"FATAL: {d.name}: the estimator returned a {est_mask.shape[:2]} mask for a "
                f"{true_mask.shape[:2]} frame. A centroid compared across grids is not a "
                f"displacement.",
                file=sys.stderr,
            )
            return 2

        iou = mask_iou(est_mask, true_mask)
        iou_per_frame.append(iou)
        if iou is None:
            frames_both_masks_empty += 1
        else:
            gt_ious.append(iou)

        pairs.append(
            (
                centroid_of_mask(est_mask, args.largest_component, args.min_area_px),
                centroid_of_mask(true_mask, args.largest_component, args.min_area_px),
            )
        )
        depth_stats.append(depth_error(est.estimate_depth(rgb), true_depth, true_mask))

    # Mixed geometry is fatal for the same reason it is in measure_geom_tol: §6 subtracts these
    # pixels from GEOM_TOL's pixels, and that is arithmetic only on one grid.
    if len(shapes) > 1:
        print(
            f"FATAL: the capture mixes frame geometries {sorted(shapes)}. EST_DRIFT_P95 is in "
            f"pixels at one resolution and cannot be measured across several.",
            file=sys.stderr,
        )
        return 2

    # WHAT THE PER-FRAME ARM DID, SNAPSHOTTED HERE AND NOT AT ARTIFACT-WRITING TIME. The
    # propagation arm reaches apple_sam2._best_box for its frame-0 seed — deliberately, so that the
    # two arms share a detector rather than two copies of one — which moves this adapter's
    # counters. Closing the bracket before that happens keeps `estimator_stats` a description of
    # segment(rgb) alone, comparable field for field with every artifact written before --arm
    # existed. The propagation arm's own detection is recorded in arm_comparison.propagator.stats.
    estimator_stats_block = stats_probe.block(
        stats_probe.since(scores_at_run_start), include_raw=True
    )

    # -- the propagation arm ---------------------------------------------------------------------
    #
    # AFTER the per-frame pass and after the geometry refusal, on purpose: it is the expensive half
    # (one clip-level model load and a forward track over every frame), and a capture that is going
    # to be refused should be refused before the GPU is asked for anything.
    propagation_pairs: list[tuple[tuple[float, float] | None, tuple[float, float] | None]] = []
    propagation_iou_per_frame: list[float | None] = []
    propagation_gt_ious: list[float] = []
    propagation_both_empty = 0
    if run_propagation:
        assert propagator is not None
        try:
            propagated = propagator.propagate(propagation_rgbs)
        except Exception as exc:  # noqa: BLE001 - every refusal in that module is deliberate
            print(
                f"FATAL: the propagation arm refused: {exc}\n"
                "       Nothing was written. A propagation arm that cannot be seeded, or that "
                "cannot ingest the clip, is a refusal and not a p95.",
                file=sys.stderr,
            )
            return 2
        if len(propagated) != len(propagation_rgbs):
            print(
                f"FATAL: the propagation arm returned {len(propagated)} masks for "
                f"{len(propagation_rgbs)} frames. A mask list that is not frame-for-frame with "
                "the capture cannot be compared against it.",
                file=sys.stderr,
            )
            return 2
        for mask, true_mask in zip(propagated, propagation_true_masks):
            if true_mask is None:
                propagation_pairs.append((None, None))
                propagation_iou_per_frame.append(None)
                continue
            mask = np.asarray(mask)
            if mask.shape[:2] != true_mask.shape[:2]:
                print(
                    f"FATAL: the propagation arm returned a {mask.shape[:2]} mask for a "
                    f"{true_mask.shape[:2]} frame. A centroid compared across grids is not a "
                    "displacement.",
                    file=sys.stderr,
                )
                return 2
            iou = mask_iou(mask, true_mask)
            propagation_iou_per_frame.append(iou)
            if iou is None:
                propagation_both_empty += 1
            else:
                propagation_gt_ious.append(iou)
            propagation_pairs.append(
                (
                    centroid_of_mask(mask, args.largest_component, args.min_area_px),
                    centroid_of_mask(true_mask, args.largest_component, args.min_area_px),
                )
            )

    values, dropped = paired_displacements(pairs)
    n_steps = len(pairs)
    coverage = (values.size / n_steps) if n_steps else 0.0
    if coverage < args.min_coverage:
        disqualified.append("coverage_below_floor")

    p95 = float(np.percentile(values, 95)) if values.size else None
    headline_valid = bool(values.size) and coverage >= args.min_coverage

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "writeup": WRITEUP,
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "est_drift_p95_px": p95,
        "headline_valid": headline_valid,
        "gate_qualified": not disqualified,
        "gate_disqualified_reasons": disqualified,
        # PER-ROUTE SINCE 2026-08-22 (PR-08-V5), AND STILL NOT A FLAG: nothing on the command
        # line can move these, they are looked up from the binding the capture header names.
        #
        # THE FALLBACK IS THE OLD UNCONDITIONAL STAMP, DELIBERATELY. A capture from anything
        # that is not in GROUND_TRUTH_BINDINGS — the fake, or a stub — keeps the exact pair of
        # values this file stamped before V5, wording included, so that widening the route
        # moves NO number and NO string on the path that already existed. (That includes the
        # Humanoid-Everyday half of the Isaac reason string, which the runbook §7 defect 4
        # argues is stale; correcting it is a judgement for whoever owns PR-08 and is
        # deliberately not made here.) What such a capture gets instead is the honest new key
        # below plus `capture_is_not_from_isaac_sim` in the reasons — neither of which existed
        # to be changed. See GROUND_TRUTH_BINDINGS for why the MuJoCo row says what it says
        # and, more importantly, for what it refuses to say.
        "is_lower_bound": (route or GROUND_TRUTH_BINDINGS["IsaacSimBinding"])["is_lower_bound"],
        "is_lower_bound_reason": (route or GROUND_TRUTH_BINDINGS["IsaacSimBinding"])[
            "is_lower_bound_reason"
        ],
        # WHICH WAY THE ERROR POINTS, AND WHETHER ANYBODY MEASURED THAT. §6 SUBTRACTS this
        # number from GEOM_TOL, so the direction is the property that decides whether an error
        # in the budget lands in the generator's favour or against it — and `is_lower_bound:
        # false` on its own would read as "so it is an upper bound", which is a claim no route
        # here has earned.
        "error_direction": (
            route["error_direction"] if route else "unknown — not a ground-truth capture"
        ),
        "error_direction_measured": (
            bool(route["error_direction_measured"]) if route else False
        ),
        "ground_truth_route": header.get("ground_truth_route"),
        "object_class": object_class,
        # Where the object came from, because "apple" appearing in two places is exactly what this
        # field used to hide. `estimator_prompt` means nobody typed it twice.
        "object_class_source": object_class_source,
        "object_class_requested": args.object_class,
        "estimator_object_text_prompt": est.object_text_prompt,
        "label_vocabulary_seen": sorted(vocab),
        "n_frames": n_steps,
        "n_frames_found": n_found,
        "n_frames_without_object_label": frames_without_label,
        "n_measured": int(values.size),
        "n_dropped": int(dropped),
        "coverage": float(coverage),
        "min_coverage": float(args.min_coverage),
        "resolution_hw": resolution_hw,
        "units": "pixels at the capture resolution; depth error in metres",
        "estimators": {
            "spec": est.spec,
            "name": est.name,
            "version": est.version,
            "gate_qualified": est.gate_qualified,
        },
        # ADDITIVE AND READ-ONLY. Nothing here is in `gate_disqualified_reasons`, nothing here is
        # subtracted from anything, and recording it discharges no blocker — the adapter's second
        # one asks for these numbers AND for somebody to read them, and the second half is a human's
        # to do. `include_raw` is true because this capture is a few hundred frames, not the 171 600
        # of a GEOM_TOL pass: the raw values are what make the distribution re-derivable, and they
        # are small enough here to keep beside it.
        "estimator_stats": estimator_stats_block,
        "geom_tol_cross_check": geom_compare,
        "centroid_displacement": distribution(values, args.hist_bin_px),
        # ADDITIVE AND READ-ONLY, in the same class as estimator_stats and independent_samples:
        # est_drift_p95_px above is the budget PR-08 §6 names and is unchanged by this block
        # existing, no gate_disqualified_reasons entry depends on it, and run_g0_gates never reads
        # it. It is here because a centroid displacement cannot see the failure this adapter's
        # first blocker is actually about — "a plausible mask on the wrong object (the plate, the
        # hand, the whole tabletop) which produces a centroid, a displacement and a p95 that all
        # look like measurements". An IoU against the renderer's exact geom-id mask can.
        "mask_vs_ground_truth_iou": iou_distribution(gt_ious, frames_both_masks_empty),
        # HOW MANY INDEPENDENT OBSERVATIONS ARE UNDER THAT p95. Additive and read-only, in the
        # same class as estimator_stats: nothing reads it back and no disqualification reason
        # depends on it. See independent_sample_block for why a frame count is not a sample size
        # on a static-prop capture.
        "independent_samples": independent_sample_block(
            header,
            [
                (
                    None
                    if est is None or true is None
                    else float(np.hypot(est[0] - true[0], est[1] - true[1]))
                )
                for est, true in pairs
            ],
        ),
        "depth_absolute_error_over_object": depth_stats,
        "capture": {
            "path": str(args.capture),
            "binding": header.get("binding"),
            "is_simulated_binding": header.get("is_simulated_binding"),
            "camera": header.get("camera"),
            # WHICH STAGE THIS WAS THE TRUE SEGMENTATION OF. §4 step 1 renders "N Isaac episodes",
            # and an artifact that does not say what was in the scene cannot be audited later: a
            # p95 over the bare g1.usd (no table, no plate, no apple) and a p95 over an
            # apple-to-plate scene look identical in this file otherwise. Copied from the capture
            # header, which is written by the run that rendered the frames; `null` on a capture
            # made before these fields existed, which is itself the answer to "was it recorded".
            "asset": header.get("asset"),
            "asset_source": header.get("asset_source"),
            "camera_prim": header.get("camera_prim"),
            "backend": header.get("backend"),
            "ground_truth_route": header.get("ground_truth_route"),
            # WHAT OBJECT THIS BUDGET WAS MEASURED ON, carried as named fields rather than as a
            # sentence in a docstring. PR-08 §4 measures the estimator's error on the apple; a
            # simulator route measures it on whatever stands in for one, and an EST_DRIFT_P95
            # that can be read without also reading the stand-in is an EST_DRIFT_P95 that will
            # be. `null` for the Isaac route, which declares no such block.
            "object_limitations": header.get("object_limitations"),
            "n_scene_states_scheduled": header.get("n_scene_states_scheduled"),
            # PR-08 §4.6's "N counted in distinct configurations, not frames".
            "n_scene_states_visited": header.get("n_scene_states_visited"),
            # WHICH SCHEDULE, AND WHETHER THE OBJECT ACTUALLY MOVED. The name is provenance; the
            # block beside it is the measurement, and only one of the two can be wrong quietly.
            # `null` on every capture made before these fields existed, which is itself the
            # answer to "was it recorded".
            "scene_schedule": header.get("scene_schedule"),
            "scene_schedule_source": header.get("scene_schedule_source"),
            # AND WHICH PATH THAT SCHEDULE DREW. The name `trajectory` is shared by every capture
            # V17 §2 pools and by the three of §5's ladder that must never be pooled with them, and
            # the only thing that tells them apart is these three cycle counts. Copying them here
            # is not decoration: `pool_est_drift_arms.py` prints them per capture, and an artifact
            # that records the schedule's NAME and not its PARAMETERS says a capture was smooth
            # without saying how smooth. `null` on every capture made before the flags existed —
            # which is itself the answer to "was it recorded", and the answer is no.
            "scene_schedule_params": header.get("scene_schedule_params"),
            "scene_schedule_params_source": header.get("scene_schedule_params_source"),
            "temporal_coherence": header.get("temporal_coherence"),
            "render_hw_requested": header.get("render_hw_requested"),
            "render_hw_source": header.get("render_hw_source"),
            "captured_utc": header.get("captured_utc"),
        },
    }
    # THE TWO ARMS, SIDE BY SIDE — and only when a second arm actually ran. A default `--arm
    # per_frame` run writes exactly the document it wrote before this block existed; adding a key
    # whose value is "we did not do that" to every artifact on disk would be a change to every
    # artifact on disk.
    if run_propagation:
        # WHICH ARM `est_drift_p95_px` AT THE TOP OF THIS DOCUMENT IS, said in the artifact instead
        # of only in this file's comments. That field is the PER-FRAME arm by construction — it is a
        # percentile over `pairs`, which only the per-frame pass fills — and until this key existed
        # the sole record of that fact was the comment beside `per_frame_arm_not_measured` above. A
        # reader of a `--arm both` artifact therefore saw three p95s (one headline, one per arm)
        # with nothing saying which arm the headline was, and `measure_geom_tol.py
        # --carry-est-drift` read the headline and wrote it into the committed gate document — so
        # the arm that reached G0b was selected by a FIELD NAME and not by anybody. That carry now
        # refuses a two-arm artifact outright unless the operator names an arm (`--est-drift-arm`),
        # because which arm PR-08 §6 subtracts is an open owner decision and the bias between the
        # arms is recorded as two-sided; this key is the same fact stated from the producing side,
        # so the artifact is legible on its own without the consumer's refusal to explain it.
        #
        # WRITTEN ONLY ON A TWO-ARM DOCUMENT, for the identical reason `arm_comparison` itself is: a
        # default `--arm per_frame` run must keep writing exactly the document it wrote before this
        # existed. Where one arm ran there is no ambiguity to resolve, and adding a key to every
        # artifact on disk to say so would be a change to every artifact on disk. `null` when the
        # per-frame arm did not run at all — in which case `est_drift_p95_px` is null beside it and
        # `per_frame_arm_not_measured` is already in the reasons.
        artifact["est_drift_p95_px_arm"] = "per_frame" if run_per_frame else None
        artifact["arm_comparison"] = arm_comparison_block(
            arm_block(
                "per_frame",
                measured=run_per_frame,
                pairs=pairs,
                iou_per_frame=iou_per_frame,
                n_both_masks_empty=frames_both_masks_empty,
                hist_bin_px=args.hist_bin_px,
                absent_because=(
                    None if run_per_frame else f"--arm {args.arm} did not drive segment(rgb)"
                ),
                extra={
                    "topology": (
                        "GroundingDINO + SAM 2 re-run independently on every frame — "
                        "estimator.segment(rgb), the contract both PR-08 §4 harnesses call"
                    ),
                    "estimator": {
                        "spec": est.spec,
                        "name": est.name,
                        "version": est.version,
                        "gate_qualified": est.gate_qualified,
                    },
                },
            ),
            arm_block(
                "propagation",
                measured=True,
                pairs=propagation_pairs,
                iou_per_frame=propagation_iou_per_frame,
                n_both_masks_empty=propagation_both_empty,
                hist_bin_px=args.hist_bin_px,
                extra={
                    "topology": (
                        "one GroundingDINO detection on frame 0, one SAM2VideoPredictor seed, the "
                        "mask propagated forward across the clip — Cosmos-Transfer2.5's topology"
                    )
                },
            ),
            arms=[a for a, on in (("per_frame", run_per_frame), ("propagation", True)) if on],
            pixels=identical_input_pixels(per_frame_pixels, propagation_pixels),
            propagator=propagator,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({k: artifact[k] for k in (
        "est_drift_p95_px", "headline_valid", "gate_qualified",
        "gate_disqualified_reasons", "coverage", "n_measured", "n_dropped",
    )}, indent=2))
    print(f"\nwrote {args.out}")
    if not artifact["gate_qualified"]:
        print(
            "NOT GATE-QUALIFIED: this number must not be subtracted from GEOM_TOL in G0b.\n"
            "  reasons: " + ", ".join(disqualified)
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
