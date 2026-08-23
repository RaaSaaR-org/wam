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

#: The COMMITTED gate artifact, beside GEOM_TOL's and for the same reason: §8 item 4 wants both
#: measured *and committed* before generation, and a path under gitignored ``runs/`` cannot be that.
DEFAULT_OUT_REL = "configs/transfer25/pr08_est_drift.json"
DEFAULT_OUT = _REPO_ROOT / DEFAULT_OUT_REL

#: GEOM_TOL's artifact. Read to cross-check the segmenter and the pixel grid, never written.
GEOM_TOL_ARTIFACT = _REPO_ROOT / "configs/transfer25/pr08_geom_tol.json"

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
        default=20,
        help="mujoco backend only: how many DISTINCT scene configurations the capture spans "
        "(default: %(default)s). PR-08 §4.6: a p95 over N frames of one pose is a percentile "
        "over one viewpoint, so N is counted in configurations and not in frames. The frames "
        "are divided evenly across them and the count lands in the capture header.",
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
                    MuJoCoGroundTruthBinding,
                    default_scene_schedule,
                )

                if args.scene_states < 1:
                    raise EstimatorUnavailable(
                        f"--scene-states must be >= 1, got {args.scene_states}"
                    )
                # THE CALLER OWNS THIS DIVISION, not the binding: `frames` and
                # `steps_per_frame` live here and the binding only ever sees steps. Spelling it
                # out means the header can record how many configurations were SCHEDULED beside
                # how many were VISITED, and a run that ends early cannot be read as if it had
                # covered the sweep.
                total_steps = max(1, int(args.frames) * int(args.steps_per_frame))
                steps_per_state = max(1, total_steps // int(args.scene_states))
                schedule = default_scene_schedule(int(args.scene_states))
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
            try:
                header = capture_frames(
                    binding, camera, args.frames, args.out, args.steps_per_frame, provenance
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

    pairs: list[tuple[tuple[float, float] | None, tuple[float, float] | None]] = []
    depth_stats: list[dict] = []
    vocab: set[str] = set()
    frames_without_label = 0
    shapes: set[tuple[int, int]] = set()

    for d in frames:
        rgb = np.load(d / "rgb.npy")
        true_depth = np.load(d / "depth.npy")
        true_ids = np.load(d / "seg_ids.npy")
        labels_raw = json.loads((d / "seg_labels.json").read_text(encoding="utf-8"))
        id_to_labels = {int(k): v for k, v in labels_raw.items()}
        shapes.add((int(true_ids.shape[0]), int(true_ids.shape[1])))

        wanted, seen = object_ids(id_to_labels, object_class)
        vocab.update(seen)
        if not wanted:
            frames_without_label += 1
            pairs.append((None, None))
            continue

        true_mask = mask_from_ids(true_ids, wanted)
        est_mask = est.segment(rgb)
        if est_mask.shape[:2] != true_mask.shape[:2]:
            print(
                f"FATAL: {d.name}: the estimator returned a {est_mask.shape[:2]} mask for a "
                f"{true_mask.shape[:2]} frame. A centroid compared across grids is not a "
                f"displacement.",
                file=sys.stderr,
            )
            return 2

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
        "estimator_stats": stats_probe.block(
            stats_probe.since(scores_at_run_start), include_raw=True
        ),
        "geom_tol_cross_check": geom_compare,
        "centroid_displacement": distribution(values, args.hist_bin_px),
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
            "render_hw_requested": header.get("render_hw_requested"),
            "render_hw_source": header.get("render_hw_source"),
            "captured_utc": header.get("captured_utc"),
        },
    }
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
