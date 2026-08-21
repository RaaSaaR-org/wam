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

**It will not use a different segmenter from GEOM_TOL's.** §4 step 2 says "the *same* segmenter",
and §6 subtracts the two numbers. Two segmenters is two different quantities and the subtraction is
not arithmetic. When ``configs/transfer25/pr08_geom_tol.json`` exists, its recorded mask method is
read and a mismatch disqualifies the run rather than being noted in passing.

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

THE NUMBER IS A LOWER BOUND, AND THE ARTIFACT SAYS SO
-----------------------------------------------------
PR-08 §4's stated weakness, unedited: *"Isaac frames are not real frames, and a monocular
estimator's error on synthetic renders is not its error on RealSense footage -- plausibly
optimistic. So EST_DRIFT_P95 is a lower bound on the real error, it is recorded as such, and a G0b
margin that only clears under a lower bound is not a pass."*

``is_lower_bound`` is therefore stamped ``true`` unconditionally and is not a flag. The confirmatory
measurement against Humanoid Everyday's real depth is off the critical path because HE is unlicensed
(§4), and if that licence resolves this run is repeated rather than adjusted.

EXIT STATUS
-----------
0   measured with a gate-qualified estimator pair, coverage above ``--min-coverage``.
2   fatal: nothing was measured (no estimator, no capture, no object label, mixed geometry).
3   measured, but the number MUST NOT be used as G0b's budget -- ungated estimator, coverage below
    the floor, a partial run, a segmenter that disagrees with GEOM_TOL's, or a resolution that does.
    The artifact is still written, because "we tried and this is what came out" is a record.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from measure_geom_tol import (  # noqa: E402
    CANDIDATE_SEGMENTERS,
    _importable,
    _local_weight_hits,
    centroid_of_mask,
    distribution,
)

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


def capture_frames(binding: Any, camera: str, n_frames: int, out: Path, steps_per_frame: int) -> dict:
    """Drive an already-constructed binding and write ground truth to ``out``. Returns the header.

    Takes the binding rather than building one so that the caller owns the Isaac boot -- and so
    that this whole path is exercisable against ``FakeIsaacBinding`` on a laptop, which is the only
    reason any of it is testable before an Isaac node exists.

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
        "is_simulated_binding": type(binding).__name__ != "IsaacSimBinding",
    }
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


def cross_check_geom_tol(resolution_hw: list[int]) -> tuple[list[str], dict]:
    """Disqualifying reasons from GEOM_TOL's committed artifact, plus what was compared.

    §6 computes ``GEOM_TOL - EST_DRIFT_P95``. That subtraction is arithmetic only if both were
    measured in the same units on the same grid with the same estimator. Nothing else in the
    pipeline checks it, and a mismatch is invisible in the result: two plausible pixel numbers
    subtract to a plausible pixel number.
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
            },
        )
    doc = json.loads(GEOM_TOL_ARTIFACT.read_text(encoding="utf-8"))
    reasons: list[str] = []
    theirs_hw = doc.get("resolution_hw") or doc.get("frame_hw")
    if theirs_hw is not None and list(theirs_hw) != list(resolution_hw):
        reasons.append("resolution_disagrees_with_geom_tol")
    if not doc.get("gate_qualified", False):
        reasons.append("geom_tol_is_not_gate_qualified")
    return reasons, {
        "geom_tol_artifact": str(GEOM_TOL_ARTIFACT.relative_to(_REPO_ROOT)),
        "geom_tol_resolution_hw": theirs_hw,
        "geom_tol_gate_qualified": doc.get("gate_qualified"),
        "geom_tol_mask_method": doc.get("mask_method"),
        "this_resolution_hw": list(resolution_hw),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="PR-08 §4 step 1 — render Isaac ground truth to a dir")
    cap.add_argument("--out", type=Path, required=True)
    cap.add_argument("--camera", default="ego")
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
    mea.add_argument("--object-class", default=DEFAULT_OBJECT_CLASS)
    mea.add_argument("--min-area-px", type=int, default=40)
    mea.add_argument("--largest-component", action="store_true", default=True)
    mea.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    mea.add_argument("--hist-bin-px", type=float, default=DEFAULT_HIST_BIN_PX)
    mea.add_argument("--limit", type=int, default=0)
    mea.add_argument("--out", type=Path, default=DEFAULT_OUT)

    args = ap.parse_args(argv)

    if args.cmd == "capture":
        try:
            if args.fake:
                sys.path.insert(0, str(_REPO_ROOT / "src"))
                from wam.robot.isaac_binding import FakeIsaacBinding

                binding = FakeIsaacBinding(
                    cameras=(args.camera,), ground_truth=("depth", "segmentation")
                )
            else:
                sys.path.insert(0, str(_REPO_ROOT / "src"))
                from wam.robot.isaac_binding import IsaacSimBinding

                binding = IsaacSimBinding(ground_truth=("depth", "segmentation"))
            header = capture_frames(
                binding, args.camera, args.frames, args.out, args.steps_per_frame
            )
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
    if header.get("is_simulated_binding", True):
        disqualified.append("capture_is_not_from_isaac_sim")

    try:
        est = resolve_estimators(args.estimators)
    except EstimatorUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not est.gate_qualified:
        disqualified.append("estimator_not_gate_qualified")

    resolution_hw = list(header.get("resolution_hw") or [])
    geom_reasons, geom_compare = cross_check_geom_tol(resolution_hw)
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

        wanted, seen = object_ids(id_to_labels, args.object_class)
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
        # Unconditional, and not a flag. PR-08 §4: Isaac frames are not real frames, so a monocular
        # estimator's error on synthetic renders is plausibly optimistic. A G0b margin that only
        # clears under a lower bound is not a pass.
        "is_lower_bound": True,
        "is_lower_bound_reason": (
            "measured on Isaac renders, not RealSense footage (PR-08 §4). The confirmatory "
            "measurement against Humanoid Everyday is blocked on that corpus's licence and is "
            "deliberately off the critical path."
        ),
        "object_class": args.object_class,
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
        "geom_tol_cross_check": geom_compare,
        "centroid_displacement": distribution(values, args.hist_bin_px),
        "depth_absolute_error_over_object": depth_stats,
        "capture": {
            "path": str(args.capture),
            "binding": header.get("binding"),
            "is_simulated_binding": header.get("is_simulated_binding"),
            "camera": header.get("camera"),
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
