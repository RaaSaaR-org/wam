#!/usr/bin/env python3
"""Why is the robot mask EMPTY on a third of this corpus's frames — absent robot, or failed detector?

``cluster/discoverer/106_measure_robot_mask_area.sbatch`` PILOT (Slurm job 189707) measured an
empty-robot-mask rate of **35.2 %** over the first three episodes of
``data/pr08-apple-640x480-h264-lossless``. ``robot_composite.check_mask`` refuses a clip on an
empty mask with no threshold ("zero is zero"), so at that rate essentially every clip refuses and
PR-08 §6 G0c cannot run. The artifact cannot say WHICH of two opposite problems it measured:

* the robot is genuinely out of shot on those frames — in which case an empty mask is the CORRECT
  answer, there are no robot pixels to composite, and what needs revisiting is the refusal; or
* the robot is in shot and ``ROBOT_TEXT_PROMPT`` / the pinned detector do not find it — in which
  case the refusal is right and the detector is what needs fixing.

The full 9.47 GPU-h area measurement is worth nothing until that is settled: a distribution measured
with a detector that fails on a third of its frames is a distribution of the failure.

WHAT THIS MODULE DOES, AND WHAT IT DELIBERATELY DOES NOT DO
----------------------------------------------------------
It answers the question with two independent instruments and joins them frame by frame.

``visible``  A NON-LEARNED reference predicate for "the robot is in shot", run on the CPU over the
             corpus. It shares no component with GroundingDINO, so it can disagree with it. See
             :func:`robot_dark_mask` for exactly what it can and cannot see — that paragraph is the
             load-bearing one and it names the failure modes rather than claiming there are none.

``detect``   The REAL masker — ``robot_composite.build_masker()``, the committed prompt, the pinned
             checkpoints, the adapter's own thresholds — on a sample of frames, recording not just
             the mask area but WHY an empty mask was empty: no boxes at all, boxes that SAM 2
             segmented to nothing, or boxes that PR-08 V9's object-grounding filter dropped as the
             apple — three different findings behind one all-False array, and the masker returns
             the same array for all three. On the empty frames it additionally reads the detector's raw
             per-phrase scores back at threshold 0, which is what separates "nothing matched" from
             "matched weakly and got filtered". That readout is an EXTRA post-processing pass over
             an already-computed forward; it changes no threshold and feeds nothing back into the
             mask. The mask path is byte-identical to ``Sam2RobotMasker.mask`` and ``--verify``
             asserts that against the real method rather than asserting it in a comment.

``report``   Joins the two and prints the four numbers the question needs: the fraction of frames
             the reference predicate calls the robot visible; whether the empty frames cluster by
             episode and within an episode by time; the detector's score distribution on empty
             frames; and the 2x2 agreement table that is the actual verdict.

``blind``    ``blind-sheet`` / ``blind-score``: the one thing the three instruments above cannot
             produce between them — a human label written down BEFORE the masker's answer is known.
             Every other sheet here, and every human inspection this project holds, shows frames
             NOMINATED by a disagreement with the masker; such a sample can describe the
             disagreements and cannot bound the cell where both instruments are quiet together.
             Three labelled arms drawn from the empty-mask population, one of them uniform and
             therefore the only one that estimates anything; tiles that show pixels and an opaque
             id and nothing else; a sealed key; and a scorer that refuses a half-filled sheet. The
             section above :func:`blind_spot_score` states the problem in full.

**IT CHANGES NOTHING AND RECOMMENDS NOTHING IN CODE.** ``ROBOT_TEXT_PROMPT`` is a committed
constant with no override and this module does not set one; it imports the prompt to record it.
No threshold in ``robot_composite`` or ``estimators.apple_sam2`` is read from the environment here,
none is written, and ``check_mask`` is not called. Whether an empty mask should refuse a clip is a
gate decision and a person's; this produces the evidence that decision needs.

THE REFERENCE PREDICATE'S OWN NUMBERS ARE REPORTED WITH THEIR SENSITIVITY, NOT COMMITTED
----------------------------------------------------------------------------------------
Any pixel predicate has constants. Two of the three here are not new — they are the window
``build_identity_calibration.cloth_mask`` already commits, read from the other side (that function
keeps pixels within 45 counts of the modal luminance and its docstring states the reason this works:
"the Dex3 gripper sits 55-70 counts below the cloth"). The third, the change threshold, is this
module's own. ``visible`` therefore measures every frame at THREE settings of each and ``report``
prints the corpus fraction under all of them, so a reader can see whether the answer depends on a
number this file chose. If it does, the answer is "ambiguous" and this module says so.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from typing import Any

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

# luma / saturation / cloth_level come from the module that already commits this corpus's colour
# predicates. Redefining them here would be a second definition of "how bright is the cloth" that
# could drift from the one the calibration corpus was built with, which is the failure PR-13 is
# about.
from build_identity_calibration import cloth_level, luma, saturation  # noqa: E402

SCHEMA_VISIBLE = "wam.robot_visibility_reference/1"
SCHEMA_DETECT = "wam.robot_detector_diagnosis/1"
SCHEMA_REPORT = "wam.robot_mask_empty_diagnosis/1"

#: The reference predicate's settings. The FIRST of each tuple is the one the summary quotes; the
#: rest exist so ``report`` can say whether the answer depends on the choice. ``dark_offset`` and
#: ``sat_max`` centre on ``build_identity_calibration.cloth_mask``'s committed window (45 counts,
#: saturation 0.25); ``change_min`` is this module's own and is swept widest for that reason.
DARK_OFFSETS = (45, 35, 55)
SAT_MAXES = (0.25,)
CHANGE_MINS = (25, 15, 40)

#: Per-pixel background model stride. The background is a per-pixel temporal MEDIAN, so it survives
#: an arm that covers any given pixel for a minority of the clip; every fifth frame is enough to
#: estimate it and keeps a 749-frame episode inside a few hundred MB.
BACKGROUND_STRIDE = 5


class DiagnosisError(RuntimeError):
    """A refusal with a message meant for an operator, not a traceback."""


# --------------------------------------------------------------------------------------------
# the non-learned reference predicate
# --------------------------------------------------------------------------------------------


def background_median(frames: np.ndarray, stride: int = BACKGROUND_STRIDE) -> np.ndarray:
    """Per-pixel temporal median of an episode — the static scene, as float32 ``(H, W, 3)``.

    A median rather than a mean because the arm is an outlier that a mean would smear across every
    frame's background, and a median rather than the first frame because nothing guarantees the
    first frame is robot-free — that is the very thing being measured.
    """
    arr = np.asarray(frames)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise DiagnosisError(f"expected (N, H, W, 3) frames; got {arr.shape}.")
    if arr.shape[0] == 0:
        raise DiagnosisError("cannot build a background from zero frames.")
    step = max(1, int(stride))
    return np.median(arr[::step].astype(np.float32), axis=0)


def robot_dark_mask(
    frame: np.ndarray,
    background: np.ndarray,
    *,
    dark_offset: int = DARK_OFFSETS[0],
    sat_max: float = SAT_MAXES[0],
    change_min: int = CHANGE_MINS[0],
) -> np.ndarray:
    """``(H, W)`` bool: pixels that are dark, near-neutral, and not part of the static scene.

    WHAT IT SEES. The G1's arm and Dex3 hand are matte black over a mid-grey cloth. The three
    clauses are conjunctive and each removes a different confusor:

    * ``luma < cloth_level - dark_offset`` — dark RELATIVE TO THIS FRAME'S OWN modal luminance, so
      it tracks the exposure drift across the corpus instead of a fixed count.
      ``build_identity_calibration.cloth_mask`` commits the same 45-count window from the other
      side and states the measurement it rests on: the gripper sits 55-70 counts below the cloth.
    * ``saturation < sat_max`` — the apple is warm and saturated and its stem and shadow are dark.
      Without this clause the apple's own dark parts count as robot on every frame of every clip.
    * ``|frame - background| > change_min`` — the cloth's fold shadows and the dark band along the
      top edge are dark, neutral and STATIC. This clause is what removes them, and it is why the
      predicate needs a whole episode rather than one frame.

    WHAT IT CANNOT SEE, STATED PLAINLY BECAUSE IT BOUNDS EVERY CONCLUSION DRAWN FROM IT:

    * **The white wrist and the bare-metal segments.** The arm carries a white/silver forearm cuff
      that is BRIGHTER than the cloth. This predicate scores none of it. So on a frame showing only
      the wrist and nothing black, the predicate says "absent" and is wrong. It therefore
      UNDERSTATES robot presence, which is the safe direction here: an "absent" call from it is
      weaker evidence than a "present" call.
    * **A few pixels of gripper at the frame edge.** Area is area; a 30-pixel fingertip entering
      from the left is indistinguishable from noise, and the classification band below is what
      keeps such frames out of both buckets instead of forcing them into one.
    * **Anything else black that moves — including the arm's own SHADOW.** The arm casts a moving
      shadow on the cloth, and a shadow is dark, neutral and not in the background model. Observed
      on this corpus in the frames just before the arm enters: the predicate scores a few hundred
      to a couple of thousand pixels of shadow while the arm itself is still out of frame. So the
      predicate OVER-calls presence at the edges of the transition — which is the direction that
      makes a "present, and the masker returned nothing" count conservative rather than flattering,
      and it is a second reason the band exists. A dropped dark object or a human hand in dark
      sleeves would score too; nothing in this corpus's frames is either, but the predicate cannot
      itself certify that, which is what the contact sheets are for.
    * **Robot pixels that coincide with the background model.** If the arm parks in one place for
      most of an episode, the median absorbs it there. The measured traces do not show this (the
      signal returns to its floor at both ends of every episode), but it is the predicate's
      structural blind spot and a reason it can only bound, not certify.

    It is NOT ground truth and no gate may read it. It exists to disagree with GroundingDINO.
    """
    return apply_setting(
        frame_fields(frame, background),
        dark_offset=dark_offset,
        sat_max=sat_max,
        change_min=change_min,
    )


def frame_fields(frame: np.ndarray, background: np.ndarray) -> dict[str, Any]:
    """The three per-pixel fields :func:`robot_dark_mask`'s clauses compare, computed ONCE.

    Split out because the sweep in :func:`all_settings` differs only in the thresholds, and
    recomputing a histogram and two channel reductions per setting made a corpus pass cost hours
    instead of minutes. :func:`apply_setting` is the comparison; nothing else changes.
    """
    a = np.asarray(frame, dtype=np.float32)
    bg = np.asarray(background, dtype=np.float32)
    if a.shape != bg.shape:
        raise DiagnosisError(f"frame {a.shape} and background {bg.shape} differ.")
    return {
        "luma": luma(a),
        "cloth_level": cloth_level(a),
        "saturation": saturation(a),
        "change": np.abs(a - bg).max(2),
    }


def apply_setting(
    fields: dict[str, Any],
    *,
    dark_offset: int,
    sat_max: float,
    change_min: int,
) -> np.ndarray:
    dark = fields["luma"] < (fields["cloth_level"] - float(dark_offset))
    neutral = fields["saturation"] < float(sat_max)
    changed = fields["change"] > float(change_min)
    return np.asarray(dark & neutral & changed)


def largest_component(mask: np.ndarray) -> int:
    """Pixel count of the largest 8-connected component of ``mask``; 0 when it is empty.

    Reported beside the raw area because a scatter of isolated noise pixels and one arm-shaped blob
    of the same total area are different observations, and a compression artifact produces the
    first.
    """
    import cv2  # noqa: PLC0415

    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 0
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        m.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return 0
    return int(stats[1:, cv2.CC_STAT_AREA].max())


def classify(area_px: int, *, absent_below: int, present_above: int) -> str:
    """``"absent"`` / ``"ambiguous"`` / ``"present"`` from a reference-predicate area.

    A BAND rather than a threshold, deliberately. The predicate's traces are strongly bimodal and a
    single cut would hide the frames that sit between the modes — the arm entering or leaving shot,
    which is exactly the population whose classification the whole question turns on. Frames in the
    band are counted and reported as their own bucket, never silently assigned.
    """
    if absent_below > present_above:
        raise DiagnosisError(
            f"absent_below={absent_below} is above present_above={present_above}; the band would "
            "be inverted and every frame would be 'ambiguous'."
        )
    if area_px < absent_below:
        return "absent"
    if area_px > present_above:
        return "present"
    return "ambiguous"


# --------------------------------------------------------------------------------------------
# corpus access
# --------------------------------------------------------------------------------------------


def read_manifest(manifest: pathlib.Path) -> list[dict]:
    payload = json.loads(pathlib.Path(manifest).read_text(encoding="utf-8"))
    episodes = list(payload.get("episodes") or ())
    if not episodes:
        raise DiagnosisError(f"{manifest} lists no episodes.")
    return episodes


def decode(video: pathlib.Path) -> np.ndarray:
    """``(N, H, W, 3)`` uint8 RGB. Same cv2 path the generation venv uses, hence the H.264 tree."""
    import cv2  # noqa: PLC0415

    capture = cv2.VideoCapture(str(video))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise DiagnosisError(
            f"{video} decoded 0 frames. cv2 opens an AV1 container, believes its header and reads "
            "nothing — point this at the H.264 tree."
        )
    return np.stack(frames)


def select_episodes(episodes: list[dict], limit: int | None, stride: int) -> list[tuple[int, dict]]:
    """``(manifest_index, entry)`` pairs, at an episode stride, then truncated.

    The index is the manifest's own so a subsampled run can still say which episode a number came
    from.
    """
    picked = list(enumerate(episodes))[:: max(1, int(stride))]
    if limit is not None:
        picked = picked[: int(limit)]
    return picked


# --------------------------------------------------------------------------------------------
# `visible`
# --------------------------------------------------------------------------------------------


def episode_visibility(
    frames: np.ndarray,
    *,
    settings: list[dict],
    frame_stride: int = 1,
) -> dict:
    """Reference-predicate area per frame, at every setting, for one episode."""
    background = background_median(frames)
    indices = list(range(0, frames.shape[0], max(1, int(frame_stride))))
    areas: dict[str, list[int]] = {setting_key(s): [] for s in settings}
    largest: list[int] = []
    primary = setting_key(settings[0])
    for index in indices:
        fields = frame_fields(frames[index], background)
        for setting in settings:
            mask = apply_setting(fields, **setting)
            key = setting_key(setting)
            areas[key].append(int(np.count_nonzero(mask)))
            if key == primary:
                largest.append(largest_component(mask))
    return {
        "n_frames": len(indices),
        "frame_indices": indices,
        "areas": areas,
        "largest_component": largest,
    }


def setting_key(setting: dict) -> str:
    return "d{dark_offset}_s{sat_max}_c{change_min}".format(**setting)


def all_settings() -> list[dict]:
    """The primary setting first, then one variation of each swept constant."""
    primary = {
        "dark_offset": DARK_OFFSETS[0],
        "sat_max": SAT_MAXES[0],
        "change_min": CHANGE_MINS[0],
    }
    out = [primary]
    for dark in DARK_OFFSETS[1:]:
        out.append(primary | {"dark_offset": dark})
    for change in CHANGE_MINS[1:]:
        out.append(primary | {"change_min": change})
    for sat in SAT_MAXES[1:]:
        out.append(primary | {"sat_max": sat})
    return out


def cmd_visible(args: argparse.Namespace) -> int:
    manifest = pathlib.Path(args.manifest)
    episodes = read_manifest(manifest)
    settings = all_settings()
    picked = select_episodes(episodes, args.limit, args.episode_stride)
    per_episode: dict[str, Any] = {}
    for position, entry in picked:
        key = str(entry.get("id"))
        frames = decode(manifest.parent / str(entry["video"]))
        record = episode_visibility(frames, settings=settings, frame_stride=args.frame_stride)
        record["episode_index"] = position
        per_episode[key] = record
        primary = record["areas"][setting_key(settings[0])]
        print(
            f"{key}  {record['n_frames']:4d} frames  "
            f"median {int(np.median(primary)):6d} px  "
            f"min {min(primary):6d}  max {max(primary):6d}",
            flush=True,
        )
    payload = {
        "schema": SCHEMA_VISIBLE,
        "produced_by": "scripts/diagnose_robot_mask_empty.py visible",
        "instrument": "non-learned dark-motion predicate; NOT ground truth; see robot_dark_mask",
        "corpus": str(manifest.parent),
        "episode_stride": int(args.episode_stride),
        "frame_stride": int(args.frame_stride),
        "background_stride": BACKGROUND_STRIDE,
        "settings": {setting_key(s): s for s in settings},
        "primary_setting": setting_key(settings[0]),
        "per_episode": per_episode,
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


# --------------------------------------------------------------------------------------------
# `detect`
# --------------------------------------------------------------------------------------------


def stratified_plan(
    visible: dict,
    *,
    absent_below: int,
    present_above: int,
    per_bucket_per_episode: int,
    episodes: int | None = None,
) -> dict[str, list[int]]:
    """A detect plan that samples all three reference buckets in every episode it touches.

    A uniform sample would be ~40 % absent, ~55 % present and ~5 % ambiguous, which spends almost
    nothing on the band where the two instruments could actually disagree. Equal quotas per bucket
    put the GPU where the question is. THE RESULTING RATES ARE NOT CORPUS RATES and nothing may read
    them as such — the corpus rates come from ``visible``, which measured every frame it saw.
    """
    primary = visible["primary_setting"]
    keys = sorted(visible["per_episode"])
    if episodes is not None:
        keys = evenly(keys, int(episodes))
    plan: dict[str, list[int]] = {}
    for key in keys:
        record = visible["per_episode"][key]
        areas = record["areas"][primary]
        buckets: dict[str, list[int]] = {"absent": [], "ambiguous": [], "present": []}
        for slot, frame_index in enumerate(record["frame_indices"]):
            buckets[classify(areas[slot], absent_below=absent_below, present_above=present_above)]\
                .append(int(frame_index))
        picked = sorted({
            index
            for items in buckets.values()
            for index in evenly(items, int(per_bucket_per_episode))
        })
        if picked:
            plan[key] = picked
    return plan


def cmd_plan(args: argparse.Namespace) -> int:
    visible = json.loads(pathlib.Path(args.visible).read_text(encoding="utf-8"))
    plan = stratified_plan(
        visible,
        absent_below=int(args.absent_below),
        present_above=int(args.present_above),
        per_bucket_per_episode=int(args.per_bucket),
        episodes=args.episodes,
    )
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(plan) + "\n", encoding="utf-8")
    print(f"{len(plan)} episode(s), {sum(len(v) for v in plan.values())} frame(s) -> {args.out}")
    return 0


def detector_readout(masker: Any, frame: np.ndarray) -> dict:
    """One detector forward, post-processed TWICE, plus the SAM 2 mask the real path would produce.

    The first post-processing is at the adapter's live ``BOX_THRESHOLD`` / ``TEXT_THRESHOLD`` and
    its boxes are what go to SAM 2 — the same call, the same spelling, the same union-of-all-boxes
    rule as ``Sam2RobotMasker._boxes`` / ``.mask``. The second is at threshold 0 and its scores go
    nowhere: they are the record of what the detector had to offer before the threshold, which is
    the only thing that can tell "nothing matched" apart from "matched weakly and got filtered".

    Reimplemented here rather than called through the masker because the masker cannot report WHY
    a mask was empty — it returns all-False for "no boxes", for "SAM 2 segmented to nothing" and
    for "V9 dropped every box as the apple" alike, and those are three different findings.
    ``--verify`` asserts this path and ``Sam2RobotMasker.mask`` agree pixel for pixel on real
    frames.

    **WHAT IS NOT REIMPLEMENTED: THE OBJECT-GROUNDING FILTER.** PR-08 V9 drops a candidate whose
    SAM 2 mask IS the apple before the union (``ROBOT_MASK_OBJECT_MAX_IOU``), and this module
    predates it, which is exactly why the ``--verify`` guard below would have started firing on
    2026-08-23 (``docs/preregistration/PR-08-RESULT-2026-08-25-v12-preconditions.md`` §2 item 2).
    The repair calls ``Sam2RobotMasker.object_grounding_keep`` — the masker's own rule, the masker's
    own counters, the masker's own second opinion from ``apple_sam2`` — rather than re-typing its
    comparison here. Two implementations of one rule that never compare their answers is the drift
    PR-13 is about, and it is what this whole diagnosis is trying to detect one level up.

    Calling the masker's filter ADVANCES THE MASKER'S FILTER COUNTERS. Nothing in this module reads
    them and no artifact here reports them; a caller that differences ``filter_counters`` around a
    composite must not also run this readout on the same masker.
    """
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from robot_composite import ROBOT_TEXT_PROMPT  # noqa: PLC0415

    module = masker._estimator()
    processor, model = module._detector()
    rgb = module._as_uint8_rgb(frame)
    height, width = rgb.shape[:2]
    inputs = processor(
        images=Image.fromarray(rgb), text=ROBOT_TEXT_PROMPT, return_tensors="pt"
    ).to(module._device())
    with torch.inference_mode():
        outputs = model(**inputs)
    kept = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=module.BOX_THRESHOLD,
        text_threshold=module.TEXT_THRESHOLD,
        target_sizes=[(height, width)],
    )[0]
    # threshold 0.0 on BOTH knobs: every query the head produced, with its score and the phrase it
    # matched. Nothing downstream reads this; it is the diagnosis.
    raw = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=0.0,
        text_threshold=0.0,
        target_sizes=[(height, width)],
    )[0]
    raw_scores = np.asarray(raw["scores"].detach().cpu(), dtype=np.float64).reshape(-1)
    boxes = np.asarray(kept["boxes"].detach().cpu(), dtype=np.float64).reshape(-1, 4)
    kept_scores = np.asarray(kept["scores"].detach().cpu(), dtype=np.float64).reshape(-1)

    record = {
        "n_boxes_kept": int(boxes.shape[0]),
        "kept_scores": [round(float(s), 5) for s in kept_scores[:12]],
        "raw_n": int(raw_scores.size),
        "raw_max": float(raw_scores.max()) if raw_scores.size else 0.0,
        "raw_top5": [round(float(s), 5) for s in np.sort(raw_scores)[::-1][:5]],
    }
    if boxes.shape[0] == 0:
        record["n_detections_segmented"] = 0
        record["n_dropped_as_object"] = 0
        record["mask_px"] = 0
        record["empty_reason"] = "no_boxes_above_threshold"
        return record

    predictor = module._predictor()
    with torch.inference_mode():
        predictor.set_image(rgb)
        masks, _scores, _logits = predictor.predict(box=boxes, multimask_output=False)
    stacked = np.asarray(masks).reshape(-1, height, width) > 0

    # PR-08 V9, asked of the masker rather than answered here. ``object_grounding_keep`` scores each
    # candidate against the frame's own colour reference and returns the survivors; the union below
    # is the same one ``Sam2RobotMasker.mask`` performs on the same array.
    #
    # Asked ONCE. ``object_grounding_iou`` counts the frames whose colour reference was empty, so a
    # second call here to log the scores beside the decision would double that count and cost a
    # second pass over every candidate mask. The decision and its two counts are what this records.
    keep = np.asarray(masker.object_grounding_keep(rgb, stacked), dtype=bool)
    record["n_detections_segmented"] = int(stacked.shape[0])
    record["n_dropped_as_object"] = int(keep.size - np.count_nonzero(keep))

    if not keep.any():
        # The mask is empty because the filter emptied it — NOT because the detector found nothing.
        # Those are opposite findings about the same all-False array and the masker cannot tell
        # them apart, which is the reason this function exists at all.
        mask = np.zeros((height, width), dtype=bool)
        record["mask_px"] = 0
        record["empty_reason"] = "all_boxes_dropped_as_object"
        record["_mask"] = mask
        return record

    mask = np.ascontiguousarray(np.any(stacked[keep], axis=0))
    record["mask_px"] = int(np.count_nonzero(mask))
    record["empty_reason"] = None if record["mask_px"] else "sam2_segmented_nothing"
    record["_mask"] = mask
    return record


def cmd_detect(args: argparse.Namespace) -> int:
    from robot_composite import ROBOT_TEXT_PROMPT, build_masker  # noqa: PLC0415

    manifest = pathlib.Path(args.manifest)
    episodes = {str(e.get("id")): e for e in read_manifest(manifest)}
    plan = json.loads(pathlib.Path(args.plan).read_text(encoding="utf-8"))
    masker = build_masker()

    # THE FRAMES TO VERIFY ARE SPREAD ACROSS THE WHOLE PLAN, NOT ITS PREFIX. A prefix of this
    # corpus's plan is the opening seconds of one episode, where the robot is out of shot and both
    # paths return all-False — an agreement that would hold for two masker implementations that
    # agree about nothing. The spread guarantees the check lands on grounded frames too.
    ordered = [(key, index) for key, indices in plan.items() for index in indices]
    to_verify = set(evenly(ordered, int(args.verify)))

    verified = 0
    results: dict[str, list[dict]] = {}
    for key, indices in plan.items():
        entry = episodes.get(key)
        if entry is None:
            raise DiagnosisError(f"{key} is not in {manifest}.")
        frames = decode(manifest.parent / str(entry["video"]))
        rows: list[dict] = []
        for index in indices:
            record = detector_readout(masker, frames[index])
            mask = record.pop("_mask", None)
            if (key, index) in to_verify:
                reference = np.asarray(masker.mask(frames[index]), dtype=bool)
                mine = mask if mask is not None else np.zeros(reference.shape, dtype=bool)
                if not np.array_equal(reference, mine):
                    raise DiagnosisError(
                        f"{key} frame {index}: this module's mask path disagrees with "
                        "Sam2RobotMasker.mask. The diagnosis would be about a different masker "
                        "than the one G0c runs; nothing here is usable until that is fixed."
                    )
                verified += 1
            record["frame_index"] = int(index)
            rows.append(record)
        results[key] = rows
        empty = sum(1 for r in rows if r["mask_px"] == 0)
        print(f"{key}  {len(rows):4d} frames  empty {empty:4d}", flush=True)

    payload = {
        "schema": SCHEMA_DETECT,
        "produced_by": "scripts/diagnose_robot_mask_empty.py detect",
        "prompt": ROBOT_TEXT_PROMPT,
        "estimator": masker.provenance(),
        "verified_against_masker_frames": verified,
        "verified_note": (
            "Frames spread evenly across the plan (not its prefix) on which this module's mask path "
            "was asserted equal to Sam2RobotMasker.mask, pixel for pixel. 0 means the diagnosis was "
            "never checked against the masker G0c actually runs."
        ),
        "corpus": str(manifest.parent),
        "per_episode": results,
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


# --------------------------------------------------------------------------------------------
# `report`
# --------------------------------------------------------------------------------------------


def phase_bucket(index: int, n_frames: int, n_buckets: int = 10) -> int:
    """Which tenth of an episode's timeline ``index`` falls in. Clamped, never out of range."""
    if n_frames <= 0:
        raise DiagnosisError("an episode with no frames has no timeline.")
    return min(n_buckets - 1, max(0, int(index * n_buckets // n_frames)))


def runs_of(flags: list[bool]) -> list[tuple[int, int]]:
    """``(start, length)`` of every maximal run of True in ``flags``.

    A third of frames spread uniformly and a third in two blocks are the same rate and opposite
    findings, and run length is what tells them apart without a plot.
    """
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i - start))
            start = None
    if start is not None:
        out.append((start, len(flags) - start))
    return out


def contingency(pairs: list[tuple[str, bool]]) -> dict[str, int]:
    """2x2(+1) table of reference-predicate class against "the real mask was empty".

    ``present_empty`` is the cell that decides the question: the reference predicate says the robot
    is in shot and the committed masker returned nothing.
    """
    table = {
        "present_empty": 0,
        "present_nonempty": 0,
        "absent_empty": 0,
        "absent_nonempty": 0,
        "ambiguous_empty": 0,
        "ambiguous_nonempty": 0,
    }
    for label, empty in pairs:
        table[f"{label}_{'empty' if empty else 'nonempty'}"] += 1
    return table


def cmd_report(args: argparse.Namespace) -> int:
    visible = json.loads(pathlib.Path(args.visible).read_text(encoding="utf-8"))
    detect = (
        json.loads(pathlib.Path(args.detect).read_text(encoding="utf-8"))
        if args.detect
        else None
    )
    primary = visible["primary_setting"]
    lo, hi = int(args.absent_below), int(args.present_above)

    # -- 1. how often is the robot in shot at all, and does the answer depend on our constants --
    by_setting: dict[str, dict] = {}
    for key in visible["settings"]:
        counts = {"absent": 0, "ambiguous": 0, "present": 0}
        for record in visible["per_episode"].values():
            for area in record["areas"][key]:
                counts[classify(area, absent_below=lo, present_above=hi)] += 1
        total = sum(counts.values())
        by_setting[key] = {
            **counts,
            "n": total,
            "present_fraction": counts["present"] / total if total else 0.0,
            "absent_fraction": counts["absent"] / total if total else 0.0,
        }

    # -- 2. clustering: by episode, and within an episode by decile of the timeline --
    per_episode: dict[str, dict] = {}
    deciles = [0] * 10
    decile_totals = [0] * 10
    run_lengths: list[int] = []
    for key, record in visible["per_episode"].items():
        areas = record["areas"][primary]
        n = len(areas)
        labels = [classify(a, absent_below=lo, present_above=hi) for a in areas]
        absent_flags = [label == "absent" for label in labels]
        for i, flag in enumerate(absent_flags):
            bucket = phase_bucket(i, n)
            decile_totals[bucket] += 1
            if flag:
                deciles[bucket] += 1
        runs = runs_of(absent_flags)
        run_lengths.extend(length for _s, length in runs)
        per_episode[key] = {
            "n_frames": n,
            "absent": sum(absent_flags),
            "absent_fraction": sum(absent_flags) / n if n else 0.0,
            "n_absent_runs": len(runs),
            "longest_absent_run": max((length for _s, length in runs), default=0),
            "first_run_starts_at_0": bool(runs and runs[0][0] == 0),
            "last_run_ends_at_end": bool(runs and runs[-1][0] + runs[-1][1] == n),
        }

    summary: dict[str, Any] = {
        "schema": SCHEMA_REPORT,
        "produced_by": "scripts/diagnose_robot_mask_empty.py report",
        "instrument_note": (
            "The reference predicate is NOT ground truth and understates robot presence by "
            "construction (it scores no bright wrist pixels). An 'absent' call from it is weaker "
            "evidence than a 'present' call."
        ),
        "band": {"absent_below_px": lo, "present_above_px": hi},
        "visibility_by_setting": by_setting,
        "per_episode": per_episode,
        "absent_by_decile": [
            {"decile": i, "absent": deciles[i], "frames": decile_totals[i],
             "fraction": deciles[i] / decile_totals[i] if decile_totals[i] else 0.0}
            for i in range(10)
        ],
        "absent_run_lengths": {
            "n_runs": len(run_lengths),
            "median": float(np.median(run_lengths)) if run_lengths else 0.0,
            "max": max(run_lengths, default=0),
        },
    }

    if detect is not None:
        pairs: list[tuple[str, bool]] = []
        empty_raw_max: list[float] = []
        nonempty_raw_max: list[float] = []
        reasons: dict[str, int] = {}
        missing = 0
        for key, rows in detect["per_episode"].items():
            record = visible["per_episode"].get(key)
            if record is None:
                missing += len(rows)
                continue
            index_of = {f: i for i, f in enumerate(record["frame_indices"])}
            areas = record["areas"][primary]
            for row in rows:
                slot = index_of.get(row["frame_index"])
                if slot is None:
                    missing += 1
                    continue
                label = classify(areas[slot], absent_below=lo, present_above=hi)
                empty = row["mask_px"] == 0
                pairs.append((label, empty))
                (empty_raw_max if empty else nonempty_raw_max).append(row["raw_max"])
                if empty:
                    reasons[row["empty_reason"] or "?"] = reasons.get(row["empty_reason"] or "?", 0) + 1
        summary["detector"] = {
            "n_frames": len(pairs),
            "frames_without_a_reference_measurement": missing,
            "contingency": contingency(pairs),
            "empty_reasons": reasons,
            "raw_max_score_on_empty_frames": _quantiles(empty_raw_max),
            "raw_max_score_on_nonempty_frames": _quantiles(nonempty_raw_max),
            "box_threshold": detect["estimator"]["box_threshold"],
            "text_threshold": detect["estimator"]["text_threshold"],
            "prompt": detect["prompt"],
        }

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_episode"}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


def stratify(
    visible: dict,
    detect: dict,
    *,
    absent_below: int,
    present_above: int,
) -> dict[str, list[tuple[str, int, dict]]]:
    """Frames grouped by the cell of the agreement table they fall in.

    The stratum names are the finding: ``empty_but_reference_present`` is "the detector failed",
    ``empty_and_reference_absent`` is "the robot is out of shot", and nothing in the numbers can
    settle which population is which without somebody looking at the pixels. Hence the sheets.
    """
    primary = visible["primary_setting"]
    out: dict[str, list[tuple[str, int, dict]]] = {}
    for key, rows in detect["per_episode"].items():
        record = visible["per_episode"].get(key)
        if record is None:
            continue
        index_of = {f: i for i, f in enumerate(record["frame_indices"])}
        areas = record["areas"][primary]
        for row in rows:
            slot = index_of.get(row["frame_index"])
            if slot is None:
                continue
            label = classify(areas[slot], absent_below=absent_below, present_above=present_above)
            empty = row["mask_px"] == 0
            name = f"{'empty' if empty else 'nonempty'}_{'and' if empty else 'but'}_reference_{label}"
            out.setdefault(name, []).append((key, row["frame_index"], row | {"reference_px": areas[slot]}))
    return out


def evenly(items: list, k: int) -> list:
    """``k`` items spread across ``items`` — never the first ``k``, which would be one episode."""
    if k >= len(items) or k <= 0:
        return list(items)
    step = len(items) / float(k)
    return [items[min(len(items) - 1, int(i * step))] for i in range(k)]


def cmd_sheet(args: argparse.Namespace) -> int:
    """Contact sheets of the frames each cell of the agreement table holds. For a person to LOOK at.

    The numbers can say the detector and the reference predicate agree; they cannot say the
    reference predicate is right. Same rig and the same disclaimer as
    ``scripts/audit_apple_masks.py``: an overlay is evidence a reviewer can check, not a verdict.
    """
    from audit_apple_masks import boundary, captioned, contact_sheet  # noqa: PLC0415

    visible = json.loads(pathlib.Path(args.visible).read_text(encoding="utf-8"))
    detect = json.loads(pathlib.Path(args.detect).read_text(encoding="utf-8"))
    manifest = pathlib.Path(args.manifest)
    episodes = {str(e.get("id")): e for e in read_manifest(manifest)}
    strata = stratify(
        visible, detect, absent_below=int(args.absent_below), present_above=int(args.present_above)
    )
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted: dict[str, list] = {}
    for name, items in sorted(strata.items()):
        wanted[name] = evenly(sorted(items, key=lambda t: (t[0], t[1])), int(args.per_stratum))

    needed: dict[str, set[int]] = {}
    for items in wanted.values():
        for key, index, _row in items:
            needed.setdefault(key, set()).add(index)
    cache: dict[tuple[str, int], np.ndarray] = {}
    backgrounds: dict[str, np.ndarray] = {}
    for key, indices in needed.items():
        frames = decode(manifest.parent / str(episodes[key]["video"]))
        backgrounds[key] = background_median(frames)
        for index in indices:
            cache[(key, index)] = frames[index]

    # The committed masker, only when asked for: it costs a GPU and the sheets are useful without
    # it. With it, a reader sees BOTH instruments on one tile, which is what makes a disagreement
    # legible rather than merely counted.
    masker = None
    if args.with_robot_mask:
        from robot_composite import build_masker  # noqa: PLC0415

        masker = build_masker()

    written = []
    for name, items in wanted.items():
        tiles = []
        for key, index, row in items:
            frame = cache[(key, index)]
            mask = robot_dark_mask(frame, backgrounds[key])
            arr = np.asarray(frame, dtype=np.uint8).copy()
            if masker is not None:
                robot = np.asarray(masker.mask(frame), dtype=bool)
                if robot.any():
                    arr[robot] = (arr[robot] * 0.5 + np.array([0, 255, 90]) * 0.5).astype(np.uint8)
            # Magenta is the REFERENCE PREDICATE, not the robot mask. Different colour from the
            # apple audit's green on purpose: these two sheet families must never be confused.
            arr[boundary(mask, thickness=2)] = (255, 60, 200)
            tiles.append(captioned(arr, [
                f"{key} f{index:05d}",
                f"robot mask {row['mask_px']} px  boxes {row['n_boxes_kept']}",
                f"detector raw max {row['raw_max']:.4f}  ({row.get('empty_reason') or 'grounded'})",
                f"reference predicate {row['reference_px']} px  [magenta]",
            ]))
        if not tiles:
            continue
        sheet = contact_sheet(
            tiles,
            f"{name} | magenta = NON-LEARNED reference predicate (NOT ground truth)"
            + (" | green = COMMITTED robot mask" if masker is not None else "")
            + f" | "
            f"prompt {detect['prompt']!r}",
            cols=4,
        )
        path = out_dir / f"{name}.png"
        sheet.save(path)
        written.append(str(path))
        print(f"{name}: {len(tiles)} tile(s) of {len(strata[name])} -> {path}")
    print(json.dumps({n: len(v) for n, v in sorted(strata.items())}, indent=2))
    return 0 if written else 1


# --------------------------------------------------------------------------------------------
# `blind-sheet` / `blind-score` — a label assigned BEFORE the masker's answer is seen
# --------------------------------------------------------------------------------------------
#
# WHAT IS WRONG WITH EVERY HUMAN LABEL THIS PROJECT HOLDS, stated by
# ``docs/preregistration/PR-08-RESULT-2026-08-25-v12-preconditions.md`` §2:
#
#   "No frame anywhere carries a human label assigned BEFORE seeing the masker's answer. Every
#    human inspection on record was of frames nominated by a disagreement with the masker. And the
#    ``absent_empty`` cell — where a masker failure would hide — has had 12 of ~190 frames
#    adjudicated, by a predicate whose own docstring says it 'scores none of' the white/silver
#    forearm and therefore 'UNDERSTATES robot presence'."
#
# A sample nominated by a disagreement can only ever describe the disagreements. It cannot bound the
# cell where BOTH instruments are wrong in the same direction — the robot in shot, the masker empty,
# and the reference predicate quiet because the only thing in frame is the bright wrist it scores at
# zero. Three arms, drawn from the SAME empty-mask population, with the reason each exists:
#
#   uniform_random        the only one that estimates anything about the population. Drawn without
#                         consulting either instrument, so its (a)/(b) split is an unbiased estimate
#                         and its interval is a real bound on the blind spot.
#   predicate_nominated   what the existing sheets already show: the reference predicate says the
#                         robot is in shot and the committed masker returned nothing. Kept because
#                         it is where a failure is densest, and marked BIASED because it is.
#   blind_spot_targeted   deliberately over-weights frames the reference predicate CANNOT score —
#                         moving, near-neutral, and too bright for its dark clause. Also biased, in
#                         the opposite direction from the arm above, and by construction.
#
# The tiles are blind: pixels and an opaque id. If a reviewer can tell which arm a tile came from,
# or what the masker said about it, the label is conditioned on the answer again and the instrument
# is the one V12 §2 already refuses.

SCHEMA_BLIND_KEY = "wam.robot_mask_blind_adjudication_key/1"
SCHEMA_BLIND_LABELS = "wam.robot_mask_blind_adjudication_labels/1"
SCHEMA_BLIND_SCORE = "wam.robot_mask_blind_adjudication_score/1"

#: The only three answers a reviewer may give. ``undecidable`` is a REAL answer on a frame whose
#: corner might be a gripper or might be a fold shadow, and it is more useful than a guess — the
#: same rule ``audit_apple_masks``'s template states for its own verdicts.
BLIND_LABEL_VALUES = ("arm_present", "arm_absent", "undecidable")

BLIND_ARMS = ("uniform_random", "predicate_nominated", "blind_spot_targeted")

#: Defaults, and what they buy. 40 uniform tiles is the smallest draw whose 95 % Wilson upper bound
#: on an all-``arm_absent`` outcome lands under 9 % — i.e. the smallest draw that can say "if the
#: masker is missing robots in this population it is missing them in under a tenth of it" rather
#: than "we saw none". 12 predicate-nominated matches the count already adjudicated by hand so the
#: two are comparable; 24 blind-spot tiles is a deliberate over-sample of ~190 frames' worst tail.
DEFAULT_BLIND_N_UNIFORM = 40
DEFAULT_BLIND_N_PREDICATE = 12
DEFAULT_BLIND_N_BLIND_SPOT = 24
DEFAULT_BLIND_PER_SHEET = 12

#: Copied, not paraphrased, from ``scripts/audit_apple_masks.py``'s ``CORRELATED_OBSERVER``
#: (``audit_apple_masks.py:245-251``), which itself copies
#: ``runs/t040-identity-prompt/calibration-2/probe_observations.json``'s seed pass. The same
#: weakness in the same words, because it is the same weakness — plus one that belongs to this
#: instrument alone and is stated after the quotation.
BLIND_CORRELATED_OBSERVER = (
    "A MODEL FILLING THIS IN IS NOT THE MEASUREMENT. Copied verbatim from "
    "scripts/audit_apple_masks.py's CORRELATED_OBSERVER (audit_apple_masks.py:245-251), which "
    "copies runs/t040-identity-prompt/calibration-2/probe_observations.json's seed pass: \"If the "
    "`observed` fields below were written by a model rather than a person, say so in "
    "`established_by` and leave `looked_at` alone. A model looking at masks produced by a pipeline "
    "another model wired up is a CORRELATED OBSERVER: it is capable of reproducing the same "
    "misreading on both sides, and blocker 1 asks for a human.\" AND ONE MORE, WHICH IS THIS "
    "SHEET'S OWN: these tiles are blind so that the answer is written down before the pipeline's "
    "answer is known. A model with the repository in front of it can recover the pipeline's answer "
    "and would be unblinding itself, which costs this instrument the one property it has that the "
    "existing labels do not. Set `established_by` to whoever looked; leave `looked_at` false "
    "unless that was a person."
)


def blind_spot_score(
    fields: dict[str, Any],
    *,
    dark_offset: int = DARK_OFFSETS[0],
    sat_max: float = SAT_MAXES[0],
    change_min: int = CHANGE_MINS[0],
) -> int:
    """Pixels that MOVED and are near-neutral but are NOT dark — the reference predicate's blind spot.

    NOT A NEW DETECTOR, and deliberately not: it is :func:`apply_setting`'s own three clauses with
    the dark one complemented, over the same :func:`frame_fields`. So this and
    :func:`robot_dark_mask` partition the moving near-neutral pixels of a frame between them, which
    is asserted in the tests rather than claimed here. A frame scoring high is a frame where
    something entered the scene that the reference predicate scores at zero — the white/silver
    forearm cuff named at :func:`robot_dark_mask`'s docstring's first blind spot, and also, being a
    pixel predicate, anything else bright that moved. It ranks candidates for a human to LOOK at;
    it decides nothing and no gate may read it.
    """
    not_dark = fields["luma"] >= (fields["cloth_level"] - float(dark_offset))
    neutral = fields["saturation"] < float(sat_max)
    changed = fields["change"] > float(change_min)
    return int(np.count_nonzero(not_dark & neutral & changed))


def blind_population(
    visible: dict,
    detect: dict,
    *,
    absent_below: int,
    present_above: int,
) -> list[dict]:
    """Every frame in ``detect`` whose committed-masker mask was EMPTY, in a deterministic order.

    Frames the reference pass never measured are KEPT, with ``predicate_verdict`` ``"unmeasured"``.
    Dropping them would make the uniform arm a sample of "empty-mask frames the reference predicate
    also saw", which is a population defined by the instrument the arm exists to be independent of.
    """
    primary = visible.get("primary_setting")
    rows: list[dict] = []
    for key, records in detect["per_episode"].items():
        record = visible["per_episode"].get(key) if primary else None
        index_of = {f: i for i, f in enumerate(record["frame_indices"])} if record else {}
        areas = record["areas"][primary] if record else []
        for row in records:
            if int(row.get("mask_px", 0)) != 0:
                continue
            slot = index_of.get(row["frame_index"])
            rows.append({
                "episode": str(key),
                "frame_index": int(row["frame_index"]),
                "mask_px": int(row.get("mask_px", 0)),
                "empty_reason": row.get("empty_reason"),
                "predicate_verdict": (
                    classify(areas[slot], absent_below=absent_below, present_above=present_above)
                    if slot is not None else "unmeasured"
                ),
                "predicate_px": int(areas[slot]) if slot is not None else None,
            })
    return sorted(rows, key=lambda r: (r["episode"], r["frame_index"]))


def draw_blind_arms(
    population: list[dict],
    *,
    seed: int,
    n_uniform: int = DEFAULT_BLIND_N_UNIFORM,
    n_predicate: int = DEFAULT_BLIND_N_PREDICATE,
    n_blind_spot: int = DEFAULT_BLIND_N_BLIND_SPOT,
) -> list[dict]:
    """The three arms, drawn under ``seed``, shuffled together, and given opaque tile ids.

    ORDER MATTERS AND IS NOT ARBITRARY. The uniform arm is drawn FIRST, from the untouched
    population, so nothing conditions it — it is the only arm that estimates a rate and the only one
    an exclusion could bias. The other two then draw from what is left, so no frame is tiled twice
    and a reviewer cannot notice a duplicate; both are already biased samples by construction and
    are labelled as such wherever they are reported.

    The tile ids are assigned AFTER the shuffle, so their order carries no arm and no episode.
    """
    missing = [r for r in population if r.get("blind_spot_px") is None]
    if missing:
        raise DiagnosisError(
            f"{len(missing)} of {len(population)} candidate frames carry no blind_spot_px. The "
            "blind-spot arm cannot be drawn without it, and drawing the other two alone would "
            "leave the reference predicate's own blind spot unsampled — which is the cell this "
            "instrument exists to reach."
        )
    rng = random.Random(int(seed))
    ordered = sorted(population, key=lambda r: (r["episode"], r["frame_index"]))
    taken: set[tuple[str, int]] = set()

    def ident(row: dict) -> tuple[str, int]:
        return (row["episode"], int(row["frame_index"]))

    drawn: list[dict] = []

    def take(rows: list[dict], k: int, arm: str) -> None:
        picked = rng.sample(rows, min(max(int(k), 0), len(rows)))
        for row in picked:
            drawn.append(dict(row, arm=arm))
            taken.add(ident(row))

    take(list(ordered), n_uniform, "uniform_random")
    take([r for r in ordered if r["predicate_verdict"] == "present" and ident(r) not in taken],
         n_predicate, "predicate_nominated")
    # Deliberately NOT random: the point of this arm is the extreme of the ranking, and a random
    # draw from a "high enough" subset would need a second coined threshold to define the subset.
    ranked = sorted(
        (r for r in ordered if ident(r) not in taken and int(r["blind_spot_px"]) > 0),
        key=lambda r: (-int(r["blind_spot_px"]), r["episode"], r["frame_index"]),
    )[: max(int(n_blind_spot), 0)]
    for row in ranked:
        drawn.append(dict(row, arm="blind_spot_targeted"))
        taken.add(ident(row))

    rng.shuffle(drawn)
    for i, row in enumerate(drawn):
        row["tile"] = f"t{i:04d}"
    return drawn


def blind_sheet_title(page: int, pages: int) -> str:
    """The header a reviewer reads. It must not carry one bit about any tile on the sheet."""
    # Kept to one line and short on purpose: the header is drawn once across the sheet, so a long
    # one runs off the right edge of a narrow sheet and takes the instruction with it. The full
    # instruction lives in BLIND_LABELS.template.json, which the reviewer has open anyway.
    return (
        f"BLIND ADJUDICATION - sheet {page} of {pages} - per tile id write "
        "arm_present / arm_absent / undecidable. Nothing here says what any instrument answered."
    )


def blind_labels_template(tiles: list[str], sheets: list[str]) -> dict:
    """One blank row per tile. It carries no episode, no id of any kind but the tile's, no number.

    Everything a reviewer could condition an answer on lives in ``BLIND_KEY.json``, which is the
    sealed half of the pair and is not needed to fill this in.
    """
    return {
        "schema": SCHEMA_BLIND_LABELS,
        "produced_by": "scripts/diagnose_robot_mask_empty.py blind-sheet",
        "established_by": "",
        "established_by_note": BLIND_CORRELATED_OBSERVER,
        "how_to_use": (
            "Open the sheets. For each tile id, look at the picture and decide ONE thing: is any "
            "part of the robot - arm, wrist, hand, a fingertip at the border - visible in it? "
            "Write arm_present, arm_absent or undecidable in `label`, and what you saw in `note`. "
            "undecidable is a real answer and is worth more than a guess. Rename this file to "
            "BLIND_LABELS.json when every row is filled, then run `blind-score`."
        ),
        "label_values": list(BLIND_LABEL_VALUES),
        "sheets": list(sheets),
        "human_review": {"looked_at": False, "established_by": ""},
        "tiles": {tile: {"label": None, "note": ""} for tile in tiles},
    }


def blind_key_payload(
    drawn: list[dict],
    population: list[dict],
    *,
    seed: int,
    corpus: str,
    absent_below: int,
    present_above: int,
    requested: dict[str, int],
    sheets: list[str],
) -> dict:
    """The sealed half: what the tiles must not reveal, kept so the labels can be joined afterwards."""
    by_arm: dict[str, int] = {arm: 0 for arm in BLIND_ARMS}
    for row in drawn:
        by_arm[row["arm"]] += 1
    reasons: dict[str, int] = {}
    verdicts: dict[str, int] = {}
    for row in population:
        reasons[row["empty_reason"] or "?"] = reasons.get(row["empty_reason"] or "?", 0) + 1
        verdicts[row["predicate_verdict"]] = verdicts.get(row["predicate_verdict"], 0) + 1
    return {
        "schema": SCHEMA_BLIND_KEY,
        "produced_by": "scripts/diagnose_robot_mask_empty.py blind-sheet",
        "seal": (
            "THIS FILE IS THE ANSWER SHEET. Reading it before filling BLIND_LABELS.json in destroys "
            "the one property this instrument has that every existing label lacks: the label was "
            "written before the pipeline's answer was known."
        ),
        "seed": int(seed),
        "corpus": str(corpus),
        "band": {"absent_below_px": int(absent_below), "present_above_px": int(present_above)},
        "population": {
            "n_frames_with_an_empty_mask": len(population),
            "by_empty_reason": reasons,
            "by_reference_predicate": verdicts,
            "note": (
                "the sampling frame for all three arms: every frame in the detect artifact whose "
                "committed-masker mask was empty, including frames the reference pass never "
                "measured. It is NOT the corpus: the detect plan is stratified."
            ),
        },
        "arms": {
            arm: {
                "requested": int(requested.get(arm, 0)),
                "drawn": by_arm[arm],
                "sampling": _BLIND_ARM_SAMPLING[arm],
                "reporting": _BLIND_ARM_REPORTING[arm],
            }
            for arm in BLIND_ARMS
        },
        "sheets": list(sheets),
        "instrument_note": (
            "blind_spot_px is scripts/diagnose_robot_mask_empty.py's own pixel score, not a "
            "detector and not ground truth; it ranks frames for a person to look at. See "
            "blind_spot_score."
        ),
        "tiles": {
            row["tile"]: {
                "episode": row["episode"],
                "frame_index": int(row["frame_index"]),
                "arm": row["arm"],
                "mask_px": int(row["mask_px"]),
                "empty_reason": row["empty_reason"],
                "predicate_verdict": row["predicate_verdict"],
                "predicate_px": row["predicate_px"],
                "blind_spot_px": int(row["blind_spot_px"]),
            }
            for row in drawn
        },
    }


_BLIND_ARM_SAMPLING = {
    "uniform_random": (
        "uniformly at random, without replacement, from every frame of the population, consulting "
        "neither the reference predicate nor the blind-spot score. Drawn FIRST, so no exclusion "
        "touches it."
    ),
    "predicate_nominated": (
        "at random from the frames the reference predicate calls robot-PRESENT, minus anything the "
        "uniform arm already took. This is the cell the existing contact sheets already show."
    ),
    "blind_spot_targeted": (
        "the highest blind_spot_score frames remaining: moving, near-neutral and too BRIGHT for "
        "the reference predicate's dark clause. Ranked, not sampled, so the arm reaches the "
        "extreme rather than a random draw from a second coined threshold."
    ),
}

_BLIND_ARM_REPORTING = {
    "uniform_random": (
        "UNBIASED over the population above. This is the only arm whose split may be read as an "
        "estimate, and the only one whose interval bounds anything."
    ),
    "predicate_nominated": (
        "BIASED, deliberately and upward: conditioned on one instrument already disagreeing with "
        "the masker. Its split is not a rate over anything."
    ),
    "blind_spot_targeted": (
        "BIASED, deliberately: an over-sample of the frames the reference predicate cannot score. "
        "Its split is not a rate over anything."
    ),
}


def cmd_blind_sheet(args: argparse.Namespace) -> int:
    """Blind contact sheets, a sealed key and a blank labels template. Renders pixels ONLY.

    The masker is never loaded here: every mask this reads was already measured by ``detect``, and
    a sheet that recomputed one would be a sheet whose tiles could differ from the key it ships
    with.
    """
    from audit_apple_masks import captioned, contact_sheet  # noqa: PLC0415

    manifest = pathlib.Path(args.manifest)
    episodes = {str(e.get("id")): e for e in read_manifest(manifest)}
    visible = json.loads(pathlib.Path(args.visible).read_text(encoding="utf-8"))
    detect = json.loads(pathlib.Path(args.detect).read_text(encoding="utf-8"))
    population = blind_population(
        visible, detect,
        absent_below=int(args.absent_below), present_above=int(args.present_above),
    )
    if not population:
        raise DiagnosisError(
            f"{args.detect} records no frame whose mask was empty, so there is nothing to "
            "adjudicate. That is a finding about the detect artifact, not about the corpus."
        )

    frames: dict[tuple[str, int], np.ndarray] = {}
    for key in sorted({row["episode"] for row in population}):
        entry = episodes.get(key)
        if entry is None:
            raise DiagnosisError(f"{key} is not in {manifest}.")
        decoded = decode(manifest.parent / str(entry["video"]))
        background = background_median(decoded)
        for row in population:
            if row["episode"] != key:
                continue
            frame = np.asarray(decoded[row["frame_index"]], dtype=np.uint8)
            row["blind_spot_px"] = blind_spot_score(frame_fields(frame, background))
            frames[(key, row["frame_index"])] = frame
        del decoded

    requested = {
        "uniform_random": int(args.n_uniform),
        "predicate_nominated": int(args.n_predicate),
        "blind_spot_targeted": int(args.n_blind_spot),
    }
    drawn = draw_blind_arms(
        population,
        seed=int(args.seed),
        n_uniform=requested["uniform_random"],
        n_predicate=requested["predicate_nominated"],
        n_blind_spot=requested["blind_spot_targeted"],
    )

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sheet = max(1, int(args.per_sheet))
    pages = (len(drawn) + per_sheet - 1) // per_sheet
    sheets: list[str] = []
    for page in range(pages):
        chunk = drawn[page * per_sheet:(page + 1) * per_sheet]
        tiles = [
            captioned(frames[(row["episode"], int(row["frame_index"]))], [row["tile"]])
            for row in chunk
        ]
        name = f"blind_sheet_{page + 1:02d}.png"
        contact_sheet(tiles, blind_sheet_title(page + 1, pages), cols=int(args.cols)).save(
            out_dir / name)
        sheets.append(name)

    key_payload = blind_key_payload(
        drawn, population,
        seed=int(args.seed), corpus=str(manifest.parent),
        absent_below=int(args.absent_below), present_above=int(args.present_above),
        requested=requested, sheets=sheets,
    )
    (out_dir / "BLIND_KEY.json").write_text(json.dumps(key_payload, indent=2) + "\n", encoding="utf-8")
    (out_dir / "BLIND_LABELS.template.json").write_text(
        json.dumps(blind_labels_template([row["tile"] for row in drawn], sheets), indent=2) + "\n",
        encoding="utf-8",
    )

    counts = {arm: key_payload["arms"][arm]["drawn"] for arm in BLIND_ARMS}
    print(f"{len(population)} frame(s) with an empty mask -> {len(drawn)} blind tile(s) {counts}")
    print(f"{pages} sheet(s), BLIND_KEY.json and BLIND_LABELS.template.json -> {out_dir}")
    return 0


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for ``k`` of ``n``. ``(0.0, 1.0)`` when ``n`` is 0.

    Wilson rather than normal-approximation because every interesting outcome here is at or near
    zero, where the normal interval is a point at zero and would read as a proof.
    """
    if n <= 0:
        return (0.0, 1.0)
    k, n = int(k), int(n)
    denom = n + z * z
    centre = (k + z * z / 2.0) / denom
    half = (z / denom) * float(np.sqrt(k * (n - k) / n + z * z / 4.0))
    return (round(max(0.0, centre - half), 6), round(min(1.0, centre + half), 6))


def join_blind_labels(key: dict, labels: dict) -> list[dict]:
    """Key rows joined to their labels, or a refusal naming what is wrong. Never a partial score.

    Three refusals, all of them the same principle: a number computed over the tiles that happen to
    be filled in is a number over a sample selected by whoever stopped filling it in, and that is
    the exact defect this whole instrument exists to remove.
    """
    key_tiles = dict(key.get("tiles") or {})
    label_tiles = dict(labels.get("tiles") or {})
    only_key = sorted(set(key_tiles) - set(label_tiles))
    only_labels = sorted(set(label_tiles) - set(key_tiles))
    if only_key or only_labels:
        raise DiagnosisError(
            "the key and the labels file do not describe the same tiles: "
            f"{len(only_key)} in the key only ({only_key[:5]}), {len(only_labels)} in the labels "
            f"only ({only_labels[:5]}). They are halves of one draw and joining them by "
            "intersection would score a subset chosen after the fact."
        )
    blank = sorted(t for t, row in label_tiles.items() if not (row or {}).get("label"))
    if blank:
        raise DiagnosisError(
            f"{len(blank)} of {len(label_tiles)} tiles are still blank ({blank[:8]}). Scoring the "
            "filled ones would report a rate over the tiles a reviewer chose to answer, which is a "
            "nominated sample again."
        )
    bad = sorted(
        t for t, row in label_tiles.items() if row["label"] not in BLIND_LABEL_VALUES)
    if bad:
        raise DiagnosisError(
            f"{len(bad)} tiles carry a label outside {list(BLIND_LABEL_VALUES)} ({bad[:8]})."
        )
    return [
        dict(key_tiles[tile], tile=tile, label=label_tiles[tile]["label"],
             note=label_tiles[tile].get("note", ""))
        for tile in sorted(key_tiles)
    ]


def blind_score_report(key: dict, labels: dict) -> dict:
    """The (a)/(b) contingency per arm. A MEASUREMENT — it settles nothing and licenses nothing.

    (a) and (b) are V12 §2's own two cases for an empty mask: (a) the robot is genuinely out of
    shot, so the empty mask is the correct answer; (b) the robot is in shot and the masker returned
    nothing. Only the uniform arm's split is an estimate of anything; the other two are marked as
    the biased samples they are, in the artifact and not only here.
    """
    rows = join_blind_labels(key, labels)
    per_arm: dict[str, Any] = {}
    for arm in BLIND_ARMS:
        mine = [r for r in rows if r["arm"] == arm]
        a = sum(1 for r in mine if r["label"] == "arm_absent")
        b = sum(1 for r in mine if r["label"] == "arm_present")
        undecided = sum(1 for r in mine if r["label"] == "undecidable")
        reasons: dict[str, int] = {}
        for r in mine:
            if r["label"] == "arm_present":
                name = r.get("empty_reason") or "?"
                reasons[name] = reasons.get(name, 0) + 1
        per_arm[arm] = {
            "n": len(mine),
            "a_robot_absent": a,
            "b_robot_present_mask_empty": b,
            "undecidable": undecided,
            "b_rate_of_decided": (b / (a + b)) if (a + b) else None,
            "b_rate_ci95_wilson": list(wilson_interval(b, a + b)),
            "b_rate_upper_bound_counting_undecidable_as_b": (
                (b + undecided) / len(mine) if mine else None),
            "b_by_empty_reason": reasons,
            "sampling": (key.get("arms", {}).get(arm, {}) or {}).get("sampling"),
            "estimate": (key.get("arms", {}).get(arm, {}) or {}).get(
                "reporting", _BLIND_ARM_REPORTING[arm]),
        }
    return {
        "schema": SCHEMA_BLIND_SCORE,
        "produced_by": "scripts/diagnose_robot_mask_empty.py blind-score",
        "this_is_a_measurement": (
            "This file reports what a reviewer wrote down against a draw taken before they saw any "
            "instrument's answer. It settles no open question, releases no gate and permits no "
            "run. What it is FOR is the one thing the existing labels cannot do: put an interval "
            "on the cell where the masker and the reference predicate could be wrong together."
        ),
        "cells": {
            "a": "the robot is genuinely out of shot; an empty mask is the correct answer",
            "b": "the robot is in shot and the committed masker returned nothing",
            "undecidable": "the reviewer could not tell from the picture; counted, never assigned",
        },
        "seed": key.get("seed"),
        "corpus": key.get("corpus"),
        "population": key.get("population"),
        "n_tiles": len(rows),
        "labels_established_by": (
            labels.get("human_review", {}).get("established_by") or labels.get("established_by")),
        "labels_looked_at_by_a_person": bool(
            labels.get("human_review", {}).get("looked_at", False)),
        "correlated_observer_warning": BLIND_CORRELATED_OBSERVER,
        "per_arm": per_arm,
    }


def cmd_blind_score(args: argparse.Namespace) -> int:
    key = json.loads(pathlib.Path(args.key).read_text(encoding="utf-8"))
    labels = json.loads(pathlib.Path(args.labels).read_text(encoding="utf-8"))
    report = blind_score_report(key, labels)
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.out:
        print(f"\nwrote {args.out}")
    return 0


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    vis = sub.add_parser("visible", help="non-learned robot-visibility reference over the corpus")
    vis.add_argument("--manifest", required=True)
    vis.add_argument("--out", required=True)
    vis.add_argument("--limit", type=int, default=None)
    vis.add_argument("--episode-stride", type=int, default=1)
    vis.add_argument("--frame-stride", type=int, default=1)
    vis.set_defaults(func=cmd_visible)

    pln = sub.add_parser("plan", help="a detect plan with equal quotas per reference bucket")
    pln.add_argument("--visible", required=True)
    pln.add_argument("--out", required=True)
    pln.add_argument("--per-bucket", type=int, default=8)
    pln.add_argument("--episodes", type=int, default=None)
    pln.add_argument("--absent-below", type=int, required=True)
    pln.add_argument("--present-above", type=int, required=True)
    pln.set_defaults(func=cmd_plan)

    det = sub.add_parser("detect", help="the committed masker on planned frames, with reasons")
    det.add_argument("--manifest", required=True)
    det.add_argument("--plan", required=True, help='JSON {"episode_000000": [0, 5, ...], ...}')
    det.add_argument("--out", required=True)
    det.add_argument("--verify", type=int, default=24,
                     help="frames to assert against Sam2RobotMasker.mask (costs one extra forward "
                          "each; 0 disables and is not recommended)")
    det.set_defaults(func=cmd_detect)

    rep = sub.add_parser("report", help="join the two and print the four answers")
    rep.add_argument("--visible", required=True)
    rep.add_argument("--detect", default=None)
    rep.add_argument("--out", required=True)
    rep.add_argument("--absent-below", type=int, required=True,
                     help="reference-predicate area below which a frame is called robot-absent")
    rep.add_argument("--present-above", type=int, required=True,
                     help="area above which it is called robot-present; between the two is "
                          "'ambiguous' and is reported as its own bucket, never assigned")
    rep.set_defaults(func=cmd_report)

    sheet = sub.add_parser("sheet", help="contact sheets of each cell of the agreement table")
    sheet.add_argument("--manifest", required=True)
    sheet.add_argument("--visible", required=True)
    sheet.add_argument("--detect", required=True)
    sheet.add_argument("--out-dir", required=True)
    sheet.add_argument("--per-stratum", type=int, default=12)
    sheet.add_argument("--with-robot-mask", action="store_true",
                       help="also tint the COMMITTED robot mask green (needs the checkpoints and a "
                            "GPU); without it the sheets show the reference predicate alone")
    sheet.add_argument("--absent-below", type=int, required=True)
    sheet.add_argument("--present-above", type=int, required=True)
    sheet.set_defaults(func=cmd_sheet)

    blind = sub.add_parser(
        "blind-sheet",
        help="BLIND contact sheets of empty-mask frames, drawn in three labelled arms, plus a "
             "sealed key and a blank labels template",
    )
    blind.add_argument("--manifest", required=True)
    blind.add_argument("--visible", required=True)
    blind.add_argument("--detect", required=True)
    blind.add_argument("--out-dir", required=True)
    blind.add_argument("--seed", type=int, required=True,
                       help="REQUIRED and recorded in BLIND_KEY.json: it decides the uniform draw "
                            "and the tile shuffle, so the whole sheet is reproducible from it")
    blind.add_argument("--absent-below", type=int, required=True)
    blind.add_argument("--present-above", type=int, required=True)
    blind.add_argument("--n-uniform", type=int, default=DEFAULT_BLIND_N_UNIFORM,
                       help="tiles drawn uniformly at random from ALL empty-mask frames — the only "
                            "arm that estimates a rate (default: %(default)s)")
    blind.add_argument("--n-predicate", type=int, default=DEFAULT_BLIND_N_PREDICATE,
                       help="tiles the reference predicate calls robot-present; a BIASED sample "
                            "(default: %(default)s)")
    blind.add_argument("--n-blind-spot", type=int, default=DEFAULT_BLIND_N_BLIND_SPOT,
                       help="tiles over-weighting the frames the reference predicate cannot score, "
                            "i.e. bright and near-neutral; also BIASED (default: %(default)s)")
    blind.add_argument("--per-sheet", type=int, default=DEFAULT_BLIND_PER_SHEET)
    blind.add_argument("--cols", type=int, default=4)
    blind.set_defaults(func=cmd_blind_sheet)

    score = sub.add_parser(
        "blind-score",
        help="join a filled-in labels file against its key and report the (a)/(b) split per arm",
    )
    score.add_argument("--key", required=True, help="BLIND_KEY.json from blind-sheet")
    score.add_argument("--labels", required=True,
                       help="BLIND_LABELS.template.json with every `label` filled in")
    score.add_argument("--out", default=None)
    score.set_defaults(func=cmd_blind_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except DiagnosisError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
