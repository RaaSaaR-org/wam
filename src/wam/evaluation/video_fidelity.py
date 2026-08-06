"""Score a generated clip against the real future it claims to predict (T-38).

The generate path has never been scored. ``scripts/hf_job_cosmos3_probe.py:408`` writes an mp4
and checks ``out_path.stat().st_size > 0``; every statement this repo makes about those clips
(``docs/hf_jobs.md``, 2026-07-30 onward) is a description of what they looked like. This module
is the arithmetic that was missing, and it exists for one question: is either backbone's
generated video close enough to the truth to be worth using as synthetic training data.

**A distance alone answers nothing, so it is never reported without its controls.** Two of them
say what a trivial system already achieves, one says what a perfect one would score here, and
one is the scale's zero and nothing more.

``frozen``
    Repeat the conditioning frame for every predicted step. This is T-36's control and it has
    already won once: the anchored Wan dream scored 16.656 against freezing's 12.020, i.e. the
    prediction was **39 % further from the truth than standing still**
    (``docs/preregistration/PR-06-RESULT.md``). It is the bar, not a courtesy.
``truth``
    The real future against itself. **0 by construction, and not a check.** This arm is
    ``frame_metrics(truth, truth)`` — one array twice — so it stays 0 for any indices, any pair
    of frame rates and any resize kernel. An earlier version of this docstring claimed it would
    notice an off-by-one; it does not, and scoring the same clip with the window moved 21 frames
    reports ``truth`` 0.0 either way. The only thing a nonzero value there can mean is a
    non-finite pixel in the source, which would otherwise poison every arm silently, and that is
    all :func:`check_truth_is_zero` is. **Nothing in this module can verify the alignment**: the
    clock and the start frame are facts about the run, not about the arrays. What the report can
    do is publish ``alignment.source_indices`` so a reader recomputes the mapping from the
    parquet, and the CLI additionally flags a generated container whose declared fps disagrees
    with ``--generated-fps``, which is the one clock error the files themselves reveal.
``other``
    A different episode's real future, read at the same relative offsets from one absolute frame
    of that episode. A **retrieval** baseline — what a system could get by looking up another
    demo — and *not* a phase-matched one, which this docstring used to claim it was. The offset
    is an absolute index into an episode of a different length (GR00T-AppleToPlate runs 249 to
    749 frames, mean 427), so the same index is a different point in the task, and the arm's
    strength swings with it. Measured at ep0@271, 72 frames, sweeping the offset into ep1,
    other/frozen ``mean_abs``: 1.118 (200), 1.033 (240), 0.978 (246), 0.829 (260), 0.697 (271),
    0.522 (290), 0.810 (310). The old claim was also backwards: 246 IS the phase-computed offset
    (271/590 of ep1's 535 frames) and at 0.978 it is the weaker control, so computing the phase
    is not the repair.

    What this arm has to be, to carry a ``beats_chance`` verdict, is an upper bound on lookup.
    So the CLI searches: it scores the control at every offset on a grid around the requested
    one, keeps the one closest to the truth, and publishes the whole curve in
    ``info["other_offset_sweep"]``. Grid +-120 frames, stride 10, against the CLI's old
    single-offset default, other/frozen ``mean_abs``: 0.533 vs 0.697 (ep0@271, 72f), 0.593 vs
    0.743 (ep10@250, 48f), 0.716 vs 0.991 (ep0@400, 48f), 0.839 vs 1.027 (ep2@200, 72f), 1.295
    vs 1.349 (ep0@150, 72f) — stronger in all five, by a quarter in two of them. Choosing by
    distance to the truth is an oracle no retrieval system has; it is deliberate, it only ever
    makes the gate harder to pass, and it is still a LOWER bound on what retrieval over 402
    episodes would reach, because it looks inside one bounded window of one neighbouring episode.

    The arm has to be reported at all because it lands *below* the frozen bar on most of this
    corpus: four of those five windows once searched (0.533, 0.593, 0.716, 0.839), and two of
    them even at the old single offset. A clip that merely matches another demo of the task
    would collect a "beats frozen" verdict here, which is why there are two verdicts.
``codec_floor``
    Optional, and the only arm that says what a *perfect* prediction would score here. The model
    arm is read out of a lossy container (``export_to_video`` writes libx264,
    ``hf_job_wan_probe.py:944``) while ``frozen`` and ``truth`` come straight off the source
    decode, so the model pays an encode the controls never pay. Measured on ep0, 72 generated
    frames at 24 fps from frame 271 (source 271..360): the byte-exact true future written with
    libx264 and read back scores ``gradient_abs`` 1.111 against the frozen bar's 2.417 — ratio
    **0.460**, where the same frames without the round trip score 0.000. On ``mean_abs`` the
    same round trip costs 0.076 of the bar and on ``mse`` 0.003, so the contamination is
    specific to the gradient metric: codec ringing is high frequency and the frozen gradient bar
    is tiny because 96 % of this corpus's frame pairs barely move (T-35). Across the five
    windows above the floor runs 0.460 to 0.723, worst on the quietest one (ep0@400) — on a
    still window nearly three quarters of the gradient bar is codec noise. A reported
    ``gradient_abs`` ratio of 0.55 is therefore not "captured half the structure"; it is below
    the floor for some windows and barely above it for the rest. Reproduce with
    ``scripts/score_generated_video.py --episode 0 --start-frame 271 --generated-fps 24``.

The headline is ``ratio_to_frozen["model"]["mean_abs"]``, because that ratio is the quantity
T-36 already established. **It is comparable with another report of the same window and with no
other report**, which an earlier version of this docstring denied when it called the ratio "the
one number comparable across geometries, clip lengths, backbones and runs". The ratio divides by
the frozen bar and the frozen bar is a fact about how much the scene happened to move: measured
here it runs 5.597 ``mean_abs`` (ep0@400, 48f) to 25.639 (ep2@200, 72f), a factor of 4.6. So an
arm of CONSTANT quality lands wherever the window puts it — ``codec_floor``, always the same
libx264 round trip, scores 0.074 (ep2@200), 0.076 (ep0@271), 0.085 (ep10@250), 0.178 (ep0@150)
and 0.296 (ep0@400) of the bar. The pre-registered margin is 10 %, so the window moves the
headline by up to nine times the quantity a verdict turns on: score Wan on ep0@271 and Cosmos3
on ep0@400, compare the ratios, and the backbone ranking published is a ranking of two windows.
:func:`check_same_window` is the refusal that makes that fail instead of publish, and
``scripts/score_generated_video.py --compare`` is it wired to two report files.

It has to clear two bars, not one: ``verdicts["beats_frozen"]`` and
``verdicts["beats_chance"]``. A PREDICTS without a BEATS_CHANCE beside it says the clip is
closer to the truth than standing still and no closer than a different demo of the same task,
which is a retrieval result rather than a prediction.

**What a pixel metric cannot see, stated plainly.** It cannot see whether the robot is the right
robot. The qualitative runs found both priors keep the visible black Dex3 hand and then invent a
generic manipulator for the arm that was never in frame — Wan a green-white tube, Cosmos3 a
bulky black-silver mechanism with a rod artifact (``docs/hf_jobs.md``). An invented arm occupies
a small, moving fraction of a 480x640 frame, and a mean over all pixels charges it almost
nothing. So a clip can score well on everything below and still be useless as training data
because it shows the wrong embodiment. These metrics measure distance to the truth; they do not
measure whether the video is of our robot, and no threshold on them may be read as if it did.

SSIM, LPIPS and FVD are deliberately absent. ``scipy`` and ``scikit-image`` are not dependencies
of this project and FVD needs an I3D checkpoint over a network; a perceptual metric
re-implemented from memory is a number nobody can reproduce. The one metric here that global
brightness cannot move is a first-difference gradient distance, which is arithmetic a reader can
check by hand — and, per ``codec_floor`` above, the one metric an mp4 round trip ruins.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from wam.evaluation.dream import FREEZE_MARGIN, as_frames_255

FIDELITY_VERSION = "video_fidelity/1"

#: Arm names. ``model`` is the thing under test; the other three are the controls it has to be
#: read against and a report is not interpretable without at least ``frozen``.
MODEL_ARM = "model"
FROZEN_ARM = "frozen"
TRUTH_ARM = "truth"
OTHER_ARM = "other"
CODEC_FLOOR_ARM = "codec_floor"

#: Every metric, all on the 0-255 uint8 scale — the convention ``docs/hf_jobs.md`` already
#: quotes ("mean abs pixel diff 2.5/255") and the one :mod:`wam.evaluation.dream` reports in.
METRICS = ("mean_abs", "mse", "gradient_abs")

#: Resize kernels this module will name in a report. Restricted on purpose: the interpolation is
#: recorded next to the numbers it produced, so it has to be a fixed vocabulary rather than
#: whatever integer flag a caller happened to have.
INTERPOLATIONS = ("area", "nearest", "linear", "cubic", "lanczos4")

#: Downsampling default. INTER_AREA averages the pixels it discards; INTER_LINEAR samples them,
#: which aliases the fine texture of a 480x640 kitchen scene into whichever grid it lands on and
#: would charge the model for our resampling. Recorded in the report either way.
DEFAULT_INTERPOLATION = "area"

VERDICT_PREDICTS = "PREDICTS"
VERDICT_NO_BETTER = "NO_BETTER_THAN_FREEZING"

#: The second gate. Beating the frozen bar is necessary and not sufficient on this corpus — a
#: different demo of the same task clears it in two of the five measured windows — so a clip
#: that does not also beat retrieval is reported as such rather than left to a reader to notice.
VERDICT_BEATS_CHANCE = "BEATS_CHANCE"
VERDICT_NO_BETTER_THAN_CHANCE = "NO_BETTER_THAN_CHANCE"
VERDICT_CHANCE_NOT_MEASURED = "CHANCE_NOT_MEASURED"


class ArmScore(BaseModel):
    """One arm's distance to the real future. All quantities on the 0-255 pixel scale."""

    model_config = ConfigDict(frozen=True)

    arm: str
    mean_abs: float = Field(description="mean |arm - real| over every pixel and scored frame")
    mse: float = Field(description="mean (arm - real)^2 — squared grey levels, not normalized")
    gradient_abs: float = Field(
        description="mean |grad(arm) - grad(real)| over both spatial first differences"
    )


class Alignment(BaseModel):
    """Exactly which frames were compared with which, and how they were made comparable.

    This is not bookkeeping. Both halves of it — resampling in time and resizing in space — fail
    by producing a plausible number rather than an error, so the report carries the full index
    list and the kernel name and a reader can recompute the mapping from the parquet.
    """

    model_config = ConfigDict(frozen=True)

    generated_fps: float
    source_fps: float
    generated_frames: int = Field(ge=1, description="frames in the clip as decoded")
    lead_context_frames: int = Field(ge=0, description="leading frames excluded as replayed real")
    scored_frames: int = Field(ge=1, description="frames actually compared — the rest were not")
    source_start_frame: int = Field(ge=0, description="real frame the FIRST SCORED frame predicts")
    source_indices: tuple[int, ...] = ()
    conditioning_source_frame: int = Field(ge=0, description="the frame the frozen arm holds")
    comparison_hw: tuple[int, int]
    interpolation: str
    resized: bool = Field(description="False when both sides already share comparison_hw")
    other_scored: bool = False
    codec_floor_scored: bool = Field(
        default=False, description="False means the report does not say what perfect would score"
    )


class FidelityReport(BaseModel):
    """One clip's distance to truth, its controls, and the ratio that is comparable across runs."""

    model_config = ConfigDict(frozen=True)

    version: str = FIDELITY_VERSION
    arms: dict[str, ArmScore]
    ratio_to_frozen: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="arm metric / frozen metric — the T-36 quantity"
    )
    alignment: Alignment
    verdicts: dict[str, str] = Field(default_factory=dict)
    info: dict[str, Any] = Field(default_factory=dict)


# ---- metrics ----------------------------------------------------------------------------


def _gradient_distance_255(x: np.ndarray, y: np.ndarray) -> float:
    """:func:`gradient_distance` on arrays :func:`as_frames_255` has already normalized.

    Split out for the same reason ``dream._static_fraction_255`` is: normalizing twice is not
    idempotent by design — the second pass sees a 0-255 float and the range guard fires — so the
    internal call has to take the normalized array rather than re-derive it.
    """
    if x.shape[-2] < 2 or x.shape[-3] < 2:
        raise ValueError(f"a gradient needs at least 2x2 pixels, got {x.shape[-3:-1]}")
    horizontal = np.abs(np.diff(x, axis=-2) - np.diff(y, axis=-2)).mean()
    vertical = np.abs(np.diff(x, axis=-3) - np.diff(y, axis=-3)).mean()
    return float((horizontal + vertical) / 2.0)


def gradient_distance(clip_a: Any, clip_b: Any) -> float:
    """Mean ``|grad(a) - grad(b)|`` over the horizontal and vertical first differences (0-255).

    The metric a global exposure shift cannot move: adding a constant to every pixel of one clip
    leaves every neighbour difference untouched, so it cancels here and does not cancel in
    :func:`frame_metrics`'s ``mean_abs``. That matters because a diffusion sample decoded from a
    VAE routinely comes back a few grey levels brighter or darker than the recording, and a
    scorer that ranked backbones on exposure would be ranking them on nothing.

    It is a difference of gradients, not a difference of gradient magnitudes: an edge that moved
    is a real error and must not cancel against an edge of the same strength somewhere else.
    """
    x = as_frames_255(clip_a)
    y = as_frames_255(clip_b)
    if x.shape != y.shape:
        raise ValueError(f"clip shape mismatch: {x.shape} vs {y.shape}")
    return _gradient_distance_255(x, y)


def frame_metrics(clip_a: Any, clip_b: Any) -> dict[str, float]:
    """All of :data:`METRICS` for one pair of equally-shaped clips, on the 0-255 scale.

    Normalization is :func:`~wam.evaluation.dream.as_frames_255`, so uint8 recordings and the
    [0, 1] floats a backbone's ``decode_video`` returns both arrive here on one scale and the
    numbers stay comparable with every motion figure already in ``dream.json``.
    """
    x = as_frames_255(clip_a)
    y = as_frames_255(clip_b)
    if x.shape != y.shape:
        raise ValueError(f"clip shape mismatch: {x.shape} vs {y.shape}")
    difference = x - y
    return {
        "mean_abs": float(np.abs(difference).mean()),
        "mse": float((difference * difference).mean()),
        "gradient_abs": _gradient_distance_255(x, y),
    }


def check_truth_is_zero(scores: Mapping[str, float]) -> None:
    """Refuse a report whose ``truth`` arm is not identically zero.

    **This does not check the alignment.** The truth arm compares one array with itself, so it
    is 0 for any indices, any clock and any resize kernel — the module docstring used to claim
    otherwise and it was wrong. Subtracting an array from itself has exactly one way to come out
    nonzero: a NaN or an infinity among the source pixels, which propagates into every other arm
    and would otherwise be reported as a finite-looking distance. That failure is real (a decode
    returning a partially written frame does it) and silent, so the guard stays; it is a
    non-finite check wearing an alignment check's name, and the name is kept because ``truth``
    being 0 is what a reader of the report is entitled to assume.
    """
    nonzero = {name: value for name, value in scores.items() if value != 0.0}
    if nonzero:
        raise ValueError(
            f"the truth arm scored {nonzero} instead of 0 — an array does not differ from "
            "itself, so a non-finite pixel reached the metrics and every arm in this report "
            "is contaminated by it"
        )


# ---- alignment in time --------------------------------------------------------------------


def align_indices(
    generated_frames: int,
    source_length: int,
    *,
    generated_fps: float,
    source_fps: float,
    source_start_frame: int,
    source_offset: int = 0,
    lead_context_frames: int = 0,
) -> np.ndarray:
    """Which real frame each SCORED generated frame is a prediction of. Absolute source indices.

    **The two sides do not share a clock.** Both generate paths write 24 fps
    (``hf_job_cosmos3_probe.py`` passes ``fps=24`` to the pipeline and to ``export_to_video``)
    and GR00T-AppleToPlate is 30 fps (``meta/info.json``). Comparing frame k with frame k drifts
    by one real frame every four generated frames — 18 frames over a 72-frame prediction, which
    on a 0.6 m reach is most of the motion, and it produces a perfectly finite number. So both
    rates are required arguments with no defaults and the map is by time:

        source = source_start_frame + floor(k * source_fps / generated_fps + 0.5)

    ``k`` counts from the FIRST SCORED frame, not from the start of the clip. Placing the time
    origin at that boundary is deliberate: a video-conditioned clip replays real frames into a
    24 fps container, so their spacing in the generated timeline is not their spacing in the
    recording, and an origin placed before the boundary would inherit that distortion and carry
    it into every predicted frame.

    Nearest neighbour, ties up, no temporal interpolation. Blending two real frames would build
    a target that no camera recorded, and a model that predicted the blend would then score
    better than one that predicted either real frame correctly.

    ``source_offset`` says which absolute index the supplied span starts at, so a caller that
    decoded only the span it needs out of a 590-frame AV1 episode can still have absolute
    indices in its report. Running past either end raises: truncating would silently score a
    shorter window than the one asked for, and the ratio would then be a different measurement
    wearing the same name.
    """
    if generated_fps <= 0.0 or source_fps <= 0.0:
        raise ValueError(f"fps must be > 0, got generated {generated_fps}, source {source_fps}")
    if lead_context_frames < 0:
        raise ValueError(f"lead_context_frames must be >= 0, got {lead_context_frames}")
    if lead_context_frames >= generated_frames:
        raise ValueError(
            f"{lead_context_frames} leading context frames of a {generated_frames}-frame clip "
            "leaves nothing predicted to score"
        )
    if source_start_frame < 0:
        raise ValueError(f"source_start_frame must be >= 0, got {source_start_frame}")

    steps = np.arange(generated_frames - lead_context_frames, dtype=np.float64)
    offsets = np.floor(steps * (source_fps / generated_fps) + 0.5).astype(np.int64)
    indices = source_start_frame + offsets

    available = range(source_offset, source_offset + source_length)
    if int(indices[-1]) not in available or int(indices[0]) not in available:
        raise ValueError(
            f"the comparison window needs source frames {int(indices[0])}..{int(indices[-1])} "
            f"but only {source_offset}..{source_offset + source_length - 1} were supplied — "
            "refusing to truncate, because a shorter window is a different measurement"
        )
    return indices


# ---- alignment in space -------------------------------------------------------------------


def _cv2_interpolation(name: str) -> int:
    """Kernel name -> cv2 flag. The name is validated before cv2 is imported, so a typo fails
    the same way with or without the optional ``data`` extra installed."""
    if name not in INTERPOLATIONS:
        raise ValueError(f"unknown interpolation {name!r}, expected one of {INTERPOLATIONS}")
    import cv2

    return {
        "area": cv2.INTER_AREA,
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
    }[name]


def resize_clip(
    frames: Any, hw: tuple[int, int], *, interpolation: str = DEFAULT_INTERPOLATION
) -> np.ndarray:
    """``[F, H, W, 3]`` -> ``[F, h, w, 3]``, or the array untouched when it is already that size.

    Returning the input unchanged on a size match is what lets a report say ``resized: False``
    and mean it. That distinction is load-bearing: the real arms (``truth``, ``frozen``,
    ``other``) pass through this and the model's own clip does not, so whenever the geometries
    differ the controls carry a resampling blur that the model arm never sees. On the corpus
    this was written for they do not differ — GR00T is 480x640 and both generate paths write
    480x640 — the asymmetry is absent. Anywhere else, read the flag before reading the ratio.
    """
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"frames must be [F, H, W, 3], got {array.shape}")
    height, width = int(hw[0]), int(hw[1])
    if height < 1 or width < 1:
        raise ValueError(f"target size must be positive, got {hw}")
    if array.shape[1:3] == (height, width):
        return array

    flag = _cv2_interpolation(interpolation)
    import cv2

    # cv2 takes (width, height); the arrays here are (height, width). Swapping them silently
    # transposes the whole scene when the frame is not square, and the metrics still return.
    return np.stack([cv2.resize(frame, (width, height), interpolation=flag) for frame in array])


def frozen_control(frame: Any, num_frames: int) -> np.ndarray:
    """One conditioning frame -> ``[num_frames, H, W, 3]`` of nothing happening.

    The trivial predictor, and on this corpus a strong one: 96 % of frame pairs move under one
    grey level (T-35), so "the scene stays as it was" is right most of the time. T-36 measured a
    fine-tuned Wan dream losing to it by 39 %.
    """
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"conditioning frame must be [H, W, 3], got {array.shape}")
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    return np.repeat(array[None, ...], num_frames, axis=0)


# ---- the scorer ---------------------------------------------------------------------------


def score_generated_video(
    generated: Any,
    source_frames: Any,
    *,
    generated_fps: float,
    source_fps: float,
    source_start_frame: int,
    source_offset: int = 0,
    lead_context_frames: int = 0,
    conditioning_source_frame: int | None = None,
    other_frames: Any | None = None,
    codec_floor_frames: Any | None = None,
    interpolation: str = DEFAULT_INTERPOLATION,
    info: Mapping[str, Any] | None = None,
) -> FidelityReport:
    """Score one generated clip against the same episode's real future, plus its controls.

    ``generated`` is ``[F, H, W, 3]`` as decoded, context frames included — they are excluded
    here rather than by the caller, because a caller that trimmed them would also have to trim
    the alignment and the two trims can disagree. ``lead_context_frames`` is how many leading
    frames are real context replayed by a Video2World-style conditioning
    (``--gen-cond-frames``); scoring those would compare the recording with itself and drive the
    reported distance toward the ``truth`` arm's zero. The returned
    :class:`Alignment` states how many frames were actually scored.

    ``source_frames`` is the real episode, ``source_offset`` the absolute index of its first
    element. ``other_frames`` is a DIFFERENT episode, positioned so that ``other_frames[0]``
    plays the role ``source_start_frame`` plays here; it is resampled at the same relative
    offsets, so the chance level is measured through the identical time map and not through a
    kinder one. **How strong that bar is, is the caller's choice and not this function's**: the
    same two episodes give other/frozen 0.522 to 1.118 depending on where the caller started the
    other one (module docstring, ``other``), and ``beats_chance`` inherits whatever it is handed.
    The CLI searches offsets and passes the strongest it finds; a caller that passes one
    arbitrary offset gets a verdict against one arbitrary offset.

    ``conditioning_source_frame`` defaults to ``source_start_frame - 1``, the last real frame
    before the first predicted one. It is an argument rather than a constant because an
    image-conditioned run and a video-conditioned run anchor at different places, and guessing
    wrong moves the frozen bar without moving anything that would complain.

    ``codec_floor_frames`` is the true future put through the same encoder the generated clip
    came out of — one scored frame for each scored frame of the clip. It is what a perfect
    prediction scores, and without it the report has no way to say that a ``gradient_abs`` ratio
    of 0.45 is the floor rather than a result (see the ``codec_floor`` entry in the module
    docstring). Optional because the library takes arrays and cannot re-encode them; the CLI
    builds it.
    """
    clip = np.asarray(generated)
    if clip.ndim != 4 or clip.shape[-1] != 3:
        raise ValueError(f"generated must be [F, H, W, 3], got {clip.shape}")
    source = np.asarray(source_frames)
    if source.ndim != 4 or source.shape[-1] != 3:
        raise ValueError(f"source_frames must be [N, H, W, 3], got {source.shape}")

    indices = align_indices(
        int(clip.shape[0]),
        int(source.shape[0]),
        generated_fps=generated_fps,
        source_fps=source_fps,
        source_start_frame=source_start_frame,
        source_offset=source_offset,
        lead_context_frames=lead_context_frames,
    )
    anchor = (
        source_start_frame - 1 if conditioning_source_frame is None else conditioning_source_frame
    )
    if not source_offset <= anchor < source_offset + source.shape[0]:
        raise ValueError(
            f"conditioning frame {anchor} is outside the supplied source frames "
            f"{source_offset}..{source_offset + source.shape[0] - 1}"
        )

    model = clip[lead_context_frames:]
    comparison_hw = (int(model.shape[1]), int(model.shape[2]))
    truth = resize_clip(source[indices - source_offset], comparison_hw, interpolation=interpolation)
    frozen = frozen_control(
        resize_clip(
            source[anchor - source_offset][None, ...], comparison_hw, interpolation=interpolation
        )[0],
        int(model.shape[0]),
    )

    arms = {
        MODEL_ARM: frame_metrics(model, truth),
        FROZEN_ARM: frame_metrics(frozen, truth),
        # One array against itself. This is the scale's zero, NOT a check on the gather above —
        # it is 0 for a wrong clock and a wrong start frame too. See check_truth_is_zero.
        TRUTH_ARM: frame_metrics(truth, truth),
    }
    check_truth_is_zero(arms[TRUTH_ARM])

    if other_frames is not None:
        other = np.asarray(other_frames)
        if other.ndim != 4 or other.shape[-1] != 3:
            raise ValueError(f"other_frames must be [N, H, W, 3], got {other.shape}")
        # The SAME relative offsets, so the retrieval control goes through the identical time
        # map. A control resampled differently from the arm it calibrates measures the resampling.
        relative = indices - source_start_frame
        if int(relative[-1]) >= other.shape[0]:
            raise ValueError(
                f"the other episode has {other.shape[0]} frames from its start point but the "
                f"comparison window reaches +{int(relative[-1])} — chance level must be measured "
                "over the same span, not a shorter one"
            )
        arms[OTHER_ARM] = frame_metrics(
            resize_clip(other[relative], comparison_hw, interpolation=interpolation), truth
        )

    if codec_floor_frames is not None:
        floor = np.asarray(codec_floor_frames)
        if floor.ndim != 4 or floor.shape[-1] != 3:
            raise ValueError(f"codec_floor_frames must be [F, H, W, 3], got {floor.shape}")
        # A floor measured over a different window is not this clip's floor. The encode cost
        # depends on what is in the frames — 72 frames of a static shelf ring far less than 72
        # frames of an arm crossing the scene — so a shorter or longer run would quietly
        # understate the very contamination the arm exists to expose.
        if int(floor.shape[0]) != int(model.shape[0]):
            raise ValueError(
                f"the codec floor has {floor.shape[0]} frames but {model.shape[0]} were scored "
                "— a floor over a different window is not this clip's floor"
            )
        arms[CODEC_FLOOR_ARM] = frame_metrics(
            resize_clip(floor, comparison_hw, interpolation=interpolation), truth
        )

    flat = [metric for metric in METRICS if arms[FROZEN_ARM][metric] == 0.0]
    if flat:
        raise ValueError(
            f"the frozen control scores 0 on {flat} — the real future is indistinguishable from "
            "the conditioning frame, so there is nothing here for a prediction to be better at. "
            "This is T-36's D1 defect: pick the window by motion, not by position in the episode"
        )
    ratios = {
        name: {metric: scores[metric] / arms[FROZEN_ARM][metric] for metric in METRICS}
        for name, scores in arms.items()
        if name != FROZEN_ARM
    }

    alignment = Alignment(
        generated_fps=float(generated_fps),
        source_fps=float(source_fps),
        generated_frames=int(clip.shape[0]),
        lead_context_frames=int(lead_context_frames),
        scored_frames=int(model.shape[0]),
        source_start_frame=int(source_start_frame),
        source_indices=tuple(int(i) for i in indices),
        conditioning_source_frame=int(anchor),
        comparison_hw=comparison_hw,
        interpolation=interpolation,
        resized=(int(source.shape[1]), int(source.shape[2])) != comparison_hw,
        other_scored=other_frames is not None,
        codec_floor_scored=codec_floor_frames is not None,
    )
    # The same pre-registered margin as PR-06's freeze gate, imported rather than restated, so a
    # win here and a win there mean the same thing: 10 % closer to the truth than doing nothing.
    #
    # Both gates read mean_abs, and which metric they read is not a free choice. mse is the
    # outlier-sensitive one — a clip 7 % closer than frozen is already 13 % closer in squared
    # error, so a gate on mse passes clips the pre-registration rejects. gradient_abs is worse
    # than either: it is blind to a global exposure shift by construction, and an mp4 round trip
    # alone spends 0.460 to 0.723 of the frozen gradient bar (the codec_floor arm), so it is the
    # metric a codec could pass on its own. A clip that is the true future plus 30 grey levels
    # is twice as far from the truth as standing still (mean_abs ratio 2.134 on this module's
    # test fixture) and scores gradient_abs 0.000 — gated on the gradient it would publish
    # PREDICTS. Pinned by test_a_clip_that_is_only_a_brightness_shift_fails_both_gates.
    verdicts = {
        "beats_frozen": (
            VERDICT_PREDICTS
            if arms[MODEL_ARM]["mean_abs"] < FREEZE_MARGIN * arms[FROZEN_ARM]["mean_abs"]
            else VERDICT_NO_BETTER
        )
    }
    # The frozen bar alone certifies nothing on this corpus: measured through the CLI, another
    # demo of the same task scores 0.533 of frozen at ep0@271 and 0.593 at ep10@250, both well
    # inside the margin. Computing the retrieval control and then not reading it is how a lookup
    # gets published as a prediction.
    if OTHER_ARM in arms:
        verdicts["beats_chance"] = (
            VERDICT_BEATS_CHANCE
            if arms[MODEL_ARM]["mean_abs"] < FREEZE_MARGIN * arms[OTHER_ARM]["mean_abs"]
            else VERDICT_NO_BETTER_THAN_CHANCE
        )
    else:
        verdicts["beats_chance"] = VERDICT_CHANCE_NOT_MEASURED
    return FidelityReport(
        arms={name: ArmScore(arm=name, **scores) for name, scores in arms.items()},
        ratio_to_frozen=ratios,
        alignment=alignment,
        verdicts=verdicts,
        info=dict(info or {}),
    )


# ---- reading two reports against each other -------------------------------------------------


#: What two reports must agree on before one ratio may be read against the other. Everything
#: here changes the frozen bar the ratio divides by, so a disagreement makes the comparison a
#: comparison of windows: ``interpolation`` is in the list because the kernel moves the arms
#: too (0.354 grey levels between ``area`` and ``linear`` on this module's test fixture).
COMPARABLE_ALIGNMENT_FIELDS = (
    "source_indices",
    "comparison_hw",
    "conditioning_source_frame",
    "generated_fps",
    "source_fps",
    "interpolation",
)

#: Which recording the window was cut from. The library only ever sees arrays, so it cannot
#: derive these; the CLI writes them into ``info`` and they are checked when BOTH reports carry
#: them. Two reports on different episodes can otherwise share every alignment field above.
COMPARABLE_INFO_KEYS = ("data_dir", "episode")


def check_same_window(
    report_a: FidelityReport, report_b: FidelityReport, *, names: tuple[str, str] = ("a", "b")
) -> None:
    """Refuse two reports whose headline ratios are not about the same window.

    The ratio is a distance divided by the frozen bar, and the bar is a property of the window:
    5.597 to 25.639 ``mean_abs`` over the five windows measured on this corpus. An arm of fixed
    quality moves by 4x across them (module docstring), against a decision margin of 10 %, so
    "Wan scored 0.60 and Cosmos3 scored 0.85" is a statement about two backbones only if both
    numbers came off the same frames. Nothing about a mismatched pair looks wrong — both reports
    are internally valid and both ratios are correctly computed — which is why this is a refusal
    and not a warning.

    ``info`` is checked for :data:`COMPARABLE_INFO_KEYS` only where both reports carry the key.
    A report made from arrays has no episode in it and cannot be forced to have one, so a
    library-only pair is compared on its alignment alone; that is weaker, and it is the most
    this module can know.
    """
    differences = []
    for field in COMPARABLE_ALIGNMENT_FIELDS:
        left, right = getattr(report_a.alignment, field), getattr(report_b.alignment, field)
        if left != right:
            if field == "source_indices" and left and right:
                left = f"{left[0]}..{left[-1]} ({len(left)} frames)"
                right = f"{right[0]}..{right[-1]} ({len(right)} frames)"
            differences.append(f"{field}: {names[0]} {left}, {names[1]} {right}")
    shared = [key for key in COMPARABLE_INFO_KEYS if key in report_a.info and key in report_b.info]
    for key in shared:
        if report_a.info[key] != report_b.info[key]:
            differences.append(
                f"info.{key}: {names[0]} {report_a.info[key]!r}, {names[1]} {report_b.info[key]!r}"
            )
    if differences:
        raise ValueError(
            "these two reports scored different windows, so their ratios are not comparable — "
            + "; ".join(differences)
        )
