#!/usr/bin/env python3
"""Look at what ``estimators.apple_sam2`` actually segments — PR-08 §4, blockers 1 and 2.

    python scripts/audit_apple_masks.py \\
        --corpus /valhalla/.../data/pr08-apple-640x480-h264-lossless \\
        --census runs/t040-identity-prompt/calibration-2/probe_census.json \\
        --out    /valhalla/.../runs/pr08-mask-audit

WHAT THIS IS FOR, IN THE BLOCKER'S OWN WORDS
--------------------------------------------
``scripts/estimators/apple_sam2.py`` sets ``GATE_QUALIFIED = False`` and blocker 1 begins
**"NOBODY HAS LOOKED AT A MASK."** Coverage 1.0 says a box was returned on every frame, not that it
was the *apple's* box, and this adapter's whole failure mode is a plausible mask on the wrong object
— the plate, the hand, the whole tabletop — which yields a centroid, a displacement and a p95 that
all look like measurements. Blocker 1 names its own discharge condition: *a human looking at a
sample of overlaid masks spanning the corpus (occluded frames, apple-out-of-frame frames, and the
grasp)*. Blocker 2 adds the operating point: ``0.15``/``0.25`` with one ``(0.10, 0.10)`` retry is
unmeasured on this corpus, and the retry *"buys detections by accepting weak ones, which on an
occluded frame can replace an honest all-False mask with a confident box on the wrong object —
inflating coverage while degrading the mask, i.e. hiding itself in the one number the harness gates
on"*; it is discharged by the same overlays **plus** the detection-score distribution and the retry
counts.

This script produces that evidence. **It does not produce the discharge.**

WHAT THIS SCRIPT WILL NOT DO
----------------------------
**It will not flip ``GATE_QUALIFIED`` and it will not edit ``GATE_QUALIFICATION_BLOCKERS``.** It
imports the adapter, drives it unmodified, reads its counters and copies its blocker tuple into the
artifact verbatim. Producing evidence and discharging a blocker are two different acts and only the
second one is a judgement; a script that could do both would be a script that discharges a blocker
by running. ``tests/test_audit_apple_masks.py`` asserts this file contains no assignment to either
name.

**It will not call a model's reading of the overlays "a human looking".** The artifact carries a
``human_review`` block whose ``looked_at`` is ``false`` until a person fills
``OBSERVATIONS.template.json`` in, and the template's own header says what
``runs/t040-identity-prompt/calibration-2/probe_observations.json`` says about its seed pass: a
model confirming an instrument built by a model of the same family is a **correlated observer**, and
what it writes down is a finding, not the check. The same rule, in the same words, because it is the
same weakness.

**It will not sample uniformly and call it spanning.** See THE SAMPLING RULE below. A uniform sample
of 402 episodes × ~427 frames would contain, in expectation, *zero* of the frames blocker 1 names:
``probe-scan`` measured the strict apple mask on all 154 447 frames of the 362 non-measured episodes
and found **48** frames below 1 200 px of visible apple, **24** of them eligible as occlusions, and
all of them in **one episode**. Hard cases in this corpus are rare and concentrated, so they are
sought deliberately and the bias is stated rather than averaged away.

**It will not decide anything from the flags.** Automatic triage cannot tell an apple from a plate;
it can only say "this mask is the size of a plate", "this centroid moved 90 px in one frame", "the
retry bought this detection", "this mask sits where the plate is". Every flag is a REQUEST FOR
ATTENTION addressed to the reviewer and the artifact says so in the field beside them.

**It will not measure the corpus rate of anything.** The retry rate, the no-detection rate and the
score distribution recorded here are over a sample that deliberately over-weights the hard frames,
so they are **biased upwards** as estimates of the corpus and are labelled that way in the artifact.
Blocker 2 asks for those numbers *from a full pass*; the full pass is the GEOM_TOL array run, which
until 2026-08-22 recorded ``apple_sam2.stats()`` nowhere, so the full-pass half of blocker 2 had
nowhere to land. It has one now: both harnesses write an ``estimator_stats`` block (per-run counters,
snapshot-and-differenced, and the pooled detection-score distribution), and the shards carry their
raw scores per episode so the merge pools them exactly. The place existing is not the evidence
existing and neither is a discharge — see ``full_pass_gap`` in the artifact, which now says which
run has to be done rather than which code has to be written.

WHAT A RE-RUN SHOWS AFTER PR-08 V6
----------------------------------
The 2026-08-22 run of this script (job 189637, 382 frames, 24 episodes) is what produced the finding
PR-08 V6 registers: **twelve frames carried a confident, well-formed mask of the PLATE**, all in
``episode_000094``, every one at IoU 0.0000 against the colour heuristic while every correct mask
scored >= 0.7492. The adapter now refuses such a mask — ``segment()`` returns all-False and counts
the frame — so a re-run brings those twelve back as **refusals** rather than as masks, carrying the
``mask_refused`` flag and a ``mask_validity_iou`` of 0.0 in ``frames[*]``, with the run's totals in
``mask_validity_filter``.

**The flagging is not weakened by this and must not be.** ``mask_refused`` is ADDITIVE: every other
flag still applies to a refused frame, and ``disagrees_with_warm_apple`` in particular — the fruit
is plainly visible and the returned mask is nowhere near it — is exactly what a WRONG refusal would
look like from here. A frame that is suspicious in any of the old ways and survives the filter is
still flagged in all of them. An all-False mask with no recorded reason is now itself a
``recorder_inconsistent``: there are exactly three reasons (no detection, empty mask, validity
refusal) and a fourth would be a step silently dropped from every coverage number downstream.

THE SAMPLING RULE
-----------------
Deterministic, no RNG, and recorded in the artifact so the sample can be rebuilt and its bias
argued with.

*Episodes.* Every episode the ``probe-scan`` census names as containing a frame below its census
threshold is **forced in** — those are the corpus's occlusion events and there are almost none of
them. The remainder of the budget is filled by walking the sorted episode enumeration at an even
stride, endpoints included, so the sample spans the corpus in its own order.

*Frames, per episode.* Six strata, in priority order, computed from a fresh per-frame scan of that
episode with the same strict warm-apple discriminator ``build_identity_calibration.apple_mask``
uses:

``census``          the census's own eligible frames for this episode — measured occlusions.
``occluded``        frames whose warm apple is below the census threshold (1 200 px).
``min_visibility``  the episode's own least-visible frames, whether or not they clear a threshold.
``border``          frames whose warm apple touches the image border — the closest this corpus
                    comes to "apple out of frame", and see WHAT THE CORPUS DOES NOT CONTAIN.
``grasp``           a window around lift-off: the first frame at which the apple's own centroid has
                    left its resting position and stays away. The grasp is not annotated in this
                    corpus, so this is a proxy and is named as one.
``spanning``        fixed quantiles of the episode timeline (0, ⅓, ⅔, 1), the ordinary frames,
                    without which this would be an audit of the hard cases only.

*Neighbours.* Every anchor outside ``spanning`` also pulls in the frame after it, so that
"the centroid jumped implausibly between adjacent frames" is a measurable statement and not an
inference across a gap of 200 frames.

WHAT THE CORPUS DOES NOT CONTAIN, WHICH IS ALSO A RESULT
--------------------------------------------------------
Blocker 1 asks for "apple-out-of-frame frames". The census's per-episode ``min_apple_warm_px`` says
the apple is essentially always visible: 362 episodes, and the strict mask drops below 1 200 px on
48 frames of one of them. If the sample contains no out-of-frame frame, the artifact says so under
``strata_not_found`` instead of quietly omitting the stratum, because "we looked and the corpus has
none" and "we did not look" are different claims that produce the same empty list.

EXIT STATUS
-----------
0   the evidence was written. **This is not a discharge and the artifact says so in three places.**
2   fatal: a refusal — the adapter would not load, the corpus would not decode, the census is
    missing and was not explicitly waived. Nothing usable was written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import build_identity_calibration as bic  # noqa: E402
import measure_geom_tol as mgt  # noqa: E402

SCHEMA = "wam.pr08_mask_audit/1"
WRITEUP = "docs/preregistration/PR-08-photoreal-augmentation.md"

#: The blocker this evidence is addressed to. Spelled out so the artifact can be read on its own.
ADDRESSES = (
    "scripts/estimators/apple_sam2.py GATE_QUALIFICATION_BLOCKERS[0] (NOBODY HAS LOOKED AT A MASK) "
    "and [1] (the operating point and the retry are unmeasured on AppleToPlate). NOT blocker [2] "
    "(per-frame segmentation vs upstream's SAM2VideoPredictor propagation), which needs an Isaac "
    "capture measured both ways and is untouched by anything here."
)

# -- the strata ------------------------------------------------------------------------------------
#
# Priority order. When an episode yields more anchors than --max-anchors-per-episode, the tail of
# THIS list is what gets dropped, so the hard cases can never be crowded out by the ordinary ones.

S_CENSUS = "census"
S_OCCLUDED = "occluded"
S_MIN_VIS = "min_visibility"
S_BORDER = "border"
S_GRASP = "grasp"
S_SPANNING = "spanning"

STRATUM_ORDER: tuple[str, ...] = (S_CENSUS, S_OCCLUDED, S_MIN_VIS, S_BORDER, S_GRASP, S_SPANNING)

#: Strata whose successor frame is pulled in too. ``spanning`` is excluded on purpose: an adjacent
#: pair of ordinary frames costs two segmenter forwards to demonstrate that a stationary apple does
#: not move, which is the one thing nobody doubts.
PAIRED_STRATA: frozenset[str] = frozenset({S_CENSUS, S_OCCLUDED, S_MIN_VIS, S_BORDER, S_GRASP})

STRATUM_MEANING: dict[str, str] = {
    S_CENSUS: (
        "a frame the probe-scan census (build_identity_calibration.py probe-scan) measured as an "
        "occlusion: the strict warm-apple mask has all but vanished AND at least half the ring "
        "around what is left is foreground, i.e. what hides the fruit is visibly the robot."
    ),
    S_OCCLUDED: (
        f"the strict warm-apple mask is below the census threshold of "
        f"{bic.NATURAL_PROBE_CENSUS_PX} px in this episode's own fresh scan."
    ),
    S_MIN_VIS: (
        "this episode's least-visible frames by warm-apple area, whether or not they clear any "
        "threshold. Included because an episode with no occlusion still has a worst frame, and the "
        "adapter's behaviour there is the question."
    ),
    S_BORDER: (
        "the warm-apple mask touches the image border — the closest this corpus comes to the "
        "'apple out of frame' case blocker 1 names. If a run reports none, that is a measurement "
        "about the corpus, recorded under strata_not_found."
    ),
    S_GRASP: (
        "a window around LIFT-OFF, defined as the first frame where the warm-apple centroid has "
        "left its resting position by at least the lift threshold and stays away for the "
        "persistence window. The grasp instant is not annotated in this corpus; this is a proxy "
        "computed from the pixels and it is named as one."
    ),
    S_SPANNING: (
        "fixed quantiles of the episode timeline (0, 1/3, 2/3, 1). The ordinary frames. Without "
        "them this would be an audit of the hard cases only, and 'the masks look fine' would be a "
        "claim about six frames of one episode."
    ),
}

# -- the sampling rule, as the artifact carries it -------------------------------------------------

SAMPLING_RULE = (
    "DETERMINISTIC, NO RNG. Episodes: every episode the probe-scan census names as containing a "
    "frame below its census threshold is forced in; the rest of the episode budget is filled by "
    "walking the sorted episode enumeration at an even stride with both endpoints included. "
    "Frames, per episode: six strata computed from a fresh per-frame scan of that episode with the "
    "strict warm-apple discriminator (r>90, r-b>50, saturation>0.35 — build_identity_calibration."
    "apple_mask's own predicate) — census, occluded, min_visibility, border, grasp, spanning — "
    "taken in that priority order up to --max-anchors-per-episode, then deduplicated by frame "
    "index. Every anchor outside `spanning` additionally pulls in the frame after it so that "
    "adjacent-frame centroid displacement is measurable."
)

SAMPLING_BIAS = (
    "THIS SAMPLE IS NOT THE CORPUS AND ITS RATES ARE NOT CORPUS RATES. It deliberately "
    "over-weights the frames blocker 1 names — occlusions, the least-visible frame of every "
    "episode, border contact, and the grasp — because a uniform sample of this corpus contains "
    "essentially none of them (probe-scan: 48 frames below 1200 px of visible apple in 154447 "
    "frames, 24 eligible, all in one episode). Consequences, stated rather than left to be "
    "inferred: the no-detection rate, the retry rate and the low end of the detection-score "
    "distribution measured here are BIASED UPWARDS as estimates of the corpus, and the mask-area "
    "distribution is biased towards small masks. The corpus-wide versions of those numbers can "
    "only come from a full pass. Conversely, a mask defect that appears on ORDINARY frames would "
    "show up in the `spanning` stratum, which is drawn without regard to difficulty."
)

NOT_A_DISCHARGE = (
    "THIS ARTIFACT DOES NOT DISCHARGE ANY BLOCKER. It is the evidence blockers 1 and 2 ask for, "
    "produced so that a person can look at it. GATE_QUALIFIED is read from the adapter and copied "
    "here; nothing in this script writes it. Discharging a blocker is a reviewable edit to "
    "GATE_QUALIFICATION_BLOCKERS made by a person who has looked, and it moves the retired wording "
    "into GATE_QUALIFICATION_DISCHARGED with the evidence rather than deleting it."
)

CORRELATED_OBSERVER = (
    "If the `observed` fields below were written by a model rather than a person, say so in "
    "`established_by` and leave `looked_at` alone. A model looking at masks produced by a pipeline "
    "another model wired up is a CORRELATED OBSERVER: it is capable of reproducing the same "
    "misreading on both sides, and blocker 1 asks for a human. The same weakness, in the same "
    "words, as runs/t040-identity-prompt/calibration-2/probe_observations.json's seed pass."
)

# -- the thresholds this script chooses, all of them arbitrary and all of them recorded -------------
#
# Every number below is a TRIAGE threshold. None of them enters a gate, none of them is compared
# against a pre-registration, and moving one changes which frames get a coloured label in a contact
# sheet — never what the segmenter did. They are named, defaulted, exposed on the CLI and written
# into the artifact so a reviewer can see the sieve they are looking through.

#: Lift-off: how far the warm-apple centroid must leave its resting position, and for how many
#: consecutive frames, before that instant counts as the grasp proxy.
DEFAULT_LIFT_PX = 12.0
DEFAULT_LIFT_PERSISTENCE = 3
#: Frame offsets around lift-off. Negative reaches into the approach, where the hand is closing on
#: the fruit and the box is at its most ambiguous.
DEFAULT_GRASP_OFFSETS: tuple[int, ...] = (-16, -8, 0, 8)

#: Quantiles of the episode timeline for the ordinary frames.
DEFAULT_SPAN_QUANTILES: tuple[float, ...] = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)

#: A mask this many times the sample's own median warm-apple area is not an apple. The plate is
#: roughly 10x the fruit in this scene and the tabletop is the whole frame, so the band is wide on
#: purpose: it is meant to catch the plate and the table, not to adjudicate a fat mask.
DEFAULT_AREA_BAND_LOW = 0.20
DEFAULT_AREA_BAND_HIGH = 6.0

#: Independently of the band: a mask covering this fraction of the frame is the tabletop.
DEFAULT_FRAME_FRACTION_CEILING = 0.12

#: Adjacent-frame centroid displacement above which a reviewer should look. At 30 fps a tabletop
#: apple carried by an arm moves a few pixels per frame; 25 px in one frame is a different object.
DEFAULT_CENTROID_JUMP_PX = 25.0

#: The mask sits where the plate is, and is too big to be the fruit sitting ON the plate.
DEFAULT_PLATE_OVERLAP_FRACTION = 0.60

#: The mask and the colour heuristic disagree while the fruit is plainly visible. This is the
#: sharpest automatic tell for "the box was on the wrong object" — and it is still not a verdict:
#: the colour heuristic is a heuristic and it fails on shadowed fruit.
DEFAULT_MIN_WARM_IOU = 0.20
DEFAULT_WARM_VISIBLE_PX = 1500

#: Detection scores below this are the weak ones the retry exists to buy. 0.25 is the adapter's own
#: TEXT_THRESHOLD, reused here so the label means something a reader can look up.
DEFAULT_LOW_SCORE = 0.25

#: measure_geom_tol's own default. Reused so the centroid this audit reports is the centroid GEOM_TOL
#: would have reported for the same mask.
DEFAULT_MIN_AREA_PX = 40

FLAG_MEANING: dict[str, str] = {
    "no_detection": "the detector found no box at either threshold; the adapter returned all-False.",
    "empty_mask": "a box was found and SAM 2 returned nothing inside it.",
    "retry_fired": "the first pass found no box and upstream's single (0.10, 0.10) retry ran. "
                   "Blocker 2's named hazard lives exactly here.",
    "retry_recovered": "the retry produced a box. This frame's contribution to coverage was bought "
                       "at the lower threshold and nothing but a reviewer's eye checks it.",
    "low_score": "the winning box scored below the low-score threshold.",
    "mask_area_above_band": "the mask is far larger than this sample's median warm-apple area.",
    "mask_area_below_band": "the mask is far smaller than this sample's median warm-apple area.",
    "mask_covers_frame": "the mask covers more of the frame than any apple could.",
    "centroid_jump": "the mask centroid moved further between this frame and the adjacent one than "
                     "a carried apple can move in one frame.",
    "plate_overlap": "most of the mask lies inside the region the plate occupies, and the mask is "
                     "too big to be the fruit resting on it.",
    "disagrees_with_warm_apple": "the fruit is plainly visible to the colour heuristic and the mask "
                                 "is somewhere else.",
    "mask_refused": "PR-08 V6's mask-validity filter refused this frame: the adapter drew a "
                    "non-empty mask, found it contained essentially none of the object, and "
                    "returned all-False. This flag is not a defect — it is the fix working, and it "
                    "is here so a reviewer can see WHICH frames it fired on rather than a count. "
                    "The mask that was refused is not shown, because it was not returned; the "
                    "detector box that produced it is.",
    "mask_refused_no_reference": "the same refusal, on a frame where the colour heuristic found no "
                                 "fruit AT ALL. Nothing here can confirm or deny the mask, so this "
                                 "is the sub-case that removes a HARD frame from the measured "
                                 "population rather than a wrong one — PR-08 V6's threat to "
                                 "validity, made countable.",
    "recorder_inconsistent": "the post-processing recorder and the adapter's own counters disagree "
                             "about what happened on this frame. Trust the counters and read this "
                             "as a defect in the audit, not in the adapter.",
}

FLAGS_ARE_TRIAGE = (
    "FLAGGING IS TRIAGE, NOT A VERDICT. Every flag below is a rule of thumb chosen by this script "
    "and recorded in `triage_thresholds`; not one of them can tell an apple from a plate. A flagged "
    "frame is a request for a reviewer's attention and an unflagged frame is not a certificate — "
    "the failure this whole exercise exists for, a confident box on the wrong object, produces a "
    "mask of plausible size in a plausible place and would clear every threshold here."
)


class AuditError(RuntimeError):
    """A refusal. Nothing usable was written."""


# -- the pixels ------------------------------------------------------------------------------------


def warm_apple_mask(rgb: np.ndarray) -> np.ndarray:
    """The strict warm-and-saturated apple mask, as ``build_identity_calibration.apple_mask`` reads it.

    Reused rather than invented: this is the discriminator ``probe-scan`` measured all 154 447
    frames with, so a frame this script calls "occluded" is occluded by the same rule the census
    used. It is NOT ground truth and is never treated as such — it is a second, non-learned opinion
    about where the fruit is, which is exactly what makes a disagreement with SAM 2 interesting.

    Unlike ``apple_mask`` this returns the raw predicate rather than one grown connected component,
    and it does not raise on a frame with no fruit in it: an occluded frame is the case of interest
    here, not an error.
    """
    arr = np.asarray(rgb)
    r = arr[:, :, 0].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    return (r > 90) & ((r - b) > 50) & (bic.saturation(arr) > 0.35)


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """(x, y) of a boolean mask, or None when it is empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def touches_border(mask: np.ndarray, margin: int = 3) -> bool:
    """Does the mask reach within ``margin`` px of any edge? The census's own test."""
    if not mask.any():
        return False
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return bool(
        xs.min() < margin or xs.max() > w - 1 - margin
        or ys.min() < margin or ys.max() > h - 1 - margin
    )


def _erode(mask: np.ndarray) -> np.ndarray:
    """4-neighbour erosion, no scipy — the same shape as ``bic._dilate``, inverted."""
    out = mask.copy()
    out[1:, :] &= mask[:-1, :]
    out[:-1, :] &= mask[1:, :]
    out[:, 1:] &= mask[:, :-1]
    out[:, :-1] &= mask[:, 1:]
    return out


def boundary(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """The outline of ``mask``, ``thickness`` px thick and drawn INWARD.

    Inward so the outline never claims territory the mask does not have; thick so it survives the
    2x downsample the contact sheets do.
    """
    if not mask.any():
        return np.zeros_like(mask)
    inner = mask
    for _ in range(max(int(thickness), 1)):
        inner = _erode(inner)
    return mask & ~inner


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boolean masks; 0.0 when both are empty."""
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return float(inter) / float(union) if union else 0.0


# -- one episode, scanned ---------------------------------------------------------------------------


@dataclass
class EpisodeScan:
    """Per-frame warm-apple statistics for one episode, and nothing that needs a GPU."""

    key: str
    n_frames: int
    size_wh: tuple[int, int]
    warm_px: np.ndarray            # (n,) int
    centroid_x: np.ndarray         # (n,) float, nan where the warm mask is empty
    centroid_y: np.ndarray
    border: np.ndarray             # (n,) bool
    median_warm_px: float
    lift_index: int | None
    lift_note: str


def scan_episode(
    frames: Iterable[np.ndarray],
    key: str,
    *,
    lift_px: float = DEFAULT_LIFT_PX,
    lift_persistence: int = DEFAULT_LIFT_PERSISTENCE,
    rest_fraction: float = 0.15,
) -> EpisodeScan:
    """One decode pass over an episode -> the statistics the sampling rule needs.

    ``frames`` yields BGR uint8, which is what every decoder in ``measure_geom_tol.DECODERS``
    produces; the flip to RGB happens here, once, for the same reason it happens once in
    ``measure_geom_tol.sam2_mask_via``: a colour-order mistake in a warm-pixel discriminator does
    not crash, it silently measures the blue channel and reports that the apple is never visible.
    """
    warm_px: list[int] = []
    cxs: list[float] = []
    cys: list[float] = []
    border: list[bool] = []
    size: tuple[int, int] | None = None
    for frame in frames:
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise AuditError(f"{key}: decoded a frame of shape {arr.shape}; this path wants BGR.")
        h, w = arr.shape[:2]
        if size is None:
            size = (int(w), int(h))
        elif size != (int(w), int(h)):
            raise AuditError(f"{key}: frame geometry changes mid-clip ({size} -> {(w, h)}).")
        warm = warm_apple_mask(arr[:, :, ::-1])
        warm_px.append(int(warm.sum()))
        c = mask_centroid(warm)
        cxs.append(c[0] if c else float("nan"))
        cys.append(c[1] if c else float("nan"))
        border.append(touches_border(warm))
    if size is None:
        raise AuditError(
            f"{key}: opened and decoded no frames. This is the failure "
            "scripts/verify_clip_decode.py exists for — the container parses and the codec does "
            "not — and it is why --decoder auto probes before choosing."
        )
    px = np.asarray(warm_px, dtype=np.int64)
    cx = np.asarray(cxs, dtype=np.float64)
    cy = np.asarray(cys, dtype=np.float64)
    lift, note = _lift_index(cx, cy, lift_px=lift_px, persistence=lift_persistence,
                            rest_fraction=rest_fraction)
    return EpisodeScan(
        key=key,
        n_frames=int(px.size),
        size_wh=size,
        warm_px=px,
        centroid_x=cx,
        centroid_y=cy,
        border=np.asarray(border, dtype=bool),
        median_warm_px=float(np.median(px)),
        lift_index=lift,
        lift_note=note,
    )


def _lift_index(
    cx: np.ndarray,
    cy: np.ndarray,
    *,
    lift_px: float,
    persistence: int,
    rest_fraction: float,
) -> tuple[int | None, str]:
    """First frame at which the apple has left its resting place and stays away.

    The apple sits still on the cloth until the hand closes on it and then travels to the plate, so
    the onset of its own motion brackets the grasp from the pixels alone. This is a PROXY: the
    contact instant is a few frames earlier than the first frame of visible travel, which is why
    the grasp window reaches backwards as well as forwards.
    """
    n = int(cx.size)
    if n < persistence + 2:
        return None, "episode too short for a lift test"
    head = max(int(round(n * rest_fraction)), 1)
    rest_x = np.nanmedian(cx[:head])
    rest_y = np.nanmedian(cy[:head])
    if not np.isfinite(rest_x) or not np.isfinite(rest_y):
        return None, "no warm apple in the opening frames, so there is no resting position to leave"
    d = np.sqrt((cx - rest_x) ** 2 + (cy - rest_y) ** 2)
    away = np.nan_to_num(d, nan=0.0) >= lift_px
    run = 0
    for i in range(n):
        run = run + 1 if away[i] else 0
        if run >= persistence:
            return i - persistence + 1, (
                f"first frame at which the warm-apple centroid is >= {lift_px:g} px from its "
                f"resting position ({rest_x:.1f}, {rest_y:.1f}) for {persistence} consecutive frames"
            )
    return None, (
        f"the warm-apple centroid never left its resting position by {lift_px:g} px for "
        f"{persistence} consecutive frames"
    )


# -- the sampling rule ------------------------------------------------------------------------------


def select_episodes(
    keys: Sequence[str],
    forced: Iterable[str],
    budget: int,
) -> tuple[list[str], dict[str, Any]]:
    """Which episodes to audit. Deterministic, forced-first, then an even stride over the rest.

    A stride and not a random draw: this sample is meant to be rebuildable from the rule alone six
    months from now, and "seed 40007" is a rule only for whoever still has the interpreter that
    produced it.
    """
    ordered = list(keys)
    if not ordered:
        raise AuditError("no episodes to select from.")
    forced_present = [k for k in ordered if k in set(forced)]
    if budget <= 0 or budget >= len(ordered):
        chosen = list(ordered)
        rule = "budget covers the corpus; every episode is audited"
        return chosen, {"forced": forced_present, "stride_picked": chosen, "rule": rule}
    remaining = [k for k in ordered if k not in set(forced_present)]
    want = max(budget - len(forced_present), 0)
    stride_picked: list[str] = []
    if want and remaining:
        if want == 1:
            idx = [0]
        else:
            idx = [int(round(i * (len(remaining) - 1) / (want - 1))) for i in range(want)]
        seen: set[int] = set()
        for i in idx:
            if i not in seen:
                seen.add(i)
                stride_picked.append(remaining[i])
    chosen = sorted(set(forced_present) | set(stride_picked), key=ordered.index)
    return chosen, {
        "forced": forced_present,
        "stride_picked": stride_picked,
        "rule": (
            f"{len(forced_present)} episode(s) forced in by the census (they contain measured "
            f"occlusions); {len(stride_picked)} drawn at an even stride over the remaining "
            f"{len(remaining)} sorted keys, endpoints included."
        ),
    }


@dataclass
class Anchor:
    """One frame this audit will segment."""

    episode: str
    frame_index: int
    stratum: str
    role: str = "anchor"
    pair_of: int | None = None
    why: str = ""


def _spread(indices: Sequence[int], k: int) -> list[int]:
    """``k`` of ``indices``, spread over the run rather than taken from its head.

    The eligible frames of one occlusion event are consecutive, so the k smallest apples are k
    copies of one instant — the same trap ``build_identity_calibration.farthest_point_pick`` exists
    to avoid, in the one dimension available here.
    """
    order = list(indices)
    if k <= 0 or not order:
        return []
    if len(order) <= k:
        return order
    if k == 1:
        return [order[len(order) // 2]]
    picks = [order[int(round(i * (len(order) - 1) / (k - 1)))] for i in range(k)]
    out: list[int] = []
    for p in picks:
        if p not in out:
            out.append(p)
    return out


def select_frames(
    scan: EpisodeScan,
    *,
    census_frames: Sequence[int] = (),
    quotas: dict[str, int] | None = None,
    max_anchors: int = 12,
    grasp_offsets: Sequence[int] = DEFAULT_GRASP_OFFSETS,
    span_quantiles: Sequence[float] = DEFAULT_SPAN_QUANTILES,
    census_px: int | None = None,
    neighbour_offset: int = 1,
) -> tuple[list[Anchor], dict[str, Any]]:
    """The per-episode half of the sampling rule. Pure: no decoding, no models, no RNG."""
    q = {S_CENSUS: 4, S_OCCLUDED: 3, S_MIN_VIS: 2, S_BORDER: 2,
         S_GRASP: len(grasp_offsets), S_SPANNING: len(span_quantiles)}
    q.update(quotas or {})
    threshold = bic.NATURAL_PROBE_CENSUS_PX if census_px is None else census_px
    n = scan.n_frames
    picks: dict[str, list[int]] = {}
    notes: dict[str, str] = {}

    picks[S_CENSUS] = _spread([i for i in sorted(set(census_frames)) if 0 <= i < n], q[S_CENSUS])

    below = [int(i) for i in np.nonzero(scan.warm_px < threshold)[0]]
    picks[S_OCCLUDED] = _spread([i for i in below if i not in picks[S_CENSUS]], q[S_OCCLUDED])
    notes[S_OCCLUDED] = f"{len(below)} frame(s) below {threshold} px of visible apple in this episode"

    order = list(np.argsort(scan.warm_px, kind="stable"))
    min_vis = [int(i) for i in order[: max(q[S_MIN_VIS] * 4, q[S_MIN_VIS])]]
    picks[S_MIN_VIS] = _spread(min_vis, q[S_MIN_VIS])

    border_idx = [int(i) for i in np.nonzero(scan.border)[0]]
    picks[S_BORDER] = _spread(border_idx, q[S_BORDER])
    notes[S_BORDER] = f"{len(border_idx)} frame(s) whose warm apple touches the border"

    if scan.lift_index is None:
        picks[S_GRASP] = []
        notes[S_GRASP] = f"no lift-off found: {scan.lift_note}"
    else:
        picks[S_GRASP] = []
        for off in grasp_offsets:
            i = int(scan.lift_index) + int(off)
            if 0 <= i < n and i not in picks[S_GRASP]:
                picks[S_GRASP].append(i)
        notes[S_GRASP] = f"lift-off at frame {scan.lift_index}: {scan.lift_note}"

    picks[S_SPANNING] = []
    for frac in span_quantiles:
        i = int(round(float(frac) * (n - 1)))
        if 0 <= i < n and i not in picks[S_SPANNING]:
            picks[S_SPANNING].append(i)

    anchors: list[Anchor] = []
    taken: set[int] = set()
    # A stratum the BUDGET cut and a stratum the CORPUS does not contain look identical in the
    # output and are opposite findings, so the first is recorded as it happens.
    dropped_by_budget: list[str] = []
    for stratum in STRATUM_ORDER:
        for i in picks.get(stratum, []):
            if len(anchors) >= max_anchors:
                if stratum not in dropped_by_budget:
                    dropped_by_budget.append(stratum)
                break
            if i in taken:
                continue
            taken.add(i)
            anchors.append(Anchor(
                episode=scan.key, frame_index=int(i), stratum=stratum,
                why=f"{stratum}: warm apple {int(scan.warm_px[i])} px "
                    f"({scan.warm_px[i] / max(scan.median_warm_px, 1.0):.2f} of episode median)",
            ))

    if neighbour_offset:
        for a in list(anchors):
            if a.stratum not in PAIRED_STRATA:
                continue
            j = a.frame_index + int(neighbour_offset)
            if 0 <= j < n and j not in taken:
                taken.add(j)
                anchors.append(Anchor(
                    episode=scan.key, frame_index=j, stratum=a.stratum, role="neighbour",
                    pair_of=a.frame_index,
                    why=f"neighbour of frame {a.frame_index}, so adjacent-frame centroid "
                        "displacement is a measurement rather than an inference across a gap",
                ))

    anchors.sort(key=lambda a: a.frame_index)
    meta = {
        "picked_per_stratum": {k: v for k, v in picks.items()},
        "dropped_by_budget": dropped_by_budget,
        "notes": notes,
        "quotas": q,
        "max_anchors_per_episode": max_anchors,
        "lift_index": scan.lift_index,
        "lift_note": scan.lift_note,
        "median_warm_px": scan.median_warm_px,
        "min_warm_px": int(scan.warm_px.min()) if scan.n_frames else None,
        "n_frames": n,
    }
    return anchors, meta


# -- driving the adapter, without modifying it -------------------------------------------------------


COUNTER_NAMES = (
    "SEGMENT_CALLS",
    "NO_DETECTION_FRAMES",
    "EMPTY_MASK_FRAMES",
    "RETRY_FRAMES",
    "RETRY_RECOVERED_FRAMES",
    # PR-08 V6's mask-validity filter. Read by NAME and defaulted to 0 by `read_counters`, so this
    # script still runs against an adapter that predates the filter — the artifact then records
    # zero refusals, which for such an adapter is the truth.
    "MASK_REFUSED_FRAMES",
    "MASK_REFUSED_NO_REFERENCE_FRAMES",
)

#: The adapter attribute holding the per-frame validity IoU, in call order. Read the same way
#: ``measure_geom_tol.ADAPTER_SCORES_ATTR`` is: an optional declaration, absent without consequence
#: beyond being recorded as absent. What it buys is that the refusal is PER FRAME evidence — a
#: future audit can show the filter fired on exactly the frames a person flagged — rather than a
#: run-level tally that has to be taken on trust.
VALIDITY_IOU_ATTR = "MASK_VALIDITY_IOU"


def _to_numpy(value: Any) -> np.ndarray:
    """Tensor or array -> ndarray, without importing torch to find out."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float64)


class RecordingProcessor:
    """A see-through wrapper around GroundingDINO's processor that records its post-processing.

    WHY A WRAPPER AND NOT A SECOND DETECTION PASS. The numbers blocker 2 asks for — the detection
    score of the box that won, and whether the ``(0.10, 0.10)`` retry fired — are computed inside
    ``apple_sam2._best_box`` and thrown away. The two ways to get them are to re-run the detector
    here with the same thresholds, which is a second implementation of upstream's rule and can
    drift from the one under audit, or to WATCH the one under audit. This watches. Every attribute
    it does not define is delegated, so the adapter drives its own processor; the only interception
    is ``post_process_grounded_object_detection``, which is a pure function of the model outputs and
    the two thresholds, so observing it changes nothing.

    One call recorded per frame means the first pass found a box. Two means the retry ran. That is
    read off the adapter's own control flow rather than inferred, and it is cross-checked against
    the adapter's counters on every frame; a disagreement is flagged as a defect in THIS file.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner(*args, **kwargs)

    def post_process_grounded_object_detection(self, *args: Any, **kwargs: Any) -> Any:
        out = self._inner.post_process_grounded_object_detection(*args, **kwargs)
        record: dict[str, Any] = {
            "threshold": kwargs.get("threshold"),
            "text_threshold": kwargs.get("text_threshold"),
        }
        try:
            res = out[0]
            scores = _to_numpy(res["scores"]).reshape(-1)
            boxes = _to_numpy(res["boxes"]).reshape(-1, 4)
            record["scores"] = [round(float(s), 6) for s in scores.tolist()]
            record["boxes"] = [[round(float(v), 2) for v in row] for row in boxes.tolist()]
        except Exception as exc:  # noqa: BLE001 - an unreadable result is a fact, not a crash
            record["error"] = f"{type(exc).__name__}: {exc}"
        self.calls.append(record)
        return out


def attach_recorder(module: Any) -> RecordingProcessor | None:
    """Load the detector and wrap its processor. None when the adapter has no such seam.

    Loading here rather than on the first frame is deliberate and matches the adapter's own rule:
    a machine missing a checkpoint should refuse before any frame is decoded, not on frame 37.
    """
    detector = getattr(module, "_detector", None)
    if not callable(detector):
        return None
    processor, model = detector()
    if isinstance(processor, RecordingProcessor):
        processor.calls.clear()
        return processor
    recorder = RecordingProcessor(processor)
    module._DETECTOR = (recorder, model)
    return recorder


def read_counters(module: Any) -> dict[str, int]:
    return {name: int(getattr(module, name, 0)) for name in COUNTER_NAMES}


def _validity_mark(module: Any) -> int | None:
    """How many validity IoUs the adapter has recorded, or None when it records none."""
    seq = getattr(module, VALIDITY_IOU_ATTR, None)
    if not isinstance(seq, Sequence) or isinstance(seq, (str, bytes)):
        return None
    return len(seq)


def _validity_since(module: Any, mark: int | None) -> float | None:
    """The single IoU recorded since ``mark``, or None when the check did not run on this frame.

    None rather than 0.0 on purpose, and for the reason ``DETECTION_SCORES`` gives for the same
    shape: 0.0 is a validity IoU that was measured and was zero — a mask on the wrong object — and
    "the check did not run" (no detection, or SAM 2 returned nothing) is not that claim.
    """
    if mark is None:
        return None
    now = _validity_mark(module)
    if now is None or now != mark + 1:
        return None
    try:
        return round(float(getattr(module, VALIDITY_IOU_ATTR)[mark]), 4)
    except (TypeError, ValueError, IndexError):
        return None


@dataclass
class FrameResult:
    """What one frame did, in the shape the artifact carries it."""

    mask: np.ndarray
    box: list[float] | None
    score: float | None
    all_scores: list[float]
    retry_fired: bool
    retry_recovered: bool
    no_detection: bool
    empty_mask: bool
    mask_refused: bool = False
    mask_refused_no_reference: bool = False
    mask_validity_iou: float | None = None
    postprocess_calls: list[dict[str, Any]] = field(default_factory=list)
    recorder_inconsistent: bool = False
    recorder_note: str = ""


def audit_one_frame(
    module: Any,
    mask_fn: Callable[[np.ndarray], np.ndarray],
    frame_bgr: np.ndarray,
    recorder: RecordingProcessor | None,
) -> FrameResult:
    """Segment one frame with the adapter as it stands and record everything it did.

    ``mask_fn`` is ``measure_geom_tol``'s own bound mask callable, so the BGR->RGB flip, the shape
    check and the refusal text are the ones GEOM_TOL uses. Nothing here re-implements the call.
    """
    if recorder is not None:
        recorder.calls.clear()
    before = read_counters(module)
    validity_before = _validity_mark(module)
    mask = np.asarray(mask_fn(frame_bgr)).astype(bool)
    after = read_counters(module)

    def delta(name: str) -> int:
        return after[name] - before[name]

    retry_fired = delta("RETRY_FRAMES") > 0
    retry_recovered = delta("RETRY_RECOVERED_FRAMES") > 0
    no_detection = delta("NO_DETECTION_FRAMES") > 0
    empty_mask = delta("EMPTY_MASK_FRAMES") > 0
    mask_refused = delta("MASK_REFUSED_FRAMES") > 0
    mask_refused_no_reference = delta("MASK_REFUSED_NO_REFERENCE_FRAMES") > 0
    # The per-frame IoU the filter decided on, taken as a DIFFERENCE for the same reason the
    # counters are: the adapter's list is cumulative over the import and a bare [-1] would read the
    # previous frame's value on any frame where the check did not run.
    validity_iou = _validity_since(module, validity_before)

    calls = list(recorder.calls) if recorder is not None else []
    scores: list[float] = []
    box: list[float] | None = None
    score: float | None = None
    if calls:
        last = calls[-1]
        scores = [float(s) for s in last.get("scores", [])]
        boxes = last.get("boxes") or []
        if scores and boxes:
            best = int(np.argmax(np.asarray(scores)))
            score = float(scores[best])
            box = [float(v) for v in boxes[best]]

    inconsistent = False
    note = ""
    if recorder is not None:
        expected_calls = 2 if retry_fired else 1
        if len(calls) != expected_calls:
            inconsistent = True
            note = (f"the recorder saw {len(calls)} post-processing call(s) and the adapter's "
                    f"counters say {expected_calls}")
        elif no_detection and box is not None:
            inconsistent = True
            note = "the adapter counted a no-detection frame and the recorder saw a winning box"
        elif not no_detection and box is None:
            inconsistent = True
            note = "the adapter did not count a no-detection frame and the recorder saw no box"
    # An all-False mask must have a recorded reason. There are exactly three — no detection, an
    # empty mask from a detected box, and a refusal by the mask-validity filter — and a fourth,
    # unexplained one would be a step silently dropped from every coverage number downstream. This
    # is the check that would notice.
    if not inconsistent and not mask.any() and not (no_detection or empty_mask or mask_refused):
        inconsistent = True
        note = ("the adapter returned an all-False mask and counted no no-detection, no empty mask "
                "and no validity refusal — a dropped step with no recorded reason")

    return FrameResult(
        mask=mask,
        box=box,
        score=score,
        all_scores=scores,
        retry_fired=retry_fired,
        retry_recovered=retry_recovered,
        no_detection=no_detection,
        empty_mask=empty_mask,
        mask_refused=mask_refused,
        mask_refused_no_reference=mask_refused_no_reference,
        mask_validity_iou=validity_iou,
        postprocess_calls=calls,
        recorder_inconsistent=inconsistent,
        recorder_note=note,
    )


# -- triage ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TriageThresholds:
    area_band_low: float = DEFAULT_AREA_BAND_LOW
    area_band_high: float = DEFAULT_AREA_BAND_HIGH
    frame_fraction_ceiling: float = DEFAULT_FRAME_FRACTION_CEILING
    centroid_jump_px: float = DEFAULT_CENTROID_JUMP_PX
    plate_overlap_fraction: float = DEFAULT_PLATE_OVERLAP_FRACTION
    min_warm_iou: float = DEFAULT_MIN_WARM_IOU
    warm_visible_px: int = DEFAULT_WARM_VISIBLE_PX
    low_score: float = DEFAULT_LOW_SCORE

    def as_dict(self) -> dict[str, Any]:
        return {
            "mask_area_band_x_median_warm_px": [self.area_band_low, self.area_band_high],
            "frame_fraction_ceiling": self.frame_fraction_ceiling,
            "centroid_jump_px": self.centroid_jump_px,
            "plate_overlap_fraction": self.plate_overlap_fraction,
            "min_warm_iou": self.min_warm_iou,
            "warm_visible_px": self.warm_visible_px,
            "low_score": self.low_score,
        }


def flag_frame(
    record: dict[str, Any],
    *,
    median_warm_px: float,
    frame_px: int,
    thresholds: TriageThresholds,
) -> list[str]:
    """Which flags this frame raises. Pure, deterministic, and not a verdict about anything."""
    flags: list[str] = []
    if record.get("no_detection"):
        flags.append("no_detection")
    if record.get("empty_mask"):
        flags.append("empty_mask")
    # ADDITIVE. The filter refusing a frame does not suppress any other flag, and nothing below is
    # skipped for a refused frame: a refusal that fired for the wrong reason has to stay visible,
    # and `disagrees_with_warm_apple` in particular still fires on a refused frame whose fruit was
    # plainly visible — which is what a false refusal would look like.
    if record.get("mask_refused"):
        flags.append("mask_refused")
    if record.get("mask_refused_no_reference"):
        flags.append("mask_refused_no_reference")
    if record.get("retry_fired"):
        flags.append("retry_fired")
    if record.get("retry_recovered"):
        flags.append("retry_recovered")
    score = record.get("detection_score")
    if score is not None and float(score) < thresholds.low_score:
        flags.append("low_score")
    area = int(record.get("mask_area_px") or 0)
    if area > 0 and median_warm_px > 0:
        if area > thresholds.area_band_high * median_warm_px:
            flags.append("mask_area_above_band")
        elif area < thresholds.area_band_low * median_warm_px:
            flags.append("mask_area_below_band")
    if frame_px > 0 and area > thresholds.frame_fraction_ceiling * frame_px:
        flags.append("mask_covers_frame")
    jump = record.get("centroid_step_px")
    if jump is not None and float(jump) > thresholds.centroid_jump_px:
        flags.append("centroid_jump")
    plate = record.get("plate_overlap_fraction")
    if (plate is not None and float(plate) >= thresholds.plate_overlap_fraction
            and median_warm_px > 0 and area > 2.0 * median_warm_px):
        flags.append("plate_overlap")
    warm_px = record.get("warm_apple_px")
    warm_iou = record.get("warm_apple_iou")
    if (warm_px is not None and warm_iou is not None
            and int(warm_px) >= thresholds.warm_visible_px
            and float(warm_iou) < thresholds.min_warm_iou):
        flags.append("disagrees_with_warm_apple")
    if record.get("recorder_inconsistent"):
        flags.append("recorder_inconsistent")
    return flags


def distribution(values: Sequence[float], *, bins: int = 20) -> dict[str, Any]:
    """Percentiles, a histogram and the count. The distribution blocker 2 asks to see."""
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "note": "no values"}
    counts, edges = np.histogram(arr, bins=bins)
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "histogram": {"counts": [int(c) for c in counts], "bin_edges": [float(e) for e in edges]},
    }


# -- overlays ------------------------------------------------------------------------------------------
#
# THIS ARTIFACT EXISTS TO BE LOOKED AT. Everything below is optimised for a person's eye and not for
# a JSON parser: the mask is tinted AND outlined so a two-pixel sliver is still visible, the
# colour-heuristic apple is outlined separately in a different colour so a disagreement is visible
# without cross-referencing a number, and the caption says what happened in words rather than in
# keys.

COLOR_MASK = (0, 255, 90)          # SAM 2's mask — green, because the fruit is red
COLOR_WARM = (60, 150, 255)        # the colour heuristic's apple — blue
COLOR_BOX = (255, 210, 0)          # GroundingDINO's winning box — yellow
COLOR_FLAG = (255, 60, 60)

LEGEND = ("green = SAM 2 mask (tint + outline) | blue = colour-heuristic apple (NOT ground "
          "truth) | yellow = detector box")


def _draw_box(arr: np.ndarray, box: Sequence[float], color: tuple[int, int, int],
              thickness: int = 2) -> None:
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    x0, x1 = sorted((max(0, min(w - 1, x0)), max(0, min(w - 1, x1))))
    y0, y1 = sorted((max(0, min(h - 1, y0)), max(0, min(h - 1, y1))))
    t = max(int(thickness), 1)
    arr[y0:min(y0 + t, h), x0:x1 + 1] = color
    arr[max(y1 - t + 1, 0):y1 + 1, x0:x1 + 1] = color
    arr[y0:y1 + 1, x0:min(x0 + t, w)] = color
    arr[y0:y1 + 1, max(x1 - t + 1, 0):x1 + 1] = color


def composite(
    rgb: np.ndarray,
    mask: np.ndarray | None,
    warm: np.ndarray | None,
    box: Sequence[float] | None,
) -> np.ndarray:
    """The frame with everything drawn on it, at full resolution."""
    out = np.asarray(rgb).astype(np.float32).copy()
    if mask is not None and mask.any():
        out[mask] = out[mask] * 0.72 + np.asarray(COLOR_MASK, dtype=np.float32) * 0.28
    arr = np.clip(out, 0, 255).astype(np.uint8)
    if warm is not None and warm.any():
        arr[boundary(warm, thickness=2)] = COLOR_WARM
    if mask is not None and mask.any():
        arr[boundary(mask, thickness=2)] = COLOR_MASK
    if box is not None:
        _draw_box(arr, box, COLOR_BOX)
    return arr


def _font(size: int) -> Any:
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no size argument
        return ImageFont.load_default()


def captioned(
    arr: np.ndarray,
    lines: Sequence[str],
    *,
    font_size: int = 13,
    flagged: bool = False,
) -> Any:
    """``arr`` with a caption band under it, as a PIL image."""
    from PIL import Image, ImageDraw

    font = _font(font_size)
    pad = 4
    line_h = font_size + 3
    band = pad * 2 + line_h * len(lines)
    h, w = arr.shape[:2]
    canvas = Image.new("RGB", (w, h + band), (18, 18, 20))
    canvas.paste(Image.fromarray(arr), (0, 0))
    draw = ImageDraw.Draw(canvas)
    if flagged:
        draw.rectangle([0, 0, w - 1, h + band - 1], outline=COLOR_FLAG, width=3)
    for i, line in enumerate(lines):
        draw.text((pad, h + pad + i * line_h), line,
                  fill=COLOR_FLAG if (flagged and i == len(lines) - 1) else (232, 232, 236),
                  font=font)
    return canvas


def contact_sheet(tiles: Sequence[Any], title: str, cols: int = 4) -> Any:
    """A grid of tiles with a header. What a reviewer actually scans."""
    from PIL import Image, ImageDraw

    if not tiles:
        raise AuditError("contact_sheet called with no tiles")
    tw = max(t.width for t in tiles)
    th = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    header = 28
    gap = 6
    sheet = Image.new(
        "RGB",
        (cols * tw + gap * (cols + 1), header + rows * th + gap * (rows + 1)),
        (10, 10, 12),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 6), title, fill=(240, 240, 244), font=_font(14))
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet.paste(tile, (gap + c * (tw + gap), header + gap + r * (th + gap)))
    return sheet


# -- provenance ----------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_census(path: Path | None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """``probe_census.json``, and the provenance block that says which one it was."""
    if path is None:
        return None, {"path": None, "used": False,
                      "note": "no census supplied; the occlusion strata rest on the fresh "
                              "per-episode scan alone, which cannot see the episodes this run did "
                              "not sample."}
    if not path.is_file():
        raise AuditError(
            f"FATAL: --census {path} does not exist. That file is `build_identity_calibration.py "
            "probe-scan`'s measurement over all 154447 frames of the 362 non-measured episodes, "
            "and it is the only thing that knows WHICH episodes contain an occlusion. Without it "
            "an episode sample drawn by stride will, on this corpus, contain no occluded frame at "
            "all and the artifact would be silently weaker than it looks. Copy it to this machine, "
            "or pass --allow-missing-census to say that a sample with no measured occlusion is "
            "what you meant."
        )
    doc = json.loads(path.read_text())
    return doc, {
        "path": str(path),
        "sha256": sha256_file(path),
        "used": True,
        "built_utc": doc.get("built_utc"),
        "rule": doc.get("rule"),
        "corpus": doc.get("corpus"),
        "note": (
            "probe-scan measured the strict warm-apple mask on every frame of the 362 non-measured "
            "episodes. Episodes it names as containing a below-threshold frame are forced into "
            "this sample; its eligible frames are taken as the `census` stratum verbatim."
        ),
    }


def census_episode_frames(census: dict[str, Any] | None) -> dict[str, list[int]]:
    """episode -> the census's own eligible frame indices."""
    out: dict[str, list[int]] = {}
    for row in (census or {}).get("eligible_frames") or []:
        ep = str(row.get("episode"))
        idx = row.get("frame_index")
        if ep and idx is not None:
            out.setdefault(ep, []).append(int(idx))
    return {k: sorted(v) for k, v in out.items()}


def census_forced_episodes(census: dict[str, Any] | None) -> list[str]:
    """Every episode the census saw a below-threshold frame in, whether or not it was eligible."""
    forced: set[str] = set(census_episode_frames(census))
    for ep, rec in ((census or {}).get("per_episode") or {}).items():
        if int(rec.get("n_below_census") or 0) > 0:
            forced.add(str(ep))
    return sorted(forced)


# -- the run -------------------------------------------------------------------------------------------


def _read_frames(clip: Path, decoder: Any, wanted: set[int]) -> dict[int, np.ndarray]:
    """Second decode pass: the BGR frames at ``wanted``, and nothing else held in memory."""
    frames, _fps = decoder.open_fn(clip)
    out: dict[int, np.ndarray] = {}
    last = max(wanted) if wanted else -1
    for i, frame in enumerate(frames):
        if i in wanted:
            out[i] = np.asarray(frame).copy()
        if i >= last:
            break
    return out


def _plate_reference(rgb: np.ndarray) -> tuple[np.ndarray | None, str]:
    """The plate region, from ``build_identity_calibration.plate_mask``, or why there is none.

    Computed ONCE per episode from its first frame: the plate does not move, and running a region
    grow per frame would cost more than the segmenter it is auditing.
    """
    try:
        return bic.plate_mask(rgb), "build_identity_calibration.plate_mask on this episode's frame 0"
    except Exception as exc:  # noqa: BLE001 - a missing plate is a fact about the frame
        return None, f"no plate found on frame 0: {type(exc).__name__}: {exc}"


def run_audit(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    frames_dir = out_dir / "frames"
    sheets_dir = out_dir / "sheets"
    for d in (out_dir, frames_dir, sheets_dir):
        d.mkdir(parents=True, exist_ok=True)

    corpus = Path(args.corpus).resolve()
    if not corpus.is_dir():
        raise AuditError(f"FATAL: --corpus {corpus} is not a directory.")

    census_path = Path(args.census).resolve() if args.census else None
    if census_path is None and not args.allow_missing_census:
        raise AuditError(
            "FATAL: no --census was given. The census is `build_identity_calibration.py "
            "probe-scan`'s measurement over all 154447 frames of the 362 non-measured episodes, "
            "and it is the only thing that knows WHICH episodes contain an occlusion. Blocker 1 "
            "asks specifically for occluded frames, and on this corpus an episode sample drawn by "
            "stride contains none of them: the census found 48 below-threshold frames in one "
            "episode out of 362. Pass --census <probe_census.json>, or --allow-missing-census to "
            "record in the artifact that this sample has no measured occlusion in it."
        )
    census, census_meta = load_census(census_path)

    episodes, layout = mgt.find_episodes(corpus, args.camera_key)
    keys = [e.key for e in episodes]
    by_key = {e.key: e for e in episodes}
    forced = [k for k in census_forced_episodes(census) if k in by_key]
    chosen_keys, episode_rule = select_episodes(keys, forced, args.episodes)

    probe_clip = by_key[chosen_keys[0]].clip
    if probe_clip is None:
        raise AuditError("FATAL: the corpus enumerated an episode with no clip to decode.")
    decoder = mgt.resolve_decoder(args.decoder, probe_clip)

    # The adapter, and the same bound mask callable GEOM_TOL uses. sam2_method() refuses loudly when
    # the checkpoints are absent, which is the refusal this run wants — before any decoding.
    method = mgt.sam2_method(args.min_area_px)
    module = mgt._import_sam2_adapter()
    recorder = attach_recorder(module)
    def mask_fn(frame_bgr: np.ndarray) -> np.ndarray:
        return method.mask_fn(frame_bgr, method)

    census_frames = census_episode_frames(census)
    per_episode: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    tiles: dict[str, list[Any]] = {}
    flagged_tiles: list[Any] = []
    print(f"=== {len(chosen_keys)} episode(s) of {len(keys)}; decoder {decoder.name} "
          f"{decoder.version}", file=sys.stderr)

    for n_ep, key in enumerate(chosen_keys, 1):
        clip = by_key[key].clip
        if clip is None:
            continue
        frames_iter, _fps = decoder.open_fn(clip)
        scan = scan_episode(frames_iter, key, lift_px=args.lift_px,
                            lift_persistence=args.lift_persistence)
        anchors, sel_meta = select_frames(
            scan,
            census_frames=census_frames.get(key, ()),
            max_anchors=args.max_anchors_per_episode,
            neighbour_offset=args.neighbour_offset,
        )
        wanted = {a.frame_index for a in anchors}
        got = _read_frames(clip, decoder, wanted)
        plate = None
        plate_note = "not computed"
        if got:
            first = got[min(got)]
            plate, plate_note = _plate_reference(np.ascontiguousarray(first[:, :, ::-1]))
        sel_meta["plate_reference"] = plate_note
        per_episode[key] = sel_meta
        print(f"===   [{n_ep}/{len(chosen_keys)}] {key}: {scan.n_frames} frames, "
              f"{len(anchors)} sampled", file=sys.stderr)

        prev_by_index: dict[int, tuple[float, float] | None] = {}
        for anchor in anchors:
            frame_bgr = got.get(anchor.frame_index)
            if frame_bgr is None:
                continue
            rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
            warm = warm_apple_mask(rgb)
            result = audit_one_frame(module, mask_fn, frame_bgr, recorder)
            mask = result.mask
            centroid = mgt.centroid_of_mask(mask, largest_component=True,
                                            min_area=args.min_area_px)
            prev_by_index[anchor.frame_index] = centroid
            step = None
            if anchor.role == "neighbour" and anchor.pair_of in prev_by_index:
                a = prev_by_index[anchor.pair_of]
                if a is not None and centroid is not None:
                    step = float(np.hypot(centroid[0] - a[0], centroid[1] - a[1]))
            area = int(mask.sum())
            plate_frac = None
            if plate is not None and area:
                plate_frac = float((mask & plate).sum()) / float(area)
            rec: dict[str, Any] = {
                "episode": key,
                "frame_index": int(anchor.frame_index),
                "stratum": anchor.stratum,
                "role": anchor.role,
                "pair_of": anchor.pair_of,
                "why_sampled": anchor.why,
                "detection_score": result.score,
                "detection_scores_all": result.all_scores,
                "box_xyxy": result.box,
                "retry_fired": result.retry_fired,
                "retry_recovered": result.retry_recovered,
                "no_detection": result.no_detection,
                "empty_mask": result.empty_mask,
                # PR-08 V6, per frame rather than only in the run's totals: this is what lets a
                # later reader check that the filter fired on exactly the frames a person flagged
                # in this artifact's predecessor, instead of taking a count on trust.
                "mask_refused": result.mask_refused,
                "mask_refused_no_reference": result.mask_refused_no_reference,
                "mask_validity_iou": result.mask_validity_iou,
                "postprocess_calls": result.postprocess_calls,
                "recorder_inconsistent": result.recorder_inconsistent,
                "recorder_note": result.recorder_note,
                "mask_area_px": area,
                "mask_centroid_xy": list(centroid) if centroid else None,
                "mask_touches_border": touches_border(mask),
                "centroid_step_px": step,
                "warm_apple_px": int(warm.sum()),
                "warm_apple_iou": round(iou(mask, warm), 4),
                "plate_overlap_fraction": None if plate_frac is None else round(plate_frac, 4),
                "episode_median_warm_px": scan.median_warm_px,
            }
            rec["flags"] = flag_frame(
                rec,
                median_warm_px=scan.median_warm_px,
                frame_px=int(mask.size),
                thresholds=args.thresholds,
            )
            records.append(rec)

            if not args.no_overlays:
                shot = composite(rgb, mask, warm, result.box)
                cap = _caption_lines(rec)
                png = frames_dir / f"{key}_f{anchor.frame_index:05d}_{anchor.role}.png"
                captioned(shot, cap, font_size=13, flagged=bool(rec["flags"])).save(png)
                rec["overlay"] = str(png.relative_to(out_dir))
                tile = captioned(shot[::2, ::2], cap, font_size=10, flagged=bool(rec["flags"]))
                tiles.setdefault(anchor.stratum, []).append(tile)
                if rec["flags"]:
                    flagged_tiles.append(tile)

    if not records:
        raise AuditError("FATAL: nothing was segmented. Nothing was written.")

    sheets: list[str] = []
    if not args.no_overlays:
        for stratum in STRATUM_ORDER:
            group = tiles.get(stratum) or []
            for i in range(0, len(group), args.sheet_tiles):
                chunk = group[i:i + args.sheet_tiles]
                title = f"{stratum} | sheet {i // args.sheet_tiles:02d} | {LEGEND}"
                path = sheets_dir / f"{stratum}-{i // args.sheet_tiles:02d}.png"
                contact_sheet(chunk, title, cols=args.sheet_cols).save(path)
                sheets.append(str(path.relative_to(out_dir)))
        for i in range(0, len(flagged_tiles), args.sheet_tiles):
            chunk = flagged_tiles[i:i + args.sheet_tiles]
            path = sheets_dir / f"flagged-{i // args.sheet_tiles:02d}.png"
            contact_sheet(chunk, f"AUTO-FLAGGED | sheet {i // args.sheet_tiles:02d} | triage, not a "
                                 f"verdict | {LEGEND}", cols=args.sheet_cols).save(path)
            sheets.insert(0, str(path.relative_to(out_dir)))

    artifact = build_artifact(
        records=records,
        per_episode=per_episode,
        episode_rule=episode_rule,
        chosen_keys=chosen_keys,
        all_keys=keys,
        layout=layout,
        corpus=corpus,
        decoder=decoder,
        method=method,
        module=module,
        census_meta=census_meta,
        sheets=sheets,
        args=args,
    )
    (out_dir / "MASK_AUDIT.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (out_dir / "OBSERVATIONS.template.json").write_text(
        json.dumps(observations_template(records, sheets), indent=2) + "\n")

    print(f"=== wrote {out_dir / 'MASK_AUDIT.json'}", file=sys.stderr)
    print(f"=== {len(records)} frame(s), {sum(1 for r in records if r['flags'])} flagged, "
          f"{len(sheets)} contact sheet(s)", file=sys.stderr)
    print("=== NOW LOOK AT THE CONTACT SHEETS and fill OBSERVATIONS.template.json in. Nothing here",
          file=sys.stderr)
    print("=== discharges a blocker: this is the evidence, the discharge is a person's edit to",
          file=sys.stderr)
    print("=== scripts/estimators/apple_sam2.py's GATE_QUALIFICATION_BLOCKERS.", file=sys.stderr)
    return 0


def _caption_lines(rec: dict[str, Any]) -> list[str]:
    score = "none" if rec["detection_score"] is None else f"{rec['detection_score']:.3f}"
    retry = "RETRY" if rec["retry_fired"] else "-"
    if rec["retry_fired"] and rec["retry_recovered"]:
        retry = "RETRY->box"
    lines = [
        f"{rec['episode']} f{rec['frame_index']:05d}  [{rec['stratum']}/{rec['role']}]",
        f"score {score}  {retry}  mask {rec['mask_area_px']}px  warm {rec['warm_apple_px']}px  "
        f"IoU {rec['warm_apple_iou']:.2f}",
    ]
    step = rec.get("centroid_step_px")
    plate = rec.get("plate_overlap_fraction")
    extra = []
    if step is not None:
        extra.append(f"step {step:.1f}px")
    if plate is not None:
        extra.append(f"plate {plate:.2f}")
    if rec["no_detection"]:
        extra.append("NO DETECTION")
    if rec["empty_mask"]:
        extra.append("EMPTY MASK")
    if rec.get("mask_refused"):
        # The IoU the filter decided on, not the returned mask's — the returned mask is empty, so
        # the caption would otherwise read "IoU 0.00" for a refusal and for an occlusion alike.
        vi = rec.get("mask_validity_iou")
        extra.append("REFUSED" + ("" if vi is None else f" (val IoU {vi:.2f})"))
    if rec.get("mask_refused_no_reference"):
        extra.append("no fruit visible")
    lines.append("  ".join(extra) if extra else " ")
    if rec["flags"]:
        # Wrapped rather than truncated: the tile is 320 px wide and a cut-off flag list drops the
        # flags that come last alphabetically, which is a silent, systematic omission.
        text = "FLAGS: " + ", ".join(rec["flags"])
        while text:
            lines.append(text[:46])
            text = text[46:]
    return lines


def build_artifact(
    *,
    records: list[dict[str, Any]],
    per_episode: dict[str, Any],
    episode_rule: dict[str, Any],
    chosen_keys: Sequence[str],
    all_keys: Sequence[str],
    layout: str,
    corpus: Path,
    decoder: Any,
    method: Any,
    module: Any,
    census_meta: dict[str, Any],
    sheets: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """The JSON half of the evidence. Everything a reader needs to argue with the sample."""
    scored = [r["detection_score"] for r in records if r["detection_score"] is not None]
    areas = [r["mask_area_px"] for r in records if r["mask_area_px"]]
    steps = [r["centroid_step_px"] for r in records if r.get("centroid_step_px") is not None]
    n = len(records)
    # "The corpus has no such frame" and "the anchor budget cut it" are opposite findings that
    # produce the same empty stratum. The candidates each episode's scan FOUND are the ones that
    # separate them, so the not-found list is computed from the picks and not from what survived.
    with_candidates = {
        s for meta in per_episode.values()
        for s, picked in (meta.get("picked_per_stratum") or {}).items() if picked
    }
    not_found = [s for s in STRATUM_ORDER if s not in with_candidates]
    truncated = sorted(
        {s for meta in per_episode.values() for s in (meta.get("dropped_by_budget") or [])},
        key=STRATUM_ORDER.index,
    )
    flag_counts: dict[str, int] = {}
    for r in records:
        for f in r["flags"]:
            flag_counts[f] = flag_counts.get(f, 0) + 1

    stats = module.stats() if callable(getattr(module, "stats", None)) else {}
    return {
        "schema": SCHEMA,
        "writeup": WRITEUP,
        "addresses": ADDRESSES,
        "not_a_discharge": NOT_A_DISCHARGE,
        "flags_are_triage": FLAGS_ARE_TRIAGE,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "produced_by": "scripts/audit_apple_masks.py",
        "slurm_job_id": args.slurm_job_id,
        "corpus": {
            "path": str(corpus),
            "layout": layout,
            "episodes_found": len(all_keys),
            "episodes_audited": len(chosen_keys),
            "episode_keys": list(chosen_keys),
        },
        "decoder": {"name": decoder.name, "version": decoder.version, "note": decoder.note},
        "sampling": {
            "rule": SAMPLING_RULE,
            "bias": SAMPLING_BIAS,
            "deterministic": True,
            "rng_used": False,
            "episode_selection": episode_rule,
            "stratum_meaning": STRATUM_MEANING,
            "strata_not_found": not_found,
            "strata_not_found_note": (
                "A stratum listed here was LOOKED FOR in every audited episode and NO episode "
                "produced a frame that met its rule. That is a measurement about the corpus, not "
                "an omission from the sample — and it is computed from the candidates the scans "
                "found, not from the frames that survived the anchor budget, which is a different "
                "and opposite reason for a stratum to be empty."
            ),
            "strata_truncated_by_budget": truncated,
            "strata_truncated_by_budget_note": (
                "These strata HAD candidates that --max-anchors-per-episode cut. They sit at the "
                "tail of the priority order, so the hard cases are never the ones dropped; raise "
                "the budget if a reviewer wants them."
            ),
            "per_episode": per_episode,
            "lift_px": args.lift_px,
            "lift_persistence": args.lift_persistence,
            "grasp_offsets": list(DEFAULT_GRASP_OFFSETS),
            "span_quantiles": list(DEFAULT_SPAN_QUANTILES),
            "neighbour_offset": args.neighbour_offset,
            "census": census_meta,
        },
        "counts": {
            "n_frames_segmented": n,
            "n_anchor_frames": sum(1 for r in records if r["role"] == "anchor"),
            "n_neighbour_frames": sum(1 for r in records if r["role"] == "neighbour"),
            "n_frames_flagged": sum(1 for r in records if r["flags"]),
            "per_stratum": {s: sum(1 for r in records if r["stratum"] == s) for s in STRATUM_ORDER},
            "flag_counts": flag_counts,
            "flag_meaning": FLAG_MEANING,
        },
        # PR-08 V6's filter, as a run-level total beside the per-frame `mask_refused` in `frames`.
        # Kept out of `blocker_2_numbers` on purpose: that block is spelled in the blocker's own
        # words and nothing here may quietly restate a blocker as satisfied.
        "mask_validity_filter": {
            "min_iou": stats.get("mask_validity_min_iou"),
            "reference": stats.get("mask_validity_reference"),
            "n_frames_refused": sum(1 for r in records if r.get("mask_refused")),
            "n_frames_refused_no_reference": sum(
                1 for r in records if r.get("mask_refused_no_reference")),
            "validity_iou_distribution": distribution(
                [r["mask_validity_iou"] for r in records if r.get("mask_validity_iou") is not None]
            ),
            "present": stats.get("mask_validity_min_iou") is not None,
            "note": (
                "The adapter refuses a non-empty mask containing essentially none of the object it "
                "claims to be, and returns all-False — which both PR-08 §4 harnesses already drop "
                "and count. `n_frames_refused_no_reference` is the sub-case where the colour "
                "reference found no fruit anywhere, i.e. the refusal removed a HARD frame rather "
                "than a wrong one; see PR-08 V6's threat to validity. A refused frame carries the "
                "`mask_refused` flag, and every other flag still applies to it — in particular "
                "`disagrees_with_warm_apple`, which is what a WRONG refusal would look like. "
                "`present` is false against an adapter that predates the filter, where the zeros "
                "above are 'no such mechanism' rather than 'it never fired'."
            ),
        },
        # The names blocker 2 uses, spelled exactly as it spells them, so the claim can be checked
        # against the blocker without a translation step.
        "blocker_2_numbers": {
            "n_frames_retry_fired": sum(1 for r in records if r["retry_fired"]),
            "n_frames_retry_recovered": sum(1 for r in records if r["retry_recovered"]),
            "n_frames_without_detection": sum(1 for r in records if r["no_detection"]),
            "n_frames_with_empty_mask": sum(1 for r in records if r["empty_mask"]),
            "no_detection_rate": round(sum(1 for r in records if r["no_detection"]) / n, 4),
            "retry_rate": round(sum(1 for r in records if r["retry_fired"]) / n, 4),
            "detection_score_distribution": distribution(scored),
            "mask_area_px_distribution": distribution(areas),
            "adjacent_frame_centroid_step_px_distribution": distribution(steps),
            "over": "THIS SAMPLE ONLY — see sampling.bias. Not a corpus rate.",
        },
        "full_pass_gap": (
            "Blocker 2 asks for these counts FROM A FULL PASS, and this sample is not one. The "
            "place they land now exists (2026-08-22): scripts/measure_geom_tol.py and "
            "scripts/measure_est_drift.py each write an `estimator_stats` block holding this run's "
            "counters — snapshotted before the pass and differenced after it, because the "
            "adapter's own counters are lifetime totals — and the detection-score distribution, "
            "with the shards carrying their raw scores per episode so the merge pools them exactly "
            "rather than approximately. What is still missing is the RUN: no full GEOM_TOL pass has "
            "been executed since, so the numbers in this artifact remain the only ones anybody has, "
            "and they are over a sample that over-weights the hard frames. Neither the block nor "
            "this sample discharges the blocker; a human still has to read them."
        ),
        "estimator": {
            "adapter_stats": stats,
            "mask_method_name": method.name,
            "mask_method_version": method.version,
            "mask_method_gate_qualified": bool(getattr(method, "gate_qualified", False)),
            "mask_method_params": getattr(method, "params", {}),
            "gate_qualified_read_from_adapter": bool(getattr(module, "GATE_QUALIFIED", False)),
            "gate_qualification_blockers_verbatim": list(
                getattr(module, "GATE_QUALIFICATION_BLOCKERS", ())),
            "blockers_discharged_by_this_run": [],
            "blockers_discharged_by_this_run_note": (
                "Empty by construction. Evidence is not a discharge; see not_a_discharge."
            ),
        },
        "triage_thresholds": args.thresholds.as_dict(),
        "human_review": {
            "looked_at": False,
            "established_by": None,
            "correlated_observer_warning": CORRELATED_OBSERVER,
            "observations_file": "OBSERVATIONS.template.json",
            "contact_sheets": list(sheets),
            "instruction": (
                "Open the contact sheets. For every frame, decide whether the green mask is the "
                "APPLE — not whether it is a plausible object. Write what you saw per frame into "
                "OBSERVATIONS.template.json, rename it to OBSERVATIONS.json, and set looked_at."
            ),
        },
        "frames": records,
    }


def observations_template(records: list[dict[str, Any]], sheets: Sequence[str]) -> dict[str, Any]:
    """The per-frame sheet a reviewer fills in — the shape ``probe_observations.json`` uses."""
    return {
        "schema": SCHEMA + "-observations",
        "step": "human review of PR-08 §4 mask overlays",
        "established_by": None,
        "established_by_note": CORRELATED_OBSERVER,
        "how_to_use": (
            "One entry per audited frame, in the order the contact sheets show them. For each: set "
            "`looked_at`, and write in `observed` what is actually in the picture — what the green "
            "mask covers, and whether that is the fruit, the plate, the hand, or the tabletop. "
            "`verdict` is one of apple / partial / wrong_object / no_mask / undecidable. An "
            "`undecidable` is a real answer on an occluded frame and is more useful than a guess."
        ),
        "contact_sheets": list(sheets),
        "verdict_values": ["apple", "partial", "wrong_object", "no_mask", "undecidable"],
        "frames": [
            {
                "episode": r["episode"],
                "frame_index": r["frame_index"],
                "stratum": r["stratum"],
                "role": r["role"],
                "overlay": r.get("overlay"),
                "flags": r["flags"],
                "detection_score": r["detection_score"],
                "mask_area_px": r["mask_area_px"],
                "looked_at": False,
                "verdict": None,
                "observed": "",
            }
            for r in records
        ],
    }


# -- CLI -------------------------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--corpus", required=True, type=Path,
                    help="the SOURCE corpus. Use the H.264 lossless tree: it is proven bit-exact "
                         "against the AV1 original and it is the only one the generation venv's "
                         "cv2 4.11.0 can read.")
    ap.add_argument("--out", required=True, type=Path, help="output directory (put it under runs/)")
    ap.add_argument("--census", type=Path, default=None,
                    help="probe_census.json from build_identity_calibration.py probe-scan")
    ap.add_argument("--allow-missing-census", action="store_true",
                    help="proceed with no census, recording that this sample contains no measured "
                         "occlusion")
    ap.add_argument("--episodes", type=int, default=24,
                    help="episode budget; 0 audits every episode")
    ap.add_argument("--max-anchors-per-episode", type=int, default=12)
    ap.add_argument("--neighbour-offset", type=int, default=1,
                    help="0 disables the adjacent-frame pairs, and with them the centroid-jump flag")
    ap.add_argument("--decoder", default="auto", choices=["auto", *mgt.DECODERS])
    ap.add_argument("--camera-key", default=None)
    ap.add_argument("--min-area-px", type=int, default=DEFAULT_MIN_AREA_PX,
                    help="measure_geom_tol's own default, reused so the centroid reported here is "
                         "the centroid GEOM_TOL would report for the same mask")
    ap.add_argument("--lift-px", type=float, default=DEFAULT_LIFT_PX)
    ap.add_argument("--lift-persistence", type=int, default=DEFAULT_LIFT_PERSISTENCE)
    ap.add_argument("--sheet-cols", type=int, default=4)
    ap.add_argument("--sheet-tiles", type=int, default=12)
    ap.add_argument("--no-overlays", action="store_true",
                    help="numbers only. Defeats the point of the exercise; here so a smoke test "
                         "can run without writing a thousand PNGs.")
    ap.add_argument("--slurm-job-id", default=None)
    for name, default in (
        ("--area-band-low", DEFAULT_AREA_BAND_LOW),
        ("--area-band-high", DEFAULT_AREA_BAND_HIGH),
        ("--frame-fraction-ceiling", DEFAULT_FRAME_FRACTION_CEILING),
        ("--centroid-jump-px", DEFAULT_CENTROID_JUMP_PX),
        ("--plate-overlap-fraction", DEFAULT_PLATE_OVERLAP_FRACTION),
        ("--min-warm-iou", DEFAULT_MIN_WARM_IOU),
        ("--low-score", DEFAULT_LOW_SCORE),
    ):
        ap.add_argument(name, type=float, default=default)
    ap.add_argument("--warm-visible-px", type=int, default=DEFAULT_WARM_VISIBLE_PX)
    args = ap.parse_args(argv)
    args.thresholds = TriageThresholds(
        area_band_low=args.area_band_low,
        area_band_high=args.area_band_high,
        frame_fraction_ceiling=args.frame_fraction_ceiling,
        centroid_jump_px=args.centroid_jump_px,
        plate_overlap_fraction=args.plate_overlap_fraction,
        min_warm_iou=args.min_warm_iou,
        warm_visible_px=args.warm_visible_px,
        low_score=args.low_score,
    )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return run_audit(args)
    except (AuditError, mgt.MethodUnavailable) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
