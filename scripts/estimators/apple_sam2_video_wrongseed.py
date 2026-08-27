"""T40_RULE_V19 §3 — the propagation arm, seeded on the WRONG object. A CONTROL, not an arm.

    WAM_PR08_CONTROL_SEED_FROM_CAPTURE=runs/pr08-est-drift/v17/A1 \\
    WAM_PR08_CONTROL_SEED_LABEL=cube \\
    .venv/bin/python scripts/measure_est_drift.py measure \\
        --capture runs/pr08-est-drift/v17/A1 --estimators estimators.apple_sam2 \\
        --arm both --propagation-module estimators.apple_sam2_video_wrongseed \\
        --out runs/pr08-est-drift/v17/EST_DRIFT-C3-wrongseed.json

WHY THIS EXISTS
---------------
``low_iou_runs`` counts contiguous frames whose ground-truth IoU is below 0.5, and V17 §4's outcome
D turns on a run of **at least ten frames**. The statistic has been computed twice in this project.
It returned 0 runs on the coherent capture, and on the lattice control it returned 10 runs whose
longest was **5** — because the lattice's object returns under the stuck mask every fifth frame and
breaks the run, so *a lattice control cannot produce a run longer than the lattice's period*
(T40_RULE_V19 §1). **Nothing has ever shown that this statistic reports a LONG run.** Until
something does, a pooled zero from it is a statistic of unknown sensitivity reporting its only
observed value, and V17 §4 reads VOID.

This module is the thing that would show it. It is a **positive control**: a failure established on
purpose so the instrument can be seen catching it, which is the option `PR-08-V13` §3.3(c) names as
*"the only option that makes the bound a measurement of the thing it is supposed to catch."*

WHAT IS CHANGED, AND IT IS ONE THING
------------------------------------
**The seed box, and nothing else.** :func:`propagate` swaps ``apple_sam2_video.seed_box`` for a
constant and then calls ``apple_sam2_video.propagate`` — the real one, not a copy. So the video
predictor, the pinned revision, the in-memory ingest that keeps the two arms bit-identical, the
``logits > 0.0`` threshold, the counters and the would-have-been-refused bookkeeping are all
literally the measured arm's code. A reimplementation here could drift from that arm, and then a
control that fired would be evidence about a different propagator than the one under test.

**The seed does not come from the detector.** It is the bounding box of a named geom's
GROUND-TRUTH mask on frame 0, read from the renderer's ``seg_ids.npy``. A control whose seed came
from GroundingDINO could fail because the detector had a bad day, which would be uninformative in
exactly the direction that matters.

WHAT IT MUST NOT BE USED FOR
----------------------------
**It is not an arm and no number it produces is an ``EST_DRIFT``.** ``measure_est_drift`` will
happily write ``est_drift_p95_px`` from a run that used it; that number is the distance from the
cube to the apple and means nothing. Read ``low_iou_runs`` and nothing else. The artifact says so
in :data:`PROPAGATION_CONTRACT` so the sentence travels with the file.

It is reached only through ``--propagation-module``. ``apple_sam2_video.SEED_FRAME_INDEX`` and its
recorded reason — *"a sweep over seed frames would be a different experiment"* — are untouched, and
the measurement path grows no seed override.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Sequence

import numpy as np

from . import apple_sam2_video as _base

#: Where the ground truth to seed on is read from, and which geom to seed on. Environment rather
#: than flags because ``--propagation-module`` takes a module name and nothing else, and inventing
#: a flag on the measurement path for a control's benefit is what this module exists to avoid.
CAPTURE_ENV = "WAM_PR08_CONTROL_SEED_FROM_CAPTURE"
LABEL_ENV = "WAM_PR08_CONTROL_SEED_LABEL"
DEFAULT_LABEL = "cube"

#: Set by :func:`propagate`, reported by :func:`stats`.
LAST_CONTROL_SEED_BOX: list[float] | None = None
LAST_CONTROL_SEED_LABEL: str | None = None

PROPAGATION_CONTRACT: dict[str, Any] = {
    "role": "POSITIVE CONTROL, NOT AN ARM",
    "rule": "T40_RULE_V19 §3",
    "what_is_changed": (
        "the seed box only. apple_sam2_video.seed_box is swapped for a constant and the REAL "
        "apple_sam2_video.propagate is then called, so the predictor, the pins, the in-memory "
        "ingest, the logits>0.0 threshold and every counter are literally the measured arm's."
    ),
    "seed_source": (
        "the bounding box of a named geom's GROUND-TRUTH mask on frame 0 of the capture, read from "
        "the renderer's seg_ids.npy. NOT from GroundingDINO: a control whose seed came from the "
        "detector could fail because the detector had a bad day."
    ),
    "est_drift_is_meaningless_here": (
        "Any est_drift_p95_px produced under this module is the distance from the seeded object to "
        "the measured one. It is not an estimator error and must never be quoted as one, pooled, "
        "or carried into configs/transfer25/pr08_est_drift.json. Read low_iou_runs and nothing "
        "else."
    ),
    "what_it_can_show": (
        "that low_iou_runs reports a LONG contiguous run when one is present — the thing no "
        "measurement in this project has yet shown."
    ),
    "what_it_cannot_show": (
        "sensitivity to a SUBTLE drift. A held wrong seed is the grossest version of limb (b); a "
        "control that fires only on a failure larger than the corpus produces proves less than it "
        "looks. T40_RULE_V19 §5."
    ),
}


def seed_box_from_capture(capture: pathlib.Path, label: str) -> np.ndarray:
    """``[x0, y0, x1, y1]`` of ``label``'s ground-truth mask on frame 0 of ``capture``."""
    frame0 = capture / "frames" / "000000"
    labels = json.loads((frame0 / "seg_labels.json").read_text())
    ids = {str(v.get("class")): int(k) for k, v in labels.items()}
    if label not in ids:
        raise ValueError(
            f"{capture} frame 0 has no geom labelled {label!r}. It labels {sorted(ids)}. "
            "A control seeded on a geom that is not in the render would seed on nothing."
        )
    seg = np.load(frame0 / "seg_ids.npy")
    mask = seg == ids[label]
    if not mask.any():
        raise ValueError(
            f"{label!r} is a named geom of {capture} but covers no pixel on frame 0, so it is "
            "occluded or off-camera there. A control cannot be seeded on an invisible object; "
            "this is refused rather than seeding on an empty box."
        )
    ys, xs = np.nonzero(mask)
    return np.asarray(
        [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())], dtype=np.float32
    )


def _resolve_seed() -> tuple[np.ndarray, str, str]:
    capture = os.environ.get(CAPTURE_ENV)
    if not capture:
        raise RuntimeError(
            f"{CAPTURE_ENV} is not set. This module is a control and has to be told which capture's "
            "ground truth to seed the wrong object from; guessing would make the control's seed a "
            "property of whatever directory happened to be around."
        )
    label = os.environ.get(LABEL_ENV) or DEFAULT_LABEL
    return seed_box_from_capture(pathlib.Path(capture), label), capture, label


def propagate(rgbs: Sequence[np.ndarray]) -> list[np.ndarray]:
    """``apple_sam2_video.propagate``, with the seed box replaced and nothing else."""
    global LAST_CONTROL_SEED_BOX, LAST_CONTROL_SEED_LABEL

    box, _capture, label = _resolve_seed()
    LAST_CONTROL_SEED_BOX = [float(v) for v in box]
    LAST_CONTROL_SEED_LABEL = label

    original = _base.seed_box
    # Swapped for the duration of ONE call and put back in a finally, the same discipline
    # `_in_memory_frames` uses for its own monkeypatch: a module left permanently patched would
    # turn the measured arm into a control the next time anything imported it.
    _base.seed_box = lambda _frame: box  # type: ignore[assignment]
    try:
        return _base.propagate(rgbs)
    finally:
        _base.seed_box = original  # type: ignore[assignment]


def available() -> bool:
    return _base.available()


def reset_counters() -> None:
    global LAST_CONTROL_SEED_BOX, LAST_CONTROL_SEED_LABEL
    LAST_CONTROL_SEED_BOX = None
    LAST_CONTROL_SEED_LABEL = None
    _base.reset_counters()


def stats() -> dict[str, Any]:
    out = dict(_base.stats())
    out.update(
        {
            "control": PROPAGATION_CONTRACT,
            "control_seed_box_xyxy": LAST_CONTROL_SEED_BOX,
            "control_seed_label": LAST_CONTROL_SEED_LABEL,
            "control_seed_capture": os.environ.get(CAPTURE_ENV),
        }
    )
    return out
