#!/usr/bin/env python3
"""Residue (i): every frame of one episode, and which of them the validity filter refuses.

    .venv/bin/python scripts/census_operating_point_episode.py \\
        --episode episode_000094 \\
        --corpus /home/humanoid/wam-t041/pr08-apple-640x480-h264-lossless \\
        --corpus /home/humanoid/wam-t041/pr08-apple-640x480 \\
        --out runs/pr08-operating-point/EPISODE_094_CENSUS.json

WHAT THIS IS FOR
----------------
``apple_sam2.GATE_QUALIFIED``'s own comment names two preconditions, and only one of them is the
propagation blocker. The other is *"somebody decides, on the record, what to do with the residue
the two 2026-08-26 entries carry forward — in particular blocker 2's residue (i), a failure the
discharged blocker predicted, occurring by a route it did not predict, on 92 frames."*

``PR-08-RESULT-2026-08-25-operating-point-on-this-corpus.md`` characterised that residue and then
named three limits on its own evidence. **This closes the second one**, which reads:

> **The exact 36 are not recorded.** The shard artifacts carry per-frame *scores* but no per-frame
> centroid-present flag, so "the 36 refusals are f109–f144" is **not** established here. What is
> established is that all 36 are in `episode_000094` and that the low-score run in that episode
> spans ~f101–f155.

It also closes the first limit — *"the phenomenon replicates; the numbers are not the same
numbers"* — by running BOTH decode trees over the same episode, so the AV1/H.264 difference is a
measured quantity rather than a caveat.

HOW, WITHOUT TOUCHING THE ADAPTER
---------------------------------
``segment()`` maintains module-level counters and returns an all-False mask for all three of its
"no usable mask" events — no detection, an empty mask from a real box, and a mask refused as being
of the wrong object. Those three are indistinguishable in the RETURN VALUE and that is exactly the
distinction residue (i) turns on. So this snapshots ``stats()`` around every single call and reads
the delta: the adapter runs unmodified, at its committed operating point, and the classification
comes from its own counters rather than from a reimplementation of its rules here.

WHAT IT DOES NOT DO
-------------------
**It does not decide anything.** Whether a refusal rate of *n* on this episode is acceptable is a
determination, it is registered blind in ``PR-08-V18``, and this script neither reads that document
nor writes a verdict. It does not flip ``GATE_QUALIFIED``, does not touch
``GATE_QUALIFICATION_BLOCKERS``, and copies the blocker tuple into its artifact verbatim so the
number and the standing objection travel together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from measure_geom_tol import centroid_of_mask as _centroid_of_mask  # noqa: E402
from robot_composite import _decode_frames  # noqa: E402

#: The harness's own floor, imported in spirit rather than re-chosen: `measure_est_drift measure`
#: and `measure_geom_tol` both default to 40, and the corpus pass that produced the 473 above ran
#: at that default.
MIN_AREA_PX = 40

SCHEMA = "wam.operating_point_census/1"
WRITEUP = "docs/preregistration/PR-08-V18-residue-i-decision-rule.md"

#: The three ways ``segment()`` returns an all-False mask, named by the counter that moves. They
#: are read as DELTAS around each call, never as totals: a total tells you how many frames of the
#: run were refused and this question is about WHICH.
EVENT_COUNTERS = (
    "n_frames_without_detection",
    "n_frames_with_empty_mask",
    "n_frames_mask_refused",
    "n_frames_mask_refused_no_reference",
    "n_frames_mask_refused_reference_not_object_scale",
)


def _centroid(mask: np.ndarray) -> list[float] | None:
    """``centroid_of_mask`` at the harness's own defaults, called rather than re-derived."""
    got = _centroid_of_mask(mask, largest_component=True, min_area=MIN_AREA_PX)
    return None if got is None else [float(got[0]), float(got[1])]


def census(episode: str, corpus: pathlib.Path, est: Any) -> dict[str, Any]:
    video = corpus / "videos" / f"{episode}.mp4"
    if not video.is_file():
        raise SystemExit(f"FATAL: {video} does not exist.")
    rows: list[dict[str, Any]] = []
    before = {k: est.stats().get(k, 0) for k in EVENT_COUNTERS}
    for index, rgb in enumerate(_decode_frames(video)):
        mask = np.asarray(est.segment(rgb), dtype=bool)
        after = {k: est.stats().get(k, 0) for k in EVENT_COUNTERS}
        fired = [k for k in EVENT_COUNTERS if after[k] > before[k]]
        before = after
        reference = est.object_color_reference(rgb)
        rows.append(
            {
                "frame": index,
                "mask_area_px": int(mask.sum()),
                "has_mask": bool(mask.any()),
                # The adapter's OWN classification of why this frame carries no mask, read from
                # its counters. An all-False mask means three different things and this is which.
                "events": fired,
                "warm_reference_px": int(reference.sum()),
                "warm_reference_frame_fraction": float(est.reference_frame_fraction(reference)),
                "reference_is_object_scale": bool(est.reference_is_object_scale(reference)),
                "mask_validity_iou": (
                    float(est.mask_validity_iou(mask, reference)) if mask.any() else None
                ),
                # THE QUANTITY THE CORPUS PASS ACTUALLY RECORDED, computed with the harness's own
                # function rather than approximated here. shard-7 reports episode_000094 as
                # `n_frames: 509, n_frames_with_centroid: 473`, i.e. 36 frames with no centroid —
                # and "no centroid" is NOT the same event as "mask refused": `centroid_of_mask`
                # also returns None when the mask's LARGEST CONNECTED COMPONENT is under
                # min_area_px, which a mask of adequate total area split into fragments can be.
                # Comparing 31 refusals against 36 missing centroids would be comparing two
                # different quantities and calling the difference a discrepancy.
                "centroid": _centroid(mask),
            }
        )
    no_centroid = [r["frame"] for r in rows if r["centroid"] is None]
    refused = [r["frame"] for r in rows if "n_frames_mask_refused" in r["events"]]
    no_detection = [r["frame"] for r in rows if "n_frames_without_detection" in r["events"]]
    empty = [r["frame"] for r in rows if "n_frames_with_empty_mask" in r["events"]]
    return {
        "corpus": str(corpus),
        "episode": episode,
        "n_frames": len(rows),
        "n_frames_with_mask": sum(1 for r in rows if r["has_mask"]),
        # Directly comparable to the corpus pass's per-episode n_frames_with_centroid.
        "n_frames_with_centroid": sum(1 for r in rows if r["centroid"] is not None),
        "no_centroid_frames": no_centroid,
        "n_no_centroid": len(no_centroid),
        "n_no_centroid_that_are_not_refusals": len(
            [f for f in no_centroid if f not in set(refused)]
        ),
        "refused_frames": refused,
        "n_refused": len(refused),
        "refused_span": [min(refused), max(refused)] if refused else None,
        "refused_is_contiguous": bool(
            refused and refused == list(range(min(refused), max(refused) + 1))
        ),
        "no_detection_frames": no_detection,
        "empty_mask_frames": empty,
        "frames": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--episode", required=True)
    ap.add_argument(
        "--corpus",
        type=pathlib.Path,
        action="append",
        required=True,
        help="repeatable. Pass BOTH decode trees to close limit 1 of the operating-point result: "
        "the AV1 original and the H.264-lossless transcode give slightly different per-frame "
        "scores, and this makes that difference a measurement instead of a caveat.",
    )
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)

    from estimators import apple_sam2 as est

    passes = [census(args.episode, corpus, est) for corpus in args.corpus]

    agreement: dict[str, Any] | None = None
    if len(passes) == 2:
        a, b = passes
        sa, sb = set(a["refused_frames"]), set(b["refused_frames"])
        agreement = {
            "meaning": (
                "The two decodes are the same recorded pixels through two codecs. A frame refused "
                "under one and not the other is a frame whose classification is a property of the "
                "codec rather than of the recording — which is the thing the operating-point "
                "result said must not be quoted interchangeably, now counted."
            ),
            "corpora": [a["corpus"], b["corpus"]],
            "n_refused": [a["n_refused"], b["n_refused"]],
            "refused_in_both": sorted(sa & sb),
            "refused_in_first_only": sorted(sa - sb),
            "refused_in_second_only": sorted(sb - sa),
            "jaccard": (len(sa & sb) / len(sa | sb)) if (sa | sb) else None,
        }

    record = {
        "schema": SCHEMA,
        "writeup": WRITEUP,
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "what_this_closes": (
            "Limit 2 of PR-08-RESULT-2026-08-25-operating-point-on-this-corpus.md — 'the exact 36 "
            "are not recorded'. They are recorded here, by frame index, from the adapter's own "
            "counters. Limit 1, the two decodes, is measured rather than caveated. Limit 3, the "
            "correlated observer, is NOT closed by this and cannot be: nothing here is a look."
        ),
        "decides": (
            "NOTHING. Whether this refusal pattern makes residue (i) acceptable is a determination "
            "registered blind in " + WRITEUP + ", and this script does not read it, does not write "
            "a verdict, and does not touch GATE_QUALIFIED or GATE_QUALIFICATION_BLOCKERS."
        ),
        "estimator": {
            "name": getattr(est, "ESTIMATOR_NAME", "unknown"),
            "version": getattr(est, "ESTIMATOR_VERSION", "unversioned"),
            "gate_qualified": bool(getattr(est, "GATE_QUALIFIED", False)),
            "segmenter_contract": dict(getattr(est, "SEGMENTER_CONTRACT", {}) or {}),
        },
        "gate_qualification_blockers": list(getattr(est, "GATE_QUALIFICATION_BLOCKERS", ())),
        "decode_agreement": agreement,
        "passes": passes,
        "estimator_stats": est.stats(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, indent=2) + "\n").encode("utf-8")
    args.out.write_bytes(payload)
    (args.out.parent / (args.out.name + ".sha256")).write_text(
        hashlib.sha256(payload).hexdigest() + "\n", encoding="utf-8"
    )
    for p in passes:
        print(
            f"{pathlib.Path(p['corpus']).name}: {p['n_frames']} frames, "
            f"{p['n_refused']} refused {p['refused_span']} contiguous={p['refused_is_contiguous']}, "
            f"{len(p['no_detection_frames'])} no-detection, {len(p['empty_mask_frames'])} empty"
        )
    if agreement:
        print(
            f"decode agreement: {len(agreement['refused_in_both'])} in both, "
            f"{len(agreement['refused_in_first_only'])} / "
            f"{len(agreement['refused_in_second_only'])} only, jaccard {agreement['jaccard']}"
        )
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
