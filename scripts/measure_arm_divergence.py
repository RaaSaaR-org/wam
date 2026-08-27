#!/usr/bin/env python3
"""T40_RULE_V17 Arm B — the two estimator arms, over the REAL corpus, disagreeing or not.

    .venv/bin/python scripts/measure_arm_divergence.py \\
        --corpus /home/humanoid/wam-t041/pr08-apple-640x480-h264-lossless \\
        --estimators estimators.apple_sam2 \\
        --out runs/pr08-est-drift/v17/ARM_DIVERGENCE.json

WHAT THIS MEASURES, AND WHAT IT CANNOT
--------------------------------------
``scripts/measure_est_drift.py`` measures each arm against a KNOWN TRUE MASK, which exists only in
a renderer. The real AppleToPlate corpus has no ground truth at all, so ``EST_DRIFT`` is not
measurable on it and nothing here is an ``EST_DRIFT``. **This measures AGREEMENT, not correctness.**
For every frame it computes the IoU between the per-frame arm's mask and the propagation arm's
mask, and then counts contiguous runs below the same 0.5 the drift rig uses.

THE INFERENCE THIS RESTS ON, STATED SO IT CAN BE REFUSED. A long run of disagreement has a cheap
explanation and an expensive one:

* cheap — the PER-FRAME arm is wrong for that stretch. Its errors are independent across frames by
  construction (it re-detects from scratch every frame), and its measured profile is exactly that:
  on the 480-frame capture it produced ONE sub-0.5 frame, a run of length 1. Independent failures
  at that rate produce a run of ten by coincidence essentially never.
* expensive — the PROPAGATION arm lost the object and stayed lost. Its failure is serially
  correlated by construction, because it carries state forward from frame 0. That is precisely
  limb (b) of the blocker, and it is why ``low_iou_runs`` was built.

So run LENGTH is what carries the attribution, which is why V17 §4's threshold is a length and not
a count, and why this script reports the length distribution rather than a single verdict.

**BOTH ARMS COULD BE WRONG TOGETHER AND THIS DESIGN CANNOT SEE IT.** A frame where both masks sit
on the plate agrees perfectly and scores IoU 1.0. That limitation is not a footnote — it is the
price of having no ground truth, it is stated in V17 §4 outcome N item 3, and it must travel with
every number this script writes.

WHAT IS NOT A FLAG, ON PURPOSE
------------------------------
No ``--threshold``: V17 §0 keeps ``LOW_IOU_THRESHOLD`` at 0.5 and this imports it rather than
restating it. No ``--episodes``: the sample is V17 §3's, fixed under ``sample_seed = 40017`` before
any of it was decoded, and it is re-derived here from the manifest rather than typed, so a corpus
that is not the one the sample was drawn over cannot be measured by accident. No ``--limit`` that
survives into a headline: a truncated run is stamped ``partial`` and disqualified, because a run
statistic over a truncated episode is a statistic about the truncation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import measure_est_drift as ed  # noqa: E402
from robot_composite import _decode_frames  # noqa: E402

SCHEMA = "wam.arm_divergence/1"
RULE = "T40_RULE_V17"
WRITEUP = "docs/preregistration/PR-08-V17-drift-rate-protocol.md"

#: V17 §3. The scheme, the seed and the size are the document's; the ids are DERIVED from the
#: corpus manifest under them rather than pasted, so this script and that document cannot disagree
#: about which 40 episodes were drawn without the derivation failing loudly.
SAMPLE_SEED = 40017
SAMPLE_SIZE = 40
SAMPLE_SCHEME = "stratified-systematic/1"

#: The ids V17 §3 wrote down. Kept as a CHECK on the derivation above, never as its source: if the
#: manifest ever lists a different 402, the draw would silently move and the registered sample
#: would quietly become a different sample.
REGISTERED_SAMPLE: tuple[str, ...] = (
    "episode_000005", "episode_000010", "episode_000021", "episode_000031", "episode_000049",
    "episode_000055", "episode_000060", "episode_000079", "episode_000083", "episode_000093",
    "episode_000107", "episode_000112", "episode_000129", "episode_000139", "episode_000148",
    "episode_000155", "episode_000165", "episode_000173", "episode_000185", "episode_000193",
    "episode_000207", "episode_000218", "episode_000223", "episode_000236", "episode_000250",
    "episode_000251", "episode_000267", "episode_000277", "episode_000287", "episode_000292",
    "episode_000301", "episode_000313", "episode_000326", "episode_000338", "episode_000348",
    "episode_000356", "episode_000366", "episode_000373", "episode_000386", "episode_000397",
)

#: V17 §4 outcome D. A propagation-side run this long or longer, that the per-frame arm does not
#: also show, is the observation that keeps the blocker open. Fixed blind, before the first frame
#: was decoded, and NOT a flag.
RUN_LENGTH_D = 10

_NO_GROUND_TRUTH = (
    "THIS IS NOT AN EST_DRIFT AND NOT AN ERROR. The real corpus carries no ground-truth mask, so "
    "every IoU here is between the TWO ARMS and measures whether they agree, not whether either is "
    "right. Two masks that are both wrong in the same place agree perfectly and score 1.0; this "
    "design cannot see that case and does not claim to."
)


def registered_sample(manifest: dict[str, Any]) -> list[str]:
    """V17 §3's draw, re-derived from the manifest under the registered scheme and seed."""
    ids = sorted(e["id"] for e in manifest["episodes"])
    total = len(ids)
    rng = random.Random(SAMPLE_SEED)
    drawn = [
        ids[rng.randrange(k * total // SAMPLE_SIZE, (k + 1) * total // SAMPLE_SIZE)]
        for k in range(SAMPLE_SIZE)
    ]
    if tuple(drawn) != REGISTERED_SAMPLE:
        raise SystemExit(
            "FATAL: re-drawing V17 §3's sample over this corpus does not reproduce the ids the "
            "document registered.\n"
            f"       corpus has {total} episodes; scheme {SAMPLE_SCHEME}, seed {SAMPLE_SEED}\n"
            f"       derived : {drawn[:3]} ... {drawn[-3:]}\n"
            f"       registered: {list(REGISTERED_SAMPLE)[:3]} ... {list(REGISTERED_SAMPLE)[-3:]}\n"
            "       Either this is a different corpus than the sample was drawn over, or the draw "
            "moved. Both make the pre-registration void; neither is repaired by measuring anyway."
        )
    return drawn


def cross_arm_iou(a: np.ndarray, b: np.ndarray) -> float | None:
    """IoU of two boolean masks. ``None`` when the union is empty — the same convention
    ``measure_est_drift.mask_iou`` uses, and the same reason: two empty masks are not a
    disagreement, and scoring them 0.0 would manufacture runs out of frames with no object."""
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return None
    return float(np.count_nonzero(a & b)) / float(union)


def episode_block(
    episode: str,
    frames: list[np.ndarray],
    estimator: Any,
    propagator: Any,
) -> dict[str, Any]:
    """One episode, both arms, the per-frame agreement and its runs."""
    per_frame_masks = [np.asarray(estimator.segment(rgb), dtype=bool) for rgb in frames]
    propagated = [np.asarray(m, dtype=bool) for m in propagator.propagate(frames)]
    if len(propagated) != len(frames):
        raise SystemExit(
            f"FATAL: {episode}: the propagation arm returned {len(propagated)} masks for "
            f"{len(frames)} frames. A mask list that does not line up with the frames cannot be "
            "compared frame by frame, and pairing them by position anyway would silently compare "
            "different instants."
        )
    ious = [cross_arm_iou(p, q) for p, q in zip(per_frame_masks, propagated)]
    runs = ed.low_iou_runs(ious, threshold=ed.LOW_IOU_THRESHOLD)
    scored = [v for v in ious if v is not None]
    return {
        "episode": episode,
        "n_frames": len(frames),
        "n_frames_scored": len(scored),
        "n_frames_both_masks_empty": sum(1 for v in ious if v is None),
        "n_frames_per_frame_arm_empty": sum(1 for m in per_frame_masks if not m.any()),
        "n_frames_propagation_arm_empty": sum(1 for m in propagated if not m.any()),
        "cross_arm_iou": {
            "meaning": _NO_GROUND_TRUTH,
            "median": float(np.median(scored)) if scored else None,
            "p05": float(np.percentile(scored, 5)) if scored else None,
            "min": float(min(scored)) if scored else None,
            "n_below_half": sum(1 for v in scored if v < ed.LOW_IOU_THRESHOLD),
        },
        "divergence_runs": runs,
        "longest_run": int(runs["longest_run"]),
        "meets_outcome_d": bool(runs["longest_run"] >= RUN_LENGTH_D),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=pathlib.Path, required=True)
    ap.add_argument("--estimators", default="estimators.apple_sam2")
    ap.add_argument("--propagation-module", default=ed.DEFAULT_PROPAGATION_MODULE)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="first N episodes of the registered sample; for smoke runs only. ANY non-zero value "
        "stamps partial=true and disqualifies the run, because a rate over part of a "
        "pre-registered sample is not the rate that sample was drawn to estimate.",
    )
    args = ap.parse_args(argv)

    proof = args.corpus / "TRANSCODE_PROOF.json"
    if args.corpus.name.endswith("h264-lossless") and not proof.is_file():
        raise SystemExit(
            f"FATAL: {args.corpus} is named lossless and carries no {proof.name}. A directory name "
            "is not evidence, and a lossy re-encode between the recorded pixels and both arms "
            "would be a confound neither arm could be separated from."
        )
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    sample = registered_sample(manifest)
    frames_by_id = {e["id"]: int(e["frames"]) for e in manifest["episodes"]}

    partial = int(args.limit) > 0
    if partial:
        sample = sample[: int(args.limit)]

    estimator = ed.resolve_estimators(args.estimators)
    propagator = ed.resolve_propagator(args.propagation_module)

    blocks: list[dict[str, Any]] = []
    for i, episode in enumerate(sample, 1):
        video = args.corpus / "videos" / f"{episode}.mp4"
        frames = [np.ascontiguousarray(f) for f in _decode_frames(video)]
        if len(frames) != frames_by_id[episode]:
            raise SystemExit(
                f"FATAL: {episode} decoded {len(frames)} frames, manifest declares "
                f"{frames_by_id[episode]}. A run statistic over a clip of the wrong length is a "
                "statistic about the decode."
            )
        block = episode_block(episode, frames, estimator, propagator)
        blocks.append(block)
        print(
            f"[{i}/{len(sample)}] {episode}: {block['n_frames']} frames, "
            f"longest divergence run {block['longest_run']}, "
            f"median cross-arm IoU {block['cross_arm_iou']['median']}",
            flush=True,
        )

    n_frames = sum(b["n_frames"] for b in blocks)
    lengths = [ln for b in blocks for ln in (b["divergence_runs"]["runs"] and
               [e - s + 1 for s, e in b["divergence_runs"]["runs"]] or [])]
    episodes_d = [b["episode"] for b in blocks if b["meets_outcome_d"]]

    record = {
        "schema": SCHEMA,
        "rule": RULE,
        "writeup": WRITEUP,
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "no_ground_truth": _NO_GROUND_TRUTH,
        "partial": partial,
        "gate_qualified": False,
        "gate_disqualified_reasons": (
            ["partial_run_limit"] if partial else []
        ) + ([] if estimator.gate_qualified else ["estimator_not_gate_qualified"]),
        "corpus": str(args.corpus),
        "sample": {
            "scheme": SAMPLE_SCHEME,
            "seed": SAMPLE_SEED,
            "size": SAMPLE_SIZE,
            "episodes": list(sample),
            "of_total_episodes": len(manifest["episodes"]),
            "registered_in": WRITEUP + " §3",
        },
        "low_iou_threshold": float(ed.LOW_IOU_THRESHOLD),
        "outcome_d_run_length": RUN_LENGTH_D,
        "n_episodes": len(blocks),
        "n_frames": n_frames,
        "n_divergence_runs": sum(int(b["divergence_runs"]["n_runs"]) for b in blocks),
        "longest_divergence_run": max([b["longest_run"] for b in blocks], default=0),
        "run_length_histogram": {
            str(k): lengths.count(k) for k in sorted(set(lengths))
        },
        "episodes_meeting_outcome_d": episodes_d,
        "outcome_d_met": bool(episodes_d),
        "rule_of_three_episode_rate_upper_95": (
            None if episodes_d else 3.0 / float(len(blocks))
        ),
        "rule_of_three_reading": (
            "not applicable — events were observed" if episodes_d else
            f"zero episodes of {len(blocks)} showed a run of >= {RUN_LENGTH_D}. The rule of three "
            f"bounds the per-episode rate at ~{3.0 / float(len(blocks)):.4f} with 95% confidence, "
            f"i.e. this clean sweep is still consistent with about "
            f"{round(3.0 / float(len(blocks)) * len(manifest['episodes']))} of the "
            f"{len(manifest['episodes'])} episodes containing one. THIS SAMPLE CAN DETECT "
            "DIVERGENCE; IT CANNOT CERTIFY ITS ABSENCE."
        ),
        "estimators": {
            "spec": estimator.spec,
            "name": estimator.name,
            "version": estimator.version,
            "gate_qualified": estimator.gate_qualified,
            "segmenter_contract": estimator.segmenter_contract,
            "object_text_prompt": estimator.object_text_prompt,
        },
        "propagation_module": args.propagation_module,
        "propagator_stats": propagator.stats() if hasattr(propagator, "stats") else None,
        "episodes": blocks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, indent=2) + "\n").encode("utf-8")
    args.out.write_bytes(payload)
    (args.out.parent / (args.out.name + ".sha256")).write_text(
        hashlib.sha256(payload).hexdigest() + "\n", encoding="utf-8"
    )
    print(
        f"\n{len(blocks)} episodes, {n_frames} frames, "
        f"{record['n_divergence_runs']} divergence runs, longest {record['longest_divergence_run']}"
    )
    print(f"outcome D met: {record['outcome_d_met']}  ->  {args.out}")
    return 0 if not partial else 3


if __name__ == "__main__":
    raise SystemExit(main())
