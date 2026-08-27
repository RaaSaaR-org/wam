#!/usr/bin/env python3
"""T40_RULE_V17 Arm A — eight capture artifacts pooled into one drift rate, without lying at the seams.

    .venv/bin/python scripts/pool_est_drift_arms.py \\
        --artifact runs/pr08-est-drift/v17/EST_DRIFT-A[1-8].json \\
        --control  runs/pr08-est-drift/v17/EST_DRIFT-C1-lattice.json \\
        --out runs/pr08-est-drift/v17/POOLED.json

WHY POOLING NEEDS ITS OWN SCRIPT AND NOT A LONGER --capture
-----------------------------------------------------------
``measure_est_drift.py measure`` takes ONE capture, and every statistic downstream is keyed to that
capture's single frame index space. Concatenating eight captures into one frame list would be the
obvious shortcut and it is wrong in a specific, silent way: ``low_iou_runs`` counts CONTIGUOUS
frames, so frame 479 of A1 and frame 0 of A2 would become neighbours, and a low-IoU frame at each
end would be reported as one run of two across a seam that does not exist. The propagation arm is
worse — it seeds from frame 0 of the list, so seven of the eight captures would be tracked from
another capture's first frame.

**So each capture is measured alone, and this pools the results under two rules:**

1. **Displacements pool; runs do not.** The pooled ``EST_DRIFT_P95`` is the 95th percentile over the
   UNION of every capture's per-frame displacements — one percentile taken once, never an average
   of eight percentiles, which is the same rule ``measure_geom_tol.py --merge`` follows for the
   16 GEOM_TOL shards. Runs are counted WITHIN each capture and then summed as counts; no run may
   span two captures because no such run exists.
2. **A capture that disagrees about the instrument is refused, not averaged in.** Different
   segmenter contract, different pixel grid, different device, a schedule that is not
   ``trajectory``, or a capture whose measured ``median_interframe_motion_px`` exceeds V17 §2's
   25.0 px coherence bound — each refuses the whole pool by name rather than dropping one artifact
   quietly.

THE CONTROL IS READ, NOT POOLED
-------------------------------
``--control`` takes the C1 artifact of V17 §5 — the committed lattice capture, whose object
teleports between neighbours. It is **never** pooled into the drift numbers: it is not a coherent
capture and its p95 would be a measurement of jump cuts. It is read for exactly one thing, the
question V17 §4 puts first: **did ``low_iou_runs`` fire at all?** A statistic that has never been
observed to report a drift that is present cannot be read as reporting its absence, and if the
control does not fire this script stamps ``outcome: V`` and refuses to compute the rest.
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

SCHEMA = "wam.est_drift_pooled/1"
RULE = "T40_RULE_V17"
WRITEUP = "docs/preregistration/PR-08-V17-drift-rate-protocol.md"

#: V17 §2. Adopted from ``tests/test_measure_est_drift.py``'s existing assertion for the trajectory
#: schedule rather than coined here, and applied to the MEASURED number in each capture's header.
COHERENCE_MAX_MEDIAN_PX = 25.0

#: V17 §4 outcome D, and V17 §5's fire condition for the control. One constant, because a control
#: that had to clear a *different* bar than the thing it controls for would not be a control.
RUN_LENGTH_D = 10

ARMS = ("per_frame", "propagation")


def _load(path: pathlib.Path) -> dict[str, Any]:
    doc = json.loads(path.read_text())
    if "arm_comparison" not in doc:
        raise SystemExit(
            f"FATAL: {path} carries no arm_comparison block, so it was measured with a single "
            "--arm. Pooling needs both arms from the same capture: the whole quantity is the "
            "DIFFERENCE between them, and a per-frame number from one capture beside a "
            "propagation number from another is not that difference."
        )
    return doc


def _instrument_key(doc: dict[str, Any]) -> tuple:
    """What must be identical across every pooled artifact for the union to be one population."""
    est = doc.get("estimators") or {}
    return (
        json.dumps(est.get("segmenter_contract"), sort_keys=True),
        tuple(doc.get("resolution_hw") or ()),
        str(doc.get("object_class")),
        str((doc.get("arm_comparison") or {}).get("propagator", {}).get("spec", "")),
        float((doc.get("arm_comparison") or {}).get("low_iou_threshold", -1.0)),
    )


def check_poolable(docs: dict[str, dict[str, Any]]) -> None:
    keys = {name: _instrument_key(doc) for name, doc in docs.items()}
    distinct = set(keys.values())
    if len(distinct) > 1:
        lines = "\n".join(f"       {n}: {k}" for n, k in sorted(keys.items()))
        raise SystemExit(
            "FATAL: these artifacts were not produced by the same instrument, so their frames are "
            "not one population and a percentile over their union means nothing.\n" + lines
        )
    for name, doc in sorted(docs.items()):
        capture = doc.get("capture") or {}
        if capture.get("scene_schedule") != "trajectory":
            raise SystemExit(
                f"FATAL: {name} was captured on the {capture.get('scene_schedule')!r} schedule. "
                "Arm A pools coherent captures only; a lattice teleports the object between "
                "neighbours, so a propagated mask crosses a cut on frame 1 and every number after "
                "it measures the cut. Pass it to --control instead, where it is read and not "
                "pooled."
            )
        median = ((capture.get("temporal_coherence") or {}).get("median_interframe_motion_px"))
        if median is None:
            raise SystemExit(
                f"FATAL: {name} records no measured median_interframe_motion_px. The schedule's "
                "NAME is not evidence that a capture is coherent — V17 §2 bounds the measured "
                "number, and an absent measurement is not a passing one."
            )
        if float(median) > COHERENCE_MAX_MEDIAN_PX:
            raise SystemExit(
                f"FATAL: {name} measures median_interframe_motion_px = {float(median):.4f}, over "
                f"V17 §2's bound of {COHERENCE_MAX_MEDIAN_PX}. It is excluded from Arm A BY NAME "
                "rather than dropped quietly; re-run the pool without it and say so in the result."
            )


def pooled_arm(docs: dict[str, dict[str, Any]], arm: str) -> dict[str, Any]:
    values: list[float] = []
    per_capture: list[dict[str, Any]] = []
    runs_total = 0
    longest = 0
    frames = 0
    measured = 0
    for name, doc in sorted(docs.items()):
        block = (doc["arm_comparison"] or {})[arm]
        if not block.get("measured"):
            raise SystemExit(
                f"FATAL: {name}'s {arm} arm is not measured ({block.get('absent_because')!r})."
            )
        dumps = block.get("displacements_px")
        if dumps is None:
            raise SystemExit(
                f"FATAL: {name}'s {arm} arm records no displacements_px. A pooled p95 is not any "
                "function of eight p95s, and re-deriving it from the 0.5 px histogram would "
                "quantise the answer to the bin width. Re-measure with a build that dumps them."
            )
        values.extend(float(v) for v in dumps)
        runs = block["low_iou_runs"]
        runs_total += int(runs["n_runs"])
        longest = max(longest, int(runs["longest_run"]))
        frames += int(block["n_frames"])
        measured += int(block["n_measured"])
        per_capture.append(
            {
                "capture": name,
                "schedule_params": (doc.get("capture") or {}).get("scene_schedule_params"),
                "median_interframe_motion_px": (
                    (doc.get("capture") or {}).get("temporal_coherence") or {}
                ).get("median_interframe_motion_px"),
                "est_drift_p95_px": block["est_drift_p95_px"],
                "n_runs": int(runs["n_runs"]),
                "longest_run": int(runs["longest_run"]),
            }
        )
    array = np.asarray(values, dtype=float)
    return {
        "arm": arm,
        "n_captures": len(docs),
        "n_frames": frames,
        "n_measured": measured,
        "coverage": (measured / frames) if frames else None,
        # ONE percentile, taken ONCE, over the union. Never a mean of the per-capture p95s.
        "pooled_est_drift_p95_px": float(np.percentile(array, 95)) if array.size else None,
        "pooled_percentiles_px": (
            {f"p{p}": float(np.percentile(array, p)) for p in (50, 90, 95, 99, 100)}
            if array.size
            else None
        ),
        "pooling_rule": (
            "the 95th percentile over the UNION of every capture's per-frame displacements. Shard "
            "percentiles are never averaged — the same rule measure_geom_tol.py --merge follows "
            "for the 16 GEOM_TOL shards."
        ),
        "n_low_iou_runs_summed": runs_total,
        "longest_low_iou_run_within_any_capture": longest,
        "runs_never_span_captures": (
            "TRUE BY CONSTRUCTION. Each capture's runs are counted inside its own frame index "
            "space and only the COUNTS are summed. Concatenating the captures would make frame "
            "479 of one the neighbour of frame 0 of the next and report a run across a seam that "
            "does not exist."
        ),
        "per_capture": per_capture,
    }


def control_verdict(path: pathlib.Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "fired": False,
            "reason": "no --control was given",
            "meaning": (
                "V17 §5 makes the control the FIRST thing outcome §4 reads. low_iou_runs has been "
                "computed once in this project, reported zero, and has never been observed to "
                "fire. Without a control, a pooled zero is not evidence of absence — it is a "
                "statistic of unknown sensitivity reporting its only observed value."
            ),
        }
    doc = _load(path)
    block = (doc["arm_comparison"] or {})["propagation"]
    runs = block["low_iou_runs"]
    fired = int(runs["longest_run"]) >= RUN_LENGTH_D
    return {
        "fired": bool(fired),
        "artifact": str(path),
        "capture": (doc.get("capture") or {}).get("path"),
        "schedule": (doc.get("capture") or {}).get("scene_schedule"),
        "median_interframe_motion_px": (
            (doc.get("capture") or {}).get("temporal_coherence") or {}
        ).get("median_interframe_motion_px"),
        "n_runs": int(runs["n_runs"]),
        "longest_run": int(runs["longest_run"]),
        "required_longest_run": RUN_LENGTH_D,
        "meaning": (
            "The control fires iff the propagation arm loses the object on a capture where it is "
            "KNOWN to be losable. It establishes that low_iou_runs can report a drift that is "
            "present. It does NOT establish sensitivity to a subtler drift on a slowly-moving "
            "object, which is what limb (b) actually worries about — see V17 §5's last paragraph."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--artifact", type=pathlib.Path, nargs="+", required=True)
    ap.add_argument("--control", type=pathlib.Path, default=None)
    ap.add_argument("--divergence", type=pathlib.Path, default=None,
                    help="Arm B's artifact from measure_arm_divergence.py, read for §4's outcome")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)

    docs = {p.stem: _load(p) for p in args.artifact}
    if len(docs) != len(args.artifact):
        raise SystemExit("FATAL: two --artifact paths share a stem; one would shadow the other.")
    check_poolable(docs)

    control = control_verdict(args.control)
    arms = {arm: pooled_arm(docs, arm) for arm in ARMS}

    divergence = None
    if args.divergence is not None:
        divergence = json.loads(args.divergence.read_text())

    # V17 §4, read in order. The first that applies is the outcome.
    if not control["fired"]:
        outcome, why = "V", (
            "VOID. V17 §5's positive control did not fire, so low_iou_runs has still never been "
            "observed to report a drift that is present. §4's remaining outcomes are NOT "
            "evaluated, and the pooled numbers below are reported for the record only — nothing "
            "may be concluded from a zero produced by a statistic of unknown sensitivity."
        )
    elif arms["propagation"]["longest_low_iou_run_within_any_capture"] >= RUN_LENGTH_D or (
        divergence is not None and divergence.get("outcome_d_met")
    ):
        outcome, why = "D", (
            f"DRIFT OBSERVED. A propagation-side run of >= {RUN_LENGTH_D} frames appeared. Limb "
            "(b) is confirmed on this evidence, the blocker's second reason is answered in the "
            "direction that KEEPS IT OPEN, and GATE_QUALIFICATION_BLOCKERS is not shortened."
        )
    elif divergence is None:
        outcome, why = "INCOMPLETE", (
            "The control fired and Arm A shows no run, but Arm B has not been measured. V17 §4's "
            "outcome N requires BOTH arms — Arm A is MuJoCo and answers 'one trajectory', while "
            "'is not a corpus' is exactly what only Arm B addresses. Pass --divergence."
        )
    else:
        outcome, why = "N", (
            "NOT OBSERVED, RATE BOUNDED. The control fired and neither arm shows a run of "
            f">= {RUN_LENGTH_D}. Under V17 §4 this is the outcome that discharges the blocker's "
            "second reason — AS A BOUND AND NOT AS AN ABSENCE, carrying the four items §4 requires "
            "in the discharge text itself."
        )

    record = {
        "schema": SCHEMA,
        "rule": RULE,
        "writeup": WRITEUP,
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "outcome": outcome,
        "outcome_reading": why,
        "positive_control": control,
        "arms": arms,
        "delta_pooled_px": (
            None
            if arms["per_frame"]["pooled_est_drift_p95_px"] is None
            or arms["propagation"]["pooled_est_drift_p95_px"] is None
            else arms["propagation"]["pooled_est_drift_p95_px"]
            - arms["per_frame"]["pooled_est_drift_p95_px"]
        ),
        "arm_b_divergence": (
            None
            if divergence is None
            else {
                "artifact": str(args.divergence),
                "n_episodes": divergence.get("n_episodes"),
                "n_frames": divergence.get("n_frames"),
                "n_divergence_runs": divergence.get("n_divergence_runs"),
                "longest_divergence_run": divergence.get("longest_divergence_run"),
                "outcome_d_met": divergence.get("outcome_d_met"),
                "rule_of_three_reading": divergence.get("rule_of_three_reading"),
                "no_ground_truth": divergence.get("no_ground_truth"),
            }
        ),
        "discharges": (
            "NOTHING BY ITSELF. Editing GATE_QUALIFICATION_BLOCKERS is a reviewable act a person "
            "makes; producing the evidence a blocker asks for is a different act from accepting "
            "it. GATE_QUALIFIED is untouched by the run that wrote this file, and has a second, "
            "independent precondition — the recorded decision on residue (i), the 92 frames — "
            "that nothing here touches."
        ),
        "simulator_caveat": (
            "Arm A is MuJoCo: an untextured 14-group convex proxy mesh, a static prop, a "
            "rasteriser that is neither ray-traced nor photoreal, with a cube distractor in the "
            "scene. T40_RULE_V14 licenses the substitution for EST_DRIFT_P95 and the arm "
            "comparison and for nothing else."
        ),
        "artifacts_pooled": sorted(str(p) for p in args.artifact),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, indent=2) + "\n").encode("utf-8")
    args.out.write_bytes(payload)
    (args.out.parent / (args.out.name + ".sha256")).write_text(
        hashlib.sha256(payload).hexdigest() + "\n", encoding="utf-8"
    )
    print(f"outcome {outcome}: {why}")
    for arm in ARMS:
        a = arms[arm]
        print(
            f"  {arm:>11}: pooled p95 {a['pooled_est_drift_p95_px']} px over {a['n_measured']} "
            f"measured frames of {a['n_frames']}, {a['n_low_iou_runs_summed']} runs, longest "
            f"{a['longest_low_iou_run_within_any_capture']}"
        )
    print(f"  control fired: {control['fired']}  ->  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
