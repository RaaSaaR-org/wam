"""T40_RULE_V17 Arm A pooling — the seams that must not be sewn, and the control that gates.

Synthetic artifacts throughout: the pooling arithmetic and every refusal are reachable without a
capture, a GPU or weights, and the one thing that must never happen — a run reported across two
captures — is asserted directly rather than hoped for.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import pool_est_drift_arms as pool  # noqa: E402


def _arm(displacements, *, n_runs=0, longest=0):
    return {
        "arm": "x",
        "measured": True,
        "absent_because": None,
        "est_drift_p95_px": float(np.percentile(displacements, 95)) if displacements else None,
        "n_frames": len(displacements),
        "n_measured": len(displacements),
        "n_dropped": 0,
        "coverage": 1.0,
        "low_iou_runs": {"n_runs": n_runs, "longest_run": longest, "runs": []},
        "displacements_px": list(displacements),
    }


def _artifact(
    tmp_path,
    name,
    *,
    per_frame,
    propagation,
    median_motion=1.3,
    schedule="trajectory",
    contract=None,
    resolution=(480, 640),
    pf_runs=(0, 0),
    pr_runs=(0, 0),
    params=None,
):
    doc = {
        "resolution_hw": list(resolution),
        "object_class": "apple",
        "estimators": {"segmenter_contract": contract or {"prompt": "apple."}},
        "capture": {
            "path": f"runs/{name}",
            "scene_schedule": schedule,
            "scene_schedule_params": params or {"turns": 1.0, "yaw_turns": 1.0, "arm_cycles": 2.0},
            "temporal_coherence": {"median_interframe_motion_px": median_motion},
        },
        "arm_comparison": {
            "low_iou_threshold": 0.5,
            "propagator": {"spec": "estimators.apple_sam2_video"},
            "per_frame": _arm(per_frame, n_runs=pf_runs[0], longest=pf_runs[1]),
            "propagation": _arm(propagation, n_runs=pr_runs[0], longest=pr_runs[1]),
        },
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(doc))
    return path


def _control(tmp_path, *, longest, n_runs=1, median_motion=65.3):
    path = _artifact(
        tmp_path,
        "C1-lattice",
        per_frame=[0.1] * 60,
        propagation=[0.2] * 60,
        median_motion=median_motion,
        schedule="lattice",
        pr_runs=(n_runs, longest),
    )
    return path


# ---------------------------------------------------------------------------------------------
# The pooling arithmetic
# ---------------------------------------------------------------------------------------------


def test_the_pooled_p95_is_one_percentile_over_the_union_not_a_mean_of_percentiles():
    """THE RULE THIS SCRIPT EXISTS FOR. Eight p95s averaged is not the p95 of eight captures, and
    the difference is largest exactly when one capture has the heavy tail — which is the case the
    gate cares about."""
    low = [0.1] * 100
    high = [0.1] * 90 + [9.0] * 10
    docs = {
        "A1": {"arm_comparison": {"per_frame": _arm(low), "propagation": _arm(low)}},
        "A2": {"arm_comparison": {"per_frame": _arm(high), "propagation": _arm(high)}},
    }
    block = pool.pooled_arm(docs, "per_frame")
    union = np.percentile(np.asarray(low + high), 95)
    mean_of_p95s = np.mean([np.percentile(low, 95), np.percentile(high, 95)])
    assert block["pooled_est_drift_p95_px"] == pytest.approx(float(union))
    assert block["pooled_est_drift_p95_px"] != pytest.approx(float(mean_of_p95s))
    assert block["n_measured"] == 200


def test_runs_are_summed_as_counts_and_never_joined_across_captures():
    """A run at the end of one capture and a run at the start of the next are TWO runs. Sewing
    them would report a contiguous drift across a seam that does not exist — the single failure
    a naive concatenation would introduce silently."""
    docs = {
        "A1": {"arm_comparison": {"propagation": _arm([0.1] * 10, n_runs=1, longest=4)}},
        "A2": {"arm_comparison": {"propagation": _arm([0.1] * 10, n_runs=1, longest=6)}},
    }
    block = pool.pooled_arm(docs, "propagation")
    assert block["n_low_iou_runs_summed"] == 2
    assert block["longest_low_iou_run_within_any_capture"] == 6, "6, not 4+6=10"
    assert "TRUE BY CONSTRUCTION" in block["runs_never_span_captures"]


def test_an_artifact_without_raw_displacements_is_refused_rather_than_approximated():
    block = _arm([0.1] * 10)
    del block["displacements_px"]
    docs = {"A1": {"arm_comparison": {"per_frame": block}}}
    with pytest.raises(SystemExit, match="quantise the answer to the bin width"):
        pool.pooled_arm(docs, "per_frame")


def test_an_unmeasured_arm_is_refused_not_treated_as_zero():
    docs = {
        "A1": {
            "arm_comparison": {
                "per_frame": {"measured": False, "absent_because": "--arm did not include it"}
            }
        }
    }
    with pytest.raises(SystemExit, match="is not measured"):
        pool.pooled_arm(docs, "per_frame")


# ---------------------------------------------------------------------------------------------
# What may not be pooled
# ---------------------------------------------------------------------------------------------


def test_two_different_segmenters_are_not_one_population(tmp_path):
    a = _artifact(tmp_path, "A1", per_frame=[0.1], propagation=[0.2])
    b = _artifact(tmp_path, "A2", per_frame=[0.1], propagation=[0.2], contract={"prompt": "pear."})
    with pytest.raises(SystemExit, match="not produced by the same instrument"):
        pool.check_poolable({p.stem: pool._load(p) for p in (a, b)})


def test_two_different_pixel_grids_are_not_one_population(tmp_path):
    a = _artifact(tmp_path, "A1", per_frame=[0.1], propagation=[0.2])
    b = _artifact(tmp_path, "A2", per_frame=[0.1], propagation=[0.2], resolution=(240, 320))
    with pytest.raises(SystemExit, match="not produced by the same instrument"):
        pool.check_poolable({p.stem: pool._load(p) for p in (a, b)})


def test_a_lattice_capture_may_not_be_pooled_into_arm_a(tmp_path):
    """Its object teleports, so a propagated mask crosses a cut on frame 1 and every number after
    it measures the cut. It belongs on --control, where it is read and not pooled."""
    a = _artifact(tmp_path, "A1", per_frame=[0.1], propagation=[0.2])
    b = _artifact(tmp_path, "L", per_frame=[0.1], propagation=[0.2], schedule="lattice")
    with pytest.raises(SystemExit, match="Pass it to --control"):
        pool.check_poolable({p.stem: pool._load(p) for p in (a, b)})


def test_a_capture_over_the_coherence_bound_is_refused_by_name(tmp_path):
    a = _artifact(tmp_path, "A1", per_frame=[0.1], propagation=[0.2])
    b = _artifact(tmp_path, "C2-t80", per_frame=[0.1], propagation=[0.2], median_motion=105.0)
    with pytest.raises(SystemExit, match="C2-t80 measures median_interframe_motion_px"):
        pool.check_poolable({p.stem: pool._load(p) for p in (a, b)})


def test_the_schedules_name_is_not_evidence_that_a_capture_is_coherent(tmp_path):
    """V17 §2 bounds the MEASURED number. An artifact that records no measurement does not pass
    by having the right schedule string."""
    a = _artifact(tmp_path, "A1", per_frame=[0.1], propagation=[0.2])
    doc = json.loads(a.read_text())
    doc["capture"]["temporal_coherence"] = {}
    a.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="NAME is not evidence"):
        pool.check_poolable({"A1": pool._load(a)})


def test_the_coherence_bound_is_the_one_the_existing_test_already_asserts():
    assert pool.COHERENCE_MAX_MEDIAN_PX == 25.0


def test_a_single_arm_artifact_cannot_be_pooled(tmp_path):
    path = tmp_path / "single.json"
    path.write_text(json.dumps({"est_drift_p95_px": 0.29}))
    with pytest.raises(SystemExit, match="carries no arm_comparison block"):
        pool._load(path)


# ---------------------------------------------------------------------------------------------
# The control, and V17 §4's outcome order
# ---------------------------------------------------------------------------------------------


def test_no_control_means_the_pool_is_void_however_good_the_numbers_are(tmp_path):
    """V17 §4 reads VOID first on purpose. A pooled zero from a statistic that has never been seen
    to fire is not evidence of absence."""
    a = _artifact(tmp_path, "A1", per_frame=[0.1] * 480, propagation=[0.2] * 480)
    out = tmp_path / "POOLED.json"
    pool.main(["--artifact", str(a), "--out", str(out)])
    got = json.loads(out.read_text())
    assert got["outcome"] == "V"
    assert "never been observed to fire" in got["positive_control"]["meaning"]


def test_a_control_that_does_not_fire_is_still_void(tmp_path):
    a = _artifact(tmp_path, "A1", per_frame=[0.1] * 480, propagation=[0.2] * 480)
    c = _control(tmp_path, longest=3)
    out = tmp_path / "POOLED.json"
    pool.main(["--artifact", str(a), "--control", str(c), "--out", str(out)])
    got = json.loads(out.read_text())
    assert got["outcome"] == "V"
    assert got["positive_control"]["fired"] is False
    assert got["positive_control"]["longest_run"] == 3


def test_a_fired_control_and_a_long_propagation_run_is_outcome_d(tmp_path):
    a = _artifact(
        tmp_path, "A1", per_frame=[0.1] * 480, propagation=[0.2] * 480, pr_runs=(1, 14)
    )
    c = _control(tmp_path, longest=41)
    out = tmp_path / "POOLED.json"
    pool.main(["--artifact", str(a), "--control", str(c), "--out", str(out)])
    got = json.loads(out.read_text())
    assert got["outcome"] == "D"
    assert "KEEPS IT OPEN" in got["outcome_reading"]


def test_a_clean_arm_a_without_arm_b_is_incomplete_and_not_outcome_n(tmp_path):
    """THE GUARD AGAINST THE EASY WIN. Arm A is MuJoCo and answers 'one trajectory'. 'Is not a
    corpus' is what only Arm B addresses, and outcome N requires both."""
    a = _artifact(tmp_path, "A1", per_frame=[0.1] * 480, propagation=[0.2] * 480)
    c = _control(tmp_path, longest=41)
    out = tmp_path / "POOLED.json"
    pool.main(["--artifact", str(a), "--control", str(c), "--out", str(out)])
    got = json.loads(out.read_text())
    assert got["outcome"] == "INCOMPLETE"
    assert "only Arm B addresses" in got["outcome_reading"]


def test_outcome_n_needs_a_fired_control_a_clean_arm_a_and_a_clean_arm_b(tmp_path):
    a = _artifact(tmp_path, "A1", per_frame=[0.1] * 480, propagation=[0.2] * 480)
    c = _control(tmp_path, longest=41)
    div = tmp_path / "DIV.json"
    div.write_text(json.dumps({
        "n_episodes": 40, "n_frames": 16846, "n_divergence_runs": 3,
        "longest_divergence_run": 2, "outcome_d_met": False,
        "rule_of_three_reading": "CAN DETECT ... CANNOT CERTIFY",
        "no_ground_truth": "agreement, not correctness",
    }))
    out = tmp_path / "POOLED.json"
    pool.main(["--artifact", str(a), "--control", str(c), "--divergence", str(div), "--out", str(out)])
    got = json.loads(out.read_text())
    assert got["outcome"] == "N"
    assert "AS A BOUND AND NOT AS AN ABSENCE" in got["outcome_reading"]
    assert got["arm_b_divergence"]["n_frames"] == 16846


def test_arm_b_alone_can_carry_outcome_d_even_when_arm_a_is_clean(tmp_path):
    """The corpus is the thing the blocker names. A divergence run there is the observation,
    whether or not the simulator reproduced it."""
    a = _artifact(tmp_path, "A1", per_frame=[0.1] * 480, propagation=[0.2] * 480)
    c = _control(tmp_path, longest=41)
    div = tmp_path / "DIV.json"
    div.write_text(json.dumps({"outcome_d_met": True, "longest_divergence_run": 61}))
    out = tmp_path / "POOLED.json"
    pool.main(["--artifact", str(a), "--control", str(c), "--divergence", str(div), "--out", str(out)])
    assert json.loads(out.read_text())["outcome"] == "D"


def test_no_outcome_of_this_script_discharges_anything(tmp_path):
    a = _artifact(tmp_path, "A1", per_frame=[0.1] * 480, propagation=[0.2] * 480)
    c = _control(tmp_path, longest=41)
    div = tmp_path / "DIV.json"
    div.write_text(json.dumps({"outcome_d_met": False}))
    out = tmp_path / "POOLED.json"
    pool.main(["--artifact", str(a), "--control", str(c), "--divergence", str(div), "--out", str(out)])
    got = json.loads(out.read_text())
    assert got["discharges"].startswith("NOTHING BY ITSELF")
    assert "residue (i)" in got["discharges"]
    assert "MuJoCo" in got["simulator_caveat"]


def test_the_artifact_carries_a_sha256_sidecar(tmp_path):
    a = _artifact(tmp_path, "A1", per_frame=[0.1] * 10, propagation=[0.2] * 10)
    out = tmp_path / "POOLED.json"
    pool.main(["--artifact", str(a), "--out", str(out)])
    side = tmp_path / "POOLED.json.sha256"
    assert side.is_file()
    import hashlib

    assert side.read_text().strip() == hashlib.sha256(out.read_bytes()).hexdigest()


def test_two_artifacts_with_the_same_stem_would_shadow_each_other(tmp_path):
    a = _artifact(tmp_path, "A1", per_frame=[0.1], propagation=[0.2])
    sub = tmp_path / "other"
    sub.mkdir()
    b = _artifact(sub, "A1", per_frame=[0.1], propagation=[0.2])
    with pytest.raises(SystemExit, match="share a stem"):
        pool.main(["--artifact", str(a), str(b), "--out", str(tmp_path / "o.json")])
