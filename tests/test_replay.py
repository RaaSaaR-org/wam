"""T-09/T-10 tests: replay ordering, hand-checked episode report, calibration roundtrip."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from wam.data import (
    CalibrationSet,
    CameraExtrinsics,
    CameraIntrinsics,
    EpisodeReader,
    EpisodeWriter,
    episode_report,
    replay_episode,
)
from wam.interfaces.schema import (
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    IMUState,
    RobotState,
)
from wam.interfaces.versioning import load_config

_REPO_ROOT = Path(__file__).resolve().parent.parent

FPS = 20.0
NS = 1_000_000_000
PERIOD_NS = 50_000_000  # nominal frame period at 20 fps
WRIST_OFFSET_NS = 2_000_000

SPEC = CanonicalSpaceSpec(joint_names=("j0", "j1"), gripper_dims=1)

# Hand-picked values -> hand-computed expectations (see test bodies).
STATE_ROWS = [
    # (t_ns, q, dq)
    (0, (0.0, 1.0), (0.0, -0.5)),
    (PERIOD_NS, (0.2, 0.8), (0.5, 0.5)),
    (2 * PERIOD_NS, (0.4, 0.6), (-1.0, 0.0)),
]
CHUNK_ROWS = [
    # (t_ns, prefix, targets, gripper): norms 0.5/0.0 and 1.0/0.5
    (0, 1, ((0.3, 0.4), (0.0, 0.0)), (0.0, 1.0)),
    (PERIOD_NS, 1, ((0.6, 0.8), (0.3, 0.4)), (1.0, 1.0)),
]


def _state(t_ns: int, q: tuple[float, ...], dq: tuple[float, ...]) -> RobotState:
    return RobotState(
        timestamp_ns=t_ns,
        q=np.asarray(q, dtype=np.float32),
        dq=np.asarray(dq, dtype=np.float32),
        imu=IMUState(
            orientation_wxyz=np.asarray([1, 0, 0, 0], dtype=np.float32),
            angular_velocity=np.zeros(3, dtype=np.float32),
            linear_acceleration=np.zeros(3, dtype=np.float32),
        ),
        gripper_state=np.asarray([0.5], dtype=np.float32),
    )


def _chunk(targets: tuple[tuple[float, ...], ...], gripper: tuple[float, ...]) -> ActionChunk:
    return ActionChunk(
        mode=ActionMode.JOINT_DELTA,
        targets=np.asarray(targets, dtype=np.float32),
        gripper_target=np.asarray(gripper, dtype=np.float32),
        dt_s=0.05,
    )


def _frame(value: int) -> np.ndarray:
    return np.full((16, 16, 3), value, dtype=np.uint8)


def _record(dir: Path) -> dict:
    states = [_state(*row) for row in STATE_ROWS]
    chunks = [(t, p, _chunk(tg, g)) for t, p, tg, g in CHUNK_ROWS]
    frame_values = {"front": (20, 100, 200), "wrist": (40, 120, 220)}
    with EpisodeWriter(dir, "ep-tiny", SPEC, FPS, "tiny synthetic episode") as writer:
        for i in range(3):
            writer.add_state(states[i])
            writer.add_frame("front", _frame(frame_values["front"][i]), i * PERIOD_NS)
            writer.add_frame(
                "wrist", _frame(frame_values["wrist"][i]), i * PERIOD_NS + WRIST_OFFSET_NS
            )
        for t, prefix, chunk in chunks:
            writer.add_action(chunk, prefix, t)
    return {"states": states, "chunks": chunks, "frame_values": frame_values}


@pytest.fixture(scope="module")
def episode(tmp_path_factory: pytest.TempPathFactory) -> dict:
    dir = tmp_path_factory.mktemp("replay") / "ep-tiny"
    truth = _record(dir)
    truth["dir"] = dir
    return truth


# -- replay ordering -------------------------------------------------------------------------


def test_replay_merges_and_orders_by_timestamp(episode: dict) -> None:
    steps = list(replay_episode(EpisodeReader(episode["dir"])))
    assert [s.t_ns for s in steps] == [
        0,
        WRIST_OFFSET_NS,
        PERIOD_NS,
        PERIOD_NS + WRIST_OFFSET_NS,
        2 * PERIOD_NS,
        2 * PERIOD_NS + WRIST_OFFSET_NS,
    ]
    # t=0: state + front frame + chunk merged into ONE step
    s0 = steps[0]
    assert s0.state is not None and s0.state.timestamp_ns == 0
    assert np.array_equal(s0.state.q, episode["states"][0].q)
    assert s0.frames is not None and set(s0.frames) == {"front"}
    assert s0.frames["front"].shape == (16, 16, 3)
    assert s0.commanded is not None and s0.executed_prefix == 1
    assert np.array_equal(s0.commanded.targets, episode["chunks"][0][2].targets)
    # wrist-only steps carry nothing else
    for idx in (1, 3, 5):
        w = steps[idx]
        assert set(w.frames or {}) == {"wrist"}
        assert w.state is None and w.commanded is None and w.executed_prefix is None
    # last state step has no chunk (only 2 chunks recorded)
    s4 = steps[4]
    assert s4.state is not None and s4.commanded is None and s4.executed_prefix is None
    # states arrive in recording order
    got_states = [s.state.timestamp_ns for s in steps if s.state is not None]
    assert got_states == [0, PERIOD_NS, 2 * PERIOD_NS]
    # lossy codec: frame content close to the recorded constant value
    for idx, cam, val_idx in ((0, "front", 0), (1, "wrist", 0), (2, "front", 1)):
        val = episode["frame_values"][cam][val_idx]
        assert abs(float(steps[idx].frames[cam].mean()) - val) < 10.0


def test_replay_same_timestamp_collisions_split_steps(tmp_path: Path) -> None:
    with EpisodeWriter(tmp_path / "dup", "ep-dup", SPEC, FPS, "") as writer:
        writer.add_state(_state(5, (0.0, 0.0), (0.0, 0.0)))
        writer.add_state(_state(5, (1.0, 1.0), (0.0, 0.0)))
        chunk = _chunk(((0.1, 0.1),), (0.5,))
        writer.add_action(chunk, 1, 5)
        writer.add_action(chunk, 0, 5)
    steps = list(replay_episode(EpisodeReader(tmp_path / "dup")))
    assert [s.t_ns for s in steps] == [5, 5, 5]
    assert steps[0].state is not None and float(steps[0].state.q[0]) == 0.0
    assert steps[0].commanded is None
    # second state opens a new step; first chunk merges into it
    assert steps[1].state is not None and float(steps[1].state.q[0]) == 1.0
    assert steps[1].commanded is not None and steps[1].executed_prefix == 1
    # second chunk collides -> third step, no state
    assert steps[2].state is None
    assert steps[2].commanded is not None and steps[2].executed_prefix == 0


# -- episode report: hand-computed values ----------------------------------------------------


def test_report_header_and_duration(episode: dict) -> None:
    r = episode_report(EpisodeReader(episode["dir"]))
    assert r.episode_id == "ep-tiny"
    assert r.instruction == "tiny synthetic episode"
    assert (r.t0_ns, r.t1_ns) == (0, 2 * PERIOD_NS + WRIST_OFFSET_NS)
    assert r.duration_s == pytest.approx(0.102)
    assert r.num_states == 3
    assert r.max_sync_error_ns == WRIST_OFFSET_NS
    # default tolerance: half the nominal frame period of the fastest camera
    assert r.sync_tolerance_ns == PERIOD_NS // 2
    assert r.flags == ()


def test_report_camera_stats(episode: dict) -> None:
    r = episode_report(EpisodeReader(episode["dir"]))
    assert set(r.cameras) == {"front", "wrist"}
    for cam in ("front", "wrist"):
        c = r.cameras[cam]
        assert c.num_frames == 3
        assert (c.width, c.height) == (16, 16)
        assert c.fps_nominal == FPS
        # 3 frames spanning exactly 2 nominal periods
        assert c.duration_s == pytest.approx(0.1)
        assert c.mean_dt_ns == pytest.approx(PERIOD_NS)
        assert c.fps_actual == pytest.approx(20.0)


def test_report_joint_coverage_hand_computed(episode: dict) -> None:
    r = episode_report(EpisodeReader(episode["dir"]))
    assert [j.name for j in r.joints] == ["j0", "j1"]
    j0, j1 = r.joints
    # j0: q in {0.0, 0.2, 0.4}; dq in {0.0, 0.5, -1.0}
    assert j0.q_min == pytest.approx(0.0)
    assert j0.q_max == pytest.approx(0.4, rel=1e-6)
    assert j0.q_mean == pytest.approx(0.2, rel=1e-6)
    assert j0.dq_abs_max == pytest.approx(1.0)
    assert j0.dq_abs_mean == pytest.approx(0.5)
    # j1: q in {1.0, 0.8, 0.6}; dq in {-0.5, 0.5, 0.0}
    assert j1.q_min == pytest.approx(0.6, rel=1e-6)
    assert j1.q_max == pytest.approx(1.0)
    assert j1.q_mean == pytest.approx(0.8, rel=1e-6)
    assert j1.dq_abs_max == pytest.approx(0.5)
    assert j1.dq_abs_mean == pytest.approx(1.0 / 3.0)


def test_report_action_stats_hand_computed(episode: dict) -> None:
    a = episode_report(EpisodeReader(episode["dir"])).actions
    assert a.num_chunks == 2
    assert a.num_steps == 4
    assert a.executed_steps == 2  # prefix 1 + 1
    assert a.executed_ratio == pytest.approx(0.5)
    assert a.modes == {"joint_delta": 2}
    # per-step L2 norms: 0.5, 0.0, 1.0, 0.5
    assert a.step_norm_mean == pytest.approx(0.5, rel=1e-6)
    assert a.step_norm_max == pytest.approx(1.0, rel=1e-6)
    assert (a.gripper_min, a.gripper_max) == (0.0, 1.0)
    # within-chunk deltas: |1-0| = 1.0, |1-1| = 0.0
    assert a.gripper_mean_abs_delta == pytest.approx(0.5)
    assert a.gripper_active_fraction == pytest.approx(0.5)


def test_report_markdown_and_flags(episode: dict) -> None:
    r = episode_report(EpisodeReader(episode["dir"]))
    md = r.render_markdown()
    assert md.startswith("# Episode report — ep-tiny")
    assert "| front | 3 | 16x16 | 20 | 20 | 50 |" in md
    assert "| j0 |" in md and "| j1 |" in md
    assert "- none" in md  # no flags
    # tightened tolerance triggers the sync flag
    strict = episode_report(EpisodeReader(episode["dir"]), sync_tolerance_ns=1)
    assert "sync_error_exceeds_tolerance" in strict.flags
    assert "- sync_error_exceeds_tolerance" in strict.render_markdown()


def test_report_empty_episode(tmp_path: Path) -> None:
    with EpisodeWriter(tmp_path / "empty", "ep-empty", SPEC, FPS, ""):
        pass
    r = episode_report(EpisodeReader(tmp_path / "empty"))
    assert r.flags == ("no_states", "no_actions", "no_cameras")
    assert r.duration_s == 0.0
    assert r.joints == () and r.cameras == {}
    assert r.sync_tolerance_ns is None
    assert r.actions.num_chunks == 0 and r.actions.executed_ratio == 0.0
    assert "(no states)" in r.render_markdown()
    assert list(replay_episode(EpisodeReader(tmp_path / "empty"))) == []


def test_report_flags_discarded_and_validity(tmp_path: Path) -> None:
    state = _state(0, (0.0, 0.0), (0.0, 0.0))
    state.validity.imu = False
    with EpisodeWriter(tmp_path / "flag", "ep-flag", SPEC, FPS, "") as writer:
        writer.add_state(state)
        writer.add_action(_chunk(((0.1, 0.1),), (0.5,)), 0, 0)  # prefix 0 = discarded
    r = episode_report(EpisodeReader(tmp_path / "flag"))
    assert "validity_gap:imu" in r.flags
    assert "discarded_chunks" in r.flags
    assert "invalid_states" not in r.flags  # invalid groups are masked, not broken


# -- CLI -------------------------------------------------------------------------------------


def _load_cli():
    path = _REPO_ROOT / "scripts" / "episode_report.py"
    spec = importlib.util.spec_from_file_location("episode_report_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_markdown_and_json(episode: dict, capsys: pytest.CaptureFixture[str]) -> None:
    cli = _load_cli()
    assert cli.main([str(episode["dir"])]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Episode report — ep-tiny")
    assert cli.main([str(episode["dir"]), "--json", "--no-verify"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["episode_id"] == "ep-tiny"
    assert payload["actions"]["num_chunks"] == 2
    assert payload["flags"] == []


# -- calibration (T-10) ----------------------------------------------------------------------


def _calibration() -> CalibrationSet:
    return CalibrationSet(
        robot="mock",
        calibrated_at="2026-07-26",
        method="unit-test",
        intrinsics={
            "front": CameraIntrinsics(
                width=64, height=64, fx=50.0, fy=51.0, cx=32.0, cy=31.5,
                distortion_model="radtan", distortion=(0.01, -0.02, 0.0, 0.0),
            )
        },
        extrinsics={
            "front": CameraExtrinsics(
                translation_m=(0.1, -0.2, 0.3), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)
            )
        },
        joint_offsets_rad={"j0": 0.001, "j1": -0.002},
        extra={"reprojection_error_px": 0.5},
    )


def test_calibration_yaml_roundtrip(tmp_path: Path) -> None:
    calib = _calibration()
    path = tmp_path / "calib.yaml"
    calib.to_yaml(path)
    back = CalibrationSet.from_yaml(path)
    assert back == calib
    assert back.config_hash() == calib.config_hash()
    assert len(calib.config_hash()) == 64 and set(calib.config_hash()) <= set("0123456789abcdef")
    # the written file passes the generic wam config gate too
    assert load_config(path)["calibration_version"] == calib.calibration_version
    # content change -> different hash
    other = calib.model_copy(update={"robot": "g1"})
    assert other.config_hash() != calib.config_hash()


def test_calibration_example_yaml_loads() -> None:
    path = _REPO_ROOT / "configs" / "calibration" / "example.yaml"
    calib = CalibrationSet.from_yaml(path)
    assert calib.robot == "g1"
    assert calib.cameras() == ("front", "wrist")
    assert calib.intrinsics["front"].distortion_model == "radtan"
    assert calib.extrinsics["wrist"].parent_frame == "right_wrist_yaw"
    # Joint-offset keys must use the canonical G1_SPEC joint names (configs/robot/g1.yaml).
    from wam.robot.g1 import G1_SPEC

    assert set(calib.joint_offsets_rad) <= set(G1_SPEC.joint_names)
    assert calib.joint_offsets_rad["right_shoulder_pitch"] == pytest.approx(0.0021)
    assert len(calib.config_hash()) == 64


def test_calibration_matrices() -> None:
    calib = _calibration()
    k = calib.intrinsics["front"].matrix()
    assert k.shape == (3, 3) and k.dtype == np.float64
    assert k[0, 0] == 50.0 and k[1, 1] == 51.0 and k[0, 2] == 32.0 and k[2, 2] == 1.0
    t = calib.extrinsics["front"].matrix()
    assert t.shape == (4, 4)
    assert np.allclose(t[:3, :3], np.eye(3))  # identity quaternion
    assert np.allclose(t[:3, 3], [0.1, -0.2, 0.3]) and t[3, 3] == 1.0
    # 90 deg about z: proper rotation, x-axis -> y-axis
    half = math.sqrt(0.5)
    rot = CameraExtrinsics(
        translation_m=(0.0, 0.0, 0.0), rotation_wxyz=(half, 0.0, 0.0, half)
    ).rotation_matrix()
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rot) == pytest.approx(1.0)
    assert np.allclose(rot @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_calibration_validation_errors() -> None:
    with pytest.raises(ValidationError, match="unit quaternion"):
        CameraExtrinsics(translation_m=(0, 0, 0), rotation_wxyz=(0.5, 0.5, 0.0, 0.0))
    with pytest.raises(ValidationError):
        CameraIntrinsics(width=64, height=64, fx=0.0, fy=50.0, cx=32.0, cy=32.0)
    with pytest.raises(ValidationError, match="finite"):
        CalibrationSet(joint_offsets_rad={"j0": float("nan")})
    with pytest.raises(ValidationError, match="calibration_version"):
        CalibrationSet(calibration_version="2.0.0")


def test_calibration_from_yaml_version_gate(tmp_path: Path) -> None:
    calib = _calibration()
    path = tmp_path / "calib.yaml"
    calib.to_yaml(path)
    import yaml

    data = yaml.safe_load(path.read_text())
    data["wam_config_version"] = "9.0.0"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="wam_config_version"):
        CalibrationSet.from_yaml(bad)
