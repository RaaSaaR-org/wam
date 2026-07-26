"""T-08 tests: sync sources, tolerance handling, mock capture session end-to-end."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wam.data import (
    EpisodeReader,
    EpisodeWriter,
    FrameSource,
    MockCameraSource,
    MockCaptureSession,
    RobotStateSource,
    StateSource,
    SyncRecorder,
    SyncToleranceError,
)
from wam.interfaces import ActionChunk, ActionMode, CanonicalSpaceSpec, Observation
from wam.robot import MockRobot
from wam.runtime.mock_loop import DummyPolicy
from wam.safety import SafetyConfig, SafetyLayer

N_JOINTS = 4
DT_S = 0.05
FPS = 5.0
MS = 1_000_000  # ns per ms


def _safety(spec: CanonicalSpaceSpec) -> SafetyLayer:
    n = spec.num_joints
    config = SafetyConfig(
        q_min=(-3.0,) * n,
        q_max=(3.0,) * n,
        dq_max=(1.5,) * n,
        ddq_max=(4.0,) * n,
        workspace_min=(0.1, -0.6, 0.6),
        workspace_max=(0.8, 0.6, 1.4),
        ee_max_lin_vel_m_s=0.5,
        ee_max_step_m=0.05,
        gripper_rate_max=2.0,
        chunk_timeout_s=0.5,
    )
    return SafetyLayer(config, spec=spec)


def _zero_chunk(num_steps: int = 2) -> ActionChunk:
    return ActionChunk(
        mode=ActionMode.JOINT_DELTA,
        targets=np.zeros((num_steps, N_JOINTS), dtype=np.float32),
        gripper_target=np.zeros(num_steps, dtype=np.float32),
        dt_s=DT_S,
    )


class NanPolicy:
    """Policy emitting NaN targets; the safety layer must replace them with a HOLD chunk."""

    def predict(self, observation: Observation) -> ActionChunk:
        targets = np.full((4, N_JOINTS), np.nan, dtype=np.float32)
        return ActionChunk(
            mode=ActionMode.JOINT_DELTA,
            targets=targets,
            gripper_target=np.zeros(4, dtype=np.float32),
            dt_s=DT_S,
        )


# -- sources ---------------------------------------------------------------------------------


def test_sources_satisfy_protocols() -> None:
    robot = MockRobot(num_joints=N_JOINTS, seed=0)
    camera = MockCameraSource(robot, "front")
    state_source = RobotStateSource(robot)
    assert isinstance(camera, FrameSource)
    assert isinstance(state_source, StateSource)


def test_mock_camera_source_offset_and_shape() -> None:
    robot = MockRobot(num_joints=N_JOINTS, seed=0)
    source = MockCameraSource(robot, "wrist", offset_ns=7 * MS)
    img, ts = source.capture()
    assert img.dtype == np.uint8 and img.shape == (64, 64, 3)
    assert ts == robot.sim_time_ns + 7 * MS
    with pytest.raises(KeyError):
        MockCameraSource(robot, "nope").capture()


# -- SyncRecorder ----------------------------------------------------------------------------


def test_sync_recorder_records_aligned(tmp_path: Path) -> None:
    robot = MockRobot(num_joints=N_JOINTS, seed=0)
    writer = EpisodeWriter(tmp_path / "ep", "ep-sync", robot.spec, FPS, "test")
    sources = [
        MockCameraSource(robot, "front"),
        MockCameraSource(robot, "wrist", offset_ns=1 * MS),
    ]
    recorder = SyncRecorder(writer, sources, RobotStateSource(robot), 5 * MS)
    for _ in range(3):
        sample = recorder.step()
        assert sample.within_tolerance
        assert sample.sync_error_ns == 1 * MS
        chunk = _zero_chunk()
        recorder.add_action(chunk, 2, sample.state.timestamp_ns)
        robot.execute(chunk, 2)
    manifest = writer.close()

    assert recorder.num_samples == 3
    assert recorder.num_chunks == 3
    assert recorder.num_violations == 0
    assert recorder.max_sync_error_ns == 1 * MS
    assert manifest.max_sync_error_ns == 1 * MS  # camera-camera spread

    reader = EpisodeReader(tmp_path / "ep")
    assert len(reader.read_states()) == 3
    assert len(reader.read_actions()) == 3
    assert reader.read_frames("front").shape[0] == 3
    np.testing.assert_array_equal(
        reader.frame_timestamps("wrist"), reader.frame_timestamps("front") + 1 * MS
    )


def test_sync_tolerance_violation_raises_before_writing(tmp_path: Path) -> None:
    robot = MockRobot(num_joints=N_JOINTS, seed=0)
    writer = EpisodeWriter(tmp_path / "ep", "ep-viol", robot.spec, FPS, "test")
    sources = [
        MockCameraSource(robot, "front"),
        MockCameraSource(robot, "wrist", offset_ns=10 * MS),
    ]
    recorder = SyncRecorder(writer, sources, RobotStateSource(robot), 1 * MS)
    with pytest.raises(SyncToleranceError):
        recorder.step()
    assert recorder.num_samples == 0
    assert recorder.num_violations == 1
    assert recorder.max_sync_error_ns == 10 * MS
    manifest = writer.close()  # nothing was written before the raise
    assert manifest.tables["states"].num_rows == 0
    assert manifest.cameras == {}


def test_sync_tolerance_violation_flagged_still_records(tmp_path: Path) -> None:
    robot = MockRobot(num_joints=N_JOINTS, seed=0)
    writer = EpisodeWriter(tmp_path / "ep", "ep-flag", robot.spec, FPS, "test")
    sources = [MockCameraSource(robot, "front", offset_ns=10 * MS)]
    recorder = SyncRecorder(
        writer, sources, RobotStateSource(robot), 1 * MS, on_violation="flag"
    )
    sample = recorder.step()
    assert not sample.within_tolerance
    assert sample.sync_error_ns == 10 * MS
    assert recorder.num_violations == 1
    assert recorder.num_samples == 1
    manifest = writer.close()
    assert manifest.tables["states"].num_rows == 1
    assert manifest.cameras["front"].num_frames == 1


def test_sync_recorder_rejects_bad_args(tmp_path: Path) -> None:
    robot = MockRobot(num_joints=N_JOINTS, seed=0)
    writer = EpisodeWriter(tmp_path / "ep", "ep-args", robot.spec, FPS, "test")
    state_source = RobotStateSource(robot)
    duplicate = [MockCameraSource(robot, "front"), MockCameraSource(robot, "front")]
    with pytest.raises(ValueError):
        SyncRecorder(writer, duplicate, state_source, 1 * MS)
    with pytest.raises(ValueError):
        SyncRecorder(writer, [], state_source, -1)
    with pytest.raises(ValueError):
        SyncRecorder(writer, [], state_source, 1 * MS, on_violation="ignore")  # type: ignore[arg-type]
    writer.close()


# -- MockCaptureSession ----------------------------------------------------------------------


def _session(robot: MockRobot, policy: object | None = None, **kwargs: object) -> MockCaptureSession:
    return MockCaptureSession(
        robot,
        policy or DummyPolicy(robot.spec, steps=8, dt_s=DT_S),
        _safety(robot.spec),
        fps=FPS,
        sync_tolerance_ns=20 * MS,
        camera_offsets_ns={"wrist": 2 * MS},
        **kwargs,  # type: ignore[arg-type]
    )


def test_capture_session_end_to_end(tmp_path: Path) -> None:
    robot = MockRobot(num_joints=N_JOINTS, seed=0)
    result = _session(robot).record_episode(
        tmp_path / "ep", "ep-cap-0001", iterations=6, prefix_steps=3, extra={"d_phase": "D0"}
    )
    assert result.iterations == 6
    assert result.interventions_total == 0
    assert result.sync_violations == 0
    assert result.max_sync_error_ns == 2 * MS
    assert result.manifest.extra["d_phase"] == "D0"

    reader = EpisodeReader(tmp_path / "ep")
    states = reader.read_states()
    actions = reader.read_actions()
    assert len(states) == 6
    assert len(actions) == 6
    ts = np.asarray([s.timestamp_ns for s in states])
    assert (np.diff(ts) > 0).all()  # robot sim clock advanced every iteration
    for chunk, executed_prefix, _ in actions:
        assert executed_prefix == 3
        assert chunk.num_steps == 8
        assert np.isfinite(chunk.targets).all()
    for camera in ("front", "wrist"):
        assert reader.read_frames(camera).shape == (6, 64, 64, 3)


def test_capture_session_deterministic(tmp_path: Path) -> None:
    def record(dir: Path) -> tuple[list, list]:
        robot = MockRobot(num_joints=N_JOINTS, seed=0)
        _session(robot).record_episode(dir, "ep-det", iterations=4, prefix_steps=2)
        reader = EpisodeReader(dir)
        return reader.read_states(), reader.read_actions()

    states_a, actions_a = record(tmp_path / "a")
    states_b, actions_b = record(tmp_path / "b")
    for sa, sb in zip(states_a, states_b):
        assert sa.timestamp_ns == sb.timestamp_ns
        np.testing.assert_array_equal(sa.q, sb.q)
        np.testing.assert_array_equal(sa.dq, sb.dq)
    for (ca, pa, ta), (cb, pb, tb) in zip(actions_a, actions_b):
        assert (pa, ta) == (pb, tb)
        np.testing.assert_array_equal(ca.targets, cb.targets)


def test_capture_session_records_safety_filtered_chunk(tmp_path: Path) -> None:
    robot = MockRobot(num_joints=N_JOINTS, seed=0)
    result = _session(robot, policy=NanPolicy()).record_episode(
        tmp_path / "ep", "ep-nan", iterations=3, prefix_steps=2
    )
    assert result.intervention_kinds.get("nan_reject", 0) >= 3
    reader = EpisodeReader(tmp_path / "ep")
    for chunk, executed_prefix, _ in reader.read_actions():
        assert np.isfinite(chunk.targets).all()  # the recorded chunk is the SAFE one
        assert executed_prefix == min(2, chunk.num_steps)


def test_capture_session_rejects_bad_args(tmp_path: Path) -> None:
    robot = MockRobot(num_joints=N_JOINTS, seed=0)
    with pytest.raises(ValueError):
        MockCaptureSession(
            robot,
            DummyPolicy(robot.spec),
            _safety(robot.spec),
            camera_offsets_ns={"nope": 1},
        )
    session = _session(robot)
    with pytest.raises(ValueError):
        session.record_episode(tmp_path / "ep", "ep", iterations=0, prefix_steps=1)
    with pytest.raises(ValueError):
        session.record_episode(tmp_path / "ep", "ep", iterations=1, prefix_steps=0)
