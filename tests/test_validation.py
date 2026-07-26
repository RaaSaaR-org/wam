"""T-11 tests: episode/dataset validation gates + record_mock_dataset CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from wam.data import (
    EpisodeWriter,
    MockCaptureSession,
    ValidationThresholds,
    validate_dataset,
    validate_episode,
)
from wam.data.validation import (
    EPISODE_GATES,
    GATE_CHECKSUMS,
    GATE_COUNTS,
    GATE_DURATION,
    GATE_EPISODE_COUNT,
    GATE_FINITE,
    GATE_READABLE,
    GATE_SYNC,
)
from wam.interfaces import (
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    IMUState,
    RobotState,
)
from wam.robot import MockRobot
from wam.runtime.mock_loop import DummyPolicy
from wam.safety import SafetyConfig, SafetyLayer

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "record_mock_dataset.py"

N_JOINTS = 4
DT_S = 0.05
MS = 1_000_000

THRESHOLDS = ValidationThresholds(sync_tolerance_ns=20 * MS)


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


def _record_clean(
    dir: Path,
    episode_id: str,
    *,
    seed: int = 0,
    iterations: int = 8,
    prefix_steps: int = 3,
    wrist_offset_ns: int = 2 * MS,
    sync_tolerance_ns: int = 20 * MS,
    on_violation: str = "raise",
) -> None:
    robot = MockRobot(num_joints=N_JOINTS, seed=seed)
    session = MockCaptureSession(
        robot,
        DummyPolicy(robot.spec, steps=8, dt_s=DT_S),
        _safety(robot.spec),
        fps=1.0 / (DT_S * prefix_steps),
        sync_tolerance_ns=sync_tolerance_ns,
        camera_offsets_ns={"wrist": wrist_offset_ns},
        on_violation=on_violation,  # type: ignore[arg-type]
    )
    session.record_episode(dir, episode_id, iterations=iterations, prefix_steps=prefix_steps)


@pytest.fixture(scope="module")
def clean_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("dataset")
    for i in range(2):
        _record_clean(root / f"ep-{i:04d}", f"ep-{i:04d}", seed=i)
    return root


# -- manual episode helper for targeted gate failures ----------------------------------------


def _state(timestamp_ns: int, *, dq_value: float = 0.0) -> RobotState:
    return RobotState(
        timestamp_ns=timestamp_ns,
        q=np.zeros(N_JOINTS, dtype=np.float32),
        dq=np.full(N_JOINTS, dq_value, dtype=np.float32),
        imu=IMUState(
            orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
            angular_velocity=np.zeros(3, dtype=np.float32),
            linear_acceleration=np.zeros(3, dtype=np.float32),
        ),
        gripper_state=np.zeros(1, dtype=np.float32),
    )


def _write_manual_episode(
    dir: Path, state_dq: list[float], frame_ts: list[int], state_ts: list[int] | None = None
) -> None:
    spec = CanonicalSpaceSpec(joint_names=tuple(f"j{i}" for i in range(N_JOINTS)))
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    with EpisodeWriter(dir, dir.name, spec, 5.0, "manual") as writer:
        ts_list = state_ts if state_ts is not None else frame_ts
        for ts, dq_value in zip(ts_list, state_dq):
            writer.add_state(_state(ts, dq_value=dq_value))
        for ts in frame_ts:
            writer.add_frame("front", img, ts)
        chunk = ActionChunk(
            mode=ActionMode.JOINT_DELTA,
            targets=np.zeros((2, N_JOINTS), dtype=np.float32),
            gripper_target=np.zeros(2, dtype=np.float32),
            dt_s=DT_S,
        )
        writer.add_action(chunk, 2, ts_list[0])


# -- clean data passes -----------------------------------------------------------------------


def test_clean_episode_passes_all_gates(clean_root: Path) -> None:
    report = validate_episode(clean_root / "ep-0000", THRESHOLDS)
    assert report.passed, report.to_json()
    assert tuple(g.name for g in report.gates) == EPISODE_GATES
    assert report.duration_s > 0.5
    assert report.failed_gates() == []


def test_clean_dataset_passes_all_gates(clean_root: Path) -> None:
    thresholds = ValidationThresholds(
        sync_tolerance_ns=20 * MS, min_episodes=2, min_total_duration_s=1.5
    )
    report = validate_dataset(clean_root, thresholds)
    assert report.passed, report.to_json()
    assert len(report.episodes) == 2
    assert all(gate.passed for gate in report.gates)


# -- exactly the right gate fails -------------------------------------------------------------


def test_corrupted_video_fails_checksum_gate_only(clean_root: Path, tmp_path: Path) -> None:
    bad = tmp_path / "ep-bad"
    shutil.copytree(clean_root / "ep-0000", bad)
    video = bad / "front.mp4"
    data = bytearray(video.read_bytes())
    data[len(data) // 2] ^= 0xFF
    video.write_bytes(bytes(data))

    report = validate_episode(bad, THRESHOLDS)
    assert not report.passed
    assert report.failed_gates() == [GATE_CHECKSUMS]
    assert len(report.gates) == 1  # remaining gates are not run on tampered data


def test_sync_violation_fails_sync_gate_only(tmp_path: Path) -> None:
    dir = tmp_path / "ep-skew"
    _record_clean(
        dir, "ep-skew", wrist_offset_ns=50 * MS, sync_tolerance_ns=20 * MS, on_violation="flag"
    )
    report = validate_episode(dir, THRESHOLDS)
    assert not report.passed
    assert report.failed_gates() == [GATE_SYNC]


def test_nan_state_fails_finite_gate_only(tmp_path: Path) -> None:
    dir = tmp_path / "ep-nan"
    _write_manual_episode(dir, state_dq=[0.0, float("nan"), 0.0], frame_ts=[0, 10 * MS, 20 * MS])
    thresholds = ValidationThresholds(sync_tolerance_ns=20 * MS, min_duration_s=0.001)
    report = validate_episode(dir, thresholds)
    assert not report.passed
    assert report.failed_gates() == [GATE_FINITE]


def test_too_few_states_fails_counts_gate_only(tmp_path: Path) -> None:
    dir = tmp_path / "ep-short"
    _write_manual_episode(dir, state_dq=[0.0], frame_ts=[0, 10 * MS], state_ts=[0])
    thresholds = ValidationThresholds(
        sync_tolerance_ns=20 * MS, min_duration_s=0.001, min_states=2
    )
    report = validate_episode(dir, thresholds)
    assert not report.passed
    assert report.failed_gates() == [GATE_COUNTS]


def test_duration_bounds_fail_duration_gate_only(clean_root: Path) -> None:
    thresholds = ValidationThresholds(
        sync_tolerance_ns=20 * MS, min_duration_s=100.0, max_duration_s=200.0
    )
    report = validate_episode(clean_root / "ep-0000", thresholds)
    assert not report.passed
    assert report.failed_gates() == [GATE_DURATION]


def test_unreadable_episode_reports_readable_gate(tmp_path: Path) -> None:
    report = validate_episode(tmp_path / "does-not-exist")
    assert not report.passed
    assert report.failed_gates() == [GATE_READABLE]


# -- dataset-level gates ----------------------------------------------------------------------


def test_dataset_min_episode_count_gate(clean_root: Path) -> None:
    thresholds = ValidationThresholds(sync_tolerance_ns=20 * MS, min_episodes=5)
    report = validate_dataset(clean_root, thresholds)
    assert not report.passed
    assert report.failed_gates() == [GATE_EPISODE_COUNT]
    assert all(episode.passed for episode in report.episodes)


def test_dataset_missing_root(tmp_path: Path) -> None:
    report = validate_dataset(tmp_path / "nope")
    assert not report.passed
    assert report.episodes == []
    assert GATE_EPISODE_COUNT in report.failed_gates()


def test_reports_serialize_to_json(clean_root: Path) -> None:
    episode_report = validate_episode(clean_root / "ep-0000", THRESHOLDS)
    parsed = json.loads(episode_report.to_json())
    assert parsed["passed"] is True
    assert len(parsed["gates"]) == len(EPISODE_GATES)

    dataset_report = validate_dataset(clean_root, THRESHOLDS)
    parsed = json.loads(dataset_report.to_json())
    assert parsed["passed"] is True
    assert len(parsed["episodes"]) == 2
    assert {gate["name"] for gate in parsed["gates"]} >= {GATE_EPISODE_COUNT}


# -- CLI --------------------------------------------------------------------------------------


def test_record_mock_dataset_script(tmp_path: Path) -> None:
    out = tmp_path / "ds"
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--out",
            str(out),
            "--episodes",
            "2",
            "--iterations",
            "8",
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out / "mock-0000" / "manifest.json").is_file()
    assert (out / "mock-0001" / "manifest.json").is_file()
    report = json.loads((out / "validation_report.json").read_text())
    assert report["passed"] is True
    assert len(report["episodes"]) == 2
