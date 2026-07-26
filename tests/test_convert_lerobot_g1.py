"""Tests for scripts/convert_lerobot_g1.py (GR00T G1 LeRobot -> WAM episode format)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wam.data import EpisodeReader, ValidationThresholds, validate_dataset
from wam.interfaces.schema import ActionMode

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_converter():
    spec = importlib.util.spec_from_file_location(
        "convert_lerobot_g1", _REPO_ROOT / "scripts" / "convert_lerobot_g1.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


conv = _load_converter()


def test_canonical_q_picks_waist_yaw_and_both_arms() -> None:
    state = np.arange(43, dtype=np.float32)
    q = conv.canonical_q(state)
    assert q.shape == (15,)
    assert q[0] == 12.0  # waist_yaw
    assert list(q[1:8]) == list(range(15, 22))  # left arm
    assert list(q[8:15]) == list(range(22, 29))  # right arm


def test_hand_synergy_is_bounded_and_monotone() -> None:
    open_hand = conv.hand_synergy(np.full(7, -1.0, dtype=np.float32))
    mid = conv.hand_synergy(np.zeros(7, dtype=np.float32))
    closed = conv.hand_synergy(np.full(7, 1.0, dtype=np.float32))
    assert open_hand == 0.0 and closed == 1.0 and mid == pytest.approx(0.5)
    assert conv.hand_synergy(np.full(7, 99.0, dtype=np.float32)) == 1.0  # clipped


def test_relabel_chunks_are_executed_state_deltas() -> None:
    rng = np.random.default_rng(0)
    q = np.cumsum(rng.uniform(-0.01, 0.01, size=(21, 15)), axis=0).astype(np.float32)
    grip = rng.uniform(0, 1, size=(21, 2)).astype(np.float32)
    chunks = conv.relabel_chunks(q, grip, chunk_steps=8, dt_s=1 / 30)
    assert len(chunks) == 2  # steps 0..7 and 8..15; remainder (16..20) dropped
    chunk, start = chunks[1]
    assert start == 8
    assert chunk.mode is ActionMode.JOINT_DELTA
    np.testing.assert_allclose(chunk.targets, np.diff(q[8:17], axis=0), atol=1e-7)
    np.testing.assert_allclose(chunk.gripper_target, grip[9:17].mean(axis=-1), atol=1e-7)


def _write_fake_lerobot_episode(root: Path, episode_index: int, n: int = 40) -> None:
    """Minimal GR00T-layout episode: 43-dim states at 30 Hz + a tiny ego mp4."""
    import cv2

    rng = np.random.default_rng(episode_index)
    base = rng.uniform(-0.3, 0.3, size=43).astype(np.float32)
    drift = rng.uniform(-0.005, 0.005, size=(n, 43)).astype(np.float32)
    state = base + np.cumsum(drift, axis=0)
    ts = (np.arange(n) / 30.0).astype(np.float32)

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "observation.state": pa.array(list(state), type=pa.list_(pa.float32())),
                "timestamp": pa.array(ts, type=pa.float32()),
            }
        ),
        data_dir / f"episode_{episode_index:06d}.parquet",
    )

    video_dir = root / "videos" / "chunk-000" / "observation.images.ego_view"
    video_dir.mkdir(parents=True, exist_ok=True)
    path = video_dir / f"episode_{episode_index:06d}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for t in range(n):
        frame = np.full((48, 64, 3), (5 * t) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    meta = root / "meta"
    meta.mkdir(exist_ok=True)
    with (meta / "episodes.jsonl").open("a") as f:
        f.write(
            json.dumps(
                {
                    "episode_index": episode_index,
                    "tasks": ["move the apple to the plate"],
                    "length": n,
                }
            )
            + "\n"
        )


def test_convert_episode_roundtrip_passes_validation_gates(tmp_path: Path) -> None:
    source = tmp_path / "lerobot"
    for i in range(2):
        _write_fake_lerobot_episode(source, i)

    out = tmp_path / "wam"
    rc = conv.main(
        [
            "--source",
            str(source),
            "--out",
            str(out),
            "--episodes",
            "2",
            "--chunk-steps",
            "8",
            "--resize",
            "24",
            "32",
        ]
    )
    assert rc == 0

    report = validate_dataset(out, ValidationThresholds(min_episodes=2, min_duration_s=0.5))
    assert report.passed, report.failed_gates()

    reader = EpisodeReader(out / "gr00t-apple-000000")
    assert reader.manifest.instruction == "move the apple to the plate"
    assert reader.manifest.extra["source"]["dataset"] == "nvidia/GR00T-N1.7-AppleToPlate"
    states = reader.read_states()
    assert len(states) == 40
    first, second = states[0], states[1]
    assert first.q.shape == (15,) and first.gripper_state.shape == (2,)
    assert not first.validity.imu and first.validity.q and first.validity.gripper
    # dq is finite-differenced from q: first row zero, second row (q1-q0)/dt
    np.testing.assert_allclose(first.dq, np.zeros(15), atol=0)
    np.testing.assert_allclose(second.dq, (second.q - first.q) * 30.0, rtol=1e-4, atol=1e-5)
    frames = reader.read_frames("ego")
    assert frames.shape == (40, 24, 32, 3)
    chunks = list(reader.read_actions())
    assert len(chunks) == 4  # floor((40-1)/8) chunks of 8 steps
    chunk, executed_prefix, _ts = chunks[0]
    assert chunk.num_steps == 8 and executed_prefix == 8
    assert float(np.abs(np.asarray(chunk.targets)).max()) < 1.0  # tanh-learnable
