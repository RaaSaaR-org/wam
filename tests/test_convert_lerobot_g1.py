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


def test_legacy_clipped_frac_measures_the_rail_the_legacy_mapping_hides() -> None:
    """The legacy mapping's clip had no compensating gate, unlike the pinned affine's.

    Clipping is silent in the written dataset and moves EVERY downstream admissibility clause in
    the passing direction: a railed channel reads as a wide, decisive, two-state gripper. The
    audit cannot see it either — by the time it reads gripper_target, the rail IS the data. So it
    has to be caught where the unclipped values still exist.
    """
    in_range = [np.zeros((4, 43), dtype=np.float32)]
    assert conv.legacy_clipped_frac(in_range) == 0.0

    off_scale = np.zeros((4, 43), dtype=np.float32)
    off_scale[:, conv.LEFT_HAND] = 1.8  # (1.8 + 1) / 2 = 1.4, railed to 1.0
    assert conv.legacy_clipped_frac([off_scale]) == pytest.approx(0.5)  # one hand of two


def test_hand_synergy_rails_an_off_scale_source_without_saying_so() -> None:
    """Why the check above lives outside hand_synergy: the function itself cannot report it.

    Two physically different hands map to the same stored value, so the information is gone
    before anything downstream could gate on it.
    """
    assert conv.hand_synergy(np.full(7, 1.8, dtype=np.float32)) == 1.0
    assert conv.hand_synergy(np.full(7, 99.0, dtype=np.float32)) == 1.0
    assert conv.LEGACY_MAX_CLIPPED_FRAC == 0.0  # same bar pinned_hand_affine already held


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


def _write_fake_lerobot_episode(
    root: Path, episode_index: int, n: int = 40, hand_offset: float = 0.0
) -> None:
    """Minimal GR00T-layout episode: 43-dim states at 30 Hz + a tiny ego mp4.

    ``hand_offset`` replaces the LEFT hand with a closing hand parked ``hand_offset`` rad off the
    [-1, 1] scale the legacy gripper formula assumes: it is then the most active joint group in
    the episode, so it is both what legacy rails and what ``active-hand`` fits. At the default
    the whole state is inside [-1, 1] and nothing rails, which is what every other test wants.
    """
    import cv2

    rng = np.random.default_rng(episode_index)
    base = rng.uniform(-0.3, 0.3, size=43).astype(np.float32)
    drift = rng.uniform(-0.005, 0.005, size=(n, 43)).astype(np.float32)
    state = base + np.cumsum(drift, axis=0)
    if hand_offset:
        closing = np.linspace(0.0, 0.5, n, dtype=np.float32)
        state[:, conv.LEFT_HAND] = hand_offset + closing[:, None]
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


def _convert_args(source: Path, out: Path, *extra: str) -> list[str]:
    return [
        "--source", str(source),
        "--out", str(out),
        "--episodes", "2",
        "--chunk-steps", "8",
        "--resize", "24", "32",
        *extra,
    ]  # fmt: skip


def test_the_converter_refuses_a_source_the_legacy_mapping_would_rail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``legacy_clipped_frac`` is only a gate if the conversion actually stops on it.

    The roundtrip above converts a source that happens to live inside [-1, 1], so it walks
    through the same branch measuring 0.0 and can never show the refusal. Here the left hand is
    off that scale: every one of its samples is replaced by a rail in the written dataset, where
    it is indistinguishable from a measurement — so the run has to end before any episode is
    written, name the fraction it measured, and name the mapping that does not need to assume a
    scale. That last part is what makes the refusal safe to be absolute: the alternative it
    points at converts THIS source, and clips nothing.
    """
    source = tmp_path / "lerobot"
    for i in range(2):
        _write_fake_lerobot_episode(source, i, hand_offset=1.8)

    out = tmp_path / "railed"
    assert conv.main(_convert_args(source, out, "--gripper-mapping", "legacy")) == 2
    err = capsys.readouterr().err
    assert "clips 0.5000 of samples" in err  # one hand of two, the measurement not just a verdict
    assert "--gripper-mapping active-hand" in err
    assert list(out.glob("gr00t-apple-*")) == []  # refused before a single episode was written

    fitted = tmp_path / "fitted"
    assert conv.main(_convert_args(source, fitted, "--gripper-mapping", "active-hand")) == 0
    reader = EpisodeReader(fitted / "gr00t-apple-000000")
    left = np.stack([s.gripper_state for s in reader.read_states()])[:, 0]
    assert 0.0 <= left.min() and left.max() <= 1.0
    # The same hand the legacy formula flattened onto one rail spans the range once the mapping
    # is fitted instead of assumed — so the refusal costs nothing that had to be given up.
    assert float(left.max() - left.min()) > 0.9
