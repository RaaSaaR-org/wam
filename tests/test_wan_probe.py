"""Tests for scripts/hf_job_wan_probe.py (readout probes on real action labels)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load("hf_job_wan_probe")
conv = _load("convert_lerobot_g1")


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
        writer.write(np.full((48, 64, 3), (5 * t) % 255, dtype=np.uint8))
    writer.release()

    meta = root / "meta"
    meta.mkdir(exist_ok=True)
    with (meta / "episodes.jsonl").open("a") as f:
        row = {
            "episode_index": episode_index,
            "tasks": ["move the apple to the plate"],
            "length": n,
        }
        f.write(json.dumps(row) + "\n")


def test_build_windows_labels_are_canonical_state_deltas(tmp_path: Path) -> None:
    source = tmp_path / "lerobot"
    for i in range(2):
        _write_fake_lerobot_episode(source, i)

    args = probe.parse_args(
        [
            "--data-dir", str(source),
            "--episodes", "2",
            "--frames", "3",
            "--height", "32",
            "--width", "32",
            "--chunk-steps", "8",
            "--windows-per-episode", "3",
        ]
    )  # fmt: skip
    windows, instruction, info = probe.build_windows(args)

    assert instruction == "move the apple to the plate"
    # 40 steps, chunks start at 0/8/16/24; start >= frames-1 leaves 8/16/24 -> 3 per episode
    assert info["windows"] == len(windows) == 6
    assert sorted({w["episode"] for w in windows}) == [0, 1]

    window = next(w for w in windows if w["episode"] == 1 and w["start"] == 8)
    assert window["frames"].shape == (3, 32, 32, 3)
    assert window["label"].shape == (8 * 15 + 8,)
    assert not window["state"].validity.imu and window["state"].validity.q

    raw = np.stack(
        pq.read_table(source / "data" / "chunk-000" / "episode_000001.parquet")[
            "observation.state"
        ].to_numpy(zero_copy_only=False)
    ).astype(np.float32)
    q = conv.canonical_q(raw)
    np.testing.assert_allclose(
        window["label"][: 8 * 15].reshape(8, 15), np.diff(q[8:17], axis=0), atol=1e-6
    )
    np.testing.assert_allclose(window["state"].q, q[8], atol=1e-6)
    np.testing.assert_allclose(window["state"].dq, (q[8] - q[7]) * 30.0, rtol=1e-3, atol=1e-4)


def test_episode_split_is_disjoint_and_by_episode() -> None:
    episode_of = np.repeat(np.arange(8), 4)
    split = probe.episode_split(episode_of)
    train, val, test = split["train"], split["val"], split["test"]
    assert len(set(train) & set(val)) == len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    assert len(train) + len(val) + len(test) == len(episode_of)
    # no episode straddles two splits
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(episode_of[split[a]]) & set(episode_of[split[b]])
    assert split["test_eps"].tolist() == [6, 7]

    with pytest.raises(ValueError, match="need >= 4 episodes"):
        probe.episode_split(np.repeat(np.arange(3), 4))


def test_probe_r2_separates_signal_from_noise() -> None:
    rng = np.random.default_rng(0)
    n, d, out = 96, 40, 6
    episode_of = np.repeat(np.arange(8), n // 8)
    signal = rng.standard_normal((n, d)).astype(np.float32)
    w_true = rng.standard_normal((d, out)).astype(np.float32)
    y = signal @ w_true + 0.01 * rng.standard_normal((n, out)).astype(np.float32)
    noise = rng.standard_normal((n, d)).astype(np.float32)

    split = probe.episode_split(episode_of)
    alphas = (1.0, 10.0, 100.0)
    good = probe.probe_r2(signal, y, split, alphas)
    bad = probe.probe_r2(noise, y, split, alphas)
    assert good["test_r2"] > 0.9
    assert bad["test_r2"] < 0.2
    assert good["test_r2"] > bad["test_r2"]


def test_analyze_probes_ranks_the_informative_block(tmp_path: Path) -> None:
    from wam.interfaces.schema import RobotState, ValidityMask

    rng = np.random.default_rng(1)
    n, d, label_dim = 96, 32, 10
    episode_of = np.repeat(np.arange(8), n // 8)
    signal = rng.standard_normal((n, d)).astype(np.float32)
    w_true = rng.standard_normal((d, label_dim)).astype(np.float32)
    labels = signal @ w_true

    pooled = np.stack(
        [rng.standard_normal((n, d)).astype(np.float32), signal,
         rng.standard_normal((n, d)).astype(np.float32)],
        axis=1,
    )  # fmt: skip
    windows = []
    for i in range(n):
        windows.append(
            {
                "label": labels[i],
                "episode": int(episode_of[i]),
                "start": i,
                "state": RobotState(
                    timestamp_ns=i,
                    q=rng.standard_normal(15).astype(np.float32),
                    dq=np.zeros(15, dtype=np.float32),
                    imu=probe._zero_imu(),
                    gripper_state=np.zeros(2, dtype=np.float32),
                    validity=ValidityMask(q=True, dq=True, imu=False, gripper=True),
                ),
            }
        )

    args = probe.parse_args(
        ["--data-dir", "unused", "--measured-blocks", "1,2", "--heuristic-blocks", "0,2"]
    )
    report = probe.smoke.Report()
    probe.analyze_probes(pooled, windows, args, report)

    assert not report.failed
    result = report.info["probe"]
    assert result["ranking_by_val_r2"][0] == 1  # the informative block wins
    assert 1 in result["suggested_blocks"]
    per_block = result["per_block"]
    # label_dim < chunk_steps*15, so the whole label counts as "joints" and gripper is absent
    assert "gripper" not in per_block["1"]
    assert per_block["1"]["joints"]["test_r2"] > 0.9 > per_block["0"]["joints"]["test_r2"]
    assert result["candidates"]["measured_1_2"]["joints"]["test_r2"] > 0.9
    assert result["verdict"]["measured_beats_heuristic"] in (True, False)
