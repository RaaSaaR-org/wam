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


def test_parse_readouts_round_trips_and_rejects_junk() -> None:
    assert probe.parse_readouts("mean,grid2x2,rand4") == (
        ("mean", None),
        ("grid", (2, 2)),
        ("rand", 4),
    )
    # order preserved, duplicates dropped, whitespace and case tolerated
    assert probe.parse_readouts(" GRID3x4 , mean ,grid3x4") == (("grid", (3, 4)), ("mean", None))
    assert [probe.readout_label(k, p) for k, p in probe.parse_readouts("mean,grid2x2,rand4")] == [
        "mean",
        "grid2x2",
        "rand4",
    ]
    assert probe.readout_width("mean", None, 8) == 8
    assert probe.readout_width("grid", (2, 2), 8) == 32
    assert probe.readout_width("rand", 4, 8) == 32

    for bad in ("grid2", "grid0x2", "gridaxb", "rand", "rand0", "pool", ""):
        with pytest.raises(ValueError):
            probe.parse_readouts(bad)


def test_mean_readout_is_the_historical_pooling() -> None:
    """`mean` must stay byte-identical to the old `.float().mean(dim=1)` — it is the anchor
    that keeps runs/wan_probe/ reproducible while the spatial readouts are added."""
    torch = pytest.importorskip("torch")

    tokens = torch.randn(1, 2 * 6 * 8, 16, dtype=torch.float32)
    out = probe.apply_readouts(tokens, (2, 6, 8), (("mean", None),))
    torch.testing.assert_close(out["mean"], tokens.float().mean(dim=1), rtol=0, atol=0)


def test_grid_readout_keeps_position_where_mean_pool_destroys_it() -> None:
    """I-1 in miniature: a signal that lives in ONE spatial cell survives the grid readout and
    is annihilated by the mean-pool, because the grid never averages across cells."""
    torch = pytest.importorskip("torch")

    grid = (2, 6, 8)  # the real probe geometry: 5 frames at 192x256 through the Wan VAE
    frames, rows, cols = grid
    dim = 4
    # Two windows that differ ONLY in which half of the frame carries the +1 activation.
    left = torch.zeros(1, frames, rows, cols, dim)
    left[:, :, :, : cols // 2] = 1.0
    right = torch.zeros(1, frames, rows, cols, dim)
    right[:, :, :, cols // 2 :] = 1.0
    tokens = torch.cat([left, right]).reshape(2, frames * rows * cols, dim)

    out = probe.apply_readouts(tokens, grid, probe.parse_readouts("mean,grid1x2"))
    # mean-pool: both windows collapse to the same vector — the difference is gone
    torch.testing.assert_close(out["mean"][0], out["mean"][1])
    # grid1x2: two cells of 1.0/0.0, mirrored between the windows
    torch.testing.assert_close(out["grid1x2"][0], torch.tensor([1.0] * dim + [0.0] * dim))
    torch.testing.assert_close(out["grid1x2"][1], torch.tensor([0.0] * dim + [1.0] * dim))


def test_grid_readout_pools_cells_and_averages_time() -> None:
    torch = pytest.importorskip("torch")

    grid = (2, 6, 8)
    frames, rows, cols = grid
    dim = 3
    # value = the spatial column index, constant in time -> a 2x2 grid must average columns
    space = torch.arange(cols, dtype=torch.float32).reshape(1, 1, 1, cols, 1)
    tokens = space.expand(1, frames, rows, cols, dim).reshape(1, frames * rows * cols, dim)

    out = probe.apply_readouts(tokens, grid, probe.parse_readouts("mean,grid1x2,grid2x2"))
    assert out["grid1x2"].shape == (1, 2 * dim)
    assert out["grid2x2"].shape == (1, 4 * dim)
    # left half = mean(0..3) = 1.5, right half = mean(4..7) = 5.5
    torch.testing.assert_close(out["grid1x2"][0], torch.tensor([1.5] * dim + [5.5] * dim))
    # rows are identical, so both grid rows repeat the same pair of column means
    torch.testing.assert_close(out["grid2x2"][0], torch.tensor(([1.5] * dim + [5.5] * dim) * 2))
    torch.testing.assert_close(out["mean"][0], torch.full((dim,), 3.5))


def test_random_control_matches_grid_width_and_is_seeded() -> None:
    """`rand<N>` is the control that makes the grid comparison mean anything: same width, same
    group sizes, geometry replaced by a seeded permutation."""
    torch = pytest.importorskip("torch")

    grid = (2, 6, 8)
    tokens = torch.randn(3, grid[0] * grid[1] * grid[2], 5)
    out = probe.apply_readouts(tokens, grid, probe.parse_readouts("grid2x2,rand4"), seed=0)
    assert out["rand4"].shape == out["grid2x2"].shape == (3, 4 * 5)
    # same tokens, so the two readouts must agree on the overall mean but not on the cells
    torch.testing.assert_close(
        out["rand4"].reshape(3, 4, 5).mean(1), out["grid2x2"].reshape(3, 4, 5).mean(1)
    )
    assert not torch.allclose(out["rand4"], out["grid2x2"])

    same = probe.apply_readouts(tokens, grid, probe.parse_readouts("rand4"), seed=0)
    other = probe.apply_readouts(tokens, grid, probe.parse_readouts("rand4"), seed=1)
    torch.testing.assert_close(out["rand4"], same["rand4"], rtol=0, atol=0)
    assert not torch.allclose(out["rand4"], other["rand4"])


def test_readout_width_matches_what_apply_readouts_produces() -> None:
    """`extract_features` preallocates from `readout_width` and assigns from `apply_readouts`,
    so a disagreement between the two would corrupt the feature matrix silently for every
    readout whose width happened to divide the allocated one."""
    torch = pytest.importorskip("torch")

    grid = (2, 6, 8)
    dim = 7
    tokens = torch.randn(1, grid[0] * grid[1] * grid[2], dim)
    readouts = probe.parse_readouts("mean,grid1x2,grid2x2,grid3x4,grid6x8,rand4,rand12,rand48")
    out = probe.apply_readouts(tokens, grid, readouts)
    for kind, param in readouts:
        label = probe.readout_label(kind, param)
        assert out[label].shape == (1, probe.readout_width(kind, param, dim)), label


def test_readouts_reject_geometry_that_does_not_fit() -> None:
    torch = pytest.importorskip("torch")

    tokens = torch.randn(1, 2 * 6 * 8, 4)
    with pytest.raises(ValueError, match="tokens but the grid"):
        probe.apply_readouts(tokens, (2, 6, 7), probe.parse_readouts("grid2x2"))
    with pytest.raises(ValueError, match="does not divide"):
        probe.apply_readouts(tokens, (2, 6, 8), probe.parse_readouts("grid4x4"))
    with pytest.raises(ValueError, match="only 48 tokens exist"):
        probe.apply_readouts(tokens, (2, 6, 8), probe.parse_readouts("rand64"))


def test_analyze_probes_scores_every_readout_and_controls_for_width() -> None:
    """The dict form ranks each readout separately, keeps the primary one in the legacy place,
    and reports the grid-vs-random control."""
    from wam.interfaces.schema import RobotState, ValidityMask

    rng = np.random.default_rng(3)
    n, d, label_dim = 96, 16, 10
    episode_of = np.repeat(np.arange(8), n // 8)
    signal = rng.standard_normal((n, d)).astype(np.float32)
    labels = signal @ rng.standard_normal((d, label_dim)).astype(np.float32)

    def stack(informative: np.ndarray) -> np.ndarray:
        return np.stack(
            [rng.standard_normal((n, informative.shape[1])).astype(np.float32),
             informative,
             rng.standard_normal((n, informative.shape[1])).astype(np.float32)],
            axis=1,
        )  # fmt: skip

    # grid2x2 sees the signal; mean sees noise only; rand4 has grid2x2's width but no signal
    features = {
        "mean": stack(rng.standard_normal((n, d)).astype(np.float32)),
        "grid2x2": stack(np.tile(signal, (1, 4))),
        "rand4": stack(rng.standard_normal((n, 4 * d)).astype(np.float32)),
    }
    windows = [
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
        for i in range(n)
    ]
    args = probe.parse_args(
        ["--data-dir", "unused", "--measured-blocks", "1,2", "--heuristic-blocks", "0,2"]
    )
    report = probe.smoke.Report()
    probe.analyze_probes(features, windows, args, report)

    assert not report.failed
    result = report.info["probe"]
    # the primary readout (first key) still occupies the legacy top-level slot
    assert result["readout"] == "mean"
    assert result["per_block"] == result["readouts"]["mean"]["per_block"]
    assert set(result["readouts"]) == {"mean", "grid2x2", "rand4"}
    assert result["readouts"]["grid2x2"]["ranking_by_val_r2"][0] == 1

    comparison = result["readout_comparison"]
    grid_r2 = comparison["per_readout"]["grid2x2"]["suggested_joints_test_r2"]
    assert grid_r2 > comparison["per_readout"]["mean"]["suggested_joints_test_r2"]
    control = comparison["grid_vs_random_control"]
    assert [c["control"] for c in control] == ["rand4"]
    assert control[0]["geometry_helps"] and control[0]["grid_r2"] == grid_r2
    assert comparison["any_geometry_gain_over_control"]
    # state_only is fitted once and shared, so every readout quotes the identical floor
    floors = {
        r["candidates"]["state_only"]["joints"]["test_r2"] for r in result["readouts"].values()
    }
    assert floors == {comparison["state_only_joints_test_r2"]}


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


# ---- LoRA on the generate path (--gen-lora) -------------------------------------------------


class _FakePeftConfig:
    def __init__(self, r: int, lora_alpha: int) -> None:
        self.r = r
        self.lora_alpha = lora_alpha


class _FakeTransformer:
    """Records what ``apply_lora`` asked of it; no DiT and no GPU involved."""

    def __init__(self) -> None:
        self.loaded: dict | None = None
        self.activated: tuple | None = None
        self.peft_config = {"wam": _FakePeftConfig(32, 64)}

    def load_lora_adapter(self, path, *, adapter_name, prefix, weight_name):
        self.loaded = {
            "path": str(path),
            "adapter_name": adapter_name,
            "prefix": prefix,
            "weight_name": weight_name,
        }

    def set_adapters(self, names, weights):
        self.activated = (names, weights)


class _FakePipe:
    def __init__(self) -> None:
        self.transformer = _FakeTransformer()


def _write_lora_dir(root: Path, *, metadata: bool = True) -> Path:
    """A minimal export_lora.py output: two tensors plus the adapter metadata."""
    import torch
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=True)
    meta = {"format": "pt"}
    if metadata:
        meta["lora_adapter_metadata"] = json.dumps({"r": 32, "lora_alpha": 64})
    save_file(
        {
            "blocks.0.attn1.to_q.lora_A.weight": torch.zeros(32, 16),
            "blocks.0.attn1.to_q.lora_B.weight": torch.zeros(16, 32),
        },
        str(root / "pytorch_lora_weights.safetensors"),
        metadata=meta,
    )
    return root


def test_apply_lora_attaches_model_relative_keys_and_sets_the_scale(tmp_path: Path) -> None:
    """``prefix=None`` is the contract: the export writes model-relative names, so the
    pipeline-level loader (which expects a ``transformer.`` prefix) would match nothing."""
    lora_dir = _write_lora_dir(tmp_path / "lora")
    args = probe.parse_args(
        ["--data-dir", "unused", "--gen-lora", str(lora_dir), "--gen-lora-scale", "0.5"]
    )
    pipe = _FakePipe()
    info = probe.apply_lora(pipe, args, probe.smoke.Report())

    assert pipe.transformer.loaded == {
        "path": str(lora_dir),
        "adapter_name": "wam",
        "prefix": None,
        "weight_name": "pytorch_lora_weights.safetensors",
    }
    assert pipe.transformer.activated == ("wam", 0.5)
    # alpha/r is 2.0, so "scale 0.5" means the adapter runs at its trained strength x 0.5.
    assert info["effective_scaling"] == pytest.approx(1.0)
    assert info["tensors"] == 2


def test_apply_lora_flags_a_missing_alpha_instead_of_silently_halving_it(tmp_path: Path) -> None:
    """Without the metadata the loader infers alpha = r, so the adapter would run at half
    strength and read as a weak fine-tune. That has to show up as a failed check."""
    lora_dir = _write_lora_dir(tmp_path / "lora", metadata=False)
    args = probe.parse_args(["--data-dir", "unused", "--gen-lora", str(lora_dir)])
    report = probe.smoke.Report()
    probe.apply_lora(_FakePipe(), args, report)
    assert "generate.lora_metadata" in report.failed


def test_apply_lora_reports_the_checkpoint_the_pixels_came_from(tmp_path: Path) -> None:
    """AC-04: a clip has to be attributable to a run, so provenance rides along into the
    report rather than living only in the directory nobody looks at."""
    lora_dir = _write_lora_dir(tmp_path / "lora")
    (lora_dir / "wam_provenance.json").write_text(
        json.dumps({"run_id": "t16-lora-seed0", "config_hash": "45ee9e60"})
    )
    args = probe.parse_args(["--data-dir", "unused", "--gen-lora", str(lora_dir)])
    info = probe.apply_lora(_FakePipe(), args, probe.smoke.Report())
    assert info["provenance"]["run_id"] == "t16-lora-seed0"


def test_apply_lora_refuses_a_directory_without_an_exported_adapter(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    args = probe.parse_args(["--data-dir", "unused", "--gen-lora", str(empty)])
    with pytest.raises(FileNotFoundError, match="export_lora.py"):
        probe.apply_lora(_FakePipe(), args, probe.smoke.Report())
