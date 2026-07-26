"""T-07 tests: episode writer/reader roundtrip, checksums, manifest versioning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wam.data import (
    EPISODE_FORMAT_VERSION,
    EpisodeChecksumError,
    EpisodeFormatError,
    EpisodeReader,
    EpisodeWriter,
    list_episodes,
)
from wam.interfaces.schema import (
    SCHEMA_VERSION,
    ActionChunk,
    ActionMode,
    NormalizationSpec,
)
from wam.robot import MockRobot

FPS = 20.0
ITERATIONS = 6
CHUNK_STEPS = 4
PREFIX = 2
WRIST_OFFSET_NS = 5
INSTRUCTION = "pick up the red cube"


def _make_chunk(rng: np.random.Generator, num_joints: int) -> ActionChunk:
    targets = rng.uniform(-0.05, 0.05, size=(CHUNK_STEPS, num_joints)).astype(np.float32)
    targets[:, 0] = 0.3  # sweep q[0] so the rendered dot moves between frames
    gripper = rng.uniform(0.0, 1.0, size=CHUNK_STEPS).astype(np.float32)
    return ActionChunk(
        mode=ActionMode.JOINT_DELTA, targets=targets, gripper_target=gripper, dt_s=0.05
    )


def _record(dir: Path) -> dict:
    """Drive a MockRobot, record an episode, return the ground truth for comparisons."""
    robot = MockRobot(num_joints=4, seed=0)
    spec = robot.spec
    rng = np.random.default_rng(0)
    norm = {
        "action": NormalizationSpec(mean=(0.0,) * 4, std=(1.0,) * 4),
        "state_q": NormalizationSpec(mean=(0.1,) * 4, std=(2.0,) * 4),
    }
    states, actions, frames, frame_ts = [], [], {"front": [], "wrist": []}, {"front": [], "wrist": []}
    with EpisodeWriter(
        dir,
        "ep-0001",
        spec,
        FPS,
        INSTRUCTION,
        normalization=norm,
        extra={"operator": "test", "d_phase": "D0"},
    ) as writer:
        for _ in range(ITERATIONS):
            state = robot.read_state()
            writer.add_state(state)
            states.append(state)

            rendered = robot.render_frames(1)
            for cam in ("front", "wrist"):
                offset = WRIST_OFFSET_NS if cam == "wrist" else 0
                ts = robot.sim_time_ns + offset
                writer.add_frame(cam, rendered[cam][0], ts)
                frames[cam].append(rendered[cam][0])
                frame_ts[cam].append(ts)

            chunk = _make_chunk(rng, spec.num_joints)
            ts = robot.sim_time_ns
            writer.add_action(chunk, PREFIX, ts)
            actions.append((chunk, PREFIX, ts))
            robot.execute(chunk, PREFIX)
        manifest = writer.close()
    return {
        "spec": spec,
        "norm": norm,
        "states": states,
        "actions": actions,
        "frames": {cam: np.stack(f) for cam, f in frames.items()},
        "frame_ts": frame_ts,
        "manifest": manifest,
    }


@pytest.fixture(scope="module")
def episode(tmp_path_factory: pytest.TempPathFactory) -> dict:
    dir = tmp_path_factory.mktemp("episodes") / "ep-0001"
    truth = _record(dir)
    truth["dir"] = dir
    return truth


def _dot_col(frame: np.ndarray) -> float:
    """Column centroid of the bright dot (pixels above half of the max brightness)."""
    gray = frame.astype(np.float64).sum(axis=2)
    mask = gray > 0.5 * gray.max()
    cols = np.where(mask.any(axis=0))[0]
    return float(cols.mean())


# -- roundtrip -------------------------------------------------------------------------------


def test_states_roundtrip_exact(episode: dict) -> None:
    reader = EpisodeReader(episode["dir"])
    got = reader.read_states()
    assert len(got) == ITERATIONS
    for orig, back in zip(episode["states"], got):
        assert back.timestamp_ns == orig.timestamp_ns
        for name in ("q", "dq", "gripper_state"):
            a, b = getattr(orig, name), getattr(back, name)
            assert b.dtype == np.float32
            assert np.array_equal(a, b), name
        assert np.array_equal(orig.imu.orientation_wxyz, back.imu.orientation_wxyz)
        assert np.array_equal(orig.imu.angular_velocity, back.imu.angular_velocity)
        assert np.array_equal(orig.imu.linear_acceleration, back.imu.linear_acceleration)
        assert back.validity.as_dict() == orig.validity.as_dict()
        assert back.schema_version == orig.schema_version
        assert back.validate(episode["spec"]) == []


def test_actions_roundtrip_exact(episode: dict) -> None:
    reader = EpisodeReader(episode["dir"])
    got = reader.read_actions()
    assert len(got) == ITERATIONS
    for (orig, prefix, ts), (back, back_prefix, back_ts) in zip(episode["actions"], got):
        assert back.mode is orig.mode
        assert back.targets.dtype == np.float32
        assert np.array_equal(orig.targets, back.targets)
        assert np.array_equal(orig.gripper_target, back.gripper_target)
        assert back.dt_s == orig.dt_s
        assert back.schema_version == orig.schema_version
        assert (back_prefix, back_ts) == (prefix, ts)
        assert back.validate(episode["spec"]) == []


def test_frames_roundtrip_structural(episode: dict) -> None:
    reader = EpisodeReader(episode["dir"])
    for cam in ("front", "wrist"):
        orig = episode["frames"][cam]
        got = reader.read_frames(cam)
        assert got.shape == orig.shape
        assert got.dtype == np.uint8
        # lossy codec: structural checks only — dot position and overall pixel closeness
        for i in range(orig.shape[0]):
            assert abs(_dot_col(orig[i]) - _dot_col(got[i])) <= 3.0, (cam, i)
        mean_err = np.abs(orig.astype(np.int32) - got.astype(np.int32)).mean()
        assert mean_err < 8.0, (cam, mean_err)
    # the dot actually moved during the episode (test is not vacuous)
    cols = [_dot_col(f) for f in episode["frames"]["front"]]
    assert max(cols) - min(cols) > 5.0


def test_frame_timestamps_and_sync_stats(episode: dict) -> None:
    reader = EpisodeReader(episode["dir"])
    for cam in ("front", "wrist"):
        ts = reader.frame_timestamps(cam)
        assert ts.dtype == np.int64
        assert ts.tolist() == episode["frame_ts"][cam]
    m = reader.manifest
    assert m.max_sync_error_ns == WRIST_OFFSET_NS
    all_ts = (
        [s.timestamp_ns for s in episode["states"]]
        + [ts for c in ("front", "wrist") for ts in episode["frame_ts"][c]]
        + [ts for _, _, ts in episode["actions"]]
    )
    assert m.t0_ns == min(all_ts)
    assert m.t1_ns == max(all_ts)


def test_manifest_contents(episode: dict) -> None:
    m = EpisodeReader(episode["dir"]).manifest
    assert m.episode_id == "ep-0001"
    assert m.format_version == EPISODE_FORMAT_VERSION
    assert m.schema_version == SCHEMA_VERSION
    assert m.spec == episode["spec"]
    assert m.instruction == INSTRUCTION
    assert m.extra == {"operator": "test", "d_phase": "D0"}
    assert set(m.cameras) == {"front", "wrist"}
    for cam, info in m.cameras.items():
        assert info.file == f"{cam}.mp4"
        assert (info.width, info.height, info.fps) == (64, 64, FPS)
        assert info.num_frames == ITERATIONS
    assert m.tables["states"].num_rows == ITERATIONS
    assert m.tables["actions"].num_rows == ITERATIONS * CHUNK_STEPS
    assert m.tables["front_timestamps"].num_rows == ITERATIONS
    # every data file is checksummed and present
    expected_files = {info.file for info in m.cameras.values()}
    expected_files |= {info.file for info in m.tables.values()}
    assert set(m.checksums) == expected_files
    for fname in expected_files:
        assert (episode["dir"] / fname).is_file()
    # normalization specs roundtrip through their dict form
    specs = m.normalization_specs()
    assert specs is not None and set(specs) == {"action", "state_q"}
    for key, spec in specs.items():
        assert spec == episode["norm"][key]


# -- integrity + versioning ------------------------------------------------------------------


def _copy_episode(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True)
    for f in src.iterdir():
        (dst / f.name).write_bytes(f.read_bytes())


def test_checksum_tamper_detected(episode: dict, tmp_path: Path) -> None:
    tampered = tmp_path / "tampered"
    _copy_episode(episode["dir"], tampered)
    target = tampered / "states.parquet"
    raw = bytearray(target.read_bytes())
    raw[10] ^= 0xFF
    target.write_bytes(bytes(raw))
    with pytest.raises(EpisodeChecksumError, match="states.parquet"):
        EpisodeReader(tampered)
    # opt-out flag skips verification
    reader = EpisodeReader(tampered, verify_checksums=False)
    assert reader.manifest.episode_id == "ep-0001"


def test_missing_file_detected(episode: dict, tmp_path: Path) -> None:
    broken = tmp_path / "broken"
    _copy_episode(episode["dir"], broken)
    (broken / "front.mp4").unlink()
    with pytest.raises(EpisodeChecksumError, match="front.mp4"):
        EpisodeReader(broken)


def test_manifest_version_mismatch_rejected(episode: dict, tmp_path: Path) -> None:
    other = tmp_path / "other"
    _copy_episode(episode["dir"], other)
    manifest = json.loads((other / "manifest.json").read_text())
    manifest["format_version"] = "1.0.0"
    (other / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(EpisodeFormatError, match="format_version"):
        EpisodeReader(other)
    manifest["format_version"] = EPISODE_FORMAT_VERSION
    manifest["schema_version"] = "2.0.0"
    (other / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(EpisodeFormatError, match="schema_version"):
        EpisodeReader(other)


# -- discovery + lifecycle -------------------------------------------------------------------


def test_list_episodes(episode: dict, tmp_path: Path) -> None:
    root = tmp_path / "root"
    _copy_episode(episode["dir"], root / "b" / "ep-0002")
    _copy_episode(episode["dir"], root / "a-ep-0001")
    (root / "junk").mkdir()
    (root / "junk" / "notes.txt").write_text("no manifest here")
    found = list_episodes(root)
    assert found == [root / "a-ep-0001", root / "b" / "ep-0002"]
    assert list_episodes(root / "does-not-exist") == []


def test_empty_episode_roundtrip(tmp_path: Path) -> None:
    robot = MockRobot(num_joints=4)
    with EpisodeWriter(tmp_path / "empty", "ep-empty", robot.spec, FPS, "") as writer:
        pass
    reader = EpisodeReader(tmp_path / "empty")
    assert reader.read_states() == []
    assert reader.read_actions() == []
    assert reader.manifest.cameras == {}
    assert reader.manifest.normalization is None
    assert reader.manifest.normalization_specs() is None
    assert (reader.manifest.t0_ns, reader.manifest.t1_ns) == (0, 0)
    with pytest.raises(KeyError):
        reader.read_frames("front")
    _ = writer  # context manager closed it; second close is a no-op
    writer.close()


def test_abort_leaves_no_manifest(tmp_path: Path) -> None:
    robot = MockRobot(num_joints=4)
    dir = tmp_path / "aborted"
    with (
        pytest.raises(RuntimeError, match="boom"),
        EpisodeWriter(dir, "ep-x", robot.spec, FPS, "") as writer,
    ):
        writer.add_frame("front", robot.render_frames(1)["front"][0], 0)
        raise RuntimeError("boom")
    assert not (dir / "manifest.json").exists()
    assert list_episodes(tmp_path) == []
    with pytest.raises(FileNotFoundError):
        EpisodeReader(dir)


def test_writer_input_validation(tmp_path: Path) -> None:
    robot = MockRobot(num_joints=4)
    frame = robot.render_frames(1)["front"][0]
    chunk = _make_chunk(np.random.default_rng(0), 4)
    with EpisodeWriter(tmp_path / "v", "ep-v", robot.spec, FPS, "") as writer:
        with pytest.raises(ValueError, match="uint8"):
            writer.add_frame("front", frame.astype(np.float32), 0)
        with pytest.raises(ValueError, match="even"):
            writer.add_frame("front", np.zeros((63, 64, 3), dtype=np.uint8), 0)
        with pytest.raises(ValueError, match="camera name"):
            writer.add_frame("../evil", frame, 0)
        writer.add_frame("front", frame, 100)
        with pytest.raises(ValueError, match="size changed"):
            writer.add_frame("front", np.zeros((32, 32, 3), dtype=np.uint8), 200)
        with pytest.raises(ValueError, match="non-decreasing"):
            writer.add_frame("front", frame, 50)
        with pytest.raises(ValueError, match="executed_prefix"):
            writer.add_action(chunk, CHUNK_STEPS + 1, 0)
    with pytest.raises(RuntimeError, match="closed"):
        writer.add_state(robot.read_state())
    with pytest.raises(FileExistsError):
        EpisodeWriter(tmp_path / "v", "ep-v2", robot.spec, FPS, "")
    with pytest.raises(ValueError, match="fps"):
        EpisodeWriter(tmp_path / "v2", "ep-v3", robot.spec, 0.0, "")
