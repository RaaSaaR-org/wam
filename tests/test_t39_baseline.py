"""Tests for the T-39 positive control (PR-07): the adapter, the oracles, the subset, the witness.

Everything here runs on CPU in ``tmp_path`` with synthetic data. No GR00T weights, no vendored
trainer, no network — the two drivers are built so that the parts which can be wrong on our side
are separable from the part that is NVIDIA's, and this file is where that separation pays off.

WHY THE ANCHORING TESTS ARE THE POINT. ``commanded_to_chunk`` turns a commanded ABSOLUTE position
into the JOINT_DELTA convention every number in this repo is scored in. Get the anchor off by one
step and it still returns a [T, 15] float32 array of plausible magnitude: no shape check, no range
check and no finiteness check fires, the eval runs to completion, and the result is a
``skill_vs_repeat_pct`` that reads as a statement about GR00T when it is a statement about our
arithmetic. That is T-37's transposed-``xmat`` failure exactly — the wrong rotation convention
produced perfectly finite euler angles that every assertion accepted — and PR-07 §8 item 3
therefore requires the mutation to be killed rather than the convention to be argued. Three
plausible mis-anchorings are constructed below and each must fail.

The fixture builds its source recording with PERFECT TRACKING (``action[i] == state[i+1]``), which
is what makes the mutation test sharp: under perfect tracking the commanded column and the
executed column carry the same information, so the correct anchoring reproduces the stored targets
EXACTLY and any other anchoring cannot. On the real corpus tracking is imperfect and the gap is
the thing ``oracle_action`` is built to measure — but a mapping that cannot reproduce the ideal
case has no business being pointed at the real one.

The episodes are written with the repo's own ``EpisodeWriter`` and the targets with
``convert_lerobot_g1.relabel_chunks``, never a hand-rolled equivalent: the claim under test is
that the oracle reproduces what the CONVERTER produced, and a fixture that reimplemented the
converter would be testing this file against itself.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wam.data import EpisodeWriter
from wam.interfaces.schema import (
    CanonicalSpaceSpec,
    IMUState,
    RobotState,
    ValidityMask,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

FPS = 30.0
DT = 1.0 / FPS
CHUNK_STEPS = 8
NUM_SAMPLES = 60
SOURCE_DIM = 43
EPISODE_ID = "gr00t-apple-000007"
EPISODE_INDEX = 7

ZERO_IMU = IMUState(
    orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
    angular_velocity=np.zeros(3, dtype=np.float32),
    linear_acceleration=np.zeros(3, dtype=np.float32),
)


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = _load("eval_t39_baseline")
tr = _load("train_t39_baseline")
convert = _load("convert_lerobot_g1")
eval_t16 = _load("eval_t16")

SPEC = CanonicalSpaceSpec(
    joint_names=tuple(f"j{i}" for i in range(15)),
    gripper_dims=2,
    ee_frame="base",
    ee_rotation_convention="quat_wxyz",
)


# ------------------------------------------------------------------------------- fixtures


def _source_recording(seed: int = 0) -> dict[str, np.ndarray]:
    """A 43-dim source episode with PERFECT position tracking.

    ``state`` is a smooth random walk; ``action[i]`` is exactly ``state[i + 1]`` — the controller
    was told to reach the position it then reached. The last command has nothing after it to
    match, so it repeats the final state, and every chunk built below stops before it.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.01, (NUM_SAMPLES, SOURCE_DIM)).astype(np.float32)
    state = np.cumsum(steps, axis=0).astype(np.float32)
    action = np.concatenate([state[1:], state[-1:]], axis=0).astype(np.float32)
    ts_ns = np.round(np.arange(NUM_SAMPLES) * DT * 1e9).astype(np.int64)
    return {"state": state, "action": action, "ts_ns": ts_ns}


def _write_converted(root: Path, raw: dict[str, np.ndarray], *, with_frames: bool = True) -> Path:
    """Convert ``raw`` into a WAM episode using the converter's OWN mapping functions."""
    state, ts_ns = raw["state"], raw["ts_ns"]
    q = convert.canonical_q(state)
    grip = convert.gripper_state(state, None)  # legacy mapping, as datasets/gr00t-apple-full uses
    chunks = convert.relabel_chunks(q, grip, chunk_steps=CHUNK_STEPS, dt_s=DT)

    episode_dir = root / EPISODE_ID
    with EpisodeWriter(episode_dir, EPISODE_ID, SPEC, FPS, "pick the apple") as writer:
        for i in range(state.shape[0]):
            ts = int(ts_ns[i])
            if with_frames:
                writer.add_frame("ego", np.full((8, 8, 3), i % 256, dtype=np.uint8), ts)
            writer.add_state(
                RobotState(
                    timestamp_ns=ts,
                    q=q[i],
                    dq=np.zeros(15, dtype=np.float32),
                    imu=ZERO_IMU,
                    gripper_state=grip[i],
                    validity=ValidityMask(q=True, dq=False, imu=False, gripper=True),
                )
            )
        for chunk, start in chunks:
            writer.add_action(
                chunk, executed_prefix=chunk.num_steps, timestamp_ns=int(ts_ns[start])
            )
    return episode_dir


def _write_raw_lerobot(root: Path, raw: dict[str, np.ndarray], indices: tuple[int, ...]) -> Path:
    """A minimal LeRobot source root holding ``indices``, all with the same recording."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "videos" / "chunk-000" / "observation.images.ego").mkdir(parents=True, exist_ok=True)

    records = []
    for index in indices:
        table = pa.table(
            {
                "observation.state": pa.array(list(raw["state"].tolist())),
                "action": pa.array(list(raw["action"].tolist())),
                "timestamp": pa.array((raw["ts_ns"] / 1e9).tolist()),
            }
        )
        pq.write_table(table, root / "data" / "chunk-000" / f"episode_{index:06d}.parquet")
        (
            root / "videos" / "chunk-000" / "observation.images.ego" / f"episode_{index:06d}.mp4"
        ).write_bytes(b"not a real mp4, only a link target")
        records.append({"episode_index": index, "length": int(raw["state"].shape[0])})

    (root / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": len(records),
                "total_frames": sum(r["length"] for r in records),
                "splits": {"train": f"0:{len(records)}"},
                "fps": FPS,
            }
        ),
        encoding="utf-8",
    )
    (root / "meta" / "tasks.jsonl").write_text('{"task_index": 0, "task": "pick"}\n')
    return root


@pytest.fixture(scope="module")
def recording() -> dict[str, np.ndarray]:
    return _source_recording()


@pytest.fixture(scope="module")
def converted(tmp_path_factory: pytest.TempPathFactory, recording) -> Path:
    return _write_converted(tmp_path_factory.mktemp("t39-converted"), recording)


@pytest.fixture(scope="module")
def mapping() -> Any:
    return ev.GripperMapping(affine=None, column=None)


# ------------------------------------------------------- the adapter, and the mutations it must kill


def _reader(episode_dir: Path):
    from wam.data.episode import EpisodeReader

    return EpisodeReader(episode_dir)


def test_oracle_state_reproduces_the_stored_targets(converted, mapping):
    """The identity ``ORACLE_STATE_FLOOR_PCT`` is set against: the label pipeline's own output.

    ``relabel_chunks`` built the stored targets out of exactly these states, so a correct rebuild
    returns them bit for bit. If this ever fails, T39_RULE_V1's first branch is right to VOID the
    experiment and blame us rather than the policy.
    """
    reader = _reader(converted)
    rebuilt = ev.oracle_state_chunks(reader, CHUNK_STEPS, mapping)
    stored = {int(t_ns): chunk for chunk, _prefix, t_ns in reader.read_actions()}
    assert rebuilt, "no oracle chunks were rebuilt at all"
    for t_ns, chunk in rebuilt.items():
        np.testing.assert_array_equal(chunk.targets, stored[t_ns].targets)
        np.testing.assert_array_equal(chunk.gripper_target, stored[t_ns].gripper_target)


def test_oracle_action_reproduces_targets_under_perfect_tracking(converted, recording, mapping):
    """``action[i] == state[i+1]`` makes the commanded and executed columns the same information.

    So the CORRECT anchoring must reproduce the stored targets to float32. This is the assertion
    the three mutants below have to break.

    BOTH channels, and the gripper one is not padding: ``relabel_chunks`` takes the gripper from
    step ``t+1`` while it takes the joint delta ACROSS ``t -> t+1``, so the two channels are
    anchored differently by construction and a single convention applied to both is wrong for one
    of them. An earlier version of this test compared only ``targets``, and a mutation that read
    the commanded gripper one step late survived it.
    """
    reader = _reader(converted)
    rebuilt = ev.oracle_action_chunks(reader, recording, CHUNK_STEPS, mapping, convert)
    stored = {int(t_ns): chunk for chunk, _prefix, t_ns in reader.read_actions()}
    assert rebuilt
    for t_ns, chunk in rebuilt.items():
        np.testing.assert_allclose(chunk.targets, stored[t_ns].targets, rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            chunk.gripper_target, stored[t_ns].gripper_target, rtol=0, atol=1e-6
        )


def _correct_chunk(recording, mapping, start: int) -> np.ndarray:
    return ev.commanded_to_chunk(
        recording["action"][start : start + CHUNK_STEPS],
        recording["state"][start],
        dt_s=DT,
        mapping=mapping,
        convert=convert,
    ).targets


@pytest.mark.parametrize(
    "mutant",
    [
        "shift_command_forward",
        "anchor_on_next_state",
        "first_difference_of_commands",
    ],
)
def test_mis_anchored_deltas_are_killed(converted, recording, mapping, mutant):
    """Three plausible wrong anchorings, each of which produces finite, plausible, wrong numbers.

    - ``shift_command_forward``      ``action[t+1] - q[t]``   — one step of lead
    - ``anchor_on_next_state``       ``action[t]   - q[t+1]`` — differencing against the outcome
    - ``first_difference_of_commands`` ``action[t+1] - action[t]`` — the commanded column's own
      first difference, which is the single most natural thing to reach for and is wrong because
      it measures the change in the SETPOINT rather than the displacement the setpoint asks for.

    Every one of them has the right shape, the right dtype and a magnitude in the same order as
    the truth. Only the comparison against the stored targets separates them.
    """
    start = 2 * CHUNK_STEPS
    action, state = recording["action"], recording["state"]
    correct = _correct_chunk(recording, mapping, start)

    if mutant == "shift_command_forward":
        wrong = ev.commanded_to_chunk(
            action[start + 1 : start + 1 + CHUNK_STEPS],
            state[start],
            dt_s=DT,
            mapping=mapping,
            convert=convert,
        ).targets
    elif mutant == "anchor_on_next_state":
        wrong = ev.commanded_to_chunk(
            action[start : start + CHUNK_STEPS],
            state[start + 1],
            dt_s=DT,
            mapping=mapping,
            convert=convert,
        ).targets
    else:
        q_cmd = convert.canonical_q(action[start : start + CHUNK_STEPS + 1])
        wrong = np.diff(q_cmd, axis=0).astype(np.float32)

    assert wrong.shape == correct.shape
    assert np.isfinite(wrong).all(), "the mutant is not even detectable by a finiteness check"
    assert not np.allclose(wrong, correct, atol=1e-6), (
        f"{mutant} produced the same chunk as the correct anchoring — the test cannot "
        "distinguish them, so it is not pinning the convention"
    )


def test_chunk_deltas_sum_to_the_total_commanded_displacement(recording, mapping):
    """Chaining through the commands is what makes an open-loop chunk add up.

    A chunk's deltas must telescope to ``canonical_q(action[start + K - 1]) -
    canonical_q(state[start])``. If steps after the first were anchored on states the policy never
    observed, this identity would not hold — and it is the identity a controller integrating the
    chunk actually depends on.
    """
    start = 3 * CHUNK_STEPS
    targets = _correct_chunk(recording, mapping, start)
    total = convert.canonical_q(recording["action"][start + CHUNK_STEPS - 1]) - convert.canonical_q(
        recording["state"][start]
    )
    np.testing.assert_allclose(targets.sum(axis=0), total, rtol=0, atol=1e-5)


# ------------------------------------------------------------- PR-10, the anchor-delay sweep


def test_the_sweep_defaults_are_the_t39_convention_exactly(converted, recording, mapping):
    """Every default is T-39's, so the archived command line produces the archived numbers.

    Pinned bit-for-bit rather than argued: the sweep was added to a scorer whose output is already
    published, and a knob that shifts the default by a float would rewrite PR-07-RESULT silently.
    """
    reader = _reader(converted)
    base = ev.oracle_action_chunks(reader, recording, CHUNK_STEPS, mapping, convert)
    explicit = ev.oracle_action_chunks(
        reader, recording, CHUNK_STEPS, mapping, convert, offset=0, margin=0, co_shift=False
    )
    assert set(base) == set(explicit)
    for t_ns, chunk in base.items():
        np.testing.assert_array_equal(chunk.targets, explicit[t_ns].targets)
        np.testing.assert_array_equal(chunk.gripper_target, explicit[t_ns].gripper_target)


@pytest.mark.parametrize("offset", [-2, -1, 1, 2])
def test_a_nonzero_offset_actually_moves_the_command_window(
    converted, recording, mapping, offset
):
    """THE MUTATION TEST PR-10 §8.3 REQUIRES, and the one failure this experiment would not notice.

    An offset knob that is silently a no-op — threaded through the signature, never reaching the
    slice — produces a perfectly flat grid, and a flat grid reads as a confident verdict **J**:
    "no offset helps, therefore the mismatch is content, not shift". That is the wrong answer
    arrived at without a single suspicious number, which is exactly the shape of T-37's transposed
    ``xmat``. So the knob is required to CHANGE something, on both signs, before any grid is run.
    """
    reader = _reader(converted)
    base = ev.oracle_action_chunks(reader, recording, CHUNK_STEPS, mapping, convert, margin=2)
    moved = ev.oracle_action_chunks(
        reader, recording, CHUNK_STEPS, mapping, convert, offset=offset, margin=2
    )
    assert set(base) == set(moved), "the margin did not hold the chunk set fixed across offsets"
    assert base, "no chunks survived the margin — the test proves nothing"
    differing = [
        t_ns
        for t_ns, chunk in base.items()
        if not np.allclose(chunk.targets, moved[t_ns].targets, atol=1e-7)
    ]
    assert differing, (
        f"offset {offset:+d} produced identical targets everywhere — the knob is a no-op and the "
        "sweep would report a flat grid as evidence of content rather than shift"
    )


def test_the_margin_holds_one_chunk_set_across_the_whole_grid(converted, recording, mapping):
    """PR-10 §2's common support: without it each cell scores a different sample set.

    The margin has to be defined by the GRID's width, not by the cell's own offset — otherwise the
    edge cells lose chunks the centre keeps and the sweep measures episode ends.
    """
    reader = _reader(converted)
    grid = {
        k: set(
            ev.oracle_action_chunks(
                reader, recording, CHUNK_STEPS, mapping, convert, offset=k, margin=4
            )
        )
        for k in range(-4, 5)
    }
    sizes = {len(v) for v in grid.values()}
    assert len(sizes) == 1, f"the grid's cells scored different numbers of chunks: {sizes}"
    unmargined = set(ev.oracle_action_chunks(reader, recording, CHUNK_STEPS, mapping, convert))
    assert grid[0] < unmargined, "margin=4 dropped nothing, so it is not restricting anything"


def test_co_shifting_the_anchor_is_a_different_experiment_from_shifting_the_command(
    converted, recording, mapping
):
    """Variants A and B must not collapse into each other, or the control is not a control."""
    reader = _reader(converted)
    a = ev.oracle_action_chunks(
        reader, recording, CHUNK_STEPS, mapping, convert, offset=2, margin=2
    )
    b = ev.oracle_action_chunks(
        reader, recording, CHUNK_STEPS, mapping, convert, offset=2, margin=2, co_shift=True
    )
    assert set(a) == set(b)
    assert any(
        not np.allclose(chunk.targets, b[t_ns].targets, atol=1e-7) for t_ns, chunk in a.items()
    ), "variant B produced variant A's chunks — co_shift never reached the anchor"


def test_commanded_to_chunk_refuses_a_wrong_width(recording, mapping):
    with pytest.raises(SystemExit, match="commanded actions must be"):
        ev.commanded_to_chunk(
            recording["action"][:CHUNK_STEPS, :10],
            recording["state"][0],
            dt_s=DT,
            mapping=mapping,
            convert=convert,
        )


# ------------------------------------------------------- the policy shares the oracle's mapping


def test_a_policy_returning_the_ground_truth_scores_exactly_like_oracle_action(
    converted, recording, mapping
):
    """The design claim ``oracle_action`` rests on, tested rather than asserted.

    PR-07 §4 calls ``oracle_action`` "the ceiling for any policy trained on that column". That is
    only true if the policy arm and the oracle arm cross into canonical units through the SAME
    code. A fake policy that returns the dataset's own action column must therefore produce
    byte-identical chunks to the oracle — if it does not, the two arms are measuring through two
    different adapters and the ceiling means nothing.
    """
    reader = _reader(converted)
    anchors = ev.raw_anchor_indices(reader, recording)
    oracle = ev.oracle_action_chunks(reader, recording, CHUNK_STEPS, mapping, convert)

    def infer(request):
        index = int(np.argmin(np.abs(recording["state"] - request["state"]).sum(axis=1)))
        return recording["action"][index : index + CHUNK_STEPS]

    policy = ev.CommandedPolicy(
        infer,
        camera="ego",
        chunk_steps=CHUNK_STEPS,
        dt_s=DT,
        mapping=mapping,
        convert=convert,
        raw_states={EPISODE_ID: recording["state"]},
        anchors={(EPISODE_ID, t): i for t, i in anchors.items()},
        episode_id=EPISODE_ID,
    )
    from wam.evaluation import build_eval_pairs

    pairs = build_eval_pairs(converted, "ego", CHUNK_STEPS)
    assert pairs
    for obs, _target, _ep in pairs:
        t_ns = int(obs.state.timestamp_ns)
        if t_ns not in oracle:
            continue
        predicted = policy.predict(obs)
        np.testing.assert_array_equal(predicted.targets, oracle[t_ns].targets)
        np.testing.assert_array_equal(predicted.gripper_target, oracle[t_ns].gripper_target)


def test_commanded_policy_refuses_a_short_horizon(converted, recording, mapping):
    """A policy that predicts fewer steps than the bar is scored over is a mismatch, not a result.

    Padding the tail would score invented steps as predictions and would flatter exactly the
    metric under test — the tail of a chunk is where ``horizon_ratio`` looks.
    """
    reader = _reader(converted)
    anchors = ev.raw_anchor_indices(reader, recording)
    policy = ev.CommandedPolicy(
        lambda request: recording["action"][: CHUNK_STEPS - 1],
        camera="ego",
        chunk_steps=CHUNK_STEPS,
        dt_s=DT,
        mapping=mapping,
        convert=convert,
        raw_states={EPISODE_ID: recording["state"]},
        anchors={(EPISODE_ID, t): i for t, i in anchors.items()},
        episode_id=EPISODE_ID,
    )
    from wam.evaluation import build_eval_pairs

    obs = build_eval_pairs(converted, "ego", CHUNK_STEPS)[0][0]
    with pytest.raises(SystemExit, match="horizon and the bar is scored"):
        policy.predict(obs)


def test_chunk_lookup_refuses_an_unanchored_timestamp():
    """A miss must be fatal, not a fallback: the nearest chunk is a neighbouring chunk."""
    policy = ev.ChunkLookupPolicy({}, episode_id="ep")
    state = RobotState(
        timestamp_ns=123,
        q=np.zeros(15, dtype=np.float32),
        dq=np.zeros(15, dtype=np.float32),
        imu=ZERO_IMU,
        gripper_state=np.zeros(2, dtype=np.float32),
        validity=ValidityMask(q=True, dq=False, imu=False, gripper=True),
    )
    from wam.interfaces import Observation

    obs = Observation(images={"ego": np.zeros((4, 4, 3), np.uint8)}, state=state, instruction="")
    with pytest.raises(SystemExit, match="no oracle chunk anchored"):
        policy.predict(obs)


# ---------------------------------------------------------------------- the gripper mapping


class _FakeManifest:
    def __init__(self, mapping: dict[str, Any], normalization: dict[str, Any] | None) -> None:
        self.extra = {"mapping": mapping}
        self._normalization = normalization

    def normalization_specs(self):
        if self._normalization is None:
            return None
        from wam.interfaces.schema import NormalizationSpec

        return {k: NormalizationSpec.from_dict(v) for k, v in self._normalization.items()}


def test_gripper_mapping_legacy_is_the_both_hand_mean():
    got = ev.gripper_mapping_from_manifest(_FakeManifest({}, None), convert)
    assert got.kind == "legacy" and got.column is None
    np.testing.assert_allclose(got.reduce(np.array([[0.2, 0.8]], np.float32)), [0.5])


def test_gripper_mapping_active_hand_reads_the_recorded_affine():
    got = ev.gripper_mapping_from_manifest(
        _FakeManifest(
            {"gripper_mapping": "active-hand", "gripper_target": "left hand only (T-31)"},
            {"gripper_target": {"mean": [-0.4386541], "std": [0.4667477], "version": "0.1.0"}},
        ),
        convert,
    )
    assert got.kind == "active-hand" and got.column == 0
    assert got.affine.offset == pytest.approx(-0.4386541)
    assert got.affine.span == pytest.approx(0.4667477)
    np.testing.assert_allclose(got.reduce(np.array([[0.25, 0.75]], np.float32)), [0.25])


def test_gripper_mapping_refuses_active_hand_without_an_affine():
    """Falling back to legacy here would rescale the whole channel and read as a delta bug."""
    with pytest.raises(SystemExit, match="records no "):
        ev.gripper_mapping_from_manifest(
            _FakeManifest({"gripper_mapping": "active-hand"}, None), convert
        )


def test_gripper_mapping_refuses_an_unnamed_hand():
    with pytest.raises(SystemExit, match="does not say which hand"):
        ev.gripper_mapping_from_manifest(
            _FakeManifest(
                {"gripper_mapping": "active-hand", "gripper_target": "whichever"},
                {"gripper_target": {"mean": [0.0], "std": [1.0], "version": "0.1.0"}},
            ),
            convert,
        )


def test_the_shipped_manifests_are_both_readable():
    """The two real conversions in this repo must parse — legacy and active-hand.

    A unit test on a fake manifest proves the parser handles what the test author imagined. This
    proves it handles what ``scripts/convert_lerobot_g1.py`` actually wrote.
    """
    from wam.data.episode import EpisodeReader, list_episodes

    for root, expected in (("gr00t-apple-full", "legacy"), ("gr00t-apple-grip", "active-hand")):
        episodes = list_episodes(_REPO_ROOT / "datasets" / root)
        if not episodes:
            pytest.skip(f"datasets/{root} not present")
        manifest = EpisodeReader(episodes[0], verify_checksums=False).manifest
        assert ev.gripper_mapping_from_manifest(manifest, convert).kind == expected


# ------------------------------------------------------------------- episode-id <-> source index


@pytest.mark.parametrize("bad", ["gr00t-apple", "ep-abc", "nope"])
def test_episode_index_refuses_ids_it_cannot_map(bad):
    with pytest.raises(SystemExit):
        ev.raw_episode_index(bad)
    with pytest.raises(SystemExit):
        tr._episode_index(bad)


def test_both_scripts_map_ids_the_same_way():
    """One convention, two scripts. A disagreement would train on one episode and score another."""
    for episode_id in ("gr00t-apple-000000", "gr00t-apple-000041", "gr00t-apple-000401"):
        assert ev.raw_episode_index(episode_id) == tr._episode_index(episode_id)


def test_read_raw_episode_refuses_a_missing_source(tmp_path, recording):
    _write_raw_lerobot(tmp_path, recording, (0,))
    with pytest.raises(SystemExit, match="is not the source of"):
        ev.read_raw_episode(tmp_path, "gr00t-apple-000099")


def test_raw_anchors_refuse_a_clock_mismatch(converted, recording):
    """A nearest-match fallback here would score the oracle one step out and report it as a
    label-space mismatch — the exact finding this arm is supposed to be able to make."""
    shifted = dict(recording)
    shifted["ts_ns"] = recording["ts_ns"] + 1
    with pytest.raises(SystemExit, match="not a timestamp of"):
        ev.raw_anchor_indices(_reader(converted), shifted)


# ------------------------------------------------------------------------ the LeRobot subset


def test_subset_links_only_the_selected_episodes(tmp_path, recording):
    source = _write_raw_lerobot(tmp_path / "src", recording, (0, 1, 2, 3))
    stats = tr.build_lerobot_subset(source, tmp_path / "subset", [1, 3])

    linked = sorted(p.name for p in (tmp_path / "subset" / "data" / "chunk-000").glob("*.parquet"))
    assert linked == ["episode_000001.parquet", "episode_000003.parquet"]
    assert all(
        (tmp_path / "subset" / "data" / "chunk-000" / name).is_symlink() for name in linked
    ), "the subset copies parquet instead of linking it"
    assert stats["episodes"] == 2

    kept = [
        json.loads(line)
        for line in (tmp_path / "subset" / "meta" / "episodes.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    assert [r["episode_index"] for r in kept] == [1, 3]

    info = json.loads((tmp_path / "subset" / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 2 * NUM_SAMPLES
    assert info["splits"] == {"train": "0:2"}
    assert info["fps"] == FPS, "unrelated info.json fields must survive the rewrite"
    assert (tmp_path / "subset" / "meta" / "tasks.jsonl").is_file()


def test_subset_metadata_actually_restricts_the_trainer(tmp_path, recording):
    """The rewritten totals are the load-bearing part.

    A trainer that trusts ``info.json`` over the directory listing — and they generally do — would
    otherwise iterate 402 episodes over a directory holding 362 and either crash or skip silently,
    training on a set nobody chose. The source says 4; the subset must not.
    """
    source = _write_raw_lerobot(tmp_path / "src", recording, (0, 1, 2, 3))
    tr.build_lerobot_subset(source, tmp_path / "subset", [2])
    assert json.loads((source / "meta" / "info.json").read_text())["total_episodes"] == 4
    assert json.loads((tmp_path / "subset" / "meta" / "info.json").read_text())["total_episodes"] == 1


def test_subset_refuses_an_unknown_metadata_layout(tmp_path, recording):
    source = _write_raw_lerobot(tmp_path / "src", recording, (0, 1))
    (source / "meta" / "info.json").unlink()
    with pytest.raises(SystemExit, match="standard meta/ layout"):
        tr.build_lerobot_subset(source, tmp_path / "subset", [0])


def test_subset_refuses_when_metadata_does_not_describe_the_selection(tmp_path, recording):
    source = _write_raw_lerobot(tmp_path / "src", recording, (0, 1))
    # The parquet is there; the metadata is not. Passing this through would hand the trainer a
    # directory it can read and a manifest that does not mention half of it.
    (source / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": NUM_SAMPLES}) + "\n"
    )
    with pytest.raises(SystemExit, match="would train on a different set"):
        tr.build_lerobot_subset(source, tmp_path / "subset", [0, 1])


def test_subset_refuses_a_missing_parquet(tmp_path, recording):
    source = _write_raw_lerobot(tmp_path / "src", recording, (0,))
    with pytest.raises(SystemExit, match="is not the source of this split"):
        tr.build_lerobot_subset(source, tmp_path / "subset", [0, 5])


def test_subset_rebuild_is_idempotent(tmp_path, recording):
    source = _write_raw_lerobot(tmp_path / "src", recording, (0, 1))
    first = tr.build_lerobot_subset(source, tmp_path / "subset", [0, 1])
    second = tr.build_lerobot_subset(source, tmp_path / "subset", [0])
    assert first["episodes"] == 2 and second["episodes"] == 1
    linked = list((tmp_path / "subset" / "data" / "chunk-000").glob("*.parquet"))
    assert len(linked) == 1, "a rebuild left the previous selection behind"


# --------------------------------------------------- the subset's normalization statistics
#
# generate_subset_stats shells out to upstream's gr00t/data/stats.py, which needs the vendored
# tree and a working gr00t install. What is OURS to get right is the refusal set, and every one of
# these failures is otherwise discovered after a GPU has been allocated. The fake vendor tree is a
# real directory with a real script in it, so the tests exercise the actual subprocess path rather
# than a mock of it.


def _fake_vendor(root: Path, body: str) -> Path:
    """A vendor tree whose gr00t/data/stats.py is ``body``, plus an importable embodiment enum.

    The enum is real rather than stubbed away because ``generate_subset_stats`` asks the vendored
    tree to resolve the tag before it shells out, and the whole point of that step is that name
    and value differ. ``XDOF`` is included precisely because ``"xdof_relative_eef_relative_joint"
    .upper()`` is not its name — the shortcut this indirection exists to rule out.
    """
    data = root / "gr00t" / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "stats.py").write_text(body)
    (root / "gr00t" / "__init__.py").write_text("")
    (data / "__init__.py").write_text("")
    (data / "embodiment_tags.py").write_text(
        "from enum import Enum\n"
        "class EmbodimentTag(Enum):\n"
        "    NEW_EMBODIMENT = 'new_embodiment'\n"
        "    XDOF = 'xdof_relative_eef_relative_joint'\n"
        "    @classmethod\n"
        "    def resolve(cls, s):\n"
        "        for m in cls:\n"
        "            if s.lower() in (m.name.lower(), m.value.lower()):\n"
        "                return m\n"
        "        raise ValueError(s)\n"
    )
    return root


def test_the_tag_is_resolved_to_the_enum_name_upstream_stats_demands(tmp_path):
    """gr00t/data/stats.py types embodiment_tag as the ENUM, so tyro offers enum NAMES and rejects
    'new_embodiment' outright. launch_finetune.py types it as str and resolves either spelling.
    One concept, two accepted spellings — the driver absorbs that so the operator never has to."""
    vendor = _fake_vendor(tmp_path / "vendor", "pass\n")
    assert tr.resolve_embodiment_tag(vendor, "new_embodiment") == "NEW_EMBODIMENT"
    assert tr.resolve_embodiment_tag(vendor, "NEW_EMBODIMENT") == "NEW_EMBODIMENT"


def test_the_tag_is_resolved_by_the_enum_and_not_by_uppercasing(tmp_path):
    """The shortcut that looks correct and is not: XDOF's value is not the shout-case of its name,
    so `.upper()` would invent XDOF_RELATIVE_EEF_RELATIVE_JOINT and be rejected."""
    vendor = _fake_vendor(tmp_path / "vendor", "pass\n")
    assert tr.resolve_embodiment_tag(vendor, "xdof_relative_eef_relative_joint") == "XDOF"


def test_an_unknown_tag_is_refused_by_the_vendored_enum(tmp_path):
    vendor = _fake_vendor(tmp_path / "vendor", "pass\n")
    with pytest.raises(SystemExit, match="not an embodiment tag"):
        tr.resolve_embodiment_tag(vendor, "definitely_not_a_robot")


def test_stats_refuse_a_vendor_tree_without_the_generator(tmp_path):
    (tmp_path / "subset").mkdir()
    (tmp_path / "config.py").write_text("")
    with pytest.raises(SystemExit, match="does not carry upstream"):
        tr.generate_subset_stats(
            tmp_path / "subset",
            vendor_root=tmp_path / "vendor",
            embodiment_tag="new_embodiment",
            modality_config=tmp_path / "config.py",
        )


def test_stats_refuse_a_missing_modality_config(tmp_path):
    """A custom tag is absent from MODALITY_CONFIGS until a config registers it, and upstream's
    generator raises rather than writing a partial set. Catching it here names the actual file."""
    vendor = _fake_vendor(tmp_path / "vendor", "raise SystemExit(0)\n")
    (tmp_path / "subset").mkdir()
    with pytest.raises(SystemExit, match="MODALITY_CONFIGS registry"):
        tr.generate_subset_stats(
            tmp_path / "subset",
            vendor_root=vendor,
            embodiment_tag="new_embodiment",
            modality_config=tmp_path / "absent.py",
        )


def test_stats_refuse_a_generator_that_fails(tmp_path):
    vendor = _fake_vendor(tmp_path / "vendor", "import sys; sys.exit(3)\n")
    (tmp_path / "subset").mkdir()
    (tmp_path / "config.py").write_text("")
    with pytest.raises(SystemExit, match="exited 3"):
        tr.generate_subset_stats(
            tmp_path / "subset",
            vendor_root=vendor,
            embodiment_tag="new_embodiment",
            modality_config=tmp_path / "config.py",
        )


def test_stats_refuse_a_generator_that_exits_zero_writing_nothing(tmp_path):
    """The one that would otherwise get through. generate_rel_stats returns EARLY and writes no
    relative_stats.json when the registered config carries no action_configs — exit code 0, no
    file, and the failure surfaces as a normalization error thousands of GPU-seconds later."""
    vendor = _fake_vendor(tmp_path / "vendor", "pass\n")
    subset = tmp_path / "subset"
    (subset / "meta").mkdir(parents=True)
    (subset / "meta" / "stats.json").write_text("{}")  # only the first of the two
    (tmp_path / "config.py").write_text("")
    with pytest.raises(SystemExit, match="returns early"):
        tr.generate_subset_stats(
            subset,
            vendor_root=vendor,
            embodiment_tag="new_embodiment",
            modality_config=tmp_path / "config.py",
        )


def test_stats_report_both_files_when_the_generator_writes_them(tmp_path):
    vendor = _fake_vendor(
        tmp_path / "vendor",
        "import pathlib, sys\n"
        "meta = pathlib.Path(sys.argv[sys.argv.index('--dataset-path') + 1]) / 'meta'\n"
        "meta.mkdir(parents=True, exist_ok=True)\n"
        "(meta / 'stats.json').write_text('{\"a\": 1}')\n"
        "(meta / 'relative_stats.json').write_text('{\"b\": 2}')\n",
    )
    subset = tmp_path / "subset"
    subset.mkdir()
    (tmp_path / "config.py").write_text("")
    written = tr.generate_subset_stats(
        subset,
        vendor_root=vendor,
        embodiment_tag="new_embodiment",
        modality_config=tmp_path / "config.py",
    )
    assert written["embodiment_tag"] == "new_embodiment"
    assert written["stats.json"] > 0 and written["relative_stats.json"] > 0


def test_relative_paths_survive_the_cwd_change_into_the_vendored_tree(tmp_path, monkeypatch):
    """The subprocess runs with ``cwd=vendor_root``, so every relative path means something else
    inside it. A relative ``--vendor-root`` doubles into ``vendor/vendor/gr00t/data/stats.py``
    and the run dies with a bare 'exited 2'; a relative ``--dataset-path`` is worse, because it
    would resolve against the vendored tree without complaint if a same-named directory existed
    there. Found on the workstation 2026-08-16 and invisible on the cluster, where the sbatch
    builds ${PROJ}-rooted absolute paths throughout — which is exactly why it needs a test."""
    _fake_vendor(
        tmp_path / "vendor",
        "import pathlib, sys\n"
        "meta = pathlib.Path(sys.argv[sys.argv.index('--dataset-path') + 1]) / 'meta'\n"
        "meta.mkdir(parents=True, exist_ok=True)\n"
        "(meta / 'stats.json').write_text('{\"a\": 1}')\n"
        "(meta / 'relative_stats.json').write_text('{\"b\": 2}')\n",
    )
    (tmp_path / "subset").mkdir()
    (tmp_path / "config.py").write_text("")

    # A decoy at the path the doubled/mis-resolved lookup would land on, so the test fails on a
    # WRONG answer rather than only on a missing-file refusal.
    (tmp_path / "vendor" / "subset").mkdir()

    monkeypatch.chdir(tmp_path)
    written = tr.generate_subset_stats(
        Path("subset"),
        vendor_root=Path("vendor"),
        embodiment_tag="new_embodiment",
        modality_config=Path("config.py"),
    )
    assert written["stats.json"] > 0 and written["relative_stats.json"] > 0
    assert (tmp_path / "subset" / "meta" / "stats.json").is_file()
    assert not (tmp_path / "vendor" / "subset" / "meta").exists(), (
        "the stats landed inside the vendored tree — the path resolved against cwd=vendor_root"
    )


# ------------------------------------------------------------- the witness, end to end


_MOCK_D1 = _REPO_ROOT / "datasets" / "mock-d1"


@pytest.mark.skipif(not _MOCK_D1.is_dir(), reason="datasets/mock-d1 not present")
def test_the_witness_this_trainer_writes_passes_the_evaluator_split_proof(tmp_path):
    """The one contract the two drivers must share, tested across both of them.

    ``train_t39_baseline`` writes ``run_metadata.json``; ``eval_t39_baseline.load_witness`` reads
    it; ``eval_t16.verify_split`` — the SAME function every other run in this repo is proven with
    — has to accept it on the disjointness path against the external witness file. Nothing about
    T-39 is scorable if this link is broken, and neither script can catch it alone.
    """
    from wam.data.episode import list_episodes

    episodes = list_episodes(_MOCK_D1)
    holdout_ids = {p.name for p in episodes[-2:]}
    train_dirs = [p for p in episodes if p.name not in holdout_ids][:3]
    train_ids = [p.name for p in train_dirs]

    holdout_file = tmp_path / "holdout.txt"
    holdout_file.write_text("\n".join(sorted(holdout_ids)) + "\n")
    train_file = tmp_path / "train.txt"
    train_file.write_text("# a rung\n\n" + "\n".join(train_ids) + "\n")

    resolved = tr.resolve_training_episodes(_MOCK_D1, set(train_ids), holdout_ids)
    assert [p.name for p in resolved] == train_ids

    snapshot = eval_t16.dataset_snapshot_hash(_MOCK_D1, resolved)
    run_dir = tmp_path / "t39-baseline-seed0"
    run_dir.mkdir()
    tr.write_witness(
        run_dir,
        run_id="t39-baseline-seed0",
        model_id="vendor/Model-Under-Test",
        trained_ids=[p.name for p in resolved],
        snapshot_ref=snapshot,
        config={"task": "T-39"},
    )

    witness = ev.load_witness(run_dir)
    assert witness.checkpoint_ref == "vendor/Model-Under-Test"
    assert list(witness.train_episode_ids) == train_ids

    proven = eval_t16.verify_split(
        _MOCK_D1,
        holdout_ids,
        witness.dataset_snapshot_ref,
        witness.train_episode_ids,
        set(train_ids),
    )
    assert {p.name for p in proven} == holdout_ids


@pytest.mark.skipif(not _MOCK_D1.is_dir(), reason="datasets/mock-d1 not present")
def test_the_split_proof_refuses_a_witness_that_lost_an_episode(tmp_path):
    """The witness has to be checked against the committed file, not against itself.

    ``eval_t16.verify_split``'s docstring records the measured hole this closes: a checkpoint that
    declared one training episode out of eight printed "split proven" and handed back episodes it
    had trained on. Re-pinned here from the T-39 side, because T-39 is the first run in this repo
    whose witness is written by a driver rather than by the trainer that produced the weights.
    """
    from wam.data.episode import list_episodes

    episodes = list_episodes(_MOCK_D1)
    holdout_ids = {p.name for p in episodes[-2:]}
    train_dirs = [p for p in episodes if p.name not in holdout_ids][:3]

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tr.write_witness(
        run_dir,
        run_id="t39",
        model_id="vendor/Model",
        trained_ids=[p.name for p in train_dirs],
        snapshot_ref=eval_t16.dataset_snapshot_hash(_MOCK_D1, train_dirs),
        config={},
    )
    witness = ev.load_witness(run_dir)
    with pytest.raises(SystemExit, match="does not describe this checkpoint"):
        eval_t16.verify_split(
            _MOCK_D1,
            holdout_ids,
            witness.dataset_snapshot_ref,
            witness.train_episode_ids,
            {p.name for p in train_dirs[:2]},  # the committed file names one fewer
        )


@pytest.mark.skipif(not _MOCK_D1.is_dir(), reason="datasets/mock-d1 not present")
def test_training_refuses_a_split_file_that_names_a_holdout_episode(tmp_path):
    """Caught at the PRODUCING end, not only at scoring time.

    If a leak is only caught by the evaluator it has already reached a checkpoint, and the
    evaluator is then the single thing standing between it and a published number.
    """
    from wam.data.episode import list_episodes

    episodes = list_episodes(_MOCK_D1)
    holdout_ids = {p.name for p in episodes[-2:]}
    leaking = {p.name for p in episodes[:2]} | {min(holdout_ids)}
    with pytest.raises(SystemExit, match="names 1 holdout episode"):
        tr.resolve_training_episodes(_MOCK_D1, leaking, holdout_ids)


def test_load_witness_refuses_a_run_without_one(tmp_path):
    with pytest.raises(SystemExit, match="cannot be PROVEN unseen"):
        ev.load_witness(tmp_path)
