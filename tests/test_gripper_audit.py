"""Tests for wam.evaluation.gripper: the dataset-level grasp audit (T-31).

The pair ``test_the_shipped_gr00t_gripper_mapping_fails_the_audit`` /
``test_the_same_hand_passes_when_the_dead_hand_is_not_averaged_in`` is the whole argument of the
item as an executable test: the same hand joints, two mappings, opposite verdicts.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wam.data.episode import EpisodeReader, EpisodeWriter
from wam.evaluation.gripper import (
    GRIPPER_HYSTERESIS_MARGIN,
    GRIPPER_MIN_DYNAMIC_RANGE,
    GRIPPER_MIN_EPISODES_WITH_GRASP_CYCLE,
    GRIPPER_MIN_EPISODES_WITH_TRANSITION,
    GRIPPER_MIN_TRANSITIONS_PER_EPISODE,
    GripperAuditReport,
    _pick_scored,
    apply_affine,
    audit_lerobot_dataset,
    audit_wam_dataset,
    channel_stats,
    dataset_affine,
    debounced_transitions,
    expected_saturated_frac,
)
from wam.interfaces.schema import (
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    IMUState,
    RobotState,
    ValidityMask,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "audit_gripper.py"

FPS = 30.0
DT = 1.0 / FPS


def _load_converter():
    spec = importlib.util.spec_from_file_location(
        "convert_lerobot_g1", _REPO_ROOT / "scripts" / "convert_lerobot_g1.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


conv = _load_converter()

# Per-joint peak-to-peak of the LEFT Dex3 hand in data/raw/gr00t_apple episode 0, rad. Only four
# of the seven joints carry the grasp — which is the first half of why averaging all seven
# attenuates it. Used to shape the fixture so the test fails for the same reason the real data
# does, not merely for a reason that resembles it.
LEFT_HAND_P2P = np.array([0.003, 0.158, 0.380, 0.690, 0.826, 0.590, 0.621], dtype=np.float32)
LEFT_HAND_REST = -0.40
RIGHT_HAND_REST = 0.036


def _trapezoid_gripper(n: int = 96) -> np.ndarray:
    """One close-then-open with real transit samples, in gripper units, touching no rail.

    A square wave has no transit samples at all, which makes it indistinguishable from a
    two-state command and useless for reasoning about clipping.
    """
    q = n // 4
    return np.concatenate(
        [
            np.full(q, 0.02),
            np.linspace(0.02, 0.98, q),
            np.full(q, 0.98),
            np.linspace(0.98, 0.02, n - 3 * q),
        ]
    )


def _clip_with_narrow_affine(values: np.ndarray) -> np.ndarray:
    """``values`` remapped by an affine fitted on the middle half of its range, then clipped.

    This is what a pinned or foreign affine does to a channel it does not cover: both dwell
    phases land on a rail and the range, crossings and transitions all grow.
    """
    return np.clip((np.asarray(values, dtype=np.float64) - 0.25) / 0.5, 0.0, 1.0)


def grasp_state_sequence(steps: int = 120, seed: int = 0) -> np.ndarray:
    """A [steps, 43] GR00T-layout state block: one close-then-open grasp on the LEFT hand only.

    The right hand is frozen at its rest posture, exactly as in the source snapshot — that is the
    second half of the mapping bug and it has to be in the fixture or the both-hand mean looks
    harmless.
    """
    rng = np.random.default_rng(seed)
    closure = np.zeros(steps, dtype=np.float32)
    closure[steps // 4 : steps // 2] = np.linspace(0.0, 1.0, steps // 4, dtype=np.float32)
    closure[steps // 2 : 3 * steps // 4] = 1.0
    closure[3 * steps // 4 :] = np.linspace(1.0, 0.0, steps - 3 * steps // 4, dtype=np.float32)

    state = rng.normal(0.0, 0.01, size=(steps, 43)).astype(np.float32)
    state[:, 29:36] = LEFT_HAND_REST + closure[:, None] * LEFT_HAND_P2P[None, :]
    state[:, 36:43] = RIGHT_HAND_REST
    return state


def test_debounced_transitions_ignores_jitter_inside_the_margin() -> None:
    """The one mechanism that separates a real grasp from dithering on the threshold."""
    steps = 200
    jitter = 0.5 + 0.05 * np.sin(np.arange(steps) * 1.7)
    assert debounced_transitions(jitter) == 0
    # ... while the undebounced count is large, which is what made a dead channel look busy.
    from wam.evaluation.gripper import crossings

    assert crossings(jitter) > 10

    square = np.where(np.arange(steps) % 40 < 20, 0.1, 0.9)
    # 200 samples, period 40 -> 4 interior edges after the first latch.
    assert debounced_transitions(square) == 9

    assert debounced_transitions(np.full(steps, 0.5)) == 0
    assert debounced_transitions(np.zeros(1)) == 0
    # A jump that only reaches the band edge does not latch: the margin is exclusive of nothing,
    # it is the level that must be REACHED (>= thr + margin).
    edge = np.array([0.5 - GRIPPER_HYSTERESIS_MARGIN, 0.5 + GRIPPER_HYSTERESIS_MARGIN])
    assert debounced_transitions(edge) == 1


def test_the_shipped_gr00t_gripper_mapping_fails_the_audit() -> None:
    """Pins the CAUSE, not the symptom: synergy attenuation, then averaging in a dead hand."""
    episodes = [grasp_state_sequence(seed=i) for i in range(6)]
    q = [conv.canonical_q(s) for s in episodes]
    series = []
    for state, q_i in zip(episodes, q):
        grip = conv.gripper_state(state)  # legacy: clip((mean(hand_7dof)+1)/2, 0, 1)
        chunks = conv.relabel_chunks(q_i, grip, chunk_steps=16, dt_s=DT)  # both-hand mean
        series.append(np.concatenate([np.asarray(c.gripper_target) for c, _ in chunks]))

    stats = channel_stats("legacy", series)
    assert stats.p2p_global < GRIPPER_MIN_DYNAMIC_RANGE
    assert stats.debounced_transitions_per_episode == 0.0
    assert not stats.admissible
    assert any("dynamic range" in r for r in stats.failed_clauses())

    # The signal is not absent, it is attenuated, in the two documented steps — each exactly a
    # halving, which is why the composite loss is 4x and why fixing only one of them is not enough.
    raw_p2p = float(np.ptp(conv.raw_synergy(episodes[0][:, 29:36])))
    synergy_p2p = float(np.ptp(conv.hand_synergy(episodes[0][:, 29:36])))
    assert synergy_p2p == pytest.approx(raw_p2p / 2, rel=1e-5)  # (mean + 1) / 2
    assert stats.p2p_per_episode_mean == pytest.approx(raw_p2p / 4, rel=1e-3)  # + the dead hand


def test_the_same_hand_passes_when_the_dead_hand_is_not_averaged_in() -> None:
    """Identical joints, the active-hand mapping: two debounced transitions in every episode."""
    episodes = [grasp_state_sequence(seed=i) for i in range(6)]
    affine = conv.fit_hand_affine(episodes)
    assert affine.active == "left"
    assert affine.p2p_left > 100 * affine.p2p_right

    series = []
    for state in episodes:
        grip = conv.gripper_state(state, affine)
        chunks = conv.relabel_chunks(
            conv.canonical_q(state), grip, chunk_steps=16, dt_s=DT, gripper_column=0
        )
        series.append(np.concatenate([np.asarray(c.gripper_target) for c, _ in chunks]))

    stats = channel_stats("active-hand", series)
    assert stats.admissible, stats.failed_clauses()
    assert stats.debounced_transitions_per_episode >= 2.0
    assert stats.episodes_with_transition_frac == 1.0


def test_a_channel_that_only_ever_closes_is_refused_although_the_mean_clause_clears_it() -> None:
    """Audit 0.1.0's hole, verbatim: a monotone ramp is not a grasp and used to be admissible.

    50 episodes of 0 -> 1 that never reopen in ANY episode score p2p 1.00, exactly 1.00 debounced
    transitions per episode and 1.00 episodes-with-a-transition — three clauses cleared, zero
    failures, while the dataset contains no close-and-release anywhere. The mean clause tolerates
    partial episodes by averaging, and (1, 1, 1, ...) averages to the same 1.0 as (2, 0, 2, 0).
    """
    ramp = [np.linspace(0.0, 1.0, 120) for _ in range(50)]
    stats = channel_stats("monotone", ramp)

    assert stats.p2p_global == pytest.approx(1.0)
    assert stats.debounced_transitions_per_episode == pytest.approx(1.0)
    assert stats.episodes_with_transition_frac == pytest.approx(1.0)
    assert stats.p2p_global >= GRIPPER_MIN_DYNAMIC_RANGE
    assert stats.debounced_transitions_per_episode >= GRIPPER_MIN_TRANSITIONS_PER_EPISODE
    assert stats.episodes_with_transition_frac >= GRIPPER_MIN_EPISODES_WITH_TRANSITION

    assert stats.episodes_with_grasp_cycle_frac == 0.0
    assert not stats.admissible
    assert [("complete grasp" in c) for c in stats.failed_clauses()] == [True]


def test_the_grasp_cycle_clause_still_tolerates_half_the_episodes_being_partial() -> None:
    """The relaxation the mean clause was written for must survive the clause that fixes it.

    ``GRIPPER_MIN_TRANSITIONS_PER_EPISODE = 1.0`` exists to tolerate approach-only and truncated
    recordings. Stating that same tolerance per-episode must not quietly demand a grasp in every
    episode — exactly half carrying one is the boundary it was derived from, and it passes.
    """
    full = [_trapezoid_gripper() for _ in range(10)]  # closes and reopens
    partial = [np.linspace(0.02, 0.98, 48) for _ in range(10)]  # closes only, never releases
    stats = channel_stats("half-partial", [*full, *partial])

    assert stats.episodes_with_grasp_cycle_frac == pytest.approx(
        GRIPPER_MIN_EPISODES_WITH_GRASP_CYCLE
    )
    assert stats.episodes_with_transition_frac == pytest.approx(1.0)
    assert stats.admissible, stats.failed_clauses()


def test_a_clipped_channel_is_named_in_reasons_even_though_the_report_passes(
    tmp_path: Path,
) -> None:
    """Clipping moves every clause the PASSING way, so it must be visible next to a PASS.

    The channel here is a clean close-then-open pushed through an affine fitted on the middle
    half of its range, so both dwell phases land on a rail. Before 0.2.0 the only trace was two
    table columns that reached neither ``failed_clauses`` nor ``reasons``.
    """
    clipped = _clip_with_narrow_affine(_trapezoid_gripper())
    for i in range(4):
        _write_wam_episode(tmp_path, f"ep{i:03d}", np.stack([clipped, clipped], axis=1))

    report = audit_wam_dataset(tmp_path)
    assert report.passed  # every clause is cleared — that is exactly the problem
    saturation = [r for r in report.reasons if "sit exactly on a [0, 1] rail" in r]
    assert saturation, report.reasons
    assert all(r.startswith(("action.gripper_target:", "state.gripper[")) for r in saturation)
    assert "NOTE (not gated)" in saturation[0]
    markdown = report.render_markdown()
    assert "## Findings" in markdown
    # The verdict line is still the clause verdict; a notice must not read as a failure.
    assert "All four clauses cleared" in markdown


def test_the_saturation_reference_comes_from_the_min_max_fit_and_not_from_a_measurement() -> None:
    """One sample per rail per episode is the loosest a dataset-level min-max fit can be."""
    assert expected_saturated_frac(num_episodes=402, num_steps=171625) == pytest.approx(
        2 * 402 / 171625
    )
    assert expected_saturated_frac(num_episodes=0, num_steps=0) == 0.0

    # A channel mapped by its OWN dataset affine saturates exactly its two extremal samples per
    # episode and must never raise the note; one fitted on a subset does.
    episodes = [np.linspace(-1.0 + i, 1.0 + i, 100) for i in range(5)]
    own_offset, own_span = dataset_affine(episodes)
    clean = channel_stats("own-fit", [apply_affine(s, own_offset, own_span) for s in episodes])
    assert not clean.clipping_suspected
    assert clean.notices() == ()

    subset_offset, subset_span = dataset_affine(episodes[:1])
    borrowed = channel_stats(
        "foreign-fit", [apply_affine(s, subset_offset, subset_span) for s in episodes]
    )
    assert borrowed.clipping_suspected
    assert len(borrowed.notices()) == 1


def test_a_clipped_channel_does_not_take_the_scored_slot_from_a_clean_one() -> None:
    """A PASS must be explained by the channel whose numbers are not inflated by the clip."""
    clean = channel_stats("clean", [_trapezoid_gripper() for _ in range(4)])
    # The same events, clipped: strictly more range and no fewer transitions, so it wins on
    # every key _pick_scored had before 0.2.0.
    clipped = channel_stats(
        "clipped", [_clip_with_narrow_affine(_trapezoid_gripper()) for _ in range(4)]
    )
    assert clipped.p2p_global > clean.p2p_global
    assert clipped.debounced_transitions_per_episode >= clean.debounced_transitions_per_episode
    assert clipped.clipping_suspected and not clean.clipping_suspected

    assert _pick_scored([clipped, clean]).name == "clean"
    assert _pick_scored([clean, clipped]).name == "clean"


def test_the_gripper_affine_is_dataset_level_not_per_episode() -> None:
    """The same physical aperture must map to the same number in every episode.

    A per-episode min-max is the tempting fix and would make a half-closed hand in a shy episode
    read as fully closed. Fitting one affine over episodes with deliberately different closure
    depths is what catches that.
    """
    deep = grasp_state_sequence(seed=1)
    shy = grasp_state_sequence(seed=2)
    shy[:, 29:36] = LEFT_HAND_REST + (shy[:, 29:36] - LEFT_HAND_REST) * 0.5

    affine = conv.fit_hand_affine([deep, shy])
    deep_max = float(affine.apply(conv.raw_synergy(deep[:, 29:36])).max())
    shy_max = float(affine.apply(conv.raw_synergy(shy[:, 29:36])).max())
    assert deep_max == pytest.approx(1.0, abs=1e-6)
    assert shy_max < 0.6  # a shallower grasp stays a shallower number

    # Per-episode normalisation would have made both exactly 1.0 — check the helper the audit
    # uses is the dataset-level one.
    offset, span = dataset_affine(
        [conv.raw_synergy(deep[:, 29:36]), conv.raw_synergy(shy[:, 29:36])]
    )
    assert (offset, span) == pytest.approx((affine.offset, affine.span), abs=1e-6)


# --- converted WAM datasets ---------------------------------------------------------------------


def _write_wam_episode(
    root: Path, name: str, gripper: np.ndarray, *, with_frames: bool = True
) -> None:
    """Minimal WAM episode whose gripper_state is ``gripper`` [n, 2] and target = column 0."""
    spec = CanonicalSpaceSpec(joint_names=("j0", "j1"), gripper_dims=gripper.shape[1])
    n = gripper.shape[0]
    zero_imu = IMUState(
        orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
        angular_velocity=np.zeros(3, dtype=np.float32),
        linear_acceleration=np.zeros(3, dtype=np.float32),
    )
    with EpisodeWriter(root / name, name, spec, FPS, "grasp") as writer:
        for t in range(n):
            ts = int(t * DT * 1e9)
            if with_frames:
                writer.add_frame("ego", np.full((8, 8, 3), t % 255, dtype=np.uint8), ts)
            writer.add_state(
                RobotState(
                    timestamp_ns=ts,
                    q=np.zeros(2, dtype=np.float32),
                    dq=np.zeros(2, dtype=np.float32),
                    imu=zero_imu,
                    gripper_state=gripper[t],
                    validity=ValidityMask(q=True, dq=True, imu=False, gripper=True),
                )
            )
        for start in range(0, n - 8, 8):
            writer.add_action(
                ActionChunk(
                    mode=ActionMode.JOINT_DELTA,
                    targets=np.zeros((8, 2), dtype=np.float32),
                    gripper_target=gripper[start : start + 8, 0].astype(np.float32),
                    dt_s=DT,
                ),
                executed_prefix=8,
                timestamp_ns=int(start * DT * 1e9),
            )


def _square_gripper(n: int = 96) -> np.ndarray:
    live = np.where((np.arange(n) // 24) % 2 == 0, 0.05, 0.95).astype(np.float32)
    return live


def test_audit_reports_each_state_gripper_column_separately(tmp_path: Path) -> None:
    """A live hand next to a frozen one must stay visible; averaging them is what hid the bug."""
    live = _square_gripper()
    frozen = np.full_like(live, 0.52)
    for i in range(3):
        _write_wam_episode(tmp_path, f"ep{i:03d}", np.stack([live, frozen], axis=1))

    report = audit_wam_dataset(tmp_path)
    assert report.num_episodes == 3
    assert [c.name for c in report.channels] == [
        "action.gripper_target",
        "state.gripper[0]",
        "state.gripper[1]",
    ]
    assert report.channel("state.gripper[0]").p2p_global == pytest.approx(0.9, abs=1e-6)
    assert report.channel("state.gripper[1]").p2p_global == pytest.approx(0.0, abs=1e-6)
    assert report.channel("state.gripper[1]").debounced_transitions_per_episode == 0.0
    assert report.passed  # the scored channel is the live one
    assert report.source_kind == "wam"


def test_audit_fails_and_names_every_broken_clause(tmp_path: Path) -> None:
    dead = np.full(96, 0.48, dtype=np.float32)
    for i in range(3):
        _write_wam_episode(tmp_path, f"ep{i:03d}", np.stack([dead, dead], axis=1))

    report = audit_wam_dataset(tmp_path)
    assert not report.passed
    assert len(report.reasons) == 4
    assert any("dynamic range" in r for r in report.reasons)
    assert any("debounced transitions" in r for r in report.reasons)
    assert any("episodes with a transition" in r for r in report.reasons)
    assert any("complete grasp" in r for r in report.reasons)
    restored = GripperAuditReport.from_json(report.to_json())
    assert restored == report
    assert "FAIL" in report.render_markdown()


def test_audit_never_decodes_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins the cost contract: no ffmpeg, no GPU, so the gate stays runnable over 402 episodes."""
    live = _square_gripper()
    for i in range(2):
        _write_wam_episode(tmp_path, f"ep{i:03d}", np.stack([live, live], axis=1))

    def explode(self: EpisodeReader, camera: str) -> np.ndarray:
        raise AssertionError("the gripper audit must never decode video")

    monkeypatch.setattr(EpisodeReader, "read_frames", explode)
    report = audit_wam_dataset(tmp_path)
    assert report.num_episodes == 2


# --- raw LeRobot snapshots ----------------------------------------------------------------------


def _write_fake_lerobot(root: Path, hand_start: int, dim: int, episodes: int = 4) -> None:
    """A LeRobot snapshot whose hand group sits at a NON-default index, declared in modality.json."""
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    n = 96
    closure = np.where((np.arange(n) // 24) % 2 == 0, 0.0, 1.0).astype(np.float32)
    for ep in range(episodes):
        state = np.zeros((n, dim), dtype=np.float32)
        state[:, hand_start : hand_start + 7] = -0.4 + closure[:, None] * LEFT_HAND_P2P[None, :]
        pq.write_table(
            pa.table(
                {
                    "observation.state": pa.array(list(state), type=pa.list_(pa.float32())),
                    "action": pa.array(list(state), type=pa.list_(pa.float32())),
                }
            ),
            data_dir / f"episode_{ep:06d}.parquet",
        )
    meta = root / "meta"
    meta.mkdir(exist_ok=True)
    groups = {"left_hand": {"start": hand_start, "end": hand_start + 7}}
    (meta / "modality.json").write_text(
        json.dumps(
            {
                "state": groups,
                "action": {
                    **groups,
                    # A group that lives in its own column: its indices are offsets into THAT
                    # column, so slicing the packed vector with them would read the wrong joints.
                    "effort_left_hand": {
                        "original_key": "action.effort_left_hand",
                        "start": 0,
                        "end": 7,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_lerobot_mode_reads_hand_slices_from_modality_json(tmp_path: Path) -> None:
    """The raw mode must not be silently tied to the 43-dim AppleToPlate layout."""
    _write_fake_lerobot(tmp_path, hand_start=5, dim=20)
    report = audit_lerobot_dataset(tmp_path)

    assert report.source_kind == "lerobot"
    assert report.passed, report.reasons
    names = {c.name for c in report.channels}
    # The chosen joint is part of the channel name: a reader must be able to see WHICH physical
    # joint the numbers are about, because "the most active one" is not a column.
    assert names == {
        "action.left_hand.max_joint[4]",
        "action.left_hand.mean",
        "state.left_hand.max_joint[4]",
        "state.left_hand.mean",
    }
    # effort_left_hand carries an original_key and must be skipped, not sliced out of `action`.
    assert not any("effort" in n for n in names)
    assert report.channel("state.left_hand.mean").debounced_transitions_per_episode >= 2.0


def _write_two_joint_lerobot(root: Path, hand_start: int = 5, dim: int = 20) -> None:
    """A snapshot where the per-episode most active joint is NOT the set's most active joint.

    Joint 3 is a one-way ramp that only moves in episode 0, over twice the range of anything
    else; joint 5 carries the actual close-and-open in all four episodes. Picking per episode
    reports joint 3 for episode 0 and joint 5 for the rest, i.e. one channel made of two
    different physical joints, and then fits one affine over the concatenation.
    """
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    n = 96
    open_close = _trapezoid_gripper(n)  # range ~0.96, two debounced transitions once mapped
    one_way = np.linspace(0.0, 2.0, n)  # range 2.0, one transition, episode 0 only
    for ep in range(4):
        state = np.zeros((n, dim), dtype=np.float32)
        state[:, hand_start + 5] = open_close
        if ep == 0:
            state[:, hand_start + 3] = one_way
        pq.write_table(
            pa.table(
                {"observation.state": pa.array(list(state), type=pa.list_(pa.float32()))}
            ),
            data_dir / f"episode_{ep:06d}.parquet",
        )
    meta = root / "meta"
    meta.mkdir(exist_ok=True)
    (meta / "modality.json").write_text(
        json.dumps({"state": {"hand": {"start": hand_start, "end": hand_start + 7}}}),
        encoding="utf-8",
    )


def test_the_active_joint_is_chosen_once_over_the_set_not_once_per_episode(
    tmp_path: Path,
) -> None:
    """A channel must be ONE physical joint; per-episode selection concatenates two.

    Mean per-episode peak-to-peak over the set: joint 3 = 2.0/4 = 0.5, joint 5 = 0.96, so joint 5
    is the set's active joint. Inside episode 0 joint 3 wins on 2.0 > 0.96, which is what the
    per-episode ``argmax`` picked — and the dataset-level affine then stretched to joint 3's
    range, pushing joint 5's grasp below the latch level in the other three episodes.
    """
    _write_two_joint_lerobot(tmp_path)
    report = audit_lerobot_dataset(tmp_path)

    channel = report.channel("state.hand.max_joint[5]")
    assert channel.num_episodes == 4
    assert channel.debounced_transitions_per_episode == pytest.approx(2.0)
    assert channel.episodes_with_grasp_cycle_frac == pytest.approx(1.0)
    assert channel.p2p_per_episode_min == pytest.approx(1.0, abs=1e-6)
    assert not any("max_joint[3]" in c.name for c in report.channels)


def test_lerobot_mode_without_modality_json_says_the_layout_is_unverified(tmp_path: Path) -> None:
    _write_fake_lerobot(tmp_path, hand_start=29, dim=43)
    (tmp_path / "meta" / "modality.json").unlink()

    report = audit_lerobot_dataset(tmp_path)
    assert report.source_kind == "lerobot(unverified-layout)"
    assert report.passed  # the fallback slices happen to be right here — but it says so


def test_script_exits_nonzero_when_the_dataset_cannot_support_a_grasp_claim(
    tmp_path: Path,
) -> None:
    """The exit code is what makes this a gate a runbook can call, not a report to read."""
    dead = np.full(96, 0.48, dtype=np.float32)
    live = _square_gripper()
    bad, good = tmp_path / "bad", tmp_path / "good"
    for i in range(2):
        _write_wam_episode(bad, f"ep{i:03d}", np.stack([dead, dead], axis=1))
        _write_wam_episode(good, f"ep{i:03d}", np.stack([live, live], axis=1))

    out = tmp_path / "audit.json"
    fail = subprocess.run(
        [sys.executable, str(_SCRIPT), str(bad), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert fail.returncode == 1, fail.stderr
    assert "FAIL" in fail.stdout
    assert GripperAuditReport.from_json(out.read_text(encoding="utf-8")).passed is False

    ok = subprocess.run(
        [sys.executable, str(_SCRIPT), str(good), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout)["passed"] is True


# --- the converter's dataset-level affine ---------------------------------------------------------


def test_the_active_hand_affine_is_a_property_of_the_conversion_set_not_of_the_hand() -> None:
    """The limitation the ``--gripper-affine`` flag exists for, stated as a measurement.

    ``active-hand`` fits over the episodes of one invocation, so the SAME physical aperture is a
    different number in a 6-episode and a 12-episode conversion. Anything comparing two
    conversions — a doc table, a second dataset, a retrain — is comparing two scales unless one
    of them pins the other's affine.
    """
    shy = [grasp_state_sequence(seed=i) for i in range(6)]
    deep = [grasp_state_sequence(seed=i) for i in range(6)]
    for state in deep:  # a second session that closes harder
        state[:, 29:36] = LEFT_HAND_REST + (state[:, 29:36] - LEFT_HAND_REST) * 1.5

    small = conv.fit_hand_affine(shy)
    large = conv.fit_hand_affine([*shy, *deep])
    aperture = np.array([LEFT_HAND_REST + 0.5 * float(LEFT_HAND_P2P.mean())], dtype=np.float32)

    assert small.span != pytest.approx(large.span)
    assert float(small.apply(aperture)[0]) != pytest.approx(float(large.apply(aperture)[0]))
    assert small.fitted_episodes == 6 and large.fitted_episodes == 12

    # Pinning the first fit onto the second set is what makes them agree — and it is refused
    # here, because the second set leaves the first set's range.
    with pytest.raises(ValueError, match="clips"):
        conv.pinned_hand_affine([*shy, *deep], small.offset, small.span)

    pinned = conv.pinned_hand_affine(shy[:3], large.offset, large.span)
    assert pinned.pinned is True
    assert float(pinned.apply(aperture)[0]) == pytest.approx(float(large.apply(aperture)[0]))


def test_a_pinned_affine_that_would_clip_is_refused_and_names_the_alternative() -> None:
    """Clipping is unrecoverable once written, so it is refused where the raw values still exist.

    Zero is not a tuned tolerance: the affine fitted on the set it is applied to clips exactly
    zero samples by construction, so it is the value the alternative achieves.
    """
    episodes = [grasp_state_sequence(seed=i) for i in range(6)]
    narrow = conv.fit_hand_affine(episodes[:1])
    narrow = narrow._replace(offset=narrow.offset + 0.05, span=narrow.span * 0.5)

    with pytest.raises(ValueError) as excinfo:
        conv.pinned_hand_affine(episodes, narrow.offset, narrow.span)
    message = str(excinfo.value)
    assert "clips" in message and "left hand" in message
    fitted = conv.fit_hand_affine(episodes)
    assert f"offset {fitted.offset:.6g}" in message  # the alternative, printed, not implied

    # The set's own fit is always accepted: it clips nothing.
    ok = conv.pinned_hand_affine(episodes, fitted.offset, fitted.span)
    assert (ok.offset, ok.span, ok.pinned) == (fitted.offset, fitted.span, True)


def _write_convertible_source(root: Path, episodes: int = 2, n: int = 96) -> None:
    """A minimal GR00T-layout snapshot the converter can run end to end (parquet + tiny mp4)."""
    import cv2

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir = root / "videos" / "chunk-000" / "observation.images.ego_view"
    video_dir.mkdir(parents=True, exist_ok=True)
    meta = root / "meta"
    meta.mkdir(exist_ok=True)

    for ep in range(episodes):
        state = grasp_state_sequence(steps=n, seed=ep)
        pq.write_table(
            pa.table(
                {
                    "observation.state": pa.array(list(state), type=pa.list_(pa.float32())),
                    "timestamp": pa.array(
                        (np.arange(n) / 30.0).astype(np.float32), type=pa.float32()
                    ),
                }
            ),
            data_dir / f"episode_{ep:06d}.parquet",
        )
        writer = cv2.VideoWriter(
            str(video_dir / f"episode_{ep:06d}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (32, 24),
        )
        for t in range(n):
            writer.write(np.full((24, 32, 3), (3 * t) % 255, dtype=np.uint8))
        writer.release()
        with (meta / "episodes.jsonl").open("a") as f:
            f.write(
                json.dumps({"episode_index": ep, "tasks": ["move the apple to the plate"]}) + "\n"
            )
    (meta / "info.json").write_text(
        json.dumps({"robot_type": "unitree_g1", "codebase_version": "v2.1", "fps": 30}),
        encoding="utf-8",
    )


def _convert(source: Path, out: Path, *extra: str) -> bytes:
    """Run the converter and return episode 0's manifest bytes."""
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
            *extra,
        ]
    )
    assert rc == 0
    return (out / "gr00t-apple-000000" / "manifest.json").read_bytes()


def test_naming_the_dataset_the_default_already_assumes_moves_no_manifest_byte(
    tmp_path: Path,
) -> None:
    """The converter's own WARNING tells you to pass --source-dataset; obeying it must be free.

    ``datasets/gr00t-apple-full``'s manifest BYTES are hashed into ``runs/t16-lora-seed0``'s
    ``dataset_snapshot_ref``, and ``scripts/eval_t16.py`` refuses to score on a mismatch. So
    recording the id the default already assumes has to be a no-op; only an id that actually
    differs (or an explicit --record-provenance) may add the extra provenance block.
    """
    source = tmp_path / "src"
    _write_convertible_source(source)

    default = _convert(source, tmp_path / "a")
    same_id = _convert(source, tmp_path / "b", "--source-dataset", conv.DEFAULT_SOURCE_DATASET)
    assert same_id == default

    other = _convert(source, tmp_path / "c", "--source-dataset", "someone/else")
    assert other != default
    other_meta = json.loads(other)["extra"]["source"]
    assert other_meta["dataset"] == "someone/else"
    assert other_meta["dataset_id_source"] == "--source-dataset"
    assert other_meta["codebase_version"] == "v2.1"  # meta/info.json rides along

    opted_in = _convert(source, tmp_path / "d", "--record-provenance")
    assert opted_in != default
    assert json.loads(opted_in)["extra"]["source"]["dataset_id_source"].startswith(
        "converter default"
    )
    assert "dataset_id_source" not in json.loads(default)["extra"]["source"]


def test_the_active_hand_conversion_records_the_affine_its_episode_count_and_the_clip(
    tmp_path: Path,
) -> None:
    """The mapping has to travel with the data: which affine, fitted on what, clipped how."""
    source = tmp_path / "src"
    _write_convertible_source(source)
    out = tmp_path / "grip"
    _convert(source, out, "--gripper-mapping", "active-hand")

    manifest = json.loads((out / "gr00t-apple-000000" / "manifest.json").read_text())
    mapping = manifest["extra"]["mapping"]
    assert mapping["gripper_mapping"] == "active-hand"
    assert mapping["gripper_clip"] == [0.0, 1.0]
    assert mapping["gripper_affine_source"] == "fitted on these 2 episodes"
    affine = conv.fit_hand_affine(
        [grasp_state_sequence(steps=96, seed=i) for i in range(2)]
    )
    assert manifest["normalization"]["gripper_target"]["mean"] == [pytest.approx(affine.offset)]
    assert manifest["normalization"]["gripper_target"]["std"] == [pytest.approx(affine.span)]

    # Pinning that same affine reproduces the mapping and says the mapping was pinned.
    pinned_out = tmp_path / "grip-pinned"
    _convert(
        source,
        pinned_out,
        "--gripper-mapping",
        "active-hand",
        "--gripper-affine",
        repr(affine.offset),
        repr(affine.span),
    )
    pinned = json.loads((pinned_out / "gr00t-apple-000000" / "manifest.json").read_text())
    assert pinned["extra"]["mapping"]["gripper_affine_source"].startswith("pinned via")
    assert pinned["checksums"] == manifest["checksums"]  # identical mapping -> identical data


def test_to_spec_is_the_pre_clip_affine_and_says_so() -> None:
    """NormalizationSpec has no clip field, so the recorded spec is not invertible at the rails."""
    affine = conv.fit_hand_affine([grasp_state_sequence(seed=i) for i in range(4)])
    spec = affine.to_spec(2)

    assert spec.mean == (affine.offset,) * 2
    assert spec.std == (affine.span,) * 2
    assert "PRE-CLIP" in conv.HandAffine.to_spec.__doc__

    # Outside the fitted range the stored channel is clipped, so inverting the spec does NOT
    # return the raw synergy — which is exactly why the clip is recorded separately.
    outside = np.array([affine.offset - affine.span], dtype=np.float32)
    stored = float(affine.apply(outside)[0])
    assert stored == 0.0
    assert stored * spec.std[0] + spec.mean[0] != pytest.approx(float(outside[0]))
