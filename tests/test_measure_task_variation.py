"""Tests for `scripts/measure_task_variation.py` — the video-free "does this need vision?" screen.

This script exists to be believed about datasets nobody has looked at yet. Its output is a verdict
that decides whether terabytes get downloaded and GPU hours get spent, so the ways it can be wrong
are not "it prints the wrong float":

  it cannot tell the      The whole tool is one discrimination: is the reach target implied by the
  two cases apart         starting state, or is it not? So the two synthetic corpora below are
                          built with the answer known — one where the grasp pose is an exact
                          linear function of the t=0 state, one where it is drawn independently of
                          it — and the screen has to separate them. A tool that returns 0.6 on both
                          reproduces every archived number and answers nothing.

  it leaks through        R^2 with folds over ROWS instead of over EPISODES is the same code with
  the folds               one line changed, prints the same table, and reports memorisation as
                          proprioception. On the archived dataset the two coincide (one row per
                          episode) so the bug would be invisible there — which is exactly why it is
                          pinned here on a design with several rows per episode, where a row-wise
                          map turns an R^2 of ~0 into ~0.9.

  it invents a grasp      Normalizing the hand synergy by its own min/max is what makes the
                          detector embodiment-independent and is also what turns a dead channel's
                          float noise into a full-scale open/close. T-31 is what that costs. The
                          dynamic-range refusal and the debounce are both tested for FIRING, not
                          only for passing.

  the headline is a       A parked limb is re-posed between episodes: large between-episode spread,
  parked limb             no within-episode motion, spread/range ratios of 18-54 on the real
                          corpus. Letting it into the L2 roughly doubles "how much does the reach
                          target move" for reasons that have nothing to do with the task.

  it stops reproducing    The archive gate is the only thing tying a run to
  PR-01                   docs/preregistration/PR-01-TASK-VARIATION.md. It is tested for firing on
                          the wrong data, for passing on the real data, and the pinned constants
                          are checked against the strings the document actually publishes.

Everything except the two real-data tests is synthetic and runs in milliseconds. The synthetic
corpora carry the answer in their construction rather than in a fixture, so a test that passes says
something about the measurement rather than about a recorded number.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_DATASET = _REPO_ROOT / "data" / "raw" / "gr00t_apple"
WRITEUP = _REPO_ROOT / "docs" / "preregistration" / "PR-01-TASK-VARIATION.md"


def _load(name: str) -> Any:
    """`test_bench_incremental_value`'s loader: import a script by path, registered in
    `sys.modules` so `@dataclass` can resolve its own annotations."""
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mtv = _load("measure_task_variation")

# -- the synthetic embodiment --------------------------------------------------------------------
#
# Four arm joints, four hand joints, one dead column and three noise columns. Deliberately NOT the
# G1's layout: the script must be driven by --arm-slice / --hand-slice, and a test on 15:29 / 29:36
# would pass just as well against hard-wired constants.

STATE_DIM = 12
ARM = slice(0, 4)
HAND = slice(4, 8)
DEAD_COLUMN = 8
"""A column that never moves. The zero-variance guard's job, present in every synthetic state."""

def _timing(index: int) -> tuple[int, int]:
    """(length, grasp_at) for synthetic episode `index` — varied, and a pure function of it.

    Varied because a corpus where every grasp lands on the same sample has a CONSTANT grasp
    instant, and "how predictable is the timing" is then undefined rather than easy. A pure
    function of the index because two runs have to build the same corpus.
    """
    return 50 + index % 17, 20 + (index * 7) % 13


def _episode(
    start_arm: np.ndarray,
    target_arm: np.ndarray,
    *,
    length: int = 60,
    grasp_at: int = 30,
    hand_amplitude: float = 1.0,
    spike: tuple[int, int] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """[length, STATE_DIM] — an arm ramping from `start_arm` to `target_arm` by `grasp_at`.

    The hand synergy is open until `grasp_at` and closed after, so the grasp pose IS `target_arm`
    by construction and every downstream number can be predicted by hand. `spike` closes the hand
    briefly before the real grasp, which is the only thing the debounce exists to survive.
    """
    rng = np.random.default_rng(seed)
    states = np.zeros((length, STATE_DIM), dtype=np.float64)
    ramp = np.clip(np.arange(length, dtype=np.float64) / grasp_at, 0.0, 1.0)[:, None]
    states[:, ARM] = start_arm[None, :] + ramp * (target_arm - start_arm)[None, :]
    synergy = np.zeros(length, dtype=np.float64)
    synergy[grasp_at:] = hand_amplitude
    if spike is not None:
        synergy[spike[0] : spike[1]] = hand_amplitude
    states[:, HAND] = synergy[:, None]
    states[:, DEAD_COLUMN] = 1.0
    states[:, DEAD_COLUMN + 1 :] = rng.normal(size=(length, STATE_DIM - DEAD_COLUMN - 1))
    return states


def _corpus(num_episodes: int, *, determined: bool, seed: int = 0) -> list[tuple[str, np.ndarray]]:
    """A synthetic dataset whose answer is known.

    `determined=True`: the grasp pose is `M @ start_arm` for a fixed matrix M, AND the grasp
    instant is an affine function of the starting pose. Both halves of the task are therefore in
    proprioception and a linear readout must recover both (R^2 -> 1, verdict BLIND-SUFFICIENT).

    `determined=False`: the grasp pose is drawn independently of the starting pose — the object
    moved and the robot cannot know where from its own joints (R^2 -> 0). This is the case the
    whole tool exists to detect, and the one a broken screen would report as predictable.
    """
    rng = np.random.default_rng(seed)
    mixing = rng.normal(size=(4, 4))
    episodes: list[tuple[str, np.ndarray]] = []
    for i in range(num_episodes):
        start = rng.normal(size=4)
        length, grasp_at = _timing(i)
        if determined:
            target = mixing @ start
            grasp_at = int(np.clip(round(26.0 + 4.0 * start[0]), 16, 34))
        else:
            target = rng.normal(size=4)
        episodes.append(
            (f"ep-{i:04d}", _episode(start, target, length=length, grasp_at=grasp_at, seed=i))
        )
    return episodes


def _collect(episodes: list[tuple[str, np.ndarray]]) -> Any:
    return mtv.collect_grasps(episodes, arm=ARM, hand=HAND)


def _write_dataset(root: Path, episodes: list[tuple[str, np.ndarray]]) -> Path:
    """Write a synthetic corpus as LeRobot-style raw parquet."""
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    for episode_id, states in episodes:
        flat = pa.array(states.reshape(-1).astype(np.float32))
        column = pa.FixedSizeListArray.from_arrays(flat, states.shape[1])
        pq.write_table(
            pa.table({mtv.DEFAULT_STATE_COLUMN: column}), data_dir / f"{episode_id}.parquet"
        )
    return root


# ================================================================================================
# The grasp detector
# ================================================================================================


def test_detect_grasp_returns_the_first_sustained_close() -> None:
    hand = np.zeros((80, 3))
    hand[20:40] = 1.0
    hand[60:] = 1.0
    assert mtv.detect_grasp(hand) == 20


def test_detect_grasp_ignores_a_spike_shorter_than_the_debounce() -> None:
    """A 4-step blip is not a grasp, and taking it as one would report the STARTING pose.

    This is the failure the debounce exists for, and it is silent: a spike near t=0 still yields a
    pose, a spread, an R^2 and a verdict, and the verdict would say the reach target barely moves
    because every "grasp pose" is a starting pose.
    """
    hand = np.zeros((80, 3))
    hand[5:9] = 1.0  # shorter than GRASP_DEBOUNCE_STEPS
    hand[40:] = 1.0
    assert mtv.detect_grasp(hand) == 40


def test_detect_grasp_debounce_length_is_the_boundary() -> None:
    """Exactly `GRASP_DEBOUNCE_STEPS` closed samples counts; one fewer does not."""
    hold = mtv.GRASP_DEBOUNCE_STEPS
    exact = np.zeros((80, 1))
    exact[10 : 10 + hold] = 1.0
    assert mtv.detect_grasp(exact) == 10

    one_short = np.zeros((80, 1))
    one_short[10 : 10 + hold - 1] = 1.0
    one_short[50:] = 1.0
    assert mtv.detect_grasp(one_short) == 50


def test_detect_grasp_refuses_a_near_constant_channel() -> None:
    """The T-31 guard. A dead channel divided by its own range is a full-scale open/close.

    Without the refusal this returns an index — the normalization maps float noise onto [0, 1] and
    the crossing is real in the normalized signal. Every number downstream would then be computed
    at an arbitrary sample of each episode, and nothing in the output would say so.
    """
    rng = np.random.default_rng(0)
    drift = np.linspace(0.0, 0.01, 200)[:, None]  # a dead channel that merely drifts
    dead = np.full((200, 7), 0.3) + drift + rng.normal(scale=1e-5, size=(200, 7))
    assert float(np.ptp(dead.mean(axis=1))) < mtv.GRASP_MIN_DYNAMIC_RANGE
    assert mtv.detect_grasp(dead) is None

    # ...and the same channel yields a confident, sustained, entirely fictional grasp the moment
    # the guard is told its range is admissible. That is what the refusal is holding back: not a
    # crash, an index — around the midpoint of the drift, in every episode, for ever.
    invented = mtv.detect_grasp(dead, min_dynamic_range=0.0)
    assert invented is not None
    assert 80 < invented < 120


def test_detect_grasp_refuses_a_channel_already_closed_at_t0() -> None:
    """No upward crossing was observed, so there is no grasp instant to report."""
    hand = np.ones((80, 3))
    hand[40:] = 3.0
    assert mtv.detect_grasp(hand) == 40  # the second step up is a crossing

    always = np.ones((80, 3))
    always[0] = 1.0 + 1e-9
    assert mtv.detect_grasp(always) is None


def test_detect_grasp_refuses_an_episode_shorter_than_the_debounce_window() -> None:
    short = np.zeros((mtv.GRASP_DEBOUNCE_STEPS, 2))
    short[2:] = 1.0
    assert mtv.detect_grasp(short) is None


def test_detect_grasp_is_scale_and_offset_invariant() -> None:
    """Per-episode normalization is what makes the screen embodiment-independent."""
    hand = np.zeros((80, 3))
    hand[33:] = 1.0
    rescaled = hand * -0.4 + 7.0  # inverted, so the "close" is now a decrease
    assert mtv.detect_grasp(hand) == 33
    assert mtv.detect_grasp(rescaled) is None  # ...and an inverted channel has no upward crossing


# ================================================================================================
# The discrimination the tool exists for
# ================================================================================================


def test_r2_is_near_one_when_the_grasp_pose_is_determined_by_the_t0_state() -> None:
    """The reach target is an exact linear function of the starting pose: vision buys nothing."""
    measurement = mtv.measure(_collect(_corpus(120, determined=True, seed=1)))
    assert measurement.num_grasps == 120
    assert measurement.r2_pose_from_state > 0.99
    assert measurement.r2_time_from_state > 0.95
    assert measurement.residual_pose_spread < 0.05 * measurement.grasp_pose_spread
    assert mtv.decide(measurement).letter == "BLIND-SUFFICIENT"


def test_r2_is_near_zero_when_the_grasp_pose_is_independent_of_the_t0_state() -> None:
    """The object moved and proprioception cannot know where to.

    This is the test the tool exists to pass. A screen that cannot distinguish this corpus from
    the one above is not a screen — it reproduces every archived number and answers nothing, and
    the only thing separating the two is what it does with the two R^2 values.
    """
    measurement = mtv.measure(_collect(_corpus(120, determined=False, seed=2)))
    assert measurement.num_grasps == 120
    assert abs(measurement.r2_pose_from_state) < 0.15
    assert abs(measurement.r2_pose_from_arm) < 0.15
    # ...and the residual is essentially the whole spread: nothing was explained away.
    assert measurement.residual_pose_spread > 0.85 * measurement.grasp_pose_spread
    assert mtv.decide(measurement).letter == "VISION-CANDIDATE"


def _corpus_with_the_answer_outside_the_arm(
    num_episodes: int = 100, seed: int = 13
) -> list[tuple[str, np.ndarray]]:
    """The reach target is determined by non-arm state columns, not by the starting arm pose.

    A robot whose base or torso pose implies where the object is, while its arm starts anywhere.
    The two R^2 rows the report prints — "from the full state" and "from the arm only" — are the
    only thing that separates this from a corpus where the arm itself carries the answer.
    """
    rng = np.random.default_rng(seed)
    beacon_width = STATE_DIM - DEAD_COLUMN - 1
    mixing = rng.normal(size=(4, beacon_width))
    episodes: list[tuple[str, np.ndarray]] = []
    for i in range(num_episodes):
        beacon = rng.normal(size=beacon_width)
        start = rng.normal(size=4)
        length, grasp_at = _timing(i)
        states = _episode(start, mixing @ beacon, length=length, grasp_at=grasp_at, seed=i)
        states[:, DEAD_COLUMN + 1 :] = beacon[None, :]
        episodes.append((f"ep-{i:04d}", states))
    return episodes


def test_the_arm_only_predictor_reads_the_arm_and_not_the_whole_state() -> None:
    """"from the t=0 arm" is a WEAKER predictor than "from the t=0 state", and must stay one.

    Handed the whole state it becomes the row above it, both rows print +0.99, and the report
    silently loses the distinction between "the answer is in the arm's own posture" and "the answer
    is somewhere else in the robot" — which is the difference between a postural task and one where
    something outside the arm is carrying the information.
    """
    measurement = mtv.measure(_collect(_corpus_with_the_answer_outside_the_arm()))
    assert measurement.r2_pose_from_state > 0.99
    assert measurement.r2_pose_from_arm < 0.2


def test_the_two_corpora_are_separated_by_the_screen() -> None:
    """Stated as one assertion, because "0.99 vs 0.02" is the product, not either number."""
    blind = mtv.measure(_collect(_corpus(120, determined=True, seed=1)))
    visual = mtv.measure(_collect(_corpus(120, determined=False, seed=2)))
    assert blind.r2_pose_from_state - visual.r2_pose_from_state > 0.8
    assert mtv.decide(blind).letter != mtv.decide(visual).letter


# ================================================================================================
# Folds over episodes, not over rows
# ================================================================================================


NUM_LEAKY_FEATURES = 30
"""More features than episodes, so a linear map CAN memorise every episode it has seen.

With fewer features than episodes the leak is only partial and the row-wise R^2 lands around 0.5 —
which is still a leak, but a weak demonstration. The point of this design is that the two fold maps
give ~1.0 and ~0.0 on the same data, so nothing about the assertion depends on a margin.
"""


def _leaky_design(
    num_episodes: int = 20, rows_per_episode: int = 5, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows that identify their episode, with a target that the features cannot explain.

    Each episode has its own feature vector and its own target, drawn INDEPENDENTLY of each other;
    the rows within an episode are near-copies of both. So the honest out-of-episode R^2 is ~0 —
    there is no relationship to learn — while a row-wise fold can read the answer off the four
    sibling rows of the same episode sitting in its fit set.
    """
    rng = np.random.default_rng(seed)
    feature = rng.normal(size=(num_episodes, NUM_LEAKY_FEATURES))
    target = rng.normal(size=(num_episodes, 2))
    ids: list[str] = []
    x: list[np.ndarray] = []
    y: list[np.ndarray] = []
    for e in range(num_episodes):
        for _ in range(rows_per_episode):
            ids.append(f"ep-{e:03d}")
            x.append(feature[e] + rng.normal(scale=1e-3, size=NUM_LEAKY_FEATURES))
            y.append(target[e] + rng.normal(scale=1e-3, size=2))
    return np.asarray(ids, dtype=object), np.stack(x), np.stack(y)


def test_row_wise_folds_inflate_r2_and_episode_wise_folds_do_not() -> None:
    """The leak this whole design is arranged to avoid, shown actually happening.

    On the archived dataset there is exactly one row per episode, so a row-wise fold map produces
    the SAME number and the bug is invisible there. It stops being invisible the moment anything
    contributes a second row per episode — and at that point the screen would report memorisation
    as proprioception, i.e. it would call a dataset blind-sufficient because it had seen the
    answer.
    """
    ids, x, y = _leaky_design()

    row_folds = np.arange(len(ids)) % mtv.CV_FOLDS
    leaked = mtv.r2_score(y, mtv.out_of_fold_predictions(x, y, row_folds))

    honest = mtv.cross_validated_r2(x, y, ids)

    assert leaked > 0.9, "the leaky design does not leak; the guard below would prove nothing"
    assert honest < 0.2
    assert leaked - honest > 0.7


def test_check_folds_are_episode_disjoint_refuses_a_row_wise_map() -> None:
    ids, _x, _y = _leaky_design()
    row_folds = np.arange(len(ids)) % mtv.CV_FOLDS
    with pytest.raises(SystemExit, match="not over episodes"):
        mtv.check_folds_are_episode_disjoint(ids, row_folds, mtv.CV_FOLDS)


def test_assign_episode_folds_keeps_every_episode_on_one_side() -> None:
    ids, _x, _y = _leaky_design()
    folds = mtv.assign_episode_folds(ids, mtv.CV_FOLDS)
    mtv.check_folds_are_episode_disjoint(ids, folds, mtv.CV_FOLDS)
    for episode_id in set(ids.tolist()):
        assert len(set(folds[ids == episode_id].tolist())) == 1


def test_assign_episode_folds_is_deterministic_and_order_independent() -> None:
    """Assignment follows the SORTED ids, so a reshuffled directory listing cannot move a fold."""
    ids = np.asarray([f"ep-{i:03d}" for i in range(23)], dtype=object)
    folds = mtv.assign_episode_folds(ids, 5)
    assert np.array_equal(folds, mtv.assign_episode_folds(ids, 5))

    shuffled_order = np.asarray(list(reversed(ids.tolist())), dtype=object)
    shuffled_folds = mtv.assign_episode_folds(shuffled_order, 5)
    for episode_id in ids.tolist():
        assert folds[ids == episode_id][0] == shuffled_folds[shuffled_order == episode_id][0]


def test_too_few_episodes_for_the_fold_count_is_refused() -> None:
    ids = np.asarray(["a", "b", "c"], dtype=object)
    with pytest.raises(SystemExit, match="cannot be split into 5 folds"):
        mtv.assign_episode_folds(ids, 5)


def test_cross_validated_r2_uses_episode_folds_even_when_rows_repeat() -> None:
    """The call site, not only the helper: `cross_validated_r2` must build the map itself."""
    ids, x, y = _leaky_design()
    assert mtv.cross_validated_r2(x, y, ids) < 0.2


# ================================================================================================
# The ridge: fit-fold standardization, zero-variance columns
# ================================================================================================


EXPECTED_RIDGE_LAMBDA = 1e-2
"""The penalty PR-01-TASK-VARIATION.md's numbers were measured at, written out here on purpose.

`_reference_out_of_fold` is only an independent implementation if it is independent of the module
it checks, and a reference that reads `mtv.RIDGE_LAMBDA` follows every change to it — including a
change of two orders of magnitude, which moves every R^2 in the report and is invisible to a
reference that moved with it.
"""


def test_the_ridge_penalty_is_the_one_the_archive_was_measured_at() -> None:
    assert mtv.RIDGE_LAMBDA == EXPECTED_RIDGE_LAMBDA
    assert mtv.CV_FOLDS == 5


def _reference_out_of_fold(
    x: np.ndarray, y: np.ndarray, folds: np.ndarray, *, standardize_on_all_rows: bool
) -> np.ndarray:
    """The same solve, written independently, with the leak as a switch.

    `standardize_on_all_rows=True` is the quiet mistake: the scored rows contribute to the scale
    the model is fitted in. It changes no shape, raises nothing and moves the number.
    """
    predictions = np.zeros(y.shape, dtype=np.float64)
    for k in np.unique(folds):
        scored, fitted = folds == k, folds != k
        source = x if standardize_on_all_rows else x[fitted]
        mean, std = source.mean(axis=0), source.std(axis=0)
        live = std > mtv.ZERO_VARIANCE_EPS
        design = np.hstack([(x[fitted][:, live] - mean[live]) / std[live], np.ones((fitted.sum(), 1))])
        query = np.hstack([(x[scored][:, live] - mean[live]) / std[live], np.ones((scored.sum(), 1))])
        gram = design.T @ design
        weights = np.linalg.solve(
            gram + EXPECTED_RIDGE_LAMBDA * np.eye(gram.shape[0]), design.T @ y[fitted]
        )
        predictions[scored] = query @ weights
    return predictions


def test_standardization_is_fitted_on_the_fit_fold_only() -> None:
    """Pinned against an independent implementation, with the leaked variant shown to differ.

    The second assertion is what gives the first one teeth: on data where the two agree, matching
    the correct reference proves nothing.
    """
    rng = np.random.default_rng(11)
    ids = np.asarray([f"ep-{i:03d}" for i in range(60)], dtype=object)
    x = rng.normal(size=(60, 6))
    x[:12] *= 40.0  # fold 0 lives on a different scale, so who computes the scale matters
    y = x @ rng.normal(size=(6, 2)) + rng.normal(scale=0.1, size=(60, 2))
    folds = mtv.assign_episode_folds(ids, mtv.CV_FOLDS)

    measured = mtv.out_of_fold_predictions(x, y, folds)
    correct = _reference_out_of_fold(x, y, folds, standardize_on_all_rows=False)
    leaked = _reference_out_of_fold(x, y, folds, standardize_on_all_rows=True)

    assert np.allclose(measured, correct, rtol=0.0, atol=1e-12)
    assert not np.allclose(measured, leaked, rtol=0.0, atol=1e-9)


def test_zero_variance_columns_are_dropped_rather_than_divided_by() -> None:
    """A constant predictor column must not become a full-scale noise feature, or a NaN."""
    rng = np.random.default_rng(3)
    ids = np.asarray([f"ep-{i:03d}" for i in range(40)], dtype=object)
    live_columns = rng.normal(size=(40, 3))
    y = live_columns @ rng.normal(size=(3, 1))
    folds = mtv.assign_episode_folds(ids, mtv.CV_FOLDS)

    with_dead = np.hstack([live_columns, np.full((40, 2), 5.0)])
    predictions = mtv.out_of_fold_predictions(with_dead, y, folds)
    assert np.isfinite(predictions).all()
    assert np.allclose(
        predictions, mtv.out_of_fold_predictions(live_columns, y, folds), rtol=0.0, atol=1e-12
    )


def test_r2_score_centres_each_target_column_on_its_own_mean() -> None:
    """Pooled over columns, but centred PER column.

    Centring on one scalar mean across columns puts the distance between the columns into SS_tot,
    which is not variation any predictor has to explain. Here that would turn an honest 0.9375 into
    0.9999 — a number that says "solved" about a fit that is off by 1 in 2.
    """
    y = np.array([[0.0, 100.0], [2.0, 102.0], [4.0, 104.0]])
    predictions = y.copy()
    predictions[0, 0] += 1.0
    # SS_res = 1. SS_tot about the per-column means [2, 102] = 8 + 8 = 16.
    assert mtv.r2_score(y, predictions) == pytest.approx(1.0 - 1.0 / 16.0)


def test_r2_score_refuses_a_constant_target() -> None:
    y = np.full((10, 2), 4.0)
    with pytest.raises(SystemExit, match="constant across every episode"):
        mtv.r2_score(y, y)


def test_out_of_fold_predictions_refuse_a_non_positive_lambda() -> None:
    x = np.zeros((10, 2))
    with pytest.raises(SystemExit, match="lambda must be > 0"):
        mtv.out_of_fold_predictions(x, x, np.arange(10) % 2, lam=0.0)


# ================================================================================================
# Parked joints
# ================================================================================================


def _corpus_with_a_parked_joint(
    num_episodes: int = 40, seed: int = 5
) -> list[tuple[str, np.ndarray]]:
    """Arm joints 0-2 reach; joint 3 is parked at a different angle in every episode.

    The parked joint's between-episode spread is deliberately huge — larger than all three live
    joints together — so if it reaches the headline it dominates it.
    """
    rng = np.random.default_rng(seed)
    episodes: list[tuple[str, np.ndarray]] = []
    for i in range(num_episodes):
        start = np.concatenate([rng.normal(size=3), [rng.normal(scale=10.0)]])
        target = start.copy()
        target[:3] = rng.normal(size=3)  # joint 3 never moves within the episode
        length, grasp_at = _timing(i)
        episodes.append(
            (f"ep-{i:04d}", _episode(start, target, length=length, grasp_at=grasp_at, seed=i))
        )
    return episodes


def test_a_parked_joint_is_excluded_from_the_headline_l2() -> None:
    """Its spread is a re-posed limb, and it is ~10x the real signal here."""
    table = _collect(_corpus_with_a_parked_joint())
    measurement = mtv.measure(table)

    assert list(measurement.live_mask) == [True, True, True, False]
    assert measurement.mean_within_range[3] == pytest.approx(0.0, abs=1e-12)
    assert measurement.between_episode_std[3] > 5.0

    live_only = float(np.linalg.norm(measurement.between_episode_std[:3]))
    everything = float(np.linalg.norm(measurement.between_episode_std))
    assert measurement.grasp_pose_spread == pytest.approx(live_only, rel=1e-12)
    assert everything > 3.0 * live_only, "the parked joint does not dominate; the test has no teeth"


def test_a_parked_joint_stays_out_of_the_r2_targets_too() -> None:
    """Not only the L2. A parked joint's grasp pose IS its starting pose, so leaving it in the
    target inflates the R^2 with a column the model gets for free."""
    table = _collect(_corpus_with_a_parked_joint())
    measurement = mtv.measure(table)

    live = measurement.live_mask
    contaminated = mtv.cross_validated_r2(
        table.start_states, table.arm_poses, table.episode_ids
    )
    assert measurement.r2_pose_from_state < contaminated - 0.3
    assert contaminated > 0.9  # the free column is worth almost all of it
    assert live.sum() == 3


def test_within_episode_range_is_a_peak_to_peak_not_a_spread() -> None:
    """The parked/live split and the motion scale both hang on this being a RANGE.

    A standard deviation of the same trajectory is ~0.3x its range and is not the "how far did this
    joint travel" the headline reads it as: it would rescale the motion denominator, move the
    spread/motion ratio by 3x, and shift which joints clear the parked threshold.
    """
    start = np.array([0.0, 0.0, 0.0, 0.0])
    target = np.array([2.0, -1.0, 0.5, 0.0])
    episodes = [
        (f"ep-{i:03d}", _episode(start, target, length=50, grasp_at=20, seed=i)) for i in range(6)
    ]
    table = _collect(episodes)
    assert table.joint_ranges[0] == pytest.approx(np.abs(target - start))
    assert table.joint_ranges[0][0] == pytest.approx(2.0)  # a std would read ~0.6 here


def test_live_joint_mask_is_a_mean_range_threshold() -> None:
    ranges = np.array([[0.0, 1.0, 0.04], [0.0, 1.0, 0.06]])
    assert list(mtv.live_joint_mask(ranges)) == [False, True, False]
    assert list(mtv.live_joint_mask(ranges, threshold=0.0)) == [False, True, True]


def test_explained_by_a_constant_is_measured_against_ONE_scalar_mean() -> None:
    """The archived quantity, whose denominator is a single scalar over every joint.

    Centre the denominator per joint instead and it collapses to 0 by construction — same name,
    same shape, and a number that would have made PR-01-TASK-VARIATION.md say the opposite thing.
    """
    poses = np.array([[0.0, 10.0], [2.0, 12.0]])
    # per-joint means [1, 11] -> SS_res 4. Grand mean 6 -> SS_tot 36 + 16 + 16 + 36 = 104.
    assert mtv._explained_by_a_constant(poses) == pytest.approx(1.0 - 4.0 / 104.0)


def test_the_archived_residual_spread_is_the_in_sample_one() -> None:
    """Two residual spreads are printed and only one is the archived number.

    They differ by 15 % on the real corpus — far too little to spot in a printout of four-decimal
    floats, and more than enough to fail a gate whose tolerance is half a unit in the last digit.
    """
    table = _collect(_corpus(120, determined=False, seed=2))
    measurement = mtv.measure(table)

    live = measurement.live_mask
    poses_live = table.arm_poses[:, live]
    expected = float(
        np.linalg.norm(
            (poses_live - mtv.in_sample_predictions(table.start_states, poses_live)).std(axis=0)
        )
    )
    assert measurement.residual_pose_spread == pytest.approx(expected, rel=1e-12)
    assert measurement.residual_pose_spread < measurement.residual_pose_spread_oof


def test_measure_refuses_a_dataset_in_which_nothing_moves() -> None:
    rng = np.random.default_rng(9)
    episodes = []
    for i in range(20):
        start = rng.normal(size=4)
        length, grasp_at = _timing(i)
        episodes.append(
            (f"ep-{i:04d}", _episode(start, start.copy(), length=length, grasp_at=grasp_at, seed=i))
        )
    with pytest.raises(SystemExit, match="no arm joint moves"):
        mtv.measure(_collect(episodes))


# ================================================================================================
# Collection and slices
# ================================================================================================


def test_refused_episodes_are_reported_and_excluded() -> None:
    episodes = _corpus(20, determined=False, seed=4)
    dead = episodes[3][1].copy()
    dead[:, HAND] = 0.5  # a flat hand channel
    episodes[3] = (episodes[3][0], dead)

    table = _collect(episodes)
    assert table.num_episodes_seen == 20
    assert table.num_grasps == 19
    assert table.refused_episode_ids == ("ep-0003",)
    assert "ep-0003" not in set(table.episode_ids.tolist())


def test_collect_refuses_slices_wider_than_the_state() -> None:
    episodes = _corpus(6, determined=False)
    with pytest.raises(SystemExit, match="state is 12 wide"):
        mtv.collect_grasps(episodes, arm=slice(0, 4), hand=slice(40, 47))


def test_collect_refuses_a_corpus_with_no_grasp_anywhere() -> None:
    episodes = [(f"ep-{i}", np.zeros((40, STATE_DIM))) for i in range(5)]
    with pytest.raises(SystemExit, match="no grasp was detected"):
        _collect(episodes)


def test_arm_state_indices_follow_the_slice() -> None:
    """The "starting arm pose" predictor reads the SAME columns the grasp pose came from."""
    table = mtv.collect_grasps(_corpus(10, determined=False), arm=slice(1, 4), hand=HAND)
    assert list(table.arm_state_indices) == [1, 2, 3]


def test_grasp_fraction_matches_the_construction() -> None:
    table = _collect(_corpus(12, determined=False))
    expected = [grasp_at / length for length, grasp_at in (_timing(i) for i in range(12))]
    assert list(table.grasp_fractions) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text,expected", [("15:29", (15, 29)), ("0:1", (0, 1)), ("29:36", (29, 36))]
)
def test_parse_slice_accepts_half_open_ranges(text: str, expected: tuple[int, int]) -> None:
    parsed = mtv.parse_slice(text)
    assert (parsed.start, parsed.stop) == expected


@pytest.mark.parametrize("text", ["15", "15:29:2", "a:b", "-7:0", "29:29", "30:29"])
def test_parse_slice_refuses_anything_else(text: str) -> None:
    with pytest.raises(SystemExit):
        mtv.parse_slice(text)


# ================================================================================================
# The CLI: layouts, gate, determinism
# ================================================================================================


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    assert mtv.main(argv) == 0
    return capsys.readouterr().out


def _synthetic_cli_args(root: Path) -> list[str]:
    return [
        "--dataset", str(root),
        "--arm-slice", "0:4",
        "--hand-slice", "4:8",
        "--archive-gate", "off",
    ]  # fmt: skip


def test_find_parquet_files_handles_the_three_layouts_and_always_sorts(tmp_path: Path) -> None:
    """Sorted at every level. Directory order is not an order, and it is what fixes the folds.

    Every layout below is created in an order that is neither sorted nor reverse-sorted, so the
    assertion cannot be satisfied by a `glob` that happens to walk the directory in a helpful
    order, nor by a reversal of it.
    """
    flat = tmp_path / "flat"
    flat.mkdir()
    for name in ("b.parquet", "a.parquet", "c.parquet"):
        (flat / name).write_bytes(b"")
    found = mtv.find_parquet_files(flat)
    assert [p.name for p in found] == ["a.parquet", "b.parquet", "c.parquet"] == sorted(
        p.name for p in found
    )

    lerobot = tmp_path / "lerobot"
    for chunk, names in (
        ("chunk-001", ("m.parquet",)),
        ("chunk-000", ("b.parquet", "a.parquet", "c.parquet")),
    ):
        (lerobot / "data" / chunk).mkdir(parents=True)
        for name in names:
            (lerobot / "data" / chunk / name).write_bytes(b"")
    found = [str(p.relative_to(lerobot / "data")) for p in mtv.find_parquet_files(lerobot)]
    assert found == sorted(found)
    assert found == [
        "chunk-000/a.parquet",
        "chunk-000/b.parquet",
        "chunk-000/c.parquet",
        "chunk-001/m.parquet",
    ]

    nested = tmp_path / "nested"
    for part in ("odd/place", "aaa"):
        (nested / part).mkdir(parents=True)
    for path in ("odd/place/z.parquet", "aaa/b.parquet", "aaa/d.parquet"):
        (nested / path).write_bytes(b"")
    found = [str(p.relative_to(nested)) for p in mtv.find_parquet_files(nested)]
    assert found == sorted(found) == ["aaa/b.parquet", "aaa/d.parquet", "odd/place/z.parquet"]

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no .parquet files found"):
        mtv.find_parquet_files(empty)
    with pytest.raises(SystemExit, match="not a file or directory"):
        mtv.find_parquet_files(tmp_path / "absent")


def test_cli_refuses_overlapping_arm_and_hand_slices(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "ds", _corpus(10, determined=False))
    with pytest.raises(SystemExit, match="overlap"):
        mtv.main(["--dataset", str(root), "--arm-slice", "0:6", "--hand-slice", "4:8"])


def test_archive_gate_fires_on_a_dataset_that_is_not_pr01s(tmp_path: Path) -> None:
    """The gate is tested for FIRING, not only for passing.

    A screen whose loader has drifted still prints a full table and a verdict; the gate is the only
    thing that turns that into a refusal.
    """
    root = _write_dataset(tmp_path / "ds", _corpus(30, determined=False))
    with pytest.raises(SystemExit, match="does NOT reproduce"):
        mtv.main(["--dataset", str(root), "--arm-slice", "0:4", "--hand-slice", "4:8"])


def test_archive_gate_off_says_it_reproduces_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_dataset(tmp_path / "ds", _corpus(30, determined=False))
    out = _run(_synthetic_cli_args(root), capsys)
    assert "ARCHIVE GATE OFF" in out
    assert "archive gate was OFF" in out
    assert "reproduce exactly" not in out


def test_two_runs_are_byte_identical(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Determinism, asserted rather than assumed. Nothing here may depend on a clock or a seed."""
    root = _write_dataset(tmp_path / "ds", _corpus(30, determined=True, seed=3))
    args = _synthetic_cli_args(root)
    assert _run(args, capsys) == _run(args, capsys)


def test_json_record_carries_the_numbers_and_the_thresholds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_dataset(tmp_path / "ds", _corpus(30, determined=True, seed=3))
    out_path = tmp_path / "record.json"
    _run([*_synthetic_cli_args(root), "--json", str(out_path)], capsys)

    record = json.loads(out_path.read_text())
    assert record["num_grasps"] == 30
    assert record["verdict"] == "BLIND-SUFFICIENT"
    assert record["live_joints"] == [0, 1, 2, 3]
    assert record["r2_pose_from_state"] > 0.99
    assert record["thresholds"]["GRASP_DEBOUNCE_STEPS"] == mtv.GRASP_DEBOUNCE_STEPS
    assert record["thresholds"]["CV_FOLDS"] == mtv.CV_FOLDS


def test_the_report_marks_every_joint_live_or_parked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_dataset(tmp_path / "ds", _corpus_with_a_parked_joint())
    out = _run(_synthetic_cli_args(root), capsys)
    assert out.count("LIVE") == 3
    assert out.count("PARKED") == 1
    assert "L2 over 3 live joints" in out


# ================================================================================================
# The verdict rule
# ================================================================================================


def _measurement(**overrides: float) -> Any:
    base: dict[str, Any] = {
        "num_episodes_seen": 10,
        "num_grasps": 10,
        "live_mask": np.array([True]),
        "between_episode_std": np.array([1.0]),
        "mean_within_range": np.array([1.0]),
        "grasp_pose_spread": 1.0,
        "within_episode_motion": 1.0,
        "constant_explained_all_arm": 0.5,
        "constant_explained_live": 0.5,
        "grasp_fraction_mean": 0.5,
        "grasp_fraction_std": 0.1,
        "r2_pose_from_state": 0.0,
        "r2_pose_from_arm": 0.0,
        "r2_pose_from_length": 0.0,
        "r2_time_from_state": 0.0,
        "residual_pose_spread": 0.5,
        "residual_pose_spread_oof": 0.5,
    }
    base.update(overrides)
    return mtv.Measurement(**base)


def test_verdict_is_blind_sufficient_only_when_both_pose_and_timing_are_predictable() -> None:
    assert mtv.decide(_measurement(r2_pose_from_state=0.95, r2_time_from_state=0.8)).letter == (
        "BLIND-SUFFICIENT"
    )
    assert mtv.decide(_measurement(r2_pose_from_state=0.95, r2_time_from_state=0.1)).letter == (
        "VISION-CANDIDATE"
    )
    assert mtv.decide(_measurement(r2_pose_from_state=0.1, r2_time_from_state=0.8)).letter == (
        "VISION-CANDIDATE"
    )


def test_an_unpredictable_but_tiny_residual_is_not_a_vision_candidate() -> None:
    """A reach target nobody can predict, that moves by a millimetre, is not worth a video model."""
    verdict = mtv.decide(_measurement(residual_pose_spread=0.01, within_episode_motion=1.0))
    assert verdict.letter == "MIXED"
    assert verdict.clauses["what is left over is large enough to matter"] is False


# ================================================================================================
# The archive itself
# ================================================================================================


def test_the_pinned_constants_are_the_numbers_the_writeup_publishes() -> None:
    """The constants must be PR-01-TASK-VARIATION.md's, not a rerun's.

    The document is the artefact people read; these constants are what the gate enforces. If they
    ever drift apart, the gate goes on passing while the writeup goes on being wrong, and nothing
    in either place says so.
    """
    text = WRITEUP.read_text(encoding="utf-8")
    assert "scripts/measure_task_variation.py" in text
    assert f"{mtv.ARCHIVED_NUM_GRASPS} of {mtv.ARCHIVED_NUM_EPISODES}" in text
    for value, decimals in (
        (mtv.ARCHIVED_GRASP_POSE_SPREAD, 4),
        (mtv.ARCHIVED_WITHIN_EPISODE_MOTION, 4),
        (mtv.ARCHIVED_CONSTANT_EXPLAINED_ALL_ARM, 4),
        (mtv.ARCHIVED_GRASP_FRACTION_MEAN, 3),
        (mtv.ARCHIVED_GRASP_FRACTION_STD, 3),
        (mtv.ARCHIVED_R2_POSE_FROM_STATE, 4),
        (mtv.ARCHIVED_R2_POSE_FROM_ARM, 4),
        (mtv.ARCHIVED_R2_POSE_FROM_LENGTH, 4),
        (mtv.ARCHIVED_R2_TIME_FROM_STATE, 4),
        (mtv.ARCHIVED_RESIDUAL_POSE_SPREAD, 4),
    ):
        assert f"{value:.{decimals}f}" in text, f"{value} is not in {WRITEUP.name}"


def test_check_archive_reports_the_drift_it_found() -> None:
    measurement = _measurement()
    with pytest.raises(SystemExit) as excinfo:
        mtv.check_archive(measurement)
    message = str(excinfo.value)
    assert "archive 0/12" in message or "archive " in message
    assert "R^2 grasp TIME from the state" in message
    assert "--archive-gate off" in message


def test_the_gate_tolerance_is_half_a_unit_in_the_last_quoted_digit() -> None:
    """Tight enough that a real drift cannot round into agreement."""
    inside = mtv.ArchivedCheck("x", 0.6623 + 4e-5, 0.6623, mtv.ARCHIVE_ATOL_4DP)
    outside = mtv.ArchivedCheck("x", 0.6623 + 6e-5, 0.6623, mtv.ARCHIVE_ATOL_4DP)
    assert inside.ok and not outside.ok

    counts = mtv.ArchivedCheck("n", 401, 402, 0.0, 0)
    assert not counts.ok, "one missing episode must not pass as a rounding difference"


# ================================================================================================
# The real dataset
# ================================================================================================

_HAVE_REAL_DATA = REAL_DATASET.is_dir() and any(REAL_DATASET.glob("data/chunk-*/*.parquet"))
_needs_real_data = pytest.mark.skipif(
    not _HAVE_REAL_DATA, reason=f"{REAL_DATASET} is not present"
)


@_needs_real_data
def test_the_archived_numbers_reproduce_on_the_real_dataset(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The gate, on the data it was archived from. ~2 s, CPU, no video decoded."""
    out = _run(["--dataset", str(REAL_DATASET)], capsys)
    assert "archive 12/12 reproduce exactly" in out
    assert "DRIFT" not in out
    assert "VERDICT VISION-CANDIDATE" in out
    assert out.count("PARKED") == 7, "the G1's right arm is parked in this corpus"


@_needs_real_data
def test_the_real_dataset_gate_fires_on_a_different_arm_slice(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Left arm only (15:22) reproduces nine of the twelve numbers and must still be refused.

    That is the case the gate is really for. A wrong slice does not produce nonsense — it produces
    a table that looks exactly right, because nine of the twelve values are live-joint-only and do
    not depend on whether the parked arm was in the slice. Only the all-arm-joints value moves, and
    a reader comparing the printout against the document by eye would not catch it.
    """
    with pytest.raises(SystemExit, match="does NOT reproduce"):
        mtv.main(["--dataset", str(REAL_DATASET), "--arm-slice", "15:22"])
    capsys.readouterr()


@_needs_real_data
def test_the_real_dataset_has_no_grasp_in_the_right_hand_channel(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """state[36:43] is the RIGHT hand, which never closes in this corpus.

    The dynamic-range guard refuses every episode rather than inventing 402 grasps out of a flat
    channel's noise — the T-31 failure, on the actual data it would happen to.
    """
    with pytest.raises(SystemExit, match="no grasp was detected in any of 402"):
        mtv.main(["--dataset", str(REAL_DATASET), "--hand-slice", "36:43"])
    capsys.readouterr()
