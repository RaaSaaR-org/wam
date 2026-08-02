#!/usr/bin/env python3
"""Does this dataset require vision? A video-free screen, from proprioception alone.

Run this on a candidate dataset BEFORE downloading terabytes of video or spending GPU hours on it.
It reads nothing but the state column of the raw parquet, runs on CPU in seconds, and answers the
only question that decides whether a video branch can help: *is there anything in this task that
the robot's own joint angles do not already contain?*

This is the generalized form of the measurement written up in
``docs/preregistration/PR-01-TASK-VARIATION.md`` (both of its measurements, 402 episodes of
``data/raw/gr00t_apple``) and of the screening criterion PR-02 formalises. That document is the
archive: every number it quotes is pinned below as a named constant and re-checked on every run
against the real dataset, so a silent drift in the loader, the detector or the solver shows up as a
failed gate rather than as a slightly different sentence in a report.

WHAT IS MEASURED
----------------
The object being manipulated lives in the video and cannot be seen here. But where the hand GOES to
get it can: the arm pose at the moment of grasp — the first debounced close of the hand — is a
proprioceptive read-out of the reach target. Three measurements follow from it.

1.  **Does the target move at all?** Between-episode spread of the grasp pose, against the
    within-episode motion range. Reported per joint and as an L2 over joints.

    Joints that do not move WITHIN an episode are excluded from the headline. A parked limb is
    re-posed between recording sessions and its between-episode spread is therefore a nuisance, not
    task variation: on ``gr00t_apple`` the frozen right arm shows spread/range ratios of 18–54
    against the live arm's 0.16–0.63, and letting it into an L2 would roughly double the headline
    spread for reasons that have nothing to do with the task. The exclusion is mechanical
    (:data:`PARKED_JOINT_RANGE_RAD`) and the per-joint table prints which side each joint fell on.

2.  **Is that variation already in proprioception?** Cross-validated R^2 of the grasp POSE from the
    state at t=0, folds over EPISODES.

3.  **Is the TIMING in proprioception?** The same, for the grasp INSTANT.

HOW TO READ THE THREE NUMBERS
-----------------------------
A **high pose R^2** means the reach target is already implied by where the robot starts: the
variation is postural, not visual, and no video model is needed to supply it. A **low timing R^2**
means WHEN to act is not in the state — that is the visual decision, and it is the part a blind
policy cannot fake. The two come apart routinely, which is why they are measured separately: on
``gr00t_apple`` the pose is 61 % predictable from the starting state while the instant is 8 %, so
the honest reading is "most of the reach is postural, the moment of the grasp is not".

Both are CEILINGS ON A BLIND LINEAR READOUT, in one direction only. A high R^2 is strong evidence
that vision is not needed for that quantity — a weak model already has it. A low R^2 is NOT evidence
that pixels would supply it; it says only that a ridge on the starting state does not, and a
stronger blind model would close some of the gap. The verdict printed at the end says
VISION-CANDIDATE for exactly that reason and never says "vision would work".

WHAT THIS DOES NOT ESTABLISH
----------------------------
Both R^2 values are computed at ONE instant per episode — the grasp — so they say nothing about the
other ~99 % of the frames, which is where a chunk-MSE metric spends its weight. And a dataset that
passes this screen can still fail for reasons it cannot see: camera placement, action-space
mismatch, or a demonstration quality problem that only shows up closed-loop.

DETERMINISM
-----------
There is no randomness in this script. Folds are assigned by position in the SORTED episode ids, the
episode order comes from ``sorted()`` over the parquet paths, and nothing is sampled, shuffled or
seeded from a clock. Two runs on the same directory print byte-identical output; that is asserted in
``tests/test_measure_task_variation.py`` rather than assumed.

    scripts/measure_task_variation.py --dataset data/raw/gr00t_apple
    scripts/measure_task_variation.py --dataset data/raw/other --archive-gate off \\
        --arm-slice 0:7 --hand-slice 7:9
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# -- the archive: PR-01-TASK-VARIATION.md's numbers, one constant each ----------------------------
#
# Measured 2026-08-02 on data/raw/gr00t_apple (402 episodes), arm slice 15:29, hand slice 29:36.
# The gate below recomputes all twelve on every run and refuses to report anything if one moved.
# Quoted to the precision the document quotes them at, which is what ARCHIVE_ATOL_4DP and
# ARCHIVE_ATOL_3DP are sized against.

ARCHIVED_NUM_EPISODES = 402
"""Parquet files found under the archived dataset. A different count is a different dataset."""

ARCHIVED_NUM_GRASPS = 402
"""Episodes in which the detector found a debounced grasp: all of them.

Gated together with the episode count because the ratio is the detector's only self-check. A
detector that silently stopped finding grasps would not crash — it would report the same headline
on a shrinking, self-selected subset of episodes.
"""

ARCHIVED_GRASP_POSE_SPREAD = 0.6623
"""Between-episode spread of the grasp pose, L2 over the LIVE arm joints (7 left-arm, rad)."""

ARCHIVED_WITHIN_EPISODE_MOTION = 1.9132
"""Mean within-episode range, L2 over the same live joints (rad). The scale the spread is read on."""

ARCHIVED_CONSTANT_EXPLAINED_ALL_ARM = 0.7241
"""'Variance explained by a constant', over ALL arm joints in the slice — parked ones included.

This is the one archived number that is NOT live-joints-only, and PR-01-TASK-VARIATION.md's rider
("Every number above is therefore left-arm only") does not hold for it: the document's 0.7241 was
computed over all 14 arm joints, the live-joint value is 0.7925, and both are printed below so the
discrepancy is visible rather than resolved silently in one direction.

The quantity itself is narrower than its name suggests, and is worth stating precisely because it
is easy to over-read as "the target barely moves". It is
``1 - SS(pose about the per-joint mean) / SS(pose about a single scalar mean over all joints)``:
the share of the total squared deviation that is explained by knowing WHICH JOINT you are looking
at rather than which episode. A large value means the joints sit at different angles from each
other — which they do, trivially — not that the reach is stereotyped. :data:`GRASP_POSE_SPREAD`
against :data:`WITHIN_EPISODE_MOTION` is the number that answers "does the target move".
"""

ARCHIVED_GRASP_FRACTION_MEAN = 0.545
"""Grasp instant as a fraction of the episode, mean over episodes."""

ARCHIVED_GRASP_FRACTION_STD = 0.064
"""...and its standard deviation. Tight in relative terms, which is not the same as predictable."""

ARCHIVED_R2_POSE_FROM_STATE = 0.6136
"""Cross-validated R^2 of the grasp pose from the full t=0 state (43 dims on the G1)."""

ARCHIVED_R2_POSE_FROM_ARM = 0.4178
"""...from the t=0 LIVE ARM JOINTS only (7 dims). The postural part of the postural part."""

ARCHIVED_R2_POSE_FROM_LENGTH = 0.0539
"""...from the episode length alone (1 dim). A control: a long episode is not a different reach."""

ARCHIVED_R2_TIME_FROM_STATE = 0.0771
"""Cross-validated R^2 of the grasp INSTANT from the t=0 state. The headline of the screen."""

ARCHIVED_RESIDUAL_POSE_SPREAD = 0.3594
"""In-sample residual grasp-pose spread after the t=0 state is regressed out (rad).

In-sample on purpose: it is the archived quantity, and it is a magnitude ("how much reach target is
left over"), not a performance claim, so the leak cannot flatter a comparison that is not being
made. The out-of-fold value is printed beside it and is the one to quote if it ever is.
"""

ARCHIVE_ATOL_4DP = 5e-5
"""Tolerance for the values PR-01-TASK-VARIATION.md quotes to four decimals.

Half a unit in the last quoted digit: the measurement must still ROUND to the archived string.
Anything looser would let a real drift through — every way this can go wrong (a moved dataset, a
changed slice, a broken debounce) moves these numbers in their first or second decimal, not their
fifth.
"""

ARCHIVE_ATOL_3DP = 5e-4
"""The same rule for the two timing values, which the document quotes to three decimals."""

# -- the method, one constant per decision -------------------------------------------------------

DEFAULT_DATASET = Path("data/raw/gr00t_apple")
"""The archived dataset. ``--dataset`` overrides it; the gate is what ties a RUN to this path."""

DEFAULT_STATE_COLUMN = "observation.state"
"""LeRobot's state column. The RAW parquet, not a converted episode: on ``gr00t_apple`` the
converted gripper channel is dead (T-31) and a grasp detector on it would find nothing, or worse,
find noise."""

DEFAULT_ARM_SLICE = "15:29"
"""G1 GR00T layout: state[15:29] are the 14 arm joints (left 15:22, right 22:29)."""

DEFAULT_HAND_SLICE = "29:36"
"""...and state[29:36] the 7 Dex3 LEFT-hand joints, which is the live hand on this corpus."""

GRASP_THRESHOLD = 0.5
"""Where the normalized hand synergy has to cross for a close to count. Midpoint of its own range."""

GRASP_DEBOUNCE_STEPS = 10
"""How many steps the synergy must STAY above the threshold for a crossing to count as a grasp.

Without it, a hand that dithers around its own midpoint — which is what a noisy or near-dead
channel does — produces a crossing within the first few steps of every episode, and the "grasp
pose" becomes the starting pose. That failure is invisible downstream: it still yields a pose per
episode, a spread, an R^2 and a verdict.
"""

GRASP_MIN_DYNAMIC_RANGE = 0.05
"""Below this peak-to-peak range the hand channel is refused and the episode contributes nothing.

Normalization by the channel's own min/max is what makes the detector embodiment-independent, and
it is also what makes it dangerous: divide a constant channel by its own range and pure numerical
noise becomes a full-scale open/close. This is the guard that stops that, and it is the direct
lesson of T-31, where a flat gripper channel read as a working one for weeks because every metric
downstream accepted whatever it was handed.
"""

PARKED_JOINT_RANGE_RAD = 0.05
"""A joint whose MEAN within-episode range is below this is parked, and stays out of the headline.

Not a tuned threshold: on ``gr00t_apple`` the live arm's smallest mean range is 0.478 rad and the
parked arm's largest is 0.014 rad, so anything in between selects the same seven joints. It is
written as a constant so a dataset where the gap is NOT 30x is a visible judgement call in the
per-joint table rather than a silent one.
"""

RIDGE_LAMBDA = 1e-2
"""Ridge penalty, FIXED rather than swept — the same value and the same reason as PR-01's.

A penalty chosen by looking at the score is a selection on the quantity being reported. This screen
exists to be trusted on datasets nobody has looked at yet, so there is nothing to select on.
"""

CV_FOLDS = 5
"""Cross-validation folds, over EPISODES. See :func:`assign_episode_folds`."""

ZERO_VARIANCE_EPS = 1e-9
"""Standard deviation below which a predictor column is dropped on that fold.

The G1 state carries columns that never move (locked joints, unused command channels).
Standardizing by their standard deviation is a division by ~0 that turns float noise into a
full-scale feature, and a ridge handed 43 of those fits the noise.
"""

# -- the screening rule --------------------------------------------------------------------------

SCREEN_POSE_R2_BLIND = 0.90
"""Above this, the reach target is in proprioception and a video branch cannot be what supplies it."""

SCREEN_TIMING_R2_BLIND = 0.50
"""Above this, WHEN to act is in proprioception too. Lower bar than the pose: the timing is one
scalar, and a blind readout that gets half of it is already most of the decision."""

SCREEN_MIN_RESIDUAL_FRACTION = 0.10
"""Residual grasp-pose spread, as a fraction of the within-episode motion, below which the
unexplained part is too small to be worth a model — whatever its R^2 says. A dataset can have an
unpredictable reach target that moves by a millimetre."""

VERDICT_CONSEQUENCE = {
    "BLIND-SUFFICIENT": (
        "where AND when to act are both implied by the starting state. A blind baseline will do "
        "well here, so this corpus cannot distinguish a video model from a linear map and should "
        "not be downloaded to try. Randomize the object placement at RECORDING time instead — it "
        "is free there and it is the difference between a corpus where vision must matter and one "
        "where it need not."
    ),
    "VISION-CANDIDATE": (
        "a blind linear readout of the starting state does NOT supply all of the task, and what is "
        "left over is large enough to matter. This is a ceiling on the BLIND side only: it says "
        "the information is not in proprioception, not that it is in the pixels."
    ),
    "MIXED": (
        "no clean call. Read the three numbers rather than the letter: which of pose and timing is "
        "blind-predictable is the finding, and the next step follows from that pattern."
    ),
}


def _fmt(value: float) -> str:
    return f"{value:.4f}"


# -- loading -------------------------------------------------------------------------------------


def parse_slice(text: str) -> slice:
    """``"15:29"`` -> ``slice(15, 29)``. Half-open, non-negative, non-empty.

    Negative bounds are refused rather than accepted with Python's wrap-around meaning: on a state
    vector ``-7:`` and ``36:43`` look interchangeable to a reader and are not to a dataset whose
    state has a different width, and the failure would be a plausible-looking number over the wrong
    joints.
    """
    parts = text.split(":")
    if len(parts) != 2:
        raise SystemExit(f"slice must be START:STOP, got {text!r}")
    try:
        start, stop = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise SystemExit(f"slice must be START:STOP with integer bounds, got {text!r}") from exc
    if start < 0 or stop < 0:
        raise SystemExit(f"slice bounds must be non-negative, got {text!r}")
    if stop <= start:
        raise SystemExit(f"slice {text!r} is empty (stop must be > start)")
    return slice(start, stop)


def find_parquet_files(dataset: Path) -> list[Path]:
    """Every episode parquet under ``dataset``, in SORTED order.

    Three layouts, tried in order: the directory IS a chunk of parquet files; it is a LeRobot root
    with ``data/chunk-*/``; or the files are somewhere below it. Sorted at every level, because the
    order fixes the fold assignment and a filesystem's directory order does not.
    """
    if dataset.is_file():
        return [dataset]
    if not dataset.is_dir():
        raise SystemExit(f"{dataset}: not a file or directory")
    direct = sorted(dataset.glob("*.parquet"))
    if direct:
        return direct
    chunked = sorted(dataset.glob("data/chunk-*/*.parquet"))
    if chunked:
        return chunked
    anywhere = sorted(dataset.rglob("*.parquet"))
    if not anywhere:
        raise SystemExit(f"{dataset}: no .parquet files found")
    return anywhere


def read_episode_states(path: Path, state_column: str) -> np.ndarray:
    """[T, state_dim] float64 from one episode parquet. Raises on anything ragged or absent."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=[state_column])
    values = np.asarray(table[state_column].to_pylist(), dtype=np.float64)
    if values.ndim != 2:
        raise SystemExit(
            f"{path}: column {state_column!r} is not a fixed-width vector per row "
            f"(got array of shape {values.shape}). Rows of differing length cannot be stacked into "
            "a state matrix, and a partial read would be a state vector with the wrong meaning."
        )
    if values.shape[0] == 0:
        raise SystemExit(f"{path}: no rows")
    return values


def iter_episode_states(
    paths: Sequence[Path], state_column: str
) -> Iterator[tuple[str, np.ndarray]]:
    """``(episode_id, states)`` per file, in the order given. The id is the file stem."""
    for path in paths:
        yield path.stem, read_episode_states(path, state_column)


# -- the grasp detector --------------------------------------------------------------------------


def detect_grasp(
    hand: np.ndarray,
    *,
    threshold: float = GRASP_THRESHOLD,
    debounce_steps: int = GRASP_DEBOUNCE_STEPS,
    min_dynamic_range: float = GRASP_MIN_DYNAMIC_RANGE,
) -> int | None:
    """Index of the first SUSTAINED close, or None if this channel has no grasp in it.

    The synergy is the mean over the hand joints, normalized by the episode's OWN min/max so the
    detector does not need to know the embodiment's joint conventions or units. A grasp is the first
    upward crossing of ``threshold`` that stays above it for ``debounce_steps`` consecutive samples.

    Two refusals, both of which return None rather than a plausible index:

    - a channel whose peak-to-peak range is below ``min_dynamic_range`` (see
      :data:`GRASP_MIN_DYNAMIC_RANGE`) — the normalization would amplify noise to full scale;
    - an episode too short to contain a debounced crossing at all.

    A channel that is already above the threshold at t=0 has no upward CROSSING and yields None,
    which is the honest answer: the close was not observed.
    """
    if debounce_steps < 1:
        raise SystemExit(f"debounce_steps must be >= 1, got {debounce_steps}")
    if hand.ndim != 2 or hand.shape[1] == 0:
        raise SystemExit(f"hand channel must be [T, joints] with joints >= 1, got {hand.shape}")
    synergy = hand.mean(axis=1)
    low, high = float(synergy.min()), float(synergy.max())
    if high - low < min_dynamic_range:
        return None
    normalized = (synergy - low) / (high - low)
    closed = normalized > threshold
    for i in range(1, len(closed) - debounce_steps + 1):
        if not closed[i - 1] and closed[i] and closed[i : i + debounce_steps].all():
            return i
    return None


# -- cross-validation, folds over episodes -------------------------------------------------------


def assign_episode_folds(episode_ids: np.ndarray, num_folds: int) -> np.ndarray:
    """[N] fold index per row, assigned over EPISODES: ``sorted(unique)[i] -> i % num_folds``.

    Over episodes and never over rows. Here the design happens to carry one row per episode, so the
    two coincide — today. They stop coinciding the moment anything contributes a second row per
    episode (a second grasp, a bootstrap, a sliding window), and at that point a row-wise map fits
    the ridge on near-copies of the rows it scores and the R^2 it reports is memorisation. The
    difference is invisible from the outside: same folds, same table, same verdict, a number that
    moved. So the map is built over episodes unconditionally and :func:`check_folds_are_episode_
    disjoint` proves it against the row tags before a single weight is fitted.

    Deterministic and seed-free, so a rerun is the same run. Round-robin over the sorted ids rather
    than contiguous blocks, so a fold is not one contiguous run of recording sessions that could
    share a session-level artefact.
    """
    if num_folds < 2:
        raise SystemExit(f"need at least 2 folds, got {num_folds}")
    unique = sorted({str(e) for e in episode_ids.tolist()})
    if len(unique) < num_folds:
        raise SystemExit(
            f"{len(unique)} episode(s) cannot be split into {num_folds} folds over episodes. "
            "Splitting over rows instead would fit the ridge on neighbours of the rows it scores."
        )
    fold_of_episode = {episode_id: i % num_folds for i, episode_id in enumerate(unique)}
    return np.asarray([fold_of_episode[str(e)] for e in episode_ids.tolist()], dtype=np.int64)


def check_folds_are_episode_disjoint(
    episode_ids: np.ndarray, fold_of_row: np.ndarray, num_folds: int
) -> None:
    """Refuse a fold map that puts one episode's rows on both sides of any fold."""
    if fold_of_row.shape[0] != episode_ids.shape[0]:
        raise SystemExit(
            f"fold map has {fold_of_row.shape[0]} entries for {episode_ids.shape[0]} rows"
        )
    for k in range(num_folds):
        scored = fold_of_row == k
        fitted = ~scored
        if not scored.any() or not fitted.any():
            raise SystemExit(f"fold {k} is empty on one side; cannot cross-validate")
        both = set(episode_ids[fitted].tolist()) & set(episode_ids[scored].tolist())
        if both:
            raise SystemExit(
                f"fold {k} would fit the ridge on {len(both)} episode(s) it then scores, e.g. "
                f"{sorted(str(e) for e in both)[:3]}. The folds are not over episodes, so the R^2 "
                "below would be read off memorised neighbours of the rows it is scored on and "
                "would overstate how much of the task is in proprioception."
            )


def out_of_fold_predictions(
    x: np.ndarray,
    y: np.ndarray,
    fold_of_row: np.ndarray,
    *,
    lam: float = RIDGE_LAMBDA,
    eps: float = ZERO_VARIANCE_EPS,
) -> np.ndarray:
    """[N, targets] ridge predictions, each row predicted by weights fitted without its fold.

    Standardization is fitted on the FIT fold only — mean, standard deviation and the live-column
    mask alike. Standardizing on all rows first is the classic quiet leak: the scored rows would
    have contributed to the scale the model is fitted in, and on 402 episodes that is worth a
    couple of points of R^2 with nothing on screen to show for it.

    Columns whose fit-fold standard deviation is below ``eps`` are dropped rather than divided by
    (:data:`ZERO_VARIANCE_EPS`). Only an intercept survives in the degenerate case where none is
    live, which predicts the fit fold's mean — the right answer when there are no usable features.
    """
    if lam <= 0.0:
        raise SystemExit(f"ridge lambda must be > 0, got {lam:g}")
    if x.shape[0] != y.shape[0]:
        raise SystemExit(f"{x.shape[0]} predictor rows vs {y.shape[0]} target rows")
    predictions = np.zeros(y.shape, dtype=np.float64)
    for k in np.unique(fold_of_row):
        scored = fold_of_row == k
        fitted = ~scored
        if not scored.any() or not fitted.any():
            raise SystemExit(f"fold {k} is empty on one side; cannot cross-validate")
        mean = x[fitted].mean(axis=0)
        std = x[fitted].std(axis=0)
        live = std > eps
        design_fit = _design(x[fitted], mean, std, live)
        design_scored = _design(x[scored], mean, std, live)
        gram = design_fit.T @ design_fit
        weights = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), design_fit.T @ y[fitted])
        predictions[scored] = design_scored @ weights
    return predictions


def _design(x: np.ndarray, mean: np.ndarray, std: np.ndarray, live: np.ndarray) -> np.ndarray:
    """Standardized live columns plus an intercept."""
    standardized = (x[:, live] - mean[live]) / std[live]
    return np.hstack([standardized, np.ones((x.shape[0], 1))])


def r2_score(y: np.ndarray, predictions: np.ndarray) -> float:
    """``1 - SS_res / SS_tot`` over every element, against the mean of each target column.

    Pooled over target columns rather than averaged per column: the question is how much of the
    reach-target variation is explained, and a per-column average would weight a joint that barely
    moves the same as the one carrying the reach.
    """
    residual = float(((y - predictions) ** 2).sum())
    total = float(((y - y.mean(axis=0)) ** 2).sum())
    if total <= 0.0:
        raise SystemExit(
            "the target is constant across every episode, so R^2 is undefined. Nothing varies here "
            "and there is nothing for any model, blind or otherwise, to predict."
        )
    return 1.0 - residual / total


def cross_validated_r2(
    x: np.ndarray,
    y: np.ndarray,
    episode_ids: np.ndarray,
    *,
    lam: float = RIDGE_LAMBDA,
    num_folds: int = CV_FOLDS,
    eps: float = ZERO_VARIANCE_EPS,
) -> float:
    """Out-of-fold R^2 with folds over episodes, having PROVEN the folds are episode-disjoint."""
    fold_of_row = assign_episode_folds(episode_ids, num_folds)
    check_folds_are_episode_disjoint(episode_ids, fold_of_row, num_folds)
    return r2_score(y, out_of_fold_predictions(x, y, fold_of_row, lam=lam, eps=eps))


def in_sample_predictions(
    x: np.ndarray, y: np.ndarray, *, lam: float = RIDGE_LAMBDA, eps: float = ZERO_VARIANCE_EPS
) -> np.ndarray:
    """The same ridge fitted on ALL rows and read back on them. LEAKED, and used for one thing.

    :data:`ARCHIVED_RESIDUAL_POSE_SPREAD` is a magnitude, not a comparison, and this is the fit it
    was archived from. Never hand this to :func:`r2_score` and print the result as a result.
    """
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    live = std > eps
    design = _design(x, mean, std, live)
    gram = design.T @ design
    weights = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), design.T @ y)
    return design @ weights


# -- collection ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class GraspTable:
    """One row per episode in which a grasp was detected, plus the ones that were refused.

    ``arm_poses`` and ``joint_ranges`` cover EVERY joint in the arm slice, parked ones included:
    the parked/live split is a reported finding (:func:`live_joint_mask`) and cannot be made before
    the ranges exist.
    """

    episode_ids: np.ndarray  # [N] str
    start_states: np.ndarray  # [N, state_dim] the full state at t=0
    arm_poses: np.ndarray  # [N, arm_joints] the arm pose at the grasp
    joint_ranges: np.ndarray  # [N, arm_joints] within-episode peak-to-peak
    grasp_indices: np.ndarray  # [N] the grasp sample index
    episode_lengths: np.ndarray  # [N] samples in the episode
    arm_state_indices: np.ndarray  # [arm_joints] absolute state column of each arm joint
    num_episodes_seen: int
    refused_episode_ids: tuple[str, ...]

    @property
    def num_grasps(self) -> int:
        return int(self.episode_ids.shape[0])

    @property
    def grasp_fractions(self) -> np.ndarray:
        """[N] grasp instant as a fraction of the episode."""
        return self.grasp_indices / self.episode_lengths


def collect_grasps(
    episodes: Iterable[tuple[str, np.ndarray]],
    *,
    arm: slice,
    hand: slice,
    threshold: float = GRASP_THRESHOLD,
    debounce_steps: int = GRASP_DEBOUNCE_STEPS,
    min_dynamic_range: float = GRASP_MIN_DYNAMIC_RANGE,
) -> GraspTable:
    """Walk episodes once, keeping the grasp pose, the motion range and the timing of each."""
    episode_ids: list[str] = []
    starts: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    ranges: list[np.ndarray] = []
    indices: list[int] = []
    lengths: list[int] = []
    refused: list[str] = []
    seen = 0

    for episode_id, states in episodes:
        seen += 1
        if states.ndim != 2:
            raise SystemExit(f"{episode_id}: states must be [T, state_dim], got {states.shape}")
        width = states.shape[1]
        if arm.stop > width or hand.stop > width:
            raise SystemExit(
                f"{episode_id}: state is {width} wide but the slices ask for arm {arm.start}:"
                f"{arm.stop} and hand {hand.start}:{hand.stop}. Check --arm-slice / --hand-slice "
                "against this embodiment; a silently truncated slice would measure other joints."
            )
        index = detect_grasp(
            states[:, hand],
            threshold=threshold,
            debounce_steps=debounce_steps,
            min_dynamic_range=min_dynamic_range,
        )
        if index is None:
            refused.append(episode_id)
            continue
        episode_ids.append(episode_id)
        starts.append(states[0])
        poses.append(states[index, arm])
        ranges.append(np.ptp(states[:, arm], axis=0))
        indices.append(index)
        lengths.append(states.shape[0])

    if not episode_ids:
        raise SystemExit(
            f"no grasp was detected in any of {seen} episode(s). Either the hand slice does not "
            "point at a hand, or the channel is flat — check the dynamic-range line in the report "
            "before believing anything about this dataset."
        )
    return GraspTable(
        episode_ids=np.asarray(episode_ids, dtype=object),
        start_states=np.stack(starts),
        arm_poses=np.stack(poses),
        joint_ranges=np.stack(ranges),
        grasp_indices=np.asarray(indices, dtype=np.float64),
        episode_lengths=np.asarray(lengths, dtype=np.float64),
        # Recorded from the SAME slice the poses were taken through, so the "starting arm pose"
        # predictor cannot drift away from the joints whose grasp pose it predicts.
        arm_state_indices=np.arange(arm.start, arm.stop, dtype=np.int64),
        num_episodes_seen=seen,
        refused_episode_ids=tuple(refused),
    )


def live_joint_mask(
    joint_ranges: np.ndarray, threshold: float = PARKED_JOINT_RANGE_RAD
) -> np.ndarray:
    """[arm_joints] bool — which joints actually MOVE within an episode.

    A joint below the threshold is parked: it holds one angle for a whole episode and is re-posed
    between episodes. Its between-episode spread is therefore real and large and about nothing, and
    including it in the headline L2 inflates "how much does the reach target move" with how much the
    operator moved a limb that is not doing the task. See :data:`PARKED_JOINT_RANGE_RAD`.
    """
    return joint_ranges.mean(axis=0) > threshold


# -- the measurement -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Measurement:
    """Everything the report prints and the gate checks, computed once."""

    num_episodes_seen: int
    num_grasps: int
    live_mask: np.ndarray
    between_episode_std: np.ndarray  # [arm_joints]
    mean_within_range: np.ndarray  # [arm_joints]
    grasp_pose_spread: float  # L2 over live joints
    within_episode_motion: float  # L2 over live joints
    constant_explained_all_arm: float
    constant_explained_live: float
    grasp_fraction_mean: float
    grasp_fraction_std: float
    r2_pose_from_state: float
    r2_pose_from_arm: float
    r2_pose_from_length: float
    r2_time_from_state: float
    residual_pose_spread: float  # in-sample; the archived one
    residual_pose_spread_oof: float  # out-of-fold; the honest one

    @property
    def residual_fraction(self) -> float:
        """Residual reach target as a fraction of the motion scale. The screen's size check."""
        return self.residual_pose_spread / max(self.within_episode_motion, 1e-12)


def _explained_by_a_constant(poses: np.ndarray) -> float:
    """``1 - SS(about the per-joint mean) / SS(about one scalar mean)`` — see the archive note.

    Reproduced exactly as PR-01-TASK-VARIATION.md computed it, including the scalar denominator.
    """
    residual = float(((poses - poses.mean(axis=0)) ** 2).sum())
    total = float(((poses - poses.mean()) ** 2).sum())
    if total <= 0.0:
        raise SystemExit("every arm joint is at the same constant angle in every episode")
    return 1.0 - residual / total


def measure(
    table: GraspTable,
    *,
    parked_range: float = PARKED_JOINT_RANGE_RAD,
    lam: float = RIDGE_LAMBDA,
    num_folds: int = CV_FOLDS,
) -> Measurement:
    """The three measurements, on the live joints, with folds over episodes."""
    live = live_joint_mask(table.joint_ranges, parked_range)
    if not live.any():
        raise SystemExit(
            f"no arm joint moves more than {parked_range:g} rad within an episode. Either the arm "
            "slice does not point at the arm, or nothing in this dataset moves — and either way "
            "the between-episode spread would be entirely parked-pose nuisance."
        )

    between = table.arm_poses.std(axis=0)
    within = table.joint_ranges.mean(axis=0)
    poses_live = table.arm_poses[:, live]
    starts = table.start_states
    arm_at_start = starts[:, table.arm_state_indices[live]]
    lengths = table.episode_lengths.reshape(-1, 1)
    grasp_time = table.grasp_indices.reshape(-1, 1)
    ids = table.episode_ids

    kw: dict[str, Any] = {"lam": lam, "num_folds": num_folds}
    residual_in_sample = poses_live - in_sample_predictions(starts, poses_live, lam=lam)
    fold_of_row = assign_episode_folds(ids, num_folds)
    check_folds_are_episode_disjoint(ids, fold_of_row, num_folds)
    residual_oof = poses_live - out_of_fold_predictions(starts, poses_live, fold_of_row, lam=lam)

    return Measurement(
        num_episodes_seen=table.num_episodes_seen,
        num_grasps=table.num_grasps,
        live_mask=live,
        between_episode_std=between,
        mean_within_range=within,
        grasp_pose_spread=float(np.linalg.norm(between[live])),
        within_episode_motion=float(np.linalg.norm(within[live])),
        constant_explained_all_arm=_explained_by_a_constant(table.arm_poses),
        constant_explained_live=_explained_by_a_constant(poses_live),
        grasp_fraction_mean=float(table.grasp_fractions.mean()),
        grasp_fraction_std=float(table.grasp_fractions.std()),
        r2_pose_from_state=cross_validated_r2(starts, poses_live, ids, **kw),
        r2_pose_from_arm=cross_validated_r2(arm_at_start, poses_live, ids, **kw),
        r2_pose_from_length=cross_validated_r2(lengths, poses_live, ids, **kw),
        r2_time_from_state=cross_validated_r2(starts, grasp_time, ids, **kw),
        residual_pose_spread=float(np.linalg.norm(residual_in_sample.std(axis=0))),
        residual_pose_spread_oof=float(np.linalg.norm(residual_oof.std(axis=0))),
    )


# -- the archive gate ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchivedCheck:
    """One archived number, what this run measured for it, and how close it has to be.

    ``decimals`` is the precision PR-01-TASK-VARIATION.md quotes the value at, and it drives both
    the printout and — through ``atol`` — the gate. A count is quoted at 0 decimals with an ``atol``
    of 0: 401 episodes of 402 is not a rounding difference, it is a different dataset.
    """

    name: str
    measured: float
    expected: float
    atol: float
    decimals: int = 4

    @property
    def ok(self) -> bool:
        return bool(abs(self.measured - self.expected) <= self.atol)

    def format(self, value: float) -> str:
        return f"{value:.{self.decimals}f}"


def archived_checks(measurement: Measurement) -> tuple[ArchivedCheck, ...]:
    """PR-01-TASK-VARIATION.md's numbers, paired with what this run measured for each."""
    four = ARCHIVE_ATOL_4DP
    three = ARCHIVE_ATOL_3DP
    return (
        ArchivedCheck(
            "episodes seen", measurement.num_episodes_seen, ARCHIVED_NUM_EPISODES, 0.0, 0
        ),
        ArchivedCheck(
            "episodes with a grasp", measurement.num_grasps, ARCHIVED_NUM_GRASPS, 0.0, 0
        ),
        ArchivedCheck(
            name="grasp-pose spread (L2, live)",
            measured=measurement.grasp_pose_spread,
            expected=ARCHIVED_GRASP_POSE_SPREAD,
            atol=four,
        ),
        ArchivedCheck(
            name="within-episode motion (L2)",
            measured=measurement.within_episode_motion,
            expected=ARCHIVED_WITHIN_EPISODE_MOTION,
            atol=four,
        ),
        ArchivedCheck(
            name="explained by a constant (all arm)",
            measured=measurement.constant_explained_all_arm,
            expected=ARCHIVED_CONSTANT_EXPLAINED_ALL_ARM,
            atol=four,
        ),
        ArchivedCheck(
            name="grasp fraction, mean",
            measured=measurement.grasp_fraction_mean,
            expected=ARCHIVED_GRASP_FRACTION_MEAN,
            atol=three,
            decimals=3,
        ),
        ArchivedCheck(
            name="grasp fraction, std",
            measured=measurement.grasp_fraction_std,
            expected=ARCHIVED_GRASP_FRACTION_STD,
            atol=three,
            decimals=3,
        ),
        ArchivedCheck(
            name="R^2 pose from the t=0 state",
            measured=measurement.r2_pose_from_state,
            expected=ARCHIVED_R2_POSE_FROM_STATE,
            atol=four,
        ),
        ArchivedCheck(
            name="R^2 pose from the t=0 arm",
            measured=measurement.r2_pose_from_arm,
            expected=ARCHIVED_R2_POSE_FROM_ARM,
            atol=four,
        ),
        ArchivedCheck(
            name="R^2 pose from episode length",
            measured=measurement.r2_pose_from_length,
            expected=ARCHIVED_R2_POSE_FROM_LENGTH,
            atol=four,
        ),
        ArchivedCheck(
            name="R^2 grasp TIME from the state",
            measured=measurement.r2_time_from_state,
            expected=ARCHIVED_R2_TIME_FROM_STATE,
            atol=four,
        ),
        ArchivedCheck(
            name="residual pose spread (in-sample)",
            measured=measurement.residual_pose_spread,
            expected=ARCHIVED_RESIDUAL_POSE_SPREAD,
            atol=four,
        ),
    )


def check_archive(measurement: Measurement) -> list[str]:
    """Return the gate's lines, or raise if any archived number has moved.

    Raises rather than warns, and raises BEFORE the verdict is printed, for the reason PR-01 gives
    about its own controls: a warning above a table of four-decimal floats is a warning nobody acts
    on, and a screen whose loader has drifted is not a screen, it is a number.
    """
    checks = archived_checks(measurement)
    lines: list[str] = []
    failed: list[str] = []
    for check in checks:
        state = "OK" if check.ok else "DRIFT"
        lines.append(
            f"  {check.name:<36}{check.format(check.measured):>10}"
            f"   archived {check.format(check.expected):>8}   {state}"
        )
        if not check.ok:
            failed.append(
                f"{check.name}: measured {check.measured:.6f}, "
                f"archived {check.format(check.expected)}"
            )
    passed = sum(1 for c in checks if c.ok)
    if failed:
        raise SystemExit(
            f"archive {passed}/{len(checks)} — this run does NOT reproduce "
            "docs/preregistration/PR-01-TASK-VARIATION.md, so none of its numbers are that "
            "document's and none are interpreted:\n  "
            + "\n  ".join(failed)
            + "\nCheck --dataset, --arm-slice, --hand-slice and --state-column. Use "
            "--archive-gate off to screen a DIFFERENT dataset."
        )
    lines.append(f"  archive {passed}/{len(checks)} reproduce exactly")
    return lines


# -- the verdict ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    letter: str
    clauses: dict[str, bool]
    lines: tuple[str, ...]


def decide(measurement: Measurement) -> Verdict:
    """Evaluate the screening rule mechanically. Nothing here is judged by eye."""
    pose_is_blind = measurement.r2_pose_from_state >= SCREEN_POSE_R2_BLIND
    timing_is_blind = measurement.r2_time_from_state >= SCREEN_TIMING_R2_BLIND
    residual_matters = measurement.residual_fraction >= SCREEN_MIN_RESIDUAL_FRACTION

    clauses = {
        "grasp pose is blind-predictable": pose_is_blind,
        "grasp timing is blind-predictable": timing_is_blind,
        "what is left over is large enough to matter": residual_matters,
    }
    where = "in proprioception"
    lines = (
        (
            f"  pose   R^2 {measurement.r2_pose_from_state:+.4f} vs {SCREEN_POSE_R2_BLIND:.2f}"
            f"   ->  {where if pose_is_blind else 'NOT ' + where}"
        ),
        (
            f"  timing R^2 {measurement.r2_time_from_state:+.4f} vs {SCREEN_TIMING_R2_BLIND:.2f}"
            f"   ->  {where if timing_is_blind else 'NOT ' + where}"
        ),
        (
            f"  residual {measurement.residual_pose_spread:.4f} rad = "
            f"{measurement.residual_fraction:.3f} of the motion scale, vs "
            f"{SCREEN_MIN_RESIDUAL_FRACTION:.2f}"
            f"   ->  {'material' if residual_matters else 'too small to chase'}"
        ),
    )

    if pose_is_blind and timing_is_blind:
        letter = "BLIND-SUFFICIENT"
    elif residual_matters and not (pose_is_blind and timing_is_blind):
        letter = "VISION-CANDIDATE"
    else:
        letter = "MIXED"
    return Verdict(letter=letter, clauses=clauses, lines=lines)


# -- report --------------------------------------------------------------------------------------


def format_report(
    measurement: Measurement,
    *,
    dataset: Path,
    num_files: int,
    arm: slice,
    hand: slice,
    state_dim: int,
    gate_lines: Sequence[str],
    gate_on: bool,
    verdict: Verdict,
    refused: Sequence[str],
) -> str:
    """The whole printout as one string, so the determinism test can compare two of them."""
    live = measurement.live_mask
    out: list[str] = []
    out.append("measure_task_variation — does this dataset require vision? (proprioception only)")
    out.append(f"dataset  {dataset}  ({num_files} parquet file(s))")
    out.append(
        f"slices   arm {arm.start}:{arm.stop} ({arm.stop - arm.start} joints)   "
        f"hand {hand.start}:{hand.stop} ({hand.stop - hand.start} joints)   state {state_dim} dims"
    )
    out.append(
        f"grasp    first upward crossing of {GRASP_THRESHOLD:g} in the per-episode normalized hand "
        f"synergy, held {GRASP_DEBOUNCE_STEPS} steps; refused below a range of "
        f"{GRASP_MIN_DYNAMIC_RANGE:g}"
    )
    out.append("")
    if gate_on:
        out.append("archived controls (gate — every number below is void unless these reproduce)")
        out.extend(gate_lines)
    else:
        out.append("ARCHIVE GATE OFF — this run reproduces nothing and its numbers are not PR-01's.")
    out.append("")

    out.append(
        f"1 — does the reach target move?   grasp detected in {measurement.num_grasps} of "
        f"{measurement.num_episodes_seen} episode(s)"
    )
    if refused:
        shown = ", ".join(refused[:5]) + (" ..." if len(refused) > 5 else "")
        out.append(f"    refused (no debounced grasp in the hand channel): {shown}")
    out.append("  joint  state          between-episode std   mean within-episode range      ratio")
    for j in range(measurement.between_episode_std.shape[0]):
        tag = "LIVE  " if live[j] else "PARKED"
        ratio = measurement.between_episode_std[j] / max(measurement.mean_within_range[j], 1e-12)
        out.append(
            f"  {j:>5}  {arm.start + j:>5}  {tag}  {measurement.between_episode_std[j]:>19.4f}"
            f"  {measurement.mean_within_range[j]:>25.4f}  {ratio:>9.3f}"
        )
    out.append(
        f"    a joint below {PARKED_JOINT_RANGE_RAD:g} rad of within-episode range is parked, and "
        "is kept out of every headline below: its between-episode spread is a re-posed limb, not "
        "task variation."
    )
    out.append("")
    ratio = measurement.grasp_pose_spread / max(measurement.within_episode_motion, 1e-12)
    live_n = int(live.sum())
    for label, value in (
        (
            f"grasp-pose spread, L2 over {live_n} live joints",
            f"{_fmt(measurement.grasp_pose_spread)} rad",
        ),
        ("within-episode motion, same joints", f"{_fmt(measurement.within_episode_motion)} rad"),
        ("ratio", f"{ratio:.4f}"),
        ("explained by a constant (live joints)", _fmt(measurement.constant_explained_live)),
        (
            "explained by a constant (ALL arm joints)",
            (
                f"{_fmt(measurement.constant_explained_all_arm)}   "
                "<- the archived value; parked joints included"
            ),
        ),
        (
            "grasp timing, fraction of the episode",
            f"mean {measurement.grasp_fraction_mean:.3f}  std {measurement.grasp_fraction_std:.3f}",
        ),
    ):
        out.append(f"  {label:<42}{value}")
    out.append("")

    out.append(
        f"2 — is it already in proprioception?   cross-validated R^2, {CV_FOLDS} folds over EPISODES"
    )
    for label, value in (
        ("grasp POSE from the full t=0 state", f"{measurement.r2_pose_from_state:+.4f}"),
        ("grasp POSE from the t=0 live arm only", f"{measurement.r2_pose_from_arm:+.4f}"),
        ("grasp POSE from the episode length", f"{measurement.r2_pose_from_length:+.4f}"),
        ("grasp POSE from a constant", "+0.0000   <- by definition"),
        (
            "residual spread, in-sample",
            (
                f"{_fmt(measurement.residual_pose_spread)} rad "
                f"(vs a raw {_fmt(measurement.grasp_pose_spread)})"
            ),
        ),
        (
            "residual spread, out-of-fold",
            f"{_fmt(measurement.residual_pose_spread_oof)} rad   <- the honest one",
        ),
    ):
        out.append(f"  {label:<42}{value}")
    out.append("")

    out.append("3 — is the TIMING in proprioception?")
    out.append(
        f"  {'grasp INSTANT from the full t=0 state':<42}{measurement.r2_time_from_state:+.4f}"
    )
    out.append("")

    out.append("screening rule")
    out.extend(verdict.lines)
    out.append("")
    out.append(f"VERDICT {verdict.letter} — {VERDICT_CONSEQUENCE[verdict.letter]}")
    if not gate_on:
        out.append("(archive gate was OFF: this verdict is about whatever data was passed.)")
    return "\n".join(out)


# -- CLI -----------------------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET,
        help="LeRobot-style raw parquet directory (default: %(default)s)",
    )  # fmt: skip
    parser.add_argument(
        "--state-column", default=DEFAULT_STATE_COLUMN,
        help="parquet column holding the state vector (default: %(default)s)",
    )  # fmt: skip
    parser.add_argument(
        "--arm-slice", default=DEFAULT_ARM_SLICE,
        help="START:STOP of the arm joints in the state (default: %(default)s, the G1's 14 arm "
        "joints; left arm 15:22, right 22:29)",
    )  # fmt: skip
    parser.add_argument(
        "--hand-slice", default=DEFAULT_HAND_SLICE,
        help="START:STOP of the hand joints the grasp is detected in (default: %(default)s, the "
        "G1's 7 left-hand joints)",
    )  # fmt: skip
    parser.add_argument(
        "--archive-gate", choices=("pr-01", "off"), default="pr-01",
        help="'pr-01' (default) refuses to report anything unless PR-01-TASK-VARIATION.md's twelve "
        "numbers reproduce. 'off' is for screening a DIFFERENT dataset, and says so in the output",
    )  # fmt: skip
    parser.add_argument("--json", type=Path, default=None, help="write the full record here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    arm = parse_slice(args.arm_slice)
    hand = parse_slice(args.hand_slice)
    if not (hand.stop <= arm.start or arm.stop <= hand.start):
        raise SystemExit(
            f"--arm-slice {args.arm_slice} and --hand-slice {args.hand_slice} overlap. The grasp "
            "would then be detected in the same channels whose pose it is used to read, and the "
            "'grasp pose' would be a restatement of the detector's own threshold."
        )

    files = find_parquet_files(args.dataset)
    table = collect_grasps(iter_episode_states(files, args.state_column), arm=arm, hand=hand)
    measurement = measure(table)
    verdict = decide(measurement)

    gate_on = args.archive_gate != "off"
    gate_lines = check_archive(measurement) if gate_on else []

    report = format_report(
        measurement,
        dataset=args.dataset,
        num_files=len(files),
        arm=arm,
        hand=hand,
        state_dim=int(table.start_states.shape[1]),
        gate_lines=gate_lines,
        gate_on=gate_on,
        verdict=verdict,
        refused=table.refused_episode_ids,
    )
    print(report)

    if args.json is not None:
        record: dict[str, Any] = {
            "writeup": "docs/preregistration/PR-01-TASK-VARIATION.md",
            "dataset": str(args.dataset),
            "num_parquet_files": len(files),
            "state_column": args.state_column,
            "arm_slice": args.arm_slice,
            "hand_slice": args.hand_slice,
            "archive_gate": args.archive_gate,
            "num_episodes_seen": measurement.num_episodes_seen,
            "num_grasps": measurement.num_grasps,
            "refused_episode_ids": list(table.refused_episode_ids),
            "live_joints": [int(arm.start + j) for j in np.flatnonzero(measurement.live_mask)],
            "between_episode_std": [float(v) for v in measurement.between_episode_std],
            "mean_within_episode_range": [float(v) for v in measurement.mean_within_range],
            "grasp_pose_spread": measurement.grasp_pose_spread,
            "within_episode_motion": measurement.within_episode_motion,
            "constant_explained_all_arm": measurement.constant_explained_all_arm,
            "constant_explained_live": measurement.constant_explained_live,
            "grasp_fraction_mean": measurement.grasp_fraction_mean,
            "grasp_fraction_std": measurement.grasp_fraction_std,
            "r2_pose_from_state": measurement.r2_pose_from_state,
            "r2_pose_from_arm": measurement.r2_pose_from_arm,
            "r2_pose_from_length": measurement.r2_pose_from_length,
            "r2_time_from_state": measurement.r2_time_from_state,
            "residual_pose_spread_in_sample": measurement.residual_pose_spread,
            "residual_pose_spread_out_of_fold": measurement.residual_pose_spread_oof,
            "verdict": verdict.letter,
            "verdict_clauses": verdict.clauses,
            "thresholds": {
                "GRASP_THRESHOLD": GRASP_THRESHOLD,
                "GRASP_DEBOUNCE_STEPS": GRASP_DEBOUNCE_STEPS,
                "GRASP_MIN_DYNAMIC_RANGE": GRASP_MIN_DYNAMIC_RANGE,
                "PARKED_JOINT_RANGE_RAD": PARKED_JOINT_RANGE_RAD,
                "RIDGE_LAMBDA": RIDGE_LAMBDA,
                "CV_FOLDS": CV_FOLDS,
                "SCREEN_POSE_R2_BLIND": SCREEN_POSE_R2_BLIND,
                "SCREEN_TIMING_R2_BLIND": SCREEN_TIMING_R2_BLIND,
                "SCREEN_MIN_RESIDUAL_FRACTION": SCREEN_MIN_RESIDUAL_FRACTION,
            },
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
