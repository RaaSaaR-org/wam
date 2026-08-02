#!/usr/bin/env python3
"""PR-01 — does the model know anything a linear map on proprioception does not?

The pre-registration is ``docs/preregistration/PR-01-incremental-value.md``, committed before this
file existed so ``git log`` proves the rule predates the number. This script IMPLEMENTS that
document and does not adjust it: every threshold below is one of its clauses, transcribed into a
named module constant so it can be grepped, diffed and versioned instead of living in a comparison
buried in a print statement. The verdict (A, B or C) is evaluated MECHANICALLY from the measured
numbers and printed. Nothing is left to a reader's judgement about which side of 0.95 a ratio fell.

WHAT QUESTION THIS ANSWERS
--------------------------
``scripts/bench_ridge_baseline.py`` established that the deployed Wan2.2-5B + LoRA checkpoint loses
1.76x on its own holdout to 7 920 linear parameters on the 32-dim robot state. That has two
readings and the ridge number alone separates neither: the model is empty (Reading A), or the
metric is empty (Reading B — chunk MSE on this data is dominated by momentum, and a metric a
no-parameter extrapolator nearly solves was never measuring skill). PR-01 is the four measurements
that tell them apart, all on CPU, in seconds, with no backbone built and no pixel read.

THE FIVE PREDICTORS
-------------------
::

    zero-delta          hold still. 0 parameters. The floor for "did anything at all".
    const-velocity      pred[k, j] = dq[j] * dt_s for all 16 steps. 0 FITTED parameters.
    ridge (dq only)     3 840 parameters, train episodes only.
    ridge (all state)   7 920 parameters, train episodes only. The bar in the pre-registration.
    model               the Wan-5B+LoRA run's own predictions, read from its predictions.jsonl.

``const-velocity`` is the load-bearing one and is worth stating precisely, because getting it wrong
would silently answer a different question. ``ActionChunk.targets`` is [16, 15] float32 of PER-STEP
joint deltas in radians, integrated onto the robot's current ``q`` by ``G1Adapter.execute``
(``src/wam/robot/g1.py``). Constant velocity over the chunk therefore means the SAME delta at every
one of the 16 steps, namely the distance the joint travels in one control period at its current
speed: ``dq * dt_s``. It is a fixed analytic rule — no least squares, no lambda, no train set, no
scaling coefficient fitted to make it look good. It exists to make Reading B visible directly: if a
rule with zero fitted parameters lands anywhere near a 7 920-parameter solve, the metric is a
momentum metric and cannot rank policies.

THE FOUR TESTS
--------------
T1  cross-fitted stacking (leak-free, PRIMARY). Two scalars ``(alpha, beta)`` in
    ``y ~ alpha*ridge + beta*model``, cross-fitted over 5 folds split OVER EPISODES inside the
    holdout — weights fitted on 4 folds, scored on the 5th, never on their own fold. ``beta`` is
    the model's incremental weight given the ridge is already there. It cannot be gamed by subset
    choice because there is no subset. The in-sample stack is printed next to it as a LEAKED
    diagnostic, so a reader can see what the cross-fitting cost and that it cost something.

T2  per-horizon breakdown (leak-free). MSE at each of the 16 chunk steps for every predictor.
    Momentum is exactly right at step 0 and decays; a model that learned the TASK rather than the
    VELOCITY should close ground as the horizon grows. No selection of any kind.

T3  branch points (target-selected; a model LOSS is decisive, a model WIN is not). The worst
    quartile of holdout chunks ranked by CONSTANT-VELOCITY error — deliberately neither of the two
    predictors being compared (``T3_RANKING_PREDICTOR``). This removes exactly the chunks the
    linear map is best at, so the thumb is on the model's side of the scale. A loss here is
    therefore strong evidence for Reading A; a win is weak and must not be reported as a win
    without T1 agreeing. Which predictor ranked the subset and which chunks it kept both go into
    the ``--json`` record, so the selection can be recomputed from the dataset instead of trusted.

T4  gripper transitions (target-selected; same one-directional admissibility as T3). Chunks whose
    demonstrated ``gripper_target`` crosses ``GRIPPER_BINARIZE_THRESHOLD``, reported both on the
    full flattened MSE and on the gripper channel alone — the joint channels are 15/16ths of the
    flattened number and would drown the event being asked about. The clause is evaluated on the
    RAW crossings (``T4_PRIMARY_DEBOUNCE``); the debounced count is printed beside it, and their
    selected chunks are recorded like T3's.

THE ARCHIVED CONTROLS ARE A HARD GATE
-------------------------------------
The pre-registration: "Every run must reproduce the archived controls (zero-delta 1.632760e-05,
model 1.112983e-05) before any new number is reported; a run that does not reproduce them is void,
not interpreted." So the gate runs FIRST and raises, and no test number is computed, let alone
printed, until it has passed. The all-state ridge at ``RIDGE_LAMBDA`` is gated with them, because
it is the denominator of two of the four decision clauses and a silent drift in it would move the
verdict without moving anything a reader is looking at.

``--archive-gate off`` exists for tests and for a run on other data, and prints a banner saying the
run reproduces nothing. It is not a way to publish a number.

WHAT IS NOT CLAIMED
-------------------
Nothing here says whether the model would work on a real robot. Every test is offline, on
demonstrations of one task, with the apple in the same place. E1 action-MSE is a DIAGNOSTIC metric
(PRD 10.4) and this is a diagnostic on that diagnostic. VERDICT A is a statement that the OFFLINE
EVIDENCE is worthless, not that the model is — which is exactly why the pre-registered follow-on
under A is a closed-loop task-success run rather than an architecture rewrite.

Nothing here tests generalization either. The ridge has no notion of where the apple is and would
not survive it moving; that is the generalization the video branch exists to buy, and no
fixed-scene holdout can test it.

And T4 in particular is only as good as the channel underneath it. This script measures the
holdout gripper channel's dynamic range against the pre-registered
``GRIPPER_MIN_DYNAMIC_RANGE`` and prints the verdict of that clause next to the T4 numbers. On a
channel that fails it, "the model beats the ridge on the gripper" is a statement about two
constants disagreeing slightly, and the printout says so rather than letting the ratio speak.

    scripts/bench_incremental_value.py \\
        --dataset datasets/gr00t-apple-full \\
        --holdout runs/t16-lora-seed0/eval-t30-regression/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import bench_ridge_baseline as ridge_baseline

from wam.evaluation.gripper import (
    GRIPPER_BINARIZE_THRESHOLD,
    GRIPPER_HYSTERESIS_MARGIN,
    GRIPPER_MIN_DYNAMIC_RANGE,
    crossings,
    debounced_transitions,
)
from wam.evaluation.offline import load_episode_ids, load_predictions_jsonl

# -- the archived controls, and what reproducing them means -------------------------------------

ARCHIVED_ZERO_DELTA_MSE = 1.632760e-05
"""``scripts/bench_ridge_baseline.py``, measured 2026-08-01. Named in PR-01 as a gate."""

ARCHIVED_MODEL_MSE = 1.112983e-05
"""The Wan-5B+LoRA run's own holdout MSE, recomputed from its predictions.jsonl. Gated."""

ARCHIVED_RIDGE_ALL_MSE = 6.330899e-06
"""The all-state ridge at ``RIDGE_LAMBDA``. Gated too: it is the denominator of two clauses."""

ARCHIVE_RTOL = 1e-6
"""Relative tolerance for the gate.

The archived values are written to seven significant digits, so the rounding alone is worth up to
~3e-7 of relative disagreement and the tolerance has to clear it. It must not clear more: every way
this measurement can be wrong (a moved dataset, a different chunk horizon, a misaligned holdout)
moves these numbers in their first three digits, not their seventh.
"""

# -- the pre-registered rule, one constant per clause --------------------------------------------

RIDGE_LAMBDA = 1e-2
"""The single ridge penalty, FIXED rather than swept.

``bench_ridge_baseline`` sweeps lambda and reports the best holdout MSE, and names that as a mild
selection on the holdout. PR-01 is a study OF holdout leakage and must not inherit one, so the
penalty is pinned instead of chosen. 1e-2 is the value the archived headline 6.330899e-06 was
quoted at; across four orders of magnitude the all-state number moves in its seventh significant
digit (6.330877e-06 at 1e1, 6.333218e-06 at 1e2), so pinning it costs nothing anyone can read.
"""

STACKING_FOLDS = 5
"""T1 cross-fitting folds. Split OVER EPISODES, never over rows — see :func:`episode_folds`."""

STACK_IMPROVEMENT_FRACTION = 0.95
"""VERDICT A clause 1: ``MSE_stack >= 0.95 * MSE_ridge``, i.e. stacking the model in buys < 5 %."""

MIN_INCREMENTAL_BETA = 0.05
"""VERDICT A clause 2, part one: a fold's ``beta`` counts as positive only above this."""

MIN_FOLDS_WITH_POSITIVE_BETA = 4
"""VERDICT A clause 2, part two: fewer than 4 of the 5 folds above ``MIN_INCREMENTAL_BETA``."""

BRANCH_POINT_QUANTILE = 0.75
"""T3 keeps the chunks ABOVE this quantile of constant-velocity error — the worst quartile."""

T3_RANKING_PREDICTOR = "const-velocity"
"""WHICH predictor T3 ranks the holdout on, as a named constant rather than a subscript in ``main``.

T3's entire one-directional admissibility rests on this being NEITHER of the two predictors being
compared. Rank on the model's own error and the subset becomes the chunks the model happens to be
best on; rank on the ridge's and it becomes the chunks the ridge is worst on. Either way the
printout still says "worst quartile", still has 260 chunks under it and still carries a verdict —
the number moves by an order of magnitude and nothing on the screen says why.

So the choice is a constant that can be grepped and diffed, the run refuses a ranking on a compared
predictor (:func:`branch_point_ranking`), and the subset it selected is written into the ``--json``
record so it can be recomputed from outside rather than taken on trust.
"""

T4_PRIMARY_DEBOUNCE = False
"""WHICH of the two gripper subsets VERDICT A's fourth clause is evaluated on.

``False`` — raw threshold crossings — is the literal reading of the pre-registration and is the
PRIMARY. The debounced count is computed and printed beside it, because where the two disagree the
disagreement is the finding about the channel; but it is a diagnostic, not the clause. The
difference is not cosmetic: on the T-16 holdout the raw subset has 17 chunks and the debounced one
has 0, i.e. swapping them turns an evaluated clause into an INDETERMINATE one and changes which
verdicts are reachable.
"""

# -- alignment ----------------------------------------------------------------------------------

ALIGNMENT_ATOL = 1e-9
"""How closely a prediction record's target must match the row it is aligned to.

Both sides are the same float32 numbers taking different routes: parquet -> float64 here, and
float32 -> ``tolist()`` -> JSON -> float64 there. Neither route loses a bit, so the honest
tolerance is "essentially exact" and anything looser would let a genuinely misaligned pair through
on two chunks that merely look similar.
"""

PREDICTOR_ORDER = (
    "zero-delta",
    "const-velocity",
    "ridge (dq only)",
    "ridge (all state)",
    "model",
)
"""Print order. ``ridge (all state)`` and ``model`` are the two the decision rule compares."""

RIDGE_KEY = "ridge (all state)"
MODEL_KEY = "model"


def _fmt(value: float) -> str:
    return f"{value:.6e}"


def mse(predicted: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error over every element — the SAME quantity the model was scored on."""
    return float(((predicted - target) ** 2).mean())


def per_chunk_mse(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """[N] — one MSE per chunk, over its own [16, 15] block. What T3 ranks on."""
    return ((predicted - target) ** 2).mean(axis=(1, 2))


def per_step_mse(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """[16] — MSE at each horizon step, averaged over chunks and joints. What T2 reports."""
    return ((predicted - target) ** 2).mean(axis=(0, 2))


# -- collection: what the ridge table does not carry ---------------------------------------------


@dataclass(frozen=True)
class ChunkExtras:
    """Per-chunk ``gripper_target`` and ``dt_s``, in ``collect_chunks``' row order.

    ``bench_ridge_baseline.ChunkTable`` carries the state and the flattened joint target and
    nothing else, because the ridge needs nothing else. PR-01 needs two more columns: ``dt_s`` for
    the constant-velocity rule and ``gripper_target`` for T4 — including on TRAIN rows, since the
    gripper-channel comparison needs a ridge fitted on that channel and a train-mean constant to
    read it against.

    Collected in a second pass rather than by editing the archived baseline, which is the control
    this whole run is gated on and is therefore left untouched. The second pass duplicates
    ``collect_chunks``' skip-short/skip-stateless rule, and a duplicated rule can drift — so the
    two are not trusted to agree, they are ASSERTED to agree row for row by
    :func:`check_extras_align`, over the episode tags rather than over the loops that produced
    them. On the holdout half they are checked a second time, against the predictions file
    (:func:`align_to_predictions`), which came from a third implementation of the same rule.
    """

    episode_ids: np.ndarray
    gripper_targets: np.ndarray  # [N, chunk_steps]
    dt_s: np.ndarray  # [N]


def collect_chunk_extras(dataset: Path, chunk_steps: int) -> ChunkExtras:
    """Walk the dataset exactly as ``collect_chunks`` does, keeping the two columns it drops."""
    from wam.data.episode import EpisodeReader, list_episodes

    episode_dirs = list_episodes(dataset)
    if not episode_dirs:
        raise SystemExit(f"{dataset}: no episodes found (nothing containing a manifest.json)")

    gripper: list[np.ndarray] = []
    dt_s: list[float] = []
    episode_ids: list[str] = []
    for episode_dir in episode_dirs:
        reader = EpisodeReader(episode_dir)
        episode_id = reader.manifest.episode_id
        if not reader.read_states():
            continue
        for chunk, _executed_prefix, _ts in reader.read_actions():
            if chunk.num_steps < chunk_steps:
                continue
            gripper.append(np.asarray(chunk.gripper_target[:chunk_steps], dtype=np.float64))
            dt_s.append(float(chunk.dt_s))
            episode_ids.append(episode_id)

    if not gripper:
        raise SystemExit(f"{dataset}: no chunk reached {chunk_steps} steps")
    return ChunkExtras(
        episode_ids=np.asarray(episode_ids),
        gripper_targets=np.asarray(gripper, dtype=np.float64),
        dt_s=np.asarray(dt_s, dtype=np.float64),
    )


def check_extras_align(table: Any, extras: ChunkExtras) -> None:
    """Refuse a second pass that did not produce ``collect_chunks``' rows, in its order."""
    if extras.episode_ids.shape[0] != table.num_rows:
        raise SystemExit(
            f"the extras pass built {extras.episode_ids.shape[0]} rows and collect_chunks built "
            f"{table.num_rows}. The two walks of the dataset disagree about which chunks exist, "
            "so no per-row column from one may be attached to the other."
        )
    if not np.array_equal(extras.episode_ids, table.episode_ids):
        bad = int(np.flatnonzero(extras.episode_ids != table.episode_ids)[0])
        raise SystemExit(
            f"the extras pass and collect_chunks disagree at row {bad}: "
            f"{extras.episode_ids[bad]!r} vs {table.episode_ids[bad]!r}"
        )


def align_to_predictions(
    table: Any,
    holdout_rows: np.ndarray,
    extras: ChunkExtras,
    predictions: list[Any],
) -> np.ndarray:
    """Row index into ``table`` for each prediction record, PROVEN by the targets.

    Every per-chunk comparison in this file — stacking, the branch-point ranking, the gripper
    subset — needs the model's prediction and the ridge's prediction to be about the SAME chunk. A
    misalignment would not crash and would not look wrong: it would quietly compare two predictors
    on two different moments and produce a number with a verdict attached to it.

    So the pairing is established by position (both walks iterate ``read_actions()`` in order,
    within an episode) and then VERIFIED against the data: the demonstrated target chunk, the
    gripper channel and ``dt_s`` recorded in the predictions file must all match the row they were
    paired with. Those came out of ``build_eval_pairs`` on a different day; if the two agree
    element-wise on 1 040 chunks the alignment is a fact rather than an assumption.
    """
    rows_by_episode: dict[str, list[int]] = {}
    for row in holdout_rows.tolist():
        rows_by_episode.setdefault(str(table.episode_ids[row]), []).append(int(row))

    taken: dict[str, int] = {}
    rows: list[int] = []
    for i, pred in enumerate(predictions):
        episode_id = str(pred.episode_id)
        available = rows_by_episode.get(episode_id)
        if available is None:
            raise SystemExit(
                f"prediction {i} is from episode {episode_id!r}, which the holdout split does not "
                "contain. The predictions file and the split have come apart."
            )
        index = taken.get(episode_id, 0)
        if index >= len(available):
            raise SystemExit(
                f"episode {episode_id!r} has {len(available)} chunk(s) in the dataset but the "
                f"predictions file scores at least {index + 1}. Check --chunk-steps."
            )
        taken[episode_id] = index + 1
        rows.append(available[index])

    short = {ep: len(v) - taken.get(ep, 0) for ep, v in rows_by_episode.items()}
    missing = {ep: n for ep, n in short.items() if n}
    if missing:
        raise SystemExit(
            f"{sum(missing.values())} holdout chunk(s) in {len(missing)} episode(s) have no "
            f"prediction, e.g. {sorted(missing)[:3]}. The ridge would then be scored on chunks the "
            "model was not, and the two numbers would not be comparable."
        )

    row_index = np.asarray(rows, dtype=np.int64)
    chunk_steps = table.chunk_steps
    num_joints = table.num_joints
    for i, (pred, row) in enumerate(zip(predictions, row_index.tolist())):
        recorded = table.targets[row].reshape(chunk_steps, num_joints)
        scored = np.asarray(pred.target.targets, dtype=np.float64)
        if scored.shape != recorded.shape:
            raise SystemExit(
                f"prediction {i}: target shape {scored.shape} != the dataset's {recorded.shape}"
            )
        if np.asarray(pred.predicted.targets).shape != recorded.shape:
            raise SystemExit(
                f"prediction {i}: the model predicted "
                f"{np.asarray(pred.predicted.targets).shape}, not {recorded.shape}. This run was "
                "scored on a different chunk horizon than --chunk-steps asks for."
            )
        if not np.allclose(scored, recorded, rtol=0.0, atol=ALIGNMENT_ATOL):
            raise SystemExit(
                f"prediction {i} ({pred.episode_id}) does not match the chunk it was aligned to: "
                f"max |diff| {float(np.abs(scored - recorded).max()):.3e} > {ALIGNMENT_ATOL:g}. "
                "Every per-chunk comparison below would be between two different moments."
            )
        grip = np.asarray(pred.target.gripper_target, dtype=np.float64)
        if not np.allclose(grip, extras.gripper_targets[row], rtol=0.0, atol=ALIGNMENT_ATOL):
            raise SystemExit(
                f"prediction {i} ({pred.episode_id}): the scored gripper channel does not match "
                "the recorded one. The extras pass is aligned to a different chunk."
            )
        if not np.isclose(float(pred.target.dt_s), float(extras.dt_s[row]), rtol=1e-9, atol=0.0):
            raise SystemExit(
                f"prediction {i} ({pred.episode_id}): dt_s {pred.target.dt_s!r} != the recorded "
                f"{extras.dt_s[row]!r}. Constant velocity would be extrapolated at the wrong rate."
            )
    return row_index


# -- the predictors ------------------------------------------------------------------------------


def constant_velocity(dq: np.ndarray, dt_s: np.ndarray, chunk_steps: int) -> np.ndarray:
    """``pred[n, k, j] = dq[n, j] * dt_s[n]`` for every step k — the no-parameter reference.

    The same delta at all ``chunk_steps`` steps, because ``targets`` are PER-STEP joint deltas that
    ``G1Adapter.execute`` integrates onto the current ``q``: an arm continuing at its present speed
    covers ``dq * dt_s`` radians in every control period, not a cumulative ``dq * k * dt_s``. The
    cumulative form would describe an arm accelerating away, would be wrong by a factor of ~8 in
    the middle of the chunk, and would make this predictor look far worse than momentum is — which
    is precisely the direction that would fake Reading A.

    Nothing is fitted. There is no coefficient in front of ``dq``, no train set and no lambda, so
    this number cannot have been tuned into agreement with anything.
    """
    return np.repeat((dq * dt_s[:, None])[:, None, :], chunk_steps, axis=1)


def ridge_weights(
    standardizer: Any, train_states: np.ndarray, train_targets: np.ndarray, lam: float
) -> np.ndarray:
    """``(X'X + lam*I)^-1 X'Y`` — ``bench_ridge_baseline.fit_ridge``'s solve, weights returned.

    ``fit_ridge`` reports only an MSE, and PR-01 needs the ridge's PER-ROW predictions (to stack
    them, to rank subsets, to break them down by horizon). Rather than trust that this repeats the
    archived solve, the caller checks it: the MSE recomputed from these weights is compared against
    ``fit_ridge``'s own return value inside the archive gate, and a disagreement is fatal.
    """
    if lam <= 0.0:
        raise SystemExit(f"lambda must be > 0, got {lam:g}")
    x = standardizer.design(train_states)
    gram = x.T @ x
    rhs = x.T @ train_targets
    return np.linalg.solve(gram + lam * np.eye(gram.shape[0]), rhs)


# -- T1: cross-fitted stacking -------------------------------------------------------------------


@dataclass(frozen=True)
class StackFold:
    """One held-out fold's stacking weights and what they were fitted on."""

    fold: int
    alpha: float
    beta: float
    num_fit_episodes: int
    num_fit_chunks: int
    num_scored_chunks: int


@dataclass(frozen=True)
class CrossFittedStack:
    """T1's leak-free result as ONE object, so the leaked one cannot stand in for it.

    ``main`` produces two stacked numbers: this one, whose weights never saw the rows they score,
    and an in-sample one printed beside it as a diagnostic of what the cross-fitting cost. On the
    T-16 holdout they differ by 0.2 % — far too little for a reader to spot the wrong one in a
    printout of six-digit floats, and easily enough to move VERDICT A's first clause across
    ``0.95 * MSE_ridge``, which is the clause PR-01 calls primary.

    Two floats are interchangeable at a call site and a swap between them is a one-token edit that
    reads as a rename. So the clause does not take a float: :func:`decide` takes this type, which
    only :func:`cross_fitted_stack` returns, and the in-sample diagnostic stays a bare float that
    raises on attribute access if it is ever passed where this belongs.
    """

    predictions: np.ndarray
    mse: float
    folds: tuple[StackFold, ...]

    @property
    def betas(self) -> list[float]:
        """Per-fold incremental weight on the model, in fold order."""
        return [f.beta for f in self.folds]

    def folds_with_beta_above(self, threshold: float) -> int:
        """How many folds put more than ``threshold`` weight on the model. VERDICT A clause 2."""
        return sum(1 for b in self.betas if b > threshold)


def episode_folds(episode_ids: np.ndarray, num_folds: int) -> dict[str, int]:
    """``{episode_id: fold}`` — folds over EPISODES, deterministically, without a seed.

    Over episodes and not over rows, for the reason ``split_by_episode`` gives: chunks inside one
    episode are consecutive overlapping views of the same motion, so a row-level fold would fit the
    stacking weights on near-copies of the rows it then scores, and ``beta`` — the entire question
    reduced to one number — would come out of memorised neighbours.

    The assignment is ``sorted(ids)[i] -> i % num_folds``. Deterministic and seed-free, so a rerun
    is the same run; round-robin over the sorted ids rather than contiguous blocks, so a fold is
    not a contiguous run of recording sessions that could share a session-level artefact.
    """
    unique = sorted({str(e) for e in episode_ids.tolist()})
    if len(unique) < num_folds:
        raise SystemExit(
            f"{len(unique)} holdout episode(s) cannot be split into {num_folds} folds over "
            "episodes. Splitting over rows instead would fit the stacking weights on neighbours "
            "of the rows they are scored on."
        )
    return {episode_id: i % num_folds for i, episode_id in enumerate(unique)}


def check_folds_are_episode_disjoint(
    episode_ids: np.ndarray, fold_of_row: np.ndarray, num_folds: int
) -> None:
    """Refuse a fold map that puts one episode's rows on both sides of any fold.

    Asserted over the row tags rather than trusted to :func:`episode_folds`, in the idiom
    ``bench_ridge_baseline.split_by_episode`` uses for the train/holdout split and for the same
    reason: ``beta`` means nothing without "these weights never saw these rows", and a row-wise map
    — ``arange % num_folds``, exactly what :func:`episode_folds` exists to prevent — is invisible
    from the outside. The header still reads "folds over EPISODES", every fold still has weights,
    and the only thing that changes is that each fold is now fitted on near-copies of the chunks it
    scores.
    """
    for k in range(num_folds):
        scored = fold_of_row == k
        fitted = ~scored
        if not scored.any() or not fitted.any():
            raise SystemExit(f"fold {k} is empty on one side; cannot cross-fit")
        both = set(episode_ids[fitted].tolist()) & set(episode_ids[scored].tolist())
        if both:
            raise SystemExit(
                f"fold {k} would fit its stacking weights on {len(both)} episode(s) it then "
                f"scores, e.g. {sorted(str(e) for e in both)[:3]}. The folds are not over "
                "episodes, so beta would come out of memorised neighbours of the rows it is read "
                "on and T1 would not be the leak-free test PR-01 calls primary."
            )


def stack_weights(
    ridge_pred: np.ndarray, model_pred: np.ndarray, target: np.ndarray
) -> tuple[float, float]:
    """Least squares ``(alpha, beta)`` in ``y ~ alpha*ridge + beta*model``, over every element.

    No intercept: the pre-registration writes the model with two terms and this implements that
    one. The targets are zero-centred deltas anyway, so an intercept would be a third parameter
    fitted to nothing.
    """
    a = np.stack([ridge_pred.reshape(-1), model_pred.reshape(-1)], axis=1)
    y = target.reshape(-1)
    try:
        w = np.linalg.solve(a.T @ a, a.T @ y)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - needs two identical predictors
        raise SystemExit(
            f"the 2x2 stacking system is singular: {exc}. That happens when the two predictors are "
            "exactly collinear, i.e. the model IS the ridge up to a scale factor — which is a "
            "finding, not a numerical accident, and is reported rather than solved around."
        ) from exc
    return float(w[0]), float(w[1])


def cross_fitted_stack(
    ridge_pred: np.ndarray,
    model_pred: np.ndarray,
    target: np.ndarray,
    episode_ids: np.ndarray,
    num_folds: int,
) -> CrossFittedStack:
    """Stack the two predictors out of fold, having PROVEN no fold's weights saw their own rows.

    See :func:`check_folds_are_episode_disjoint`: the fold map is not trusted because it came out
    of :func:`episode_folds`, it is checked against the row tags before a single weight is fitted.
    """
    folds = episode_folds(episode_ids, num_folds)
    fold_of_row = np.asarray([folds[str(e)] for e in episode_ids.tolist()], dtype=np.int64)
    check_folds_are_episode_disjoint(episode_ids, fold_of_row, num_folds)

    stacked = np.empty_like(target)
    records: list[StackFold] = []
    for k in range(num_folds):
        scored = fold_of_row == k
        fitted = ~scored
        alpha, beta = stack_weights(ridge_pred[fitted], model_pred[fitted], target[fitted])
        stacked[scored] = alpha * ridge_pred[scored] + beta * model_pred[scored]
        records.append(
            StackFold(
                fold=k,
                alpha=alpha,
                beta=beta,
                num_fit_episodes=len(set(episode_ids[fitted].tolist())),
                num_fit_chunks=int(fitted.sum()),
                num_scored_chunks=int(scored.sum()),
            )
        )
    return CrossFittedStack(predictions=stacked, mse=mse(stacked, target), folds=tuple(records))


# -- T3 / T4: the two target-selected subsets ----------------------------------------------------


def branch_point_mask(cv_error: np.ndarray, quantile: float) -> np.ndarray:
    """The worst ``1 - quantile`` of chunks by CONSTANT-VELOCITY error.

    Ranked on constant velocity on purpose: it is neither of the two predictors being compared, so
    the subset cannot be the model's own good chunks or the ridge's own bad ones. The selection is
    still ON THE TARGET and is therefore admissible in one direction only — it strips out the
    chunks where the arm is simply continuing, which are exactly the chunks a linear map on ``dq``
    is best at, so it is biased in the MODEL's favour and a model loss here is the decisive result.

    Selection is by RANK, not by ``np.quantile`` on the values: a quantile threshold on a
    distribution with ties can return a subset of any size, and "the worst quartile" has to be a
    quartile for the count in the printout to mean what it says.
    """
    n = cv_error.shape[0]
    keep = max(1, round(n * (1.0 - quantile)))
    order = np.argsort(-cv_error, kind="stable")
    mask = np.zeros(n, dtype=bool)
    mask[order[:keep]] = True
    return mask


def branch_point_ranking(predicted: dict[str, np.ndarray], target: np.ndarray) -> np.ndarray:
    """[N] per-chunk error of T3's ranking predictor, refusing to rank on a compared one.

    The one argument :func:`branch_point_mask` takes is the only thing standing between T3 and a
    selection effect, and it is invisible in the output: every ranking produces a quartile, a chunk
    count and a threshold, and only the numbers underneath move. So the choice is made from
    ``T3_RANKING_PREDICTOR`` rather than written inline, and pointing it at either side of the
    comparison is refused here rather than discovered in review.
    """
    if T3_RANKING_PREDICTOR in (RIDGE_KEY, MODEL_KEY):
        raise SystemExit(
            f"T3 would rank its subset on {T3_RANKING_PREDICTOR!r}, which is one of the two "
            "predictors it then compares. The worst quartile by a predictor's own error is that "
            "predictor's worst chunks, so the comparison would be a selection effect and T3's "
            "one-directional admissibility argument would be empty."
        )
    if T3_RANKING_PREDICTOR not in predicted:
        raise SystemExit(
            f"T3's ranking predictor {T3_RANKING_PREDICTOR!r} was not computed; have "
            f"{sorted(predicted)}"
        )
    return per_chunk_mse(predicted[T3_RANKING_PREDICTOR], target)


def gripper_transition_mask(gripper_targets: np.ndarray, *, debounce: bool) -> np.ndarray:
    """Chunks whose demonstrated gripper channel crosses ``GRIPPER_BINARIZE_THRESHOLD``.

    Two readings of the pre-registration's "crossing the 0.5 threshold used by
    ``wam.evaluation.gripper``", and both are reported because the choice between them is not
    obviously the author's:

    - ``debounce=False`` — raw threshold crossings, ``gripper.crossings``. The literal reading, and
      the PRIMARY: it is the subset the decision rule's T4 clause is evaluated on.
    - ``debounce=True`` — ``gripper.debounced_transitions``, the hysteresis-corrected count that
      the same module exists to provide. A channel that merely sits ON the threshold and dithers
      produces a stream of raw crossings and reads as a busy gripper; that artefact is the one that
      made ``gripper_accuracy`` look like a grasp metric on a dead channel.

    Reporting both is not hedging. If they disagree, the disagreement IS the finding about the
    channel, and a reader who only saw the primary would draw a conclusion from noise.
    """
    if debounce:
        counts = [
            debounced_transitions(
                g, threshold=GRIPPER_BINARIZE_THRESHOLD, margin=GRIPPER_HYSTERESIS_MARGIN
            )
            for g in gripper_targets
        ]
    else:
        counts = [crossings(g, threshold=GRIPPER_BINARIZE_THRESHOLD) for g in gripper_targets]
    return np.asarray(counts, dtype=np.int64) > 0


# -- the verdict ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """The pre-registered rule, evaluated. ``letter`` is A, B or C and nothing else.

    ``clauses`` maps each clause of VERDICT A to True, False, or None for "not evaluable on this
    data". None is not a pass: VERDICT A requires ALL FOUR clauses to hold, so a clause that cannot
    be evaluated makes A unreachable and the run lands in C — which is the honest outcome and is
    what "the exact pattern of which tests split is reported" was written for. VERDICT B never
    mentions T4 and stays reachable.
    """

    letter: str
    clauses: dict[str, bool | None]
    lines: tuple[str, ...]


def decide(
    mse_ridge: float,
    stack: CrossFittedStack,
    t3_model: float,
    t3_ridge: float,
    t4_gripper_model: float | None,
    t4_gripper_ridge: float | None,
) -> Verdict:
    """Evaluate PR-01's decision rule mechanically. No number here is judged by eye.

    ``stack`` is the CROSS-FITTED result and is taken as an object rather than as its MSE, so the
    in-sample stack — a float sitting three lines away in ``main`` that is always slightly better —
    cannot be handed to clause 1 by a one-token edit.
    """
    betas = stack.betas
    mse_stack = stack.mse
    folds_positive = stack.folds_with_beta_above(MIN_INCREMENTAL_BETA)

    stack_buys_little = mse_stack >= STACK_IMPROVEMENT_FRACTION * mse_ridge
    beta_inconsistent = folds_positive < MIN_FOLDS_WITH_POSITIVE_BETA
    model_loses_t3 = t3_model >= t3_ridge
    if t4_gripper_model is None or t4_gripper_ridge is None:
        model_loses_t4: bool | None = None
    else:
        model_loses_t4 = t4_gripper_model >= t4_gripper_ridge

    clauses: dict[str, bool | None] = {
        "T1 stack buys < 5 %": stack_buys_little,
        "T1 beta not consistently positive": beta_inconsistent,
        "T3 model does not beat the ridge": model_loses_t3,
        "T4 model does not beat the ridge on the gripper channel": model_loses_t4,
    }
    if model_loses_t4 is None:
        t4_line = "  T4  gripper subset is empty — clause not evaluable  ->  INDETERMINATE"
    else:
        t4_line = (
            f"  T4  model {_fmt(t4_gripper_model)} vs ridge {_fmt(t4_gripper_ridge)}"
            f"  ->  {_bool(model_loses_t4)}"
        )
    lines = (
        (
            f"  T1  MSE_stack {_fmt(mse_stack)} vs {STACK_IMPROVEMENT_FRACTION:g} * MSE_ridge "
            f"{_fmt(STACK_IMPROVEMENT_FRACTION * mse_ridge)}  ->  {_bool(stack_buys_little)}"
        ),
        (
            f"  T1  folds with beta > {MIN_INCREMENTAL_BETA:g}: {folds_positive}/{len(betas)} "
            f"(need < {MIN_FOLDS_WITH_POSITIVE_BETA} for A)  ->  {_bool(beta_inconsistent)}"
        ),
        f"  T3  model {_fmt(t3_model)} vs ridge {_fmt(t3_ridge)}  ->  {_bool(model_loses_t3)}",
        t4_line,
    )

    if all(v is True for v in clauses.values()):
        letter = "A"
    elif (not model_loses_t3) and (not stack_buys_little) and (not beta_inconsistent):
        letter = "B"
    else:
        letter = "C"
    return Verdict(letter=letter, clauses=clauses, lines=lines)


def _bool(value: bool | None) -> str:
    return "n/a" if value is None else ("holds" if value else "fails")


VERDICT_CONSEQUENCE = {
    "A": (
        "the model carries no incremental information. Pre-registered consequence: I-8 (~125 "
        "GPU-h) is NOT submitted, no further LoRA scaling is proposed on this architecture, "
        "offline chunk MSE is retired as a ranking metric, and D2/D3 stay unimplemented."
    ),
    "B": (
        "the metric was the problem. Pre-registered consequence: the branch-point subset becomes "
        "the headline metric and a gated WAM-Bench rung, full-holdout MSE is demoted, prior ladder "
        "results are re-scored before being quoted, and I-8 is reconsidered on the NEW axis."
    ),
    "C": (
        "mixed. No global claim is made. The pattern above is the result; the next step is chosen "
        "from that pattern rather than from a headline."
    ),
}


# -- the archive gate ----------------------------------------------------------------------------


def check_archived_controls(zero_delta: float, model: float, ridge_all: float) -> list[str]:
    """Raise unless all three archived numbers reproduce. Returns the lines to print.

    PR-01: "a run that does not reproduce them is void, not interpreted". So this runs before any
    PR-01 number is computed, and it raises rather than warning — a warning above a table of
    six-digit numbers is a warning nobody acts on.
    """
    archived = (
        ("zero-delta (hold still)", zero_delta, ARCHIVED_ZERO_DELTA_MSE),
        ("model (Wan-5B+LoRA)", model, ARCHIVED_MODEL_MSE),
        (f"ridge (all state, lambda {RIDGE_LAMBDA:g})", ridge_all, ARCHIVED_RIDGE_ALL_MSE),
    )
    lines: list[str] = []
    failed: list[str] = []
    for name, measured, expected in archived:
        ok = bool(np.isclose(measured, expected, rtol=ARCHIVE_RTOL, atol=0.0))
        lines.append(
            f"  {name:<34}{_fmt(measured)}   expected {_fmt(expected)}   {'OK' if ok else 'DRIFT'}"
        )
        if not ok:
            failed.append(f"{name}: measured {_fmt(measured)}, archived {_fmt(expected)}")
    if failed:
        raise SystemExit(
            "the archived controls did not reproduce, so this run is VOID and none of its numbers "
            "are interpreted (PR-01):\n  "
            + "\n  ".join(failed)
            + "\nCheck --dataset, --chunk-steps and that the predictions file is the T-16 run's."
        )
    return lines


# -- CLI -----------------------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="episode directory root")
    parser.add_argument(
        "--holdout",
        type=Path,
        required=True,
        help="the run's predictions.jsonl. A plain episode id list is NOT enough here: PR-01 "
        "compares the model per chunk, so it needs the model's predictions, not just the split",
    )
    parser.add_argument(
        "--chunk-steps", type=int, default=ridge_baseline.DEFAULT_CHUNK_STEPS,
        help="chunk horizon the targets are truncated to (default: %(default)s)",
    )  # fmt: skip
    parser.add_argument(
        "--archive-gate",
        choices=("t16", "off"),
        default="t16",
        help="'t16' (default) refuses to report anything unless the archived controls reproduce. "
        "'off' is for tests and for other data, and says so in the output",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the full record here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.chunk_steps < 1:
        raise SystemExit(f"--chunk-steps must be >= 1, got {args.chunk_steps}")

    holdout_ids = load_episode_ids(args.holdout)
    if not holdout_ids:
        raise SystemExit(f"{args.holdout}: no episode ids found")
    try:
        predictions = load_predictions_jsonl(args.holdout)
    except ValueError as exc:
        raise SystemExit(
            f"{args.holdout}: not a readable predictions.jsonl ({exc}). PR-01 compares the model "
            "chunk by chunk, so an episode id list cannot stand in for the run's own predictions."
        ) from exc
    if not predictions:
        raise SystemExit(
            f"{args.holdout}: no prediction records. PR-01 compares the model chunk by chunk, so "
            "an episode id list cannot stand in for the run's own predictions."
        )

    table = ridge_baseline.collect_chunks(args.dataset, args.chunk_steps)
    extras = collect_chunk_extras(args.dataset, args.chunk_steps)
    check_extras_align(table, extras)
    train_mask, holdout_mask = ridge_baseline.split_by_episode(table, holdout_ids)
    rows = align_to_predictions(table, np.flatnonzero(holdout_mask), extras, predictions)

    steps, joints = table.chunk_steps, table.num_joints
    train_states = table.states[train_mask]
    train_targets = table.targets[train_mask]
    train_gripper = extras.gripper_targets[train_mask]
    states = table.states[rows]
    target = table.targets[rows].reshape(-1, steps, joints)
    gripper_target = extras.gripper_targets[rows]
    dt_s = extras.dt_s[rows]
    episode_ids = table.episode_ids[rows]
    num_chunks = target.shape[0]

    print("PR-01 — incremental value of the model over the blind proprioceptive ridge")
    print(f"dataset  {args.dataset}")
    print(f"holdout  {args.holdout}")
    print(
        f"chunks   holdout {num_chunks} ({len(holdout_ids)} eps) / train {train_states.shape[0]} "
        f"({len(set(table.episode_ids[train_mask].tolist()))} eps)  |  target {steps} x {joints}"
        f"  |  ridge lambda {RIDGE_LAMBDA:g}, fitted on train episodes only"
    )

    groups = ridge_baseline.feature_groups(joints, table.gripper_dims, table.state_dim)
    predicted: dict[str, np.ndarray] = {
        "zero-delta": np.zeros_like(target),
        # Through the SAME column mask the "ridge (dq only)" ablation is built from, so "which
        # columns are dq" has one definition. Handed `q` instead, this still returns a plausible
        # six-digit number and quietly changes both the Reading-B reference row and T3's subset.
        "const-velocity": constant_velocity(states[:, groups["dq"]], dt_s, steps),
        "model": np.stack([np.asarray(p.predicted.targets, dtype=np.float64) for p in predictions]),
    }
    standardizers = {
        RIDGE_KEY: ridge_baseline.Standardizer.fit(train_states, columns=groups["all"]),
        "ridge (dq only)": ridge_baseline.Standardizer.fit(train_states, columns=groups["dq"]),
    }
    for name, standardizer in standardizers.items():
        weights = ridge_weights(standardizer, train_states, train_targets, RIDGE_LAMBDA)
        predicted[name] = (standardizer.design(states) @ weights).reshape(-1, steps, joints)

    # -- the gate. Nothing below is computed until these three reproduce. -----------------------
    measured = {name: mse(p, target) for name, p in predicted.items()}
    cross_check = ridge_baseline.fit_ridge(
        standardizers[RIDGE_KEY],
        train_states,
        train_targets,
        states,
        table.targets[rows],
        RIDGE_LAMBDA,
    ).holdout_mse
    if not np.isclose(measured[RIDGE_KEY], cross_check, rtol=1e-12, atol=0.0):
        raise SystemExit(
            f"the ridge predictions here score {_fmt(measured[RIDGE_KEY])} but "
            f"bench_ridge_baseline.fit_ridge scores {_fmt(cross_check)} on the same rows. "
            "ridge_weights is not repeating the archived solve."
        )
    archived_model = ridge_baseline.model_mse_from_predictions(args.holdout)
    if archived_model is None or not np.isclose(
        measured[MODEL_KEY], archived_model[0], rtol=1e-12, atol=0.0
    ):
        raise SystemExit(
            "the model MSE assembled here does not match bench_ridge_baseline's own reading of "
            f"the same file ({_fmt(measured[MODEL_KEY])} vs "
            f"{'n/a' if archived_model is None else _fmt(archived_model[0])})."
        )

    print()
    if args.archive_gate == "off":
        print("ARCHIVE GATE OFF — this run reproduces nothing and its numbers are not PR-01's.")
        gate_lines: list[str] = []
    else:
        print("archived controls (gate — every number below is void unless these reproduce)")
        gate_lines = check_archived_controls(
            measured["zero-delta"], measured[MODEL_KEY], measured[RIDGE_KEY]
        )
        print("\n".join(gate_lines))

    print()
    print("full holdout MSE (mean over the flattened target chunk)")
    for name in PREDICTOR_ORDER:
        print(f"  {name:<20}{_fmt(measured[name])}")

    # -- T1 ---------------------------------------------------------------------------------------
    stack = cross_fitted_stack(
        predicted[RIDGE_KEY], predicted[MODEL_KEY], target, episode_ids, STACKING_FOLDS
    )
    alpha_all, beta_all = stack_weights(predicted[RIDGE_KEY], predicted[MODEL_KEY], target)
    mse_stack_leaked = mse(
        alpha_all * predicted[RIDGE_KEY] + beta_all * predicted[MODEL_KEY], target
    )
    folds_positive = stack.folds_with_beta_above(MIN_INCREMENTAL_BETA)

    print()
    print(f"T1 — cross-fitted stacking, {STACKING_FOLDS} folds over EPISODES (leak-free, primary)")
    print(f"  {'fold':>4}{'fit eps':>9}{'fit chunks':>12}{'scored':>8}{'alpha':>12}{'beta':>12}")
    for f in stack.folds:
        print(
            f"  {f.fold:>4}{f.num_fit_episodes:>9}{f.num_fit_chunks:>12}"
            f"{f.num_scored_chunks:>8}{f.alpha:>12.4f}{f.beta:>12.4f}"
        )
    print(f"  cross-fitted stack   {_fmt(stack.mse)}   "
          f"{stack.mse / measured[RIDGE_KEY]:.4f} x the ridge")  # fmt: skip
    print(
        f"  in-sample stack      {_fmt(mse_stack_leaked)}   LEAKED (alpha {alpha_all:.4f}, beta "
        f"{beta_all:.4f} fitted on the rows they score) — diagnostic only, never a result"
    )
    print(f"  folds with beta > {MIN_INCREMENTAL_BETA:g}: {folds_positive}/{STACKING_FOLDS}")

    # -- T2 ---------------------------------------------------------------------------------------
    per_step = {name: per_step_mse(p, target) for name, p in predicted.items()}
    print()
    print("T2 — per-horizon breakdown, MSE at each chunk step (leak-free, no selection)")
    print("  step  " + "".join(f"{name:>18}" for name in PREDICTOR_ORDER))
    for k in range(steps):
        print(f"  {k:>4}  " + "".join(f"{_fmt(per_step[name][k]):>18}" for name in PREDICTOR_ORDER))

    # -- T3 ---------------------------------------------------------------------------------------
    cv_error = branch_point_ranking(predicted, target)
    branch = branch_point_mask(cv_error, BRANCH_POINT_QUANTILE)
    t3 = {name: mse(p[branch], target[branch]) for name, p in predicted.items()}
    print()
    print(
        f"T3 — branch points: worst quartile by {T3_RANKING_PREDICTOR!r} error "
        f"({int(branch.sum())} of {num_chunks} chunks, threshold "
        f"{_fmt(float(cv_error[branch].min()))})"
    )
    print("     target-selected: a model LOSS is decisive, a model WIN is not (needs T1 to agree)")
    for name in PREDICTOR_ORDER:
        print(f"  {name:<20}{_fmt(t3[name])}")

    # -- T4 ---------------------------------------------------------------------------------------
    transition = gripper_transition_mask(gripper_target, debounce=T4_PRIMARY_DEBOUNCE)
    transition_debounced = gripper_transition_mask(gripper_target, debounce=True)
    grip_p2p = float(gripper_target.max() - gripper_target.min())
    grip_model = np.stack(
        [np.asarray(p.predicted.gripper_target, dtype=np.float64) for p in predictions]
    )
    if np.array_equal(grip_model, gripper_target):
        raise SystemExit(
            f"the model's gripper predictions are bit-identical to the demonstrated channel on all "
            f"{num_chunks} chunks. No model reproduces a float32 target exactly; this is what "
            "reading `target` where `predicted` was meant looks like, and it would print "
            "0.000000e+00 as a model result under a passing archive gate."
        )
    grip_standardizer = standardizers[RIDGE_KEY]
    grip_weights = ridge_weights(grip_standardizer, train_states, train_gripper, RIDGE_LAMBDA)
    grip_ridge = grip_standardizer.design(states) @ grip_weights
    # The same cross-check the joint ridge gets, for the same reason: `fit_ridge` is named the
    # train rows here, so a gripper ridge that had been fitted on the holdout it scores — a
    # two-argument edit, and a change in the fourth decimal — cannot reach the clause.
    grip_cross_check = ridge_baseline.fit_ridge(
        grip_standardizer, train_states, train_gripper, states, gripper_target, RIDGE_LAMBDA
    ).holdout_mse
    if not np.isclose(mse(grip_ridge, gripper_target), grip_cross_check, rtol=1e-12, atol=0.0):
        raise SystemExit(
            f"the gripper ridge here scores {_fmt(mse(grip_ridge, gripper_target))} but "
            f"bench_ridge_baseline.fit_ridge scores {_fmt(grip_cross_check)} on the same rows from "
            "the same train episodes. The weights T4's clause is decided on are not that solve."
        )
    grip_constant = np.repeat(train_gripper.mean(axis=0)[None, :], num_chunks, axis=0)

    print()
    print(
        f"T4 — gripper-transition chunks: {int(transition.sum())} of {num_chunks} cross "
        f"{GRIPPER_BINARIZE_THRESHOLD:g} raw, {int(transition_debounced.sum())} debounced "
        f"(margin {GRIPPER_HYSTERESIS_MARGIN:g}); the clause is evaluated on the "
        f"{'DEBOUNCED' if T4_PRIMARY_DEBOUNCE else 'RAW'} subset"
    )
    print("     target-selected: same one-directional admissibility as T3")
    t4: dict[str, float] = {}
    t4_gripper: dict[str, float] = {}
    if not transition.any():
        print(
            "  no chunk crosses the threshold — T4 measures nothing on this holdout and its "
            "clause is INDETERMINATE, which makes VERDICT A unreachable rather than free."
        )
    else:
        t4 = {name: mse(p[transition], target[transition]) for name, p in predicted.items()}
        for name in PREDICTOR_ORDER:
            print(f"  {name:<20}{_fmt(t4[name])}")
        t4_gripper = {
            "model": mse(grip_model[transition], gripper_target[transition]),
            "ridge (all state)": mse(grip_ridge[transition], gripper_target[transition]),
            "constant (train mean)": mse(grip_constant[transition], gripper_target[transition]),
        }
        print("  gripper channel alone (the joint channels are 15/16ths of the numbers above)")
        for name, value in t4_gripper.items():
            print(f"    {name:<22}{_fmt(value)}")
    print(
        f"  channel admissibility: holdout gripper p2p {grip_p2p:.4f} vs the pre-registered "
        f"GRIPPER_MIN_DYNAMIC_RANGE {GRIPPER_MIN_DYNAMIC_RANGE:g} — "
        + (
            "PASS"
            if grip_p2p >= GRIPPER_MIN_DYNAMIC_RANGE
            else "FAIL. There is no open/close event in this holdout, so the gripper numbers "
            "above are two near-constants disagreeing and must not be read as a grasp result."
        )
    )

    # -- the verdict -------------------------------------------------------------------------------
    verdict = decide(
        mse_ridge=measured[RIDGE_KEY],
        stack=stack,
        t3_model=t3[MODEL_KEY],
        t3_ridge=t3[RIDGE_KEY],
        t4_gripper_model=t4_gripper.get("model"),
        t4_gripper_ridge=t4_gripper.get("ridge (all state)"),
    )
    print()
    print("VERDICT A clauses (A requires ALL four; B never mentions T4)")
    print("\n".join(verdict.lines))
    print()
    print(f"VERDICT {verdict.letter} — {VERDICT_CONSEQUENCE[verdict.letter]}")
    if args.archive_gate == "off":
        print("(archive gate was OFF: this verdict is about whatever data was passed, not PR-01.)")

    if args.json is not None:
        record: dict[str, Any] = {
            "preregistration": "docs/preregistration/PR-01-incremental-value.md",
            "dataset": str(args.dataset),
            "holdout": str(args.holdout),
            "chunk_steps": args.chunk_steps,
            "ridge_lambda": RIDGE_LAMBDA,
            "archive_gate": args.archive_gate,
            "num_holdout_chunks": num_chunks,
            "num_holdout_episodes": len(holdout_ids),
            "num_train_chunks": int(train_states.shape[0]),
            "full_holdout_mse": measured,
            "t1": {
                "folds": [
                    {
                        "fold": f.fold,
                        "alpha": f.alpha,
                        "beta": f.beta,
                        "num_fit_episodes": f.num_fit_episodes,
                        "num_fit_chunks": f.num_fit_chunks,
                        "num_scored_chunks": f.num_scored_chunks,
                    }
                    for f in stack.folds
                ],
                "mse_stack": stack.mse,
                "mse_stack_in_sample_leaked": mse_stack_leaked,
                "folds_with_positive_beta": folds_positive,
            },
            "t2_per_step_mse": {name: [float(v) for v in per_step[name]] for name in per_step},
            # `ranked_on` and `selected_chunks` are the provenance of a TARGET-SELECTED subset:
            # which predictor chose it and which rows it kept. Without them a reader can see that
            # T3 took a quartile but not of what, and every ranking — including one on a compared
            # predictor — produces an identical-looking header with different numbers under it.
            "t3": {
                "quantile": BRANCH_POINT_QUANTILE,
                "ranked_on": T3_RANKING_PREDICTOR,
                "num_chunks": int(branch.sum()),
                "threshold": float(cv_error[branch].min()),
                "selected_chunks": [int(i) for i in np.flatnonzero(branch)],
                "mse": t3,
            },
            "t4": {
                "primary_debounce": T4_PRIMARY_DEBOUNCE,
                "num_chunks": int(transition.sum()),
                "num_chunks_debounced": int(transition_debounced.sum()),
                "selected_chunks": [int(i) for i in np.flatnonzero(transition)],
                "gripper_p2p": grip_p2p,
                "gripper_channel_admissible": grip_p2p >= GRIPPER_MIN_DYNAMIC_RANGE,
                "mse": t4,
                "gripper_channel_mse": t4_gripper,
            },
            "verdict": verdict.letter,
            "verdict_clauses": verdict.clauses,
            "thresholds": {
                "RIDGE_LAMBDA": RIDGE_LAMBDA,
                "STACKING_FOLDS": STACKING_FOLDS,
                "STACK_IMPROVEMENT_FRACTION": STACK_IMPROVEMENT_FRACTION,
                "MIN_INCREMENTAL_BETA": MIN_INCREMENTAL_BETA,
                "MIN_FOLDS_WITH_POSITIVE_BETA": MIN_FOLDS_WITH_POSITIVE_BETA,
                "BRANCH_POINT_QUANTILE": BRANCH_POINT_QUANTILE,
                "T3_RANKING_PREDICTOR": T3_RANKING_PREDICTOR,
                "T4_PRIMARY_DEBOUNCE": T4_PRIMARY_DEBOUNCE,
            },
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
