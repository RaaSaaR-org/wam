#!/usr/bin/env python3
"""The blind proprioception bars: what a model that never opens the camera already scores.

Two predictors from the 32-dim robot state at chunk time (q15 + dq15 + gripper2) to the flattened
[16, 15] action chunk. No frames are read. No backbone is built. Nothing is trained on a GPU. On
the T-16 holdout both of them beat the deployed model:

    blind nonlinear ceiling 5.431371e-06   983 280 parameters, a random-Fourier ridge
    ridge, all state        6.330899e-06     7 920 parameters, one np.linalg.solve
    ridge, dq only          6.869239e-06     3 840 parameters
    model (Wan-5B+LoRA)     1.112983e-05    82.5M trainable parameters
    ridge, q only           1.348259e-05
    ridge, gripper only     1.550558e-05
    zero-delta (hold still) 1.632760e-05

Measured on ``datasets/gr00t-apple-full`` against
``runs/t16-lora-seed0/eval-t30-regression/predictions.jsonl`` — the linear rows on 2026-08-01,
independently reproduced by two separate implementations before being written down, the ceiling on
2026-08-02. The linear map is **1.76x better** than the fine-tune; velocity alone is still
**1.62x better**. Position alone and gripper alone both lose to it, so the win is not
"proprioception is trivially sufficient" — it is specifically ``dq``, the one channel that says
where the arm is already going.

TWO BARS, AND THEY DO NOT SAY THE SAME THING
--------------------------------------------
THE LINEAR BAR (``ridge, all state``) is the best a *linear* map can do knowing everything about
the body and nothing about the world. A visual policy that scores above it has not demonstrated
that it uses vision — it has demonstrated that it is a worse proprioceptive regressor than a
matrix solve. Whatever its backbone cost in parameters, GPU-hours or LoRA rank, it has not
earned it.

THE NONLINEAR CEILING (``blind nonlinear ceiling``) exists because the linear bar UNDERSTATES what
is knowable without a camera: nothing says the blind-optimal predictor is linear, and on this
holdout it is not — 4 096 random Fourier features over the same 32 dims reach 5.431371e-06 against
the linear 6.330877e-06, a further 1.17x with the camera still shut. A score above the ceiling has
not demonstrated *anything a blind regressor could not do*. That is the stronger and more
uncomfortable statement, and it is the one an expensive model has to answer: clearing the linear
bar only means "better than a matrix solve", while clearing the ceiling is the first evidence that
the world model contributed something the body did not already imply.

Both are answerable without looking at the robot's camera, which is what makes the comparison mean
something and also what makes it run on a laptop instead of a GPU. The linear rows take seconds;
the ceiling's hyperparameter search takes ~2 minutes of BLAS and can be skipped with
``--no-ceiling`` when only the linear controls are wanted.

THE CEILING'S HYPERPARAMETERS NEVER TOUCH THE HOLDOUT
-----------------------------------------------------
A ceiling tuned on the data it is a ceiling *for* is not a ceiling, it is a fit — and it would be
the single easiest way to manufacture a bar no model can clear. So the width, the RBF bandwidth
and the ridge penalty are chosen on an INNER validation split of the TRAIN episodes only,
episode-disjoint from both the train remainder and the holdout, and the chosen config is then
refitted on all train episodes and scored once. :func:`inner_validation_episodes` RAISES if a
holdout episode reaches the search at all — a runtime check over the row tags, not a comment —
and ``tests/test_bench_ridge_baseline.py`` pins it with a leak control that shows choosing on the
holdout instead scores detectably better.

WHAT IS NOT CLAIMED
-------------------
That either bar is a policy. Both are fitted on the demonstrations of a single task, neither has
any notion of where the apple is, and neither would survive the apple moving — which is exactly
the generalization the video branch exists to buy. E1 action-MSE is a DIAGNOSTIC metric
(PRD 10.4) and these baselines are a diagnostic on that diagnostic. Losing to them does not make a
model useless; it makes the *offline MSE evidence* for that model worthless, which is a narrower
and much more actionable statement.

That the ceiling is the true blind optimum. It is the best of a 48-point grid of one particular
nonlinear family, so it is a LOWER BOUND on what proprioception affords, and the error is in the
conservative direction: the real blind optimum is at most 5.431371e-06, so a model scoring between
the true optimum and this number escapes the verdict rather than being falsely convicted. The
number is also mildly draw-dependent — the search stream picks width 4096 over 8192 on a val gap
in the third significant digit, and seeding each grid point independently instead flips that to
8192 and 5.388504e-06, a 0.8% lower ceiling. Anything that hinges on the sixth digit of this row
is not a finding.

``--lam`` for the LINEAR rows is swept and the best holdout MSE is reported, which is a mild
selection on the holdout and is named here rather than hidden. It buys almost nothing: across four
orders of magnitude of lambda the all-state number moves in its seventh significant digit —
6.330899e-06 at 1e-2, the grid minimum 6.330877e-06 at 1e1, 6.333218e-06 at 1e2. Quoting any of
them makes the same claim to the six digits anyone reads. The baseline is not tuned into its win,
and the full per-lambda table is printed so a reader can see that for themselves instead of taking
it on trust. The ceiling does not get that latitude, because it is the row with enough capacity to
abuse it.

    scripts/bench_ridge_baseline.py \\
        --dataset datasets/gr00t-apple-full \\
        --holdout runs/t16-lora-seed0/eval-t30-regression/predictions.jsonl

``--holdout`` takes a plain one-per-line episode list OR a ``predictions.jsonl`` directly, via the
repo's own :func:`~wam.evaluation.offline.load_episode_ids`. Pointing it at the predictions is the
better habit: the split then has exactly one definition, and when the file is a predictions.jsonl
the model's own MSE is recomputed from it and printed alongside — so the comparison is against a
number produced by that run, never against one copied out of a report.
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

from wam.evaluation.offline import load_episode_ids

DEFAULT_CHUNK_STEPS = 16
"""Matches the T-16 head and ``build_eval_pairs``' contract, so the rows line up with a bench run."""

DEFAULT_LAMBDAS = (1e-2, 1e-1, 1.0, 10.0, 100.0)
"""The swept ridge penalties. Four orders of magnitude, and the answer barely moves across them."""

FEATURE_GROUPS = ("all", "q", "dq", "gripper")
"""Reported ablations. Each is a claim about WHICH proprioceptive channel carries the prediction,
and the answer (``dq``, not ``q``) is the part of this baseline that is actually informative."""

CEILING_WIDTHS = (4096, 8192)
"""Random-Fourier feature counts searched for the nonlinear ceiling.

Both are far more features than the 32 dims they are drawn over and than the ~9.5k train chunks,
so the ridge penalty is what keeps the system honest, not the width. The grid is small on purpose:
every point costs a Gram matrix of its own width, and the val gap between these two is already in
the third significant digit."""

CEILING_GAMMAS = (0.005, 0.01, 0.02, 0.05)
"""RBF bandwidths. ``cos(z @ W + b)`` with ``W ~ N(0, 2*gamma)`` approximates an RBF kernel of
bandwidth ``gamma`` on the STANDARDIZED state, so these are "how far apart do two robot states
have to be before they stop predicting each other" measured in per-dimension standard deviations.
The search lands on the smallest — the map the data supports is smooth."""

CEILING_LAMBDAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
"""Ridge penalties for the ceiling. Higher than :data:`DEFAULT_LAMBDAS` because there are two to
three orders of magnitude more features to shrink."""

CEILING_VAL_EPISODES = 40
"""Inner validation episodes, drawn from TRAIN only. Matches the holdout's own episode count, so
the config is chosen on a split the same size as the one it will be scored on."""

CEILING_SPLIT_SEED = 1
"""Seed for the inner train/validation episode shuffle."""

CEILING_SEARCH_SEED = 0
"""Seed for the ONE feature stream the grid search draws from.

One stream advanced across the grid, not a fresh seed per point, because that is what was measured
and the archived 5.431371e-06 is the config that stream selected. It makes the search
order-dependent, which is a real fragility and is named in the module docstring rather than hidden:
re-seeding per grid point picks the other width and lands 0.8% lower."""

CEILING_REFIT_SEED = 7
"""Seed for the feature draw of the final refit on all train episodes.

Deliberately NOT the search stream's. The features are redrawn anyway — the search standardizes on
the inner-fit episodes and the refit on all of them, so the two feature maps differ regardless —
and a separate seed makes it impossible to read the ceiling as the search's best val score
recycled onto the holdout."""

ARCHIVED_T16 = {
    "zero_delta": "1.632760e-05",
    "model": "1.112983e-05",
    "ceiling": "5.431371e-06",
    "ridge_all_lam0.01": "6.330899e-06",
    "ridge_all_lam10": "6.330877e-06",
    "ridge_all_lam100": "6.333218e-06",
    "ridge_dq_lam1": "6.869239e-06",
    "ridge_q_lam1": "1.348259e-05",
    "ridge_gripper_lam0.01": "1.550558e-05",
}
"""Every number this module writes down, keyed so each is unambiguous about its lambda.

These are the control. Three separate measurements — PR-01's incremental-value verdict, the
momentum follow-up and the ceiling below — are stated relative to these rows, so a change anywhere
in ``collect_chunks``, ``Standardizer`` or ``fit_ridge`` that moves one of them invalidates work
that has already been reported. :func:`check_archived` therefore re-checks them on every run that
matches the archived shape and RAISES instead of printing a quietly different table.

Compared as the six-decimal strings this script prints: that is the precision the numbers were
published at, and matching it means "reproduces to the digit" rather than "close enough"."""

ARCHIVED_T16_SHAPE = {
    "num_train_chunks": 9476,
    "num_holdout_chunks": 1040,
    "num_train_episodes": 362,
    "num_holdout_episodes": 40,
    "state_dim": 32,
    "target_dim": 240,
}
"""The fingerprint that says this run IS the archived one. Checking the numbers on any other
dataset would be nonsense, so the control arms itself on an exact shape match and stays silent
otherwise."""


# -- collection ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkTable:
    """One row per recorded action chunk: the state at chunk time and the flattened target.

    ``states`` is [N, 2*num_joints + gripper_dims] and ``targets`` is [N, chunk_steps*num_joints],
    both float64 — these are differences of small numbers and the metric is a mean of squares, so
    the accumulation runs in the same precision ``wam.evaluation`` uses.

    ``episode_ids`` is what makes the split checkable after the fact rather than by construction:
    every row carries the episode it came from, so "no holdout episode is in the train matrix" is
    an assertion over data instead of a property of the loop that built it.
    """

    states: np.ndarray
    targets: np.ndarray
    episode_ids: np.ndarray
    num_joints: int
    gripper_dims: int
    chunk_steps: int

    @property
    def num_rows(self) -> int:
        return int(self.states.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def target_dim(self) -> int:
        return int(self.targets.shape[1])

    def mask_for(self, episode_ids: set[str]) -> np.ndarray:
        """Boolean row mask selecting the rows recorded in ``episode_ids``."""
        return np.isin(self.episode_ids, sorted(episode_ids))

    def select(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.states[mask], self.targets[mask]


def state_vector(state: Any) -> np.ndarray:
    """``concat(q, dq, gripper_state)`` as float64 — the whole input this baseline is allowed.

    IMU is left out on purpose. It is marked invalid on every episode of the converted GR00T
    datasets (``ValidityMask(imu=False)``), so including it would add columns that are either
    constant or meaningless and would make the parameter count a worse description of what the
    model actually reads.
    """
    return np.concatenate(
        [
            np.asarray(state.q, dtype=np.float64).ravel(),
            np.asarray(state.dq, dtype=np.float64).ravel(),
            np.atleast_1d(np.asarray(state.gripper_state, dtype=np.float64)).ravel(),
        ]
    )


def collect_chunks(dataset: Path, chunk_steps: int) -> ChunkTable:
    """Build the (state, target) table over every episode under ``dataset``.

    The pairing rule is ``build_eval_pairs``' (``src/wam/evaluation/offline.py``), deliberately
    duplicated rather than called: for each recorded chunk take the LAST state at or before the
    chunk timestamp — never the next one, which would hand the baseline an observation from after
    the decision it is being asked to make — skip chunks shorter than ``chunk_steps`` and truncate
    longer ones. Same rule as the evaluator means the ridge is scored on the same chunks the model
    was, which is the only reason the two numbers may be printed on adjacent lines.

    What is NOT shared is the frame read. ``build_eval_pairs`` decodes the episode's mp4 to build
    an ``Observation``; this baseline never looks at a pixel, so calling it would spend minutes of
    video decoding to throw the result away. That difference is the point of the script, not a
    shortcut around it.
    """
    from wam.data.episode import EpisodeReader, list_episodes

    episode_dirs = list_episodes(dataset)
    if not episode_dirs:
        raise SystemExit(f"{dataset}: no episodes found (nothing containing a manifest.json)")

    states: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    episode_ids: list[str] = []
    geometry: tuple[int, int] | None = None

    for episode_dir in episode_dirs:
        reader = EpisodeReader(episode_dir)
        spec = reader.manifest.spec
        episode_id = reader.manifest.episode_id
        if geometry is None:
            geometry = (spec.num_joints, spec.gripper_dims)
        elif geometry != (spec.num_joints, spec.gripper_dims):
            raise SystemExit(
                f"{episode_id}: canonical space {(spec.num_joints, spec.gripper_dims)} differs "
                f"from {geometry} seen earlier. One design matrix cannot hold two geometries, and "
                "padding them together would fit a map across robots that never existed."
            )

        recorded = reader.read_states()
        if not recorded:
            continue
        state_ts = np.asarray([s.timestamp_ns for s in recorded], dtype=np.int64)
        for chunk, _executed_prefix, ts in reader.read_actions():
            if chunk.num_steps < chunk_steps:
                continue
            target = np.asarray(chunk.targets[:chunk_steps], dtype=np.float64)
            index = max(int(np.searchsorted(state_ts, ts, side="right")) - 1, 0)
            states.append(state_vector(recorded[index]))
            targets.append(target.reshape(-1))
            episode_ids.append(episode_id)

    if not states:
        raise SystemExit(
            f"{dataset}: no chunk reached {chunk_steps} steps, so there is nothing to fit. "
            "Check --chunk-steps against the head this dataset was recorded for."
        )
    widths = {x.shape[0] for x in states}
    if len(widths) != 1:
        raise SystemExit(
            f"{dataset}: ragged state vectors across episodes: widths {sorted(widths)}"
        )

    assert geometry is not None
    return ChunkTable(
        states=np.asarray(states, dtype=np.float64),
        targets=np.asarray(targets, dtype=np.float64),
        episode_ids=np.asarray(episode_ids),
        num_joints=geometry[0],
        gripper_dims=geometry[1],
        chunk_steps=chunk_steps,
    )


def split_by_episode(table: ChunkTable, holdout_ids: set[str]) -> tuple[np.ndarray, np.ndarray]:
    """``(train mask, holdout mask)`` — split STRICTLY by episode, and prove it.

    Chunks inside one episode are consecutive, overlapping views of the same motion: neighbouring
    rows share states to within sensor noise. A row-level split would therefore put near-copies of
    every holdout row into the train matrix and the baseline would score its own training data —
    the single most likely way for a number like 6.33e-06 to be wrong. So the split is by episode,
    and the disjointness is ASSERTED here over the row tags rather than trusted to the loop above.

    Refuses a holdout id the dataset does not contain, which is the other quiet failure: a
    misspelled or moved episode silently shrinks the holdout and the ridge is scored on fewer,
    easier chunks than the model was.
    """
    present = set(table.episode_ids.tolist())
    missing = sorted(holdout_ids - present)
    if missing:
        raise SystemExit(
            f"{len(missing)} holdout episode(s) are not in the dataset: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}. The ridge would then be scored on "
            "a smaller holdout than the model was, and the two numbers would not be comparable."
        )
    holdout_mask = table.mask_for(holdout_ids)
    train_mask = ~holdout_mask
    if not holdout_mask.any():
        raise SystemExit("the holdout selects no chunks — nothing to score")
    if not train_mask.any():
        raise SystemExit("the holdout is the whole dataset — nothing to fit on")

    leaked = set(table.episode_ids[train_mask].tolist()) & holdout_ids
    if leaked:
        raise SystemExit(f"split is not episode-clean: {sorted(leaked)[:5]} appear on both sides")
    return train_mask, holdout_mask


# -- the fit ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Standardizer:
    """Per-dimension mean/std fitted on TRAIN ROWS ONLY, plus the columns worth keeping.

    Fitting normalisation on the union is the classic quiet leak: the holdout's own mean and scale
    end up inside the transform, the ridge is handed a holdout that has been centred using
    knowledge of itself, and every number downstream is optimistic by an amount nobody can bound
    afterwards. Which set the statistics came from is therefore a property of this object, not a
    comment in the caller.

    ``keep`` also drops train dims with zero variance. Those carry no information by construction
    and standardizing them would divide by zero; kept unstandardized they are exact duplicates of
    the bias column, which is what turns the normal equations singular at small lambda.
    """

    mean: np.ndarray
    std: np.ndarray
    keep: np.ndarray

    @classmethod
    def fit(cls, train_states: np.ndarray, columns: np.ndarray | None = None) -> Standardizer:
        """Fit on ``train_states`` [N, S], optionally restricted to a boolean ``columns`` mask."""
        mean = train_states.mean(axis=0)
        std = train_states.std(axis=0)
        keep = std > 0.0
        if columns is not None:
            keep = keep & columns
        return cls(mean=mean, std=std, keep=keep)

    @property
    def num_features(self) -> int:
        return int(self.keep.sum())

    @property
    def dropped(self) -> tuple[int, ...]:
        """Indices dropped for zero train variance (ignoring ones excluded by ``columns``)."""
        return tuple(int(i) for i in np.flatnonzero(self.std <= 0.0))

    def design(self, states: np.ndarray) -> np.ndarray:
        """[N, num_features + 1]: kept dims standardized with the TRAIN stats, plus a bias column.

        The bias rides inside the penalized system rather than being fitted separately. On
        standardized features the two agree to well below the seventh significant digit of any
        number this script prints, and one matrix is one thing to get wrong instead of two.
        """
        if states.shape[1] != self.keep.shape[0]:
            raise ValueError(
                f"state width {states.shape[1]} does not match the fitted {self.keep.shape[0]}"
            )
        z = (states[:, self.keep] - self.mean[self.keep]) / self.std[self.keep]
        return np.hstack([z, np.ones((z.shape[0], 1), dtype=np.float64)])


@dataclass(frozen=True)
class RidgeFit:
    """One (feature group, lambda) readout."""

    group: str
    lam: float
    holdout_mse: float
    num_features: int
    num_parameters: int


def fit_ridge(
    standardizer: Standardizer,
    train_states: np.ndarray,
    train_targets: np.ndarray,
    holdout_states: np.ndarray,
    holdout_targets: np.ndarray,
    lam: float,
    *,
    group: str = "all",
) -> RidgeFit:
    """Solve ``(X'X + lam*I) W = X'Y`` on the normal equations and score W on the holdout.

    Pure numpy on purpose. The whole claim is that this is a matrix solve rather than a model:
    reaching for sklearn or torch to compute 7 920 numbers would obscure exactly the thing the
    baseline is evidence for, and would add a dependency to a script whose selling point is that
    it runs anywhere in seconds. ``np.linalg.solve`` on the normal equations is well-conditioned
    here because the features are standardized and ``lam > 0`` — the system is symmetric positive
    definite by construction, and a ``LinAlgError`` therefore means an input problem, not a
    numerical one, so it is reported as such rather than silently pinv'd away.
    """
    if lam <= 0.0:
        raise SystemExit(
            f"lambda must be > 0, got {lam:g}. At 0 this is ordinary least squares and any two "
            "collinear state dims make the normal equations singular."
        )
    x = standardizer.design(train_states)
    gram = x.T @ x
    rhs = x.T @ train_targets
    try:
        weights = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), rhs)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - guarded by lam > 0 and `keep`
        raise SystemExit(
            f"the {group} normal equations are singular at lambda={lam:g}: {exc}. With "
            "standardized features and a positive lambda this should not happen; check for "
            "duplicated state dims."
        ) from exc
    predicted = standardizer.design(holdout_states) @ weights
    return RidgeFit(
        group=group,
        lam=float(lam),
        holdout_mse=float(((predicted - holdout_targets) ** 2).mean()),
        num_features=standardizer.num_features,
        num_parameters=int(weights.size),
    )


def feature_groups(num_joints: int, gripper_dims: int, state_dim: int) -> dict[str, np.ndarray]:
    """Boolean column masks for the reported ablations, over the raw state layout.

    Layout is ``state_vector``'s: q first, then dq, then the gripper columns. Reporting the three
    channels separately is what turns "a linear model wins" into something usable — ``dq`` alone
    beating the fine-tune while ``q`` alone loses to it says the signal is short-horizon momentum,
    which is a statement about what the demonstrations contain and about what a video backbone
    would have to add on top of it.
    """
    q = np.zeros(state_dim, dtype=bool)
    dq = np.zeros(state_dim, dtype=bool)
    gripper = np.zeros(state_dim, dtype=bool)
    q[:num_joints] = True
    dq[num_joints : 2 * num_joints] = True
    gripper[2 * num_joints : 2 * num_joints + gripper_dims] = True
    return {
        "all": np.ones(state_dim, dtype=bool),
        "q": q,
        "dq": dq,
        "gripper": gripper,
    }


# -- the blind nonlinear ceiling ------------------------------------------------------------------


@dataclass(frozen=True)
class CeilingConfig:
    """One point of the random-Fourier grid: width, RBF bandwidth, ridge penalty."""

    width: int
    gamma: float
    lam: float

    def __str__(self) -> str:
        return f"width {self.width}  gamma {self.gamma:g}  lambda {self.lam:g}"


@dataclass(frozen=True)
class CeilingFit:
    """The chosen config, what it scored on the INNER validation split, and its holdout MSE.

    ``val_mse`` is carried next to ``holdout_mse`` on purpose. They are computed on disjoint
    episodes by two different fits, so a reader can see the selection score and the reported score
    side by side and check that the ceiling was not simply the best of 48 tries on the holdout —
    which is the one way a number like this is usually wrong.
    """

    config: CeilingConfig
    holdout_mse: float
    val_mse: float
    num_parameters: int
    num_state_features: int
    num_inner_fit_episodes: int
    num_val_episodes: int
    num_inner_fit_rows: int
    num_val_rows: int
    grid: tuple[tuple[CeilingConfig, float], ...]

    @property
    def num_configs(self) -> int:
        return len(self.grid)


def inner_validation_episodes(
    train_episode_ids: np.ndarray,
    holdout_ids: set[str],
    *,
    num_val_episodes: int = CEILING_VAL_EPISODES,
    seed: int = CEILING_SPLIT_SEED,
) -> tuple[set[str], set[str]]:
    """``(inner fit episodes, inner validation episodes)``, drawn from TRAIN ONLY — and prove it.

    This is the one property that makes the ceiling meaningful. A nonlinear regressor with ~10^6
    parameters and a free bandwidth can be tuned to almost any number you like on a 1 040-chunk
    holdout; a ceiling chosen that way says nothing about what proprioception affords and
    everything about how many configs were tried. So the search never sees the holdout, and that
    is checked HERE, at runtime, over the episode tags of the rows actually handed in — not
    asserted in a docstring and not guaranteed by the caller having been written correctly.

    Two ways it raises, both of which have happened to somebody:

    - a holdout episode is present in ``train_episode_ids``. Then the "train" rows are not train
      rows, the split above them leaked, and every config score below is contaminated.
    - the selected validation set intersects the holdout. Unreachable given the first check, which
      is exactly why it is worth stating: it is the invariant, and an invariant that is only ever
      implied by another check is one refactoring away from being false.

    ``SystemExit`` rather than ``assert`` so that ``python -O`` cannot remove the guard.
    """
    present = sorted(set(train_episode_ids.tolist()))
    contaminated = sorted(set(present) & holdout_ids)
    if contaminated:
        raise SystemExit(
            f"the ceiling's hyperparameter search was handed {len(contaminated)} HOLDOUT "
            f"episode(s): {contaminated[:5]}{'...' if len(contaminated) > 5 else ''}. Choosing "
            "width, bandwidth or lambda on data the ceiling is then reported on turns the bar into "
            "a fit, and the number would be unusable rather than merely optimistic."
        )
    if num_val_episodes < 1:
        raise SystemExit(f"--ceiling-val-episodes must be >= 1, got {num_val_episodes}")
    if num_val_episodes >= len(present):
        raise SystemExit(
            f"--ceiling-val-episodes {num_val_episodes} leaves nothing to fit on: the train side "
            f"has {len(present)} episode(s)"
        )

    episodes = list(present)
    np.random.default_rng(seed).shuffle(episodes)
    validation = set(episodes[:num_val_episodes])
    inner_fit = set(episodes[num_val_episodes:])

    leaked = sorted(validation & holdout_ids)
    if leaked:  # pragma: no cover - unreachable while the check above stands, and that is the point
        raise SystemExit(
            f"the inner validation split selected holdout episode(s) {leaked[:5]} — the search "
            "would be choosing its hyperparameters on the data it is scored on"
        )
    if validation & inner_fit:  # pragma: no cover - set arithmetic, stated because load-bearing
        raise SystemExit("the inner split is not episode-disjoint")
    return inner_fit, validation


def ceiling_scores(
    standardizer: Standardizer,
    fit_states: np.ndarray,
    fit_targets: np.ndarray,
    eval_states: np.ndarray,
    *,
    width: int,
    gamma: float,
    lambdas: tuple[float, ...],
    rng: np.random.Generator,
) -> dict[float, np.ndarray]:
    """One random-Fourier draw, one Gram, ``{lambda: predictions on eval_states}``.

    ``cos(z @ W + b)`` with ``W ~ N(0, 2*gamma)`` and ``b ~ U(0, 2*pi)`` is Rahimi & Recht's
    approximation to an RBF kernel of bandwidth ``gamma``: a kernel ridge regression that stays a
    plain ``np.linalg.solve`` instead of becoming an N x N kernel matrix, which matters because the
    whole point of this script is that it is arithmetic rather than training.

    The draw comes from a caller-owned ``rng`` rather than a seed, because the archived search is
    ONE stream advanced across the grid and reproducing 5.431371e-06 requires that exact sequence.

    All lambdas share one Gram and one right-hand side. That is not just speed: re-forming
    ``X'X`` per lambda would be six chances for the six numbers to differ in their last bits for
    no reason, and the grid is compared at exactly that resolution.
    """
    if width < 1:
        raise SystemExit(f"ceiling width must be >= 1, got {width}")
    if gamma <= 0.0:
        raise SystemExit(f"ceiling gamma must be > 0, got {gamma:g}")
    z_fit = standardizer.design(fit_states)[:, :-1]
    z_eval = standardizer.design(eval_states)[:, :-1]
    weights = rng.normal(0.0, np.sqrt(2.0 * gamma), (z_fit.shape[1], width))
    offsets = rng.uniform(0.0, 2.0 * np.pi, width)

    design_fit = np.hstack([np.cos(z_fit @ weights + offsets), np.ones((z_fit.shape[0], 1))])
    design_eval = np.hstack([np.cos(z_eval @ weights + offsets), np.ones((z_eval.shape[0], 1))])
    gram = design_fit.T @ design_fit
    rhs = design_fit.T @ fit_targets
    eye = np.eye(gram.shape[0])

    predictions: dict[float, np.ndarray] = {}
    for lam in lambdas:
        if lam <= 0.0:
            raise SystemExit(f"ceiling lambda must be > 0, got {lam:g}")
        try:
            solved = np.linalg.solve(gram + lam * eye, rhs)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - guarded by lam > 0
            raise SystemExit(
                f"the ceiling's normal equations are singular at width={width}, lambda={lam:g}: "
                f"{exc}"
            ) from exc
        predictions[float(lam)] = design_eval @ solved
    return predictions


def fit_ceiling(
    train_states: np.ndarray,
    train_targets: np.ndarray,
    train_episode_ids: np.ndarray,
    holdout_states: np.ndarray,
    holdout_targets: np.ndarray,
    holdout_ids: set[str],
    *,
    widths: tuple[int, ...] = CEILING_WIDTHS,
    gammas: tuple[float, ...] = CEILING_GAMMAS,
    lambdas: tuple[float, ...] = CEILING_LAMBDAS,
    num_val_episodes: int = CEILING_VAL_EPISODES,
) -> CeilingFit:
    """Choose on an inner split of TRAIN, refit on all of TRAIN, score once on the holdout.

    The order of the three steps is the whole argument, so it is worth being explicit about what
    each one is allowed to see:

    1. ``inner_validation_episodes`` splits the TRAIN episodes into an inner-fit set and a
       validation set, and refuses outright if a holdout episode is among them.
    2. every ``(width, gamma, lambda)`` is fitted on the inner-fit rows and scored on the
       validation rows. The holdout arrays are not referenced anywhere in this loop — they are not
       even standardized yet, because the standardizer used here is fitted on the inner-fit rows.
    3. the single winner is refitted on ALL train rows, with a fresh feature draw and a
       standardizer fitted on all train rows, and scored on the holdout exactly once.

    Step 3 uses more data than step 2, so the reported number is not the validation score and is
    not expected to equal it. It is also not corrected for the selection in step 2 — with 48
    configs scored on 40 held-out-from-train episodes there is some optimism in the *choice*, but
    it lands on which of two near-tied widths is picked rather than on the holdout MSE, which was
    computed after the choice was frozen.
    """
    inner_fit_ids, val_ids = inner_validation_episodes(
        train_episode_ids, holdout_ids, num_val_episodes=num_val_episodes
    )
    val_mask = np.isin(train_episode_ids, sorted(val_ids))
    fit_mask = ~val_mask
    inner_states, inner_targets = train_states[fit_mask], train_targets[fit_mask]
    val_states, val_targets = train_states[val_mask], train_targets[val_mask]

    inner_standardizer = Standardizer.fit(inner_states)
    search_rng = np.random.default_rng(CEILING_SEARCH_SEED)
    grid: list[tuple[CeilingConfig, float]] = []
    for width in widths:
        for gamma in gammas:
            predicted = ceiling_scores(
                inner_standardizer,
                inner_states,
                inner_targets,
                val_states,
                width=width,
                gamma=gamma,
                lambdas=lambdas,
                rng=search_rng,
            )
            for lam, prediction in predicted.items():
                grid.append(
                    (
                        CeilingConfig(width=width, gamma=gamma, lam=lam),
                        float(((prediction - val_targets) ** 2).mean()),
                    )
                )
    if not grid:
        raise SystemExit("the ceiling grid is empty — nothing to select")
    grid.sort(key=lambda row: row[1])
    best, val_mse = grid[0]

    standardizer = Standardizer.fit(train_states)
    holdout_prediction = ceiling_scores(
        standardizer,
        train_states,
        train_targets,
        holdout_states,
        width=best.width,
        gamma=best.gamma,
        lambdas=(best.lam,),
        rng=np.random.default_rng(CEILING_REFIT_SEED),
    )[best.lam]
    return CeilingFit(
        config=best,
        holdout_mse=float(((holdout_prediction - holdout_targets) ** 2).mean()),
        val_mse=val_mse,
        num_parameters=(best.width + 1) * int(train_targets.shape[1]),
        num_state_features=standardizer.num_features,
        num_inner_fit_episodes=len(inner_fit_ids),
        num_val_episodes=len(val_ids),
        num_inner_fit_rows=int(fit_mask.sum()),
        num_val_rows=int(val_mask.sum()),
        grid=tuple(grid),
    )


# -- controls -----------------------------------------------------------------------------------


def zero_delta_mse(holdout_targets: np.ndarray) -> float:
    """``mean(target^2)`` — what holding perfectly still scores. The floor for "did anything"."""
    return float((holdout_targets**2).mean())


def model_mse_from_predictions(path: Path) -> tuple[float, int] | None:
    """The model's own MSE recomputed from a ``predictions.jsonl``, or ``None`` for an id list.

    Recomputed rather than read out of the neighbouring ``bench.json``, because the comparison is
    only worth printing if both sides are computed here from the same definition — mean squared
    error over the flattened target chunk. A number lifted from a report cannot be checked and can
    silently be a different metric (bench.json also carries a critical-subset MSE that differs by
    more than the effect being measured).

    Detection is by content, not by filename: ``load_episode_ids`` already accepts both shapes, so
    a caller may legitimately pass either, and a plain id list simply has no model to compare to.
    """
    total = 0.0
    chunks = 0
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("{"):
            return None
        try:
            record = json.loads(line)
            predicted = np.asarray(record["predicted"]["targets"], dtype=np.float64)
            target = np.asarray(record["target"]["targets"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"{path}:{line_no}: not a readable prediction record: {exc}") from exc
        if predicted.shape != target.shape:
            raise SystemExit(
                f"{path}:{line_no}: predicted {predicted.shape} != target {target.shape}"
            )
        total += float(((predicted - target) ** 2).mean())
        chunks += 1
    if chunks == 0:
        return None
    return total / chunks, chunks


def check_archived(measured: dict[str, float | None], shape: dict[str, int]) -> str | None:
    """Re-verify every number this module wrote down, or raise. ``None`` off the archived shape.

    The reason this is a hard failure and not a warning: the rows in :data:`ARCHIVED_T16` are the
    denominator of three separate write-ups. PR-01's verdict is stated as ratios against the ridge,
    the momentum follow-up is stated as ratios against zero-delta and const-velocity, and the
    ceiling below is stated as a ratio against the linear bar. If ``collect_chunks`` starts pairing
    one chunk differently, or ``Standardizer`` changes which dims it drops, every one of those
    ratios silently becomes a claim about data nobody measured — and the printed table would look
    completely normal. A run that cannot reproduce its own archive is not a run with a slightly
    different number, it is a run whose comparisons have quietly stopped meaning anything.

    Comparison is on the six-decimal strings, because that is the precision the numbers were
    published at. A tolerance would be a decision about how much drift is acceptable, and there is
    no such amount: these are deterministic closed-form solves on a frozen dataset.
    """
    if shape != ARCHIVED_T16_SHAPE:
        return None
    moved = [
        (key, expected, _fmt(measured[key]))
        for key, expected in ARCHIVED_T16.items()
        if measured.get(key) is not None and _fmt(measured[key]) != expected
    ]
    if moved:
        lines = "\n".join(f"    {key:<24} archived {a}  now {b}" for key, a, b in moved)
        raise SystemExit(
            f"{len(moved)} archived number(s) MOVED on the T-16 shape this run matches:\n{lines}\n"
            "These are the control for PR-01, the momentum follow-up and the nonlinear ceiling, "
            "all of which are stated as ratios against them. Whatever changed, revert it and "
            "report the drift — a re-derived table is not a substitute for the one already cited."
        )
    checked = sum(1 for key in ARCHIVED_T16 if measured.get(key) is not None)
    skipped = len(ARCHIVED_T16) - checked
    tail = f" ({skipped} not computed this run)" if skipped else ""
    return f"archive   {checked}/{len(ARCHIVED_T16)} archived numbers reproduce exactly{tail}"


# -- CLI ----------------------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="episode directory root, e.g. datasets/gr00t-apple-full",
    )
    parser.add_argument(
        "--holdout",
        type=Path,
        required=True,
        help="a predictions.jsonl OR a one-per-line episode id list. Prefer the predictions: the "
        "split then has one definition and the model's own MSE is recomputed alongside",
    )
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=DEFAULT_CHUNK_STEPS,
        help="chunk horizon the targets are truncated to (default: %(default)s)",
    )
    parser.add_argument(
        "--lam",
        type=float,
        action="append",
        default=None,
        help="ridge penalty; repeatable. Default: " + " ".join(f"{v:g}" for v in DEFAULT_LAMBDAS),
    )
    parser.add_argument(
        "--no-ceiling",
        action="store_true",
        help="skip the blind NONLINEAR ceiling (a ~2 minute hyperparameter search). The linear "
        "rows still print; the run then reports a bar that understates what is knowable blind",
    )
    parser.add_argument(
        "--ceiling-val-episodes",
        type=int,
        default=CEILING_VAL_EPISODES,
        help="inner validation episodes for the ceiling's hyperparameter search, taken from the "
        "TRAIN episodes only (default: %(default)s)",
    )
    parser.add_argument(
        "--ceiling-width",
        type=int,
        action="append",
        default=None,
        help="random-Fourier feature count to search; repeatable. Default: "
        + " ".join(f"{v:g}" for v in CEILING_WIDTHS),
    )
    parser.add_argument(
        "--ceiling-gamma",
        type=float,
        action="append",
        default=None,
        help="RBF bandwidth to search; repeatable. Default: "
        + " ".join(f"{v:g}" for v in CEILING_GAMMAS),
    )
    parser.add_argument(
        "--ceiling-lam",
        type=float,
        action="append",
        default=None,
        help="ridge penalty to search for the ceiling; repeatable. Default: "
        + " ".join(f"{v:g}" for v in CEILING_LAMBDAS),
    )
    parser.add_argument("--json", type=Path, default=None, help="write the full record here")
    return parser.parse_args(argv)


def _fmt(value: float) -> str:
    return f"{value:.6e}"


def _mse_at(results: dict[str, list[RidgeFit]], group: str, lam: float) -> float | None:
    """The holdout MSE of one ``(group, lambda)`` cell, or ``None`` if this run did not fit it.

    The archive is keyed by lambda rather than by "best", because "best" is an argmin over whatever
    ``--lam`` happened to be and would compare two different fits across two runs. A custom
    ``--lam`` simply leaves the cell uncomputed, and :func:`check_archived` skips it rather than
    reporting a mismatch that is really a different question.
    """
    for fit in results.get(group, ()):
        if fit.lam == lam:
            return fit.holdout_mse
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    lambdas = tuple(args.lam) if args.lam else DEFAULT_LAMBDAS
    if args.chunk_steps < 1:
        raise SystemExit(f"--chunk-steps must be >= 1, got {args.chunk_steps}")

    holdout_ids = load_episode_ids(args.holdout)
    if not holdout_ids:
        raise SystemExit(f"{args.holdout}: no episode ids found")

    table = collect_chunks(args.dataset, args.chunk_steps)
    train_mask, holdout_mask = split_by_episode(table, holdout_ids)
    train_states, train_targets = table.select(train_mask)
    holdout_states, holdout_targets = table.select(holdout_mask)
    num_train_eps = len(set(table.episode_ids[train_mask].tolist()))

    print(f"dataset  {args.dataset}")
    print(f"holdout  {args.holdout}")
    print(
        f"chunks   train {train_states.shape[0]} ({num_train_eps} eps) / "
        f"holdout {holdout_states.shape[0]} ({len(holdout_ids)} eps)  |  "
        f"state {table.state_dim} dims (q{table.num_joints} + dq{table.num_joints} + "
        f"gripper{table.gripper_dims})  |  target {table.chunk_steps} x {table.num_joints} = "
        f"{table.target_dim}"
    )

    groups = feature_groups(table.num_joints, table.gripper_dims, table.state_dim)
    dropped = Standardizer.fit(train_states).dropped
    if dropped:
        print(f"dropped  {len(dropped)} state dim(s) with zero train variance: {list(dropped)}")

    results: dict[str, list[RidgeFit]] = {}
    for group in FEATURE_GROUPS:
        standardizer = Standardizer.fit(train_states, columns=groups[group])
        if standardizer.num_features == 0:
            print(f"skipping {group}: no live columns on train")
            continue
        results[group] = [
            fit_ridge(
                standardizer,
                train_states,
                train_targets,
                holdout_states,
                holdout_targets,
                lam,
                group=group,
            )
            for lam in lambdas
        ]

    print()
    print("holdout MSE (mean over the flattened target chunk), by feature group and lambda")
    header = f"  {'group':<9}{'params':>8}  " + "  ".join(f"{lam:>12g}" for lam in lambdas)
    print(header + f"  {'best':>12}")
    for group, fits in results.items():
        best = min(fits, key=lambda f: f.holdout_mse)
        cells = "  ".join(f"{_fmt(f.holdout_mse):>12}" for f in fits)
        print(f"  {group:<9}{fits[0].num_parameters:>8}  {cells}  {_fmt(best.holdout_mse):>12}")

    zero = zero_delta_mse(holdout_targets)
    model = model_mse_from_predictions(args.holdout)
    print()
    print("controls on the same chunks")
    print(f"  zero-delta (hold still)      {_fmt(zero)}")
    if model is None:
        print(
            "  model                        n/a — --holdout is an episode id list, so there is "
            "no run to score against"
        )
    else:
        model_mse, model_chunks = model
        print(f"  model (from predictions)     {_fmt(model_mse)}   over {model_chunks} chunks")
        if model_chunks != holdout_states.shape[0]:
            print(
                f"  WARNING: the run scored {model_chunks} chunks, this pass built "
                f"{holdout_states.shape[0]}. The two MSEs are then over different rows and are "
                "NOT comparable — check --chunk-steps and that the dataset has not moved."
            )

    ceiling: CeilingFit | None = None
    if not args.no_ceiling:
        print()
        print(
            f"blind NONLINEAR ceiling — random-Fourier ridge on the same {table.state_dim} dims, "
            "camera still shut"
        )
        ceiling = fit_ceiling(
            train_states,
            train_targets,
            table.episode_ids[train_mask],
            holdout_states,
            holdout_targets,
            holdout_ids,
            widths=tuple(args.ceiling_width) if args.ceiling_width else CEILING_WIDTHS,
            gammas=tuple(args.ceiling_gamma) if args.ceiling_gamma else CEILING_GAMMAS,
            lambdas=tuple(args.ceiling_lam) if args.ceiling_lam else CEILING_LAMBDAS,
            num_val_episodes=args.ceiling_val_episodes,
        )
        print(
            f"  inner split    {ceiling.num_inner_fit_episodes} fit / {ceiling.num_val_episodes} "
            f"validation episodes ({ceiling.num_inner_fit_rows} / {ceiling.num_val_rows} chunks), "
            "TRAIN only — no holdout episode is reachable from the search"
        )
        print(
            f"  searched       {ceiling.num_configs} configs on validation, best {ceiling.config} "
            f"at val {_fmt(ceiling.val_mse)}"
        )
        print(
            f"  ceiling        {_fmt(ceiling.holdout_mse)}  from {ceiling.num_parameters} "
            f"parameters ({ceiling.config.width} random features + bias, x {table.target_dim} "
            "outputs), refitted on all train episodes"
        )

    print()
    best_all = min(results["all"], key=lambda f: f.holdout_mse)
    print(
        f"ridge (all state, lambda {best_all.lam:g})  {_fmt(best_all.holdout_mse)}  from "
        f"{best_all.num_parameters} parameters "
        f"({best_all.num_features} features + bias, x {table.target_dim} outputs)"
    )
    print(f"  vs zero-delta   {zero / best_all.holdout_mse:.2f}x better")
    if model is not None:
        ratio = model[0] / best_all.holdout_mse
        verdict = (
            f"the blind linear map is {ratio:.2f}x BETTER than the model — that model has not "
            "demonstrated it uses vision at all"
            if ratio > 1.0
            else f"the model is {1.0 / ratio:.2f}x better than the blind linear map"
        )
        print(f"  vs the model    {ratio:.2f}x   <- {verdict}")

    if ceiling is not None:
        print()
        print(f"ceiling (blind nonlinear)  {_fmt(ceiling.holdout_mse)}")
        print(
            f"  vs the linear bar  {best_all.holdout_mse / ceiling.holdout_mse:.2f}x lower — "
            "that much of the linear bar's headroom was nonlinearity in the body, not the world"
        )
        if model is not None:
            ratio = model[0] / ceiling.holdout_mse
            verdict = (
                f"the model is {ratio:.2f}x WORSE than a blind nonlinear regressor — it has not "
                "demonstrated anything proprioception alone could not do"
                if ratio > 1.0
                else f"the model is {1.0 / ratio:.2f}x better than the blind nonlinear ceiling — "
                "the first evidence here that is not explainable without the camera"
            )
            print(f"  vs the model       {ratio:.2f}x   <- {verdict}")
    else:
        print()
        print(
            "--no-ceiling: the NONLINEAR bar was not computed, so 'beats the ridge' above is the "
            "weaker of the two claims a blind regressor can make"
        )

    shape = {
        "num_train_chunks": int(train_states.shape[0]),
        "num_holdout_chunks": int(holdout_states.shape[0]),
        "num_train_episodes": num_train_eps,
        "num_holdout_episodes": len(holdout_ids),
        "state_dim": table.state_dim,
        "target_dim": table.target_dim,
    }
    archive = check_archived(
        {
            "zero_delta": zero,
            "model": None if model is None else model[0],
            "ceiling": None if ceiling is None else ceiling.holdout_mse,
            "ridge_all_lam0.01": _mse_at(results, "all", 0.01),
            "ridge_all_lam10": _mse_at(results, "all", 10.0),
            "ridge_all_lam100": _mse_at(results, "all", 100.0),
            "ridge_dq_lam1": _mse_at(results, "dq", 1.0),
            "ridge_q_lam1": _mse_at(results, "q", 1.0),
            "ridge_gripper_lam0.01": _mse_at(results, "gripper", 0.01),
        },
        shape,
    )
    if archive is not None:
        print()
        print(archive)

    if args.json is not None:
        record: dict[str, Any] = {
            "dataset": str(args.dataset),
            "holdout": str(args.holdout),
            "chunk_steps": args.chunk_steps,
            "lambdas": list(lambdas),
            "num_train_chunks": int(train_states.shape[0]),
            "num_holdout_chunks": int(holdout_states.shape[0]),
            "num_train_episodes": num_train_eps,
            "num_holdout_episodes": len(holdout_ids),
            "state_dim": table.state_dim,
            "target_dim": table.target_dim,
            "num_joints": table.num_joints,
            "gripper_dims": table.gripper_dims,
            "dropped_state_dims": list(dropped),
            "zero_delta_mse": zero,
            "model_mse": None if model is None else model[0],
            "model_chunks": None if model is None else model[1],
            "archive_checked": archive is not None,
            "ceiling": None
            if ceiling is None
            else {
                "holdout_mse": ceiling.holdout_mse,
                "val_mse": ceiling.val_mse,
                "width": ceiling.config.width,
                "gamma": ceiling.config.gamma,
                "lam": ceiling.config.lam,
                "num_parameters": ceiling.num_parameters,
                "num_state_features": ceiling.num_state_features,
                "num_configs": ceiling.num_configs,
                "num_inner_fit_episodes": ceiling.num_inner_fit_episodes,
                "num_val_episodes": ceiling.num_val_episodes,
                "num_inner_fit_chunks": ceiling.num_inner_fit_rows,
                "num_val_chunks": ceiling.num_val_rows,
                # The whole search, so the selection can be re-read instead of taken on trust.
                "val_grid": [
                    {"width": c.width, "gamma": c.gamma, "lam": c.lam, "val_mse": v}
                    for c, v in ceiling.grid
                ],
            },
            "groups": {
                group: {
                    "num_parameters": fits[0].num_parameters,
                    "num_features": fits[0].num_features,
                    "mse_by_lambda": {f"{f.lam:g}": f.holdout_mse for f in fits},
                    "best_lambda": min(fits, key=lambda f: f.holdout_mse).lam,
                    "best_mse": min(f.holdout_mse for f in fits),
                }
                for group, fits in results.items()
            },
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
