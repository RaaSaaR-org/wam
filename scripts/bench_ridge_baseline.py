#!/usr/bin/env python3
"""The proprioception-only ridge: the bar a visual model must clear to have used its eyes.

A linear least-squares map from the 32-dim robot state at chunk time (q15 + dq15 + gripper2)
to the flattened [16, 15] action chunk. No frames are read. No backbone is built. Nothing is
trained. It is 7 920 numbers solved in closed form, and on the T-16 holdout it beats the
deployed model:

    ridge, all state      6.330899e-06     7 920 parameters, one np.linalg.solve
    ridge, dq only        6.869239e-06     3 840 parameters
    model (Wan-5B+LoRA)   1.112983e-05     82.5M trainable parameters
    ridge, q only         1.348259e-05
    ridge, gripper only   1.550558e-05
    zero-delta (hold still) 1.632760e-05

Measured 2026-08-01 on ``datasets/gr00t-apple-full`` against
``runs/t16-lora-seed0/eval-t30-regression/predictions.jsonl``, and independently reproduced by
two separate implementations before being written down. The linear map is **1.76x better** than
the fine-tune; velocity alone is still **1.62x better**. Position alone and gripper alone both
lose to it, so the win is not "proprioception is trivially sufficient" — it is specifically ``dq``,
the one channel that says where the arm is already going.

WHY THIS SHIPS AS A PERMANENT BASELINE
--------------------------------------
Zero-delta and repeat-last-action (``wam.evaluation.bench``) ask whether a policy beats holding
still and whether it beats the last command. Both are answerable without looking at the robot at
all. This one is different in kind: it is the best a model can do *knowing everything about the
body and nothing about the world*. A visual policy that scores above it has not demonstrated that
it uses vision — it has demonstrated that it is a worse proprioceptive regressor than a matrix
solve. Whatever its backbone costs in parameters, GPU-hours or LoRA rank, it has not earned it.

That makes the ridge the bar for the claim, not just another row in a table. It belongs next to
every WAM-Bench readout for the same reason the majority-class rate belongs next to a gripper
accuracy: without it the number is unreadable, and the failure it hides is the expensive one.

It is deliberately BLIND. The camera is never opened, which is what makes the comparison mean
something and also what makes it run in seconds on a laptop instead of minutes on a GPU. Any
future run can be checked against it for free.

WHAT IS NOT CLAIMED
-------------------
That the ridge is a policy. It is fitted on the demonstrations of a single task, it has no notion
of where the apple is, and it would not survive the apple moving — which is exactly the
generalization the video branch exists to buy. E1 action-MSE is a DIAGNOSTIC metric (PRD 10.4)
and this baseline is a diagnostic on that diagnostic. Losing to it does not make a model useless;
it makes the *offline MSE evidence* for that model worthless, which is a narrower and much more
actionable statement.

``--lam`` is swept and the best holdout MSE is reported, which is a mild selection on the holdout
and is named here rather than hidden. It buys almost nothing: across four orders of magnitude of
lambda the all-state number moves in its seventh significant digit — 6.330899e-06 at 1e-2, the
grid minimum 6.330877e-06 at 1e1, 6.333218e-06 at 1e2. Quoting any of them makes the same claim to
the six digits anyone reads. The baseline is not tuned into its win, and the full per-lambda table
is printed so a reader can see that for themselves instead of taking it on trust.

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
    parser.add_argument("--json", type=Path, default=None, help="write the full record here")
    return parser.parse_args(argv)


def _fmt(value: float) -> str:
    return f"{value:.6e}"


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
