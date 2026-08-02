#!/usr/bin/env python3
"""PR-03's blind control suite and power gate — CPU only, no checkpoint, no GPU.

`docs/preregistration/PR-03-grasp-anticipation.md` scores one thing: whether a fine-tune can
anticipate a grasp flip that no *blind* (no-vision) predictor can. This script computes the blind
side of that comparison, which is everything needed to decide whether the refit is worth
submitting at all:

  GATE 1, the archive gate  restrict the run to the original 40-episode T-18 holdout (a subset of
                            PR-03's 150 by construction) and the five controls must reproduce
                            `PR-01-GRIPPER.md`'s table -- +-0.5 points for the two zero-parameter
                            rules, +-2.0 for the three fitted ones.
  GATE 2, the power gate    n_postflip >= 2000 and the episode-bootstrap CI half-width on the
                            ceiling <= 3.5 points. Below that resolution the refit cannot
                            distinguish a real effect from noise and PR-03 says do not submit it.

Both gates are pre-registered; this script does not decide them, it measures them and prints the
comparison. Run:

    .venv/bin/python scripts/bench_grasp_anticipation.py \
        --dataset datasets/gr00t-apple-grip \
        --holdout configs/splits/pr03_holdout_150.txt

`--restrict configs/splits/t18_holdout_episodes.txt` scores the same fitted predictors on the
40-episode subset, which is gate 1. The training pool is always "dataset minus --holdout", so a
restricted score changes what is SCORED and never what was FITTED.

WHAT COUNTS AS A FLIP. Three definitions, all reported, because `PR-01-GRIPPER.md` quoted the most
generous one without saying so and had to correct itself. All three latch with the shipped
hysteresis dead band (`wam.evaluation.gripper.latched_states`, threshold 0.5 +- 0.10) so the metric
and the admissibility gate cannot disagree about what a grasp is; they differ only in what context
the latch may use:

  episode-latch    the latch runs over the whole episode, and the step before the chunk's first
                   target step is the chunk's own start frame. A chunk sees the state the robot
                   was actually in. PRIMARY -- it is the one the audit's per-episode counts use.
  self-contained   the latch is rebuilt from the chunk's 16 target values alone. A chunk that
                   opens without a decisive closed sample inside itself is not a transition.
  label-steps      the episode latch, but without the carry-in from the chunk start: only changes
                   BETWEEN target steps count.

Post-flip steps are those at and after the first transition index in the chunk. Accuracy is on the
0.5-binarized channel.

HONEST LIMITATION, and it bounds gate 1. `PR-01-GRIPPER.md`'s numbers were measured in a scratch
script that was never committed, so the archived table cannot be reproduced by re-running its code
-- only by re-implementing its description, which is what this file is. A miss inside tolerance is
therefore evidence the channel survived conversion; a miss OUTSIDE tolerance is ambiguous between
"the channel differs" and "this re-implementation differs", and must be reported as ambiguous
rather than as a channel finding. Committing this script is what stops that recurring.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

CHUNK_STEPS = 16
ACTIVE_HAND = 0
"""Column of ``gripper_state`` the ``active-hand`` conversion writes the moving hand into."""

LAGS = (0, 1, 2, 3, 4, 6, 8, 12)
"""Frame lags the blind ceiling sees, ~0.4 s of history at 30 fps (PR-01-GRIPPER's 8-lag family)."""

N_FEATURES = 2048
GAMMA_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
"""RBF bandwidths as multiples of ``1 / n_input_dims``, not absolute numbers.

Squared distances between standardized points grow like the dimension, so a fixed grid tuned on a
32-dim input collapses the kernel to the identity on a 256-dim one (exp(-0.02 * 512) is 0) and the
"ceiling" silently becomes a memorizer that generalizes at chance. Scaling by 1/D keeps the same
grid meaningful whichever lag set is used.
"""
LAMBDAS = (1e-3, 1e-1, 1e1, 1e3)
TIME_BINS = 20
BOOTSTRAP_RESAMPLES = 5000

#: PR-01-GRIPPER.md's archived post-flip accuracies on the 40-episode holdout, and the tolerance
#: PR-03 pre-registers for each: the two zero-parameter rules are deterministic given the channel,
#: the three fitted ones depend on a random-feature draw and a hyperparameter search.
ARCHIVED = {
    "repeat-last": (19.70, 0.5),
    "const-velocity": (62.88, 0.5),
    "ridge-32": (24.09, 2.0),
    "time-only": (39.09, 2.0),
    "ceiling": (70.91, 2.0),
}

N_POSTFLIP_MIN = 2000
CI_HALFWIDTH_MAX = 3.5


@dataclass
class Episode:
    """One episode's blind state stream and its non-overlapping 16-step gripper target chunks."""

    episode_id: str
    features: np.ndarray  # (n_chunks, D) blind state at each chunk's start frame
    targets: np.ndarray  # (n_chunks, 16) demonstrated gripper target
    observed: np.ndarray  # (n_chunks,) gripper at the chunk's start frame
    velocity: np.ndarray  # (n_chunks,) backward difference of the gripper at that frame
    phase: np.ndarray  # (n_chunks,) start frame / episode length, in [0, 1)
    latched_start: np.ndarray  # (n_chunks,) episode latch state at the start frame
    latched_steps: np.ndarray  # (n_chunks, 16) episode latch state at each target step
    flips: dict[str, np.ndarray] = field(default_factory=dict)  # definition -> first flip index


def _lagged(stream: np.ndarray, frame: int) -> np.ndarray:
    """``stream`` at ``frame`` minus each lag, clamped at the episode start (no future reads)."""
    return np.concatenate([stream[max(frame - lag, 0)] for lag in LAGS])


def load_episode(root: Path, episode_id: str) -> Episode | None:
    from wam.evaluation.gripper import latched_states

    states = pq.read_table(
        root / episode_id / "states.parquet", columns=["q", "dq", "gripper_state"]
    ).to_pydict()
    q = np.asarray([np.asarray(v, dtype=np.float64) for v in states["q"]])
    dq = np.asarray([np.asarray(v, dtype=np.float64) for v in states["dq"]])
    grip = np.asarray([np.asarray(v, dtype=np.float64) for v in states["gripper_state"]])
    n_frames = len(q)

    actions = pq.read_table(
        root / episode_id / "actions.parquet", columns=["chunk_idx", "step_idx", "gripper_target"]
    ).to_pydict()
    chunk_idx = np.asarray(actions["chunk_idx"], dtype=np.int64)
    step_idx = np.asarray(actions["step_idx"], dtype=np.int64)
    gripper_target = np.asarray(actions["gripper_target"], dtype=np.float64).reshape(-1)

    stream = np.hstack([q, dq, grip])
    active = grip[:, ACTIVE_HAND]
    latch = latched_states(active)

    rows, feats, obs, vel, phase, lat_start, lat_steps = [], [], [], [], [], [], []
    for chunk in sorted(set(chunk_idx.tolist())):
        sel = chunk_idx == chunk
        if int(sel.sum()) < CHUNK_STEPS:
            continue  # a trailing partial chunk is not a 16-step chunk
        order = np.argsort(step_idx[sel])
        target = gripper_target[sel][order][:CHUNK_STEPS]
        start = int(chunk) * CHUNK_STEPS
        if start + CHUNK_STEPS >= n_frames:
            continue
        # The converter writes gripper_target[k] = the active hand's synergy at start+k+1. If that
        # does not hold, every flip index below is off by a frame and every number is wrong, so it
        # is checked rather than assumed.
        expected = active[start + 1 : start + CHUNK_STEPS + 1]
        if not np.allclose(target, expected, atol=1e-6):
            raise SystemExit(
                f"{episode_id} chunk {chunk}: gripper_target does not match "
                f"gripper_state[start+1:][:, {ACTIVE_HAND}] — the chunk/frame alignment this "
                "script assumes does not hold for this dataset"
            )
        rows.append(target)
        feats.append(_lagged(stream, start))
        obs.append(active[start])
        vel.append(active[start] - active[start - 1] if start > 0 else 0.0)
        phase.append(start / max(n_frames, 1))
        lat_start.append(latch[start])
        lat_steps.append(latch[start + 1 : start + CHUNK_STEPS + 1])
    if not rows:
        return None

    episode = Episode(
        episode_id=episode_id,
        features=np.asarray(feats),
        targets=np.asarray(rows),
        observed=np.asarray(obs),
        velocity=np.asarray(vel),
        phase=np.asarray(phase),
        latched_start=np.asarray(lat_start),
        latched_steps=np.asarray(lat_steps),
    )
    episode.flips = {name: fn(episode) for name, fn in FLIP_DEFINITIONS.items()}
    return episode


# --------------------------------------------------------------------- the three flip definitions
#
# Each returns, per chunk, the index of the first transition among the 16 target steps, or -1.


def _first_change(states: np.ndarray) -> np.ndarray:
    """First column where a decided latch differs from the previous decided one, else -1."""
    out = np.full(len(states), -1, dtype=np.int64)
    for i, row in enumerate(states):
        previous = -1
        for k, value in enumerate(row):
            if value < 0:
                continue
            if previous >= 0 and value != previous:
                out[i] = k
                break
            previous = value
    return out


def flips_episode_latch(ep: Episode) -> np.ndarray:
    """Episode-wide latch, with the chunk's own start frame as the step-before state."""
    return _first_change(np.hstack([ep.latched_start[:, None], ep.latched_steps])) - 1


def flips_self_contained(ep: Episode) -> np.ndarray:
    """Latch rebuilt from the chunk's 16 target values alone — no carry-in from the episode."""
    from wam.evaluation.gripper import latched_states

    return _first_change(np.asarray([latched_states(row) for row in ep.targets]))


def flips_label_steps(ep: Episode) -> np.ndarray:
    """Episode latch, but only changes BETWEEN target steps count — no carry-in from the start."""
    return _first_change(ep.latched_steps)


FLIP_DEFINITIONS = {
    "episode-latch": flips_episode_latch,
    "self-contained": flips_self_contained,
    "label-steps": flips_label_steps,
}
PRIMARY_DEFINITION = "episode-latch"


# ------------------------------------------------------------------------------ blind predictors


def _ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    a = np.hstack([x, np.ones((len(x), 1))])
    return np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ y)


def _apply(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.hstack([x, np.ones((len(x), 1))]) @ coef


def _standardize(fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, std = fit.mean(0), fit.std(0)
    return mean, np.where(std > 1e-9, std, 1.0)


def _time_basis(phase: np.ndarray) -> np.ndarray:
    """One-hot over TIME_BINS deciles-of-episode + the raw phase — state-free by construction."""
    index = np.clip((phase * TIME_BINS).astype(int), 0, TIME_BINS - 1)
    basis = np.zeros((len(phase), TIME_BINS + 1))
    basis[np.arange(len(phase)), index] = 1.0
    basis[:, TIME_BINS] = phase
    return basis


def _rff(x: np.ndarray, weights: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0 / weights.shape[1]) * np.cos(x @ weights + offset)


def _postflip_accuracy(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return 100.0 * float(((pred >= 0.5) == (target >= 0.5))[mask].mean())


def _stack(episodes: list[Episode], attr: str) -> np.ndarray:
    return np.concatenate([getattr(ep, attr) for ep in episodes])


def build_predictions(
    train: list[Episode], score: list[Episode], seed: int
) -> dict[str, np.ndarray]:
    """Every blind predictor's (n_chunks, 16) prediction on ``score``, fitted only on ``train``."""
    rng = np.random.default_rng(seed)
    steps = np.arange(1, CHUNK_STEPS + 1)[None, :]

    x_train, y_train = _stack(train, "features"), _stack(train, "targets")
    x_score = _stack(score, "features")
    observed, velocity = _stack(score, "observed"), _stack(score, "velocity")

    preds: dict[str, np.ndarray] = {
        "repeat-last": np.repeat(observed[:, None], CHUNK_STEPS, axis=1),
        "const-velocity": observed[:, None] + velocity[:, None] * steps,
    }

    # The 32-dim state control sees only the current frame: q, dq, gripper (lag 0).
    lag0 = x_train.shape[1] // len(LAGS)
    mean, std = _standardize(x_train[:, :lag0])
    coef = _ridge((x_train[:, :lag0] - mean) / std, y_train, 1e-1)
    preds["ridge-32"] = _apply((x_score[:, :lag0] - mean) / std, coef)

    coef = _ridge(_time_basis(_stack(train, "phase")), y_train, 1e-1)
    preds["time-only"] = _apply(_time_basis(_stack(score, "phase")), coef)

    preds["ceiling"] = _fit_ceiling(train, x_train, y_train, x_score, rng)
    return preds


def _fit_ceiling(
    train: list[Episode],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_score: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """RFF + ridge, hyperparameters chosen on an INNER episode-disjoint split of train only.

    Two choices make this a ceiling rather than just another predictor, and both raise the bar the
    model has to clear — i.e. both err against over-claiming:

    * **Fitted on the BINARIZED target**, unlike the ridge-32 control, which regresses the
      continuous channel. PR-03 scores accuracy at a 0.5 threshold, and a least-squares fit to the
      continuous value is mean-seeking: it lands on the stale side of the threshold exactly where
      the metric looks, which is why the linear controls score *below chance* post-flip even
      though their span contains the const-velocity rule. A ceiling fitted for MSE would not be a
      ceiling on accuracy, it would be a fourth way of restating the momentum result.
    * **Selected on inner post-flip accuracy**, so the hyperparameters target the scored quantity.

    The result is asserted to dominate every other blind predictor; if it does not, this family is
    not a ceiling on that holdout and the caller is told rather than quietly given a low bar.
    """
    counts = [len(ep.targets) for ep in train]
    bounds = np.cumsum([0, *counts])
    inner_episodes = set(range(0, len(train), 5))  # every 5th episode, episode-disjoint
    inner = np.zeros(len(x_train), dtype=bool)
    for i in inner_episodes:
        inner[bounds[i] : bounds[i + 1]] = True
    fit = ~inner

    inner_flip = np.concatenate(
        [ep.flips[PRIMARY_DEFINITION] for i, ep in enumerate(train) if i in inner_episodes]
    )
    inner_targets = np.concatenate(
        [ep.targets for i, ep in enumerate(train) if i in inner_episodes]
    )
    inner_mask = postflip_mask(inner_flip)

    mean, std = _standardize(x_train[fit])
    z_train, z_score = (x_train - mean) / std, (x_score - mean) / std
    labels = (y_train >= 0.5).astype(np.float64)

    best: tuple[float, np.ndarray] | None = None
    for scale in GAMMA_SCALES:
        gamma = scale / z_train.shape[1]
        weights = rng.normal(0.0, np.sqrt(2 * gamma), (z_train.shape[1], N_FEATURES))
        offset = rng.uniform(0.0, 2 * np.pi, N_FEATURES)
        # The raw standardized block rides along so the model can always fall back on the linear
        # solution; without it a badly scaled kernel can score below a zero-parameter rule.
        f_train = np.hstack([_rff(z_train, weights, offset), z_train])
        f_score = np.hstack([_rff(z_score, weights, offset), z_score])
        for lam in LAMBDAS:
            coef = _ridge(f_train[fit], labels[fit], lam)
            accuracy = _postflip_accuracy(_apply(f_train[inner], coef), inner_targets, inner_mask)
            if not np.isnan(accuracy) and (best is None or accuracy > best[0]):
                best = (accuracy, _apply(f_score, _ridge(f_train, labels, lam)))
    if best is None:
        raise SystemExit("no inner post-flip steps — cannot select the ceiling's hyperparameters")
    return best[1]


def postflip_mask(flip: np.ndarray) -> np.ndarray:
    """(n_chunks, 16) boolean: steps at and after the chunk's first flip; all-False if none."""
    steps = np.arange(CHUNK_STEPS)[None, :]
    return (flip[:, None] >= 0) & (steps >= flip[:, None])


def preflip_mask(flip: np.ndarray) -> np.ndarray:
    steps = np.arange(CHUNK_STEPS)[None, :]
    return (flip[:, None] >= 0) & (steps < flip[:, None])


def window_mask(flip: np.ndarray, width: int = 4) -> np.ndarray:
    steps = np.arange(CHUNK_STEPS)[None, :]
    return (flip[:, None] >= 0) & (steps >= flip[:, None]) & (steps < flip[:, None] + width)


def bootstrap_halfwidth(correct: list[np.ndarray], total: list[np.ndarray], seed: int = 0) -> float:
    """Episode-level bootstrap half-width in accuracy points: resample EPISODES, not steps.

    Steps inside one episode are not independent — a single grasp contributes a whole run of
    post-flip steps — so a step-level interval would be optimistic by roughly the run length.
    """
    rng = np.random.default_rng(seed)
    hits, counts = np.asarray(correct, dtype=np.float64), np.asarray(total, dtype=np.float64)
    keep = counts > 0
    hits, counts = hits[keep], counts[keep]
    if len(hits) < 2 or counts.sum() == 0:
        return float("nan")
    draws = rng.integers(0, len(hits), size=(BOOTSTRAP_RESAMPLES, len(hits)))
    accuracies = 100.0 * hits[draws].sum(axis=1) / np.maximum(counts[draws].sum(axis=1), 1e-9)
    low, high = np.percentile(accuracies, [2.5, 97.5])
    return float((high - low) / 2.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=_REPO_ROOT / "datasets" / "gr00t-apple-grip"
    )
    parser.add_argument(
        "--holdout", type=Path, default=_REPO_ROOT / "configs" / "splits" / "pr03_holdout_150.txt"
    )
    parser.add_argument(
        "--restrict",
        type=Path,
        default=None,
        help="score only these episodes (must be a subset of --holdout); gate 1 uses the T-18 40",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="also write the JSON report here")
    args = parser.parse_args(argv)

    from wam.data.episode import list_episodes
    from wam.evaluation import load_episode_ids

    present = [p.name for p in list_episodes(args.dataset)]
    holdout_ids = load_episode_ids(args.holdout)
    missing = sorted(holdout_ids - set(present))
    if missing:
        raise SystemExit(f"--holdout lists {len(missing)} episode(s) absent: {missing[:5]}")
    score_ids = holdout_ids
    if args.restrict is not None:
        score_ids = load_episode_ids(args.restrict)
        if not score_ids <= holdout_ids:
            raise SystemExit(
                "--restrict is not a subset of --holdout: scoring episodes the training pool "
                "contains would report a training score"
            )

    train = [
        ep
        for ep in (load_episode(args.dataset, name) for name in present if name not in holdout_ids)
        if ep is not None
    ]
    score = [
        ep
        for ep in (load_episode(args.dataset, name) for name in sorted(score_ids))
        if ep is not None
    ]
    print(f"train {len(train)} episodes / {sum(len(e.targets) for e in train)} chunks")
    print(f"score {len(score)} episodes / {sum(len(e.targets) for e in score)} chunks")

    preds = build_predictions(train, score, args.seed)
    targets = _stack(score, "targets")
    report: dict[str, object] = {
        "dataset": str(args.dataset),
        "holdout": str(args.holdout),
        "restrict": str(args.restrict) if args.restrict else None,
        "train_episodes": len(train),
        "score_episodes": len(score),
        "definitions": {},
    }

    for definition in FLIP_DEFINITIONS:
        flip = np.concatenate([ep.flips[definition] for ep in score])
        post, pre, window = postflip_mask(flip), preflip_mask(flip), window_mask(flip)
        rows = {
            name: {
                "postflip": _postflip_accuracy(pred, targets, post),
                "preflip": _postflip_accuracy(pred, targets, pre),
                "k_to_k3": _postflip_accuracy(pred, targets, window),
            }
            for name, pred in preds.items()
        }
        per_episode = _per_episode_counts(score, preds["ceiling"], definition)
        best_other = max(row["postflip"] for name, row in rows.items() if name != "ceiling")
        entry = {
            "transition_chunks": int((flip >= 0).sum()),
            "n_postflip": int(post.sum()),
            "episodes_with_transition": int(sum(1 for c in per_episode[1] if c > 0)),
            "ci_halfwidth": bootstrap_halfwidth(*per_episode, seed=args.seed),
            # A "ceiling" a zero-parameter rule beats is not a ceiling. Reported rather than
            # asserted, because the honest response is to strengthen the family and re-run, not
            # to score a model against a bar that is known to be too low.
            "ceiling_dominates": bool(rows["ceiling"]["postflip"] >= best_other),
            "best_non_ceiling": best_other,
            "predictors": rows,
        }
        report["definitions"][definition] = entry  # type: ignore[index]
        _print_definition(definition, entry)

    _print_gates(report, restricted=args.restrict is not None)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.out}")
    return 0


def _per_episode_counts(
    score: list[Episode], ceiling: np.ndarray, definition: str
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-episode (correct, total) post-flip step counts for the ceiling — the bootstrap's unit."""
    correct, total, offset = [], [], 0
    for ep in score:
        n = len(ep.targets)
        mask = postflip_mask(ep.flips[definition])
        hit = (ceiling[offset : offset + n] >= 0.5) == (ep.targets >= 0.5)
        correct.append(float(hit[mask].sum()))
        total.append(float(mask.sum()))
        offset += n
    return correct, total


def _print_definition(definition: str, entry: dict) -> None:
    print(f"\n## {definition}")
    print(
        f"transition chunks {entry['transition_chunks']} · post-flip steps {entry['n_postflip']} "
        f"· episodes with a transition {entry['episodes_with_transition']} "
        f"· ceiling CI half-width {entry['ci_halfwidth']:.2f} pts"
    )
    print(f"{'predictor':18s} {'post-flip':>10s} {'pre-flip':>10s} {'k..k+3':>10s}")
    for name, row in entry["predictors"].items():
        print(f"{name:18s} {row['postflip']:9.2f}% {row['preflip']:9.2f}% {row['k_to_k3']:9.2f}%")
    if not entry["ceiling_dominates"]:
        print(
            f"  WARNING: the ceiling ({entry['predictors']['ceiling']['postflip']:.2f}%) is beaten "
            f"by another blind predictor ({entry['best_non_ceiling']:.2f}%). It is not a ceiling "
            "on this holdout, and no model may be scored against it until the family is fixed."
        )


def _print_gates(report: dict, *, restricted: bool) -> None:
    primary = report["definitions"][PRIMARY_DEFINITION]  # type: ignore[index]
    if restricted:
        print("\n## GATE 1 — archive gate (PR-03), scored on the T-18 40")
        worst_ok = True
        for name, (archived, tolerance) in ARCHIVED.items():
            value = primary["predictors"][name]["postflip"]
            delta = value - archived
            ok = abs(delta) <= tolerance
            worst_ok &= ok
            print(
                f"  {name:18s} {value:6.2f}% vs archived {archived:6.2f}% "
                f"(Δ {delta:+.2f}, tol ±{tolerance:.1f}) {'PASS' if ok else 'MISS'}"
            )
        print(f"  => {'PASS' if worst_ok else 'MISS — see the limitation note in this file'}")
        return

    print("\n## GATE 2 — power gate (PR-03), scored on the full holdout")
    n_postflip = primary["n_postflip"]
    halfwidth = primary["ci_halfwidth"]
    n_ok = n_postflip >= N_POSTFLIP_MIN
    ci_ok = halfwidth <= CI_HALFWIDTH_MAX
    print(f"  n_postflip    {n_postflip:6d}  >= {N_POSTFLIP_MIN}   {'PASS' if n_ok else 'FAIL'}")
    print(
        f"  ci_halfwidth  {halfwidth:6.2f}  <= {CI_HALFWIDTH_MAX}   {'PASS' if ci_ok else 'FAIL'}"
    )
    print(f"  minimum detectable effect ~{2 * halfwidth:.1f} points")
    print(
        f"  => {'PASS — the refit is submittable' if n_ok and ci_ok else 'FAIL — PR-03 says do not submit the refit'}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
