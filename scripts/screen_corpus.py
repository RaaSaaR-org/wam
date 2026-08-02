#!/usr/bin/env python3
"""Screen a demonstration corpus: does a BLIND baseline fail on it? (PR-04, T-34)

PR-02 formalised the one question worth asking of a candidate dataset, and it is not "does it
match our hands and cameras". It is whether a predictor with no vision at all — only
proprioception and the clock — can already do the task's action prediction. If it can, no
world-action result measured on that corpus can distinguish "the video branch works" from
"momentum works", and every negative this project has recorded is unfalsifiable on it.

That screen was applied to somebody else's dataset (HIW-500) from a script that was never
committed. This is the same screen as committed, runnable code, aimed at OUR OWN recordings
BEFORE we scale them — because the whole lesson of the GR00T corpus is that we discovered
M2 = 0.333 after building everything on it, not before.

THE THREE NUMBERS
-----------------
Fit on train episodes, scored on episode-disjoint holdout episodes, on the arm action chunks:

  M1  momentum share          (mse_zero - mse_constvel) / (mse_zero - mse_ceiling)
      The fraction of everything a blind model could achieve that a ZERO-PARAMETER rule
      already achieves. High M1 means the metric is mostly measuring inertia.

  M2  blind-unreachable share  mse_ceiling / mse_zero
      The share of target energy the best blind model CANNOT reach. This is the room a
      vision model has to prove itself in. Low M2 means there is nothing to win.

  M3  grasp liveness           debounced open/close transitions per episode
      Events per episode on the gripper channel. Zero means the corpus cannot say anything
      about grasping at all, whatever else it shows.

M1 and M2 are computed against a blind nonlinear CEILING (random Fourier features over lagged
q/dq/gripper plus the raw standardized block), not against a linear model: the claim "blind
cannot reach this" is only worth making against the strongest blind predictor we can fit.

VALIDATION
----------
``--expect gr00t`` scores against the archived values this screen was defined by
(``docs/preregistration/PR-02-RESULT.md``): M1 +0.660, M2 0.333, M3 2.01. Run it on
``datasets/gr00t-apple-grip`` with the committed 40-episode holdout and it must reproduce
them, or this implementation is not the screen PR-02 ran and its verdicts on new corpora mean
nothing. That check is the thing PR-03's gate 1 could not do, because the code it needed to
compare against had never been committed.

Usage
-----
    .venv/bin/python scripts/screen_corpus.py --dataset datasets/gr00t-apple-grip \\
        --holdout configs/splits/t18_holdout_episodes.txt --expect gr00t
    .venv/bin/python scripts/screen_corpus.py --dataset datasets/d1-pilot --out runs/pr04/pilot.json

Torch-free, CPU only, no GPU, no robot.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

#: Non-overlapping chunk length, matching the converter and every archived number.
CHUNK_STEPS = 16
#: Proprioception lags handed to the blind model, in frames. 8 lags over ~0.4 s at 30 Hz.
LAGS = (0, 1, 2, 3, 4, 6, 8, 12)
#: Random Fourier features for the blind ceiling, and the bandwidth/ridge grids searched.
N_FEATURES = 2048
GAMMA_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)  # multiples of 1/D, never absolute — see PR-03-RESULT
LAMBDAS = (1e-3, 1e-1, 1e1, 1e3)
SEED = 0

#: PR-04 gate thresholds. Fixed here, in code, before any pilot is recorded.
M1_MAX = 0.45
M2_MIN = 0.45
M3_MIN = 2.0

#: docs/preregistration/PR-02-RESULT.md, measured on the 402-episode GR00T corpus.
ARCHIVED = {"gr00t": {"m1": 0.660, "m2": 0.333, "m3": 2.01}}
#: Tolerances for --expect. M1/M2 are ratios of MSEs and reproduce tightly; M3 is a count.
EXPECT_TOL = {"m1": 0.02, "m2": 0.02, "m3": 0.05}


@dataclass
class Episode:
    episode_id: str
    features: np.ndarray  # (n_chunks, D) lagged blind state at each chunk's start frame
    targets: np.ndarray  # (n_chunks, 16 * 15) demonstrated arm deltas, flattened
    constvel: np.ndarray  # (n_chunks, 16 * 15) dq[start] * dt_s held across the chunk
    #: Debounced transitions PER GRIPPER CHANNEL. Which channel is the live one is a property
    #: of the corpus, not of this script — a rig whose moving hand is index 1 is a perfectly
    #: healthy rig, and hardcoding index 0 would report M3 = 0.00 for it and route PR-04 to
    #: verdict C ("the recording or conversion killed the channel"), which is the exact
    #: inversion of the failure G3 exists to catch.
    transitions: np.ndarray  # (gripper_dims,)


def _lagged(stream: np.ndarray, frame: int) -> np.ndarray:
    """``stream`` at ``frame`` minus each lag, clamped at the episode start (no future reads)."""
    return np.concatenate([stream[max(frame - lag, 0)] for lag in LAGS])


def load_episode(root: Path, episode_id: str) -> Episode | None:
    from wam.evaluation.gripper import debounced_transitions

    states = pq.read_table(
        root / episode_id / "states.parquet", columns=["q", "dq", "gripper_state"]
    ).to_pydict()
    q = np.asarray([np.asarray(v, dtype=np.float64) for v in states["q"]])
    dq = np.asarray([np.asarray(v, dtype=np.float64) for v in states["dq"]])
    grip = np.asarray([np.asarray(v, dtype=np.float64) for v in states["gripper_state"]])
    n_frames = len(q)

    actions = pq.read_table(
        root / episode_id / "actions.parquet",
        columns=["chunk_idx", "step_idx", "targets", "dt_s"],
    ).to_pydict()
    chunk_idx = np.asarray(actions["chunk_idx"], dtype=np.int64)
    step_idx = np.asarray(actions["step_idx"], dtype=np.int64)
    targets = np.asarray([np.asarray(v, dtype=np.float64) for v in actions["targets"]])
    dt_s = float(np.asarray(actions["dt_s"], dtype=np.float64)[0])

    stream = np.hstack([q, dq, grip])
    rows, feats, cvel = [], [], []
    for chunk in sorted(set(chunk_idx.tolist())):
        sel = chunk_idx == chunk
        if int(sel.sum()) < CHUNK_STEPS:
            continue  # a trailing partial chunk is not a 16-step chunk
        order = np.argsort(step_idx[sel])
        block = targets[sel][order][:CHUNK_STEPS]
        start = int(chunk) * CHUNK_STEPS
        if start + CHUNK_STEPS >= n_frames:
            continue
        rows.append(block.reshape(-1))
        feats.append(_lagged(stream, start))
        # Const-velocity: hold the measured joint velocity at the chunk start for all 16 steps.
        # Zero-parameter, strictly causal, and the rule PR-01 found beats the fine-tune.
        cvel.append(np.tile(dq[start] * dt_s, (CHUNK_STEPS, 1)).reshape(-1))
    if not rows:
        return None

    return Episode(
        episode_id=episode_id,
        features=np.asarray(feats),
        targets=np.asarray(rows),
        constvel=np.asarray(cvel),
        transitions=np.asarray(
            [debounced_transitions(grip[:, c]) for c in range(grip.shape[1])], dtype=np.int64
        ),
    )


# ------------------------------------------------------------------------------- blind predictors


def _ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    gram = x.T @ x + lam * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ y)


def _standardize(fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = fit.mean(axis=0)
    scale = fit.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return mean, scale


def _rff(x: np.ndarray, weights: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0 / weights.shape[1]) * np.cos(x @ weights + offset)


def _design(z: np.ndarray, weights: np.ndarray, offset: np.ndarray) -> np.ndarray:
    # The raw standardized block is appended so the ceiling can never be WORSE than the linear
    # model it contains — a "ceiling" a simpler predictor beats is not a ceiling (PR-03-RESULT).
    return np.hstack([_rff(z, weights, offset), z, np.ones((len(z), 1))])


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def fit_ceiling(
    train: list[Episode], holdout: list[Episode], seed: int = SEED
) -> tuple[float, dict]:
    """Best blind nonlinear predictor of the arm chunk, selected on an episode-disjoint inner
    split of the TRAINING episodes only — the holdout is never used to choose anything."""
    rng = np.random.default_rng(seed)
    cut = max(1, int(0.8 * len(train)))
    order = rng.permutation(len(train))
    inner_fit = [train[i] for i in order[:cut]]
    inner_val = [train[i] for i in order[cut:]] or [train[i] for i in order[:1]]

    x_fit = np.vstack([e.features for e in inner_fit])
    y_fit = np.vstack([e.targets for e in inner_fit])
    x_val = np.vstack([e.features for e in inner_val])
    y_val = np.vstack([e.targets for e in inner_val])
    mean, scale = _standardize(x_fit)
    z_fit, z_val = (x_fit - mean) / scale, (x_val - mean) / scale

    best = (np.inf, None)
    for gamma_scale in GAMMA_SCALES:
        gamma = gamma_scale / z_fit.shape[1]
        w = rng.normal(0.0, np.sqrt(2.0 * gamma), (z_fit.shape[1], N_FEATURES))
        b = rng.uniform(0.0, 2.0 * np.pi, N_FEATURES)
        f_fit, f_val = _design(z_fit, w, b), _design(z_val, w, b)
        for lam in LAMBDAS:
            coef = _ridge(f_fit, y_fit, lam)
            score = _mse(f_val @ coef, y_val)
            if score < best[0]:
                best = (score, (gamma_scale, lam, w, b))

    gamma_scale, lam, w, b = best[1]  # type: ignore[misc]
    # Refit on ALL training episodes at the selected hyperparameters, then score the holdout.
    x_all = np.vstack([e.features for e in train])
    y_all = np.vstack([e.targets for e in train])
    mean, scale = _standardize(x_all)
    coef = _ridge(_design((x_all - mean) / scale, w, b), y_all, lam)
    x_out = np.vstack([e.features for e in holdout])
    y_out = np.vstack([e.targets for e in holdout])
    pred = _design((x_out - mean) / scale, w, b) @ coef
    return _mse(pred, y_out), {"gamma_scale": gamma_scale, "lambda": lam, "inner_mse": best[0]}


def screen(train: list[Episode], holdout: list[Episode], seed: int = SEED) -> dict:
    y = np.vstack([e.targets for e in holdout])
    cv = np.vstack([e.constvel for e in holdout])
    mse_zero = _mse(np.zeros_like(y), y)
    mse_constvel = _mse(cv, y)
    mse_ceiling, selection = fit_ceiling(train, holdout, seed=seed)

    span = mse_zero - mse_ceiling
    m1 = (mse_zero - mse_constvel) / span if span > 0 else float("nan")
    m2 = mse_ceiling / mse_zero if mse_zero > 0 else float("nan")
    # M3 is scored on the corpus's OWN live channel: the one with the most transitions summed
    # over every episode. Ties and all-dead corpora fall to channel 0 and report 0.00, which is
    # the honest answer — there is no live channel to find.
    per_channel = np.sum([e.transitions for e in holdout + train], axis=0)
    active_hand = int(np.argmax(per_channel))
    m3 = float(np.mean([e.transitions[active_hand] for e in holdout + train]))

    # A ceiling a zero-parameter rule beats is not a ceiling, and every M1/M2 read off it is
    # meaningless. PR-03 hit exactly this and only caught it because the number was absurd.
    dominates = bool(mse_ceiling <= mse_constvel and mse_ceiling <= mse_zero)
    return {
        "m1_momentum_share": m1,
        "m2_blind_unreachable": m2,
        "m3_transitions_per_episode": m3,
        "m3_active_hand": active_hand,
        "m3_transitions_by_hand": [int(v) for v in per_channel],
        "ceiling_dominates": dominates,
        "mse": {
            "zero_delta": mse_zero,
            "const_velocity": mse_constvel,
            "blind_ceiling": mse_ceiling,
        },
        "selection": selection,
        "episodes": {"train": len(train), "holdout": len(holdout)},
        "chunks": {"train": int(sum(len(e.targets) for e in train)), "holdout": len(y)},
        "gates": {
            "m1_max": M1_MAX,
            "m2_min": M2_MIN,
            "m3_min": M3_MIN,
            "m1_pass": bool(m1 <= M1_MAX),
            "m2_pass": bool(m2 >= M2_MIN),
            "m3_pass": bool(m3 >= M3_MIN),
        },
    }


# ------------------------------------------------------------------------------------------- cli


def _json_safe(value: object) -> object:
    """Recursively replace non-finite floats with None, so the artifact is valid JSON."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def read_split(path: Path) -> list[str]:
    ids = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not ids:
        raise SystemExit(f"{path}: no episode ids")
    return ids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--holdout", type=Path, help="split file; default is a seeded 20% split")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--expect", choices=sorted(ARCHIVED), help="validate against archived values")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    all_ids = sorted(p.name for p in args.dataset.iterdir() if (p / "states.parquet").is_file())
    if not all_ids:
        raise SystemExit(f"{args.dataset}: no episodes")
    if args.holdout:
        held = read_split(args.holdout)
        missing = sorted(set(held) - set(all_ids))
        if missing:
            raise SystemExit(f"{args.holdout}: {len(missing)} episodes not in the dataset")
        held_set = set(held)
    else:
        order = np.random.default_rng(args.seed).permutation(len(all_ids))
        n_held = max(1, len(all_ids) // 5)
        held_set = {all_ids[i] for i in order[:n_held]}

    train, holdout = [], []
    for episode_id in all_ids:
        episode = load_episode(args.dataset, episode_id)
        if episode is None:
            continue
        (holdout if episode_id in held_set else train).append(episode)
    if not train or not holdout:
        raise SystemExit("need at least one train and one holdout episode with full chunks")

    report = screen(train, holdout, seed=args.seed)
    report["dataset"] = str(args.dataset)
    report["holdout_file"] = str(args.holdout) if args.holdout else None

    print(f"dataset: {args.dataset}")
    print(f"episodes: {len(train)} train / {len(holdout)} holdout")
    m = report["mse"]
    print(
        f"  mse   zero-delta {m['zero_delta']:.6e}  const-velocity {m['const_velocity']:.6e}  "
        f"blind ceiling {m['blind_ceiling']:.6e}"
    )
    gates = report["gates"]
    for key, label, value, bar, ok in (
        (
            "m1",
            "M1 momentum share    ",
            report["m1_momentum_share"],
            f"<= {M1_MAX}",
            gates["m1_pass"],
        ),
        (
            "m2",
            "M2 blind-unreachable ",
            report["m2_blind_unreachable"],
            f">= {M2_MIN}",
            gates["m2_pass"],
        ),
        (
            "m3",
            "M3 transitions/episode",
            report["m3_transitions_per_episode"],
            f">= {M3_MIN}",
            gates["m3_pass"],
        ),
    ):
        del key
        print(f"  {label} {value:8.4f}   bar {bar:>8}   {'PASS' if ok else 'FAIL'}")
    print(
        f"  (M3 scored on gripper channel {report['m3_active_hand']}; "
        f"transitions by channel {report['m3_transitions_by_hand']})"
    )
    if not report["ceiling_dominates"]:
        print(
            "  G4 FAIL: the blind ceiling is beaten by a zero-parameter rule, so M1 and M2 above\n"
            "  are VOID (M1's denominator can even be negative). PR-04 verdict E: refit the\n"
            "  ceiling; this says nothing about the corpus."
        )

    status = 0
    if args.expect:
        want = ARCHIVED[args.expect]
        print(f"\n--expect {args.expect}: reproduce the archived screen")
        for key, label in (
            ("m1", "m1_momentum_share"),
            ("m2", "m2_blind_unreachable"),
            ("m3", "m3_transitions_per_episode"),
        ):
            got, ref, tol = report[label], want[key], EXPECT_TOL[key]
            ok = abs(got - ref) <= tol
            status |= 0 if ok else 1
            print(
                f"  {key.upper()}  measured {got:8.4f}  archived {ref:8.4f}  "
                f"delta {got - ref:+.4f}  tol +-{tol}  {'OK' if ok else 'MISS'}"
            )
        report["expect"] = {"reference": args.expect, "reproduced": status == 0}

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # json.dumps emits a bare NaN token for a non-finite float, which Python and jq accept
        # but every standards-conformant reader rejects outright. A collapsed ceiling is exactly
        # when M1 goes NaN, i.e. the artifact most in need of inspection would be the one that
        # cannot be parsed. null instead: absent, which is what it means.
        args.out.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.out}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
