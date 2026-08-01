#!/usr/bin/env python3
"""Why the flow readout cannot contract noise: the velocity head's gain is FLAT in t.

``ActionVelocityHead`` (``src/wam/training/joint.py``) receives the flow timestep as ONE RAW
SCALAR concatenated beside ``latent_dim`` latent columns and ``feature_dim`` feature columns —
``self.mlp(cat([z_t, feats, t_col], dim=-1))``, nothing else. Under WAM's rectified-flow
convention (``src/wam/training/losses.py``: ``x_t = (1-t) x0 + t x1`` with ``x0`` the noise and
``x1`` the clean latent, target ``v = x1 - x0``), the field a sampler needs at ``z_t`` is

    v* = (x1 - z_t) / (1 - t)

— an error-correcting term whose GAIN on that error is ``1/(1-t)``, running 1 -> 32 over the
32-step grid ``JointWorldActionModel.sample_action_chunk`` deploys. Concatenation is ADDITIVE
conditioning: a t column can SHIFT the output, it cannot MULTIPLY it by a function of t except
through whatever curvature the MLP's nonlinearities happen to lend it. So the head settles on a
t-flat gain, and a constant-gain linear field contracts the residual by a fixed factor per step
and cannot reach zero at any step count — which is the mechanism behind the T-30 sweep converging
onto a chunk that is still mostly noise (``joint.py``, ``sample_action_chunk``).

That is a claim about a specific 3105 -> 256 -> 32 MLP, so this script measures it instead of
arguing it: on any checkpoint, on CPU, in seconds, with no training and without constructing a
backbone — building the model would materialize a frozen multi-GB Wan tower to run one MLP.

    scripts/probe_velocity_head.py \\
        --checkpoint runs/t16-lora-seed0/checkpoints/step-020000

Three statistics, all read off the head alone:

  first-layer blocks   Frobenius norm of ``mlp.0.weight``'s ``[z | feats | t]`` column blocks.
                       The BUDGET: how much first-layer weight the head spends on each input.
                       Per column as well as per block, because t is one column against
                       thousands and a raw block norm reads as "t is ignored" either way.

  S_t                  t-flatness, ``max_t ||v(t) - vbar|| / ||vbar||`` with ``(z, features)``
                       held fixed and only t swept over the deployed grid — the whole variation
                       the timestep input buys. Printed next to what the CORRECT field scores on
                       the same grid, which is an exact number rather than a simulation: with
                       ``(z, features)`` fixed, ``v* = c/(1-t)`` for a FIXED vector ``c``, so its
                       direction never moves and ``S_t`` collapses to the spread of the scalar
                       gain (see :func:`ideal_t_flatness`).

  ghat                 ``mean -diag(dv/dz)``, the per-step contraction the sampler actually gets,
                       against the ``1/(1-t)`` it should get. The Jacobian is taken through the
                       head at a single chunk step, which loses nothing: the head is applied
                       per step, so one step IS the map. Off-diagonal Frobenius mass is reported
                       beside it — a diagonal-mean gain is only a summary of the field if the map
                       is roughly diagonal, and saying so is the guard on the summary.

Reference values, measured 2026-08-01 on ``runs/t16-lora-seed0/checkpoints/step-020000`` at this
script's defaults (32-step grid, 64 samples, seed 0):

  first-layer blocks   latent 23.284 (4.116/col) | feats 51.049 (0.921/col) | t 1.681 (1.681/col)
  S_t                  0.0388 at feature scale 0.1 falling to 0.0093 at scale 5.0 — against 6.885
                       for the correct field on the same grid, i.e. ~180x too flat where the
                       head's t-dependence is at its largest
  ghat                 4.877 / 3.720 / 2.539 / 1.459 / 0.615 over feature scales 0.1 .. 5.0, and
                       CONSTANT in t to three significant figures at every one of them (4.877 at
                       t=0, 4.881 at t=0.5, 4.879 at t=0.969 — over a span where ``1/(1-t)`` has
                       gone from 1 to 32)
  offdiag/diag         0.38 .. 0.51, so the diagonal mean is a fair summary of the map

The Monte-Carlo error on ``ghat`` is a percent or so at these settings, not a rounding artifact:
``--samples 32`` moves the feature-scale-0.1 column to 4.859 / 4.858 / 4.852. It moves the
flatness reading by nothing at all, which is the point — the gap being reported is 32x.

Re-run it on any future checkpoint for free; the numbers are the acceptance criterion for a
timestep-embedding change to the head, not just a description of this one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from wam.training._utils import CHECKPOINT_CONFIG_KEY
from wam.training.joint import ActionVelocityHead, JointTrainingConfig

MODEL_FILENAME = "model.safetensors"

VELOCITY_PREFIX = "velocity_head."
"""The state-dict prefix the head is checkpointed under, frozen by every archived run."""

DEFAULT_FEATURE_SCALES = (0.1, 0.5, 1.0, 2.0, 5.0)
"""Sweeps the feature input from far below to far above unit scale.

The head is not linear, so its gain is a function of where in feature space it is probed, and a
single scale would leave "flat in t" open to the objection that it was measured in a dead corner.
The pooled features a real observation produces are not standard normal either — this brackets
them rather than pretending to reproduce them.
"""

DEFAULT_STEPS = 32
"""Grid size of the deployed flow readout (``--flow-steps 32``, the T-30 sweep's setting)."""

DEFAULT_SAMPLES = 64


def resolve_model_path(checkpoint: Path) -> Path:
    """``--checkpoint`` (a file, a step dir, or a run dir) -> the weights file."""
    path = checkpoint / MODEL_FILENAME if checkpoint.is_dir() else checkpoint
    if not path.is_file():
        raise SystemExit(f"{path} missing — not a restorable checkpoint")
    return path


def load_velocity_head(model_path: Path) -> tuple[ActionVelocityHead, JointTrainingConfig]:
    """``(head, config)`` restored from a checkpoint — the head and NOTHING else.

    The head's shape is not guessable from the tensors alone (``velocity_hidden_dims`` is, but
    the latent/feature split of the first layer's columns is not), so it comes from the config
    the checkpoint carries. That config travels inside the file and cannot go missing the way a
    sidecar or an operator's memory can.

    ``strict=True`` is the load, and it is not a stylistic preference. ``load_state_dict`` with
    ``strict=False`` leaves every unmatched tensor at its RANDOM INIT, and a randomly-initialized
    MLP has a t-flat gain too — so a partial load would produce exactly the finding this script
    exists to report, at full confidence, from a checkpoint that never trained this module.
    """
    from safetensors import safe_open

    with safe_open(str(model_path), framework="pt") as handle:
        metadata = handle.metadata() or {}
        if CHECKPOINT_CONFIG_KEY not in metadata:
            raise SystemExit(
                f"{model_path}: no {CHECKPOINT_CONFIG_KEY!r} in the safetensors metadata, so the "
                "head's latent_dim / feature_dim / velocity_hidden_dims are unknown. Every "
                "checkpoint written by wam.training._utils.save_checkpoint carries the config "
                "that produced it; this file was not written by WAM, or was rewritten without it."
            )
        # Only the head's own tensors are pulled through. safe_open reads lazily, so probing a
        # 330 MB T-16 checkpoint touches the ~850 kB that belong to the head; load_file would
        # deserialize the LoRA and every other branch to run one MLP.
        names = list(handle.keys())
        state = {
            name[len(VELOCITY_PREFIX) :]: handle.get_tensor(name)
            for name in names
            if name.startswith(VELOCITY_PREFIX)
        }
        if not state:
            raise SystemExit(
                f"{model_path}: no {VELOCITY_PREFIX}* tensors. Only a JOINT checkpoint has a "
                "velocity head — an action-only run (T-13) never trains a flow branch, and there "
                "is no timestep conditioning in it to probe."
            )
        config = JointTrainingConfig.model_validate(json.loads(metadata[CHECKPOINT_CONFIG_KEY]))

    head = ActionVelocityHead(
        config.action_encoder.latent_dim,
        config.backbone.feature_dim,
        tuple(config.velocity_hidden_dims),
    )
    try:
        head.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise SystemExit(
            f"{model_path}: the {VELOCITY_PREFIX}* tensors do not fit the head its own config "
            f"describes ({config.action_encoder.latent_dim} latent + "
            f"{config.backbone.feature_dim} features + 1, hidden "
            f"{tuple(config.velocity_hidden_dims)}):\n  {error}\n"
            "Refusing to load the subset non-strictly: the missing tensors would stay at their "
            "random init, and a random MLP measures as t-flat too — the finding would be "
            "indistinguishable from a real one."
        ) from error
    head.eval()
    return head, config


def sampler_grid(steps: int, t0: float = 0.0) -> tuple[float, ...]:
    """The t values the deployed sampler evaluates the head at.

    ``sample_action_chunk``'s grid verbatim: ``{t0, t0 + dt, ..., 1 - dt}`` for
    ``dt = (1 - t0)/steps``. It never reaches 1.0 because training draws t from ``torch.rand``,
    whose support is [0, 1) — t=1.0 is a timestep the head has never once been evaluated at, and
    probing there would measure extrapolation rather than the deployed field.
    """
    if steps < 1:
        raise SystemExit(f"--steps must be >= 1, got {steps}")
    dt = (1.0 - t0) / steps
    return tuple(t0 + k * dt for k in range(steps))


def ideal_t_flatness(grid: tuple[float, ...]) -> float:
    """``S_t`` of the analytically correct field on ``grid`` — exact, not simulated.

    :func:`t_flatness` holds ``(z, features)`` fixed and sweeps only t. Under that sweep the
    correct field ``v* = (x1 - z_t)/(1 - t)`` is ``c/(1 - t)`` for a FIXED vector ``c``: its
    direction never moves, so ``||v(t) - vbar|| / ||vbar||`` collapses to
    ``|1/(1-t) - m| / m`` with ``m`` the mean gain over the grid. That makes the reference a
    property of the grid alone — the same number for every clean latent, every observation and
    every checkpoint — which is what makes it worth printing next to a measurement.
    """
    gains = [1.0 / (1.0 - t) for t in grid]
    mean = sum(gains) / len(gains)
    return max(abs(gain - mean) for gain in gains) / mean


def first_layer_blocks(
    head: nn.Module, latent_dim: int, feature_dim: int
) -> dict[str, tuple[int, float]]:
    """``{block: (columns, Frobenius norm)}`` over ``mlp.0.weight``'s ``[z | feats | t]`` blocks.

    The budget the head allocates to each input group, before any behaviour is measured. Column
    counts travel with the norms because they are what makes the comparison honest: t is ONE
    column against thousands of feature columns, so a small t block is the expected shape of the
    layer and not yet evidence of anything.
    """
    weight = head.mlp[0].weight.detach()
    expected = latent_dim + feature_dim + 1
    if weight.shape[1] != expected:
        raise SystemExit(
            f"first layer takes {weight.shape[1]} columns, but the config describes "
            f"{latent_dim} + {feature_dim} + 1 = {expected}"
        )
    return {
        "latent": (latent_dim, float(weight[:, :latent_dim].norm())),
        "feats": (feature_dim, float(weight[:, latent_dim : latent_dim + feature_dim].norm())),
        "t": (1, float(weight[:, -1].norm())),
    }


@torch.no_grad()
def t_flatness(
    head: nn.Module,
    latent_dim: int,
    feature_dim: int,
    grid: tuple[float, ...],
    *,
    chunk_steps: int,
    feature_scale: float,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """``(mean S_t, mean ||vbar||)`` — how much the output moves when ONLY t moves.

    Per sample: draw one ``(z, features)`` pair, hold it, evaluate the head at every t on the
    grid, and take ``max_t ||v(t) - vbar|| / ||vbar||``. Normalizing by the mean's own norm is
    what makes the statistic comparable across feature scales and against
    :func:`ideal_t_flatness`; the un-normalized ``||vbar||`` is returned beside it so a small
    ratio produced by a large denominator is visible rather than inferred.

    The mean and the residuals are taken in float64, for the reason ``wam.evaluation`` reduces in
    float64: this statistic is a difference of nearly-equal numbers, and on the head it exists to
    measure they are equal to five significant figures. Accumulating 32 float32 velocities in
    float32 leaves ~1e-7 of rounding in ``vbar`` alone — which is not a small error next to a
    measured S_t of 0.009, it is a floor the statistic could never go below. In float64 a head
    that genuinely cannot see t scores exactly 0.0, and the calibration point is a fact about the
    head rather than about the accumulator.
    """
    generator = torch.Generator().manual_seed(seed)
    scores: list[float] = []
    norms: list[float] = []
    for _ in range(samples):
        latent = torch.randn(1, chunk_steps, latent_dim, generator=generator)
        features = torch.randn(1, feature_dim, generator=generator) * feature_scale
        velocities = torch.stack(
            [head(latent, features, torch.full((1,), t)) for t in grid]
        ).double()
        mean = velocities.mean(dim=0)
        norm = float(mean.norm())
        if norm == 0.0:
            # A head whose t-average is exactly zero has no scale to measure the spread against.
            return float("nan"), 0.0
        scores.append(max(float((velocity - mean).norm()) / norm for velocity in velocities))
        norms.append(norm)
    return sum(scores) / len(scores), sum(norms) / len(norms)


def latent_gain(
    head: nn.Module,
    latent_dim: int,
    feature_dim: int,
    *,
    t: float,
    feature_scale: float,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """``(ghat, offdiag/diag)`` at one timestep: the contraction ``-dv/dz`` the sampler gets.

    ``ghat = mean -diag(dv/dz)``. The sign is the one that makes it readable as a gain: the
    sampler steps ``z <- z + v dt``, so a field that pulls the latent toward the data has
    ``dv/dz`` NEGATIVE, and ``ghat > 0`` is the rate at which error is removed. The correct
    field's is exactly ``1/(1-t)``.

    Taken at a single chunk step, which loses nothing: ``ActionVelocityHead`` applies one MLP per
    step with no cross-step term, so one step is the entire map. The Jacobian is the exact
    derivative of the float32 graph (autograd, not finite differences), so the only sampling
    error here is over ``(z, features)``.

    The off-diagonal Frobenius ratio is the guard on the summary: a diagonal mean describes the
    field only if the map is roughly diagonal, and a head whose Jacobian is dominated by
    off-diagonal mixing would make ``ghat`` a number about nothing.
    """
    generator = torch.Generator().manual_seed(seed)
    diagonals: list[float] = []
    ratios: list[float] = []
    for _ in range(samples):
        features = torch.randn(1, feature_dim, generator=generator) * feature_scale
        timestep = torch.full((1,), t)

        def one_step(latent: Tensor, features: Tensor = features, timestep: Tensor = timestep):
            return head(latent.reshape(1, 1, latent_dim), features, timestep).reshape(latent_dim)

        latent = torch.randn(latent_dim, generator=generator)
        jacobian = torch.autograd.functional.jacobian(one_step, latent, vectorize=True)
        contraction = -jacobian
        diagonal = torch.diagonal(contraction)
        off_diagonal = contraction - torch.diag(diagonal)
        diagonals.append(float(diagonal.mean()))
        ratios.append(float(off_diagonal.norm()) / max(float(diagonal.norm()), 1e-12))
    return sum(diagonals) / len(diagonals), sum(ratios) / len(ratios)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=f"a run/step dir or the {MODEL_FILENAME} inside it",
    )
    parser.add_argument(
        "--feature-scale",
        type=float,
        action="append",
        default=None,
        help="std of the probe features; repeatable (default: "
        f"{' '.join(str(scale) for scale in DEFAULT_FEATURE_SCALES)})",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="probe draws per (statistic, feature scale, t) cell (default: %(default)s)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="sampler grid size — the t values the deployed readout uses (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=0, help="default: %(default)s")
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="also write the measurement as a machine-readable record",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.samples < 1:
        raise SystemExit(f"--samples must be >= 1, got {args.samples}")
    scales = tuple(args.feature_scale) if args.feature_scale else DEFAULT_FEATURE_SCALES

    model_path = resolve_model_path(args.checkpoint)
    head, config = load_velocity_head(model_path)
    latent_dim = config.action_encoder.latent_dim
    feature_dim = config.backbone.feature_dim
    chunk_steps = config.head.num_steps
    grid = sampler_grid(args.steps)
    # First, middle and last of the deployed grid. Three points are enough because the question
    # is not the shape of ghat(t) but whether it moves AT ALL against a reference that goes
    # 1 -> 1/(1-t_last) across the same span; duplicates collapse for a one- or two-step grid.
    probe_ts = tuple(sorted({grid[0], grid[len(grid) // 2], grid[-1]}))

    print(f"checkpoint  {model_path}")
    print(
        f"head        latent {latent_dim} | features {feature_dim} | hidden "
        f"{tuple(config.velocity_hidden_dims)} | in_dim {latent_dim + feature_dim + 1} | "
        f"chunk {chunk_steps} steps"
    )
    print(
        f"grid        {len(grid)} steps, t in [{grid[0]:g}, {grid[-1]:g}] — the correct field's "
        f"gain 1/(1-t) runs {1.0 / (1.0 - grid[0]):.2f} -> {1.0 / (1.0 - grid[-1]):.2f} across it"
    )
    print(f"probe       {args.samples} samples, seed {args.seed}, CPU")
    print()

    blocks = first_layer_blocks(head, latent_dim, feature_dim)
    print("first-layer weight blocks of mlp.0 — what the head spends on each input group")
    print(f"{'block':>8} {'columns':>9} {'Frobenius':>12} {'per column':>12}")
    for name, (columns, norm) in blocks.items():
        print(f"{name:>8} {columns:>9} {norm:>12.3f} {norm / columns**0.5:>12.4f}")
    print()

    print("t-flatness  S_t = max_t ||v(t) - vbar|| / ||vbar||, with (z, features) held fixed")
    print(f"{'feat_scale':>12} {'S_t':>12} {'||vbar||':>12}")
    flatness: list[dict[str, float]] = []
    for scale in scales:
        score, norm = t_flatness(
            head,
            latent_dim,
            feature_dim,
            grid,
            chunk_steps=chunk_steps,
            feature_scale=scale,
            samples=args.samples,
            seed=args.seed,
        )
        flatness.append({"feature_scale": scale, "s_t": score, "mean_norm": norm})
        print(f"{scale:>12g} {score:>12.5f} {norm:>12.4e}")
    ideal_flatness = ideal_t_flatness(grid)
    print(f"  the correct field scores S_t = {ideal_flatness:.3f} on this grid (exact, see")
    print("  ideal_t_flatness: every v* = c/(1-t) has it, whatever c is)")
    print()

    # The gain probe draws its own features rather than reusing the flatness stream: the two
    # statistics are independent measurements of one head, and sharing a draw would make them
    # correlated for no reason anyone reading the table would expect.
    print("latent gain  ghat = mean -diag(dv/dz) — the per-step contraction the sampler gets")
    print(f"{'feat_scale':>12} {'t':>9} {'ghat':>10} {'offdiag/diag':>14} {'ideal 1/(1-t)':>14}")
    gains: list[dict[str, float]] = []
    for scale in scales:
        for t in probe_ts:
            ghat, off_ratio = latent_gain(
                head,
                latent_dim,
                feature_dim,
                t=t,
                feature_scale=scale,
                samples=args.samples,
                seed=args.seed + 1,
            )
            gains.append(
                {
                    "feature_scale": scale,
                    "t": t,
                    "ghat": ghat,
                    "offdiag_over_diag": off_ratio,
                    "ideal_gain": 1.0 / (1.0 - t),
                }
            )
            print(
                f"{scale:>12g} {t:>9.5f} {ghat:>10.4f} {off_ratio:>14.4f} {1.0 / (1.0 - t):>14.3f}"
            )
    print()

    # The reading, in the terms the head is actually charged with: a MULTIPLICATIVE factor in t.
    needed = (1.0 / (1.0 - probe_ts[-1])) / (1.0 / (1.0 - probe_ts[0]))
    spans = []
    for scale in scales:
        row = [entry["ghat"] for entry in gains if entry["feature_scale"] == scale]
        low, high = min(row), max(row)
        if low > 0.0:
            spans.append(high / low)
    measured = f"{min(spans):.3f}x - {max(spans):.3f}x" if spans else "undefined (ghat crosses 0)"
    print(
        f"reading: across t in [{probe_ts[0]:g}, {probe_ts[-1]:g}] the measured gain moves "
        f"{measured} where the correct\n  field moves {needed:.1f}x. A gain that does not grow "
        "with t removes a fixed fraction of the residual\n  per step, so the sampler converges "
        "to a fixed noise floor instead of to the data — no step\n  count fixes that. It is a "
        "head/objective change (a timestep embedding, or a step index the\n  head can read), not "
        "a sampler one."
    )

    if args.json is not None:
        record: dict[str, Any] = {
            "checkpoint": str(model_path),
            "latent_dim": latent_dim,
            "feature_dim": feature_dim,
            "velocity_hidden_dims": list(config.velocity_hidden_dims),
            "chunk_steps": chunk_steps,
            "samples": args.samples,
            "seed": args.seed,
            "grid": {"steps": len(grid), "t_first": grid[0], "t_last": grid[-1]},
            "first_layer_blocks": {
                name: {"columns": columns, "frobenius": norm, "per_column": norm / columns**0.5}
                for name, (columns, norm) in blocks.items()
            },
            "t_flatness": flatness,
            "ideal_t_flatness": ideal_flatness,
            "gain": gains,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
