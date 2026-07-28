"""Training losses (T-16/T-17, PRD §10.4, FR-03).

All losses are plain tensor functions: torch in, scalar tensor out, no module state, fully
differentiable, deterministic. Reductions are means over the contributing elements so loss
magnitudes are comparable across batch sizes and chunk lengths.

Loss inventory (PRD §10.4):
- video loss:      ``video_flow_loss`` — velocity MSE in the video latent space.
- action loss:     ``action_flow_matching_loss`` (flow target) or ``action_regression_loss``
                   (L1/L2 on chunk targets, used by the action-only baseline, T-13).
- alignment loss:  ``alignment_loss`` — COSINE variant (documented below), not InfoNCE.
- smoothness:      ``smoothness_loss`` — second-difference penalty on action chunks.
- limit penalty:   ``limit_penalty`` — soft squared hinge outside ±limit in normalized space.

Rectified-flow convention (``make_flow_targets``): with noise ``x0``, data ``x1`` and time
``t ∈ [0, 1]``: ``x_t = (1 - t) * x0 + t * x1`` and the velocity target is the constant
``v = x1 - x0`` (so ``x1 = x_t + (1 - t) * v``).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

__all__ = [
    "action_flow_matching_loss",
    "action_regression_loss",
    "alignment_loss",
    "limit_penalty",
    "make_flow_targets",
    "smoothness_loss",
    "video_flow_loss",
]


def _broadcast_t(t: Tensor | float, ndim: int, batch: int) -> Tensor:
    """Timestep scalar or [B] vector -> [B, 1, ..., 1] tensor broadcastable to ``ndim`` dims."""
    t_vec = torch.as_tensor(t, dtype=torch.float32)
    if t_vec.ndim == 0:
        t_vec = t_vec.reshape(1).expand(batch)
    if t_vec.ndim != 1 or t_vec.shape[0] != batch:
        raise ValueError(
            f"t must be a scalar or [B={batch}] vector, got shape {tuple(t_vec.shape)}"
        )
    return t_vec.reshape(batch, *([1] * (ndim - 1)))


def make_flow_targets(x0: Tensor, x1: Tensor, t: Tensor | float) -> tuple[Tensor, Tensor]:
    """Rectified-flow interpolation: ``(x_t, velocity_target)``.

    ``x0`` is the noise sample, ``x1`` the clean data, both ``[B, ...]`` with identical shapes;
    ``t`` is a scalar or ``[B]`` vector in [0, 1]. Returns ``x_t = (1 - t) x0 + t x1`` and the
    velocity target ``v = x1 - x0`` (shape of ``x1``).
    """
    if x0.shape != x1.shape:
        raise ValueError(f"x0/x1 shape mismatch: {tuple(x0.shape)} vs {tuple(x1.shape)}")
    t_b = _broadcast_t(t, x1.ndim, x1.shape[0]).to(x1.device)
    x_t = (1.0 - t_b) * x0 + t_b * x1
    return x_t, x1 - x0


def _masked_mean(per_element: Tensor, mask: Tensor | None) -> Tensor:
    """Mean of ``per_element``; with ``mask`` (bool/0-1, broadcastable) only over valid entries."""
    if mask is None:
        return per_element.mean()
    weights = mask.to(dtype=per_element.dtype)
    while weights.ndim < per_element.ndim:
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(per_element)
    denom = weights.sum()
    if denom.item() == 0:
        return per_element.sum() * 0.0  # keeps graph + dtype, loss contribution is zero
    return (per_element * weights).sum() / denom


def action_flow_matching_loss(
    model_velocity: Tensor, target_velocity: Tensor, mask: Tensor | None = None
) -> Tensor:
    """Flow-matching action loss: MSE between predicted and target velocity (PRD §10.4).

    Shapes ``[B, T, L]`` (any trailing layout works); optional ``mask`` ``[B]``/``[B, T]``
    (bool or 0-1) excludes padded or invalid steps from the mean.
    """
    if model_velocity.shape != target_velocity.shape:
        raise ValueError(
            f"velocity shape mismatch: {tuple(model_velocity.shape)} vs "
            f"{tuple(target_velocity.shape)}"
        )
    return _masked_mean((model_velocity - target_velocity).pow(2), mask)


def action_regression_loss(
    pred: Tensor,
    target: Tensor,
    kind: Literal["l1", "l2"] = "l2",
    mask: Tensor | None = None,
) -> Tensor:
    """Direct regression on chunk targets — the action-only baseline objective (T-13).

    ``kind='l2'`` is mean squared error, ``kind='l1'`` mean absolute error. Optional ``mask``
    as in :func:`action_flow_matching_loss`.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    diff = pred - target
    if kind == "l2":
        per_element = diff.pow(2)
    elif kind == "l1":
        per_element = diff.abs()
    else:
        raise ValueError(f"kind must be 'l1' or 'l2', got {kind!r}")
    return _masked_mean(per_element, mask)


def video_flow_loss(
    velocity_pred: Tensor, velocity_target: Tensor, mask: Tensor | None = None
) -> Tensor:
    """Video branch loss: velocity MSE in the backbone's video latent space (PRD §10.4).

    The loss lives in WHATEVER space ``FlowBackbone.encode_video`` returns: pixels in [0, 1]
    for the tiny identity VAE (``[B, F, H, W, 3]``), VAE latents for a real backbone
    (``[B, C, F', H', W']``). Magnitudes are therefore NOT comparable across backbones — a
    ``weights.video`` tuned against tiny means nothing against Wan.

    With a real VAE the latents MUST be mean/std-normalized before the flow target is built.
    Raw Wan latents carry per-channel scales of order 1e1, so the squared velocity error is
    two-plus orders of magnitude above the action terms and the video branch swamps action
    learning outright (R-07) — the failure looks like "action losses plateau", not like a
    video bug.

    Optional ``mask`` (bool or 0-1, broadcastable over the leading dims) restricts the mean to
    valid entries — e.g. dropping the conditioning-frame latents that are copied in, not
    predicted. ``mask=None`` is a plain mean over all elements.
    """
    if velocity_pred.shape != velocity_target.shape:
        raise ValueError(
            f"velocity shape mismatch: {tuple(velocity_pred.shape)} vs "
            f"{tuple(velocity_target.shape)}"
        )
    return _masked_mean((velocity_pred - velocity_target).pow(2), mask)


def alignment_loss(video_features: Tensor, action_features: Tensor) -> Tensor:
    """Optional video/action alignment loss (PRD §10.4) — COSINE variant.

    Chosen over InfoNCE: a contrastive objective needs large batches and a temperature to be
    meaningful, while D1 overfit batches are tiny (PRD 10.2). Cosine distance
    ``1 - cos(video, action)`` per sample is stable at any batch size and keeps the two pooled
    representations directionally consistent; 0 == perfectly aligned, 2 == opposite.

    Inputs ``[B, D]`` (or ``[B, S, D]``, mean-pooled over the middle dim) with equal ``D``.
    """
    v = video_features.mean(dim=1) if video_features.ndim == 3 else video_features
    a = action_features.mean(dim=1) if action_features.ndim == 3 else action_features
    if v.ndim != 2 or a.ndim != 2 or v.shape != a.shape:
        raise ValueError(
            f"expected matching [B, D] features, got {tuple(video_features.shape)} vs "
            f"{tuple(action_features.shape)}"
        )
    return (1.0 - torch.nn.functional.cosine_similarity(v, a, dim=-1)).mean()


def smoothness_loss(targets: Tensor) -> Tensor:
    """Second-difference (acceleration) penalty on action chunks ``[B, T, D]`` (PRD §10.4).

    Mean of ``(x[t+1] - 2 x[t] + x[t-1])^2``; exactly zero for constant and linear ramps, so
    intentional motion is not penalized (only jerk). Returns 0 for chunks with T < 3.
    """
    if targets.ndim != 3:
        raise ValueError(f"targets: expected [B, T, D], got shape {tuple(targets.shape)}")
    if targets.shape[1] < 3:
        return targets.sum() * 0.0
    second = targets[:, 2:] - 2.0 * targets[:, 1:-1] + targets[:, :-2]
    return second.pow(2).mean()


def limit_penalty(targets: Tensor, limit: float = 1.0) -> Tensor:
    """Soft squared hinge outside ``±limit`` in NORMALIZED action space (PRD §10.4).

    ``mean(relu(|x| - limit)^2)`` — zero inside the band, quadratic outside. This is a training
    regularizer only; hard limits stay in the deterministic safety layer (FR-07).
    """
    if limit <= 0:
        raise ValueError(f"limit must be > 0, got {limit}")
    return torch.relu(targets.abs() - limit).pow(2).mean()
