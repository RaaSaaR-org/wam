"""Action head: backbone features -> bounded action chunks (FR-04, T-12).

``ActionHead`` is a lightweight decoder on backbone intermediate features (PRD 9.6). Targets
are tanh-bounded to (-1, 1) and interpreted DIRECTLY as physical canonical units (per-step
rad / m deltas — the MVP pipeline is identity-normalized end-to-end, so training data must
keep per-step |targets| < 1; ``EpisodeDataset`` enforces this). Hard limits live downstream
in the safety layer. The gripper command is sigmoid-bounded to (0, 1).

Tensor shapes (T = num_steps, D = target_dim, G = gripper_dims, B = batch, F = feature_dim):
- ``forward(features)``: ``[B, F]`` -> ``{"targets": [B, T, D], "gripper": [B, T, G]}``
- ``decode(features)``:  ``[F]`` or ``[*, F]`` (leading dims mean-pooled) -> ActionChunk with
  ``targets`` [T, D] float32, ``gripper_target`` [T] float32 (mean over G), ``dt_s`` from config.
"""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import nn

from wam.interfaces.schema import ActionChunk, ActionMode

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}


class ActionHeadConfig(BaseModel):
    """Configuration for :class:`ActionHead`.

    ``num_steps`` follows the MVP chunk-length guidance of 8-32 steps (PRD 9.10) and is
    configurable within that range. ``dt_s`` is the control period stamped into decoded chunks.
    """

    model_config = ConfigDict(frozen=True)

    feature_dim: int = Field(ge=1)
    num_steps: int = Field(ge=8, le=32)
    target_dim: int = Field(ge=1)
    gripper_dims: int = Field(default=1, ge=1)
    mode: ActionMode = ActionMode.JOINT_DELTA
    dt_s: float = Field(default=0.05, gt=0)
    hidden_dims: tuple[int, ...] = Field(default=(64,), min_length=1, max_length=3)
    use_layernorm: bool = True
    activation: Literal["relu", "gelu", "tanh", "silu"] = "gelu"

    @field_validator("hidden_dims")
    @classmethod
    def _positive_hidden(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(h < 1 for h in v):
            raise ValueError("hidden_dims entries must be >= 1")
        return v


class ActionHead(nn.Module):
    """MLP trunk + two bounded heads implementing ``wam.interfaces.ActionDecoder``.

    Deterministic init under ``torch.manual_seed``. Output is in physical canonical units
    (identity normalization, MVP); it must still pass the deterministic safety layer before
    reaching any robot adapter (FR-07).
    """

    def __init__(self, config: ActionHeadConfig) -> None:
        super().__init__()
        self.config = config
        act = _ACTIVATIONS[config.activation]
        layers: list[nn.Module] = []
        in_dim = config.feature_dim
        for hidden in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            if config.use_layernorm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(act())
            in_dim = hidden
        self.trunk = nn.Sequential(*layers)
        self.target_head = nn.Linear(in_dim, config.num_steps * config.target_dim)
        self.gripper_head = nn.Linear(in_dim, config.num_steps * config.gripper_dims)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Batched training forward: features [B, feature_dim] ->
        ``{"targets": [B, T, D] in (-1, 1), "gripper": [B, T, G] in (0, 1)}``.
        """
        if features.ndim != 2 or features.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"features: expected shape [B, {self.config.feature_dim}], "
                f"got {tuple(features.shape)}"
            )
        batch = features.shape[0]
        hidden = self.trunk(features.to(torch.float32))
        targets = torch.tanh(self.target_head(hidden)).view(
            batch, self.config.num_steps, self.config.target_dim
        )
        gripper = torch.sigmoid(self.gripper_head(hidden)).view(
            batch, self.config.num_steps, self.config.gripper_dims
        )
        return {"targets": targets, "gripper": gripper}

    @torch.no_grad()
    def decode(self, features: torch.Tensor) -> ActionChunk:
        """Inference path (ActionDecoder protocol): features -> canonical ActionChunk.

        Accepts ``[feature_dim]`` or ``[*, feature_dim]`` (all leading dims, e.g. backbone
        tokens, are mean-pooled). The chunk's scalar per-step gripper command is the mean over
        ``gripper_dims``; ``mode`` and ``dt_s`` come from the config.
        """
        device = next(self.parameters()).device
        feats = torch.as_tensor(features, dtype=torch.float32, device=device)
        if feats.ndim == 0 or feats.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"features: expected last dim {self.config.feature_dim}, "
                f"got shape {tuple(feats.shape)}"
            )
        if feats.ndim > 1:
            feats = feats.reshape(-1, self.config.feature_dim).mean(dim=0)
        out = self.forward(feats.unsqueeze(0))
        targets = out["targets"][0].cpu().numpy()  # [T, D] float32
        gripper = out["gripper"][0].mean(dim=-1).cpu().numpy()  # [T] float32
        return ActionChunk(
            mode=self.config.mode,
            targets=targets,
            gripper_target=gripper,
            dt_s=self.config.dt_s,
        )
