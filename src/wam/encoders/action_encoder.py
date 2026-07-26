"""Action chunk encoder — TRAINING ONLY (PRD 9.5, FR-03, T-16 prep).

``ActionChunkEncoder`` embeds demonstrated action chunks into per-step latents that can be
modeled jointly with video latents. It is never part of the runtime inference path.

Per step, targets and gripper command are concatenated and passed through a small MLP; a
learned positional embedding (by step index) is added so the latent sequence is order-aware.

Tensor shapes (T = steps, D = target_dim, G = gripper_dims, B = batch, L = latent_dim):
- ``encode(chunk)``:            ActionChunk -> ``[T, L]`` float32
- ``forward(targets, gripper)``: ``[B, T, D]``, ``[B, T, G]`` -> ``[B, T, L]`` float32
"""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import nn

from wam.interfaces.schema import ActionChunk

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}


class ActionChunkEncoderConfig(BaseModel):
    """Configuration for :class:`ActionChunkEncoder`.

    ``max_steps`` bounds the learned positional table; chunks longer than this are rejected.
    """

    model_config = ConfigDict(frozen=True)

    latent_dim: int = Field(ge=1)
    hidden_dims: tuple[int, ...] = Field(default=(64,), min_length=1, max_length=3)
    target_dim: int = Field(ge=1)
    gripper_dims: int = Field(default=1, ge=1)
    max_steps: int = Field(default=32, ge=1)
    use_layernorm: bool = True
    activation: Literal["relu", "gelu", "tanh", "silu"] = "gelu"

    @field_validator("hidden_dims")
    @classmethod
    def _positive_hidden(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(h < 1 for h in v):
            raise ValueError("hidden_dims entries must be >= 1")
        return v


class ActionChunkEncoder(nn.Module):
    """Per-step MLP + positional embedding implementing ``wam.interfaces.ActionEncoder``.

    Deterministic init under ``torch.manual_seed``. Training-only module.
    """

    def __init__(self, config: ActionChunkEncoderConfig) -> None:
        super().__init__()
        self.config = config
        act = _ACTIVATIONS[config.activation]
        layers: list[nn.Module] = []
        in_dim = config.target_dim + config.gripper_dims
        for hidden in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            if config.use_layernorm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(act())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, config.latent_dim))
        self.mlp = nn.Sequential(*layers)
        self.pos_embedding = nn.Embedding(config.max_steps, config.latent_dim)

    @property
    def latent_dim(self) -> int:
        """Output latent dimensionality (ActionEncoder protocol)."""
        return self.config.latent_dim

    def forward(self, targets: torch.Tensor, gripper: torch.Tensor) -> torch.Tensor:
        """Batched training forward: targets [B, T, D], gripper [B, T, G] -> [B, T, latent_dim]."""
        if targets.ndim != 3 or targets.shape[-1] != self.config.target_dim:
            raise ValueError(
                f"targets: expected shape [B, T, {self.config.target_dim}], "
                f"got {tuple(targets.shape)}"
            )
        if gripper.ndim != 3 or gripper.shape[-1] != self.config.gripper_dims:
            raise ValueError(
                f"gripper: expected shape [B, T, {self.config.gripper_dims}], "
                f"got {tuple(gripper.shape)}"
            )
        if gripper.shape[:2] != targets.shape[:2]:
            raise ValueError(
                f"targets/gripper B,T mismatch: {tuple(targets.shape[:2])} vs "
                f"{tuple(gripper.shape[:2])}"
            )
        num_steps = targets.shape[1]
        if num_steps > self.config.max_steps:
            raise ValueError(f"chunk has {num_steps} steps, max_steps={self.config.max_steps}")
        x = torch.cat([targets.to(torch.float32), gripper.to(torch.float32)], dim=-1)
        latents = self.mlp(x)  # [B, T, L]
        positions = torch.arange(num_steps, device=latents.device)
        return latents + self.pos_embedding(positions)  # pos broadcasts over batch

    def encode(self, chunk: ActionChunk) -> torch.Tensor:
        """ActionChunk -> latent [T, latent_dim] (ActionEncoder protocol).

        The canonical chunk carries a scalar gripper command per step ([T]); it is broadcast
        across ``gripper_dims`` before encoding.
        """
        device = next(self.parameters()).device
        targets = torch.as_tensor(chunk.targets, dtype=torch.float32, device=device)
        if targets.ndim != 2 or targets.shape[-1] != self.config.target_dim:
            raise ValueError(
                f"chunk.targets: expected shape [T, {self.config.target_dim}], "
                f"got {tuple(targets.shape)}"
            )
        gripper = torch.as_tensor(chunk.gripper_target, dtype=torch.float32, device=device)
        if gripper.ndim != 1 or gripper.shape[0] != targets.shape[0]:
            raise ValueError(
                f"chunk.gripper_target: expected shape [{targets.shape[0]}], "
                f"got {tuple(gripper.shape)}"
            )
        gripper = gripper.unsqueeze(-1).expand(-1, self.config.gripper_dims)
        return self.forward(targets.unsqueeze(0), gripper.unsqueeze(0))[0]
