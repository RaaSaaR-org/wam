"""Trainable proprioceptive state encoder (FR-02, PRD 9.4, T-12).

``StateMLP`` maps the canonical robot state (q, dq, imu, gripper) to a fixed-size embedding
with a small MLP (2-4 linear layers). Field groups flagged invalid in the validity mask are
replaced by learned per-group 'missing' embeddings so missing/failed sensors never crash or
poison the forward pass (FR-02). Masking uses ``torch.where`` which is differentiable-safe:
gradients flow to the real input where valid and to the missing embedding where invalid, and
NaN/Inf values inside invalid groups cannot propagate.

Tensor shapes (N = num_joints, G = gripper_dims, B = batch, E = embedding_dim):
- ``encode(state)``:  RobotState -> ``[E]`` float32
- ``forward(batch)``: ``{"q": [B, N], "dq": [B, N], "imu": [B, 10], "gripper": [B, G],``
  ``"validity": [B, 4] (optional, bool/0-1 float, group order q, dq, imu, gripper)}``
  -> ``[B, E]`` float32
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import nn

from wam.interfaces.schema import RobotState

IMU_DIM = 10  # orientation quaternion wxyz (4) + angular velocity (3) + linear acceleration (3)

_GROUP_ORDER: tuple[str, ...] = ("q", "dq", "imu", "gripper")

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}


class StateMLPConfig(BaseModel):
    """Configuration for :class:`StateMLP`.

    ``hidden_dims`` holds the hidden layer widths; with the output projection the MLP has
    ``len(hidden_dims) + 1`` linear layers, constrained to the PRD 9.4 range of 2-4.
    """

    model_config = ConfigDict(frozen=True)

    embedding_dim: int = Field(ge=1)
    hidden_dims: tuple[int, ...] = Field(default=(128, 128), min_length=1, max_length=3)
    num_joints: int = Field(ge=1)
    gripper_dims: int = Field(default=1, ge=0)
    use_layernorm: bool = True
    activation: Literal["relu", "gelu", "tanh", "silu"] = "gelu"

    @field_validator("hidden_dims")
    @classmethod
    def _positive_hidden(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(h < 1 for h in v):
            raise ValueError("hidden_dims entries must be >= 1")
        return v

    @property
    def input_dim(self) -> int:
        """Concatenated raw input width: q(N) + dq(N) + imu(10) + gripper(G)."""
        return 2 * self.num_joints + IMU_DIM + self.gripper_dims


class StateMLP(nn.Module):
    """MLP state encoder implementing the ``wam.interfaces.StateEncoder`` protocol.

    Input = concat(q, dq, imu, gripper); each field group owns a learned 'missing' embedding
    of the group's raw width, substituted (via ``torch.where``) wherever the validity mask
    flags the group invalid. Deterministic init under ``torch.manual_seed``.
    """

    def __init__(self, config: StateMLPConfig) -> None:
        super().__init__()
        self.config = config
        self._group_dims: dict[str, int] = {
            "q": config.num_joints,
            "dq": config.num_joints,
            "imu": IMU_DIM,
            "gripper": config.gripper_dims,
        }
        self.missing = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(dim) * 0.02) for name, dim in self._group_dims.items()}
        )
        act = _ACTIVATIONS[config.activation]
        layers: list[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            if config.use_layernorm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(act())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, config.embedding_dim))
        self.mlp = nn.Sequential(*layers)

    @property
    def embedding_dim(self) -> int:
        """Output embedding dimensionality (StateEncoder protocol)."""
        return self.config.embedding_dim

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Batched training forward.

        ``batch`` keys: ``q`` [B, N], ``dq`` [B, N], ``imu`` [B, 10], ``gripper`` [B, G] and
        optionally ``validity`` [B, 4] (bool or 0/1 float; group order q, dq, imu, gripper;
        omitted == all valid). Returns embeddings [B, embedding_dim].
        """
        groups: dict[str, torch.Tensor] = {}
        for name in _GROUP_ORDER:
            if name not in batch:
                raise KeyError(f"batch missing required key {name!r}")
            x = batch[name]
            expected = self._group_dims[name]
            if x.ndim != 2 or x.shape[-1] != expected:
                raise ValueError(f"{name}: expected shape [B, {expected}], got {tuple(x.shape)}")
            groups[name] = x.to(dtype=torch.float32)

        batch_size = groups["q"].shape[0]
        validity = batch.get("validity")
        if validity is None:
            validity = torch.ones(
                batch_size, len(_GROUP_ORDER), dtype=torch.bool, device=groups["q"].device
            )
        else:
            if validity.shape != (batch_size, len(_GROUP_ORDER)):
                raise ValueError(
                    f"validity: expected shape [{batch_size}, {len(_GROUP_ORDER)}], "
                    f"got {tuple(validity.shape)}"
                )
            validity = validity.bool()

        parts: list[torch.Tensor] = []
        for idx, name in enumerate(_GROUP_ORDER):
            mask = validity[:, idx].unsqueeze(-1)  # [B, 1] bool, broadcasts over the group dim
            # Differentiable-safe: where() never multiplies by the invalid values, so NaN/Inf
            # in masked-out groups neither reach the output nor the backward pass.
            parts.append(torch.where(mask, groups[name], self.missing[name]))
        return self.mlp(torch.cat(parts, dim=-1))

    def encode(self, state: RobotState) -> torch.Tensor:
        """Single-state inference path (StateEncoder protocol): RobotState -> [embedding_dim].

        Groups flagged invalid in ``state.validity`` are never read from the state (arbitrary
        garbage/None is tolerated there); the learned missing embedding is used instead.
        """
        device = next(self.parameters()).device
        batch: dict[str, torch.Tensor] = {}
        flags = state.validity.as_dict()
        raw = {
            "q": state.q,
            "dq": state.dq,
            "imu": None,
            "gripper": state.gripper_state,
        }
        if flags["imu"]:
            raw["imu"] = np.concatenate(
                [
                    np.asarray(state.imu.orientation_wxyz, dtype=np.float32).reshape(-1),
                    np.asarray(state.imu.angular_velocity, dtype=np.float32).reshape(-1),
                    np.asarray(state.imu.linear_acceleration, dtype=np.float32).reshape(-1),
                ]
            )
        for name in _GROUP_ORDER:
            dim = self._group_dims[name]
            if flags[name]:
                arr = np.asarray(raw[name], dtype=np.float32).reshape(-1)
                if arr.shape[0] != dim:
                    raise ValueError(f"{name}: expected {dim} values, got {arr.shape[0]}")
                batch[name] = torch.from_numpy(arr.copy()).to(device).unsqueeze(0)
            else:
                batch[name] = torch.zeros(1, dim, dtype=torch.float32, device=device)
        batch["validity"] = torch.tensor(
            [[flags[name] for name in _GROUP_ORDER]], dtype=torch.bool, device=device
        )
        return self.forward(batch)[0]
