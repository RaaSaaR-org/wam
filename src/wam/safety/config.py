"""Safety layer configuration (FR-07, PRD §11.2).

Contracts:
- All limits are expressed in PHYSICAL canonical units (rad, rad/s, rad/s^2, m, m/s;
  gripper in [0, 1]) — the same units decoders emit (identity normalization, MVP).
- Per-joint tuples fix the canonical joint order (must match CanonicalSpaceSpec.joint_names).
- Frozen pydantic model: validated at construction, immutable afterwards.
- Torch-free; numpy only.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SAFETY_CONFIG_VERSION = "0.1.0"


def _all_finite(name: str, values: tuple[float, ...]) -> tuple[float, ...]:
    if any(not math.isfinite(v) for v in values):
        raise ValueError(f"{name}: all entries must be finite")
    return values


class SafetyConfig(BaseModel):
    """Deterministic limit set applied to every action chunk before execution.

    - ``q_min``/``q_max``: per-joint position limits [rad], canonical joint order.
    - ``dq_max``/``ddq_max``: per-joint max |velocity| [rad/s] / |acceleration| [rad/s^2].
    - ``workspace_min``/``workspace_max``: EE-mode AABB [m] in the frame the fk callable
      reports positions in (``CanonicalSpaceSpec.ee_frame``).
    - ``ee_max_lin_vel_m_s``: max EE translation speed [m/s] per chunk step.
    - ``ee_max_step_m``: max per-step EE translation magnitude [m]; the ONLY workspace bound
      applied when no fk callable is available.
    - ``gripper_rate_max``: max |gripper_target| change per second (gripper unit is [0, 1]).
    - ``chunk_timeout_s``: watchdog timeout between chunk arrivals/feeds.
    - ``timeout_policy``: what the watchdog decides on expiry ('hold' or 'stop').
    - ``hold_dt_s``: dt used for synthesized HOLD chunks when the incoming dt is unusable.
    """

    model_config = ConfigDict(frozen=True)

    version: str = SAFETY_CONFIG_VERSION
    q_min: tuple[float, ...] = Field(min_length=1)
    q_max: tuple[float, ...] = Field(min_length=1)
    dq_max: tuple[float, ...] = Field(min_length=1)
    ddq_max: tuple[float, ...] = Field(min_length=1)
    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]
    ee_max_lin_vel_m_s: float = Field(gt=0)
    ee_max_step_m: float = Field(gt=0)
    gripper_rate_max: float = Field(gt=0)
    chunk_timeout_s: float = Field(gt=0)
    timeout_policy: Literal["hold", "stop"] = "hold"
    hold_dt_s: float = Field(default=0.1, gt=0)

    @field_validator("q_min", "q_max", "workspace_min", "workspace_max")
    @classmethod
    def _finite(cls, v: tuple[float, ...], info) -> tuple[float, ...]:
        return _all_finite(info.field_name, v)

    @field_validator("dq_max", "ddq_max")
    @classmethod
    def _finite_positive(cls, v: tuple[float, ...], info) -> tuple[float, ...]:
        _all_finite(info.field_name, v)
        if any(x <= 0.0 for x in v):
            raise ValueError(f"{info.field_name}: all entries must be > 0")
        return v

    @field_validator("ee_max_lin_vel_m_s", "ee_max_step_m", "gripper_rate_max", "chunk_timeout_s")
    @classmethod
    def _finite_scalar(cls, v: float, info) -> float:
        if not math.isfinite(v):
            raise ValueError(f"{info.field_name}: must be finite")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> SafetyConfig:
        n = len(self.q_min)
        for name in ("q_max", "dq_max", "ddq_max"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name}: length {len(getattr(self, name))} != len(q_min) {n}")
        if any(lo >= hi for lo, hi in zip(self.q_min, self.q_max)):
            raise ValueError("q_min must be < q_max elementwise")
        if any(lo >= hi for lo, hi in zip(self.workspace_min, self.workspace_max)):
            raise ValueError("workspace_min must be < workspace_max elementwise")
        return self

    @property
    def num_joints(self) -> int:
        return len(self.q_min)

    def q_min_arr(self) -> np.ndarray:
        """[num_joints] float64."""
        return np.asarray(self.q_min, dtype=np.float64)

    def q_max_arr(self) -> np.ndarray:
        """[num_joints] float64."""
        return np.asarray(self.q_max, dtype=np.float64)

    def dq_max_arr(self) -> np.ndarray:
        """[num_joints] float64."""
        return np.asarray(self.dq_max, dtype=np.float64)

    def ddq_max_arr(self) -> np.ndarray:
        """[num_joints] float64."""
        return np.asarray(self.ddq_max, dtype=np.float64)

    def workspace_min_arr(self) -> np.ndarray:
        """[3] float64."""
        return np.asarray(self.workspace_min, dtype=np.float64)

    def workspace_max_arr(self) -> np.ndarray:
        """[3] float64."""
        return np.asarray(self.workspace_max, dtype=np.float64)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SafetyConfig:
        """Load and validate a config from a YAML mapping file."""
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise TypeError(f"{path}: expected a YAML mapping, got {type(raw).__name__}")
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        """Write the config as a YAML mapping (roundtrips through ``from_yaml``)."""
        Path(path).write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))
