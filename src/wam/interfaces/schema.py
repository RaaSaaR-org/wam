"""Canonical, versioned robot state/action schema (PRD Anhang A, FR-06).

Contracts:
- This module is robot-agnostic. Robot-specific joint mapping, units and limits live ONLY in
  robot adapters (``wam.robot``).
- Array containers (RobotState, ActionChunk, IMUState) are plain dataclasses on the hot path.
  Construction never raises on bad values: schema stores, safety rejects. ``validate()`` flags
  problems as a list of human-readable strings (empty list == valid).
- Specs/configs (CanonicalSpaceSpec, NormalizationSpec) are pydantic models and DO validate at
  construction.
- Torch-free by design; numpy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "0.1.0"

_IMU_QUAT_DIMS = 4
_IMU_VEC_DIMS = 3


def _schema_major(version: str) -> str:
    return version.split(".", 1)[0]


class ActionMode(str, Enum):
    """Canonical action representation (PRD OD-02: joint delta vs. end-effector delta)."""

    JOINT_DELTA = "joint_delta"
    EE_DELTA = "ee_delta"


class CanonicalSpaceSpec(BaseModel):
    """Robot-agnostic definition of the canonical state/action space.

    ``joint_names`` fixes the canonical joint ORDER for all ``q``/``dq``/``targets`` arrays.
    Mapping to a physical robot's joint indices/units is the robot adapter's job (FR-06).
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    joint_names: tuple[str, ...] = Field(min_length=1)
    gripper_dims: int = Field(default=1, ge=0)
    ee_frame: str = Field(
        default="base",
        description="Reference frame in which EE_DELTA targets are expressed.",
    )
    ee_rotation_convention: str = Field(
        default="quat_wxyz",
        description="Rotation parametrization convention for EE_DELTA targets.",
    )

    @field_validator("joint_names")
    @classmethod
    def _unique_joint_names(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(v)) != len(v):
            raise ValueError("joint_names must be unique")
        return v

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def target_dim(self, mode: ActionMode) -> int:
        """Expected per-step target dimensionality ``D`` for the given action mode."""
        if mode is ActionMode.JOINT_DELTA:
            return self.num_joints
        # xyz translation delta + rotation delta in the declared convention.
        rot = _IMU_QUAT_DIMS if self.ee_rotation_convention == "quat_wxyz" else _IMU_VEC_DIMS
        return _IMU_VEC_DIMS + rot


def _check_array(
    issues: list[str],
    name: str,
    arr: np.ndarray,
    shape: tuple[int, ...] | None,
    check_finite: bool = True,
) -> None:
    if not isinstance(arr, np.ndarray):
        issues.append(f"{name}: expected np.ndarray, got {type(arr).__name__}")
        return
    if arr.dtype != np.float32:
        issues.append(f"{name}: expected dtype float32, got {arr.dtype}")
    if shape is not None and arr.shape != shape:
        issues.append(f"{name}: expected shape {shape}, got {arr.shape}")
    if check_finite and arr.size > 0 and not np.isfinite(arr).all():
        issues.append(f"{name}: contains NaN/Inf")


@dataclass
class ValidityMask:
    """Per field-group validity flags. Policies MUST handle groups flagged invalid/missing."""

    q: bool = True
    dq: bool = True
    imu: bool = True
    gripper: bool = True

    def as_dict(self) -> dict[str, bool]:
        return {"q": self.q, "dq": self.dq, "imu": self.imu, "gripper": self.gripper}


@dataclass
class IMUState:
    """IMU sample. orientation is a unit quaternion in w,x,y,z order."""

    orientation_wxyz: np.ndarray  # [4] float32
    angular_velocity: np.ndarray  # [3] float32, rad/s
    linear_acceleration: np.ndarray  # [3] float32, m/s^2

    def validate(self) -> list[str]:
        issues: list[str] = []
        _check_array(issues, "imu.orientation_wxyz", self.orientation_wxyz, (_IMU_QUAT_DIMS,))
        _check_array(issues, "imu.angular_velocity", self.angular_velocity, (_IMU_VEC_DIMS,))
        _check_array(issues, "imu.linear_acceleration", self.linear_acceleration, (_IMU_VEC_DIMS,))
        return issues


@dataclass
class RobotState:
    """Canonical proprioceptive state. Joint order is fixed by CanonicalSpaceSpec.joint_names.

    Construction never raises on bad values; use ``validate()`` (safety layer rejects).
    """

    timestamp_ns: int
    q: np.ndarray  # [num_joints] float32, canonical order
    dq: np.ndarray  # [num_joints] float32, canonical order
    imu: IMUState
    gripper_state: np.ndarray  # [gripper_dims] float32
    validity: ValidityMask = field(default_factory=ValidityMask)
    schema_version: str = SCHEMA_VERSION

    def validate(self, spec: CanonicalSpaceSpec | None = None) -> list[str]:
        """Flag structural problems. Only groups marked valid in ``validity`` are checked."""
        issues: list[str] = []
        if self.timestamp_ns < 0:
            issues.append(f"timestamp_ns: must be >= 0, got {self.timestamp_ns}")
        n = spec.num_joints if spec is not None else None
        if self.validity.q:
            _check_array(issues, "q", self.q, (n,) if n is not None else None)
        if self.validity.dq:
            _check_array(issues, "dq", self.dq, (n,) if n is not None else None)
        if (
            self.validity.q
            and self.validity.dq
            and isinstance(self.q, np.ndarray)
            and isinstance(self.dq, np.ndarray)
            and self.q.shape != self.dq.shape
        ):
            issues.append(f"q/dq shape mismatch: {self.q.shape} vs {self.dq.shape}")
        if self.validity.imu:
            issues.extend(self.imu.validate())
        if self.validity.gripper:
            g = spec.gripper_dims if spec is not None else None
            _check_array(issues, "gripper_state", self.gripper_state, (g,) if g else None)
        if _schema_major(self.schema_version) != _schema_major(SCHEMA_VERSION):
            issues.append(
                f"schema_version: incompatible major ({self.schema_version} vs {SCHEMA_VERSION})"
            )
        return issues


@dataclass
class ActionChunk:
    """Chunk of T future action steps in the canonical action space, PHYSICAL units.

    - ``targets``: [T, D] float32 in physical canonical units — rad deltas for JOINT_DELTA,
      m (+ quaternion) for EE_DELTA; D depends on ``mode`` (CanonicalSpaceSpec.target_dim).
      The MVP pipeline is identity-normalized end-to-end: decoders emit physical units and
      the safety layer compares them against physical limits directly (``NormalizationSpec``
      is parked, see its docstring).
    - ``gripper_target``: [T] float32 in [0, 1].
    - ``dt_s``: control period per step; total horizon == ``duration``.
    - T is configuration-driven (MVP guidance 8-32) and intentionally NOT enforced here.
    Construction never raises on bad values; use ``validate()`` (safety layer rejects).
    """

    mode: ActionMode
    targets: np.ndarray  # [T, D] float32, physical canonical units (rad / m)
    gripper_target: np.ndarray  # [T] float32 in [0, 1]
    dt_s: float
    schema_version: str = SCHEMA_VERSION

    @property
    def num_steps(self) -> int:
        return int(self.targets.shape[0])

    @property
    def duration(self) -> float:
        """Total chunk horizon in seconds (num_steps * dt_s)."""
        return self.num_steps * self.dt_s

    def validate(self, spec: CanonicalSpaceSpec | None = None) -> list[str]:
        issues: list[str] = []
        if not isinstance(self.mode, ActionMode):
            issues.append(f"mode: expected ActionMode, got {type(self.mode).__name__}")
        if not isinstance(self.targets, np.ndarray) or self.targets.ndim != 2:
            issues.append("targets: expected np.ndarray of shape [T, D]")
        else:
            t, d = self.targets.shape
            if t < 1:
                issues.append("targets: T must be >= 1")
            _check_array(issues, "targets", self.targets, (t, d))
            if spec is not None and isinstance(self.mode, ActionMode):
                expected_d = spec.target_dim(self.mode)
                if d != expected_d:
                    issues.append(f"targets: expected D={expected_d} for {self.mode}, got {d}")
            _check_array(issues, "gripper_target", self.gripper_target, (t,))
        if isinstance(self.gripper_target, np.ndarray) and self.gripper_target.size > 0:
            finite = self.gripper_target[np.isfinite(self.gripper_target)]
            if finite.size > 0 and (finite.min() < 0.0 or finite.max() > 1.0):
                issues.append("gripper_target: values outside [0, 1]")
        if not (isinstance(self.dt_s, (int, float)) and math.isfinite(self.dt_s) and self.dt_s > 0):
            issues.append(f"dt_s: must be finite and > 0, got {self.dt_s}")
        if _schema_major(self.schema_version) != _schema_major(SCHEMA_VERSION):
            issues.append(
                f"schema_version: incompatible major ({self.schema_version} vs {SCHEMA_VERSION})"
            )
        return issues


class NormalizationSpec(BaseModel):
    """Per-dimension affine normalization: ``z = (x - mean) / std``.

    PARKED FOR THE MVP: the shipped pipeline is identity-normalized end-to-end — episode
    manifests may STORE a spec (provenance only), but nothing in training or runtime applies
    it. ``EpisodeDataset`` refuses episodes that declare a non-identity spec for action
    targets rather than silently training on raw units. Wiring this up (dataset applies,
    checkpoint persists, policy denormalizes) is a deliberate follow-up, not implied by
    storing a spec.

    - Applies to the LAST axis of the input (per-dim stats over D dims).
    - Computation and results are float64 (full precision); cast to float32 only when storing
      (e.g. into ActionChunk.targets).
    - Roundtrip contract: ``denormalize(normalize(x)).astype(np.float32)`` reproduces float32
      inputs bit-exactly.
    - Serializable via ``to_dict()``/``from_dict()``; versioned, major must match.
    """

    model_config = ConfigDict(frozen=True)

    version: str = SCHEMA_VERSION
    mean: tuple[float, ...] = Field(min_length=1)
    std: tuple[float, ...] = Field(min_length=1)

    @field_validator("std")
    @classmethod
    def _std_positive(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if any(not math.isfinite(s) or s <= 0.0 for s in v):
            raise ValueError("std entries must be finite and > 0")
        return v

    @model_validator(mode="after")
    def _same_length(self) -> NormalizationSpec:
        if len(self.mean) != len(self.std):
            raise ValueError(f"mean/std length mismatch: {len(self.mean)} vs {len(self.std)}")
        return self

    @property
    def dim(self) -> int:
        return len(self.mean)

    def _stats(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.mean, dtype=np.float64),
            np.asarray(self.std, dtype=np.float64),
        )

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """x: [..., dim] -> normalized float64 array of the same shape."""
        mean, std = self._stats()
        return (np.asarray(x, dtype=np.float64) - mean) / std

    def denormalize(self, z: np.ndarray) -> np.ndarray:
        """z: [..., dim] -> denormalized float64 array of the same shape."""
        mean, std = self._stats()
        return np.asarray(z, dtype=np.float64) * std + mean

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "mean": list(self.mean), "std": list(self.std)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizationSpec:
        version = str(data.get("version", ""))
        if _schema_major(version) != _schema_major(SCHEMA_VERSION):
            raise ValueError(
                f"incompatible NormalizationSpec version {version!r}, expected major "
                f"{_schema_major(SCHEMA_VERSION)}.x.x"
            )
        return cls(version=version, mean=tuple(data["mean"]), std=tuple(data["std"]))
