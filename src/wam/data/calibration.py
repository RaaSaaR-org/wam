"""Camera/kinematics calibration storage (T-10, code part).

Storage + validation ONLY — there is no calibration solver in here. Producing the numbers
(ChArUco boards, hand-eye, joint-zero measurement) is a hardware workflow (docs/teleop.md);
this module makes the RESULT versioned, hashable and loadable so every episode/rollout can
reference the exact calibration it was recorded under (FR-10, R-04).

Contracts:
- Frozen pydantic models; validated at construction, immutable afterwards.
- ``CalibrationSet.from_yaml``/``to_yaml`` roundtrip losslessly; the YAML carries the usual
  top-level ``wam_config_version`` gate (compatible with ``wam.interfaces.load_config``).
- ``CalibrationSet.config_hash()`` is the canonical content hash — store it in
  ``EpisodeWriter(extra={"calibration_hash": ...})`` and in run logs.
- Units: meters, radians; quaternions in w,x,y,z order (matches the canonical schema).
- Torch-free; numpy + pydantic + yaml only.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from wam.interfaces.versioning import WAM_CONFIG_VERSION
from wam.interfaces.versioning import config_hash as _config_hash

CALIBRATION_VERSION = "0.1.0"

#: |quaternion norm - 1| beyond this is rejected (YAML rounding stays well below).
_QUAT_NORM_TOL = 1e-3

__all__ = [
    "CALIBRATION_VERSION",
    "CalibrationSet",
    "CameraExtrinsics",
    "CameraIntrinsics",
]


def _major(version: str) -> str:
    return version.split(".", 1)[0]


def _all_finite(name: str, values: tuple[float, ...]) -> tuple[float, ...]:
    if any(not math.isfinite(v) for v in values):
        raise ValueError(f"{name}: all entries must be finite")
    return values


class CameraIntrinsics(BaseModel):
    """Pinhole intrinsics of one camera at its calibrated resolution.

    ``distortion`` coefficients follow ``distortion_model`` ('none' -> empty, 'radtan' ->
    OpenCV k1 k2 p1 p2 [k3], 'fisheye' -> k1..k4); the length is model-dependent and not
    enforced here.
    """

    model_config = ConfigDict(frozen=True)

    width: int = Field(ge=1)
    height: int = Field(ge=1)
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    distortion_model: Literal["none", "radtan", "fisheye"] = "none"
    distortion: tuple[float, ...] = ()

    @field_validator("fx", "fy", "cx", "cy")
    @classmethod
    def _finite_scalar(cls, v: float, info: Any) -> float:
        if not math.isfinite(v):
            raise ValueError(f"{info.field_name}: must be finite")
        return v

    @field_validator("distortion")
    @classmethod
    def _finite_distortion(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        return _all_finite("distortion", v)

    def matrix(self) -> np.ndarray:
        """Camera matrix K as [3, 3] float64."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


class CameraExtrinsics(BaseModel):
    """Pose of one camera in ``parent_frame``: T_parent_camera (camera -> parent points)."""

    model_config = ConfigDict(frozen=True)

    parent_frame: str = "base"
    translation_m: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]  # unit quaternion, w,x,y,z order

    @field_validator("translation_m")
    @classmethod
    def _finite_translation(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        _all_finite("translation_m", v)
        return v

    @field_validator("rotation_wxyz")
    @classmethod
    def _unit_quaternion(
        cls, v: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        _all_finite("rotation_wxyz", v)
        norm = math.sqrt(sum(c * c for c in v))
        if abs(norm - 1.0) > _QUAT_NORM_TOL:
            raise ValueError(f"rotation_wxyz: expected unit quaternion, |q| = {norm:.6f}")
        return v

    def rotation_matrix(self) -> np.ndarray:
        """Rotation part as [3, 3] float64."""
        w, x, y, z = self.rotation_wxyz
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def matrix(self) -> np.ndarray:
        """Homogeneous transform T_parent_camera as [4, 4] float64."""
        t = np.eye(4, dtype=np.float64)
        t[:3, :3] = self.rotation_matrix()
        t[:3, 3] = np.asarray(self.translation_m, dtype=np.float64)
        return t


class CalibrationSet(BaseModel):
    """One versioned calibration snapshot: camera intrinsics/extrinsics + joint offsets.

    - ``intrinsics``/``extrinsics`` are keyed by camera name (episode camera names);
      the key sets may differ (a camera may have intrinsics before its mount is measured).
    - ``joint_offsets_rad``: kinematic zero-offset per canonical joint name, ADDED to raw
      joint readings by the robot adapter. Absent joints mean offset 0.
    - ``calibrated_at``/``method`` are provenance strings (ISO-8601 date, tool name).
    """

    model_config = ConfigDict(frozen=True)

    calibration_version: str = CALIBRATION_VERSION
    robot: str = ""
    calibrated_at: str = ""
    method: str = ""
    intrinsics: dict[str, CameraIntrinsics] = Field(default_factory=dict)
    extrinsics: dict[str, CameraExtrinsics] = Field(default_factory=dict)
    joint_offsets_rad: dict[str, float] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("calibration_version")
    @classmethod
    def _version_major_match(cls, v: str) -> str:
        if _major(v) != _major(CALIBRATION_VERSION):
            raise ValueError(
                f"incompatible calibration_version {v!r}, "
                f"expected major {_major(CALIBRATION_VERSION)}.x.x"
            )
        return v

    @field_validator("joint_offsets_rad")
    @classmethod
    def _finite_offsets(cls, v: dict[str, float]) -> dict[str, float]:
        for name, value in v.items():
            if not math.isfinite(value):
                raise ValueError(f"joint_offsets_rad[{name!r}]: must be finite")
        return v

    def cameras(self) -> tuple[str, ...]:
        """Sorted union of all camera names appearing in intrinsics or extrinsics."""
        return tuple(sorted(set(self.intrinsics) | set(self.extrinsics)))

    def config_hash(self) -> str:
        """Deterministic sha256 of the calibration content (FR-10 traceability)."""
        return _config_hash(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CalibrationSet:
        """Load and validate a calibration YAML (top-level ``wam_config_version`` gated)."""
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise TypeError(f"{path}: expected a YAML mapping, got {type(raw).__name__}")
        data = dict(raw)
        wam_version = data.pop("wam_config_version", None)
        if wam_version is not None and _major(str(wam_version)) != _major(WAM_CONFIG_VERSION):
            raise ValueError(
                f"{path}: incompatible wam_config_version {wam_version!r}, "
                f"expected major {_major(WAM_CONFIG_VERSION)}.x.x"
            )
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Write as a YAML mapping (roundtrips through ``from_yaml``, hash-stable)."""
        payload: dict[str, Any] = {
            "wam_config_version": WAM_CONFIG_VERSION,
            **self.model_dump(mode="json"),
        }
        Path(path).write_text(yaml.safe_dump(payload, sort_keys=False))
