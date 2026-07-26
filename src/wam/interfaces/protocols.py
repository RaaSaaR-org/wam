"""Versioned runtime-checkable protocols for all swappable WAM modules (FR-09, T-02).

Contracts:
- Torch-free by design. Tensor-valued inputs/outputs of encoders/backbones are typed ``Any``;
  the documented contract is: framework tensors (e.g. torch.Tensor) whose LAST dimension is the
  documented feature/embedding dim. Array data crossing the robot/safety boundary is numpy only.
- All protocols are structural (``@runtime_checkable``): any object with matching members
  conforms; use ``isinstance`` checks, never inheritance requirements.
- Backbones are swappable (FR-09/AC-05): FLUX 3 Dev and an open I2V fallback must both fit
  behind ``BackboneAdapter`` without changes to the data schema or robot API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from wam.interfaces.schema import ActionChunk, RobotState

INTERFACES_VERSION = "0.1.0"


@dataclass
class Observation:
    """Single policy input. ``images`` maps camera name (e.g. 'front', 'wrist') to an
    HxWxC uint8/float32 numpy array; ``instruction`` is the language task string."""

    images: dict[str, np.ndarray]
    state: RobotState
    instruction: str


@dataclass
class SafetyIntervention:
    """One deterministic safety-layer intervention. Every intervention MUST be logged (FR-07).

    ``kind`` is a stable machine-readable identifier (e.g. 'joint_limit', 'velocity_limit',
    'workspace', 'nan_reject', 'watchdog_timeout'); ``detail`` is human-readable context.
    """

    kind: str
    detail: str
    timestamp_ns: int


@runtime_checkable
class StateEncoder(Protocol):
    """Trainable proprioception adapter (FR-02). Must tolerate field groups flagged invalid
    in ``state.validity`` (missing sensors must not crash encoding)."""

    @property
    def embedding_dim(self) -> int:
        """Output embedding dimensionality."""
        ...

    def encode(self, state: RobotState) -> Any:
        """RobotState -> embedding tensor with last dim == embedding_dim."""
        ...


@runtime_checkable
class ActionEncoder(Protocol):
    """TRAINING ONLY (PRD 9.5): embeds demonstrated chunks into the joint latent space.
    Never part of the runtime inference path."""

    @property
    def latent_dim(self) -> int:
        """Output latent dimensionality."""
        ...

    def encode(self, chunk: ActionChunk) -> Any:
        """ActionChunk -> action latent tensor with last dim == latent_dim."""
        ...


@runtime_checkable
class ActionDecoder(Protocol):
    """Lightweight decoder from backbone intermediate features to a canonical chunk (FR-04).
    Output is in PHYSICAL canonical units (rad / m; gripper in [0, 1]) — the MVP pipeline
    uses identity normalization end-to-end (NormalizationSpec is parked, see its docstring).
    Hard limits are enforced downstream by the safety layer."""

    def decode(self, features: Any) -> ActionChunk:
        """features: tensor with last dim == BackboneAdapter.feature_dim -> ActionChunk."""
        ...


@runtime_checkable
class BackboneAdapter(Protocol):
    """Uniform interface over video backbones (FLUX 3 Dev, open I2V fallback — FR-09/AC-05).

    Must expose INTERMEDIATE features: conditioning inputs are embedded separately, then
    ``features()`` runs the backbone and returns intermediate activations the ActionDecoder
    consumes. Swapping the backbone must not change data schema or robot API.
    """

    @property
    def name(self) -> str:
        """Stable backbone identifier (for AC-04 traceability), e.g. 'flux3-dev', 'wan2.1-i2v'."""
        ...

    @property
    def feature_dim(self) -> int:
        """Last dimension of the tensors returned by ``features()``."""
        ...

    def condition_video(self, video: Any) -> Any:
        """Past frames/video latents -> backbone-native video conditioning context."""
        ...

    def condition_text(self, text: str) -> Any:
        """Language instruction -> backbone-native text conditioning context."""
        ...

    def condition_state(self, state_embedding: Any) -> Any:
        """StateEncoder output -> backbone-native state conditioning context."""
        ...

    def features(self, video_ctx: Any, text_ctx: Any, state_ctx: Any) -> Any:
        """Run the backbone on the conditioning contexts and return intermediate features
        (tensor, last dim == feature_dim) for the ActionDecoder; may also drive video
        prediction internally."""
        ...


@runtime_checkable
class SafetyFilter(Protocol):
    """Deterministic action gate (FR-07). No ML. No learned component may bypass it.
    Rejects or safely projects invalid actions; every change is reported as an intervention."""

    def filter(
        self, state: RobotState, chunk: ActionChunk
    ) -> tuple[ActionChunk, list[SafetyIntervention]]:
        """Returns a safe (possibly projected/truncated) chunk plus all interventions applied.
        An empty intervention list means the chunk passed unchanged."""
        ...


@runtime_checkable
class RobotAdapter(Protocol):
    """HAL between the canonical schema and one robot API (FR-06). Owns joint mapping, units,
    calibration and vendor limits. The ONLY place robot-specific mapping may live."""

    @property
    def limits(self) -> dict[str, np.ndarray]:
        """Canonical-order limit arrays. Required keys: 'q_min', 'q_max', 'dq_max' ([num_joints]
        float32); optional: 'ddq_max', 'gripper_min', 'gripper_max'."""
        ...

    def read_state(self) -> RobotState:
        """Latest synchronized state in the canonical schema."""
        ...

    def execute(self, chunk: ActionChunk, prefix_steps: int) -> None:
        """Execute only the first ``prefix_steps`` steps of ``chunk`` (receding horizon, FR-05).
        The chunk MUST already have passed the SafetyFilter."""
        ...

    def hold(self) -> None:
        """Hold current position. Timeout recovery: never keep extrapolating a stale action."""
        ...

    def estop(self) -> None:
        """Emergency stop. Must be safe to call at any time, from any thread."""
        ...


@runtime_checkable
class Policy(Protocol):
    """Full observation -> action mapping used by the closed-loop runtime (FR-05).
    Output must still pass the SafetyFilter before reaching the RobotAdapter."""

    def predict(self, observation: Observation) -> ActionChunk:
        """One planning step: fresh observation -> next canonical action chunk."""
        ...
