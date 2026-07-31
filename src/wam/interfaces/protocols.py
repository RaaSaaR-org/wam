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

INTERFACES_VERSION = "0.3.0"


@dataclass
class Observation:
    """Single policy input. ``images`` maps camera name (e.g. 'front', 'wrist') to an
    HxWxC uint8/float32 numpy array; ``instruction`` is the language task string.

    ``image_history`` optionally carries the *preceding* frames for a camera as a single
    ``[T, H, W, C]`` stack, oldest first, **whose last entry is** ``images[key]``. It exists
    because a video backbone is trained on a moving clip and, without it, has to be shown the
    same still N times — which carries no motion at all (T-29, ``docs/improvements.md`` I-7).

    Optional on purpose, and the invariant above is what makes it safe to be optional: the
    history is a strict superset of ``images``, so a policy that ignores it is still correct,
    and a producer that fills it wrongly is detectable rather than silently degrading. Sources
    that genuinely have no history (a single render, the first cycles of a closed loop) leave
    it ``None``; the policy then falls back to tiling and says so.
    """

    images: dict[str, np.ndarray]
    state: RobotState
    instruction: str
    image_history: dict[str, np.ndarray] | None = None


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
class FlowBackbone(BackboneAdapter, Protocol):
    """A :class:`BackboneAdapter` that also exposes the rectified-flow training pathway
    (T-16, PRD §10.3). ``JointWorldActionModel`` depends on exactly this — never on a
    concrete backbone class (FR-09).

    The flow convention is WAM's, not any backbone's: with ``x0`` noise and ``x1`` clean,
    ``x_t = (1 - t) * x0 + t * x1`` and ``v = x1 - x0``, so ``t = 1`` is clean. Each adapter
    maps ``t`` onto whatever its own scheduler expects (Wan counts denoising steps
    *downwards* from 1000, so it flips both the timestep and the velocity sign internally).
    Nothing outside the adapter may learn about that.

    "Latents" is whatever space the backbone trains in: pixels for the tiny identity VAE,
    normalized VAE latents for Wan. The video flow loss lives in that space, so its
    magnitude is **not** comparable across backbones.
    """

    def encode_video(self, video: Any) -> Any:
        """Raw frames (uint8 [B, F, H, W, 3] / [F, H, W, 3], or float) -> clean flow latents.

        This is the backbone's VAE (identity for tiny). The result is what the flow target
        is built from, so it must be deterministic and gradient-free.
        """
        ...

    def decode_video(self, video_latents: Any) -> Any:
        """Flow latents -> pixel frames [B, F, H, W, 3]. Inverse of :meth:`encode_video`."""
        ...

    def forward_flow(
        self, video_latents: Any, t: Any, text_ctx: Any, state_ctx: Any
    ) -> tuple[Any, Any]:
        """One denoising pass -> ``(velocity_pred, features)``.

        ``video_latents`` is the *noised* latent at ``t`` (scalar or [B] in [0, 1]).
        ``velocity_pred`` has EXACTLY the shape of ``video_latents``; ``features`` is
        [B, S, feature_dim] for the action branch, video tokens first.
        """
        ...

    def num_video_tokens(self, video_latents: Any) -> int:
        """How many LEADING tokens of ``features`` are video tokens.

        A method, not a property: tiny reads it off its config, Wan derives it from the
        latent geometry of the batch it was handed.
        """
        ...

    def frozen_part_names(self) -> tuple[str, ...]:
        """Attribute names of the parts frozen at construction (PRD §10.3 step 4).

        Fed into the model's frozen-parts registry; for Wan these are the VAE and the text
        tower, for tiny the text embedding tables.
        """
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
