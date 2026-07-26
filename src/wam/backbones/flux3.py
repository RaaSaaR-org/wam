"""FLUX 3 Dev backbone adapter STUB (M5, OD-06 — access/weights/license pending).

Status: protocol-conformant placeholder. Every method except ``name``/``feature_dim`` raises
``NotImplementedError('FLUX 3 access pending — OD-06')``. Kept in-tree so the registry, the
swap tests (FR-09/AC-05) and downstream code exercise the third backbone name today.

Planned integration surface (per FLUX 3 announcement [R3] and FLUX 3 x mimic recipe [R4]):

- ``condition_video()``: past frames through the FLUX 3 video VAE (frozen) -> video latents.
- ``condition_text()``: FLUX 3 native text tower (frozen) -> text context tokens.
- ``condition_state()``: linear projection of the StateEncoder embedding into the FLUX token
  stream (trainable robotics adapter).
- ``features()``: intermediate activations of selected FLUX video-path blocks,
  ``[B, S, feature_dim]`` — mimic-style: the action decoder reads video-path intermediate
  features, never final pixels.
- ``feature_dim`` default (4096) is a PLACEHOLDER until OD-06 fixes the real channel width;
  it is constructor-overridable so downstream shape plumbing can be exercised.

Swapping this adapter in must not change the data schema or the robot API (FR-09/AC-05).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

FLUX3_NAME = "flux3-dev"
FLUX3_PLACEHOLDER_FEATURE_DIM = 4096

_PENDING_MSG = "FLUX 3 access pending — OD-06"


class Flux3Adapter:
    """``BackboneAdapter``-conformant stub for FLUX 3 Dev (preferred backbone, M5)."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        feature_dim: int = FLUX3_PLACEHOLDER_FEATURE_DIM,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self._feature_dim = int(feature_dim)

    @property
    def name(self) -> str:
        return FLUX3_NAME

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def condition_video(self, video: Any) -> Any:
        raise NotImplementedError(_PENDING_MSG)

    def condition_text(self, text: str) -> Any:
        raise NotImplementedError(_PENDING_MSG)

    def condition_state(self, state_embedding: Any) -> Any:
        raise NotImplementedError(_PENDING_MSG)

    def features(self, video_ctx: Any, text_ctx: Any, state_ctx: Any) -> Any:
        raise NotImplementedError(_PENDING_MSG)
