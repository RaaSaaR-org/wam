"""Action-only baseline policy + trainer (T-13, PRD §10.3 step 3, M2 overfit gate).

``ActionOnlyModel`` composes the M0/M2 building blocks — ``StateMLP`` (FR-02),
``TinyVideoBackbone`` features (FR-09) and ``ActionHead`` (FR-04) — into a policy that
regresses action chunks directly from (frames, instruction, state). No video prediction:
this is the baseline the world-action model must beat (AC-07).

``ActionOnlyTrainer`` is a standard supervised loop: AdamW, gradient clipping, seeded and
deterministic on CPU. ``overfit()`` is the D1 go/no-go gate helper (P6: overfit first).
Checkpoints are safetensors files with the config and a ``RunMetadata`` (config_hash,
checkpoint_ref) embedded — every artifact is traceable (FR-10, AC-04).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn
from torch.utils.data import Dataset

from wam.backbones.tiny import TinyBackboneConfig, TinyVideoBackbone
from wam.decoders import ActionHead, ActionHeadConfig
from wam.encoders import StateMLP, StateMLPConfig
from wam.interfaces.protocols import Observation
from wam.interfaces.schema import ActionChunk
from wam.interfaces.versioning import RunMetadata, load_config

from ._utils import (
    encode_instructions,
    iterate_batches,
    load_checkpoint_raw,
    resolve_frame_context,
    save_checkpoint,
)
from .losses import action_regression_loss, limit_penalty, smoothness_loss
from .monitor import TrainingMonitor

__all__ = [
    "ActionLossWeights",
    "ActionOnlyConfig",
    "ActionOnlyModel",
    "ActionOnlyTrainer",
    "load_action_only_checkpoint",
]


class ActionLossWeights(BaseModel):
    """Weights of the action-only objective terms (PRD §10.4)."""

    model_config = ConfigDict(frozen=True)

    action: float = Field(default=1.0, ge=0)
    gripper: float = Field(default=0.5, ge=0)
    smoothness: float = Field(default=0.0, ge=0)
    limit: float = Field(default=0.0, ge=0)


class ActionOnlyConfig(BaseModel):
    """Model + optimization config for the action-only baseline. Frozen (config_hash-able)."""

    model_config = ConfigDict(frozen=True)

    state: StateMLPConfig
    backbone: TinyBackboneConfig = TinyBackboneConfig()
    head: ActionHeadConfig

    seed: int = 0
    device: str = "cpu"
    lr: float = Field(default=3e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    grad_clip: float = Field(default=1.0, gt=0)
    batch_size: int = Field(default=16, ge=1)
    steps: int = Field(default=300, ge=1)
    loss: str = Field(default="l2", pattern="^(l1|l2)$")
    weights: ActionLossWeights = ActionLossWeights()
    limit_margin: float = Field(default=0.95, gt=0, le=1.0)
    camera: str = "front"

    @model_validator(mode="after")
    def _consistent_dims(self) -> ActionOnlyConfig:
        if self.backbone.state_embedding_dim != self.state.embedding_dim:
            raise ValueError(
                f"backbone.state_embedding_dim {self.backbone.state_embedding_dim} != "
                f"state.embedding_dim {self.state.embedding_dim}"
            )
        if self.head.feature_dim != self.backbone.feature_dim:
            raise ValueError(
                f"head.feature_dim {self.head.feature_dim} != "
                f"backbone.feature_dim {self.backbone.feature_dim}"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ActionOnlyConfig:
        """Load ``configs/training/action_only.yaml`` (versioned via ``load_config``)."""
        data = load_config(path)
        if "training" not in data:
            raise ValueError(f"{path}: missing top-level 'training' section")
        return cls.model_validate(data["training"])


class ActionOnlyModel(nn.Module):
    """StateMLP + TinyVideoBackbone features + ActionHead; implements the Policy protocol."""

    def __init__(self, config: ActionOnlyConfig) -> None:
        super().__init__()
        self.config = config
        self.state_encoder = StateMLP(config.state)
        self.backbone = TinyVideoBackbone(config.backbone)
        self.action_head = ActionHead(config.head)

    def pooled_features(self, batch: Mapping[str, Any]) -> Tensor:
        """(frames, instruction, state) -> mean-pooled backbone features [B, feature_dim]."""
        frames = torch.as_tensor(batch["frames"])
        state_batch = {k: batch[k] for k in ("q", "dq", "imu", "gripper") if k in batch}
        if "validity" in batch and batch["validity"] is not None:
            state_batch["validity"] = batch["validity"]
        state_emb = self.state_encoder(state_batch)  # [B, E]
        video_ctx = self.backbone.condition_video(frames)
        text_ctx = encode_instructions(
            self.backbone, batch.get("instruction", ""), state_emb.shape[0]
        )
        state_ctx = self.backbone.condition_state(state_emb)
        features = self.backbone.features(video_ctx, text_ctx, state_ctx)  # [B, S, D]
        return features.mean(dim=1)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        """Training forward -> ``{"targets": [B, T, D], "gripper": [B, T, G]}``."""
        return self.action_head(self.pooled_features(batch))

    @torch.no_grad()
    def predict(self, observation: Observation, *, camera: str | None = None) -> ActionChunk:
        """Policy protocol: one Observation -> one canonical ActionChunk.

        Frames come from :func:`~wam.training._utils.resolve_frame_context` — the real window if
        ``observation.image_history`` has one, else the single frame tiled to ``num_frames``.
        ``camera`` overrides the trained ``config.camera`` for a deployment that names the same
        view differently (sim renders ``head``, the converted episodes trained on ``ego``) —
        same override, and the same frame resolution, as ``JointWorldActionModel.predict``, so an
        AC-07 comparison can put both models on the identical observation stream.
        """
        camera = camera if camera is not None else self.config.camera
        frames = resolve_frame_context(observation, camera, self.config.backbone.num_frames)
        state_emb = self.state_encoder.encode(observation.state)  # [E]
        video_ctx = self.backbone.condition_video(frames.unsqueeze(0))
        text_ctx = self.backbone.condition_text(observation.instruction)
        state_ctx = self.backbone.condition_state(state_emb)
        features = self.backbone.features(video_ctx, text_ctx, state_ctx)
        return self.action_head.decode(features[0])


class ActionOnlyTrainer:
    """Seeded AdamW loop for :class:`ActionOnlyModel` with monitoring + checkpoints."""

    def __init__(self, config: ActionOnlyConfig, model: ActionOnlyModel | None = None) -> None:
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        self.model = (model or ActionOnlyModel(config)).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.metadata: RunMetadata | None = None  # set by load_checkpoint

    # -- loss --------------------------------------------------------------------------

    def compute_losses(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        """Weighted loss dict incl. ``'total'``; targets/gripper regression + regularizers."""
        out = self.model(batch)
        cfg = self.config
        targets = torch.as_tensor(batch["targets"], dtype=torch.float32, device=self.device)
        gripper = torch.as_tensor(batch["gripper_target"], dtype=torch.float32, device=self.device)
        if gripper.ndim == 2:  # [B, T] scalar command -> broadcast over gripper_dims
            gripper = gripper.unsqueeze(-1).expand_as(out["gripper"])
        losses = {
            "action": action_regression_loss(out["targets"], targets, kind=cfg.loss),
            "gripper": action_regression_loss(out["gripper"], gripper, kind=cfg.loss),
            "smoothness": smoothness_loss(out["targets"]),
            "limit": limit_penalty(out["targets"], limit=cfg.limit_margin),
        }
        w = cfg.weights
        losses["total"] = (
            w.action * losses["action"]
            + w.gripper * losses["gripper"]
            + w.smoothness * losses["smoothness"]
            + w.limit * losses["limit"]
        )
        return losses

    # -- training ----------------------------------------------------------------------

    def train(
        self,
        data: Dataset | Mapping[str, Any],
        steps: int | None = None,
        *,
        monitor: TrainingMonitor | None = None,
    ) -> list[dict[str, float]]:
        """Run ``steps`` optimizer steps (default ``config.steps``); returns per-step history.

        ``data`` is either an ``EpisodeDataset``-style torch Dataset or a full tensor batch
        (mapping with frames/q/dq/imu/gripper/targets/gripper_target[/validity/instruction]).
        A ``TrainingMonitor`` may raise ``TrainingDiverged`` mid-run.
        """
        num_steps = steps if steps is not None else self.config.steps
        self.model.train()
        history: list[dict[str, float]] = []
        batches = iterate_batches(
            data,
            steps=num_steps,
            batch_size=self.config.batch_size,
            seed=self.config.seed,
            device=self.device,
        )
        for step, batch in enumerate(batches):
            snapshot = TrainingMonitor.snapshot_params(self.model) if monitor else None
            losses = self.compute_losses(batch)
            self.optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            )
            self.optimizer.step()

            entry = {name: float(value.detach()) for name, value in losses.items()}
            entry["step"] = float(step)
            entry["grad_norm"] = grad_norm
            history.append(entry)
            if monitor is not None:
                monitor.record_step(
                    step,
                    {k: v for k, v in entry.items() if k not in ("step", "grad_norm")},
                    grad_norms=TrainingMonitor.module_grad_norms(self.model),
                    update_ratio=TrainingMonitor.param_update_ratio(self.model, snapshot or {}),
                )
        return history

    def overfit(
        self,
        data: Dataset | Mapping[str, Any],
        steps: int | None = None,
        target_loss: float = 1e-2,
    ) -> bool:
        """D1 overfit gate (T-13, P6): train and check final total loss <= ``target_loss``."""
        history = self.train(data, steps)
        return bool(history) and history[-1]["total"] <= target_loss

    # -- checkpointing (FR-10, AC-04) ----------------------------------------------------

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        run_id: str = "action_only",
        dataset_snapshot_ref: str | None = None,
        git_commit: str | None = None,
    ) -> RunMetadata:
        """Write safetensors weights + embedded config and RunMetadata; returns the metadata."""
        metadata = RunMetadata.create(
            run_id,
            self.config,
            checkpoint_ref=str(Path(path)),
            dataset_snapshot_ref=dataset_snapshot_ref,
            git_commit=git_commit,
        )
        save_checkpoint(self.model, self.config, path, metadata)
        self.metadata = metadata
        return metadata

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> ActionOnlyTrainer:
        """Rebuild a trainer (model + config) from a checkpoint; weights are bit-exact."""
        model, metadata = load_action_only_checkpoint(path)
        trainer = cls(model.config, model=model)
        trainer.metadata = metadata
        return trainer


def load_action_only_checkpoint(path: str | Path) -> tuple[ActionOnlyModel, RunMetadata]:
    """Load ``(ActionOnlyModel, RunMetadata)`` from a safetensors checkpoint (bit-exact)."""
    state_dict, config_dict, metadata = load_checkpoint_raw(path)
    config = ActionOnlyConfig.model_validate(config_dict)
    model = ActionOnlyModel(config)
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata
