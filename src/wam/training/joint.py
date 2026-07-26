"""Joint world-action training: co-denoise video + action latents (T-16, FR-03, PRD §10.3).

``JointWorldActionModel`` couples the backbone flow pathway with the action branch:

- video branch: rectified-flow on the backbone's video "latents" (for tiny, pixels — the
  identity VAE) via ``backbone.forward_flow`` -> velocity prediction + shared features;
- action branch: demonstrated chunks are embedded by ``ActionChunkEncoder`` (training only,
  PRD 9.5), noised with the SAME rectified-flow schedule and timestep as the video latents,
  and an ``ActionVelocityHead`` predicts the action-latent velocity from
  ``[z_t | pooled shared features | t]`` per step. Sharing one ``t`` keeps the two branches
  co-denoised (PRD §10.3); separate schedules remain an R-07 mitigation lever.
  The flow-matching input and target are built from a DETACHED copy of the action latent:
  with gradients flowing from the flow target into the encoder, AdamW can collapse the
  encoder to a constant latent c (then v_target = c - noise is an exact function of
  (z_t, t) and action_flow -> 0 while encoding zero action information). The encoder is
  instead anchored by a small reconstruction decoder (``action_recon``) that regresses
  (targets, gripper) back from the clean latent — the latent must stay action-informative.
- decoder branch: ``ActionHead`` regresses the clean chunk from the shared features (inverse
  dynamics, PRD §8.2) so smoothness/limit regularizers act in normalized action space.

Frozen-parts registry: text/VAE-equivalents are frozen for the MVP (PRD §10.3 step 4). For
the tiny backbone that is the text embedding table + text positional table (the "text
encoder"); the tiny "VAE" is the identity and has no parameters. The registry lives in
``JointWorldActionModel.frozen_parts`` and freezing happens at construction.

``JointTrainer`` optimizes the weighted sum of the PRD §10.4 loss dict:
video / action_flow / action_recon / action_reg / gripper / alignment / smoothness / limit.
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
from wam.encoders import ActionChunkEncoder, ActionChunkEncoderConfig, StateMLP, StateMLPConfig
from wam.interfaces.versioning import RunMetadata, load_config

from ._utils import (
    encode_instructions,
    iterate_batches,
    load_checkpoint_raw,
    save_checkpoint,
)
from .losses import (
    action_flow_matching_loss,
    action_regression_loss,
    alignment_loss,
    limit_penalty,
    make_flow_targets,
    smoothness_loss,
    video_flow_loss,
)
from .monitor import TrainingMonitor

__all__ = [
    "ActionVelocityHead",
    "JointLossWeights",
    "JointTrainer",
    "JointTrainingConfig",
    "JointWorldActionModel",
    "load_joint_checkpoint",
]


class JointLossWeights(BaseModel):
    """Weights of the combined joint objective (PRD §10.4; R-07 tuning surface)."""

    model_config = ConfigDict(frozen=True)

    video: float = Field(default=1.0, ge=0)
    action_flow: float = Field(default=1.0, ge=0)
    action_recon: float = Field(default=1.0, ge=0)
    action_reg: float = Field(default=1.0, ge=0)
    gripper: float = Field(default=0.5, ge=0)
    alignment: float = Field(default=0.1, ge=0)
    smoothness: float = Field(default=0.0, ge=0)
    limit: float = Field(default=0.0, ge=0)


class JointTrainingConfig(BaseModel):
    """Model + optimization config for joint video/action training. Frozen."""

    model_config = ConfigDict(frozen=True)

    state: StateMLPConfig
    backbone: TinyBackboneConfig = TinyBackboneConfig()
    action_encoder: ActionChunkEncoderConfig
    head: ActionHeadConfig
    velocity_hidden_dims: tuple[int, ...] = Field(default=(64,), min_length=1, max_length=3)

    seed: int = 0
    device: str = "cpu"
    lr: float = Field(default=3e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    grad_clip: float = Field(default=1.0, gt=0)
    batch_size: int = Field(default=16, ge=1)
    steps: int = Field(default=200, ge=1)
    weights: JointLossWeights = JointLossWeights()
    limit_margin: float = Field(default=0.95, gt=0, le=1.0)
    camera: str = "front"

    @model_validator(mode="after")
    def _consistent_dims(self) -> JointTrainingConfig:
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
        if self.action_encoder.target_dim != self.head.target_dim:
            raise ValueError(
                f"action_encoder.target_dim {self.action_encoder.target_dim} != "
                f"head.target_dim {self.head.target_dim}"
            )
        if self.action_encoder.max_steps < self.head.num_steps:
            raise ValueError(
                f"action_encoder.max_steps {self.action_encoder.max_steps} < "
                f"head.num_steps {self.head.num_steps}"
            )
        if any(h < 1 for h in self.velocity_hidden_dims):
            raise ValueError("velocity_hidden_dims entries must be >= 1")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> JointTrainingConfig:
        """Load ``configs/training/joint.yaml`` (versioned via ``load_config``)."""
        data = load_config(path)
        if "training" not in data:
            raise ValueError(f"{path}: missing top-level 'training' section")
        return cls.model_validate(data["training"])


class ActionVelocityHead(nn.Module):
    """Per-step MLP: ``[z_t | pooled features | t] -> action-latent velocity`` [B, T, L]."""

    def __init__(self, latent_dim: int, feature_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = latent_dim + feature_dim + 1  # + raw timestep as one scalar feature
        for hidden in hidden_dims:
            layers.extend([nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, latent_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z_t: Tensor, pooled_features: Tensor, t: Tensor) -> Tensor:
        """``z_t`` [B, T, L], ``pooled_features`` [B, D], ``t`` [B] -> velocity [B, T, L]."""
        batch, steps, _ = z_t.shape
        feats = pooled_features[:, None, :].expand(batch, steps, -1)
        t_col = t.reshape(batch, 1, 1).expand(batch, steps, 1).to(z_t.dtype)
        return self.mlp(torch.cat([z_t, feats, t_col], dim=-1))


class JointWorldActionModel(nn.Module):
    """Backbone flow pathway + ActionChunkEncoder + action velocity head + ActionHead."""

    #: module attribute names frozen at construction (text/VAE equivalents, PRD §10.3 step 4)
    FROZEN_PART_NAMES: tuple[str, ...] = ("text_embedding", "text_pos")

    def __init__(self, config: JointTrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = TinyVideoBackbone(config.backbone)
        self.state_encoder = StateMLP(config.state)
        self.action_encoder = ActionChunkEncoder(config.action_encoder)  # training only
        self.velocity_head = ActionVelocityHead(
            config.action_encoder.latent_dim,
            config.backbone.feature_dim,
            config.velocity_hidden_dims,
        )
        self.action_head = ActionHead(config.head)
        self.align_proj = nn.Linear(config.action_encoder.latent_dim, config.backbone.feature_dim)
        # Reconstruction anchor for the action encoder: latent [B, T, L] -> (targets, gripper)
        # per step. Without it (and with detached flow targets) nothing would prevent the
        # encoder latent from collapsing to an input-independent constant.
        enc = config.action_encoder
        recon_hidden = enc.hidden_dims[-1]
        self.action_recon = nn.Sequential(
            nn.Linear(enc.latent_dim, recon_hidden),
            nn.GELU(),
            nn.Linear(recon_hidden, enc.target_dim + enc.gripper_dims),
        )
        # Frozen-parts registry: name -> parameter-bearing module/parameter, frozen in place.
        self.frozen_parts: dict[str, nn.Module | nn.Parameter] = {
            name: getattr(self.backbone, name) for name in self.FROZEN_PART_NAMES
        }
        for part in self.frozen_parts.values():
            if isinstance(part, nn.Parameter):
                part.requires_grad_(False)
            else:
                for param in part.parameters():
                    param.requires_grad_(False)

    def frozen_parameter_names(self) -> tuple[str, ...]:
        """Fully-qualified names of all frozen parameters (for tests/audits)."""
        return tuple(name for name, param in self.named_parameters() if not param.requires_grad)

    def co_denoise(
        self,
        batch: Mapping[str, Any],
        t: Tensor,
        *,
        video_noise: Tensor | None = None,
        action_noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        """One co-denoising pass with a shared timestep ``t`` [B] in [0, 1].

        Returns velocity predictions + targets for both branches, the shared features, the
        decoded chunk and the pooled alignment features (see module docstring).
        """
        frames = torch.as_tensor(batch["frames"])
        if frames.dtype == torch.uint8:
            video_clean = frames.float() / 255.0
        else:
            video_clean = frames.float()
        if video_clean.ndim == 4:
            video_clean = video_clean[None]
        batch_size = video_clean.shape[0]

        targets = torch.as_tensor(batch["targets"], dtype=torch.float32)
        gripper = torch.as_tensor(batch["gripper_target"], dtype=torch.float32)
        if gripper.ndim == 2:  # [B, T] scalar command -> broadcast over gripper_dims
            gripper = gripper.unsqueeze(-1).expand(-1, -1, self.config.action_encoder.gripper_dims)
        action_clean = self.action_encoder(targets, gripper)  # [B, T, L]
        # DETACHED for flow matching: the action_flow loss trains only the velocity head.
        # Gradients from the flow target into the encoder open a collapse shortcut (see
        # module docstring); the encoder is trained by action_recon (+ alignment) instead.
        action_latent = action_clean.detach()

        if video_noise is None:
            video_noise = torch.randn(
                video_clean.shape, generator=generator, dtype=video_clean.dtype
            ).to(video_clean.device)
        if action_noise is None:
            action_noise = torch.randn(
                action_latent.shape, generator=generator, dtype=action_latent.dtype
            ).to(action_latent.device)

        video_t, video_velocity = make_flow_targets(video_noise, video_clean, t)
        action_t, action_velocity = make_flow_targets(action_noise, action_latent, t)

        state_batch = {k: batch[k] for k in ("q", "dq", "imu", "gripper") if k in batch}
        if "validity" in batch and batch["validity"] is not None:
            state_batch["validity"] = batch["validity"]
        state_emb = self.state_encoder(state_batch)
        text_ctx = encode_instructions(self.backbone, batch.get("instruction", ""), batch_size)
        state_ctx = self.backbone.condition_state(state_emb)

        video_velocity_pred, features = self.backbone.forward_flow(video_t, t, text_ctx, state_ctx)
        pooled = features.mean(dim=1)  # [B, D] shared conditioning for the action branch
        action_velocity_pred = self.velocity_head(action_t, pooled, t)
        decoded = self.action_head(pooled)
        recon = self.action_recon(action_clean)  # [B, T, D+G] — the encoder's anchor
        target_dim = self.config.action_encoder.target_dim

        num_video = self.config.backbone.num_video_tokens
        return {
            "video_velocity_pred": video_velocity_pred,
            "video_velocity_target": video_velocity,
            "action_velocity_pred": action_velocity_pred,
            "action_velocity_target": action_velocity,
            "features": features,
            "decoded_targets": decoded["targets"],
            "decoded_gripper": decoded["gripper"],
            "recon_targets": recon[..., :target_dim],
            "recon_gripper": recon[..., target_dim:],
            "video_feature": features[:, :num_video].mean(dim=1),  # [B, D]
            "action_feature": self.align_proj(action_clean.mean(dim=1)),  # [B, D]
        }


class JointTrainer:
    """Seeded AdamW loop over the combined weighted loss dict (T-16/T-17)."""

    def __init__(self, config: JointTrainingConfig, model: JointWorldActionModel | None = None):
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        self.model = (model or JointWorldActionModel(config)).to(self.device)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=config.lr, weight_decay=config.weight_decay
        )
        self._rng = torch.Generator().manual_seed(config.seed)
        self.metadata: RunMetadata | None = None

    def compute_losses(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        """Weighted PRD §10.4 loss dict incl. ``'total'`` for one batch."""
        cfg = self.config
        batch_size = torch.as_tensor(batch["frames"]).shape[0]
        t = torch.rand(batch_size, generator=self._rng).to(self.device)
        out = self.model.co_denoise(batch, t, generator=self._rng)

        targets = torch.as_tensor(batch["targets"], dtype=torch.float32, device=self.device)
        gripper = torch.as_tensor(batch["gripper_target"], dtype=torch.float32, device=self.device)
        if gripper.ndim == 2:
            gripper = gripper.unsqueeze(-1).expand_as(out["decoded_gripper"])

        losses = {
            "video": video_flow_loss(out["video_velocity_pred"], out["video_velocity_target"]),
            "action_flow": action_flow_matching_loss(
                out["action_velocity_pred"], out["action_velocity_target"]
            ),
            # Encoder anchor: reconstruct the demonstrated chunk from the action latent
            # (the flow branch is detached from the encoder — see co_denoise).
            "action_recon": (
                action_regression_loss(out["recon_targets"], targets)
                + action_regression_loss(out["recon_gripper"], gripper)
            ),
            "action_reg": action_regression_loss(out["decoded_targets"], targets),
            "gripper": action_regression_loss(out["decoded_gripper"], gripper),
            "alignment": alignment_loss(out["video_feature"], out["action_feature"]),
            "smoothness": smoothness_loss(out["decoded_targets"]),
            "limit": limit_penalty(out["decoded_targets"], limit=cfg.limit_margin),
        }
        w = cfg.weights
        losses["total"] = (
            w.video * losses["video"]
            + w.action_flow * losses["action_flow"]
            + w.action_recon * losses["action_recon"]
            + w.action_reg * losses["action_reg"]
            + w.gripper * losses["gripper"]
            + w.alignment * losses["alignment"]
            + w.smoothness * losses["smoothness"]
            + w.limit * losses["limit"]
        )
        return losses

    def train(
        self,
        data: Dataset | Mapping[str, Any],
        steps: int | None = None,
        *,
        monitor: TrainingMonitor | None = None,
    ) -> list[dict[str, float]]:
        """Run ``steps`` optimizer steps (default ``config.steps``); returns per-step history."""
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
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.grad_clip,
                )
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

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        run_id: str = "joint",
        dataset_snapshot_ref: str | None = None,
        git_commit: str | None = None,
    ) -> RunMetadata:
        """Write safetensors weights + embedded config and RunMetadata (FR-10, AC-04)."""
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
    def load_checkpoint(cls, path: str | Path) -> JointTrainer:
        """Rebuild a trainer (model + config) from a checkpoint; weights are bit-exact."""
        model, metadata = load_joint_checkpoint(path)
        trainer = cls(model.config, model=model)
        trainer.metadata = metadata
        return trainer


def load_joint_checkpoint(path: str | Path) -> tuple[JointWorldActionModel, RunMetadata]:
    """Load ``(JointWorldActionModel, RunMetadata)`` from a safetensors checkpoint."""
    state_dict, config_dict, metadata = load_checkpoint_raw(path)
    config = JointTrainingConfig.model_validate(config_dict)
    model = JointWorldActionModel(config)
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata
