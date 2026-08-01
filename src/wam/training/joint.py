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

Two readouts, one deployed by default (T-30, ``docs/improvements.md`` I-3). ``predict`` uses the
regression branch (``ActionHead``); ``predict(..., flow_steps=n)`` instead integrates the trained
flow branch from noise and decodes the sampled latent through ``action_recon``. That promotes
``action_recon`` from "the encoder's collapse anchor" to "the only latent->action decoder that
exists", which is why its attribute name and its ``nn.Sequential`` layout are frozen: every
archived checkpoint carries the state-dict keys ``action_recon.0.*`` / ``action_recon.2.*``, and
renaming or reshaping the module strands ``runs/t16-lora-seed0`` rather than merely breaking a
test. The flow readout is OFF by default so every number recorded before it existed reproduces.
Sampling it from pure noise evaluates the velocity head outside the (features, t) pairing it was
trained on, so ``sample_action_chunk`` also carries the two control arms the T-30 rule is keyed
on — ``mean_of`` (k draws averaged, the MSE-fair comparison against a mean-seeking head) and
``t0`` (start the integration inside the region where the two branches' timesteps agree).

The model is backbone-agnostic (FR-09): it depends on the ``FlowBackbone`` protocol and never
on a concrete class. The backbone is either built from ``config.backbone`` — a discriminated
union tagged by ``kind`` — or injected ready-made, which is how a multi-GB adapted DiT is
loaded once and reused. Everything backbone-shaped is asked of the instance: the video latent
space (``encode_video``), how many leading feature tokens are video (``num_video_tokens``) and
which parts to freeze (``frozen_part_names``).

Frozen-parts registry: text/VAE-equivalents are frozen for the MVP (PRD §10.3 step 4). The
backbone names them — for tiny the text embedding table + text positional table (its "VAE" is
the identity and has no parameters), for Wan the VAE and the text tower. The registry lives in
``JointWorldActionModel.frozen_parts`` and freezing happens at construction.

``JointTrainer`` optimizes the weighted sum of the PRD §10.4 loss dict:
video / action_flow / action_recon / action_reg / gripper / alignment / smoothness / limit.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn
from torch.utils.data import Dataset

from wam.backbones.registry import build_backbone
from wam.backbones.tiny import TinyBackboneConfig
from wam.backbones.wan_i2v import WanBackboneConfig
from wam.decoders import ActionHead, ActionHeadConfig
from wam.encoders import ActionChunkEncoder, ActionChunkEncoderConfig, StateMLP, StateMLPConfig
from wam.interfaces.protocols import FlowBackbone, Observation
from wam.interfaces.schema import ActionChunk
from wam.interfaces.versioning import RunMetadata, load_config

from ._utils import (
    encode_instructions,
    iterate_batches,
    load_checkpoint_raw,
    resolve_frame_context,
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
    "BackboneConfig",
    "JointLossWeights",
    "JointTrainer",
    "JointTrainingConfig",
    "JointWorldActionModel",
    "build_action_recon",
    "load_joint_checkpoint",
]

#: Tagged union of every trainable backbone config (FR-09). The ``kind`` discriminator is what
#: keeps a Wan section from validating as an all-defaults tiny config; ``build_backbone``
#: dispatches on the same tag, so adding a backbone touches exactly this alias and the registry.
BackboneConfig = Annotated[TinyBackboneConfig | WanBackboneConfig, Field(discriminator="kind")]


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
    backbone: BackboneConfig = TinyBackboneConfig()
    action_encoder: ActionChunkEncoderConfig
    head: ActionHeadConfig
    velocity_hidden_dims: tuple[int, ...] = Field(default=(64,), min_length=1, max_length=3)

    seed: int = 0
    device: str = "cpu"
    lr: float = Field(default=3e-3, gt=0)
    #: Separate LR for everything under ``backbone.``; ``None`` == one group at ``lr``. An
    #: adapted large backbone wants a far smaller LR than freshly-initialized heads.
    backbone_lr: float | None = Field(default=None, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    grad_clip: float = Field(default=1.0, gt=0)
    batch_size: int = Field(default=16, ge=1)
    steps: int = Field(default=200, ge=1)
    weights: JointLossWeights = JointLossWeights()
    limit_margin: float = Field(default=0.95, gt=0, le=1.0)
    camera: str = "front"

    @model_validator(mode="before")
    @classmethod
    def _tag_legacy_backbone(cls, data: Any) -> Any:
        """Tag an untagged ``backbone`` section as ``kind='tiny'``.

        The discriminated union would otherwise reject every artifact written before it
        existed: the shipped ``configs/training/*.yaml`` and the config dict embedded in every
        checkpoint on disk. Tagging on the way in keeps those loading bit-identically while new
        configs must name their backbone explicitly.
        """
        if isinstance(data, Mapping):
            backbone = data.get("backbone")
            if isinstance(backbone, Mapping) and "kind" not in backbone:
                return {**data, "backbone": {**backbone, "kind": "tiny"}}
        return data

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


def build_action_recon(config: ActionChunkEncoderConfig) -> nn.Sequential:
    """The latent -> (targets, gripper) decoder: ``[B, T, L] -> [B, T, D + G]``.

    A free function, not an inline block in ``__init__``, because two places need the identical
    module: the model, and ``scripts/check_action_latent.py``, which rebuilds it from a
    checkpoint's embedded config WITHOUT constructing a backbone (that would mean loading tens of
    GB of Wan weights to run an MLP). Two hand-copied definitions would drift, and the drift would
    show up as a mysteriously wrong reconstruction rather than as a load error.

    The layout is frozen: ``Linear -> GELU -> Linear`` gives every archived checkpoint the
    state-dict keys ``action_recon.0.*`` and ``action_recon.2.*``. Inserting a layer renumbers
    them and strands every checkpoint written before the change.
    """
    hidden = config.hidden_dims[-1]
    return nn.Sequential(
        nn.Linear(config.latent_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, config.target_dim + config.gripper_dims),
    )


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

    #: Fallback frozen-part names for a backbone that predates ``frozen_part_names`` (tiny's
    #: text embedding + text positional table, PRD §10.3 step 4). The backbone is authoritative.
    FROZEN_PART_NAMES: tuple[str, ...] = ("text_embedding", "text_pos")

    def __init__(self, config: JointTrainingConfig, backbone: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        # Injection point for an already-loaded backbone: a multi-GB DiT is built and adapted
        # once by the caller and handed in, instead of being rebuilt per model instance.
        resolved = backbone if backbone is not None else build_backbone(config.backbone)
        if not isinstance(resolved, nn.Module):
            raise TypeError(
                f"backbone must be an nn.Module, got {type(resolved).__name__}: the joint model "
                "registers it as a submodule, so a plain object would leave its parameters out "
                "of .parameters() (nothing to optimize), out of .state_dict() (nothing "
                "checkpointed) and out of .to(device) (silent CPU/GPU split)"
            )
        if not isinstance(resolved, FlowBackbone):
            missing = sorted(
                name
                for name in getattr(FlowBackbone, "__protocol_attrs__", ())
                if not hasattr(resolved, name)
            )
            raise TypeError(
                f"backbone {type(resolved).__name__} does not implement FlowBackbone; "
                f"missing: {missing or 'unknown (protocol members unavailable)'}"
            )
        self.backbone = resolved
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
        # encoder latent from collapsing to an input-independent constant. Since T-30 it is also
        # the decoder of the optional flow readout (see sample_action_chunk).
        self.action_recon = build_action_recon(config.action_encoder)
        # Frozen-parts registry: name -> parameter-bearing module/parameter, frozen in place.
        # The BACKBONE names its own frozen parts (tiny: text tables; Wan: VAE + text tower) —
        # the model must not know which those are (FR-09). An empty tuple is legitimate: a
        # backbone that arrives pre-frozen (LoRA-adapted, base weights already requires_grad=
        # False) has nothing left for us to freeze.
        part_names = tuple(
            getattr(self.backbone, "frozen_part_names", lambda: self.FROZEN_PART_NAMES)()
        )
        self.frozen_parts: dict[str, nn.Module | nn.Parameter] = {
            name: getattr(self.backbone, name) for name in part_names
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

    def trainable_state_dict(self) -> dict[str, Tensor]:
        """``state_dict`` restricted to parameters with ``requires_grad``.

        The checkpoint payload for an adapted large backbone: LoRA/adapter tensors and heads
        only, so a run does not write the frozen base weights into every checkpoint. Reload it
        with ``strict=False`` on top of the same base weights.
        """
        trainable = {name for name, param in self.named_parameters() if param.requires_grad}
        return {name: value for name, value in self.state_dict().items() if name in trainable}

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
        # The backbone owns the pixel->latent step (identity for tiny, a real VAE otherwise),
        # so the flow target is built in whatever space that backbone trains in. Batching and
        # uint8 scaling are its business too — nothing here may assume a pixel layout.
        video_clean = self.backbone.encode_video(batch["frames"])
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
        # A real backbone runs in bf16/fp16; the loss and the action branch stay in fp32 so the
        # small action terms are not rounded away against the video term (R-07). No-ops on fp32.
        video_velocity_pred = video_velocity_pred.float()
        features = features.float()
        pooled = features.mean(dim=1)  # [B, D] shared conditioning for the action branch
        action_velocity_pred = self.velocity_head(action_t, pooled, t)
        decoded = self.action_head(pooled)
        recon = self.action_recon(action_clean)  # [B, T, D+G] — the encoder's anchor
        target_dim = self.config.action_encoder.target_dim

        # Asked of the backbone, not the config: Wan derives the count from the latent geometry
        # of THIS batch, tiny reads it off its config.
        num_video = self.backbone.num_video_tokens(video_t)
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

    @torch.no_grad()
    def sample_action_chunk(
        self,
        pooled: Tensor,
        *,
        steps: int,
        seed: int = 0,
        mean_of: int = 1,
        t0: float = 0.0,
        init_latent: Tensor | None = None,
    ) -> ActionChunk:
        """Sample a chunk from the trained FLOW branch instead of regressing it (T-30, I-3).

        The deployed readout regresses the whole chunk in one shot from the pooled features, and
        a single-shot L2 regressor under a one-to-many conditional is mean-seeking: it emits the
        average of the futures compatible with the observation, which is smoother and smaller
        than any of them. Measured on ``runs/t16-lora-seed0/checkpoints/step-020000`` over the
        1 040 archived holdout chunks: the regression head's chunks have RMS 0.00226 against the
        demonstrations' 0.00404 (a 44 % magnitude shortfall) and ``smoothness_ratio`` 0.29.

        That pair is **consistent with** mean-seeking. It is not a signature of it, and an earlier
        version of this docstring said it was. What the numbers actually rule out is a
        bounded-output artifact: max |target| is 0.0192 against a ``tanh``, and ``limit_penalty``
        only bites outside ±0.95, so neither bound is active. Two alternatives survive, and each
        produces small, smooth chunks on its own:

        * **the one-sided jerk regulariser.** ``configs/training/joint_wan_gr00t.yaml`` sets
          ``weights.smoothness = 0.01`` and :meth:`JointTrainer.compute_losses` applies
          ``smoothness_loss`` to ``decoded_targets`` — the regression head's output — and to
          nothing else. The flow branch is never charged for jerk. A smoothness gap between the
          two readouts is therefore the expected consequence of the objective, whether or not
          either head models the conditional.
        * **L2 shrinkage.** ``action_reg`` is an MSE against targets whose own RMS is 0.004 and
          AdamW runs with ``weight_decay``; a head that under-shoots magnitude uniformly is
          indistinguishable, on RMS and jerk alone, from one that averages modes.

        Telling those apart needs an intervention (drop ``weights.smoothness`` to 0 and retrain,
        or score a multi-modal task), not this readout swap. What this readout swap can do is ask
        whether the branch that was trained as a distribution carries anything the regression
        head does not: integrate ``velocity_head``, decode through ``action_recon``.

        **Integration direction — get this wrong and the sampler silently emits noise that still
        decodes to a finite, plausible-looking chunk.** WAM's convention lives in one place,
        :func:`~wam.training.losses.make_flow_targets`: ``x_t = (1-t) x0 + t x1`` with ``x0`` the
        NOISE and ``x1`` the clean latent, so **t=0 is noise and t=1 is clean** (the inverse of
        the diffusers/SD3 convention). Sampling therefore starts at ``z ~ N(0, I)`` and steps
        FORWARD, ``z <- z + v * dt``, with t ascending.

        The t-grid is ``{t0, t0 + dt, ..., 1 - dt}`` for ``dt = (1 - t0)/n`` and deliberately
        never reaches 1.0: :meth:`JointTrainer.compute_losses` draws t from ``torch.rand``, whose
        support is [0, 1), so t=1.0 is a timestep the head has never once been evaluated at.

        **T-30 job 184670 tripped the sampler-broken guard on every arm; the guard was wrong.**
        Audited 2026-08-01 against the training path, and this loop is faithful to it — the
        invariants are now pinned by ``TestFlowSamplerAgainstTheTrainingPath`` rather than argued:
        the z it hands the head at each t_k is bit-for-bit ``make_flow_targets(noise, clean,
        t_k)[0]``, the pooled vector :meth:`predict` passes is bit-for-bit the one
        :meth:`co_denoise` builds, and a head that returns ``co_denoise``'s own
        ``action_velocity_target`` drives this loop onto the demonstrated latent. On the archived
        checkpoint ``action_recon(action_encoder(chunk))`` scores 8.10e-07 — the pre-registered
        ceiling — so the decode side is exact too. What the arms measured is the FIELD, and the
        budget it has to work against. A chunk latent has RMS 1.42, but essentially all of it is
        the step-index positional embedding, identical for every chunk; the part that varies with
        the demonstrated chunk — the whole content the sampler has to hit — has per-element std
        0.049, against the N(0, I) the flow starts from. Fitting the archived seed0-vs-seed1
        spread back through ``action_recon`` puts the noise surviving 32 steps at ~0.10: the
        integration removes ~90 % of it, and what is left is still 2x the content, which is where
        RMS/demo 4.38 comes from. The guard's "RMS > 3x the demonstrations'" crosses at a residual
        of ~0.085 — it demands >91.5 % noise removal and reads the missing 1.5 points as an
        integration bug. ``ActionVelocityHead`` takes t as ONE raw scalar beside 32 latent and
        3072 feature inputs (first-layer weight-block Frobenius norms on the archived checkpoint:
        1.68 for t, 23.3 for the latent, 51.0 for the features), and its measured d(v)/d(z) gain
        comes out FLAT in t at every feature scale probed, where a straight path needs 1/(1-t)
        (1 at t=0, 33 at t=0.97). A constant-gain linear field contracts noise by a fixed factor
        and cannot reach zero at any step count, which is also why the sweep converges. Fixing
        that is a head/objective change (timestep embedding, or a step index the head can read),
        not a sampler one.

        **The conditioning mismatch this sampler runs on, and the arm that measures it.**
        Training only ever paired (features from video noised to t, action latent noised to the
        SAME t) — :meth:`co_denoise` shares one ``t`` across both branches. :meth:`predict`
        computes ONE backbone pass at t=1 on the CLEAN observation and this loop reuses those
        features at every t_k, so at small t_k the head is asked about (near-pure-noise z,
        clean-video features, t≈0) — a combination it was never trained on. The faithful
        alternative costs n backbone passes AND requires noising the observed frame, which
        destroys the observation the readout exists to read. It is therefore not run, and a
        negative result from the default ``t0=0`` path is a statement about the flow branch **as
        sampled this way**, never about the flow branch as trained (same shape of confound as
        I-7, which is why the T-30 pre-registration names it as its own branch).

        ``t0`` is the control for exactly that. With ``t0 > 0`` and ``init_latent`` the
        integration starts at t0 from ``(1-t0)·noise + t0·init_latent``, i.e. only in the region
        where the head's ``t`` and the features' ``t=1`` roughly agree. :meth:`predict` builds
        ``init_latent`` by encoding the REGRESSION head's own chunk through ``action_encoder`` —
        the only clean-latent estimate available at inference (the demonstrated chunk is not, and
        starting from it would make this an oracle, not a readout). The warm-started arm
        therefore INHERITS the regression mean and cannot on its own show that the flow branch
        models the conditional; what it can do is separate "the branch is dead" from "the branch
        was sampled outside its training region", which a ``t0=0`` negative cannot tell apart.

        ``mean_of`` averages k independent draws (seeds ``seed .. seed+k-1``) in ACTION space
        before the chunk is returned. It exists because the headline metric is MSE-derived and
        MSE structurally penalises sampling: for any calibrated conditional,
        ``E‖a - draw‖² = E‖a - mean‖² + E‖draw - mean‖²``, so a single unbiased draw scores worse
        than the conditional mean by exactly the conditional variance — the metric would charge
        the sampler for the one property the regression head is suspected of lacking. k draws
        leave 1/k of that penalty, which makes ``mean_of=k`` the apples-to-apples MSE comparison
        against the regression head, while ``mean_of=1`` stays the arm that measures what a
        deployed sampler would emit (magnitude, jerk, draw-to-draw spread).

        Two range decisions, both deliberate. The gripper is clamped into [0, 1] **after** the
        draws are averaged, because ``action_recon`` ends in a bare ``nn.Linear`` where
        ``ActionHead`` has a sigmoid, and :meth:`ActionChunk.validate` rejects a gripper outside
        the schema range — an unclamped chunk would be refused downstream rather than merely
        being slightly wrong. The targets are NOT clamped: squashing them would apply a
        nonlinearity the model never trained through, and hard limits belong to the deterministic
        safety layer (FR-07), not here.

        Seeding re-creates a CPU generator per draw rather than advancing a stream, so the same
        (checkpoint, observation, seed) yields the same chunk on every call and on every device —
        a CUDA generator draws a different sequence for the same seed, which would make a cluster
        rollout and a workstation rollout incomparable. This is what keeps T-25's bit-identical
        rollouts valid with the sampler switched on.

        The price is worth naming: because the stream is not advanced, an eval pass integrates
        EVERY observation from the same noise vector (or, with ``mean_of``, the same k of them).
        Per chunk that is still a fair draw — the noise does not depend on the observation — but
        the errors across a holdout are correlated through it, so an aggregate is conditioned on
        that one vector and its spread is not the iid spread. The T-30 rule handles this by
        measuring rather than assuming: a second arm at ``flow_seed=1`` re-runs the whole holdout
        from a different vector, and the band the verdict uses is that measured difference.

        ``pooled`` is ``[feature_dim]`` or ``[1, feature_dim]``: the mean over the token axis of
        the shared features, i.e. exactly the reduction :meth:`co_denoise` applies before the
        action branch.
        """
        if steps < 1:
            raise ValueError(f"flow steps must be >= 1, got {steps}")
        if mean_of < 1:
            raise ValueError(f"flow mean_of must be >= 1, got {mean_of}")
        if not 0.0 <= t0 < 1.0:
            # t0=1 would integrate zero steps and return the warm start unchanged, which is the
            # regression chunk laundered through action_recon and reported as a flow arm.
            raise ValueError(f"flow t0 must be in [0, 1), got {t0}")
        if t0 > 0.0 and init_latent is None:
            raise ValueError(
                "flow t0 > 0 needs init_latent: starting at t0 from PURE noise is off the "
                "probability path in a second way rather than fixing the first one — z_t0 has to "
                "be (1-t0)*noise + t0*(a clean-latent estimate) for the arm to mean anything"
            )
        if t0 == 0.0 and init_latent is not None:
            # Same rule as --flow-steps without --flow-sampler: an argument that silently does
            # nothing makes an archived command line unreadable.
            raise ValueError("init_latent was given with t0 = 0, where it is never read")
        feature_dim = self.config.backbone.feature_dim
        feats = pooled.unsqueeze(0) if pooled.ndim == 1 else pooled
        if feats.ndim != 2 or feats.shape[0] != 1 or feats.shape[-1] != feature_dim:
            raise ValueError(
                f"pooled: expected [{feature_dim}] or [1, {feature_dim}], got {tuple(pooled.shape)}"
            )
        feats = feats.float()
        enc = self.config.action_encoder
        shape = (1, self.config.head.num_steps, enc.latent_dim)
        init: Tensor | None = None
        if init_latent is not None:
            init = init_latent.unsqueeze(0) if init_latent.ndim == 2 else init_latent
            if tuple(init.shape) != shape:
                raise ValueError(
                    f"init_latent: expected {shape[1:]} or {shape}, got {tuple(init_latent.shape)}"
                )
            init = init.to(dtype=torch.float32, device=feats.device)
        dt = (1.0 - t0) / steps
        total: Tensor | None = None
        for draw in range(mean_of):
            # CPU generator then .to(device) — the same pattern co_denoise uses for its flow
            # noise, and the reason a seed means the same draw on CPU, CUDA and MPS alike.
            generator = torch.Generator().manual_seed(int(seed) + draw)
            noise = torch.randn(*shape, generator=generator, dtype=torch.float32).to(feats.device)
            z = noise if init is None else (1.0 - t0) * noise + t0 * init
            for k in range(steps):
                t = torch.full((1,), t0 + k * dt, dtype=torch.float32, device=feats.device)
                z = z + self.velocity_head(z, feats, t) * dt
            # Averaged in ACTION space, not latent space: the MSE identity that justifies this
            # arm holds for the quantity being scored, and action_recon is not linear, so a mean
            # of latents is not the mean of the chunks they decode to.
            step = self.action_recon(z)[0]  # [T, D + G]
            total = step if total is None else total + step
        assert total is not None  # mean_of >= 1 is enforced above
        decoded = total / mean_of  # exact for mean_of=1, so that path stays bit-identical
        targets = decoded[:, : enc.target_dim].float().cpu().numpy()
        gripper = decoded[:, enc.target_dim :].mean(dim=-1).clamp(0.0, 1.0).float().cpu().numpy()
        return ActionChunk(
            mode=self.config.head.mode,
            targets=targets,
            gripper_target=gripper,
            dt_s=self.config.head.dt_s,
        )

    @torch.no_grad()
    def predict(
        self,
        observation: Observation,
        *,
        camera: str | None = None,
        flow_steps: int | None = None,
        flow_seed: int = 0,
        flow_mean_k: int = 1,
        flow_t0: float = 0.0,
    ) -> ActionChunk:
        """Policy protocol: one :class:`Observation` -> one canonical :class:`ActionChunk`.

        This is the **representation-only** readout of the joint model, and it is the reason a
        world-action model can close a 2 Hz loop at all: one forward pass at the CLEAN end of
        the flow, then ``ActionHead`` on the shared features. The video velocity comes back and
        is discarded — no iterative denoising, no video sampled at test time, one backbone pass
        per control cycle rather than one per denoising step. The video branch's whole job at
        inference is to have shaped the features during training (AC-07).

        ``t = 1`` follows from WAM's flow convention (``x_t = (1-t)*x0 + t*x1``, x1 clean): the
        observed frame enters unnoised, which is the only timestep at which the backbone is
        being asked to read a real observation rather than a partially destroyed one.

        The frames come from :func:`~wam.training._utils.resolve_frame_context`: the real window
        when ``observation.image_history`` carries one, otherwise the single frame **tiled** to
        ``num_frames``. Training always fed a real window, so the tiled path is a train/inference
        mismatch and not a neutral default — it is what every world-action number recorded before
        2026-08-01 was measured with. T-29 has since put a size on the mismatch, on one
        checkpoint: 10.65 pp of ``skill_vs_repeat_pct`` (−32.45 % tiled, −21.80 % windowed), and
        27.64 pp on L2's task-critical chunks, which is where deleting the motion should hurt
        most. Neither crosses a gate; L1 still fails. Only ``t16-lora-seed0`` has been scored both
        ways, so a tiled number and a windowed one are not comparable (T-29,
        ``docs/improvements.md`` I-7). ``camera`` overrides
        the trained ``config.camera``, for the case where the deployment names the same view
        differently (sim: ``head``); it changes which key is read, never what the model expects.

        ``flow_steps`` switches the ACTION readout — not the backbone pass, which is unchanged —
        from the regression head to :meth:`sample_action_chunk` with that many Euler steps
        (T-30, ``docs/improvements.md`` I-3). ``None`` is the default and leaves the regression
        branch byte-identical, so re-running an archived evaluation reproduces it instead of
        quietly redefining what every number before this flag existed meant.

        ``flow_mean_k`` and ``flow_t0`` select the two T-30 control arms and both default to the
        plain single draw from noise, so they too change nothing on any recorded path.
        ``flow_mean_k`` averages k draws (the MSE-fair comparison against a mean-seeking head —
        see :meth:`sample_action_chunk`). ``flow_t0`` warm-starts the integration at t0 from the
        regression chunk re-encoded by ``action_encoder``, which is the only clean-latent
        estimate available at inference; that is the arm which measures whether a t0=0 negative
        is about the branch or about sampling it outside the region it was trained in.
        """
        if flow_steps is None and (flow_mean_k != 1 or flow_t0 != 0.0):
            raise ValueError(
                f"flow_mean_k={flow_mean_k}/flow_t0={flow_t0} only apply to the flow readout, "
                "and flow_steps is None (the regression head). Silently ignoring them would let "
                "an arm be recorded under a name for something that never ran."
            )
        key = camera if camera is not None else self.config.camera
        device = next(self.parameters()).device
        frames = resolve_frame_context(observation, key, self.config.backbone.num_frames).to(device)
        video_latents = self.backbone.encode_video(frames.unsqueeze(0))
        t = torch.ones(1, dtype=torch.float32, device=device)
        state_emb = self.state_encoder.encode(observation.state)
        text_ctx = self.backbone.condition_text(observation.instruction)
        state_ctx = self.backbone.condition_state(state_emb)
        _, features = self.backbone.forward_flow(video_latents, t, text_ctx, state_ctx)
        if flow_steps is not None:
            init_latent = None
            if flow_t0 > 0.0:
                # The warm start's clean-latent estimate: the regression head's own chunk, put
                # back through the encoder the flow branch was trained against. It is the same
                # backbone pass, so this arm costs one ActionHead + one encoder call, not a
                # second observation — and it deliberately does NOT use the demonstrated chunk,
                # which is available in an offline eval and would turn the arm into an oracle.
                init_latent = self.action_encoder.encode(self.action_head.decode(features[0]))
            # mean over the token axis, the reduction co_denoise feeds the velocity head — the
            # same vector action_head.decode() forms internally, so the two readouts differ in
            # how they turn features into a chunk and in nothing else.
            return self.sample_action_chunk(
                features.float().mean(dim=1),
                steps=flow_steps,
                seed=flow_seed,
                mean_of=flow_mean_k,
                t0=flow_t0,
                init_latent=init_latent,
            )
        # decode() mean-pools the token axis in fp32 — same reduction co_denoise applies before
        # the action branch, so a chunk predicted here matches the one trained against.
        return self.action_head.decode(features[0])


class JointTrainer:
    """Seeded AdamW loop over the combined weighted loss dict (T-16/T-17)."""

    def __init__(
        self,
        config: JointTrainingConfig,
        model: JointWorldActionModel | None = None,
        *,
        backbone: nn.Module | None = None,
    ):
        if model is not None and backbone is not None:
            raise TypeError("pass either model= or backbone=, not both")
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        self.model = (model or JointWorldActionModel(config, backbone=backbone)).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self._param_groups(), lr=config.lr, weight_decay=config.weight_decay
        )
        self._rng = torch.Generator().manual_seed(config.seed)
        self.metadata: RunMetadata | None = None

    def _param_groups(self) -> list[Any]:
        """Trainable parameters as AdamW input — one group, or backbone/rest at separate LRs.

        With ``backbone_lr`` unset this returns a flat parameter LIST, not a one-element group
        dict: that is the exact call shape the trainer has always used, so optimizer state
        dicts written by earlier runs keep loading unchanged. Both groups are emitted even when
        empty so the saved layout does not depend on how much of the backbone is trainable.
        """
        named = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
        if self.config.backbone_lr is None:
            return [param for _, param in named]
        return [
            {
                "params": [p for name, p in named if name.startswith("backbone.")],
                "lr": self.config.backbone_lr,
            },
            {
                "params": [p for name, p in named if not name.startswith("backbone.")],
                "lr": self.config.lr,
            },
        ]

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

    def step(
        self,
        batch: Mapping[str, Any],
        *,
        monitor: TrainingMonitor | None = None,
        step: int = 0,
    ) -> dict[str, float]:
        """One forward/backward/clip/update on ``batch`` -> the history entry for that step.

        The unit an external training loop drives (LR schedules, gradient accumulation,
        distributed sampling, mid-run checkpointing): ``train`` is nothing but this in a loop
        over ``iterate_batches``. ``step`` is only the label written into the entry and handed
        to the monitor; it does not affect the update.
        """
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
        if monitor is not None:
            monitor.record_step(
                step,
                {k: v for k, v in entry.items() if k not in ("step", "grad_norm")},
                grad_norms=TrainingMonitor.module_grad_norms(self.model),
                update_ratio=TrainingMonitor.param_update_ratio(self.model, snapshot or {}),
            )
        return entry

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
        batches = iterate_batches(
            data,
            steps=num_steps,
            batch_size=self.config.batch_size,
            seed=self.config.seed,
            device=self.device,
        )
        return [
            self.step(batch, monitor=monitor, step=index) for index, batch in enumerate(batches)
        ]

    # -- resume / checkpointing ------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Everything a resume needs BESIDES the weights: optimizer moments + RNG streams.

        Both RNG streams matter and they are different objects: ``self._rng`` seeds the
        per-step timestep and flow noise (so a resumed run sees the same noise schedule),
        while the global torch stream drives the DataLoader shuffle and any dropout.
        """
        state: dict[str, Any] = {
            "optimizer": self.optimizer.state_dict(),
            "rng": self._rng.get_state(),
            "torch_rng": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore :meth:`state_dict`. CUDA RNG is skipped when the device is unavailable."""
        self.optimizer.load_state_dict(state["optimizer"])
        self._rng.set_state(state["rng"])
        torch.random.set_rng_state(state["torch_rng"])
        if "cuda_rng" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(list(state["cuda_rng"]))

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        run_id: str = "joint",
        dataset_snapshot_ref: str | None = None,
        git_commit: str | None = None,
        state_dict: Mapping[str, Tensor] | None = None,
    ) -> RunMetadata:
        """Write safetensors weights + embedded config and RunMetadata (FR-10, AC-04).

        ``state_dict`` overrides the payload — pass ``model.trainable_state_dict()`` to write
        adapters only instead of a frozen multi-GB base (reload with ``strict=False``).
        """
        metadata = RunMetadata.create(
            run_id,
            self.config,
            checkpoint_ref=str(Path(path)),
            dataset_snapshot_ref=dataset_snapshot_ref,
            git_commit=git_commit,
        )
        save_checkpoint(self.model, self.config, path, metadata, state_dict=state_dict)
        self.metadata = metadata
        return metadata

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> JointTrainer:
        """Rebuild a trainer (model + config) from a checkpoint; weights are bit-exact."""
        model, metadata = load_joint_checkpoint(path)
        trainer = cls(model.config, model=model)
        trainer.metadata = metadata
        return trainer


def load_joint_checkpoint(
    path: str | Path, *, backbone: nn.Module | None = None, strict: bool = True
) -> tuple[JointWorldActionModel, RunMetadata]:
    """Load ``(JointWorldActionModel, RunMetadata)`` from a safetensors checkpoint.

    ``backbone`` injects an already-loaded backbone instead of rebuilding one from the embedded
    config; ``strict=False`` accepts an adapters-only checkpoint (see
    :meth:`JointWorldActionModel.trainable_state_dict`), where the base weights come from that
    injected backbone rather than from the file.
    """
    state_dict, config_dict, metadata = load_checkpoint_raw(path)
    config = JointTrainingConfig.model_validate(config_dict)
    model = JointWorldActionModel(config, backbone=backbone)
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model, metadata
