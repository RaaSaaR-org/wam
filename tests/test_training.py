"""Tests for wam.training: datasets, action-only baseline, joint trainer, monitor.

Covers T-13 (overfit gate), T-16 (joint co-denoising, frozen parts), T-17 (monitoring,
divergence detection) plus checkpoint traceability (FR-10, AC-04). CPU-only, deterministic,
tiny dims.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch import nn

from wam.backbones.registry import build_backbone, build_backbone_config
from wam.backbones.tiny import TinyBackboneConfig, TinyVideoBackbone
from wam.backbones.wan_i2v import WanBackboneConfig
from wam.decoders import ActionHeadConfig
from wam.encoders import ActionChunkEncoderConfig, StateMLPConfig
from wam.interfaces.protocols import Observation, Policy
from wam.interfaces.schema import (
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    IMUState,
    RobotState,
    ValidityMask,
)
from wam.interfaces.versioning import RunMetadata, config_hash
from wam.training import (
    ActionOnlyConfig,
    ActionOnlyModel,
    ActionOnlyTrainer,
    EpisodeDataset,
    JointTrainer,
    JointTrainingConfig,
    JointWorldActionModel,
    TrainingDiverged,
    TrainingMonitor,
    TrainingMonitorConfig,
    collate_episode_batch,
    load_action_only_checkpoint,
    load_joint_checkpoint,
)
from wam.training._utils import resolve_frame_context

REPO_ROOT = Path(__file__).resolve().parents[1]

NUM_JOINTS = 6
GRIPPER_DIMS = 1
NUM_FRAMES = 2
IMAGE_HW = 16
NUM_STEPS = 8
TARGET_DIM = NUM_JOINTS
FEATURE_DIM = 32
STATE_DIM = 16

SPEC = CanonicalSpaceSpec(joint_names=tuple(f"joint_{i}" for i in range(NUM_JOINTS)))


def tiny_backbone_config() -> TinyBackboneConfig:
    return TinyBackboneConfig(
        feature_dim=FEATURE_DIM,
        patch_size=8,
        depth=1,
        num_heads=4,
        num_frames=NUM_FRAMES,
        image_hw=(IMAGE_HW, IMAGE_HW),
        state_embedding_dim=STATE_DIM,
    )


def action_only_config(**overrides) -> ActionOnlyConfig:
    kwargs: dict = {
        "state": StateMLPConfig(embedding_dim=STATE_DIM, hidden_dims=(32,), num_joints=NUM_JOINTS),
        "backbone": tiny_backbone_config(),
        "head": ActionHeadConfig(
            feature_dim=FEATURE_DIM, num_steps=NUM_STEPS, target_dim=TARGET_DIM, hidden_dims=(64,)
        ),
        "lr": 5e-3,
        "seed": 0,
    }
    kwargs.update(overrides)
    return ActionOnlyConfig(**kwargs)


def joint_config(**overrides) -> JointTrainingConfig:
    kwargs: dict = {
        "state": StateMLPConfig(embedding_dim=STATE_DIM, hidden_dims=(32,), num_joints=NUM_JOINTS),
        "backbone": tiny_backbone_config(),
        "action_encoder": ActionChunkEncoderConfig(
            latent_dim=16, target_dim=TARGET_DIM, hidden_dims=(32,)
        ),
        "head": ActionHeadConfig(
            feature_dim=FEATURE_DIM, num_steps=NUM_STEPS, target_dim=TARGET_DIM, hidden_dims=(64,)
        ),
        "lr": 3e-3,
        "seed": 0,
    }
    kwargs.update(overrides)
    return JointTrainingConfig(**kwargs)


def make_batch(batch_size: int = 32, seed: int = 1) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "frames": torch.randint(
            0, 256, (batch_size, NUM_FRAMES, IMAGE_HW, IMAGE_HW, 3), generator=g, dtype=torch.uint8
        ),
        "q": torch.randn(batch_size, NUM_JOINTS, generator=g),
        "dq": torch.randn(batch_size, NUM_JOINTS, generator=g),
        "imu": torch.randn(batch_size, 10, generator=g),
        "gripper": torch.rand(batch_size, GRIPPER_DIMS, generator=g),
        "targets": torch.rand(batch_size, NUM_STEPS, TARGET_DIM, generator=g) * 1.6 - 0.8,
        "gripper_target": torch.rand(batch_size, NUM_STEPS, generator=g),
        "instruction": "pick the red cube",
    }


def make_state(rng: np.random.Generator, ts: int) -> RobotState:
    return RobotState(
        timestamp_ns=ts,
        q=rng.standard_normal(NUM_JOINTS).astype(np.float32),
        dq=rng.standard_normal(NUM_JOINTS).astype(np.float32),
        imu=IMUState(
            orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
            angular_velocity=np.zeros(3, dtype=np.float32),
            linear_acceleration=np.zeros(3, dtype=np.float32),
        ),
        gripper_state=np.array([0.5], dtype=np.float32),
        validity=ValidityMask(),
    )


def write_episode(
    dir: Path,
    *,
    num_frames: int = 12,
    chunk_len: int = NUM_STEPS,
    target_scale: float = 0.8,
    normalization: dict | None = None,
) -> None:
    from wam.data.episode import EpisodeWriter

    rng = np.random.default_rng(0)
    with EpisodeWriter(
        dir,
        dir.name,
        SPEC,
        fps=10.0,
        instruction="pick the red cube",
        normalization=normalization,
    ) as writer:
        for i in range(num_frames):
            ts = 1_000_000_000 + i * 100_000_000
            img = rng.integers(0, 256, (IMAGE_HW, IMAGE_HW, 3), dtype=np.uint8)
            writer.add_frame("front", img, ts)
            writer.add_state(make_state(rng, ts))
            if i % 2 == 0:
                targets = rng.random((chunk_len, TARGET_DIM)).astype(np.float32)
                targets = targets * (2.0 * target_scale) - target_scale
                chunk = ActionChunk(
                    mode=ActionMode.JOINT_DELTA,
                    targets=targets,
                    gripper_target=rng.random(chunk_len).astype(np.float32),
                    dt_s=0.05,
                )
                writer.add_action(chunk, executed_prefix=chunk_len // 2, timestamp_ns=ts)


# -- EpisodeDataset -------------------------------------------------------------------------


class TestEpisodeDataset:
    def test_windowing_shapes_and_len(self, tmp_path: Path) -> None:
        write_episode(tmp_path / "ep000")
        ds = EpisodeDataset(tmp_path, camera="front", num_frames=NUM_FRAMES, chunk_steps=NUM_STEPS)
        assert len(ds) == 6  # one sample per recorded chunk
        sample = ds[0]
        assert sample["frames"].shape == (NUM_FRAMES, IMAGE_HW, IMAGE_HW, 3)
        assert sample["frames"].dtype == torch.uint8
        assert sample["q"].shape == (NUM_JOINTS,)
        assert sample["dq"].shape == (NUM_JOINTS,)
        assert sample["imu"].shape == (10,)
        assert sample["gripper"].shape == (GRIPPER_DIMS,)
        assert sample["validity"].shape == (4,)
        assert sample["validity"].dtype == torch.bool
        assert sample["targets"].shape == (NUM_STEPS, TARGET_DIM)
        assert sample["targets"].dtype == torch.float32
        assert sample["gripper_target"].shape == (NUM_STEPS,)
        assert sample["instruction"] == "pick the red cube"

    def test_collate(self, tmp_path: Path) -> None:
        write_episode(tmp_path / "ep000")
        ds = EpisodeDataset(tmp_path, camera="front", num_frames=NUM_FRAMES, chunk_steps=NUM_STEPS)
        batch = collate_episode_batch([ds[i] for i in range(3)])
        assert batch["frames"].shape == (3, NUM_FRAMES, IMAGE_HW, IMAGE_HW, 3)
        assert batch["targets"].shape == (3, NUM_STEPS, TARGET_DIM)
        assert batch["instruction"] == ["pick the red cube"] * 3

    def test_short_chunks_are_skipped(self, tmp_path: Path) -> None:
        write_episode(tmp_path / "ep000", chunk_len=4)
        ds = EpisodeDataset(tmp_path, camera="front", chunk_steps=NUM_STEPS)
        with pytest.raises(ValueError, match="no usable samples"):
            len(ds)

    def test_longer_chunks_are_truncated(self, tmp_path: Path) -> None:
        write_episode(tmp_path / "ep000", chunk_len=12)
        ds = EpisodeDataset(tmp_path, camera="front", chunk_steps=NUM_STEPS)
        assert ds[0]["targets"].shape == (NUM_STEPS, TARGET_DIM)

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        ds = EpisodeDataset(tmp_path / "nothing_here")
        with pytest.raises(FileNotFoundError):
            len(ds)

    def test_module_imports_without_touching_data_api(self) -> None:
        # Construction must not require wam.data (lazy import contract).
        ds = EpisodeDataset("/definitely/not/a/path")
        assert ds.camera == "front"

    def test_rejects_non_identity_target_normalization(self, tmp_path: Path) -> None:
        # The pipeline applies NO normalization (identity end-to-end): an episode whose
        # manifest claims z-scored targets must be rejected, not silently trained raw.
        from wam.interfaces.schema import NormalizationSpec

        spec = NormalizationSpec(mean=(0.0,) * TARGET_DIM, std=(0.02,) * TARGET_DIM)
        write_episode(tmp_path / "ep000", normalization={"targets": spec})
        ds = EpisodeDataset(tmp_path, camera="front", chunk_steps=NUM_STEPS)
        with pytest.raises(ValueError, match="non-identity"):
            len(ds)

    def test_identity_normalization_spec_is_accepted(self, tmp_path: Path) -> None:
        from wam.interfaces.schema import NormalizationSpec

        spec = NormalizationSpec(mean=(0.0,) * TARGET_DIM, std=(1.0,) * TARGET_DIM)
        write_episode(tmp_path / "ep000", normalization={"targets": spec})
        ds = EpisodeDataset(tmp_path, camera="front", chunk_steps=NUM_STEPS)
        assert len(ds) > 0

    def test_rejects_targets_outside_tanh_range(self, tmp_path: Path) -> None:
        # Shipped decoders are tanh-bounded to (-1, 1): per-step deltas >= 1 are silently
        # unlearnable (loss floor, no error) — the dataset must fail loudly instead.
        write_episode(tmp_path / "ep000", target_scale=1.5)
        ds = EpisodeDataset(tmp_path, camera="front", chunk_steps=NUM_STEPS)
        with pytest.raises(ValueError, match="tanh"):
            len(ds)


# -- Action-only baseline (T-13) -------------------------------------------------------------


class TestActionOnlyModel:
    def test_forward_shapes(self) -> None:
        torch.manual_seed(0)
        model = ActionOnlyModel(action_only_config())
        out = model(make_batch(batch_size=3))
        assert out["targets"].shape == (3, NUM_STEPS, TARGET_DIM)
        assert out["gripper"].shape == (3, NUM_STEPS, GRIPPER_DIMS)
        assert out["targets"].abs().max() < 1.0  # tanh-bounded
        assert (out["gripper"] > 0).all() and (out["gripper"] < 1).all()

    def test_deterministic_construction(self) -> None:
        cfg = action_only_config()
        batch = make_batch(batch_size=2)
        torch.manual_seed(0)
        out_a = ActionOnlyModel(cfg)(batch)
        torch.manual_seed(0)
        out_b = ActionOnlyModel(cfg)(batch)
        assert torch.equal(out_a["targets"], out_b["targets"])

    def test_predict_is_policy(self) -> None:
        torch.manual_seed(0)
        model = ActionOnlyModel(action_only_config())
        assert isinstance(model, Policy)
        rng = np.random.default_rng(3)
        obs = Observation(
            images={"front": rng.integers(0, 256, (IMAGE_HW, IMAGE_HW, 3), dtype=np.uint8)},
            state=make_state(rng, ts=0),
            instruction="pick",
        )
        chunk = model.predict(obs)
        assert chunk.targets.shape == (NUM_STEPS, TARGET_DIM)
        assert chunk.validate(SPEC) == []

    @pytest.mark.skipif(
        not torch.backends.mps.is_available() and not torch.cuda.is_available(),
        reason="needs a non-CPU device",
    )
    def test_predict_from_numpy_obs_on_accelerator(self) -> None:
        device = "mps" if torch.backends.mps.is_available() else "cuda"
        torch.manual_seed(0)
        model = ActionOnlyModel(action_only_config()).to(device)
        rng = np.random.default_rng(3)
        obs = Observation(
            images={"front": rng.integers(0, 256, (IMAGE_HW, IMAGE_HW, 3), dtype=np.uint8)},
            state=make_state(rng, ts=0),
            instruction="pick",
        )
        chunk = model.predict(obs)
        assert chunk.targets.shape == (NUM_STEPS, TARGET_DIM)

    def test_dim_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="state_embedding_dim"):
            action_only_config(
                state=StateMLPConfig(embedding_dim=8, hidden_dims=(32,), num_joints=NUM_JOINTS)
            )


class TestActionOnlyTrainer:
    def test_overfit_32_samples(self) -> None:
        # T-13 go/no-go gate: 32 random samples to near-zero in <= 300 CPU steps.
        trainer = ActionOnlyTrainer(action_only_config())
        assert trainer.overfit(make_batch(batch_size=32), steps=300, target_loss=5e-3)

    def test_history_records_all_components(self) -> None:
        trainer = ActionOnlyTrainer(action_only_config())
        history = trainer.train(make_batch(batch_size=4), steps=3)
        assert len(history) == 3
        for key in ("total", "action", "gripper", "smoothness", "limit", "grad_norm", "step"):
            assert key in history[0]
        assert history[-1]["total"] < history[0]["total"]

    def test_trains_from_episode_dataset(self, tmp_path: Path) -> None:
        write_episode(tmp_path / "ep000")
        ds = EpisodeDataset(tmp_path, camera="front", num_frames=NUM_FRAMES, chunk_steps=NUM_STEPS)
        trainer = ActionOnlyTrainer(action_only_config(batch_size=4))
        history = trainer.train(ds, steps=4)
        assert len(history) == 4
        assert all(np.isfinite(entry["total"]) for entry in history)

    def test_checkpoint_roundtrip_bit_exact(self, tmp_path: Path) -> None:
        trainer = ActionOnlyTrainer(action_only_config())
        trainer.train(make_batch(batch_size=4), steps=2)
        path = tmp_path / "action_only.safetensors"
        metadata = trainer.save_checkpoint(path, run_id="run-1", git_commit="deadbeef")

        assert metadata.config_hash == config_hash(trainer.config)  # AC-04
        assert metadata.checkpoint_ref == str(path)
        assert isinstance(metadata, RunMetadata)

        model, loaded_meta = load_action_only_checkpoint(path)
        assert loaded_meta == metadata
        batch = make_batch(batch_size=3, seed=9)
        trainer.model.eval()
        with torch.no_grad():
            expected = trainer.model(batch)
            actual = model(batch)
        assert torch.equal(expected["targets"], actual["targets"])
        assert torch.equal(expected["gripper"], actual["gripper"])

    def test_load_checkpoint_rebuilds_trainer(self, tmp_path: Path) -> None:
        trainer = ActionOnlyTrainer(action_only_config())
        path = tmp_path / "ckpt.safetensors"
        trainer.save_checkpoint(path, git_commit="deadbeef")
        restored = ActionOnlyTrainer.load_checkpoint(path)
        assert restored.config == trainer.config
        assert restored.metadata is not None
        assert restored.metadata.checkpoint_ref == str(path)

    def test_yaml_config_loads(self) -> None:
        cfg = ActionOnlyConfig.from_yaml(REPO_ROOT / "configs" / "training" / "action_only.yaml")
        assert cfg.head.num_steps == 8
        assert cfg.head.mode is ActionMode.JOINT_DELTA
        assert cfg.backbone.state_embedding_dim == cfg.state.embedding_dim


# -- Joint world-action training (T-16) ------------------------------------------------------


class TestJointWorldActionModel:
    def test_co_denoise_shapes(self) -> None:
        torch.manual_seed(0)
        cfg = joint_config()
        model = JointWorldActionModel(cfg)
        batch = make_batch(batch_size=3)
        t = torch.tensor([0.2, 0.5, 0.8])
        out = model.co_denoise(batch, t, generator=torch.Generator().manual_seed(0))
        video_shape = (3, NUM_FRAMES, IMAGE_HW, IMAGE_HW, 3)
        latent = cfg.action_encoder.latent_dim
        assert out["video_velocity_pred"].shape == video_shape
        assert out["video_velocity_target"].shape == video_shape
        assert out["action_velocity_pred"].shape == (3, NUM_STEPS, latent)
        assert out["action_velocity_target"].shape == (3, NUM_STEPS, latent)
        assert out["decoded_targets"].shape == (3, NUM_STEPS, TARGET_DIM)
        assert out["video_feature"].shape == (3, FEATURE_DIM)
        assert out["action_feature"].shape == (3, FEATURE_DIM)

    def test_frozen_parts_registry(self) -> None:
        torch.manual_seed(0)
        model = JointWorldActionModel(joint_config())
        assert set(model.frozen_parts) == {"text_embedding", "text_pos"}
        frozen = model.frozen_parameter_names()
        assert "backbone.text_embedding.weight" in frozen
        assert "backbone.text_pos" in frozen
        assert not model.backbone.text_embedding.weight.requires_grad

    def test_frozen_parts_do_not_train(self) -> None:
        trainer = JointTrainer(joint_config())
        text_before = trainer.model.backbone.text_embedding.weight.detach().clone()
        video_before = trainer.model.backbone.video_proj.weight.detach().clone()
        trainer.train(make_batch(batch_size=4), steps=5)
        assert torch.equal(trainer.model.backbone.text_embedding.weight, text_before)
        assert not torch.equal(trainer.model.backbone.video_proj.weight, video_before)

    def test_gradients_reach_both_branches(self) -> None:
        trainer = JointTrainer(joint_config())
        losses = trainer.compute_losses(make_batch(batch_size=2))
        losses["total"].backward()
        norms = TrainingMonitor.module_grad_norms(trainer.model)
        for module in ("backbone", "action_encoder", "velocity_head", "action_head"):
            assert norms[module] > 0.0, f"no gradient reached {module}"

    def test_flow_targets_are_detached_from_encoder(self) -> None:
        # Collapse-shortcut regression: the flow-matching target/input must NOT carry
        # gradients into the ActionChunkEncoder (else AdamW can drive the encoder to a
        # constant latent and action_flow -> 0 while encoding zero action information).
        torch.manual_seed(0)
        model = JointWorldActionModel(joint_config())
        out = model.co_denoise(
            make_batch(batch_size=2),
            torch.tensor([0.3, 0.6]),
            generator=torch.Generator().manual_seed(0),
        )
        assert not out["action_velocity_target"].requires_grad
        assert out["recon_targets"].requires_grad  # the anchor DOES train the encoder

    def test_action_flow_trains_only_velocity_head_recon_anchors_encoder(self) -> None:
        trainer = JointTrainer(joint_config())
        losses = trainer.compute_losses(make_batch(batch_size=2))
        losses["action_flow"].backward(retain_graph=True)
        enc_grads = [p.grad for p in trainer.model.action_encoder.parameters()]
        assert all(g is None or torch.all(g == 0) for g in enc_grads)
        vel_norm = sum(
            float(p.grad.norm())
            for p in trainer.model.velocity_head.parameters()
            if p.grad is not None
        )
        assert vel_norm > 0.0
        trainer.model.zero_grad(set_to_none=True)
        losses["action_recon"].backward()
        enc_norm = sum(
            float(p.grad.norm())
            for p in trainer.model.action_encoder.parameters()
            if p.grad is not None
        )
        assert enc_norm > 0.0  # reconstruction anchor reaches the encoder


class _StraightPathField(nn.Module):
    """Velocity-head stub: the exact field of the straight path from any z to a fixed ``x1``.

    ``v(z, t) = (x1 - z) / (1 - t)`` is the analytic velocity of a rectified-flow trajectory that
    ends at ``x1``. On the sampler's grid it makes forward Euler EXACT, so the direction test has
    no tolerance to tune and cannot go flaky.
    """

    def __init__(self, x1: torch.Tensor) -> None:
        super().__init__()
        self.x1 = x1

    def forward(self, z_t: torch.Tensor, pooled: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (self.x1 - z_t) / (1.0 - float(t[0]))


class _RecordingRecon(nn.Module):
    """``action_recon`` stub that captures the sampled latent and returns a well-formed chunk."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.last: torch.Tensor | None = None

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        self.last = latent.detach().clone()
        return torch.zeros(*latent.shape[:-1], self.out_dim)


class TestFlowSampler:
    """T-30 / I-3: reading the chunk out of the trained flow branch instead of regressing it.

    Everything here runs on the velocity head and ``action_recon`` alone — no backbone weights,
    no GPU. That is the claim under test as much as it is a convenience: the sampler has to work
    on an ARCHIVED checkpoint, without retraining and without touching the frozen base.
    """

    def _model(self) -> JointWorldActionModel:
        torch.manual_seed(0)
        model = JointWorldActionModel(joint_config())
        model.eval()
        return model

    def _obs(self) -> Observation:
        rng = np.random.default_rng(3)
        return Observation(
            images={"front": rng.integers(0, 256, (IMAGE_HW, IMAGE_HW, 3), dtype=np.uint8)},
            state=make_state(rng, ts=0),
            instruction="pick",
        )

    def _pooled(self, model: JointWorldActionModel, seed: int = 7) -> torch.Tensor:
        return torch.randn(1, FEATURE_DIM, generator=torch.Generator().manual_seed(seed))

    def _initial_noise(self, model: JointWorldActionModel, seed: int) -> torch.Tensor:
        """The exact draw ``sample_action_chunk`` starts from, so a test can integrate alongside."""
        return torch.randn(
            1,
            model.config.head.num_steps,
            model.config.action_encoder.latent_dim,
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.float32,
        )

    def test_euler_integration_lands_exactly_on_the_clean_latent(self) -> None:
        """Pins the convention: in WAM t=0 is NOISE and t=1 is CLEAN (losses.make_flow_targets),
        so the sampler integrates forward. With the closed-form straight-path field substituted
        for the velocity head, forward Euler on the grid {k/n} reaches x1 to the last bit."""
        model = self._model()
        latent_dim = model.config.action_encoder.latent_dim
        x1 = torch.full((1, model.config.head.num_steps, latent_dim), 0.25)
        model.velocity_head = _StraightPathField(x1)
        recon = _RecordingRecon(TARGET_DIM + model.config.action_encoder.gripper_dims)
        model.action_recon = recon

        model.sample_action_chunk(self._pooled(model), steps=8, seed=0)

        assert recon.last is not None
        assert torch.allclose(recon.last, x1, atol=1e-6, rtol=0)

    def test_flipping_the_velocity_sign_walks_away_from_the_clean_latent(self) -> None:
        """The anti-test, and the reason the one above exists. Integrating ``z -= v*dt`` — what
        the diffusers/SD3 convention (t=1 noise) would suggest — still returns a finite chunk that
        decodes to plausible-looking numbers; only the latent shows it went the wrong way. With
        this field the error grows by exactly (n+1), so the wrong direction fails loudly."""
        model = self._model()
        latent_dim = model.config.action_encoder.latent_dim
        steps = 8
        x1 = torch.full((1, model.config.head.num_steps, latent_dim), 0.25)
        field = _StraightPathField(x1)

        z = self._initial_noise(model, seed=0)
        start_error = float((z - x1).norm())
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((1,), k * dt)
            z = z - field(z, self._pooled(model), t) * dt

        assert float((z - x1).norm()) == pytest.approx(start_error * (steps + 1), rel=1e-4)

    def test_the_timestep_grid_stays_inside_the_training_support(self) -> None:
        """``compute_losses`` draws t from ``torch.rand``, support [0, 1). Evaluating the head at
        t=1.0 asks it for a timestep it was never once trained at — and it would not error, it
        would just answer badly."""
        model = self._model()
        seen: list[float] = []

        class _Recorder(nn.Module):
            def forward(self, z_t: torch.Tensor, pooled: torch.Tensor, t: torch.Tensor):
                seen.append(float(t[0]))
                return torch.zeros_like(z_t)

        model.velocity_head = _Recorder()
        model.sample_action_chunk(self._pooled(model), steps=4, seed=0)

        assert seen == [0.0, 0.25, 0.5, 0.75]

    def test_flow_steps_none_is_byte_identical_to_the_regression_head(self) -> None:
        """The archived-run guarantee: adding the flow branch must not move a single bit of the
        default path, or every world-action number on record silently changes meaning."""
        model = self._model()
        obs = self._obs()

        # The pre-T-30 body of predict(), spelled out, run against the same model instance.
        device = next(model.parameters()).device
        frames = resolve_frame_context(obs, model.config.camera, NUM_FRAMES).to(device)
        with torch.no_grad():
            video_latents = model.backbone.encode_video(frames.unsqueeze(0))
            state_ctx = model.backbone.condition_state(model.state_encoder.encode(obs.state))
            _, features = model.backbone.forward_flow(
                video_latents,
                torch.ones(1, dtype=torch.float32, device=device),
                model.backbone.condition_text(obs.instruction),
                state_ctx,
            )
            historical = model.action_head.decode(features[0])

        for chunk in (model.predict(obs), model.predict(obs, flow_steps=None)):
            assert np.array_equal(chunk.targets, historical.targets)
            assert np.array_equal(chunk.gripper_target, historical.gripper_target)
            assert chunk.dt_s == historical.dt_s and chunk.mode == historical.mode

    def test_the_flow_readout_actually_differs_from_the_regression_head(self) -> None:
        """The complement: if the two readouts returned the same chunk the A/B would measure
        nothing, and a mis-wired flag would look exactly like a null result."""
        model = self._model()
        obs = self._obs()
        assert not np.array_equal(
            model.predict(obs).targets, model.predict(obs, flow_steps=4).targets
        )

    def test_the_flow_path_decodes_through_action_recon_not_the_action_head(self) -> None:
        """``action_recon`` is documented as a training-only anchor; the flow readout promotes it
        to the deployed decoder. Perturbing it must move the chunk and perturbing ``ActionHead``
        must not — otherwise the 'flow branch' is the regression head wearing a flag."""
        model = self._model()
        obs = self._obs()
        before = model.predict(obs, flow_steps=4).targets

        with torch.no_grad():
            model.action_head.target_head.weight.add_(1.0)
        assert np.array_equal(model.predict(obs, flow_steps=4).targets, before)

        with torch.no_grad():
            model.action_recon[-1].weight.add_(1.0)
        assert not np.array_equal(model.predict(obs, flow_steps=4).targets, before)

    def test_identical_observations_give_identical_chunks(self) -> None:
        """The determinism contract runtime/policies.py:12 promises and T-25's bit-identical
        MuJoCo rollouts rest on: the seed is re-drawn per call, never advanced across calls."""
        model = self._model()
        obs = self._obs()
        first = model.predict(obs, flow_steps=6, flow_seed=3)
        second = model.predict(obs, flow_steps=6, flow_seed=3)
        assert np.array_equal(first.targets, second.targets)
        assert np.array_equal(first.gripper_target, second.gripper_target)

    def test_different_seeds_give_different_chunks(self) -> None:
        """It has to be a sampler over a distribution, not a deterministic map wearing a seed —
        which is precisely what a mode-collapsed velocity head would quietly decay into."""
        model = self._model()
        obs = self._obs()
        assert not np.array_equal(
            model.predict(obs, flow_steps=6, flow_seed=0).targets,
            model.predict(obs, flow_steps=6, flow_seed=1).targets,
        )

    def test_gripper_is_clamped_into_the_schema_range(self) -> None:
        """``action_recon`` ends in a bare Linear where ``ActionHead`` has a sigmoid, so nothing
        bounds the gripper. An out-of-range command fails ActionChunk.validate and the
        deterministic safety layer rejects the chunk — the clamp is what keeps it deployable."""
        model = self._model()
        with torch.no_grad():
            model.action_recon[-1].bias.add_(50.0)
        chunk = model.sample_action_chunk(self._pooled(model), steps=4, seed=0)
        assert chunk.validate(SPEC) == []

        with torch.no_grad():
            model.action_recon[-1].bias.sub_(100.0)
        assert model.sample_action_chunk(self._pooled(model), steps=4, seed=0).validate(SPEC) == []

    def test_targets_are_left_unsquashed_for_the_safety_layer(self) -> None:
        """Deliberately NOT clamped: squashing would apply a nonlinearity the model never trained
        through, and hard limits are the deterministic safety layer's job (FR-07). A large
        magnitude must reach the caller as a large magnitude."""
        model = self._model()
        with torch.no_grad():
            model.action_recon[-1].bias.add_(50.0)
        chunk = model.sample_action_chunk(self._pooled(model), steps=4, seed=0)
        assert np.abs(chunk.targets).max() > 1.0

    def test_sampling_needs_only_the_velocity_head_and_action_recon(self) -> None:
        """The 'works on an archived checkpoint, no retraining' claim as a regression guard: the
        sampler must not reach for the backbone, which for T-16 means tens of GB of frozen Wan."""
        model = self._model()

        def _forbidden(*args, **kwargs):
            raise AssertionError("sample_action_chunk ran a backbone pass")

        model.backbone.forward_flow = _forbidden  # type: ignore[method-assign]
        chunk = model.sample_action_chunk(self._pooled(model), steps=4, seed=0)
        assert chunk.targets.shape == (NUM_STEPS, TARGET_DIM)
        assert chunk.validate(SPEC) == []

    def test_a_saved_checkpoint_carries_every_module_the_sampler_needs(self) -> None:
        """T-16 wrote adapters only (``trainable_state_dict``). If a future freezing change drops
        ``action_recon`` from the trainable set, the flow readout silently becomes a retrain —
        this fails first instead."""
        model = self._model()
        keys = set(model.trainable_state_dict())
        for prefix in ("velocity_head.", "action_recon.", "action_encoder."):
            assert any(k.startswith(prefix) for k in keys), prefix
        # The layout every archived checkpoint's state dict is keyed by — renaming strands them.
        assert {"action_recon.0.weight", "action_recon.2.weight"} <= keys

    def test_zero_or_negative_steps_are_rejected(self) -> None:
        model = self._model()
        for steps in (0, -1):
            with pytest.raises(ValueError, match="flow steps"):
                model.sample_action_chunk(self._pooled(model), steps=steps, seed=0)

    def test_a_wrongly_shaped_pooled_vector_is_rejected(self) -> None:
        """Feeding the token axis in un-pooled would broadcast into a batch of chunks and return
        the first one, which is a wrong answer rather than an error."""
        model = self._model()
        with pytest.raises(ValueError, match="pooled"):
            model.sample_action_chunk(torch.zeros(3, FEATURE_DIM), steps=4, seed=0)


class TestFlowSamplerControlArms:
    """T-30's two control arms: ``mean_of`` (k draws averaged) and ``t0`` (warm start).

    Both exist because of how the T-30 verdict would otherwise be misread. ``mean_of`` is there
    because ``skill_vs_repeat_pct`` is MSE-derived and one unbiased draw is worth exactly the
    conditional variance MORE error than the conditional mean — the metric charges a sampler for
    sampling, which is the property the regression head is on trial for lacking. ``t0`` is there
    because the sampler feeds the velocity head t=1 features at every timestep, a pairing it was
    never trained on, so a from-noise negative cannot distinguish "the branch is dead" from "we
    sampled it outside its training region".
    """

    def _model(self) -> JointWorldActionModel:
        torch.manual_seed(0)
        model = JointWorldActionModel(joint_config())
        model.eval()
        return model

    def _obs(self) -> Observation:
        rng = np.random.default_rng(3)
        return Observation(
            images={"front": rng.integers(0, 256, (IMAGE_HW, IMAGE_HW, 3), dtype=np.uint8)},
            state=make_state(rng, ts=0),
            instruction="pick",
        )

    def _pooled(self, seed: int = 7) -> torch.Tensor:
        return torch.randn(1, FEATURE_DIM, generator=torch.Generator().manual_seed(seed))

    def _latent(self, model: JointWorldActionModel, value: float) -> torch.Tensor:
        return torch.full(
            (1, model.config.head.num_steps, model.config.action_encoder.latent_dim), value
        )

    def test_the_default_arms_reproduce_the_sampler_that_existed_before_them(self) -> None:
        """The archived-run guarantee, one level below ``flow_steps=None``: adding the arms must
        not move a single bit of the plain sampler either, or every ``+flow32s0`` number recorded
        before they existed silently changes meaning. The pre-arm loop is spelled out here rather
        than trusted, because ``t0 + k*dt`` and ``(1-t0)/steps`` are new arithmetic on that path.
        """
        model = self._model()
        steps, seed = 5, 3
        pooled = self._pooled()

        z = torch.randn(
            1,
            model.config.head.num_steps,
            model.config.action_encoder.latent_dim,
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.float32,
        )
        dt = 1.0 / steps
        with torch.no_grad():
            for k in range(steps):
                t = torch.full((1,), k * dt, dtype=torch.float32)
                z = z + model.velocity_head(z, pooled.float(), t) * dt
            decoded = model.action_recon(z)[0]
        target_dim = model.config.action_encoder.target_dim

        chunk = model.sample_action_chunk(pooled, steps=steps, seed=seed)

        assert np.array_equal(chunk.targets, decoded[:, :target_dim].numpy())

    def test_averaging_k_draws_returns_the_mean_of_those_k_chunks(self) -> None:
        """The arm's whole claim is an arithmetic one: E‖a-draw‖² = E‖a-mean‖² + Var, so the
        MSE-fair comparison against a mean-seeking head is the MEAN of k draws. If this returned
        anything else — a mean of latents, or k copies of one draw — the T-30 table would compare
        two different estimators and call the difference a readout difference."""
        model = self._model()
        pooled = self._pooled()
        singles = [
            model.sample_action_chunk(pooled, steps=4, seed=11 + i).targets for i in range(3)
        ]

        averaged = model.sample_action_chunk(pooled, steps=4, seed=11, mean_of=3).targets

        assert np.allclose(averaged, np.mean(singles, axis=0), atol=1e-6, rtol=0)

    def test_the_k_draws_use_k_distinct_seeds(self) -> None:
        """k copies of one draw would average to that draw, leave the conditional variance in the
        score untouched, and look exactly like a working arm on every shape assertion."""
        model = self._model()
        pooled = self._pooled()

        single = model.sample_action_chunk(pooled, steps=4, seed=0).targets
        averaged = model.sample_action_chunk(pooled, steps=4, seed=0, mean_of=4).targets

        assert not np.array_equal(single, averaged)

    def test_averaging_stays_deterministic_across_calls(self) -> None:
        """The determinism contract has to survive the arm: seeds ``seed..seed+k-1`` are re-drawn
        per call, never advanced, or two evaluations of one checkpoint stop being comparable."""
        model = self._model()
        pooled = self._pooled()
        first = model.sample_action_chunk(pooled, steps=4, seed=2, mean_of=3)
        second = model.sample_action_chunk(pooled, steps=4, seed=2, mean_of=3)
        assert np.array_equal(first.targets, second.targets)
        assert np.array_equal(first.gripper_target, second.gripper_target)

    def test_a_warm_start_evaluates_the_head_only_at_and_above_t0(self) -> None:
        """The point of the arm: the head is asked ONLY about the region where its own t and the
        features' t=1 roughly agree. A grid that still visited t≈0 would re-introduce the exact
        confound the arm exists to measure, while looking like it had removed it."""
        model = self._model()
        seen: list[float] = []

        class _Recorder(nn.Module):
            def forward(self, z_t: torch.Tensor, pooled: torch.Tensor, t: torch.Tensor):
                seen.append(float(t[0]))
                return torch.zeros_like(z_t)

        model.velocity_head = _Recorder()
        model.sample_action_chunk(
            self._pooled(), steps=4, seed=0, t0=0.6, init_latent=self._latent(model, 0.25)
        )

        assert seen == pytest.approx([0.6, 0.7, 0.8, 0.9])
        assert max(seen) < 1.0  # torch.rand's support is [0, 1); t=1 was never trained

    def test_a_warm_start_begins_from_the_init_latent_noised_to_t0(self) -> None:
        """``z_t0`` must be ``(1-t0)*noise + t0*init``. Starting at t0 from PURE noise would be
        off the probability path in a second way rather than fixing the first, and the arm would
        quietly measure nothing while reporting a number."""
        model = self._model()
        recon = _RecordingRecon(TARGET_DIM + model.config.action_encoder.gripper_dims)
        model.action_recon = recon

        class _Zero(nn.Module):
            def forward(self, z_t: torch.Tensor, pooled: torch.Tensor, t: torch.Tensor):
                return torch.zeros_like(z_t)

        model.velocity_head = _Zero()
        init = self._latent(model, 0.25)
        noise = torch.randn(
            *init.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float32
        )

        model.sample_action_chunk(self._pooled(), steps=4, seed=0, t0=0.6, init_latent=init)

        assert recon.last is not None
        assert torch.allclose(recon.last, 0.4 * noise + 0.6 * init, atol=1e-6, rtol=0)

    def test_predict_warm_starts_from_the_regression_head_so_it_inherits_that_mean(self) -> None:
        """The honesty test for the arm's interpretation. ``predict`` has no clean action latent
        at inference, so it re-encodes the REGRESSION chunk — which means a warm-started score
        cannot be read as 'the flow branch models the conditional'. Perturbing ``ActionHead`` must
        move this readout (it does not move the from-noise one, pinned in TestFlowSampler)."""
        model = self._model()
        obs = self._obs()
        before = model.predict(obs, flow_steps=4, flow_t0=0.6).targets

        with torch.no_grad():
            model.action_head.target_head.weight.add_(1.0)

        assert not np.array_equal(model.predict(obs, flow_steps=4, flow_t0=0.6).targets, before)

    def test_the_arms_at_their_defaults_leave_predict_byte_identical(self) -> None:
        """``mean_of=1``/``t0=0`` are the recorded path. ``total / 1`` and ``0.0 + k*dt`` are
        exact in IEEE-754 and this is what says so out loud."""
        model = self._model()
        obs = self._obs()
        plain = model.predict(obs, flow_steps=6, flow_seed=1)
        explicit = model.predict(obs, flow_steps=6, flow_seed=1, flow_mean_k=1, flow_t0=0.0)
        assert np.array_equal(plain.targets, explicit.targets)
        assert np.array_equal(plain.gripper_target, explicit.gripper_target)

    def test_a_warm_start_without_an_init_latent_is_rejected(self) -> None:
        model = self._model()
        with pytest.raises(ValueError, match="init_latent"):
            model.sample_action_chunk(self._pooled(), steps=4, seed=0, t0=0.6)

    def test_an_init_latent_that_would_never_be_read_is_rejected(self) -> None:
        """Same rule as ``--flow-steps`` without ``--flow-sampler``: an argument that silently
        does nothing makes an archived command line unreadable."""
        model = self._model()
        with pytest.raises(ValueError, match="never read"):
            model.sample_action_chunk(
                self._pooled(), steps=4, seed=0, init_latent=self._latent(model, 0.25)
            )

    def test_a_wrongly_shaped_init_latent_is_rejected(self) -> None:
        """It would broadcast against the noise instead of failing, warm-starting from something
        that is not a chunk latent at all."""
        model = self._model()
        with pytest.raises(ValueError, match="init_latent"):
            model.sample_action_chunk(
                self._pooled(),
                steps=4,
                seed=0,
                t0=0.5,
                init_latent=torch.zeros(1, model.config.head.num_steps + 1, 3),
            )

    def test_t0_outside_zero_to_one_is_rejected(self) -> None:
        """t0=1 integrates zero steps and hands back the warm start unchanged — the regression
        chunk laundered through ``action_recon`` and reported as a flow arm."""
        model = self._model()
        for t0 in (1.0, 1.5, -0.1):
            with pytest.raises(ValueError, match="t0"):
                model.sample_action_chunk(
                    self._pooled(), steps=4, seed=0, t0=t0, init_latent=self._latent(model, 0.25)
                )

    def test_fewer_than_one_draw_is_rejected(self) -> None:
        model = self._model()
        for mean_of in (0, -2):
            with pytest.raises(ValueError, match="mean_of"):
                model.sample_action_chunk(self._pooled(), steps=4, seed=0, mean_of=mean_of)

    def test_the_arms_are_rejected_when_the_flow_readout_is_off(self) -> None:
        """Ignoring them silently would let an arm be recorded under the name of a run that never
        happened — the regression head's chunk filed as ``+flow32s0k8``."""
        model = self._model()
        obs = self._obs()
        for kwargs in ({"flow_mean_k": 4}, {"flow_t0": 0.5}):
            with pytest.raises(ValueError, match="flow_mean_k|flow_t0"):
                model.predict(obs, **kwargs)


class TestJointTrainer:
    def test_loss_dict_keys_and_decrease(self) -> None:
        trainer = JointTrainer(joint_config())
        history = trainer.train(make_batch(batch_size=8), steps=30)
        expected = {
            "video",
            "action_flow",
            "action_recon",
            "action_reg",
            "gripper",
            "alignment",
            "smoothness",
            "limit",
            "total",
        }
        assert expected <= set(history[0])
        assert all(np.isfinite(entry["total"]) for entry in history)
        assert history[-1]["total"] < history[0]["total"]

    def test_checkpoint_roundtrip_bit_exact(self, tmp_path: Path) -> None:
        trainer = JointTrainer(joint_config())
        trainer.train(make_batch(batch_size=4), steps=2)
        path = tmp_path / "joint.safetensors"
        metadata = trainer.save_checkpoint(path, run_id="joint-1", git_commit="deadbeef")
        assert metadata.config_hash == config_hash(trainer.config)

        model, loaded_meta = load_joint_checkpoint(path)
        assert loaded_meta == metadata
        batch = make_batch(batch_size=2, seed=5)
        t = torch.tensor([0.3, 0.7])
        noise_v = torch.randn(2, NUM_FRAMES, IMAGE_HW, IMAGE_HW, 3)
        noise_a = torch.randn(2, NUM_STEPS, trainer.config.action_encoder.latent_dim)
        trainer.model.eval()
        with torch.no_grad():
            expected = trainer.model.co_denoise(batch, t, video_noise=noise_v, action_noise=noise_a)
            actual = model.co_denoise(batch, t, video_noise=noise_v, action_noise=noise_a)
        assert torch.equal(expected["action_velocity_pred"], actual["action_velocity_pred"])
        assert torch.equal(expected["video_velocity_pred"], actual["video_velocity_pred"])
        assert torch.equal(expected["decoded_targets"], actual["decoded_targets"])

    def test_yaml_config_loads(self) -> None:
        cfg = JointTrainingConfig.from_yaml(REPO_ROOT / "configs" / "training" / "joint.yaml")
        assert cfg.action_encoder.max_steps >= cfg.head.num_steps
        assert cfg.action_encoder.target_dim == cfg.head.target_dim


# -- backbone swappability (FR-09): tagged config union ---------------------------------------


WAN_SECTION: dict = {
    "kind": "wan_i2v",
    "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "feature_dim": FEATURE_DIM,  # cross-validated against head.feature_dim
    "state_embedding_dim": STATE_DIM,  # ... and against state.embedding_dim
}


class TestBackboneConfigUnion:
    def test_shipped_joint_yaml_stays_tiny(self) -> None:
        cfg = JointTrainingConfig.from_yaml(REPO_ROOT / "configs" / "training" / "joint.yaml")
        assert type(cfg.backbone) is TinyBackboneConfig
        assert cfg.backbone.kind == "tiny"

    def test_tagged_wan_section_validates_as_wan(self) -> None:
        cfg = joint_config(backbone=WAN_SECTION)
        assert type(cfg.backbone) is WanBackboneConfig
        assert cfg.backbone.model_id == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    def test_untagged_wan_shaped_section_is_rejected(self) -> None:
        # Without the discriminator this used to validate as an all-defaults 64-dim tiny
        # config and silently train the wrong model.
        untagged = {k: v for k, v in WAN_SECTION.items() if k != "kind"}
        with pytest.raises(ValidationError):
            joint_config(backbone=untagged)

    def test_json_round_trip_preserves_the_union_member(self) -> None:
        cfg = joint_config(backbone=WAN_SECTION)
        restored = JointTrainingConfig.model_validate_json(cfg.model_dump_json())
        assert type(restored.backbone) is WanBackboneConfig
        assert restored == cfg

    def test_legacy_untagged_tiny_section_still_validates(self) -> None:
        # The shape of every config dict embedded in a checkpoint written before the union.
        data = joint_config().model_dump()
        data["backbone"].pop("kind")
        assert type(JointTrainingConfig.model_validate(data).backbone) is TinyBackboneConfig

    def test_wan_config_is_hashable_into_config_hash(self) -> None:
        # AC-04: a torch dtype object anywhere in the config would make this raise TypeError.
        assert isinstance(config_hash(joint_config(backbone=WAN_SECTION)), str)


class TestBuildBackbone:
    def test_untagged_mapping_is_tagged_tiny(self) -> None:
        config = build_backbone_config({"feature_dim": FEATURE_DIM, "num_frames": NUM_FRAMES})
        assert type(config) is TinyBackboneConfig
        assert config.feature_dim == FEATURE_DIM

    def test_tagged_wan_mapping(self) -> None:
        config = build_backbone_config({"kind": "wan_i2v", "lora_rank": 8})
        assert type(config) is WanBackboneConfig
        assert config.lora_rank == 8

    def test_untagged_wan_shaped_mapping_raises(self) -> None:
        with pytest.raises(ValidationError):
            build_backbone_config({"model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "lora_rank": 8})

    def test_config_instance_passes_through(self) -> None:
        config = tiny_backbone_config()
        assert build_backbone_config(config) == config

    def test_builds_the_tiny_module(self) -> None:
        backbone = build_backbone(tiny_backbone_config())
        assert isinstance(backbone, TinyVideoBackbone)
        assert backbone.config == tiny_backbone_config()

    def test_unknown_kind_raises(self) -> None:
        class Bogus:
            kind = "sora"

        with pytest.raises(ValueError, match="unknown backbone kind"):
            build_backbone(Bogus())  # type: ignore[arg-type]

    def test_name_registry_is_untouched(self) -> None:
        # build_backbone is a second entry point, not a replacement: AC-05 still enumerates
        # every adapter by name.
        from wam.backbones import available_backbones

        assert available_backbones() == ("flux3", "tiny", "wan_i2v")


# -- backbone swappability (FR-09): module injection ------------------------------------------


class FakeFlowBackbone(nn.Module):
    """A second, independent FlowBackbone in PIXEL space (one video token per frame).

    Deliberately NOT a TinyVideoBackbone subclass and deliberately trivial: its only job is to
    prove ``JointWorldActionModel`` talks to the protocol and never to a concrete backbone.
    """

    def __init__(self) -> None:
        super().__init__()
        self.text_table = nn.Embedding(8, FEATURE_DIM)  # the "text tower" — the frozen part
        self.pixel_in = nn.Linear(3, FEATURE_DIM)
        self.pixel_out = nn.Linear(FEATURE_DIM, 3)
        self.state_in = nn.Linear(STATE_DIM, FEATURE_DIM)

    @property
    def name(self) -> str:
        return "fake-flow"

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM

    def encode_video(self, video) -> torch.Tensor:
        frames = torch.as_tensor(video)
        frames = frames[None] if frames.ndim == 4 else frames
        return frames.float() / 255.0 if frames.dtype == torch.uint8 else frames.float()

    def decode_video(self, video_latents) -> torch.Tensor:
        return torch.as_tensor(video_latents).float()

    def condition_video(self, video) -> torch.Tensor:
        return self.pixel_in(self.encode_video(video)).mean(dim=(2, 3))

    def condition_text(self, text: str) -> torch.Tensor:
        return self.text_table(torch.tensor([[len(text) % 8]]))

    def condition_state(self, state_embedding) -> torch.Tensor:
        emb = torch.as_tensor(state_embedding, dtype=torch.float32)
        return self.state_in(emb)[:, None, :]

    def features(self, video_ctx, text_ctx, state_ctx) -> torch.Tensor:
        batch = video_ctx.shape[0]
        text = text_ctx.expand(batch, -1, -1) if text_ctx.shape[0] == 1 else text_ctx
        return torch.cat([video_ctx, text, state_ctx], dim=1)

    def forward_flow(self, video_latents, t, text_ctx, state_ctx) -> tuple[torch.Tensor, ...]:
        hidden = self.pixel_in(torch.as_tensor(video_latents).float())
        hidden = hidden + torch.as_tensor(t, dtype=torch.float32).reshape(-1, 1, 1, 1, 1)
        return self.pixel_out(hidden), self.features(hidden.mean(dim=(2, 3)), text_ctx, state_ctx)

    def num_video_tokens(self, video_latents=None) -> int:
        return NUM_FRAMES

    def frozen_part_names(self) -> tuple[str, ...]:
        return ("text_table",)


class LatentFlowBackbone(FakeFlowBackbone):
    """FlowBackbone whose ``encode_video`` returns VAE-SHAPED latents ``[B, 8, 2, 3, 4]``.

    The whole point is the shape: nothing in ``JointWorldActionModel`` may assume the
    ``[B, F, H, W, 3]`` pixel layout, and the 24 video tokens (F' x H' x W') must come from
    ``num_video_tokens(latents)`` — the tiny config would claim 8 here.
    """

    LATENT_SHAPE = (8, 2, 3, 4)  # (C, F', H', W')

    def __init__(self) -> None:
        super().__init__()
        self.token_in = nn.Linear(self.LATENT_SHAPE[0], FEATURE_DIM)
        self.token_out = nn.Linear(FEATURE_DIM, self.LATENT_SHAPE[0])

    def encode_video(self, video) -> torch.Tensor:
        pixels = super().encode_video(video)
        numel = int(np.prod(self.LATENT_SHAPE))
        flat = pixels.reshape(pixels.shape[0], -1)[:, :numel]
        return flat.reshape(pixels.shape[0], *self.LATENT_SHAPE).detach()

    def decode_video(self, video_latents) -> torch.Tensor:
        # Shape-correct stand-in, not a true inverse — no test round-trips through it.
        latents = torch.as_tensor(video_latents).float()
        pooled = latents.mean(dim=(1, 3, 4))  # [B, F']
        return pooled[:, :, None, None, None].expand(-1, NUM_FRAMES, IMAGE_HW, IMAGE_HW, 3)

    def forward_flow(self, video_latents, t, text_ctx, state_ctx) -> tuple[torch.Tensor, ...]:
        latents = torch.as_tensor(video_latents).float()
        batch = latents.shape[0]
        channels, *grid = self.LATENT_SHAPE
        tokens = latents.permute(0, 2, 3, 4, 1).reshape(batch, -1, channels)
        hidden = self.token_in(tokens)
        hidden = hidden + torch.as_tensor(t, dtype=torch.float32).reshape(batch, 1, 1)
        velocity = self.token_out(hidden).reshape(batch, *grid, channels).permute(0, 4, 1, 2, 3)
        return velocity, self.features(hidden, text_ctx, state_ctx)

    def num_video_tokens(self, video_latents=None) -> int:
        if video_latents is None:
            return int(np.prod(self.LATENT_SHAPE[1:]))
        _, _, frames, height, width = torch.as_tensor(video_latents).shape
        return frames * height * width


class TestBackboneInjection:
    def test_injected_backbone_is_used_and_drives_the_frozen_registry(self) -> None:
        torch.manual_seed(0)
        fake = FakeFlowBackbone()
        model = JointWorldActionModel(joint_config(), backbone=fake)
        assert model.backbone is fake
        assert set(model.frozen_parts) == {"text_table"}  # NOT tiny's text_embedding/text_pos
        assert model.frozen_parameter_names() == ("backbone.text_table.weight",)

    def test_gradients_reach_the_injected_backbone(self) -> None:
        trainer = JointTrainer(joint_config(), backbone=FakeFlowBackbone())
        trainer.compute_losses(make_batch(batch_size=2))["total"].backward()
        assert TrainingMonitor.module_grad_norms(trainer.model)["backbone"] > 0.0

    def test_pixel_fake_trains(self) -> None:
        trainer = JointTrainer(joint_config(), backbone=FakeFlowBackbone())
        history = trainer.train(make_batch(batch_size=4), steps=5)
        assert all(np.isfinite(entry["total"]) for entry in history)
        assert history[-1]["total"] < history[0]["total"]

    def test_empty_frozen_part_names_yields_an_empty_registry(self) -> None:
        # A backbone that arrives pre-frozen (LoRA-adapted) has nothing left for us to freeze.
        class PreFrozen(FakeFlowBackbone):
            def frozen_part_names(self) -> tuple[str, ...]:
                return ()

        model = JointWorldActionModel(joint_config(), backbone=PreFrozen())
        assert model.frozen_parts == {}
        assert model.frozen_parameter_names() == ()

    def test_non_module_backbone_is_rejected(self) -> None:
        # Structurally a FlowBackbone (an adapter wrapping a third-party pipeline often is),
        # but not an nn.Module — so its parameters would never reach the optimizer.
        class PlainAdapter:
            name = "plain"
            feature_dim = FEATURE_DIM

            def __getattr__(self, item: str):
                return lambda *args, **kwargs: None

        with pytest.raises(TypeError, match="nn.Module"):
            JointWorldActionModel(joint_config(), backbone=PlainAdapter())

    def test_incomplete_backbone_lists_the_missing_members(self) -> None:
        with pytest.raises(TypeError, match="forward_flow") as excinfo:
            JointWorldActionModel(joint_config(), backbone=nn.Linear(2, 2))
        assert "encode_video" in str(excinfo.value)

    def test_model_and_backbone_are_mutually_exclusive(self) -> None:
        model = JointWorldActionModel(joint_config(), backbone=FakeFlowBackbone())
        with pytest.raises(TypeError, match="not both"):
            JointTrainer(joint_config(), model, backbone=FakeFlowBackbone())


class TestLatentSpaceBackbone:
    def test_shapes_line_up_and_the_loss_is_finite(self) -> None:
        trainer = JointTrainer(joint_config(), backbone=LatentFlowBackbone())
        batch = make_batch(batch_size=2)
        out = trainer.model.co_denoise(
            batch, torch.tensor([0.3, 0.7]), generator=torch.Generator().manual_seed(0)
        )
        latent_shape = (2, *LatentFlowBackbone.LATENT_SHAPE)
        assert out["video_velocity_pred"].shape == latent_shape
        assert out["video_velocity_target"].shape == latent_shape
        assert out["features"].shape == (2, 24 + 2, FEATURE_DIM)  # 24 video + text + state
        assert out["video_feature"].shape == (2, FEATURE_DIM)

        losses = trainer.compute_losses(batch)
        assert torch.isfinite(losses["total"])
        losses["total"].backward()
        norms = TrainingMonitor.module_grad_norms(trainer.model)
        for module in ("backbone", "action_encoder", "velocity_head", "action_head"):
            assert norms[module] > 0.0, f"no gradient reached {module}"

    def test_token_count_comes_from_the_batch_not_the_config(self) -> None:
        backbone = LatentFlowBackbone()
        latents = backbone.encode_video(make_batch(batch_size=2)["frames"])
        assert backbone.num_video_tokens(latents) == 24
        assert tiny_backbone_config().num_video_tokens == 8  # what the old code would have used


# -- external training loop: step(), param groups, resume -------------------------------------


class TestTrainerStepAndResume:
    def test_step_matches_train_bitwise(self) -> None:
        from wam.training._utils import prepare_tensor_batch

        batch = make_batch(batch_size=4)
        looped = JointTrainer(joint_config())
        history = looped.train(batch, steps=3)

        manual = JointTrainer(joint_config())
        manual.model.train()
        prepared = prepare_tensor_batch(batch, manual.device)
        manual_history = [manual.step(prepared, step=index) for index in range(3)]

        assert manual_history == history  # identical loss dicts, float for float
        params = zip(looped.model.named_parameters(), manual.model.named_parameters(), strict=True)
        for (name, a), (_, b) in params:
            assert torch.equal(a, b), name

    def test_single_param_group_without_backbone_lr(self) -> None:
        trainer = JointTrainer(joint_config())
        assert len(trainer.optimizer.param_groups) == 1
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(trainer.config.lr)

    def test_backbone_lr_splits_the_param_groups(self) -> None:
        trainer = JointTrainer(joint_config(backbone_lr=1e-4))
        backbone, rest = trainer.optimizer.param_groups
        assert backbone["lr"] == pytest.approx(1e-4)
        assert rest["lr"] == pytest.approx(trainer.config.lr)
        expected = {id(p) for p in trainer.model.backbone.parameters() if p.requires_grad}
        assert {id(p) for p in backbone["params"]} == expected
        assert not {id(p) for p in rest["params"]} & expected

    def test_state_dict_resumes_optimizer_and_rng(self) -> None:
        batch = make_batch(batch_size=4)
        source = JointTrainer(joint_config())
        source.train(batch, steps=2)
        weights = {name: value.clone() for name, value in source.model.state_dict().items()}
        trainer_state = source.state_dict()
        expected = source.train(batch, steps=1)[0]

        resumed = JointTrainer(joint_config())
        resumed.model.load_state_dict(weights)
        resumed.load_state_dict(trainer_state)
        assert resumed.train(batch, steps=1)[0] == expected

    def test_trainable_state_dict_round_trips_with_strict_false(self, tmp_path: Path) -> None:
        trainer = JointTrainer(joint_config())
        trainer.train(make_batch(batch_size=4), steps=2)
        payload = trainer.model.trainable_state_dict()
        assert "backbone.text_embedding.weight" not in payload  # frozen parts are left out
        assert "backbone.video_proj.weight" in payload

        path = tmp_path / "adapters.safetensors"
        trainer.save_checkpoint(path, state_dict=payload)
        with pytest.raises(RuntimeError, match="Missing key"):
            load_joint_checkpoint(path)

        model, _ = load_joint_checkpoint(path, strict=False)
        reference = dict(trainer.model.named_parameters())
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert torch.equal(param, reference[name]), name


class TestEncodeInstructions:
    def test_padding_token_is_one_column_wide(self) -> None:
        # A real text tower pads "" out to its full window; using the whole thing as the
        # padding token would blow up the concat width instead of adding one column.
        from wam.training._utils import encode_instructions

        class WideEmptyTextTower:
            def condition_text(self, text: str) -> torch.Tensor:
                length = len(text.split()) or 512
                return torch.full((1, length, 8), float(length))

        ctx = encode_instructions(WideEmptyTextTower(), ["lift", "lift the cube"], 2)
        assert ctx.shape == (2, 3, 8)
        assert torch.equal(ctx[0, 1:], torch.full((2, 8), 512.0))
        assert torch.equal(ctx[1], torch.full((3, 8), 3.0))


# -- shipped configs x shipped dataset (drift gate) -------------------------------------------


MOCK_D1_EPISODE = REPO_ROOT / "datasets" / "mock-d1" / "d1-0000"


@pytest.mark.skipif(not MOCK_D1_EPISODE.exists(), reason="datasets/mock-d1 not present")
class TestShippedConfigsConsumeShippedDataset:
    """The documented 1:1 config path must consume the documented dataset path: every
    shipped episode producer records MockRobot's 64x64 frames, so the shipped training
    yamls must run one real loss step over datasets/mock-d1 without shape errors."""

    def _batch(self, cfg) -> dict:
        ds = EpisodeDataset(
            MOCK_D1_EPISODE,
            camera=cfg.camera,
            num_frames=cfg.backbone.num_frames,
            chunk_steps=cfg.head.num_steps,
        )
        return collate_episode_batch([ds[0]])

    def test_action_only_yaml_runs_one_loss_step(self) -> None:
        cfg = ActionOnlyConfig.from_yaml(REPO_ROOT / "configs" / "training" / "action_only.yaml")
        torch.manual_seed(cfg.seed)
        losses = ActionOnlyTrainer(cfg).compute_losses(self._batch(cfg))
        assert torch.isfinite(losses["total"])

    def test_joint_yaml_runs_one_loss_step(self) -> None:
        cfg = JointTrainingConfig.from_yaml(REPO_ROOT / "configs" / "training" / "joint.yaml")
        torch.manual_seed(cfg.seed)
        losses = JointTrainer(cfg).compute_losses(self._batch(cfg))
        assert torch.isfinite(losses["total"])
        assert "action_recon" in losses


# -- Monitoring (T-17, R-07) -----------------------------------------------------------------


class TestTrainingMonitor:
    def test_records_history(self) -> None:
        monitor = TrainingMonitor()
        record = monitor.record_step(
            0, {"total": 1.0, "action": 0.5}, grad_norms={"global": 2.0}, update_ratio=0.01
        )
        assert monitor.history == [record]
        assert record["total"] == 1.0
        assert record["grad_norms"]["global"] == 2.0
        assert record["update_ratio"] == 0.01
        assert monitor.ema == pytest.approx(1.0)

    def test_total_defaults_to_sum(self) -> None:
        monitor = TrainingMonitor()
        record = monitor.record_step(0, {"a": 1.0, "b": 2.0})
        assert record["total"] == pytest.approx(3.0)

    def test_nan_loss_raises(self) -> None:
        monitor = TrainingMonitor()
        with pytest.raises(TrainingDiverged, match="non-finite"):
            monitor.record_step(0, {"total": float("nan")})

    def test_inf_component_raises(self) -> None:
        monitor = TrainingMonitor()
        with pytest.raises(TrainingDiverged, match="non-finite"):
            monitor.record_step(0, {"total": 1.0, "video": float("inf")})

    def test_loss_spike_raises_after_warmup(self) -> None:
        monitor = TrainingMonitor(TrainingMonitorConfig(warmup_steps=5, divergence_factor=10.0))
        for step in range(6):
            monitor.record_step(step, {"total": 1.0})
        with pytest.raises(TrainingDiverged, match="exceeds"):
            monitor.record_step(6, {"total": 100.0})

    def test_no_spike_check_during_warmup(self) -> None:
        monitor = TrainingMonitor(TrainingMonitorConfig(warmup_steps=5))
        monitor.record_step(0, {"total": 1.0})
        monitor.record_step(1, {"total": 100.0})  # within warmup: no raise
        assert len(monitor.history) == 2

    def test_trainer_integration_catches_injected_nan(self) -> None:
        trainer = ActionOnlyTrainer(action_only_config())
        batch = make_batch(batch_size=4)
        batch["targets"] = batch["targets"].clone()
        batch["targets"][0, 0, 0] = float("nan")  # poisons the action loss
        with pytest.raises(TrainingDiverged, match="non-finite"):
            trainer.train(batch, steps=3, monitor=TrainingMonitor())

    def test_trainer_integration_records(self) -> None:
        trainer = ActionOnlyTrainer(action_only_config())
        monitor = TrainingMonitor()
        trainer.train(make_batch(batch_size=4), steps=3, monitor=monitor)
        assert len(monitor.history) == 3
        record = monitor.history[0]
        assert record["grad_norms"]["global"] > 0
        assert {"state_encoder", "backbone", "action_head"} <= set(record["grad_norms"])
        assert record["update_ratio"] > 0

    def test_grad_and_update_helpers(self) -> None:
        torch.manual_seed(0)
        model = torch.nn.Linear(2, 2)
        assert TrainingMonitor.global_grad_norm(model) == 0.0
        before = TrainingMonitor.snapshot_params(model)
        model(torch.randn(3, 2)).sum().backward()
        assert TrainingMonitor.global_grad_norm(model) > 0.0
        with torch.no_grad():
            model.weight.add_(1.0)
        assert TrainingMonitor.param_update_ratio(model, before) > 0.0

    def test_to_jsonl_stamps_every_line(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor()
        monitor.record_step(0, {"total": 1.0})
        monitor.record_step(1, {"total": 0.5})
        metadata = RunMetadata.create(
            "mon-run", {"wam_config_version": "0.1.0"}, git_commit="deadbeef"
        )
        path = monitor.to_jsonl(tmp_path / "train.jsonl", metadata)
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(lines) == 3  # metadata + 2 steps
        assert lines[0]["kind"] == "run_metadata"
        assert all(line["run_id"] == "mon-run" for line in lines)
        assert all(line["config_hash"] == metadata.config_hash for line in lines)
        assert [line["step"] for line in lines[1:]] == [0, 1]
        assert all(line["kind"] == "training_step" for line in lines[1:])


class TestResolveFrameContext:
    """T-29 / I-7: what a policy shows the backbone. Both ``predict()`` implementations go
    through this, so the action-only and world-action models cannot be fed different clips —
    an AC-07 comparison between them is only meaningful if they are not."""

    NUM_FRAMES = 4

    def _obs(self, history: np.ndarray | None = None) -> Observation:
        rng = np.random.default_rng(11)
        image = rng.integers(0, 256, (IMAGE_HW, IMAGE_HW, 3), dtype=np.uint8)
        if history is not None:
            history = np.concatenate([history, image[None]], axis=0)
        return Observation(
            images={"front": image},
            state=make_state(rng, ts=0),
            instruction="pick",
            image_history=None if history is None else {"front": history},
        )

    def _window(self, n: int) -> np.ndarray:
        return np.stack([np.full((IMAGE_HW, IMAGE_HW, 3), i, dtype=np.uint8) for i in range(n)])

    def test_without_history_the_single_frame_is_tiled(self) -> None:
        frames = resolve_frame_context(self._obs(), "front", self.NUM_FRAMES)
        assert frames.shape == (self.NUM_FRAMES, IMAGE_HW, IMAGE_HW, 3)
        for i in range(1, self.NUM_FRAMES):
            assert torch.equal(frames[i], frames[0])  # no motion whatsoever

    def test_with_history_the_real_window_is_used(self) -> None:
        obs = self._obs(self._window(self.NUM_FRAMES - 1))
        frames = resolve_frame_context(obs, "front", self.NUM_FRAMES)
        assert frames.shape == (self.NUM_FRAMES, IMAGE_HW, IMAGE_HW, 3)
        assert not torch.equal(frames[0], frames[-1])
        assert torch.equal(frames[-1], torch.as_tensor(obs.images["front"]))

    def test_wrong_length_history_is_rejected_not_resampled(self) -> None:
        """Silently padding or truncating would reintroduce exactly the class of bug T-29 is
        about: the model being fed something other than what it was trained on, quietly."""
        obs = self._obs(self._window(self.NUM_FRAMES))  # one frame too many
        with pytest.raises(ValueError, match="image_history"):
            resolve_frame_context(obs, "front", self.NUM_FRAMES)

    def test_history_not_ending_at_the_observation_is_rejected(self) -> None:
        """The Observation invariant. A history misaligned in time is the failure mode that
        produces a plausible number from the wrong frames — it must not be silently accepted."""
        rng = np.random.default_rng(5)
        obs = Observation(
            images={"front": rng.integers(0, 256, (IMAGE_HW, IMAGE_HW, 3), dtype=np.uint8)},
            state=make_state(rng, ts=0),
            instruction="pick",
            image_history={"front": self._window(self.NUM_FRAMES)},
        )
        with pytest.raises(ValueError, match="last frame"):
            resolve_frame_context(obs, "front", self.NUM_FRAMES)

    def test_missing_camera_is_reported_with_the_available_keys(self) -> None:
        with pytest.raises(KeyError, match="wrist"):
            resolve_frame_context(self._obs(), "wrist", self.NUM_FRAMES)

    def test_history_for_another_camera_is_ignored(self) -> None:
        """A rolling buffer that only tracks one view must not change what a different view
        gets — it falls back to tiling, which is correct rather than an error."""
        obs = self._obs()
        obs.image_history = {"wrist": self._window(self.NUM_FRAMES)}
        frames = resolve_frame_context(obs, "front", self.NUM_FRAMES)
        assert torch.equal(frames[0], frames[-1])

    def test_both_policies_resolve_frames_identically(self) -> None:
        """The AC-07 guarantee, checked rather than documented: same observation in, same clip
        to the backbone, for the action-only and the world-action model alike."""
        import inspect

        for cls in (ActionOnlyModel, JointWorldActionModel):
            source = inspect.getsource(cls.predict)
            assert "resolve_frame_context(" in source, f"{cls.__name__}.predict"
            assert ".expand(" not in source, (
                f"{cls.__name__}.predict tiles frames itself instead of going through "
                "resolve_frame_context — that is how the two paths drifted apart before"
            )

    def test_tiling_and_an_explicitly_tiled_history_agree(self) -> None:
        """The reproducibility guarantee for every pre-T-29 number: the default path is exactly
        'a history of N copies of the current frame', so nothing about the archived runs changed
        when the history branch was added. Verified once against the real d1-full-gen-seed0
        checkpoint on real chunks (bit-identical) and pinned here for the general case."""
        obs = self._obs()
        tiled = resolve_frame_context(obs, "front", self.NUM_FRAMES)
        obs.image_history = {"front": np.repeat(obs.images["front"][None], self.NUM_FRAMES, axis=0)}
        explicit = resolve_frame_context(obs, "front", self.NUM_FRAMES)
        assert torch.equal(tiled, explicit)
