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
