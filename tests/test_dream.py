"""Dream sampling + motion metrics (T-35).

The load-bearing test is :class:`TestSamplerAgainstTheTrainingPath`: rectified flow has a
CONSTANT velocity along its path (``v = x1 - x0`` does not depend on t), so Euler integration
with the true field is exact at any step count. A backbone that returns
``co_denoise``'s own velocity target must therefore drive the sampler onto the clean latent
to floating-point precision — no tolerance argument, no "close enough". That pins the direction,
the grid and the update rule at once, which matters because getting any of them wrong still
produces a finite, plausible-looking clip (the same failure mode T-30 documented for actions).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from wam.backbones.tiny import TinyBackboneConfig, TinyVideoBackbone
from wam.decoders import ActionHeadConfig
from wam.encoders import ActionChunkEncoderConfig, StateMLPConfig
from wam.evaluation.dream import (
    MOTION_FLOOR_RATIO,
    STATIC_THRESHOLD,
    as_frames_255,
    build_report,
    measure_clips,
    motion_energy,
    motion_ratio,
    pair_distance,
    sample_video,
    static_fraction,
    vae_roundtrip,
)
from wam.training.joint import JointTrainingConfig, JointWorldActionModel
from wam.training.losses import make_flow_targets

NUM_JOINTS = 4
NUM_FRAMES = 4
IMAGE_HW = 16
NUM_STEPS = 8
TARGET_DIM = NUM_JOINTS
GRIPPER_DIMS = 1
FEATURE_DIM = 32
STATE_DIM = 16


def tiny_joint_model(seed: int = 0) -> JointWorldActionModel:
    torch.manual_seed(seed)
    config = JointTrainingConfig(
        state=StateMLPConfig(embedding_dim=STATE_DIM, hidden_dims=(32,), num_joints=NUM_JOINTS),
        backbone=TinyBackboneConfig(
            feature_dim=FEATURE_DIM,
            patch_size=8,
            depth=1,
            num_heads=4,
            num_frames=NUM_FRAMES,
            image_hw=(IMAGE_HW, IMAGE_HW),
            state_embedding_dim=STATE_DIM,
        ),
        action_encoder=ActionChunkEncoderConfig(
            latent_dim=16, target_dim=TARGET_DIM, hidden_dims=(32,)
        ),
        head=ActionHeadConfig(
            feature_dim=FEATURE_DIM, num_steps=NUM_STEPS, target_dim=TARGET_DIM, hidden_dims=(64,)
        ),
        seed=0,
    )
    model = JointWorldActionModel(config)
    model.eval()
    return model


def make_batch(batch_size: int = 2, seed: int = 1) -> dict:
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


class OracleFlowBackbone(TinyVideoBackbone):
    """A backbone whose ``forward_flow`` returns the TRUE rectified-flow velocity.

    Not a mock of the interface — a real ``TinyVideoBackbone`` with one method replaced, so
    encode/decode, conditioning and the feature contract are the shipped ones and only the field
    is substituted. ``v = clean - noise`` is what ``make_flow_targets`` hands the loss, so a
    sampler faithful to the training path must integrate it exactly onto ``clean``.
    """

    def set_oracle(self, noise: torch.Tensor, clean: torch.Tensor) -> None:
        self._oracle = (noise, clean)
        self.seen_t: list[float] = []
        self.seen_z: list[torch.Tensor] = []

    def forward_flow(self, video_latents, t, text_ctx, state_ctx):
        noise, clean = self._oracle
        self.seen_t.append(float(torch.as_tensor(t).reshape(-1)[0]))
        self.seen_z.append(torch.as_tensor(video_latents).clone())
        features = torch.zeros(
            clean.shape[0], self.config.num_video_tokens, self.config.feature_dim
        )
        return clean - noise, features


def oracle_model(batch: dict) -> tuple[JointWorldActionModel, torch.Tensor, torch.Tensor]:
    """A joint model whose backbone integrates the exact field, plus that field's endpoints."""
    model = tiny_joint_model()
    config = model.config.backbone
    backbone = OracleFlowBackbone(config)
    clean = backbone.encode_video(batch["frames"])
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(0), dtype=clean.dtype)
    backbone.set_oracle(noise, clean)
    model.backbone = backbone
    return model, noise, clean


# ---- metrics ----------------------------------------------------------------------------


class TestFrameNormalization:
    def test_uint8_is_already_0_255(self):
        frames = np.full((1, 2, 4, 4, 3), 200, dtype=np.uint8)
        assert as_frames_255(frames).max() == pytest.approx(200.0)

    def test_float_is_wams_0_1_convention_and_gets_scaled(self):
        frames = np.full((1, 2, 4, 4, 3), 1.0, dtype=np.float32)
        assert as_frames_255(frames).max() == pytest.approx(255.0)

    def test_a_second_normalization_is_rejected_not_silently_doubled(self):
        # The guard that caught a real double-scale during T-35's first run.
        once = as_frames_255(np.full((1, 2, 4, 4, 3), 1.0, dtype=np.float32))
        with pytest.raises(ValueError, match="0, 1"):
            as_frames_255(once)

    def test_unbatched_clip_gets_a_batch_axis(self):
        assert as_frames_255(np.zeros((2, 4, 4, 3), dtype=np.uint8)).shape == (1, 2, 4, 4, 3)

    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match=r"\[B, F, H, W, 3\]"):
            as_frames_255(np.zeros((2, 4, 4), dtype=np.uint8))


class TestMotionMetrics:
    def test_a_frozen_clip_has_zero_motion(self):
        frames = np.tile(np.full((1, 1, 4, 4, 3), 130, dtype=np.uint8), (1, 5, 1, 1, 1))
        assert motion_energy(frames) == 0.0
        assert static_fraction(frames) == 1.0

    def test_motion_is_the_mean_absolute_frame_difference_in_0_255(self):
        frames = np.zeros((1, 3, 2, 2, 3), dtype=np.uint8)
        frames[0, 1] = 10
        frames[0, 2] = 40
        # |10-0| then |40-10| over two pairs -> (10 + 30) / 2
        assert motion_energy(frames) == pytest.approx(20.0)

    def test_single_frame_clip_cannot_move(self):
        assert motion_energy(np.zeros((1, 1, 4, 4, 3), dtype=np.uint8)) == 0.0
        assert static_fraction(np.zeros((1, 1, 4, 4, 3), dtype=np.uint8)) == 1.0

    def test_static_fraction_counts_pairs_below_one_grey_level(self):
        frames = np.zeros((1, 3, 4, 4, 3), dtype=np.float32)
        frames[0, 1] = (STATIC_THRESHOLD * 0.5) / 255.0  # a still pair
        frames[0, 2] = (STATIC_THRESHOLD * 8.0) / 255.0  # a moving pair
        assert static_fraction(frames) == pytest.approx(0.5)

    def test_measure_clips_reports_geometry_and_per_clip_motion(self):
        frames = np.zeros((3, 4, 8, 6, 3), dtype=np.uint8)
        frames[1, 1:] = 20
        metrics = measure_clips(frames, arm="gt")
        assert (metrics.clips, metrics.frames, metrics.height, metrics.width) == (3, 4, 8, 6)
        assert len(metrics.motion_per_clip) == 3
        assert metrics.motion_per_clip[0] == 0.0 and metrics.motion_per_clip[1] > 0

    def test_pair_distance_needs_matching_shapes(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            pair_distance(
                np.zeros((1, 2, 4, 4, 3), dtype=np.uint8), np.zeros((1, 3, 4, 4, 3), dtype=np.uint8)
            )

    def test_motion_ratio_refuses_a_zero_reference(self):
        with pytest.raises(ValueError, match="must be > 0"):
            motion_ratio(1.0, 0.0)


class TestReport:
    def _arms(self, dream_scale: float) -> dict:
        rng = np.random.default_rng(0)
        recon = rng.integers(0, 256, (2, 4, 8, 8, 3)).astype(np.uint8)
        dream = np.zeros_like(recon, dtype=np.float32)
        dream[:, 1:] = dream_scale  # motion in [0, 1] units
        return {"recon": recon, "dream": dream}

    def test_a_still_dream_gets_the_static_verdict(self):
        report = build_report(self._arms(0.0))
        assert report.verdicts["dream.moves"] == "STATIC"
        assert report.motion_ratio["dream"] < MOTION_FLOOR_RATIO

    def test_a_moving_dream_clears_the_floor(self):
        report = build_report(self._arms(1.0))  # a full-range step every frame
        assert report.motion_ratio["dream"] >= MOTION_FLOOR_RATIO
        assert report.verdicts["dream.moves"] == "MOVES"

    def test_the_reference_arm_must_be_present(self):
        with pytest.raises(KeyError, match="reference arm"):
            build_report({"dream": np.zeros((1, 2, 4, 4, 3), dtype=np.uint8)})

    def test_fine_tune_verdict_needs_the_seed_null_to_beat(self):
        rng = np.random.default_rng(1)
        recon = rng.integers(0, 256, (2, 4, 8, 8, 3)).astype(np.uint8)
        base = rng.integers(0, 256, (2, 4, 8, 8, 3)).astype(np.uint8)
        arms = {"recon": recon, "base": base, "base_seed1": base + 1, "lora": base + 60}
        report = build_report(
            arms, pairs={"lora_vs_base": ("lora", "base"), "base_seed_null": ("base", "base_seed1")}
        )
        assert report.verdicts["fine_tune_changed_the_prior"] == "CHANGED"

    def test_a_fine_tune_inside_its_own_sampling_noise_is_indistinguishable(self):
        rng = np.random.default_rng(2)
        recon = rng.integers(0, 256, (2, 4, 8, 8, 3)).astype(np.uint8)
        base = rng.integers(0, 100, (2, 4, 8, 8, 3)).astype(np.uint8)
        arms = {"recon": recon, "base": base, "base_seed1": base + 30, "lora": base + 2}
        report = build_report(
            arms, pairs={"lora_vs_base": ("lora", "base"), "base_seed_null": ("base", "base_seed1")}
        )
        assert report.verdicts["fine_tune_changed_the_prior"] == "INDISTINGUISHABLE"


# ---- the sampler ------------------------------------------------------------------------


class TestSamplerAgainstTheTrainingPath:
    def test_the_true_field_integrates_exactly_onto_the_observation(self):
        """Euler on a constant field is exact — so this holds at ANY step count."""
        batch = make_batch()
        model, _, clean = oracle_model(batch)
        for steps in (1, 4, 32):
            frames = sample_video(model, batch, steps=steps, seed=0)
            expected = model.backbone.decode_video(clean)
            assert torch.allclose(frames, expected, atol=1e-5), f"steps={steps}"

    def test_the_z_handed_to_the_backbone_is_the_training_interpolant(self):
        batch = make_batch()
        model, noise, clean = oracle_model(batch)
        sample_video(model, batch, steps=8, seed=0)
        for t, z in zip(model.backbone.seen_t, model.backbone.seen_z):
            expected, _ = make_flow_targets(noise, clean, t)
            assert torch.allclose(z, expected, atol=1e-5), f"t={t}"

    def test_the_grid_starts_at_t0_and_stops_short_of_one(self):
        """t=1.0 is a level the flow was never trained at (t ~ U[0, 1))."""
        batch = make_batch()
        model, _, _ = oracle_model(batch)
        sample_video(model, batch, steps=4, seed=0)
        assert model.backbone.seen_t == pytest.approx([0.0, 0.25, 0.5, 0.75])
        assert max(model.backbone.seen_t) < 1.0

    def test_a_warm_start_shortens_the_grid_from_t0(self):
        batch = make_batch()
        model, _, _ = oracle_model(batch)
        sample_video(model, batch, steps=2, seed=0, t0=0.5)
        assert model.backbone.seen_t == pytest.approx([0.5, 0.75])

    def test_integrating_backwards_would_not_reach_the_observation(self):
        """Guards the sign: the same loop with a negated field must FAIL the exact test."""
        batch = make_batch()
        model, noise, clean = oracle_model(batch)
        model.backbone.set_oracle(clean, noise)  # field reversed
        frames = sample_video(model, batch, steps=8, seed=0)
        assert not torch.allclose(frames, model.backbone.decode_video(clean), atol=1e-3)


class TestSamplerContract:
    def test_it_runs_on_a_stock_tiny_model_and_returns_pixels(self):
        batch = make_batch()
        model = tiny_joint_model()
        frames = sample_video(model, batch, steps=3, seed=0)
        assert frames.shape == (2, NUM_FRAMES, IMAGE_HW, IMAGE_HW, 3)
        # An untrained model's latents decode out of range and the metric clips them; what must
        # NOT happen is the sampler being unmeasurable, which is why the guard clips here and
        # only rejects a clearly-0-255 array.
        assert measure_clips(frames, arm="sample").motion >= 0.0

    def test_the_seed_is_the_only_source_of_variation(self):
        batch = make_batch()
        model = tiny_joint_model()
        a = sample_video(model, batch, steps=3, seed=7)
        b = sample_video(model, batch, steps=3, seed=7)
        c = sample_video(model, batch, steps=3, seed=8)
        assert torch.equal(a, b)
        assert not torch.equal(a, c)

    def test_state_conditioning_actually_reaches_the_sample(self):
        """The whole reason this route exists next to the diffusers 'generate future' tab."""
        batch = make_batch()
        model = tiny_joint_model()
        other = dict(batch)
        other["q"] = batch["q"] + 5.0
        assert not torch.equal(
            sample_video(model, batch, steps=3, seed=0),
            sample_video(model, other, steps=3, seed=0),
        )

    def test_anchoring_pins_the_leading_frames_to_the_observation(self):
        batch = make_batch()
        model = tiny_joint_model()
        frames = sample_video(model, batch, steps=3, seed=0, anchor_latent_frames=2)
        observed = model.backbone.encode_video(batch["frames"])
        assert torch.allclose(frames[:, :2], observed[:, :2], atol=1e-6)
        assert not torch.allclose(frames[:, 2:], observed[:, 2:], atol=1e-3)

    def test_anchoring_needs_the_backbone_to_declare_its_frame_axis(self):
        batch = make_batch()
        model = tiny_joint_model()

        class Undeclared(TinyVideoBackbone):
            latent_frame_axis = None

        model.backbone = Undeclared(model.config.backbone)
        with pytest.raises(TypeError, match="latent_frame_axis"):
            sample_video(model, batch, steps=2, seed=0, anchor_latent_frames=1)

    def test_free_sampling_works_without_a_declared_axis(self):
        batch = make_batch()
        model = tiny_joint_model()

        class Undeclared(TinyVideoBackbone):
            latent_frame_axis = None

        model.backbone = Undeclared(model.config.backbone)
        assert sample_video(model, batch, steps=2, seed=0).shape[0] == 2

    def test_more_anchors_than_frames_is_rejected(self):
        batch = make_batch()
        model = tiny_joint_model()
        with pytest.raises(ValueError, match="anchor_latent_frames"):
            sample_video(model, batch, steps=2, seed=0, anchor_latent_frames=NUM_FRAMES + 1)

    @pytest.mark.parametrize("steps,t0", [(0, 0.0), (2, 1.0), (2, -0.1)])
    def test_invalid_schedules_are_rejected(self, steps, t0):
        batch = make_batch()
        model = tiny_joint_model()
        with pytest.raises(ValueError):
            sample_video(model, batch, steps=steps, seed=0, t0=t0)

    def test_roundtrip_is_the_identity_on_the_tiny_backbone(self):
        batch = make_batch()
        model = tiny_joint_model()
        assert torch.allclose(
            vae_roundtrip(model, batch["frames"]), batch["frames"].float() / 255.0, atol=1e-6
        )


# ---- the script's window handling --------------------------------------------------------


class TestWindowSelection:
    def test_clamped_start_of_episode_windows_are_dropped(self):
        from dream import drop_padded_windows

        padded = {"frames": torch.full((4, 2, 2, 3), 7, dtype=torch.uint8)}
        moving = {"frames": torch.arange(4 * 2 * 2 * 3, dtype=torch.uint8).reshape(4, 2, 2, 3)}
        kept, dropped = drop_padded_windows([padded, moving, padded])
        assert dropped == 2 and kept == [moving]

    def test_a_genuinely_still_clip_survives(self):
        """0.07 motion at the end of an episode is data, not an artifact — exact equality only."""
        from dream import drop_padded_windows

        nearly = torch.full((4, 2, 2, 3), 7, dtype=torch.uint8)
        nearly[2, 0, 0, 0] = 8
        kept, dropped = drop_padded_windows([{"frames": nearly}])
        assert dropped == 0 and len(kept) == 1

    def test_strip_anchor_removes_exactly_the_copied_pixel_frames(self):
        from dream import strip_anchor

        frames = torch.zeros((1, 9, 2, 2, 3))
        assert strip_anchor(frames, 0, 4).shape[1] == 9
        assert strip_anchor(frames, 1, 4).shape[1] == 8  # 1 + 0*4 covered
        assert strip_anchor(frames, 2, 4).shape[1] == 4  # 1 + 1*4 covered


class TestSpaceBatchPath:
    """`batch_from_windows` is what the ZeroGPU Space samples from — it must agree with the
    local path, because a Space that assembles its batch differently is measuring a different
    thing under the same arm names."""

    def _window(self, seed: int) -> dict:
        from wam.interfaces.schema import IMUState, RobotState, ValidityMask

        rng = np.random.default_rng(seed)
        state = RobotState(
            timestamp_ns=seed,
            q=rng.standard_normal(NUM_JOINTS).astype(np.float32),
            dq=rng.standard_normal(NUM_JOINTS).astype(np.float32),
            imu=IMUState(
                orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
                angular_velocity=np.zeros(3, dtype=np.float32),
                linear_acceleration=np.zeros(3, dtype=np.float32),
            ),
            gripper_state=np.array([0.5], dtype=np.float32),
            validity=ValidityMask(q=True, dq=True, imu=False, gripper=True),
        )
        return {
            "frames": rng.integers(
                0, 256, (NUM_FRAMES, IMAGE_HW, IMAGE_HW, 3), dtype=np.dtype("uint8")
            ),
            "state": state,
            "label": rng.standard_normal(NUM_STEPS * (NUM_JOINTS + 1)).astype(np.float32) * 0.01,
            "episode": seed,
            "start": 10,
        }

    def test_it_produces_a_batch_the_sampler_accepts(self):
        from dream import batch_from_windows

        batch = batch_from_windows([self._window(0), self._window(1)], "pick the apple")
        model = tiny_joint_model()
        frames = sample_video(model, batch, steps=2, seed=0)
        assert frames.shape == (2, NUM_FRAMES, IMAGE_HW, IMAGE_HW, 3)

    def test_the_keys_match_the_local_dataset_path(self):
        from dream import batch_from_windows

        batch = batch_from_windows([self._window(0)], "pick the apple")
        for key in ("frames", "q", "dq", "imu", "gripper", "validity", "targets", "gripper_target"):
            assert key in batch, key
        assert batch["targets"].shape == (1, NUM_STEPS, NUM_JOINTS)
        assert batch["gripper_target"].shape == (1, NUM_STEPS)

    def test_clamped_windows_are_dropped_here_too(self):
        from dream import batch_from_windows

        padded = self._window(0)
        padded["frames"] = np.tile(padded["frames"][:1], (NUM_FRAMES, 1, 1, 1))
        batch = batch_from_windows([padded, self._window(1)], "pick the apple")
        assert batch["frames"].shape[0] == 1

    def test_an_all_padded_set_fails_loudly_rather_than_scoring_nothing(self):
        from dream import batch_from_windows

        padded = self._window(0)
        padded["frames"] = np.tile(padded["frames"][:1], (NUM_FRAMES, 1, 1, 1))
        with pytest.raises(ValueError, match="no usable windows"):
            batch_from_windows([padded], "pick the apple")


class TestCheckpointResolution:
    """The Space downloads a repo root; `load_checkpoint_raw` opens a file. One resolver."""

    def test_a_directory_resolves_to_its_safetensors(self, tmp_path):
        from dream import resolve_checkpoint

        (tmp_path / "model.safetensors").write_bytes(b"")
        (tmp_path / "trainer_state.pt").write_bytes(b"")  # 660 MB in reality; never wanted
        assert resolve_checkpoint(tmp_path) == str(tmp_path / "model.safetensors")

    def test_a_file_is_passed_through(self, tmp_path):
        from dream import resolve_checkpoint

        path = tmp_path / "step-020000.safetensors"
        path.write_bytes(b"")
        assert resolve_checkpoint(path) == str(path)

    def test_a_directory_without_one_says_what_it_found(self, tmp_path):
        from dream import resolve_checkpoint

        (tmp_path / "other.safetensors").write_bytes(b"")
        with pytest.raises(FileNotFoundError, match="other.safetensors"):
            resolve_checkpoint(tmp_path)
