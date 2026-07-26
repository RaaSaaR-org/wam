"""Tests for wam.encoders: StateMLP (FR-02) and ActionChunkEncoder (training only)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from wam.encoders import ActionChunkEncoder, ActionChunkEncoderConfig, StateMLP, StateMLPConfig
from wam.interfaces import (
    ActionChunk,
    ActionEncoder,
    ActionMode,
    IMUState,
    RobotState,
    StateEncoder,
    ValidityMask,
)

NUM_JOINTS = 3
GRIPPER_DIMS = 1
EMBED_DIM = 16
LATENT_DIM = 12
TARGET_DIM = 3


def make_state_config(**overrides) -> StateMLPConfig:
    kwargs: dict = {
        "embedding_dim": EMBED_DIM,
        "hidden_dims": (16, 16),
        "num_joints": NUM_JOINTS,
        "gripper_dims": GRIPPER_DIMS,
    }
    kwargs.update(overrides)
    return StateMLPConfig(**kwargs)


def make_state(validity: ValidityMask | None = None, imu_nan: bool = False) -> RobotState:
    rng = np.random.default_rng(7)
    imu = IMUState(
        orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
        angular_velocity=rng.normal(size=3).astype(np.float32),
        linear_acceleration=rng.normal(size=3).astype(np.float32),
    )
    if imu_nan:
        imu.angular_velocity = np.full(3, np.nan, dtype=np.float32)
    return RobotState(
        timestamp_ns=1_000_000,
        q=rng.normal(size=NUM_JOINTS).astype(np.float32),
        dq=rng.normal(size=NUM_JOINTS).astype(np.float32),
        imu=imu,
        gripper_state=np.array([0.5], dtype=np.float32),
        validity=validity or ValidityMask(),
    )


def make_state_batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(3)
    return {
        "q": torch.randn(batch_size, NUM_JOINTS, generator=g),
        "dq": torch.randn(batch_size, NUM_JOINTS, generator=g),
        "imu": torch.randn(batch_size, 10, generator=g),
        "gripper": torch.rand(batch_size, GRIPPER_DIMS, generator=g),
    }


class TestStateMLPConfig:
    def test_input_dim(self):
        assert make_state_config().input_dim == 2 * NUM_JOINTS + 10 + GRIPPER_DIMS

    def test_hidden_dims_bounds(self):
        with pytest.raises(ValidationError):
            make_state_config(hidden_dims=())  # < 2 linear layers
        with pytest.raises(ValidationError):
            make_state_config(hidden_dims=(8, 8, 8, 8))  # > 4 linear layers
        with pytest.raises(ValidationError):
            make_state_config(hidden_dims=(8, 0))
        make_state_config(hidden_dims=(8,))
        make_state_config(hidden_dims=(8, 8, 8))


class TestStateMLP:
    def test_protocol_conformance(self):
        torch.manual_seed(0)
        enc = StateMLP(make_state_config())
        assert isinstance(enc, StateEncoder)

    def test_embedding_dim_property(self):
        torch.manual_seed(0)
        assert StateMLP(make_state_config()).embedding_dim == EMBED_DIM

    def test_encode_shape_dtype(self):
        torch.manual_seed(0)
        enc = StateMLP(make_state_config())
        emb = enc.encode(make_state())
        assert isinstance(emb, torch.Tensor)
        assert emb.shape == (EMBED_DIM,)
        assert emb.dtype == torch.float32
        assert torch.isfinite(emb).all()

    def test_forward_batch_shape(self):
        torch.manual_seed(0)
        enc = StateMLP(make_state_config())
        out = enc(make_state_batch(5))
        assert out.shape == (5, EMBED_DIM)
        assert torch.isfinite(out).all()

    def test_validity_mask_changes_output(self):
        torch.manual_seed(0)
        enc = StateMLP(make_state_config())
        valid = enc.encode(make_state())
        masked = enc.encode(make_state(validity=ValidityMask(gripper=False)))
        assert not torch.allclose(valid, masked)

    def test_invalid_group_with_nan_is_safe(self):
        torch.manual_seed(0)
        enc = StateMLP(make_state_config())
        emb = enc.encode(make_state(validity=ValidityMask(imu=False), imu_nan=True))
        assert torch.isfinite(emb).all()

    def test_all_groups_invalid(self):
        torch.manual_seed(0)
        enc = StateMLP(make_state_config())
        mask = ValidityMask(q=False, dq=False, imu=False, gripper=False)
        emb = enc.encode(make_state(validity=mask))
        assert emb.shape == (EMBED_DIM,)
        assert torch.isfinite(emb).all()

    def test_forward_validity_mask_changes_output_and_is_nan_safe(self):
        torch.manual_seed(0)
        enc = StateMLP(make_state_config())
        batch = make_state_batch(2)
        base = enc(batch)
        poisoned = dict(batch)
        poisoned["imu"] = batch["imu"].clone()
        poisoned["imu"][0] = float("nan")
        poisoned["validity"] = torch.tensor([[1, 1, 0, 1], [1, 1, 1, 1]], dtype=torch.bool)
        out = enc(poisoned)
        assert torch.isfinite(out).all()
        assert not torch.allclose(out[0], base[0])  # masked row uses missing embedding
        assert torch.allclose(out[1], base[1])  # untouched row unchanged

    def test_gradients_flow_to_all_params(self):
        torch.manual_seed(0)
        enc = StateMLP(make_state_config())
        batch = make_state_batch(4)
        # Each group has at least one valid and one invalid row so every parameter
        # (MLP weights AND all missing embeddings) participates.
        batch["validity"] = torch.tensor(
            [[1, 1, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 0]], dtype=torch.bool
        )
        enc(batch).sum().backward()
        for name, param in enc.named_parameters():
            assert param.grad is not None, f"no grad for {name}"
            assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"
        for name in ("q", "dq", "imu", "gripper"):
            assert enc.missing[name].grad.abs().sum() > 0, f"zero grad for missing[{name}]"

    def test_deterministic_init(self):
        torch.manual_seed(42)
        enc_a = StateMLP(make_state_config())
        torch.manual_seed(42)
        enc_b = StateMLP(make_state_config())
        for (name_a, pa), (name_b, pb) in zip(enc_a.named_parameters(), enc_b.named_parameters()):
            assert name_a == name_b
            assert torch.equal(pa, pb)
        assert torch.equal(enc_a.encode(make_state()), enc_b.encode(make_state()))

    def test_forward_rejects_bad_shapes(self):
        torch.manual_seed(0)
        enc = StateMLP(make_state_config())
        batch = make_state_batch(2)
        batch["q"] = torch.randn(2, NUM_JOINTS + 1)
        with pytest.raises(ValueError):
            enc(batch)


def make_action_encoder_config(**overrides) -> ActionChunkEncoderConfig:
    kwargs: dict = {
        "latent_dim": LATENT_DIM,
        "hidden_dims": (16,),
        "target_dim": TARGET_DIM,
        "gripper_dims": 1,
        "max_steps": 16,
    }
    kwargs.update(overrides)
    return ActionChunkEncoderConfig(**kwargs)


def make_chunk(num_steps: int = 8, constant: bool = False) -> ActionChunk:
    rng = np.random.default_rng(11)
    if constant:
        targets = np.full((num_steps, TARGET_DIM), 0.25, dtype=np.float32)
        gripper = np.full(num_steps, 0.5, dtype=np.float32)
    else:
        targets = rng.uniform(-1, 1, size=(num_steps, TARGET_DIM)).astype(np.float32)
        gripper = rng.uniform(0, 1, size=num_steps).astype(np.float32)
    return ActionChunk(
        mode=ActionMode.JOINT_DELTA, targets=targets, gripper_target=gripper, dt_s=0.05
    )


class TestActionChunkEncoder:
    def test_protocol_conformance(self):
        torch.manual_seed(0)
        enc = ActionChunkEncoder(make_action_encoder_config())
        assert isinstance(enc, ActionEncoder)
        assert enc.latent_dim == LATENT_DIM

    def test_encode_shape(self):
        torch.manual_seed(0)
        enc = ActionChunkEncoder(make_action_encoder_config())
        latent = enc.encode(make_chunk(num_steps=8))
        assert isinstance(latent, torch.Tensor)
        assert latent.shape == (8, LATENT_DIM)
        assert latent.dtype == torch.float32
        assert torch.isfinite(latent).all()

    def test_positional_info_distinguishes_identical_steps(self):
        torch.manual_seed(0)
        enc = ActionChunkEncoder(make_action_encoder_config())
        latent = enc.encode(make_chunk(num_steps=8, constant=True))
        assert not torch.allclose(latent[0], latent[1])

    def test_forward_batch_shape(self):
        torch.manual_seed(0)
        enc = ActionChunkEncoder(make_action_encoder_config())
        g = torch.Generator().manual_seed(5)
        targets = torch.randn(2, 8, TARGET_DIM, generator=g)
        gripper = torch.rand(2, 8, 1, generator=g)
        out = enc(targets, gripper)
        assert out.shape == (2, 8, LATENT_DIM)

    def test_rejects_too_many_steps(self):
        torch.manual_seed(0)
        enc = ActionChunkEncoder(make_action_encoder_config(max_steps=8))
        with pytest.raises(ValueError):
            enc.encode(make_chunk(num_steps=9))

    def test_rejects_wrong_target_dim(self):
        torch.manual_seed(0)
        enc = ActionChunkEncoder(make_action_encoder_config(target_dim=TARGET_DIM + 1))
        with pytest.raises(ValueError):
            enc.encode(make_chunk())

    def test_gripper_broadcast_to_multi_dim(self):
        torch.manual_seed(0)
        enc = ActionChunkEncoder(make_action_encoder_config(gripper_dims=2))
        latent = enc.encode(make_chunk(num_steps=8))
        assert latent.shape == (8, LATENT_DIM)

    def test_gradients_flow_to_all_params(self):
        torch.manual_seed(0)
        enc = ActionChunkEncoder(make_action_encoder_config())
        g = torch.Generator().manual_seed(5)
        targets = torch.randn(2, 8, TARGET_DIM, generator=g)
        gripper = torch.rand(2, 8, 1, generator=g)
        enc(targets, gripper).sum().backward()
        for name, param in enc.named_parameters():
            assert param.grad is not None, f"no grad for {name}"
            assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"
        # Positional rows beyond T=8 must be untouched; used rows must receive gradient.
        pos_grad = enc.pos_embedding.weight.grad
        assert pos_grad[:8].abs().sum() > 0
        assert pos_grad[8:].abs().sum() == 0

    def test_deterministic_init(self):
        torch.manual_seed(42)
        enc_a = ActionChunkEncoder(make_action_encoder_config())
        torch.manual_seed(42)
        enc_b = ActionChunkEncoder(make_action_encoder_config())
        for pa, pb in zip(enc_a.parameters(), enc_b.parameters()):
            assert torch.equal(pa, pb)
        chunk = make_chunk()
        assert torch.equal(enc_a.encode(chunk), enc_b.encode(chunk))
