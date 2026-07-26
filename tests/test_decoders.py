"""Tests for wam.decoders.ActionHead (FR-04): bounded chunked action decoding."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from wam.decoders import ActionHead, ActionHeadConfig
from wam.interfaces import ActionChunk, ActionDecoder, ActionMode, CanonicalSpaceSpec

FEATURE_DIM = 24
NUM_STEPS = 8
TARGET_DIM = 3

SPEC = CanonicalSpaceSpec(joint_names=("j0", "j1", "j2"), gripper_dims=1)


def make_config(**overrides) -> ActionHeadConfig:
    kwargs: dict = {
        "feature_dim": FEATURE_DIM,
        "num_steps": NUM_STEPS,
        "target_dim": TARGET_DIM,
        "gripper_dims": 1,
        "mode": ActionMode.JOINT_DELTA,
        "dt_s": 0.05,
        "hidden_dims": (16,),
    }
    kwargs.update(overrides)
    return ActionHeadConfig(**kwargs)


def make_features(*shape: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(9)
    return torch.randn(*shape, generator=g)


class TestActionHeadConfig:
    def test_num_steps_range(self):
        with pytest.raises(ValidationError):
            make_config(num_steps=7)
        with pytest.raises(ValidationError):
            make_config(num_steps=33)
        assert make_config(num_steps=8).num_steps == 8
        assert make_config(num_steps=32).num_steps == 32

    def test_dt_s_positive(self):
        with pytest.raises(ValidationError):
            make_config(dt_s=0.0)


class TestActionHead:
    def test_protocol_conformance(self):
        torch.manual_seed(0)
        head = ActionHead(make_config())
        assert isinstance(head, ActionDecoder)

    def test_forward_shapes_and_bounds(self):
        torch.manual_seed(0)
        head = ActionHead(make_config(gripper_dims=2))
        out = head(make_features(4, FEATURE_DIM))
        assert set(out.keys()) == {"targets", "gripper"}
        assert out["targets"].shape == (4, NUM_STEPS, TARGET_DIM)
        assert out["gripper"].shape == (4, NUM_STEPS, 2)
        assert (out["targets"] > -1).all() and (out["targets"] < 1).all()
        assert (out["gripper"] > 0).all() and (out["gripper"] < 1).all()

    def test_forward_rejects_bad_shape(self):
        torch.manual_seed(0)
        head = ActionHead(make_config())
        with pytest.raises(ValueError):
            head(make_features(4, FEATURE_DIM + 1))

    def test_decode_valid_chunk(self):
        torch.manual_seed(0)
        head = ActionHead(make_config())
        chunk = head.decode(make_features(FEATURE_DIM))
        assert isinstance(chunk, ActionChunk)
        assert chunk.validate(SPEC) == []
        assert chunk.mode is ActionMode.JOINT_DELTA
        assert chunk.num_steps == NUM_STEPS
        assert chunk.dt_s == pytest.approx(0.05)
        assert chunk.targets.shape == (NUM_STEPS, TARGET_DIM)
        assert chunk.targets.dtype == np.float32
        assert chunk.gripper_target.shape == (NUM_STEPS,)
        assert chunk.gripper_target.dtype == np.float32

    def test_decode_pools_leading_dims(self):
        torch.manual_seed(0)
        head = ActionHead(make_config())
        chunk_1d = head.decode(make_features(FEATURE_DIM))
        # A token sequence [S, F] whose mean equals the single feature vector decodes equally.
        stacked = make_features(FEATURE_DIM).unsqueeze(0).repeat(5, 1)
        chunk_2d = head.decode(stacked)
        assert chunk_2d.validate(SPEC) == []
        np.testing.assert_allclose(chunk_1d.targets, chunk_2d.targets, atol=1e-6)

    def test_decode_rejects_wrong_feature_dim(self):
        torch.manual_seed(0)
        head = ActionHead(make_config())
        with pytest.raises(ValueError):
            head.decode(make_features(FEATURE_DIM + 2))

    def test_decode_multi_gripper_dims_still_valid(self):
        torch.manual_seed(0)
        head = ActionHead(make_config(gripper_dims=2))
        chunk = head.decode(make_features(FEATURE_DIM))
        assert chunk.validate(SPEC) == []
        assert chunk.gripper_target.shape == (NUM_STEPS,)

    def test_decode_ee_delta_mode(self):
        torch.manual_seed(0)
        head = ActionHead(make_config(mode=ActionMode.EE_DELTA, target_dim=7))
        chunk = head.decode(make_features(FEATURE_DIM))
        assert chunk.mode is ActionMode.EE_DELTA
        assert chunk.validate(SPEC) == []  # SPEC.target_dim(EE_DELTA) == 3 + 4 == 7

    def test_gradients_flow_to_all_params_and_input(self):
        torch.manual_seed(0)
        head = ActionHead(make_config())
        features = make_features(3, FEATURE_DIM).requires_grad_(True)
        out = head(features)
        (out["targets"].sum() + out["gripper"].sum()).backward()
        assert features.grad is not None
        for name, param in head.named_parameters():
            assert param.grad is not None, f"no grad for {name}"
            assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"
            assert param.grad.abs().sum() > 0, f"zero grad for {name}"

    def test_decode_requires_no_grad_context(self):
        torch.manual_seed(0)
        head = ActionHead(make_config())
        features = make_features(FEATURE_DIM).requires_grad_(True)
        chunk = head.decode(features)  # must not raise despite requires_grad input
        assert chunk.validate(SPEC) == []

    def test_deterministic_init_and_decode(self):
        torch.manual_seed(42)
        head_a = ActionHead(make_config())
        torch.manual_seed(42)
        head_b = ActionHead(make_config())
        for pa, pb in zip(head_a.parameters(), head_b.parameters()):
            assert torch.equal(pa, pb)
        chunk_a = head_a.decode(make_features(FEATURE_DIM))
        chunk_b = head_b.decode(make_features(FEATURE_DIM))
        np.testing.assert_array_equal(chunk_a.targets, chunk_b.targets)
        np.testing.assert_array_equal(chunk_a.gripper_target, chunk_b.gripper_target)
