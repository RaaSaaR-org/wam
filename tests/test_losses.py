"""Hand-checked tests for wam.training.losses (T-16/T-17, PRD §10.4)."""

from __future__ import annotations

import pytest
import torch

from wam.training import (
    action_flow_matching_loss,
    action_regression_loss,
    alignment_loss,
    limit_penalty,
    make_flow_targets,
    smoothness_loss,
    video_flow_loss,
)


class TestMakeFlowTargets:
    def test_scalar_t_interpolation(self) -> None:
        x0 = torch.zeros(2, 3, 4)
        x1 = torch.ones(2, 3, 4)
        x_t, velocity = make_flow_targets(x0, x1, 0.25)
        assert torch.allclose(x_t, torch.full((2, 3, 4), 0.25))
        assert torch.allclose(velocity, torch.ones(2, 3, 4))

    def test_endpoints(self) -> None:
        x0 = torch.randn(2, 5)
        x1 = torch.randn(2, 5)
        x_t0, _ = make_flow_targets(x0, x1, 0.0)
        x_t1, _ = make_flow_targets(x0, x1, 1.0)
        assert torch.allclose(x_t0, x0)
        assert torch.allclose(x_t1, x1)

    def test_per_sample_t(self) -> None:
        x0 = torch.zeros(2, 3)
        x1 = torch.ones(2, 3)
        x_t, _ = make_flow_targets(x0, x1, torch.tensor([0.0, 1.0]))
        assert torch.allclose(x_t[0], x0[0])
        assert torch.allclose(x_t[1], x1[1])

    def test_velocity_reconstructs_data(self) -> None:
        # Rectified flow identity: x1 == x_t + (1 - t) * v.
        x0 = torch.randn(3, 4)
        x1 = torch.randn(3, 4)
        t = torch.tensor([0.1, 0.5, 0.9])
        x_t, v = make_flow_targets(x0, x1, t)
        assert torch.allclose(x_t + (1.0 - t)[:, None] * v, x1, atol=1e-6)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            make_flow_targets(torch.zeros(2, 3), torch.zeros(2, 4), 0.5)

    def test_bad_t_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="t must be"):
            make_flow_targets(torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(3))


class TestActionFlowMatchingLoss:
    def test_hand_computed_mse(self) -> None:
        pred = torch.tensor([[[1.0, 2.0]]])
        target = torch.zeros(1, 1, 2)
        # mean(1^2, 2^2) = 2.5
        assert action_flow_matching_loss(pred, target).item() == pytest.approx(2.5)

    def test_zero_at_optimum(self) -> None:
        v = torch.randn(2, 4, 3)
        assert action_flow_matching_loss(v, v.clone()).item() == 0.0

    def test_mask_excludes_samples(self) -> None:
        pred = torch.stack([torch.ones(4, 2), torch.full((4, 2), 3.0)])
        target = torch.zeros(2, 4, 2)
        mask = torch.tensor([1.0, 0.0])
        # Only sample 0 contributes: mean(1^2) = 1.0 (unmasked would be (1+9)/2 = 5).
        assert action_flow_matching_loss(pred, target, mask).item() == pytest.approx(1.0)
        assert action_flow_matching_loss(pred, target).item() == pytest.approx(5.0)

    def test_step_mask(self) -> None:
        pred = torch.ones(1, 2, 3)
        target = torch.zeros(1, 2, 3)
        mask = torch.tensor([[1.0, 0.0]])  # [B, T]
        assert action_flow_matching_loss(pred, target, mask).item() == pytest.approx(1.0)

    def test_all_masked_is_zero(self) -> None:
        loss = action_flow_matching_loss(torch.ones(1, 2, 3), torch.zeros(1, 2, 3), torch.zeros(1))
        assert loss.item() == 0.0

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="mismatch"):
            action_flow_matching_loss(torch.zeros(1, 2, 3), torch.zeros(1, 2, 4))


class TestActionRegressionLoss:
    def test_l2_hand_computed(self) -> None:
        pred = torch.tensor([[1.0, 2.0]])
        target = torch.zeros(1, 2)
        assert action_regression_loss(pred, target, kind="l2").item() == pytest.approx(2.5)

    def test_l1_hand_computed(self) -> None:
        pred = torch.tensor([[1.0, -2.0]])
        target = torch.zeros(1, 2)
        assert action_regression_loss(pred, target, kind="l1").item() == pytest.approx(1.5)

    def test_default_is_l2(self) -> None:
        pred = torch.tensor([[2.0]])
        target = torch.zeros(1, 1)
        assert action_regression_loss(pred, target).item() == pytest.approx(4.0)

    def test_invalid_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            action_regression_loss(torch.zeros(1, 1), torch.zeros(1, 1), kind="huber")  # type: ignore[arg-type]

    def test_mask(self) -> None:
        pred = torch.tensor([[1.0], [5.0]])
        target = torch.zeros(2, 1)
        mask = torch.tensor([True, False])
        assert action_regression_loss(pred, target, kind="l1", mask=mask).item() == pytest.approx(
            1.0
        )


class TestVideoFlowLoss:
    def test_hand_computed(self) -> None:
        pred = torch.full((1, 2, 4, 4, 3), 2.0)
        target = torch.zeros(1, 2, 4, 4, 3)
        assert video_flow_loss(pred, target).item() == pytest.approx(4.0)

    def test_zero_at_optimum(self) -> None:
        v = torch.randn(2, 2, 4, 4, 3)
        assert video_flow_loss(v, v.clone()).item() == 0.0

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="mismatch"):
            video_flow_loss(torch.zeros(1, 2, 4, 4, 3), torch.zeros(1, 3, 4, 4, 3))


class TestAlignmentLoss:
    def test_identical_is_zero(self) -> None:
        feats = torch.randn(4, 8)
        assert alignment_loss(feats, feats.clone()).item() == pytest.approx(0.0, abs=1e-6)

    def test_opposite_is_two(self) -> None:
        feats = torch.randn(4, 8)
        assert alignment_loss(feats, -feats).item() == pytest.approx(2.0, abs=1e-6)

    def test_orthogonal_is_one(self) -> None:
        video = torch.tensor([[1.0, 0.0]])
        action = torch.tensor([[0.0, 1.0]])
        assert alignment_loss(video, action).item() == pytest.approx(1.0, abs=1e-6)

    def test_three_dim_inputs_are_pooled(self) -> None:
        video = torch.randn(3, 5, 8)
        action = video.mean(dim=1, keepdim=True).expand(3, 2, 8)
        assert alignment_loss(video, action).item() == pytest.approx(0.0, abs=1e-5)

    def test_dim_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="matching"):
            alignment_loss(torch.zeros(2, 8), torch.zeros(2, 4))


class TestSmoothnessLoss:
    def test_linear_ramp_is_zero(self) -> None:
        # Second difference of a linear ramp is exactly zero (motion is not penalized).
        ramp = torch.linspace(-1, 1, steps=10).reshape(1, 10, 1).expand(2, 10, 3)
        assert smoothness_loss(ramp.contiguous()).item() == pytest.approx(0.0, abs=1e-7)

    def test_hand_computed_kink(self) -> None:
        # [0, 0, 1]: second difference = 1 - 0 + 0 = 1 -> squared mean = 1.
        targets = torch.tensor([[[0.0], [0.0], [1.0]]])
        assert smoothness_loss(targets).item() == pytest.approx(1.0)

    def test_short_chunks_are_zero(self) -> None:
        assert smoothness_loss(torch.randn(2, 2, 3)).item() == 0.0

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[B, T, D\]"):
            smoothness_loss(torch.zeros(2, 3))


class TestLimitPenalty:
    def test_inside_band_is_zero(self) -> None:
        assert limit_penalty(torch.full((2, 4, 3), 0.9), limit=1.0).item() == 0.0

    def test_hand_computed_outside(self) -> None:
        # |1.5| - 1.0 = 0.5 -> 0.25; |-2| - 1 = 1 -> 1.0; mean = 0.625.
        targets = torch.tensor([[[1.5, -2.0]]])
        assert limit_penalty(targets, limit=1.0).item() == pytest.approx(0.625)

    def test_nonpositive_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="limit"):
            limit_penalty(torch.zeros(1, 1, 1), limit=0.0)


class TestDifferentiability:
    def test_gradients_flow_through_all_losses(self) -> None:
        torch.manual_seed(0)
        pred = torch.randn(2, 4, 3, requires_grad=True)
        target = torch.randn(2, 4, 3)
        video_pred = torch.randn(1, 2, 4, 4, 3, requires_grad=True)
        feats = torch.randn(2, 8, requires_grad=True)
        total = (
            action_flow_matching_loss(pred, target)
            + action_regression_loss(pred, target, kind="l1")
            + smoothness_loss(pred)
            + limit_penalty(pred, limit=0.5)
            + video_flow_loss(video_pred, torch.zeros_like(video_pred))
            + alignment_loss(feats, torch.randn(2, 8))
        )
        total.backward()
        for tensor in (pred, video_pred, feats):
            assert tensor.grad is not None
            assert torch.isfinite(tensor.grad).all()

    def test_flow_targets_carry_gradients(self) -> None:
        x0 = torch.randn(2, 3, requires_grad=True)
        x1 = torch.randn(2, 3, requires_grad=True)
        x_t, v = make_flow_targets(x0, x1, 0.3)
        (x_t.sum() + v.sum()).backward()
        assert x0.grad is not None and x1.grad is not None
