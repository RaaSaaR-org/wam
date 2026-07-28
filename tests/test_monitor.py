"""Tests for the ``TrainingMonitor`` parameter-snapshot helpers (T-17, R-07).

The record/divergence behaviour is covered in ``tests/test_training.py``; what lives here is
the snapshot surface a large adapted backbone depends on, where cloning every frozen weight
once per step is the difference between a monitored run and an OOM.
"""

from __future__ import annotations

import torch
from torch import nn

from wam.training import TrainingMonitor


def _model() -> nn.Module:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    for param in model[0].parameters():  # stands in for a frozen backbone
        param.requires_grad_(False)
    return model


class TestSnapshotParams:
    def test_default_covers_every_parameter(self) -> None:
        snapshot = TrainingMonitor.snapshot_params(_model())
        assert set(snapshot) == {"0.weight", "0.bias", "1.weight", "1.bias"}

    def test_trainable_only_skips_frozen(self) -> None:
        snapshot = TrainingMonitor.snapshot_params(_model(), trainable_only=True)
        assert set(snapshot) == {"1.weight", "1.bias"}

    def test_snapshot_is_a_detached_clone(self) -> None:
        model = _model()
        snapshot = TrainingMonitor.snapshot_params(model, trainable_only=True)
        with torch.no_grad():
            model[1].weight.add_(1.0)
        assert not torch.equal(snapshot["1.weight"], model[1].weight)
        assert not snapshot["1.weight"].requires_grad

    def test_update_ratio_over_a_trainable_only_snapshot(self) -> None:
        # param_update_ratio ignores names absent from `before`, so a trainable-only snapshot
        # measures the trainable subspace instead of being diluted by frozen base weights.
        model = _model()
        before = TrainingMonitor.snapshot_params(model, trainable_only=True)
        assert TrainingMonitor.param_update_ratio(model, before) == 0.0
        with torch.no_grad():
            model[1].weight.add_(1.0)
        assert TrainingMonitor.param_update_ratio(model, before) > 0.0
