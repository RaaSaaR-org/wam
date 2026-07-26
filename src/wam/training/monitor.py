"""Training monitoring + divergence detection (T-17, R-07, PRD §10.4 Monitoring).

``TrainingMonitor`` records per-step loss components, gradient norms (global and per
top-level module) and the parameter update ratio, and raises ``TrainingDiverged`` when
training goes off the rails:

- any non-finite (NaN/Inf) loss component -> immediate ``TrainingDiverged``;
- total loss > ``divergence_factor`` x EMA(total) after ``warmup_steps`` recorded steps.

R-07 (video loss dominating action learning) is what the per-module gradient norms are for:
comparing e.g. ``grad/backbone`` vs ``grad/action_head`` makes branch imbalance visible.

The monitor is passive: trainers call ``record_step`` once per optimizer step; the history is
exported via ``to_jsonl`` through ``JsonlRunLogger`` so every line is stamped with
``run_id`` + ``config_hash`` (AC-04).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from wam.interfaces.versioning import JsonlRunLogger, RunMetadata

__all__ = ["TrainingDiverged", "TrainingMonitor", "TrainingMonitorConfig"]


class TrainingDiverged(RuntimeError):
    """Raised when a loss is non-finite or the total loss explodes vs its EMA."""


class TrainingMonitorConfig(BaseModel):
    """Divergence-detection thresholds. Frozen (hashable into config_hash)."""

    model_config = ConfigDict(frozen=True)

    ema_beta: float = Field(default=0.98, gt=0.0, lt=1.0)
    divergence_factor: float = Field(default=10.0, gt=1.0)
    warmup_steps: int = Field(default=20, ge=1, description="Recorded steps before ratio checks.")
    check_nonfinite: bool = True


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


class TrainingMonitor:
    """Records loss components / grad norms / update ratios; detects divergence."""

    def __init__(self, config: TrainingMonitorConfig | None = None) -> None:
        self.config = config or TrainingMonitorConfig()
        self.history: list[dict[str, Any]] = []
        self._ema = 0.0  # zero-init + bias correction (Adam-style)
        self._steps_recorded = 0

    @property
    def ema(self) -> float | None:
        """Bias-corrected EMA of the total loss (``None`` before the first record)."""
        if self._steps_recorded == 0:
            return None
        correction = 1.0 - self.config.ema_beta**self._steps_recorded
        return self._ema / correction

    # -- recording ---------------------------------------------------------------------

    def record_step(
        self,
        step: int,
        losses: Mapping[str, Any],
        *,
        grad_norms: Mapping[str, float] | None = None,
        update_ratio: float | None = None,
    ) -> dict[str, Any]:
        """Record one optimizer step; raises ``TrainingDiverged`` on NaN/Inf or loss blow-up.

        ``losses`` maps component name -> float/0-dim tensor; the ``'total'`` entry drives the
        EMA divergence check (defaults to the sum of all components if absent).
        """
        values = {name: _to_float(v) for name, v in losses.items()}
        if not values:
            raise ValueError("losses must contain at least one component")
        total = values["total"] if "total" in values else sum(values.values())

        if self.config.check_nonfinite:
            for name, value in values.items():
                if not math.isfinite(value):
                    raise TrainingDiverged(
                        f"non-finite loss component {name!r}={value} at step {step}"
                    )
            if not math.isfinite(total):
                raise TrainingDiverged(f"non-finite total loss {total} at step {step}")

        ema = self.ema
        if (
            ema is not None
            and self._steps_recorded >= self.config.warmup_steps
            and total > self.config.divergence_factor * ema
        ):
            raise TrainingDiverged(
                f"total loss {total:.6g} exceeds {self.config.divergence_factor} x "
                f"EMA {ema:.6g} at step {step}"
            )

        beta = self.config.ema_beta
        self._ema = beta * self._ema + (1.0 - beta) * total
        self._steps_recorded += 1

        record: dict[str, Any] = {"step": int(step), "losses": values, "total": total}
        if grad_norms is not None:
            record["grad_norms"] = {name: float(v) for name, v in grad_norms.items()}
        if update_ratio is not None:
            record["update_ratio"] = float(update_ratio)
        self.history.append(record)
        return record

    # -- gradient / parameter statistics -------------------------------------------------

    @staticmethod
    def global_grad_norm(model: nn.Module) -> float:
        """L2 norm over all parameter gradients (0.0 if no gradients are populated)."""
        total = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total += float(param.grad.detach().pow(2).sum().item())
        return math.sqrt(total)

    @staticmethod
    def module_grad_norms(model: nn.Module) -> dict[str, float]:
        """``{'global': g, '<child>': g_child, ...}`` gradient norms per top-level child."""
        norms = {"global": TrainingMonitor.global_grad_norm(model)}
        for name, child in model.named_children():
            norms[name] = TrainingMonitor.global_grad_norm(child)
        return norms

    @staticmethod
    def snapshot_params(model: nn.Module) -> dict[str, torch.Tensor]:
        """Detached parameter clones for a later ``param_update_ratio`` comparison."""
        return {name: param.detach().clone() for name, param in model.named_parameters()}

    @staticmethod
    def param_update_ratio(model: nn.Module, before: Mapping[str, torch.Tensor]) -> float:
        """``||theta_now - theta_before|| / (||theta_before|| + eps)`` over all parameters."""
        delta_sq = 0.0
        ref_sq = 0.0
        for name, param in model.named_parameters():
            prev = before.get(name)
            if prev is None:
                continue
            delta_sq += float((param.detach() - prev).pow(2).sum().item())
            ref_sq += float(prev.pow(2).sum().item())
        return math.sqrt(delta_sq) / (math.sqrt(ref_sq) + 1e-12)

    # -- export ------------------------------------------------------------------------

    def to_jsonl(self, path: str | Path, metadata: RunMetadata) -> Path:
        """Write metadata + one ``kind='training_step'`` record per step via JsonlRunLogger."""
        target = Path(path)
        with JsonlRunLogger(target, metadata) as logger:
            logger.log_metadata()
            for record in self.history:
                logger.log({"kind": "training_step", **record})
        return target
