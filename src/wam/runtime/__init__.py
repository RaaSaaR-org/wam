"""Closed-loop executor (receding horizon, replanning) and inference server.

The checkpoint policies are exported lazily so ``import wam.runtime`` stays torch-free;
everything else here is numpy/pydantic only.
"""

from __future__ import annotations

from typing import Any

from wam.runtime.executor import (
    ClosedLoopExecutor,
    ExecutorConfig,
    RolloutResult,
    run_rollouts,
)
from wam.runtime.mock_loop import (
    DEFAULT_INSTRUCTION,
    DummyPolicy,
    LoopResult,
    run_mock_loop,
)

__all__ = [
    "DEFAULT_INSTRUCTION",
    "CheckpointPolicy",
    "ClosedLoopExecutor",
    "DummyPolicy",
    "ExecutorConfig",
    "JointCheckpointPolicy",
    "LoopResult",
    "RolloutResult",
    "run_mock_loop",
    "run_rollouts",
]

_LAZY_POLICIES = ("CheckpointPolicy", "JointCheckpointPolicy")


def __getattr__(name: str) -> Any:
    if name in _LAZY_POLICIES:
        from wam.runtime import policies

        return getattr(policies, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
