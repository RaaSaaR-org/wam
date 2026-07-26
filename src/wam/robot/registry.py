"""Robot adapter registry: name -> ``RobotAdapter`` factory (FR-06).

Contract: ``get_robot`` is the single construction entry point for runtime/config code;
adding a robot means adding a factory here, never branching on names elsewhere.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from wam.interfaces.protocols import RobotAdapter
from wam.robot.g1 import G1Adapter
from wam.robot.mock import MockRobot

_FACTORIES: dict[str, Callable[..., RobotAdapter]] = {
    "mock": MockRobot,
    "g1": G1Adapter,
}


def available_robots() -> tuple[str, ...]:
    """Registered robot names, sorted."""
    return tuple(sorted(_FACTORIES))


def get_robot(name: str, **cfg: Any) -> RobotAdapter:
    """Construct a robot adapter by name ('mock' | 'g1'); ``cfg`` is passed to the factory.
    Raises ValueError for unknown names. Construction never touches hardware (G1 connects
    lazily via ``connect()``)."""
    key = name.lower()
    factory = _FACTORIES.get(key)
    if factory is None:
        raise ValueError(f"unknown robot {name!r}; available: {list(available_robots())}")
    return factory(**cfg)
