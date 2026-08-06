"""Robot adapter registry: name -> ``RobotAdapter`` factory (FR-06).

Contract: ``get_robot`` is the single construction entry point for runtime/config code;
adding a robot means adding a factory here, never branching on names elsewhere.

Two tiers, one entry point:

- ``_FACTORIES`` — always importable, always constructible. ``available_robots()`` returns
  exactly these: the tuple means "names this environment can construct RIGHT NOW", which is
  what callers (and the test suite) enumerate and instantiate.
- ``_LAZY_FACTORIES`` — factories whose MODULE needs an OPTIONAL dependency (e.g. 'mujoco_g1'
  needs ``mujoco``, extra ``wam[sim]``). They are registered by dotted path and imported on
  FIRST USE, so importing this module — and constructing every other robot — never requires
  that dependency. They are listed by ``optional_robots()``, NOT by ``available_robots()``,
  because their constructibility depends on the install; ``get_robot`` accepts them either
  way and lets the factory raise its own error naming the missing package.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from wam.interfaces.protocols import RobotAdapter
from wam.robot.g1 import G1Adapter
from wam.robot.mock import MockRobot

_FACTORIES: dict[str, Callable[..., RobotAdapter]] = {
    "mock": MockRobot,
    "g1": G1Adapter,
}

# name -> (module, attribute), imported on the first get_robot() call. "mujoco_g1" pulls in
# mujoco (extra: wam[sim]) plus the fetched scene assets (scripts/fetch_g1_model.py); its
# factory raises RuntimeError naming the install fix when the package is absent.
#
# "isaac_g1" is lazy for a different reason: wam.robot.isaac_g1 imports fine anywhere (it is
# deliberately torch-free and isaacsim-free at module scope), but it is only CONSTRUCTIBLE
# inside the Isaac Sim python, which cannot contain this repo's torch — isaacsim-core 6.0.1
# pins torch 2.11.0 and uv.lock resolves 2.13.0. So it belongs with the optional robots, and
# IsaacG1Robot's constructor raises RuntimeError naming that two-venv split. Registering it
# here rather than in _FACTORIES also keeps `get_robot("mock")` from importing the Isaac
# stack on machines that will never run it.
_LAZY_FACTORIES: dict[str, tuple[str, str]] = {
    "mujoco_g1": ("wam.robot.mujoco_g1", "MujocoG1Robot"),
    "isaac_g1": ("wam.robot.isaac_g1", "IsaacG1Robot"),
}


def available_robots() -> tuple[str, ...]:
    """Robot names constructible in ANY install, sorted. Robots behind an optional
    dependency are listed separately by :func:`optional_robots`."""
    return tuple(sorted(_FACTORIES))


def optional_robots() -> tuple[str, ...]:
    """Robot names registered but gated behind an OPTIONAL dependency, sorted.

    ``get_robot`` constructs these like any other name; without the dependency installed the
    factory raises with the installation hint. Kept out of :func:`available_robots` on
    purpose — that tuple is enumerated and instantiated by callers, and these names are only
    constructible in some installs.
    """
    return tuple(sorted(_LAZY_FACTORIES))


# Resolved lazy factories. Deliberately NOT merged into _FACTORIES: that would make
# available_robots() depend on whether someone already constructed the optional robot.
_LAZY_CACHE: dict[str, Callable[..., RobotAdapter]] = {}


def _resolve(key: str) -> Callable[..., RobotAdapter] | None:
    factory = _FACTORIES.get(key) or _LAZY_CACHE.get(key)
    if factory is None and key in _LAZY_FACTORIES:
        module_name, attr = _LAZY_FACTORIES[key]
        factory = getattr(import_module(module_name), attr)
        _LAZY_CACHE[key] = factory  # import once per process
    return factory


def get_robot(name: str, **cfg: Any) -> RobotAdapter:
    """Construct a robot adapter by name ('mock' | 'g1' | 'mujoco_g1'); ``cfg`` is passed to
    the factory. Raises ValueError for unknown names. Construction never touches hardware
    (G1 connects lazily via ``connect()``; 'mujoco_g1' is simulation only)."""
    key = name.lower()
    factory = _resolve(key)
    if factory is None:
        raise ValueError(
            f"unknown robot {name!r}; available: {list(available_robots())}, "
            f"optional: {list(optional_robots())}"
        )
    return factory(**cfg)
