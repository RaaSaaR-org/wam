"""Backbone adapter registry: name -> ``BackboneAdapter`` factory (FR-09/AC-05, T-15).

Contract: ``get_backbone`` is the single construction entry point for runtime/config code;
adding a backbone means adding a factory here, never branching on names elsewhere. Factories
import their module lazily so listing/constructing torch-free skeletons never imports torch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from wam.interfaces.protocols import BackboneAdapter


def _make_tiny(**cfg: Any) -> BackboneAdapter:
    from wam.backbones.tiny import TinyBackboneConfig, TinyVideoBackbone

    config = cfg.pop("config", None)
    if config is not None:
        if cfg:
            raise TypeError(f"pass either config= or field kwargs, not both: {sorted(cfg)}")
        return TinyVideoBackbone(config)
    return TinyVideoBackbone(TinyBackboneConfig(**cfg) if cfg else None)


def _make_wan_i2v(**cfg: Any) -> BackboneAdapter:
    from wam.backbones.wan_i2v import WanI2VAdapter

    return WanI2VAdapter(**cfg)


def _make_flux3(**cfg: Any) -> BackboneAdapter:
    from wam.backbones.flux3 import Flux3Adapter

    return Flux3Adapter(**cfg)


_FACTORIES: dict[str, Callable[..., BackboneAdapter]] = {
    "tiny": _make_tiny,
    "wan_i2v": _make_wan_i2v,
    "flux3": _make_flux3,
}


def available_backbones() -> tuple[str, ...]:
    """Registered backbone names, sorted."""
    return tuple(sorted(_FACTORIES))


def get_backbone(name: str, **cfg: Any) -> BackboneAdapter:
    """Construct a backbone adapter by name ('tiny' | 'wan_i2v' | 'flux3'); ``cfg`` is passed
    to the factory. Raises ValueError for unknown names. Construction never downloads weights
    (wan_i2v/flux3 load lazily and raise until their integrations land)."""
    key = name.lower()
    factory = _FACTORIES.get(key)
    if factory is None:
        raise ValueError(f"unknown backbone {name!r}; available: {list(available_backbones())}")
    return factory(**cfg)
