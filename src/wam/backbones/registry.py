"""Backbone adapter registry: name -> ``BackboneAdapter`` factory (FR-09/AC-05, T-15).

Contract: ``get_backbone`` is the single construction entry point for runtime/config code;
adding a backbone means adding a factory here, never branching on names elsewhere. Factories
import their module lazily so listing/constructing torch-free skeletons never imports torch.

Two entry points, two audiences:

- ``get_backbone(name, **cfg)`` — the AC-05 conformance path: build every registered adapter
  from a bare name, no weights, no torch.
- ``build_backbone_config`` / ``build_backbone`` — the TRAINING path: a validated, tagged
  config in, a ready ``nn.Module`` out. These deliberately dispatch on ``config.kind`` rather
  than a name string, so a training config carries its own fully-typed backbone section
  (FR-09: nothing outside this module and ``wan_flow.py`` may know which backbone is in use).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from wam.interfaces.protocols import BackboneAdapter

if TYPE_CHECKING:  # torch-domain imports: annotations only, never at runtime
    from torch import nn

    from wam.backbones.tiny import TinyBackboneConfig
    from wam.backbones.wan_i2v import WanBackboneConfig


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


def _make_cosmos3_edge(**cfg: Any) -> BackboneAdapter:
    from wam.backbones.cosmos3_edge import Cosmos3EdgeAdapter, Cosmos3EdgeConfig

    config = cfg.pop("config", None)
    if config is not None:
        if cfg:
            raise TypeError(f"pass either config= or field kwargs, not both: {sorted(cfg)}")
        return Cosmos3EdgeAdapter(config)
    return Cosmos3EdgeAdapter(Cosmos3EdgeConfig(**cfg) if cfg else None)


_FACTORIES: dict[str, Callable[..., BackboneAdapter]] = {
    "tiny": _make_tiny,
    "wan_i2v": _make_wan_i2v,
    "flux3": _make_flux3,
    "cosmos3_edge": _make_cosmos3_edge,
}


def available_backbones() -> tuple[str, ...]:
    """Registered backbone names, sorted."""
    return tuple(sorted(_FACTORIES))


def get_backbone(name: str, **cfg: Any) -> BackboneAdapter:
    """Construct a backbone adapter by name ('tiny' | 'wan_i2v' | 'flux3' | 'cosmos3_edge');
    ``cfg`` is passed to the factory. Raises ValueError for unknown names. Construction never
    downloads weights (wan_i2v/flux3/cosmos3_edge load lazily and raise until their
    integrations land)."""
    key = name.lower()
    factory = _FACTORIES.get(key)
    if factory is None:
        raise ValueError(f"unknown backbone {name!r}; available: {list(available_backbones())}")
    return factory(**cfg)


# -- training path: tagged config -> module ------------------------------------------------


def build_backbone_config(data: Any) -> TinyBackboneConfig | WanBackboneConfig:
    """Validate a backbone config section into the right union member.

    An untagged mapping is tagged ``kind='tiny'`` first. That back-compat step is load-bearing
    in two places that must keep working unchanged: every shipped YAML under ``configs/``
    (written before the union existed) and every config dict embedded in a pre-existing
    checkpoint. Tagging beats a non-discriminated union here — pydantic would otherwise try
    ``TinyBackboneConfig`` on a Wan section and, before ``extra='forbid'``, quietly hand back
    an all-defaults 64-dim tiny config.
    """
    from pydantic import TypeAdapter

    # The union lives in the torch layer (both members are defined next to their adapters), so
    # it is imported here rather than at module scope: importing wam.backbones stays torch-free.
    from wam.training.joint import BackboneConfig

    if isinstance(data, Mapping) and "kind" not in data:
        data = {**data, "kind": "tiny"}
    return TypeAdapter(BackboneConfig).validate_python(data)


def build_backbone(
    config: TinyBackboneConfig | WanBackboneConfig, *, load: bool = False
) -> nn.Module:
    """Construct the ``FlowBackbone`` module for a tagged backbone config.

    ``load=False`` (the default) builds the module skeleton only — no weights, no downloads —
    which is what tests, config validation and checkpoint restore need. ``load=True`` pulls the
    real weights and is a heavyweight, network- and disk-touching operation.
    """
    kind = config.kind
    if kind == "tiny":
        from wam.backbones.tiny import TinyVideoBackbone

        return TinyVideoBackbone(config)  # type: ignore[arg-type]
    if kind == "wan_i2v":
        try:
            from wam.backbones.wan_flow import WanFlowBackbone
        except ImportError as exc:  # never silently fall back to tiny — that trains junk
            raise ImportError(
                "backbone kind 'wan_i2v' requires wam.backbones.wan_flow and its optional "
                f"dependencies (diffusers, peft): {exc}"
            ) from exc

        backbone = WanFlowBackbone(config)  # type: ignore[arg-type]
        if load:
            backbone.load()
        return backbone
    raise ValueError(f"unknown backbone kind {kind!r}")
