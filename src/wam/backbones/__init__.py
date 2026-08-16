"""Video/multimodal backbones behind one adapter interface (FR-09/AC-05, T-15).

FLUX 3 Dev preferred (stub until OD-06), Wan2.1-I2V open fallback (skeleton, OD-04),
Cosmos3-Edge for the edge sub-project (skeleton, weights not staged — E-01/E-02/E-03), and a
tiny fully-functional backbone for tests/overfit. ``TinyVideoBackbone``/``TinyBackboneConfig``
are exposed lazily so importing this package never requires torch.
"""

from __future__ import annotations

from typing import Any

from wam.backbones.cosmos3_edge import Cosmos3EdgeAdapter, Cosmos3EdgeConfig
from wam.backbones.flux3 import Flux3Adapter
from wam.backbones.registry import available_backbones, get_backbone
from wam.backbones.wan_i2v import WanI2VAdapter

__all__ = [
    "Cosmos3EdgeAdapter",
    "Cosmos3EdgeConfig",
    "Flux3Adapter",
    "TinyBackboneConfig",
    "TinyVideoBackbone",
    "WanI2VAdapter",
    "available_backbones",
    "get_backbone",
]

_LAZY_TORCH_EXPORTS = ("TinyVideoBackbone", "TinyBackboneConfig")


def __getattr__(name: str) -> Any:
    if name in _LAZY_TORCH_EXPORTS:
        from wam.backbones import tiny

        return getattr(tiny, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
