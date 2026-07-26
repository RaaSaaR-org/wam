"""Deterministic safety layer: limits, projection, NaN/Inf rejection, watchdog (FR-07).

No learned components live in this package, and no learned component may bypass it.
Torch-free by design; numpy only.
"""

from wam.safety.config import SAFETY_CONFIG_VERSION, SafetyConfig
from wam.safety.layer import FkCallable, SafetyLayer
from wam.safety.watchdog import Watchdog, WatchdogAction

__all__ = [
    "SAFETY_CONFIG_VERSION",
    "FkCallable",
    "SafetyConfig",
    "SafetyLayer",
    "Watchdog",
    "WatchdogAction",
]
