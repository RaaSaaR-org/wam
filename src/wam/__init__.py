"""WAM — modular World Action Model: joint video/action prediction with deterministic safety.

Top-level convenience re-exports only; the authoritative contracts live in
``wam.interfaces`` (schema + protocols), ``wam.safety`` and ``wam.robot``.
"""

from wam.interfaces import ActionChunk, RobotState
from wam.robot import get_robot
from wam.safety import SafetyLayer

__version__ = "0.0.1"

__all__ = [
    "ActionChunk",
    "RobotState",
    "SafetyLayer",
    "__version__",
    "get_robot",
]
