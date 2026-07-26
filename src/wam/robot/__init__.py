"""Hardware abstraction: canonical schema <-> robot API (Unitree G1, mock).

Robot-specific joint mapping, units, calibration and limits live ONLY here (FR-06).
"""

from wam.robot.g1 import G1_JOINT_MAP, G1_NUM_MOTORS, G1_SPEC, G1Adapter, G1Config
from wam.robot.mock import MockRobot
from wam.robot.registry import available_robots, get_robot

__all__ = [
    "G1_JOINT_MAP",
    "G1_NUM_MOTORS",
    "G1_SPEC",
    "G1Adapter",
    "G1Config",
    "MockRobot",
    "available_robots",
    "get_robot",
]
