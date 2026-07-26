"""Action decoder: backbone features -> normalized action chunks (joint deltas / EE deltas + gripper)."""

from wam.decoders.action_head import ActionHead, ActionHeadConfig

__all__ = ["ActionHead", "ActionHeadConfig"]
