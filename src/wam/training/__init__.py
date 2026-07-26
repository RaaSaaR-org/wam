"""Losses, datasets, trainers and monitoring (T-13/T-16/T-17, PRD §10).

Torch domain: everything here may import torch. Losses live in ``wam.training.losses``,
the action-only baseline in ``wam.training.action_only`` (M2 overfit gate), joint
video/action flow-matching in ``wam.training.joint`` (M3) and divergence monitoring in
``wam.training.monitor`` (R-07).
"""

from .action_only import (
    ActionLossWeights,
    ActionOnlyConfig,
    ActionOnlyModel,
    ActionOnlyTrainer,
    load_action_only_checkpoint,
)
from .datasets import EpisodeDataset, collate_episode_batch
from .joint import (
    ActionVelocityHead,
    JointLossWeights,
    JointTrainer,
    JointTrainingConfig,
    JointWorldActionModel,
    load_joint_checkpoint,
)
from .losses import (
    action_flow_matching_loss,
    action_regression_loss,
    alignment_loss,
    limit_penalty,
    make_flow_targets,
    smoothness_loss,
    video_flow_loss,
)
from .monitor import TrainingDiverged, TrainingMonitor, TrainingMonitorConfig

__all__ = [
    "ActionLossWeights",
    "ActionOnlyConfig",
    "ActionOnlyModel",
    "ActionOnlyTrainer",
    "ActionVelocityHead",
    "EpisodeDataset",
    "JointLossWeights",
    "JointTrainer",
    "JointTrainingConfig",
    "JointWorldActionModel",
    "TrainingDiverged",
    "TrainingMonitor",
    "TrainingMonitorConfig",
    "action_flow_matching_loss",
    "action_regression_loss",
    "alignment_loss",
    "collate_episode_batch",
    "limit_penalty",
    "load_action_only_checkpoint",
    "load_joint_checkpoint",
    "make_flow_targets",
    "smoothness_loss",
    "video_flow_loss",
]
