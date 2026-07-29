"""Runtime policies for the closed-loop executor (T-19).

Contracts:
- This module is the ONLY torch user in ``wam.runtime``: it loads a trained checkpoint and
  serves the ``Policy`` protocol. The executor itself stays torch-free.
- Two checkpoint kinds, two classes, because the two models are different artifacts:
  :class:`CheckpointPolicy` serves the action-only baseline (``ActionOnlyModel``, T-13) and
  :class:`JointCheckpointPolicy` serves the world-action model (``JointWorldActionModel``,
  T-16). They are NOT interchangeable — a joint checkpoint carries an action encoder and a
  velocity head the action-only loader has no slots for, so pointing the wrong class at a
  file fails loudly at load rather than quietly mispredicting.
- Deterministic inference: the model is restored bit-exact in eval mode and every
  prediction runs under ``torch.no_grad()`` — identical observations yield identical
  chunks (no dropout, no autograd graph, no RNG).
- Traceability (AC-04): the checkpoint's embedded :class:`RunMetadata` (run_id,
  config_hash, checkpoint_ref, dataset_snapshot_ref) is exposed so every rollout can be
  tied to checkpoint + dataset snapshot + config hash.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from wam.interfaces import ActionChunk, Observation, RunMetadata

# Re-exported for convenience: the deterministic M0 sinusoid policy.
from wam.runtime.mock_loop import DummyPolicy
from wam.training import (
    ActionOnlyModel,
    JointWorldActionModel,
    load_action_only_checkpoint,
    load_joint_checkpoint,
)

__all__ = ["CheckpointPolicy", "DummyPolicy", "JointCheckpointPolicy", "load_joint_policy"]


def load_joint_policy(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    camera: str | None = None,
) -> JointCheckpointPolicy:
    """Load a joint checkpoint of EITHER kind, building the frozen base only when required.

    A self-contained checkpoint (tiny backbone) restores directly. A Wan-backed one does not:
    ``WanFlowBackbone`` keeps the DiT, VAE and text tower out of the module tree, so the file holds
    only ``backbone.lora.*`` and ``backbone.state_proj.*`` and the base has to arrive separately.

    The branch is taken from the embedded config's ``requires_external_weights``, not from a
    sidecar file or a guess at tensor names — the config is inside the checkpoint, so it travels
    with it and cannot go missing. Callers that skip this and construct
    :class:`JointCheckpointPolicy` directly work only for the self-contained case.

    Building the base is the expensive step (weights off disk, tens of GB for Wan), which is why
    it happens only on the branch that needs it.
    """
    from wam.backbones.registry import build_backbone
    from wam.training._utils import load_checkpoint_raw
    from wam.training.joint import JointTrainingConfig

    _, config_dict, _ = load_checkpoint_raw(checkpoint_path)
    config = JointTrainingConfig.model_validate(config_dict)
    if not config.backbone.requires_external_weights:
        return JointCheckpointPolicy(checkpoint_path, device=device, camera=camera)
    backbone = build_backbone(config.backbone, load=True)
    return JointCheckpointPolicy(
        checkpoint_path, device=device, backbone=backbone, strict=False, camera=camera
    )


class CheckpointPolicy:
    """A trained ``ActionOnlyModel`` checkpoint served as a runtime ``Policy``.

    Loads via ``wam.training.load_action_only_checkpoint`` (safetensors with embedded
    config JSON + RunMetadata), moves the model to ``device`` and keeps it in eval mode.
    ``predict`` maps one :class:`Observation` to one canonical :class:`ActionChunk`.
    """

    def __init__(
        self, checkpoint_path: str | Path, device: str = "cpu", *, camera: str | None = None
    ) -> None:
        model, metadata = load_action_only_checkpoint(checkpoint_path)
        self._model: ActionOnlyModel = model.to(torch.device(device))
        self._model.eval()
        self._metadata = metadata
        self._checkpoint_path = Path(checkpoint_path)
        self._device = str(device)
        self._camera = camera

    @property
    def model(self) -> ActionOnlyModel:
        return self._model

    @property
    def camera(self) -> str:
        """The ``Observation.images`` key actually read (the override, or the trained one)."""
        return self._camera if self._camera is not None else self._model.config.camera

    @property
    def metadata(self) -> RunMetadata:
        """Checkpoint provenance: run_id, config_hash, checkpoint/dataset refs (AC-04)."""
        return self._metadata

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def device(self) -> str:
        return self._device

    def predict(self, observation: Observation) -> ActionChunk:
        """Policy protocol: Observation -> ActionChunk, deterministic, gradient-free."""
        with torch.no_grad():
            return self._model.predict(observation, camera=self._camera)


class JointCheckpointPolicy:
    """A trained ``JointWorldActionModel`` (world-action) checkpoint served as a ``Policy``.

    The world-action counterpart of :class:`CheckpointPolicy`. ``predict`` runs the model's
    representation-only readout — one backbone pass at the clean flow timestep, ``ActionHead``
    on the shared features, video velocity discarded — so a rollout costs one forward pass per
    control cycle and no test-time video generation.

    ``backbone`` + ``strict=False`` are the T-16 path: an adapters-only checkpoint
    (``model.trainable_state_dict()``) carries LoRA tensors and heads but not the frozen
    multi-GB base, so the base arrives as an already-loaded backbone instance and the file
    fills in the rest. With no ``backbone`` the embedded config rebuilds one from scratch,
    which is what a self-contained tiny-backbone checkpoint wants.

    ``camera`` overrides which ``Observation.images`` key is read. Deployments name views
    differently from datasets (the MuJoCo scene renders ``head``/``wrist_left``, the converted
    GR00T episodes trained on ``ego``), and the alternative — silently falling back to some
    other camera — would put a policy on a robot looking through the wrong lens. The override
    is explicit, exposed as :attr:`camera`, and folded into the rollout's ``config_hash``.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        *,
        backbone: nn.Module | None = None,
        strict: bool = True,
        camera: str | None = None,
    ) -> None:
        model, metadata = load_joint_checkpoint(checkpoint_path, backbone=backbone, strict=strict)
        self._model: JointWorldActionModel = model.to(torch.device(device))
        self._model.eval()
        self._metadata = metadata
        self._checkpoint_path = Path(checkpoint_path)
        self._device = str(device)
        self._camera = camera

    @property
    def model(self) -> JointWorldActionModel:
        return self._model

    @property
    def metadata(self) -> RunMetadata:
        """Checkpoint provenance: run_id, config_hash, checkpoint/dataset refs (AC-04)."""
        return self._metadata

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def device(self) -> str:
        return self._device

    @property
    def camera(self) -> str:
        """The ``Observation.images`` key actually read (the override, or the trained one)."""
        return self._camera if self._camera is not None else self._model.config.camera

    def predict(self, observation: Observation) -> ActionChunk:
        """Policy protocol: Observation -> ActionChunk, deterministic, gradient-free."""
        with torch.no_grad():
            return self._model.predict(observation, camera=self._camera)
