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
  chunks (no dropout, no autograd graph, no RNG). This survives the T-30 flow sampler
  because that sampler re-seeds a CPU generator per call instead of advancing a shared
  stream: a stream would make chunk N depend on how many predictions preceded it, and
  T-25's two bit-identical MuJoCo rollouts rest on exactly that not being the case.
- Traceability (AC-04): the checkpoint's embedded :class:`RunMetadata` (run_id,
  config_hash, checkpoint_ref, dataset_snapshot_ref) is exposed so every rollout can be
  tied to checkpoint + dataset snapshot + config hash.
"""

from __future__ import annotations

from pathlib import Path

import torch
from pydantic import BaseModel
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


def _relocate_backbone(backbone_config: BaseModel, *, source: str | Path | None, device: str):
    """Point a trained backbone config at THIS machine's weights and device.

    Where the frozen weights sit is machine-local. ``train_t16_lora`` keeps it off the committed
    YAML on purpose and folds ``--backbone-source`` in at run time, so ``config_hash`` matches
    across two machines training the identical model (AC-04). The consequence lands here: a
    checkpoint trained on the cluster carries *that cluster's* absolute path, so loading it
    anywhere else has to substitute a local one or ``build_backbone`` goes looking for weights
    that are not on this disk.

    ``device`` is overridden for the same reason it is at training time: the frozen tower is
    materialized by ``build_backbone(load=True)`` BEFORE the policy moves the model, so a
    checkpoint that recorded ``cuda`` would allocate tens of GB of VRAM on the way to a CPU
    policy — or just fail outright on a box with no GPU.

    ``config_hash`` is deliberately NOT recomputed. These two fields record where a run happened,
    not what was trained, which is precisely why they were excluded from the hash to begin with.
    """
    fields = type(backbone_config).model_fields
    updates: dict[str, object] = {}
    if source is not None:
        if "checkpoint_path" not in fields:
            raise ValueError(
                f"backbone kind {backbone_config.kind!r} holds no weights of its own, "
                "so a backbone source does not apply to it"
            )
        updates["checkpoint_path"] = str(source)
    if "device" in fields and backbone_config.device != device:
        updates["device"] = device
    if not updates:
        return backbone_config
    # model_validate rather than model_copy: a substituted path should face the same validation a
    # freshly parsed config does, instead of slipping in unchecked.
    return type(backbone_config).model_validate({**backbone_config.model_dump(), **updates})


def load_joint_policy(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    camera: str | None = None,
    backbone_source: str | Path | None = None,
    flow_steps: int | None = None,
    flow_seed: int = 0,
) -> JointCheckpointPolicy:
    """Load a joint checkpoint of EITHER kind, building the frozen base only when required.

    A self-contained checkpoint (tiny backbone) restores directly. A Wan-backed one does not:
    ``WanFlowBackbone`` keeps the DiT, VAE and text tower out of the module tree, so the file holds
    only ``backbone.lora.*`` and ``backbone.state_proj.*`` and the base has to arrive separately.

    The branch is taken from the embedded config's ``requires_external_weights``, not from a
    sidecar file or a guess at tensor names — the config is inside the checkpoint, so it travels
    with it and cannot go missing. Callers that skip this and construct
    :class:`JointCheckpointPolicy` directly work only for the self-contained case.

    ``backbone_source`` relocates the frozen weights for this machine (see
    :func:`_relocate_backbone`); without it the path recorded at training time is used as-is,
    which is right on the machine that trained and wrong everywhere else.

    Building the base is the expensive step (weights off disk, tens of GB for Wan), which is why
    it happens only on the branch that needs it.

    ``flow_steps``/``flow_seed`` select the T-30 flow readout (``None`` = the regression head, the
    historical default). They are wired HERE rather than in each caller because this one function
    is what ``scripts/eval_t16.py``, ``scripts/rollout.py`` and ``scripts/serve_policy.py`` all
    load through: an offline A/B that concluded in favour of the sampler would otherwise need a
    second, subtly different implementation before it could be closed-loop tested.
    """
    from wam.backbones.registry import build_backbone
    from wam.training._utils import load_checkpoint_raw
    from wam.training.joint import JointTrainingConfig

    _, config_dict, _ = load_checkpoint_raw(checkpoint_path)
    config = JointTrainingConfig.model_validate(config_dict)
    if not config.backbone.requires_external_weights:
        if backbone_source is not None:
            raise ValueError(
                f"backbone kind {config.backbone.kind!r} is self-contained: its weights are in "
                "the checkpoint, so there is no external source to point at"
            )
        return JointCheckpointPolicy(
            checkpoint_path,
            device=device,
            camera=camera,
            flow_steps=flow_steps,
            flow_seed=flow_seed,
        )
    backbone = build_backbone(
        _relocate_backbone(config.backbone, source=backbone_source, device=device), load=True
    )
    return JointCheckpointPolicy(
        checkpoint_path,
        device=device,
        backbone=backbone,
        strict=False,
        camera=camera,
        flow_steps=flow_steps,
        flow_seed=flow_seed,
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

    ``flow_steps`` swaps the action readout for the T-30 flow sampler
    (:meth:`~wam.training.joint.JointWorldActionModel.sample_action_chunk`); ``None``, the
    default, keeps the regression head every recorded run was produced with. A policy that
    decodes its chunks differently is a different policy, so the setting is exposed as
    :attr:`flow_steps`/:attr:`flow_seed` for the same reason ``camera`` is — a rollout has to be
    able to say which readout produced it.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        *,
        backbone: nn.Module | None = None,
        strict: bool = True,
        camera: str | None = None,
        flow_steps: int | None = None,
        flow_seed: int = 0,
    ) -> None:
        # Rejected at construction rather than at the first predict(): a Wan-backed policy has
        # already spent minutes building a multi-GB base by the time predict() is reached.
        if flow_steps is not None and flow_steps < 1:
            raise ValueError(f"flow_steps must be >= 1 or None, got {flow_steps}")
        model, metadata = load_joint_checkpoint(checkpoint_path, backbone=backbone, strict=strict)
        self._model: JointWorldActionModel = model.to(torch.device(device))
        self._model.eval()
        self._metadata = metadata
        self._checkpoint_path = Path(checkpoint_path)
        self._device = str(device)
        self._camera = camera
        self._flow_steps = flow_steps
        self._flow_seed = flow_seed

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

    @property
    def flow_steps(self) -> int | None:
        """Euler steps of the T-30 flow readout, or ``None`` for the regression head."""
        return self._flow_steps

    @property
    def flow_seed(self) -> int:
        """Seed of the flow readout's noise draw (ignored when :attr:`flow_steps` is ``None``)."""
        return self._flow_seed

    def predict(self, observation: Observation) -> ActionChunk:
        """Policy protocol: Observation -> ActionChunk, deterministic, gradient-free.

        Deterministic with the flow readout too: the seed is passed through unchanged on every
        call, so the sampler re-draws the same noise rather than walking a stream (see the module
        docstring's determinism contract).
        """
        with torch.no_grad():
            return self._model.predict(
                observation,
                camera=self._camera,
                flow_steps=self._flow_steps,
                flow_seed=self._flow_seed,
            )
