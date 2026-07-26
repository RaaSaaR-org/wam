"""Runtime policies for the closed-loop executor (T-19).

Contracts:
- :class:`CheckpointPolicy` is the ONLY torch user in ``wam.runtime``: it loads a trained
  ``ActionOnlyModel`` checkpoint and serves the ``Policy`` protocol. The executor itself
  stays torch-free.
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

from wam.interfaces import ActionChunk, Observation, RunMetadata

# Re-exported for convenience: the deterministic M0 sinusoid policy.
from wam.runtime.mock_loop import DummyPolicy
from wam.training import ActionOnlyModel, load_action_only_checkpoint

__all__ = ["CheckpointPolicy", "DummyPolicy"]


class CheckpointPolicy:
    """A trained ``ActionOnlyModel`` checkpoint served as a runtime ``Policy``.

    Loads via ``wam.training.load_action_only_checkpoint`` (safetensors with embedded
    config JSON + RunMetadata), moves the model to ``device`` and keeps it in eval mode.
    ``predict`` maps one :class:`Observation` to one canonical :class:`ActionChunk`.
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        model, metadata = load_action_only_checkpoint(checkpoint_path)
        self._model: ActionOnlyModel = model.to(torch.device(device))
        self._model.eval()
        self._metadata = metadata
        self._checkpoint_path = Path(checkpoint_path)
        self._device = str(device)

    @property
    def model(self) -> ActionOnlyModel:
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

    def predict(self, observation: Observation) -> ActionChunk:
        """Policy protocol: Observation -> ActionChunk, deterministic, gradient-free."""
        with torch.no_grad():
            return self._model.predict(observation)
