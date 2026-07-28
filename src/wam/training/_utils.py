"""Shared internals for the trainers (T-13/T-16). Not part of the public API."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from wam.interfaces.versioning import RunMetadata

from .datasets import collate_episode_batch

_BATCH_TENSOR_KEYS = ("frames", "q", "dq", "imu", "gripper", "targets", "gripper_target")

CHECKPOINT_CONFIG_KEY = "wam_config_json"
CHECKPOINT_METADATA_KEY = "wam_run_metadata_json"


def encode_instructions(backbone: Any, instruction: str | Sequence[str], batch: int) -> Tensor:
    """Instruction(s) -> text context tokens for ``backbone.features``.

    ``backbone`` is annotated ``Any``, not ``nn.Module``: adapters only have to satisfy the
    structural ``BackboneAdapter``/``FlowBackbone`` protocols, and the ones wrapping a
    third-party pipeline are plain classes holding a module, not modules themselves.

    A single string (or a list with one unique value) is encoded once as ``[1, T, D]`` and
    broadcast by the backbone. Mixed per-sample instructions are encoded individually and
    right-padded with the empty-text padding token embedding to a common length.
    """
    if isinstance(instruction, str):
        return backbone.condition_text(instruction)
    texts = list(instruction)
    if len(texts) != batch:
        raise ValueError(f"got {len(texts)} instructions for batch size {batch}")
    if len(set(texts)) == 1:
        return backbone.condition_text(texts[0])
    ctxs = [backbone.condition_text(text) for text in texts]
    max_len = max(ctx.shape[1] for ctx in ctxs)
    # ONE token wide, whatever the backbone's empty-text context length is: a real text tower
    # pads "" out to its full max_text_tokens, so using the whole thing would expand to
    # (max_len - len) * max_text_tokens columns and silently break the concat width.
    pad = backbone.condition_text("")[:, :1]  # [1, 1, D] deterministic padding token
    padded = [
        torch.cat([ctx, pad.expand(1, max_len - ctx.shape[1], -1)], dim=1)
        if ctx.shape[1] < max_len
        else ctx
        for ctx in ctxs
    ]
    return torch.cat(padded, dim=0)


def prepare_tensor_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Validate + move a training batch to ``device`` (keeps ``instruction`` as-is)."""
    out: dict[str, Any] = {}
    for key in _BATCH_TENSOR_KEYS:
        if key not in batch:
            raise KeyError(f"batch missing required key {key!r}")
        out[key] = torch.as_tensor(batch[key]).to(device)
    if "validity" in batch and batch["validity"] is not None:
        out["validity"] = torch.as_tensor(batch["validity"]).to(device)
    out["instruction"] = batch.get("instruction", "")
    return out


def iterate_batches(
    data: Dataset | Mapping[str, Any],
    *,
    steps: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Iterator[dict[str, Any]]:
    """Yield exactly ``steps`` training batches from a dataset or a full tensor batch.

    A ``Mapping`` is treated as one fixed full batch (overfit mode, D1 gate); a torch
    ``Dataset`` is cycled through a seeded shuffling ``DataLoader`` (deterministic order).
    """
    if isinstance(data, Mapping):
        fixed = prepare_tensor_batch(data, device)
        for _ in range(steps):
            yield fixed
        return
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        data,
        batch_size=min(batch_size, len(data)),  # type: ignore[arg-type]
        shuffle=True,
        generator=generator,
        collate_fn=collate_episode_batch,
        num_workers=0,
        drop_last=False,
    )
    produced = 0
    while produced < steps:
        for batch in loader:
            if produced >= steps:
                return
            yield prepare_tensor_batch(batch, device)
            produced += 1


def save_checkpoint(
    model: nn.Module,
    config: Any,
    path: str | Path,
    metadata: RunMetadata,
    *,
    state_dict: Mapping[str, Tensor] | None = None,
) -> Path:
    """Serialize ``model.state_dict()`` to safetensors with config + RunMetadata metadata.

    ``state_dict`` overrides what is written. That is how an adapted large backbone stays
    checkpointable: ``JointWorldActionModel.trainable_state_dict()`` holds the LoRA/adapter
    tensors only, so a run does not copy the frozen multi-GB base weights into every
    checkpoint. Restoring such a file needs ``strict=False`` plus the same base weights.
    """
    from safetensors.torch import save_file

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    source = model.state_dict() if state_dict is None else state_dict
    state = {name: tensor.contiguous() for name, tensor in source.items()}
    save_file(
        state,
        str(target),
        metadata={
            CHECKPOINT_CONFIG_KEY: config.model_dump_json(),
            CHECKPOINT_METADATA_KEY: json.dumps(metadata.to_dict(), sort_keys=True),
        },
    )
    return target


def load_checkpoint_raw(path: str | Path) -> tuple[dict[str, Tensor], dict[str, Any], RunMetadata]:
    """Read a safetensors checkpoint -> ``(state_dict, config_dict, RunMetadata)``."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    source = Path(path)
    with safe_open(str(source), framework="pt") as f:
        meta = f.metadata() or {}
    if CHECKPOINT_CONFIG_KEY not in meta or CHECKPOINT_METADATA_KEY not in meta:
        raise ValueError(f"{source}: not a WAM checkpoint (missing embedded config/metadata)")
    config_dict = json.loads(meta[CHECKPOINT_CONFIG_KEY])
    run_metadata = RunMetadata.model_validate(json.loads(meta[CHECKPOINT_METADATA_KEY]))
    return load_file(str(source)), config_dict, run_metadata
