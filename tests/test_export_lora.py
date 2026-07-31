"""Tests for scripts/export_lora.py (WAM checkpoint -> stock diffusers LoRA directory)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export_lora = _load("export_lora")

_BACKBONE = {
    "kind": "wan_i2v",
    "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "num_frames": 9,
    "image_hw": [128, 160],
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "lora_targets": ["to_q", "to_k", "to_v", "to_out.0", "net.0.proj", "net.2"],
    "lora_blocks": None,
}


def _write_checkpoint(path: Path, *, backbone: dict | None = None, adapter: str = "wam") -> Path:
    """A checkpoint shaped like a real T-16 one: two adapted modules plus an action branch."""
    section = {**_BACKBONE, **(backbone or {})}
    tensors: dict[str, torch.Tensor] = {}
    for module in ("blocks__0__attn1__to_q", "blocks__7__ffn__net__0__proj"):
        tensors[f"backbone.lora.{module}__lora_A__{adapter}__weight"] = torch.randn(32, 3072)
        tensors[f"backbone.lora.{module}__lora_B__{adapter}__weight"] = torch.randn(3072, 32)
    tensors["backbone.state_proj.weight"] = torch.randn(4096, 32)
    tensors["backbone.state_proj.bias"] = torch.randn(4096)
    tensors["action_head.target_head.weight"] = torch.randn(240, 256)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(path),
        metadata={
            "wam_config_json": json.dumps({"backbone": section, "weights": {"video": 0.5}}),
            "wam_run_metadata_json": json.dumps(
                {
                    "run_id": "t16-lora-seed0",
                    "config_hash": "45ee9e60",
                    "dataset_snapshot_ref": "sha256:598f193f",
                    "git_commit": "78fc56d",
                }
            ),
        },
    )
    return path


def test_unmangle_recovers_the_peft_parameter_name() -> None:
    """The two transforms the ParameterDict alias forced: "__" back to ".", adapter name gone."""
    assert (
        export_lora.unmangle(
            "backbone.lora.blocks__0__attn1__to_out__0__lora_A__wam__weight", adapter_name="wam"
        )
        == "blocks.0.attn1.to_out.0.lora_A.weight"
    )


def test_unmangle_rejects_a_key_from_a_differently_named_adapter() -> None:
    """Silently keeping the wrong segment would produce keys no loader matches, and a LoRA that
    loads zero tensors reads as "the fine-tune did nothing" rather than as a bad export."""
    with pytest.raises(ValueError, match="different adapter name"):
        export_lora.unmangle(
            "backbone.lora.blocks__0__attn1__to_q__lora_A__other__weight", adapter_name="wam"
        )


def test_export_writes_the_diffusers_layout_with_unprefixed_keys(tmp_path: Path) -> None:
    summary = export_lora.export(
        _write_checkpoint(tmp_path / "model.safetensors"), tmp_path / "out"
    )
    weight_path = tmp_path / "out" / "pytorch_lora_weights.safetensors"
    assert weight_path.is_file()
    with safe_open(str(weight_path), framework="pt") as handle:
        keys = set(handle.keys())
    assert keys == {
        "blocks.0.attn1.to_q.lora_A.weight",
        "blocks.0.attn1.to_q.lora_B.weight",
        "blocks.7.ffn.net.0.proj.lora_A.weight",
        "blocks.7.ffn.net.0.proj.lora_B.weight",
    }
    # The action branch is not part of a video-prior adapter, and no "transformer." prefix:
    # load_lora_adapter(prefix=None) matches model-relative names.
    assert not any(k.startswith(("action_", "transformer.", "backbone.")) for k in keys)
    assert summary["tensors"] == 4
    assert (tmp_path / "out" / "state_proj.safetensors").is_file()


def test_export_stores_the_alpha_so_the_adapter_loads_at_trained_strength(tmp_path: Path) -> None:
    """The whole reason the metadata is written: diffusers infers ``alpha = r`` without it, and
    we train at alpha=64 / r=32, so the adapter would load at half the strength it learned."""
    export_lora.export(_write_checkpoint(tmp_path / "model.safetensors"), tmp_path / "out")
    with safe_open(
        str(tmp_path / "out" / "pytorch_lora_weights.safetensors"), framework="pt"
    ) as handle:
        metadata = handle.metadata()
    assert metadata["format"] == "pt"  # _fetch_state_dict rejects a file without it
    saved = json.loads(metadata["lora_adapter_metadata"])
    assert (saved["r"], saved["lora_alpha"]) == (32, 64)
    assert sorted(saved["target_modules"]) == sorted(_BACKBONE["lora_targets"])


def test_export_carries_the_provenance_of_the_run_it_came_from(tmp_path: Path) -> None:
    """A clip has to be attributable to a checkpoint (AC-04), and the fields are copied, not
    recomputed — the export changes nothing the hashes refer to."""
    export_lora.export(_write_checkpoint(tmp_path / "model.safetensors"), tmp_path / "out")
    prov = json.loads((tmp_path / "out" / "wam_provenance.json").read_text())
    assert prov["run_id"] == "t16-lora-seed0"
    assert prov["config_hash"] == "45ee9e60"
    assert prov["dataset_snapshot_ref"] == "sha256:598f193f"
    assert prov["lora"] == {
        "rank": 32,
        "alpha": 64,
        "targets": _BACKBONE["lora_targets"],
        "tensors": 4,
    }
    # Recorded because generating at another geometry is out of the adapter's distribution.
    assert prov["trained_geometry"] == {"num_frames": 9, "image_hw": [128, 160]}


def test_export_refuses_a_checkpoint_that_carries_its_own_backbone(tmp_path: Path) -> None:
    """A tiny-backbone run has no adapter to export; exporting an empty file would be worse
    than failing, because it looks like a fine-tune that changed nothing."""
    path = tmp_path / "tiny.safetensors"
    save_file(
        {"backbone.blocks.0.weight": torch.randn(4, 4)},
        str(path),
        metadata={"wam_config_json": json.dumps({"backbone": {"kind": "tiny"}})},
    )
    with pytest.raises(ValueError, match="holds no backbone.lora"):
        export_lora.export(path, tmp_path / "out")


def test_export_refuses_a_block_restricted_adapter_rather_than_guess(tmp_path: Path) -> None:
    """``lora_blocks`` makes add_lora() build an anchored regex; emitting the plain suffix list
    instead would attach the adapter to blocks it was never trained on."""
    path = _write_checkpoint(tmp_path / "model.safetensors", backbone={"lora_blocks": [2, 10]})
    with pytest.raises(NotImplementedError, match="restricts LoRA to specific blocks"):
        export_lora.export(path, tmp_path / "out")
