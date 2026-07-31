#!/usr/bin/env python3
"""Export a WAM checkpoint's LoRA into the stock diffusers layout.

`WanFlowBackbone.save_adapter` already writes this layout, but it needs a *loaded* adapter —
19 GB of frozen base weights on a GPU — purely to hand `save_lora_adapter` a module tree to
walk. Everything that layout actually contains is already in the checkpoint. This script does
the same conversion as a pure key rename, so the file can be produced on a laptop from the
330 MB checkpoint alone:

    .venv/bin/python scripts/export_lora.py \
        --checkpoint runs/t16-lora-seed0/checkpoints/step-020000/model.safetensors \
        --out runs/t16-lora-seed0/lora-diffusers

The point of the export is that the result loads into a plain
``WanImageToVideoPipeline`` — so you can *watch* what the fine-tune did to the video prior
without any of WAM in the process (see ``docs/hf_jobs.md``, the Space's "generate future" tab).

Three transforms, all reversible:

1. ``backbone.lora.blocks__0__attn1__to_q__lora_A__wam__weight``
   -> ``blocks.0.attn1.to_q.lora_A.weight``. The ``__`` is ``WanFlowBackbone``'s mangling
   (``nn.ParameterDict`` keys may not contain "."), the dropped ``.wam`` is the adapter name,
   which is exactly what peft's ``get_peft_model_state_dict`` strips on save.
2. The peft ``LoraConfig`` is rebuilt from the config the checkpoint carries and written to the
   safetensors metadata under ``lora_adapter_metadata`` — the same key ``save_lora_adapter``
   uses. **This is not cosmetic.** Without it, ``load_lora_adapter`` infers the config from the
   weights alone, and inference sets ``lora_alpha = r``; we train at ``alpha=64, r=32``, so the
   adapter would silently load at half the strength it was trained with.
3. ``backbone.state_proj.*`` goes to its own file, mirroring ``save_adapter``.

What the exported adapter is NOT: the whole trained model. The action branch (head, encoders,
velocity head) stays behind in the checkpoint, and a diffusers pipeline has no state input, so
the state projection cannot be applied there either — the DiT was trained with state tokens
concatenated onto the text context (``wan_i2v.py:605``) and generates here without them. This
file answers "what did the fine-tune do to the video prior", not "how well does WAM act".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_LORA_PREFIX = "backbone.lora."
_STATE_PROJ_PREFIX = "backbone.state_proj."
_DOT_REPLACEMENT = "__"  # wam.backbones.wan_flow._DOT_REPLACEMENT
_LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"
_STATE_PROJ_FILE = "state_proj.safetensors"
_METADATA_KEY = "lora_adapter_metadata"  # diffusers.loaders.lora_base.LORA_ADAPTER_METADATA_KEY


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", type=Path, required=True, help="WAM model.safetensors")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument(
        "--adapter-name",
        default="wam",
        help="adapter name to strip from the keys (must match the trained one)",
    )
    return p.parse_args(argv)


def unmangle(key: str, *, adapter_name: str) -> str:
    """One checkpoint LoRA key -> its peft/diffusers name."""
    stripped = key[len(_LORA_PREFIX) :].replace(_DOT_REPLACEMENT, ".")
    marker = f".{adapter_name}."
    if marker not in stripped:
        raise ValueError(
            f"LoRA key {key!r} carries no {marker!r} segment — it was trained under a different "
            f"adapter name; pass the right --adapter-name"
        )
    # Anchored on the dotted segment rather than the bare name: a loose replace would also eat
    # the substring out of a module called e.g. "wamble".
    return stripped.replace(marker, ".", 1)


def lora_config_metadata(backbone: dict[str, Any]) -> dict[str, Any]:
    """The peft ``LoraConfig`` for this checkpoint, as ``save_lora_adapter`` would store it.

    Built through peft rather than hand-rolled so the dict carries every field the loader
    expects, at whatever peft version is installed — the fields we care about are r, lora_alpha,
    lora_dropout and target_modules, but the loader reads the whole thing.
    """
    from peft import LoraConfig

    targets: str | list[str] = [str(name) for name in backbone["lora_targets"]]
    if backbone.get("lora_blocks") is not None:
        raise NotImplementedError(
            "this checkpoint restricts LoRA to specific blocks, which add_lora() encodes as an "
            "anchored regex; reproduce that spec here before exporting"
        )
    config = LoraConfig(
        r=int(backbone["lora_rank"]),
        lora_alpha=int(backbone["lora_alpha"]),
        lora_dropout=float(backbone["lora_dropout"]),
        target_modules=targets,
    )
    as_dict = config.to_dict()
    # peft keeps target_modules as a set; safetensors metadata is JSON, and save_lora_adapter
    # applies the same flattening.
    return {k: sorted(v) if isinstance(v, set) else v for k, v in as_dict.items()}


def export(checkpoint: Path, out: Path, *, adapter_name: str = "wam") -> dict[str, Any]:
    """Write the diffusers LoRA directory; returns a summary of what went into it."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(str(checkpoint), framework="pt") as handle:
        meta = handle.metadata() or {}
        keys = list(handle.keys())
        lora = {
            unmangle(k, adapter_name=adapter_name): handle.get_tensor(k)
            for k in keys
            if k.startswith(_LORA_PREFIX)
        }
        state_proj = {
            k[len(_STATE_PROJ_PREFIX) :]: handle.get_tensor(k)
            for k in keys
            if k.startswith(_STATE_PROJ_PREFIX)
        }

    if not lora:
        raise ValueError(
            f"{checkpoint} holds no {_LORA_PREFIX}* tensors — it is not a LoRA-adapted "
            "checkpoint (a tiny-backbone run carries its whole backbone instead)"
        )
    if "wam_config_json" not in meta:
        raise ValueError(
            f"{checkpoint} carries no wam_config_json metadata; cannot recover the LoRA config"
        )

    config = json.loads(meta["wam_config_json"])
    backbone = config["backbone"]
    if backbone.get("kind") != "wan_i2v":
        raise ValueError(f"expected a wan_i2v backbone, got {backbone.get('kind')!r}")

    adapter_metadata = lora_config_metadata(backbone)
    out.mkdir(parents=True, exist_ok=True)
    weight_path = out / _LORA_WEIGHT_NAME
    save_file(
        {k: v.contiguous() for k, v in lora.items()},
        str(weight_path),
        # "format" is what tells safetensors' readers this is torch; save_lora_adapter sets it
        # too, and _fetch_state_dict rejects a file without it.
        metadata={
            "format": "pt",
            _METADATA_KEY: json.dumps(adapter_metadata, indent=2, sort_keys=True),
        },
    )
    if state_proj:
        save_file({k: v.contiguous() for k, v in state_proj.items()}, str(out / _STATE_PROJ_FILE))

    run_metadata = json.loads(meta.get("wam_run_metadata_json", "{}"))
    provenance = {
        # Copied verbatim, not recomputed: these are the three things AC-04 traceability needs,
        # and the export changes none of what they refer to.
        "run_id": run_metadata.get("run_id"),
        "config_hash": run_metadata.get("config_hash"),
        "dataset_snapshot_ref": run_metadata.get("dataset_snapshot_ref"),
        "git_commit": run_metadata.get("git_commit"),
        "source_checkpoint": str(checkpoint),
        "base_model": backbone.get("model_id"),
        "trained_geometry": {
            "num_frames": backbone.get("num_frames"),
            "image_hw": backbone.get("image_hw"),
        },
        "lora": {
            "rank": backbone["lora_rank"],
            "alpha": backbone["lora_alpha"],
            "targets": list(backbone["lora_targets"]),
            "tensors": len(lora),
        },
        "video_loss_weight": config.get("weights", {}).get("video"),
        "excluded": "action branch and state projection are not applicable to a diffusers pipeline",
    }
    (out / "wam_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )

    dtypes = sorted({str(v.dtype).removeprefix("torch.") for v in lora.values()})
    return {
        "weight_path": str(weight_path),
        "tensors": len(lora),
        "dtypes": dtypes,
        "bytes": weight_path.stat().st_size,
        "state_proj_tensors": len(state_proj),
        "adapter_metadata": adapter_metadata,
        "provenance": provenance,
        "torch": torch.__version__,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.checkpoint.is_file():
        print(f"no such checkpoint: {args.checkpoint}", file=sys.stderr)
        return 2
    summary = export(args.checkpoint, args.out, adapter_name=args.adapter_name)
    prov = summary["provenance"]
    print(f"wrote {summary['weight_path']}")
    print(
        f"  {summary['tensors']} LoRA tensors, {summary['dtypes']}, {summary['bytes'] / 1e6:.1f} MB"
    )
    print(
        f"  r={prov['lora']['rank']} alpha={prov['lora']['alpha']} targets={prov['lora']['targets']}"
    )
    print(
        f"  state_proj: {summary['state_proj_tensors']} tensors (not usable by a diffusers pipeline)"
    )
    print(
        f"  run_id={prov['run_id']} config_hash={prov['config_hash'][:12] if prov['config_hash'] else None}"
    )
    print(
        f"  trained at {prov['trained_geometry']['num_frames']} frames {prov['trained_geometry']['image_hw']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
