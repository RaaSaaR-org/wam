"""WanFlowBackbone: the trainable module boundary around WanI2VAdapter (T-16, PRD §10.3).

Everything here runs on CPU with NO Wan weights: a REAL (190k-param) ``WanTransformer3DModel``
at toy dimensions plus a VAE/umT5 fake attached through ``WanI2VAdapter.attach``. The DiT has
to be real — the point of this class is where peft may inject, what ``.to()`` does to those
parameters and whether the readout hooks stay graph-connected, and a stub would happily agree
with a wrong implementation.

The load-bearing test is :meth:`TestModuleBoundary.test_only_the_adapters_are_registered`: if
the DiT, the VAE or the text tower ever become submodules, ``state_dict()`` grows to ~10 GB,
``TrainingMonitor.snapshot_params`` clones that per monitored step, and "save adapters only"
stops being structural.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from wam.backbones.wan_flow import WanFlowBackbone
from wam.backbones.wan_i2v import WanBackboneConfig, WanI2VAdapter
from wam.decoders import ActionHeadConfig
from wam.encoders import ActionChunkEncoderConfig, StateMLPConfig
from wam.interfaces import FlowBackbone
from wam.training import JointTrainer, JointTrainingConfig, TrainingMonitor

pytest.importorskip("diffusers")
pytest.importorskip("peft")

Z = 4  # latent channels == DiT in_channels: the TI2V-5B layout, the only one flow supports
INNER_DIM = 64  # 2 heads x 32
TEXT_DIM = 16
LAYERS = 3
SPATIAL = 8
TEMPORAL = 4
FEATURE_BLOCKS = (1, 2)
IMAGE_HW = (32, 48)  # -> latents [B, 4, 2, 4, 6] -> 12 video tokens
NUM_FRAMES = 5
STATE_DIM = 16
LATENTS_MEAN = 0.25
LATENTS_STD = 4.0

NUM_JOINTS = 6
NUM_STEPS = 8

_WAN_KWARGS = {
    "patch_size": (1, 2, 2),
    "num_attention_heads": 2,
    "attention_head_dim": 32,
    "in_channels": Z,
    "out_channels": Z,
    "text_dim": TEXT_DIM,
    "freq_dim": 16,
    "ffn_dim": 128,
    "num_layers": LAYERS,
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
    "eps": 1e-6,
}


def tiny_wan(**overrides) -> nn.Module:
    """A real WanTransformer3DModel — same code path as the 5B, toy dimensions."""
    from diffusers import WanTransformer3DModel

    torch.manual_seed(0)
    return WanTransformer3DModel(**{**_WAN_KWARGS, **overrides})


class _Config(dict):
    """Attribute access over a dict, like diffusers' FrozenDict configs."""

    def __getattr__(self, item: str):
        try:
            return self[item]
        except KeyError as err:
            raise AttributeError(item) from err


class _LatentDist:
    def __init__(self, value: torch.Tensor) -> None:
        self._value = value

    def mode(self) -> torch.Tensor:
        return self._value


class _FakeWanVAE(nn.Module):
    """Wan-VAE stand-in: real stride/normalization semantics, hand-computable latents."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _Config(
            z_dim=Z,
            latents_mean=[LATENTS_MEAN] * Z,
            latents_std=[LATENTS_STD] * Z,
            temperal_downsample=[True, False, True],
            scale_factor_spatial=SPATIAL,
            scale_factor_temporal=TEMPORAL,
        )
        self.gain = nn.Parameter(torch.ones(1))  # gives _device_of()/dtype something to read

    def encode(self, pixels: torch.Tensor) -> _Config:
        latent_frames = 1 + (pixels.shape[2] - 1) // TEMPORAL
        gray = pixels.mean(dim=1, keepdim=True)[:, :, :latent_frames]
        pooled = torch.nn.functional.avg_pool3d(gray, kernel_size=(1, SPATIAL, SPATIAL))
        gains = torch.arange(1, Z + 1, dtype=pooled.dtype).view(1, Z, 1, 1, 1)
        return _Config(latent_dist=_LatentDist(pooled * gains))

    def decode(self, latents: torch.Tensor) -> _Config:
        _, _, frames, height, width = latents.shape
        pixels = latents.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1, -1)
        size = ((frames - 1) * TEMPORAL + 1, height * SPATIAL, width * SPATIAL)
        return _Config(sample=torch.nn.functional.interpolate(pixels, size=size, mode="nearest"))


class _StubTextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(64, TEXT_DIM)

    def forward(self, ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> _Config:
        return _Config(last_hidden_state=self.embed(ids))


class _StubTokenizer:
    def __call__(self, prompts, padding=None, max_length=8, truncation=True, **_kwargs):
        ids = torch.arange(1, max_length + 1).repeat(len(prompts), 1) % 64
        mask = torch.ones(len(prompts), max_length, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}


def wan_config(**overrides) -> WanBackboneConfig:
    """The shipped Wan config shape, shrunk to the toy DiT (fp32/CPU, rank-1 adapters)."""
    kwargs: dict = {
        "dtype": "float32",
        "device": "cpu",
        "feature_dim": INNER_DIM,
        "num_frames": NUM_FRAMES,
        "image_hw": IMAGE_HW,
        "state_embedding_dim": STATE_DIM,
        "max_text_tokens": 8,
        "feature_blocks": FEATURE_BLOCKS,
        "lora_rank": 1,
        "lora_alpha": 2,
        "gradient_checkpointing": True,
    }
    kwargs.update(overrides)
    return WanBackboneConfig(**kwargs)


def make_backbone(
    config: WanBackboneConfig | None = None,
    *,
    transformer: nn.Module | None = None,
    adapter: WanI2VAdapter | None = None,
    lora_dir=None,
) -> WanFlowBackbone:
    backbone = WanFlowBackbone(config or wan_config(), adapter=adapter)
    backbone.attach(
        transformer=tiny_wan() if transformer is None else transformer,
        vae=_FakeWanVAE(),
        text_encoder=_StubTextEncoder(),
        tokenizer=_StubTokenizer(),
        lora_dir=lora_dir,
    )
    return backbone


def flow_inputs(backbone: WanFlowBackbone, *, batch: int = 1):
    """(clean latents [B, 4, 2, 4, 6], text ctx, state ctx) from GR00T-shaped raw frames."""
    rng = np.random.default_rng(0)
    video = rng.integers(0, 256, size=(batch, NUM_FRAMES, 120, 160, 3), dtype=np.uint8)
    latents = backbone.encode_video(video)
    text = backbone.condition_text("pick up the red cube")
    state = backbone.condition_state(torch.zeros(batch, STATE_DIM))
    return latents, text, state


def randomize_lora(backbone: WanFlowBackbone) -> None:
    """Move the adapters off their init.

    peft initializes lora_B to ZERO, so at step 0 the gradient w.r.t. every lora_A is exactly
    zero — a real property of LoRA, not a bug, but it makes "all adapters receive gradient"
    unobservable. Any run past step 1 is in the regime this simulates.
    """
    with torch.no_grad():
        for param in backbone.lora.values():
            param.normal_(std=0.05)


# ---- module boundary: the 20-GB guard --------------------------------------------------------


class TestModuleBoundary:
    def test_only_the_adapters_are_registered(self) -> None:
        backbone = make_backbone()
        dit = backbone.adapter._transformer
        state = backbone.state_dict()

        assert state, "an attached backbone must checkpoint its adapters"
        assert all(key.startswith(("lora.", "state_proj.")) for key in state), sorted(state)
        # Not one DiT/VAE/text-tower tensor name may appear anywhere in the keys.
        frozen_names = {n for n, _ in dit.named_parameters() if "lora_" not in n}
        frozen_names |= {n for n, _ in backbone.vae.named_parameters()}
        frozen_names |= {n for n, _ in backbone.text_encoder.named_parameters()}
        keys = " ".join(state)
        assert frozen_names and not any(name in keys for name in frozen_names)

        wrapper_params = sum(int(p.numel()) for p in backbone.parameters())
        dit_params = sum(int(p.numel()) for n, p in dit.named_parameters() if "lora_" not in n)
        assert 0 < wrapper_params < 0.05 * dit_params, (wrapper_params, dit_params)

        children = dict(backbone.named_children())
        assert set(children) == {"state_proj", "lora"}
        assert "transformer" not in children and "vae" not in children

    def test_the_adapter_is_a_plain_attribute(self) -> None:
        """nn.Module.__setattr__ intercepts Parameter/Module/Tensor — and nothing else."""
        backbone = make_backbone()
        assert not isinstance(backbone.adapter, nn.Module)
        assert "_adapter" in backbone.__dict__  # __dict__, not _modules
        assert "_adapter" not in dict(backbone.named_modules())

        frozen_ids = {
            id(p)
            for module in backbone._held_modules()
            for n, p in module.named_parameters()
            if "lora_" not in n
        }
        assert frozen_ids and not (frozen_ids & {id(p) for p in backbone.parameters()})

    def test_an_unloaded_backbone_has_nothing_to_checkpoint(self) -> None:
        backbone = WanFlowBackbone(wan_config())
        assert not backbone.is_loaded
        assert backbone.state_dict() == {} and list(backbone.parameters()) == []

    def test_registry_resolves_the_wan_kind_to_this_class(self) -> None:
        """``build_backbone`` imports wan_flow lazily; that path is live now that it exists."""
        from wam.backbones.registry import build_backbone

        backbone = build_backbone(wan_config())
        assert isinstance(backbone, WanFlowBackbone)
        assert isinstance(backbone, FlowBackbone)
        assert not backbone.is_loaded  # load=False must not touch disk or network

    def test_a_module_adapter_is_refused(self) -> None:
        class ModuleAdapter(nn.Module, WanI2VAdapter):  # type: ignore[misc]
            pass

        with pytest.raises(TypeError, match="10 GB"):
            WanFlowBackbone(wan_config(), adapter=ModuleAdapter())


# ---- protocol conformance --------------------------------------------------------------------


class TestProtocol:
    def test_conforms_to_flow_backbone_and_names_its_frozen_parts(self) -> None:
        backbone = make_backbone()
        assert isinstance(backbone, FlowBackbone)
        assert backbone.frozen_part_names() == ("vae", "text_encoder")
        # The frozen-parts registry resolves those names with getattr — properties, not
        # submodules, so the registry works without the weights entering state_dict().
        assert backbone.vae is backbone.adapter._vae
        assert backbone.text_encoder is backbone.adapter._text_encoder
        assert backbone.name == "wan2.1-i2v"
        assert backbone.feature_dim == INNER_DIM

    def test_encode_video_applies_the_configured_image_hw(self) -> None:
        """The protocol's encode_video takes no size: DiT-legal grids are the backbone's job."""
        backbone = make_backbone()
        raw = np.full((1, NUM_FRAMES, 120, 160, 3), 200, dtype=np.uint8)  # GR00T-shaped
        latents = backbone.encode_video(raw)
        assert tuple(latents.shape) == (1, Z, 2, 4, 6)  # 120x160 -> 32x48, F' = 1 + (5-1)//4
        assert latents.dtype == torch.float32
        assert backbone.num_video_tokens(latents) == 2 * (4 // 2) * (6 // 2) == 12
        frames = backbone.decode_video(latents)
        assert tuple(frames.shape) == (1, NUM_FRAMES, *IMAGE_HW, 3)

    def test_setup_applies_the_config_lora_and_checkpointing_settings(self) -> None:
        backbone = make_backbone()
        assert backbone.adapter._transformer.gradient_checkpointing
        assert len(backbone.lora) == LAYERS * 10 * 2  # 10 Linears per block, lora_A + lora_B
        assert backbone.state_proj is not None
        assert backbone.state_proj.in_features == STATE_DIM
        assert all(p.requires_grad for p in backbone.parameters())

        restricted = make_backbone(wan_config(lora_blocks=(2,)))
        assert {name.split("__")[1] for name in restricted.lora} == {"2"}


# ---- aliasing --------------------------------------------------------------------------------


class TestLoraAliasing:
    def test_entries_are_the_dit_parameters_themselves(self) -> None:
        backbone = make_backbone()
        live = dict(backbone.adapter._transformer.named_parameters())
        assert len(backbone.lora) == len(backbone.adapter.lora_parameters())
        for key in backbone.lora:
            assert "." not in key and "__" in key  # ParameterDict rejects dotted keys
            assert backbone.lora[key] is live[key.replace("__", ".")]

    def test_identity_survives_a_dtype_cast(self) -> None:
        """AdamW/clip_grad_norm_ see the wrapper's parameters; the DiT computes with its own."""
        backbone = make_backbone()
        before = dict(backbone.lora.items())

        backbone.to(torch.bfloat16)

        live = dict(backbone.adapter._transformer.named_parameters())
        for key, param in backbone.lora.items():
            assert param is before[key]
            assert param is live[key.replace("__", ".")]
            assert param.dtype == torch.bfloat16
        # DEVICE moves are forwarded, dtype casts never are: the frozen towers keep the
        # dtypes §10.3 assigned them (bf16 DiT / fp32 VAE), whatever the caller asked for.
        base = [p for n, p in live.items() if "lora_" not in n]
        assert base and all(p.dtype == torch.float32 for p in base)
        assert backbone.vae.gain.dtype == torch.float32

    def test_a_device_move_is_forwarded_and_the_aliases_are_repointed(self) -> None:
        """Cross-device ``_apply`` REPLACES parameters — the aliases must follow the DiT."""
        backbone = make_backbone()
        backbone.to("cpu")  # genuine no-op on a CPU model: nothing may move
        assert next(backbone.adapter._transformer.parameters()).device.type == "cpu"

        backbone.to("meta")  # a device the test machine always has

        assert all(next(m.parameters()).device.type == "meta" for m in backbone._held_modules())
        assert next(backbone.state_proj.parameters()).device.type == "meta"
        assert backbone.adapter.device == "meta"  # the adapter places pixels/timesteps on it
        live = dict(backbone.adapter._transformer.named_parameters())
        assert all(backbone.lora[k] is live[k.replace("__", ".")] for k in backbone.lora)

    def test_device_map_placement_is_never_overridden(self) -> None:
        """With device_map, accelerate already sharded the model; a blanket .to() would OOM."""
        backbone = make_backbone(wan_config(device_map="cpu"))
        backbone.to("meta")
        assert all(next(m.parameters()).device.type == "cpu" for m in backbone._held_modules())


# ---- geometry gates --------------------------------------------------------------------------


class TestGeometryChecks:
    def test_feature_dim_mismatch_fails_at_load(self) -> None:
        with pytest.raises(RuntimeError, match="inner dim"):
            make_backbone(wan_config(feature_dim=INNER_DIM * 2))

    def test_illegal_image_hw_fails_at_load(self) -> None:
        """Every batch is resized to image_hw, so a wrong value is a config bug, not a data bug."""
        with pytest.raises(ValueError, match="not DiT-legal"):
            make_backbone(wan_config(image_hw=(30, 48)))

    def test_i2v_channel_packing_is_refused_up_front(self) -> None:
        with pytest.raises(NotImplementedError, match="in_channels"):
            make_backbone(transformer=tiny_wan(in_channels=2 * Z + TEMPORAL))

    def test_feature_blocks_are_rechecked_against_the_real_block_count(self) -> None:
        """The adapter's __init__ can only check against the deepest known Wan (40 blocks)."""
        deep = wan_config(feature_blocks=(30, 35))  # constructs fine, breaks on a 3-block DiT
        with pytest.raises(ValueError, match=f"out of range for a {LAYERS}-block DiT"):
            make_backbone(deep)

    def test_an_injected_adapter_must_read_out_the_configured_depths(self) -> None:
        foreign = WanI2VAdapter(feature_blocks=(0, 2), dtype="float32", max_text_tokens=8)
        with pytest.raises(RuntimeError, match="reads out blocks"):
            make_backbone(adapter=foreign)


# ---- flow pathway ----------------------------------------------------------------------------


class TestForwardFlow:
    def test_shapes_dtypes_and_graph_connection(self) -> None:
        backbone = make_backbone()
        latents, text, state = flow_inputs(backbone, batch=2)
        backbone.train()

        velocity, features = backbone.forward_flow(latents, 0.3, text, state)

        assert tuple(velocity.shape) == tuple(latents.shape) == (2, Z, 2, 4, 6)
        assert tuple(features.shape) == (2, 12, INNER_DIM)
        assert velocity.dtype == features.dtype == torch.float32
        # Under gradient checkpointing the hooked block outputs must stay in the graph.
        assert features.requires_grad and velocity.requires_grad

    def test_detached_features_raise_instead_of_training_on_constants(self, monkeypatch) -> None:
        backbone = make_backbone()
        latents, text, state = flow_inputs(backbone)
        velocity, features = backbone.forward_flow(latents, 0.3, text, state)
        monkeypatch.setattr(
            backbone.adapter, "forward_flow", lambda *a, **k: (velocity, features.detach())
        )

        backbone.train()
        with pytest.raises(RuntimeError, match="detached from the autograd graph"):
            backbone.forward_flow(latents, 0.3, text, state)

        # Inference has nothing to train, so the assertion must not fire there.
        backbone.eval()
        _, detached = backbone.forward_flow(latents, 0.3, text, state)
        assert not detached.requires_grad


# ---- checkpointing ---------------------------------------------------------------------------


class TestSaveAdapter:
    def test_round_trips_through_the_peft_directory(self, tmp_path) -> None:
        backbone = make_backbone()
        randomize_lora(backbone)
        with torch.no_grad():
            backbone.state_proj.weight.normal_()

        path = backbone.save_adapter(tmp_path / "adapter")
        assert (path / "pytorch_lora_weights.safetensors").exists()  # diffusers layout
        assert (path / "state_proj.safetensors").exists()
        info = json.loads((path / "backbone.json").read_text())
        assert info["config"]["kind"] == "wan_i2v"
        assert info["feature_blocks"] == list(FEATURE_BLOCKS)
        assert info["lora_tensors"] == len(backbone.lora)
        assert info["trainable_parameters"] == sum(int(p.numel()) for p in backbone.parameters())

        restored = make_backbone(lora_dir=path)
        assert set(restored.lora) == set(backbone.lora)
        assert all(torch.equal(backbone.lora[k], restored.lora[k]) for k in backbone.lora)
        assert torch.equal(backbone.state_proj.weight, restored.state_proj.weight)

    def test_a_missing_state_projection_is_not_silently_reinitialized(self, tmp_path) -> None:
        backbone = make_backbone()
        path = backbone.save_adapter(tmp_path / "adapter")
        (path / "state_proj.safetensors").unlink()
        with pytest.raises(FileNotFoundError, match="state projection"):
            make_backbone(lora_dir=path)


# ---- end to end: the joint model -------------------------------------------------------------


def joint_config(**overrides) -> JointTrainingConfig:
    """GR00T-shaped joint config (15 joints shrunk to 6) over the toy Wan backbone."""
    kwargs: dict = {
        "state": StateMLPConfig(embedding_dim=STATE_DIM, hidden_dims=(32,), num_joints=NUM_JOINTS),
        "backbone": wan_config(),
        "action_encoder": ActionChunkEncoderConfig(
            latent_dim=16, target_dim=NUM_JOINTS, hidden_dims=(32,)
        ),
        "head": ActionHeadConfig(
            feature_dim=INNER_DIM, num_steps=NUM_STEPS, target_dim=NUM_JOINTS, hidden_dims=(32,)
        ),
        "velocity_hidden_dims": (32,),
        "lr": 1e-3,
        "backbone_lr": 1e-3,
        "seed": 0,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return JointTrainingConfig(**kwargs)


def make_batch(batch_size: int = 2) -> dict:
    g = torch.Generator().manual_seed(1)
    return {
        "frames": torch.randint(
            0, 256, (batch_size, NUM_FRAMES, 120, 160, 3), generator=g, dtype=torch.uint8
        ),
        "q": torch.randn(batch_size, NUM_JOINTS, generator=g),
        "dq": torch.randn(batch_size, NUM_JOINTS, generator=g),
        "imu": torch.randn(batch_size, 10, generator=g),
        "gripper": torch.rand(batch_size, 1, generator=g),
        "targets": torch.rand(batch_size, NUM_STEPS, NUM_JOINTS, generator=g) * 1.6 - 0.8,
        "gripper_target": torch.rand(batch_size, NUM_STEPS, generator=g),
        "instruction": "pick up the red cube",
    }


class TestJointTraining:
    def test_gradients_reach_the_adapters_and_nothing_else(self) -> None:
        backbone = make_backbone()
        randomize_lora(backbone)
        trainer = JointTrainer(joint_config(), backbone=backbone)

        trainer.model.train()
        trainer.compute_losses(make_batch())["total"].backward()

        for key, param in backbone.lora.items():
            assert param.grad is not None, key
            assert torch.isfinite(param.grad).all(), key
            assert float(param.grad.abs().sum()) > 0.0, key
        for name, param in backbone.state_proj.named_parameters():
            assert param.grad is not None and float(param.grad.abs().sum()) > 0.0, name
        # The frozen base never even allocates a gradient buffer (attach() froze it).
        for name, param in backbone.adapter._transformer.named_parameters():
            if "lora_" not in name:
                assert param.grad is None, name
        assert TrainingMonitor.module_grad_norms(trainer.model)["backbone"] > 0.0

    def test_two_steps_move_the_adapters_and_leave_the_base_bit_identical(self) -> None:
        backbone = make_backbone()
        randomize_lora(backbone)
        trainer = JointTrainer(joint_config(), backbone=backbone)
        dit = backbone.adapter._transformer
        base_before = {n: p.detach().clone() for n, p in dit.named_parameters() if "lora_" not in n}
        lora_before = {k: p.detach().clone() for k, p in backbone.lora.items()}
        proj_before = backbone.state_proj.weight.detach().clone()

        history = trainer.train(make_batch(), steps=2)

        assert [np.isfinite(entry["total"]) for entry in history] == [True, True]
        assert all(not torch.equal(lora_before[k], p) for k, p in backbone.lora.items())
        assert not torch.equal(proj_before, backbone.state_proj.weight)
        for name, param in dit.named_parameters():
            if "lora_" not in name:
                assert torch.equal(base_before[name], param), name

    def test_the_checkpoint_payload_is_adapters_only(self) -> None:
        backbone = make_backbone()
        model = JointTrainer(joint_config(), backbone=backbone).model
        payload = model.trainable_state_dict()
        backbone_keys = [k for k in payload if k.startswith("backbone.")]
        assert backbone_keys
        assert all(k.startswith(("backbone.lora.", "backbone.state_proj.")) for k in backbone_keys)
        # ... and it loads back through the aliases into the DiT the wrapper holds.
        with torch.no_grad():
            for param in backbone.lora.values():
                param.normal_()
        model.load_state_dict(payload, strict=False)
        live = dict(backbone.adapter._transformer.named_parameters())
        key = next(iter(backbone.lora))
        assert torch.equal(payload[f"backbone.lora.{key}"], live[key.replace("__", ".")])
