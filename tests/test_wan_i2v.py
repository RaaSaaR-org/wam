"""WanI2VAdapter integration logic, exercised against stub modules (T-15, OD-04).

These stubs mirror the diffusers Wan API surface the adapter depends on — ``.blocks`` on the
DiT, ``vae.encode(...).latent_dist``, ``latents_mean/std``, ``temperal_downsample``,
``last_hidden_state`` on the text encoder — so shape/plumbing regressions are caught offline.
The real weights are validated on GPU by ``scripts/hf_job_wan_smoke.py`` (see docs/hf_jobs.md).
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from wam.backbones.wan_i2v import WanI2VAdapter, default_feature_blocks

TEXT_DIM = 32
INNER_DIM = 24  # 4 heads x 6
LATENT_CHANNELS = 8
NUM_LAYERS = 8
PATCH = (1, 2, 2)
VAE_SPATIAL = 8
VAE_TEMPORAL = 4


class _Config(dict):
    """Attribute access over a dict, like diffusers' FrozenDict configs."""

    def __getattr__(self, item: str):
        try:
            return self[item]
        except KeyError as err:
            raise AttributeError(item) from err


class _Block(nn.Module):
    """Residual-stream block: [B, S, inner] -> [B, S, inner], deterministic."""

    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index
        self.linear = nn.Linear(INNER_DIM, INNER_DIM)

    def forward(self, hidden_states: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        return hidden_states + torch.tanh(self.linear(hidden_states)) * (self.index + 1)


class _StubTransformer(nn.Module):
    """Minimal WanTransformer3DModel stand-in: patchify -> blocks -> hook target."""

    def __init__(self, in_channels: int = LATENT_CHANNELS, image_dim: int | None = None) -> None:
        super().__init__()
        self.config = _Config(
            num_layers=NUM_LAYERS,
            num_attention_heads=4,
            attention_head_dim=6,
            in_channels=in_channels,
            out_channels=LATENT_CHANNELS,
            text_dim=TEXT_DIM,
            patch_size=list(PATCH),
            image_dim=image_dim,
        )
        self.patch = nn.Conv3d(in_channels, INNER_DIM, kernel_size=PATCH, stride=PATCH)
        self.text_embedder = nn.Linear(TEXT_DIM, INNER_DIM)
        self.blocks = nn.ModuleList(_Block(i) for i in range(NUM_LAYERS))
        self.seen: dict[str, object] = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        **_kwargs,
    ):
        self.seen = {
            "in_channels": int(hidden_states.shape[1]),
            "timestep": timestep,
            "text_len": int(encoder_hidden_states.shape[1]),
            "image": encoder_hidden_states_image,
        }
        tokens = self.patch(hidden_states.float()).flatten(2).transpose(1, 2)  # [B, S, inner]
        context = self.text_embedder(encoder_hidden_states.float())
        for block in self.blocks:
            tokens = block(tokens + context.mean(dim=1, keepdim=True))
        return tokens if not return_dict else (tokens,)


class _LatentDist:
    def __init__(self, value: torch.Tensor) -> None:
        self._value = value

    def mode(self) -> torch.Tensor:
        return self._value


class _StubVAE(nn.Module):
    """Causal-3D-VAE stand-in with Wan's stride semantics and latent normalization stats."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _Config(
            z_dim=LATENT_CHANNELS,
            latents_mean=[0.1] * LATENT_CHANNELS,
            latents_std=[2.0] * LATENT_CHANNELS,
            temperal_downsample=[True, False, True],  # -> temporal 4, spatial 8
        )
        self.proj = nn.Conv3d(3, LATENT_CHANNELS, kernel_size=1)

    def encode(self, pixels: torch.Tensor) -> _Config:
        frames = pixels.shape[2]
        latent_frames = 1 + (frames - 1) // VAE_TEMPORAL
        projected = self.proj(pixels)[:, :, :latent_frames]
        latents = torch.nn.functional.avg_pool3d(
            projected, kernel_size=(1, VAE_SPATIAL, VAE_SPATIAL)
        )
        return _Config(latent_dist=_LatentDist(latents))


class _StubTextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(64, TEXT_DIM)

    def forward(self, ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> _Config:
        return _Config(last_hidden_state=self.embed(ids))


class _StubTokenizer:
    def __call__(self, prompts, padding=None, max_length=8, truncation=True, **_kwargs):
        batch = len(prompts)
        ids = torch.arange(1, max_length + 1).repeat(batch, 1) % 64
        mask = torch.ones(batch, max_length, dtype=torch.long)
        mask[:, max_length // 2 :] = 0  # exercise the padding-zeroing path
        return {"input_ids": ids, "attention_mask": mask}


def make_adapter(
    *, in_channels: int = LATENT_CHANNELS, blocks=None, seed: int = 0
) -> WanI2VAdapter:
    torch.manual_seed(seed)
    adapter = WanI2VAdapter(feature_blocks=blocks, dtype="float32", max_text_tokens=8)
    adapter.attach(
        transformer=_StubTransformer(in_channels=in_channels),
        vae=_StubVAE(),
        text_encoder=_StubTextEncoder(),
        tokenizer=_StubTokenizer(),
    )
    return adapter


def make_video(frames: int = 5, height: int = 32, width: int = 48) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(1, frames, height, width, 3), dtype=np.uint8)


# ---- geometry ------------------------------------------------------------------------------


def test_default_feature_blocks_matches_14b_default():
    assert default_feature_blocks(40) == (20, 30)
    assert default_feature_blocks(30) == (15, 22)
    with pytest.raises(ValueError):
        default_feature_blocks(1)


def test_attach_derives_geometry_from_configs():
    adapter = make_adapter()
    geometry = adapter.describe()
    assert adapter.is_loaded
    assert geometry["feature_dim"] == INNER_DIM  # not the 5120 default any more
    assert geometry["num_layers"] == NUM_LAYERS
    assert geometry["feature_blocks"] == [4, 6]  # auto: mid/late depth
    assert geometry["latent_channels"] == LATENT_CHANNELS
    assert (geometry["vae_temporal_stride"], geometry["vae_spatial_stride"]) == (4, 8)


def test_explicit_scale_factors_win_over_the_downsample_derivation():
    """Regression: a real Wan2.2-TI2V-5B run produced H/16 latents where we predicted H/8.

    Its VAE pixel-shuffles by patch_size on top of three downsample stages, so
    2**len(temperal_downsample) understates the compression. Those configs carry
    scale_factor_spatial/temporal; Wan2.1 configs carry neither and must still derive.
    """
    vae = _StubVAE()
    vae.config["scale_factor_spatial"] = 16
    vae.config["scale_factor_temporal"] = 4
    adapter = WanI2VAdapter(dtype="float32")
    adapter.attach(transformer=_StubTransformer(), vae=vae)
    geometry = adapter.describe()
    assert (geometry["vae_spatial_stride"], geometry["vae_temporal_stride"]) == (16, 4)
    # Token count follows the stride, so it has to move with it.
    assert adapter.expected_token_count(5, 256, 448) == 2 * (256 // 16 // 2) * (448 // 16 // 2)


def test_token_grid_is_the_factorization_of_the_token_count():
    """The spatial probe readout (I-1) reshapes [B, S, D] to [B, F', H', W', D], so the grid
    and the count must never disagree — and the grid must track the VAE stride and patch size."""
    vae = _StubVAE()
    vae.config["scale_factor_spatial"] = 16
    vae.config["scale_factor_temporal"] = 4
    adapter = WanI2VAdapter(dtype="float32")
    adapter.attach(transformer=_StubTransformer(), vae=vae)

    # The recorded probe geometry: 5 context frames at 192x256 -> 2 latent frames, 6x8 tokens.
    assert adapter.token_grid(5, 192, 256) == (2, 6, 8)
    for frames, height, width in ((5, 192, 256), (1, 32, 48), (9, 256, 448)):
        grid = adapter.token_grid(frames, height, width)
        assert grid[0] * grid[1] * grid[2] == adapter.expected_token_count(frames, height, width)


def test_device_map_is_recorded_and_suppresses_the_second_placement():
    """With device_map, accelerate has already placed every shard — attach must not re-move.

    Loading a 34 GB repo inside a 16 GB-RAM ZeroGPU Space depends on this path
    (deploy/wan-smoke-space/); the placement itself is exercised only on a real GPU.
    """
    adapter = WanI2VAdapter(dtype="float32", device_map="cuda")
    assert adapter.describe()["device_map"] == "cuda"
    assert WanI2VAdapter(dtype="float32").describe()["device_map"] is None

    moved: list[str] = []

    class _RecordingVAE(_StubVAE):
        def to(self, device):
            moved.append(str(device))
            return self

    adapter.attach(transformer=_StubTransformer(), vae=_RecordingVAE(), move_to_device=False)
    assert moved == []


def test_feature_blocks_out_of_range_for_shallow_model():
    adapter = WanI2VAdapter(feature_blocks=(30,), dtype="float32")
    with pytest.raises(ValueError, match="out of range for a 8-block DiT"):
        adapter.attach(transformer=_StubTransformer(), vae=_StubVAE())


def test_missing_blocks_attribute_is_a_clear_error():
    class NoBlocks(nn.Module):
        config = _Config(num_layers=2, num_attention_heads=1, attention_head_dim=2, in_channels=1)

    with pytest.raises(RuntimeError, match="has no '.blocks'"):
        WanI2VAdapter(dtype="float32").attach(transformer=NoBlocks(), vae=_StubVAE())


# ---- conditioning --------------------------------------------------------------------------


def test_condition_video_shapes_and_normalization():
    adapter = make_adapter()
    ctx = adapter.condition_video(make_video(frames=5, height=32, width=48))
    latents = ctx["latents"]
    assert tuple(latents.shape) == (1, LATENT_CHANNELS, 2, 4, 6)  # F'=1+(5-1)//4
    assert ctx["image_embeds"] is None  # no CLIP tower on this checkpoint
    assert torch.isfinite(latents).all()


def test_condition_video_rejects_unaligned_frame_size():
    adapter = make_adapter()
    with pytest.raises(ValueError, match="must be a multiple of 16x16"):
        adapter.condition_video(make_video(frames=5, height=30, width=48))


def test_condition_video_accepts_unbatched_and_float_input():
    adapter = make_adapter()
    frames = make_video(frames=5)[0]
    from_uint8 = adapter.condition_video(frames)["latents"]
    as_float = torch.from_numpy(frames).float() / 127.5 - 1.0
    from_float = adapter.condition_video(as_float)["latents"]
    assert torch.allclose(from_uint8, from_float, atol=1e-5)


def test_condition_text_zeroes_padding():
    adapter = make_adapter()
    ctx = adapter.condition_text("pick up the red cube")
    assert tuple(ctx.shape) == (1, 8, TEXT_DIM)
    assert torch.count_nonzero(ctx[:, 4:]) == 0  # stub masks the second half


def test_condition_state_projects_into_text_space():
    adapter = make_adapter()
    token = adapter.condition_state(np.zeros((1, 16), dtype=np.float32))
    assert tuple(token.shape) == (1, 1, TEXT_DIM)
    assert adapter.state_projection.in_features == 16
    assert adapter.state_projection.weight.requires_grad  # the one trainable block
    assert tuple(adapter.condition_state(np.zeros(16, dtype=np.float32)).shape) == (1, 1, TEXT_DIM)
    with pytest.raises(ValueError, match="state embedding dim changed"):
        adapter.condition_state(np.zeros((1, 5), dtype=np.float32))


# ---- feature readout -----------------------------------------------------------------------


def test_features_shape_matches_expected_token_count():
    adapter = make_adapter()
    video = make_video(frames=5, height=32, width=48)
    ctx = adapter.condition_video(video)
    text = adapter.condition_text("pick up the red cube")
    state = adapter.condition_state(np.zeros((1, 16), dtype=np.float32))
    features = adapter.features(ctx, text, state)
    tokens = adapter.expected_token_count(5, 32, 48)
    assert tokens == 2 * (32 // 8 // 2) * (48 // 8 // 2)
    assert tuple(features.shape) == (1, tokens, INNER_DIM)
    assert torch.isfinite(features).all()


def test_features_are_deterministic_and_hooks_are_removed():
    adapter = make_adapter()
    ctx = adapter.condition_video(make_video())
    text = adapter.condition_text("task")
    first = adapter.features(ctx, text, None)
    second = adapter.features(ctx, text, None)
    assert torch.equal(first, second)
    hooks = [len(b._forward_hooks) for b in adapter._transformer.blocks]
    assert hooks == [0] * NUM_LAYERS  # no leaked handles


def test_state_token_is_appended_to_the_text_context():
    adapter = make_adapter()
    ctx = adapter.condition_video(make_video())
    text = adapter.condition_text("task")
    adapter.features(ctx, text, None)
    without = adapter._transformer.seen["text_len"]
    adapter.features(ctx, text, adapter.condition_state(np.zeros((1, 16), dtype=np.float32)))
    assert adapter._transformer.seen["text_len"] == without + 1


def test_features_average_the_selected_blocks_only():
    adapter = make_adapter(blocks=(1,))
    other = make_adapter(blocks=(7,))
    other._transformer.load_state_dict(adapter._transformer.state_dict())
    ctx = adapter.condition_video(make_video())
    text = adapter.condition_text("task")
    assert not torch.allclose(adapter.features(ctx, text, None), other.features(ctx, text, None))


def test_i2v_channel_layout_is_built_for_36_channel_checkpoints():
    """Wan2.1-I2V DiTs take [latents, temporal mask, condition latents] = 2*z + stride."""
    adapter = make_adapter(in_channels=2 * LATENT_CHANNELS + VAE_TEMPORAL)
    ctx = adapter.condition_video(make_video())
    adapter.features(ctx, adapter.condition_text("task"), None)
    assert adapter._transformer.seen["in_channels"] == 2 * LATENT_CHANNELS + VAE_TEMPORAL


def test_unknown_channel_layout_fails_loudly():
    adapter = make_adapter(in_channels=LATENT_CHANNELS + 3)
    ctx = adapter.condition_video(make_video())
    with pytest.raises(RuntimeError, match="cannot build DiT input"):
        adapter.features(ctx, adapter.condition_text("task"), None)


def test_features_require_text_context():
    adapter = make_adapter()
    ctx = adapter.condition_video(make_video())
    with pytest.raises(ValueError, match="text_ctx is required"):
        adapter.features(ctx, None, None)


def test_timestep_is_passed_as_a_batch_tensor():
    adapter = make_adapter()
    ctx = adapter.condition_video(make_video())
    adapter.timestep = 7
    adapter.features(ctx, adapter.condition_text("task"), None)
    timestep = adapter._transformer.seen["timestep"]
    assert timestep.dtype == torch.long and timestep.tolist() == [7]


def test_offload_moves_components_and_conditioning_still_works():
    adapter = make_adapter()
    adapter.offload("text_encoder")
    assert tuple(adapter.condition_text("task").shape) == (1, 8, TEXT_DIM)
    with pytest.raises(ValueError, match="unknown component"):
        adapter.offload("nope")


def test_features_by_block_covers_all_blocks_and_features_averages_them():
    adapter = make_adapter()
    ctx = adapter.condition_video(make_video())
    text = adapter.condition_text("task")
    per_block = adapter.features_by_block(ctx, text, None, blocks=tuple(range(NUM_LAYERS)))
    assert sorted(per_block) == list(range(NUM_LAYERS))
    tokens = adapter.expected_token_count(5, 32, 48)
    assert all(tuple(t.shape) == (1, tokens, INNER_DIM) for t in per_block.values())
    averaged = torch.stack([per_block[i] for i in adapter.feature_blocks], dim=0).mean(dim=0)
    assert torch.allclose(adapter.features(ctx, text, None), averaged)
    with pytest.raises(ValueError, match="blocks must be non-empty"):
        adapter.features_by_block(ctx, text, None, blocks=(NUM_LAYERS,))
    hooks = [len(b._forward_hooks) for b in adapter._transformer.blocks]
    assert hooks == [0] * NUM_LAYERS  # no leaked handles, also not on the error path


# ---- readout ablation (scripts/hf_job_wan_smoke.py --ablate) -------------------------------


def _load_smoke_cli():
    spec = importlib.util.spec_from_file_location(
        "wan_smoke_cli",
        Path(__file__).resolve().parent.parent / "scripts" / "hf_job_wan_smoke.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _ContentTokenizer(_StubTokenizer):
    """The base stub ignores prompt text; the ablation's instruction probe needs it not to."""

    def __call__(self, prompts, **kwargs):
        batch = super().__call__(prompts, **kwargs)
        offset = sum(map(ord, "".join(prompts))) % 61
        batch["input_ids"] = (batch["input_ids"] + offset) % 64
        return batch


def test_ablation_ranking_normalizes_each_probe_column():
    smoke = _load_smoke_cli()
    base = {0: torch.ones(1, 4), 1: torch.ones(1, 4)}
    probes = {"motion": {0: torch.full((1, 4), 2.0), 1: torch.ones(1, 4)}}
    per_block, scores = smoke.ablation_ranking(base, probes)
    assert per_block[1]["motion"] == 0.0
    assert per_block[0]["motion"] == pytest.approx(0.5)  # ||1||*2 / max(2, 4)
    assert (scores[0], scores[1]) == (1.0, 0.0)


def test_run_ablation_probes_every_block_and_suggests_a_pair():
    smoke = _load_smoke_cli()
    adapter = make_adapter()
    adapter._tokenizer = _ContentTokenizer()
    args = argparse.Namespace(frames=5, height=32, width=48, instruction="pick up the red cube")
    report = smoke.Report()
    smoke.run_ablation(adapter, args, report)
    assert report.failed == []  # includes: every probe moved features somewhere
    ablation = report.info["ablation"]
    assert len(ablation["per_block"]) == NUM_LAYERS
    assert len(ablation["suggested_blocks"]) == 2
    assert ablation["default_blocks"] == [4, 6]
    hooks = [len(b._forward_hooks) for b in adapter._transformer.blocks]
    assert hooks == [0] * NUM_LAYERS


def test_run_ablation_flags_a_dead_conditioning_path():
    """With the content-blind stub tokenizer, the instruction probe cannot move features."""
    smoke = _load_smoke_cli()
    adapter = make_adapter()
    args = argparse.Namespace(frames=5, height=32, width=48, instruction="pick up the red cube")
    report = smoke.Report()
    smoke.run_ablation(adapter, args, report)
    assert report.failed == ["ablation.instruction_moves_features"]


def test_action_head_consumes_the_features():
    """The point of the whole exercise: backbone features -> canonical ActionChunk."""
    from wam.decoders.action_head import ActionHead, ActionHeadConfig

    adapter = make_adapter()
    ctx = adapter.condition_video(make_video())
    features = adapter.features(ctx, adapter.condition_text("task"), None)
    head = ActionHead(ActionHeadConfig(feature_dim=INNER_DIM, num_steps=16, target_dim=7))
    chunk = head.decode(features)
    assert chunk.targets.shape == (16, 7)
    assert np.isfinite(chunk.targets).all()


# ---- flow pathway (T-16): real tiny WanTransformer3DModel, no weights ----------------------
#
# Everything above runs against hand-written stubs. The flow pathway cannot: its whole job is
# to line WAM's conventions up with the REAL Wan DiT (timestep direction, velocity sign, token
# geometry, where peft may inject), and a stub would happily agree with a wrong implementation.
# So these tests build a genuine WanTransformer3DModel at toy dimensions (13.6k params, CPU,
# milliseconds) and pair it with a VAE fake whose latents are hand-computable.

Z = 4  # latent channels == DiT in_channels, i.e. the TI2V-5B layout (no 2z+mask packing)
FLOW_INNER_DIM = 16  # 2 heads x 8
FLOW_TEXT_DIM = 16
FLOW_LAYERS = 3
FLOW_SPATIAL = 8
FLOW_TEMPORAL = 4
LATENTS_MEAN = 0.25
LATENTS_STD = 4.0

_WAN_KWARGS = {
    "patch_size": (1, 2, 2),
    "num_attention_heads": 2,
    "attention_head_dim": 8,
    "in_channels": Z,
    "out_channels": Z,
    "text_dim": FLOW_TEXT_DIM,
    "freq_dim": 16,
    "ffn_dim": 32,
    "num_layers": FLOW_LAYERS,
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
    "eps": 1e-6,
}


def _wan_transformer_cls():
    diffusers = pytest.importorskip("diffusers")
    return diffusers.WanTransformer3DModel


def tiny_wan(**overrides) -> torch.nn.Module:
    """A real (13.6k-param) WanTransformer3DModel — same code path as the 5B, toy dimensions."""
    torch.manual_seed(0)
    return _wan_transformer_cls()(**{**_WAN_KWARGS, **overrides})


def constant_wan(value: float) -> torch.nn.Module:
    """Real Wan DiT with its OUTPUT pinned to a constant; the blocks (and hooks) still run.

    Lets a test read forward_flow's sign convention straight off the return value instead of
    re-deriving what the DiT would have predicted.
    """
    base = _wan_transformer_cls()

    class _ConstantWan(base):  # type: ignore[misc, valid-type]
        def forward(self, *args, **kwargs):
            out = super().forward(*args, **kwargs)
            sample = out[0] if isinstance(out, tuple) else out.sample
            return (torch.full_like(sample, self.constant),)

    torch.manual_seed(0)
    model = _ConstantWan(**_WAN_KWARGS)
    model.constant = float(value)
    return model


class _FakeWanVAE(nn.Module):
    """Wan-VAE stand-in: real stride/normalization semantics, hand-computable latents.

    Latent channel c is ``(c + 1) *`` the 8x8 block mean of the frame, so a constant clip has a
    latent this test file can predict in closed form — which is the only way to check that
    resize, [-1, 1] mapping and latents_mean/std are applied in the right order.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = _Config(
            z_dim=Z,
            latents_mean=[LATENTS_MEAN] * Z,
            latents_std=[LATENTS_STD] * Z,
            temperal_downsample=[True, False, True],
            scale_factor_spatial=FLOW_SPATIAL,
            scale_factor_temporal=FLOW_TEMPORAL,
        )
        self.gain = nn.Parameter(torch.ones(1))  # gives _device_of()/dtype something to read

    def encode(self, pixels: torch.Tensor) -> _Config:
        latent_frames = 1 + (pixels.shape[2] - 1) // FLOW_TEMPORAL
        gray = pixels.mean(dim=1, keepdim=True)[:, :, :latent_frames]
        pooled = torch.nn.functional.avg_pool3d(gray, kernel_size=(1, FLOW_SPATIAL, FLOW_SPATIAL))
        gains = torch.arange(1, Z + 1, dtype=pooled.dtype).view(1, Z, 1, 1, 1)
        return _Config(latent_dist=_LatentDist(pooled * gains))

    def decode(self, latents: torch.Tensor) -> _Config:
        _, _, frames, height, width = latents.shape
        pixels = latents.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1, -1)
        size = ((frames - 1) * FLOW_TEMPORAL + 1, height * FLOW_SPATIAL, width * FLOW_SPATIAL)
        return _Config(sample=torch.nn.functional.interpolate(pixels, size=size, mode="nearest"))


def flow_adapter(*, transformer: torch.nn.Module | None = None, **kwargs) -> WanI2VAdapter:
    adapter = WanI2VAdapter(dtype="float32", **kwargs)
    adapter.attach(
        transformer=transformer if transformer is not None else tiny_wan(), vae=_FakeWanVAE()
    )
    return adapter


def flow_inputs(adapter: WanI2VAdapter, *, batch: int = 1, frames: int = 5):
    """(latents, text_ctx, state_ctx) for a 32x48 clip -> latents [B, 4, 2, 4, 6], 12 tokens."""
    video = make_video(frames=frames, height=32, width=48).repeat(batch, axis=0)
    latents = adapter.encode_video(video)
    text = torch.zeros(1, 4, FLOW_TEXT_DIM)
    state = torch.zeros(batch, 1, FLOW_TEXT_DIM)
    return latents, text, state


def test_flow_adapter_conforms_to_the_flow_backbone_protocol():
    from wam.interfaces import FlowBackbone

    adapter = flow_adapter()
    assert isinstance(adapter, FlowBackbone)
    assert adapter.frozen_part_names() == ("_vae",)  # no text/image tower attached here


def test_forward_dit_returns_the_output_and_the_features_in_one_forward():
    """The refactor's reason to exist: readout used to run the DiT and throw the result away."""
    adapter = flow_adapter()
    latents, text, state = flow_inputs(adapter)
    calls: list[int] = []
    adapter._transformer.register_forward_hook(lambda *_a: calls.append(1))

    output, captured = adapter._forward_dit(latents, text, state)
    assert len(calls) == 1
    assert tuple(output.shape) == tuple(latents.shape)  # [B, z, F', h, w], not a token grid
    assert sorted(captured) == [1, 2]  # default_feature_blocks(3)
    assert all(tuple(a.shape) == (1, 12, FLOW_INNER_DIM) for a in captured.values())


def test_forward_dit_default_timestep_reproduces_the_probe_constant():
    """runs/wan_probe numbers were taken at a constant long timestep — keep that byte-for-byte."""
    adapter = flow_adapter()
    adapter.timestep = 7
    latents, text, state = flow_inputs(adapter)
    seen: list[torch.Tensor] = []
    adapter._transformer.register_forward_pre_hook(
        lambda _m, _a, kw: seen.append(kw["timestep"]), with_kwargs=True
    )
    adapter._forward_dit(latents, text, state)
    assert seen[0].dtype == torch.long and seen[0].tolist() == [7]


def test_forward_flow_returns_velocity_in_latent_shape_and_video_tokens():
    adapter = flow_adapter()
    latents, text, state = flow_inputs(adapter)
    velocity, features = adapter.forward_flow(latents, 0.4, text, state)
    assert tuple(velocity.shape) == tuple(latents.shape) == (1, Z, 2, 4, 6)
    tokens = adapter.num_video_tokens(latents)
    assert tokens == 2 * (4 // 2) * (6 // 2) == 12
    assert tuple(features.shape) == (1, tokens, adapter.feature_dim)
    assert velocity.dtype == features.dtype == torch.float32
    assert torch.isfinite(velocity).all() and torch.isfinite(features).all()


def test_forward_flow_negates_the_dit_velocity():
    """Wan's DiT points noise-ward, WAM's target is v = x1 - x0. The minus lives in one place."""
    adapter = flow_adapter(transformer=constant_wan(0.75))
    latents, text, state = flow_inputs(adapter)
    velocity, _ = adapter.forward_flow(latents, 0.4, text, state)
    assert torch.equal(velocity, torch.full_like(velocity, -0.75))


def test_forward_flow_maps_t_onto_wans_downward_schedule():
    adapter = flow_adapter()
    latents, text, state = flow_inputs(adapter, batch=2)
    seen: list[torch.Tensor] = []
    adapter._transformer.register_forward_pre_hook(
        lambda _m, _a, kw: seen.append(kw["timestep"]), with_kwargs=True
    )

    adapter.forward_flow(latents, 0.25, text, state)  # scalar broadcasts over the batch
    assert seen[-1].dtype == torch.float32
    assert seen[-1].tolist() == [750.0, 750.0]

    adapter.forward_flow(latents, torch.tensor([0.0, 1.0]), text, state)
    assert seen[-1].tolist() == [1000.0, 0.0]  # t=1 is CLEAN -> Wan step 0

    with pytest.raises(ValueError, match="does not match latent batch"):
        adapter.forward_flow(latents, torch.tensor([0.1, 0.2, 0.3]), text, state)


def test_forward_flow_broadcasts_a_single_instruction_over_the_batch():
    adapter = flow_adapter()
    latents, text, state = flow_inputs(adapter, batch=3)
    assert text.shape[0] == 1  # one instruction, three clips
    _, features = adapter.forward_flow(latents, 0.5, text, state)
    assert features.shape[0] == 3


def test_forward_flow_refuses_the_i2v_channel_packing():
    """2z+mask I2V checkpoints can still be READ from; training semantics are unresolved."""
    adapter = flow_adapter(transformer=tiny_wan(in_channels=2 * Z + FLOW_TEMPORAL))
    latents, text, state = flow_inputs(adapter)
    adapter.features(latents, text, state)  # readout is fine: [latents, mask, latents]
    with pytest.raises(NotImplementedError, match="in_channels"):
        adapter.forward_flow(latents, 0.5, text, state)


def test_encode_video_resizes_to_a_dit_legal_grid_and_normalizes():
    """GR00T-G1 frames are 120x160 and 120 is not a multiple of vae_spatial * patch (16)."""
    adapter = flow_adapter()
    raw = np.full((1, 5, 120, 160, 3), 200, dtype=np.uint8)
    with pytest.raises(ValueError, match="must be a multiple of 16x16"):
        adapter.encode_video(raw)

    latents = adapter.encode_video(raw, image_hw=(128, 160))
    assert tuple(latents.shape) == (1, Z, 2, 16, 20)  # F' = 1 + (5-1)//4
    assert latents.dtype == torch.float32
    pixel = 200 / 127.5 - 1.0  # uint8 -> [-1, 1]; a constant clip survives the resize exactly
    expected = torch.tensor(
        [((c + 1) * pixel - LATENTS_MEAN) / LATENTS_STD for c in range(Z)], dtype=torch.float32
    )
    assert torch.allclose(latents[0, :, 0, 0, 0], expected, atol=1e-6)
    assert torch.allclose(latents[0], latents[0, :, :1].expand_as(latents[0]), atol=1e-6)


def test_decode_video_round_trips_the_shape_into_pixels():
    adapter = flow_adapter()
    latents = adapter.encode_video(make_video(frames=5, height=32, width=48))
    frames = adapter.decode_video(latents)
    assert tuple(frames.shape) == (1, 5, 32, 48, 3)  # (F'-1)*4+1 pixel frames, s=8 upsampling
    assert float(frames.min()) >= 0.0 and float(frames.max()) <= 1.0
    # denormalize is the exact inverse of the normalization encode_video applied.
    assert torch.allclose(
        adapter.denormalize_latents(latents), (latents * LATENTS_STD) + LATENTS_MEAN, atol=1e-6
    )


def test_num_video_tokens_rejects_a_non_latent_tensor():
    adapter = flow_adapter()
    with pytest.raises(ValueError, match=r"video_latents must be"):
        adapter.num_video_tokens(torch.zeros(1, Z, 4, 6))


# ---- LoRA ----------------------------------------------------------------------------------


def lora_adapter(**kwargs) -> tuple[WanI2VAdapter, int]:
    pytest.importorskip("peft")
    adapter = flow_adapter()
    trainable = adapter.add_lora(rank=2, alpha=4, dropout=0.0, **kwargs)
    return adapter, trainable


def test_add_lora_unfreezes_exactly_the_lora_parameters():
    """attach() froze the whole DiT; peft injects into that frozen tree (wan_i2v.py attach)."""
    adapter, trainable = lora_adapter()
    params = adapter.lora_parameters()
    assert trainable > 0 and trainable == sum(p.numel() for p in params.values())
    assert len(params) == FLOW_LAYERS * 10 * 2  # 10 Linears per block, lora_A + lora_B
    unfrozen = {n for n, p in adapter._transformer.named_parameters() if p.requires_grad}
    assert unfrozen == set(params)
    assert all(p.requires_grad and p.dtype == torch.float32 for p in params.values())
    base = [p for n, p in adapter._transformer.named_parameters() if "lora_" not in n]
    assert base and not any(p.requires_grad for p in base)


def test_add_lora_still_allows_the_readout_hooks():
    """get_peft_model() would rename the tree to base_model.model.blocks.i and break these."""
    adapter, _ = lora_adapter()
    latents, text, state = flow_inputs(adapter)
    _, features = adapter.forward_flow(latents, 0.5, text, state)
    assert tuple(features.shape) == (1, 12, adapter.feature_dim)
    assert [len(b._forward_hooks) for b in adapter._transformer.blocks] == [0] * FLOW_LAYERS


def test_add_lora_blocks_restricts_the_injection_to_those_depths():
    adapter, _ = lora_adapter(blocks=(1, 2))
    depths = {name.split(".")[1] for name in adapter.lora_parameters()}
    assert depths == {"1", "2"}  # block 0 untouched
    assert len(adapter.lora_parameters()) == 2 * 10 * 2
    with pytest.raises(ValueError, match=r"blocks must be non-empty indices"):
        flow_adapter().add_lora(rank=2, alpha=4, blocks=(FLOW_LAYERS,))


def test_second_add_lora_is_refused():
    adapter, _ = lora_adapter()
    with pytest.raises(RuntimeError, match="already attached"):
        adapter.add_lora(rank=2, alpha=4)


def test_lora_state_dict_round_trip_is_bit_exact():
    adapter, _ = lora_adapter()
    with torch.no_grad():
        for param in adapter.lora_parameters().values():
            param.normal_()
    saved = adapter.lora_state_dict()

    other, _ = lora_adapter()
    other.load_lora_state_dict(saved)
    assert all(torch.equal(saved[n], p.detach().cpu()) for n, p in other.lora_parameters().items())
    with pytest.raises(ValueError, match="does not match the injected adapter"):
        other.load_lora_state_dict({n: v for n, v in list(saved.items())[:-1]})


def test_save_lora_writes_the_diffusers_layout_and_loads_back(tmp_path):
    adapter, trainable = lora_adapter()
    with torch.no_grad():
        for param in adapter.lora_parameters().values():
            param.normal_()
    weights = adapter.save_lora(tmp_path / "lora")
    assert weights.exists() and weights.name.endswith(".safetensors")

    fresh = flow_adapter()
    assert fresh.load_lora(tmp_path / "lora") == trainable
    expected = adapter.lora_state_dict()
    assert all(
        torch.equal(expected[n], p.detach().cpu()) for n, p in fresh.lora_parameters().items()
    )


def test_trainable_parameters_cover_lora_and_the_state_projection():
    adapter, _ = lora_adapter()
    adapter.build_state_projection(16)
    names = adapter.trainable_parameters()
    assert {n.split(".")[0] for n in names} == {"transformer", "state_proj"}
    assert names["state_proj.weight"].dtype == torch.float32
    assert all(p.requires_grad for p in names.values())


def test_features_stay_graph_connected_under_gradient_checkpointing():
    """Reentrant checkpointing detaches block outputs — our features ARE those outputs."""
    adapter, _ = lora_adapter()
    adapter.enable_gradient_checkpointing()
    assert adapter._transformer.gradient_checkpointing
    latents, text, state = flow_inputs(adapter)
    velocity, features = adapter.forward_flow(latents, 0.3, text, state)
    assert features.requires_grad and velocity.requires_grad

    (features.square().mean() + velocity.square().mean()).backward()
    grads = [p.grad for p in adapter.lora_parameters().values() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0.0 for g in grads)
    adapter.enable_gradient_checkpointing(False)
    assert not adapter._transformer.gradient_checkpointing


# ---- config ---------------------------------------------------------------------------------


def test_wan_backbone_config_is_frozen_torch_free_and_hashable():
    from pydantic import ValidationError

    from wam.backbones.wan_i2v import DEFAULT_LORA_TARGETS, WanBackboneConfig
    from wam.interfaces.versioning import config_hash

    cfg = WanBackboneConfig(model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers", feature_blocks=(2, 10))
    assert cfg.kind == "wan_i2v"
    assert cfg.lora_targets == DEFAULT_LORA_TARGETS
    assert cfg.dtype == "bfloat16" and cfg.lora_param_dtype == "float32"
    assert config_hash(cfg) == config_hash(cfg.model_copy())  # every field is JSON-primitive
    with pytest.raises(ValidationError):  # frozen: configs go into run metadata (AC-04)
        cfg.dtype = "float32"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        WanBackboneConfig(patch_size=8)  # a tiny-shaped field must not validate silently
