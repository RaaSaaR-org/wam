"""WanI2VAdapter integration logic, exercised against stub modules (T-15, OD-04).

These stubs mirror the diffusers Wan API surface the adapter depends on — ``.blocks`` on the
DiT, ``vae.encode(...).latent_dist``, ``latents_mean/std``, ``temperal_downsample``,
``last_hidden_state`` on the text encoder — so shape/plumbing regressions are caught offline.
The real weights are validated on GPU by ``scripts/hf_job_wan_smoke.py`` (see docs/hf_jobs.md).
"""

from __future__ import annotations

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
