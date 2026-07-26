"""T-15 tests: backbone adapters behind one interface (FR-09/AC-05).

Covers: tiny end-to-end feature extraction shapes, determinism under seed, protocol/signature
conformance for all three adapters, registry, and finite grads through the flow pathway.
CPU-only, deterministic, tiny dims.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from wam.backbones import available_backbones, get_backbone
from wam.backbones.flux3 import Flux3Adapter
from wam.backbones.tiny import TinyBackboneConfig, TinyVideoBackbone
from wam.backbones.wan_i2v import WAN_DIT_HIDDEN_DIM, WanI2VAdapter
from wam.interfaces.protocols import BackboneAdapter

TINY_CFG = {
    "feature_dim": 32,
    "patch_size": 8,
    "depth": 1,
    "num_heads": 4,
    "num_frames": 2,
    "image_hw": (16, 16),
    "text_vocab": 64,
    "max_text_tokens": 8,
    "state_embedding_dim": 8,
}
NUM_VIDEO_TOKENS = 2 * (16 // 8) * (16 // 8)  # 8
INSTRUCTION = "pick up the red cube"  # 5 hash tokens
PROTOCOL_METHODS = ("condition_video", "condition_text", "condition_state", "features")


def make_tiny(seed: int = 0) -> TinyVideoBackbone:
    torch.manual_seed(seed)
    return TinyVideoBackbone(TinyBackboneConfig(**TINY_CFG))


def make_video(batch: int | None = 2, dtype=np.uint8) -> np.ndarray:
    rng = np.random.RandomState(7)
    shape = (2, 16, 16, 3) if batch is None else (batch, 2, 16, 16, 3)
    if dtype == np.uint8:
        return rng.randint(0, 256, size=shape, dtype=np.uint8)
    return rng.rand(*shape).astype(np.float32)


def all_adapters() -> list[BackboneAdapter]:
    return [make_tiny(), WanI2VAdapter(), Flux3Adapter()]


# ---- protocol / signature conformance (all three) ----------------------------------------


def test_protocol_conformance_all_adapters():
    for adapter in all_adapters():
        assert isinstance(adapter, BackboneAdapter), type(adapter).__name__
        assert isinstance(adapter.name, str) and adapter.name
        assert isinstance(adapter.feature_dim, int) and adapter.feature_dim > 0


def test_signature_conformance_all_adapters():
    for adapter in all_adapters():
        for method in PROTOCOL_METHODS:
            proto_params = list(inspect.signature(getattr(BackboneAdapter, method)).parameters)
            impl_params = list(inspect.signature(getattr(type(adapter), method)).parameters)
            assert impl_params == proto_params, f"{type(adapter).__name__}.{method}"


def test_adapter_names_are_distinct():
    names = {a.name for a in all_adapters()}
    assert names == {"tiny", "wan2.1-i2v", "flux3-dev"}


# ---- tiny: end-to-end feature extraction --------------------------------------------------


def test_tiny_feature_shapes_batched_uint8():
    tiny = make_tiny()
    video_ctx = tiny.condition_video(make_video(batch=2))
    text_ctx = tiny.condition_text(INSTRUCTION)
    state_ctx = tiny.condition_state(torch.randn(2, 8))

    assert video_ctx.shape == (2, NUM_VIDEO_TOKENS, 32)
    assert text_ctx.shape == (1, 5, 32)
    assert state_ctx.shape == (2, 1, 32)

    feats = tiny.features(video_ctx, text_ctx, state_ctx)
    assert feats.shape == (2, NUM_VIDEO_TOKENS + 5 + 1, 32)
    assert feats.shape[-1] == tiny.feature_dim
    assert torch.isfinite(feats).all()


def test_tiny_accepts_unbatched_float_video_and_1d_state():
    tiny = make_tiny()
    video_ctx = tiny.condition_video(make_video(batch=None, dtype=np.float32))
    state_ctx = tiny.condition_state(torch.randn(8))
    feats = tiny.features(video_ctx, tiny.condition_text("lift"), state_ctx)
    assert feats.shape == (1, NUM_VIDEO_TOKENS + 1 + 1, 32)
    assert torch.isfinite(feats).all()


def test_tiny_rejects_bad_shapes():
    tiny = make_tiny()
    with pytest.raises(ValueError):
        tiny.condition_video(np.zeros((2, 8, 8, 3), dtype=np.uint8))  # wrong H, W
    with pytest.raises(ValueError):
        tiny.condition_state(torch.randn(2, 5))  # wrong embedding dim
    with pytest.raises(ValueError):
        tiny.features(torch.randn(2, 3, 32), tiny.condition_text("x"), torch.randn(2, 1, 32))


def test_tiny_uint8_matches_manual_scaling():
    tiny = make_tiny()
    video = make_video(batch=1)
    out_uint8 = tiny.condition_video(video)
    out_float = tiny.condition_video(video.astype(np.float32) / 255.0)
    assert torch.allclose(out_uint8, out_float)


# ---- tiny: determinism ---------------------------------------------------------------------


def test_tiny_determinism_under_seed():
    video = make_video()
    state = torch.ones(2, 8)

    outs = []
    for _ in range(2):
        tiny = make_tiny(seed=0)
        feats = tiny.features(
            tiny.condition_video(video),
            tiny.condition_text(INSTRUCTION),
            tiny.condition_state(state),
        )
        outs.append(feats)
    assert torch.equal(outs[0], outs[1])

    tiny = make_tiny(seed=0)
    ctx = (
        tiny.condition_video(video),
        tiny.condition_text(INSTRUCTION),
        tiny.condition_state(state),
    )
    assert torch.equal(tiny.features(*ctx), tiny.features(*ctx))  # forward has no RNG


def test_tiny_text_hash_embedding_deterministic_and_text_sensitive():
    tiny = make_tiny()
    a1 = tiny.condition_text("place the cup on the marker")
    a2 = tiny.condition_text("place the cup on the marker")
    b = tiny.condition_text("open the drawer")
    assert torch.equal(a1, a2)
    assert a1.shape[-1] == 32 and b.shape == (1, 3, 32)
    assert not torch.equal(a1[:, :3], b)  # different instructions -> different tokens
    assert tiny.condition_text("").shape == (1, 1, 32)  # empty text -> one padding token


# ---- tiny: video branch + flow pathway -----------------------------------------------------


def test_tiny_predict_video_latents_shape():
    tiny = make_tiny()
    feats = tiny.features(
        tiny.condition_video(make_video()),
        tiny.condition_text(INSTRUCTION),
        tiny.condition_state(torch.randn(2, 8)),
    )
    latents = tiny.predict_video_latents(feats)
    assert latents.shape == (2, 2, 16, 16, 3)
    assert torch.isfinite(latents).all()
    with pytest.raises(ValueError):
        tiny.predict_video_latents(torch.randn(2, NUM_VIDEO_TOKENS - 1, 32))


def test_tiny_forward_flow_shapes_and_finite_grads():
    tiny = make_tiny()
    noisy = torch.randn(2, 2, 16, 16, 3, requires_grad=True)
    t = torch.tensor([0.25, 0.75])
    text_ctx = tiny.condition_text(INSTRUCTION)
    state_ctx = tiny.condition_state(torch.randn(2, 8))

    velocity, feats = tiny.forward_flow(noisy, t, text_ctx, state_ctx)
    assert velocity.shape == noisy.shape
    assert feats.shape == (2, NUM_VIDEO_TOKENS + 5 + 1, 32)

    loss = velocity.pow(2).mean() + feats.pow(2).mean()
    loss.backward()
    assert noisy.grad is not None
    assert torch.isfinite(noisy.grad).all()
    assert noisy.grad.abs().sum() > 0
    param_grads = [p.grad for p in tiny.parameters() if p.grad is not None]
    assert param_grads, "flow pathway must reach model parameters"
    assert all(torch.isfinite(g).all() for g in param_grads)


def test_tiny_forward_flow_scalar_t_and_determinism():
    noisy = torch.full((1, 2, 16, 16, 3), 0.5)
    tiny = make_tiny(seed=1)
    ctx = (tiny.condition_text("grasp"), tiny.condition_state(torch.zeros(1, 8)))
    v1, f1 = tiny.forward_flow(noisy, 0.5, *ctx)
    v2, f2 = tiny.forward_flow(noisy, torch.tensor(0.5), *ctx)
    assert torch.equal(v1, v2) and torch.equal(f1, f2)
    with pytest.raises(ValueError):
        tiny.forward_flow(noisy, torch.tensor([0.1, 0.9]), *ctx)  # t batch mismatch


# ---- wan_i2v skeleton: error path only -----------------------------------------------------


def test_wan_i2v_skeleton_error_path():
    wan = WanI2VAdapter(checkpoint_path="/nonexistent/wan2.1-i2v-14b")
    assert wan.name == "wan2.1-i2v"
    assert wan.feature_dim == WAN_DIT_HIDDEN_DIM == 5120
    assert not wan.is_loaded
    with pytest.raises(RuntimeError, match="Wan2.1 weights not available"):
        wan.load()
    with pytest.raises(RuntimeError, match="Wan2.1 weights not available"):
        wan.condition_video(np.zeros((1, 2, 16, 16, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="Wan2.1 weights not available"):
        wan.condition_text(INSTRUCTION)
    with pytest.raises(RuntimeError, match="Wan2.1 weights not available"):
        wan.condition_state(np.zeros((1, 8), dtype=np.float32))
    with pytest.raises(RuntimeError, match="Wan2.1 weights not available"):
        wan.features(None, None, None)
    with pytest.raises(RuntimeError, match="Wan2.1 weights not available"):
        WanI2VAdapter().load()  # no checkpoint path at all
    with pytest.raises(ValueError):
        WanI2VAdapter(feature_blocks=(99,))  # out of range for 40 blocks


# ---- flux3 stub ----------------------------------------------------------------------------


def test_flux3_stub_raises_pending():
    flux = Flux3Adapter()
    assert flux.name == "flux3-dev"
    assert flux.feature_dim == 4096
    assert Flux3Adapter(feature_dim=1024).feature_dim == 1024
    for call in (
        lambda: flux.condition_video(None),
        lambda: flux.condition_text(INSTRUCTION),
        lambda: flux.condition_state(None),
        lambda: flux.features(None, None, None),
    ):
        with pytest.raises(NotImplementedError, match="FLUX 3 access pending — OD-06"):
            call()


# ---- registry (AC-05: swap by name) --------------------------------------------------------


def test_registry_lists_and_constructs_all():
    assert available_backbones() == ("flux3", "tiny", "wan_i2v")
    torch.manual_seed(0)
    tiny = get_backbone("tiny", **TINY_CFG)
    assert isinstance(tiny, TinyVideoBackbone) and tiny.feature_dim == 32
    assert isinstance(get_backbone("wan_i2v"), WanI2VAdapter)
    assert isinstance(get_backbone("FLUX3"), Flux3Adapter)  # case-insensitive
    for name in available_backbones():
        torch.manual_seed(0)
        assert isinstance(get_backbone(name), BackboneAdapter)


def test_registry_tiny_accepts_config_object_and_rejects_mixing():
    torch.manual_seed(0)
    tiny = get_backbone("tiny", config=TinyBackboneConfig(**TINY_CFG))
    assert tiny.feature_dim == 32
    with pytest.raises(TypeError):
        get_backbone("tiny", config=TinyBackboneConfig(**TINY_CFG), feature_dim=16)


def test_registry_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown backbone"):
        get_backbone("sora")
