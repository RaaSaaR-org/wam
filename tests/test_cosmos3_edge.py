"""E-04 tests: the Cosmos3-Edge backbone adapter behind the existing contract (FR-09/AC-05).

Covers: registry construction from a bare name with no torch and no weights, protocol and
signature conformance, config defaults and validation, and every refusal path — missing
weights, the unsettled action width [?], the absent proprioception input, and the unverified
feature readout depth.

CPU-only, no downloads, no torch import required by the adapter itself (this module imports
torch only because the shared pytest environment already has it; see
``test_module_imports_without_torch_or_diffusers`` for the actual AC-05 guarantee).
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from wam.backbones import available_backbones, get_backbone
from wam.backbones.cosmos3_edge import (
    COSMOS3_EDGE_BASE_MODEL_ID,
    COSMOS3_EDGE_HIDDEN_DIM,
    COSMOS3_EDGE_MAX_ACTION_DIM,
    COSMOS3_EDGE_NAME,
    COSMOS3_EDGE_NUM_EMBODIMENT_DOMAINS,
    COSMOS3_EDGE_NUM_LAYERS,
    G1_DEX3_ACTION_DIM,
    POLICY_DOMAIN_NAME,
    Cosmos3EdgeAdapter,
    Cosmos3EdgeConfig,
)
from wam.interfaces.protocols import BackboneAdapter

INSTRUCTION = "pick up the red cube and place it on the plate"
PROTOCOL_METHODS = ("condition_video", "condition_text", "condition_state", "features")


def make_frames(shape: tuple[int, ...] = (3, 16, 20, 3)) -> np.ndarray:
    rng = np.random.RandomState(11)
    return rng.randint(0, 256, size=shape, dtype=np.uint8)


# ---- AC-05: bare name, no weights, no torch ------------------------------------------------


def test_registry_lists_and_constructs_cosmos3_edge():
    assert "cosmos3_edge" in available_backbones()
    adapter = get_backbone("cosmos3_edge")
    assert isinstance(adapter, Cosmos3EdgeAdapter)
    assert isinstance(adapter, BackboneAdapter)
    assert adapter.name == COSMOS3_EDGE_NAME == "cosmos3-edge"
    assert not adapter.is_loaded


def test_registry_is_case_insensitive_and_passes_kwargs():
    assert isinstance(get_backbone("COSMOS3_EDGE"), Cosmos3EdgeAdapter)
    adapter = get_backbone("cosmos3_edge", feature_dim=512, raw_action_dim=8)
    assert adapter.feature_dim == 512
    assert adapter.require_raw_action_dim() == 8


def test_registry_accepts_config_object_and_rejects_mixing():
    adapter = get_backbone("cosmos3_edge", config=Cosmos3EdgeConfig(feature_dim=64))
    assert adapter.feature_dim == 64
    with pytest.raises(TypeError):
        get_backbone("cosmos3_edge", config=Cosmos3EdgeConfig(), feature_dim=64)
    with pytest.raises(TypeError):
        Cosmos3EdgeAdapter(Cosmos3EdgeConfig(), feature_dim=64)


def test_module_imports_without_torch_or_diffusers():
    """AC-05, the load-bearing one: importing and constructing must not pull in torch."""
    code = (
        "import sys\n"
        "from wam.backbones.registry import get_backbone\n"
        "a = get_backbone('cosmos3_edge')\n"
        "assert a.name == 'cosmos3-edge' and a.feature_dim > 0\n"
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)\n"
        "assert 'diffusers' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ---- protocol / signature conformance ------------------------------------------------------


def test_protocol_conformance():
    adapter = Cosmos3EdgeAdapter()
    assert isinstance(adapter, BackboneAdapter)
    assert isinstance(adapter.name, str) and adapter.name
    assert isinstance(adapter.feature_dim, int) and adapter.feature_dim > 0


def test_signature_conformance():
    adapter = Cosmos3EdgeAdapter()
    for method in PROTOCOL_METHODS:
        proto_params = list(inspect.signature(getattr(BackboneAdapter, method)).parameters)
        impl_params = list(inspect.signature(getattr(type(adapter), method)).parameters)
        assert impl_params == proto_params, method


# ---- config defaults -----------------------------------------------------------------------


def test_config_defaults_match_the_verified_checkpoint_facts():
    cfg = Cosmos3EdgeConfig()
    assert cfg.model_id == "nvidia/Cosmos3-Edge-Policy-DROID"
    assert COSMOS3_EDGE_BASE_MODEL_ID == "nvidia/Cosmos3-Edge"
    assert cfg.checkpoint_path is None
    assert cfg.allow_download is False  # never block on a 9.2 GB pull implicitly
    assert cfg.dtype == "bfloat16"  # the only precision NVIDIA tests
    assert cfg.feature_dim == COSMOS3_EDGE_HIDDEN_DIM == 2048
    assert cfg.domain_name == POLICY_DOMAIN_NAME == "droid_lerobot"
    assert cfg.action_chunk_size == 32
    assert cfg.conditioning_fps == 15.0
    assert cfg.num_inference_steps == 4
    assert cfg.guidance_scale == 3.0
    assert cfg.resolution_tier == 480
    assert cfg.view_point == "ego_view"
    assert cfg.allow_empty_instruction is False
    assert cfg.requires_external_weights is True
    # The two constants that must never be silently retyped as one another.
    assert COSMOS3_EDGE_MAX_ACTION_DIM == 64
    assert COSMOS3_EDGE_NUM_EMBODIMENT_DOMAINS == 32
    assert COSMOS3_EDGE_NUM_LAYERS == 28
    assert G1_DEX3_ACTION_DIM == 28 < COSMOS3_EDGE_MAX_ACTION_DIM


def test_config_has_no_default_raw_action_dim():
    """[?] diffusers says 10 for droid_lerobot, the model card says 8 — so: no default."""
    assert Cosmos3EdgeConfig().raw_action_dim is None


def test_config_is_frozen_and_torch_free():
    cfg = Cosmos3EdgeConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.feature_dim = 1  # type: ignore[misc]
    for value in vars(cfg).values():
        assert value is None or isinstance(value, (str, int, float, bool)), value


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feature_dim": 0},
        {"action_chunk_size": 0},
        {"conditioning_fps": 0.0},
        {"num_inference_steps": 0},
        {"guidance_scale": 0.0},
        {"resolution_tier": 512},
        {"view_point": "drone_view"},
        {"dtype": "fp8"},
        {"domain_name": ""},
        {"raw_action_dim": 0},
        {"raw_action_dim": COSMOS3_EDGE_MAX_ACTION_DIM + 1},
    ],
)
def test_config_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        Cosmos3EdgeConfig(**kwargs)


def test_describe_is_json_primitive_and_reports_unsettled_width():
    described = Cosmos3EdgeAdapter().describe()
    assert described["name"] == "cosmos3-edge"
    assert described["loaded"] is False
    assert described["raw_action_dim"] is None  # [?] stays visibly unsettled in run logs
    assert described["max_action_dim"] == 64
    assert described["source"] == "nvidia/Cosmos3-Edge-Policy-DROID"


# ---- weights-absent error paths -------------------------------------------------------------


def test_load_without_staged_weights_raises_actionable_error():
    adapter = Cosmos3EdgeAdapter()
    with pytest.raises(RuntimeError, match="Cosmos3-Edge weights not available") as excinfo:
        adapter.load()
    message = str(excinfo.value)
    assert "checkpoint_path" in message and "allow_download" in message
    assert not adapter.is_loaded


def test_load_with_missing_checkpoint_path_names_the_path():
    adapter = Cosmos3EdgeAdapter(checkpoint_path="/nonexistent/cosmos3-edge-policy-droid")
    with pytest.raises(RuntimeError, match="Cosmos3-Edge weights not available") as excinfo:
        adapter.load()
    assert "/nonexistent/cosmos3-edge-policy-droid" in str(excinfo.value)


def test_features_requires_weights():
    with pytest.raises(RuntimeError, match="Cosmos3-Edge weights not available"):
        Cosmos3EdgeAdapter().features(None, None, None)


def test_features_refuses_even_when_loaded_because_readout_depth_is_unverified():
    adapter = Cosmos3EdgeAdapter()
    adapter._loaded = True  # simulate staged weights without touching 9.2 GB
    with pytest.raises(NotImplementedError, match="readout depth is unverified"):
        adapter.features(None, None, None)


def test_require_raw_action_dim_refuses_to_guess():
    with pytest.raises(NotImplementedError, match="UNSETTLED") as excinfo:
        Cosmos3EdgeAdapter().require_raw_action_dim()
    message = str(excinfo.value)
    assert "10" in message and "8" in message  # both disputed widths are named
    assert "droid_lerobot" in message
    assert Cosmos3EdgeAdapter(raw_action_dim=10).require_raw_action_dim() == 10


def test_condition_state_refuses_because_there_is_no_state_input():
    with pytest.raises(NotImplementedError, match="no proprioception input"):
        Cosmos3EdgeAdapter().condition_state(np.zeros((1, 32), dtype=np.float32))


# ---- conditioning (weights-free by design) ---------------------------------------------------


def test_condition_video_selects_the_current_frame():
    adapter = Cosmos3EdgeAdapter()
    clip = make_frames((3, 16, 20, 3))
    out = adapter.condition_video(clip)
    assert out["image"].shape == (16, 20, 3)
    assert out["image"].dtype == np.uint8
    np.testing.assert_array_equal(out["image"], clip[-1])  # LAST entry = current observation
    assert out["resolution_tier"] == 480
    assert out["chunk_size"] == 32
    assert out["domain_name"] == "droid_lerobot"

    single = adapter.condition_video(clip[-1])
    np.testing.assert_array_equal(single["image"], clip[-1])
    batched = adapter.condition_video(clip[None])
    np.testing.assert_array_equal(batched["image"], clip[-1])


def test_condition_video_accepts_unit_range_float():
    adapter = Cosmos3EdgeAdapter()
    frames = make_frames((2, 8, 8, 3))
    out = adapter.condition_video(frames.astype(np.float32) / 255.0)
    assert out["image"].dtype == np.uint8
    np.testing.assert_allclose(out["image"], frames[-1], atol=1)


def test_condition_video_rejects_bad_shapes_and_ranges():
    adapter = Cosmos3EdgeAdapter()
    with pytest.raises(ValueError, match="single clip"):
        adapter.condition_video(make_frames((2, 3, 8, 8, 3)))
    with pytest.raises(ValueError, match="3 channels"):
        adapter.condition_video(make_frames((3, 8, 8, 4)))
    with pytest.raises(ValueError, match=r"float video must be in \[0, 1\]"):
        adapter.condition_video(make_frames((2, 8, 8, 3)).astype(np.float32))


def test_condition_text_builds_the_action_caption_fields():
    caption = Cosmos3EdgeAdapter().condition_text(INSTRUCTION)
    assert caption["description"] == INSTRUCTION
    assert caption["view_point"] == "ego_view"
    assert caption["fps"] == 15.0
    assert caption["chunk_size"] == 32
    assert caption["negative_prompt"] == ""  # NVIDIA's tuned choice for all action modes
    assert caption["guidance_scale"] == 3.0


def test_condition_text_rejects_empty_unless_opted_in():
    """E-01: text is structurally required, so '' is a degraded case, not a null path."""
    adapter = Cosmos3EdgeAdapter()
    for empty in ("", "   "):
        with pytest.raises(ValueError, match="empty instruction rejected") as excinfo:
            adapter.condition_text(empty)
        assert "unconditional prior" in str(excinfo.value)
    opted_in = Cosmos3EdgeAdapter(allow_empty_instruction=True)
    assert opted_in.condition_text("")["description"] == ""


def test_condition_text_rejects_non_strings():
    with pytest.raises(TypeError, match="text must be a str"):
        Cosmos3EdgeAdapter().condition_text(None)  # type: ignore[arg-type]
