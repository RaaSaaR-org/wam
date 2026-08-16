"""Tests for ``scripts/t39_policy_shim.py``.

The shim is pure plumbing, which is exactly why it needs tests: every defect it can have —
a permuted joint vector, a dropped key, an off-by-one horizon, legs quietly filled with a
plausible pose — produces a finite, correctly shaped array that scores. None of them raise.

``Gr00tPolicy`` is faked rather than loaded. The real one wants 24 GB of weights and a GPU, and
what is under test here is the mapping between its dict-of-groups and our 43-dim vector, which is
independent of what the network computes.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_shim():
    """Import the shim fresh, so a test that stubs ``gr00t`` cannot leak into the next one."""
    path = _REPO_ROOT / "scripts" / "t39_policy_shim.py"
    spec = importlib.util.spec_from_file_location("_t39_policy_shim_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The AppleToPlate 43-dim layout, restated here as the test's own fixture so that a change to
# convert_lerobot_g1's constants shows up as a failure rather than being silently agreed with.
STATE_GROUPS = [
    ("left_leg", 6),
    ("right_leg", 6),
    ("waist", 3),
    ("left_arm", 7),
    ("right_arm", 7),
    ("left_hand", 7),
    ("right_hand", 7),
]
ACTION_KEYS = [
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "waist",
    "navigate_command",
    "base_height_command",
    "effort_left_arm",
    "effort_right_arm",
    "effort_left_hand",
    "effort_right_hand",
    "effort_waist",
]
ACTION_DIMS = {"navigate_command": 3, "base_height_command": 1}
HORIZON = 16


class _Dim:
    def __init__(self, value: int) -> None:
        self._value = value

    def item(self) -> int:
        return self._value


class _ModalityConfig:
    def __init__(self, delta_indices, modality_keys):
        self.delta_indices = delta_indices
        self.modality_keys = modality_keys


class _FakePolicy:
    """Stands in for ``Gr00tPolicy``, recording what the shim hands it."""

    instances: list["_FakePolicy"] = []

    def __init__(
        self,
        tag,
        model_path,
        *,
        device,
        strict=True,
        state_groups=None,
        video_delta_indices=None,
    ):
        self.tag = tag
        self.model_path = model_path
        self.device = device
        self.strict = strict
        self.last_observation = None
        self.embodiment_tag = types.SimpleNamespace(value=tag, name=tag.upper())
        self.language_key = "annotation.human.task_description"

        groups = state_groups if state_groups is not None else STATE_GROUPS
        self.modality_configs = {
            "video": _ModalityConfig(
                [0] if video_delta_indices is None else list(video_delta_indices), ["ego_view"]
            ),
            "state": _ModalityConfig([0], [name for name, _ in groups]),
            "action": _ModalityConfig(list(range(HORIZON)), list(ACTION_KEYS)),
        }
        norm = {tag: {"state": {name: {"dim": _Dim(dim)} for name, dim in groups}}}
        self.processor = types.SimpleNamespace(
            state_action_processor=types.SimpleNamespace(norm_params=norm)
        )
        self._dims = dict(groups)
        _FakePolicy.instances.append(self)

    def get_action(self, observation):
        self.last_observation = observation
        action = {}
        for index, key in enumerate(ACTION_KEYS):
            dim = ACTION_DIMS.get(key, self._dims.get(key.replace("effort_", ""), 7))
            # A distinct constant per key, so a mis-placed slice is visible in the output.
            action[key] = np.full((1, HORIZON, dim), float(index + 1), dtype=np.float32)
        return action, {}


@pytest.fixture(autouse=True)
def _reset_instances():
    _FakePolicy.instances = []
    yield
    _FakePolicy.instances = []


@pytest.fixture
def stub_gr00t(monkeypatch):
    """Install a fake ``gr00t.policy.gr00t_policy`` module, parameterised per test."""

    def install(**policy_kwargs):
        module = types.ModuleType("gr00t.policy.gr00t_policy")

        def factory(tag, model_path, *, device, strict=True):
            return _FakePolicy(tag, model_path, device=device, strict=strict, **policy_kwargs)

        module.Gr00tPolicy = factory
        pkg = types.ModuleType("gr00t")
        policy_pkg = types.ModuleType("gr00t.policy")
        monkeypatch.setitem(sys.modules, "gr00t", pkg)
        monkeypatch.setitem(sys.modules, "gr00t.policy", policy_pkg)
        monkeypatch.setitem(sys.modules, "gr00t.policy.gr00t_policy", module)

    return install


def _checkpoint(tmp_path: Path, tags=("new_embodiment",)) -> Path:
    root = tmp_path / "checkpoint-10000"
    (root / "experiment_cfg").mkdir(parents=True)
    conf = {"data": {"datasets": [{"embodiment_tag": tag, "mix_ratio": 1.0} for tag in tags]}}
    (root / "experiment_cfg" / "conf.yaml").write_text(yaml.safe_dump(conf))
    return root


def _observation(frames=1, height=8, width=10):
    return {
        "video": np.zeros((frames, height, width, 3), dtype=np.uint8),
        "state": np.arange(43, dtype=np.float32),
        "instruction": "pick up the apple and put it on the plate",
    }


# ------------------------------------------------------------------------------ tag resolution


def test_tag_is_read_from_the_checkpoints_own_conf(tmp_path, stub_gr00t):
    stub_gr00t()
    shim = _load_shim()
    shim.build_policy(_checkpoint(tmp_path), "cpu")
    assert _FakePolicy.instances[0].tag == "new_embodiment"


def test_env_override_wins_over_the_conf(tmp_path, stub_gr00t, monkeypatch):
    stub_gr00t()
    shim = _load_shim()
    monkeypatch.setenv(shim.TAG_ENV_VAR, "unitree_g1")
    shim.build_policy(_checkpoint(tmp_path), "cpu")
    assert _FakePolicy.instances[0].tag == "unitree_g1"


def test_missing_conf_refuses_rather_than_guessing(tmp_path, stub_gr00t):
    stub_gr00t()
    shim = _load_shim()
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(SystemExit, match="cannot tell which embodiment slot"):
        shim.build_policy(bare, "cpu")


def test_two_tags_in_one_conf_refuses(tmp_path, stub_gr00t):
    stub_gr00t()
    shim = _load_shim()
    root = _checkpoint(tmp_path, tags=("new_embodiment", "unitree_g1"))
    with pytest.raises(SystemExit, match="embodiment tag"):
        shim.build_policy(root, "cpu")


# --------------------------------------------------------------------------- layout cross-check


def test_permuted_state_keys_are_refused(tmp_path, stub_gr00t):
    """The arms swapped with the hands still totals 43 and would score."""
    permuted = [
        ("left_leg", 6),
        ("right_leg", 6),
        ("waist", 3),
        ("left_hand", 7),
        ("right_hand", 7),
        ("left_arm", 7),
        ("right_arm", 7),
    ]
    stub_gr00t(state_groups=permuted)
    shim = _load_shim()
    with pytest.raises(SystemExit, match="would be permuted"):
        shim.build_policy(_checkpoint(tmp_path), "cpu")


def test_wrong_total_dim_is_refused(tmp_path, stub_gr00t):
    stub_gr00t(state_groups=[("left_arm", 7), ("right_arm", 7)])
    shim = _load_shim()
    with pytest.raises(SystemExit, match="was not trained on this corpus"):
        shim.build_policy(_checkpoint(tmp_path), "cpu")


# ---------------------------------------------------------------------------------- inference


def test_output_shape_and_dtype(tmp_path, stub_gr00t):
    stub_gr00t()
    shim = _load_shim()
    infer = shim.build_policy(_checkpoint(tmp_path), "cpu")
    out = infer(_observation())
    assert out.shape == (HORIZON, 43)
    assert out.dtype == np.float32


def test_legs_are_nan_and_everything_else_is_finite(tmp_path, stub_gr00t):
    stub_gr00t()
    shim = _load_shim()
    infer = shim.build_policy(_checkpoint(tmp_path), "cpu")
    out = infer(_observation())
    assert np.isnan(out[:, 0:12]).all(), "legs are not predicted and must not look like they are"
    assert np.isfinite(out[:, 12:43]).all()


def test_each_action_key_lands_in_its_own_slice(tmp_path, stub_gr00t):
    """The fake returns a distinct constant per key, so a swap is visible in the values."""
    stub_gr00t()
    shim = _load_shim()
    infer = shim.build_policy(_checkpoint(tmp_path), "cpu")
    out = infer(_observation())
    expected = {key: float(index + 1) for index, key in enumerate(ACTION_KEYS)}
    assert (out[:, 12:15] == expected["waist"]).all()
    assert (out[:, 15:22] == expected["left_arm"]).all()
    assert (out[:, 22:29] == expected["right_arm"]).all()
    assert (out[:, 29:36] == expected["left_hand"]).all()
    assert (out[:, 36:43] == expected["right_hand"]).all()


def test_non_joint_action_keys_are_dropped_not_folded_in(tmp_path, stub_gr00t):
    """navigate_command / base_height_command / efforts have no column and must not take one."""
    stub_gr00t()
    shim = _load_shim()
    infer = shim.build_policy(_checkpoint(tmp_path), "cpu")
    out = infer(_observation())
    dropped = {
        float(ACTION_KEYS.index(key) + 1)
        for key in ACTION_KEYS
        if key.startswith("effort_") or key.endswith("_command")
    }
    present = set(np.unique(out[:, 12:43]).tolist())
    assert not (present & dropped), f"a dropped key's value reached the vector: {present & dropped}"


def test_state_is_split_into_the_groups_the_policy_asked_for(tmp_path, stub_gr00t):
    stub_gr00t()
    shim = _load_shim()
    infer = shim.build_policy(_checkpoint(tmp_path), "cpu")
    infer(_observation())
    handed = _FakePolicy.instances[0].last_observation
    assert set(handed["state"]) == {name for name, _ in STATE_GROUPS}
    # arange(43) means each group's values are its own source indices.
    assert handed["state"]["left_arm"].shape == (1, 1, 7)
    np.testing.assert_array_equal(handed["state"]["left_arm"][0, 0], np.arange(15, 22))
    np.testing.assert_array_equal(handed["state"]["right_hand"][0, 0], np.arange(36, 43))


def test_video_is_batched_and_trimmed_to_the_trained_window(tmp_path, stub_gr00t):
    stub_gr00t()
    shim = _load_shim()
    infer = shim.build_policy(_checkpoint(tmp_path), "cpu")
    infer(_observation(frames=4))
    handed = _FakePolicy.instances[0].last_observation
    assert handed["video"]["ego_view"].shape == (1, 1, 8, 10, 3)


def test_video_is_tiled_when_the_eval_passes_a_single_frame(tmp_path, stub_gr00t):
    """The eval's own wording for the no --frame-history case is 'tiled by the policy'.

    A checkpoint that wants a two-frame window must still be scorable when the eval hands over one
    frame, and the tile must go at the FRONT: the delta indices are non-positive offsets from now,
    so the frame the eval passed is the current one and belongs last.
    """
    stub_gr00t(video_delta_indices=[-1, 0])
    shim = _load_shim()
    infer = shim.build_policy(_checkpoint(tmp_path), "cpu")
    obs = _observation(frames=1)
    obs["video"][:] = 7
    out = infer(obs)
    handed = _FakePolicy.instances[0].last_observation
    assert handed["video"]["ego_view"].shape == (1, 2, 8, 10, 3)
    assert (handed["video"]["ego_view"] == 7).all()
    assert out.shape == (HORIZON, 43)


def test_video_window_keeps_the_most_recent_frame_last(tmp_path, stub_gr00t):
    """Trimming a longer history must drop the OLDEST frames, never the current one."""
    stub_gr00t(video_delta_indices=[-1, 0])
    shim = _load_shim()
    infer = shim.build_policy(_checkpoint(tmp_path), "cpu")
    obs = _observation(frames=4)
    for index in range(4):
        obs["video"][index] = index
    infer(obs)
    handed = _FakePolicy.instances[0].last_observation["video"]["ego_view"]
    assert handed.shape == (1, 2, 8, 10, 3)
    assert (handed[0, 0] == 2).all() and (handed[0, 1] == 3).all()


def test_instruction_is_passed_through_under_the_language_key(tmp_path, stub_gr00t):
    stub_gr00t()
    shim = _load_shim()
    infer = shim.build_policy(_checkpoint(tmp_path), "cpu")
    obs = _observation()
    infer(obs)
    handed = _FakePolicy.instances[0].last_observation
    assert handed["language"]["annotation.human.task_description"] == [[obs["instruction"]]]


def test_wrong_state_dim_is_refused(tmp_path, stub_gr00t):
    stub_gr00t()
    shim = _load_shim()
    infer = shim.build_policy(_checkpoint(tmp_path), "cpu")
    obs = _observation()
    obs["state"] = np.zeros(15, dtype=np.float32)
    with pytest.raises(SystemExit, match="the contract is 43"):
        infer(obs)


def test_float_video_is_refused(tmp_path, stub_gr00t):
    stub_gr00t()
    shim = _load_shim()
    infer = shim.build_policy(_checkpoint(tmp_path), "cpu")
    obs = _observation()
    obs["video"] = obs["video"].astype(np.float32)
    with pytest.raises(SystemExit, match="must be uint8"):
        infer(obs)


def test_no_arithmetic_on_joint_values(tmp_path, stub_gr00t):
    """PR-07 §8 item 6: the shim adapts shapes, it does not convert units.

    The fake returns a known constant per key; if the shim added an anchor, applied a scale or
    re-relativised anything, the constant would not survive to the output.
    """
    stub_gr00t()
    shim = _load_shim()
    infer = shim.build_policy(_checkpoint(tmp_path), "cpu")
    out = infer(_observation())
    assert (out[:, 15:22] == float(ACTION_KEYS.index("left_arm") + 1)).all()
