#!/usr/bin/env python3
"""``POLICY_ENTRYPOINT`` shim — NVIDIA's vendored ``Gr00tPolicy`` behind our eval's contract.

PR-07 §8 item 6 sanctions exactly this file and fixes its limits: *"a shim inside the t39 venv may
adapt to it, but may not convert into canonical units — that happens once, in our code, shared with
the oracle."* So everything here is shape, key order and dtype. There is no arithmetic on joint
values anywhere below, and that is the property to preserve if this file is ever edited.

The contract, restated from :func:`eval_t39_baseline.load_commanded_policy`::

    build_policy(model_dir: Path, device: str) -> callable
    callable({"video": uint8 [T, H, W, 3], "state": float32 [43], "instruction": str})
        -> float32 [horizon, 43]   ABSOLUTE positions in the source 43-dim layout

Use it as::

    POLICY_ENTRYPOINT=t39_policy_shim:build_policy
    MODEL_DIR=$PROJ/runs/t39-baseline-seed0/checkpoints/checkpoint-10000

WHY THE VENDORED CALL ALREADY RETURNS ABSOLUTE POSITIONS, which is the one claim this file rests
on and the one worth checking against the source rather than believing. Our action config marks
``left_arm``/``right_arm`` RELATIVE and the other ten keys ABSOLUTE, so a naive reading is that the
arm keys come back as deltas and the shim owes them an addition. It does not.
``Gr00tPolicy._get_action`` passes the observation's own states into
``processor.decode_action(..., batched_states)``, whose docstring is *"Undo action normalization
and convert relative actions to absolute"*, and which delegates to
``StateActionProcessor.unapply_action`` — step 2 of that function walks ``action_configs`` and, for
every key whose ``rep`` is ``RELATIVE``, adds the reference state back. By the time the dict reaches
us it is absolute in source units for every key. Adding an anchor here would apply the conversion
twice, and the result would be finite, correctly shaped and wrong — the failure mode PR-07 §3 keeps
naming.

THE LEGS ARE NOT PREDICTED, AND ARE RETURNED AS NaN ON PURPOSE. The 43-dim source layout is
``legs [0:12] waist [12:15] arms [15:29] hands [29:43]``, but the trained action space has no leg
key at all — ``configs/groot/new_embodiment_config_defaults.py`` says so in as many words ("the
legs are observed but never commanded"). Twelve dims of the returned vector are therefore not
predictions, and something has to go in them. NaN rather than a plausible hold-the-current-pose
fill, because the eval provably never reads those columns (``convert_lerobot_g1.canonical_q``
takes ``WAIST_YAW`` plus the two arm slices; the gripper channel takes the two hand slices), so
today the choice is inert either way — and on the day someone extends the canonical mapping to
include legs, NaN turns this into an immediate, unmissable failure while a hold-fill would quietly
score twelve invented columns as if the policy had produced them.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent

TAG_ENV_VAR = "T39_EMBODIMENT_TAG"
"""Override for the tag read out of the checkpoint. Not a default — see :func:`_embodiment_tag`."""


def _load_convert() -> Any:
    """Import ``scripts/convert_lerobot_g1.py`` for its layout constants.

    Reused rather than restated. The 43-dim slices exist in exactly one place in this repo and a
    second copy here would be free to drift from the one that actually built the dataset — the
    same reason ``eval_t39_baseline._load_script`` exists. This runs inside the eval's process, so
    ``src/`` is already on ``sys.path`` and the module's ``wam.*`` imports resolve.
    """
    path = _REPO_ROOT / "scripts" / "convert_lerobot_g1.py"
    spec = importlib.util.spec_from_file_location("_t39_shim_convert", path)
    if spec is None or spec.loader is None:  # pragma: no cover - unreachable for a real file
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _embodiment_tag(model_dir: Path) -> str:
    """The tag this checkpoint was post-trained under, read out of the checkpoint.

    Read rather than passed, because the contract's factory signature has no room for it and a
    constant here could disagree with what 70_train_t39_baseline.sbatch actually trained. The
    checkpoint records it in ``experiment_cfg/conf.yaml`` under ``data.datasets[*].embodiment_tag``
    — the trainer's own account of the run — so this cannot drift from the weights it describes.

    ``statistics.json`` in the same directory carries the whole 32-way bank, so "the checkpoint has
    exactly one embodiment" is not a question that can be asked of it; the conf is the only file
    that says which slot was tuned.
    """
    override = os.environ.get(TAG_ENV_VAR)
    if override:
        return override

    import yaml

    conf_path = model_dir / "experiment_cfg" / "conf.yaml"
    if not conf_path.is_file():
        raise SystemExit(
            f"{conf_path} missing — cannot tell which embodiment slot {model_dir} was trained "
            f"into. Set {TAG_ENV_VAR} if you know it from the run's own log, but check it: the "
            "base model has untrained random weights in every slot, and scoring one of those "
            "produces numbers rather than an error."
        )
    conf = yaml.safe_load(conf_path.read_text())
    tags = {
        dataset["embodiment_tag"]
        for dataset in conf.get("data", {}).get("datasets", [])
        if dataset.get("embodiment_tag")
    }
    if len(tags) != 1:
        raise SystemExit(
            f"{conf_path} records {sorted(tags) or 'no'} embodiment tag(s); this shim scores one "
            f"policy over one embodiment. Set {TAG_ENV_VAR} to disambiguate."
        )
    return tags.pop()


def _state_slices(policy: Any, tag_value: str, convert: Any) -> dict[str, slice]:
    """``{state_key: slice}`` over the 43-dim source vector, derived from the checkpoint.

    Derived, never hardcoded: the source vector is the concatenation of the state groups in the
    order ``modality.json`` lists them, which is what the processor's ``norm_params`` records, and
    which is how ``processing_gr00t_n1d7.decode_action`` slices the action side. Taking the order
    from anywhere else would let a key-order mismatch through, and a permuted joint vector is the
    canonical example of an error that returns finite, plausible, wrong numbers.

    The cross-check against ``convert_lerobot_g1`` is the point of the function. Both sides are
    independently derived — one from the checkpoint, one from the constants the dataset was built
    with — so agreement is evidence and disagreement is a stop.
    """
    processor = policy.processor.state_action_processor
    keys = list(policy.modality_configs["state"].modality_keys)

    slices: dict[str, slice] = {}
    start = 0
    for key in keys:
        dim = int(processor.norm_params[tag_value]["state"][key]["dim"].item())
        slices[key] = slice(start, start + dim)
        start += dim

    if start != convert.SOURCE_STATE_DIM:
        raise SystemExit(
            f"the checkpoint's state groups {keys} total {start} dims, but the source layout is "
            f"{convert.SOURCE_STATE_DIM}. This checkpoint was not trained on this corpus."
        )

    expected = {
        "left_arm": convert.LEFT_ARM,
        "right_arm": convert.RIGHT_ARM,
        "left_hand": convert.LEFT_HAND,
        "right_hand": convert.RIGHT_HAND,
    }
    for key, want in expected.items():
        got = slices.get(key)
        if got is None or (got.start, got.stop) != (want.start, want.stop):
            raise SystemExit(
                f"state key {key!r} sits at {got} in the checkpoint but at {want} in "
                f"convert_lerobot_g1. The joint vector would be permuted, and every metric would "
                "still be finite. Refusing."
            )
    waist = slices.get("waist")
    if waist is None or not (waist.start <= convert.WAIST_YAW < waist.stop):
        raise SystemExit(
            f"waist slice {waist} does not contain WAIST_YAW={convert.WAIST_YAW}; the canonical "
            "mapping reads that column and would be reading someone else's joint."
        )
    return slices


def _one_video_key(policy: Any) -> str:
    keys = list(policy.modality_configs["video"].modality_keys)
    if len(keys) != 1:
        raise SystemExit(
            f"this checkpoint expects {len(keys)} camera(s) ({keys}); the eval hands the shim one "
            "camera's frames and cannot say which of several they are."
        )
    return keys[0]


def build_policy(model_dir: Path | str, device: str) -> Any:
    """Build the vendored policy and return the callable the eval's contract asks for."""
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    model_dir = Path(model_dir)
    convert = _load_convert()
    tag = _embodiment_tag(model_dir)

    policy = Gr00tPolicy(tag, str(model_dir), device=device, strict=True)
    tag_value = policy.embodiment_tag.value

    state_slices = _state_slices(policy, tag_value, convert)
    video_key = _one_video_key(policy)
    language_key = policy.language_key
    video_steps = len(policy.modality_configs["video"].delta_indices)
    state_steps = len(policy.modality_configs["state"].delta_indices)
    action_keys = list(policy.modality_configs["action"].modality_keys)

    # The action keys that have a home in the 43-dim source layout. The other seven the config
    # trains — navigate_command, base_height_command and the five efforts — are real outputs with
    # no column here: efforts are torques and the two commands are base-level, none of which the
    # 43-dim joint-position vector represents. They are dropped, not folded in anywhere.
    commanded_keys = [key for key in action_keys if key in state_slices]
    if not commanded_keys:
        raise SystemExit(
            f"none of the checkpoint's action keys {action_keys} name a state group "
            f"{sorted(state_slices)}; nothing could be written into the source vector."
        )

    print(
        f"[t39_policy_shim] {model_dir}\n"
        f"[t39_policy_shim]   embodiment {tag} -> {tag_value}, device {device}\n"
        f"[t39_policy_shim]   camera {video_key} x{video_steps}, state x{state_steps}\n"
        f"[t39_policy_shim]   writing {commanded_keys} into the 43-dim vector\n"
        f"[t39_policy_shim]   dropping {[k for k in action_keys if k not in state_slices]} "
        f"(no column in the source layout)\n"
        f"[t39_policy_shim]   legs return NaN — never predicted, never read",
        flush=True,
    )

    def infer(observation: dict[str, Any]) -> np.ndarray:
        video = np.asarray(observation["video"])
        state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
        instruction = observation["instruction"]

        if state.shape[0] != convert.SOURCE_STATE_DIM:
            raise SystemExit(
                f"the eval handed the shim a {state.shape[0]}-dim state; the contract is "
                f"{convert.SOURCE_STATE_DIM}."
            )
        if video.ndim != 4 or video.shape[-1] != 3:
            raise SystemExit(f"video must be [T, H, W, 3], got {video.shape}")
        if video.dtype != np.uint8:
            raise SystemExit(
                f"video must be uint8 (the vendored check_observation enforces it), got "
                f"{video.dtype}"
            )

        # Trim or tile to the horizon the checkpoint was trained with. Taking the LAST frames
        # because delta_indices are non-positive offsets from now; repeating the EARLIEST frame
        # when short, which is the eval's own "tiled by the policy" wording for the case where it
        # passes a single frame and the model wants a window.
        if video.shape[0] > video_steps:
            video = video[-video_steps:]
        elif video.shape[0] < video_steps:
            pad = np.repeat(video[:1], video_steps - video.shape[0], axis=0)
            video = np.concatenate([pad, video], axis=0)

        batched_state = {
            key: np.repeat(state[None, None, sl], state_steps, axis=1).astype(np.float32)
            for key, sl in state_slices.items()
        }
        vendored_observation = {
            "video": {video_key: video[None, ...]},
            "state": batched_state,
            "language": {language_key: [[instruction]] * 1},
        }

        action, _info = policy.get_action(vendored_observation)

        horizon = int(action[commanded_keys[0]].shape[1])
        commanded = np.full((horizon, convert.SOURCE_STATE_DIM), np.nan, dtype=np.float32)
        for key in commanded_keys:
            values = np.asarray(action[key], dtype=np.float32)
            if values.shape[0] != 1:
                raise SystemExit(f"action {key!r} came back batched {values.shape}, expected B=1")
            commanded[:, state_slices[key]] = values[0, :horizon, :]
        return commanded

    return infer
