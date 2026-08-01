"""`scripts/run_ablation.py` — the T-18 AC-07 driver.

The point of this file is one deletion. `JointPolicy` used to be a third, hand-written copy of
the world-action predict path: its own frame tiling, its own `forward_flow` call, its own
pooling. Both real `predict()` implementations moved to `resolve_frame_context` in T-29; this
copy did not, because nothing imported it and nothing tested it — and it is the copy that
produced the archived `t18-real-ablation-seed0` number (skill_vs_repeat_pct −129.0 %).

So the "single definition of the frame window" claim was false for exactly the script whose
output the AC-07 verdict rests on. The fix was to delete the copy and delegate to
`JointWorldActionModel.predict`. These tests pin the two facts that made that deletion safe,
so a future reader does not have to re-derive them from the backbone internals:

1. the old expression and the new one agree bit-for-bit on the tiny backbone, so no archived
   number moves;
2. the delegation actually reaches `resolve_frame_context`, i.e. `--frame-history` now works
   here at all — which it never did before.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from test_training import (
    GRIPPER_DIMS,
    IMAGE_HW,
    NUM_FRAMES,
    NUM_JOINTS,
    joint_config,
)

from wam.interfaces import IMUState, Observation, RobotState
from wam.training.joint import JointWorldActionModel


def _model() -> JointWorldActionModel:
    torch.manual_seed(0)
    model = JointWorldActionModel(joint_config())
    model.eval()
    return model


def _state() -> RobotState:
    rng = np.random.default_rng(3)
    return RobotState(
        timestamp_ns=1_000,
        q=rng.standard_normal(NUM_JOINTS).astype(np.float32),
        dq=rng.standard_normal(NUM_JOINTS).astype(np.float32),
        imu=IMUState(
            orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
            angular_velocity=np.zeros(3, dtype=np.float32),
            linear_acceleration=np.zeros(3, dtype=np.float32),
        ),
        gripper_state=rng.random(GRIPPER_DIMS).astype(np.float32),
    )


def _frames(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (n, IMAGE_HW, IMAGE_HW, 3), dtype=np.uint8)


def _observation(*, history: bool) -> Observation:
    camera = joint_config().camera
    window = _frames(NUM_FRAMES, seed=7)
    return Observation(
        images={camera: window[-1]},
        state=_state(),
        instruction="pick the red cube",
        image_history={camera: window} if history else None,
    )


def _legacy_predict(model: JointWorldActionModel, observation: Observation):
    """The exact expression `JointPolicy.predict` carried before it was deleted.

    Kept verbatim on purpose: it is the definition of what every archived world-action number
    was produced by, and the only way to show the deletion changed nothing is to still be able
    to evaluate it.
    """
    cfg = model.config
    camera = cfg.camera
    image = torch.as_tensor(observation.images[camera])
    frames = image.unsqueeze(0).expand(cfg.backbone.num_frames, -1, -1, -1)
    state_emb = model.state_encoder.encode(observation.state)
    text_ctx = model.backbone.condition_text(observation.instruction)
    state_ctx = model.backbone.condition_state(state_emb)
    t = torch.ones(1)
    _, feats = model.backbone.forward_flow(frames.unsqueeze(0), t, text_ctx, state_ctx)
    pooled = feats.mean(dim=1)
    return model.action_head.decode(pooled[0])


def test_deleting_the_third_predict_copy_moves_no_archived_number() -> None:
    """The deleted hand-written path and `model.predict` agree bit-for-bit.

    Two differences had to cancel for this to hold, and both were verified in the backbone
    rather than assumed: `forward_flow` normalizes uint8 through `_to_video_tensor` itself, so
    skipping `encode_video` was a no-op on the tiny backbone (it is NOT on a real VAE, which is
    the other reason the copy had to go), and `ActionHead.decode` mean-pools leading dims
    internally, so `decode(features[0])` is the same reduction as `decode(feats.mean(1)[0])`.
    """
    from run_ablation import JointPolicy

    model = _model()
    obs = _observation(history=False)

    with torch.no_grad():
        legacy = _legacy_predict(model, obs)
    current = JointPolicy(model).predict(obs)

    np.testing.assert_array_equal(current.targets, legacy.targets)
    np.testing.assert_array_equal(current.gripper_target, legacy.gripper_target)
    assert current.mode == legacy.mode
    assert current.dt_s == legacy.dt_s


def test_the_delegation_actually_reaches_the_frame_window() -> None:
    """With a history present the policy must produce a different chunk than the tiled path.

    This is the capability the old copy did not have at all: it read `observation.images` and
    nothing else, so `--frame-history` would have been silently ignored had anyone passed it.
    A weaker assertion (that it merely runs) would pass against the deleted code too.
    """
    from run_ablation import JointPolicy

    model = _model()
    policy = JointPolicy(model)

    tiled = policy.predict(_observation(history=False))
    windowed = policy.predict(_observation(history=True))

    assert not np.array_equal(tiled.targets, windowed.targets), (
        "a real frame window produced the same chunk as one frame tiled — the history is "
        "being dropped somewhere between build_eval_pairs and the backbone"
    )


def test_the_tiled_default_is_still_the_default() -> None:
    """No history on the observation must reproduce the tiled expression, not raise or drift.

    `run_ablation.py` defaults to tiled precisely so re-running it reproduces the archived
    t18-real-ablation-seed0 number instead of quietly redefining it.
    """
    from run_ablation import JointPolicy

    model = _model()
    obs = _observation(history=False)

    with torch.no_grad():
        legacy = _legacy_predict(model, obs)

    np.testing.assert_array_equal(JointPolicy(model).predict(obs).targets, legacy.targets)
