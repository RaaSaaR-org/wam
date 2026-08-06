"""Tests for :mod:`wam.robot.kinematics` — FK from canonical joints to an EE pose (T-37).

Skips whole-module when the optional ``mujoco`` extra or the fetched Menagerie model is
absent, matching ``tests/test_mujoco_g1.py``.

What these pin is the part that can be wrong *silently*. An FK bug does not raise: it returns
plausible-looking metres, and the action sequence built from them is quietly about a different
robot. So the assertions are geometric facts that a mis-resolved joint, a transposed matrix or a
dropped wrap would break — not shape checks.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="needs the optional wam[sim] extra")

from wam.robot import G1_SPEC
from wam.robot.kinematics import G1Kinematics, ee_action_sequence
from wam.robot.mujoco_transport import DEFAULT_SCENE_PATH, _repo_root

if not (_repo_root() / DEFAULT_SCENE_PATH).is_file():  # pragma: no cover - env dependent
    pytest.skip("scene not fetched: scripts/fetch_g1_model.py", allow_module_level=True)

NUM_JOINTS = len(G1_SPEC.joint_names)
LEFT_ELBOW = list(G1_SPEC.joint_names).index("left_elbow")
RIGHT_ELBOW = list(G1_SPEC.joint_names).index("right_elbow")
LEFT_SHOULDER_PITCH = list(G1_SPEC.joint_names).index("left_shoulder_pitch")


@pytest.fixture(scope="module")
def kin() -> G1Kinematics:
    return G1Kinematics()


def test_the_zero_pose_is_reproducible_and_the_ee_is_where_a_g1_hand_is(kin: G1Kinematics) -> None:
    zero = np.zeros((2, NUM_JOINTS))
    poses = kin.ee_poses(zero)
    assert np.allclose(poses[0], poses[1]), "same q must give the same pose"
    x, y, z = poses[0, :3]
    # Left hand of a standing 1.27 m G1 with arms down: left of centre, roughly hip height.
    assert 0.0 < y < 0.6, f"left hand should be on the +y side, got y={y}"
    assert 0.4 < z < 1.3, f"hand should be between hip and shoulder height, got z={z}"
    assert abs(x) < 0.6, f"hand should be near the sagittal plane at rest, got x={x}"


def test_moving_a_left_arm_joint_moves_the_left_ee_and_a_right_arm_joint_does_not(
    kin: G1Kinematics,
) -> None:
    """The single assertion that a positional joint mapping would fail.

    ``_qpos_adr`` is resolved by name; if it were resolved by MJCF order the canonical index
    for ``left_elbow`` would address some other joint, and this test is how you would find out.
    """
    base = np.zeros((1, NUM_JOINTS))
    left = base.copy()
    left[0, LEFT_ELBOW] = -0.8
    right = base.copy()
    right[0, RIGHT_ELBOW] = -0.8

    p0 = kin.ee_poses(base)[0, :3]
    moved = np.linalg.norm(kin.ee_poses(left)[0, :3] - p0)
    unmoved = np.linalg.norm(kin.ee_poses(right)[0, :3] - p0)
    assert moved > 0.05, f"left elbow must move the left EE, moved {moved:.4f} m"
    assert unmoved == 0.0, f"right elbow must not move the LEFT EE, moved {unmoved:.4f} m"


def test_the_pose_is_continuous_in_the_joint_angle(kin: G1Kinematics) -> None:
    """Halving the joint step must roughly halve the EE displacement.

    A ratio near 2 is what a smooth kinematic chain gives; a mis-set qpos slot (e.g. writing
    into a free-joint quaternion) makes this ratio wander far from 2.
    """
    q = np.zeros((3, NUM_JOINTS))
    q[1, LEFT_SHOULDER_PITCH] = 0.1
    q[2, LEFT_SHOULDER_PITCH] = 0.2
    p = kin.ee_poses(q)[:, :3]
    small = np.linalg.norm(p[1] - p[0])
    large = np.linalg.norm(p[2] - p[0])
    assert small > 1e-4
    assert 1.8 < large / small < 2.2, f"expected ~2x, got {large / small:.3f}"


def test_the_reported_euler_angles_reconstruct_the_scenes_own_rotation_matrix(
    kin: G1Kinematics,
) -> None:
    """The orientation convention, pinned against MuJoCo rather than against itself.

    ``roll/pitch/yaw`` here means ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``. Nothing else in the
    codebase can catch getting this wrong: a transposed matrix yields the *inverse* rotation,
    which is still a perfectly finite, plausible-looking triple of angles, and every shape,
    range and continuity assertion still passes. It would just describe a different robot to
    whichever action port consumed it.
    """
    import mujoco as mj

    model, data = kin._model, kin._data
    body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, kin.ee_body)

    rng = np.random.default_rng(0)
    q = rng.uniform(-0.6, 0.6, size=(5, NUM_JOINTS))
    poses = kin.ee_poses(q)

    def rx(a: float) -> np.ndarray:
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def ry(a: float) -> np.ndarray:
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def rz(a: float) -> np.ndarray:
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    for t in range(q.shape[0]):
        data.qpos[kin._qpos_adr] = q[t]
        mj.mj_kinematics(model, data)
        expected = data.xmat[body].reshape(3, 3).copy()
        roll, pitch, yaw = poses[t, 3:]
        assert np.allclose(rz(yaw) @ ry(pitch) @ rx(roll), expected, atol=1e-9)
        # And the transpose is genuinely a different matrix here, so the check has teeth.
        assert not np.allclose(expected, expected.T, atol=1e-3)


def test_non_finite_joints_are_refused_rather_than_producing_nan_actions(
    kin: G1Kinematics,
) -> None:
    q = np.zeros((2, NUM_JOINTS))
    q[1, LEFT_ELBOW] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        kin.ee_poses(q)


def test_the_wrong_joint_count_is_refused(kin: G1Kinematics) -> None:
    with pytest.raises(ValueError, match=r"\[N, 15\]"):
        kin.ee_poses(np.zeros((2, NUM_JOINTS - 1)))
    with pytest.raises(ValueError, match=r"\[N, 15\]"):
        kin.ee_poses(np.zeros(NUM_JOINTS))


def test_an_action_is_a_displacement_not_a_pose(kin: G1Kinematics) -> None:
    """A held pose must produce zero action. This is what separates delta from absolute — and
    an absolute-pose port fed our deltas (or the reverse) fails silently, not loudly."""
    q = np.tile(np.linspace(0.0, 0.3, NUM_JOINTS), (4, 1))
    actions = ee_action_sequence(q, np.zeros(4), kinematics=kin)
    assert actions.shape == (3, 7)
    assert np.abs(actions[:, :6]).max() == 0.0, "a static arm must command zero displacement"


def test_the_gripper_channel_is_binary_and_reads_the_step_being_entered(
    kin: G1Kinematics,
) -> None:
    q = np.zeros((4, NUM_JOINTS))
    grip = np.array([0.0, 0.9, 0.9, 0.1])
    actions = ee_action_sequence(q, grip, kinematics=kin, grip_threshold=0.5)
    # Three actions for four frames; each carries the gripper at the frame it lands on.
    assert actions[:, 6].tolist() == [1.0, 1.0, 0.0]
    assert set(np.unique(actions[:, 6])) <= {0.0, 1.0}


class _StubKinematics:
    """Returns poses chosen by the test. ``ee_action_sequence`` only ever calls ``ee_poses``,
    so this exercises the differencing and wrapping without needing joint angles that happen
    to put a real G1 wrist on the branch cut."""

    def __init__(self, poses: np.ndarray) -> None:
        self._poses = np.asarray(poses, dtype=np.float64)

    def ee_poses(self, q: np.ndarray) -> np.ndarray:
        assert len(q) == len(self._poses)
        return self._poses


def test_a_yaw_crossing_pi_reads_as_a_small_rotation() -> None:
    """Without the wrap, one frame of a wrist rotation past +-pi reads as a ~2*pi action —
    larger than every real motion in an episode, and it would dominate any ridge fitted on it."""
    poses = np.zeros((2, 6))
    poses[0, 5] = np.pi - 0.1  # yaw just below +pi
    poses[1, 5] = -np.pi + 0.1  # ... continues 0.2 rad the short way, across the branch cut
    actions = ee_action_sequence(
        np.zeros((2, NUM_JOINTS)), np.zeros(2), kinematics=_StubKinematics(poses)
    )
    unwrapped = poses[1, 5] - poses[0, 5]
    assert abs(unwrapped) > 6.0, "precondition: the raw difference is a ~2pi jump"
    assert actions[0, 5] == pytest.approx(0.2, abs=1e-9)


def test_translation_channels_are_not_wrapped() -> None:
    """Only the three angular channels get the modulo. Wrapping a position would silently cap
    every displacement at pi metres, which no assertion on this corpus's millimetre steps
    would ever notice."""
    poses = np.zeros((2, 6))
    poses[1, 0] = 4.0  # > pi, would come back as 4 - 2pi if the wrap were applied to xyz
    actions = ee_action_sequence(
        np.zeros((2, NUM_JOINTS)), np.zeros(2), kinematics=_StubKinematics(poses)
    )
    assert actions[0, 0] == pytest.approx(4.0)


def test_mismatched_gripper_and_joint_lengths_are_refused(kin: G1Kinematics) -> None:
    with pytest.raises(ValueError, match="gripper has"):
        ee_action_sequence(np.zeros((4, NUM_JOINTS)), np.zeros(3), kinematics=kin)


def test_one_frame_cannot_form_a_displacement(kin: G1Kinematics) -> None:
    with pytest.raises(ValueError, match="at least 2 frames"):
        ee_action_sequence(np.zeros((1, NUM_JOINTS)), np.zeros(1), kinematics=kin)


def test_an_unknown_ee_body_fails_at_construction_not_at_first_use() -> None:
    with pytest.raises(ValueError, match="no body named"):
        G1Kinematics(ee_body="left_hand_palm_link")  # not in the Menagerie model
