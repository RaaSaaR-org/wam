"""Forward kinematics for the canonical G1 arm: joint angles -> end-effector pose.

Why this exists: every action-conditioned video prior published so far (Cosmos-Predict2's
Bridge post-training, IRASim, the RoboCasa/LIBERO policies) takes actions as **end-effector
displacement + gripper**, not joint deltas. Our canonical action space is joint-delta (OD-02)
and our recorded corpus is joint-space, so feeding any of those ports the truth about our robot
needs FK. That is all this module does.

**It is not a second robot backend.** No physics, no stepping, no contacts, no actuators — the
model is loaded once and only ``mj_kinematics`` runs, which is a pure function of ``qpos``. It
shares ``configs/sim/g1_scene.xml`` with :mod:`wam.robot.mujoco_transport` deliberately: the
same MJCF that the closed loop runs on is the one that defines where the hand is, so an FK
number here and a rendered frame there cannot disagree about the arm's geometry.

**Frame.** The scene welds the pelvis, so the world frame and the robot base frame differ by a
fixed transform and *displacements are identical in both*. Absolute positions are therefore
reported in world coordinates and are only meaningful as differences — which is what every
action port consumes anyway.

**The 14 joints this does not set.** ``canonical_q`` is 15 DoF (waist yaw + two 7-DoF arms);
the scene has 29 body joints. The legs and the remaining waist axes stay at the model's zero
pose. For a welded-base upper-body reach that is exactly right; for a walking robot it would
not be, and this module would need the floating base.

Requires the optional ``mujoco`` extra (``wam[sim]``) and the fetched Menagerie model
(``scripts/fetch_g1_model.py``), same as the sim backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from wam.robot.g1 import G1_SPEC

#: MJCF body whose frame is taken as the end effector. The Dex3-1 has no palm body in the
#: Menagerie model, so the last wrist link — the hand's root, the body every finger hangs off —
#: is the closest thing to a tool frame that the asset actually defines.
DEFAULT_EE_BODY = "left_wrist_yaw_link"

#: Canonical joint name -> MJCF name. Same suffix rule as ``mujoco_transport``; duplicated as a
#: constant rather than imported so this module does not depend on the transport.
MJCF_JOINT_SUFFIX = "_joint"


def _load_model(scene_path: str | Path | None) -> Any:
    from wam.robot.mujoco_transport import DEFAULT_SCENE_PATH, _repo_root, _require_mujoco

    mj = _require_mujoco()
    path = Path(scene_path) if scene_path is not None else DEFAULT_SCENE_PATH
    if not path.is_absolute():
        path = _repo_root() / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"scene_path: no such MJCF file: {path}")
    return mj, mj.MjModel.from_xml_path(str(path))


def _mat_to_rpy(mat: np.ndarray) -> np.ndarray:
    """3x3 rotation -> (roll, pitch, yaw) radians, with ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

    That composition is EXTRINSIC XYZ, equivalently intrinsic ZYX — spelled out because the two
    are one capital letter apart in the library a consumer will reach for and are not the same
    rotation: ``scipy...Rotation.from_euler("xyz", rpy)`` reconstructs this, ``"XYZ"`` does not.
    The same statement, as an assertion against MuJoCo's own matrix, is
    ``test_the_reported_euler_angles_reconstruct_the_scenes_own_rotation_matrix``.

    Bridge-style action ports quote orientation as roll/pitch/yaw, so that is what this returns.
    Euler angles are a poor rotation representation in general — they are discontinuous at
    pitch = +-pi/2 — but the consumer is a port that was trained on them, and a gimbal-lock
    artefact in the input is preferable to silently feeding a different convention. Callers that
    want a well-behaved orientation should read the matrix.
    """
    sy = float(-mat[2, 0])
    sy = min(1.0, max(-1.0, sy))
    pitch = float(np.arcsin(sy))
    if abs(sy) < 1.0 - 1e-6:
        roll = float(np.arctan2(mat[2, 1], mat[2, 2]))
        yaw = float(np.arctan2(mat[1, 0], mat[0, 0]))
    else:  # gimbal lock: roll and yaw are not separable, fold everything into yaw
        roll = 0.0
        yaw = float(np.arctan2(-mat[0, 1], mat[1, 1]))
    return np.array([roll, pitch, yaw], dtype=np.float64)


class G1Kinematics:
    """FK over the 15 canonical joints. Construct once, call :meth:`ee_poses` per episode."""

    def __init__(
        self,
        scene_path: str | Path | None = None,
        ee_body: str = DEFAULT_EE_BODY,
    ) -> None:
        mj, model = _load_model(scene_path)
        self._mj, self._model = mj, model
        self._data = mj.MjData(model)

        names = [f"{n}{MJCF_JOINT_SUFFIX}" for n in G1_SPEC.joint_names]
        ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n) for n in names]
        missing = [n for n, i in zip(names, ids, strict=True) if i < 0]
        if missing:
            raise ValueError(f"scene is missing canonical joints: {missing}")
        # Resolved BY NAME, like the transport — MJCF ordering is not the canonical ordering and
        # a positional assumption here would silently drive the wrong joints.
        self._qpos_adr = model.jnt_qposadr[np.asarray(ids)].astype(np.int64)

        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ee_body)
        if body_id < 0:
            raise ValueError(f"scene has no body named {ee_body!r}")
        self._body_id = int(body_id)
        self.ee_body = ee_body

    @property
    def num_joints(self) -> int:
        return len(self._qpos_adr)

    def ee_poses(self, q: np.ndarray) -> np.ndarray:
        """``[N, 15]`` canonical joint angles -> ``[N, 6]`` ``(x, y, z, roll, pitch, yaw)``.

        Non-finite input is refused rather than propagated: ``mj_kinematics`` will happily
        produce NaN poses, and a NaN action silently poisons everything downstream of it.
        """
        q = np.asarray(q, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != self.num_joints:
            raise ValueError(f"q must be [N, {self.num_joints}], got {q.shape}")
        if not np.isfinite(q).all():
            raise ValueError("q contains non-finite values")

        out = np.empty((q.shape[0], 6), dtype=np.float64)
        for t in range(q.shape[0]):
            self._data.qpos[self._qpos_adr] = q[t]
            self._mj.mj_kinematics(self._model, self._data)
            out[t, :3] = self._data.xpos[self._body_id]
            out[t, 3:] = _mat_to_rpy(self._data.xmat[self._body_id].reshape(3, 3))
        return out


def ee_action_sequence(
    q: np.ndarray,
    gripper: np.ndarray,
    *,
    kinematics: G1Kinematics | None = None,
    scene_path: str | Path | None = None,
    grip_threshold: float = 0.5,
) -> np.ndarray:
    """``[N, 15]`` joints + ``[N]`` grasp synergy -> ``[N-1, 7]`` Bridge-shaped actions.

    Layout ``[dx, dy, dz, droll, dpitch, dyaw, gripper]``: the first six are the *displacement*
    of the end-effector frame from step ``t`` to ``t+1``, the seventh is the gripper command at
    ``t+1``, binarized at ``grip_threshold`` because the ports published so far take a binary
    open/close rather than a width.

    Angular differences are wrapped into ``[-pi, pi]``, so a yaw crossing +-pi reads as the small
    rotation it is instead of a 2*pi jump that would dwarf every real motion in the episode.
    """
    q = np.asarray(q, dtype=np.float64)
    gripper = np.asarray(gripper, dtype=np.float64).reshape(-1)
    if gripper.shape[0] != q.shape[0]:
        raise ValueError(f"gripper has {gripper.shape[0]} steps, q has {q.shape[0]}")
    if q.shape[0] < 2:
        raise ValueError("need at least 2 frames to form a displacement")

    kin = kinematics if kinematics is not None else G1Kinematics(scene_path)
    poses = kin.ee_poses(q)
    delta = np.diff(poses, axis=0)
    delta[:, 3:] = (delta[:, 3:] + np.pi) % (2 * np.pi) - np.pi
    grip = (gripper[1:] > grip_threshold).astype(np.float64)
    return np.concatenate([delta, grip[:, None]], axis=1)
