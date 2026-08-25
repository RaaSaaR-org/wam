"""MuJoCo ground-truth capture binding for PR-08 §4 step 1 (`EST_DRIFT_P95`).

    THIS IS A CAPTURE SHIM, NOT A SECOND ROBOT ADAPTER AND NOT A TRANSPORT.

    ``MujocoG1Transport`` is how MuJoCo drives the G1 through the ``G1Transport`` seam and
    nothing here replaces it. This module exists for exactly one caller —
    ``scripts/measure_est_drift.py capture`` — and implements exactly the members that
    caller touches: ``ground_truth_channels``, ``step``, ``get_physics_step_count``,
    ``render_frame``, ``render_depth``, ``render_segmentation``, ``close``. It does not
    command joints for a rollout, it does not carry an e-stop, and it must never be
    registered as a robot. ``capture_frames`` takes a *binding* rather than building one
    precisely so a second ground-truth source is a shim of this size instead of a fork.

WHY THIS EXISTS

  PR-08 §4 step 1 says "render N Isaac episodes with ground-truth depth + segmentation".
  ``docs/isaac-est-drift-runbook.md`` §6 prices that route at roughly a week, of which the
  install is about an hour and **the missing Isaac scene is the rest**: the bare ``g1.usd``
  has no table, no plate and no apple, so a capture from it measures nothing (§4.2). The
  project owner chose MuJoCo instead (2026-08-22), registered as
  ``docs/preregistration/PR-08-V5-ground-truth-route.md``, rule ``T40_RULE_V5``.

  The load-bearing observation is that ``EST_DRIFT_P95`` is defined purely on
  **segmentation** — the p95 centroid displacement between the estimated mask and the true
  mask. §4 step 3's depth error is **recorded, not gated** (``apple_sam2``'s docstring says
  so outright). A route with ground-truth masks therefore produces the gated number in
  full, and MuJoCo has exact per-pixel geom-id segmentation on CPU, headless, with no
  install and no network.

  **V5 does not license generation.** ``T40_RULE_V1`` §1's forbid list is untouched.

WHICH WAY THIS ROUTE'S ERROR POINTS, AND WHY THAT IS THE WHOLE ARGUMENT FOR IT

  §6 SUBTRACTS ``EST_DRIFT_P95`` from ``GEOM_TOL``. Subtracting a number that is too small
  leaves the tolerance too wide, so an optimistic error budget always lands in the
  generator's favour. That is PR-08 §4's own stated weakness about Isaac: RTX-photoreal
  frames flatter a detector trained on photographs, so the measured drift is a *lower
  bound* and the gate it feeds is *looser* than the truth.

  MuJoCo's rasteriser is markedly less photoreal than Isaac's RTX path, so the same
  detector does **worse** on these frames. Worse detection means a larger p95, a larger
  subtraction, a smaller ``GEOM_TOL − EST_DRIFT_P95`` and therefore a **stricter** G0b.
  The error lands against the generator, which is the safe direction. **That is an
  argument, not a measurement**, and the artifact records it as such: nothing in this file
  claims the number is an upper bound.

  The practical consequence for anyone editing this module: **when a rendering choice would
  flatter the estimator, take the one that does not.** Two such choices are made and
  labelled below — the orange cube is left in the scene as a distractor, and the hands are
  left where they occlude.

WHAT IS GROUND TRUTH HERE AND WHAT IS NOT

- **Segmentation is exact.** ``mujoco.Renderer.enable_segmentation_rendering()`` returns the
  geom that owns each pixel, from the same rasteriser that drew the RGB. There is no
  annotation, no threshold and no model in that path. This is the gated quantity.
- **Depth is exact but is a DIFFERENT QUANTITY from Isaac's.** MuJoCo's depth buffer is
  distance to the **image plane** (Replicator would call this ``distance_to_image_plane``);
  ``isaac_binding.GROUND_TRUTH_ANNOTATORS`` wires ``distance_to_camera``, which is
  **euclidean ray length**, and warns that comparing the two inflates the error by
  ``1 / cos(angle off the optical axis)`` — 1.41 at 45°. Neither is converted into the
  other here. The channel is stamped in the capture header
  (``depth_semantics``) so a reader cannot mistake one for the other, and it costs the gate
  nothing because §4 step 3's depth error is recorded and not gated. Worth stating plainly:
  a metric monocular estimator predicts image-plane depth, so on this axis the MuJoCo
  channel is the *closer* comparison of the two, and it is the Isaac path that owes a
  conversion.
- **Background depth is ``inf``, deliberately.** MuJoCo returns (nearly) the far clip plane
  for a ray that hit no geometry; ``distance_to_camera`` returns ``inf`` and
  ``measure_est_drift``'s ``depth_error`` excludes non-finite pixels and counts them. Such
  pixels are therefore mapped to ``inf`` here (:meth:`render_depth`), identified from the
  **segmentation buffer** rather than from a threshold on the far plane — see that method for
  the measurement that rules the threshold out — so that a rig which forgets to mask fails the
  same way against both bindings instead of quietly averaging in a 44-metre sky.
- **The pose schedule is not physics.** The object is a static prop moved by writing
  ``model.body_pos`` between scene states; it is not dropped, not grasped, and carries no
  contacts. A calibration capture measures *the estimator*, not a manipulation.

THE OBJECT, AND THE LIMITATION THAT TRAVELS WITH THE NUMBER

  The runbook's stated objection to this route was that ``configs/sim/g1_scene.xml``'s only
  graspable object is an **orange cube**, and a budget for "finding a cube in a MuJoCo
  render" transferred to "finding an apple in a RealSense frame" is a different quantity.
  It also breaks the gate outright: the estimator would have to be prompted
  ``"orange cube."``, ``apple_sam2.SEGMENTER_CONTRACT`` carries the prompt, and
  ``measure_est_drift.cross_check_geom_tol`` compares that block field for field against the
  committed ``configs/transfer25/pr08_geom_tol.json`` (``object_text_prompt: "apple."``) —
  so a cube run is stamped ``segmenter_params_disagree_with_geom_tol`` and disqualified.

  An apple mesh **was** found reachable offline on this box (see
  :data:`OBJECT_MESH_SEARCH_PATHS`), so the cube substitution is not taken. What remains,
  and what :meth:`limitations` puts in the capture header as first-class named fields rather
  than in a docstring nobody reads next to the number:

  1. the mesh is a **convex-decomposition proxy** of a scanned apple, not the scan;
  2. it is rendered **untextured**, under one flat material — MuJoCo shades it, nothing
     paints it;
  3. the scene is a **rasterised** scene, which is the whole basis of the conservative-
     direction argument above and is not a defect to be fixed;
  4. the apple is a static prop, so its pose varies across scene states and not within one.

  ``EST_DRIFT_P95`` produced this way cannot be read without also reading what object it was
  measured on, which is the point of putting these in the artifact.

Torch-free; numpy only. ``mujoco`` is imported inside :meth:`MuJoCoGroundTruthBinding.__init__`
and never at module scope, so importing this module on a machine with no MuJoCo and no GL
raises nothing — the same isolation trick ``isaac_binding`` and ``mujoco_transport`` use, and
what lets every refusal below be unit-tested without a GL context.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ONE definition of "what a segmentation frame is" and "which ground-truth channel names
# exist", imported from the module that defined the seam rather than restated. Two copies is
# how a rig comes to be written against one shape and handed the other; and `capture_frames`
# reads `.ids` / `.id_to_labels` off whatever it is given.
from wam.robot.isaac_binding import (
    GROUND_TRUTH_ANNOTATORS,
    SegmentationFrame,
    _validate_ground_truth,
)

__all__ = [
    "DEFAULT_ARM_ACTUATORS",
    "DEFAULT_OBJECT_LABEL",
    "DEFAULT_SCENE",
    "MUJOCO_MISSING_MSG",
    "OBJECT_MESH_ENV_VAR",
    "OBJECT_MESH_SEARCH_PATHS",
    "MuJoCoGroundTruthBinding",
    "ObjectMesh",
    "SceneState",
    "default_scene_schedule",
    "load_obj_mesh",
    "mesh_missing_message",
    "resolve_object_mesh",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]

MUJOCO_MISSING_MSG = (
    "MuJoCo ground-truth capture needs the optional 'sim' extra — `uv pip install mujoco`. "
    "Nothing is fetched by this module."
)

#: The committed scene. UNFORKED: this is the same file ``MujocoG1Transport`` loads, the same
#: table at z = 0.72 m, the same measured reach envelope and the same `ready` keyframe. The
#: apple is added to a *compiled-from-spec copy* of it (see :meth:`_build_model`) and this file
#: on disk is never edited — a calibration capture that quietly moved the E2 scene would make
#: every earlier E2 result unreproducible.
DEFAULT_SCENE = _REPO_ROOT / "configs/sim/g1_scene.xml"

#: The geom name given to the added object, and therefore the label text
#: ``measure_est_drift.object_ids`` matches on. It is spelled to equal the estimator's own
#: ``OBJECT_TEXT_PROMPT`` with the trailing period stripped (``"apple." -> "apple"``), because
#: ``object_ids`` forgives case and surrounding whitespace and NOTHING else — a stage calling
#: the fruit ``apple_01`` or ``Fruit`` matches nothing, every frame lands in
#: ``n_frames_without_object_label`` and the run measures nothing with no crash (runbook §4.5).
DEFAULT_OBJECT_LABEL = "apple"

#: Environment override for the object mesh, checked first by :func:`resolve_object_mesh`.
OBJECT_MESH_ENV_VAR = "WAM_PR08_OBJECT_MESH"

#: Where an apple mesh is looked for, in order, and **nothing is ever downloaded**.
#:
#: ``assets/`` is gitignored on purpose (".gitignore": *third-party models fetched, never
#: vendored*), which is why the mesh is not committed beside this file and why the first
#: entry is the repo-local drop point rather than a tracked path — the same arrangement
#: ``configs/sim/g1_scene.xml`` already has with the Menagerie STLs that
#: ``scripts/fetch_g1_model.py`` puts under ``assets/mujoco/unitree_g1``.
#:
#: The remaining entries are copies that were **measured present on this box on 2026-08-22**:
#: ManiSkill2-real2sim's ``apple`` object, vendored inside SimplerEnv inside Isaac-GR00T's
#: external dependencies. That tree is Apache-2.0 (``ManiSkill2_real2sim/LICENSE``) inside an
#: MIT repo (``SimplerEnv/LICENSE``). A path that does not exist is skipped in silence; if
#: none exists the refusal names every one of them (:func:`mesh_missing_message`).
OBJECT_MESH_SEARCH_PATHS: tuple[Path, ...] = (
    _REPO_ROOT / "assets/mujoco/objects/apple/collision.obj",
    Path.home()
    / "IsaacLab-Arena/submodules/Isaac-GR00T/external_dependencies/SimplerEnv"
    / "ManiSkill2_real2sim/data/custom/models/apple/collision.obj",
    Path.home()
    / "Dokumente/Unitree/g1_quest_teleop/gr00t_teleop/upstream/IsaacLab-Arena"
    / "submodules/Isaac-GR00T/external_dependencies/SimplerEnv"
    / "ManiSkill2_real2sim/data/custom/models/apple/collision.obj",
)

#: The two actuators the scene schedule nudges, so that "arm configuration" in PR-08 §4.6's
#: "N distinct scene states" is a real axis and not a word. Resolved BY NAME against the
#: compiled model; a model that has neither is not an error (the schedule then varies object
#: pose only) but it IS recorded, because a silently one-axis capture reads exactly like a
#: two-axis one in the artifact.
DEFAULT_ARM_ACTUATORS: tuple[str, str] = (
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
)

#: MuJoCo's own object-type code for a geom, restated as an int so this module can compare
#: against a segmentation buffer without importing mujoco at module scope. Asserted against
#: ``mujoco.mjtObj.mjOBJ_GEOM`` at construction rather than trusted (:meth:`_check_objtype`).
_MJOBJ_GEOM = 5


# -- the object mesh -------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectMesh:
    """Triangles for the added object, plus how they were obtained.

    ``source`` and ``groups_merged`` are not decoration: they travel into the capture header,
    and "which apple was this" is the first question anyone auditing an ``EST_DRIFT_P95``
    measured on a stand-in object has to be able to answer without re-running anything.
    """

    verts: np.ndarray
    faces: np.ndarray
    source: Path
    groups_merged: int

    @property
    def extent_m(self) -> tuple[float, float, float]:
        """Axis-aligned bounding-box size in metres, as loaded (before MuJoCo re-centres it)."""
        lo = self.verts.min(axis=0)
        hi = self.verts.max(axis=0)
        return (float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2]))


def mesh_missing_message(env_value: str | None = None) -> str:
    """Name every path that was looked in and what would have to change. Never a fallback.

    Same shape and the same reason as ``measure_est_drift._missing_message``: a capture that
    silently fell back to the orange cube would produce a p95 that is a budget for a different
    object, disqualified for a reason (``segmenter_params_disagree_with_geom_tol``) that names
    the prompt rather than the substitution.
    """
    lines = [
        "FATAL: no object mesh for the PR-08 §4 capture scene was found, and none is fetched.",
        "       Nothing was rendered.",
        "",
        f"       ${OBJECT_MESH_ENV_VAR}: {env_value!r}" if env_value else
        f"       ${OBJECT_MESH_ENV_VAR}: unset",
        "",
        "       looked in, in order:",
    ]
    for path in OBJECT_MESH_SEARCH_PATHS:
        lines.append(f"         [{'x' if path.is_file() else ' '}] {path}")
    lines += [
        "",
        "       Put an OBJ (or STL) apple mesh at the first path, or point "
        f"${OBJECT_MESH_ENV_VAR} / --object-mesh at one.",
        "       DO NOT substitute the scene's orange cube: apple_sam2.SEGMENTER_CONTRACT pins",
        "       object_text_prompt='apple.', configs/transfer25/pr08_geom_tol.json commits the",
        "       same string, and measure_est_drift compares that block field for field — a cube",
        "       run is stamped segmenter_params_disagree_with_geom_tol and can never be a gate",
        "       input. See docs/preregistration/PR-08-V5-ground-truth-route.md §4.",
    ]
    return "\n".join(lines)


def resolve_object_mesh(explicit: str | Path | None = None) -> Path:
    """``--object-mesh`` > ``$WAM_PR08_OBJECT_MESH`` > :data:`OBJECT_MESH_SEARCH_PATHS`.

    An explicitly named path that does not exist raises naming it, rather than falling through
    to the search: the operator said which mesh, and quietly rendering a different apple is the
    failure this whole module is careful about.
    """
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"--object-mesh {path} does not exist. Nothing was rendered.")
        return path
    env_value = os.environ.get(OBJECT_MESH_ENV_VAR)
    if env_value:
        path = Path(env_value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"${OBJECT_MESH_ENV_VAR} = {env_value!r} does not exist. Nothing was rendered."
            )
        return path
    for candidate in OBJECT_MESH_SEARCH_PATHS:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(mesh_missing_message(env_value))


def load_obj_mesh(path: str | Path) -> ObjectMesh:
    """Parse an OBJ into ONE triangle soup — every ``o``/``g`` group merged.

    **MUJOCO'S OWN OBJ LOADER KEEPS ONLY THE FIRST OBJECT GROUP, AND THAT IS WHY THIS PARSER
    EXISTS.** Measured 2026-08-22 on this box with mujoco 3.10.0: handing
    ``.../models/apple/collision.obj`` (14 ``o`` groups, 17 723 vertices, 35 390 faces — a
    convex decomposition) straight to ``MjSpec.add_mesh(file=...)`` compiles to
    ``mesh_vertnum = 17723`` and ``mesh_facenum = 316``. All the vertices, one group's faces.
    It renders as a small chip of an apple, covers ~24 px where the whole fruit covers ~1 900,
    and **nothing raises** — the exact failure shape this repository keeps naming: a plausible
    frame, a plausible centroid, a plausible p95, silently measured on the wrong geometry.

    So the file is read here, all groups are merged, polygons are fanned into triangles, and
    the arrays go to MuJoCo through ``uservert``/``userface``. No derived asset is written to
    disk: a generated OBJ sitting beside the original is a second copy that can drift from it.

    Raises ``ValueError`` when the file yields no triangles, naming the file — an OBJ that
    parses to nothing is indistinguishable downstream from an object that is never in frame.
    """
    path = Path(path)
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    groups = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                # An OBJ vertex line may carry per-vertex colour after xyz (this apple's does:
                # `v x y z r g b`). Take the first three and ignore the rest rather than
                # refusing the file — the colour is not used, MuJoCo shades the geom.
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                idx = []
                for token in line.split()[1:]:
                    raw = int(token.split("/")[0])
                    # OBJ indices are 1-based, and NEGATIVE indices are relative to the end of
                    # the vertex list *so far*. Both spellings appear in the wild; getting the
                    # negative case wrong scrambles the mesh instead of failing.
                    idx.append(raw - 1 if raw > 0 else len(verts) + raw)
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
            elif line.startswith(("o ", "g ")):
                groups += 1
    if not faces or not verts:
        raise ValueError(
            f"{path} yielded {len(verts)} vertices and {len(faces)} triangles — there is no "
            "object to render, and a capture from an empty mesh measures nothing without "
            "crashing."
        )
    return ObjectMesh(
        verts=np.asarray(verts, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int32),
        source=path,
        groups_merged=groups,
    )


# -- the scene schedule ----------------------------------------------------------------------


@dataclass(frozen=True)
class SceneState:
    """One distinct scene configuration: where the object is, and how the arms are held.

    PR-08 §4.6 (runbook): *"N has to be counted in distinct configurations, not frames"* — on
    a static stage N frames are N nearly identical samples and a "95th percentile" over them
    is a percentile over one viewpoint. This dataclass is what makes the count auditable, and
    the capture header records how many of them a run actually visited.

    **``object_xy`` and not ``object_pos``**: the height is DERIVED, not chosen — the table top
    at z = 0.72 m plus the resolved mesh's own half-height, so the object rests on the table
    whatever mesh was found. A z in this dataclass would be a field the binding ignores, which
    is a silent disagreement between what a caller wrote and what was rendered.
    """

    object_xy: tuple[float, float]
    object_yaw_rad: float
    arm_pitch_offset_rad: float


#: Object placements, in metres, in the world frame the scene header documents: the table top
#: is at z = 0.72 m and spans x 0.24..0.64, y -0.32..0.32. These stay inside that top with a
#: margin, and they deliberately include the far corners where the hands occlude the object and
#: the near edge where it is largest — an easy-frames-only sweep would flatter the estimator,
#: which is the one thing this route must not do.
_SCHEDULE_X: tuple[float, ...] = (0.30, 0.36, 0.42, 0.48)
_SCHEDULE_Y: tuple[float, ...] = (-0.20, -0.10, 0.0, 0.10, 0.20)

#: Shoulder-pitch offsets applied (mirrored) to the two arms. Small on purpose: the `ready`
#: keyframe is the one pose this scene's clearances were measured at, and a large sweep would
#: be driving the arm into the table rather than varying the view.
_SCHEDULE_ARM: tuple[float, ...] = (-0.15, 0.0, 0.15)


def default_scene_schedule(n_states: int = 20) -> tuple[SceneState, ...]:
    """A deterministic sweep of ``n_states`` distinct configurations. No RNG.

    Determinism is the repo convention and it is load-bearing twice over: a capture has to be
    re-renderable to be audited (AC-04), and a p95 that moves under reseeding is a p95 nobody
    can compare across estimator revisions.

    The object walks the x/y lattice above; yaw and the arm offset advance on their own strides
    so that consecutive states differ in more than one way. The height is not here — see
    :class:`SceneState`.
    """
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1, got {n_states}")
    lattice = [(x, y) for x in _SCHEDULE_X for y in _SCHEDULE_Y]
    states: list[SceneState] = []
    for i in range(n_states):
        x, y = lattice[i % len(lattice)]
        states.append(
            SceneState(
                object_xy=(x, y),
                object_yaw_rad=(i % 5) * (2.0 * math.pi / 5.0),
                arm_pitch_offset_rad=_SCHEDULE_ARM[i % len(_SCHEDULE_ARM)],
            )
        )
    return tuple(states)


#: The trajectory schedule's envelope, DERIVED from the lattice above rather than typed again.
#:
#: PR-08-V5 §4.5 registers that *"any change to the capture scene that raises the object's
#: visibility is a change that must be argued in a further V-document, not made in a commit"*. The
#: placements the lattice sweeps are what that clause was registered over, so the trajectory is
#: confined to their bounding box and the arm to their amplitude: every pose the smooth schedule
#: visits is inside the region the committed one already reaches, and no pose is nearer the camera,
#: further from the hands or otherwise easier than the lattice already goes. Deriving it is the
#: point — a second copy of ``0.30..0.48`` here is how the two would come to disagree, and the one
#: that disagreed upward would be the one that quietly widened the envelope.
_TRAJECTORY_CENTER_XY: tuple[float, float] = (
    0.5 * (min(_SCHEDULE_X) + max(_SCHEDULE_X)),
    0.5 * (min(_SCHEDULE_Y) + max(_SCHEDULE_Y)),
)
_TRAJECTORY_RADII_XY: tuple[float, float] = (
    0.5 * (max(_SCHEDULE_X) - min(_SCHEDULE_X)),
    0.5 * (max(_SCHEDULE_Y) - min(_SCHEDULE_Y)),
)
_TRAJECTORY_ARM_AMPLITUDE_RAD: float = max(abs(v) for v in _SCHEDULE_ARM)


def trajectory_scene_schedule(
    n_frames: int,
    *,
    turns: float = 1.0,
    yaw_turns: float = 1.0,
    arm_cycles: float = 2.0,
) -> tuple[SceneState, ...]:
    """``n_frames`` states on a SMOOTH, continuous path — one state per FRAME, no jump cuts.

    WHAT THIS IS FOR, AND WHAT IT IS NOT FOR

    :func:`default_scene_schedule` is a sweep of *distinct configurations* and is exactly right for
    the quantity ``EST_DRIFT_P95`` names: a per-frame percentile wants independent viewpoints, and
    PR-08 §4.6's *"N counted in configurations, not frames"* is that requirement. It is exactly
    wrong for anything that PROPAGATES. ``apple_sam2``'s third gate-qualification blocker names its
    own discharge condition — measure *"the same capture BOTH ways — this adapter per frame, and
    the video predictor propagating from frame 0 — and recording the two p95s"* — and that
    experiment is not merely unrun against the lattice, it is **unrunnable**: the lattice teleports
    the object between neighbouring frames (measured on the head camera at 240x320: up to ~64 px,
    and 55.9 / 65.3 / 290.1 px at 480x640), so propagating from frame 0 crosses a cut on frame 1
    and every number after it is a measurement of the cut.

    **This function builds the capture that makes that experiment possible. It does not run it, it
    is not the propagation arm, and producing a temporally coherent capture discharges nothing.**

    THE DESIGN, AND WHY EACH PART OF IT

    * **One state per frame, not per configuration.** Continuity is the product here, so the
      caller drives ``steps_per_state == steps_per_frame`` and the state index advances once per
      rendered frame. ``n_frames`` is therefore the count of *both*, and unlike the lattice — which
      saturates at 60 distinct tuples and repeats — asking for more frames buys more configurations
      forever, because a curve sampled more finely is still a curve.
    * **A closed ellipse in the table plane**, centred and sized by
      :data:`_TRAJECTORY_CENTER_XY` / :data:`_TRAJECTORY_RADII_XY`, which are the lattice's own
      bounding box (see there for the V5 §4.5 reason). The per-frame increment is
      ``2*pi*turns*r/n_frames`` — **O(1/n)**, so more frames make the capture *smoother* rather
      than longer with the same cuts, which is the property a propagation arm actually needs. The
      path is closed so that the last frame is a neighbour of the first: a run that wraps has no
      seam either.
    * **The object MOVES, and it moves on every frame.** An ellipse has no stationary point, so
      there is no run of duplicate frames anywhere in the capture — which matters because a static
      prop dressed up as a trajectory would hand the propagation arm the lattice's duplicates
      without the jump that makes them visible. The capture rig measures this rather than trusting
      it (``measure_est_drift``'s ``temporal_coherence`` block); the name of this function is not
      evidence.
    * **The arm sweeps, on its own rate.** ``arm_cycles`` defaults to 2 so the shoulder-pitch
      offset is not a function of the object's y — two axes that advance in lockstep are one axis.
      The amplitude is the lattice's own (:data:`_TRAJECTORY_ARM_AMPLITUDE_RAD`), which keeps the
      sweep inside the clearances the ``ready`` keyframe was measured at *and* keeps the Dex3 hands
      passing in front of the object exactly as V5 §4.5 requires them to.
    * **Yaw is monotone, not wrapped.** ``_apply_state`` turns it into a quaternion, so magnitude
      is irrelevant to the render; leaving it unwrapped means the *state values* are continuous
      too, and a reader diffing consecutive entries sees no 2*pi discontinuity that is not in the
      picture.
    * **No RNG, and no envelope override.** Determinism is the repo convention (AC-04). There is
      deliberately no parameter for the centre or the radii: widening the envelope is the change
      V5 §4.5 says must be argued in a document, and a keyword argument is how it would instead be
      made in a commit.

    ``turns``, ``yaw_turns`` and ``arm_cycles`` count *complete cycles over the whole capture*, so
    every one of them is scale-free in ``n_frames`` and none of them can reintroduce a cut.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    cx, cy = _TRAJECTORY_CENTER_XY
    rx, ry = _TRAJECTORY_RADII_XY
    states: list[SceneState] = []
    for i in range(n_frames):
        # i / n_frames and NOT i / (n_frames - 1): a uniform increment and a CLOSED loop, so the
        # step between the last frame and a wrap back to the first is the same size as every other
        # step. Dividing by n-1 would land the final frame exactly on the first one, which is a
        # duplicate configuration at the one place nobody looks.
        s = i / float(n_frames)
        theta = 2.0 * math.pi * float(turns) * s
        states.append(
            SceneState(
                object_xy=(cx + rx * math.cos(theta), cy + ry * math.sin(theta)),
                object_yaw_rad=2.0 * math.pi * float(yaw_turns) * s,
                arm_pitch_offset_rad=_TRAJECTORY_ARM_AMPLITUDE_RAD
                * math.sin(2.0 * math.pi * float(arm_cycles) * s),
            )
        )
    return tuple(states)


#: The schedules ``measure_est_drift.py --schedule`` may name, so the CLI's choices and the
#: functions behind them cannot drift apart. ``lattice`` is first and is the default everywhere:
#: every capture in ``runs/pr08-est-drift/`` was made with it and none of them may change meaning.
SCENE_SCHEDULES: dict[str, Any] = {
    "lattice": default_scene_schedule,
    "trajectory": trajectory_scene_schedule,
}


# -- the binding -----------------------------------------------------------------------------


class MuJoCoGroundTruthBinding:
    """Ground-truth RGB + depth + segmentation out of ``configs/sim/g1_scene.xml``, headless.

    Construction order, and each step is a thing that can refuse:

    1. import ``mujoco`` (never at module scope), and check the geom object-type code this
       module compares against;
    2. resolve and parse the object mesh (:func:`resolve_object_mesh`, :func:`load_obj_mesh`);
    3. compile the committed scene **plus** the object through ``MjSpec`` — the file on disk is
       not edited and the object carries no joint, so ``nq`` is unchanged and the scene's own
       ``ready`` keyframe stays valid;
    4. resolve the object geom, the object body and the two arm actuators BY NAME;
    5. build the offscreen renderer, which is the only step that needs a GL context.

    Ground truth is **opt-in**, exactly as in ``isaac_binding``: without ``ground_truth=`` the
    depth and segmentation calls RAISE rather than returning zeros. There is no rollout on this
    path that would pay for the extra passes, but the symmetry is the point — a rig that only
    ever meets one binding has to meet the same errors from both.

    **There is no warmup.** MuJoCo's first render is a real render, so ``render_frame`` never
    returns ``None``. That is a difference from Isaac (up to 20 empty frames) and not a hidden
    one: ``capture_frames`` handles ``None`` from either, and a binding that fabricated a
    warmup would be inventing a failure mode to look symmetric.
    """

    def __init__(
        self,
        *,
        scene: str | Path = DEFAULT_SCENE,
        object_mesh: str | Path | None = None,
        object_label: str = DEFAULT_OBJECT_LABEL,
        object_rgba: Sequence[float] = (0.78, 0.07, 0.07, 1.0),
        cameras: Sequence[str] = ("head",),
        render_hw: tuple[int, int] = (480, 640),
        ground_truth: Sequence[str] = (),
        schedule: Sequence[SceneState] | None = None,
        steps_per_state: int = 1,
        keyframe: str = "ready",
        arm_actuators: Sequence[str] = DEFAULT_ARM_ACTUATORS,
        build_renderer: bool = True,
    ) -> None:
        """Compile the capture scene and (optionally) open the renderer.

        ``cameras``: names that must exist in the MJCF — validated against the compiled model,
        which costs a sub-second compile rather than Isaac's full boot, so the check is made
        where the answer actually is. ``render_hw``: (H, W); refused when it exceeds the
        scene's ``<visual><global offwidth/offheight>``, because MuJoCo would otherwise clamp
        the offscreen buffer and hand back a differently-sized frame that ``measure`` would
        disqualify two machines later. ``steps_per_state``: how many physics steps one entry of
        the schedule holds — the caller knows ``frames × steps_per_frame`` and this class does
        not, so the caller does that division. ``build_renderer=False`` compiles everything and
        opens no GL context, which is how the refusals below are tested without a display.
        """
        try:
            import mujoco  # noqa: PLC0415 — module-scope import would need MuJoCo to import us
        except ImportError as exc:  # pragma: no cover - exercised only where mujoco is absent
            raise RuntimeError(MUJOCO_MISSING_MSG) from exc
        self._mj = mujoco
        self._check_objtype()

        if len(render_hw) != 2 or any(int(v) < 1 for v in render_hw):
            raise ValueError(f"render_hw must be two positive ints, got {render_hw!r}")
        if steps_per_state < 1:
            raise ValueError(f"steps_per_state must be >= 1, got {steps_per_state}")
        label = str(object_label).strip()
        if not label:
            raise ValueError(
                "object_label is empty — measure_est_drift.object_ids matches the capture's "
                "label text against the estimator's prompt, and an unnamed object matches "
                "nothing while raising nothing."
            )
        self._object_label = label
        self._render_hw = (int(render_hw[0]), int(render_hw[1]))
        self._ground_truth = _validate_ground_truth(ground_truth)
        self._cameras = tuple(str(c) for c in cameras)
        if not self._cameras:
            raise ValueError("cameras must name at least one camera in the MJCF")

        self._scene_path = Path(scene)
        if not self._scene_path.is_file():
            raise FileNotFoundError(f"scene {self._scene_path} does not exist")
        self._mesh = load_obj_mesh(resolve_object_mesh(object_mesh))
        self._object_rgba = tuple(float(v) for v in object_rgba)

        self._model = self._build_model()
        self._data = mujoco.MjData(self._model)

        self._object_bid = self._require_id(mujoco.mjtObj.mjOBJ_BODY, label, "body")
        self._object_gid = self._require_id(mujoco.mjtObj.mjOBJ_GEOM, label, "geom")
        self._validate_cameras()

        # Half-height of the object as MuJoCo ended up storing it, so the schedule can put the
        # object ON the table rather than through it or hovering above it. geom_size for a mesh
        # geom is the mesh's own half-extent, which is why this is read and not assumed.
        self._object_half_z = float(self._model.geom_size[self._object_gid][2])

        self._arm_actuator_ids: dict[str, int] = {}
        for name in arm_actuators:
            aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, str(name))
            if aid >= 0:
                self._arm_actuator_ids[str(name)] = aid

        self._keyframe = str(keyframe)
        self._keyframe_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, self._keyframe)
        if self._keyframe_id < 0:
            raise ValueError(
                f"the scene has no keyframe {self._keyframe!r}; it has "
                f"{[mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(self._model.nkey)]}"
                " — look keyframes up by NAME (configs/sim/g1_scene.xml header)"
            )

        self._schedule = tuple(schedule) if schedule is not None else default_scene_schedule()
        if not self._schedule:
            raise ValueError("schedule must contain at least one SceneState")
        self._steps_per_state = int(steps_per_state)

        self._steps = 0
        self._state_index = -1
        self._states_visited = 0
        self._closed = False
        self._renderer: Any = None

        self.reset()
        if build_renderer:
            self._renderer = self._make_renderer()

    # -- construction helpers ---------------------------------------------------------------

    def _check_objtype(self) -> None:
        """Assert MuJoCo's geom object-type code, never assume it.

        :data:`_MJOBJ_GEOM` exists so the segmentation buffer can be interpreted without
        importing mujoco at module scope. A vendor enum that moved would silently turn every
        geom pixel into background — full run, zero coverage, no crash.
        """
        actual = int(self._mj.mjtObj.mjOBJ_GEOM)
        if actual != _MJOBJ_GEOM:
            raise RuntimeError(
                f"mujoco.mjtObj.mjOBJ_GEOM is {actual}, not {_MJOBJ_GEOM} — this module reads "
                "the segmentation buffer's object-type channel against that constant. Update "
                "_MJOBJ_GEOM rather than letting every geom read as background."
            )

    def _build_model(self) -> Any:
        """Compile the committed scene with the object added, WITHOUT touching the scene file.

        ``MjSpec`` rather than a second MJCF that ``<include>``s the first: the scene's element
        order is load-bearing (its own header explains why ``<keyframe>`` precedes the vendor
        ``<include>`` and ``<compiler>`` follows it) and ``meshdir`` is resolved against the
        TOP-LEVEL model file, so a wrapper file would have to either re-point ``meshdir`` — and
        break every Menagerie STL — or reach the object mesh through a ``../..`` path relative
        to the vendor asset directory. Building the spec avoids both: the mesh arrives as
        vertex and face arrays and needs no directory at all.

        **The object carries no joint.** That keeps ``nq`` at the scene's own 71, so the
        ``ready`` keyframe still describes this model; MuJoCo zero-pads a short keyframe rather
        than refusing it, and a zero-padded free joint would put the object at the world origin
        inside the floor with nothing said.
        """
        spec = self._mj.MjSpec.from_file(str(self._scene_path))
        mesh = spec.add_mesh()
        mesh.name = f"wam_{self._object_label}"
        mesh.uservert = self._mesh.verts.reshape(-1)
        mesh.userface = self._mesh.faces.reshape(-1)
        body = spec.worldbody.add_body()
        body.name = self._object_label
        body.pos = [0.36, 0.0, 0.76]
        geom = body.add_geom()
        geom.name = self._object_label
        geom.type = self._mj.mjtGeom.mjGEOM_MESH
        geom.meshname = mesh.name
        geom.rgba = list(self._object_rgba)
        # NO CONTACTS. The object is a calibration prop that the schedule teleports; leaving it
        # collidable would let a scheduled pose intersect the hand and shove the arm, turning a
        # deterministic capture into a physics accident that still renders.
        geom.contype = 0
        geom.conaffinity = 0
        return spec.compile()

    def _require_id(self, objtype: Any, name: str, what: str) -> int:
        ident = self._mj.mj_name2id(self._model, objtype, name)
        if ident < 0:
            raise RuntimeError(f"the compiled model has no {what} named {name!r}")
        return int(ident)

    def _validate_cameras(self) -> None:
        """Every requested camera must exist in the MJCF, named.

        Checked against the compiled model rather than a constant, because the MJCF is where
        the answer is and the compile is sub-second. The refusal lists every camera the scene
        has, which is the fix.
        """
        have = [
            self._mj.mj_id2name(self._model, self._mj.mjtObj.mjOBJ_CAMERA, i)
            for i in range(self._model.ncam)
        ]
        missing = [c for c in self._cameras if c not in have]
        if missing:
            raise ValueError(
                f"unknown camera(s) {missing} in {self._scene_path.name}; it has {have}. "
                "MuJoCo has no viewport camera equivalent to Isaac's `persp` — name one of the "
                "scene's own."
            )

    def _make_renderer(self) -> Any:
        """The single reused offscreen renderer, mirroring ``mujoco_g1``'s one-renderer rule.

        Refuses a grid larger than the scene's offscreen buffer BEFORE rendering: MuJoCo
        clamps ``<visual><global offwidth/offheight>`` silently, and a capture that came back
        at a different size than it asked for is exactly ``resolution_disagrees_with_geom_tol``
        discovered one machine too late.
        """
        height, width = self._render_hw
        offw = int(self._model.vis.global_.offwidth)
        offh = int(self._model.vis.global_.offheight)
        if width > offw or height > offh:
            raise ValueError(
                f"render_hw {height}x{width} exceeds the scene's offscreen buffer "
                f"{offh}x{offw} (<visual><global offwidth/offheight>). MuJoCo would clamp it "
                "and hand back a differently-sized frame, which `measure` disqualifies as "
                "resolution_disagrees_with_geom_tol."
            )
        try:
            return self._mj.Renderer(self._model, height=height, width=width)
        except Exception as exc:  # noqa: BLE001 — mujoco.FatalError is not a RuntimeError
            # MEASURED 2026-08-23 on this workstation: with no DISPLAY and no MUJOCO_GL, this
            # line raises `mujoco.FatalError: gladLoadGL error`, whose base is `Exception` and
            # not `RuntimeError` — so `measure_est_drift.main`'s
            # `except (FileNotFoundError, ValueError, RuntimeError)` does NOT catch it and the
            # operator gets a traceback and exit 1 instead of `FATAL: ...` and exit 2. Every
            # other way this constructor can fail is a named refusal; the one that fires first
            # on a headless box was the one that crashed. `tests/test_mujoco_binding.py` never
            # saw it because it renders in a subprocess with MUJOCO_GL=egl already set.
            #
            # THE BACKEND IS NOT CHOSEN HERE. MuJoCo binds its GL backend at `import mujoco`,
            # so a setenv from inside this process is a no-op — the fix is in the operator's
            # environment, and quietly picking one for them would make the capture's renderer
            # depend on which import happened to come first.
            if isinstance(exc, (ValueError, RuntimeError, FileNotFoundError)):
                raise
            raise RuntimeError(
                f"MuJoCo could not create an offscreen GL context ({type(exc).__name__}: "
                f"{exc}). Nothing was rendered.\n"
                f"       $MUJOCO_GL: {os.environ.get('MUJOCO_GL') or 'unset'}\n"
                f"       $DISPLAY:   {os.environ.get('DISPLAY') or 'unset'}\n"
                "       Headless (no DISPLAY) needs an explicit backend, set BEFORE the "
                "process starts — MuJoCo binds it at `import mujoco` and a later setenv does "
                "nothing:\n"
                "           MUJOCO_GL=egl    python scripts/measure_est_drift.py capture ...\n"
                "           MUJOCO_GL=osmesa python scripts/measure_est_drift.py capture ...  "
                "(software, no GPU)\n"
                "       `egl` is what tests/test_mujoco_binding.py's render subprocess uses."
            ) from exc

    # -- the members `capture_frames` uses ---------------------------------------------------

    @property
    def camera_names(self) -> tuple[str, ...]:
        return self._cameras

    @property
    def ground_truth_channels(self) -> tuple[str, ...]:
        """Which of :data:`~wam.robot.isaac_binding.GROUND_TRUTH_ANNOTATORS` are attached."""
        return self._ground_truth

    @property
    def physics_dt(self) -> float:
        return float(self._model.opt.timestep)

    @property
    def scene_states_visited(self) -> int:
        """How many entries of the schedule this run has actually applied.

        The number PR-08 §4.6 asks for and the artifact had no field for. Recorded rather than
        computed later from ``frames // steps_per_state``, because a run that ended early would
        make that arithmetic overstate it.
        """
        return self._states_visited

    def get_physics_step_count(self) -> int:
        """Physics steps since construction, as an exact ``int``.

        Same contract as the Isaac binding's: an equality test upstream, never a float and
        never derived from sim time.
        """
        self._require_open("get_physics_step_count")
        return int(self._steps)

    def step(self, steps: int = 1) -> None:
        """Advance physics by exactly ``steps``, applying the scene schedule on the way.

        The active state index is ``floor(tick / steps_per_state)``, so the first block of steps
        is state 0 and each change lands on the step that crosses a boundary. That is what turns
        N frames into N frames over ``ceil(N * steps_per_frame / steps_per_state)`` distinct
        configurations, which is PR-08 §4.6's "N counted in configurations, not frames".

        Rendering never calls this — the tick advances only here.
        """
        self._require_open("step")
        if steps < 0:
            raise ValueError(f"steps must be >= 0, got {steps}")
        for _ in range(int(steps)):
            if self._steps % self._steps_per_state == 0:
                self._apply_state(self._steps // self._steps_per_state)
            self._mj.mj_step(self._model, self._data)
            self._steps += 1

    def reset(self) -> None:
        """Restore the ``ready`` keyframe and schedule state 0. Does NOT rewind the tick.

        The raw-counter rule is the Isaac binding's, kept so that a rig which resets mid-run
        behaves the same against either.
        """
        self._require_open("reset")
        self._mj.mj_resetDataKeyframe(self._model, self._data, self._keyframe_id)
        self._base_ctrl = np.asarray(self._data.ctrl, dtype=np.float64).copy()
        self._state_index = -1
        self._apply_state(0)

    def render_frame(self, camera: str) -> np.ndarray | None:
        """One RGB frame as ``uint8 (H, W, 3)``, WITHOUT stepping physics.

        Never ``None`` here (see the class docstring): MuJoCo has no warmup.
        """
        renderer = self._begin_render(camera, "rgb", "render_frame")
        renderer.update_scene(self._data, camera=camera)
        return np.asarray(renderer.render(), dtype=np.uint8).copy()

    def render_depth(self, camera: str) -> np.ndarray | None:
        """Ground-truth depth as ``float32 (H, W)`` in metres, no physics step.

        **Distance to the IMAGE PLANE, not euclidean ray length** — see the module docstring;
        the two differ by ``1/cos`` off-axis and this one is not converted into the other.

        Rays that hit no geometry are mapped to ``inf``, which is what ``distance_to_camera``
        reports and what ``measure_est_drift.depth_error`` excludes and counts. Leaving them in
        would put a 44-metre "sky" into a depth error whose units are metres of tabletop.

        **Background is identified from the SEGMENTATION buffer and not from the far plane**,
        even though that costs a second render pass. MuJoCo's float32 depth reconstruction
        lands *near* the far clip and not *at* it — measured on this scene 2026-08-22, the
        farthest background pixel came back at 44.36677 m against a far plane of 44.38440 m, so
        a ``depth >= far`` test fires on nothing and the only alternative is a tolerance
        somebody coined. Asking the segmenter which geom owns the pixel is exact and needs no
        number. The pass runs even when the ``segmentation`` channel was not requested: it is
        this method's own business, not a channel the caller is being handed.
        """
        renderer = self._begin_render(camera, "depth", "render_depth")
        renderer.update_scene(self._data, camera=camera)
        renderer.enable_depth_rendering()
        try:
            depth = np.asarray(renderer.render(), dtype=np.float32).copy()
        finally:
            # The renderer is reused for every channel and every camera; a mode left enabled by
            # a raising render would make the NEXT call return depth where RGB was asked for.
            renderer.disable_depth_rendering()
        depth[self._render_geom_ids(camera) == 0] = np.inf
        return depth

    def render_segmentation(self, camera: str) -> SegmentationFrame | None:
        """Exact per-pixel geom ids plus their labels, no physics step.

        The buffer MuJoCo hands back is ``int32 (H, W, 2)`` — ``(object id, object type)``,
        with ``(-1, -1)`` for background. It is converted to the shape the rig already reads:
        ids are ``geom_id + 1`` in ``uint32`` so that **0 is background**, matching Replicator's
        unlabelled id, and ``id_to_labels`` is ``{id: {"class": <geom name>}}`` — the shape
        ``isaac_binding.SegmentationFrame`` documents and ``measure_est_drift.label_text``
        indexes.

        Pixels whose object type is not a geom are mapped to background rather than trusted:
        the type channel exists because the id channel alone is ambiguous across object types,
        and reading a site id as a geom id would label a pixel with whatever geom shares the
        number.

        **Every NAMED geom in the model gets an entry, not only the visible ones.** A frame
        where the object is fully occluded then has the label present and an empty mask, which
        ``paired_displacements`` drops and counts as coverage — the honest bucket. Listing only
        visible geoms would file the same frame under ``n_frames_without_object_label``, i.e.
        as "this scene has no apple in it", which is a different claim.
        """
        self._begin_render(camera, "segmentation", "render_segmentation")
        return SegmentationFrame(
            ids=self._render_geom_ids(camera), id_to_labels=self.geom_id_to_labels()
        )

    def _render_geom_ids(self, camera: str) -> np.ndarray:
        """``uint32 (H, W)`` of ``geom_id + 1``, 0 = background. The raw half of the above.

        Separate from :meth:`render_segmentation` because :meth:`render_depth` needs the same
        buffer to find the background, and it needs it whether or not the caller asked for the
        segmentation *channel*. Guards live in the public methods, not here.
        """
        renderer = self._renderer
        renderer.update_scene(self._data, camera=camera)
        renderer.enable_segmentation_rendering()
        try:
            raw = np.asarray(renderer.render())
        finally:
            renderer.disable_segmentation_rendering()
        objid = raw[..., 0].astype(np.int64)
        objtype = raw[..., 1].astype(np.int64)
        return np.where((objtype == _MJOBJ_GEOM) & (objid >= 0), objid + 1, 0).astype(np.uint32)

    def geom_id_to_labels(self) -> dict[int, Any]:
        """``{geom_id + 1: {"class": <geom name>}}`` for every named geom in the model."""
        labels: dict[int, Any] = {}
        for gid in range(self._model.ngeom):
            name = self._mj.mj_id2name(self._model, self._mj.mjtObj.mjOBJ_GEOM, gid)
            if name:
                labels[gid + 1] = {"class": name}
        return labels

    def close(self) -> None:
        """Release the renderer. Idempotent; every other method raises afterwards."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._closed = True

    # -- provenance ------------------------------------------------------------------------

    def limitations(self) -> dict[str, Any]:
        """The named, first-class caveats that travel with this capture into the artifact.

        These are fields and not prose because ``EST_DRIFT_P95`` measured this way must not be
        readable without them (module docstring). ``object_is_substituted`` is the one a reader
        checks first: it is ``False`` here only because an apple mesh was found, and it would be
        ``True`` — with ``object_label`` naming the stand-in — for any other object.

        **What ``object_is_substituted`` cannot tell you** is that the *mesh* is an apple. It
        compares the LABEL, which is the thing the estimator's prompt has to match; a cube mesh
        labelled ``apple`` would report ``False``. ``object_mesh_source``,
        ``object_mesh_triangles`` and ``object_mesh_extent_m`` are the record for that, and they
        are recorded for exactly this reason — no field can catch somebody naming a mesh
        something it is not, so the artifact carries what the mesh WAS instead.
        """
        return {
            "object_label": self._object_label,
            "object_is_substituted": self._object_label != DEFAULT_OBJECT_LABEL,
            "object_mesh_source": str(self._mesh.source),
            "object_mesh_is_convex_decomposition_proxy": self._mesh.groups_merged > 1,
            "object_mesh_groups_merged": int(self._mesh.groups_merged),
            "object_mesh_triangles": int(self._mesh.faces.shape[0]),
            "object_mesh_extent_m": [round(v, 6) for v in self._mesh.extent_m],
            "object_is_untextured": True,
            "object_is_static_prop": True,
            "renderer": "mujoco rasteriser (not ray-traced, not photoreal)",
            "distractors_left_in_scene": ["cube"],
            "depth_semantics": "distance_to_image_plane (NOT distance_to_camera)",
        }

    def provenance(self) -> dict[str, Any]:
        """What the capture header should record about this binding. Merged by the caller."""
        return {
            "backend": "mujoco",
            "mujoco_version": getattr(self._mj, "__version__", None),
            "scene": str(self._scene_path),
            "scene_source": "configs/sim/g1_scene.xml (unmodified; object added via MjSpec)",
            "keyframe": self._keyframe,
            "physics_dt": self.physics_dt,
            "steps_per_state": self._steps_per_state,
            "n_scene_states_scheduled": len(self._schedule),
            "arm_actuators_varied": sorted(self._arm_actuator_ids),
            "object_limitations": self.limitations(),
        }

    # -- internals -------------------------------------------------------------------------

    def _apply_state(self, index: int) -> None:
        """Place the object and hold the arms for schedule entry ``index`` (cyclic).

        Writes ``model.body_pos``/``model.body_quat`` — legitimate for a jointless body, whose
        pose lives in the model rather than in ``qpos`` — and the two arm actuator targets.
        ``mj_forward`` is called so a render taken before the next ``step()`` already shows the
        new pose rather than the previous state's.
        """
        state = self._schedule[index % len(self._schedule)]
        if index != self._state_index:
            self._state_index = index
            self._states_visited += 1
        x, y = state.object_xy
        # Table top at z = 0.72 m (scene header, measured). Half the mesh's own height puts the
        # object ON it, whatever mesh was resolved.
        self._model.body_pos[self._object_bid] = [x, y, 0.72 + self._object_half_z]
        half = 0.5 * float(state.object_yaw_rad)
        self._model.body_quat[self._object_bid] = [math.cos(half), 0.0, 0.0, math.sin(half)]
        self._data.ctrl[:] = self._base_ctrl
        for i, (_, aid) in enumerate(sorted(self._arm_actuator_ids.items())):
            sign = 1.0 if i % 2 == 0 else -1.0
            self._data.ctrl[aid] = self._base_ctrl[aid] + sign * state.arm_pitch_offset_rad
        self._mj.mj_forward(self._model, self._data)

    def _begin_render(self, camera: str, channel: str, op: str) -> Any:
        """The guards every render method shares -> the renderer.

        Refuses an unknown camera with a ``ValueError`` and an unattached ground-truth channel
        with a ``RuntimeError``, in the same shape and for the same reason as
        ``FakeIsaacBinding._begin_render``: a rig that only ever meets one binding has to meet
        the same errors from both.
        """
        self._require_open(op)
        if camera not in self._cameras:
            raise ValueError(f"unknown camera {camera!r}; have {list(self._cameras)}")
        if channel != "rgb" and channel not in self._ground_truth:
            raise RuntimeError(
                f"{op}: the {channel!r} channel "
                f"({GROUND_TRUTH_ANNOTATORS.get(channel, channel)!r} in Replicator's naming) "
                f"is not attached — pass ground_truth={(channel,)!r} when constructing the "
                f"binding. Attached here: {['rgb', *self._ground_truth]}"
            )
        if self._renderer is None:
            raise RuntimeError(
                f"{op}: this binding was built with build_renderer=False and owns no GL "
                "context. That mode exists so the refusals can be tested without a display; "
                "it cannot render."
            )
        return self._renderer

    def _require_open(self, op: str) -> None:
        if self._closed:
            raise RuntimeError(f"{op}: this MuJoCoGroundTruthBinding is closed")
