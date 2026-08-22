"""G1 + Dex3 28-dim vectors <-> the WAM canonical space (T-043, FR-06, FR-02).

The 13 ``unitreerobotics/G1_Dex3_*`` sets (LeRobot v3.0, 3 152 episodes, Apache-2.0) ship a
flat ``float32[28]`` for both ``observation.state`` and ``action``. This repo's canonical
space is different — 15 joints (``waist_yaw`` + both arms, proximal to distal) + 2 scalar
gripper dims (``configs/robot/g1.yaml``, :data:`wam.robot.g1.G1_SPEC`). This module is the
ONLY place the two meet, in the same spirit as ``G1_JOINT_MAP`` in :mod:`wam.robot.g1`.

WHAT IS MEASURED AND WHAT IS NOT — read this before using anything below.

- **[OK] Block order is ARM-FIRST**: ``[0:14]`` arm, ``[14:28]`` hand. Measured 2026-08-15
  across all 13 sets, two independent lines of evidence, both recorded in
  ``.mc/tasks/todo/T-043-convert-the-3152-g1-dex3-episodes-recorded-28-dim-labels.md`` §1 and
  ``docs/contracts/vla-training-consumer.md`` §3.1:
  (1) mechanical joint limits from every set's ``meta/stats.json`` — one-sided dims (a range
  endpoint at 0 ± 2e-3) number **0 of 14** in ``[0:14]`` and **4-10 of 14** in ``[14:28]``,
  unanimous across 13 independently recorded sets, with the far end railing at a clean
  100.0°/100.1° (7 sets) or 120.0° (5 sets); (2) an explicit modality spec from a pipeline
  that produced a working model — ``vla-training/groot/modality_g1_dex3.json:2-9``,
  ``"state"/"action": {"arms": {0, 14}, "hands": {14, 28}}``.
  A HAND-FIRST ordering appears in older documents of this project. It is a real measurement
  of a DIFFERENT corpus (``USC-PSI-Lab/Humanoid-Everyday-G1``, LeRobot v2.1) and is wrong for
  these sets; it was corrected in five documents on 2026-08-15.
- **[?] Left/right order WITHIN each 14-dim block is UNVERIFIED.** Not defaulted here. See
  :class:`SideOrder`.
- **[?] Intra-hand joint order is UNVERIFIED, with THREE mutually inconsistent orderings on
  record.** Not defaulted here. See :class:`HandJointOrder`.
- **[?] The 7-joint order WITHIN one arm is not measured either** — see
  :data:`WITHIN_ARM_ORDER_IS_MEASURED`. This module assumes it already equals the canonical
  proximal-to-distal order, i.e. the source->canonical map applies the SIDE permutation only.
  That assumption is named, flagged and isolated in :func:`arm_block_to_canonical_arms` so a
  measurement can be applied in one place.
- **[OK] The corpus has NO waist column** (28 = 7+7 arm + 7+7 hand; no waist, no legs), while
  the canonical space HAS ``waist_yaw``. It is not invented here: see :func:`to_canonical_q`
  and :func:`canonical_joint_validity`, which follow the converter's IMU-absent precedent
  (FR-02: fill a placeholder, flag the group invalid, record it in the manifest).
- **[OK] Dex3 hands are 7 DoF each**; the canonical gripper channel is one scalar per hand in
  [0, 1]. The [0, 1] affine for THIS hand has not been fitted, and T-31's affine from
  ``GR00T-N1.7-AppleToPlate`` is not transferable (different hand, different range, and two
  hand variants at 100°/120° here). :func:`to_canonical_gripper` therefore REQUIRES an
  explicit :class:`HandSynergyAffine` and refuses to invent one.

Nothing in this module reads the corpus. Of the four unverified degrees of freedom above,
THREE are caller-supplied — side order, intra-hand order and the gripper affine — and omitting
any of them raises :class:`UnverifiedOrderingError` (or, for the affine, refuses) naming the
measurement that would settle it.

The FOURTH is an assumption, and it is the one thing here that is not caller-supplied: the
7-joint order **within** one arm is taken to already equal the canonical proximal-to-distal
order (:data:`WITHIN_ARM_ORDER_IS_MEASURED` is ``False``). It is deliberately not a required
parameter — the grasp-synergy path is permutation-invariant and does not need it, and forcing
it would make every conversion here unusable for the one path that is fully determined. It is
confined to :func:`arm_block_to_canonical_arms` so a measurement lands in exactly one place.
Read it as: **an arm-joint conversion out of this module is only as correct as that
assumption, and nothing here has tested it.**

Torch-free; numpy + pydantic only, like the rest of ``wam.robot``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict

from wam.robot.g1 import G1_SPEC

# -- verified block boundaries ------------------------------------------------------------

SOURCE_VECTOR_DIM: Final[int] = 28
"""Width of the source ``observation.state`` / ``action`` vector. [OK] meta/info.json, 13/13
sets, measured 2026-08-15 (T-043 §2)."""

# [OK] MEASURED 2026-08-15 across all 13 unitreerobotics/G1_Dex3_* sets — arm-first. See the
# module docstring for the two independent evidence lines. Do NOT swap these to hand-first:
# that ordering belongs to USC-PSI-Lab/Humanoid-Everyday-G1 and transposing arm and hand here
# produces a model that trains, converges and moves the wrong joints.
ARM_BLOCK: Final[slice] = slice(0, 14)
HAND_BLOCK: Final[slice] = slice(14, 28)

JOINTS_PER_ARM: Final[int] = 7
"""[OK] 14 arm dims = 7 + 7, two 7-DoF arms and no waist (T-043 §2-3)."""

JOINTS_PER_HAND: Final[int] = 7
"""[OK] Dex3-1 is 7 DoF per hand; 14 hand dims = 7 + 7 (OD-01, T-043 §2)."""

WITHIN_ARM_ORDER_IS_MEASURED: Final[bool] = False
"""[?] Whether the 7-joint order inside ONE arm has been correlated against the parquet.

False. The block order was measured; this was not. This module assumes source order inside a
side already equals the canonical proximal-to-distal order, so :func:`arm_block_to_canonical_arms`
applies the side permutation ONLY. Settled by the same method that settled the block order:
correlate each source column against a joint whose motion is known (T-041's method), or read it
off a producer-side spec for these sets. If it turns out to differ, the fix is one permutation
in :func:`arm_block_to_canonical_arms` and nowhere else.
"""

WAIST_ABSENT_FILL: Final[float] = 0.0
"""Placeholder written into the canonical ``waist_yaw`` slot, which the corpus does not have.

It is NOT a measurement and must never be read as one — :func:`canonical_joint_validity` flags
that slot False and :func:`provenance` records the absence for the manifest. The precedent is
``scripts/convert_lerobot_g1.py``, which writes an identity quaternion and zero vectors for the
absent IMU and sets ``validity.imu=False`` (FR-02 missing-group path).
"""

# Canonical geometry, DERIVED from G1_SPEC rather than restated — the canonical space is the
# contract (configs/robot/g1.yaml <-> wam.robot.g1.G1_SPEC, enforced by tests/test_versioning.py)
# and a second hardcoded copy of it here is exactly the drift CLAUDE.md warns about.
CANONICAL_JOINT_NAMES: Final[tuple[str, ...]] = G1_SPEC.joint_names
NUM_CANONICAL_JOINTS: Final[int] = G1_SPEC.num_joints
NUM_CANONICAL_GRIPPERS: Final[int] = G1_SPEC.gripper_dims
WAIST_YAW_CANONICAL_INDEX: Final[int] = CANONICAL_JOINT_NAMES.index("waist_yaw")
CANONICAL_LEFT_ARM: Final[slice] = slice(
    CANONICAL_JOINT_NAMES.index("left_shoulder_pitch"),
    CANONICAL_JOINT_NAMES.index("left_shoulder_pitch") + JOINTS_PER_ARM,
)
CANONICAL_RIGHT_ARM: Final[slice] = slice(
    CANONICAL_JOINT_NAMES.index("right_shoulder_pitch"),
    CANONICAL_JOINT_NAMES.index("right_shoulder_pitch") + JOINTS_PER_ARM,
)


# -- the unverified degrees of freedom, as explicit choices --------------------------------


class UnverifiedOrderingError(ValueError):
    """Raised when a caller asks for a conversion that depends on something unmeasured.

    A ``ValueError`` subclass so callers that only catch ``ValueError`` still stop; a distinct
    class so a converter can tell "you have not decided yet" from "your array is the wrong
    shape".
    """


class SideOrder(str, Enum):
    """[?] Which half of a 14-dim block is the LEFT limb. UNVERIFIED for this corpus.

    Both 14-dim blocks (arm and hand) are two 7-DoF limbs concatenated, and which one comes
    first has never been correlated against the parquet for the ``unitreerobotics/G1_Dex3_*``
    sets. Two possibilities, no default: choosing wrong mirrors the robot, and a mirrored
    dataset trains to convergence while reaching with the wrong arm.
    """

    LEFT_FIRST = "left_first"
    """Source ``block[0:7]`` is the left limb, ``block[7:14]`` the right."""

    RIGHT_FIRST = "right_first"
    """Source ``block[0:7]`` is the right limb, ``block[7:14]`` the left."""


class HandJointOrder(str, Enum):
    """[?] Intra-hand joint order. THREE mutually inconsistent orderings are on record.

    Recorded in ``docs/contracts/vla-training-consumer.md`` §3.2 and T-043 §1 (2026-08-15).
    None of them has been correlated against these sets' parquet, and T-041's lesson applies
    verbatim: a source that was wrong about the block order earns no trust about the finger
    order. There is a known-wrong default, so this module has none.
    """

    CORPUS_CARD_THUMB_FIRST_SYMMETRIC = "corpus_card_thumb_first_symmetric"
    """The ``unitreerobotics`` dataset card: thumb-first and SYMMETRIC across hands.

    Only the description is on record here, not a per-joint list — see
    :func:`hand_joint_names`, which refuses to invent one.
    """

    ARENA_INDEX_FIRST = "arena_index_first"
    """Arena's ordering: index-first. Description only, no per-joint list on record."""

    NVIDIA_ASYMMETRIC = "nvidia_asymmetric"
    """NVIDIA's ApplePnP pipeline: ASYMMETRIC across hands, fully specified.

    [OK] for corpus A (``nvidia/GR00T-N1.7-AppleToPlate``, 43-dim), read from
    ``vla-training/eval/exported_leapp.yaml:72-87`` and recorded at
    ``docs/contracts/vla-training-consumer.md §2.4:232-233``; the decoder-side permutations back to
    that order are ``[4,5,6,2,3,0,1]`` (left) / ``[4,5,6,0,1,2,3]`` (right)
    (``eval/onnx_leapp_server.py:104-111``). [?] as applied to THIS corpus: it is a candidate,
    not a measurement.
    """


# Per-joint element names, per ordering, as (left, right). ``None`` where only a DESCRIPTION is
# on record and no per-joint list — inventing the missing halves ("thumb-first symmetric" does
# not say whether index precedes middle) is exactly the failure mode this module exists to
# prevent, so those entries stay None and hand_joint_names() raises.
_HAND_JOINT_NAMES: Final[dict[HandJointOrder, tuple[tuple[str, ...], tuple[str, ...]] | None]] = {
    HandJointOrder.CORPUS_CARD_THUMB_FIRST_SYMMETRIC: None,
    HandJointOrder.ARENA_INDEX_FIRST: None,
    HandJointOrder.NVIDIA_ASYMMETRIC: (
        ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"),
        ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"),
    ),
}

_SETTLE_SIDE_ORDER = (
    "left/right order within the 14-dim block: correlate one side's columns against a frame "
    "range where only one arm is known to move (T-041's method, "
    "docs/contracts/vla-training-consumer.md §3.2), or against cam_left_high pixels"
)
_SETTLE_HAND_ORDER = (
    "intra-hand joint order: three mutually inconsistent orderings are on record (the corpus "
    "card's thumb-first symmetric, Arena's index-first, and NVIDIA's asymmetric left "
    "[4,5,6,2,3,0,1] / right [4,5,6,0,1,2,3]); correlate the 7 columns of one hand against a "
    "grasp whose finger sequence is known, do not read the card"
)
_SETTLE_BLOCKER = (
    "both need the action parquets, which have NOT been fetched — cluster/discoverer/"
    "92_fetch_g1_corpus.sbatch took meta/ + videos/ only (T-043 §4: 415 files, 647 MB, "
    "Apache-2.0, and an ask before it is fetched)"
)


class G1Dex3Layout(BaseModel):
    """The two source-layout choices this corpus has not settled, stated explicitly.

    Both fields are REQUIRED and have no default, deliberately: a default would be a guess
    wearing the clothes of a decision, and both guesses fail silently (a mirrored robot, a
    scrambled hand). Construct one only from a recorded measurement, and set ``measured=True``
    plus ``evidence`` when you have it so the manifest can carry the difference between a
    measurement and a working assumption.
    """

    model_config = ConfigDict(frozen=True)

    side_order: SideOrder
    hand_joint_order: HandJointOrder
    measured: bool = False
    """False = a working assumption ([?]). True = correlated against the parquet ([OK])."""

    evidence: str = ""
    """Where the ordering came from; recorded verbatim by :func:`provenance`."""

    @property
    def mark(self) -> str:
        """House-rule provenance mark for this layout: ``[OK]`` or ``[?]``."""
        return "[OK]" if self.measured else "[?]"


def require_layout(layout: G1Dex3Layout | None) -> G1Dex3Layout:
    """Return ``layout``, or raise naming exactly what must be measured to settle it.

    Every public conversion in this module funnels through here, so "I forgot to decide" can
    never be silently answered by a module-level default.
    """
    if layout is None:
        raise UnverifiedOrderingError(
            "G1_Dex3 28-dim layout not specified. The block order IS measured ([OK] arm-first: "
            f"ARM_BLOCK={ARM_BLOCK}, HAND_BLOCK={HAND_BLOCK}, 13/13 sets, 2026-08-15), but two "
            "things inside those blocks are NOT, and this module will not guess either.\n"
            f"  1. {_SETTLE_SIDE_ORDER}\n"
            f"  2. {_SETTLE_HAND_ORDER}\n"
            f"Blocker: {_SETTLE_BLOCKER}\n"
            "Until then, pass an explicit G1Dex3Layout(side_order=SideOrder..., "
            "hand_joint_order=HandJointOrder..., measured=False, evidence='...') and record it "
            "in the dataset manifest via provenance() so the assumption travels with the data."
        )
    if not isinstance(layout, G1Dex3Layout):
        raise TypeError(f"layout must be a G1Dex3Layout, got {type(layout).__name__}")
    return layout


def hand_joint_names(layout: G1Dex3Layout | None, side: str) -> tuple[str, ...]:
    """Per-joint element names for one hand under ``layout``'s ordering.

    Raises :class:`UnverifiedOrderingError` for the two orderings that exist on record only as
    a DESCRIPTION ("thumb-first symmetric", "index-first") and not as a per-joint list. The
    canonical mapping in this module never needs these names — the grasp synergy is a mean and
    is permutation-invariant within a hand — but anything per-finger does, and that is where
    guessing the missing half would go wrong.
    """
    resolved = require_layout(layout)
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    names = _HAND_JOINT_NAMES[resolved.hand_joint_order]
    if names is None:
        raise UnverifiedOrderingError(
            f"hand_joint_names: {resolved.hand_joint_order.value} is on record as a DESCRIPTION "
            "only, not as a per-joint list — e.g. 'thumb-first symmetric' does not say whether "
            "index precedes middle, and 'index-first' does not say where the thumb's three "
            "joints go. Writing the missing half here would be an invention. To settle it: "
            f"{_SETTLE_HAND_ORDER}. Only HandJointOrder.NVIDIA_ASYMMETRIC carries a full "
            "per-joint list (docs/contracts/vla-training-consumer.md §2.4:232-233), and that list is "
            "[OK] for corpus A and [?] as applied here."
        )
    return names[0] if side == "left" else names[1]


# -- array helpers -------------------------------------------------------------------------


def _as_source(vec: Any, name: str = "vec28") -> np.ndarray:
    """Validate and cast a ``[..., 28]`` source vector to float32."""
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim == 0 or arr.shape[-1] != SOURCE_VECTOR_DIM:
        raise ValueError(
            f"{name}: expected last dim {SOURCE_VECTOR_DIM}, got shape {arr.shape}"
        )
    return arr


def _as_block(block: Any, name: str) -> np.ndarray:
    """Validate and cast a ``[..., 14]`` limb-pair block to float32."""
    arr = np.asarray(block, dtype=np.float32)
    width = 2 * JOINTS_PER_ARM
    if arr.ndim == 0 or arr.shape[-1] != width:
        raise ValueError(f"{name}: expected last dim {width}, got shape {arr.shape}")
    return arr


def split_blocks(vec28: Any) -> tuple[np.ndarray, np.ndarray]:
    """``[..., 28]`` -> ``(arm[..., 14], hand[..., 14])`` using the MEASURED block boundaries.

    This is the one operation in the module that rests entirely on measured facts, so it takes
    no layout: :data:`ARM_BLOCK` / :data:`HAND_BLOCK` are the 2026-08-15 measurement.
    """
    arr = _as_source(vec28)
    return arr[..., ARM_BLOCK].copy(), arr[..., HAND_BLOCK].copy()


def join_blocks(arm14: Any, hand14: Any) -> np.ndarray:
    """``(arm[..., 14], hand[..., 14])`` -> ``[..., 28]``. Exact inverse of :func:`split_blocks`."""
    return np.concatenate(
        [_as_block(arm14, "arm14"), _as_block(hand14, "hand14")], axis=-1
    ).astype(np.float32)


def _side_slices(layout: G1Dex3Layout) -> tuple[slice, slice]:
    """(left, right) slices into a 14-dim limb-pair block, per the declared side order."""
    first = slice(0, JOINTS_PER_ARM)
    second = slice(JOINTS_PER_ARM, 2 * JOINTS_PER_ARM)
    if layout.side_order is SideOrder.LEFT_FIRST:
        return first, second
    return second, first


# -- arm block <-> canonical arms -----------------------------------------------------------


def arm_block_to_canonical_arms(arm14: Any, layout: G1Dex3Layout | None) -> np.ndarray:
    """Source arm block ``[..., 14]`` -> canonical arm order ``[..., 14]`` (left 7, then right 7).

    Applies the SIDE permutation only. The 7-joint order inside one arm is passed through
    unchanged, under the explicitly flagged assumption that it already equals the canonical
    proximal-to-distal order (:data:`WITHIN_ARM_ORDER_IS_MEASURED` is False and says how to
    settle it). If a measurement ever contradicts that, the per-side permutation belongs here
    and nowhere else.
    """
    resolved = require_layout(layout)
    arr = _as_block(arm14, "arm14")
    left, right = _side_slices(resolved)
    return np.concatenate([arr[..., left], arr[..., right]], axis=-1).astype(np.float32)


def canonical_arms_to_arm_block(arms14: Any, layout: G1Dex3Layout | None) -> np.ndarray:
    """Canonical arm order ``[..., 14]`` -> source arm block ``[..., 14]``.

    Exact inverse of :func:`arm_block_to_canonical_arms` for the same ``layout``.
    """
    resolved = require_layout(layout)
    arr = _as_block(arms14, "arms14")
    canon_left = arr[..., :JOINTS_PER_ARM]
    canon_right = arr[..., JOINTS_PER_ARM:]
    if resolved.side_order is SideOrder.LEFT_FIRST:
        return np.concatenate([canon_left, canon_right], axis=-1).astype(np.float32)
    return np.concatenate([canon_right, canon_left], axis=-1).astype(np.float32)


# -- canonical q ----------------------------------------------------------------------------


def canonical_joint_validity() -> np.ndarray:
    """Per-canonical-joint validity for anything converted from this corpus: ``[15]`` bool.

    ``waist_yaw`` is False — the corpus has no waist column, so that slot is a placeholder
    (:data:`WAIST_ABSENT_FILL`), not a measurement. Every arm joint is True.

    Why a separate array instead of ``ValidityMask``: :class:`wam.interfaces.schema.ValidityMask`
    is GROUP-level (``q``/``dq``/``imu``/``gripper``), and ``q`` as a group is genuinely valid
    here — 14 of its 15 entries are measured. Flipping ``validity.q`` False would discard the
    arms; leaving it True and saying nothing would present a fabricated waist as measurement.
    So the group flag stays True and this array, plus :func:`provenance`, carries the absence
    into the manifest. Extending the schema with per-joint validity is a change to
    ``src/wam/interfaces/`` (change-with-care, CLAUDE.md) and is T-043's decision to make, not
    this module's.
    """
    valid = np.ones(NUM_CANONICAL_JOINTS, dtype=bool)
    valid[WAIST_YAW_CANONICAL_INDEX] = False
    return valid


def to_canonical_q(
    vec28: Any,
    layout: G1Dex3Layout | None,
    *,
    waist_fill: float = WAIST_ABSENT_FILL,
) -> np.ndarray:
    """Source state ``[..., 28]`` -> canonical ``q [..., 15]`` (waist_yaw + left arm + right arm).

    The hand block is DROPPED here (it becomes the 2-dim gripper channel — see
    :func:`to_canonical_gripper`), and ``waist_yaw`` is filled with ``waist_fill``, which is a
    placeholder and is flagged invalid by :func:`canonical_joint_validity`. The absence is
    recorded, never measured: see :data:`WAIST_ABSENT_FILL`.
    """
    resolved = require_layout(layout)
    arm14, _ = split_blocks(vec28)
    arms = arm_block_to_canonical_arms(arm14, resolved)
    waist = np.full(arms.shape[:-1] + (1,), float(waist_fill), dtype=np.float32)
    return np.concatenate([waist, arms], axis=-1).astype(np.float32)


def to_canonical_joint_delta(vec28_delta: Any, layout: G1Dex3Layout | None) -> np.ndarray:
    """Source-space joint DELTA ``[..., 28]`` -> canonical JOINT_DELTA targets ``[..., 15]``.

    The waist channel is exactly ``0.0`` — never ``waist_fill`` — because a delta is a command:
    a non-zero waist delta would move a joint this corpus never recorded. Zero is a hold, which
    is the only honest command for an absent measurement (and matches how
    :meth:`wam.robot.g1.G1Adapter.execute` treats unmapped motors).
    """
    resolved = require_layout(layout)
    arm14, _ = split_blocks(vec28_delta)
    arms = arm_block_to_canonical_arms(arm14, resolved)
    waist = np.zeros(arms.shape[:-1] + (1,), dtype=np.float32)
    return np.concatenate([waist, arms], axis=-1).astype(np.float32)


def canonical_q_to_arm_block(
    q15: Any, layout: G1Dex3Layout | None, *, require_waist_zero: bool = True
) -> np.ndarray:
    """Canonical ``q [..., 15]`` -> source arm block ``[..., 14]``, dropping ``waist_yaw``.

    The inverse direction is lossy by construction: the corpus has no waist column, so a
    canonical vector whose ``waist_yaw`` carries a value cannot be represented. Rather than
    dropping that value silently, ``require_waist_zero`` refuses it — for JOINT_DELTA targets
    a non-zero waist entry is a commanded motion that would vanish, which is precisely the kind
    of silent loss this module exists to prevent. Pass ``require_waist_zero=False`` only for
    absolute ``q`` whose waist slot is the known placeholder.
    """
    resolved = require_layout(layout)
    arr = np.asarray(q15, dtype=np.float32)
    if arr.ndim == 0 or arr.shape[-1] != NUM_CANONICAL_JOINTS:
        raise ValueError(
            f"q15: expected last dim {NUM_CANONICAL_JOINTS}, got shape {arr.shape}"
        )
    waist = arr[..., WAIST_YAW_CANONICAL_INDEX]
    if require_waist_zero and not np.all(waist == 0.0):
        raise ValueError(
            "canonical_q_to_arm_block: canonical waist_yaw is non-zero "
            f"(max |waist| = {float(np.abs(waist).max())}), but the G1_Dex3 28-dim space has NO "
            "waist column — the value would be dropped silently. For a JOINT_DELTA chunk that "
            "means a commanded waist motion disappearing. Zero the waist channel first, or pass "
            "require_waist_zero=False if this is an absolute q whose waist slot is the "
            "WAIST_ABSENT_FILL placeholder."
        )
    arms = np.concatenate(
        [arr[..., CANONICAL_LEFT_ARM], arr[..., CANONICAL_RIGHT_ARM]], axis=-1
    )
    return canonical_arms_to_arm_block(arms, resolved)


# -- hand block -> the 2-dim canonical gripper ----------------------------------------------


class HandSynergyAffine(NamedTuple):
    """The dataset-level grasp-synergy mapping ``clip((x - offset) / span, 0, 1)``.

    Same shape of object as ``scripts/convert_lerobot_g1.HandAffine``, deliberately NOT imported
    from it: that file is a script loaded by path in the tests, its affine was fitted on a
    different corpus with a different hand, and T-043 requires this one to be re-derived rather
    than inherited. ``source`` is required and free-form — it is what makes two datasets
    comparable or provably not (see the converter's module docstring on why a dataset-level fit
    is not dataset-INDEPENDENT).

    A ``NamedTuple`` for the same reason the converter's is one: it survives being loaded
    outside ``sys.modules``.
    """

    offset: float
    span: float
    source: str

    def apply(self, values: Any) -> np.ndarray:
        """``clip((x - offset) / span, 0, 1)``, float32."""
        if not self.span > 0.0:
            raise ValueError(f"HandSynergyAffine.span must be > 0, got {self.span}")
        arr = np.asarray(values, dtype=np.float32)
        return np.clip((arr - self.offset) / self.span, 0.0, 1.0).astype(np.float32)

    def clipped_frac(self, values: Any) -> float:
        """Fraction of ``values`` this affine would push outside [0, 1] BEFORE clipping.

        The converter refuses a pinned affine that clips anything at all, because a clipped
        sample is indistinguishable from a measurement in the written dataset and moves every
        clause of ``scripts/audit_gripper.py`` in the passing direction. Same reasoning applies
        to whatever affine T-043 fits here; this method is the measurement that gate needs.
        """
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if not self.span > 0.0 or arr.size == 0:
            return 0.0
        z = (arr - self.offset) / self.span
        return float(((z < 0.0) | (z > 1.0)).mean())


def hand_block_to_raw_synergy(vec28: Any, layout: G1Dex3Layout | None) -> np.ndarray:
    """Source state ``[..., 28]`` -> RAW per-hand grasp synergy ``[..., 2]`` = ``[left, right]``.

    "Raw" means the un-normalized mean of one hand's 7 joints, in SOURCE joint units — the
    signal a dataset-level affine has to be fitted on. Fitting on an already-mapped [0, 1]
    channel inherits exactly the attenuation the fit exists to undo (T-31).

    The mean is permutation-invariant, so :class:`HandJointOrder` does not change this output —
    but :class:`SideOrder` decides which column is the left hand, which it very much does
    change, and the ordering still has to be declared so it reaches the manifest.
    """
    resolved = require_layout(layout)
    _, hand14 = split_blocks(vec28)
    left, right = _side_slices(resolved)
    return np.stack(
        [hand14[..., left].mean(axis=-1), hand14[..., right].mean(axis=-1)], axis=-1
    ).astype(np.float32)


def to_canonical_gripper(
    vec28: Any, layout: G1Dex3Layout | None, affine: HandSynergyAffine | None
) -> np.ndarray:
    """Source state ``[..., 28]`` -> canonical gripper ``[..., 2]`` in [0, 1] (``[left, right]``).

    ``affine`` is REQUIRED and has no default. The [0, 1] mapping for this hand has not been
    fitted, and T-31's affine from ``GR00T-N1.7-AppleToPlate`` is explicitly not transferable:
    different hand, different range, and this corpus has TWO hand variants (``Unitree_G1`` at a
    120° mechanical limit, ``Unitree_G1_Dex3`` at 100°) whose poolability is itself an open
    question. Passing ``None`` raises rather than assuming a scale, because assuming a scale is
    what the legacy ``clip((mean+1)/2, 0, 1)`` mapping did and it silently railed a real signal.
    """
    resolved = require_layout(layout)
    raw = hand_block_to_raw_synergy(vec28, resolved)
    if affine is None:
        raise UnverifiedOrderingError(
            "to_canonical_gripper: no HandSynergyAffine given, and there is no default to fall "
            "back on. The canonical gripper is a scalar per hand in [0, 1]; turning this hand's "
            "7 joint angles into it needs a scale that has NOT been measured for the "
            "unitreerobotics/G1_Dex3_* corpus. Do not inherit T-31's AppleToPlate affine: "
            "different hand, different range, and this corpus has two hand variants (100° vs "
            "120° mechanical limit) that may not even be poolable. To settle it: fit "
            "clip((mean(hand_7dof) - offset) / span, 0, 1) over the whole conversion set from "
            "the action parquets (the converter's fit_hand_affine is the pattern), check it "
            "clips nothing with HandSynergyAffine.clipped_frac, verify with "
            "scripts/audit_gripper.py, and record it in the manifest's normalization slot. "
            "hand_block_to_raw_synergy() gives you the un-normalized signal to fit on."
        )
    return affine.apply(raw)


# -- one-shot conversion + manifest provenance ----------------------------------------------


@dataclass(frozen=True)
class CanonicalUpperBody:
    """One source vector mapped into the canonical space, with its absences attached.

    ``joint_validity`` is per-canonical-joint (``waist_yaw`` False) and is NOT the schema's
    group-level :class:`~wam.interfaces.schema.ValidityMask` — see
    :func:`canonical_joint_validity` for why the two are different objects.
    """

    q: np.ndarray  # [..., 15] float32, canonical order
    gripper: np.ndarray  # [..., 2] float32 in [0, 1]
    joint_validity: np.ndarray  # [15] bool
    layout: G1Dex3Layout
    waist_fill: float


def to_canonical(
    vec28: Any,
    layout: G1Dex3Layout | None,
    *,
    gripper_affine: HandSynergyAffine | None,
    waist_fill: float = WAIST_ABSENT_FILL,
) -> CanonicalUpperBody:
    """Source state ``[..., 28]`` -> canonical ``q`` + gripper + the validity of what is missing."""
    resolved = require_layout(layout)
    return CanonicalUpperBody(
        q=to_canonical_q(vec28, resolved, waist_fill=waist_fill),
        gripper=to_canonical_gripper(vec28, resolved, gripper_affine),
        joint_validity=canonical_joint_validity(),
        layout=resolved,
        waist_fill=float(waist_fill),
    )


def provenance(
    layout: G1Dex3Layout | None,
    *,
    gripper_affine: HandSynergyAffine | None = None,
    waist_fill: float = WAIST_ABSENT_FILL,
) -> dict[str, Any]:
    """Machine-readable mapping record for an episode manifest's ``extra['mapping']`` slot.

    Everything unverified is marked ``[?]`` and carries the measurement that would settle it,
    so a later reader can tell a decision from an assumption without re-reading this module.
    """
    resolved = require_layout(layout)
    record: dict[str, Any] = {
        "converter_module": __name__,
        "source_vector_dim": SOURCE_VECTOR_DIM,
        "block_order": {
            "value": f"arm-first: arm {ARM_BLOCK.start}:{ARM_BLOCK.stop}, "
            f"hand {HAND_BLOCK.start}:{HAND_BLOCK.stop}",
            "mark": "[OK]",
            "evidence": "measured 2026-08-15 over all 13 unitreerobotics/G1_Dex3_* sets — "
            "one-sided joint limits in meta/stats.json (0/14 vs 4-10/14, unanimous) and "
            "vla-training/groot/modality_g1_dex3.json:2-9; see T-043 §1",
        },
        "side_order": {
            "value": resolved.side_order.value,
            "mark": resolved.mark,
            "settled_by": _SETTLE_SIDE_ORDER,
            "evidence": resolved.evidence,
        },
        "hand_joint_order": {
            "value": resolved.hand_joint_order.value,
            "mark": resolved.mark,
            "settled_by": _SETTLE_HAND_ORDER,
            "evidence": resolved.evidence,
            "note": "the per-hand grasp synergy is a mean and therefore permutation-invariant, "
            "so this choice does not change today's canonical output — it is recorded because "
            "anything per-finger does depend on it",
        },
        "within_arm_joint_order": {
            "value": "assumed identical to the canonical proximal-to-distal order "
            "(side permutation only)",
            "mark": "[OK]" if WITHIN_ARM_ORDER_IS_MEASURED else "[?]",
            "settled_by": "correlate each source arm column against a joint whose motion is "
            "known (T-041's method); the fix, if any, is one permutation in "
            "arm_block_to_canonical_arms",
        },
        "waist_yaw": {
            "value": f"ABSENT in source; canonical slot {WAIST_YAW_CANONICAL_INDEX} filled with "
            f"{float(waist_fill)} and flagged invalid",
            "mark": "[OK]",
            "evidence": "the 28-dim vector is 7+7 arm + 7+7 hand with no waist and no leg "
            "columns (T-043 §2-3); JOINT_DELTA targets carry an exact 0.0 waist delta (hold)",
        },
        "dropped": ["waist_roll", "waist_pitch", "legs", "per-finger hand joints"],
        "gripper": {
            "value": "per-hand grasp synergy = mean of the 7 Dex3 joints, mapped to [0, 1] by a "
            "dataset-level affine",
            "mark": "[?]",
            "settled_by": "fit the affine over the whole conversion set from the action "
            "parquets and verify with scripts/audit_gripper.py; T-31's AppleToPlate affine is "
            "NOT transferable (different hand, different range, two hand variants at 100°/120°)",
        },
    }
    if gripper_affine is not None:
        record["gripper"]["affine"] = {
            "offset": gripper_affine.offset,
            "span": gripper_affine.span,
            "clip": [0.0, 1.0],
            "source": gripper_affine.source,
        }
    return record
