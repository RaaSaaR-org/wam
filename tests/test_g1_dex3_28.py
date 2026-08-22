"""Tests for the G1 + Dex3 28-dim <-> canonical mapping (T-043).

The point of this module is what it REFUSES to do, so most of these tests assert refusals.
The block split is measured and is tested as a round-trip; everything inside the blocks is
unverified and is tested to raise rather than to silently pick an ordering.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from wam.interfaces.schema import ActionMode, CanonicalSpaceSpec
from wam.robot.g1 import G1_SPEC
from wam.robot.g1_dex3_28 import (
    ARM_BLOCK,
    CANONICAL_LEFT_ARM,
    CANONICAL_RIGHT_ARM,
    HAND_BLOCK,
    JOINTS_PER_ARM,
    JOINTS_PER_HAND,
    NUM_CANONICAL_GRIPPERS,
    NUM_CANONICAL_JOINTS,
    SOURCE_VECTOR_DIM,
    WAIST_ABSENT_FILL,
    WAIST_YAW_CANONICAL_INDEX,
    WITHIN_ARM_ORDER_IS_MEASURED,
    G1Dex3Layout,
    HandJointOrder,
    HandSynergyAffine,
    SideOrder,
    UnverifiedOrderingError,
    arm_block_to_canonical_arms,
    canonical_arms_to_arm_block,
    canonical_joint_validity,
    canonical_q_to_arm_block,
    hand_block_to_raw_synergy,
    hand_joint_names,
    join_blocks,
    provenance,
    require_layout,
    split_blocks,
    to_canonical,
    to_canonical_gripper,
    to_canonical_joint_delta,
    to_canonical_q,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

# A layout is a DECLARED assumption, never a default — every test that needs one builds it
# explicitly and says so, which is exactly what production callers have to do.
ASSUMED = G1Dex3Layout(
    side_order=SideOrder.LEFT_FIRST,
    hand_joint_order=HandJointOrder.NVIDIA_ASYMMETRIC,
    measured=False,
    evidence="test fixture — a working assumption, not a measurement",
)
ASSUMED_RIGHT_FIRST = ASSUMED.model_copy(update={"side_order": SideOrder.RIGHT_FIRST})

AFFINE = HandSynergyAffine(offset=0.0, span=1.0, source="test fixture, not a fitted affine")


def _ramp() -> np.ndarray:
    """A 28-vector whose every entry is its own index — transpositions are visible by eye."""
    return np.arange(SOURCE_VECTOR_DIM, dtype=np.float32)


# -- the verified part: block boundaries and their round-trip -------------------------------


def test_block_boundaries_are_the_measured_arm_first_split() -> None:
    """[OK] 2026-08-15, 13/13 unitreerobotics/G1_Dex3_* sets. Hand-first is a DIFFERENT corpus.

    This assertion is the whole reason the module exists: arm and hand transposed produces a
    model that trains, converges and moves the wrong joints.
    """
    assert (ARM_BLOCK.start, ARM_BLOCK.stop) == (0, 14)
    assert (HAND_BLOCK.start, HAND_BLOCK.stop) == (14, 28)
    assert JOINTS_PER_ARM == JOINTS_PER_HAND == 7
    assert 2 * JOINTS_PER_ARM + 2 * JOINTS_PER_HAND == SOURCE_VECTOR_DIM


def test_split_blocks_round_trips_and_needs_no_layout() -> None:
    vec = _ramp()
    arm, hand = split_blocks(vec)
    assert arm.shape == (14,) and hand.shape == (14,)
    assert list(arm) == list(range(0, 14))
    assert list(hand) == list(range(14, 28))
    np.testing.assert_array_equal(join_blocks(arm, hand), vec)


def test_split_blocks_round_trips_batched() -> None:
    rng = np.random.default_rng(0)
    batch = rng.normal(size=(5, 3, SOURCE_VECTOR_DIM)).astype(np.float32)
    arm, hand = split_blocks(batch)
    assert arm.shape == (5, 3, 14) and hand.shape == (5, 3, 14)
    np.testing.assert_array_equal(join_blocks(arm, hand), batch)


def test_split_blocks_does_not_alias_the_input() -> None:
    """A view would let a caller's edit reach back into the source array."""
    vec = _ramp()
    arm, _ = split_blocks(vec)
    arm[0] = 999.0
    assert vec[0] == 0.0


def test_arm_block_round_trips_under_both_side_orders() -> None:
    rng = np.random.default_rng(1)
    arm = rng.normal(size=(4, 14)).astype(np.float32)
    for layout in (ASSUMED, ASSUMED_RIGHT_FIRST):
        canonical = arm_block_to_canonical_arms(arm, layout)
        np.testing.assert_array_equal(canonical_arms_to_arm_block(canonical, layout), arm)


def test_side_order_actually_swaps_the_arms() -> None:
    """The parameter must not be decorative: the two declarations differ, and by a full limb."""
    arm = np.arange(14, dtype=np.float32)
    left_first = arm_block_to_canonical_arms(arm, ASSUMED)
    right_first = arm_block_to_canonical_arms(arm, ASSUMED_RIGHT_FIRST)
    assert list(left_first) == list(range(14))
    assert list(right_first) == list(range(7, 14)) + list(range(7))
    assert not np.array_equal(left_first, right_first)


# -- the unverified parts: refusing to guess ------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda: require_layout(None),
        lambda: arm_block_to_canonical_arms(np.zeros(14, dtype=np.float32), None),
        lambda: canonical_arms_to_arm_block(np.zeros(14, dtype=np.float32), None),
        lambda: to_canonical_q(np.zeros(28, dtype=np.float32), None),
        lambda: to_canonical_joint_delta(np.zeros(28, dtype=np.float32), None),
        lambda: canonical_q_to_arm_block(np.zeros(15, dtype=np.float32), None),
        lambda: hand_block_to_raw_synergy(np.zeros(28, dtype=np.float32), None),
        lambda: to_canonical_gripper(np.zeros(28, dtype=np.float32), None, AFFINE),
        lambda: to_canonical(np.zeros(28, dtype=np.float32), None, gripper_affine=AFFINE),
        lambda: hand_joint_names(None, "left"),
        lambda: provenance(None),
    ],
)
def test_every_conversion_refuses_an_unspecified_ordering(call) -> None:
    with pytest.raises(UnverifiedOrderingError):
        call()


def test_the_refusal_names_what_must_be_measured() -> None:
    """A refusal that does not say how to lift itself just gets worked around."""
    with pytest.raises(UnverifiedOrderingError) as exc:
        require_layout(None)
    msg = str(exc.value)
    assert "left/right" in msg
    assert "intra-hand" in msg
    assert "correlate" in msg
    assert "parquet" in msg  # names the blocker: the action parquets were never fetched
    assert "arm-first" in msg  # what IS measured is stated, so the refusal is scoped


def test_unverified_ordering_error_is_a_value_error() -> None:
    """Callers that only catch ValueError must still stop rather than continue with a guess."""
    assert issubclass(UnverifiedOrderingError, ValueError)


def test_layout_has_no_defaults_for_either_unverified_field() -> None:
    """A default here would be a guess wearing the clothes of a decision."""
    with pytest.raises(ValidationError):
        G1Dex3Layout()
    with pytest.raises(ValidationError):
        G1Dex3Layout(side_order=SideOrder.LEFT_FIRST)
    with pytest.raises(ValidationError):
        G1Dex3Layout(hand_joint_order=HandJointOrder.NVIDIA_ASYMMETRIC)
    with pytest.raises(ValidationError):
        G1Dex3Layout(side_order="left", hand_joint_order=HandJointOrder.NVIDIA_ASYMMETRIC)


def test_layout_defaults_to_unmeasured_and_marks_itself_so() -> None:
    assert ASSUMED.measured is False
    assert ASSUMED.mark == "[?]"
    measured = ASSUMED.model_copy(update={"measured": True, "evidence": "correlated vs parquet"})
    assert measured.mark == "[OK]"


def test_all_three_recorded_hand_orderings_are_offered_and_none_is_default() -> None:
    """Three mutually inconsistent orderings are on record; the module names all three."""
    assert {o.value for o in HandJointOrder} == {
        "corpus_card_thumb_first_symmetric",
        "arena_index_first",
        "nvidia_asymmetric",
    }


def test_hand_joint_names_refuses_the_two_description_only_orderings() -> None:
    """'thumb-first symmetric' does not say whether index precedes middle. Do not invent it."""
    for order in (
        HandJointOrder.CORPUS_CARD_THUMB_FIRST_SYMMETRIC,
        HandJointOrder.ARENA_INDEX_FIRST,
    ):
        layout = ASSUMED.model_copy(update={"hand_joint_order": order})
        with pytest.raises(UnverifiedOrderingError) as exc:
            hand_joint_names(layout, "left")
        assert "DESCRIPTION" in str(exc.value)


def test_hand_joint_names_returns_the_one_fully_specified_ordering_and_it_is_asymmetric() -> None:
    left = hand_joint_names(ASSUMED, "left")
    right = hand_joint_names(ASSUMED, "right")
    assert len(left) == len(right) == JOINTS_PER_HAND
    # docs/contracts/vla-training-consumer.md §2.4:232-233 — left is middle-before-index, right is
    # index-before-middle. Symmetrically "repairing" this is a documented mistake.
    assert left == ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
    assert right == ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
    assert left != right


def test_contract_hand_order_citation_still_points_at_the_hand_table() -> None:
    """The one automated reader of the consumer contract.

    ``g1_dex3_28`` cites ``docs/contracts/vla-training-consumer.md`` by LINE NUMBER for the
    only fully specified hand ordering on record. Nothing parsed that document until this test,
    which is exactly why ``PR-08`` §8 item 2 could describe the wrong dataset for a month without
    anything going red (``T40_RULE_V7`` §6). A citation that drifts sends the next reader to the
    wrong lines, and the failure this whole module exists to prevent is a silently transposed
    finger pair. So: follow the citation, and check it lands where it claims.
    """
    module_src = (_REPO_ROOT / "src" / "wam" / "robot" / "g1_dex3_28.py").read_text()
    cites = re.findall(
        r"vla-training-consumer\.md\s*§2\.4:(\d+)-(\d+)",
        module_src,
    )
    assert cites, "g1_dex3_28.py no longer cites the contract's §2.4 hand table by line"
    assert len(set(cites)) == 1, f"the module cites §2.4 at disagreeing lines: {set(cites)}"

    first, last = (int(n) for n in cites[0])
    contract = (_REPO_ROOT / "docs" / "contracts" / "vla-training-consumer.md").read_text()
    lines = contract.splitlines()
    cited = "\n".join(lines[first - 1 : last])

    # The asymmetry is the point: symmetrically "repairing" it is a documented mistake.
    left, right = hand_joint_names(ASSUMED, "left"), hand_joint_names(ASSUMED, "right")
    assert "**state** `left_hand`" in cited and "**state** `right_hand`" in cited
    for name in left:
        assert name in cited, f"{name!r} missing from the cited contract lines {first}-{last}"
    for name in right:
        assert name in cited
    assert cited.index("middle_0, middle_1") < cited.index("index_0, index_1, middle_0")

    # And the ordering is ATTESTED, not MEASURED, for this corpus too — the contract must keep
    # saying so. If somebody measures it against the parquet, this assertion is the place to
    # record that, together with HandJointOrder's own mark.
    assert "ATTESTED, not MEASURED" in contract
    assert provenance(ASSUMED)["hand_joint_order"]["mark"] == "[?]"


def test_within_arm_order_is_flagged_unmeasured() -> None:
    """The block order was measured; the joint order inside one arm was not. Say so."""
    assert WITHIN_ARM_ORDER_IS_MEASURED is False
    assert provenance(ASSUMED)["within_arm_joint_order"]["mark"] == "[?]"


# -- waist: absent, not fabricated ----------------------------------------------------------


def test_canonical_waist_is_flagged_invalid_not_measured() -> None:
    valid = canonical_joint_validity()
    assert valid.shape == (NUM_CANONICAL_JOINTS,)
    assert valid.dtype == np.bool_
    assert valid[WAIST_YAW_CANONICAL_INDEX] == False  # noqa: E712 — numpy bool, not `is False`
    assert valid.sum() == NUM_CANONICAL_JOINTS - 1  # every arm joint stays valid


def test_to_canonical_q_fills_the_absent_waist_with_a_flagged_placeholder() -> None:
    q = to_canonical_q(_ramp(), ASSUMED)
    assert q.shape == (NUM_CANONICAL_JOINTS,)
    assert q[WAIST_YAW_CANONICAL_INDEX] == WAIST_ABSENT_FILL
    assert not canonical_joint_validity()[WAIST_YAW_CANONICAL_INDEX]
    # The arm block is carried through unchanged (left first under this layout).
    assert list(q[CANONICAL_LEFT_ARM]) == list(range(0, 7))
    assert list(q[CANONICAL_RIGHT_ARM]) == list(range(7, 14))


def test_joint_delta_waist_is_exactly_zero_regardless_of_fill() -> None:
    """A delta is a command. A non-zero waist delta would move a joint nobody recorded."""
    delta = to_canonical_joint_delta(np.full(SOURCE_VECTOR_DIM, 0.5, dtype=np.float32), ASSUMED)
    assert delta[WAIST_YAW_CANONICAL_INDEX] == 0.0
    assert np.all(delta[1:] == 0.5)


def test_hand_block_never_leaks_into_canonical_q() -> None:
    """The whole arm/hand transposition failure mode, asserted directly."""
    vec = np.zeros(SOURCE_VECTOR_DIM, dtype=np.float32)
    vec[HAND_BLOCK] = 7.0
    q = to_canonical_q(vec, ASSUMED)
    assert np.all(q == 0.0)


def test_canonical_q_to_arm_block_refuses_to_drop_a_commanded_waist() -> None:
    q = np.zeros(NUM_CANONICAL_JOINTS, dtype=np.float32)
    q[WAIST_YAW_CANONICAL_INDEX] = 0.3
    with pytest.raises(ValueError, match="no waist column|NO waist column"):
        canonical_q_to_arm_block(q, ASSUMED)
    arm = canonical_q_to_arm_block(q, ASSUMED, require_waist_zero=False)
    assert arm.shape == (14,)


def test_canonical_q_round_trips_through_the_arm_block() -> None:
    rng = np.random.default_rng(2)
    vec = rng.normal(size=SOURCE_VECTOR_DIM).astype(np.float32)
    vec[HAND_BLOCK] = 0.0  # the hand block is not representable in canonical q by design
    for layout in (ASSUMED, ASSUMED_RIGHT_FIRST):
        q = to_canonical_q(vec, layout)
        arm = canonical_q_to_arm_block(q, layout, require_waist_zero=False)
        np.testing.assert_allclose(arm, vec[ARM_BLOCK])


def test_the_absent_waist_is_flagged_per_joint_and_only_on_the_waist() -> None:
    """Documented contract: the schema's ``ValidityMask`` is group-level, so ``q`` stays valid.

    Flipping that group flag False would discard 14 measured arm joints to describe one absent
    one; the per-joint array plus the manifest record is what carries the absence instead. So
    what has to hold is that the per-joint array marks the waist and nothing else.
    """
    validity = canonical_joint_validity()
    # The absence is carried per-joint, on exactly one index...
    assert validity[WAIST_YAW_CANONICAL_INDEX] == False  # noqa: E712
    # ...and on no other: every arm joint the corpus really measured stays valid. THIS is the
    # assertion that fails if a future change flags the whole q group instead of the one joint.
    assert int(np.count_nonzero(validity)) == len(validity) - 1


# -- gripper: no inherited scale ------------------------------------------------------------


def test_gripper_refuses_to_invent_a_scale() -> None:
    with pytest.raises(UnverifiedOrderingError) as exc:
        to_canonical_gripper(_ramp(), ASSUMED, None)
    msg = str(exc.value)
    assert "audit_gripper" in msg
    assert "100" in msg and "120" in msg  # the two hand variants are named


def test_raw_synergy_is_unnormalized_and_side_aware() -> None:
    vec = np.zeros(SOURCE_VECTOR_DIM, dtype=np.float32)
    vec[14:21] = 0.4  # first hand in the block
    vec[21:28] = 0.9  # second hand in the block
    left_first = hand_block_to_raw_synergy(vec, ASSUMED)
    right_first = hand_block_to_raw_synergy(vec, ASSUMED_RIGHT_FIRST)
    np.testing.assert_allclose(left_first, [0.4, 0.9], rtol=1e-6)
    np.testing.assert_allclose(right_first, [0.9, 0.4], rtol=1e-6)


def test_raw_synergy_is_permutation_invariant_within_a_hand() -> None:
    """Why HandJointOrder does not change today's output — and is still required, to be recorded."""
    rng = np.random.default_rng(3)
    vec = np.zeros(SOURCE_VECTOR_DIM, dtype=np.float32)
    vec[HAND_BLOCK] = rng.normal(size=14).astype(np.float32)
    shuffled = vec.copy()
    shuffled[14:21] = vec[14:21][::-1]
    np.testing.assert_allclose(
        hand_block_to_raw_synergy(vec, ASSUMED),
        hand_block_to_raw_synergy(shuffled, ASSUMED),
        rtol=1e-6,
    )


def test_affine_maps_into_the_unit_interval_and_reports_its_own_clipping() -> None:
    affine = HandSynergyAffine(offset=0.0, span=1.0, source="unit test")
    np.testing.assert_allclose(affine.apply(np.array([0.0, 0.5, 1.0])), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(affine.apply(np.array([-1.0, 2.0])), [0.0, 1.0])
    assert affine.clipped_frac(np.array([0.0, 0.5, 1.0])) == 0.0
    assert affine.clipped_frac(np.array([-1.0, 0.5, 2.0])) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        HandSynergyAffine(offset=0.0, span=0.0, source="unit test").apply(np.zeros(3))


# -- shape / dtype contracts ----------------------------------------------------------------


@pytest.mark.parametrize("bad", [np.zeros(27), np.zeros(43), np.zeros((4, 14)), np.float32(1.0)])
def test_source_width_is_enforced(bad) -> None:
    with pytest.raises(ValueError, match="expected last dim 28"):
        split_blocks(bad)


def test_canonical_width_is_enforced() -> None:
    with pytest.raises(ValueError, match="expected last dim 15"):
        canonical_q_to_arm_block(np.zeros(14, dtype=np.float32), ASSUMED)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int32])
def test_outputs_are_float32_whatever_goes_in(dtype) -> None:
    vec = np.arange(SOURCE_VECTOR_DIM, dtype=dtype)
    out = to_canonical(vec, ASSUMED, gripper_affine=AFFINE)
    assert out.q.dtype == np.float32
    assert out.gripper.dtype == np.float32
    assert to_canonical_joint_delta(vec, ASSUMED).dtype == np.float32
    assert split_blocks(vec)[0].dtype == np.float32


def test_to_canonical_batches_and_keeps_the_joint_validity_unbatched() -> None:
    rng = np.random.default_rng(4)
    batch = rng.uniform(size=(6, SOURCE_VECTOR_DIM)).astype(np.float32)
    out = to_canonical(batch, ASSUMED, gripper_affine=AFFINE)
    assert out.q.shape == (6, NUM_CANONICAL_JOINTS)
    assert out.gripper.shape == (6, NUM_CANONICAL_GRIPPERS)
    assert out.joint_validity.shape == (NUM_CANONICAL_JOINTS,)
    assert (out.gripper >= 0.0).all() and (out.gripper <= 1.0).all()


# -- agreement with the canonical space ------------------------------------------------------


def test_canonical_geometry_is_derived_from_g1_spec_not_restated() -> None:
    assert NUM_CANONICAL_JOINTS == G1_SPEC.num_joints == 15
    assert NUM_CANONICAL_GRIPPERS == G1_SPEC.gripper_dims == 2
    assert G1_SPEC.joint_names[WAIST_YAW_CANONICAL_INDEX] == "waist_yaw"
    assert G1_SPEC.joint_names[CANONICAL_LEFT_ARM] == tuple(
        n for n in G1_SPEC.joint_names if n.startswith("left_")
    )
    assert G1_SPEC.joint_names[CANONICAL_RIGHT_ARM] == tuple(
        n for n in G1_SPEC.joint_names if n.startswith("right_")
    )
    # 15 = waist_yaw + 7 + 7, and the corpus supplies only the 14.
    assert 1 + 2 * JOINTS_PER_ARM == NUM_CANONICAL_JOINTS


def test_canonical_q_width_matches_the_joint_delta_target_dim() -> None:
    """What comes out of here must be a legal ActionChunk target row for this spec."""
    spec = CanonicalSpaceSpec(joint_names=G1_SPEC.joint_names, gripper_dims=G1_SPEC.gripper_dims)
    delta = to_canonical_joint_delta(_ramp(), ASSUMED)
    assert delta.shape[-1] == spec.target_dim(ActionMode.JOINT_DELTA)


def test_provenance_records_every_unverified_choice_with_its_mark() -> None:
    record = provenance(ASSUMED, gripper_affine=AFFINE)
    assert record["block_order"]["mark"] == "[OK]"
    assert record["side_order"]["mark"] == "[?]"
    assert record["hand_joint_order"]["mark"] == "[?]"
    assert record["waist_yaw"]["mark"] == "[OK]"  # the ABSENCE is what is measured
    assert "settled_by" in record["side_order"]
    assert "settled_by" in record["hand_joint_order"]
    assert record["gripper"]["affine"]["source"] == AFFINE.source
    assert "waist_roll" in record["dropped"]


def test_provenance_upgrades_the_mark_once_the_layout_is_measured() -> None:
    measured = ASSUMED.model_copy(
        update={"measured": True, "evidence": "correlated against the parquet, T-043 AC-1/2"}
    )
    record = provenance(measured)
    assert record["side_order"]["mark"] == "[OK]"
    assert record["side_order"]["evidence"].startswith("correlated")
