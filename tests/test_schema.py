"""Unit tests for wam.interfaces.schema (E0: shapes, normalization, validation flags)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from wam.interfaces.schema import (
    SCHEMA_VERSION,
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    IMUState,
    NormalizationSpec,
    RobotState,
    ValidityMask,
)

NUM_JOINTS = 7


@pytest.fixture()
def spec() -> CanonicalSpaceSpec:
    return CanonicalSpaceSpec(joint_names=tuple(f"j{i}" for i in range(NUM_JOINTS)))


def make_imu() -> IMUState:
    return IMUState(
        orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
        angular_velocity=np.zeros(3, dtype=np.float32),
        linear_acceleration=np.zeros(3, dtype=np.float32),
    )


def make_state(n: int = NUM_JOINTS) -> RobotState:
    return RobotState(
        timestamp_ns=1_000_000,
        q=np.zeros(n, dtype=np.float32),
        dq=np.zeros(n, dtype=np.float32),
        imu=make_imu(),
        gripper_state=np.zeros(1, dtype=np.float32),
    )


def make_chunk(t: int = 16, d: int = NUM_JOINTS) -> ActionChunk:
    return ActionChunk(
        mode=ActionMode.JOINT_DELTA,
        targets=np.zeros((t, d), dtype=np.float32),
        gripper_target=np.zeros(t, dtype=np.float32),
        dt_s=0.05,
    )


class TestCanonicalSpaceSpec:
    def test_num_joints_and_defaults(self, spec: CanonicalSpaceSpec) -> None:
        assert spec.num_joints == NUM_JOINTS
        assert spec.gripper_dims == 1
        assert spec.schema_version == SCHEMA_VERSION

    def test_duplicate_joint_names_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CanonicalSpaceSpec(joint_names=("a", "a"))

    def test_empty_joint_names_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CanonicalSpaceSpec(joint_names=())

    def test_target_dim_per_mode(self, spec: CanonicalSpaceSpec) -> None:
        assert spec.target_dim(ActionMode.JOINT_DELTA) == NUM_JOINTS
        assert spec.target_dim(ActionMode.EE_DELTA) == 7  # xyz + quat_wxyz


class TestRobotState:
    def test_valid_state_has_no_issues(self, spec: CanonicalSpaceSpec) -> None:
        assert make_state().validate(spec) == []
        assert make_state().validate() == []  # spec-less structural check

    def test_construction_does_not_raise_on_bad_values(self) -> None:
        # Schema stores, safety rejects: NaN must survive construction.
        s = make_state()
        s.q[0] = np.nan
        assert any("q" in i and "NaN" in i for i in s.validate())

    def test_wrong_joint_count_flagged(self, spec: CanonicalSpaceSpec) -> None:
        issues = make_state(n=NUM_JOINTS + 1).validate(spec)
        assert any("q" in i and "shape" in i for i in issues)

    def test_wrong_dtype_flagged(self, spec: CanonicalSpaceSpec) -> None:
        s = make_state()
        s.dq = s.dq.astype(np.float64)
        assert any("dq" in i and "float32" in i for i in s.validate(spec))

    def test_invalid_group_is_skipped(self, spec: CanonicalSpaceSpec) -> None:
        s = make_state()
        s.imu.orientation_wxyz = np.full(4, np.nan, dtype=np.float32)
        s.validity = ValidityMask(imu=False)
        assert s.validate(spec) == []  # flagged missing -> not checked

    def test_negative_timestamp_flagged(self) -> None:
        s = make_state()
        s.timestamp_ns = -1
        assert any("timestamp_ns" in i for i in s.validate())

    def test_validity_mask_as_dict(self) -> None:
        assert ValidityMask(dq=False).as_dict() == {
            "q": True,
            "dq": False,
            "imu": True,
            "gripper": True,
        }


class TestActionChunk:
    def test_valid_chunk(self, spec: CanonicalSpaceSpec) -> None:
        assert make_chunk().validate(spec) == []

    def test_duration_property(self) -> None:
        c = make_chunk(t=20)
        assert c.num_steps == 20
        assert c.duration == pytest.approx(20 * 0.05)

    def test_t_is_not_hard_coded(self, spec: CanonicalSpaceSpec) -> None:
        for t in (1, 8, 32, 64):
            assert make_chunk(t=t).validate(spec) == []

    def test_nan_targets_flagged_not_raised(self) -> None:
        c = make_chunk()
        c.targets[0, 0] = np.inf
        assert any("targets" in i and "NaN" in i for i in c.validate())

    def test_gripper_shape_mismatch_flagged(self) -> None:
        c = make_chunk(t=16)
        c.gripper_target = np.zeros(8, dtype=np.float32)
        assert any("gripper_target" in i and "shape" in i for i in c.validate())

    def test_gripper_range_flagged(self) -> None:
        c = make_chunk()
        c.gripper_target[0] = 1.5
        assert any("gripper_target" in i and "[0, 1]" in i for i in c.validate())

    def test_wrong_target_dim_for_mode_flagged(self, spec: CanonicalSpaceSpec) -> None:
        c = make_chunk(d=NUM_JOINTS + 2)
        assert any("targets" in i and "D=" in i for i in c.validate(spec))

    def test_bad_dt_flagged(self) -> None:
        c = make_chunk()
        c.dt_s = 0.0
        assert any("dt_s" in i for i in c.validate())

    def test_wrong_schema_major_flagged(self) -> None:
        c = make_chunk()
        c.schema_version = "1.0.0"
        assert any("schema_version" in i for i in c.validate())


class TestNormalizationSpec:
    def test_roundtrip_exact_float32(self) -> None:
        rng = np.random.default_rng(0)
        spec = NormalizationSpec(
            mean=tuple(rng.normal(size=NUM_JOINTS)),
            std=tuple(rng.uniform(0.1, 3.0, size=NUM_JOINTS)),
        )
        x = rng.normal(scale=2.0, size=(32, NUM_JOINTS)).astype(np.float32)
        z = spec.normalize(x)
        back = spec.denormalize(z).astype(np.float32)
        assert z.dtype == np.float64  # full precision until storage
        np.testing.assert_array_equal(back, x)  # bit-exact roundtrip contract

    def test_normalize_values(self) -> None:
        spec = NormalizationSpec(mean=(1.0, -2.0), std=(2.0, 4.0))
        z = spec.normalize(np.array([[3.0, 2.0]], dtype=np.float32))
        np.testing.assert_allclose(z, [[1.0, 1.0]])

    def test_dict_roundtrip_versioned(self) -> None:
        spec = NormalizationSpec(mean=(0.5,), std=(1.5,))
        d = spec.to_dict()
        assert d["version"] == SCHEMA_VERSION
        assert NormalizationSpec.from_dict(d) == spec

    def test_from_dict_rejects_incompatible_major(self) -> None:
        with pytest.raises(ValueError, match="incompatible"):
            NormalizationSpec.from_dict({"version": "1.0.0", "mean": [0.0], "std": [1.0]})

    def test_nonpositive_std_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NormalizationSpec(mean=(0.0,), std=(0.0,))

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NormalizationSpec(mean=(0.0, 0.0), std=(1.0,))

    def test_dim(self) -> None:
        assert NormalizationSpec(mean=(0.0,) * 5, std=(1.0,) * 5).dim == 5
