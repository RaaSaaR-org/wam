"""Unit tests for wam.safety (T-04, FR-07, PRD §11.2): config, layer, watchdog."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wam.interfaces import (
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    IMUState,
    RobotState,
    SafetyFilter,
    ValidityMask,
)
from wam.safety import SafetyConfig, SafetyLayer, Watchdog, WatchdogAction


def make_config(n: int = 1, **overrides: object) -> SafetyConfig:
    base: dict[str, object] = {
        "q_min": tuple([-10.0] * n),
        "q_max": tuple([10.0] * n),
        "dq_max": tuple([100.0] * n),
        "ddq_max": tuple([1e6] * n),
        "workspace_min": (0.0, 0.0, 0.0),
        "workspace_max": (1.0, 1.0, 1.0),
        "ee_max_lin_vel_m_s": 10.0,
        "ee_max_step_m": 1.0,
        "gripper_rate_max": 100.0,
        "chunk_timeout_s": 0.5,
    }
    base.update(overrides)
    return SafetyConfig(**base)  # type: ignore[arg-type]


def make_state(
    q: list[float],
    dq: list[float] | None = None,
    gripper: float = 0.0,
    validity: ValidityMask | None = None,
) -> RobotState:
    n = len(q)
    return RobotState(
        timestamp_ns=1_000,
        q=np.asarray(q, dtype=np.float32),
        dq=np.asarray(dq if dq is not None else [0.0] * n, dtype=np.float32),
        imu=IMUState(
            orientation_wxyz=np.asarray([1, 0, 0, 0], dtype=np.float32),
            angular_velocity=np.zeros(3, dtype=np.float32),
            linear_acceleration=np.zeros(3, dtype=np.float32),
        ),
        gripper_state=np.asarray([gripper], dtype=np.float32),
        validity=validity if validity is not None else ValidityMask(),
    )


def joint_chunk(
    deltas: list[list[float]], dt: float = 0.1, gripper: list[float] | None = None
) -> ActionChunk:
    t = len(deltas)
    return ActionChunk(
        mode=ActionMode.JOINT_DELTA,
        targets=np.asarray(deltas, dtype=np.float32),
        gripper_target=np.asarray(gripper if gripper is not None else [0.0] * t, dtype=np.float32),
        dt_s=dt,
    )


def ee_chunk(xyz: list[list[float]], dt: float = 0.1) -> ActionChunk:
    targets = [[*d, 1.0, 0.0, 0.0, 0.0] for d in xyz]
    return ActionChunk(
        mode=ActionMode.EE_DELTA,
        targets=np.asarray(targets, dtype=np.float32),
        gripper_target=np.zeros(len(xyz), dtype=np.float32),
        dt_s=dt,
    )


def kinds(interventions: list) -> list[str]:
    return [i.kind for i in interventions]


# --------------------------------------------------------------------- config


class TestSafetyConfig:
    def test_valid_and_num_joints(self) -> None:
        cfg = make_config(3)
        assert cfg.num_joints == 3
        assert cfg.timeout_policy == "hold"
        assert cfg.q_min_arr().dtype == np.float64

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_config(2, dq_max=(1.0,))

    def test_inverted_limits_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_config(1, q_min=(1.0,), q_max=(-1.0,))
        with pytest.raises(ValueError):
            make_config(1, workspace_min=(2.0, 0.0, 0.0))

    def test_nonpositive_rates_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_config(1, gripper_rate_max=0.0)
        with pytest.raises(ValueError):
            make_config(1, chunk_timeout_s=-1.0)
        with pytest.raises(ValueError):
            make_config(1, ddq_max=(0.0,))

    def test_bad_timeout_policy_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_config(1, timeout_policy="ignore")

    def test_yaml_roundtrip(self, tmp_path: Path) -> None:
        cfg = make_config(2, timeout_policy="stop", chunk_timeout_s=0.25)
        path = tmp_path / "safety.yaml"
        cfg.to_yaml(path)
        assert SafetyConfig.from_yaml(path) == cfg

    def test_from_yaml_text(self, tmp_path: Path) -> None:
        path = tmp_path / "s.yaml"
        path.write_text(
            "q_min: [-1.0]\nq_max: [1.0]\ndq_max: [2.0]\nddq_max: [10.0]\n"
            "workspace_min: [0.0, 0.0, 0.0]\nworkspace_max: [1.0, 1.0, 1.0]\n"
            "ee_max_lin_vel_m_s: 0.5\nee_max_step_m: 0.05\n"
            "gripper_rate_max: 2.0\nchunk_timeout_s: 0.5\n"
        )
        cfg = SafetyConfig.from_yaml(path)
        assert cfg.num_joints == 1
        assert cfg.dq_max == (2.0,)

    def test_from_yaml_non_mapping_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("- 1\n- 2\n")
        with pytest.raises(TypeError):
            SafetyConfig.from_yaml(path)


# ------------------------------------------------------------ layer: reject


class TestReject:
    def test_conforms_to_safety_filter_protocol(self) -> None:
        assert isinstance(SafetyLayer(make_config(1)), SafetyFilter)

    def test_nan_targets_rejected_to_hold(self) -> None:
        layer = SafetyLayer(make_config(2))
        state = make_state([0.0, 0.0], gripper=0.7)
        chunk = joint_chunk([[0.1, np.nan], [0.1, 0.1]])
        safe, iv = layer.filter(state, chunk)
        assert kinds(iv) == ["nan_reject"]
        assert safe.num_steps == 1
        assert np.array_equal(safe.targets, np.zeros((1, 2), dtype=np.float32))
        assert safe.gripper_target[0] == pytest.approx(0.7)
        assert safe.validate() == []

    def test_inf_gripper_rejected_to_hold(self) -> None:
        layer = SafetyLayer(make_config(1))
        chunk = joint_chunk([[0.1]], gripper=[np.inf])
        safe, iv = layer.filter(make_state([0.0]), chunk)
        assert kinds(iv) == ["nan_reject"]
        assert np.array_equal(safe.targets, np.zeros((1, 1), dtype=np.float32))

    def test_nan_dt_rejected_uses_hold_dt(self) -> None:
        layer = SafetyLayer(make_config(1, hold_dt_s=0.2))
        chunk = joint_chunk([[0.1]], dt=float("nan"))
        safe, iv = layer.filter(make_state([0.0]), chunk)
        assert kinds(iv) == ["nan_reject"]
        assert safe.dt_s == pytest.approx(0.2)

    def test_nonpositive_dt_schema_rejected(self) -> None:
        layer = SafetyLayer(make_config(1))
        safe, iv = layer.filter(make_state([0.0]), joint_chunk([[0.1]], dt=0.0))
        assert kinds(iv) == ["schema_reject"]
        assert safe.validate() == []

    def test_wrong_target_dim_schema_rejected(self) -> None:
        layer = SafetyLayer(make_config(2))
        safe, iv = layer.filter(make_state([0.0, 0.0]), joint_chunk([[0.1]]))
        assert kinds(iv) == ["schema_reject"]
        assert safe.targets.shape == (1, 2)

    def test_invalid_state_q_rejected(self) -> None:
        layer = SafetyLayer(make_config(1))
        state = make_state([0.0], validity=ValidityMask(q=False))
        _, iv = layer.filter(state, joint_chunk([[0.1]]))
        assert kinds(iv) == ["state_reject"]

    def test_ee_hold_chunk_is_identity_delta(self) -> None:
        layer = SafetyLayer(make_config(1))
        chunk = ee_chunk([[np.nan, 0.0, 0.0]])
        safe, iv = layer.filter(make_state([0.0]), chunk)
        assert kinds(iv) == ["nan_reject"]
        assert safe.mode is ActionMode.EE_DELTA
        expected = np.asarray([[0, 0, 0, 1, 0, 0, 0]], dtype=np.float32)
        assert np.array_equal(safe.targets, expected)


# ----------------------------------------------------------- layer: limits


class TestJointLimits:
    def test_clean_chunk_passes_unchanged(self) -> None:
        spec = CanonicalSpaceSpec(joint_names=("j0", "j1"))
        layer = SafetyLayer(make_config(2), spec=spec)
        chunk = joint_chunk([[0.01, -0.01], [0.02, 0.0]], gripper=[0.1, 0.2])
        safe, iv = layer.filter(make_state([0.0, 0.0], gripper=0.1), chunk)
        assert iv == []
        np.testing.assert_allclose(safe.targets, chunk.targets, atol=1e-7)
        np.testing.assert_allclose(safe.gripper_target, chunk.gripper_target, atol=1e-7)
        assert safe.mode is ActionMode.JOINT_DELTA
        assert safe.dt_s == chunk.dt_s

    def test_position_clamped_exactly_to_limit(self) -> None:
        cfg = make_config(1, q_min=(-0.5,), q_max=(0.5,))
        layer = SafetyLayer(cfg)
        chunk = joint_chunk([[0.3], [0.3], [0.3]])
        safe, iv = layer.filter(make_state([0.0]), chunk)
        q_final = float(np.sum(safe.targets.astype(np.float64)))
        assert q_final == pytest.approx(0.5, abs=1e-6)
        assert safe.targets[2, 0] == pytest.approx(0.0, abs=1e-7)
        assert kinds(iv).count("joint_limit") == 2  # steps 1 and 2 clamped
        # Never exceeds the limit at any intermediate step.
        cum = np.cumsum(safe.targets.astype(np.float64))
        assert np.all(cum <= 0.5 + 1e-6)

    def test_velocity_scaling_numerically_exact(self) -> None:
        cfg = make_config(1, dq_max=(1.0,))
        layer = SafetyLayer(cfg)
        chunk = joint_chunk([[0.5], [0.05]], dt=0.1)  # 5 rad/s then 0.5 rad/s
        safe, iv = layer.filter(make_state([0.0]), chunk)
        np.testing.assert_allclose(
            safe.targets, np.asarray([[0.1], [0.05]], dtype=np.float32), atol=1e-7
        )
        assert kinds(iv) == ["velocity_limit"]
        assert "step 0" in iv[0].detail

    def test_acceleration_limited_ramp(self) -> None:
        cfg = make_config(1, ddq_max=(10.0,))  # dv_max = 1 rad/s per 0.1 s step
        layer = SafetyLayer(cfg)
        chunk = joint_chunk([[0.5], [0.5]], dt=0.1)  # asks for 5 rad/s immediately
        safe, iv = layer.filter(make_state([0.0], dq=[0.0]), chunk)
        # v: 0 -> 1 -> 2 rad/s; deltas 0.1 then 0.2.
        np.testing.assert_allclose(
            safe.targets, np.asarray([[0.1], [0.2]], dtype=np.float32), atol=1e-6
        )
        assert kinds(iv) == ["accel_limit", "accel_limit"]

    def test_negative_direction_symmetric(self) -> None:
        cfg = make_config(1, dq_max=(1.0,))
        layer = SafetyLayer(cfg)
        safe, iv = layer.filter(make_state([0.0]), joint_chunk([[-0.5]], dt=0.1))
        assert safe.targets[0, 0] == pytest.approx(-0.1, abs=1e-7)
        assert kinds(iv) == ["velocity_limit"]

    def test_output_always_within_all_limits(self) -> None:
        cfg = make_config(
            2, q_min=(-0.2, -0.2), q_max=(0.2, 0.2), dq_max=(0.5, 0.5), ddq_max=(5.0, 5.0)
        )
        layer = SafetyLayer(cfg)
        rng = np.random.default_rng(0)
        deltas = (rng.standard_normal((16, 2)) * 0.5).astype(np.float32).tolist()
        state = make_state([0.1, -0.1], dq=[0.3, -0.3])
        safe, _ = layer.filter(state, joint_chunk(deltas, dt=0.05))
        d = safe.targets.astype(np.float64)
        dt = 0.05
        v = d / dt
        v_all = np.vstack([np.asarray([[0.3, -0.3]]), v])
        assert np.all(np.abs(v) <= 0.5 + 1e-6)  # velocity
        assert np.all(np.abs(np.diff(v_all, axis=0)) <= 5.0 * dt + 1e-6)  # acceleration
        q = np.asarray([0.1, -0.1]) + np.cumsum(d, axis=0)
        assert np.all(q <= 0.2 + 1e-6) and np.all(q >= -0.2 - 1e-6)  # position


class TestEELimits:
    def test_workspace_clamped_with_fk(self) -> None:
        layer = SafetyLayer(make_config(1), fk=lambda s: np.asarray([0.9, 0.5, 0.5]))
        safe, iv = layer.filter(make_state([0.0]), ee_chunk([[0.3, 0.0, 0.0]]))
        assert kinds(iv) == ["workspace"]
        np.testing.assert_allclose(safe.targets[0, :3], [0.1, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(safe.targets[0, 3:], [1, 0, 0, 0], atol=1e-7)

    def test_workspace_integration_across_steps(self) -> None:
        layer = SafetyLayer(make_config(1), fk=lambda s: np.asarray([0.8, 0.5, 0.5]))
        safe, iv = layer.filter(make_state([0.0]), ee_chunk([[0.15, 0.0, 0.0], [0.15, 0.0, 0.0]]))
        # 0.8 -> 0.95 (ok) -> 1.1 clamped to 1.0: second delta becomes 0.05.
        np.testing.assert_allclose(safe.targets[:, 0], [0.15, 0.05], atol=1e-6)
        assert kinds(iv) == ["workspace"]

    def test_no_fk_bounds_step_magnitude_and_flags_skip(self) -> None:
        layer = SafetyLayer(make_config(1, ee_max_step_m=0.05))
        safe, iv = layer.filter(make_state([0.0]), ee_chunk([[0.3, 0.0, 0.0]]))
        assert set(kinds(iv)) == {"ee_step_limit", "workspace_skipped"}
        assert float(np.linalg.norm(safe.targets[0, :3])) == pytest.approx(0.05, abs=1e-6)

    def test_no_fk_clean_chunk_still_flags_skip(self) -> None:
        layer = SafetyLayer(make_config(1))
        _, iv = layer.filter(make_state([0.0]), ee_chunk([[0.01, 0.0, 0.0]]))
        assert kinds(iv) == ["workspace_skipped"]

    def test_ee_linear_velocity_scaled(self) -> None:
        layer = SafetyLayer(
            make_config(1, ee_max_lin_vel_m_s=0.5), fk=lambda s: np.asarray([0.5, 0.5, 0.5])
        )
        safe, iv = layer.filter(make_state([0.0]), ee_chunk([[0.3, 0.4, 0.0]], dt=0.1))
        # |d| = 0.5 m in 0.1 s = 5 m/s -> scaled to 0.05 m, direction preserved.
        np.testing.assert_allclose(safe.targets[0, :3], [0.03, 0.04, 0.0], atol=1e-6)
        assert "velocity_limit" in kinds(iv)

    def test_broken_fk_falls_back_to_skip(self) -> None:
        def bad_fk(_: RobotState) -> np.ndarray:
            raise RuntimeError("fk unavailable")

        layer = SafetyLayer(make_config(1), fk=bad_fk)
        _, iv = layer.filter(make_state([0.0]), ee_chunk([[0.01, 0.0, 0.0]]))
        assert kinds(iv) == ["workspace_skipped"]


class TestOutOfLimitsStartRecovery:
    """A start state outside the limits must ramp back at legal speed, never snap."""

    def test_joint_start_beyond_q_max_recovers_at_velocity_limit(self) -> None:
        cfg = make_config(1, q_min=(-1.0,), q_max=(1.0,), dq_max=(2.0,), ddq_max=(1e6,))
        layer = SafetyLayer(cfg)
        dt = 0.02
        chunk = joint_chunk([[0.0], [0.0], [0.0]], dt=dt)
        safe, iv = layer.filter(make_state([1.3]), chunk)  # 0.3 rad beyond q_max
        d = safe.targets.astype(np.float64)
        # Every output step obeys the velocity limit — no 0.3 rad snap in one 20 ms step.
        assert np.all(np.abs(d) <= 2.0 * dt + 1e-9)
        assert d[0, 0] == pytest.approx(-2.0 * dt, abs=1e-9)
        assert "joint_limit_recovery" in kinds(iv)
        # q ramps monotonically back toward q_max.
        q = 1.3 + np.cumsum(d)
        assert np.all(np.diff(np.concatenate([[1.3], q])) <= 0.0)
        assert q[-1] == pytest.approx(1.3 - 3 * 2.0 * dt, abs=1e-6)  # float32 state input

    def test_joint_recovery_is_noop_inside_limits(self) -> None:
        cfg = make_config(1, q_min=(-0.5,), q_max=(0.5,), dq_max=(100.0,))
        layer = SafetyLayer(cfg)
        _, iv = layer.filter(make_state([0.0]), joint_chunk([[0.3], [0.3], [0.3]]))
        assert "joint_limit_recovery" not in kinds(iv)  # position clamp only shrinks steps

    def test_ee_start_outside_workspace_reenters_at_velocity_limit(self) -> None:
        # fk reports x = 1.2 m, AABB max x = 1.0: re-entry must be bounded by the EE
        # velocity limit, not a 0.2 m jump in one step.
        layer = SafetyLayer(
            make_config(1, ee_max_lin_vel_m_s=0.5),
            fk=lambda s: np.asarray([1.2, 0.5, 0.5]),
        )
        dt = 0.05
        max_step = 0.5 * dt
        safe, iv = layer.filter(
            make_state([0.0]), ee_chunk([[0.0, 0.0, 0.0]] * 3, dt=dt)
        )
        norms = np.linalg.norm(safe.targets[:, :3].astype(np.float64), axis=1)
        assert np.all(norms <= max_step + 1e-9)
        np.testing.assert_allclose(safe.targets[:, 0], [-max_step] * 3, atol=1e-9)
        assert "workspace_recovery" in kinds(iv)
        assert "workspace" in kinds(iv)

    def test_ee_recovery_is_noop_inside_workspace(self) -> None:
        layer = SafetyLayer(make_config(1), fk=lambda s: np.asarray([0.9, 0.5, 0.5]))
        _, iv = layer.filter(make_state([0.0]), ee_chunk([[0.3, 0.0, 0.0]]))
        assert "workspace_recovery" not in kinds(iv)  # clamp inside the AABB only shrinks


class TestGripper:
    def test_rate_limited_ramp(self) -> None:
        layer = SafetyLayer(make_config(1, gripper_rate_max=2.0))
        chunk = joint_chunk([[0.0], [0.0]], dt=0.1, gripper=[1.0, 1.0])
        safe, iv = layer.filter(make_state([0.0], gripper=0.0), chunk)
        np.testing.assert_allclose(safe.gripper_target, [0.2, 0.4], atol=1e-6)
        assert kinds(iv) == ["gripper_rate", "gripper_rate"]

    def test_out_of_range_clamped_then_rate_limited(self) -> None:
        layer = SafetyLayer(make_config(1, gripper_rate_max=2.0))
        chunk = joint_chunk([[0.0]], dt=0.1, gripper=[1.5])
        safe, iv = layer.filter(make_state([0.0], gripper=0.9), chunk)
        assert safe.gripper_target[0] == pytest.approx(1.0, abs=1e-6)
        assert kinds(iv) == ["gripper_range"]

    def test_invalid_gripper_state_skips_rate_anchor(self) -> None:
        layer = SafetyLayer(make_config(1, gripper_rate_max=2.0))
        state = make_state([0.0], gripper=0.0, validity=ValidityMask(gripper=False))
        chunk = joint_chunk([[0.0]], dt=0.1, gripper=[1.0])
        safe, iv = layer.filter(state, chunk)
        assert safe.gripper_target[0] == pytest.approx(1.0)
        assert iv == []


# ------------------------------------------------------------ layer: purity


class TestPurity:
    def test_inputs_never_mutated(self) -> None:
        layer = SafetyLayer(make_config(1, dq_max=(1.0,), gripper_rate_max=2.0))
        state = make_state([0.0], gripper=0.0)
        chunk = joint_chunk([[0.5], [0.5]], dt=0.1, gripper=[1.0, 1.0])
        targets_before = chunk.targets.copy()
        gripper_before = chunk.gripper_target.copy()
        q_before = state.q.copy()
        safe, iv = layer.filter(state, chunk)
        assert len(iv) > 0 and safe is not chunk
        assert np.array_equal(chunk.targets, targets_before)
        assert np.array_equal(chunk.gripper_target, gripper_before)
        assert np.array_equal(state.q, q_before)

    def test_deterministic_and_counter_monotonic(self) -> None:
        layer = SafetyLayer(make_config(1, dq_max=(1.0,)))
        state = make_state([0.0])
        chunk = joint_chunk([[0.5]], dt=0.1)
        safe1, iv1 = layer.filter(state, chunk)
        count_after_first = layer.intervention_count
        safe2, iv2 = layer.filter(state, chunk)
        assert np.array_equal(safe1.targets, safe2.targets)
        assert kinds(iv1) == kinds(iv2)
        assert count_after_first == len(iv1)
        assert layer.intervention_count == len(iv1) + len(iv2)

    def test_intervention_timestamps_from_state(self) -> None:
        layer = SafetyLayer(make_config(1, dq_max=(1.0,)))
        _, iv = layer.filter(make_state([0.0]), joint_chunk([[0.5]]))
        assert all(i.timestamp_ns == 1_000 for i in iv)


# ------------------------------------------------------------------ watchdog


class TestWatchdog:
    def test_unfed_is_expired_fail_safe(self) -> None:
        w = Watchdog(timeout_s=0.5)
        assert w.expired(0)
        assert w.decide(0) is WatchdogAction.HOLD
        iv = w.intervention(0)
        assert iv is not None and iv.kind == "watchdog_timeout"

    def test_feed_then_expiry_boundary(self) -> None:
        w = Watchdog(timeout_s=0.5)
        w.feed(1_000_000_000)
        assert not w.expired(1_400_000_000)
        assert not w.expired(1_500_000_000)  # exactly at deadline: not expired
        assert w.expired(1_500_000_001)
        assert w.decide(1_400_000_000) is None
        assert w.decide(1_600_000_000) is WatchdogAction.HOLD
        assert w.intervention(1_400_000_000) is None

    def test_refeed_rearms(self) -> None:
        w = Watchdog(timeout_s=0.1)
        w.feed(0)
        assert w.expired(200_000_000)
        w.feed(200_000_000)
        assert not w.expired(250_000_000)
        assert w.last_feed_ns == 200_000_000

    def test_stop_policy_and_from_config(self) -> None:
        cfg = make_config(1, chunk_timeout_s=0.25, timeout_policy="stop")
        w = Watchdog.from_config(cfg)
        assert w.action is WatchdogAction.STOP
        assert w.timeout_ns == 250_000_000
        w.feed(0)
        assert w.decide(300_000_000) is WatchdogAction.STOP

    def test_nonpositive_timeout_rejected(self) -> None:
        with pytest.raises(ValueError):
            Watchdog(timeout_s=0.0)
