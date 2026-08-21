"""Tests for the Isaac Sim binding seam (FR-06, E2).

Everything here runs on a Mac with no Isaac Sim, no GPU and no CUDA — which is the whole
point. ``IsaacSimBinding`` was written against NVIDIA's documentation and CANNOT be executed
here; ``scripts/preflight_isaac.py`` is what tests it, on the box. What CAN be tested here is
the half that the rest of the backend actually depends on:

1. :class:`FakeIsaacBinding`'s own contract — the tick is an exact ``int`` and advances by
   exactly N, a render never advances it, gains round-trip, ``kp = 0`` is accepted. The
   transport's staleness detection and its e-stop drain are built ON these properties, so a
   fake that got them wrong would make every test above it meaningless.
2. The name resolution, which is shared code and is the failure this seam fears most (a
   permuted joint map moves a physical arm silently).
3. The torch-free promise, in a subprocess, because ``isaac_transport.py`` has to run in a
   venv where torch 2.13 does not exist.

Style note: every assertion here is on the CONTRACT, never on the fake's integrator
constants. The integrator is a caricature and is allowed to change; "the tick advanced by
exactly 7" is not.
"""

from __future__ import annotations

import importlib.util
import numbers
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wam.robot import isaac_binding  # for monkeypatching module-level naming candidates
from wam.robot.g1_transport import DEX3_FINGER_JOINTS, G1_MOTOR_JOINT_NAMES
from wam.robot.isaac_binding import (
    BODY_NAME_CANDIDATES,
    DEFAULT_ASSET_SUBPATH,
    EFFORT_GETTER_CANDIDATES,
    EXPECTED_NUM_DOFS,
    FINGER_NAME_CANDIDATES,
    GROUND_TRUTH_ANNOTATORS,
    FakeIsaacBinding,
    IsaacBinding,
    IsaacSimBinding,
    SegmentationFrame,
    _batch,
    _row,
    _to_numpy,
    fake_g1_dof_names,
    resolve_g1_dof_indices,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_preflight():
    """Load ``scripts/preflight_isaac.py`` as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "preflight_isaac", _REPO_ROOT / "scripts" / "preflight_isaac.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["preflight_isaac"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake() -> FakeIsaacBinding:
    return FakeIsaacBinding()


def _flat(binding: FakeIsaacBinding, value: float) -> np.ndarray:
    return np.full(binding.num_dofs, value, dtype=np.float32)


# -- the tick ------------------------------------------------------------------------------


def test_the_tick_is_an_exact_int_not_a_float_and_not_a_numpy_integer(
    fake: FakeIsaacBinding,
) -> None:
    """``G1Adapter.read_state`` decides staleness by EQUALITY against the previous tick.

    ``type(...) is int`` rather than ``isinstance``: ``bool`` and every ``np.integer`` pass an
    isinstance check, and a float that happens to be integral passes none of the obvious ones
    while still making equality meaningless a few million steps in.
    """
    assert type(fake.get_physics_step_count()) is int
    fake.step(3)
    assert type(fake.get_physics_step_count()) is int


@pytest.mark.parametrize("steps", [0, 1, 7, 33])
def test_the_tick_advances_by_exactly_the_number_of_steps(
    fake: FakeIsaacBinding, steps: int
) -> None:
    """Exactly N, not "at least N": a tick that advanced by a variable amount would make the
    watchdog fire at random, and step(0) must freeze it completely."""
    before = fake.get_physics_step_count()
    fake.step(steps)
    assert fake.get_physics_step_count() - before == steps


def test_the_tick_never_advances_on_a_read_or_a_write(fake: FakeIsaacBinding) -> None:
    """Only physics moves the clock. A getter or a gain write that ticked would hide a
    stalled simulation behind a healthy-looking staleness check."""
    fake.step(2)
    before = fake.get_physics_step_count()
    fake.get_dof_positions()
    fake.get_dof_velocities()
    fake.get_dof_efforts()
    fake.set_dof_position_targets(_flat(fake, 0.1))
    fake.set_dof_gains(_flat(fake, 10.0), _flat(fake, 1.0))
    fake.get_dof_gains()
    assert fake.get_physics_step_count() == before


def test_a_negative_step_count_is_refused(fake: FakeIsaacBinding) -> None:
    with pytest.raises(ValueError, match="steps must be >= 0"):
        fake.step(-1)


# -- rendering -----------------------------------------------------------------------------


def test_render_does_not_advance_the_tick(fake: FakeIsaacBinding) -> None:
    """The guarantee the whole render path rests on: the adapter owns the clock.

    A render that stepped behind its back would corrupt staleness detection AND silently
    widen the ``dq_max * dt`` clip, which is only a velocity limit if dt is what we think.
    """
    fake.step(5)
    before = fake.get_physics_step_count()
    for _ in range(4):
        assert fake.render_frame("persp") is not None
    assert fake.get_physics_step_count() == before


def test_a_frame_is_uint8_hw3_and_not_blank() -> None:
    """A uniform frame would pass a shape check and poison a dataset silently, so the fake
    must produce real variance — otherwise a downstream "frame is not blank" gate cannot
    fail against it either."""
    binding = FakeIsaacBinding(render_hw=(32, 48))
    frame = binding.render_frame("persp")
    assert frame is not None
    assert frame.dtype == np.uint8
    assert frame.shape == (32, 48, 3)
    assert float(frame.astype(np.float64).std()) > 1.0


def test_the_frame_changes_when_the_robot_moves(fake: FakeIsaacBinding) -> None:
    """Otherwise "the image changed after execute()" is an assertion that cannot fail."""
    first = fake.render_frame("persp")
    fake.set_dof_gains(_flat(fake, 400.0), _flat(fake, 20.0))
    fake.set_dof_position_targets(_flat(fake, 0.5))
    fake.step(20)
    assert not np.array_equal(first, fake.render_frame("persp"))


def test_the_warmup_returns_none_before_it_returns_a_frame() -> None:
    """The first frames come back empty — up to 20 of them in NVIDIA's own test. Code that
    does not gate on ``is not None`` records black frames, which pass the T-11 data-quality
    gates. The fake reproduces the warmup so that gate can be tested."""
    binding = FakeIsaacBinding(warmup_frames=3)
    assert [binding.render_frame("persp") for _ in range(3)] == [None, None, None]
    assert binding.render_frame("persp") is not None


def test_an_unknown_camera_is_refused(fake: FakeIsaacBinding) -> None:
    with pytest.raises(ValueError, match="unknown camera"):
        fake.render_frame("wrist_left")


# -- ground truth (PR-08 §4: the estimator error budget) ------------------------------------


def test_ground_truth_is_off_by_default_and_asking_anyway_raises(fake: FakeIsaacBinding) -> None:
    """The default binding is exactly the binding that existed before: rgb and nothing else.

    Refusing is the whole design. A binding with no depth annotator has no depth, and the
    tempting alternative — hand back zeros, or a plausible constant — puts a fabricated
    distance into ``EST_DRIFT_P95``, which is a number a gate is computed from. Nothing
    downstream could tell it from a measurement.
    """
    assert fake.ground_truth_channels == ()
    assert fake.render_frame("persp") is not None
    for call in (fake.render_depth, fake.render_segmentation):
        with pytest.raises(RuntimeError, match="not attached"):
            call("persp")


def test_the_refusal_names_the_knob_that_fixes_it(fake: FakeIsaacBinding) -> None:
    with pytest.raises(RuntimeError) as excinfo:
        fake.render_depth("persp")
    message = str(excinfo.value)
    assert "ground_truth=('depth',)" in message
    assert "distance_to_camera" in message  # the vendor name, so a preflight report is greppable


def test_only_the_requested_channels_are_attached() -> None:
    """Asking for depth must not quietly enable segmentation: the rig pays per AOV per frame,
    and a channel nobody asked for is a cost nobody budgeted."""
    binding = FakeIsaacBinding(ground_truth=("depth",))
    assert binding.ground_truth_channels == ("depth",)
    assert binding.render_depth("persp") is not None
    with pytest.raises(RuntimeError, match="not attached"):
        binding.render_segmentation("persp")


def test_an_unknown_ground_truth_channel_is_refused_at_construction() -> None:
    """At construction, not at the first render: the rig boots a GPU before it renders."""
    with pytest.raises(ValueError, match="unknown ground-truth channel 'normals'"):
        FakeIsaacBinding(ground_truth=("depth", "normals"))
    with pytest.raises(ValueError, match="unknown ground-truth channel 'normals'"):
        IsaacSimBinding(ground_truth=("normals",))


def test_a_bare_string_is_refused_instead_of_iterated_into_letters() -> None:
    """``ground_truth="depth"`` is the obvious typo and Python would iterate it into five
    single-character channels; the error has to name the mistake, not the letter ``d``."""
    with pytest.raises(ValueError, match=r"got the string 'depth'"):
        FakeIsaacBinding(ground_truth="depth")


def test_channels_are_deduplicated_in_the_order_given() -> None:
    binding = FakeIsaacBinding(ground_truth=("segmentation", "depth", "segmentation"))
    assert binding.ground_truth_channels == ("segmentation", "depth")


def test_depth_is_float32_hw_in_metres() -> None:
    binding = FakeIsaacBinding(render_hw=(32, 48), ground_truth=("depth",))
    depth = binding.render_depth("persp")
    assert depth is not None
    assert depth.dtype == np.float32
    assert depth.shape == (32, 48)
    finite = depth[np.isfinite(depth)]
    assert finite.size and float(finite.min()) > 0.0


def test_depth_reports_the_background_as_inf_rather_than_a_finite_sentinel() -> None:
    """``distance_to_camera`` gives ``inf`` where a ray hit nothing, and the binding passes it
    through untouched. A rig that forgets to mask has to blow up here, on a laptop, rather
    than report a p95 depth error of ``inf`` after a GPU-hour of rendering — or, worse, a
    finite sentinel that reads like a real distance."""
    depth = FakeIsaacBinding(ground_truth=("depth",)).render_depth("persp")
    assert depth is not None
    assert np.isinf(depth).any()
    assert np.isfinite(depth).any()


def test_segmentation_gives_integer_ids_and_the_labels_that_explain_them() -> None:
    """Ids without the map are unusable: Replicator numbers them per annotator, so "which id
    is the apple" is answerable only through ``id_to_labels``. Keys are ``int`` — the vendor
    hands them back as strings, and a string-keyed lookup finds nothing and says nothing."""
    binding = FakeIsaacBinding(render_hw=(32, 48), ground_truth=("segmentation",))
    seg = binding.render_segmentation("persp")
    assert seg is not None
    assert seg.ids.dtype == np.uint32
    assert seg.ids.shape == (32, 48)
    assert all(isinstance(key, int) for key in seg.id_to_labels)
    labelled = {value["class"] for value in seg.id_to_labels.values()}
    assert "apple" in labelled
    assert set(np.unique(seg.ids)) <= set(seg.id_to_labels)


def test_the_object_centroid_moves_with_the_pose() -> None:
    """PR-08 §4 measures the displacement between the true and the estimated object centroid.
    A fake whose mask never moved would make every assertion about that displacement — and
    every rig bug that freezes it — invisible."""
    binding = FakeIsaacBinding(ground_truth=("segmentation",))
    apple = next(
        i
        for i, v in binding.render_segmentation("persp").id_to_labels.items()
        if v["class"] == "apple"
    )

    def centroid(seg: Any) -> float:
        columns = np.nonzero(seg.ids == apple)[1]
        assert columns.size, "the object left the frame; the fake must always paint it"
        return float(columns.mean())

    before = centroid(binding.render_segmentation("persp"))
    binding.set_dof_gains(_flat(binding, 400.0), _flat(binding, 20.0))
    binding.set_dof_position_targets(_flat(binding, 0.5))
    binding.step(20)
    assert centroid(binding.render_segmentation("persp")) != before


def test_ground_truth_renders_never_advance_the_tick() -> None:
    """Same guarantee as rgb, and it has to hold per channel: the adapter owns the clock."""
    binding = FakeIsaacBinding(ground_truth=("depth", "segmentation"))
    binding.step(5)
    before = binding.get_physics_step_count()
    binding.render_frame("persp")
    binding.render_depth("persp")
    binding.render_segmentation("persp")
    assert binding.get_physics_step_count() == before


def test_each_channel_warms_up_on_its_own_schedule() -> None:
    """The three annotators are independent objects on the box. A rig that asked for rgb
    first must not be handed an unwarmed depth frame because someone shared a counter."""
    binding = FakeIsaacBinding(ground_truth=("depth",), warmup_frames=2)
    assert [binding.render_frame("persp") for _ in range(2)] == [None, None]
    assert binding.render_frame("persp") is not None
    assert binding.render_depth("persp") is None  # its own first call, still warming up
    assert binding.render_depth("persp") is None
    assert binding.render_depth("persp") is not None


def test_an_unknown_camera_is_refused_for_every_channel() -> None:
    binding = FakeIsaacBinding(ground_truth=("depth", "segmentation"))
    for call in (binding.render_frame, binding.render_depth, binding.render_segmentation):
        with pytest.raises(ValueError, match="unknown camera"):
            call("wrist_left")


def test_ground_truth_calls_raise_after_close() -> None:
    binding = FakeIsaacBinding(ground_truth=("depth", "segmentation"))
    binding.close()
    for call in (binding.render_depth, binding.render_segmentation):
        with pytest.raises(RuntimeError, match="closed"):
            call("persp")


# -- gains ---------------------------------------------------------------------------------


def test_gains_round_trip_verbatim(fake: FakeIsaacBinding) -> None:
    """The caller owns the gains — nothing may silently re-tune them. (This is why the
    backend is raw Isaac Sim and not Isaac Lab, whose DCMotorCfg actuator model computes
    torque in Python and neutralises the sim PD gains.)"""
    kp, kd = _flat(fake, 123.0), _flat(fake, 4.5)
    fake.set_dof_gains(kp, kd)
    got_kp, got_kd = fake.get_dof_gains()
    np.testing.assert_array_equal(got_kp, kp)
    np.testing.assert_array_equal(got_kd, kd)


def test_zero_kp_is_accepted_and_not_clamped_to_a_floor(fake: FakeIsaacBinding) -> None:
    """``kp = 0`` IS the e-stop damping mode. A binding that quietly floored it would turn
    every emergency stop into a position hold at whatever target was last commanded."""
    fake.set_dof_gains(np.zeros(fake.num_dofs, np.float32), _flat(fake, 20.0))
    got_kp, got_kd = fake.get_dof_gains()
    assert not got_kp.any()
    np.testing.assert_array_equal(got_kd, _flat(fake, 20.0))


def test_zero_kp_with_damping_brings_a_moving_joint_to_rest(fake: FakeIsaacBinding) -> None:
    """The property the e-stop actually needs: damping decays velocity rather than ringing."""
    fake.set_dof_gains(_flat(fake, 400.0), _flat(fake, 2.0))
    fake.set_dof_position_targets(_flat(fake, 1.0))
    fake.step(10)
    moving = float(np.abs(fake.get_dof_velocities()).max())
    assert moving > 0.0
    fake.set_dof_gains(np.zeros(fake.num_dofs, np.float32), _flat(fake, 50.0))
    fake.step(200)
    assert float(np.abs(fake.get_dof_velocities()).max()) < 0.05 * moving


def test_negative_gains_are_refused(fake: FakeIsaacBinding) -> None:
    with pytest.raises(ValueError, match="kp/kd entries must be >= 0"):
        fake.set_dof_gains(_flat(fake, -1.0), _flat(fake, 1.0))


# -- integration behaviour -----------------------------------------------------------------


def test_nothing_moves_without_gains_and_the_commanded_joint_moves_with_them(
    fake: FakeIsaacBinding,
) -> None:
    """Positions must respond to targets THROUGH the gains, and only through them — a fake
    that snapped to the target would make every gain test above it vacuous."""
    fake.set_dof_position_targets(_flat(fake, 0.4))
    fake.step(50)
    assert not fake.get_dof_positions().any()

    fake.set_dof_gains(_flat(fake, 400.0), _flat(fake, 40.0))
    fake.step(50)
    reached = fake.get_dof_positions()
    assert np.all(reached > 0.0)
    assert np.all(reached <= 0.4 + 1e-6)


def test_efforts_follow_the_pd_law(fake: FakeIsaacBinding) -> None:
    """Diagnostic channel, but it has to be a function of the command or it is decoration."""
    fake.set_dof_gains(_flat(fake, 10.0), _flat(fake, 0.0))
    fake.set_dof_position_targets(_flat(fake, 2.0))
    np.testing.assert_allclose(fake.get_dof_efforts(), 20.0, atol=1e-5)


def test_reset_restores_the_pose_but_not_the_tick(fake: FakeIsaacBinding) -> None:
    """``get_num_physics_steps`` is a raw counter on the real binding, so a caller that
    resets mid-episode must call ``G1Adapter.forget_tick()`` — that is only true if the fake
    behaves the same way."""
    fake.set_dof_gains(_flat(fake, 400.0), _flat(fake, 20.0))
    fake.set_dof_position_targets(_flat(fake, 0.3))
    fake.step(20)
    assert fake.get_dof_positions().any()
    tick = fake.get_physics_step_count()
    fake.reset()
    assert not fake.get_dof_positions().any()
    assert not fake.get_dof_velocities().any()
    assert fake.get_physics_step_count() == tick


def test_readbacks_are_copies_not_live_views(fake: FakeIsaacBinding) -> None:
    q = fake.get_dof_positions()
    q[0] = 99.0
    assert fake.get_dof_positions()[0] == 0.0


# -- pre-physics callbacks: the e-stop drain point -----------------------------------------


def test_callbacks_fire_once_per_physics_step_in_registration_order(
    fake: FakeIsaacBinding,
) -> None:
    calls: list[str] = []
    fake.register_pre_physics_callback(lambda: calls.append("a"))
    fake.register_pre_physics_callback(lambda: calls.append("b"))
    fake.step(3)
    assert calls == ["a", "b"] * 3


def test_a_callback_registered_after_stepping_only_sees_later_steps(
    fake: FakeIsaacBinding,
) -> None:
    fake.step(2)
    calls: list[int] = []
    fake.register_pre_physics_callback(lambda: calls.append(1))
    fake.step(2)
    assert len(calls) == 2


def test_a_wedged_main_thread_never_drains_the_callback_and_never_ticks() -> None:
    """The e-stop's REAL limitation, reproduced rather than documented away.

    The Omniverse API is main-thread-only, so an e-stop from a watchdog thread can only latch
    a flag for a PHYSICS_PRE_STEP callback to drain. If the main loop is wedged, the drain
    never happens and NOTHING reaches the simulator — where ``DdsG1Transport.emergency_damp()``
    would already have put damping on the DDS wire. That is not parity and this test is the
    proof, not a footnote.
    """
    binding = FakeIsaacBinding()
    drained: list[int] = []
    binding.register_pre_physics_callback(lambda: drained.append(1))
    binding.step(2)
    assert len(drained) == 2

    binding.wedge_main_thread()
    tick = binding.get_physics_step_count()
    binding.step(50)
    assert drained == [1, 1]
    assert binding.get_physics_step_count() == tick

    binding.release_main_thread()
    binding.step(1)
    assert len(drained) == 3


# -- the main-thread rule ------------------------------------------------------------------


def test_isaac_calls_are_refused_off_the_main_thread(fake: FakeIsaacBinding) -> None:
    """NVIDIA documents the prohibition but not the consequence of breaking it, so assume the
    worst — silently wrong readings. The fake enforces it too, deliberately: that is what
    proves the transport's ``emergency_damp()`` latch never touches the binding from the
    watchdog thread it is called on.
    """
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            fake.step(1)
        except BaseException as exc:  # noqa: BLE001 - the point is to capture it
            errors.append(exc)

    thread = threading.Thread(target=worker, name="watchdog")
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "main-thread-only" in str(errors[0])


def test_the_main_thread_guard_can_be_switched_off_for_worker_driven_tests() -> None:
    binding = FakeIsaacBinding(enforce_main_thread=False)
    ok: list[int] = []
    thread = threading.Thread(target=lambda: (binding.step(1), ok.append(1)))
    thread.start()
    thread.join()
    assert ok == [1]
    assert binding.get_physics_step_count() == 1


# -- lifecycle -----------------------------------------------------------------------------


def test_close_is_idempotent_and_everything_raises_afterwards(fake: FakeIsaacBinding) -> None:
    fake.close()
    fake.close()
    assert fake.is_closed
    with pytest.raises(RuntimeError, match="closed"):
        fake.step(1)
    with pytest.raises(RuntimeError, match="closed"):
        fake.get_dof_positions()
    with pytest.raises(RuntimeError, match="closed"):
        fake.render_frame("persp")


# -- name resolution: the failure this seam fears most -------------------------------------


def test_the_expected_dof_count_is_29_body_plus_two_dex3_hands() -> None:
    assert EXPECTED_NUM_DOFS == 43
    assert EXPECTED_NUM_DOFS == len(G1_MOTOR_JOINT_NAMES) + 2 * len(DEX3_FINGER_JOINTS)


def test_the_fakes_dof_order_is_deliberately_not_canonical(fake: FakeIsaacBinding) -> None:
    """PhysX walks the articulation breadth-first from the base link. Any code that indexes
    positionally has to break against this fake, on a laptop, instead of on a robot."""
    assert len(fake.dof_names) == EXPECTED_NUM_DOFS
    assert fake.dof_indices.body != tuple(range(len(G1_MOTOR_JOINT_NAMES)))


def test_every_canonical_joint_resolves_to_the_dof_that_carries_its_name(
    fake: FakeIsaacBinding,
) -> None:
    """The one assertion that stands between a permuted map and a physical arm going
    somewhere nobody asked for: index i of the body map must be the DOF actually NAMED after
    canonical joint i."""
    names, idx = fake.dof_names, fake.dof_indices
    for slot, canonical in enumerate(G1_MOTOR_JOINT_NAMES):
        assert names[idx.body[slot]] == idx.body_pattern.format(name=canonical)
    for side, indices, pattern in (
        ("left", idx.left, idx.left_pattern),
        ("right", idx.right, idx.right_pattern),
    ):
        for slot, finger in enumerate(DEX3_FINGER_JOINTS):
            assert names[indices[slot]] == pattern.format(side=side, finger=finger)
    assert len(set(idx.body) | set(idx.left) | set(idx.right)) == EXPECTED_NUM_DOFS


def test_two_canonical_joints_may_not_share_one_dof(monkeypatch: pytest.MonkeyPatch) -> None:
    """A convention that maps two canonical joints onto one DOF is refused, not resolved.

    This is the failure with no symptom: resolution succeeds, ``num_dofs`` is 43, every readback
    is a plausible float, and two canonical joints quietly drive one physical actuator. Nothing
    downstream can detect it — ``G1Adapter`` gathers by hard-coded index and gets exactly what it
    asked for.

    The shipped candidate patterns cannot produce a collision; a future one can, which is the
    whole point of a runtime guard rather than a test over today's tuples. The pattern injected
    here is the realistic mistake — one that forgets to vary ``{finger}``, so all seven fingers
    of a hand format to the same name.
    """
    names = fake_g1_dof_names("{name}", "{side}_hand_{finger}")
    monkeypatch.setattr(isaac_binding, "FINGER_NAME_CANDIDATES", ("{side}_shoulder_pitch",))

    with pytest.raises(ValueError, match="same DOF index"):
        resolve_g1_dof_indices(names)


def test_an_alternative_naming_convention_still_resolves() -> None:
    """The USD is a conversion of the vendor URDF and the converter may not have preserved
    the ``_joint`` suffix. Which convention the shipped asset uses is a DISCOVERY (preflight
    check G), so more than one has to work."""
    names = fake_g1_dof_names(BODY_NAME_CANDIDATES[1], FINGER_NAME_CANDIDATES[1])
    resolved = resolve_g1_dof_indices(names)
    assert resolved.body_pattern == BODY_NAME_CANDIDATES[1]
    assert resolved.left_pattern == FINGER_NAME_CANDIDATES[1]
    assert names[resolved.body[0]] == G1_MOTOR_JOINT_NAMES[0]


def test_a_missing_joint_is_named_and_the_actual_dof_names_are_dumped() -> None:
    """A resolution failure is a config/data finding, so the message has to carry the data:
    which joint is missing, and what the articulation actually has."""
    names = list(fake_g1_dof_names())
    names[names.index("left_elbow_joint")] = "left_elbow_link_joint"
    with pytest.raises(ValueError) as excinfo:
        resolve_g1_dof_indices(names)
    message = str(excinfo.value)
    assert "left_elbow_joint" in message
    assert "left_elbow_link_joint" in message
    assert "waist_yaw_joint" in message


def test_a_missing_finger_is_reported_against_the_right_hand() -> None:
    names = list(fake_g1_dof_names())
    names[names.index("right_hand_thumb_2_joint")] = "right_hand_thumb_two_joint"
    with pytest.raises(ValueError, match="right hand"):
        resolve_g1_dof_indices(names)


def test_duplicate_dof_names_are_refused_instead_of_guessed() -> None:
    names = list(fake_g1_dof_names())
    names[-1] = names[0]
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_g1_dof_indices(names)


def test_a_binding_with_broken_names_still_constructs_but_refuses_to_map() -> None:
    """Construction must not be where this fails, or the fake could never be used to test
    the failure."""
    binding = FakeIsaacBinding(dof_names=["a", "b", "c"])
    assert binding.num_dofs == 3
    with pytest.raises(ValueError, match="cannot resolve"):
        _ = binding.dof_indices


def test_the_naming_candidates_agree_with_the_preflight_copies() -> None:
    """The candidate patterns exist twice on purpose — ``scripts/preflight_isaac.py`` must be
    able to test the vendor API without believing anything this module believes. Two copies
    of a convention is two chances to diverge, so the divergence is asserted away here rather
    than discovered on the box."""
    preflight = _load_preflight()
    assert preflight.BODY_NAME_CANDIDATES == BODY_NAME_CANDIDATES
    assert preflight.FINGER_NAME_CANDIDATES == FINGER_NAME_CANDIDATES
    assert preflight.EXPECTED_NUM_DOFS == EXPECTED_NUM_DOFS


def test_the_preflight_probes_the_same_asset_and_effort_getters_this_module_uses() -> None:
    """The other two constants duplicated across that seam, pinned for the same reason.

    ``DEFAULT_ASSET_SUBPATH`` is the sharp one: the preflight's whole value is that it
    validated the DoF count, the joint names and the gain round-trip on THE USD THE BINDING
    LOADS. Let the two subpaths drift and the preflight goes green on one asset while
    ``IsaacSimBinding`` loads another — the report says 43 named DOFs and the robot moves the
    wrong arm, which is precisely the failure the by-name resolution exists to prevent.
    ``EFFORT_GETTER_CANDIDATES`` is milder (effort readback is diagnostic only), but the
    preflight RECORDS which name this build exposes so the binding can stop guessing; a
    recorded answer to a different question is worse than no answer.
    """
    preflight = _load_preflight()
    assert preflight.DEFAULT_ASSET_SUBPATH == DEFAULT_ASSET_SUBPATH
    assert preflight.EFFORT_GETTER_CANDIDATES == EFFORT_GETTER_CANDIDATES


def test_the_preflight_probes_the_same_ground_truth_annotators_this_module_attaches() -> None:
    """Third constant across that seam, pinned for the sharp version of the same reason: the
    preflight is the ONLY thing that will ever execute ``distance_to_camera`` or
    ``semantic_segmentation`` before the calibration rig does. Let the names drift and check N
    goes green on an annotator the binding never asks for, while the binding asks for one
    nobody proved exists — and the first symptom is a rig that renders nothing on the box."""
    preflight = _load_preflight()
    assert preflight.GROUND_TRUTH_ANNOTATORS == dict(GROUND_TRUTH_ANNOTATORS)
    assert dict(GROUND_TRUTH_ANNOTATORS) == {
        "depth": "distance_to_camera",
        "segmentation": "semantic_segmentation",
    }
    # The init params too: a preflight that proved a COLORISED segmentation works would say
    # nothing about the annotator the binding actually asks for.
    assert preflight.ANNOTATOR_INIT_PARAMS == {
        channel: dict(params) for channel, params in isaac_binding._ANNOTATOR_INIT_PARAMS.items()
    }
    assert isaac_binding._ANNOTATOR_INIT_PARAMS["segmentation"]["colorize"] is False


# -- the boundary helpers ------------------------------------------------------------------


class _WarpLike:
    """Stands in for a ``warp.array``: convertible only via ``.numpy()``."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def numpy(self) -> np.ndarray:
        return self._array


def test_a_warp_like_batched_readback_is_squeezed_to_one_row() -> None:
    """``Articulation`` is a view over N prims and returns ``(N, D)`` warp arrays. Both
    conversions happen in one place so no caller has to remember which shape it holds."""
    row = _row(_WarpLike(np.arange(43, dtype=np.float32).reshape(1, 43)), 43, "positions")
    assert row.shape == (43,)
    assert row[7] == 7.0
    assert isinstance(row, np.ndarray)
    assert not isinstance(_to_numpy(_WarpLike(np.zeros(3))), _WarpLike)


def test_an_unbatched_readback_is_accepted_and_a_multi_robot_batch_is_not() -> None:
    assert _row(np.zeros((43,), dtype=np.float32), 43, "positions").shape == (43,)
    with pytest.raises(RuntimeError, match="more than one robot"):
        _row(np.zeros((2, 43), dtype=np.float32), 43, "positions")


def test_a_command_is_reshaped_to_the_batch_isaac_wants_and_validated() -> None:
    assert _batch(np.zeros(43, dtype=np.float32), 43, "targets").shape == (1, 43)
    with pytest.raises(ValueError, match=r"expected shape \(43,\)"):
        _batch(np.zeros(42, dtype=np.float32), 43, "targets")
    with pytest.raises(ValueError, match="non-finite"):
        _batch(np.full(43, np.nan, dtype=np.float32), 43, "targets")


def test_non_finite_targets_never_reach_the_simulation(fake: FakeIsaacBinding) -> None:
    """A NaN reaching PhysX corrupts the articulation for the rest of the episode while the
    readbacks may stay finite — undetectable downstream, so it stops at the seam."""
    with pytest.raises(ValueError, match="non-finite"):
        fake.set_dof_position_targets(np.full(fake.num_dofs, np.inf, dtype=np.float32))
    fake.set_dof_gains(_flat(fake, 100.0), _flat(fake, 10.0))
    fake.step(5)
    assert np.isfinite(fake.get_dof_positions()).all()


# -- structural promises -------------------------------------------------------------------


def test_the_fake_satisfies_the_binding_protocol(fake: FakeIsaacBinding) -> None:
    assert isinstance(fake, IsaacBinding)


def test_a_physics_rate_that_does_not_survive_physx_truncation_is_refused() -> None:
    """``PhysxScene.set_dt`` does ``steps_per_second = int(1.0 / dt)`` and nothing warns when
    that truncation loses a hertz — Isaac Lab ships a live victim (dt=0.0167 -> 59, not 60).
    1364 of the first 20000 integer rates round-trip wrong, 99 Hz among them.

    This is one of only two constructor guards that can be exercised on a machine with no
    Isaac Sim: it runs BEFORE the vendor import, which is deliberate — a rate that is going
    to be silently wrong should be rejected before a GPU is booted for it.
    """
    with pytest.raises(ValueError, match="truncation"):
        IsaacSimBinding(physics_hz=99)
    with pytest.raises(ValueError, match="physics_hz must be > 0"):
        IsaacSimBinding(physics_hz=0)


def test_the_real_bindings_tick_rule_is_the_one_the_preflight_and_the_transport_apply() -> None:
    """All three had to agree on what "the tick is an integer" MEANS, and they did not.

    ``scripts/preflight_isaac.py`` and ``IsaacG1Transport.step_count`` test
    ``numbers.Integral``; this method used to test ``(int, np.integer)``. Anything Integral but
    neither concrete type — a pybind counter that registers itself, say — passed the preflight
    green on the box and then made every ``read_state()`` raise ``TypeError`` on the first
    control step. Nobody can run ``IsaacSimBinding`` here, so its ``_sim`` is stubbed; the rule
    under test is a plain isinstance check and does not need Isaac to be meaningful.

    A float is still refused, which is the thing that actually matters: staleness upstream is
    an equality test and a clock that only almost repeats fires the watchdog at random.
    """

    class PybindCounter(numbers.Integral):
        """Integral by registration, neither ``int`` nor ``np.integer`` — exact under ``==``."""

        def __init__(self, value: int) -> None:
            self._value = value

        def __int__(self) -> int:
            return self._value

        def __getattr__(self, name: str) -> Any:  # pragma: no cover - abstract-method filler
            raise AttributeError(name)

    for method in PybindCounter.__abstractmethods__ - {"__int__"}:
        setattr(PybindCounter, method, lambda *a, **k: NotImplemented)
    PybindCounter.__abstractmethods__ = frozenset()

    binding = object.__new__(IsaacSimBinding)
    binding._closed = False

    class _Sim:
        def __init__(self, value: Any) -> None:
            self.value = value

        def get_num_physics_steps(self) -> Any:
            return self.value

    binding._sim = _Sim(PybindCounter(7))
    tick = binding.get_physics_step_count()
    assert type(tick) is int
    assert tick == 7

    binding._sim = _Sim(np.int64(7))
    assert binding.get_physics_step_count() == 7

    for refused in (7.0, True, "7"):
        binding._sim = _Sim(refused)
        with pytest.raises(TypeError, match="integer counter"):
            binding.get_physics_step_count()


class _StubAnnotator:
    """One Replicator annotator reduced to the one method the binding calls."""

    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def get_data(self) -> Any:
        return self.payload


class _StubRenderer:
    """``RenderingManager`` reduced to ``render()``, counting the calls."""

    def __init__(self) -> None:
        self.renders = 0

    def render(self) -> None:
        self.renders += 1


def _stubbed_real_binding(**payloads: Any) -> IsaacSimBinding:
    """An :class:`IsaacSimBinding` with its two render dependencies replaced, and no Isaac.

    Same bare-object trick as the tick test above: ``__init__`` never runs, so no vendor
    import happens. It exists because the annotator-to-numpy path in ``IsaacSimBinding``
    is otherwise UNEXECUTED CODE on every machine that is not the box — the payload shapes
    are guesses, and a guess that is never even run against a stub is two guesses.
    """
    binding = object.__new__(IsaacSimBinding)
    binding._closed = False
    binding._rendering = _StubRenderer()
    binding._annotators = {
        "persp": {channel: _StubAnnotator(payload) for channel, payload in payloads.items()}
    }
    return binding


def test_the_real_binding_parses_the_documented_segmentation_payload() -> None:
    """``semantic_segmentation`` is the odd one out: it returns ``{"data": ..., "info": ...}``
    rather than a bare array, and the string ids in ``idToLabels`` have to become ints or the
    object can never be found in the mask."""
    binding = _stubbed_real_binding(
        segmentation={
            "data": np.full((4, 5, 1), 2, dtype=np.uint32),
            "info": {"idToLabels": {"0": {"class": "BACKGROUND"}, "2": {"class": "apple"}}},
        }
    )
    seg = binding.render_segmentation("persp")
    assert isinstance(seg, SegmentationFrame)
    assert seg.ids.shape == (4, 5)
    assert seg.ids.dtype == np.uint32
    assert seg.id_to_labels == {0: {"class": "BACKGROUND"}, 2: {"class": "apple"}}


def test_the_real_binding_also_accepts_a_bare_segmentation_array() -> None:
    """Which of the two shapes a given Replicator build produces is UNVERIFIED (preflight
    check N records it), so the parser must survive either — with an empty label map, which
    is honest, rather than a fabricated one."""
    binding = _stubbed_real_binding(segmentation=np.zeros((4, 5), dtype=np.uint32))
    seg = binding.render_segmentation("persp")
    assert seg is not None
    assert seg.ids.shape == (4, 5)
    assert seg.id_to_labels == {}


def test_the_real_binding_refuses_a_colorized_segmentation() -> None:
    """``colorize=False`` is requested at construction; an (H, W, 4) uint8 array means it did
    not take effect, and a centroid measured off a palette is not a measurement of geometry."""
    binding = _stubbed_real_binding(segmentation=np.zeros((4, 5, 4), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="colorize"):
        binding.render_segmentation("persp")


def test_the_real_bindings_depth_is_squeezed_to_hw_float32_and_keeps_inf() -> None:
    payload = np.full((4, 5, 1), 2.5, dtype=np.float32)
    payload[0, 0, 0] = np.inf
    binding = _stubbed_real_binding(depth=payload)
    depth = binding.render_depth("persp")
    assert depth is not None
    assert depth.shape == (4, 5)
    assert depth.dtype == np.float32
    assert np.isinf(depth[0, 0])
    assert depth[1, 1] == np.float32(2.5)


def test_the_real_binding_reports_a_depth_shape_it_does_not_understand() -> None:
    binding = _stubbed_real_binding(depth=np.zeros((4, 5, 3), dtype=np.float32))
    with pytest.raises(RuntimeError, match="distance_to_camera"):
        binding.render_depth("persp")


def test_the_real_bindings_ground_truth_warmup_returns_none_and_still_rendered() -> None:
    """An empty annotator buffer is the warmup, on all three channels alike: return ``None``
    and let the caller retry. Zeros would be a depth map of a plane 0 m from the lens."""
    binding = _stubbed_real_binding(
        depth=np.zeros((0,), dtype=np.float32), segmentation={"data": None, "info": {}}
    )
    assert binding.render_depth("persp") is None
    assert binding.render_segmentation("persp") is None
    assert binding._rendering.renders == 2


def test_the_real_binding_refuses_a_channel_it_never_attached() -> None:
    """The rgb-only binding is the default one, so this is the message an operator who forgot
    ``ground_truth=`` will actually see."""
    binding = _stubbed_real_binding(rgb=np.zeros((4, 5, 4), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="not attached"):
        binding.render_depth("persp")
    with pytest.raises(ValueError, match="unknown camera"):
        binding.render_frame("wrist_left")


def test_the_real_binding_without_isaac_names_the_two_venv_split_instead_of_raising_import() -> (
    None
):
    """The operator most likely to hit this ran ``rollout.py --robot isaac_g1`` in the WAM
    venv, and a bare ``ModuleNotFoundError: isaacsim`` would send them off to pip-install it
    there — which cannot work: isaacsim-core 6.0.1 pins torch 2.11.0 and this repo resolves
    2.13.0. The message has to carry the actual topology."""
    with pytest.raises(RuntimeError) as excinfo:
        IsaacSimBinding()
    message = str(excinfo.value)
    assert "isaac-python" in message
    assert "--policy remote" in message
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_the_module_imports_and_runs_without_isaac_sim_and_without_torch() -> None:
    """The Isaac venv cannot contain this repo's torch (isaacsim-core 6.0.1 pins 2.11.0,
    uv.lock resolves 2.13.0), so the whole Isaac side of the split must be torch-free.

    Checked in a SUBPROCESS: the pytest process has torch imported by other test modules, so
    an in-process ``'torch' not in sys.modules`` would be testing the test runner. And this
    machine has no Isaac Sim at all, so the import succeeding here is itself the proof that
    nothing at module scope reaches for the vendor stack.
    """
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed, so this check could not fail")
    code = (
        "import sys\n"
        "import wam.robot.isaac_binding as b\n"
        "binding = b.FakeIsaacBinding()\n"
        "binding.register_pre_physics_callback(lambda: None)\n"
        "binding.set_dof_gains([1.0] * 43, [0.1] * 43)\n"
        "binding.step(3)\n"
        "binding.render_frame('persp')\n"
        "gt = b.FakeIsaacBinding(ground_truth=('depth', 'segmentation'))\n"
        "assert gt.render_depth('persp').shape == gt.render_segmentation('persp').ids.shape\n"
        "assert binding.dof_indices.body\n"
        "leaked = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))\n"
        "assert not leaked, leaked\n"
        "print('clean')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=_REPO_ROOT, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout
