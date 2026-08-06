#!/usr/bin/env python3
"""Preflight for the Isaac Sim backend — run this FIRST on the 5090 box, before anything else.

    isaac-python scripts/preflight_isaac.py --out runs/preflight/isaac.json

WHY THIS EXISTS
    ``wam.robot.isaac_transport`` was written against NVIDIA's *documentation* for Isaac Sim
    6.0.1, on a machine with no Isaac Sim and no GPU. Every other backend in this repo was
    written against something that could be run: ``FakeG1Transport`` against itself,
    ``MujocoG1Transport`` against MuJoCo on a laptop, ``DdsG1Transport`` against a real
    CycloneDDS bus in an arm64 container (T-25a). This one could not be, so the binding layer
    carries assumptions instead of measurements.

    This script converts each of those assumptions into a check that fails loudly. It is the
    Isaac equivalent of ``docker/dds/conformance.py``: it does not test WAM, it tests that the
    vendor API is shaped the way the binding believes it is. Run it before trusting a single
    number out of an Isaac rollout.

WHAT IT DELIBERATELY DOES NOT DO
    It does not import ``wam.robot.isaac_transport`` and exercise it. That would conflate two
    questions — "is the vendor API what we think" and "is our code correct against it" — and
    the first has to be answered first. ``tests/test_isaac_g1.py`` answers the second, on CPU,
    against a stub.

THE DISCOVERY PART
    Two things are genuinely unknown until this runs, and both are recorded rather than
    asserted: the articulation's DOF *names* (PhysX orders joints breadth-first from the base
    link, which is neither URDF order nor ours) and the joint *naming convention* in the
    shipped USD. Check G tries several candidate conventions, reports which one matched, and
    dumps the full name list into the report either way. A mismatch here is a config change,
    not a code change — which is why it must be data in a report and not a traceback.

Exit code 0 iff every check passes.
"""

from __future__ import annotations

import argparse
import json
import numbers
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any

# The 29 motor slots and 7 finger joints per hand, in canonical order. Imported rather than
# copied: two copies of a 29-entry ordering is two chances to permute it, and a permuted map
# is silent (see the note on G1_MOTOR_JOINT_NAMES in wam/robot/g1_transport.py). This import
# needs only numpy + pydantic + pyyaml, so it is safe inside Isaac's python.
try:
    from wam.robot.g1_transport import DEX3_FINGER_JOINTS, G1_MOTOR_JOINT_NAMES
except ImportError:  # pragma: no cover - only when wam is not installed in the isaac venv
    print(
        "FATAL: cannot import wam. Install it into the Isaac venv:\n"
        "  pip install -e '.[serve]'   # base deps are numpy/pydantic/pyyaml — no torch",
        file=sys.stderr,
    )
    raise

#: Candidate joint-naming conventions for the shipped G1 USD, tried in order. The Unitree URDF
#: (g1_29dof_with_hand_rev_1_0) uses ``<canonical>_joint`` for the body and
#: ``<side>_hand_<finger>_joint`` for the fingers, but the USD is a conversion of it and the
#: converter may or may not preserve the suffix. UNVERIFIED until this script runs.
BODY_NAME_CANDIDATES: tuple[str, ...] = ("{name}_joint", "{name}")
FINGER_NAME_CANDIDATES: tuple[str, ...] = (
    "{side}_hand_{finger}_joint",
    "{side}_hand_{finger}",
    "{side}_{finger}_joint",
)

#: Candidate names for the Articulation's measured-effort getter. Unlike positions,
#: velocities and gains, this one is NOT in the documentation the binding was written from,
#: so it is a guess with fallbacks. RECORDED, NOT CHECKED: effort readback is diagnostic
#: only — it is not part of the ``G1Transport`` low-state contract (q, dq, imu, gripper,
#: tick_ns) — and failing a whole preflight on a symbol no rollout needs would block the box
#: for nothing. ``isaac_binding.IsaacSimBinding`` resolves the same list at construction and
#: raises naming all of them only if someone actually asks for efforts.
EFFORT_GETTER_CANDIDATES: tuple[str, ...] = (
    "get_dof_efforts",
    "get_measured_dof_efforts",
    "get_dof_forces",
    "get_measured_joint_efforts",
)

#: Isaac ships the G1 under the asset root; this is the 29-DoF revision WITH the Dex3-1 hands,
#: i.e. the same kinematic model as the MuJoCo Menagerie file this repo already fetches.
#: The Isaac Lab copy (ISAACLAB_NUCLEUS_DIR) is a legacy 23-DoF G1 — do not point at it.
DEFAULT_ASSET_SUBPATH = "/Isaac/Robots/Unitree/G1/g1.usd"

#: 29 body + 2 x 7 fingers. Asserted, not assumed: the DoF count was never read out of the
#: USD itself during research, only cross-checked against the vendor URDF.
EXPECTED_NUM_DOFS = 29 + 2 * len(DEX3_FINGER_JOINTS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--asset", default=None, help="USD path (default: asset root + G1 g1.usd)")
    p.add_argument("--hz", type=int, default=500, help="physics rate; int(1/dt) must be exact")
    p.add_argument("--device", default="cuda:0", help="physics device")
    p.add_argument("--camera-hw", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    p.add_argument("--warmup-frames", type=int, default=20, help="max render ticks for a frame")
    p.add_argument("--out", default=None, help="write the JSON report here")
    p.add_argument("--gui", action="store_true", help="run with a window (default: headless)")
    return p.parse_args(argv)


class Report:
    """Collects check results; any failure makes the preflight exit non-zero."""

    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.info: dict[str, Any] = {}

    def check(self, name: str, ok: bool, detail: Any = "") -> bool:
        self.checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
        return bool(ok)

    def expect(self, name: str, actual: Any, expected: Any) -> bool:
        return self.check(name, actual == expected, f"got {actual!r}, expected {expected!r}")

    def record(self, key: str, value: Any) -> None:
        self.info[key] = value

    @property
    def failed(self) -> list[str]:
        return [c["name"] for c in self.checks if not c["ok"]]


def check_environment(report: Report) -> None:
    """A. Interpreter and OS. isaacsim 6.0.1 pins python ==3.12.* exactly, not as a floor."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    report.record("python", sys.version.split()[0])
    report.record("platform", platform.platform())
    report.check(
        "python_is_3_12",
        version == "3.12",
        f"{version} (isaacsim 6.0.1 requires ==3.12.*; 5.1 wanted 3.11)",
    )
    report.check("platform_is_linux", sys.platform == "linux", sys.platform)


def check_api_surface(report: Report) -> dict[str, Any]:
    """D. Every symbol the binding imports must exist and be callable.

    This is the check that says whether ``isaac_transport.py`` will even import on this
    machine. It is deliberately name-by-name rather than one blanket import, so a rename
    between Isaac releases points at the symbol that moved instead of at a bare ImportError.
    """
    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.rendering_manager import RenderingManager
    from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager

    wanted = {
        "Articulation.get_dof_positions": getattr(Articulation, "get_dof_positions", None),
        "Articulation.get_dof_velocities": getattr(Articulation, "get_dof_velocities", None),
        "Articulation.set_dof_position_targets": getattr(
            Articulation, "set_dof_position_targets", None
        ),
        "Articulation.set_dof_gains": getattr(Articulation, "set_dof_gains", None),
        "Articulation.get_dof_gains": getattr(Articulation, "get_dof_gains", None),
        "Articulation.get_dof_indices": getattr(Articulation, "get_dof_indices", None),
        # Used by check_determinism below and by IsaacBinding.reset (episode reset). Named
        # here too so a rename points at the symbol instead of at a traceback further down.
        "Articulation.reset": getattr(Articulation, "reset", None),
        "SimulationManager.setup_simulation": getattr(SimulationManager, "setup_simulation", None),
        "SimulationManager.step": getattr(SimulationManager, "step", None),
        "SimulationManager.get_physics_dt": getattr(SimulationManager, "get_physics_dt", None),
        "SimulationManager.get_num_physics_steps": getattr(
            SimulationManager, "get_num_physics_steps", None
        ),
        "SimulationManager.register_callback": getattr(
            SimulationManager, "register_callback", None
        ),
        "RenderingManager.set_dt": getattr(RenderingManager, "set_dt", None),
        "RenderingManager.render": getattr(RenderingManager, "render", None),
        "stage_utils.set_stage_units": getattr(stage_utils, "set_stage_units", None),
        "stage_utils.add_reference_to_stage": getattr(stage_utils, "add_reference_to_stage", None),
        "app_utils.play": getattr(app_utils, "play", None),
        "app_utils.update_app": getattr(app_utils, "update_app", None),
    }
    missing = sorted(name for name, obj in wanted.items() if not callable(obj))
    report.check(
        "api_surface_complete",
        not missing,
        "all present" if not missing else f"MISSING/uncallable: {missing}",
    )
    report.check(
        "physics_pre_step_event",
        hasattr(SimulationEvent, "PHYSICS_PRE_STEP"),
        "SimulationEvent.PHYSICS_PRE_STEP — the e-stop drain point",
    )

    # Discovery, not a gate — see EFFORT_GETTER_CANDIDATES.
    effort_getter = next(
        (n for n in EFFORT_GETTER_CANDIDATES if callable(getattr(Articulation, n, None))), None
    )
    report.record("effort_getter", effort_getter)
    print(
        f"[INFO] effort getter: {effort_getter or 'NONE of ' + str(list(EFFORT_GETTER_CANDIDATES))}"
        " (diagnostic only; no rollout needs it)",
        flush=True,
    )
    return {
        "Articulation": Articulation,
        "SimulationManager": SimulationManager,
        "SimulationEvent": SimulationEvent,
        "RenderingManager": RenderingManager,
        "stage_utils": stage_utils,
        "app_utils": app_utils,
    }


def check_rates(report: Report, api: dict[str, Any], hz: int, device: str) -> None:
    """E. Set every clock BEFORE play, and prove the int(1/dt) truncation did not bite.

    PhysxScene.set_dt does ``steps_per_second = int(1.0 / dt)``. A float round-trip can land
    on 499 when you asked for 500, and nothing warns. Isaac Lab ships a live victim of this
    (dt=0.0167 -> 59, not 60). Changing rates AFTER play() is a known fatal crash in 6.0, so
    this is also the only moment it is safe to do.
    """
    sm, rm = api["SimulationManager"], api["RenderingManager"]
    dt = 1.0 / hz
    report.check("hz_is_exact", int(1.0 / dt) == hz, f"int(1/{dt}) == {int(1.0 / dt)}, want {hz}")
    sm.setup_simulation(dt=dt, device=device)
    rm.set_dt(dt)
    actual = sm.get_physics_dt()
    report.record("physics_dt", actual)
    report.record("physics_hz", hz)
    report.check(
        "physics_dt_round_trips",
        abs(actual - dt) < 1e-12,
        f"set {dt!r}, got {actual!r} — a mismatch here silently changes every velocity limit",
    )


def _resolve_names(
    dof_names: list[str], canonical: tuple[str, ...], candidates: tuple[str, ...], **fmt: str
) -> tuple[str | None, list[str]]:
    """Return the first naming convention that covers every canonical name, and the misses."""
    available = set(dof_names)
    best_missing: list[str] = list(canonical)
    for pattern in candidates:
        wanted = [pattern.format(name=n, finger=n, **fmt) for n in canonical]
        missing = [w for w in wanted if w not in available]
        if not missing:
            return pattern, []
        if len(missing) < len(best_missing):
            best_missing = missing
    return None, best_missing


def check_articulation(report: Report, api: dict[str, Any], asset: str) -> Any:
    """F+G. Load the G1, count its DOFs, and DISCOVER the joint naming convention.

    The DoF ordering is PhysX's breadth-first walk from the base link — not URDF order and
    not ours. Indexing positionally here would give a plausible-looking robot doing the wrong
    thing, undetectable without hardware. So: resolve every joint by name, and record the
    whole name list in the report so a future asset swap shows up as a diff.
    """
    stage_utils, Articulation = api["stage_utils"], api["Articulation"]
    stage_utils.set_stage_units(meters_per_unit=1.0, kilograms_per_unit=1.0)
    stage_utils.add_reference_to_stage(usd_path=asset, path="/World/G1")
    report.record("asset", asset)

    robot = Articulation("/World/G1")
    api["app_utils"].play()
    api["app_utils"].update_app(steps=2)  # tensor backend only becomes valid here

    dof_names = list(robot.dof_names)
    report.record("num_dofs", int(robot.num_dofs))
    report.record("dof_names", dof_names)
    report.expect("num_dofs_is_43", int(robot.num_dofs), EXPECTED_NUM_DOFS)

    body_pattern, body_missing = _resolve_names(
        dof_names, G1_MOTOR_JOINT_NAMES, BODY_NAME_CANDIDATES
    )
    report.record("body_joint_pattern", body_pattern)
    report.check(
        "body_joints_resolve",
        body_pattern is not None,
        f"pattern {body_pattern!r}"
        if body_pattern
        else f"no candidate covers all 29: {body_missing}",
    )
    for side in ("left", "right"):
        pattern, missing = _resolve_names(
            dof_names, DEX3_FINGER_JOINTS, FINGER_NAME_CANDIDATES, side=side
        )
        report.record(f"{side}_finger_pattern", pattern)
        report.check(
            f"{side}_fingers_resolve",
            pattern is not None,
            f"pattern {pattern!r}" if pattern else f"no candidate covers all 7: {missing}",
        )
    return robot


def check_tick(report: Report, api: dict[str, Any]) -> None:
    """H. The tick must be an integer that advances by exactly N over step(steps=N).

    ``G1Adapter.read_state`` decides staleness by EQUALITY against the previous tick. A float
    clock would make that comparison meaningless, and a tick that advances by a variable
    amount would make the watchdog fire at random. get_num_physics_steps is a raw counter.
    """
    sm = api["SimulationManager"]
    before = sm.get_num_physics_steps()
    sm.step(steps=7)
    after = sm.get_num_physics_steps()
    # numbers.Integral, not int: a pybind counter may surface as numpy.int64, which is NOT an
    # int but is still exact under ==. A float is what must fail here, and does. bool is
    # Integral and is excluded explicitly. isaac_binding.IsaacSimBinding applies the same rule
    # and coerces with int(), so the two agree on what "the tick is an integer" means.
    report.check(
        "tick_is_integer",
        all(
            isinstance(v, numbers.Integral) and not isinstance(v, bool) for v in (before, after)
        ),
        f"{type(before).__name__} -> equality comparison is only safe on an exact type "
        "(a numpy integer passes; a float must not)",
    )
    report.expect("tick_advances_exactly_7", after - before, 7)

    idle_before = sm.get_num_physics_steps()
    sm.step(steps=0)
    report.expect("tick_frozen_on_zero_steps", sm.get_num_physics_steps() - idle_before, 0)


def check_pre_step_callback(report: Report, api: dict[str, Any]) -> None:
    """L. The e-stop drain point must actually fire, on the main thread, once per step.

    The Omniverse API is main-thread-only, so an e-stop arriving on a watchdog thread cannot
    touch Isaac at all: ``isaac_transport.emergency_damp()`` latches a pending-damp flag in
    pure Python and a ``PHYSICS_PRE_STEP`` subscriber applies the damping on the main thread.
    If that subscription silently never fires, the e-stop is a no-op and NOTHING else in the
    system would notice — the latch would still stop further motor commands, so the failure
    is invisible from above. That is why this is a preflight check and not a unit test.

    Neither ``register_callback``'s signature nor the payload it delivers to a
    PHYSICS_PRE_STEP subscriber is documented. Both are RECORDED here (the binding's
    dispatcher takes ``*args, **kwargs`` precisely because they are unknown), and a
    registration that raises is reported as a failed check rather than as a traceback that
    would take the whole ``--out`` report with it.
    """
    sm, events = api["SimulationManager"], api["SimulationEvent"]
    seen: dict[str, Any] = {"calls": 0, "args": None, "kwargs": None, "threads": set()}

    def on_pre_step(*args: Any, **kwargs: Any) -> None:
        seen["calls"] += 1
        seen["args"] = [type(a).__name__ for a in args]
        seen["kwargs"] = sorted(kwargs)
        seen["threads"].add(threading.current_thread().name)

    try:
        handle = sm.register_callback(on_pre_step, event=events.PHYSICS_PRE_STEP)
    except Exception as err:  # noqa: BLE001 - any failure here is a reportable result
        report.check(
            "pre_step_callback_registers",
            False,
            f"{type(err).__name__}: {err} — without this the Isaac e-stop cannot reach the "
            "simulator at all",
        )
        return
    report.check("pre_step_callback_registers", True, f"handle {handle!r}")

    try:
        sm.step(steps=3)
    finally:
        try:
            sm.deregister_callback(handle)
        except Exception as err:  # noqa: BLE001 - teardown detail, recorded not fatal
            report.record("deregister_callback_error", f"{type(err).__name__}: {err}")

    report.record("pre_step_callback_args", seen["args"])
    report.record("pre_step_callback_kwargs", seen["kwargs"])
    report.record("pre_step_callback_threads", sorted(seen["threads"]))
    report.expect("pre_step_callback_fires_once_per_step", seen["calls"], 3)
    report.check(
        "pre_step_callback_runs_on_the_main_thread",
        seen["threads"] == {threading.main_thread().name},
        f"{sorted(seen['threads'])} — the drain writes gains, which is main-thread-only",
    )


def check_render_does_not_step(report: Report, api: dict[str, Any]) -> None:
    """I. The guarantee the whole render path rests on.

    ``render_frames`` must never advance physics: the adapter owns the clock, and a render
    that steps behind its back would corrupt staleness detection and make the dq_max*dt clip
    stop being a velocity limit. RenderingManager.render() flips /app/player/playSimulations
    off, updates once, and restores it. This proves it on THIS build.
    """
    sm, rm = api["SimulationManager"], api["RenderingManager"]
    before = sm.get_num_physics_steps()
    rm.render()
    rm.render()
    after = sm.get_num_physics_steps()
    report.expect("render_advances_no_physics", after - before, 0)


def check_gains(report: Report, robot: Any) -> None:
    """J. The caller owns the gains — nothing may silently override them.

    This is why the backend is raw Isaac Sim and not Isaac Lab: Isaac Lab's explicit actuator
    models (DCMotorCfg, used for the G1's legs in G1_29DOF_CFG) compute torque in Python and
    neutralize the sim's PD gains. A sim that quietly re-tunes what we command makes every
    safety-intervention rate and every action label uncalibrated.
    """
    import numpy as np

    n = int(robot.num_dofs)
    kp = np.full((1, n), 123.0, dtype=np.float32)
    kd = np.full((1, n), 4.5, dtype=np.float32)
    robot.set_dof_gains(stiffnesses=kp, dampings=kd, update_default_gains=False)
    got_kp, got_kd = robot.get_dof_gains()
    got_kp, got_kd = np.asarray(got_kp.numpy()), np.asarray(got_kd.numpy())
    report.check(
        "gains_round_trip",
        np.allclose(got_kp, 123.0, atol=1e-3) and np.allclose(got_kd, 4.5, atol=1e-3),
        f"kp {float(got_kp.flat[0]):.4f} (want 123.0), kd {float(got_kd.flat[0]):.4f} (want 4.5)",
    )

    zero = np.zeros((1, n), dtype=np.float32)
    robot.set_dof_gains(stiffnesses=zero, dampings=kd, update_default_gains=False)
    got_kp, _ = robot.get_dof_gains()
    report.check(
        "zero_kp_accepted",
        bool(np.allclose(np.asarray(got_kp.numpy()), 0.0, atol=1e-6)),
        "kp=0 is the e-stop damping mode — it must not be clamped to a floor",
    )


def check_determinism(report: Report, api: dict[str, Any], robot: Any) -> None:
    """G'. Two identical short runs from the same state must agree.

    Not a guarantee NVIDIA makes across machines (GPU work scheduling reorders float ops), so
    this only establishes same-process reproducibility — which is the weaker claim the
    rollout manifest should carry. Recorded as a measured max-abs difference, not a boolean.
    """
    import numpy as np

    sm = api["SimulationManager"]
    n = int(robot.num_dofs)
    target = np.zeros((1, n), dtype=np.float32)

    def run() -> np.ndarray:
        robot.reset()
        sm.step(steps=1)
        robot.set_dof_position_targets(target)
        sm.step(steps=50)
        return np.asarray(robot.get_dof_positions().numpy()).copy()

    a, b = run(), run()
    delta = float(np.max(np.abs(a - b)))
    report.record("determinism_max_abs_delta_rad", delta)
    report.check(
        "same_process_determinism",
        delta == 0.0,
        f"max |q_a - q_b| = {delta:.3e} rad over 50 steps (0.0 = bit-identical)",
    )


def check_camera(report: Report, api: dict[str, Any], hw: tuple[int, int], warmup: int) -> None:
    """K. A real uint8 frame, and the warmup handled explicitly.

    The first frames come back as ``None``, not black — up to 20 of them in NVIDIA's own test.
    Code that does not gate on ``is not None`` records black frames into a dataset, which is
    strictly worse than a crash: black frames pass the T-11 data-quality gates and poison
    training silently. So the binding must fail loudly here, and this proves the timeout works.
    """
    import numpy as np
    import omni.replicator.core as rep

    rm = api["RenderingManager"]
    height, width = hw
    rep.orchestrator.set_capture_on_play(False)
    render_product = rep.create.render_product("/OmniverseKit_Persp", resolution=(width, height))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(render_product)

    frame, ticks = None, 0
    for ticks in range(1, warmup + 1):
        rm.render()
        data = annotator.get_data()
        if data is not None and getattr(data, "size", 0) > 0:
            frame = np.asarray(data)
            break

    report.record("camera_warmup_ticks", ticks)
    if not report.check(
        "camera_returns_a_frame",
        frame is not None,
        f"still None after {warmup} render ticks — raise --warmup-frames or add a dome light",
    ):
        return

    report.expect("camera_dtype_uint8", str(frame.dtype), "uint8")
    report.check(
        "camera_shape",
        frame.shape[:2] == (height, width) and frame.shape[2] in (3, 4),
        f"{frame.shape} (annotator returns RGBA; the binding slices [:, :, :3])",
    )
    std = float(np.asarray(frame[:, :, :3], dtype=np.float64).std())
    report.record("camera_pixel_std", std)
    report.check(
        "camera_frame_is_not_blank",
        std > 1.0,
        f"pixel std {std:.3f} — a uniform frame means no lighting, and it would pass a shape check",
    )


def check_threading(report: Report) -> None:
    """M. The main-thread rule, and the guard that enforces it.

    NVIDIA documents the prohibition ("All methods must be called from the main thread") but
    NOT the consequence of breaking it. Assume the worst — silently wrong readings — because
    that is the failure mode a safety layer cannot detect. This check does NOT violate the
    rule; it verifies that the identity test the binding uses to REFUSE such a call works.
    """
    main_id = threading.main_thread().ident
    seen: dict[str, Any] = {}

    def worker() -> None:
        seen["ident"] = threading.get_ident()
        seen["is_main"] = threading.current_thread() is threading.main_thread()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    report.check(
        "main_thread_guard_discriminates",
        seen.get("ident") != main_id and seen.get("is_main") is False,
        "a non-main thread is detectable, so the binding can refuse instead of corrupting state",
    )
    report.check(
        "preflight_runs_on_main_thread",
        threading.current_thread() is threading.main_thread(),
        "every Isaac call below was made from the main thread",
    )


def _import_simulation_app(report: Report) -> Any | None:
    """``isaacsim.SimulationApp``, or None with the reason recorded as a failed check.

    A traceback here would be the same defect this script exists to prevent: a preflight whose
    job is to report a broken environment must REPORT it, not raise out of ``main`` before the
    ``--out`` report is written. Not having Isaac Sim installed is the single most likely state
    of a box on which someone runs this, and it is a check result like any other.
    """
    try:
        # SimulationApp MUST be constructed before any other omni import — the extension system
        # provides those modules and is not loaded until this returns.
        from isaacsim import SimulationApp
    except ImportError as err:
        report.check(
            "isaacsim_importable",
            False,
            f"{err} — run this with Isaac Sim's own interpreter "
            "(`isaac-python scripts/preflight_isaac.py`), not the repo venv. Nothing below "
            "this line ran, so nothing about this box's Isaac install is proven.",
        )
        return None
    report.check("isaacsim_importable", True, "isaacsim.SimulationApp resolved")
    return SimulationApp


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = Report()
    started = time.perf_counter()

    check_environment(report)
    check_threading(report)

    simulation_app = _import_simulation_app(report)
    if simulation_app is not None:
        app = simulation_app({"headless": not args.gui, "renderer": "RaytracedLighting"})
        try:
            import isaacsim

            report.record("isaacsim_version", getattr(isaacsim, "__version__", "unknown"))
            report.check("simulation_app_started", True, f"headless={not args.gui}")

            api = check_api_surface(report)
            check_rates(report, api, args.hz, args.device)

            asset = args.asset
            if asset is None:
                from isaacsim.storage.native import get_assets_root_path

                root = get_assets_root_path()
                report.check(
                    "asset_root_resolves", bool(root), root or "get_assets_root_path() -> None"
                )
                asset = f"{root}{DEFAULT_ASSET_SUBPATH}"

            robot = check_articulation(report, api, asset)
            check_tick(report, api)
            check_pre_step_callback(report, api)
            check_render_does_not_step(report, api)
            check_gains(report, robot)
            check_determinism(report, api, robot)
            check_camera(report, api, tuple(args.camera_hw), args.warmup_frames)
        finally:
            app.close()

    report.record("wall_s", round(time.perf_counter() - started, 1))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"checks": report.checks, "info": report.info}, indent=2))
        print(f"\nreport -> {out}", flush=True)

    if report.failed:
        print(f"\nFAILED: {', '.join(report.failed)}", flush=True)
        return 1
    print(f"\nALL {len(report.checks)} CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
