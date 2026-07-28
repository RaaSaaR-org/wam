#!/usr/bin/env python3
"""DDS conformance check for :class:`wam.robot.g1_transport.DdsG1Transport` (T-21, OD-08).

Contracts:

- **Runs INSIDE the arm64 container, never on the macOS host.** It needs
  ``unitree_sdk2py`` + CycloneDDS, which only exist in the image built from the sibling
  ``Dockerfile``. The WAM repo is bind-mounted at ``/wam``; this script imports the
  working-tree ``wam.robot``, not a copy.
- **No robot.** A fake DDS peer stands in for the G1's onboard controller: it publishes
  ``LowState_``/``HandState_`` and subscribes ``LowCmd_``/``HandCmd_`` on the same domain.
  The peer runs as a SEPARATE PROCESS on purpose — two independent DomainParticipants must
  discover each other and exchange RTPS over the loopback interface, which is a strictly
  stronger claim than one process talking to itself. Parent<->peer coordination uses a
  stdin/stdout line protocol (sentinel-prefixed JSON), never DDS, so a DDS failure can
  never be masked by the control channel.
- **Every check reports exactly one PASS / FAIL / SKIP line.** A check whose prerequisite
  failed is SKIPped with the blocking reason, never silently dropped. Exit code is 0 only
  when nothing FAILed; SKIPs do not fail the run (they are recorded in README.md).
- **What this validates:** the wire/transport layer only — IDL field mapping, CRC, topic
  names, tick semantics, DDS discovery. It is NOT a physics simulation and proves nothing
  about the robot's reaction to a command.
- Torch-free; numpy + the vendor SDK only.

Usage (inside the container)::

    python3 /wam/docker/dds/conformance.py [--interface lo] [--domain 0]
    python3 /wam/docker/dds/conformance.py --peer      # internal: fake-robot child process
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# The repo is mounted at /wam; PYTHONPATH already points at /wam/src in the image, but keep
# the script runnable from a plain `python3 docker/dds/conformance.py` too.
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

#: Peer responses are prefixed with this so the SDK's own chatter on stdout cannot corrupt
#: the control protocol.
PEER_SENTINEL = "@@PEER@@ "
PEER_STATE_RATE_HZ = 200.0
PEER_TICK_STEP_MS = 5  # 200 Hz -> 5 ms per published LowState_ (deterministic, no wall clock)
NUM_MOTORS = 29
NUM_MOTOR_SLOTS = 35
DEX3_MOTORS = 7


# ==========================================================================================
# Fake robot peer (child process)
# ==========================================================================================


class FakeG1Peer:
    """Stands in for the G1's onboard controller on the DDS bus.

    Publishes ``rt/lowstate`` at a fixed rate with a deterministic tick, mirrors commanded
    hand positions on ``rt/dex3/{left,right}/state``, and records the last ``rt/lowcmd`` /
    ``rt/dex3/*/cmd`` it received together with an INDEPENDENTLY RECOMPUTED CRC.
    """

    def __init__(self, domain_id: int, interface: str) -> None:
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import (
            unitree_hg_msg_dds__HandState_,
            unitree_hg_msg_dds__LowState_,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
            HandCmd_,
            HandState_,
            LowCmd_,
            LowState_,
        )
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(domain_id, interface)
        self._crc = CRC()

        self._lock = threading.Lock()
        self._q = np.zeros(NUM_MOTORS, dtype=np.float64)
        self._dq = np.zeros(NUM_MOTORS, dtype=np.float64)
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._gyro = np.zeros(3)
        self._acc = np.array([0.0, 0.0, 9.81])
        self._hand_q = np.zeros(2, dtype=np.float64)
        self._mode_machine = 4  # arbitrary non-zero: proves the transport echoes it back
        self._tick_ms = 1000
        self._paused = False
        self._last_cmd: dict[str, Any] | None = None
        self._cmd_count = 0
        self._last_hand_cmd: list[dict[str, Any] | None] = [None, None]

        self._state = unitree_hg_msg_dds__LowState_()
        self._hand_state = [unitree_hg_msg_dds__HandState_() for _ in range(2)]

        self._state_pub = ChannelPublisher("rt/lowstate", LowState_)
        self._state_pub.Init()
        self._hand_state_pubs = []
        for topic in ("rt/dex3/left/state", "rt/dex3/right/state"):
            pub = ChannelPublisher(topic, HandState_)
            pub.Init()
            self._hand_state_pubs.append(pub)

        self._cmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self._cmd_sub.Init(self._on_low_cmd, 0)
        self._hand_cmd_subs = []
        for side, topic in enumerate(("rt/dex3/left/cmd", "rt/dex3/right/cmd")):
            sub = ChannelSubscriber(topic, HandCmd_)
            sub.Init(self._make_hand_cmd_handler(side), 0)
            self._hand_cmd_subs.append(sub)

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._publish_loop, name="peer_state", daemon=True)

    # -- lifecycle -------------------------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    # -- publishing ------------------------------------------------------------------------

    def _publish_loop(self) -> None:
        period = 1.0 / PEER_STATE_RATE_HZ
        while not self._stop.is_set():
            with self._lock:
                paused = self._paused
                if not paused:
                    self._tick_ms += PEER_TICK_STEP_MS
                    self._fill_state()
                    state = self._state
                    hands = list(self._hand_state)
            if not paused:
                self._state_pub.Write(state)
                for pub, hand in zip(self._hand_state_pubs, hands):
                    pub.Write(hand)
            time.sleep(period)

    def _fill_state(self) -> None:
        """Caller holds the lock."""
        state = self._state
        state.tick = self._tick_ms
        state.mode_machine = self._mode_machine
        state.mode_pr = 0
        for i in range(NUM_MOTORS):
            state.motor_state[i].q = float(self._q[i])
            state.motor_state[i].dq = float(self._dq[i])
        state.imu_state.quaternion = [float(v) for v in self._quat]
        state.imu_state.gyroscope = [float(v) for v in self._gyro]
        state.imu_state.accelerometer = [float(v) for v in self._acc]
        state.crc = self._crc.Crc(state)
        for side in range(2):
            for i in range(DEX3_MOTORS):
                self._hand_state[side].motor_state[i].q = float(self._hand_q[side])

    # -- subscriptions ---------------------------------------------------------------------

    def _on_low_cmd(self, msg: Any) -> None:
        received_crc = int(msg.crc)
        # Recompute over the received message: the vendor CRC covers every field except the
        # trailing crc word, so this is a genuine independent check, not an echo.
        msg.crc = 0
        expected = int(self._crc.Crc(msg))
        msg.crc = received_crc
        record = {
            "mode_pr": int(msg.mode_pr),
            "mode_machine": int(msg.mode_machine),
            "crc": received_crc,
            "crc_expected": expected,
            "crc_valid": received_crc == expected,
            "q": [float(msg.motor_cmd[i].q) for i in range(NUM_MOTOR_SLOTS)],
            "dq": [float(msg.motor_cmd[i].dq) for i in range(NUM_MOTOR_SLOTS)],
            "tau": [float(msg.motor_cmd[i].tau) for i in range(NUM_MOTOR_SLOTS)],
            "kp": [float(msg.motor_cmd[i].kp) for i in range(NUM_MOTOR_SLOTS)],
            "kd": [float(msg.motor_cmd[i].kd) for i in range(NUM_MOTOR_SLOTS)],
            "mode": [int(msg.motor_cmd[i].mode) for i in range(NUM_MOTOR_SLOTS)],
        }
        with self._lock:
            self._last_cmd = record
            self._cmd_count += 1

    def _make_hand_cmd_handler(self, side: int) -> Callable[[Any], None]:
        def _handler(msg: Any) -> None:
            motors = list(msg.motor_cmd)
            record = {
                "n": len(motors),
                "q": [float(m.q) for m in motors],
                "kp": [float(m.kp) for m in motors],
                "kd": [float(m.kd) for m in motors],
                "mode": [int(m.mode) for m in motors],
            }
            with self._lock:
                self._last_hand_cmd[side] = record
                # Mirror the command into the reported hand state (a perfect servo), so the
                # gripper read path has something to observe.
                if motors:
                    self._hand_q[side] = float(np.mean([m.q for m in motors]))

        return _handler

    # -- control protocol ------------------------------------------------------------------

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op")
        if op == "set_state":
            with self._lock:
                for key, target in (("q", self._q), ("dq", self._dq)):
                    if key in request:
                        target[:] = np.asarray(request[key], dtype=np.float64)
                if "quat" in request:
                    self._quat = np.asarray(request["quat"], dtype=np.float64)
                if "gyro" in request:
                    self._gyro = np.asarray(request["gyro"], dtype=np.float64)
                if "acc" in request:
                    self._acc = np.asarray(request["acc"], dtype=np.float64)
                if "hand" in request:
                    self._hand_q[:] = np.asarray(request["hand"], dtype=np.float64)
            return {"ok": True}
        if op == "pause":
            with self._lock:
                self._paused = True
            return {"ok": True}
        if op == "resume":
            with self._lock:
                self._paused = False
            return {"ok": True}
        if op == "last_cmd":
            with self._lock:
                return {"ok": True, "cmd": self._last_cmd, "count": self._cmd_count}
        if op == "last_hand_cmd":
            with self._lock:
                return {"ok": True, "hand_cmd": list(self._last_hand_cmd)}
        if op == "reset_cmd":
            with self._lock:
                self._last_cmd = None
                self._cmd_count = 0
                self._last_hand_cmd = [None, None]
            return {"ok": True}
        if op == "quit":
            return {"ok": True, "bye": True}
        return {"ok": False, "error": f"unknown op {op!r}"}


def run_peer(domain_id: int, interface: str) -> int:
    """Child-process entry point: serve the control protocol on stdin/stdout."""
    peer = FakeG1Peer(domain_id, interface)
    peer.start()
    _emit({"ok": True, "ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"ok": False, "error": f"bad request: {exc}"})
            continue
        try:
            response = peer.handle(request)
        except Exception as exc:  # noqa: BLE001 - the parent must see the failure, not a hang
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        _emit(response)
        if response.get("bye"):
            break
    peer.stop()
    return 0


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(PEER_SENTINEL + json.dumps(payload) + "\n")
    sys.stdout.flush()


# ==========================================================================================
# Peer handle (parent process)
# ==========================================================================================


class PeerProcess:
    """Parent-side handle for the fake-robot child process."""

    def __init__(self, domain_id: int, interface: str, timeout_s: float = 20.0) -> None:
        self._timeout_s = timeout_s
        self._responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self._proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--peer",
                "--domain",
                str(domain_id),
                "--interface",
                interface,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # Cyclone's own diagnostics stay visible in the run log
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._read_loop, name="peer_reader", daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            if line.startswith(PEER_SENTINEL):
                try:
                    self._responses.put(json.loads(line[len(PEER_SENTINEL) :]))
                except json.JSONDecodeError:
                    pass
            else:
                sys.stderr.write("[peer] " + line)

    def request(self, **payload: Any) -> dict[str, Any]:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()
        try:
            response = self._responses.get(timeout=self._timeout_s)
        except queue.Empty as exc:
            raise RuntimeError(f"peer did not answer {payload.get('op')!r} in time") from exc
        if not response.get("ok"):
            raise RuntimeError(f"peer error on {payload.get('op')!r}: {response.get('error')}")
        return response

    def wait_ready(self) -> None:
        try:
            first = self._responses.get(timeout=self._timeout_s)
        except queue.Empty as exc:
            raise RuntimeError("peer process never reported DDS ready") from exc
        if not first.get("ready"):
            raise RuntimeError(f"unexpected first peer message: {first}")

    def shutdown(self) -> None:
        try:
            if self._proc.poll() is None:
                self.request(op="quit")
        except Exception as exc:  # noqa: BLE001 - shutdown is best effort, never fatal
            sys.stderr.write(f"[peer] shutdown request failed: {type(exc).__name__}: {exc}\n")
        finally:
            if self._proc.poll() is None:
                self._proc.terminate()
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()


# ==========================================================================================
# Report
# ==========================================================================================


@dataclass
class Row:
    name: str
    status: str
    detail: str


class Report:
    """Ordered PASS/FAIL/SKIP ledger. A failure blocks every later check as SKIP."""

    def __init__(self) -> None:
        self.rows: list[Row] = []
        self.blocked_by: str | None = None

    def run(self, name: str, fn: Callable[[], str]) -> bool:
        if self.blocked_by is not None:
            self._add(name, "SKIP", f"prerequisite {self.blocked_by!r} failed")
            return False
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001 - a check failure must be reported, not raised
            self._add(name, "FAIL", f"{type(exc).__name__}: {exc}")
            return False
        self._add(name, "PASS", detail)
        return True

    def run_blocking(self, name: str, fn: Callable[[], str]) -> bool:
        """Like :meth:`run`, but a failure here SKIPs everything after it."""
        ok = self.run(name, fn)
        if not ok and self.blocked_by is None:
            self.blocked_by = name
        return ok

    def skip(self, name: str, reason: str) -> None:
        self._add(name, "SKIP", reason)

    def _add(self, name: str, status: str, detail: str) -> None:
        self.rows.append(Row(name, status, detail))
        print(f"[{status}] {name:<26} {detail}", flush=True)

    def summary(self) -> int:
        counts = {
            status: sum(r.status == status for r in self.rows)
            for status in ("PASS", "FAIL", "SKIP")
        }
        print()
        print(f"{counts['PASS']} PASS  {counts['FAIL']} FAIL  {counts['SKIP']} SKIP")
        return 1 if counts["FAIL"] else 0


# ==========================================================================================
# Checks
# ==========================================================================================


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _close(actual: Any, expected: Any, tol: float = 1e-6) -> bool:
    return bool(
        np.allclose(
            np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64), atol=tol
        )
    )


def _wait_until(predicate: Callable[[], bool], message: str, timeout_s: float = 3.0) -> None:
    """Poll ``predicate`` until it holds; raise ``TimeoutError(message)`` if it never does."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise TimeoutError(f"{message} (within {timeout_s}s)")


def _wait_for_new_tick(
    transport: Any, previous: int | None, timeout_s: float = 3.0
) -> dict[str, Any]:
    """Poll until the transport reports a tick different from ``previous``."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        low = transport.read_low_state()
        if previous is None or low["tick_ns"] != previous:
            return low
        time.sleep(0.005)
    raise TimeoutError(
        f"no LowState_ with a new tick within {timeout_s}s (last tick_ns={previous})"
    )


def _wait_for_cmd(peer: PeerProcess, minimum_count: int, timeout_s: float = 3.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        response = peer.request(op="last_cmd")
        if response["count"] >= minimum_count and response["cmd"] is not None:
            return response["cmd"]
        last = response
        time.sleep(0.01)
    raise TimeoutError(f"peer received no LowCmd_ within {timeout_s}s (last={last})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--interface", default="lo", help="network interface for DDS (default: lo)")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain id (default: 0)")
    parser.add_argument("--peer", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.peer:
        return run_peer(args.domain, args.interface)

    print("WAM G1 DDS transport conformance")
    print(f"  interface={args.interface!r} domain={args.domain} python={sys.version.split()[0]}")
    print()

    report = Report()
    ctx: dict[str, Any] = {}

    # -- 1. vendor SDK ---------------------------------------------------------------------
    def check_sdk_import() -> str:
        import unitree_sdk2py  # noqa: F401
        from unitree_sdk2py.utils.crc import CRC

        crc = CRC()
        lib = getattr(crc, "crc_lib", None)
        _assert(
            lib is not None, "CRC fell back to the pure-Python path (native crc lib not loaded)"
        )
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

        value = crc.Crc(unitree_hg_msg_dds__LowCmd_())
        _assert(isinstance(value, int), f"CRC returned {type(value).__name__}, expected int")
        return f"unitree_sdk2py imported; native CRC lib loaded; Crc(default LowCmd_)={value}"

    report.run_blocking("sdk_import", check_sdk_import)

    # -- 2. DDS init on loopback -----------------------------------------------------------
    def check_dds_init() -> str:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        ChannelFactoryInitialize(args.domain, args.interface)
        return f"ChannelFactoryInitialize({args.domain}, {args.interface!r}) ok"

    report.run_blocking("dds_init_loopback", check_dds_init)

    # -- 3. fake robot peer ----------------------------------------------------------------
    def check_peer() -> str:
        peer = PeerProcess(args.domain, args.interface)
        peer.wait_ready()
        ctx["peer"] = peer
        return "fake G1 peer process up (separate DomainParticipant, publishing rt/lowstate)"

    report.run_blocking("fake_peer_process", check_peer)

    try:
        # -- 4. transport connect ----------------------------------------------------------
        def check_connect() -> str:
            from wam.robot.g1 import G1Config
            from wam.robot.g1_transport import DdsG1Transport, G1Transport

            config = G1Config(network_interface=args.interface)
            transport = DdsG1Transport(config, domain_id=args.domain, state_timeout_s=10.0)
            _assert(
                isinstance(transport, G1Transport), "DdsG1Transport does not satisfy G1Transport"
            )
            transport.open()  # RuntimeError(SDK_MISSING_MSG) here would mean the SDK path failed
            _assert(transport.is_open, "open() returned without marking the transport open")
            ctx["transport"] = transport
            return f"DdsG1Transport.open() ok on {args.interface!r}, first LowState_ received"

        report.run_blocking("transport_connect", check_connect)

        # .get(): when a blocking check failed every later check is SKIPped, so these
        # closures never run — but the names must still resolve while they are being defined.
        peer: PeerProcess = ctx.get("peer")  # type: ignore[assignment]
        transport = ctx.get("transport")

        # -- 5. LowState_ round trip -------------------------------------------------------
        def check_lowstate() -> str:
            q_expected = np.array([0.01 * (i + 1) for i in range(NUM_MOTORS)])
            dq_expected = np.array([-0.002 * (i + 1) for i in range(NUM_MOTORS)])
            quat = [0.7071068, 0.0, 0.7071068, 0.0]
            gyro = [0.1, -0.2, 0.3]
            acc = [0.0, 0.0, 9.75]
            peer.request(
                op="set_state",
                q=q_expected.tolist(),
                dq=dq_expected.tolist(),
                quat=quat,
                gyro=gyro,
                acc=acc,
            )
            # Two fresh ticks: the first may still carry the pre-update sample.
            low = _wait_for_new_tick(transport, None)
            low = _wait_for_new_tick(transport, low["tick_ns"])
            low = _wait_for_new_tick(transport, low["tick_ns"])
            _assert(low["q"].shape == (NUM_MOTORS,), f"q shape {low['q'].shape}")
            _assert(low["q"].dtype == np.float32, f"q dtype {low['q'].dtype}")
            _assert(_close(low["q"], q_expected), f"q mismatch: {low['q'][:4]} vs {q_expected[:4]}")
            _assert(_close(low["dq"], dq_expected), "dq mismatch")
            _assert(_close(low["imu"]["quat_wxyz"], quat), "imu quaternion mismatch")
            _assert(_close(low["imu"]["gyro"], gyro), "imu gyro mismatch")
            _assert(_close(low["imu"]["acc"], acc), "imu acc mismatch")
            first = low["tick_ns"]
            second = _wait_for_new_tick(transport, first)["tick_ns"]
            _assert(second > first, f"tick did not advance: {first} -> {second}")
            _assert(
                (second - first) % 1_000_000 == 0,
                f"tick_ns is not a whole number of 1 ms vendor ticks: {second - first} ns",
            )
            ctx["tick_ns"] = second
            return f"q/dq/imu match the peer's payload; tick_ns advanced {first} -> {second}"

        report.run("lowstate_roundtrip", check_lowstate)

        # -- 6. LowCmd_ round trip + CRC ---------------------------------------------------
        def check_lowcmd() -> str:
            peer.request(op="reset_cmd")
            q_target = np.array([0.001 * (i + 1) for i in range(NUM_MOTORS)])
            dq_target = np.array([0.05 * (i + 1) for i in range(NUM_MOTORS)])
            kp = np.full(NUM_MOTORS, 20.0)
            kd = np.full(NUM_MOTORS, 0.5)
            transport.write_motor_cmd(q_target, dq_target, kp, kd)
            cmd = _wait_for_cmd(peer, 1)
            _assert(_close(cmd["q"][:NUM_MOTORS], q_target), "LowCmd_ q payload mismatch")
            _assert(_close(cmd["dq"][:NUM_MOTORS], dq_target), "LowCmd_ dq payload mismatch")
            _assert(_close(cmd["kp"][:NUM_MOTORS], kp), "LowCmd_ kp payload mismatch")
            _assert(_close(cmd["kd"][:NUM_MOTORS], kd), "LowCmd_ kd payload mismatch")
            _assert(_close(cmd["tau"][:NUM_MOTORS], np.zeros(NUM_MOTORS)), "LowCmd_ tau must be 0")
            _assert(
                all(m == 1 for m in cmd["mode"][:NUM_MOTORS]),
                "motors 0..28 must be mode=1 (enable)",
            )
            _assert(
                all(m == 0 for m in cmd["mode"][NUM_MOTORS:]),
                "unused motor slots 29..34 must stay disabled (mode=0)",
            )
            _assert(
                cmd["crc_valid"],
                f"CRC invalid: got {cmd['crc']}, peer recomputed {cmd['crc_expected']}",
            )
            _assert(
                cmd["mode_machine"] == 4,
                f"mode_machine not echoed from LowState_: {cmd['mode_machine']}",
            )
            return (
                f"29-motor payload exact; slots 29..34 disabled; mode_machine echoed; "
                f"CRC {cmd['crc']} verified by the peer"
            )

        report.run("lowcmd_roundtrip_crc", check_lowcmd)

        # -- 7. CRC has teeth --------------------------------------------------------------
        def check_crc_detects_corruption() -> str:
            from unitree_sdk2py.core.channel import ChannelPublisher
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

            peer.request(op="reset_cmd")
            raw_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
            raw_pub.Init()
            bad = unitree_hg_msg_dds__LowCmd_()
            bad.motor_cmd[0].q = 0.42
            bad.crc = 12345  # deliberately wrong
            raw_pub.Write(bad)
            cmd = _wait_for_cmd(peer, 1)
            _assert(not cmd["crc_valid"], "peer accepted a deliberately corrupted CRC")
            raw_pub.Close()
            return "a deliberately wrong CRC is rejected by the peer (the CRC check is not vacuous)"

        report.run("lowcmd_crc_detects_corruption", check_crc_detects_corruption)

        # -- 8. emergency damp -------------------------------------------------------------
        def check_damp() -> str:
            peer.request(op="reset_cmd")
            transport.emergency_damp()
            cmd = _wait_for_cmd(peer, 1)
            expected_kd = transport.damp_kd
            _assert(_close(cmd["kp"][:NUM_MOTORS], np.zeros(NUM_MOTORS)), "damping must set kp=0")
            _assert(
                _close(cmd["kd"][:NUM_MOTORS], np.full(NUM_MOTORS, expected_kd)),
                "damping kd mismatch",
            )
            _assert(_close(cmd["q"][:NUM_MOTORS], np.zeros(NUM_MOTORS)), "damping must set q=0")
            _assert(_close(cmd["dq"][:NUM_MOTORS], np.zeros(NUM_MOTORS)), "damping must set dq=0")
            _assert(_close(cmd["tau"][:NUM_MOTORS], np.zeros(NUM_MOTORS)), "damping must set tau=0")
            _assert(cmd["crc_valid"], "damping command CRC invalid")
            return f"kp=0, kd={expected_kd} on all 29 motors, q=dq=tau=0, CRC valid"

        report.run("emergency_damp", check_damp)

        # -- 9. Dex3 hand command / state --------------------------------------------------
        def check_gripper() -> str:
            peer.request(op="reset_cmd")
            transport.write_gripper_cmd(0.3, 0.7)
            deadline = time.monotonic() + 3.0
            hand_cmd = [None, None]
            while time.monotonic() < deadline:
                hand_cmd = peer.request(op="last_hand_cmd")["hand_cmd"]
                if all(h is not None for h in hand_cmd):
                    break
                time.sleep(0.01)
            _assert(all(h is not None for h in hand_cmd), "peer received no Dex3 HandCmd_")
            for side, expected in enumerate((0.3, 0.7)):
                _assert(
                    hand_cmd[side]["n"] == DEX3_MOTORS,
                    f"hand {side}: {hand_cmd[side]['n']} motors, want 7",
                )
                _assert(
                    _close(hand_cmd[side]["q"], [expected] * DEX3_MOTORS), f"hand {side} q mismatch"
                )
            # The peer mirrors the command into HandState_, so the read path closes the loop.
            deadline = time.monotonic() + 3.0
            gripper = None
            while time.monotonic() < deadline:
                low = transport.read_low_state()
                gripper = low.get("gripper")
                if gripper is not None and _close(gripper, [0.3, 0.7], tol=1e-5):
                    break
                time.sleep(0.01)
            _assert(gripper is not None, "no 'gripper' key: Dex3 HandState_ never arrived")
            _assert(
                _close(gripper, [0.3, 0.7], tol=1e-5),
                f"gripper readback {gripper}, want [0.3, 0.7]",
            )
            return "Dex3 HandCmd_ (7 motors/hand) delivered on rt/dex3/*/cmd; HandState_ readback matches"

        report.run("dex3_gripper_roundtrip", check_gripper)

        # -- 10. full adapter over the wire ------------------------------------------------
        def check_adapter() -> str:
            from wam.interfaces.schema import ActionChunk, ActionMode
            from wam.robot.g1 import G1_JOINT_MAP, G1Adapter, G1Config

            q_zero = np.zeros(NUM_MOTORS)
            peer.request(op="set_state", q=q_zero.tolist(), dq=q_zero.tolist())
            _wait_until(
                lambda: _close(transport.read_low_state()["q"], q_zero),
                "peer never reported the zeroed joint state",
            )
            adapter = G1Adapter(G1Config(network_interface=args.interface), transport=transport)
            adapter.connect()
            state = adapter.read_state()
            _assert(state.q.shape == (len(G1_JOINT_MAP),), f"canonical q shape {state.q.shape}")
            _assert(bool(state.validity.q), "fresh sample reported as invalid")

            n_joints = len(G1_JOINT_MAP)
            delta = 0.01
            targets = np.full((3, n_joints), delta, dtype=np.float32)
            chunk = ActionChunk(
                mode=ActionMode.JOINT_DELTA,
                targets=targets,
                gripper_target=np.full(3, 0.5, dtype=np.float32),
                dt_s=0.02,
            )
            # prefix_steps=1: exactly ONE LowCmd_ goes out, so the assertion below cannot be
            # confused by DDS keep-last dropping an intermediate step of a longer prefix.
            peer.request(op="reset_cmd")
            adapter.execute(chunk, prefix_steps=1)
            cmd = _wait_for_cmd(peer, 1)
            motor_q = np.asarray(cmd["q"][:NUM_MOTORS])
            for _, motor_index in G1_JOINT_MAP:
                _assert(
                    abs(motor_q[motor_index] - delta) < 1e-5,
                    f"motor {motor_index} got {motor_q[motor_index]}, want {delta}",
                )
            leg_indices = list(range(12))
            _assert(
                _close(motor_q[leg_indices], np.zeros(len(leg_indices))),
                "unmapped leg motors must hold their current position (0 here)",
            )
            _assert(cmd["crc_valid"], "adapter-issued command has an invalid CRC")
            return (
                f"G1Adapter.read_state() -> {n_joints} canonical joints; execute() lands "
                f"{delta} rad on exactly the mapped motor indices, legs held, CRC valid"
            )

        report.run("adapter_closed_loop", check_adapter)

        # -- 11. stale tick ----------------------------------------------------------------
        def check_stale() -> str:
            from wam.robot.g1 import G1Adapter, G1Config

            adapter = G1Adapter(G1Config(network_interface=args.interface), transport=transport)
            adapter.connect()
            _wait_for_new_tick(transport, None)
            fresh = adapter.read_state()
            _assert(bool(fresh.validity.q), "state before the pause should be fresh")
            peer.request(op="pause")
            time.sleep(0.1)  # let the last in-flight sample land
            adapter.read_state()  # consumes the final fresh tick
            stale = adapter.read_state()
            _assert(not stale.validity.q, "q validity survived a frozen vendor tick")
            _assert(not stale.validity.imu, "imu validity survived a frozen vendor tick")
            peer.request(op="resume")
            return "peer stops publishing -> tick frozen -> adapter clears every validity flag"

        report.run("stale_tick_degrades_validity", check_stale)

    finally:
        transport = ctx.get("transport")
        if transport is not None:
            transport.close()
        peer_proc = ctx.get("peer")
        if peer_proc is not None:
            peer_proc.shutdown()

    return report.summary()


if __name__ == "__main__":
    sys.exit(main())
