"""T-20 tests: wire protocol round-trips, PolicyServer + RemotePolicy end-to-end,
error envelopes, timeout -> TimeoutError, wire version gate.

All CPU-only and deterministic; async parts run via asyncio.run inside sync tests
(pytest-asyncio is intentionally not used).
"""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
import pytest
from websockets.asyncio.client import connect as ws_connect

from wam.interfaces import (
    INTERFACES_VERSION,
    SCHEMA_VERSION,
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    IMUState,
    Observation,
    RobotState,
    ValidityMask,
)
from wam.interfaces.protocols import Policy
from wam.runtime import wire
from wam.runtime.client import RemotePolicy, RemotePolicyError
from wam.runtime.mock_loop import DummyPolicy
from wam.runtime.server import PolicyServer

SPEC = CanonicalSpaceSpec(
    joint_names=tuple(f"joint_{i}" for i in range(6)),
    gripper_dims=1,
)


def _make_state(seed: int = 0, timestamp_ns: int = 1_000_000_000) -> RobotState:
    rng = np.random.default_rng(seed)
    return RobotState(
        timestamp_ns=timestamp_ns,
        q=rng.uniform(-3, 3, SPEC.num_joints).astype(np.float32),
        dq=rng.uniform(-2, 2, SPEC.num_joints).astype(np.float32),
        imu=IMUState(
            orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
            angular_velocity=rng.uniform(-1, 1, 3).astype(np.float32),
            linear_acceleration=rng.uniform(-9, 9, 3).astype(np.float32),
        ),
        gripper_state=rng.uniform(0, 1, SPEC.gripper_dims).astype(np.float32),
        validity=ValidityMask(q=True, dq=True, imu=False, gripper=True),
    )


def _make_observation(seed: int = 0) -> Observation:
    rng = np.random.default_rng(seed + 100)
    images = {
        "front": rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8),
        "wrist": rng.integers(0, 256, size=(16, 24, 3), dtype=np.uint8),
    }
    return Observation(
        images=images,
        state=_make_state(seed),
        instruction="Greife die rote Tasse — vorsichtig öffnen.",
    )


def _make_chunk(seed: int = 0, steps: int = 8) -> ActionChunk:
    rng = np.random.default_rng(seed + 200)
    return ActionChunk(
        mode=ActionMode.JOINT_DELTA,
        targets=rng.uniform(-1, 1, (steps, SPEC.num_joints)).astype(np.float32),
        gripper_target=rng.uniform(0, 1, steps).astype(np.float32),
        dt_s=0.05,
    )


class SlowPolicy:
    """Policy that always overruns the client deadline."""

    def __init__(self, spec: CanonicalSpaceSpec, sleep_s: float) -> None:
        self._inner = DummyPolicy(spec)
        self._sleep_s = sleep_s

    def predict(self, observation: Observation) -> ActionChunk:
        time.sleep(self._sleep_s)
        return self._inner.predict(observation)


class ExplodingPolicy:
    def predict(self, observation: Observation) -> ActionChunk:
        raise ValueError("kaboom")


# ---------------------------------------------------------------------------
# wire round-trips
# ---------------------------------------------------------------------------


def test_wire_observation_roundtrip_bit_exact() -> None:
    obs = _make_observation(seed=3)
    # awkward but legal float32 values must survive bit-exact
    obs.state.q[0] = np.float32(-0.0)
    obs.state.q[1] = np.float32(1e-39)  # subnormal
    obs.state.q[2] = np.float32(0.1)

    data = wire.encode_observation(obs)
    assert isinstance(data, bytes)
    decoded = wire.decode_observation(data)

    assert decoded.instruction == obs.instruction
    assert set(decoded.images) == set(obs.images)
    for cam, img in obs.images.items():
        assert decoded.images[cam].dtype == np.uint8
        assert decoded.images[cam].shape == img.shape
        assert decoded.images[cam].tobytes() == img.tobytes()

    s0, s1 = obs.state, decoded.state
    assert s1.timestamp_ns == s0.timestamp_ns
    assert s1.schema_version == s0.schema_version
    for name in ("q", "dq", "gripper_state"):
        a0, a1 = getattr(s0, name), getattr(s1, name)
        assert a1.dtype == np.float32
        assert a1.tobytes() == a0.tobytes()  # bit-exact, incl. -0.0 and subnormals
    for name in ("orientation_wxyz", "angular_velocity", "linear_acceleration"):
        assert getattr(s1.imu, name).tobytes() == getattr(s0.imu, name).tobytes()
    assert s1.validity.as_dict() == s0.validity.as_dict()
    assert s1.validate(SPEC) == []


def test_wire_chunk_roundtrip_bit_exact() -> None:
    chunk = _make_chunk(seed=7)
    chunk.targets[0, 0] = np.float32(-0.0)
    chunk.targets[0, 1] = np.float32(0.1)

    decoded = wire.decode_chunk(wire.encode_chunk(chunk))
    assert decoded.mode is ActionMode.JOINT_DELTA
    assert decoded.dt_s == chunk.dt_s
    assert decoded.schema_version == chunk.schema_version
    assert decoded.targets.dtype == np.float32
    assert decoded.targets.shape == chunk.targets.shape
    assert decoded.targets.tobytes() == chunk.targets.tobytes()
    assert decoded.gripper_target.tobytes() == chunk.gripper_target.tobytes()
    assert decoded.validate(SPEC) == []
    # double round-trip is stable
    again = wire.decode_chunk(wire.encode_chunk(decoded))
    assert again.targets.tobytes() == chunk.targets.tobytes()


def test_wire_envelope_and_errors() -> None:
    env = wire.decode_envelope(wire.encode_envelope("ping", {"x": 1}, "id-1"))
    assert env["wire_version"] == wire.WIRE_VERSION
    assert env["type"] == "ping"
    assert env["msg_id"] == "id-1"
    assert env["payload"] == {"x": 1}

    with pytest.raises(wire.WireError) as exc_info:
        wire.decode_envelope(b"not json at all")
    assert exc_info.value.code == wire.ERR_BAD_REQUEST

    with pytest.raises(wire.WireError) as exc_info:
        wire.decode_envelope(json.dumps({"type": "ping"}).encode())  # no wire_version
    assert exc_info.value.code == wire.ERR_BAD_REQUEST

    with pytest.raises(wire.WireError) as exc_info:
        wire.decode_envelope(wire.encode_envelope("ping", None, "x", wire_version="1.0.0"))
    assert exc_info.value.code == wire.ERR_VERSION_MISMATCH

    # same major, newer minor: accepted
    env = wire.decode_envelope(wire.encode_envelope("ping", None, "x", wire_version="0.9.5"))
    assert env["wire_version"] == "0.9.5"

    with pytest.raises(wire.WireError):
        wire.chunk_from_payload({"mode": "joint_delta"})  # missing arrays
    with pytest.raises(wire.WireError):
        wire.observation_from_payload("not a dict")


# ---------------------------------------------------------------------------
# server end-to-end
# ---------------------------------------------------------------------------


def test_server_end_to_end_predict_with_dummy_policy() -> None:
    policy = DummyPolicy(SPEC)
    server = PolicyServer(policy, host="127.0.0.1", port=0)
    thread, port = server.run_in_thread()
    assert port > 0  # port=0 resolved to a real port
    try:
        with RemotePolicy(f"ws://127.0.0.1:{port}", timeout_s=5.0) as remote:
            assert isinstance(remote, Policy)  # runtime-checkable protocol

            obs = _make_observation(seed=1)
            remote_chunk = remote.predict(obs)
            local_chunk = policy.predict(obs)  # DummyPolicy is deterministic + stateless

            assert remote_chunk.validate(SPEC) == []
            assert remote_chunk.mode is local_chunk.mode
            assert remote_chunk.dt_s == local_chunk.dt_s
            assert remote_chunk.targets.tobytes() == local_chunk.targets.tobytes()
            assert remote_chunk.gripper_target.tobytes() == local_chunk.gripper_target.tobytes()

            # several sequential predicts on one connection
            for seed in (2, 3):
                obs_i = _make_observation(seed=seed)
                got = remote.predict(obs_i)
                want = policy.predict(obs_i)
                assert got.targets.tobytes() == want.targets.tobytes()

            info = remote.info()
            assert info["policy_class"] == "DummyPolicy"
            assert info["wire_version"] == wire.WIRE_VERSION
            assert info["schema_version"] == SCHEMA_VERSION
            assert info["interfaces_version"] == INTERFACES_VERSION
            assert set(info["request_types"]) == {"predict", "ping", "info"}

            pong = remote.ping()
            assert pong["server_time_ns"] > 0
    finally:
        server.stop()
        thread.join(timeout=10.0)
    assert not thread.is_alive()


def test_remote_policy_reconnects_after_drop() -> None:
    policy = DummyPolicy(SPEC)
    server = PolicyServer(policy, host="127.0.0.1", port=0)
    thread, port = server.run_in_thread()
    try:
        with RemotePolicy(f"ws://127.0.0.1:{port}", timeout_s=5.0) as remote:
            obs = _make_observation(seed=4)
            first = remote.predict(obs)

            # kill the underlying connection behind the client's back
            ws, loop = remote._ws, remote._loop
            assert ws is not None and loop is not None
            asyncio.run_coroutine_threadsafe(ws.close(), loop).result(timeout=5.0)

            second = remote.predict(obs)  # transparently reconnects
            assert second.targets.tobytes() == first.targets.tobytes()
    finally:
        server.stop()
        thread.join(timeout=10.0)


def test_server_error_envelope_on_garbage_connection_survives() -> None:
    server = PolicyServer(DummyPolicy(SPEC), host="127.0.0.1", port=0)
    thread, port = server.run_in_thread()

    async def scenario() -> None:
        async with ws_connect(f"ws://127.0.0.1:{port}") as ws:
            # 1) raw garbage -> bad_request error envelope
            await ws.send(b"\xff\xfenot json")
            reply = json.loads(await ws.recv())
            assert reply["type"] == "error"
            assert reply["code"] == wire.ERR_BAD_REQUEST
            assert reply["wire_version"] == wire.WIRE_VERSION

            # 2) valid envelope, unknown type -> unknown_type, msg_id echoed
            await ws.send(wire.encode_envelope("selfdestruct", None, "m-2"))
            reply = json.loads(await ws.recv())
            assert reply["type"] == "error"
            assert reply["code"] == wire.ERR_UNKNOWN_TYPE
            assert reply["msg_id"] == "m-2"

            # 3) predict with broken payload -> bad_payload
            await ws.send(wire.encode_envelope("predict", {"images": {}}, "m-3"))
            reply = json.loads(await ws.recv())
            assert reply["type"] == "error"
            assert reply["code"] == wire.ERR_BAD_PAYLOAD

            # 4) connection is still alive: a normal ping succeeds
            await ws.send(wire.encode_envelope("ping", {"n": 1}, "m-4"))
            reply = json.loads(await ws.recv())
            assert reply["type"] == "ping"
            assert reply["msg_id"] == "m-4"
            assert reply["payload"]["echo"] == {"n": 1}

    try:
        asyncio.run(scenario())
    finally:
        server.stop()
        thread.join(timeout=10.0)


def test_server_version_gate() -> None:
    server = PolicyServer(DummyPolicy(SPEC), host="127.0.0.1", port=0)
    thread, port = server.run_in_thread()

    async def scenario() -> None:
        async with ws_connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(wire.encode_envelope("info", None, "v-1", wire_version="1.0.0"))
            reply = json.loads(await ws.recv())
            assert reply["type"] == "error"
            assert reply["code"] == wire.ERR_VERSION_MISMATCH

            # compatible minor bump still answered
            await ws.send(wire.encode_envelope("info", None, "v-2", wire_version="0.99.0"))
            reply = json.loads(await ws.recv())
            assert reply["type"] == "info"
            assert reply["payload"]["policy_class"] == "DummyPolicy"

    try:
        asyncio.run(scenario())
    finally:
        server.stop()
        thread.join(timeout=10.0)


def test_policy_exception_becomes_error_envelope_and_client_raises() -> None:
    server = PolicyServer(ExplodingPolicy(), host="127.0.0.1", port=0)
    thread, port = server.run_in_thread()
    try:
        with RemotePolicy(f"ws://127.0.0.1:{port}", timeout_s=5.0) as remote:
            with pytest.raises(RemotePolicyError) as exc_info:
                remote.predict(_make_observation())
            assert exc_info.value.code == wire.ERR_POLICY_ERROR
            assert "kaboom" in exc_info.value.detail
            # connection still serves info afterwards
            assert remote.info()["policy_class"] == "ExplodingPolicy"
    finally:
        server.stop()
        thread.join(timeout=10.0)


def test_remote_policy_timeout_raises_timeouterror() -> None:
    server = PolicyServer(SlowPolicy(SPEC, sleep_s=0.6), host="127.0.0.1", port=0)
    thread, port = server.run_in_thread()
    try:
        remote = RemotePolicy(f"ws://127.0.0.1:{port}", timeout_s=0.15)
        try:
            start = time.monotonic()
            with pytest.raises(TimeoutError):
                remote.predict(_make_observation())
            elapsed = time.monotonic() - start
            assert elapsed < 0.5  # deadline enforced, does not wait for the slow policy
        finally:
            remote.close()
    finally:
        server.stop()
        thread.join(timeout=10.0)


def test_remote_policy_close_is_idempotent_and_blocks_reuse() -> None:
    remote = RemotePolicy("ws://127.0.0.1:1", timeout_s=0.2)
    remote.close()
    remote.close()
    with pytest.raises(RuntimeError):
        remote.info()


def test_connect_failure_surfaces_within_timeout() -> None:
    # nothing listens on this port; RemotePolicy must fail fast, not hang
    with RemotePolicy("ws://127.0.0.1:9", timeout_s=0.3) as remote:
        start = time.monotonic()
        with pytest.raises((TimeoutError, RemotePolicyError, OSError)):
            remote.predict(_make_observation())
        assert time.monotonic() - start < 3.0
