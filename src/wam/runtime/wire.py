"""Versioned WebSocket wire protocol for the WAM inference server (T-20).

Design:
- stdlib ``json`` + base64 only — no msgpack, no pickle, no new dependencies.
- Arrays travel as ``{"dtype", "shape", "data"}`` with base64 of the raw little-endian
  buffer: float32 state/action data round-trips BIT-EXACT, images stay uint8.
- Every message is an envelope ``{wire_version, msg_id, type, payload}``; errors are
  ``{wire_version, msg_id, type: "error", code, detail}``. A major-version mismatch is
  rejected by the server with code ``version_mismatch``.

The wire layer is transport-agnostic and torch-free; it only depends on
``wam.interfaces`` dataclasses (schema stays canonical — FR-06).
"""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np

from wam.interfaces import ActionChunk, ActionMode, IMUState, Observation, RobotState, ValidityMask

WIRE_VERSION = "0.1.0"

#: request types the server understands (also reported by the ``info`` reply).
REQUEST_TYPES = ("predict", "ping", "info")

ERROR_TYPE = "error"

# Stable machine-readable error codes.
ERR_BAD_REQUEST = "bad_request"
ERR_BAD_PAYLOAD = "bad_payload"
ERR_VERSION_MISMATCH = "version_mismatch"
ERR_UNKNOWN_TYPE = "unknown_type"
ERR_POLICY_ERROR = "policy_error"


class WireError(ValueError):
    """Malformed or incompatible wire data. ``code`` is a stable error-envelope code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def wire_major(version: str) -> str:
    """Major component of a wire version string ('0.1.0' -> '0')."""
    return str(version).split(".", 1)[0]


# ---------------------------------------------------------------------------
# array <-> json
# ---------------------------------------------------------------------------


def array_to_json(arr: np.ndarray) -> dict[str, Any]:
    """Encode a numpy array as ``{dtype, shape, data(b64 of little-endian bytes)}``."""
    a = np.ascontiguousarray(arr)
    if a.dtype.byteorder == ">":  # normalize to little-endian on the wire
        a = a.astype(a.dtype.newbyteorder("<"))
    return {
        "dtype": a.dtype.str.lstrip("<>=|"),
        "shape": [int(d) for d in a.shape],
        "data": base64.b64encode(a.tobytes()).decode("ascii"),
    }


def array_from_json(data: Any) -> np.ndarray:
    """Decode :func:`array_to_json` output. Bit-exact for the encoded buffer."""
    if not isinstance(data, dict):
        raise WireError(ERR_BAD_PAYLOAD, f"array must be an object, got {type(data).__name__}")
    try:
        dtype = np.dtype(str(data["dtype"])).newbyteorder("<")
        shape = tuple(int(d) for d in data["shape"])
        raw = base64.b64decode(str(data["data"]), validate=True)
        arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
    except (KeyError, ValueError, TypeError) as exc:
        raise WireError(ERR_BAD_PAYLOAD, f"bad array field: {exc}") from None
    # native byte order, writable copy (frombuffer is read-only)
    return arr.astype(arr.dtype.newbyteorder("="), copy=True)


def _f32(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# Observation <-> payload / bytes
# ---------------------------------------------------------------------------


def observation_to_payload(obs: Observation) -> dict[str, Any]:
    """Observation -> JSON-safe payload dict (images keep dtype, state full-fidelity f32)."""
    state = obs.state
    return {
        "images": {str(name): array_to_json(img) for name, img in obs.images.items()},
        "state": {
            "timestamp_ns": int(state.timestamp_ns),
            "q": array_to_json(_f32(state.q)),
            "dq": array_to_json(_f32(state.dq)),
            "imu": {
                "orientation_wxyz": array_to_json(_f32(state.imu.orientation_wxyz)),
                "angular_velocity": array_to_json(_f32(state.imu.angular_velocity)),
                "linear_acceleration": array_to_json(_f32(state.imu.linear_acceleration)),
            },
            "gripper_state": array_to_json(_f32(state.gripper_state)),
            "validity": {k: bool(v) for k, v in state.validity.as_dict().items()},
            "schema_version": str(state.schema_version),
        },
        "instruction": str(obs.instruction),
    }


def observation_from_payload(payload: Any) -> Observation:
    """Inverse of :func:`observation_to_payload`. Raises :class:`WireError` on bad data."""
    if not isinstance(payload, dict):
        raise WireError(ERR_BAD_PAYLOAD, "observation payload must be an object")
    try:
        state_p = payload["state"]
        imu_p = state_p["imu"]
        validity_p = state_p.get("validity", {})
        state = RobotState(
            timestamp_ns=int(state_p["timestamp_ns"]),
            q=array_from_json(state_p["q"]),
            dq=array_from_json(state_p["dq"]),
            imu=IMUState(
                orientation_wxyz=array_from_json(imu_p["orientation_wxyz"]),
                angular_velocity=array_from_json(imu_p["angular_velocity"]),
                linear_acceleration=array_from_json(imu_p["linear_acceleration"]),
            ),
            gripper_state=array_from_json(state_p["gripper_state"]),
            validity=ValidityMask(
                q=bool(validity_p.get("q", True)),
                dq=bool(validity_p.get("dq", True)),
                imu=bool(validity_p.get("imu", True)),
                gripper=bool(validity_p.get("gripper", True)),
            ),
            schema_version=str(state_p.get("schema_version", RobotState.schema_version)),
        )
        images = {str(name): array_from_json(img) for name, img in dict(payload["images"]).items()}
        instruction = str(payload.get("instruction", ""))
    except WireError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise WireError(ERR_BAD_PAYLOAD, f"bad observation payload: {exc!r}") from None
    return Observation(images=images, state=state, instruction=instruction)


def encode_observation(obs: Observation) -> bytes:
    """Observation -> canonical JSON bytes (utf-8)."""
    return _dumps(observation_to_payload(obs))


def decode_observation(data: bytes | str) -> Observation:
    """JSON bytes/str -> Observation (state arrays f32 bit-exact)."""
    return observation_from_payload(_loads(data))


# ---------------------------------------------------------------------------
# ActionChunk <-> payload / bytes
# ---------------------------------------------------------------------------


def chunk_to_payload(chunk: ActionChunk) -> dict[str, Any]:
    """ActionChunk -> JSON-safe payload dict (targets/gripper f32 bit-exact)."""
    return {
        "mode": str(chunk.mode.value),
        "targets": array_to_json(_f32(chunk.targets)),
        "gripper_target": array_to_json(_f32(chunk.gripper_target)),
        "dt_s": float(chunk.dt_s),
        "schema_version": str(chunk.schema_version),
    }


def chunk_from_payload(payload: Any) -> ActionChunk:
    """Inverse of :func:`chunk_to_payload`. Raises :class:`WireError` on bad data."""
    if not isinstance(payload, dict):
        raise WireError(ERR_BAD_PAYLOAD, "chunk payload must be an object")
    try:
        chunk = ActionChunk(
            mode=ActionMode(str(payload["mode"])),
            targets=array_from_json(payload["targets"]),
            gripper_target=array_from_json(payload["gripper_target"]),
            dt_s=float(payload["dt_s"]),
            schema_version=str(payload.get("schema_version", ActionChunk.schema_version)),
        )
    except WireError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise WireError(ERR_BAD_PAYLOAD, f"bad chunk payload: {exc!r}") from None
    return chunk


def encode_chunk(chunk: ActionChunk) -> bytes:
    """ActionChunk -> canonical JSON bytes (utf-8), f32 bit-exact round-trip."""
    return _dumps(chunk_to_payload(chunk))


def decode_chunk(data: bytes | str) -> ActionChunk:
    """JSON bytes/str -> ActionChunk."""
    return chunk_from_payload(_loads(data))


# ---------------------------------------------------------------------------
# envelopes
# ---------------------------------------------------------------------------


def encode_envelope(
    msg_type: str,
    payload: Any,
    msg_id: str,
    *,
    wire_version: str = WIRE_VERSION,
) -> bytes:
    """Request/response envelope ``{wire_version, msg_id, type, payload}`` as JSON bytes.

    ``wire_version`` is overridable for version-gate tests only.
    """
    return _dumps(
        {
            "wire_version": str(wire_version),
            "msg_id": str(msg_id),
            "type": str(msg_type),
            "payload": payload,
        }
    )


def encode_error(code: str, detail: str, msg_id: str | None = None) -> bytes:
    """Error envelope ``{wire_version, msg_id, type: "error", code, detail}``."""
    return _dumps(
        {
            "wire_version": WIRE_VERSION,
            "msg_id": msg_id,
            "type": ERROR_TYPE,
            "code": str(code),
            "detail": str(detail),
        }
    )


def decode_envelope(data: bytes | str) -> dict[str, Any]:
    """Parse an envelope and gate the wire version.

    Raises :class:`WireError` with ``bad_request`` (not JSON / not an object / missing
    fields) or ``version_mismatch`` (major version differs from :data:`WIRE_VERSION`).
    """
    envelope = _loads(data)
    if not isinstance(envelope, dict):
        raise WireError(ERR_BAD_REQUEST, "envelope must be a JSON object")
    version = envelope.get("wire_version")
    if not isinstance(version, str) or not version:
        raise WireError(ERR_BAD_REQUEST, "envelope missing 'wire_version'")
    if wire_major(version) != wire_major(WIRE_VERSION):
        raise WireError(
            ERR_VERSION_MISMATCH,
            f"wire_version {version!r} incompatible with server {WIRE_VERSION!r} (major mismatch)",
        )
    if not isinstance(envelope.get("type"), str):
        raise WireError(ERR_BAD_REQUEST, "envelope missing 'type'")
    return envelope


# ---------------------------------------------------------------------------
# json helpers
# ---------------------------------------------------------------------------


def _dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _loads(data: bytes | str) -> Any:
    if isinstance(data, (bytes, bytearray)):
        try:
            data = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WireError(ERR_BAD_REQUEST, f"message is not utf-8: {exc}") from None
    try:
        return json.loads(data)
    except json.JSONDecodeError as exc:
        raise WireError(ERR_BAD_REQUEST, f"message is not valid JSON: {exc}") from None
