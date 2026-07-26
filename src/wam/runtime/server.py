"""WebSocket inference server for WAM policies (T-20, M4).

Separation of concerns (PRD FR-05/FR-07): this process runs POLICY INFERENCE only.
It never talks to motors and runs no safety code — the deterministic safety layer,
watchdog and low-level control stay on the robot side (the client). A slow or dead
server therefore can never block robot safety: the client times out and the executor
treats it as a deadline miss.

Transport: ``websockets`` 16.x (asyncio implementation), JSON wire protocol from
:mod:`wam.runtime.wire`. One request -> one response; errors are per-message
(error envelope), the connection stays open.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from typing import Any

from websockets.asyncio.server import Server, ServerConnection
from websockets.asyncio.server import serve as ws_serve
from websockets.exceptions import ConnectionClosed

from wam.interfaces import INTERFACES_VERSION, SCHEMA_VERSION
from wam.interfaces.protocols import Policy
from wam.runtime import wire

__all__ = ["PolicyServer"]

_START_TIMEOUT_S = 10.0


class PolicyServer:
    """Serves a :class:`~wam.interfaces.protocols.Policy` over WebSocket.

    - ``predict``: payload = wire observation -> payload = wire chunk. Inference runs in
      the default executor thread so the event loop (and other connections) never block.
    - ``ping``: latency probe; echoes the request payload and adds ``server_time_ns``.
    - ``info``: policy class + wire/schema/interfaces versions + supported request types.

    Use ``await serve()`` inside an existing event loop, or ``run_in_thread()`` from
    synchronous code/tests (supports ``port=0`` -> OS-assigned port).
    """

    def __init__(self, policy: Policy, host: str = "127.0.0.1", port: int = 0) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._server: Server | None = None
        self._started = threading.Event()
        self._startup_error: BaseException | None = None
        self._actual_port: int | None = None

    # ------------------------------------------------------------------ serve

    @property
    def actual_port(self) -> int | None:
        """Bound port once serving (resolves ``port=0``), else None."""
        return self._actual_port

    async def serve(self) -> None:
        """Bind, serve until :meth:`stop`, then shut down gracefully."""
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        try:
            async with ws_serve(self._handle_connection, self._host, self._port) as server:
                self._server = server
                sockets = server.sockets or ()
                if sockets:
                    self._actual_port = int(sockets[0].getsockname()[1])
                self._started.set()
                await self._stop_event.wait()
                server.close(close_connections=True)
                await server.wait_closed()
        except BaseException as exc:  # startup failures must unblock run_in_thread()
            self._startup_error = exc
            self._started.set()
            raise
        finally:
            self._server = None

    def run_in_thread(self) -> tuple[threading.Thread, int]:
        """Start :meth:`serve` on a background thread; return ``(thread, actual_port)``.

        For tests and simple hosts. Raises RuntimeError if the server fails to bind.
        """
        self._started.clear()
        self._startup_error = None
        thread = threading.Thread(target=self._thread_main, name="wam-policy-server", daemon=True)
        thread.start()
        if not self._started.wait(timeout=_START_TIMEOUT_S):
            self.stop()
            thread.join(timeout=_START_TIMEOUT_S)
            raise RuntimeError("PolicyServer did not start within timeout")
        if self._startup_error is not None:
            thread.join(timeout=_START_TIMEOUT_S)
            raise RuntimeError(f"PolicyServer failed to start: {self._startup_error!r}")
        assert self._actual_port is not None
        return thread, self._actual_port

    def _thread_main(self) -> None:
        # Failures are recorded in _startup_error by serve(); never kill the interpreter.
        with contextlib.suppress(BaseException):
            asyncio.run(self.serve())

    def stop(self) -> None:
        """Request graceful shutdown. Thread-safe, idempotent, safe before start."""
        loop, stop_event = self._loop, self._stop_event
        if loop is None or stop_event is None:
            return
        try:
            loop.call_soon_threadsafe(stop_event.set)
        except RuntimeError:  # loop already closed
            pass

    # ------------------------------------------------------------- connection

    async def _handle_connection(self, connection: ServerConnection) -> None:
        """Per-message try/except: any failure -> error envelope, connection stays up."""
        try:
            async for message in connection:
                await connection.send(await self._handle_message(message))
        except ConnectionClosed:
            pass

    async def _handle_message(self, message: bytes | str) -> bytes:
        msg_id: str | None = None
        try:
            envelope = wire.decode_envelope(message)
            raw_id = envelope.get("msg_id")
            msg_id = str(raw_id) if raw_id is not None else None
            msg_type = envelope.get("type")
            payload = envelope.get("payload")
            if msg_type == "predict":
                reply = await self._predict(payload)
            elif msg_type == "ping":
                reply = self._ping(payload)
            elif msg_type == "info":
                reply = self._info()
            else:
                raise wire.WireError(
                    wire.ERR_UNKNOWN_TYPE,
                    f"unknown request type {msg_type!r}; supported: {wire.REQUEST_TYPES}",
                )
            return wire.encode_envelope(str(msg_type), reply, msg_id or "")
        except wire.WireError as exc:
            return wire.encode_error(exc.code, exc.detail, msg_id)
        except Exception as exc:  # noqa: BLE001 — server must survive any handler bug
            return wire.encode_error(wire.ERR_POLICY_ERROR, f"{type(exc).__name__}: {exc}", msg_id)

    # --------------------------------------------------------------- handlers

    async def _predict(self, payload: Any) -> dict[str, Any]:
        observation = wire.observation_from_payload(payload)
        loop = asyncio.get_running_loop()
        # Run inference in the default executor: the loop keeps serving pings /
        # other connections while the policy (possibly torch) is busy.
        chunk = await loop.run_in_executor(None, self._policy.predict, observation)
        return wire.chunk_to_payload(chunk)

    def _ping(self, payload: Any) -> dict[str, Any]:
        return {"echo": payload, "server_time_ns": time.time_ns()}

    def _info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "policy_class": type(self._policy).__name__,
            "wire_version": wire.WIRE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "interfaces_version": INTERFACES_VERSION,
            "request_types": list(wire.REQUEST_TYPES),
        }
        # AC-04 provenance over the wire: policies that expose RunMetadata (e.g.
        # CheckpointPolicy) advertise checkpoint/dataset refs so robot-side rollout
        # logs stay traceable even when inference runs remotely.
        metadata = getattr(self._policy, "metadata", None)
        to_dict = getattr(metadata, "to_dict", None)
        if callable(to_dict):
            try:
                info["metadata"] = dict(to_dict())
            except Exception as exc:  # noqa: BLE001 — info must never take the connection down
                info["metadata_error"] = f"{type(exc).__name__}: {exc}"
        return info
