"""Synchronous WebSocket client implementing the Policy protocol (T-20, M4).

``RemotePolicy`` lets the robot-side executor treat a remote inference server exactly
like a local policy: ``predict(Observation) -> ActionChunk``, blocking, with a hard
timeout. Past ``timeout_s`` it raises builtin :class:`TimeoutError` — the closed-loop
executor treats that as a deadline miss and falls back to its deterministic safety
behavior (hold/stop). The learned model can therefore never stall the robot (FR-05/07).

Internals: a private asyncio event loop on a daemon thread; the WebSocket connection is
opened lazily on first use and re-opened transparently after a drop or timeout.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import threading
import uuid
from concurrent.futures import TimeoutError as _FutureTimeoutError
from typing import Any

from typing_extensions import Self
from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from wam.interfaces import ActionChunk, Observation
from wam.runtime import wire

__all__ = ["RemotePolicy", "RemotePolicyError"]


class RemotePolicyError(RuntimeError):
    """Server-side error envelope surfaced to the caller. ``code`` is the wire code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class RemotePolicy:
    """Policy-protocol client for :class:`~wam.runtime.server.PolicyServer`.

    Synchronous by design (the executor is synchronous); context manager; thread-safe
    for one caller at a time (calls are serialized by an internal lock).
    """

    def __init__(self, uri: str, timeout_s: float = 1.0) -> None:
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        self._uri = uri
        self._timeout_s = float(timeout_s)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ws: ClientConnection | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._msg_counter = itertools.count()

    # ------------------------------------------------------------- Policy API

    def predict(self, observation: Observation) -> ActionChunk:
        """Remote ``predict``; raises TimeoutError past ``timeout_s`` (deadline miss)."""
        payload = wire.observation_to_payload(observation)
        reply = self._request("predict", payload)
        return wire.chunk_from_payload(reply)

    def info(self) -> dict[str, Any]:
        """Server/policy metadata (policy class, versions, supported request types)."""
        reply = self._request("info", None)
        return dict(reply) if isinstance(reply, dict) else {"info": reply}

    def ping(self) -> dict[str, Any]:
        """Latency probe; returns the server's ping reply (includes server_time_ns)."""
        reply = self._request("ping", {"client": "wam.runtime.client"})
        return dict(reply) if isinstance(reply, dict) else {"echo": reply}

    # ---------------------------------------------------------------- request

    def _request(self, msg_type: str, payload: Any) -> Any:
        with self._lock:
            if self._closed:
                raise RuntimeError("RemotePolicy is closed")
            loop = self._ensure_loop()
            msg_id = f"{next(self._msg_counter)}-{uuid.uuid4().hex[:8]}"
            future = asyncio.run_coroutine_threadsafe(
                self._request_async(msg_type, payload, msg_id), loop
            )
            try:
                return future.result(timeout=self._timeout_s)
            except _FutureTimeoutError:
                future.cancel()
                # Drop the connection: a late reply must not desync the next request.
                self._drop_connection(loop)
                raise TimeoutError(
                    f"remote policy {msg_type!r} exceeded timeout_s={self._timeout_s}"
                ) from None

    async def _request_async(self, msg_type: str, payload: Any, msg_id: str) -> Any:
        """Send one request, await the matching reply; one transparent reconnect."""
        for attempt in (0, 1):
            ws = await self._ensure_connection()
            try:
                await ws.send(wire.encode_envelope(msg_type, payload, msg_id))
                while True:
                    envelope = wire.decode_envelope(await ws.recv())
                    if envelope.get("type") == wire.ERROR_TYPE:
                        raise RemotePolicyError(
                            str(envelope.get("code", "unknown")),
                            str(envelope.get("detail", "")),
                        )
                    if envelope.get("msg_id") == msg_id:
                        return envelope.get("payload")
                    # stale reply from an earlier timed-out request — discard
            except (ConnectionClosed, OSError):
                self._ws = None
                if attempt:  # second failure: give up
                    raise
        raise AssertionError("unreachable")

    # ------------------------------------------------------------- connection

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Lazily start the private event-loop thread."""
        if self._loop is None:
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever, name="wam-remote-policy", daemon=True
            )
            thread.start()
            self._loop = loop
            self._thread = thread
        return self._loop

    async def _ensure_connection(self) -> ClientConnection:
        if self._ws is None:
            self._ws = await ws_connect(self._uri, open_timeout=self._timeout_s)
        return self._ws

    def _drop_connection(self, loop: asyncio.AbstractEventLoop) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            asyncio.run_coroutine_threadsafe(self._close_ws(ws), loop)

    @staticmethod
    async def _close_ws(ws: ClientConnection) -> None:
        with contextlib.suppress(Exception):  # best-effort cleanup
            await ws.close()

    # ----------------------------------------------------------------- close

    def close(self) -> None:
        """Close connection and stop the internal loop thread. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop, thread, ws = self._loop, self._thread, self._ws
            self._loop = None
            self._thread = None
            self._ws = None
        if loop is None:
            return
        if ws is not None:
            future = asyncio.run_coroutine_threadsafe(self._close_ws(ws), loop)
            with contextlib.suppress(Exception):  # best-effort cleanup
                future.result(timeout=self._timeout_s)
        # cancel stragglers (keepalive / late replies) so the loop shuts down cleanly
        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(_cancel_pending_tasks(), loop).result(timeout=2.0)
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5.0)
        loop.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


async def _cancel_pending_tasks() -> None:
    """Cancel every task on the private loop except the current one, then reap them."""
    current = asyncio.current_task()
    tasks = [task for task in asyncio.all_tasks() if task is not current]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
