# `wam.runtime` — closed loop and inference server

**TL;DR** — Runs the receding-horizon control loop (observe → predict → filter → execute a
prefix → re-plan) and, optionally, serves the policy over WebSocket so inference can live on a
different machine. The executor is torch-free; only `policies.py` imports torch.

## Files

| File | Contains |
|------|----------|
| `executor.py` | `ClosedLoopExecutor`, `ExecutorConfig`, `RolloutResult`, `run_rollouts` (T-19) |
| `mock_loop.py` | `run_mock_loop`, `DummyPolicy` — the M0 end-to-end loop, no hardware (T-03) |
| `policies.py` | `CheckpointPolicy` — a trained checkpoint served as a `Policy` |
| `server.py` | `PolicyServer` — WebSocket inference server (T-20) |
| `client.py` | `RemotePolicy` — the client, also a `Policy` |
| `wire.py` | Versioned JSON wire protocol |

## The replanning semantic (FR-05)

Every cycle produces a **fresh** chunk from a **fresh** observation; only the first
`prefix_steps` of the safety-filtered chunk are executed and **the remainder is discarded** —
the next prediction replaces it. The robot only ever sees prefixes; no stale chunk tail
survives a re-plan.

Per cycle: `read_state` → render frames (if the adapter can) → `Observation` → timed
`policy.predict` → watchdog/deadline gates → `safety.filter` → `robot.execute(chunk, prefix)`
→ feed watchdog → success check.

## What the executor refuses to do

- **Never execute a late prediction.** A chunk arriving after `policy_deadline_ms` is
  discarded and `robot.hold()` is commanded. The watchdog is deliberately *not* fed on that
  path, so a persistently late policy eventually trips it.
- **Never keep the watchdog armed on bad data.** When the safety filter rejects the cycle
  (`nan_reject` / `schema_reject` / `state_reject` → HOLD chunk), the watchdog is not fed.
- **Never let a frozen robot hide.** A stalled robot also freezes `state.timestamp_ns` — which
  is the watchdog's clock. So the executor separately measures the uninterrupted reject streak
  on the **host** clock and escalates (HOLD or STOP, per the watchdog's action) once it exceeds
  the watchdog timeout.
- **Re-arm only after safety.** After a watchdog expiry the watchdog is fed only once the safe
  state has actually been commanded.

**Two clocks, on purpose.** `now_ns` and the normal watchdog feed/expiry run on *robot* time
(`state.timestamp_ns` — simulated for the mock, wall-clock-ish on hardware). Policy latency and
the stale-state escalation run on the *host* clock, because a stalled robot freezes its own
timestamps. `clock` is injectable so tests enforce deadlines deterministically.

## Logging

Exactly two record kinds, the shared rollout log contract: one `control_cycle` line per cycle
and one `rollout_summary` per rollout, each stamped with `run_id` + `config_hash` by
`JsonlRunLogger` (AC-04). `below_min_policy_rate` lives **only** on `RolloutResult`, not in the
summary record — the record keeps fixed contract keys and consumers derive the flag from
`policy_rate_hz`. The MVP floor is ≥ 2 Hz (PRD §11.1).

## `mock_loop.py`

The M0 loop: no hardware, no wall-clock control flow. Robot time is the mock adapter's
simulated clock, and watchdog time is injected from it, so runs are deterministic.

`DummyPolicy` is a sinusoid joint-delta policy whose phase derives **only** from
`observation.state.timestamp_ns` — identical timestamps yield bit-identical chunks (stateless
and replayable). Per joint j of N: `q_j(t) = A·(j/N)·(1 - cos(2πt/period))`, which starts at
zero velocity and has bounded velocity and acceleration. The defaults sit inside
`configs/safety/default.yaml`, so a clean run needs **zero** interventions — any intervention in
a mock run is a real signal.

`stall_at` injects simulated policy stalls (watchdog time jumps *after* prediction) so the
expiry path can be exercised deterministically.

## Remote inference (`server.py`, `client.py`, `wire.py`)

**Separation of concerns.** The server process runs **policy inference only**. It never talks to
motors and runs no safety code — the safety layer, watchdog and low-level control stay on the
robot side. A slow or dead server therefore cannot block robot safety: the client times out, and
the executor treats it as a deadline miss and holds.

`PolicyServer` handles `predict`, `ping` and `info`. Inference runs in the default executor
thread, so the event loop keeps serving pings and other connections while a (possibly torch)
policy is busy. Errors are **per message** — an error envelope goes back and the connection
stays open. `info` also advertises the policy's `RunMetadata` when it has one, so robot-side
rollout logs stay traceable even when inference is remote (AC-04).

`RemotePolicy` implements the `Policy` protocol, so the executor cannot tell it apart from a
local policy. It is synchronous with a hard timeout (raising builtin `TimeoutError` past
`timeout_s`), backed by a private event loop on a daemon thread; the connection opens lazily and
re-opens transparently after a drop. On timeout the connection is **dropped on purpose** — a
late reply must not desync the next request — and stale replies whose `msg_id` doesn't match are
discarded.

`wire.py` is stdlib `json` + base64 only: no msgpack, no pickle, no new dependencies. Arrays
travel as `{dtype, shape, data}` with base64 of the raw little-endian buffer, so float32 state
and action data round-trips **bit-exact** and images stay uint8. Every message is an envelope
`{wire_version, msg_id, type, payload}`; a **major** wire-version mismatch is rejected with
`version_mismatch`.

## `policies.py`

`CheckpointPolicy` is the only torch user in this package. It loads a trained `ActionOnlyModel`
checkpoint (safetensors with embedded config + `RunMetadata`), keeps it in eval mode and runs
every prediction under `torch.no_grad()` — identical observations give identical chunks. Its
`metadata` property carries the provenance the rollout log needs.
