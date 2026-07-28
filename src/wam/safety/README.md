# `wam.safety` — deterministic safety layer

**TL;DR** — The gate between policy output and robot. No ML, no torch, no randomness: identical
inputs produce identical outputs. Nothing learned may bypass it (FR-07). Every modification is
reported as a logged `SafetyIntervention`.

## Files

| File | Contains |
|------|----------|
| `config.py` | `SafetyConfig` — the frozen limit set, YAML-loadable |
| `layer.py` | `SafetyLayer` — reject / project / hold, per step and per joint |
| `watchdog.py` | `Watchdog` — chunk-timeout detection, caller-driven clock |

## `SafetyConfig`

All limits in **physical** canonical units — the same units decoders emit (identity
normalization, MVP). Per-joint tuples fix the canonical joint order and must match
`CanonicalSpaceSpec.joint_names`. Frozen pydantic: validated once at construction.

```
q_min / q_max        per-joint position limits [rad]
dq_max / ddq_max     per-joint max |velocity| [rad/s] / |acceleration| [rad/s²]
workspace_min/max    EE-mode AABB [m], in the frame the fk callable reports
ee_max_lin_vel_m_s   max EE translation speed [m/s] per step
ee_max_step_m        max per-step EE translation — the ONLY workspace bound without fk
gripper_rate_max     max gripper change per second (gripper unit is [0, 1])
chunk_timeout_s      watchdog timeout
timeout_policy       'hold' | 'stop' — the watchdog's decision on expiry
hold_dt_s            dt for synthesized HOLD chunks when the incoming dt is unusable
```

`from_yaml` / `to_yaml` roundtrip.

## `SafetyLayer.filter(state, chunk) -> (safe_chunk, interventions)`

An empty intervention list means the chunk passed unchanged. `filter` never mutates its inputs;
the only internal mutation is a monotonic `intervention_count`.

**1. Reject → HOLD.** Unusable input is replaced by a zero-delta single-step hold chunk:

| Kind | Trigger |
|------|---------|
| `nan_reject` | NaN/Inf anywhere in targets, gripper or `dt_s` |
| `schema_reject` | wrong shape / dtype / mode / `dt <= 0` / version mismatch |
| `state_reject` | the state cannot be integrated (invalid, wrong width, `q` flagged invalid) |

Out-of-range gripper values are *not* rejected — they are projectable, and get clamped below.

**2. Project, step-wise.** Never truncation of the whole chunk; per step, per joint.

Joint mode applies the limits in the order **acceleration → velocity → position**, which is what
makes them hold *simultaneously*: the velocity clamp can only shrink `|dv|`, and the position
clamp can only shrink the step while the running position is inside `[q_min, q_max]`.

EE mode scales translation deltas to `ee_max_lin_vel_m_s * dt`, then — if a forward-kinematics
callable was supplied — integrates positions and clamps them into the workspace AABB. Without
`fk` only the per-step magnitude is bounded (`ee_max_step_m`) and every filtered chunk carries a
`workspace_skipped` intervention. Quaternion rotation deltas pass through unmodified in v0: a
documented limitation, not an oversight.

**Out-of-limits start states never snap back.** If the robot begins outside its position limits
or outside the AABB (overtravel, miscalibration, someone moved the arm by hand), the plain
projection would produce one arbitrarily fast re-entry step. That step is re-clipped to the
velocity limit — `joint_limit_recovery` / `workspace_recovery` — so re-entry ramps back at legal
speed over several steps. It is a no-op whenever the position was already inside.

Gripper targets are clamped to `[0, 1]` (`gripper_range`) and rate-limited against the current
gripper state (`gripper_rate`).

## `Watchdog`

Caller-driven: **no threads, no wall clock**. Time is injected as monotonic nanoseconds, which
makes expiry fully deterministic and testable.

```
feed(now_ns)        re-arm — call on every accepted chunk / heartbeat
expired(now_ns)     True iff never fed, or > timeout since the last feed (at the deadline: not expired)
decide(now_ns)      WatchdogAction.HOLD | STOP, or None
intervention(...)   loggable record on expiry
```

Fail-safe by construction: a watchdog that has never been fed is **expired**. Expiry never
auto-resets; only `feed()` re-arms. On timeout the answer is HOLD or STOP — never "keep
executing the stale chunk".
