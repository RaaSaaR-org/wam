# `wam.interfaces` — core contracts

**TL;DR** — The vocabulary every other package speaks: what a robot state is, what an action
chunk is, and which methods a swappable module must expose. Robot-agnostic, torch-free, and
versioned. Change with care — everything depends on this.

## Files

| File | Contains |
|------|----------|
| `schema.py` | Canonical state/action data types (PRD Anhang A, FR-06) |
| `protocols.py` | Structural interfaces for every swappable module (FR-09) |
| `versioning.py` | Config loading, deterministic config hashes, run provenance (FR-10, AC-04) |

## `schema.py` — the data

- `CanonicalSpaceSpec` — robot-agnostic definition of the space. `joint_names` fixes the
  canonical joint **order** used by every `q` / `dq` / `targets` array. `target_dim(mode)`
  gives the per-step width `D`.
- `RobotState` — `timestamp_ns`, `q`, `dq`, `imu`, `gripper_state`, plus a `ValidityMask`
  marking which field groups are actually trustworthy. Policies must survive a group flagged
  invalid; a missing sensor is not a crash.
- `ActionChunk` — `[T, D]` targets in **physical** units (rad for `JOINT_DELTA`, m + quaternion
  for `EE_DELTA`), a `[T]` gripper target in `[0, 1]`, and `dt_s`. `duration == T * dt_s`.
- `ActionMode` — `JOINT_DELTA` vs `EE_DELTA` (PRD OD-02, still open).
- `NormalizationSpec` — affine per-dim normalization. **Parked for the MVP**: the pipeline is
  identity-normalized end to end. A spec may be stored in a manifest for provenance, but nothing
  applies it, and `EpisodeDataset` refuses episodes declaring a non-identity spec rather than
  silently training on raw units.

**Store vs. reject.** Array containers are plain dataclasses on the hot path — construction
*never* raises on bad values. `validate()` returns a list of human-readable problems (empty list
== valid) and the safety layer is what rejects. Specs and configs are pydantic models and *do*
validate at construction, because they are built once, not per control step.

## `protocols.py` — the seams

All protocols are `@runtime_checkable` and structural: any object with matching members
conforms, no inheritance required. Use `isinstance`, never a base class.

```
StateEncoder    RobotState  -> embedding tensor
ActionEncoder   ActionChunk -> latent tensor        (TRAINING ONLY, never at runtime)
BackboneAdapter condition_{video,text,state}() then features() -> intermediate features
ActionDecoder   features    -> ActionChunk          (physical units)
SafetyFilter    (state, chunk) -> (safe chunk, [SafetyIntervention])
RobotAdapter    read_state() / execute(chunk, prefix_steps) / hold() / estop()
Policy          Observation -> ActionChunk
```

Plus two carrier dataclasses: `Observation` (camera images by name + state + instruction) and
`SafetyIntervention` (stable machine-readable `kind`, human-readable `detail`, timestamp).

Tensor-valued arguments are typed `Any` to keep this package torch-free; the contract is that
the **last** dimension is the documented feature dim. Anything crossing the robot or safety
boundary is numpy.

## `versioning.py` — traceability

- `config_hash(obj)` — SHA-256 over a canonicalized JSON form. Identical logical content hashes
  identically regardless of key order, tuple-vs-list, or pydantic-model-vs-dict. Unsupported
  types raise instead of being silently coerced, because lossy coercion would break the hash.
- `RunMetadata` — frozen provenance record: `run_id`, `config_hash`, `git_commit`, schema and
  interface versions, checkpoint ref, dataset snapshot ref, timestamp. Clock and git commit are
  injectable for deterministic tests.
- `JsonlRunLogger` — append-only JSONL; stamps **every** record with `run_id` + `config_hash`,
  so any rollout traces back to checkpoint + dataset + config (AC-04). Stamps override
  caller-supplied keys.
- `load_config(path)` — YAML loader that enforces a top-level `wam_config_version` whose major
  matches `WAM_CONFIG_VERSION`. Mismatch fails loudly.

## Versions

`SCHEMA_VERSION`, `INTERFACES_VERSION`, `WAM_CONFIG_VERSION` — all `0.1.0`. Only the **major**
component is enforced; a major mismatch is an error, not a warning.
