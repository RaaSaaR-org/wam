# Teleop + Calibration Workflow (T-10)

Status: the teleoperation system is **OD-07 (open)**. This page fixes the workflow contract
that any candidate must satisfy; it does not pick the device.

## Candidate systems (OD-07)

- **Leader–follower arm** (kinematically similar leader, e.g. scaled twin): lowest latency,
  direct joint mapping into the canonical space — preferred for the D0/D1 sets.
- **VR controller / hand tracking**: EE-delta streams + IK; more setup, needed later for
  bimanual work.

Either way the device must deliver operator commands at >= 20 Hz, map into the canonical
action space (`ActionMode.JOINT_DELTA` baseline, OD-02), and expose a deadman switch. The
physical E-stop stays independent of the teleop stack (PRD §11.2).

## Recording flow (teleop -> EpisodeWriter)

1. All producers stamp with ONE monotonic clock (robot host `CLOCK_MONOTONIC`); cameras are
   stamped at capture, not at encode (FR-01).
2. Operator input -> canonical `ActionChunk` -> **SafetyLayer** -> robot adapter. Teleop is
   never allowed to bypass safety; interventions are logged like policy interventions.
3. Per control tick the recorder calls `EpisodeWriter.add_state` (canonical `RobotState`
   from the robot adapter), `add_frame` per camera, and `add_action(chunk,
   executed_prefix, ts)` where `executed_prefix` counts the steps actually executed before
   replanning/abort (0 = chunk discarded).
4. `EpisodeWriter(extra=...)` records: `operator_id` (pseudonymized), `d_phase` (D0/D1/...),
   `teleop_device`, and `calibration_hash` = `CalibrationSet.config_hash()`.
5. On `close()` the manifest carries `max_sync_error_ns`; gate: it must stay below half a
   frame period (checked by `episode_report`, flag `sync_error_exceeds_tolerance`).
6. Review each episode with `scripts/episode_report.py <dir>` before it enters a dataset
   snapshot (T-11 automates this gate).

## Calibration checklist (before every recording block)

1. **Intrinsics** per camera (ChArUco board, full FoV coverage) -> `CalibrationSet.intrinsics`;
   record reprojection error in `extra` (gate: < 1 px RMS).
2. **Extrinsics** camera -> parent frame (hand-eye for wrist cam, board-in-base for fixed
   cam) -> `CalibrationSet.extrinsics`.
3. **Joint zero-offsets**: drive to the mechanical reference pose, record deltas ->
   `joint_offsets_rad` (applied only in the robot adapter, FR-06).
4. **Timestamp sync check**: record 30 s, run `episode_report`, verify sync error + fps.
5. Save as `configs/calibration/<date>.yaml` (schema: `configs/calibration/example.yaml`),
   never edit an existing file. Re-calibrate whenever a camera or mount moved, after any
   collision, and at the start of each D-phase.
