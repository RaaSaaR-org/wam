# `wam.data` — episode format, capture, replay, validation

**TL;DR** — Records demonstrations to disk in a versioned, checksummed format; reads them back
bit-exactly; and runs the dataset validation gates. Data quality is a **release gate**, not a
nice-to-have. Torch-free (numpy + pyarrow, plus av/opencv for video).

## Files

| File | Contains |
|------|----------|
| `episode.py` | `EpisodeWriter` / `EpisodeReader` / `EpisodeManifest`, `list_episodes` (PRD Anhang A) |
| `capture.py` | `SyncRecorder`, source protocols, `MockCaptureSession` (FR-01) |
| `replay.py` | `replay_episode`, `episode_report` (FR-08) |
| `validation.py` | `validate_episode` / `validate_dataset` — the gates (T-11, R-04) |
| `calibration.py` | `CalibrationSet` — storage and validation of calibration results |

## On-disk layout

```
<episode_dir>/
  manifest.json               versioned index: spec, cameras, tables, checksums, stats
  <camera>.mp4                one lossy video stream per camera (PyAV/x264, cv2 fallback)
  <camera>_timestamps.parquet per-frame capture timestamps
  states.parquet              canonical RobotState rows
  actions.parquet             ActionChunk steps flattened via chunk_idx / step_idx
```

The timestamp sidecars exist because **mp4 pts are not sufficient for sync analysis** — the
capture time of a frame is not the same thing as its presentation time.

Guarantees:

- **Streaming writer.** Frames are encoded as they arrive, never all held in RAM. State/action
  rows are buffered as small plain rows and flushed on `close()`.
- **Exact roundtrip** for states and actions (float32, bit-exact through parquet). Video
  roundtrip is *structural* only: shape and dtype exact, pixel values within lossy tolerance.
- **Checksums.** The manifest carries a sha256 of every data file; the reader verifies them on
  open (`verify_checksums=False` opts out).
- **Versioned.** `EPISODE_FORMAT_VERSION` and the schema major must match on read.
- **No half-written episodes.** If an exception escapes the `with` block, encoders are released
  but **no manifest is written** — so the partial directory is invisible to `list_episodes`.

`normalization` in the manifest is provenance only. Nothing in the MVP pipeline applies it.

## Capture (`capture.py`)

**Caller-driven stepping: no threads, no wall-clock sleeps.** One `SyncRecorder.step()` pulls
one state and one frame per camera, checks the timestamp spread against `sync_tolerance_ns`,
and *only then* writes. That makes capture deterministic and lets it run on MockRobot's
simulated clock.

A violating sample is never partially written: with `on_violation="raise"` the step raises
*before* anything reaches the writer; with `"flag"` it is recorded, marked
`within_tolerance=False` and counted. `max_sync_error_ns` tracks the worst spread seen,
including raised and flagged samples.

Sources are pull protocols — `FrameSource.capture() -> (rgb, ts)` and
`StateSource.capture() -> RobotState`. `MockCameraSource` supports a fixed `offset_ns` to model
per-sensor clock skew and exercise the tolerance path.

`MockCaptureSession` wires MockRobot + policy + safety into a full episode without hardware.
Per iteration: sync sample → policy → safety filter → **record the SAFE chunk** (post-filter —
what was actually commanded) with the actually executed prefix → execute on the mock robot.

## Replay and reports (`replay.py`)

`replay_episode(reader)` merges states, frames and commanded chunks into one time-ordered
stream of `ReplayStep`. Events sharing a timestamp merge into one step; a collision on an
already-filled slot (a second state, a second frame of the same camera, a second chunk) starts
a new step at the same `t_ns`, so **no event is ever dropped**.

`episode_report(reader)` computes an `EpisodeReport` from stored tables + manifest only — the
video is **not** decoded. Statistics are float64 aggregates over bit-exact float32 rows, so they
reproduce exactly offline (FR-08). It covers per-camera timing (nominal vs actual fps), per-joint
range and velocity coverage, and action aggregates (executed ratio, step norms, gripper
activity). `render_markdown()` prints the lot.

`flags` are findings for gates and operators — `no_states`, `validity_gap:<group>`,
`discarded_chunks`, `sync_error_exceeds_tolerance`, … — not hard errors. A broken episode still
yields a report.

## Validation gates (`validation.py`)

Release gates. **Gates never raise:** a crashing gate is a failed gate with the exception in
`detail`, and bad data means failed gates, not an exception.

| Episode gate | Checks |
|---|---|
| `readable` | the directory opens at all |
| `checksums` | every file matches its manifest sha256 |
| `monotonic_timestamps` | states strictly increasing; actions and frames non-decreasing |
| `sync_error` | camera↔camera spread and worst state↔frame alignment vs tolerance |
| `finite_values` | no NaN/Inf in states or chunks |
| `state_coverage` | valid-flagged fields have canonical shapes; q/dq coverage ≥ minimum |
| `counts` | table rows match the manifest; target dims and executed prefixes consistent |
| `duration` | within `[min_duration_s, max_duration_s]` |
| `frame_integrity` | every stream decodes and is non-empty |

If the **checksum gate fails, the remaining gates are skipped** — their results would be
meaningless on tampered data — and the report contains exactly that one failed gate.

Dataset level adds `episode_count`, `total_duration`, `episodes_valid` and `unique_episode_ids`.

All thresholds live in `ValidationThresholds` (frozen). The defaults suit the mock D0 setup and
**must be tuned per real deployment**. Reports are JSON-serializable so they can be archived
next to the dataset snapshot (AC-04).

## Calibration (`calibration.py`)

Storage and validation only — **there is no solver here**. Producing the numbers (ChArUco,
hand-eye, joint-zero measurement) is a hardware workflow. This module makes the *result*
versioned, hashable and loadable, so every episode and rollout can reference the exact
calibration it was recorded under.

`CameraIntrinsics` (pinhole + distortion model), `CameraExtrinsics` (`T_parent_camera`, unit
quaternion enforced) and `CalibrationSet` (per-camera intrinsics/extrinsics + per-joint zero
offsets + provenance). Units are meters and radians, quaternions w,x,y,z — same as the canonical
schema. `config_hash()` is the content hash to store in
`EpisodeWriter(extra={"calibration_hash": ...})` and in run logs. YAML roundtrips losslessly and
is gated on `wam_config_version`.
