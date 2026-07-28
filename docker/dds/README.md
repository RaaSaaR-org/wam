# `docker/dds` — Unitree DDS conformance track

**TL;DR** — Runs `wam.robot.g1_transport.DdsG1Transport` against a fake G1 on a real
CycloneDDS bus, inside a `linux/arm64` container, on this Mac, with no robot. It validates the
**wire/transport layer**: IDL field mapping, CRC, topic names, tick semantics, DDS discovery.
It is **not** a simulator and says nothing about how a robot would move.

```bash
docker/dds/run.sh          # build + run; exit 0 = no FAIL
```

Current status (2026-07-27, Docker Desktop 28.5.2, arm64): **11 PASS, 0 FAIL, 0 SKIP**.

---

## Why this exists

`g1_transport.py` was the only hardware-stubbed file in the repo: `DdsG1Transport` raised
`NotImplementedError` on every method, so the entire vendor-facing surface was documentation.
The G1 EDU4's onboard computer is a Jetson Orin — **arm64 Linux**, the same target as an arm64
Linux container on an Apple-silicon Mac. Nothing here is emulated, so the vendor SDK, the
CycloneDDS build and the wire format are the real ones. That makes it possible to implement
and exercise the DDS methods months before the robot arrives.

The DDS methods in `src/wam/robot/g1_transport.py` are now implemented. `G1Transport`,
`FakeG1Transport` and `tests/test_g1.py` are untouched; `pytest tests/test_g1.py
tests/test_robot.py` passes on the host, where `unitree_sdk2py` is absent and every
`DdsG1Transport` method still raises `RuntimeError("G1 hardware support requires
unitree_sdk2py")`.

## Files

| File | Contains |
|------|----------|
| `Dockerfile` | ubuntu:22.04 arm64, CycloneDDS + `unitree_sdk2py` from pinned source, two-stage (204 MB final) |
| `conformance.py` | The check itself + the fake-robot peer (`--peer` child mode) |
| `run.sh` | Build the image, run the check with the repo mounted read-only at `/wam` |

## What is in the image

| Component | Pin | Note |
|-----------|-----|------|
| `ubuntu` | `22.04` (arm64) | matches the Jetson-class userspace |
| CycloneDDS (C) | tag `0.10.2` = `9995905` | built with `BUILD_IDLC=ON`, installed to `/opt/cyclonedds` (3.7 MB) |
| `cyclonedds` (Python) | `0.10.2` | no arm64 wheel exists; built from sdist against `CYCLONEDDS_HOME` |
| `unitree_sdk2py` | commit `65691c8` (v1.0.1) | no upstream tags, so the commit is the pin |
| `numpy` / `pydantic` / `pyyaml` | `1.26.4` / `2.9.2` / `6.0.2` | `wam.robot` is torch-free by contract |

Two implementation details worth remembering:

- **`unitree_sdk2py` is installed editable (`pip install -e`) on purpose.** Its `setup.py`
  declares no `package_data`, so a normal install silently drops
  `unitree_sdk2py/utils/lib/crc_aarch64.so` — the native CRC routine. Editable keeps the
  checkout, and therefore the `.so`, on disk. The `sdk_import` check asserts the native
  library actually loaded rather than falling back.
- **`--no-deps`**: the vendor `install_requires` pulls `opencv-python` (~200 MB on arm64)
  although the package contains zero `import cv2`. Skipping it keeps the image at 204 MB.

`docker build` emits two `FromPlatformFlagConstDisallowed` lint warnings because both `FROM`
lines hard-pin `linux/arm64`. That is deliberate — this image is arm64-only by contract, and
an accidental amd64 build would quietly load `crc_amd64.so` and invalidate the whole premise.

## The checks

Each line of output is exactly one `PASS` / `FAIL` / `SKIP`. A failed check blocks the rest,
which are reported as `SKIP` with the blocking reason — never dropped.

| # | Check | Establishes |
|---|-------|-------------|
| 1 | `sdk_import` | `unitree_sdk2py` imports; the native aarch64 CRC library loads (no silent Python fallback); `CRC().Crc()` returns an int |
| 2 | `dds_init_loopback` | `ChannelFactoryInitialize(0, "lo")` builds a CycloneDDS domain + participant |
| 3 | `fake_peer_process` | A **second process** with its own DomainParticipant comes up and publishes `rt/lowstate` |
| 4 | `transport_connect` | `DdsG1Transport(G1Config(...)).open()` succeeds — the `RuntimeError("requires unitree_sdk2py")` path is *not* hit — and the first `LowState_` arrives |
| 5 | `lowstate_roundtrip` | `q`[29]/`dq`[29] equal what the peer published; IMU quaternion/gyro/accel map correctly; `tick_ns` advances in whole milliseconds |
| 6 | `lowcmd_roundtrip_crc` | `write_motor_cmd` produces a `LowCmd_` the peer receives with the exact 29-motor `q`/`dq`/`kp`/`kd` payload, `tau=0`, `mode=1` on 0..28 and `mode=0` on the unused slots 29..34, `mode_machine` echoed from the state, and a CRC the peer **recomputes and confirms** |
| 7 | `lowcmd_crc_detects_corruption` | A deliberately wrong CRC is rejected — check 6 is not vacuous |
| 8 | `emergency_damp` | The damping command is `kp=0`, `kd=damp_kd` on all 29 motors with `q=dq=tau=0` and a valid CRC |
| 9 | `dex3_gripper_roundtrip` | `write_gripper_cmd` delivers a 7-motor `HandCmd_` per hand on `rt/dex3/{left,right}/cmd`; the mirrored `HandState_` comes back through `read_low_state()["gripper"]` |
| 10 | `adapter_closed_loop` | The **whole `G1Adapter`** over DDS: `read_state()` returns the 15 canonical joints, `execute()` lands the commanded delta on exactly the mapped motor indices, unmapped legs hold, CRC valid |
| 11 | `stale_tick_degrades_validity` | Peer stops publishing → `tick_ns` freezes → the adapter clears every validity flag (the watchdog path, over real DDS) |

The peer runs as a **separate process**, not a thread, so checks 4-11 require two independent
participants to discover each other and exchange RTPS over loopback. Parent↔peer coordination
goes through a stdin/stdout JSON line protocol, never through DDS, so a DDS failure can never
be masked by the control channel.

Run it directly for options:

```bash
docker run --rm --init -v "$PWD:/wam:ro" wam-dds-conformance:latest \
  python3 /wam/docker/dds/conformance.py --interface lo --domain 0
```

## What this does **not** prove

Read this section before quoting "11 PASS" anywhere.

1. **It is not a physics simulation.** The peer echoes whatever it is told to publish. No
   contact, no dynamics, no gravity, no actuator model. A command that would make the robot
   fall over passes every check here. Physics belongs to the MuJoCo track.
2. **Self-consistency, not vendor conformance.** Both sides use the same `unitree_sdk2py` IDL
   classes and the same `CRC.Crc()`. That the vendor's Python struct layout matches the
   robot's C++ `LowCmd` byte-for-byte is asserted by Unitree, not verified here. A vendor-side
   layout change would be invisible to this test.
3. **Dex3 hand details are unverified.** The topic names `rt/dex3/{left,right}/{cmd,state}`,
   the RIS mode-byte packing (`id | status<<4 | timeout<<7`) and the finger gains come from
   vendor documentation. The gripper mapping itself (command all 7 finger joints to the same
   angle, report their mean) is an explicit **placeholder open/close proxy**, not a grasp
   policy.
4. **No vendor RPC services.** `MotionSwitcherClient` (release the built-in controller) and
   `LocoClient().Damp()` need a live service peer on the robot and are not exercised. This is
   why `emergency_damp()` is implemented as a *wire* damping command rather than a service
   call — a wire command still works when the vendor service is dead, which is exactly when an
   e-stop matters.
5. **`rt/arm_sdk` is implemented but untested.** `DdsG1Transport(cmd_topic=G1_ARM_SDK_TOPIC,
   arm_sdk_weight=1.0)` writes the blend weight into `motor_cmd[29].q`, but no peer here
   validates the vendor's blending semantics.
6. **No timing or load characterisation.** The peer publishes at 200 Hz; the robot publishes
   `LowState_` at 500 Hz and expects `LowCmd_` at up to 500 Hz. Latency, jitter, packet loss,
   MTU and sustained-rate behaviour on a real switch are all untested.
7. **`G1Config` limits and gains remain placeholders** (`kp=20`, `kd=0.5`, `q∈±90°`,
   `dq_max=2 rad/s`, `damp_kd=2.0`). They are shaped correctly and clipped correctly; whether
   they are *right* is OD-08.
8. **`mode_pr`/`mode_machine` semantics are pass-through.** The transport echoes the reported
   `mode_machine` and always sends `mode_pr = 0` (series). Whether that is correct for a given
   23-DoF/29-DoF machine is a hardware question.

### Skipped checks

The current run reports **0 SKIP**. Nothing in the intended scope was silently dropped. Three
things were deliberately *not attempted*, for the reasons above, and are tracked as bring-up
steps rather than as skipped checks: vendor RPC services (4), `rt/arm_sdk` blending (5), and
rate/latency characterisation (6).

## The multicast caveat (why this stays inside the container)

DDS discovery (SPDP) normally uses multicast. Inside the container CycloneDDS logs:

```
selected interface "lo" is not multicast-capable: disabling multicast
```

and falls back to unicast discovery on localhost. Both participants live in the same network
namespace, so this works and is exactly why the fake peer is co-located.

**Do not try to bridge this to the macOS host or to a real robot.** On Docker Desktop for Mac
every container runs inside a LinuxKit VM behind a NAT:

- multicast/RTPS SPDP is not forwarded across that VM boundary in either direction, so a
  container cannot discover a robot on the LAN (and vice versa);
- `--network host` does **not** help on macOS — it gives you the *VM's* network namespace, not
  the Mac's LAN interfaces;
- configuring CycloneDDS `<Discovery><Peers>` for unicast to the robot's IP still requires the
  packets to be routable to the robot's subnet, which the VM NAT does not provide.

The supported paths for talking to a real G1 are: run on the robot's onboard Jetson, run in a
container **on a Linux host** with `--network host`, or run natively on a Linux workstation on
the robot's network. None of those change the code — only `G1Config.network_interface`.

## Remaining steps for real-hardware bring-up

In order. Steps 1-3 are prerequisites for *any* motion.

1. **Network.** Put a Linux host (or the onboard Jetson) on the robot's network
   (`192.168.123.x`). Set `G1Config.network_interface` to the NIC facing the robot; keep
   `domain_id=0`. Confirm `rt/lowstate` arrives at ~500 Hz before sending anything.
2. **Release the built-in controller.** The vendor motion service owns the motors until
   `MotionSwitcherClient().ReleaseMode()` is called (see `example/g1/low_level/`). This is
   **not** part of `DdsG1Transport` — decide deliberately whether it belongs in `open()` or in
   an operator-run script, because it is the moment the robot stops holding itself up.
3. **Verify the e-stop first.** With the robot suspended/supported, call `estop()` and confirm
   the joints actually go limp-damped. Add `LocoClient().Damp()` as an escalation on top of the
   wire damping once the service is reachable. Nothing autonomous runs before this passes.
4. **Start on `rt/arm_sdk`, not `rt/lowcmd`.** `DdsG1Transport(cmd_topic=G1_ARM_SDK_TOPIC,
   arm_sdk_weight=1.0)` keeps the legs with the vendor controller and blends in arm targets.
   `rt/lowcmd` takes over all 29 motors at once and is an E3-grade decision, not a first-contact
   one.
5. **Confirm the machine variant.** Check the reported `mode_machine`, whether the waist
   roll/pitch joints exist (23-DoF locks them) and whether `mode_pr` should be `0` (series) or
   `1` (parallel A/B) for the ankles on your unit.
6. **Replace the placeholder limits and gains** in `G1Config` from the vendor datasheet
   (OD-08), and version them under `configs/`. Re-confirm `damp_kd`.
7. **Fix the gripper mapping.** Verify the Dex3 topic names and the RIS mode byte on hardware,
   determine the real per-joint vendor range, set `gripper_vendor_min/max`, and replace the
   mean-of-7 placeholder with a real grasp mapping.
8. **Characterise timing.** `G1Adapter.execute()` paces commands on the wall clock; the
   `dq_max * dt` clip is only a velocity limit if commands really arrive `dt_s` apart. Measure
   jitter at the chosen `control_dt_s` under load.
9. **Then, and only then, E2 → E3.** Sim, replay, and finally the real robot per the
   evaluation ladder.

## Maintenance

```bash
NO_CACHE=1 docker/dds/run.sh            # clean rebuild (~4 min: CycloneDDS + cyclonedds-python)
docker image rm wam-dds-conformance:latest
docker builder prune                    # this image's build cache; host disk is tight
```

Adding a check: write a `check_*() -> str` closure in `main()` (raise to fail, return the
detail string to pass) and register it with `report.run(...)`. Anything the fake robot must do
gets an `op` in `FakeG1Peer.handle`.
