---
id: T-25a
aliases:
- T-25a
title: "Docker DDS conformance track on a real CycloneDDS bus"
slug: docker-dds-conformance-track-on-a-real-cyclonedds-bus
status: done
priority: 3
owner: ''
projects: []
customers: []
tags:
- m4
- hardware
sprint: ''
depends_on: []
due_date: ''
created: 2026-07-29
updated: 2026-07-29
---

# Docker DDS conformance track on a real CycloneDDS bus

## Description

Docker DDS conformance track: run the real `DdsG1Transport` against a fake G1 on a real CycloneDDS
bus in a `linux/arm64` container (same arch as the EDU4's Jetson Orin), which turns the last
hardware-stubbed file into implemented code —
`docker/dds/{Dockerfile,conformance.py,run.sh,README.md}`, `docker/dds/run.sh` — *✅ ran 2026-07-27:
**11 PASS / 0 FAIL / 0 SKIP**, reproduced 4× (SDK import incl. native aarch64 CRC, DDS init,
separate-process peer, connect, LowState round-trip, LowCmd round-trip + CRC, CRC-corruption
negative check, `emergency_damp`, Dex3 gripper round-trip, full `G1Adapter` closed loop, stale-tick
validity). `DdsG1Transport`'s five methods are implemented; on the host without `unitree_sdk2py`
they still raise `RuntimeError` and `tests/test_g1.py` is untouched.* **Wire layer only, not vendor
conformance:** both sides use the same `unitree_sdk2py` IDL + CRC, so a vendor-side layout change is
invisible; no physics, no vendor RPC services (`MotionSwitcherClient`, `LocoClient().Damp()` — hence
`emergency_damp()` is a *wire* command), `rt/arm_sdk` implemented but untested, no timing/load
characterisation, Dex3 topic names + RIS mode byte from documentation only, `G1Config` limits/gains
still placeholders (OD-08). Full list + 9 ordered bring-up steps: `docker/dds/README.md`

---

Migrated from `TASKS.md` (milestone M4) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
