---
id: T-16a
aliases:
- T-16a
title: "Make T-16 runnable on the real backbone"
slug: make-t-16-runnable-on-the-real-backbone
status: done
priority: 3
owner: ''
projects: []
customers: []
tags:
- m3
- interfaces
- backbone
sprint: ''
depends_on: []
due_date: ''
created: 2026-07-29
updated: 2026-07-29
---

# Make T-16 runnable on the real backbone

## Description

Make T-16 runnable on the real backbone — *done 2026-07-27, 583 tests green, all CPU-testable
without Wan weights*: (a) new `FlowBackbone` protocol
(`encode_video`/`decode_video`/`forward_flow`/`num_video_tokens`/`frozen_part_names`,
`INTERFACES_VERSION` 0.2.0); `JointWorldActionModel(config, backbone=…)` depends only on it, and the
backbone config is a `kind`-discriminated union so an untagged Wan block raises instead of silently
validating as tiny; (b) `WanI2VAdapter.add_lora()` (peft on the DiT blocks, targets
`to_q/to_k/to_v/to_out.0/net.0.proj/net.2`) + `WanFlowBackbone` (`src/wam/backbones/wan_flow.py`),
which keeps the 5B DiT/VAE/umT5 **out of the module tree** and aliases only the LoRA params —
adapter-only checkpoints are structural, not a flag; (c) `scripts/train_t16_lora.py` with
`SIGUSR1`→checkpoint→exit 0, `--resume latest`, `DONE` sentinel, and a bitwise resume test. Wan's
downward timestep schedule and velocity sign live only in `wan_i2v.forward_flow`

---

Migrated from `TASKS.md` (milestone M3) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
