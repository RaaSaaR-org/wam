# `wam.decoders` — backbone features → action chunk

**TL;DR** — One lightweight head (`ActionHead`) that reads the backbone's intermediate features
and emits a canonical `ActionChunk` (FR-04, PRD 9.6). It bounds its own output softly; **hard**
limits are the safety layer's job, downstream.

## `action_head.py`

An MLP trunk plus two bounded heads:

```
forward(features)  [B, F] -> {"targets": [B, T, D] via tanh   -> (-1, 1),
                              "gripper": [B, T, G] via sigmoid -> (0, 1)}
decode(features)   [F] or [*, F] -> ActionChunk               ActionDecoder protocol
```

`decode` accepts extra leading dimensions (e.g. backbone tokens) and **mean-pools** them, then
reduces the per-step gripper across `gripper_dims` to the single scalar the canonical chunk
carries. `mode` and `dt_s` come from the config.

## The unit contract

Targets are tanh-bounded to `(-1, 1)` and interpreted **directly as physical canonical units** —
per-step rad or m deltas. The MVP pipeline is identity-normalized end to end, so there is no
denormalization step anywhere. The practical consequence: **training data must keep per-step
`|targets| < 1`**, otherwise the head can never reach it. `EpisodeDataset` enforces this rather
than letting it fail silently.

`num_steps` is config-driven within the MVP guidance of 8–32 (PRD 9.10). Init is deterministic
under `torch.manual_seed`.

Whatever comes out of here still has to pass `wam.safety.SafetyLayer` before it may reach any
robot adapter (FR-07).
