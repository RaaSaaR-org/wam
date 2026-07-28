# `wam.encoders` — non-visual inputs → embeddings

**TL;DR** — Two small trainable MLPs that turn non-image data into tensors the backbone can
condition on. `StateMLP` runs at training *and* inference; `ActionChunkEncoder` is
**training-only** and never touches the runtime path.

## Files

| File | Contains |
|------|----------|
| `state_mlp.py` | `StateMLP` — canonical robot state → embedding (FR-02, PRD 9.4) |
| `action_encoder.py` | `ActionChunkEncoder` — demonstrated chunk → per-step latents (FR-03, PRD 9.5) |

Frozen text/VAE encoders are not implemented here — those live inside the backbone adapters.

## `StateMLP`

Input is `concat(q, dq, imu, gripper)`; `IMU_DIM = 10` (quaternion wxyz 4 + angular velocity 3
+ linear acceleration 3). A 2–4 layer MLP (PRD 9.4 range, enforced by the config) maps it to a
fixed embedding.

```
encode(state)  RobotState -> [E]                          inference, StateEncoder protocol
forward(batch) {q [B,N], dq [B,N], imu [B,10], gripper [B,G],
                validity [B,4] optional} -> [B, E]        training
```

**Missing sensors are a first-class case.** Each field group owns a learned "missing"
embedding, substituted via `torch.where` wherever the validity mask flags that group invalid.
`where` is the point: it never multiplies by the invalid values, so NaN/Inf inside a masked-out
group reaches neither the output nor the backward pass, and gradients still flow — to the real
input where valid, to the missing embedding where not. In `encode`, invalid groups are never
even read from the state, so garbage or `None` there is tolerated.

Validity group order is fixed: `q, dq, imu, gripper`. Init is deterministic under
`torch.manual_seed`.

## `ActionChunkEncoder`

Per step, targets and gripper command are concatenated, run through a small MLP, and a learned
positional embedding (by step index) is added so the latent sequence is order-aware.

```
encode(chunk)             ActionChunk -> [T, L]                ActionEncoder protocol
forward(targets, gripper) [B,T,D], [B,T,G] -> [B, T, L]        training
```

`max_steps` bounds the positional table; longer chunks are rejected rather than truncated. The
canonical chunk carries one scalar gripper command per step (`[T]`), broadcast across
`gripper_dims` before encoding.

This module exists so demonstrated actions can be modeled jointly with video latents during
training. At runtime the model *produces* actions — it never encodes them.
