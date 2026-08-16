# `wam.backbones` — video backbones behind one interface

**TL;DR** — Four interchangeable implementations of `BackboneAdapter` (FR-09/AC-05). Swapping
one for another must not change the data schema or the robot API. Nothing here is hard-wired
anywhere else — construct via `get_backbone(name, **cfg)`.

## Files

| File | Backbone | State |
|------|----------|-------|
| `registry.py` | `get_backbone` / `available_backbones` | — |
| `tiny.py` | `tiny` — self-contained patchify transformer | **fully functional** |
| `wan_i2v.py` | `wan2.1-i2v` — diffusers-backed open fallback | functional, needs weights |
| `flux3.py` | `flux3-dev` — preferred backbone | stub, blocked on OD-06 |
| `cosmos3_edge.py` | `cosmos3-edge` — edge sub-project target | skeleton, weights not staged |

Factories import their module lazily, so listing or constructing a torch-free skeleton never
pulls in torch. Construction never downloads weights.

## The interface

```
condition_video(video)   past frames -> backbone-native video context
condition_text(text)     instruction -> text context
condition_state(emb)     StateEncoder output -> state context
features(v, t, s)        -> intermediate activations [B, S, feature_dim]
```

The action decoder reads **intermediate** features, never final pixels.

## `tiny.py`

A small patchify transformer: same interface, tiny dims, CPU-fast, deterministic. Used for unit
tests and the D1 overfit gate.

Self-contained by design — **no external tokenizer, no downloads**. Text conditioning is a
deterministic crc32 hash embedding; the "VAE" is the identity, so tiny operates directly in
pixel space and "video latents" means float frames `[B, F, H, W, 3]`. No dropout and no RNG in
forward; construction under `torch.manual_seed` reproduces bit-for-bit on CPU.

**Token layout contract:** sequences are `[video | text | state]` with video tokens **first**
(`config.num_video_tokens` of them). `predict_video_latents` depends on that ordering.

Beyond the protocol it also offers `predict_video_latents` (video-branch loss),
`forward_flow(latents, t, text, state) -> (velocity, features)` for joint flow-matching
training (T-16, PRD §10.3), and the `FlowBackbone` extras `encode_video` / `decode_video` /
`num_video_tokens` / `frozen_part_names`.

`TinyBackboneConfig` sets `extra="forbid"` on purpose: the backbone config is a discriminated
union, and an untagged Wan-shaped dict would otherwise validate into an all-defaults 64-dim
tiny config and silently train the wrong model. Fail loudly instead.

## `wan_i2v.py`

Works with any Wan variant shipping `WanTransformer3DModel` + `AutoencoderKLWan` + umT5 — e.g.
`Wan2.2-TI2V-5B` (30 blocks, 3072 dim) or `Wan2.1-I2V-14B-480P` (40 blocks, 5120 dim). **All
geometry is read from the loaded configs**; the `WAN_*` constants are only 14B defaults used
for reporting before `load()`.

The module imports torch-free — torch, diffusers and transformers are imported inside `load()`,
and nothing downloads implicitly (`local_files_only=True` unless `allow_download`).

Pipeline (DreamZero recipe):

- `condition_video` → Wan-VAE latents, normalized with `latents_mean/std`; posterior **mode**,
  not a sample, because a feature extractor must be deterministic. I2V checkpoints with a CLIP
  tower also get the image embedding of the *last* observed frame (`hidden_states[-2]`).
- `condition_text` → frozen umT5, padding positions zeroed.
- `condition_state` → **one extra token in text-context space**, appended to the text context;
  the DiT's own text embedder maps it into the residual stream. This projection is the adapter's
  **only trainable parameter block** (built lazily, exposed as `state_projection`).
- `features` → forward hooks on the residual-stream output of `feature_blocks` (default:
  mid/late depth, `num_layers//2` and `3*num_layers//4`), averaged. `features_by_block` returns
  them unaveraged for readout ablations.

Everything is frozen and in eval mode on attach; LoRA fine-tuning unfreezes explicitly in the
training code. `offload(...)` moves named components to CPU for peak-VRAM relief — the umT5
tower is only needed once per rollout for a fixed instruction. `device_map` streams shards
straight to the target device instead of materializing the checkpoint in host RAM first
(required when host RAM < checkpoint; also just faster — 7.4 s for the 5B).

Hook outputs are shape-checked against `feature_dim`, so a block whose output is not the
residual stream fails loudly rather than feeding garbage downstream.

## `flux3.py`

Protocol-conformant placeholder. `name` and `feature_dim` work; everything else raises
`NotImplementedError("FLUX 3 access pending — OD-06")`. It stays in-tree so the registry and
the backbone-swap tests exercise a third name today. `feature_dim` defaults to 4096 — a
placeholder, constructor-overridable, so downstream shape plumbing can already be tested. The
planned integration surface is documented in the module docstring.

## `cosmos3_edge.py`

`nvidia/Cosmos3-Edge-Policy-DROID` (default) / `nvidia/Cosmos3-Edge` (base) — the 4B checkpoint
`subprojects/edge-wam/` is built around. Torch- and diffusers-free at import; `load()` does the
heavy lifting and raises an actionable `RuntimeError` until a ~9.2 GB snapshot is staged.

`Cosmos3EdgeConfig` is a **stdlib frozen dataclass, deliberately not in the `BackboneConfig`
union** — post-training a Cosmos 3 embodiment row is unpriced (E-02), so no training config may
select this backbone yet. Verified constants: hidden size 2048 over 28 layers, global
`action_dim` 64 shared by all embodiments, 32 embodiment rows, BF16 only.

Three methods refuse instead of guessing, each naming what would settle it:

- `condition_state()` — **always** raises. Neither `Cosmos3OmniPipeline.__call__` nor
  `Cosmos3OmniTransformer.forward` has a proprioception input, even though NVIDIA's DROID
  recipe trains with `use_state=true`. A fabricated state slot would run cleanly on nothing.
- `features()` — raises "weights missing" unloaded, `NotImplementedError` loaded: the readout
  *width* is known (2048), the readout *depth* is not, and Cosmos 3 publishes no recipe.
- `require_raw_action_dim()` — raises until `raw_action_dim` is set explicitly. diffusers 0.39.0
  says 10 for `droid_lerobot`; the model card says 8. `Cosmos3EdgeConfig.raw_action_dim` has no
  default on purpose.

`condition_text()` models E-01 honestly: text is **structurally required** (`prompt` is a
required positional, `text_tokenizer` a required component, and action tokens cross-attend to
text in every layer), so `""` is a *degraded* case rather than a null path and must be opted
into with `allow_empty_instruction=True`. For a single-task MVP, pass a constant correct
sentence. `condition_video()` selects the current frame (the last of the history) — Cosmos 3
policy mode conditions on one image — and is weights-free.
