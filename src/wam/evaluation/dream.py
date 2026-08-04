"""Dream sampling: run the video branch forward and measure whether it moves (T-35).

The video branch has never been *watched* through WAM's own conventions. Two routes existed,
and neither is this one:

* ``JointWorldActionModel.predict`` runs one backbone pass at t=1 and **throws the video
  velocity away** (``joint.py``, "no iterative denoising, no video sampled at test time"). The
  video branch's whole job at inference is to have shaped the features during training.
* the "generate future" tab (``scripts/hf_job_wan_probe.py:generate_future``) samples the
  *exported LoRA* inside a stock ``WanImageToVideoPipeline``. That is a diffusers pipeline, so
  it is image-conditioned, uses Wan's own scheduler and CFG, and — the part that matters —
  **cannot supply proprioception**. The DiT was trained with the state token concatenated onto
  the text context (``wan_i2v.py:605``); the pipeline has no state input, so every clip
  measured on 2026-07-30 (``docs/hf_jobs.md``, "the fine-tune is geometry-bound") was generated
  with that input missing.

This module is the third route: integrate ``FlowBackbone.forward_flow`` in WAM's convention,
with the text **and state** context the model was actually trained against, at the geometry it
was trained at. It is the only path on which a dream is conditioned on the robot.

**What that fixes, and what it does not.** It removes one train/inference mismatch — the same
*class* of defect T-29 found in the frame context, and the one confound the archived generate-tab
numbers could not control for. It does not make a dream evidence that the policy works, and it
does not make a dream training data: see :func:`motion_ratio`'s docstring for why the ratio,
not the raw number, is the readable quantity.

Sampling direction, which is the one thing that silently produces garbage if wrong: WAM's
convention lives in :func:`~wam.training.losses.make_flow_targets` — ``x_t = (1-t) x0 + t x1``
with ``x0`` the NOISE and ``x1`` clean, so **t=0 is noise, t=1 is clean**, the inverse of the
diffusers/SD3 convention. Sampling starts at ``z ~ N(0, I)`` and steps FORWARD with t ascending.
The t-grid stops one ``dt`` short of 1.0 because ``JointTrainer.compute_losses`` draws t from
``torch.rand`` (support [0, 1)), so t=1.0 is a level the flow was never trained at — same
reasoning, same grid, as :meth:`~wam.training.joint.JointWorldActionModel.sample_action_chunk`.

**One property this sampler has that the action sampler does not.** T-30's action sampler reuses
ONE set of features computed at t=1 for every t_k, so at small t_k the velocity head is asked
about a (near-noise latent, clean-video features) pair it never saw in training — a documented
confound on every T-30 arm. Here the backbone is re-evaluated at each t_k, on the latent noised
to exactly that t_k, which is precisely the pairing ``co_denoise`` trained. The video sampler is
faithful to the training conditioning; it costs n backbone passes to be, and that is why the
action branch cannot afford the same fidelity at 2 Hz.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

DREAM_VERSION = "dream/1"

# Motion is reported in 0-255 pixel units to stay comparable with the ad-hoc numbers already
# recorded in docs/hf_jobs.md ("the fine-tune is geometry-bound", 2026-07-30): base 29.48 and
# LoRA 0.73 at 9x128x160, base 2.93 at 49x480x640. Those were scratch output from the Space and
# no committed code reproduced them until this module — the same gap PR-04 closed for the
# corpus screen, and the reason that table's numbers are two implementations short of canonical.
MOTION_SCALE = 255.0

#: A pair of consecutive frames counts as STILL below this mean absolute difference (0-255).
#: 1.0 is one quantization step of an 8-bit channel: below it, nothing survives being written
#: to a PNG. Not tuned — picked as the smallest defensible unit and fixed here.
STATIC_THRESHOLD = 1.0

#: Above this peak a float array is 0-255 already and normalizing it again would multiply by
#: 255 a second time. Set well above what an untrained model overshoots [0, 1] by (single
#: digits) and well below 255, so the two cases cannot be confused in either direction.
_MAX_UNIT_PEAK = 16.0

#: Pre-registered floor for :func:`motion_ratio`. A dream that moves less than half as much as
#: the VAE round-trip of the very clip it was conditioned on is not a candidate for anything
#: downstream. Fixed BEFORE the first Wan run (T-35), so the number cannot be chosen to fit.
MOTION_FLOOR_RATIO = 0.5


class ClipMetrics(BaseModel):
    """Motion statistics of one arm's clips. All pixel quantities in 0-255."""

    model_config = ConfigDict(frozen=True)

    arm: str
    clips: int = Field(ge=0)
    frames: int = Field(ge=0)
    height: int = Field(ge=0)
    width: int = Field(ge=0)
    motion: float = Field(description="mean |frame[t+1] - frame[t]| over all pixels and clips")
    motion_per_clip: tuple[float, ...] = ()
    pixel_std: float = Field(description="std over all pixels — a collapsed clip loses contrast")
    static_fraction: float = Field(
        ge=0.0, le=1.0, description=f"share of frame pairs moving < {STATIC_THRESHOLD} (0-255)"
    )


class DreamReport(BaseModel):
    """One dream measurement: every arm, the ratios, and the two verdicts they support."""

    model_config = ConfigDict(frozen=True)

    version: str = DREAM_VERSION
    arms: dict[str, ClipMetrics]
    motion_ratio: dict[str, float] = Field(
        default_factory=dict, description="arm motion / reference (VAE round-trip) motion"
    )
    reference_arm: str = "recon"
    pair_distance: dict[str, float] = Field(
        default_factory=dict, description="mean |arm_a - arm_b| (0-255) at matched seed"
    )
    verdicts: dict[str, str] = Field(default_factory=dict)
    info: dict[str, Any] = Field(default_factory=dict)


# ---- frame normalization ---------------------------------------------------------------


def as_frames_255(frames: Any) -> np.ndarray:
    """Anything frame-shaped -> float32 ``[B, F, H, W, 3]`` in 0-255.

    Both conventions in this codebase arrive here: recorded episodes are uint8 (``EpisodeReader``
    /``EpisodeDataset``) and ``FlowBackbone.decode_video`` returns float in [0, 1] (WAM's pixel
    convention, not Wan's [-1, 1]). Rescaling by peak value would be a guess, so the dtype
    decides: integers are already 0-255, floats are [0, 1] and get multiplied. A float array that
    is secretly 0-255 would come back 255x too large and fail the range check below rather than
    quietly halving every motion number downstream.
    """
    import torch

    if isinstance(frames, torch.Tensor):
        array = frames.detach().to("cpu").numpy()
    else:
        array = np.asarray(frames)
    if array.ndim == 4:
        array = array[None, ...]
    if array.ndim != 5 or array.shape[-1] != 3:
        raise ValueError(f"frames must be [B, F, H, W, 3] or [F, H, W, 3], got {array.shape}")
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.float32)
    out = array.astype(np.float32)
    peak = float(np.abs(out).max()) if out.size else 0.0
    if peak > _MAX_UNIT_PEAK:
        raise ValueError(
            f"float frames must be WAM's [0, 1] pixel convention, got peak {peak:.3f} — an "
            "already-scaled 0-255 float array would be multiplied a second time here"
        )
    # Clipped, not rejected, below the guard: a real decoded clip is already in range
    # (``decode_video`` clamps), but an UNTRAINED model's latents decode to whatever they
    # decode to, and a diagnostic that cannot be pointed at an untrained model is a diagnostic
    # that can only be run once the answer no longer matters.
    return np.clip(out, 0.0, 1.0) * MOTION_SCALE


# ---- metrics ---------------------------------------------------------------------------


def motion_energy(frames: Any) -> float:
    """Mean absolute frame-to-frame difference in 0-255 — the archived ``motion`` definition."""
    array = as_frames_255(frames)
    if array.shape[1] < 2:
        return 0.0
    return float(np.abs(np.diff(array, axis=1)).mean())


def _static_fraction_255(array: np.ndarray, threshold: float) -> float:
    """``static_fraction`` on an array :func:`as_frames_255` has already normalized.

    Split out because normalizing twice is not idempotent by design — the second pass sees a
    0-255 float and the range guard fires, which is the behaviour that makes an accidental
    double-scale loud instead of a silent 255x.
    """
    if array.shape[1] < 2:
        return 1.0
    per_pair = np.abs(np.diff(array, axis=1)).mean(axis=(2, 3, 4))  # [B, F-1]
    return float((per_pair < threshold).mean())


def static_fraction(frames: Any, *, threshold: float = STATIC_THRESHOLD) -> float:
    """Share of consecutive frame pairs whose mean absolute difference is below ``threshold``.

    Motion is a mean and a mean hides its shape: a clip that jumps once and then freezes and a
    clip that drifts uniformly can report the same number. This separates them, which is the
    difference between "predicts a short motion then stops" and "predicts nothing".
    """
    return _static_fraction_255(as_frames_255(frames), threshold)


def measure_clips(frames: Any, *, arm: str) -> ClipMetrics:
    """All statistics of one arm in a single pass over its clips."""
    array = as_frames_255(frames)
    batch, num_frames, height, width, _ = array.shape
    if num_frames >= 2:
        per_clip = np.abs(np.diff(array, axis=1)).mean(axis=(1, 2, 3, 4))
    else:
        per_clip = np.zeros((batch,), dtype=np.float32)
    return ClipMetrics(
        arm=arm,
        clips=batch,
        frames=num_frames,
        height=height,
        width=width,
        motion=float(per_clip.mean()) if batch else 0.0,
        motion_per_clip=tuple(round(float(v), 4) for v in per_clip),
        pixel_std=float(array.std()) if array.size else 0.0,
        static_fraction=_static_fraction_255(array, STATIC_THRESHOLD),
    )


def pair_distance(frames_a: Any, frames_b: Any) -> float:
    """Mean ``|a - b|`` in 0-255 between two arms' clips, which must be the same shape.

    The only quantity here that can answer "did the fine-tune move the video branch at all",
    and only against its own null: compare ``d(lora, base)`` with ``d(base_seed0, base_seed1)``,
    the same model's spread across sampling noise. A fine-tune whose dreams differ from the base
    by less than the base differs from itself has not been shown to change the prior — the
    comparison being unavailable is exactly why the archived table had to argue from what the
    clips *looked like*.
    """
    a = as_frames_255(frames_a)
    b = as_frames_255(frames_b)
    if a.shape != b.shape:
        raise ValueError(f"arm shape mismatch: {a.shape} vs {b.shape}")
    return float(np.abs(a - b).mean())


def motion_ratio(arm_motion: float, reference_motion: float) -> float:
    """``arm / reference`` — the readable quantity, where the raw motion number is not.

    Absolute motion is not comparable across geometries: the archived table has the *base* prior
    at 29.48 (9x128x160, "psychedelic colour noise") and at 2.93 (49x480x640, a coherent grasp),
    a 10x spread that is about resolution and clip length, not about imagining more. Dividing by
    the VAE round-trip of the same recorded clips cancels geometry, codec and camera, and leaves
    the one question: does the dream move like the data it was trained on.
    """
    if reference_motion <= 0.0:
        raise ValueError(f"reference motion must be > 0 to divide by, got {reference_motion}")
    return arm_motion / reference_motion


# ---- sampling --------------------------------------------------------------------------


def vae_roundtrip(model: Any, frames: Any) -> Any:
    """``decode_video(encode_video(frames))`` — the reference arm, and the ceiling for any dream.

    A sampled clip is decoded from latents, so it inherits everything the VAE loses: resizing to
    the backbone's legal grid (GR00T's 120x160 -> 128x160), the temporal stride, and the codec.
    Scoring a dream against RAW recorded frames would charge it for all of that. This arm is the
    same pipeline with the flow removed, so a ratio against it isolates the flow.
    """
    import torch

    with torch.no_grad():
        return model.backbone.decode_video(model.backbone.encode_video(frames))


def sample_video(
    model: Any,
    batch: Mapping[str, Any],
    *,
    steps: int = 32,
    seed: int = 0,
    anchor_latent_frames: int = 0,
    t0: float = 0.0,
) -> Any:
    """Integrate the video branch -> decoded frames ``[B, F, H, W, 3]`` in [0, 1].

    ``batch`` is an ``EpisodeDataset`` batch (``frames``, ``q``, ``dq``, ``imu``, ``gripper``,
    optional ``validity``/``instruction``) — the same mapping :meth:`co_denoise` consumes, so the
    conditioning is built by the same code paths training used and cannot drift from them.

    ``anchor_latent_frames`` pins the leading latent frames to the observed clip by replacement:
    after each Euler step those positions are overwritten with the observation noised to the
    current t. **This is an intervention, not a mode the model was trained in** — training noised
    the whole window and denoised the whole window, so a free sample (the default, ``0``) is the
    faithful arm and an anchored one is the arm that makes the clip a *future prediction* rather
    than a text+state-conditioned sample. Wan's causal VAE maps latent frame 0 to pixel frame 0
    and then groups of ``temporal_stride``, so 1 anchor ~ the start frame; the correspondence is
    approximate and anchored frames must be excluded before scoring motion, or the arm imports
    the real clip's motion and reports it as imagination.

    ``t0`` > 0 warm-starts from the observed latent and is an ORACLE here — it injects the clean
    future the sampler is supposed to produce. It exists for one purpose, measuring how much of a
    negative is the sampler rather than the branch, and any arm using it must be labelled.
    """
    import torch

    from wam.training._utils import encode_instructions
    from wam.training.losses import make_flow_targets

    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if not 0.0 <= t0 < 1.0:
        raise ValueError(f"t0 must be in [0, 1), got {t0}")

    with torch.no_grad():
        clean = model.backbone.encode_video(batch["frames"])
        batch_size = int(clean.shape[0])
        # WHICH axis holds frames is the backbone's business, not this function's (FR-09): Wan
        # latents are [B, z, F', h, w] and the tiny backbone's "latents" are pixels,
        # [B, F, H, W, 3]. Hard-coding axis 2 here would silently pin an image ROW instead of a
        # frame on any backbone but Wan — a bug with the right shapes and no error.
        frame_axis = getattr(model.backbone, "latent_frame_axis", None)
        if anchor_latent_frames and frame_axis is None:
            raise TypeError(
                f"{type(model.backbone).__name__} does not declare latent_frame_axis, so which "
                "axis to anchor is unknown — free sampling (anchor_latent_frames=0) works on any "
                "backbone"
            )
        latent_frames = int(clean.shape[frame_axis]) if frame_axis is not None else 0
        if not 0 <= anchor_latent_frames <= latent_frames:
            raise ValueError(
                f"anchor_latent_frames must be in [0, {latent_frames}], got {anchor_latent_frames}"
            )
        pin = (slice(None),) * int(frame_axis or 0) + (slice(0, anchor_latent_frames),)

        # CPU generator then .to(device): the same pattern co_denoise and sample_action_chunk
        # use, so a seed means the same draw regardless of which device the run lands on.
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype).to(clean.device)

        state_batch = {k: batch[k] for k in ("q", "dq", "imu", "gripper") if k in batch}
        if batch.get("validity") is not None:
            state_batch["validity"] = batch["validity"]
        state_emb = model.state_encoder(state_batch)
        text_ctx = encode_instructions(model.backbone, batch.get("instruction", ""), batch_size)
        # The input the diffusers "generate future" route structurally cannot supply.
        state_ctx = model.backbone.condition_state(state_emb)

        z = noise if t0 == 0.0 else make_flow_targets(noise, clean, t0)[0]
        dt = (1.0 - t0) / steps
        for k in range(steps):
            t = t0 + k * dt
            t_vec = torch.full((batch_size,), t, dtype=torch.float32, device=clean.device)
            if anchor_latent_frames:
                pinned, _ = make_flow_targets(noise, clean, t_vec)
                z[pin] = pinned[pin]
            velocity, _ = model.backbone.forward_flow(z, t_vec, text_ctx, state_ctx)
            z = z + velocity.to(z.dtype) * dt
        if anchor_latent_frames:
            z[pin] = clean[pin]
        return model.backbone.decode_video(z)


# ---- report ----------------------------------------------------------------------------


def build_report(
    arms: Mapping[str, Any],
    *,
    reference_arm: str = "recon",
    pairs: Mapping[str, tuple[str, str]] | None = None,
    info: Mapping[str, Any] | None = None,
) -> DreamReport:
    """Measure every arm, ratio them against ``reference_arm``, and record the two verdicts.

    ``pairs`` names the arm-vs-arm distances to compute, e.g.
    ``{"lora_vs_base": ("lora", "base"), "base_seed_null": ("base", "base_seed1")}``. The second
    is not optional decoration: without a same-model null, a nonzero ``lora_vs_base`` says only
    that sampling is stochastic.
    """
    measured = {name: measure_clips(frames, arm=name) for name, frames in arms.items()}
    if reference_arm not in measured:
        raise KeyError(
            f"reference arm {reference_arm!r} is not among {sorted(measured)} — a motion number "
            "without the VAE round-trip to divide by is not comparable across geometries"
        )
    reference = measured[reference_arm].motion
    ratios = {
        name: round(motion_ratio(metrics.motion, reference), 4)
        for name, metrics in measured.items()
    }
    distances = {
        label: round(pair_distance(arms[a], arms[b]), 4) for label, (a, b) in (pairs or {}).items()
    }

    verdicts: dict[str, str] = {}
    for name in measured:
        if name == reference_arm:
            continue
        verdicts[f"{name}.moves"] = "MOVES" if ratios[name] >= MOTION_FLOOR_RATIO else "STATIC"
    if "lora_vs_base" in distances and "base_seed_null" in distances:
        null = distances["base_seed_null"]
        verdicts["fine_tune_changed_the_prior"] = (
            "CHANGED" if distances["lora_vs_base"] > null else "INDISTINGUISHABLE"
        )
    return DreamReport(
        arms=measured,
        motion_ratio=ratios,
        reference_arm=reference_arm,
        pair_distance=distances,
        verdicts=verdicts,
        info=dict(info or {}),
    )
