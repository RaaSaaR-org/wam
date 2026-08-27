#!/usr/bin/env python
"""The PROPAGATION arm of PR-08 §4's segmenter comparison — one mask, seeded on frame 0, tracked.

WHY THIS FILE EXISTS AT ALL
---------------------------
``scripts/estimators/apple_sam2.py``'s **third** gate-qualification blocker names its own discharge
condition and this module is one half of it:

    "upstream drives SAM2VideoPredictor.init_state(video_path=...) and PROPAGATES one mask across
    the clip, while this adapter re-detects and re-segments every frame independently, because
    segment(rgb) is the contract both harnesses call. […] Discharged by: measuring the same Isaac
    capture BOTH ways — this adapter per frame, and the video predictor propagating from frame 0 —
    and recording the two p95s."

``apple_sam2`` is not touched, and cannot be: ``segment(rgb)`` is a per-frame contract and a video
predictor is a clip-level object. So the second arm lives here, drives the SAME GroundingDINO
detector at the SAME operating point on frame 0, seeds SAM 2 with the box that detector selected,
and then propagates — which is Cosmos-Transfer2.5's ``sam2_model.py`` topology rather than ours.

**IT DISCHARGES NOTHING.** ``GATE_QUALIFIED`` and ``GATE_QUALIFICATION_BLOCKERS`` are imported
read-only and are not written by anything here or by anything that calls this. Producing the
evidence a blocker names and accepting it are two different acts, and only one of them is a
session's to perform.

THE CONFOUND THIS MODULE IS BUILT AROUND
----------------------------------------
``SAM2VideoPredictor.init_state`` ingests **a directory of JPEG files** (``sam2.utils.misc.
load_video_frames`` accepts a JPEG folder or an MP4 and raises ``NotImplementedError`` on anything
else). Our captures are lossless ``rgb.npy``.

If the per-frame arm saw raw arrays and this arm saw a JPEG transcode of them, the difference
between the two p95s would be **the codec plus propagation**, in unknown proportions, reported as
propagation. That is not a weaker measurement — it is a void one, and it would not look wrong.

So nothing is ever encoded. :func:`frames_to_normalized_tensor` performs upstream's own
``_load_img_as_tensor`` arithmetic — ``Image.resize`` to the model grid, ``/255``, then the
ImageNet mean/std — directly on the ``uint8`` arrays the harness already holds, and
:func:`propagate` installs it in place of ``load_video_frames`` for the duration of exactly one
``init_state`` call. Everything else ``init_state`` does — every key of the inference state, the
whole tracking path, ``propagate_in_video`` — is the installed ``sam2`` package's code, unmodified.

``tests/test_apple_sam2_video_propagation.py`` asserts the ingest is **bitwise** upstream's ingest
of a lossless file of the same frames, and separately that a JPEG round trip of those frames would
NOT have been, which is the demonstration that the confound was real and was excluded rather than
assumed away.

WHAT IS THE SAME AS THE PER-FRAME ARM, AND WHAT IS DELIBERATELY NOT
-------------------------------------------------------------------
Same, so that the difference between the two p95s is attributable to propagation and to nothing
else: the SAM 2 checkpoint and its pinned revision, the GroundingDINO checkpoint and its pinned
revision, the ``apple.`` prompt, ``threshold=0.15`` / ``text_threshold=0.25``, the single
``(0.10, 0.10)`` retry, highest-score box selection, the offline hub, and the frames themselves.
The detector is reached through ``apple_sam2._best_box``, i.e. it is literally the same code —
including its counters, which is why the harness's ``estimator_stats`` block sees this arm's
detection too.

NOT the same, on purpose, and both are recorded in :data:`PROPAGATION_CONTRACT`:

1. **The detector runs on frame 0 only.** That is what "propagating from frame 0" means, and it is
   upstream's topology. A per-frame re-detection here would just be the other arm again.
2. **``apple_sam2``'s mask-validity colour filter is NOT applied.** Cosmos-Transfer2.5 has no such
   filter, and this arm exists to measure what the generator's segmenter does. Applying a per-frame
   validity refusal to a propagated mask would also erase precisely the failure the experiment is
   looking for: a mask that has drifted onto the table would be refused, counted as "no mask", and
   the run of low-IoU frames — blocker 3's failure mode (b) — would never appear. What the filter
   WOULD have refused is counted anyway, in :func:`stats`, so the choice is visible instead of
   silent — and it is counted over **both** of ``segment()``'s refusal branches, because that
   counter's only use is to be set beside the per-frame arm's ``n_frames_mask_refused`` and two
   counters mirroring different branch sets do not subtract.

WHAT IT REFUSES TO DO
---------------------
**It will not start a propagation it cannot seed.** If GroundingDINO finds no box on frame 0 the
run raises :class:`PropagationSeedNotFound` rather than returning 480 empty masks — which would
read out of the harness as a propagation arm that tracked nothing, i.e. as a fact about SAM 2.

**It will not rescale a float image**, and it will not ingest a clip that changes shape partway.
Same reasons as ``apple_sam2``: the first is two different pictures from one array, the second has
no resolution to compare a centroid at.

**It will not fetch anything.** The checkpoint is resolved through ``apple_sam2``'s own
``_offline_hub`` / ``_require_cached`` at ``apple_sam2.SAM2_MODEL_REVISION``, so this arm loads the
weights the artifact names or refuses.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator, Sequence

import numpy as np

from . import apple_sam2

#: What this arm is, for the artifact. The per-frame arm's counterpart is
#: ``apple_sam2.ESTIMATOR_NAME``; two arms sharing one name in one document is how one comes to be
#: read as the other.
PROPAGATION_NAME = "grounding-dino(frame0)+sam2-video-predictor"

#: Seeded on frame 0 and only frame 0. Not a flag: "propagating from frame 0" is the blocker's own
#: words and a sweep over seed frames would be a different experiment.
SEED_FRAME_INDEX = 0

#: The object id handed to the predictor. One object; there is one budget.
SEED_OBJECT_ID = 1

#: Kept off the GPU by default. The inference state holds the WHOLE clip preprocessed to
#: ``image_size`` x ``image_size`` float32 — 480 frames at 1024 is ~6 GB — and a propagation arm
#: that OOMs on the capture it was built for is not an arm.
OFFLOAD_VIDEO_TO_CPU = True
OFFLOAD_STATE_TO_CPU = True

PROPAGATION_VERSION = (
    f"{PROPAGATION_NAME};"
    f"seg={apple_sam2.SAM2_MODEL_CHECKPOINT}@{apple_sam2.SAM2_MODEL_REVISION};"
    f"det={apple_sam2.GROUNDING_DINO_MODEL_CHECKPOINT}@"
    f"{apple_sam2.GROUNDING_DINO_MODEL_REVISION};"
    f"prompt={apple_sam2.OBJECT_TEXT_PROMPT!r};"
    f"seed_frame={SEED_FRAME_INDEX};ingest=in_memory_arrays"
)

#: What this arm refuses to be read as. Copied verbatim into the comparison artifact, beside the
#: number, because a number that satisfies the letter of a discharge condition is exactly the
#: number somebody will read as the discharge.
_DISCHARGES = (
    "NOTHING. This arm PRODUCES the measurement apple_sam2's third gate-qualification blocker "
    "names; accepting it is a separate act and a human's. GATE_QUALIFIED and "
    "GATE_QUALIFICATION_BLOCKERS are imported read-only here and are untouched by any run of "
    "this module, and the harness still stamps estimator_not_gate_qualified."
)

#: Everything a reader of the artifact needs in order to know that the two arms were comparable,
#: recorded as fields rather than argued in a docstring nobody will open.
PROPAGATION_CONTRACT: dict[str, Any] = {
    "name": PROPAGATION_NAME,
    "version": PROPAGATION_VERSION,
    "upstream": apple_sam2.UPSTREAM_PROPAGATION,
    "topology": (
        "GroundingDINO on frame 0 -> highest-scoring box -> "
        "SAM2VideoPredictor.add_new_points_or_box(frame_idx=0) -> propagate_in_video() forward "
        "across the whole clip. One detection, one seed, N propagated masks."
    ),
    # THE CONFOUND, AND THE MECHANISM THAT EXCLUDES IT.
    "frame_ingest": (
        "in-memory uint8 arrays. sam2.utils.misc.load_video_frames is replaced, for the duration "
        "of exactly one init_state call, by frames_to_normalized_tensor, which performs upstream's "
        "own _load_img_as_tensor arithmetic (PIL resize to image_size, /255, ImageNet mean/std) on "
        "the arrays the harness already holds. No file is written and no image is encoded, so this "
        "arm and the per-frame arm are handed the same bytes."
    ),
    "jpeg_encoded": False,
    "lossy_encode_anywhere": False,
    "why_that_matters": (
        "SAM2VideoPredictor conventionally ingests a directory of JPEGs and our captures are "
        "lossless rgb.npy. Had one arm seen a transcode of the other's frames, the difference "
        "between the two p95s would be the codec plus propagation reported as propagation — void "
        "rather than merely weak, and not visibly wrong."
    ),
    "seed": {
        "frame_index": SEED_FRAME_INDEX,
        "object_id": SEED_OBJECT_ID,
        "detector": apple_sam2.GROUNDING_DINO_MODEL_CHECKPOINT,
        "prompt": apple_sam2.OBJECT_TEXT_PROMPT,
        "box_selection": apple_sam2.BOX_SELECTION,
        "box_threshold": apple_sam2.BOX_THRESHOLD,
        "text_threshold": apple_sam2.TEXT_THRESHOLD,
        "retry_box_threshold": apple_sam2.RETRY_BOX_THRESHOLD,
        "retry_text_threshold": apple_sam2.RETRY_TEXT_THRESHOLD,
        "no_seed_is_a_refusal_not_an_empty_run": True,
    },
    "sam2_checkpoint": apple_sam2.SAM2_MODEL_CHECKPOINT,
    "sam2_revision": apple_sam2.SAM2_MODEL_REVISION,
    "grounding_dino_checkpoint": apple_sam2.GROUNDING_DINO_MODEL_CHECKPOINT,
    "grounding_dino_revision": apple_sam2.GROUNDING_DINO_MODEL_REVISION,
    "object_text_prompt": apple_sam2.OBJECT_TEXT_PROMPT,
    # THE ONE THING THAT IS DELIBERATELY NOT THE SAME, named where a reader will find it.
    "mask_validity_filter_applied": False,
    "mask_validity_filter_reason": (
        "Cosmos-Transfer2.5 has no such filter and this arm exists to measure what the "
        "generator's segmenter does. Applying a per-frame validity refusal to a propagated mask "
        "would also erase the failure this experiment looks for: a mask drifted onto the table "
        "would be refused, counted as 'no mask', and the run of consecutive low-IoU frames — "
        "blocker 3's failure mode (b) — would never appear. What the filter would have refused is "
        "counted in stats() anyway, so the choice is visible rather than silent."
    ),
    # WHICH OF THIS ARM'S COUNTERS IS WHICH OF THE PER-FRAME ARM'S, stated as a field because the
    # only reason to record "what the filter would have refused" is to subtract it from what the
    # filter DID refuse on the other arm — and a subtraction of two counters that mirror different
    # sets of branches is not a difference of anything. Since PR-08 V10 segment() refuses a
    # non-empty mask on TWO branches; this module mirrored one of them until 2026-08-27, which made
    # the pair non-comparable on any capture whose background is warm enough to move the colour
    # reference off the fruit (measured 37.2-56.4 % of the frame on train-01-oak-tungsten, against
    # the 0.10 bound). The MuJoCo captures that have actually run are unaffected, and the record is
    # thinner than "the counter reads zero on all three": EST_DRIFT-mujoco-trajectory-f480 carries
    # n_frames_mask_refused_reference_not_object_scale = 0 against 1 refusal in 480 calls, while
    # s60-f720 (25 of 720) and s20-f240 (12 of 240) were measured by a PRE-V10 adapter and carry no
    # such field at all — there the branch could not have fired because it did not exist. Either
    # way the misfire is not in those numbers, so this repairs the meaning BEFORE the first warm
    # capture rather than after it.
    "colour_filter_mirror": {
        "n_frames_the_colour_filter_would_have_refused": "apple_sam2 MASK_REFUSED_FRAMES",
        "n_frames_the_colour_filter_would_have_refused_reference_not_object_scale": (
            "apple_sam2 MASK_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES (inside the total)"
        ),
        "n_frames_the_colour_filter_would_have_refused_no_reference": (
            "apple_sam2 MASK_REFUSED_NO_REFERENCE_FRAMES (inside the total)"
        ),
        "n_frames_the_colour_filter_mirror_could_not_evaluate": (
            "no counterpart: segment() refuses the RUN for a label the filter has no reference "
            "for, so on those frames there is no per-frame refusal count to mirror and the zeros "
            "above are not evidence of agreement."
        ),
        "branches_mirrored": ("reference_is_object_scale", "mask_validity_iou"),
        "branch_order": "segment()'s own: object-scale first, and the IoU test only when it passes",
    },
    "discharges": _DISCHARGES,
}


class PropagationSeedNotFound(RuntimeError):
    """GroundingDINO found no box on the seed frame, so there is no mask to propagate.

    A refusal rather than a clip of empty masks: the second reads out of the harness as coverage
    0.0, i.e. as a fact about what SAM 2's tracker did, when the true statement is that the
    tracker was never started.
    """


# -- counters, in the shape apple_sam2.stats() uses ------------------------------------------------

#: Clips propagated by this process.
PROPAGATION_RUNS = 0
#: Frames propagated across, summed over runs.
PROPAGATED_FRAMES = 0
#: Propagated masks that came back empty. Not an error — the tracker reports the object as absent,
#: which on this capture is what an occlusion looks like.
EMPTY_PROPAGATED_FRAMES = 0
#: Frames whose propagated mask would have been REFUSED by apple_sam2's colour filter had it been
#: applied. Recorded, never acted on: see PROPAGATION_CONTRACT["mask_validity_filter_reason"].
#:
#: THE WHOLE POINT OF THIS COUNTER IS THAT IT IS THE PER-FRAME ARM'S ``MASK_REFUSED_FRAMES``
#: COMPUTED OVER THIS ARM'S MASKS, so it must mirror **every** branch on which ``segment()``
#: returns all-False and increments that counter, in ``segment()``'s own order. Since PR-08 V10
#: there are two of them (``apple_sam2.py``, the ``reference_is_object_scale`` branch and the
#: ``mask_validity_iou`` branch), and this module mirrored only the second until 2026-08-27. A
#: mirror of one branch is not a smaller mirror — on any warm-background capture the per-frame arm
#: refuses through the branch that was missing, so the two arms' refusal counts stop being the same
#: quantity and the propagation-vs-per-frame comparison becomes a comparison of two different
#: refusal populations. The three counters below are the same decomposition ``apple_sam2.stats()``
#: publishes, so the arms can be subtracted term by term rather than only in total.
WOULD_HAVE_BEEN_REFUSED_FRAMES = 0
#: Inside :data:`WOULD_HAVE_BEEN_REFUSED_FRAMES`, not beside it — exactly as
#: ``MASK_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES`` sits inside ``MASK_REFUSED_FRAMES``. The V10
#: branch: the colour reference covers more of the frame than an object can, so it cannot arbitrate
#: this mask in either direction and ``segment()`` refuses the frame as undecidable.
WOULD_HAVE_BEEN_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES = 0
#: Also inside :data:`WOULD_HAVE_BEEN_REFUSED_FRAMES`, mirroring ``MASK_REFUSED_NO_REFERENCE_FRAMES``:
#: the IoU branch fired on a frame where the reference is EMPTY, i.e. "nothing here can confirm the
#: mask" rather than "the mask is demonstrably the plate". ``apple_sam2`` keeps the two apart
#: because they have opposite implications for the budget; an arm that pooled them would answer that
#: question differently from the arm it is compared against.
WOULD_HAVE_BEEN_REFUSED_NO_REFERENCE_FRAMES = 0
#: Frames on which the mirror could not be computed AT ALL because the colour filter has no
#: reference for the label this process was told to segment (``WAM_PR08_OBJECT_PROMPT="plate."``,
#: PR-08 V10 §2). ``segment()`` does not refuse such a frame — it refuses the whole RUN, before any
#: counter moves — so there is no per-frame number to mirror and a zero in the counters above would
#: be read as "the filter would have refused nothing", which is the opposite of what happened. This
#: arm does not adopt the run-level refusal, because whether §6's second label is measured with the
#: filter off is an OWNER decision (PR-08 V10 §2's closing paragraph, and it is unanswered); what it
#: does instead is make the gap countable rather than silent.
FILTER_MIRROR_UNEVALUABLE_FRAMES = 0
#: The seed box, as [x0, y0, x1, y1], of the most recent run.
LAST_SEED_BOX: list[float] | None = None


def reset_counters() -> None:
    """Zero the counters. The models are ``apple_sam2``'s and are not touched here."""
    global PROPAGATION_RUNS, PROPAGATED_FRAMES, EMPTY_PROPAGATED_FRAMES
    global WOULD_HAVE_BEEN_REFUSED_FRAMES, LAST_SEED_BOX
    global WOULD_HAVE_BEEN_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES
    global WOULD_HAVE_BEEN_REFUSED_NO_REFERENCE_FRAMES, FILTER_MIRROR_UNEVALUABLE_FRAMES
    PROPAGATION_RUNS = PROPAGATED_FRAMES = EMPTY_PROPAGATED_FRAMES = 0
    WOULD_HAVE_BEEN_REFUSED_FRAMES = 0
    WOULD_HAVE_BEEN_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES = 0
    WOULD_HAVE_BEEN_REFUSED_NO_REFERENCE_FRAMES = 0
    FILTER_MIRROR_UNEVALUABLE_FRAMES = 0
    LAST_SEED_BOX = None


def stats() -> dict[str, Any]:
    """What this arm did, for the artifact. Cumulative over the process, like ``apple_sam2``'s.

    The three ``would_have_refused`` numbers are the per-frame arm's ``n_frames_mask_refused``,
    ``n_frames_mask_refused_reference_not_object_scale`` and ``n_frames_mask_refused_no_reference``
    recomputed over THIS arm's masks, with the same nesting: the last two are inside the first.
    They are only comparable with the per-frame arm's if every refusal branch is mirrored, which is
    why ``colour_filter_mirror_covers_both_refusal_branches`` is stated as a field a reader of the
    artifact can check rather than left as a property of the source.
    """
    return {
        "n_propagation_runs": PROPAGATION_RUNS,
        "n_frames_propagated": PROPAGATED_FRAMES,
        "n_frames_empty_propagated_mask": EMPTY_PROPAGATED_FRAMES,
        "n_frames_the_colour_filter_would_have_refused": WOULD_HAVE_BEEN_REFUSED_FRAMES,
        "n_frames_the_colour_filter_would_have_refused_reference_not_object_scale": (
            WOULD_HAVE_BEEN_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES
        ),
        "n_frames_the_colour_filter_would_have_refused_no_reference": (
            WOULD_HAVE_BEEN_REFUSED_NO_REFERENCE_FRAMES
        ),
        "n_frames_the_colour_filter_mirror_could_not_evaluate": FILTER_MIRROR_UNEVALUABLE_FRAMES,
        "colour_filter_mirror_covers_both_refusal_branches": True,
        "colour_filter_applied": False,
        "seed_box_xyxy": list(LAST_SEED_BOX) if LAST_SEED_BOX is not None else None,
    }


def available() -> bool:
    """Are the weights here, at their pinned revisions? Delegated — they are the same weights."""
    return apple_sam2.available()


# -- the ingest, which is where the whole comparison lives or dies ---------------------------------


def frames_to_normalized_tensor(
    frames: Sequence[np.ndarray],
    image_size: int,
    *,
    offload_video_to_cpu: bool,
    compute_device: Any,
    img_mean: Sequence[float] = (0.485, 0.456, 0.406),
    img_std: Sequence[float] = (0.229, 0.224, 0.225),
) -> tuple[Any, int, int]:
    """``(images, video_height, video_width)`` from in-memory ``uint8`` RGB frames.

    A drop-in for ``sam2.utils.misc.load_video_frames``'s return, computed with the arithmetic
    ``_load_img_as_tensor`` uses on a file — ``Image.convert("RGB").resize((S, S))``, ``/255``,
    then the ImageNet mean/std — so that the ONLY difference from upstream's path is that no image
    was encoded on the way in. That is asserted bitwise in the tests, against the same frames
    written as lossless PNGs and read back through upstream's own loader.

    Refuses a float array (``apple_sam2._as_uint8_rgb``'s reason: ``[0, 1]`` and ``[0, 255]`` are
    indistinguishable from the array alone and rescale to different pictures) and a clip whose
    frames are not all one grid (a centroid compared across grids is not a displacement).
    """
    import torch
    from PIL import Image

    if not len(frames):
        raise ValueError("nothing to propagate: the clip is empty.")
    checked = [apple_sam2._as_uint8_rgb(f) for f in frames]
    shapes = {f.shape[:2] for f in checked}
    if len(shapes) > 1:
        raise ValueError(
            f"the clip mixes frame geometries {sorted(shapes)} and a propagated mask is scored "
            "against ground truth at ONE grid. Every frame must be one grid."
        )
    height, width = checked[0].shape[:2]

    images = torch.zeros(len(checked), 3, image_size, image_size, dtype=torch.float32)
    for i, frame in enumerate(checked):
        resized = np.array(Image.fromarray(frame).convert("RGB").resize((image_size, image_size)))
        images[i] = torch.from_numpy(resized / 255.0).permute(2, 0, 1)
    mean = torch.tensor(tuple(img_mean), dtype=torch.float32)[:, None, None]
    std = torch.tensor(tuple(img_std), dtype=torch.float32)[:, None, None]
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        mean = mean.to(compute_device)
        std = std.to(compute_device)
    images -= mean
    images /= std
    return images, height, width


@contextlib.contextmanager
def _in_memory_frames(frames: Sequence[np.ndarray]) -> Iterator[None]:
    """Install :func:`frames_to_normalized_tensor` as ``load_video_frames`` for one ``init_state``.

    The narrowest seam that gets arrays into the predictor: the name is rebound in
    ``sam2.sam2_video_predictor``'s own namespace for the duration of the call and put back
    afterwards, so every other line of ``init_state`` — and the entire tracking path — is the
    installed package's code, unmodified. Building the inference state by hand instead would fork
    forty lines of upstream state setup, and a fork is what goes stale.
    """
    import sam2.sam2_video_predictor as predictor_module

    original = predictor_module.load_video_frames

    def _load(
        video_path,
        image_size,
        offload_video_to_cpu,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
        async_loading_frames=False,
        compute_device=None,
    ):
        return frames_to_normalized_tensor(
            frames,
            image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            compute_device=compute_device,
            img_mean=img_mean,
            img_std=img_std,
        )

    predictor_module.load_video_frames = _load
    try:
        yield
    finally:
        predictor_module.load_video_frames = original


# -- the video predictor, loaded once per process at the pinned revision ---------------------------

_VIDEO_PREDICTOR: Any | None = None


def reset_models() -> None:
    """Drop the cached video predictor. ``apple_sam2``'s own caches are its business."""
    global _VIDEO_PREDICTOR
    _VIDEO_PREDICTOR = None


def _video_predictor() -> Any:
    """``SAM2VideoPredictor`` at :data:`apple_sam2.SAM2_MODEL_REVISION`, cached.

    Reproduces ``build_sam2_video_predictor_hf``'s two steps the way ``apple_sam2._predictor``
    reproduces ``build_sam2_hf``'s, and for the same reason: only the download half can carry a
    revision, and the ``_hf`` helper does not pass one. Same repository, same commit, same config
    name as the per-frame arm's image predictor — the weights are identical and the difference
    between the arms is the tracker.
    """
    global _VIDEO_PREDICTOR
    if _VIDEO_PREDICTOR is None:
        ckpt = apple_sam2.CHECKPOINTS[1]
        apple_sam2._require_cached(ckpt)
        import sam2.build_sam as build_sam_mod
        from huggingface_hub import hf_hub_download

        filenames = getattr(build_sam_mod, "HF_MODEL_ID_TO_FILENAMES", None)
        build_video = getattr(build_sam_mod, "build_sam2_video_predictor", None)
        if filenames is None or build_video is None:
            raise apple_sam2.EstimatorCheckpointUnusable(
                apple_sam2._sam2_api_message(
                    "HF_MODEL_ID_TO_FILENAMES and build_sam2_video_predictor"
                )
            )
        if ckpt.repo_id not in filenames:
            raise apple_sam2.EstimatorCheckpointUnusable(
                f"`sam2` does not know the checkpoint {ckpt.repo_id!r}: it is not in "
                "sam2.build_sam.HF_MODEL_ID_TO_FILENAMES, so there is no config file to build the "
                "video predictor with."
            )
        config_name, checkpoint_name = filenames[ckpt.repo_id]
        with apple_sam2._offline_hub():
            ckpt_path = hf_hub_download(
                repo_id=ckpt.repo_id,
                filename=checkpoint_name,
                revision=ckpt.revision,
                local_files_only=apple_sam2._local_files_only(),
            )
            _VIDEO_PREDICTOR = build_video(
                config_file=config_name, ckpt_path=ckpt_path, device=apple_sam2._device()
            )
    return _VIDEO_PREDICTOR


def seed_box(frame: np.ndarray) -> np.ndarray | None:
    """The box the propagation is seeded with: ``apple_sam2``'s own detector, unchanged.

    Reached through ``apple_sam2._best_box`` rather than reimplemented, so the prompt, both
    thresholds, the single retry, the box-selection rule and the two retry counters are literally
    the per-frame arm's. If this were a copy, the two arms could drift apart in the detector — and
    the difference between their p95s would then not be propagation.
    """
    return apple_sam2._best_box(apple_sam2._as_uint8_rgb(frame))


def propagate(rgbs: Sequence[np.ndarray]) -> list[np.ndarray]:
    """One mask per frame, seeded on frame 0 and PROPAGATED forward across the clip.

    This is the arm blocker 3 asks for. ``rgbs`` are the harness's own in-memory frames and they
    are never written, encoded or re-read; see :data:`PROPAGATION_CONTRACT` for why that is the
    load-bearing property of this function and not an implementation detail.
    """
    global PROPAGATION_RUNS, PROPAGATED_FRAMES, EMPTY_PROPAGATED_FRAMES
    global WOULD_HAVE_BEEN_REFUSED_FRAMES, LAST_SEED_BOX
    # EVERY counter this function touches is declared here, and the reason is a concrete bug rather
    # than style: a name incremented with `+=` and NOT named in a `global` is a local, so the first
    # frame the new branch fires on raises UnboundLocalError — i.e. the run dies on exactly the
    # warm-background frame the branch was added to count, and a source-grep test would still pass.
    global WOULD_HAVE_BEEN_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES
    global WOULD_HAVE_BEEN_REFUSED_NO_REFERENCE_FRAMES, FILTER_MIRROR_UNEVALUABLE_FRAMES

    import torch

    frames = [apple_sam2._as_uint8_rgb(f) for f in rgbs]
    if not frames:
        raise ValueError("nothing to propagate: the clip is empty.")
    height, width = frames[SEED_FRAME_INDEX].shape[:2]

    predictor = _video_predictor()
    box = seed_box(frames[SEED_FRAME_INDEX])
    if box is None:
        raise PropagationSeedNotFound(
            "GroundingDINO returned no box for "
            f"{apple_sam2.OBJECT_TEXT_PROMPT!r} on frame {SEED_FRAME_INDEX} at "
            f"threshold={apple_sam2.BOX_THRESHOLD}/{apple_sam2.TEXT_THRESHOLD} (retry "
            f"{apple_sam2.RETRY_BOX_THRESHOLD}/{apple_sam2.RETRY_TEXT_THRESHOLD}), so there is no "
            "mask to propagate and the tracker was never started. This is refused rather than "
            "returning a clip of empty masks, which would read as a fact about SAM 2's tracker."
        )
    LAST_SEED_BOX = [float(v) for v in np.asarray(box).reshape(-1)]

    with _in_memory_frames(frames):
        state = predictor.init_state(
            video_path=frames,
            offload_video_to_cpu=OFFLOAD_VIDEO_TO_CPU,
            offload_state_to_cpu=OFFLOAD_STATE_TO_CPU,
        )
    try:
        with torch.inference_mode():
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=SEED_FRAME_INDEX,
                obj_id=SEED_OBJECT_ID,
                box=np.asarray(box, dtype=np.float32).reshape(4),
            )
            out: dict[int, np.ndarray] = {}
            for frame_idx, _obj_ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=SEED_FRAME_INDEX
            ):
                # `logits` is (n_objects, 1, H, W) at the VIDEO resolution. One object, and > 0 is
                # SAM 2's own mask threshold — the same one the image predictor's `predict` applies.
                out[int(frame_idx)] = np.asarray(
                    (logits[0, 0] > 0.0).detach().cpu().numpy(), dtype=bool
                )
    finally:
        with contextlib.suppress(Exception):
            predictor.reset_state(state)

    masks: list[np.ndarray] = []
    for i in range(len(frames)):
        mask = out.get(i)
        if mask is None:
            # propagate_in_video visits every frame from the seed forward, so this cannot happen
            # on a forward pass from frame 0 — and if it ever does, an all-False mask that says so
            # is better than a KeyError halfway through a 480-frame run.
            mask = np.zeros((height, width), dtype=bool)
        masks.append(mask)
        if not mask.any():
            # An empty mask is not a refusal on either arm: segment() counts it in EMPTY_MASK_FRAMES
            # and returns BEFORE the validity check, so the mirror must not run here either.
            EMPTY_PROPAGATED_FRAMES += 1
        elif not apple_sam2.mask_validity_reference_is_defined():
            # segment() would not have refused this FRAME — it would have refused the RUN, on its
            # first call and before any counter moved (PR-08 V10). There is therefore no per-frame
            # refusal to mirror, and counting these frames as "the filter would have refused
            # nothing" would manufacture agreement between the arms out of a label the filter does
            # not cover. Asked per frame rather than hoisted, because segment() asks per call.
            FILTER_MIRROR_UNEVALUABLE_FRAMES += 1
        else:
            # BOTH of segment()'s refusal branches, in segment()'s own order and with segment()'s
            # own nesting. The object-scale test GATES the IoU test there rather than sitting beside
            # it (apple_sam2, PR-08 V10): when the reference covers the scene it refuses a correct
            # mask of a restyled apple and ACCEPTS a mask of the warm background, so reaching the
            # IoU comparison at all is the fail-OPEN half of the same defect. Mirroring the two as
            # independent tests would therefore count a different set of frames than the arm this
            # counter exists to be compared with.
            reference = apple_sam2.object_color_reference(frames[i])
            if not apple_sam2.reference_is_object_scale(reference):
                WOULD_HAVE_BEEN_REFUSED_FRAMES += 1
                WOULD_HAVE_BEEN_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES += 1
            elif apple_sam2.mask_validity_iou(mask, reference) < apple_sam2.MASK_VALIDITY_MIN_IOU:
                WOULD_HAVE_BEEN_REFUSED_FRAMES += 1
                if not reference.any():
                    WOULD_HAVE_BEEN_REFUSED_NO_REFERENCE_FRAMES += 1

    PROPAGATION_RUNS += 1
    PROPAGATED_FRAMES += len(masks)
    return masks
