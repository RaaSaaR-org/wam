#!/usr/bin/env python3
"""PR-08 §4's estimator pair: GroundingDINO -> SAM 2 for the mask, Depth-Anything-V2 for the metres.

    segment(rgb)        -> (H, W) bool     the apple in THIS frame; ALL-FALSE when it is not there
    estimate_depth(rgb) -> (H, W) float32  METRES from the camera

This is the one thing PR-08 §8 item 4 was waiting on. Both halves of that item need an object
segmenter and §4 step 2 requires it to be **the same one**, so a single adapter closes ``GEOM_TOL``
(``scripts/measure_geom_tol.py``, which reaches this module directly and also accepts masks dumped
from it through ``--masks`` + a ``masks.meta.json``) and ``EST_DRIFT_P95``
(``scripts/measure_est_drift.py measure --estimators estimators.apple_sam2``) together. Wiring two segmenters would have been the cheaper-looking option and it would make §6's
``GEOM_TOL - EST_DRIFT_P95`` a subtraction of two different quantities — two plausible pixel numbers
that subtract to a plausible pixel number, with nothing in the pipeline able to notice.

WHICH NUMBER IS THE GATE
------------------------
``EST_DRIFT_P95`` is the **p95 of the CENTROID displacement**, in pixels, between the mask
:func:`segment` returns and Isaac's ground-truth mask of the same frame. It is the only number here
that enters a gate: §6 G0b holds the restyled corpus to ``GEOM_TOL - EST_DRIFT_P95``. So the gate
rides entirely on :func:`segment`.

:func:`estimate_depth` exists because §4 **step 3** asks for the absolute depth error in metres
alongside the centroid displacement, and because Transfer2.5 consumes an estimated depth map as a
conditioning signal (§4). It is **recorded, not gated**. Saying which is which matters: a reader who
assumes the depth error is the budget will tune the wrong half of this file.

WHY THE GENERATOR'S OWN SEGMENTER, DOWN TO THE CHECKPOINT ID
------------------------------------------------------------
§4 step 2's "the same segmenter" has a weak reading (the same one on both sides of *our*
measurement) and a strong one (the same one the *generator* will use, so that the drift we budget
for is the drift the generator actually commits). This module takes the strong reading, and the
checkpoint ids below are not chosen — they are read off Cosmos-Transfer2.5's own auxiliary
segmenter, ``cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_model.py``, which names
``SAM2_MODEL_CHECKPOINT = "facebook/sam2-hiera-large"`` and
``GROUNDING_DINO_MODEL_CHECKPOINT = "IDEA-Research/grounding-dino-base"`` and drives them as
*text prompt -> GroundingDINO box -> SAM 2 mask*. ``sam2`` 1.1.0 is already installed in the
Transfer2.5 venv on Discoverer+ with its hiera configs (T-040 notes, verified 2026-08-21), so this
adapter is the missing piece and not a new dependency.

Anything that would make our estimate differ from the generator's — a different SAM 2 size, a
different detector, a hand-placed point prompt — makes ``EST_DRIFT_P95`` a budget for an error the
generator does not commit and leaves the one it does commit unbudgeted.

**The strong reading reaches past the weights, and as of 2026-08-22 so does this module.** A
checkpoint id is not an operating point: the same GroundingDINO at ``threshold=0.35`` and at
``threshold=0.15`` returns different boxes on different frames, which is a different mask, a
different centroid and a different no-detection rate. Upstream's numbers are therefore adopted
verbatim — ``0.15`` / ``0.25``, one retry at ``(0.10, 0.10)`` when no box is found at all, and the
highest-scoring box — and they are pinned in :data:`SEGMENTER_CONTRACT`, committed to
``configs/transfer25/pr08_geom_tol.json`` before the measurement, and cross-checked by
``measure_est_drift`` field for field. They are not ours to improve; see the comment above
:data:`BOX_THRESHOLD`.

**One difference remains and it is not papered over.** Upstream propagates a single mask across the
clip with ``SAM2VideoPredictor``; this adapter segments each frame independently, because
``segment(rgb)`` is the contract both PR-08 §4 harnesses call. That is the LAST blocker in
:data:`GATE_QUALIFICATION_BLOCKERS` — named that way and not by an index, because the tuple shrinks
as blockers are discharged and an index in a comment goes stale silently — with the argument that
it biases ``EST_DRIFT_P95`` in BOTH
directions — inflating the per-frame tail we do measure, and hiding the tracking drift we do not.

A REPO ID IS NOT A CHECKPOINT ID: EVERY LOAD IS PINNED TO A COMMIT
------------------------------------------------------------------
``facebook/sam2-hiera-large`` names a repository, not weights. The repository can move, and a gate
number traceable only to "whatever ``main`` was that day" is not traceable at all (AC-04). So each
of the three checkpoints carries a 40-hex commit beside it — ``*_REVISION_DEFAULT`` below, read off
the HF API on 2026-08-22 — and that revision is threaded through **every** hub call this module
makes: the availability probe, the pre-load cache check, and all three loaders. It is also stamped
into :data:`ESTIMATOR_VERSION`, which is what the artifact records, so the committed gate number
identifies the exact weights.

This matters in a second, sharper way. ``cluster/discoverer/102_stage_sam2_weights.sbatch`` stages
these repos **at those commits**, and ``huggingface_hub`` writes no ``refs/main`` entry when the
requested revision is a commit hash. A cache probe with no revision therefore cannot see a
correctly staged cache: it reports "not cached" on the one machine where the weights actually are,
and the documented escape hatch (``WAM_PR08_ALLOW_DOWNLOAD=1``) would then fetch ``main`` — a
DIFFERENT revision from the staged, checksum-verified one, with nothing recording that it differed.
Pinning the probe is what makes the staged cache visible and the fallback unnecessary.

**Single source of truth.** The ids and revisions live here and nowhere else; the staging job takes
them from this file — extracting them, or restating them behind a check that FAILS the job when the
restated value and the value here disagree — so that the two cannot drift silently. The extraction
contract is: every pin is a module-level assignment of a bare string literal, one per line, named
``<THING>_MODEL_ID_DEFAULT`` / ``<THING>_MODEL_REVISION_DEFAULT``, e.g.

    sed -n 's/^SAM2_MODEL_REVISION_DEFAULT = "\\([0-9a-f]\\{40\\}\\)"$/\\1/p' scripts/estimators/apple_sam2.py

Nothing but the six ``*_DEFAULT`` lines may be reformatted without updating that job.

THE DEPTH CHECKPOINT IS METRIC ON PURPOSE, AND THE RELATIVE ONE IS REFUSED
--------------------------------------------------------------------------
:data:`DEPTH_MODEL_CHECKPOINT` defaults to ``depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf``.
The flagship Depth-Anything-V2 checkpoints (``...-V2-Large-hf`` and friends) are **relative**: they
emit an affine-free *inverse* depth, arbitrarily scaled per image, where a LARGER value means
NEARER. Subtracting that from Isaac's ``distance_to_camera`` metres yields a number with the units
of nothing, ordered backwards, and — this is the part that costs — it looks exactly like a depth
error in metres in the artifact, under a key called ``mean_m``. That is the failure this repository
keeps naming: it does not crash, it sets a number, and every reader downstream inherits it.
``depth-anything/Video-Depth-Anything-Large`` is relative in exactly this sense and is NOT the
checkpoint this pair uses, whatever a staging job may have fetched first.

So the metric variant is the default, ``metric`` vs ``relative`` is **read off the loaded model's
config** rather than inferred from the id (an id can be overridden; a config cannot lie), it is
recorded in :data:`DEPTH_ESTIMATION_TYPE` / :data:`DEPTH_IS_METRIC`, and a relative checkpoint makes
:func:`estimate_depth` **refuse at load time** — before frame 0, not on frame 37 — naming the
checkpoint, the config value, the env var and the metric alternatives. ``Indoor`` rather than
``Outdoor`` because the metric heads are domain-fine-tuned and carry a ``max_depth`` (20 m vs 80 m):
AppleToPlate is a tabletop at well under 2 m and the Isaac calibration renders are the same scene,
so the outdoor head would spend its output range on distances that never occur. ``Large`` because
this runs once over a few dozen calibration frames, not inside the generation loop, so accuracy
costs nothing that matters here.

WHAT IT REFUSES TO DO
---------------------
**It will not return a stale mask.** A frame where GroundingDINO finds nothing is a REAL EVENT in
this corpus — the Dex3 hand occludes the apple, or the apple leaves frame — and it returns an
all-False mask. ``measure_est_drift`` turns that into ``centroid_of_mask(...) is None``, which is
DROPPED and COUNTED into ``coverage``. Returning the previous frame's mask, or raising, would each
destroy that: one invents a displacement that was never observed, the other kills a run over a
frame that is not an error.

**It will not return a mask of an object it was not asked for.** As of 2026-08-22 every non-empty
mask is checked against a second, non-learned opinion about where the fruit is — the warm-and-
saturated colour predicate ``build_identity_calibration.apple_mask`` uses, reimplemented here as
:func:`object_color_reference` — and a mask whose IoU against it is below
:data:`MASK_VALIDITY_MIN_IOU` is REFUSED: :func:`segment` returns all-False, exactly as it does for a
frame with no detection, and the frame is counted in :data:`MASK_REFUSED_FRAMES`. This is a validity
check on the OUTPUT and it is emphatically NOT a change to the detection — see the block above
:data:`MASK_VALIDITY_MIN_IOU`, and PR-08 V6 (``docs/preregistration/PR-08-V6-mask-validity.md``),
which registers it.

**And it will not pretend to have decided a frame that check cannot decide.** As of 2026-08-23 the
reference is checked before it is used as one, because it is a predicate for ONE object under ONE
appearance and this module was applying it to any label on any pixels. A label outside
:data:`MASK_VALIDITY_REFERENCE_LABELS` refuses the RUN — :class:`MaskValidityReferenceUndefined`,
rather than the ``coverage: 0.0`` that a ``plate.`` pass used to report as a fact about the corpus —
and a frame whose reference covers more than
:data:`MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION` of the picture is refused as undecidable and
counted in :data:`MASK_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES`, because on a warm-table restyle
the predicate stops describing the fruit and starts describing the table. PR-08 V10
(``docs/preregistration/PR-08-V10-mask-validity-reference-scope.md``), **UNSIGNED**.

**It will not rescale a float image.** ``rgb`` must be ``uint8``. A float array in [0, 1] and a
float array in [0, 255] are indistinguishable from the array alone and rescale to different
pictures, which is a different detection, which is a different centroid.

**It will not fetch 3 GB because someone ran a script, and it enforces that rather than intending
it.** ``from_pretrained`` downloads by default, and this project's rule is that nothing is fetched
at scale without asking (T-040; and the provider forbids agents on the login node at all). Unless
``WAM_PR08_ALLOW_DOWNLOAD=1`` says the decision has been taken, every load runs inside
:func:`_offline_hub`, which sets ``huggingface_hub.constants.HF_HUB_OFFLINE = True`` for the
duration — ``constants.is_offline_mode()`` is read at request time by the hub's own HTTP hook
(verified against the installed ``huggingface_hub`` 1.25.1), so a request cannot leave the machine —
*and* passes ``local_files_only=True`` to each loader, because ``transformers`` keeps its own copy
of that flag from its import. Belt and braces on purpose: the pre-check alone was a check a load
could walk past. The refusal names every checkpoint, its pinned revision and every cache directory
it looked in, and :func:`available` answers "are they here, at those revisions?" without loading
anything — which is what ``measure_geom_tol``'s ``--method auto`` probes before it will select this
adapter, because a segmenter running without its checkpoints does not crash. It returns empty
masks, every step drops, and ``coverage: 0.0`` reads as a fact about the corpus.

**It will not load SAM 2 through ``SAM2ImagePredictor.from_pretrained``.** That path reaches
``sam2.build_sam.build_sam2_hf`` -> ``_hf_download(model_id)``, which calls ``hf_hub_download``
with no ``revision`` and forwards none of its caller's kwargs (read off ``sam2`` 1.1.0's
``build_sam.py``, 2026-08-22), so it resolves ``refs/main`` and cannot be pinned. This module
performs the same two steps itself — resolve the checkpoint file at :data:`SAM2_MODEL_REVISION`,
then ``build_sam2(config_file=..., ckpt_path=...)`` with the id's own config name from
``HF_MODEL_ID_TO_FILENAMES`` — so the weights SAM 2 loads are the weights the artifact names. If
those two names are absent from the installed ``sam2``, this refuses; it does not fall back to the
unpinned loader.

**It will not check the weights only where the happy path happens to look.** Both models are loaded
at the START of :func:`segment`, before detection, so a machine missing SAM 2's checkpoint refuses
on frame 0 rather than on the first frame that happens to contain a detectable apple. A capture in
which nothing is detected would otherwise run to completion with no segmenter at all and report
``coverage: 0.0`` as a fact about the corpus — the exact outcome :func:`available` exists to
prevent.

**It will not paper over a shape change in transformers.** The depth pipeline's post-processing
moved between transformers versions (the image processor's ``post_process_depth_estimation`` now
resizes to the source grid; it did not always). If the returned map is not the input's ``(H, W)``,
this refuses — a depth error averaged across two grids is not a depth error.

**It will not import its dependencies lazily enough to hide them.** Package *availability* is
checked at import time and raises an :class:`ImportError` subclass, so
``measure_est_drift.resolve_estimators`` catches it and prints the whole message instead of a
traceback. Model *loading* stays lazy and module-level-cached, one load per process. Missing
*weights* are discovered at first load and are deliberately NOT catchable that way: by then a run is
underway, and an artifact written from a model that never loaded is the thing gate qualification
exists to prevent.

WHAT IT CANNOT REFUSE ALONE, AND SO EXPORTS SO THAT SOMETHING ELSE CAN
-----------------------------------------------------------------------
The object this module looks for and the object the harness scores it against used to be two
independent knobs: :data:`OBJECT_TEXT_PROMPT` (``$WAM_PR08_OBJECT_PROMPT``) here, and
``measure_est_drift --object-class`` there. Change one and not the other and an apple mask is
compared against a plate's ground truth: a large but entirely plausible p95, no crash, no drop in
coverage. This module still cannot see that flag — but it no longer has to, because the flag now
DEFAULTS to this module's prompt and an explicit value naming a different object is fatal there.
The same shape applies to everything else that has to be equal on both sides of §6's subtraction:
this module cannot check it, so it EXPORTS it, once, as :data:`SEGMENTER_CONTRACT`, and the two
harnesses do the checking. A constant that is only in the code cannot be cross-checked by a script
reading two JSON artifacts six months later.

GATE QUALIFICATION
------------------
:data:`GATE_QUALIFIED` is ``False``. See :data:`GATE_QUALIFICATION_BLOCKERS` for the specific,
checkable conditions and the reasoning; flipping it is a reviewable edit to that tuple, not a
judgement someone re-makes from scratch. Conditions that HAVE been discharged move to
:data:`GATE_QUALIFICATION_DISCHARGED` with the evidence, rather than disappearing — a blocker that
vanishes between two commits looks identical whether it was satisfied or deleted, and only one of
those is allowed to shorten this list. ``measure_est_drift`` reads the flag with a default of
``False`` and stamps ``estimator_not_gate_qualified`` into the artifact, which still gets written —
"we tried and this is what came out" is a record.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import re
import sys
from typing import Any, Iterator, NamedTuple

import numpy as np


# -- the loud failures -----------------------------------------------------------------------------
#
# Defined before the constants because the pin checks below raise them.


class EstimatorDependencyMissing(ImportError):
    """Something this pair needs is absent or unusable. Fatal, never fallen back from.

    Subclasses :class:`ImportError` on purpose: ``measure_est_drift.resolve_estimators`` wraps
    ``importlib.import_module`` in ``except ImportError`` and re-raises as ``EstimatorUnavailable``
    with the text attached, so a missing package prints the whole diagnosis and exits 2 instead of
    dumping a traceback out of ``main``.

    Not every refusal raised here IS an import failure — missing weights and an unusable checkpoint
    are not — so those two have their own subclasses below. A caller that only wants "the module
    would not import" must catch :class:`EstimatorDependencyMissing` and re-raise anything whose
    type says otherwise, rather than reading every one of these as "not importable".
    """


class EstimatorWeightsMissing(EstimatorDependencyMissing):
    """A checkpoint is not in the local hub cache at its pinned revision, and fetching is refused."""


class EstimatorCheckpointUnusable(EstimatorDependencyMissing):
    """A checkpoint (or its pin) is present but cannot produce the quantity this module promises."""


class MaskValidityReferenceUndefined(RuntimeError):
    """The mask-validity filter has no reference for the object this process was told to segment.

    NOT an :class:`EstimatorDependencyMissing`, deliberately: nothing is absent, nothing failed to
    import, and no weight is missing. The models are fine and the frames are fine. What is absent is
    a SECOND OPINION about where the named object is, and the filter that PR-08 V6 put in the gate
    path cannot decide a frame without one.

    It is a refusal rather than a silently disabled filter (which would measure the label with no
    validity check at all, in an artifact whose committed contract says one ran) and rather than a
    per-frame all-False mask (which is what the module did before PR-08 V10 and is how this defect
    hid: 20 of 20 frames refused reads out of the harness as ``coverage: 0.0``, a fact about the
    corpus, when the true statement is that the filter is not defined for this label).
    """


# -- what this pair is, in the words the artifact will carry ---------------------------------------
#
# ``measure_est_drift`` records only ``name`` and ``version`` for the estimator pair, so the version
# string carries the three checkpoints, their revisions and the two detection thresholds. Anything
# not in here is invisible to whoever reads the artifact in six months, and "which SAM 2 was that?"
# is exactly the question that makes a committed gate number unusable.
#
# THE SIX LINES BELOW ARE THE SINGLE SOURCE OF TRUTH for the pins, and
# cluster/discoverer/102_stage_sam2_weights.sbatch reads them out of this file rather than deciding
# for itself what to stage (see the module docstring for the format that makes that possible).
# Keep each one a bare string literal on one line.

#: Cosmos-Transfer2.5's own segmenter checkpoints, copied from its ``sam2_model.py``, with the
#: commits they resolved to on the HF API on 2026-08-22.
SAM2_MODEL_ID_DEFAULT = "facebook/sam2-hiera-large"
SAM2_MODEL_REVISION_DEFAULT = "e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251"
GROUNDING_DINO_MODEL_ID_DEFAULT = "IDEA-Research/grounding-dino-base"
GROUNDING_DINO_MODEL_REVISION_DEFAULT = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"

#: METRIC, not relative — see the module docstring. depth-estimation, ungated, 2026-08-22.
DEPTH_MODEL_ID_DEFAULT = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
DEPTH_MODEL_REVISION_DEFAULT = "d2fc6a93601aabb1139a3bf0ebfcb4e89c67817f"

#: Overridable so a different tier can be measured, never so a local path can be hardcoded: an id
#: plus a commit is resolvable on any machine and reproducible in an artifact, a path under
#: someone's ``$HOME`` is neither. An id and its revision are ONE setting — overriding half of a
#: pair is refused below, because a new id at the old commit resolves to nothing and an old id at a
#: new commit is a silently different checkpoint.
SAM2_MODEL_CHECKPOINT = os.environ.get("WAM_PR08_SAM2_CHECKPOINT", SAM2_MODEL_ID_DEFAULT)
SAM2_MODEL_REVISION = os.environ.get("WAM_PR08_SAM2_REVISION", SAM2_MODEL_REVISION_DEFAULT)
GROUNDING_DINO_MODEL_CHECKPOINT = os.environ.get(
    "WAM_PR08_GROUNDING_DINO_CHECKPOINT", GROUNDING_DINO_MODEL_ID_DEFAULT
)
GROUNDING_DINO_MODEL_REVISION = os.environ.get(
    "WAM_PR08_GROUNDING_DINO_REVISION", GROUNDING_DINO_MODEL_REVISION_DEFAULT
)

#: Overriding this to a relative checkpoint is allowed and then :func:`estimate_depth` refuses,
#: loudly, rather than returning inverse disparity under a key called ``mean_m``.
DEPTH_MODEL_CHECKPOINT = os.environ.get("WAM_PR08_DEPTH_CHECKPOINT", DEPTH_MODEL_ID_DEFAULT)
DEPTH_MODEL_REVISION = os.environ.get("WAM_PR08_DEPTH_REVISION", DEPTH_MODEL_REVISION_DEFAULT)

#: The metric Depth-Anything-V2 heads, named so the refusal can point at one instead of saying "a
#: metric checkpoint". Not used to DECIDE whether the loaded model is metric — that is read off its
#: config — only to make the failure actionable.
METRIC_DEPTH_CHECKPOINT_SUGGESTIONS: tuple[str, ...] = (
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
)

#: GroundingDINO wants a lowercase, period-terminated phrase; that is its documented input format,
#: not a preference. "Apple" or "apple" (no period) parse differently in its text encoder and yield
#: fewer detections, which shows up as lower ``coverage`` and a p95 over the frames that happened to
#: survive — a quiet, plausible, wrong number. The normalisation is therefore applied and RECORDED
#: rather than assumed, and the raw value is kept beside it.
OBJECT_TEXT_PROMPT_RAW = os.environ.get("WAM_PR08_OBJECT_PROMPT", "apple.")


def normalize_prompt(prompt: str) -> str:
    """Lowercase and period-terminate, GroundingDINO's documented phrase format."""
    text = prompt.strip().lower()
    if not text:
        raise ValueError("WAM_PR08_OBJECT_PROMPT is empty — there is no object to segment.")
    return text if text.endswith(".") else text + "."


OBJECT_TEXT_PROMPT = normalize_prompt(OBJECT_TEXT_PROMPT_RAW)

# THESE FOUR NUMBERS ARE NOT OURS TO TUNE. THEY ARE THE GENERATOR'S, AND THAT IS THE POINT.
#
# They are read off Cosmos-Transfer2.5's own auxiliary segmenter,
# ``cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_model.py`` (the cluster copy, read
# 2026-08-22), which calls
#
#     processor.post_process_grounded_object_detection(..., threshold=0.15, text_threshold=0.25, ...)
#     if len(boxes) == 0:   # exactly one retry, at threshold=0.1, text_threshold=0.1
#     sorted_indices = np.argsort(scores)[::-1]   # the highest-scoring box wins
#
# The previous values here were 0.35/0.25 — GroundingDINO's demo defaults, which we had copied
# because they are what everyone copies. That made them OUR numbers, and a number of ours sitting in
# the detection path is exactly what PR-08 §4 step 2 forbids: it asks for **the same segmenter** the
# generator will use, so that ``EST_DRIFT_P95`` budgets the error the generator actually commits.
# At 0.35 we were budgeting for a stricter detector than the one that will draw the conditioning
# masks — a different no-detection rate, a different set of surviving frames, a different p95 — and
# calling the difference zero.
#
# So: do not "tune" these on AppleToPlate. A threshold that reads better on our corpus makes this
# adapter a different segmenter from the generator's, which does not improve the budget, it makes it
# a budget for something else. The only edit that is ever correct here is one that follows upstream
# after re-reading upstream, and it has to move :data:`SEGMENTER_CONTRACT` and the committed
# ``configs/transfer25/pr08_geom_tol.json`` with it or the cross-check will refuse the run.
BOX_THRESHOLD = float(os.environ.get("WAM_PR08_BOX_THRESHOLD", "0.15"))
TEXT_THRESHOLD = float(os.environ.get("WAM_PR08_TEXT_THRESHOLD", "0.25"))

#: Upstream's single retry when the first pass returns no box at all. It is a real behaviour of the
#: generator's segmenter and not a robustness flourish: it decides which frames the generator has a
#: mask for, and therefore which frames it constrains geometrically. Dropping it would leave us
#: measuring a stricter detector than the one that runs; adding a SECOND retry, or looping down, is
#: equally forbidden for the same reason. Exactly one, at exactly these two numbers.
RETRY_BOX_THRESHOLD = float(os.environ.get("WAM_PR08_RETRY_BOX_THRESHOLD", "0.1"))
RETRY_TEXT_THRESHOLD = float(os.environ.get("WAM_PR08_RETRY_TEXT_THRESHOLD", "0.1"))

#: Upstream takes ``np.argsort(scores)[::-1]`` and uses index 0 — the highest-scoring box. This
#: adapter already did that, for a reason of its own (see :func:`_best_box`: the largest box a text
#: prompt returns on a tabletop is routinely the plate-plus-apple region, and a box that CONTAINS
#: the apple is an ambiguous prompt for SAM 2). The two agreeing is worth stating rather than
#: leaving as a coincidence a later edit could break from either side.
BOX_SELECTION = "highest_score"

# -- THE VALIDITY CHECK ON THE OUTPUT, WHICH IS NOT A CHANGE TO THE DETECTION ----------------------
#
# WHAT IT IS FOR. Job 189637 drove this adapter over 382 frames of AppleToPlate (24 episodes) and a
# local CPU audit over 169 more, and both found the same defect: on twelve frames the detector
# returned a confident, well-formed box on THE PLATE, and SAM 2 dutifully segmented it. Those masks
# are ~30 900 px against a median apple of 6 185, they sit at 0.97-0.98 plate overlap, they score
# 0.167-0.309 where the correct masks score a median 0.829 — and they produce a centroid, a
# displacement and a p95 that all look exactly like measurements. That is the failure mode
# GATE_QUALIFICATION_BLOCKERS's first entry names in as many words. In episode_000094 the segmenter
# OSCILLATES between the two objects (f00149 plate -> f00150 apple, 471 px -> f00151 apple, 670 px
# -> f00152 plate), so the corruption hits both tails at once: near-zero displacements while it is
# locked on the stationary plate, and a recorded 245.9 px step at every switch.
#
# WHY IT IS NOT A SEGMENTER CHANGE, WHICH §4 STEP 2 WOULD FORBID. It alters no threshold, no prompt,
# no retry and no box rule: the detector runs at the generator's operating point, SAM 2 is prompted
# with exactly the box upstream's rule selects, and the mask drawn is bit for bit the mask that was
# drawn before. What changes is WHICH FRAMES WE ARE WILLING TO MEASURE ON — this adapter's callers
# already drop and count frames with no mask, and a refused frame joins them. §4 step 2 asks that
# GEOM_TOL and EST_DRIFT_P95 come from the same segmenter at the same operating point, and they
# still do, because the filter is applied identically on both sides of §6's subtraction by living
# HERE, in the one module both harnesses call. Do NOT "fix" the plate masks by raising
# BOX_THRESHOLD: that would be our detector rather than the generator's, and it would budget for an
# error nobody commits (see the comment above BOX_THRESHOLD).
#
# WHY THE EXACT THRESHOLD DOES NOT MATTER, WHICH IS THE ARGUMENT FOR ADMITTING IT AT ALL. Over the
# 382 audited frames the two populations do not overlap and are not close to overlapping: every
# correct mask scores IoU >= 0.7492 against the colour reference and every plate mask scores exactly
# 0.0000. Any cut in (0.0, 0.7492) produces the IDENTICAL partition of those frames, so this number
# is not a coined one — it is a value read off a gap. That claim is not left as prose:
# ``tests/test_apple_sam2_estimator.py`` sweeps the range against the audit's own recorded IoUs and
# fails if the partition ever moves. Registered as PR-08 V6 (T40_RULE_V6),
# docs/preregistration/PR-08-V6-mask-validity.md, before any gate number is measured with it.
#
# NO ENV OVERRIDE, DELIBERATELY. Every other knob here is overridable because upstream could move
# it. This one is ours, it decides which frames enter a committed measurement, and a knob that can
# silently change the measured population between two runs is precisely what SEGMENTER_CONTRACT
# exists to prevent — so it is a constant, it is IN that contract, and moving it means editing this
# file, the committed configs/transfer25/pr08_geom_tol.json and a pre-registration together.
MASK_VALIDITY_MIN_IOU = 0.10

#: The reference, named in the contract so an artifact says which second opinion was used. It is
#: NOT ground truth and is never treated as such — it is a non-learned opinion about where the fruit
#: is, which is exactly what makes a disagreement with SAM 2 informative. The same three numbers
#: ``build_identity_calibration.apple_mask`` uses, which is what ``probe-scan`` measured all 154 447
#: frames of this corpus with, so "the fruit is not visible here" means the same thing in the census
#: and in this refusal.
MASK_VALIDITY_REFERENCE = "warm_saturated_rgb(r>90, r-b>50, saturation>0.35)"

# -- WHERE THAT REFERENCE IS DEFINED, WHICH V6 DID NOT SAY AND THIS MODULE ASSUMED ------------------
#
# Registered as PR-08 V10 (T40_RULE_V10), docs/preregistration/PR-08-V10-mask-validity-reference-
# scope.md. UNSIGNED as this lands: nothing measured under it may be quoted until the project owner
# signs it. V6 is NOT edited, weakened or superseded — every threshold, counter and frame decision
# it registered still applies wherever its reference is defined. V10 only makes "wherever" explicit,
# because the module was treating a predicate about ONE object under ONE lighting as a predicate
# about any object under any lighting, and failing in two measured ways when it was not.
#
# DEFECT 1 — THE `plate.` PASS REFUSED 100 % OF FRAMES, ON THE SOURCE CORPUS. run_g0_gates documents
# §6's plate half as a second pass of the same script with WAM_PR08_OBJECT_PROMPT="plate.", which
# reaches segment() here. Measured on 20 source frames of episode_000000 (workstation GPU, this
# adapter unmodified): n_segment_calls 20, n_frames_mask_refused 20, validity IoUs all 0.0000,
# non-empty on 0 of 20 -- while the detector was doing its job perfectly, scoring 0.7524-0.7773 on
# every frame. The matched control on the SAME twenty frames with "apple." refuses nothing and
# scores 0.9686-0.9744. So this is not the corpus and not the detector: a correct plate mask
# contains no warm fruit pixels and scores ~0 against a warm-fruit reference, exactly as V6's own
# audit records (0.0000 on all twelve plate masks). The plate half of §6 could not be measured at
# all, and the failure presented as `coverage: 0.0` -- a fact about the corpus.
#
# DEFECT 2 — ON A RESTYLE THE REFERENCE DOES NOT GO QUIET, IT MOVES TO THE TABLE. V6 §5.3 anticipated
# the reference not firing on generated pixels, argued it fails CLOSED, and relies on
# MASK_REFUSED_NO_REFERENCE_FRAMES to separate "the segmenter is wrong on this corpus" from "the
# reference does not fit this corpus". Measured on job 189926's committed contact sheets -- the
# first restyled frames this project has -- on train-01-oak-tungsten (a bright green Granny Smith on
# warm oak under tungsten, its committed prompt) the predicate returns 40.5-56.4 % OF THE FRAME, all
# of it warm oak table, and about a fortieth of that on the fruit. So the reference is non-empty, the
# sub-case counter stays 0, and the twelve frames are recorded as "the mask was wrong" when the true
# statement is "the reference does not fit here". What was refused was a CORRECT mask: the detector
# put the same box [188,127,236,176] on the apple in both styles, SAM 2 drew ~1 830 px for it, and
# the same mask is KEPT on train-02-linen-overcast at IoU 0.46-0.49 and REFUSED on
# train-01-oak-tungsten at IoU 0.025-0.030. Nothing about the mask changed; the reference moved.
#
# WHAT V10 CHANGES, AND WHAT IT DELIBERATELY DOES NOT. It does not make the reference style-aware,
# and it does not make it a function of the paired source frame -- see the rule's §3 for why the
# second one is worse than the defect it would fix (under --g0b-percentile 100 it would refuse
# exactly the frames whose displacement IS the verdict). It states the reference's scope and refuses
# outside it. Fail closed, never open: a filter that cannot decide refuses loudly, and it never
# quietly accepts, and it never quietly turns itself off.

#: The object labels :func:`object_color_reference` is a reference FOR. Exactly one, because exactly
#: one has ever been measured: "the only saturated warm thing in any of these frames is the fruit"
#: is a claim about the fruit and says nothing about the plate, the cloth or the hand. A label that
#: is not in here has no second opinion available, so :func:`segment` refuses the RUN rather than
#: refusing every frame in it.
#:
#: ADDING A LABEL HERE IS NOT A CODE CHANGE, IT IS A PRE-REGISTRATION. A predicate for the plate
#: would be a colour discriminator for a neutral-white object on a cloth this module's own reference
#: docstring records as "neutral to within two counts" -- i.e. several numbers coined by us, in the
#: gate path, with no measured gap to read them off, which is exactly what PR-08 §4 step 2 and V6 §4
#: forbid. Whether §6's plate half is measured unfiltered instead is the project owner's call, and
#: it needs the committed contract to say so (this module cannot say it alone: `mask_validity_min_iou`
#: is cross-checked field for field against configs/transfer25/pr08_geom_tol.json, so an artifact
#: whose filter silently did not run would still claim it did).
MASK_VALIDITY_REFERENCE_LABELS: frozenset[str] = frozenset({"apple."})

#: The largest fraction of a frame the reference may cover and still be a reference to THE OBJECT.
#: Above it the predicate has latched onto the background and its own stated corpus assumption is
#: false on this frame, so the filter cannot decide the frame and refuses it -- counted apart, in
#: :data:`MASK_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES`, which is the counter V6 §5.3 needed and
#: did not have.
#:
#: A VALUE READ OFF A GAP, NOT A COINED ONE, and the gap is wide under both instruments that have
#: measured it (the rule's §4 carries the whole table):
#:
#:   SOURCE, full resolution, this module's own predicate
#:     17 307 frames -- every frame of 40 episodes, 2026-08-23, local  max 3.00 %
#:     382-frame committed audit, job 189637 (warm_apple_px)           max 2.90 %
#:     169-frame local CPU audit                                       max 2.80 %
#:     154 447-frame census, 362 episodes (per-episode medians)        max 2.52 %
#:   RESTYLE, via job 189926's contact sheets (half resolution; the sheet path inflates this
#:   predicate by a measured 1.93-1.94x, established on the SOURCE half of the same sheets)
#:     train-02-linen-overcast, filter working correctly    4.27-4.39 %  (2.20-2.26 % deflated)
#:     train-01-oak-tungsten, filter mis-firing            40.5 -56.4 %  (20.9 -29.1 % deflated)
#:
#: So the gap is (4.39 %, 40.5 %) on the raw sheet scale and (3.00 %, 20.9 %) deflated, and 0.10 is
#: the one round value inside BOTH -- which is the point, because the deflation factor is measured on
#: warm-red fruit pixels and its transfer to warm-oak pixels is not established. The bound does not
#: depend on it. The numeral coinciding with :data:`MASK_VALIDITY_MIN_IOU` is a coincidence: that one
#: is an IoU between two masks, this one is a fraction of a frame, and they are never compared.
#:
#: NO ENV OVERRIDE, for MASK_VALIDITY_MIN_IOU's reason exactly: it decides which frames enter a
#: committed measurement, and a knob that can silently change the measured population between two
#: runs is what the segmenter contract exists to prevent.
MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION = 0.10

#: THE ONE REMAINING DIFFERENCE FROM UPSTREAM, named so it cannot be mistaken for an equivalence.
#: Upstream drives ``SAM2VideoPredictor.from_pretrained(...).init_state(video_path=...)`` and
#: PROPAGATES one mask through the clip; this adapter segments each frame independently, because the
#: contract both PR-08 §4 harnesses call is ``segment(rgb) -> mask`` on one frame. See
#: :data:`GATE_QUALIFICATION_BLOCKERS` for the argument about which way that biases the budget — it
#: is not one way.
PROPAGATION = "per_frame"
UPSTREAM_PROPAGATION = "sam2_video_predictor"

#: The pixel grid both halves of §6's subtraction must be denominated in: AppleToPlate's source
#: frames at 640x480, written ``[height, width]`` because that is the order ``resolution_hw`` uses
#: on both sides. This is a claim about the CORPUS recorded in the contract so the two rigs can be
#: joined on it before either runs; it is not enforcement. The enforcement is
#: ``measure_est_drift.cross_check_geom_tol`` comparing the committed grid against the capture's
#: actual one, and ``measure_geom_tol`` refusing a corpus with mixed geometry.
PIXEL_GRID_HW: tuple[int, int] = (480, 640)

#: ``cuda`` when torch says so, resolved at first load rather than at import so that importing this
#: module never touches a GPU. The device changes nothing about the numbers and everything about
#: whether the run finishes this week.
DEVICE_OVERRIDE = os.environ.get("WAM_PR08_DEVICE") or None

#: Downloads are the project owner's call (T-040: staging these is a ~3 GB fetch). Unset means the
#: weights must already be cached — and, since this is also what switches offline enforcement on,
#: means no hub request may leave the machine at all.
ALLOW_DOWNLOAD = os.environ.get("WAM_PR08_ALLOW_DOWNLOAD", "") == "1"

ESTIMATOR_NAME = "grounding-dino+sam2+depth-anything-v2"
ESTIMATOR_VERSION = (
    f"det={GROUNDING_DINO_MODEL_CHECKPOINT}@{GROUNDING_DINO_MODEL_REVISION};"
    f"seg={SAM2_MODEL_CHECKPOINT}@{SAM2_MODEL_REVISION};"
    f"depth={DEPTH_MODEL_CHECKPOINT}@{DEPTH_MODEL_REVISION};"
    f"prompt={OBJECT_TEXT_PROMPT!r};"
    f"box_thr={BOX_THRESHOLD};text_thr={TEXT_THRESHOLD};"
    f"retry_box_thr={RETRY_BOX_THRESHOLD};retry_text_thr={RETRY_TEXT_THRESHOLD};"
    f"box_sel={BOX_SELECTION};prop={PROPAGATION};"
    # In the version string because it decides which frames a recorded number was measured on, and
    # an artifact whose version cannot answer "was the mask-validity filter on?" cannot be compared
    # against one measured before it existed.
    f"mask_val_min_iou={MASK_VALIDITY_MIN_IOU};"
    # PR-08 V10, and here for V6's reason: it decides which frames a recorded number was measured
    # on. run_g0_gates.instrument_disagreements compares this string BETWEEN THE TWO SIDES of G0b,
    # so a source record dumped before this token existed and a restyled record dumped after it
    # refuse to be compared -- which is correct, they are two instruments, and it is the reason the
    # token is here rather than only in stats(). Nothing committed carries ESTIMATOR_VERSION, so no
    # cross-check against configs/transfer25/pr08_geom_tol.json moves.
    f"mask_val_ref_max_frac={MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION}"
)

#: The join key PR-08 §4 step 2 is checked on. It is :data:`ESTIMATOR_NAME` and it is spelled once:
#: ``measure_geom_tol`` stamps it into ``mask_method.name``, ``measure_est_drift`` into
#: ``estimators.name``, and the committed contract below into ``segmenter.method_name``. Three
#: recordings of one string, so equality is by construction rather than by anyone remembering.
MASK_METHOD_NAME = ESTIMATOR_NAME

#: THE COMMITTED SEGMENTER CONTRACT, in the shape ``configs/transfer25/pr08_geom_tol.json`` carries
#: it under ``segmenter``.
#:
#: PR-08 §4 step 2 says GEOM_TOL and EST_DRIFT_P95 must come from "the same segmenter", and §6
#: subtracts them. Before this dict existed the only thing either artifact recorded about the
#: segmenter was a NAME, so two runs could share a name and disagree about the prompt, the
#: thresholds, the retry and the box rule — every one of which changes which frames get a mask and
#: where its centroid lands — and the subtraction would still look like arithmetic.
#:
#: It is committed BEFORE the measurement, which is the other half of its job. A mask method chosen
#: (or quietly adjusted) after seeing GEOM_TOL is the failure the style partition is committed early
#: to prevent, and "we used the same segmenter" is not a checkable claim unless the claim was
#: written down first. ``measure_est_drift.cross_check_geom_tol`` compares this dict field for field
#: against the committed file and DISQUALIFIES the run on any disagreement.
SEGMENTER_CONTRACT: dict[str, Any] = {
    "method_name": MASK_METHOD_NAME,
    "detector": {
        "repo": GROUNDING_DINO_MODEL_CHECKPOINT,
        "revision": GROUNDING_DINO_MODEL_REVISION,
    },
    "segmenter": {"repo": SAM2_MODEL_CHECKPOINT, "revision": SAM2_MODEL_REVISION},
    "depth": {"repo": DEPTH_MODEL_CHECKPOINT, "revision": DEPTH_MODEL_REVISION},
    "object_text_prompt": OBJECT_TEXT_PROMPT,
    "box_threshold": BOX_THRESHOLD,
    "text_threshold": TEXT_THRESHOLD,
    "retry_box_threshold": RETRY_BOX_THRESHOLD,
    "retry_text_threshold": RETRY_TEXT_THRESHOLD,
    "box_selection": BOX_SELECTION,
    # NOT a detection parameter — a validity check on the output, and therefore a statement about
    # WHICH FRAMES were measured rather than about how a mask was drawn. It is in the contract for
    # exactly the reason the thresholds are: GEOM_TOL and EST_DRIFT_P95 are subtracted in §6, and a
    # GEOM_TOL measured with the filter minus an EST_DRIFT_P95 measured without it is a subtraction
    # over two different frame populations that would still look like arithmetic. Absence counts as
    # a disagreement in contract_disagreements(), so an artifact from before PR-08 V6 now fails the
    # cross-check instead of pooling silently.
    "mask_validity_min_iou": MASK_VALIDITY_MIN_IOU,
    "mask_validity_reference": MASK_VALIDITY_REFERENCE,
    "propagation": PROPAGATION,
    "upstream_propagation": UPSTREAM_PROPAGATION,
    "pixel_grid_hw": list(PIXEL_GRID_HW),
}

#: Every condition that has to be true before this pair may set ``EST_DRIFT_P95`` or ``GEOM_TOL``.
#: Written out rather than summarised because :data:`GATE_QUALIFIED` is a claim, and a claim whose
#: grounds are not written down gets flipped by whoever is in a hurry.
GATE_QUALIFICATION_BLOCKERS: tuple[str, ...] = (
    "NOBODY HAS LOOKED AT A MASK. The 2026-08-21 wording of this blocker ('never executed, no "
    "checkpoint staged') is withdrawn as stale: job 189583 staged all three checkpoints at the "
    "pinned revisions and verified them, and job 189588 drove this adapter end to end over the "
    "AppleToPlate corpus in the GEOM_TOL pilot — 720 frames, two passes, 480x640, coverage 1.0 on "
    "both. CITATION CAVEAT, because a blocker tuple is the load-bearing record of what is and is "
    "not established: 189583 is recorded in .mc/tasks/todo/T-040-*.md, but 189588 IS NOT RECORDED "
    "ANYWHERE TRACKED IN THIS REPOSITORY — its artifact was not readable from the session that "
    "wrote this line, and the job id is an untracked claim until GEOM_TOL_PILOT.json lands. It is "
    "also evidence about a configuration THIS FILE HAS SINCE REPLACED: that pilot necessarily ran "
    "at the old operating point (box_threshold 0.35, no retry branch), so it is weaker evidence "
    "for the current adapter than its numbers suggest. So the module runs and produces output. "
    "What that does NOT establish is that the output "
    "is right: coverage 1.0 says a box was returned on every frame, not that it was the APPLE's "
    "box, and this adapter's whole failure mode is a plausible mask on the wrong object (the "
    "plate, the hand, the whole tabletop) which produces a centroid, a displacement and a p95 that "
    "all look like measurements. Lowering BOX_THRESHOLD to upstream's 0.15 with a 0.10 retry — "
    "correct, and required by §4 step 2 — makes coverage an even weaker witness than it was at "
    "0.35, because more frames now get a box and none of them get checked. Discharged by: a human "
    "looking at a sample of overlaid masks spanning the corpus (occluded frames, apple-out-of-frame "
    "frames, and the grasp), and/or a mask-vs-ground-truth IoU distribution from the Isaac capture "
    "recorded beside the centroid displacement. Neither exists.",
    "BOX_THRESHOLD / TEXT_THRESHOLD / the retry are unmeasured on AppleToPlate, and after 2026-08-22 "
    "that is a narrower objection than it was. They are no longer 'upstream demo defaults we "
    "happened to copy' (0.35/0.25): they are Cosmos-Transfer2.5's own operating point, read off its "
    "sam2_model.py, which is precisely what §4 step 2 asks for. The choice-defect half of this "
    "blocker is therefore DISCHARGED and inverted — measuring these on our corpus and moving them "
    "to whatever reads best would MAKE this a different segmenter from the generator's, and the "
    "budget would then be a budget for an error nobody commits. What survives is not a choice, it "
    "is an unknown: nothing has measured what this operating point does on THIS corpus, and the "
    "retry at (0.10, 0.10) buys detections by accepting weak ones, which on an occluded frame can "
    "replace an honest all-False mask with a confident box on the wrong object. That inflates "
    "coverage while degrading the mask, i.e. it hides itself in the one number the harness gates "
    "on. Discharged by the same evidence as blocker 1, plus the recorded detection-score "
    "distribution and retry counts (n_frames_retry_fired / n_frames_retry_recovered) from a full "
    "pass, so the retry's contribution is visible rather than assumed.",
    "PER-FRAME SEGMENTATION IS NOT UPSTREAM'S PROPAGATION, and it is the one difference left. "
    "Everything else in §4 step 2's 'the same segmenter' now matches Cosmos-Transfer2.5's "
    "sam2_model.py exactly — both checkpoints at pinned revisions, the 'apple.' phrase, "
    "threshold=0.15 / text_threshold=0.25, the single (0.10, 0.10) retry when no box is found, and "
    "highest-score box selection. But upstream drives SAM2VideoPredictor.init_state(video_path=...) "
    "and PROPAGATES one mask across the clip, while this adapter re-detects and re-segments every "
    "frame independently, because segment(rgb) is the contract both harnesses call. The bias is "
    "TWO-SIDED, which is why this cannot be waved through as conservative: (a) independent "
    "re-detection jitters frame to frame where propagation is temporally smooth, so our tail — and "
    "EST_DRIFT_P95 is a p95, i.e. the tail — is INFLATED relative to the generator's, which "
    "subtracts more from GEOM_TOL and tightens G0b (safe); (b) propagation's own characteristic "
    "failure, drifting off the object and staying off for a run of frames, is invisible to a "
    "per-frame estimator that recovers on the next frame, so the generator commits an error our "
    "budget never sees (unsafe). PR-08 §4 already stamps is_lower_bound: true for a different "
    "reason; with (a) and (b) together this number is neither a lower nor an upper bound on the "
    "generator's mask error, and a G0b margin that clears only under it is not a pass. Discharged "
    "by: measuring the same Isaac capture BOTH ways — this adapter per frame, and the video "
    "predictor propagating from frame 0 — and recording the two p95s, so the direction and size of "
    "the difference are a measurement rather than the argument above.",
)

#: What used to be in the tuple above and is not any more, with the evidence that removed it. A
#: blocker that simply DISAPPEARS between two commits is indistinguishable from a blocker somebody
#: deleted because it was in the way, and the whole value of the tuple is that it cannot be reduced
#: quietly. Kept in ``stats()`` too, so the artifact carries the shrinking as well as the remainder.
GATE_QUALIFICATION_DISCHARGED: tuple[str, ...] = (
    "2026-08-22 — 'never executed, no checkpoint staged anywhere in this project' (T-040, "
    "2026-08-21). Withdrawn by measurement: job 189583 staged facebook/sam2-hiera-large, "
    "IDEA-Research/grounding-dino-base and depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf "
    "at the revisions pinned in this file and verified them (PR08_ESTIMATORS_STAGED.json, 5.0 GB); "
    "job 189588 ran this adapter over the AppleToPlate corpus for 720 frames in two passes at "
    "480x640. NOT replaced by 'and the output is correct' — see blocker 1, which is what is left of "
    "it.",
    "2026-08-22 — 'Cosmos-Transfer2.5's prompt text, thresholds and mask-selection rule were NOT "
    "read, so the same segmenter means the same weights driven our way'. Discharged by reading "
    "cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_model.py on the cluster and adopting its "
    "operating point verbatim: threshold=0.15, text_threshold=0.25, one retry at (0.10, 0.10) when "
    "no box is found, highest-scoring box wins. Partially: the propagation difference it also "
    "exposed is now a blocker in its own right.",
    "2026-08-22 — 'the committed contract can be OVERWRITTEN by the measurement it constrains'. "
    "configs/transfer25/pr08_geom_tol.json is measure_geom_tol.py's default --out and its --merge "
    "target, so the first real GEOM_TOL run replaced the pre-commitment with a document that "
    "mentioned no segmenter anywhere; the cross-check then reported "
    "geom_tol_does_not_record_segmenter_params on every later run — failing closed, and closed "
    "forever. Discharged in measure_geom_tol.py, on both write paths, exactly as this blocker "
    "specified: sam2_method() records SEGMENTER_CONTRACT into mask_method.params.segmenter (where "
    "the cross-check already looks), merge_committed_contract() compares the block already at "
    "--out field for field against the adapter this run drove and REFUSES the whole run — exit 2, "
    "nothing written — on any disagreement, then copies the contract section forward verbatim, and "
    "refuse_default_out_without_contract() refuses to write the tracked path at all when no "
    "contract is sitting in it. The file is one document in two declared sections "
    "(contract_fields / measurement_fields) rather than two files, because measure_est_drift, "
    "run_g0_gates and 102_stage_sam2_weights.sbatch all resolve the tolerance AND its segmenter "
    "through that single path.",
    "2026-08-22 — 'the object segmented and the object scored against are set independently "
    "($WAM_PR08_OBJECT_PROMPT here, measure_est_drift --object-class there), so a mismatch produces "
    "a large plausible p95 rather than a refusal'. Discharged in the harness, which is where it "
    "belonged: measure_est_drift's --object-class now DEFAULTS to this module's own "
    "OBJECT_TEXT_PROMPT and an explicit value that names a different object is FATAL (exit 2, "
    "nothing written) instead of being measured. measure_geom_tol has no such flag at all — its "
    "sam2 method takes the prompt from this module — so there is no longer a second place where the "
    "object is chosen. This module still cannot see the flag; it no longer has to.",
)

#: Opt-IN, and this module still does not opt in. THREE conditions above are open, and the two that
#: matter most are cheap to state: nobody has looked at a mask this adapter produced (the FIRST
#: blocker), and it is not yet the same segmenter the generator runs — it re-detects per frame where
#: Transfer2.5 propagates (the LAST). Counted rather than indexed on purpose: the tuple shrank on
#: 2026-08-22 when the committed-contract blocker was discharged, and a comment that said "blocker
#: 4" went on pointing at whatever had moved into that slot. :data:`GATE_QUALIFICATION_DISCHARGED`
#: now carries four conditions closed by measurement rather than by deletion, and one of the three
#: that remain is inverted — which is progress and is not permission. ``measure_est_drift`` reads this flag with a default of
#: False and stamps ``estimator_not_gate_qualified``; the artifact is still written, and exits 3.
GATE_QUALIFIED = False


# -- the pins are checked at import, not at first load ----------------------------------------------


class Checkpoint(NamedTuple):
    """One repo, pinned. ``what`` and ``approx_size`` exist to make the refusals actionable."""

    repo_id: str
    revision: str
    what: str
    approx_size: str


#: Every checkpoint this pair loads. One list so that :func:`available`, :func:`_require_cached` and
#: the refusals cannot disagree about what "the weights" means.
CHECKPOINTS: tuple[Checkpoint, ...] = (
    Checkpoint(
        GROUNDING_DINO_MODEL_CHECKPOINT,
        GROUNDING_DINO_MODEL_REVISION,
        "GroundingDINO detector",
        "~700 MB",
    ),
    Checkpoint(SAM2_MODEL_CHECKPOINT, SAM2_MODEL_REVISION, "SAM 2 segmenter", "~900 MB"),
    Checkpoint(
        DEPTH_MODEL_CHECKPOINT,
        DEPTH_MODEL_REVISION,
        "Depth-Anything-V2 depth estimator",
        "~1.3 GB",
    ),
)

#: The same three, in the shape ``measure_geom_tol._adapter_checkpoints`` reads: it records what an
#: adapter DECLARES it loads, and "an estimator is its weights as much as its code". ``id@commit``
#: rather than the bare id, so GEOM_TOL's provenance and EST_DRIFT_P95's ``estimator.version`` name
#: the same weights and can be joined to the staging manifest.
ESTIMATOR_CHECKPOINTS: dict[str, str] = {
    "detector": f"{CHECKPOINTS[0].repo_id}@{CHECKPOINTS[0].revision}",
    "segmenter": f"{CHECKPOINTS[1].repo_id}@{CHECKPOINTS[1].revision}",
    "depth": f"{CHECKPOINTS[2].repo_id}@{CHECKPOINTS[2].revision}",
}

#: ``(id env var, revision env var, checkpoint)`` — the pairs that have to be overridden together.
_PIN_ENV_PAIRS: tuple[tuple[str, str, Checkpoint], ...] = (
    ("WAM_PR08_GROUNDING_DINO_CHECKPOINT", "WAM_PR08_GROUNDING_DINO_REVISION", CHECKPOINTS[0]),
    ("WAM_PR08_SAM2_CHECKPOINT", "WAM_PR08_SAM2_REVISION", CHECKPOINTS[1]),
    ("WAM_PR08_DEPTH_CHECKPOINT", "WAM_PR08_DEPTH_REVISION", CHECKPOINTS[2]),
)

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _check_pins() -> None:
    """Refuse a moving pointer, and refuse half an override. Same rule as the staging job's.

    ``main``/``HEAD``/a tag names whatever upstream pushed last, which is a different answer on a
    different day: recording it in ``ESTIMATOR_VERSION`` records nothing. And an id overridden
    without its revision (or the reverse) is worse than either alone — the id and the commit stop
    describing the same weights, and the artifact would name a pair that never existed.
    """
    for id_var, rev_var, ckpt in _PIN_ENV_PAIRS:
        id_set, rev_set = id_var in os.environ, rev_var in os.environ
        if id_set != rev_set:
            given, missing = (id_var, rev_var) if id_set else (rev_var, id_var)
            raise EstimatorCheckpointUnusable(
                "\n".join([
                    f"FATAL: {given} is set but {missing} is not, so the {ckpt.what}'s id and its",
                    "       commit no longer describe the same weights. A new repo at the old",
                    "       commit resolves to nothing; the old repo at a new commit is a silently",
                    "       different checkpoint. Set both, or neither. Nothing was written.",
                    "",
                    f"       {id_var}  = {ckpt.repo_id}",
                    f"       {rev_var} = {ckpt.revision}",
                ])
            )
        if not _COMMIT_SHA.match(ckpt.revision):
            raise EstimatorCheckpointUnusable(
                "\n".join([
                    f"FATAL: {rev_var}={ckpt.revision!r} is not a 40-hex commit sha. A branch, a tag",
                    "       or an empty string names whatever upstream pushed last, which is a",
                    "       different answer on a different day, and ESTIMATOR_VERSION would then",
                    "       identify no particular weights. AC-04 asks the opposite. Read a commit",
                    "       off the HF API and pin it:",
                    "",
                    f"         curl -s https://huggingface.co/api/models/{ckpt.repo_id} \\",
                    "           | python3 -c 'import json,sys;print(json.load(sys.stdin)[\"sha\"])'",
                ])
            )


# -- observed state, for the caller that wants to record it ----------------------------------------
#
# Filled in at load and during use. Every counter here is CUMULATIVE OVER THE LIFETIME OF THE
# IMPORT, not per run: nothing resets them, and two measurements driven from one interpreter share
# them. That is deliberate — a module that reset its own counters would give a caller no way to
# distinguish "this run detected nothing" from "somebody reset them mid-run" — and it is the reason
# ``measure_geom_tol.EstimatorStatsProbe`` snapshots them before the pass and DIFFERENCES afterwards
# rather than reading them straight into an artifact. A caller that wants per-run numbers must do
# the same; a caller that copies these values verbatim is recording a total, and should say so.
#
# Since 2026-08-22 both harnesses do record them (``estimator_stats`` in
# ``configs/transfer25/pr08_geom_tol.json`` and in ``pr08_est_drift.json``), which is where the
# full-pass half of ``GATE_QUALIFICATION_BLOCKERS``'s second entry lands. Recording it is not
# discharging it: the blocker asks for the numbers AND for somebody to read them.

#: ``"metric"``, ``"relative"``, or None before the depth model has been loaded. Read off the loaded
#: config; absent from the config is treated as ``"relative"``, because that is what transformers'
#: ``DepthAnythingConfig`` defaults to and because an unstated claim is not a claim.
DEPTH_ESTIMATION_TYPE: str | None = None
DEPTH_IS_METRIC: bool | None = None
DEPTH_MAX_DEPTH_M: float | None = None

SEGMENT_CALLS = 0
NO_DETECTION_FRAMES = 0
EMPTY_MASK_FRAMES = 0

#: Frames on which a non-empty mask was drawn and then REFUSED by the mask-validity check, and of
#: those, the ones where the colour reference found no fruit at all in the frame. Three events that
#: all end in an all-False mask and MUST NOT be collapsed into one number:
#:
#:   ``NO_DETECTION_FRAMES``          the detector found no box at either threshold.
#:   ``EMPTY_MASK_FRAMES``            a box was found and SAM 2 returned nothing inside it.
#:   ``MASK_REFUSED_FRAMES``          a box was found, SAM 2 filled it, and what it filled was not
#:                                    the object — the plate, the hand, the tabletop.
#:
#: The split of the third one matters as much as the third one existing. ``*_NO_REFERENCE_FRAMES``
#: is the sub-case where the colour reference itself is empty, i.e. THE FRUIT IS NOT VISIBLE AT ALL
#: — a genuinely hard frame, refused because nothing here can confirm the mask, not because the mask
#: was demonstrably wrong. That is the threat to validity PR-08 V6 records: those refusals remove
#: hard frames from the measured population, and for a p95 that gets SUBTRACTED from GEOM_TOL that
#: plausibly errs in the generator's favour. It is counted so the size of the effect is a number
#: rather than a worry — and so that a pass over a corpus this colour predicate does not fit (an
#: Isaac render, a restyled clip) announces itself as "the reference found nothing on N frames"
#: instead of as a low coverage nobody can explain.
MASK_REFUSED_FRAMES = 0
MASK_REFUSED_NO_REFERENCE_FRAMES = 0

#: PR-08 V10. The OTHER sub-case of "the reference does not fit here", and the one V6 §5.3 assumed
#: could not happen: the reference is not empty, it is enormous — the predicate has latched onto the
#: background, so it describes the scene rather than the object and cannot arbitrate any mask on this
#: frame. Counted INSIDE ``MASK_REFUSED_FRAMES`` and not beside it, exactly as
#: ``MASK_REFUSED_NO_REFERENCE_FRAMES`` is, so that the identity ``n_frames_without_detection +
#: n_frames_with_empty_mask + n_frames_mask_refused`` still spans the whole coverage shortfall this
#: module is responsible for. A consumer that wants the three-way split reads the two sub-cases and
#: subtracts.
#:
#: THE TWO SUB-CASES ARE NOT SYMMETRIC AND MUST NOT BE POOLED. An empty reference on the SOURCE
#: corpus means the fruit is occluded or out of frame — a hard frame, refused because nothing can
#: confirm the mask, which is V6 §5.1's recorded bias. An empty reference on a restyle whose apple is
#: not warm means the reference does not fit, which is a different finding with a different fix. This
#: module cannot tell those two apart from one frame and does not pretend to: see
#: stats()['mask_validity_reference_scope'].
MASK_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES = 0

#: The validity IoU of every frame the check RAN on, in call order — the same design, for the same
#: reasons, as :data:`DETECTION_SCORES`: raw values pool exactly through JSON where a histogram only
#: pools if it was binned identically, and a distribution recorded as a digest cannot answer the
#: question nobody has asked yet. The check runs on frames that got a box AND a non-empty mask, so
#: ``len(MASK_VALIDITY_IOU) == SEGMENT_CALLS - NO_DETECTION_FRAMES - EMPTY_MASK_FRAMES``, and it is
#: not index-aligned to frames. Cumulative like every counter here.
#:
#: It is what makes the refusal PER-FRAME evidence rather than a tally: an audit rig that segments
#: one frame at a time reads the value this frame appended and can show the filter fired on exactly
#: the frames a person flagged. ``scripts/audit_apple_masks.py`` does precisely that.
MASK_VALIDITY_IOU: list[float] = []

#: PR-08 V10. The fraction of the frame the colour reference covered, one per frame the check ran on,
#: in call order and index-aligned to :data:`MASK_VALIDITY_IOU` (both are appended in the same
#: branch, before either decides anything). Raw values for :data:`DETECTION_SCORES`'s reasons.
#:
#: It is the evidence that makes :data:`MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION` checkable in any
#: artifact this adapter writes, rather than a number a reader has to take on trust. The condition
#: that would make the bound matter — a corpus, a restyle or a render on which the applicable and
#: inapplicable populations are NOT separated by a gap — is a distribution with mass between 0.03 and
#: 0.21, and it is visible here the moment it occurs. If it ever does, the correct response is a
#: further version alongside V10, not a bound moved inside it.
MASK_VALIDITY_REFERENCE_FRACTION: list[float] = []

#: Frames where the first pass found nothing and upstream's single (0.10, 0.10) retry fired, and of
#: those, the ones where it produced a box. The gap between them and ``NO_DETECTION_FRAMES`` is the
#: only visible trace of how much of this run's ``coverage`` was bought at the lower threshold.
RETRY_FRAMES = 0
RETRY_RECOVERED_FRAMES = 0

#: The detection score of the box that WON, one entry per frame where a box was found at all, in
#: call order. Appended by :func:`_best_box`; frames with no detection append NOTHING, so this list
#: is shorter than ``SEGMENT_CALLS`` by exactly ``NO_DETECTION_FRAMES`` and is not index-aligned to
#: the frames — ``n_frames_without_detection`` is where those live.
#:
#: WHY THE RAW VALUES AND NOT A HISTOGRAM. Two reasons, both about the merge.
#: ``measure_geom_tol`` runs as an 8-way array and pools its shards; raw values pool exactly through
#: JSON (``float`` -> ``repr`` -> ``float`` is the identity) while two histograms only pool if they
#: were binned identically, which is the same argument that makes the shards emit raw
#: displacements. And a distribution recorded as a digest cannot answer a question nobody asked
#: yet, which for this list is the whole point: it is the evidence
#: ``GATE_QUALIFICATION_BLOCKERS``'s second entry asks for, and the question it is meant to answer
#: — how much of ``coverage`` was bought at the retry's lower threshold — is READ OFF THE VALUES.
#: A score below :data:`BOX_THRESHOLD` can only have come from the ``(0.10, 0.10)`` retry, because
#: the first pass discards everything under it; so ``[s for s in DETECTION_SCORES if s <
#: BOX_THRESHOLD]`` is exactly the retry's contribution, and it is a measurement rather than the
#: assumption the blocker objects to. Cumulative like the counters above, and snapshotted the same
#: way.
#:
#: Cheap: a full GEOM_TOL pass is ~171 600 frames, i.e. ~1.4 MB of float in memory and ~215 kB of
#: JSON per shard, beside the ~430 kB of displacements a shard already carries.
DETECTION_SCORES: list[float] = []


def stats() -> dict[str, Any]:
    """What this pair did, in a shape a caller can drop into an artifact verbatim."""
    return {
        "estimator_name": ESTIMATOR_NAME,
        "estimator_version": ESTIMATOR_VERSION,
        "gate_qualified": GATE_QUALIFIED,
        "gate_qualification_blockers": list(GATE_QUALIFICATION_BLOCKERS),
        "gate_qualification_discharged": list(GATE_QUALIFICATION_DISCHARGED),
        "segmenter_contract": dict(SEGMENTER_CONTRACT),
        "detector_checkpoint": GROUNDING_DINO_MODEL_CHECKPOINT,
        "detector_revision": GROUNDING_DINO_MODEL_REVISION,
        "segmenter_checkpoint": SAM2_MODEL_CHECKPOINT,
        "segmenter_revision": SAM2_MODEL_REVISION,
        "depth_checkpoint": DEPTH_MODEL_CHECKPOINT,
        "depth_revision": DEPTH_MODEL_REVISION,
        "object_text_prompt": OBJECT_TEXT_PROMPT,
        "object_text_prompt_raw": OBJECT_TEXT_PROMPT_RAW,
        "object_text_prompt_note": (
            "measure_est_drift's --object-class now defaults to this string and refuses an explicit "
            "value that names a different object, so the two knobs can no longer disagree "
            "silently; measure_geom_tol has no such flag and takes the prompt from here. Recorded "
            "anyway, because a reader of the artifact checks the claim rather than trusting it."
        ),
        "box_threshold": BOX_THRESHOLD,
        "text_threshold": TEXT_THRESHOLD,
        "retry_box_threshold": RETRY_BOX_THRESHOLD,
        "retry_text_threshold": RETRY_TEXT_THRESHOLD,
        "box_selection": BOX_SELECTION,
        "mask_validity_min_iou": MASK_VALIDITY_MIN_IOU,
        "mask_validity_reference": MASK_VALIDITY_REFERENCE,
        "mask_validity_reference_labels": sorted(MASK_VALIDITY_REFERENCE_LABELS),
        "mask_validity_reference_max_frame_fraction": MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION,
        "mask_validity_reference_is_defined_for_this_prompt": mask_validity_reference_is_defined(),
        "propagation": PROPAGATION,
        "upstream_propagation": UPSTREAM_PROPAGATION,
        "detection_params_source": (
            "cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_model.py — the generator's own "
            "segmenter, read on the cluster 2026-08-22. These are NOT tuned for this corpus and "
            "must not be: PR-08 §4 step 2 asks for the generator's operating point, not a better "
            "one."
        ),
        "depth_estimation_type": DEPTH_ESTIMATION_TYPE,
        "depth_is_metric": DEPTH_IS_METRIC,
        "depth_max_depth_m": DEPTH_MAX_DEPTH_M,
        "downloads_permitted": ALLOW_DOWNLOAD,
        "n_segment_calls": SEGMENT_CALLS,
        "n_frames_without_detection": NO_DETECTION_FRAMES,
        "n_frames_with_empty_mask": EMPTY_MASK_FRAMES,
        "n_frames_retry_fired": RETRY_FRAMES,
        "n_frames_retry_recovered": RETRY_RECOVERED_FRAMES,
        "n_frames_mask_refused": MASK_REFUSED_FRAMES,
        "n_frames_mask_refused_no_reference": MASK_REFUSED_NO_REFERENCE_FRAMES,
        "n_frames_mask_refused_reference_not_object_scale": (
            MASK_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES
        ),
        "n_mask_validity_iou": len(MASK_VALIDITY_IOU),
        "n_mask_validity_reference_fraction": len(MASK_VALIDITY_REFERENCE_FRACTION),
        "n_detection_scores": len(DETECTION_SCORES),
        "detection_scores_attr": "DETECTION_SCORES",
        "detection_scores_meaning": (
            "the module attribute named by detection_scores_attr holds the winning box's score "
            "for every frame where a box was found, in call order; frames with no detection are "
            "absent from it, so len(DETECTION_SCORES) == n_segment_calls - "
            "n_frames_without_detection. The VALUES are not recorded here — a full pass is ~171 600 "
            "of them and stats() is embedded verbatim in several artifacts — they are read off the "
            "attribute by measure_geom_tol/measure_est_drift, which pool and bin them. A score "
            "below box_threshold can only have come from the (retry_box_threshold, "
            "retry_text_threshold) pass, so the part of the distribution under box_threshold IS "
            "the retry's contribution, measured rather than assumed."
        ),
        "counters_are_cumulative": (
            "every n_* count above, and n_detection_scores, is a total since this module was "
            "imported. Nothing resets them, so two measurements driven from one interpreter share "
            "them. A caller wanting THIS RUN's numbers snapshots them before the pass and "
            "differences afterwards, as measure_geom_tol.EstimatorStatsProbe does; a caller that "
            "copies them verbatim is recording a lifetime total and should say so in its artifact."
        ),
        "retry_meaning": (
            "frames where the first pass at (box_threshold, text_threshold) found no box and "
            "upstream's single (retry_box_threshold, retry_text_threshold) pass ran; 'recovered' is "
            "the subset where it produced one. Those recovered frames are the part of `coverage` "
            "bought at the lower threshold, and nothing checks that their box is the apple — see "
            "gate_qualification_blockers."
        ),
        "no_detection_meaning": (
            "an all-False mask, which measure_geom_tol/measure_est_drift drop and count. It is the "
            "hand occluding the apple or the apple leaving frame, not an estimator error, and it is "
            "never folded in as a zero displacement."
        ),
        "empty_mask_meaning": (
            "the detector found a box and SAM 2 returned nothing inside it. That drops the step "
            "exactly like a no-detection frame, but it is NOT the same event — it is the segmenter "
            "failing on a frame where the object was found — so it is counted separately. "
            "n_frames_without_detection + n_frames_with_empty_mask + n_frames_mask_refused is the "
            "whole of the coverage shortfall this module is responsible for."
        ),
        "mask_validity_iou_attr": "MASK_VALIDITY_IOU",
        "mask_validity_meaning": (
            "a non-empty mask whose IoU against the colour reference named in "
            "mask_validity_reference is below mask_validity_min_iou is REFUSED: segment() returns "
            "all-False and the frame is counted in n_frames_mask_refused. It is a THIRD event, not "
            "a no-detection frame and not an empty mask: the detector found a box, SAM 2 filled it, "
            "and what it filled contained essentially none of the object claimed. The per-frame "
            "values are on the module attribute named by mask_validity_iou_attr, in call order, one "
            "per frame the check ran on — so len(MASK_VALIDITY_IOU) == n_segment_calls - "
            "n_frames_without_detection - n_frames_with_empty_mask. This is a check on the OUTPUT "
            "and changes no threshold, prompt, retry or box rule; the detection operating point is "
            "still the generator's, as PR-08 §4 step 2 requires. Registered as PR-08 V6 "
            "(T40_RULE_V6), docs/preregistration/PR-08-V6-mask-validity.md."
        ),
        "mask_validity_reference_fraction_attr": "MASK_VALIDITY_REFERENCE_FRACTION",
        "mask_validity_reference_scope": (
            "PR-08 V10 (T40_RULE_V10, docs/preregistration/PR-08-V10-mask-validity-reference-"
            "scope.md, UNSIGNED as this is recorded — nothing measured under it may be quoted "
            "until the project owner signs it). The reference named in mask_validity_reference is "
            "a predicate for ONE object under ONE appearance, and this module used to apply it to "
            "any label on any pixels. Two consequences were MEASURED rather than supposed. (1) A "
            "label outside mask_validity_reference_labels now refuses the RUN — "
            "MaskValidityReferenceUndefined — instead of refusing every frame: at "
            "WAM_PR08_OBJECT_PROMPT='plate.' twenty of twenty source frames of episode_000000 were "
            "refused at validity IoU 0.0000 with detection scores 0.7524-0.7773, and the harness "
            "reported that as coverage 0.0, a fact about the corpus. (2) A frame whose reference "
            "covers more than mask_validity_reference_max_frame_fraction of the picture is refused "
            "as undecidable and counted in n_frames_mask_refused_reference_not_object_scale, "
            "because on job 189926's train-01-oak-tungsten the predicate returns 40.5-56.4 % of "
            "the frame — warm oak table, not fruit — so it refused a CORRECT mask of a green apple "
            "(IoU 0.025-0.030, the same box and the same ~1 830 px mask that is KEPT at IoU "
            "0.46-0.49 on train-02-linen-overcast) while n_frames_mask_refused_no_reference stayed "
            "0. WHAT IS STILL NOT SEPARATED: an EMPTY reference means 'the fruit is occluded or out "
            "of frame' on the source corpus and 'the reference does not fit' on a restyle whose "
            "apple is not warm, and one frame cannot tell those apart. V10 does not claim to; it "
            "closes the other end, which V6 §5.3 assumed could not open."
        ),
        "mask_validity_threat_to_validity": (
            "n_frames_mask_refused_no_reference counts the refusals where the colour reference "
            "found NO fruit anywhere in the frame — the fruit is fully occluded or out of frame, so "
            "nothing here can confirm any mask. Refusing those removes HARD frames from the "
            "measured population, which for EST_DRIFT_P95 — a p95 that is SUBTRACTED from GEOM_TOL "
            "— plausibly makes the number smaller and the resulting tolerance WIDER, i.e. it errs "
            "in the generator's favour. That is recorded rather than glossed; PR-08 V6 §5 says what "
            "would measure it."
        ),
    }


#: Package -> what it provides here. Probed at run time so the failure names what is actually absent
#: on THIS interpreter rather than a list someone wrote down once.
REQUIRED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("torch", "runs all three models; also decides cuda vs cpu"),
    ("transformers", "GroundingDINO (AutoModelForZeroShotObjectDetection + AutoProcessor) and "
                     "pipeline('depth-estimation') for Depth-Anything-V2"),
    ("sam2", "SAM 2 image predictor — `sam2` 1.1.0, as shipped in the Cosmos-Transfer2.5 venv"),
    ("huggingface_hub", "resolves each checkpoint at its pinned revision, and is what offline mode "
                        "is enforced through"),
    ("PIL", "Pillow — the depth pipeline takes a PIL image, not an ndarray, on the pinned version"),
)


def _hub_cache_dirs() -> list[str]:
    """Every place a hub cache could be, reported in the failures.

    So that "the weights are downloaded, honest" is checkable against a directory listing rather
    than against someone's memory of a job that ran last week on a different filesystem.
    """
    from pathlib import Path

    named = [
        ("HF_HOME", os.environ.get("HF_HOME")),
        ("HF_HUB_CACHE", os.environ.get("HF_HUB_CACHE")),
        ("HUGGINGFACE_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE")),
        ("TRANSFORMERS_CACHE", os.environ.get("TRANSFORMERS_CACHE")),
    ]
    out = [f"{k}={v}" for k, v in named if v]
    default = Path.home() / ".cache" / "huggingface" / "hub"
    out.append(f"{default} ({'exists' if default.is_dir() else 'ABSENT'})")
    return out


def _importable(module: str) -> bool:
    """Is ``module`` importable by this interpreter?

    ``sys.modules`` is consulted first and a ``None`` entry counts as NOT importable — that is
    Python's own documented sentinel for a blocked import, and it is what lets a test prove the
    refusal without uninstalling anything.
    """
    if module in sys.modules:
        return sys.modules[module] is not None
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def missing_package_message(absent: list[tuple[str, str]]) -> str:
    """The loud failure. Names every place that was looked in and what would have to change.

    Deliberately shaped like ``measure_geom_tol.no_segmenter_message()``: whoever hits this is
    holding the same problem from the other end, and the two messages have to be recognisably the
    same refusal or the second one reads as a different, smaller obstacle.
    """
    present = [(m, why) for m, why in REQUIRED_PACKAGES if _importable(m)]
    lines = [
        "FATAL: estimators.apple_sam2 cannot run here, so neither GEOM_TOL nor EST_DRIFT_P95 can be",
        "       measured with it. Nothing was written.",
        "",
        f"       interpreter: {sys.executable}",
        "",
        "       required packages NOT importable by this interpreter:",
    ]
    lines += [f"         - {m:<16} {why}" for m, why in absent] or ["         (none)"]
    if present:
        lines += ["", "       importable:"]
        lines += [f"         - {m:<16} {why}" for m, why in present]
    lines += [
        "",
        "       checkpoints this module would load (id@commit, never local paths):",
    ]
    lines += [
        f"         - {c.what:<34} {c.repo_id}@{c.revision}" for c in CHECKPOINTS
    ]
    lines += [
        "",
        "       hub cache locations looked at:",
    ]
    lines += [f"         - {d}" for d in _hub_cache_dirs()]
    lines += [
        "",
        "       What would have to change:",
        "",
        "       1. Run this inside the Cosmos-Transfer2.5 venv, which already has sam2 1.1.0,",
        "          transformers, torch and timm — cluster/discoverer/98_build_transfer25_env.sbatch",
        "          builds it. This workstation's .venv has no sam2 and is not the place.",
        "       2. Stage the three checkpoints into the hub cache AT THE COMMITS ABOVE —",
        "          cluster/discoverer/102_stage_sam2_weights.sbatch does exactly that and takes its",
        "          ids and revisions from this file. That is a ~3 GB download and the repo rule is",
        "          that nothing is fetched at scale without asking first. ASK. Once the decision is",
        "          taken, WAM_PR08_ALLOW_DOWNLOAD=1 permits this module to fetch them itself.",
        "",
        "       There is no fallback and there will not be one. PR-08 §4 step 2 requires the SAME",
        "       segmenter on both sides of the subtraction in §6, so a stand-in here does not",
        "       produce a worse EST_DRIFT_P95 — it produces a number that is not subtractable from",
        "       GEOM_TOL at all. See scripts/measure_geom_tol.py's no_segmenter_message().",
    ]
    return "\n".join(lines)


def _check_packages() -> None:
    absent = [(m, why) for m, why in REQUIRED_PACKAGES if not _importable(m)]
    if absent:
        raise EstimatorDependencyMissing(missing_package_message(absent))


# Import time, not first-call time. A run that is going to fail for want of `sam2`, or because
# someone pinned a branch, should fail while `resolve_estimators` is still holding the exception and
# can print it, not two subcommands later inside a frame loop. Loading the MODELS stays lazy; this
# only asks whether the packages exist and whether the pins are pins.
_check_packages()
_check_pins()


def missing_weights_message(ckpt: Checkpoint, exc: Exception) -> str:
    """Refusal for a checkpoint that is not in the cache at its pin and may not be fetched."""
    return "\n".join([
        f"FATAL: the {ckpt.what} checkpoint {ckpt.repo_id!r} is not in the local hub cache at the",
        "       revision this module is pinned to, and downloading is not permitted for this run.",
        "       Nothing was written.",
        "",
        f"       pinned revision:  {ckpt.revision}",
        f"       approximate download size: {ckpt.approx_size}",
        f"       hub error: {type(exc).__name__}: {exc}",
        "",
        "       hub cache locations looked at:",
        *[f"         - {d}" for d in _hub_cache_dirs()],
        "",
        "       A cache staged at a DIFFERENT commit fails this check too, and that is the point:",
        "       the artifact names the revision above, so loading any other one would make the",
        "       recorded gate number identify weights it was not measured with.",
        "",
        "       Staging these weights is a download at scale and therefore the project owner's",
        "       call (T-040), and this cluster's login node forbids it outright. Either stage the",
        "       checkpoint from a compute node (cluster/discoverer/102_stage_sam2_weights.sbatch,",
        "       which reads its pins out of this file) and re-run, or set WAM_PR08_ALLOW_DOWNLOAD=1",
        "       to say that the decision has been taken and this process may fetch it.",
    ])


def _cache_probe(repo_id: str, revision: str) -> Exception | None:
    """None when the hub reports ``repo_id`` at ``revision`` resolvable from the local cache alone.

    That is a weaker claim than "every file is present and intact": ``snapshot_download`` with
    ``local_files_only=True`` returns the snapshot folder subject to the hub's own completeness
    check, which depends on what tree listing was cached. It is the strongest claim available
    without a network, and it is exactly the claim :func:`available` needs to make.

    ``revision`` is load-bearing, not decoration. ``huggingface_hub`` writes no ``refs/main`` entry
    for a cache staged at a commit sha, so an unpinned probe reports "not cached" on precisely the
    machines where the weights ARE staged — and the escape hatch would then fetch a different
    revision. ``local_files_only=True`` is the other load-bearing argument: this is the check that
    decides whether a fetch is about to happen, so it must never be able to cause one.

    A broken or absent ``huggingface_hub`` is NOT swallowed into "no cache" — that would be a
    misdiagnosis of a package problem as a weights problem — so the import sits outside the guard.
    """
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo_id=repo_id, revision=revision, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - the hub raises several unrelated types for "no cache"
        return exc
    return None


def available() -> bool:
    """Can this adapter run right now, without fetching anything?

    ``measure_geom_tol``'s ``--method auto`` probes this before it will select the sam2 method, and
    the reason it has to is that a segmenter running without its checkpoints does not crash: it
    returns empty masks, every step drops, and ``coverage: 0.0`` reads as a statement about the
    corpus rather than about the missing weights.

    Every probe carries its checkpoint's pinned revision, so this answers "are the weights this
    module would actually load here?" rather than "is something by that name here?".

    :data:`ALLOW_DOWNLOAD` deliberately does NOT make this true. Permission to fetch is not the same
    claim as "the weights are here", and ``auto`` picking a method must never be the thing that
    starts a 3 GB download on a node that may not be allowed to make one. An explicit
    ``--method sam2`` with ``WAM_PR08_ALLOW_DOWNLOAD=1`` is the way to say that on purpose.

    A broken ``huggingface_hub`` RAISES here rather than returning False: "the hub library is not
    working" is not "the weights are absent", and ``measure_geom_tol`` already treats a probe that
    raises as unavailable while keeping the exception text.
    """
    return all(_cache_probe(c.repo_id, c.revision) is None for c in CHECKPOINTS)


def _require_cached(ckpt: Checkpoint) -> None:
    """Refuse to fetch unless someone said so. No-op when :data:`ALLOW_DOWNLOAD`.

    The three loaders reach the hub by three different code paths (transformers' ``from_pretrained``,
    the pipeline factory, and ``huggingface_hub.hf_hub_download`` for SAM 2), so asking the cache
    directly, at the pin, before any of them runs is the one check that reaches all three and can
    name the checkpoint in its refusal.

    It is a pre-check and it is not the enforcement: :func:`_offline_hub` is, and every loader is
    additionally passed ``local_files_only``. A pre-check on its own is a guard a load can walk
    past — a moved ref, a file missing from a partial snapshot, an etag revalidation.
    """
    if ALLOW_DOWNLOAD:
        return
    exc = _cache_probe(ckpt.repo_id, ckpt.revision)
    if exc is not None:
        raise EstimatorWeightsMissing(missing_weights_message(ckpt, exc)) from exc


@contextlib.contextmanager
def _offline_hub() -> Iterator[None]:
    """Make a hub request IMPOSSIBLE for the duration, unless downloading was permitted.

    ``huggingface_hub.constants.is_offline_mode()`` reads the module global at CALL time and the
    hub's request hook raises ``OfflineModeIsEnabled`` when it is set (checked against the installed
    huggingface_hub 1.25.1), so assigning it here does block requests — the older comment in this
    file claiming it was read once at import and could not be set at runtime was simply wrong.

    ``transformers`` copies the flag at ITS import, which is why this is belt and braces: every
    loader is also passed ``local_files_only``. The previous value is restored, because this module
    is a library and a process-wide switch it never turns back off is a bug for its caller.
    """
    if ALLOW_DOWNLOAD:
        yield
        return
    try:
        from huggingface_hub import constants as hub_constants
    except ImportError as exc:  # pragma: no cover - REQUIRED_PACKAGES makes this unreachable
        raise EstimatorDependencyMissing(
            "FATAL: huggingface_hub.constants is not importable, so offline mode cannot be "
            "enforced and a load could fetch weights this run is not permitted to fetch. "
            f"Nothing was written. ({type(exc).__name__}: {exc})"
        ) from exc
    previous = getattr(hub_constants, "HF_HUB_OFFLINE", False)
    hub_constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        hub_constants.HF_HUB_OFFLINE = previous


def _local_files_only() -> bool:
    """What every loader passes so that transformers' own import-time copy of the offline flag
    cannot let a fetch through. :data:`ALLOW_DOWNLOAD` inverts it, which is the whole of the
    escape hatch."""
    return not ALLOW_DOWNLOAD


# -- input validation ------------------------------------------------------------------------------


def _as_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    """``(H, W, 3)`` uint8, or a loud refusal.

    A float array is refused rather than rescaled: [0, 1] and [0, 255] floats are indistinguishable
    from the array alone and rescale to different pictures, which is a different detection, which is
    a different centroid — and the centroid is the gate. RGBA is accepted with the alpha dropped
    because Replicator's ``rgb`` annotator hands back four channels (``isaac_binding.render_frame``
    already drops it, but a capture written by anything else may not have).
    """
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(
            f"segment/estimate_depth expect (H, W, 3|4) RGB, got shape {arr.shape}. "
            "A single-channel or batched array is not a frame."
        )
    if arr.dtype != np.uint8:
        raise ValueError(
            f"segment/estimate_depth expect a uint8 frame, got {arr.dtype}. This is not rescaled "
            "here on purpose: a float frame in [0, 1] and one in [0, 255] look identical to numpy "
            "and produce different detections, and the resulting centroid error would be silent. "
            "Convert at the source, where the range is known."
        )
    return np.ascontiguousarray(arr[:, :, :3])


# -- the second opinion the validity check is made of ----------------------------------------------


def object_color_reference(rgb: np.ndarray) -> np.ndarray:
    """``(H, W)`` bool: where a non-learned colour predicate says the fruit is.

    The warm-and-saturated discriminator ``build_identity_calibration.apple_mask`` uses, restated
    here rather than imported: this module is imported as ``estimators.apple_sam2`` by two harnesses
    and a cluster job, and reaching back into ``scripts/build_identity_calibration.py`` from inside
    it would make the estimator depend on the calibration tooling's import graph for a three-line
    predicate. The two copies are pinned to each other by a test that asserts they agree pixel for
    pixel (``tests/test_apple_sam2_estimator.py``), which is the check that a restatement needs and
    an import would not have.

    IT IS NOT GROUND TRUTH, and nothing here treats it as such. It cannot say where the apple's
    boundary is, it fails on shadowed fruit, and a mask that agrees with it is not thereby correct.
    The only claim it is used for is the one it can support: a mask containing essentially NONE of
    the warm, saturated pixels in a frame that has some is not a mask of the fruit. On this corpus
    the cloth and the plate are neutral to within two counts and the robot is black or bare metal
    (the identity-calibration seed observations), so the only saturated warm thing in any of these
    frames is the fruit.

    Raw predicate, not one grown connected component, and no minimum area: an occluded frame is the
    case of interest here, not an error.
    """
    arr = _as_uint8_rgb(rgb)
    r = arr[:, :, 0].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    mx, mn = arr.max(2), arr.min(2)
    saturation = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)
    return (r > 90) & ((r - b) > 50) & (saturation > 0.35)


def mask_validity_iou(mask: np.ndarray, reference: np.ndarray) -> float:
    """Intersection over union of two boolean masks; 0.0 when the union is empty.

    Symmetric on purpose. "How much of the mask is object" alone would pass a mask that covers the
    apple AND the whole plate, and "how much of the object is masked" alone would pass a mask that
    covers the entire frame. The failure this filter exists for produces both shapes.
    """
    inter = int(np.logical_and(mask, reference).sum())
    union = int(np.logical_or(mask, reference).sum())
    return float(inter) / float(union) if union else 0.0


def reference_frame_fraction(reference: np.ndarray) -> float:
    """Fraction of the frame the colour reference covers. ``0.0`` on an empty array."""
    ref = np.asarray(reference, dtype=bool)
    return float(ref.sum()) / float(ref.size) if ref.size else 0.0


def reference_is_object_scale(reference: np.ndarray) -> bool:
    """Is this reference the size of AN OBJECT, or the size of a SCENE? (PR-08 V10.)

    :func:`object_color_reference` justifies itself with *"the only saturated warm thing in any of
    these frames is the fruit"*. That is a claim about AppleToPlate's real pixels under
    AppleToPlate's real lighting, and PR-08 is a pre-registration whose committed prompts change the
    lighting and the fruit's colour on purpose. When the claim is false the predicate does not go
    quiet — measured on job 189926's ``train-01-oak-tungsten``, it returns 40.5-56.4 % of the frame,
    all of it warm oak table. A predicate covering half the scene cannot arbitrate whether any
    particular mask is the fruit, in either direction: it refuses a correct mask of a green apple
    (measured: IoU 0.025-0.030) and it would ACCEPT a mask of the table (IoU 1.0 against itself).
    Both are silent, and the second is worse.

    So the reference is checked before it is used as one. An EMPTY reference is deliberately
    ``True`` here — it is not a scene, and V6 already handles it, refusing the frame through the IoU
    and counting it in :data:`MASK_REFUSED_NO_REFERENCE_FRAMES`. V10 changes nothing about that
    path; it adds the other end of the distribution, which V6 did not have.

    Exported rather than kept private because ``scripts/robot_composite.py`` (PR-08 V9) drives this
    module's reference over generated pixels too and is exposed to the same defect by its own §5.1.
    V10 does not wire it there — that file belongs to V9 and to whoever signs it — but there is now
    ONE definition of "this reference is applicable here" available to both, which is V9's own stated
    reason for reaching into this module rather than restating anything.
    """
    return reference_frame_fraction(reference) <= MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION


def mask_validity_reference_is_defined() -> bool:
    """Does the filter have a reference for the object THIS PROCESS was told to segment?

    Cheap, side-effect free and importable, so a harness can find out before decoding a corpus
    instead of after. :func:`segment` asks the same question and refuses on the answer.
    """
    return OBJECT_TEXT_PROMPT in MASK_VALIDITY_REFERENCE_LABELS


def _require_mask_validity_reference() -> None:
    """Refuse the run when the mask-validity filter has no reference for this label. (PR-08 V10.)"""
    if mask_validity_reference_is_defined():
        return
    raise MaskValidityReferenceUndefined(
        f"REFUSED: the mask-validity filter has no reference for {OBJECT_TEXT_PROMPT!r}.\n"
        f"       WAM_PR08_OBJECT_PROMPT={OBJECT_TEXT_PROMPT_RAW!r} normalises to "
        f"{OBJECT_TEXT_PROMPT!r}, and PR-08 V6 requires every non-empty mask to clear\n"
        f"       mask_validity_min_iou={MASK_VALIDITY_MIN_IOU} against "
        f"{MASK_VALIDITY_REFERENCE} — a predicate for the FRUIT, which\n"
        "       this module applied to every label unconditionally. The only label it is a "
        "reference for is: "
        + ", ".join(sorted(MASK_VALIDITY_REFERENCE_LABELS))
        + ".\n"
        "\n"
        "       THIS IS NOT A FACT ABOUT THE CORPUS AND MUST NOT BE READ AS ONE. Measured on 20 "
        "source frames of\n"
        "       episode_000000 before this refusal existed: the detector scored 0.7524-0.7773 on "
        "every frame, SAM 2\n"
        "       drew a mask on every frame, and the filter refused all twenty at validity IoU "
        "0.0000 — because a\n"
        "       correct plate mask contains no warm fruit pixels. The harness then reported "
        "coverage 0.0 and the\n"
        "       gate refused for the wrong reason. The same twenty frames at "
        "WAM_PR08_OBJECT_PROMPT='apple.' are\n"
        "       refused zero times, at validity IoU 0.9686-0.9744.\n"
        "\n"
        "       What this does NOT mean: that the label cannot be measured. It means the FILTER is "
        "not defined for\n"
        "       it, and this module may not decide alone whether to measure the label without one — "
        "the committed\n"
        "       contract configs/transfer25/pr08_geom_tol.json states mask_validity_min_iou and is "
        "cross-checked\n"
        "       field for field, so a run that quietly skipped the filter would write an artifact "
        "claiming it ran.\n"
        "       Registering a reference for this label, or registering that it is measured "
        "unfiltered, is a further\n"
        "       version alongside PR-08 V10 and the project owner's signature — not an edit to this "
        "file.\n"
        "       docs/preregistration/PR-08-V10-mask-validity-reference-scope.md §2."
    )


def _device() -> str:
    if DEVICE_OVERRIDE:
        return DEVICE_OVERRIDE
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# -- the models, loaded once per process ------------------------------------------------------------
#
# Module-level caches, not lru_cache, so that reset_models() can drop them and so that a reader can
# see there is exactly one of each. Loading SAM 2 hiera-large per frame would turn a few-minute
# calibration into an overnight one and would change no number, which is the kind of cost that gets
# discovered after the run.

_DETECTOR: tuple[Any, Any] | None = None
_PREDICTOR: Any | None = None
_DEPTH_PIPE: Any | None = None


def reset_models() -> None:
    """Drop the cached models.

    Exists so a test can prove the cache is a cache, and so a long-lived process can give the VRAM
    back between stages. It does not reset the counters: those describe the run, not the models.
    """
    global _DETECTOR, _PREDICTOR, _DEPTH_PIPE
    _DETECTOR = _PREDICTOR = _DEPTH_PIPE = None


def _detector() -> tuple[Any, Any]:
    """GroundingDINO ``(processor, model)``, cached, at :data:`GROUNDING_DINO_MODEL_REVISION`."""
    global _DETECTOR
    if _DETECTOR is None:
        ckpt = CHECKPOINTS[0]
        _require_cached(ckpt)
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        with _offline_hub():
            processor = AutoProcessor.from_pretrained(
                ckpt.repo_id, revision=ckpt.revision, local_files_only=_local_files_only()
            )
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                ckpt.repo_id, revision=ckpt.revision, local_files_only=_local_files_only()
            ).to(_device())
        model.eval()
        _DETECTOR = (processor, model)
    return _DETECTOR


def _sam2_api_message(missing: str) -> str:
    return "\n".join([
        f"FATAL: the installed `sam2` does not expose {missing}, so the SAM 2 checkpoint cannot be",
        "       loaded at a pinned revision. Nothing was written.",
        "",
        "       Why this module does not simply call SAM2ImagePredictor.from_pretrained: that path",
        "       reaches sam2.build_sam.build_sam2_hf -> _hf_download(model_id), which calls",
        "       hf_hub_download WITHOUT a revision and forwards none of its caller's kwargs, so it",
        "       resolves refs/main. On a cache staged at a commit there is no such ref, and where",
        "       there is one it may be a different commit from the one this run records. So the",
        "       two steps are performed here instead, with the pin.",
        "",
        f"       pinned segmenter: {SAM2_MODEL_CHECKPOINT}@{SAM2_MODEL_REVISION}",
        "",
        "       Check the sam2 version in the Cosmos-Transfer2.5 venv (this was written against",
        "       sam2 1.1.0's build_sam.py). There is no fallback to the unpinned loader: it would",
        "       load weights the artifact does not name.",
    ])


def _predictor() -> Any:
    """SAM 2 image predictor, cached, at :data:`SAM2_MODEL_REVISION`.

    Reproduces ``build_sam2_hf``'s two steps — resolve the repo's checkpoint file, then
    ``build_sam2`` on the config name that repo maps to — because only the first of them can carry
    a revision, and ``build_sam2_hf`` does not. Same checkpoint Cosmos-Transfer2.5 drives, same
    config, at a commit this run can name.
    """
    global _PREDICTOR
    if _PREDICTOR is None:
        ckpt = CHECKPOINTS[1]
        _require_cached(ckpt)
        import sam2.build_sam as build_sam_mod
        from huggingface_hub import hf_hub_download
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        filenames = getattr(build_sam_mod, "HF_MODEL_ID_TO_FILENAMES", None)
        build_sam2 = getattr(build_sam_mod, "build_sam2", None)
        if filenames is None or build_sam2 is None:
            raise EstimatorCheckpointUnusable(
                _sam2_api_message("HF_MODEL_ID_TO_FILENAMES and build_sam2")
            )
        if ckpt.repo_id not in filenames:
            raise EstimatorCheckpointUnusable(
                "\n".join([
                    f"FATAL: `sam2` does not know the checkpoint {ckpt.repo_id!r}: it is not in",
                    "       sam2.build_sam.HF_MODEL_ID_TO_FILENAMES, so there is no config file to",
                    "       build it with. Nothing was written.",
                    "",
                    "       ids this sam2 knows:",
                    *[f"         - {k}" for k in sorted(filenames)],
                    "",
                    "       Set WAM_PR08_SAM2_CHECKPOINT and WAM_PR08_SAM2_REVISION together to one",
                    "       of those, or install the sam2 that ships the one you want.",
                ])
            )
        config_name, checkpoint_name = filenames[ckpt.repo_id]
        with _offline_hub():
            ckpt_path = hf_hub_download(
                repo_id=ckpt.repo_id,
                filename=checkpoint_name,
                revision=ckpt.revision,
                local_files_only=_local_files_only(),
            )
            model = build_sam2(
                config_file=config_name, ckpt_path=ckpt_path, device=_device()
            )
        _PREDICTOR = SAM2ImagePredictor(model)
    return _PREDICTOR


def _depth_pipeline() -> Any:
    """Depth-Anything-V2 depth-estimation pipeline, cached, and METRIC or it does not return.

    The metric check happens here rather than in :func:`estimate_depth` so that a relative
    checkpoint stops the run before frame 0. Discovering it on frame 37 means 37 frames of a
    finished-looking artifact already exist.
    """
    global _DEPTH_PIPE, DEPTH_ESTIMATION_TYPE, DEPTH_IS_METRIC, DEPTH_MAX_DEPTH_M
    if _DEPTH_PIPE is None:
        ckpt = CHECKPOINTS[2]
        _require_cached(ckpt)
        from transformers import pipeline

        with _offline_hub():
            pipe = pipeline(
                "depth-estimation",
                model=ckpt.repo_id,
                revision=ckpt.revision,
                local_files_only=_local_files_only(),
                device=_device(),
            )
        config = getattr(getattr(pipe, "model", None), "config", None)
        # Absent means relative: that is transformers' own DepthAnythingConfig default, and an
        # unstated claim is not a claim — the same rule measure_geom_tol applies to gate_qualified.
        DEPTH_ESTIMATION_TYPE = str(getattr(config, "depth_estimation_type", "relative"))
        DEPTH_IS_METRIC = DEPTH_ESTIMATION_TYPE == "metric"
        max_depth = getattr(config, "max_depth", None)
        DEPTH_MAX_DEPTH_M = float(max_depth) if max_depth is not None else None
        if not DEPTH_IS_METRIC:
            raise EstimatorCheckpointUnusable(
                "\n".join([
                    f"FATAL: {DEPTH_MODEL_CHECKPOINT!r} is a {DEPTH_ESTIMATION_TYPE.upper()} depth",
                    "       checkpoint. PR-08 §4 step 3 asks for the ABSOLUTE depth error in metres",
                    "       against Isaac's distance_to_camera, and a relative Depth-Anything map is",
                    "       affine-free INVERSE depth: arbitrary per-image scale, and larger means",
                    "       NEARER. Subtracting it from metres produces a number with no units,",
                    "       ordered backwards, that lands in the artifact under a key called",
                    "       'mean_m' and looks entirely reasonable. That is the expensive failure in",
                    "       this project, so nothing is returned.",
                    "",
                    "       read from the loaded model config, not from the id:",
                    f"         depth_estimation_type = {DEPTH_ESTIMATION_TYPE!r}",
                    f"         max_depth             = {DEPTH_MAX_DEPTH_M!r}",
                    f"         revision              = {DEPTH_MODEL_REVISION!r}",
                    "",
                    "       Set WAM_PR08_DEPTH_CHECKPOINT (and WAM_PR08_DEPTH_REVISION with it) to a",
                    "       metric head, e.g.:",
                    *[f"         - {c}" for c in METRIC_DEPTH_CHECKPOINT_SUGGESTIONS],
                    "",
                    "       Note which number is the gate: EST_DRIFT_P95 is the p95 of the CENTROID",
                    "       displacement and does not depend on depth at all. This refusal protects",
                    "       §4 step 3's recorded depth error, not the budget — and it is still a",
                    "       refusal, because a recorded number that is wrong is read as a measured",
                    "       one.",
                ])
            )
        _DEPTH_PIPE = pipe
    return _DEPTH_PIPE


# -- the contract ------------------------------------------------------------------------------------


def _best_box(rgb: np.ndarray) -> np.ndarray | None:
    """Highest-scoring GroundingDINO box for the prompt, as ``[x0, y0, x1, y1]``, or None.

    Upstream's rule, step for step. ``cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_model.py``
    post-processes at ``threshold=0.15, text_threshold=0.25``, and **only when that returns no box
    at all** repeats the post-processing once at ``(0.10, 0.10)``. It then takes
    ``np.argsort(scores)[::-1]`` and uses index 0.

    Highest score and not largest area — which is also our own reason, arrived at independently and
    now confirmed against upstream: on a tabletop scene the largest box a text prompt returns is
    routinely the table or the whole plate-plus-apple region, and a box that contains the apple is
    not the same prompt for SAM 2 as a box that IS the apple — it hands SAM 2 an ambiguous prompt
    and the mask lands on whichever object dominates it. The centroid then tracks the plate and the
    number still looks like a number.

    The retry runs the DETECTOR once and post-processes twice, rather than running the whole
    forward pass again: ``post_process_grounded_object_detection`` is a pure function of ``outputs``
    and the two thresholds, so a second forward pass would cost a second inference to produce
    identical logits. Upstream re-post-processes for the same reason.

    Two counters exist because the retry is the part of upstream's rule most likely to be doing
    something we would not want if we could see it: it buys a detection by accepting a weak one, and
    on a frame where the apple is genuinely occluded a weak box lands on something else. That turns
    an honest all-False mask into a confident wrong one — which RAISES coverage while LOWERING mask
    quality, i.e. it hides in exactly the number the harness gates on. It is upstream's behaviour so
    it stays; making it countable is the least this module can do about it. The WINNING SCORE is
    kept too, in :data:`DETECTION_SCORES` — a count says how often the retry fired, and only the
    scores say how weak the detections it bought were.
    """
    global RETRY_FRAMES, RETRY_RECOVERED_FRAMES

    import torch
    from PIL import Image

    processor, model = _detector()
    h, w = rgb.shape[:2]
    inputs = processor(
        images=Image.fromarray(rgb), text=OBJECT_TEXT_PROMPT, return_tensors="pt"
    ).to(_device())
    with torch.inference_mode():
        outputs = model(**inputs)

    def post_process(box_thr: float, text_thr: float) -> tuple[np.ndarray, np.ndarray]:
        # `threshold=` is transformers >= 4.51's spelling; it was `box_threshold=` before, and the
        # pinned env is 4.51.3. If that pin moves backwards this raises TypeError, which is the
        # right outcome — the alternative is the old kwarg silently keeping its 0.25 default while
        # ours is ignored, i.e. a different detection threshold than the one recorded in
        # ESTIMATOR_VERSION and in the committed contract.
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_thr,
            text_threshold=text_thr,
            target_sizes=[(h, w)],
        )[0]
        got = np.asarray(results["scores"].detach().cpu(), dtype=np.float64).reshape(-1)
        return got, np.asarray(results["boxes"].detach().cpu(), dtype=np.float64).reshape(-1, 4)

    scores, boxes = post_process(BOX_THRESHOLD, TEXT_THRESHOLD)
    if scores.size == 0:
        # EXACTLY ONE retry, and only on "no box at all". Retrying on a low score, or looping the
        # thresholds down until something appears, would be a detector of our own design — see the
        # comment on BOX_THRESHOLD for why that is not an improvement but a different measurement.
        RETRY_FRAMES += 1
        scores, boxes = post_process(RETRY_BOX_THRESHOLD, RETRY_TEXT_THRESHOLD)
        if scores.size == 0:
            return None
        RETRY_RECOVERED_FRAMES += 1
    best = int(np.argmax(scores))
    # AFTER the retry branch and before the return, so it records the score of the box that is
    # actually handed to SAM 2 — including a retry's, which is the one this list exists for. A
    # frame that reached neither return appends nothing; see DETECTION_SCORES on the alignment.
    DETECTION_SCORES.append(float(scores[best]))
    return boxes[best]


def segment(rgb: np.ndarray) -> np.ndarray:
    """``(H, W)`` bool mask of the object in THIS frame. All-False when it is not found.

    text prompt -> GroundingDINO box -> SAM 2 mask, which is the topology
    Cosmos-Transfer2.5's own ``sam2_model.py`` uses, with its own checkpoints. This is the function
    the gate rides on: ``EST_DRIFT_P95`` is the p95 of the displacement between this mask's centroid
    and the ground-truth mask's, and ``GEOM_TOL`` is measured from masks this same function produced.

    **Both models are loaded before the first detection**, not on demand, so that a missing or
    unpinned SAM 2 checkpoint refuses on frame 0. Reaching the segmenter only through the detected
    branch means a capture where nothing is detected finishes with no segmenter present at all and
    reports ``coverage: 0.0`` as a fact about the corpus.

    **No detection is not an error.** The Dex3 hand occludes the apple and the apple leaves frame,
    repeatedly, in this corpus. Those frames come back all-False, ``centroid_of_mask`` turns that
    into ``None``, and the callers DROP AND COUNT them into ``coverage``. Raising would kill a run
    over a normal event; returning the previous mask would invent a displacement that was never
    observed and would pull the p95 down, which WIDENS G0b — conservative-looking and backwards.
    An empty mask from a box that WAS detected drops identically but is a different event, so it is
    counted separately in :data:`EMPTY_MASK_FRAMES`; a coverage shortfall with no recorded
    explanation is a run someone has to do again to find out why.

    **A mask of the wrong object is refused, and is a THIRD event.** A non-empty mask containing
    essentially none of the warm, saturated pixels :func:`object_color_reference` finds is not a
    mask of the fruit — on this corpus it is the plate — and it comes back all-False, counted in
    :data:`MASK_REFUSED_FRAMES` and never folded into either of the two above. It is a check on the
    OUTPUT: the detection operating point, the prompt, the retry and the box rule are untouched, so
    this is still the segmenter §4 step 2 asks for, applied identically to both sides of §6's
    subtraction by living in the module both harnesses call.

    **The reference is checked before it is used as one, and a label it does not cover refuses the
    RUN** (PR-08 V10). Two things this function used to do quietly and now does loudly:
    ``WAM_PR08_OBJECT_PROMPT="plate."`` raises :class:`MaskValidityReferenceUndefined` on the first
    call instead of refusing all twenty frames of the pass and reporting ``coverage: 0.0``; and a
    frame whose colour reference covers more than
    :data:`MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION` of the picture is refused as a frame this
    filter cannot decide — counted in
    :data:`MASK_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES`, inside
    :data:`MASK_REFUSED_FRAMES` — rather than being arbitrated by a predicate that has moved to the
    background. Neither is a change to the detection path, and neither can ACCEPT a frame the module
    accepted before: V10's only possible effect on a frame is a refusal, and its only possible effect
    on a run is a refusal.
    """
    global SEGMENT_CALLS, NO_DETECTION_FRAMES, EMPTY_MASK_FRAMES
    global MASK_REFUSED_FRAMES, MASK_REFUSED_NO_REFERENCE_FRAMES
    global MASK_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES

    # BEFORE ANY COUNTER MOVES AND BEFORE ANY WEIGHT LOADS. A label the filter has no reference for
    # is not a frame this module can decide — refusing each frame instead reads out of the harness
    # as coverage 0.0, i.e. as a fact about the corpus. PR-08 V10.
    _require_mask_validity_reference()

    frame = _as_uint8_rgb(rgb)
    SEGMENT_CALLS += 1
    h, w = frame.shape[:2]

    _detector()
    predictor = _predictor()

    box = _best_box(frame)
    if box is None:
        NO_DETECTION_FRAMES += 1
        return np.zeros((h, w), dtype=bool)

    import torch

    with torch.inference_mode():
        predictor.set_image(frame)
        masks, _scores, _logits = predictor.predict(
            box=box[None, :], multimask_output=False
        )
    mask = np.asarray(masks).reshape(-1, h, w)[0] > 0
    mask = mask.astype(bool)
    if not mask.any():
        EMPTY_MASK_FRAMES += 1
        return mask

    # THE VALIDITY CHECK, and it is a check on this mask rather than on how it was produced: the
    # box above is the box upstream's rule selected at upstream's thresholds, and the mask is the
    # one SAM 2 drew for it. Nothing is re-detected, re-prompted or re-drawn. All that is decided
    # here is whether this frame is one we are willing to MEASURE on — the same decision the two
    # callers already make for every frame this function returns all-False for. See the block above
    # MASK_VALIDITY_MIN_IOU for why that is not the segmenter change §4 step 2 forbids, and for why
    # the exact threshold is a value read off a gap rather than a coined number.
    reference = object_color_reference(frame)
    overlap = mask_validity_iou(mask, reference)
    MASK_VALIDITY_IOU.append(overlap)
    MASK_VALIDITY_REFERENCE_FRACTION.append(reference_frame_fraction(reference))

    # PR-08 V10, and it gates the WHOLE decision rather than only the refusal branch. When the
    # reference covers the scene it cannot arbitrate this mask in either direction: it refuses a
    # correct mask of a restyled apple (measured 0.025-0.030 on train-01-oak-tungsten) and it
    # ACCEPTS a mask of the warm background (1.0 against itself). Reaching the IoU test at all would
    # therefore be the fail-OPEN half of the same defect. Counted inside MASK_REFUSED_FRAMES, and
    # apart from it, so V6 §5.3's question — is the segmenter wrong here, or does the reference not
    # fit here — has an answer in the artifact instead of a zero that means neither.
    if not reference_is_object_scale(reference):
        MASK_REFUSED_FRAMES += 1
        MASK_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES += 1
        return np.zeros((h, w), dtype=bool)

    if overlap < MASK_VALIDITY_MIN_IOU:
        MASK_REFUSED_FRAMES += 1
        if not reference.any():
            # The fruit is not visible at all, so this refusal is "nothing here can confirm the
            # mask", not "the mask is demonstrably the plate". Counted apart because the two have
            # opposite implications for the budget — see stats()['mask_validity_threat_to_validity'].
            MASK_REFUSED_NO_REFERENCE_FRAMES += 1
        return np.zeros((h, w), dtype=bool)
    return mask


def estimate_depth(rgb: np.ndarray) -> np.ndarray:
    """``(H, W)`` float32 depth in METRES from the camera.

    Recorded for PR-08 §4 step 3's absolute depth error; it is NOT ``EST_DRIFT_P95``, which is the
    centroid displacement :func:`segment` produces. Metres are guaranteed by the load-time refusal in
    :func:`_depth_pipeline`: a relative checkpoint never gets this far.

    The returned map is checked against the input grid rather than trusted. transformers moved depth
    post-processing into the image processor's ``post_process_depth_estimation`` partway through the
    4.x line, and before that the pipeline handed back the raw patch-grid tensor. A depth error
    averaged over two different grids is not a depth error, and it does not look wrong.
    """
    frame = _as_uint8_rgb(rgb)
    from PIL import Image

    out = _depth_pipeline()(Image.fromarray(frame))
    predicted = out["predicted_depth"]
    depth = np.asarray(
        predicted.detach().cpu().numpy() if hasattr(predicted, "detach") else predicted,
        dtype=np.float32,
    )
    depth = np.squeeze(depth)
    if depth.shape != frame.shape[:2]:
        raise RuntimeError(
            f"the depth pipeline returned a {depth.shape} map for a {frame.shape[:2]} frame. "
            "It is not resized here: the pipeline's own post-processing is what resizes to the "
            "source grid, and a shape mismatch means that post-processing did not run — which "
            "means the values are not what this module claims they are either. Check the "
            "transformers version against the pin in the Cosmos-Transfer2.5 venv."
        )
    return depth
