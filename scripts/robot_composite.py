#!/usr/bin/env python3
"""PR-08 §6 G0c — the real robot's pixels, composited back over every generated frame.

    composite_clip(source, generated, ...)  ->  the generated clip, with the robot's REAL pixels
                                                copied back in, in place, every frame, no exceptions

G0c is quoted here in full because everything in this file is a consequence of it and of nothing
else:

    "``video_fidelity`` provably cannot see the generic-manipulator defect
    (``runs/backbone_eval/video/embodiment_grid.png``), and any IoU threshold on the robot mask
    would be a coined number. So the real robot's pixels are **unconditionally composited back**
    over every generated frame, using the robot segmentation mask. The defect cannot enter, and no
    threshold is needed. Robot-mask IoU between source and generated is still recorded, as a
    diagnostic on the generator, never as a gate."

WHY THIS IS NOT A CHECKER, AND WHY IT LIVES IN THE GENERATION PATH
------------------------------------------------------------------
G0a and G0b are gates: they measure a finished corpus and refuse it. G0c is not, and reading it as
one is the mistake this file exists to make impossible. Its sentence is "the defect cannot enter",
not "the defect is detected and rejected" — because the only instrument that could detect it,
``video_fidelity``, has been measured against the defect and cannot see it. A checker downstream of
generation would therefore be a checker with no detector: it would have to invent an IoU threshold,
which §6 forbids in the same breath, and it would pass every clip whose generated manipulator
happened to land in roughly the right place.

So the compositing happens between the model writing its mp4 and the driver calling the unit a
success, in ``restyle_transfer25.run_unit``. A unit that reaches ``status: "success"`` has been
composited; a unit that has not been composited never gets that status, and its ``vision.mp4`` is
renamed out of the way so the harvest cannot file it even if someone later misreads the record.
There is no flag, no environment variable and no backend that turns this off. ``--backend null``
composites too — it is a placeholder generator, not a placeholder pipeline.

WHERE THE MASK COMES FROM, AND WHY THERE IS NO SECOND SEGMENTATION STACK
------------------------------------------------------------------------
The robot's pixels come from the SOURCE frame; a mask says which pixels those are. The corpus does
not carry one. ``scripts/build_pr08_source.py`` says so on purpose — *"``depth`` and
``segmentation`` are absent from every episode entry, and that is a decision rather than an
oversight"* — and PR-08 §4 records the same fact from the other side: AppleToPlate ships one RGB
camera and every conditioning signal but Canny is estimated. Isaac's ``semantic_segmentation``
annotator is §8 item 5 and renders sim, not the real teleop frames this path restyles. So there is
no recorded robot segmentation anywhere in this project, and the honest answer is that the mask has
to be estimated from the source RGB.

It is estimated with **the pair that is already pinned**: ``scripts/estimators/apple_sam2.py``,
GroundingDINO -> SAM 2, at the commits that module fixes, driven with a robot text prompt instead of
``"apple."``. Not a second segmenter, not a new dependency, not a different checkpoint — the same
weights, the same thresholds, the same offline enforcement and the same loud refusals, reached
through that module's own ``_detector()`` / ``_predictor()``. PR-08 §4 step 2's rule is "the SAME
segmenter", and ``measure_geom_tol`` and ``measure_est_drift`` both already ride on this one; adding
a second stack here would put a third estimator in a pipeline whose geometry budget is a subtraction
of two numbers that are only comparable because they came from one.

**The prompt is a committed constant, not a flag.** :data:`ROBOT_TEXT_PROMPT` has no environment
override and no command-line switch, deliberately, and unlike ``apple_sam2.OBJECT_TEXT_PROMPT`` —
which takes ``$WAM_PR08_OBJECT_PROMPT`` — it cannot be moved per run. A per-run prompt is a per-run
decision about which pixels are protected from the generator, taken by whoever typed the command,
recorded nowhere anybody would look, and invisible in the output: a narrower prompt yields a smaller
mask, a smaller mask lets more generated manipulator through, and the clip still looks fine. That is
exactly the defect G0c exists to exclude, arriving through G0c's own configuration.

**The box rule is the union of every detection, not the best one.** ``apple_sam2._best_box`` takes
the highest-scoring box and argues for it at length, and that argument is right for the apple: the
apple is one object, and a box that merely CONTAINS it hands SAM 2 an ambiguous prompt whose mask
lands on the plate. The robot is not one object in this view — it is two arms and at least one Dex3
hand — and the two selection rules fail in opposite directions. For the apple, over-coverage is the
danger (the centroid tracks the plate). For the robot, UNDER-coverage is the danger, and it is the
whole danger: a mask that misses the second arm leaves that arm's generated pixels in the frame,
which is the generic-manipulator defect entering exactly as if there were no composite at all. So
every box the detector returns above its threshold is segmented and the masks are OR-ed. Over-
coverage is not free either — it makes the restyle weaker and, past a point, vacuous — and that is
what :func:`check_mask`'s area bound is for.

**Upstream's one retry is NOT run here, and that is a deliberate divergence rather than an
oversight.** ``apple_sam2._best_box`` — following Cosmos-Transfer2.5's own ``sam2_model.py`` — post-
processes once at ``(BOX_THRESHOLD, TEXT_THRESHOLD)`` and, *only when that returns no box at all*,
repeats the post-processing once at ``(0.10, 0.10)``. That retry is right where it lives and wrong
here, because the two callers do OPPOSITE things with an ungrounded frame. ``apple_sam2``'s callers
DROP such a frame and count it into ``coverage``, so the retry recovers data that would otherwise be
lost. G0c REFUSES the clip (:func:`check_mask`), loudly, by name. So here the retry does not recover
a frame — it *suppresses a refusal*, by buying a detection at a confidence the pinned adapter itself
describes as "accepting a weak one", and ``_best_box``'s own docstring says what a weak box does on
an occluded frame: it "lands on something else", turning "an honest all-False mask into a confident
wrong one". A confident wrong robot mask composites source pixels over some non-robot region and
leaves the generated manipulator exactly where it was — the defect entering silently, in place of a
refusal that would have been printed. Between a loud refusal and a quiet wrong mask this file takes
the refusal, every time.

The cost of that choice is real and is stated rather than hidden: this makes the robot masker a
STRICTER detector than the one PR-08 §4 step 2 pins as "the same segmenter". It is recorded as such
— :meth:`Sam2RobotMasker.provenance` carries ``upstream_retry_not_run`` with upstream's two
thresholds and this argument's summary, and the string is part of the mask cache key — so a later
reader cannot mistake these masks for ones the adapter would have produced. Anyone who reverses this
decision must move that record with it, because the cached masks are keyed on it.

A DETECTION THAT IS THE APPLE IS NOT A ROBOT, AND IS DROPPED BEFORE THE UNION
------------------------------------------------------------------------------
The union rule above has an exposure the ``apple_sam2`` path does not, and it was found by
measurement rather than by reading: **on frames where the robot is out of shot, this prompt grounds
the apple.** GroundingDINO grounds phrases, it does not decide that a phrase is absent, so
``"robot arm. robotic hand. robotic gripper."`` against a tablecloth, a plate and a piece of fruit
returns its best-scoring box above 0.15 and that box lands on the fruit. ``d739a87`` measured it and
``runs/pr08-robot-mask-empty/`` records it: the robot is genuinely absent from ~36 % of this
corpus's frames (verdict ABSENT, which is settled and is not what this section is about), and on the
robot-absent frames of the 40-episode sample the masker returned a non-empty mask on 98 of 240.
Re-segmenting every box of those 710 frames puts 146 detections at IoU 0.94-0.98 against
``apple_sam2.object_color_reference`` — they are the apple, and a person has looked at them
(``runs/pr08-robot-mask-apple/sheet_absent_now_empty.png``).

**Under G0c an apple inside the robot mask is the worst shape a defect can have, because it is a
SILENT PASS.** The robot mask is the region composited back from the source, so the generated apple
is overwritten by the source apple: the object the task is about stops being restyled, and arms B
and C become arm A for that object while still costing their GPU hours. No gate downstream can see
it. G0a measures labels. G0b measures geometry, and a pixel-identical apple has moved zero pixels —
it does not merely pass G0b, it passes it perfectly. The robot-mask IoU is "a diagnostic on the
generator, never a gate" by §6's own sentence. An apple-sized mask is ~0.02 of the frame, far below
any plausible area bound, so :func:`check_mask` sees nothing wrong either. The empty-mask refusal
never fires, because the mask is not empty.

So every candidate mask is scored against the frame's own colour reference before the union, and one
that is essentially coincident with it is dropped: **a candidate that is the apple is not a robot.**
The threshold, where it came from and why its exact value is irrelevant over a 0.42-wide interval
are on :data:`ROBOT_MASK_OBJECT_MAX_IOU`.

Four properties of the fix, each of which is a decision rather than an implementation detail:

* **It fails LOUD, in the direction this file already fails.** Dropping every candidate leaves an
  all-False mask, and :func:`check_mask`'s existing "zero is zero" refusal takes the clip. There is
  no fallback that keeps a weak box to avoid a refusal — that is the same thing upstream's
  ``(0.10, 0.10)`` retry would have been, refused three paragraphs up for the same reason. The
  filter therefore makes G0c refuse MORE clips, never fewer, and it cannot manufacture a pass.
* **It cannot make G0c workable and does not pretend to.** ``DIAGNOSIS.json`` already concluded that
  with a median 152 robot-absent frames per episode "every clip refuses. G0c as written cannot
  produce a single composited clip on this corpus." That is unchanged and is a separate open
  decision. What this removes is the *other* outcome — the clips that would have been composited
  with the apple frozen, and passed.
* **It is a check on the OUTPUT, not on the detection.** The prompt, ``box_threshold``,
  ``text_threshold``, the absent retry and the union rule are all untouched; nothing is re-detected,
  re-prompted or re-drawn, and no mask is altered. All that is decided is whether a mask SAM 2
  already drew is admitted to the union. Same argument, same shape, as ``T40_RULE_V6`` §3 makes for
  ``apple_sam2.segment``.
* **It is not the IoU threshold §6 refuses.** §6's refused number is a *gate*: a pass/fail cut on
  the robot-mask IoU between source and generated, i.e. a verdict on the generator. This one is
  computed on the SOURCE frame alone, before any generated pixel exists, compares our own estimator
  against a non-learned second opinion, and its only possible effect is a refusal. It gates nothing
  and licenses nothing. It is still a number in this path, which is why it is pre-registered rather
  than merely commented — ``docs/preregistration/PR-08-V9-robot-mask-object-grounding.md``.

HARD EDGE, NO FEATHER, NO DILATION — AND THE ARGUMENT, BECAUSE THE BURDEN IS ON FEATHERING
-------------------------------------------------------------------------------------------
A binary mask leaves a one-pixel discontinuity where the source's robot meets the generated
background, and h264 will ring around it. That is a real cost and it is paid on purpose.

A feather is an alpha ramp across the boundary band, which means that in that band the output is a
blend of source and generated pixels. The band is the robot's SILHOUETTE. The generic-manipulator
defect is a defect of silhouette — the priors draw a plausible generic gripper where the G1's Dex3
hand should be (``docs/hf_jobs.md``), and the way that shows is the outline. So a feather admits the
defect back in, at reduced opacity, at the single location where it carries the most information,
while making it harder to see than it would have been without any composite at all. "The defect
cannot enter" and "the defect enters at alpha 0.5 along the hand's outline" are not the same
sentence.

Two further reasons, either of which would be enough on its own:

* **Any feather width is a coined number.** §6 opens "No threshold is coined" and refuses an IoU
  threshold on this very mask in its own text. A three-pixel ramp is a three-pixel decision about
  how much generated manipulator is acceptable, and nothing in the corpus derives it.
* **A seam is honest and a blend is not.** A hard edge produces a visible artifact that a reader can
  attribute to a known operation. A feathered edge produces a clip that looks better and contains
  something that is not in the source, in a region no downstream gate can measure.

The same argument forbids dilating the mask "to be safe": a dilation radius is a coined number too,
and it composites source pixels over generated BACKGROUND, which quietly shrinks the restyle by an
amount nobody chose. The mask is used exactly as the segmenter returned it.

WHAT IT REFUSES TO DO
---------------------
**It will not composite an empty mask.** A frame whose robot mask is empty is a frame where the
composite is the identity and the generated manipulator went straight into the corpus. That is the
one failure this gate is built to make impossible, so it is a refusal and not a warning, it fires
per frame, and it takes the clip with it. There is no number in this check: zero is zero.

**It will not decide by itself how much of a frame a robot may cover.** An over-large mask is the
other half of the same failure — a mask that has grounded on the table or the whole scene composites
the source back over everything, the "restyle" is a no-op, and arms B and C silently become arm A.
The bound on it is NOT coined here. PR-08 §6's own discipline is that the two geometry numbers are
"derived from the corpus itself", and the corpus statistic that would derive this one — the
distribution of robot-mask area fraction over the source frames — **has never been measured**. That
is the whole of the reason, and it is a statement about a measurement rather than about a machine:
the pinned GroundingDINO and SAM 2 checkpoints ARE staged (``102_stage_sam2_weights.sbatch`` put
them on the cluster and the local hub cache carries the same pinned revisions as of 2026-08-22), and
nobody has pointed them at the source frames and written down what came back. A staged checkpoint is
not a distribution. So this
module **refuses rather than coins**: :func:`load_area_bound` requires the committed artifact
``configs/transfer25/pr08_robot_mask_area.json``, that artifact is produced by this file's own
``measure`` mode, and the measure mode writes the DISTRIBUTION with ``max_frame_fraction: null`` —
it does not set the bound either.

That last point is deliberate and is worth the extra step. The natural "derived" bound is the
maximum area fraction observed over the source corpus, and it is useless: the composite runs on the
same frames the maximum was taken over, so the check can never fire. Any bound that CAN fire sits
above the observed maximum by a margin, and that margin is a coined number whichever way it is
dressed. There is no honest way for a script to pick it. So the measurement is automated, the
decision is not, and until somebody writes a number and their reasoning into that artifact this
pipeline refuses to generate. That is the intended behaviour: PR-08 §1 does not license generation
today anyway, and a G0c that quietly picked 0.6 would be a threshold pre-registered by a default.

**It will not composite frames it cannot pair.** Generated and source must have the same frame
count and the same grid, and the source's decoded length must match the manifest's ``frames``.
Compositing generated frame *i* with source frame *j* would put the robot from one instant into the
scene of another — geometry drift manufactured by the gate that exists to protect geometry, and G0b
would then score it as a generator defect.

**It will not leave an uncomposited file called ``vision.mp4``.** On any refusal the caller
quarantines the model's output under ``vision.uncomposited.mp4``. The harvest in
``97_transfer25_restyle.sbatch`` looks for ``vision.mp4`` plus a success status; renaming the file
means the first of those two independent conditions is false as well.

THE IoU IS A DIAGNOSTIC ON THE GENERATOR AND NEVER A GATE
----------------------------------------------------------
PR-08 §6 says this twice (once in G0c, once in V2 §0 and V3 §1's unchanged-gates tables), so it is
written into the record under a key that cannot be skim-read as anything else:
``robot_mask_iou_source_vs_generated.THIS_IS_A_DIAGNOSTIC_ON_THE_GENERATOR_AND_NEVER_A_GATE``.

It is measured between the robot mask of the source frame and the robot mask of the **raw generated
frame, before compositing**. After compositing the source's robot pixels ARE the generated frame's
robot pixels, so a post-composite IoU is 1.0 by construction and would be a measurement of this
file rather than of Cosmos-Transfer2.5. Nothing in this pipeline compares these numbers to a
threshold, and adding one would be an amendment to ``T40_RULE_V1``, not a code change.

THE COST, AND WHY THE MASK IS CACHED PER SOURCE EPISODE
--------------------------------------------------------
The composite needs one robot mask per source frame: ~172 000 frames for the corpus. The committed
partition asks for 25 style-instances of it, and a naive implementation would run GroundingDINO plus
SAM 2 four million times — far more compute than the generation it is protecting. It does not have
to: **the robot mask is a property of the SOURCE frame**, so it is identical across all 25
restyles of an episode. It is therefore computed once per source video and cached, keyed on the
source bytes, the prompt and the estimator version, so a cache entry can never be reused across a
changed input or a changed segmenter.

The IoU diagnostic cannot be cached — its second mask is of the generated frame — so it is sampled
on a stride. A stride is a coined number, and it is admissible here for exactly one reason: this
number gates nothing, so a sampling rate cannot become a finding. The COMPOSITE has no stride and
never will.

MEASURING THE DISTRIBUTION TAKES LONGER THAN THE WALL, SO IT IS SHARDED AND MERGED
------------------------------------------------------------------------------------
``measure`` is one GroundingDINO forward plus one SAM 2 forward per frame over 171 625 source
frames, one frame at a time. Every Discoverer+ QoS caps at a four-hour MaxWall, and the measurement
does not fit inside it with any margin, so ``--shard I --num-shards N`` measures the episodes that
hash to one shard and ``--merge`` pools the shards. ``cluster/discoverer/106_measure_robot_mask_area
.sbatch`` drives both, and ``scripts/measure_geom_tol.py`` solved the same arithmetic first — the
partition rule, the refusals and the shape of the merge are deliberately the same, because two
different sharding designs in one repository is two things to get right instead of one.

**THREE OF THE FIVE NUMBERS DO NOT DECOMPOSE, WHICH IS THE WHOLE REASON --merge EXISTS.** The
artifact reports min, median, p95, p99 and max. Only the first and the last recombine from per-shard
summaries; the median of the shard medians is a different statistic with the same units and a
plausible magnitude, and the same is true of a p95 and a p99. A bound will sit above these numbers
and its written rationale will quote them, and nothing downstream re-derives either — so that error
would be permanent and invisible. Shards therefore emit the RAW per-frame area fractions and the
merge takes the five numbers ONCE over the pooled population, rebuilt in the manifest's own
enumeration order, so the merged artifact is identical to what a single un-sharded run would have
written. That is asserted rather than claimed, in ``tests/test_robot_composite_shards.py``.

**A MERGE CANNOT LAUNDER ``measurement_qualified``.** The six conditions in
:data:`MERGE_CONDITIONS` are evaluated and written into the artifact per condition: the shards tile
the corpus exactly once, every shard ran at stride 1 with no limit, every shard calls itself
qualified, and the shards agree on the estimator, the source manifest and the prompt. Any false one
stamps ``measurement_qualified: false`` with the reasons, :func:`load_area_bound` refuses the file
by name, and the exit code is non-zero. Inputs that cannot be pooled AT ALL — a duplicated shard, an
episode in a shard it does not hash to, a shard that kept only its own summary — are refused with
nothing written, because there the pool itself would be wrong rather than incomplete.

Neither mode coins the bound. ``max_frame_fraction`` is null on every path here, merge included.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The robot text prompt, COMMITTED HERE AND NOWHERE ELSE. No environment override, no flag — see
#: the module docstring: a per-run prompt is a per-run decision about which pixels the generator is
#: allowed to touch, and it is invisible in the output.
#:
#: Three phrases rather than one because GroundingDINO grounds phrases, and this view carries an
#: arm, a hand and a gripper that a single noun does not reliably cover. Period-terminated and
#: lowercase is GroundingDINO's documented input format, not a preference — ``apple_sam2`` makes the
#: same normalisation and argues it there. The phrase set's ADEQUACY on this corpus is unproven and
#: the ``measure`` mode below is what would show it, frame by frame, before the first clip.
ROBOT_TEXT_PROMPT = "robot arm. robotic hand. robotic gripper."

# Checked at import rather than assumed, and checked HERE rather than by calling
# ``apple_sam2.normalize_prompt`` — that module reaches transformers and sam2 at import time, and
# this one is imported by the driver on machines that have neither. The rule it restates is
# GroundingDINO's documented phrase format, and ``apple_sam2`` argues at length why getting it wrong
# is expensive: "Apple" or "apple" without the period parse differently in its text encoder and
# yield fewer detections. Fewer robot detections is a smaller mask, and a smaller mask is generated
# manipulator left in the frame — the defect G0c exists to exclude, arriving through a capital
# letter.
assert ROBOT_TEXT_PROMPT == ROBOT_TEXT_PROMPT.strip().lower(), ROBOT_TEXT_PROMPT
assert ROBOT_TEXT_PROMPT.endswith("."), ROBOT_TEXT_PROMPT

#: THE ROBOT-MASK VALIDITY FILTER'S ONE NUMBER. A detection whose SAM 2 mask overlaps the frame's
#: apple this much IS the apple, and is dropped before the union — see the module docstring's
#: "A DETECTION THAT IS THE APPLE IS NOT A ROBOT" section for why, and PR-08 V9 for the licence.
#:
#: HOW 0.70 WAS CHOSEN, AND FROM WHICH MEASUREMENT. Not tuned, and not a midpoint of nothing: it is
#: a value read off a measured gap, the same defence ``apple_sam2.MASK_VALIDITY_MIN_IOU`` makes and
#: for the same reason — a number introduced into a gate path must not be able to become the
#: finding. Every GroundingDINO box above the committed operating point on the 710 frames of
#: ``runs/pr08-robot-mask-empty/plan_corpus.json`` (40 episodes, the stratified plan the ABSENT
#: diagnosis was produced from) was segmented and scored against
#: ``apple_sam2.object_color_reference``. 2 845 detections, and the two populations do not touch:
#:
#:     apple detections   :  IoU in [0.9364, 0.9847]   (146 of them; the mask IS the colour region)
#:     everything else    :  IoU <= 0.5131             (2 699 of them; robot, plate, tablecloth,
#:                                                      and gripper-over-apple boxes at 0.19-0.51)
#:
#: Nothing at all lands in (0.5131, 0.9364), so EVERY cut in that open interval produces the
#: identical partition of those 2 845 detections, and 0.70 sits inside it with 0.187 of margin
#: below and 0.236 above. ``tests/test_robot_composite_object_filter.py`` sweeps the interval
#: against the measured IoUs and asserts the partition never moves, and asserts that the value this
#: module ships lies strictly inside it — so the insensitivity is checked rather than claimed.
#:
#: The gap is not an accident of this sample, it is what the two shapes are. A mask of the apple and
#: the warm-saturated colour predicate are two outlines of one object, so they agree at ~0.95. A
#: mask of anything else on this corpus — the robot is black and bare metal, the cloth and the plate
#: are neutral to within two counts (``apple_sam2.object_color_reference``'s own note) — contains
#: essentially none of those pixels. The only in-between shape is a real robot detection whose box
#: also swallows the fruit during a grasp, and those are the 0.19-0.51 tail: KEPT, correctly, and
#: the measurement shows dropping the apple box beside them removes 140 px of a 31 710 px mask.
#:
#: NO ENVIRONMENT OVERRIDE AND NO FLAG, for ``ROBOT_TEXT_PROMPT``'s reason exactly. This decides
#: which pixels the generator is allowed to touch. A per-run value would be a per-run decision about
#: that, taken on a submit line, recorded nowhere anybody would look, and invisible in the output.
#: Moving it means editing this file, :data:`SEGMENTER_IDENTITY_FIELDS`' consequences (every cached
#: mask and any committed area bound stop matching) and a pre-registration, together.
ROBOT_MASK_OBJECT_MAX_IOU = 0.70

#: The committed artifact carrying the area bound, tracked rather than under ``runs/`` for
#: ``measure_geom_tol``'s reason: ``runs/`` is gitignored, so an artifact written there can never be
#: the pre-commitment the rule asks for.
AREA_BOUND_ARTIFACT = REPO_ROOT / "configs" / "transfer25" / "pr08_robot_mask_area.json"

#: Every field the bound artifact must carry before it may be used. A file that says only
#: ``{"max_frame_fraction": 0.6}`` is a coined number in a committed file's clothing: it cannot say
#: which segmenter measured the distribution the bound sits above, on how many frames, or of which
#: corpus, so nothing downstream could tell it apart from a guess. Same discipline as
#: ``measure_geom_tol.CROSS_CHECK_FIELDS_REQUIRED``.
AREA_BOUND_FIELDS_REQUIRED = (
    "max_frame_fraction",
    "bound_rationale",
    "measured",
    "measurement_qualified",
    "estimator",
    "prompt",
    "source_manifest_sha256",
)

#: The fields of a masker's ``provenance()`` that IDENTIFY the segmenter — everything that changes
#: which pixels come back for a given frame. :func:`segmenter_identity` is the one definition, and
#: it has two consumers on purpose: :meth:`MaskCache.key` (a cached mask may not survive any of
#: these changing) and :func:`load_area_bound`'s cross-check (a committed area bound may not survive
#: any of these changing either). Those two used to be independent, and the failure that cost is
#: exactly the one PR-13 is about: two copies of the same measurement drifting apart. Re-pinning
#: GroundingDINO correctly invalidated every cached mask while the bound measured under the OLD
#: weights was silently reused — and a bound that no longer matches its distribution either never
#: fires (over-large masks composite the source back over the whole frame, the restyle is a no-op,
#: arms B and C become arm A) or fires on everything. One tuple, two readers, no drift.
SEGMENTER_IDENTITY_FIELDS = (
    "version",
    "prompt",
    "box_threshold",
    "text_threshold",
    "box_rule",
    "upstream_retry_not_run",
    # The object-grounding filter changes WHICH DETECTIONS SURVIVE, so it changes the mask for a
    # given frame, so it belongs here by this tuple's own definition. Both consequences are the
    # intended ones and neither is a side effect to be worked around: a mask cached before the
    # filter existed is a different mask and must not be reused, and an area-fraction distribution
    # measured before it existed is a distribution of a different masker and must not be sat above.
    # (No such bound exists today — configs/transfer25/pr08_robot_mask_area.json is not in the tree
    # — so nothing committed is invalidated by adding this; it is here so that nothing committed
    # LATER can be reused across the filter being changed or removed.)
    "object_grounding_filter",
)


def segmenter_identity(provenance: dict) -> dict:
    """The identifying subset of a ``provenance()`` dict, missing fields included as ``None``.

    Missing keys are kept rather than dropped so that a provenance which stopped declaring one of
    them compares UNEQUAL to one that declared it, instead of comparing equal by absence. Absence-
    permissive comparison is the failure ``measure_geom_tol``'s cross-check was repaired for.
    """
    return {field: provenance.get(field) for field in SEGMENTER_IDENTITY_FIELDS}

#: libx264 at a near-visually-lossless CRF. The composite is exact in the ARRAY — see
#: :func:`composite_frame` — and the re-encode is the same lossy step every generated clip already
#: pays, but it is a SECOND one for the generated pixels (model encode, our decode, our encode), so
#: the quality knob is set high and named rather than left at a library default.
ENCODE_CODEC = "libx264"
ENCODE_FFMPEG_PARAMS = ("-crf", "10")

#: Exit codes of the ``measure`` mode, distinct because a shell has to tell three outcomes apart
#: without parsing JSON, and the middle one is the one that gets mistaken for success. Same spelling
#: as ``measure_geom_tol``'s (0 / non-zero fatal / 3 not gate-qualified) so that an operator who has
#: run one recognises the other.
EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_MEASUREMENT_NOT_QUALIFIED = 3


class CompositeError(RuntimeError):
    """A refusal, as opposed to a crash. Carries a message meant for the operator.

    Deliberately not a subclass of ``restyle_transfer25.DriverError``: this module is imported by
    that driver and not the other way round. The driver translates.
    """


# --------------------------------------------------------------------------------------------
# video in, video out
# --------------------------------------------------------------------------------------------
#
# A thin decoder rather than an import of ``scripts/score_generated_video.py``, whose ``_decode`` is
# the pattern followed here down to the AV1 fallback. That module imports
# ``wam.evaluation.video_fidelity`` at module scope, which drags the whole scoring stack — and
# torch — into a driver that runs inside the Transfer2.5 venv on a GPU node and has no use for a
# single metric in it. The duplication is eight lines; the coupling would not be.


def _decode_frames(path: pathlib.Path) -> Iterator[np.ndarray]:
    """Yield RGB uint8 frames. cv2 first, imageio's bundled ffmpeg as the AV1 fallback.

    GR00T ships AV1 and the pip ``opencv-python`` wheel's FFmpeg cannot decode it: cv2 opens the
    file happily and then reads ZERO frames, which is a silent empty clip rather than an error. The
    same workaround is in ``convert_lerobot_g1.py``, ``hf_job_wan_probe.py`` and
    ``score_generated_video.py``; the source corpus is exactly the AV1 case.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    produced = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            produced += 1
            yield np.ascontiguousarray(bgr[:, :, ::-1])
    finally:
        capture.release()
    if produced:
        return
    import imageio.v3 as iio

    for rgb in iio.imiter(str(path), plugin="FFMPEG"):
        yield np.ascontiguousarray(np.asarray(rgb))


def decode_clip(path: pathlib.Path) -> np.ndarray:
    """Whole clip as uint8 RGB ``[F, H, W, 3]``. Refuses an empty decode rather than returning it.

    A zero-frame decode is the AV1-into-cv2 failure above and it is indistinguishable from a
    zero-length video. Either way, a clip with no frames that reached the composite silently is a
    clip that was never composited.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        raise CompositeError(f"no such video: {path}")
    frames = list(_decode_frames(path))
    if not frames:
        raise CompositeError(
            f"{path} decoded to zero frames. cv2 opens a file it cannot decode and then reads "
            "nothing (the corpus is AV1), so this is reported rather than treated as an empty clip."
        )
    stacked = np.stack(frames)
    if stacked.ndim != 4 or stacked.shape[3] != 3 or stacked.dtype != np.uint8:
        raise CompositeError(
            f"{path} decoded to {stacked.shape} {stacked.dtype}; this path handles uint8 RGB only."
        )
    return stacked


def container_fps(path: pathlib.Path) -> float | None:
    """The rate the container declares, or None. Used for the re-encode only, never to align.

    Frames are paired by INDEX here, because the labels are paired by index; a container rate is
    metadata a writer chose. It matters only so that the composited file plays at the rate the
    source did.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        value = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    return value if value and value > 0 else None


def encode_clip(frames: np.ndarray, path: pathlib.Path, fps: float) -> None:
    """Write ``[F, H, W, 3]`` uint8 RGB, then decode it back and check what landed.

    The read-back is not paranoia about ffmpeg. ``imageio``'s writer pads frames up to a multiple of
    its ``macro_block_size`` unless told not to, and a silently padded clip is a clip whose pixels no
    longer line up with the source's — the same off-by-geometry failure the frame-count check below
    refuses, arriving through the encoder instead of through the model. ``macro_block_size=1`` turns
    the padding off and this asserts that it did.
    """
    import imageio.v3 as iio

    arr = np.asarray(frames)
    iio.imwrite(
        str(path),
        arr,
        fps=fps,
        codec=ENCODE_CODEC,
        ffmpeg_params=list(ENCODE_FFMPEG_PARAMS),
        macro_block_size=1,
    )
    back = decode_clip(path)
    if back.shape != arr.shape:
        raise CompositeError(
            f"the composited clip was written as {arr.shape} and decodes as {back.shape}. The "
            "encoder changed the geometry or the frame count, so the pixels no longer line up with "
            "the actions carried over from the source recording."
        )


# --------------------------------------------------------------------------------------------
# the mask
# --------------------------------------------------------------------------------------------


class Sam2RobotMasker:
    """The robot mask, from ``scripts/estimators/apple_sam2.py``'s pinned GroundingDINO + SAM 2.

    Everything model-shaped is reached through that module: its detector, its predictor, its device
    resolution, its uint8 contract, its detection thresholds, its offline enforcement and its
    refusals. What differs is three things, all argued in the module docstring: the prompt is
    :data:`ROBOT_TEXT_PROMPT` instead of its ``OBJECT_TEXT_PROMPT``, every box above threshold is
    segmented and OR-ed instead of only the highest-scoring one, and a candidate mask that IS the
    apple is dropped before that union (:data:`ROBOT_MASK_OBJECT_MAX_IOU`).

    The import is lazy — inside ``_estimator()`` — because ``apple_sam2`` reaches ``transformers``
    and ``sam2`` at import time and this module is imported by the driver, whose other paths and
    whose tests must run on a machine with neither.

    :attr:`filter_counters` accumulate over the masker's life; :meth:`filter_record` reads them and
    :func:`composite_clip` differences it, because a filter whose firing is not recorded cannot be
    told apart from a corpus that never triggered it.
    """

    #: Cumulative, and never reset by anything here. Differenced by the caller.
    _COUNTERS = (
        "frames_masked",
        "detections_segmented",
        "detections_dropped_as_object",
        "frames_with_a_dropped_detection",
        "frames_emptied_by_the_filter",
        "frames_with_no_object_reference",
    )

    def __init__(self) -> None:
        self._module: Any = None
        self.filter_counters: dict[str, int] = dict.fromkeys(self._COUNTERS, 0)

    # -- what the filter did ---------------------------------------------------------------------

    def filter_record(self) -> dict:
        """The counters plus the constants they were produced under, as one readable block."""
        return {
            "rule": (
                "a detection whose SAM 2 mask has IoU > max_iou against the frame's own colour "
                "reference IS that object and is dropped before the union; if that leaves nothing, "
                "the mask is empty and check_mask refuses the clip"
            ),
            "max_iou": float(ROBOT_MASK_OBJECT_MAX_IOU),
            "reference": self._object_reference_name(),
            **{name: int(self.filter_counters[name]) for name in self._COUNTERS},
        }

    def _object_reference_name(self) -> str:
        module = self._estimator()
        name = getattr(module, "MASK_VALIDITY_REFERENCE", None)
        if not name:
            raise CompositeError(
                "scripts/estimators/apple_sam2.py no longer declares MASK_VALIDITY_REFERENCE, so "
                "this module cannot say WHICH second opinion decided that a detection was the "
                "apple rather than the robot. Every G0c record makes that claim; an unnameable "
                "predicate in 10 050 records is worse than a refusal here."
            )
        return str(name)

    # -- the pinned pair, loaded once ----------------------------------------------------------

    def _estimator(self) -> Any:
        if self._module is None:
            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
            import estimators.apple_sam2 as apple_sam2  # type: ignore

            self._module = apple_sam2
        return self._module

    def preflight(self) -> None:
        """Load both models NOW, before the first clip, and refuse here if they are not staged.

        ``apple_sam2.segment`` already loads both models before its first detection so that a
        machine missing SAM 2's checkpoint refuses on frame 0 rather than on the first frame that
        happens to contain something. This is the same argument one level up: a missing checkpoint
        is a fact about the RUN, not about a unit, and discovering it inside the per-unit guard
        would turn one run-level refusal into N identical per-unit errors that look like a flaky
        generator and burn a pass of the chunk's rail.
        """
        try:
            module = self._estimator()
            module._detector()
            module._predictor()
        except CompositeError:
            raise
        except Exception as exc:  # noqa: BLE001 — translated, not swallowed; the text is kept whole
            # ``apple_sam2`` raises ``EstimatorDependencyMissing``, an ImportError subclass carrying
            # a multi-paragraph diagnosis that names every checkpoint, every pinned revision and
            # every cache directory it looked in. Letting that escape ``main`` would print it as a
            # traceback, which the sbatch reads as "the driver crashed" rather than as a refusal it
            # can act on. Re-raised as the one type this module promises so the driver's translation
            # turns it into "FATAL: <the whole diagnosis>" and exit 1.
            raise CompositeError(
                "the robot segmenter could not be loaded, so PR-08 §6 G0c cannot composite and "
                "nothing may be generated:\n"
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def provenance(self) -> dict:
        """What produced the masks, in the words the per-unit record will carry.

        ``version`` IS BUILT HERE RATHER THAN COPIED FROM ``apple_sam2.ESTIMATOR_VERSION``, and the
        reason is a lie that would otherwise land in every clip's record. That string embeds the
        adapter's own ``OBJECT_TEXT_PROMPT`` — ``'apple.'`` — because it is written for the GEOM_TOL
        and EST_DRIFT_P95 measurements, which segment the apple. These masks are of the ROBOT. A
        record whose ``version`` says ``prompt='apple.'`` next to a ``prompt`` field saying
        ``'robot arm. …'`` is worse than one that says neither: it invites the reader to believe the
        apple prompt produced the robot mask, and there is no way to tell from the artifact which
        one is the truth. So the two checkpoints that actually made this mask are quoted directly
        from the adapter's pins, the prompt beside them is OURS, and the adapter's own string is
        carried unchanged under ``adapter_version`` so nothing is hidden.

        The depth checkpoint is deliberately absent: it makes no mask and naming it here would
        suggest the composite depends on it.

        ``version`` is also part of the mask cache key, so re-pinning either checkpoint invalidates
        every cached mask — which is the point of putting the revisions in it rather than the ids.

        ``upstream_retry_not_run`` IS THE OTHER DIVERGENCE FROM THE PINNED ADAPTER, WRITTEN DOWN.
        The module docstring argues it: upstream's single retry at ``(0.10, 0.10)`` recovers a
        dropped frame for ``apple_sam2``'s callers and would suppress a REFUSAL here, buying a weak
        detection that ``_best_box`` itself says can land on something else. This file takes the
        refusal — and says so, because a mask made by a stricter detector than the one PR-08 §4
        step 2 pins is not the same mask, and a reader of one clip's record must be able to see
        that without reading this file. The two upstream thresholds are quoted from the adapter so
        the claim stays checkable; if the adapter stops declaring them, that claim is no longer
        checkable and this refuses rather than recording a sentence about a retry it can no longer
        name. The string is in the cache key: reversing this decision invalidates every cached mask,
        which is correct, because every one of them would be a different mask.
        """
        module = self._estimator()
        retry_box = getattr(module, "RETRY_BOX_THRESHOLD", None)
        retry_text = getattr(module, "RETRY_TEXT_THRESHOLD", None)
        if retry_box is None or retry_text is None:
            raise CompositeError(
                "scripts/estimators/apple_sam2.py no longer declares RETRY_BOX_THRESHOLD / "
                "RETRY_TEXT_THRESHOLD, so this module cannot record WHICH upstream retry it "
                "deliberately does not run (see the robot_composite docstring's box-rule section). "
                "Every G0c record makes that claim about the segmenter; an unverifiable claim in "
                "10 050 records is worse than a refusal here. Re-read the adapter and update "
                "Sam2RobotMasker.provenance()."
            )
        retry_note = (
            f"upstream retries once at ({retry_box}, {retry_text}) when the first pass grounds "
            "nothing; THIS masker does not. apple_sam2's callers DROP an ungrounded frame, so the "
            "retry recovers data there; G0c REFUSES the clip, so here it would only suppress a "
            "refusal by accepting a weak box that can land on something other than the robot — a "
            "confident wrong mask in place of a loud refusal."
        )
        return {
            "name": "grounding-dino+sam2 (estimators.apple_sam2 pins, robot prompt)",
            "version": (
                f"det={module.GROUNDING_DINO_MODEL_CHECKPOINT}"
                f"@{module.GROUNDING_DINO_MODEL_REVISION};"
                f"seg={module.SAM2_MODEL_CHECKPOINT}@{module.SAM2_MODEL_REVISION};"
                f"prompt={ROBOT_TEXT_PROMPT!r};"
                f"box_thr={module.BOX_THRESHOLD};text_thr={module.TEXT_THRESHOLD};"
                f"retry=none(upstream={retry_box}/{retry_text})"
            ),
            "adapter": "estimators.apple_sam2",
            "adapter_version": str(getattr(module, "ESTIMATOR_VERSION", "unversioned")),
            "adapter_version_note": (
                "the adapter's own version string, whose prompt= field is its APPLE prompt because "
                "that is what GEOM_TOL and EST_DRIFT_P95 segment. The robot mask's prompt is the "
                "'prompt' field above."
            ),
            "prompt": ROBOT_TEXT_PROMPT,
            "box_threshold": float(module.BOX_THRESHOLD),
            "text_threshold": float(module.TEXT_THRESHOLD),
            "box_rule": "union of every detection above threshold (see robot_composite docstring)",
            "upstream_retry_not_run": retry_note,
            # THE THIRD DIVERGENCE FROM THE PINNED ADAPTER, WRITTEN DOWN, for the same reason as the
            # one above it: this changes which pixels come back from the source for a given frame,
            # so a reader of one clip's record must be able to see it without reading this file. It
            # is in SEGMENTER_IDENTITY_FIELDS and therefore in the mask cache key: turning the
            # filter off, or moving its number, invalidates every cached mask, which is correct,
            # because every one of them would be a different mask.
            "object_grounding_filter": (
                f"a detection whose SAM 2 mask has IoU > {ROBOT_MASK_OBJECT_MAX_IOU} against "
                f"{self._object_reference_name()} is the OBJECT, not the robot, and is dropped "
                "before the union; if that empties the mask, check_mask refuses the clip. "
                "PR-08 V9. The detection operating point, the prompt and the union rule are "
                "unchanged and no mask is altered."
            ),
        }

    # -- one frame -----------------------------------------------------------------------------

    def _boxes(self, frame: np.ndarray) -> np.ndarray:
        """Every GroundingDINO box for :data:`ROBOT_TEXT_PROMPT`, as ``[N, 4]``; ``[0, 4]`` if none.

        ``apple_sam2._best_box`` cannot be reused: it reads that module's ``OBJECT_TEXT_PROMPT``
        global and returns the single highest-scoring box. Both are right for the apple and wrong
        here — see the module docstring's box-rule paragraph. The processor call is otherwise
        identical, ``threshold=`` spelling included, because a divergence there would be a different
        detection threshold from the one ``ESTIMATOR_VERSION`` records.

        ONE PASS, AND UPSTREAM'S RETRY IS NOT ONE OF THEM. ``_best_box`` post-processes a second
        time at ``(RETRY_BOX_THRESHOLD, RETRY_TEXT_THRESHOLD)`` when the first pass grounds nothing;
        this deliberately does not, and :meth:`provenance` records the omission under
        ``upstream_retry_not_run`` so it is in every clip's record and in the mask cache key. The
        argument is in the module docstring and it turns on the caller: an ungrounded frame is
        DROPPED by ``apple_sam2``'s callers and REFUSED here, so the retry recovers data there and
        would suppress a refusal here. What it would put in its place is what ``_best_box``'s own
        docstring warns about — a weak box that "lands on something else", i.e. a confident wrong
        robot mask, which composites source pixels over a non-robot region and leaves the generated
        manipulator standing. That is the defect entering silently instead of a refusal being
        printed. This makes the robot masker stricter than the pinned adapter, which is a real
        divergence from "the same segmenter" and is why it is written into the record rather than
        left to be rediscovered from the code.

        ``module.BOX_THRESHOLD`` and ``module.TEXT_THRESHOLD`` are read live rather than copied, so
        this tracks the adapter's operating point (0.15/0.25 since 2026-08-22, read off
        Cosmos-Transfer2.5's own ``sam2_model.py``). They are also settable from the environment
        there, which is why ``97_transfer25_restyle.sbatch`` unsets those variables before the
        driver runs: a detection threshold typed into a submit script is a per-run decision about
        which pixels the generator is allowed to touch, which is exactly what this module's
        docstring refuses for the prompt.
        """
        import torch
        from PIL import Image

        module = self._estimator()
        processor, model = module._detector()
        h, w = frame.shape[:2]
        inputs = processor(
            images=Image.fromarray(frame), text=ROBOT_TEXT_PROMPT, return_tensors="pt"
        ).to(module._device())
        with torch.inference_mode():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=module.BOX_THRESHOLD,
            text_threshold=module.TEXT_THRESHOLD,
            target_sizes=[(h, w)],
        )[0]
        boxes = np.asarray(results["boxes"].detach().cpu(), dtype=np.float64).reshape(-1, 4)
        return boxes

    def object_grounding_iou(self, frame: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """``(N,)`` float: each candidate mask's IoU against the frame's own colour reference.

        Both halves come from the pinned adapter — ``object_color_reference`` is the warm-and-
        saturated predicate ``T40_RULE_V6`` already runs on every ``segment()`` call, and
        ``mask_validity_iou`` is the symmetric IoU it scores with. They are reached rather than
        restated so that there is exactly one definition of "this region is the apple" in the
        repository: two copies of a discriminator drifting apart is the failure PR-13 is about, and
        this one would drift silently, because the two callers never compare their answers.

        Symmetric IoU rather than "how much of the candidate is apple-coloured", deliberately. The
        one-sided containment ratio would also drop the whole-tablecloth masks — they contain the
        fruit — and those are :func:`check_mask`'s area bound's business, not this filter's. A
        filter that quietly took over another check's failure mode would turn an over-large-mask
        refusal into an empty-mask refusal and change what the operator is told.
        """
        module = self._estimator()
        for name in ("object_color_reference", "mask_validity_iou"):
            if getattr(module, name, None) is None:
                raise CompositeError(
                    f"scripts/estimators/apple_sam2.py no longer declares {name}(), which is the "
                    "second opinion this module uses to tell a robot detection from a detection "
                    "that has grounded on the apple (PR-08 V9). Without it every robot-absent "
                    "frame would silently composite the SOURCE apple over the generated one, which "
                    "no PR-08 gate can see. Re-read the adapter and update "
                    "Sam2RobotMasker.object_grounding_iou()."
                )
        reference = module.object_color_reference(frame)
        if not reference.any():
            self.filter_counters["frames_with_no_object_reference"] += 1
        return np.asarray(
            [float(module.mask_validity_iou(m, reference)) for m in masks], dtype=np.float64
        )

    def mask(self, rgb: np.ndarray) -> np.ndarray:
        """``(H, W)`` bool: every pixel this frame's robot occupies. All-False when nothing grounds.

        All-False is returned rather than raised because the caller is what decides what an empty
        robot mask means, and here it means the clip is refused — see :func:`check_mask`. Making the
        masker raise would conflate "the segmenter could not run" with "the segmenter ran and found
        no robot", and those two need different messages.

        **A detection that is the apple is dropped before the union**, at
        :data:`ROBOT_MASK_OBJECT_MAX_IOU`, and the module docstring argues why. Note the ordering:
        the drop happens BEFORE the OR, per candidate, not on the finished union. On a grasp frame
        the detector returns both real robot boxes and a box on the fruit; filtering the union would
        have to choose between discarding the robot and admitting the apple, and per-candidate
        filtering has to do neither. Dropping every candidate is how this returns all-False, which
        is the loud path — there is no fallback that keeps the best-scoring reject.
        """
        module = self._estimator()
        frame = module._as_uint8_rgb(rgb)
        h, w = frame.shape[:2]
        self.filter_counters["frames_masked"] += 1

        boxes = self._boxes(frame)
        if boxes.shape[0] == 0:
            return np.zeros((h, w), dtype=bool)

        import torch

        predictor = module._predictor()
        with torch.inference_mode():
            predictor.set_image(frame)
            masks, _scores, _logits = predictor.predict(box=boxes, multimask_output=False)
        stacked = np.asarray(masks).reshape(-1, h, w) > 0
        self.filter_counters["detections_segmented"] += int(stacked.shape[0])

        overlaps = self.object_grounding_iou(frame, stacked)
        keep = overlaps <= ROBOT_MASK_OBJECT_MAX_IOU
        dropped = int(stacked.shape[0] - np.count_nonzero(keep))
        if dropped:
            self.filter_counters["detections_dropped_as_object"] += dropped
            self.filter_counters["frames_with_a_dropped_detection"] += 1
            if not keep.any():
                # Every candidate was the apple, so this frame has no robot detection left and the
                # mask is empty. check_mask refuses the clip on it, by name and with no threshold —
                # which is the intended outcome and the whole reason nothing weaker is admitted
                # here to avoid it.
                self.filter_counters["frames_emptied_by_the_filter"] += 1
                return np.zeros((h, w), dtype=bool)
        return np.ascontiguousarray(np.any(stacked[keep], axis=0))


def build_masker() -> Sam2RobotMasker:
    """The one and only robot masker. Takes no arguments, ON PURPOSE.

    Every parameter that could weaken a mask — the prompt, the box rule, the thresholds, the
    checkpoints — is either a committed constant here or a pin in ``apple_sam2``. This factory takes
    nothing because there is nothing to choose: an argument here would be the configuration surface
    the module docstring refuses to open, and ``restyle_transfer25`` would then have to decide what
    to pass, which puts the decision on the command line after all.
    """
    return Sam2RobotMasker()


# --------------------------------------------------------------------------------------------
# the mask cache
# --------------------------------------------------------------------------------------------


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class MaskCache:
    """Robot masks for one source video, stored bit-packed and keyed on everything that could change.

    The key is the source file's sha256 plus the prompt plus the estimator version. Keying on the
    path would be the cheap version and it would be wrong twice over: a regenerated source corpus
    reuses the same paths, and a re-pinned SAM 2 produces different masks from the same bytes. A
    stale mask reused across either change composites the wrong pixels — the failure mode with no
    symptom, because the output still looks like a robot.

    ``np.packbits`` because 427 frames of 640x480 bool is 131 MB unpacked and 16 MB packed, and this
    file is written once per source episode and read 25 times.
    """

    def __init__(self, root: pathlib.Path) -> None:
        self.root = pathlib.Path(root)

    def _entry(self, key: str) -> pathlib.Path:
        return self.root / f"{key}.npz"

    @staticmethod
    def key(source_video: pathlib.Path, provenance: dict) -> str:
        payload = json.dumps(
            {
                "source_sha256": _file_sha256(pathlib.Path(source_video)),
                # The SAME tuple ``load_area_bound`` cross-checks the committed bound against, so a
                # segmenter change that invalidates a cached mask cannot leave the bound standing.
                "segmenter": segmenter_identity(provenance),
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, key: str, shape: tuple[int, int, int]) -> np.ndarray | None:
        entry = self._entry(key)
        if not entry.is_file():
            return None
        with np.load(entry) as data:
            packed, stored = data["packed"], tuple(int(x) for x in data["shape"])
        if stored != tuple(shape):
            # Same bytes, same segmenter, different frame count: the decoder saw something else
            # this time. Recompute rather than reshape — a mask array that has to be reshaped to fit
            # is not this clip's mask.
            return None
        return np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape).astype(bool)

    def put(self, key: str, masks: np.ndarray) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        # ``.tmp.npz`` and not ``.npz.tmp``: ``np.savez_compressed`` APPENDS ``.npz`` to any name
        # that does not already end in it, so the second spelling writes ``…npz.tmp.npz`` and the
        # rename that follows fails on a file that was never there.
        #
        # THE PID IS IN THE NAME BECAUSE THIS DIRECTORY IS DELIBERATELY SHARED. 97's generation
        # invocation passes one ``--mask-cache ${OUT}/robot_masks`` for the whole RUN_ID so the
        # train, eval and identity submissions reuse each other's masks, and those are separate
        # jobs restyling the SAME source episodes — so they compute the SAME key at the same time.
        # With a fixed temporary name two writers share one file: the second truncates what the
        # first is still writing and both then rename the torn result into place. The entry that
        # lands is a zip nobody can read, every clip that then hits it fails its unit, and the
        # symptom looks like a flaky generator rather than like a cache. Per-writer temporaries plus
        # an atomic rename make the loser of the race harmless — both wrote the same masks, because
        # the key is a hash of the source bytes and the segmenter.
        tmp = self._entry(key).with_name(f"{key}.{os.getpid()}.tmp.npz")
        np.savez_compressed(
            tmp, packed=np.packbits(masks.astype(bool)), shape=np.asarray(masks.shape)
        )
        tmp.replace(self._entry(key))


# --------------------------------------------------------------------------------------------
# the area bound — measured, never coined
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AreaBound:
    """The largest fraction of a frame a robot mask may cover before the clip is refused.

    ``cross_checked`` is the difference between a bound that has been VALIDATED (its shape is right,
    its number is in range, it carries a rationale) and one that has been shown to belong to THIS
    run — same segmenter, same source corpus. Only the second may composite, and that is enforced by
    :class:`CompositeContext` rather than by remembering to ask: see :func:`load_area_bound`.
    """

    max_frame_fraction: float
    artifact: pathlib.Path
    artifact_sha256: str
    rationale: str
    cross_checked: bool = False
    cross_checked_against: dict | None = None

    def record(self) -> dict:
        return {
            "max_frame_fraction": self.max_frame_fraction,
            "artifact": str(self.artifact),
            "artifact_sha256": self.artifact_sha256,
            "rationale": self.rationale,
            # In every clip's record, because "the bound was checked against the segmenter that
            # actually made these masks" is a per-clip claim and a reader of one clip's
            # sample_outputs.json must be able to see it without holding the whole run.
            "cross_checked": self.cross_checked,
            "cross_checked_against": self.cross_checked_against,
        }


def area_bound_missing_message(path: pathlib.Path, why: str) -> str:
    """The refusal, written so the reader knows exactly what to measure and what to decide.

    A refusal that names no remedy is how a project ends up with a default: the next person needs a
    number, finds no instructions, and picks one.
    """
    return "\n".join([
        f"FATAL: no usable robot-mask area bound at {path}.",
        f"       {why}",
        "",
        "       WHAT THIS BOUND IS. PR-08 §6 G0c composites the real robot's pixels back over every",
        "       generated frame. A robot mask that has grounded on the table, or on the whole",
        "       scene, composites the SOURCE back over everything: the restyle becomes a no-op, and",
        "       arms B and C silently become arm A. Refusing such a frame needs a number, and §6",
        "       opens with 'No threshold is coined.'",
        "",
        "       WHY THIS FILE WILL NOT PICK ONE. The corpus statistic that would derive it — the",
        "       distribution of robot-mask area fraction over the SOURCE frames — HAS NEVER BEEN",
        "       MEASURED. Not 'cannot be': the pinned GroundingDINO and SAM 2 checkpoints are",
        "       staged (102_stage_sam2_weights.sbatch on the cluster; the same pinned revisions in",
        "       the local hub cache). Nobody has pointed them at the source frames and written down",
        "       what came back, and a staged checkpoint is not a distribution. The obvious derived",
        "       bound is useless anyway: the maximum observed over the source corpus can never be",
        "       exceeded by a composite that runs on those same frames, so a bound that CAN fire",
        "       sits above it by a margin, and that margin is coined however it is dressed. So the",
        "       measurement is automated and the decision is not.",
        "",
        "       WHAT TO DO.",
        "         1. Make the pinned checkpoints reachable OFFLINE for the process that measures",
        "            (cluster/discoverer/102_stage_sam2_weights.sbatch stages them on the cluster;",
        "            scripts/estimators/apple_sam2.py names every revision and cache directory it",
        "            looks in, and refuses by name if one is absent).",
        "         2. Measure the distribution over the SOURCE corpus, WHOLE — no --limit, and",
        "            --stride 1. A truncated measurement is stamped measurement_qualified: false",
        "            and this loader refuses it:",
        "              python scripts/robot_composite.py measure \\",
        "                  --manifest <SOURCE>/manifest.json \\",
        f"                  --out {AREA_BOUND_ARTIFACT}",
        "            It writes min/median/p95/p99/max, the empty-mask frame count, the estimator",
        "            and the manifest hash, and leaves max_frame_fraction null.",
        "         3. Read that distribution, write max_frame_fraction and bound_rationale into the",
        "            artifact, and COMMIT it. The rationale is not decoration: it is the record of a",
        "            decision a script refused to make.",
        "",
        "       Until then this pipeline generates nothing, which is also what PR-08 §1 says.",
    ])


def load_area_bound(
    path: pathlib.Path | None = None,
    *,
    expect_segmenter: dict | None = None,
    expect_source_manifest: pathlib.Path | None = None,
) -> AreaBound:
    """Read and validate the committed bound. Every failure names the remedy.

    VALIDATING A BOUND AND ACCEPTING IT FOR A RUN ARE TWO DIFFERENT THINGS, and this function does
    the first always and the second only when told what to compare against. Called bare it checks
    shape, range, rationale and qualification, and returns a bound with ``cross_checked=False``,
    which :class:`CompositeContext` REFUSES to composite with — so "forgot to cross-check" is not a
    reachable state, it is a refusal. ``build_context`` is the only thing that builds a context and
    it has no way to skip either expectation.

    ``expect_segmenter`` is a masker's ``provenance()``; only :func:`segmenter_identity`'s fields are
    compared, which is exactly the tuple :meth:`MaskCache.key` keys a cached mask on. The rule is one
    sentence: *if a change would invalidate a cached mask, it invalidates the bound too.* Re-pinning
    GroundingDINO, editing :data:`ROBOT_TEXT_PROMPT` or moving a detection threshold changes the area
    distribution, and a bound measured under the old one either never fires — over-large masks
    composite the source back over the whole frame and the restyle silently becomes a no-op, the
    exact failure the bound exists to catch — or fires on everything.

    ``expect_source_manifest`` is the SOURCE manifest this run will restyle. Its sha256 must be the
    one the distribution was measured over: a bound is a statement about a corpus, and holding
    corpus A's distribution over corpus B is the same drift by another route.
    """
    path = pathlib.Path(path or AREA_BOUND_ARTIFACT)
    if not path.is_file():
        raise CompositeError(area_bound_missing_message(path, "The artifact does not exist."))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise CompositeError(area_bound_missing_message(path, f"It is not JSON: {exc}")) from exc
    missing = [k for k in AREA_BOUND_FIELDS_REQUIRED if k not in payload]
    if missing:
        raise CompositeError(
            area_bound_missing_message(
                path,
                f"It is missing {missing}. A bound that cannot say which segmenter measured the "
                "distribution it sits above, on which corpus, is a coined number in a committed "
                "file's clothing.",
            )
        )
    # THIS COMES BEFORE THE BOUND'S OWN VALIDITY, and the order is the message. An artifact whose
    # distribution is not the corpus's cannot be repaired by filling in a number, so the operator
    # has to hear "re-measure" before they are invited to decide.
    qualified = payload.get("measurement_qualified")
    if qualified is not True:
        # ``measure_geom_tol`` refuses exactly this and for exactly this reason: a bound sitting
        # above a distribution measured over three episodes at stride 30 is indistinguishable, at
        # load time, from one measured over the whole corpus — unless the artifact says so and the
        # loader reads it. The measure mode stamps this false for any --limit or any --stride > 1.
        reasons = payload.get("measurement_disqualified_reasons") or []
        raise CompositeError(
            area_bound_missing_message(
                path,
                f"measurement_qualified is {qualified!r}, not true"
                + (f" ({'; '.join(str(r) for r in reasons)})" if reasons else "")
                + ". The distribution under this bound is a smoke run, not the corpus. A bound is "
                "a claim about the whole source corpus; re-measure with no --limit and --stride 1.",
            )
        )
    value = payload["max_frame_fraction"]
    if value is None:
        raise CompositeError(
            area_bound_missing_message(
                path,
                "It carries max_frame_fraction: null — the distribution has been measured and the "
                "bound has not been decided. That null is written by the measure mode on purpose; "
                "replacing it is a human decision with a written rationale.",
            )
        )
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CompositeError(
            area_bound_missing_message(path, f"max_frame_fraction is {value!r}, not a number.")
        )
    value = float(value)
    if not 0.0 < value < 1.0:
        # STRICTLY below 1.0, and that end of the interval is the one that matters. check_mask
        # refuses on ``fraction > bound.max_frame_fraction`` and an area fraction can never exceed
        # 1.0, so a committed bound of exactly 1.0 makes the over-large refusal UNREACHABLE — half
        # of this check switched off, in a committed file, passing every other validation, with a
        # rationale string beside it that nobody would read twice. The empty-mask half would still
        # fire, so the failure is silent: over-large masks composite the source back over the whole
        # frame, the restyle becomes a no-op, and arms B and C become arm A at full GPU cost.
        raise CompositeError(
            area_bound_missing_message(
                path,
                f"max_frame_fraction is {value}, which is outside (0, 1). It is a FRACTION OF THE "
                "FRAME, not a pixel count — and 1.0 itself is refused rather than clamped: a mask "
                "cannot cover MORE than a frame, so a bound of 1.0 is not a loose bound, it is the "
                "over-large refusal deleted. If the intent is 'almost anything', write the number "
                "the distribution supports and say so in bound_rationale.",
            )
        )
    rationale = str(payload.get("bound_rationale") or "")
    if not rationale.strip():
        raise CompositeError(
            area_bound_missing_message(
                path,
                "bound_rationale is empty. The rationale is the record of the decision this script "
                "refused to make; without it the committed number is indistinguishable from a "
                "default someone typed.",
            )
        )

    if payload.get("prompt") != ROBOT_TEXT_PROMPT:
        # No caller input is needed for this one — the prompt is a committed constant in this file,
        # so a bound measured under a different one is stale on its face. A narrower prompt yields a
        # smaller mask and a smaller area distribution; a bound sitting above THAT distribution is
        # not above this one.
        raise CompositeError(
            area_bound_missing_message(
                path,
                f"it was measured with prompt {payload.get('prompt')!r} and this build's "
                f"ROBOT_TEXT_PROMPT is {ROBOT_TEXT_PROMPT!r}. The prompt decides which pixels are "
                "protected from the generator, so it decides the area distribution the bound sits "
                "above. Re-measure and re-decide.",
            )
        )

    cross_checked_against: dict | None = None
    if expect_segmenter is not None or expect_source_manifest is not None:
        cross_checked_against = {}
    if expect_segmenter is not None:
        want = segmenter_identity(expect_segmenter)
        got = segmenter_identity(payload.get("estimator") or {})
        if got != want:
            differing = sorted(k for k in want if want[k] != got[k])
            raise CompositeError(
                area_bound_missing_message(
                    path,
                    "it was measured by a DIFFERENT segmenter from the one this run will use; "
                    f"{differing} disagree.\n"
                    f"       committed: {json.dumps(got, sort_keys=True)}\n"
                    f"       this run:  {json.dumps(want, sort_keys=True)}\n"
                    "       These are the same fields the mask cache is keyed on, and the rule is "
                    "one sentence: a change that invalidates a cached mask invalidates the bound "
                    "too. A re-pinned detector produces a different area distribution, and the old "
                    "bound then either never fires — over-large masks composite the source back "
                    "over the whole frame and the restyle becomes a no-op — or fires everywhere.",
                )
            )
        cross_checked_against["segmenter"] = want
    if expect_source_manifest is not None:
        manifest = pathlib.Path(expect_source_manifest)
        if not manifest.is_file():
            raise CompositeError(
                f"FATAL: the source manifest {manifest} does not exist, so the committed area "
                "bound cannot be checked against the corpus this run will restyle."
            )
        want_sha = _file_sha256(manifest)
        got_sha = str(payload.get("source_manifest_sha256"))
        if got_sha != want_sha:
            raise CompositeError(
                area_bound_missing_message(
                    path,
                    "it was measured over a DIFFERENT source corpus.\n"
                    f"       committed source_manifest_sha256: {got_sha}\n"
                    f"       {manifest}: {want_sha}\n"
                    "       A bound is a claim about a corpus: the largest fraction of a frame a "
                    "robot mask covers in THESE episodes. Applied to other episodes it is a number "
                    "with no distribution behind it, which is the coined threshold §6 refuses.",
                )
            )
        cross_checked_against["source_manifest_sha256"] = want_sha
        cross_checked_against["source_manifest"] = str(manifest)

    return AreaBound(
        max_frame_fraction=value,
        artifact=path,
        artifact_sha256=_file_sha256(path),
        rationale=rationale,
        # Both expectations, or it is not cross-checked. A half-check is a bound that matches the
        # segmenter and was measured on another corpus, or the reverse, and neither may composite.
        cross_checked=expect_segmenter is not None and expect_source_manifest is not None,
        cross_checked_against=cross_checked_against,
    )


# --------------------------------------------------------------------------------------------
# the composite itself
# --------------------------------------------------------------------------------------------


def composite_frame(
    source: np.ndarray, generated: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """``mask ? source : generated``, pixel-exact, hard-edged.

    Three lines and the whole of G0c. ``np.where`` on the boolean mask broadcast over the channel
    axis: inside the mask every channel is the source's byte, outside it every channel is the
    generator's byte, and there is no third case — no blend, no ramp, no interpolation. See the
    module docstring for why a feather is refused; the burden was on feathering and it does not
    clear it.

    Exactness matters beyond tidiness. The claim G0c makes is "the defect cannot enter", and it is
    only true if the robot region contains no generated information at all. A single blended pixel
    row along the silhouette would make it "the defect enters attenuated", which is a different and
    much weaker sentence, and one no downstream gate could measure.
    """
    src = np.asarray(source)
    gen = np.asarray(generated)
    if src.shape != gen.shape:
        raise CompositeError(f"source frame {src.shape} and generated frame {gen.shape} disagree.")
    m = np.asarray(mask, dtype=bool)
    if m.shape != src.shape[:2]:
        raise CompositeError(f"mask {m.shape} does not fit frame {src.shape[:2]}.")
    return np.where(m[:, :, None], src, gen)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boolean masks. Two empty masks are 1.0, not NaN.

    The degenerate case is decided rather than left to 0/0 because this number is written into an
    artifact: a NaN there would be read as "not measured" by anyone skimming, when what actually
    happened is that neither the source nor the generated frame contained a robot — a fact worth
    recording as agreement. It cannot arise on the composite path anyway, where an empty SOURCE mask
    has already refused the clip; it can arise in the measure mode.
    """
    x = np.asarray(a, dtype=bool)
    y = np.asarray(b, dtype=bool)
    if x.shape != y.shape:
        raise CompositeError(f"IoU wants two masks of one shape; got {x.shape} and {y.shape}.")
    union = int(np.count_nonzero(x | y))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(x & y)) / float(union)


def check_mask(mask: np.ndarray, *, frame_index: int, bound: AreaBound, source: str) -> float:
    """Refuse an empty or over-large robot mask on THIS frame. Returns the area fraction.

    Both refusals are clip-fatal on purpose. A corpus is trained on as a whole, and "9 frames out of
    427 had no robot composited" is not a smaller version of the failure — those 9 frames carry a
    generated manipulator into the training set exactly as 427 would, and no gate downstream looks at
    frames individually.
    """
    m = np.asarray(mask, dtype=bool)
    covered = int(np.count_nonzero(m))
    if covered == 0:
        raise CompositeError(
            f"{source}: the robot mask is EMPTY on frame {frame_index}.\n"
            "       An empty robot mask means the composite is the identity on that frame and the "
            "GENERATED manipulator went straight into the corpus — the one failure PR-08 §6 G0c "
            "exists to make impossible. There is no threshold in this check and no number to "
            "loosen: zero is zero.\n"
            f"       The segmenter is {ROBOT_TEXT_PROMPT!r} through scripts/estimators/apple_sam2.py. "
            "If the robot is genuinely absent from this frame the SOURCE corpus is not what PR-08 §3 "
            "describes; if it is present, the prompt or the detector thresholds do not find it, and "
            "that has to be fixed before generation rather than skipped per frame."
        )
    fraction = covered / float(m.size)
    if fraction > bound.max_frame_fraction:
        raise CompositeError(
            f"{source}: the robot mask covers {fraction:.4f} of frame {frame_index}, above the "
            f"committed bound {bound.max_frame_fraction} ({bound.artifact}).\n"
            "       A mask this large has grounded on something that is not the robot — the table, "
            "or the whole scene. Compositing it copies the SOURCE back over everything, the restyle "
            "becomes a no-op, and arms B and C silently become arm A while still costing their GPU "
            "hours.\n"
            f"       The bound's own rationale: {bound.rationale}"
        )
    return fraction


@dataclass(frozen=True)
class CompositeContext:
    """Everything the composite needs, resolved once per run and then not negotiable.

    Frozen, and built only by :func:`build_context`, so that nothing between ``main`` and the frame
    loop can substitute a different masker or a looser bound halfway through a chunk.
    """

    masker: Any
    bound: AreaBound
    iou_stride: int
    cache: MaskCache | None

    def __post_init__(self) -> None:
        """No context may hold a bound that was never checked against THIS run's segmenter.

        The check lives here rather than in ``build_context`` because this is the type every
        compositing path must go through, including one constructed directly. A bound loaded bare —
        validated for shape but never compared to the segmenter that will make the masks or to the
        corpus it will run on — is the drift failure in :func:`load_area_bound`'s docstring, and
        making it a refusal at construction means there is no reachable way to composite with one.
        """
        if not self.bound.cross_checked:
            raise CompositeError(
                f"FATAL: the area bound {self.bound.artifact} was validated but never cross-checked "
                "against this run.\n"
                "       load_area_bound() must be given BOTH expect_segmenter (the provenance of "
                "the masker that will make the masks) and expect_source_manifest (the corpus this "
                "run restyles). A bound measured under a different segmenter or over a different "
                "corpus is a number with no distribution behind it — see load_area_bound's "
                "docstring. build_context() supplies both and is the intended way to get here."
            )

    def provenance(self) -> dict:
        return {
            "rule": (
                "PR-08 §6 G0c — the real robot's pixels are unconditionally composited back over "
                "every generated frame, using the robot segmentation mask. No threshold, no "
                "feather, no opt-out."
            ),
            "masker": self.masker.provenance(),
            "area_bound": self.bound.record(),
            "edge": "hard binary mask, no feather and no dilation (see robot_composite docstring)",
            "iou_stride": self.iou_stride,
            "mask_cache": str(self.cache.root) if self.cache else None,
        }

    def composite(
        self,
        *,
        source_video: pathlib.Path,
        generated_video: pathlib.Path,
        expected_frames: int | None = None,
    ) -> dict:
        return composite_clip(
            source_video=source_video,
            generated_video=generated_video,
            context=self,
            expected_frames=expected_frames,
        )


def check_video_io() -> None:
    """Refuse now if this process cannot decode or encode video. A run-level fact, checked run-level.

    ``cv2`` and ``imageio``/``imageio-ffmpeg`` are what the composite reads and writes with, and on
    the cluster this driver runs inside Cosmos-Transfer2.5's venv rather than ours. A missing wheel
    discovered inside the per-unit guard becomes N identical ImportErrors that look like a flaky
    generator, spend a pass of the chunk's rail, and send the operator to the model. Discovered here
    it is one line naming the package.
    """
    missing = []
    for module, wheel in (("cv2", "opencv-python-headless"), ("imageio", "imageio"),
                          ("imageio_ffmpeg", "imageio-ffmpeg")):
        try:
            __import__(module)
        except ImportError:
            missing.append(f"{module} (pip install {wheel})")
    if missing:
        raise CompositeError(
            "PR-08 §6 G0c composites the real robot back over every generated frame, which means "
            "decoding the source, decoding the generated clip and re-encoding the result. This "
            f"process cannot: {', '.join(missing)}. Refusing before the first clip rather than once "
            "per unit — the venv is the fix, not the retry."
        )


def build_context(
    *,
    source_manifest: pathlib.Path,
    area_bound_path: pathlib.Path | None = None,
    iou_stride: int = 10,
    cache_dir: pathlib.Path | None = None,
    preflight: bool = True,
) -> CompositeContext:
    """Resolve the masker, the bound and the cache — refusing here rather than mid-chunk.

    ``preflight=True`` loads GroundingDINO and SAM 2 before the first unit. A missing checkpoint is a
    fact about the run: discovered inside the per-unit guard it becomes N identical per-unit errors
    that read like a flaky generator, the chunk's pass rail spends a pass on them, and the operator
    is sent to look at the wrong thing.

    ``source_manifest`` IS REQUIRED AND HAS NO DEFAULT, and it is not an input to the composite: it
    is what the committed area bound is cross-checked against. A bound is a claim about a corpus,
    and the caller is the only one that knows which corpus this run restyles. Required rather than
    optional because an optional cross-check is a cross-check that gets skipped.

    The ORDER below is deliberate. The masker is built and preflighted BEFORE the bound is loaded,
    because the bound is checked against the masker's provenance and that provenance is only
    truthful once the pinned adapter has actually been imported — a provenance built from an
    adapter that could not load would be a description of a segmenter that is not there.

    There is no ``enabled`` argument and no way to obtain a context that does not composite. The
    driver holds one of these or it holds nothing, and holding nothing means it never got past
    ``main``'s setup.
    """
    if int(iou_stride) < 1:
        raise CompositeError(f"--iou-stride must be >= 1; got {iou_stride}.")
    check_video_io()
    masker = build_masker()
    if preflight:
        masker.preflight()
    # Once, here, at run level: ``provenance()`` reads the pinned adapter's constants and refuses if
    # it can no longer describe what made the masks. Left to first use it would raise inside the
    # per-unit guard or, worse, inside main's banner line, where it is a traceback rather than a
    # refusal the sbatch can act on.
    provenance = masker.provenance()
    bound = load_area_bound(
        area_bound_path, expect_segmenter=provenance, expect_source_manifest=source_manifest
    )
    return CompositeContext(
        masker=masker,
        bound=bound,
        iou_stride=int(iou_stride),
        cache=MaskCache(cache_dir) if cache_dir is not None else None,
    )


def source_masks(
    source_video: pathlib.Path, frames: np.ndarray, context: CompositeContext
) -> tuple[np.ndarray, bool]:
    """Robot masks for every source frame, from the cache when the cache holds this exact input.

    Returns ``(masks, from_cache)``. The masks are NOT validated here — :func:`composite_clip`
    validates them frame by frame, including the ones that came out of the cache, because a cache
    written before the bound was tightened must not skip the check the bound exists to make.
    """
    key = None
    if context.cache is not None:
        key = MaskCache.key(source_video, context.masker.provenance())
        hit = context.cache.get(key, frames.shape[:3])
        if hit is not None:
            return hit, True
    masks = np.stack([np.asarray(context.masker.mask(frame), dtype=bool) for frame in frames])
    if masks.shape != frames.shape[:3]:
        raise CompositeError(
            f"the masker returned {masks.shape} for {frames.shape[:3]} source frames. A mask on one "
            "grid applied to a frame on another is not a mask."
        )
    if context.cache is not None and key is not None:
        context.cache.put(key, masks)
    return masks, False


def composite_clip(
    *,
    source_video: pathlib.Path,
    generated_video: pathlib.Path,
    context: CompositeContext,
    expected_frames: int | None = None,
) -> dict:
    """Composite the real robot back over ``generated_video``, IN PLACE, and return the record.

    In place, and atomically: the composited frames are encoded to a sibling temporary file which
    then REPLACES the generated clip. The harvest keys on the existence of ``vision.mp4``, so a
    half-written composite under that name would be filed as a finished clip. A crash mid-encode
    leaves the model's original output in place and the caller quarantines it.

    Every frame is composited. There is no stride, no sampling and no early exit: the loop runs from
    0 to F-1 and the count it reports is asserted against the frame count before the record is
    written, so "composited" in the artifact means what it says.
    """
    source_video = pathlib.Path(source_video)
    generated_video = pathlib.Path(generated_video)

    src = decode_clip(source_video)
    gen = decode_clip(generated_video)

    if expected_frames is not None and src.shape[0] != int(expected_frames):
        raise CompositeError(
            f"{source_video} decodes {src.shape[0]} frames, the manifest declares "
            f"{int(expected_frames)}. The actions are carried over from the recording by INDEX, so "
            "a source whose length disagrees with the label column pairs every frame after the "
            "divergence with the wrong action — silently, with no decode error. Rebuild the source "
            "with scripts/build_pr08_source.py, which checks exactly this."
        )
    if gen.shape != src.shape:
        raise CompositeError(
            f"{generated_video} is {gen.shape} and {source_video} is {src.shape}. Compositing frame "
            "i of one over frame i of the other requires them to be the same clip at the same "
            "geometry; a mismatch would put the robot from one instant into the scene of another, "
            "which is geometry drift manufactured by the gate that exists to protect geometry — and "
            "G0b would then score it as a generator defect."
        )

    before_filter = dict(getattr(context.masker, "filter_counters", {}) or {})
    masks, from_cache = source_masks(source_video, src, context)
    # Differenced HERE and not after the loop: the IoU diagnostic below runs the masker over the
    # GENERATED frames too, and the object filter's behaviour there is a different question — the
    # colour reference describes the SOURCE corpus's apple, and a restyle whose whole point is to
    # change how the scene looks may not fire it at all. Pooling the two would make a block that
    # answers neither.
    after_filter = dict(getattr(context.masker, "filter_counters", {}) or {})

    fractions: list[float] = []
    ious: list[float] = []
    iou_frames: list[int] = []
    out = np.empty_like(gen)
    for index in range(src.shape[0]):
        mask = masks[index]
        fractions.append(
            check_mask(mask, frame_index=index, bound=context.bound, source=str(generated_video))
        )
        out[index] = composite_frame(src[index], gen[index], mask)
        # The IoU is measured against the RAW generated frame, before this loop's own output
        # replaces it. After compositing the two masks agree by construction and the number would
        # be a measurement of this file rather than of Cosmos-Transfer2.5.
        if index % context.iou_stride == 0:
            ious.append(mask_iou(mask, np.asarray(context.masker.mask(gen[index]), dtype=bool)))
            iou_frames.append(index)

    if len(fractions) != src.shape[0]:
        raise CompositeError(
            f"composited {len(fractions)} of {src.shape[0]} frames. Reported rather than rounded up: "
            "a partially composited clip is a clip that carries the generated manipulator on the "
            "frames it missed."
        )

    fps = container_fps(source_video) or container_fps(generated_video)
    if fps is None:
        raise CompositeError(
            f"neither {source_video} nor {generated_video} declares a frame rate, so the composited "
            "clip cannot be written at the source's. Frames are paired by index and the rate "
            "changes no pairing, but a clip whose declared rate came from nowhere is a clip whose "
            "record cannot say what it plays at."
        )

    tmp = generated_video.with_suffix(".composited.tmp.mp4")
    try:
        encode_clip(out, tmp, fps)
        tmp.replace(generated_video)
    finally:
        tmp.unlink(missing_ok=True)

    return {
        **context.provenance(),
        "composited": True,
        "frames_composited": int(src.shape[0]),
        "frames_total": int(src.shape[0]),
        "fps": float(fps),
        "source_video": str(source_video),
        "mask_source_frames_from_cache": bool(from_cache),
        "robot_mask_object_filter": {
            "note": (
                "PR-08 V9. Counted over the SOURCE frames of this clip only; the robot-mask IoU "
                "diagnostic below masks the generated frames and is deliberately not pooled here. "
                "All-zero counts with masks_from_cache true mean the masks predate this run, NOT "
                "that the filter never fired — the cache key carries the filter, so a hit is a hit "
                "against the same filter."
            ),
            "masks_from_cache": bool(from_cache),
            "max_iou": float(ROBOT_MASK_OBJECT_MAX_IOU),
            **{
                name: int(after_filter.get(name, 0) - before_filter.get(name, 0))
                for name in sorted(set(after_filter) | set(before_filter))
            },
        },
        "mask_area_fraction": {
            "min": float(np.min(fractions)),
            "mean": float(np.mean(fractions)),
            "max": float(np.max(fractions)),
        },
        "robot_mask_iou_source_vs_generated": {
            "THIS_IS_A_DIAGNOSTIC_ON_THE_GENERATOR_AND_NEVER_A_GATE": True,
            "note": (
                "PR-08 §6 G0c: 'Robot-mask IoU between source and generated is still recorded, as a "
                "diagnostic on the generator, never as a gate.' T40_RULE_V2 §0 and T40_RULE_V3 §1 "
                "both repeat it in their unchanged-gates tables. Nothing in this pipeline compares "
                "these numbers to a threshold; §6 says an IoU threshold on the robot mask 'would be "
                "a coined number' and refuses one. Adding one is an amendment to T40_RULE_V1, not a "
                "code change."
            ),
            "measured_on": "the RAW generated frames, before compositing (after it, it is 1.0 by construction)",
            "frames_sampled": len(iou_frames),
            "stride": context.iou_stride,
            "stride_note": (
                "a sampling rate for a diagnostic that gates nothing, so it cannot become a "
                "finding. The COMPOSITE itself has no stride: every frame, always."
            ),
            "mean": float(np.mean(ious)) if ious else None,
            "min": float(np.min(ious)) if ious else None,
            "max": float(np.max(ious)) if ious else None,
        },
    }


# --------------------------------------------------------------------------------------------
# the measure mode — the distribution, and deliberately not the decision
# --------------------------------------------------------------------------------------------


#: Schema of the artifact a bound may sit above — written by a whole-corpus ``measure`` run and by
#: ``--merge``. Added so the merge can tell a shard from a finished measurement without guessing at
#: the shape; ``load_area_bound`` does not read it, and a pre-existing artifact without it is still
#: a valid bound.
AREA_SCHEMA = "wam.robot_mask_area/1"

#: Schema of ONE shard's artifact. A different string from :data:`AREA_SCHEMA` on purpose: a shard
#: is not a small measurement, it is a piece of one, and the two must not be interchangeable in a
#: directory scan, on a command line, or under a reader's eye.
AREA_SHARD_SCHEMA = "wam.robot_mask_area_shard/1"

#: The partition rule, recorded in every shard and RE-DERIVED by the merge rather than trusted.
#: Identical to ``measure_geom_tol.SHARD_ASSIGNMENT`` — deliberately the same rule, so that an
#: operator who has read one of these two jobs has read both, and a test asserts the two functions
#: agree key for key.
SHARD_ASSIGNMENT = (
    "int.from_bytes(blake2b(episode_key.utf8, digest_size=8).digest(), 'big') % num_shards"
)

#: Verbatim the sentence a whole-corpus run has always written. Hoisted into a constant only so the
#: merge writes the SAME words; changing it changes what a non-sharded run produces.
BOUND_NOTE = (
    "max_frame_fraction is null ON PURPOSE. This file measures the distribution; it does "
    "not choose the bound. The observed maximum below cannot fire on the frames it was "
    "measured over, and any bound above it carries a margin nothing in the corpus derives "
    "— so choosing one is a decision with a written rationale, not a computation. Fill in "
    "max_frame_fraction and bound_rationale, then commit this file."
)

#: The six conditions a MERGE must satisfy before its pooled distribution is one a bound may sit
#: above. Named as data rather than as prose because the merge writes the verdict per condition
#: into the artifact and the sbatch reads it: "which of these failed" has to survive JSON.
#:
#: They are not the whole of what the merge checks. The refusals in :func:`merge_shard_records`
#: come FIRST and are fatal with nothing written, because they are the cases where the pool itself
#: would be wrong — an episode counted twice, a per_episode entry a shard was never assigned, a
#: shard that kept only its own summary. These six are the cases where the pool is honest
#: arithmetic over something that is not the corpus, which is exactly what
#: ``measurement_qualified: false`` means everywhere else in this file.
MERGE_CONDITIONS: tuple[str, ...] = (
    "shards_tile_the_corpus_exactly_once",
    "every_shard_at_stride_1_with_no_limit",
    "every_shard_measurement_qualified",
    "shards_agree_on_estimator",
    "shards_agree_on_source_manifest_sha256",
    "shards_agree_on_prompt",
)

#: Everything the merge checks, named. The first ten are fatal and write nothing; the last six are
#: :data:`MERGE_CONDITIONS` and stamp ``measurement_qualified: false``.
MERGE_REFUSALS_CHECKED: tuple[str, ...] = (
    "a path named to --merge is unreadable, is not JSON, or is not a shard artifact",
    "a finished measurement was handed to --merge as if it were a shard",
    "--merge was given no shard artifacts at all",
    "two artifacts claim the same shard index",
    "the shards disagree on num_shards, so they are pieces of two different partitions",
    "the shards enumerated different corpora (corpus_episode_keys_sha256)",
    "a shard carries no usable 'shard' block or no corpus_episode_keys",
    "a shard holds an episode that does not hash to it under the recorded assignment rule",
    "a shard reports a per_episode entry for an episode it was not assigned",
    "a shard does not account for every episode it was assigned",
    "a shard kept only its own summary and no raw per-frame area fractions",
    "the shards do not tile the corpus exactly once (a shard is missing)",
    "a shard was measured with --limit, or at --stride > 1",
    "a shard is itself measurement_qualified: false",
    "the shards disagree on the estimator that made the masks",
    "the shards disagree on the source manifest they measured",
    "the shards disagree on the robot prompt",
)


def shard_of(episode_key: str, num_shards: int) -> int:
    """Which shard owns this episode. Deterministic across processes, machines and Python builds.

    ``hash(episode_key) % num_shards`` is the obvious spelling and it is a trap: ``PYTHONHASHSEED``
    is randomised per interpreter, so every task of the same Slurm array would compute a DIFFERENT
    partition of the same corpus. The failure is not a crash — it is a set of shard artifacts that
    together cover some episodes twice and others never, each internally consistent.

    Byte-for-byte ``measure_geom_tol.shard_of``. Copied rather than imported because that module
    reaches numpy and a large argument parser and this one is imported by
    ``restyle_transfer25`` inside the Transfer2.5 venv on a GPU node; a test asserts the two agree
    over a corpus of keys, so the copy cannot drift silently.
    """
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    digest = hashlib.blake2b(episode_key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % num_shards


def corpus_keys_digest(keys: list[str]) -> str:
    """A stable digest of one corpus enumeration. Newline-joined so no key can absorb another."""
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def _select_episodes(
    keys: list[str],
    episodes: list[dict],
    *,
    shard: int | None,
    num_shards: int | None,
) -> list[tuple[int, dict]]:
    """The episodes this run measures, each paired with its position in the FULL enumeration."""
    if shard is None and num_shards is None:
        return list(enumerate(episodes))
    if shard is None or num_shards is None:
        raise CompositeError(
            "--shard and --num-shards go together. One without the other is a partition with an "
            "unknown denominator,\n"
            "       and an artifact from it could not be merged: the merge needs to know how many "
            "shards to insist on\n"
            "       before it can say one is missing."
        )
    num_shards = int(num_shards)
    shard = int(shard)
    if num_shards < 1:
        raise CompositeError(f"--num-shards {num_shards} is not a positive integer.")
    if not 0 <= shard < num_shards:
        raise CompositeError(
            f"--shard {shard} is out of range for --num-shards {num_shards}: the shards are "
            f"0..{num_shards - 1}.\n"
            "       A shard index nobody will merge produces an artifact that looks finished and "
            "is unreachable; an\n"
            "       out-of-range one measures the empty set and reports it as a clean run."
        )
    return [(i, ep) for i, (k, ep) in enumerate(zip(keys, episodes)) if shard_of(k, num_shards) == shard]


def _shard_block(
    shard: int, num_shards: int, selected: list[tuple[int, dict]], all_keys: list[str]
) -> dict:
    """The provenance a shard carries so the merge can check it rather than trust it."""
    return {
        "index": int(shard),
        "num_shards": int(num_shards),
        "assignment": SHARD_ASSIGNMENT,
        "assignment_note": (
            "Assignment is a digest of the episode ID, not a slice of the episode LIST. Adding or "
            "removing a clip therefore moves that clip only; a range would renumber every episode "
            "after it and silently re-partition a corpus whose shards are computed by different "
            "jobs at different times. The merge re-derives this rule and refuses a shard holding "
            "an episode that does not hash to it."
        ),
        # WHICH EPISODES, not how many. A count cannot prove coverage: eight shards reporting 50
        # episodes each sum to 400 whether they covered 400 distinct episodes or 380 with 20
        # counted twice. The merge takes the union of these and compares it to the enumeration.
        "episode_keys": [str(ep.get("id")) for _, ep in selected],
        "episode_indices": [int(i) for i, _ in selected],
        "n_episodes_in_shard": len(selected),
        "corpus_episode_keys_sha256": corpus_keys_digest(all_keys),
    }


def _measured_block(
    fractions: list[float],
    empty: int,
    *,
    episode_ids: list[str],
    total_episodes: int,
    limit: int | None,
    stride: int,
) -> dict:
    """The distribution block, computed ONCE over whatever population it is handed.

    The merge hands it the pooled population and the measurement hands it its own, which is the
    whole of why the merged artifact equals an un-sharded one: there is one implementation of
    min/median/p95/p99/max and neither path recombines a percentile from summaries.
    """
    arr = np.asarray(fractions, dtype=np.float64)
    return {
        "frames": int(arr.size),
        "episodes": len(episode_ids),
        "episodes_in_manifest": int(total_episodes),
        "episode_ids": list(episode_ids),
        # Recorded even when None, because "no --limit was given" and "the field predates the
        # flag" are different facts and an absent key cannot tell them apart.
        "limit": None if limit is None else int(limit),
        "stride": int(stride),
        "empty_frames": int(empty),
        "empty_frame_fraction": float(empty) / float(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def measure_source_mask_area(
    manifest: pathlib.Path,
    *,
    masker: Any,
    limit: int | None = None,
    stride: int = 1,
    shard: int | None = None,
    num_shards: int | None = None,
) -> dict:
    """Robot-mask area fraction over the SOURCE corpus — the distribution the bound sits above.

    With ``shard``/``num_shards`` this measures only the episodes that hash to that shard and
    writes a :data:`AREA_SHARD_SCHEMA` artifact carrying the RAW per-frame fractions;
    :func:`merge_shard_records` pools those into the artifact a bound may sit above. The corpus is
    171 625 frames and the wall is four hours, which is the same arithmetic
    ``scripts/measure_geom_tol.py`` already answers this way — see the module docstring's sharding
    section and ``cluster/discoverer/106_measure_robot_mask_area.sbatch``.

    Writes no bound. See :func:`load_area_bound`'s refusal for why: the only bound this could derive
    honestly (the observed maximum) can never fire on the frames it was derived from, and any bound
    that can fire sits above it by a margin nothing in the corpus determines. So this measures and a
    human decides, in writing, in a committed file.

    ``empty_frames`` is the number that matters most on the first run. An empty robot mask refuses a
    clip, so if the prompt or the thresholds leave frames empty on the SOURCE corpus, every clip
    containing one of those frames will refuse — and that is a fact worth learning here, before
    10 050 clips of GPU time, rather than from a chunk of refusals.

    IT WILL NOT LET A SMOKE TEST BECOME THE BOUND. ``--limit`` and ``--stride`` exist because the
    first run of this on a corpus is a shakedown, and a shakedown that writes an artifact
    indistinguishable from the real measurement is how a bound ends up sitting above a distribution
    of three episodes at every 30th frame. So a truncated run is STAMPED: ``measurement_qualified:
    false`` with the reasons, and :func:`load_area_bound` refuses such an artifact by name. Both
    knobs truncate in the direction that matters — fewer episodes and fewer frames can only lower
    the observed maximum, so a bound chosen above a truncated maximum can sit BELOW the real one and
    refuse honest clips, or be nudged up to compensate, which is coining. ``measure_geom_tol`` makes
    the same refusal for the same reason and this follows it deliberately.
    """
    manifest = pathlib.Path(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    episodes = list(payload.get("episodes") or ())
    if not episodes:
        raise CompositeError(f"{manifest} lists no episodes.")
    total_episodes = len(episodes)
    # THE ENUMERATION IS THE MANIFEST'S, TAKEN BEFORE --limit TRUNCATES IT. What a shard recorded as
    # "the corpus it saw" has to be the same list whether or not that run was truncated: the merge
    # digests it to prove the shards measured ONE corpus, and a truncated run whose digest differed
    # would be refused as a different corpus instead of being stamped as the truncated measurement
    # it is. The truncation is recorded where it belongs — measured.limit, and the disqualification
    # reasons — and shows up in the merge as episodes nobody covered.
    all_keys = [str(entry.get("id")) for entry in episodes]
    stride = int(stride)
    if stride < 1:
        raise CompositeError(f"--stride must be >= 1; got {stride}.")
    disqualified: list[str] = []
    if limit is not None:
        episodes = episodes[: int(limit)]
        disqualified.append(
            f"--limit {int(limit)}: {len(episodes)} of the manifest's {total_episodes} episodes "
            "were measured, so this distribution is not the corpus's"
        )
    if stride > 1:
        disqualified.append(
            f"--stride {stride}: every {stride}th frame was measured, so the observed maximum is a "
            "maximum over a subsample and can only understate the corpus's"
        )

    keys = [str(entry.get("id")) for entry in episodes]
    # THE PARTITION IS KEYED ON THE EPISODE ID, so two episodes carrying one id is not a cosmetic
    # defect: both would hash to the same shard, the merge's coverage arithmetic would count one
    # episode where two were measured, and the pooled distribution would be over a corpus nobody
    # can name. Refused whether or not this run is sharded — an un-sharded artifact from such a
    # manifest is what a later sharded run would be compared against.
    duplicates = sorted({k for k in all_keys if all_keys.count(k) > 1})
    if duplicates:
        raise CompositeError(
            f"{manifest} lists {len(duplicates)} episode id(s) more than once: "
            + ", ".join(duplicates[:8]) + ("..." if len(duplicates) > 8 else "") + "\n"
            "       Episode ids are what the shard partition is keyed on and what the merge proves "
            "coverage with, so\n"
            "       they have to identify an episode. Fix the manifest; nothing here guesses which "
            "clip was meant."
        )

    selected = _select_episodes(keys, episodes, shard=shard, num_shards=num_shards)
    sharding = shard is not None or num_shards is not None

    per_episode: list[dict] = []
    fractions: list[float] = []
    empty = 0
    measured_episodes: list[str] = []
    for position, entry in selected:
        video = manifest.parent / str(entry["video"])
        frames = decode_clip(video)
        episode_fractions: list[float] = []
        episode_empty = 0
        for index in range(0, frames.shape[0], max(1, int(stride))):
            mask = np.asarray(masker.mask(frames[index]), dtype=bool)
            covered = int(np.count_nonzero(mask))
            if covered == 0:
                episode_empty += 1
            episode_fractions.append(covered / float(mask.size))
        fractions.extend(episode_fractions)
        empty += episode_empty
        measured_episodes.append(str(entry.get("id")))
        per_episode.append({
            # The episode's position in the manifest's OWN enumeration, not a serial number within
            # the shard. It is what lets the merge rebuild the pooled array in the order an
            # un-sharded run built it, which is what makes the merged artifact identical rather
            # than merely close.
            "episode_index": int(position),
            "episode": str(entry.get("id")),
            "n_frames": len(episode_fractions),
            "empty_frames": int(episode_empty),
            # RAW, one float per measured frame. A median and two percentiles do not decompose
            # across shards — the median of the shard medians is a different statistic — so a shard
            # that reported only its own summary could be averaged and never merged. See
            # merge_shard_records.
            "area_fractions": episode_fractions,
        })

    if not fractions:
        if sharding:
            raise CompositeError(
                f"shard {shard} of {num_shards} was assigned no frames at all "
                f"({len(selected)} of the {len(episodes)} episode(s) enumerated hash to it).\n"
                "       That is a statement about the PARTITION, not about the corpus: --num-shards "
                "is larger than the\n"
                "       number of episodes, or the episodes it was assigned decoded nothing. An "
                "artifact here would be a\n"
                "       well-formed shard contributing no frames, and the merge would pool it and "
                "prove a coverage it\n"
                "       never had. Lower --num-shards, or fix the clips this shard was assigned."
            )
        raise CompositeError(f"{manifest}: nothing was measured.")

    measured = _measured_block(
        fractions,
        empty,
        episode_ids=measured_episodes,
        total_episodes=total_episodes,
        limit=limit,
        stride=stride,
    )
    estimator = masker.provenance()
    source_manifest_sha256 = _file_sha256(manifest)

    if not sharding:
        return {
            "schema": AREA_SCHEMA,
            "max_frame_fraction": None,
            "bound_rationale": "",
            "bound_note": BOUND_NOTE,
            # A bound may only sit above a distribution that IS the corpus's. False here is not a
            # warning to weigh: load_area_bound refuses the artifact.
            "measurement_qualified": not disqualified,
            "measurement_disqualified_reasons": disqualified,
            "measured": measured,
            "estimator": estimator,
            "prompt": ROBOT_TEXT_PROMPT,
            "source_manifest": str(manifest),
            "source_manifest_sha256": source_manifest_sha256,
        }

    return {
        "schema": AREA_SHARD_SCHEMA,
        "shard": _shard_block(int(shard), int(num_shards), selected, all_keys),
        # The whole enumeration this shard saw, so the merge can NAME the episodes nobody covered
        # instead of only counting them. Shard-only: it never reaches the merged artifact.
        "corpus_episode_keys": list(all_keys),
        "per_episode": per_episode,
        # A SHARD IS NOT A DISTRIBUTION AND MUST NEVER BE READABLE AS ONE. It carries no
        # bound_rationale at all, so load_area_bound refuses it on AREA_BOUND_FIELDS_REQUIRED
        # before it ever reads a number — a shard artifact copied to the committed path is a
        # refusal, not a bound over a twelfth of the corpus.
        "max_frame_fraction": None,
        "max_frame_fraction_is_null_because": (
            "this is one shard of a partition, not the corpus. The five numbers under 'measured' "
            "are this shard's own and are printed as a sanity check; the artifact a bound may sit "
            "above is written by --merge, which pools the raw per-frame fractions below and takes "
            "min/median/p95/p99/max ONCE over the pooled population. Percentiles do not decompose."
        ),
        "measurement_qualified": not disqualified,
        "measurement_disqualified_reasons": disqualified,
        "measured": measured,
        "estimator": estimator,
        "prompt": ROBOT_TEXT_PROMPT,
        "source_manifest": str(manifest),
        "source_manifest_sha256": source_manifest_sha256,
    }


def _read_shard_json(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CompositeError(f"FATAL: --merge could not read {path}: {exc}") from exc
    except ValueError as exc:
        raise CompositeError(
            f"FATAL: --merge could not parse {path} as JSON: {exc}\n"
            "       A truncated shard artifact is what a job killed at the wall leaves behind. "
            "Re-run that shard;\n"
            "       merging around it would drop its frames and the merge would then be a "
            "distribution over part of\n"
            "       the corpus wearing the name of the whole."
        ) from exc


def collect_shard_records(paths: list[pathlib.Path]) -> list[tuple[pathlib.Path, dict]]:
    """Read the shard artifacts named on the command line. Directories are expanded, files are not.

    A directory is scanned for ``*.json`` and anything that is not a shard artifact is SKIPPED with
    a line on stderr — the merge job's own output directory holds the pilot artifact and, after the
    first successful merge, the merged one, and a scan that refused on those would be unusable. A
    path named EXPLICITLY is never skipped: the operator said that file, and quietly ignoring it is
    how a merge comes to be missing a shard that was right there on the command line.
    """
    found: list[tuple[pathlib.Path, dict]] = []
    for path in paths:
        path = pathlib.Path(path)
        if path.is_dir():
            for candidate in sorted(path.glob("*.json")):
                rec = _read_shard_json(candidate)
                if rec.get("schema") != AREA_SHARD_SCHEMA:
                    print(
                        f"--merge: skipping {candidate} (schema {rec.get('schema')!r}, not "
                        f"{AREA_SHARD_SCHEMA!r})",
                        file=sys.stderr,
                    )
                    continue
                found.append((candidate, rec))
            continue
        if not path.exists():
            raise CompositeError(
                f"FATAL: --merge {path} does not exist.\n"
                "       Named explicitly, so it is not skipped: a shard the operator asked for and "
                "that is not there\n"
                "       is a missing shard, not a filter."
            )
        rec = _read_shard_json(path)
        if rec.get("schema") != AREA_SHARD_SCHEMA:
            raise CompositeError(
                f"FATAL: --merge {path} carries schema {rec.get('schema')!r}, not "
                f"{AREA_SHARD_SCHEMA!r}.\n"
                f"       A {AREA_SCHEMA!r} artifact is a finished distribution, not an input to a "
                "merge, and merging one\n"
                "       in would pool a corpus with itself. Nothing here guesses which you meant."
            )
        found.append((path, rec))
    if not found:
        raise CompositeError(
            "FATAL: --merge found no shard artifacts at all in "
            + ", ".join(str(p) for p in paths) + ".\n"
            "       A merge over zero shards is not an empty result, it is a missing input. Shard "
            "artifacts carry\n"
            f"       schema {AREA_SHARD_SCHEMA!r} and are written by --shard I --num-shards N."
        )
    return found


def _by_shard_lines(items: list[tuple[int, Any]]) -> str:
    return "".join(f"         shard {i}: {v}\n" for i, v in sorted(items, key=lambda t: t[0]))


def _agreement(values: list[tuple[int, Any]]) -> tuple[bool, Any]:
    """Do the shards agree on this field, and if so on what. Comparison is on canonical JSON."""
    distinct = {json.dumps(v, sort_keys=True, default=str) for _, v in values}
    if len(distinct) == 1:
        return True, values[0][1]
    return False, None


def merge_shard_records(
    loaded: list[tuple[pathlib.Path, dict]],
) -> tuple[dict, list[str], dict]:
    """Pool the shards into the artifact a bound may sit above, or refuse.

    Returns ``(record, reasons, conditions)``. The record is always the honest arithmetic over
    whatever landed; ``conditions`` says, per name in :data:`MERGE_CONDITIONS`, whether this pool
    is the corpus's; ``reasons`` says why not. Any false condition stamps
    ``measurement_qualified: false`` into the record, which :func:`load_area_bound` refuses by
    name, and the caller exits :data:`EXIT_MEASUREMENT_NOT_QUALIFIED`.

    WHAT REFUSES INSTEAD OF STAMPING, AND WHY THE LINE IS WHERE IT IS. Everything raised below is a
    case where the POOL would be wrong rather than incomplete: an episode weighted twice, a
    per_episode entry from an episode the shard was never assigned, a shard that kept only its own
    median. There is no honest distribution to write in those cases, not even a disqualified one,
    so nothing is written. A MISSING shard is different in kind — the arithmetic over the shards
    that landed is exactly right about the frames it saw and merely is not the corpus — and that is
    what ``measurement_qualified: false`` has always meant in this file. It is written, stamped,
    and refused by the loader, exactly as a ``--limit`` run is.

    THE FIVE NUMBERS ARE TAKEN ONCE, OVER THE POOLED POPULATION. A median and a p95 do not
    decompose: the median of the shard medians is a different statistic, and on a corpus where the
    robot parks out of frame for part of every episode the two differ substantially while both look
    entirely reasonable. Shards therefore emit RAW per-frame fractions — exact through JSON, whose
    float repr is the shortest round-tripping string since Python 3.1 — and the pool is rebuilt in
    the manifest's own enumeration order, so this artifact is identical to what a single un-sharded
    run would have written.
    """
    records = [rec for _, rec in loaded]

    # -- shape --------------------------------------------------------------------------------
    for path, rec in loaded:
        block = rec.get("shard")
        if not isinstance(block, dict) or "index" not in block or "num_shards" not in block:
            raise CompositeError(
                f"FATAL: {path} declares schema {AREA_SHARD_SCHEMA!r} but carries no usable "
                "'shard' block.\n"
                "       The merge reads index, num_shards, episode_keys and episode_indices out of "
                "it; without them\n"
                "       there is nothing to check coverage against and the artifact is not a shard, "
                "whatever its schema says."
            )

    # -- REFUSAL: the shards belong to two different partitions ---------------------------------
    counts = {int(rec["shard"]["num_shards"]) for rec in records}
    if len(counts) != 1:
        raise CompositeError(
            "FATAL: the shard artifacts disagree on num_shards: "
            + ", ".join(str(c) for c in sorted(counts)) + ".\n"
            + "".join(
                f"         {p}: shard {rec['shard']['index']} of {rec['shard']['num_shards']}\n"
                for p, rec in loaded
            )
            + "       These are pieces of two DIFFERENT partitions of the corpus. Pooling them "
            "would count the episodes\n"
            "       the two partitions happen to share twice and drop the rest. Re-run one "
            "partition whole."
        )
    num_shards = counts.pop()

    # -- REFUSAL: two artifacts claim the same shard --------------------------------------------
    seen: dict[int, pathlib.Path] = {}
    for path, rec in loaded:
        idx = int(rec["shard"]["index"])
        if idx in seen:
            raise CompositeError(
                f"FATAL: two shard artifacts both claim shard index {idx} of {num_shards}:\n"
                f"         {seen[idx]}\n"
                f"         {path}\n"
                "       One of them is stale — a re-run that wrote to a new path, or a directory "
                "scan that picked up an\n"
                "       old copy. Merging both pools that shard's frames twice, which moves the "
                "median and the two\n"
                "       percentiles toward whatever those episodes did. Name the shard artifacts "
                "explicitly, or clear\n"
                "       the stale one."
            )
        seen[idx] = path

    # -- REFUSAL: the shards did not enumerate the same corpus -----------------------------------
    # ``expected`` below is taken from ONE shard's record and the tiling arithmetic is done against
    # it, so if the shards saw different enumerations that arithmetic is about a corpus that never
    # existed. Each shard digests the enumeration it saw independently, so agreement across N of
    # these is evidence that they all measured one corpus.
    digests = [(int(rec["shard"]["index"]), rec["shard"].get("corpus_episode_keys_sha256"))
               for rec in records]
    if not _agreement(digests)[0]:
        raise CompositeError(
            "FATAL: the shards enumerated different corpora (corpus_episode_keys_sha256):\n"
            + _by_shard_lines(digests)
            + "       Each shard digests the episode list it saw. Different digests mean the "
            "corpus changed between\n"
            "       shards, or they were pointed at different trees — either way the partition "
            "they belong to no\n"
            "       longer exists and its coverage cannot be proved. Re-run one partition against "
            "one corpus."
        )

    # -- REFUSAL: an episode is in a shard it does not hash to ----------------------------------
    # The merge does not take a shard's word for which episodes belong to it. Re-deriving the rule
    # catches an artifact written by an older assignment, a hand-edited file, and the
    # PYTHONHASHSEED class of bug that the rule exists to make impossible.
    for path, rec in loaded:
        idx = int(rec["shard"]["index"])
        wrong = [k for k in rec["shard"].get("episode_keys", []) if shard_of(k, num_shards) != idx]
        if wrong:
            raise CompositeError(
                f"FATAL: {path} claims shard {idx} of {num_shards} but holds {len(wrong)} "
                "episode(s) that do not hash to it: "
                + ", ".join(wrong[:8]) + ("..." if len(wrong) > 8 else "") + "\n"
                f"       The rule is {SHARD_ASSIGNMENT}, and the merge re-derives it rather than "
                "trusting the artifact.\n"
                "       A shard whose membership does not follow it was produced by a different "
                "partition rule, and the\n"
                "       other shards' coverage cannot be reasoned about alongside it."
            )

    # -- REFUSAL: a shard measured somebody else's episode, or did not account for its own -------
    for path, rec in loaded:
        block = rec["shard"]
        assigned = [str(k) for k in block.get("episode_keys", [])]
        measured = [str(ep.get("episode")) for ep in rec.get("per_episode", [])]
        stray = [k for k in measured if k not in set(assigned)]
        if stray:
            raise CompositeError(
                f"FATAL: {path} claims shard {block.get('index')} of {num_shards} but reports "
                f"per_episode entries for {len(stray)} episode(s) it was not assigned: "
                + ", ".join(stray[:8]) + ("..." if len(stray) > 8 else "") + "\n"
                "       Those frames are in this shard's pool and in whichever shard the keys hash "
                "to, so the merged\n"
                "       percentiles weight them twice while the coverage arithmetic still adds up. "
                "Re-run that shard."
            )
        if len(measured) != len(assigned):
            unaccounted = [k for k in assigned if k not in set(measured)]
            raise CompositeError(
                f"FATAL: {path} was assigned {len(assigned)} episode(s) and reports "
                f"{len(measured)} measured.\n"
                + (
                    "       UNACCOUNTED FOR (" + str(len(unaccounted)) + "): "
                    + ", ".join(unaccounted[:12]) + ("..." if len(unaccounted) > 12 else "") + "\n"
                    if unaccounted else ""
                )
                + "       The merge proves coverage from what the shards MEASURED, not from what "
                "they were assigned — a\n"
                "       shard that silently measured nothing would otherwise satisfy the coverage "
                "check while contributing\n"
                "       no frames, and the artifact would state a coverage it never had. Unlike "
                "the object masks in\n"
                "       measure_geom_tol, EVERY frame of every assigned episode yields an area "
                "fraction here — an empty\n"
                "       mask is 0.0 and is counted, never skipped — so there is no legitimate way "
                "for these two to differ.\n"
                "       Re-run that shard."
            )

    # -- REFUSAL: a shard kept only its summary --------------------------------------------------
    entries: list[tuple[int, dict]] = []
    for path, rec in loaded:
        for ep in rec.get("per_episode", []):
            if "episode_index" not in ep or "area_fractions" not in ep:
                raise CompositeError(
                    f"FATAL: {path} has a per_episode entry for {ep.get('episode')!r} with no "
                    "episode_index or no\n"
                    "       area_fractions. The merge pools the RAW per-frame area fractions — a "
                    "median and a p95 do not\n"
                    "       decompose, so a shard that reports only its own summary cannot be "
                    "merged, only averaged, and\n"
                    "       averaging percentiles is the wrong number. Re-run that shard with this "
                    "version of the script."
                )
            entries.append((int(ep["episode_index"]), ep))
    entries.sort(key=lambda t: t[0])

    # -- REFUSAL: no shard records the enumeration -----------------------------------------------
    expected: list[str] = [str(k) for k in (records[0].get("corpus_episode_keys") or [])]
    if not expected:
        raise CompositeError(
            f"FATAL: {loaded[0][0]} does not record corpus_episode_keys, so the merge cannot prove "
            "it saw every\n"
            "       episode — only that the shard indices 0..N-1 are present, which is a statement "
            "about files and not\n"
            "       about the corpus. A merge that cannot prove coverage is not a merge."
        )

    # -- the six conditions ----------------------------------------------------------------------
    conditions = {name: True for name in MERGE_CONDITIONS}
    reasons: list[str] = []

    covered: dict[str, list[int]] = {}
    for rec in records:
        for key in rec["shard"].get("episode_keys", []):
            covered.setdefault(str(key), []).append(int(rec["shard"]["index"]))
    missing_shards = sorted(set(range(num_shards)) - set(seen))
    uncovered = [k for k in expected if k not in covered]
    unexpected = [k for k in covered if k not in set(expected)]
    if missing_shards or uncovered or unexpected:
        conditions["shards_tile_the_corpus_exactly_once"] = False
        if missing_shards:
            reasons.append(
                "shard(s) " + ", ".join(str(i) for i in missing_shards) + f" of {num_shards} are "
                "missing, so this distribution is over part of the corpus"
            )
        if uncovered:
            reasons.append(
                f"{len(uncovered)} episode(s) were never measured: "
                + ", ".join(uncovered[:12]) + ("..." if len(uncovered) > 12 else "")
            )
        if unexpected:
            reasons.append(
                f"{len(unexpected)} measured episode(s) are not in the enumeration: "
                + ", ".join(unexpected[:12]) + ("..." if len(unexpected) > 12 else "")
            )

    for rec in sorted(records, key=lambda r: int(r["shard"]["index"])):
        idx = rec["shard"]["index"]
        block = rec.get("measured") or {}
        if block.get("limit") is not None or int(block.get("stride") or 1) > 1:
            conditions["every_shard_at_stride_1_with_no_limit"] = False
            reasons.append(
                f"shard {idx} was measured with limit={block.get('limit')!r} and "
                f"stride={block.get('stride')!r}; a truncated shard understates the maximum, and "
                "the pool inherits that"
            )
        if rec.get("measurement_qualified") is not True:
            conditions["every_shard_measurement_qualified"] = False
            why = "; ".join(str(r) for r in (rec.get("measurement_disqualified_reasons") or []))
            reasons.append(
                f"shard {idx} is measurement_qualified: {rec.get('measurement_qualified')!r}"
                + (f" ({why})" if why else "")
            )

    field_checks = (
        ("shards_agree_on_estimator", "estimator",
         lambda r: segmenter_identity(r.get("estimator") or {}),
         "The estimator identity is the tuple the mask cache is keyed on. Two segmenters pooled "
         "into one distribution is not a distribution, it is a mixture, and a bound above it is "
         "above neither."),
        ("shards_agree_on_source_manifest_sha256", "source_manifest_sha256",
         lambda r: r.get("source_manifest_sha256"),
         "A bound is a claim about ONE corpus. Two manifests pooled into one distribution is the "
         "same drift load_area_bound's corpus cross-check exists to catch, arriving from inside."),
        ("shards_agree_on_prompt", "prompt", lambda r: r.get("prompt"),
         "The prompt decides which pixels are protected from the generator, so it decides the area "
         "distribution. A narrower prompt yields a smaller mask and a smaller distribution."),
    )
    disagreements: dict[str, list] = {}
    agreed: dict[str, Any] = {}
    for condition, field, get, why in field_checks:
        values = [(int(rec["shard"]["index"]), get(rec)) for rec in records]
        ok, value = _agreement(values)
        if ok:
            agreed[field] = value
            continue
        conditions[condition] = False
        disagreements[field] = [{"shard": i, "value": v} for i, v in sorted(values, key=lambda t: t[0])]
        reasons.append(
            f"the shards disagree on {field}, so they did not measure one quantity:\n"
            + _by_shard_lines(values)
            + f"       {why}"
        )

    # -- pooling. Nothing above this line has looked at an area fraction. -------------------------
    fractions: list[float] = []
    empty = 0
    episode_ids: list[str] = []
    for _, ep in entries:
        fractions.extend(float(v) for v in ep["area_fractions"])
        empty += int(ep.get("empty_frames") or 0)
        episode_ids.append(str(ep.get("episode")))
    if not fractions:
        raise CompositeError(
            "FATAL: the shards named to --merge carry no area fractions between them, so there is "
            "nothing to pool.\n"
            "       That is a missing input, not an empty distribution."
        )

    total_episodes = int(
        (records[0].get("measured") or {}).get("episodes_in_manifest") or len(expected)
    )
    limits = [(rec.get("measured") or {}).get("limit") for rec in records]
    strides = [int((rec.get("measured") or {}).get("stride") or 1) for rec in records]
    measured = _measured_block(
        fractions,
        empty,
        episode_ids=episode_ids,
        total_episodes=total_episodes,
        # The pooled run's own truncation, taken from the shards rather than from this command
        # line: a merge has no --limit and no --stride and must not be able to launder one away.
        limit=next((v for v in limits if v is not None), None),
        stride=max(strides) if strides else 1,
    )

    qualified = all(conditions.values())
    record = {
        "schema": AREA_SCHEMA,
        "max_frame_fraction": None,
        "bound_rationale": "",
        "bound_note": BOUND_NOTE,
        "measurement_qualified": qualified,
        "measurement_disqualified_reasons": reasons,
        "measured": measured,
        # None when the shards disagreed. The artifact is stamped false in that case and
        # load_area_bound refuses it on the qualification check BEFORE it reads either field, so a
        # null here can never be mistaken for a segmenter or a prompt; merged_from.disagreements
        # carries what each shard actually said.
        "estimator": _agreed_estimator(records, conditions),
        "prompt": agreed.get("prompt"),
        "source_manifest": _agreed_source_manifest(records, conditions),
        "source_manifest_sha256": agreed.get("source_manifest_sha256"),
        "merged_from": {
            "num_shards": num_shards,
            "assignment": SHARD_ASSIGNMENT,
            "merged_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "shards": [
                {
                    "index": int(rec["shard"]["index"]),
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "n_episodes": int(rec["shard"]["n_episodes_in_shard"]),
                    "n_frames": int((rec.get("measured") or {}).get("frames") or 0),
                    "shard_max": (rec.get("measured") or {}).get("max"),
                    "shard_median": (rec.get("measured") or {}).get("median"),
                }
                for path, rec in sorted(loaded, key=lambda t: int(t[1]["shard"]["index"]))
            ],
            "pooling": (
                "min/median/p95/p99/max taken ONCE over the pooled per-frame area fractions. Shard "
                "percentiles are never recombined: a median does not decompose, and neither does a "
                "p95 — the median of the shard medians is a different statistic with the same "
                "units and a plausible magnitude. Shards emit raw float64 fractions (exact through "
                "JSON: float repr is the shortest round-tripping string since Python 3.1) and the "
                "pool is rebuilt in the manifest's own enumeration order, so this artifact is "
                "identical to what a single un-sharded run would have written."
            ),
            "qualification": conditions,
            "qualification_note": (
                "measurement_qualified is the AND of these. Every false one is a reason in "
                "measurement_disqualified_reasons and load_area_bound refuses this artifact by "
                "name. The refusals that write nothing at all are listed under refusals_checked "
                "and are the cases where the pool itself would be wrong rather than incomplete."
            ),
            "refusals_checked": list(MERGE_REFUSALS_CHECKED),
            "disagreements": disagreements,
            "coverage_proof": {
                "corpus_episodes": len(expected),
                "assigned_episodes": len(covered),
                "measured_episodes": len(episode_ids),
                "how": (
                    "Per shard: every measured episode is one the shard was assigned, and measured "
                    "== assigned (every frame of every assigned episode yields a fraction, an "
                    "empty mask included). Across shards: the assigned sets tile the enumeration "
                    "exactly once. Together those give measured == the corpus, which is the claim "
                    "a bound over the whole source corpus needs."
                ),
            },
        },
    }
    return record, reasons, conditions


def _agreed_estimator(records: list[dict], conditions: dict) -> Any:
    """Shard 0's estimator when the shards agreed on their identity, else None.

    The comparison that decides agreement is :func:`segmenter_identity` — the same tuple the mask
    cache is keyed on and the same one ``load_area_bound`` cross-checks — but what is WRITTEN is
    the full provenance block, so the merged artifact carries everything an un-sharded run would.
    """
    if not conditions.get("shards_agree_on_estimator"):
        return None
    by_index = {int(rec["shard"]["index"]): rec for rec in records}
    return by_index[min(by_index)].get("estimator")


def _agreed_source_manifest(records: list[dict], conditions: dict) -> Any:
    if not conditions.get("shards_agree_on_source_manifest_sha256"):
        return None
    by_index = {int(rec["shard"]["index"]): rec for rec in records}
    return by_index[min(by_index)].get("source_manifest")


def merge_main(merge: list[pathlib.Path], out: pathlib.Path) -> int:
    """``--merge``: pool the shard artifacts into the artifact a bound may sit above."""
    loaded = collect_shard_records(list(merge))
    record, reasons, conditions = merge_shard_records(loaded)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record["measured"], indent=2, sort_keys=True))

    merged = record["merged_from"]
    print(
        f"\nmerged {merged['num_shards']} shard(s), {record['measured']['episodes']} of "
        f"{record['measured']['episodes_in_manifest']} episodes, "
        f"{record['measured']['frames']} frames",
        file=sys.stderr,
    )
    for s in merged["shards"]:
        print(
            f"  shard {s['index']:>3}  {s['n_episodes']:>4} ep  {s['n_frames']:>7} frames  "
            f"max {s['shard_max']}  {s['path']}",
            file=sys.stderr,
        )
    # The shard maxima are printed and are NOT the distribution. Their spread beside the pooled
    # numbers is the cheapest possible reminder that the two are different statistics.
    print(
        "pooled min/median/p95/p99/max — taken ONCE over every frame, never recombined from the "
        "shard summaries above",
        file=sys.stderr,
    )
    print(f"\nwrote {out} with max_frame_fraction: null — read the distribution and decide.")
    if not record["measurement_qualified"]:
        failed = [k for k, v in conditions.items() if not v]
        print(
            "\nTHIS IS NOT THE MEASUREMENT A BOUND MAY SIT ABOVE. Conditions that failed: "
            + ", ".join(failed) + "\n  - " + "\n  - ".join(reasons)
            + f"\nmeasurement_qualified: false is stamped into {out} and load_area_bound refuses "
            "it.",
            file=sys.stderr,
        )
        return EXIT_MEASUREMENT_NOT_QUALIFIED
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="mode", required=True)
    measure = sub.add_parser(
        "measure",
        help="measure the robot-mask area distribution over the SOURCE corpus (sets no bound)",
    )
    # NOT required=True any more, and checked in _check_measure_flags() instead: --merge reads
    # shard artifacts and never opens a clip, never imports the segmenter and never touches a GPU,
    # which is exactly why the merge job runs on the free CPU QoS with no data mounted. Every other
    # invocation still refuses without it, with a reason attached.
    measure.add_argument("--manifest", type=pathlib.Path, default=None)
    measure.add_argument("--out", type=pathlib.Path, default=AREA_BOUND_ARTIFACT)
    measure.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "first N episodes only — A SHAKEDOWN, NOT A MEASUREMENT. Any value stamps "
            "measurement_qualified: false into the artifact and exits "
            f"{EXIT_MEASUREMENT_NOT_QUALIFIED}; load_area_bound refuses such a file."
        ),
    )
    measure.add_argument(
        "--stride",
        type=int,
        default=1,
        help=(
            "every Nth frame. Anything above 1 measures a subsample, whose maximum can only "
            "understate the corpus's, so it stamps measurement_qualified: false and exits "
            f"{EXIT_MEASUREMENT_NOT_QUALIFIED} exactly as --limit does."
        ),
    )
    measure.add_argument(
        "--shard",
        type=int,
        default=None,
        metavar="I",
        help=(
            "measure only the episodes that hash to shard I of --num-shards, and write a "
            f"{AREA_SHARD_SCHEMA} artifact carrying the RAW per-frame area fractions. The corpus "
            "is 171 625 frames against a 4 h MaxWall, so the distribution is produced by an array "
            "and merged. Assignment is a blake2b digest of the EPISODE ID, never a slice of the "
            "episode list, so adding or removing a clip moves that clip only. Refuses to write "
            "the tracked default --out."
        ),
    )
    measure.add_argument(
        "--num-shards",
        type=int,
        default=None,
        metavar="N",
        help=(
            "how many shards the corpus is partitioned into. Goes together with --shard: the "
            "merge needs the denominator before it can say a shard is missing."
        ),
    )
    measure.add_argument(
        "--merge",
        type=pathlib.Path,
        nargs="+",
        default=None,
        metavar="SHARD_JSON",
        help=(
            "pool shard artifacts into the distribution at --out. Paths may be files or "
            f"directories (a directory is scanned for {AREA_SHARD_SCHEMA} artifacts; anything else "
            "in it is skipped with a note, while a file named explicitly is never skipped). "
            "min/median/p95/p99/max are taken ONCE over the pooled per-frame fractions — shard "
            "percentiles are never recombined — and the merge refuses outright, writing nothing, "
            "on inputs that cannot be pooled (a duplicated shard, an episode in the wrong shard, a "
            "shard that kept only its summary). A pool that is honest but is not the corpus's — a "
            "missing shard, a truncated shard, shards that disagree about the segmenter, the "
            "manifest or the prompt — is written with measurement_qualified: false, the failing "
            f"conditions named, and exit {EXIT_MEASUREMENT_NOT_QUALIFIED}. --manifest is not "
            "needed and no GPU is used."
        ),
    )
    return ap


def _check_measure_flags(args: argparse.Namespace) -> None:
    """Refuse the flag combinations that would silently measure the wrong thing.

    ``parser.error`` would exit 2 with a usage block and no argument about WHY, and every one of
    these has a why that is worth more than the usage block.
    """
    merging = args.merge is not None
    sharding = args.shard is not None or args.num_shards is not None

    if merging and sharding:
        raise CompositeError(
            "FATAL: --merge and --shard/--num-shards name two different jobs on one command line.\n"
            "       --shard MEASURES one piece of the corpus; --merge POOLS the pieces into the "
            "distribution a bound\n"
            "       may sit above. Nothing here picks one and drops the other."
        )
    if merging:
        if args.manifest is not None:
            raise CompositeError(
                "FATAL: --merge does not read the corpus, so --manifest names something it will "
                "not open.\n"
                "       The shards each recorded the manifest they measured and its sha256, and "
                "the merge checks them\n"
                "       against each other — which is a stronger claim than re-reading a manifest "
                "this job was handed.\n"
                "       Drop --manifest."
            )
        if args.limit is not None or args.stride != 1:
            raise CompositeError(
                "FATAL: --merge takes no --limit and no --stride. Those truncate a MEASUREMENT, "
                "and a merge measures\n"
                "       nothing: it pools what the shards measured. Accepting them here would let "
                "a merge widen or narrow\n"
                "       a distribution after the fact, which is the one thing "
                "measurement_qualified exists to make\n"
                "       impossible. The shards' own limit and stride are read out of their "
                "artifacts and carried into the\n"
                "       merged record."
            )
        return
    if args.manifest is None:
        raise CompositeError(
            "FATAL: --manifest is required. (It is optional only under --merge, which reads shard "
            "artifacts and not\n"
            "       the corpus.)"
        )
    if sharding and args.out == AREA_BOUND_ARTIFACT:
        raise CompositeError(
            f"FATAL: --shard refuses to write the tracked default {AREA_BOUND_ARTIFACT}.\n"
            "       N array tasks writing one path is a race whose winner is whichever task "
            "finished last, and what it\n"
            "       leaves behind is one shard of the corpus sitting at the path load_area_bound "
            "reads. Give each shard\n"
            "       its own --out (the sbatch uses shard-<index>.json), and let --merge write the "
            "artifact."
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _check_measure_flags(args)
        if args.merge is not None:
            # Before the masker: a merge opens no clip and imports no segmenter, which is why the
            # merge job runs on the free CPU QoS. Building the masker here would tie the cheapest
            # step in the chain to the most expensive precondition for no reason.
            return merge_main(list(args.merge), args.out)
        masker = build_masker()
        masker.preflight()
        record = measure_source_mask_area(
            args.manifest,
            masker=masker,
            limit=args.limit,
            stride=args.stride,
            shard=args.shard,
            num_shards=args.num_shards,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(record["measured"], indent=2, sort_keys=True))
        if record["schema"] == AREA_SHARD_SCHEMA:
            block = record["shard"]
            print(
                f"\nwrote {args.out}: SHARD {block['index']} of {block['num_shards']}, "
                f"{block['n_episodes_in_shard']} episode(s). The five numbers above are this "
                "shard's own and are NOT the corpus's — a percentile does not decompose. Merge "
                "every shard of the partition:\n"
                f"  python {pathlib.Path(__file__).name} measure --merge <dir> --out "
                f"{AREA_BOUND_ARTIFACT}"
            )
        else:
            print(
                f"\nwrote {args.out} with max_frame_fraction: null — read the distribution and "
                "decide."
            )
        if not record["measurement_qualified"]:
            # The artifact is still written — a shakedown's numbers are worth reading — but the
            # shell must be able to tell it apart from the real thing without parsing JSON, and a
            # zero here is what turns a smoke test into "the measurement ran, fine".
            print(
                "\nTHIS IS NOT THE MEASUREMENT A BOUND MAY SIT ABOVE:\n  - "
                + "\n  - ".join(record["measurement_disqualified_reasons"])
                + f"\nmeasurement_qualified: false is stamped into {args.out} and load_area_bound "
                "refuses it. Re-run with no --limit and --stride 1.",
                file=sys.stderr,
            )
            return EXIT_MEASUREMENT_NOT_QUALIFIED
        return EXIT_OK
    except CompositeError as exc:
        print(f"{exc}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
