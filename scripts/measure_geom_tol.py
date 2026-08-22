#!/usr/bin/env python3
"""Measure ``GEOM_TOL`` — the tolerance PR-08 §6 G0b holds the restyled corpus to.

    GEOM_TOL := the median per-step object-centroid displacement, in pixels, in the SOURCE clips.

WHY THIS NUMBER AND NOT A CHOSEN ONE. G0b asks whether a restyle moved the geometry. The labels in
this experiment are the recorded teleop trajectory carried over unchanged, so a generator error is
only a corrupted *input* — right up until the geometry moves, at which point the carried-over label
describes a different scene than the pixels and the training pair is a lie. The tolerance for "moved
the geometry" therefore cannot be coined: it is whatever one action step actually moves the scene,
measured on the corpus itself, and committed BEFORE the first clip is generated. That is the whole
of ``T40_RULE_V1``'s G0b clause and it is why this script exists.

    .venv/bin/python scripts/measure_geom_tol.py \\
        --corpus /path/to/GR00T-N1.7-AppleToPlate --masks /path/to/apple-masks
    # -> configs/transfer25/pr08_geom_tol.json (+ .sha256), the TRACKED, committed artifact

WHERE THE NUMBER LANDS, AND WHY IT IS NOT UNDER runs/
-----------------------------------------------------
PR-08 §8 item 4 requires GEOM_TOL to be "measured and COMMITTED" before a single clip is generated.
``runs/`` is gitignored (``.gitignore``: ``runs/``), so an artifact written there can never be
committed and therefore can never be the pre-commitment the rule asks for — it is scratch that
happens to have the right shape. The default output is consequently a TRACKED path,
``configs/transfer25/pr08_geom_tol.json``, absolute and anchored to the repository root rather than
to the caller's working directory, and a ``.sha256`` sidecar is written next to it — the same
discipline ``configs/transfer25/pr08_style_partition.json`` already uses, so the file the gate reads
can be proved to be the file that was committed. ``--out`` still points anywhere for scratch or
diagnostic runs; ``--dump-displacements`` is scratch by nature and belongs under ``runs/``.

A CWD-relative default would be worse than a wrong path: the script would exit 0 having written the
gate artifact under whatever directory the caller happened to be in, while every consumer looked for
it under the repository and reported it missing.

WHAT IT REFUSES TO DO
---------------------
**It will not invent a centroid.** An object centroid needs a segmenter that can find the apple. One
is wired — ``--method sam2``, the shared PR-08 §4 adapter, see the next paragraph — and it can only
run where its checkpoints are staged. On a machine without them there is no segmenter here at all,
which is checked at run time rather than assumed, and the failure names every package and weight
directory it looked in before refusing. The obvious stand-in — threshold
the red pixels — produces a *plausible* number on a red apple photographed on a table, and a
plausible-but-wrong GEOM_TOL is the single most expensive failure this measurement has, because it
is invisible: it does not crash, it does not look odd, it just sets the gate to the wrong place and
every downstream verdict inherits it. So the heuristic exists here, but it is called
``hsv-red-diagnostic``, it can only be reached by typing that name, it stamps
``gate_qualified: false`` into the artifact, and it exits non-zero. ``--method auto`` never selects
it; with no segmenter wired, ``auto`` fails loudly and measures nothing.

**It will not select a segmenter that has not said it can run.** One real segmenter is wired:
``--method sam2`` drives ``scripts/estimators/apple_sam2.py`` — the SAME adapter
``scripts/measure_est_drift.py`` measures ``EST_DRIFT_P95`` with, because PR-08 §4 step 2 says "the
*same* segmenter" and §6 subtracts the two numbers, so two adapters would be two quantities and the
subtraction would not be arithmetic. The adapter is imported LAZILY, inside the call, never at module
import time: this module is imported by ``measure_est_drift`` and by tests that must run with no GPU,
no network and no weights, and an adapter that touches ``transformers`` at import time would drag all
three into every one of them. ``--method auto`` selects it only when the adapter itself declares its
weights are present (``available() -> bool`` or ``WEIGHTS_AVAILABLE: bool``). An adapter that is
silent about its weights is NOT selected and ``auto`` refuses exactly as it did before, because a
segmenter selected without its checkpoints does not crash: it returns empty masks, every step drops,
and ``coverage: 0.0`` reads as a fact about the corpus rather than about the missing weights.

**It will not decode a frame after the adapter has already said no.** ``--method sam2`` is the
explicit spelling, and explicit is not a licence to ignore an answer that was already given: when the
adapter's ``available()`` returns False, the weights are absent, and the first ``segment()`` call
will raise. Refusing there — before a single frame is decoded, exit 2, no artifact — is the same
refusal one frame earlier, and it keeps the failure inside the EXIT STATUS table below instead of a
traceback out of ``main``. The one way past it is the adapter declaring ``ALLOW_DOWNLOAD = True``
(``WAM_PR08_ALLOW_DOWNLOAD=1``), which is a human saying "fetch them, on purpose". Whatever the
adapter raises later — a missing checkpoint found mid-run, a CUDA failure — is caught at the call
site and re-raised as this module's own fatal refusal, with the adapter's message verbatim. No
partial artifact, no plausible number.

**It will not take two segmenters on one command line.** ``--method sam2 --masks DIR`` names two,
and the old order of tests silently used one and dropped the other's provenance. It is refused: the
artifact can only record which estimator produced GEOM_TOL if the command line only had one.

THE JOIN KEY: THE FIELD THAT MAKES §4 STEP 2 CHECKABLE
------------------------------------------------------
``measure_est_drift.cross_check_geom_tol()`` reads this artifact and copies ``mask_method`` into its
own ``geom_tol_cross_check`` block. **The two rigs join on ``mask_method.name``**, which for the
sam2 method is the adapter's own ``ESTIMATOR_NAME`` — the identical string, read off the identical
module, that ``measure_est_drift`` records as ``estimators.name`` (``Estimators.__init__``). Equality
is therefore by construction and not by convention: there is no second place where a segmenter is
named. The pixel grid joins on ``resolution_hw`` (``[height, width]``), which is the key that
function looks for and which is written here for that reason; ``frame_width``/``frame_height`` stay
as they were.

**Both consumer-side limits this section used to record are now CLOSED**, on 2026-08-22, by the
change to ``scripts/measure_est_drift.py`` that this module deliberately did not make when the join
key was introduced. Recorded rather than deleted, because the artifact fields that exist to make
them survivable — ``cross_check_limits``, ``consumer_asserts`` — were built for them and are still
written:

1.  ``cross_check_geom_tol()`` now ASSERTS that the two names are equal, rather than only recording
    ``mask_method``. A disagreement is the disqualifying reason ``mask_method_disagrees_with_estimator``.
    Until that landed, ``mask_method.name != estimators.name`` was visible in both artifacts and
    caught by nobody, and a consumer of ``GEOM_TOL - EST_DRIFT_P95`` had to check it by hand — which
    is why it is still the FIRST line of ``consumer_asserts``: a reader holding two OLD artifacts
    gets no benefit from a check that only runs at measurement time.
2.  That comparison is no longer ABSENCE-PERMISSIVE. ``if theirs_hw is not None and ... != ...``
    meant an artifact with no ``resolution_hw`` at all passed the grid check by saying nothing —
    exactly what a hand-written or a stale artifact looks like. Each field the reader needs now
    disqualifies by its own name (``geom_tol_does_not_record_<field>``) when it is absent.

This side keeps the belt regardless: every field that function reads is checked present and non-null
BEFORE the artifact is written (``CROSS_CHECK_FIELDS_REQUIRED``, ``missing_cross_check_fields()``),
and a record missing one is a fatal refusal with nothing written rather than an artifact that reads
clean downstream. A test asserts that the set of fields ``cross_check_geom_tol`` actually reads —
parsed out of its source, not copied from it — is the set this module guarantees, so the day the
reader grows a field, this module is told. That test is what caught the join key being added to the
reader without being declared here.

**It will not overwrite the pre-commitment it was measured under.** The default ``--out`` already
holds the COMMITTED SEGMENTER CONTRACT — the detector, the segmenter, the depth model, their pinned
revisions, the prompt, both threshold pairs, the box rule and the pixel grid, written down before
the number so that PR-08 §4 step 2's "the same segmenter" is a checkable claim and not a
recollection. Until 2026-08-22 the first real GEOM_TOL run replaced that file with a document that
mentioned no segmenter anywhere: the pre-commitment was destroyed by the measurement it existed to
constrain, and ``measure_est_drift`` then refused every later run with
``geom_tol_does_not_record_segmenter_params`` — closed, and closed forever.
``merge_committed_contract()`` now runs before a byte is written on BOTH paths (measure and
``--merge``): it compares the committed block field for field against the adapter this run drove,
refuses the whole run on any disagreement, and otherwise copies the contract section forward
verbatim into the artifact. ``refuse_default_out_without_contract()`` covers the other half — the
tracked path may not be written when no contract is sitting in it, because an artifact measured
against nothing looks exactly like one measured against the contract. The document is one file with
two sections (``contract_fields`` / ``measurement_fields``) rather than two files because three
consumers already resolve the tolerance AND its segmenter through this single path; see
:data:`CONTRACT_SECTION_FIELDS` for that argument in full.

**It will not average away a missing object.** When the Dex3 hand occludes the apple, or the apple
leaves frame, that step has no displacement. It is DROPPED and COUNTED — never folded in as a zero.
Zeros would pull the median down, which tightens the gate, which looks conservative and is simply
wrong. ``coverage`` is in the artifact and a run below ``--min-coverage`` reports its number with
``headline_valid: false`` rather than quietly.

**It will not let a smoke test become the gate.** ``--limit`` and ``--max-frames`` exist so the
pipeline can be exercised in seconds, and they are exactly the shape of a silent corruption:
``coverage`` is a fraction of the steps that were ACTUALLY DECODED, so ``--limit 3`` over a
402-episode corpus reports ``coverage: 1.000`` — a perfect score over 0.7% of the corpus — and every
other field looks like a finished measurement. Any non-zero ``--limit`` or ``--max-frames`` therefore
forces ``gate_qualified: false``, records the reason in ``gate_disqualified_reasons``, and exits 3.
``n_episodes`` and ``n_episodes_found`` are both written (the latter counted BEFORE ``--limit``
truncates), so a consumer can assert they match rather than trusting a flag.

**It will not compare pixels across resolutions.** Every clip must share one frame geometry. §4
subtracts ``EST_DRIFT_P95`` (also in pixels) from this number, and that subtraction is arithmetic
only if both were measured on the same grid. Mixed geometry is fatal.

THREE PLACES PR-08 §6 IS SILENT, RECORDED RATHER THAN RESOLVED BY FIAT
----------------------------------------------------------------------
``T40_RULE_V1`` is registered and is not edited (``docs/handoff.md`` §3). None of these makes it
wrong; each is a place it does not say, so this script pins a choice, records the choice in the
artifact, and makes it a flag so the other reading is one argument away.

1.  **"per-step" is undefined.** At 30 fps a step could be one frame or one control tick, and
    GEOM_TOL scales roughly linearly with whichever it is — a 10x misreading is a 10x wrong gate.
    This script defaults to ONE FRAME at source fps and writes ``step_frames``, ``fps`` and
    ``step_seconds`` into the artifact. If the action column later shows the control tick is a
    decimation of frames, re-run with ``--step-frames`` and the two numbers sit side by side.
2.  **Units are implied.** §4 fixes ``EST_DRIFT_P95`` as "centroid displacement in pixels" and §6
    subtracts it, so GEOM_TOL is pixels at the source resolution. The artifact says so explicitly,
    with the width and height it was measured at.
3.  **Object vs plate.** G0b's prose gates "Object *and* plate centroids"; the tolerance is derived
    from the OBJECT centroid alone. The plate is near-static, so a tolerance derived from the apple
    is loose for the plate. That does not block computing GEOM_TOL; it blocks applying one number to
    both, and the artifact carries the warning rather than silently widening the plate's budget.

WHAT THIS DOES NOT UNBLOCK. G0b holds the generator to ``GEOM_TOL - EST_DRIFT_P95``, and
``EST_DRIFT_P95`` is a separate measurement this script does not make: §4 steps 1-4 need Isaac
ground-truth depth and segmentation, which need the annotators of §4 step 0. The artifact records
``est_drift_p95_px: null`` together with ``est_drift_p95_blocked_by``, which is re-derived from
``src/wam/robot/isaac_binding.py`` on every run rather than written down here and left to go stale.
GEOM_TOL is worth committing early on its
own — it is a property of the corpus, not of the generator — but it does not on its own license
generation, and neither does anything else here.

THE MEASUREMENT DOES NOT FIT THE MACHINE, SO IT IS SHARDED — AND THE MEDIAN IS THE HARD PART
--------------------------------------------------------------------------------------------
The pilot (cluster job 189588, ``runs/pr08-geom-tol/GEOM_TOL_PILOT.json``) measured the full run at
**4.005 GPU-h** and wrote ``single_job_feasible: false`` with ``recommended_time_limit: 05:30:00``.
Discoverer+ enforces ``MaxWall = 04:00:00`` on *every* QoS (``docs/discoverer.md`` §QoS) and
``MaxJobsPU = 4``, so the committed number cannot be produced by one job however the request is
written. It is produced by several and then joined:

    # one array task each, N of them, each writing its own shard artifact
    measure_geom_tol.py --corpus C --method sam2 --shard I --num-shards N --out runs/.../shard-I.json
    # and then, once, the committed artifact:
    measure_geom_tol.py --merge runs/.../shard-*.json --out configs/transfer25/pr08_geom_tol.json

**Why ``--shard I --num-shards N`` over an ``--episode-range A:B``.** Both compose with
``find_episodes()``, which already returns one sorted, stable enumeration. A range does not compose
with anything else: it is an index into that list, so inserting or dropping one clip **renumbers
every episode after it** and silently re-partitions the corpus — and this is a *resumable chain*,
where shard 3 may be computed on Tuesday and shard 7 re-run on Wednesday after a preemption. With
ranges, a corpus that grew by one clip in between yields a set of shard artifacts that overlap on
some episodes and skip others, and every one of them is individually well-formed. The assignment
here is instead a stable digest of the **episode key**::

    shard(key) = int.from_bytes(blake2b(key.encode(), digest_size=8).digest(), "big") % num_shards

so adding or removing an episode moves **that episode only** and leaves every other episode where it
was. It is ``blake2b`` and not ``hash()`` for a reason that is not style: ``PYTHONHASHSEED`` is
randomised per interpreter, so ``hash(key) % N`` would assign the same episode to a different shard
in every task of the same array — producing duplicates and gaps at once, from code that looks
deterministic. The rule is recorded in every shard artifact and the merge **re-derives it** and
refuses a shard holding an episode that does not hash to it.

**A shard artifact is not a GEOM_TOL and cannot be mistaken for one.** It carries
``schema: wam.geom_tol_shard/1``, ``is_shard: true`` and ``GEOM_TOL_px: null`` — its own median is
recorded as ``shard_median_px``, a diagnostic. Its ``gate_qualified`` means one thing only, *this
shard is fit to be merged*: a gate-qualified mask method, no ``--limit``/``--max-frames``, coverage
over the floor. Whether the CORPUS was covered is not a question a shard can answer, and it does not
pretend to. ``--shard`` also refuses to write the tracked default ``--out``, because N array tasks
writing one path is a race whose winner is whichever task finished last.

**The median does not decompose, so nothing is summarised before the merge.** The median of N shard
medians is not the median of the pooled displacements — it is a different statistic with the same
units and a plausible magnitude, and on a bimodal park-then-transfer corpus the two can differ by a
lot while both look entirely reasonable. Nothing downstream re-derives GEOM_TOL, so that error would
be permanent and invisible: the single worst failure this measurement has. Each shard therefore
emits **every per-step displacement it measured**, per episode, and the merge takes ONE median over
the pooled set.

**The representation is exact, not approximate, and it is exact for a stated reason.** Displacements
are float64 and are written into the shard artifact as JSON numbers. ``json.dumps`` renders a float
with ``repr``, which since Python 3.1 is the *shortest string that round-trips*, and ``json.loads``
parses it back with ``float()`` — so ``float -> JSON -> float`` is the identity on every finite
float64, with no error bound to prove because there is no error. The merge additionally rebuilds the
pooled array **in the corpus's own enumeration order** (every episode carries its ``episode_index``,
its position in the un-sharded enumeration), so the concatenated array the merge medians is the
*same array in the same order* the un-sharded run would have built. That is stronger than it needs
to be for the median, which is order-invariant, and it is what makes ``mean_px``, ``std_px`` and the
histogram — which are **not** order-invariant in floating point — identical too. A test asserts the
merged artifact equals the un-sharded artifact exactly, field for field, on the same fixture.

**What the merge refuses, each with its own message, because a merge that cannot prove it saw every
episode is not a merge:** a shard is missing (or two claim the same index, or they disagree on
``num_shards``); the shards did not enumerate the same corpus; they disagree on the mask method, on
the decoder, on the pixel grid, or on ``step_frames``; one reports ``gate_qualified: false`` while
another reports true; a shard holds an episode that does not hash to it; or the union of covered
episodes is not the full corpus. None of these is a warning and none has a permissive branch.

WHAT THE ESTIMATOR SAW, RECORDED BESIDE WHAT WAS MEASURED
---------------------------------------------------------
The artifact carries an ``estimator_stats`` block: what ``estimators.apple_sam2.stats()`` said after
the pass, and what THIS RUN did to its counters — frames with no detection, frames whose mask came
back empty, frames where upstream's ``(0.10, 0.10)`` retry fired and the ones it recovered, plus the
distribution of the winning detection scores. It exists because the adapter's own second
gate-qualification blocker asks for exactly those numbers *from a full pass*, and until 2026-08-22
this script read none of them: the 402-episode, ~171 600-frame run that would produce the evidence
threw it away. The scores matter as much as the counts — the part of the distribution below the
adapter's ``box_threshold`` can only have come from the retry, so "how much of ``coverage`` did the
retry buy" is read off the values instead of assumed.

Three properties, none of them free. The counters are SNAPSHOTTED AND DIFFERENCED, because the
adapter's are lifetime totals of the import and two runs in one interpreter would otherwise each
record the other's frames. The scores are carried RAW, per episode, in the shard artifacts, and the
merge concatenates them in the corpus's own enumeration order before binning — the same rule the
displacements follow, and for the same reason: a distribution decomposes no better than a median
does. And an adapter that exports no ``stats()`` is recorded as ABSENT WITH A REASON, never as
zeros, because "nobody looked" and "it never happened" are different claims that a later reader has
to be able to tell apart.

**It is additive and it discharges nothing.** No refusal reads it, no exit code depends on it, and
it is not an input to ``gate_qualified``; a test runs the previous version of this script over the
same fixture and asserts every field it already wrote is unchanged. Recording evidence is not
accepting it — ``GATE_QUALIFIED`` in the adapter, and the blocker this serves, are a human's to
retire.

EXIT STATUS
-----------
0   measured with a gate-qualified mask method, coverage above ``--min-coverage``.
2   fatal: nothing was measured and NOTHING was written — no segmenter, no clips, no mask
    provenance, mixed geometry, two segmenters named on one command line, an adapter that says its
    checkpoints are absent, or an estimator that raised while segmenting. **Every merge refusal
    lands here too** — a missing shard, a disagreement between shards, an incomplete corpus — and
    writes nothing, because the merge's output IS the committed GEOM_TOL and a partial one has no
    honest form. Every way this script can fail lands here; a traceback out of ``main`` would be a
    bug in the script, not a fourth status.
3   measured, but the number MUST NOT be used as G0b's tolerance — an ungated mask method, coverage
    below the floor, or a partial run (``--limit`` / ``--max-frames``). The artifact is still
    written: PR-08 §6 requires GEOM_TOL to be recorded regardless of verdict, and "we tried and this
    is what came out" is a record. ``gate_disqualified_reasons`` says which of the four it was.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from prepare_cosmos_corpus import resolve_camera  # noqa: E402

SCHEMA = "wam.geom_tol/1"

#: A SHARD's schema, deliberately different from ``SCHEMA``. A shard artifact is a partial
#: measurement that is *fit to be merged*, not a GEOM_TOL, and the one thing that must never happen
#: is for one of them to be committed as the number. A consumer that checks ``schema`` rejects it
#: without having to know that sharding exists at all.
#:
#: NO CONSUMER CHECKS IT TODAY, and that is stated here rather than assumed: neither
#: ``97_transfer25_restyle.sbatch``'s GEOM_CONSTANTS block nor ``scripts/run_g0_gates.py``
#: ``gate_budget()`` reads ``schema`` off this artifact — both go straight for the number. So the
#: field that actually stops a shard from being quoted as GEOM_TOL is ``GEOM_TOL_px: null``, which
#: BOTH of them refuse on ("carries no GEOM_TOL" / "records GEOM_TOL_px = null"), with
#: ``is_shard: true`` and this schema as the belt beside those braces. If a consumer ever starts
#: checking ``schema``, this comment becomes true rather than aspirational; until then the null is
#: the load-bearing one and must not be "helpfully" filled in with ``shard_median_px``.
SHARD_SCHEMA = "wam.geom_tol_shard/1"

WRITEUP = "docs/preregistration/PR-08-photoreal-augmentation.md"

#: How an episode is assigned to a shard. Spelled once, recorded into every shard artifact, and
#: RE-DERIVED by the merge — the merge does not take a shard's word for which episodes belong to it.
SHARD_ASSIGNMENT = "int.from_bytes(blake2b(episode_key.utf8, digest_size=8).digest(), 'big') % num_shards"

#: The COMMITTED gate artifact. Tracked, not gitignored, and anchored to the repository root rather
#: than to the caller's CWD — PR-08 §8 item 4 wants GEOM_TOL *committed* before generation, and a
#: path under ``runs/`` (gitignored) or a path that moves with the shell's CWD cannot be that. See
#: the module docstring; ``configs/transfer25/pr08_style_partition.json`` is the precedent.
DEFAULT_OUT_REL = "configs/transfer25/pr08_geom_tol.json"
DEFAULT_OUT = _REPO_ROOT / DEFAULT_OUT_REL

#: The keys of the CONTRACT SECTION of that document — the half that is committed BEFORE the
#: measurement and that no measurement may alter. Carried forward VERBATIM by
#: :func:`merge_committed_contract` into every artifact written over the contract.
#:
#: ONE FILE, TWO SECTIONS, AND WHY NOT TWO FILES. The alternative considered was a separate
#: pre-measurement contract at its own path, with the measurement carrying a copy into its own
#: artifact. It was rejected for a reason that is specific rather than aesthetic: three consumers
#: already resolve "the tolerance and the segmenter that produced it" through ONE path.
#: ``measure_est_drift.GEOM_TOL_ARTIFACT``, ``run_g0_gates.GEOM_CONFIG_DEFAULT`` and
#: ``102_stage_sam2_weights.sbatch``'s ``artifact_id()`` all name
#: ``configs/transfer25/pr08_geom_tol.json``, and ``run_g0_gates`` is explicitly built for it —
#: ``GEOM_TOL_KEYS`` accepts both spellings of the number and ``config_instrument()`` accepts both
#: spellings of the segmenter block, "one path written by two producers", in its own words.
#: Splitting the file would have moved the number away from the path those three read, i.e. it
#: would have fixed the overwrite by breaking the join, and two of the three are other people's
#: files. So the contract stays where its consumers already look, and it is protected by a
#: REFUSAL at the write site instead of by a filename.
#:
#: What makes "committed before" checkable is not the path: it is that these keys go into git
#: before the first measurement and come back out of the measured artifact byte for byte. A run
#: that would change any of them writes nothing at all.
CONTRACT_SECTION_FIELDS: tuple[str, ...] = (
    "spec_version", "what_this_is", "contract_fields", "measurement_fields", "segmenter",
)

#: The MEASUREMENT SECTION's slots: null in the committed contract, filled by the measurement.
#: ``geom_tol_px`` is filled to the SAME value as this module's own ``GEOM_TOL_px`` on purpose —
#: ``run_g0_gates._first_present`` refuses a document that states one quantity under two spellings
#: that disagree, and a null left beside a measured number is precisely that disagreement, so
#: leaving the contract's slot alone would make the gate unreachable in a new way.
#:
#: ``est_drift_estimator_name`` is the JOIN KEY and is a measurement slot rather than a courtesy.
#: PR-08 §4 step 2 requires both halves of ``GEOM_TOL - EST_DRIFT_P95`` to come from ONE segmenter;
#: ``run_g0_gates._ca_mask_method_name`` is the consumer that checks it, and until 2026-08-22 this
#: producer wrote no spelling of it at all — so that assertion could only ever come back "could not
#: check", which costs the run its gate qualification. A G0b that structurally cannot return 0 is
#: as blocking as a wrong one. The slot exists so the name is carried BESIDE the number instead of
#: living in the memory of whoever merged the two artifacts, and
#: :func:`refuse_unnamed_est_drift` makes writing the number without it impossible.
CONTRACT_MEASUREMENT_FIELDS: tuple[str, ...] = (
    "geom_tol_px", "geom_tol_source",
    "est_drift_p95_px", "est_drift_source", "est_drift_estimator_name",
    "gate_margin_px",
)

#: The one spelling this producer writes for the EST_DRIFT_P95 half's segmenter name, and the first
#: spelling ``run_g0_gates._ca_mask_method_name`` looks for.
EST_DRIFT_NAME_FIELD = "est_drift_estimator_name"

#: Fraction of steps that must yield a displacement before the median is called a measurement.
#: Not a threshold on the corpus — a threshold on how much of the corpus the estimator could see.
DEFAULT_MIN_COVERAGE = 0.90

#: Histogram resolution for the recorded distribution, in pixels.
DEFAULT_HIST_BIN_PX = 0.5

EXIT_OK = 0
EXIT_FATAL = 2
EXIT_NOT_GATE_QUALIFIED = 3

#: Segmenters that could produce a gate-qualified apple mask, and what each one needs. Probed at
#: run time so the failure message names what is actually absent on THIS machine rather than a
#: list someone wrote down once.
CANDIDATE_SEGMENTERS: tuple[tuple[str, str], ...] = (
    ("sam2", "SAM 2 — `sam2` package plus a hiera checkpoint, prompted with a point/box"),
    ("segment_anything", "SAM 1 — `segment_anything` package plus a ViT-H/L/B checkpoint"),
    ("ultralytics", "YOLO-seg — `ultralytics` package plus a `*-seg` checkpoint"),
    ("groundingdino", "GroundingDINO — text-prompted 'apple' box, then a mask head"),
)

#: Where local weights would be if anyone had fetched them. Globbed, never fetched.
WEIGHT_SEARCH_GLOBS: tuple[tuple[Path, str], ...] = (
    (Path.home() / "models", "*"),
    (Path.home() / ".cache" / "huggingface" / "hub", "models--*"),
)
WEIGHT_NAME_HINTS = ("sam", "seg", "dino", "owl")

#: The ONE estimator adapter both halves of PR-08 §4 go through. ``measure_est_drift.py`` imports it
#: as ``--estimators estimators.apple_sam2``; this script reaches the identical module through
#: ``--method sam2``. Spelled once, here, because §4 step 2's "the same segmenter" is only checkable
#: if there is exactly one place that says which one it is.
SAM2_ADAPTER_SPEC = "estimators.apple_sam2"
SAM2_ADAPTER_FILE = _REPO_ROOT / "scripts" / "estimators" / "apple_sam2.py"

#: The module attribute an adapter uses to say a fetch has been authorised. ``available()`` is
#: deliberately False while the checkpoints are absent even when a download IS permitted (see the
#: adapter's own docstring), so this is the ONE thing that distinguishes "the weights are missing
#: and nobody said to get them" from "the weights are missing and a human said fetch them". Read off
#: the adapter, never inferred from an environment variable read here.
ADAPTER_DOWNLOAD_ATTR = "ALLOW_DOWNLOAD"

#: Every field ``measure_est_drift.cross_check_geom_tol()`` reads out of this artifact. Listed here
#: so the two ends can be shown to agree by a test that PARSES that function rather than by two
#: prose paragraphs that drift. ``frame_hw`` is its legacy fallback for ``resolution_hw`` and is not
#: written by this module.
CROSS_CHECK_FIELDS_READ_BY_EST_DRIFT: tuple[str, ...] = (
    "resolution_hw", "frame_hw", "gate_qualified", "mask_method", "name",
    # Added 2026-08-22, when the reader stopped judging "the same segmenter" by its NAME. It now
    # compares the committed SEGMENTER BLOCK — prompt, both thresholds, the retry pair, the box
    # rule, the propagation mode, the checkpoint pins — field for field against the adapter's
    # ``SEGMENTER_CONTRACT``, and falls back to that block's own ``method_name`` and
    # ``pixel_grid_hw`` when the document is the pre-measurement CONTRACT
    # (``configs/transfer25/pr08_geom_tol.json`` as committed before either number exists) rather
    # than a measured artifact. Those two names are what the reader's ``doc.get`` calls now name.
    #
    # THIS MODULE NOW WRITES THAT BLOCK, IN BOTH PLACES THE READER LOOKS, AND THEY MEAN DIFFERENT
    # THINGS ON PURPOSE (closed 2026-08-22; apple_sam2 blocker 3):
    #   ``mask_method.params.segmenter``  — what the adapter this run actually drove declared.
    #                                        Written on every artifact, at any --out.
    #   top-level ``segmenter``            — the COMMITTED contract, copied forward verbatim from
    #                                        the document already at --out by
    #                                        merge_committed_contract(), which refuses the whole
    #                                        run if the two disagree in any field.
    # So a top-level block always means "this came out of the pre-commitment" and a params block
    # always means "this is what ran", and an artifact carrying both has had them compared. Before
    # this existed, the first real GEOM_TOL run overwrote the committed contract with a document
    # that had no ``segmenter`` anywhere, and every later est_drift run refused with
    # ``geom_tol_does_not_record_segmenter_params`` — failing closed, and closed forever.
    #
    # ``segmenter`` and ``params`` are the reader's own ``doc.get`` literals, and they are declared
    # here because the reader's lookup moved into the helper ``committed_segmenter_contract()``:
    # the guard test walks that helper too, precisely so a read that moves out of
    # ``cross_check_geom_tol``'s body cannot leave this tuple understating what is read.
    "method_name", "pixel_grid_hw", "segmenter", "params",
)

#: The subset this module GUARANTEES to write, present and non-null, in every artifact it produces.
#: It used to matter more than it does: the reader's grid comparison was absence-permissive
#: (``if theirs_hw is not None and ...``), so an artifact missing ``resolution_hw`` passed its grid
#: check by saying nothing, and a check that says nothing reads downstream as a check that passed.
#: **Repaired on the reader's side 2026-08-22** — ``cross_check_geom_tol`` now disqualifies on
#: ``geom_tol_does_not_record_<field>`` for each of these. Guaranteeing them here is still the
#: right thing: it means this module never hands the reader an artifact that trips that refusal.
CROSS_CHECK_FIELDS_REQUIRED: tuple[str, ...] = ("resolution_hw", "gate_qualified", "mask_method")

#: The CLI spelling. Distinct from the method NAME that lands in the artifact, which is the adapter's
#: ESTIMATOR_NAME — see the join key in the module docstring.
SAM2_METHOD_CLI = "sam2"

#: Checkpoints Cosmos-Transfer2.5 itself names for this pipeline
#: (``cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_model.py``), so the generator's own
#: segmenter is the one the tolerance is measured with. Used ONLY to make the failure message
#: concrete: the artifact quotes checkpoints the ADAPTER declares and never these, because what
#: Cosmos names upstream is not evidence about what this adapter loaded.
COSMOS_SAM2_CHECKPOINT_HINTS: tuple[tuple[str, str], ...] = (
    ("SAM2_MODEL_CHECKPOINT", "facebook/sam2-hiera-large"),
    ("GROUNDING_DINO_MODEL_CHECKPOINT", "IDEA-Research/grounding-dino-base"),
)

#: Module attributes the adapter may use to name the weights it loaded. Anything found here goes
#: verbatim into ``mask_method.params.checkpoints`` and into the provenance string.
CHECKPOINT_ATTRS: tuple[str, ...] = (
    "SAM2_MODEL_CHECKPOINT",
    "GROUNDING_DINO_MODEL_CHECKPOINT",
    "DEPTH_MODEL_CHECKPOINT",
)


# -- mask methods --------------------------------------------------------------------------------
#
# A mask method turns one frame (or one stored mask) into a binary object mask. Every method is
# named, versioned and stamped gate_qualified into the artifact, because the identical estimator has
# to be re-runnable on the restyled clips at gate time — a tolerance measured with one estimator and
# applied with another compares two different quantities.


class MethodUnavailable(RuntimeError):
    """Raised when the requested mask method cannot run here. Always fatal, never fallen back from."""


@dataclass
class MaskMethod:
    name: str
    version: str
    gate_qualified: bool
    #: Where the frames come from: "video" decodes the clips, "masks" reads them off disk.
    frames_from: str
    params: dict[str, Any] = field(default_factory=dict)
    provenance: str = ""
    #: frame -> object mask, for methods whose ``frames_from`` is "video". Held ON the method rather
    #: than looked up by name in the decode loop: the sam2 method's ``name`` is the ADAPTER's
    #: ESTIMATOR_NAME (that is the join key §4 step 2 needs), so a name-keyed branch would stop
    #: matching the day the adapter renames itself — and would then quietly segment with something
    #: else while still stamping the adapter's name into the artifact. Never serialized; the
    #: ``mask_method`` block is built field by field.
    mask_fn: "Callable[[np.ndarray, MaskMethod], np.ndarray] | None" = None
    #: The estimator ADAPTER MODULE behind this method, when there is one, so the run can record
    #: what it saw (:class:`EstimatorStatsProbe`). Held for the same reason ``mask_fn`` is: it is
    #: the object the frames actually went through, not a name that might resolve to a second
    #: import of the same file. None for every method that is not an adapter — the hsv diagnostic
    #: and the precomputed-mask reader have no estimator to ask, and that is recorded as absent
    #: rather than as zeros. Never serialized.
    stats_module: Any = None


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _local_weight_hits() -> list[str]:
    """Directory names under the local weight roots that look like a segmenter. Read-only."""
    hits: list[str] = []
    for root, pattern in WEIGHT_SEARCH_GLOBS:
        if not root.is_dir():
            continue
        try:
            for entry in sorted(root.glob(pattern)):
                low = entry.name.lower()
                if any(h in low for h in WEIGHT_NAME_HINTS):
                    hits.append(str(entry))
        except OSError:
            continue
    return hits


def no_segmenter_message() -> str:
    """The loud failure. Names every place that was looked in and what would have to change."""
    present = [(m, why) for m, why in CANDIDATE_SEGMENTERS if _importable(m)]
    absent = [(m, why) for m, why in CANDIDATE_SEGMENTERS if not _importable(m)]
    weights = _local_weight_hits()

    lines = [
        "FATAL: no gate-qualified object segmenter is wired, so no object centroid can be computed",
        "       and GEOM_TOL cannot be measured. Nothing was written.",
        "",
        f"       interpreter: {sys.executable}",
        "",
        "       segmenter packages NOT importable by this interpreter:",
    ]
    lines += [f"         - {m:<18} {why}" for m, why in absent] or ["         (none — see below)"]
    if present:
        lines += ["", "       importable, but this script has no code path for them yet:"]
        lines += [f"         - {m:<18} {why}" for m, why in present]
    lines += ["", "       local segmentation weights found:"]
    lines += [f"         - {w}" for w in weights] or [
        "         (none — nothing matching *sam*/*seg*/*dino*/*owl* under "
        + ", ".join(str(r) for r, _ in WEIGHT_SEARCH_GLOBS) + ")"
    ]
    lines += [
        "",
        "       Three ways forward, in order of how little they cost:",
        "",
        "       1. Run a segmenter elsewhere, dump per-frame masks, and point --masks at them:",
        "            <masks>/masks.meta.json          {\"method\": ..., \"version\": ...,",
        "                                              \"gate_qualified\": true}",
        "            <masks>/<clip-stem>/000000.npy   (or .png) one binary mask per frame",
        "          This script then measures with --method precomputed and records that",
        "          provenance verbatim, so the same estimator can be re-run at gate time.",
        "",
        "       2. Fetch a segmenter (SAM2-Hiera-tiny is ~150 MB). That is a download, and the",
        "          repo rule is that nothing is downloaded at scale without asking first. ASK.",
        "",
        "       3. --method hsv-red-diagnostic thresholds red pixels. It needs nothing, it is NOT",
        "          a segmenter, its output is stamped gate_qualified: false, and it exits 3. It",
        "          exists to exercise this pipeline end to end, never to set G0b's tolerance.",
        "",
        # Re-derived from src/wam/robot/isaac_binding.py at run time, exactly as the artifact's
        # est_drift_p95_blocked_by is, and for the same reason: that file is under active change
        # (commit 5ef3535 wired two annotators this message used to say were absent), and a refusal
        # that prints a hardcoded fact next to a computed one teaches the reader to trust neither.
        "       Related, and separately blocking — EST_DRIFT_P95, checked against the source just",
        "       now rather than asserted here:",
        f"         {_est_drift_blocker()}",
    ]
    return "\n".join(lines)


def load_precomputed_method(masks_root: Path) -> MaskMethod:
    """Read ``masks.meta.json`` and take the segmenter's word for what it is — in writing.

    The sidecar is mandatory. A directory of masks with no statement of what produced them cannot be
    recorded in the artifact, and an artifact that cannot say which estimator produced GEOM_TOL
    cannot be re-run against the restyled clips, which is the only thing the number is for.
    """
    if not masks_root.is_dir():
        raise MethodUnavailable(f"FATAL: --masks {masks_root} is not a directory.")
    sidecar = masks_root / "masks.meta.json"
    if not sidecar.is_file():
        raise MethodUnavailable(
            f"FATAL: {sidecar} is missing.\n"
            "       Masks with no provenance cannot be recorded, and GEOM_TOL is only usable if\n"
            "       the identical estimator can be re-run on the restyled clips at gate time.\n"
            '       Write {"method": "<segmenter>", "version": "<pinned rev>", '
            '"gate_qualified": true}.'
        )
    try:
        meta = json.loads(sidecar.read_text())
    except json.JSONDecodeError as exc:
        raise MethodUnavailable(f"FATAL: {sidecar} is not valid JSON: {exc}") from exc
    missing = [k for k in ("method", "version") if not meta.get(k)]
    if missing:
        raise MethodUnavailable(
            f"FATAL: {sidecar} does not declare {', '.join(missing)}. "
            "An unnamed or unversioned estimator is not provenance."
        )
    # An unstated claim is not a claim: gate_qualified defaults to False.
    return MaskMethod(
        name=str(meta["method"]),
        version=str(meta["version"]),
        gate_qualified=bool(meta.get("gate_qualified", False)),
        frames_from="masks",
        params={k: v for k, v in meta.items() if k not in ("method", "version", "gate_qualified")},
        provenance=str(sidecar),
    )


def hsv_red_method(min_area: int) -> MaskMethod:
    """The heuristic. Deliberately named so nobody can pass it by accident or quote it by mistake."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - cv2 is in the venv
        raise MethodUnavailable(
            "FATAL: --method hsv-red-diagnostic needs cv2 and this interpreter has none: "
            f"{sys.executable}"
        ) from exc
    return MaskMethod(
        name="hsv-red-diagnostic",
        version=f"1 (cv2 {cv2.__version__})",
        gate_qualified=False,
        frames_from="video",
        params={
            "hue_lo": [0, 10], "hue_hi": [170, 180], "sat_min": 100, "val_min": 60,
            "open_kernel": 3, "min_area_px": min_area,
            "component": "largest connected component by area",
        },
        # Behaviourally identical to calling hsv_red_mask directly, which is what the decode loop
        # used to do. It is attached here so that the loop has ONE way to reach a segmenter and no
        # default to fall back to — see the refusal in episode_centroids_from_video.
        mask_fn=hsv_red_mask,
    )


def hsv_red_mask(frame: np.ndarray, method: MaskMethod) -> np.ndarray:
    """BGR frame -> binary mask of the reddest connected blob. See the refusal in the docstring."""
    import cv2

    p = method.params
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo = np.array([p["hue_lo"][0], p["sat_min"], p["val_min"]], dtype=np.uint8)
    hi = np.array([p["hue_lo"][1], 255, 255], dtype=np.uint8)
    lo2 = np.array([p["hue_hi"][0], p["sat_min"], p["val_min"]], dtype=np.uint8)
    hi2 = np.array([p["hue_hi"][1], 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi) | cv2.inRange(hsv, lo2, hi2)
    k = int(p["open_kernel"])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    return mask > 0


# -- the shared PR-08 §4 adapter -------------------------------------------------------------------
#
# Everything below reaches ``scripts/estimators/apple_sam2.py``, the module ``measure_est_drift.py``
# also drives. Nothing here imports it at module scope: see the docstring's lazy-import paragraph.


def _try_import_sam2_adapter() -> tuple[Any | None, str]:
    """``(module, why)``. ``module`` is None when the adapter cannot be reached from here.

    Broad ``except Exception`` on purpose. An adapter that raises ``OSError`` reaching for a
    checkpoint, or ``RuntimeError`` on a CUDA probe, is exactly as unusable as one that is not on
    ``sys.path``, and the difference matters only to the message — which names the type. Letting an
    unexpected type escape here would turn "no segmenter is staged" into a traceback that reads like
    a bug in the measurement.
    """
    import importlib

    try:
        return importlib.import_module(SAM2_ADAPTER_SPEC), "imported"
    except Exception as exc:  # noqa: BLE001 - see docstring
        return None, f"{type(exc).__name__}: {exc}"


def _import_sam2_adapter() -> Any:
    """Import the shared estimator adapter, or fail loudly. Never falls back."""
    module, why = _try_import_sam2_adapter()
    if module is None:
        raise MethodUnavailable(sam2_unavailable_message(why))
    return module


def _adapter_checkpoints(module: Any) -> dict[str, str]:
    """The weight identifiers the adapter DECLARES. Never inferred, never defaulted.

    A tolerance is re-run against the restyled clips at gate time with "the same estimator", and an
    estimator is its weights as much as its code: ``sam2-hiera-large`` and ``sam2-hiera-tiny`` are
    the same package and two different segmenters. So the checkpoints come from the adapter or they
    do not appear, and an adapter that names none cannot be gate-qualified here whatever it claims
    about itself.
    """
    found: dict[str, str] = {}
    declared = getattr(module, "ESTIMATOR_CHECKPOINTS", None)
    if isinstance(declared, dict):
        found.update({str(k): str(v) for k, v in declared.items() if v})
    elif isinstance(declared, (list, tuple)):
        found.update({f"checkpoint_{i}": str(v) for i, v in enumerate(declared) if v})
    for attr in CHECKPOINT_ATTRS:
        value = getattr(module, attr, None)
        if isinstance(value, str) and value:
            found[attr] = value
    return found


def _adapter_weights_status(module: Any) -> tuple[bool | None, str]:
    """Does the adapter say its weights are on this machine? ``None`` means it did not say.

    ``None`` is not "probably fine". ``--method auto`` treats it as a refusal, because the only
    alternative would be for this script to go looking for a checkpoint directory on the adapter's
    behalf, and "a path called sam2-hiera-large exists" is a different claim from "this adapter can
    segment". Guessing wrong there does not crash: an unloaded segmenter returns empty masks, every
    step is dropped as "object not visible", and the artifact reports a coverage floor breach that
    looks like a property of the corpus.
    """
    probe = getattr(module, "available", None)
    if callable(probe):
        try:
            ok = bool(probe())
        except Exception as exc:  # noqa: BLE001 - a probe that raises is not an available segmenter
            return False, f"{SAM2_ADAPTER_SPEC}.available() raised {type(exc).__name__}: {exc}"
        return ok, f"{SAM2_ADAPTER_SPEC}.available() returned {ok}"
    flag = getattr(module, "WEIGHTS_AVAILABLE", None)
    if isinstance(flag, bool):
        return flag, f"{SAM2_ADAPTER_SPEC}.WEIGHTS_AVAILABLE is {flag}"
    return None, (
        f"{SAM2_ADAPTER_SPEC} declares neither available() nor WEIGHTS_AVAILABLE, so it has made "
        "no statement about whether its checkpoints are on this machine"
    )


def _adapter_may_fetch(module: Any) -> bool:
    """Has the adapter been told, by a human, that it may download its checkpoints?

    ``available()`` stays False while the weights are absent even when a fetch is authorised — the
    adapter says so itself — so without this the explicit ``--method sam2 WAM_PR08_ALLOW_DOWNLOAD=1``
    route, which is the whole point of having an explicit route, would be unreachable. Read off the
    adapter; this module never reads the environment variable itself, because the permission belongs
    to the thing that would do the fetching.
    """
    return getattr(module, ADAPTER_DOWNLOAD_ATTR, False) is True


def sam2_weights_absent_message(weights_note: str) -> str:
    """``--method sam2`` when the adapter has ALREADY said its checkpoints are not here.

    Typing the method explicitly says which segmenter to use. It does not overrule an answer the
    adapter has already given: the first ``segment()`` call raises ``EstimatorDependencyMissing``,
    which is an ``ImportError`` and not this module's ``MethodUnavailable``, so before this refusal
    existed it escaped ``main`` as a traceback, exit 1 — outside the documented EXIT STATUS table —
    after the decode loop had already been entered. Refusing here is that same refusal one frame
    earlier, with an exit code the caller can act on and nothing written.
    """
    return "\n".join([
        f"FATAL: --method {SAM2_METHOD_CLI} was requested and {SAM2_ADAPTER_SPEC} says its "
        "checkpoints are NOT on this machine.",
        "       Nothing was written and no frame was decoded.",
        "",
        f"       declaration  {weights_note}",
        f"       interpreter  {sys.executable}",
        f"       module       {SAM2_ADAPTER_SPEC} ({SAM2_ADAPTER_FILE.relative_to(_REPO_ROOT)})",
        "",
        "       Naming the method explicitly chooses WHICH segmenter. It does not overrule the",
        "       adapter's own statement that it cannot run: segment() would raise on the first",
        "       frame, and a run that dies mid-decode has already spent the decode and still has",
        "       no number. This is that refusal, before the first frame.",
        "",
        "       Two ways forward:",
        "",
        "       1. Stage the checkpoints on this machine and re-run. The adapter's own refusal",
        "          names each one and every cache directory it looked in.",
        f"       2. Authorise the fetch on purpose: WAM_PR08_ALLOW_DOWNLOAD=1 sets "
        f"{SAM2_ADAPTER_SPEC}.{ADAPTER_DOWNLOAD_ATTR},",
        "          and this refusal steps aside for it. That is a multi-GB download; the repo rule",
        "          is that nothing is downloaded at scale without asking first. ASK.",
    ])


def sam2_call_failed_message(method_name: str, exc: BaseException) -> str:
    """The adapter raised while segmenting a frame. Fatal here, never fallen back from.

    ``EstimatorDependencyMissing`` subclasses ``ImportError`` and every other thing a segmenter can
    raise mid-run (an OSError reaching for a checkpoint, a CUDA RuntimeError) is likewise not this
    module's ``MethodUnavailable``, so all of them used to leave ``main`` as a traceback and exit 1.
    Translating them here puts the failure inside the EXIT STATUS table and keeps the adapter's own
    message, which is the one that names its checkpoints and its venv, verbatim in the output.
    """
    return "\n".join([
        f"FATAL: {method_name!r} raised while segmenting a frame — {type(exc).__name__}.",
        "       The measurement is abandoned and nothing was written. A partial or resumed run is",
        "       not this corpus's GEOM_TOL, and dropping the frame would move the median.",
        "",
        "       ---- the estimator's own message, verbatim " + "-" * 36,
        "",
        str(exc).rstrip() or f"({type(exc).__name__} carried no message)",
    ])


def _first_line(text: str) -> str:
    """A one-line summary of a multi-line failure, for a one-line field.

    Nothing is lost by it: the adapter's own message is reproduced verbatim below the summary,
    because that message is the one that names its checkpoints, its venv and its download switch,
    and paraphrasing someone else's refusal is how a reader ends up debugging the wrong machine.
    """
    stripped = text.strip()
    return stripped.splitlines()[0] if stripped else text


def sam2_unavailable_message(reason: str) -> str:
    """The loud failure for ``--method sam2``. Names every place looked in and what must change."""
    hits = _local_weight_hits()
    lines = [
        f"FATAL: --method {SAM2_METHOD_CLI} needs the shared PR-08 §4 estimator adapter and cannot "
        "use it here.",
        "       Nothing was written.",
        "",
        f"       reason       {_first_line(reason)}",
        f"       interpreter  {sys.executable}",
        f"       module       {SAM2_ADAPTER_SPEC}",
        f"       file         {SAM2_ADAPTER_FILE} "
        f"({'present' if SAM2_ADAPTER_FILE.is_file() else 'ABSENT'})",
        f"       package init {SAM2_ADAPTER_FILE.parent / '__init__.py'} "
        f"({'present' if (SAM2_ADAPTER_FILE.parent / '__init__.py').is_file() else 'ABSENT'})",
        "",
        "       the checkpoints this pipeline is built on — Cosmos-Transfer2.5 names them itself in",
        "       cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_model.py, which is why it is the",
        "       generator's OWN segmenter and the strongest reading of §4 step 2:",
    ]
    lines += [f"         - {a:<32} {v}" for a, v in COSMOS_SAM2_CHECKPOINT_HINTS]
    lines += ["", "       local segmentation weights found:"]
    lines += [f"         - {w}" for w in hits] or [
        "         (none — nothing matching *sam*/*seg*/*dino*/*owl* under "
        + ", ".join(str(r) for r, _ in WEIGHT_SEARCH_GLOBS) + ")"
    ]
    lines += [
        "",
        "       What would have to change, in order of how little it costs:",
        "",
        f"       1. Write/repair {SAM2_ADAPTER_FILE.relative_to(_REPO_ROOT)} against the estimator",
        "          contract in scripts/measure_est_drift.py (Estimators):",
        "            segment(rgb) -> (H, W) bool mask     estimate_depth(rgb) -> (H, W) float32 m",
        "          plus ESTIMATOR_NAME / ESTIMATOR_VERSION / GATE_QUALIFIED, and a declaration of",
        "          the checkpoints it loads (ESTIMATOR_CHECKPOINTS, or "
        + "/".join(CHECKPOINT_ATTRS) + ").",
        "",
        "       2. Stage the checkpoints. That is a download, and the repo rule is that nothing is",
        "          downloaded at scale without asking first. ASK.",
        "",
        "       This is ONE fix for BOTH halves of PR-08 §8 item 4: the identical module is what",
        f"       scripts/measure_est_drift.py measure --estimators {SAM2_ADAPTER_SPEC} runs, and §4",
        "       step 2 requires GEOM_TOL and EST_DRIFT_P95 to come from the SAME segmenter.",
    ]
    if reason.strip().count("\n"):
        lines += [
            "",
            "       ---- the adapter's own message, verbatim "
            + "-" * 40,
            "",
            reason.rstrip(),
        ]
    return "\n".join(lines)


def sam2_auto_declined(reason: str) -> str:
    """Why ``--method auto`` did not take the adapter. Appended to the standing refusal, never instead."""
    return "\n".join([
        "       The shared PR-08 §4 estimator adapter was checked too, and NOT selected:",
        "",
        f"         module      {SAM2_ADAPTER_SPEC}",
        f"         file        {SAM2_ADAPTER_FILE} "
        f"({'present' if SAM2_ADAPTER_FILE.is_file() else 'ABSENT'})",
        f"         why not     {_first_line(reason)}",
        "",
        "       --method auto selects the adapter only when the ADAPTER declares its weights are",
        "       present, via available() -> bool or WEIGHTS_AVAILABLE: bool. Nothing here probes a",
        "       checkpoint directory on its behalf: a segmenter auto-selected without its weights",
        "       does not crash, it returns empty masks, and the run reports a coverage floor breach",
        "       that reads as a property of the corpus rather than as a missing download.",
        f"       --method {SAM2_METHOD_CLI} types it explicitly, and when the adapter has said its",
        "       checkpoints are absent it refuses with that declaration quoted rather than this",
        "       message — the same answer at a different door, not a way past it. The way past it",
        "       is staging the weights, or authorising the fetch on purpose "
        "(WAM_PR08_ALLOW_DOWNLOAD=1).",
    ])


def sam2_mask_via(module: Any) -> Callable[[np.ndarray, MaskMethod], np.ndarray]:
    """Bind the adapter into a frame -> mask callable. The closure holds the module, not a name."""

    def mask(frame: np.ndarray, method: MaskMethod) -> np.ndarray:
        """One decoded frame -> the adapter's object mask, handed over in the adapter's colour order.

        cv2 decodes BGR and the estimator contract says ``segment(rgb)``. A GroundingDINO prompt of
        "apple" evaluated on channel-swapped pixels neither crashes nor returns nothing: it grounds
        on whatever, in a world where red is blue, most looks like an apple, and GEOM_TOL becomes
        the median displacement of that. The swap happens here, once, rather than being left to
        whichever estimator is wired next.
        """
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise MethodUnavailable(
                f"FATAL: {method.name!r} was handed a frame of shape {arr.shape}; this path decodes "
                "3-channel BGR video and the adapter's contract is segment(rgb). Nothing here "
                "guesses a channel order."
            )
        try:
            raw = module.segment(np.ascontiguousarray(arr[:, :, ::-1]))
        except MethodUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - see sam2_call_failed_message
            raise MethodUnavailable(sam2_call_failed_message(method.name, exc)) from exc
        out = np.asarray(raw)
        if out.shape[:2] != arr.shape[:2]:
            raise MethodUnavailable(
                f"FATAL: {method.name!r} returned a {out.shape[:2]} mask for a {arr.shape[:2]} "
                "frame. A centroid taken on one grid and a displacement reported in another is not "
                "a displacement — the same refusal measure_est_drift.py makes."
            )
        return out

    return mask


def sam2_method(min_area: int) -> MaskMethod:
    """The gate-qualifiable method: the generator's own segmenter, reached through the shared adapter.

    Gate qualification is OPT-IN three times over, and every half is the adapter's to assert. The
    module must say ``GATE_QUALIFIED = True`` — absent means false, exactly as ``measure_est_drift``
    reads it, so a stub cannot become a gate input by being importable. It must NAME ITS WEIGHTS, or
    qualification is withheld here regardless of what it claims: the artifact's only job is to make
    the identical estimator re-runnable on the restyled clips, and "sam2" without a checkpoint is a
    family of segmenters, not one. And it must EXPORT ``SEGMENTER_CONTRACT``, because PR-08 §4 step
    2's "the same segmenter" is uncheckable against a module that never said what operating point it
    ran at — and an uncheckable requirement reads downstream exactly like a satisfied one. That
    contract goes into ``params["segmenter"]``, which is where both the committed-contract guard on
    this side and ``measure_est_drift``'s cross-check look for it.
    """
    module = _import_sam2_adapter()

    # The same contract measure_est_drift.Estimators enforces, enforced identically. estimate_depth
    # is required even though GEOM_TOL never calls it: §4 step 2 wants ONE estimator behind both
    # numbers, and a module that cannot measure the budget cannot be that one — GEOM_TOL would be
    # left with no subtractable partner and nothing downstream would notice until §6.
    for fn in ("segment", "estimate_depth"):
        if not callable(getattr(module, fn, None)):
            raise MethodUnavailable(
                f"FATAL: {SAM2_ADAPTER_SPEC} does not define {fn}(rgb).\n"
                "       The estimator contract is scripts/measure_est_drift.py (Estimators): "
                "segment(rgb) -> (H, W)\n"
                "       bool mask, estimate_depth(rgb) -> (H, W) float32 metres. PR-08 §4 step 2 "
                "requires GEOM_TOL\n"
                "       and EST_DRIFT_P95 to come from the SAME module, so a module that only half "
                "satisfies the\n"
                "       contract cannot produce either."
            )

    checkpoints = _adapter_checkpoints(module)
    declared_gate = bool(getattr(module, "GATE_QUALIFIED", False))
    # The adapter's own account of the operating point it runs at, recorded verbatim into the
    # artifact. This is what makes the committed contract checkable at all: without it the artifact
    # says only ESTIMATOR_NAME, and the same adapter at two box thresholds reports that same name
    # while producing two different tolerances.
    contract = getattr(module, "SEGMENTER_CONTRACT", None)
    contract = dict(contract) if isinstance(contract, Mapping) else None
    withheld_reasons: list[str] = []
    if declared_gate and not checkpoints:
        withheld_reasons.append(
            f"{SAM2_ADAPTER_SPEC} sets GATE_QUALIFIED=True but names no checkpoints (looked for "
            f"ESTIMATOR_CHECKPOINTS and {', '.join(CHECKPOINT_ATTRS)}). A tolerance that cannot say "
            "which weights produced it cannot be re-run with the same estimator at gate time, which "
            "is the only thing GEOM_TOL is for."
        )
    if declared_gate and contract is None:
        withheld_reasons.append(
            f"{SAM2_ADAPTER_SPEC} sets GATE_QUALIFIED=True but exports no SEGMENTER_CONTRACT, so "
            "nothing can compare this run against the contract committed beside GEOM_TOL. PR-08 §4 "
            "step 2's 'the same segmenter' would be uncheckable, and an uncheckable requirement "
            "reads downstream exactly like a satisfied one."
        )
    withheld: str | None = " ".join(withheld_reasons) or None
    available, weights_note = _adapter_weights_status(module)
    may_fetch = _adapter_may_fetch(module)
    # The adapter has already answered, and "I typed the method out in full" is not a rebuttal. The
    # decode loop would raise EstimatorDependencyMissing on frame 0 — an ImportError, not a
    # MethodUnavailable — and leave main as a traceback with no artifact and an undocumented exit 1.
    # Silence (available is None) is NOT refused here: that is auto's rule, and an adapter that
    # simply never wrote a probe has stated nothing to overrule. Anything it raises later is caught
    # at the call site instead.
    if available is False and not may_fetch:
        raise MethodUnavailable(sam2_weights_absent_message(weights_note))

    name = str(getattr(module, "ESTIMATOR_NAME", SAM2_ADAPTER_SPEC))
    version = str(getattr(module, "ESTIMATOR_VERSION", "unversioned"))
    provenance = "; ".join([
        f"{SAM2_ADAPTER_SPEC} ({SAM2_ADAPTER_FILE.relative_to(_REPO_ROOT)})",
        f"the SAME module scripts/measure_est_drift.py measure --estimators {SAM2_ADAPTER_SPEC} "
        "uses, per PR-08 §4 step 2",
        ("checkpoints: " + ", ".join(f"{k}={v}" for k, v in sorted(checkpoints.items())))
        if checkpoints else "checkpoints: NONE DECLARED BY THE ADAPTER",
        weights_note,
    ])

    return MaskMethod(
        name=name,
        version=version,
        gate_qualified=declared_gate and bool(checkpoints) and contract is not None,
        frames_from="video",
        params={
            "cli_method": SAM2_METHOD_CLI,
            # WHERE THE READER LOOKS. measure_est_drift.committed_segmenter_contract() falls back
            # to mask_method.params.segmenter when the document is a measured artifact rather than
            # the pre-measurement contract, and the merge's mask-method refusal compares this
            # block across shards. Recording the adapter's declaration here — not a re-derivation,
            # the dict itself — is what turns "the same segmenter" from a name into a comparison.
            "segmenter": contract,
            "estimator_spec": SAM2_ADAPTER_SPEC,
            "estimator_module_file": str(SAM2_ADAPTER_FILE.relative_to(_REPO_ROOT)),
            "estimator_contract": (
                "segment(rgb) -> (H, W) bool mask; estimate_depth(rgb) -> (H, W) float32 metres "
                "(scripts/measure_est_drift.py Estimators)"
            ),
            "checkpoints": checkpoints or None,
            "checkpoints_declared_by_adapter": bool(checkpoints),
            "adapter_declares_gate_qualified": declared_gate,
            "gate_qualification_withheld_reason": withheld,
            "weights_declaration": weights_note,
            "weights_available": available,
            # False here can only mean a human authorised the fetch (WAM_PR08_ALLOW_DOWNLOAD=1);
            # unauthorised-and-absent is refused before any frame is decoded, so it cannot reach an
            # artifact at all. Recorded because "the weights were downloaded during the measurement"
            # is provenance a reader of the committed number is entitled to.
            "adapter_download_authorised": may_fetch,
            "color_order_in": "RGB — cv2 decodes BGR and the adapter is handed frame[:, :, ::-1]",
            "component": "largest connected component by area",
            "min_area_px": min_area,
            # Stated in the artifact as well as in the code, because the consumer that has to check
            # §4 step 2 reads the JSON and not this file.
            "cross_check_join_key": (
                "mask_method.name == pr08_est_drift.json estimators.name; both are this adapter's "
                "ESTIMATOR_NAME, read off the same module"
            ),
        },
        provenance=provenance,
        mask_fn=sam2_mask_via(module),
        # The module itself, so main() can record what this adapter saw beside what the harness
        # measured — the retry counts and the detection-score distribution the adapter's own second
        # gate-qualification blocker asks for "from a full pass". See EstimatorStatsProbe.
        stats_module=module,
    )


def auto_sam2_method(min_area: int) -> tuple[MaskMethod | None, str]:
    """``--method auto``'s one chance to find a segmenter, and the reason it did not take it.

    Returns ``(None, why)`` rather than raising: ``auto``'s refusal is ``no_segmenter_message()``,
    unchanged, and this text is APPENDED to it. Replacing that message would narrow a refusal that
    names every package and weight directory looked in down to one about a single adapter.
    """
    module, why = _try_import_sam2_adapter()
    if module is None:
        return None, sam2_auto_declined(f"not importable — {_first_line(why)}")
    available, weights_note = _adapter_weights_status(module)
    if available is not True:
        return None, sam2_auto_declined(weights_note)
    try:
        return sam2_method(min_area), ""
    except MethodUnavailable as exc:
        return None, sam2_auto_declined(str(exc).splitlines()[0])


# -- centroids -----------------------------------------------------------------------------------


def centroid_of_mask(mask: np.ndarray, largest_component: bool, min_area: int) -> tuple[float, float] | None:
    """(x, y) centroid in pixels, or None when the object is not visible in this frame.

    None is the whole point of the return type. Occlusion by the hand and the apple leaving frame
    are real events in this corpus; returning (0, 0) or the previous centroid would turn them into
    displacements that were never observed.
    """
    binary = np.asarray(mask)
    if binary.dtype != bool:
        binary = binary > 0
    if not binary.any():
        return None
    if largest_component:
        import cv2

        n, _, stats, centroids = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=8
        )
        if n <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = int(np.argmax(areas)) + 1
        if int(stats[best, cv2.CC_STAT_AREA]) < min_area:
            return None
        cx, cy = centroids[best]
        return float(cx), float(cy)
    if int(binary.sum()) < min_area:
        return None
    ys, xs = np.nonzero(binary)
    return float(xs.mean()), float(ys.mean())


def displacements(centroids: list[tuple[float, float] | None], step: int) -> tuple[np.ndarray, int]:
    """Per-step Euclidean centroid displacement, and the number of steps that could not be measured.

    Steps overlap: every offset i -> i+step is one step. A step is measurable only when BOTH
    endpoints have a centroid.
    """
    if step < 1:
        raise ValueError("step must be >= 1")
    out: list[float] = []
    dropped = 0
    for i in range(len(centroids) - step):
        a, b = centroids[i], centroids[i + step]
        if a is None or b is None:
            dropped += 1
            continue
        out.append(float(np.hypot(b[0] - a[0], b[1] - a[1])))
    return np.asarray(out, dtype=float), dropped


def distribution(values: np.ndarray, bin_px: float) -> dict[str, Any]:
    """The FULL distribution, because PR-08 asks for it and because a median alone hides bimodality.

    A corpus whose steps are mostly ~0 px (the arm is parked) with a tail at 30 px (the transfer)
    has a small median and a gate that the transfer phase cannot pass. That is visible here and
    invisible in a single number.
    """
    if values.size == 0:
        return {"n": 0}
    pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    q = np.percentile(values, pcts)
    top = float(np.ceil(values.max() / bin_px) * bin_px) if values.max() > 0 else bin_px
    edges = np.arange(0.0, top + bin_px, bin_px)
    counts, edges = np.histogram(values, bins=edges)
    return {
        "n": int(values.size),
        "min_px": float(values.min()),
        "max_px": float(values.max()),
        "mean_px": float(values.mean()),
        "std_px": float(values.std(ddof=0)),
        "percentiles_px": {f"p{p}": float(v) for p, v in zip(pcts, q)},
        "histogram": {
            "bin_px": float(bin_px),
            "bin_edges_px": [float(e) for e in edges],
            "counts": [int(c) for c in counts],
        },
    }


# -- what the estimator adapter saw, recorded beside what the harness measured --------------------
#
# WHY THIS EXISTS AND WHAT IT IS NOT. ``scripts/estimators/apple_sam2.py`` counts the frames on
# which it found no box, the frames on which the segmenter came back empty, the frames on which
# upstream's single ``(0.10, 0.10)`` retry fired and the ones it recovered, and it keeps the winning
# detection score for every frame it did find a box on. Until 2026-08-22 NEITHER harness read any of
# it, so the 4-GPU-h, 402-episode, ~171 600-frame GEOM_TOL pass produced none of those numbers and
# threw the evidence away — while the adapter's own second gate-qualification blocker asks for
# exactly "the recorded detection-score distribution and retry counts (n_frames_retry_fired /
# n_frames_retry_recovered) from a full pass, so the retry's contribution is visible rather than
# assumed". This is the place that evidence lands. It is NOT a discharge of that blocker and nothing
# here touches ``GATE_QUALIFIED``: producing evidence and accepting it are two different acts, and
# the second one is a human's.
#
# THREE PROPERTIES IT HAS TO HAVE, none of which is free:
#
# 1. IT DESCRIBES THIS RUN. The adapter's counters are cumulative over the lifetime of the import
#    and nothing resets them, so two measurements driven from one interpreter (a test session, a
#    sweep, a future in-process merge) would each record the other's frames. They are snapshotted
#    before the pass and DIFFERENCED afterwards; the adapter's own counters are never written to
#    from here, because a harness that resets somebody else's module state breaks the next caller.
# 2. THE SCORES SURVIVE SHARDING, EXACTLY. ``--shard`` runs the corpus in 8 pieces and ``--merge``
#    pools them, and a distribution does not decompose any more than a median does. So a shard
#    records the RAW scores per episode, exactly as it records raw per-step displacements, and the
#    merge concatenates them in the corpus's own enumeration order before binning — which is what
#    makes the merged artifact's score distribution identical, float for float, to the one an
#    un-sharded run would have written, rather than approximately equal to it.
# 3. AN ADAPTER WITHOUT ``stats()`` MUST NOT BREAK EITHER HARNESS. The contract both of them call is
#    ``segment(rgb)`` / ``estimate_depth(rgb)``; ``stats()`` is an extra this one happens to offer.
#    A module that does not offer it, or one whose ``stats()`` raises, is recorded as ABSENT WITH A
#    REASON. Never as zeros: "the retry fired 0 times" and "nobody asked" are different claims, and
#    a reader of the artifact must be able to tell them apart.

#: The keys of ``stats()`` that count THIS RUN's frames once differenced. Anything else the adapter
#: reports is descriptive (its name, its pins, its thresholds, its blockers) and is recorded as-is.
#: A key listed here and absent from ``stats()`` records ``null``, not 0.
ADAPTER_RUN_COUNTERS: tuple[str, ...] = (
    "n_segment_calls",
    "n_frames_without_detection",
    "n_frames_with_empty_mask",
    "n_frames_retry_fired",
    "n_frames_retry_recovered",
    # PR-08 V6's mask-validity filter: frames on which the adapter drew a mask and then REFUSED it
    # because it contained essentially none of the object. Differenced like the rest, and listed
    # here rather than left to the descriptive half, because "the filter fired on 12 frames of this
    # run" and "it has fired 12 times since this interpreter started" are different claims and only
    # the first one belongs beside a coverage number.
    "n_frames_mask_refused",
    "n_frames_mask_refused_no_reference",
    "n_mask_validity_iou",
    "n_detection_scores",
)

#: The module attribute holding the per-frame winning detection scores, in call order. Read off the
#: adapter the same way ``ADAPTER_DOWNLOAD_ATTR`` and ``SEGMENTER_CONTRACT`` are — an optional
#: declaration, not part of the estimator contract, absent without consequence beyond being recorded
#: as absent.
ADAPTER_SCORES_ATTR = "DETECTION_SCORES"

#: Bin width of the recorded score histogram. Scores live in [0, 1] and the interesting boundaries
#: are the adapter's own ``box_threshold`` (0.15) and ``retry_box_threshold`` (0.10), so the bins
#: are fine enough to separate them. The exact counts either side of ``box_threshold`` do not depend
#: on this: they are counted from the raw values, against the threshold the adapter reports.
SCORE_HIST_BIN = 0.05

#: One string, written into every ``estimator_stats`` block on every path, because it is true on
#: all of them and a note that differs between the merged artifact and the un-sharded one would be
#: a difference a reader has to account for before believing the two are the same measurement.
ADAPTER_STATS_SCOPE = (
    "THIS RUN ONLY. The adapter's counters are cumulative over the lifetime of the import and "
    "nothing resets them, so they are snapshotted before the first frame and differenced after the "
    "last; a non-zero counters_at_start_of_run means this interpreter had already driven the "
    "adapter and this_run is still this run's. On a MERGED artifact this_run is the sum of the "
    "shards' differences, which is the same set of frames an un-sharded run would have segmented."
)

#: Why the two counter snapshots are null in a merged artifact. Also one string on every path: the
#: merged and un-sharded blocks then differ only in the fields that are genuinely about the process
#: rather than about the corpus.
PROCESS_LOCAL_COUNTERS_NOTE = (
    "counters_at_start_of_run and counters_at_end_of_run are lifetime totals of the PROCESS that "
    "ran the pass, not of this corpus. A merged artifact leaves them null — its shards ran in eight "
    "processes and there is no such total for the merge — and lists each shard's under per_shard; "
    "the shards' own artifacts carry them in full. this_run and detection_scores are about the "
    "frames and are the same numbers either way."
)


def score_distribution(values: np.ndarray, box_threshold: float | None) -> dict[str, Any]:
    """The detection-score distribution, plus the retry's share of it.

    Deliberately NOT :func:`distribution`: that one names every key ``*_px`` because it describes
    pixels, and a confidence score reported in pixels is a unit error waiting to be quoted. The
    shape is otherwise the same, for the same reason — a median alone hides bimodality, and on the
    169-frame local audit the scores were sharply bimodal (p25 0.758 for the masks that were right,
    0.155-0.264 for every flagged one), which is the entire finding.

    ``n_below_box_threshold`` is the number that answers the blocker: the first pass discards
    everything under ``box_threshold``, so a recorded score below it can only have come from the
    retry pass. ``null`` when the adapter did not say what its threshold was — inferring one from
    the data would be this function deciding what the retry did.
    """
    if values.size == 0:
        return {"n": 0, "box_threshold": box_threshold, "n_below_box_threshold": None}
    pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    q = np.percentile(values, pcts)
    # Rounded, so the edges READ as 0.10 and 0.15 — the adapter's retry and primary thresholds —
    # rather than as 0.15000000000000002, which is what stepping a float by 0.05 produces and which
    # would put a score of exactly box_threshold in the bin below it. The counts either side of the
    # threshold are not taken from this histogram in any case; see n_below_box_threshold.
    edges = np.round(np.linspace(0.0, 1.0, int(round(1.0 / SCORE_HIST_BIN)) + 1), 4)
    counts, edges = np.histogram(values, bins=edges)
    below = (int(np.count_nonzero(values < box_threshold))
             if isinstance(box_threshold, (int, float)) and not isinstance(box_threshold, bool)
             else None)
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "percentiles": {f"p{p}": float(v) for p, v in zip(pcts, q)},
        "histogram": {
            "bin": float(SCORE_HIST_BIN),
            "bin_edges": [float(e) for e in edges],
            "counts": [int(c) for c in counts],
        },
        "box_threshold": box_threshold,
        "n_below_box_threshold": below,
        "n_below_box_threshold_meaning": (
            "detections the first pass would have discarded, so they came from the "
            "(retry_box_threshold, retry_text_threshold) retry. This is the retry's contribution to "
            "coverage, measured. Nothing here says those masks are on the right object — that is "
            "the adapter's first gate-qualification blocker and it is not answered by a number."
        ),
    }


def estimator_stats_absent(why: str) -> dict[str, Any]:
    """The record for a run whose estimator reported nothing. Explicit, and never zeros."""
    return {
        "recorded": False,
        "absent_because": why,
        "adapter": None,
        "counters_at_start_of_run": None,
        "counters_at_end_of_run": None,
        "this_run": None,
        "per_shard": None,
        "detection_scores": {"recorded": False, "absent_because": why, "n": None,
                             "distribution": None},
        "note": (
            "Absent is not zero. A run that recorded no adapter statistics says nothing about how "
            "often the detector failed or the retry fired; a run that recorded zeros says those "
            "things did not happen. Nothing downstream may read this block as the second claim."
        ),
    }


class EstimatorStatsProbe:
    """Snapshot an estimator adapter's counters, then report what THIS run did to them.

    Constructed before the first frame and read after the last. Holds the module, never a name: the
    adapter is reached through the same object the harness segments with, so a probe cannot end up
    describing a different import of the same module.
    """

    def __init__(self, module: Any | None, absent_because: str | None) -> None:
        self.module = module
        self.absent_because = absent_because
        self.spec: str | None = (
            None if module is None else (getattr(module, "__name__", None) or repr(module)))
        self.start: dict[str, Any] | None = None
        if module is not None and absent_because is None:
            stats, why = self._read()
            if stats is None:
                self.absent_because = why
            else:
                self.start = {k: stats.get(k) for k in ADAPTER_RUN_COUNTERS}

    @classmethod
    def open(cls, module: Any | None, *, why_absent: str | None = None) -> "EstimatorStatsProbe":
        """A probe on ``module``, or one that records ``why_absent`` and measures nothing."""
        if module is None:
            return cls(None, why_absent or "no estimator module was involved in this run")
        return cls(module, None)

    def _read(self) -> tuple[dict[str, Any] | None, str | None]:
        """``stats()`` as a dict, or None and the reason. Never raises into the harness."""
        fn = getattr(self.module, "stats", None)
        if not callable(fn):
            return None, (
                f"{self.spec!r} exports no callable stats(). The estimator contract is "
                "segment(rgb) / estimate_depth(rgb) and stats() is an optional extra, so this is "
                "not an error — it is the reason nothing was recorded."
            )
        try:
            out = fn()
        except Exception as exc:  # noqa: BLE001 - a broken stats() must not lose a measurement
            return None, (
                f"{self.spec!r} stats() raised {type(exc).__name__}: {exc}. The measurement is "
                "unaffected — stats() is read for the record only and is never on the path that "
                "produces a displacement."
            )
        if not isinstance(out, Mapping):
            return None, f"{self.spec!r} stats() returned {type(out).__name__}, not a mapping."
        return dict(out), None

    def mark(self) -> int | None:
        """How many scores the adapter has recorded so far, or None if it records none.

        The unit of :meth:`since`. Taken per episode as well as per run, so a shard can attribute
        its raw scores to the episode they came from and the merge can rebuild the pool in the
        corpus's own order.
        """
        if self.module is None or self.absent_because is not None:
            return None
        seq = getattr(self.module, ADAPTER_SCORES_ATTR, None)
        if not isinstance(seq, Sequence) or isinstance(seq, (str, bytes)):
            return None
        return len(seq)

    def since(self, mark: int | None) -> list[float] | None:
        """The scores recorded since ``mark``, or None when there are none to attribute.

        Returns None rather than ``[]`` when the list SHRANK: a caller that reset the adapter's
        state mid-run has invalidated the difference, and an empty list would read as "no detection
        scored", which is a claim about the corpus.
        """
        if mark is None:
            return None
        now = self.mark()
        if now is None or now < mark:
            return None
        try:
            return [float(v) for v in getattr(self.module, ADAPTER_SCORES_ATTR)[mark:now]]
        except (TypeError, ValueError):
            # A list holding something that is not a number is not a score distribution, and it is
            # not worth a four-GPU-hour run either: nothing here is on the path that produces a
            # displacement, so it records an absence and the measurement continues.
            return None

    def block(self, scores: list[float] | None, *, include_raw: bool) -> dict[str, Any]:
        """The artifact block: what the adapter is, what this run did, and the scores it recorded.

        ``scores`` is this run's raw list, in call order (or None when the adapter records none).

        ``include_raw`` decides whether the values themselves are written here. ``measure_est_drift``
        sets it: its capture is a few hundred frames, it is never sharded, and keeping the values
        beside the distribution makes the distribution re-derivable. ``measure_geom_tol`` does not —
        its raw values go per episode, into ``per_episode[*].detection_scores`` on the shard path,
        which is where the merge needs them and is exactly where ``displacements_px`` goes.
        """
        if self.module is None or self.absent_because is not None:
            return estimator_stats_absent(
                self.absent_because or "no estimator module was involved in this run")
        end, why = self._read()
        if end is None:
            return estimator_stats_absent(why or "stats() became unreadable during the run")

        start = self.start or {}
        this_run: dict[str, Any] = {}
        went_backwards: list[str] = []
        for key in ADAPTER_RUN_COUNTERS:
            a, b = start.get(key), end.get(key)
            if not isinstance(a, int) or not isinstance(b, int) or isinstance(a, bool):
                this_run[key] = None
                continue
            if b < a:
                went_backwards.append(key)
                this_run[key] = None
                continue
            this_run[key] = b - a
        return {
            "recorded": True,
            "absent_because": None,
            "source": f"{self.spec}.stats()",
            "module": self.spec,
            "scope": ADAPTER_STATS_SCOPE,
            # stats() IN FULL, split by key and not summarised: `adapter` is every key that
            # describes the estimator (its name, its pins, its thresholds, its blockers, its own
            # prose about what each counter means) and `counters_at_end_of_run` is the rest, which
            # is exactly ADAPTER_RUN_COUNTERS. The union of the two IS stats(). They are separated
            # because one half is a property of the ADAPTER and pools across shards by being
            # identical, while the other is a property of the PROCESS and does not pool at all.
            "adapter": {k: v for k, v in end.items() if k not in set(ADAPTER_RUN_COUNTERS)},
            "counters_at_start_of_run": {k: start.get(k) for k in ADAPTER_RUN_COUNTERS},
            "counters_at_end_of_run": {k: end.get(k) for k in ADAPTER_RUN_COUNTERS},
            "this_run": this_run,
            "counters_went_backwards": went_backwards or None,
            "counters_went_backwards_meaning": (
                "a counter that ended below where it started was reset by something during the "
                "run, so its difference is not this run's count and is recorded as null rather "
                "than as a plausible number."
            ) if went_backwards else None,
            "process_local_counters_note": PROCESS_LOCAL_COUNTERS_NOTE,
            "per_shard": None,
            "detection_scores": self._scores_block(scores, end, include_raw=include_raw),
        }

    def _scores_block(self, scores: list[float] | None, end: Mapping,
                      *, include_raw: bool) -> dict[str, Any]:
        if scores is None:
            return {
                "recorded": False,
                "absent_because": (
                    f"{self.spec!r} exports no {ADAPTER_SCORES_ATTR} list of per-frame detection "
                    "scores (or it was reset during the run, which makes the difference "
                    "meaningless). The counts in this_run are unaffected."
                ),
                "n": None,
                "distribution": None,
            }
        thr = end.get("box_threshold")
        block: dict[str, Any] = {
            "recorded": True,
            "absent_because": None,
            "attr": ADAPTER_SCORES_ATTR,
            "n": len(scores),
            "meaning": (
                "the winning box's detection score for every frame of THIS RUN where a box was "
                "found, in call order. Frames with no detection are absent, so n == "
                "this_run.n_segment_calls - this_run.n_frames_without_detection."
            ),
            "distribution": score_distribution(
                np.asarray(scores, dtype=float),
                float(thr) if isinstance(thr, (int, float)) and not isinstance(thr, bool) else None,
            ),
        }
        if include_raw:
            # THE RAW VALUES. A distribution does not decompose and a binned one cannot be
            # re-derived from its bins, so wherever they fit they are kept: that is here for a
            # capture, and per episode for a corpus (see the docstring).
            block["values"] = list(scores)
        return block

# -- corpus discovery ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Episode:
    key: str
    clip: Path | None


def find_episodes(corpus: Path, camera_key: str | None) -> tuple[list[Episode], str]:
    """Enumerate episodes of a LeRobot v2.1 root or of a flat directory of clips.

    Returns the episodes and the layout name, which goes into the artifact — the same corpus reached
    two ways is the same measurement, but only if the artifact says which way.
    """
    if not corpus.exists():
        raise MethodUnavailable(f"FATAL: --corpus {corpus} does not exist.")
    info_path = corpus / "meta" / "info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        key = resolve_camera(info, camera_key, str(corpus))
        clips = sorted((corpus / "videos").rglob(f"{key}/episode_*.mp4"))
        if not clips:
            raise MethodUnavailable(
                f"FATAL: {corpus} declares camera {key!r} but no "
                f"videos/**/{key}/episode_*.mp4 were found. Only meta/ was fetched?"
            )
        return [Episode(c.stem, c) for c in clips], f"lerobot-{info.get('codebase_version', '?')}"
    clips = sorted(corpus.rglob("*.mp4"))
    if not clips:
        raise MethodUnavailable(
            f"FATAL: {corpus} has no meta/info.json and no *.mp4 under it — nothing to measure. "
            "An empty scan is not a pass."
        )
    return [Episode(c.stem, c) for c in clips], "clip-dir"


def read_mask_dir(mask_dir: Path) -> list[np.ndarray]:
    """Per-frame masks, ordered by filename. ``.npy`` needs numpy only; ``.png``/``.jpg`` need cv2."""
    files = sorted(p for p in mask_dir.iterdir()
                   if p.suffix.lower() in (".npy", ".png", ".jpg", ".jpeg"))
    masks: list[np.ndarray] = []
    for f in files:
        if f.suffix.lower() == ".npy":
            masks.append(np.load(f))
        else:
            import cv2

            arr = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if arr is None:
                raise MethodUnavailable(f"FATAL: {f} did not decode as an image.")
            masks.append(arr)
    return masks


def episode_centroids_from_masks(mask_dir: Path, min_area: int) -> tuple[list[tuple[float, float] | None], tuple[int, int]]:
    masks = read_mask_dir(mask_dir)
    if not masks:
        raise MethodUnavailable(f"FATAL: {mask_dir} contains no .npy/.png masks.")
    shapes = {tuple(m.shape[:2]) for m in masks}
    if len(shapes) != 1:
        raise MethodUnavailable(
            f"FATAL: {mask_dir} mixes mask geometries {sorted(shapes)}. Pixels from different "
            "grids are not the same unit and GEOM_TOL would be meaningless."
        )
    h, w = next(iter(shapes))
    cents = [centroid_of_mask(m, largest_component=False, min_area=min_area) for m in masks]
    return cents, (int(w), int(h))


# -- decoders ------------------------------------------------------------------------------------
#
# A CORPUS IS ONLY READABLE BY THE DECODER THAT WILL ACTUALLY READ IT, and until 2026-08-22 this
# module had exactly one and never said so. cv2 was not a choice here, it was an assumption.
#
# Job 189585 -- the first cluster run of this script -- decoded ZERO frames of the PR-08 corpus:
#
#     [av1 @ ...] Missing Sequence Header.
#     FATAL: episode_000000.mp4 opened but decoded no frames
#
# The corpus is AV1 (its manifest says so: codecs ["av1"], materialized "copy") and the cv2 4.11.0
# in the generator's own venv is built against an avcodec with no AV1 decoder. cv2.VideoCapture
# does not raise on that. It opens the file, reports 590 frames, 30 fps and 640x480 off the
# container header, and then fails every read -- which is a corpus that looks measured and is not.
# scripts/verify_clip_decode.py exists because job 186357 lost a GPU hour and 372 captions to the
# identical shape on a sibling corpus, and the same trap was still armed here.
#
# Measured on the cluster (job 189586, runs/pr08-geom-tol/CLIP_DECODE_PROBE.json), in the venv that
# matters: pyav decodes it via libdav1d, imageio and torchvision decode it, decord fails, and there
# is no ffmpeg on PATH to transcode with even if transcoding were the right answer. It is not
# obviously the right answer: Cosmos-Transfer2.5 is handed the PATH and decodes it itself, so a
# re-encoded copy would put a lossy transcode between the tolerance and the pixels the generator
# sees -- at a scale of a fraction of a pixel, which is the unit GEOM_TOL is denominated in.
# Reading the same bytes with a working decoder leaves the evidence alone.
#
# BGR, NOT RGB, AND NOT BY ACCIDENT. cv2 yields BGR and sam2_mask_via() flips it once with
# frame[:, :, ::-1] because the adapter contract is segment(rgb). Every decoder here therefore
# yields BGR too. A decoder that handed back RGB would not crash -- GroundingDINO would ground
# "apple" on channel-swapped pixels in a world where red is blue, and GEOM_TOL would become the
# median displacement of whatever that found.
#
# WHICH DECODER RAN IS PROVENANCE and goes into the artifact beside mask_method, for the same
# reason: two numbers produced by two different readers of the same file are not obviously the same
# quantity, and the artifact is the only place that can say which one produced this one.


@dataclass(frozen=True)
class Decoder:
    """One way of turning a clip into BGR frames, named and versioned for the artifact."""

    name: str
    version: str
    #: clip -> (iterator of BGR uint8 frames, fps). The iterator is lazy on purpose: a 749-frame
    #: episode held whole is 690 MB, and this module only ever needs one frame at a time.
    open_fn: Callable[[Path], tuple[Any, float]]
    note: str = ""


def _module_version(module: str) -> str:
    """Version string, or a marker saying it is not installed. Never raises: an absent decoder is a
    fact the artifact should be able to state, not a reason this module fails to import."""
    try:
        mod = importlib.import_module(module)
    except Exception:  # noqa: BLE001 - absent, or broken on import; both are "not usable"
        return "<not importable>"
    return str(getattr(mod, "__version__", "<no __version__>"))


def _cv2_open(clip: Path) -> tuple[Any, float]:
    import cv2

    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise MethodUnavailable(f"FATAL: cv2 could not open {clip}.")
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    def frames():
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    return
                yield frame
        finally:
            cap.release()

    return frames(), fps


def _pyav_open(clip: Path) -> tuple[Any, float]:
    import av

    container = av.open(str(clip))
    try:
        stream = container.streams.video[0]
    except IndexError as exc:
        container.close()
        raise MethodUnavailable(f"FATAL: {clip} carries no video stream.") from exc
    rate = stream.average_rate or stream.guessed_rate
    fps = float(rate) if rate else 0.0

    def frames():
        try:
            for frame in container.decode(video=0):
                # bgr24 is cv2's order, deliberately -- see the seam header. to_ndarray does the
                # conversion in libswscale rather than by a numpy view, so the array is contiguous
                # and uint8 exactly as VideoCapture would have returned it.
                yield frame.to_ndarray(format="bgr24")
        finally:
            container.close()

    return frames(), fps


DECODERS: dict[str, Decoder] = {
    "cv2": Decoder(
        name="cv2",
        version=_module_version("cv2"),
        open_fn=_cv2_open,
        note="cv2.VideoCapture. Cannot decode AV1 in the Cosmos-Transfer2.5 venv (job 189586).",
    ),
    "pyav": Decoder(
        name="pyav",
        version=_module_version("av"),
        open_fn=_pyav_open,
        note="av.open + libswscale to bgr24. Decodes the PR-08 AV1 corpus via libdav1d.",
    ),
}


def decoder_probe(decoder: Decoder, clip: Path) -> tuple[bool, str]:
    """Can this decoder pull ONE frame out of this clip? Returns (ok, detail).

    One frame is the whole question. The failure this exists for is not a corrupt file -- it is a
    decoder that opens the container, believes the header, and returns nothing, so the distinction
    that matters is between zero frames and one.
    """
    try:
        frames, _ = decoder.open_fn(clip)
        for frame in frames:
            arr = np.asarray(frame)
            return True, f"decoded a {arr.shape} frame"
        return False, "opened the container and decoded no frames"
    except MethodUnavailable as exc:
        return False, _first_line(str(exc))
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def resolve_decoder(name: str, probe_clip: Path) -> Decoder:
    """Pick the decoder, and never pick one that cannot read this corpus.

    ``auto`` is not "whatever imports". It PROBES, in order, and takes the first that actually
    returns a frame from the corpus being measured -- because the failure mode is silent and a
    decoder that imports cleanly is not evidence of anything. The choice and the reason both go
    into the artifact.
    """
    if name != "auto":
        decoder = DECODERS[name]
        ok, detail = decoder_probe(decoder, probe_clip)
        if not ok:
            raise MethodUnavailable(
                f"FATAL: --decoder {name} {detail} on {probe_clip.name}.\n"
                f"       {decoder.note}\n"
                "       This is the failure that reports 590 frames off the container header and "
                "measures none of\n"
                "       them. Try --decoder auto, which probes every decoder against this corpus "
                "before choosing."
            )
        return replace(decoder, note=f"{decoder.note} Probed on {probe_clip.name}: {detail}.")

    tried: list[str] = []
    for candidate in DECODERS.values():
        ok, detail = decoder_probe(candidate, probe_clip)
        tried.append(f"{candidate.name} ({candidate.version}): {detail}")
        if ok:
            return replace(
                candidate,
                note=(f"{candidate.note} Selected by --decoder auto after probing "
                      f"{probe_clip.name}; tried in order: " + "; ".join(tried) + "."),
            )
    raise MethodUnavailable(
        f"FATAL: no decoder known to this script could read {probe_clip}.\n"
        + "".join(f"       {line}\n" for line in tried)
        + "       The container parses and no codec here decodes it. Nothing was measured, which "
        "is not a pass.\n"
        "       scripts/verify_clip_decode.py reports the same thing per clip over a whole corpus."
    )


def episode_centroids_from_video(clip: Path, method: MaskMethod, min_area: int, max_frames: int,
                                decoder: Decoder) -> tuple[list[tuple[float, float] | None], tuple[int, int], float]:
    # No default and no name lookup, and checked before a single frame is decoded. A video method
    # with no segmenter attached is a bug in resolve_method; the one thing that must not happen is
    # for it to be papered over with the red-pixel heuristic while the artifact still carries the
    # other method's name.
    if method.mask_fn is None:
        raise MethodUnavailable(
            f"FATAL: mask method {method.name!r} decodes video but carries no segmenter "
            "(mask_fn is None). resolve_method must attach one; nothing here guesses which."
        )
    frames, fps = decoder.open_fn(clip)
    cents: list[tuple[float, float] | None] = []
    size: tuple[int, int] | None = None
    for frame in frames:
        if 0 < max_frames <= len(cents):
            # The generator holds an open container; abandoning it mid-iteration is what its
            # `finally` is for. Closing here explicitly would be closing it twice.
            break
        h, w = frame.shape[:2]
        if size is None:
            size = (int(w), int(h))
        elif size != (int(w), int(h)):
            raise MethodUnavailable(f"FATAL: {clip} changes frame geometry mid-clip.")
        cents.append(
            centroid_of_mask(method.mask_fn(frame, method), largest_component=True,
                             min_area=min_area)
        )
    if size is None:
        raise MethodUnavailable(
            f"FATAL: {clip} opened but decoded no frames with {decoder.name} "
            f"{decoder.version} — the container parses and the codec does not. "
            "See scripts/verify_clip_decode.py, and --decoder auto, which probes before choosing."
        )
    return cents, size, fps


# -- CLI -----------------------------------------------------------------------------------------


def _est_drift_blocker() -> str:
    """Why ``EST_DRIFT_P95`` is still null, checked against the code rather than asserted.

    PR-08 §4 step 0 needs two Replicator annotators attached before the calibration rig exists at
    all. That file is under active change, so the artifact reports what it found today instead of
    carrying a sentence that quietly goes stale.
    """
    binding = _REPO_ROOT / "src" / "wam" / "robot" / "isaac_binding.py"
    try:
        text = binding.read_text()
    except OSError:
        return f"could not read {binding} to check PR-08 §4 step 0"
    missing = [a for a in ("distance_to_camera", "semantic_segmentation") if a not in text]
    if missing:
        return (f"PR-08 §4 step 0 incomplete: {', '.join(missing)} absent from "
                f"src/wam/robot/isaac_binding.py, so there is no ground-truth segmentation to "
                f"calibrate a monocular estimator against")
    return ("PR-08 §4 step 0 annotators are named in src/wam/robot/isaac_binding.py; steps 1-4 "
            "(render, estimate, compare, p95) have still not been run here")


def missing_cross_check_fields(record: dict[str, Any]) -> list[str]:
    """Fields ``measure_est_drift.cross_check_geom_tol()`` needs that this record does not carry.

    The reader used to pass a missing field by saying nothing (``if theirs_hw is not None and
    ...``); since 2026-08-22 each absent field is its own disqualifying
    ``geom_tol_does_not_record_<field>``. This guard is therefore no longer the only thing standing
    between an incomplete artifact and a clean-looking cross-check — but it is still the half this
    module owns, and it is the better half: it means an artifact written here never trips that
    refusal at all, rather than tripping it hours later on a machine with no corpus mounted.

    ``None`` counts as missing. A null ``resolution_hw`` is not a grid, and a record that reached
    this point without one measured nothing it can name the units of.
    """
    return [k for k in CROSS_CHECK_FIELDS_REQUIRED if record.get(k) is None]


# -- the committed segmenter contract --------------------------------------------------------------
#
# PR-08 §4 step 2 says GEOM_TOL and EST_DRIFT_P95 must come from "the same segmenter", and §6
# subtracts them. The claim is only checkable if the segmenter was written down BEFORE either
# number existed, which is what the contract section of DEFAULT_OUT_REL is. The two functions below
# are the single implementation of "where the block lives" and "on which fields do two of them
# disagree": ``measure_est_drift`` imports both from here rather than keeping its own copies,
# because a reader and a writer that look for one block in two different places is exactly how a
# cross-check comes to pass by looking somewhere empty.


def committed_segmenter_contract(doc: Mapping) -> tuple[dict[str, Any] | None, str | None]:
    """The segmenter block of a GEOM_TOL document, and where in it the block was found.

    Two shapes legitimately live at ``configs/transfer25/pr08_geom_tol.json``, and both are looked
    for, in this order:

    * top-level ``segmenter`` — the CONTRACT, committed before the measurement and copied forward
      verbatim by :func:`merge_committed_contract` into every artifact written over it. Top-level
      means "this came out of the pre-commitment".
    * ``mask_method.params.segmenter`` — what the adapter that produced THIS artifact declared.
      Written on every sam2 artifact at any ``--out``, including a shard, which is what lets the
      merge's mask-method refusal compare segmenters and not only names.

    ``(None, None)`` means the document says nothing about its segmenter, which is a reason for the
    caller to refuse and never a reason to proceed: "the file used to say so before it was
    overwritten" is not a property anything downstream can read.
    """
    block = doc.get("segmenter")
    if isinstance(block, Mapping):
        return dict(block), "segmenter"
    method = doc.get("mask_method")
    if isinstance(method, Mapping):
        params = method.get("params")
        if isinstance(params, Mapping) and isinstance(params.get("segmenter"), Mapping):
            return dict(params["segmenter"]), "mask_method.params.segmenter"
    return None, None


def _canonical(value: Any) -> Any:
    """The value as JSON would round-trip it, so a tuple and a list compare equal.

    The committed side has been through ``json.dumps``/``json.loads`` and the module side has not,
    so ``(480, 640) != [480, 640]`` would otherwise be reported as a disagreement about the pixel
    grid. A cross-check that cries wolf about serialisation gets switched off, and then it is not
    checking the thing it exists for either.
    """
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def contract_disagreements(ours: Mapping, theirs: Mapping) -> list[dict[str, Any]]:
    """Every field on which two segmenter contracts differ, named one by one.

    Field by field rather than a whole-dict inequality, because "the segmenters disagree" is not an
    actionable message: the fix for a different ``box_threshold`` (someone edited a constant) and
    for a different ``segmenter.revision`` (someone measured with other weights) are different, and
    a reader of the artifact should not have to diff two JSON blobs by eye to find out which.

    A field present on one side and absent on the other counts as a disagreement. Absence is not
    agreement anywhere else in this cross-check and it is not here: a contract that has grown a
    field the committed one never had is, precisely, a segmenter the committed one did not describe.
    """
    out: list[dict[str, Any]] = []
    for key in sorted(set(ours) | set(theirs)):
        mine, yours = _canonical(ours.get(key)), _canonical(theirs.get(key))
        if mine != yours:
            out.append({"field": key, "geom_tol": yours, "this_run": mine})
    return out


def _contract_section_keys(existing: Mapping) -> list[str]:
    """Which top-level keys of ``existing`` are the contract section.

    Read off the document's own ``contract_fields`` when it states one, so that a contract which
    grows a key is carried forward without this module being edited — and falls back to
    :data:`CONTRACT_SECTION_FIELDS` for a hand-written or older document that never declared the
    list. The fallback is deliberately not "everything that is not a measurement field": that would
    silently promote a stray key someone left in the file to part of the pre-commitment.
    """
    declared = existing.get("contract_fields")
    if isinstance(declared, list) and declared and all(isinstance(k, str) for k in declared):
        return list(declared)
    return list(CONTRACT_SECTION_FIELDS)


def document_mask_method_name(doc: Mapping) -> str | None:
    """The segmenter name a GEOM_TOL document states, under either of its two spellings.

    ``mask_method.name`` is what a MEASURED artifact carries; ``segmenter.method_name`` is what the
    committed contract carries before any measurement. Both are the estimator module's
    ``ESTIMATOR_NAME`` by construction, and one function answers "which segmenter does this
    document claim" for every caller — the same reason ``committed_segmenter_contract`` exists.
    """
    method = doc.get("mask_method")
    if isinstance(method, Mapping) and isinstance(method.get("name"), str):
        return method["name"]
    contract, _where = committed_segmenter_contract(doc)
    name = (contract or {}).get("method_name")
    return name if isinstance(name, str) else None


def document_pixel_grid(doc: Mapping) -> tuple[list[int] | None, str | None]:
    """``[H, W]`` a GEOM_TOL document states, and where. Measured spellings first, contract last.

    A measured artifact leads with ``resolution_hw``; the pre-measurement contract has only
    ``segmenter.pixel_grid_hw``, committed before a frame was decoded. Same precedence, same
    reason, as ``measure_est_drift.cross_check_geom_tol``.
    """
    for key in ("resolution_hw", "frame_hw"):
        value = doc.get(key)
        if value is not None and len(list(value)) == 2:
            return [int(v) for v in value], key
    contract, where = committed_segmenter_contract(doc)
    grid = (contract or {}).get("pixel_grid_hw")
    if grid is not None and len(list(grid)) == 2:
        return [int(v) for v in grid], f"{where}.pixel_grid_hw"
    return None, None


def refuse_unnamed_est_drift(record: Mapping, out: Path) -> None:
    """An ``est_drift_p95_px`` may not be written without the segmenter that produced it.

    THE HOLE THIS CLOSES. PR-08 §6 subtracts ``EST_DRIFT_P95`` from ``GEOM_TOL`` and §4 step 2
    requires both to come from the SAME segmenter. The two numbers are measured by two scripts into
    two artifacts and merged into this one document, and the merge is the moment the join key can
    be lost: ``scripts/measure_est_drift.py`` records its segmenter as ``estimators.name`` in ITS
    artifact, and a number copied across without that string leaves the committed document stating
    a difference whose two halves nothing can be shown to share. ``run_g0_gates`` then reports "the
    artifact records no estimator name for the EST_DRIFT_P95 half", which costs every G0b run its
    gate qualification — a gate that cannot say yes, which is as blocking as one that says yes
    wrongly.

    So the number and the name are written together or neither is written. ``None`` drift is not a
    violation: the contract is committed with every measurement slot null on purpose.

    A name that DISAGREES with this document's own ``mask_method.name`` is refused too, and that is
    not the same check as the consumer's — this one fires while the file is being written, when the
    fix is free.
    """
    drift = record.get("est_drift_p95_px")
    if drift is None:
        return
    name = record.get(EST_DRIFT_NAME_FIELD)
    ours = document_mask_method_name(record)
    if not isinstance(name, str) or not name.strip():
        raise MethodUnavailable(
            f"FATAL: {out} would state est_drift_p95_px = {drift!r} and no "
            f"{EST_DRIFT_NAME_FIELD}.\n"
            "       PR-08 §4 step 2 requires GEOM_TOL and EST_DRIFT_P95 to be measured with the "
            "SAME segmenter and §6\n"
            "       subtracts them. The two numbers are produced by two scripts into two "
            "artifacts; the name is the\n"
            "       join key, and a merge that drops it leaves a difference whose halves nobody "
            "can pair. run_g0_gates\n"
            "       reports exactly that and refuses to gate on it, permanently.\n"
            f"       Write the segmenter name beside the number — it is `estimators.name` in "
            "configs/transfer25/\n"
            "       pr08_est_drift.json — or use --carry-est-drift, which reads both out of that "
            "artifact and cannot\n"
            "       forget one. Nothing was written."
        )
    if ours is not None and str(name) != str(ours):
        raise MethodUnavailable(
            f"FATAL: {out} would name two segmenters: GEOM_TOL {ours!r} and EST_DRIFT_P95 "
            f"{name!r}.\n"
            "       PR-08 §6 subtracts the two numbers, and two segmenters subtract to a plausible "
            "pixel number that\n"
            "       means nothing — the failure this whole artifact is built against. Nothing was "
            "written."
        )


def merge_committed_contract(out: Path, record: dict[str, Any]) -> dict[str, Any] | None:
    """Carry the committed contract at ``out`` into ``record``, or REFUSE the whole run.

    THE FAILURE THIS PREVENTS. ``out`` defaults to the tracked path that already holds the
    pre-measurement segmenter contract, and ``--merge`` writes there too. Without this function the
    first real GEOM_TOL run replaced that contract with a document that mentioned no segmenter
    anywhere, so the pre-commitment was destroyed by the measurement it was written to constrain,
    and every subsequent ``measure_est_drift`` run refused with
    ``geom_tol_does_not_record_segmenter_params`` — permanently, and correctly, because after the
    overwrite nothing could prove which segmenter GEOM_TOL had been measured with.

    WHAT IT DOES, IN THE ONLY ORDER THAT IS SAFE. Before a byte is written:

    1.  a document already at ``out`` that carries no segmenter block is scratch and is overwritten
        as before — this is only about a file that made the pre-commitment;
    2.  a run that cannot state its OWN segmenter (``--method precomputed``, ``hsv-red-diagnostic``,
        an adapter with no ``SEGMENTER_CONTRACT``) may not land on a document that made one: it is
        refused, because "we used the same segmenter" is unanswerable afterwards;
    3.  any field on which the two contracts disagree refuses the run, naming the fields. This is
        the whole point of committing the method early. A thresholds-tweaked adapter re-run over the
        same corpus produces a different tolerance and would look identical in the artifact;
    4.  on agreement the contract section is copied forward VERBATIM from the document on disk —
        not re-rendered from the adapter — so the bytes that were committed are the bytes that
        survive, and the measurement fills only the measurement slots.

    Returns the contract block that was carried, or ``None`` when ``out`` held no contract. Raises
    :class:`MethodUnavailable`, which every caller turns into exit 2 with nothing written.
    """
    if not out.exists():
        return None
    try:
        existing = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        # An unreadable or non-JSON file at --out is not a contract and cannot be compared against
        # one. Refusing here would block a re-run over a truncated artifact from a killed job,
        # which is a normal thing to do and not a pre-commitment being destroyed.
        return None
    if not isinstance(existing, dict):
        return None
    theirs, where = committed_segmenter_contract(existing)
    if theirs is None:
        return None

    # WHAT RAN, not what some earlier file said. The record's top-level ``segmenter`` can only be a
    # contract carried in from wherever this record was templated (a shard's own --out, say), so
    # comparing it against the target's contract would compare two committed documents to each
    # other and never look at the adapter. ``mask_method.params.segmenter`` is the adapter's own
    # declaration and is what §4 step 2 asks about, so it wins here — the reverse of the reader's
    # precedence, which is looking for the pre-commitment and rightly prefers the top level.
    method = record.get("mask_method")
    params = method.get("params") if isinstance(method, Mapping) else None
    ours: dict[str, Any] | None = None
    ours_where: str | None = None
    if isinstance(params, Mapping) and isinstance(params.get("segmenter"), Mapping):
        ours, ours_where = dict(params["segmenter"]), "mask_method.params.segmenter"
    else:
        ours, ours_where = committed_segmenter_contract(record)
    if ours is None:
        raise MethodUnavailable(
            f"FATAL: {out} carries a committed segmenter contract at {where!r}, and this run "
            "cannot state\n"
            "       which segmenter it used, so it must not overwrite it.\n"
            "       PR-08 §4 step 2 requires GEOM_TOL and EST_DRIFT_P95 to come from the SAME "
            "segmenter and §6\n"
            "       subtracts them. A tolerance written over that contract by a method that "
            "declares no contract of\n"
            "       its own leaves nothing downstream can check the claim against — the file would "
            "still look like a\n"
            "       finished gate artifact.\n"
            f"       Either measure with --method {SAM2_METHOD_CLI} (the adapter the contract "
            "describes), or send this\n"
            "       run's artifact somewhere else with --out."
        )

    disagreements = contract_disagreements(ours, theirs)
    if disagreements:
        lines = [
            f"FATAL: this run's segmenter disagrees with the contract committed at {out} "
            f"({where}):\n"
        ]
        for d in disagreements:
            lines.append(f"         {d['field']}: committed {d['geom_tol']!r}, this run "
                         f"{d['this_run']!r}\n")
        lines.append(
            "       That contract was committed BEFORE the measurement precisely so this "
            "comparison could be\n"
            "       made. A segmenter adjusted after seeing the number it produces is the failure "
            "the committed\n"
            "       style partition exists to prevent, and the adjustment is invisible in the "
            "result: the same\n"
            "       adapter at two thresholds returns two plausible tolerances under one "
            "ESTIMATOR_NAME.\n"
            "       Nothing is written. Either run the segmenter the contract describes, or change "
            "the contract as\n"
            "       a reviewed commit of its own, before the measurement and never after it."
        )
        raise MethodUnavailable("".join(lines))

    for key in _contract_section_keys(existing):
        if key in existing:
            record[key] = existing[key]
    record.setdefault("contract_fields", list(CONTRACT_SECTION_FIELDS))
    record.setdefault("measurement_fields", list(CONTRACT_MEASUREMENT_FIELDS))

    # THE MEASUREMENT SLOTS, one rule each, because they fail in three different directions.
    #
    # geom_tol_px is written to the SAME value as this module's own GEOM_TOL_px — including None —
    # because run_g0_gates._first_present() refuses a document stating one quantity under two
    # spellings that disagree, and a null slot left beside a measured number is that disagreement.
    #
    # The EST_DRIFT_P95 pair is NOT this script's to measure, and it is not this script's to erase
    # either. The record arrives with est_drift_p95_px hardcoded None, so a plain overwrite would
    # silently null a budget somebody had already carried into the committed file — a measurement
    # of GEOM_TOL deleting a measurement of something else. Whatever the committed document holds
    # is kept unless this run actually has a value, which it never does today.
    #
    # gate_margin_px is DERIVED and is therefore re-derived, never carried: a margin computed
    # against the previous tolerance would disagree with this artifact's own arithmetic, which
    # run_g0_gates refuses outright — correctly, and after the corpus has been re-measured.
    # The UNION of what the document declares and what this module knows, not one or the other. A
    # document committed before a slot existed declares the shorter list, and taking its list alone
    # would silently drop the newer slot from the artifact written over it — which for
    # ``est_drift_estimator_name`` means carrying a drift number forward while losing the name that
    # says which segmenter produced it, and that pair is refused three lines below. Extra slots the
    # document declares are honoured too: this module is not the only thing allowed to grow one.
    for key in dict.fromkeys(
        [*(existing.get("measurement_fields") or ()), *CONTRACT_MEASUREMENT_FIELDS]
    ):
        if record.get(key) is None:
            record[key] = existing.get(key)
    record["geom_tol_px"] = record.get("GEOM_TOL_px")
    record["geom_tol_source"] = (
        f"{record.get('measured_by')} {record.get('measured_date')} "
        f"git={record.get('git_commit')} mask_method="
        f"{(record.get('mask_method') or {}).get('name')!r} "
        f"gate_qualified={bool(record.get('gate_qualified'))}"
    )
    refuse_unnamed_est_drift(record, out)
    if record.get("est_drift_p95_px") is not None:
        # The blocker is re-derived on every run and says "steps 1-4 have not been run here". Left
        # beside a budget somebody carried in, it is a document contradicting itself about whether
        # the number exists — and gate_budget() prints that string as the reason when the budget
        # is missing, so a stale one would explain an absence that is not there.
        record["est_drift_p95_blocked_by"] = None
    tol, drift = record.get("geom_tol_px"), record.get("est_drift_p95_px")
    record["gate_margin_px"] = (
        float(tol) - float(drift)
        if isinstance(tol, (int, float)) and isinstance(drift, (int, float)) else None
    )
    record["committed_contract_carried_from"] = {
        "path": str(out),
        "found_at": where,
        "compared_against": ours_where,
        "fields": _contract_section_keys(existing),
        "note": (
            "The contract section of the document previously at this path, copied forward "
            "verbatim after every field of its segmenter block was compared against the adapter "
            "this run drove. The run would have been refused, with nothing written, on any "
            "disagreement — see measure_geom_tol.merge_committed_contract()."
        ),
    }
    return dict(theirs)


def refuse_default_out_without_contract(out: Path) -> None:
    """The tracked GEOM_TOL path may not be written unless the pre-commitment is sitting in it.

    ``merge_committed_contract`` protects a contract that is THERE. This protects the case where it
    is not: a deleted, renamed or never-created contract would let a measurement write the tracked
    path with nothing to have been checked against, and the resulting artifact is indistinguishable
    from one that was checked. PR-08 §4 step 2's "the same segmenter" is a claim about what was
    written down first, so a first measurement with nothing written down first is not a measurement
    this gate can use — and ``git checkout`` is a cheaper fix than a re-run of the corpus.

    Only ``DEFAULT_OUT`` is guarded. Any other ``--out`` is scratch or a diagnostic and is free.
    """
    if out != DEFAULT_OUT:
        return
    if out.exists():
        try:
            doc = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            doc = None
        if isinstance(doc, dict) and committed_segmenter_contract(doc)[0] is not None:
            return
    raise MethodUnavailable(
        f"FATAL: {DEFAULT_OUT_REL} carries no committed segmenter contract, and this run would "
        "write it.\n"
        "       That file is the PR-08 §4 step 2 pre-commitment: the detector, the segmenter, the "
        "depth model,\n"
        "       their pinned revisions, the prompt, both threshold pairs, the box rule and the "
        "pixel grid, written\n"
        "       down BEFORE the number so that 'GEOM_TOL and EST_DRIFT_P95 came from the same "
        "segmenter' is a\n"
        "       checkable claim rather than a recollection. Measuring onto that path without it "
        "produces an\n"
        "       artifact that looks exactly like one that was checked.\n"
        "       Restore it (`git checkout -- " + DEFAULT_OUT_REL + "`) — or, if this really is the "
        "first one, commit\n"
        "       the contract with the four measured fields null, and then measure. Use --out for "
        "anything else."
    )


def sidecar_path(out: Path) -> Path:
    """``<artifact>.sha256`` — the same sidecar name ``check_style_partition.py --write-hash`` uses."""
    return out.parent / (out.name + ".sha256")


def write_artifact(out: Path, record: dict[str, Any]) -> tuple[Path, str]:
    """Write the artifact and its ``.sha256`` sidecar. Returns the sidecar path and the digest.

    The sidecar is not decoration. GEOM_TOL is a pre-commitment: it is measured once, committed, and
    then every later gate quotes it. A digest committed alongside is what makes "the file the gate
    read is the file that was committed" checkable by anyone with ``sha256sum``, rather than a
    matter of trusting that nobody edited a JSON file between the measurement and the run.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, indent=2) + "\n").encode("utf-8")
    out.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    side = sidecar_path(out)
    side.write_text(digest + "\n", encoding="utf-8")
    return side, digest


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# -- sharding, and the merge that puts the median back together -----------------------------------
#
# The module docstring argues WHY this shape and not an --episode-range, why blake2b and not hash(),
# and why nothing is summarised before the merge. What follows is the mechanism.


def shard_of(episode_key: str, num_shards: int) -> int:
    """Which shard owns this episode. Deterministic across processes, machines and Python builds.

    ``hash(episode_key) % num_shards`` is the obvious spelling and it is a trap: ``PYTHONHASHSEED``
    is randomised per interpreter, so every task of the same Slurm array would compute a DIFFERENT
    partition of the same corpus. The failure is not a crash — it is a set of shard artifacts that
    together cover some episodes twice and others never, each of them internally consistent. A
    keyed-by-content digest has no such freedom.

    ``digest_size=8`` is 64 bits taken big-endian: far more entropy than the ~402 keys need, and
    fixed here so the partition is reproducible from the recorded rule alone.
    """
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    digest = hashlib.blake2b(episode_key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % num_shards


def select_shard(episodes: list[Episode], shard: int, num_shards: int) -> list[tuple[int, Episode]]:
    """This shard's episodes, each paired with its index in the FULL enumeration.

    The index is not decoration and it is not a serial number within the shard. It is the episode's
    position in ``find_episodes()``'s sorted enumeration of the whole corpus, and it is what lets
    the merge rebuild the pooled displacement array in exactly the order an un-sharded run would
    have built it — which is what makes the merged mean, std and histogram identical rather than
    merely close. The median would survive any order; those three would not.
    """
    return [(i, ep) for i, ep in enumerate(episodes) if shard_of(ep.key, num_shards) == shard]


def _shard_block(shard: int, num_shards: int, selected: list[tuple[int, Episode]],
                 all_keys: list[str]) -> dict[str, Any]:
    """The provenance a shard carries so the merge can check it rather than trust it."""
    return {
        "index": shard,
        "num_shards": num_shards,
        "assignment": SHARD_ASSIGNMENT,
        "assignment_note": (
            "Assignment is a digest of the episode KEY, not a slice of the episode LIST. Adding or "
            "removing a clip therefore moves that clip only; a range would renumber every episode "
            "after it and silently re-partition a corpus whose shards are computed by different "
            "jobs at different times. The merge re-derives this rule and refuses a shard holding "
            "an episode that does not hash to it."
        ),
        # WHICH EPISODES, not how many. A count cannot prove coverage: eight shards reporting 50
        # episodes each sum to 400 whether they covered 400 distinct episodes or 380 with 20
        # counted twice. The merge takes the union of these and compares it to the enumeration.
        "episode_keys": [ep.key for _, ep in selected],
        "episode_indices": [i for i, _ in selected],
        "n_episodes_in_shard": len(selected),
        # The whole corpus as this shard saw it, digested. Every shard enumerates independently, so
        # agreement across N of these is evidence that they all measured the same corpus — and the
        # full list beside it is what lets the merge NAME the episodes nobody covered instead of
        # only counting them.
        "corpus_episode_keys_sha256": corpus_keys_digest(all_keys),
    }


def corpus_keys_digest(keys: list[str]) -> str:
    """A stable digest of one corpus enumeration. Newline-joined so no key can absorb another."""
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


#: Fields that exist only on a shard artifact and must not survive into the committed one. Dropped
#: explicitly rather than by an allow-list, so a field added to shards later fails loudly in the
#: test that compares merged against un-sharded rather than leaking into the gate artifact.
SHARD_ONLY_FIELDS: tuple[str, ...] = (
    "shard", "is_shard", "shard_median_px", "geom_tol_px_is_null_because", "corpus_episode_keys",
    "gate_qualified_scope",
)


def _read_shard_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except OSError as exc:
        raise MethodUnavailable(f"FATAL: --merge could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MethodUnavailable(
            f"FATAL: --merge could not parse {path} as JSON: {exc}\n"
            "       A truncated shard artifact is what a job killed at the wall leaves behind. "
            "Re-run that shard;\n"
            "       merging around it would drop its episodes and the merge would then be a median "
            "over part of the\n"
            "       corpus wearing the name of the whole."
        ) from exc


def collect_shard_records(paths: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    """Read the shard artifacts named on the command line. Directories are expanded, files are not.

    A directory is scanned for ``*.json`` and anything that is not a shard artifact is SKIPPED with
    a line on stderr — the merge job's own output directory holds the pilot artifact and, after the
    first successful merge, the merged one, and a scan that refused on those would be unusable. A
    path named EXPLICITLY is never skipped: the operator said that file, and quietly ignoring it is
    how a merge comes to be missing a shard that was right there on the command line.
    """
    found: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        if path.is_dir():
            for candidate in sorted(path.glob("*.json")):
                rec = _read_shard_json(candidate)
                if rec.get("schema") != SHARD_SCHEMA:
                    print(f"--merge: skipping {candidate} (schema "
                          f"{rec.get('schema')!r}, not {SHARD_SCHEMA!r})", file=sys.stderr)
                    continue
                found.append((candidate, rec))
            continue
        if not path.exists():
            raise MethodUnavailable(
                f"FATAL: --merge {path} does not exist.\n"
                "       Named explicitly, so it is not skipped: a shard the operator asked for and "
                "that is not there\n"
                "       is a missing shard, not a filter."
            )
        rec = _read_shard_json(path)
        if rec.get("schema") != SHARD_SCHEMA:
            raise MethodUnavailable(
                f"FATAL: --merge {path} carries schema {rec.get('schema')!r}, not "
                f"{SHARD_SCHEMA!r}.\n"
                f"       A {SCHEMA!r} artifact is a finished GEOM_TOL, not an input to a merge, and "
                "merging one in\n"
                "       would pool a corpus with itself. Nothing here guesses which you meant."
            )
        found.append((path, rec))
    if not found:
        raise MethodUnavailable(
            "FATAL: --merge found no shard artifacts at all in "
            + ", ".join(str(p) for p in paths) + ".\n"
            f"       A merge over zero shards is not an empty result, it is a missing input. Shard "
            f"artifacts carry\n"
            f"       schema {SHARD_SCHEMA!r} and are written by --shard I --num-shards N."
        )
    return found


def _by_shard_lines(items: list[tuple[int, Any]]) -> str:
    return "".join(f"         shard {i}: {v}\n" for i, v in sorted(items, key=lambda t: t[0]))


def _refuse_on_disagreement(label: str, values: list[tuple[int, Any]], why: str) -> None:
    """One field, one refusal, one reason why the disagreement matters. Never a warning."""
    distinct = {json.dumps(v, sort_keys=True) for _, v in values}
    if len(distinct) <= 1:
        return
    raise MethodUnavailable(
        f"FATAL: the shards disagree on {label}, so they did not measure one quantity:\n"
        + _by_shard_lines(values)
        + f"       {why}\n"
        "       Nothing is written. Re-run the disagreeing shard(s) under the conditions the "
        "others ran under;\n"
        "       there is no reconciliation here, because there is no correct way to pool two "
        "different measurements."
    )


def merge_estimator_stats(loaded: list[tuple[Path, dict[str, Any]]],
                          entries: list[tuple[int, dict[str, Any], list[float]]]) -> dict[str, Any]:
    """Pool the shards' ``estimator_stats`` into one block, exactly as the displacements are pooled.

    THE COUNTS SUM. ``this_run`` counts frames, and a shard's frames are disjoint from every other
    shard's — the merge has already refused, above, unless the shards partition the corpus exactly
    once — so the sums are the counts an un-sharded run over the same corpus would have reported.
    The lifetime totals do NOT sum: eight processes' totals are eight overlapping statements about
    eight interpreters. They are nulled here and listed per shard instead.

    THE SCORES CONCATENATE IN THE CORPUS'S ORDER, not in shard order — out of
    ``per_episode[*].detection_scores`` and through the same ``episode_index`` sort the
    displacements use. That is what makes the merged distribution identical to the un-sharded one
    rather than approximately equal to it: ``mean`` and ``std`` are floating-point sums, and a sum
    is only bit-identical if the order is.

    NOTHING HERE REFUSES. Every other refusal in this file guards a number the gate quotes; this
    block is recorded evidence and feeds nothing — not ``gate_qualified``, not ``GEOM_TOL_px``, not
    ``coverage``. A shard written by a version that did not record it, or a set of shards that
    disagree about whether it was recorded, produces an ABSENCE WITH A REASON. Never zeros, and
    never a merge that dies on the way to a tolerance somebody spent four GPU-hours on.
    """
    by_index = sorted(((int(rec["shard"]["index"]), path, rec) for path, rec in loaded),
                      key=lambda t: t[0])
    blocks: list[tuple[int, Path, dict[str, Any]]] = []
    for idx, path, rec in by_index:
        block = rec.get("estimator_stats")
        if not isinstance(block, dict):
            return estimator_stats_absent(
                f"shard {idx} ({path}) carries no estimator_stats block, so there is nothing to "
                "pool. A shard written before this field existed looks exactly like this; re-run "
                "that shard with this version of the script if the evidence is wanted. The "
                "tolerance is unaffected — nothing in GEOM_TOL is derived from these counts."
            )
        blocks.append((idx, path, block))

    if any(b.get("recorded") is not True for _, _, b in blocks):
        first = blocks[0][2]
        if all(b == first for _, _, b in blocks):
            # Every shard reported the SAME absence for the same reason — a precomputed-mask run,
            # say, where no estimator was involved at all. Carrying it forward verbatim is what
            # makes the merged artifact equal to the un-sharded one on that path.
            return dict(first)
        return estimator_stats_absent(
            "the shards disagree about the estimator statistics: "
            + "; ".join(
                f"shard {idx} recorded none ({b.get('absent_because')})"
                if b.get("recorded") is not True else f"shard {idx} recorded them"
                for idx, _, b in blocks)
            + ". Pooling a shard that measured the adapter with one that did not would report the "
            "counts of part of the corpus as if they were the whole of it."
        )

    template = dict(blocks[0][2])
    went_backwards = sorted({k for _, _, b in blocks
                             for k in (b.get("counters_went_backwards") or ())})
    this_run: dict[str, Any] = {}
    for key in ADAPTER_RUN_COUNTERS:
        parts = [(b.get("this_run") or {}).get(key) for _, _, b in blocks]
        this_run[key] = (
            sum(int(v) for v in parts)
            if parts and all(isinstance(v, int) and not isinstance(v, bool) for v in parts)
            else None
        )

    # The raw scores, attributed to episodes by the shards and re-sorted into the corpus's own
    # enumeration order by the caller. A shard that recorded a distribution but no per-episode
    # values cannot be pooled exactly, and an approximate pool is not offered.
    missing_values = [ep.get("episode") for _, ep, _ in entries
                      if not isinstance(ep.get("detection_scores"), list)]
    pooled_scores: list[float] | None = None
    scores_absent: str | None = None
    if not all((b.get("detection_scores") or {}).get("recorded") is True for _, _, b in blocks):
        scores_absent = "; ".join(
            f"shard {idx}: {(b.get('detection_scores') or {}).get('absent_because')}"
            for idx, _, b in blocks
            if (b.get("detection_scores") or {}).get("recorded") is not True
        )
    elif missing_values:
        scores_absent = (
            f"{len(missing_values)} measured episode(s) carry no raw detection_scores "
            f"({', '.join(str(k) for k in missing_values[:6])}"
            + ("..." if len(missing_values) > 6 else "") + "). A distribution does not decompose, "
            "so the merge pools the RAW scores or records nothing: a shard that reports only its "
            "own binned distribution can be averaged, not merged, and averaging distributions is "
            "the same wrong answer as averaging medians."
        )
    else:
        pooled_scores = [float(v) for _, ep, _ in entries for v in ep["detection_scores"]]

    if pooled_scores is None:
        scores_block: dict[str, Any] = {
            "recorded": False,
            "absent_because": scores_absent or "no shard recorded per-frame detection scores",
            "n": None,
            "distribution": None,
        }
    else:
        first_scores = template.get("detection_scores") or {}
        thr = (first_scores.get("distribution") or {}).get("box_threshold")
        scores_block = {
            "recorded": True,
            "absent_because": None,
            "attr": first_scores.get("attr", ADAPTER_SCORES_ATTR),
            "n": len(pooled_scores),
            "meaning": first_scores.get("meaning"),
            "distribution": score_distribution(
                np.asarray(pooled_scores, dtype=float),
                float(thr) if isinstance(thr, (int, float)) and not isinstance(thr, bool) else None,
            ),
        }

    template.update({
        "counters_at_start_of_run": {k: None for k in ADAPTER_RUN_COUNTERS},
        "counters_at_end_of_run": {k: None for k in ADAPTER_RUN_COUNTERS},
        "this_run": this_run,
        "counters_went_backwards": went_backwards or None,
        # Carried with its own list rather than templated from shard 0, which may not be the shard
        # it happened in: a named defect with a null explanation beside it is a worse record than
        # either half alone.
        "counters_went_backwards_meaning": (
            "at least one shard's counter ended below where it started, so its difference was not "
            "that shard's count and was recorded as null; this_run therefore counts fewer frames "
            "than the corpus was segmented over. The shard artifacts say which."
        ) if went_backwards else None,
        "per_shard": [
            {
                "index": idx,
                "path": str(path),
                "this_run": b.get("this_run"),
                "counters_at_start_of_run": b.get("counters_at_start_of_run"),
                "counters_at_end_of_run": b.get("counters_at_end_of_run"),
                "n_detection_scores": (b.get("detection_scores") or {}).get("n"),
            }
            for idx, path, b in blocks
        ],
        "detection_scores": scores_block,
    })
    return template


def merge_shard_records(loaded: list[tuple[Path, dict[str, Any]]],
                        out: Path) -> tuple[dict[str, Any], np.ndarray]:
    """Pool the shards into the committed GEOM_TOL artifact, or refuse.

    Every refusal below is its own message naming its own failure, and every one of them is fatal
    with nothing written. The reason they are not warnings is the same reason GEOM_TOL exists at
    all: it is committed once, quoted by every later gate, and re-derived by nobody. A merge that
    proceeds past a missing shard produces a median over part of the corpus that is indistinguishable
    from the real thing — right units, plausible magnitude, wrong number, forever.
    """
    records = [rec for _, rec in loaded]
    paths = [p for p, _ in loaded]

    # -- shape: every shard must carry the block the rest of this function reads ------------------
    for path, rec in loaded:
        block = rec.get("shard")
        if not isinstance(block, dict) or "index" not in block or "num_shards" not in block:
            raise MethodUnavailable(
                f"FATAL: {path} declares schema {SHARD_SCHEMA!r} but carries no usable 'shard' "
                "block.\n"
                "       The merge reads index, num_shards, episode_keys and episode_indices out of "
                "it; without\n"
                "       them there is nothing to check coverage against and the artifact is not a "
                "shard, whatever\n"
                "       its schema says."
            )

    # -- REFUSAL 1a: the shards disagree about how many shards there are --------------------------
    counts = {int(rec["shard"]["num_shards"]) for rec in records}
    if len(counts) != 1:
        raise MethodUnavailable(
            "FATAL: the shard artifacts disagree on num_shards: "
            + ", ".join(str(c) for c in sorted(counts)) + ".\n"
            + "".join(f"         {p}: shard {rec['shard']['index']} of "
                      f"{rec['shard']['num_shards']}\n" for p, rec in loaded)
            + "       These are pieces of two DIFFERENT partitions of the corpus. Pooling them "
            "would double-count\n"
            "       the episodes the two partitions happen to share and drop the rest. Re-run one "
            "partition whole."
        )
    num_shards = counts.pop()

    # -- REFUSAL 1b: two artifacts claim the same shard index -------------------------------------
    seen: dict[int, Path] = {}
    for path, rec in loaded:
        idx = int(rec["shard"]["index"])
        if idx in seen:
            raise MethodUnavailable(
                f"FATAL: two shard artifacts both claim shard index {idx} of {num_shards}:\n"
                f"         {seen[idx]}\n"
                f"         {path}\n"
                "       One of them is stale — a re-run that wrote to a new path, or a directory "
                "scan that picked\n"
                "       up an old copy. Merging both pools that shard's episodes twice, which "
                "shifts the median\n"
                "       toward whatever those episodes did. Name the shard artifacts explicitly, "
                "or clear the stale one."
            )
        seen[idx] = path

    # -- REFUSAL 1c: a shard is missing ------------------------------------------------------------
    missing = sorted(set(range(num_shards)) - set(seen))
    if missing:
        raise MethodUnavailable(
            "FATAL: shard(s) " + ", ".join(str(i) for i in missing) + f" of {num_shards} are "
            "missing.\n"
            + "".join(f"         have shard {i}: {seen[i]}\n" for i in sorted(seen))
            + "       GEOM_TOL is the median over the WHOLE source corpus. A median over the "
            "shards that happen\n"
            "       to have landed has the right units and a plausible magnitude and is a "
            "different number, and\n"
            "       nothing downstream re-derives it. Re-submit the missing array task(s) and "
            "merge again."
        )

    # -- REFUSAL 2: the shards did not enumerate the same corpus ----------------------------------
    _refuse_on_disagreement(
        "the corpus they enumerated (corpus_episode_keys_sha256)",
        [(int(rec["shard"]["index"]), rec["shard"].get("corpus_episode_keys_sha256"))
         for rec in records],
        "Each shard enumerates the corpus itself and digests the result. Different digests mean "
        "the corpus changed between shards, or they were pointed at different corpora — either "
        "way the partition they belong to no longer exists and its coverage cannot be proved.",
    )

    # -- REFUSAL 3: the mask method ---------------------------------------------------------------
    _refuse_on_disagreement(
        "the mask method that produced the centroids",
        [(int(rec["shard"]["index"]), rec.get("mask_method")) for rec in records],
        "PR-08 §4 step 2 requires ONE segmenter behind GEOM_TOL, because §6 subtracts "
        "EST_DRIFT_P95 from it and that subtraction is arithmetic only between two numbers from "
        "the same estimator. Two segmenters pooled into one median is not a tolerance, it is a "
        "mixture.",
    )

    # -- REFUSAL 3b: the corpus path and the camera ------------------------------------------------
    # Both are carried into the COMMITTED artifact from shard 0's template, so a disagreement means
    # the artifact names one corpus (one camera) while half its displacements came from another.
    # ``corpus_episode_keys_sha256`` does not catch either: it digests episode KEYS, and two roots
    # holding the same episode names agree on it while holding different pixels.
    _refuse_on_disagreement(
        "the corpus they measured",
        [(int(rec["shard"]["index"]), rec.get("corpus")) for rec in records],
        "GEOM_TOL is a property of ONE source corpus and the merged artifact names one path. Two "
        "roots with the same episode names digest identically, so nothing else here would notice.",
    )
    _refuse_on_disagreement(
        "camera_key, which selects the pixels",
        [(int(rec["shard"]["index"]), rec.get("camera_key")) for rec in records],
        "The camera decides which view the centroid moved in. Two cameras pooled into one median "
        "is a mixture in the same way two segmenters are, and it belongs in this list beside "
        "resolution_hw and step_frames for the same reason.",
    )

    # -- REFUSAL 4: the decoder --------------------------------------------------------------------
    # NAME AND VERSION ONLY, deliberately. `decoder.note` records which clip --decoder auto probed,
    # and each shard probes its OWN first clip, so comparing the whole block would refuse every
    # correct merge that ever ran. What has to agree is which reader read the pixels.
    _refuse_on_disagreement(
        "the decoder that read the pixels",
        [(int(rec["shard"]["index"]),
          None if rec.get("decoder") is None
          else [rec["decoder"].get("name"), rec["decoder"].get("version")]) for rec in records],
        "Two readers of the same bytes are not obviously the same quantity — the corpus is AV1 and "
        "this project has already lost a run to a decoder that reported frames off a container "
        "header and decoded none of them (job 189585). Which one read the pixels is provenance, "
        "and it has to be one answer.",
    )

    # -- REFUSAL 5: the pixel grid -----------------------------------------------------------------
    _refuse_on_disagreement(
        "the pixel grid (resolution_hw)",
        [(int(rec["shard"]["index"]), rec.get("resolution_hw")) for rec in records],
        "GEOM_TOL is denominated in pixels and §4 subtracts EST_DRIFT_P95, also in pixels, from "
        "it. Displacements measured on two grids are two units, and a median over the pool is a "
        "number in neither.",
    )

    # -- REFUSAL 6: the step ------------------------------------------------------------------------
    # Not asked for by the task list and not optional either: PR-08 §6 never defines "per-step",
    # GEOM_TOL scales ~linearly with the reading, and two shards run at different --step-frames pool
    # into a median that is wrong by a factor nobody can recover from the artifact.
    _refuse_on_disagreement(
        "step_frames, the step PR-08 §6 leaves undefined",
        [(int(rec["shard"]["index"]), rec.get("step_frames")) for rec in records],
        "GEOM_TOL scales roughly linearly with what a step is taken to be. Shards run at two steps "
        "pool into a median that is a mixture of two tolerances and is proportionally wrong "
        "against either.",
    )

    # The floor the shards were judged against, and the bins their distributions were counted into.
    # Neither is a free parameter of the merge: taking them from --min-coverage / --hist-bin-px here
    # would let a merge re-judge and re-bin shards silently, which is the same class of failure as
    # averaging their medians. They come from the shards, and the shards must agree.
    _refuse_on_disagreement(
        "min_coverage, the floor they were judged against",
        [(int(rec["shard"]["index"]), rec.get("min_coverage")) for rec in records],
        "The coverage floor is the threshold this measurement is judged against. Shards judged "
        "against two floors cannot be pooled into one verdict.",
    )
    _refuse_on_disagreement(
        "the histogram bin width",
        [(int(rec["shard"]["index"]),
          (rec.get("distribution") or {}).get("histogram", {}).get("bin_px")) for rec in records],
        "The recorded distribution is binned, PR-08 §6 asks for the distribution and not only the "
        "median, and two bin widths do not pool into one histogram.",
    )

    # -- REFUSAL 7: one shard says it is fit to merge and another says it is not --------------------
    by_index: dict[int, dict[str, Any]] = {int(rec["shard"]["index"]): rec for rec in records}
    quals = {i: bool(rec.get("gate_qualified")) for i, rec in by_index.items()}
    if len(set(quals.values())) != 1:
        yes = [i for i, q in sorted(quals.items()) if q]
        no = [i for i, q in sorted(quals.items()) if not q]
        raise MethodUnavailable(
            "FATAL: the shards disagree on gate_qualified — "
            + f"{', '.join(str(i) for i in yes)} say true, "
            + f"{', '.join(str(i) for i in no)} say false.\n"
            + "".join(
                "         shard {}: {}\n".format(
                    i, "; ".join(by_index[i].get("gate_disqualified_reasons")
                                or ["(none recorded)"]))
                for i in no)
            + "       A shard's gate_qualified means 'this shard is fit to be merged': a "
            "gate-qualified mask method,\n"
            "       no --limit and no --max-frames, coverage over the floor. Pooling a fit shard "
            "with an unfit one\n"
            "       produces an artifact that is neither — its gate flag would be a claim about "
            "some of its own\n"
            "       inputs. Re-run the disqualified shard(s), or, if the disqualification is real, "
            "merge nothing:\n"
            "       the corpus, not the merge, is what has to change."
        )

    # -- REFUSAL 8: an episode is in a shard it does not hash to -----------------------------------
    # The merge does not take a shard's word for which episodes belong to it. Re-deriving the rule
    # catches an artifact written by an older assignment, a hand-edited file, and the PYTHONHASHSEED
    # class of bug that the rule exists to make impossible.
    for path, rec in loaded:
        idx = int(rec["shard"]["index"])
        wrong = [k for k in rec["shard"].get("episode_keys", [])
                 if shard_of(k, num_shards) != idx]
        if wrong:
            raise MethodUnavailable(
                f"FATAL: {path} claims shard {idx} of {num_shards} but holds "
                f"{len(wrong)} episode(s) that do not hash to it: "
                + ", ".join(wrong[:8]) + ("..." if len(wrong) > 8 else "") + "\n"
                f"       The rule is {SHARD_ASSIGNMENT}, and the merge re-derives it rather than "
                "trusting the\n"
                "       artifact. A shard whose membership does not follow it was produced by a "
                "different partition\n"
                "       rule, and the other shards' coverage cannot be reasoned about alongside it."
            )

    # -- REFUSAL 9a: a shard did not account for every episode it was assigned ---------------------
    #
    # THE PROOF USED TO BE WEAKER THAN THE CLAIM IT WROTE DOWN. Coverage was taken over each shard's
    # ASSIGNED ``episode_keys`` and compared to the corpus, while the pooled displacements come from
    # ``per_episode``. An episode assigned to a shard and never measured by it is in the first list
    # and absent from the second, so the union came out complete, ``refusals_checked`` recorded
    # "the union of covered episodes is not the corpus, exactly once each" as a check that had been
    # performed, and the merged artifact stated corpus coverage it had not established. Reachable on
    # the ``--method precomputed`` path (an episode whose mask directory is missing is skipped) and
    # exactly the shape a hand-edited or older-version shard has.
    #
    # This is parity with the un-sharded run, not a new standard: there, every selected episode
    # either lands in ``per_episode`` or is named in ``episodes_skipped_no_masks``, by construction
    # of the loop. The merge now checks that same identity per shard. The skip LIST is truncated to
    # 50 names in the artifact, so the identity is checked on the COUNT — which is not truncated —
    # and the names are used to say which episodes when they are all there.
    for path, rec in loaded:
        block = rec["shard"]
        assigned = list(block.get("episode_keys", []))
        measured = [ep.get("episode") for ep in rec.get("per_episode", [])]
        n_skipped = int(rec.get("n_episodes_skipped_no_masks") or 0)
        # Measured keys must be a SUBSET of the assigned ones, or the count identity below can be
        # satisfied by a shard that measured somebody else's episode and skipped one of its own —
        # which is a double-count in the pooled median and a hole in the corpus at the same time.
        stray = [k for k in measured if k not in set(assigned)]
        if stray:
            raise MethodUnavailable(
                f"FATAL: {path} claims shard {block.get('index')} of {num_shards} but reports "
                f"per_episode entries for {len(stray)} episode(s) it was not assigned: "
                + ", ".join(str(k) for k in stray[:8]) + ("..." if len(stray) > 8 else "") + "\n"
                "       Those displacements are in this shard's pool and in whichever shard the "
                "keys hash to, so the\n"
                "       merged median weights them twice while the coverage arithmetic still adds "
                "up. Re-run that shard."
            )
        if len(measured) + n_skipped == len(assigned):
            continue
        skipped_named = list(rec.get("episodes_skipped_no_masks") or [])
        unaccounted = [k for k in assigned
                       if k not in set(measured) and k not in set(skipped_named)]
        raise MethodUnavailable(
            f"FATAL: {path} was assigned {len(assigned)} episode(s) and accounts for "
            f"{len(measured)} measured + {n_skipped} skipped = {len(measured) + n_skipped}.\n"
            + ("       UNACCOUNTED FOR (" + str(len(unaccounted)) + "): "
               + ", ".join(unaccounted[:12]) + ("..." if len(unaccounted) > 12 else "") + "\n"
               if unaccounted else "")
            + "       The merge proves corpus coverage from what the shards MEASURED, not from "
            "what they were\n"
            "       assigned — a shard that silently measured nothing would otherwise satisfy the "
            "coverage check\n"
            "       while contributing no displacements, and the committed artifact would state a "
            "coverage it never\n"
            "       had. An un-sharded run puts every episode in exactly one of the two lists, so "
            "a shard that does\n"
            "       not was written by a different version of this script or edited afterwards. "
            "Re-run that shard."
        )

    # -- REFUSAL 9b: the union of covered episodes is not the corpus -------------------------------
    expected: list[str] = list(records[0].get("corpus_episode_keys") or [])
    if not expected:
        raise MethodUnavailable(
            f"FATAL: {paths[0]} does not record corpus_episode_keys, so the merge cannot prove it "
            "saw every\n"
            "       episode — only that the shard indices 0..N-1 are present, which is a statement "
            "about files\n"
            "       and not about the corpus. A merge that cannot prove coverage is not a merge."
        )
    covered: dict[str, list[int]] = {}
    for rec in records:
        for key in rec["shard"].get("episode_keys", []):
            covered.setdefault(key, []).append(int(rec["shard"]["index"]))
    uncovered = [k for k in expected if k not in covered]
    doubled = {k: v for k, v in covered.items() if len(v) > 1}
    unexpected = [k for k in covered if k not in set(expected)]
    if uncovered or doubled or unexpected:
        lines = [
            "FATAL: the shards do not cover the corpus exactly once, so this is not a merge of the "
            f"corpus.\n"
            f"       corpus enumerates {len(expected)} episode(s); the shards cover "
            f"{len(covered)} distinct.\n"
        ]
        if uncovered:
            lines.append("       NEVER MEASURED (" + str(len(uncovered)) + "): "
                         + ", ".join(uncovered[:12]) + ("..." if len(uncovered) > 12 else "") + "\n")
        if doubled:
            lines.append("       MEASURED TWICE (" + str(len(doubled)) + "): "
                         + ", ".join(f"{k} in shards {v}" for k, v in list(doubled.items())[:6])
                         + ("..." if len(doubled) > 6 else "") + "\n")
        if unexpected:
            lines.append("       NOT IN THE CORPUS (" + str(len(unexpected)) + "): "
                         + ", ".join(unexpected[:12]) + ("..." if len(unexpected) > 12 else "")
                         + "\n")
        lines.append(
            "       An episode measured twice is weighted twice in the median; an episode never "
            "measured is a\n"
            "       hole in a number that PR-08 §6 defines over the whole source corpus. Neither is "
            "detectable\n"
            "       downstream. Fix the partition and merge again."
        )
        raise MethodUnavailable("".join(lines))

    # -- pooling. Nothing above this line has looked at a displacement. ----------------------------
    #
    # Rebuilt in the corpus's own enumeration order (episode_index), which is the order an
    # un-sharded run concatenates in. The median would not care; mean_px, std_px and the histogram
    # counts are floating-point sums and DO care, and "the merged artifact equals the un-sharded
    # one exactly" is a far easier property to test than "equals it in the fields we thought of".
    entries: list[tuple[int, dict[str, Any], list[float]]] = []
    for path, rec in loaded:
        for ep in rec.get("per_episode", []):
            if "episode_index" not in ep or "displacements_px" not in ep:
                raise MethodUnavailable(
                    f"FATAL: {path} has a per_episode entry for {ep.get('episode')!r} with no "
                    "episode_index or no\n"
                    "       displacements_px. The merge pools the RAW per-step displacements — a "
                    "median does not\n"
                    "       decompose, so a shard that reports only its own median cannot be "
                    "merged, only averaged,\n"
                    "       and averaging medians is the wrong number. Re-run that shard with this "
                    "version of the script."
                )
            entries.append((int(ep["episode_index"]), ep, list(ep["displacements_px"])))
    entries.sort(key=lambda t: t[0])

    pooled = [np.asarray(d, dtype=float) for _, _, d in entries]
    values = np.concatenate(pooled) if pooled else np.asarray([], dtype=float)
    n_dropped = sum(int(ep["n_steps_dropped"]) for _, ep, _ in entries)
    n_frames = sum(int(ep["n_frames"]) for _, ep, _ in entries)
    n_steps_total = int(values.size + n_dropped)
    coverage = float(values.size / n_steps_total) if n_steps_total else 0.0
    geom_tol = float(np.median(values)) if values.size else None

    # The raw per-episode arrays are the shard's contribution to the pool and not part of the
    # committed artifact: both are dropped here, and both for the same reason — the pooled
    # statistic above is what the merge exists to produce, and the values it was taken over are
    # still in the shard artifacts named under merged_from.shards.
    per_episode = [{k: v for k, v in ep.items()
                    if k not in ("displacements_px", "detection_scores")} for _, ep, _ in entries]
    ep_medians = [e["median_px"] for e in per_episode if e["median_px"] is not None]

    # Shard 0's record is the template for every field that is a property of the RUN rather than of
    # the data — corpus, layout, units, grid, decoder, mask_method, step, the notes and the consumer
    # asserts. Every one of those has just been proved identical across the shards by a refusal
    # above, so "shard 0's" and "the shards'" are the same thing; taking it from a record rather
    # than rebuilding it means the merged artifact cannot be missing a field the un-sharded run
    # writes. Everything that is a property of the DATA is recomputed below, from the pool.
    template = dict(by_index[0])
    min_coverage = float(template["min_coverage"])
    bin_px = float((template.get("distribution") or {}).get("histogram", {}).get(
        "bin_px", DEFAULT_HIST_BIN_PX))

    coverage_ok = coverage >= min_coverage
    headline_valid = bool(values.size and coverage_ok)
    method_gate_ok = bool((template.get("mask_method") or {}).get("gate_qualified"))
    partial = any(bool(rec.get("partial_measurement")) for rec in records)

    reasons: list[str] = []
    if not values.size:
        reasons.append("no step yielded a displacement")
    if not coverage_ok:
        reasons.append(f"pooled coverage {coverage:.3f} < min_coverage {min_coverage}")
    if not method_gate_ok:
        reasons.append(
            f"mask method {(template.get('mask_method') or {}).get('name')!r} is not gate-qualified")
    for rec in sorted(records, key=lambda r: int(r["shard"]["index"])):
        for r in rec.get("gate_disqualified_reasons") or []:
            reasons.append(f"shard {rec['shard']['index']}: {r}")
    gate_ok = bool(headline_valid and method_gate_ok and not partial and not reasons)

    merged = dict(template)
    for key in SHARD_ONLY_FIELDS:
        merged.pop(key, None)
    # And any contract section shard 0 happened to carry in from ITS --out. The merged artifact's
    # contract comes from the document at the merge's own --out, compared against the adapter the
    # shards ran; a block templated in from a shard would be a second, unchecked provenance for the
    # one field whose entire purpose is provenance.
    for key in CONTRACT_SECTION_FIELDS:
        merged.pop(key, None)
    merged.update({
        "schema": SCHEMA,
        "measured_by": "scripts/measure_geom_tol.py --merge",
        "measured_date": date.today().isoformat(),
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),

        "artifact_path": str(out),
        "artifact_sha256_sidecar": str(sidecar_path(out)),
        "artifact_is_tracked_default": out == DEFAULT_OUT,

        "merged_from": {
            "num_shards": num_shards,
            "assignment": SHARD_ASSIGNMENT,
            "shards": [
                {
                    "index": int(rec["shard"]["index"]),
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "n_episodes": int(rec["shard"]["n_episodes_in_shard"]),
                    "n_steps_measured": int(rec["n_steps_measured"]),
                    "shard_median_px": rec.get("shard_median_px"),
                    "decoder_note": (rec.get("decoder") or {}).get("note"),
                }
                for path, rec in sorted(loaded, key=lambda t: int(t[1]["shard"]["index"]))
            ],
            "pooling": (
                "The per-step displacements of every episode, pooled and medianed once. A median "
                "does NOT decompose across shards: the median of the shard medians is a different "
                "statistic, and on a park-then-transfer corpus it can differ substantially while "
                "looking entirely reasonable. Shards therefore emit raw float64 displacements — "
                "exact through JSON, whose float repr is the shortest round-tripping string since "
                "Python 3.1 — and the pool is rebuilt in the corpus's own enumeration order, so "
                "this artifact is identical to what a single un-sharded run would have written."
            ),
            "refusals_checked": [
                "a shard is missing, duplicated, or disagrees on num_shards",
                "the shards did not enumerate the same corpus",
                "the shards disagree on the corpus path or on camera_key",
                "the shards disagree on the mask method (including its segmenter contract)",
                "the shards disagree on the decoder (name and version)",
                "the shards disagree on the pixel grid (resolution_hw)",
                "the shards disagree on step_frames, min_coverage or the histogram bin",
                "one shard is gate_qualified and another is not",
                "a shard holds an episode that does not hash to it",
                "a shard reports a per_episode entry for an episode it was not assigned",
                "a shard does not account for every episode it was assigned, as measured "
                "(per_episode) + skipped (n_episodes_skipped_no_masks)",
                "the union of assigned episodes is not the corpus, exactly once each",
            ],
            # THE COVERAGE PROOF, IN THE NUMBERS IT WAS ACTUALLY MADE ON. It used to be made on the
            # ASSIGNED episode keys while the artifact claimed coverage of the MEASURED ones — a
            # shard that silently measured nothing passed. Both counts are recorded now so a reader
            # can see which claim was proved rather than take the sentence above for it.
            "coverage_proof": {
                "corpus_episodes": len(expected),
                "assigned_episodes": len(covered),
                "measured_episodes": len(per_episode),
                "skipped_no_masks": sum(int(rec.get("n_episodes_skipped_no_masks") or 0)
                                        for rec in records),
                "how": (
                    "Per shard: every measured episode is one the shard was assigned, and "
                    "measured + skipped == assigned. Across shards: the assigned sets partition "
                    "the corpus exactly once. Together those give measured + skipped == the "
                    "corpus, which is the claim GEOM_TOL's definition over the whole source corpus "
                    "needs. skipped_no_masks > 0 means the median is over less than the corpus and "
                    "is the same caveat an un-sharded run carries in the same field."
                ),
            },
        },

        "n_episodes": len(per_episode),
        "n_episodes_found": int(template["n_episodes_found"]),
        "n_episodes_skipped_no_masks": sum(int(rec.get("n_episodes_skipped_no_masks") or 0)
                                           for rec in records),
        "episodes_skipped_no_masks": [
            k for rec in sorted(records, key=lambda r: int(r["shard"]["index"]))
            for k in (rec.get("episodes_skipped_no_masks") or [])
        ][:50],
        "n_frames": n_frames,

        "GEOM_TOL_px": geom_tol,
        "geom_tol_px_median_of_episode_medians": (
            float(np.median(ep_medians)) if ep_medians else None
        ),
        "headline_valid": headline_valid,
        "gate_qualified": gate_ok,
        "gate_disqualified_reasons": reasons,
        "partial_measurement": partial,
        "limit": max(int(rec.get("limit") or 0) for rec in records),
        "max_frames": max(int(rec.get("max_frames") or 0) for rec in records),
        "n_steps_total": n_steps_total,
        "n_steps_measured": int(values.size),
        "n_steps_dropped_object_not_visible": n_dropped,
        "coverage": coverage,
        "min_coverage": min_coverage,

        "distribution": distribution(values, bin_px),
        "per_episode": per_episode,
        "displacements_npy": None,

        # POOLED, not templated from shard 0. The counts sum because the shards partition the
        # corpus exactly once (proved by the refusals above) and the scores concatenate in the
        # corpus's own enumeration order, so this block is the block an un-sharded run would have
        # written — apart from the two lifetime totals, which belong to eight processes and are
        # recorded per shard instead. Additive and read-only: nothing here feeds gate_qualified.
        "estimator_stats": merge_estimator_stats(loaded, entries),

        # Re-derived HERE rather than carried from shard 0: it is read off
        # src/wam/robot/isaac_binding.py, the file it describes is under active change, and a
        # sentence copied out of an artifact written hours earlier is exactly how it goes stale.
        "est_drift_p95_blocked_by": _est_drift_blocker(),
    })
    return merged, values


def _check_mode_flags(args: argparse.Namespace) -> None:
    """Refuse the flag combinations that would silently measure the wrong thing.

    argparse could express some of this, but ``parser.error`` exits 2 with a usage block and no
    argument about WHY — and every one of these has a why that is worth more than the usage block.
    """
    merging = args.merge is not None
    sharding = args.shard is not None or args.num_shards is not None
    carrying = args.carry_est_drift is not None

    if carrying and (merging or sharding):
        raise MethodUnavailable(
            "FATAL: --carry-est-drift names a different job from --merge/--shard.\n"
            "       It MEASURES NOTHING: it copies an already-measured EST_DRIFT_P95 and the "
            "segmenter that produced\n"
            "       it into the committed document. Run it on its own, after the measurement it "
            "carries."
        )
    if carrying:
        return

    if merging and sharding:
        raise MethodUnavailable(
            "FATAL: --merge and --shard/--num-shards name two different jobs on one command line.\n"
            "       --shard MEASURES one piece of the corpus; --merge POOLS the pieces into the "
            "committed number.\n"
            "       Nothing here picks one and drops the other."
        )
    if merging:
        return

    if args.corpus is None:
        raise MethodUnavailable(
            "FATAL: --corpus is required. (It is optional only under --merge, which reads shard "
            "artifacts and\n"
            "       not the corpus — the shards each recorded the enumeration they saw, and the "
            "merge checks them\n"
            "       against each other.)"
        )
    if sharding and (args.shard is None or args.num_shards is None):
        raise MethodUnavailable(
            "FATAL: --shard and --num-shards go together. One without the other is a partition "
            "with an unknown\n"
            "       denominator, and an artifact from it could not be merged: the merge needs to "
            "know how many\n"
            "       shards to insist on before it can say one is missing."
        )
    if not sharding:
        return
    if args.num_shards < 1:
        raise MethodUnavailable(
            f"FATAL: --num-shards {args.num_shards} is not a positive integer."
        )
    if not 0 <= args.shard < args.num_shards:
        raise MethodUnavailable(
            f"FATAL: --shard {args.shard} is out of range for --num-shards {args.num_shards}: "
            f"the shards are 0..{args.num_shards - 1}.\n"
            "       A shard index nobody will merge produces an artifact that looks finished and "
            "is unreachable;\n"
            "       an out-of-range one measures the empty set and reports it as a clean run."
        )
    if args.out == DEFAULT_OUT:
        raise MethodUnavailable(
            f"FATAL: --shard refuses to write the tracked default {DEFAULT_OUT_REL}.\n"
            f"       {args.num_shards} array tasks writing one path is a race whose winner is "
            "whichever task finished\n"
            "       last, and what it leaves behind is one shard of the corpus sitting at the "
            "path the gate reads\n"
            "       as GEOM_TOL. Give each shard its own --out (the sbatch uses "
            "shard-<index>.json), and let\n"
            "       --merge write the committed artifact."
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # NOT required=True any more, and checked in _check_mode_flags() instead: --merge reads shard
    # artifacts and never touches the corpus, so the merge job can run on the free CPU QoS with no
    # data mounted. Every other invocation still refuses without it, with a reason attached.
    ap.add_argument("--corpus", type=Path, default=None,
                    help="the SOURCE corpus: a LeRobot v2.1 root (meta/info.json + videos/) or a "
                         "directory of .mp4 clips. GEOM_TOL is a property of the source, never of "
                         "the generated clips. Required unless --merge")
    ap.add_argument("--camera-key", default=None,
                    help="video feature to measure when the root declares more than one; never "
                         "guessed")
    ap.add_argument("--method",
                    choices=("auto", "precomputed", SAM2_METHOD_CLI, "hsv-red-diagnostic"),
                    default="auto",
                    help=f"'auto' (default) uses --masks if given, else the shared "
                         f"{SAM2_ADAPTER_SPEC} adapter IF it declares its weights are present, and "
                         "otherwise FAILS naming the missing segmenter. "
                         f"'{SAM2_METHOD_CLI}' drives that adapter explicitly — the same module "
                         "scripts/measure_est_drift.py measures EST_DRIFT_P95 with, which is what "
                         "PR-08 §4 step 2's 'the same segmenter' means; it still refuses, before "
                         "decoding anything, if the adapter says its checkpoints are absent and no "
                         "download was authorised. Naming a method AND --masks names two "
                         "segmenters and is refused. 'hsv-red-diagnostic' is not a segmenter and "
                         "exits 3")
    ap.add_argument("--masks", type=Path, default=None,
                    help="directory of per-frame object masks, one subdirectory per clip stem, "
                         "plus a masks.meta.json naming and versioning the segmenter that made them")
    ap.add_argument("--step-frames", type=int, default=1,
                    help="frames per 'step' (default: %(default)s). PR-08 §6 does not define the "
                         "step; the choice is recorded in the artifact and GEOM_TOL scales with it")
    ap.add_argument("--limit", type=int, default=0,
                    help="measure at most N episodes (0 = all). A sample is a smoke test, not the "
                         "committed number: any non-zero value forces gate_qualified=false and "
                         "exit 3")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="decode at most N frames per clip (0 = all). Partial in the same way as "
                         "--limit, and disqualifies the run from the gate for the same reason")
    ap.add_argument("--decoder", choices=("auto", *DECODERS), default="auto",
                    help="which decoder turns clips into frames (default: %(default)s). 'auto' "
                         "PROBES each one against this corpus and takes the first that actually "
                         "returns a frame — a decoder that imports is not evidence that it can "
                         "read AV1, and the failure is silent (job 189585 read 590 frames off a "
                         "container header and decoded none of them). The choice and the probe "
                         "results are recorded in the artifact")
    ap.add_argument("--min-area-px", type=int, default=40,
                    help="masks smaller than this count as 'object not visible' (default: "
                         "%(default)s)")
    ap.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                    help="fraction of steps that must yield a displacement before the median is "
                         "called a measurement (default: %(default)s)")
    ap.add_argument("--hist-bin-px", type=float, default=DEFAULT_HIST_BIN_PX,
                    help="histogram bin width in pixels (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"JSON artifact path (default: the TRACKED {DEFAULT_OUT_REL}, absolute and "
                         "anchored to the repo root, plus a .sha256 sidecar). PR-08 §6 requires "
                         "GEOM_TOL to be recorded regardless of verdict and §8 item 4 requires it "
                         "COMMITTED, which a path under gitignored runs/ can never be")
    ap.add_argument("--dump-displacements", type=Path, default=None,
                    help="also write every displacement as a .npy, for a distribution nobody has "
                         "to trust a histogram for. Scratch/diagnostic — runs/ is the place for it")
    ap.add_argument("--shard", type=int, default=None, metavar="I",
                    help="measure only the episodes that hash to shard I of --num-shards. The "
                         "pilot measured the full run at 4.005 GPU-h against a 4 h MaxWall, so the "
                         "committed number is produced by several jobs and merged. Assignment is a "
                         "blake2b digest of the EPISODE KEY, never a slice of the episode list, so "
                         "adding or removing a clip moves that clip only. Writes a "
                         f"{SHARD_SCHEMA} artifact with GEOM_TOL_px null — a shard is not a "
                         "tolerance — and refuses to write the tracked default --out")
    ap.add_argument("--num-shards", type=int, default=None, metavar="N",
                    help="how many shards the corpus is partitioned into. Goes together with "
                         "--shard: the merge needs the denominator before it can say a shard is "
                         "missing")
    ap.add_argument("--merge", type=Path, nargs="+", default=None, metavar="SHARD_JSON",
                    help="pool shard artifacts into the committed GEOM_TOL at --out. Paths may be "
                         "files or directories (a directory is scanned for "
                         f"{SHARD_SCHEMA} artifacts; anything else in it is skipped with a note, "
                         "while a file named explicitly is never skipped). The median is taken "
                         "ONCE over the pooled per-step displacements — shard medians are never "
                         "averaged — and the merge refuses, loudly and separately, on a missing or "
                         "duplicated shard, a disagreement about the mask method / decoder / pixel "
                         "grid / step / floor / bins, a gate_qualified split, an episode in the "
                         "wrong shard, or a union that is not the corpus exactly once. --corpus is "
                         "not needed and --min-coverage / --hist-bin-px are taken from the shards")
    ap.add_argument("--carry-est-drift", type=Path, default=None, metavar="EST_DRIFT_JSON",
                    help="carry EST_DRIFT_P95 out of scripts/measure_est_drift.py's artifact into "
                         "the committed document at --out, together with the segmenter name that "
                         f"produced it ({EST_DRIFT_NAME_FIELD}) — the join key PR-08 §4 step 2 "
                         "requires and the one a person merging two files by hand can forget. "
                         "Refuses a disqualified or null measurement, a different segmenter name, "
                         "a segmenter whose parameters disagree field for field, and a different "
                         "pixel grid. Measures nothing and needs no corpus")
    return ap.parse_args(argv)


def resolve_method(args: argparse.Namespace) -> MaskMethod:
    """Pick the mask method, or refuse. The order below is the whole of the selection policy.

    ``--masks`` still wins under ``auto``: masks on disk were produced deliberately and carry their
    own named, versioned provenance, and silently preferring a locally importable adapter over the
    thing the operator pointed at would change which estimator produced GEOM_TOL without saying so.
    """
    # Two segmenters named on one command line, and only one of them can produce the number. The
    # old order of tests picked the typed method and dropped --masks along with its masks.meta.json
    # provenance, without a word in the artifact — which is the exact failure the rest of this
    # function argues against, committed by this function.
    if args.masks is not None and args.method in (SAM2_METHOD_CLI, "hsv-red-diagnostic"):
        raise MethodUnavailable(
            f"FATAL: --method {args.method} and --masks {args.masks} name two different segmenters "
            "and only one\n"
            "       can have produced GEOM_TOL. Nothing here picks for you: the artifact's only "
            "job is to say\n"
            "       which estimator measured this number so the identical one can be re-run on the "
            "restyled\n"
            "       clips at gate time, and a silently discarded --masks makes that record a "
            "guess.\n"
            f"       Drop --masks to use {args.method}, or drop --method to measure the masks "
            "(--method precomputed)."
        )
    if args.method == "hsv-red-diagnostic":
        return hsv_red_method(args.min_area_px)
    if args.method == SAM2_METHOD_CLI:
        return sam2_method(args.min_area_px)
    if args.method == "precomputed" or (args.method == "auto" and args.masks is not None):
        if args.masks is None:
            raise MethodUnavailable("FATAL: --method precomputed needs --masks.")
        return load_precomputed_method(args.masks)
    method, declined = auto_sam2_method(args.min_area_px)
    if method is not None:
        return method
    # The standing refusal, unchanged and first: it names every segmenter package and every weight
    # directory this machine was checked for. The adapter paragraph is appended to it, never
    # substituted for it.
    raise MethodUnavailable(no_segmenter_message() + "\n\n" + declined)


def merge_main(args: argparse.Namespace) -> int:
    """``--merge``: pool the shard artifacts into the committed GEOM_TOL, or refuse and write nothing."""
    try:
        loaded = collect_shard_records(list(args.merge))
        record, values = merge_shard_records(loaded, args.out)
    except MethodUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_FATAL

    # The same belt the measuring path wears, for the same reason: the consumer disqualifies its
    # own run over each of these by name. A merged artifact is MORE exposed to it, not less — it is
    # assembled from a template rather than built field by field.
    absent = missing_cross_check_fields(record)
    if absent:
        print(
            "FATAL: the merged record is missing " + ", ".join(absent) + ", which "
            "scripts/measure_est_drift.py\n"
            "       cross_check_geom_tol() reads. The shard artifacts do not carry those fields, "
            "so they were\n"
            "       written by a different version of this script. Nothing was written.",
            file=sys.stderr,
        )
        return EXIT_FATAL

    # The merge's --out is the committed path too — that is the whole point of the merge — so it
    # wears the identical contract guard. A merged artifact reaches it through shard 0's template,
    # which carries mask_method.params.segmenter from the adapter every shard ran; the merge has
    # already refused if two shards disagreed about mask_method, so what is compared here is one
    # segmenter, not a mixture.
    try:
        refuse_default_out_without_contract(args.out)
        carried = merge_committed_contract(args.out, record)
    except MethodUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_FATAL
    if carried is not None:
        print(f"contract    carried forward from {args.out} — the shards' segmenter agrees with it "
              "field for field", file=sys.stderr)

    if args.dump_displacements:
        args.dump_displacements.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.dump_displacements, values)
        record["displacements_npy"] = str(args.dump_displacements)

    side, digest = write_artifact(args.out, record)

    merged = record["merged_from"]
    print(f"merged      {merged['num_shards']} shard(s), "
          f"{record['n_episodes']} of {record['n_episodes_found']} episodes, "
          f"{record['n_frames']} frames, "
          f"{record['n_steps_measured']}/{record['n_steps_total']} steps "
          f"(coverage {record['coverage']:.3f})", file=sys.stderr)
    for s in merged["shards"]:
        print(f"  shard {s['index']:>3}  {s['n_episodes']:>4} ep  "
              f"{s['n_steps_measured']:>7} steps  median "
              f"{s['shard_median_px']}  {s['path']}", file=sys.stderr)
    # The shard medians are printed and are NOT the answer. Their spread beside the pooled median is
    # the cheapest possible reminder that the two are different statistics, in the one place an
    # operator is guaranteed to look.
    print(f"GEOM_TOL    {record['GEOM_TOL_px'] if record['GEOM_TOL_px'] is None else round(record['GEOM_TOL_px'], 4)} px"
          f" — ONE median over the pooled displacements, not an average of the shard medians above",
          file=sys.stderr)
    print(f"wrote       {args.out}", file=sys.stderr)
    print(f"sha256      {digest}  ({side})", file=sys.stderr)

    if not record["gate_qualified"]:
        print("\nNOT GATE-QUALIFIED: the merge completed and the number MUST NOT be committed as "
              "GEOM_TOL:", file=sys.stderr)
        for r in record["gate_disqualified_reasons"]:
            print(f"                    - {r}", file=sys.stderr)
    return EXIT_OK if record["gate_qualified"] else EXIT_NOT_GATE_QUALIFIED


# -- carrying EST_DRIFT_P95 across, which used to be a person copying two numbers by hand ----------


#: What ``scripts/measure_est_drift.py`` stamps on its artifact. Checked rather than assumed: this
#: mode reads a number out of another script's file and writes it into the committed gate document,
#: and "it was JSON and it had the right key" is not a reason to believe it is that number.
EST_DRIFT_SCHEMA = "wam.est_drift/1"


def _artifact_rel(path: Path) -> str:
    """Repo-relative when it is under the repo, ABSOLUTE otherwise. Never raises.

    Absolute and not as-typed: this string goes into ``est_drift_source``, which is how a reader
    two months later finds the file the number came out of, and a relative path is only meaningful
    beside the working directory nobody recorded.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(_REPO_ROOT))
    except ValueError:
        return str(resolved)


def est_drift_measurement(path: Path) -> dict[str, Any]:
    """The EST_DRIFT_P95 half, read out of its own artifact and checked before it is quoted.

    Every refusal here is a way the subtraction in PR-08 §6 stops being arithmetic: a disqualified
    run (a fake capture, a partial run, an ungated estimator, a segmenter that disagrees with
    GEOM_TOL's), a null number, or an artifact that does not say which segmenter produced it. The
    last one is the whole point of this mode: the name travels WITH the number, mechanically,
    instead of being retyped from memory by whoever merged the two files.
    """
    if not path.is_file():
        raise MethodUnavailable(
            f"FATAL: no EST_DRIFT_P95 artifact at {path}. scripts/measure_est_drift.py writes it "
            "(PR-08 §4 steps 1-4)."
        )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MethodUnavailable(f"FATAL: {path} is not readable JSON: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != EST_DRIFT_SCHEMA:
        raise MethodUnavailable(
            f"FATAL: {path} does not carry schema {EST_DRIFT_SCHEMA!r} (got "
            f"{(doc or {}).get('schema')!r} ). This mode copies a measured number into the "
            "committed gate\n       document and will not do that from a file it cannot identify."
        )
    if not doc.get("gate_qualified"):
        raise MethodUnavailable(
            f"FATAL: {path} records gate_qualified = {doc.get('gate_qualified')!r}, so its p95 "
            "MUST NOT be subtracted from\n       GEOM_TOL. measure_est_drift's own reasons: "
            + "; ".join(doc.get("gate_disqualified_reasons") or ["(none recorded)"])
            + "\n       Nothing was written."
        )
    drift = doc.get("est_drift_p95_px")
    if not isinstance(drift, (int, float)):
        raise MethodUnavailable(
            f"FATAL: {path} records est_drift_p95_px = {drift!r}. There is no number to carry."
        )
    estimators = doc.get("estimators")
    name = estimators.get("name") if isinstance(estimators, Mapping) else None
    if not isinstance(name, str) or not name.strip():
        raise MethodUnavailable(
            f"FATAL: {path} records est_drift_p95_px = {drift} and no estimators.name, so the "
            "segmenter that\n"
            "       produced it cannot be named. PR-08 §4 step 2's 'the SAME segmenter' is the "
            "join between the two\n"
            "       halves of GEOM_TOL - EST_DRIFT_P95 and there is nothing here to join on. "
            "Nothing was written."
        )
    return {
        "est_drift_p95_px": float(drift),
        "name": name,
        "resolution_hw": doc.get("resolution_hw"),
        "segmenter_contract": (doc.get("geom_tol_cross_check") or {}).get(
            "this_segmenter_contract"
        ),
        "measured_utc": doc.get("measured_utc"),
        "is_lower_bound": doc.get("is_lower_bound"),
        "path": path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def carry_est_drift_main(args: argparse.Namespace) -> int:
    """``--carry-est-drift``: fill the EST_DRIFT_P95 slots of the committed document, or refuse.

    WHY THIS EXISTS AS CODE RATHER THAN AS A SENTENCE IN A RUNBOOK. The two constants PR-08 §8
    item 4 requires are measured by two scripts into two files, and the gate reads ONE document.
    Somebody has to merge them, and until this mode existed that somebody was a person with a text
    editor — who could copy the number and not the segmenter name, at which point
    ``run_g0_gates._ca_mask_method_name`` can only answer "could not check" and no G0b run can ever
    return 0. The name is not a nicety attached to the number; it is the only evidence that the
    subtraction is arithmetic.

    This reads GEOM_TOL's own document, refuses every way the two halves can fail to be one
    measurement — a different segmenter name, a segmenter whose parameters differ field for field,
    a different pixel grid — and writes the number, the name, a provenance string naming the
    artifact and its digest, and the re-derived margin. Nothing else in the document is touched.
    """
    est = est_drift_measurement(args.carry_est_drift)
    out: Path = args.out
    if not out.is_file():
        raise MethodUnavailable(
            f"FATAL: {out} does not exist. --carry-est-drift fills two slots in the COMMITTED "
            "GEOM_TOL document;\n       it does not create one, because a document with an "
            "EST_DRIFT_P95 and no committed segmenter\n       contract is exactly what PR-08 §4 "
            "step 2 cannot be checked against."
        )
    refuse_default_out_without_contract(out)
    try:
        doc = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MethodUnavailable(f"FATAL: {out} is not readable JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise MethodUnavailable(f"FATAL: {out} is not a JSON object.")

    theirs, contract_at = committed_segmenter_contract(doc)
    if theirs is None:
        raise MethodUnavailable(
            f"FATAL: {out} records no segmenter block, so 'the same segmenter' cannot be checked "
            "against it.\n       Restore the committed contract from git; do not write an "
            "EST_DRIFT_P95 into a document that\n       cannot be joined to one."
        )

    # THE SAME SEGMENTER, checked as a segmenter and not as a string — the identical pair of
    # comparisons measure_est_drift.cross_check_geom_tol() makes from the other side, made here
    # because this is the moment the two numbers become one document.
    ours_grid, grid_at = document_pixel_grid(doc)
    if est["resolution_hw"] is not None and ours_grid is not None:
        if [int(v) for v in est["resolution_hw"]] != ours_grid:
            raise MethodUnavailable(
                f"FATAL: the two halves were measured on two pixel grids: GEOM_TOL {ours_grid} "
                f"({grid_at}) vs\n       EST_DRIFT_P95 {list(est['resolution_hw'])} "
                f"({est['path']}). PR-08 §6 subtracts them and that is\n       arithmetic on one "
                "grid only. Nothing was written."
            )
    if isinstance(est["segmenter_contract"], Mapping):
        disagreements = contract_disagreements(est["segmenter_contract"], theirs)
        if disagreements:
            lines = [
                "FATAL: the segmenter EST_DRIFT_P95 was measured with disagrees with the contract "
                f"committed at\n       {out} ({contract_at}):\n"
            ]
            for d in disagreements:
                lines.append(f"         {d['field']}: committed {d['geom_tol']!r}, est_drift "
                             f"{d['this_run']!r}\n")
            lines.append(
                "       A name is the one property of a segmenter that does not change when its "
                "behaviour does, so\n"
                "       the block is compared field for field. Nothing was written."
            )
            raise MethodUnavailable("".join(lines))

    record = dict(doc)
    record["est_drift_p95_px"] = est["est_drift_p95_px"]
    record[EST_DRIFT_NAME_FIELD] = est["name"]
    record["est_drift_source"] = (
        f"{_artifact_rel(est['path'])} measured_utc={est['measured_utc']} "
        f"sha256={est['sha256']} estimator={est['name']!r} "
        f"is_lower_bound={bool(est['is_lower_bound'])}"
    )
    # Declared, so that a later measure_geom_tol run carries both slots forward rather than
    # dropping the name it never wrote itself.
    declared = record.get("measurement_fields")
    record["measurement_fields"] = list(
        dict.fromkeys([*(declared or ()), *CONTRACT_MEASUREMENT_FIELDS])
    )
    # The name and the number land together or neither lands: this is the same guard the measuring
    # and merging paths wear, here checking the join it exists for.
    refuse_unnamed_est_drift(record, out)
    if "est_drift_p95_blocked_by" in record:
        record["est_drift_p95_blocked_by"] = None

    tol_key, tol = None, None
    for key in ("geom_tol_px", "GEOM_TOL_px"):
        if isinstance(record.get(key), (int, float)):
            tol_key, tol = key, float(record[key])
            break
    record["gate_margin_px"] = None if tol is None else tol - est["est_drift_p95_px"]

    side, digest = write_artifact(out, record)
    print(f"carried     EST_DRIFT_P95 {est['est_drift_p95_px']} px from "
          f"{_artifact_rel(est['path'])}", file=sys.stderr)
    print(f"segmenter   {est['name']} — matches this document's own mask method", file=sys.stderr)
    print(f"wrote       {out}", file=sys.stderr)
    print(f"sha256      {digest}  ({side})", file=sys.stderr)
    if tol is None:
        print("\nGEOM_TOL is still null in this document, so gate_margin_px is null and G0b has "
              "no budget.\n  This carry is half of PR-08 §8 item 4; measure GEOM_TOL for the "
              "other half.", file=sys.stderr)
        return EXIT_NOT_GATE_QUALIFIED
    margin = record["gate_margin_px"]
    print(f"margin      {tol_key} {tol} - EST_DRIFT_P95 {est['est_drift_p95_px']} = "
          f"{margin:.6f} px", file=sys.stderr)
    if margin <= 0.0:
        print("\nNON-POSITIVE MARGIN. PR-08 §6: that is the finding — record it. The move is a "
              "better estimator,\n  never a wider gate. run_g0_gates refuses this document.",
              file=sys.stderr)
        return EXIT_NOT_GATE_QUALIFIED
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        _check_mode_flags(args)
    except MethodUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_FATAL

    if args.carry_est_drift is not None:
        try:
            return carry_est_drift_main(args)
        except MethodUnavailable as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_FATAL

    if args.merge is not None:
        return merge_main(args)

    try:
        method = resolve_method(args)
        episodes, layout = find_episodes(args.corpus, args.camera_key)
    except MethodUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_FATAL

    # Counted BEFORE --limit truncates and before --shard selects, so n_episodes_found is what the
    # corpus HAS and n_episodes is what was measured. Taking it after truncation would make the two
    # agree by construction and hide exactly the partial run this pair exists to expose.
    n_episodes_found = len(episodes)
    all_keys = [ep.key for ep in episodes]

    # (index in the FULL enumeration, episode). The index is what the merge sorts on to rebuild the
    # pool in the order an un-sharded run would have concatenated it; it is written on every
    # per_episode entry on BOTH paths, so the merged artifact and the un-sharded one have the same
    # shape as well as the same numbers.
    selected: list[tuple[int, Episode]] = list(enumerate(episodes))
    if args.shard is not None:
        selected = select_shard(episodes, args.shard, args.num_shards)
        print(f"shard       {args.shard} of {args.num_shards}: "
              f"{len(selected)} of {n_episodes_found} episodes", file=sys.stderr)
        if not selected:
            # An empty shard cannot produce an honest artifact: it never decodes a frame, so it has
            # no resolution_hw, and an artifact without that field is one the consumer's cross-check
            # passes by saying nothing. Refusing here NAMES THE PARTITION as the cause. Left to fall
            # through, the run would die a few lines later on "no episode yielded any frames", which
            # is a sentence about the corpus and would send an operator to look at the clips.
            print(
                f"FATAL: shard {args.shard} of {args.num_shards} holds no episodes. The corpus has "
                f"{n_episodes_found}, and\n"
                "       assignment is a digest of the episode key, so a shard count near or above "
                "the episode count\n"
                "       leaves shards empty by chance. An empty shard cannot be written: it decodes "
                "no frame, so it\n"
                "       has no pixel grid to record, and an artifact with a null resolution_hw is "
                "one the consumer's\n"
                "       cross-check passes by comparing nothing. Lower --num-shards and re-run the "
                "whole partition —\n"
                "       the other shards belong to the old one and the merge will refuse to mix "
                "them.",
                file=sys.stderr,
            )
            return EXIT_FATAL
    if args.limit:
        selected = selected[: args.limit]
    # AFTER --limit, not before. A shard artifact's episode_keys is the list the merge's coverage
    # proof is built on, so it has to be what was actually measured — otherwise a --limit inside a
    # shard would leave the block claiming episodes that have no per_episode entry, and the union
    # would look complete while the pool was short. (--limit already forces gate_qualified false;
    # this makes the coverage check catch it too, independently.)
    shard_block: dict[str, Any] | None = (
        None if args.shard is None
        else _shard_block(args.shard, args.num_shards, selected, all_keys)
    )
    episodes = [ep for _, ep in selected]

    # OPENED BEFORE THE FIRST FRAME, and after the method is resolved so it can hold the module the
    # frames will actually go through. The adapter's counters are lifetime totals; this snapshots
    # them now so what lands in the artifact is this run's arithmetic and not a total shared with
    # whatever else this interpreter has driven.
    stats_probe = EstimatorStatsProbe.open(
        getattr(method, "stats_module", None),
        why_absent=(
            f"mask method {method.name!r} is not backed by an estimator adapter module "
            f"(frames_from={method.frames_from!r}), so there is no stats() to read. Only "
            f"--method {SAM2_METHOD_CLI} reaches one."
        ),
    )
    scores_at_run_start = stats_probe.mark()

    per_episode: list[dict[str, Any]] = []
    pooled: list[np.ndarray] = []
    skipped_no_masks: list[str] = []
    geometry: tuple[int, int] | None = None
    fps_seen: set[float] = set()
    n_frames = 0
    n_dropped = 0

    # Resolved ONCE, against a clip of the corpus actually being measured, and before any frame is
    # decoded — so a corpus this machine cannot read fails in a second rather than after the first
    # episode of four hundred. The masks path has no decoder and asks for none.
    decoder: Decoder | None = None
    try:
        if method.frames_from != "masks":
            probe_clip = next((ep.clip for ep in episodes if ep.clip is not None), None)
            if probe_clip is None:
                raise MethodUnavailable(
                    "FATAL: this method decodes video and not one episode carries a clip path."
                )
            decoder = resolve_decoder(args.decoder, probe_clip)
            print(f"decoder     {decoder.name} {decoder.version}", file=sys.stderr)
        for ep_index, ep in selected:
            scores_at_episode_start = stats_probe.mark()
            if method.frames_from == "masks":
                mask_dir = args.masks / ep.key
                if not mask_dir.is_dir():
                    skipped_no_masks.append(ep.key)
                    continue
                cents, size = episode_centroids_from_masks(mask_dir, args.min_area_px)
                fps = 0.0
            else:
                assert decoder is not None  # resolved above whenever frames_from != "masks"
                cents, size, fps = episode_centroids_from_video(
                    ep.clip, method, args.min_area_px, args.max_frames, decoder
                )
            if geometry is None:
                geometry = size
            elif geometry != size:
                raise MethodUnavailable(
                    f"FATAL: {ep.key} is {size[0]}x{size[1]} but the first episode was "
                    f"{geometry[0]}x{geometry[1]}. GEOM_TOL is in pixels and §4 subtracts "
                    "EST_DRIFT_P95 from it, which is arithmetic only on a shared grid."
                )
            if fps:
                fps_seen.add(round(fps, 3))
            d, dropped = displacements(cents, args.step_frames)
            n_frames += len(cents)
            n_dropped += dropped
            pooled.append(d)
            entry = {
                "episode": ep.key,
                # Position in the FULL enumeration of the corpus, not a counter over what this run
                # measured. It is the merge's sort key, and it is written on the un-sharded path
                # too so that the merged artifact and the un-sharded one are the same object.
                "episode_index": ep_index,
                "clip": str(ep.clip) if ep.clip else None,
                "n_frames": len(cents),
                "n_frames_with_centroid": int(sum(c is not None for c in cents)),
                "n_steps": len(cents) - args.step_frames if len(cents) > args.step_frames else 0,
                "n_steps_measured": int(d.size),
                "n_steps_dropped": int(dropped),
                "median_px": float(np.median(d)) if d.size else None,
                "p95_px": float(np.percentile(d, 95)) if d.size else None,
            }
            episode_scores = stats_probe.since(scores_at_episode_start)
            if shard_block is not None:
                # THE RAW DISPLACEMENTS, and only on the shard path. A median does not decompose,
                # so the merge cannot work from this episode's median — it needs the numbers the
                # median was taken over. float -> JSON -> float is exact (json.dumps renders a
                # float with repr, the shortest round-tripping string since Python 3.1), so this
                # is a lossless representation and there is no error bound to state. It is ~430 kB
                # per shard on the real corpus and it is NOT carried into the merged artifact.
                entry["displacements_px"] = [float(x) for x in d]
                if episode_scores is not None:
                    # THE RAW DETECTION SCORES OF THIS EPISODE, for the same reason and on the same
                    # path. A distribution decomposes no better than a median does, and pooling the
                    # shards' scores in shard order rather than in the corpus's own order would
                    # change mean, std and every histogram count — the merged artifact would then be
                    # approximately, rather than exactly, the un-sharded one. Attributing them to
                    # the episode is what lets the merge sort them by episode_index like everything
                    # else. ~215 kB per shard on the real corpus, and NOT carried into the merged
                    # artifact.
                    entry["detection_scores"] = list(episode_scores)
            per_episode.append(entry)
    except MethodUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_FATAL

    if not per_episode:
        print("FATAL: no episode yielded any frames — nothing was measured, which is not a pass.\n"
              + (f"       {len(skipped_no_masks)} clip(s) had no mask directory under "
                 f"{args.masks}." if skipped_no_masks else ""),
              file=sys.stderr)
        return EXIT_FATAL

    values = np.concatenate(pooled) if pooled else np.asarray([], dtype=float)
    n_steps_total = int(values.size + n_dropped)
    coverage = float(values.size / n_steps_total) if n_steps_total else 0.0
    geom_tol = float(np.median(values)) if values.size else None
    ep_medians = [e["median_px"] for e in per_episode if e["median_px"] is not None]

    coverage_ok = coverage >= args.min_coverage
    headline_valid = bool(values.size and coverage_ok)

    # A PARTIAL run is not a measurement of this corpus, and coverage cannot notice: coverage is a
    # fraction of the steps that were DECODED, so --limit 3 over 402 episodes reports 1.000 and
    # every other field reads like a finished run. The disqualification is therefore structural —
    # derived from the flags, not from the numbers those flags produced.
    partial_reasons: list[str] = []
    if args.limit:
        partial_reasons.append(
            f"--limit {args.limit}: {len(per_episode)} of {n_episodes_found} episodes in the "
            "corpus were measured, so this is a sample and not the corpus. coverage is computed "
            "over the decoded steps ONLY and cannot detect this."
        )
    if args.max_frames:
        # The flag only reaches the decoder, so on the --masks path it changes nothing. It still
        # disqualifies: a run asked for a truncated measurement, and a gate artifact is not the
        # place to reason about which code path happened to honour the request.
        applied = method.frames_from == "video"
        partial_reasons.append(
            f"--max-frames {args.max_frames}: each clip was truncated to its first "
            f"{args.max_frames} decoded frame(s), so the phases of the episode past that point "
            "(typically the transfer, which is where the displacement is) were never seen."
            if applied else
            f"--max-frames {args.max_frames} was requested. It applies to video decoding only and "
            f"this run read precomputed masks, so no truncation happened — but a run that asked to "
            "be truncated is not the committed measurement and is not treated as one."
        )

    gate_disqualified_reasons: list[str] = []
    if not values.size:
        gate_disqualified_reasons.append("no step yielded a displacement")
    if not coverage_ok:
        gate_disqualified_reasons.append(
            f"coverage {coverage:.3f} < --min-coverage {args.min_coverage}"
        )
    if not method.gate_qualified:
        gate_disqualified_reasons.append(
            f"mask method {method.name!r} is not gate-qualified"
        )
    gate_disqualified_reasons += partial_reasons

    gate_ok = bool(headline_valid and method.gate_qualified and not partial_reasons)
    fps = sorted(fps_seen)[0] if len(fps_seen) == 1 else None

    # The run's scores, in the order the frames were segmented, which for the un-sharded path is the
    # corpus's own enumeration order and for a shard is that order restricted to its episodes. Read
    # once here rather than concatenated out of per_episode, so the two agree by construction on the
    # path where both exist.
    run_scores = stats_probe.since(scores_at_run_start)

    record: dict[str, Any] = {
        "schema": SCHEMA,
        "writeup": WRITEUP,
        "rule": "T40_RULE_V1",
        "gate": "PR-08 §6 G0b",
        "measured_by": "scripts/measure_geom_tol.py",
        "measured_date": date.today().isoformat(),
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),

        "artifact_path": str(args.out),
        "artifact_sha256_sidecar": str(sidecar_path(args.out)),
        "artifact_is_tracked_default": args.out == DEFAULT_OUT,
        # What a consumer (cluster/discoverer/97_transfer25_restyle.sbatch) must assert before
        # quoting GEOM_TOL_px as G0b's tolerance. Written into the artifact rather than only into
        # the consumer so the two cannot drift apart silently.
        "consumer_asserts": [
            # First, because it is the one nothing checks automatically: cross_check_geom_tol()
            # RECORDS mask_method and does not compare it. Two segmenters subtract to a plausible
            # pixel number, which is the invisible failure this whole artifact exists against.
            "mask_method.name == pr08_est_drift.json estimators.name — PR-08 §4 step 2's 'the SAME "
            "segmenter'. Both are the estimator module's ESTIMATOR_NAME. A consumer holding a "
            f"finished artifact checks it against this document's own {EST_DRIFT_NAME_FIELD}, "
            "which --carry-est-drift writes beside est_drift_p95_px and refuses to omit "
            "(refuse_unnamed_est_drift); a budget carried across without it cannot be joined to "
            "this half at all and is refused. The measurement-time half of the same check is "
            "measure_est_drift.cross_check_geom_tol(), which disqualifies its own run with "
            "mask_method_disagrees_with_estimator — that one runs against whatever was on disk at "
            "the time and a later consumer gets no benefit from it",
            "the segmenter block agrees, and not only the name: the committed contract (top-level "
            "`segmenter`) and what ran (mask_method.params.segmenter) pin the detector, the "
            "segmenter, the depth model AND their revisions, the prompt, both threshold pairs, the "
            "box rule and the propagation mode. A name is the one property of a segmenter that "
            "does not change when its behaviour does; measure_geom_tol REFUSES to write over a "
            "committed contract this run disagrees with, and cross_check_geom_tol() disqualifies "
            "on segmenter_params_disagree_with_geom_tol",
            "resolution_hw == the [height, width] EST_DRIFT_P95 was measured at. GEOM_TOL - "
            "EST_DRIFT_P95 is arithmetic only on one grid. cross_check_geom_tol() compares it, and "
            "since 2026-08-22 an artifact that does not record it is disqualified by name rather "
            "than passing the comparison by saying nothing",
            "gate_qualified == true",
            "partial_measurement == false",
            "n_episodes == n_episodes_found",
            "step_frames == the step the consumer intends to gate under (GEOM_TOL scales ~linearly "
            "with it; see step_definition)",
            "coverage >= min_coverage",
            "sha256sum <artifact> matches <artifact>.sha256",
        ],
        # What the consumer of these two artifacts DOES and DOES NOT check, stated in the
        # machine-readable artifact and not only in prose — the field whose whole job is to tell a
        # later reader what was and was not compared. The two limits recorded here until
        # 2026-08-22 ("the estimator name is recorded, not compared" and "the grid comparison is
        # absence-permissive") were both closed that day and have been REPLACED rather than
        # deleted: an artifact that keeps publishing a closed limit teaches its reader to re-check
        # something by hand, and an artifact that silently drops one teaches nothing.
        "cross_check_limits": {
            "checked_by": "scripts/measure_est_drift.py cross_check_geom_tol()",
            "fields_it_reads": list(CROSS_CHECK_FIELDS_READ_BY_EST_DRIFT),
            "fields_this_artifact_guarantees": list(CROSS_CHECK_FIELDS_REQUIRED),
            "what_the_reader_now_enforces": (
                "The estimator NAME is compared, not merely copied "
                "(mask_method_disagrees_with_estimator). The committed segmenter block is compared "
                "field for field against the adapter's SEGMENTER_CONTRACT "
                "(segmenter_params_disagree_with_geom_tol). Neither comparison is "
                "absence-permissive any more: each field the reader needs and does not find is its "
                "own geom_tol_does_not_record_<field>, so an artifact that says nothing is "
                "disqualified rather than passed."
            ),
            "it_only_runs_at_measurement_time": (
                "All of the above happens inside a measure_est_drift run, against whatever is at "
                "configs/transfer25/pr08_geom_tol.json at that moment. A consumer that later picks "
                "up two finished artifacts and subtracts them is checked by nobody, which is why "
                "consumer_asserts exists and why it leads with the join key."
            ),
            "the_committed_contract_is_write_protected_here": (
                "measure_geom_tol.merge_committed_contract() compares the committed segmenter "
                "block already at --out against the adapter this run drove and REFUSES the whole "
                "run — exit 2, nothing written, no sidecar — on any disagreement, then copies the "
                "contract section forward verbatim. Writing the tracked default path with no "
                "contract present in it is refused outright. So the pre-commitment PR-08 §4 step 2 "
                "is checked against cannot be destroyed by the measurement it constrains, which is "
                "what used to happen the first time GEOM_TOL was measured."
            ),
        },

        "corpus": str(args.corpus),
        "corpus_layout": layout,
        "camera_key": args.camera_key,
        "n_episodes_found": n_episodes_found,
        "n_episodes": len(per_episode),
        "n_episodes_skipped_no_masks": len(skipped_no_masks),
        "episodes_skipped_no_masks": skipped_no_masks[:50],
        "n_frames": n_frames,

        "units": (f"pixels at {geometry[0]}x{geometry[1]}" if geometry else "pixels"),
        "frame_width": geometry[0] if geometry else None,
        "frame_height": geometry[1] if geometry else None,
        # THE GRID JOIN KEY, and it is [height, width] in that order because that is the order the
        # other end writes. measure_est_drift.cross_check_geom_tol() looks for "resolution_hw"
        # (falling back to "frame_hw") and compares it element for element against the Isaac
        # capture's [H, W]. Until this key existed that comparison read None and appended no reason:
        # the one check standing between §6 and a subtraction of pixels measured on two different
        # grids was a no-op, which is worse than absent because it reports a clean cross-check.
        # frame_width/frame_height stay exactly as they were; this duplicates them for the consumer.
        "resolution_hw": [geometry[1], geometry[0]] if geometry else None,
        "fps": fps,
        "step_frames": args.step_frames,
        "step_seconds": (args.step_frames / fps) if fps else None,
        "step_definition": (
            f"one step = {args.step_frames} source frame(s), overlapping offsets i -> i+"
            f"{args.step_frames}. PR-08 §6 does not define 'step'; this is the choice made here "
            "and GEOM_TOL scales with it."
        ),

        # PROVENANCE, beside mask_method and for the identical reason. Two numbers produced by two
        # different readers of the same bytes are not obviously the same quantity. It is NOT part of
        # the cross-check measure_est_drift.py runs: that side's frames come out of a renderer, so a
        # decoder field there would be a field about nothing.
        "decoder": ({
            "name": decoder.name,
            "version": decoder.version,
            "selected": args.decoder,
            "note": decoder.note,
        } if decoder is not None else None),

        "mask_method": {
            "name": method.name,
            "version": method.version,
            "gate_qualified": method.gate_qualified,
            "frames_from": method.frames_from,
            "provenance": method.provenance or None,
            "params": method.params,
            "centroid_rule": ("largest connected component by area"
                              if method.frames_from == "video"
                              else "mean of all nonzero mask pixels"),
            "min_area_px": args.min_area_px,
        },

        # WHAT THE ESTIMATOR SAW WHILE THIS RAN, beside what this measured. Additive and read-only:
        # nothing here feeds gate_qualified, no refusal reads it, and an adapter that exports no
        # stats() records an absence with a reason rather than zeros. See EstimatorStatsProbe for
        # why it is a difference and not a total, and for what the blocker it serves does and does
        # not consider discharged.
        # include_raw is FALSE on every path here, shards included, and that is the displacements'
        # rule and not an omission: a shard's raw values live in per_episode[*].detection_scores,
        # attributed to the episode that produced them, because that is what lets the merge rebuild
        # the pool in the corpus's own order. A second, un-attributed copy of the same list at the
        # top of the shard would be ~215 kB that can only ever disagree with the first.
        "estimator_stats": stats_probe.block(run_scores, include_raw=False),

        "GEOM_TOL_px": geom_tol,
        "geom_tol_px_median_of_episode_medians": (
            float(np.median(ep_medians)) if ep_medians else None
        ),
        "headline_valid": headline_valid,
        "gate_qualified": gate_ok,
        "gate_disqualified_reasons": gate_disqualified_reasons,
        "partial_measurement": bool(partial_reasons),
        "limit": args.limit,
        "max_frames": args.max_frames,
        "n_steps_total": n_steps_total,
        "n_steps_measured": int(values.size),
        "n_steps_dropped_object_not_visible": n_dropped,
        "coverage": coverage,
        "min_coverage": args.min_coverage,
        "coverage_scope": (
            "fraction of the steps ACTUALLY DECODED that yielded a displacement — it says nothing "
            "about how much of the corpus was decoded. Assert n_episodes == n_episodes_found and "
            "partial_measurement == false for that."
        ),

        "distribution": distribution(values, args.hist_bin_px),
        "per_episode": per_episode,
        "displacements_npy": str(args.dump_displacements) if args.dump_displacements else None,

        "est_drift_p95_px": None,
        "est_drift_p95_blocked_by": _est_drift_blocker(),
        "geom_tol_minus_est_drift_px": None,
        "notes": [
            "G0b holds the generator to GEOM_TOL - EST_DRIFT_P95. EST_DRIFT_P95 is null here — "
            "see est_drift_p95_blocked_by, checked against src/wam/robot/isaac_binding.py at run "
            "time. This number does not on its own license generation.",
            "G0b's prose gates object AND plate centroids; this tolerance is derived from the "
            "OBJECT centroid only. The plate is near-static, so applying this number to the plate "
            "is loose. PR-08 §6 is silent on the split and is registered, so it is recorded here "
            "rather than resolved.",
            "Steps where the object was not visible (hand occlusion, apple out of frame) are "
            "dropped and counted in n_steps_dropped_object_not_visible, never folded in as zero "
            "displacement.",
            "GEOM_TOL is a property of the SOURCE corpus. Re-running this script on restyled "
            "clips measures something else; the gate compares restyled centroids against their "
            "source frame's, with this number as the tolerance.",
            "GEOM_TOL scales ~linearly with step_frames, which PR-08 §6 does not define. A "
            "consumer must assert step_frames is the step it intends to gate under; quoting this "
            "number against a different step is a proportionally wrong gate, silently.",
            "PR-08 §8 item 4 requires GEOM_TOL to be measured AND COMMITTED before generation, so "
            f"the committed artifact is the tracked {DEFAULT_OUT_REL} with a .sha256 sidecar. An "
            "artifact under gitignored runs/ cannot be a pre-commitment; runs/ is for scratch and "
            "for --dump-displacements.",
            "That same tracked path holds the COMMITTED SEGMENTER CONTRACT, written before this "
            "measurement so that PR-08 §4 step 2's 'the same segmenter' is checkable. Writing this "
            "artifact does not replace it: merge_committed_contract() compared the committed block "
            "field for field against the adapter this run drove, would have refused the whole run "
            "with nothing written on any disagreement, and copied the contract section forward "
            "verbatim — see committed_contract_carried_from when this artifact carries one. A "
            "diff of this file against its previous version that changes spec_version, "
            "what_this_is or segmenter is not a measurement and must not be committed.",
            "PR-08 §4 step 2 requires EST_DRIFT_P95 to be measured with the SAME segmenter as "
            "this number, and the two artifacts join on mask_method.name: scripts/measure_est_drift"
            ".py records that same string as estimators.name (both are the estimator module's "
            "ESTIMATOR_NAME) and its cross_check_geom_tol() copies this whole block into "
            "geom_tol_cross_check. Two different names is two different quantities and "
            "GEOM_TOL - EST_DRIFT_P95 is then not arithmetic. The pixel grid joins on "
            "resolution_hw, [height, width].",
            "coverage is measured over the steps that were DECODED. --limit and --max-frames "
            "therefore cannot lower it, and both force gate_qualified=false on their own; see "
            "partial_measurement and gate_disqualified_reasons.",
        ],
    }

    # -- a shard is not a tolerance, and the record has to make that impossible to miss ------------
    #
    # Three independent statements, because one is a flag someone can overlook: the SCHEMA is not
    # the gate's schema (a consumer that checks it rejects this file without knowing sharding
    # exists), GEOM_TOL_px is null (a consumer that reads the number gets nothing to quote), and
    # is_shard says so in words. `gate_qualified` is left exactly as computed above, and on this
    # path it means one thing only: THIS SHARD IS FIT TO BE MERGED — gate-qualified mask method, no
    # --limit or --max-frames, coverage over the floor. Whether the CORPUS was covered is not a
    # question a shard can answer, and the merge is where it is asked.
    if shard_block is not None:
        record["schema"] = SHARD_SCHEMA
        record["is_shard"] = True
        record["shard"] = shard_block
        record["corpus_episode_keys"] = all_keys
        record["shard_median_px"] = geom_tol
        record["GEOM_TOL_px"] = None
        record["geom_tol_px_is_null_because"] = (
            f"This is shard {args.shard} of {args.num_shards}, not a measurement of the corpus. "
            "GEOM_TOL is defined by PR-08 §6 as the median over the SOURCE CLIPS, and a median "
            "does not decompose: pooling the shard medians is a different statistic. This shard's "
            "own median is recorded as shard_median_px, for diagnostics only. Run "
            "`measure_geom_tol.py --merge <shard artifacts>` to produce GEOM_TOL."
        )
        record["gate_qualified_scope"] = (
            "On a shard artifact gate_qualified means THIS SHARD IS FIT TO BE MERGED: a "
            "gate-qualified mask method, neither --limit nor --max-frames, and coverage over the "
            "floor. It is NOT a claim that the corpus was covered — no shard can make that claim. "
            "The merge refuses if one shard says true while another says false."
        )

    # Checked against the record that is about to be written, not against the code that built it.
    # Each of these fields is one the consumer needs, and an artifact missing one is refused by the
    # consumer under its own name (``geom_tol_does_not_record_<field>``) hours later, on a machine
    # that no longer has the corpus mounted. Catching it here costs nothing and is the same answer.
    # Nothing is written when it fires.
    absent = missing_cross_check_fields(record)
    if absent:
        print(
            "FATAL: this record is missing " + ", ".join(absent) + ", which "
            "scripts/measure_est_drift.py\n"
            "       cross_check_geom_tol() reads and disqualifies its own run over. Nothing was "
            "written. This is a\n"
            "       bug in measure_geom_tol.py, not in the corpus.",
            file=sys.stderr,
        )
        return EXIT_FATAL

    # THE COMMITTED CONTRACT, BEFORE A BYTE IS WRITTEN. --out defaults to the tracked path that
    # holds the pre-measurement segmenter contract; this either carries it forward verbatim or
    # refuses the run outright. See merge_committed_contract() for the failure it prevents.
    try:
        refuse_default_out_without_contract(args.out)
        carried = merge_committed_contract(args.out, record)
    except MethodUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_FATAL
    if carried is not None:
        print(f"contract    carried forward from {args.out} — this run's segmenter agrees with it "
              "field for field", file=sys.stderr)

    side, digest = write_artifact(args.out, record)
    if args.dump_displacements:
        args.dump_displacements.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.dump_displacements, values)

    print(f"corpus      {args.corpus} ({layout})", file=sys.stderr)
    print(f"method      {method.name} v{method.version} "
          f"(gate_qualified={method.gate_qualified})", file=sys.stderr)
    print(f"measured    {len(per_episode)} of {n_episodes_found} episodes, {n_frames} frames, "
          f"{values.size}/{n_steps_total} steps (coverage {coverage:.3f})", file=sys.stderr)
    print(f"step        {args.step_frames} frame(s)"
          + (f" = {args.step_frames / fps:.4f} s at {fps} fps" if fps else ""), file=sys.stderr)
    if shard_block is None:
        print(f"GEOM_TOL    {geom_tol if geom_tol is None else round(geom_tol, 4)} px"
              + (f" at {geometry[0]}x{geometry[1]}" if geometry else ""), file=sys.stderr)
    else:
        print(f"shard med   {geom_tol if geom_tol is None else round(geom_tol, 4)} px"
              + (f" at {geometry[0]}x{geometry[1]}" if geometry else "")
              + " — DIAGNOSTIC. This is not GEOM_TOL and averaging these across shards is not "
                "GEOM_TOL either;", file=sys.stderr)
        print("            a median does not decompose. Run --merge over the shard artifacts.",
              file=sys.stderr)
    print(f"wrote       {args.out}", file=sys.stderr)
    print(f"sha256      {digest}  ({side})", file=sys.stderr)

    if partial_reasons:
        print("\nPARTIAL MEASUREMENT — gate_qualified is false and this artifact MUST NOT be "
              "committed as GEOM_TOL:", file=sys.stderr)
        for r in partial_reasons:
            print(f"                    - {r}", file=sys.stderr)
        print("                    Re-run with neither --limit nor --max-frames to produce the "
              "committed number.", file=sys.stderr)
    if not method.gate_qualified:
        print("\nNOT GATE-QUALIFIED: this number came from "
              f"{method.name!r}, which is not a segmenter whose output PR-08 §6 G0b can be set "
              "from.\n"
              "                    Do NOT quote it as GEOM_TOL. It is recorded, and stamped, so "
              "that it cannot be\n"
              "                    mistaken for the committed value later.", file=sys.stderr)
    if not coverage_ok:
        print(f"\nCOVERAGE {coverage:.3f} < --min-coverage {args.min_coverage}: the object was "
              "invisible for too much of the corpus\n"
              "                    for the median over the rest to describe it. headline_valid is "
              "false.", file=sys.stderr)
    return EXIT_OK if gate_ok else EXIT_NOT_GATE_QUALIFIED


if __name__ == "__main__":
    raise SystemExit(main())
