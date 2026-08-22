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

EXIT STATUS
-----------
0   measured with a gate-qualified mask method, coverage above ``--min-coverage``.
2   fatal: nothing was measured and NOTHING was written — no segmenter, no clips, no mask
    provenance, mixed geometry, two segmenters named on one command line, an adapter that says its
    checkpoints are absent, or an estimator that raised while segmenting. Every way this script can
    fail lands here; a traceback out of ``main`` would be a bug in the script, not a fourth status.
3   measured, but the number MUST NOT be used as G0b's tolerance — an ungated mask method, coverage
    below the floor, or a partial run (``--limit`` / ``--max-frames``). The artifact is still
    written: PR-08 §6 requires GEOM_TOL to be recorded regardless of verdict, and "we tried and this
    is what came out" is a record. ``gate_disqualified_reasons`` says which of the four it was.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from prepare_cosmos_corpus import resolve_camera  # noqa: E402

SCHEMA = "wam.geom_tol/1"
WRITEUP = "docs/preregistration/PR-08-photoreal-augmentation.md"

#: The COMMITTED gate artifact. Tracked, not gitignored, and anchored to the repository root rather
#: than to the caller's CWD — PR-08 §8 item 4 wants GEOM_TOL *committed* before generation, and a
#: path under ``runs/`` (gitignored) or a path that moves with the shell's CWD cannot be that. See
#: the module docstring; ``configs/transfer25/pr08_style_partition.json`` is the precedent.
DEFAULT_OUT_REL = "configs/transfer25/pr08_geom_tol.json"
DEFAULT_OUT = _REPO_ROOT / DEFAULT_OUT_REL

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

    Gate qualification is OPT-IN twice over, and both halves are the adapter's to assert. The module
    must say ``GATE_QUALIFIED = True`` — absent means false, exactly as ``measure_est_drift`` reads
    it, so a stub cannot become a gate input by being importable. And it must NAME ITS WEIGHTS, or
    qualification is withheld here regardless of what it claims: the artifact's only job is to make
    the identical estimator re-runnable on the restyled clips, and "sam2" without a checkpoint is a
    family of segmenters, not one.
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
    withheld: str | None = None
    if declared_gate and not checkpoints:
        withheld = (
            f"{SAM2_ADAPTER_SPEC} sets GATE_QUALIFIED=True but names no checkpoints (looked for "
            f"ESTIMATOR_CHECKPOINTS and {', '.join(CHECKPOINT_ATTRS)}). A tolerance that cannot say "
            "which weights produced it cannot be re-run with the same estimator at gate time, which "
            "is the only thing GEOM_TOL is for."
        )
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
        gate_qualified=declared_gate and bool(checkpoints),
        frames_from="video",
        params={
            "cli_method": SAM2_METHOD_CLI,
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


def episode_centroids_from_video(clip: Path, method: MaskMethod, min_area: int, max_frames: int) -> tuple[list[tuple[float, float] | None], tuple[int, int], float]:
    # No default and no name lookup, and checked before a single frame is decoded. A video method
    # with no segmenter attached is a bug in resolve_method; the one thing that must not happen is
    # for it to be papered over with the red-pixel heuristic while the artifact still carries the
    # other method's name.
    if method.mask_fn is None:
        raise MethodUnavailable(
            f"FATAL: mask method {method.name!r} decodes video but carries no segmenter "
            "(mask_fn is None). resolve_method must attach one; nothing here guesses which."
        )
    import cv2

    cap = cv2.VideoCapture(str(clip))
    try:
        if not cap.isOpened():
            raise MethodUnavailable(f"FATAL: cv2 could not open {clip}.")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cents: list[tuple[float, float] | None] = []
        size: tuple[int, int] | None = None
        while max_frames <= 0 or len(cents) < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
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
                f"FATAL: {clip} opened but decoded no frames — the container parses and the codec "
                "does not. See scripts/verify_clip_decode.py."
            )
        return cents, size, fps
    finally:
        cap.release()


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

    Its grid comparison is ``if theirs_hw is not None and list(theirs_hw) != list(resolution_hw)``:
    absence is silence, and silence there is indistinguishable downstream from a comparison that
    ran and agreed. The same shape as the ``gate_qualified`` default-permissiveness this repo has
    already removed once (``97_transfer25_restyle.sbatch``: "saying nothing is exactly what a
    fabricated artifact does"). The reader is not this module's to fix — this is the half that is:
    an artifact that would be read permissively is never written at all.

    ``None`` counts as missing. A null ``resolution_hw`` is exactly the value that makes the
    consumer's check say nothing.
    """
    return [k for k in CROSS_CHECK_FIELDS_REQUIRED if record.get(k) is None]


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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--corpus", type=Path, required=True,
                    help="the SOURCE corpus: a LeRobot v2.1 root (meta/info.json + videos/) or a "
                         "directory of .mp4 clips. GEOM_TOL is a property of the source, never of "
                         "the generated clips")
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        method = resolve_method(args)
        episodes, layout = find_episodes(args.corpus, args.camera_key)
    except MethodUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_FATAL

    # Counted BEFORE --limit truncates, so n_episodes_found is what the corpus HAS and n_episodes is
    # what was measured. Taking it after truncation would make the two agree by construction and
    # hide exactly the partial run this pair exists to expose.
    n_episodes_found = len(episodes)
    if args.limit:
        episodes = episodes[: args.limit]

    per_episode: list[dict[str, Any]] = []
    pooled: list[np.ndarray] = []
    skipped_no_masks: list[str] = []
    geometry: tuple[int, int] | None = None
    fps_seen: set[float] = set()
    n_frames = 0
    n_dropped = 0

    try:
        for ep in episodes:
            if method.frames_from == "masks":
                mask_dir = args.masks / ep.key
                if not mask_dir.is_dir():
                    skipped_no_masks.append(ep.key)
                    continue
                cents, size = episode_centroids_from_masks(mask_dir, args.min_area_px)
                fps = 0.0
            else:
                cents, size, fps = episode_centroids_from_video(
                    ep.clip, method, args.min_area_px, args.max_frames
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
            per_episode.append({
                "episode": ep.key,
                "clip": str(ep.clip) if ep.clip else None,
                "n_frames": len(cents),
                "n_frames_with_centroid": int(sum(c is not None for c in cents)),
                "n_steps": len(cents) - args.step_frames if len(cents) > args.step_frames else 0,
                "n_steps_measured": int(d.size),
                "n_steps_dropped": int(dropped),
                "median_px": float(np.median(d)) if d.size else None,
                "p95_px": float(np.percentile(d, 95)) if d.size else None,
            })
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
            "segmenter'. Both are the estimator module's ESTIMATOR_NAME. NOTHING ENFORCES THIS: "
            "measure_est_drift.cross_check_geom_tol() copies mask_method into geom_tol_cross_check "
            "without comparing it, so the consumer must",
            "resolution_hw == the [height, width] EST_DRIFT_P95 was measured at. GEOM_TOL - "
            "EST_DRIFT_P95 is arithmetic only on one grid. cross_check_geom_tol() does compare "
            "this — but only when the key is present, so a consumer reading an artifact from "
            "anywhere else must assert it EXISTS before trusting a clean grid check",
            "gate_qualified == true",
            "partial_measurement == false",
            "n_episodes == n_episodes_found",
            "step_frames == the step the consumer intends to gate under (GEOM_TOL scales ~linearly "
            "with it; see step_definition)",
            "coverage >= min_coverage",
            "sha256sum <artifact> matches <artifact>.sha256",
        ],
        # The two places the CONSUMER of these two artifacts is weaker than it reads, stated in the
        # machine-readable artifact and not only in prose. Neither is repairable from this side:
        # both live in scripts/measure_est_drift.py.
        "cross_check_limits": {
            "checked_by": "scripts/measure_est_drift.py cross_check_geom_tol()",
            "fields_it_reads": list(CROSS_CHECK_FIELDS_READ_BY_EST_DRIFT),
            "fields_this_artifact_guarantees": list(CROSS_CHECK_FIELDS_REQUIRED),
            "estimator_name_is_recorded_not_compared": (
                "cross_check_geom_tol() copies mask_method into geom_tol_cross_check and never "
                "asserts mask_method.name == estimators.name. PR-08 §4 step 2 requires the SAME "
                "segmenter; a mismatch is caught by nobody automatically. See consumer_asserts[0]."
            ),
            "grid_comparison_is_absence_permissive": (
                "cross_check_geom_tol() compares resolution_hw only when the key is present "
                "('if theirs_hw is not None and ...'), so an artifact WITHOUT it passes the grid "
                "check by saying nothing. This module refuses to write such an artifact at all "
                "(missing_cross_check_fields), which closes the case it controls and not the "
                "reader; an artifact from any other producer must be checked for the key's "
                "PRESENCE before its clean grid check means anything."
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

    # Checked against the record that is about to be written, not against the code that built it.
    # A field that is absent or null here is a field the consumer's cross-check reads as silence,
    # and its grid comparison treats silence as agreement — so an artifact missing one would report
    # a clean cross-check of two grids nobody compared. Nothing is written when it fires.
    absent = missing_cross_check_fields(record)
    if absent:
        print(
            "FATAL: this record is missing " + ", ".join(absent) + ", which "
            "scripts/measure_est_drift.py\n"
            "       cross_check_geom_tol() reads. Its grid comparison is absence-permissive, so an "
            "artifact\n"
            "       without those fields does not fail the cross-check — it passes it silently, "
            "having compared\n"
            "       nothing. Nothing was written. This is a bug in measure_geom_tol.py, not in the "
            "corpus.",
            file=sys.stderr,
        )
        return EXIT_FATAL

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
    print(f"GEOM_TOL    {geom_tol if geom_tol is None else round(geom_tol, 4)} px"
          + (f" at {geometry[0]}x{geometry[1]}" if geometry else ""), file=sys.stderr)
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
