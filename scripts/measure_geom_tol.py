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
**It will not invent a centroid.** An object centroid needs a segmenter that can find the apple.
Nothing in this repo, this virtualenv or this machine's local weights can (checked at run time, and
the failure names every package and weight directory it looked for). The obvious stand-in — threshold
the red pixels — produces a *plausible* number on a red apple photographed on a table, and a
plausible-but-wrong GEOM_TOL is the single most expensive failure this measurement has, because it
is invisible: it does not crash, it does not look odd, it just sets the gate to the wrong place and
every downstream verdict inherits it. So the heuristic exists here, but it is called
``hsv-red-diagnostic``, it can only be reached by typing that name, it stamps
``gate_qualified: false`` into the artifact, and it exits non-zero. ``--method auto`` never selects
it; with no segmenter wired, ``auto`` fails loudly and measures nothing.

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
2   fatal: nothing was measured (no segmenter, no clips, no mask provenance, mixed geometry).
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
from typing import Any

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
        "       Related, and separately blocking: PR-08 §4 step 0 (EST_DRIFT_P95) needs the",
        "       distance_to_camera and semantic_segmentation annotators in",
        "       src/wam/robot/isaac_binding.py, which today wires only \"rgb\".",
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
                centroid_of_mask(hsv_red_mask(frame, method), largest_component=True,
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
    ap.add_argument("--method", choices=("auto", "precomputed", "hsv-red-diagnostic"),
                    default="auto",
                    help="'auto' (default) uses --masks if given and otherwise FAILS, naming the "
                         "missing segmenter. 'hsv-red-diagnostic' is not a segmenter and exits 3")
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
    if args.method == "hsv-red-diagnostic":
        return hsv_red_method(args.min_area_px)
    if args.method == "precomputed" or (args.method == "auto" and args.masks is not None):
        if args.masks is None:
            raise MethodUnavailable("FATAL: --method precomputed needs --masks.")
        return load_precomputed_method(args.masks)
    raise MethodUnavailable(no_segmenter_message())


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
            "gate_qualified == true",
            "partial_measurement == false",
            "n_episodes == n_episodes_found",
            "step_frames == the step the consumer intends to gate under (GEOM_TOL scales ~linearly "
            "with it; see step_definition)",
            "coverage >= min_coverage",
            "sha256sum <artifact> matches <artifact>.sha256",
        ],

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
            "coverage is measured over the steps that were DECODED. --limit and --max-frames "
            "therefore cannot lower it, and both force gate_qualified=false on their own; see "
            "partial_measurement and gate_disqualified_reasons.",
        ],
    }

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
