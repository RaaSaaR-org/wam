#!/usr/bin/env python3
"""Contact sheets of a BAND of the pooled robot-mask area distribution — PR-08 V13 §3.2.

    # the open upper tail (no upper limit)
    PYTHONPATH=src .venv/bin/python scripts/render_area_tail_sheet.py \\
        --pooled runs/pr08-robot-mask-area/POOLED.json \\
        --corpus /home/humanoid/wam-t041/pr08-apple-640x480-h264-lossless \\
        --out    runs/pr08-area-tail-look \\
        --max-frames 48

    # a closed band: everything immediately BELOW the widest measured gap
    PYTHONPATH=src .venv/bin/python scripts/render_area_tail_sheet.py \\
        --threshold 0.36 --max-fraction 0.6015462239583333 --max-frames 44 \\
        --out runs/pr08-below-bound-look

WHY THIS EXISTS, IN THE RULE'S OWN WORDS
----------------------------------------
``docs/preregistration/PR-08-V13-area-bound-decision-rule.md`` §3.2 fixes the minimum content of the
``bound_rationale`` that a committed area bound must carry, and one of its five items is **whether
those frames were looked at, and what they were**. The project owner is about to make that decision.
There is currently nothing for them to look at: the pooled artifact is 171 625 floats, the gap
analysis is eight candidate cuts, and neither is a picture of a mask.

**This script produces exactly that evidence and NOTHING else.** It never writes a bound, it never
proposes one, it sets no reviewer-confirmation flag, and it discharges no blocker and no gate.
Producing evidence and deciding that the evidence is sufficient are two different acts and only the
second one is a judgement — ``scripts/audit_apple_masks.py`` says this about PR-08 §4's blockers in
the same words, because it is the same weakness. A script that could do both would discharge a
decision by being executed.

THE DISTINCTION THE SHEETS HAVE TO LET A PERSON MAKE
----------------------------------------------------
V13 §2 names it and it is the only reason the bound exists:

  *"§6 G0c composites the real robot's pixels back over every generated frame. The bound exists for
  one failure: a robot mask that has grounded on the table, the background, or the whole scene."*

and, pointing the other way,

  *"A bound that fires on legitimate frames destroys clips. The robot genuinely occupies a large
  fraction of some frames — a near-camera arm at the grasp is not a defect."*

So every tile here is one question: **is this a legitimately near-camera arm at the grasp, on which
the bound must not fire, or a mask that has grounded on the scene, on which it must?** No number in
this artifact answers that and none is offered as answering it. The area fraction is precisely the
statistic that cannot tell the two apart — that is why the tail has to be looked at rather than
percentiled.

THE HARDWARE CAVEAT, WHICH IS LOAD-BEARING
------------------------------------------
**The distribution was measured on the cluster's H200. These masks are RE-RENDERED here, on this
workstation's RTX 5090.** They are therefore not the same execution that produced the numbers, and
this project already has a measurement of that gap:
``docs/preregistration/PR-08-RESULT-2026-08-25-detector-noise-floor.md`` §3.3 records the same
masker, same pins, same tree and same 1 603 frames disagreeing between an RTX 5090 and the H200 on
the empty-mask count by 13 of 1 603 (0.81 %), and attributes it — as a mechanism consistent with the
observation, not as a proven cause — to a detection threshold sitting ~0.001 above the detector's
noise floor.

**That documented disagreement concerns frames within ~0.001 of the detection threshold. These are
not those frames**: the tail is two thresholds and three orders of magnitude away from the noise
floor, and a 0.68-area mask is not a coin-flip detection. So the caveat is not a reason to distrust
the sheets — it is a reason to CHECK them, per frame, which is what this does. Every tile carries
the fraction the H200 recorded and the fraction this GPU just recomputed, and a tile whose two
fractions disagree by more than ``--mismatch-tolerance`` is drawn with a red border and counted.

**The mismatch count belongs in any rationale that quotes these sheets.** It is the only thing that
makes a tile evidence about *the mask that produced the number* rather than a picture of a
differently-executed mask that happens to come from the same frame index.

WHAT IT WILL NOT DO
-------------------
- **It will not write, propose or imply a value for the bound.** ``TAIL_SAMPLE.json`` carries
  ``writes_a_bound: false`` and a ``not_a_discharge`` sentence, and
  ``tests/test_render_area_tail_sheet.py`` asserts that neither this source nor the artifact ever
  assigns one.
- **It will not touch the masker.** ``scripts/robot_composite.py``'s ``Sam2RobotMasker`` is driven
  unmodified at its committed operating point with the committed ``ROBOT_TEXT_PROMPT``. No
  threshold, prompt or config is read from the command line, because ``build_masker()`` deliberately
  takes no arguments.
- **It will not decode frames its own way.** Frames come from ``robot_composite.decode_clip``, which
  is the function ``measure_source_mask_area`` used to produce the distribution and which
  ``cluster/discoverer/106_measure_robot_mask_area.sbatch`` drove. If the tile were decoded by any
  other path it would not be showing the mask that produced the number, and the recorded-vs-
  recomputed comparison would be measuring the decoder instead of the GPU. The decoded frame count
  is checked against the pooled artifact's own ``n_frames`` per episode and the run refuses on a
  disagreement rather than indexing into a differently-decoded clip.
- **It will not call a model's reading of these overlays "a person looking".** This process is a
  correlated observer of an instrument built by models of the same family; what it writes down is a
  finding, not the check. The artifact records no reviewer confirmation of any kind and there is no
  field here for one.

THE BAND, AND WHY A LOOK BELOW THE GAP IS THE CONTROL FOR A LOOK ABOVE IT
-------------------------------------------------------------------------
``--threshold`` and ``--max-fraction`` together define a **band** ``threshold <= f <= max_fraction``
over the *measured* distribution. ``--max-fraction`` defaults to ``None``, which leaves the upper end
open and selects the upper tail exactly as before.

**The band is a selection window and nothing else.** It says which frames get rendered so that a
person can look at them. Both of its edges are measured features of the distribution — the edges of
the widest measured gap in ``AREA_GAP_ANALYSIS.json`` are the obvious ones to quote, because they are
where the distribution itself separates. **NEITHER EDGE IS A BOUND AND NEITHER IS A CANDIDATE
BOUND.** Rendering a band does not propose its edges, and nothing in this module or its artifact may
be read as proposing them.

The reason the upper limit exists is that V13 §3.1 asks for a bound that separates **two**
populations, and a sheet of the open upper tail is a look at one of them. A population that is
separable only when read from above has not been shown to be separable: if the frames just below the
gap look the same as the frames just above it, then the gap is a gap in the *statistic* and not a
line between two kinds of frame, and no sheet of the tail alone can reveal that. **So a look below
the gap is the control for a look above it**, and the band is what makes that control renderable
with the same instrument, the same decode path, the same masker and the same recorded-vs-recomputed
comparison — which is the only way the two looks are comparable at all.

THE SELECTION RULE
------------------
Deterministic, no RNG, and written into the artifact verbatim so the sample can be rebuilt from the
artifact alone and its bias argued with:

1. Every ``(episode, frame_index, recorded_fraction)`` in the pooled artifact whose
   ``recorded_fraction >= --threshold`` — and, when ``--max-fraction`` is given, whose
   ``recorded_fraction <= --max-fraction`` — is a candidate. ``area_fractions`` is indexed by frame
   index, so the index IS the frame number.
2. Candidates are grouped by episode and the episodes are sorted by id.
3. The budget ``--max-frames`` is handed out one frame at a time, cycling through those sorted
   episodes, until the budget is spent or every candidate is taken. An episode already exhausted is
   skipped on later passes.
4. Within an episode, its quota is taken from its own candidate list at an even stride with both
   endpoints included (``floor(i * (m - 1) / (k - 1) + 0.5)``, ties up). A quota of one takes that
   episode's first tail frame, because one sample cannot include two endpoints.

**The bias this leaves is stated rather than averaged away**, in the artifact's ``selection_rule``
block: when the budget is smaller than the number of episodes carrying tail frames — 48 against 175
at the default threshold — step 3's first pass ends before the later episodes are reached, so the
sample spans the first N episode ids in sorted order and not the whole corpus. It is a sample of the
tail, deliberately even ACROSS episodes so that one pathological episode cannot fill the sheets, and
it is not a random sample of the tail. ``population.episodes_not_sampled`` says how many episodes
went unseen.

The default ``--threshold`` is ``0.6802766927083334``: the ``tail_edge_above`` of the widest gap in
``runs/pr08-robot-mask-area/AREA_GAP_ANALYSIS.json``, which is the smallest measured fraction above
that gap. The default ``--max-fraction`` is ``None``, so the default selection is the open upper
tail: 1 385 frames in 175 episodes sit at or above the default threshold. **Both numbers are
measured edges of a measured distribution, and NEITHER IS A BOUND OR A CANDIDATE BOUND.**
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import audit_apple_masks as aam  # noqa: E402  (captioned/contact_sheet/boundary conventions)


class TailLookError(RuntimeError):
    """A refusal this script makes on purpose, printed as a refusal rather than a traceback."""


# --------------------------------------------------------------------------------------------
# the constants that are about the PICTURE, and nothing else
# --------------------------------------------------------------------------------------------

#: The measured upper edge of the widest gap in AREA_GAP_ANALYSIS.json. An edge of the tail
#: population, quoted so the sample is reproducible. NOT a bound and not a candidate bound.
DEFAULT_TAIL_EDGE = 0.6802766927083334

#: The default upper limit of the band: none at all. With this the selection is the open upper tail
#: and the behaviour is exactly what it was before --max-fraction existed.
DEFAULT_MAX_FRACTION = None

#: Magenta. Nothing in this corpus — red apple, white plate, wood table, grey-black arm — is this
#: colour, so a tinted region is unambiguously the mask and not the scene.
COLOR_MASK = (255, 0, 220)

#: Reused from audit_apple_masks so the two audits' red means the same thing: LOOK AT THIS.
COLOR_FLAG = aam.COLOR_FLAG

#: Written into the artifact in full. The sheet header carries SHEET_LEGEND, which is the same
#: statement trimmed to what fits across four tiles at the default font — a title that runs off the
#: right edge of the picture is a legend nobody can read.
LEGEND = ("magenta = robot mask, RE-RENDERED on this machine (tint + outline) | "
          "red border = re-render disagrees with the measured fraction by more than the tolerance")

SHEET_LEGEND = "magenta = re-rendered robot mask | red border = mismatch"

SCHEMA = "pr08-area-tail-look/1"

NOT_A_DISCHARGE = (
    "This artifact is EVIDENCE and not a decision. It discharges no blocker and no gate: not "
    "PR-08 §6 G0c, not §4's blockers, not T40_RULE_V13, which stays an UNSIGNED DRAFT that only "
    "the project owner may sign. It writes no bound and proposes none, and it records no "
    "confirmation that a person has reviewed these sheets — a model reading overlays produced by "
    "an instrument built by models of the same family is a correlated observer, and what it writes "
    "down is a finding, not the check."
)

WHAT_THIS_IS_FOR = (
    "T40_RULE_V13 §3.2 requires a bound_rationale to state 'whether those frames were looked at, "
    "and what they were'. Nothing existed for a person to look at. These sheets are that and only "
    "that."
)

THE_DISTINCTION = (
    "V13 §2: the bound exists for a robot mask that has grounded on the table, the background or "
    "the whole scene, and it must NOT fire on a legitimately near-camera arm at the grasp, which "
    "is not a defect. The area fraction cannot tell those two apart; that is why the tail is "
    "looked at instead of percentiled. No number in this artifact adjudicates a tile."
)

HARDWARE_CAVEAT = (
    "THE DISTRIBUTION WAS MEASURED ON THE CLUSTER'S H200; THESE MASKS ARE RE-RENDERED ON THIS "
    "WORKSTATION'S RTX 5090, so they are not the same execution that produced the numbers. "
    "docs/preregistration/PR-08-RESULT-2026-08-25-detector-noise-floor.md section 3.3 documents "
    "cross-hardware disagreement between exactly these two machines on exactly this masker (13 of "
    "1603 frames, 0.81 %, on the empty-mask count), attributed as a consistent mechanism rather "
    "than a proven cause to a detection threshold ~0.001 above the detector's noise floor. That "
    "concerns frames within ~0.001 of the threshold and these tail frames are not those. The "
    "per-frame recorded-vs-recomputed comparison below is what makes each tile evidence about the "
    "mask that produced the number, and the mismatch count belongs in any rationale that quotes "
    "these sheets."
)

SELECTION_RULE_TEXT = (
    "Deterministic, no RNG. (1) Every (episode, frame_index, recorded_fraction) in the pooled "
    "artifact with recorded_fraction >= threshold, and (when max_fraction is not null) "
    "recorded_fraction <= max_fraction, is a candidate; area_fractions is indexed by "
    "frame index. (2) Candidates are grouped by episode and the episodes sorted by id. (3) The "
    "budget max_frames is handed out one frame at a time, cycling through those sorted episodes, "
    "until the budget is spent or every candidate is taken; an exhausted episode is skipped on "
    "later passes. (4) Within an episode its quota is taken from its own candidate list at an even "
    "stride with both endpoints included, index floor(i * (m - 1) / (k - 1) + 0.5) for i in 0..k-1 "
    "(ties go up, deduplicated, sorted); a "
    "quota of 1 takes that episode's first tail frame, because one sample cannot include two "
    "endpoints."
)

SELECTION_BIAS_TEXT = (
    "When max_frames is smaller than the number of episodes carrying tail frames (48 against 175 "
    "at the default threshold), step 3's first pass ends before the later episodes are reached, so "
    "the sample spans the first N episode ids in sorted order rather than the whole corpus. It is "
    "deliberately even ACROSS episodes so that one pathological episode cannot fill the sheets, "
    "and it is NOT a random sample of the tail. See population.episodes_not_sampled."
)

THRESHOLD_NOTE = (
    "The default threshold is AREA_GAP_ANALYSIS.json's widest-gap tail_edge_above: the smallest "
    "measured fraction above the widest measured gap, i.e. a measured edge of the tail population. "
    "It selects which frames are rendered. IT IS NOT A BOUND AND NOT A CANDIDATE BOUND, and "
    "nothing here may be read as proposing it as one."
)

BAND_NOTE = (
    "threshold and max_fraction together are a BAND, threshold <= f <= max_fraction, and the band "
    "is a SELECTION WINDOW OVER THE MEASURED DISTRIBUTION and nothing else: it says which frames "
    "get rendered so a person can look at them. A null upper edge leaves the band open at the top, "
    "which is the upper tail and the default. Both edges are measured features of a measured "
    "distribution — the edges of the widest measured gap are the obvious ones to quote, because "
    "that is where the distribution itself separates. NEITHER EDGE IS A BOUND AND NEITHER IS A "
    "CANDIDATE BOUND. Rendering a band does not propose its edges, and nothing in this artifact "
    "may be read as proposing either of them as one."
)

WHY_A_BAND = (
    "T40_RULE_V13 §3.1 asks for a bound that separates TWO populations, and a sheet of the open "
    "upper tail is a look at one of them. A LOOK BELOW THE GAP IS THE CONTROL FOR A LOOK ABOVE IT: "
    "a population that is separable only when read from above has not been shown to be separable. "
    "If the frames immediately below the gap look like the frames immediately above it, the gap is "
    "a gap in the statistic and not a line between two kinds of frame, and no sheet of the tail "
    "alone can show that. The band makes that control renderable with the same instrument, the "
    "same decode path, the same masker and the same recorded-vs-recomputed comparison, which is "
    "the only thing that makes the two looks comparable."
)


# --------------------------------------------------------------------------------------------
# the pooled artifact
# --------------------------------------------------------------------------------------------


def load_pooled(path: pathlib.Path) -> dict:
    """The pooled per-frame artifact, or a refusal.

    ``measurement_qualified`` must be exactly ``True``. ``robot_composite.measure_source_mask_area``
    stamps ``false`` on any run truncated by ``--limit`` or ``--stride``, and ``load_area_bound``
    refuses such an artifact by name; rendering sheets from one would put a picture of a shakedown
    in front of the person deciding the bound, labelled as the corpus. ``is not True`` rather than
    a truthiness test, because ``1`` and ``"true"`` are not the stamp.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        raise TailLookError(f"no such pooled artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    qualified = payload.get("measurement_qualified")
    if qualified is not True:
        raise TailLookError(
            f"{path} carries measurement_qualified={qualified!r}, not True.\n"
            "       That stamp means the distribution is not the corpus's — a --limit or --stride "
            "truncation, or a pilot.\n"
            "       load_area_bound refuses such an artifact by name and so does this: sheets "
            "rendered from it would be a\n"
            "       picture of a shakedown shown to the person deciding a bound over the corpus."
        )
    episodes = list(payload.get("per_episode") or ())
    if not episodes:
        raise TailLookError(f"{path} lists no per_episode records.")
    return payload


def tail_candidates(
    pooled: dict,
    threshold: float,
    max_fraction: float | None = DEFAULT_MAX_FRACTION,
) -> "OrderedDict[str, list[tuple[int, float]]]":
    """``{episode_id: [(frame_index, recorded_fraction), ...]}`` for every frame inside the band.

    The band is ``threshold <= recorded_fraction <= max_fraction``, both ends INCLUSIVE, and
    ``max_fraction=None`` — the default — leaves the upper end open, which is the upper tail and
    exactly the behaviour this function had before the upper limit existed. **The band is a
    selection window over the measured distribution; neither of its edges is a bound or a candidate
    bound** (see BAND_NOTE, which the artifact carries verbatim).

    Keyed in sorted episode order and each list in frame order, which is what makes the selection
    below reproducible from the artifact alone rather than from this dict's construction order.
    """
    threshold = float(threshold)
    upper = None if max_fraction is None else float(max_fraction)
    if upper is not None and upper < threshold:
        raise TailLookError(
            f"--max-fraction {upper!r} is below --threshold {threshold!r}; the band "
            f"{threshold} <= f <= {upper} is empty by construction.\n"
            "       Neither number is a bound, but a band that cannot contain a frame is a typo "
            "rather than a selection."
        )
    grouped: dict[str, list[tuple[int, float]]] = {}
    for record in pooled.get("per_episode") or ():
        episode = str(record.get("episode"))
        fractions = record.get("area_fractions") or ()
        hits = [(int(i), float(f)) for i, f in enumerate(fractions)
                if float(f) >= threshold and (upper is None or float(f) <= upper)]
        if hits:
            grouped[episode] = hits
    return OrderedDict((key, grouped[key]) for key in sorted(grouped))


def even_stride_indices(count: int, take: int) -> list[int]:
    """``take`` positions out of ``count``, evenly spread, BOTH ENDPOINTS INCLUDED.

    ``take == 1`` returns ``[0]``: one sample cannot include two endpoints, and taking the first is
    the choice that does not depend on the episode's length.
    """
    count = int(count)
    take = int(take)
    if count <= 0 or take <= 0:
        return []
    if take >= count:
        return list(range(count))
    if take == 1:
        return [0]
    step = (count - 1) / (take - 1)
    # floor(x + 0.5) rather than round(), because round() is banker's rounding and a rule whose
    # tie-breaking depends on the parity of the result cannot be re-derived from the sentence in
    # the artifact. Ties go up, always, and the artifact says so.
    return sorted({math.floor(i * step + 0.5) for i in range(take)})


def select_frames(
    candidates: "OrderedDict[str, list[tuple[int, float]]]",
    budget: int,
) -> list[dict]:
    """The sample, as ``[{episode, frame_index, recorded_fraction}, ...]``.

    Round-robin across the sorted episodes, then an even stride inside each — see SELECTION_RULE_TEXT
    and the module docstring's bias note. Nothing here consults an RNG, a clock, a filesystem or a
    hash, so the same pooled artifact and the same band give the same frames forever. This function
    never sees the band: it is handed whatever ``tail_candidates`` selected, so adding an upper
    limit to the band changed nothing about how the budget is handed out.
    """
    budget = int(budget)
    if budget <= 0:
        raise TailLookError(f"--max-frames must be >= 1; got {budget}.")
    episodes = list(candidates)
    quotas = dict.fromkeys(episodes, 0)
    remaining = budget
    # One pass hands out one frame per episode; repeated until the budget is spent or every
    # candidate has been claimed. `progress` is what stops the loop when every episode is full.
    while remaining > 0:
        progress = False
        for episode in episodes:
            if remaining <= 0:
                break
            if quotas[episode] < len(candidates[episode]):
                quotas[episode] += 1
                remaining -= 1
                progress = True
        if not progress:
            break

    selected: list[dict] = []
    for episode in episodes:
        quota = quotas[episode]
        if not quota:
            continue
        hits = candidates[episode]
        for position in even_stride_indices(len(hits), quota):
            frame_index, fraction = hits[position]
            selected.append({
                "episode": episode,
                "frame_index": int(frame_index),
                "recorded_fraction": float(fraction),
            })
    return selected


# --------------------------------------------------------------------------------------------
# the picture
# --------------------------------------------------------------------------------------------


def overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The frame with the mask tinted AND outlined, at full resolution.

    Tinted so its extent is readable and outlined so a thin sliver survives the 2x downsample the
    contact sheets do — ``audit_apple_masks.composite``'s reasoning, with this audit's own colour.
    """
    out = np.asarray(rgb).astype(np.float32).copy()
    mask = np.asarray(mask, dtype=bool)
    if mask.any():
        out[mask] = out[mask] * 0.62 + np.asarray(COLOR_MASK, dtype=np.float32) * 0.38
    arr = np.clip(out, 0, 255).astype(np.uint8)
    if mask.any():
        arr[aam.boundary(mask, thickness=2)] = COLOR_MASK
    return arr


def caption_lines(record: dict, *, tolerance: float) -> list[str]:
    """What the tile says under the picture. BOTH fractions, always, and the delta.

    The recomputed number is not a decoration: it is the only thing that ties this RTX 5090 render
    to the H200 measurement the bound would be argued from.
    """
    delta = float(record["delta"])
    lines = [
        f"{record['episode']} / frame {int(record['frame_index']):05d}",
        f"recorded={float(record['recorded_fraction']):.6f}  "
        f"recomputed={float(record['recomputed_fraction']):.6f}",
        f"delta={delta:+.6f}",
    ]
    if record.get("mismatch"):
        lines.append(f"MISMATCH > {float(tolerance):g}: re-render disagrees with the measurement")
    return lines


def band_text(threshold: float, max_fraction: float | None) -> str:
    """The band as it is printed on a sheet. Says both edges when there are two.

    Neither edge is a bound; this is the selection window the tiles were drawn from.
    """
    if max_fraction is None:
        return f"fraction >= {float(threshold):.6f}"
    return f"{float(threshold):.6f} <= fraction <= {float(max_fraction):.6f}"


def sheet_title(index: int, threshold: float, count: int,
                max_fraction: float | None = DEFAULT_MAX_FRACTION) -> str:
    return (f"PR-08 V13 3.2 AREA BAND LOOK | sheet {index:02d} | {count} tiles | "
            f"{band_text(threshold, max_fraction)} | {SHEET_LEGEND}")


# --------------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------------


def _git_commit(root: pathlib.Path) -> str | None:
    """The commit this RENDER ran from, recorded beside the pooled artifact's own, never merged."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except Exception:  # noqa: BLE001 — provenance is recorded or recorded absent, never guessed
        return None
    value = out.stdout.strip()
    return value or None


def _torch_runtime() -> dict:
    """What actually ran the masker, so the hardware caveat is checkable and not just asserted."""
    record: dict[str, Any] = {}
    try:
        import torch

        record["torch"] = str(torch.__version__)
        record["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            record["gpu"] = str(torch.cuda.get_device_name(0))
            record["cuda"] = str(torch.version.cuda)
    except Exception as exc:  # noqa: BLE001
        record["torch"] = f"unavailable: {type(exc).__name__}: {exc}"
    return record


# --------------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------------


def _manifest_episodes(manifest: pathlib.Path) -> dict[str, dict]:
    payload = json.loads(pathlib.Path(manifest).read_text(encoding="utf-8"))
    return {str(entry.get("id")): entry for entry in payload.get("episodes") or ()}


def _pooled_frame_counts(pooled: dict) -> dict[str, int]:
    return {str(r.get("episode")): int(r.get("n_frames") or 0) for r in pooled["per_episode"]}


def render(
    selected: Sequence[dict],
    *,
    manifest: pathlib.Path,
    pooled: dict,
    masker: Any,
    decode_clip: Any,
    tolerance: float,
    full_frames_dir: pathlib.Path | None,
) -> list[dict]:
    """Decode, re-mask, recompute and overlay every selected frame, episode by episode.

    One decode per episode, because ``decode_clip`` returns the whole clip and that is the same call
    ``measure_source_mask_area`` made — the frame at index ``i`` here is the frame whose fraction is
    at ``area_fractions[i]`` there, or the run refuses.
    """
    entries = _manifest_episodes(manifest)
    counts = _pooled_frame_counts(pooled)
    by_episode: "OrderedDict[str, list[dict]]" = OrderedDict()
    for record in selected:
        by_episode.setdefault(record["episode"], []).append(dict(record))

    out: list[dict] = []
    for episode, records in by_episode.items():
        entry = entries.get(episode)
        if entry is None:
            raise TailLookError(
                f"{manifest} does not list {episode}, which the pooled artifact measured. The "
                "sheets would be showing\n       a different corpus than the distribution."
            )
        video = pathlib.Path(manifest).parent / str(entry["video"])
        frames = decode_clip(video)
        if int(frames.shape[0]) != counts.get(episode):
            raise TailLookError(
                f"{video} decodes {int(frames.shape[0])} frames; the pooled artifact measured "
                f"{counts.get(episode)} for {episode}.\n"
                "       area_fractions is indexed by frame index, so a tile drawn from this decode "
                "would not be the frame\n"
                "       that produced the number. Refused rather than rendered."
            )
        for record in records:
            index = int(record["frame_index"])
            rgb = frames[index]
            mask = np.asarray(masker.mask(rgb), dtype=bool)
            recomputed = float(np.count_nonzero(mask)) / float(mask.size)
            delta = recomputed - float(record["recorded_fraction"])
            record["recomputed_fraction"] = recomputed
            record["delta"] = float(delta)
            record["mismatch"] = bool(abs(delta) > float(tolerance))
            record["episode_index"] = int(
                next((r.get("episode_index") for r in pooled["per_episode"]
                      if str(r.get("episode")) == episode), -1)
            )
            shot = overlay(rgb, mask)
            lines = caption_lines(record, tolerance=tolerance)
            if full_frames_dir is not None:
                png = full_frames_dir / f"{episode}-{index:05d}.png"
                aam.captioned(shot, lines, font_size=13,
                              flagged=record["mismatch"]).save(png)
                record["full_frame"] = png.name
            record["_tile"] = aam.captioned(shot[::2, ::2], lines, font_size=10,
                                            flagged=record["mismatch"])
            out.append(record)
    return out


def write_sheets(
    rendered: Sequence[dict],
    sheets_dir: pathlib.Path,
    *,
    threshold: float,
    per_sheet: int,
    cols: int,
    max_fraction: float | None = DEFAULT_MAX_FRACTION,
) -> list[str]:
    """The contact sheets. Every header carries the whole band, not just its lower edge."""
    names: list[str] = []
    for start in range(0, len(rendered), per_sheet):
        chunk = list(rendered[start:start + per_sheet])
        number = start // per_sheet
        name = f"area-tail-{number:02d}.png"
        sheet = aam.contact_sheet([r["_tile"] for r in chunk],
                                  sheet_title(number, threshold, len(chunk), max_fraction),
                                  cols=cols)
        sheet.save(sheets_dir / name)
        for record in chunk:
            record["sheet"] = name
        names.append(name)
    return names


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Contact sheets of a band of the pooled robot-mask area distribution "
                    "(PR-08 V13 §3.2 evidence). The band is a selection window over the measured "
                    "distribution; neither of its edges is a bound or a candidate bound. Writes no "
                    "bound and discharges nothing.",
    )
    parser.add_argument("--pooled", type=pathlib.Path,
                        default=pathlib.Path("runs/pr08-robot-mask-area/POOLED.json"))
    parser.add_argument("--corpus", type=pathlib.Path,
                        default=pathlib.Path("/home/humanoid/wam-t041/"
                                             "pr08-apple-640x480-h264-lossless"))
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("runs/pr08-area-tail-look"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_TAIL_EDGE,
                        help="lower edge of the band, INCLUSIVE: select frames whose RECORDED "
                             "fraction is at or above this. A measured edge; not a bound.")
    parser.add_argument("--max-fraction", type=float, default=DEFAULT_MAX_FRACTION,
                        help="upper edge of the band, INCLUSIVE. Default: none, leaving the band "
                             "open at the top, which is the upper tail. A measured edge; not a "
                             "bound and not a candidate bound.")
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--sheet-tiles", type=int, default=12)
    parser.add_argument("--sheet-cols", type=int, default=4)
    parser.add_argument("--mismatch-tolerance", type=float, default=0.01)
    parser.add_argument("--no-full-frames", action="store_true",
                        help="skip the full-resolution per-frame PNGs; sheets only.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    repo = pathlib.Path(__file__).resolve().parent.parent

    try:
        pooled = load_pooled(args.pooled)
        manifest = pathlib.Path(args.corpus) / "manifest.json"
        if not manifest.is_file():
            raise TailLookError(f"no such corpus manifest: {manifest}")

        candidates = tail_candidates(pooled, args.threshold, args.max_fraction)
        if not candidates:
            raise TailLookError(
                f"no frame in {args.pooled} is inside the band "
                f"{band_text(args.threshold, args.max_fraction)}. "
                "There is nothing to look at in this band."
            )
        selected = select_frames(candidates, args.max_frames)

        out_dir = pathlib.Path(args.out)
        sheets_dir = out_dir / "sheets"
        frames_dir = out_dir / "frames"
        for directory in (out_dir, sheets_dir):
            directory.mkdir(parents=True, exist_ok=True)
        full_frames_dir = None
        if not args.no_full_frames:
            frames_dir.mkdir(parents=True, exist_ok=True)
            full_frames_dir = frames_dir

        import robot_composite as rc  # noqa: PLC0415 — torch/cv2 stay out of import time

        masker = rc.build_masker()
        masker.preflight()

        rendered = render(
            selected,
            manifest=manifest,
            pooled=pooled,
            masker=masker,
            decode_clip=rc.decode_clip,
            tolerance=args.mismatch_tolerance,
            full_frames_dir=full_frames_dir,
        )
        sheets = write_sheets(rendered, sheets_dir, threshold=args.threshold,
                              per_sheet=args.sheet_tiles, cols=args.sheet_cols,
                              max_fraction=args.max_fraction)

        frames_block = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rendered]
        mismatches = [r for r in frames_block if r.get("mismatch")]
        total_candidates = sum(len(v) for v in candidates.values())
        # Counted independently of the band's upper edge, so the open-tail population stays
        # readable from an artifact that rendered a closed band.
        at_or_above = sum(
            1
            for record in pooled.get("per_episode") or ()
            for f in (record.get("area_fractions") or ())
            if float(f) >= float(args.threshold)
        )

        artifact = {
            "schema": SCHEMA,
            "produced_by": "scripts/render_area_tail_sheet.py",
            "produced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rule": "T40_RULE_V13 §3.2 — 'whether those frames were looked at, and what they were'",
            "writes_a_bound": False,
            "not_a_discharge": NOT_A_DISCHARGE,
            "what_this_is_for": WHAT_THIS_IS_FOR,
            "the_distinction": THE_DISTINCTION,
            "records_no_reviewer_confirmation": (
                "There is no field here in which any reviewer confirmation could be recorded, and "
                "this process is not a reviewer."
            ),
            "hardware_caveat": HARDWARE_CAVEAT,
            "band": {
                "lower_inclusive": float(args.threshold),
                "upper_inclusive": (None if args.max_fraction is None
                                    else float(args.max_fraction)),
                "open_at_the_top": args.max_fraction is None,
                "selection": band_text(args.threshold, args.max_fraction),
                "note": BAND_NOTE,
                "why_a_band": WHY_A_BAND,
                "neither_edge_is_a_bound": (
                    "Neither the lower edge nor the upper edge is a bound, and neither is a "
                    "candidate bound. They select which frames were rendered and nothing else."
                ),
            },
            "threshold": {
                "value": float(args.threshold),
                "default": float(DEFAULT_TAIL_EDGE),
                "source": "runs/pr08-robot-mask-area/AREA_GAP_ANALYSIS.json "
                          "candidate_gaps[widest].tail_edge_above",
                "note": THRESHOLD_NOTE,
                "max_fraction": (None if args.max_fraction is None
                                 else float(args.max_fraction)),
                "max_fraction_default": DEFAULT_MAX_FRACTION,
                "max_fraction_note": BAND_NOTE,
            },
            "selection_rule": {
                "rule": SELECTION_RULE_TEXT,
                "bias": SELECTION_BIAS_TEXT,
                "rng": "none. No RNG, no clock, no hash and no filesystem order enters the "
                       "selection; it is a pure function of the pooled artifact, the threshold and "
                       "max_frames.",
                "max_frames": int(args.max_frames),
                "band": band_text(args.threshold, args.max_fraction),
                "reproduce": (
                    "load_pooled -> tail_candidates(pooled, threshold, max_fraction) -> "
                    "select_frames(candidates, max_frames) in scripts/render_area_tail_sheet.py. "
                    "The frames[] list below is the output of that composition."
                ),
            },
            "source": {
                "pooled_artifact": str(args.pooled),
                "git_commit": pooled.get("git_commit"),
                "source_manifest_sha256": pooled.get("source_manifest_sha256"),
                "prompt": pooled.get("prompt"),
                "estimator": pooled.get("estimator"),
                "corpus": str(args.corpus),
                "corpus_manifest": str(manifest),
            },
            "render": {
                "git_commit": _git_commit(repo),
                "git_commit_note": "the commit THIS RENDER ran from, recorded beside the pooled "
                                   "artifact's own commit and never merged with it",
                "runtime": _torch_runtime(),
                "masker": {
                    "class": "scripts/robot_composite.py Sam2RobotMasker (unmodified, committed "
                             "operating point, committed ROBOT_TEXT_PROMPT)",
                    "provenance": masker.provenance(),
                    "filter_record": masker.filter_record(),
                },
                "decode_path": "robot_composite.decode_clip — the same function "
                               "measure_source_mask_area used, per "
                               "cluster/discoverer/106_measure_robot_mask_area.sbatch",
            },
            "population": {
                "threshold": float(args.threshold),
                "max_fraction": (None if args.max_fraction is None
                                 else float(args.max_fraction)),
                "band": band_text(args.threshold, args.max_fraction),
                "frames_in_band": int(total_candidates),
                "frames_at_or_above_threshold": int(at_or_above),
                "episodes_with_tail_frames": len(candidates),
                "episodes_sampled": len({r["episode"] for r in frames_block}),
                "episodes_not_sampled": len(candidates) - len({r["episode"]
                                                               for r in frames_block}),
                "frames_rendered": len(frames_block),
            },
            "mismatch": {
                "tolerance": float(args.mismatch_tolerance),
                "definition": "abs(recorded_fraction - recomputed_fraction) > tolerance",
                "count": len(mismatches),
                "frames": [{"episode": m["episode"], "frame_index": m["frame_index"],
                            "recorded_fraction": m["recorded_fraction"],
                            "recomputed_fraction": m["recomputed_fraction"],
                            "delta": m["delta"]} for m in mismatches],
                "note": "This count belongs in any rationale that quotes these sheets. It is what "
                        "makes a tile evidence about the mask that produced the number rather "
                        "than about a differently-executed mask from the same frame index.",
            },
            "sheets": [f"sheets/{name}" for name in sheets],
            "legend": LEGEND,
            "sheet_header_legend": SHEET_LEGEND,
            "full_frames": (None if full_frames_dir is None else "frames/"),
            "frames": frames_block,
        }
        (out_dir / "TAIL_SAMPLE.json").write_text(
            json.dumps(artifact, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    except TailLookError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    print(f"=== rendered  {len(frames_block)} frame(s) from "
          f"{artifact['population']['episodes_sampled']} episode(s)")
    print(f"=== band      {band_text(args.threshold, args.max_fraction)} "
          "(a selection window; neither edge is a bound)")
    print(f"=== population {total_candidates} frame(s) in the band, in "
          f"{len(candidates)} episode(s)")
    print(f"=== mismatch  {len(mismatches)} frame(s) beyond {args.mismatch_tolerance}")
    for name in sheets:
        print(f"=== sheet     {sheets_dir / name}")
    print(f"=== artifact  {out_dir / 'TAIL_SAMPLE.json'}")
    print("=== this writes no bound, proposes none, and discharges nothing.")
    print("=== neither edge of the band is a bound or a candidate bound.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
