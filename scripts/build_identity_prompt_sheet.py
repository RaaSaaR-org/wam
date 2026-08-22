#!/usr/bin/env python3
"""T-040 / ``T40-TODO-01-identity-prompt-provenance`` — the evidence harness for arm C's prompt.

    .venv/bin/python scripts/build_identity_prompt_sheet.py build-sheet \\
        --manifest <SOURCE>/manifest.json --out runs/t040-identity-prompt/
    #   ... a human or a VLM fills the blank `verdict` field in sheet.jsonl ...
    .venv/bin/python scripts/build_identity_prompt_sheet.py verdict \\
        --sheet runs/t040-identity-prompt/
    # -> configs/transfer25/pr08_identity_prompt_evidence.json (+ .sha256), and a TOML fragment

WHAT IS ACTUALLY OPEN. ``configs/transfer25/styles.toml`` carries one OPEN blocking TODO, and this
script is its `evidence_required`, which reads in full: *"the sampled episode ids, the sample size,
and the per-episode verdicts"*. The claim it guards is that ``[identity_style].prompt`` was
assembled from a T-041 machine caption of **one** clip (``episode_000135_clip000``, see
``[source].caption_provenance``) and is applied unchanged to all 402 episodes, and that nobody has
checked it describes the other 401.

WHY THAT BLOCKS 4 020 CLIPS RATHER THAN BEING A TIDINESS ITEM. Arm C is the
generator-fingerprint control, and PR-08 §5 defines it as the arm "whose style prompt *is* the
source's own appearance". A control is only a control if that is true. If the caption is wrong for
most episodes — a different cloth, a different apple, different lighting — then arm C is a weak
RESTYLE and not an identity pass: it carries some of arm B's diversity, B − C understates the
diversity effect, and the headline's attribution between "diversity helped" and "generator
fingerprint" breaks silently, with every clip looking perfect. The frame match [volume] enforces
does not help, because a frame-matched control with the wrong prompt is still the wrong control.

THIS SCRIPT IS THE BUILD-SHEET HALF, AND IT DOES NOT JUDGE
----------------------------------------------------------
The split is `build-sheet` / judge / `verdict`, the same three steps PR-09 §8 item 4 registered for
T-041, and the middle step is **deliberately not implemented here and no judge is named anywhere in
this file**. That is not modesty about scope. T-041's entire run came back **VOID on G0b** because
the VLM judge could not clear its own 20/20 calibration set: forensics over the recorded
``scores.jsonl`` found it had answered the literal string ``"NO"`` to all 80 items — a constant
classifier, zero abstentions, ten "correct" calibration answers earned purely by the negatives
being NO-labelled. Every downstream number in that run (``base_failures: 30``,
``G0a_defect_present: true``, ``b = c = 0``) was vacuous rather than wrong. Presuming a judge here,
before anyone has shown an instrument that can answer this question, would repeat that failure
exactly — and this time the artifact it corrupts is a *pre-commitment* that licenses 4 020 clips.

So the sheet is **judge-agnostic**: one row per sampled episode, a frame path, the committed prompt
verbatim, and a blank ``verdict``. A person filling it in by looking at forty frames and a VLM
filling it in over the same forty frames write into the identical field, and ``verdict`` applies the
identical rule to whichever one comes back. Choosing an instrument, and showing it can clear a
calibration set before it is trusted, is a separate decision that this script deliberately leaves
to whoever makes it.

WHAT `verdict` REFUSES TO DO
----------------------------
**It will not emit an overall pass.** There is no threshold on the mismatch rate anywhere in this
file, because the TODO's own `action` says an inconstant appearance IS the finding: *"If the
appearance is not constant across the 402, that is the finding and arm C needs a per-episode
identity prompt rather than one shared string."* A script that folded forty verdicts into PASS/FAIL
would be coining the very threshold that decision turns on, after the numbers existed. It reports
the counts and it reports the disagreements, and the reader decides.

**It will not emit a verdict from a partially-filled sheet.** A blank row is not a `match`, and the
average over the filled rows is not the sample the seed selected — it is a self-selected subset of
it, biased in an unknown direction by whatever made a filler skip those rows. Any blank, any token
outside the fixed vocabulary, any `mismatch` with no axis named: refusal, naming every offending
row, writing nothing.

**It will not emit a verdict from a sheet that is not the sample that was drawn.** A DELETED row is
not a blank row, and deleting is the easier spelling for any tool that round-trips a JSONL: without
a check it slips past the refusal above and the artifact reports the survivors as the sample,
gate-qualified, with the strata it lost invisible. So the draw is pinned when the sheet is built —
``sampled_episodes``, ``sample_size``, and a ``sheet_id`` digest over seed, size, fraction, ids and
prompt that is carried on the meta AND on every row — and the rows are checked against it when the
sheet is read. Deletion, duplication and substitution are all refusals; a meta edited to match
doctored rows no longer hashes to the id its rows carry.

**It will not silently accept a sheet filled against different pixels.** Every row carries the
sha256 of the frame it names. If a frame on disk no longer matches, the verdict would describe
frames nobody looked at, so that is fatal; if a frame is simply gone the verdict cannot be
confirmed against anything and the run is stamped not gate-qualified rather than passed. A row
carrying one of the pair and not the other did not come from ``build-sheet``, and it is refused
rather than skipped: a frame with no recorded digest is a frame that check cannot run on, and the
skip would be invisible in the artifact.

**It will not infer gate qualification from a missing field.** The build stamps ``gate_qualified``
and its reasons; the verdict step READS both, refuses a meta that carries neither, and refuses one
whose stamp and reasons disagree. Deriving qualification from an empty reasons list made ``true``
the default for any meta that lost the key — including a meta whose own ``gate_qualified`` was
``false``.

**It will not quote a prompt nobody committed.** The sheet records ``[identity_style].prompt``
verbatim plus the partition's content hash. If the committed prompt has changed since the sheet was
built, the evidence is about a string that is no longer arm C's, and that is fatal rather than a
note in passing.

**It will not let a smoke run become the evidence.** ``--sample-size`` below the floor,
``--skip-frames``, a frame that failed to extract, or a manifest holding fewer episodes than the
partition's committed 402 all stamp ``gate_qualified: false``, record the reason, and exit 3 — the
same rule, for the same reason, as ``measure_geom_tol.py``'s ``--limit``. The artifact is still
written, because "we tried and this is what came out" is a record.

WHAT `gate_qualified` MEANS HERE, AND WHAT IT DOES NOT
------------------------------------------------------
It is a statement about the **admissibility of the evidence**, never about arm C. ``true`` means a
full sample was drawn under a recorded seed, every frame was extracted and still matches, and every
row was filled with a legal token. A run can be gate-qualified and report thirty-nine mismatches;
that is a gate-qualified measurement of a broken control. Reading ``gate_qualified: true`` as "arm
C is fine" would be the same category error as reading T-041's ``calibration_correct: 10/20`` as
half a pass.

THE SAMPLE, AND WHY IT IS NOT A PLAIN RANDOM ONE
-------------------------------------------------
The TODO asks for "a random sample of episodes **spanning the corpus**", and those two words are
the whole design constraint. 402 episodes recorded in one order carry unknown session structure —
a tablecloth straightened between sessions, a lamp moved, an afternoon of daylight — and the
question being asked is precisely whether appearance is constant ACROSS that structure. A plain
uniform draw of 40 from 402 clusters often enough to matter: it can leave a run of a hundred
consecutive episodes untouched, and a sample that accidentally covers one recording session answers
a different question than the one asked, while looking exactly like the right answer.

So: **stratified systematic sampling**. The episode ids are sorted, split into ``N`` contiguous
strata of near-equal size, and ONE episode is drawn uniformly at random from each. Every stratum is
represented by construction, so no gap longer than two strata can exist, while the draw inside each
stratum stays random. The alternative that also spans — a fixed stride, every ``402/N``-th episode —
was rejected: a deterministic lattice aliases with any periodicity in the recording order (sessions
of ten, a re-grip every fifth episode) and would then sample the same within-session position every
time, which is a systematically unrepresentative sample that no amount of re-running would reveal.

Sorting is on the episode **id**, not on manifest order: the id carries the recording index, that
index is the axis being spanned, and a manifest written in a different order would otherwise change
the sample without changing the seed.

``--sample-size`` is part of the sample's identity, not a knob: the strata are derived from it, so
40 and 41 select two different sets under the same seed. Both it and the seed go into the sheet and
into the evidence artifact, and rebuilding a sheet into a directory that already holds one under a
different seed is refused unless ``--allow-reseed`` is passed — re-drawing after seeing verdicts is
sample-shopping, and it should cost a flag that shows up in the shell history. A sheet that already
carries ANSWERS is refused outright, flag or no flag and identical parameters or not: that flag's
own text says "if the first draw was never filled", and rebuilding over a filled sheet would
replace forty verdicts with forty blanks while looking exactly like a first draw.

WHAT 40 BUYS, STATED SO IT CANNOT BE OVER-READ. Forty is the default because zero mismatches in 40
bounds the mismatch rate at roughly 3/40 = 7.5 % with 95 % confidence (the rule of three; the
finite-population correction at 40-of-402 makes that bound conservative). In episodes, a clean
sweep of the sample is still consistent with ~30 of the 402 disagreeing. **This sample can detect
inconstancy; it cannot certify constancy**, and the TODO asks for the former. If the decision needs
the latter, the answer is a census, not a bigger sample.

THE FRAME, AND ITS ONE HONEST WEAKNESS
---------------------------------------
One frame per episode, at a fixed fraction through the clip (default 0.10), chosen a priori and
recorded. Early rather than middle or late because the prompt asserts five things at once — apple,
cloth, background, lighting, and the white plate — and all five are unoccluded only before the
transfer: after the grasp the apple is inside the Dex3 hand, and after the release it is on the
plate with the hand over the region the prompt describes. Not frame 0, because the first frame of a
recording is the least representative one available (the scene may still be settling, exposure may
still be converging), and it is also the frame most likely to differ from its own clip.

The fraction is a flag, and it is recorded per row together with the resolved frame index and the
clip's frame count, so a different choice is one argument away and two runs under two fractions sit
side by side rather than overwriting each other. It is a flag with a range, though: a fraction
outside [0, 1] is refused rather than clamped (clamped, ``--frame-fraction 9`` would take the LAST
frame of every clip — the one instant the rubric calls unjudgeable — while the artifact went on
recording "early, before the transfer"), and a fraction that is legal but is not the early rule
fixed a priori (0 itself, or past 0.25) still runs, records what it actually did in ``frame_rule``,
and is stamped not gate-qualified. What it CANNOT see is a change that happens later
in the episode — a light switched mid-clip, a hand entering with a different sleeve. One frame per
episode is what the TODO's "sample-check" asks for and it is a sample within a sample; that
limitation is stamped into the artifact rather than left for a reader to work out.

The extraction decodes forward to the requested frame (``select=eq(n,IDX)``) instead of seeking.
``-ss`` lands on the nearest keyframe, so the frame that reached disk would not be the frame the
row claims, and every verdict would be about a slightly different instant than the record says.

EXIT STATUS
-----------
0   `build-sheet`: a full sample, every frame extracted. `verdict`: every row filled legally,
    coverage above ``--min-coverage``, frames still matching. In neither case a statement about
    arm C.
2   fatal: nothing usable was written (no manifest, wrong resolution, a missing video, an --out
    that cannot be honoured, a frame fraction outside [0, 1], a sheet that is unparseable,
    partially-filled or illegally-filled, rows deleted from / duplicated in / substituted into the
    drawn sample, a meta that does not hash to its own sheet_id or carries no gate stamp, a frame
    whose bytes changed or whose digest was removed, a prompt that no longer matches the committed
    one, an existing sheet that already carries answers).
3   produced, but MUST NOT be quoted as the TODO's evidence — a partial sample, skipped or failed
    frames, a manifest that is not the 402-episode corpus, missing frames at verdict time, a frame
    fraction outside the early rule, or coverage below the floor. The artifact is still written and
    says which.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import check_style_partition as csp  # noqa: E402
from measure_geom_tol import _git_commit, write_artifact  # noqa: E402

# The manifest reader the GENERATOR uses, imported rather than re-implemented. The sheet has to be
# drawn from exactly the corpus `97` will restyle — including its refusal of anything that is not
# 640x480, because the 120x160 converted tree has a perfectly valid manifest that would sail
# through a generic check while being a corpus PR-08 §3 forbids. A second reader here would be a
# second chance to disagree with the driver about what the corpus is.
from restyle_transfer25 import DriverError, load_manifest  # noqa: E402

SHEET_SCHEMA = "wam.identity_prompt_sheet/1"
EVIDENCE_SCHEMA = "wam.identity_prompt_evidence/1"
WRITEUP = "docs/preregistration/PR-08-photoreal-augmentation.md"
TODO_ID = "T40-TODO-01-identity-prompt-provenance"

DEFAULT_STYLES = _REPO_ROOT / "configs" / "transfer25" / "styles.toml"

#: The COMMITTED evidence artifact. Tracked, and anchored to the repository root rather than to the
#: caller's CWD, for the reason ``measure_geom_tol.py`` spells out: the TODO closes by putting this
#: evidence into a committed file before the first identity clip, and ``runs/`` is gitignored, so an
#: artifact written there can never be the pre-commitment. The FRAMES belong under ``runs/`` — they
#: are bulk, and the evidence names them by digest rather than carrying them.
DEFAULT_EVIDENCE_OUT_REL = "configs/transfer25/pr08_identity_prompt_evidence.json"
DEFAULT_EVIDENCE_OUT = _REPO_ROOT / DEFAULT_EVIDENCE_OUT_REL

#: Arbitrary, fixed before any frame was looked at, and recorded in every artifact. It is NOT one of
#: the 7001-7015 generation seeds in ``[seed_schedule]``: those identify a clip's initial latent and
#: reusing one here would put two unrelated meanings on one number in the same experiment.
DEFAULT_SAMPLE_SEED = 40001

#: See the module docstring for what 40 buys and what it does not. Below this, the run is a smoke
#: test of the harness and is stamped as one.
DEFAULT_SAMPLE_SIZE = 40
MIN_GATE_QUALIFIED_SAMPLE = 40

#: A tenth of the way in: past the first frame, before the reach closes on the apple.
DEFAULT_FRAME_FRACTION = 0.10

#: The a-priori frame rule is EARLY, and the artifact's ``frame_rule`` says so in words. A fraction
#: past this is a different question — after the grasp the apple is inside the Dex3 hand and after
#: the release the hand is over the region the prompt describes — so the run is still produced and
#: is stamped not gate-qualified, the same way ``--sample-size`` below the floor is. Exactly 0.0 is
#: disqualifying for the reason the module docstring gives: frame 0 is the least representative
#: frame of a recording. Neither is fatal; both are recorded.
MAX_GATE_QUALIFIED_FRAME_FRACTION = 0.25

#: Fraction of the sampled rows that must carry a decidable verdict (`match` or `mismatch`) before
#: the evidence is admissible. Borrowed from ``measure_geom_tol.py`` and ``measure_est_drift.py``
#: rather than coined here, and it means the same thing they mean: a threshold on how much of the
#: sample could be answered at all, never a threshold on the answer.
DEFAULT_MIN_COVERAGE = 0.90

EXIT_OK = 0
EXIT_FATAL = 2
EXIT_NOT_GATE_QUALIFIED = 3

#: The fixed vocabulary. Three values and no fourth, because "mostly matches" is the token that
#: would let a reader read a pass into a sheet nobody passed.
VERDICT_VALUES = ("match", "mismatch", "unsure")

#: Which clause of the prompt a `mismatch` disagrees with. The first four are T-040's allowed
#: variation axes, so a mismatch on one of them is the ordinary finding: arm C would need a
#: per-episode prompt on that axis. `plate` is listed separately and is NOT an allowed axis — the
#: prompt asserts a white round plate and every style ends by holding the plate fixed, so a plate
#: disagreement in the SOURCE is a different and worse finding than a different tablecloth.
MISMATCH_AXES = ("apple", "table", "background", "lighting", "plate", "other")

#: The question, fixed here so that a person and a model answer the SAME one. Writing a question is
#: not choosing a judge: no model, server, endpoint or API appears anywhere in this file, and see
#: the docstring for why that is deliberate rather than unfinished.
RUBRIC = """\
You are shown ONE frame taken from ONE episode of the source corpus, and the identity prompt that
was committed for that corpus. The prompt was written from a machine caption of a DIFFERENT episode
of the same corpus. The question is whether it also describes this one.

Read every clause of the prompt against the frame: the apple, the surface it rests on, the
background, the lighting, and the white plate. Ignore where things are and ignore what the robot is
doing — position and geometry are not what this prompt claims, and are not what is being checked.

Fill `verdict` with exactly one of:

  match     every clause of the prompt is true of this frame.
  mismatch  at least one clause is false. List which in `mismatched_axes`, using only the words
            apple, table, background, lighting, plate, other, and say in `notes` what you actually
            see instead.
  unsure    this frame cannot settle it (the hand occludes the apple, the frame is too dark to
            judge the cloth). `unsure` is an abstention, not a soft mismatch, and it is reported
            separately from both.

Leave `verdict` blank only if you did not look. A blank row makes the whole sheet unusable, by
design: the verdict step refuses a partially-filled sheet rather than averaging over the rows
someone happened to answer.\
"""


class SheetError(RuntimeError):
    """A refusal. Every one of these means the sheet or the verdict would claim something untrue."""


# --------------------------------------------------------------------------------------------
# the committed prompt
# --------------------------------------------------------------------------------------------


def read_identity_style(styles_path: Path) -> dict[str, Any]:
    """``[identity_style]`` plus the partition's content hash, read from the committed TOML.

    Read through ``check_style_partition`` so the prompt and the digest come from the one reader
    that defines what the partition hash is over (the document minus ``[hash]`` and ``[consumer]``).
    A local re-implementation would be a second definition of the digest, and the failure mode of
    two digest definitions is a recorded hash that identifies nothing — which is defect 3 in
    ``check_style_partition``'s own docstring.
    """
    try:
        doc = csp.load(styles_path)
    except csp.Failure as exc:
        raise SheetError(str(exc)) from exc

    style = doc.get("identity_style")
    if not isinstance(style, dict):
        raise SheetError(
            f"{styles_path} has no [identity_style] table. Arm C's prompt is never synthesised — "
            "see the file's own [consumer] note, where an earlier draft of `97` invented one and "
            "that was recorded as a defect because it would make the control unattributable."
        )
    prompt = style.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise SheetError(f"{styles_path}: [identity_style] carries no usable 'prompt'.")

    todo = next(
        (t for t in (doc.get("blocking_todos") or []) if t.get("id") == TODO_ID),
        None,
    )
    source = doc.get("source") or {}
    return {
        "prompt": prompt,
        "axes": {k: style.get(k) for k in ("apple", "table", "background", "lighting")},
        "style_id": style.get("id"),
        "repeats": style.get("repeats"),
        "partition_rule": doc.get("rule"),
        "partition_content_sha256": csp.content_hash(doc),
        "caption_provenance": source.get("caption_provenance"),
        "committed_episodes": source.get("episodes"),
        "todo_status": (todo or {}).get("status"),
        "todo_evidence_required": (todo or {}).get("evidence_required"),
    }


# --------------------------------------------------------------------------------------------
# the sample
# --------------------------------------------------------------------------------------------


def strata_bounds(n_episodes: int, sample_size: int) -> list[tuple[int, int]]:
    """``sample_size`` contiguous half-open index ranges covering ``0..n_episodes``.

    ``k*M//N`` rather than an equal split, because 402 does not divide by 40 and rounding each
    boundary independently is the only split that leaves no episode outside a stratum and no
    stratum empty. An empty stratum would silently shrink the sample below the size recorded in the
    artifact, which is one of the three things the TODO asks for.
    """
    if sample_size <= 0:
        raise SheetError("sample size must be positive.")
    if sample_size > n_episodes:
        raise SheetError(
            f"cannot draw {sample_size} episodes from a corpus of {n_episodes}: the sample would "
            "have to repeat an episode, and a repeated episode counted twice in the per-episode "
            "verdicts is not a sample of the corpus."
        )
    return [
        (k * n_episodes // sample_size, (k + 1) * n_episodes // sample_size)
        for k in range(sample_size)
    ]


def sample_episode_ids(episode_ids: list[str], sample_size: int, seed: int) -> list[str]:
    """One episode per stratum, drawn uniformly inside it. Deterministic in (ids, size, seed).

    Sorted first, and sorted on the id: the id carries the recording index, spanning that index is
    the entire point of stratifying, and taking the manifest's own order would mean a manifest
    rewritten in a different order silently produces a different sample under the same seed — a
    change of evidence with no change of any recorded parameter.
    """
    ordered = sorted(episode_ids)
    rng = random.Random(seed)
    return [ordered[rng.randrange(lo, hi)] for lo, hi in strata_bounds(len(ordered), sample_size)]


# --------------------------------------------------------------------------------------------
# the frame
# --------------------------------------------------------------------------------------------


def check_frame_fraction(fraction: float) -> float:
    """Refuse a fraction that is not a position inside the clip. Never clamps one into range.

    ``frame_index`` clamps, and a clamp is the wrong instrument for an out-of-range *request*:
    ``--frame-fraction 9.0`` would resolve to the LAST frame of every clip while the artifact's
    ``frame_rule`` went on saying the frame was taken early, before the transfer — forty verdicts
    about the one instant the rubric calls unjudgeable, recorded under a rule the run did not
    follow. The clamp stays, but only for what it was documented for: rounding inside a 1-frame
    clip. Anything outside [0, 1] is a request this harness cannot honour, so it refuses instead of
    honouring a different one. (``not (0 <= f <= 1)`` also catches NaN, which every comparison
    would otherwise wave through.)
    """
    try:
        f = float(fraction)
    except (TypeError, ValueError) as exc:
        raise SheetError(f"--frame-fraction {fraction!r} is not a number.") from exc
    if not 0.0 <= f <= 1.0:
        raise SheetError(
            f"--frame-fraction {f} is outside [0, 1]. A fraction is a position inside the clip; "
            "out of range it would be silently clamped to frame 0 or to the last frame, and the "
            "sheet would still record the a-priori rule ('early, before the transfer') that the "
            "run did not follow. Pass a fraction in [0, 1]."
        )
    return f


def sample_size_disqualification(sample_size: int) -> str | None:
    """Why a sample below the floor is a smoke run, or None. Written once, applied at both steps.

    Both ``build-sheet`` and ``verdict`` derive this, and they derive it from the same function so
    the two strings are identical and the verdict step's copy dedupes against the one it inherited.
    Two derivations of the same rule would be two chances to disagree about it, and the reason the
    verdict step derives it at all is that inheriting is not enough: emptying
    ``gate_disqualified_reasons`` in the meta and setting ``gate_qualified: true`` is internally
    consistent and does not change the sheet_id, so a five-episode smoke run could otherwise be
    hand-promoted into the gate-qualified evidence for 4 020 clips.
    """
    if sample_size >= MIN_GATE_QUALIFIED_SAMPLE:
        return None
    return (
        f"--sample-size {sample_size} < {MIN_GATE_QUALIFIED_SAMPLE}: a smaller sample is "
        "a smoke test of this harness. It cannot bound the mismatch rate at the 7.5% the "
        "default is chosen for, and nothing in the filled sheet would look any different."
    )


def frame_fraction_disqualification(fraction: float) -> str | None:
    """Why a legal-but-unregistered fraction costs gate qualification, or None."""
    if fraction == 0.0:
        return (
            "--frame-fraction 0.0 resolves to frame 0, and the first frame of a recording is the "
            "least representative one available (the scene may still be settling, the exposure "
            "may still be converging). The rows are written and can be looked at; they are not "
            "the sample this harness pre-registered."
        )
    if fraction > MAX_GATE_QUALIFIED_FRAME_FRACTION:
        return (
            f"--frame-fraction {fraction} > {MAX_GATE_QUALIFIED_FRAME_FRACTION}: the frame rule "
            "fixed before any frame was looked at is EARLY, because the apple, the cloth, the "
            "background, the lighting and the plate are all unoccluded only before the transfer. "
            "Later in the clip the apple is inside the hand or on the plate under it, so a "
            "mismatch and an occlusion are no longer distinguishable."
        )
    return None


def frame_rule_text(fraction: float) -> str:
    """The recorded rule, written from the fraction actually used rather than from the default.

    A fixed string here would keep asserting 'early, before the transfer' for a run taken at 0.9,
    which is the artifact recording a rule the run did not follow.
    """
    base = (f"one frame per episode at index round({fraction} * (n_frames - 1)), decoded forward "
            "(select=eq(n,IDX)), never seeked. ")
    if fraction == 0.0:
        return base + (
            "Frame 0 — NOT the pre-registered rule: the first frame of a recording is the least "
            "representative one available, and this run is stamped not gate-qualified for it."
        )
    if fraction <= MAX_GATE_QUALIFIED_FRAME_FRACTION:
        return base + (
            "Early rather than middle or late because the apple, the cloth, the background, the "
            "lighting and the plate are all unoccluded only before the transfer."
        )
    return base + (
        f"NOT the pre-registered rule, which is early (<= {MAX_GATE_QUALIFIED_FRAME_FRACTION}) "
        "because the apple, the cloth, the background, the lighting and the plate are all "
        "unoccluded only before the transfer. At this fraction the grasp or the release may "
        "already have happened, so this run is stamped not gate-qualified."
    )


def frame_index(n_frames: int, fraction: float) -> int:
    """The representative frame's 0-based index. Clamped, so a 1-frame clip resolves to 0.

    The fraction is checked first: the clamp exists for rounding, not to absorb a fraction outside
    [0, 1] into a frame at one end of the clip — see ``check_frame_fraction``.
    """
    if n_frames <= 0:
        raise SheetError("episode declares no frames.")
    f = check_frame_fraction(fraction)
    return max(0, min(n_frames - 1, int(round(f * (n_frames - 1)))))


def extract_frame(video: Path, index: int, out: Path, ffmpeg: str = "ffmpeg") -> None:
    """Decode forward to frame ``index`` and write it as PNG. Never seeks.

    ``-ss`` would be far faster and would land on the nearest keyframe, so the file on disk would
    not be the frame the sheet's row claims it is. Every verdict would then be about a slightly
    different instant than the record says, and nothing downstream could detect it — the PNG looks
    like a perfectly good frame of the right episode. A full forward decode of a few hundred frames
    of 640x480 is a price worth paying to keep the row honest.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-v", "error", "-nostdin", "-y",
        "-i", str(video),
        "-vf", f"select=eq(n\\,{index})",
        "-fps_mode", "passthrough",
        "-frames:v", "1",
        str(out),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise SheetError(
            f"{ffmpeg} not found. Pass --ffmpeg, or put ffmpeg's bin on PATH — on the cluster that "
            "is ${FFMPEG_PREFIX}/bin. Nothing was extracted; the sheet is not written from a "
            "guess about what the frames would have shown."
        ) from exc
    if proc.returncode != 0 or not out.is_file():
        raise SheetError(
            f"ffmpeg failed to extract frame {index} of {video}:\n{proc.stderr.strip()}"
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------
# sheet layout
# --------------------------------------------------------------------------------------------


def sheet_paths(target: Path) -> tuple[Path, Path]:
    """(sheet.jsonl, sheet_meta.json) from either the sheet directory or the .jsonl itself.

    Decided on the suffix rather than on ``is_dir()``, because at build time the directory does not
    exist yet and an existence test would silently resolve ``--out runs/x`` to the *file*
    ``runs/x`` — writing the rows over the directory the frames were about to go into.
    """
    if target.suffix == ".jsonl":
        return target, target.parent / "sheet_meta.json"
    return target / "sheet.jsonl", target / "sheet_meta.json"


def sheet_id(seed: int, sample_size: int, fraction: float, ids: list[str], prompt: str) -> str:
    """A digest of everything that decides WHICH sheet this is.

    Carried on the meta and on every row, so a set of filled rows cannot be joined to a meta they
    were not drawn under. Two sheets differing only in the seed are two different samples of the
    corpus, and pasting one's answers under the other's provenance would produce evidence for a
    sample nobody drew.
    """
    payload = json.dumps(
        {"seed": seed, "sample_size": sample_size, "frame_fraction": fraction,
         "episodes": ids, "prompt": prompt},
        sort_keys=True, ensure_ascii=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# --------------------------------------------------------------------------------------------
# build-sheet
# --------------------------------------------------------------------------------------------


def cmd_build_sheet(args: argparse.Namespace) -> int:
    # Before anything is decoded: --out is the directory that will hold sheet.jsonl, sheet_meta.json
    # and frames/. Handed a .jsonl, `sheet_paths` resolves the rows onto that path while the frames
    # go to `<path>/frames/`, which creates the same path as a DIRECTORY — the run then extracts
    # every frame and dies on IsADirectoryError writing the rows. Refuse the path it cannot honour
    # instead of honouring half of it. (`verdict --sheet` does take the .jsonl.)
    if args.out.suffix == ".jsonl":
        raise SheetError(
            f"--out {args.out} names a .jsonl file, but build-sheet writes a DIRECTORY holding "
            "sheet.jsonl, sheet_meta.json and frames/. Pass the directory. Only "
            "`verdict --sheet` accepts the .jsonl itself."
        )
    args.frame_fraction = check_frame_fraction(args.frame_fraction)
    identity = read_identity_style(args.styles)
    episodes = load_manifest(args.manifest)          # raises DriverError on anything not 640x480
    source_root = args.manifest.parent

    ids = sample_episode_ids(list(episodes), args.sample_size, args.seed)
    sid = sheet_id(args.seed, args.sample_size, args.frame_fraction, ids, identity["prompt"])

    sheet_file, meta_file = sheet_paths(args.out)
    _refuse_silent_reseed(sheet_file, meta_file, sid, args)
    # Frames of a previous draw are frames of episodes this sheet does not name. Left in place they
    # are forty-odd plausible PNGs of the right corpus sitting next to the ones under judgement,
    # with nothing in the directory saying which draw they belong to.
    stale = _clear_stale_frames(args.out / "frames", ids)

    rows: list[dict[str, Any]] = []
    frame_failures: list[str] = []
    for ep_id in ids:
        entry = episodes[ep_id]
        video = source_root / str(entry["video"])
        if not video.is_file():
            # Not a per-row failure: the manifest disagreeing with the tree means the sheet would
            # name pixels that are not there, and `97` would hit the same missing file. Refuse.
            raise SheetError(
                f"{ep_id}: {video} is missing, though {args.manifest} lists it. The manifest and "
                "the corpus tree do not describe one corpus; rebuild it with "
                "scripts/build_pr08_source.py rather than sampling around the gap."
            )
        n_frames = int(entry["frames"])
        idx = frame_index(n_frames, args.frame_fraction)
        frame = args.out / "frames" / f"{ep_id}.png"

        digest: str | None = None
        if args.skip_frames:
            frame_rel: str | None = None
        else:
            try:
                extract_frame(video, idx, frame, args.ffmpeg)
                digest = sha256_file(frame)
                frame_rel = str(frame)
            except SheetError as exc:
                # One unreadable clip must not cost the other 39 extractions, and it must not be
                # papered over either: the row is written with a null frame, the episode is named
                # in the artifact, and the run loses gate qualification.
                frame_failures.append(f"{ep_id}: {exc}")
                frame_rel = None

        rows.append({
            "sheet_id": sid,
            "episode": ep_id,
            "video": str(video),
            "n_frames": n_frames,
            "frame_index": idx,
            "frame_fraction": args.frame_fraction,
            "frame": frame_rel,
            "frame_sha256": digest,
            "prompt": identity["prompt"],
            # The three blank fields. `verdict` is the one the TODO's evidence_required names; the
            # other two are what turn a mismatch into something arm C can be repaired from, because
            # "it does not match" does not say which clause to make per-episode.
            "verdict": "",
            "mismatched_axes": [],
            "notes": "",
        })

    disq: list[str] = []
    floor_reason = sample_size_disqualification(args.sample_size)
    if floor_reason:
        disq.append(floor_reason)
    if args.skip_frames:
        disq.append(
            "--skip-frames: the sheet names no pixels, so nobody can fill it by looking. It "
            "exercises the sampling and the row schema and is not evidence about any episode."
        )
    if frame_failures:
        disq.append(
            f"{len(frame_failures)} frame(s) failed to extract, so those episodes cannot be "
            "judged and the sample is short of the size recorded here."
        )
    fraction_reason = frame_fraction_disqualification(args.frame_fraction)
    if fraction_reason:
        disq.append(fraction_reason)
    committed_n = identity.get("committed_episodes")
    if isinstance(committed_n, int) and len(episodes) != committed_n:
        disq.append(
            f"the manifest holds {len(episodes)} episodes; the committed partition's [source] "
            f"declares {committed_n}. A sample spanning a different corpus than the one arm C "
            "generates over says nothing about arm C."
        )

    meta = {
        "schema": SHEET_SCHEMA,
        "writeup": WRITEUP,
        "todo": TODO_ID,
        "evidence_required": identity["todo_evidence_required"],
        "todo_status_at_build": identity["todo_status"],
        "built_by": "scripts/build_identity_prompt_sheet.py build-sheet",
        "built_date": date.today().isoformat(),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),

        "sheet_id": sid,
        "sheet": str(sheet_file),
        "manifest": str(args.manifest),
        "n_episodes_in_manifest": len(episodes),
        "styles": str(args.styles),
        "partition_rule": identity["partition_rule"],
        "partition_content_sha256": identity["partition_content_sha256"],
        "identity_style_id": identity["style_id"],
        "identity_repeats": identity["repeats"],
        "prompt": identity["prompt"],
        "prompt_axes": identity["axes"],
        "caption_provenance": identity["caption_provenance"],

        "sample_seed": args.seed,
        "sample_size": args.sample_size,
        "sample_scheme": (
            "stratified-systematic/1 — episode ids sorted, split into sample_size contiguous "
            "strata of near-equal size (bounds k*M//N), one episode drawn uniformly at random "
            "inside each under sample_seed. Spans the recording order by construction; a fixed "
            "stride was rejected because it aliases with any periodicity in that order."
        ),
        "sample_strata": [
            {"stratum": k, "lo": lo, "hi": hi, "episode": ids[k]}
            for k, (lo, hi) in enumerate(strata_bounds(len(episodes), args.sample_size))
        ],
        "sampled_episodes": ids,

        "frame_fraction": args.frame_fraction,
        "frame_rule": frame_rule_text(args.frame_fraction),
        "frames_extracted": sum(1 for r in rows if r["frame"]),
        "frame_failures": frame_failures,
        "stale_frames_removed": stale,

        "rubric": RUBRIC,
        "verdict_values": list(VERDICT_VALUES),
        "mismatch_axes": list(MISMATCH_AXES),
        "judge": None,
        "judge_note": (
            "Deliberately unset and not chosen by this script. T-041 came back VOID because its "
            "VLM judge answered a constant 'NO' to all 80 items and could not clear its own 20/20 "
            "calibration set; presuming a judge here would repeat that failure into a "
            "pre-commitment that licenses 4020 clips. A human and a VLM fill this identical field "
            "and `verdict` applies the identical rule to either."
        ),

        "gate_qualified": not disq,
        "gate_disqualified_reasons": disq,
        "gate_qualified_scope": (
            "admissibility of the evidence, never a statement about arm C. A gate-qualified sheet "
            "can come back all mismatches — that is a gate-qualified measurement of a broken "
            "control."
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    sheet_file.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )
    meta_file.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"manifest    {args.manifest} ({len(episodes)} episodes)", file=sys.stderr)
    print(f"sample      {args.sample_size} of {len(episodes)}, seed {args.seed}, "
          f"stratified-systematic", file=sys.stderr)
    print(f"span        {ids[0]} .. {ids[-1]}", file=sys.stderr)
    print(f"frames      {meta['frames_extracted']}/{len(rows)} at fraction "
          f"{args.frame_fraction}", file=sys.stderr)
    if stale:
        print(f"cleared     {len(stale)} frame(s) of a previous, unfilled draw", file=sys.stderr)
    print(f"wrote       {sheet_file}", file=sys.stderr)
    print(f"            {meta_file}", file=sys.stderr)
    print(f"sheet_id    {sid}", file=sys.stderr)
    print("\nNEXT: a human or a VLM fills `verdict` (and `mismatched_axes` / `notes` on a "
          "mismatch)\n      in every row. No judge is named by this script, on purpose — see its "
          "docstring.\n      Then: build_identity_prompt_sheet.py verdict --sheet "
          f"{args.out}", file=sys.stderr)
    for reason in disq:
        print(f"\nNOT GATE-QUALIFIED: {reason}", file=sys.stderr)
    if disq:
        print("\n                    The sheet is written anyway — 'we tried and this is what came "
              "out' is a\n                    record — and the verdict step inherits every reason "
              "above.", file=sys.stderr)
    return EXIT_OK if not disq else EXIT_NOT_GATE_QUALIFIED


def _clear_stale_frames(frames_dir: Path, ids: list[str]) -> list[str]:
    """Delete ``frames/<episode>.png`` for episodes this draw does not name. Returns what went.

    Only reachable once ``_refuse_silent_reseed`` has established that no sheet in this directory
    carries an answer, so nothing being deleted here was ever judged.
    """
    if not frames_dir.is_dir():
        return []
    keep = set(ids)
    gone: list[str] = []
    for p in sorted(frames_dir.glob("*.png")):
        if p.stem not in keep:
            try:
                p.unlink()
            except OSError:
                continue
            gone.append(p.name)
    return gone


def _filled_rows(sheet_file: Path) -> list[str]:
    """Episodes in an existing sheet that already carry a verdict. Best effort, never raises."""
    if not sheet_file.is_file():
        return []
    filled: list[str] = []
    try:
        text = sheet_file.read_text()
    except OSError:
        return []
    for ln in text.splitlines():
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and str(r.get("verdict") or "").strip():
            filled.append(str(r.get("episode")))
    return filled


def _refuse_silent_reseed(sheet_file: Path, meta_file: Path, sid: str,
                          args: argparse.Namespace) -> None:
    """Rebuilding over an existing sheet under a different seed costs a flag.

    Drawing a second sample after seeing the first one's verdicts is sample-shopping, and it is
    invisible afterwards: the new sheet looks exactly like a first draw. Making it cost
    ``--allow-reseed`` does not prevent it — nothing here can — but it puts the decision in the
    shell history and in this refusal's text instead of nowhere.

    ``--allow-reseed``'s own message says "if the first draw was never filled", so that is now
    checked rather than asserted: a sheet carrying answers is never overwritten, with or without
    the flag, and not under an identical sheet_id either — a plain re-run of the same command would
    otherwise silently replace forty verdicts with forty blanks and look like a fresh build.
    """
    filled = _filled_rows(sheet_file)
    if filled:
        raise SheetError(
            f"{sheet_file} already carries {len(filled)} filled verdict(s) "
            f"({', '.join(filled[:6])}{' …' if len(filled) > 6 else ''}).\n"
            "       Rebuilding would overwrite answers somebody produced by looking at frames, and "
            "re-drawing after seeing verdicts is sample-shopping that leaves no trace in the "
            "result. Write the new sample to a different --out. --allow-reseed covers a first draw "
            "that was never filled; this one was."
        )
    if args.allow_reseed or not meta_file.is_file():
        return
    try:
        old = json.loads(meta_file.read_text())
    except (OSError, json.JSONDecodeError):
        return
    if old.get("sheet_id") == sid:
        return
    raise SheetError(
        f"{meta_file} already holds sheet {old.get('sheet_id')} drawn under seed "
        f"{old.get('sample_seed')} / size {old.get('sample_size')} / fraction "
        f"{old.get('frame_fraction')}, and this run would replace it with {sid} under seed "
        f"{args.seed} / size {args.sample_size} / fraction {args.frame_fraction}.\n"
        "       Re-drawing after seeing a sheet's verdicts is sample-shopping and leaves no trace "
        "in the result. Write the new sample to a different --out, or pass --allow-reseed if the "
        "first draw was never filled."
    )


# --------------------------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------------------------


def read_filled_sheet(sheet_file: Path, meta_file: Path) -> tuple[list[dict], dict]:
    if not sheet_file.is_file():
        raise SheetError(f"{sheet_file} missing — build-sheet writes it.")
    if not meta_file.is_file():
        raise SheetError(
            f"{meta_file} missing. The rows alone do not carry the seed, the scheme or the strata, "
            "and 'the sample size' is one of the three things the TODO asks for. Without the meta "
            "the filled rows are answers to an unrecorded question."
        )
    # Hand-editing this JSONL is the intended workflow, so malformed JSON is an expected input and
    # not an internal error: it exits FATAL naming the file and the line, the same way the missing
    # -key check below does, rather than as a traceback from json's own frames.
    try:
        meta = json.loads(meta_file.read_text())
    except json.JSONDecodeError as exc:
        raise SheetError(f"{meta_file} is not valid JSON ({exc}). It is hand-editable and it was "
                         "hand-edited into something no reader can parse.") from exc
    if not isinstance(meta, dict):
        raise SheetError(f"{meta_file} does not hold a JSON object.")

    rows = []
    for n, ln in enumerate(sheet_file.read_text().splitlines(), 1):
        if not ln.strip():
            continue
        try:
            row = json.loads(ln)
        except json.JSONDecodeError as exc:
            raise SheetError(
                f"{sheet_file}:{n} is not valid JSON ({exc}). One row per line, and the row is "
                "edited in place — a truncated or re-wrapped line is not a verdict."
            ) from exc
        if not isinstance(row, dict):
            raise SheetError(f"{sheet_file}:{n} is not a JSON object.")
        rows.append(row)
    if not rows:
        raise SheetError(f"{sheet_file} has no rows.")
    # A row missing a schema field would otherwise surface as a KeyError three functions later,
    # from a traceback that names neither the file nor the row. A filled sheet comes back from
    # whatever tool a filler used, so "it is the shape build-sheet wrote" is a check, not a given.
    for n, r in enumerate(rows, 1):
        missing = [k for k in ("sheet_id", "episode", "verdict") if k not in r]
        if missing:
            raise SheetError(
                f"{sheet_file}:{n} is missing {missing} — it is not a row this script wrote. "
                "Fill the sheet in place rather than regenerating it from another format."
            )

    strays = sorted({r.get("sheet_id") for r in rows} - {meta.get("sheet_id")})
    if strays:
        raise SheetError(
            f"{sheet_file} carries rows from sheet(s) {strays} but {meta_file} describes "
            f"{meta.get('sheet_id')}. Answers drawn under one sample cannot be reported under "
            "another sample's provenance."
        )
    check_sample_identity(rows, meta, sheet_file, meta_file)
    return rows, meta


def check_sample_identity(rows: list[dict], meta: dict, sheet_file: Path, meta_file: Path) -> None:
    """The rows must be EXACTLY the episodes the draw pinned — no deletion, no duplicate, no swap.

    This is the hole the blank-row refusal on its own leaves open, and it is the easier one to fall
    into: any tool that round-trips a JSONL can drop a line, and a *deleted* row is not a blank row.
    Without this check the verdict step re-derives ``sampled_episodes`` and ``sample_size`` from
    whatever rows are present, so five deleted rows produce a gate-qualified artifact reporting a
    35-episode sample with coverage 1.0 — self-consistent down to the rule-of-three note, five
    strata unrepresented, and nothing anywhere saying a sample of forty was drawn. That is the
    "self-selected subset of it, biased in an unknown direction by whatever made a filler skip
    those rows" the module docstring refuses, arriving by a different spelling.

    A duplicate is the same hole from the other side: ``sample_episode_ids`` refuses a repeated
    episode at draw time ("a repeated episode counted twice in the per-episode verdicts is not a
    sample of the corpus"), so the verdict step enforces the invariant its own draw step names.

    Order is not part of the identity — a filler may sort the file — but membership and multiplicity
    are, and ``sampled_episodes`` on the meta is where the draw recorded them.
    """
    drawn = meta.get("sampled_episodes")
    size = meta.get("sample_size")
    if not isinstance(drawn, list) or not drawn or not all(isinstance(e, str) for e in drawn):
        raise SheetError(
            f"{meta_file} carries no usable `sampled_episodes`, so there is nothing to check the "
            "rows against. 'the sampled episode ids' is the first of the three things the TODO's "
            "evidence_required names; re-deriving them from whichever rows survived would make "
            "the artifact describe the surviving rows and call them the sample."
        )
    if isinstance(size, bool) or not isinstance(size, int) or size != len(drawn):
        raise SheetError(
            f"{meta_file}: `sample_size` {size!r} does not match its own {len(drawn)} "
            "`sampled_episodes`. The meta contradicts itself about what was drawn."
        )

    present = [str(r["episode"]) for r in rows]
    seen: set[str] = set()
    repeated: set[str] = set()
    for ep in present:
        (repeated if ep in seen else seen).add(ep)
    dupes = sorted(repeated)
    if dupes:
        raise SheetError(
            f"{sheet_file} lists {len(dupes)} episode(s) more than once: {', '.join(dupes[:12])}\n"
            "       A repeated episode counted twice in the per-episode verdicts is not a sample "
            "of the corpus — the draw refuses to produce one, so the verdict refuses to report "
            "one. It would inflate `sample_size` past the rows that exist and deflate coverage "
            "against a denominator nobody drew."
        )

    drawn_set, present_set = set(drawn), set(present)
    lost = [e for e in drawn if e not in present_set]
    extra = sorted(present_set - drawn_set)
    if lost or extra:
        parts = []
        if lost:
            parts.append(f"{len(lost)} drawn episode(s) have no row: "
                         f"{', '.join(lost[:12])}{' …' if len(lost) > 12 else ''}")
        if extra:
            parts.append(f"{len(extra)} row(s) name an episode that was never drawn: "
                         f"{', '.join(extra[:12])}{' …' if len(extra) > 12 else ''}")
        raise SheetError(
            f"{sheet_file} is not the sheet {meta_file} describes — " + "; ".join(parts) + ".\n"
            "       The sample is pinned when it is drawn and checked when it is read, because a "
            "deleted row is not a blank row: it would shrink the sample past the partial-fill "
            "refusal and the artifact would report the survivors as the sample, gate-qualified, "
            "with every stratum it lost invisible. Fill the sheet in place."
        )
    if len(present) != size:
        raise SheetError(
            f"{sheet_file} holds {len(present)} rows; {meta_file} pinned a sample of {size}."
        )


def check_fill(rows: list[dict]) -> None:
    """Refuse anything that is not a complete, legal fill. Never repairs, never defaults."""
    blank = [r["episode"] for r in rows if not str(r.get("verdict") or "").strip()]
    if blank:
        raise SheetError(
            f"{len(blank)} of {len(rows)} rows have a blank `verdict`: "
            f"{', '.join(blank[:12])}{' …' if len(blank) > 12 else ''}\n"
            "       No verdict is emitted from a partially-filled sheet. The filled rows are not "
            "the sample the seed drew — they are a subset of it selected by whatever made a filler "
            "skip the rest, and 'the sampled episode ids' would then name episodes nobody judged."
        )

    bad_token = [(r["episode"], r["verdict"]) for r in rows
                 if str(r["verdict"]).strip() not in VERDICT_VALUES]
    if bad_token:
        shown = ", ".join(f"{e}={v!r}" for e, v in bad_token[:12])
        raise SheetError(
            f"{len(bad_token)} row(s) carry a verdict outside {VERDICT_VALUES}: {shown}\n"
            "       Mapping an unrecognised token onto one of the three would be this script "
            "guessing what a filler meant, and every such guess lands in a committed record."
        )

    no_axis = [r["episode"] for r in rows
               if str(r["verdict"]).strip() == "mismatch" and not (r.get("mismatched_axes") or [])]
    if no_axis:
        raise SheetError(
            f"{len(no_axis)} `mismatch` row(s) name no axis: {', '.join(no_axis[:12])}\n"
            "       'the disagreements' is what the TODO's action asks to be recorded, and a "
            "mismatch with no clause named cannot tell anyone whether arm C needs a per-episode "
            "apple, a per-episode cloth, or a per-episode light."
        )

    illegal_axis = sorted({
        a for r in rows for a in (r.get("mismatched_axes") or []) if a not in MISMATCH_AXES
    })
    if illegal_axis:
        raise SheetError(
            f"unknown mismatch axis/axes {illegal_axis}; allowed: {list(MISMATCH_AXES)}."
        )

    contradictory = [r["episode"] for r in rows
                     if str(r["verdict"]).strip() != "mismatch" and (r.get("mismatched_axes") or [])]
    if contradictory:
        raise SheetError(
            f"{len(contradictory)} row(s) name a mismatched axis while their verdict is not "
            f"`mismatch`: {', '.join(contradictory[:12])}. The row says two different things and "
            "picking one of them here would be inventing the answer."
        )


def check_frames(rows: list[dict]) -> list[str]:
    """Frames must still be the bytes that were judged. Returns disqualification reasons.

    Four cases, and the third one is why the earlier ``if not path or not recorded: continue`` was
    wrong. ``build-sheet`` writes ``frame`` and ``frame_sha256`` together or writes neither, so:

    * both set   — the ordinary row. Bytes changed is FATAL (the verdict describes pixels nobody
      looked at); the file gone is disqualifying (what was judged cannot be confirmed).
    * both null  — the row names no pixels (``--skip-frames``, or an extraction that failed). It
      cannot have been judged by looking, so it costs gate qualification here as well as at build
      time; the reasons are about different moments and both belong in the record.
    * one of the two null — not a row this script wrote. A row naming a frame with no digest is
      exempt from the check the module docstring calls fatal, and nulling one field is the cheapest
      way to make a changed frame pass: the check that is skipped is invisible in the artifact,
      which reports ``gate_qualified: true`` with no note. Refuse the row instead.
    """
    missing: list[str] = []
    changed: list[str] = []
    unpixeled: list[str] = []
    half: list[str] = []
    for r in rows:
        recorded = r.get("frame_sha256")
        path = r.get("frame")
        if not path and not recorded:
            unpixeled.append(r["episode"])
            continue
        if not path or not recorded:
            half.append(r["episode"])
            continue
        p = Path(path)
        if not p.is_file():
            missing.append(r["episode"])
            continue
        if sha256_file(p) != recorded:
            changed.append(r["episode"])
    if half:
        raise SheetError(
            f"{len(half)} row(s) carry a frame path without its sha256, or a sha256 without a "
            f"frame: {', '.join(half[:12])}\n"
            "       build-sheet writes both or neither, so this row was edited. A frame with no "
            "recorded digest is a frame the integrity check cannot run on, and skipping it "
            "silently is exactly how a verdict comes to describe pixels nobody looked at. Rebuild "
            "the sheet rather than repairing the row."
        )
    if changed:
        raise SheetError(
            f"{len(changed)} frame(s) no longer match the digest recorded when the sheet was "
            f"built: {', '.join(changed[:12])}\n"
            "       The verdicts on those rows are about pixels that are no longer there, so this "
            "evidence would describe frames nobody looked at. Rebuild the sheet and re-fill it."
        )

    reasons: list[str] = []
    if missing:
        reasons.append(
            f"{len(missing)} frame(s) named by the sheet are gone at verdict time "
            f"({', '.join(missing[:8])}{'…' if len(missing) > 8 else ''}), so what was judged "
            "cannot be confirmed."
        )
    if unpixeled:
        reasons.append(
            f"{len(unpixeled)} row(s) name no frame at all "
            f"({', '.join(unpixeled[:8])}{'…' if len(unpixeled) > 8 else ''}): nobody could have "
            "filled them by looking, so whatever verdict they carry is not a verdict about pixels."
        )
    return reasons


def check_meta_gate(meta: dict) -> list[str]:
    """The build's own gate stamp, read rather than inferred. Returns its reasons.

    ``list(meta.get('gate_disqualified_reasons') or [])`` inferred qualification from the ABSENCE
    of reasons, which makes ``gate_qualified: true`` the default for any meta that does not carry
    the key — including one whose own ``gate_qualified`` is ``false``, and including one whose
    reasons list somebody emptied. Gate qualification is opt-in: it is a claim the build has to
    have made explicitly, in a meta that agrees with itself, and there is no reading of a missing
    field that supports it.
    """
    if meta.get("schema") != SHEET_SCHEMA:
        raise SheetError(
            f"sheet meta declares schema {meta.get('schema')!r}, not {SHEET_SCHEMA!r}. It is not a "
            "meta this script wrote, and its gate stamp cannot be read as one."
        )
    qualified = meta.get("gate_qualified")
    reasons = meta.get("gate_disqualified_reasons")
    if not isinstance(qualified, bool) or not isinstance(reasons, list):
        raise SheetError(
            "the sheet meta carries no `gate_qualified` / `gate_disqualified_reasons` pair "
            f"(got {qualified!r} / {type(reasons).__name__}). Gate qualification is opt-in and is "
            "never inferred from a missing field: without the build's own stamp this run cannot "
            "say whether the sample it reports was a full one. Rebuild the sheet."
        )
    if qualified != (not reasons):
        raise SheetError(
            f"the sheet meta says gate_qualified={qualified} while carrying {len(reasons)} "
            "disqualification reason(s). The two disagree, so the meta was edited after the build "
            "wrote it, and reporting either one would be this script picking which edit to "
            "believe."
        )
    out = [str(r) for r in reasons]
    if not qualified and not out:                      # unreachable via the check above; belt.
        out.append("the sheet was stamped not gate-qualified at build time.")
    return out


def check_meta_integrity(meta: dict) -> None:
    """``sheet_id`` recomputed from the meta's own fields. A doctored meta cannot pin a sample.

    The rows are checked against ``sampled_episodes`` (see ``check_sample_identity``), so the
    remaining way to shrink a sample is to edit BOTH: delete five rows and delete the same five
    ids from the meta. ``sheet_id`` is a digest of seed, size, fraction, ids and prompt, and it is
    also written into every row, so recomputing it here closes that loop — the edited meta no
    longer hashes to the id its own rows carry, and the rows cannot be re-stamped without saying
    so in all forty of them.
    """
    fields = ("sample_seed", "sample_size", "frame_fraction", "sampled_episodes", "prompt")
    absent = [f for f in fields if meta.get(f) is None]
    if absent:
        raise SheetError(
            f"the sheet meta is missing {absent}, so the sample it describes cannot be "
            "reconstructed or checked."
        )
    recomputed = sheet_id(
        int(meta["sample_seed"]), int(meta["sample_size"]), float(meta["frame_fraction"]),
        [str(e) for e in meta["sampled_episodes"]], str(meta["prompt"]),
    )
    if recomputed != meta.get("sheet_id"):
        raise SheetError(
            f"the sheet meta records sheet_id {meta.get('sheet_id')} but its own seed, sample "
            f"size, frame fraction, episode ids and prompt hash to {recomputed}. Something was "
            "edited after the draw. The sheet_id is what pins WHICH sample this is; a meta that "
            "does not hash to its own id can pin nothing, and the rows all carry the old id."
        )


def build_evidence(rows: list[dict], meta: dict, min_coverage: float,
                   styles_path: Path) -> dict[str, Any]:
    identity = read_identity_style(styles_path)
    if identity["prompt"] != meta.get("prompt"):
        raise SheetError(
            f"{styles_path}'s [identity_style].prompt is not the string this sheet was built "
            "against. The verdicts answer a question about a prompt that is no longer arm C's, so "
            "reporting them as this TODO's evidence would attach a measurement to the wrong "
            "string. Rebuild the sheet against the committed prompt."
        )

    meta_disq = check_meta_gate(meta)
    check_meta_integrity(meta)
    frame_reasons = check_frames(rows)

    verdicts = {r["episode"]: str(r["verdict"]).strip() for r in rows}
    counts = {v: sum(1 for x in verdicts.values() if x == v) for v in VERDICT_VALUES}
    decidable = counts["match"] + counts["mismatch"]
    coverage = decidable / len(rows) if rows else 0.0

    disagreements = [
        {
            "episode": r["episode"],
            "axes": list(r.get("mismatched_axes") or []),
            "notes": str(r.get("notes") or "").strip(),
            "frame": r.get("frame"),
            "frame_index": r.get("frame_index"),
        }
        for r in rows if verdicts[r["episode"]] == "mismatch"
    ]
    unsure = [
        {"episode": r["episode"], "notes": str(r.get("notes") or "").strip()}
        for r in rows if verdicts[r["episode"]] == "unsure"
    ]
    axis_counts = {
        a: sum(1 for d in disagreements if a in d["axes"]) for a in MISMATCH_AXES
    }

    # Inherited from the build's own stamp (read, not inferred), then this run's own findings.
    disq = list(meta_disq) + list(frame_reasons)
    # The two build-time rules that are still derivable from the pinned sample are re-derived here
    # rather than only inherited: a meta with its reasons emptied and `gate_qualified: true` is
    # self-consistent and hashes to its own sheet_id, so inheritance alone would let a five-episode
    # smoke run, or a frame taken after the transfer, be hand-promoted into the evidence. Identical
    # strings by construction, so an inherited reason is not repeated.
    for derived in (sample_size_disqualification(len(rows)),
                    frame_fraction_disqualification(float(meta["frame_fraction"]))):
        if derived and derived not in disq:
            disq.append(derived)
    if coverage < min_coverage:
        disq.append(
            f"coverage {coverage:.3f} < --min-coverage {min_coverage}: {counts['unsure']} of "
            f"{len(rows)} rows abstained, so too little of the sample was decidable for the "
            "counts over the rest to describe it."
        )

    return {
        "schema": EVIDENCE_SCHEMA,
        "writeup": WRITEUP,
        "todo": TODO_ID,
        "evidence_required": meta.get("evidence_required"),
        "produced_by": "scripts/build_identity_prompt_sheet.py verdict",
        "produced_date": date.today().isoformat(),
        "produced_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),

        # -- provenance of the sample --------------------------------------------------------
        "sheet_id": meta.get("sheet_id"),
        "sheet": meta.get("sheet"),
        "manifest": meta.get("manifest"),
        "n_episodes_in_manifest": meta.get("n_episodes_in_manifest"),
        "partition_rule": meta.get("partition_rule"),
        "partition_content_sha256_at_build": meta.get("partition_content_sha256"),
        "partition_content_sha256_now": identity["partition_content_sha256"],
        "prompt": meta.get("prompt"),
        "caption_provenance": meta.get("caption_provenance"),
        "sample_scheme": meta.get("sample_scheme"),
        "sample_seed": meta.get("sample_seed"),
        "frame_fraction": meta.get("frame_fraction"),
        "frame_rule": meta.get("frame_rule"),

        # -- the three things evidence_required names ----------------------------------------
        # Both of these are checked against the meta's pinned draw before they are written (see
        # `check_sample_identity`), so they are the sample that was drawn and not the rows that
        # survived. `sample_size_at_build` and `sample_strata` are carried so a READER can see the
        # same thing without re-running anything: a shrunk sample is otherwise undetectable in an
        # artifact that re-derives its own denominator.
        "sampled_episodes": [r["episode"] for r in rows],
        "sample_size": len(rows),
        "per_episode_verdicts": verdicts,
        "sample_size_at_build": meta.get("sample_size"),
        "sampled_episodes_at_build": meta.get("sampled_episodes"),
        "sample_strata": meta.get("sample_strata"),

        # -- what the reader decides from ----------------------------------------------------
        "verdict_counts": counts,
        "mismatch_axis_counts": axis_counts,
        "disagreements": disagreements,
        "abstentions": unsure,
        "coverage": coverage,
        "min_coverage": min_coverage,
        "coverage_scope": (
            "fraction of the sampled rows carrying a decidable verdict (match or mismatch). It "
            "says nothing about how many matched."
        ),

        "gate_qualified": not disq,
        "gate_disqualified_reasons": disq,
        "judge": meta.get("judge"),
        "judge_note": meta.get("judge_note"),

        "notes": [
            "NO OVERALL PASS IS COMPUTED, AND THIS IS NOT AN OVERSIGHT. The TODO's action says an "
            "inconstant appearance IS the finding: 'If the appearance is not constant across the "
            "402, that is the finding and arm C needs a per-episode identity prompt rather than "
            "one shared string.' Folding these counts into PASS/FAIL would coin the threshold "
            "that decision turns on, after the numbers existed.",
            "gate_qualified is about the admissibility of this evidence, never about arm C. A "
            "gate-qualified run can report every episode as a mismatch.",
            "The sample can DETECT inconstancy; it cannot certify constancy. Zero mismatches in "
            f"{len(rows)} bounds the rate at about {3 / max(len(rows), 1):.1%} at 95% (rule of "
            "three), which over 402 episodes still permits roughly "
            f"{round(3 / max(len(rows), 1) * (meta.get('n_episodes_in_manifest') or 0))} "
            "disagreeing episodes. If the decision needs certainty, the answer is a census.",
            "One frame per episode. A change that happens later in an episode — a light switched "
            "mid-clip — is outside what this sample can see, and a per-episode verdict here is a "
            "verdict about one instant of that episode.",
            "This evidence does not close the TODO and does not license generation. PR-08 §1 "
            "gates generation on every §8 item; closing this TODO is an edit to "
            "configs/transfer25/styles.toml made by whoever reads these counts.",
        ],
    }


def evidence_toml(ev: dict[str, Any]) -> str:
    """The three required items as a TOML fragment, pasteable under the [[blocking_todos]] entry.

    Deliberately emits NO ``status`` line. `closes_by` says the TODO closes by setting
    ``status = "CLOSED"`` *plus* the evidence, and which of those the counts below justify is the
    reader's call, not this script's — emitting a pre-typed CLOSED would put the decision in the
    clipboard.

    Strings go through ``json.dumps`` because TOML basic strings use JSON's escapes for everything
    a filler's free-text note can contain, and a raw quote or backslash pasted into the committed
    partition would break the file the generator reads.
    """
    def s(x: Any) -> str:
        return json.dumps("" if x is None else str(x), ensure_ascii=True)

    counts = ev["verdict_counts"]
    lines = [
        f"# --- {TODO_ID}: evidence, produced by",
        "#     scripts/build_identity_prompt_sheet.py verdict. Paste inside the [[blocking_todos]]",
        "#     entry. NOTE: blocking_todos is INSIDE the partition content hash (only [hash] and",
        "#     [consumer] are excluded), so after pasting, re-run",
        "#       scripts/check_style_partition.py --write-hash --emit-json",
        "#     or the sidecar and the rendering disagree with the file and the verifier fails.",
        f"evidence_produced = {s(ev['produced_date'])}",
        f"evidence_sheet_id = {s(ev['sheet_id'])}",
        f"evidence_gate_qualified = {'true' if ev['gate_qualified'] else 'false'}",
        f"evidence_sample_seed = {int(ev['sample_seed'])}",
        f"evidence_sample_size = {int(ev['sample_size'])}",
        f"evidence_sample_scheme = {s(ev['sample_scheme'])}",
        f"evidence_frame_rule = {s(ev['frame_rule'])}",
        "evidence_sampled_episodes = [",
    ]
    lines += [f"  {s(e)}," for e in ev["sampled_episodes"]]
    lines += [
        "]",
        "evidence_verdict_counts = { "
        + ", ".join(f"{k} = {counts[k]}" for k in VERDICT_VALUES)
        + " }",
        "evidence_verdicts = [",
    ]
    lines += [f"  {s(f'{ep} = {v}')}," for ep, v in ev["per_episode_verdicts"].items()]
    lines += ["]", "evidence_disagreements = ["]
    for d in ev["disagreements"]:
        note = " ".join(d["notes"].split())
        axes = ",".join(d["axes"])
        lines.append("  " + s(f"{d['episode']} [{axes}] {note}".strip()) + ",")
    lines += [
        "]",
        f"evidence_coverage = {ev['coverage']:.4f}",
        "# No overall pass is emitted: an inconstant appearance IS the finding (see the TODO's",
        "# `action`), so whether these counts close this item is the reader's decision.",
    ]
    return "\n".join(lines) + "\n"


def cmd_verdict(args: argparse.Namespace) -> int:
    sheet_file, meta_file = sheet_paths(args.sheet)
    rows, meta = read_filled_sheet(sheet_file, meta_file)
    check_fill(rows)
    ev = build_evidence(rows, meta, args.min_coverage, args.styles)

    side, digest = write_artifact(args.out, ev)
    toml_text = evidence_toml(ev)
    if args.emit_toml:
        args.emit_toml.parent.mkdir(parents=True, exist_ok=True)
        args.emit_toml.write_text(toml_text, encoding="utf-8")

    c = ev["verdict_counts"]
    print(f"sheet       {sheet_file} ({ev['sample_size']} episodes, seed {ev['sample_seed']})",
          file=sys.stderr)
    print(f"verdicts    match {c['match']} | mismatch {c['mismatch']} | unsure {c['unsure']} "
          f"(coverage {ev['coverage']:.3f})", file=sys.stderr)
    if ev["disagreements"]:
        print("axes        " + ", ".join(
            f"{a}:{n}" for a, n in ev["mismatch_axis_counts"].items() if n), file=sys.stderr)
        for d in ev["disagreements"][:12]:
            print(f"            {d['episode']} [{','.join(d['axes'])}] "
                  f"{' '.join(d['notes'].split())[:90]}", file=sys.stderr)
    print(f"wrote       {args.out}", file=sys.stderr)
    print(f"sha256      {digest}  ({side})", file=sys.stderr)
    if args.emit_toml:
        print(f"toml        {args.emit_toml}", file=sys.stderr)
    print("\nNO OVERALL PASS IS COMPUTED. The TODO's action makes an inconstant appearance the "
          "finding\n(arm C would then need a per-episode prompt), so whether these counts close "
          "the item is\nthe reader's decision and not this script's.", file=sys.stderr)
    for reason in ev["gate_disqualified_reasons"]:
        print(f"\nNOT GATE-QUALIFIED: {reason}", file=sys.stderr)
    if not args.emit_toml:
        print("\n" + toml_text)
    return EXIT_OK if ev["gate_qualified"] else EXIT_NOT_GATE_QUALIFIED


# --------------------------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-sheet", help="sample episodes, cut frames, write a BLANK sheet")
    b.add_argument("--manifest", required=True, type=Path,
                   help="SOURCE/manifest.json from scripts/build_pr08_source.py — the same file "
                        "`97` restyles from, read with the driver's own loader")
    b.add_argument("--styles", type=Path, default=DEFAULT_STYLES,
                   help="the committed partition carrying [identity_style].prompt")
    b.add_argument("--out", required=True, type=Path,
                   help="DIRECTORY for sheet.jsonl, sheet_meta.json and frames/ — put it under "
                        "runs/; only the verdict artifact is committed. Not a .jsonl path: that "
                        "is what `verdict --sheet` takes")
    b.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                   help=f"episodes to draw, one per stratum (default {DEFAULT_SAMPLE_SIZE}; below "
                        f"{MIN_GATE_QUALIFIED_SAMPLE} is a smoke run and is stamped as one)")
    b.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED,
                   help=f"sampling seed, recorded everywhere (default {DEFAULT_SAMPLE_SEED})")
    b.add_argument("--frame-fraction", type=float, default=DEFAULT_FRAME_FRACTION,
                   help=f"where in the clip the representative frame is taken from, in [0, 1] "
                        f"(default {DEFAULT_FRAME_FRACTION}; outside the range is refused, and 0 "
                        f"or past {MAX_GATE_QUALIFIED_FRAME_FRACTION} is not the early rule fixed "
                        f"a priori and is stamped as such)")
    b.add_argument("--skip-frames", action="store_true",
                   help="write the rows without extracting anything; disqualifies the sheet")
    b.add_argument("--ffmpeg", default="ffmpeg")
    b.add_argument("--allow-reseed", action="store_true",
                   help="overwrite an UNFILLED sheet in --out that was drawn under different "
                        "parameters. A sheet carrying answers is never overwritten, with or "
                        "without this flag — write the new sample to a different --out")

    v = sub.add_parser("verdict", help="read a FILLED sheet, emit the TODO's evidence")
    v.add_argument("--sheet", required=True, type=Path,
                   help="the build-sheet output directory, or sheet.jsonl itself")
    v.add_argument("--styles", type=Path, default=DEFAULT_STYLES)
    v.add_argument("--out", type=Path, default=DEFAULT_EVIDENCE_OUT,
                   help=f"evidence artifact (default the tracked {DEFAULT_EVIDENCE_OUT_REL})")
    v.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                   help="fraction of rows that must be decidable (not `unsure`)")
    v.add_argument("--emit-toml", type=Path, default=None,
                   help="write the pasteable TOML fragment here instead of to stdout")

    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.cmd == "build-sheet":
            return cmd_build_sheet(args)
        return cmd_verdict(args)
    except (SheetError, DriverError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return EXIT_FATAL


if __name__ == "__main__":
    raise SystemExit(main())
