"""Gripper-channel admissibility audit (T-31): is there a grasp in this data at all?

Why this exists. WAM-Bench emits ``gripper_accuracy`` for every run, and on our own data that
number was 0.85-0.89 while the demonstrated channel never crossed the binarization threshold —
it was measuring how often two constants agree. T-27 recorded a warning next to it and drew the
wrong conclusion from it: "the demonstrated gripper never opens or closes, this dataset cannot
support a grasping claim". The audit below is what turns that sentence into a measurement, and
running it in ``--lerobot`` mode on the source snapshot shows the grasp IS in the data and our
converter flattened it (``scripts/convert_lerobot_g1.py``, ``docs/benchmark.md``).

Contracts:
- **Never decodes video.** The whole point is a gate cheap enough to run over a full dataset
  before anything is trained on it: parquet columns only, no GPU, no ffmpeg. Pinned by
  ``tests/test_gripper_audit.py::test_audit_never_decodes_video``.
- Torch-free, numpy + pyarrow only.
- A DATASET-level verdict against pre-registered constants, unlike ``episode_report`` (FR-08)
  which is per-episode and whose ``ActionReport`` is a frozen contract.
- Any normalization applied to a raw channel is DATASET-level, and so is every CHOICE that
  precedes it (which joint, which hand). Per-episode is the tempting option and is wrong for both:
  a per-episode min-max makes the same physical aperture mean different values in different
  episodes, and a per-episode "most active joint" concatenates two different physical joints into
  one channel. Neither is learnable and neither is comparable across episodes.
- ``passed`` is decided by the pre-registered clauses alone. Findings that are true but cannot be
  turned into a clause without refusing a legitimate channel ride in ``reasons`` as notices — see
  ``GripperChannelStats.notices``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from wam.evaluation.offline import GRIPPER_BINARIZE_THRESHOLD, _fmt

AUDIT_VERSION = "0.2.0"
"""Version of the audit RULE SET, bumped when a clause is added or changed.

0.2.0 adds the grasp-cycle clause (``GRIPPER_MIN_EPISODES_WITH_GRASP_CYCLE``) and the saturation
notice, after 0.1.0 was shown to admit a channel containing no grasp anywhere: 50 episodes of a
pure monotone ramp 0 -> 1, which never closes and reopens in ANY episode, scored p2p 1.00, 1.00
debounced transitions per episode and 1.00 episodes-with-a-transition, i.e. admissible with zero
failed clauses. Every field added since 0.1.0 carries a default, so a report written by 0.1.0
still parses through ``from_json``.
"""

# --- pre-registered admissibility constants ----------------------------------------------------
#
# Fixed before any dataset is audited. Each carries the reasoning that produced it, never the
# number it happened to score — a threshold chosen after reading a measurement measures nothing.

GRIPPER_MIN_DYNAMIC_RANGE = 0.25
"""Below this peak-to-peak range the gripper channel carries no open/close event to score.

Unchanged in value and meaning from where it was first pre-registered (T-27, formerly
``benchmark.GRIPPER_MIN_DYNAMIC_RANGE``, which now imports it from here so the gate and the
audit that reports it cannot drift apart).
"""

GRIPPER_HYSTERESIS_MARGIN = 0.10
"""Half-width of the dead band around the binarization threshold, in gripper units.

Derived, not measured: it must stay below ``GRIPPER_MIN_DYNAMIC_RANGE / 2`` (= 0.125), because a
channel that exactly clears the range floor spans threshold +/- 0.125 and has to be able to reach
both latch levels. A wider margin would make the two gates contradict each other — the range
clause would admit a channel the transition clause could never see move.
"""

GRIPPER_MIN_TRANSITIONS_PER_EPISODE = 1.0
"""Mean debounced open/close transitions per episode required of an admissible channel.

Derived from the task, not from a run: a pick-and-place demonstration closes on the object and
opens to release it, i.e. 2 transitions. Requiring a MEAN of 1.0 tolerates half the episodes
being partial (approach-only, truncated recording) while still refusing a channel that is
constant.

On its own this clause does NOT refuse a channel with no grasp in it, and audit 0.1.0 shipped
believing it did. A monotone ramp 0 -> 1 that never reopens clears it at exactly 1.00 in every
episode; so does any channel whose value only ever goes one way. The relaxation from 2 to a MEAN
of 1.0 swallowed the very event being tested, which is why the tolerance it encodes is now ALSO
stated per-episode, where a mean cannot launder it — see
``GRIPPER_MIN_EPISODES_WITH_GRASP_CYCLE``.
"""

GRIPPER_MIN_EPISODES_WITH_GRASP_CYCLE = 0.5
"""Fraction of episodes that must contain a COMPLETE grasp: >= 2 debounced transitions.

Not a new tolerance and not fitted to any measurement — it is the mean clause's own derivation
written where it cannot be averaged away. ``GRIPPER_MIN_TRANSITIONS_PER_EPISODE = 1.0`` is
exactly "up to half the episodes may be partial": 2 transitions in half of them and 0 in the
other half averages to 1.0. A mean cannot express that, and the arithmetic that produces 1.0 from
(2, 0) also produces it from (1, 1) — one-way motion in every episode, no grasp anywhere. This
clause says the same thing about episodes instead of about their mean, so the close-and-release
the task is defined by has to appear in at least the half the mean was already assuming.
"""

GRIPPER_MIN_EPISODES_WITH_TRANSITION = 0.8
"""Fraction of episodes that must contain at least one debounced transition.

The mean above can be carried by a minority of very active episodes, so this is the second,
per-episode clause: at most one episode in five may contain no grasp at all before the set stops
being a grasping dataset and becomes a reaching dataset with occasional grasps. Numerically equal
to ``benchmark.CRITICAL_QUANTILE`` by coincidence, deliberately NOT the same symbol — the two
quantities answer different questions and must be able to move independently.
"""

_SATURATION_EPS = 1e-6
"""Tolerance for calling a [0, 1] gripper value clipped."""


def expected_saturated_frac(num_episodes: int, num_steps: int) -> float:
    """The rail mass a DATASET-level min-max affine over the audited set can produce.

    Derived from the mapping, never from an observed number. ``(x - lo) / (hi - lo)`` over the
    audited set puts its two extremal samples exactly on 0.0 and 1.0 and everything else strictly
    inside, so a dataset-level fit saturates 2 samples. Letting every episode touch the fitted
    extremes independently — the loosest a fit without dwell can be — gives ``2 * num_episodes``.
    Rail mass above that was produced by an affine that is NOT this set's own range: either a
    pinned/foreign affine (``scripts/convert_lerobot_g1.py --gripper-affine``) or a fixed formula
    like the legacy ``clip((mean + 1) / 2, 0, 1)``.
    """
    if num_steps <= 0:
        return 0.0
    return min(1.0, 2.0 * max(num_episodes, 1) / num_steps)


def latched_states(
    values: np.ndarray,
    *,
    threshold: float = GRIPPER_BINARIZE_THRESHOLD,
    margin: float = GRIPPER_HYSTERESIS_MARGIN,
) -> np.ndarray:
    """Per-sample latched state: 1 closed, 0 open, -1 before the first decisive sample.

    The same dead band :func:`debounced_transitions` counts with, but kept per sample and
    forward-filled, so a caller can ask *where* a transition happened rather than only how many
    there were. PR-03's grasp-anticipation metric needs the index — it scores the steps at and
    after a flip — and giving it its own latch would let the metric and the admissibility gate
    disagree about what a grasp is. :func:`debounced_transitions` is defined in terms of this
    function for exactly that reason; the two cannot drift.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    raw = np.where(v >= threshold + margin, 1, np.where(v <= threshold - margin, 0, -1))
    out = np.empty(raw.shape, dtype=np.int8)
    state = -1
    for i, sample in enumerate(raw):
        if sample >= 0:
            state = int(sample)
        out[i] = state
    return out


def debounced_transitions(
    values: np.ndarray,
    *,
    threshold: float = GRIPPER_BINARIZE_THRESHOLD,
    margin: float = GRIPPER_HYSTERESIS_MARGIN,
) -> int:
    """Number of open<->close transitions in ``values``, with a hysteresis dead band.

    A sample latches the channel closed only at ``threshold + margin`` and open only at
    ``threshold - margin``; samples inside the band leave the latched state unchanged. Without
    that band a channel that merely sits ON the threshold and dithers produces a stream of
    crossings and reads as a busy gripper — exactly the artefact that made ``gripper_accuracy``
    look like a grasp metric on a dead channel. The undebounced count is reported separately
    (``crossings``), so a FAIL can be read rather than merely asserted.
    """
    seq = latched_states(values, threshold=threshold, margin=margin)
    decided = seq[seq >= 0]
    if decided.size < 2:
        return 0
    return int((np.diff(decided) != 0).sum())


def crossings(values: np.ndarray, *, threshold: float = GRIPPER_BINARIZE_THRESHOLD) -> int:
    """Raw threshold crossings, no dead band — the number ``debounced_transitions`` corrects."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size < 2:
        return 0
    return int((np.diff((v >= threshold).astype(np.int8)) != 0).sum())


class GripperChannelStats(BaseModel):
    """Per-channel measurements. Pure description — the verdict lives in the report."""

    model_config = ConfigDict(frozen=True)

    name: str
    num_episodes: int = Field(ge=0)
    num_steps: int = Field(ge=0)
    p2p_global: float
    p2p_per_episode_mean: float
    p2p_per_episode_min: float
    frac_at_or_above_threshold: float
    frac_saturated_low: float
    frac_saturated_high: float
    crossings_per_episode: float
    debounced_transitions_per_episode: float
    episodes_with_transition_frac: float
    episodes_with_grasp_cycle_frac: float = 0.0
    majority_class_pct: float

    @property
    def admissible(self) -> bool:
        """True when this channel clears all four pre-registered clauses."""
        return not self.failed_clauses()

    @property
    def frac_saturated(self) -> float:
        """Fraction of samples sitting exactly on a [0, 1] rail — the clipping evidence."""
        return self.frac_saturated_low + self.frac_saturated_high

    @property
    def clipping_suspected(self) -> bool:
        """True when more samples sit on a rail than this set's own min-max fit could put there.

        Not an admissibility clause; see :meth:`notices` for why it cannot be one.
        """
        return self.frac_saturated > expected_saturated_frac(self.num_episodes, self.num_steps)

    def failed_clauses(self) -> tuple[str, ...]:
        """Every failed clause, each naming the measured value that failed it."""
        out: list[str] = []
        if self.p2p_global < GRIPPER_MIN_DYNAMIC_RANGE:
            out.append(
                f"dynamic range {self.p2p_global:.4f} < {GRIPPER_MIN_DYNAMIC_RANGE} "
                "(no open/close event to score)"
            )
        if self.debounced_transitions_per_episode < GRIPPER_MIN_TRANSITIONS_PER_EPISODE:
            out.append(
                f"debounced transitions/episode {self.debounced_transitions_per_episode:.3f} < "
                f"{GRIPPER_MIN_TRANSITIONS_PER_EPISODE} (margin {GRIPPER_HYSTERESIS_MARGIN})"
            )
        if self.episodes_with_grasp_cycle_frac < GRIPPER_MIN_EPISODES_WITH_GRASP_CYCLE:
            out.append(
                f"episodes with a complete grasp (>= 2 debounced transitions) "
                f"{self.episodes_with_grasp_cycle_frac:.3f} < "
                f"{GRIPPER_MIN_EPISODES_WITH_GRASP_CYCLE} (the channel moves, but it never "
                "closes AND reopens)"
            )
        if self.episodes_with_transition_frac < GRIPPER_MIN_EPISODES_WITH_TRANSITION:
            out.append(
                f"episodes with a transition {self.episodes_with_transition_frac:.3f} < "
                f"{GRIPPER_MIN_EPISODES_WITH_TRANSITION}"
            )
        return tuple(out)

    def notices(self) -> tuple[str, ...]:
        """Findings that are true of the channel but are NOT grounds for refusing the dataset.

        Only one today: clipping. Every mapping in this repo ends in ``clip(..., 0, 1)``, and
        clipping moves all four admissibility clauses in the PASSING direction — a clipped
        channel has a larger ``p2p_global``, more raw crossings and more debounced transitions
        than the same channel unclipped. So the audit has to be able to say "these values were
        clipped, not measured", which before 0.2.0 it could not: ``frac_saturated_low/high`` sat
        in the table and reached neither ``failed_clauses`` nor ``reasons``.

        Deliberately NOT a clause, and this is the part worth reading. Saturation alone cannot
        separate a defect from a design: a two-state gripper COMMAND channel is saturated at both
        rails by construction, and ``data/raw/gr00t_apple`` contains exactly that —
        ``action.left_hand.max_joint[0]`` sits on a rail for 97.6 % of its samples and is the
        cleanest grasp signal in the snapshot (2.04 debounced transitions per episode, a complete
        cycle in 99.8 % of episodes). A clause tight enough to catch that would refuse it. A
        channel dwelling against a mechanical stop saturates for legitimate reasons too. So this
        lands in ``GripperAuditReport.reasons`` on PASS and on FAIL alike, and the exact gate
        lives where the mapping is applied and the unclipped values still exist
        (``convert_lerobot_g1.pinned_hand_affine`` refuses a pinned ``--gripper-affine`` that
        clips at all).
        """
        if not self.clipping_suspected or self.p2p_global <= 0.0:
            # A constant channel has no rails to be pushed onto — the mapping collapses it to a
            # single value and the range clause already refuses it. Reporting "100 % saturated"
            # there describes the collapse, not a clip, and would bury the real notices.
            return ()
        ref = expected_saturated_frac(self.num_episodes, self.num_steps)
        note = (
            f"NOTE (not gated): {self.frac_saturated:.4f} of samples sit exactly on a [0, 1] "
            f"rail ({self.frac_saturated_low:.4f} low / {self.frac_saturated_high:.4f} high), "
            f"above the {ref:.6f} a dataset-level min-max affine over these "
            f"{self.num_episodes} episode(s) can produce. Either this channel is a two-state "
            "command rather than a measurement, or it was mapped by an affine that is not this "
            "set's own range (manifest `mapping.gripper_synergy`). In the second case range, "
            "crossings and transitions are all inflated and every clause above reads as an "
            "upper bound."
        )
        return (note,)


class GripperAuditReport(BaseModel):
    """Dataset-level verdict: may any result from this data be described as being about grasping?

    ``passed`` is the gate ``scripts/audit_gripper.py`` turns into an exit code. In ``wam`` mode
    it is the verdict on ``scored_channel`` alone — the commanded ``gripper_target``, because that
    is the channel ``bench_metrics`` binarizes and therefore the only one a bench number can be
    about. In ``lerobot`` mode it is true when ANY hand channel is admissible, which is the
    question that mode exists to answer: is the signal present in the source at all?

    ``reasons`` is everything a reader has to see, not only what failed: the scored channel's
    failed clauses when ``passed`` is False, plus every channel's non-fatal notices
    (:meth:`GripperChannelStats.notices`) in both cases. ``passed`` is decided by
    ``failed_clauses()`` alone, so a notice can never flip a verdict — but it can never be
    silently absent either, which is what a clipped channel needs.

    ``report_version`` names the clause set that produced ``passed``. An older report still parses
    (fields added since carry defaults) and keeps the verdict it was written with, but its
    channels must NOT have their clauses re-derived: a default is not a measurement, and a 0.1.0
    report carries no ``episodes_with_grasp_cycle_frac``. Re-run the audit instead — it is
    seconds over a full dataset, which is the whole reason it never decodes video.
    """

    model_config = ConfigDict(frozen=True)

    report_version: str = AUDIT_VERSION
    root: str
    source_kind: str
    num_episodes: int = Field(ge=0)
    channels: tuple[GripperChannelStats, ...]
    scored_channel: str
    passed: bool
    reasons: tuple[str, ...]
    thresholds: dict[str, float]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> GripperAuditReport:
        return cls.model_validate_json(text)

    def channel(self, name: str) -> GripperChannelStats:
        for ch in self.channels:
            if ch.name == name:
                return ch
        raise KeyError(f"no channel {name!r}; have {[c.name for c in self.channels]}")

    def render_markdown(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            f"# Gripper audit — `{self.root}`",
            "",
            (
                f"**{verdict}** · source `{self.source_kind}` · {self.num_episodes} episode(s) "
                f"· scored channel `{self.scored_channel}`"
            ),
            "",
            (
                "| channel | p2p | p2p/ep mean | >= thr | sat lo | sat hi | cross/ep | "
                "debounced/ep | eps with transition | eps with grasp cycle | majority % |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for ch in self.channels:
            mark = "**" if ch.name == self.scored_channel else ""
            lines.append(
                f"| {mark}{ch.name}{mark} | {_fmt(ch.p2p_global)} | "
                f"{_fmt(ch.p2p_per_episode_mean)} | {ch.frac_at_or_above_threshold:.3f} | "
                f"{ch.frac_saturated_low:.3f} | {ch.frac_saturated_high:.3f} | "
                f"{ch.crossings_per_episode:.2f} | {ch.debounced_transitions_per_episode:.2f} | "
                f"{ch.episodes_with_transition_frac:.3f} | "
                f"{ch.episodes_with_grasp_cycle_frac:.3f} | {ch.majority_class_pct:.1f} |"
            )
        lines += ["", "## Gates", ""]
        lines += [f"- `{k}` = {v:g}" for k, v in sorted(self.thresholds.items())]
        if self.passed:
            lines += [
                "",
                "All four clauses cleared: this data can support a claim about grasping.",
            ]
        # Printed on PASS too: a notice that only shows up on failures is a notice nobody sees on
        # the run that mattered. `passed` above is the verdict; these are the caveats on it.
        if self.reasons:
            lines += ["", "## Findings", ""]
            lines += [f"- {r}" for r in self.reasons]
        return "\n".join(lines) + "\n"


def _thresholds() -> dict[str, float]:
    return {
        "binarize_threshold": GRIPPER_BINARIZE_THRESHOLD,
        "hysteresis_margin": GRIPPER_HYSTERESIS_MARGIN,
        "min_dynamic_range": GRIPPER_MIN_DYNAMIC_RANGE,
        "min_episodes_with_grasp_cycle": GRIPPER_MIN_EPISODES_WITH_GRASP_CYCLE,
        "min_episodes_with_transition": GRIPPER_MIN_EPISODES_WITH_TRANSITION,
        "min_transitions_per_episode": GRIPPER_MIN_TRANSITIONS_PER_EPISODE,
    }


def channel_stats(
    name: str,
    per_episode: Sequence[np.ndarray],
    *,
    threshold: float = GRIPPER_BINARIZE_THRESHOLD,
    margin: float = GRIPPER_HYSTERESIS_MARGIN,
) -> GripperChannelStats:
    """Aggregate one channel's per-episode series into the audited statistics.

    ``per_episode`` must already be in gripper units ([0, 1]); mapping raw joint angles into that
    range is the caller's job precisely because the mapping must be dataset-level (see the module
    docstring) and this function only ever sees one episode at a time.
    """
    series = [np.asarray(s, dtype=np.float64).reshape(-1) for s in per_episode]
    series = [s for s in series if s.size]
    if not series:
        return GripperChannelStats(
            name=name,
            num_episodes=0,
            num_steps=0,
            p2p_global=0.0,
            p2p_per_episode_mean=0.0,
            p2p_per_episode_min=0.0,
            frac_at_or_above_threshold=0.0,
            frac_saturated_low=0.0,
            frac_saturated_high=0.0,
            crossings_per_episode=0.0,
            debounced_transitions_per_episode=0.0,
            episodes_with_transition_frac=0.0,
            episodes_with_grasp_cycle_frac=0.0,
            majority_class_pct=0.0,
        )
    flat = np.concatenate(series)
    p2p_ep = np.asarray([float(s.max() - s.min()) for s in series])
    trans = np.asarray(
        [debounced_transitions(s, threshold=threshold, margin=margin) for s in series]
    )
    cross = np.asarray([crossings(s, threshold=threshold) for s in series])
    frac_closed = float((flat >= threshold).mean())
    return GripperChannelStats(
        name=name,
        num_episodes=len(series),
        num_steps=int(flat.size),
        p2p_global=float(flat.max() - flat.min()),
        p2p_per_episode_mean=float(p2p_ep.mean()),
        p2p_per_episode_min=float(p2p_ep.min()),
        frac_at_or_above_threshold=frac_closed,
        frac_saturated_low=float((flat <= _SATURATION_EPS).mean()),
        frac_saturated_high=float((flat >= 1.0 - _SATURATION_EPS).mean()),
        crossings_per_episode=float(cross.mean()),
        debounced_transitions_per_episode=float(trans.mean()),
        episodes_with_transition_frac=float((trans > 0).mean()),
        # >= 2, not > 0: one transition is a hand that closed (or opened) and stayed there, which
        # is half of a grasp. The pair is what makes the mean clause unable to hide a monotone
        # channel behind an average — see GRIPPER_MIN_EPISODES_WITH_GRASP_CYCLE.
        episodes_with_grasp_cycle_frac=float((trans >= 2).mean()),
        majority_class_pct=float(max(frac_closed, 1.0 - frac_closed) * 100.0),
    )


def dataset_affine(per_episode: Sequence[np.ndarray]) -> tuple[float, float]:
    """DATASET-level (offset, span) mapping a raw channel onto [0, 1]; span 0 when constant.

    One affine for the whole set, deliberately: with a per-episode min-max the same physical
    aperture would map to a different number in every episode, which destroys the only property
    that makes the channel learnable and comparable across episodes.
    """
    lo = np.inf
    hi = -np.inf
    for s in per_episode:
        a = np.asarray(s, dtype=np.float64).reshape(-1)
        if a.size:
            lo = min(lo, float(a.min()))
            hi = max(hi, float(a.max()))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 0.0
    return lo, float(hi - lo)


def apply_affine(values: np.ndarray, offset: float, span: float) -> np.ndarray:
    """``(values - offset) / span`` clipped to [0, 1]; a zero span maps everything to 0.0.

    When ``(offset, span)`` came from :func:`dataset_affine` over the same series the clip is a
    no-op beyond the two extremal samples. It is not a no-op for a pinned or foreign affine, and
    then it hides how far outside the range the values were: that is what
    ``frac_saturated_low/high`` and :meth:`GripperChannelStats.notices` exist to surface.
    """
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    if span <= 0.0:
        return np.zeros_like(a)
    return np.clip((a - offset) / span, 0.0, 1.0)


def _findings(
    channels: Sequence[GripperChannelStats], scored: GripperChannelStats, *, any_channel: bool
) -> tuple[bool, tuple[str, ...]]:
    """``(passed, reasons)``. Notices are appended for EVERY channel, PASS or FAIL.

    The verdict comes from ``failed_clauses()`` only; notices ride along so a clipped channel
    cannot pass unremarked. They are gathered from every channel and not only from the scored
    one, because in ``wam`` mode the scored channel is fixed and a clipped state column is
    exactly the kind of thing the reader needs to see next to a PASS.
    """
    passed = any(c.admissible for c in channels) if any_channel else scored.admissible
    out: list[str] = []
    if not passed:
        out += [f"{scored.name}: {r}" for r in scored.failed_clauses()]
    for channel in channels:
        out += [f"{channel.name}: {n}" for n in channel.notices()]
    return passed, tuple(out)


def _pick_scored(channels: Sequence[GripperChannelStats]) -> GripperChannelStats:
    """The channel a verdict is reported against: the most convincing one on offer.

    Ordered by admissibility first so a PASS is never explained by a dead channel, then by
    whether clipping is suspected, then by debounced transitions, then by range. Clipping earns a
    channel more transitions and more range than it has, so without that second key the channel
    most likely to win this comparison is the one whose numbers are least trustworthy.
    """
    return max(
        channels,
        key=lambda c: (
            c.admissible,
            not c.clipping_suspected,
            c.debounced_transitions_per_episode,
            c.p2p_global,
        ),
    )


# --- converted WAM datasets --------------------------------------------------------------------


def audit_wam_dataset(
    root: str | Path,
    *,
    verify_checksums: bool = True,
    max_episodes: int | None = None,
) -> GripperAuditReport:
    """Audit a converted WAM dataset directory (``manifest.json`` per episode).

    Scores ``ActionChunk.gripper_target`` — the channel ``bench_metrics`` binarizes, so the only
    one a reported ``gripper_accuracy`` can be about. Each ``RobotState.gripper_state`` column is
    reported separately as a diagnostic; averaging them is what hid a frozen hand in the first
    place, so the audit refuses to average anything.

    Reads ``read_actions``/``read_states`` only — never ``read_frames``.
    """
    from wam.data.episode import EpisodeReader, list_episodes

    root_path = Path(root)
    dirs = list_episodes(root_path)
    if max_episodes is not None:
        dirs = dirs[:max_episodes]

    target_series: list[np.ndarray] = []
    state_series: dict[int, list[np.ndarray]] = {}
    for d in dirs:
        reader = EpisodeReader(d, verify_checksums=verify_checksums)
        parts = [
            np.asarray(chunk.gripper_target, dtype=np.float64)[: max(int(executed), 1)]
            for chunk, executed, _ts in reader.read_actions()
        ]
        if parts:
            # Only the executed prefix of each chunk: whole chunks overlap whenever the emission
            # stride is shorter than the horizon (FR-05's regime) and would count the same
            # physical transition twice.
            target_series.append(np.concatenate(parts))
        states = reader.read_states()
        if states:
            grip = np.stack([np.asarray(s.gripper_state, dtype=np.float64) for s in states])
            for col in range(grip.shape[1]):
                state_series.setdefault(col, []).append(grip[:, col])

    channels = [channel_stats("action.gripper_target", target_series)]
    channels += [
        channel_stats(f"state.gripper[{col}]", series)
        for col, series in sorted(state_series.items())
    ]
    scored = channels[0]
    passed, reasons = _findings(channels, scored, any_channel=False)
    return GripperAuditReport(
        root=str(root_path),
        source_kind="wam",
        num_episodes=len(dirs),
        channels=tuple(channels),
        scored_channel=scored.name,
        passed=passed,
        reasons=reasons,
        thresholds=_thresholds(),
    )


# --- raw LeRobot snapshots ---------------------------------------------------------------------

# Fallback slices for a snapshot without meta/modality.json (GR00T G1 43-dim layout). Used only
# when the layout cannot be read, and the report says so via source_kind rather than pretending
# the slices were verified.
_FALLBACK_GROUPS: dict[str, tuple[int, int]] = {"left_hand": (29, 36), "right_hand": (36, 43)}

_HAND_GROUP_HINTS = ("hand", "gripper")


def _load_modality_groups(source: Path) -> tuple[dict[str, dict[str, tuple[int, int]]], bool]:
    """``{'state'|'action': {group: (start, end)}}`` from meta/modality.json; flag = verified.

    Groups carrying an ``original_key`` live in their own parquet column (LeRobot's effort/command
    channels) and are skipped: their indices are offsets into that column, not into the packed
    state/action vector, so slicing the packed vector with them would silently read the wrong
    joints.
    """
    path = source / "meta" / "modality.json"
    if not path.is_file():
        return {"state": dict(_FALLBACK_GROUPS), "action": dict(_FALLBACK_GROUPS)}, False
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, tuple[int, int]]] = {}
    for section in ("state", "action"):
        groups: dict[str, tuple[int, int]] = {}
        for name, entry in (raw.get(section) or {}).items():
            if not isinstance(entry, dict) or "original_key" in entry:
                continue
            if not any(h in name.lower() for h in _HAND_GROUP_HINTS):
                continue
            groups[name] = (int(entry["start"]), int(entry["end"]))
        out[section] = groups
    if not out["state"] and not out["action"]:
        return {"state": dict(_FALLBACK_GROUPS), "action": dict(_FALLBACK_GROUPS)}, False
    return out, True


def _lerobot_episode_files(source: Path, max_episodes: int | None) -> list[Path]:
    files = sorted((source / "data").rglob("episode_*.parquet"))
    return files if max_episodes is None else files[:max_episodes]


def _active_joint(episodes: Sequence[np.ndarray]) -> int:
    """Index of the group's most active joint, chosen ONCE over the whole audited set.

    Ranked on the MEAN PER-EPISODE peak-to-peak, the same rule
    ``convert_lerobot_g1.fit_hand_affine`` uses to pick the active hand, and for the same reason:
    a joint that merely rests at a different angle in each session has a large global range and
    no motion inside any episode, and ranking on the global range would elect it.
    """
    if not episodes:
        return 0
    spread = np.mean(
        [block.max(axis=0) - block.min(axis=0) for block in episodes], axis=0, dtype=np.float64
    )
    return int(np.argmax(spread))


def audit_lerobot_dataset(
    source: str | Path, *, max_episodes: int | None = None
) -> GripperAuditReport:
    """Audit a raw LeRobot v2.1 snapshot, before any WAM conversion has touched it.

    This is the mode that answers "is the grasp in the source?" — run it whenever a converted set
    fails, because a FAIL there is equally consistent with a bad dataset and a bad converter, and
    only this mode tells the two apart.

    Every hand/gripper group named in ``meta/modality.json`` is reported for both
    ``observation.state`` and ``action``, each as two channels: the 7-joint mean (what our
    converter uses) and the single most active joint (what the mean averages away), named
    ``…max_joint[i]`` after the index it settled on. Both are mapped to [0, 1] by a DATASET-level
    affine, and ``i`` is likewise chosen ONCE over the whole set (:func:`_active_joint`).

    Read the transition columns, not ``p2p_global``: raw joints have no natural [0, 1] scale, so
    the affine defines the global range to BE 1.0 for any channel that moved at all — including a
    frozen hand whose micrometre of drift it stretches across the whole range. That is why the
    verdict has four clauses and not one; a dead hand clears the range clause here and still
    fails, on 0.00 debounced transitions.
    """
    import pyarrow.parquet as pq

    source_path = Path(source)
    groups, verified = _load_modality_groups(source_path)
    files = _lerobot_episode_files(source_path, max_episodes)

    column_for = {"state": "observation.state", "action": "action"}
    blocks: dict[str, list[np.ndarray]] = {}
    for path in files:
        wanted = sorted({column_for[s] for s in groups if groups[s]})
        table = pq.read_table(path, columns=wanted)
        for section, section_groups in groups.items():
            if not section_groups:
                continue
            col = column_for[section]
            arr = np.stack(table[col].to_numpy(zero_copy_only=False)).astype(np.float64)
            for name, (start, end) in section_groups.items():
                if end > arr.shape[1]:
                    continue
                blocks.setdefault(f"{section}.{name}", []).append(arr[:, start:end])

    # The whole group is kept per episode and reduced only afterwards, because both reductions
    # are DATASET-level: the mean is per-sample, but the "most active joint" is a choice about
    # the group and picking it per episode would report a different physical joint in different
    # episodes and concatenate them into one channel (the module docstring forbids exactly this
    # for the affine; the same argument applies to the column selection that precedes it).
    raw: dict[str, list[np.ndarray]] = {}
    for key, episodes in blocks.items():
        raw[f"{key}.mean"] = [block.mean(axis=1) for block in episodes]
        joint = _active_joint(episodes)
        raw[f"{key}.max_joint[{joint}]"] = [block[:, joint] for block in episodes]

    channels: list[GripperChannelStats] = []
    for name in sorted(raw):
        offset, span = dataset_affine(raw[name])
        channels.append(channel_stats(name, [apply_affine(s, offset, span) for s in raw[name]]))
    if not channels:
        channels = [channel_stats("action.none", [])]
    scored = _pick_scored(channels)
    passed, reasons = _findings(channels, scored, any_channel=True)
    return GripperAuditReport(
        root=str(source_path),
        source_kind="lerobot" if verified else "lerobot(unverified-layout)",
        num_episodes=len(files),
        channels=tuple(channels),
        scored_channel=scored.name,
        passed=passed,
        reasons=reasons,
        thresholds=_thresholds(),
    )
