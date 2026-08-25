#!/usr/bin/env python3
"""Apply ``T40_RULE_V13`` §3.1 to the pooled robot-mask area distribution.

**This script does not choose a bound and cannot write one.** V13 §3 is explicit that
``max_frame_fraction`` stays a human decision with a written rationale; what a script can do
is assemble the evidence that decision is required to cite, and refuse to go further.
So this reports candidate gaps and their attribution, and prints the §3.2 checklist with the
items only a person can fill marked as such.

WRITTEN AND COMMITTED BEFORE THE POOLED DISTRIBUTION EXISTED, which is the same ordering
argument V13 §1 makes about itself: after the numbers are visible nobody, including the author,
can prove the analysis was not fitted to them. The shards were still ``PENDING`` at the commit
that introduced this file.

THE ZERO SPIKE IS NOT THE GAP
-----------------------------
About a third of source frames carry **no** robot mask at all and are recorded as area fraction
exactly ``0.0`` (``empty_frame_fraction`` 0.331 on shard 0). A gap-finder run over the raw pooled
population therefore finds an enormous discontinuity between ``0.0`` and the smallest real mask
(~0.12) and reports it as "a measured gap". **That is not the gap V13 means.** V13 §3.1 step 2
asks for a discontinuity "separating a bulk (masks that are the robot) from a tail (masks that
are the scene)", and an empty mask is neither -- it is the *absence* of a measurement, which is
the whole subject of ``T40_RULE_V12``. An empty mask also has fraction 0.0 and so can never
exceed any bound, meaning it is untouched by this decision in either direction.

So the separation analysis runs over the NON-EMPTY population and the empty count is reported
beside it rather than mixed into it. Both populations are reported; neither is hidden.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

#: Gaps narrower than this are not discontinuities, they are the spacing of a continuum. Fixed
#: here rather than chosen later; it exists so "the largest gap" cannot be reported as a finding
#: when the largest gap is 1e-4 wide.
MIN_GAP_WIDTH = 0.01

#: A "tail" of one frame is an outlier, not a population. V13 §3.1 step 2 asks for a population.
MIN_TAIL_FRAMES = 2

#: Candidate cuts are only sought above this quantile of the non-empty population. The decision
#: is about an UPPER tail -- masks that have grounded on the scene are larger than the robot, not
#: smaller -- and a gap in the lower half is not evidence about it.
TAIL_SEARCH_FROM_QUANTILE = 0.50


def pooled_fractions(payload: dict) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    """Every per-frame fraction in the artifact, plus the same split per episode.

    Reads ``per_episode`` rather than ``measured``: the percentiles in ``measured`` are summary
    statistics and a gap is not visible in five numbers, which is the reason V13 §3.1 step 1 asks
    for "the per-frame fractions behind them".
    """
    per_episode: list[tuple[str, np.ndarray]] = []
    for entry in payload["per_episode"]:
        key = entry.get("episode") or entry.get("id") or entry.get("key") or "<unnamed>"
        per_episode.append((str(key), np.asarray(entry["area_fractions"], dtype=np.float64)))
    if not per_episode:
        raise SystemExit("per_episode is empty: nothing to pool")
    return np.concatenate([f for _, f in per_episode]), per_episode


def candidate_gaps(nonempty: np.ndarray, *, top: int = 8) -> list[dict]:
    """Discontinuities in the upper part of the non-empty population, widest first.

    A candidate is a pair of adjacent order statistics whose separation is at least
    :data:`MIN_GAP_WIDTH`, with at least :data:`MIN_TAIL_FRAMES` above it. Reporting several
    rather than one is deliberate: "the largest gap" is a statistic, and V13 asks a human to
    judge whether a gap separates two POPULATIONS. That judgement needs the alternatives visible.
    """
    values = np.unique(nonempty)
    if values.size < 2:
        return []
    floor = float(np.quantile(nonempty, TAIL_SEARCH_FROM_QUANTILE))
    out: list[dict] = []
    for lower, upper in zip(values[:-1], values[1:]):
        if lower < floor:
            continue
        width = float(upper - lower)
        if width < MIN_GAP_WIDTH:
            continue
        above = int(np.count_nonzero(nonempty > lower))
        if above < MIN_TAIL_FRAMES:
            continue
        out.append(
            {
                # The two edges V13 §3.1 step 3 requires a rationale to name.
                "bulk_edge_below": float(lower),
                "tail_edge_above": float(upper),
                "gap_width": width,
                "frames_above": above,
                "frames_above_fraction_of_nonempty": above / float(nonempty.size),
            }
        )
    out.sort(key=lambda g: g["gap_width"], reverse=True)
    return out[:top]


def tail_attribution(per_episode: list[tuple[str, np.ndarray]], cut: float) -> dict:
    """Which episodes the frames above ``cut`` come from.

    V13 §3.1 step 2: this is "the check that the tail is a *failure mode* and not simply the
    grasp frames of every episode". The distinction is visible here and nowhere else -- a tail
    spread thinly over all 402 episodes and a tail concentrated in three are the same percentile
    and completely different findings.
    """
    rows = []
    for key, fractions in per_episode:
        n_above = int(np.count_nonzero(fractions > cut))
        if n_above:
            rows.append(
                {
                    "episode": key,
                    "frames_above": n_above,
                    "episode_frames": int(fractions.size),
                    "share_of_episode": n_above / float(fractions.size),
                }
            )
    rows.sort(key=lambda r: r["frames_above"], reverse=True)
    total_above = sum(r["frames_above"] for r in rows)
    return {
        "cut": float(cut),
        "episodes_contributing": len(rows),
        "episodes_total": len(per_episode),
        "frames_above_total": total_above,
        "concentration_top_episode": (
            rows[0]["frames_above"] / total_above if rows and total_above else 0.0
        ),
        "per_episode": rows,
    }


def analyze(payload: dict) -> dict:
    pooled, per_episode = pooled_fractions(payload)
    empty_mask = pooled == 0.0
    nonempty = pooled[~empty_mask]
    gaps = candidate_gaps(nonempty)
    report = {
        "schema": "pr08-area-gap-analysis/1",
        "rule": "T40_RULE_V13 §3.1",
        "writes_a_bound": False,
        "source": {
            "git_commit": payload.get("git_commit"),
            "source_manifest_sha256": payload.get("source_manifest_sha256"),
            "prompt": payload.get("prompt"),
            "estimator": payload.get("estimator"),
        },
        "population": {
            "frames_total": int(pooled.size),
            "frames_empty_mask": int(empty_mask.sum()),
            "empty_mask_fraction": float(empty_mask.mean()),
            "frames_nonempty": int(nonempty.size),
            "episodes": len(per_episode),
        },
        # Reported so a reader can see the zero spike was excluded on purpose and by how much
        # it would have distorted the search. See this module's docstring.
        "why_nonempty_only": (
            "empty masks are recorded as 0.0, are the subject of T40_RULE_V12 rather than of an "
            "area bound, and cannot exceed any bound; including them manufactures a spurious "
            "discontinuity between 0.0 and the smallest real mask"
        ),
        "nonempty_distribution": {
            "min": float(nonempty.min()),
            "median": float(np.median(nonempty)),
            "p95": float(np.percentile(nonempty, 95)),
            "p99": float(np.percentile(nonempty, 99)),
            "max": float(nonempty.max()),
        },
        "candidate_gaps": gaps,
    }
    if gaps:
        report["attribution_for_widest_gap"] = tail_attribution(
            per_episode, gaps[0]["bulk_edge_below"]
        )
        report["verdict"] = "candidate gap(s) found — V13 §3.1 step 3; a human places the bound"
    else:
        # V13 §3.3 is the load-bearing half and this is the branch that reaches it.
        report["verdict"] = (
            "NO SEPARABLE GAP — V13 §3.3 applies: no bound may be committed under this rule. "
            "The honest outcomes are (a) leave max_frame_fraction null, (b) register a different "
            "instrument, (c) establish the failure population deliberately. V13 does not choose."
        )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifact", type=pathlib.Path, help="merged ROBOT_MASK_AREA artifact")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)

    payload = json.loads(args.artifact.read_text())
    if payload.get("measurement_qualified") is False:
        # load_area_bound refuses such an artifact by name; analysing it would produce a bound
        # rationale citing a distribution nothing may sit above. V13 §3.4's last bullet.
        print(
            "REFUSED: artifact carries measurement_qualified: false — "
            f"{payload.get('measurement_disqualified_reasons')}",
            file=sys.stderr,
        )
        return 2

    report = analyze(payload)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)

    print("\n=== T40_RULE_V13 §3.2 — what a bound_rationale must contain ===", file=sys.stderr)
    for item, who in [
        ("the two edges of the gap, as numbers", "above, per candidate"),
        ("frames and EPISODES above the bound, absolute and fractional", "above, in attribution"),
        ("whether those frames were LOOKED AT, and what they were", "*** A PERSON. Not here. ***"),
        ("the commit and source_manifest_sha256 measured over", "above, in source"),
        ("that the bound was never validated against a known-bad mask", "*** A PERSON. ***"),
    ]:
        print(f"  - {item}\n      -> {who}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
