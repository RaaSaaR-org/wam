"""Tests for PR-03's blind control suite (``scripts/bench_grasp_anticipation.py``).

The script decides whether ~20-40 GPU-hours get spent, so the parts that can be wrong quietly are
pinned here: where a flip is judged to be, which steps count as post-flip, and whether the
bootstrap resamples episodes rather than steps. All of it is arithmetic over small hand-built
arrays and needs no dataset.

The one thing NOT asserted here is agreement with ``PR-01-GRIPPER.md``'s archived table. That
document's numbers came from a script that was never committed, so agreement is a *measurement*
this file cannot fake into existence — it is reported by the gate itself, with the ambiguity
spelled out in the script docstring.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASET_GRIP = _REPO_ROOT / "datasets" / "gr00t-apple-grip"
_SPLITS = _REPO_ROOT / "configs" / "splits"

requires_grip_dataset = pytest.mark.skipif(
    not _DATASET_GRIP.is_dir(), reason="datasets/gr00t-apple-grip not present"
)


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec because @dataclass resolves annotations through sys.modules; without
    # this the Episode dataclass fails to build at import time.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = _load("bench_grasp_anticipation")


def _episode(targets: np.ndarray, latched_start: np.ndarray, latched_steps: np.ndarray) -> Any:
    return bench.Episode(
        episode_id="e",
        features=np.zeros((len(targets), 1)),
        targets=targets,
        observed=np.zeros(len(targets)),
        velocity=np.zeros(len(targets)),
        phase=np.zeros(len(targets)),
        latched_start=latched_start,
        latched_steps=latched_steps,
    )


# ------------------------------------------------------------------ the latch, shared with T-31


def test_latched_states_and_debounced_transitions_cannot_disagree() -> None:
    """PR-03's metric and T-31's admissibility gate must mean the same thing by "a grasp".

    ``debounced_transitions`` is defined in terms of ``latched_states``; this asserts the property
    that refactor exists to guarantee, on a series that exercises the dead band in both
    directions and dithers inside it.
    """
    from wam.evaluation.gripper import debounced_transitions, latched_states

    series = np.array([0.2, 0.5, 0.52, 0.9, 0.55, 0.45, 0.8, 0.1, 0.5, 0.05])
    latch = latched_states(series)
    decided = latch[latch >= 0]
    assert int((np.diff(decided) != 0).sum()) == debounced_transitions(series)


def test_latched_states_forward_fills_the_dead_band_and_marks_the_lead_in() -> None:
    """Samples inside the band keep the previous state; samples before any decision are -1."""
    from wam.evaluation.gripper import latched_states

    latch = latched_states(np.array([0.5, 0.52, 0.9, 0.55, 0.45, 0.1]))
    assert latch.tolist() == [-1, -1, 1, 1, 1, 0]


# ------------------------------------------------------------------------ the three definitions


def test_episode_latch_uses_the_chunk_start_as_the_step_before() -> None:
    """A chunk whose very first target step already differs from the robot's current state flips
    at step 0. Without the carry-in that transition is invisible, which is the whole difference
    between this definition and ``label-steps``."""
    steps = np.ones((1, 16), dtype=np.int8)
    ep = _episode(np.ones((1, 16)), np.array([0], dtype=np.int8), steps)
    assert bench.flips_episode_latch(ep).tolist() == [0]
    assert bench.flips_label_steps(ep).tolist() == [-1]


def test_label_steps_finds_a_flip_between_target_steps() -> None:
    steps = np.array([[0] * 5 + [1] * 11], dtype=np.int8)
    ep = _episode(np.ones((1, 16)), np.array([0], dtype=np.int8), steps)
    assert bench.flips_label_steps(ep).tolist() == [5]


def test_self_contained_ignores_context_and_needs_both_levels_inside_the_chunk() -> None:
    """A chunk that is entirely closed has no transition of its own, however it got there."""
    closed = np.full((1, 16), 0.9)
    ep = _episode(closed, np.array([0], dtype=np.int8), np.ones((1, 16), dtype=np.int8))
    assert bench.flips_self_contained(ep).tolist() == [-1]
    assert bench.flips_episode_latch(ep).tolist() == [0]


def test_a_flip_index_skips_undecided_samples_rather_than_counting_them() -> None:
    """Dead-band samples are not evidence of a transition, and they do not shift the index.

    The flip is reported where the channel next becomes DECISIVE, not where it first left the
    previous level — otherwise every dithering sample would advance the index and the post-flip
    window would start early, on steps the demonstrator had not yet committed to.
    """
    assert bench._first_change(np.array([[1, -1, -1, 0]], dtype=np.int8)).tolist() == [3]
    assert bench._first_change(np.array([[-1, -1, 1, 1]], dtype=np.int8)).tolist() == [-1]


# ------------------------------------------------------------------------------------- the masks


def test_postflip_and_preflip_masks_partition_a_transition_chunk() -> None:
    """Every step of a transition chunk is exactly one of pre-flip or post-flip."""
    flip = np.array([5])
    post, pre = bench.postflip_mask(flip), bench.preflip_mask(flip)
    assert not (post & pre).any()
    assert (post | pre).all()
    assert post.sum() == 11 and pre.sum() == 5


def test_a_chunk_without_a_flip_contributes_no_steps_to_any_window() -> None:
    """Non-transition chunks are the 92 % of the holdout the metric deliberately does not score;
    leaking them in would turn post-flip accuracy back into the full-holdout momentum metric."""
    flip = np.array([-1])
    assert not bench.postflip_mask(flip).any()
    assert not bench.preflip_mask(flip).any()
    assert not bench.window_mask(flip).any()


def test_the_k_to_k3_window_is_four_steps_and_clipped_by_the_chunk_end() -> None:
    assert bench.window_mask(np.array([5])).sum() == 4
    assert bench.window_mask(np.array([14])).sum() == 2  # steps 14, 15 — no reading past the end


# --------------------------------------------------------------------------------- the bootstrap


def test_bootstrap_halfwidth_is_zero_when_every_episode_is_perfect() -> None:
    """No between-episode variance, so no interval — a guard against a resampler that is
    resampling something other than what it claims."""
    assert bench.bootstrap_halfwidth([10.0, 10.0, 10.0], [10.0, 10.0, 10.0]) == 0.0


def test_bootstrap_resamples_episodes_not_steps() -> None:
    """The interval must be driven by disagreement BETWEEN episodes.

    Two datasets with identical step-level accuracy (50 %) but opposite structure: one where every
    episode is half right, one where half the episodes are all right and half all wrong. A
    step-level bootstrap gives them the same interval; an episode-level one gives the second a far
    wider one, which is the whole reason PR-01-GRIPPER's CI is +-7.6 and not +-4.0.
    """
    uniform = bench.bootstrap_halfwidth([5.0] * 8, [10.0] * 8)
    clustered = bench.bootstrap_halfwidth([10.0, 0.0] * 4, [10.0] * 8)
    assert clustered > 3 * uniform


def test_bootstrap_ignores_episodes_with_no_postflip_steps() -> None:
    """Most episodes contribute nothing; counting them as zero-accuracy would invent a result."""
    with_empties = bench.bootstrap_halfwidth([5.0, 0.0, 3.0], [10.0, 0.0, 10.0])
    without = bench.bootstrap_halfwidth([5.0, 3.0], [10.0, 10.0])
    assert with_empties == without


# ------------------------------------------------------------------------------ accuracy scoring


def test_accuracy_binarizes_both_sides_at_the_threshold() -> None:
    pred = np.array([[0.51, 0.49]])
    target = np.array([[0.99, 0.01]])
    assert bench._postflip_accuracy(pred, target, np.ones((1, 2), dtype=bool)) == 100.0


def test_accuracy_is_nan_rather_than_zero_when_nothing_is_scored() -> None:
    """An empty window is "not measured", not "measured and wrong" — the difference decides
    whether a definition silently contributes a 0 % to a comparison."""
    value = bench._postflip_accuracy(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2), bool))
    assert np.isnan(value)


# --------------------------------------------------------------------- refusals that need data


@requires_grip_dataset
def test_the_gate_refuses_to_score_episodes_it_trained_on() -> None:
    """``--restrict`` outside ``--holdout`` would fit on an episode and then score it."""
    with pytest.raises(SystemExit, match="not a subset"):
        bench.main(
            [
                "--dataset", str(_DATASET_GRIP),
                "--holdout", str(_SPLITS / "t18_holdout_episodes.txt"),
                "--restrict", str(_SPLITS / "pr03_holdout_150.txt"),
            ]
        )  # fmt: skip
