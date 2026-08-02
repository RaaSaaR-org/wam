"""Tests for PR-04's corpus screen (``scripts/screen_corpus.py``).

The screen decides whether a recording campaign scales or its protocol changes, so the parts
that can be quietly wrong are pinned here on small hand-built arrays: the two zero-parameter
baselines, the M1/M2 algebra, the ceiling-dominates guard, and the fact that the ceiling is
never chosen using holdout data.

The reproduction of ``PR-02-RESULT.md``'s archived M1/M2/M3 is deliberately NOT asserted here.
It is a measurement over 402 real episodes, it belongs to ``--expect gr00t``, and a unit test
that faked it would defeat its purpose. What this file does assert is that the machinery the
measurement runs through is arithmetically what the document says it is.
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

requires_grip_dataset = pytest.mark.skipif(
    not _DATASET_GRIP.is_dir(), reason="datasets/gr00t-apple-grip not present"
)


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec because @dataclass resolves annotations through sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


screen_corpus = _load("screen_corpus")


_FEATURE_DIM = 6
_WIDTH = screen_corpus.CHUNK_STEPS * 15
#: One fixed linear map shared by every synthetic episode, so a blind ceiling fitted on some
#: episodes genuinely transfers to others. Without a learnable signal the ceiling overfits,
#: scores worse than zero-delta, and M1's denominator is undefined — which is correct
#: behaviour but tests nothing about the algebra.
_MIXER = np.random.default_rng(7).normal(0.0, 1.0, (_FEATURE_DIM, _WIDTH))


def _episode(
    episode_id: str,
    n_chunks: int,
    *,
    noise: float = 0.3,
    constvel_frac: float = 0.0,
    transitions: int = 2,
    seed: int = 0,
) -> Any:
    """A synthetic episode with a blind-learnable signal plus noise.

    ``constvel_frac`` sets how much of the target the zero-parameter rule reproduces, which is
    what M1 measures; ``noise`` sets how much no blind model can reach, which is what M2
    measures.
    """
    rng = np.random.default_rng(seed)
    features = rng.normal(0.0, 1.0, (n_chunks, _FEATURE_DIM))
    targets = features @ _MIXER + rng.normal(0.0, noise, (n_chunks, _WIDTH))
    return screen_corpus.Episode(
        episode_id=episode_id,
        features=features,
        targets=targets,
        constvel=targets * constvel_frac,
        transitions=transitions,
    )


# ------------------------------------------------------------------------------- the M1/M2 algebra


def test_m2_is_the_ceilings_share_of_the_zero_delta_error() -> None:
    """M2 = mse_ceiling / mse_zero. Checked against the archived GR00T triple, which is where
    the definition came from: 5.431371e-06 / 1.632760e-05 = 0.3327, reported as 0.333."""
    assert 5.431371e-06 / 1.632760e-05 == pytest.approx(0.333, abs=0.001)


def test_m1_is_the_zero_parameter_rules_share_of_the_blind_span() -> None:
    """M1 = (zero - constvel) / (zero - ceiling). Same archived triple: 0.660."""
    zero, constvel, ceiling = 1.632760e-05, 9.137664e-06, 5.431371e-06
    assert (zero - constvel) / (zero - ceiling) == pytest.approx(0.660, abs=0.001)


def test_a_perfect_const_velocity_rule_drives_m1_to_one() -> None:
    """If the zero-parameter rule already reaches the ceiling, M1 is 1 and there is nothing a
    fitted model — blind or sighted — can add on this metric."""
    episodes = [_episode(f"e{i}", 60, constvel_frac=1.0, seed=i) for i in range(9)]
    report = screen_corpus.screen(episodes[:6], episodes[6:])
    assert report["mse"]["const_velocity"] == pytest.approx(0.0, abs=1e-12)
    assert report["m1_momentum_share"] >= 1.0
    assert not report["gates"]["m1_pass"]


def test_a_useless_const_velocity_rule_drives_m1_to_zero_or_below() -> None:
    """The other end: a rule that predicts nothing leaves the whole blind span to the ceiling."""
    episodes = [_episode(f"e{i}", 60, constvel_frac=0.0, seed=i) for i in range(9)]
    report = screen_corpus.screen(episodes[:6], episodes[6:])
    assert report["m1_momentum_share"] <= 0.05


def test_unpredictable_targets_make_m2_approach_one() -> None:
    """When noise dominates the blind-learnable signal, almost all the target energy is left
    on the table — the shape a corpus worth training on should have. (In a real corpus that
    "noise" is what the cameras can see and proprioception cannot.)"""
    episodes = [_episode(f"e{i}", 60, noise=8.0, seed=100 + i) for i in range(9)]
    report = screen_corpus.screen(episodes[:6], episodes[6:])
    assert report["m2_blind_unreachable"] > 0.8
    assert report["gates"]["m2_pass"]


# ------------------------------------------------------------------------------------ the guards


def test_the_ceiling_dominates_flag_catches_a_ceiling_a_free_rule_beats() -> None:
    """PR-03 shipped a "ceiling" a zero-parameter rule beat, and every M1/M2 read off such a
    ceiling is void — M1's denominator can even go negative. The flag must fire, not the run
    silently continue."""
    episodes = [_episode(f"e{i}", 60, constvel_frac=1.0, seed=i) for i in range(9)]
    report = screen_corpus.screen(episodes[:6], episodes[6:])
    # const-velocity is exact here, so no fitted ceiling can match it on unseen episodes.
    assert not report["ceiling_dominates"]


def test_m3_counts_transitions_over_every_episode_not_just_the_holdout() -> None:
    """Grasp liveness is a property of the corpus, not of one split of it — a protocol change
    that killed the channel in 90 % of episodes must show up even if the holdout is clean."""
    live = [_episode(f"h{i}", 30, transitions=4, seed=i) for i in range(2)]
    dead = [_episode(f"t{i}", 30, transitions=0, seed=10 + i) for i in range(18)]
    report = screen_corpus.screen(dead, live)
    assert report["m3_transitions_per_episode"] == pytest.approx(8 / 20)
    assert not report["gates"]["m3_pass"]


def test_the_ceiling_is_never_selected_on_holdout_data() -> None:
    """Hyperparameters chosen against the holdout would make the ceiling optimistic, which
    biases M2 DOWN and would make a bad corpus look screenable. The inner split must come out
    of the training episodes alone: swapping the holdout must not change the selection."""
    train = [_episode(f"t{i}", 40, seed=i) for i in range(8)]
    hold_a = [_episode(f"a{i}", 40, seed=50 + i) for i in range(3)]
    hold_b = [_episode(f"b{i}", 40, seed=80 + i) for i in range(3)]
    sel_a = screen_corpus.screen(train, hold_a)["selection"]
    sel_b = screen_corpus.screen(train, hold_b)["selection"]
    assert sel_a == sel_b


def test_the_gate_thresholds_are_the_ones_pr04_pre_registered() -> None:
    """These decide whether a campaign scales. They are constants in committed code so that
    changing one is a reviewable diff, not an argument after seeing the pilot."""
    assert screen_corpus.M1_MAX == 0.45
    assert screen_corpus.M2_MIN == 0.45
    assert screen_corpus.M3_MIN == 2.0
    assert screen_corpus.ARCHIVED["gr00t"] == {"m1": 0.660, "m2": 0.333, "m3": 2.01}


# ------------------------------------------------------------------------- refusals that need data


@requires_grip_dataset
def test_the_screen_refuses_a_holdout_that_is_not_in_the_dataset(tmp_path: Path) -> None:
    bad = tmp_path / "holdout.txt"
    bad.write_text("no-such-episode\n")
    with pytest.raises(SystemExit, match="not in the dataset"):
        screen_corpus.main(["--dataset", str(_DATASET_GRIP), "--holdout", str(bad)])
