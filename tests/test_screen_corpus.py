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
        # Per gripper channel, as the real loader reports it: channel 0 live, channel 1 dead —
        # the shape of the corpus we have (T-31).
        transitions=np.asarray([transitions, 0], dtype=np.int64),
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


# --------------------------------------------------------- load_episode, on a synthetic corpus
#
# Added after adversarial review: nothing exercised load_episode, and three mutants inside it
# each flipped a PR-04 gate verdict on real data — a future-reading _lagged (the exact leak this
# screen exists to detect) turned an M1 FAIL into a PASS.


def _write_episode(
    root: Path,
    episode_id: str,
    *,
    n_frames: int = 80,
    dt_s: float = 1.0 / 30.0,
    live_hand: int = 0,
    gripper_dims: int = 2,
) -> None:
    """A minimal on-disk episode in the WAM format: states.parquet + actions.parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(abs(hash(episode_id)) % (2**32))
    q = np.cumsum(rng.normal(0.0, 0.01, (n_frames, 15)), axis=0)
    dq = np.diff(q, axis=0, prepend=q[:1]) / dt_s
    grip = np.zeros((n_frames, gripper_dims))
    # One clean close-and-reopen on the live hand: two debounced transitions.
    grip[n_frames // 4 : 3 * n_frames // 4, live_hand] = 1.0

    (root / episode_id).mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "q": [row.tolist() for row in q],
                "dq": [row.tolist() for row in dq],
                "gripper_state": [row.tolist() for row in grip],
            }
        ),
        root / episode_id / "states.parquet",
    )

    chunk_idx, step_idx, targets = [], [], []
    n_chunks = -(-n_frames // screen_corpus.CHUNK_STEPS)  # ceil: the last one may be partial
    for chunk in range(n_chunks):
        for step in range(screen_corpus.CHUNK_STEPS):
            frame = chunk * screen_corpus.CHUNK_STEPS + step
            if frame + 1 >= n_frames:
                break
            chunk_idx.append(chunk)
            step_idx.append(step)
            targets.append((q[frame + 1] - q[frame]).tolist())
    pq.write_table(
        pa.table(
            {
                "chunk_idx": chunk_idx,
                "step_idx": step_idx,
                "targets": targets,
                "dt_s": [dt_s] * len(chunk_idx),
            }
        ),
        root / episode_id / "actions.parquet",
    )


def test_lagged_features_never_read_the_future() -> None:
    """The leak this whole screen exists to detect, at its own front door.

    A blind predictor that can see past the chunk start is not blind: on real data, flipping
    ``_lagged`` to read forward instead of backward took M1 from 0.5989 to 0.3845 and turned an
    M1 gate FAIL into a PASS — a corpus whose blind baseline disqualifies it gets approved for
    scaling. A strictly increasing stream makes any forward read detectable by value alone.
    """
    stream = np.arange(40, dtype=np.float64).reshape(40, 1)
    feats = screen_corpus._lagged(stream, 20)
    assert feats.max() <= stream[20].item(), "a feature came from after the chunk start"
    assert feats.min() >= 0.0  # clamped at the episode start, not wrapped
    assert list(feats) == [20.0, 19.0, 18.0, 17.0, 16.0, 14.0, 12.0, 8.0]
    # Clamping at the start must not invent negative frames.
    assert list(screen_corpus._lagged(stream, 1)) == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_load_episode_builds_const_velocity_from_dq_times_dt_s(tmp_path: Path) -> None:
    """The zero-parameter baseline M1 is measured against. Dropping the dt_s factor leaves the
    prediction 30x too large; on real data that alone drove M1 to -1482.79 and flipped its gate
    FAIL -> PASS, because M1's denominator is a difference that can change sign."""
    _write_episode(tmp_path, "e0", n_frames=80, dt_s=1.0 / 30.0)
    episode = screen_corpus.load_episode(tmp_path, "e0")
    assert episode is not None

    states = __import__("pyarrow.parquet", fromlist=["read_table"]).read_table(
        tmp_path / "e0" / "states.parquet", columns=["dq"]
    ).to_pydict()
    dq = np.asarray([np.asarray(v) for v in states["dq"]])
    # Chunk 1, not 0: the recorder writes dq[0] = 0, so chunk 0 cannot tell a missing dt_s
    # factor from a present one and would make the discriminator below vacuous.
    chunk, start = 1, screen_corpus.CHUNK_STEPS
    assert np.abs(dq[start]).max() > 0.0
    expected = np.tile(dq[start] * (1.0 / 30.0), (screen_corpus.CHUNK_STEPS, 1)).reshape(-1)
    np.testing.assert_allclose(episode.constvel[chunk], expected, rtol=1e-12)
    # And it is genuinely scaled: the raw dq would be 30x bigger.
    assert not np.allclose(episode.constvel[chunk], np.tile(dq[start], 16), rtol=1e-6)


def test_m3_finds_the_live_gripper_channel_whichever_one_it_is(tmp_path: Path) -> None:
    """A rig whose moving hand is channel 1 is a healthy rig. Hardcoding channel 0 reported
    M3 = 0.00 for it, which PR-04 maps to verdict C — "the recording or conversion killed the
    channel, fix the pipeline" — on data whose channel is demonstrably alive. That is the exact
    inversion of the failure G3 exists to catch."""
    for hand in (0, 1):
        root = tmp_path / f"hand{hand}"
        for i in range(6):
            _write_episode(root, f"e{i}", live_hand=hand)
        episodes = [screen_corpus.load_episode(root, f"e{i}") for i in range(6)]
        report = screen_corpus.screen(episodes[:4], episodes[4:])
        assert report["m3_active_hand"] == hand
        assert report["m3_transitions_per_episode"] == pytest.approx(2.0)
        assert report["gates"]["m3_pass"]


def test_a_trailing_partial_chunk_is_not_scored(tmp_path: Path) -> None:
    """A short final chunk would be scored against 16 steps' worth of target energy while
    holding fewer, quietly deflating every MSE in the report."""
    import pyarrow.parquet as pq

    _write_episode(tmp_path, "e0", n_frames=88)  # 5 full chunks + a 7-step trailing one
    on_disk = pq.read_table(tmp_path / "e0" / "actions.parquet", columns=["chunk_idx"]).to_pydict()
    assert len(set(on_disk["chunk_idx"])) == 6, "the fixture must actually write a partial chunk"

    episode = screen_corpus.load_episode(tmp_path, "e0")
    assert episode is not None
    assert episode.targets.shape[1] == screen_corpus.CHUNK_STEPS * 15
    assert len(episode.targets) == 5  # the 6th is partial and is dropped


def test_the_json_artifact_has_no_bare_nan_token(tmp_path: Path) -> None:
    """PR-04 makes runs/pr04/pilot.json the recorded artifact of the gate decision, and a
    collapsed ceiling is exactly when M1 goes NaN — so the run most in need of inspection was
    the one no standards-conformant reader could parse."""
    safe = screen_corpus._json_safe({"a": float("nan"), "b": [1.0, float("inf")], "c": 2.0})
    assert safe == {"a": None, "b": [1.0, None], "c": 2.0}
    import json

    assert "NaN" not in json.dumps(safe)
