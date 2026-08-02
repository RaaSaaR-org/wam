"""Tests for `scripts/bench_ridge_baseline.py` — the blind bars a visual model has to beat.

The script's headline claim is that 7 920 linear parameters fitted on proprioception alone score
6.330899e-06 on the T-16 holdout against the deployed Wan-5B+LoRA model's 1.112983e-05, and that a
blind NONLINEAR regressor on the same 32 dims goes further still, to 5.431371e-06. Numbers that
large only mean something if the fits themselves are trustworthy, so the tests here are not "does
it print a number". They are the four ways a baseline like this is usually wrong:

  it does not fit    On data where the target IS an exact linear function of the state, the ridge
                     has to recover it and score ~0. Without this calibration a broken solve looks
                     identical to "the state does not predict the action". The ceiling gets the
                     matching calibration in the other direction: on a target that is a NONLINEAR
                     function of the state it has to beat the linear ridge decisively, or it is
                     just an expensive way to redo one.
  it leaks           The mutation that matters, and it now has two shapes. For the ridge: on data
                     where the target is INDEPENDENT of the state the holdout MSE must land at the
                     zero-delta baseline and NOT below it, reinforced by a DELIBERATELY leaked
                     split that scores detectably better. For the ceiling: the hyperparameters must
                     be chosen on an inner split of TRAIN only — the guard raises when a holdout
                     episode reaches the search, and the leak control here chooses the width,
                     bandwidth and lambda ON THE HOLDOUT and comes out four orders of magnitude
                     "better", which is what an unpoliced ceiling would be worth.
  it normalises on   Train and holdout are built with different means and the holdout must come
  the union          out transformed by the TRAIN statistics — visibly off-centre, not re-centred
                     using knowledge of itself.
  it silently        A zero-variance state dim standardizes to 0/0 and, kept raw, duplicates the
  degenerates        bias column. Both are dropped, and the rank deficiency they would have caused
                     is asserted directly so the drop is shown to be necessary.

Plus two contract tests. The pairing rule must be `build_eval_pairs`' rule, checked by running both
over the same episode and comparing row for row — that is what licenses printing the ridge's MSE
next to a model's on the same line. And `check_archived` must actually refuse a moved number, since
it is the only thing standing between a refactor and three published write-ups quietly becoming
claims about data nobody measured.

Nothing here touches `datasets/gr00t-apple-full` or `runs/`. The episodes are synthesized in
`tmp_path` with the repo's own `EpisodeWriter` (never a hand-rolled parquet writer), and — apart
from the one test that cross-checks against `build_eval_pairs` — they carry no frames at all,
because the baseline never opens a camera and encoding video for it would be paying for pixels
that are the whole point of not being read. The ceiling is exercised at a toy grid (32/128 random
features, not 4096/8192): what is under test is the SEARCH DISCIPLINE, which is identical at both
scales, and a 8193-wide solve per grid point would put minutes into a unit test to re-measure a
number the script already prints.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wam.data import EpisodeWriter
from wam.interfaces.schema import (
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    IMUState,
    RobotState,
    ValidityMask,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Small enough to write in milliseconds, big enough that the design matrix is over-determined.
NUM_JOINTS = 3
GRIPPER_DIMS = 2
CHUNK_STEPS = 4
STATE_DIM = 2 * NUM_JOINTS + GRIPPER_DIMS  # 8
TARGET_DIM = CHUNK_STEPS * NUM_JOINTS  # 12
NUM_PARAMETERS = (STATE_DIM + 1) * TARGET_DIM  # 108, the toy-scale 7920
FPS = 20.0
DT = 1.0 / FPS
CHUNKS_PER_EPISODE = 20

TRAIN_EPISODES = tuple(f"ep-tr-{i:02d}" for i in range(8))
HOLDOUT_EPISODES = tuple(f"ep-ho-{i:02d}" for i in range(3))

#: A toy version of the ceiling's grid. 32/128 random features instead of 4096/8192, because what
#: is under test is where the hyperparameters are CHOSEN, and that is the same code at both scales.
CEILING_WIDTHS = (32, 128)
CEILING_GAMMAS = (0.05, 0.5)
CEILING_LAMBDAS = (1e-2, 1.0, 1e5)
CEILING_VAL_EPISODES = 2

SPEC = CanonicalSpaceSpec(
    joint_names=tuple(f"j{i}" for i in range(NUM_JOINTS)),
    gripper_dims=GRIPPER_DIMS,
)
ZERO_IMU = IMUState(
    orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
    angular_velocity=np.zeros(3, dtype=np.float32),
    linear_acceleration=np.zeros(3, dtype=np.float32),
)


def _load(name: str) -> Any:
    """`test_rescore_archived`'s loader, plus the `sys.modules` registration `@dataclass` needs
    to resolve its own annotations on a module imported by path."""
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rb = _load("bench_ridge_baseline")


# -- synthesis ---------------------------------------------------------------------------------


def _write_episode(
    root: Path,
    episode_id: str,
    states: np.ndarray,
    targets: np.ndarray,
    *,
    with_frames: bool = False,
    chunk_lengths: tuple[int, ...] | None = None,
) -> Path:
    """One WAM episode: row ``i`` of ``states`` at ``t=i*DT``, with a chunk commanded at that
    same timestamp. ``targets`` is [n, chunk_steps, num_joints].

    Frames are omitted by default — the baseline never decodes a camera, so recording one would
    only slow the fixture down. ``with_frames`` exists for the single test that cross-checks the
    pairing rule against ``build_eval_pairs``, which does need an image to build an Observation.

    ``chunk_lengths`` overrides the per-chunk horizon, so the skip-short/truncate-long contract
    can be exercised with the writer the rest of the repo uses rather than a fixture of its own.
    """
    episode_dir = root / episode_id
    with EpisodeWriter(episode_dir, episode_id, SPEC, FPS, "pick the apple") as writer:
        for i, (x, y) in enumerate(zip(states, targets)):
            ts = round(i * DT * 1e9)
            if with_frames:
                writer.add_frame("ego", np.full((8, 8, 3), i % 256, dtype=np.uint8), ts)
            writer.add_state(
                RobotState(
                    timestamp_ns=ts,
                    q=np.asarray(x[:NUM_JOINTS], dtype=np.float32),
                    dq=np.asarray(x[NUM_JOINTS : 2 * NUM_JOINTS], dtype=np.float32),
                    imu=ZERO_IMU,
                    gripper_state=np.asarray(x[2 * NUM_JOINTS :], dtype=np.float32),
                    validity=ValidityMask(q=True, dq=True, imu=False, gripper=True),
                )
            )
            steps = CHUNK_STEPS if chunk_lengths is None else chunk_lengths[i]
            chunk_targets = np.asarray(y, dtype=np.float32)
            if steps <= CHUNK_STEPS:
                chunk_targets = chunk_targets[:steps]
            else:
                pad = np.zeros((steps - CHUNK_STEPS, NUM_JOINTS), dtype=np.float32)
                chunk_targets = np.concatenate([chunk_targets, pad])
            writer.add_action(
                ActionChunk(
                    mode=ActionMode.JOINT_DELTA,
                    targets=chunk_targets,
                    gripper_target=np.zeros(steps, dtype=np.float32),
                    dt_s=DT,
                ),
                executed_prefix=1,
                timestamp_ns=ts,
            )
    return episode_dir


def _linear_dataset(root: Path, seed: int = 0) -> Path:
    """A dataset whose target is EXACTLY ``W @ state + b`` — the calibration case."""
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.05, (STATE_DIM, TARGET_DIM))
    bias = rng.normal(0.0, 0.01, TARGET_DIM)
    for episode_id in (*TRAIN_EPISODES, *HOLDOUT_EPISODES):
        states = rng.normal(0.0, 1.0, (CHUNKS_PER_EPISODE, STATE_DIM))
        targets = (states @ weights + bias).reshape(CHUNKS_PER_EPISODE, CHUNK_STEPS, NUM_JOINTS)
        _write_episode(root, episode_id, states, targets)
    return root


def _independent_dataset(root: Path, seed: int = 1) -> Path:
    """A dataset whose target is drawn independently of the state, zero-mean.

    Zero-mean matters: with a non-zero mean the ridge would legitimately learn the intercept and
    beat ``mean(target^2)`` without any leak, and the mutation would lose its meaning.
    """
    rng = np.random.default_rng(seed)
    for episode_id in (*TRAIN_EPISODES, *HOLDOUT_EPISODES):
        states = rng.normal(0.0, 1.0, (CHUNKS_PER_EPISODE, STATE_DIM))
        targets = rng.normal(0.0, 0.05, (CHUNKS_PER_EPISODE, CHUNK_STEPS, NUM_JOINTS))
        _write_episode(root, episode_id, states, targets)
    return root


def _memorizable_dataset(root: Path, seed: int = 2) -> tuple[Path, tuple[str, ...]]:
    """Per-episode SIGNATURE states and per-episode CONSTANT targets — memorizable, unlearnable.

    Within an episode the state is one random vector (plus a whisper of noise so the columns are
    not degenerate) and the target is one random constant. A linear map over 8 features plus a
    bias can therefore interpolate every episode it is shown exactly, and knows nothing at all
    about an episode it is not. That gap is what a leaked split converts into a small number.
    """
    rng = np.random.default_rng(seed)
    episodes = tuple(f"ep-sig-{i:02d}" for i in range(6))
    for episode_id in episodes:
        signature = rng.normal(0.0, 1.0, STATE_DIM)
        constant = rng.normal(0.0, 0.05, TARGET_DIM)
        states = signature[None, :] + rng.normal(0.0, 1e-3, (CHUNKS_PER_EPISODE, STATE_DIM))
        targets = np.repeat(constant[None, :], CHUNKS_PER_EPISODE, axis=0).reshape(
            CHUNKS_PER_EPISODE, CHUNK_STEPS, NUM_JOINTS
        )
        _write_episode(root, episode_id, states, targets)
    return root, episodes


def _split(table: Any, holdout_ids: tuple[str, ...]) -> tuple[np.ndarray, ...]:
    train_mask, holdout_mask = rb.split_by_episode(table, set(holdout_ids))
    train_x, train_y = table.select(train_mask)
    holdout_x, holdout_y = table.select(holdout_mask)
    return train_x, train_y, holdout_x, holdout_y


def _best_mse(table: Any, holdout_ids: tuple[str, ...], columns: np.ndarray | None = None) -> float:
    train_x, train_y, holdout_x, holdout_y = _split(table, holdout_ids)
    standardizer = rb.Standardizer.fit(train_x, columns=columns)
    return min(
        rb.fit_ridge(standardizer, train_x, train_y, holdout_x, holdout_y, lam).holdout_mse
        for lam in rb.DEFAULT_LAMBDAS
    )


# -- 1. it fits: calibration on an exactly linear target ----------------------------------------


def test_an_exactly_linear_target_is_recovered_almost_perfectly(tmp_path: Path) -> None:
    """Calibrates the fit. If the ridge cannot recover a relation that IS linear, then a large
    MSE elsewhere says nothing about the data — it says the solve is broken, and the two failures
    are indistinguishable from the printed number alone.

    The residual is not exactly zero and should not be asserted to be: the episodes are stored as
    float32 and every lambda > 0 shrinks the coefficients slightly. Both effects land many orders
    of magnitude below the signal, so the bound is stated relative to the zero-delta baseline —
    the ridge has to explain essentially all of the variance, not merely some of it.
    """
    table = rb.collect_chunks(_linear_dataset(tmp_path / "linear"), CHUNK_STEPS)
    _, _, _, holdout_y = _split(table, HOLDOUT_EPISODES)

    mse = _best_mse(table, HOLDOUT_EPISODES)
    zero = rb.zero_delta_mse(holdout_y)

    assert zero > 0.0
    assert mse < zero * 1e-6, f"ridge {mse:.3e} vs zero-delta {zero:.3e} — the fit is not linear"


def test_the_parameter_count_is_features_plus_bias_times_outputs(tmp_path: Path) -> None:
    """The headline is a parameter count as much as an MSE, so it has to be the real one: 7 920
    on the T-16 geometry is (32 state dims + 1 bias) x 240 outputs, and this is that arithmetic
    at toy scale rather than a constant repeated from the docstring."""
    table = rb.collect_chunks(_linear_dataset(tmp_path / "linear"), CHUNK_STEPS)
    train_x, train_y, holdout_x, holdout_y = _split(table, HOLDOUT_EPISODES)

    fit = rb.fit_ridge(rb.Standardizer.fit(train_x), train_x, train_y, holdout_x, holdout_y, 1e-2)

    assert (table.state_dim, table.target_dim) == (STATE_DIM, TARGET_DIM)
    assert fit.num_features == STATE_DIM
    assert fit.num_parameters == NUM_PARAMETERS


# -- 2. it does not leak: an independent target must score the zero-delta baseline ---------------


def test_a_target_independent_of_the_state_lands_on_the_zero_delta_baseline(
    tmp_path: Path,
) -> None:
    """The mutation that catches a leaking split.

    When the target carries no information about the state, the best any honest linear map can do
    is predict the train mean — which on zero-mean targets is the zero-delta baseline. Scoring
    BELOW it is not a better model, it is holdout information having reached the fit, and it is
    the single failure that would make the 1.76x headline meaningless.

    Slightly ABOVE is expected and allowed: 9 parameters per output fitted on 160 rows overfit a
    little, and that overfit shows up as holdout error. Only the downside is a bug.
    """
    table = rb.collect_chunks(_independent_dataset(tmp_path / "independent"), CHUNK_STEPS)
    _, _, _, holdout_y = _split(table, HOLDOUT_EPISODES)

    mse = _best_mse(table, HOLDOUT_EPISODES)
    zero = rb.zero_delta_mse(holdout_y)

    assert mse > zero * 0.9, f"ridge {mse:.4e} beat zero-delta {zero:.4e} on noise — leak"
    assert mse == pytest.approx(zero, rel=0.3)


# -- 3. the split is by episode, and the guard has teeth -----------------------------------------


def test_no_holdout_episode_reaches_the_train_matrix(tmp_path: Path) -> None:
    """Asserted over the row tags, not over the loop that built them. Chunks within an episode
    are overlapping views of one motion, so a row-level split would put near-copies of every
    holdout row into train and the baseline would be scoring its own training data."""
    table = rb.collect_chunks(_linear_dataset(tmp_path / "linear"), CHUNK_STEPS)

    train_mask, holdout_mask = rb.split_by_episode(table, set(HOLDOUT_EPISODES))

    train_ids = set(table.episode_ids[train_mask].tolist())
    holdout_ids = set(table.episode_ids[holdout_mask].tolist())
    assert holdout_ids == set(HOLDOUT_EPISODES)
    assert train_ids == set(TRAIN_EPISODES)
    assert not (train_ids & holdout_ids)
    assert int(train_mask.sum()) == len(TRAIN_EPISODES) * CHUNKS_PER_EPISODE
    assert int(holdout_mask.sum()) == len(HOLDOUT_EPISODES) * CHUNKS_PER_EPISODE
    assert not (train_mask & holdout_mask).any()


def test_a_leaked_split_scores_detectably_better(tmp_path: Path) -> None:
    """Proves the episode split is doing work, rather than passing because nothing was at stake.

    On per-episode-signature data a linear map can memorize every episode it sees and nothing
    about one it does not. Fitting on train only has to score far worse than fitting on train PLUS
    the holdout — if the two agree, the split is not separating anything and every other
    assertion in this file is satisfied by a broken implementation too.
    """
    root, episodes = _memorizable_dataset(tmp_path / "signature")
    table = rb.collect_chunks(root, CHUNK_STEPS)
    holdout = episodes[-2:]
    train_x, train_y, holdout_x, holdout_y = _split(table, holdout)

    honest = rb.fit_ridge(
        rb.Standardizer.fit(train_x), train_x, train_y, holdout_x, holdout_y, 1e-2
    ).holdout_mse
    # The leak, made explicitly: the holdout rows are inside the fit.
    leaked_x = np.vstack([train_x, holdout_x])
    leaked_y = np.vstack([train_y, holdout_y])
    leaked = rb.fit_ridge(
        rb.Standardizer.fit(leaked_x), leaked_x, leaked_y, holdout_x, holdout_y, 1e-2
    ).holdout_mse

    assert leaked < honest / 100.0, f"leaked {leaked:.4e} vs honest {honest:.4e} — no separation"
    assert honest > rb.zero_delta_mse(holdout_y) * 0.5


def test_a_holdout_id_the_dataset_does_not_have_is_refused(tmp_path: Path) -> None:
    """A moved or misspelled episode would quietly shrink the holdout, and the ridge would then
    be scored on fewer chunks than the model it is printed next to."""
    table = rb.collect_chunks(_linear_dataset(tmp_path / "linear"), CHUNK_STEPS)

    with pytest.raises(SystemExit, match="not in the dataset"):
        rb.split_by_episode(table, {*HOLDOUT_EPISODES, "ep-does-not-exist"})


def test_a_holdout_covering_everything_is_refused(tmp_path: Path) -> None:
    """Nothing to fit on is a different failure from a bad fit, and must not be reported as one."""
    table = rb.collect_chunks(_linear_dataset(tmp_path / "linear"), CHUNK_STEPS)

    with pytest.raises(SystemExit, match="nothing to fit on"):
        rb.split_by_episode(table, {*TRAIN_EPISODES, *HOLDOUT_EPISODES})


# -- 4. normalisation comes from train only ------------------------------------------------------


def test_normalisation_statistics_are_fitted_on_train_only(tmp_path: Path) -> None:
    """Train and holdout are built with deliberately different means, so a standardizer fitted on
    the union is distinguishable from one fitted on train.

    The assertion that matters is the second one: the transformed holdout must stay visibly
    OFF-CENTRE. A holdout centred at ~0 would mean its own mean had been used to centre it, which
    is the quiet leak — every downstream number optimistic by an amount nobody can bound after
    the fact.
    """
    root = tmp_path / "shifted"
    rng = np.random.default_rng(7)
    offset = 10.0
    for episode_id in TRAIN_EPISODES:
        states = rng.normal(0.0, 1.0, (CHUNKS_PER_EPISODE, STATE_DIM))
        _write_episode(
            root, episode_id, states, rng.normal(0, 0.05, (CHUNKS_PER_EPISODE, CHUNK_STEPS, 3))
        )
    for episode_id in HOLDOUT_EPISODES:
        states = rng.normal(offset, 1.0, (CHUNKS_PER_EPISODE, STATE_DIM))
        _write_episode(
            root, episode_id, states, rng.normal(0, 0.05, (CHUNKS_PER_EPISODE, CHUNK_STEPS, 3))
        )

    table = rb.collect_chunks(root, CHUNK_STEPS)
    train_x, _, holdout_x, _ = _split(table, HOLDOUT_EPISODES)
    standardizer = rb.Standardizer.fit(train_x)

    assert standardizer.mean == pytest.approx(train_x.mean(axis=0))
    assert standardizer.std == pytest.approx(train_x.std(axis=0))
    pooled = np.vstack([train_x, holdout_x])
    assert not np.allclose(standardizer.mean, pooled.mean(axis=0), atol=0.5)

    design = standardizer.design(holdout_x)
    expected = (holdout_x - train_x.mean(axis=0)) / train_x.std(axis=0)
    assert design[:, :-1] == pytest.approx(expected)
    assert np.all(design[:, -1] == 1.0)
    assert np.abs(design[:, :-1].mean(axis=0)).min() > 5.0, "the holdout was re-centred on itself"


# -- 5. degenerate state dims -------------------------------------------------------------------


def test_zero_variance_state_dims_are_dropped_without_a_singular_system(tmp_path: Path) -> None:
    """A constant state dim standardizes to 0/0, and kept raw it is a scalar multiple of the bias
    column — which is exactly what makes the normal equations singular at small lambda. Both dims
    below are constant on every episode; the rank check is what shows the drop is necessary
    rather than tidy.
    """
    root = tmp_path / "degenerate"
    rng = np.random.default_rng(11)
    for episode_id in (*TRAIN_EPISODES, *HOLDOUT_EPISODES):
        states = rng.normal(0.0, 1.0, (CHUNKS_PER_EPISODE, STATE_DIM))
        states[:, 1] = 0.5  # a constant that duplicates the bias column
        states[:, 4] = 0.0  # a constant that is identically zero
        targets = rng.normal(0.0, 0.05, (CHUNKS_PER_EPISODE, CHUNK_STEPS, NUM_JOINTS))
        _write_episode(root, episode_id, states, targets)

    table = rb.collect_chunks(root, CHUNK_STEPS)
    train_x, train_y, holdout_x, holdout_y = _split(table, HOLDOUT_EPISODES)
    standardizer = rb.Standardizer.fit(train_x)

    assert standardizer.dropped == (1, 4)
    assert standardizer.num_features == STATE_DIM - 2
    design = standardizer.design(train_x)
    assert design.shape == (train_x.shape[0], STATE_DIM - 2 + 1)
    assert np.isfinite(design).all()

    # Why the drop is not cosmetic: with those columns kept, the design matrix is rank deficient.
    raw = np.hstack([train_x, np.ones((train_x.shape[0], 1))])
    assert np.linalg.matrix_rank(raw) < raw.shape[1]
    assert np.linalg.matrix_rank(design) == design.shape[1]

    fit = rb.fit_ridge(
        standardizer, train_x, train_y, holdout_x, holdout_y, min(rb.DEFAULT_LAMBDAS)
    )
    assert np.isfinite(fit.holdout_mse)
    assert fit.num_parameters == (STATE_DIM - 2 + 1) * TARGET_DIM


def test_a_non_positive_lambda_is_refused(tmp_path: Path) -> None:
    """Zero is ordinary least squares, and any two collinear state dims make it singular. Refusing
    is better than returning a pinv solution nobody asked for under a name that says ridge."""
    table = rb.collect_chunks(_linear_dataset(tmp_path / "linear"), CHUNK_STEPS)
    train_x, train_y, holdout_x, holdout_y = _split(table, HOLDOUT_EPISODES)

    with pytest.raises(SystemExit, match="lambda must be > 0"):
        rb.fit_ridge(rb.Standardizer.fit(train_x), train_x, train_y, holdout_x, holdout_y, 0.0)


# -- the pairing rule has to be build_eval_pairs' rule --------------------------------------------


def test_the_pairs_match_build_eval_pairs_row_for_row(tmp_path: Path) -> None:
    """The licence for printing the ridge's MSE next to a model's.

    ``collect_chunks`` re-implements the pairing rather than calling ``build_eval_pairs``, because
    that function decodes the episode's video to build an Observation and this baseline is blind
    by design. A copied rule can drift, and a drifted one would score the ridge on chunks the
    model was never evaluated on while both numbers still looked comparable. So the two are run
    over the same episode and compared row for row.
    """
    from wam.evaluation.offline import build_eval_pairs

    root = tmp_path / "framed"
    rng = np.random.default_rng(3)
    states = rng.normal(0.0, 1.0, (12, STATE_DIM))
    targets = rng.normal(0.0, 0.05, (12, CHUNK_STEPS, NUM_JOINTS))
    _write_episode(root, "ep-framed", states, targets, with_frames=True)

    table = rb.collect_chunks(root, CHUNK_STEPS)
    reference = build_eval_pairs(root / "ep-framed", "ego", CHUNK_STEPS)

    assert table.num_rows == len(reference)
    for row, (observation, target, episode_id) in enumerate(reference):
        assert table.episode_ids[row] == episode_id
        assert table.states[row] == pytest.approx(rb.state_vector(observation.state))
        assert table.targets[row] == pytest.approx(
            np.asarray(target.targets, dtype=np.float64).reshape(-1)
        )


def test_short_chunks_are_skipped_and_long_ones_truncated(tmp_path: Path) -> None:
    """``EpisodeDataset``'s contract, so the rows line up with what training and evaluation saw.
    A short chunk padded instead of skipped would inject zeros the demonstrations never held."""
    root = tmp_path / "ragged"
    rng = np.random.default_rng(5)
    states = rng.normal(0.0, 1.0, (4, STATE_DIM))
    targets = rng.normal(0.0, 0.05, (4, CHUNK_STEPS, NUM_JOINTS))
    _write_episode(root, "ep-ragged", states, targets, chunk_lengths=(CHUNK_STEPS, 1, 7, 3))

    table = rb.collect_chunks(root, CHUNK_STEPS)

    assert table.num_rows == 2  # the 1-step and 3-step chunks are gone
    assert table.target_dim == TARGET_DIM
    assert table.targets[0] == pytest.approx(targets[0].reshape(-1), abs=1e-6)
    assert table.targets[1] == pytest.approx(targets[2].reshape(-1), abs=1e-6)


def test_the_state_is_the_last_one_at_or_before_the_chunk(tmp_path: Path) -> None:
    """Never the next state. Taking the following one would hand the baseline an observation from
    after the decision it is being asked to make — a leak in time rather than in episodes, and
    one that would flatter every number here."""
    from wam.data.episode import EpisodeReader

    root = _linear_dataset(tmp_path / "linear")
    table = rb.collect_chunks(root, CHUNK_STEPS)

    episode_id = TRAIN_EPISODES[0]
    rows = np.flatnonzero(table.episode_ids == episode_id)
    recorded = EpisodeReader(root / episode_id).read_states()
    assert len(rows) == CHUNKS_PER_EPISODE
    for i, row in enumerate(rows):
        # Chunk i was commanded at exactly state i's timestamp, so "at or before" picks state i.
        assert table.states[row] == pytest.approx(rb.state_vector(recorded[i]))


# -- 6. the blind nonlinear ceiling ---------------------------------------------------------------


def _nonlinear_dataset(root: Path, seed: int = 21) -> Path:
    """A target that is a nonlinear function of the state with NO linear component at all.

    ``(state**2 - 1) @ C`` on standard-normal states: every output is uncorrelated with every
    input by construction (``E[x*(x^2-1)] = 0`` under N(0,1)), so the best linear map is the
    intercept and the linear bar sits at the zero-delta baseline no matter how well it is fitted.
    An RBF-kernel regressor recovers it easily. That is the whole distinction the ceiling exists
    to measure, arranged so the two predictors cannot be confused for one another.
    """
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(0.0, 1.0, (STATE_DIM, TARGET_DIM))
    for episode_id in (*TRAIN_EPISODES, *HOLDOUT_EPISODES):
        states = rng.normal(0.0, 1.0, (CHUNKS_PER_EPISODE, STATE_DIM))
        targets = (states**2 - 1.0) @ coefficients
        _write_episode(
            root,
            episode_id,
            states,
            targets.reshape(CHUNKS_PER_EPISODE, CHUNK_STEPS, NUM_JOINTS),
        )
    return root


def _holdout_barely_moves_dataset(root: Path, seed: int = 17) -> Path:
    """Train carries a strong signal; the HOLDOUT is very nearly still.

    Not realism — leverage. The holdout has a property the train side does not, and exactly one
    hyperparameter reaches it: shrink hard enough and the prediction collapses towards the train
    mean, which on this holdout is almost exactly right. A search that is allowed to look at the
    holdout finds that immediately; an honest one, which only ever sees train-like validation
    episodes, picks the opposite end of the lambda grid because on train-like data shrinking that
    hard is terrible. So the two procedures are separated by construction rather than by luck.
    """
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 1.0, (STATE_DIM, TARGET_DIM))
    for episode_id in (*TRAIN_EPISODES, *HOLDOUT_EPISODES):
        states = rng.normal(0.0, 1.0, (CHUNKS_PER_EPISODE, STATE_DIM))
        if episode_id in HOLDOUT_EPISODES:
            targets = rng.normal(0.0, 1e-3, (CHUNKS_PER_EPISODE, TARGET_DIM))
        else:
            targets = states @ weights
        _write_episode(
            root,
            episode_id,
            states,
            targets.reshape(CHUNKS_PER_EPISODE, CHUNK_STEPS, NUM_JOINTS),
        )
    return root


def _fit_ceiling(table: Any, holdout_ids: tuple[str, ...] = HOLDOUT_EPISODES, **kwargs: Any) -> Any:
    """`fit_ceiling` on the toy grid, wired the way `main` wires it."""
    train_mask, holdout_mask = rb.split_by_episode(table, set(holdout_ids))
    train_x, train_y = table.select(train_mask)
    holdout_x, holdout_y = table.select(holdout_mask)
    options: dict[str, Any] = {
        "widths": CEILING_WIDTHS,
        "gammas": CEILING_GAMMAS,
        "lambdas": CEILING_LAMBDAS,
        "num_val_episodes": CEILING_VAL_EPISODES,
    }
    options.update(kwargs)
    return rb.fit_ceiling(
        train_x,
        train_y,
        table.episode_ids[train_mask],
        holdout_x,
        holdout_y,
        set(holdout_ids),
        **options,
    )


def _grid_scored_on(
    table: Any, holdout_ids: tuple[str, ...] = HOLDOUT_EPISODES
) -> list[tuple[Any, float]]:
    """Every grid point refitted on all train rows and scored ON THE HOLDOUT — i.e. the leak.

    This is what an unpoliced ceiling would report: the minimum of this list. It uses the same
    refit seed the honest path uses, so the honest config is a member of it and the comparison is
    "same fits, different chooser" rather than two different procedures.
    """
    train_mask, holdout_mask = rb.split_by_episode(table, set(holdout_ids))
    train_x, train_y = table.select(train_mask)
    holdout_x, holdout_y = table.select(holdout_mask)
    standardizer = rb.Standardizer.fit(train_x)
    scored: list[tuple[Any, float]] = []
    for width in CEILING_WIDTHS:
        for gamma in CEILING_GAMMAS:
            predictions = rb.ceiling_scores(
                standardizer,
                train_x,
                train_y,
                holdout_x,
                width=width,
                gamma=gamma,
                lambdas=CEILING_LAMBDAS,
                rng=np.random.default_rng(rb.CEILING_REFIT_SEED),
            )
            for lam, predicted in predictions.items():
                scored.append(
                    (
                        rb.CeilingConfig(width=width, gamma=gamma, lam=lam),
                        float(((predicted - holdout_y) ** 2).mean()),
                    )
                )
    return scored


def test_the_inner_split_is_drawn_from_train_only_and_is_episode_disjoint(tmp_path: Path) -> None:
    """The property the whole ceiling rests on, asserted over the episode tags themselves."""
    table = rb.collect_chunks(_linear_dataset(tmp_path / "linear"), CHUNK_STEPS)
    train_mask, _ = rb.split_by_episode(table, set(HOLDOUT_EPISODES))

    inner_fit, validation = rb.inner_validation_episodes(
        table.episode_ids[train_mask],
        set(HOLDOUT_EPISODES),
        num_val_episodes=CEILING_VAL_EPISODES,
    )

    assert len(validation) == CEILING_VAL_EPISODES
    assert not (validation & inner_fit)
    assert inner_fit | validation == set(TRAIN_EPISODES)
    assert not (validation & set(HOLDOUT_EPISODES))
    assert not (inner_fit & set(HOLDOUT_EPISODES))


def test_the_inner_split_refuses_to_run_if_a_holdout_episode_reaches_it() -> None:
    """The runtime check the ceiling's meaning depends on, exercised directly.

    A nonlinear regressor with ~10^6 parameters and a free bandwidth can be tuned to nearly any
    number on a 1 040-chunk holdout, so a ceiling whose hyperparameters saw the holdout is not a
    ceiling — it is a bar built to the height the builder wanted. If this test ever fails, the
    number in the module docstring stops being evidence about proprioception.
    """
    tagged = np.asarray([*TRAIN_EPISODES, "ep-ho-01", *TRAIN_EPISODES])

    with pytest.raises(SystemExit, match="HOLDOUT"):
        rb.inner_validation_episodes(
            tagged, {"ep-ho-01"}, num_val_episodes=CEILING_VAL_EPISODES
        )


def test_fit_ceiling_refuses_train_rows_that_carry_a_holdout_episode(tmp_path: Path) -> None:
    """The guard has to be WIRED, not merely present. Here the caller mislabels the split — the
    rows are right but the tags claim a holdout episode is on the train side — and the search has
    to stop rather than quietly choose its hyperparameters on it."""
    table = rb.collect_chunks(_linear_dataset(tmp_path / "linear"), CHUNK_STEPS)
    train_mask, holdout_mask = rb.split_by_episode(table, set(HOLDOUT_EPISODES))
    train_x, train_y = table.select(train_mask)
    holdout_x, holdout_y = table.select(holdout_mask)
    mislabelled = table.episode_ids[train_mask].copy()
    mislabelled[0] = HOLDOUT_EPISODES[0]

    with pytest.raises(SystemExit, match="HOLDOUT"):
        rb.fit_ceiling(
            train_x,
            train_y,
            mislabelled,
            holdout_x,
            holdout_y,
            set(HOLDOUT_EPISODES),
            widths=CEILING_WIDTHS,
            gammas=CEILING_GAMMAS,
            lambdas=CEILING_LAMBDAS,
            num_val_episodes=CEILING_VAL_EPISODES,
        )


def test_an_inner_validation_split_that_leaves_nothing_to_fit_is_refused(tmp_path: Path) -> None:
    """Asking for every train episode as validation is a different failure from a bad ceiling and
    must not be reported as one."""
    table = rb.collect_chunks(_linear_dataset(tmp_path / "linear"), CHUNK_STEPS)
    train_mask, _ = rb.split_by_episode(table, set(HOLDOUT_EPISODES))

    with pytest.raises(SystemExit, match="nothing to fit on"):
        rb.inner_validation_episodes(
            table.episode_ids[train_mask],
            set(HOLDOUT_EPISODES),
            num_val_episodes=len(TRAIN_EPISODES),
        )
    with pytest.raises(SystemExit, match=">= 1"):
        rb.inner_validation_episodes(
            table.episode_ids[train_mask], set(HOLDOUT_EPISODES), num_val_episodes=0
        )


def test_choosing_the_ceiling_hyperparameters_on_the_holdout_is_detectably_better(
    tmp_path: Path,
) -> None:
    """The leak control. Without it every other assertion about the ceiling is satisfied by a
    procedure that tunes itself on the data it reports.

    Same feature draws, same refits, same grid — the only difference is who picks. The honest
    picker sees only train-like validation episodes; the leaking picker sees the holdout, notices
    it barely moves, and takes the most-shrunk config on the grid. The gap is orders of magnitude,
    which is the actual size of the thing the guard is preventing: a ceiling nobody can clear,
    reported as a fact about proprioception.
    """
    table = rb.collect_chunks(_holdout_barely_moves_dataset(tmp_path / "leaky"), CHUNK_STEPS)

    honest = _fit_ceiling(table)
    leaked_config, leaked = min(_grid_scored_on(table), key=lambda row: row[1])

    assert leaked < honest.holdout_mse / 100.0, (
        f"leaked {leaked:.4e} vs honest {honest.holdout_mse:.4e} — choosing on the holdout bought "
        "nothing here, so this dataset no longer separates the two procedures and the guard is "
        "being tested against a case that cannot fail"
    )
    # And the honest search really did land somewhere else, rather than agreeing by accident.
    assert honest.config != leaked_config
    assert honest.config.lam < leaked_config.lam


def test_the_ceiling_beats_the_linear_ridge_when_the_target_is_nonlinear(tmp_path: Path) -> None:
    """Calibration, and the reason the row exists at all.

    On a target that is a nonlinear function of the state the linear bar is structurally too low.
    If the ceiling cannot clear it here, then it is not measuring nonlinearity — it is an
    expensive way to recompute a ridge, and reporting it as a stronger bar would be false.
    """
    table = rb.collect_chunks(_nonlinear_dataset(tmp_path / "nonlinear"), CHUNK_STEPS)
    _, _, _, holdout_y = _split(table, HOLDOUT_EPISODES)

    ceiling = _fit_ceiling(table)
    linear = _best_mse(table, HOLDOUT_EPISODES)
    zero = rb.zero_delta_mse(holdout_y)

    assert linear > zero * 0.9, "no linear component should pin the ridge to the zero-delta floor"
    assert ceiling.holdout_mse < linear / 5.0
    assert ceiling.holdout_mse < zero / 5.0


def test_the_ceiling_does_not_beat_zero_delta_on_a_target_independent_of_the_state(
    tmp_path: Path,
) -> None:
    """The ridge's no-leak mutation, repeated for the ceiling — which has three more orders of
    magnitude of capacity and therefore three more orders of magnitude of ways to cheat. With no
    relation to find, the honest procedure must shrink to the train mean and land AT the zero-delta
    baseline; anything below it is holdout information inside the fit."""
    table = rb.collect_chunks(_independent_dataset(tmp_path / "independent"), CHUNK_STEPS)
    _, _, _, holdout_y = _split(table, HOLDOUT_EPISODES)

    ceiling = _fit_ceiling(table)
    zero = rb.zero_delta_mse(holdout_y)

    assert ceiling.holdout_mse > zero * 0.9, (
        f"ceiling {ceiling.holdout_mse:.4e} beat zero-delta {zero:.4e} on noise — leak"
    )
    assert ceiling.holdout_mse == pytest.approx(zero, rel=0.3)


def test_the_ceilings_parameter_count_and_split_sizes_are_the_real_ones(tmp_path: Path) -> None:
    """983 280 on the T-16 geometry is (4096 random features + 1 bias) x 240 outputs, and this is
    that arithmetic at toy scale rather than a constant copied from the docstring."""
    table = rb.collect_chunks(_nonlinear_dataset(tmp_path / "nonlinear"), CHUNK_STEPS)

    ceiling = _fit_ceiling(table)

    assert ceiling.config.width in CEILING_WIDTHS
    assert ceiling.num_parameters == (ceiling.config.width + 1) * TARGET_DIM
    assert ceiling.num_configs == len(CEILING_WIDTHS) * len(CEILING_GAMMAS) * len(CEILING_LAMBDAS)
    assert ceiling.num_val_episodes == CEILING_VAL_EPISODES
    assert ceiling.num_inner_fit_episodes == len(TRAIN_EPISODES) - CEILING_VAL_EPISODES
    assert ceiling.num_val_rows == CEILING_VAL_EPISODES * CHUNKS_PER_EPISODE
    assert ceiling.num_inner_fit_rows + ceiling.num_val_rows == (
        len(TRAIN_EPISODES) * CHUNKS_PER_EPISODE
    )
    # The selection score and the reported score are computed on disjoint episodes by two fits.
    assert ceiling.val_mse != ceiling.holdout_mse


def test_a_degenerate_ceiling_hyperparameter_is_refused(tmp_path: Path) -> None:
    """A zero bandwidth is a constant feature map and a non-positive lambda is unregularized
    kernel regression on a rank-deficient system. Both are input errors, not numerical ones."""
    table = rb.collect_chunks(_linear_dataset(tmp_path / "linear"), CHUNK_STEPS)
    train_x, train_y, holdout_x, _ = _split(table, HOLDOUT_EPISODES)
    standardizer = rb.Standardizer.fit(train_x)

    with pytest.raises(SystemExit, match="gamma must be > 0"):
        rb.ceiling_scores(
            standardizer, train_x, train_y, holdout_x,
            width=8, gamma=0.0, lambdas=(1.0,), rng=np.random.default_rng(0),
        )  # fmt: skip
    with pytest.raises(SystemExit, match="lambda must be > 0"):
        rb.ceiling_scores(
            standardizer, train_x, train_y, holdout_x,
            width=8, gamma=0.5, lambdas=(0.0,), rng=np.random.default_rng(0),
        )  # fmt: skip


# -- 7. the archived numbers are the control, so they are checked ---------------------------------


def test_the_archive_reproduces_when_nothing_has_moved() -> None:
    measured = {key: float(value) for key, value in rb.ARCHIVED_T16.items()}

    message = rb.check_archived(measured, dict(rb.ARCHIVED_T16_SHAPE))

    assert message is not None
    assert f"{len(rb.ARCHIVED_T16)}/{len(rb.ARCHIVED_T16)}" in message


def test_a_moved_archived_number_raises_rather_than_printing_a_different_table() -> None:
    """The teeth. Three write-ups are stated as ratios against these rows, so a refactor that moves
    one has invalidated work that is already cited — and the printed table would look entirely
    normal. The drift asserted here is in the SEVENTH significant digit, because that is the
    resolution the numbers were published at and there is no amount of drift that is fine."""
    measured = {key: float(value) for key, value in rb.ARCHIVED_T16.items()}
    measured["ridge_all_lam10"] *= 1.0 + 1e-6

    with pytest.raises(SystemExit, match="MOVED"):
        rb.check_archived(measured, dict(rb.ARCHIVED_T16_SHAPE))


def test_the_archive_stays_silent_on_a_dataset_that_is_not_the_archived_one() -> None:
    """Checking T-16's numbers against some other dataset would fail every run for no reason, so
    the control arms itself on an exact shape match and says nothing otherwise."""
    measured = {key: 1.0 for key in rb.ARCHIVED_T16}
    shape = dict(rb.ARCHIVED_T16_SHAPE)
    shape["num_holdout_chunks"] += 1

    assert rb.check_archived(measured, shape) is None


def test_a_number_this_run_did_not_compute_is_skipped_not_failed() -> None:
    """``--no-ceiling`` and a custom ``--lam`` leave cells uncomputed. Missing is not moved."""
    measured: dict[str, float | None] = {
        key: float(value) for key, value in rb.ARCHIVED_T16.items()
    }
    measured["ceiling"] = None

    message = rb.check_archived(measured, dict(rb.ARCHIVED_T16_SHAPE))

    assert message is not None and "1 not computed" in message


# -- the CLI ------------------------------------------------------------------------------------


def _write_predictions(path: Path, table: Any, holdout_ids: tuple[str, ...], offset: float) -> None:
    """A ``predictions.jsonl`` for the holdout rows whose model MSE is exactly ``offset**2``.

    Written through the library's own serializer, so the file the script parses is the same shape
    every real run produces, and the number it recomputes can be checked against arithmetic
    instead of against a second implementation of the metric.
    """
    from wam.evaluation.offline import ChunkPrediction, save_predictions_jsonl

    predictions = []
    for row in range(table.num_rows):
        episode_id = str(table.episode_ids[row])
        if episode_id not in holdout_ids:
            continue
        target = table.targets[row].reshape(CHUNK_STEPS, NUM_JOINTS).astype(np.float32)
        predictions.append(
            ChunkPrediction(
                predicted=ActionChunk(
                    mode=ActionMode.JOINT_DELTA,
                    targets=(target + np.float32(offset)),
                    gripper_target=np.zeros(CHUNK_STEPS, dtype=np.float32),
                    dt_s=DT,
                ),
                target=ActionChunk(
                    mode=ActionMode.JOINT_DELTA,
                    targets=target,
                    gripper_target=np.zeros(CHUNK_STEPS, dtype=np.float32),
                    dt_s=DT,
                ),
                episode_id=episode_id,
                t_ns=row,
            )
        )
    save_predictions_jsonl(predictions, path)


def test_the_cli_reports_every_group_the_controls_and_the_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end on a predictions.jsonl holdout: the split, the ablations, the zero-delta control
    and the model's own recomputed MSE all have to come out of one invocation, because a reader
    comparing a model against this baseline needs them on one screen or they will be compared
    across two runs that need not agree."""
    root = _linear_dataset(tmp_path / "linear")
    table = rb.collect_chunks(root, CHUNK_STEPS)
    predictions = tmp_path / "predictions.jsonl"
    _write_predictions(predictions, table, HOLDOUT_EPISODES, offset=0.01)
    out = tmp_path / "ridge.json"

    exit_code = rb.main(
        [
            "--dataset", str(root),
            "--holdout", str(predictions),
            "--chunk-steps", str(CHUNK_STEPS),
            "--no-ceiling",
            "--json", str(out),
        ]
    )  # fmt: skip

    assert exit_code == 0

    printed = capsys.readouterr().out
    assert "zero-delta (hold still)" in printed
    assert "model (from predictions)" in printed
    # --no-ceiling has to say so: a reader must not take the linear bar for the stronger claim.
    assert "--no-ceiling: the NONLINEAR bar was not computed" in printed

    record = json.loads(out.read_text())
    assert record["num_holdout_chunks"] == len(HOLDOUT_EPISODES) * CHUNKS_PER_EPISODE
    assert record["num_train_chunks"] == len(TRAIN_EPISODES) * CHUNKS_PER_EPISODE
    assert record["num_holdout_episodes"] == len(HOLDOUT_EPISODES)
    assert set(record["groups"]) == set(rb.FEATURE_GROUPS)
    assert record["groups"]["all"]["num_parameters"] == NUM_PARAMETERS
    assert record["model_chunks"] == len(HOLDOUT_EPISODES) * CHUNKS_PER_EPISODE
    # predicted = target + 0.01 everywhere, so the recomputed model MSE is exactly 1e-4.
    assert record["model_mse"] == pytest.approx(1e-4, rel=1e-4)
    assert record["zero_delta_mse"] > 0.0
    assert record["ceiling"] is None
    # The calibration case again, this time through the CLI's own reported numbers.
    assert record["groups"]["all"]["best_mse"] < record["zero_delta_mse"] * 1e-6


def test_the_cli_reports_the_ceiling_and_the_split_it_was_chosen_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ceiling is an ADDITIVE row: everything the linear run printed still prints, and the
    ceiling arrives with the evidence that it was not tuned on the holdout — the inner split's
    episode counts, the number of configs searched and the validation score of the winner, all on
    the same screen as the holdout number they licence."""
    root = _nonlinear_dataset(tmp_path / "nonlinear")
    holdout = tmp_path / "holdout.txt"
    holdout.write_text("\n".join(HOLDOUT_EPISODES) + "\n")
    out = tmp_path / "ridge.json"

    exit_code = rb.main(
        [
            "--dataset", str(root),
            "--holdout", str(holdout),
            "--chunk-steps", str(CHUNK_STEPS),
            "--ceiling-val-episodes", str(CEILING_VAL_EPISODES),
            *[a for w in CEILING_WIDTHS for a in ("--ceiling-width", str(w))],
            *[a for g in CEILING_GAMMAS for a in ("--ceiling-gamma", str(g))],
            *[a for l in CEILING_LAMBDAS for a in ("--ceiling-lam", str(l))],
            "--json", str(out),
        ]
    )  # fmt: skip

    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "blind NONLINEAR ceiling" in printed
    assert "TRAIN only" in printed
    assert "zero-delta (hold still)" in printed  # still additive

    record = json.loads(out.read_text())
    ceiling = record["ceiling"]
    assert ceiling["num_val_episodes"] == CEILING_VAL_EPISODES
    assert ceiling["num_inner_fit_episodes"] == len(TRAIN_EPISODES) - CEILING_VAL_EPISODES
    assert len(ceiling["val_grid"]) == ceiling["num_configs"]
    assert ceiling["num_parameters"] == (ceiling["width"] + 1) * TARGET_DIM
    # The whole point of the row: on a nonlinear target it clears the linear bar.
    assert ceiling["holdout_mse"] < record["groups"]["all"]["best_mse"] / 2.0
    assert record["archive_checked"] is False


def test_a_plain_episode_id_list_is_accepted_and_reports_no_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``load_episode_ids`` takes both shapes, so the script must too — and with an id list there
    is no run to compare against, which has to be said rather than left as a blank line."""
    root = _linear_dataset(tmp_path / "linear")
    holdout = tmp_path / "holdout.txt"
    holdout.write_text("# the reviewed split\n\n" + "\n".join(HOLDOUT_EPISODES) + "\n")
    out = tmp_path / "ridge.json"

    exit_code = rb.main(
        [
            "--dataset", str(root),
            "--holdout", str(holdout),
            "--chunk-steps", str(CHUNK_STEPS),
            "--no-ceiling",
            "--json", str(out),
        ]
    )  # fmt: skip

    assert exit_code == 0
    assert "no run to score against" in capsys.readouterr().out
    record = json.loads(out.read_text())
    assert record["model_mse"] is None and record["model_chunks"] is None


def test_the_dq_ablation_is_reported_separately(tmp_path: Path) -> None:
    """The ablation that carries the finding on the real data (``dq`` alone beats the fine-tune,
    ``q`` alone does not), so each channel has to be fitted on its own columns and not read off a
    slice of the all-state weights."""
    root = tmp_path / "dq-only"
    rng = np.random.default_rng(13)
    weights = rng.normal(0.0, 0.05, (NUM_JOINTS, TARGET_DIM))
    for episode_id in (*TRAIN_EPISODES, *HOLDOUT_EPISODES):
        states = rng.normal(0.0, 1.0, (CHUNKS_PER_EPISODE, STATE_DIM))
        targets = (states[:, NUM_JOINTS : 2 * NUM_JOINTS] @ weights).reshape(
            CHUNKS_PER_EPISODE, CHUNK_STEPS, NUM_JOINTS
        )
        _write_episode(root, episode_id, states, targets)

    table = rb.collect_chunks(root, CHUNK_STEPS)
    groups = rb.feature_groups(table.num_joints, table.gripper_dims, table.state_dim)
    _, _, _, holdout_y = _split(table, HOLDOUT_EPISODES)
    zero = rb.zero_delta_mse(holdout_y)

    assert int(groups["q"].sum()) == NUM_JOINTS
    assert int(groups["dq"].sum()) == NUM_JOINTS
    assert int(groups["gripper"].sum()) == GRIPPER_DIMS
    assert _best_mse(table, HOLDOUT_EPISODES, groups["dq"]) < zero * 1e-6
    assert _best_mse(table, HOLDOUT_EPISODES, groups["q"]) > zero * 0.5
    assert _best_mse(table, HOLDOUT_EPISODES, groups["gripper"]) > zero * 0.5


def test_an_empty_dataset_is_refused(tmp_path: Path) -> None:
    """An empty root is a mistyped path, and a baseline that returns 0 chunks quietly is a
    baseline that will be quoted as though it had been measured."""
    (tmp_path / "empty").mkdir()

    with pytest.raises(SystemExit, match="no episodes found"):
        rb.collect_chunks(tmp_path / "empty", CHUNK_STEPS)
