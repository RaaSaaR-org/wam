"""Tests for `scripts/bench_incremental_value.py` — PR-01's four tests and its mechanical verdict.

PR-01 exists to decide whether ~125 GPU-h are spent. Its whole value is that the answer comes out
of a rule fixed in advance rather than out of a reading of a table, so the ways this script can be
wrong are not "it prints the wrong float" — they are:

  it leaks in T1       The stacking weights are two numbers, and two numbers fitted on the rows
                       they are scored on will reproduce anything. `beta` is THE answer to the
                       pre-registered question, so the cross-fitting has to be shown to cost
                       something. It is, by a DELIBERATELY leaked control that scores ~0 where the
                       leak-free path scores worse than predicting nothing at all.
  it leaks in T3       The branch-point subset is chosen using the target, which is admissible in
                       one direction only — and only because the ranking predictor is neither of
                       the two being compared. A subset ranked on the model's own error is the
                       same code with one argument changed, so the second leaked control does
                       exactly that and has to come out detectably different.
  it misaligns         Nothing crashes when the model's chunk `i` is compared against the ridge's
                       chunk `i+1`. Every per-chunk number in T1/T3/T4 would then be about two
                       different moments, and the verdict would still print. The alignment is
                       therefore proven against the data and the proof is tested with a
                       deliberately permuted predictions file.
  it extrapolates      `pred[k, j] = dq[j] * dt_s` at EVERY step, not `k * dq * dt_s`. The
  the wrong shape      cumulative form is wrong by ~8x mid-chunk and would make the no-parameter
                       reference look far worse than momentum is — which is the direction that
                       fakes Reading A. Pinned against a dataset where it must score exactly 0.
  it reproduces        A run that does not reproduce zero-delta 1.632760e-05 and model
  nothing              1.112983e-05 is void. The gate is tested for firing, not only for passing.

Plus the verdict rule itself, driven through every branch: A needs all four clauses, B never
mentions T4, and an unevaluable T4 clause makes A unreachable rather than free.

AND THE CALL SITE, NOT ONLY THE HELPER
--------------------------------------
A leak control that hands hand-built arrays to `branch_point_mask` proves that FUNCTION ranks what
it is given. It says nothing about what `main` gives it — and that is where every leak PR-01 is
about would actually live: T3 ranked on the model's own error, the folds row-wise, the LEAKED
in-sample stack passed to clause 1, the model's gripper channel scored against its own target,
constant velocity read off `q`. Each is a one-token edit at a call site, each still prints a
plausible six-digit table under a passing archive gate, and each moves a pre-registered clause.

So the CLI tests here assert VALUES and SUBSETS recomputed independently from the dataset — not
that a section header was printed. The subsets T3 and T4 select are written into the `--json`
record precisely so they can be checked from outside instead of trusted, and the two comparisons
the verdict is made of are pinned to numbers this file computes for itself.

Nothing here touches `datasets/gr00t-apple-full` or `runs/`. Episodes are synthesized in `tmp_path`
with the repo's own `EpisodeWriter`, carry no frames (this measurement never opens a camera), and
the predictions.jsonl is written through `wam.evaluation.offline`'s own serializer so the file the
script parses is the shape every real run produces.
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
from wam.evaluation.gripper import crossings
from wam.evaluation.offline import (
    ChunkPrediction,
    load_predictions_jsonl,
    save_predictions_jsonl,
)
from wam.interfaces.schema import (
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    IMUState,
    RobotState,
    ValidityMask,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

NUM_JOINTS = 3
GRIPPER_DIMS = 2
CHUNK_STEPS = 4
STATE_DIM = 2 * NUM_JOINTS + GRIPPER_DIMS  # 8
FPS = 20.0
DT = 1.0 / FPS
CHUNKS_PER_EPISODE = 20

#: 10 holdout episodes so 5 folds over EPISODES is possible with 2 episodes in each.
TRAIN_EPISODES = tuple(f"ep-tr-{i:02d}" for i in range(12))
HOLDOUT_EPISODES = tuple(f"ep-ho-{i:02d}" for i in range(10))

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
    """`test_bench_ridge_baseline`'s loader, plus the `sys.modules` registration `@dataclass`
    needs to resolve its own annotations on a module imported by path."""
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


biv = _load("bench_incremental_value")
#: Reached THROUGH the script under test, never loaded a second time: two module objects would
#: mean two `ChunkTable` classes and the dataclass identity checks below would compare strangers.
rb = biv.ridge_baseline


# -- synthesis -----------------------------------------------------------------------------------


def _write_episode(
    root: Path,
    episode_id: str,
    states: np.ndarray,
    targets: np.ndarray,
    gripper: np.ndarray,
) -> Path:
    """One WAM episode: row ``i`` of ``states`` at ``t = i*DT``, with a chunk commanded there.

    ``targets`` is [n, CHUNK_STEPS, NUM_JOINTS] and ``gripper`` is [n, CHUNK_STEPS] in [0, 1].
    No frames: this measurement never decodes a camera, so encoding video for it would be paying
    for the pixels whose absence is the point.
    """
    episode_dir = root / episode_id
    with EpisodeWriter(episode_dir, episode_id, SPEC, FPS, "pick the apple") as writer:
        for i, (x, y, g) in enumerate(zip(states, targets, gripper)):
            ts = round(i * DT * 1e9)
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
            writer.add_action(
                ActionChunk(
                    mode=ActionMode.JOINT_DELTA,
                    targets=np.asarray(y, dtype=np.float32),
                    gripper_target=np.asarray(g, dtype=np.float32),
                    dt_s=DT,
                ),
                executed_prefix=1,
                timestamp_ns=ts,
            )
    return episode_dir


def _gripper_series(
    rng: np.random.Generator, n: int, *, transitions: bool, dither: bool = False
) -> np.ndarray:
    """[n, CHUNK_STEPS] gripper commands. With ``transitions``, some chunks cross 0.5.

    Without them the channel is deliberately NARROW (0.20-0.30, peak-to-peak ~0.10) rather than
    merely low: that is the shape a flattened gripper actually has on the converted GR00T data, and
    it exercises both the empty-subset path and the ``GRIPPER_MIN_DYNAMIC_RANGE`` clause at once.

    ``dither`` adds chunks that sit ON the threshold and wobble across it. Those are raw crossings
    and are NOT debounced transitions, so the two counts come apart — which is the only condition
    under which a test can see WHICH of them T4's clause is evaluated on. With every chunk either
    flat or a genuine ramp the two counts agree and the question is unaskable.
    """
    if not transitions:
        return rng.uniform(0.20, 0.30, (n, CHUNK_STEPS))
    base = rng.uniform(0.05, 0.45, (n, CHUNK_STEPS))
    kind = rng.random(n)
    ramp = np.linspace(0.2, 0.8, CHUNK_STEPS)[None, :]
    crossing = kind < 0.3
    base[crossing] = ramp + rng.normal(0.0, 0.01, (int(crossing.sum()), CHUNK_STEPS))
    if dither:
        wobble = (kind >= 0.3) & (kind < 0.6)
        pattern = np.array([0.499, 0.501] * ((CHUNK_STEPS + 1) // 2))[:CHUNK_STEPS]
        base[wobble] = pattern[None, :]
    return np.clip(base, 0.0, 1.0)


def _dataset(root: Path, seed: int = 0, *, transitions: bool = True, dither: bool = False) -> Path:
    """A dataset where the target is a linear function of the state plus noise.

    Linear-plus-noise on purpose: the ridge then genuinely predicts, so the stacking system is well
    conditioned and T3's subset is not a subset of pure noise. What the model does on top of it is
    supplied per test by the predictions file, which is where the interesting variation lives.
    """
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.05, (STATE_DIM, CHUNK_STEPS * NUM_JOINTS))
    for episode_id in (*TRAIN_EPISODES, *HOLDOUT_EPISODES):
        states = rng.normal(0.0, 1.0, (CHUNKS_PER_EPISODE, STATE_DIM))
        noise = rng.normal(0.0, 0.01, (CHUNKS_PER_EPISODE, CHUNK_STEPS * NUM_JOINTS))
        targets = (states @ weights + noise).reshape(CHUNKS_PER_EPISODE, CHUNK_STEPS, NUM_JOINTS)
        gripper = _gripper_series(rng, CHUNKS_PER_EPISODE, transitions=transitions, dither=dither)
        _write_episode(root, episode_id, states, targets, gripper)
    return root


def _momentum_dataset(root: Path, seed: int = 4) -> Path:
    """A dataset where every chunk IS ``dq * dt`` repeated — momentum, exactly.

    The reference predictor must then score 0 THROUGH THE CLI, which is the only form of the check
    that can see which state columns ``main`` sliced. ``q`` and ``dq`` are independent draws of the
    same distribution, so a run that read ``q`` produces an ordinary-looking number here and an
    ordinary-looking number on real data; only "exactly zero on a dataset built from dq" separates
    them.
    """
    rng = np.random.default_rng(seed)
    for episode_id in (*TRAIN_EPISODES, *HOLDOUT_EPISODES):
        states = rng.normal(0.0, 1.0, (CHUNKS_PER_EPISODE, STATE_DIM))
        dq = states[:, NUM_JOINTS : 2 * NUM_JOINTS]
        targets = np.repeat((dq * DT)[:, None, :], CHUNK_STEPS, axis=1)
        gripper = _gripper_series(rng, CHUNKS_PER_EPISODE, transitions=True)
        _write_episode(root, episode_id, states, targets, gripper)
    return root


def _write_predictions(
    path: Path,
    root: Path,
    holdout_ids: tuple[str, ...],
    *,
    noise: float = 0.02,
    seed: int = 99,
    gripper_offset: float | None = None,
) -> None:
    """A predictions.jsonl for the holdout, written through the library's own serializer.

    ``gripper_offset`` replaces the noisy gripper prediction with the demonstrated channel plus a
    known constant, so the gripper-channel MSE the CLI reports has one arithmetically predictable
    value. ``0.0`` is the degenerate case: a "model" that reports the target back, which is what a
    run reading ``target`` where it meant ``predicted`` looks like from the outside.
    """
    from wam.data.episode import EpisodeReader, list_episodes

    rng = np.random.default_rng(seed)
    records: list[ChunkPrediction] = []
    for episode_dir in list_episodes(root):
        reader = EpisodeReader(episode_dir)
        episode_id = reader.manifest.episode_id
        if episode_id not in holdout_ids:
            continue
        for chunk, _executed, ts in reader.read_actions():
            target = np.asarray(chunk.targets, dtype=np.float32)
            grip = np.asarray(chunk.gripper_target, dtype=np.float32)
            if gripper_offset is None:
                predicted_grip = np.clip(
                    grip + rng.normal(0.0, noise, grip.shape), 0.0, 1.0
                ).astype(np.float32)
            else:
                predicted_grip = (grip + gripper_offset).astype(np.float32)
            records.append(
                ChunkPrediction(
                    predicted=ActionChunk(
                        mode=ActionMode.JOINT_DELTA,
                        targets=(target + rng.normal(0.0, noise, target.shape)).astype(np.float32),
                        gripper_target=predicted_grip,
                        dt_s=float(chunk.dt_s),
                    ),
                    target=ActionChunk(
                        mode=ActionMode.JOINT_DELTA,
                        targets=target,
                        gripper_target=grip,
                        dt_s=float(chunk.dt_s),
                    ),
                    episode_id=episode_id,
                    t_ns=int(ts),
                )
            )
    save_predictions_jsonl(records, path)


def _prepared(tmp_path: Path, **kwargs: Any) -> tuple[Path, Path, Any, Any, np.ndarray]:
    """``(root, predictions, table, extras, holdout_row_index)`` — the script's own front half."""
    root = _dataset(tmp_path / "ds", **kwargs)
    predictions_path = tmp_path / "predictions.jsonl"
    _write_predictions(predictions_path, root, HOLDOUT_EPISODES)
    table = rb.collect_chunks(root, CHUNK_STEPS)
    extras = biv.collect_chunk_extras(root, CHUNK_STEPS)
    biv.check_extras_align(table, extras)
    _, holdout_mask = rb.split_by_episode(table, set(HOLDOUT_EPISODES))
    rows = biv.align_to_predictions(
        table, np.flatnonzero(holdout_mask), extras, load_predictions_jsonl(predictions_path)
    )
    return root, predictions_path, table, extras, rows


def _run_cli(root: Path, predictions: Path, out: Path) -> dict[str, Any]:
    """One full invocation with the gate off, returning the ``--json`` record it wrote."""
    exit_code = biv.main(
        [
            "--dataset", str(root),
            "--holdout", str(predictions),
            "--chunk-steps", str(CHUNK_STEPS),
            "--archive-gate", "off",
            "--json", str(out),
        ]
    )  # fmt: skip
    assert exit_code == 0
    return json.loads(out.read_text())


def _holdout_view(root: Path, predictions: Path) -> tuple[Any, np.ndarray, np.ndarray, Any]:
    """``(table, holdout rows, target [N, K, J], extras)`` — the CLI's inputs, rebuilt here.

    Recomputed rather than read out of the record, so a test can compare what the run REPORTED
    against what the dataset says instead of against the run's own opinion of it.
    """
    table = rb.collect_chunks(root, CHUNK_STEPS)
    extras = biv.collect_chunk_extras(root, CHUNK_STEPS)
    _, holdout_mask = rb.split_by_episode(table, set(HOLDOUT_EPISODES))
    rows = biv.align_to_predictions(
        table, np.flatnonzero(holdout_mask), extras, load_predictions_jsonl(predictions)
    )
    target = table.targets[rows].reshape(-1, CHUNK_STEPS, NUM_JOINTS)
    return table, rows, target, extras


# -- the pre-registered constants ARE the document's constants ------------------------------------


def test_the_thresholds_are_the_pre_registered_ones() -> None:
    """PR-01 is annotated, never edited; a threshold that must change is versioned as PR-01b. So
    the constants are pinned here too — a silent edit to one of them changes a verdict that a
    ~125 GPU-h decision hangs on, and would otherwise pass review as a diff to a print statement.
    """
    assert biv.STACK_IMPROVEMENT_FRACTION == 0.95
    assert biv.MIN_INCREMENTAL_BETA == 0.05
    assert biv.MIN_FOLDS_WITH_POSITIVE_BETA == 4
    assert biv.STACKING_FOLDS == 5
    assert biv.BRANCH_POINT_QUANTILE == 0.75
    assert biv.ARCHIVED_ZERO_DELTA_MSE == 1.632760e-05
    assert biv.ARCHIVED_MODEL_MSE == 1.112983e-05
    assert biv.ARCHIVED_RIDGE_ALL_MSE == 6.330899e-06


def test_the_two_subset_choices_are_constants_and_are_the_pre_registered_ones() -> None:
    """The two places PR-01 selects on the target, pinned by name rather than by value.

    T3's ranking predictor may not be either side of the comparison it feeds — that single fact is
    the whole of T3's one-directional admissibility — and T4's clause is the RAW crossing subset,
    which on the T-16 holdout has 17 chunks where the debounced one has 0. Both are silent choices:
    every alternative prints the same headers with different numbers underneath.
    """
    assert biv.T3_RANKING_PREDICTOR == "const-velocity"
    assert biv.T3_RANKING_PREDICTOR not in (biv.RIDGE_KEY, biv.MODEL_KEY)
    assert biv.T4_PRIMARY_DEBOUNCE is False


# -- 1. the constant-velocity reference ------------------------------------------------------------


def test_constant_velocity_is_the_same_delta_at_every_step() -> None:
    """The shape of the extrapolation, which is the part that is easy to get wrong.

    ``targets`` are PER-STEP deltas that ``G1Adapter.execute`` integrates onto the current ``q``,
    so an arm continuing at its present speed covers ``dq * dt_s`` in EVERY control period. The
    cumulative form ``k * dq * dt_s`` describes an arm accelerating away; it is wrong by ~8x in the
    middle of a 16-step chunk, and it is wrong in the direction that makes the no-parameter
    reference look bad — i.e. the direction that would fake Reading A.
    """
    dq = np.array([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
    dt = np.array([0.1, 0.25])

    pred = biv.constant_velocity(dq, dt, 4)

    assert pred.shape == (2, 4, 3)
    for k in range(4):
        assert pred[:, k, :] == pytest.approx(dq * dt[:, None])
    cumulative = np.stack([(k + 1) * dq * dt[:, None] for k in range(4)], axis=1)
    assert not np.allclose(pred, cumulative), "this is the cumulative rule, not constant velocity"


def test_constant_velocity_scores_exactly_zero_when_the_arm_does_continue(tmp_path: Path) -> None:
    """Calibration. On a dataset built so that every chunk IS ``dq * dt`` repeated, the reference
    has to score 0 — otherwise a large number elsewhere says nothing about momentum, it says the
    extrapolator is broken, and the two are indistinguishable from the printed value."""
    root = tmp_path / "momentum"
    rng = np.random.default_rng(4)
    for episode_id in (*TRAIN_EPISODES[:2], *HOLDOUT_EPISODES[:2]):
        states = rng.normal(0.0, 1.0, (CHUNKS_PER_EPISODE, STATE_DIM))
        dq = states[:, NUM_JOINTS : 2 * NUM_JOINTS]
        step = (dq * DT)[:, None, :]
        targets = np.repeat(step, CHUNK_STEPS, axis=1)
        _write_episode(
            root,
            episode_id,
            states,
            targets,
            _gripper_series(rng, CHUNKS_PER_EPISODE, transitions=False),
        )

    table = rb.collect_chunks(root, CHUNK_STEPS)
    extras = biv.collect_chunk_extras(root, CHUNK_STEPS)
    biv.check_extras_align(table, extras)
    target = table.targets.reshape(-1, CHUNK_STEPS, NUM_JOINTS)
    pred = biv.constant_velocity(
        table.states[:, NUM_JOINTS : 2 * NUM_JOINTS], extras.dt_s, CHUNK_STEPS
    )

    zero = rb.zero_delta_mse(table.targets)
    assert zero > 0.0
    assert biv.mse(pred, target) < zero * 1e-8


# -- 2. alignment --------------------------------------------------------------------------------


def test_the_alignment_maps_every_prediction_onto_its_own_chunk(tmp_path: Path) -> None:
    """One row per prediction, in the predictions file's order, each tagged with its own episode."""
    _, predictions_path, table, _, rows = _prepared(tmp_path)
    predictions = load_predictions_jsonl(predictions_path)

    assert rows.shape[0] == len(predictions)
    assert len(set(rows.tolist())) == rows.shape[0], "two predictions share a row"
    for pred, row in zip(predictions, rows.tolist()):
        assert str(table.episode_ids[row]) == pred.episode_id
        assert table.targets[row].reshape(CHUNK_STEPS, NUM_JOINTS) == pytest.approx(
            np.asarray(pred.target.targets, dtype=np.float64)
        )


def test_two_predictions_swapped_inside_one_episode_are_refused(tmp_path: Path) -> None:
    """The failure that does not crash and does not look wrong.

    Position pairing alone would happily hand chunk 0's model prediction to chunk 1's ridge
    prediction; the numbers would still be six digits and the verdict would still print. So the
    pairing is verified against the demonstrated target, and that check has to fire here.
    """
    _, predictions_path, table, extras, _ = _prepared(tmp_path)
    lines = predictions_path.read_text(encoding="utf-8").splitlines()
    lines[0], lines[1] = lines[1], lines[0]  # same episode, adjacent chunks
    swapped = tmp_path / "swapped.jsonl"
    swapped.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _, holdout_mask = rb.split_by_episode(table, set(HOLDOUT_EPISODES))
    with pytest.raises(SystemExit, match="does not match the chunk it was aligned to"):
        biv.align_to_predictions(
            table, np.flatnonzero(holdout_mask), extras, load_predictions_jsonl(swapped)
        )


def test_a_holdout_chunk_with_no_prediction_is_refused(tmp_path: Path) -> None:
    """A short predictions file would score the ridge on chunks the model never saw, and the two
    MSEs would be printed on adjacent lines anyway."""
    _, predictions_path, table, extras, _ = _prepared(tmp_path)
    lines = predictions_path.read_text(encoding="utf-8").splitlines()
    trimmed = tmp_path / "trimmed.jsonl"
    trimmed.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    _, holdout_mask = rb.split_by_episode(table, set(HOLDOUT_EPISODES))
    with pytest.raises(SystemExit, match="have no prediction"):
        biv.align_to_predictions(
            table, np.flatnonzero(holdout_mask), extras, load_predictions_jsonl(trimmed)
        )


def test_the_extras_pass_is_asserted_against_collect_chunks_not_trusted(tmp_path: Path) -> None:
    """``collect_chunk_extras`` duplicates ``collect_chunks``' skip rule, and a duplicated rule can
    drift. The guard is over the row tags, and it has to have teeth: a single dropped row must be
    refused rather than shifting every gripper column by one chunk."""
    _, _, table, extras, _ = _prepared(tmp_path)

    assert np.array_equal(extras.episode_ids, table.episode_ids)
    assert extras.gripper_targets.shape == (table.num_rows, CHUNK_STEPS)
    assert extras.dt_s == pytest.approx(np.full(table.num_rows, DT))

    drifted = biv.ChunkExtras(
        episode_ids=extras.episode_ids[1:],
        gripper_targets=extras.gripper_targets[1:],
        dt_s=extras.dt_s[1:],
    )
    with pytest.raises(SystemExit, match="rows and collect_chunks built"):
        biv.check_extras_align(table, drifted)


def test_extras_reordered_at_the_same_length_are_refused(tmp_path: Path) -> None:
    """The reordering the length check cannot see, and the one that would do the most damage.

    A dropped row shortens the pass and is caught by the count. A PERMUTATION does not: every row
    still exists, the count still matches, and every ``gripper_target``/``dt_s`` is attached to the
    wrong chunk — including on the TRAIN rows, which feed the gripper ridge T4's clause is decided
    against and which the predictions file never re-checks. So the guard is over the tags, and it
    has to fire on a same-length swap between two different episodes.
    """
    _, _, table, extras, _ = _prepared(tmp_path)
    episode_ids = extras.episode_ids.copy()
    gripper = extras.gripper_targets.copy()
    dt_s = extras.dt_s.copy()
    other = int(np.flatnonzero(episode_ids != episode_ids[0])[0])
    for column in (episode_ids, gripper, dt_s):
        column[[0, other]] = column[[other, 0]]

    reordered = biv.ChunkExtras(episode_ids=episode_ids, gripper_targets=gripper, dt_s=dt_s)

    assert reordered.episode_ids.shape == extras.episode_ids.shape
    with pytest.raises(SystemExit, match="disagree at row"):
        biv.check_extras_align(table, reordered)


def _rewrite_first_record(predictions_path: Path, out: Path, **target_fields: Any) -> Path:
    """Copy the predictions file with ``target`` fields of record 0 overwritten."""
    lines = predictions_path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["target"].update(target_fields)
    lines[0] = json.dumps(record)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def test_a_prediction_whose_gripper_channel_disagrees_is_refused(tmp_path: Path) -> None:
    """The alignment is proven on three fields, and each leg has to be able to fire on its own.

    The joint targets agreeing is not proof that the EXTRAS pass points at the same chunk: the
    gripper column and ``dt_s`` come from a different walk of the dataset, and T4's subset and the
    constant-velocity rate are read off exactly those. So a record whose targets still match but
    whose gripper channel does not must be refused.
    """
    _, predictions_path, table, extras, _ = _prepared(tmp_path)
    bad = _rewrite_first_record(
        predictions_path,
        tmp_path / "bad_gripper.jsonl",
        gripper_target=[0.123] * CHUNK_STEPS,
    )

    _, holdout_mask = rb.split_by_episode(table, set(HOLDOUT_EPISODES))
    with pytest.raises(SystemExit, match="scored gripper channel does not match"):
        biv.align_to_predictions(
            table, np.flatnonzero(holdout_mask), extras, load_predictions_jsonl(bad)
        )


def test_a_prediction_whose_dt_s_disagrees_is_refused(tmp_path: Path) -> None:
    """``dt_s`` is the rate the no-parameter reference extrapolates at. A record that disagrees with
    the recorded chunk is either a different chunk or a different control period, and both make the
    constant-velocity row a number about something else."""
    _, predictions_path, table, extras, _ = _prepared(tmp_path)
    bad = _rewrite_first_record(predictions_path, tmp_path / "bad_dt.jsonl", dt_s=DT * 2.0)

    _, holdout_mask = rb.split_by_episode(table, set(HOLDOUT_EPISODES))
    with pytest.raises(SystemExit, match="Constant velocity would be extrapolated"):
        biv.align_to_predictions(
            table, np.flatnonzero(holdout_mask), extras, load_predictions_jsonl(bad)
        )


# -- 3. the ridge path is bench_ridge_baseline's ridge ---------------------------------------------


def test_the_ridge_weights_reproduce_fit_ridges_own_mse(tmp_path: Path) -> None:
    """PR-01 needs the ridge's PER-ROW predictions, which ``fit_ridge`` does not return. So the
    solve is repeated here — and a repeated solve that has drifted would move the denominator of
    two decision clauses without moving anything a reader is looking at."""
    _, _, table, _, rows = _prepared(tmp_path)
    train_mask, _ = rb.split_by_episode(table, set(HOLDOUT_EPISODES))
    train_x, train_y = table.select(train_mask)
    holdout_x, holdout_y = table.states[rows], table.targets[rows]
    standardizer = rb.Standardizer.fit(train_x)

    weights = biv.ridge_weights(standardizer, train_x, train_y, biv.RIDGE_LAMBDA)
    ours = biv.mse(standardizer.design(holdout_x) @ weights, holdout_y)
    theirs = rb.fit_ridge(
        standardizer, train_x, train_y, holdout_x, holdout_y, biv.RIDGE_LAMBDA
    ).holdout_mse

    assert ours == pytest.approx(theirs, rel=1e-12)


# -- 4. T1: the folds are over episodes, and the cross-fitting is leak-free ------------------------


def test_folds_are_over_episodes_and_every_episode_lands_in_exactly_one() -> None:
    """Chunks inside one episode are overlapping views of one motion, so a row-level fold would fit
    the two stacking weights on near-copies of the rows it then scores — and ``beta`` is the whole
    pre-registered question reduced to one number."""
    episode_ids = np.asarray([f"ep-{i:02d}" for i in range(10) for _ in range(7)])

    folds = biv.episode_folds(episode_ids, biv.STACKING_FOLDS)

    assert set(folds) == set(episode_ids.tolist())
    assert sorted(set(folds.values())) == list(range(biv.STACKING_FOLDS))
    for k in range(biv.STACKING_FOLDS):
        scored = {ep for ep, f in folds.items() if f == k}
        fitted = {ep for ep, f in folds.items() if f != k}
        assert scored and fitted
        assert not (scored & fitted)


def test_fewer_episodes_than_folds_is_refused() -> None:
    """Falling back to a row split would be silent and would produce a plausible ``beta``."""
    with pytest.raises(SystemExit, match="cannot be split into"):
        biv.episode_folds(np.asarray(["ep-00", "ep-01"]), biv.STACKING_FOLDS)


def test_a_stack_fitted_on_its_own_fold_scores_detectably_better() -> None:
    """THE leak control for T1. Proves the cross-fitting is doing work, not merely passing.

    The construction: the "model" prediction is ``+target`` in some folds and ``-target`` in the
    others. Weights fitted ON their own fold find ``beta = +-1`` and reconstruct the target
    exactly; weights fitted on the OTHER four folds see the signs cancel, land near ``beta ~ -0.5``
    for a fold whose sign is ``+1``, and score WORSE than predicting nothing.

    That gap is exactly what a leak converts into a small number, and it is the number PR-01's
    primary test reports. If the two agreed, `cross_fitted_stack` would be satisfied by an
    implementation that fits on everything.
    """
    rng = np.random.default_rng(21)
    episodes = [f"ep-{i:02d}" for i in range(10)]
    per_episode = 30
    episode_ids = np.asarray([e for e in episodes for _ in range(per_episode)])
    n = episode_ids.shape[0]
    target = rng.normal(0.0, 1.0, (n, CHUNK_STEPS, NUM_JOINTS))
    ridge = rng.normal(0.0, 1.0, (n, CHUNK_STEPS, NUM_JOINTS))  # a weak, independent predictor

    folds = biv.episode_folds(episode_ids, biv.STACKING_FOLDS)
    sign_of_fold = {0: 1.0, 1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0}
    signs = np.asarray([sign_of_fold[folds[e]] for e in episode_ids.tolist()])
    model = signs[:, None, None] * target

    stack = biv.cross_fitted_stack(ridge, model, target, episode_ids, biv.STACKING_FOLDS)
    honest = stack.mse

    # The leak, made explicitly: each fold's weights fitted on the fold they are scored on.
    leaked = np.empty_like(target)
    fold_of_row = np.asarray([folds[e] for e in episode_ids.tolist()])
    for k in range(biv.STACKING_FOLDS):
        own = fold_of_row == k
        alpha, beta = biv.stack_weights(ridge[own], model[own], target[own])
        leaked[own] = alpha * ridge[own] + beta * model[own]

    floor = float((target**2).mean())
    assert biv.mse(leaked, target) < floor * 1e-3, "the leaked control did not reconstruct"
    assert honest > floor * 0.5, f"cross-fitted {honest:.4e} is near the leaked fit — T1 leaks"
    assert honest > biv.mse(leaked, target) * 100.0
    assert all(abs(b) < 0.9 for b in stack.betas), "a fold recovered its own sign out of fold"
    assert stack.mse == pytest.approx(biv.mse(stack.predictions, target))


def test_a_row_wise_fold_map_is_refused_over_the_row_tags() -> None:
    """THE guard on T1's folds, and the mutation it exists for.

    ``arange % num_folds`` is the row-level split ``episode_folds``' docstring forbids, and it is
    invisible from the outside: the header still says "folds over EPISODES", every fold still has
    weights, and the stack moves by a tenth of a percent while every fold's fit set now contains
    chunks from the very episodes it scores. So which rows a fold was fitted on is asserted over
    the tags — the idiom ``split_by_episode`` uses on the train/holdout split — rather than trusted
    to the loop that built the map.
    """
    episode_ids = np.asarray([f"ep-{i:02d}" for i in range(10) for _ in range(7)])
    folds = biv.episode_folds(episode_ids, biv.STACKING_FOLDS)
    by_episode = np.asarray([folds[e] for e in episode_ids.tolist()], dtype=np.int64)
    by_row = np.arange(episode_ids.shape[0], dtype=np.int64) % biv.STACKING_FOLDS

    biv.check_folds_are_episode_disjoint(episode_ids, by_episode, biv.STACKING_FOLDS)
    with pytest.raises(SystemExit, match="it then scores"):
        biv.check_folds_are_episode_disjoint(episode_ids, by_row, biv.STACKING_FOLDS)


def test_the_cross_fitted_stack_never_beats_the_in_sample_one(tmp_path: Path) -> None:
    """A general property of least squares, and therefore a cheap standing check on the whole T1
    path: two parameters fitted on the rows they score cannot do worse there than two fitted
    elsewhere. If cross-fitting ever came out ahead, the folds are not what they claim to be.

    The per-fold bookkeeping is asserted alongside it, because ``num_fit_episodes`` is the number
    that says out loud whether the folds were over episodes: with 10 holdout episodes in 5 folds it
    is 8, and a row-wise map would print 10.
    """
    _, predictions_path, table, _, rows = _prepared(tmp_path)
    train_mask, _ = rb.split_by_episode(table, set(HOLDOUT_EPISODES))
    train_x, train_y = table.select(train_mask)
    standardizer = rb.Standardizer.fit(train_x)
    weights = biv.ridge_weights(standardizer, train_x, train_y, biv.RIDGE_LAMBDA)

    target = table.targets[rows].reshape(-1, CHUNK_STEPS, NUM_JOINTS)
    ridge = (standardizer.design(table.states[rows]) @ weights).reshape(target.shape)
    model = np.stack(
        [
            np.asarray(p.predicted.targets, dtype=np.float64)
            for p in load_predictions_jsonl(predictions_path)
        ]
    )

    stack = biv.cross_fitted_stack(
        ridge, model, target, table.episode_ids[rows], biv.STACKING_FOLDS
    )
    alpha, beta = biv.stack_weights(ridge, model, target)
    in_sample = biv.mse(alpha * ridge + beta * model, target)

    assert len(stack.folds) == biv.STACKING_FOLDS
    assert sum(r.num_scored_chunks for r in stack.folds) == target.shape[0]
    assert stack.mse >= in_sample
    per_fold = len(HOLDOUT_EPISODES) // biv.STACKING_FOLDS
    assert [r.num_fit_episodes for r in stack.folds] == [
        len(HOLDOUT_EPISODES) - per_fold
    ] * biv.STACKING_FOLDS
    assert [r.num_fit_chunks for r in stack.folds] == [
        target.shape[0] - per_fold * CHUNKS_PER_EPISODE
    ] * biv.STACKING_FOLDS


# -- 5. T2 ----------------------------------------------------------------------------------------


def test_per_step_mse_has_one_entry_per_chunk_step_and_averages_to_the_total() -> None:
    """T2 is a decomposition, not a second metric: the 16 (here 4) numbers have to be the same
    error the headline reports, sliced by horizon. If they are not, "the model closes ground at
    long horizon" would be a statement about a different quantity than the one being decided on."""
    rng = np.random.default_rng(31)
    target = rng.normal(0.0, 1.0, (50, CHUNK_STEPS, NUM_JOINTS))
    pred = target + rng.normal(0.0, 0.1, target.shape)

    steps = biv.per_step_mse(pred, target)

    assert steps.shape == (CHUNK_STEPS,)
    assert float(steps.mean()) == pytest.approx(biv.mse(pred, target))
    assert biv.per_chunk_mse(pred, target).shape == (50,)
    assert float(biv.per_chunk_mse(pred, target).mean()) == pytest.approx(biv.mse(pred, target))


# -- 6. T3: the subset is ranked on constant velocity, and that choice matters ----------------------


def test_the_branch_point_subset_is_the_worst_quartile_by_rank() -> None:
    """A quartile has to be a quartile: ``np.quantile`` on a distribution with ties returns a
    subset of any size, and the printed "260 of 1040" would then be a different claim per run."""
    rng = np.random.default_rng(41)
    error = rng.random(1000)

    mask = biv.branch_point_mask(error, biv.BRANCH_POINT_QUANTILE)

    assert int(mask.sum()) == 250
    assert error[mask].min() >= error[~mask].max()

    tied = np.ones(1000)
    assert int(biv.branch_point_mask(tied, biv.BRANCH_POINT_QUANTILE).sum()) == 250


def test_ranking_the_subset_on_the_models_own_error_flatters_it() -> None:
    """The leak control for T3. Same code, one argument changed, and the answer has to move.

    The subset is chosen using the target, which PR-01 admits in ONE direction only — and only
    because the ranking predictor (constant velocity) is neither of the two being compared. Here
    the model and the ridge are drawn from the SAME error distribution and are independent, so an
    honest ranking leaves them level. Ranking on the model's own error and keeping the chunks it
    does BEST on hands it a large win out of nothing.

    The per-chunk error scale is drawn per chunk (some moments are easy for a predictor, some are
    not), which is what makes a subset choice able to move the answer at all — and is the property
    a real model has.

    If this test passed with the honest ranking too, the ranking predictor would not be doing any
    work and T3's one-directional admissibility argument would be empty.
    """
    rng = np.random.default_rng(51)
    n = 8000
    shape = (n, CHUNK_STEPS, NUM_JOINTS)
    target = rng.normal(0.0, 1.0, shape)

    def _predictor() -> np.ndarray:
        scale = rng.lognormal(-0.7, 0.5, n)[:, None, None]
        return target + scale * rng.normal(0.0, 1.0, shape)

    ridge, model, cv = _predictor(), _predictor(), _predictor()

    honest = biv.branch_point_mask(biv.per_chunk_mse(cv, target), biv.BRANCH_POINT_QUANTILE)
    # The leak: keep the chunks the MODEL happens to be best on, by ranking on its own error.
    model_error = biv.per_chunk_mse(model, target)
    leaked = biv.branch_point_mask(-model_error, biv.BRANCH_POINT_QUANTILE)

    honest_ratio = biv.mse(model[honest], target[honest]) / biv.mse(ridge[honest], target[honest])
    leaked_ratio = biv.mse(model[leaked], target[leaked]) / biv.mse(ridge[leaked], target[leaked])

    assert honest_ratio == pytest.approx(1.0, rel=0.1), "the neutral ranking is not neutral"
    assert leaked_ratio < 0.5, "selecting on the model's own error did not flatter it"


def test_ranking_t3_on_either_compared_predictor_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The leak above is a one-word edit at the call site, so the call site refuses it.

    ``branch_point_mask`` cannot tell what it was handed; only the caller knows whether the ranking
    predictor is one of the two being compared. That check therefore lives where the choice is
    made, and it fires for BOTH sides — ranking on the ridge's error is the mirror-image selection
    effect and would flatter the model just as hard.
    """
    predicted = {
        "const-velocity": np.zeros((4, CHUNK_STEPS, NUM_JOINTS)),
        biv.RIDGE_KEY: np.zeros((4, CHUNK_STEPS, NUM_JOINTS)),
        biv.MODEL_KEY: np.zeros((4, CHUNK_STEPS, NUM_JOINTS)),
    }
    target = np.ones((4, CHUNK_STEPS, NUM_JOINTS))

    assert biv.branch_point_ranking(predicted, target).shape == (4,)

    for compared in (biv.RIDGE_KEY, biv.MODEL_KEY):
        monkeypatch.setattr(biv, "T3_RANKING_PREDICTOR", compared)
        with pytest.raises(SystemExit, match="one of the two predictors it then compares"):
            biv.branch_point_ranking(predicted, target)


# -- 7. T4: the gripper subset ---------------------------------------------------------------------


def test_the_gripper_subset_is_exactly_the_chunks_that_cross_the_threshold() -> None:
    """Checked against ``wam.evaluation.gripper.crossings`` itself, so the subset cannot drift away
    from the threshold the rest of the repo binarizes at."""
    grip = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],  # never crosses
            [0.2, 0.4, 0.6, 0.8],  # opens
            [0.9, 0.8, 0.2, 0.1],  # closes
            [0.5, 0.5, 0.5, 0.5],  # sits exactly on the threshold, never crosses
        ]
    )

    mask = biv.gripper_transition_mask(grip, debounce=False)

    assert mask.tolist() == [False, True, True, False]
    assert mask.tolist() == [crossings(g) > 0 for g in grip]


def test_the_debounced_subset_refuses_a_channel_that_only_dithers() -> None:
    """The reading the pre-registration's phrasing leaves open, and the reason both are reported.

    A channel sitting ON the threshold produces a stream of raw crossings and reads as a busy
    gripper — the exact artefact that made ``gripper_accuracy`` look like a grasp metric on a dead
    channel. The hysteresis count in the same module refuses it. A reader who only saw the primary
    subset would be drawing a conclusion from noise, so the script prints both counts.
    """
    dither = np.array([[0.499, 0.501, 0.499, 0.501], [0.502, 0.498, 0.502, 0.498]])
    real = np.array([[0.1, 0.2, 0.8, 0.9]])

    assert biv.gripper_transition_mask(dither, debounce=False).all()
    assert not biv.gripper_transition_mask(dither, debounce=True).any()
    assert biv.gripper_transition_mask(real, debounce=True).all()


# -- 8. the archive gate ---------------------------------------------------------------------------


def test_the_archive_gate_passes_on_the_archived_numbers() -> None:
    lines = biv.check_archived_controls(
        biv.ARCHIVED_ZERO_DELTA_MSE, biv.ARCHIVED_MODEL_MSE, biv.ARCHIVED_RIDGE_ALL_MSE
    )
    assert len(lines) == 3
    assert all("OK" in line for line in lines)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_the_archive_gate_refuses_a_run_that_does_not_reproduce(index: int) -> None:
    """ "A run that does not reproduce them is void, not interpreted" — so it raises, and the
    caller never gets the chance to print a table under a warning nobody reads. Every one of the
    three controls has to be able to fire on its own."""
    values = [biv.ARCHIVED_ZERO_DELTA_MSE, biv.ARCHIVED_MODEL_MSE, biv.ARCHIVED_RIDGE_ALL_MSE]
    values[index] *= 1.001  # a tenth of a percent — far below anything a reader would notice

    with pytest.raises(SystemExit, match="VOID"):
        biv.check_archived_controls(*values)


def test_the_archive_gate_tolerates_only_the_written_precision() -> None:
    """The archived numbers carry seven significant digits, so the tolerance has to clear their
    rounding and nothing more. A drift in the sixth digit is a different measurement."""
    assert biv.check_archived_controls(
        biv.ARCHIVED_ZERO_DELTA_MSE * (1 + 2e-7),
        biv.ARCHIVED_MODEL_MSE,
        biv.ARCHIVED_RIDGE_ALL_MSE,
    )
    with pytest.raises(SystemExit, match="VOID"):
        biv.check_archived_controls(
            biv.ARCHIVED_ZERO_DELTA_MSE * (1 + 1e-5),
            biv.ARCHIVED_MODEL_MSE,
            biv.ARCHIVED_RIDGE_ALL_MSE,
        )


# -- 9. the verdict rule ---------------------------------------------------------------------------

_RIDGE = 1.0e-05
_FLAT_BETAS = [0.0, 0.01, -0.02, 0.0, 0.03]  # never consistently positive
_POSITIVE_BETAS = [0.4, 0.5, 0.45, 0.6, 0.02]  # 4 of 5 folds


def _stack(mse_stack: float, betas: list[float]) -> Any:
    """A ``CrossFittedStack`` carrying exactly what the decision rule reads off it."""
    folds = tuple(
        biv.StackFold(
            fold=k,
            alpha=1.0,
            beta=beta,
            num_fit_episodes=8,
            num_fit_chunks=160,
            num_scored_chunks=40,
        )
        for k, beta in enumerate(betas)
    )
    return biv.CrossFittedStack(
        predictions=np.zeros((1, CHUNK_STEPS, NUM_JOINTS)), mse=mse_stack, folds=folds
    )


_A_CASE: dict[str, Any] = {
    "mse_ridge": _RIDGE,
    "stack": _stack(_RIDGE, _FLAT_BETAS),  # buys nothing, beta never consistently positive
    "t3_model": 2.0e-05,
    "t3_ridge": 1.0e-05,  # the model loses where the thumb was on its scale
    "t4_gripper_model": 2.0e-04,
    "t4_gripper_ridge": 1.0e-04,  # and loses on the gripper channel
}
_B_CASE: dict[str, Any] = {
    "mse_ridge": _RIDGE,
    "stack": _stack(0.5e-05, _POSITIVE_BETAS),  # stacking the model in halves the error
    "t3_model": 0.5e-05,
    "t3_ridge": 1.0e-05,  # the model wins at the branch points
    "t4_gripper_model": 2.0e-04,
    "t4_gripper_ridge": 1.0e-04,
}


def test_verdict_a_needs_all_four_clauses() -> None:
    verdict = biv.decide(**_A_CASE)
    assert verdict.letter == "A"
    assert all(v is True for v in verdict.clauses.values())


@pytest.mark.parametrize(
    "override",
    [
        {"stack": _stack(0.5e-05, _FLAT_BETAS)},  # stacking buys a lot
        {"stack": _stack(_RIDGE, _POSITIVE_BETAS)},  # beta consistently positive
        {"t3_model": 0.5e-05},  # the model wins at the branch points
        {"t4_gripper_model": 0.5e-04},  # the model wins on the gripper channel
    ],
)
def test_breaking_any_single_clause_takes_verdict_a_away(override: dict[str, Any]) -> None:
    """Four clauses joined by AND: each one has to be load-bearing on its own, or the rule is
    shorter than the document says it is."""
    assert biv.decide(**{**_A_CASE, **override}).letter != "A"


def test_verdict_b_is_a_t3_win_with_t1_agreeing() -> None:
    verdict = biv.decide(**_B_CASE)
    assert verdict.letter == "B"


@pytest.mark.parametrize(
    "override",
    [
        {"stack": _stack(0.96e-05, _POSITIVE_BETAS)},  # T1 does not agree: stacking buys < 5 %
        {"stack": _stack(0.5e-05, [0.4, 0.5, 0.45, 0.0, 0.02])},  # only 3 of 5 folds
    ],
)
def test_a_t3_win_without_t1_is_not_verdict_b(override: dict[str, Any]) -> None:
    """ "If it wins here, that is weak evidence and must not be reported as a win without T1
    agreeing." A T3 win on its own is VERDICT C, and C makes no global claim."""
    assert biv.decide(**{**_B_CASE, **override}).letter == "C"


def test_clause_one_is_decided_on_the_cross_fitted_stack_and_cannot_take_a_float() -> None:
    """The substitution the printout cannot show: the LEAKED in-sample stack into clause 1.

    The two numbers sit three lines apart in ``main``, are both called a stack, and on the T-16
    holdout differ by 0.2 % — 6.177562e-06 leaked against 6.190756e-06 honest, either side of a
    0.95 * MSE_ridge threshold of 6.014354e-06 on a run where they happen to fall together. Here
    they are placed on opposite sides of it, so the swap is the difference between VERDICT A and
    VERDICT C rather than a digit nobody reads.

    ``decide`` therefore takes the cross-fitted OBJECT, which only ``cross_fitted_stack`` produces;
    the in-sample number is a bare float and cannot be passed where it belongs.
    """
    honest = 0.96e-05  # above 0.95 * ridge: the model buys < 5 %, clause holds
    leaked = 0.94e-05  # below it: reads as the model buying something
    assert honest > biv.STACK_IMPROVEMENT_FRACTION * _RIDGE > leaked

    assert biv.decide(**{**_A_CASE, "stack": _stack(honest, _FLAT_BETAS)}).letter == "A"
    assert biv.decide(**{**_A_CASE, "stack": _stack(leaked, _FLAT_BETAS)}).letter == "C"

    with pytest.raises(AttributeError):
        biv.decide(**{**_A_CASE, "stack": leaked})


def test_an_unevaluable_t4_makes_verdict_a_unreachable_but_leaves_b_alone() -> None:
    """An empty gripper subset is not a free pass on a clause VERDICT A requires. It is also not a
    problem for VERDICT B, which never mentions T4 — so the two have to move differently."""
    a_like = biv.decide(**{**_A_CASE, "t4_gripper_model": None, "t4_gripper_ridge": None})
    b_like = biv.decide(**{**_B_CASE, "t4_gripper_model": None, "t4_gripper_ridge": None})

    assert a_like.letter == "C"
    assert a_like.clauses["T4 model does not beat the ridge on the gripper channel"] is None
    assert any("INDETERMINATE" in line for line in a_like.lines)
    assert b_like.letter == "B"


# -- 10. the CLI ------------------------------------------------------------------------------------


def test_the_cli_runs_all_four_tests_and_prints_a_mechanical_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end. Everything PR-01 asks for has to come out of ONE invocation: a reader deciding
    on ~125 GPU-h needs the four tests, the controls and the verdict on one screen, or they will be
    compared across runs that need not agree."""
    root = _dataset(tmp_path / "ds")
    predictions = tmp_path / "predictions.jsonl"
    _write_predictions(predictions, root, HOLDOUT_EPISODES)

    record = _run_cli(root, predictions, tmp_path / "pr01.json")

    printed = capsys.readouterr().out
    assert "ARCHIVE GATE OFF" in printed
    for marker in ("T1 —", "T2 —", "T3 —", "T4 —", "VERDICT"):
        assert marker in printed

    assert record["verdict"] in ("A", "B", "C")
    assert record["num_holdout_chunks"] == len(HOLDOUT_EPISODES) * CHUNKS_PER_EPISODE
    assert record["num_holdout_episodes"] == len(HOLDOUT_EPISODES)
    assert len(record["t1"]["folds"]) == biv.STACKING_FOLDS
    assert (
        sum(f["num_scored_chunks"] for f in record["t1"]["folds"]) == record["num_holdout_chunks"]
    )
    assert set(record["full_holdout_mse"]) == set(biv.PREDICTOR_ORDER)
    assert all(len(v) == CHUNK_STEPS for v in record["t2_per_step_mse"].values())
    assert record["t3"]["num_chunks"] == round(record["num_holdout_chunks"] * 0.25)
    assert record["t4"]["num_chunks"] > 0
    assert record["t4"]["num_chunks_debounced"] <= record["t4"]["num_chunks"]
    assert record["thresholds"]["STACK_IMPROVEMENT_FRACTION"] == biv.STACK_IMPROVEMENT_FRACTION
    # The leaked in-sample stack is reported next to the cross-fitted one, and cannot be worse.
    assert record["t1"]["mse_stack_in_sample_leaked"] <= record["t1"]["mse_stack"]

    # Every fold is fitted on 4/5 of the EPISODES, not on 4/5 of the rows.
    per_fold = len(HOLDOUT_EPISODES) // biv.STACKING_FOLDS
    assert [f["num_fit_episodes"] for f in record["t1"]["folds"]] == [
        len(HOLDOUT_EPISODES) - per_fold
    ] * biv.STACKING_FOLDS

    # The clause line a reader decides on carries the CROSS-FITTED number, not the leaked one.
    assert record["t1"]["mse_stack"] != record["t1"]["mse_stack_in_sample_leaked"]
    clause = next(line for line in printed.splitlines() if line.startswith("  T1  MSE_stack"))
    assert biv._fmt(record["t1"]["mse_stack"]) in clause
    assert biv._fmt(record["t1"]["mse_stack_in_sample_leaked"]) not in clause


def test_the_cli_reports_the_cross_fitted_stack_this_test_computes_itself(tmp_path: Path) -> None:
    """T1's number, rebuilt from the dataset rather than read back out of the run's own record.

    The record can only say what the run believed; this recomputes the ridge, the model array, the
    episode fold map and the stack from the episodes on disk and requires the run to match. It is
    the assertion that would fail if ``main`` stacked something other than the two predictors it
    printed, or folded the rows some other way, or reported the in-sample fit as the primary.
    """
    root = _dataset(tmp_path / "ds")
    predictions = tmp_path / "predictions.jsonl"
    _write_predictions(predictions, root, HOLDOUT_EPISODES)

    record = _run_cli(root, predictions, tmp_path / "pr01.json")

    table, rows, target, _ = _holdout_view(root, predictions)
    train_mask, _ = rb.split_by_episode(table, set(HOLDOUT_EPISODES))
    train_x, train_y = table.select(train_mask)
    standardizer = rb.Standardizer.fit(train_x)
    weights = biv.ridge_weights(standardizer, train_x, train_y, biv.RIDGE_LAMBDA)
    ridge = (standardizer.design(table.states[rows]) @ weights).reshape(target.shape)
    model = np.stack(
        [
            np.asarray(p.predicted.targets, dtype=np.float64)
            for p in load_predictions_jsonl(predictions)
        ]
    )
    stack = biv.cross_fitted_stack(
        ridge, model, target, table.episode_ids[rows], biv.STACKING_FOLDS
    )

    assert record["full_holdout_mse"][biv.RIDGE_KEY] == pytest.approx(
        biv.mse(ridge, target), rel=1e-12
    )
    assert record["full_holdout_mse"][biv.MODEL_KEY] == pytest.approx(
        biv.mse(model, target), rel=1e-12
    )
    assert record["t1"]["mse_stack"] == pytest.approx(stack.mse, rel=1e-12)
    assert [f["beta"] for f in record["t1"]["folds"]] == pytest.approx(stack.betas, rel=1e-12)


def test_the_cli_ranks_t3_on_constant_velocity_over_dq(tmp_path: Path) -> None:
    """WHICH chunks T3 kept, recomputed from the dataset and compared row index for row index.

    This is the assertion the leak control on ``branch_point_mask`` cannot make. That one proves
    the function ranks what it is handed; this one proves what ``main`` hands it. Ranked on the
    model's own error the subset is still a quartile, still prints a threshold, and still carries
    T3's verdict clause — only the membership changes, so membership is what gets compared.

    It pins the constant-velocity input at the same time: built from ``q`` instead of ``dq`` the
    ranking is a different one and the selected rows move.
    """
    root = _dataset(tmp_path / "ds")
    predictions = tmp_path / "predictions.jsonl"
    _write_predictions(predictions, root, HOLDOUT_EPISODES)

    record = _run_cli(root, predictions, tmp_path / "pr01.json")

    table, rows, target, extras = _holdout_view(root, predictions)
    dq = table.states[rows][:, NUM_JOINTS : 2 * NUM_JOINTS]
    cv_error = biv.per_chunk_mse(biv.constant_velocity(dq, extras.dt_s[rows], CHUNK_STEPS), target)
    expected = biv.branch_point_mask(cv_error, biv.BRANCH_POINT_QUANTILE)

    assert record["t3"]["ranked_on"] == "const-velocity"
    assert record["t3"]["selected_chunks"] == np.flatnonzero(expected).tolist()
    assert record["t3"]["threshold"] == pytest.approx(float(cv_error[expected].min()), rel=1e-12)
    assert record["t3"]["mse"]["const-velocity"] == pytest.approx(
        biv.mse(
            biv.constant_velocity(dq, extras.dt_s[rows], CHUNK_STEPS)[expected], target[expected]
        ),
        rel=1e-12,
    )
    # A ranking on q would select a different quartile — otherwise the check above proves nothing.
    q_error = biv.per_chunk_mse(
        biv.constant_velocity(table.states[rows][:, :NUM_JOINTS], extras.dt_s[rows], CHUNK_STEPS),
        target,
    )
    q_mask = biv.branch_point_mask(q_error, biv.BRANCH_POINT_QUANTILE)
    assert not np.array_equal(q_mask, expected), "q and dq rank the same; the fixture is degenerate"


def test_the_t3_subset_does_not_move_when_only_the_model_changes(tmp_path: Path) -> None:
    """T3's subset must be a function of the DATA, not of the run being judged.

    Two predictions files over the same episodes: identical demonstrated targets, different model
    output. A subset ranked on constant velocity is byte-identical between them. A subset ranked on
    the model's own error is not — and that is the whole of T3's one-directional admissibility,
    stated as an experiment rather than as a sentence in a docstring.
    """
    root = _dataset(tmp_path / "ds")
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    _write_predictions(first, root, HOLDOUT_EPISODES, seed=1, noise=0.02)
    _write_predictions(second, root, HOLDOUT_EPISODES, seed=2, noise=0.30)

    a = _run_cli(root, first, tmp_path / "a.json")
    b = _run_cli(root, second, tmp_path / "b.json")

    assert a["full_holdout_mse"]["model"] != b["full_holdout_mse"]["model"], "same model twice"
    assert a["t3"]["selected_chunks"] == b["t3"]["selected_chunks"]
    assert a["t3"]["threshold"] == b["t3"]["threshold"]
    assert a["t3"]["mse"]["const-velocity"] == b["t3"]["mse"]["const-velocity"]
    assert a["t3"]["mse"][biv.RIDGE_KEY] == b["t3"]["mse"][biv.RIDGE_KEY]


def test_the_cli_extrapolates_from_dq_and_scores_zero_on_a_momentum_dataset(
    tmp_path: Path,
) -> None:
    """The calibration of the load-bearing reference row, THROUGH the CLI.

    On a dataset where every chunk is exactly ``dq * dt`` repeated, the no-parameter reference has
    to score 0 — and only a run that sliced the ``dq`` columns can. ``q`` and ``dq`` are
    independent draws here, so a run reading ``q`` returns an ordinary six-digit number: the same
    shape of answer, about a different question, on the row Reading B is argued from.
    """
    root = _momentum_dataset(tmp_path / "ds")
    predictions = tmp_path / "predictions.jsonl"
    _write_predictions(predictions, root, HOLDOUT_EPISODES)

    record = _run_cli(root, predictions, tmp_path / "pr01.json")

    zero_delta = record["full_holdout_mse"]["zero-delta"]
    assert zero_delta > 0.0
    assert record["full_holdout_mse"]["const-velocity"] < zero_delta * 1e-8
    assert all(v < zero_delta * 1e-8 for v in record["t2_per_step_mse"]["const-velocity"])


def test_the_cli_says_so_when_no_chunk_crosses_the_gripper_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dead gripper channel is the likely case on this data, and it must not read as a passed
    clause. T4 then measures nothing, its clause is INDETERMINATE, and VERDICT A is unreachable —
    which is a result about the dataset and has to be printed as one."""
    root = _dataset(tmp_path / "ds", seed=7, transitions=False)
    predictions = tmp_path / "predictions.jsonl"
    _write_predictions(predictions, root, HOLDOUT_EPISODES)

    record = _run_cli(root, predictions, tmp_path / "pr01.json")

    printed = capsys.readouterr().out
    assert "no chunk crosses the threshold" in printed
    assert "INDETERMINATE" in printed

    assert record["t4"]["num_chunks"] == 0
    assert record["t4"]["gripper_channel_admissible"] is False
    assert record["verdict"] == "C"


def test_the_cli_evaluates_t4_on_the_raw_crossing_subset(tmp_path: Path) -> None:
    """WHICH of the two gripper subsets the clause is decided on, on data where they differ.

    A channel that dithers on the threshold produces raw crossings and no debounced transitions.
    On the T-16 holdout that difference is 17 chunks against 0 — the difference between an
    evaluated clause and an INDETERMINATE one, i.e. between VERDICT A being reachable and not. So
    the fixture is built so the two counts come apart, and the RAW count is the one the record must
    report as T4's.
    """
    root = _dataset(tmp_path / "ds", seed=11, transitions=True, dither=True)
    predictions = tmp_path / "predictions.jsonl"
    _write_predictions(predictions, root, HOLDOUT_EPISODES)

    record = _run_cli(root, predictions, tmp_path / "pr01.json")

    _, rows, _, extras = _holdout_view(root, predictions)
    gripper = extras.gripper_targets[rows]
    raw = np.asarray([crossings(g) > 0 for g in gripper])

    assert record["t4"]["primary_debounce"] is False
    assert record["t4"]["num_chunks"] == int(raw.sum())
    assert record["t4"]["selected_chunks"] == np.flatnonzero(raw).tolist()
    assert 0 < record["t4"]["num_chunks_debounced"] < record["t4"]["num_chunks"], (
        "the fixture no longer separates raw from debounced crossings"
    )
    assert record["t4"]["gripper_channel_mse"]["model"] > 0.0


def test_the_cli_scores_the_models_own_gripper_predictions(tmp_path: Path) -> None:
    """T4's clause is decided on the model's gripper output, so that is what has to be read.

    The predictions here carry the demonstrated channel plus a known constant, which fixes the
    channel MSE at that constant squared. A run that scored ``target`` against ``target`` prints
    0.000000e+00 — a perfect model result, under a passing archive gate, on the clause — and every
    structural assertion about T4 still holds. An arithmetic one does not.
    """
    offset = 0.03
    root = _dataset(tmp_path / "ds")
    predictions = tmp_path / "predictions.jsonl"
    _write_predictions(predictions, root, HOLDOUT_EPISODES, gripper_offset=offset)

    record = _run_cli(root, predictions, tmp_path / "pr01.json")

    assert record["t4"]["num_chunks"] > 0
    assert record["t4"]["gripper_channel_mse"]["model"] == pytest.approx(offset**2, rel=1e-4)


def test_a_model_that_hands_back_the_demonstrated_gripper_channel_is_refused(
    tmp_path: Path,
) -> None:
    """The degenerate case of the above, and the one that looks like a triumph.

    No model reproduces a float32 target exactly on every chunk of a holdout. Bit-identical
    predictions are what reading ``target`` where ``predicted`` was meant looks like from outside,
    so the run refuses to score them rather than printing the perfect number it would produce.
    """
    root = _dataset(tmp_path / "ds")
    predictions = tmp_path / "predictions.jsonl"
    _write_predictions(predictions, root, HOLDOUT_EPISODES, gripper_offset=0.0)

    with pytest.raises(SystemExit, match="bit-identical to the demonstrated channel"):
        _run_cli(root, predictions, tmp_path / "pr01.json")


def test_the_gripper_ridge_is_fitted_on_the_train_episodes_only(tmp_path: Path) -> None:
    """T4's OTHER side. The gripper channel needs its own ridge, and that is a second fit — with a
    second chance to be fitted on the rows it scores.

    Rebuilt here from the train episodes and compared to what the run reported, plus the leak
    control that gives the comparison teeth: the same solve fitted on the holdout scores strictly
    better on the holdout, so "fitted on train" is a measurable claim rather than a comment.
    """
    root = _dataset(tmp_path / "ds")
    predictions = tmp_path / "predictions.jsonl"
    _write_predictions(predictions, root, HOLDOUT_EPISODES)

    record = _run_cli(root, predictions, tmp_path / "pr01.json")

    table, rows, _, extras = _holdout_view(root, predictions)
    train_mask, _ = rb.split_by_episode(table, set(HOLDOUT_EPISODES))
    subset = np.asarray(record["t4"]["selected_chunks"], dtype=np.int64)
    holdout_states = table.states[rows]
    holdout_gripper = extras.gripper_targets[rows]

    honest = rb.Standardizer.fit(table.states[train_mask])
    honest_weights = biv.ridge_weights(
        honest, table.states[train_mask], extras.gripper_targets[train_mask], biv.RIDGE_LAMBDA
    )
    honest_mse = biv.mse(
        (honest.design(holdout_states) @ honest_weights)[subset], holdout_gripper[subset]
    )

    leaked = rb.Standardizer.fit(holdout_states)
    leaked_weights = biv.ridge_weights(leaked, holdout_states, holdout_gripper, biv.RIDGE_LAMBDA)
    leaked_mse = biv.mse(
        (leaked.design(holdout_states) @ leaked_weights)[subset], holdout_gripper[subset]
    )

    reported = record["t4"]["gripper_channel_mse"][biv.RIDGE_KEY]
    assert reported == pytest.approx(honest_mse, rel=1e-12)
    assert leaked_mse < honest_mse, "fitting on the holdout did not help; the control is empty"


def test_the_cli_refuses_a_plain_episode_id_list(tmp_path: Path) -> None:
    """``bench_ridge_baseline`` accepts one because it only needs the split. PR-01 compares the
    model chunk by chunk, so an id list is a missing input, not a degraded mode."""
    root = _dataset(tmp_path / "ds")
    holdout = tmp_path / "holdout.txt"
    holdout.write_text("\n".join(HOLDOUT_EPISODES) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="not a readable predictions.jsonl"):
        biv.main(
            [
                "--dataset", str(root),
                "--holdout", str(holdout),
                "--chunk-steps", str(CHUNK_STEPS),
                "--archive-gate", "off",
            ]
        )  # fmt: skip


def test_the_cli_refuses_the_real_archived_numbers_on_synthetic_data(tmp_path: Path) -> None:
    """The gate is on by default, and the default is the one that matters: a run on the wrong
    dataset must die before it prints a verdict, not after."""
    root = _dataset(tmp_path / "ds")
    predictions = tmp_path / "predictions.jsonl"
    _write_predictions(predictions, root, HOLDOUT_EPISODES)

    with pytest.raises(SystemExit, match="VOID"):
        biv.main(
            [
                "--dataset", str(root),
                "--holdout", str(predictions),
                "--chunk-steps", str(CHUNK_STEPS),
            ]
        )  # fmt: skip
