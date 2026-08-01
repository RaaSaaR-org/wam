"""Tests for `scripts/rescore_archived.py` — re-scoring a finished run without retraining it.

The script exists because the AC-07 table is currently mixed-mode: `t16-lora-seed0` was
re-measured with the real frame window (T-29), the other two arms are still tiled-only, and a
table with one windowed row and two tiled rows is not a comparison. Re-scoring the two archived
arms is what makes it one again.

So the two tests that carry the most weight are not "does it produce a number". They are:

  --verify-tiled       A re-score is only trustworthy if the same code path, in the mode the
                       archive was measured in, reproduces the archived number; otherwise the
                       --frame-history number is a fresh measurement wearing a re-score's name
                       and the delta it reports is partly the tooling. The control is pinned
                       from BOTH sides — it must pass on an archive its own checkpoint produced
                       and fail when the checkpoint is not the one that produced it. A control
                       that cannot fail is decoration.
  --frame-history      The flag has to reach the model, not just the run name. A wiring break
                       reports "the window makes no difference", which is the most expensive
                       possible false negative: it is the answer the mixed-mode table would be
                       corrected TO.

Everything runs on CPU with the shipped `configs/training/*.yaml`, the fixture pattern
`test_eval_t16.py` established: module-scoped, tiny, no GPU, no network, no 16 MB checkpoint,
no real dataset.

The dataset is recorded here rather than taken from `datasets/mock-d1`, and that is forced by
what is being tested: every frame of every mock-d1 episode is byte-identical to the first
(measured — max |frame_i - frame_0| is 0.0 across all 8 episodes and both cameras), so a real
`num_frames` window and one tiled frame are the SAME input there and no value-level
frame-history test is possible on it. That is also why `test_eval_t16.py`'s frame-history test
can only assert the run name. `MockRobot` renders a dot whose column encodes `q[0]`, so driving
`q[0]` across its range gives episodes whose frames actually move.

The archived runs are synthesized: an untrained model saved as a flat `checkpoint.safetensors`
with `train_episode_ids=None`, plus the tiled `predictions.jsonl` and `bench.json` its own
forward pass produces — exactly the shape `runs/t18-real-ablation-seed0` and
`runs/d1-full-gen-seed0` have on disk. Those archived predictions are built from the library
primitives (`build_eval_pairs` / `evaluate_policy`), the way `overfit_d1.py` and
`run_ablation.py` built the real ones, and never by calling the script under test — an archive
produced by the thing being verified would make the control circular.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wam.data import EpisodeWriter
from wam.interfaces.schema import ActionChunk, ActionMode
from wam.robot import MockRobot

_REPO_ROOT = Path(__file__).resolve().parent.parent
_JOINT_YAML = _REPO_ROOT / "configs" / "training" / "joint.yaml"
_ACTION_ONLY_YAML = _REPO_ROOT / "configs" / "training" / "action_only.yaml"

#: Matched to both shipped configs: 6 joints, 64x64 frames, one camera named ``front``.
NUM_JOINTS = 6
IMAGE_HW = (64, 64)
CAMERA = "front"
FPS = 20.0
CHUNK_STEPS = 8
PREFIX_STEPS = 1
ITERATIONS = 14
#: Per-step delta on q[0], applied once per iteration (``PREFIX_STEPS``). Large enough that the
#: rendered dot moves ~2 columns between frames — a window and a tiled still have to be visibly
#: different inputs, or the frame-history test proves nothing — and small enough that
#: ``ITERATIONS * SWEEP`` stays inside MockRobot's ``[-pi, pi]``, since a saturated q[0] parks
#: the dot and the last frames stop moving.
SWEEP = 0.2

EPISODES = ("ep-0000", "ep-0001", "ep-0002")
#: Two of the three, so the holdout is not the whole dataset.
HOLDOUT = ("ep-0001", "ep-0002")


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rs = _load("rescore_archived")


def _record_episode(episode_dir: Path, episode_id: str, seed: int) -> None:
    """One episode whose frames MOVE, driven by a sweep of ``q[0]``."""
    robot = MockRobot(num_joints=NUM_JOINTS, seed=seed, cameras=(CAMERA,), image_hw=IMAGE_HW)
    rng = np.random.default_rng(seed)
    with EpisodeWriter(episode_dir, episode_id, robot.spec, FPS, "greife die rote Tasse") as writer:
        for _ in range(ITERATIONS):
            writer.add_state(robot.read_state())
            writer.add_frame(CAMERA, robot.render_frames(1)[CAMERA][0], robot.sim_time_ns)
            targets = rng.uniform(-0.02, 0.02, (CHUNK_STEPS, NUM_JOINTS)).astype(np.float32)
            targets[:, 0] = SWEEP
            chunk = ActionChunk(
                mode=ActionMode.JOINT_DELTA,
                targets=targets,
                gripper_target=rng.uniform(0.0, 1.0, CHUNK_STEPS).astype(np.float32),
                dt_s=0.05,
            )
            writer.add_action(chunk, PREFIX_STEPS, robot.sim_time_ns)
            robot.execute(chunk, PREFIX_STEPS)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("dataset") / "moving"
    root.mkdir(parents=True)
    for seed, episode_id in enumerate(EPISODES):
        _record_episode(root / episode_id, episode_id, seed)
    return root


def test_the_fixture_episodes_actually_move(dataset: Path) -> None:
    """Guards the guard. Every frame-history assertion below is worthless on a dataset whose
    frames never change — which is exactly the trap `datasets/mock-d1` is (max |Δ| 0.0 between
    any two of its frames), and the reason this dataset is recorded rather than reused."""
    from wam.data import EpisodeReader

    frames = EpisodeReader(dataset / EPISODES[0]).read_frames(CAMERA)
    consecutive = [
        float(np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16)).max())
        for i in range(1, len(frames))
    ]
    assert min(consecutive) > 0.0, "frames are static — a window and a tiled still are equal"


def _dataset_hash(dataset: Path) -> str:
    from wam.data.episode import list_episodes

    return rs.ev.dataset_snapshot_hash(dataset, list_episodes(dataset))


def _build_model(kind: str, seed: int):
    """An untrained model of either kind, deterministic in ``seed``.

    Untrained is deliberate: what is under test is the re-scoring machinery, and a random-init
    model exercises every path a fine-tuned one does at a fraction of the wall time. The
    ``seed`` is what makes the "wrong checkpoint" test possible at all.
    """
    torch.manual_seed(seed)
    if kind == "joint":
        from wam.training.joint import JointTrainingConfig, JointWorldActionModel

        config = JointTrainingConfig.from_yaml(_JOINT_YAML)
        model = JointWorldActionModel(config)
    else:
        from wam.training.action_only import ActionOnlyConfig, ActionOnlyModel

        config = ActionOnlyConfig.from_yaml(_ACTION_ONLY_YAML)
        model = ActionOnlyModel(config)
    model.eval()
    return model, config


def _archive(
    run_dir: Path,
    kind: str,
    dataset: Path,
    *,
    holdout: tuple[str, ...] = HOLDOUT,
    checkpoint_seed: int | None = None,
) -> Path:
    """Synthesize a run directory shaped exactly like the two archived AC-07 arms.

    Flat ``checkpoint.safetensors`` at the run root, ``train_episode_ids=None`` (the shape of
    every pre-I-8 checkpoint), a ``dataset_snapshot_ref`` over the whole dataset, and the TILED
    ``predictions.jsonl`` + ``bench.json`` the model's own forward pass produces.

    ``checkpoint_seed`` writes DIFFERENT weights than the ones that produced the predictions.
    That is not a curiosity: it is the only way to show ``--verify-tiled`` detects a re-score
    that is not reproducing its archive, as opposed to merely re-reading a json.
    """
    from wam.evaluation import (
        bench_metrics,
        build_eval_pairs,
        evaluate_policy,
        save_predictions_jsonl,
    )
    from wam.interfaces import RunMetadata
    from wam.training._utils import save_checkpoint

    run_dir.mkdir(parents=True, exist_ok=True)
    model, config = _build_model(kind, seed=0)

    pairs = []
    for episode_id in sorted(holdout):
        pairs.extend(build_eval_pairs(dataset / episode_id, config.camera, config.head.num_steps))
    predictions = evaluate_policy(model, pairs)
    save_predictions_jsonl(predictions, run_dir / "predictions.jsonl")
    bench = bench_metrics(predictions, run_name=run_dir.name)
    (run_dir / "bench.json").write_text(bench.to_json() + "\n")
    (run_dir / "bench.md").write_text(bench.render_markdown())

    saved = model if checkpoint_seed is None else _build_model(kind, checkpoint_seed)[0]
    metadata = RunMetadata.create(
        run_dir.name,
        config,
        checkpoint_ref=str(run_dir / rs.FLAT_CHECKPOINT),
        dataset_snapshot_ref=_dataset_hash(dataset),
        git_commit="0" * 40,
    )
    assert metadata.train_episode_ids is None, "the archived shape is train_episode_ids=None"
    save_checkpoint(saved, config, run_dir / rs.FLAT_CHECKPOINT, metadata)
    return run_dir


def _holdout_file(tmp_path: Path, ids: tuple[str, ...], name: str = "holdout.txt") -> Path:
    path = tmp_path / name
    path.write_text("# a comment line, and a blank one\n\n" + "\n".join(ids) + "\n")
    return path


def _run(run_dir: Path, out: Path, holdout: Path, dataset: Path, *extra: str) -> int:
    return rs.main(
        [
            "--run-dir", str(run_dir),
            "--out", str(out),
            "--dataset", str(dataset),
            "--holdout", str(holdout),
            "--device", "cpu",
            *extra,
        ]
    )  # fmt: skip


@pytest.fixture(scope="module")
def holdout_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _holdout_file(tmp_path_factory.mktemp("split"), HOLDOUT)


@pytest.fixture(scope="module")
def archived(dataset: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """One archived run per checkpoint kind — the shapes of the two arms still to be re-scored."""
    root = tmp_path_factory.mktemp("archived")
    return {
        "joint": _archive(root / "t18-like", "joint", dataset),
        "action_only": _archive(root / "d1-like", "action_only", dataset),
    }


def _predicted_chunks(out: Path) -> list:
    return [
        json.loads(line)["predicted"]["targets"]
        for line in (out / "predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]


# -- it re-scores, and it re-scores the right thing -------------------------------------------


@pytest.mark.parametrize("kind", ["joint", "action_only"])
def test_rescores_an_archived_run_and_writes_every_artifact(
    archived: dict[str, Path], holdout_file: Path, dataset: Path, tmp_path: Path, kind: str
) -> None:
    """Both archived arms have to be scorable by ONE entry point, or the AC-07 table gets
    re-scored by two scripts and the comparison inherits their differences."""
    out = tmp_path / f"{kind}-scored"

    assert _run(archived[kind], out, holdout_file, dataset) == 0

    for name in ("predictions.jsonl", "e1.json", "e1.md", "bench.json", "bench.md", "timing.json"):
        assert (out / name).is_file(), name
    assert not (out / "UNVERIFIED_DATASET").exists()

    episode_ids = {
        json.loads(line)["episode_id"]
        for line in (out / "predictions.jsonl").read_text().splitlines()
        if line.strip()
    }
    assert episode_ids == set(HOLDOUT), "re-scored something other than the archived holdout"


@pytest.mark.parametrize("kind", ["joint", "action_only"])
def test_the_model_kind_is_sniffed_from_the_checkpoint_not_declared(
    archived: dict[str, Path], holdout_file: Path, dataset: Path, tmp_path: Path, kind: str
) -> None:
    """No CLI flag picks the loader. The two are not interchangeable, and a mispaired one can
    restore a subset of the tensors and still hand back a plausible chunk — a wrong number that
    looks exactly like a right one. The config travels inside the file; the caller does not."""
    out = tmp_path / f"{kind}-sniff"
    assert _run(archived[kind], out, holdout_file, dataset) == 0
    assert json.loads((out / "rescore.json").read_text())["model_kind"] == kind


def test_sniffing_refuses_a_checkpoint_that_is_neither_kind(tmp_path: Path) -> None:
    """Guessing is the failure mode being designed out: an unrecognised config shape must stop
    the run rather than default to a loader that might partially succeed."""
    from safetensors.torch import save_file

    from wam.interfaces import RunMetadata
    from wam.training._utils import CHECKPOINT_CONFIG_KEY, CHECKPOINT_METADATA_KEY

    path = tmp_path / "odd.safetensors"
    metadata = RunMetadata.create("odd", {"a": 1}, git_commit="0" * 40)
    save_file(
        {"w": torch.zeros(2)},
        str(path),
        metadata={
            CHECKPOINT_CONFIG_KEY: json.dumps({"backbone": {}, "head": {}}),
            CHECKPOINT_METADATA_KEY: json.dumps(metadata.to_dict(), sort_keys=True),
        },
    )

    with pytest.raises(SystemExit, match="neither 'action_encoder'"):
        rs.sniff_checkpoint(path)


# -- the control ------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["joint", "action_only"])
def test_verify_tiled_reproduces_the_archive(
    archived: dict[str, Path], holdout_file: Path, dataset: Path, tmp_path: Path, kind: str
) -> None:
    """The control, from the side that must pass.

    Same checkpoint, same holdout, same tiled readout, same device -> the re-score has to come
    back bit-for-bit, not merely within tolerance. Anything less and the +frame_history delta
    produced by this same code path is partly tooling. (On the real archives exactness holds
    only on the device they were produced on — both recorded ``mps``; here the archive is
    produced on CPU, so bit-for-bit is the honest expectation.)
    """
    out = tmp_path / f"{kind}-verified"

    assert _run(archived[kind], out, holdout_file, dataset, "--verify-tiled") == 0

    verify = json.loads((out / "rescore.json").read_text())["verify"]
    assert verify["passed"]
    assert verify["bit_identical"], f"max |Δ| {verify['max_abs_prediction_delta']}"
    assert verify["chunks_compared"] > 0


def test_verify_tiled_fails_when_the_checkpoint_did_not_produce_the_archive(
    holdout_file: Path, dataset: Path, tmp_path: Path
) -> None:
    """The control, from the side that makes it a control.

    A check that only ever passes proves nothing. Here the archived predictions come from one
    model and the checkpoint holds another's weights — precisely the "this number was not
    produced by this file" case the control exists to catch, and one that no amount of
    re-reading bench.json would reveal.
    """
    run_dir = _archive(tmp_path / "mismatched", "action_only", dataset, checkpoint_seed=17)

    with pytest.raises(SystemExit, match="does NOT reproduce the archived number"):
        _run(run_dir, tmp_path / "nope", holdout_file, dataset, "--verify-tiled")

    # The failing evidence is written, not discarded: the numbers that disagreed ARE the finding.
    verify = json.loads((tmp_path / "nope" / "rescore.json").read_text())["verify"]
    assert verify["passed"] is False
    assert verify["max_abs_prediction_delta"] > 0.0


def test_verify_tiled_fails_on_an_archived_mse_that_moved(
    archived: dict[str, Path], holdout_file: Path, dataset: Path, tmp_path: Path
) -> None:
    """The gate is on the number, not on a checksum of the file: an archive whose recorded mse
    is 1 % away from what the checkpoint reproduces is not the archive of this checkpoint."""
    run_dir = tmp_path / "edited"
    shutil.copytree(archived["action_only"], run_dir)
    bench = json.loads((run_dir / "bench.json").read_text())
    bench["mse"] = bench["mse"] * 1.01
    (run_dir / "bench.json").write_text(json.dumps(bench))

    with pytest.raises(SystemExit, match="does NOT reproduce the archived number"):
        _run(run_dir, tmp_path / "nope", holdout_file, dataset, "--verify-tiled")


def test_verify_tiled_is_refused_with_frame_history(
    archived: dict[str, Path], holdout_file: Path, dataset: Path, tmp_path: Path
) -> None:
    """The archived numbers are TILED. Checking a windowed re-score against them would fail on
    exactly the effect being measured — and a caller who then widened the tolerance to make it
    pass would have disabled the control to accommodate the finding."""
    with pytest.raises(SystemExit, match="--verify-tiled with --frame-history"):
        _run(
            archived["joint"],
            tmp_path / "nope",
            holdout_file,
            dataset,
            "--verify-tiled",
            "--frame-history",
        )
    assert not (tmp_path / "nope").exists()  # refused before anything was written


def test_verify_tiled_needs_something_to_verify_against(
    archived: dict[str, Path], holdout_file: Path, dataset: Path, tmp_path: Path
) -> None:
    """A control with no baseline is not a control, and silently degrading to "scored fine" is
    how a missing archive turns into a claim of reproduction."""
    run_dir = tmp_path / "no-bench"
    shutil.copytree(archived["joint"], run_dir)
    (run_dir / "bench.json").unlink()

    with pytest.raises(SystemExit, match="--verify-tiled needs"):
        _run(run_dir, tmp_path / "nope", holdout_file, dataset, "--verify-tiled")


# -- the measurement: --frame-history ----------------------------------------------------------


def test_frame_history_changes_the_predictions_and_names_itself(
    archived: dict[str, Path], holdout_file: Path, dataset: Path, tmp_path: Path
) -> None:
    """The flag has to reach the model, not just the run name.

    A wiring break here reports "the window makes no difference" — the most expensive possible
    false negative, because that is the answer the mixed-mode table would be corrected TO. The
    report has to say which mode it is for the same reason: two predictions.jsonl from one
    checkpoint differ only in what the policy was shown, so an unlabelled pair reads as two
    checkpoints.
    """
    from wam.evaluation import BenchReport

    tiled, windowed = tmp_path / "tiled", tmp_path / "windowed"

    assert _run(archived["joint"], tiled, holdout_file, dataset) == 0
    assert _run(archived["joint"], windowed, holdout_file, dataset, "--frame-history") == 0

    assert _predicted_chunks(tiled) != _predicted_chunks(windowed)
    assert BenchReport.from_json((windowed / "bench.json").read_text()).run_name.endswith(
        "+frame_history"
    )
    record = json.loads((windowed / "rescore.json").read_text())
    assert record["frame_history"] is True
    assert record["num_frames"] == 4  # the trained backbone.num_frames, not a hardcoded window


def test_the_default_stays_the_archived_tiled_path(
    archived: dict[str, Path], holdout_file: Path, dataset: Path, tmp_path: Path
) -> None:
    """OFF by default, same as eval_t16: the default invocation reproduces the archive rather
    than quietly redefining what every number before 2026-08-01 meant."""
    from wam.evaluation import BenchReport

    out = tmp_path / "default"
    assert _run(archived["action_only"], out, holdout_file, dataset) == 0

    bench = BenchReport.from_json((out / "bench.json").read_text())
    assert "+frame_history" not in bench.run_name
    timing = json.loads((out / "timing.json").read_text())
    assert timing["readout_tag"] == ""
    assert timing["frame_history"] is False


def test_the_comparison_line_reports_archived_recomputed_and_the_delta(
    archived: dict[str, Path],
    holdout_file: Path,
    dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The single line the script exists to print. It must name the archived value, the
    recomputed one AND the mode of each, because the archive is always tiled and the whole
    point of the exercise is that the two are not the same measurement."""
    out = tmp_path / "compared"
    assert _run(archived["joint"], out, holdout_file, dataset, "--frame-history") == 0

    printed = capsys.readouterr().out
    assert "skill_vs_repeat_pct  archived" in printed
    assert "(tiled)" in printed and "(real window)" in printed
    assert " pp" in printed

    record = json.loads((out / "rescore.json").read_text())
    assert record["delta_pp"] == pytest.approx(
        record["recomputed"]["skill_vs_repeat_pct"] - record["archived"]["skill_vs_repeat_pct"]
    )


# -- the holdout has to be the reviewed one ----------------------------------------------------


def test_refuses_a_holdout_that_is_not_the_reviewed_split(
    archived: dict[str, Path], dataset: Path, tmp_path: Path
) -> None:
    """The recovered holdout is the run's own; the committed split file is what says that
    holdout belongs in the AC-07 table. A re-score on a different one produces a number that
    would sit in that table next to numbers it cannot be compared with."""
    wrong = _holdout_file(tmp_path, ("ep-0000", "ep-0002"), "wrong.txt")

    with pytest.raises(SystemExit, match="holdout is not the reviewed one"):
        _run(archived["joint"], tmp_path / "nope", wrong, dataset)


def test_refuses_a_run_with_no_archived_predictions_to_recover_the_holdout_from(
    archived: dict[str, Path], holdout_file: Path, dataset: Path, tmp_path: Path
) -> None:
    """These checkpoints carry ``train_episode_ids=None``, so the predictions are the ONLY
    record of what was scored. Without them there is nothing to recover, and falling back to
    the split file would score episodes the run may never have been evaluated on."""
    run_dir = tmp_path / "no-preds"
    shutil.copytree(archived["joint"], run_dir)
    (run_dir / "predictions.jsonl").unlink()

    with pytest.raises(SystemExit, match="holdout of an archived run is recovered"):
        _run(run_dir, tmp_path / "nope", holdout_file, dataset)


def test_refuses_when_the_chunk_count_no_longer_matches_the_archive(
    archived: dict[str, Path], holdout_file: Path, dataset: Path, tmp_path: Path
) -> None:
    """Same episodes, different chunks is not a re-score. The episode ids can match perfectly
    while the chunking rule or the episode contents have moved underneath — and the resulting
    number would be compared against the archive as if nothing had."""
    run_dir = tmp_path / "extra-chunk"
    shutil.copytree(archived["joint"], run_dir)
    lines = (run_dir / "predictions.jsonl").read_text().splitlines()
    (run_dir / "predictions.jsonl").write_text("\n".join([*lines, lines[0]]) + "\n")

    with pytest.raises(SystemExit, match="chunk count changed"):
        _run(run_dir, tmp_path / "nope", holdout_file, dataset)


# -- provenance: the dataset has to be the trained one -----------------------------------------


def _tampered_dataset(dataset: Path, destination: Path) -> Path:
    """A copy of the dataset with one manifest edited — same ids, different bytes."""
    shutil.copytree(dataset, destination)
    manifest = destination / EPISODES[0] / "manifest.json"
    manifest.write_text(manifest.read_text().replace('"instruction"', '"Instruction"', 1))
    return destination


def test_refuses_a_dataset_that_no_longer_hashes_to_the_recorded_snapshot(
    archived: dict[str, Path], holdout_file: Path, dataset: Path, tmp_path: Path
) -> None:
    """``dataset_snapshot_ref`` is the one provenance claim these archived checkpoints make.
    Re-scoring against bytes that have moved produces a number that cannot be compared with the
    archived one, which is the entire purpose of running this script."""
    tampered = _tampered_dataset(dataset, tmp_path / "dataset")

    with pytest.raises(SystemExit, match="not the one this run trained on"):
        _run(archived["joint"], tmp_path / "nope", holdout_file, tampered)


def test_skip_dataset_check_scores_but_brands_the_output(
    archived: dict[str, Path], holdout_file: Path, dataset: Path, tmp_path: Path
) -> None:
    """The escape hatch has to leave a trace on disk, or an unverified number is
    indistinguishable from a verified one six months later."""
    tampered = _tampered_dataset(dataset, tmp_path / "dataset")
    out = tmp_path / "unverified"

    assert _run(archived["joint"], out, holdout_file, tampered, "--skip-dataset-check") == 0
    assert "not proven" in (out / "UNVERIFIED_DATASET").read_text()
    assert json.loads((out / "rescore.json").read_text())["dataset_checked"] is False


# -- the archive must survive the re-score -----------------------------------------------------


def test_refuses_to_write_its_output_into_the_run_dir(
    archived: dict[str, Path], holdout_file: Path, dataset: Path
) -> None:
    """Every artifact name is fixed, so ``--out == --run-dir`` overwrites the archived
    predictions.jsonl and bench.json — the baseline the re-score exists to be compared against.
    That is the one mistake this script must not be able to make: it destroys the evidence
    rather than producing a wrong number about it."""
    run_dir = archived["action_only"]
    before = (run_dir / "bench.json").read_text()

    with pytest.raises(SystemExit, match="--out may not be --run-dir"):
        _run(run_dir, run_dir, holdout_file, dataset)

    assert (run_dir / "bench.json").read_text() == before


def test_a_second_readout_refuses_to_overwrite_the_first_ones_artifacts(
    archived: dict[str, Path], holdout_file: Path, dataset: Path, tmp_path: Path
) -> None:
    """eval_t16's --out guard, reused rather than re-derived: scoring the windowed arm into the
    tiled arm's directory would leave an A/B of one arm against itself and nothing on disk to
    show it had happened."""
    out = tmp_path / "one-dir"
    assert _run(archived["joint"], out, holdout_file, dataset) == 0
    first = (out / "bench.json").read_text()

    with pytest.raises(SystemExit, match="already holds artifacts from a different readout"):
        _run(archived["joint"], out, holdout_file, dataset, "--frame-history")

    assert (out / "bench.json").read_text() == first
    # Re-running the SAME arm stays idempotent — a re-score after an interrupted pass has to work.
    assert _run(archived["joint"], out, holdout_file, dataset) == 0


# -- checkpoint resolution ---------------------------------------------------------------------


def test_resolve_prefers_the_flat_checkpoint_and_falls_back_to_step_dirs(
    archived: dict[str, Path], tmp_path: Path
) -> None:
    """The archived arms hold a flat ``checkpoint.safetensors``; ``train_t16_lora`` runs hold
    ``checkpoints/step-N/model.safetensors``. One entry point has to read both, or re-scoring
    the AC-07 table needs two scripts again."""
    run_dir = tmp_path / "layouts"
    shutil.copytree(archived["joint"], run_dir)
    flat = run_dir / rs.FLAT_CHECKPOINT
    assert rs.resolve_archived_checkpoint(run_dir, rs.AUTO) == flat

    step = run_dir / "checkpoints" / "step-000010"
    step.mkdir(parents=True)
    shutil.copy(flat, step / "model.safetensors")
    assert rs.resolve_archived_checkpoint(run_dir, rs.AUTO) == flat, "the flat file must win"

    flat.unlink()
    assert rs.resolve_archived_checkpoint(run_dir, rs.AUTO) == step / "model.safetensors"

    shutil.rmtree(run_dir / "checkpoints")
    with pytest.raises(SystemExit, match="not a restorable run"):
        rs.resolve_archived_checkpoint(run_dir, rs.AUTO)


def test_an_explicit_checkpoint_file_is_taken_as_given(
    archived: dict[str, Path], holdout_file: Path, dataset: Path, tmp_path: Path
) -> None:
    """``--checkpoint <file>`` must not be re-derived through a filename convention: an archived
    checkpoint can be called anything, and a re-score should never have to guess at a path it
    was told."""
    elsewhere = tmp_path / "weights-copy.safetensors"
    shutil.copy(archived["action_only"] / rs.FLAT_CHECKPOINT, elsewhere)
    out = tmp_path / "explicit"

    assert (
        _run(archived["action_only"], out, holdout_file, dataset, "--checkpoint", str(elsewhere))
        == 0
    )
    assert json.loads((out / "rescore.json").read_text())["checkpoint"] == str(elsewhere)


# -- the output has to be usable by the tools that read the AC-07 table -------------------------


def test_run_bench_compare_accepts_two_rescored_arms(
    archived: dict[str, Path],
    holdout_file: Path,
    dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point is a table that can be compared again, and ``run_bench.py --compare`` is
    what builds it — it refuses runs whose holdouts differ. So the artifacts this script writes
    have to be accepted by it unchanged, which is why the file names and the ``bench_metrics``
    call are eval_t16's rather than a second dialect of them.
    """
    rb = _load("run_bench")
    a, b = tmp_path / "arm-a", tmp_path / "arm-b"
    assert _run(archived["joint"], a, holdout_file, dataset, "--frame-history") == 0
    assert _run(archived["action_only"], b, holdout_file, dataset, "--frame-history") == 0

    monkeypatch.setattr(sys, "argv", ["run_bench.py", str(a), str(b), "--compare", "--no-write"])
    assert rb.main() == 0


def test_the_reports_round_trip_and_name_the_run_they_came_from(
    archived: dict[str, Path], holdout_file: Path, dataset: Path, tmp_path: Path
) -> None:
    """An artifact that cannot be parsed back cannot be re-scored or compared, and one that does
    not carry the run id is a number with no run attached."""
    from wam.evaluation import BenchReport, E1Report

    out = tmp_path / "roundtrip"
    assert _run(archived["joint"], out, holdout_file, dataset) == 0

    e1 = E1Report.from_json((out / "e1.json").read_text())
    bench = BenchReport.from_json((out / "bench.json").read_text())

    assert e1.mse >= 0.0
    assert bench.level in range(-1, 5)
    assert bench.run_name.startswith(archived["joint"].name)
