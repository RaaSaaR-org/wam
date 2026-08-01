"""Tests for scripts/train_t16_lora.py — the T-16 requeue chain (M3, AC-04).

The cluster caps every job at 4 hours and requeues, so the deliverable run is a CHAIN of
processes sharing one ``--out-dir``. What is under test here is therefore not "does it train"
but "is a chain indistinguishable from one long process": the bitwise-equivalence test is the
one that protects every other property in this file.

Everything runs on CPU with the tiny backbone over ``datasets/mock-d1`` and the shipped
``configs/training/joint.yaml`` (which is dimensioned for exactly that dataset) — no Wan
weights, no GPU, no network.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from wam.training import JointTrainer, JointTrainingConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "train_t16_lora.py"
_JOINT_YAML = _REPO_ROOT / "configs" / "training" / "joint.yaml"
_GR00T_YAML = _REPO_ROOT / "configs" / "training" / "joint_gr00t.yaml"
_WAN_TRAINING_YAML = _REPO_ROOT / "configs" / "training" / "joint_wan_gr00t.yaml"
_WAN_MODEL_YAML = _REPO_ROOT / "configs" / "model" / "wan22_ti2v_5b.yaml"
_MOCK_D1 = _REPO_ROOT / "datasets" / "mock-d1"

# batch 24 over mock-d1's 200 samples -> 9 batches/epoch, so a 12-step run crosses an epoch
# boundary AND hits a short final batch. Both are exactly where a naive resume goes wrong.
BATCH_SIZE = 24
STEPS = 12
HALF = 6


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("train_t16_lora", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tw = _load_script()

pytestmark = pytest.mark.skipif(not _MOCK_D1.is_dir(), reason="datasets/mock-d1 not present")


def _argv(out_dir: Path, **overrides: Any) -> list[str]:
    """The flag set of a chain job, pointed at the CPU fixtures."""
    args: dict[str, Any] = {
        "--training-config": str(_JOINT_YAML),
        "--dataset": str(_MOCK_D1),
        "--out-dir": str(out_dir),
        "--steps": str(STEPS),
        "--batch-size": str(BATCH_SIZE),
        "--log-every": "1",
        "--device": "cpu",
    }
    args.update({k: v for k, v in overrides.items() if v is not None})
    argv: list[str] = []
    for key, value in args.items():
        argv.append(key)
        if value != "":  # "" marks a store_true flag
            argv.append(str(value))
    return argv


def _step_records(out_dir: Path) -> list[dict[str, Any]]:
    """Per-step loss entries from the run log, stripped of run-identity keys."""
    lines = (out_dir / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    return [
        {k: v for k, v in record.items() if k not in ("kind", "run_id", "config_hash")}
        for record in records
        if record.get("kind") == "step"
    ]


def _final_state(out_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint = tw.CheckpointManager(out_dir).latest()
    assert checkpoint is not None
    weights = load_file(str(checkpoint / tw.MODEL_FILENAME))
    payload = torch.load(
        checkpoint / tw.TRAINER_STATE_FILENAME, map_location="cpu", weights_only=False
    )
    return weights, payload


def _stop_after(monkeypatch: pytest.MonkeyPatch, *, step_label: int, signum: int) -> None:
    """Send ``signum`` to ourselves from inside the step whose label is ``step_label``.

    Stands in for Slurm's ``--signal=B:USR1@300`` (and for preemption's SIGTERM): the signal
    lands MID-run, not between invocations, which is the case the boundary check exists for.
    """
    original = JointTrainer.step

    def patched(self: JointTrainer, batch: Any, **kwargs: Any) -> dict[str, float]:
        entry = original(self, batch, **kwargs)
        if int(entry["step"]) == step_label:
            os.kill(os.getpid(), signum)
        return entry

    monkeypatch.setattr(JointTrainer, "step", patched)


# -- the chain is indistinguishable from one long run -----------------------------------------


def test_interrupted_run_matches_uninterrupted_bitwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """12 steps straight == 6 + checkpoint + fresh restore + 6, bit for bit.

    This is the test the whole requeue architecture rests on. If it ever fails, every T-16
    result is really a sequence of unrelated partial runs, and no amount of "the loss went
    down" tells the difference. It covers all three restored streams at once: weights,
    AdamW's exp_avg/exp_avg_sq, and the sampler position (the losses would diverge on step 7
    alone if the resumed run re-shuffled from the top).
    """
    straight, chained = tmp_path / "straight", tmp_path / "chained"

    assert tw.main(_argv(straight)) == 0

    with monkeypatch.context() as patch:
        _stop_after(patch, step_label=HALF - 1, signum=signal.SIGUSR1)
        assert tw.main(_argv(chained)) == 0
    first_chunk = tw.CheckpointManager(chained).latest()
    assert first_chunk is not None and first_chunk.name == "step-000006"
    assert not (chained / tw.DONE_FILENAME).exists()  # half-done must not look finished

    assert tw.main(_argv(chained, **{"--resume": "latest"})) == 0
    assert (chained / tw.DONE_FILENAME).is_file()

    straight_weights, straight_state = _final_state(straight)
    chained_weights, chained_state = _final_state(chained)

    assert chained_state["step"] == STEPS
    assert straight_weights.keys() == chained_weights.keys()
    differing = [
        name
        for name, tensor in straight_weights.items()
        if not torch.equal(tensor, chained_weights[name])
    ]
    assert differing == []

    # AdamW moments, not just the weights: identical parameters with drifted moments produce
    # different updates from step 13 onwards, i.e. the damage would only show up later.
    straight_moments = straight_state["trainer"]["optimizer"]["state"]
    chained_moments = chained_state["trainer"]["optimizer"]["state"]
    assert straight_moments.keys() == chained_moments.keys()
    for key, moments in straight_moments.items():
        for name in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(moments[name], chained_moments[key][name]), f"{key}.{name}"
        assert moments["step"] == chained_moments[key]["step"]

    straight_history, chained_history = _step_records(straight), _step_records(chained)
    assert [entry["step"] for entry in straight_history] == list(range(STEPS))
    assert straight_history == chained_history


def test_resume_keeps_the_config_hash_of_the_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every chunk of the chain logs the same config_hash (AC-04)."""
    out_dir = tmp_path / "chain"
    with monkeypatch.context() as patch:
        _stop_after(patch, step_label=HALF - 1, signum=signal.SIGUSR1)
        assert tw.main(_argv(out_dir)) == 0
    # --lr on the resuming job is deliberately ignored: the checkpoint's config wins, so the
    # chain cannot silently turn into two experiments sharing one checkpoint lineage.
    assert tw.main(_argv(out_dir, **{"--resume": "latest", "--lr": "0.05"})) == 0

    lines = (out_dir / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
    hashes = {json.loads(line)["config_hash"] for line in lines if line.strip()}
    assert len(hashes) == 1

    checkpoint = tw.CheckpointManager(out_dir).latest()
    assert checkpoint is not None
    config_dict, _metadata = tw._read_embedded_config(checkpoint / tw.MODEL_FILENAME)
    assert config_dict["lr"] == JointTrainingConfig.from_yaml(_JOINT_YAML).lr


# -- signals stop at a step boundary and always leave a checkpoint -----------------------------


@pytest.mark.parametrize("signum", [signal.SIGUSR1, signal.SIGTERM])
def test_signal_stops_at_the_next_step_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signum: int
) -> None:
    """A signal raised mid-run finishes the step, checkpoints, exits 0 and leaves no DONE."""
    out_dir = tmp_path / "signalled"
    before = signal.getsignal(signal.SIGINT)
    with monkeypatch.context() as patch:
        _stop_after(patch, step_label=2, signum=signum)  # during the 3rd step
        assert tw.main(_argv(out_dir)) == 0

    manager = tw.CheckpointManager(out_dir)
    checkpoint = manager.latest()
    assert checkpoint is not None
    assert checkpoint.name == "step-000003"  # the step in flight was completed, not discarded
    assert tw.CheckpointManager.is_complete(checkpoint)
    assert Path(os.path.realpath(out_dir / tw.LATEST_LINK)) == checkpoint
    assert not (out_dir / tw.DONE_FILENAME).exists()  # not finished -> the chain must requeue
    assert signal.getsignal(signal.SIGINT) is before  # handlers restored for the test session

    payload = torch.load(
        checkpoint / tw.TRAINER_STATE_FILENAME, map_location="cpu", weights_only=False
    )
    assert payload["step"] == 3
    assert payload["batch_in_epoch"] == 3  # the sampler position rides along, not just weights


def test_stop_signal_handler_only_sets_a_flag() -> None:
    """The handler must never raise — it runs at an arbitrary bytecode inside the step."""
    before = signal.getsignal(signal.SIGUSR1)
    stop = tw.StopSignal().install([signal.SIGUSR1])
    try:
        assert not stop.requested
        os.kill(os.getpid(), signal.SIGUSR1)
        os.kill(os.getpid(), signal.SIGUSR1)  # a second one must not escalate into an exception
        assert stop.requested and stop.count == 2
        assert stop.name == "SIGUSR1"
    finally:
        stop.restore()
    assert signal.getsignal(signal.SIGUSR1) is before


# -- sampler resume ---------------------------------------------------------------------------


def _dataset() -> Any:
    from wam.training import EpisodeDataset

    config = JointTrainingConfig.from_yaml(_JOINT_YAML)
    return EpisodeDataset(
        _MOCK_D1,
        camera=config.camera,
        num_frames=config.backbone.num_frames,
        chunk_steps=config.head.num_steps,
    )


def test_epoch_feeder_seek_is_exact_across_an_epoch_boundary() -> None:
    """17 consumed batches and a fresh feeder seeked to that position agree on what comes next.

    17 is chosen to overshoot one epoch (9 batches at batch 24 over 200 samples), because the
    resume that matters happens wherever the 4-hour cap falls — usually mid-epoch, sometimes
    across the reshuffle.
    """
    dataset = _dataset()
    device = torch.device("cpu")
    consumed = tw.EpochFeeder(dataset, batch_size=BATCH_SIZE, seed=0, device=device)
    stream = iter(consumed)
    for _ in range(17):
        next(stream)

    assert consumed.position == divmod(17, consumed.batches_per_epoch)
    assert consumed.position[0] == 1  # genuinely crossed into the next epoch's reshuffle

    resumed = tw.EpochFeeder(dataset, batch_size=BATCH_SIZE, seed=0, device=device)
    resumed.seek(*consumed.position)
    resumed_stream = iter(resumed)
    for _ in range(5):
        expected, actual = next(stream), next(resumed_stream)
        assert expected.keys() == actual.keys()
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, actual[key]), key
            else:
                assert value == actual[key], key


def test_epoch_feeder_reshuffles_per_epoch_and_covers_the_dataset() -> None:
    """Each epoch is a full permutation, and consecutive epochs differ (no fixed order)."""
    dataset = _dataset()
    feeder = tw.EpochFeeder(dataset, batch_size=BATCH_SIZE, seed=0, device=torch.device("cpu"))
    first, second = feeder.epoch_order(0), feeder.epoch_order(1)
    assert sorted(first) == list(range(len(dataset)))
    assert first != second
    assert feeder.epoch_order(0) == first  # pure function of (seed, epoch)
    other_seed = tw.EpochFeeder(dataset, batch_size=BATCH_SIZE, seed=1, device=torch.device("cpu"))
    assert other_seed.epoch_order(0) != first


def test_epoch_feeder_rejects_a_position_from_a_different_batch_size() -> None:
    feeder = tw.EpochFeeder(_dataset(), batch_size=BATCH_SIZE, seed=0, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="batches at batch_size"):
        feeder.seek(0, feeder.batches_per_epoch + 1)


# -- checkpoint store -------------------------------------------------------------------------


@pytest.fixture()
def trainer() -> JointTrainer:
    config = JointTrainingConfig.from_yaml(_JOINT_YAML)
    return JointTrainer(config)


def test_checkpoint_manager_prunes_and_leaves_no_partial_directories(
    tmp_path: Path, trainer: JointTrainer
) -> None:
    """5 saves at total_limit=3 keep the newest 3, and nothing half-written survives."""
    manager = tw.CheckpointManager(tmp_path, total_limit=3)
    for step in range(5):
        manager.save(trainer, step=step, epoch=0, batch_in_epoch=step, run_id="prune-test")

    kept = [path.name for path in manager.existing()]
    assert kept == ["step-000002", "step-000003", "step-000004"]
    leftovers = [
        child.name
        for child in (tmp_path / tw.CHECKPOINT_DIRNAME).iterdir()
        if child.name.endswith((".tmp", ".stale"))
    ]
    assert leftovers == []
    # The newest can never be the pruned one — that is what makes prune safe to call right
    # after the symlink swap.
    assert Path(os.path.realpath(tmp_path / tw.LATEST_LINK)).name == "step-000004"
    assert manager.latest() == manager.existing()[-1]


def test_latest_falls_back_to_the_newest_intact_checkpoint(
    tmp_path: Path, trainer: JointTrainer
) -> None:
    """A truncated newest checkpoint must not abort the chain — an older one still works."""
    manager = tw.CheckpointManager(tmp_path, total_limit=3)
    for step in range(3):
        manager.save(trainer, step=step, epoch=0, batch_in_epoch=step, run_id="fallback-test")

    newest = manager.latest()
    assert newest is not None and newest.name == "step-000002"
    (newest / tw.MODEL_FILENAME).unlink()  # a write killed between the two files

    assert not tw.CheckpointManager.is_complete(newest)
    assert manager.latest() is not None
    assert manager.latest().name == "step-000001"


def test_latest_survives_a_dangling_symlink(tmp_path: Path, trainer: JointTrainer) -> None:
    manager = tw.CheckpointManager(tmp_path, total_limit=3)
    for step in range(2):
        manager.save(trainer, step=step, epoch=0, batch_in_epoch=0, run_id="dangling-test")
    (tmp_path / tw.LATEST_LINK).unlink()
    os.symlink("checkpoints/step-999999", tmp_path / tw.LATEST_LINK, target_is_directory=True)

    assert manager.latest() is not None
    assert manager.latest().name == "step-000001"


def test_restore_rejects_a_checkpoint_missing_trainable_tensors(
    tmp_path: Path, trainer: JointTrainer
) -> None:
    """A silently incomplete restore would resume a head from its fresh random init."""
    manager = tw.CheckpointManager(tmp_path)
    path = manager.save(trainer, step=0, epoch=0, batch_in_epoch=0, run_id="strict-test")
    weights = load_file(str(path / tw.MODEL_FILENAME))
    dropped = next(name for name in weights if name.startswith("action_head."))

    from safetensors import safe_open

    with safe_open(str(path / tw.MODEL_FILENAME), framework="pt") as handle:
        metadata = handle.metadata()
    from safetensors.torch import save_file

    save_file(
        {k: v for k, v in weights.items() if k != dropped},
        str(path / tw.MODEL_FILENAME),
        metadata=metadata,
    )
    with pytest.raises(RuntimeError, match="TRAINABLE tensor"):
        manager.restore(trainer, path)


# -- the production checkpoint payload and the wall-clock budget -------------------------------


def test_adapter_only_chain_resumes_from_trainable_tensors_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--save-adapter-only writes the trainable tensors only and still resumes exactly.

    This is the production path for Wan: writing a frozen multi-GB base into every 30-minute
    checkpoint is what would make the interval unaffordable. The frozen parameters are
    therefore legitimately absent from the file and must come from the rebuilt backbone —
    which is precisely the gap ``CheckpointManager.restore`` has to allow without also
    allowing an absent HEAD.
    """
    out_dir = tmp_path / "adapters"
    with monkeypatch.context() as patch:
        _stop_after(patch, step_label=HALF - 1, signum=signal.SIGUSR1)
        assert tw.main(_argv(out_dir, **{"--save-adapter-only": ""})) == 0

    checkpoint = tw.CheckpointManager(out_dir).latest()
    assert checkpoint is not None
    saved = set(load_file(str(checkpoint / tw.MODEL_FILENAME)))
    config = JointTrainingConfig.model_validate(
        tw._read_embedded_config(checkpoint / tw.MODEL_FILENAME)[0]
    )
    reference = JointTrainer(config)
    assert saved == set(reference.model.trainable_state_dict())
    frozen = set(reference.model.frozen_parameter_names())
    assert frozen and not (saved & frozen)  # the frozen text tables really are left out

    assert tw.main(_argv(out_dir, **{"--save-adapter-only": "", "--resume": "latest"})) == 0
    assert (out_dir / tw.DONE_FILENAME).is_file()
    assert [entry["step"] for entry in _step_records(out_dir)] == list(range(STEPS))


def test_max_hours_stops_before_the_first_step_and_still_checkpoints(tmp_path: Path) -> None:
    """Out of wall clock == exit 0 with a checkpoint and no DONE, so the chain requeues.

    ``--max-hours 0`` is the degenerate form of the 4-hour cap: zero progress must still
    produce a restorable state rather than an empty directory the next job cannot resume.
    """
    out_dir = tmp_path / "no-time"
    assert tw.main(_argv(out_dir, **{"--max-hours": "0"})) == 0

    checkpoint = tw.CheckpointManager(out_dir).latest()
    assert checkpoint is not None and checkpoint.name == "step-000000"
    assert not (out_dir / tw.DONE_FILENAME).exists()

    # ...and the next job in the chain picks it up and finishes the budget.
    assert tw.main(_argv(out_dir, **{"--resume": "latest"})) == 0
    assert (out_dir / tw.DONE_FILENAME).is_file()


def test_grad_accum_consumes_one_batch_per_micro_step(tmp_path: Path) -> None:
    """--grad-accum N draws N batches per optimizer step (the point: a bigger effective batch
    than fits in VRAM), and the sampler position accounts for every one of them."""
    out_dir = tmp_path / "accum"
    assert tw.main(_argv(out_dir, **{"--grad-accum": "2", "--steps": "4"})) == 0

    checkpoint = tw.CheckpointManager(out_dir).latest()
    assert checkpoint is not None and checkpoint.name == "step-000004"
    payload = torch.load(
        checkpoint / tw.TRAINER_STATE_FILENAME, map_location="cpu", weights_only=False
    )
    assert payload["batch_in_epoch"] == 8
    history = _step_records(out_dir)
    assert len(history) == 4
    assert all(math.isfinite(entry["total"]) for entry in history)


# -- the I-8 rung flag and the training set recorded in the checkpoint -------------------------


def _rung_file(path: Path, ids: tuple[str, ...]) -> Path:
    path.write_text("# a rung, with a comment and a blank line\n\n" + "\n".join(ids) + "\n")
    return path


def test_train_episodes_selects_exactly_the_listed_episodes_and_records_them(
    tmp_path: Path,
) -> None:
    """A rung trains on its file and nothing else, and says so in its own provenance.

    The recorded list is what lets the evaluator prove a rung's holdout was unseen at all:
    a 40-episode run is not the complement of the holdout, so the complement proof — the only
    one that existed before I-8 — refuses it by construction.
    """
    rung = _rung_file(tmp_path / "rung.txt", ("d1-0002", "d1-0005"))
    out_dir = tmp_path / "rung-run"

    assert tw.main(_argv(out_dir, **{"--train-episodes": str(rung), "--steps": "2"})) == 0

    metadata = json.loads((out_dir / "run_metadata.json").read_text())
    # list_episodes order, not the file's order and not a set: the snapshot digest is
    # sequential, so the evaluator has to be able to replay this list verbatim.
    assert metadata["train_episode_ids"] == ["d1-0002", "d1-0005"]

    _config, embedded = tw._read_embedded_config(
        tw.CheckpointManager(out_dir).latest() / tw.MODEL_FILENAME
    )
    assert embedded.train_episode_ids == ("d1-0002", "d1-0005")
    assert embedded.dataset_snapshot_ref == metadata["dataset_snapshot_ref"]


def test_a_run_without_the_rung_flag_still_records_what_it_trained_on(tmp_path: Path) -> None:
    """The field is not conditional on --train-episodes. A full run records its full set, so
    'no ids recorded' means exactly one thing: a checkpoint written before I-8."""
    out_dir = tmp_path / "full-run"
    assert tw.main(_argv(out_dir, **{"--steps": "2"})) == 0

    metadata = json.loads((out_dir / "run_metadata.json").read_text())
    assert metadata["train_episode_ids"] == [f"d1-{i:04d}" for i in range(8)]


def test_train_episodes_overlapping_the_holdout_is_fatal(tmp_path: Path) -> None:
    """The leak has to die at the PRODUCING end too.

    Belt and braces against the evaluator's own check: if a rung file naming a held-out episode
    is allowed to reach a checkpoint, the only thing standing between it and a published number
    is somebody remembering to point --holdout at the right file.
    """
    rung = _rung_file(tmp_path / "leaky.txt", ("d1-0002", "d1-0006"))
    holdout = _rung_file(tmp_path / "holdout.txt", ("d1-0006", "d1-0007"))

    with pytest.raises(SystemExit, match="share 1 episode"):
        tw.main(
            _argv(
                tmp_path / "nope",
                **{"--train-episodes": str(rung), "--exclude-episodes": str(holdout)},
            )
        )


def test_train_episodes_listing_an_absent_episode_is_fatal(tmp_path: Path) -> None:
    """Mirrors --exclude-episodes: a rung file that does not line up with the dataset is a
    typo, not a smaller rung, and silently training on 39 of 40 would go unnoticed."""
    rung = _rung_file(tmp_path / "ghost.txt", ("d1-0002", "d1-9999"))

    with pytest.raises(SystemExit, match="absent from"):
        tw.main(_argv(tmp_path / "nope", **{"--train-episodes": str(rung)}))


def test_resuming_a_chain_against_a_different_episode_set_is_fatal(tmp_path: Path) -> None:
    """The copy-paste that would otherwise produce a wrong-but-self-consistent proof.

    Three I-8 rungs share one dataset root and differ only in --train-episodes. Resuming
    rung-120's --out-dir with rung-40's file loads rung-120's weights and stamps rung-40's
    snapshot ref onto the final checkpoint — after which the evaluator's disjointness proof
    PASSES on a lie. ``_report_config_drift`` cannot catch it: the episode list is not part of
    the config, so both jobs have the identical config hash.
    """
    first = _rung_file(tmp_path / "a.txt", ("d1-0000", "d1-0001", "d1-0002"))
    second = _rung_file(tmp_path / "b.txt", ("d1-0003", "d1-0004", "d1-0005"))
    out_dir = tmp_path / "chain"

    assert tw.main(_argv(out_dir, **{"--train-episodes": str(first), "--steps": "2"})) == 0
    (out_dir / tw.DONE_FILENAME).unlink()  # pretend the chain was preempted, not finished

    with pytest.raises(SystemExit, match="REFUSING TO RESUME"):
        tw.main(
            _argv(
                out_dir, **{"--train-episodes": str(second), "--resume": "latest", "--steps": "4"}
            )
        )
    # ...and resuming with the SAME rung still works, or the guard would have broken the chain.
    assert (
        tw.main(
            _argv(out_dir, **{"--train-episodes": str(first), "--resume": "latest", "--steps": "4"})
        )
        == 0
    )


def test_the_rung_flag_is_reported_in_the_dry_run_line(tmp_path: Path) -> None:
    """The dry run is the login-node pre-flight; if it does not say which rung it resolved,
    the one thing that distinguishes three otherwise identical submissions is invisible."""
    import contextlib
    import io

    rung = _rung_file(tmp_path / "rung.txt", ("d1-0002", "d1-0005"))
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert (
            tw.main(_argv(tmp_path / "dry", **{"--train-episodes": str(rung), "--dry-run": ""}))
            == 0
        )

    assert "2 train episodes" in buffer.getvalue()
    assert "rung.txt" in buffer.getvalue()


# -- sentinels and the flag surface ------------------------------------------------------------


def test_done_sentinel_short_circuits(tmp_path: Path) -> None:
    """A finished run must not train one more chunk when the scheduler requeues it anyway."""
    out_dir = tmp_path / "finished"
    out_dir.mkdir()
    (out_dir / tw.DONE_FILENAME).write_text("{}\n")

    assert tw.main(_argv(out_dir, **{"--resume": "latest"})) == 0
    assert not (out_dir / tw.CHECKPOINT_DIRNAME).exists()
    assert not (out_dir / "training_log.jsonl").exists()


def test_resume_latest_without_any_checkpoint_starts_fresh(tmp_path: Path) -> None:
    """The first job of a chain passes the same --resume latest as the tenth."""
    manager = tw.CheckpointManager(tmp_path / "empty")
    assert tw._resolve_resume(manager, "latest") is None
    assert tw._resolve_resume(manager, "none") is None


def test_resume_with_an_unusable_explicit_path_fails_loudly(tmp_path: Path) -> None:
    manager = tw.CheckpointManager(tmp_path)
    with pytest.raises(SystemExit, match="not a complete checkpoint"):
        tw._resolve_resume(manager, str(tmp_path / "nope"))


def test_backbone_source_is_ignored_for_a_backbone_without_checkpoint_path(
    tmp_path: Path,
) -> None:
    """--backbone-source is a machine-local path; a backbone kind without that field says so
    instead of failing extra='forbid' validation."""
    training = {"backbone": {"kind": "tiny", "feature_dim": 64}}
    block = tw._resolve_backbone_block(
        training, None, source="/scratch/weights", allow_download=True
    )
    assert "checkpoint_path" not in block
    assert "allow_download" not in block
    assert "device" not in block  # tiny holds no weights of its own; nothing to place


def test_device_flag_reaches_the_backbone_block_not_just_the_trainer() -> None:
    """--device must override the backbone's own device, or load() lands on the wrong one.

    --backbone-config REPLACES the training YAML's backbone block, so the device declared there
    is gone and WanBackboneConfig falls back to its "cuda" default. The backbone is built with
    load=True before JointTrainer moves the model, so this is not cosmetic: it materializes the
    frozen tower on a device the user explicitly did not ask for.
    """
    training = {"backbone": {"kind": "wan_i2v", "device": "cuda"}}
    assert (
        tw._resolve_backbone_block(training, None, source=None, allow_download=False, device="cpu")[
            "device"
        ]
        == "cpu"
    )
    # ...and with no --device the config's own choice stands.
    assert (
        tw._resolve_backbone_block(training, None, source=None, allow_download=False, device=None)[
            "device"
        ]
        == "cuda"
    )


def test_dim_mismatch_between_backbone_and_head_is_not_patched(tmp_path: Path) -> None:
    """The head width must never be silently rewritten to fit a swapped backbone."""
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(
        'wam_config_version: "0.1.0"\n'
        "backbone:\n"
        "  kind: tiny\n"
        "  feature_dim: 128\n"  # joint.yaml's head.feature_dim is 64
        "  patch_size: 8\n"
        "  num_heads: 4\n"
        "  num_frames: 4\n"
        "  image_hw: [64, 64]\n"
        "  state_embedding_dim: 32\n"
    )
    argv = _argv(tmp_path / "out", **{"--backbone-config": str(backbone_yaml), "--dry-run": ""})
    with pytest.raises(ValueError, match="head.feature_dim"):
        tw.main(argv)


@pytest.mark.skipif(
    not (_WAN_TRAINING_YAML.is_file() and _WAN_MODEL_YAML.is_file()),
    reason="Wan configs not present",
)
def test_shipped_wan_config_pair_splices_and_validates(tmp_path: Path) -> None:
    """The two YAMLs the sbatch pairs must produce one valid config — checked without weights.

    They are edited independently (model config = architecture + LoRA surface, training config
    = shapes + optimizer), so the failure mode is a dim that agrees with neither. Catching it
    here costs a second; catching it on the cluster costs a queue slot on an H200.
    """
    args = tw._parse_args(
        [
            "--backbone-config",
            str(_WAN_MODEL_YAML),
            "--backbone-source",
            "/scratch/models/Wan2.2-TI2V-5B",
            "--training-config",
            str(_WAN_TRAINING_YAML),
            "--dataset",
            str(_MOCK_D1),
            "--out-dir",
            str(tmp_path),
        ]
    )
    config = tw._build_yaml_config(args)
    assert config.backbone.kind == "wan_i2v"
    # The machine-local weight path is injected from the CLI, never from the committed YAML.
    assert config.backbone.checkpoint_path == "/scratch/models/Wan2.2-TI2V-5B"
    assert config.head.feature_dim == config.backbone.feature_dim
    assert config.state.embedding_dim == config.backbone.state_embedding_dim


def test_sbatch_flag_set_parses_in_a_subprocess(tmp_path: Path) -> None:
    """The EXACT flag set cluster/discoverer/50_train_t16.sbatch passes, on CPU, no weights.

    A typo in a flag name only surfaces on the cluster otherwise — after the job has queued,
    been scheduled and burned a slot to die in argparse.
    """
    backbone_yaml = tmp_path / "tiny_backbone.yaml"
    backbone_yaml.write_text(  # the tiny twin of configs/model/wan22_ti2v_5b.yaml
        'wam_config_version: "0.1.0"\n'
        "backbone:\n"
        "  kind: tiny\n"
        "  feature_dim: 64\n"
        "  patch_size: 20\n"
        "  depth: 2\n"
        "  num_heads: 4\n"
        "  num_frames: 9\n"
        "  image_hw: [120, 160]\n"
        "  text_vocab: 256\n"
        "  max_text_tokens: 16\n"
        "  state_embedding_dim: 32\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--backbone-config",
            str(backbone_yaml),
            "--backbone-source",
            str(tmp_path / "weights"),
            "--training-config",
            str(_GR00T_YAML),
            "--dataset",
            str(_MOCK_D1),
            "--camera",
            "ego",
            "--out-dir",
            str(tmp_path / "run"),
            "--resume",
            "latest",
            "--checkpoint-every-min",
            "30",
            "--checkpoints-total-limit",
            "3",
            "--save-adapter-only",
            "--device",
            "cuda",  # a string in the config until a trainer is built; --dry-run builds none
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "dry run OK" in result.stdout
    assert "backbone=tiny" in result.stdout
    assert "checkpoint_path" in result.stdout  # the splice ran and reported the ignored flag


def _sbatch_trainer_flags(path: Path) -> list[str]:
    """The ``--flag`` names in an sbatch's ``python "${TRAIN}" ... &`` invocation.

    Read out of the file rather than copied into the test, so a flag renamed in one place and
    not the other fails here instead of on an H200 after the job has queued.
    """
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('python "${TRAIN}"'))
    end = next(i for i, line in enumerate(lines[start:], start) if line.rstrip().endswith("&"))
    block = " ".join(lines[start : end + 1])
    return [token for token in block.split() if token.startswith("--")]


def test_i8_rung_sbatch_flag_set_parses_in_a_subprocess(tmp_path: Path) -> None:
    """The EXACT flag set cluster/discoverer/55_train_i8_rung.sbatch passes, on CPU.

    55 is submitted three times against the deliverable allocation. A typo in --train-episodes
    surfaces only after the job has queued, been scheduled and burned an H200 slot to die in
    argparse — and the rung it was supposed to train is then simply missing from the curve.
    """
    sbatch = _REPO_ROOT / "cluster" / "discoverer" / "55_train_i8_rung.sbatch"
    flags = _sbatch_trainer_flags(sbatch)
    assert "--train-episodes" in flags, "55 no longer passes the rung file"
    assert set(flags) >= set(_sbatch_trainer_flags(sbatch.with_name("50_train_t16.sbatch"))), (
        "55 dropped a flag 50_train_t16.sbatch passes — the rungs must differ from the T-16 "
        "run in the training SET only"
    )

    backbone_yaml = tmp_path / "tiny_backbone.yaml"
    backbone_yaml.write_text(  # the tiny twin of configs/model/wan22_ti2v_5b.yaml
        'wam_config_version: "0.1.0"\n'
        "backbone:\n"
        "  kind: tiny\n"
        "  feature_dim: 64\n"
        "  patch_size: 20\n"
        "  depth: 2\n"
        "  num_heads: 4\n"
        "  num_frames: 9\n"
        "  image_hw: [120, 160]\n"
        "  text_vocab: 256\n"
        "  max_text_tokens: 16\n"
        "  state_embedding_dim: 32\n"
    )
    rung = _rung_file(tmp_path / "i8_train_002.txt", ("d1-0000", "d1-0001"))
    holdout = _rung_file(tmp_path / "holdout.txt", ("d1-0006", "d1-0007"))
    values = {
        "--backbone-config": str(backbone_yaml),
        "--backbone-source": str(tmp_path / "weights"),
        "--training-config": str(_GR00T_YAML),
        "--dataset": str(_MOCK_D1),
        "--exclude-episodes": str(holdout),
        "--train-episodes": str(rung),
        "--seed": "0",
        "--steps": "20000",
        "--camera": "ego",
        "--out-dir": str(tmp_path / "run"),
        "--resume": "latest",
        "--checkpoint-every-min": "30",
        "--checkpoints-total-limit": "3",
        "--save-adapter-only": None,
        "--device": "cuda",  # a string in the config until a trainer is built; --dry-run builds none
    }
    assert list(values) == flags, "55's flag list drifted away from this test"

    argv: list[str] = []
    for flag, value in values.items():
        argv.append(flag)
        if value is not None:
            argv.append(value)
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), *argv, "--dry-run"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "dry run OK" in result.stdout
    assert "2 train episodes, 2 held out" in result.stdout
