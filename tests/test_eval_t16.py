"""Tests for scripts/eval_t16.py — scoring a fine-tune on episodes it never saw.

Everything runs on CPU with the tiny backbone over ``datasets/mock-d1`` and the shipped
``configs/training/joint.yaml``, the same fixtures ``test_training_resume.py`` uses: no Wan
weights, no GPU, no network. The Wan-specific part of the script is one branch
(``adapter_only`` -> build the frozen base), and the tiny path exercises everything else.

The tests that matter most here are the refusals. An offline verdict is only worth reading if the
episodes it was computed on were genuinely held out, and "genuinely" cannot be taken on trust from
a filename: the trainer's ``dataset_snapshot_ref`` covers the episodes it actually trained on, so
recomputing it either matches or the split is not what it claims.

Since I-8 there are two proofs, and both are pinned below. A checkpoint that records its
``train_episode_ids`` is proven DISJOINT from the holdout; one that does not (every checkpoint
written before I-8, reconstructed here by the ``archived`` fixture) keeps the older COMPLEMENT
proof unchanged. The choice is made by the checkpoint, never by a flag — a guard that can be
switched off to silence a warning is not a guard, so ``--skip-split-check`` remains the only
escape hatch and it still brands its own output.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_JOINT_YAML = _REPO_ROOT / "configs" / "training" / "joint.yaml"
_MOCK_D1 = _REPO_ROOT / "datasets" / "mock-d1"

HOLDOUT = ("d1-0006", "d1-0007")
#: An I-8-shaped rung: a strict subset of the training set that is neither the whole dataset
#: nor the complement of HOLDOUT. d1-0003..d1-0005 are trained on by neither — which is exactly
#: what the complement proof cannot express and the disjointness proof can.
RUNG = ("d1-0000", "d1-0001", "d1-0002")
#: mock-d1 minus HOLDOUT: what the ``trained`` fixture is fed. It is written to a file and passed
#: as ``--train-episodes`` because the disjointness proof REFUSES to score without an external
#: witness — the checkpoint's ids and its snapshot hash are two fields of one self-description,
#: so checking them against each other cannot fail. Every fresh checkpoint records
#: ``train_episode_ids`` (``train_t16_lora.py``), so this is the normal path now and the
#: complement proof is reachable only through the ``archived`` fixture.
TRAINED = ("d1-0000", "d1-0001", "d1-0002", "d1-0003", "d1-0004", "d1-0005")
STEPS = 4
BATCH_SIZE = 16


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = _load("eval_t16")
tw = _load("train_t16_lora")

pytestmark = pytest.mark.skipif(not _MOCK_D1.is_dir(), reason="datasets/mock-d1 not present")


def _holdout_file(tmp_path: Path, ids: tuple[str, ...], name: str = "holdout.txt") -> Path:
    path = tmp_path / name
    path.write_text("# a comment line, and a blank one\n\n" + "\n".join(ids) + "\n")
    return path


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A tiny joint checkpoint trained with HOLDOUT excluded -> (run_dir, holdout file)."""
    root = tmp_path_factory.mktemp("t16")
    out_dir = root / "run"
    holdout = _holdout_file(root, HOLDOUT)
    rc = tw.main(
        [
            "--training-config", str(_JOINT_YAML),
            "--dataset", str(_MOCK_D1),
            "--out-dir", str(out_dir),
            "--exclude-episodes", str(holdout),
            "--steps", str(STEPS),
            "--batch-size", str(BATCH_SIZE),
            "--device", "cpu",
        ]
    )  # fmt: skip
    assert rc == 0
    return out_dir, holdout


@pytest.fixture(scope="module")
def trained_subset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A rung: trained on RUNG only, with HOLDOUT also excluded -> (run_dir, holdout file).

    Its training set is neither the dataset nor the complement of the holdout, so the proof
    that existed before I-8 refuses it by construction — which is the whole reason the
    disjointness path exists.
    """
    root = tmp_path_factory.mktemp("t16-rung")
    out_dir = root / "run"
    holdout = _holdout_file(root, HOLDOUT)
    rung = root / "rung.txt"
    rung.write_text("\n".join(RUNG) + "\n")
    rc = tw.main(
        [
            "--training-config", str(_JOINT_YAML),
            "--dataset", str(_MOCK_D1),
            "--out-dir", str(out_dir),
            "--exclude-episodes", str(holdout),
            "--train-episodes", str(rung),
            "--steps", str(STEPS),
            "--batch-size", str(BATCH_SIZE),
            "--device", "cpu",
        ]
    )  # fmt: skip
    assert rc == 0
    return out_dir, holdout


def _first_chunk(out: Path) -> list:
    """The first predicted chunk in a run's ``predictions.jsonl`` — the cheapest evidence that a
    readout flag reached the model rather than only the report's run_name."""
    line = (out / "predictions.jsonl").read_text().splitlines()[0]
    return json.loads(line)["predicted"]["targets"]


def _copy_run(run_dir: Path, destination: Path) -> Path:
    """A writable copy of a module-scoped run, so a test may tamper with its checkpoint."""
    import shutil

    shutil.copytree(run_dir, destination, symlinks=True)
    return destination


def _copy_dataset(destination: Path) -> Path:
    """A writable copy of mock-d1. Manifest bytes and relative names are preserved, so the
    dataset snapshot hash is identical and a checkpoint trained on the original still proves
    its split against the copy — which is what makes tampering tests possible at all."""
    import shutil

    shutil.copytree(_MOCK_D1, destination)
    return destination


def _rewrite_run_metadata(run_dir: Path, *, drop: tuple[str, ...] = (), **updates: Any) -> None:
    """Patch the RunMetadata embedded in a checkpoint, leaving the weights untouched."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    from wam.training._utils import CHECKPOINT_METADATA_KEY

    model_path = ev.resolve_checkpoint(run_dir, "latest")
    weights = load_file(str(model_path))
    with safe_open(str(model_path), framework="pt") as handle:
        metadata = dict(handle.metadata())
    record = json.loads(metadata[CHECKPOINT_METADATA_KEY])
    for key in drop:
        record.pop(key, None)
    record.update(updates)
    metadata[CHECKPOINT_METADATA_KEY] = json.dumps(record, sort_keys=True)
    save_file(weights, str(model_path), metadata=metadata)


@pytest.fixture(scope="module")
def archived(trained: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``trained`` with ``train_episode_ids`` REMOVED — the shape of every pre-I-8 checkpoint.

    Built by deletion rather than by an old code path on purpose: this is exactly what
    ``runs/t16-lora-seed0``'s embedded metadata looks like on disk, and it is the only way the
    complement proof is still reachable now that fresh runs always record their episode list.
    """
    copied = _copy_run(trained[0], tmp_path_factory.mktemp("t16-archived") / "run")
    _rewrite_run_metadata(copied, drop=("train_episode_ids",))
    return copied


def _run_eval(
    run_dir: Path,
    holdout: Path,
    out: Path,
    *extra: str,
    dataset: Path | None = None,
    train_episodes: tuple[str, ...] | None = TRAINED,
) -> int:
    """Run the evaluator, writing ``train_episodes`` out as the ``--train-episodes`` witness.

    The witness stands in for the reviewed, committed split file (``configs/splits/i8_train_*``).
    It defaults to the ``trained`` fixture's episode list because that is the common case; pass
    ``RUNG`` for the subset run. ``None`` omits the flag entirely — on the complement path the
    witness is optional (and cross-checked when given), on the disjointness path omitting it is
    the refusal this whole mechanism exists for.
    """
    witness: list[str] = []
    if train_episodes is not None:
        path = out.parent / f"{out.name}-train-episodes.txt"
        path.write_text("\n".join(train_episodes) + "\n")
        witness = ["--train-episodes", str(path)]
    return ev.main(
        [
            "--run-dir", str(run_dir),
            "--dataset", str(dataset or _MOCK_D1),
            "--holdout", str(holdout),
            "--device", "cpu",
            "--out", str(out),
            *witness,
            *extra,
        ]
    )  # fmt: skip


def test_scores_the_holdout_and_writes_every_artifact(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    run_dir, holdout = trained
    out = tmp_path / "scored"

    assert _run_eval(run_dir, holdout, out) == 0

    for name in ("predictions.jsonl", "e1.json", "e1.md", "bench.json", "bench.md"):
        assert (out / name).is_file(), name
    assert not (out / "UNPROVEN_SPLIT").exists()

    episode_ids = {
        json.loads(line)["episode_id"]
        for line in (out / "predictions.jsonl").read_text().splitlines()
        if line.strip()
    }
    assert episode_ids == set(HOLDOUT), "scored something other than the holdout"


def test_e1_and_bench_artifacts_parse_back(trained: tuple[Path, Path], tmp_path: Path) -> None:
    """Both reports must round-trip: an unreadable artifact cannot be re-scored or compared."""
    from wam.evaluation import BenchReport, E1Report

    run_dir, holdout = trained
    out = tmp_path / "roundtrip"
    _run_eval(run_dir, holdout, out)

    e1 = E1Report.from_json((out / "e1.json").read_text())
    bench = BenchReport.from_json((out / "bench.json").read_text())

    assert e1.mse >= 0.0
    assert bench.level in range(-1, 5)
    assert 0.0 <= bench.score <= 100.0


def test_refuses_a_holdout_that_was_not_excluded(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """The guard: a "holdout" made of episodes the run trained on is a training score."""
    run_dir, _ = trained
    wrong = _holdout_file(tmp_path, ("d1-0000", "d1-0001"), "wrong.txt")

    with pytest.raises(SystemExit, match="records training on holdout episode"):
        _run_eval(run_dir, wrong, tmp_path / "nope")


def test_refuses_a_partially_overlapping_holdout(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """One trained episode smuggled into the holdout is enough, and it must be named."""
    run_dir, _ = trained
    leaky = _holdout_file(tmp_path, (*HOLDOUT, "d1-0000"), "leaky.txt")

    with pytest.raises(SystemExit, match="records training on holdout episode"):
        _run_eval(run_dir, leaky, tmp_path / "nope")


def test_refuses_the_disjointness_proof_without_an_external_witness(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """The hole the witness exists to close: with no ``--train-episodes`` the recorded ids and
    the recorded hash are two fields of ONE self-description, so the check compares the
    checkpoint against itself and cannot fail. Refusing to score is the only safe answer, and it
    must not be reachable by simply omitting an argument."""
    run_dir, holdout = trained

    with pytest.raises(SystemExit, match="needs an external witness"):
        _run_eval(run_dir, holdout, tmp_path / "nope", train_episodes=None)


def test_refuses_a_witness_that_disagrees_with_the_checkpoint(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """The witness is only worth requiring if a wrong one is rejected: the split file is the
    reviewed artifact, and a checkpoint that trained on a different set is not the run it
    describes — even when that checkpoint is perfectly self-consistent."""
    run_dir, holdout = trained

    with pytest.raises(SystemExit, match="does not describe this checkpoint"):
        _run_eval(run_dir, holdout, tmp_path / "nope", train_episodes=RUNG)


def test_a_witness_is_accepted_and_cross_checked_on_the_complement_path(
    archived: Path, trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """``--train-episodes`` must be safe to pass on EVERY call, including the complement path.

    The first version of the witness fix REFUSED the flag here, reasoning that the complement
    proof does not consult it. That was a regression with teeth: it forced every caller to know
    which proof a checkpoint would take before it could build a command line, and all four eval
    sbatch files got it wrong in the other direction — none passed the flag at all, so
    ``62_eval_i8_curve.sbatch`` could not score a single rung. Accepting the file and checking it
    against the complement is strictly stronger than ignoring it and removes the fork.
    """
    _, holdout = trained
    assert _run_eval(archived, holdout, tmp_path / "cross", train_episodes=TRAINED) == 0


def test_a_witness_that_is_not_the_complement_is_refused_on_the_complement_path(
    archived: Path, trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """Accepting the witness is only worth anything if a wrong one still stops the run."""
    _, holdout = trained

    with pytest.raises(SystemExit, match="does not list the complement"):
        _run_eval(archived, holdout, tmp_path / "nope", train_episodes=RUNG)


def test_the_witness_comparison_counts_repeats(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """Multiset, not set: ``dataset_snapshot_ref`` is a hash over the recorded SEQUENCE.

    A checkpoint declaring an id twice hashes a different sequence than the witness names, so a
    set comparison would authorise a training set the reviewed file never described.
    """
    run_dir = _copy_run(trained[0], tmp_path / "dupe")
    _rewrite_run_metadata(run_dir, train_episode_ids=[*TRAINED, TRAINED[0]])
    _, holdout = trained

    with pytest.raises(SystemExit, match="does not describe this checkpoint"):
        _run_eval(run_dir, holdout, tmp_path / "nope", train_episodes=TRAINED)


def test_every_eval_sbatch_passes_the_split_witness() -> None:
    """The regression no test caught: ``verify_split`` grew a required argument and not one
    caller was updated, so four cluster jobs were one ``set -euo pipefail`` away from dying on
    their first eval — and the suite stayed green because it never reads the sbatch files.

    Scoring is the whole point of these four jobs; a flag they cannot omit belongs in a test.
    """
    jobs = sorted((_REPO_ROOT / "cluster" / "discoverer").glob("6?_eval_*.sbatch"))
    assert jobs, "no eval sbatch files found — did they move?"
    missing = [p.name for p in jobs if "--train-episodes" not in p.read_text()]
    assert not missing, f"eval jobs with no split witness: {missing}"


def test_skip_split_check_scores_but_marks_the_output(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """The escape hatch has to leave a trace, or an unproven number looks like a proven one."""
    run_dir, _ = trained
    wrong = _holdout_file(tmp_path, ("d1-0000",), "wrong.txt")
    out = tmp_path / "unproven"

    # No witness: --skip-split-check never consults one, and leaving it on the command line
    # would suggest something was checked.
    assert _run_eval(run_dir, wrong, out, "--skip-split-check", train_episodes=None) == 0

    assert (out / "predictions.jsonl").is_file()
    assert "not proven unseen" in (out / "UNPROVEN_SPLIT").read_text()


def test_rejects_a_holdout_episode_missing_from_the_dataset(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    run_dir, _ = trained
    ghost = _holdout_file(tmp_path, ("d1-0006", "d1-9999"), "ghost.txt")

    with pytest.raises(SystemExit, match="absent from"):
        _run_eval(run_dir, ghost, tmp_path / "nope")


def test_resolve_checkpoint_prefers_latest_then_falls_back(trained: tuple[Path, Path]) -> None:
    """A scratch purge or a truncated write can lose the symlink; the newest step still works."""
    run_dir, _ = trained

    via_link = ev.resolve_checkpoint(run_dir, "latest")
    assert via_link.name == "model.safetensors"

    link = run_dir / "latest"
    target = link.resolve()
    # Restored VERBATIM, and it is relative because train_t16_lora writes it relative. Putting an
    # absolute link back instead corrupts the module-scoped fixture from a distance: _copy_run
    # uses copytree(symlinks=True), so the copy's `latest` would still point into THIS run dir,
    # and the `archived` fixture's metadata rewrite would land on the original checkpoint —
    # silently moving every later test onto the complement proof.
    original = os.readlink(link)
    link.unlink()
    try:
        via_scan = ev.resolve_checkpoint(run_dir, "latest")
        assert via_scan.resolve() == (target / "model.safetensors").resolve()
    finally:
        link.symlink_to(original, target_is_directory=True)
    assert not os.path.isabs(os.readlink(link))


def test_resolve_checkpoint_rejects_an_empty_run_dir(tmp_path: Path) -> None:
    (tmp_path / "checkpoints").mkdir()
    with pytest.raises(SystemExit, match="no checkpoints"):
        ev.resolve_checkpoint(tmp_path, "latest")


def test_load_episode_ids_reads_lists_and_predictions_jsonl(tmp_path: Path) -> None:
    """The trainer's exclusion and this eval's holdout must read one file the same way."""
    from wam.evaluation import load_episode_ids

    plain = _holdout_file(tmp_path, ("ep-a", "ep-b"), "plain.txt")
    preds = tmp_path / "predictions.jsonl"
    preds.write_text('{"episode_id": "ep-a", "t_ns": 0}\n{"episode_id": "ep-b", "t_ns": 1}\n')

    assert load_episode_ids(plain) == {"ep-a", "ep-b"}
    assert load_episode_ids(preds) == {"ep-a", "ep-b"}
    assert load_episode_ids(plain) == tw._load_excluded_ids(plain)


# -- I-8 / T-32: the split proof generalised from complementarity to disjointness -------------


def test_scores_a_subset_run_whose_training_set_is_provably_disjoint(
    trained_subset: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole point of the change: an I-8 rung is scorable.

    Its training set is 3 of mock-d1's 8 episodes, so it is not the complement of anything —
    but the holdout is still provably unseen, which is the only claim that makes the number
    mean something. No UNPROVEN_SPLIT marker: this is a proof, not an escape hatch.
    """
    run_dir, holdout = trained_subset
    out = tmp_path / "rung-scored"

    assert _run_eval(run_dir, holdout, out, train_episodes=RUNG) == 0

    assert "split proven (disjoint): 3 train / 2 holdout" in capsys.readouterr().out
    assert not (out / "UNPROVEN_SPLIT").exists()
    episode_ids = {
        json.loads(line)["episode_id"]
        for line in (out / "predictions.jsonl").read_text().splitlines()
        if line.strip()
    }
    assert episode_ids == set(HOLDOUT)


def test_old_checkpoints_without_train_episode_ids_keep_the_complement_proof(
    archived: Path, trained: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every archived checkpoint predates the field, so it must still be judged by the rule it
    was recorded under — and that rule must still refuse what it always refused."""
    holdout = trained[1]

    assert _run_eval(archived, holdout, tmp_path / "archived", train_episodes=None) == 0
    assert "split proven (complement): 6 train / 2 holdout" in capsys.readouterr().out

    wrong = _holdout_file(tmp_path, ("d1-0000", "d1-0001"), "wrong.txt")
    with pytest.raises(SystemExit, match="REFUSING TO SCORE — split not provable \\(complement\\)"):
        _run_eval(archived, wrong, tmp_path / "nope", train_episodes=None)
    leaky = _holdout_file(tmp_path, (*HOLDOUT, "d1-0000"), "leaky.txt")
    with pytest.raises(SystemExit, match="REFUSING TO SCORE — split not provable \\(complement\\)"):
        _run_eval(archived, leaky, tmp_path / "nope", train_episodes=None)


def test_refuses_when_the_recorded_training_set_intersects_the_holdout(
    trained_subset: tuple[Path, Path], tmp_path: Path
) -> None:
    """The property that must survive the generalisation.

    ``dataset_snapshot_ref`` is left alone, so the aggregate hash check is not what catches
    this — only the disjointness test is. If this ever stops raising, the change has become
    "a flag that turns the guard off" and every rung number is worthless.
    """
    run_dir = _copy_run(trained_subset[0], tmp_path / "leaky-run")
    _rewrite_run_metadata(run_dir, train_episode_ids=[*RUNG, "d1-0006"])
    holdout = _holdout_file(tmp_path, HOLDOUT, "holdout.txt")

    # Witness and checkpoint AGREE here — the leak is in the reviewed split file itself, which
    # is the case a witness cannot catch and this check must.
    with pytest.raises(SystemExit, match="records training on holdout episode"):
        _run_eval(run_dir, holdout, tmp_path / "nope", train_episodes=(*RUNG, "d1-0006"))
    assert not (tmp_path / "nope" / "predictions.jsonl").exists()  # refused before any forward


def test_refuses_when_a_recorded_training_episode_changed_on_disk(
    trained_subset: tuple[Path, Path], tmp_path: Path
) -> None:
    """The id list alone is not the proof. The recorded ids still have to hash to the bytes the
    checkpoint says it trained on, or a checkpoint could simply assert a convenient split."""
    dataset = _copy_dataset(tmp_path / "dataset")
    holdout = _holdout_file(tmp_path, HOLDOUT, "holdout.txt")
    run_dir = trained_subset[0]

    # Unmodified copy: identical manifest bytes -> identical hash -> still provable.
    assert _run_eval(run_dir, holdout, tmp_path / "ok", dataset=dataset, train_episodes=RUNG) == 0

    manifest = dataset / RUNG[0] / "manifest.json"
    manifest.write_text(manifest.read_text().replace('"instruction"', '"Instruction"', 1))

    with pytest.raises(SystemExit, match="REFUSING TO SCORE — split not provable \\(disjoint\\)"):
        _run_eval(run_dir, holdout, tmp_path / "nope", dataset=dataset, train_episodes=RUNG)


def test_refuses_a_holdout_episode_byte_identical_in_content_to_a_trained_one(
    archived: Path, trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """The guarantee the complement rule never had, checked ON the complement path.

    An episode duplicated under a second id defeats the split without changing a single id-based
    hash: the holdout is still 'not in the training set' by name, and the complement hash still
    matches because the holdout's manifest is not part of it. Hashing manifest['checksums'] and
    not the whole manifest is what catches it — the manifest embeds ``episode_id``, so a
    whole-file digest can never match across a rename.
    """
    dataset = _copy_dataset(tmp_path / "dataset")
    holdout_manifest = dataset / HOLDOUT[0] / "manifest.json"
    trained_checksums = json.loads((dataset / "d1-0000" / "manifest.json").read_text())["checksums"]
    record = json.loads(holdout_manifest.read_text())
    record["checksums"] = trained_checksums  # same recording, second id
    holdout_manifest.write_text(json.dumps(record))

    with pytest.raises(SystemExit, match="byte-identical in content to trained ones"):
        _run_eval(archived, trained[1], tmp_path / "nope", dataset=dataset, train_episodes=None)


def test_recorded_training_order_is_used_verbatim_not_re_sorted(
    trained_subset: tuple[Path, Path], tmp_path: Path
) -> None:
    """``dataset_snapshot_hash`` is a SEQUENTIAL digest over the episodes in iteration order, so
    the evaluator has to replay the recorded order rather than sort it. Feeding a reversed list
    whose SET is correct must therefore refuse: if this ever passes, the evaluator has started
    reconstructing an order of its own and the hash has stopped binding ids to bytes.
    """
    run_dir = _copy_run(trained_subset[0], tmp_path / "reordered-run")
    _rewrite_run_metadata(run_dir, train_episode_ids=list(reversed(RUNG)))
    holdout = _holdout_file(tmp_path, HOLDOUT, "holdout.txt")

    # The witness matches as a SET, which is all it is compared on — so the order defect has to
    # be caught by the hash, exactly as it would be for a real committed split file.
    with pytest.raises(SystemExit, match="REFUSING TO SCORE — split not provable \\(disjoint\\)"):
        _run_eval(run_dir, holdout, tmp_path / "nope", train_episodes=RUNG)


def test_refuses_when_the_checkpoint_lists_an_episode_the_dataset_does_not_have(
    trained_subset: tuple[Path, Path], tmp_path: Path
) -> None:
    """--dataset pointing somewhere plausible but wrong has to say so, rather than silently
    scoring against whatever subset happens to be present."""
    run_dir = _copy_run(trained_subset[0], tmp_path / "ghost-run")
    _rewrite_run_metadata(run_dir, train_episode_ids=[*RUNG, "d1-9999"])
    holdout = _holdout_file(tmp_path, HOLDOUT, "holdout.txt")

    with pytest.raises(SystemExit, match="trained on 1 episode\\(s\\) absent from"):
        _run_eval(run_dir, holdout, tmp_path / "nope", train_episodes=(*RUNG, "d1-9999"))


# -- T-29 / I-7: --frame-history ------------------------------------------------------------


def test_frame_history_runs_end_to_end_and_names_itself_in_the_report(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """The T-29 A/B has to be readable months later. Two predictions.jsonl from ONE checkpoint
    differ only in what the policy was shown, so the report must say which — otherwise the
    comparison silently becomes checkpoint-vs-checkpoint."""
    from wam.evaluation import BenchReport

    run_dir, holdout = trained
    out = tmp_path / "with-history"

    assert _run_eval(run_dir, holdout, out, "--frame-history") == 0

    bench = BenchReport.from_json((out / "bench.json").read_text())
    assert bench.run_name.endswith("+frame_history")
    assert bench.num_predictions > 0


def test_default_stays_the_historical_tiled_path(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """Off by default, so re-scoring an archived run reproduces it instead of quietly
    redefining what every number before 2026-07-30 meant."""
    from wam.evaluation import BenchReport

    run_dir, holdout = trained
    out = tmp_path / "default"

    assert _run_eval(run_dir, holdout, out) == 0
    assert "+frame_history" not in BenchReport.from_json((out / "bench.json").read_text()).run_name


# -- T-30 / I-3: --flow-sampler -------------------------------------------------------------


def test_flow_sampler_runs_end_to_end_and_names_itself_in_the_report(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """Two predictions.jsonl from ONE checkpoint can now differ in the readout as well as in the
    frames, so the report has to carry both or the T-30 A/B is unreadable after the fact."""
    from wam.evaluation import BenchReport

    run_dir, holdout = trained
    out = tmp_path / "flow"

    assert _run_eval(run_dir, holdout, out, "--flow-sampler", "--flow-steps", "8") == 0

    bench = BenchReport.from_json((out / "bench.json").read_text())
    assert bench.run_name.endswith("+flow8s0")
    assert bench.num_predictions > 0


def test_the_default_stays_the_regression_head(trained: tuple[Path, Path], tmp_path: Path) -> None:
    """OFF by default: re-scoring an archived run reproduces it rather than silently swapping in
    a different decoder and calling the result the same run."""
    from wam.evaluation import BenchReport

    run_dir, holdout = trained
    out = tmp_path / "regression"

    assert _run_eval(run_dir, holdout, out) == 0
    assert "+flow" not in BenchReport.from_json((out / "bench.json").read_text()).run_name


def test_flow_steps_without_the_sampler_flag_is_refused(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """A half-typed command must not score the regression head into a flow-named --out dir —
    that produces an artifact whose filename and contents disagree, months before anyone looks."""
    run_dir, holdout = trained

    for flag, value in (
        ("--flow-steps", "16"),
        ("--flow-seed", "1"),
        ("--flow-mean-k", "8"),
        ("--flow-t0", "0.6"),
    ):
        with pytest.raises(SystemExit, match="--flow-steps"):
            _run_eval(run_dir, holdout, tmp_path / "nope", flag, value)


def test_the_flow_flag_actually_changes_the_predictions(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """The flag has to reach the model, not just the run name. A wiring break here would make the
    A/B report 'no difference' — the most expensive possible false negative."""
    run_dir, holdout = trained
    regression, flow = tmp_path / "a", tmp_path / "b"

    assert _run_eval(run_dir, holdout, regression) == 0
    assert _run_eval(run_dir, holdout, flow, "--flow-sampler", "--flow-steps", "4") == 0

    assert _first_chunk(regression) != _first_chunk(flow)


def test_timing_json_records_which_readout_it_timed(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """Wall time is the only number here that cannot be recomputed from the archived artifacts,
    and the sbatch A/B reads it to check the sampler against the executor's 500 ms deadline."""
    run_dir, holdout = trained
    out = tmp_path / "timed"

    assert _run_eval(run_dir, holdout, out, "--flow-sampler", "--flow-steps", "8") == 0

    timing = json.loads((out / "timing.json").read_text())
    assert timing["flow_steps"] == 8
    assert timing["flow_seed"] == 0
    assert timing["num_chunks"] > 0
    assert timing["ms_per_chunk"] > 0.0


# -- T-30 control arms: --flow-mean-k and --flow-t0 ------------------------------------------


def test_the_mean_of_k_arm_names_itself_and_reaches_the_sampler(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """The MSE-fair arm of the T-30 rule. It has to be distinguishable from the single-draw arm
    in the archive (or the two get compared as if they were one readout twice) AND it has to
    actually average — a flag that only reached the run name would report the single-draw
    penalty under the mean's name, which is the branch the rule keys on."""
    from wam.evaluation import BenchReport

    run_dir, holdout = trained
    single, averaged = tmp_path / "k1", tmp_path / "k4"

    assert _run_eval(run_dir, holdout, single, "--flow-sampler", "--flow-steps", "8") == 0
    assert (
        _run_eval(
            run_dir, holdout, averaged, "--flow-sampler", "--flow-steps", "8", "--flow-mean-k", "4"
        )
        == 0
    )

    assert BenchReport.from_json((averaged / "bench.json").read_text()).run_name.endswith(
        "+flow8s0k4"
    )
    assert _first_chunk(single) != _first_chunk(averaged)


def test_the_warm_start_arm_names_itself_and_reaches_the_sampler(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """The conditioning-mismatch control. Same two requirements, and the tag has to carry the t0
    value: a warm start at 0.6 and one at 0.9 are different measurements of the same question."""
    from wam.evaluation import BenchReport

    run_dir, holdout = trained
    plain, warm = tmp_path / "t0", tmp_path / "t06"

    assert _run_eval(run_dir, holdout, plain, "--flow-sampler", "--flow-steps", "8") == 0
    assert (
        _run_eval(run_dir, holdout, warm, "--flow-sampler", "--flow-steps", "8", "--flow-t0", "0.6")
        == 0
    )

    assert BenchReport.from_json((warm / "bench.json").read_text()).run_name.endswith(
        "+flow8s0t0.6"
    )
    assert _first_chunk(plain) != _first_chunk(warm)


def test_timing_json_records_the_control_arms_and_the_tag(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """``timing.json`` is what the --out guard reads back, and the sbatch verdict reads its
    ms/chunk against the executor's deadline — for THIS arm, not for whichever one ran last."""
    run_dir, holdout = trained
    out = tmp_path / "armed"

    arms = ("--flow-sampler", "--flow-steps", "8", "--flow-mean-k", "2", "--flow-t0", "0.5")
    assert _run_eval(run_dir, holdout, out, *arms) == 0

    timing = json.loads((out / "timing.json").read_text())
    assert timing["flow_mean_k"] == 2
    assert timing["flow_t0"] == 0.5
    assert timing["readout_tag"] == "+flow8s0k2t0.5"
    assert timing["run_name"].endswith(timing["readout_tag"])


def test_the_control_arms_are_refused_before_the_checkpoint_is_loaded(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """Out-of-range values must not be found by the first predict(): for a Wan checkpoint that is
    minutes of base-weight loading later, and the run has already booked the GPU."""
    run_dir, holdout = trained

    with pytest.raises(SystemExit, match="--flow-steps must be >= 1"):
        _run_eval(run_dir, holdout, tmp_path / "nope", "--flow-sampler", "--flow-steps", "0")
    with pytest.raises(SystemExit, match="--flow-mean-k must be >= 1"):
        _run_eval(run_dir, holdout, tmp_path / "nope", "--flow-sampler", "--flow-mean-k", "0")
    with pytest.raises(SystemExit, match=r"--flow-t0 must be in \[0, 1\)"):
        _run_eval(run_dir, holdout, tmp_path / "nope", "--flow-sampler", "--flow-t0", "1.0")
    assert not (tmp_path / "nope").exists()


# -- one --out per readout -------------------------------------------------------------------


def test_a_second_readout_refuses_to_overwrite_the_first_ones_artifacts(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """The A/B's own failure mode: every artifact name is fixed and --out defaults to --run-dir,
    so scoring B into A's directory replaced A with B and left an A/B of B against itself, with
    nothing on disk to show it had happened. Two help strings asked for a separate --out; that
    was the whole enforcement."""
    run_dir, holdout = trained
    out = tmp_path / "one-dir"

    assert _run_eval(run_dir, holdout, out) == 0
    first = (out / "bench.json").read_text()

    with pytest.raises(SystemExit, match="already holds artifacts from a different readout"):
        _run_eval(run_dir, holdout, out, "--flow-sampler", "--flow-steps", "8")
    with pytest.raises(SystemExit, match="already holds artifacts from a different readout"):
        _run_eval(run_dir, holdout, out, "--frame-history")

    assert (out / "bench.json").read_text() == first  # refused before anything was rewritten


def test_re_running_the_same_readout_into_the_same_out_stays_allowed(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """The guard must not break a re-score after an interrupted pass: the same arm reproduces the
    same four files, and refusing that would push people toward --skip-split-check-shaped
    workarounds."""
    run_dir, holdout = trained
    out = tmp_path / "same-arm"

    assert _run_eval(run_dir, holdout, out, "--flow-sampler", "--flow-steps", "8") == 0
    assert _run_eval(run_dir, holdout, out, "--flow-sampler", "--flow-steps", "8") == 0

    assert json.loads((out / "timing.json").read_text())["readout_tag"] == "+flow8s0"


def test_the_out_guard_identifies_an_arm_that_predates_timing_json(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """Archived runs are the ones this protects: they were written before ``timing.json`` existed,
    so the guard has to fall back to ``bench.json``'s run_name — which has carried the readout
    suffix since the day there was more than one readout."""
    run_dir, holdout = trained
    out = tmp_path / "archived-arm"

    assert _run_eval(run_dir, holdout, out, "--frame-history") == 0
    (out / "timing.json").unlink()

    with pytest.raises(SystemExit, match="already holds artifacts from a different readout"):
        _run_eval(run_dir, holdout, out)
    assert _run_eval(run_dir, holdout, out, "--frame-history") == 0


def test_the_shared_loader_carries_the_flow_readout_into_the_policy(
    trained: tuple[Path, Path],
) -> None:
    """``load_joint_policy`` is the one entry point eval_t16, rollout.py and serve_policy.py all
    load through, so wiring it there is what lets the closed loop inherit this without a second
    implementation. Pinned here because nothing else exercises the keyword end to end."""
    from wam.runtime.policies import load_joint_policy

    run_dir, _ = trained
    model_path = ev.resolve_checkpoint(run_dir, "latest")

    regression = load_joint_policy(model_path, device="cpu")
    sampled = load_joint_policy(model_path, device="cpu", flow_steps=4, flow_seed=2)

    assert regression.flow_steps is None
    assert sampled.flow_steps == 4 and sampled.flow_seed == 2

    with pytest.raises(ValueError, match="flow_steps"):
        load_joint_policy(model_path, device="cpu", flow_steps=0)
