"""Tests for scripts/eval_t16.py — scoring a fine-tune on episodes it never saw.

Everything runs on CPU with the tiny backbone over ``datasets/mock-d1`` and the shipped
``configs/training/joint.yaml``, the same fixtures ``test_training_resume.py`` uses: no Wan
weights, no GPU, no network. The Wan-specific part of the script is one branch
(``adapter_only`` -> build the frozen base), and the tiny path exercises everything else.

The test that matters most here is the refusal. An offline verdict is only worth reading if the
episodes it was computed on were genuinely held out, and "genuinely" cannot be taken on trust from
a filename: the trainer's ``dataset_snapshot_ref`` covers the episodes it actually trained on, so
recomputing it over ``dataset - holdout`` either matches or the split is not what it claims.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_JOINT_YAML = _REPO_ROOT / "configs" / "training" / "joint.yaml"
_MOCK_D1 = _REPO_ROOT / "datasets" / "mock-d1"

HOLDOUT = ("d1-0006", "d1-0007")
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


def _run_eval(run_dir: Path, holdout: Path, out: Path, *extra: str) -> int:
    return ev.main(
        [
            "--run-dir", str(run_dir),
            "--dataset", str(_MOCK_D1),
            "--holdout", str(holdout),
            "--device", "cpu",
            "--out", str(out),
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
    """The guard: a different split does not hash to what the checkpoint trained on."""
    run_dir, _ = trained
    wrong = _holdout_file(tmp_path, ("d1-0000", "d1-0001"), "wrong.txt")

    with pytest.raises(SystemExit, match="REFUSING TO SCORE"):
        _run_eval(run_dir, wrong, tmp_path / "nope")


def test_refuses_a_partially_overlapping_holdout(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """One trained episode smuggled into the holdout is enough to change the hash."""
    run_dir, _ = trained
    leaky = _holdout_file(tmp_path, (*HOLDOUT, "d1-0000"), "leaky.txt")

    with pytest.raises(SystemExit, match="REFUSING TO SCORE"):
        _run_eval(run_dir, leaky, tmp_path / "nope")


def test_skip_split_check_scores_but_marks_the_output(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    """The escape hatch has to leave a trace, or an unproven number looks like a proven one."""
    run_dir, _ = trained
    wrong = _holdout_file(tmp_path, ("d1-0000",), "wrong.txt")
    out = tmp_path / "unproven"

    assert _run_eval(run_dir, wrong, out, "--skip-split-check") == 0

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
    link.unlink()
    try:
        via_scan = ev.resolve_checkpoint(run_dir, "latest")
        assert via_scan.resolve() == (target / "model.safetensors").resolve()
    finally:
        link.symlink_to(target, target_is_directory=True)


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
