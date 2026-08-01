"""Tests for the committed I-8 rung splits and ``scripts/make_rung_splits.py`` (T-32).

These are DATA tests: they assert properties of files under ``configs/splits/`` that the
data-scaling curve's interpretation depends on. If the nesting breaks, the curve stops
measuring size and starts measuring "which episodes happened to be drawn"; if a rung and the
holdout overlap, every number the curve produces is a training score.

SKIPPING IS PER-TEST, NEVER MODULE-LEVEL. ``datasets/gr00t-apple-full`` is git-ignored
(``.gitignore`` un-ignores only ``mock-d0/`` and ``mock-d1/``), so a module-level skipif would
make this entire file a no-op on every machine except the one that recorded the dataset — CI,
a fresh clone, a reviewer. Most of what is checked here is pure arithmetic over committed text
files and needs no dataset at all: nesting, rung/holdout disjointness, episode counts, and the
generator's refusal paths. Those run everywhere. Only the assertions that must open the
parquet or enumerate the real episode directories carry ``requires_dataset``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASET = _REPO_ROOT / "datasets" / "gr00t-apple-full"
_SPLITS = _REPO_ROOT / "configs" / "splits"
_HOLDOUT = _SPLITS / "t18_holdout_episodes.txt"
_GENERATOR = _REPO_ROOT / "scripts" / "make_rung_splits.py"
_PREREGISTRATION = _REPO_ROOT / "cluster" / "discoverer" / "62_eval_i8_curve.sbatch"

#: The chunk length ``configs/training/joint_wan_gr00t.yaml`` trains at (``head.num_steps``).
#: ``EpisodeDataset`` drops any recorded chunk shorter than this, so it is what turns an
#: episode count into a sample count.
CHUNK_STEPS = 16

#: Rung size -> (episode count, chunk samples). The sample counts are what the stage-2
#: equal-EPOCH step budgets (2 290 / 6 748 for 16.88 epochs) are derived from, so they are
#: pinned here: a silent dataset change has to invalidate that derivation loudly rather than
#: leave two rungs quietly training for the wrong number of passes.
EXPECTED = {40: (40, 1085), 120: (120, 3197), 362: (362, 9476)}

#: ``configs/training/joint_wan_gr00t.yaml``: steps 20000, batch_size 8. The stage-1 budget
#: every rung trains for (``55_train_i8_rung.sbatch``, ``STEPS`` default).
STAGE1_STEPS = 20000
STAGE1_BATCH = 8

#: The stage-2 equal-EPOCH control budgets, quoted as literals by ``55_train_i8_rung.sbatch``
#: and by the I-8 pre-registration. Both verdict A and verdict C are held PROVISIONAL until
#: this control reports, so a budget that no longer equalises epochs would silently turn the
#: control into a second uncontrolled run.
STAGE2_STEPS = {40: 2290, 120: 6748}

#: The per-rung epoch counts the pre-registration prints next to the action_reg diagnostic.
STAGE1_EPOCHS = {40: 147.5, 120: 50.0, 362: 16.9}

requires_dataset = pytest.mark.skipif(
    not _DATASET.is_dir(), reason="datasets/gr00t-apple-full not present"
)


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ids(path: Path) -> list[str]:
    from wam.evaluation import load_episode_ids

    return sorted(load_episode_ids(path))


def _rung_path(size: int) -> Path:
    return _SPLITS / f"i8_train_{size:03d}.txt"


def _samples(episode_ids: list[str]) -> int:
    """Chunks of at least CHUNK_STEPS steps — what ``EpisodeDataset`` will actually yield.

    Counted from the actions parquet rather than by building the dataset: this reads a single
    int column per episode instead of decoding 402 videos.
    """
    total = 0
    for episode_id in episode_ids:
        column = pq.read_table(
            _DATASET / episode_id / "actions.parquet", columns=["chunk_idx"]
        ).to_pydict()["chunk_idx"]
        sizes: dict[int, int] = {}
        for value in column:
            sizes[int(value)] = sizes.get(int(value), 0) + 1
        total += sum(1 for rows in sizes.values() if rows >= CHUNK_STEPS)
    return total


# ------------------------------------------------------- committed files, no dataset needed


def test_rung_files_are_nested() -> None:
    """40 ⊂ 120 ⊂ 362, strictly.

    Without this the three rungs are three different random subsets and a difference between
    them confounds "how much data" with "which episodes" — the one thing the whole experiment
    is designed to hold fixed.
    """
    r40, r120, r362 = (set(_ids(_rung_path(n))) for n in (40, 120, 362))
    assert r40 < r120 < r362


def test_rung_files_are_disjoint_from_the_t18_holdout() -> None:
    """A rung naming a scored episode makes every number on the curve a training score.

    Pure set arithmetic over four committed text files: it needs no dataset, and it is the
    single invariant whose failure would invalidate the entire curve, so it must run on every
    machine rather than only on the one holding the parquet.
    """
    holdout = set(_ids(_HOLDOUT))
    assert holdout, "the T-18 holdout file is empty — nothing would be held out"
    for size in EXPECTED:
        ids = set(_ids(_rung_path(size)))
        assert not (ids & holdout), f"rung {size} overlaps the holdout"


def test_rung_files_have_the_declared_episode_counts() -> None:
    """Episode count is a line count, not a dataset property.

    Split out from the sample-count assertion on purpose: a rung file that lost or gained ids
    is detectable from the committed artifact alone, and gating that on a git-ignored dataset
    would mean nobody but the recording machine ever notices.
    """
    for size, (episodes, _samples_expected) in EXPECTED.items():
        ids = _ids(_rung_path(size))
        assert len(ids) == episodes, f"rung {size} episode count"
        assert len(set(ids)) == len(ids), f"rung {size} names an episode twice"


def test_the_stage2_equal_epoch_budgets_follow_from_the_pinned_sample_counts() -> None:
    """steps_rung = STAGE1_STEPS * samples_rung / samples_362 — the batch size cancels.

    The stage-2 control is the evidence both verdict A and verdict C are held PROVISIONAL for
    (``62_eval_i8_curve.sbatch``, I8_RULE_V2). If EXPECTED's sample counts ever move without
    2290/6748 moving with them, that control silently stops equalising epochs and the confound
    it exists to remove is back — with a run that still claims to have removed it.
    """
    for size, steps in STAGE2_STEPS.items():
        derived = round(STAGE1_STEPS * EXPECTED[size][1] / EXPECTED[362][1])
        assert derived == steps, f"rung {size}: equal-epoch budget is {derived}, not {steps}"


def test_the_preregistration_quotes_the_epoch_counts_the_sample_counts_imply() -> None:
    """147.5 / 50.0 / 16.9 epochs at N = 40 / 120 / 362, as printed next to action_reg.

    Those numbers are the whole argument for the epoch confound: they are what makes rung 40's
    training loss comparable to nothing at rung 362. A comment quoting stale epochs next to a
    live gate is how a reader ends up trusting a rule whose premise has moved.
    """
    text = _PREREGISTRATION.read_text(encoding="utf-8")
    for size, epochs in STAGE1_EPOCHS.items():
        derived = STAGE1_STEPS * STAGE1_BATCH / EXPECTED[size][1]
        assert round(derived, 1) == epochs, f"rung {size}: {derived:.1f} epochs, not {epochs}"
    assert "147.5 / 50.0 / 16.9" in text, "the pre-registration no longer quotes these epochs"
    assert "STEPS=2290 / 6748" in text, "the pre-registration no longer quotes stage 2's budgets"


def test_make_rung_splits_refuses_a_rung_larger_than_the_pool() -> None:
    """Asking for more episodes than exist is a typo, not a bigger rung."""
    mk = _load("make_rung_splits")
    with pytest.raises(SystemExit, match="exceeds"):
        mk.rung_ids(["a", "b", "c"], 4, 0)


# ------------------------------------------------------------ needs the recorded parquet


@requires_dataset
def test_rung_files_name_only_episodes_present_in_the_dataset() -> None:
    """A rung id the dataset does not contain trains on fewer episodes than the file claims."""
    from wam.data.episode import list_episodes

    present = {p.name for p in list_episodes(_DATASET)}
    for size in EXPECTED:
        ids = set(_ids(_rung_path(size)))
        assert ids <= present, f"rung {size} names episodes absent from {_DATASET}"


@requires_dataset
def test_rung_files_have_the_declared_sample_counts() -> None:
    """Chunk samples, not episodes — the quantity the step budgets are derived from."""
    for size, (_episodes, samples) in EXPECTED.items():
        assert _samples(_ids(_rung_path(size))) == samples, f"rung {size} sample count"


@requires_dataset
def test_the_largest_rung_is_exactly_the_dataset_minus_the_holdout() -> None:
    """Rung 362 is NOT re-trained — it is ``runs/t16-lora-seed0`` reused. That reuse is only
    legitimate if the committed file describes precisely the set that run trained on."""
    from wam.data.episode import list_episodes

    holdout = set(_ids(_HOLDOUT))
    complement = {p.name for p in list_episodes(_DATASET) if p.name not in holdout}
    assert set(_ids(_rung_path(362))) == complement


@requires_dataset
def test_rung_selection_is_not_a_sorted_prefix_of_the_dataset() -> None:
    """The permutation has to be real. A sorted prefix would take a contiguous block of one
    recording session, so rung 40 would differ from rung 362 in WHEN it was recorded as well as
    in how much of it there is — and the curve could not tell those two apart."""
    from wam.data.episode import list_episodes

    holdout = set(_ids(_HOLDOUT))
    pool = [p.name for p in list_episodes(_DATASET) if p.name not in holdout]
    assert _ids(_rung_path(40)) != pool[:40]
    assert _ids(_rung_path(120)) != pool[:120]


@requires_dataset
def test_make_rung_splits_is_deterministic_for_a_seed(tmp_path: Path) -> None:
    """Re-running the generator reproduces the committed files byte for byte.

    That is what makes the committed artifact reviewable: the selection is pinned by git AND
    re-derivable, so nobody has to trust that the file came from the code next to it.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(_GENERATOR),
            "--dataset", str(_DATASET),
            "--holdout", str(_HOLDOUT),
            "--seed", "0",
            "--rungs", "40,120,362",
            "--out", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        check=False,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    for size in EXPECTED:
        name = f"i8_train_{size:03d}.txt"
        assert (tmp_path / name).read_bytes() == (_SPLITS / name).read_bytes(), name


@requires_dataset
def test_make_rung_splits_refuses_a_holdout_that_is_not_in_the_dataset(tmp_path: Path) -> None:
    """Rungs generated against a holdout the dataset does not contain would be proven against
    a different split than the evaluator later checks."""
    mk = _load("make_rung_splits")
    ghost = tmp_path / "ghost.txt"
    ghost.write_text("gr00t-apple-999999\n")
    with pytest.raises(SystemExit, match="absent from"):
        mk.main(
            [
                "--dataset", str(_DATASET),
                "--holdout", str(ghost),
                "--rungs", "2",
                "--out", str(tmp_path),
            ]
        )  # fmt: skip
