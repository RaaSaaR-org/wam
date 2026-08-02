"""Tests for the committed splits: the I-8 rungs (T-32) and the PR-03 widened holdout.

These are DATA tests: they assert properties of files under ``configs/splits/`` that an
experiment's interpretation depends on. If the rungs' nesting breaks, the data-scaling curve
stops measuring size and starts measuring "which episodes happened to be drawn"; if a rung and
the holdout overlap, every number the curve produces is a training score. If the PR-03 holdout
stops being a superset of the T-18 one, its archive gate stops comparing like with like and an
already-scored episode can drift back into training.

The two experiments vary opposite things — the rungs vary the training set against a fixed
holdout, PR-03 varies the holdout — so their splits are NOT interchangeable, and the overlap
between them is pinned below rather than left to be discovered by a run.

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
_DATASET_GRIP = _REPO_ROOT / "datasets" / "gr00t-apple-grip"
_SPLITS = _REPO_ROOT / "configs" / "splits"
_HOLDOUT = _SPLITS / "t18_holdout_episodes.txt"
_PR03_HOLDOUT = _SPLITS / "pr03_holdout_150.txt"
_GENERATOR = _REPO_ROOT / "scripts" / "make_rung_splits.py"
_WIDENER = _REPO_ROOT / "scripts" / "widen_holdout.py"
_PREREGISTRATION = _REPO_ROOT / "cluster" / "discoverer" / "62_eval_i8_curve.sbatch"
_PR03 = _REPO_ROOT / "docs" / "preregistration" / "PR-03-grasp-anticipation.md"

#: PR-03 widens the T-18 holdout to this many episodes, leaving this many to train on.
PR03_HOLDOUT_SIZE = 150
PR03_TRAIN_POOL = 252

#: Overlap between each committed I-8 rung and the PR-03 holdout, as PR-03 and
#: ``scripts/widen_holdout.py``'s warning both quote it. Not a defect — the rungs were generated
#: against the 40-episode holdout — but it is the reason the two experiments may not share splits.
PR03_RUNG_OVERLAP = {40: 40, 120: 110, 362: 110}

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
requires_grip_dataset = pytest.mark.skipif(
    not _DATASET_GRIP.is_dir(), reason="datasets/gr00t-apple-grip not present"
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


# ------------------------------------------- the PR-03 widened holdout and widen_holdout.py
#
# Same contract as above, one axis different: the rungs vary the TRAINING set against a fixed
# holdout, PR-03 varies the HOLDOUT because its metric is bottlenecked by how many grasp flips
# are scored, not by how many are trained on (docs/preregistration/PR-03-grasp-anticipation.md).


def test_pr03_holdout_contains_every_t18_holdout_episode() -> None:
    """Nesting, and it buys two things PR-03 depends on.

    Every number ever measured on the 40 stays a sub-metric of the 150 — which is what makes
    PR-03's archive gate ("restrict to the original 40 and reproduce PR-01-GRIPPER's table")
    meaningful rather than a comparison of two different episode sets. And widening can only
    ever move episodes OUT of training: if the new holdout could drop one of the 40, an episode
    that had already been scored would silently become trainable.
    """
    base, wide = set(_ids(_HOLDOUT)), set(_ids(_PR03_HOLDOUT))
    assert base < wide, "the PR-03 holdout is not a strict superset of the T-18 holdout"


def test_pr03_holdout_has_the_declared_episode_count() -> None:
    """150 episodes, none named twice — a line count, checkable without the dataset."""
    ids = _ids(_PR03_HOLDOUT)
    assert len(ids) == PR03_HOLDOUT_SIZE
    assert len(set(ids)) == len(ids), "the PR-03 holdout names an episode twice"


def test_the_i8_rungs_overlap_the_pr03_holdout_by_the_documented_amount() -> None:
    """The two experiments may NOT share splits, and the overlap is pinned so that stays visible.

    This is not a defect to fix: the rungs were generated against the 40-episode holdout, and
    they are correct against it. It is a landmine — running T-32's rungs against PR-03's holdout
    trains on scored episodes — so the exact overlap is asserted here and quoted in PR-03. If a
    future regeneration changes it, the number in the document has to change with it.
    """
    wide = set(_ids(_PR03_HOLDOUT))
    for size, expected in PR03_RUNG_OVERLAP.items():
        assert len(set(_ids(_rung_path(size))) & wide) == expected, f"rung {size} overlap"


def test_pr03_quotes_the_holdout_size_and_training_pool_it_was_generated_with() -> None:
    """The pre-registration's numbers and the committed file must describe the same split.

    PR-03 fixes its decision rule against "150 episodes / 252 to train on". A file that no longer
    matches would leave a binding document arguing about a split that does not exist.
    """
    text = _PR03.read_text(encoding="utf-8")
    assert f"{PR03_HOLDOUT_SIZE} episodes" in text
    assert f"{PR03_TRAIN_POOL} episodes" in text
    header = _PR03_HOLDOUT.read_text(encoding="utf-8")
    assert f"training pool: {PR03_TRAIN_POOL} episodes" in header


def test_widen_holdout_refuses_to_shrink_a_holdout() -> None:
    """Shrinking would move an already-SCORED episode into a training set, silently."""
    wh = _load("widen_holdout")
    with pytest.raises(SystemExit, match="smaller than"):
        wh.widened_ids(["c", "d"], {"a", "b", "e"}, 2, 0)


def test_widen_holdout_refuses_a_size_larger_than_the_dataset() -> None:
    """Asking for more episodes than exist is a typo, not a wider holdout."""
    wh = _load("widen_holdout")
    with pytest.raises(SystemExit, match="only"):
        wh.widened_ids(["c", "d"], {"a", "b"}, 9, 0)


def test_widen_holdout_keeps_the_base_and_adds_the_difference() -> None:
    """The arithmetic the whole artifact rests on, on a fixture small enough to read."""
    wh = _load("widen_holdout")
    out = wh.widened_ids(["c", "d", "e", "f"], {"a", "b"}, 5, 0)
    assert len(out) == 5
    assert {"a", "b"} <= set(out)
    assert out == sorted(out)


# ------------------------------------------------------------ needs the recorded parquet


@requires_grip_dataset
def test_pr03_holdout_names_only_episodes_present_in_the_grip_dataset() -> None:
    """A holdout id the dataset does not contain scores fewer episodes than the file claims."""
    from wam.data.episode import list_episodes

    present = {p.name for p in list_episodes(_DATASET_GRIP)}
    assert set(_ids(_PR03_HOLDOUT)) <= present


@requires_grip_dataset
def test_pr03_holdout_leaves_the_declared_training_pool() -> None:
    """252 episodes left to refit on — the number PR-03's decision rule is written against."""
    from wam.data.episode import list_episodes

    wide = set(_ids(_PR03_HOLDOUT))
    pool = [p.name for p in list_episodes(_DATASET_GRIP) if p.name not in wide]
    assert len(pool) == PR03_TRAIN_POOL


@requires_grip_dataset
def test_pr03_holdout_additions_are_not_a_sorted_prefix() -> None:
    """The permutation has to be real, for the same reason the rungs' does.

    A sorted prefix would draw the 110 additions from one contiguous recording session, so the
    widened holdout would differ from the base one in WHEN it was recorded as well as in size —
    and the archive gate could not tell a channel change from a session change.
    """
    from wam.data.episode import list_episodes

    base = set(_ids(_HOLDOUT))
    pool = [p.name for p in list_episodes(_DATASET_GRIP) if p.name not in base]
    added = sorted(set(_ids(_PR03_HOLDOUT)) - base)
    assert added != pool[:110]


def _widen(out: Path) -> str:
    result = subprocess.run(
        [
            sys.executable,
            str(_WIDENER),
            "--dataset", str(_DATASET_GRIP),
            "--base", str(_HOLDOUT),
            "--size", str(PR03_HOLDOUT_SIZE),
            "--seed", "0",
            "--out", str(out),
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        check=False,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    return out.read_text(encoding="utf-8")


@requires_grip_dataset
def test_widen_holdout_is_deterministic_for_a_seed(tmp_path: Path) -> None:
    """Same dataset, base, seed and --out -> byte-identical output, header included."""
    out = tmp_path / "pr03_holdout_150.txt"
    assert _widen(out) == _widen(out)


@requires_grip_dataset
def test_widen_holdout_reproduces_the_committed_file(tmp_path: Path) -> None:
    """The committed artifact is re-derivable from the code next to it, so nobody has to trust it.

    Everything but one line is compared literally. The exception is the ``--out`` line of the
    reproduce command, which quotes the path the file was actually written to — that is the
    point of recording it (a header naming a path the file is not at would be worse than no
    header) and it is why the generated text is patched back to the committed path here rather
    than the header being made to lie about a tmp directory.
    """
    out = tmp_path / "pr03_holdout_150.txt"
    generated = _widen(out).replace(f"--out {out}", "--out configs/splits/pr03_holdout_150.txt")
    assert generated == _PR03_HOLDOUT.read_text(encoding="utf-8")


@requires_grip_dataset
def test_widen_holdout_refuses_a_base_that_is_not_in_the_dataset(tmp_path: Path) -> None:
    """A base the dataset does not contain would widen a split the evaluator never checks."""
    wh = _load("widen_holdout")
    ghost = tmp_path / "ghost.txt"
    ghost.write_text("gr00t-apple-999999\n")
    with pytest.raises(SystemExit, match="absent from"):
        wh.main(
            [
                "--dataset", str(_DATASET_GRIP),
                "--base", str(ghost),
                "--size", "2",
                "--out", str(tmp_path / "out.txt"),
            ]
        )  # fmt: skip
