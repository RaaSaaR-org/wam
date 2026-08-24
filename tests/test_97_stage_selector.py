"""Tests for `T40_RULE_V11`'s stage selector, inside `97_transfer25_restyle.sbatch`.

WHY THIS FILE EXISTS AT ALL. The selector is python embedded in an sbatch, and until it was written
nothing in `tests/` EXECUTED any of that python — the existing suites read the file as text and
assert on properties of the source. That is enough for "the contract names four variables" and not
nearly enough for a slicing rule, because the failure mode here is silent: a mis-staged run produces
a plausible work list, spends GPU-hours, and yields a corpus that is not the registered experiment.

`T40_RULE_V11` §2.4 says it in the document's own words — a stage that generated four train styles
against ten identity repeats "would pass every check in the file". The two guards below are the
enforcement behind §0's promise that arms B and C stay matched *at every stage*, so a test that
only checked the happy path would leave exactly the property that was missing still unverified.
Each guard therefore gets a partition built to break it.

The heredoc is extracted verbatim and run as a subprocess, the same way
`test_restyle_transfer25._harvest_source` runs 97's harvest. Re-typing the logic here would test a
copy of the selector rather than the selector.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

SBATCH_97 = pathlib.Path(__file__).resolve().parents[1] / "cluster/discoverer/97_transfer25_restyle.sbatch"
REAL_RENDERING = (pathlib.Path(__file__).resolve().parents[1]
                  / "configs/transfer25/pr08_style_partition.json")

#: `T40_RULE_V11` §2's determination names these four, in this order. Hard-coded here on purpose:
#: the test's job is to catch the selector drifting away from the rule, and a expectation derived
#: from the same file the selector reads would drift along with it.
STAGE1_STYLES = ["train-01-oak-tungsten", "train-02-linen-overcast",
                 "train-03-melamine-fluorescent", "train-04-slate-lowkey"]


def _expansion_source() -> str:
    """97's work-list expansion heredoc, verbatim. Refuses rather than guessing."""
    text = SBATCH_97.read_text(encoding="utf-8")
    start = text.index('python - "${SOURCE}/manifest.json"')
    body = text[start:]
    opener = body.index("<<'PY'\n") + len("<<'PY'\n")
    end = body.index("\nPY\n", opener)
    source = body[opener:end]
    assert "repeat_span" in source, "97's expansion no longer carries a stage selector at all"
    assert "ARM C IS NOT FRAME-MATCHED AT STAGE" in source, (
        "97's expansion no longer carries the stage-level arm-C guard T40_RULE_V11 §2.4 requires")
    return source


def _manifest(tmp_path: pathlib.Path, n_episodes: int, frames: int = 100) -> pathlib.Path:
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({
        "resolution": [640, 480],
        "episodes": [{"id": f"episode_{i:06d}", "frames": frames} for i in range(n_episodes)],
    }), encoding="utf-8")
    return p


def _run(tmp_path: pathlib.Path, *, stage: str, style_set: str,
         rendering: pathlib.Path, n_episodes: int = 402,
         idx: int = 1, total: int = 1, k1: int = 4) -> tuple[int, str, dict | None, list[dict]]:
    """Run the expansion. Returns (returncode, stderr+stdout, partition_facts, work rows)."""
    script = tmp_path / "expand_from_97.py"
    script.write_text(_expansion_source(), encoding="utf-8")
    out = tmp_path / f"work-{stage}-{style_set}.jsonl"
    facts = tmp_path / f"facts-{stage}-{style_set}.json"
    proc = subprocess.run(
        [sys.executable, str(script), str(_manifest(tmp_path, n_episodes)), str(rendering),
         style_set, str(idx), str(total), str(out), "0", str(facts), stage, str(k1)],
        capture_output=True, text=True)
    payload = json.loads(facts.read_text()) if facts.is_file() else None
    rows = ([json.loads(line) for line in out.read_text().splitlines() if line]
            if out.is_file() else [])
    return proc.returncode, proc.stdout + proc.stderr, payload, rows


# -- the committed partition, staged --------------------------------------------------------------

def test_stage_1_is_the_four_styles_the_rule_names_in_the_order_it_names_them(tmp_path):
    """A prefix of the committed order, never a selection.

    If this ever became a filter or a sort, the stage could be chosen after seeing which styles
    rendered well — which is the failure the committed partition exists to prevent, arriving by a
    route the partition hash cannot see.
    """
    rc, log, facts, rows = _run(tmp_path, stage="1", style_set="train", rendering=REAL_RENDERING)
    assert rc == 0, log
    assert facts["stage_styles"] == STAGE1_STYLES
    assert facts["stage_instances_per_set"]["train"] == 4
    assert sorted({r["style"] for r in rows}) == sorted(STAGE1_STYLES)
    assert all(r["stage"] == "1" for r in rows), "T40_RULE_V11 §2.4 requires the stage per clip"


def test_identity_is_cut_on_the_repeat_axis_because_it_is_one_style(tmp_path):
    """The asymmetry §2.4 flags as easy to get wrong.

    TRAIN is ten styles at repeats=1 so its stage is a slice of the STYLE LIST; IDENTITY is one
    style at repeats=10 so its stage is a slice of the REPEAT RANGE. Slicing identity's style list
    would take all ten repeats or none, and arm C would silently be 2.5x arm B.
    """
    rc, log, facts, rows = _run(tmp_path, stage="1", style_set="identity", rendering=REAL_RENDERING)
    assert rc == 0, log
    assert facts["stage_instances_per_set"]["identity"] == 4
    assert sorted({r["repeat"] for r in rows}) == [0, 1, 2, 3]
    assert len({r["style"] for r in rows}) == 1, "identity is one style; the cut is on repeats"

    rc, log, facts, rows = _run(tmp_path, stage="2", style_set="identity", rendering=REAL_RENDERING)
    assert rc == 0, log
    assert sorted({r["repeat"] for r in rows}) == [4, 5, 6, 7, 8, 9]


def test_the_two_stages_partition_the_committed_partition_exactly(tmp_path):
    """No clip generated twice, none left behind — the property that makes staging safe.

    Checked on the UNITS rather than on the counts, because two stages can agree on totals while
    overlapping: a slice bug that took ``[:4]`` and ``[3:]`` differs from the truth by one style
    and by nothing else visible in an instance count.
    """
    def units(stage, style_set):
        rc, log, _, rows = _run(tmp_path, stage=stage, style_set=style_set,
                                rendering=REAL_RENDERING)
        assert rc == 0, log
        return {(r["unit"], r["seed"]) for r in rows}

    for style_set in ("train", "identity"):
        one, two = units("1", style_set), units("2", style_set)
        every = units("all", style_set)
        assert one & two == set(), f"{style_set}: stages overlap"
        assert one | two == every, f"{style_set}: the stages do not cover the committed partition"

    # eval is DEFERRED out of stage 1, not split, so it is whole in stage 2.
    assert units("2", "eval") == units("all", "eval")


def test_stage_1_prices_the_eval_set_at_zero_and_does_not_pretend_it_is_gone(tmp_path):
    """Deferred is not cut. The ceiling gate reads these numbers, so a stage that dropped eval from
    the record entirely would price stage 2 against a partition it could no longer describe."""
    _, _, facts, _ = _run(tmp_path, stage="1", style_set="train", rendering=REAL_RENDERING)
    assert facts["stage_instances_per_set"]["eval"] == 0
    # ...while the WHOLE-partition figures are untouched: the ceiling is a bound on the experiment
    # and does not move because one invocation generates less of it.
    assert facts["style_instances_per_set"] == {"train": 10, "eval": 5, "identity": 10}
    assert facts["style_instances_whole_partition"] == 25
    assert facts["whole_partition_clips"] == 10050


def test_the_committed_stage_1_arithmetic_is_the_one_the_rule_registered(tmp_path):
    """8 of 25 style-instances and 3 216 clips — V11 §2's numbers, recomputed from the rendering."""
    _, _, facts, _ = _run(tmp_path, stage="1", style_set="train", rendering=REAL_RENDERING)
    staged = facts["stage_instances_per_set"]
    assert sum(staged.values()) == 8
    assert facts["stage_clips"] == 402 * 4
    assert sum(staged.values()) * 402 == 3216


# -- the guards, each against a partition built to break it ----------------------------------------

def _rendering(tmp_path: pathlib.Path, *, train_repeats: list[int], identity_repeats: int,
               name: str, identity_seeds: list[int] | None = None) -> pathlib.Path:
    """A rendering shaped like the committed one but with the repeat structure under test.

    Every field the expansion checks BEFORE the stage guards is made self-consistent on purpose: a
    fixture that tripped the volume check or the whole-file seed check would exit with a different
    message and the test would pass while proving nothing about the guard it names.
    """
    seeds = iter(range(8001, 9000))
    train = [{"id": f"train-{i:02d}", "prompt": "p", "repeats": r,
              "seeds": [next(seeds) for _ in range(r)]}
             for i, r in enumerate(train_repeats, start=1)]
    train_seed_set = sorted({s for st in train for s in st["seeds"]})
    identity = [{"id": "identity-source", "prompt": "p", "repeats": identity_repeats,
                 "seeds": identity_seeds if identity_seeds is not None else list(train_seed_set)}]
    ev = [{"id": f"eval-{i:02d}", "prompt": "p", "repeats": 1, "seeds": [9500 + i]}
          for i in range(1, 6)]
    inst = sum(train_repeats) + len(ev) + identity_repeats
    payload = {
        "schema": "wam.style_partition/1", "rule": "T40_STYLES_V1", "blocking_todos": [],
        "seed_schedule": {"blocks": {"train": [8001, 8999]}, "assignment": "a", "rule": "r"},
        "train": train, "eval": ev, "identity": identity,
        "volume": {"episodes": 402, "style_instances": inst,
                   "whole_partition_clips": 402 * inst},
    }
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_the_stage_level_arm_c_guard_fires_where_the_whole_partition_one_cannot(tmp_path):
    """The exact hole T40_RULE_V11 §2.4 names, reproduced.

    Non-uniform train repeats make the WHOLE-partition counts agree (14 == 14) while stage 1's
    disagree (8 vs 4). The pre-existing guard is computed over the whole rendering and passes; if
    the stage-level guard were missing, this partition would generate an arm C at half arm B's
    volume and B − C would measure volume while being read as diversity.
    """
    rendering = _rendering(tmp_path, train_repeats=[2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
                           identity_repeats=14, name="nonuniform")

    # The whole-partition guard is satisfied by this fixture — proved, not assumed, by running the
    # unstaged path over the same file and finding no refusal.
    rc, log, facts, _ = _run(tmp_path, stage="all", style_set="train", rendering=rendering)
    assert rc == 0, log
    assert facts["style_instances_per_set"]["train"] == facts["style_instances_per_set"]["identity"]

    rc, log, _, _ = _run(tmp_path, stage="1", style_set="train", rendering=rendering)
    assert rc != 0
    assert "ARM C IS NOT FRAME-MATCHED AT STAGE 1" in log
    assert "arm B 8 clips per episode and arm C 4" in log


def test_the_stage_level_seed_guard_fires_when_the_counts_agree_but_the_seeds_do_not(tmp_path):
    """Equal clip counts drawn from different seed blocks is still a confound.

    Arm C isolates the generator's fingerprint and can only do that seed for seed, so a count
    comparison is not sufficient — and it is precisely the case a count comparison cannot see.
    """
    rendering = _rendering(tmp_path, train_repeats=[1] * 10, identity_repeats=10,
                           name="shuffled",
                           # Same ten seeds, so the whole-file set comparison still passes; the
                           # ORDER differs, so the stage-1 prefixes span different blocks.
                           identity_seeds=list(reversed(range(8001, 8011))))
    rc, log, _, _ = _run(tmp_path, stage="all", style_set="train", rendering=rendering)
    assert rc == 0, log

    rc, log, _, _ = _run(tmp_path, stage="1", style_set="train", rendering=rendering)
    assert rc != 0
    assert "DO NOT SPAN THE SAME SEED SET AT STAGE 1" in log


def test_a_partition_too_small_to_stage_is_refused_rather_than_silently_emptied(tmp_path):
    """identity repeats <= k would make stage 2 empty for it, and an empty arm C is not a control."""
    # Four train styles at repeats=1 so the seed sets still match at four instances a side; the
    # fixture has to clear every earlier check or this test proves nothing about the one it names.
    rendering = _rendering(tmp_path, train_repeats=[1] * 4, identity_repeats=4, name="tiny")
    rc, log, _, _ = _run(tmp_path, stage="2", style_set="identity", rendering=rendering)
    assert rc != 0
    assert "repeats=4" in log and "stage 2 would be empty" in log


@pytest.mark.parametrize("style_set", ["train", "identity"])
def test_the_work_sha_material_separates_the_stages(tmp_path, style_set):
    """Two stages must not be able to resume into one another's chunk directory.

    The stage is hashed into WORK_SHA as well as carried in CHUNK_TAG, so the stamp check refuses a
    cross-stage resume on its own terms rather than only because the unit sets happen to differ.
    """
    shas = set()
    for stage in ("1", "2", "all"):
        rc, log, _, _ = _run(tmp_path, stage=stage, style_set=style_set, rendering=REAL_RENDERING)
        assert rc == 0, log
        shas.add((tmp_path / f"work-{stage}-{style_set}.jsonl.sha256").read_text().strip())
    assert len(shas) == 3, "two stages hash to the same work list"
