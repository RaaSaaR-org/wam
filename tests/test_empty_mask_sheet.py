"""The empty-mask (a)/(b) instrument — `T40_RULE_V15` §§2-3.

Two things here are load-bearing and neither is obvious from reading the scripts.

``stratify`` IS V15 §2's stratum definition. The protocol's five strata are not a convenience of
the sampler; they are named in a registered rule, the allocation is stated against them, and §5's
quantity re-weights by their population sizes. A silent change here produces a run that looks like
the protocol and is not it.

The tiles MUST NOT LEAK. V15 §3 lists what a tile may not carry — the mask, the area fraction, the
stratum, the episode id, the frame index — and every one of those answers the question the tile
asks. The page builder is the only place that enforces it, so it is tested rather than trusted.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name: str):
    """Import a script by path, registered in ``sys.modules`` before execution.

    ``exec_module`` without the registration makes ``dataclasses``/``typing`` resolve the module's
    own globals to ``None`` mid-definition; this project has hit that once already.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sheet = _load("render_empty_mask_sheet")


# --------------------------------------------------------------------------------------------
# stratify — V15 §2
# --------------------------------------------------------------------------------------------

def test_stratify_covers_exactly_the_empty_frames_and_nothing_else():
    fractions = [0.0, 0.0, 0.3, 0.0, 0.4, 0.0, 0.0]
    got = {index for index, _ in sheet.stratify(fractions)}
    assert got == {i for i, v in enumerate(fractions) if v == 0.0}


def test_stratify_assigns_each_empty_frame_exactly_one_stratum():
    fractions = [0.0] * 3 + [0.2] + [0.0] * 4 + [0.2] + [0.0] * 2
    indices = [index for index, _ in sheet.stratify(fractions)]
    assert len(indices) == len(set(indices))


def test_leading_and_trailing_runs_are_named_by_position_not_by_length():
    # A three-frame run at the start is S1 and a three-frame run at the end is S2, even though a
    # three-frame run in the middle would be S4. Position is the whole point of the split.
    fractions = [0.0] * 3 + [0.5] * 4 + [0.0] * 3
    strata = dict(sheet.stratify(fractions))
    assert [strata[i] for i in (0, 1, 2)] == ["S1_lead"] * 3
    assert [strata[i] for i in (7, 8, 9)] == ["S2_trail"] * 3


@pytest.mark.parametrize(
    ("run_length", "expected"),
    [(1, "S3_int_1_2"), (2, "S3_int_1_2"), (3, "S4_int_3_25"),
     (25, "S4_int_3_25"), (26, "S5_int_26plus"), (99, "S5_int_26plus")],
)
def test_interior_run_binning_boundaries(run_length, expected):
    fractions = [0.5] + [0.0] * run_length + [0.5]
    strata = dict(sheet.stratify(fractions))
    assert {strata[i] for i in range(1, run_length + 1)} == {expected}


def test_an_interior_run_is_binned_by_its_own_length_not_the_episode_total():
    # Two interior runs, 1 frame and 30 frames. Binning the second by the sum would call both S5.
    fractions = [0.5] + [0.0] + [0.5] * 3 + [0.0] * 30 + [0.5]
    strata = dict(sheet.stratify(fractions))
    assert strata[1] == "S3_int_1_2"
    assert strata[5] == "S5_int_26plus"


def test_a_wholly_empty_episode_is_all_leading_run_rather_than_raising():
    assert sheet.stratify([0.0] * 4) == [(i, "S1_lead") for i in range(4)]


def test_an_episode_with_no_empty_frame_yields_nothing():
    assert sheet.stratify([0.1, 0.2, 0.3]) == []


# --------------------------------------------------------------------------------------------
# the sample — V15 §3
# --------------------------------------------------------------------------------------------

def _pooled(n_episodes: int = 40, n_frames: int = 120) -> dict:
    episodes = []
    for e in range(n_episodes):
        fractions = [0.0] * 10 + [0.4] * (n_frames - 40) + [0.0] * 30
        for offset in (15, 40, 41, 60, 61, 62):
            fractions[offset] = 0.0
        episodes.append({"episode": f"episode_{e:06d}", "area_fractions": fractions})
    return {"per_episode": episodes, "measurement_qualified": True}


def test_the_draw_is_reproducible_under_its_seed():
    pooled = _pooled()
    allocation = {"S1_lead": 5, "S2_trail": 5, "S3_int_1_2": 5, "S4_int_3_25": 5, "S5_int_26plus": 0}
    first = sheet.draw(pooled, sheet.SAMPLE_SEED, allocation)
    second = sheet.draw(pooled, sheet.SAMPLE_SEED, allocation)
    assert [(r["episode"], r["frame_index"]) for r in first] == \
           [(r["episode"], r["frame_index"]) for r in second]


def test_a_different_seed_draws_a_different_sample():
    pooled = _pooled()
    allocation = {"S1_lead": 5, "S2_trail": 5, "S3_int_1_2": 5, "S4_int_3_25": 5, "S5_int_26plus": 0}
    a = sheet.draw(pooled, sheet.SAMPLE_SEED, allocation)
    b = sheet.draw(pooled, sheet.SAMPLE_SEED + 1, allocation)
    assert [(r["episode"], r["frame_index"]) for r in a] != \
           [(r["episode"], r["frame_index"]) for r in b]


def test_the_draw_honours_the_allocation_per_stratum():
    allocation = {"S1_lead": 7, "S2_trail": 3, "S3_int_1_2": 4, "S4_int_3_25": 2, "S5_int_26plus": 0}
    drawn = sheet.draw(_pooled(), sheet.SAMPLE_SEED, allocation)
    counts = {k: 0 for k in allocation}
    for record in drawn:
        counts[record["stratum"]] += 1
    assert counts == allocation


def test_tiles_are_numbered_contiguously_in_presentation_order():
    allocation = {"S1_lead": 5, "S2_trail": 5, "S3_int_1_2": 5, "S4_int_3_25": 5, "S5_int_26plus": 0}
    drawn = sheet.draw(_pooled(), sheet.SAMPLE_SEED, allocation)
    assert [r["tile"] for r in drawn] == list(range(len(drawn)))


def test_the_presentation_order_is_shuffled_across_strata():
    # Unshuffled, the draw comes out stratum by stratum and the first tiles would all share one.
    # A reader who notices that has been handed the stratum, which V15 §3 forbids.
    allocation = {"S1_lead": 20, "S2_trail": 20, "S3_int_1_2": 20, "S4_int_3_25": 20, "S5_int_26plus": 0}
    drawn = sheet.draw(_pooled(), sheet.SAMPLE_SEED, allocation)
    assert len({r["stratum"] for r in drawn[:8]}) > 1


def test_an_unsatisfiable_allocation_is_refused_rather_than_quietly_shrunk():
    with pytest.raises(SystemExit, match="not satisfiable"):
        sheet.draw(_pooled(n_episodes=1), sheet.SAMPLE_SEED, {**dict.fromkeys(sheet.ALLOCATION, 0),
                                                              "S3_int_1_2": 10_000})


def test_the_registered_allocation_and_seed_are_the_ones_v15_names():
    # V15 §3 states these numerals. If someone changes the script, this test is the tripwire that
    # says the run is no longer the registered protocol — it is not a style assertion.
    assert sheet.SAMPLE_SEED == 40015
    assert sheet.ALLOCATION == {"S1_lead": 60, "S2_trail": 60,
                                "S3_int_1_2": 40, "S4_int_3_25": 40, "S5_int_26plus": 40}
    assert sum(sheet.ALLOCATION.values()) == 240


# --------------------------------------------------------------------------------------------
# the page — V15 §3's list of things a tile may not carry
# --------------------------------------------------------------------------------------------

@pytest.mark.skipif(not (REPO_ROOT / "runs/pr08-empty-mask-look/SAMPLE.json").is_file(),
                    reason="the sample has not been rendered in this working tree")
def test_the_built_page_leaks_no_episode_stratum_or_frame_index():
    page = (REPO_ROOT / "runs/pr08-empty-mask-look/page.html").read_text()
    sample = json.loads((REPO_ROOT / "runs/pr08-empty-mask-look/SAMPLE.json").read_text())
    # The page is one long line of base64, so search the payload's KEYS rather than substrings that
    # base64 would produce by chance.
    tiles = json.loads(page.split('id="tiles">', 1)[1].split("</script>", 1)[0])
    assert {k for tile in tiles for k in tile} == {"tile", "image"}
    assert len(tiles) == len(sample["tiles"])
    for stratum in sheet.ALLOCATION:
        assert stratum not in page
    assert "episode_0" not in page


@pytest.mark.skipif(not (REPO_ROOT / "runs/pr08-empty-mask-look/SAMPLE.json").is_file(),
                    reason="the sample has not been rendered in this working tree")
def test_the_sample_key_records_what_the_page_deliberately_omits():
    # The mapping has to exist somewhere or the verdicts cannot be weighted back to the population.
    sample = json.loads((REPO_ROOT / "runs/pr08-empty-mask-look/SAMPLE.json").read_text())
    assert sample["sample_seed"] == sheet.SAMPLE_SEED
    assert sample["allocation"] == sheet.ALLOCATION
    assert sample["verdicts_accepted"] == ["yes", "no", "cannot_tell"]
    for record in sample["tiles"]:
        assert {"tile", "episode", "frame_index", "stratum", "file"} <= set(record)
