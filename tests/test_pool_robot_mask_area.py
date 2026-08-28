"""scripts/pool_robot_mask_area.py — the pool, pinned to the evidence T40_RULE_V20 §2 registered.

WHY THIS FILE EXISTS. Slurm job 190981 asked for an H200 and died six seconds later because
`runs/pr08-robot-mask-area/POOLED.json` is not on the cluster — and it is not on the cluster because
no committed script writes it. The fix reads the sixteen shard artifacts instead. That fix is only
legitimate if the rebuild is the SAME evidence, so the load-bearing test here is the one that checks
the rebuild against the file `T40_RULE_V20` §2 registered **by sha256**, field for field.

The refusal tests build synthetic shards, so they run anywhere. The chain tests read the real shards
under gitignored `runs/` and skip when it is absent — stated rather than hidden, because a skipped
test proves nothing and the reader should know which of these two kinds they are looking at.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import statistics
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

import pool_robot_mask_area as prma  # noqa: E402

_SHARDS = _REPO / "runs/pr08-robot-mask-area/shards"
_POOLED = _REPO / "runs/pr08-robot-mask-area/POOLED.json"
_BOUND = _REPO / "configs/transfer25/pr08_robot_mask_area.json"

_needs_shards = pytest.mark.skipif(
    not _SHARDS.is_dir(), reason=f"{_SHARDS} is under gitignored runs/ and is not on this machine"
)
_needs_pooled = pytest.mark.skipif(
    not _POOLED.is_file(), reason=f"{_POOLED} is under gitignored runs/ and is not on this machine"
)


# ------------------------------------------------------------------------------------------------
# synthetic shards — enough shape for the refusals, no dependency on runs/
# ------------------------------------------------------------------------------------------------

_ESTIMATOR = {"adapter": "estimators.apple_sam2", "adapter_version": "det=x;seg=y"}


def _episode_for(index: int, num_shards: int, salt: int = 0) -> str:
    """An episode id that actually hashes to ``index``, found by counting rather than asserted."""
    n = salt
    while True:
        candidate = f"episode_{n:06d}"
        if prma._shard_of(candidate, num_shards) == index:
            return candidate
        n += 1


def _entry(episode: str, position: int, *, n_frames: int = 4, empty: int = 0,
           fractions: list[float] | None = None) -> dict:
    if fractions is None:
        fractions = [0.0] * empty + [0.1] * (n_frames - empty)
    return {
        "episode_index": position,
        "episode": episode,
        "n_frames": len(fractions),
        "empty_frames": empty,
        "area_fractions": fractions,
    }


def _shard(index: int, num_shards: int, entries: list[dict], **over) -> dict:
    record = {
        "schema": prma.SHARD_SCHEMA,
        "shard": {
            "index": index,
            "num_shards": num_shards,
            "corpus_episode_keys_sha256": "c0ffee" * 10 + "abcd",
        },
        "corpus_episode_keys": [e["episode"] for e in entries],
        "per_episode": entries,
        "measured": {"stride": 1},
        "measurement_qualified": True,
        "measurement_disqualified_reasons": [],
        "prompt": "robot arm. robotic hand. robotic gripper.",
        "estimator": _ESTIMATOR,
        "git_commit": "deadbeef",
        "source_manifest": "/x/manifest.json",
        "source_manifest_sha256": "ab" * 32,
    }
    record.update(over)
    return record


def _write(directory: pathlib.Path, records: dict[int, dict]) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    for index, record in records.items():
        (directory / f"shard-{index}.json").write_text(json.dumps(record), encoding="utf-8")
    return directory


def _two_shards(tmp_path: pathlib.Path) -> pathlib.Path:
    a, b = _episode_for(0, 2), _episode_for(1, 2)
    return _write(tmp_path / "shards", {
        0: _shard(0, 2, [_entry(a, 0)], corpus_episode_keys=[a, b]),
        1: _shard(1, 2, [_entry(b, 1)], corpus_episode_keys=[a, b]),
    })


# ------------------------------------------------------------------------------------------------
# the chain: shards -> the pool -> the file V20 §2 registered
# ------------------------------------------------------------------------------------------------


@_needs_shards
def test_the_rebuild_is_the_corpus_and_recovers_what_pooled_json_could_not_state() -> None:
    """§A.7 of the F5 investigation caught "402 episodes / 171 625 frames, stride 1" being quoted
    against POOLED.json, which carries none of those three fields. All three are derivable here."""
    pooled = prma.pool_shard_dir(_SHARDS)
    assert pooled["measurement_qualified"] is True, pooled["measurement_disqualified_reasons"]
    assert pooled["n_episodes"] == 402
    assert pooled["n_frames"] == 171_625
    assert pooled["stride"] == 1
    assert pooled["num_shards"] == 16
    assert len(pooled["pooled_from"]) == 16
    assert [s["index"] for s in pooled["pooled_from"]] == list(range(16))


@_needs_shards
@_needs_pooled
def test_the_rebuild_reproduces_the_pooled_evidence_v20_registered_by_sha256() -> None:
    """THE LOAD-BEARING TEST. Replacing POOLED.json with a derivation over its own shards is only
    not-a-rule-change if the two are the same evidence. `T40_RULE_V20` §2 registered the file by
    hash, so the hash is checked first and the contents second — a rebuild matching a file that is
    no longer the registered one would prove nothing at all."""
    digest = hashlib.sha256(_POOLED.read_bytes()).hexdigest()
    assert digest == prma.V20_REGISTERED_POOLED_SHA256, (
        "POOLED.json is no longer the file T40_RULE_V20 §2 registered; the equivalence below would "
        "be against something else"
    )
    rule = (_REPO / "docs/preregistration/PR-08-V20-timing-episode-registration.md").read_text()
    assert prma.V20_REGISTERED_POOLED_SHA256 in " ".join(rule.split()), (
        "the constant this module pins is not the one the rule document registers"
    )

    ok, why = prma.equivalent(prma.pool_shard_dir(_SHARDS), json.loads(_POOLED.read_text()))
    assert ok, why


@_needs_shards
def test_the_v20_criterion_lands_on_the_registered_episode_from_the_shards_alone() -> None:
    """V20 §3 recomputed over the rebuilt pool, with no POOLED.json anywhere in the derivation."""
    pooled = prma.pool_shard_dir(_SHARDS)
    bound = json.loads(_BOUND.read_text())["max_frame_fraction"]
    eps = pooled["per_episode"]
    survivors = [e for e in eps if e["empty_frames"] == 0 and max(e["area_fractions"]) <= bound]
    median = statistics.median([e["n_frames"] for e in eps])
    pick = min(survivors, key=lambda e: (abs(e["n_frames"] - median), str(e["episode"])))

    assert len(survivors) == 17
    assert median == 421.5
    assert pick["episode"] == "episode_000371"
    assert pick["n_frames"] == 422


# ------------------------------------------------------------------------------------------------
# what refuses, and what is merely stamped
# ------------------------------------------------------------------------------------------------


def test_a_clean_pair_of_shards_pools(tmp_path: pathlib.Path) -> None:
    pooled = prma.pool_shard_dir(_two_shards(tmp_path))
    assert pooled["measurement_qualified"] is True
    assert pooled["n_episodes"] == 2
    assert pooled["schema"] == prma.POOLED_SCHEMA


def test_a_missing_shard_is_stamped_and_written_not_raised(tmp_path: pathlib.Path) -> None:
    """`robot_composite.merge_shard_records`' own line, kept: the arithmetic over what landed is
    exactly right about the frames it saw and merely is not the corpus."""
    a, b = _episode_for(0, 2), _episode_for(1, 2)
    directory = _write(tmp_path / "shards", {0: _shard(0, 2, [_entry(a, 0)],
                                                       corpus_episode_keys=[a, b])})
    pooled = prma.pool_shard_dir(directory)
    assert pooled["measurement_qualified"] is False
    joined = " ".join(pooled["measurement_disqualified_reasons"])
    assert "did not land" in joined
    assert "in no shard" in joined


def test_a_duplicated_shard_index_refuses(tmp_path: pathlib.Path) -> None:
    directory = _two_shards(tmp_path)
    (directory / "shard-01.json").write_text(
        (directory / "shard-0.json").read_text(), encoding="utf-8"
    )
    with pytest.raises(prma.PoolError, match="appears twice"):
        prma.pool_shard_dir(directory)


def test_an_episode_in_the_wrong_shard_refuses(tmp_path: pathlib.Path) -> None:
    """The partition is re-derived, never trusted: a shard holding an episode it was not assigned
    means some other shard holds it too, and that episode is weighted twice."""
    a, b = _episode_for(0, 2), _episode_for(1, 2)
    directory = _write(tmp_path / "shards", {
        0: _shard(0, 2, [_entry(a, 0), _entry(b, 1)], corpus_episode_keys=[a, b]),
        1: _shard(1, 2, [_entry(b, 1)], corpus_episode_keys=[a, b]),
    })
    with pytest.raises(prma.PoolError, match="hashes to shard"):
        prma.pool_shard_dir(directory)


def test_a_summary_only_shard_refuses(tmp_path: pathlib.Path) -> None:
    directory = _two_shards(tmp_path)
    record = json.loads((directory / "shard-1.json").read_text())
    record["per_episode"] = []
    (directory / "shard-1.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(prma.PoolError, match="no per_episode list"):
        prma.pool_shard_dir(directory)


@pytest.mark.parametrize(
    "field, value",
    [
        ("prompt", "a different prompt"),
        ("source_manifest_sha256", "cd" * 32),
        ("estimator", {"adapter": "estimators.something_else"}),
    ],
)
def test_shards_that_disagree_about_provenance_refuse(
    tmp_path: pathlib.Path, field: str, value: object
) -> None:
    """A distribution is a claim ABOUT a corpus measured BY a segmenter. Two shards disagreeing on
    either are two measurements, and pooling them describes neither."""
    directory = _two_shards(tmp_path)
    record = json.loads((directory / "shard-1.json").read_text())
    record[field] = value
    (directory / "shard-1.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(prma.PoolError, match=f"disagree about {field}"):
        prma.pool_shard_dir(directory)


def test_shards_that_disagree_about_stride_refuse(tmp_path: pathlib.Path) -> None:
    directory = _two_shards(tmp_path)
    record = json.loads((directory / "shard-1.json").read_text())
    record["measured"]["stride"] = 4
    (directory / "shard-1.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(prma.PoolError, match="disagree about stride"):
        prma.pool_shard_dir(directory)


def test_a_merged_artifact_in_the_shard_directory_refuses_rather_than_being_skipped(
    tmp_path: pathlib.Path,
) -> None:
    """The merged artifact and the committed bound carry no per-frame fractions. Silently skipping
    one is how a pool over fifteen shards gets reported as a pool over sixteen."""
    directory = _two_shards(tmp_path)
    (directory / "shard-2.json").write_text(
        json.dumps({"schema": "wam.robot_mask_area/1", "max_frame_fraction": 0.64}),
        encoding="utf-8",
    )
    with pytest.raises(prma.PoolError, match="not 'wam.robot_mask_area_shard/1'"):
        prma.pool_shard_dir(directory)


def test_a_truncated_fraction_list_refuses(tmp_path: pathlib.Path) -> None:
    directory = _two_shards(tmp_path)
    record = json.loads((directory / "shard-1.json").read_text())
    record["per_episode"][0]["area_fractions"] = [0.1]
    (directory / "shard-1.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(prma.PoolError, match="area fractions for"):
        prma.pool_shard_dir(directory)


def test_a_disqualified_shard_disqualifies_the_pool(tmp_path: pathlib.Path) -> None:
    directory = _two_shards(tmp_path)
    record = json.loads((directory / "shard-1.json").read_text())
    record["measurement_qualified"] = False
    record["measurement_disqualified_reasons"] = ["--stride 4"]
    (directory / "shard-1.json").write_text(json.dumps(record), encoding="utf-8")
    pooled = prma.pool_shard_dir(directory)
    assert pooled["measurement_qualified"] is False
    assert "itself disqualified" in " ".join(pooled["measurement_disqualified_reasons"])


def test_an_empty_directory_refuses(tmp_path: pathlib.Path) -> None:
    empty = tmp_path / "shards"
    empty.mkdir()
    with pytest.raises(prma.PoolError, match="holds no"):
        prma.pool_shard_dir(empty)


# ------------------------------------------------------------------------------------------------
# the one entry point every consumer uses
# ------------------------------------------------------------------------------------------------


def test_load_area_evidence_takes_a_directory_or_a_file(tmp_path: pathlib.Path) -> None:
    """One function, so "which evidence answered this" has one answer on both machines."""
    directory = _two_shards(tmp_path)
    from_dir = prma.load_area_evidence(directory)
    written = tmp_path / "POOLED.json"
    written.write_text(json.dumps(from_dir), encoding="utf-8")
    ok, why = prma.equivalent(from_dir, prma.load_area_evidence(written))
    assert ok, why


def test_load_area_evidence_refuses_a_file_without_per_episode(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "not-a-pool.json"
    path.write_text(json.dumps({"max_frame_fraction": 0.64}), encoding="utf-8")
    with pytest.raises(prma.PoolError, match="no per_episode list"):
        prma.load_area_evidence(path)


def test_equivalent_ignores_order_and_nothing_else(tmp_path: pathlib.Path) -> None:
    """Order is the ONE difference between POOLED.json and a rebuild from its shards, and no
    consumer can see it — the criterion is set membership and the median is over n_frames."""
    pooled = prma.pool_shard_dir(_two_shards(tmp_path))
    shuffled = dict(pooled, per_episode=list(reversed(pooled["per_episode"])))
    ok, _ = prma.equivalent(pooled, shuffled)
    assert ok

    moved = json.loads(json.dumps(pooled))
    moved["per_episode"][0]["area_fractions"][0] += 1e-12
    ok, why = prma.equivalent(pooled, moved)
    assert not ok and "area_fractions differs" in why


# ------------------------------------------------------------------------------------------------
# the consumers, held to the same evidence
# ------------------------------------------------------------------------------------------------


def test_every_consumer_reaches_the_pool_through_this_module() -> None:
    """The defect this module exists to end: four consumers each reading an untracked file directly.
    A fifth that grows its own `json.loads(POOLED.json)` puts the repo back where it started."""
    screen = (_REPO / "cluster/discoverer/97_transfer25_restyle.sbatch").read_text()
    assert "from pool_robot_mask_area import PoolError, load_area_evidence" in screen
    assert "MASK_AREA_EVIDENCE:-${PROJ}/runs/pr08-robot-mask-area/shards}" in screen
    # And the message that sent an operator to a merge which cannot produce this is gone.
    assert "MERGE=1 writes POOLED.json" not in screen
    assert "MERGE=1 does NOT write a per-episode pool" in screen

    wrapper = (_REPO / "cluster/discoverer/submit_timing_episode.sh").read_text()
    assert "from pool_robot_mask_area import PoolError, equivalent, load_area_evidence" in wrapper
    assert "EVIDENCE=${EVIDENCE:-runs/pr08-robot-mask-area/shards}" in wrapper

    renderer = (_REPO / "scripts/render_area_tail_sheet.py").read_text()
    assert "import pool_robot_mask_area" in renderer
    assert "pool_robot_mask_area.pool_shard_dir(path)" in renderer
