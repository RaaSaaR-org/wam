"""Tests for ``scripts/robot_composite.py measure --shard/--merge`` — PR-08 §6 G0c's area bound.

The distribution these tests are about is the one ``load_area_bound``'s refusal names: the robot-
mask area fraction over the 171 625 frames of the SOURCE corpus. It is measured once, a human reads
it and writes a bound above it, and nothing downstream re-derives either. So the failures worth
pinning are not "it crashed":

  a percentile is         ``max_frame_fraction`` will sit above a distribution whose median, p95 and
  recombined from        p99 are quoted in the rationale. None of those three decompose across
  shard summaries        shards — the median of the shard medians is a different statistic with the
                         same units and a plausible magnitude. Shards therefore emit RAW per-frame
                         fractions and the merge takes the five numbers ONCE over the pool, and the
                         proof is that the merged artifact equals a whole-corpus one exactly.

  a partial merge        Eight shards, one killed at the wall, seven merged: the arithmetic is right
  looks finished         about the frames it saw and is not the corpus's. It is stamped
                         ``measurement_qualified: false`` with the failing condition named, and
                         ``load_area_bound`` refuses it — the same treatment ``--limit`` gets.

  the partition moves    ``hash()`` is seeded per interpreter, so eight array tasks would compute
  between array tasks    eight different partitions of one corpus, each internally consistent, and
                         together they would cover some episodes twice and others never.

  a script picks the     ``max_frame_fraction`` stays null on every path here. The observed maximum
  bound                  cannot fire on the frames it was measured over, and any bound above it
                         carries a margin nothing in the corpus derives.

Every corpus here is synthetic and carries its answer in its construction: the masker covers exactly
the number of pixels encoded in the frame's first byte, so every area fraction is arithmetic.
Nothing decodes video and nothing needs the real AppleToPlate corpus.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import robot_composite as rc  # noqa: E402

CANVAS = 16                      # 16x16 = 256 px, so a fraction is <covered>/256


def _frames(codes: list[int]) -> np.ndarray:
    """One clip. Frame ``i`` carries ``codes[i]`` in its first byte and nothing else."""
    arr = np.zeros((len(codes), CANVAS, CANVAS, 3), dtype=np.uint8)
    for i, code in enumerate(codes):
        arr[i, 0, 0, 0] = code
    return arr


class CodeMasker:
    """Covers exactly ``frame[0, 0, 0]`` pixels, so the area fraction is known before the run."""

    def __init__(self) -> None:
        self.calls = 0
        self.preflighted = 0

    def preflight(self) -> None:
        self.preflighted += 1

    def provenance(self) -> dict:
        return {
            "name": "code-masker",
            "version": "test",
            "prompt": rc.ROBOT_TEXT_PROMPT,
            "box_threshold": 0.15,
            "text_threshold": 0.25,
            "box_rule": "test rectangle",
            "upstream_retry_not_run": "test masker; there is no upstream retry to skip",
        }

    def mask(self, rgb) -> np.ndarray:
        self.calls += 1
        arr = np.asarray(rgb)
        covered = int(arr[0, 0, 0])
        flat = np.zeros(arr.shape[0] * arr.shape[1], dtype=bool)
        flat[:covered] = True
        return flat.reshape(arr.shape[:2])


@pytest.fixture(autouse=True)
def masker(monkeypatch):
    """Every test gets the code masker; none of them can reach GroundingDINO."""
    instance = CodeMasker()
    monkeypatch.setattr(rc, "build_masker", lambda: instance)
    return instance


#: Eleven episodes whose per-frame area fractions differ enough between shards that the median of
#: the shard medians is a different number from the pooled median. Deliberately lopsided: the same
#: shape ``measure_geom_tol``'s eleven-episode fixture has, for the same reason.
def _eleven() -> dict[str, list[int]]:
    episodes: dict[str, list[int]] = {}
    for i in range(11):
        base = 4 + i * 9                       # 4..94 covered pixels
        n = 5 + (i % 4)                        # 5..8 frames
        episodes[f"episode_{i:06d}"] = [base + (j * (i + 1)) % 17 for j in range(n)]
    return episodes


@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    return _make_corpus(tmp_path, monkeypatch, _eleven())


def _make_corpus(tmp_path, monkeypatch, episodes: dict[str, list[int]], *,
                 name: str = "source") -> pathlib.Path:
    """A manifest plus empty clip files, with ``decode_clip`` answering from the codes.

    The .mp4 files are empty and are never opened: what is being measured here is the partition and
    the pooling, and a real encode would put h264 rounding between the fixture and its own answer.
    """
    root = tmp_path / name
    (root / "videos").mkdir(parents=True, exist_ok=True)
    entries = []
    table: dict[str, np.ndarray] = {}
    for stem, codes in episodes.items():
        clip = root / "videos" / f"{stem}.mp4"
        clip.write_bytes(b"")
        table[str(clip.resolve())] = _frames(codes)
        entries.append({"id": stem, "frames": len(codes), "video": f"videos/{stem}.mp4"})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "resolution": [CANVAS, CANVAS],
        "source": {"total_episodes": len(entries),
                   "total_frames": sum(len(c) for c in episodes.values())},
        "episodes": entries,
    }), encoding="utf-8")

    existing = getattr(rc, "_TEST_CLIPS", None)
    if existing is None:
        monkeypatch.setattr(rc, "decode_clip",
                            lambda p: rc._TEST_CLIPS[str(pathlib.Path(p).resolve())])
        monkeypatch.setattr(rc, "_TEST_CLIPS", {}, raising=False)
    rc._TEST_CLIPS.update(table)
    return manifest


def _whole(manifest: pathlib.Path, out: pathlib.Path, *extra: str) -> int:
    return rc.main(["measure", "--manifest", str(manifest), "--out", str(out), *extra])


def _sharded(manifest: pathlib.Path, tmp_path: pathlib.Path, n: int, *extra: str,
             tag: str = "shard") -> list[str]:
    paths = []
    for i in range(n):
        p = tmp_path / f"{tag}-{i}.json"
        assert rc.main(["measure", "--manifest", str(manifest), "--out", str(p),
                        "--shard", str(i), "--num-shards", str(n), *extra]) == rc.EXIT_OK
        paths.append(str(p))
    return paths


def _merge(paths, out: pathlib.Path) -> int:
    return rc.main(["measure", "--merge", *[str(p) for p in paths], "--out", str(out)])


def _load(path) -> dict:
    return json.loads(pathlib.Path(path).read_text())


def _edit(path, **changes) -> str:
    rec = _load(path)
    rec.update(changes)
    pathlib.Path(path).write_text(json.dumps(rec), encoding="utf-8")
    return str(path)


# -- THE important one: the merged distribution is the whole-corpus distribution, exactly ---------


#: The one field that CANNOT match, because it describes how the artifact was produced rather than
#: what was measured. The list is short on purpose: an exact whole-record comparison is a far easier
#: property to defend than "equal in the fields we thought to check".
_PROVENANCE_ONLY = {"merged_from"}


def test_the_merged_artifact_is_the_whole_corpus_artifact_exactly(corpus, tmp_path) -> None:
    """Merged from three shards equals measured in one pass. Exactly — ``==``, not ``approx``.

    A merge that pooled correctly but recombined the percentiles from the shard summaries would
    pass an approximate check on the median and be wrong on p95 and p99 by an amount nothing
    downstream could notice. So the assertion is on the WHOLE record minus ``merged_from``, and the
    exactness is not a coincidence: shards emit raw float64 fractions, ``float -> JSON -> float`` is
    the identity, and the pool is rebuilt in the manifest's own enumeration order.
    """
    full = tmp_path / "full.json"
    assert _whole(corpus, full) == rc.EXIT_OK
    reference = _load(full)

    merged_path = tmp_path / "merged.json"
    assert _merge(_sharded(corpus, tmp_path, 3), merged_path) == rc.EXIT_OK
    merged = _load(merged_path)

    for stat in ("min", "median", "p95", "p99", "max"):
        assert merged["measured"][stat] == reference["measured"][stat], f"{stat}, bit for bit"
    differing = sorted(k for k in set(reference) | set(merged)
                       if k not in _PROVENANCE_ONLY and reference.get(k) != merged.get(k))
    assert differing == [], f"merged and whole-corpus disagree on {differing}"
    assert merged["schema"] == rc.AREA_SCHEMA
    assert merged["measurement_qualified"] is True


def test_the_five_numbers_cannot_be_recombined_from_the_shard_summaries(corpus, tmp_path) -> None:
    """The failure the whole design exists against, made visible on this fixture.

    If the merge had pooled the shard summaries instead of the raw frames, at least one of the five
    would come out different. This asserts they DO differ here, so the equality test above cannot be
    passing by accident on a fixture where every route gives the same answer.
    """
    paths = _sharded(corpus, tmp_path, 3)
    shards = [_load(p)["measured"] for p in paths]
    merged_path = tmp_path / "merged.json"
    assert _merge(paths, merged_path) == rc.EXIT_OK
    pooled = _load(merged_path)["measured"]

    assert float(np.median([s["median"] for s in shards])) != pytest.approx(pooled["median"]), (
        "this fixture is supposed to separate the two statistics")
    assert float(np.median([s["p95"] for s in shards])) != pytest.approx(pooled["p95"])
    assert float(np.max([s["p99"] for s in shards])) != pytest.approx(pooled["p99"])
    # min and max DO decompose, and that is worth stating rather than leaving implicit.
    assert min(s["min"] for s in shards) == pooled["min"]
    assert max(s["max"] for s in shards) == pooled["max"]


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_every_partition_of_one_corpus_merges_to_the_same_distribution(corpus, tmp_path, n) -> None:
    """The partition is an implementation detail of the schedule and must leave no trace.

    Stops at 4 because 11 episodes into 5 leaves a shard empty by chance, which is its own refusal
    (``test_a_partition_wider_than_the_corpus_names_the_partition``) rather than a merge.
    """
    full = tmp_path / "full.json"
    assert _whole(corpus, full) == rc.EXIT_OK
    merged_path = tmp_path / f"merged-{n}.json"
    assert _merge(_sharded(corpus, tmp_path, n, tag=f"p{n}"), merged_path) == rc.EXIT_OK
    assert _load(merged_path)["measured"] == _load(full)["measured"]


def test_the_merge_does_not_depend_on_the_order_the_shards_were_named_in(corpus, tmp_path) -> None:
    paths = _sharded(corpus, tmp_path, 4)
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    assert _merge(paths, first) == rc.EXIT_OK
    assert _merge(list(reversed(paths)), second) == rc.EXIT_OK
    assert _load(first)["measured"] == _load(second)["measured"]


def test_a_directory_of_shards_merges_and_skips_what_is_not_a_shard(corpus, tmp_path, capsys):
    """The merge job is pointed at a directory that also holds the pilot and the previous merge."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    for i in range(3):
        p = shard_dir / f"shard-{i}.json"
        assert rc.main(["measure", "--manifest", str(corpus), "--out", str(p),
                        "--shard", str(i), "--num-shards", "3"]) == rc.EXIT_OK
    (shard_dir / "PILOT.json").write_text(json.dumps({"schema": "something.else/1"}))

    out = tmp_path / "merged.json"
    assert _merge([shard_dir], out) == rc.EXIT_OK
    assert "skipping" in capsys.readouterr().err
    assert _load(out)["merged_from"]["num_shards"] == 3


# -- the partition itself -------------------------------------------------------------------------


def test_the_partition_does_not_depend_on_the_interpreters_hash_seed() -> None:
    """Eight array tasks are eight interpreters. ``hash()`` would give them eight partitions."""
    prog = (
        "import sys; sys.path.insert(0, %r); import robot_composite as m;"
        "print(','.join(str(m.shard_of('episode_%%06d' %% i, 8)) for i in range(11)))"
        % str(_REPO_ROOT / "scripts")
    )
    import os
    outs = []
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outs.append(subprocess.run([sys.executable, "-c", prog], env=env, check=True,
                                   capture_output=True, text=True).stdout.strip())
    assert outs[0] == outs[1], f"the partition moved with PYTHONHASHSEED: {outs}"
    assert outs[0] == ",".join(str(rc.shard_of(f"episode_{i:06d}", 8)) for i in range(11))


def test_the_partition_is_the_one_measure_geom_tol_uses() -> None:
    """Copied rather than imported, so a test has to be the thing that stops it drifting."""
    import measure_geom_tol as mgt

    assert rc.SHARD_ASSIGNMENT == mgt.SHARD_ASSIGNMENT
    keys = [f"episode_{i:06d}" for i in range(402)]
    for n in (2, 3, 4, 8, 16):
        assert [rc.shard_of(k, n) for k in keys] == [mgt.shard_of(k, n) for k in keys], n


def test_adding_an_episode_moves_that_episode_and_no_other(tmp_path, monkeypatch) -> None:
    """A range would renumber every episode after the insertion; a key digest moves one."""
    before = _eleven()
    manifest_a = _make_corpus(tmp_path, monkeypatch, before, name="a")
    after = dict(list(before.items())[:5] + [("episode_999999", [7, 9, 11])]
                 + list(before.items())[5:])
    manifest_b = _make_corpus(tmp_path, monkeypatch, after, name="b")

    def assignment(manifest):
        keys = [e["id"] for e in _load(manifest)["episodes"]]
        return {k: rc.shard_of(k, 4) for k in keys}

    a, b = assignment(manifest_a), assignment(manifest_b)
    assert all(b[k] == v for k, v in a.items()), "an existing episode changed shard"
    assert set(b) - set(a) == {"episode_999999"}


def test_every_episode_index_is_its_place_in_the_full_enumeration(corpus, tmp_path) -> None:
    """Not a serial number within the shard: it is what rebuilds the pool in corpus order."""
    keys = [e["id"] for e in _load(corpus)["episodes"]]
    for path in _sharded(corpus, tmp_path, 3):
        rec = _load(path)
        for entry in rec["per_episode"]:
            assert keys[entry["episode_index"]] == entry["episode"]
        assert rec["shard"]["episode_indices"] == [e["episode_index"] for e in rec["per_episode"]]


def test_a_shard_carries_the_raw_fractions_and_the_merge_carries_none(corpus, tmp_path) -> None:
    """The raw arrays are the shard's contribution to the pool, not part of the distribution.

    They stay in the shard artifacts, which ``merged_from.shards`` names and digests.
    """
    paths = _sharded(corpus, tmp_path, 3)
    total = 0
    for p in paths:
        rec = _load(p)
        for entry in rec["per_episode"]:
            assert len(entry["area_fractions"]) == entry["n_frames"]
            total += entry["n_frames"]
    out = tmp_path / "merged.json"
    assert _merge(paths, out) == rc.EXIT_OK
    merged = _load(out)
    assert "per_episode" not in merged and "area_fractions" not in json.dumps(merged)
    assert merged["measured"]["frames"] == total
    assert {s["sha256"] for s in merged["merged_from"]["shards"]}


# -- the six qualification conditions, one test each ----------------------------------------------


def _conditions(path) -> dict:
    return _load(path)["merged_from"]["qualification"]


def test_a_qualified_merge_says_so_on_every_condition(corpus, tmp_path) -> None:
    out = tmp_path / "merged.json"
    assert _merge(_sharded(corpus, tmp_path, 3), out) == rc.EXIT_OK
    rec = _load(out)
    assert rec["measurement_qualified"] is True
    assert rec["measurement_disqualified_reasons"] == []
    assert set(_conditions(out)) == set(rc.MERGE_CONDITIONS)
    assert all(_conditions(out).values())


def test_a_missing_shard_is_a_distribution_over_part_of_the_corpus(corpus, tmp_path, capsys):
    """It is written, stamped and refused by the loader — not silently dropped and not merged."""
    paths = _sharded(corpus, tmp_path, 4)
    out = tmp_path / "merged.json"
    assert _merge(paths[:3], out) == rc.EXIT_MEASUREMENT_NOT_QUALIFIED

    rec = _load(out)
    assert rec["measurement_qualified"] is False
    assert _conditions(out)["shards_tile_the_corpus_exactly_once"] is False
    assert all(v for k, v in _conditions(out).items()
               if k != "shards_tile_the_corpus_exactly_once")
    assert any("missing" in r for r in rec["measurement_disqualified_reasons"])
    assert any("never measured" in r for r in rec["measurement_disqualified_reasons"])
    assert "shards_tile_the_corpus_exactly_once" in capsys.readouterr().err
    with pytest.raises(rc.CompositeError, match="smoke run|not the corpus"):
        rc.load_area_bound(out)


def test_a_shard_measured_with_a_limit_disqualifies_the_merge(corpus, tmp_path) -> None:
    """--limit on a shard is a shakedown, and a merge must not launder one into a distribution."""
    paths = _sharded(corpus, tmp_path, 2)
    limited = tmp_path / "limited.json"
    assert rc.main(["measure", "--manifest", str(corpus), "--out", str(limited),
                    "--shard", "0", "--num-shards", "2",
                    "--limit", "6"]) == rc.EXIT_MEASUREMENT_NOT_QUALIFIED

    out = tmp_path / "merged.json"
    assert _merge([limited, paths[1]], out) == rc.EXIT_MEASUREMENT_NOT_QUALIFIED
    cond = _conditions(out)
    assert cond["every_shard_at_stride_1_with_no_limit"] is False
    assert cond["every_shard_measurement_qualified"] is False
    assert _load(out)["measured"]["limit"] == 6


def test_a_shard_measured_at_a_stride_disqualifies_the_merge(corpus, tmp_path) -> None:
    paths = _sharded(corpus, tmp_path, 2)
    strided = tmp_path / "strided.json"
    assert rc.main(["measure", "--manifest", str(corpus), "--out", str(strided),
                    "--shard", "0", "--num-shards", "2",
                    "--stride", "3"]) == rc.EXIT_MEASUREMENT_NOT_QUALIFIED

    out = tmp_path / "merged.json"
    assert _merge([strided, paths[1]], out) == rc.EXIT_MEASUREMENT_NOT_QUALIFIED
    assert _conditions(out)["every_shard_at_stride_1_with_no_limit"] is False
    assert _load(out)["measured"]["stride"] == 3


def test_a_shard_that_calls_itself_unqualified_disqualifies_the_merge(corpus, tmp_path) -> None:
    """The flag is checked as well as the flags that would set it.

    A shard artifact from an older version, or one edited by hand, can carry
    ``measurement_qualified: false`` for a reason this merge has no other way to see.
    """
    paths = _sharded(corpus, tmp_path, 2)
    _edit(paths[0], measurement_qualified=False,
          measurement_disqualified_reasons=["the operator says so"])

    out = tmp_path / "merged.json"
    assert _merge(paths, out) == rc.EXIT_MEASUREMENT_NOT_QUALIFIED
    cond = _conditions(out)
    assert cond["every_shard_measurement_qualified"] is False
    assert cond["every_shard_at_stride_1_with_no_limit"] is True, (
        "the two conditions are separate: this shard was not truncated")
    assert any("the operator says so" in r
               for r in _load(out)["measurement_disqualified_reasons"])


@pytest.mark.parametrize(
    "condition, field, value",
    [
        ("shards_agree_on_estimator", "estimator",
         {"name": "other", "version": "other", "prompt": rc.ROBOT_TEXT_PROMPT,
          "box_threshold": 0.9, "text_threshold": 0.9, "box_rule": "other",
          "upstream_retry_not_run": "other"}),
        ("shards_agree_on_source_manifest_sha256", "source_manifest_sha256", "0" * 64),
        ("shards_agree_on_prompt", "prompt", "robot."),
    ],
)
def test_shards_that_did_not_measure_one_quantity_disqualify_the_merge(
        corpus, tmp_path, condition, field, value) -> None:
    """Three fields, three conditions, and the merged artifact names the field as null.

    Null rather than shard 0's value: the artifact is stamped ``measurement_qualified: false`` and
    ``load_area_bound`` refuses it on that before it ever reads either field, so a null here cannot
    be mistaken for a segmenter or a prompt — while shard 0's value in that slot could be.
    """
    paths = _sharded(corpus, tmp_path, 2)
    _edit(paths[1], **{field: value})

    out = tmp_path / "merged.json"
    assert _merge(paths, out) == rc.EXIT_MEASUREMENT_NOT_QUALIFIED
    rec = _load(out)
    assert _conditions(out)[condition] is False
    assert rec[field] is None
    assert [d["value"] for d in rec["merged_from"]["disagreements"][field]] != []
    assert any(field in r for r in rec["measurement_disqualified_reasons"])
    with pytest.raises(rc.CompositeError):
        rc.load_area_bound(out)


def test_an_estimator_that_differs_only_outside_the_identity_still_merges(corpus, tmp_path) -> None:
    """The comparison is ``segmenter_identity``, the same tuple the mask cache is keyed on.

    A field outside it — the adapter's own version string, say — is provenance and not a different
    segmenter, and refusing on it would refuse every correct merge whose shards ran on two nodes.
    """
    paths = _sharded(corpus, tmp_path, 2)
    rec = _load(paths[1])
    rec["estimator"]["adapter_version"] = "a different string"
    pathlib.Path(paths[1]).write_text(json.dumps(rec), encoding="utf-8")

    out = tmp_path / "merged.json"
    assert _merge(paths, out) == rc.EXIT_OK
    assert _conditions(out)["shards_agree_on_estimator"] is True


# -- the refusals that write nothing --------------------------------------------------------------


def _refuses(paths, out: pathlib.Path, capsys, needle: str) -> None:
    assert _merge(paths, out) == rc.EXIT_REFUSED
    assert not out.exists(), "a refusal that wrote an artifact is not a refusal"
    assert needle in capsys.readouterr().err


def test_two_artifacts_claiming_the_same_shard_are_refused(corpus, tmp_path, capsys) -> None:
    paths = _sharded(corpus, tmp_path, 2)
    twin = tmp_path / "twin.json"
    twin.write_text(pathlib.Path(paths[0]).read_text(), encoding="utf-8")
    _refuses([*paths, twin], tmp_path / "merged.json", capsys, "both claim shard index 0")


def test_shards_from_two_different_partitions_are_refused(corpus, tmp_path, capsys) -> None:
    two = _sharded(corpus, tmp_path, 2, tag="two")
    three = _sharded(corpus, tmp_path, 3, tag="three")
    _refuses([two[0], three[1]], tmp_path / "merged.json", capsys, "disagree on num_shards")


def test_an_episode_in_a_shard_it_does_not_hash_to_is_refused(corpus, tmp_path, capsys) -> None:
    paths = _sharded(corpus, tmp_path, 2)
    rec = _load(paths[0])
    stolen = _load(paths[1])["shard"]["episode_keys"][0]
    rec["shard"]["episode_keys"].append(stolen)
    pathlib.Path(paths[0]).write_text(json.dumps(rec), encoding="utf-8")
    _refuses(paths, tmp_path / "merged.json", capsys, "do not hash to it")


def test_a_shard_that_pooled_an_episode_it_was_not_assigned_is_refused(
        corpus, tmp_path, capsys) -> None:
    """A double count in the pool while the coverage arithmetic still adds up."""
    paths = _sharded(corpus, tmp_path, 2)
    rec = _load(paths[0])
    rec["per_episode"].append(_load(paths[1])["per_episode"][0])
    pathlib.Path(paths[0]).write_text(json.dumps(rec), encoding="utf-8")
    _refuses(paths, tmp_path / "merged.json", capsys, "it was not assigned")


def test_a_shard_that_did_not_measure_what_it_was_assigned_is_refused(
        corpus, tmp_path, capsys) -> None:
    paths = _sharded(corpus, tmp_path, 2)
    rec = _load(paths[0])
    dropped = rec["per_episode"].pop()
    pathlib.Path(paths[0]).write_text(json.dumps(rec), encoding="utf-8")
    _merge(paths, tmp_path / "merged.json")
    err = capsys.readouterr().err
    assert "UNACCOUNTED FOR" in err and dropped["episode"] in err


def test_a_shard_that_kept_only_its_summary_cannot_be_merged(corpus, tmp_path, capsys) -> None:
    """A percentile does not decompose, so a summary-only shard can be averaged and not merged."""
    paths = _sharded(corpus, tmp_path, 2)
    rec = _load(paths[0])
    for entry in rec["per_episode"]:
        entry.pop("area_fractions")
    pathlib.Path(paths[0]).write_text(json.dumps(rec), encoding="utf-8")
    _refuses(paths, tmp_path / "merged.json", capsys, "no\n       area_fractions")


def test_a_shard_with_no_enumeration_cannot_prove_coverage(corpus, tmp_path, capsys) -> None:
    paths = _sharded(corpus, tmp_path, 2)
    for p in paths:
        _edit(p, corpus_episode_keys=[])
    _refuses(paths, tmp_path / "merged.json", capsys, "cannot prove")


def test_a_shard_with_no_shard_block_is_not_a_shard(corpus, tmp_path, capsys) -> None:
    paths = _sharded(corpus, tmp_path, 2)
    _edit(paths[0], shard={})
    _refuses(paths, tmp_path / "merged.json", capsys, "no usable 'shard' block")


def test_a_finished_measurement_is_not_an_input_to_a_merge(corpus, tmp_path, capsys) -> None:
    full = tmp_path / "full.json"
    assert _whole(corpus, full) == rc.EXIT_OK
    _refuses([full], tmp_path / "merged.json", capsys, "not an input to a merge")


def test_a_shard_named_explicitly_and_absent_is_a_missing_shard_not_a_filter(
        tmp_path, capsys) -> None:
    _refuses([tmp_path / "nope.json"], tmp_path / "merged.json", capsys, "does not exist")


def test_a_merge_over_no_shards_is_a_missing_input(tmp_path, capsys) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    _refuses([empty], tmp_path / "merged.json", capsys, "no shard artifacts at all")


def test_a_truncated_shard_artifact_is_named_rather_than_merged_around(
        corpus, tmp_path, capsys) -> None:
    paths = _sharded(corpus, tmp_path, 2)
    pathlib.Path(paths[0]).write_text('{"schema": "wam.robot_mask_area_shard/1", "sh',
                                      encoding="utf-8")
    _refuses(paths, tmp_path / "merged.json", capsys, "could not parse")


# -- the flags ------------------------------------------------------------------------------------


def test_a_shard_refuses_to_write_the_tracked_artifact_path(corpus, capsys) -> None:
    """N array tasks writing one path is a race whose winner is whichever finished last.

    This asserted ``not AREA_BOUND_ARTIFACT.exists()`` until 2026-08-26, when the bound was
    decided under T40_RULE_V13 and that path became a committed file. The assertion was wrong
    even before it failed: it conflated *the shard did not write this path* with *nothing has
    ever written this path*, so it would have passed for the wrong reason on any tree where the
    bound was simply undecided. What the refusal actually promises is that a shard run leaves
    the tracked artifact BYTE-FOR-BYTE ALONE — including when it is already there, which is the
    case that matters now and is the only case where a race could destroy anything.
    """
    before = (
        rc.AREA_BOUND_ARTIFACT.read_bytes() if rc.AREA_BOUND_ARTIFACT.exists() else None
    )
    assert rc.main(["measure", "--manifest", str(corpus),
                    "--shard", "0", "--num-shards", "4"]) == rc.EXIT_REFUSED
    assert "refuses to write the tracked default" in capsys.readouterr().err
    after = (
        rc.AREA_BOUND_ARTIFACT.read_bytes() if rc.AREA_BOUND_ARTIFACT.exists() else None
    )
    assert after == before, "a refused shard run touched the tracked area-bound artifact"


def test_shard_and_num_shards_go_together(corpus, tmp_path, capsys) -> None:
    assert rc.main(["measure", "--manifest", str(corpus), "--out", str(tmp_path / "s.json"),
                    "--shard", "0"]) == rc.EXIT_REFUSED
    assert "go together" in capsys.readouterr().err


def test_a_shard_index_outside_the_partition_is_refused(corpus, tmp_path, capsys) -> None:
    assert rc.main(["measure", "--manifest", str(corpus), "--out", str(tmp_path / "s.json"),
                    "--shard", "4", "--num-shards", "4"]) == rc.EXIT_REFUSED
    assert "out of range" in capsys.readouterr().err


def test_merging_and_sharding_on_one_command_line_are_refused(corpus, tmp_path, capsys) -> None:
    assert rc.main(["measure", "--merge", str(tmp_path / "x.json"), "--shard", "0",
                    "--num-shards", "2", "--out", str(tmp_path / "m.json")]) == rc.EXIT_REFUSED
    assert "two different jobs" in capsys.readouterr().err


def test_a_merge_takes_no_manifest_and_no_truncation(corpus, tmp_path, capsys) -> None:
    out = tmp_path / "m.json"
    assert rc.main(["measure", "--merge", str(tmp_path / "x.json"),
                    "--manifest", str(corpus), "--out", str(out)]) == rc.EXIT_REFUSED
    assert "does not read the corpus" in capsys.readouterr().err
    assert rc.main(["measure", "--merge", str(tmp_path / "x.json"),
                    "--out", str(out), "--stride", "3"]) == rc.EXIT_REFUSED
    assert "takes no --limit and no --stride" in capsys.readouterr().err


def test_measure_still_requires_a_manifest_when_it_is_not_merging(tmp_path, capsys) -> None:
    assert rc.main(["measure", "--out", str(tmp_path / "x.json")]) == rc.EXIT_REFUSED
    assert "--manifest is required" in capsys.readouterr().err


def test_a_merge_never_builds_the_segmenter(corpus, tmp_path, monkeypatch) -> None:
    """The merge job runs on the free CPU QoS with no GPU and no weights staged."""
    paths = _sharded(corpus, tmp_path, 2)

    def explode():
        raise AssertionError("--merge built the masker")

    monkeypatch.setattr(rc, "build_masker", explode)
    assert _merge(paths, tmp_path / "merged.json") == rc.EXIT_OK


def test_a_partition_wider_than_the_corpus_names_the_partition(corpus, tmp_path, capsys) -> None:
    """An empty shard would be a well-formed artifact contributing no frames."""
    assert rc.main(["measure", "--manifest", str(corpus), "--out", str(tmp_path / "s.json"),
                    "--shard", "0", "--num-shards", "500"]) == rc.EXIT_REFUSED
    assert "was assigned no frames at all" in capsys.readouterr().err


def test_a_manifest_with_a_repeated_episode_id_is_refused(tmp_path, monkeypatch, capsys) -> None:
    """Ids are what the partition is keyed on and what the merge proves coverage with."""
    episodes = _eleven()
    manifest = _make_corpus(tmp_path, monkeypatch, episodes)
    doc = _load(manifest)
    doc["episodes"].append(dict(doc["episodes"][0]))
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    assert _whole(manifest, tmp_path / "x.json") == rc.EXIT_REFUSED
    assert "more than once" in capsys.readouterr().err


# -- the bound is still nobody's to coin -----------------------------------------------------------


def test_no_path_here_coins_a_bound(corpus, tmp_path) -> None:
    """Requirement zero: this file measures, and it does not decide."""
    full = tmp_path / "full.json"
    assert _whole(corpus, full) == rc.EXIT_OK
    paths = _sharded(corpus, tmp_path, 3)
    merged_path = tmp_path / "merged.json"
    assert _merge(paths, merged_path) == rc.EXIT_OK

    for path in (full, merged_path, *[pathlib.Path(p) for p in paths]):
        rec = _load(path)
        assert rec["max_frame_fraction"] is None, path
        assert not str(rec.get("bound_rationale") or "").strip(), path
    # And what the merge writes is still refused by the loader for exactly that reason.
    with pytest.raises(rc.CompositeError, match="has been measured and the bound has not"):
        rc.load_area_bound(merged_path)


def test_a_shard_artifact_can_never_be_read_as_a_bound(corpus, tmp_path) -> None:
    """It carries no bound_rationale at all, so the loader refuses it on shape.

    A shard copied to the committed path by a tired operator has to be a refusal and not a bound
    over a twelfth of the corpus.
    """
    shard = pathlib.Path(_sharded(corpus, tmp_path, 3)[0])
    assert "bound_rationale" not in _load(shard)
    with pytest.raises(rc.CompositeError, match="missing"):
        rc.load_area_bound(shard)


def test_the_module_suggests_no_number_anywhere(tmp_path) -> None:
    """No default, no heuristic, no 'suggested' value — checked against the source, not asserted.

    ``max_frame_fraction`` is only ever assigned ``None`` in this module. A line that assigned it a
    float would be a threshold pre-registered by a default, which is what PR-08 §6's "No threshold
    is coined" forbids.
    """
    source = (_REPO_ROOT / "scripts" / "robot_composite.py").read_text(encoding="utf-8")
    coined = [line.strip() for line in source.splitlines()
              if '"max_frame_fraction":' in line
              and not line.lstrip().startswith("#")
              and "None" not in line
              # AreaBound.record() echoes the number it READ out of the committed artifact. That is
              # the one place a float may appear, and it is a copy rather than a choice.
              and "self.max_frame_fraction" not in line]
    assert coined == [], f"a max_frame_fraction that is not null is written somewhere: {coined}"
    assert 'AreaBound(' in source and 'max_frame_fraction=value' in source, (
        "the only float that reaches max_frame_fraction must be the one READ out of the committed "
        "artifact")
    for banned in ("suggested_bound", "recommended_bound", "default_bound"):
        assert banned not in source


# -- the change is additive: a whole-corpus run produces what it always did -------------------------


_BASELINE_COMMIT = "98d402a"


def _previous_version(tmp_path: pathlib.Path):
    """``scripts/robot_composite.py`` as of the commit before this change, imported side by side."""
    import importlib.util

    src = subprocess.run(["git", "show", f"{_BASELINE_COMMIT}:scripts/robot_composite.py"],
                         cwd=str(_REPO_ROOT), capture_output=True, text=True, check=True).stdout
    anchor = "REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]"
    assert src.count(anchor) == 1, "robot_composite no longer derives REPO_ROOT that way"
    src = src.replace(anchor, f"REPO_ROOT = pathlib.Path({str(_REPO_ROOT)!r})")
    path = tmp_path / "robot_composite_baseline.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("robot_composite_baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sharding_changed_no_number_a_whole_corpus_run_already_produced(
        corpus, tmp_path, monkeypatch, masker) -> None:
    """THE WHOLE CLAIM OF THIS CHANGE, checked against the script as it was rather than asserted.

    The shard and merge modes are additive: a run with neither flag must write what it wrote at
    ``98d402a``. The way to be sure of that is not to read the diff — it is to run the previous
    version over the same fixture and compare the two records key for key.
    """
    old = _previous_version(tmp_path)
    monkeypatch.setattr(old, "decode_clip", rc.decode_clip)

    before = old.measure_source_mask_area(corpus, masker=masker)
    after = rc.measure_source_mask_area(corpus, masker=masker)

    assert set(after) - set(before) == {"schema", "git_commit", "git_commit_source"}, (
        # git_commit/_source were added 2026-08-24 after the GEOM_TOL array was found to have
        # measured six of its sixteen shards with a superseded adapter, unnoticed, because the
        # cluster copy is an rsync target with no .git and nothing in the artifact could say so.
        "a key appeared that was not declared")
    assert set(before) - set(after) == set(), "a key the previous version wrote went missing"
    differing = sorted(k for k in before if before[k] != after[k])
    assert differing == [], f"sharding changed {differing}"

    # And on the truncated path, where the disqualification is what a reader depends on.
    before = old.measure_source_mask_area(corpus, masker=masker, limit=3, stride=2)
    after = rc.measure_source_mask_area(corpus, masker=masker, limit=3, stride=2)
    assert set(after) - set(before) == {"schema", "git_commit", "git_commit_source"}
    assert sorted(k for k in before if before[k] != after[k]) == []


def test_the_merged_artifact_is_what_the_previous_version_would_have_written(
        corpus, tmp_path, monkeypatch, masker) -> None:
    """The merge's own record-building path, against the same baseline.

    ``merged_from`` and ``schema`` are the merge's; everything else must be the number the previous
    version produced from one pass over the same corpus.
    """
    old = _previous_version(tmp_path)
    monkeypatch.setattr(old, "decode_clip", rc.decode_clip)
    before = old.measure_source_mask_area(corpus, masker=masker)

    out = tmp_path / "merged.json"
    assert _merge(_sharded(corpus, tmp_path, 4), out) == rc.EXIT_OK
    after = _load(out)

    assert set(after) - set(before) == {"schema", "merged_from",
                                       "git_commit", "git_commit_source"}
    differing = sorted(k for k in before if before[k] != after[k])
    assert differing == [], f"the merge changed {differing}"


def test_shards_that_enumerated_different_corpora_are_refused(corpus, tmp_path, capsys) -> None:
    """The tiling arithmetic is done against one shard's enumeration, so they must all be one.

    Two trees holding the same episode ids digest identically and are caught by the source-manifest
    condition instead; this catches the case where the episode LIST itself moved between shards.
    """
    paths = _sharded(corpus, tmp_path, 2)
    rec = _load(paths[1])
    rec["shard"]["corpus_episode_keys_sha256"] = "0" * 64
    pathlib.Path(paths[1]).write_text(json.dumps(rec), encoding="utf-8")
    _refuses(paths, tmp_path / "merged.json", capsys, "enumerated different corpora")


# -- the two failures 2026-08-23 put on the record, pinned so they cannot come back ---------------


_SBATCH = _REPO_ROOT / "cluster" / "discoverer" / "106_measure_robot_mask_area.sbatch"


def test_a_shard_is_qualified_by_its_own_truncation_and_by_nothing_in_the_adapter() -> None:
    """``measurement_qualified`` is a function of ``--limit``/``--stride`` and of nothing else.

    Commit ``4102e2e`` records the defect this pins, in the sibling harness: ``measure_geom_tol``
    coupled its shard's exit code to ``apple_sam2.GATE_QUALIFIED``, so every ARITHMETICALLY CORRECT
    shard exited 3, the sbatch treated any non-zero return as fatal, the resume validator refused a
    ``gate_qualified: false`` artifact — and an array of any width could never converge. Sharding
    was bought for resumability and handed it straight back.

    This module is not coupled that way and must not become so: a gate-qualification flag is a
    statement about whether the ESTIMATOR may produce a gate number, and ``measurement_qualified``
    is a statement about whether THIS RUN saw the whole corpus. Two different facts. The behaviour
    is already covered — ``_sharded`` asserts ``EXIT_OK`` on every untruncated shard — so what is
    added here is the coupling itself, checked against the source, because the behavioural test
    would keep passing right up until the day someone imports the flag.
    """
    module = (_REPO_ROOT / "scripts" / "robot_composite.py").read_text(encoding="utf-8")
    for banned in ("GATE_QUALIFIED", "GATE_QUALIFICATION_BLOCKERS", "gate_qualified"):
        offenders = [line.strip() for line in module.splitlines() if banned in line]
        assert offenders == [], (
            f"robot_composite.py now mentions {banned}: {offenders}. If the intent is to couple "
            "this measurement's qualification to the adapter's gate flag, read 4102e2e first — "
            "that is the change that made every correct shard exit 3."
        )
    assert "measurement_qualified" in module


def test_the_sbatch_recipes_tile_the_partition_and_fit_the_submission_limits() -> None:
    """106's committed SHARD waves cover 0..N-1 exactly once, and MERGE insists on the same N.

    A header that recommends one N while its merge recipe insists on another produces a partition
    that can never be merged, and the operator finds out after the GPU is spent. The submission
    limits are checked in the same breath because they are why the waves exist at all:
    ``MaxSubmitJobsPU=8`` counts every array task as a submission and ``DenyOnLimit`` rejects the
    surplus rather than queueing it, so a wave may not exceed 8 indices; ``%k`` throttles RUNNING
    tasks only and must stay within ``MaxJobsPU=4``.
    """
    import re

    text = _SBATCH.read_text(encoding="utf-8")
    header = text.split("#SBATCH", 1)[0]

    waves = re.findall(
        r"SHARD=1\s+NUM_SHARDS=(\d+)\s+sbatch\s+--array=(\d+)-(\d+)%(\d+)", header)
    assert waves, "106's header no longer carries an explicit SHARD recipe"

    ns = {int(w[0]) for w in waves}
    assert len(ns) == 1, f"the SHARD waves disagree about NUM_SHARDS: {sorted(ns)}"
    n = ns.pop()

    covered: list[int] = []
    for _, lo, hi, throttle in waves:
        indices = list(range(int(lo), int(hi) + 1))
        assert len(indices) <= 8, (
            f"wave {lo}-{hi} is {len(indices)} submissions against MaxSubmitJobsPU=8; %N throttles "
            "running tasks only and buys no submission slot")
        assert int(throttle) <= 4, f"%{throttle} exceeds MaxJobsPU=4"
        covered.extend(indices)
    assert sorted(covered) == list(range(n)), (
        f"the waves cover {sorted(covered)}, which is not 0..{n - 1} exactly once")

    merges = re.findall(r"MERGE=1\s+NUM_SHARDS=(\d+)", header)
    assert merges, "106's header no longer carries a MERGE recipe"
    assert {int(m) for m in merges} == {n}, (
        f"the MERGE recipe insists on NUM_SHARDS={merges} while the waves build {n}. A merge over "
        "a different N is stamped measurement_qualified:false and load_area_bound refuses it.")
