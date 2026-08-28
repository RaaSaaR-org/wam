"""The TIMING=1 path of `97_transfer25_restyle.sbatch`: which unit it times, and what it records.

WHY THIS FILE EXISTS. PR-08 §8 item 3 is one measurement — "one timed episode on an H200" — and
this job is the only thing that can make it. Everything about that measurement that can be wrong
BEFORE the GPU is asked for is checked here, because after it the cost is an H200 slot and the
outcome is still no `THROUGHPUT.json`.

The two properties under test are the two the item cannot close without:

  * THE TIMED UNIT MUST BE ONE THIS JOB CAN FINISH. `head -1` of a deterministically sorted work
    list is `episode_000000__train-01-oak-tungsten__r00`, and that episode's robot mask is EMPTY on
    254 of its 590 source frames — `robot_composite.check_mask` refuses a clip on ONE empty frame
    (scripts/robot_composite.py:1388, "zero is zero"). So the deterministic choice is a choice that
    is refused every time it is made, on evidence that has been sitting in
    `runs/pr08-robot-mask-area/POOLED.json` since 2026-08-25. The precondition under test reads that
    evidence before the driver is invoked.
  * THE TIMED UNIT MUST BE NAMEABLE FROM THE ARTIFACT. A seconds-per-frame whose episode nobody can
    identify prices 4.29 M frames against an anecdote, so `THROUGHPUT.json` has to carry which unit
    was timed, how it was selected, and what said it was admissible.

Both heredocs are extracted verbatim and executed as subprocesses, the same way
`test_97_stage_selector` runs the work-list expansion. Re-typing either here would test a copy.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SBATCH_97 = REPO / "cluster/discoverer/97_transfer25_restyle.sbatch"
#: The real pooled measurement. Under `runs/`, therefore gitignored and therefore optional: the
#: tests that need a corpus build their own. The one test that uses this asserts the actual
#: consequence on the actual corpus and skips where the artifact was never fetched.
POOLED = REPO / "runs/pr08-robot-mask-area/POOLED.json"
BOUND_ARTIFACT = REPO / "configs/transfer25/pr08_robot_mask_area.json"


def _heredoc(anchor: str) -> str:
    """The `python - ... <<'PY' ... PY` block whose command line starts at `anchor`, verbatim."""
    text = SBATCH_97.read_text(encoding="utf-8")
    assert anchor in text, f"97 no longer carries a heredoc starting {anchor!r}"
    body = text[text.index(anchor):]
    # `<<'PY'` may be followed by more of the command line (`|| exit 1`); bash starts the body at
    # the next newline and so does this.
    opener = body.index("\n", body.index("<<'PY'")) + 1
    end = body.index("\nPY\n", opener)
    return body[opener:end]


PRECONDITION_ANCHOR = 'python - "${TIMING_UNIT}" "${MASK_AREA_EVIDENCE}"'
THROUGHPUT_ANCHOR = 'python - "${TIMING_UNIT}" "${TIMING_WALL}" "${THROUGHPUT}"'


def _manifest(tmp_path: pathlib.Path, episodes: dict[str, int]) -> pathlib.Path:
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({
        "resolution": [640, 480],
        "episodes": [{"id": k, "frames": v} for k, v in episodes.items()],
    }), encoding="utf-8")
    return p


def _pooled(tmp_path: pathlib.Path, manifest: pathlib.Path,
            per_episode: list[dict], **over) -> pathlib.Path:
    payload = {
        "schema": "wam.robot_mask_area/1",
        "measurement_qualified": True,
        "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "prompt": "robot arm. robotic hand. robotic gripper.",
        "per_episode": per_episode,
    }
    payload.update(over)
    p = tmp_path / "POOLED.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _bound(tmp_path: pathlib.Path, value: float | None = 0.64091145833333329) -> pathlib.Path:
    p = tmp_path / "pr08_robot_mask_area.json"
    p.write_text(json.dumps({"schema": "wam.robot_mask_area/1", "max_frame_fraction": value}),
                 encoding="utf-8")
    return p


def _episode(name: str, fractions: list[float]) -> dict:
    return {"episode": name, "n_frames": len(fractions), "area_fractions": fractions,
            "empty_frames": sum(1 for f in fractions if f == 0.0)}


def _precondition(tmp_path: pathlib.Path, *, unit: dict, pooled: pathlib.Path,
                  manifest: pathlib.Path, bound: pathlib.Path) -> tuple[int, str, dict | None]:
    script = tmp_path / "precondition_from_97.py"
    script.write_text(_heredoc(PRECONDITION_ANCHOR), encoding="utf-8")
    out = tmp_path / "timing_unit_admissibility.json"
    # argv[6] is ${WAM}: since 2026-08-28 the screen imports scripts/pool_robot_mask_area.py so
    # that it accepts a DIRECTORY of shard artifacts as well as a pooled file. The repo root is
    # what the sbatch passes, and passing the real one here is what keeps these tests exercising the
    # screen's own import rather than a stub of it.
    proc = subprocess.run(
        [sys.executable, str(script), json.dumps(unit), str(pooled), str(manifest), str(bound),
         str(out), str(REPO)],
        capture_output=True, text=True)
    record = json.loads(out.read_text()) if out.is_file() else None
    return proc.returncode, proc.stdout + proc.stderr, record


UNIT_0 = {"unit": "episode_000000__train-01-oak-tungsten__r00", "episode": "episode_000000",
          "frames": 590, "style": "train-01-oak-tungsten", "repeat": 0, "seed": 7001, "stage": "1"}


def _shard_dir(tmp_path: pathlib.Path, manifest: pathlib.Path,
               per_episode: list[dict]) -> pathlib.Path:
    """The SHARDS, which since 2026-08-28 are what the screen reads by default.

    Job 190981 asked for an H200 and died in six seconds because the pooled file the screen used to
    require is on exactly one workstation and no committed script writes it. The shards are on the
    cluster, so the screen accepts a directory of them; this builds the smallest one that is a whole
    partition, with each episode placed in the shard it actually hashes to.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import pool_robot_mask_area as prma

    # Two, so the partition is a partition: with one shard every episode trivially hashes to it and
    # the placement check below would never be exercised.
    num_shards = 2
    directory = tmp_path / "shards"
    directory.mkdir()
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    keys = [e["episode"] for e in per_episode]
    for index in range(num_shards):
        mine = [e for e in per_episode if prma._shard_of(e["episode"], num_shards) == index]
        (directory / f"shard-{index}.json").write_text(json.dumps({
            "schema": "wam.robot_mask_area_shard/1",
            "shard": {"index": index, "num_shards": num_shards,
                      "corpus_episode_keys_sha256": "ab" * 32},
            "corpus_episode_keys": keys,
            "per_episode": [dict(e, episode_index=keys.index(e["episode"])) for e in mine],
            "measured": {"stride": 1},
            "measurement_qualified": True,
            "measurement_disqualified_reasons": [],
            "prompt": "robot arm. robotic hand. robotic gripper.",
            "estimator": {"adapter": "estimators.apple_sam2"},
            "git_commit": "deadbeef",
            "source_manifest": str(manifest),
            "source_manifest_sha256": sha,
        }), encoding="utf-8")
    return directory


def test_the_screen_admits_an_episode_from_a_shard_DIRECTORY(tmp_path) -> None:
    """The regression job 190981 was. The screen must reach the same verdict from the shards as
    from a pooled file, and must record WHICH shards said so — a directory has no bytes to hash, so
    an artifact naming one without its parts would cite evidence nobody can pin down."""
    episodes = [_episode("episode_000115", [0.10, 0.22, 0.31, 0.19]),
                _episode("episode_000116", [0.05, 0.07, 0.09, 0.11])]
    manifest = _manifest(tmp_path, {"episode_000115": 4, "episode_000116": 4})
    shards = _shard_dir(tmp_path, manifest, episodes)
    unit = dict(UNIT_0, unit="episode_000115__s__r00", episode="episode_000115", frames=4)

    rc, out, record = _precondition(tmp_path, unit=unit, pooled=shards, manifest=manifest,
                                    bound=_bound(tmp_path))
    assert rc == 0, out
    assert record is not None
    assert record["max_area_fraction"] == pytest.approx(0.31)
    assert record["evidence"] == str(shards)
    assert record["evidence_shards"] is not None
    assert len(record["evidence_shards"]) == 2
    assert all(len(part["sha256"]) == 64 for part in record["evidence_shards"])
    # The digest is over the parts, so it moves when any shard does and is not a hash of a path.
    assert record["evidence_sha256"] == hashlib.sha256(
        "".join(part["sha256"] for part in record["evidence_shards"]).encode()
    ).hexdigest()


def test_the_screen_refuses_a_shard_directory_that_is_not_the_whole_partition(tmp_path) -> None:
    """A pool over part of the corpus is stamped measurement_qualified false, and the screen already
    refuses that stamp by name. What must not happen is admitting on it."""
    episodes = [_episode("episode_000115", [0.10, 0.22, 0.31, 0.19]),
                _episode("episode_000116", [0.05, 0.07, 0.09, 0.11])]
    manifest = _manifest(tmp_path, {"episode_000115": 4, "episode_000116": 4})
    shards = _shard_dir(tmp_path, manifest, episodes)
    survivor = next(p for p in sorted(shards.glob("shard-*.json")))
    for other in sorted(shards.glob("shard-*.json")):
        if other != survivor:
            other.unlink()
    unit = dict(UNIT_0, unit="episode_000115__s__r00", episode="episode_000115", frames=4)
    rc, out, _ = _precondition(tmp_path, unit=unit, pooled=shards, manifest=manifest,
                               bound=_bound(tmp_path))
    assert rc != 0, out
    assert "measurement_qualified" in out


# ------------------------------------------------------------------------------------------------
# The defect itself: the choice is made before the GPU, or it is not a choice, it is a bill.
# ------------------------------------------------------------------------------------------------
def test_the_timed_unit_is_screened_before_the_driver_is_invoked() -> None:
    """Order is the whole property. Screening after the driver call would cost what it saves."""
    text = SBATCH_97.read_text(encoding="utf-8")
    timing = text.index("if (( TIMING_MODE )); then")
    driver = text.index('python "${RESTYLE_DRIVER}"', timing)
    assert PRECONDITION_ANCHOR in text[timing:driver], (
        "97's TIMING branch invokes the restyle driver without first screening the timed unit "
        "against the committed source-mask evidence. The unit `head -1` picks is refused by G0c on "
        "every submission, so the run costs an H200 slot and produces no THROUGHPUT.json.")


def test_refuses_the_head_of_the_work_list_when_its_source_masks_are_empty(tmp_path) -> None:
    manifest = _manifest(tmp_path, {"episode_000000": 4})
    pooled = _pooled(tmp_path, manifest, [_episode("episode_000000", [0.0, 0.0, 0.11, 0.12])])
    rc, out, record = _precondition(tmp_path, unit=dict(UNIT_0, frames=4), pooled=pooled,
                                    manifest=manifest, bound=_bound(tmp_path))
    assert rc != 0, out
    assert "episode_000000" in out
    assert "EMPTY on 2 of its 4 source frames, first at frame 0" in out, out
    assert record is None, "an admissibility record must not be written for a refused unit"


@pytest.mark.skipif(not POOLED.is_file(), reason="runs/ is gitignored; pooled measurement absent")
def test_the_real_corpus_refuses_the_real_head_of_the_work_list(tmp_path) -> None:
    """The consequence, on the artifact the cluster actually reads: 254 of 590 frames are empty."""
    pooled = json.loads(POOLED.read_text())
    entry = {e["episode"]: e for e in pooled["per_episode"]}["episode_000000"]
    assert entry["empty_frames"] == 254 and entry["n_frames"] == 590, entry
    manifest = _manifest(tmp_path, {"episode_000000": entry["n_frames"]})
    local = _pooled(tmp_path, manifest, [entry])
    rc, out, record = _precondition(tmp_path, unit=UNIT_0, pooled=local, manifest=manifest,
                                    bound=BOUND_ARTIFACT)
    assert rc != 0, out
    assert "254" in out, out
    assert record is None


def test_refuses_an_episode_whose_mask_exceeds_the_committed_bound(tmp_path) -> None:
    manifest = _manifest(tmp_path, {"episode_000042": 3})
    pooled = _pooled(tmp_path, manifest, [_episode("episode_000042", [0.10, 0.98, 0.11])])
    unit = dict(UNIT_0, unit="episode_000042__s__r00", episode="episode_000042", frames=3)
    rc, out, record = _precondition(tmp_path, unit=unit, pooled=pooled, manifest=manifest,
                                    bound=_bound(tmp_path))
    assert rc != 0, out
    assert "0.98" in out and "0.6409" in out, out
    assert record is None


def test_refuses_when_the_evidence_measured_a_different_corpus(tmp_path) -> None:
    """A bound and a distribution are claims ABOUT a corpus; over another one they say nothing."""
    manifest = _manifest(tmp_path, {"episode_000007": 2})
    pooled = _pooled(tmp_path, manifest, [_episode("episode_000007", [0.1, 0.1])],
                     source_manifest_sha256="0" * 64)
    unit = dict(UNIT_0, episode="episode_000007", frames=2)
    rc, out, _ = _precondition(tmp_path, unit=unit, pooled=pooled, manifest=manifest,
                               bound=_bound(tmp_path))
    assert rc != 0, out
    assert "source_manifest_sha256" in out, out


def test_refuses_when_the_episode_is_not_in_the_evidence(tmp_path) -> None:
    manifest = _manifest(tmp_path, {"episode_000009": 2})
    pooled = _pooled(tmp_path, manifest, [_episode("episode_000008", [0.1, 0.1])])
    unit = dict(UNIT_0, episode="episode_000009", frames=2)
    rc, out, _ = _precondition(tmp_path, unit=unit, pooled=pooled, manifest=manifest,
                               bound=_bound(tmp_path))
    assert rc != 0, out
    assert "episode_000009" in out


def test_refuses_when_the_evidence_is_absent_or_unqualified(tmp_path) -> None:
    """Fails CLOSED: an unprovable unit is not timed, because proving it costs nothing here."""
    manifest = _manifest(tmp_path, {"episode_000000": 2})
    rc, out, _ = _precondition(tmp_path, unit=dict(UNIT_0, frames=2), pooled=tmp_path / "nope.json",
                               manifest=manifest, bound=_bound(tmp_path))
    assert rc != 0 and "nope.json" in out, out

    pooled = _pooled(tmp_path, manifest, [_episode("episode_000000", [0.1, 0.1])],
                     measurement_qualified=False)
    rc, out, _ = _precondition(tmp_path, unit=dict(UNIT_0, frames=2), pooled=pooled,
                               manifest=manifest, bound=_bound(tmp_path))
    assert rc != 0 and "measurement_qualified" in out, out


def test_refuses_when_the_area_bound_has_not_been_decided(tmp_path) -> None:
    """`max_frame_fraction: null` is the artifact's own "no bound chosen yet". Half a check is none."""
    manifest = _manifest(tmp_path, {"episode_000000": 2})
    pooled = _pooled(tmp_path, manifest, [_episode("episode_000000", [0.1, 0.1])])
    rc, out, _ = _precondition(tmp_path, unit=dict(UNIT_0, frames=2), pooled=pooled,
                               manifest=manifest, bound=_bound(tmp_path, None))
    assert rc != 0 and "max_frame_fraction" in out, out


def test_refuses_when_the_evidence_counts_different_frames_than_the_manifest(tmp_path) -> None:
    manifest = _manifest(tmp_path, {"episode_000000": 590})
    pooled = _pooled(tmp_path, manifest, [_episode("episode_000000", [0.1, 0.1])])
    rc, out, _ = _precondition(tmp_path, unit=UNIT_0, pooled=pooled, manifest=manifest,
                               bound=_bound(tmp_path))
    assert rc != 0, out
    assert "590" in out and "2" in out


def test_admits_a_clean_episode_and_records_what_made_it_admissible(tmp_path) -> None:
    manifest = _manifest(tmp_path, {"episode_000115": 4})
    pooled = _pooled(tmp_path, manifest, [_episode("episode_000115", [0.10, 0.22, 0.31, 0.19])])
    unit = dict(UNIT_0, unit="episode_000115__s__r00", episode="episode_000115", frames=4)
    rc, out, record = _precondition(tmp_path, unit=unit, pooled=pooled, manifest=manifest,
                                    bound=_bound(tmp_path))
    assert rc == 0, out
    assert record is not None
    assert record["episode"] == "episode_000115"
    assert record["empty_frames"] == 0
    assert record["max_area_fraction"] == pytest.approx(0.31)
    assert record["max_frame_fraction_bound"] == pytest.approx(0.64091145833333329)
    assert record["evidence_sha256"] == hashlib.sha256(pooled.read_bytes()).hexdigest()
    assert "head -1" in record["selection_rule"], (
        "the timed unit's selection rule has to be IN the record: a throughput number whose unit "
        "nobody can reproduce the choice of is not a measurement of anything")


# ------------------------------------------------------------------------------------------------
# The other half: the artifact has to name the unit and its admissibility, or the number floats.
# ------------------------------------------------------------------------------------------------
def _throughput(tmp_path: pathlib.Path, *, admissibility: pathlib.Path | str,
                corpus_frames: str = "171625") -> tuple[int, str, dict | None]:
    script = tmp_path / "throughput_from_97.py"
    script.write_text(_heredoc(THROUGHPUT_ANCHOR), encoding="utf-8")
    troot = tmp_path / "timing_raw" / "episode_000115__s__r00"
    troot.mkdir(parents=True, exist_ok=True)
    (troot / "sample_outputs.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    out = tmp_path / "THROUGHPUT.json"
    unit = json.dumps(dict(UNIT_0, unit="episode_000115__s__r00", episode="episode_000115",
                           frames=100))
    proc = subprocess.run(
        [sys.executable, str(script), unit, "200", str(out), "nvidia/cosmos-transfer2.5",
         "rev0", "", "depth:0.5,seg:0.5", corpus_frames, str(troot.parent), str(admissibility)],
        capture_output=True, text=True)
    payload = json.loads(out.read_text()) if out.is_file() else None
    return proc.returncode, proc.stdout + proc.stderr, payload


def test_the_throughput_artifact_carries_the_timed_units_admissibility(tmp_path) -> None:
    adm = tmp_path / "adm.json"
    adm.write_text(json.dumps({
        "episode": "episode_000115", "n_frames": 100, "empty_frames": 0,
        "max_area_fraction": 0.31, "max_frame_fraction_bound": 0.64091145833333329,
        "evidence": "/valhalla/runs/pr08-robot-mask-area/POOLED.json", "evidence_sha256": "ab" * 32,
        "selection_rule": "head -1 of the deterministically sorted work list",
    }), encoding="utf-8")
    rc, out, payload = _throughput(tmp_path, admissibility=adm)
    assert rc == 0, out
    assert payload is not None
    assert payload["episode"] == "episode_000115"
    assert payload["timed_unit"] == "episode_000115__s__r00"
    assert payload["timed_unit_admissibility"]["evidence_sha256"] == "ab" * 32
    assert payload["timed_unit_admissibility"]["selection_rule"].startswith("head -1")
    assert payload["frames_per_variant"] == 171625


def test_a_throughput_artifact_without_the_screens_record_says_so_in_its_own_fields(tmp_path):
    """Not a refusal, and the asymmetry is the point.

    By the time this block runs the GPU-hours are spent, so refusing here would discard a paid
    measurement over a missing note about it — the failure this whole cluster of defects is. The
    refusal belongs to the screen, which runs before the slot is asked for. What is required here
    is that the artifact cannot be read as screened when it was not.
    """
    rc, out, payload = _throughput(tmp_path, admissibility=tmp_path / "missing.json")
    assert rc == 0, out
    assert payload is not None
    assert payload["timed_unit_admissibility"] is None
    assert "NOT RECORDED" in payload["timed_unit_admissibility_note"]


def test_frames_per_variant_is_read_from_the_partition_facts_not_coined(tmp_path) -> None:
    """Defect 12's regression guard: 172_000 was a typed number where a counted one existed.

    The literal is still allowed to appear in the prose that explains what it was; what may never
    come back is a FRAMES_PER_VARIANT that is anything other than the counted corpus.
    """
    body = _heredoc(THROUGHPUT_ANCHOR)
    assert "FRAMES_PER_VARIANT = int(corpus_frames)" in body, body
    adm = tmp_path / "adm.json"
    adm.write_text(json.dumps({"episode": "episode_000115", "selection_rule": "head -1 x"}),
                   encoding="utf-8")
    rc, out, payload = _throughput(tmp_path, admissibility=adm, corpus_frames="0")
    assert rc != 0 and "corpus_frames" in out, out
