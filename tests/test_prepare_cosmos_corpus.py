"""Tests for ``scripts/prepare_cosmos_corpus.py`` (PR-09 §8 item 1).

The script does three things a copy does not, and each is a place where a bug is silent rather
than loud — the run would still start, still log a loss, and still produce a checkpoint:

1. **The filters.** cosmos-framework's loader drops clips > 61 s and windows < 61 frames without
   saying so. If we do not mirror them, the manifest reports a corpus larger than the one that
   trained, and every per-episode claim afterwards is off by the difference.
2. **The split.** PR-09 §5 requires the eval prompts to come from episodes SFT never saw. A split
   that depends on directory order is not the seeded split the manifest claims, and "held out"
   becomes unfalsifiable.
3. **Camera resolution.** A source with two cameras trained on the wrong view produces a finite,
   plausible, wrong run. The script must refuse rather than pick.

These build synthetic LeRobot v2.1 roots — no dataset needed, so they run in CI and on a fresh
clone. The mp4s are byte blobs; nothing here decodes video.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import prepare_cosmos_corpus as pcc  # noqa: E402


def make_root(
    tmp: Path,
    name: str,
    lengths: list[int],
    fps: float = 30.0,
    cameras: tuple[str, ...] = ("observation.images.ego",),
    chunk_size: int = 1000,
    write_videos: bool = True,
) -> Path:
    root = tmp / name
    (root / "meta").mkdir(parents=True)
    features = {
        cam: {"dtype": "video", "shape": [480, 640, 3], "info": {"video.fps": fps}}
        for cam in cameras
    }
    features["observation.state"] = {"dtype": "float32", "shape": [43]}
    (root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v2.1", "fps": fps, "chunks_size": chunk_size,
                    "features": features})
    )
    with (root / "meta" / "episodes.jsonl").open("w") as fh:
        for i, n in enumerate(lengths):
            fh.write(json.dumps({"episode_index": i, "length": n, "tasks": ["do the thing"]}) + "\n")
    if write_videos:
        for cam in cameras:
            for i in range(len(lengths)):
                p = pcc.episode_video_path(root, cam, i, chunk_size)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"\x00mp4" + bytes([i % 251]) * 32)
    return root


# --- 1. the filters ------------------------------------------------------------------------


def test_drops_clips_below_the_framework_frame_floor(tmp_path):
    root = make_root(tmp_path, "src", [60, 61, 200])
    clips, dropped = pcc.scan_source(root, "src", None, pcc.MIN_WINDOW_FRAMES)
    assert [c.frames for c in clips] == [61, 200]
    assert dropped["too_short"] == 1


def test_drops_clips_over_61_seconds(tmp_path):
    # 1830 frames @30 fps = 61.0 s exactly (kept); 1831 is over (dropped).
    root = make_root(tmp_path, "src", [1830, 1831])
    clips, dropped = pcc.scan_source(root, "src", None, pcc.MIN_WINDOW_FRAMES)
    assert [c.frames for c in clips] == [1830]
    assert dropped["too_long"] == 1


def test_num_video_frames_raises_the_floor_but_never_lowers_it(tmp_path):
    root = make_root(tmp_path, "src", [61, 100, 200])
    clips, _ = pcc.scan_source(root, "src", None, max(pcc.MIN_WINDOW_FRAMES, 121))
    assert [c.frames for c in clips] == [200]
    # A recipe asking for fewer frames must not defeat the loader's own floor.
    clips, _ = pcc.scan_source(root, "src", None, max(pcc.MIN_WINDOW_FRAMES, 10))
    assert [c.frames for c in clips] == [61, 100, 200]


def test_missing_video_is_counted_not_crashed(tmp_path):
    root = make_root(tmp_path, "src", [100, 100], write_videos=True)
    pcc.episode_video_path(root, "observation.images.ego", 1, 1000).unlink()
    clips, dropped = pcc.scan_source(root, "src", None, pcc.MIN_WINDOW_FRAMES)
    assert len(clips) == 1 and dropped["missing_video"] == 1


def test_manifest_counts_equal_what_was_placed(tmp_path):
    """The whole point of mirroring the filters: the manifest may not overstate the corpus."""
    root = make_root(tmp_path, "src", [30, 100, 120, 140, 2000, 160])
    out = tmp_path / "out"
    pcc.main(["--source", str(root), "--out", str(out), "--val-episodes", "1", "--seed", "0"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["counts"] == {"train": 3, "val": 1}
    assert len(list((out / "train" / "videos").glob("*.mp4"))) == 3
    assert len(list((out / "val" / "videos").glob("*.mp4"))) == 1
    assert manifest["sources"]["src"]["dropped"] == {
        "too_short": 1, "too_long": 1, "missing_video": 0
    }


# --- 2. the split --------------------------------------------------------------------------


def test_split_is_disjoint_and_covers_everything(tmp_path):
    root = make_root(tmp_path, "src", [100] * 20)
    out = tmp_path / "out"
    pcc.main(["--source", str(root), "--out", str(out), "--val-episodes", "5", "--seed", "0"])
    m = json.loads((out / "manifest.json").read_text())
    train = {c["uuid"] for c in m["clips"]["train"]}
    val = {c["uuid"] for c in m["clips"]["val"]}
    assert not (train & val)
    assert len(train) == 15 and len(val) == 5


def test_split_is_reproducible_across_runs_and_directory_order(tmp_path):
    """A split that moves when the filesystem returns a different order is not seeded."""
    a = make_root(tmp_path / "a", "src", [100] * 20)
    b = make_root(tmp_path / "b", "src", [100] * 20)
    outs = []
    for i, root in enumerate((a, b)):
        out = tmp_path / f"out{i}"
        pcc.main(["--source", str(root), "--out", str(out), "--val-episodes", "6", "--seed", "7"])
        m = json.loads((out / "manifest.json").read_text())
        outs.append(sorted(c["episode_index"] for c in m["clips"]["val"]))
    assert outs[0] == outs[1]


def test_different_seeds_give_different_holdouts(tmp_path):
    root = make_root(tmp_path, "src", [100] * 40)
    got = []
    for seed in (0, 1):
        out = tmp_path / f"o{seed}"
        pcc.main(["--source", str(root), "--out", str(out), "--val-episodes", "8",
                  "--seed", str(seed)])
        m = json.loads((out / "manifest.json").read_text())
        got.append(sorted(c["episode_index"] for c in m["clips"]["val"]))
    assert got[0] != got[1]


def test_refuses_a_holdout_that_swallows_the_corpus(tmp_path):
    root = make_root(tmp_path, "src", [100] * 5)
    with pytest.raises(SystemExit):
        pcc.main(["--source", str(root), "--out", str(tmp_path / "o"), "--val-episodes", "5"])


# --- 3. camera resolution and pooling ------------------------------------------------------


def test_refuses_to_guess_between_two_cameras(tmp_path):
    root = make_root(tmp_path, "src", [100], cameras=("cam.left", "cam.right"))
    with pytest.raises(SystemExit) as e:
        pcc.scan_source(root, "src", None, pcc.MIN_WINDOW_FRAMES)
    assert "--camera-key" in str(e.value)


def test_named_camera_is_honoured_and_an_unknown_one_is_fatal(tmp_path):
    root = make_root(tmp_path, "src", [100], cameras=("cam.left", "cam.right"))
    clips, _ = pcc.scan_source(root, "src", "cam.right", pcc.MIN_WINDOW_FRAMES)
    assert clips and "cam.right" in clips[0].src_path
    with pytest.raises(SystemExit):
        pcc.scan_source(root, "src", "cam.middle", pcc.MIN_WINDOW_FRAMES)


def test_pooled_sources_keep_distinct_uuids(tmp_path):
    """Two corpora both starting at episode_000000 must not collide in one flat videos/ dir."""
    a = make_root(tmp_path, "apple", [100] * 3)
    b = make_root(tmp_path, "bread", [100] * 3)
    out = tmp_path / "out"
    pcc.main(["--source", str(a), "--source", str(b), "--out", str(out),
              "--val-episodes", "1", "--seed", "0"])
    names = [p.name for p in (out / "train" / "videos").glob("*.mp4")]
    names += [p.name for p in (out / "val" / "videos").glob("*.mp4")]
    assert len(names) == len(set(names)) == 6
    assert any(n.startswith("apple_") for n in names)
    assert any(n.startswith("bread_") for n in names)


def test_per_source_camera_keys(tmp_path):
    """Pooled corpora do not agree on camera names; one global key would pick a wrong view."""
    a = make_root(tmp_path, "apple", [100] * 2, cameras=("obs.ego", "obs.wrist"))
    b = make_root(tmp_path, "bread", [100] * 2, cameras=("cam_high", "cam_wrist"))
    out = tmp_path / "out"
    pcc.main(["--source", str(a), "--camera-key", "obs.ego",
              "--source", str(b), "--camera-key", "cam_high",
              "--out", str(out), "--val-episodes", "1", "--seed", "0"])
    m = json.loads((out / "manifest.json").read_text())
    assert m["sources"]["apple"]["camera_key"] == "obs.ego"
    assert m["sources"]["bread"]["camera_key"] == "cam_high"
    for c in m["clips"]["train"] + m["clips"]["val"]:
        expected = "obs.ego" if c["source_id"] == "apple" else "cam_high"
        assert f"/{expected}/" in c["src_path"]


def test_camera_key_count_must_match_source_count(tmp_path):
    a = make_root(tmp_path, "a", [100], cameras=("x", "y"))
    b = make_root(tmp_path, "b", [100], cameras=("x", "y"))
    c = make_root(tmp_path, "c", [100], cameras=("x", "y"))
    with pytest.raises(SystemExit) as e:
        pcc.main(["--source", str(a), "--source", str(b), "--source", str(c),
                  "--camera-key", "x", "--camera-key", "y",
                  "--out", str(tmp_path / "o"), "--val-episodes", "1"])
    assert "one per source" in str(e.value)


def test_single_camera_key_applies_to_all_sources(tmp_path):
    a = make_root(tmp_path, "a", [100] * 2, cameras=("shared", "other"))
    b = make_root(tmp_path, "b", [100] * 2, cameras=("shared", "other"))
    out = tmp_path / "out"
    pcc.main(["--source", str(a), "--source", str(b), "--camera-key", "shared",
              "--out", str(out), "--val-episodes", "1", "--seed", "0"])
    m = json.loads((out / "manifest.json").read_text())
    assert {s["camera_key"] for s in m["sources"].values()} == {"shared"}


def test_chunked_layout_is_resolved(tmp_path):
    root = make_root(tmp_path, "src", [100] * 5, chunk_size=2)
    clips, dropped = pcc.scan_source(root, "src", None, pcc.MIN_WINDOW_FRAMES)
    assert len(clips) == 5 and dropped["missing_video"] == 0
    assert "chunk-002" in clips[4].src_path


# --- provenance ----------------------------------------------------------------------------


def test_manifest_hash_changes_when_a_clip_changes(tmp_path):
    """AC-04: a corpus that was silently rebuilt has to be detectable."""
    root = make_root(tmp_path, "src", [100] * 4)
    first = tmp_path / "o1"
    pcc.main(["--source", str(root), "--out", str(first), "--val-episodes", "1", "--seed", "0"])
    before = (first / "MANIFEST_SHA256").read_text()

    pcc.episode_video_path(root, "observation.images.ego", 0, 1000).write_bytes(b"\x00mp4CHANGED")
    second = tmp_path / "o2"
    pcc.main(["--source", str(root), "--out", str(second), "--val-episodes", "1", "--seed", "0"])
    assert (second / "MANIFEST_SHA256").read_text() != before


def test_copy_mode_produces_real_files_not_links(tmp_path):
    root = make_root(tmp_path, "src", [100] * 3)
    out = tmp_path / "out"
    pcc.main(["--source", str(root), "--out", str(out), "--val-episodes", "1",
              "--seed", "0", "--copy"])
    for p in (out / "train" / "videos").glob("*.mp4"):
        assert not p.is_symlink()


def test_dry_run_writes_nothing(tmp_path):
    root = make_root(tmp_path, "src", [100] * 3)
    out = tmp_path / "out"
    pcc.main(["--source", str(root), "--out", str(out), "--val-episodes", "1", "--dry-run"])
    assert not out.exists()


def test_not_a_lerobot_root_is_fatal(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit) as e:
        pcc.main(["--source", str(tmp_path / "empty"), "--out", str(tmp_path / "o")])
    assert "LeRobot" in str(e.value)


def _retag(root: Path, version: str) -> Path:
    """Rewrite only codebase_version, leaving the v2.1 layout in place.

    Deliberate: it reproduces what is NOT true of the real repos, so the test can prove the
    refusal comes from the declared version rather than from a file that happens to be missing.
    """
    p = root / "meta" / "info.json"
    info = json.loads(p.read_text())
    info["codebase_version"] = version
    p.write_text(json.dumps(info))
    return root


@pytest.mark.parametrize("version", ["v3.0", "v1.6", "unknown"])
def test_refuses_any_codebase_version_it_cannot_read(tmp_path, version):
    """All 13 unitreerobotics/G1_Dex3_* sets are v3.0, which this script cannot read.

    v3.0 concatenates episodes into a few large mp4s and moves the boundaries into
    meta/episodes/*/*.parquet, so a clip has to be CUT OUT by timestamp. Jobs 186353 and 186354
    died on a FileNotFoundError for episodes.jsonl, which reads like a broken download and cost
    real time pointed at the fetch step. The refusal has to name the format.
    """
    root = _retag(make_root(tmp_path, "src", [200, 200]), version)
    with pytest.raises(SystemExit) as e:
        pcc.main(["--source", str(root), "--out", str(tmp_path / "o")])
    msg = str(e.value)
    assert version in msg and "v2.1" in msg
    # Naming the version is not enough — it must say what is different, or the reader is left
    # thinking a re-download will fix it.
    assert "timestamp" in msg


def test_the_supported_version_is_not_merely_whatever_the_fixture_says(tmp_path):
    """Guard against the check passing because both sides read the same field.

    An untagged root is refused too, so `codebase_version` absent can never be mistaken for
    compatible — which is how a hand-assembled corpus would most plausibly slip through.
    """
    root = make_root(tmp_path, "src", [200, 200])
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["codebase_version"] == pcc._SUPPORTED_CODEBASE == "v2.1"
    del info["codebase_version"]
    (root / "meta" / "info.json").write_text(json.dumps(info))
    with pytest.raises(SystemExit) as e:
        pcc.main(["--source", str(root), "--out", str(tmp_path / "o")])
    assert "unknown" in str(e.value)
