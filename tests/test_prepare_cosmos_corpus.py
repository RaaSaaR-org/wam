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


@pytest.mark.parametrize("version", ["v1.6", "v4.0", "unknown"])
def test_refuses_any_codebase_version_it_cannot_read(tmp_path, version):
    """Two layouts are supported; a third is refused by name rather than guessed at.

    Jobs 186353 and 186354 died on a FileNotFoundError for episodes.jsonl, which reads like a
    broken download and cost real time pointed at the fetch step. Guessing the layout from what
    files happen to be present is worse still: it is how a corpus silently becomes one episode
    repeated.
    """
    root = _retag(make_root(tmp_path, "src", [200, 200]), version)
    with pytest.raises(SystemExit) as e:
        pcc.main(["--source", str(root), "--out", str(tmp_path / "o")])
    msg = str(e.value)
    assert version in msg and "v2.1" in msg and "v3.0" in msg
    # Naming the version is not enough — it must say what is different, or the reader is left
    # thinking a re-download will fix it.
    assert "timestamp" in msg


# --- 5. materialization: the clips that actually reach the decoder ---------------------------
#
# These are the only tests here that touch real video. Everything above works on byte blobs
# because it is testing bookkeeping; transcoding is the one step whose output is judged by a
# decoder, and a blob cannot fail the way job 186357 failed.

import shutil  # noqa: E402
import subprocess  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


def write_real_video(path: Path, frames: int, fps: float = 30.0, codec: str = "libx264") -> Path:
    """A genuinely decodable mp4 of exactly ``frames`` frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=size=64x48:rate={fps:g}",
         "-frames:v", str(frames), "-c:v", codec, "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


def real_root(tmp: Path, name: str, lengths: list[int], fps: float = 30.0) -> Path:
    root = make_root(tmp, name, lengths, fps=fps, write_videos=False)
    for i, n in enumerate(lengths):
        write_real_video(pcc.episode_video_path(root, "observation.images.ego", i, 1000), n, fps)
    return root


def probe(path: Path, entry: str) -> str:
    return subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", f"stream={entry}", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


@needs_ffmpeg
def test_transcode_writes_real_h264_files_with_the_promised_frame_count(tmp_path):
    """The whole point of --transcode: a file the consumer's decoder can read.

    LeRobot writes AV1 by default. vLLM's OpenCV opened those files, reported the right frame
    count from the container header, and then decoded zero frames — 372 clips captioned, 0
    captions produced, one GPU hour spent. H.264 yuv420p is the format nothing argues with.
    """
    root = real_root(tmp_path, "src", [80, 90, 100])
    out = tmp_path / "out"
    pcc.main(["--source", str(root), "--out", str(out), "--val-episodes", "1",
              "--seed", "0", "--mode", "transcode"])
    clips = sorted((out / "train" / "videos").glob("*.mp4"))
    assert clips, "transcode produced no clips"
    for p in clips:
        assert not p.is_symlink()
        assert probe(p, "codec_name") == "h264"
        assert int(probe(p, "nb_frames")) in (80, 90, 100)


@needs_ffmpeg
def test_transcode_records_the_source_hash_and_the_trained_hash_separately(tmp_path):
    """Re-encoding breaks the identity between "what we downloaded" and "what trains".

    AC-04 needs both: src_sha256 answers "which upstream bytes", sha256 answers "which pixels did
    the run actually see". Collapsing them to one field means a corpus re-encoded with different
    settings is indistinguishable from the original.
    """
    root = real_root(tmp_path, "src", [80] * 3)
    out = tmp_path / "out"
    pcc.main(["--source", str(root), "--out", str(out), "--val-episodes", "1",
              "--seed", "0", "--mode", "transcode"])
    manifest = json.loads((out / "manifest.json").read_text())
    for clip in manifest["clips"]["train"]:
        assert clip["src_sha256"] and clip["sha256"]
        assert clip["src_sha256"] != clip["sha256"], "re-encoded output cannot hash to its source"
    assert manifest["materialization"]["mode"] == "transcode"
    assert manifest["materialization"]["encoder"] == "libx264"


def test_link_mode_leaves_the_two_hashes_equal(tmp_path):
    """A symlinked clip IS its source, and the manifest should say so rather than imply a copy."""
    root = make_root(tmp_path, "src", [100] * 3)
    out = tmp_path / "out"
    pcc.main(["--source", str(root), "--out", str(out), "--val-episodes", "1", "--seed", "0"])
    manifest = json.loads((out / "manifest.json").read_text())
    for clip in manifest["clips"]["train"]:
        assert clip["src_sha256"] == clip["sha256"]
    assert manifest["materialization"]["mode"] == "link"
    assert manifest["materialization"]["encoder"] is None


@needs_ffmpeg
def test_transcode_cuts_the_requested_window_out_of_a_shared_file(tmp_path):
    """v3.0's concatenated layout in miniature: take one second out of the middle of a file.

    This is the operation the 13 Unitree sets need and v2.1 never did. If the window is ignored,
    every clip is the whole concatenated file and the corpus is silently one episode repeated.
    """
    src = write_real_video(tmp_path / "shared.mp4", 300)  # 10 s @ 30 fps
    clip = pcc.Clip(uuid="w", source_id="s", episode_index=0, src_path=str(src),
                    frames=30, fps=30.0, duration_s=1.0, width=64, height=48, task="t",
                    from_s=4.0, to_s=5.0)
    got = pcc.materialize(clip, tmp_path / "d", "transcode", "libx264", "ffmpeg", "ffprobe", 2)
    assert abs(int(probe(tmp_path / "d" / "w.mp4", "nb_frames")) - 30) <= 2
    assert got.sha256 and got.src_sha256 != got.sha256


@needs_ffmpeg
def test_a_cut_that_lands_short_is_rejected_rather_than_written(tmp_path):
    """The failure that would otherwise be invisible.

    A truncated clip still plays, still captions, still trains. Nothing downstream can tell that
    the episode it was promised is not the episode it got — so the check has to be here. The
    realistic cause is a window that runs past the end of its file: metadata that disagrees with
    the mp4 it points at, which is exactly what a partial download produces.
    """
    src = write_real_video(tmp_path / "shared.mp4", 60)
    clip = pcc.Clip(uuid="bad", source_id="s", episode_index=0, src_path=str(src),
                    frames=300, fps=30.0, duration_s=10.0, width=64, height=48, task="t",
                    from_s=0.0, to_s=10.0)
    with pytest.raises(pcc.TranscodeError) as e:
        pcc.materialize(clip, tmp_path / "d", "transcode", "libx264", "ffmpeg", "ffprobe", 2)
    assert "300" in str(e.value)


@needs_ffmpeg
def test_the_window_lands_on_the_right_episode_not_merely_the_right_length(tmp_path):
    """Frame count proves the clip is the right SIZE. This proves it is the right PLACE.

    A cut that is the correct length but starts at the wrong offset is the worst outcome available
    here: every count reconciles, the manifest is consistent, the corpus trains — and the clip
    labelled "episode 12" contains episode 11. The two solid colours make the distinction
    mechanical: if the window is honoured the frame is blue, if the seek is dropped it is red.
    """
    def solid(name: str, colour: str, seconds: int) -> Path:
        p = tmp_path / name
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", f"color=c={colour}:size=64x48:rate=30:duration={seconds}",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)], check=True, capture_output=True)
        return p

    red, blue = solid("red.mp4", "red", 5), solid("blue.mp4", "blue", 5)
    listing = tmp_path / "list.txt"
    listing.write_text(f"file '{red}'\nfile '{blue}'\n")
    shared = tmp_path / "shared.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "concat",
                    "-safe", "0", "-i", str(listing), "-c", "copy", str(shared)],
                   check=True, capture_output=True)

    # Episode 1 of this "dataset" is the blue half: [5 s, 10 s).
    clip = pcc.Clip(uuid="ep1", source_id="s", episode_index=1, src_path=str(shared),
                    frames=90, fps=30.0, duration_s=3.0, width=64, height=48, task="t",
                    from_s=5.0, to_s=8.0)
    pcc.materialize(clip, tmp_path / "d", "transcode", "libx264", "ffmpeg", "ffprobe", 2)

    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(tmp_path / "d" / "ep1.mp4"),
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True).stdout
    r, g, b = raw[0], raw[1], raw[2]
    assert b > 128 and r < 96, f"expected the blue half, got rgb({r},{g},{b}) — the seek was lost"


def test_a_windowed_clip_refuses_to_be_symlinked(tmp_path):
    """Linking a window would publish the whole shared file under one episode's name.

    Silent and catastrophic: every clip in the split becomes the same multi-episode video, the
    counts still add up, and the corpus looks finished.
    """
    src = tmp_path / "shared.mp4"
    src.write_bytes(b"\x00mp4")
    clip = pcc.Clip(uuid="w", source_id="s", episode_index=0, src_path=str(src),
                    frames=30, fps=30.0, duration_s=1.0, width=64, height=48, task="t",
                    from_s=4.0, to_s=5.0)
    for mode in ("link", "copy"):
        with pytest.raises(pcc.TranscodeError) as e:
            pcc.materialize(clip, tmp_path / f"d-{mode}", mode, "libx264", "ffmpeg", "ffprobe", 2)
        assert "--transcode" in str(e.value)


# --- 6. LeRobot v3.0: a clip is a window, not a file -----------------------------------------
#
# 13 of the 14 pre-registered sources are v3.0. Episodes are concatenated into a handful of mp4s
# and each episode is a [from_timestamp, to_timestamp) window inside one of them. Every bug in
# this reader produces a corpus that still trains: a dropped seek gives every clip the same
# opening frames, a per-episode file lookup reads the wrong camera's file, an inclusive end
# appends a frame of the next episode to all of them.

pa = pytest.importorskip("pyarrow", reason="v3.0 boundaries are parquet")
import pyarrow.parquet as pq  # noqa: E402


def make_root_v30(
    tmp: Path,
    name: str,
    lengths: list[int],
    fps: float = 30.0,
    cameras: tuple[str, ...] = ("observation.images.cam_left_high",),
    file_index_of: dict[str, list[int]] | None = None,
    write_videos: bool = True,
) -> Path:
    """A minimal but faithful v3.0 root.

    ``file_index_of`` maps camera -> per-episode mp4 index, which is how the real datasets behave:
    each camera rolls over to a new file on its own byte budget, so the same episode lives in
    file-000 for one camera and file-001 for another.
    """
    root = tmp / name
    (root / "meta").mkdir(parents=True)
    features = {
        cam: {"dtype": "video", "shape": [48, 64, 3],
              "info": {"video.fps": fps, "video.codec": "av1", "video.width": 64,
                       "video.height": 48}}
        for cam in cameras
    }
    features["observation.state"] = {"dtype": "float32", "shape": [43]}
    video_path = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "fps": fps, "chunks_size": 1000,
        "video_path": video_path, "features": features,
    }))

    file_index_of = file_index_of or {cam: [0] * len(lengths) for cam in cameras}
    columns: dict[str, list] = {
        "episode_index": list(range(len(lengths))),
        "length": list(lengths),
        "tasks": [["do the thing"] for _ in lengths],
    }
    # Timestamps accumulate within one mp4 and reset to 0.0 whenever that camera rolls over —
    # the behaviour the reader depends on for the windows to mean anything.
    for cam in cameras:
        files = file_index_of[cam]
        froms, tos, cursor, current = [], [], 0.0, files[0]
        for n, f in zip(lengths, files):
            if f != current:
                cursor, current = 0.0, f
            froms.append(cursor)
            cursor += n / fps
            tos.append(cursor)
        columns[f"videos/{cam}/chunk_index"] = [0] * len(lengths)
        columns[f"videos/{cam}/file_index"] = list(files)
        columns[f"videos/{cam}/from_timestamp"] = froms
        columns[f"videos/{cam}/to_timestamp"] = tos

    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.table(columns), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    if write_videos:
        for cam in cameras:
            per_file: dict[int, int] = {}
            for n, f in zip(lengths, file_index_of[cam]):
                per_file[f] = per_file.get(f, 0) + n
            for f, total in per_file.items():
                dest = root / video_path.format(video_key=cam, chunk_index=0, file_index=f)
                if shutil.which("ffmpeg"):
                    write_real_video(dest, total, fps)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(b"\x00mp4")
    return root


def test_v30_turns_each_episode_into_a_window_of_the_shared_file(tmp_path):
    root = make_root_v30(tmp_path, "u", [90, 120, 150], write_videos=False)
    (root / "videos" / "observation.images.cam_left_high" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "observation.images.cam_left_high" / "chunk-000" / "file-000.mp4").write_bytes(b"\x00mp4")

    clips, dropped = pcc.scan_source(root, "u", None, pcc.MIN_WINDOW_FRAMES)
    assert [c.frames for c in clips] == [90, 120, 150]
    assert dropped["missing_video"] == 0
    # Contiguous, cumulative, and in the coordinates of the file they point at.
    assert [c.from_s for c in clips] == pytest.approx([0.0, 3.0, 7.0])
    assert [c.to_s for c in clips] == pytest.approx([3.0, 7.0, 12.0])
    assert all(c.src_path.endswith("file-000.mp4") for c in clips)
    assert all(c.src_codec == "av1" for c in clips)


def test_v30_resolves_the_video_file_per_camera_not_once_per_episode(tmp_path):
    """The gotcha that silently reads the wrong file for most of a corpus.

    Cameras roll over independently: in the real G1_Dex3_BlockStacking set, episode 50 is in
    file-001 for cam_left_high while the other three cameras are still on file-000. Resolving the
    file once per episode and reusing it across keys reads a real, valid, wrong mp4.
    """
    cams = ("observation.images.cam_left_high", "observation.images.cam_right_high")
    root = make_root_v30(
        tmp_path, "u", [90, 90, 90], cameras=cams, write_videos=False,
        # left rolls over at episode 1, right stays in one file throughout
        file_index_of={cams[0]: [0, 1, 1], cams[1]: [0, 0, 0]},
    )
    for cam, files in ((cams[0], (0, 1)), (cams[1], (0,))):
        d = root / "videos" / cam / "chunk-000"
        d.mkdir(parents=True)
        for f in files:
            (d / f"file-{f:03d}.mp4").write_bytes(b"\x00mp4")

    left, _ = pcc.scan_source(root, "u", cams[0], pcc.MIN_WINDOW_FRAMES)
    right, _ = pcc.scan_source(root, "u", cams[1], pcc.MIN_WINDOW_FRAMES)
    assert [Path(c.src_path).name for c in left] == ["file-000.mp4", "file-001.mp4", "file-001.mp4"]
    assert [Path(c.src_path).name for c in right] == ["file-000.mp4"] * 3
    # The rollover resets the clock; episode 1 starts at 0.0 in its new file, not at 3.0.
    assert [c.from_s for c in left] == pytest.approx([0.0, 0.0, 3.0])
    assert [c.from_s for c in right] == pytest.approx([0.0, 3.0, 6.0])


def test_v30_names_the_camera_when_its_columns_are_absent(tmp_path):
    """A wrong camera key looks exactly like a corrupt parquet unless the error says otherwise."""
    root = make_root_v30(tmp_path, "u", [90, 90], write_videos=False)
    with pytest.raises(SystemExit) as e:
        pcc.scan_source(root, "u", "observation.images.nope", pcc.MIN_WINDOW_FRAMES)
    assert "nope" in str(e.value)


def test_v30_without_the_episodes_parquet_says_what_is_missing(tmp_path):
    root = make_root_v30(tmp_path, "u", [90], write_videos=False)
    shutil.rmtree(root / "meta" / "episodes")
    with pytest.raises(SystemExit) as e:
        pcc.scan_source(root, "u", None, pcc.MIN_WINDOW_FRAMES)
    assert "meta/episodes" in str(e.value)


@needs_ffmpeg
def test_v30_end_to_end_produces_one_real_clip_per_episode(tmp_path):
    """The whole v3.0 path: concatenated AV1-shaped source in, one H.264 file per episode out."""
    root = make_root_v30(tmp_path, "u", [90, 120, 150])
    out = tmp_path / "out"
    pcc.main(["--source", str(root), "--out", str(out), "--val-episodes", "1",
              "--seed", "0", "--mode", "transcode"])
    produced = sorted((out / "train" / "videos").glob("*.mp4")) + \
        sorted((out / "val" / "videos").glob("*.mp4"))
    assert len(produced) == 3
    counts = sorted(int(probe(p, "nb_frames")) for p in produced)
    assert counts == [90, 120, 150]
    assert all(probe(p, "codec_name") == "h264" for p in produced)


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
