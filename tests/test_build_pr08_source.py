"""Tests for ``scripts/build_pr08_source.py`` — the SOURCE corpus 97 reads.

The script's whole job is refusing to write a manifest that claims something the snapshot does not
support, so almost every test here is a refusal. The one that matters most is the frame-count
check: the actions are carried over by index, so a video whose length disagrees with
``episodes.jsonl`` pairs every later label with a different instant than its pixels — with no
decode error, and with the restyle looking perfect.

``ffprobe`` is injected as a path, so these run without ffmpeg: each test writes a stub that prints
the JSON a real ffprobe would. That is deliberate rather than convenient — it lets the disagreement
cases (metadata says one thing, pixels say another) be constructed at all, which a real encoder
would make tedious and slow.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

import build_pr08_source as bps  # noqa: E402


def _stub_ffprobe(tmp_path: pathlib.Path, streams: dict | list[dict], name: str = "ffprobe") -> str:
    """A fake ffprobe printing a fixed payload. One dict = same answer for every file."""
    payload = streams if isinstance(streams, list) else [streams]
    script = tmp_path / name
    script.write_text(f"#!/bin/sh\ncat <<'JSON'\n{json.dumps({'streams': payload})}\nJSON\n")
    script.chmod(0o755)
    return str(script)


def _snapshot(
    tmp_path: pathlib.Path,
    *,
    episodes: list[dict] | None = None,
    shape: list[int] | None = None,
    with_videos: bool = True,
    with_episodes_file: bool = True,
) -> pathlib.Path:
    root = tmp_path / "snap"
    (root / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "chunks_size": 1000,
        "total_episodes": 2,
        "total_frames": 15,
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            bps.VIDEO_KEY: {
                "dtype": "video",
                "shape": shape if shape is not None else [480, 640, 3],
                "names": ["height", "width", "channels"],
            }
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    eps = (
        episodes
        if episodes is not None
        else [
            {"episode_index": 0, "length": 10},
            {"episode_index": 1, "length": 5},
        ]
    )
    if with_episodes_file:
        (root / "meta" / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in eps) + "\n")
    if with_videos:
        vdir = root / "videos" / "chunk-000" / bps.VIDEO_KEY
        vdir.mkdir(parents=True)
        for e in eps:
            (vdir / f"episode_{int(e['episode_index']):06d}.mp4").write_bytes(b"\x00")
    return root


# ------------------------------------------------------------------------------------------
# the happy path
# ------------------------------------------------------------------------------------------


def test_builds_a_manifest_the_driver_contract_accepts(tmp_path: pathlib.Path) -> None:
    snap = _snapshot(tmp_path)
    probe = _stub_ffprobe(
        tmp_path,
        [{"width": 640, "height": 480, "codec_name": "av1", "nb_frames": "10"}],
    )
    out = tmp_path / "out"

    # Two episodes of different declared length against one stub answer would fail the frame check,
    # which is the point of the next test. Here both are 10.
    snap_eps = [{"episode_index": 0, "length": 10}, {"episode_index": 1, "length": 10}]
    (snap / "meta" / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in snap_eps) + "\n")

    manifest = bps.build(snap, out, ffprobe=probe, revision="deadbeef", repo_id="acme/corpus")

    assert manifest["resolution"] == [640, 480], "97 and the driver both hard-fail anything else"
    assert manifest["fps"] == 30
    assert [e["id"] for e in manifest["episodes"]] == ["episode_000000", "episode_000001"]
    assert [e["frames"] for e in manifest["episodes"]] == [10, 10]
    assert manifest["source"]["revision"] == "deadbeef"
    assert manifest["source"]["repo_id"] == "acme/corpus"
    assert manifest["source"]["codecs"] == ["av1"]
    for entry in manifest["episodes"]:
        assert (out / entry["video"]).exists(), f"{entry['video']} does not resolve"


def test_no_episode_claims_a_conditioning_map_it_does_not_have(tmp_path: pathlib.Path) -> None:
    """An absent map means 'let Transfer2.5 estimate it'; a NAMED missing map is fatal downstream.

    Until PR-08 §8 items 4 and 5 land there are no maps, so the honest manifest claims none.
    """
    snap = _snapshot(tmp_path, episodes=[{"episode_index": 0, "length": 10}])
    probe = _stub_ffprobe(
        tmp_path, {"width": 640, "height": 480, "codec_name": "av1", "nb_frames": "10"}
    )
    manifest = bps.build(snap, tmp_path / "out", ffprobe=probe)
    for entry in manifest["episodes"]:
        assert "depth" not in entry
        assert "segmentation" not in entry


def test_the_manifest_gets_a_sha256_sidecar(tmp_path: pathlib.Path) -> None:
    """The discipline pr08_style_partition.json uses: the file the gate read is provably the file
    that was built."""
    import hashlib

    out = tmp_path / "out"
    manifest = {"resolution": [640, 480], "episodes": [{"id": "e", "frames": 1, "video": "v"}]}
    bps.write_manifest(out, manifest)
    text = (out / "manifest.json").read_text()
    digest = (out / "manifest.json.sha256").read_text().split()[0]
    assert digest == hashlib.sha256(text.encode()).hexdigest()


# ------------------------------------------------------------------------------------------
# the refusals
# ------------------------------------------------------------------------------------------


def test_a_video_shorter_than_its_label_column_is_refused(tmp_path: pathlib.Path) -> None:
    """THE test. A silent desync of pixels from carried-over labels is the one defect no later
    gate catches: G0a checks the actions are unchanged (they are), G0b checks geometry within a
    clip (it is fine), and neither compares the two lengths."""
    snap = _snapshot(tmp_path, episodes=[{"episode_index": 0, "length": 10}])
    probe = _stub_ffprobe(
        tmp_path, {"width": 640, "height": 480, "codec_name": "av1", "nb_frames": "9"}
    )
    with pytest.raises(bps.BuildError, match=r"9 frames and episodes\.jsonl declares length 10"):
        bps.build(snap, tmp_path / "out", ffprobe=probe)


def test_a_downscaled_snapshot_is_refused_by_metadata(tmp_path: pathlib.Path) -> None:
    """The 120x160 tree has a perfectly valid manifest and would sail through a generic check."""
    snap = _snapshot(tmp_path, shape=[120, 160, 3])
    probe = _stub_ffprobe(
        tmp_path, {"width": 160, "height": 120, "codec_name": "av1", "nb_frames": "10"}
    )
    with pytest.raises(bps.BuildError, match=r"160x120.*does not rescale"):
        bps.build(snap, tmp_path / "out", ffprobe=probe)


def test_metadata_and_pixels_disagreeing_is_refused_rather_than_resolved(
    tmp_path: pathlib.Path,
) -> None:
    """info.json says 640x480 and the file is not. Trusting either silently would be a guess."""
    snap = _snapshot(tmp_path, episodes=[{"episode_index": 0, "length": 10}])
    probe = _stub_ffprobe(
        tmp_path, {"width": 320, "height": 240, "codec_name": "av1", "nb_frames": "10"}
    )
    with pytest.raises(bps.BuildError, match=r"is 320x240.*metadata and the pixels disagree"):
        bps.build(snap, tmp_path / "out", ffprobe=probe)


def test_a_metadata_only_snapshot_says_so_instead_of_raising_filenotfound(
    tmp_path: pathlib.Path,
) -> None:
    """This is the exact state of the HF cache on the workstation: info.json, no episodes.jsonl."""
    snap = _snapshot(tmp_path, with_episodes_file=False)
    probe = _stub_ffprobe(
        tmp_path, {"width": 640, "height": 480, "codec_name": "av1", "nb_frames": "10"}
    )
    with pytest.raises(bps.BuildError, match=r"metadata-only download"):
        bps.build(snap, tmp_path / "out", ffprobe=probe)


def test_a_missing_video_is_refused(tmp_path: pathlib.Path) -> None:
    snap = _snapshot(tmp_path, episodes=[{"episode_index": 0, "length": 10}], with_videos=False)
    probe = _stub_ffprobe(
        tmp_path, {"width": 640, "height": 480, "codec_name": "av1", "nb_frames": "10"}
    )
    with pytest.raises(bps.BuildError, match=r"missing.*fetch the dataset in full"):
        bps.build(snap, tmp_path / "out", ffprobe=probe)


def test_an_unreadable_frame_count_is_refused_not_guessed(tmp_path: pathlib.Path) -> None:
    snap = _snapshot(tmp_path, episodes=[{"episode_index": 0, "length": 10}])
    probe = _stub_ffprobe(
        tmp_path, {"width": 640, "height": 480, "codec_name": "av1", "nb_frames": "N/A"}
    )
    with pytest.raises(bps.BuildError, match=r"no frame count, so it cannot be verified"):
        bps.build(snap, tmp_path / "out", ffprobe=probe)


def test_packets_stand_in_when_the_container_carries_no_nb_frames(tmp_path: pathlib.Path) -> None:
    """Not every container writes nb_frames. One packet is one frame for these single-stream mp4s,
    and counting packets does not decode."""
    snap = _snapshot(tmp_path, episodes=[{"episode_index": 0, "length": 10}])
    probe = _stub_ffprobe(
        tmp_path, {"width": 640, "height": 480, "codec_name": "av1", "nb_read_packets": "10"}
    )
    manifest = bps.build(snap, tmp_path / "out", ffprobe=probe)
    assert manifest["episodes"][0]["frames"] == 10


# ------------------------------------------------------------------------------------------
# the small pieces
# ------------------------------------------------------------------------------------------


def test_video_relpath_follows_the_chunking_the_snapshot_declares() -> None:
    info = {
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "chunks_size": 1000,
    }
    assert bps.video_relpath(info, 0, "cam") == "videos/chunk-000/cam/episode_000000.mp4"
    assert bps.video_relpath(info, 1500, "cam") == "videos/chunk-001/cam/episode_001500.mp4"


def test_a_missing_camera_key_lists_the_ones_that_exist(tmp_path: pathlib.Path) -> None:
    info = {"features": {"observation.images.other": {"shape": [480, 640, 3]}}}
    with pytest.raises(bps.BuildError, match=r"observation\.images\.other"):
        bps.declared_resolution(info, bps.VIDEO_KEY)


def test_limit_is_labelled_a_smoke_corpus_on_stdout(tmp_path: pathlib.Path, capsys) -> None:
    """A truncated corpus that looks like the real one is a way to time 2 episodes and report 402."""
    snap = _snapshot(tmp_path)
    probe = _stub_ffprobe(
        tmp_path, {"width": 640, "height": 480, "codec_name": "av1", "nb_frames": "10"}
    )
    rc = bps.main(
        [
            "--snapshot",
            str(snap),
            "--out",
            str(tmp_path / "out"),
            "--ffprobe",
            probe,
            "--limit",
            "1",
        ]
    )
    assert rc == 0
    assert "SMOKE corpus" in capsys.readouterr().out


def test_a_refusal_exits_nonzero_from_main(tmp_path: pathlib.Path) -> None:
    """A job that treats exit 0 as 'corpus built' must not get one from a refusal."""
    snap = _snapshot(tmp_path, shape=[120, 160, 3])
    probe = _stub_ffprobe(
        tmp_path, {"width": 160, "height": 120, "codec_name": "av1", "nb_frames": "10"}
    )
    rc = bps.main(["--snapshot", str(snap), "--out", str(tmp_path / "out"), "--ffprobe", probe])
    assert rc == 1
