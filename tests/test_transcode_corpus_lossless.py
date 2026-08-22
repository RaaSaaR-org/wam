"""Tests for ``scripts/transcode_corpus_lossless.py`` — the AV1 -> H.264 re-encode of the PR-08 corpus.

The script exists because Cosmos-Transfer2.5 reads clips with a ``cv2`` that cannot decode AV1, and
it is *allowed* to exist only because the re-encode is bit-exact. So the thing under test is not
"does it produce an mp4" — it is "does it refuse when the mp4 is not the same video". Almost every
test here is therefore a refusal, and the ones that pass exist mainly to prove the refusals are not
firing on everything.

REAL CODECS, NOT STUBS. Unlike ``tests/test_build_pr08_source.py``, which injects a fake ffprobe,
these tests encode and decode actual video with PyAV. That is deliberate: the defect this script
guards against lives *in* the codec layer (a decoder that returns zero frames, an encoder that
drops a GOP, a chroma round trip that costs 7/255), and a stub that returns whatever the test
wants would prove nothing about it. The synthetic clips are tiny — 8 frames of 64x48 — so the price
is fractions of a second.

THE SOURCE FIXTURE IS AV1 ON PURPOSE where the encoder is available, because "it works on an H.264
source" is a weaker claim than the one being made. ``libsvtav1`` is not universal, so the fixture
falls back to H.264 and says so rather than skipping the whole module.
"""

from __future__ import annotations

import json
import pathlib
import sys
from fractions import Fraction

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

av = pytest.importorskip("av", reason="PyAV is the only decoder that can drive a lossless transcode")

import transcode_corpus_lossless as tcl  # noqa: E402


# ------------------------------------------------------------------------------------------------
# fixtures
# ------------------------------------------------------------------------------------------------

W, H, N_FRAMES, FPS = 64, 48, 8, 30


def _noise_frames(n: int = N_FRAMES, seed: int = 0) -> list[np.ndarray]:
    """High-entropy RGB frames.

    Noise rather than a gradient: a lossy encoder reproduces a smooth ramp almost perfectly, so a
    gentle fixture would let ``test_a_lossy_setting_is_caught_and_refused`` pass for the wrong
    reason. Noise is the content that makes a lossy setting visibly lossy in eight frames.
    """
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(n)]


def _source_codec() -> str:
    """AV1 if this build can encode it, else H.264 — the corpus is AV1 and the fixture should be."""
    for name in ("libsvtav1", "libaom-av1"):
        try:
            av.codec.Codec(name, "w")
            return name
        except Exception:  # noqa: BLE001 - not built in; that is an answer, not an error
            continue
    return "libx264"


def _write_clip(path: pathlib.Path, frames: list[np.ndarray], codec: str | None = None) -> None:
    """Write a clip the transcoder will read. Lossless, so the fixture's own codec is not a variable."""
    codec = codec or _source_codec()
    opts = {"crf": "0"} if codec == "libx264" else {"crf": "0", "preset": "8"}
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), "w")
    try:
        stream = container.add_stream(codec, rate=FPS, options=opts)
        stream.codec_context.width = W
        stream.codec_context.height = H
        stream.codec_context.pix_fmt = "yuv420p"
        # Named, not inherited: a stream's ``time_base`` is None until the codec is opened, so
        # reading it here to stamp the frames is the AttributeError that greeted the first run.
        time_base = Fraction(1, FPS)
        stream.codec_context.time_base = time_base
        stream.time_base = time_base
        for i, arr in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24").reformat(format="yuv420p")
            frame.pts = i
            frame.time_base = time_base
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()


def _decoded_frame_count(path: pathlib.Path) -> int:
    container = av.open(str(path))
    try:
        return sum(1 for _ in container.decode(video=0))
    finally:
        container.close()


@pytest.fixture()
def corpus(tmp_path: pathlib.Path) -> pathlib.Path:
    """A two-episode PR-08 source tree: ``manifest.json`` + ``videos/``, the shape 97 reads.

    Frame counts in the manifest are the DECODED counts, not the requested ones — some encoders
    emit a different number than they were fed, and a fixture that lied about its own lengths would
    make the manifest-agreement test fail for a reason that has nothing to do with the transcoder.
    """
    root = tmp_path / "src-corpus"
    episodes = []
    for idx in range(2):
        rel = f"videos/episode_{idx:06d}.mp4"
        _write_clip(root / rel, _noise_frames(seed=idx))
        episodes.append(
            {"id": f"episode_{idx:06d}", "frames": _decoded_frame_count(root / rel), "video": rel}
        )
    manifest = {
        "resolution": [W, H],
        "fps": FPS,
        "video_key": "observation.images.ego_view",
        "source": {
            "repo_id": "nvidia/GR00T-N1.7-AppleToPlate",
            "revision": "deadbeef",
            "codebase_version": "v2.1",
            "total_episodes": 2,
            "total_frames": sum(e["frames"] for e in episodes),
            "codecs": ["av1"],
            "materialized": "symlink",
        },
        "episodes": episodes,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root


def _run(corpus: pathlib.Path, out: pathlib.Path, *extra: str) -> int:
    return tcl.main(["--corpus", str(corpus), "--out", str(out), "--jobs", "1", *extra])


def _proof(out: pathlib.Path) -> dict:
    return json.loads((out / "TRANSCODE_PROOF.json").read_text())


# ------------------------------------------------------------------------------------------------
# the happy path, which is the only one allowed to write a manifest
# ------------------------------------------------------------------------------------------------


def test_a_clip_round_trips_bit_exactly_and_the_artifact_records_the_proof(
    corpus: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    out = tmp_path / "out"
    assert _run(corpus, out) == 0

    proof = _proof(out)
    assert proof["complete"] is True
    assert proof["clips_proven_bit_exact"] == 2
    assert proof["clips_refused"] == 0
    # FULL, not SAMPLED. A reader who sees this artifact must be able to tell without re-running it
    # whether every frame was compared or only some.
    assert proof["verification"]["mode"] == "FULL"
    assert proof["verification"]["stride"] == 1

    for clip in proof["clips"]:
        assert clip["ok"] is True
        # The number the whole exercise turns on, recorded per clip and in both spaces.
        assert clip["max_abs_delta"] == {"yuv420p": 0, "rgb24": 0}
        assert clip["frames_source"] == clip["frames_output"] > 0
        assert clip["frames_compared"] == clip["frames_output"]
        # Provenance: someone comparing a GEOM_TOL measured on one tree with a restyle run on the
        # other needs to be able to check that these are the same bytes they were handed.
        assert len(clip["source_sha256"]) == 64
        assert len(clip["output_sha256"]) == 64
        assert clip["source_sha256"] != clip["output_sha256"]
        assert clip["output_bytes"] > 0

    assert proof["decoder"]["name"] == "pyav"
    assert proof["encoder"]["crf"] == "0"
    assert proof["libav"]


def test_the_pixels_on_disk_are_the_same_pixels_not_just_the_recorded_delta(
    corpus: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Re-derive the claim independently of the script that made it.

    Every other assertion in this file reads a number the script wrote about itself. This one
    decodes both trees from scratch and compares them, so a verifier that always reported zero
    would be caught here rather than believed everywhere.
    """
    out = tmp_path / "out"
    assert _run(corpus, out) == 0

    for name in ("episode_000000.mp4", "episode_000001.mp4"):
        a = av.open(str(corpus / "videos" / name))
        b = av.open(str(out / "videos" / name))
        try:
            pairs = list(zip(a.decode(video=0), b.decode(video=0), strict=True))
        finally:
            a.close()
            b.close()
        assert pairs, f"{name} decoded nothing on at least one side"
        for x, y in pairs:
            assert np.array_equal(
                x.to_ndarray(format="yuv420p"), y.to_ndarray(format="yuv420p")
            ), f"{name} differs in its native planes"


def test_the_new_manifest_says_h264_and_names_the_manifest_it_descends_from(
    corpus: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A manifest still claiming ``av1`` would send the next reader hunting a decoder they do not need."""
    import hashlib

    out = tmp_path / "out"
    assert _run(corpus, out) == 0

    new = json.loads((out / "manifest.json").read_text())
    src = json.loads((corpus / "manifest.json").read_text())

    assert new["source"]["codecs"] == ["h264"]
    assert new["source"]["materialized"] == tcl.MATERIALIZED
    # Carried through verbatim: none of it changed, and re-deriving it would only create a chance
    # to disagree with the tree this one descends from.
    assert new["resolution"] == src["resolution"]
    assert new["fps"] == src["fps"]
    assert new["video_key"] == src["video_key"]
    assert new["source"]["revision"] == src["source"]["revision"]
    assert [e["id"] for e in new["episodes"]] == [e["id"] for e in src["episodes"]]
    assert [e["frames"] for e in new["episodes"]] == [e["frames"] for e in src["episodes"]]

    t = new["source"]["transcode"]
    assert t["lossless"] is True
    assert t["from_codecs"] == ["av1"]
    assert t["of_manifest_sha256"] == hashlib.sha256(
        (corpus / "manifest.json").read_bytes()
    ).hexdigest()
    # The proof is named AND pinned. A pointer to a file anyone can edit is not evidence.
    assert t["proof"] == "TRANSCODE_PROOF.json"
    assert t["proof_sha256"] == hashlib.sha256(
        (out / "TRANSCODE_PROOF.json").read_bytes()
    ).hexdigest()


def test_both_manifest_and_proof_get_a_sha256_sidecar(
    corpus: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    import hashlib

    out = tmp_path / "out"
    assert _run(corpus, out) == 0
    for name in ("manifest.json", "TRANSCODE_PROOF.json"):
        blob = (out / name).read_bytes()
        side = (out / f"{name}.sha256").read_text()
        assert side == f"{hashlib.sha256(blob).hexdigest()}  {name}\n"


# ------------------------------------------------------------------------------------------------
# refusals
# ------------------------------------------------------------------------------------------------


def test_a_lossy_setting_is_caught_by_the_comparison_and_refused(
    corpus: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """``--crf 28`` produces a perfectly valid, perfectly watchable, perfectly wrong mp4.

    This is the test that makes the word "lossless" in the new manifest mean something. Without it
    the script is a re-encoder that writes a manifest asserting a property nobody checked.
    """
    out = tmp_path / "out"
    assert _run(corpus, out, "--crf", "28") == 1

    proof = _proof(out)
    assert proof["complete"] is False
    assert proof["clips_refused"] == 2
    assert proof["clips_proven_bit_exact"] == 0
    assert all("NOT bit-exact" in c["error"] for c in proof["clips"])

    # The tree must not be usable. No manifest is the mechanism: 97_transfer25_restyle.sbatch reads
    # ${SOURCE}/manifest.json and refuses without one.
    assert not (out / "manifest.json").exists()
    # And the lossy files never took the names the manifest would have pointed at.
    assert list((out / "videos").glob("*.mp4")) == []


def test_a_dropped_frame_is_refused_even_though_every_kept_frame_is_bit_exact(
    corpus: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An encoder that silently drops its last GOP is the whole reason the count is asserted.

    The pixels of every surviving frame are identical, so the delta check passes; only the count
    catches it. Downstream that defect pairs every later action label with a different instant than
    its pixels, with no decode error anywhere.
    """
    real = tcl.transcode_clip

    def short(src, dst, decoder, enc):
        rec = real(src, dst, decoder, enc)
        # Re-encode the output one frame shorter, in place, using the script's own machinery.
        keep = []
        container = av.open(str(dst))
        try:
            keep = list(container.decode(video=0))[:-1]
        finally:
            container.close()
        # A plain ``.mp4`` name, because ``dst`` here is the ``.part`` file and libav cannot guess
        # a muxer from that suffix — the same trap ``container_format_for`` exists for.
        rebuilt = dst.parent / "shortened.mp4"
        _write_clip(rebuilt, [f.to_ndarray(format="rgb24") for f in keep], codec="libx264")
        rebuilt.replace(dst)
        return rec

    monkeypatch.setattr(tcl, "transcode_clip", short)
    out = tmp_path / "out"
    assert _run(corpus, out) == 1

    proof = _proof(out)
    assert proof["clips_refused"] == 2
    for clip in proof["clips"]:
        assert "frames" in clip["error"] and "source holds" in clip["error"]
    assert not (out / "manifest.json").exists()


def test_a_manifest_frame_count_that_disagrees_with_the_pixels_is_refused_not_laundered(
    corpus: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The transcode is not the place to quietly correct a source manifest that is already wrong."""
    manifest = json.loads((corpus / "manifest.json").read_text())
    manifest["episodes"][0]["frames"] += 1
    (corpus / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    out = tmp_path / "out"
    assert _run(corpus, out) == 1
    bad = [c for c in _proof(out)["clips"] if not c["ok"]]
    assert len(bad) == 1
    assert "source manifest declares" in bad[0]["error"]


def test_a_decoder_that_returns_zero_frames_refuses_rather_than_writing_an_empty_output(
    corpus: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Jobs 186357 and 189584, in a unit test.

    The container opens, the header is believed, no pixels arrive. Without this refusal the run
    would write 402 valid, empty mp4s and a manifest declaring 171 625 frames.
    """
    real_open = tcl.DECODERS["pyav"].open_fn

    def empty(clip):
        _, info = real_open(clip)
        return iter(()), info

    monkeypatch.setitem(tcl.DECODERS, "pyav", tcl.DECODERS["pyav"].__class__(
        name="pyav", version="stub", open_fn=empty, note="decodes nothing, on purpose"))

    out = tmp_path / "out"
    # It does not even get as far as a clip: resolve_decoder probes the corpus first and no
    # decoder returns a frame, so the run stops at the environment check.
    assert _run(corpus, out) == 2
    assert not (out / "videos").exists() or list((out / "videos").glob("*.mp4")) == []
    assert not (out / "manifest.json").exists()


def test_a_decoder_that_dies_only_on_the_second_clip_still_refuses_and_leaves_no_output(
    corpus: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe clip decoding is not a promise about the other 401.

    Separated from the test above because the two failures have different shapes: one is caught by
    ``resolve_decoder`` before any work, the other by ``transcode_clip``'s zero-frame guard after
    a container has already been opened for writing.
    """
    real_open = tcl.DECODERS["pyav"].open_fn

    def flaky(clip):
        frames, info = real_open(clip)
        if clip.name.endswith("000001.mp4"):
            return iter(()), info
        return frames, info

    monkeypatch.setitem(tcl.DECODERS, "pyav", tcl.DECODERS["pyav"].__class__(
        name="pyav", version="stub", open_fn=flaky, note="dies on episode 1"))

    out = tmp_path / "out"
    assert _run(corpus, out) == 1
    proof = _proof(out)
    assert proof["clips_proven_bit_exact"] == 1
    bad = [c for c in proof["clips"] if not c["ok"]]
    assert len(bad) == 1 and "ZERO frames" in bad[0]["error"]
    # The good clip is on disk; the bad one left nothing behind, not even a .part.
    assert (out / "videos" / "episode_000000.mp4").is_file()
    assert not (out / "videos" / "episode_000001.mp4").exists()
    assert list((out / "videos").glob("*.part")) == []
    assert not (out / "manifest.json").exists()


def test_a_corpus_without_a_manifest_is_a_usage_error_not_an_empty_success(
    tmp_path: pathlib.Path
) -> None:
    assert _run(tmp_path / "nothing-here", tmp_path / "out") == 2


def test_a_manifest_naming_a_clip_the_tree_does_not_hold_is_refused_before_any_work(
    corpus: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A dangling symlink in the source tree looks exactly like this."""
    (corpus / "videos" / "episode_000001.mp4").unlink()
    out = tmp_path / "out"
    assert _run(corpus, out) == 2
    assert not (out / "TRANSCODE_PROOF.json").exists()


# ------------------------------------------------------------------------------------------------
# resume
# ------------------------------------------------------------------------------------------------


def test_resume_skips_a_verified_clip_without_re_encoding_it(
    corpus: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    assert _run(corpus, out) == 0
    first = {p.name: p.read_bytes() for p in sorted((out / "videos").glob("*.mp4"))}

    def forbidden(*a, **k):  # noqa: ANN002, ANN003
        raise AssertionError("re-encoded a clip that was already on disk and verifiable")

    monkeypatch.setattr(tcl, "transcode_clip", forbidden)
    assert _run(corpus, out) == 0

    proof = _proof(out)
    assert proof["clips_resumed"] == 2
    assert proof["clips_proven_bit_exact"] == 2
    # Byte-identical: a resume that rewrote the files would invalidate every sha256 already quoted.
    assert {p.name: p.read_bytes() for p in sorted((out / "videos").glob("*.mp4"))} == first
    # And it still proved them, rather than copying a verdict forward.
    assert all(c["max_abs_delta"] == {"yuv420p": 0, "rgb24": 0} for c in proof["clips"])
    assert all(c["frames_compared"] == c["frames_output"] for c in proof["clips"])


def test_resume_re_verifies_rather_than_trusting_that_a_file_exists(
    corpus: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The failure mode: an interrupted run leaves a short mp4 that looks finished on an ``ls``.

    Existence is not a proof, and a resume that treated it as one would carry a truncated clip into
    the shipped tree with a manifest declaring the full length.
    """
    out = tmp_path / "out"
    assert _run(corpus, out) == 0

    victim = out / "videos" / "episode_000000.mp4"
    good_bytes = victim.read_bytes()
    # Overwrite with a genuinely shorter clip — still a valid mp4, still decodable, wrong length.
    _write_clip(victim, _noise_frames(n=N_FRAMES - 2, seed=0), codec="libx264")
    assert victim.read_bytes() != good_bytes

    assert _run(corpus, out) == 0
    proof = _proof(out)
    rec = next(c for c in proof["clips"] if c["id"] == "episode_000000")
    # It noticed, said so, and rebuilt it.
    assert rec["resumed"] is False
    assert "resume_reverify_failed" in rec
    assert rec["ok"] is True
    assert rec["frames_output"] == rec["frames_source"]
    assert victim.read_bytes() == good_bytes
    # The clip that was fine was not touched.
    assert next(c for c in proof["clips"] if c["id"] == "episode_000001")["resumed"] is True


def test_no_resume_re_encodes_everything(
    corpus: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    assert _run(corpus, out) == 0

    calls: list[str] = []
    real = tcl.transcode_clip
    monkeypatch.setattr(
        tcl, "transcode_clip",
        lambda src, dst, d, e: (calls.append(src.name), real(src, dst, d, e))[1],
    )
    assert _run(corpus, out, "--no-resume") == 0
    assert sorted(calls) == ["episode_000000.mp4", "episode_000001.mp4"]
    assert _proof(out)["clips_resumed"] == 0


# ------------------------------------------------------------------------------------------------
# sampling, which must never be silent
# ------------------------------------------------------------------------------------------------


def test_sampling_is_not_the_default_and_stamps_itself_all_over_the_artifact(
    corpus: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    out = tmp_path / "out"
    assert _run(corpus, out, "--verify-stride", "3") == 0

    proof = _proof(out)
    assert proof["verification"]["mode"] == "SAMPLED"
    assert proof["verification"]["stride"] == 3
    assert "does NOT prove" in proof["verification"]["note"]
    for clip in proof["clips"]:
        # Frame counts stay FULL under sampling — a dropped last GOP must not be able to hide
        # behind a stride.
        assert clip["frames_source"] == clip["frames_output"] == N_FRAMES
        assert clip["frames_compared"] < clip["frames_output"]


def test_a_stride_below_one_is_a_usage_error(corpus: pathlib.Path, tmp_path: pathlib.Path) -> None:
    assert _run(corpus, tmp_path / "out", "--verify-stride", "0") == 2


# ------------------------------------------------------------------------------------------------
# the pieces, tested directly
# ------------------------------------------------------------------------------------------------


def test_verify_pair_refuses_a_shorter_output_and_names_both_counts(
    corpus: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    src = corpus / "videos" / "episode_000000.mp4"
    short = tmp_path / "short.mp4"
    _write_clip(short, _noise_frames(seed=0)[:-1], codec="libx264")

    decoder = tcl.DECODERS["pyav"]
    with pytest.raises(tcl.ProofFailed) as excinfo:
        tcl.verify_pair(src, short, decoder, declared_frames=None)
    assert f"{N_FRAMES - 1} frames" in str(excinfo.value)
    assert f"source holds {N_FRAMES}" in str(excinfo.value)


def test_verify_pair_refuses_a_longer_output_too(
    corpus: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Asymmetry here would be a real hole: the ``zip`` a reader expects stops at the shorter side."""
    src = corpus / "videos" / "episode_000000.mp4"
    longer = tmp_path / "long.mp4"
    _write_clip(longer, _noise_frames(n=N_FRAMES + 3, seed=0), codec="libx264")

    with pytest.raises(tcl.ProofFailed, match="frames"):
        tcl.verify_pair(src, longer, tcl.DECODERS["pyav"], declared_frames=None)


def test_the_rgb_round_trip_this_script_avoids_really_is_lossy(
    corpus: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Pin the measurement the module docstring rests on, so nobody "simplifies" the transcode path.

    Rebuilding each frame through ``VideoFrame.from_ndarray(..., 'rgb24')`` and encoding at
    ``crf=0`` looks lossless and is not — the loss is in the chroma round trip, not the codec, so
    no encoder setting recovers it. If a future edit routes the pixels through RGB, this test says
    what the resulting refusals mean.
    """
    src = corpus / "videos" / "episode_000000.mp4"
    via_rgb = tmp_path / "via_rgb.mp4"

    container = av.open(str(src))
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    _write_clip(via_rgb, frames, codec="libx264")  # crf=0, and still not bit-exact

    with pytest.raises(tcl.ProofFailed, match="NOT bit-exact"):
        tcl.verify_pair(src, via_rgb, tcl.DECODERS["pyav"], declared_frames=None)


def test_resolve_decoder_probes_the_actual_corpus_before_choosing(
    corpus: pathlib.Path
) -> None:
    clip = corpus / "videos" / "episode_000000.mp4"
    chosen = tcl.resolve_decoder("auto", clip)
    assert chosen.name == "pyav"
    # The note is provenance: which decoder, why, and on what evidence.
    assert clip.name in chosen.note
    assert "decoded a" in chosen.note


def test_resolve_decoder_refuses_a_named_decoder_that_cannot_read_this_corpus(
    corpus: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_open = tcl.DECODERS["pyav"].open_fn
    monkeypatch.setitem(tcl.DECODERS, "pyav", tcl.DECODERS["pyav"].__class__(
        name="pyav", version="stub",
        open_fn=lambda c: (iter(()), real_open(c)[1]), note="decodes nothing"))
    with pytest.raises(tcl.TranscodeError, match="decoded no frames"):
        tcl.resolve_decoder("pyav", corpus / "videos" / "episode_000000.mp4")


def test_the_part_file_suffix_does_not_confuse_the_muxer(tmp_path: pathlib.Path) -> None:
    """``av.open`` cannot guess a muxer from ``.mp4.part``; the format is named, not sniffed."""
    assert tcl.container_format_for(tmp_path / "episode_000000.mp4.part") == "mp4"
    assert tcl.container_format_for(tmp_path / "episode_000000.mp4") == "mp4"
    with pytest.raises(tcl.TranscodeError, match="not an .mp4"):
        tcl.container_format_for(tmp_path / "episode_000000.mkv")


def test_timestamps_are_carried_rather_than_nulled(
    corpus: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The bug that killed the first attempt: ``frame.pts = None`` survives 24 frames and then
    fails muxing with errno 22. Carried pts are monotonic and the repair counter stays at zero."""
    src = corpus / "videos" / "episode_000000.mp4"
    dst = tmp_path / "out.mp4"
    rec = tcl.transcode_clip(src, dst, tcl.DECODERS["pyav"], tcl.EncoderSpec())
    assert rec["pts_repaired"] == 0
    assert rec["frames_encoded"] == N_FRAMES

    container = av.open(str(dst))
    try:
        pts = [f.pts for f in container.decode(video=0)]
    finally:
        container.close()
    assert all(p is not None for p in pts)
    assert pts == sorted(pts) and len(set(pts)) == len(pts)


def test_a_stream_with_no_timestamps_is_rebuilt_monotonically_and_counted(
    corpus: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The other half of the timestamp story: a source that carries no pts at all.

    Nulling ``frame.pts`` is what killed the first attempt at this script, so the repair is not
    "leave it to the encoder" — it is to rebuild the timeline from the frame rate, starting at 0,
    and to say in the artifact how many frames needed it. A clip whose timing was reconstructed is
    a different claim than one whose timing was copied.
    """
    real_open = tcl.DECODERS["pyav"].open_fn

    def unstamped(clip):
        frames, info = real_open(clip)

        def stripped():
            for frame in frames:
                frame.pts = None
                yield frame

        return stripped(), info

    src = corpus / "videos" / "episode_000000.mp4"
    dst = tmp_path / "out.mp4"
    decoder = tcl.DECODERS["pyav"].__class__(
        name="pyav", version="stub", open_fn=unstamped, note="hands back frames with no pts")

    rec = tcl.transcode_clip(src, dst, decoder, tcl.EncoderSpec())
    assert rec["frames_encoded"] == N_FRAMES
    assert rec["pts_repaired"] == N_FRAMES  # every one of them, and the artifact will say so

    container = av.open(str(dst))
    try:
        pts = [f.pts for f in container.decode(video=0)]
    finally:
        container.close()
    assert pts[0] == 0
    assert pts == sorted(pts) and len(set(pts)) == len(pts)
    # The pixels are still the pixels: rebuilding a timeline must not touch a plane.
    assert tcl.verify_pair(src, dst, tcl.DECODERS["pyav"], declared_frames=N_FRAMES)[
        "max_abs_delta"
    ] == {"yuv420p": 0, "rgb24": 0}


def test_the_encoder_spec_is_recorded_verbatim_so_two_runs_can_be_compared(
    corpus: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Output sha256s are only comparable if the encoder that made them is stated.

    ``threads`` in particular: x264's frame threading changes the bitstream without changing a
    pixel, so a digest from an auto-threaded run does not reproduce on a pinned one.
    """
    out = tmp_path / "out"
    assert _run(corpus, out, "--preset", "veryfast", "--encoder-threads", "1") == 0
    enc = _proof(out)["encoder"]
    assert enc == {"codec": "libx264", "crf": "0", "preset": "veryfast", "threads": 1}
    assert "preset=veryfast" in json.loads(
        (out / "manifest.json").read_text()
    )["source"]["transcode"]["encoder"]


def test_limit_writes_a_manifest_for_only_the_clips_it_actually_produced(
    corpus: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A smoke tree must not carry a manifest row for a clip that is not on disk."""
    out = tmp_path / "out"
    assert _run(corpus, out, "--limit", "1") == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert [e["id"] for e in manifest["episodes"]] == ["episode_000000"]
    assert (out / "videos" / "episode_000000.mp4").is_file()
    assert not (out / "videos" / "episode_000001.mp4").exists()


def test_the_local_cv2_readback_refuses_to_present_itself_as_the_gate(
    corpus: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """It is the wrong cv2 and the artifact has to say so.

    The build that matters is the generation venv's 4.11.0/avcodec 59.37.100. A green tick from
    this workstation's cv2 is worth exactly what job 186357's happy ffprobe was worth, and an
    artifact that let a reader mistake one for the other would be re-arming the same trap.
    """
    out = tmp_path / "out"
    assert _run(corpus, out) == 0
    readback = _proof(out).get("cv2_readback")
    if readback is None or not readback.get("ran"):
        pytest.skip("cv2 is not importable by this interpreter; nothing to assert")
    assert "verify_clip_decode.py" in readback["NOT_THE_GATE"]
    assert readback["interpreter"] == sys.executable
    assert readback["cv2_version"]
