#!/usr/bin/env python3
"""Re-encode a PR-08 source corpus from AV1 to H.264 **without changing a single pixel**, and prove it.

    .venv/bin/python scripts/transcode_corpus_lossless.py \\
        --corpus /home/humanoid/wam-t041/pr08-apple-640x480 \\
        --out    /home/humanoid/wam-t041/pr08-apple-640x480-h264-lossless \\
        --jobs 8

WHY THIS EXISTS — THREE CLUSTER JOBS, ONE DEFECT
================================================
The AppleToPlate corpus is AV1. Cosmos-Transfer2.5 reads clips with ``cv2.VideoCapture``
(``cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_utils.py``), and the ``cv2`` 4.11.0 in the
generation venv is built against ``avcodec 59.37.100``, which has no AV1 decoder. ``VideoCapture``
does not raise on that. It opens the container, believes the header, reports 590 frames at
640x480/30fps, and then fails every read. Three jobs have now died on exactly that shape:

* **186357** captioned 372 clips and wrote 0 captions — nothing crashed, the model was simply
  handed ``array([], shape=(0, 480, 640, 3))`` 372 times.
* **189585**, the GEOM_TOL pilot, died at 7 s on ``Missing Sequence Header``.
* **189584** (2026-08-21 13:51), the PR-08 §8 item 3 throughput measurement, watched upstream's own
  SAM2 helper print ``Number of frames: 590`` and then ``Done extracting frames. 0 frames
  extracted``, followed by ``RuntimeError: no images found``. It refused to write
  ``THROUGHPUT.json``, correctly: a wall clock around a failed unit is not a throughput number.

Job **189586** probed the generation venv and closed off the cheap escapes: ``cv2`` 4.11.0 cannot
decode this corpus; ``av`` 16.0.1 (libdav1d) and ``imageio`` 2.37.0 can; and
``ffmpeg_av1_decoders`` is ``null`` because **there is no ffmpeg CLI on PATH in that environment at
all**. Putting one there would not help either — ``cv2`` bundles its own libav and would keep using
it. The corpus has to arrive in a codec that ``cv2`` build can read.

WHY LOSSLESS, AND WHY THAT WORD HAS TO BE EARNED
================================================
The standing objection to transcoding (recorded in ``scripts/measure_geom_tol.py``'s decoder seam)
was that a lossy re-encode would sit between ``GEOM_TOL`` and the pixels the generator actually
sees — and ``GEOM_TOL`` is denominated in *fractions of a pixel*, so "visually identical" is not a
defence. That objection dissolves only if the transcode is **bit-exact**, and only if bit-exactness
is *proven per clip* rather than assumed from a flag.

Two things had to be measured to get there, and one of them is a trap:

* Decoding AV1 to ``rgb24``, rebuilding frames with ``VideoFrame.from_ndarray(..., 'rgb24')`` and
  encoding to ``yuv420p`` at ``crf=0`` is **NOT** bit-exact — max channel delta 7-10/255. That is
  the RGB<->YUV chroma round-trip, not the codec, and no encoder setting fixes it. **Never route
  the pixels through RGB.**
* Handing the decoded ``av.VideoFrame`` **straight to the encoder in its native ``yuv420p``
  planes**, with the output ``pix_fmt`` copied from the input and ``crf=0``, **is** bit-exact:
  max channel delta 0 in both ``yuv420p`` and ``rgb24`` across all 590 frames of
  ``episode_000000.mp4``.

So this script never materialises an ndarray on the transcode path. Arrays appear only in the
verifier, where their whole job is to disagree with each other.

THE TIMESTAMP BUG, HANDLED RATHER THAN NULLED
=============================================
Setting ``frame.pts = None`` before encoding lets libx264 invent timestamps. That survives 24
frames and then dies on a real 590-frame clip with ``av.error.ArgumentError: Invalid argument
(errno 22)`` out of ``mux`` — the muxer is handed packets whose timestamps do not fit the stream's
time base. The fix is not to null the pts, it is to own them: the output stream's ``time_base`` is
copied from the input (1/15360 here, not 1/30), each frame's ``time_base`` is set explicitly, and
the source pts is carried through unchanged. A pts that is missing or non-monotonic — which
reordered streams can produce — is *regenerated* from the frame rate and **counted in the artifact**
under ``pts_repaired``, because a clip whose timing had to be rebuilt is a different claim than one
whose timing was copied, and the reader deserves to be told which they have.

WHAT IS AND IS NOT PROVEN HERE
==============================
Proven, per clip, by decoding both files and comparing arrays: identical frame **count**, and
identical **pixels** in the native ``yuv420p`` planes *and* in ``rgb24``. The plane comparison is
the strong one (YUV->RGB clips, so two different YUV values can land on one RGB value); the
``rgb24`` comparison is the one a reader can map onto what the consumer sees. Both must be zero.
The frame-count assertion is not a formality: an encoder that silently drops its last GOP is
precisely the class of failure this whole exercise is about.

**Not** proven here: that the *cluster's* ``cv2`` 4.11.0 can read the output. This workstation's
``cv2`` is a different build, so its opinion is worth exactly what job 186357's ffprobe was worth.
``--check-cv2`` runs the local check anyway and records which cv2 said so, but the gate that counts
is ``scripts/verify_clip_decode.py`` run **with the generation venv's interpreter** on the shipped
tree. Do not skip it.

REFUSAL SEMANTICS
=================
A clip that is not bit-exact, or whose frame count disagrees with either its source or the source
manifest, is a **refusal**. Its output is never renamed into place, and the run does not write a
``manifest.json`` for the new tree — which leaves the tree unusable by
``cluster/discoverer/97_transfer25_restyle.sbatch``, exactly as intended. ``TRANSCODE_PROOF.json``
is written either way, because the evidence of a failed run is still evidence; it carries
``complete: false`` and names every refusal.

Exit codes::

    0  every clip proven bit-exact; manifest and proof written
    1  at least one clip refused; proof written, manifest deliberately withheld
    2  environment or usage: corpus missing, manifest unreadable, no decoder proven on the input
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterator

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Bumped when the shape of TRANSCODE_PROOF.json changes in a way a reader must notice.
PROOF_SCHEMA = "1.0.0"

#: Written into the new manifest's ``source.materialized``. ``build_pr08_source.py`` writes
#: "symlink" or "copy"; this tree is neither, and calling it either would be a lie about whether
#: the bytes on disk are the recorded bytes.
MATERIALIZED = "transcode"

#: CRF 0 is *the* lossless setting for libx264 in YUV. It is not a parameter: every other value is
#: lossy, and a lossy value would be caught by the verifier one clip later and refused anyway, so
#: exposing it would only buy a slower way to fail. ``--crf`` exists solely so the test suite can
#: prove the verifier catches a lossy setting.
LOSSLESS_CRF = "0"


class TranscodeError(RuntimeError):
    """A refusal. Every one of these means the output tree would not be what the manifest claims."""


class ProofFailed(TranscodeError):
    """A clip did not survive the round trip. Distinguished so the CLI can count it separately."""


# --------------------------------------------------------------------------------------------
# decoders
#
# A CORPUS IS ONLY READABLE BY THE DECODER THAT WILL ACTUALLY READ IT. That sentence has cost this
# project three cluster jobs (see the module docstring) and it applies to this script with extra
# force, because a decoder that silently yields zero frames here would not merely mismeasure
# something — it would write a zero-frame output file and, without the frame-count assertion,
# call it a success. So the decoder is PROBED against the corpus being transcoded before a single
# clip is written, exactly as scripts/measure_geom_tol.py's resolve_decoder() does.
#
# WHY THE TABLE HAS ONE ENTRY. The transcode path needs native ``av.VideoFrame`` objects: the whole
# bit-exactness result depends on the planes never being converted to RGB and back. cv2 hands back
# BGR ndarrays and imageio hands back RGB ndarrays, so neither *can* be an entry in this table
# however well it decodes — they are listed in ``UNUSABLE_FOR_TRANSCODE`` so that absence reads as
# a decision rather than an oversight. The table shape is kept because the day a second
# native-frame reader exists it should slot in and be probed like the first, not be special-cased.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipInfo:
    """What the transcoder must copy from the input stream so the output is the same video."""

    width: int
    height: int
    pix_fmt: str
    time_base: Fraction
    rate: Fraction
    codec: str
    declared_frames: int


@dataclass(frozen=True)
class Decoder:
    """One way of turning a clip into NATIVE frames, named and versioned for the artifact."""

    name: str
    version: str
    #: clip -> (lazy iterator of native frames, ClipInfo). Lazy on purpose: a 590-frame 640x480
    #: episode held whole as yuv420p is 272 MB, and nothing here needs more than one frame at once.
    open_fn: Callable[[Path], tuple[Iterator[Any], ClipInfo]]
    note: str = ""


#: Decoders that work but cannot be used for a lossless transcode, and why. Recorded in the
#: artifact so "why not cv2, it decodes fine on my box" is answered in the evidence file.
UNUSABLE_FOR_TRANSCODE = {
    "cv2": "hands back BGR ndarrays; routing the pixels through RGB costs 7-10/255 in chroma",
    "imageio": "hands back RGB ndarrays; same round-trip loss",
}


def _module_version(module: str) -> str:
    """Version string, or a marker saying it is not installed.

    Never raises: an absent decoder is a fact the artifact should be able to state, not a reason
    this module fails to import.
    """
    try:
        mod = importlib.import_module(module)
    except Exception:  # noqa: BLE001 - absent, or broken on import; both are "not usable"
        return "<not importable>"
    return str(getattr(mod, "__version__", "<no __version__>"))


def _pyav_open(clip: Path) -> tuple[Iterator[Any], ClipInfo]:
    import av

    container = av.open(str(clip))
    try:
        stream = container.streams.video[0]
    except IndexError as exc:
        container.close()
        raise TranscodeError(f"FATAL: {clip} carries no video stream.") from exc

    cc = stream.codec_context
    rate = stream.average_rate or stream.guessed_rate or Fraction(30, 1)
    info = ClipInfo(
        width=int(cc.width),
        height=int(cc.height),
        pix_fmt=str(cc.pix_fmt),
        time_base=Fraction(stream.time_base) if stream.time_base else Fraction(1, int(rate)),
        rate=Fraction(rate),
        codec=str(cc.name),
        declared_frames=int(stream.frames or 0),
    )

    def frames() -> Iterator[Any]:
        try:
            for frame in container.decode(video=0):
                yield frame
        finally:
            container.close()

    return frames(), info


DECODERS: dict[str, Decoder] = {
    "pyav": Decoder(
        name="pyav",
        version=_module_version("av"),
        open_fn=_pyav_open,
        note="av.open, native VideoFrame planes, no RGB conversion anywhere on the transcode path. "
        "Decodes the PR-08 AV1 corpus via libdav1d (job 189586).",
    ),
}


def decoder_probe(decoder: Decoder, clip: Path) -> tuple[bool, str]:
    """Can this decoder pull ONE frame out of this clip? Returns (ok, detail).

    One frame is the whole question. The failure this exists for is not a corrupt file — it is a
    decoder that opens the container, believes the header, and returns nothing, so the only
    distinction that matters is between zero frames and one.
    """
    try:
        frames, info = decoder.open_fn(clip)
        for frame in frames:
            return True, f"decoded a {info.width}x{info.height} {info.pix_fmt} frame from {info.codec}"
        return False, "opened the container and decoded no frames"
    except TranscodeError as exc:
        return False, str(exc).splitlines()[0]
    except Exception as exc:  # noqa: BLE001 - any import/codec failure is "not usable", not a crash
        return False, f"{type(exc).__name__}: {exc}"


def resolve_decoder(name: str, probe_clip: Path) -> Decoder:
    """Pick the decoder, and never pick one that cannot read this corpus.

    ``auto`` is not "whatever imports". It PROBES, in order, and takes the first that actually
    returns a frame from the corpus being transcoded — because the failure mode is silent and a
    decoder that imports cleanly is not evidence of anything. The choice and the reason both go
    into the artifact.
    """
    if name != "auto":
        if name not in DECODERS:
            raise TranscodeError(f"FATAL: unknown decoder {name!r}; known: {sorted(DECODERS)}")
        decoder = DECODERS[name]
        ok, detail = decoder_probe(decoder, probe_clip)
        if not ok:
            raise TranscodeError(
                f"FATAL: --decoder {name} {detail} on {probe_clip.name}.\n"
                f"       {decoder.note}\n"
                "       This is the failure that reports the frame count off the container header "
                "and decodes none of\n"
                "       them. Nothing was transcoded, which is not the same as nothing being wrong."
            )
        return replace(decoder, note=f"{decoder.note} Probed on {probe_clip.name}: {detail}.")

    tried: list[str] = []
    for candidate in DECODERS.values():
        ok, detail = decoder_probe(candidate, probe_clip)
        tried.append(f"{candidate.name} ({candidate.version}): {detail}")
        if ok:
            return replace(
                candidate,
                note=(
                    f"{candidate.note} Selected by --decoder auto after probing "
                    f"{probe_clip.name}; tried in order: " + "; ".join(tried) + "."
                ),
            )
    raise TranscodeError(
        f"FATAL: no decoder known to this script could read {probe_clip}.\n"
        + "".join(f"       {line}\n" for line in tried)
        + "       The container parses and no codec here decodes it. Nothing was transcoded, "
        "which is not a pass.\n"
        "       scripts/verify_clip_decode.py reports the same thing per clip over a whole corpus."
    )


# --------------------------------------------------------------------------------------------
# the transcode itself
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EncoderSpec:
    """Everything about the encoder that could change a byte of the output, in one picklable lump.

    It is frozen and it goes into the artifact verbatim. Two people comparing output sha256s need
    to be able to see, without reading this file, whether they ran the same encoder.
    """

    codec: str = "libx264"
    crf: str = LOSSLESS_CRF
    #: PRESET IS A CHOICE, NOT A PREFERENCE, and here is the measurement behind it. On
    #: episode_000000.mp4 (590 frames, 640x480, source 3 076 473 B), single-threaded, lossless:
    #:   ultrafast  27 399 057 B   0.2 s
    #:   veryfast   22 364 357 B   1.1 s
    #:   medium     21 272 359 B   3.5 s
    #:   slow       21 845 386 B   3.0 s   (auto-threaded; frame threading costs bitrate)
    #: The spread from veryfast to medium is 4.9 % of ~6 GB, which is worth ~2.4 s per clip on a
    #: corpus transcoded once and then shipped to a cluster; the spread from medium to slow is
    #: nothing at all. So: medium. Nothing downstream can tell the difference — that is the point
    #: of lossless — so this trades only bytes against local wall clock.
    preset: str = "medium"
    #: PINNED, and pinned to 1 rather than left on auto. x264's frame threading changes the
    #: bitstream (not the pixels), so an output sha256 recorded by an 8-way run would not reproduce
    #: on a 1-way one. Pinning makes the recorded digests a property of the input and this spec,
    #: independent of --jobs. Parallelism lives in the process pool instead, where it does not
    #: touch the bytes.
    threads: int = 1

    def options(self) -> dict[str, str]:
        return {"crf": self.crf, "preset": self.preset}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def container_format_for(dst: Path) -> str:
    """Container format name for ``dst``, chosen explicitly rather than sniffed from the suffix.

    Outputs are written to ``<name>.mp4.part`` and renamed only once proven, and libav cannot guess
    a muxer from ``.part`` — it raises ``ValueError: Could not determine output format``. Naming the
    muxer is also the more honest form: the container is a decision this script makes, not something
    it inherits from a temporary filename.
    """
    stem = dst.name[: -len(".part")] if dst.name.endswith(".part") else dst.name
    suffix = Path(stem).suffix.lower()
    if suffix != ".mp4":
        raise TranscodeError(
            f"FATAL: {stem} is not an .mp4. This script writes the mp4 container only — "
            "97_transfer25_restyle.sbatch and the source manifest both name .mp4 clips."
        )
    return "mp4"


def transcode_clip(src: Path, dst: Path, decoder: Decoder, enc: EncoderSpec) -> dict[str, Any]:
    """Re-encode one clip, native planes only. Returns encode-side facts; proves nothing by itself.

    The pixels are never converted. ``frame`` comes out of the decoder in ``yuv420p`` and goes into
    the encoder in ``yuv420p``; the only things this function chooses are the container, the codec
    and the timestamps. If it ever grows a ``to_ndarray`` on this path, the bit-exactness result in
    the module docstring stops holding and the verifier will start refusing every clip.
    """
    import av

    frames, info = decoder.open_fn(src)

    dst.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(dst), "w", format=container_format_for(dst))
    try:
        stream = container.add_stream(enc.codec, rate=info.rate, options=enc.options())
        cc = stream.codec_context
        cc.width = info.width
        cc.height = info.height
        # THE line. Same pix_fmt in as out is what makes crf=0 mean "these exact planes".
        cc.pix_fmt = info.pix_fmt
        cc.thread_count = enc.threads
        # Timestamps: the OUTPUT stream carries the INPUT's time base, so a carried-over pts means
        # the same instant on both sides. Copying 1/30 here instead of the container's 1/15360
        # would silently rescale every timestamp.
        cc.time_base = info.time_base
        stream.time_base = info.time_base

        step = max(1, round(1 / (info.rate * info.time_base)))
        written = 0
        pts_repaired = 0
        last_pts: int | None = None
        for frame in frames:
            pts = frame.pts
            if pts is None or (last_pts is not None and pts <= last_pts):
                # Missing or non-monotonic. Rebuild from the frame rate rather than handing the
                # muxer something it will reject with errno 22 five hundred frames from now.
                pts = 0 if last_pts is None else last_pts + step
                pts_repaired += 1
            frame.pts = pts
            frame.time_base = info.time_base
            last_pts = pts
            for packet in stream.encode(frame):
                container.mux(packet)
            written += 1
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()

    if written == 0:
        # Refuse here rather than let a 0-byte-of-video mp4 reach the verifier. This is the exact
        # shape of jobs 186357/189584: the reader opened the file, the header was believed, and no
        # pixels moved. An empty output that "verified" against an empty source would be worse.
        raise ProofFailed(
            f"FATAL: {src.name} decoded ZERO frames with {decoder.name} {decoder.version} — "
            "the container parses and the codec does not. Nothing was written."
        )

    return {
        "frames_encoded": written,
        "pts_repaired": pts_repaired,
        "source_codec": info.codec,
        "pix_fmt": info.pix_fmt,
        "width": info.width,
        "height": info.height,
        "fps": float(info.rate),
        "time_base": str(info.time_base),
    }


# --------------------------------------------------------------------------------------------
# the proof
# --------------------------------------------------------------------------------------------

#: yuv420p is the STRONG comparison: it is the space the codec actually operates in, and YUV->RGB
#: clips, so two distinct YUV values can land on one RGB value — an rgb24-only check could pass on
#: planes that differ. rgb24 is the LEGIBLE comparison: it is what the consumer's cv2 hands its
#: model. Requiring both means the artifact answers "are the files identical" and "does the
#: generator see the same picture" without the reader having to reason about colour spaces.
COMPARE_SPACES = ("yuv420p", "rgb24")


def verify_pair(
    src: Path,
    dst: Path,
    decoder: Decoder,
    *,
    declared_frames: int | None,
    stride: int = 1,
) -> dict[str, Any]:
    """Decode both files and refuse unless they are the same video, frame for frame.

    Returns a record. Raises ``ProofFailed`` on any disagreement — this is the function that makes
    the word "lossless" in the manifest mean something, so it has no warning level.

    ``stride`` > 1 compares every k-th frame. Frame COUNTS are always full on both sides regardless
    of stride, because a dropped last GOP is the failure this is here to catch and sampling must
    not be able to hide it.
    """
    import numpy as np

    a_frames, a_info = decoder.open_fn(src)
    b_frames, b_info = decoder.open_fn(dst)

    maxima = {space: 0 for space in COMPARE_SPACES}
    first_bad: dict[str, Any] | None = None
    n_src = n_dst = n_compared = 0

    a_iter, b_iter = iter(a_frames), iter(b_frames)
    sentinel = object()
    while True:
        a = next(a_iter, sentinel)
        b = next(b_iter, sentinel)
        if a is sentinel and b is sentinel:
            break
        if a is not sentinel:
            n_src += 1
        if b is not sentinel:
            n_dst += 1
        if a is sentinel or b is sentinel:
            # One side ran out. Keep draining the other so the counts in the refusal message are
            # the real ones — "output is short by 1" and "output is short by 300" are different
            # bugs and the reader should not have to guess which they have.
            for _ in a_iter:
                n_src += 1
            for _ in b_iter:
                n_dst += 1
            break
        idx = n_src - 1
        if idx % stride:
            continue
        n_compared += 1
        for space in COMPARE_SPACES:
            x = a.to_ndarray(format=space)
            y = b.to_ndarray(format=space)
            if x.shape != y.shape:
                raise ProofFailed(
                    f"FATAL: {dst.name} frame {idx} is {y.shape} in {space} and the source is "
                    f"{x.shape}. Geometry changed; this is not a re-encode of that clip."
                )
            # Equality is the expected answer and ``array_equal`` answers it without allocating a
            # widened copy of two 640x480 arrays 171 625 times. The magnitude is only ever needed
            # to DESCRIBE a failure, so it is computed only when there is one — that widening is
            # 35 % of this function's runtime on the happy path, and the happy path is every clip.
            if np.array_equal(x, y):
                continue
            delta = int(np.abs(x.astype(np.int32) - y.astype(np.int32)).max())
            if delta > maxima[space]:
                maxima[space] = delta
            if first_bad is None:
                first_bad = {"frame": idx, "space": space, "max_abs_delta": delta}

    if n_src != n_dst:
        raise ProofFailed(
            f"FATAL: {dst.name} holds {n_dst} frames and its source holds {n_src}.\n"
            "       An encoder that silently drops its last GOP is precisely the failure this "
            "corpus already lost three\n"
            "       cluster jobs to. The actions are carried over by index, so a short clip pairs "
            "every later label\n"
            "       with a different instant than its pixels — with no decode error anywhere."
        )
    if declared_frames is not None and n_src != declared_frames:
        raise ProofFailed(
            f"FATAL: {src.name} decodes {n_src} frames but the source manifest declares "
            f"{declared_frames}.\n"
            "       Refusing to copy that disagreement into a new manifest — the transcode is not "
            "the place to launder it."
        )
    if first_bad is not None:
        raise ProofFailed(
            f"FATAL: {dst.name} is NOT bit-exact. First difference at frame {first_bad['frame']} "
            f"in {first_bad['space']}, max abs channel delta {first_bad['max_abs_delta']}/255 "
            f"(per-space maxima: {maxima}).\n"
            "       A lossy re-encode sits between GEOM_TOL and the pixels the generator sees, at "
            "a scale of a fraction\n"
            "       of a pixel. That is the objection this whole exercise exists to dissolve, and "
            "it is not dissolved here.\n"
            "       If the pixels were routed through rgb24 anywhere, that is the cause: the "
            "RGB<->YUV chroma round trip\n"
            "       costs 7-10/255 on its own, with any encoder setting."
        )

    return {
        "frames_source": n_src,
        "frames_output": n_dst,
        "frames_compared": n_compared,
        "compare_stride": stride,
        "max_abs_delta": dict(maxima),
        "source_codec": a_info.codec,
        "output_codec": b_info.codec,
        "source_pix_fmt": a_info.pix_fmt,
        "output_pix_fmt": b_info.pix_fmt,
    }


# --------------------------------------------------------------------------------------------
# per-clip worker (top level so ProcessPoolExecutor can pickle it)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipJob:
    episode_id: str
    src: Path
    dst: Path
    declared_frames: int | None
    stride: int
    enc: EncoderSpec
    decoder_name: str
    resume: bool


def process_clip(job: ClipJob) -> dict[str, Any]:
    """Transcode (or resume) one clip and prove it. Never raises: returns a record either way.

    One bad clip in a corpus of 402 is a fact to report beside its neighbours, not a reason to
    abandon a run that has already spent minutes of CPU. The CLI decides what a failed record
    means; this function's only job is to make sure the record is true.
    """
    decoder = DECODERS[job.decoder_name]
    started = time.time()
    rec: dict[str, Any] = {
        "id": job.episode_id,
        "source": str(job.src),
        "source_real": str(job.src.resolve()),
        "output": job.dst.name,
        "ok": False,
        "resumed": False,
        "error": "",
    }
    part = job.dst.with_suffix(job.dst.suffix + ".part")
    try:
        rec["source_sha256"] = sha256_file(job.src)
        rec["source_bytes"] = job.src.stat().st_size

        if job.resume and job.dst.is_file():
            # RESUME RE-VERIFIES; it does not trust the file's existence. A half-written mp4 from
            # an interrupted run is exactly the artefact that looks finished on an `ls` and is
            # short by a GOP, and the frame-count check below is the only thing that can tell.
            try:
                proof = verify_pair(
                    job.src, job.dst, decoder,
                    declared_frames=job.declared_frames, stride=job.stride,
                )
                rec.update(proof)
                rec["output_sha256"] = sha256_file(job.dst)
                rec["output_bytes"] = job.dst.stat().st_size
                rec["resumed"] = True
                rec["ok"] = True
                rec["seconds"] = round(time.time() - started, 3)
                return rec
            except ProofFailed as exc:
                # Not a refusal yet — an existing output that fails its proof is re-encoded, and
                # only a freshly encoded one that fails is a refusal. Recorded so a run that had to
                # redo work says so.
                rec["resume_reverify_failed"] = str(exc).splitlines()[0]

        if part.exists():
            part.unlink()
        encoded = transcode_clip(job.src, part, decoder, job.enc)
        rec.update(encoded)

        proof = verify_pair(
            job.src, part, decoder, declared_frames=job.declared_frames, stride=job.stride
        )
        rec.update(proof)

        # Only now does it get its real name. An unproven file never occupies the path the manifest
        # will point at, so an interrupted run leaves a .part behind rather than a plausible lie.
        os.replace(part, job.dst)
        rec["output_sha256"] = sha256_file(job.dst)
        rec["output_bytes"] = job.dst.stat().st_size
        rec["ok"] = True
    except TranscodeError as exc:
        rec["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - survive and report; the run decides what it means
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if not rec["ok"] and part.exists():
            part.unlink()
        rec["seconds"] = round(time.time() - started, 3)
    return rec


# --------------------------------------------------------------------------------------------
# manifest + artifact
# --------------------------------------------------------------------------------------------


def read_source_manifest(corpus: Path) -> tuple[dict[str, Any], str]:
    """The source manifest and the sha256 of its exact bytes.

    The digest is of the file as it is on disk, not of a re-serialisation of the parsed object:
    the new manifest claims descent from *those bytes*, and json.dumps of a dict that round-tripped
    through Python is a different string as soon as anyone reorders a key.
    """
    p = corpus / "manifest.json"
    if not p.is_file():
        raise TranscodeError(
            f"FATAL: {p} missing. --corpus wants the PR-08 source tree (manifest.json + videos/), "
            "the one scripts/build_pr08_source.py writes."
        )
    raw = p.read_bytes()
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TranscodeError(f"FATAL: {p} is not JSON: {exc}") from exc
    if not manifest.get("episodes"):
        raise TranscodeError(f"FATAL: {p} lists no episodes — there is nothing to transcode.")
    return manifest, hashlib.sha256(raw).hexdigest()


def descendant_manifest(
    source: dict[str, Any],
    *,
    source_manifest_sha256: str,
    source_corpus: Path,
    enc: EncoderSpec,
    decoder: Decoder,
    proof_name: str,
    proof_sha256: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """The new tree's manifest: ``build_pr08_source.py``'s schema, with the codec told the truth.

    A faithful descendant, which means two things that pull in opposite directions. Everything the
    source asserted about the *recording* — resolution, fps, video_key, repo_id, revision, episode
    ids and frame counts — is carried through verbatim, because none of it changed and a
    re-derivation would only introduce a chance to disagree. Everything the source asserted about
    the *bytes on disk* is re-stated from this run: ``codecs`` says h264 (a manifest that still
    said av1 would send the next reader looking for a decoder they do not need), ``materialized``
    says transcode, and a ``transcode`` block carries the ancestry and points at the proof.

    Frame counts come from the VERIFIER, not from the source manifest, even though they must be
    equal — verify_pair refuses when they are not, so by the time this runs the two agree, and
    taking the measured one means this manifest can never claim a length nobody counted.
    """
    by_id = {r["id"]: r for r in records}
    episodes = []
    for ep in source["episodes"]:
        # --limit leaves the tail of the source manifest with no record. Only episodes that were
        # actually transcoded AND proven get an entry: a manifest row for a clip that is not on
        # disk is the dangling reference build_pr08_source refuses on the way in.
        rec = by_id.get(ep["id"])
        if rec is None:
            continue
        episodes.append(
            {"id": ep["id"], "frames": int(rec["frames_output"]), "video": f"videos/{rec['output']}"}
        )

    src_block = dict(source.get("source") or {})
    src_block["codecs"] = ["h264"]
    src_block["materialized"] = MATERIALIZED
    src_block["transcode"] = {
        "of": str(source_corpus),
        "of_manifest_sha256": source_manifest_sha256,
        "from_codecs": sorted((source.get("source") or {}).get("codecs") or []),
        "to_codec": "h264",
        "lossless": True,
        "decoder": f"{decoder.name} {decoder.version}",
        "encoder": f"{enc.codec} crf={enc.crf} preset={enc.preset} threads={enc.threads}",
        "pix_fmt_policy": "output pix_fmt copied from input; frames never converted to RGB",
        "proof": proof_name,
        "proof_sha256": proof_sha256,
        "proven": "every clip: frame count and max abs channel delta 0 in yuv420p and rgb24",
        "why": (
            "Cosmos-Transfer2.5 reads clips with cv2.VideoCapture and the generation venv's cv2 "
            "4.11.0 cannot decode AV1 (jobs 186357, 189585, 189584). Bit-exactness is what keeps "
            "this transcode out from between GEOM_TOL and the pixels the generator sees."
        ),
    }

    out = dict(source)
    out["source"] = src_block
    out["episodes"] = episodes
    return out


def write_json_with_sidecar(path: Path, payload: dict[str, Any]) -> tuple[str, bytes]:
    """Write JSON plus its ``.sha256`` sidecar, and return the digest.

    The same discipline ``build_pr08_source.write_manifest`` and ``measure_geom_tol.write_artifact``
    use. The sidecar is what makes "the file the gate read is the file that was written" checkable
    with ``sha256sum`` instead of trusted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    path.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    (path.parent / (path.name + ".sha256")).write_text(f"{digest}  {path.name}\n")
    return digest, blob


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=10,
        )
    except Exception:  # noqa: BLE001 - provenance is nice to have, never worth failing a run over
        return None
    return out.stdout.strip() or None


def cv2_readback(clips: list[Path], limit: int) -> dict[str, Any]:
    """Ask the LOCAL cv2 whether it can decode the output, and say loudly whose opinion that is.

    This is deliberately not the gate. The cv2 that matters is the one inside the Cosmos-Transfer2.5
    venv on the cluster (4.11.0, avcodec 59.37.100); this workstation runs a different build, and a
    different build's success is worth exactly what job 186357's happy ffprobe was worth. What it
    *can* do is catch a catastrophic container mistake before 6 GB travel over the wire.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {"ran": False, "reason": "cv2 not importable by this interpreter"}

    checked, failed = [], []
    for clip in clips[: limit or len(clips)]:
        cap = cv2.VideoCapture(str(clip))
        try:
            ok, frame = (False, None)
            if cap.isOpened():
                ok, frame = cap.read()
            good = bool(ok) and frame is not None and getattr(frame, "size", 0) > 0 \
                and bool(np.isfinite(frame.astype("float32").mean()))
            checked.append(clip.name)
            if not good:
                failed.append(clip.name)
        finally:
            cap.release()
    return {
        "ran": True,
        "cv2_version": cv2.__version__,
        "interpreter": sys.executable,
        "checked": len(checked),
        "failed": failed,
        "NOT_THE_GATE": (
            "This is THIS workstation's cv2, not the generation venv's 4.11.0/avcodec 59.37.100. "
            "Run scripts/verify_clip_decode.py with the generation venv's interpreter on the "
            "shipped tree before believing the corpus is readable there."
        ),
    }


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def build_jobs(
    corpus: Path,
    out: Path,
    manifest: dict[str, Any],
    *,
    enc: EncoderSpec,
    decoder_name: str,
    stride: int,
    resume: bool,
    limit: int | None,
) -> list[ClipJob]:
    episodes = manifest["episodes"]
    if limit is not None:
        episodes = episodes[:limit]
    jobs = []
    for ep in episodes:
        src = corpus / ep["video"]
        if not src.exists():
            raise TranscodeError(
                f"FATAL: {src} is missing (episode {ep['id']}). The source manifest names a clip "
                "the tree does not hold; a dangling symlink looks like this too."
            )
        jobs.append(
            ClipJob(
                episode_id=ep["id"],
                src=src,
                dst=out / "videos" / Path(ep["video"]).name,
                declared_frames=int(ep["frames"]) if ep.get("frames") is not None else None,
                stride=stride,
                enc=enc,
                decoder_name=decoder_name,
                resume=resume,
            )
        )
    return jobs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", required=True, type=Path,
                    help="PR-08 source tree: manifest.json + videos/.")
    ap.add_argument("--out", required=True, type=Path,
                    help="New tree. Written fresh; existing verified clips in it are resumed.")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1),
                    help="Process pool width. The encoder itself is pinned to --encoder-threads so "
                         "this never changes a byte of the output.")
    ap.add_argument("--preset", default=EncoderSpec.preset,
                    help="libx264 preset. See EncoderSpec for the measurement behind the default.")
    ap.add_argument("--encoder-threads", type=int, default=EncoderSpec.threads,
                    help="Pinned so output digests do not depend on --jobs. 0 means x264's auto.")
    ap.add_argument("--crf", default=LOSSLESS_CRF,
                    help="LEAVE THIS ALONE. Only 0 is lossless; any other value is refused one "
                         "clip later by the verifier. Exposed so the tests can prove that.")
    ap.add_argument("--decoder", choices=("auto", *DECODERS), default="auto",
                    help="'auto' probes each decoder against this corpus and takes the first that "
                         "actually returns a frame.")
    ap.add_argument("--verify-stride", type=int, default=1,
                    help="Compare every k-th frame. 1 (the default) compares ALL of them. Any "
                         "other value is sampling and is stamped SAMPLED in the artifact.")
    ap.add_argument("--limit", type=int, default=None,
                    help="First N episodes; smoke runs only. Stamped in the artifact.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Re-encode every clip even if a verified output already exists.")
    ap.add_argument("--fail-fast", action="store_true",
                    help="Stop at the first refusal instead of reporting all of them.")
    ap.add_argument("--check-cv2", type=int, default=8, metavar="N",
                    help="Read back N outputs with the LOCAL cv2 (0 disables). Not the gate — see "
                         "cv2_readback().")
    args = ap.parse_args(argv)

    if args.verify_stride < 1:
        print("FATAL: --verify-stride must be >= 1.", file=sys.stderr)
        return 2

    try:
        manifest, manifest_sha = read_source_manifest(args.corpus)
    except TranscodeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    enc = EncoderSpec(crf=args.crf, preset=args.preset, threads=args.encoder_threads)

    try:
        probe_clip = args.corpus / manifest["episodes"][0]["video"]
        if not probe_clip.exists():
            raise TranscodeError(
                f"FATAL: {probe_clip} missing — cannot probe a decoder against a corpus whose "
                "first clip is not there."
            )
        decoder = resolve_decoder(args.decoder, probe_clip)
        jobs = build_jobs(
            args.corpus, args.out, manifest,
            enc=enc, decoder_name=decoder.name, stride=args.verify_stride,
            resume=not args.no_resume, limit=args.limit,
        )
    except TranscodeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    (args.out / "videos").mkdir(parents=True, exist_ok=True)

    banner = [
        f"corpus   {args.corpus}  ({len(manifest['episodes'])} episodes, "
        f"codecs {(manifest.get('source') or {}).get('codecs')})",
        f"out      {args.out}",
        f"decoder  {decoder.name} {decoder.version}",
        f"encoder  {enc.codec} crf={enc.crf} preset={enc.preset} threads={enc.threads}",
        f"jobs     {args.jobs}   clips {len(jobs)}",
    ]
    if args.verify_stride > 1:
        banner.append(
            f"!! SAMPLED VERIFICATION: --verify-stride {args.verify_stride}. Only every "
            f"{args.verify_stride}th frame is compared. Frame counts are still full."
        )
    if args.crf != LOSSLESS_CRF:
        banner.append(f"!! --crf {args.crf} IS NOT LOSSLESS. Expect every clip to refuse.")
    for line in banner:
        print(line, file=sys.stderr)

    started = time.time()
    records: list[dict[str, Any]] = []
    if args.jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(process_clip, j): j for j in jobs}
            for n, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                records.append(rec)
                _progress(n, len(jobs), rec)
                if args.fail_fast and not rec["ok"]:
                    for other in futures:
                        other.cancel()
                    break
    else:
        for n, job in enumerate(jobs, 1):
            rec = process_clip(job)
            records.append(rec)
            _progress(n, len(jobs), rec)
            if args.fail_fast and not rec["ok"]:
                break
    wall = time.time() - started

    records.sort(key=lambda r: r["id"])
    good = [r for r in records if r["ok"]]
    bad = [r for r in records if not r["ok"]]
    complete = not bad and len(records) == len(jobs)

    proof: dict[str, Any] = {
        "schema": PROOF_SCHEMA,
        "complete": complete,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git_commit(),
        "argv": sys.argv,
        "host": platform.node(),
        "interpreter": sys.executable,
        "source_corpus": str(args.corpus),
        "source_manifest_sha256": manifest_sha,
        "source_codecs": (manifest.get("source") or {}).get("codecs"),
        "output_tree": str(args.out),
        "decoder": {"name": decoder.name, "version": decoder.version, "note": decoder.note},
        "decoders_unusable_for_transcode": UNUSABLE_FOR_TRANSCODE,
        "libav": _libav_versions(),
        "encoder": dataclasses.asdict(enc),
        "verification": {
            "mode": "FULL" if args.verify_stride == 1 else "SAMPLED",
            "stride": args.verify_stride,
            "spaces": list(COMPARE_SPACES),
            "criterion": "max abs channel delta == 0 in every space, and frame counts equal on "
                         "both sides and against the source manifest",
            "note": (
                "Every frame of every clip was decoded and compared."
                if args.verify_stride == 1
                else f"!! SAMPLED: only every {args.verify_stride}th frame was compared. Frame "
                     "counts are still full on both sides. This artifact does NOT prove the tree "
                     "is bit-exact."
            ),
        },
        "limit": args.limit,
        "wall_seconds": round(wall, 2),
        "clips_total": len(jobs),
        "clips_proven_bit_exact": len(good),
        "clips_refused": len(bad),
        "clips_resumed": sum(1 for r in good if r.get("resumed")),
        "pts_repaired_total": sum(int(r.get("pts_repaired") or 0) for r in records),
        "bytes_source": sum(int(r.get("source_bytes") or 0) for r in good),
        "bytes_output": sum(int(r.get("output_bytes") or 0) for r in good),
        "frames_total": sum(int(r.get("frames_output") or 0) for r in good),
        "clips": records,
    }
    if args.check_cv2 and good:
        proof["cv2_readback"] = cv2_readback(
            [args.out / "videos" / r["output"] for r in good], args.check_cv2
        )

    proof_path = args.out / "TRANSCODE_PROOF.json"
    proof_sha, _ = write_json_with_sidecar(proof_path, proof)
    print(f"\nwrote {proof_path}  sha256 {proof_sha[:16]}...", file=sys.stderr)

    if not complete:
        print(
            f"\nREFUSED: {len(bad)}/{len(jobs)} clips did not survive the round trip.",
            file=sys.stderr,
        )
        for r in bad[:10]:
            print(f"  {r['id']}: {(r['error'] or 'unknown').splitlines()[0]}", file=sys.stderr)
        if len(bad) > 10:
            print(f"  ... and {len(bad) - 10} more (all of them are in the proof)", file=sys.stderr)
        print(
            "\nNo manifest.json was written for the output tree, so it is not usable as a PR-08\n"
            "source and 97_transfer25_restyle.sbatch will refuse it. That is the intended outcome:\n"
            "a tree whose losslessness is unproven must not be able to stand in for the corpus.",
            file=sys.stderr,
        )
        return 1

    if args.limit is not None:
        # A limited tree is a smoke tree. Writing it a manifest that looks like the corpus's is how
        # a 5-episode directory ends up quoted as 402 episodes three weeks from now.
        print(
            f"\n--limit {args.limit} was used: this is a SMOKE tree. Its manifest declares "
            f"{len(good)} episodes,\nnot {len(manifest['episodes'])}. Do not ship it as the corpus.",
            file=sys.stderr,
        )

    new_manifest = descendant_manifest(
        manifest,
        source_manifest_sha256=manifest_sha,
        source_corpus=args.corpus,
        enc=enc,
        decoder=decoder,
        proof_name=proof_path.name,
        proof_sha256=proof_sha,
        records=good,
    )
    mdigest, _ = write_json_with_sidecar(args.out / "manifest.json", new_manifest)

    total_src = proof["bytes_source"] or 1
    print(
        f"\nOK: {len(good)}/{len(jobs)} clips PROVEN bit-exact "
        f"({proof['frames_total']} frames, stride {args.verify_stride})\n"
        f"    {proof['bytes_source']:,} B in -> {proof['bytes_output']:,} B out "
        f"({proof['bytes_output'] / total_src:.2f}x)\n"
        f"    {wall:.1f} s wall, {proof['clips_resumed']} resumed, "
        f"{proof['pts_repaired_total']} pts repaired\n"
        f"    wrote {args.out / 'manifest.json'}  sha256 {mdigest[:16]}...\n"
        f"    NEXT: verify with the CONSUMER's decoder before shipping —\n"
        f"      <generation-venv>/bin/python scripts/verify_clip_decode.py {args.out / 'videos'}",
        file=sys.stderr,
    )
    return 0


def _progress(n: int, total: int, rec: dict[str, Any]) -> None:
    if rec["ok"]:
        mark = "resume" if rec.get("resumed") else "ok"
        detail = f"{rec.get('frames_output')} frames, delta {rec.get('max_abs_delta')}"
    else:
        mark = "REFUSED"
        detail = (rec["error"] or "unknown").splitlines()[0]
    print(f"[{n}/{total}] {rec['id']} {mark}: {detail} ({rec['seconds']}s)", file=sys.stderr)


def _libav_versions() -> dict[str, str]:
    try:
        import av

        return {k: ".".join(str(x) for x in v) for k, v in av.library_versions.items()}
    except Exception:  # noqa: BLE001
        return {}


if __name__ == "__main__":
    raise SystemExit(main())
