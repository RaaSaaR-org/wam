"""Score a generated mp4 against the real future it claims to predict. CPU, no network.

Closes the gap that ``scripts/hf_job_cosmos3_probe.py:408`` left: the generate path writes a clip
and asserts ``st_size > 0``, and nothing has ever compared the imagined future to the recorded
one. Everything this repo says about those clips is a description of what they looked like.

The arms and what they are for live in :mod:`wam.evaluation.video_fidelity`. The short version:
``frozen`` (hold the conditioning frame) is the bar, because T-36 measured the anchored Wan dream
landing 39 % further from the truth than freezing; ``other`` is a retrieval baseline — a
different episode of the same task, at the offset into it that comes CLOSEST to the truth out of
a searched grid — which beats the frozen bar on a good third of this corpus's windows;
``codec_floor`` is the true future through the generated clip's own encoder, which is what
perfect scores. ``truth`` is 0 by construction and checks nothing — an off-by-one window reports
0 there too.

Read ``ratio_to_frozen.model.mean_abs`` against BOTH verdicts. Below 0.9 the clip is closer to
the truth than standing still; that is ``beats_frozen`` and it is not on its own evidence of
prediction, because the searched retrieval arm reaches 0.533 (ep0@271) and 0.593 (ep10@250) of
the frozen bar without predicting anything. ``beats_chance`` is the one that says the clip did
something a lookup into another demo could not. At or above 1.0 standing still would have been
the better prediction. On ``gradient_abs`` read ``ratio_to_frozen.codec_floor.gradient_abs``
first: an mp4 round trip alone costs 0.46 to 0.72 of the frozen bar there depending on how still
the window is, so a model at 0.55 has captured almost nothing.

**The ratio is about the window it was scored on.** The frozen bar it divides by ranges 5.597 to
25.639 ``mean_abs`` over five measured windows of this corpus, and the constant-quality
``codec_floor`` arm ranges 0.074 to 0.296 of it — a 4x spread against a 10 % decision margin. So
two backbones are comparable only on identical ``--episode``, ``--start-frame`` and frame count.
``--compare`` reads a previous report and refuses the comparison when the windows differ.

A good ratio still does not make the clip training data — a pixel metric cannot see that both
priors invent a generic manipulator where the G1's arm should be (``docs/hf_jobs.md``), and that
is the defect that would poison a VLA.

Usage:

    .venv/bin/python scripts/score_generated_video.py \\
        --generated runs/backbone_eval/cosmos3_future.mp4 --episode 0 --start-frame 151 \\
        --context-frames 97 --generated-fps 24

    .venv/bin/python scripts/score_generated_video.py \\
        --generated runs/backbone_eval/wan_future.mp4 --episode 0 --start-frame 151 \\
        --context-frames 97 --generated-fps 24 --out runs/backbone_eval/wan_fidelity.json \\
        --compare runs/backbone_eval/video_fidelity.json

``--start-frame`` is the real frame the FIRST PREDICTED generated frame corresponds to. For a
video-conditioned run that is one past the last context frame, and ``--context-frames`` must be
set to the number of real frames replayed at the head of the clip — scoring those compares the
recording with itself and would drag the result toward zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from wam.evaluation.video_fidelity import (
    DEFAULT_INTERPOLATION,
    INTERPOLATIONS,
    METRICS,
    MODEL_ARM,
    FidelityReport,
    align_indices,
    check_same_window,
    resize_clip,
    score_generated_video,
)

VIDEO_KEY = "observation.images.ego_view"

#: What both generate paths write (``export_to_video`` at ``hf_job_wan_probe.py:944`` and
#: ``hf_job_cosmos3_probe.py:438``). The floor arm has to go through the same encoder the model
#: arm went through or it is measuring some other codec's loss.
CODEC_FLOOR_CODEC = "libx264"

#: The retrieval arm is aligned to the other episode by ABSOLUTE frame index, and episodes of
#: this corpus run 249 to 749 frames, so that index is not a phase. Measured at ep0@271 the same
#: two episodes give other/frozen ``mean_abs`` anywhere from 0.522 to 1.118 depending on it, so a
#: single offset makes ``beats_chance`` a verdict about the offset. Hence the grid: 120 frames is
#: 4 s either side at 30 fps, and the stride is a cost knob — 25 candidates cost ~1.2 s of numpy,
#: ~0.5 s of extra AV1 decoding and ~300 MB held at 480x640 on a 72-frame window, and a finer
#: grid buys about 0.01 of ratio (stride 10 finds 281 at 0.533; the 1-frame optimum nearby is
#: 290 at 0.522).
OTHER_SEARCH_RADIUS = 120
OTHER_SEARCH_STRIDE = 10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generated", required=True, help="the mp4 a backbone produced")
    p.add_argument("--data-dir", default="data/raw/gr00t_apple")
    p.add_argument("--episode", type=int, required=True, help="episode the clip was conditioned on")
    p.add_argument(
        "--start-frame", type=int, required=True, help="real frame the first PREDICTED frame is"
    )
    p.add_argument("--context-frames", type=int, default=0, help="leading real frames replayed")
    p.add_argument("--generated-fps", type=float, default=24.0)
    p.add_argument("--source-fps", type=float, default=None, help="default: meta/info.json")
    p.add_argument(
        "--conditioning-frame",
        type=int,
        default=None,
        help="real frame the frozen arm holds (default: --start-frame minus one)",
    )
    p.add_argument(
        "--other-episode", type=int, default=None, help="chance-level episode (default: +1)"
    )
    p.add_argument(
        "--other-start-frame",
        type=int,
        default=None,
        help="CENTRE of the retrieval arm's offset search (default: --start-frame)",
    )
    p.add_argument(
        "--other-search-radius",
        type=int,
        default=OTHER_SEARCH_RADIUS,
        help=(
            f"search +-N frames around it in steps of {OTHER_SEARCH_STRIDE} and keep the offset "
            "closest to the truth; 0 pins the single offset (default: %(default)s)"
        ),
    )
    p.add_argument(
        "--compare",
        default=None,
        help="a previous report JSON; prints both headlines, refuses a different window",
    )
    p.add_argument(
        "--no-codec-floor",
        action="store_true",
        help="skip the arm that measures what a perfect prediction scores through the codec",
    )
    p.add_argument("--interpolation", choices=INTERPOLATIONS, default=DEFAULT_INTERPOLATION)
    p.add_argument("--out", default="runs/backbone_eval/video_fidelity.json")
    return p.parse_args(argv)


def episode_video(data_dir: Path, episode: int) -> Path:
    return data_dir / "videos" / "chunk-000" / VIDEO_KEY / f"episode_{episode:06d}.mp4"


def episode_lengths(data_dir: Path) -> dict[int, int]:
    """``meta/episodes.jsonl`` -> {episode: frame count}, empty when the mirror has no meta.

    Used only to give :func:`align_indices` something to refuse against before any decoding
    happens: a 590-frame AV1 episode takes seconds to walk, and finding out at the end that the
    window ran past it is the same answer arrived at expensively.
    """
    path = data_dir / "meta" / "episodes.jsonl"
    if not path.is_file():
        return {}
    out: dict[int, int] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[int(row["episode_index"])] = int(row["length"])
    return out


def _decode(path: Path) -> Iterator[np.ndarray]:
    """Yield RGB uint8 frames. cv2 first, imageio's bundled ffmpeg as the AV1 fallback.

    GR00T ships AV1 and the pip ``opencv-python`` wheel's FFmpeg cannot decode it, so cv2 opens
    the file happily and then reads zero frames — the same failure ``convert_lerobot_g1.py:402``
    and ``hf_job_wan_probe.py:817`` already work around. Generated clips are H.264 and cv2 reads
    them, so both paths are exercised by a normal run.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    produced = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            produced += 1
            yield np.ascontiguousarray(bgr[:, :, ::-1])
    finally:
        capture.release()
    if produced:
        return
    import imageio.v3 as iio

    for rgb in iio.imiter(str(path), plugin="FFMPEG"):
        yield np.ascontiguousarray(np.asarray(rgb))


def read_frames(path: Path, indices: Sequence[int] | None = None) -> np.ndarray:
    """Decode ``path`` to uint8 RGB ``[F, H, W, 3]`` at its native geometry.

    With ``indices`` it collects only those frames and stops at the last one, so scoring a
    72-frame window costs ~90 decoded frames rather than a whole episode. A requested frame the
    file does not contain raises instead of being dropped, which is the same refusal
    :func:`align_indices` makes, one layer down where the file length is actually known.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"no such video: {path}")
    if indices is None:
        frames = list(_decode(Path(path)))
    else:
        wanted = {int(i) for i in indices}
        last = max(wanted)
        found: dict[int, np.ndarray] = {}
        for i, rgb in enumerate(_decode(Path(path))):
            if i in wanted:
                found[i] = rgb
            if i >= last:
                break
        missing = sorted(wanted - set(found))
        if missing:
            raise ValueError(f"{path} has no frames {missing[:5]} (asked for {len(wanted)})")
        frames = [found[int(i)] for i in indices]
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames)


def container_fps(path: Path) -> float | None:
    """The fps the mp4 declares, for the report only — never used to align.

    A container's rate is metadata a writer chose; the rate the sampler actually generated at is
    a fact about the run. They agree until someone re-encodes, and at that point silently
    trusting the file would move every index in the comparison.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        value = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    return value if value > 0 else None


def codec_floor_clip(truth: np.ndarray, fps: float) -> np.ndarray | None:
    """The true future through the generated clip's own encoder — what a PERFECT clip scores.

    The model arm is read out of a lossy container and ``frozen`` and ``truth`` come straight off
    the source decode, so the model pays an encode the controls never pay. Measured on ep0, 72
    frames at 24 fps from frame 271 (source 271..360): the byte-exact true future written with
    libx264 and read back scores ``gradient_abs`` 1.111 against the frozen bar's 2.417 — ratio
    0.460 — where the same frames without the round trip score 0.000. ``mean_abs`` costs 0.076 of
    the bar and ``mse`` 0.003, so this is a gradient-metric problem and not a general one. On the
    stillest window measured (ep0@400, 48f) the gradient floor is 0.723.

    Returns ``None`` and says why on stderr instead of aborting: h264 refuses odd frame
    dimensions and the encoder is an optional wheel, and losing the disclosure arm is a smaller
    loss than losing the run. ``alignment.codec_floor_scored`` then reads False and the gradient
    numbers in that report have no floor beside them.
    """
    import tempfile

    frames = np.asarray(truth).astype(np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "codec_floor.mp4"
        try:
            import imageio.v3 as iio

            iio.imwrite(path, frames, fps=fps, codec=CODEC_FLOOR_CODEC)
            decoded = read_frames(path)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            # imageio-ffmpeg absent (ImportError), h264 refusing an odd frame size or the ffmpeg
            # child failing (OSError/RuntimeError), or the round trip decoding to nothing
            # (ValueError, from read_frames).
            print(f"no codec floor ({CODEC_FLOOR_CODEC}): {exc}", file=sys.stderr)
            return None
    if decoded.shape != frames.shape:
        print(
            f"no codec floor: the round trip returned {decoded.shape} for {frames.shape}",
            file=sys.stderr,
        )
        return None
    return decoded


def retrieval_arm(
    video: Path,
    truth: np.ndarray,
    relative: np.ndarray,
    *,
    centre: int,
    other_length: int | None,
    radius: int,
    interpolation: str,
) -> tuple[np.ndarray, int, list[tuple[int, float]]]:
    """The STRONGEST lookup into ``video`` on a grid of start offsets, and the whole curve.

    The arm answers "what would a system get by looking up another demo", so a verdict that
    survives it has to survive the best lookup available, not the one at whichever absolute index
    the source window happened to start at. Measured at ep0@271 against ep1, the grid finds 281
    at 0.533 of the frozen bar where the unsearched default (271) gives 0.697; over the five
    windows in the module docstring the searched bar is stronger in all five (0.533/0.593/0.716/
    0.839/1.295 against 0.697/0.743/0.991/1.027/1.349).

    Ranked on ``mean_abs`` because that is what both verdicts read, and at the comparison
    geometry because that is where they will be read. Selecting the minimum uses the truth, which
    a retrieval system does not have — deliberately: this arm is an upper bound on lookup, so an
    oracle only ever makes ``beats_chance`` harder to earn. It remains a LOWER bound on retrieval
    over the whole corpus, which would search 402 episodes rather than one bounded window of one.

    ``other_length`` may be ``None`` when the mirror has no ``meta/episodes.jsonl``; there is
    then no bound to filter candidates against and the caller gets the single centre offset back,
    which is the pre-search behaviour.
    """
    span = int(relative[-1])
    if radius <= 0 or other_length is None:
        offsets = [centre]
    else:
        steps = radius // OTHER_SEARCH_STRIDE
        offsets = [centre + k * OTHER_SEARCH_STRIDE for k in range(-steps, steps + 1)]
        offsets = [o for o in offsets if o >= 0 and o + span < other_length]
        if not offsets:
            raise ValueError(
                f"{video.name} has {other_length} frames and the comparison window needs "
                f"{span + 1} from its start point — no offset within +-{radius} of {centre} "
                "fits, and a chance level measured over a shorter span is not this window's"
            )
    low, high = min(offsets), max(offsets) + span
    decoded = read_frames(video, range(low, high + 1))

    target = np.asarray(truth, dtype=np.float32)
    curve: list[tuple[int, float]] = []
    for offset in offsets:
        candidate = resize_clip(
            decoded[(offset - low) + relative], truth.shape[1:3], interpolation=interpolation
        )
        curve.append((offset, float(np.abs(candidate.astype(np.float32) - target).mean())))
    best = min(curve, key=lambda row: row[1])[0]
    # The contiguous span from the chosen offset, not the gathered frames: score_generated_video
    # applies the relative offsets itself, so that the retrieval control goes through the same
    # time map as the model arm and cannot be resampled more kindly.
    return decoded[best - low : best - low + span + 1], best, curve


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    generated_path = Path(args.generated)

    source_fps = args.source_fps
    if source_fps is None:
        info_path = data_dir / "meta" / "info.json"
        if not info_path.is_file():
            print(f"no {info_path}; pass --source-fps", file=sys.stderr)
            return 1
        source_fps = float(json.loads(info_path.read_text())["fps"])

    generated = read_frames(generated_path)
    lengths = episode_lengths(data_dir)
    episode_length = lengths.get(args.episode)
    if episode_length is None:
        # Unknown length: let the decoder be the authority instead of guessing a bound. The
        # refusal still happens, in read_frames, just after the walk rather than before it.
        episode_length = args.start_frame + int(generated.shape[0] * source_fps) + 1

    anchor = args.start_frame - 1 if args.conditioning_frame is None else args.conditioning_frame
    indices = align_indices(
        int(generated.shape[0]),
        episode_length,
        generated_fps=args.generated_fps,
        source_fps=source_fps,
        source_start_frame=args.start_frame,
        lead_context_frames=args.context_frames,
    )
    if anchor < 0:
        print(f"conditioning frame {anchor} is before the episode starts", file=sys.stderr)
        return 1

    # One contiguous span covering the frozen arm's anchor and every scored frame, so the report
    # can keep absolute source indices while only ~1.25x the clip length is ever decoded.
    span = range(min(anchor, int(indices[0])), int(indices[-1]) + 1)
    source = read_frames(episode_video(data_dir, args.episode), span)

    # Both the floor and the offset search compare against the truth at the GENERATED geometry,
    # resampled with the kernel the report names, because that is the path the model arm took:
    # generate at that size, encode, decode. Built from the source geometry instead, the floor
    # would carry a resize the model arm never paid for.
    scored_hw = (int(generated.shape[1]), int(generated.shape[2]))
    truth_at_hw = resize_clip(
        source[indices - span.start], scored_hw, interpolation=args.interpolation
    )

    other_episode = args.other_episode if args.other_episode is not None else args.episode + 1
    other_centre = (
        args.other_start_frame if args.other_start_frame is not None else args.start_frame
    )
    other_length = lengths.get(other_episode)
    search_radius = args.other_search_radius
    if search_radius > 0 and other_length is None:
        # Without a length there is no bound to reject candidates against, and a read past the
        # end of the file would refuse the whole run. Reported as the radius actually used, so
        # the JSON cannot claim a search that did not happen.
        print(
            f"no length for episode {other_episode} in meta/episodes.jsonl: the retrieval arm "
            f"stays at the single offset {other_centre}",
            file=sys.stderr,
        )
        search_radius = 0
    relative = indices - args.start_frame
    other, other_start, other_sweep = retrieval_arm(
        episode_video(data_dir, other_episode),
        truth_at_hw,
        relative,
        centre=other_centre,
        other_length=other_length,
        radius=search_radius,
        interpolation=args.interpolation,
    )

    codec_floor = None
    if not args.no_codec_floor:
        codec_floor = codec_floor_clip(truth_at_hw, args.generated_fps)

    # The one clock error the files themselves reveal. The truth arm cannot catch a wrong fps —
    # it is 0 whatever the indices are — and container_fps is deliberately not used to align
    # (a re-encode changes it legitimately), so the disagreement is reported, not resolved.
    container = container_fps(generated_path)
    fps_mismatch = container is not None and abs(container - args.generated_fps) > 0.01
    if fps_mismatch:
        print(
            f"WARNING: {generated_path.name} declares {container:g} fps but --generated-fps is "
            f"{args.generated_fps:g}. Every index below is computed from the latter; if the "
            "sampler really ran at the container's rate the whole window is misaligned.",
            file=sys.stderr,
        )

    report = score_generated_video(
        generated,
        source,
        generated_fps=args.generated_fps,
        source_fps=source_fps,
        source_start_frame=args.start_frame,
        source_offset=span.start,
        lead_context_frames=args.context_frames,
        conditioning_source_frame=anchor,
        other_frames=other,
        codec_floor_frames=codec_floor,
        interpolation=args.interpolation,
        info={
            "generated": str(generated_path),
            "generated_container_fps": container,
            "generated_fps_mismatch": fps_mismatch,
            "generated_hw": [int(generated.shape[1]), int(generated.shape[2])],
            "codec_floor_codec": None if codec_floor is None else CODEC_FLOOR_CODEC,
            "data_dir": str(data_dir),
            "episode": args.episode,
            "episode_length": episode_length,
            "other_episode": other_episode,
            "other_start_frame": other_start,
            "other_offset_centre": other_centre,
            # The curve the verdict is one point of, in mean_abs grey levels — divide by
            # arms.frozen.mean_abs for the ratios the docstrings quote. Published because a
            # single number here cannot show that the same two episodes span 0.522 to 1.118.
            "other_offset_sweep": [[int(offset), value] for offset, value in other_sweep],
            "other_search_radius": int(search_radius),
            "other_search_stride": OTHER_SEARCH_STRIDE,
            "source_hw": [int(source.shape[1]), int(source.shape[2])],
        },
    )

    align = report.alignment
    print(
        f"\n{generated_path.name}: {align.generated_frames} frames at {align.generated_fps:g} fps, "
        f"{align.lead_context_frames} context excluded -> {align.scored_frames} scored"
        f"\nsource episode {args.episode} at {align.source_fps:g} fps, frames "
        f"{align.source_indices[0]}..{align.source_indices[-1]}, frozen holds "
        f"{align.conditioning_source_frame}"
        f"\ncompared at {align.comparison_hw[0]}x{align.comparison_hw[1]}, "
        f"resized={align.resized} ({align.interpolation})"
    )
    weakest = max(other_sweep, key=lambda row: row[1])
    print(
        f"retrieval arm: episode {other_episode}, {len(other_sweep)} offset(s) around "
        f"{other_centre}, using {other_start} (mean_abs {min(v for _, v in other_sweep):.3f}; "
        f"the weakest offset tried, {weakest[0]}, scores {weakest[1]:.3f})\n"
    )
    # Both ratios, because the gradient one is the metric the mp4 round trip contaminates and a
    # reader who only sees mean_abs cannot tell a gradient result from the codec floor.
    header = "  ".join(f"{m:>12}" for m in METRICS)
    print(f"{'arm':>11}  {header}   ratio(mean_abs)  ratio(grad)")
    for name, score in report.arms.items():
        ratios = report.ratio_to_frozen.get(name, {})
        cells = "".join(
            "           —" if ratios.get(metric) is None else f"{ratios[metric]:12.3f}"
            for metric in ("mean_abs", "gradient_abs")
        )
        row = "  ".join(f"{getattr(score, metric):12.4f}" for metric in METRICS)
        print(f"{name:>11}  {row}  {cells}")
    if not align.codec_floor_scored:
        print("\n  no codec_floor arm: this report does not say what a perfect clip would score")
    for name, verdict in report.verdicts.items():
        print(f"\nVERDICT {name}: {verdict}")
    print(
        "  (a pixel metric cannot see whether the arm in frame is a G1 — a passing ratio is not "
        "a training-data verdict)"
    )
    # Said next to the verdicts and not only in a docstring, because the mistake it prevents is
    # made while reading this output: the ratios above divide by a bar that is 4.6x larger on
    # ep2@200 than on ep0@400, and the margin a verdict turns on is 10 %.
    print(
        f"  (these ratios are about episode {args.episode} frames {align.source_indices[0]}.."
        f"{align.source_indices[-1]} at {align.comparison_hw[0]}x{align.comparison_hw[1]} and "
        "about no other window — to rank two backbones, score both here and use --compare)"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2))
    print(f"wrote {out}")

    if args.compare is not None:
        # After the write, so a refused comparison still leaves this run's report on disk.
        previous = FidelityReport.model_validate_json(Path(args.compare).read_text())
        try:
            check_same_window(report, previous, names=(str(out), args.compare))
        except ValueError as exc:
            print(f"\nREFUSED --compare: {exc}", file=sys.stderr)
            return 2
        headline = report.ratio_to_frozen[MODEL_ARM]["mean_abs"]
        before = previous.ratio_to_frozen[MODEL_ARM]["mean_abs"]
        print(
            f"\nsame window as {args.compare}. ratio_to_frozen.model.mean_abs: "
            f"{headline:.3f} here, {before:.3f} there ({headline - before:+.3f}; negative means "
            "this clip is the closer one, and a difference under 0.1 is inside the margin the "
            "verdicts are drawn at)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
