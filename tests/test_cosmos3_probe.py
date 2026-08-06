"""Tests for scripts/hf_job_cosmos3_probe.py — which pixels a ``--generate`` run conditions on.

The failure these pin is silent. The CLI path — which is what an HF Job runs — called the
single-frame loader unconditionally, so ``--gen-cond-frames 97`` sampled from one frame while
the report still said 97 and ``conditioning: video`` never appeared. Nothing raises; only the
clip and the report differ, and only if someone compares them against a Space run of the same
flags.

The second silent failure pinned here is the degenerate length: as many conditioning frames as
output frames leaves the pipeline nothing to denoise, and the mp4 it exports is the input clip
round-tripped through the VAE. Both failures produce a passing report, so what the assertions
below are on is the recorded call and the recorded array — not shapes.

The third is the report itself. It is all a finished job leaves behind besides the mp4: the
process exit code is derived from ``report.failed`` and the ``generate`` info block is the
AC-04 record tying the clip to the seed and settings it came from. Nothing in the run reads it
back, so a check that cannot fail and a settings block that is never written both look exactly
like a healthy run — which is why the report is asserted on here and not just constructed.

Video I/O is stubbed here, so no dataset, no weights and no GPU are involved.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cosmos = _load("hf_job_cosmos3_probe")


def _stub_loaders(monkeypatch: pytest.MonkeyPatch, *, clip_frames: int) -> dict[str, int]:
    """Replace both conditioning loaders with recorders; the caller reads which one ran."""
    calls: dict[str, int] = {}

    def fake_clip(args: Any) -> np.ndarray:
        calls["clip"] = args.gen_cond_frames
        return np.zeros((clip_frames, 8, 8, 3), dtype=np.uint8)

    def fake_frame(args: Any) -> np.ndarray:
        calls["frame"] = args.gen_frame
        return np.zeros((8, 8, 3), dtype=np.uint8)

    monkeypatch.setattr(cosmos, "load_gen_clip", fake_clip)
    monkeypatch.setattr(cosmos.wanprobe, "load_gen_frame", fake_frame)
    return calls


def _conditioning_main_handed_to_generate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: list[str]
) -> np.ndarray:
    """Run ``main --generate`` with sampling stubbed out; return the array it would have used."""
    seen: dict[str, np.ndarray] = {}

    def fake_generate(args: Any, image_rgb: np.ndarray, report: Any) -> dict[str, Any]:
        seen["image"] = image_rgb
        return {}

    monkeypatch.setattr(cosmos, "generate_future", fake_generate)
    argv = [
        "--data-dir", str(tmp_path),
        "--generate",
        "--out", str(tmp_path / "report.json"),
        *extra,
    ]  # fmt: skip
    assert cosmos.main(argv) == 0
    return seen["image"]


def test_asking_for_video_conditioning_on_the_cli_path_reaches_the_clip_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_loaders(monkeypatch, clip_frames=9)
    image = _conditioning_main_handed_to_generate(
        monkeypatch, tmp_path, ["--gen-cond-frames", "9", "--gen-frame", "150"]
    )
    # the clip loader ran and the single-frame loader did NOT: the whole bug was that
    # `frame` appeared here instead, one flag silently doing nothing
    assert calls == {"clip": 9}
    assert image.ndim == 4 and image.shape[0] == 9


def test_a_run_that_asks_for_one_frame_still_takes_the_single_frame_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default is image mode, and it must keep using cv2-then-imageio ``load_gen_frame`` —
    every recorded single-frame future in runs/ came through that reader."""
    calls = _stub_loaders(monkeypatch, clip_frames=9)
    image = _conditioning_main_handed_to_generate(monkeypatch, tmp_path, ["--gen-frame", "150"])
    assert calls == {"frame": 150}
    assert image.ndim == 3


def _stub_imiter(monkeypatch: pytest.MonkeyPatch, count: int = 20) -> dict[str, Any]:
    """Stand in for the FFMPEG decode and record what it was asked to open."""
    frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(count)]
    read: dict[str, Any] = {}

    def fake_imiter(path: str, plugin: str | None = None):
        read["path"] = path
        read["plugin"] = plugin
        return iter(frames)

    monkeypatch.setattr("imageio.v3.imiter", fake_imiter)
    return read


def test_the_clip_ends_at_the_requested_frame_and_reaches_back_cond_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--gen-frame is the reference frame in image mode, so the clip has to *end* there: a
    window that ran forward instead would condition on frames the image-mode run never saw and
    the two modes would no longer be comparable."""
    _stub_imiter(monkeypatch)

    args = cosmos.parse_args(
        ["--data-dir", str(tmp_path), "--gen-frame", "10", "--gen-cond-frames", "5"]
    )
    clip = cosmos.load_gen_clip(args)
    assert clip.shape == (5, 4, 4, 3)
    assert [int(f[0, 0, 0]) for f in clip] == [6, 7, 8, 9, 10]

    short = cosmos.parse_args(
        ["--data-dir", str(tmp_path), "--gen-frame", "3", "--gen-cond-frames", "9"]
    )
    with pytest.raises(ValueError, match="no 9-frame history"):
        cosmos.load_gen_clip(short)

    # The mirror case: the window starts inside the episode but the episode ends inside the
    # window. A short read must raise rather than return a partial clip — everything downstream
    # (cond_frames in the report, the clean-latent count, the degenerate-length guard) is
    # counted off this array, so a silently shorter one is a run misdescribed everywhere at once.
    _stub_imiter(monkeypatch, count=8)
    truncated = cosmos.parse_args(
        ["--data-dir", str(tmp_path), "--gen-frame", "10", "--gen-cond-frames", "5"]
    )
    with pytest.raises(ValueError, match="read 2 frames"):
        cosmos.load_gen_clip(truncated)


def test_the_clip_is_read_from_the_ego_view_of_the_requested_episode_via_the_ffmpeg_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Video mode has to open the same file the image-mode control opens.

    ``wanprobe.load_gen_frame`` reads ``ego_view`` of ``--gen-episode``; if this loader drifted
    to another camera or indexed the path by ``--gen-frame``, the two conditioning modes would
    be looking at different footage and the comparison this whole switch exists to enable would
    be meaningless — and nothing would raise. ``plugin="FFMPEG"`` is not decoration: GR00T ships
    AV1 and imageio's default plugin does not decode it.
    """
    read = _stub_imiter(monkeypatch)
    args = cosmos.parse_args(
        ["--data-dir", str(tmp_path), "--gen-episode", "7", "--gen-frame", "10",
         "--gen-cond-frames", "5"]
    )  # fmt: skip
    cosmos.load_gen_clip(args)

    assert Path(read["path"]) == (
        tmp_path / "videos" / "chunk-000" / "observation.images.ego_view" / "episode_000007.mp4"
    )
    assert read["plugin"] == "FFMPEG"


class _FakeVae:
    def __init__(self) -> None:
        self.tiled = False

    def enable_tiling(self) -> None:
        self.tiled = True


class _FakePipe:
    """Records the sampling call; no DiT, no VAE, no GPU."""

    def __init__(self) -> None:
        self.vae = _FakeVae()
        self.call: dict[str, Any] | None = None
        self.tiled_at_call = False

    def __call__(self, **kwargs: Any) -> Any:
        self.call = kwargs
        # sampled *and* decoded inside this call, so tiling switched on afterwards is too late
        self.tiled_at_call = self.vae.tiled
        return SimpleNamespace(video=[np.zeros((8, 8, 3), dtype=np.uint8)])


# Every sampling flag gets a value distinct from every other, and the frame is deliberately
# NOT square: a fixture of equal numbers hides a swapped height/width and a dropped resize.
GEN_PROMPT = "pick up the apple and place it on the plate"
GEN_FLAGS = ["--gen-num-frames", "17", "--gen-steps", "1", "--gen-height", "32",
             "--gen-width", "64", "--gen-guidance", "3.5", "--gen-prompt", GEN_PROMPT]  # fmt: skip


def _gen_args(tmp_path: Path, extra: list[str]) -> Any:
    return cosmos.parse_args(
        ["--data-dir", str(tmp_path), "--device", "cpu",
         "--gen-out", str(tmp_path / "future.mp4"), *GEN_FLAGS, *extra]
    )  # fmt: skip


def _write_mp4(video: Any, path: str, fps: int) -> None:
    Path(path).write_bytes(b"mp4")


def _generate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    image: np.ndarray,
    extra: list[str],
    export: Any = _write_mp4,
) -> tuple[dict[str, Any], _FakePipe, Any]:
    """Run ``generate_future`` against the fake pipe; hand back what the run left behind.

    The report is returned as well as the info dict because it is the only artifact the two
    real callers read: ``finalize`` turns ``report.failed`` into the process exit code and the
    Space's generate tab turns it into the DONE/FAILED verdict.
    """
    import diffusers.utils

    pipe = _FakePipe()
    report = cosmos.smoke.Report()
    monkeypatch.setattr(cosmos, "load_pipeline", lambda args, device: pipe)
    monkeypatch.setattr(diffusers.utils, "export_to_video", export)
    return cosmos.generate_future(_gen_args(tmp_path, extra), image, report), pipe, report


def test_the_report_counts_the_conditioning_frames_the_pipeline_actually_saw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nine frames handed over while the flag says five: both the report and the latent
    conditioning index list have to follow the array, not the request."""
    # each frame a distinct constant, so the saved start frame is identifiable
    clip = np.stack([np.full((16, 16, 3), 10 * i, dtype=np.uint8) for i in range(9)])
    info, pipe, _ = _generate(monkeypatch, tmp_path, clip, ["--gen-cond-frames", "5"])

    assert info["conditioning"] == "video"
    assert info["cond_frames"] == 9
    assert pipe.call is not None
    assert len(pipe.call["video"]) == 9
    # 9 pixel frames -> 3 clean latent frames; from the flag it would have been 2, and the
    # pipeline would treat one more latent frame as clean than the clip can fill
    assert pipe.call["condition_frame_indexes_vision"] == (0, 1, 2)
    # the png filed next to the mp4 is the reference frame the clip ends on, not the one it
    # starts on — it is what a video-conditioned run gets compared against
    from PIL import Image

    saved = np.asarray(Image.open(info["start_png"]))
    assert saved.min() == saved.max() == 80


def test_the_sampling_call_asks_for_the_geometry_and_the_prompt_the_run_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole payload, not just the conditioning.

    A swapped ``prompt``/``negative_prompt`` samples the quality-negative as the instruction, a
    sampling ``fps`` that drifts from the reported one misaligns frame-distance scoring against
    the 30 fps corpus, and a transposed height/width hands the model a different aspect ratio
    than the report claims. None of those raise and all of them invalidate a backbone
    comparison, so the recorded call is compared field by field.
    """
    clip = np.stack([np.full((16, 16, 3), 10 * i, dtype=np.uint8) for i in range(9)])
    info, pipe, _ = _generate(monkeypatch, tmp_path, clip, ["--gen-cond-frames", "9"])

    assert pipe.call is not None
    assert pipe.call["video"][0].size == (64, 32)  # PIL is (width, height)
    payload = {k: v for k, v in pipe.call.items() if k not in ("video", "generator")}
    assert payload == {
        "prompt": GEN_PROMPT,
        "negative_prompt": cosmos.COSMOS_NEGATIVE,
        "condition_frame_indexes_vision": (0, 1, 2),
        "num_frames": 17,
        "height": 32,
        "width": 64,
        "fps": 24,
        "num_inference_steps": 1,
        "guidance_scale": 3.5,
        "enable_safety_check": False,
    }
    assert (info["prompt"], info["fps"], info["size"]) == (GEN_PROMPT, 24, [32, 64])


def test_the_generator_is_seeded_with_the_number_the_report_attributes_the_clip_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--gen-seed`` is what makes a recorded clip re-samplable from its own report.

    A generator seeded with anything other than the reported number is invisible in the clip
    and in the log. It surfaces only when someone re-runs the recorded settings, gets different
    pixels, and finds that the frame distance a dream was scored with (T-36) cannot be
    regenerated — the same flag-says-one-thing/run-does-another failure the ``cond_frames``
    fix in this module was written for.
    """
    clip = np.stack([np.full((16, 16, 3), 10 * i, dtype=np.uint8) for i in range(9)])
    info, pipe, _ = _generate(
        monkeypatch, tmp_path, clip, ["--gen-cond-frames", "9", "--gen-seed", "7"]
    )

    assert pipe.call is not None
    # 7 is the default of no other flag and equals no other value in GEN_FLAGS, so a generator
    # wired to the wrong argument cannot read as 7 by coincidence
    assert pipe.call["generator"].initial_seed() == 7
    assert info["seed"] == 7


def test_the_run_leaves_a_report_holding_the_mp4_check_and_the_settings_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report is the run's only output besides the clip, and nothing re-reads it in-process.

    ``generate.wrote_mp4`` is the only check a ``--generate`` run registers and ``finalize``
    turns ``report.failed`` into the exit code, so that one entry is the whole difference
    between a job that reports success and one that reports failure. ``info["generate"]`` is
    the record that ties the stored mp4 to the prompt, conditioning and seed it came from; an
    mp4 without it cannot be traced back to a run (AC-04).
    """
    clip = np.stack([np.full((16, 16, 3), 10 * i, dtype=np.uint8) for i in range(9)])
    info, _, report = _generate(monkeypatch, tmp_path, clip, ["--gen-cond-frames", "9"])

    assert [c["name"] for c in report.checks] == ["generate.load", "generate.wrote_mp4"]
    assert report.failed == []
    # what the caller gets back and what the run records have to be the same block: the other
    # tests here assert on the return value, and that is only worth anything if it is what the
    # report json ends up holding
    assert report.info["generate"] == info


def test_an_export_that_leaves_no_playable_file_fails_the_run_instead_of_passing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both halves of the mp4 check, because the exit code means nothing without them.

    ``export_to_video`` is the one step here that depends on the host's ffmpeg build — the same
    codec surface that already forced imageio on the read side for GR00T's AV1 — and when it
    fails it leaves either no file or an empty one rather than raising. A job that prints
    ALL CHECKS PASSED and exits 0 with no clip is worse than one that crashes: the absence is
    found much later, by whoever goes looking for the video.
    """
    clip = np.stack([np.full((16, 16, 3), 10 * i, dtype=np.uint8) for i in range(9)])

    def write_empty(video: Any, path: str, fps: int) -> None:
        Path(path).write_bytes(b"")

    def write_nothing(video: Any, path: str, fps: int) -> None:
        return None

    _, _, empty = _generate(monkeypatch, tmp_path, clip, ["--gen-cond-frames", "9"], write_empty)
    assert empty.failed == ["generate.wrote_mp4"]

    _, _, absent = _generate(
        monkeypatch,
        tmp_path,
        clip,
        ["--gen-cond-frames", "9", "--gen-out", str(tmp_path / "never_encoded.mp4")],
        write_nothing,
    )
    assert absent.failed == ["generate.wrote_mp4"]


def test_the_vae_is_switched_to_tiled_decoding_before_the_sampler_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not a preference: the reason is stated on the line itself, and it is a memory ceiling.

    The fp32 VAE decode lands on top of the ~32 GB bf16 transformer that is already resident,
    and tiling is what keeps the peak inside a 48 GB card. What a stubbed pipe can check is
    only that the call happens and that it happens *before* sampling — the decode runs inside
    the pipeline call, so tiling switched on afterwards would be a no-op. The failure mode
    without it is a CUDA OOM at the end of sampling, i.e. after the full weight load has
    already been paid for.
    """
    clip = np.stack([np.full((16, 16, 3), 10 * i, dtype=np.uint8) for i in range(9)])
    _, pipe, _ = _generate(monkeypatch, tmp_path, clip, ["--gen-cond-frames", "9"])

    assert pipe.tiled_at_call


def test_a_clip_as_long_as_the_future_it_conditions_is_refused_before_the_weights_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As many conditioning frames as output frames means nothing is denoised.

    Every latent frame is then clean conditioning, the velocity is masked to zero everywhere and
    the exported mp4 is a VAE round-trip of the real input clip. Nothing raises inside the
    pipeline — ``check_inputs`` only rejects an index past the latent timeline, which is
    ``cond > num``, not ``cond == num`` — and the report cannot tell the two apart: it records
    ``num_frames`` and ``cond_frames`` but not the difference. Scored the way T-36 scored the
    dream, such a clip matches ground truth almost exactly because it *is* ground truth, and
    would read as the video prior finally beating the frozen-frame control.
    """

    def explode(args: Any, device: str) -> Any:
        raise AssertionError("32 GB of weights is a long wait for an argument error")

    monkeypatch.setattr(cosmos, "load_pipeline", explode)
    clip = np.zeros((17, 16, 16, 3), dtype=np.uint8)  # --gen-num-frames is 17
    with pytest.raises(ValueError, match="leave nothing to predict"):
        cosmos.generate_future(
            _gen_args(tmp_path, ["--gen-cond-frames", "17"]), clip, cosmos.smoke.Report()
        )
    # and it counts the frames handed over, not the frames asked for — a guard that trusted the
    # flag would wave through exactly the mismatch the rest of this module exists to prevent
    with pytest.raises(ValueError, match="leave nothing to predict"):
        cosmos.generate_future(
            _gen_args(tmp_path, ["--gen-cond-frames", "9"]), clip, cosmos.smoke.Report()
        )


@pytest.mark.parametrize(
    "flags,clip_frames,message",
    [
        (["--gen-num-frames", "18"], 9, "gen-num-frames must satisfy"),
        (["--gen-cond-frames", "8"], 8, "gen-cond-frames must satisfy"),
        (["--gen-height", "33"], 9, "multiples of 32"),
        (["--gen-width", "48"], 9, "multiples of 32"),
    ],
)
def test_a_length_or_frame_size_the_latent_grid_cannot_hold_is_refused_before_the_weights_load(
    flags: list[str],
    clip_frames: int,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each of these mis-describes a run rather than failing it, so each has to raise here.

    ``--gen-cond-frames 8`` is the sharp one. The clip is 8 pixel frames, the index list handed
    to the pipeline covers ``(8-1)//4+1 = 2`` latent frames, and 8 frames do not fill a whole
    number of the VAE's 4-frame temporal groups — so either a latent frame holding real
    conditioning pixels is left to the denoise loop, or one holding none is pinned as clean.
    Which of the two depends on the encoder's rounding, and neither is a run comparable with a
    9-frame one; the degenerate-length guard stays quiet throughout (2 latent frames of 5).
    A 4k+1 length is the only one where pixel frames and latent frames agree, and the 32-pixel
    grid is the same constraint spatially. All of them are cheap to check, and the alternative
    is discovering the argument after 32 GB of weights have loaded.
    """

    def explode(args: Any, device: str) -> Any:
        raise AssertionError("32 GB of weights is a long wait for an argument error")

    monkeypatch.setattr(cosmos, "load_pipeline", explode)
    clip = np.zeros((clip_frames, 16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match=message):
        cosmos.generate_future(_gen_args(tmp_path, flags), clip, cosmos.smoke.Report())


def test_an_image_mode_run_reports_one_conditioning_frame_whatever_the_flag_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag is a request; ``cond_frames`` is the record. They came apart once already, and
    a report that echoes the request is exactly what hid it."""
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    info, pipe, _ = _generate(monkeypatch, tmp_path, frame, ["--gen-cond-frames", "9"])

    assert info["conditioning"] == "image"
    assert info["cond_frames"] == 1
    assert pipe.call is not None and "video" not in pipe.call
    # the single conditioning frame goes in at the requested geometry too, or the two modes
    # are no longer sampling the same picture
    assert pipe.call["image"].size == (64, 32)
