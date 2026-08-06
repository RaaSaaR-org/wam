"""Scoring a generated clip against the real future (T-38).

Almost none of this is about the metrics. Mean absolute difference is four characters of numpy
and it is not where a fidelity number goes wrong. It goes wrong in the two conversions that make
the two clips comparable at all — 24 fps against 30 fps, the model's grid against the camera's —
and in the leading frames of a video-conditioned clip, which are the recording replayed and
score near zero against themselves. Each of those failures returns a finite, plausible,
publishable number, so each gets a test that fails when the guard is removed.

The metrics get exact-value tests anyway, for a reason an earlier version of this file got
wrong: every fixture here used a spatially constant difference, and on a constant difference
``mean|d|`` equals ``|mean(d)|`` equals ``sqrt(mse)``. Three different reductions all passed.
So the differences below change sign across the frame, take two values, and sit in one colour
channel, and the gradient numbers are asserted exactly rather than as ``> 0``.

Synthetic arrays throughout, except the CLI tests, which write throwaway mp4s into ``tmp_path``.
Real clips are lossy and would let an off-by-one hide inside codec noise — except where the
codec loss IS the measurement, which is what the ``codec_floor`` arm exists for.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from wam.evaluation.dream import FREEZE_MARGIN
from wam.evaluation.video_fidelity import (
    DEFAULT_INTERPOLATION,
    INTERPOLATIONS,
    METRICS,
    VERDICT_BEATS_CHANCE,
    VERDICT_CHANCE_NOT_MEASURED,
    VERDICT_NO_BETTER,
    VERDICT_NO_BETTER_THAN_CHANCE,
    VERDICT_PREDICTS,
    align_indices,
    check_same_window,
    check_truth_is_zero,
    frame_metrics,
    frozen_control,
    gradient_distance,
    resize_clip,
    score_generated_video,
)

HEIGHT, WIDTH = 12, 16


def moving_episode(length: int, *, seed: int = 0, hw: tuple[int, int] = (HEIGHT, WIDTH)):
    """A synthetic episode where every frame differs from every other.

    A bright square walks across a noise background, one pixel per frame. The walk is what makes
    an index error visible: on a static clip the wrong frame and the right frame are the same
    picture, which is exactly the regime T-36 found this project measuring in by accident.
    """
    height, width = hw
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 60, (length, height, width, 3), dtype=np.uint8)
    for t in range(length):
        row, col = 2 + (t % (height - 4)), 1 + (t % (width - 3))
        frames[t, row : row + 2, col : col + 2] = 240
    return frames


class TestPixelMetrics:
    def test_a_clip_scores_zero_against_itself_on_every_metric(self):
        clip = moving_episode(6)
        assert frame_metrics(clip, clip) == dict.fromkeys(METRICS, 0.0)

    def test_the_mean_absolute_difference_is_in_grey_levels_not_unit_scale(self):
        """The convention ``docs/hf_jobs.md`` already quotes ("mean abs pixel diff 2.5/255")."""
        a = np.full((3, 4, 4, 3), 200, dtype=np.uint8)
        b = np.full((3, 4, 4, 3), 190, dtype=np.uint8)
        assert frame_metrics(a, b)["mean_abs"] == pytest.approx(10.0)

    def test_a_prediction_wrong_in_both_directions_does_not_cancel_to_nothing(self):
        """The headline metric's reduction. ``|mean(x - y)|`` instead of ``mean|x - y|`` passes
        every constant-difference fixture in this file and reports 0.0 for a clip wrong by 40
        grey levels at every single pixel — and zero-mean high-frequency error is precisely the
        shape a VAE-decoded frame comes back with, so that mutation would score the actual
        failure mode as perfect."""
        truth = np.full((2, 4, 4, 3), 128, dtype=np.uint8)
        checker = (np.indices((4, 4)).sum(axis=0) % 2).astype(bool)
        model = np.broadcast_to(
            np.where(checker[None, :, :, None], 168, 88).astype(np.uint8), (2, 4, 4, 3)
        )
        signed_mean = float(np.mean(model.astype(np.int32) - truth.astype(np.int32)))
        assert signed_mean == 0.0, "precondition: the error cancels exactly if it is not absolute"
        assert frame_metrics(model, truth)["mean_abs"] == pytest.approx(40.0)

    def test_the_mse_is_the_square_of_the_grey_level_difference(self):
        a = np.full((3, 4, 4, 3), 200, dtype=np.uint8)
        b = np.full((3, 4, 4, 3), 190, dtype=np.uint8)
        assert frame_metrics(a, b)["mse"] == pytest.approx(100.0)

    def test_the_mse_is_the_mean_of_the_squares_and_not_the_square_of_the_mean(self):
        """The two are identical on any constant difference, which is what every other mse
        fixture here uses. On a two-valued one they part: half the pixels off by 2 and half by 8
        give mean|d| = 5 (so mean|d|^2 = 25) and a mean of squares of 34. mse is the
        outlier-sensitive metric or it is a second name for mean_abs."""
        truth = np.full((1, 4, 4, 3), 100, dtype=np.uint8)
        model = truth.copy()
        model[:, :2] = 102
        model[:, 2:] = 108
        scores = frame_metrics(model, truth)
        assert scores["mean_abs"] == pytest.approx(5.0)
        assert scores["mse"] == pytest.approx(34.0)

    @pytest.mark.parametrize("channel", [0, 1, 2])
    def test_a_difference_in_one_colour_channel_alone_is_still_scored(self, channel):
        """All three channels or two thirds of the image is silently not scored. The defect the
        qualitative runs actually found — a green-white tube where the G1's arm should be
        (``docs/hf_jobs.md``) — is a chroma defect, invisible to a luminance-only or
        red-channel-only reduction."""
        truth = np.full((2, 4, 4, 3), 100, dtype=np.uint8)
        shifted = truth.copy()
        shifted[..., channel] = 130
        assert frame_metrics(shifted, truth)["mean_abs"] == pytest.approx(10.0)

    def test_a_backbones_zero_to_one_floats_land_on_the_same_scale_as_a_recording(self):
        """``decode_video`` returns [0, 1] and the corpus is uint8; both must read as 0-255."""
        recorded = np.full((2, 4, 4, 3), 255, dtype=np.uint8)
        decoded = np.ones((2, 4, 4, 3), dtype=np.float32)
        assert frame_metrics(recorded, decoded)["mean_abs"] == pytest.approx(0.0)

    def test_a_uniform_brightness_shift_moves_the_pixel_metrics_but_not_the_gradient_one(self):
        """The reason the third metric exists. A VAE round-trip routinely returns the scene a few
        grey levels brighter; a scorer that ranked backbones on exposure ranks them on nothing."""
        clip = moving_episode(5).astype(np.float32)
        shifted = clip + 7.0
        scores = frame_metrics(shifted / 255.0, clip / 255.0)
        assert scores["mean_abs"] == pytest.approx(7.0, abs=1e-3)
        assert scores["gradient_abs"] == pytest.approx(0.0, abs=1e-4)

    def test_an_edge_that_moved_does_show_up_in_the_gradient_metric(self):
        """...so the brightness invariance above is not the metric being blind to everything."""
        a = np.zeros((1, 6, 6, 3), dtype=np.uint8)
        a[0, :, :3] = 200
        b = np.zeros((1, 6, 6, 3), dtype=np.uint8)
        b[0, :, :4] = 200  # the same edge, one column to the right
        assert gradient_distance(a, b) > 10.0

    def test_an_edge_that_moved_vertically_is_seen_too(self):
        """Both spatial directions are summed, and only this catches it: a row-constant image has
        no horizontal first difference at all, so a horizontal-only gradient scores a stripe that
        slid up the frame as a perfect prediction. The G1's arm enters from the side and lifts."""
        a = np.zeros((1, 6, 6, 3), dtype=np.uint8)
        a[0, :3, :] = 200
        b = np.zeros((1, 6, 6, 3), dtype=np.uint8)
        b[0, :4, :] = 200  # the same edge, one row down
        assert np.abs(np.diff(a, axis=-2)).max() == 0, "precondition: nothing varies horizontally"
        assert gradient_distance(a, b) > 10.0

    def test_the_gradient_distance_of_a_hand_checkable_edge_is_exactly_one_grey_level(self):
        """The only absolute gradient number in this file, and it has to be one, because the
        metric is reported on the 0-255 scale into ``runs/backbone_eval/`` and compared against
        ``dream.json``'s motion figures across report versions.

        Two frames of 2x3 pixels, an 8-level edge down the left column of the second frame. The
        horizontal first differences disagree by 8 on 6 of 24 elements -> 2.0; the two rows are
        identical so the vertical term is 0; the two terms are averaged -> 1.0. Each of the three
        choices in that sentence is a mutation that survives everything else here: ``.max()``
        gives 4.0, dropping the ``/2`` gives 2.0, scoring only the first frame gives 0.0.
        """
        a = np.zeros((2, 2, 3, 3), dtype=np.uint8)
        b = a.copy()
        b[1, :, 0, :] = 8
        assert gradient_distance(a, b) == pytest.approx(1.0)

    def test_a_clip_that_is_right_at_the_first_frame_and_wrong_at_the_last_is_charged_for_it(self):
        """Scope, not shape. A 72-frame prediction that is right at t=0 and diverges afterwards
        must not score like a perfect one; a difference taken on the first frame only returns 0
        here and the reported number is then a claim about 71 frames nobody looked at."""
        right_then_wrong = moving_episode(8)
        truth = right_then_wrong.copy()
        truth[-1] = moving_episode(8, seed=7)[-1]
        assert np.array_equal(right_then_wrong[0], truth[0]), "precondition: t=0 is correct"
        assert gradient_distance(right_then_wrong, truth) > 0.0

    def test_the_gradient_is_a_difference_of_gradients_not_of_their_magnitudes(self):
        """An edge of the same strength somewhere else is still an error. Comparing |grad a| with
        |grad b| would cancel a mirrored scene against the original and call it perfect."""
        a = np.zeros((1, 4, 6, 3), dtype=np.uint8)
        a[0, :, :3] = 200
        mirrored = a[:, :, ::-1].copy()
        assert gradient_distance(a, mirrored) > 0.0

    def test_mismatched_shapes_are_refused_rather_than_broadcast(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            frame_metrics(np.zeros((2, 4, 4, 3), np.uint8), np.zeros((3, 4, 4, 3), np.uint8))

    def test_a_clip_thinner_than_two_pixels_has_no_gradient(self):
        with pytest.raises(ValueError, match="2x2"):
            gradient_distance(np.zeros((1, 1, 4, 3), np.uint8), np.zeros((1, 1, 4, 3), np.uint8))

    def test_a_nonzero_truth_arm_means_a_non_finite_pixel_and_not_a_misalignment(self):
        """The guard is real but narrow, and it used to be documented as something it is not.
        An array does not differ from itself, so the only thing that can make this fire is a NaN
        or an infinity among the source pixels — which propagates into every arm and would
        otherwise be published as a finite-looking distance."""
        check_truth_is_zero(dict.fromkeys(METRICS, 0.0))
        with pytest.raises(ValueError, match="non-finite pixel"):
            check_truth_is_zero({"mean_abs": float("nan"), "mse": 0.0, "gradient_abs": 0.0})
        with pytest.raises(ValueError, match="non-finite pixel"):
            check_truth_is_zero({"mean_abs": 0.0, "mse": 1e-12, "gradient_abs": 0.0})


class TestTimeAlignment:
    """Comparing frame k with frame k drifts one real frame every four generated ones — 18 over
    a 72-frame prediction, most of a reach. It never raises; it just returns a worse number."""

    def test_a_24_fps_clip_maps_onto_30_fps_source_frames_by_time_not_by_index(self):
        indices = align_indices(
            5, 200, generated_fps=24.0, source_fps=30.0, source_start_frame=100
        )
        # 0, 1.25, 2.5, 3.75, 5.0 real frames after the start; nearest, ties up.
        assert indices.tolist() == [100, 101, 103, 104, 105]

    def test_equal_frame_rates_map_one_to_one(self):
        indices = align_indices(4, 50, generated_fps=30.0, source_fps=30.0, source_start_frame=7)
        assert indices.tolist() == [7, 8, 9, 10]

    def test_assuming_the_rates_match_would_compare_different_frames(self):
        """The defect itself, pinned: the same clip against the same episode, one wrong fps."""
        by_time = align_indices(9, 99, generated_fps=24.0, source_fps=30.0, source_start_frame=0)
        as_if_equal = align_indices(
            9, 99, generated_fps=24.0, source_fps=24.0, source_start_frame=0
        )
        assert by_time.tolist() != as_if_equal.tolist()
        assert int(by_time[-1]) - int(as_if_equal[-1]) == 2

    def test_the_clock_starts_at_the_first_scored_frame_not_at_the_start_of_the_clip(self):
        """A video-conditioned clip replays real frames into a 24 fps container, so their spacing
        in the generated timeline is not their spacing in the recording. Anchoring the origin at
        the replay/prediction boundary keeps that distortion out of every predicted frame."""
        indices = align_indices(
            7, 200, generated_fps=24.0, source_fps=30.0, source_start_frame=100,
            lead_context_frames=3,
        )
        assert indices.tolist() == [100, 101, 103, 104]

    def test_a_window_running_past_the_end_of_the_episode_is_refused(self):
        with pytest.raises(ValueError, match="refusing to truncate"):
            align_indices(20, 30, generated_fps=24.0, source_fps=30.0, source_start_frame=25)

    def test_a_window_starting_before_the_supplied_span_is_refused(self):
        with pytest.raises(ValueError, match="refusing to truncate"):
            align_indices(
                4, 30, generated_fps=30.0, source_fps=30.0, source_start_frame=5, source_offset=10
            )

    def test_a_window_whose_first_frame_is_early_is_refused_even_when_its_last_is_inside(self):
        """Both ends are bounded, and only this pins the near end. The case above starts at 5 and
        runs to 8, so its LAST index is outside the span too and a tail-only guard still fires;
        this one straddles the boundary. Without the near-end half, ``indices - source_offset``
        goes negative, numpy wraps it to the end of the array, and the report is about the tail
        of the episode while naming frames 8..11."""
        with pytest.raises(ValueError, match="refusing to truncate"):
            align_indices(
                4, 30, generated_fps=30.0, source_fps=30.0, source_start_frame=8, source_offset=10
            )

    def test_an_offset_span_still_reports_absolute_source_indices(self):
        indices = align_indices(
            3, 30, generated_fps=30.0, source_fps=30.0, source_start_frame=40, source_offset=38
        )
        assert indices.tolist() == [40, 41, 42]

    @pytest.mark.parametrize("generated_fps,source_fps", [(0.0, 30.0), (24.0, 0.0), (-24.0, 30.0)])
    def test_a_nonpositive_frame_rate_is_refused(self, generated_fps, source_fps):
        with pytest.raises(ValueError, match="fps must be > 0"):
            align_indices(
                4, 99, generated_fps=generated_fps, source_fps=source_fps, source_start_frame=0
            )

    def test_more_context_than_the_clip_has_leaves_nothing_to_score(self):
        with pytest.raises(ValueError, match="nothing predicted to score"):
            align_indices(
                4, 99, generated_fps=24.0, source_fps=30.0, source_start_frame=0,
                lead_context_frames=4,
            )


class TestSpatialAlignment:
    def test_a_size_match_is_a_no_op_and_the_report_says_so(self):
        clip = moving_episode(4)
        assert resize_clip(clip, (HEIGHT, WIDTH)) is clip
        assert score_of(scored(clip_hw=(HEIGHT, WIDTH))).alignment.resized is False

    def test_a_geometry_change_is_recorded_as_one(self):
        report = score_of(scored(clip_hw=(HEIGHT // 2, WIDTH // 2)))
        assert report.alignment.resized is True
        assert report.alignment.comparison_hw == (HEIGHT // 2, WIDTH // 2)

    def test_the_interpolation_used_is_named_in_the_metadata(self):
        report = score_of(scored(clip_hw=(HEIGHT // 2, WIDTH // 2)), interpolation="nearest")
        assert report.alignment.interpolation == "nearest"

    def test_two_interpolations_give_different_numbers_on_the_same_pair(self):
        """So recording the kernel is not decoration — the choice moves the result."""
        args = scored(clip_hw=(HEIGHT // 2, WIDTH // 2))
        area = score_of(args, interpolation="area").arms["frozen"].mean_abs
        nearest = score_of(args, interpolation="nearest").arms["frozen"].mean_abs
        assert area != pytest.approx(nearest)

    def test_every_arm_is_resampled_with_the_kernel_the_report_names(self):
        """The test above reads only ``frozen``, and that number moves whenever EITHER side of
        the pair moves — so it passes whichever single arm ignores the argument and uses a fixed
        kernel, while ``alignment.interpolation`` goes on naming the requested one. The truth arm
        cannot stand in for the check: truth against itself is 0 for every kernel. So each arm is
        recomputed against the kernel the report claims, separately.
        """
        episode = moving_episode(60, seed=0)
        other = moving_episode(60, seed=99)
        hw = (HEIGHT // 2, WIDTH // 2)
        clip = resize_clip(episode[40:46], hw)
        for kernel in ("area", "nearest"):
            report = score_generated_video(
                clip, episode, generated_fps=30.0, source_fps=30.0, source_start_frame=20,
                other_frames=other[20:], interpolation=kernel,
            )
            indices = list(report.alignment.source_indices)
            truth = resize_clip(episode[indices], hw, interpolation=kernel)
            anchor = resize_clip(episode[19][None], hw, interpolation=kernel)[0]
            expected = {
                "model": clip,
                "frozen": frozen_control(anchor, len(indices)),
                "other": resize_clip(other[indices], hw, interpolation=kernel),
            }
            for arm, frames in expected.items():
                assert report.arms[arm].mean_abs == pytest.approx(
                    frame_metrics(frames, truth)["mean_abs"]
                ), f"the {arm} arm was not resampled with {kernel}"

    def test_every_named_kernel_is_the_cv2_kernel_of_that_name(self):
        """The vocabulary is a promise about what happened, and the two tests above cannot hold
        it. ``test_two_interpolations...`` only wants two names to differ, and area resampled as
        linear still differs from nearest by 15.5 grey levels here while differing from real
        area by 0.354 — the swap hides inside the test's own margin. ``test_every_arm_is_
        resampled...`` recomputes through ``resize_clip`` itself, so a consistently wrong mapping
        cancels on both sides. Only cv2 can say which kernel ran, so this asks cv2.
        """
        import cv2

        named = {
            "area": cv2.INTER_AREA,
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4,
        }
        assert set(named) == set(INTERPOLATIONS), "a new kernel name needs a flag pinned here"
        clip = moving_episode(6)
        hw = (HEIGHT // 2, WIDTH // 2)
        for name, flag in named.items():
            direct = np.stack(
                [cv2.resize(frame, (hw[1], hw[0]), interpolation=flag) for frame in clip]
            )
            assert np.array_equal(resize_clip(clip, hw, interpolation=name), direct), (
                f"{name!r} did not resample with cv2.{name.upper()} — the report would record "
                f"{name!r} beside numbers produced by some other kernel"
            )

    def test_an_unknown_interpolation_is_refused_by_name(self):
        with pytest.raises(ValueError, match="unknown interpolation"):
            resize_clip(moving_episode(2), (6, 8), interpolation="bicubic")

    def test_a_non_square_resize_keeps_height_as_height(self):
        """cv2 takes (width, height) and these arrays are (height, width); swapping them
        transposes the scene and every metric still returns a number."""
        out = resize_clip(moving_episode(2, hw=(20, 40)), (5, 8))
        assert out.shape == (2, 5, 8, 3)


class TestContextFramesAreNotScored:
    """A video-conditioned clip's first N frames ARE the recording. Scoring them compares the
    episode with itself, which is the ``truth`` arm's zero, and drags the total toward it."""

    def _clip_with_replayed_context(self, context: int, predicted: int):
        episode = moving_episode(80)
        real = episode[10 : 10 + context]
        # The predicted tail is a different part of the episode: wrong, but plausibly wrong.
        wrong = episode[50 : 50 + predicted]
        return np.concatenate([real, wrong]), episode

    def test_the_replayed_context_is_excluded_and_the_count_is_reported(self):
        clip, episode = self._clip_with_replayed_context(6, 8)
        report = score_generated_video(
            clip, episode, generated_fps=30.0, source_fps=30.0, source_start_frame=16,
            lead_context_frames=6,
        )
        assert report.alignment.generated_frames == 14
        assert report.alignment.scored_frames == 8
        assert report.alignment.lead_context_frames == 6

    def test_scoring_the_replayed_context_inflates_the_result(self):
        """The whole point. Same clip, same episode; only the exclusion differs."""
        clip, episode = self._clip_with_replayed_context(6, 8)
        honest = score_generated_video(
            clip, episode, generated_fps=30.0, source_fps=30.0, source_start_frame=16,
            lead_context_frames=6,
        )
        inflated = score_generated_video(
            clip, episode, generated_fps=30.0, source_fps=30.0, source_start_frame=10,
            conditioning_source_frame=9,
        )
        # Six frames of the recording against itself score 0, so the reported distance is the
        # honest one diluted by the context fraction — here 8/14 of it, a 43 % discount for
        # nothing but mislabelling replay as prediction.
        assert inflated.arms["model"].mean_abs == pytest.approx(
            honest.arms["model"].mean_abs * 8 / 14, rel=1e-4
        )

    def test_the_scored_frames_are_the_tail_of_the_clip_not_its_head(self):
        clip, episode = self._clip_with_replayed_context(6, 8)
        report = score_generated_video(
            clip, episode, generated_fps=30.0, source_fps=30.0, source_start_frame=16,
            lead_context_frames=6,
        )
        # Frames 16..23 of the episode are what the tail claims to predict.
        assert report.alignment.source_indices == tuple(range(16, 24))


def scored(*, clip_hw: tuple[int, int] = (HEIGHT, WIDTH), length: int = 6):
    """A (generated clip, episode, other episode) triple where the clip is a wrong prediction."""
    episode = moving_episode(60, seed=0)
    other = moving_episode(60, seed=99)
    clip = resize_clip(episode[40 : 40 + length], clip_hw)
    return clip, episode, other


def score_of(triple, **overrides):
    clip, episode, other = triple
    kwargs = {
        "generated_fps": 30.0,
        "source_fps": 30.0,
        "source_start_frame": 20,
        "other_frames": other[20:],
        "interpolation": DEFAULT_INTERPOLATION,
    }
    kwargs.update(overrides)
    return score_generated_video(clip, episode, **kwargs)


class TestControls:
    def test_the_truth_arm_is_exactly_zero(self):
        truth = score_of(scored()).arms["truth"]
        assert (truth.mean_abs, truth.mse, truth.gradient_abs) == (0.0, 0.0, 0.0)

    def test_the_truth_arm_is_still_zero_when_the_window_is_off_by_twenty_one_frames(self):
        """The negative result, made executable so nobody re-reads that zero as a passed check.

        ``truth`` is one array against itself, so it is 0 for the right indices and 0 for indices
        21 frames away, and the module docstring may not present it as the arm that catches a
        misaligned gather. Nothing in this module can: the clock and the start frame are facts
        about the run. What DOES move is the model arm, which is why the report publishes
        ``source_indices`` for a reader to check against the parquet.
        """
        episode = moving_episode(60)
        clock = {"generated_fps": 30.0, "source_fps": 30.0}
        right = score_generated_video(episode[30:36], episode, source_start_frame=20, **clock)
        wrong = score_generated_video(episode[30:36], episode, source_start_frame=41, **clock)
        assert right.alignment.source_indices != wrong.alignment.source_indices
        assert right.arms["truth"].mean_abs == 0.0
        assert wrong.arms["truth"].mean_abs == 0.0
        assert right.arms["model"].mean_abs != pytest.approx(wrong.arms["model"].mean_abs)

    def test_the_frozen_arm_holds_the_frame_before_the_first_predicted_one(self):
        episode = moving_episode(60)
        report = score_generated_video(
            episode[30:36], episode, generated_fps=30.0, source_fps=30.0, source_start_frame=20
        )
        assert report.alignment.conditioning_source_frame == 19
        expected = frame_metrics(frozen_control(episode[19], 6), episode[20:26])
        assert report.arms["frozen"].mean_abs == pytest.approx(expected["mean_abs"])

    def test_the_anchor_can_be_named_explicitly_and_moves_the_bar(self):
        """Image-conditioned and video-conditioned runs anchor at different places, and guessing
        wrong moves the bar without moving anything that would complain."""
        episode = moving_episode(60)
        kwargs = {"generated_fps": 30.0, "source_fps": 30.0, "source_start_frame": 20}
        default = score_generated_video(episode[30:36], episode, **kwargs)
        named = score_generated_video(
            episode[30:36], episode, conditioning_source_frame=5, **kwargs
        )
        assert named.alignment.conditioning_source_frame == 5
        assert named.arms["frozen"].mean_abs != pytest.approx(default.arms["frozen"].mean_abs)

    def test_an_anchor_outside_the_supplied_frames_is_refused(self):
        episode = moving_episode(60)
        with pytest.raises(ValueError, match="conditioning frame"):
            score_generated_video(
                episode[30:36], episode, generated_fps=30.0, source_fps=30.0,
                source_start_frame=0,
            )

    def test_a_trimmed_source_span_scores_the_same_as_the_whole_episode(self):
        """The CLI decodes ~1.25x the clip out of a 590-frame AV1 file rather than all of it, and
        ``source_offset`` is how it says which absolute frame element 0 is. Getting that wrong
        shifts every arm by the same amount, so nothing looks broken — the report is just about
        different frames than the ones it names."""
        episode = moving_episode(60)
        whole = score_generated_video(
            episode[40:46], episode, generated_fps=24.0, source_fps=30.0, source_start_frame=20
        )
        trimmed = score_generated_video(
            episode[40:46], episode[19:30], generated_fps=24.0, source_fps=30.0,
            source_start_frame=20, source_offset=19,
        )
        assert trimmed.alignment.source_indices == whole.alignment.source_indices
        assert trimmed.arms["model"].mean_abs == pytest.approx(whole.arms["model"].mean_abs)
        assert trimmed.arms["frozen"].mean_abs == pytest.approx(whole.arms["frozen"].mean_abs)

    def test_the_other_arm_reads_a_different_episode_at_the_same_offsets(self):
        episode = moving_episode(60, seed=0)
        other = moving_episode(60, seed=99)
        report = score_generated_video(
            episode[30:36], episode, generated_fps=24.0, source_fps=30.0, source_start_frame=20,
            other_frames=other[20:],
        )
        absolute = list(report.alignment.source_indices)
        relative = [i - 20 for i in absolute]
        expected = frame_metrics(other[20:][relative], episode[absolute])
        assert report.arms["other"].mean_abs == pytest.approx(expected["mean_abs"])
        assert report.alignment.other_scored is True

    def test_a_report_without_the_other_arm_says_so(self):
        report = score_of(scored(), other_frames=None)
        assert "other" not in report.arms
        assert report.alignment.other_scored is False

    def test_a_too_short_other_episode_is_refused_rather_than_scored_on_less(self):
        episode = moving_episode(60, seed=0)
        other = moving_episode(60, seed=99)
        with pytest.raises(ValueError, match="same span"):
            score_generated_video(
                episode[30:36], episode, generated_fps=24.0, source_fps=30.0,
                source_start_frame=20, other_frames=other[20:23],
            )

    def test_a_non_finite_source_pixel_stops_the_report_instead_of_becoming_a_verdict(self):
        """:func:`check_truth_is_zero` unit-tested in isolation says nothing about the scorer.

        A partially written decode puts a NaN in the source; ``as_frames_255``'s range guard does
        not fire on it (``NaN > peak`` is False) and it propagates into every arm. Without the
        call inside :func:`score_generated_video` the run does not fail — it returns a report
        whose arms are all NaN, whose ratios are all NaN, and whose verdicts read
        NO_BETTER_THAN_FREEZING and NO_BETTER_THAN_CHANCE, because ``nan < x`` is False. A
        corrupted decode would be published as a failed prediction.
        """
        episode = moving_episode(60).astype(np.float32) / 510.0 + 0.2
        poisoned = episode.copy()
        poisoned[22, 0, 0, 0] = np.nan  # frame 22 is inside the scored window 20..25
        with pytest.raises(ValueError, match="non-finite pixel"):
            score_generated_video(
                episode[30:36], poisoned, generated_fps=30.0, source_fps=30.0,
                source_start_frame=20,
            )

    def test_a_window_where_nothing_happens_is_refused_rather_than_dividing_by_zero(self):
        """T-36's D1 defect: on the first and last window of a GR00T episode the arm is not in
        frame, freezing is perfect, and the ratio the whole report is read from is 0/0."""
        still = np.full((60, HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
        with pytest.raises(ValueError, match="frozen control scores 0"):
            score_generated_video(
                moving_episode(6), still, generated_fps=30.0, source_fps=30.0,
                source_start_frame=20,
            )


class TestVerdictAndRatio:
    def _report(self, error: float):
        """A clip that is ``error`` of the way from the truth back to the frozen frame.

        Pixels live in [0.2, 0.7], not [0, 1]: ``as_frames_255`` clips a float clip to WAM's
        pixel convention, so an arm that overshoots the frozen frame (``error > 1``, the T-36
        regime) would be silently clamped and the ratio would read lower than it is.
        """
        episode = moving_episode(60).astype(np.float32) / 510.0 + 0.2
        truth = episode[20:26]
        frozen = np.repeat(episode[19][None], 6, axis=0)
        clip = truth + (frozen - truth) * error
        return score_generated_video(
            clip, episode, generated_fps=30.0, source_fps=30.0, source_start_frame=20
        )

    def test_the_ratio_is_the_model_divided_by_frozen_on_every_metric(self):
        report = self._report(0.5)
        for metric in METRICS:
            assert report.ratio_to_frozen["model"][metric] == pytest.approx(
                getattr(report.arms["model"], metric) / getattr(report.arms["frozen"], metric)
            )

    def test_a_perfect_prediction_beats_frozen(self):
        report = self._report(0.0)
        assert report.verdicts["beats_frozen"] == VERDICT_PREDICTS
        assert report.ratio_to_frozen["model"]["mean_abs"] == pytest.approx(0.0)

    def test_predicting_the_conditioning_frame_does_not_beat_frozen(self):
        """A model that learned "nothing changes" reproduces the baseline exactly and must not be
        recorded as having predicted anything."""
        report = self._report(1.0)
        assert report.verdicts["beats_frozen"] == VERDICT_NO_BETTER
        assert report.ratio_to_frozen["model"]["mean_abs"] == pytest.approx(1.0)

    def test_a_five_percent_win_is_not_enough(self):
        report = self._report(0.95)
        assert report.arms["model"].mean_abs < report.arms["frozen"].mean_abs
        assert report.verdicts["beats_frozen"] == VERDICT_NO_BETTER
        assert FREEZE_MARGIN == 0.9

    def test_a_t36_shaped_result_reads_as_worse_than_standing_still(self):
        """T-36 measured 16.656 against freezing's 12.020 — a ratio of 1.39. The number this
        module exists to produce, reproduced on arithmetic anyone can check."""
        report = self._report(1.39)
        assert report.ratio_to_frozen["model"]["mean_abs"] == pytest.approx(1.39, abs=1e-3)
        assert report.verdicts["beats_frozen"] == VERDICT_NO_BETTER

    def test_the_gate_reads_the_headline_metric_and_not_the_outlier_sensitive_one(self):
        """The other fixtures here sit at 0.0, 0.95, 1.0 and 1.39, where mean_abs and mse land on
        the same side of the margin, so the gate could be computed from either. At 0.93 they
        part: the mean_abs ratio is 0.930 and the squared-error ratio is 0.865, and a gate on mse
        would publish PREDICTS for a clip the pre-registered PR-06 threshold rejects."""
        report = self._report(0.93)
        assert report.ratio_to_frozen["model"]["mean_abs"] == pytest.approx(0.93, abs=1e-3)
        assert report.ratio_to_frozen["model"]["mse"] < FREEZE_MARGIN
        assert report.verdicts["beats_frozen"] == VERDICT_NO_BETTER


class TestChanceIsTheSecondBar:
    """Beating the frozen bar is necessary and not sufficient on this corpus.

    Measured through the CLI on the local mirror, a different demo of the same task at the same
    phase scores other/frozen ``mean_abs`` 0.697 (ep0@271, 72f), 0.743 (ep10@250, 48f), 0.991
    (ep0@400, 48f), 1.027 (ep2@200, 72f) and 1.349 (ep0@150, 72f). Two of five windows are inside
    the 10 % margin, so a clip a lookup into the training set could match would earn PREDICTS.
    The report computed that control already; the defect was emitting one verdict that ignored it.
    """

    def _report(self, *, model_error: float, other_error: float):
        """Model and chance arms placed at chosen fractions of the frozen bar's distance.

        Same linear family as :class:`TestVerdictAndRatio`, so both ratios come out equal to the
        fractions asked for and the thresholds are readable off the call.
        """
        episode = moving_episode(60).astype(np.float32) / 510.0 + 0.2
        truth = episode[20:26]
        frozen = np.repeat(episode[19][None], 6, axis=0)
        return score_generated_video(
            truth + (frozen - truth) * model_error,
            episode,
            generated_fps=30.0,
            source_fps=30.0,
            source_start_frame=20,
            other_frames=truth + (frozen - truth) * other_error,
        )

    def test_a_clip_that_beats_freezing_but_not_another_demo_is_not_recorded_as_predicting(self):
        report = self._report(model_error=0.8, other_error=0.7)
        assert report.verdicts["beats_frozen"] == VERDICT_PREDICTS
        assert report.verdicts["beats_chance"] == VERDICT_NO_BETTER_THAN_CHANCE

    def test_a_clip_that_clears_both_bars_says_so(self):
        report = self._report(model_error=0.5, other_error=0.9)
        assert report.verdicts["beats_frozen"] == VERDICT_PREDICTS
        assert report.verdicts["beats_chance"] == VERDICT_BEATS_CHANCE

    def test_a_five_percent_win_over_the_other_episode_is_not_enough_either(self):
        """The same pre-registered margin on both bars, so a win means one thing in this report."""
        report = self._report(model_error=0.95 * 0.8, other_error=0.8)
        assert report.arms["model"].mean_abs < report.arms["other"].mean_abs
        assert report.verdicts["beats_chance"] == VERDICT_NO_BETTER_THAN_CHANCE

    def test_a_clip_that_is_only_a_brightness_shift_fails_both_gates(self):
        """Which metric the two gates read, pinned. Every other fixture in this file and in
        :class:`TestVerdictAndRatio` is a linear blend ``truth + (frozen - truth) * error``, on
        which all three metric ratios equal ``error`` by construction — so both gates could be
        computed from ``gradient_abs`` and nothing here would notice.

        This clip is the true future plus a uniform 30 grey levels. That is more than twice as
        far from the truth as standing still (mean_abs ratio 2.134) and it is exactly what
        ``gradient_abs`` is built to ignore, so the gradient ratio is 0.000 and a gradient-gated
        report would publish PREDICTS *and* BEATS_CHANCE for a clip that predicted nothing at
        all. The failure mode is not hypothetical: a VAE-decoded frame routinely comes back a few
        grey levels off, which is why the gradient metric exists — as a diagnostic, not a gate.
        """
        episode = moving_episode(60).astype(np.float32) / 510.0 + 0.2
        truth = episode[20:26]
        frozen = np.repeat(episode[19][None], 6, axis=0)
        report = score_generated_video(
            truth + 30.0 / 255.0,
            episode,
            generated_fps=30.0,
            source_fps=30.0,
            source_start_frame=20,
            other_frames=truth + (frozen - truth) * 0.9,
        )
        assert report.ratio_to_frozen["model"]["mean_abs"] == pytest.approx(2.134, abs=1e-3)
        assert report.ratio_to_frozen["model"]["gradient_abs"] == pytest.approx(0.0, abs=1e-6)
        # Precondition: on the gradient both gates would have passed, so this is a real fork.
        model_gradient = report.arms["model"].gradient_abs
        assert model_gradient < FREEZE_MARGIN * report.arms["frozen"].gradient_abs
        assert model_gradient < FREEZE_MARGIN * report.arms["other"].gradient_abs
        assert report.verdicts["beats_frozen"] == VERDICT_NO_BETTER
        assert report.verdicts["beats_chance"] == VERDICT_NO_BETTER_THAN_CHANCE

    def test_a_report_with_no_chance_arm_says_the_bar_was_not_measured(self):
        """Silence would read as a pass. A verdict dict holding only ``beats_frozen`` looks like
        a clip that cleared everything asked of it."""
        report = score_of(scored(), other_frames=None)
        assert report.verdicts["beats_chance"] == VERDICT_CHANCE_NOT_MEASURED


class TestCodecFloor:
    """What a PERFECT prediction scores, which is not zero.

    The model arm is read out of a lossy container while ``frozen`` and ``truth`` come straight
    off the source decode, so the model pays an encode the controls never pay. Measured through
    the CLI on ep0, 72 frames at 24 fps from frame 271: the byte-exact true future written with
    libx264 and read back scores ``gradient_abs`` 1.111 against the frozen bar's 2.417 — ratio
    0.460 — where the same frames without the round trip score 0.000. ``mean_abs`` costs 0.076 of
    the bar and ``mse`` 0.003, so it is the gradient metric specifically: codec ringing is high
    frequency and the frozen gradient bar is tiny because 96 % of this corpus's frame pairs barely
    move (T-35). Across five windows the floor runs 0.460 to 0.723. Undisclosed, a model at 0.55
    reads as half the structure captured.
    """

    def _ringing(self, truth: np.ndarray) -> np.ndarray:
        """A stand-in for codec loss: a few grey levels on every other pixel. Synthetic, because
        a real encode is not reproducible across ffmpeg builds; the CLI test uses a real one."""
        ring = np.zeros(truth.shape, dtype=np.int16)
        ring[:, ::2, ::2] = 4
        return np.clip(truth.astype(np.int16) + ring, 0, 255).astype(np.uint8)

    def test_the_floor_arm_is_scored_against_the_truth_like_every_other_arm(self):
        episode = moving_episode(60)
        floor = self._ringing(episode[20:26])
        report = score_generated_video(
            episode[30:36], episode, generated_fps=30.0, source_fps=30.0, source_start_frame=20,
            codec_floor_frames=floor,
        )
        expected = frame_metrics(floor, episode[20:26])
        assert report.alignment.codec_floor_scored is True
        assert report.arms["codec_floor"].gradient_abs == pytest.approx(expected["gradient_abs"])
        assert report.ratio_to_frozen["codec_floor"]["gradient_abs"] == pytest.approx(
            expected["gradient_abs"] / report.arms["frozen"].gradient_abs
        )

    def test_high_frequency_error_costs_twice_as_much_on_the_gradient_metric(self):
        """Why the floor is a gradient problem and not a general one, on arithmetic.

        The stand-in adds 4 grey levels to every other pixel of every other row: a quarter of the
        pixels, so ``mean_abs`` is 4/4 = 1.0. Every first difference across or down that pattern
        flips between 0 and 4, so both gradient terms average 2.0 and the metric is 2.0 — twice
        the mean error, from an error that moves no region of the image anywhere. On the real
        corpus the gap is far wider because the frozen gradient bar it is divided by is tiny.

        This does NOT reproduce the measured 0.460 ratio: these frames are a noise walk, and 96 %
        of the real corpus's frame pairs barely move (T-35), which is what makes the real floor
        that large. The corpus number lives in the class docstring with the window to reproduce
        it on; what is pinned here is the mechanism.
        """
        episode = moving_episode(60)
        report = score_generated_video(
            episode[30:36], episode, generated_fps=30.0, source_fps=30.0, source_start_frame=20,
            codec_floor_frames=self._ringing(episode[20:26]),
        )
        assert report.arms["codec_floor"].mean_abs == pytest.approx(1.0)
        assert report.arms["codec_floor"].gradient_abs == pytest.approx(2.0)

    def test_a_report_without_the_floor_does_not_claim_to_know_what_perfect_scores(self):
        report = score_of(scored())
        assert report.alignment.codec_floor_scored is False
        assert "codec_floor" not in report.arms

    def test_a_floor_measured_over_a_different_window_is_refused(self):
        """The encode cost depends on what is in the frames — 72 frames of a static shelf ring
        far less than 72 of an arm crossing the scene — so a floor over a shorter run would
        understate exactly the contamination it exists to expose."""
        episode = moving_episode(60)
        with pytest.raises(ValueError, match="different window"):
            score_generated_video(
                episode[30:36], episode, generated_fps=30.0, source_fps=30.0,
                source_start_frame=20, codec_floor_frames=episode[20:24],
            )


class TestOneRatioIsOnlyComparableWithTheSameWindow:
    """The headline ratio divides by the frozen bar, and the bar is a property of the window.

    Measured on the local mirror the bar runs 5.597 ``mean_abs`` (ep0@400, 48f) to 25.639
    (ep2@200, 72f), and the ``codec_floor`` arm — always the same libx264 round trip, i.e.
    constant quality — scores 0.074, 0.076, 0.085, 0.178 and 0.296 of it across five windows.
    A 4x spread against a 10 % decision margin: reading Wan's ratio on one window against
    Cosmos3's on another ranks the windows. Both reports are individually valid, which is why
    this has to be a refusal rather than something a careful reader would spot.
    """

    def test_two_reports_of_the_same_window_compare(self):
        check_same_window(score_of(scored()), score_of(scored()))

    def test_a_report_of_a_later_window_in_the_same_episode_is_refused(self):
        with pytest.raises(ValueError, match="different windows"):
            check_same_window(score_of(scored()), score_of(scored(), source_start_frame=25))

    def test_the_refusal_names_the_frames_each_report_scored(self):
        with pytest.raises(ValueError, match=r"source_indices: wan 20\.\.25"):
            check_same_window(
                score_of(scored()),
                score_of(scored(), source_start_frame=25),
                names=("wan", "cosmos3"),
            )

    def test_a_different_resize_kernel_is_a_different_window(self):
        """It moves the arms — 0.354 grey levels between area and linear on this fixture — so
        two ratios produced with different kernels are two measurements, not one comparison."""
        args = scored(clip_hw=(HEIGHT // 2, WIDTH // 2))
        with pytest.raises(ValueError, match="interpolation"):
            check_same_window(
                score_of(args, interpolation="area"), score_of(args, interpolation="nearest")
            )

    def test_the_same_frame_numbers_of_a_different_episode_are_a_different_window(self):
        """Frames 20..25 of episode 0 and of episode 7 share every alignment field there is. The
        recording is only visible in ``info``, which the CLI writes and the library cannot."""
        a = score_of(scored(), info={"data_dir": "data/raw/gr00t_apple", "episode": 0})
        b = score_of(scored(), info={"data_dir": "data/raw/gr00t_apple", "episode": 7})
        with pytest.raises(ValueError, match="info.episode"):
            check_same_window(a, b)

    def test_reports_made_from_arrays_carry_no_episode_and_are_compared_on_alignment_alone(self):
        """Stated as a limit rather than hidden: a library caller can hand this function two
        reports from two different corpora and, if the windows line up, it will pass them."""
        check_same_window(score_of(scored(), info={}), score_of(scored(), info={"episode": 7}))


class TestReportShape:
    def test_it_serializes_to_json_with_the_indices_it_compared(self):
        report = score_of(scored())
        payload = json.loads(report.model_dump_json())
        assert payload["version"] == "video_fidelity/1"
        assert len(payload["alignment"]["source_indices"]) == payload["alignment"]["scored_frames"]
        assert set(payload["arms"]) == {"model", "frozen", "truth", "other"}

    def test_the_info_a_caller_supplies_is_carried_through(self):
        report = score_of(scored(), info={"generated": "cosmos3_future.mp4"})
        assert report.info["generated"] == "cosmos3_future.mp4"

    def test_a_clip_that_is_not_frames_is_refused(self):
        with pytest.raises(ValueError, match=r"\[F, H, W, 3\]"):
            score_generated_video(
                np.zeros((4, 4, 3), np.uint8), moving_episode(20), generated_fps=30.0,
                source_fps=30.0, source_start_frame=5,
            )


class TestFrozenControl:
    def test_it_repeats_one_frame_and_nothing_moves(self):
        frame = moving_episode(1)[0]
        held = frozen_control(frame, 5)
        assert held.shape == (5, HEIGHT, WIDTH, 3)
        assert np.abs(np.diff(held, axis=0)).max() == 0

    def test_a_clip_is_not_a_conditioning_frame(self):
        with pytest.raises(ValueError, match=r"\[H, W, 3\]"):
            frozen_control(moving_episode(3), 5)


def write_mp4(path: Path, frames: np.ndarray, fps: int) -> Path:
    """A throwaway H.264 clip in ``tmp_path``. No fixture is committed: the point of the CLI
    test is the plumbing (find the mirror, read the fps out of meta, write the report), and
    codec loss would let an off-by-one hide inside it if anything else were asserted here."""
    import imageio.v3 as iio

    iio.imwrite(path, frames, fps=fps, codec="libx264")
    return path


class TestCli:
    """The thin wrapper: it must find the mirror's videos, read the fps out of meta, and write a
    report to runs/backbone_eval/. mp4 is lossy, so this asserts structure, not pixel values."""

    def _mirror(self, tmp_path: Path, fps: int = 30) -> Path:
        root = tmp_path / "mirror"
        videos = root / "videos" / "chunk-000" / "observation.images.ego_view"
        videos.mkdir(parents=True)
        (root / "meta").mkdir()
        (root / "meta" / "info.json").write_text(json.dumps({"fps": fps}))
        lengths = []
        for episode in (0, 1):
            frames = moving_episode(40, seed=episode, hw=(32, 48))
            write_mp4(videos / f"episode_{episode:06d}.mp4", frames, fps)
            lengths.append({"episode_index": episode, "length": 40})
        (root / "meta" / "episodes.jsonl").write_text(
            "\n".join(json.dumps(row) for row in lengths)
        )
        return root

    def test_it_scores_a_clip_against_the_local_mirror_and_writes_json(self, tmp_path):
        import score_generated_video as cli

        root = self._mirror(tmp_path)
        clip = write_mp4(tmp_path / "dream.mp4", moving_episode(8, seed=5, hw=(32, 48)), 24)
        out = tmp_path / "runs" / "backbone_eval" / "video_fidelity.json"

        code = cli.main(
            [
                "--generated", str(clip), "--data-dir", str(root), "--episode", "0",
                "--start-frame", "10", "--generated-fps", "24", "--out", str(out),
            ]
        )
        assert code == 0
        payload = json.loads(out.read_text())
        assert payload["alignment"]["source_fps"] == 30.0  # read from meta/info.json
        assert payload["alignment"]["generated_fps"] == 24.0
        assert payload["alignment"]["source_indices"][:3] == [10, 11, 13]  # by time, not by index
        assert payload["alignment"]["conditioning_source_frame"] == 9
        assert payload["info"]["other_episode"] == 1
        assert payload["arms"]["truth"]["mean_abs"] == 0.0
        assert payload["verdicts"]["beats_frozen"] in {VERDICT_PREDICTS, VERDICT_NO_BETTER}

    def test_the_chance_arm_reads_the_other_episode_and_not_the_one_it_conditioned_on(
        self, tmp_path
    ):
        """``info.other_episode`` naming a different file is provenance the report asserts, not
        provenance it demonstrates. Pointing the chance arm at the conditioned episode instead
        collapses it toward the frozen bar — the scale the whole report is read against — while
        the JSON goes on naming the right file. So the arm is recomputed from episode 1's frames,
        at the offset the search settled on (``info.other_start_frame``, which the test below is
        the one that pins).
        """
        import score_generated_video as cli

        root = self._mirror(tmp_path)
        clip = write_mp4(tmp_path / "dream.mp4", moving_episode(8, seed=5, hw=(32, 48)), 24)
        out = tmp_path / "fidelity.json"
        cli.main(
            [
                "--generated", str(clip), "--data-dir", str(root), "--episode", "0",
                "--start-frame", "10", "--generated-fps", "24", "--out", str(out),
            ]
        )
        payload = json.loads(out.read_text())

        indices = payload["alignment"]["source_indices"]
        offset = payload["info"]["other_start_frame"] - payload["alignment"]["source_start_frame"]
        videos = root / "videos" / "chunk-000" / "observation.images.ego_view"
        chance = cli.read_frames(videos / "episode_000001.mp4", [i + offset for i in indices])
        truth = cli.read_frames(videos / "episode_000000.mp4", indices)
        assert payload["arms"]["other"]["mean_abs"] == pytest.approx(
            frame_metrics(chance, truth)["mean_abs"], rel=1e-6
        )

    def _mirror_hiding_a_match(self, tmp_path: Path, *, at: int, fps: int = 30) -> Path:
        """A mirror whose episode 1 replays episode 0's scored window starting at frame ``at``.

        The retrieval arm exists to say what a lookup into another demo of the task achieves, and
        on the real corpus that lookup is somewhere else in the other episode than at the same
        absolute index: episodes run 249 to 749 frames. Here the match is planted, so the offset
        that finds it is known and the search either lands on it or does not.
        """
        root = tmp_path / "mirror"
        videos = root / "videos" / "chunk-000" / "observation.images.ego_view"
        videos.mkdir(parents=True)
        (root / "meta").mkdir()
        (root / "meta" / "info.json").write_text(json.dumps({"fps": fps}))
        episode = moving_episode(40, seed=0, hw=(32, 48))
        other = moving_episode(40, seed=1, hw=(32, 48))
        other[at : at + 10] = episode[10:20]
        write_mp4(videos / "episode_000000.mp4", episode, fps)
        write_mp4(videos / "episode_000001.mp4", other, fps)
        (root / "meta" / "episodes.jsonl").write_text(
            "\n".join(json.dumps({"episode_index": e, "length": 40}) for e in (0, 1))
        )
        return root

    def _half_way_clip(self, tmp_path: Path, root: Path) -> Path:
        """A clip half way between the conditioning frame and the truth: beats frozen, predicts
        nothing a lookup could not. Written through libx264 like a real generated clip."""
        import score_generated_video as cli

        video = root / "videos" / "chunk-000" / "observation.images.ego_view" / "episode_000000.mp4"
        indices = align_indices(8, 40, generated_fps=24.0, source_fps=30.0, source_start_frame=10)
        truth = cli.read_frames(video, indices).astype(np.float32)
        frozen = cli.read_frames(video, [9])[0].astype(np.float32)
        return write_mp4(tmp_path / "dream.mp4", (0.5 * truth + 0.5 * frozen).astype(np.uint8), 24)

    def test_the_retrieval_arm_is_read_where_the_other_demo_matches_not_at_the_index_asked_for(
        self, tmp_path
    ):
        """The offset is an absolute index into an episode of a different length, so it is not a
        phase, and the arm's strength swings with it — 0.522 to 1.118 of the frozen bar at
        ep0@271 on the real mirror. A verdict read off one arbitrary point on that curve is a
        verdict about the point, so the CLI searches the grid and keeps the strongest lookup.
        """
        import score_generated_video as cli

        root = self._mirror_hiding_a_match(tmp_path, at=30)
        clip = write_mp4(tmp_path / "dream.mp4", moving_episode(8, seed=5, hw=(32, 48)), 24)
        out = tmp_path / "fidelity.json"
        cli.main(
            [
                "--generated", str(clip), "--data-dir", str(root), "--episode", "0",
                "--start-frame", "10", "--generated-fps", "24", "--out", str(out),
            ]
        )
        payload = json.loads(out.read_text())
        sweep = {offset: value for offset, value in payload["info"]["other_offset_sweep"]}

        assert payload["info"]["other_offset_centre"] == 10
        assert payload["info"]["other_start_frame"] == 30, "the planted match was not found"
        assert sorted(sweep) == [0, 10, 20, 30], "the grid is published, not just its winner"
        # The arm the verdict reads is the offset the search chose, not some other one.
        assert payload["arms"]["other"]["mean_abs"] == pytest.approx(sweep[30], rel=1e-6)
        # And the offset asked for is a materially weaker control — which is the whole finding.
        # Measured here: 7.12 grey levels at 30 against 11.50 at 10, i.e. 0.62. Not smaller,
        # because both episodes are random noise written through libx264 and the codec spends
        # most of its error budget on the background rather than on the planted match.
        assert sweep[30] < 0.75 * sweep[10]

    def test_a_clip_a_lookup_into_the_other_demo_matches_is_not_certified_as_beating_chance(
        self, tmp_path
    ):
        """The consequence, both ways round. The same clip on the same window: pinned to the
        requested offset it is published as having beaten retrieval, and against the strongest
        lookup in the same episode it has not. ``--other-search-radius 0`` is the old behaviour,
        kept so an archived single-offset number can be reproduced.
        """
        import score_generated_video as cli

        root = self._mirror_hiding_a_match(tmp_path, at=30)
        clip = self._half_way_clip(tmp_path, root)
        common = ["--generated", str(clip), "--data-dir", str(root), "--episode", "0",
                  "--start-frame", "10", "--generated-fps", "24"]
        pinned, searched = tmp_path / "pinned.json", tmp_path / "searched.json"
        cli.main([*common, "--other-search-radius", "0", "--out", str(pinned)])
        cli.main([*common, "--out", str(searched)])

        one_offset = json.loads(pinned.read_text())
        best_offset = json.loads(searched.read_text())
        assert one_offset["info"]["other_start_frame"] == 10
        assert best_offset["info"]["other_start_frame"] == 30
        assert one_offset["ratio_to_frozen"]["model"]["mean_abs"] == pytest.approx(
            best_offset["ratio_to_frozen"]["model"]["mean_abs"], rel=1e-9
        ), "precondition: the clip and the window are identical, only the control moved"
        assert one_offset["verdicts"]["beats_frozen"] == VERDICT_PREDICTS
        assert best_offset["verdicts"]["beats_frozen"] == VERDICT_PREDICTS
        assert one_offset["verdicts"]["beats_chance"] == VERDICT_BEATS_CHANCE
        assert best_offset["verdicts"]["beats_chance"] == VERDICT_NO_BETTER_THAN_CHANCE

    def test_the_codec_floor_is_this_windows_truth_through_the_generated_clips_own_encoder(
        self, tmp_path
    ):
        """The model arm is the only one read out of an mp4, so without this arm the gradient
        numbers have no floor beside them and 0.55 reads as half the structure captured. Which
        makes the floor's own content load-bearing, and ``> 0`` does not check it: the same
        assertion passes when the floor is built from the frames one index early (measured 0.873
        instead of 0.318 through this CLI), from the frozen anchor repeated (0.955), or encoded
        at the source's 30 fps instead of the clip's 24 (0.347). A floor of 0.955 would make
        every gradient result look like codec noise. So it is recomputed here from the frames the
        report says it scored, through the same round trip at the fps the clip declares.
        """
        import score_generated_video as cli

        root = self._mirror(tmp_path)
        clip = write_mp4(tmp_path / "dream.mp4", moving_episode(8, seed=5, hw=(32, 48)), 24)
        out = tmp_path / "fidelity.json"
        cli.main(
            [
                "--generated", str(clip), "--data-dir", str(root), "--episode", "0",
                "--start-frame", "10", "--generated-fps", "24", "--out", str(out),
            ]
        )
        payload = json.loads(out.read_text())
        assert payload["alignment"]["codec_floor_scored"] is True
        assert payload["info"]["codec_floor_codec"] == "libx264"
        assert payload["alignment"]["resized"] is False, "precondition: no resize to reproduce"

        videos = root / "videos" / "chunk-000" / "observation.images.ego_view"
        indices = payload["alignment"]["source_indices"]
        truth = cli.read_frames(videos / "episode_000000.mp4", indices)
        expected = frame_metrics(cli.codec_floor_clip(truth, 24.0), truth)
        for metric in METRICS:
            assert payload["arms"]["codec_floor"][metric] == pytest.approx(
                expected[metric], rel=1e-6
            ), f"the floor's {metric} is not this window's truth through libx264 at 24 fps"
        # A round trip is lossy, so perfect is not 0 — that is the entire point of the arm.
        assert payload["arms"]["codec_floor"]["gradient_abs"] > 0.0
        assert payload["ratio_to_frozen"]["codec_floor"]["gradient_abs"] > 0.0

    def test_the_offset_flag_names_the_centre_of_the_search_and_is_honoured(self, tmp_path):
        import score_generated_video as cli

        root = self._mirror_hiding_a_match(tmp_path, at=30)
        clip = write_mp4(tmp_path / "dream.mp4", moving_episode(8, seed=5, hw=(32, 48)), 24)
        out = tmp_path / "fidelity.json"
        cli.main(
            [
                "--generated", str(clip), "--data-dir", str(root), "--episode", "0",
                "--start-frame", "10", "--generated-fps", "24", "--other-start-frame", "20",
                "--other-search-radius", "0", "--out", str(out),
            ]
        )
        payload = json.loads(out.read_text())
        assert payload["info"]["other_offset_centre"] == 20
        assert payload["info"]["other_start_frame"] == 20
        assert payload["info"]["other_offset_sweep"] == [
            [20, pytest.approx(payload["arms"]["other"]["mean_abs"], rel=1e-6)]
        ]

    def test_a_mirror_without_episode_lengths_does_not_claim_a_search_it_could_not_run(
        self, tmp_path
    ):
        """No ``meta/episodes.jsonl`` means no bound to reject candidate offsets against, and a
        candidate running past the end of the file would refuse the whole run. The arm falls back
        to the single requested offset — and the report says ``other_search_radius`` 0, because a
        recorded radius of 120 beside a one-point sweep would read as a search that happened."""
        import score_generated_video as cli

        root = self._mirror(tmp_path)
        (root / "meta" / "episodes.jsonl").unlink()
        clip = write_mp4(tmp_path / "dream.mp4", moving_episode(8, seed=5, hw=(32, 48)), 24)
        out = tmp_path / "fidelity.json"
        assert (
            cli.main(
                [
                    "--generated", str(clip), "--data-dir", str(root), "--episode", "0",
                    "--start-frame", "10", "--generated-fps", "24", "--out", str(out),
                ]
            )
            == 0
        )
        payload = json.loads(out.read_text())
        assert payload["info"]["other_search_radius"] == 0
        assert payload["info"]["other_start_frame"] == 10
        assert len(payload["info"]["other_offset_sweep"]) == 1

    def test_the_output_discloses_the_window_and_the_offsets_the_chance_arm_was_tried_at(
        self, tmp_path, capsys
    ):
        """Both disclosures are printed where the mistake is made — while reading the table —
        and not only in a docstring: that the ratios belong to one window, and that the chance
        arm is the best of several offsets rather than the single one the flags named."""
        import score_generated_video as cli

        root = self._mirror(tmp_path)
        clip = write_mp4(tmp_path / "dream.mp4", moving_episode(8, seed=5, hw=(32, 48)), 24)
        cli.main(
            [
                "--generated", str(clip), "--data-dir", str(root), "--episode", "0",
                "--start-frame", "10", "--generated-fps", "24",
                "--out", str(tmp_path / "fidelity.json"),
            ]
        )
        printed = capsys.readouterr().out
        assert "retrieval arm: episode 1, 4 offset(s) around 10, using" in printed
        assert "episode 0 frames 10..19 at 32x48 and about no other window" in printed

    def test_comparing_two_reports_of_different_windows_is_refused(self, tmp_path):
        """The tool exists to rank Wan against Cosmos3, and the frozen bar it divides by is 4.6x
        larger on one measured window of this corpus than on another. Nothing about the two
        reports looks wrong, so the refusal has to be in the comparison itself."""
        import score_generated_video as cli

        root = self._mirror(tmp_path)
        clip = write_mp4(tmp_path / "dream.mp4", moving_episode(8, seed=5, hw=(32, 48)), 24)
        common = ["--generated", str(clip), "--data-dir", str(root), "--episode", "0",
                  "--generated-fps", "24"]
        first, second = tmp_path / "first.json", tmp_path / "second.json"
        assert cli.main([*common, "--start-frame", "10", "--out", str(first)]) == 0
        assert (
            cli.main([*common, "--start-frame", "20", "--out", str(second),
                      "--compare", str(first)])
            == 2
        )

    def test_comparing_two_reports_of_the_same_window_prints_both_headlines(
        self, tmp_path, capsys
    ):
        import score_generated_video as cli

        root = self._mirror(tmp_path)
        clip = write_mp4(tmp_path / "dream.mp4", moving_episode(8, seed=5, hw=(32, 48)), 24)
        other_clip = write_mp4(tmp_path / "dream2.mp4", moving_episode(8, seed=6, hw=(32, 48)), 24)
        common = ["--data-dir", str(root), "--episode", "0", "--start-frame", "10",
                  "--generated-fps", "24"]
        first, second = tmp_path / "first.json", tmp_path / "second.json"
        cli.main([*common, "--generated", str(clip), "--out", str(first)])
        assert (
            cli.main([*common, "--generated", str(other_clip), "--out", str(second),
                      "--compare", str(first)])
            == 0
        )
        printed = capsys.readouterr().out
        theirs = json.loads(first.read_text())["ratio_to_frozen"]["model"]["mean_abs"]
        ours = json.loads(second.read_text())["ratio_to_frozen"]["model"]["mean_abs"]
        assert f"{ours:.3f} here, {theirs:.3f} there" in printed

    def test_the_floor_can_be_skipped_and_the_report_then_admits_it_does_not_know(self, tmp_path):
        import score_generated_video as cli

        root = self._mirror(tmp_path)
        clip = write_mp4(tmp_path / "dream.mp4", moving_episode(8, seed=5, hw=(32, 48)), 24)
        out = tmp_path / "fidelity.json"
        cli.main(
            [
                "--generated", str(clip), "--data-dir", str(root), "--episode", "0",
                "--start-frame", "10", "--generated-fps", "24", "--no-codec-floor",
                "--out", str(out),
            ]
        )
        payload = json.loads(out.read_text())
        assert payload["alignment"]["codec_floor_scored"] is False
        assert "codec_floor" not in payload["arms"]
        assert payload["info"]["codec_floor_codec"] is None

    def test_a_container_rate_that_contradicts_the_declared_one_is_flagged(self, tmp_path):
        """The one clock error the files themselves reveal, and the reason it is worth reporting:
        the truth arm cannot catch a wrong fps — it is 0 whatever the indices are — so declaring
        24 for a clip the sampler wrote at 30 silently shifts every index and every arm.

        Flagged rather than refused, because ``container_fps`` is deliberately not the authority:
        a re-encode changes the container's rate without changing what was generated.
        """
        import score_generated_video as cli

        root = self._mirror(tmp_path)
        clip = write_mp4(tmp_path / "dream.mp4", moving_episode(8, seed=5, hw=(32, 48)), 30)
        honest = tmp_path / "honest.json"
        wrong = tmp_path / "wrong.json"
        common = ["--generated", str(clip), "--data-dir", str(root), "--episode", "0",
                  "--start-frame", "10"]
        cli.main([*common, "--generated-fps", "30", "--out", str(honest)])
        cli.main([*common, "--generated-fps", "24", "--out", str(wrong)])

        assert json.loads(honest.read_text())["info"]["generated_fps_mismatch"] is False
        misdeclared = json.loads(wrong.read_text())
        assert misdeclared["info"]["generated_fps_mismatch"] is True
        assert misdeclared["info"]["generated_container_fps"] == pytest.approx(30.0)

    def test_it_excludes_the_replayed_context_when_told_about_it(self, tmp_path):
        import score_generated_video as cli

        root = self._mirror(tmp_path)
        clip = write_mp4(tmp_path / "dream.mp4", moving_episode(8, seed=5, hw=(32, 48)), 24)
        out = tmp_path / "fidelity.json"
        cli.main(
            [
                "--generated", str(clip), "--data-dir", str(root), "--episode", "0",
                "--start-frame", "10", "--context-frames", "3", "--out", str(out),
            ]
        )
        payload = json.loads(out.read_text())
        assert payload["alignment"]["generated_frames"] == 8
        assert payload["alignment"]["scored_frames"] == 5

    def test_it_only_decodes_the_frames_it_needs(self, tmp_path):
        """A 590-frame AV1 episode is expensive to walk; the span is 1.25x the clip, not the
        episode. Also the guard that a requested frame missing from the file raises."""
        import score_generated_video as cli

        root = self._mirror(tmp_path)
        video = root / "videos" / "chunk-000" / "observation.images.ego_view" / "episode_000000.mp4"
        frames = cli.read_frames(video, range(4, 9))
        assert frames.shape == (5, 32, 48, 3)
        with pytest.raises(ValueError, match="has no frames"):
            cli.read_frames(video, range(38, 44))

    def test_a_window_past_the_end_of_the_episode_fails_before_any_decoding(self, tmp_path):
        import score_generated_video as cli

        root = self._mirror(tmp_path)
        clip = write_mp4(tmp_path / "dream.mp4", moving_episode(8, seed=5, hw=(32, 48)), 24)
        with pytest.raises(ValueError, match="refusing to truncate"):
            cli.main(
                [
                    "--generated", str(clip), "--data-dir", str(root), "--episode", "0",
                    "--start-frame", "36", "--out", str(tmp_path / "x.json"),
                ]
            )
