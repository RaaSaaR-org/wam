"""The propagation arm's frame ingest, and the confound it exists to exclude.

`apple_sam2`'s THIRD gate-qualification blocker asks for the same capture measured both ways —
this adapter per frame, and `SAM2VideoPredictor` propagating one mask from frame 0. Upstream's
predictor ingests a directory of **JPEGs**; our captures are lossless `rgb.npy`. An arm that saw
JPEGs against an arm that saw raw arrays would be measuring the codec and calling it propagation,
and the whole comparison would be void.

So `estimators.apple_sam2_video` ingests the arrays directly and this file is the proof that its
ingest is upstream's ingest — bitwise — with the lossy step removed. No weight is loaded here and
no GPU is touched: the assertions are about pixels going in, which is the half that decides whether
the two p95s are comparable at all.

Nothing here discharges anything. `GATE_QUALIFIED` and `GATE_QUALIFICATION_BLOCKERS` are read, not
written, and the last test asserts exactly that.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

torch = pytest.importorskip("torch")
misc = pytest.importorskip("sam2.utils.misc")
PIL_Image = pytest.importorskip("PIL.Image")

from estimators import apple_sam2  # noqa: E402
from estimators import apple_sam2_video as vid  # noqa: E402

IMAGE_SIZE = 64


def _clip(n=4, h=48, w=64):
    """Frames with structure a codec will visibly disagree with: hard edges and saturated colour."""
    rng = np.random.default_rng(20260825)
    out = []
    for i in range(n):
        f = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        f[10:30, 8 + i : 28 + i] = np.array([220, 30, 20], dtype=np.uint8)
        out.append(f)
    return out


def _upstream_tensor_from_pngs(tmp_path, frames):
    """What upstream's own loader produces from LOSSLESS files of the same frames."""
    paths = []
    for i, f in enumerate(frames):
        p = tmp_path / f"{i}.png"
        PIL_Image.fromarray(f).save(p)
        paths.append(p)
    stack = torch.zeros(len(frames), 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
    for i, p in enumerate(paths):
        stack[i], height, width = misc._load_img_as_tensor(str(p), IMAGE_SIZE)
    return stack, height, width


def test_the_in_memory_ingest_is_bitwise_upstreams_ingest_of_a_lossless_file(tmp_path):
    """Not "close enough". The propagation arm must resize, scale and normalise EXACTLY as
    `_load_img_as_tensor` does, or the two arms differ in preprocessing as well as in propagation
    and the difference between the p95s stops being attributable."""
    frames = _clip()
    ours, h, w = vid.frames_to_normalized_tensor(
        frames, IMAGE_SIZE, offload_video_to_cpu=True, compute_device=torch.device("cpu")
    )
    theirs, uh, uw = _upstream_tensor_from_pngs(tmp_path, frames)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
    theirs = (theirs - mean) / std
    assert (h, w) == (uh, uw) == frames[0].shape[:2]
    assert torch.equal(ours, theirs)


def test_a_jpeg_route_would_have_changed_the_pixels_which_is_why_there_is_no_jpeg_route(tmp_path):
    """The confound, demonstrated rather than asserted. If this test ever goes green the other
    way — JPEG and lossless agreeing bitwise — the argument above is wrong and worth re-reading."""
    frames = _clip()
    lossless, _, _ = vid.frames_to_normalized_tensor(
        frames, IMAGE_SIZE, offload_video_to_cpu=True, compute_device=torch.device("cpu")
    )
    jpeg_frames = []
    for i, f in enumerate(frames):
        p = tmp_path / f"{i}.jpg"
        PIL_Image.fromarray(f).save(p, quality=95)
        jpeg_frames.append(np.array(PIL_Image.open(p).convert("RGB")))
    through_jpeg, _, _ = vid.frames_to_normalized_tensor(
        jpeg_frames, IMAGE_SIZE, offload_video_to_cpu=True, compute_device=torch.device("cpu")
    )
    assert not torch.equal(lossless, through_jpeg)


def test_the_ingest_refuses_a_float_frame_for_the_same_reason_the_per_frame_arm_does(tmp_path):
    """A float array in [0, 1] and one in [0, 255] are indistinguishable from the array alone and
    rescale to different pictures — which is a different mask. `apple_sam2` refuses it; an arm
    that quietly accepted it would not be seeing the same pixels."""
    with pytest.raises(Exception):  # noqa: B017 - the adapter's own refusal type is its business
        vid.frames_to_normalized_tensor(
            [np.zeros((8, 8, 3), dtype=np.float32)],
            IMAGE_SIZE,
            offload_video_to_cpu=True,
            compute_device=torch.device("cpu"),
        )


def test_the_ingest_refuses_a_clip_whose_frames_are_not_all_one_grid():
    """A propagated mask is compared against ground truth at the capture's resolution; a clip that
    changes shape partway has no such resolution."""
    frames = [np.zeros((8, 8, 3), np.uint8), np.zeros((8, 9, 3), np.uint8)]
    with pytest.raises(ValueError, match="one grid|shape"):
        vid.frames_to_normalized_tensor(
            frames, IMAGE_SIZE, offload_video_to_cpu=True, compute_device=torch.device("cpu")
        )


def test_the_module_declares_its_mechanism_and_that_it_encodes_nothing():
    """The artifact records this dict verbatim. "Both arms saw the same pixels" has to be a
    checkable claim in the file, not a sentence in a commit message."""
    contract = vid.PROPAGATION_CONTRACT
    assert contract["jpeg_encoded"] is False
    assert contract["lossy_encode_anywhere"] is False
    assert "in-memory" in contract["frame_ingest"].lower()
    assert contract["seed"]["frame_index"] == 0
    assert contract["upstream"] == apple_sam2.UPSTREAM_PROPAGATION
    # Same weights as the per-frame arm, at the same pin, or the arms differ in more than
    # propagation and the difference between the p95s is not attributable to it.
    assert contract["sam2_checkpoint"] == apple_sam2.SAM2_MODEL_CHECKPOINT
    assert contract["sam2_revision"] == apple_sam2.SAM2_MODEL_REVISION
    assert contract["object_text_prompt"] == apple_sam2.OBJECT_TEXT_PROMPT


def test_the_source_contains_no_encode_path_at_all():
    """Belt and braces on the one defect that would void the comparison silently: a helper added
    later that writes frames out to disk before handing them to the predictor."""
    src = (_REPO / "scripts" / "estimators" / "apple_sam2_video.py").read_text(encoding="utf-8")
    for banned in (".jpg", ".save(", "imwrite", "imencode", "tobytes", "BytesIO"):
        assert banned not in src, f"{banned!r} in the propagation module's source"


def test_this_module_discharges_nothing():
    """The blockers are a person's to close. A propagation arm existing is evidence, and evidence
    is not a verdict.

    This asserted ``len(...) == 3`` until 2026-08-26, when blockers 1 and 2 were discharged on
    their own evidence and the count went stale — which is the wrong tripwire, because a count has
    to be edited by every legitimate discharge and therefore stops guarding anything. The invariant
    this test is actually about is narrower and does not decay: **the blocker THIS module is the
    evidence for must still be open**, and this module must not be what closes it.
    """
    assert apple_sam2.GATE_QUALIFIED is False
    assert any(
        b.startswith("PER-FRAME SEGMENTATION IS NOT UPSTREAM'S PROPAGATION")
        for b in apple_sam2.GATE_QUALIFICATION_BLOCKERS
    ), "the propagation blocker is gone from GATE_QUALIFICATION_BLOCKERS"
    assert not any(
        "PER-FRAME SEGMENTATION IS NOT UPSTREAM'S PROPAGATION" in d
        for d in apple_sam2.GATE_QUALIFICATION_DISCHARGED
    ), "the propagation blocker was discharged; this module may not be the thing that did it"
    assert "NOTHING" in vid.PROPAGATION_CONTRACT["discharges"]
