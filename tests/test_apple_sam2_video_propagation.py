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
written, and the last test asserts exactly that — including now that the flag has moved: it flipped
on 2026-08-27 on a project owner's determination, and this module is not among the grounds it names.
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


#: The determination that flipped ``apple_sam2.GATE_QUALIFIED`` on 2026-08-27, by file name.
_DETERMINATION = "PR-08-RESULT-2026-08-27-residue-i-is-contained-and-the-flag-flips.md"


def _adapter_flag_comment() -> str:
    """The comment block committed immediately above the adapter's ``GATE_QUALIFIED``, as text.

    That block is where the adapter states what its flag rests on, so it is the only place a test
    can check that THIS module is not among those grounds.
    """
    lines = (_REPO / "scripts" / "estimators" / "apple_sam2.py").read_text(
        encoding="utf-8"
    ).splitlines()
    at = [n for n, line in enumerate(lines) if line.startswith("GATE_QUALIFIED = ")]
    assert len(at) == 1, "exactly one module-level assignment to GATE_QUALIFIED"
    end = at[0]
    start = end
    while start > 0 and lines[start - 1].startswith("#:"):
        start -= 1
    assert start < end, "GATE_QUALIFIED must keep the comment block that states its grounds"
    return " ".join(line[2:].strip() for line in lines[start:end])


def test_this_module_discharges_nothing():
    """The blockers are a person's to close. A propagation arm existing is evidence, and evidence
    is not a verdict.

    This asserted ``len(...) == 3`` until 2026-08-26, when blockers 1 and 2 were discharged on
    their own evidence and the count went stale. It then asserted the propagation blocker was still
    OPEN, and that went stale on 2026-08-27 when it was discharged. It also asserted
    ``GATE_QUALIFIED is False``, and that went stale later the same day when a project owner's
    determination flipped it. All three were the wrong tripwire for the same reason: they track
    STATE, which every legitimate decision changes, rather than the invariant this test is about —
    **whatever closes that blocker or moves that flag, it may not be the mere existence of this
    module.** Producing the evidence a blocker asks for and accepting it are different acts, and
    only the second may shorten a tuple or move a flag.

    So the guard reads the DISCHARGE and the FLAG'S GROUNDS and checks what each rests on: the
    owner's decision (T40_RULE_V14) and a measurement whose criterion was registered before the
    first capture was rendered (T40_RULE_V17) for the discharge; a named owner determination on
    disk for the flag — and in neither case "a propagation arm was written". That invariant does
    not decay, because it is about grounds rather than dates.
    """
    # The flag is True since 2026-08-27, and this file is nowhere in the reasons it gives.
    assert apple_sam2.GATE_QUALIFIED is True
    grounds = _adapter_flag_comment()
    assert _DETERMINATION in grounds, "the flag must name what flipped it"
    assert (_REPO / "docs" / "preregistration" / _DETERMINATION).is_file(), (
        "and the thing it names must be a document somebody can read, not a filename"
    )
    assert "apple_sam2_video" not in grounds, (
        "the propagation arm is evidence; it may not appear as a ground of the flag"
    )
    discharged = [
        d
        for d in apple_sam2.GATE_QUALIFICATION_DISCHARGED
        if "PER-FRAME SEGMENTATION IS NOT UPSTREAM'S PROPAGATION" in d
    ]
    assert len(discharged) == 1, "the propagation wording must be readable in exactly one place"
    entry = discharged[0]
    assert "T40_RULE_V14" in entry, "its Isaac/MuJoCo half is the owner's decision, not ours"
    assert "T40_RULE_V17" in entry, "its corpus half is a measurement with a pre-registered rule"
    # The specific bad discharge this test exists against: the arm existing, or the arm's own
    # numbers, being treated as the thing that closed the blocker the arm was built to answer.
    assert "outcome N" in entry, "the discharge must name the registered outcome it reached"
    # ONLY THE GROUNDS ARE CHECKED, not the whole entry. The retired wording is quoted verbatim
    # after the `>>>` marker and it names this module as what DROVE both arms — which is true, and
    # is exactly the distinction: being the evidence is allowed, being the reason is not.
    grounds = entry.split(">>>", 1)[0]
    assert vid.__name__.split(".")[-1] not in grounds, (
        "this module must not appear among the grounds of the discharge it is the evidence for"
    )
    assert "answered by the owner" in grounds and "registered one BEFORE" in grounds
    assert "NOTHING" in vid.PROPAGATION_CONTRACT["discharges"]


# -- the refusal mirror: what the colour filter WOULD have refused, on BOTH branches ---------------
#
# `stats()["n_frames_the_colour_filter_would_have_refused"]` exists for exactly one purpose: to be
# set beside the per-frame arm's `n_frames_mask_refused` so that the two arms' p95s are known to
# have been computed over comparable frame populations. That purpose fails silently if the two
# counters mirror different sets of branches, and since PR-08 V10 `segment()` has TWO branches on
# which it returns an all-False mask and increments `MASK_REFUSED_FRAMES`: the object-scale branch
# and the IoU branch. This module mirrored only the second until 2026-08-27.
#
# The tests below run BOTH arms over the same frames with the weights stubbed out — no GPU, no
# checkpoint, no network — and compare the counters the two arms actually moved. A source grep would
# not do: the proposed fix that these tests replace passed its own grep test while raising
# UnboundLocalError on the first frame it was meant to count.


class _FakeImagePredictor:
    """`apple_sam2._predictor()`'s surface, returning a mask the test chose.

    The per-frame arm's detection and segmentation are not what is under test here — the refusal
    accounting downstream of them is — and loading SAM 2 to assert on a counter would make this a
    test that skips whenever the weights are absent.
    """

    def __init__(self, mask: np.ndarray) -> None:
        self._mask = mask

    def set_image(self, frame: np.ndarray) -> None:
        pass

    def predict(self, box=None, multimask_output=False):
        return self._mask[None].astype(np.float32), None, None


class _FakeVideoPredictor:
    """`SAM2VideoPredictor`'s surface as `propagate()` uses it, yielding masks the test chose.

    `propagate_in_video` hands back logits at the video resolution and the module thresholds them at
    > 0, exactly as the real predictor's caller does, so the masks that reach the counting loop are
    the masks named here.
    """

    def __init__(self, masks) -> None:
        self._masks = list(masks)

    def init_state(self, video_path, offload_video_to_cpu=True, offload_state_to_cpu=True):
        return {"n_frames": len(self._masks)}

    def add_new_points_or_box(self, **kwargs):
        return None

    def propagate_in_video(self, state, start_frame_idx=0):
        for i, mask in enumerate(self._masks):
            logits = torch.where(torch.from_numpy(mask), 1.0, -1.0)[None, None]
            yield i, [vid.SEED_OBJECT_ID], logits

    def reset_state(self, state):
        return None


def _warm_scene_frame(h=48, w=64) -> np.ndarray:
    """A frame the colour reference CANNOT arbitrate: a warm table filling 40 % of the picture.

    This is `train-01-oak-tungsten` as `object_color_reference` sees it, at measured proportions
    rather than invented ones: PR-08 V10 §4.1 and the 2026-08-27 audit of job 189926's contact
    sheets measure the reference at 37.18-56.40 % of the panel on that style, against
    `MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION = 0.10`. 40 % is inside that measured population.
    """
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 2] = 200                          # cold everywhere: r low, so r > 90 fails
    frame[int(h * 0.6):, :] = (190, 120, 60)      # warm oak: r-b = 130 > 50, sat = 0.684 > 0.35
    return frame


def _table_mask(frame: np.ndarray) -> np.ndarray:
    """The mask that makes the missing branch VISIBLE: the warm table itself.

    Its IoU against the colour reference is ~1.0, so the IoU branch — the only one this module
    mirrored — says "keep". The object-scale branch, which `segment()` reaches FIRST, refuses it.
    A mask that both branches refuse would not distinguish the fixed module from the broken one.
    """
    mask = np.zeros(frame.shape[:2], dtype=bool)
    mask[int(frame.shape[0] * 0.6):, :] = True
    return mask


def _refusal_counters_of_the_per_frame_arm() -> tuple[int, int, int]:
    return (
        apple_sam2.MASK_REFUSED_FRAMES,
        apple_sam2.MASK_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES,
        apple_sam2.MASK_REFUSED_NO_REFERENCE_FRAMES,
    )


def _refusal_counters_of_the_propagation_arm() -> tuple[int, int, int]:
    return (
        vid.WOULD_HAVE_BEEN_REFUSED_FRAMES,
        vid.WOULD_HAVE_BEEN_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES,
        vid.WOULD_HAVE_BEEN_REFUSED_NO_REFERENCE_FRAMES,
    )


def _run_both_arms(monkeypatch, frames, masks):
    """The same frames and the same masks through both arms; returns each arm's counter DELTAS.

    Deltas rather than absolutes because `apple_sam2`'s counters are cumulative over the process and
    the harness reads them the same way (`counters_at_start_of_run` / `counters_at_end_of_run`).
    """
    monkeypatch.setattr(apple_sam2, "_detector", lambda: (None, None))
    monkeypatch.setattr(apple_sam2, "_best_box", lambda frame: np.array([0.0, 0.0, 1.0, 1.0]))

    before_pf = _refusal_counters_of_the_per_frame_arm()
    for frame, mask in zip(frames, masks):
        monkeypatch.setattr(apple_sam2, "_predictor", lambda mask=mask: _FakeImagePredictor(mask))
        apple_sam2.segment(frame)
    after_pf = _refusal_counters_of_the_per_frame_arm()

    monkeypatch.setattr(vid, "_video_predictor", lambda: _FakeVideoPredictor(masks))
    monkeypatch.setattr(vid, "seed_box", lambda frame: np.array([0.0, 0.0, 1.0, 1.0]))
    before_prop = _refusal_counters_of_the_propagation_arm()
    got = vid.propagate(frames)
    after_prop = _refusal_counters_of_the_propagation_arm()
    assert len(got) == len(frames)

    return (
        tuple(a - b for a, b in zip(after_pf, before_pf)),
        tuple(a - b for a, b in zip(after_prop, before_prop)),
    )


def test_the_fixture_reproduces_the_object_scale_misfire_rather_than_assuming_it():
    """The premise, measured on the fixture: the reference is scene-scale AND the IoU passes.

    Both halves matter. If the reference were object-scale the frame would not exercise the missing
    branch at all; if the IoU failed, the branch this module already had would refuse the frame and
    the two arms would agree by accident.
    """
    frame = _warm_scene_frame()
    reference = apple_sam2.object_color_reference(frame)
    fraction = apple_sam2.reference_frame_fraction(reference)
    assert fraction > apple_sam2.MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION, (
        f"the fixture must reproduce the measured misfire; got {fraction:.4f}"
    )
    assert apple_sam2.reference_is_object_scale(reference) is False
    iou = apple_sam2.mask_validity_iou(_table_mask(frame), reference)
    assert iou >= apple_sam2.MASK_VALIDITY_MIN_IOU, (
        "the IoU branch must PASS this mask, or the test cannot tell a two-branch mirror from a "
        f"one-branch one; got {iou:.4f}"
    )


def test_both_arms_count_the_object_scale_refusal_identically(monkeypatch):
    """THE DEFECT, AS A TEST. Warm background, mask on the table: `segment()` refuses on the V10
    object-scale branch, and until 2026-08-27 the propagation arm counted zero.

    The assertion is equality of the two arms' counters, not a fixed number, because the claim the
    artifact makes about this pair is a comparison and not a level. Before the fix this reads
    `(1, 1, 0) != (0, 0, 0)` on a single frame — the same 1-vs-0 disagreement the three MuJoCo
    captures already show between `n_frames_mask_refused` and this counter.
    """
    frames = [_warm_scene_frame()]
    masks = [_table_mask(frames[0])]
    per_frame, propagation = _run_both_arms(monkeypatch, frames, masks)
    assert per_frame == (1, 1, 0), "the per-frame arm must refuse via the object-scale branch"
    assert propagation == per_frame, (
        "the propagation arm's would-have-refused counters must be the per-frame arm's refusal "
        "counters recomputed over this arm's masks; a mirror of one of the two branches is a "
        "comparison of two different refusal populations, not a smaller comparison"
    )


def test_both_arms_count_the_iou_branch_and_its_empty_reference_case_identically(monkeypatch):
    """The branch this module always had, plus the sub-attribution inside it, still agree.

    A frame with no warm pixels at all: the reference is EMPTY, which `reference_is_object_scale`
    deliberately calls object-scale (an empty reference is not a scene), so `segment()` falls
    through to the IoU branch, refuses at IoU 0.0 and attributes the refusal to "nothing here can
    confirm the mask" rather than to "the mask is demonstrably the plate". The two have opposite
    implications for the budget, so an arm that pooled them would not be answering the same
    question as the arm it is subtracted from.
    """
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[:, :, 2] = 200                       # cold everywhere: the reference is empty
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:12, 4:12] = True
    per_frame, propagation = _run_both_arms(monkeypatch, [frame], [mask])
    assert per_frame == (1, 0, 1), "empty reference -> IoU branch, counted as no_reference"
    assert propagation == per_frame


def test_a_mask_the_filter_accepts_moves_no_counter_on_either_arm(monkeypatch):
    """The counterpart, so the mirror cannot be satisfied by counting everything.

    A warm apple on a cold background — the corpus the reference was built for, where its own
    justification holds — is accepted by both branches, and neither arm records a refusal.
    """
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[:, :, 2] = 200
    frame[10:16, 10:16] = (220, 30, 20)        # ~3.5 % of the frame: an object-scale reference
    mask = np.zeros((32, 32), dtype=bool)
    mask[10:16, 10:16] = True
    per_frame, propagation = _run_both_arms(monkeypatch, [frame], [mask])
    assert per_frame == (0, 0, 0)
    assert propagation == per_frame


def test_an_empty_propagated_mask_is_not_mirrored_because_segment_returns_before_the_filter(
    monkeypatch,
):
    """`segment()` counts an empty mask in EMPTY_MASK_FRAMES and returns BEFORE the validity check.

    A mirror that ran the filter on an empty mask would refuse it (IoU 0.0 against a non-empty
    reference) and manufacture a refusal the per-frame arm never records.
    """
    frame = _warm_scene_frame()
    empty = np.zeros(frame.shape[:2], dtype=bool)
    before = vid.EMPTY_PROPAGATED_FRAMES
    per_frame, propagation = _run_both_arms(monkeypatch, [frame], [empty])
    assert per_frame == (0, 0, 0)
    assert propagation == (0, 0, 0)
    assert vid.EMPTY_PROPAGATED_FRAMES == before + 1


def test_a_label_the_filter_has_no_reference_for_is_counted_as_unevaluable_not_as_agreement(
    monkeypatch,
):
    """The zero that would otherwise be read as "the filter would have refused nothing".

    For a label outside `MASK_VALIDITY_REFERENCE_LABELS` the per-frame arm refuses the RUN on its
    first call, before any counter moves (PR-08 V10 §2) — so there is no per-frame refusal count in
    existence to compare against. This arm does not adopt that run-level refusal, because whether
    §6's second label is measured with the filter off is an owner decision and not this module's;
    what it must not do is report a silent zero that reads as agreement.
    """
    monkeypatch.setattr(apple_sam2, "OBJECT_TEXT_PROMPT", "plate.")
    assert apple_sam2.mask_validity_reference_is_defined() is False
    with pytest.raises(apple_sam2.MaskValidityReferenceUndefined):
        apple_sam2.segment(_warm_scene_frame())

    frames = [_warm_scene_frame()]
    masks = [_table_mask(frames[0])]
    monkeypatch.setattr(vid, "_video_predictor", lambda: _FakeVideoPredictor(masks))
    monkeypatch.setattr(vid, "seed_box", lambda frame: np.array([0.0, 0.0, 1.0, 1.0]))
    before_refused = vid.WOULD_HAVE_BEEN_REFUSED_FRAMES
    before_unevaluable = vid.FILTER_MIRROR_UNEVALUABLE_FRAMES
    vid.propagate(frames)
    assert vid.WOULD_HAVE_BEEN_REFUSED_FRAMES == before_refused
    assert vid.FILTER_MIRROR_UNEVALUABLE_FRAMES == before_unevaluable + 1
    assert vid.stats()["n_frames_the_colour_filter_mirror_could_not_evaluate"] >= 1


def test_the_counters_are_reported_and_zeroed_as_a_set():
    """A counter that `reset_counters()` forgets carries the previous run's frames into this one,
    and a counter `stats()` forgets never reaches the artifact at all — which is how a mirror comes
    to be believed rather than checked."""
    vid.WOULD_HAVE_BEEN_REFUSED_FRAMES = 3
    vid.WOULD_HAVE_BEEN_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES = 2
    vid.WOULD_HAVE_BEEN_REFUSED_NO_REFERENCE_FRAMES = 1
    vid.FILTER_MIRROR_UNEVALUABLE_FRAMES = 4
    reported = vid.stats()
    assert reported["n_frames_the_colour_filter_would_have_refused"] == 3
    assert reported["n_frames_the_colour_filter_would_have_refused_reference_not_object_scale"] == 2
    assert reported["n_frames_the_colour_filter_would_have_refused_no_reference"] == 1
    assert reported["n_frames_the_colour_filter_mirror_could_not_evaluate"] == 4
    assert reported["colour_filter_mirror_covers_both_refusal_branches"] is True
    vid.reset_counters()
    zeroed = vid.stats()
    for key, value in zeroed.items():
        if key.startswith("n_frames") or key == "n_propagation_runs":
            assert value == 0, f"{key} survived reset_counters()"


def test_the_contract_names_which_of_this_arms_counters_is_which_of_the_per_frame_arms():
    """The artifact records PROPAGATION_CONTRACT verbatim, and the only use of a would-have-refused
    count is a subtraction against the other arm's. A reader has to be able to see which counter
    pairs with which, and that both branches are in the pair, without reading this module."""
    mirror = vid.PROPAGATION_CONTRACT["colour_filter_mirror"]
    assert set(mirror["branches_mirrored"]) == {"reference_is_object_scale", "mask_validity_iou"}
    assert "MASK_REFUSED_FRAMES" in mirror["n_frames_the_colour_filter_would_have_refused"]
    for key in mirror:
        if key.startswith("n_frames"):
            assert key in vid.stats(), f"{key} is described in the contract and never reported"


def test_the_flag_this_arm_publishes_is_pinned_to_segments_actual_branch_COUNT():
    """`stats()["colour_filter_mirror_covers_both_refusal_branches"]` is a hard-coded `True` that
    travels into every EST_DRIFT artifact, so something has to make it false when it stops being
    true.

    The behavioural tests above run both arms over frames chosen to hit the two branches that exist
    today; by construction they cannot hit a THIRD branch nobody has written yet, which is exactly
    the change that would make the published flag a lie and turn the two arms' refusal counts back
    into different quantities. This one is deliberately a source-level tripwire on `apple_sam2` —
    read-only, nothing here writes to that module — and it fires in the commit that adds the branch
    rather than in the artifact that was pooled with it.
    """
    source = (_REPO / "scripts" / "estimators" / "apple_sam2.py").read_text(encoding="utf-8")
    body = source[source.index("\ndef segment(") : source.index("\ndef estimate_depth(")]
    refusal_branches = body.count("MASK_REFUSED_FRAMES += 1")
    mirrored = vid.PROPAGATION_CONTRACT["colour_filter_mirror"]["branches_mirrored"]
    assert refusal_branches == len(mirrored), (
        f"apple_sam2.segment() now increments MASK_REFUSED_FRAMES on {refusal_branches} branches "
        f"and this arm mirrors {len(mirrored)} ({list(mirrored)}). Until the counting loop in "
        "apple_sam2_video.propagate() follows, the two arms' refusal counters are computed over "
        "different frame populations and their difference is not a difference of anything — while "
        "stats() keeps publishing colour_filter_mirror_covers_both_refusal_branches: True."
    )
    assert vid.stats()["colour_filter_mirror_covers_both_refusal_branches"] is True
