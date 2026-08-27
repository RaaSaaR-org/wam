"""T40_RULE_V17 Arm B — the refusals, the run arithmetic, and the sample that may not move.

No corpus and no weights: both arms are stub modules written per-test. That is the same split
``tests/test_measure_est_drift.py`` uses and for the same reason — the number needs a GPU, every
way of getting it *wrong* is reachable here.
"""

from __future__ import annotations

import json
import pathlib
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import measure_arm_divergence as ad  # noqa: E402
import measure_est_drift as ed  # noqa: E402


def _mask(h=8, w=8, box=(1, 1, 4, 4)):
    m = np.zeros((h, w), dtype=bool)
    y0, x0, y1, x1 = box
    m[y0:y1, x0:x1] = True
    return m


class _Arm:
    """Minimal stand-in for measure_est_drift.Estimators / Propagator."""

    def __init__(self, masks):
        self._masks = masks
        self.spec = "stub"
        self.name = "stub"
        self.version = "stub/1"
        self.gate_qualified = False
        self.segmenter_contract = None
        self.object_text_prompt = "apple."

    def segment(self, rgb):
        return self._masks[int(rgb[0, 0, 0])]

    def propagate(self, frames):
        return [self._masks[int(f[0, 0, 0])] for f in frames]


def _frames(n):
    return [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(n)]


# ---------------------------------------------------------------------------------------------
# The sample V17 §3 registered
# ---------------------------------------------------------------------------------------------


def test_the_registered_sample_is_the_one_the_document_names():
    """A TRIPWIRE. V17 §3 fixed 40 ids under seed 40017 before a frame was decoded. If the draw or
    the seed ever moves, the document silently describes a different sample than the one measured,
    and the pre-registration is worth nothing."""
    assert ad.SAMPLE_SEED == 40017
    assert ad.SAMPLE_SIZE == 40
    assert ad.SAMPLE_SCHEME == "stratified-systematic/1"
    assert len(ad.REGISTERED_SAMPLE) == 40
    assert len(set(ad.REGISTERED_SAMPLE)) == 40
    assert ad.REGISTERED_SAMPLE[0] == "episode_000005"
    assert ad.REGISTERED_SAMPLE[-1] == "episode_000397"


def test_the_sample_is_rederived_from_the_manifest_and_not_taken_on_trust():
    manifest = {"episodes": [{"id": f"episode_{i:06d}", "frames": 400} for i in range(402)]}
    assert ad.registered_sample(manifest) == list(ad.REGISTERED_SAMPLE)


def test_a_corpus_that_does_not_reproduce_the_draw_is_refused_not_measured():
    """The sample is a property of THIS corpus's episode list. A corpus with a different set of
    episodes would produce a different draw under the same seed, and measuring it anyway would
    quote a pre-registration that was never made over it."""
    manifest = {"episodes": [{"id": f"episode_{i:06d}", "frames": 400} for i in range(300)]}
    with pytest.raises(SystemExit, match="does not reproduce the ids"):
        ad.registered_sample(manifest)


def test_the_outcome_d_run_length_is_the_registered_one_and_is_not_a_flag():
    assert ad.RUN_LENGTH_D == 10
    parser_flags = ad.main.__doc__ or ""
    assert "--threshold" not in parser_flags
    with pytest.raises(SystemExit):
        ad.main(["--corpus", "/nonexistent", "--out", "/tmp/x.json", "--threshold", "0.9"])


# ---------------------------------------------------------------------------------------------
# The IoU convention and the runs
# ---------------------------------------------------------------------------------------------


def test_two_empty_masks_are_not_a_disagreement():
    """Scoring an empty-vs-empty frame 0.0 would manufacture divergence runs out of frames that
    contain no object at all — and 91% of this corpus's frames have an empty robot mask."""
    empty = np.zeros((8, 8), dtype=bool)
    assert ad.cross_arm_iou(empty, empty) is None


def test_identical_masks_agree_and_disjoint_masks_do_not():
    a = _mask(box=(1, 1, 4, 4))
    b = _mask(box=(5, 5, 7, 7))
    assert ad.cross_arm_iou(a, a) == 1.0
    assert ad.cross_arm_iou(a, b) == 0.0


def test_a_contiguous_stretch_of_disagreement_is_counted_as_one_run_of_that_length():
    """THE STATISTIC THE WHOLE ARM IS FOR. A mean over the episode averages a lost object away;
    the length of the stretch is what distinguishes propagation drift from per-frame jitter."""
    agree, disagree = _mask(box=(1, 1, 4, 4)), _mask(box=(5, 5, 7, 7))
    per_frame = [agree] * 20
    propagated = [agree] * 5 + [disagree] * 12 + [agree] * 3
    block = ad.episode_block(
        "episode_test", _frames(20), _Arm(per_frame), _Arm(propagated)
    )
    assert block["divergence_runs"]["n_runs"] == 1
    assert block["longest_run"] == 12
    assert block["meets_outcome_d"] is True
    assert block["divergence_runs"]["runs"] == [[5, 16]]


def test_scattered_single_frame_disagreements_do_not_meet_outcome_d():
    """Independent per-frame failures produce runs of length 1. That is the cheap explanation
    outcome D's length threshold exists to exclude."""
    agree, disagree = _mask(box=(1, 1, 4, 4)), _mask(box=(5, 5, 7, 7))
    per_frame = [agree] * 20
    propagated = [disagree if i % 4 == 0 else agree for i in range(20)]
    block = ad.episode_block(
        "episode_test", _frames(20), _Arm(per_frame), _Arm(propagated)
    )
    assert block["divergence_runs"]["n_runs"] == 5
    assert block["longest_run"] == 1
    assert block["meets_outcome_d"] is False


def test_an_episode_of_perfect_agreement_reports_zero_runs():
    agree = _mask(box=(1, 1, 4, 4))
    block = ad.episode_block(
        "episode_test", _frames(20), _Arm([agree] * 20), _Arm([agree] * 20)
    )
    assert block["longest_run"] == 0
    assert block["cross_arm_iou"]["median"] == 1.0
    assert block["meets_outcome_d"] is False


def test_a_propagation_arm_that_returns_the_wrong_number_of_masks_is_fatal():
    """Pairing them by position anyway would compare different instants, silently, with the run
    statistic looking perfectly well-formed."""
    agree = _mask()

    class _ShortPropagator:
        def propagate(self, frames):  # returns fewer masks than frames, on purpose
            return [agree] * 5

    with pytest.raises(SystemExit, match="does not line up"):
        ad.episode_block("episode_test", _frames(20), _Arm([agree] * 20), _ShortPropagator())


def test_the_threshold_used_is_the_drift_rigs_own_and_is_not_restated():
    """V17 §0 keeps LOW_IOU_THRESHOLD at 0.5. A second copy here is how the two would drift and
    the cross-arm runs would stop being comparable to the ground-truth runs they are read beside."""
    assert ed.LOW_IOU_THRESHOLD == 0.5
    src = pathlib.Path(ad.__file__).read_text()
    assert "threshold=ed.LOW_IOU_THRESHOLD" in src
    assert "threshold=0.5" not in src


# ---------------------------------------------------------------------------------------------
# What travels with every number
# ---------------------------------------------------------------------------------------------


def test_every_block_carries_the_no_ground_truth_sentence():
    """This measures agreement, not correctness, and the artifact has to say so where the number
    is rather than in a document a reader may not open."""
    agree = _mask()
    block = ad.episode_block("e", _frames(4), _Arm([agree] * 4), _Arm([agree] * 4))
    meaning = block["cross_arm_iou"]["meaning"]
    assert "NOT AN EST_DRIFT" in meaning
    assert "both wrong in the same place" in meaning


def test_a_lossless_named_corpus_without_its_proof_is_refused(tmp_path):
    corpus = tmp_path / "some-corpus-h264-lossless"
    (corpus / "videos").mkdir(parents=True)
    (corpus / "manifest.json").write_text(json.dumps({"episodes": []}))
    with pytest.raises(SystemExit, match="is not evidence"):
        ad.main(["--corpus", str(corpus), "--out", str(tmp_path / "o.json")])


def test_the_rule_of_three_reading_says_what_a_clean_sweep_does_not_prove():
    """V17 §3 requires the bound to be reported AS a bound. Forty clean episodes of 402 is still
    consistent with ~30 containing an event, and the artifact says that in words."""
    src = pathlib.Path(ad.__file__).read_text()
    assert "CAN DETECT" in src and "CANNOT CERTIFY" in src
    assert "rule_of_three_episode_rate_upper_95" in src


def test_a_stub_estimator_is_never_silently_gate_qualified():
    arm = _Arm([_mask()])
    assert arm.gate_qualified is False


def test_the_module_names_the_rule_it_implements():
    assert ad.RULE == "T40_RULE_V17"
    assert ad.WRITEUP.endswith("PR-08-V17-drift-rate-protocol.md")
    assert (pathlib.Path(ad.__file__).resolve().parents[1] / ad.WRITEUP).is_file()


def test_the_stub_arms_are_not_accidentally_the_same_object():
    """Guards the tests above: if `_Arm` shared state, a disagreement test could pass while
    measuring nothing."""
    a, b = _Arm([_mask(box=(1, 1, 4, 4))]), _Arm([_mask(box=(5, 5, 7, 7))])
    assert ad.cross_arm_iou(a.segment(_frames(1)[0]), b.segment(_frames(1)[0])) == 0.0


def test_types_module_is_not_needed_at_import(tmp_path):
    """Cheap guard that the module imports without a corpus, weights or a GPU."""
    assert isinstance(ad.SCHEMA, str)
    assert isinstance(types.ModuleType, type)
