"""C40's scorer, tested against the failure it exists to catch.

T-041's judge answered the literal string ``"NO"`` to all 80 items and its calibration reported
``calibration_correct: 10/20`` — a number that reads like partial credit. Every test below asks the
same question in a different spelling: **does a classifier that never looked at anything get
caught?** If any of them ever starts passing, the scorer has stopped being able to detect the one
failure mode it was written for, and no number it prints about a real instrument means anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import build_identity_calibration as C  # noqa: E402

CAL_DIR = REPO / "runs" / "t040-identity-prompt" / "calibration"


# --------------------------------------------------------------------------------------------
# a stand-in key with the registered class sizes, so the tests do not depend on runs/ artifacts
# --------------------------------------------------------------------------------------------


def make_key() -> dict[str, dict]:
    """15 positives / 15 negatives / 10 probes, the sizes the five floors are written over."""
    key: dict[str, dict] = {}
    n = 0

    def add(cls: str, axis: str | None = None, side: str | None = None,
            permitted: list[str] | None = None) -> None:
        nonlocal n
        n += 1
        key[f"item_{n:04d}"] = {
            "kind": "calibration", "class": cls, "episode": f"episode_{n:06d}",
            "mutation": f"synthetic:{cls}", "side": side,
            "required_verdict": C.REQUIRED_VERDICT[cls],
            "required_axis": axis,
            "permitted_axes": permitted if permitted is not None else ([axis] if axis else []),
        }

    for _ in range(C.N_POSITIVE):
        add("positive")
    for axis in ("apple", "table", "background", "lighting", "plate"):
        add("negative", axis, "image")
        add("negative", axis, "image")
        add("negative", axis, "prompt")
    for _ in range(C.N_PROBE):
        add("probe")
    return key


def perfect_answers(key: dict[str, dict]) -> dict[str, dict]:
    return {
        i: {"verdict": k["required_verdict"],
            "mismatched_axes": [k["required_axis"]] if k["required_axis"] else []}
        for i, k in key.items()
    }


def floor_named(scores: dict, fragment: str) -> dict:
    return next(f for f in scores["floors"] if fragment in f["name"])


# --------------------------------------------------------------------------------------------
# the four degenerate vectors the whole set exists to reject
# --------------------------------------------------------------------------------------------


def test_a_constant_match_answer_vector_fails_the_calibration():
    key = make_key()
    s = C.score_answers(key, C.degenerate_vectors(key)["constant_match"])
    assert not s["passed"]
    assert floor_named(s, "positives")["value"] == C.N_POSITIVE          # it aces exactly one class
    assert floor_named(s, "negatives answered")["value"] == 0
    assert floor_named(s, "abstention probes")["value"] == 0


def test_a_constant_mismatch_answer_vector_with_one_axis_fails_the_calibration():
    """T-041's own judge, transliterated: one token, one axis, never a decision."""
    key = make_key()
    s = C.score_answers(key, C.degenerate_vectors(key)["constant_mismatch_one_axis"])
    assert not s["passed"]
    assert floor_named(s, "positives")["value"] == 0
    assert floor_named(s, "abstention probes")["value"] == 0
    # It earns exactly the three `table` negatives on the axis line, which is the arithmetic
    # docs §3.5 predicts, and its aggregate over all forty would be the 37.5 % that looked like
    # partial credit in T-041. The report is five numbers and the first one is a zero.
    assert floor_named(s, "negatives naming")["value"] == 3


def test_a_constant_unsure_answer_vector_fails_four_of_the_five_floors():
    key = make_key()
    s = C.score_answers(key, C.degenerate_vectors(key)["constant_unsure"])
    assert not s["passed"]
    assert floor_named(s, "abstention probes")["value"] == C.N_PROBE     # the one line it clears
    assert floor_named(s, "leaked")["value"] == C.N_POSITIVE + C.N_NEGATIVE
    # docs §3.5 says "Fails three of five lines" for this vector; it fails FOUR, because a token
    # that is never `mismatch` cannot earn an axis either and the axis line is its own floor. The
    # doc undercounts, the scorer does not, and the direction of the disagreement is the safe one.
    assert sum(1 for f in s["floors"] if not f["passed"]) == 4


def test_a_uniform_coin_flip_fails_the_calibration():
    key = make_key()
    s = C.score_answers(key, C.degenerate_vectors(key)["uniform_coin_flip"])
    assert not s["passed"]


def test_a_constant_mismatch_naming_every_axis_still_fails_the_registered_axis_rule():
    """The classifier that games the PERMISSIVE reading of "named the required axis".

    Answering `mismatch` with all six axes names the required one on every negative. That is why
    the registered floor is the strict reading — required axis named, at most two axes in total,
    all of them inside the item's own `permitted_axes` — and why the permissive count is reported
    beside it rather than as it.
    """
    key = make_key()
    s = C.score_answers(key, C.degenerate_vectors(key)["constant_mismatch_all_axes"])
    assert not s["passed"]
    assert s["counts"]["negative_axes_permissive"] == C.N_NEGATIVE
    assert s["counts"]["negative_axes"] == 0


def test_every_degenerate_vector_fails_and_the_generator_produces_all_of_them():
    key = make_key()
    vectors = C.degenerate_vectors(key)
    assert {"constant_match", "constant_mismatch_one_axis", "constant_unsure",
            "uniform_coin_flip"} <= set(vectors)
    for name, vec in vectors.items():
        assert not C.score_answers(key, vec)["passed"], f"{name} passed the calibration"


# --------------------------------------------------------------------------------------------
# the per-axis requirement is what makes a negative harder than a coin flip
# --------------------------------------------------------------------------------------------


def test_the_per_axis_requirement_makes_a_negative_much_harder_than_a_coin_flip():
    """A negative needs the token AND the axis, so guessing it is 2/21, not 1/2 and not 1/3.

    docs §3.5 makes this argument with a 1/18 per-item figure for a uniform-over-six-axes guesser.
    The registered rule here admits a second axis, so the arithmetic is redone rather than quoted:
    6 singletons + 15 pairs = 21 axis answers a guesser can give, 6 of them contain the required
    axis.
    """
    p = C.coin_probabilities()
    assert p["p_negative_token_and_axis_correct"] == pytest.approx(2 / 21)
    assert p["p_negative_token_and_axis_correct"] < 0.5 * p["p_negative_token_correct"]
    key = "P(>=%d of %d negative axes)" % (C.FLOOR_NEGATIVE_AXIS, C.N_NEGATIVE)
    assert p[key] < 1e-11
    # And the token-only floor is already far out of a guesser's reach, so the axis line is not
    # carrying the whole argument on its own.
    assert p["P(>=%d of %d negative tokens)" % (C.FLOOR_NEGATIVE_TOKEN, C.N_NEGATIVE)] < 1e-5


def test_naming_an_axis_outside_the_items_permitted_set_loses_the_axis_but_keeps_the_token():
    key = make_key()
    answers = perfect_answers(key)
    victim = next(i for i, k in key.items()
                  if k["class"] == "negative" and k["required_axis"] == "apple")
    answers[victim] = {"verdict": "mismatch", "mismatched_axes": ["apple", "plate"]}
    s = C.score_answers(key, answers)
    row = next(r for r in s["per_item"] if r["item_id"] == victim)
    assert row["token_correct"] and not row["axis_correct"]
    assert row["axis_correct_permissive"]          # the permissive reading would have paid out


def test_a_permitted_second_axis_keeps_the_axis_credit():
    """An image-side cloth tint really does falsify the background clause: naming both is right."""
    key = make_key()
    victim = next(i for i, k in key.items()
                  if k["class"] == "negative" and k["required_axis"] == "table")
    key[victim]["permitted_axes"] = ["background", "table"]
    answers = perfect_answers(key)
    answers[victim] = {"verdict": "mismatch", "mismatched_axes": ["table", "background"]}
    assert C.score_answers(key, answers)["passed"]


def test_naming_three_axes_loses_the_axis_credit_even_when_all_three_are_permitted():
    key = make_key()
    victim = next(i for i, k in key.items()
                  if k["class"] == "negative" and k["required_axis"] == "lighting")
    key[victim]["permitted_axes"] = ["background", "lighting", "table"]
    answers = perfect_answers(key)
    answers[victim] = {"verdict": "mismatch",
                       "mismatched_axes": ["lighting", "table", "background"]}
    s = C.score_answers(key, answers)
    assert not next(r for r in s["per_item"] if r["item_id"] == victim)["axis_correct"]


# --------------------------------------------------------------------------------------------
# the floors themselves
# --------------------------------------------------------------------------------------------


def test_a_perfect_answer_vector_clears_all_five_floors():
    """The scorer has to be passable, or the four failures above prove nothing."""
    key = make_key()
    s = C.score_answers(key, perfect_answers(key))
    assert s["passed"]
    assert [f["value"] for f in s["floors"]] == [C.N_POSITIVE, C.N_NEGATIVE, C.N_NEGATIVE,
                                                 C.N_PROBE, 0]


def test_two_abstentions_among_the_decidable_items_fail_the_leakage_floor_on_their_own():
    """docs §3.4's fifth line, which is the one that is easy to leave out.

    Without it, an instrument abstains its way past every hard item, clears the first four lines on
    what is left, and then abstains through the real forty — where --min-coverage 0.90 stamps the
    run not gate-qualified after the wall-clock has been spent.
    """
    key = make_key()
    answers = perfect_answers(key)
    decidable = [i for i, k in key.items() if k["class"] in ("positive", "negative")]
    for i in decidable[:2]:
        answers[i] = {"verdict": "unsure", "mismatched_axes": []}
    s = C.score_answers(key, answers)
    assert not s["passed"]
    assert [f["name"] for f in s["floors"] if not f["passed"]] == [
        floor_named(s, "leaked")["name"]] or not floor_named(s, "leaked")["passed"]
    assert floor_named(s, "leaked")["value"] == 2


def test_one_abstention_among_the_decidable_items_is_still_within_the_leakage_ceiling():
    key = make_key()
    answers = perfect_answers(key)
    victim = next(i for i, k in key.items() if k["class"] == "positive")
    answers[victim] = {"verdict": "unsure", "mismatched_axes": []}
    s = C.score_answers(key, answers)
    assert floor_named(s, "leaked")["passed"]
    assert not floor_named(s, "positives")["passed"] or floor_named(s, "positives")["value"] == 14


def test_the_scorer_reports_five_numbers_and_never_an_aggregate():
    key = make_key()
    s = C.score_answers(key, perfect_answers(key))
    assert len(s["floors"]) == 5
    assert "aggregate" not in s and "score" not in s and "percent" not in s


def test_the_scorer_refuses_a_calibration_that_is_only_partly_answered():
    key = make_key()
    answers = perfect_answers(key)
    answers.pop(sorted(answers)[0])
    with pytest.raises(C.CalibrationError, match="no answer"):
        C.score_answers(key, answers)


def test_the_scorer_refuses_a_key_whose_class_sizes_are_not_the_registered_ones():
    key = make_key()
    key.pop(next(i for i, k in key.items() if k["class"] == "probe"))
    with pytest.raises(C.CalibrationError, match="pass rule is written over"):
        C.score_answers(key, perfect_answers(key))


def test_an_illegal_verdict_token_is_recorded_rather_than_mapped_onto_a_legal_one():
    """T-041's judge answered the literal string "NO", which is in nobody's vocabulary."""
    key = make_key()
    answers = {i: {"verdict": "NO", "mismatched_axes": []} for i in key}
    s = C.score_answers(key, answers)
    assert not s["passed"]
    assert s["illegal_tokens"] == ["no"]
    assert s["counts"] == {"positives_match": 0, "negative_tokens": 0, "negative_axes": 0,
                           "negative_axes_permissive": 0, "probes_unsure": 0,
                           "abstention_leakage": 0}


# --------------------------------------------------------------------------------------------
# the prompt template — the items are substitutions into it, so it has to be the committed string
# --------------------------------------------------------------------------------------------


def test_the_template_reproduces_the_committed_identity_prompt_byte_for_byte():
    import check_style_partition as csp  # noqa: PLC0415
    doc = csp.load(REPO / "configs" / "transfer25" / "styles.toml")
    C.check_template(doc["identity_style"]["prompt"])


def test_falsifying_an_axis_rewrites_every_slot_that_axis_owns():
    """The prompt names the apple twice and the plate three times.

    A prompt that says "green apple" in the first sentence and "red/yellow apple" in the fourth is
    incoherent, not falsified on one clause, and an instrument that answers `mismatch` to it has
    told us nothing about whether it looked at the picture.
    """
    for axis, replacement in C.FALSIFIED_SLOTS.items():
        slots = C.substitute(C.COMMITTED_SLOTS, axis, replacement)
        rendered = C.render_prompt(slots)
        for slot in C.AXIS_SLOTS[axis]:
            assert C.COMMITTED_SLOTS[slot] not in rendered, f"{axis}: {slot} survived"
            assert replacement[slot] in rendered
        untouched = [s for s in C.SLOT_NAMES if s not in C.AXIS_SLOTS[axis]]
        for slot in untouched:
            assert C.COMMITTED_SLOTS[slot] in rendered


def test_a_partial_axis_substitution_is_refused():
    with pytest.raises(C.CalibrationError, match="contradicts itself"):
        C.substitute(C.COMMITTED_SLOTS, "plate", {"plate_long": "A small dark square tray"})


def test_a_template_that_does_not_reproduce_the_committed_prompt_is_fatal():
    with pytest.raises(C.CalibrationError, match="does not reproduce"):
        C.check_template("A red apple on a table.")


# --------------------------------------------------------------------------------------------
# the built artifacts, when they are there
# --------------------------------------------------------------------------------------------


def _key_or_skip() -> dict:
    p = CAL_DIR / "key.json"
    if not p.is_file():
        pytest.skip("no built C40 in runs/ — build it with build_identity_calibration.py build")
    return json.loads(p.read_text())


def test_the_built_c40_has_the_class_sizes_the_pass_rule_is_written_over():
    items = _key_or_skip()["items"]
    cal = [v for v in items.values() if v["kind"] == "calibration"]
    counts = {c: sum(1 for v in cal if v["class"] == c) for c in ("positive", "negative", "probe")}
    assert counts == {"positive": C.N_POSITIVE, "negative": C.N_NEGATIVE, "probe": C.N_PROBE}
    assert sum(1 for v in items.values() if v["kind"] == "real_row") == 40


def test_the_built_c40_puts_three_negatives_on_each_of_the_five_axes_across_both_sides():
    items = _key_or_skip()["items"]
    neg = [v for v in items.values() if v["kind"] == "calibration" and v["class"] == "negative"]
    for axis in ("apple", "table", "background", "lighting", "plate"):
        on_axis = [v for v in neg if v["required_axis"] == axis]
        assert len(on_axis) == 3, axis
        assert {v["side"] for v in on_axis} == {"image", "prompt"}, (
            f"{axis} is all one mutation side; docs §3.3 wants both, because an instrument that "
            "passes only the image-side items is detecting artefacts and one that passes only the "
            "prompt-side items is reading the text and not the picture")


def test_the_issued_items_carry_nothing_but_a_frame_and_a_prompt():
    """The blinding, checked as a fact about the file rather than as an intention."""
    p = CAL_DIR / "items.jsonl"
    if not p.is_file():
        pytest.skip("no built C40 in runs/")
    rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    assert len(rows) == 80
    for r in rows:
        assert set(r) == {"item_id", "frame", "prompt"}
        assert Path(r["frame"]).name == f"{r['item_id']}.png", (
            "the FILENAME is the leak: `frames/episode_000003.png` tells the instrument it is "
            "looking at a real corpus row and `neg19_..._img-table_blue.png` tells it the answer")


# --------------------------------------------------------------------------------------------
# attempt 2's probes: found in the corpus rather than composited over it
#
# Attempt 1's ten probes pasted the corpus's own gripper over the apple, and eight of ten were
# answered `mismatch` — three of those naming `other` and calling the pasted region a "smeared,
# blocky black artifact". `mismatch: other` is a defensible answer to "there is an artefact over
# the apple", so those items may have been measuring artefact-detection rather than abstention.
# Everything below is about the replacement: probes that are UNMODIFIED corpus frames, whose only
# warrant is a measurement plus somebody having looked.
# --------------------------------------------------------------------------------------------

CAL2_DIR = REPO / "runs" / "t040-identity-prompt" / "calibration-2"


def _natural_assertions(**over) -> dict:
    a = {"apple_warm_px": 300, "apple_warm_ratio": 0.05, "ring_foreground_fraction": 0.9,
         "looked_at": True, "other_clauses_true": True, "frame_index": 133,
         "episode_median_apple_warm_px": 5633.0, "undecidable_because": "the hand is over it"}
    a.update(over)
    return a


def _frame(value: float = 0.0):
    np = pytest.importorskip("numpy")
    return np.full((8, 8, 3), value, dtype=np.float32)


def test_the_two_probe_sets_share_the_thirty_decidable_items_exactly():
    """The half that passed must not be quietly re-drawn while the half that failed is rebuilt.

    Re-running a calibration after a failure is only honest if the change is confined to what
    failed. If `item_plan` ever let the positives or the negatives differ between the attempts,
    attempt 2's first three floors would be about a different set of items than attempt 1's and
    the two runs could not be compared at all.
    """
    a = C.item_plan(C.PROBE_SET_COMPOSITE)
    b = C.item_plan(C.PROBE_SET_NATURAL)
    assert a[:30] == b[:30] == C.ITEM_PLAN_DECIDABLE
    assert len(a) == len(b) == 40
    assert {r[0] for r in a[:30]} == {"positive", "negative"}
    assert all(r[2] == "probe:natural" for r in b[30:])
    assert C.ITEM_PLAN == a          # the plan attempt 1 ran is still the whole plan


def test_an_unknown_probe_set_is_refused_rather_than_defaulted():
    with pytest.raises(C.CalibrationError, match="unknown probe set"):
        C.item_plan("whatever")


def test_a_natural_probe_whose_pixels_were_touched_is_refused():
    """The one property a natural probe has that a composite does not: nothing was done to it."""
    with pytest.raises(C.CalibrationError, match="corpus's own pixels"):
        C._assert_mutation_landed("probe:natural", _frame(0.0), _frame(1.0),
                                  _natural_assertions())


def test_a_natural_probe_with_too_much_apple_left_is_refused():
    """docs §4: a probe a careful person can still answer is a hard positive wearing a label."""
    with pytest.raises(C.CalibrationError, match="still visible"):
        C._assert_mutation_landed(
            "probe:natural", _frame(), _frame(),
            _natural_assertions(apple_warm_px=C.NATURAL_PROBE_MAX_WARM_PX + 1))


def test_a_natural_probe_with_nothing_in_front_of_the_apple_is_refused():
    """An apple that is absent from the SCENE is a `mismatch`, not an abstention.

    This is the failure that separates "the hand occludes the apple" from "the fruit was carried
    off". Both leave the strict mask empty; only the first is what `unsure` means.
    """
    with pytest.raises(C.CalibrationError, match="nothing visible is hiding it"):
        C._assert_mutation_landed(
            "probe:natural", _frame(), _frame(),
            _natural_assertions(ring_foreground_fraction=0.1))


def test_a_natural_probe_nobody_looked_at_is_refused():
    with pytest.raises(C.CalibrationError, match="observation record is incomplete"):
        C._assert_mutation_landed("probe:natural", _frame(), _frame(),
                                  _natural_assertions(looked_at=False))


def test_a_natural_probe_whose_other_clauses_are_not_confirmed_is_refused():
    with pytest.raises(C.CalibrationError, match="observation record is incomplete"):
        C._assert_mutation_landed("probe:natural", _frame(), _frame(),
                                  _natural_assertions(other_clauses_true=False))


def test_the_natural_probe_thresholds_are_stricter_than_the_census_they_are_drawn_from():
    """The census records what came CLOSE, so the artifact shows the pool and not only the pick."""
    assert C.NATURAL_PROBE_MAX_WARM_PX < C.NATURAL_PROBE_CENSUS_PX
    assert 0.0 < C.NATURAL_PROBE_MIN_RING_FOREGROUND <= 1.0


def test_the_probe_picker_refuses_to_pad_a_pool_that_is_too_small():
    """docs §3.4's denominators are counts. Nine probes are scored against a rule nobody wrote."""
    frames = [{"apple_warm_px": i, "thumb": [float(i)]} for i in range(C.N_PROBE - 1)]
    with pytest.raises(C.CalibrationError, match="eligible probe frame"):
        C.farthest_point_pick(frames, C.N_PROBE)


def test_the_probe_picker_returns_distinct_frames_and_starts_from_the_least_visible_apple():
    pytest.importorskip("numpy")
    frames = [{"apple_warm_px": 1000 - i, "thumb": [float(i) * 10], "frame_index": i}
              for i in range(20)]
    picked = C.farthest_point_pick(frames, C.N_PROBE)
    assert len(picked) == C.N_PROBE
    assert len({p["frame_index"] for p in picked}) == C.N_PROBE
    # farthest-point sampling spends the spread the pool has: it must not return a run of
    # neighbours, which is what taking the ten smallest apples out of one occlusion would give.
    assert max(p["frame_index"] for p in picked) - min(p["frame_index"] for p in picked) >= 15


def _probe_obs_or_skip() -> dict:
    p = CAL2_DIR / "probe_observations.json"
    if not p.is_file():
        pytest.skip("no attempt-2 probe observations in runs/")
    return json.loads(p.read_text())


def test_every_natural_probe_records_that_somebody_looked_and_why_it_cannot_be_answered():
    """These labels are not facts about a transformation, so this record is all they have."""
    doc = _probe_obs_or_skip()
    assert len(doc["probes"]) == C.N_PROBE
    for p in doc["probes"]:
        assert p["looked_at"] is True
        assert p["undecidable_because"].strip()
        assert p["observed"].strip()
        assert p["other_clauses_true"] is True
        assert p["apple_warm_px"] <= C.NATURAL_PROBE_MAX_WARM_PX
        assert p["ring_foreground_fraction"] >= C.NATURAL_PROBE_MIN_RING_FOREGROUND


def test_the_probe_census_covers_the_corpus_and_excludes_every_measured_episode():
    """The census is the evidence that the ten were not eye-picked — and the record of how few
    frames in this corpus qualify at all."""
    p = CAL2_DIR / "probe_census.json"
    if not p.is_file():
        pytest.skip("no attempt-2 probe census in runs/")
    doc = json.loads(p.read_text())
    measured = set(doc["rule"]["measured_episodes_excluded"])
    assert len(measured) == 40
    assert doc["corpus"]["episodes_scanned"] == 402 - len(measured)
    assert doc["corpus"]["frames_eligible"] >= C.N_PROBE
    for r in doc["picked"]:
        assert r["episode"] not in measured


def test_a_natural_probe_may_not_come_from_a_measured_episode():
    doc = _probe_obs_or_skip()
    sheet = REPO / "runs" / "t040-identity-prompt" / "sheet.jsonl"
    measured = {json.loads(ln)["episode"] for ln in sheet.read_text().splitlines() if ln.strip()}
    for p in doc["probes"]:
        assert p["episode"] not in measured, "docs §3.1: no calibration frame is a measured frame"


def _key2_or_skip() -> dict:
    p = CAL2_DIR / "key.json"
    if not p.is_file():
        pytest.skip("no built attempt-2 C40 in runs/")
    return json.loads(p.read_text())


def test_attempt_twos_probes_are_all_natural_and_none_of_its_items_was_composited():
    items = _key2_or_skip()["items"]
    probes = [v for v in items.values() if v.get("class") == "probe"]
    assert len(probes) == C.N_PROBE
    assert {v["mutation"] for v in probes} == {"probe:natural"}
    assert {v["required_verdict"] for v in probes} == {"unsure"}


def test_attempt_twos_probe_frames_are_the_corpus_frames_byte_for_byte():
    """Verified against the images the census extracted, not against an intention in the meta."""
    doc = _probe_obs_or_skip()
    meta = CAL2_DIR / "build_meta.json"
    if not meta.is_file():
        pytest.skip("no built attempt-2 C40 in runs/")
    built = {i["episode"] + str(i["assertions"].get("frame_index")): i
             for i in json.loads(meta.read_text())["items"] if i["class"] == "probe"}
    assert len(built) == C.N_PROBE
    import hashlib
    for p in doc["probes"]:
        item = built[p["episode"] + str(p["frame_index"])]
        digest = hashlib.sha256(Path(item["built_frame"]).read_bytes()).hexdigest()
        assert digest == p["frame_sha256"], (
            "the item issued to the instrument is not the frame that was looked at")


def test_the_two_attempts_issued_the_same_thirty_decidable_items():
    """Attempt 2 changed the probes and NOTHING else — checked on the pixels, not on the plan."""
    import hashlib
    for d in (CAL_DIR, CAL2_DIR):
        if not (d / "build_meta.json").is_file():
            pytest.skip("both attempts must be built for this comparison")
    def decidable(d: Path) -> dict[str, str]:
        return {i["tag"]: hashlib.sha256(Path(i["built_frame"]).read_bytes()).hexdigest()
                for i in json.loads((d / "build_meta.json").read_text())["items"]
                if i["class"] != "probe"}
    a, b = decidable(CAL_DIR), decidable(CAL2_DIR)
    assert len(a) == 30
    assert a == b


def test_the_two_attempts_were_shuffled_under_different_seeds():
    """A re-run under the same seed inherits which of the four judges sees which item."""
    for d in (CAL_DIR, CAL2_DIR):
        if not (d / "key.json").is_file():
            pytest.skip("both attempts must be built for this comparison")
    a = json.loads((CAL_DIR / "key.json").read_text())["shuffle_seed"]
    b = json.loads((CAL2_DIR / "key.json").read_text())["shuffle_seed"]
    assert a != b


def test_attempt_twos_items_carry_nothing_but_a_frame_and_a_prompt():
    p = CAL2_DIR / "items.jsonl"
    if not p.is_file():
        pytest.skip("no built attempt-2 C40 in runs/")
    rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    assert len(rows) == 80
    for r in rows:
        assert set(r) == {"item_id", "frame", "prompt"}
        assert Path(r["frame"]).name == f"{r['item_id']}.png"


def test_every_natural_probe_carries_the_same_prompt_as_the_positives():
    """A probe differs from a positive in the PICTURE, and must not differ in the sentence.

    If the probes carried a different string, an instrument could separate the class without ever
    looking at a frame — and `unsure` would then be a fact about the text.
    """
    p, k = CAL2_DIR / "items.jsonl", CAL2_DIR / "key.json"
    if not (p.is_file() and k.is_file()):
        pytest.skip("no built attempt-2 C40 in runs/")
    key = json.loads(k.read_text())["items"]
    prompts = {json.loads(ln)["item_id"]: json.loads(ln)["prompt"]
               for ln in p.read_text().splitlines() if ln.strip()}
    probe = {prompts[i] for i, v in key.items() if v.get("class") == "probe"}
    positive = {prompts[i] for i, v in key.items() if v.get("class") == "positive"}
    assert len(probe) == 1 and probe == positive
