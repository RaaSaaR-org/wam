"""Tests for ``scripts/check_style_partition.py`` and the committed partition it guards.

The committed file passing is the least interesting assertion here, so it is one test out of
several. The rest inject each violation the checker exists to catch, because a checker that has
never rejected anything is indistinguishable from one that returns PASS unconditionally — and this
one runs exactly once in the life of the experiment, before generation, when there is nothing else
left to notice the problem.

The four properties under test are the four ways PR-08 §5's sentence can be true in prose and
false in the file:

1. a style in both sets (checked on the axis slugs, not only on the ids — different names for the
   same appearance is the failure a name-level check cannot see);
2. the identity control appearing in a set, which turns arm C's control into arm B's data;
3. a prompt that moves geometry, which desynchronises the carried-over action labels from the
   pixels (T-040) and which G0b would only catch after the generation is paid for;
4. a hash that does not identify the partition — either not reproducible, or unchanged by a change
   to a style.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import check_style_partition as csp  # noqa: E402

PARTITION = REPO / "configs" / "transfer25" / "styles.toml"


@pytest.fixture
def doc() -> dict:
    return csp.load(PARTITION)


def test_committed_partition_passes() -> None:
    assert csp.main([]) == 0


def test_committed_shape(doc: dict) -> None:
    """The counts PR-08 §9 prices the experiment at, pinned so a silent edit is a test failure."""
    assert len(doc["train_styles"]) == 10
    assert len(doc["eval_styles"]) == 5
    assert doc["identity_style"]["id"] == "identity-source"
    assert doc["rule"] == "T40_STYLES_V1"


def test_committed_arm_c_is_frame_matched(doc: dict) -> None:
    """PR-08 §5's "same frame count", pinned as a number rather than left to the checker.

    The committed V1 of this file shipped one identity style against ten train styles — arm C at
    402 clips against arm B's 4020 — so B > C was equally consistent with "diversity helped" and
    "more data helped", which is the single confound arm C exists to remove. The fix repeats the
    identity style ten times per episode under ten committed seeds. Arm B is not subsampled: that
    is what the `train_repeats == 1` assertion is here to hold.
    """
    vol = doc["volume"]
    assert vol["train_repeats"] == 1
    assert vol["identity_repeats"] == len(doc["train_styles"]) == 10
    assert doc["identity_style"]["repeats"] == 10
    assert vol["train_clips"] == vol["identity_clips"] == 4020
    assert len(set(doc["seed_schedule"]["blocks"]["identity"])) == 10


def test_frame_mismatch_rejected(doc: dict) -> None:
    """The regression that shipped once already: arm C a tenth of arm B."""
    doc["identity_style"]["repeats"] = 1
    doc["volume"]["identity_repeats"] = 1
    doc["volume"]["identity_clips"] = 402
    with pytest.raises(csp.Failure, match="ARM C IS NOT FRAME-MATCHED"):
        csp.check_volume(doc)


def test_committed_volume_matches_the_registered_v2_figures(doc: dict) -> None:
    """The numbers T40_RULE_V2 §3.1 fixes, pinned here so this config cannot drift from the rule.

    PR-08-V2 §3.1's table: arm B 4 020 (unchanged), arm C 4 020 (was 402), B + C ~8 040 "roughly
    double", EVAL_STYLES 5 x 402 = 2 010 (unchanged), whole partition 10 050 clips over 25
    style-instances — which §3.2 makes the basis the GPU-h ceiling is derived over.
    """
    vol = doc["volume"]
    assert (vol["train_clips"], vol["identity_clips"], vol["eval_clips"]) == (4020, 4020, 2010)
    assert vol["training_clips"] == 8040
    assert vol["whole_partition_clips"] == 10050
    assert vol["style_instances"] == 25
    assert vol["gpu_h_ceiling_scope"] == "whole-partition"
    assert vol["gpu_h_ceiling_basis_clips"] == 10050
    assert doc["rule_v2"] == "T40_RULE_V2"


def test_ceiling_basis_must_be_the_whole_partition(doc: dict) -> None:
    """A per-set allowance wearing the whole-partition name is the failure V2 §3.2 describes."""
    doc["volume"]["gpu_h_ceiling_basis_clips"] = doc["volume"]["train_clips"]
    with pytest.raises(csp.Failure, match="gpu_h_ceiling_basis_clips"):
        csp.check_volume(doc)


def test_whole_partition_total_rejected(doc: dict) -> None:
    doc["volume"]["whole_partition_clips"] = 8040  # the training total, mistaken for the basis
    with pytest.raises(csp.Failure, match="whole_partition_clips"):
        csp.check_volume(doc)


def test_volume_disagreeing_with_the_styles_rejected(doc: dict) -> None:
    """[volume] restates what `repeats` says, so the restatement is checked, never trusted."""
    doc["identity_style"]["repeats"] = 5
    with pytest.raises(csp.Failure, match="identity_repeats"):
        csp.check_volume(doc)


def test_volume_arithmetic_rejected(doc: dict) -> None:
    doc["volume"]["train_clips"] = 4021
    with pytest.raises(csp.Failure, match="train_clips"):
        csp.check_volume(doc)


@pytest.mark.parametrize("bad", [0, -1, None, True, "10", 1.0])
def test_repeats_must_be_a_positive_int(doc: dict, bad: object) -> None:
    """`repeats` is read by the sbatch's work-list expansion; a bool or a string is not a count."""
    doc["train_styles"][0]["repeats"] = bad
    with pytest.raises(csp.Failure, match="repeats"):
        csp.check_structure(doc)


def test_repeats_is_required_not_defaulted(doc: dict) -> None:
    """An absent field is a thing to infer, and inferring it is how arm C came out at 402."""
    del doc["train_styles"][0]["repeats"]
    with pytest.raises(csp.Failure, match="repeats"):
        csp.check_structure(doc)


# ------------------------------------------------------------------------------------------
# THE SEED SCHEDULE. The second defect that shipped in this file and the one that could have
# manufactured a p rather than merely blurred one.
#
# As first committed, every TRAIN and every EVAL style had `repeats = 1` under the rule "repeat r
# uses repeat_seeds[r]; a style with repeats = 1 uses repeat_seeds[0]". So arm B's 4 020 clips
# were ALL at seed 7001, the eval set's 2 010 clips were ALSO all at 7001, and arm C was the only
# arm spanning 7001-7010. Two things follow:
#
#   1. arm B varied the prompt at a fixed seed while arm C varied the seed at a fixed prompt, so
#      B − C was a difference between two effects rather than the isolation of one; and
#   2. arm B's TRAINING clips shared their initial latent with the EVAL clips the headline is
#      scored on and arm C's mostly did not, so any transferable seed-specific generator
#      fingerprint matched B to the scoring distribution more closely than C for a reason that is
#      not diversity.
#
# (2) is the confound arm C exists to remove, re-entering through the seed schedule. The repair
# costs zero extra clips and is only available before the first clip exists, which is why these
# are tests and not review notes. The two negative cases below must fail if the rule is reverted.
# ------------------------------------------------------------------------------------------

def test_committed_seed_schedule(doc: dict) -> None:
    """The schedule as committed: same block for the two arms, a disjoint one for the eval set."""
    sched = doc["seed_schedule"]
    assert sched["assignment"] == csp.SEED_ASSIGNMENT
    assert sched["rule"] == csp.SEED_RULE  # normative, pinned to the code
    assert sched["one_seed_per_style_instance"] is True
    blocks = sched["blocks"]
    assert blocks["train"] == list(range(7001, 7011))
    assert blocks["identity"] == list(range(7001, 7011))
    assert blocks["eval"] == list(range(7011, 7016))
    assert set(blocks["train"]) == set(blocks["identity"])
    assert not set(blocks["eval"]) & (set(blocks["train"]) | set(blocks["identity"]))


def test_the_seed_arithmetic_the_file_states(doc: dict) -> None:
    """10 styles x 1 repeat and 1 style x 10 repeats both span the ten seeds, 402 clips each.

    This is the claim the fix rests on and it is not self-evident: "style i uses blocks.train[i]"
    only gives arm B ten seeds because arm B has ten styles, and arm C reaches the same ten from
    the other direction (one style, ten repeats). Worked here rather than trusted.
    """
    per = csp.resolved_seeds(doc)
    train = [s for style_seeds in per["train"].values() for s in style_seeds]
    identity = [s for style_seeds in per["identity"].values() for s in style_seeds]
    ev = [s for style_seeds in per["eval"].values() for s in style_seeds]

    assert len(per["train"]) == 10 and all(len(v) == 1 for v in per["train"].values())
    assert len(per["identity"]) == 1 and all(len(v) == 10 for v in per["identity"].values())
    assert sorted(train) == sorted(identity) == list(range(7001, 7011))
    assert sorted(ev) == list(range(7011, 7016))
    # Not merely the same SET: the same seed used the same number of times in both arms, so the
    # match is seed-by-seed. 402 episodes x one instance per seed = 402 clips per seed per arm.
    eps = doc["volume"]["episodes"]
    assert len(train) == len(identity) == 10
    assert doc["volume"]["train_clips"] == doc["volume"]["identity_clips"] == eps * 10 == 4020
    assert doc["volume"]["eval_clips"] == eps * 5 == 2010


def test_train_and_identity_must_span_the_same_seed_set(doc: dict) -> None:
    """NEGATIVE CASE 1 — reverting to per-arm seeds must fail.

    The exact shape of the original defect: arm B on one seed, arm C on ten. If this stops
    raising, B − C is a difference of seeds as well as of prompts.
    """
    doc["seed_schedule"]["blocks"]["train"] = [7001] * 10
    with pytest.raises(csp.Failure, match="duplicate seeds"):
        csp.check_seed_schedule(doc)
    # …and with distinct-but-different seeds, which the duplicate check cannot see:
    doc["seed_schedule"]["blocks"]["train"] = list(range(7101, 7111))
    with pytest.raises(csp.Failure, match="ARM B AND ARM C DO NOT SPAN THE SAME SEED SET"):
        csp.check_seed_schedule(doc)


def test_eval_seeds_must_not_intersect_the_training_blocks(doc: dict) -> None:
    """NEGATIVE CASE 2 — the one that can manufacture a p.

    Sharing even one seed between the eval block and a training block gives whichever arm owns
    that training clip a fingerprint match to the scoring distribution. The original schedule
    shared ALL of them with arm B.
    """
    doc["seed_schedule"]["blocks"]["eval"] = [7001, 7012, 7013, 7014, 7015]
    with pytest.raises(csp.Failure, match="EVAL SEEDS INTERSECT A TRAINING SEED BLOCK"):
        csp.check_seed_schedule(doc)


def test_the_original_broken_schedule_is_rejected_end_to_end(doc: dict) -> None:
    """The file exactly as it was committed this morning, rebuilt and re-checked.

    One shared ten-seed list under "repeat r uses seeds[r]; repeats = 1 uses seeds[0]" resolves to
    seed[0] for every TRAIN and every EVAL style. Expressed in the current schema that is a train
    block of ten copies of 7001 and an eval block of five copies of 7001 — and it must not pass.
    """
    doc["seed_schedule"]["blocks"]["train"] = [7001] * 10
    doc["seed_schedule"]["blocks"]["eval"] = [7001] * 5
    with pytest.raises(csp.Failure):
        csp.check_seed_schedule(doc)


def test_one_seed_per_style_instance_is_enforced(doc: dict) -> None:
    """A block longer or shorter than its set is a schedule that is not the one that ran."""
    short = dict(doc["seed_schedule"]["blocks"])
    doc["seed_schedule"]["blocks"] = {**short, "train": short["train"][:3]}
    with pytest.raises(csp.Failure, match="style-instances per episode"):
        csp.check_seed_schedule(doc)
    doc["seed_schedule"]["blocks"] = {**short, "eval": short["eval"] + [7016]}
    with pytest.raises(csp.Failure, match="style-instances per episode"):
        csp.check_seed_schedule(doc)


def test_duplicate_seeds_rejected(doc: dict) -> None:
    """Two style-instances under one seed are one sample of the generator's variance, not two."""
    doc["seed_schedule"]["blocks"]["eval"][1] = doc["seed_schedule"]["blocks"]["eval"][0]
    with pytest.raises(csp.Failure, match="duplicate seeds"):
        csp.check_seed_schedule(doc)


def test_seed_rule_string_is_pinned_to_the_code(doc: dict) -> None:
    """The rule is normative text, and the last one lived only in a comment — which is the defect.

    `assignment` is what the checker and the sbatch dispatch on; a partition naming an assignment
    nothing implements would be generated under whatever the consumer happens to do instead.
    """
    doc["seed_schedule"]["assignment"] = "repeat-index"
    with pytest.raises(csp.Failure, match="assignment"):
        csp.check_seed_schedule(doc)


def test_missing_seed_schedule_rejected(doc: dict) -> None:
    del doc["seed_schedule"]
    with pytest.raises(csp.Failure, match="no \\[seed_schedule\\]"):
        csp.check_seed_schedule(doc)


def test_seed_schedule_is_inside_the_partition_hash(doc: dict) -> None:
    """The schedule, not only the seed values — the values were already committed and were not
    what went wrong. A change to who gets which seed must move the digest PR-08 §6 records."""
    before = csp.content_hash(doc)
    doc["seed_schedule"]["blocks"]["eval"] = [7001, 7002, 7003, 7004, 7005]
    assert csp.content_hash(doc) != before


def test_rendering_carries_resolved_seeds_per_style(doc: dict) -> None:
    """The consumer reads a resolved list, never an index rule of its own.

    97_transfer25_restyle.sbatch does `seeds_of(s)[r]`. Re-deriving the rule in the consumer is
    how "repeats = 1 uses seeds[0]" reached the work list in the first place, so the rendering
    ships the answer: seeds[r] is the seed for repeat r, one entry per repeat.
    """
    styles = json.loads(csp.DEFAULT_DERIVED.read_text())
    for half in ("train", "eval", "identity"):
        for s in styles[half]:
            assert len(s["seeds"]) == s["repeats"]
            assert len(set(s["seeds"])) == len(s["seeds"])
    assert [s["seeds"][0] for s in styles["train"]] == list(range(7001, 7011))
    assert styles["identity"][0]["seeds"] == list(range(7001, 7011))
    assert [s["seeds"][0] for s in styles["eval"]] == list(range(7011, 7016))
    train_seeds = {x for s in styles["train"] for x in s["seeds"]}
    identity_seeds = {x for s in styles["identity"] for x in s["seeds"]}
    eval_seeds = {x for s in styles["eval"] for x in s["seeds"]}
    assert train_seeds == identity_seeds
    assert not eval_seeds & (train_seeds | identity_seeds)
    assert styles["seed_schedule"]["assignment"] == csp.SEED_ASSIGNMENT


def test_hash_rule_string_reproduces_the_committed_digest(doc: dict) -> None:
    """FIX for the rule string that named one exclusion while the code had two.

    ``[hash].rule`` is normative: it is what a reader recomputing PR-08 §6's partition hash
    follows. It used to read "document minus [hash]" while ``HASH_EXCLUDES`` has always been
    ``("hash", "consumer")``, so anyone following it got a digest that could not match the
    committed one — and the natural reading of that mismatch, "the partition changed", is the
    exact alarm the value exists to raise. This pins the string to the code.
    """
    assert doc["hash"]["excludes"] == list(csp.HASH_EXCLUDES)
    for excluded in csp.HASH_EXCLUDES:
        assert f"[{excluded}]" in doc["hash"]["rule"]

    # Follow the rule string literally and land on the committed sidecar.
    body = {k: v for k, v in doc.items() if k not in ("hash", "consumer")}
    canonical = json.dumps(body, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    assert digest == (PARTITION.parent / (PARTITION.name + ".sha256")).read_text().strip()


def test_open_blocking_todos_are_reported_but_do_not_fail(doc: dict) -> None:
    """PR-08 §1 licenses committing this file and forbids generating from it.

    So an open TODO must not turn the check red: a red check on a file that is correct as
    committed only teaches people to stop running the check. It must still be visible.

    CHANGED 2026-08-22, and the change matters. This used to read the property off the real
    document, which worked only for as long as the real document happened to have an open item —
    and the moment T40-TODO-01 closed, the test failed while the behaviour it guards was still
    correct. A test that goes red because the project made progress is a test that will be deleted
    rather than read. The property is now exercised against a synthetic OPEN item, so it holds
    whatever the committed file happens to contain, and the committed file's own state is asserted
    separately below.
    """
    injected = dict(doc)
    injected["blocking_todos"] = list(doc["blocking_todos"]) + [{
        "id": "T40-TODO-99-synthetic",
        "status": "OPEN",
        "blocks": "STYLE_SET=synthetic",
    }]
    lines = csp.check_blocking_todos(injected)
    assert any("T40-TODO-99-synthetic" in line for line in lines)
    assert any("block GENERATION, not this check" in line for line in lines)
    assert csp.main([]) == 0  # reported, not enforced


def test_the_committed_document_has_no_open_blocking_todos(doc: dict) -> None:
    """The committed state, asserted on purpose rather than as a side effect of the test above.

    Every blocking todo blocks a STYLE_SET. This going red means something re-opened an item and
    a generation submission for that style set will refuse — which is the intended behaviour, but
    it should be a deliberate edit and not a surprise.
    """
    open_ = [t["id"] for t in doc["blocking_todos"] if t["status"] == "OPEN"]
    assert open_ == [], f"open blocking todos: {open_}"
    lines = csp.check_blocking_todos(doc)
    assert any("0 open" in line for line in lines)


def test_identity_prompt_provenance_is_a_record_not_prose(doc: dict) -> None:
    """FIX: the one-caption identity prompt was a comment; it is now a blocking record.

    ``[identity_style].prompt`` is assembled from a T-041 machine caption of a single clip and
    applied to all 402 episodes. If it is wrong for most of them, arm C is a weak restyle rather
    than an identity pass, and the frame match does not save it — a frame-matched control with the
    wrong prompt is still the wrong control.
    """
    todos = {t["id"]: t for t in doc["blocking_todos"]}
    todo = todos["T40-TODO-01-identity-prompt-provenance"]
    assert "identity" in todo["blocks"]
    assert "episode_000135_clip000" in doc["source"]["caption_provenance"]

    # CLOSED 2026-08-22. A record does not stop being a record when it closes — it stops being one
    # if it closes without the evidence it named, which is exactly how a blocking item becomes the
    # comment it was promoted out of being. So closure is checked against the item's OWN
    # evidence_required field rather than against a list repeated here.
    assert todo["status"] == "CLOSED"
    assert todo["evidence_required"] == (
        "the sampled episode ids, the sample size, and the per-episode verdicts"
    )
    assert len(todo["evidence_sampled_episodes"]) == todo["evidence_sample_size"]
    assert len(todo["evidence_verdicts"]) == todo["evidence_sample_size"]
    assert set(todo["evidence_verdict_counts"]) == {"match", "mismatch", "unsure"}
    assert sum(todo["evidence_verdict_counts"].values()) == todo["evidence_sample_size"]
    assert todo["evidence_gate_qualified"] is True
    assert todo["evidence_coverage"] == pytest.approx(1.0)

    # The verdict list and the counts are two spellings of one measurement, so they are compared
    # rather than both trusted.
    tallied: dict[str, int] = {"match": 0, "mismatch": 0, "unsure": 0}
    for row in todo["evidence_verdicts"]:
        episode, _, verdict = row.partition(" = ")
        assert episode in todo["evidence_sampled_episodes"]
        tallied[verdict.strip()] += 1
    assert tallied == todo["evidence_verdict_counts"]

    # The corpus disagreed with the caption about the caption's OWN clip, so the prompt is no
    # longer the concatenation of [identity_style.source_caption]. The quotes stay as provenance;
    # this pins the disagreement so a later edit cannot quietly re-align them and lose the finding.
    assert "dark grey cloth" in doc["identity_style"]["prompt"]
    assert "black" not in doc["identity_style"]["prompt"]
    assert "black" in doc["identity_style"]["source_caption"]["background_setting"]


def test_id_overlap_rejected(doc: dict) -> None:
    doc["eval_styles"][0]["id"] = doc["train_styles"][0]["id"]
    with pytest.raises(csp.Failure, match="duplicate style ids"):
        csp.check_structure(doc)


def test_slug_overlap_rejected_even_with_distinct_ids(doc: dict) -> None:
    """The one a name-level check misses: two ids, one appearance.

    'walnut-veneer' in TRAIN and the same slug in EVAL means the headline reports generalisation
    to a shift that did not happen. check_structure is happy — the ids differ — so this has to be
    caught on the axis.
    """
    doc["eval_styles"][0]["table"] = doc["train_styles"][6]["table"]
    csp.check_structure(doc)  # ids still unique; the id-level check cannot see it
    with pytest.raises(csp.Failure, match="share table slugs"):
        csp.check_disjoint(doc)


def test_identity_in_train_rejected(doc: dict) -> None:
    doc["train_styles"].append(copy.deepcopy(doc["identity_style"]))
    with pytest.raises(csp.Failure):
        csp.check_structure(doc)  # duplicate id
    doc["train_styles"][-1]["id"] = "train-11-smuggled-identity"
    with pytest.raises(csp.Failure, match="share apple slugs"):
        csp.check_disjoint(doc)


@pytest.mark.parametrize(
    "sentence,rule",
    [
        ("The apple is moved slightly to the right.", "position"),
        ("The plate is repositioned nearer the arm.", "position"),
        ("The table is raised to a greater height.", "height"),
        ("An extra apple sits alongside.", "object count"),
        ("A larger apple.", "height / scale"),
        ("The robot holds a different pose.", "robot pose"),
    ],
)
def test_geometry_terms_rejected(doc: dict, sentence: str, rule: str) -> None:
    doc["train_styles"][0]["prompt"] += " " + sentence
    with pytest.raises(csp.Failure, match="geometry-moving term"):
        csp.check_geometry_terms(doc)


def test_geometry_lint_scans_slugs_not_only_prompts(doc: dict) -> None:
    """A slug is style text too — 'apple-moved-left' would be a violation wearing a short name."""
    doc["eval_styles"][0]["background"] = "wall-shifted-back"
    with pytest.raises(csp.Failure, match="geometry-moving term"):
        csp.check_geometry_terms(doc)


# ------------------------------------------------------------------------------------------
# The demonstrated false negatives. These five sentences were appended to a committed TRAIN
# style's prompt by a reviewer and PASSED the lint as it was first written — five prompts that
# each move the apple, the camera or the table, and the only automated check between a prompt
# and a paid-for generation run said nothing. Every one of them is here, verbatim, with the word
# the lint is expected to catch it on, so that reverting the vocabulary fails the suite loudly
# rather than quietly reopening the hole.
#
# The second element is asserted against the REPORT, not just against "something raised": the
# report names the matched word, and a test that only asserted "raises" would still pass if the
# sentence were caught by some unrelated term for an unrelated reason.
# ------------------------------------------------------------------------------------------
REVIEWER_BYPASSES = [
    ("The apple sits to the left of the plate.", "left"),
    ("The apple rests behind the plate.", "behind"),
    ("The apple is placed on top of the plate.", "on top of"),
    ("The apple is nudged toward the plate.", "toward"),
    ("The camera is panned and the table is tall.", "panned"),
    ("The camera is panned and the table is tall.", "tall"),
]


@pytest.mark.parametrize("sentence,expected", REVIEWER_BYPASSES)
def test_reviewer_bypass_sentences_are_rejected(doc: dict, sentence: str, expected: str) -> None:
    doc["train_styles"][0]["prompt"] += " " + sentence
    with pytest.raises(csp.Failure, match="geometry-moving term") as exc:
        csp.check_geometry_terms(doc)
    assert expected.lower() in str(exc.value).lower(), (
        f"the lint rejected {sentence!r} but not on {expected!r}: {exc.value}"
    )


@pytest.mark.parametrize("sentence,expected", REVIEWER_BYPASSES)
def test_reviewer_bypass_sentences_are_rejected_in_a_slug_too(
    doc: dict, sentence: str, expected: str
) -> None:
    """Same five, hidden where they are short: a slug is style text and is scanned as such."""
    doc["eval_styles"][0]["background"] = sentence.lower().strip(".").replace(" ", "-")
    with pytest.raises(csp.Failure, match="geometry-moving term"):
        csp.check_geometry_terms(doc)


# The vocabulary the reviewer's five sentences showed to be missing, each word on its own so a
# partial revert is a specific failure rather than one opaque one. "The apple is X the plate" is
# not always grammatical ("the apple is edge the plate"); grammar is irrelevant to a term lint,
# and the generator would not be asked politely either.
REQUIRED_VOCABULARY = [
    "left", "right", "behind", "front", "above", "below", "under", "over", "beneath", "beside",
    "next to", "toward", "towards", "edge", "corner", "centre", "center", "pan", "tall", "nudge",
    "tilt", "angle", "closer", "farther", "further", "on top of", "underneath", "alongside",
    "adjacent", "opposite", "middle", "atop", "near",
]


@pytest.mark.parametrize("word", REQUIRED_VOCABULARY)
def test_required_geometry_vocabulary_is_covered(doc: dict, word: str) -> None:
    doc["train_styles"][0]["prompt"] += f" The apple is {word} the plate."
    with pytest.raises(csp.Failure, match="geometry-moving term"):
        csp.check_geometry_terms(doc)


@pytest.mark.parametrize(
    "word",
    # Inflections the V1 suffix list could not reach: a dropped -e ("move" -> "moving") and a
    # doubled final consonant ("pan" -> "panned"). "moving" passing while "moved" failed is not a
    # rule anyone could have followed, and "panned" is one of the five.
    ["moving", "sliding", "nudging", "nudged", "panned", "panning", "gripping", "placing",
     "sliding", "tallest", "nearest", "rotating"],
)
def test_inflected_forms_are_reached(doc: dict, word: str) -> None:
    doc["train_styles"][0]["prompt"] += f" The apple is {word}."
    with pytest.raises(csp.Failure, match="geometry-moving term"):
        csp.check_geometry_terms(doc)


# ------------------------------------------------------------------------------------------
# The other half of a blunt lint: what it must NOT flag. Every string here is legitimate text
# from the committed partition or from T-040's ALLOWED side (appearance, lighting), and each one
# is a word that was considered for the vocabulary and rejected because of this collision. A
# future widening that breaks one of these fails here instead of on the reviewer's desk.
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "clause",
    [
        # The invariance clause, VERBATIM AND WHOLE. It is two sentences, not one: quoting only
        # "Scene geometry, camera framing…" appends a fragment that _scannable cannot match, so the
        # unstripped 'geometry'/'camera' legitimately trip the lint — and check_structure would
        # reject such a prompt anyway. The half-quote was this test's bug, not the lint's.
        "The white plate keeps its own appearance."
        " Scene geometry, camera framing and the robot are unchanged.",
        "against a plain white plaster wall",                           # 'against', in all 16
        "lit by warm tungsten light from one side with soft shadows",   # 'one', 'side'
        "lit by a single dim desk lamp",                                # 'single'
        "amber evening light casting long soft shadows",                # 'long'
        "against a deep teal wall",                                     # 'deep'
        # WAS "a dark crimson apple with a mirror gloss". 'mirror' is now a forbidden stem: it is
        # what catches "The scene is mirrored horizontally.", a probe that previously got through.
        # A mirror gloss is a surface finish on T-040's ALLOWED side, so the committed prompt was
        # reworded to "polished" (same finish) rather than the stem being dropped — styles.toml
        # §615-621 records that trade. This case tracks the reworded text.
        "a dark crimson apple with a polished gloss",                   # the reworded finish
        "lit by bright fluorescent tubes overhead",                     # 'overhead', 'bright'
        "even, bright, top-down lighting with minimal shadows",         # 'top-down', 'top'
        "on a white melamine table top",                                # 'table top'
        "lit by overexposed flat white light",                          # 'over'
        "cool overcast daylight that is evenly diffused",               # 'over'
        "against varnished wood panelling",                             # 'pan'
        "a black cloth covering a flat surface",                        # 'cover' vs 'over'
    ],
)
def test_legitimate_appearance_text_is_not_flagged(doc: dict, clause: str) -> None:
    doc["train_styles"][0]["prompt"] += " " + clause
    csp.check_geometry_terms(doc)


def test_contrast_between_is_the_one_recorded_gap(doc: dict) -> None:
    """'between' is deliberately absent from the vocabulary, and the reason is arm C.

    [identity_style].prompt quotes the T-041 caption verbatim — "Contrast between the black
    background and the white plate…" — and PR-08 §5 makes arm C's prompt *the source's own
    appearance*. Rewording the control's text to satisfy a lint would turn the fingerprint
    control into a weak restyle, so the term was dropped instead and the positional senses are
    covered by 'in between' / 'midway' / 'halfway' / 'middle'. This test pins the trade so it is
    a decision on the record rather than an oversight, and fails if the identity prompt is ever
    reworded (at which point 'between' can go back in).
    """
    assert "Contrast between" in doc["identity_style"]["prompt"]
    csp.check_geometry_terms(doc)  # must not raise, on the committed document
    doc["train_styles"][0]["prompt"] += " The apple is in between the plate and the arm."
    with pytest.raises(csp.Failure, match="geometry-moving term"):
        csp.check_geometry_terms(doc)


def test_missing_invariance_clause_rejected(doc: dict) -> None:
    s = doc["train_styles"][0]
    s["prompt"] = s["prompt"].replace(doc["invariance_clause"], "").strip()
    with pytest.raises(csp.Failure, match="invariance clause"):
        csp.check_structure(doc)


def test_hash_matches_the_committed_sidecar(doc: dict) -> None:
    sidecar = PARTITION.parent / (PARTITION.name + ".sha256")
    assert sidecar.read_text().strip() == csp.content_hash(doc)


def test_hash_ignores_comments_and_key_order(tmp_path: Path, doc: dict) -> None:
    """Rewording a comment must not invalidate a finished run's recorded hash; nothing else may
    be free. The hash is over the parsed content, so this is a property of canonical_json."""
    commented = tmp_path / "styles.toml"
    commented.write_text("# an added comment, no semantic change\n" + PARTITION.read_text())
    assert csp.content_hash(csp.load(commented)) == csp.content_hash(doc)
    assert csp.content_hash(dict(reversed(list(doc.items())))) == csp.content_hash(doc)


def test_hash_changes_when_any_style_changes(doc: dict) -> None:
    before = csp.content_hash(doc)
    for mutate in (
        lambda d: d["train_styles"][0].__setitem__("apple", "green-granny-2"),
        lambda d: d["eval_styles"][0].__setitem__("prompt", "A different prompt."),
        lambda d: d["identity_style"].__setitem__("id", "identity-other"),
        lambda d: d["train_styles"].pop(),
        lambda d: d.__setitem__("rule", "T40_STYLES_V2"),
    ):
        mutated = copy.deepcopy(doc)
        mutate(mutated)
        assert csp.content_hash(mutated) != before


def test_derived_rendering_matches_the_source(doc: dict) -> None:
    assert csp.DEFAULT_DERIVED.read_text() == csp.emit_json(doc)


def test_derived_rendering_satisfies_the_generators_own_guard(doc: dict) -> None:
    """The shape 97_transfer25_restyle.sbatch actually indexes, and its own overlap check.

    Reproduced here rather than trusted: the sbatch does ``json.loads`` then ``styles["train"]``
    and ``{s["id"] for s in ...}``, and exits FATAL if the halves intersect. If the rendering ever
    stops satisfying that, the failure would otherwise appear on the cluster.
    """
    styles = json.loads(csp.DEFAULT_DERIVED.read_text())
    for half in ("train", "eval", "identity"):
        assert isinstance(styles[half], list) and styles[half]
        for s in styles[half]:
            assert s["id"] and s["prompt"]
    assert not {s["id"] for s in styles["train"]} & {s["id"] for s in styles["eval"]}
    assert styles["source_content_sha256"] == csp.content_hash(doc)


def test_derived_drift_is_a_failure(tmp_path: Path, doc: dict) -> None:
    drifted = tmp_path / "pr08_style_partition.json"
    payload = json.loads(csp.emit_json(doc))
    payload["eval"].append(payload["train"][0])  # the exact edit the source forbids
    drifted.write_text(json.dumps(payload))
    with pytest.raises(csp.Failure, match="drifted"):
        csp.check_derived(doc, drifted, write=False)


def test_hash_excludes_its_own_description_and_the_wiring(doc: dict) -> None:
    """[hash] describes how the hash is taken, so it cannot be inside it; [consumer] is a path and
    two key names, and repointing it must not orphan a finished run's recorded value."""
    before = csp.content_hash(doc)
    doc["hash"]["sidecar"] = "somewhere/else.sha256"
    doc["consumer"]["rendering"] = "configs/cosmos3/pr08_style_partition.json"
    assert csp.content_hash(doc) == before
