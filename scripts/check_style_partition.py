#!/usr/bin/env python3
"""Verify the committed TRAIN_STYLES / EVAL_STYLES partition (PR-08 §5, §8 item 6).

WHAT THIS GUARDS, and why prose in the pre-registration is not enough. PR-08 §6's headline is arm
B against arm A on ``EVAL_STYLES``, "disjoint from ``TRAIN_STYLES`` by the committed partition".
Three ways that sentence can be true on paper and false in the file, each of which would be
invisible in a finished run:

1. **Overlap.** A style in both sets makes the "held-out" eval domain partly trained-on, and the
   headline number then contains an in-domain component nobody can subtract out afterwards.
   Checked on ids *and* on the four axis slugs, because different ids can name the same
   appearance and the appearance is what the policy sees.
2. **A moved object.** T-040: "anything that moves an object desynchronizes pixels from labels and
   the arm grasps empty air." The actions are the recorded teleop trajectory, carried over
   unchanged, so a prompt that repositions the apple produces a training frame whose label is
   simply wrong. G0b (PR-08 §6) catches it after generation, at the cost of the generation; this
   catches the *prompt* before anything is submitted. It is a lint over text and is deliberately
   blunt — see FORBIDDEN_TERMS.
3. **A drifting hash.** PR-08 §6 records "the style partition hash" with the verdict. If the
   recorded value cannot be recomputed from the committed file, the record identifies nothing.
4. **An unmatched arm C.** PR-08 §5 gives arm C the *same frame count* as arm B — that is the
   whole mechanism by which the control separates "diversity helped" from "more data helped". One
   identity style against ten train styles is a 10x volume gap wearing the word "control", so the
   partition commits a per-style ``repeats`` and a ``[volume]`` block and this script fails if the
   arithmetic in it stops holding.
5. **A seed schedule that re-creates the confound arm C removes.** Same frame count is not the
   only thing arm C has to hold fixed. If arm B's clips are generated under different seeds than
   arm C's, then B − C is a difference of seeds as well as of prompts; and if the EVAL clips share
   seeds with one arm's training clips and not the other's, then any transferable seed-specific
   generator fingerprint matches that arm to the scoring distribution for a reason that is not
   diversity — which can manufacture a p on the headline. Both were live in the file as first
   committed: every TRAIN and every EVAL style had ``repeats = 1`` under the rule "repeats = 1
   uses repeat_seeds[0]", so all 4 020 arm-B clips and all 2 010 eval clips were seed 7001 while
   arm C alone spanned ten. ``check_seed_schedule`` fails if the train and identity blocks stop
   spanning the same seed set, and fails if the eval block intersects either of them.

Exits non-zero on any failure and names it. ``--write-hash`` writes the sidecar; nothing else does.

    .venv/bin/python scripts/check_style_partition.py
    .venv/bin/python scripts/check_style_partition.py --write-hash    # after an intended change
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import tomllib

REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PARTITION = REPO / "configs" / "transfer25" / "styles.toml"
# The consumer shape. cluster/discoverer/97_transfer25_restyle.sbatch reads the partition with
# json.loads and indexes styles["train"] / styles["eval"] / styles["identity"], then byte-hashes
# the file with sha256sum. styles.toml is the *source of truth* — it is the thing with the
# argument for each value next to the value, in the genre configs/cosmos3/t041_eval_selection.toml
# established — and this is its rendering, generated, never hand-edited. Committing both is only
# safe because the check below refuses to pass when they disagree.
DEFAULT_DERIVED = REPO / "configs" / "transfer25" / "pr08_style_partition.json"

AXES = ("apple", "table", "background", "lighting")

# The three sets, as (key in [seed_schedule.blocks] / [volume] prefix / the rendering, report name).
SETS = (("train", "TRAIN_STYLES"), ("eval", "EVAL_STYLES"), ("identity", "IDENTITY"))

# The committed seed-assignment rule, pinned to the code the way [hash].rule is. There is exactly
# one implemented assignment and the file has to name it: a schedule that lives only in a comment
# is what put every arm-B training clip and every eval clip on one shared seed.
SEED_ASSIGNMENT = "style-instance-index"
SEED_RULE = (
    "seed = blocks[<set>][style_index * repeats + repeat_index]; "
    "style_index and repeat_index are 0-based, styles in committed order"
)

# The geometry lint's vocabulary. Built from T-040's own wording — "Allowed: apple texture/colour/
# variety, table material, background, lighting. Forbidden: apple or plate *position*, table
# height, robot pose, object count" — expanded into the words a prompt would actually use for each
# forbidden thing, grouped so a hit reports *which* rule it broke.
#
# IT IS BLUNT ON PURPOSE. It cannot tell "a larger apple" (forbidden: scale is geometry) from "a
# larger wall" (harmless), so it rejects both and the style is reworded. The asymmetry is the right
# one: a false positive costs one edit to a file nobody has generated from yet, a false negative
# costs a corpus whose labels describe a different scene than its pixels. If a term here is truly
# wrong, delete it in a V2 with the reason — do not add an inline exemption, because an exemption
# is exactly the shape a real violation would take.
#
# ------------------------------------------------------------------------------------------
# VOCABULARY EXTENSION 2026-08-15 — the lint had a demonstrated FALSE NEGATIVE, and it is the
# only automated thing between a prompt and a paid-for generation run.
#
# All five of these were appended to a committed TRAIN style's prompt and passed the lint clean:
#
#     "The apple sits to the left of the plate."       moves the apple. no verb of motion in it.
#     "The apple rests behind the plate."              moves the apple.
#     "The apple is placed on top of the plate."       moves the apple, and ends the episode.
#     "The apple is nudged toward the plate."          moves the apple.
#     "The camera is panned and the table is tall."    moves the camera and the table height.
#
# The hole was structural, not a few missing words: every group named an ACTION ("move", "shift",
# "raise") and none named a PLACE. A prompt does not have to say it moved anything — it states
# where the thing is, and the generator puts it there. So there is now a spatial-relation group,
# and the words a static placement uses (left/behind/on top of/toward/…) are in it.
#
# Two more things the five sentences exposed:
#   - "panned" and "nudging" were unreachable from the stems "pan"/"nudge": the inflection suffix
#     handled neither a doubled consonant nor a dropped -e, so even "moving" was passing. That is
#     fixed in _term_pattern below, and it is a bigger repair than the word list.
#   - "tall" was absent while "taller" was present, i.e. the comparative was forbidden and the
#     plain adjective was not. Comparatives are now generated from the plain stem.
#
# ------------------------------------------------------------------------------------------
# SECOND VOCABULARY EXTENSION, SAME DAY, AND IT IS THE SAME BUG AS THE LAST LINE ABOVE — the
# claim "comparatives are now generated from the plain stem" WAS NOT TRUE OF THE LIST.
#
# An independent review put 59 probe sentences through this lint and 48 of them passed; six of
# those were then appended to real committed TRAIN styles and passed end to end, one per forbidden
# category:
#
#     "A giant apple on a tiny plate."                  scale
#     "A dozen apples scattered across the table."      count
#     "The plate is gone and the apple is absent."      count (by deletion)
#     "A low table, waist-high."                        table height
#     "The apple sits between the plate and the arm."   position, stated not performed
#     "Shot from overhead, a wide close-up."            framing
#
# ROOT CAUSE, and it is one line long: only "tall" was ever converted to a plain stem, because
# "tall" was the word the previous demonstration used. The other nine adjective pairs were left
# as the COMPARATIVE ONLY — bigger/big, larger/large, smaller/small, higher/high, lower/low,
# shorter/short, closer/close, farther/far, fewer/few — so the lint forbade "a bigger apple" and
# allowed "a big apple". The fix is not nine more words. It is that FORBIDDEN_TERMS now holds
# PLAIN STEMS ONLY and _term_pattern derives -er/-est/-s/-ed/-ing from them (including the
# consonant-doubling, dropped-e and y->i spellings English uses first), so adding one stem covers
# its whole paradigm and the sentence above is finally true of the code. tests/
# test_style_partition.py::test_comparatives_are_derived_and_not_listed pins it.
#
# THE INVARIANCE CLAUSE IS NOW REMOVED BEFORE THE SCAN, which is what makes the framing category
# closable at all. Every prompt ends with the committed clause ("Scene geometry, camera framing
# and the robot are unchanged."), and check_structure already requires it there verbatim; that is
# why "camera", "framing" and "robot" were previously recorded as unusable and why "Shot from
# overhead, a wide close-up." had nothing to hit. Deleting the clause — the exact committed
# string, nothing else — before matching leaves the style's OWN words, and "camera", "frame"
# (-> framing), "robot" and "geometry" become ordinary forbidden terms. This is not an exemption:
# it removes a constant this file itself pins, not a pattern a style author controls.
#
# THREE WORDS ARE DELIBERATELY NOT HERE, each because it collides with committed text that must
# not be reworded. Recorded as gaps rather than left to be rediscovered:
#
#   "between"  — [identity_style].prompt carries "Contrast between the black background and the
#                white plate…", quoted VERBATIM from the T-041 caption named in
#                [source].caption_provenance. Arm C's prompt *is* the source's own appearance
#                (PR-08 §5); rewording the control's text to satisfy a lint would make it a weak
#                restyle, which is the one thing arm C must not be. "in between", "midway",
#                "halfway" and "middle" are here instead. The demonstrated bypass that used it
#                ("The apple sits between the plate and the arm.") is now caught twice over by
#                "sit" and by "arm", i.e. by the words that make it a placement, so the gap is
#                narrower than it looks. If the identity prompt ever stops being a verbatim
#                quote, add it.
#   "long"     — "casting long soft shadows" (train-07) is a lighting description, and shadow
#                length is on T-040's ALLOWED side. "short" has no such collision and IS here.
#   "without"  — the same verbatim caption carries "the lighting highlights the objects without
#                creating harsh contrasts". "without a" / "without the" / "without any" are here
#                instead: the object-absence sense takes a noun phrase, the manner sense in the
#                caption takes a gerund, and that is a narrower TERM rather than an exemption on
#                a wider one.
#
# ONE COMMITTED STYLE WAS REWORDED to let a stem in, and it is recorded at the style itself:
# train-04's "low key" became "moody" (and its lighting slug "low-key-desk-lamp" became
# "moody-dim-desk-lamp") so that "low" — the plain stem of the already-forbidden "lower" — could
# be forbidden. The alternative was to keep "lower" as a comparative-only entry, which is the
# exact defect this extension exists to remove. The light it describes is unchanged.
#
# Every term here was checked against all 108 committed style strings before being added;
# "side", "one", "single", "deep", "mirror", "overhead" and "top-down" are still rejected for
# colliding with legitimate lighting text.
# ------------------------------------------------------------------------------------------
# EVERY ENTRY BELOW IS A PLAIN STEM. No "-er", no "-est", no "-ing", no plural: _term_pattern
# derives those, so "big" forbids bigger/biggest, "close" forbids closer/closest/closing/close-up,
# and "grip" forbids gripped/gripping/gripper. Adding an inflected form here is not wrong so much
# as a sign the paradigm is being maintained by hand again, which is how "lower" ended up
# forbidden while "low" was allowed. (Words that merely END in -er — "another", "corner",
# "further" — are stems in their own right and are fine.)
FORBIDDEN_TERMS: dict[str, tuple[str, ...]] = {
    "position (apple or plate)": (
        "position", "reposition", "relocate", "move", "shift", "slide", "displace", "offset",
        "translate", "rotate", "turn", "tilt", "rearrange", "layout", "swap", "beside",
        "geometry",
        # A placement stated rather than performed — "the apple is placed on the plate" — is the
        # same desynchronised label as "the apple is moved".
        "place", "placement", "put", "arrange", "align", "nudge",
        # …and a placement stated with no verb of placing at all. "The apple SITS between the
        # plate and the arm" / "the apple RESTS behind the plate": these are the verbs a caption
        # reaches for when it describes where a thing already is, which is exactly the instruction
        # a text-conditioned generator obeys.
        "sit", "rest", "lie", "lay", "stand", "perch", "balance", "hang", "dangle", "hover",
        "float", "occupy",
        # Verbs that rewrite the scene before the recorded trajectory starts, or end it early.
        "stack", "pile", "heap", "scatter", "strew", "spread", "push", "pull", "drag", "drop",
        "topple", "tip", "knock", "roll", "toss", "throw", "flip", "insert", "hide", "obscure",
        "occlude", "overlap", "touch", "hold", "prop",
        # The scene as a whole put through a transform. "The scene is mirrored horizontally"
        # swaps left for right in the pixels while the recorded joint trajectory still reaches
        # right — the T-040 failure exactly, and it names no object at all.
        "mirror", "invert", "flop", "upside", "sideways", "horizontal", "vertical",
        "squash", "squeeze", "stretch", "compress", "deform", "warp", "skew",
        # The object's own shape. A halved or sliced apple is not the apple the gripper closed on.
        "cut", "slice", "split", "half",
    ),
    # WHERE a thing is. The group V1 did not have at all: none of these words implies motion, and
    # every one of them relocates an object when a text-conditioned generator reads it.
    "spatial relation / direction": (
        "left", "right", "above", "below", "under", "underneath", "over", "beneath", "behind",
        "front", "next to", "on top of", "atop", "toward", "adjacent", "alongside", "opposite",
        "near", "nearby", "edge", "corner", "centre", "center", "central", "middle", "midway",
        "halfway", "in between", "forward", "backward",
        "across", "along", "beyond", "past", "amid", "among", "amongst", "inside", "outside",
        "within", "into", "onto", "away", "aside", "diagonal", "row", "surround", "border",
        "flank", "abut", "rear", "elsewhere",
    ),
    "distance / framing": (
        "close", "closeup", "far", "distance", "apart", "angle", "zoom", "crop",
        # Camera moves. The invariance clause promises the framing is unchanged; a prompt that
        # pans or orbits breaks the same pixel-to-label correspondence a moved apple does.
        "pan", "dolly", "orbit", "swivel", "viewpoint", "vantage", "perspective",
        # Reachable only because the invariance clause is deleted before the scan (see
        # _scannable): these words are in every committed prompt, and in none of them twice.
        "camera", "frame", "shoot", "lens", "focal", "telephoto", "macro", "wide", "aerial",
        "eye", "handheld", "track", "truck", "pedestal", "composition",
        "field of view", "point of view", "pov",
    ),
    "height / scale": (
        "height", "high", "low", "tall", "short", "raise", "lift", "elevate", "lofty",
        "large", "small", "big", "size", "resize", "scale", "giant", "gigantic", "huge",
        "enormous", "massive", "colossal", "tiny", "miniature", "oversized", "undersized",
        "shrink", "enlarge", "magnify", "expand", "thick", "thin", "narrow", "shallow",
        # Table height stated against a body — "a low table, waist-high".
        "waist", "knee", "chest", "hip",
        # The comparison marker itself, which catches the construction rather than the adjective.
        # "The table is deeper than before" is a scale change whose adjective ("deep") cannot be
        # forbidden — "deep shadows", "deep teal", "deep red" are all committed, allowed text —
        # so the word that makes it a comparison is forbidden instead. It also backstops every
        # comparative the stem list has not thought of yet.
        "than",
    ),
    "object count": (
        "add", "additional", "extra", "another", "second", "third", "fourth", "few", "many",
        "more", "duplicate", "multiple", "several", "pair", "couple", "dozen", "numerous",
        "group", "cluster", "bunch", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "count", "twin", "copy", "clone", "replicate",
        # Counting DOWN is a count change too: a prompt that deletes the plate desynchronises the
        # labels exactly as thoroughly as one that adds a second apple.
        "remove", "delete", "omit", "erase", "gone", "absent", "miss", "vanish", "disappear",
        "empty", "bare", "none", "nothing", "nowhere", "no longer", "remain",
        "without a", "without the", "without any",
        "twice", "double", "triple", "crowd", "mound",
    ),
    "robot pose": (
        "pose", "posture", "reach", "grip", "grasp", "bend", "lean", "robot",
        "arm", "elbow", "wrist", "shoulder", "hand", "finger", "thumb", "claw", "joint",
        "manipulator", "effector", "extend", "retract", "fold", "tuck", "crouch", "upright",
    ),
}

# Simple English inflections. Matching on the bare stem would miss "repositioned"; matching on a
# substring would flag "highlights" for "high", so the boundary is not optional.
_INFLECTIONS = r"(?:s|es|d|ed|ing|er|ers|est)?"

# …and the three spelling changes English makes before those suffixes, without which the suffix
# list is a decoration: "move" + "ing" is "moving" (drop the -e), "pan" + "ed" is "panned" (double
# the final consonant) and "tiny" + "er" is "tinier" (y -> i). The first two were reachable holes
# — "moving" and "panned" passed the V1 lint while "moved" and "pans" did not, which is not a rule
# anybody could have followed; the third is what makes "tiny"/"tinier" and "empty"/"emptier" one
# entry instead of two.
_DOUBLES = re.compile(r"[^aeiou][aeiou][bdglmnprt]$")
_CONSONANT_Y = re.compile(r"[^aeiou]y$")

# The handful of paradigms English does not build with a suffix. Listed against their stem rather
# than as separate terms, so the stem stays the unit that is maintained: adding "far" still buys
# farther/furthest, and a reader looking for "shot" finds it under "shoot".
_IRREGULAR: dict[str, tuple[str, ...]] = {
    "far": ("farther", "farthest", "further", "furthest"),
    "hold": ("held",),
    "half": ("halve", "halves"),
    "take": ("took", "taken"),
    "lie": ("lying", "lain"),
    "lay": ("laid",),
    "stand": ("stood",),
    "hide": ("hidden",),
    "strew": ("strewn",),
    "throw": ("thrown", "threw"),
    "shoot": ("shot",),
}


def stems_of(term: str) -> set[str]:
    """The spellings a plain stem takes before _INFLECTIONS is appended.

    This is the function that makes "FORBIDDEN_TERMS holds plain stems" true rather than a claim
    in a comment. One stem in, its whole paradigm out:

        "big"   -> big, bigg      -> big/bigs/bigger/biggest
        "close" -> close, clos    -> close/closes/closer/closest/closing/close-up
        "tiny"  -> tiny, tini     -> tiny/tinier/tiniest
        "far"   -> far, farther…  -> far/farther/farthest/further/furthest

    Deliberately over-generates. The extra stems ("overr", "putt", "mani") are not English and
    match nothing; the cost of a stem that never fires is zero, and the cost of a missing one is a
    corpus of frames whose labels describe a different scene.
    """
    stems = {term}
    if term.endswith("e"):
        stems.add(term[:-1])            # nudge -> nudging, move -> moving, slide -> slid
    if len(term) >= 3 and _DOUBLES.search(term):
        stems.add(term + term[-1])      # pan -> panned/panning, grip -> gripped/gripper
    if len(term) >= 3 and _CONSONANT_Y.search(term):
        stems.add(term[:-1] + "i")      # tiny -> tinier/tiniest, empty -> emptier
    stems.update(_IRREGULAR.get(term, ()))
    return stems


def _term_pattern(term: str) -> re.Pattern[str]:
    """One compiled matcher per forbidden term: the term, its spelling variants, and a suffix."""
    alt = "|".join(re.escape(s) for s in sorted(stems_of(term), key=len, reverse=True))
    return re.compile(rf"\b(?:{alt}){_INFLECTIONS}\b", re.IGNORECASE)


# Compiled once, in report order: {rule: {term: pattern}}.
_TERM_PATTERNS: dict[str, dict[str, re.Pattern[str]]] = {
    rule: {term: _term_pattern(term) for term in terms}
    for rule, terms in FORBIDDEN_TERMS.items()
}


class Failure(Exception):
    """A check failed. Message is the report line."""


def rel(path: pathlib.Path) -> str:
    """A path for the report, repo-relative when it can be and never an exception when it cannot.

    ``Path.relative_to`` RAISES ValueError when the path is not under REPO, and a path handed in
    as ``configs/transfer25/styles.toml`` is not: it is relative to the cwd, so ``resolve()`` is
    what makes it comparable at all, and ``--partition /elsewhere/styles.toml`` is legitimately
    outside the tree. This used to be an unguarded ``args.partition.relative_to(REPO)`` in the
    success path of main(), i.e. AFTER every check had passed — so the one thing it could produce
    was a traceback and exit 1 over a partition that was fine. Reporting is not a check and must
    not be able to fail one.
    """
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def load(path: pathlib.Path) -> dict:
    if not path.is_file():
        raise Failure(f"partition file missing: {path}")
    return tomllib.loads(path.read_text())


HASH_EXCLUDES = ("hash", "consumer")


def canonical_json(doc: dict) -> str:
    """The bytes the partition hash is taken over: the document minus [hash] and [consumer].

    Same serialisation as make_t041_eval_prompts.py — ensure_ascii, no whitespace, sorted keys —
    so the hash is a function of the content and not of the formatting or the key order.

    [hash] is excluded because it describes the hash; a file cannot contain its own digest.
    [consumer] is excluded because it is operational wiring — which path the generator reads the
    rendering from — and the value PR-08 §6 records has to answer "which styles was this run
    generated under". Repointing a path must not invalidate the recorded hash of a finished run,
    and it cannot hide a style change: [consumer] holds a path and two key names, and the
    rendering it points at is checked against this document on every run of this script.
    """
    body = {k: v for k, v in doc.items() if k not in HASH_EXCLUDES}
    return json.dumps(body, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def content_hash(doc: dict) -> str:
    return hashlib.sha256(canonical_json(doc).encode()).hexdigest()


def styles_of(doc: dict) -> dict[str, list[dict]]:
    """The three groups, keyed by the name used in reports. Identity is a group of one."""
    return {
        "TRAIN_STYLES": list(doc.get("train_styles") or []),
        "EVAL_STYLES": list(doc.get("eval_styles") or []),
        "IDENTITY": [doc["identity_style"]] if doc.get("identity_style") else [],
    }


# --------------------------------------------------------------------------------------------
# checks. Each returns a list of report lines and raises Failure on a violation.
# --------------------------------------------------------------------------------------------

def check_structure(doc: dict) -> list[str]:
    for field in ("schema", "rule", "committed", "pre_registration", "task", "invariance_clause"):
        if not doc.get(field):
            raise Failure(f"missing provenance field: {field}")
    groups = styles_of(doc)
    if not groups["IDENTITY"]:
        raise Failure("no [identity_style] — arm C (real+identity) has no control to generate from")
    for name, styles in groups.items():
        if not styles:
            raise Failure(f"{name} is empty")
        for s in styles:
            for field in ("id", "prompt", *AXES):
                if not str(s.get(field, "")).strip():
                    raise Failure(f"{name}: style {s.get('id', '?')!r} has no {field}")
            if doc["invariance_clause"] not in s["prompt"]:
                raise Failure(
                    f"{name}: style {s['id']!r} does not carry the invariance clause. The "
                    "constraint has to be in the text the generator reads, not only in the file."
                )
            # `repeats` is how many clips this style contributes per episode, and it is REQUIRED
            # rather than defaulted. The consumer's work-list expansion reads it; a missing field
            # would be a thing to infer, and the last time arm C's size was inferred it came out
            # a tenth of arm B's. bool is an int in Python and `repeats = true` is not a count.
            r = s.get("repeats")
            if not isinstance(r, int) or isinstance(r, bool) or r < 1:
                raise Failure(
                    f"{name}: style {s['id']!r} has repeats={r!r}; it must be an integer >= 1. "
                    "Every style states how many clips per episode it contributes — the sbatch's "
                    "work-list expansion reads this rather than assuming 1."
                )
    ids = [s["id"] for styles in groups.values() for s in styles]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise Failure(f"duplicate style ids: {dupes}")
    prompts = [s["prompt"] for styles in groups.values() for s in styles]
    if len(set(prompts)) != len(prompts):
        raise Failure("two styles carry byte-identical prompts — they are one style, not two")
    return [f"structure   OK  {len(ids)} styles, all fields present, ids and prompts unique"]


def check_volume(doc: dict) -> list[str]:
    """The clip arithmetic, and arm C's frame match to arm B (PR-08 §5).

    ``[volume]`` restates in numbers what ``repeats`` says per style. Restating invites drift, so
    the restatement is checked rather than trusted: every count here is recomputed from the styles
    and compared. The load-bearing line is ``identity_clips == train_clips`` — arm C is the
    generator-fingerprint control, "same generator, same pipeline, same frame count, no added
    diversity", and a control that is a tenth of the arm it controls for cannot distinguish
    "diversity helped" from "more data helped". That is the one confound arm C exists to remove.
    """
    vol = doc.get("volume")
    if not vol:
        raise Failure(
            "no [volume] — the clip counts are what make arm C's frame match to arm B checkable, "
            "and PR-08 §5 requires the match."
        )
    for field in ("episodes", "train_style_count", "train_repeats", "train_clips",
                  "identity_style_count", "identity_repeats", "identity_clips",
                  "eval_style_count", "eval_repeats", "eval_clips_per_episode", "eval_clips",
                  "training_clips", "whole_partition_clips", "style_instances",
                  "gpu_h_ceiling_scope", "gpu_h_ceiling_basis_clips"):
        if vol.get(field) is None:
            raise Failure(f"[volume] is missing {field}")

    eps = vol["episodes"]
    src_eps = (doc.get("source") or {}).get("episodes")
    if src_eps is not None and eps != src_eps:
        raise Failure(f"[volume].episodes {eps} != [source].episodes {src_eps}")

    groups = styles_of(doc)
    out = []
    for prefix, name in (("train", "TRAIN_STYLES"), ("eval", "EVAL_STYLES"),
                         ("identity", "IDENTITY")):
        styles = groups[name]
        if vol[f"{prefix}_style_count"] != len(styles):
            raise Failure(
                f"[volume].{prefix}_style_count = {vol[f'{prefix}_style_count']} but {name} has "
                f"{len(styles)} styles"
            )
        # One repeat count per set, so the set has a single well-defined per-episode clip rate.
        seen = {s["repeats"] for s in styles}
        if seen != {vol[f"{prefix}_repeats"]}:
            raise Failure(
                f"[volume].{prefix}_repeats = {vol[f'{prefix}_repeats']} but {name} styles declare "
                f"{sorted(seen)}. The declared rate and the styles' own field must agree."
            )

    # Every set runs over every episode: T40_RULE_V2 §3.1 prices EVAL_STYLES at 5 x 402 like the
    # others, and the sbatch's work-list strides over the whole manifest rather than an eval subset.
    for prefix in ("train", "identity", "eval"):
        want = eps * vol[f"{prefix}_style_count"] * vol[f"{prefix}_repeats"]
        if vol[f"{prefix}_clips"] != want:
            raise Failure(
                f"[volume].{prefix}_clips = {vol[f'{prefix}_clips']}, but {eps} episodes x "
                f"{vol[f'{prefix}_style_count']} styles x {vol[f'{prefix}_repeats']} repeats = "
                f"{want}"
            )
    want_eval_rate = vol["eval_style_count"] * vol["eval_repeats"]
    if vol["eval_clips_per_episode"] != want_eval_rate:
        raise Failure(
            f"[volume].eval_clips_per_episode = {vol['eval_clips_per_episode']}, but "
            f"{vol['eval_style_count']} styles x {vol['eval_repeats']} repeats = {want_eval_rate}"
        )

    if vol["identity_clips"] != vol["train_clips"]:
        raise Failure(
            f"ARM C IS NOT FRAME-MATCHED: identity {vol['identity_clips']} clips vs train "
            f"{vol['train_clips']}. PR-08 §5 gives arm C the SAME FRAME COUNT as arm B — that is "
            "how the control separates 'diversity helped' from 'more data helped'. Raise "
            "[identity_style].repeats to match the TRAIN_STYLES count; do not subsample arm B, "
            "which is the arm the experiment is about."
        )

    # The two totals T40_RULE_V2 §3 names. They answer different questions and are both checked:
    # training_clips is what reaches a training corpus (arms B + C), whole_partition_clips is the
    # basis the GPU-h ceiling is derived over.
    want_training = vol["train_clips"] + vol["identity_clips"]
    if vol["training_clips"] != want_training:
        raise Failure(
            f"[volume].training_clips = {vol['training_clips']}, but train {vol['train_clips']} + "
            f"identity {vol['identity_clips']} = {want_training} (T40_RULE_V2 §3.1: arms B + C)."
        )
    want_whole = want_training + vol["eval_clips"]
    if vol["whole_partition_clips"] != want_whole:
        raise Failure(
            f"[volume].whole_partition_clips = {vol['whole_partition_clips']}, but train "
            f"{vol['train_clips']} + identity {vol['identity_clips']} + eval {vol['eval_clips']} = "
            f"{want_whole}."
        )
    want_instances = sum(vol[f"{p}_style_count"] * vol[f"{p}_repeats"]
                         for p in ("train", "identity", "eval"))
    if vol["style_instances"] != want_instances:
        raise Failure(
            f"[volume].style_instances = {vol['style_instances']}, but the sets declare "
            f"{want_instances}. T40_RULE_V2 §3.1 prices the run per style-instance."
        )
    if vol["gpu_h_ceiling_basis_clips"] != vol["whole_partition_clips"]:
        raise Failure(
            f"[volume].gpu_h_ceiling_basis_clips = {vol['gpu_h_ceiling_basis_clips']} but the "
            f"whole partition is {vol['whole_partition_clips']} clips. T40_RULE_V2 §3.2 derives "
            "the ceiling over the whole partition; a basis that is not the whole partition is a "
            "per-set allowance wearing the whole-partition name."
        )

    if vol["gpu_h_ceiling_scope"] != "whole-partition":
        raise Failure(
            f"[volume].gpu_h_ceiling_scope = {vol['gpu_h_ceiling_scope']!r}. PR-08 §8 item 3's "
            "ceiling covers train + eval + identity; the sbatch's gate is per-invocation and "
            "would approve the same ceiling once per STYLE_SET."
        )

    out.append(
        f"volume      OK  train {vol['train_clips']} = identity {vol['identity_clips']} clips "
        f"({eps} episodes); arm C frame-matched to arm B"
    )
    out.append(
        f"            whole partition {vol['whole_partition_clips']} clips over "
        f"{vol['style_instances']} style-instances = the GPU-h ceiling basis"
    )
    return out


# --------------------------------------------------------------------------------------------
# The seed schedule. Its own check because it is its own failure mode: every count in [volume] can
# be right while the schedule underneath them re-creates the confound arm C exists to remove.
# --------------------------------------------------------------------------------------------

def instances_of(styles: list[dict]) -> int:
    """Style-instances per episode in a set: sum of `repeats` over its styles."""
    return sum(int(s["repeats"]) for s in styles)


def resolved_seeds(doc: dict) -> dict[str, dict[str, list[int]]]:
    """Apply the committed assignment rule: ``{set_key: {style_id: [seed per repeat]}}``.

    ``seed_index = style_index * repeats + repeat_index``. With one repeat count per set (which
    check_volume enforces) that enumerates the set's (style, repeat) pairs bijectively onto
    0 .. instances-1, so one seed per style-instance is exactly a block as long as the set.

    This is where the schedule is *resolved*, and the rendering carries the result rather than the
    rule — the consumer must never re-derive an index rule of its own, because the rule it would
    have re-derived last time ("repeats = 1 uses seeds[0]") is the defect.
    """
    blocks = (doc.get("seed_schedule") or {}).get("blocks") or {}
    groups = styles_of(doc)
    out: dict[str, dict[str, list[int]]] = {}
    for key, name in SETS:
        block = list(blocks.get(key) or [])
        per: dict[str, list[int]] = {}
        for i, s in enumerate(groups[name]):
            n = int(s["repeats"])
            seeds = []
            for r in range(n):
                j = i * n + r
                if j >= len(block):
                    raise Failure(
                        f"[seed_schedule.blocks].{key} has {len(block)} seeds but style "
                        f"{s['id']!r} (index {i}, repeats {n}) needs index {j}. Every "
                        "style-instance is generated under a committed seed; there is no default."
                    )
                seeds.append(int(block[j]))
            per[s["id"]] = seeds
        out[key] = per
    return out


def check_seed_schedule(doc: dict) -> list[str]:
    """Which style-instance is generated under which initial-noise seed (T40_STYLES_V1, amended).

    THE TWO RULES THAT ARE NOT ABOUT TIDINESS. Both were violated by the file as first committed,
    where every TRAIN and every EVAL style had ``repeats = 1`` and the rule was "repeat r uses
    repeat_seeds[r]; repeats = 1 uses repeat_seeds[0]" — so arm B's 4 020 clips were all seed 7001,
    the eval set's 2 010 clips were all seed 7001, and arm C alone spanned 7001-7010:

    * ``train`` and ``identity`` must span the SAME seed set. Otherwise B − C is a difference of
      seeds as well as of prompts, and arm C stops isolating diversity — which is the only thing
      it is for.
    * ``eval`` must be DISJOINT from both. Otherwise training clips in one arm share an initial
      latent with the clips the headline is scored on, and any transferable seed-specific
      generator fingerprint matches that arm to the scoring distribution for a reason unrelated to
      diversity. That inflates the headline and it can manufacture a p.

    Both are cheap to satisfy — the fix costs zero extra clips — and both are only available
    before the first clip exists, which is why they are enforced here rather than reviewed later.
    """
    sched = doc.get("seed_schedule")
    if not isinstance(sched, dict):
        raise Failure(
            "no [seed_schedule] — the seed VALUES are not the commitment, the assignment is. A "
            "bare list of ten seeds is satisfied by a schedule that puts all of arm B and all of "
            "the eval set on one of them, which is what the first committed version did."
        )
    if sched.get("assignment") != SEED_ASSIGNMENT:
        raise Failure(
            f"[seed_schedule].assignment = {sched.get('assignment')!r}; this checker and the "
            f"consumer implement exactly one rule, {SEED_ASSIGNMENT!r}. A partition naming an "
            "assignment nothing implements would be generated under whatever the consumer does "
            "instead."
        )
    if sched.get("rule") != SEED_RULE:
        raise Failure(
            "[seed_schedule].rule does not match the implemented rule. It is normative — it is "
            "what a reader recomputing a clip's seed by hand follows — so it is pinned to the "
            f"code:\n  file {sched.get('rule')!r}\n  code {SEED_RULE!r}"
        )
    if sched.get("one_seed_per_style_instance") is not True:
        raise Failure(
            "[seed_schedule].one_seed_per_style_instance must be true. A block longer than its "
            "set commits a seed no clip is generated under; a shorter one makes some "
            "style-instance's seed a thing to infer."
        )
    blocks = sched.get("blocks")
    if not isinstance(blocks, dict):
        raise Failure("[seed_schedule.blocks] is missing — one committed seed block per set")

    groups = styles_of(doc)
    spans: dict[str, list[int]] = {}
    for key, name in SETS:
        block = blocks.get(key)
        if not isinstance(block, list) or not block:
            raise Failure(f"[seed_schedule.blocks].{key} is missing or empty")
        for v in block:
            if not isinstance(v, int) or isinstance(v, bool):
                raise Failure(f"[seed_schedule.blocks].{key} holds {v!r}; seeds are integers")
        if len(set(block)) != len(block):
            raise Failure(
                f"[seed_schedule.blocks].{key} has duplicate seeds: {block}. Two style-instances "
                "under one seed are one sample of the generator's variance recorded twice."
            )
        want = instances_of(groups[name])
        if len(block) != want:
            raise Failure(
                f"[seed_schedule.blocks].{key} commits {len(block)} seeds but {name} expands to "
                f"{want} style-instances per episode. one_seed_per_style_instance means exactly "
                "one: a longer block commits a seed nothing runs under, a shorter one leaves a "
                "style-instance without a committed seed."
            )
        spans[key] = block

    # Resolve the schedule the way the rendering does, so the two rules below are checked against
    # what will actually be generated rather than against the block as written.
    per_style = resolved_seeds(doc)
    used = {key: sorted({s for seeds in per_style[key].values() for s in seeds}) for key, _ in SETS}
    for key in used:
        if used[key] != sorted(spans[key]):
            raise Failure(
                f"the assignment does not use every seed in [seed_schedule.blocks].{key}: block "
                f"{sorted(spans[key])}, used {used[key]}."
            )

    train, identity, ev = set(used["train"]), set(used["identity"]), set(used["eval"])

    if train != identity:
        raise Failure(
            "ARM B AND ARM C DO NOT SPAN THE SAME SEED SET: train "
            f"{sorted(train)} vs identity {sorted(identity)} (only in train "
            f"{sorted(train - identity)}, only in identity {sorted(identity - train)}). PR-08 §5's "
            "arm C is the control that decides whether a gain in arm B is diversity or the "
            "generator; if the two arms are also generated under different seeds then B - C is a "
            "difference of seeds AND prompts and isolates nothing. Give the train block one seed "
            "per TRAIN style and the identity block the same seeds, one per repeat — it costs no "
            "extra clips."
        )

    if ev & (train | identity):
        raise Failure(
            "EVAL SEEDS INTERSECT A TRAINING SEED BLOCK: "
            f"{sorted(ev & (train | identity))}. The headline is scored on EVAL_STYLES clips, so "
            "an eval clip generated from the same initial latent as a training clip hands the arm "
            "that owns that training clip a match to the scoring distribution that comes from the "
            "generator's seed-specific behaviour and not from anything the experiment is about — "
            "it inflates the headline and it can manufacture a p. This is the confound arm C "
            "exists to remove, re-entering through the seed schedule. Give EVAL_STYLES a seed "
            "block disjoint from both training blocks."
        )

    eps = (doc.get("volume") or {}).get("episodes")
    per_seed = f", {eps} clips per seed" if isinstance(eps, int) else ""
    return [
        f"seeds       OK  assignment {SEED_ASSIGNMENT!r}; one seed per style-instance",
        f"            train {sorted(train)}",
        f"            identity spans the identical set — arm B and arm C differ only in prompt"
        f"{per_seed}",
        f"            eval {sorted(ev)} disjoint from both training blocks",
    ]


def check_blocking_todos(doc: dict) -> list[str]:
    """Open items that block GENERATION, reported and not enforced — and that is deliberate.

    PR-08 §1 licenses committing this file and forbids generating from it, so an open TODO must
    not make the checker fail: a red check on a file that is correct-as-committed teaches people
    to stop running the checker, which is the failure this whole script is trying to avoid. They
    are printed on every run instead, because prose in a comment is what a blocking item stops
    being when it becomes a record.
    """
    todos = doc.get("blocking_todos") or []
    lines = []
    for t in todos:
        for field in ("id", "status", "blocks"):
            if not str(t.get(field, "")).strip():
                raise Failure(f"blocking_todos entry missing {field}: {t!r}")
        if t["status"] not in ("OPEN", "CLOSED"):
            raise Failure(f"blocking_todos {t['id']}: status {t['status']!r} is not OPEN/CLOSED")
    open_ = [t for t in todos if t["status"] == "OPEN"]
    if not open_:
        lines.append(f"todos       OK  {len(todos)} recorded, 0 open")
        return lines
    lines.append(
        f"todos       {len(open_)} OPEN of {len(todos)} — these block GENERATION, not this check"
    )
    for t in open_:
        lines.append(f"            {t['id']}  blocks: {t['blocks']}")
    return lines


def check_disjoint(doc: dict) -> list[str]:
    groups = styles_of(doc)
    out = []
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ida = {s["id"] for s in groups[a]}
            idb = {s["id"] for s in groups[b]}
            if ida & idb:
                raise Failure(f"{a} and {b} share style ids: {sorted(ida & idb)}")
            for axis in AXES:
                sa = {s[axis] for s in groups[a]}
                sb = {s[axis] for s in groups[b]}
                if sa & sb:
                    raise Failure(
                        f"{a} and {b} share {axis} slugs: {sorted(sa & sb)}. Different ids do not "
                        "make different appearance, and the headline is read off the difference."
                    )
            out.append(f"disjoint    OK  {a} ∩ {b} = ∅ on id and on all {len(AXES)} axes")
    # Stated separately because PR-08 §5 gives it its own reason: identity is a control, and a
    # control that is also a style is not a control.
    ident = groups["IDENTITY"][0]["id"]
    out.append(f"identity    OK  {ident!r} is in neither TRAIN_STYLES nor EVAL_STYLES")
    return out


def _strings(node, path: str = ""):
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _strings(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _strings(v, f"{path}[{i}]")


def _scannable(text: str, clause: str) -> str:
    """The style's OWN words: the committed invariance clause deleted, everything else kept.

    Every prompt ends with the clause verbatim — check_structure fails the file if one does not —
    and the clause says "Scene geometry, camera framing and the robot are unchanged." Scanning it
    is what made "camera", "framing", "robot" and "geometry" unusable as forbidden terms, which is
    why the framing category had nothing to catch "Shot from overhead, a wide close-up." with.

    Deleting a constant this file pins is not an exemption: the string removed is fixed by
    [invariance_clause] and identical in all 16 styles, so a style author cannot widen it, and a
    prompt that alters one word of it ("…are changed") is no longer that constant and is scanned
    in full. All occurrences go, so a second copy cannot smuggle anything either.
    """
    return text.replace(clause, " ") if clause else text


def check_geometry_terms(doc: dict) -> list[str]:
    clause = str(doc.get("invariance_clause") or "")
    hits: list[str] = []
    scanned = 0
    for name, styles in styles_of(doc).items():
        for s in styles:
            for field, text in _strings(s):
                scanned += 1
                probe = _scannable(text, clause)
                for rule, patterns in _TERM_PATTERNS.items():
                    for term, pattern in patterns.items():
                        m = pattern.search(probe)
                        if m:
                            hits.append(
                                f"  {name} {s['id']}.{field}: {m.group(0)!r} "
                                f"[forbidden: {rule}, stem {term!r}] in {text!r}"
                            )
    if hits:
        raise Failure("geometry-moving term(s) in style text:\n" + "\n".join(hits))
    n_terms = sum(len(t) for t in FORBIDDEN_TERMS.values())
    return [f"geometry    OK  {scanned} strings scanned against {n_terms} forbidden stems, 0 hits"]


def check_hash(doc: dict, path: pathlib.Path, write: bool) -> list[str]:
    digest = content_hash(doc)

    # Reproducible means: same content, same digest, regardless of how the document got here.
    # Re-parsing the file and shuffling the key order must both land on the same value, or the
    # digest is a hash of the parse rather than of the partition.
    again = content_hash(load(path))
    if again != digest:
        raise Failure(f"hash not reproducible across a re-read: {digest} != {again}")
    shuffled = content_hash(dict(reversed(list(doc.items()))))
    if shuffled != digest:
        raise Failure(f"hash depends on key order: {digest} != {shuffled}")

    sidecar = path.parent / (path.name + ".sha256")
    if write:
        sidecar.write_text(digest + "\n")
        return [f"hash        WROTE {rel(sidecar)}", f"            {digest}"]
    if not sidecar.is_file():
        raise Failure(f"sidecar {sidecar} missing — run with --write-hash to create it")
    recorded = sidecar.read_text().strip()
    if recorded != digest:
        raise Failure(
            f"sidecar hash does not match the file's content:\n  sidecar  {recorded}\n"
            f"  computed {digest}\nThe partition changed without the recorded hash changing, or "
            "the other way round. PR-08 §6 records this value with the verdict."
        )
    return [
        "hash        OK  reproducible across re-read and key reordering; sidecar matches",
        f"            {digest}",
    ]


def emit_json(doc: dict) -> str:
    """The generator-facing rendering of the partition, deterministic to the byte.

    Key names are the consumer's ("train", "eval", "identity"), not the TOML's, because the
    consumer is a committed sbatch and renaming its keys is not this file's call. The source's
    content hash travels with the rendering so a work-unit stamp can name which partition it came
    from even though the two files hash differently (one is content, one is bytes).

    ``repeats`` rides on every style, and ``volume`` on the payload, because the consumer's
    work-list expansion has to know that the identity set is ten clips per episode and the train
    set is one. Inferring it is what produced an arm C at a tenth of arm B's volume.

    ``seeds`` rides on every style as the schedule ALREADY RESOLVED — ``seeds[r]`` is the seed for
    repeat r of that style, in repeat order, one entry per repeat. The consumer therefore never
    re-derives an index rule, which matters because the rule it used to re-derive ("repeats = 1
    uses seeds[0]") put all 4 020 arm-B clips and all 2 010 eval clips on one shared seed while
    arm C spanned ten. ``seed_schedule`` travels too, so the rendering carries the rule and the
    blocks next to the resolved values and a reader can check one against the other.
    ``blocking_todos`` rides along so an operator reading only the generator-facing file still
    sees what must close before generation.
    """
    per_style = resolved_seeds(doc)

    def style(set_key: str):
        def one(s: dict) -> dict:
            return {"id": s["id"], "prompt": s["prompt"], "repeats": s["repeats"],
                    "seeds": per_style[set_key][s["id"]], **{a: s[a] for a in AXES}}
        return one

    payload = {
        "schema": doc["schema"],
        "rule": doc["rule"],
        "committed": doc["committed"],
        "pre_registration": doc["pre_registration"],
        "source": "configs/transfer25/styles.toml",
        "source_content_sha256": content_hash(doc),
        "volume": doc["volume"],
        "seed_schedule": doc["seed_schedule"],
        "blocking_todos": doc.get("blocking_todos") or [],
        "train": [style("train")(s) for s in doc["train_styles"]],
        "eval": [style("eval")(s) for s in doc["eval_styles"]],
        "identity": [style("identity")(doc["identity_style"])],
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=1) + "\n"


def check_derived(doc: dict, derived: pathlib.Path, write: bool) -> list[str]:
    text = emit_json(doc)
    if write:
        derived.write_text(text)
        (derived.parent / (derived.name + ".sha256")).write_text(
            hashlib.sha256(text.encode()).hexdigest() + "\n")
        return [f"derived     WROTE {rel(derived)}"]
    if not derived.is_file():
        return [f"derived     NOTE  {rel(derived)} not generated (--emit-json)"]
    if derived.read_text() != text:
        raise Failure(
            f"{derived} has drifted from {DEFAULT_PARTITION.name}. Regenerate it with "
            "--emit-json; do not hand-edit it. The generator reads the derived file, so a drift "
            "means clips were restyled under a partition that is not the committed one."
        )
    return ["derived     OK  consumer JSON rendering matches the committed source"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition", type=pathlib.Path, default=DEFAULT_PARTITION)
    ap.add_argument("--write-hash", action="store_true",
                    help="write the sidecar .sha256 instead of checking it (intended changes only)")
    ap.add_argument("--derived", type=pathlib.Path, default=DEFAULT_DERIVED,
                    help="generator-facing JSON rendering (97_transfer25_restyle.sbatch reads it)")
    ap.add_argument("--emit-json", action="store_true",
                    help="regenerate --derived from the committed source instead of checking it")
    args = ap.parse_args(argv)

    try:
        doc = load(args.partition)
        lines = check_structure(doc)
        lines += check_volume(doc)
        lines += check_seed_schedule(doc)
        lines += check_disjoint(doc)
        lines += check_geometry_terms(doc)
        lines += check_hash(doc, args.partition, args.write_hash)
        lines += check_derived(doc, args.derived, args.emit_json)
        lines += check_blocking_todos(doc)
    except Failure as exc:
        print(f"FAIL  {args.partition}", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1

    groups = styles_of(doc)
    print(f"{rel(args.partition)}  rule={doc['rule']}  committed={doc['committed']}")
    print(f"  TRAIN_STYLES {len(groups['TRAIN_STYLES'])}   EVAL_STYLES {len(groups['EVAL_STYLES'])}"
          f"   identity {len(groups['IDENTITY'])} (control, in neither)")
    vol = doc["volume"]
    print(f"  arm B/D {vol['train_clips']} clips (x{vol['train_repeats']})   "
          f"arm C {vol['identity_clips']} clips (x{vol['identity_repeats']})   "
          f"eval {vol['eval_clips_per_episode']}/episode")
    for line in lines:
        print("  " + line)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
