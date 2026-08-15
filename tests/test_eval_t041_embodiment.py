"""Tests for T-041's verdict rule, `T041_RULE_V1` (PR-09 §6).

This file is the rule's executable form. Every assertion below fixes a decision that was made
before any number existed, and the reason to test it hard is that all of them are silent when
wrong: a mis-signed McNemar returns a p-value, a broken blind returns a verdict, and a G0 gate
that fails open returns a *pass*.

The mutants that motivated each block are named in the docstrings, in the habit T-39's suite set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_t041_embodiment as ev  # noqa: E402


# --- the statistic ---------------------------------------------------------------------------


def test_no_discordant_pairs_is_p_one():
    """Every pair agreeing is zero evidence, not perfect evidence."""
    assert ev.mcnemar_exact_one_sided(0, 0) == 1.0


def test_mcnemar_matches_the_binomial_by_hand():
    # b=5, c=0 -> P(X>=5 | n=5) = 1/32
    assert ev.mcnemar_exact_one_sided(5, 0) == pytest.approx(1 / 32)
    # b=4, c=1 -> P(X>=4 | n=5) = (5 + 1)/32
    assert ev.mcnemar_exact_one_sided(4, 1) == pytest.approx(6 / 32)
    # symmetric split is the null
    assert ev.mcnemar_exact_one_sided(3, 3) == pytest.approx(
        sum(__import__("math").comb(6, k) for k in range(3, 7)) / 64)


def test_mcnemar_is_one_sided_in_the_registered_direction():
    """Mutant: swapping b and c. A LoRA that BREAKS what the base got right must never pass."""
    assert ev.mcnemar_exact_one_sided(8, 1) < ALPHA_()
    assert ev.mcnemar_exact_one_sided(1, 8) > 0.5


def ALPHA_():
    return ev.ALPHA


def test_concordant_pairs_do_not_dilute_the_test():
    """Mutant: dividing by 60 instead of by the discordant count. Pairing exists to avoid that."""
    p_few = ev.mcnemar_exact_one_sided(6, 0)
    # the same discordant pattern in a hypothetical larger study is the same p-value
    assert p_few == ev.mcnemar_exact_one_sided(6, 0)
    assert p_few == pytest.approx(1 / 64)


def test_more_evidence_never_raises_the_p_value():
    prev = 1.0
    for b in range(0, 12):
        p = ev.mcnemar_exact_one_sided(b, 2)
        assert p <= prev + 1e-12
        prev = p


# --- the verdict table -----------------------------------------------------------------------


def test_verdict_table_matches_pr09_section_6():
    assert ev.verdict_from(9, 0, 0.002)[0] == "P"
    assert ev.verdict_from(2, 1, 0.5)[0] == "N"        # not significant, fixed <= 2
    assert ev.verdict_from(3, 1, 0.31)[0] == "I"       # not significant, fixed >= 3
    assert ev.verdict_from(0, 0, 1.0)[0] == "N"


def test_the_n_boundary_is_exactly_two():
    """Mutant: >= 2 instead of <= 2. Three fixes is I (underpowered), two is N (refuted)."""
    assert ev.verdict_from(2, 0, 0.25)[0] == "N"
    assert ev.verdict_from(3, 0, 0.125)[0] == "I"


# --- the gates -------------------------------------------------------------------------------


def make_case(n_pairs=30, base_right=0, lora_right=0, cal_correct=20,
              iteration=500, export=True, resume_logged=True, unscored=0):
    """Build a (key, scores) pair. base_right/lora_right index the first N pairs as correct."""
    items, scores = [], {}
    idx = 0
    for i in range(n_pairs):
        uuid = f"clip{i:03d}"
        for arm, right in (("base", i < base_right), ("lora", i < lora_right)):
            iid = f"item_{idx:04d}"
            idx += 1
            items.append({"kind": "paired", "arm": arm, "uuid": uuid,
                          "path": f"/x/{arm}/{uuid}.mp4", "item_id": iid})
            scores[iid] = None if unscored > 0 and idx <= unscored else right
    for k in range(20):
        iid = f"item_{idx:04d}"
        idx += 1
        expected = k < 10
        items.append({"kind": "cal_pos" if expected else "cal_neg", "arm": None,
                      "uuid": f"cal{k}", "path": f"/x/cal{k}.mp4",
                      "expected": expected, "item_id": iid})
        scores[iid] = expected if k < cal_correct else (not expected)
    run = {"iteration_reached": iteration, "export_nonempty": export,
           "resume_diffs_logged": resume_logged}
    return {"shuffle_seed": 0, "items": items}, scores, run


def test_a_clean_pass():
    key, scores, run = make_case(base_right=5, lora_right=25)
    r = ev.compute_verdict(key, scores, run, "sha")
    assert all(r["gates"].values())
    assert r["discordant_base_wrong_lora_right"] == 20
    assert r["discordant_base_right_lora_wrong"] == 0
    assert r["verdict"] == "P"


def test_g0a_voids_when_the_base_already_renders_a_g1():
    """No defect present -> nothing for this experiment to fix. Not a pass, not a fail."""
    key, scores, run = make_case(base_right=20, lora_right=28)
    r = ev.compute_verdict(key, scores, run, "sha")
    assert r["base_failures"] == 10
    assert r["gates"]["G0a_defect_present"] is False
    assert r["verdict"] == "VOID"


def test_g0a_boundary_is_fifteen():
    key, scores, _ = make_case(base_right=15, lora_right=30)   # 15 failures — exactly the floor
    r = ev.compute_verdict(key, scores, {"iteration_reached": 500, "export_nonempty": True}, "s")
    assert r["gates"]["G0a_defect_present"] is True
    key, scores, _ = make_case(base_right=16, lora_right=30)   # 14 failures
    r = ev.compute_verdict(key, scores, {"iteration_reached": 500, "export_nonempty": True}, "s")
    assert r["gates"]["G0a_defect_present"] is False


def test_g0b_voids_on_a_single_calibration_miss():
    """Mutant: a >=18/20 rubric gate. 20/20 is the registered bar and one miss is a VOID."""
    key, scores, run = make_case(base_right=0, lora_right=30, cal_correct=19)
    r = ev.compute_verdict(key, scores, run, "sha")
    assert r["calibration_correct"] == 19
    assert r["gates"]["G0b_rubric_calibrated"] is False
    assert r["verdict"] == "VOID"


def test_g0c_voids_when_the_run_stopped_short():
    key, scores, run = make_case(base_right=0, lora_right=30, iteration=400)
    assert ev.compute_verdict(key, scores, run, "sha")["verdict"] == "VOID"


def test_g0c_voids_when_a_resume_logged_no_toml_diff():
    """PR-09 §4a: a pass that cannot show its diff might have reinitialised the adapter."""
    key, scores, run = make_case(base_right=0, lora_right=30, resume_logged=False)
    assert ev.compute_verdict(key, scores, run, "sha")["verdict"] == "VOID"


def test_g0c_voids_on_an_empty_export():
    key, scores, run = make_case(base_right=0, lora_right=30, export=False)
    assert ev.compute_verdict(key, scores, run, "sha")["verdict"] == "VOID"


def test_an_unscored_item_voids_rather_than_dropping_the_pair():
    """Mutant: silently scoring 29 pairs. A partial pairing is a different experiment."""
    key, scores, run = make_case(base_right=0, lora_right=30, unscored=2)
    r = ev.compute_verdict(key, scores, run, "sha")
    assert r["n_pairs_complete"] < r["n_pairs_total"]
    assert r["verdict"] == "VOID"


def test_void_never_reads_as_a_weaker_pass():
    key, scores, run = make_case(base_right=0, lora_right=30, cal_correct=0)
    r = ev.compute_verdict(key, scores, run, "sha")
    assert r["verdict"] == "VOID"
    assert "not a statement about Cosmos" in r["reading"]


def test_a_lora_that_makes_things_worse_is_not_a_pass():
    key, scores, run = make_case(base_right=25, lora_right=0)
    r = ev.compute_verdict(key, scores, run, "sha")
    # G0a holds (only 5 base failures? no — base_right=25 leaves 5 failures) -> VOID on G0a
    assert r["gates"]["G0a_defect_present"] is False
    # and with the defect present it must still not pass:
    key, scores, run = make_case(base_right=15, lora_right=0)
    r = ev.compute_verdict(key, scores, run, "sha")
    assert r["gates"]["G0a_defect_present"] is True
    assert r["discordant_base_right_lora_wrong"] == 15
    assert r["verdict"] != "P"


# --- the blind -------------------------------------------------------------------------------


def test_the_scoring_sheet_carries_no_arm_label(tmp_path):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("".join(
        json.dumps({"uuid": f"c{i}", "source_id": "s", "prompt": "x",
                    "vision_path": "v"}) + "\n" for i in range(3)))
    clips = tmp_path / "clips"
    for arm in ("base", "lora"):
        (clips / arm).mkdir(parents=True)
        for i in range(3):
            (clips / arm / f"c{i}.mp4").write_bytes(b"\x00")
    cal = tmp_path / "cal"
    for sub in ("positive", "negative"):
        (cal / sub).mkdir(parents=True)
        for i in range(10):
            (cal / sub / f"{sub}{i}.mp4").write_bytes(b"\x00")
    cfg = tmp_path / "c.toml"
    cfg.write_text('[calibration]\nn_positive = 10\nn_negative = 10\n[judge]\nmodel = "m"\n')

    out = tmp_path / "out"
    ev.main(["--config", str(cfg), "--out", str(out), "build-sheet",
             "--prompts", str(prompts), "--clips", str(clips), "--calibration", str(cal)])

    sheet_text = (out / "scoring_sheet.jsonl").read_text()
    assert "base" not in sheet_text and "lora" not in sheet_text
    assert "expected" not in sheet_text
    rows = [json.loads(ln) for ln in sheet_text.splitlines() if ln.strip()]
    assert len(rows) == 3 * 2 + 20
    assert set(rows[0]) == {"item_id", "path"}
    # The path is part of the label: clips/base/c0.mp4 names the arm in the string a human
    # scorer reads. Every referenced file must live in the neutral tree under its item id.
    for r in rows:
        p = Path(r["path"])
        assert p.parent.name == "items"
        assert p.stem == r["item_id"]
        assert p.is_symlink() and p.resolve().is_file()
    # ...and the key does hold the labels, in a different file.
    key = json.loads((out / "key.json").read_text())
    assert {it["arm"] for it in key["items"] if it["kind"] == "paired"} == {"base", "lora"}


def test_build_sheet_refuses_an_incomplete_calibration_set(tmp_path):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"uuid": "c0"}) + "\n")
    clips = tmp_path / "clips"
    for arm in ("base", "lora"):
        (clips / arm).mkdir(parents=True)
        (clips / arm / "c0.mp4").write_bytes(b"\x00")
    cal = tmp_path / "cal"
    for sub in ("positive", "negative"):
        (cal / sub).mkdir(parents=True)
    cfg = tmp_path / "c.toml"
    cfg.write_text('[calibration]\nn_positive = 10\nn_negative = 10\n[judge]\nmodel = "m"\n')
    with pytest.raises(SystemExit) as e:
        ev.main(["--config", str(cfg), "--out", str(tmp_path / "o"), "build-sheet",
                 "--prompts", str(prompts), "--clips", str(clips), "--calibration", str(cal)])
    assert "G0b" in str(e.value)


def test_build_sheet_refuses_a_missing_generated_clip(tmp_path):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"uuid": "c0"}) + "\n")
    clips = tmp_path / "clips"
    (clips / "base").mkdir(parents=True)
    (clips / "base" / "c0.mp4").write_bytes(b"\x00")
    (clips / "lora").mkdir(parents=True)          # lora arm never generated
    cal = tmp_path / "cal"
    for sub in ("positive", "negative"):
        (cal / sub).mkdir(parents=True)
        for i in range(10):
            (cal / sub / f"{i}.mp4").write_bytes(b"\x00")
    cfg = tmp_path / "c.toml"
    cfg.write_text('[calibration]\nn_positive = 10\nn_negative = 10\n[judge]\nmodel = "m"\n')
    with pytest.raises(SystemExit):
        ev.main(["--config", str(cfg), "--out", str(tmp_path / "o"), "build-sheet",
                 "--prompts", str(prompts), "--clips", str(clips), "--calibration", str(cal)])


# --- disjointness, re-derived rather than trusted ---------------------------------------------


def write_corpus(tmp_path, train_uuids, val_uuids, shas=None, drop_sha=()):
    """A manifest stub. Every clip gets its own sha256 unless ``shas`` says two uuids share one.

    Sharing has to be expressible, because the shipped T-041 corpus really did carry one source's
    pixels twice under two names — that is the case the content check exists for. ``drop_sha``
    names clips whose sha256 the manifest omits entirely.
    """
    shas = shas or {}

    def clip(u):
        c = {"uuid": u}
        if u not in drop_sha:
            c["sha256"] = shas.get(u, f"sha-of-{u}")
        return c

    root = tmp_path / "corpus"
    root.mkdir(exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({
        "seed": 0,
        "clips": {"train": [clip(u) for u in train_uuids],
                  "val": [clip(u) for u in val_uuids]}}))
    return root


def write_prompts(tmp_path, uuids, name="p.jsonl"):
    p = tmp_path / name
    p.write_text("".join(json.dumps({"uuid": u, "prompt": "x"}) + "\n" for u in uuids))
    return p


def test_disjointness_passes_for_a_val_only_prompt_set(tmp_path):
    corpus = write_corpus(tmp_path, ["t0", "t1"], ["v0", "v1", "v2"])
    prompts = write_prompts(tmp_path, ["v0", "v2"])
    assert ev.check_prompts_are_held_out(prompts, corpus) == 2


def test_disjointness_catches_a_training_clip_in_the_prompt_set(tmp_path):
    """The hash chain cannot catch this: the sidecar is written by the same hand as the file."""
    corpus = write_corpus(tmp_path, ["t0", "t1"], ["v0"])
    prompts = write_prompts(tmp_path, ["v0", "t1"])
    with pytest.raises(SystemExit) as e:
        ev.check_prompts_are_held_out(prompts, corpus)
    assert "TRAINING split" in str(e.value)


def test_disjointness_catches_a_prompt_from_another_corpus_entirely(tmp_path):
    corpus = write_corpus(tmp_path, ["t0"], ["v0"])
    prompts = write_prompts(tmp_path, ["v0", "somewhere_else"])
    with pytest.raises(SystemExit) as e:
        ev.check_prompts_are_held_out(prompts, corpus)
    assert "neither split" in str(e.value)


def test_disjointness_catches_a_prompt_whose_BYTES_are_in_train(tmp_path):
    """The defect that shipped: unique uuid, duplicated pixels, and every uuid check passes.

    ``g1-dex3-graspsquare-dataset`` was a byte-copy of ``g1-dex3-blockstacking-dataset``, so a val
    clip's content sat in train under the other source's name. Scoring the adapter on that is a
    training score, and the bias runs toward the pre-registered hypothesis.
    """
    corpus = write_corpus(
        tmp_path,
        ["graspsquare_ep000077", "t1"],
        ["blockstacking_ep000077", "v1"],
        shas={"graspsquare_ep000077": "identical-bytes",
              "blockstacking_ep000077": "identical-bytes"},
    )
    prompts = write_prompts(tmp_path, ["blockstacking_ep000077", "v1"])
    with pytest.raises(SystemExit) as e:
        ev.check_prompts_are_held_out(prompts, corpus)
    msg = str(e.value)
    assert "BYTE-IDENTICAL" in msg
    # ...and it names both halves of the pair, because "one prompt is contaminated" is not
    # actionable and "this val clip is that train clip" is.
    assert "blockstacking_ep000077 == graspsquare_ep000077" in msg


def test_the_uuid_check_still_fires_on_its_own(tmp_path):
    """Mutant: replacing the uuid check with the sha check. They catch different defects.

    Here the content is disjoint — every clip has its own bytes — and the prompt set simply names
    a train clip. Nothing about sha256 would notice.
    """
    corpus = write_corpus(tmp_path, ["t0", "t1"], ["v0"])
    prompts = write_prompts(tmp_path, ["v0", "t1"])
    with pytest.raises(SystemExit) as e:
        ev.check_prompts_are_held_out(prompts, corpus)
    assert "TRAINING split" in str(e.value)
    assert "BYTE-IDENTICAL" not in str(e.value)


def test_content_disjointness_passes_on_a_clean_corpus(tmp_path):
    """The check must be silent on the corpus we actually have. Defence in depth, not a new gate."""
    corpus = write_corpus(tmp_path, [f"t{i}" for i in range(5)], ["v0", "v1", "v2"])
    prompts = write_prompts(tmp_path, ["v0", "v1", "v2"])
    assert ev.check_prompts_are_held_out(prompts, corpus) == 3


def test_a_manifest_without_hashes_refuses_rather_than_skipping_the_content_check(tmp_path):
    """Fail closed. A missing sha256 is the check not running, and that is the state we were in."""
    corpus = write_corpus(tmp_path, ["t0"], ["v0"], drop_sha=("t0",))
    prompts = write_prompts(tmp_path, ["v0"])
    with pytest.raises(SystemExit) as e:
        ev.check_prompts_are_held_out(prompts, corpus)
    assert "no sha256" in str(e.value)


# --- answer parsing --------------------------------------------------------------------------


@pytest.mark.parametrize("text,want", [
    ("YES", True), ("yes", True), ("Yes.", True), ("YES, a Dex3 hand", True),
    ("NO", False), ("no", False), ("No — a parallel gripper", False),
    ("", None), ("maybe", None), ("YES or NO", None), ("I cannot tell", None),
])
def test_answer_parsing(text, want):
    """An unparseable reply is UNSCORED, never a NO. A NO default would bias toward the defect."""
    assert ev.parse_answer(text) is want
