"""The ingest for a person's mask verdicts, tested on the ways it could silently lie.

Every test here is about the same risk: this tool expands one statement about a sheet into many
per-frame verdicts, so a mistake attributes a judgement to a person who never made it.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location(
    "record_mask_audit_verdicts", REPO_ROOT / "scripts" / "record_mask_audit_verdicts.py"
)
assert spec is not None and spec.loader is not None
rec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rec)


def make_audit(tmp_path: pathlib.Path, strata: dict[str, int]) -> pathlib.Path:
    """An audit dir whose sheets on disk match the chunking the tool reconstructs."""
    frames = []
    for stratum, n in strata.items():
        for i in range(n):
            frames.append(
                {
                    "episode": f"episode_{i:06d}",
                    "frame_index": i,
                    "stratum": stratum,
                    "role": "anchor",
                    "flags": [],
                    "detection_score": 0.8,
                    "mask_area_px": 100,
                    "looked_at": False,
                    "verdict": None,
                    "observed": "",
                }
            )
    audit = tmp_path / "audit"
    (audit / "sheets").mkdir(parents=True)
    for stratum, n in strata.items():
        for s in range((n + rec.SHEET_TILES - 1) // rec.SHEET_TILES):
            (audit / "sheets" / f"{stratum}-{s:02d}.png").write_bytes(b"")
    (audit / "OBSERVATIONS.template.json").write_text(
        json.dumps({"schema": "x", "step": "y", "contact_sheets": [], "frames": frames})
    )
    return audit


def load(audit: pathlib.Path, **kw):
    return rec.build(
        json.loads((audit / "OBSERVATIONS.template.json").read_text()),
        sheets_dir=audit / "sheets",
        **kw,
    )


def test_frames_on_sheets_nobody_named_are_never_marked_looked_at(tmp_path) -> None:
    """The core guarantee. A reviewer who saw one sheet has judged twelve frames, not 24."""
    audit = make_audit(tmp_path, {"grasp": 24})
    out = load(audit, reviewer="p", reviewed={"grasp-00": "apple"}, exceptions=[])

    seen = [f for f in out["frames"] if f["looked_at"]]
    unseen = [f for f in out["frames"] if not f["looked_at"]]
    assert len(seen) == 12 and len(unseen) == 12
    assert all(f["sheet"] == "grasp-00" for f in seen)
    assert all(f["verdict"] is None and f["verdict_source"] is None for f in unseen)
    assert out["coverage"]["frames_reviewed_fraction"] == pytest.approx(0.5)


def test_an_exception_on_an_unreviewed_sheet_is_refused(tmp_path) -> None:
    """Otherwise it records a verdict on a frame nobody looked at -- the exact failure."""
    audit = make_audit(tmp_path, {"grasp": 24})
    with pytest.raises(SystemExit, match="not passed with --sheet"):
        load(
            audit,
            reviewer="p",
            reviewed={"grasp-00": "apple"},
            exceptions=[("episode_000020", 20, "wrong_object", "")],
        )


def test_the_sheet_default_and_an_explicit_exception_stay_distinguishable(tmp_path) -> None:
    """A reader must be able to tell the twelve implied verdicts from the one that was typed."""
    audit = make_audit(tmp_path, {"grasp": 12})
    out = load(
        audit,
        reviewer="p",
        reviewed={"grasp-00": "apple"},
        exceptions=[("episode_000003", 3, "wrong_object", "plate")],
    )

    by_src = {f["verdict_source"] for f in out["frames"]}
    assert by_src == {"sheet_default", "explicit_exception"}
    odd = next(f for f in out["frames"] if f["verdict_source"] == "explicit_exception")
    assert odd["verdict"] == "wrong_object" and odd["observed"] == "plate"
    assert out["verdict_tally"] == {"apple": 11, "wrong_object": 1}


def test_a_flagged_sheet_may_not_be_used_as_a_review_unit(tmp_path) -> None:
    """Its tiles also sit on their stratum sheets; a default would write each verdict twice."""
    audit = make_audit(tmp_path, {"grasp": 12})
    (audit / "sheets" / "flagged-00.png").write_bytes(b"")
    with pytest.raises(SystemExit, match="not stratum sheets"):
        load(audit, reviewer="p", reviewed={"flagged-00": "apple"}, exceptions=[])


def test_a_mapping_that_disagrees_with_the_sheets_on_disk_refuses(tmp_path) -> None:
    """An off-by-one attaches a person's verdict to tiles they never saw. Refuse, do not guess."""
    audit = make_audit(tmp_path, {"grasp": 24})
    (audit / "sheets" / "grasp-01.png").unlink()
    with pytest.raises(SystemExit, match="disagrees with the sheets on disk"):
        load(audit, reviewer="p", reviewed={"grasp-00": "apple"}, exceptions=[])


def test_it_records_which_of_blocker_1s_three_strata_the_sample_missed(tmp_path) -> None:
    """Blocker 1 names occluded, apple-out-of-frame and the grasp. Coverage is recorded, not assumed."""
    audit = make_audit(tmp_path, {"grasp": 12, "occluded": 12, "min_visibility": 12})

    partial = load(audit, reviewer="p", reviewed={"grasp-00": "apple"}, exceptions=[])
    assert partial["blocker_1_named_strata"]["sample_spans_what_blocker_1_names"] is False
    assert set(partial["blocker_1_named_strata"]["not_covered"]) == {"occluded", "min_visibility"}

    full = load(
        audit,
        reviewer="p",
        exceptions=[],
        reviewed={"grasp-00": "apple", "occluded-00": "apple", "min_visibility-00": "apple"},
    )
    assert full["blocker_1_named_strata"]["sample_spans_what_blocker_1_names"] is True
    assert full["blocker_1_named_strata"]["not_covered"] == []


def test_it_discharges_nothing_and_says_so(tmp_path) -> None:
    audit = make_audit(tmp_path, {"grasp": 12})
    out = load(audit, reviewer="p", reviewed={"grasp-00": "apple"}, exceptions=[])
    assert "evidence, not a discharge" in out["discharges_no_blocker"]
    assert "GATE_QUALIFIED" not in json.dumps(out)


def test_the_reconstruction_matches_the_real_audit_on_disk() -> None:
    """The mapping is only trustworthy if it holds against the artifact it will actually read."""
    audit = REPO_ROOT / "runs" / "pr08-mask-audit"
    if not (audit / "OBSERVATIONS.template.json").exists():
        pytest.skip("runs/pr08-mask-audit not present (gitignored)")
    template = json.loads((audit / "OBSERVATIONS.template.json").read_text())
    mapping = rec.sheet_index(template["frames"])
    rec.verify_mapping(mapping, audit / "sheets")  # raises SystemExit on any disagreement


def test_a_verdict_outside_the_fixed_vocabulary_is_refused() -> None:
    with pytest.raises(SystemExit, match="verdict must be one of"):
        rec.parse_exception("episode_000000:0=looks_fine")
