"""The comparison must produce questions, never a score.

Two properties carry the whole design. A disagreement that crosses the "is it the apple" line must
be separated from one that only differs in severity, because the first is a disagreement about
what is in the picture and the second is a matter of degree. And nowhere may the artifact compute
an accuracy, a precision or a correctness rate — that would silently make one of two correlated
observers into ground truth.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _module():
    spec = importlib.util.spec_from_file_location(
        "_cso", REPO_ROOT / "scripts" / "compare_second_opinion.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cso = _module()


def human(**pairs) -> dict:
    return {
        key: {"verdict": verdict, "section": "mask", "source": "sheet_default", "sheet": "s-00"}
        for key, verdict in pairs.items()
    }


def opinion(**pairs) -> dict:
    return {
        key: {"verdict": verdict, "confidence": "high", "saw": "", "sheet": "s-00", "group": 1}
        for key, verdict in pairs.items()
    }


def test_a_severity_difference_is_not_a_disagreement():
    """`apple` vs `partial` is a matter of degree; both say the mask is on the apple."""
    result = cso.compare(human(a="apple"), opinion(a="partial"))
    assert result["n_disagreements"] == 0
    assert result["n_severity_only"] == 1


def test_crossing_the_line_is_a_disagreement():
    result = cso.compare(human(a="no_mask"), opinion(a="apple"))
    assert result["n_disagreements"] == 1
    assert result["disagreements"][0]["recorded"] == "no_mask"
    assert result["disagreements"][0]["second_opinion"] == "apple"


def test_the_tail_vocabulary_splits_the_same_way():
    assert cso.classify("arm") == "clean"
    assert cso.classify("table") == cso.classify("mixed") == "not_clean"
    assert cso.classify("undecidable") == "undecided"
    with pytest.raises(cso.CompareError):
        cso.classify("banana")


def test_identical_verdicts_are_counted_but_claim_nothing():
    result = cso.compare(human(a="apple", b="apple"), opinion(a="apple", b="apple"))
    assert result["n_identical"] == 2
    assert result["n_disagreements"] == 0


def test_tiles_only_one_side_saw_are_counted_not_dropped():
    result = cso.compare(human(a="apple", b="apple"), opinion(a="apple", c="apple"))
    assert result["n_tiles_compared"] == 1
    assert result["n_recorded_not_reviewed_blind"] == 1
    assert result["n_reviewed_blind_not_recorded"] == 1


def test_the_artifact_computes_no_accuracy():
    """An accuracy figure would make one correlated observer into ground truth."""
    artifact = cso.build(
        human(a="apple", b="no_mask"), opinion(a="apple", b="apple"), pathlib.Path("x")
    )
    def keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key.lower()
                yield from keys(value)
        elif isinstance(node, list):
            for value in node:
                yield from keys(value)

    # The PROSE may say "no accuracy figure is computed"; no FIELD may be one.
    for banned in ("accuracy", "precision", "recall", "correct_rate", "error_rate", "f1"):
        assert not any(banned in key for key in keys(artifact))
    assert "nothing" in artifact["what_agreement_establishes"].lower()
    assert artifact["not_a_discharge"]


def test_two_groups_claiming_the_same_tile_are_refused(tmp_path):
    """Overlapping assignments would double-count and silently change every number."""
    (tmp_path / "group-1.json").write_text(
        json.dumps({"group": 1, "tiles": [{"key": "e:1", "verdict": "apple"}]})
    )
    (tmp_path / "group-2.json").write_text(
        json.dumps({"group": 2, "tiles": [{"key": "e:1", "verdict": "apple"}]})
    )
    with pytest.raises(cso.CompareError) as excinfo:
        cso.load_opinions(tmp_path)
    assert "judged by two groups" in str(excinfo.value)


def test_no_opinions_at_all_is_refused(tmp_path):
    with pytest.raises(cso.CompareError):
        cso.load_opinions(tmp_path)


def test_a_frame_nobody_looked_at_is_not_compared(tmp_path):
    """`looked_at: false` frames carry `verdict: null` and must never enter the comparison."""
    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "MASK_AUDIT_VERDICTS.json").write_text(
        json.dumps(
            {
                "frames": [
                    {"episode": "e", "frame_index": 1, "looked_at": True, "verdict": "apple"},
                    {"episode": "e", "frame_index": 2, "looked_at": False, "verdict": None},
                ]
            }
        )
    )
    loaded = cso.load_human(audit, tmp_path / "missing")
    assert set(loaded) == {"e:1"}
