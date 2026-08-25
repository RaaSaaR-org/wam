"""Moving verdicts out of the page must not move them onto different frames.

The one property that matters here is the round trip: the per-sheet defaults plus exceptions handed
to ``record_mask_audit_verdicts.py`` must expand back to exactly the per-tile verdicts the page
held. If they do not, a reviewer's judgement lands on a frame they never saw and no artifact
downstream can tell.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ing = _module("_ing", "ingest_review_page.py")
rec = _module("_rec", "record_mask_audit_verdicts.py")


def make_audit(spec: list[tuple[str, int]]) -> dict:
    """``spec`` is ``(stratum, count)`` pairs; frames are numbered within the whole sample."""
    frames = []
    n = 0
    for stratum, count in spec:
        for _ in range(count):
            frames.append(
                {"episode": f"episode_{n // 50:06d}", "frame_index": n, "stratum": stratum}
            )
            n += 1
    return {"frames": frames}


def make_sample(keys: list[str]) -> dict:
    frames = []
    for key in keys:
        episode, _, frame = key.partition(":")
        frames.append(
            {
                "episode": episode,
                "frame_index": int(frame),
                "recorded_fraction": 0.9,
                "recomputed_fraction": 0.9,
                "mismatch": False,
                "sheet": "area-tail-00.png",
            }
        )
    return {"frames": frames, "threshold": {"value": 0.68}}


def test_defaults_plus_exceptions_expand_back_to_every_tile():
    audit = make_audit([("grasp", 24), ("census", 8), ("occluded", 6)])
    keys = [f"{f['episode']}:{f['frame_index']}" for f in audit["frames"]]
    verdicts = dict.fromkeys(keys, "apple")
    verdicts[keys[3]] = "wrong_object"
    verdicts[keys[25]] = "no_mask"
    for key in keys[24:31]:  # the census sheet goes mostly bad, so its DEFAULT flips
        verdicts[key] = "wrong_object"

    defaults, exceptions, _detail = ing.derive_sheets(verdicts, audit, rec)
    mapping = rec.sheet_index(audit["frames"])
    rebuilt = {
        f"{f['episode']}:{f['frame_index']}": defaults[mapping[i]]
        for i, f in enumerate(audit["frames"])
    }
    for item in exceptions:
        where, _, what = item.partition("=")
        rebuilt[where] = what.partition(":")[0]
    assert rebuilt == verdicts


def test_a_sheet_whose_majority_is_bad_defaults_to_bad():
    """Compression must follow the reviewer, not an assumption that masks are usually fine."""
    audit = make_audit([("min_visibility", 12)])
    keys = [f"{f['episode']}:{f['frame_index']}" for f in audit["frames"]]
    defaults, exceptions, detail = ing.derive_sheets(dict.fromkeys(keys, "no_mask"), audit, rec)
    assert defaults == {"min_visibility-00": "no_mask"}
    assert exceptions == []
    assert detail["min_visibility-00"]["tally"] == {"no_mask": 12}


def test_a_tie_resolves_deterministically():
    audit = make_audit([("grasp", 4)])
    keys = [f"{f['episode']}:{f['frame_index']}" for f in audit["frames"]]
    verdicts = {keys[0]: "apple", keys[1]: "apple", keys[2]: "partial", keys[3]: "partial"}
    first = ing.derive_sheets(verdicts, audit, rec)[0]
    second = ing.derive_sheets(dict(reversed(list(verdicts.items()))), audit, rec)[0]
    assert first == second


def test_a_key_neither_sample_names_is_refused():
    audit = make_audit([("grasp", 4)])
    sample = make_sample([])
    state = {"verdicts": {"episode_000999:7": "apple"}}
    with pytest.raises(ing.IngestError) as excinfo:
        ing.split_keys(state, audit, sample)
    assert "neither sample contains" in str(excinfo.value)


def test_the_two_sections_vocabularies_do_not_cross():
    audit = make_audit([("grasp", 2)])
    sample = make_sample(["episode_000000:900"])
    state = {"verdicts": {"episode_000000:0": "table", "episode_000000:900": "mixed"}}
    mask, tail = ing.split_keys(state, audit, sample)
    with pytest.raises(ing.IngestError) as excinfo:
        ing.check_vocabulary(mask, tail, rec)
    assert "not an apple-mask verdict" in str(excinfo.value)

    state = {"verdicts": {"episode_000000:0": "apple", "episode_000000:900": "apple"}}
    mask, tail = ing.split_keys(state, audit, sample)
    with pytest.raises(ing.IngestError) as excinfo:
        ing.check_vocabulary(mask, tail, rec)
    assert "not an area-tail verdict" in str(excinfo.value)


def test_a_frame_in_both_samples_is_refused_rather_than_guessed():
    audit = make_audit([("grasp", 2)])
    sample = make_sample(["episode_000000:0"])
    with pytest.raises(ing.IngestError) as excinfo:
        ing.split_keys({"verdicts": {}}, audit, sample)
    assert "different questions" in str(excinfo.value)


def test_state_is_read_from_the_page_and_from_a_json_copy(tmp_path):
    state = {"verdicts": {"episode_000000:0": "apple"}, "saved_at": "2026-08-26T00:00:00Z"}
    page = tmp_path / "review.html"
    page.write_text(
        '<p>x</p><script type="application/json" id="wam-state">'
        + json.dumps(state)
        + "</script><p>y</p>",
        encoding="utf-8",
    )
    assert ing.read_state(page) == state

    copy = tmp_path / "state.json"
    copy.write_text(json.dumps(state), encoding="utf-8")
    assert ing.read_state(copy) == state


def test_a_page_without_the_state_block_is_refused(tmp_path):
    page = tmp_path / "not-the-page.html"
    page.write_text("<p>nothing here</p>", encoding="utf-8")
    with pytest.raises(ing.IngestError):
        ing.read_state(page)


def test_an_empty_reviewer_is_refused(tmp_path):
    page = tmp_path / "review.html"
    page.write_text(
        '<script type="application/json" id="wam-state">{"verdicts":{}}</script>', encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        ing.main(["--page", str(page), "--reviewer", "   "])


def test_the_tail_artifact_carries_no_bound():
    sample = make_sample(["episode_000004:221", "episode_000032:263"])
    tail = {"episode_000004:221": "mixed", "episode_000032:263": "table"}
    artifact = ing.tail_artifact(
        tail, sample, "human", {"saved_at": "2026-08-26T00:00:00Z"}, pathlib.Path("review.html")
    )
    assert artifact["writes_a_bound"] is False
    assert artifact["tally"] == {"mixed": 1, "table": 1}
    assert artifact["established_by"] == "human"
    # The refusal SENTENCE names the field; no KEY may.
    def keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keys(value)
        elif isinstance(node, list):
            for value in node:
                yield from keys(value)

    assert "max_frame_fraction" not in set(keys(artifact))
