"""``T40_RULE_V13`` §3.1 applied to a distribution, on distributions with KNOWN answers.

These fixtures are synthetic on purpose. The point of the script is to be trustworthy on a
distribution nobody has seen yet, so the tests fix its behaviour on cases where the right answer
is known by construction -- including the case it would be easiest to get wrong, the zero spike.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location(
    "analyze_area_gap", REPO_ROOT / "scripts" / "analyze_area_gap.py"
)
assert spec is not None and spec.loader is not None
gap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gap)


def artifact(per_episode: dict[str, list[float]], **extra) -> dict:
    payload = {
        "git_commit": "0" * 40,
        "source_manifest_sha256": "f" * 64,
        "prompt": "robot arm. robotic hand. robotic gripper.",
        "estimator": {"name": "test"},
        "per_episode": [{"episode": k, "area_fractions": v} for k, v in per_episode.items()],
    }
    payload.update(extra)
    return payload


def test_the_zero_spike_is_not_reported_as_the_gap() -> None:
    """The failure this script exists to avoid.

    A third of real frames carry no robot mask and are recorded as 0.0. The distance from 0.0 to
    the smallest real mask dwarfs every other discontinuity, so a gap-finder over the raw pooled
    population reports it and calls a bound placed in it 'measured'. It is not a gap between two
    populations of MASK; it is the boundary between mask and no-mask.
    """
    fractions = [0.0] * 300 + [0.20 + 0.0001 * i for i in range(700)]
    report = gap.analyze(artifact({"episode_000001": fractions}))

    assert report["population"]["frames_empty_mask"] == 300
    assert report["population"]["frames_nonempty"] == 700
    # The continuum above 0.2 has no discontinuity, so the honest answer is 'no gap'.
    for candidate in report["candidate_gaps"]:
        assert candidate["bulk_edge_below"] > 0.0
        assert candidate["tail_edge_above"] > 0.1
    assert report["candidate_gaps"] == []
    assert "V13 §3.3" in report["verdict"]


def test_a_genuinely_bimodal_population_yields_a_gap_naming_both_edges() -> None:
    """V13 §3.1 step 3: a rationale that cannot name both edges has not found a gap."""
    bulk = [0.10 + 0.0001 * i for i in range(500)]  # ends at 0.1499
    tail = [0.80 + 0.0001 * i for i in range(20)]  # starts at 0.80
    report = gap.analyze(artifact({"episode_000001": bulk + tail}))

    assert report["candidate_gaps"], "a 0.65-wide separation must be found"
    widest = report["candidate_gaps"][0]
    assert widest["bulk_edge_below"] == pytest.approx(0.1499, abs=1e-6)
    assert widest["tail_edge_above"] == pytest.approx(0.80, abs=1e-6)
    assert widest["frames_above"] == 20
    assert "step 3" in report["verdict"]


def test_a_unimodal_continuum_reaches_the_refusal_branch() -> None:
    """V13 §3.3 is 'the load-bearing half' and must be reachable, not theoretical."""
    report = gap.analyze(artifact({"e": [0.05 + 0.0005 * i for i in range(1000)]}))
    assert report["candidate_gaps"] == []
    assert "NO SEPARABLE GAP" in report["verdict"]
    assert "(a) leave max_frame_fraction null" in report["verdict"]


def test_a_single_outlier_is_not_a_tail_population() -> None:
    """One frame above a gap is an outlier. V13 §3.1 step 2 asks for a population."""
    report = gap.analyze(artifact({"e": [0.10 + 0.0001 * i for i in range(500)] + [0.97]}))
    assert report["candidate_gaps"] == []


def test_attribution_separates_a_concentrated_tail_from_a_universal_one() -> None:
    """The check V13 §3.1 step 2 names: is the tail a failure mode, or every episode's grasp?

    Same percentile, opposite meaning -- so the two must not produce the same report.
    """
    bulk = [0.10 + 0.0001 * i for i in range(100)]
    concentrated = artifact({"bad": bulk + [0.9] * 30, **{f"ok{i}": list(bulk) for i in range(9)}})
    universal = artifact({f"ep{i}": bulk + [0.9] * 3 for i in range(10)})

    c = gap.analyze(concentrated)["attribution_for_widest_gap"]
    u = gap.analyze(universal)["attribution_for_widest_gap"]

    assert c["episodes_contributing"] == 1
    assert c["concentration_top_episode"] == pytest.approx(1.0)
    assert u["episodes_contributing"] == 10
    assert u["concentration_top_episode"] == pytest.approx(0.1)
    assert c["frames_above_total"] == u["frames_above_total"] == 30


def test_it_refuses_an_artifact_load_area_bound_would_refuse(tmp_path: pathlib.Path) -> None:
    """A disqualified measurement may not become a bound rationale. V13 §3.4, last bullet."""
    path = tmp_path / "a.json"
    path.write_text(
        json.dumps(
            artifact(
                {"e": [0.1, 0.9]},
                measurement_qualified=False,
                measurement_disqualified_reasons=["a shard was measured with --limit"],
            )
        )
    )
    assert gap.main([str(path)]) == 2


def test_it_never_writes_a_bound() -> None:
    """The script's whole contract: V13 §3 keeps the number a human decision."""
    report = gap.analyze(artifact({"e": [0.1] * 50 + [0.9] * 5}))
    assert report["writes_a_bound"] is False
    assert "max_frame_fraction" not in report
    text = json.dumps(report)
    assert "bound_rationale" not in text
