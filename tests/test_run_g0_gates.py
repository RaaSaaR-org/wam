"""Tests for PR-08 §6's G0a and G0b VOID gates (``scripts/run_g0_gates.py``).

What is pinned here is what can be quietly wrong in a gate: the boundary (a gate is exercised
exactly where it is nearest to changing its answer), every refusal that must NOT degrade into a
default, and the exit codes — because the exit code is how the sbatch and every operator will read
this script, and a runner whose verdict and status disagree is worse than no runner.

The two gates are tested at different altitudes on purpose. **G0a's arithmetic is
``screen_corpus``'s**, driven through its own ``--expect`` machinery, so what is asserted here is
the boundary of the comparison and the coupling to that script — plus one end-to-end run over a
tiny synthetic corpus, which is the only thing that proves the injection actually reaches
``screen_corpus.main``'s CLI. **G0b's refusals are this file's own** and are asserted one by one:
each of them is a place where a wrong default would produce a plausible pass.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


g0 = _load("run_g0_gates")
screen_corpus = _load("screen_corpus")


@pytest.fixture(autouse=True)
def _restore_archived():
    """``register_source_reference`` mutates ``screen_corpus.ARCHIVED`` by design; undo it.

    The injection is per-process and deliberate (the source's triple is a measurement, never a
    committed constant), but a test that left it behind would let one test's reference decide
    another test's verdict.
    """
    saved = dict(screen_corpus.ARCHIVED)
    yield
    screen_corpus.ARCHIVED.clear()
    screen_corpus.ARCHIVED.update(saved)


SOURCE_METRICS = {"m1": 0.660, "m2": 0.333, "m3": 2.01}


def _report(metrics: dict[str, float], **extra: Any) -> dict[str, Any]:
    """A minimal ``screen_corpus`` artifact — the fields G0a reads out of one."""
    report = {
        "m1_momentum_share": metrics["m1"],
        "m2_blind_unreachable": metrics["m2"],
        "m3_transitions_per_episode": metrics["m3"],
        "ceiling_dominates": True,
        "episodes": {"train": 362, "holdout": 40},
        "holdout_file": "configs/splits/t18_holdout_episodes.txt",
    }
    report.update(extra)
    return report


# ------------------------------------------------------------------- G0a: the identity check


def test_a_restyle_that_reproduces_the_sources_metrics_exactly_passes_g0a() -> None:
    """The whole point of G0a: a restyle changes no action, so the numbers must be the source's."""
    deltas = g0.expect_deltas(screen_corpus, SOURCE_METRICS, dict(SOURCE_METRICS))
    assert g0.deltas_verdict(deltas) == "PASS"
    assert all(row["abs_delta"] == 0.0 for row in deltas)


#: The boundary is asserted from a ZERO baseline, so that ``ref + tol`` is exactly representable
#: and ``restyled - source`` is exactly ``tol``. Off a baseline like 0.660 it is not: 0.02 has no
#: exact binary form, ``0.660 + 0.02 - 0.660`` lands a few ULPs above 0.02, and the test would be
#: asserting the rounding of the fixture rather than the operator in the gate.
ZERO_METRICS = {"m1": 0.0, "m2": 0.0, "m3": 0.0}


@pytest.mark.parametrize("metric", ["m1", "m2", "m3"])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_a_deviation_of_exactly_expect_tol_still_reproduces_on_each_metric(
    metric: str, sign: float
) -> None:
    """The boundary is inclusive, exactly as ``screen_corpus.main`` reads it (``<= tol``).

    Pinned per metric because the three tolerances differ (0.02 / 0.02 / 0.05) and a runner that
    applied one of them to all three would pass this test on m1 and gate m3 more than twice as
    tightly as PR-08 §6 says.
    """
    tol = screen_corpus.EXPECT_TOL[metric]
    restyled = dict(ZERO_METRICS)
    restyled[metric] = sign * tol
    deltas = g0.expect_deltas(screen_corpus, ZERO_METRICS, restyled)
    row = next(r for r in deltas if r["metric"] == metric)
    assert row["abs_delta"] == tol and row["tol"] == tol
    assert g0.deltas_verdict(deltas) == "PASS"


@pytest.mark.parametrize("metric", ["m1", "m2", "m3"])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_a_deviation_one_ulp_past_expect_tol_is_void_on_each_metric(
    metric: str, sign: float
) -> None:
    """One representable step outside, in either direction, is VOID.

    One ULP rather than a comfortable margin because a gate is most likely to be exercised exactly
    where it changes its answer, and ``<`` vs ``<=`` is invisible anywhere else.
    """
    tol = screen_corpus.EXPECT_TOL[metric]
    restyled = dict(ZERO_METRICS)
    restyled[metric] = sign * np.nextafter(tol, np.inf)
    deltas = g0.expect_deltas(screen_corpus, ZERO_METRICS, restyled)
    assert not next(r for r in deltas if r["metric"] == metric)["within_tol"]
    assert g0.deltas_verdict(deltas) == "VOID"


def test_a_deviation_far_past_expect_tol_is_void_on_a_realistic_baseline() -> None:
    """The same reading off the archived GR00T triple, so the boundary tests' zero baseline is
    not the only shape the comparison is ever seen in."""
    restyled = dict(SOURCE_METRICS)
    restyled["m3"] = SOURCE_METRICS["m3"] + 0.5
    assert g0.deltas_verdict(g0.expect_deltas(screen_corpus, SOURCE_METRICS, restyled)) == "VOID"


def test_g0a_carries_no_copy_of_the_tolerances_and_reads_screen_corpus_own() -> None:
    """PR-08 §6 calls EXPECT_TOL "the script's own archived tolerances". There is one copy."""
    deltas = g0.expect_deltas(screen_corpus, SOURCE_METRICS, dict(SOURCE_METRICS))
    assert {r["metric"]: r["tol"] for r in deltas} == {
        k: screen_corpus.EXPECT_TOL[k] for k in ("m1", "m2", "m3")
    }
    source = (_REPO_ROOT / "scripts" / "run_g0_gates.py").read_text()
    assert "EXPECT_TOL = {" not in source, "the tolerances must be read, never re-declared"


def test_the_source_reference_is_registered_where_screen_corpus_looks_it_up() -> None:
    """G0a works by pointing ``--expect`` at the SOURCE instead of the archived gr00t triple."""
    key = g0.register_source_reference(screen_corpus, SOURCE_METRICS)
    assert screen_corpus.ARCHIVED[key] == {"m1": 0.660, "m2": 0.333, "m3": 2.01}
    assert key in sorted(screen_corpus.ARCHIVED)  # what main() builds --expect's choices from


def test_a_screen_corpus_that_stopped_reading_archived_at_call_time_is_refused() -> None:
    """If the coupling ever breaks, argparse would reject the key or fall back to 'gr00t'.

    Both failures are silent in the artifact, so the coupling is asserted rather than assumed.
    """

    class Fake:
        ARCHIVED: dict[str, Any] = {}
        EXPECT_TOL = dict(screen_corpus.EXPECT_TOL)

        @staticmethod
        def main(argv: list[str] | None = None) -> int:  # no ARCHIVED lookup at all
            return 0

    with pytest.raises(g0.GateRefusal, match="ARCHIVED"):
        g0.assert_expect_machinery_is_live(Fake)


def test_a_disagreement_between_the_two_g0a_readings_refuses_rather_than_picking_one() -> None:
    """``screen_corpus``'s exit status and the recorded deltas cannot disagree unless this file
    has drifted from the script it drives. Reporting either answer would be choosing silently."""
    moved = {"m1": 0.9, "m2": 0.333, "m3": 2.01}
    with pytest.raises(g0.GateRefusal, match="disagree"):
        g0.g0a_record(screen_corpus, _report(SOURCE_METRICS), _report(moved), 0, "pr08-source")


def test_g0a_records_the_deltas_and_not_only_the_boolean() -> None:
    """"the labels moved by 0.003" and "by 0.9" are the same boolean and different findings."""
    moved = {"m1": 0.9, "m2": 0.333, "m3": 2.01}
    record = g0.g0a_record(screen_corpus, _report(SOURCE_METRICS), _report(moved), 1, "pr08-source")
    assert record["verdict"] == "VOID"
    row = next(r for r in record["deltas"] if r["metric"] == "m1")
    assert row["delta"] == pytest.approx(0.24)


def test_a_screen_whose_ceiling_is_beaten_by_a_zero_parameter_rule_refuses() -> None:
    """screen_corpus's own G4: M1 and M2 are VOID there, and an identity check between two VOID
    numbers is not evidence about the labels in either direction."""
    with pytest.raises(g0.GateRefusal, match="ceiling_dominates"):
        g0.g0a_record(
            screen_corpus,
            _report(SOURCE_METRICS, ceiling_dominates=False),
            _report(SOURCE_METRICS),
            0,
            "pr08-source",
        )


def test_two_screens_over_different_episode_counts_refuse_rather_than_reporting_a_verdict() -> None:
    """A restyle emits one clip per source clip. Different counts are two corpora, not a verdict."""
    record = g0.g0a_record(
        screen_corpus,
        _report(SOURCE_METRICS),
        _report(SOURCE_METRICS, episodes={"train": 300, "holdout": 40}),
        0,
        "pr08-source",
    )
    assert record["verdict"] == "REFUSED"
    assert record["measured_verdict"] == "PASS"
    assert record["structural_mismatch"]
    assert "not evidence about the labels" in record["refusal"]


def test_a_deviation_measured_between_two_different_corpora_is_not_reported_as_a_void() -> None:
    """The failure this refusal exists for, and the one a demote-only-a-PASS rule could not catch.

    When the episode counts differ AND the metrics deviate, the deviation has two explanations —
    the labels moved, or the two screens were over different corpora — and a VOID under T40_RULE_V1
    is a formal indictment of the generation pipeline. The runner's own comment already says a
    structural mismatch means "this is not the check §6 asks for"; that reasoning does not stop
    applying because the numbers happen to disagree.
    """
    moved = dict(SOURCE_METRICS, m3=SOURCE_METRICS["m3"] + 5.0)
    record = g0.g0a_record(
        screen_corpus,
        _report(SOURCE_METRICS),
        _report(moved, episodes={"train": 300, "holdout": 40}),
        1,
        "pr08-source",
    )
    assert record["verdict"] == "REFUSED"
    assert record["measured_verdict"] == "VOID"
    assert "is NOT reported as G0a's verdict" in record["refusal"]


# ------------------------------------------------------- G0b: the derived budget and its refusals


DEFAULT_CONTRACT = {
    "method_name": "grounding-dino+sam2+depth-anything-v2",
    "detector": {"repo": "IDEA-Research/grounding-dino-base", "revision": "12bdfa3"},
    "object_text_prompt": "apple.",
    "box_threshold": 0.15,
    "pixel_grid_hw": [480, 640],
}


def _geom_config(
    tmp_path: Path,
    *,
    geom_tol: float | None = 4.0,
    drift: float | None = 1.0,
    margin: Any = "derive",
    contract: dict[str, Any] | None = None,
    sources: bool = True,
    sidecar: bool | str = True,
    name: str = "grounding-dino+sam2+depth-anything-v2",
    step_frames: Any = 1,
    **extra: Any,
) -> Path:
    """A committed tolerance artifact in the schema ``configs/transfer25/pr08_geom_tol.json`` uses.

    ``step_frames`` is written by default because G0b asserts it on every run: the gate compares
    source frame i to restyled frame i, so a tolerance measured at any other step is a budget of
    the wrong width. Pass ``step_frames=None`` for the artifact that does not state it.
    """
    block = dict(contract if contract is not None else DEFAULT_CONTRACT)
    block["method_name"] = name
    doc: dict[str, Any] = {
        "spec_version": "1.0.0",
        "what_this_is": "test fixture",
        "segmenter": block,
        "geom_tol_px": geom_tol,
        "geom_tol_source": "runs/pr08-geom-tol/geom_tol.json" if sources else None,
        "est_drift_p95_px": drift,
        "est_drift_source": "runs/pr08-est-drift/est_drift.json" if sources else None,
        "gate_margin_px": (
            (None if (geom_tol is None or drift is None) else geom_tol - drift)
            if margin == "derive"
            else margin
        ),
    }
    if step_frames is not None:
        doc["step_frames"] = step_frames
    doc.update(extra)
    path = tmp_path / "pr08_geom_tol.json"
    path.write_text(json.dumps(doc, indent=2))
    # measure_geom_tol.py writes this sidecar beside every artifact, and the generation sbatch
    # refuses one without it. The fixture writes it by default so that the sidecar check is not
    # silently disqualifying every other assertion in this file.
    if sidecar:
        import hashlib

        digest = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if sidecar is True
            else str(sidecar)
        )
        (tmp_path / "pr08_geom_tol.json.sha256").write_text(digest + "\n")
    return path


def _centroids(
    tmp_path: Path,
    side: str,
    clips: dict[str, Any] | None = None,
    *,
    name: str = "grounding-dino+sam2+depth-anything-v2",
    version: str = "det=gd@12bdfa3;seg=sam2@e6a8e88",
    grid: tuple[int, int] = (480, 640),
    gate_qualified: bool = True,
    contract: dict[str, Any] | None = DEFAULT_CONTRACT,
) -> Path:
    if clips is None:
        clips = {
            "episode_000000": {
                "object": [[100.0, 100.0], [101.0, 100.0]],
                "plate": [[300.0, 200.0], [300.0, 200.0]],
            }
        }
    segmenter: dict[str, Any] = {
        "name": name,
        "version": version,
        "gate_qualified": gate_qualified,
    }
    if contract is not None:
        segmenter["contract"] = contract
    path = tmp_path / f"centroids-{side}.json"
    path.write_text(
        json.dumps(
            {
                "schema": g0.CENTROID_SCHEMA,
                "side": side,
                "segmenter": segmenter,
                "resolution_hw": list(grid),
                "clips": clips,
            },
            indent=2,
        )
    )
    return path


def _shift(clips: dict[str, Any], label: str, frame: int, dx: float) -> dict[str, Any]:
    """The restyled side with one centroid moved — the only thing G0b is about."""
    out = json.loads(json.dumps(clips))
    out["episode_000000"][label][frame][0] += dx
    return out


SOURCE_CLIPS = {
    "episode_000000": {
        "object": [[100.0, 100.0], [101.0, 100.0], [102.0, 100.0]],
        "plate": [[300.0, 200.0], [300.0, 200.0], [300.0, 200.0]],
    }
}


def _args(tmp_path: Path, geom: Path, source: Path, restyled: Path, *extra: str) -> Any:
    return g0.parse_args(
        [
            "--gates", "g0b",
            "--geom-config", str(geom),
            "--source-centroids", str(source),
            "--restyled-centroids", str(restyled),
            "--out", str(tmp_path / "g0.json"),
            *extra,
        ]
    )


def _run(tmp_path: Path, *argv: str) -> tuple[int, dict[str, Any]]:
    """Drive ``main`` exactly as the sbatch would, and read back the artifact it wrote."""
    out = tmp_path / "g0.json"
    code = g0.main([*argv, "--out", str(out)])
    return code, json.loads(out.read_text())


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"geom_tol": None}, "geom_tol_px = null"),
        ({"drift": None}, "est_drift_p95_px = null"),
    ],
)
def test_g0b_refuses_when_either_half_of_the_budget_is_null(
    tmp_path: Path, kwargs: dict[str, Any], needle: str
) -> None:
    """A null term is a measurement that has not been made. The gate does not assume zero, and it
    does not warn and continue: an assumed-zero drift budget hands the generator the WIDEST
    possible tolerance, produced by the absence of a measurement."""
    doc = json.loads(_geom_config(tmp_path, **kwargs).read_text())
    with pytest.raises(g0.GateRefusal) as exc:
        g0.gate_budget(doc, tmp_path / "pr08_geom_tol.json")
    assert needle in str(exc.value)
    assert "does NOT assume zero" in str(exc.value) or "does not assume one" in str(exc.value)


@pytest.mark.parametrize("geom_tol,drift", [(1.0, 1.0), (0.5, 2.0)])
def test_g0b_refuses_a_non_positive_margin_and_quotes_the_rule(
    tmp_path: Path, geom_tol: float, drift: float
) -> None:
    """PR-08 §6, in the message rather than paraphrased: below zero the estimator is the problem
    and generation does not start. Widening the gate is not available from here."""
    doc = json.loads(_geom_config(tmp_path, geom_tol=geom_tol, drift=drift).read_text())
    with pytest.raises(g0.GateRefusal) as exc:
        g0.gate_budget(doc, tmp_path / "pr08_geom_tol.json")
    assert g0.NON_POSITIVE_MARGIN_QUOTE in str(exc.value)


def test_g0b_runs_on_the_smallest_positive_margin(tmp_path: Path) -> None:
    """The other side of the same boundary: > 0 is a budget, however small."""
    doc = json.loads(_geom_config(tmp_path, geom_tol=1.0, drift=1.0 - 1e-9).read_text())
    budget = g0.gate_budget(doc, tmp_path / "pr08_geom_tol.json")
    assert budget["gate_margin_px"] > 0.0


def test_a_config_whose_stated_margin_disagrees_with_its_own_subtraction_is_refused(
    tmp_path: Path,
) -> None:
    """``gate_margin_px`` is checked, not trusted. An artifact that disagrees with its own
    arithmetic cannot be quoted for either half of it."""
    doc = json.loads(_geom_config(tmp_path, geom_tol=4.0, drift=1.0, margin=9.0).read_text())
    with pytest.raises(g0.GateRefusal, match="disagrees with its own arithmetic"):
        g0.gate_budget(doc, tmp_path / "pr08_geom_tol.json")


def test_a_tolerance_stating_the_same_number_under_two_disagreeing_spellings_is_refused(
    tmp_path: Path,
) -> None:
    """``configs/transfer25/pr08_geom_tol.json`` has TWO producers writing two schemas into it —
    the committed PR-08 gate schema (``geom_tol_px``) and ``measure_geom_tol.py``'s default
    ``--out`` (``GEOM_TOL_px``). Accepting both spellings is necessary; choosing between them by
    key order is not. A stale 4.0 quoted over a freshly measured 9.0 is a gate that is 5 px tighter
    than anyone decided, with nothing but a ``geom_tol_key`` field to show for it.
    """
    doc = json.loads(_geom_config(tmp_path, geom_tol=4.0, drift=1.0, GEOM_TOL_px=9.0).read_text())
    with pytest.raises(g0.GateRefusal, match="two spellings that disagree"):
        g0.gate_budget(doc, tmp_path / "pr08_geom_tol.json")


def test_two_spellings_that_agree_are_not_a_disagreement(tmp_path: Path) -> None:
    """One producer writing both names for the same measured number has contradicted nothing, and
    refusing it would make the artifact this repository actually produces ungateable."""
    doc = json.loads(_geom_config(tmp_path, geom_tol=4.0, drift=1.0, GEOM_TOL_px=4.0).read_text())
    budget = g0.gate_budget(doc, tmp_path / "pr08_geom_tol.json")
    assert budget["gate_margin_px"] == pytest.approx(3.0)


def test_a_tolerance_that_names_no_segmenter_is_refused(tmp_path: Path) -> None:
    """§4 step 2's "the same segmenter" is uncheckable against a tolerance that never said which."""
    doc = json.loads(_geom_config(tmp_path).read_text())
    doc.pop("segmenter")
    with pytest.raises(g0.GateRefusal, match="names no segmenter"):
        g0.config_instrument(doc, tmp_path / "pr08_geom_tol.json")


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"name": "some-other-segmenter"}, "different segmenters"),
        ({"version": "det=gd@OTHER;seg=sam2@e6a8e88"}, "different segmenter VERSIONS"),
        ({"grid": (240, 320)}, "different pixel grids"),
    ],
)
def test_g0b_refuses_when_the_two_sides_used_different_instruments(
    tmp_path: Path, kwargs: dict[str, Any], needle: str
) -> None:
    """Verified FROM the records. Centroids from two segmenters, or on two grids, differ by a
    plausible number of pixels that is not a geometry measurement."""
    geom = _geom_config(tmp_path)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(
        tmp_path,
        "restyled",
        SOURCE_CLIPS,
        contract=None,  # the contract block would fire its own disagreement first
        **kwargs,
    )
    with pytest.raises(g0.GateRefusal) as exc:
        g0.run_g0b(_args(tmp_path, geom, source, restyled))
    assert needle in str(exc.value)


def test_g0b_refuses_a_side_whose_pinned_operating_point_moved(tmp_path: Path) -> None:
    """Two runs can share a segmenter NAME while disagreeing about every number under it — the
    committed contract's own words. A moved box threshold is a different segmenter."""
    geom = _geom_config(tmp_path)
    moved = dict(DEFAULT_CONTRACT, box_threshold=0.35)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS, contract=moved)
    with pytest.raises(g0.GateRefusal, match="box_threshold"):
        g0.run_g0b(_args(tmp_path, geom, source, restyled))


def test_g0b_refuses_centroids_measured_on_a_grid_the_tolerance_was_not(tmp_path: Path) -> None:
    """The budget is in pixels at 640x480; a displacement measured elsewhere is another unit."""
    geom = _geom_config(tmp_path)
    small = dict(DEFAULT_CONTRACT, pixel_grid_hw=[240, 320])
    source = _centroids(tmp_path, "source", SOURCE_CLIPS, grid=(240, 320), contract=small)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS, grid=(240, 320), contract=small)
    with pytest.raises(g0.GateRefusal, match="pixel budget on one grid"):
        g0.run_g0b(_args(tmp_path, geom, source, restyled))


def test_g0b_refuses_a_clip_whose_frame_count_changed(tmp_path: Path) -> None:
    """A restyle emits one frame per source frame. A different count is proof frames were dropped,
    duplicated or reordered, and index-by-index comparison after that compares different moments."""
    geom = _geom_config(tmp_path)
    short = json.loads(json.dumps(SOURCE_CLIPS))
    short["episode_000000"]["object"] = short["episode_000000"]["object"][:-1]
    short["episode_000000"]["plate"] = short["episode_000000"]["plate"][:-1]
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", short)
    with pytest.raises(g0.GateRefusal, match="dropped, duplicated or reordered"):
        g0.run_g0b(_args(tmp_path, geom, source, restyled))


def test_g0b_refuses_a_restyled_clip_with_no_source_clip_to_compare_against(
    tmp_path: Path,
) -> None:
    orphan = {"episode_999999__styleA": dict(SOURCE_CLIPS["episode_000000"])}
    geom = _geom_config(tmp_path)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", orphan)
    with pytest.raises(g0.GateRefusal, match="name a source clip"):
        g0.run_g0b(_args(tmp_path, geom, source, restyled))


def test_a_restyle_that_moves_nothing_passes_g0b_and_records_the_distribution(
    tmp_path: Path,
) -> None:
    """The healthy case — and the artifact carries the whole margin distribution, per clip, not a
    boolean: a gate that clears at the median and fails at the p99 is a different fact."""
    geom = _geom_config(tmp_path)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", _shift(SOURCE_CLIPS, "object", 1, 0.25))
    record = g0.run_g0b(_args(tmp_path, geom, source, restyled))
    assert record["verdict"] == "PASS"
    clip = record["per_clip"][0]["labels"]["object"]
    assert clip["margin_px"]["percentiles"]["p50"] == pytest.approx(3.0)
    assert clip["margin_px"]["min"] == pytest.approx(2.75)
    assert record["by_label"]["object"]["margin_distribution"]["n"] == 3
    # The distribution ACROSS clips is over the worst margin of each clip/label: the object's
    # 2.75 px and the near-static plate's full 3.0 px.
    assert record["worst_margin_across_clips_px"]["p0"] == pytest.approx(2.75)
    assert record["worst_margin_across_clips_px"]["p100"] == pytest.approx(3.0)


def test_a_restyle_that_moves_the_object_past_the_budget_is_void(tmp_path: Path) -> None:
    """The gate's whole purpose: the carried-over label now describes a different scene than the
    pixels, so the training pair is a lie and PR-08 §6's word for that is VOID."""
    geom = _geom_config(tmp_path)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", _shift(SOURCE_CLIPS, "object", 1, 9.0))
    record = g0.run_g0b(_args(tmp_path, geom, source, restyled))
    assert record["verdict"] == "VOID"
    assert record["void_rows"] and "9.000 px > budget 3.000 px" in record["void_rows"][0]
    assert record["per_clip"][0]["labels"]["object"]["frames_outside_budget"] == 1


def test_a_displacement_exactly_at_the_budget_still_passes(tmp_path: Path) -> None:
    """``<=``, as everywhere else in this repository's gates, and asserted at the boundary."""
    geom = _geom_config(tmp_path)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", _shift(SOURCE_CLIPS, "object", 1, 3.0))
    record = g0.run_g0b(_args(tmp_path, geom, source, restyled))
    assert record["verdict"] == "PASS"
    assert record["by_label"]["object"]["worst_margin_px"] == pytest.approx(0.0)


def test_the_plate_half_of_g0b_never_running_is_not_a_pass(tmp_path: Path) -> None:
    """§6 gates object AND plate. Half a gate reporting PASS is the failure this document is
    written against, so it reports NOT_GATE_QUALIFIED and exit 3 instead."""
    object_only = {"episode_000000": {"object": SOURCE_CLIPS["episode_000000"]["object"]}}
    geom = _geom_config(tmp_path)
    source = _centroids(tmp_path, "source", object_only)
    restyled = _centroids(tmp_path, "restyled", object_only)
    record = g0.run_g0b(_args(tmp_path, geom, source, restyled))
    assert record["verdict"] == "NOT_GATE_QUALIFIED"
    assert record["measured_verdict"] == "PASS"
    assert any("plate" in r for r in record["not_gate_qualified_reasons"])


def test_a_frame_where_either_side_lost_the_object_is_dropped_and_counted(tmp_path: Path) -> None:
    """Never folded in as zero displacement: a zero would pull every statistic down exactly where
    the gate saw nothing, which looks conservative and is backwards."""
    occluded = json.loads(json.dumps(SOURCE_CLIPS))
    occluded["episode_000000"]["object"][1] = None
    geom = _geom_config(tmp_path)
    source = _centroids(tmp_path, "source", occluded)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)
    record = g0.run_g0b(_args(tmp_path, geom, source, restyled))
    assert record["by_label"]["object"]["n_dropped_object_not_visible"] == 1
    assert record["by_label"]["object"]["n_measured"] == 2
    assert record["coverage"] < 1.0


def test_a_side_whose_segmenter_does_not_claim_gate_qualification_cannot_stand_as_the_gate(
    tmp_path: Path,
) -> None:
    """An unstated claim is not a claim — the rule measure_geom_tol applies to its own inputs."""
    geom = _geom_config(tmp_path)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS, gate_qualified=False)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)
    record = g0.run_g0b(_args(tmp_path, geom, source, restyled))
    assert record["verdict"] == "NOT_GATE_QUALIFIED"


def test_a_tolerance_with_no_stated_measurement_behind_it_cannot_stand_as_the_gate(
    tmp_path: Path,
) -> None:
    """``geom_tol_source``/``est_drift_source`` null: a number with no measurement behind it."""
    geom = _geom_config(tmp_path, sources=False)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)
    record = g0.run_g0b(_args(tmp_path, geom, source, restyled))
    assert record["verdict"] == "NOT_GATE_QUALIFIED"
    assert any("geom_tol_source" in r for r in record["not_gate_qualified_reasons"])


def test_a_tolerance_edited_after_it_was_committed_is_refused(tmp_path: Path) -> None:
    """The one move that slips a hand-written tolerance past a checked digest is editing the
    artifact and leaving the sidecar alone. Re-measure, do not re-hash."""
    geom = _geom_config(tmp_path, sidecar="0" * 64)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)
    with pytest.raises(g0.GateRefusal, match="committed .sha256 sidecar"):
        g0.run_g0b(_args(tmp_path, geom, source, restyled))


def test_a_tolerance_with_no_committed_digest_cannot_stand_as_the_gate(tmp_path: Path) -> None:
    """Softer than the sbatch, which refuses outright — but not silent, which is the failure."""
    geom = _geom_config(tmp_path, sidecar=False)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)
    record = g0.run_g0b(_args(tmp_path, geom, source, restyled))
    assert record["verdict"] == "NOT_GATE_QUALIFIED"
    assert record["tolerance_digest"]["verified"] is None


def test_a_partial_comparison_cannot_stand_as_the_gate(tmp_path: Path) -> None:
    """``--limit`` is a smoke test. Coverage is computed over the frames that WERE compared and
    cannot notice that most of the corpus was not."""
    geom = _geom_config(tmp_path)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)
    record = g0.run_g0b(_args(tmp_path, geom, source, restyled, "--limit", "1"))
    assert record["verdict"] == "NOT_GATE_QUALIFIED"


# ------------------------------------------- the margin distribution, which is a SIGNED quantity


def test_a_margin_distribution_containing_negative_margins_actually_shows_them() -> None:
    """The histogram exists to answer "how far outside the budget", and a negative margin IS
    outside the budget. Binned from zero upward — which is right for displacements and wrong for
    margins — every failing frame falls outside the bins and the counts say the corpus was clean.
    """
    dist = g0.signed_distribution(np.asarray([-7.0, -7.0, -7.0]), 0.5)
    assert dist["n"] == 3
    assert sum(dist["histogram"]["counts"]) == 3
    assert min(dist["histogram"]["bin_edges_px"]) <= -7.0
    assert dist["min_px"] == pytest.approx(-7.0)


@pytest.mark.parametrize(
    "values",
    [
        [-7.0, -7.0, -7.0],          # every frame outside: the all-negative case
        [-3.25, 0.0, 4.5],           # straddling zero
        [2.0, 2.0],                  # every value identical and on a bin edge
        [-0.1],                      # one value, inside one bin
    ],
)
def test_the_signed_histogram_counts_every_value_it_reports_an_n_for(values: list[float]) -> None:
    """A counts array that does not sum to its own ``n`` is the wrong number that gets quoted."""
    dist = g0.signed_distribution(np.asarray(values), 0.5)
    assert sum(dist["histogram"]["counts"]) == dist["n"] == len(values)


def test_the_failing_tail_is_visible_in_the_artifact_of_a_void_run(tmp_path: Path) -> None:
    """End to end: the VOID case is exactly the case the margin distribution exists for."""
    geom = _geom_config(tmp_path)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", _shift(SOURCE_CLIPS, "object", 1, 9.0))
    record = g0.run_g0b(_args(tmp_path, geom, source, restyled))
    assert record["verdict"] == "VOID"
    dist = record["by_label"]["object"]["margin_distribution"]
    assert dist["min_px"] == pytest.approx(-6.0)
    assert sum(dist["histogram"]["counts"]) == dist["n"] == 3
    # the frame that blew the budget is IN a bin, not dropped on the floor
    negative_bins = [
        count
        for count, edge in zip(dist["histogram"]["counts"], dist["histogram"]["bin_edges_px"])
        if edge < 0
    ]
    assert sum(negative_bins) == 1


def test_the_pooled_percentile_is_not_named_like_the_statistic_that_gates(tmp_path: Path) -> None:
    """Two quantities, one name, free to drift: the per-clip percentile decides the verdict and the
    pooled one never gates anything. At p100 they coincide; at any lower percentile they need not,
    because percentiles do not compose."""
    geom = _geom_config(tmp_path)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", _shift(SOURCE_CLIPS, "object", 1, 0.5))
    record = g0.run_g0b(_args(tmp_path, geom, source, restyled))
    assert "gate_statistic_px" not in record["by_label"]["object"]
    assert record["by_label"]["object"]["pooled_statistic_px"] == pytest.approx(0.5)
    assert record["per_clip"][0]["labels"]["object"]["gate_statistic_px"] == pytest.approx(0.5)
    assert "PER CLIP" in record["criterion"]["statistic"]


# ------------------------------- the producer's consumer_asserts, honoured by the consumer


#: The leading tokens of ``scripts/measure_geom_tol.py``'s own ``consumer_asserts`` entries. The
#: prose is elided; dispatch is on the token, so that is what has to be right.
#:
#: THIS IS A COPY AND A COPY GOES STALE. It is a fixture — it drives the handlers, it does not
#: prove the producer writes these and only these. That is
#: ``test_every_entry_the_producer_actually_writes_dispatches_to_a_handler_here``, which parses the
#: list out of the producer's source; the entry this copy was missing on 2026-08-22 (the segmenter
#: block) would have refused every real G0b run and nothing here would have noticed.
PRODUCER_ASSERTS = [
    "mask_method.name == pr08_est_drift.json estimators.name — the SAME segmenter",
    "resolution_hw == the [height, width] EST_DRIFT_P95 was measured at",
    "gate_qualified == true",
    "partial_measurement == false",
    "n_episodes == n_episodes_found",
    "step_frames == the step the consumer intends to gate under",
    "coverage >= min_coverage",
    "sha256sum <artifact> matches <artifact>.sha256",
]

MEASURED_FIELDS: dict[str, Any] = {
    "consumer_asserts": PRODUCER_ASSERTS,
    "gate_qualified": True,
    "partial_measurement": False,
    "n_episodes": 402,
    "n_episodes_found": 402,
    "coverage": 0.97,
    "min_coverage": 0.90,
    # The spelling `scripts/measure_geom_tol.py --carry-est-drift` actually writes. This fixture
    # used to say `estimators: {"name": ...}` — a key no producer in this repository writes, only
    # measure_est_drift's OWN artifact does — so every test here passed against a document shaped
    # like nothing that can exist, while every real G0b run lost its qualification on the one
    # assertion this fixture was inventing an answer for.
    "est_drift_estimator_name": "grounding-dino+sam2+depth-anything-v2",
}

#: The same fixture as it looks BEFORE the budget is carried in: the tolerance is measured, the
#: drift half is not, and the name slot is empty because the number it names is.
MEASURED_FIELDS_WITHOUT_THE_NAME = {
    k: v for k, v in MEASURED_FIELDS.items() if k != "est_drift_estimator_name"
}


def test_every_entry_of_the_producers_consumer_asserts_list_is_checked(tmp_path: Path) -> None:
    """measure_geom_tol writes that checklist INTO the artifact so the two sides cannot drift in
    prose. A consumer that reads the tolerance and not the checklist is the drift."""
    geom = _geom_config(tmp_path, **MEASURED_FIELDS)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)
    record = g0.run_g0b(_args(tmp_path, geom, source, restyled))
    block = record["tolerance_consumer_asserts"]
    assert block["declared_by_artifact"] == len(PRODUCER_ASSERTS)
    assert all(row["checked"] is True for row in block["checked"]), block["checked"]
    assert record["verdict"] == "PASS"


def test_a_consumer_assertion_this_runner_has_no_handler_for_refuses(tmp_path: Path) -> None:
    """The point of the table: the next assertion the producer grows is implemented here or it
    stops this runner. The one outcome not available is ignoring it silently."""
    geom = _geom_config(
        tmp_path,
        **dict(MEASURED_FIELDS, consumer_asserts=PRODUCER_ASSERTS + ["depth_units == metres"]),
    )
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)
    with pytest.raises(g0.GateRefusal, match="no handler for"):
        g0.run_g0b(_args(tmp_path, geom, source, restyled))


@pytest.mark.parametrize(
    "override,needle",
    [
        # gate_qualified=false is caught by run_g0b's own earlier refusal, which quotes the
        # measurement's recorded reasons; the handler is the belt to that braces.
        ({"gate_qualified": False}, "not usable as a gate"),
        ({"partial_measurement": True}, "partial_measurement = True"),
        ({"n_episodes": 3}, "measured 3 of 402 episodes"),
        ({"coverage": 0.5}, "coverage 0.500 < min_coverage"),
        ({"est_drift_estimator_name": "some-other-segmenter"}, "name two segmenters"),
    ],
)
def test_a_declared_consumer_assertion_that_does_not_hold_refuses(
    tmp_path: Path, override: dict[str, Any], needle: str
) -> None:
    """Each of these is a condition the producer said must be true before its number is quoted."""
    geom = _geom_config(tmp_path, **dict(MEASURED_FIELDS, **override))
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)
    with pytest.raises(g0.GateRefusal) as exc:
        g0.run_g0b(_args(tmp_path, geom, source, restyled))
    assert needle in str(exc.value)


def test_a_declared_assertion_whose_own_field_the_artifact_omits_refuses(tmp_path: Path) -> None:
    """An entry whose field is missing can never be satisfied, and passing it by saying nothing is
    the default-permissiveness the list exists against."""
    fields = dict(MEASURED_FIELDS)
    fields.pop("partial_measurement")
    geom = _geom_config(tmp_path, **fields)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)
    with pytest.raises(g0.GateRefusal, match="does not carry 'partial_measurement'"):
        g0.run_g0b(_args(tmp_path, geom, source, restyled))


def test_a_tolerance_assembled_by_the_producer_can_actually_reach_a_pass(tmp_path: Path) -> None:
    """THE END-TO-END PROPERTY, and the one this file could not see.

    Every other test here builds the tolerance document by hand, so a fixture that invents a field
    no producer writes makes the gate look reachable while no real artifact can reach it. This one
    starts from a document with EST_DRIFT_P95 still null, carries the budget across with
    ``scripts/measure_geom_tol.py --carry-est-drift`` — the only thing in this repository that
    writes those slots — and then gates on what that wrote. Before 2026-08-22 nothing wrote
    ``est_drift_estimator_name`` and this run could only ever be NOT_GATE_QUALIFIED."""
    mgt = _load("measure_geom_tol")
    geom = _geom_config(tmp_path, drift=None, **MEASURED_FIELDS_WITHOUT_THE_NAME)
    est = tmp_path / "pr08_est_drift.json"
    est.write_text(json.dumps({
        "schema": "wam.est_drift/1",
        "gate_qualified": True,
        "est_drift_p95_px": 1.0,
        "is_lower_bound": True,
        "measured_utc": "2026-08-22T00:00:00+00:00",
        "resolution_hw": [480, 640],
        "estimators": {"name": "grounding-dino+sam2+depth-anything-v2"},
        "geom_tol_cross_check": {"this_segmenter_contract": dict(DEFAULT_CONTRACT)},
    }, indent=2))

    assert mgt.main(["--carry-est-drift", str(est), "--out", str(geom)]) == mgt.EXIT_OK
    doc = json.loads(geom.read_text())
    assert doc["est_drift_p95_px"] == 1.0
    assert doc[mgt.EST_DRIFT_NAME_FIELD] == "grounding-dino+sam2+depth-anything-v2"

    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)
    record = g0.run_g0b(_args(tmp_path, geom, source, restyled))
    assert record["verdict"] == "PASS"
    assert record["not_gate_qualified_reasons"] == []
    assert record["budget"]["gate_margin_px"] == pytest.approx(3.0)


def test_the_est_drift_artifacts_own_spelling_of_the_name_is_honoured_too(tmp_path: Path) -> None:
    """``estimators.name`` is how ``scripts/measure_est_drift.py`` records the segmenter in ITS
    artifact, and a document assembled by pasting that block in states the same fact. One path
    writes the join key and two spellings state it; neither is a reason to stop checking."""
    fields = dict(MEASURED_FIELDS)
    fields.pop("est_drift_estimator_name")
    fields["estimators"] = {"name": "grounding-dino+sam2+depth-anything-v2"}
    geom = _geom_config(tmp_path, **fields)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)
    record = g0.run_g0b(_args(tmp_path, geom, source, restyled))
    assert record["verdict"] == "PASS"


def test_a_budget_whose_segmenter_the_artifact_never_names_is_refused(tmp_path: Path) -> None:
    """A tolerance that states EST_DRIFT_P95 and no segmenter for it cannot be gated on at all.

    This used to come back as a soft "could not check", and until 2026-08-22 that was the ONLY
    outcome available: ``measure_geom_tol`` wrote no spelling of the drift half's estimator name,
    so every G0b run against a measured tolerance lost its qualification for that reason alone — a
    gate structurally unable to return 0, which blocks generation exactly as a wrong one would.
    ``--carry-est-drift`` now writes ``est_drift_estimator_name`` beside the number and refuses to
    write either alone, so an artifact carrying a budget and no name was assembled by hand and the
    one thing PR-08 §4 step 2 asks about it cannot be established."""
    fields = dict(MEASURED_FIELDS)
    fields.pop("est_drift_estimator_name")
    geom = _geom_config(tmp_path, **fields)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)
    with pytest.raises(g0.GateRefusal, match="names no segmenter for it"):
        g0.run_g0b(_args(tmp_path, geom, source, restyled))


def test_a_declared_assertion_this_artifact_cannot_settle_costs_the_run_its_qualification(
    tmp_path: Path,
) -> None:
    """The soft branch, on the one assertion that legitimately has one: a handler that cannot
    settle its entry says so in the record and does not stand as the gate, rather than reporting
    the entry as checked. The absent .sha256 sidecar is that case — an artifact can be the
    committed one without a digest sitting beside it, and a DISAGREEING digest refuses outright."""
    geom = _geom_config(tmp_path, sidecar=False, **MEASURED_FIELDS)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)
    record = g0.run_g0b(_args(tmp_path, geom, source, restyled))
    assert record["verdict"] == "NOT_GATE_QUALIFIED"
    assert any("could not check" in r for r in record["not_gate_qualified_reasons"])


def test_every_entry_the_producer_actually_writes_dispatches_to_a_handler_here() -> None:
    """The fixture above is a COPY of the producer's list, and a copy goes stale in silence.

    This reads the ``consumer_asserts`` list literal out of ``scripts/measure_geom_tol.py``'s own
    source — the same trick that file uses to force ``measure_est_drift``'s field reads to be
    declared — and asserts this runner can dispatch every entry of it. An entry with no handler
    REFUSES the gate, which is the right direction and is also a G0b that cannot run at all
    against the artifact its own producer writes; without this test the first thing to notice
    would be an exit 2 on the cluster.
    """
    import ast

    tree = ast.parse((_REPO_ROOT / "scripts" / "measure_geom_tol.py").read_text(encoding="utf-8"))
    entries: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "consumer_asserts"
                and isinstance(value, ast.List)
            ):
                entries = [
                    e.value for e in value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                ]
    assert entries, "no consumer_asserts list literal found in scripts/measure_geom_tol.py"
    assert len(entries) >= 8, entries
    unhandled = [e for e in entries if g0.consumer_assert_handler(e)[1] is None]
    assert not unhandled, (
        "scripts/measure_geom_tol.py writes consumer assertions this runner would refuse the gate "
        f"over rather than check: {unhandled}"
    )


def test_a_measured_tolerance_whose_two_segmenter_blocks_disagree_is_refused(
    tmp_path: Path,
) -> None:
    """The committed contract and the contract the measurement ran under are the same document in
    two places, and the producer's guard against them differing runs at MEASUREMENT time only. A
    consumer holding the finished artifact has to check it itself, or a moved box threshold makes
    GEOM_TOL a tolerance for an instrument that is not the one G0b gates with."""
    entry = (
        "the segmenter block agrees, and not only the name: the committed contract (top-level "
        "`segmenter`) and what ran (mask_method.params.segmenter) pin the detector, the segmenter, "
        "the depth model AND their revisions, the prompt, both threshold pairs, the box rule and "
        "the propagation mode"
    )
    ran = dict(DEFAULT_CONTRACT)
    fields = dict(
        MEASURED_FIELDS,
        consumer_asserts=[entry] + PRODUCER_ASSERTS,
        mask_method={"name": MEASURED_FIELDS["est_drift_estimator_name"], "params": {"segmenter": ran}},
    )
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)

    # agreeing: the entry is checked and the run stands
    record = g0.run_g0b(_args(tmp_path, _geom_config(tmp_path, **fields), source, restyled))
    checked = {row.get("token"): row["checked"] for row in record["tolerance_consumer_asserts"]["checked"]}
    assert checked["the segmenter block agrees"] is True
    assert record["verdict"] == "PASS"

    # one threshold moved between the commitment and the measurement
    moved = dict(fields)
    moved["mask_method"] = {
        "name": MEASURED_FIELDS["est_drift_estimator_name"],
        "params": {"segmenter": dict(ran, box_threshold=0.35)},
    }
    with pytest.raises(g0.GateRefusal, match="box_threshold"):
        g0.run_g0b(_args(tmp_path, _geom_config(tmp_path, **moved), source, restyled))


def test_a_declared_segmenter_block_assertion_with_only_one_block_to_compare_refuses(
    tmp_path: Path,
) -> None:
    """An impossible comparison is not a satisfied one: the committed pre-measurement contract on
    its own cannot answer "did the measurement run under it"."""
    entry = "the segmenter block agrees, and not only the name: ..."
    fields = dict(MEASURED_FIELDS, consumer_asserts=[entry] + PRODUCER_ASSERTS)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)
    with pytest.raises(g0.GateRefusal, match="mask_method.params.segmenter"):
        g0.run_g0b(_args(tmp_path, _geom_config(tmp_path, **fields), source, restyled))


def test_a_sentence_entry_this_runner_does_not_recognise_still_refuses(tmp_path: Path) -> None:
    """The phrase table must not become a wildcard. "the" is the leading token of the one sentence
    entry the producer writes, and if "the" were a dispatch key every future entry beginning with
    it would be silently handled by the wrong handler — the same drift, one level down."""
    fields = dict(
        MEASURED_FIELDS,
        consumer_asserts=PRODUCER_ASSERTS + ["the depth model agrees with the one §4 step 3 used"],
    )
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)
    with pytest.raises(g0.GateRefusal, match="no handler for"):
        g0.run_g0b(_args(tmp_path, _geom_config(tmp_path, **fields), source, restyled))


def test_a_non_finite_displacement_refuses_instead_of_being_reported_as_a_void(
    tmp_path: Path,
) -> None:
    """``json.loads`` accepts a bare ``NaN``, ``np.percentile`` propagates it, and ``NaN <= budget``
    is False — so an unreadable centroid record would be reported as a VOID, which is a formal
    indictment of the generation pipeline, and the histogram behind it would then fail to bin at
    all. The number is refused where it enters."""
    with pytest.raises(g0.GateRefusal, match="non-finite displacement"):
        g0.paired_displacements(
            [(100.0, 100.0), (101.0, 100.0)],
            [(100.0, 100.0), (float("nan"), 100.0)],
            "episode_000000",
            "object",
        )
    geom = _geom_config(tmp_path)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    broken = json.loads(json.dumps(SOURCE_CLIPS))
    broken["episode_000000"]["object"][1] = [float("nan"), 100.0]
    restyled = _centroids(tmp_path, "restyled", broken)
    code, record = _run(
        tmp_path,
        "--gates", "g0b",
        "--geom-config", str(geom),
        "--source-centroids", str(source),
        "--restyled-centroids", str(restyled),
    )
    assert code == g0.EXIT_REFUSED
    assert record["gates"]["G0b"]["verdict"] == "REFUSED"
    assert "non-finite displacement" in record["gates"]["G0b"]["refusal"]


def test_a_tolerance_measured_at_a_different_step_is_refused_and_not_quoted(
    tmp_path: Path,
) -> None:
    """G0b compares source frame i to restyled frame i. GEOM_TOL scales roughly linearly with the
    step it was measured at, so a tolerance measured at 3 is a gate about three times too loose —
    and nothing about the resulting number would look wrong."""
    geom = _geom_config(tmp_path, step_frames=3)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)
    with pytest.raises(g0.GateRefusal, match="step_frames = 3"):
        g0.run_g0b(_args(tmp_path, geom, source, restyled))


def test_a_tolerance_that_does_not_state_its_step_cannot_stand_as_the_gate(tmp_path: Path) -> None:
    """Asserted whether or not the artifact declares the assertion, because it is the one field
    whose silent disagreement produces a wrong budget rather than a missing one."""
    geom = _geom_config(tmp_path, step_frames=None)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)
    record = g0.run_g0b(_args(tmp_path, geom, source, restyled))
    assert record["verdict"] == "NOT_GATE_QUALIFIED"
    assert any("step_frames" in r for r in record["not_gate_qualified_reasons"])


def test_the_min_coverage_floor_is_the_producers_and_not_a_second_opinion() -> None:
    """A threshold that gates and is not derived from anything is a coined threshold. This one is
    borrowed, and this test is what makes the borrowing real rather than a comment."""
    geom = _load("measure_geom_tol")
    assert g0.MIN_COVERAGE_DEFAULT == geom.DEFAULT_MIN_COVERAGE
    assert g0.parse_args(["--gates", "g0b"]).min_coverage == geom.DEFAULT_MIN_COVERAGE


# --------------------------------------------- the pinned operating point, on every path it can be


def test_a_side_that_states_no_operating_point_cannot_stand_as_the_gate(tmp_path: Path) -> None:
    """``contract_disagreements`` compares the fields both sides state, so a side stating none
    passes by saying nothing. Name, version and grid were still compared — that is a real check and
    it is not the check the committed contract was written to replace."""
    geom = _geom_config(tmp_path)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS, contract=None)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS, contract=None)
    record = g0.run_g0b(_args(tmp_path, geom, source, restyled))
    assert record["verdict"] == "NOT_GATE_QUALIFIED"
    reasons = record["not_gate_qualified_reasons"]
    assert sum("states no segmenter contract" in r for r in reasons) == 2


def test_the_clips_path_reads_the_operating_point_off_the_adapter_it_actually_ran() -> None:
    """The measuring path used to reach the comparison with nothing to compare. The adapter exports
    SEGMENTER_CONTRACT precisely so a script reading two JSON artifacts later can check it."""

    class _Method:
        name = "grounding-dino+sam2+depth-anything-v2"
        params = {
            "estimator_module_file": "scripts/estimators/apple_sam2.py",
            "estimator_spec": "estimators.apple_sam2",
        }

    contract, note = g0.adapter_segmenter_contract(_Method())
    assert contract is not None and "apple_sam2" in note
    for key in ("detector", "segmenter", "depth", "box_threshold", "object_text_prompt"):
        assert key in contract


def test_a_method_with_no_adapter_behind_it_records_the_absence_instead_of_guessing() -> None:
    """--method precomputed has no module to read a contract off. The absence is recorded and costs
    the run its gate qualification; it is never filled in from the committed contract, which would
    make the comparison compare the tolerance against itself."""

    class _Method:
        name = "precomputed"
        params: dict[str, Any] = {}

    contract, note = g0.adapter_segmenter_contract(_Method())
    assert contract is None
    assert "estimator_module_file" in note


# -------------------------------------------------- the pairing, which a restyle makes many-to-one


def test_many_restyled_clips_of_one_source_clip_are_all_compared_against_it(tmp_path: Path) -> None:
    """A restyle emits 25 style-instances per source clip. The pairing is declared by the record,
    never guessed from a naming convention."""
    many = {
        f"episode_000000__style{i}": dict(
            SOURCE_CLIPS["episode_000000"], source_clip="episode_000000"
        )
        for i in range(3)
    }
    geom = _geom_config(tmp_path)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", many)
    record = g0.run_g0b(_args(tmp_path, geom, source, restyled))
    assert record["verdict"] == "PASS"
    assert record["n_clips_compared"] == 3
    assert {row["source_clip"] for row in record["per_clip"]} == {"episode_000000"}


def test_a_source_map_that_names_no_source_for_a_measured_clip_refuses(tmp_path: Path) -> None:
    """A clip with no declared source cannot be compared to anything, and falling back to pairing
    it by name is the guess the map exists to replace."""
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"episode_000000__styleA": "episode_000000"}))
    assert g0.load_source_map(path, "restyled") == {"episode_000000__styleA": "episode_000000"}
    path.write_text(json.dumps({"source_of": {"a": "b"}}))
    assert g0.load_source_map(path, "restyled") == {"a": "b"}
    path.write_text(json.dumps({"episode_000000__styleA": None}))
    with pytest.raises(g0.GateRefusal, match="not to a source clip key"):
        g0.load_source_map(path, "restyled")


def test_the_by_name_pairing_failure_names_the_flag_that_declares_the_real_one(
    tmp_path: Path,
) -> None:
    """The mode was unusable on the only corpus it will ever be pointed at, and said nothing about
    why. A refusal the reader cannot act on is a refusal that gets worked around."""
    geom = _geom_config(tmp_path)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    orphan = {"episode_000000__styleA": dict(SOURCE_CLIPS["episode_000000"])}
    restyled = _centroids(tmp_path, "restyled", orphan)
    # A centroid RECORD declares its own pairing, so the hint is only offered when the pairing was
    # assumed by name -- which is what the --restyled-clips path does without the map.
    side = g0.load_centroid_record(restyled, "restyled")
    side.notes.append("PAIRING ASSUMED BY NAME: fixture stands in for the --restyled-clips path")
    monkeyed = _args(tmp_path, geom, source, restyled)
    original = g0.resolve_side

    def _resolve(args: Any, which: str) -> Any:
        return side if which == "restyled" else original(args, which)

    g0.resolve_side = _resolve  # type: ignore[assignment]
    try:
        with pytest.raises(g0.GateRefusal, match="--restyled-source-map"):
            g0.run_g0b(monkeyed)
    finally:
        g0.resolve_side = original  # type: ignore[assignment]


# --------------------------------------------------------------------------- the exit code table


def test_the_documented_exit_codes_are_the_ones_the_module_returns() -> None:
    """The module docstring is the table operators and the sbatch will read. It is asserted
    against the constants rather than trusted, because a docstring cannot be run."""
    doc = g0.__doc__ or ""
    assert "EXIT STATUS" in doc
    for code, verdict in ((0, "PASS"), (2, "REFUSED"), (3, "NOT_GATE_QUALIFIED"), (4, "VOID")):
        assert f"{code}   {verdict}" in doc
        assert g0.VERDICT_EXIT[verdict] == code
    assert set(g0.VERDICT_EXIT) == set(g0.VERDICT_ORDER)


def test_every_flag_the_documented_invocation_uses_is_a_flag_this_runner_has() -> None:
    """The docstring hands a job script a command line to copy. A command line in a comment that
    the parser would reject is worse than no command line: it looks authoritative and exits 2 on a
    flag, which reads to the operator as the gate refusing."""
    import inspect
    import re

    doc = g0.__doc__ or ""
    section = doc[doc.index("DRIVING THIS FROM A JOB SCRIPT"):]
    source = inspect.getsource(g0.parse_args)
    flags = {f for f in re.findall(r"--[a-z0-9][a-z0-9-]+", section)}
    assert flags, "the documented invocation names no flags at all"
    for flag in sorted(flags):
        assert f'"{flag}"' in source, f"{flag} is documented and is not a flag of this runner"


def test_a_determined_void_outranks_an_unrelated_refusal() -> None:
    """Documented precedence. A reader who is told "fix your inputs" and never learns a VOID was
    determined is the one failure this ordering prevents; every non-zero code blocks equally."""
    assert g0.worst_verdict(["REFUSED", "VOID"]) == "VOID"
    assert g0.worst_verdict(["PASS", "REFUSED"]) == "REFUSED"
    assert g0.worst_verdict(["PASS", "NOT_GATE_QUALIFIED"]) == "NOT_GATE_QUALIFIED"
    assert g0.worst_verdict([]) == "REFUSED"


def test_the_exit_code_carries_the_verdict_end_to_end(tmp_path: Path) -> None:
    """Four runs, four codes, through ``main`` — which is how the sbatch will read this gate."""
    geom = _geom_config(tmp_path)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)

    passing = _centroids(tmp_path, "restyled", _shift(SOURCE_CLIPS, "object", 1, 0.5))
    code, art = _run(
        tmp_path, "--gates", "g0b", "--geom-config", str(geom),
        "--source-centroids", str(source), "--restyled-centroids", str(passing),
    )
    assert (code, art["verdict"]) == (g0.EXIT_PASS, "PASS")

    code, art = _run(
        tmp_path, "--gates", "g0b", "--geom-config", str(geom),
        "--source-centroids", str(source), "--restyled-centroids", str(passing), "--limit", "1",
    )
    assert (code, art["verdict"]) == (g0.EXIT_NOT_GATE_QUALIFIED, "NOT_GATE_QUALIFIED")

    moved = _centroids(tmp_path, "restyled-void", _shift(SOURCE_CLIPS, "object", 1, 12.0))
    code, art = _run(
        tmp_path, "--gates", "g0b", "--geom-config", str(geom),
        "--source-centroids", str(source), "--restyled-centroids", str(moved),
    )
    assert (code, art["verdict"]) == (g0.EXIT_VOID, "VOID")
    assert art["gates"]["G0b"]["void_rows"]

    unmeasured = _geom_config(tmp_path, drift=None)
    code, art = _run(
        tmp_path, "--gates", "g0b", "--geom-config", str(unmeasured),
        "--source-centroids", str(source), "--restyled-centroids", str(passing),
    )
    assert (code, art["verdict"]) == (g0.EXIT_REFUSED, "REFUSED")
    assert "est_drift_p95_px = null" in art["gates"]["G0b"]["refusal"]


def test_g0c_is_recorded_as_not_evaluated_rather_than_omitted(tmp_path: Path) -> None:
    """A gate record that silently omits one of §6's three gates reads downstream as three passes."""
    geom = _geom_config(tmp_path)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)
    _code, art = _run(
        tmp_path, "--gates", "g0b", "--geom-config", str(geom),
        "--source-centroids", str(source), "--restyled-centroids", str(restyled),
    )
    assert art["gates"]["G0c"]["verdict"] == "NOT_EVALUATED_HERE"
    assert "composit" in art["gates"]["G0c"]["why"]


# --------------------------------------------------------------------------------- --explain mode


def test_explain_names_every_missing_input_and_what_would_produce_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The mode this script will spend most of its life in: PR-08 §8 has four open items and the
    two constants G0b needs have never been measured."""
    code = g0.main(
        ["--explain", "--gates", "g0b", "--geom-config", str(tmp_path / "absent.json")]
    )
    out = capsys.readouterr().out
    assert code == g0.EXIT_REFUSED
    assert "MISSING" in out and "measure_geom_tol.py" in out and "measure_est_drift.py" in out
    assert "licenses generation" in out


def test_explain_exits_zero_when_every_input_is_there(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    geom = _geom_config(tmp_path)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)
    code = g0.main(
        [
            "--explain", "--gates", "g0b", "--geom-config", str(geom),
            "--source-centroids", str(source), "--restyled-centroids", str(restyled),
        ]
    )
    assert code == g0.EXIT_PASS
    assert "drop --explain to run them" in capsys.readouterr().out


def test_explain_reports_a_present_config_whose_numbers_are_still_null(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """"the file exists" and "the two numbers in it subtract to something positive" are different
    facts, and the second decides whether G0b can run at all."""
    geom = _geom_config(tmp_path, geom_tol=None, drift=None)
    source = _centroids(tmp_path, "source", SOURCE_CLIPS)
    restyled = _centroids(tmp_path, "restyled", SOURCE_CLIPS)
    code = g0.main(
        [
            "--explain", "--gates", "g0b", "--geom-config", str(geom),
            "--source-centroids", str(source), "--restyled-centroids", str(restyled),
        ]
    )
    # Every FILE is present and the gate STILL cannot run, which is a refusal and not a pass:
    # PR-08 §4 requires the estimator to be characterised before generation, never after.
    assert code == g0.EXIT_REFUSED
    assert "budget cannot be formed" in capsys.readouterr().out


# ------------------------------------------- G0a end to end: the injection reaches the actual CLI


def _build_corpus(root: Path, *, n_episodes: int = 6, n_chunks: int = 3, grip_events: int = 3,
                  seed: int = 0) -> Path:
    """A tiny corpus in the shape ``screen_corpus.load_episode`` reads.

    The targets are a fixed linear map of the lagged blind state, so the blind ceiling genuinely
    transfers between episodes and ``ceiling_dominates`` is true — without that, M1's denominator
    is undefined and the identity check would be comparing two numbers screen_corpus itself calls
    VOID. ``grip_events`` is the knob the VOID case turns: the gripper channel is where M3 lives,
    and a restyle that changed it would be a label pipeline that dropped transitions.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    chunk, dim, lags = 16, 2, (0, 1, 2, 3, 4, 6, 8, 12)
    mixer = np.random.default_rng(7).normal(0.0, 1.0, (6 * len(lags), chunk * dim))
    root.mkdir(parents=True, exist_ok=True)
    for episode in range(n_episodes):
        frames = n_chunks * chunk + 20
        q = np.cumsum(rng.normal(0.0, 0.01, (frames, dim)), axis=0)
        dq = np.vstack([np.zeros((1, dim)), np.diff(q, axis=0)])
        grip = np.zeros((frames, 2))
        for k in range(grip_events):
            grip[k * 13 + 5 : k * 13 + 11, 0] = 1.0
        directory = root / f"episode_{episode:04d}"
        directory.mkdir(exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "q": [[float(v) for v in row] for row in q],
                    "dq": [[float(v) for v in row] for row in dq],
                    "gripper_state": [[float(v) for v in row] for row in grip],
                }
            ),
            directory / "states.parquet",
        )
        stream = np.hstack([q, dq, grip])
        chunk_idx, step_idx, targets = [], [], []
        for c in range(n_chunks):
            start = c * chunk
            feature = np.concatenate([stream[max(start - lag, 0)] for lag in lags])
            block = (feature @ mixer).reshape(chunk, dim) + rng.normal(0.0, 0.02, (chunk, dim))
            for step in range(chunk):
                chunk_idx.append(c)
                step_idx.append(step)
                targets.append([float(v) for v in block[step]])
        pq.write_table(
            pa.table(
                {
                    "chunk_idx": chunk_idx,
                    "step_idx": step_idx,
                    "targets": targets,
                    "dt_s": [0.0333] * len(chunk_idx),
                }
            ),
            directory / "actions.parquet",
        )
    return root


@pytest.fixture()
def _fast_screen(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Drive the REAL ``screen_corpus.main`` — with a small ceiling, and nothing else changed.

    The end-to-end runs exist to prove one thing no unit test can: that the source's measured triple
    really is reachable through ``--expect`` on the actual command line. That needs the real parser,
    the real loader and the real comparison loop. It does NOT need 2048 random Fourier features or a
    20-point hyperparameter search: at the defaults the two screens take about three minutes, all of
    it inside ridge solves whose SIZE and GRID are ``screen_corpus``'s own design, pinned in
    ``tests/test_screen_corpus.py``, and nothing this gate does depends on either. The blind ceiling
    still runs, still fits, and still has to dominate — only smaller.

    ``run_g0_gates.load_script`` is redirected to the module this file already imported, so the
    smaller ceiling reaches the module the gate actually drives.
    """
    monkeypatch.setattr(screen_corpus, "N_FEATURES", 64)
    monkeypatch.setattr(screen_corpus, "GAMMA_SCALES", (1.0,))
    monkeypatch.setattr(screen_corpus, "LAMBDAS", (1e-1,))
    monkeypatch.setattr(
        g0, "load_script", lambda name: screen_corpus if name == "screen_corpus" else _load(name)
    )
    return screen_corpus


def _holdout(tmp_path: Path) -> Path:
    path = tmp_path / "holdout.txt"
    path.write_text("episode_0004\nepisode_0005\n")
    return path


def test_a_restyle_that_carried_the_labels_over_unchanged_passes_g0a_end_to_end(
    tmp_path: Path, _fast_screen: Any
) -> None:
    """The identity check, through ``screen_corpus.main --expect pr08-source`` on a real corpus.

    This is the only test that proves the injected reference is reachable from the CLI: everything
    else asserts the comparison, and argparse would reject an unregistered ``--expect`` key with a
    SystemExit that no unit test of ``expect_deltas`` would ever see.
    """
    source = _build_corpus(tmp_path / "source")
    restyled = _build_corpus(tmp_path / "restyled")  # identical: a restyle changes no action
    code, art = _run(
        tmp_path,
        "--gates", "g0a",
        "--source-dataset", str(source),
        "--restyled-dataset", str(restyled),
        "--holdout", str(_holdout(tmp_path)),
    )
    record = art["gates"]["G0a"]
    assert (code, art["verdict"]) == (g0.EXIT_PASS, "PASS"), record
    assert record["reference_key"] == g0.SOURCE_REFERENCE_KEY
    assert record["screen_corpus_expect_status"] == 0
    assert all(row["abs_delta"] == pytest.approx(0.0, abs=1e-9) for row in record["deltas"])


def test_a_restyle_whose_gripper_channel_moved_is_void_end_to_end(
    tmp_path: Path, _fast_screen: Any
) -> None:
    """M3 is debounced gripper transitions per episode. A generation pipeline that dropped or
    reordered them changes it far past EXPECT_TOL — which is exactly the defect §6 says this gate
    is looking for, and it is not a finding about the corpus."""
    source = _build_corpus(tmp_path / "source", grip_events=3)
    restyled = _build_corpus(tmp_path / "restyled", grip_events=1)
    code, art = _run(
        tmp_path,
        "--gates", "g0a",
        "--source-dataset", str(source),
        "--restyled-dataset", str(restyled),
        "--holdout", str(_holdout(tmp_path)),
    )
    record = art["gates"]["G0a"]
    assert (code, art["verdict"]) == (g0.EXIT_VOID, "VOID"), record
    assert record["screen_corpus_expect_status"] != 0
    m3 = next(row for row in record["deltas"] if row["metric"] == "m3")
    assert not m3["within_tol"] and m3["delta"] == pytest.approx(-4.0)
