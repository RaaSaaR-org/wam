"""Tests for scripts/preflight_gpu.py — the gate before the first GPU minute on a new box.

Everything here runs on CPU, on a Mac, with no CUDA and no Wan weights. That is the whole design
constraint: the script exists because the 5090 box cannot be tested from here, so the parts that
*can* be tested from here — the budget arithmetic, the extras parsing, the dependency probe, the
kernel-probe bodies, the exit-code rule — are pure functions with their inputs passed in.

Two things these tests are careful about, both scar tissue from TASKS.md T-29 (a regression guard
that ran on data which could not exercise the path it guarded):

1. The extras tests read the REAL pyproject.toml, not a fixture. The bug being guarded was that
   ``docs/local_gpu.md`` §0 told the user ``pip install -e '.[dev]'`` while ``dev`` is only
   [pytest, ruff, peft] — a fixture copy would keep passing after someone fixed or broke the
   real file, which is the one thing this check is for. That coupling has since fired once, as
   designed: pyproject gained a ``local`` extra (the flat union the runbook now installs) and
   these assertions were updated to it rather than pinned to a copy.
2. The budget tests assert verdicts that DIFFER between cards. A budget test on a single card
   size cannot distinguish "the arithmetic works" from "everything returns FITS".
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves annotations through sys.modules[__module__],
    # and without this the class bodies in the script raise on import.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pf = _load("preflight_gpu")


# ---------------------------------------------------------------------------------------------
# exit code


def test_exit_code_is_zero_only_without_fail() -> None:
    assert pf.exit_code([]) == 0
    assert pf.exit_code(["PASS", "PASS"]) == 0
    assert pf.exit_code(["PASS", "WARN", "PASS"]) == 0, "a WARN that blocks stops being read"
    assert pf.exit_code(["PASS", "FAIL"]) == 1
    assert pf.exit_code(["FAIL"]) == 1


def test_report_tracks_failures_and_warnings_separately() -> None:
    report = pf.Report(echo=False)
    report.section("t")
    report.check("a", True, "fine")
    report.soft("b", False, "meh", fix="do the thing")
    report.check("c", False, "broken", fix="fix the thing")

    assert report.failed == ["c"]
    assert report.warned == ["b"]
    assert pf.exit_code(report.statuses()) == 1


# ---------------------------------------------------------------------------------------------
# VRAM budget — the deliverable


def test_verdict_boundaries() -> None:
    # headroom exactly 0 is not "fits": the card has to hold the CUDA context too.
    assert pf.verdict(peak_gb=34.36, total_gb=34.36) == "WILL OOM"
    assert pf.verdict(peak_gb=40.0, total_gb=34.36) == "WILL OOM"
    assert pf.verdict(peak_gb=33.0, total_gb=34.36) == "TIGHT"
    assert pf.verdict(peak_gb=32.37, total_gb=34.36) == "TIGHT"  # headroom 1.99 < 2.0
    assert pf.verdict(peak_gb=32.30, total_gb=34.36) == "FITS"  # headroom 2.06
    assert pf.verdict(peak_gb=24.28, total_gb=34.36) == "FITS"


def test_the_oom_threshold_is_the_measured_allocator_overhead_not_zero() -> None:
    """Positive nominal headroom is not enough: allocated < occupied by the allocator's slack.

    This boundary is the whole reason the dream row can report anything useful. Against a total
    of 34.36 that row (32.54) reads TIGHT on every plausible card, so the check could only fire
    on hardware nobody sells. What makes it fire is measuring free VRAM AND admitting that the
    driver is asked for ALLOCATOR_OVERHEAD_GB more than max_memory_allocated() reports.
    """
    assert pf.ALLOCATOR_OVERHEAD_GB == 0.90  # measured, job 183599:129-130

    # A 5090 with a desktop compositor up: 32.90 GB free, dream's measured 32.54 GB peak.
    # +0.36 GB of nominal headroom, and 0.54 GB short once the slack is counted.
    assert pf.verdict(peak_gb=32.54, total_gb=32.90) == "WILL OOM"

    # Exactly at the threshold is still OOM; a hair above it is TIGHT.
    assert pf.verdict(peak_gb=32.00, total_gb=32.90) == "WILL OOM"  # headroom == 0.90
    assert pf.verdict(peak_gb=31.99, total_gb=32.90) == "TIGHT"  # headroom 0.91

    # And the headless case, where it genuinely is only tight.
    assert pf.verdict(peak_gb=32.54, total_gb=34.05) == "TIGHT"  # headroom 1.51


def test_verdict_tight_margin_is_exclusive_at_the_boundary() -> None:
    """headroom == margin is FITS; anything under it is TIGHT. Pinned so it cannot drift."""
    assert pf.verdict(peak_gb=10.0, total_gb=12.0, tight_margin_gb=2.0) == "FITS"
    assert pf.verdict(peak_gb=10.01, total_gb=12.0, tight_margin_gb=2.0) == "TIGHT"


def _row(rows, needle: str):
    matches = [r for r in rows if needle in r.entry.entry_point]
    assert len(matches) == 1, f"expected exactly one {needle!r} row, got {len(matches)}"
    return matches[0]


def test_dream_is_not_safe_on_a_32_gib_card() -> None:
    """32 GiB = 34.36 decimal GB, and dream's measured peak is 32.54 decimal GB.

    That is 1.82 GB of headroom against a 0.90 GB allocator overhead measured on job 183599 —
    below the TIGHT margin, and the artifact it came from includes the load transient while
    excluding the CUDA context. So: not FITS, and the script must not say it is.
    """
    rows = pf.budget_rows(pf.NOMINAL_RTX5090_TOTAL_GB)
    dream = _row(rows, "dream.py")

    assert dream.entry.peak_gb == pf.DREAM_MOTION_PEAK_GB == 32.54
    assert dream.verdict == "TIGHT"
    assert dream.verdict != "FITS"
    assert dream.entry.measured is True


def test_dream_will_oom_on_a_24_gb_card() -> None:
    """The WILL OOM branch has to be reachable from a real entry, not only from verdict()."""
    rows = pf.budget_rows(24.0)
    assert _row(rows, "dream.py").verdict == "WILL OOM"
    assert _row(rows, "WanImageToVideoPipeline").verdict == "WILL OOM"


def test_the_budget_plans_against_free_vram_not_the_board_total() -> None:
    """A compositor holding 1.5 GB must cost you 1.5 GB of budget, not nothing.

    Section 4 only WARNs below 90% free, so a desktop compositor passes there. If the budget
    then plans against the board total, that memory is spent twice and no check ever notices.
    """
    report = pf.Report(echo=False)
    pf.section_budget(report, {"vram_total_gb": 34.19, "vram_free_gb": 31.60}, None)

    budget = report.info["budget"]
    assert budget["card_total_gb"] == 31.60, "planned against the total, not what is free"
    assert "FREE" in budget["card_source"] and "2.59 GB already held" in budget["card_source"]

    dream = next(r for r in budget["rows"] if "dream.py" in r["entry_point"])
    assert dream["verdict"] == "WILL OOM"
    check = next(c for c in report.checks if c.name == "budget.scripts/dream.py")
    assert check.status == pf.STATUS_FAIL, "a measured row that will OOM has to FAIL, not WARN"


def test_the_budget_says_so_when_it_had_to_fall_back_to_the_total() -> None:
    """Falling back is fine; doing it silently is not — the number is optimistic and must say so."""
    report = pf.Report(echo=False)
    pf.section_budget(report, {"vram_total_gb": 34.19, "vram_free_gb": None}, None)

    source = report.info["budget"]["card_source"]
    assert report.info["budget"]["card_total_gb"] == 34.19
    assert "optimistic" in source


def test_load_transient_is_not_a_constant() -> None:
    """It was hardcoded WARN — the same verdict on a 12 GB laptop and an 80 GB H100."""

    def transient(free_gb: float) -> str:
        report = pf.Report(echo=False)
        pf.section_budget(report, {"vram_total_gb": free_gb, "vram_free_gb": free_gb}, None)
        return next(c for c in report.checks if c.name == "budget.load_transient").status

    assert transient(80.0) == pf.STATUS_PASS
    assert transient(34.19) == pf.STATUS_WARN


def test_everything_fits_on_an_80_gb_card() -> None:
    rows = pf.budget_rows(80.0)
    assert [r.verdict for r in rows] == ["FITS"] * len(rows)
    assert all(r.headroom_gb > 40 for r in rows)


def test_smoke_readout_fits_a_32_gib_card_with_room() -> None:
    rows = pf.budget_rows(pf.NOMINAL_RTX5090_TOTAL_GB)
    smoke = _row(rows, "hf_job_wan_smoke.py")

    assert smoke.entry.peak_gb == pf.WAN_SMOKE_PEAK_GB == 24.28
    assert smoke.verdict == "FITS"
    assert smoke.headroom_gb == pytest.approx(10.08, abs=0.01)


def test_headroom_is_card_minus_peak() -> None:
    rows = pf.budget_rows(50.0)
    for row in rows:
        assert row.headroom_gb == pytest.approx(50.0 - row.entry.peak_gb)
        assert row.total_gb == 50.0


def test_estimates_are_labelled_as_estimates_not_as_measurements() -> None:
    """The repo's rule: never print a number you did not compute without saying so."""
    rows = pf.budget_rows(pf.NOMINAL_RTX5090_TOTAL_GB)
    training = _row(rows, "train_t16_lora.py")
    evaluation = _row(rows, "eval_t16.py")

    assert training.entry.measured is False
    assert evaluation.entry.measured is False
    for row in (training, evaluation):
        assert "NOT MEASURED" in row.entry.provenance or "ESTIMATE" in row.entry.provenance

    rendered = pf.render_budget_table(rows, "34.36 GB")
    assert "(est)" in rendered
    assert "~27.70" in rendered, "an estimate must not render like a measurement"
    assert "NOT MEASURED" in rendered


def test_training_row_agrees_with_the_config_header_and_the_runbook() -> None:
    """One number, three files. The runbook says §0c IS this table; then it has to be.

    ``configs/training/joint_wan_gr00t_5090.yaml`` derives the resident floor from safetensors
    headers (25.50) and the batch-2 peak from it (27.70); ``docs/local_gpu.md`` §0c and §5 print
    both. This file used to carry 23.70, from a VAE term taken at bf16 when ``wan_i2v.py:289-291``
    hard-wires fp32 — a 1.8 GB gap between two tables that claim to be the same table.
    """
    floor, peak = pf.TRAINING_FLOOR_GB_ESTIMATE, pf.TRAINING_BATCH2_PEAK_GB_ESTIMATE
    assert (floor, peak) == (25.50, 27.70)
    assert peak > floor, "the batch-2 peak has to sit above the resident floor it is built on"

    yaml_header = (_REPO_ROOT / "configs" / "training" / "joint_wan_gr00t_5090.yaml").read_text()
    assert "25.5005" in yaml_header, "the config header no longer derives this floor"
    assert "27.7 GB" in yaml_header, "the config header no longer derives this batch-2 peak"

    runbook = (_REPO_ROOT / "docs" / "local_gpu.md").read_text()
    assert "| **resident floor** | **25.50** |" in runbook
    assert "~27.7" in runbook

    row = _row(pf.budget_rows(pf.NOMINAL_RTX5090_TOTAL_GB), "train_t16_lora.py")
    assert row.entry.peak_gb == peak
    assert f"{floor:.2f}" in row.entry.caveat, "the measured floor must travel with the estimate"


def test_every_measured_entry_carries_its_artifact_path() -> None:
    """Provenance is the point. A peak without the run that produced it is an assertion."""
    for row in pf.budget_rows(pf.NOMINAL_RTX5090_TOTAL_GB):
        if not row.entry.measured:
            continue
        assert "runs/" in row.entry.provenance, row.entry.entry_point
        assert ".json:" in row.entry.provenance, row.entry.entry_point


def test_every_entry_names_a_lever() -> None:
    for row in pf.budget_rows(34.36):
        assert row.entry.lever.strip(), row.entry.entry_point


def test_offload_reachability_is_read_from_the_scripts_not_asserted(tmp_path: Path) -> None:
    """Whether an entry point can reach offload() is a property of the tree, not a constant.

    It was true of hf_job_wan_smoke.py alone when this was written and is being wired into the
    others; a hardcoded sentence would have gone stale the same week.
    """
    (tmp_path / "a.py").write_text("p.add_argument('--offload-text', action='store_true')")
    (tmp_path / "b.py").write_text("no flag here")

    found = pf.scripts_exposing_flag(tmp_path, pf.OFFLOAD_FLAG, ("a.py", "b.py"))
    assert found == {"a.py": True, "b.py": False}


def test_flag_reachability_treats_an_unreadable_script_as_unreachable(tmp_path: Path) -> None:
    """Fail toward 'the lever is not there' — the other direction sends the user into an OOM."""
    assert pf.scripts_exposing_flag(tmp_path, pf.OFFLOAD_FLAG, ("gone.py",)) == {"gone.py": False}


def test_flag_reachability_covers_the_real_entry_points() -> None:
    reach = pf.scripts_exposing_flag(_REPO_ROOT / "scripts", pf.OFFLOAD_FLAG)
    device_map = pf.scripts_exposing_flag(_REPO_ROOT / "scripts", pf.DEVICE_MAP_FLAG)

    assert set(reach) == set(pf.OFFLOAD_ENTRY_POINTS)
    # the one that has always had both; every other entry is whatever the tree currently says
    assert reach["hf_job_wan_smoke.py"] is True
    assert device_map["hf_job_wan_smoke.py"] is True


def test_the_load_transient_is_called_out_as_unmeasured() -> None:
    """Two of the peaks exclude the load. A budget that hides that is worse than no budget."""
    assert "reset_peak_memory_stats" in pf.LOAD_TRANSIENT_WARNING
    assert str(pf.DREAM_MOTION_PEAK_GB) in pf.LOAD_TRANSIENT_WARNING
    assert "nothing in runs/ measures it" in pf.LOAD_TRANSIENT_WARNING
    assert "eval_t16" in pf.LOAD_TRANSIENT_WARNING


def test_rendered_table_carries_provenance_and_the_offload_lever() -> None:
    rendered = pf.render_budget_table(pf.budget_rows(34.36), "34.36 GB assumed")

    assert "runs/smoke/183599/wan_smoke_report.json:129" in rendered
    assert "runs/dream/t36-zerogpu-motion-seed0/dream.json:265" in rendered
    # the biggest single lever, named by its implementation site rather than by a stale claim
    assert "wan_i2v.py:397" in rendered
    assert "budget.offload_lever" in rendered
    assert "DECIMAL GB" in rendered


def test_render_handles_an_empty_table() -> None:
    assert pf.render_budget_table([], "whatever") == "(no budget entries)"


def test_budget_row_serializes_for_json() -> None:
    row = pf.budget_rows(34.36)[0]
    payload = row.as_dict()
    assert set(payload) >= {"entry_point", "peak_gb", "measured", "verdict", "provenance", "lever"}
    json.dumps(payload)  # must not raise


# ---------------------------------------------------------------------------------------------
# pyproject extras — parsed from the REAL file, so the fix line cannot drift


def test_extras_parse_against_the_real_pyproject() -> None:
    extras = pf.parse_pyproject_extras((_REPO_ROOT / "pyproject.toml").read_text())

    assert set(extras) == {
        "(core)",
        "train",
        "data",
        "serve",
        "wan",
        "jobs",
        "sim",
        "isaac",
        "dev",
        "local",
    }
    # DELIBERATELY EMPTY, and it has to survive parsing as such: installing wam[isaac] in this
    # venv must not pull isaacsim-core, which pins torch 2.11.0 against uv.lock's 2.13.0. The
    # regex fallback used to drop empty extras while tomllib kept them, so python 3.10 and 3.11
    # disagreed about what this file declares.
    assert extras["isaac"] == ()
    # typing-extensions is core because `from typing_extensions import Self` is module-level in
    # wam.interfaces.versioning — it used to arrive only via pydantic.
    assert extras["(core)"] == ("numpy", "pydantic", "pyyaml", "typing-extensions")
    assert extras["data"] == ("pyarrow", "opencv-python", "av")
    # imageio/imageio-ffmpeg are dev-only on purpose: nothing in src/ imports them, but
    # tests/test_cosmos3_probe.py imports the probe script to check its frame arithmetic.
    assert extras["dev"] == ("pytest", "ruff", "peft", "imageio", "imageio-ffmpeg")
    assert "diffusers" in extras["wan"] and "transformers" in extras["wan"]
    # `local` is the one-bracket install docs/local_gpu.md §0 prescribes. It has to be a superset
    # of what the runbook actually runs, or the runbook is back to failing on import.
    for dist in ("torch", "diffusers", "pyarrow", "opencv-python", "av", "websockets", "mujoco"):
        assert dist in extras["local"], f"{dist} missing from the local extra"
    # Declared last, so cover_extras' declaration-order tie-break still prefers a narrower group.
    assert list(extras)[-1] == "local"


def test_local_extra_does_not_drift_from_the_groups_it_unions() -> None:
    """`local` is a flat copy of five groups, so it can silently fall behind any of them.

    This is the cost of NOT writing it as `wam[wan,data,...]` (which would blind the extras
    parser). The guard is the compensation: add a package to `wan` and forget `local`, and the
    runbook's one-bracket install quietly stops installing it — which is the exact class of bug
    `local` was introduced to kill.

    `jobs` is deliberately outside the union: huggingface_hub is for launching work ON HF Jobs,
    not for running it locally, and diffusers pulls it in anyway.
    """
    extras = pf.parse_pyproject_extras((_REPO_ROOT / "pyproject.toml").read_text())
    unioned = ("train", "data", "serve", "wan", "sim", "dev")

    covered = set(extras["local"])
    for group in unioned:
        missing = set(extras[group]) - covered
        assert not missing, f"{group} has {sorted(missing)} that `local` does not install"
    # And nothing invented: everything in `local` comes from one of those groups.
    from_groups = {d for group in unioned for d in extras[group]}
    assert covered <= from_groups, f"`local` adds {sorted(covered - from_groups)} out of nowhere"


def test_the_shipped_bug_dev_extra_provides_none_of_the_gpu_dependencies() -> None:
    """docs/local_gpu.md §0 used to say `pip install -e '.[dev]'`. This is why it could not work.

    If someone fixes pyproject or the runbook, this test fails and the fix line the script prints
    has to be revisited — which is the intended coupling.
    """
    extras = pf.parse_pyproject_extras((_REPO_ROOT / "pyproject.toml").read_text())
    dev = set(extras["dev"]) | set(extras["(core)"])

    for missing in (
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "pyarrow",
        "opencv-python",
        "av",
        "websockets",
        "torch",
    ):
        assert pf.canonical_dist(missing) not in dev, f"{missing} is now in dev — update the fix"


def test_regex_fallback_matches_tomllib_on_the_real_pyproject() -> None:
    """python 3.10 has no tomllib (pyproject.toml:5 allows 3.10), so the fallback must agree."""
    text = (_REPO_ROOT / "pyproject.toml").read_text()
    assert pf._extras_via_regex(text) == pf.parse_pyproject_extras(text)


def test_regex_fallback_handles_single_line_and_commented_tables() -> None:
    text = (
        "[project]\n"
        'name = "wam"\n'
        'dependencies = ["numpy", "pydantic>=2"]\n'
        "[project.optional-dependencies]\n"
        '# wan = ["never-installed"]   <- a comment, not a dependency\n'
        "wan = [\n"
        '    "torch",       # trailing comment\n'
        '    "diffusers>=0.35",\n'
        "]\n"
        "[build-system]\n"
        'requires = ["hatchling"]\n'
    )
    parsed = pf._extras_via_regex(text)

    assert parsed == {"(core)": ("numpy", "pydantic"), "wan": ("torch", "diffusers")}
    assert "never-installed" not in parsed["wan"]
    assert "hatchling" not in {d for dists in parsed.values() for d in dists}


def test_requirement_dist_strips_specifiers_and_normalizes() -> None:
    assert pf.requirement_dist("peft>=0.14") == "peft"
    assert pf.requirement_dist("websockets >= 12") == "websockets"
    assert pf.requirement_dist("opencv_python") == "opencv-python"
    assert pf.requirement_dist("torch[opt]==2.9.0") == "torch"
    assert pf.requirement_dist('mujoco; sys_platform == "linux"') == "mujoco"


def test_extras_providing_finds_every_group_that_installs_a_package() -> None:
    extras = pf.parse_pyproject_extras((_REPO_ROOT / "pyproject.toml").read_text())

    assert set(pf.extras_providing("peft", extras)) == {"wan", "dev", "local"}
    assert pf.extras_providing("pyarrow", extras) == ("data", "local")
    assert pf.extras_providing("mujoco", extras) == ("sim", "local")
    assert pf.extras_providing("nothing-at-all", extras) == ()


# ---------------------------------------------------------------------------------------------
# dependency probe + the corrected pip line


def test_probe_reports_missing_packages_with_the_extras_that_provide_them() -> None:
    extras = pf.parse_pyproject_extras((_REPO_ROOT / "pyproject.toml").read_text())
    present = {"numpy", "pydantic", "yaml", "typing_extensions", "pytest", "ruff"}

    rows = pf.probe_requirements(pf.REQUIREMENTS, extras, probe=lambda m: m in present)

    by_module = {r["module"]: r for r in rows}
    assert by_module["numpy"]["present"] is True
    assert by_module["pyarrow"]["present"] is False
    assert by_module["pyarrow"]["extras"] == ["data", "local"]
    assert by_module["cv2"]["distribution"] == "opencv-python"
    assert set(by_module["peft"]["extras"]) == {"wan", "dev", "local"}
    # A core dependency, so no bracket names it — `pip install -e .` already brings it.
    assert by_module["typing_extensions"]["extras"] == ["(core)"]


def test_probe_covers_the_packages_the_dev_extra_forgets() -> None:
    """Guard against someone quietly shortening REQUIREMENTS to whatever happens to be installed."""
    modules = {r.module for r in pf.REQUIREMENTS}
    assert {
        "pyarrow",
        "cv2",
        "av",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "websockets",
    } <= modules


def test_a_probe_that_never_finds_anything_reports_everything_missing() -> None:
    """The test that proves the probe can fail — see the module docstring."""
    extras = pf.parse_pyproject_extras((_REPO_ROOT / "pyproject.toml").read_text())
    rows = pf.probe_requirements(pf.REQUIREMENTS, extras, probe=lambda _m: False)
    assert all(r["present"] is False for r in rows)
    assert len(rows) == len(pf.REQUIREMENTS)


def test_cover_extras_picks_a_minimal_deterministic_set() -> None:
    extras = pf.parse_pyproject_extras((_REPO_ROOT / "pyproject.toml").read_text())

    # `data` and `local` both cover all three; the tie breaks on declaration order, and `local`
    # is declared last precisely so the narrower group wins here.
    chosen, uncovered = pf.cover_extras(["pyarrow", "av", "opencv-python"], extras)
    assert chosen == ["data"]
    assert uncovered == []

    # Spanning three narrow groups, `local` covers strictly more than any of them, so it wins on
    # count — which is the point of having it: one bracket instead of three that drift.
    chosen, uncovered = pf.cover_extras(["diffusers", "pyarrow", "websockets"], extras)
    assert chosen == ["local"]
    assert uncovered == []

    # Multi-extra covers still have to be deterministic where `local` does NOT apply. `jobs` and
    # `sim` are outside it, so this is the case that still exercises the ordering rule.
    chosen, uncovered = pf.cover_extras(["huggingface-hub", "mujoco"], extras)
    assert chosen == ["jobs", "sim"], "declaration order, so the command is stable"
    assert uncovered == []


def test_cover_extras_reports_what_no_extra_provides() -> None:
    extras = pf.parse_pyproject_extras((_REPO_ROOT / "pyproject.toml").read_text())
    chosen, uncovered = pf.cover_extras(["typing_extensions", "pyarrow"], extras)

    assert chosen == ["data"]
    assert uncovered == ["typing-extensions"]


def test_cover_extras_never_names_the_core_group() -> None:
    """`pip install -e '.[(core)]'` is not a thing — a bare install already brings those."""
    extras = pf.parse_pyproject_extras((_REPO_ROOT / "pyproject.toml").read_text())
    chosen, uncovered = pf.cover_extras(["numpy"], extras)

    assert "(core)" not in chosen
    assert uncovered == ["numpy"]


def test_pip_command_is_the_corrected_runbook_line() -> None:
    """The whole point of section 5: turn the shipped `.[dev]` into something that works."""
    extras = pf.parse_pyproject_extras((_REPO_ROOT / "pyproject.toml").read_text())
    missing = [
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "ftfy",
        "peft",
        "pyarrow",
        "opencv-python",
        "av",
        "websockets",
    ]

    command = pf.pip_command(missing, extras)

    # One bracket, and the SAME one docs/local_gpu.md §0 prints. Before the `local` extra existed
    # this was `.[data,serve,wan]` — three groups the runbook had to keep in sync by hand.
    assert command == "pip install -e '.[local]'"


def test_pip_command_puts_the_cu128_wheel_first_when_torch_is_missing() -> None:
    """An extra would pull torch off the default index — on Blackwell that wheel is useless."""
    extras = pf.parse_pyproject_extras((_REPO_ROOT / "pyproject.toml").read_text())
    command = pf.pip_command(["torch", "diffusers"], extras, torch_missing=True)

    lines = command.splitlines()
    assert "download.pytorch.org/whl/cu128" in lines[0]
    assert lines[1] == "pip install -e '.[wan]'"


def test_pip_command_is_empty_when_nothing_is_missing() -> None:
    extras = pf.parse_pyproject_extras((_REPO_ROOT / "pyproject.toml").read_text())
    assert pf.pip_command([], extras) == ""


# ---------------------------------------------------------------------------------------------
# torch / kernel-launch helpers


def test_is_kernel_image_error_matches_the_blackwell_failure() -> None:
    assert pf.is_kernel_image_error(
        "CUDA error: no kernel image is available for execution on the device"
    )
    assert pf.is_kernel_image_error("no kernel image is available")
    assert not pf.is_kernel_image_error("CUDA out of memory. Tried to allocate 2.00 GiB")
    assert not pf.is_kernel_image_error("")


def test_kernel_failure_fix_offers_cu128_only_for_the_kernel_image_error() -> None:
    fix = pf.kernel_failure_fix(
        "CUDA error: no kernel image is available for execution on the device", (12, 0)
    )
    assert "cu128" in fix
    assert "sm_120" in fix

    other = pf.kernel_failure_fix("CUDA out of memory", (12, 0))
    assert "cu128" not in other


def test_cuda_toolkit_verdict_warns_on_blackwell_against_an_old_wheel() -> None:
    status, detail, fix = pf.cuda_toolkit_verdict((12, 0), "12.4")

    # WARN, never FAIL: section 3 launches real kernels and that measurement outranks this string
    # comparison. See the docstring on cuda_toolkit_verdict.
    assert status == pf.STATUS_WARN
    assert "sm_120" in detail
    assert "cu128" in fix


def test_cuda_toolkit_verdict_passes_a_matched_pair() -> None:
    status, detail, fix = pf.cuda_toolkit_verdict((12, 0), "12.8")
    assert status == pf.STATUS_PASS
    assert fix == ""
    assert "sm_120" in detail


def test_cuda_toolkit_verdict_flags_a_cpu_only_wheel_on_a_real_card() -> None:
    status, detail, fix = pf.cuda_toolkit_verdict((12, 0), None)
    assert status == pf.STATUS_WARN
    assert "CPU-only" in detail
    assert "cu128" in fix


def test_cuda_toolkit_verdict_is_quiet_without_a_card() -> None:
    status, _detail, fix = pf.cuda_toolkit_verdict(None, "12.8")
    assert status == pf.STATUS_WARN
    assert fix == ""


def test_older_card_on_an_older_wheel_is_fine() -> None:
    """Ampere on CUDA 12.1 is a perfectly good pairing — the check must not cry wolf."""
    status, _detail, _fix = pf.cuda_toolkit_verdict((8, 6), "12.1")
    assert status == pf.STATUS_PASS


def test_parse_version_tuple() -> None:
    assert pf.parse_version_tuple("12.8") == (12, 8)
    assert pf.parse_version_tuple("2.9.0+cu128") == (2, 9, 0, 128)
    assert pf.parse_version_tuple(None) is None
    assert pf.parse_version_tuple("") is None


def test_kernel_probes_cover_the_paths_this_repo_actually_launches() -> None:
    """SDPA is in the list because nothing here sets attn_implementation — it IS the attention."""
    names = [name for name, _fn in pf.kernel_probes()]
    assert names == ["bf16_matmul", "fp32_matmul", "sdpa", "conv2d", "layernorm"]


class _FakeCuda:
    def __init__(self) -> None:
        self.synchronized = 0

    def synchronize(self) -> None:
        self.synchronized += 1

    def is_available(self) -> bool:
        return True


class _FakeTorch:
    """Real torch, with 'cuda' rewritten to 'cpu' — or with every launch raising.

    This is how the section that matters gets tested without a Blackwell card: the probes, the
    RuntimeError handling and the fix line are all exercised; only the silicon is fake.
    """

    def __init__(self, real, raises: Exception | None = None) -> None:
        self._real = real
        self._raises = raises
        self.cuda = _FakeCuda()

    def __getattr__(self, name: str):
        return getattr(self._real, name)

    def randn(self, *shape, device=None, dtype=None):
        if self._raises is not None:
            raise self._raises
        return self._real.randn(*shape, device="cpu", dtype=dtype)


def test_kernel_section_fails_loudly_on_the_kernel_image_error() -> None:
    """The failure docs/local_gpu.md §0's old one-liner aimed at and could not see. Every probe
    must report it."""
    torch = pytest.importorskip("torch")
    err = RuntimeError("CUDA error: no kernel image is available for execution on the device")
    fake = _FakeTorch(torch, raises=err)
    report = pf.Report(echo=False)
    report.section("3. kernel launch proof")

    pf.section_kernels(report, fake, {"cuda_available": True, "capability": [12, 0]})

    assert len(report.failed) == len(pf.kernel_probes())
    assert report.failed[0] == "kernel.bf16_matmul"
    fixes = [c.fix for c in report.checks]
    assert all("cu128" in f for f in fixes)
    assert all("sm_120" in f for f in fixes)
    assert report.info["kernels"]["ran"] is True


def test_kernel_section_passes_and_synchronizes_when_the_kernels_run() -> None:
    torch = pytest.importorskip("torch")
    fake = _FakeTorch(torch)
    report = pf.Report(echo=False)
    report.section("3. kernel launch proof")

    pf.section_kernels(report, fake, {"cuda_available": True, "capability": [12, 0]})

    assert report.failed == []
    # a launch that is never synchronized proves nothing — the async queue can swallow the error
    assert fake.cuda.synchronized == len(pf.kernel_probes())
    assert all(op["ok"] for op in report.info["kernels"]["ops"].values())


def test_kernel_section_says_so_instead_of_pretending_when_there_is_no_gpu() -> None:
    report = pf.Report(echo=False)
    report.section("3. kernel launch proof")

    pf.section_kernels(report, None, {"cuda_available": False})

    assert report.failed == []
    assert report.warned == ["kernels.skipped"]
    assert report.info["kernels"]["ran"] is False


def test_kernel_probes_run_on_cpu_and_produce_finite_output() -> None:
    """Not a GPU proof — a proof that the probe bodies themselves are correct.

    If one of these were shaped wrong it would fail on the 5090 for the wrong reason and get
    blamed on the wheel, which is exactly the confusion this script exists to remove.
    """
    torch = pytest.importorskip("torch")
    for name, fn in pf.kernel_probes():
        out = fn(torch, "cpu")
        assert torch.isfinite(out).all(), name
        assert float(out.float().std()) > 0.0, name


# ---------------------------------------------------------------------------------------------
# memory


def test_parse_meminfo_total_gb() -> None:
    text = "MemTotal:       65856248 kB\nMemFree:        12345678 kB\n"
    assert pf.parse_meminfo_total_gb(text) == pytest.approx(67.44, abs=0.01)


def test_parse_meminfo_returns_none_without_the_field() -> None:
    assert pf.parse_meminfo_total_gb("SwapTotal: 12 kB\n") is None
    assert pf.parse_meminfo_total_gb("") is None


def test_host_ram_floor_is_above_the_wan_checkpoint_size() -> None:
    """The floor exists because ~22-24 GB of weights land in host RAM before the card sees them."""
    assert pf.HOST_RAM_FLOOR_GB >= 32.0
    assert pf.HOST_RAM_FLOOR_GB > pf.WAN_SMOKE_PEAK_GB


def test_memory_section_fails_a_box_that_cannot_hold_the_weights(monkeypatch) -> None:
    """16 GB of RAM cannot materialize a 22-24 GB checkpoint, and device_map is out of reach."""
    monkeypatch.setattr(pf, "host_ram_gb", lambda: (16.0, "fake"))
    report = pf.Report(echo=False)

    pf.section_memory(report, None, {"cuda_available": False})

    assert "host.ram" in report.failed
    fix = next(c.fix for c in report.checks if c.name == "host.ram")
    assert "device_map" in fix


def test_memory_section_passes_a_box_with_enough_ram(monkeypatch) -> None:
    monkeypatch.setattr(pf, "host_ram_gb", lambda: (67.4, "fake"))
    report = pf.Report(echo=False)

    memory = pf.section_memory(report, None, {"cuda_available": False})

    assert report.failed == []
    assert memory["host_ram_gb"] == 67.4
    assert memory["vram_total_gb"] is None


def test_host_ram_gb_returns_a_number_and_a_source_on_this_machine() -> None:
    ram, source = pf.host_ram_gb()
    assert source != "unavailable", "neither /proc/meminfo nor sysconf worked here"
    assert ram is not None and ram > 1.0


# ---------------------------------------------------------------------------------------------
# assets


def test_backbone_source_rejects_a_dir_missing_the_towers(tmp_path: Path) -> None:
    (tmp_path / "transformer").mkdir()
    found = pf.inspect_backbone_source(tmp_path)

    assert found["exists"] is True
    assert found["missing"] == ["vae", "text_encoder"]


def test_backbone_source_accepts_a_complete_snapshot(tmp_path: Path) -> None:
    for sub in pf.BACKBONE_SUBDIRS:
        (tmp_path / sub).mkdir()
    (tmp_path / "transformer" / "shard.safetensors").write_bytes(b"x" * 2048)

    found = pf.inspect_backbone_source(tmp_path)

    assert found["missing"] == []
    assert found["size_bytes"] == 2048
    assert found["size_gb"] == 0.0  # rounded for humans; a real snapshot is ~20 GB


def test_backbone_source_on_a_missing_dir(tmp_path: Path) -> None:
    found = pf.inspect_backbone_source(tmp_path / "nope")
    assert found["exists"] is False
    assert found["missing"] == list(pf.BACKBONE_SUBDIRS)


def test_dataset_counts_episode_dirs_by_manifest(tmp_path: Path) -> None:
    for name in ("ep-0000", "ep-0001"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "manifest.json").write_text("{}")
    (tmp_path / "not-an-episode").mkdir()

    found = pf.inspect_dataset(tmp_path)

    assert found["exists"] is True
    assert found["episodes"] == 2


def test_dataset_counts_the_real_mock_dataset() -> None:
    found = pf.inspect_dataset(_REPO_ROOT / "datasets" / "mock-d1")
    if not found["exists"]:  # datasets/ is not always present in a fresh clone
        pytest.skip("datasets/mock-d1 not checked out")
    assert found["episodes"] >= 1


def test_checkpoint_finds_the_weights(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"x" * 1024)
    (tmp_path / "trainer_state.pt").write_bytes(b"y")

    found = pf.inspect_checkpoint(tmp_path)

    assert found["weights"] == "model.safetensors"
    assert found["resumable"] is True


def test_checkpoint_without_weights_is_reported(tmp_path: Path) -> None:
    found = pf.inspect_checkpoint(tmp_path)
    assert found["exists"] is True
    assert found["weights"] is None


# ---------------------------------------------------------------------------------------------
# end to end, on this CPU-only machine


def test_main_runs_and_writes_a_json_report(tmp_path: Path, capsys) -> None:
    out = tmp_path / "preflight.json"
    code = pf.main(["--json", str(out), "--card-gb", "34.36"])

    payload = json.loads(out.read_text())
    assert isinstance(code, int)
    assert payload["ok"] == (code == 0)
    assert {c["section"] for c in payload["checks"]} >= {
        "1. environment",
        "2. torch",
        "3. kernel launch proof",
        "4. memory",
        "5. python dependencies",
        "7. VRAM budget on this card",
    }
    assert payload["info"]["budget"]["card_total_gb"] == 34.36
    assert len(payload["info"]["budget"]["rows"]) == len(pf.budget_entries())
    assert payload["info"]["budget"]["load_transient_warning"]
    assert "budget.load_transient" in {c["name"] for c in payload["checks"]}
    # the budget table is printed, not buried in the JSON
    assert "runs/smoke/183599" in capsys.readouterr().out


def test_main_reports_a_missing_backbone_source_as_a_failure(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    code = pf.main(["--backbone-source", str(tmp_path / "absent"), "--json", str(out), "--quiet"])

    payload = json.loads(out.read_text())
    names = {c["name"]: c["status"] for c in payload["checks"]}
    assert names["assets.backbone_source"] == "FAIL"
    assert code == 1


def test_main_with_a_card_that_cannot_hold_dream(tmp_path: Path) -> None:
    """A 24 GB card must be told that dream.py will not run on it."""
    out = tmp_path / "r.json"
    pf.main(["--card-gb", "24.0", "--json", str(out), "--quiet"])

    payload = json.loads(out.read_text())
    verdicts = {r["entry_point"]: r["verdict"] for r in payload["info"]["budget"]["rows"]}
    assert verdicts["scripts/dream.py"] == "WILL OOM"


def test_quiet_leaves_stdout_as_parseable_json(capsys) -> None:
    """--quiet is the machine-readable mode; a stray table on stdout makes it unparseable."""
    pf.main(["--quiet", "--card-gb", "24.0"])

    payload = json.loads(capsys.readouterr().out)
    assert "checks" in payload
    assert payload["info"]["budget"]["card_total_gb"] == 24.0
    # the table is still delivered, just inside the payload rather than beside it
    assert "DECIMAL GB" in payload["info"]["budget"]["table"]


def test_parse_args_defaults_keep_every_new_flag_opt_in() -> None:
    args = pf.parse_args([])
    assert isinstance(args, argparse.Namespace)
    assert args.backbone_source is None
    assert args.dataset is None
    assert args.checkpoint is None
    assert args.card_gb is None
    assert args.deep_import is False
    assert args.json_out is None
    assert args.quiet is False
