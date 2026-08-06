"""Tests for scripts/compare_backbones.py — the Wan-vs-Cosmos head-to-head driver (T-38).

Nothing here touches the network. The two ZeroGPU Spaces are exercised only through their
artifacts: a probe report is a JSON document, and every claim this driver makes is a function of
two of them plus a baselines artifact.

What these pin is the part that can be wrong *silently*. A driver that compared two backbones
scored on different windows would print a clean table with a meaningless delta in it, and a
width-matched arm that was not reproducible from its seed would make a lucky projection look like
a finding — which is exactly the mistake T-37 recorded. So the agreement check gets one test per
way the two runs can differ, including the ones the reports do not shout about (a field present in
one report and absent in the other, two different dataset snapshots).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    """Import a script by path. Registered in ``sys.modules`` first — ``@dataclass`` resolves its
    own module out of there, and a script loaded without it dies at class-creation time."""
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cb = _load("compare_backbones")

SNAPSHOT = "/cache/datasets--nvidia--GR00T-N1.7-AppleToPlate/snapshots/d89c126a71"
FRAMES, RESIZE, CHUNK_STEPS = 3, [64, 64], 4
#: The two comparators the fixture artifact supports, and they are deliberately different rows:
#: on the real 12-episode artifact the val-argmax and the test-argmax are different feature sets
#: 0.029 joints apart, and a fixture where they coincide cannot tell the two protocols apart.
BEST_ON_VAL = "past_joint_proj_s1_plus_state"
BEST_ON_TEST = "past_ee_plus_state"
CFG = {"chunk_steps": CHUNK_STEPS}
NUM_JOINTS = 15
STATE_DIM = 2 * NUM_JOINTS + 2
LABEL_DIM = CHUNK_STEPS * (NUM_JOINTS + 1)
JOINT_DIM = CHUNK_STEPS * NUM_JOINTS


def _report(
    *,
    episodes: int = 8,
    feature_dim: int = 3072,
    num_layers: int = 30,
    blocks: tuple[int, int] = (2, 10),
    joints: float = 0.3652,
    gripper: float = 0.6976,
    best_block_joints: float = 0.5,
    split: dict[str, list[int]] | None = None,
    dataset: str = SNAPSHOT,
    windows: int | None = None,
    extra_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A probe report with the fields this driver reads, shaped like ``runs/wan_probe/*.json``."""
    episode_list = list(range(episodes))
    n_windows = windows if windows is not None else episodes * 8
    split = split or {
        "train": episode_list[:-4],
        "val": episode_list[-4:-2],
        "test": episode_list[-2:],
    }
    data = {
        "dataset": dataset,
        "episodes": episode_list,
        "windows": n_windows,
        "frames": FRAMES,
        "resize": list(RESIZE),
        "chunk_steps": CHUNK_STEPS,
        "label_dim": LABEL_DIM,
        "instruction": "move the apple to the plate",
    }
    data.update(extra_data or {})
    pair = f"suggested_{blocks[0]}_{blocks[1]}"
    return {
        "ok": True,
        "checks": [
            {
                "name": "probe.features_finite",
                "ok": True,
                "detail": f"({n_windows}, {num_layers}, {feature_dim})",
            }
        ],
        "info": {
            "geometry": {"num_layers": num_layers, "feature_dim": feature_dim},
            "timings": {"forwards_s": 7.2, "per_window_s": 0.075, "peak_vram_gb": 24.61},
            "data": data,
            "probe": {
                "split_episodes": split,
                "suggested_blocks": list(blocks),
                "per_block": {
                    "0": {
                        "joints": {"test_r2": best_block_joints, "val_r2": 0.1},
                        "gripper": {"test_r2": 0.9, "val_r2": 0.1},
                    },
                    "1": {
                        "joints": {"test_r2": 0.01, "val_r2": 0.4},
                        "gripper": {"test_r2": 0.02, "val_r2": 0.4},
                    },
                },
                "candidates": {
                    "measured_20_29": {
                        "joints": {"test_r2": 0.9, "val_r2": 0.9},
                        "gripper": {"test_r2": 0.9, "val_r2": 0.9},
                    },
                    pair: {
                        "joints": {"test_r2": joints, "val_r2": 0.4},
                        "gripper": {"test_r2": gripper, "val_r2": 0.7},
                    },
                    "state_only": {
                        "joints": {"test_r2": 0.4563, "val_r2": 0.547},
                        "gripper": {"test_r2": 0.8812, "val_r2": 0.887},
                    },
                },
            },
        },
    }


def _baselines_doc(episodes: int = 8, windows: int | None = None) -> dict[str, Any]:
    """A ``probe_action_baselines`` artifact for the same windows.

    The rows are the real 12-episode artifact's, kept in their measured relation to each other:
    the val-argmax and the test-argmax are different feature sets, ``past_joint`` is the weakest
    on joints and the strongest on gripper, and the two excluded rows (floor, control) would both
    win on val if the exclusions ever stopped working. Four ways for the selection to be wrong,
    each visible as a different feature set coming back.
    """
    return {
        "windows": {
            "dataset": "data/raw/gr00t_apple",
            "episodes": list(range(episodes)),
            "windows": windows if windows is not None else episodes * 8,
            "window_select": "linspace",
            "frames": FRAMES,
            "resize": list(RESIZE),
            "chunk_steps": CHUNK_STEPS,
        },
        "results": {
            "state_only": {
                "joints": {"test_r2": 0.4563, "val_r2": 0.547, "alpha": 10.0},
                "gripper": {"test_r2": 0.8812, "val_r2": 0.887, "alpha": 10.0},
                "dim": 32,
            },
            "past_ee_plus_state": {
                "joints": {"test_r2": 0.5407, "val_r2": 0.4922, "alpha": 10.0},
                "gripper": {"test_r2": 0.8424, "val_r2": 0.9, "alpha": 10.0},
                "dim": 144,
            },
            "past_joint_proj_s1_plus_state": {
                "joints": {"test_r2": 0.5118, "val_r2": 0.5465, "alpha": 10.0},
                "gripper": {"test_r2": 0.8100, "val_r2": 0.8, "alpha": 10.0},
                "dim": 144,
            },
            "past_joint": {
                "joints": {"test_r2": -0.0950, "val_r2": 0.41, "alpha": 10.0},
                "gripper": {"test_r2": 0.9999, "val_r2": 0.5, "alpha": 10.0},
                "dim": 256,
            },
            "past_ee_shuffled": {
                "joints": {"test_r2": 0.9999, "val_r2": 0.9, "alpha": 10.0},
                "gripper": {"test_r2": 0.9999, "val_r2": 0.9, "alpha": 10.0},
                "dim": 112,
            },
        },
    }


def _baselines(doc: dict[str, Any] | None = None, path: str = "x/action_baselines.json") -> Any:
    """What ``load_baselines`` hands the table, assembled from a fixture artifact."""
    return cb._baselines_from(Path(path), doc or _baselines_doc())


def _fake_windows(
    episodes: int = 8, per_episode: int = 12, seed: int = 0
) -> list[dict[str, Any]]:
    """Windows whose joint label channels are a fixed linear image of the state, plus noise.

    The signal is deliberately real: a shuffled control can only be shown to destroy the pairing
    if there is a pairing to destroy.

    The **gripper** channels — everything past ``chunk_steps * 15`` — are pure noise, and that is
    what makes the two columns of every table row distinguishable. With signal in both, a joints
    number and a gripper number look alike, and swapping the two label blocks anywhere downstream
    is invisible. Here a swap reads as a joints column near zero.
    """
    rng = np.random.default_rng(seed)
    mixer = rng.standard_normal((STATE_DIM, JOINT_DIM)).astype(np.float32)
    windows = []
    for episode in range(episodes):
        for _ in range(per_episode):
            q = rng.standard_normal(NUM_JOINTS).astype(np.float32)
            dq = rng.standard_normal(NUM_JOINTS).astype(np.float32)
            grip = rng.standard_normal(2).astype(np.float32)
            state = np.concatenate([q, dq, grip])
            joint_label = state @ mixer + 0.05 * rng.standard_normal(JOINT_DIM).astype(np.float32)
            label = np.concatenate(
                [joint_label, rng.standard_normal(LABEL_DIM - JOINT_DIM).astype(np.float32)]
            )
            windows.append(
                {
                    "episode": episode,
                    "start": 20,
                    "label": label.astype(np.float32),
                    "state": SimpleNamespace(q=q, dq=dq, gripper_state=grip),
                }
            )
    return windows


def _args(**overrides: Any) -> Any:
    base = [
        "--frames", str(FRAMES),
        "--height", str(RESIZE[0]),
        "--width", str(RESIZE[1]),
        "--chunk-steps", str(CHUNK_STEPS),
    ]  # fmt: skip
    args = cb.parse_args(base)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# ---- the agreement check ---------------------------------------------------------------------


def test_two_reports_built_on_the_same_windows_pass_the_agreement_check() -> None:
    cosmos = _report(feature_dim=4096, num_layers=36, blocks=(11, 24))
    reports = {"wan": _report(), "cosmos": cosmos}
    cfg = cb.assert_same_windows(reports, cb.requested_config(_args(), 8))
    assert cfg["episodes"] == list(range(8))
    assert cfg["split_episodes"]["test"] == [6, 7]


def test_a_different_episode_list_between_the_two_reports_is_a_mismatch() -> None:
    reports = {"wan": _report(episodes=8), "cosmos": _report(episodes=12)}
    with pytest.raises(cb.WindowMismatch, match="episodes"):
        cb.assert_same_windows(reports)


def test_a_different_train_val_test_split_is_a_mismatch_even_when_the_episode_list_agrees() -> None:
    """The defect the episode list alone cannot see: same corpus, different held-out episodes."""
    other = {"train": [0, 1, 2, 3, 4], "val": [5], "test": [6, 7]}
    reports = {"wan": _report(), "cosmos": _report(split=other)}
    episodes = [report["info"]["data"]["episodes"] for report in reports.values()]
    assert episodes[0] == episodes[1], "the fixture must differ only in the split"
    with pytest.raises(cb.WindowMismatch, match="split_episodes"):
        cb.assert_same_windows(reports)


def test_a_field_present_in_one_report_and_absent_in_the_other_is_a_mismatch() -> None:
    """Two reports agree about ``window_select`` only if both omit it — one-sided is unknown."""
    reports = {"wan": _report(), "cosmos": _report(extra_data={"window_select": "motion"})}
    with pytest.raises(cb.WindowMismatch, match="window_select"):
        cb.assert_same_windows(reports)


def test_the_mismatch_message_names_the_flag_that_would_state_the_missing_field() -> None:
    """The two deployed Spaces carry different vintages of the probe, so this is the mismatch that
    actually fires in practice. An unactionable failure would just get the check deleted."""
    reports = {"wan": _report(extra_data={"window_select": "linspace"}), "cosmos": _report()}
    with pytest.raises(cb.WindowMismatch, match="--assume-default window_select"):
        cb.assert_same_windows(reports)


def test_an_assumed_default_reconciles_a_field_only_one_report_records() -> None:
    reports = {"wan": _report(extra_data={"window_select": "linspace"}), "cosmos": _report()}
    cfg = cb.assert_same_windows(reports, None, ("window_select",))
    assert cfg["window_select"] == "linspace"
    assert cfg["assumed_defaults"]["window_select"] == {
        "value": "linspace",
        "assumed_for": ["cosmos"],
    }, "the assumption has to travel into the artifact, not stay in the shell history"


def test_a_recorded_value_that_is_not_the_default_cannot_be_assumed_away() -> None:
    """``motion`` selects the highest-motion windows: a different subpopulation, not a default."""
    reports = {"wan": _report(extra_data={"window_select": "motion"}), "cosmos": _report()}
    with pytest.raises(cb.WindowMismatch, match="window_select"):
        cb.assert_same_windows(reports, None, ("window_select",))


def test_nothing_is_assumed_when_the_reports_that_do_record_the_field_disagree() -> None:
    """Three reports, two of which recorded a value and disagree. There is no single value to
    stand in for the third, and picking one would be inventing the answer."""
    reports = {
        "wan": _report(extra_data={"window_select": "linspace"}),
        "cosmos": _report(),
        "other": _report(extra_data={"window_select": "motion"}),
    }
    with pytest.raises(cb.WindowMismatch, match="window_select"):
        cb.assert_same_windows(reports, None, ("window_select",))


def test_only_fields_with_a_documented_default_can_be_assumed() -> None:
    configs = {"wan": cb.window_config(_report()), "cosmos": cb.window_config(_report())}
    with pytest.raises(ValueError, match="instruction"):
        cb.apply_assumed_defaults(configs, ("instruction",))


def test_two_spaces_that_read_different_dataset_snapshots_are_a_mismatch() -> None:
    other = SNAPSHOT.replace("d89c126a71", "0000000000")
    reports = {"wan": _report(), "cosmos": _report(dataset=other)}
    with pytest.raises(cb.WindowMismatch, match="dataset revision"):
        cb.assert_same_windows(reports)


def test_a_local_mirror_path_is_not_read_as_a_snapshot_revision() -> None:
    """A local run and a Space run cannot be checked for corpus identity, so it is not claimed."""
    assert cb.dataset_revision(_report()) == "d89c126a71"
    assert cb.dataset_revision(_report(dataset="data/raw/gr00t_apple")) is None
    reports = {"wan": _report(), "cosmos": _report(dataset="data/raw/gr00t_apple")}
    cb.assert_same_windows(reports)


def test_the_returned_windows_are_checked_against_what_was_requested() -> None:
    """Both Spaces can agree with each other and still not have run what was asked for."""
    reports = {"wan": _report(episodes=8), "cosmos": _report(episodes=8)}
    with pytest.raises(cb.WindowMismatch, match="requested"):
        cb.assert_same_windows(reports, cb.requested_config(_args(), 24))


def test_the_agreement_check_reports_every_disagreement_and_not_just_the_first() -> None:
    reports = {
        "wan": _report(),
        "cosmos": _report(episodes=12, extra_data={"instruction": "something else"}),
    }
    with pytest.raises(cb.WindowMismatch) as excinfo:
        cb.assert_same_windows(reports)
    message = str(excinfo.value)
    assert "episodes" in message
    assert "instruction" in message
    assert "split_episodes" in message


def test_comparing_a_single_report_against_nothing_is_refused() -> None:
    with pytest.raises(cb.WindowMismatch):
        cb.assert_same_windows({"wan": _report()})


# ---- reading a report ------------------------------------------------------------------------


def test_the_headline_row_is_the_val_selected_pair_not_the_best_number_in_the_file() -> None:
    """``measured_20_29`` scores 0.9 here and is not the headline: it is a fixed block pair that
    means different depths in the two Spaces, and the best single block is chosen on test."""
    row = cb.backbone_row(_report(joints=0.3652, best_block_joints=0.88))
    assert row["candidate"] == "suggested_2_10"
    assert row["joints"] == pytest.approx(0.3652)
    assert row["best_single_block"]["joints"] == pytest.approx(0.88)


def test_the_feature_shape_and_width_come_from_the_measured_check() -> None:
    row = cb.backbone_row(_report(episodes=8, feature_dim=4096, num_layers=36, blocks=(11, 24)))
    assert row["feature_shape"] == [64, 36, 4096]
    assert row["dim"] == 2 * 4096, "a two-block candidate is twice the residual width"


def test_the_shape_is_still_found_when_the_wan_space_prefixes_it_with_the_readout_name() -> None:
    """Wan writes ``mean(96, 30, 3072)`` and Cosmos writes ``(96, 36, 4096)``. Same field, two
    formats, and reading only one of them would make the width of one backbone unknown."""
    prefixed = _report()
    prefixed["checks"][0]["detail"] = "mean(96, 30, 3072)"
    assert cb.feature_shape(prefixed) == [96, 30, 3072]


def test_a_report_whose_ridge_never_ran_is_refused_rather_than_scored() -> None:
    broken = _report()
    broken["info"].pop("probe")
    with pytest.raises(cb.ProbeFailed):
        cb.backbone_row(broken)


def test_the_model_id_and_host_are_read_back_out_of_the_space_log() -> None:
    log = 'host: {"space_id": "huhn511/wam-wan-smoke", "ram_gb": 104.0}\nmodel: nvidia/Cosmos3-Nano'
    header = cb.parse_log_header(log)
    assert header["model_id"] == "nvidia/Cosmos3-Nano"
    assert header["host"]["space_id"] == "huhn511/wam-wan-smoke"


# ---- the comparators -------------------------------------------------------------------------


def test_the_best_input_only_row_excludes_the_floor_and_every_shuffled_control() -> None:
    """``past_ee_shuffled`` is 0.9999 in this fixture on purpose: a control must never be able to
    become the bar a backbone is asked to clear. Neither may the floor, which wins on val here."""
    results = _baselines_doc()["results"]
    name, _ = cb.best_input_only(results)
    assert name == BEST_ON_VAL
    assert results["state_only"]["joints"]["val_r2"] > results[name]["joints"]["val_r2"]
    assert {n for n, _ in cb.input_only_rows(results)} == {
        BEST_ON_VAL,
        BEST_ON_TEST,
        "past_joint",
    }


def test_the_comparator_is_selected_on_validation_like_the_backbone_rows_are() -> None:
    """Both sides of the published table have to be selected the same way. The backbone row is
    each report's val-selected block pair, so an argmax on test here would put 0.029 joints of
    selection optimism (the measured 12-episode gap) in the column it is compared against."""
    results = _baselines_doc()["results"]
    name, row = cb.best_input_only(results)
    assert name == BEST_ON_VAL
    assert row["joints"]["test_r2"] == pytest.approx(0.5118)
    assert row["gripper"]["test_r2"] == pytest.approx(0.8100), "the row travels whole"

    optimistic_name, optimistic = cb.best_input_only_on_test(results)
    assert optimistic_name == BEST_ON_TEST
    assert optimistic["joints"]["test_r2"] == pytest.approx(0.5407)
    assert optimistic["joints"]["test_r2"] > row["joints"]["test_r2"], (
        "the fixture must keep the two selections apart, or neither is pinned"
    )


def test_the_weakest_input_only_row_never_becomes_the_bar() -> None:
    """``past_joint`` is −0.095 on joints and the best gripper in the file. An argmax that slipped
    to min, or to the gripper column, would quote it — and 'both backbones lose to the bar' is
    measured against whatever this returns."""
    name, _ = cb.best_input_only(_baselines_doc()["results"])
    assert name != "past_joint"
    assert cb.best_input_only_on_test(_baselines_doc()["results"])[0] != "past_joint"


def test_a_baselines_artifact_built_on_other_episodes_is_not_quoted(tmp_path: Path) -> None:
    """The 12-episode floor next to a 48-episode probe is the exact error T-37 recorded."""
    (tmp_path / "action_baselines.json").write_text(json.dumps(_baselines_doc(episodes=8)))
    args = _args(baselines_dir=str(tmp_path), data_dir=str(tmp_path / "nope"))
    floor = cb.load_baselines(cb.window_config(_report()), args)["floor"]
    assert floor["joints"]["test_r2"] == pytest.approx(0.4563)
    with pytest.raises(FileNotFoundError, match="--episodes 24"):
        cb.load_baselines(cb.window_config(_report(episodes=24)), args)


def test_a_floor_fitted_on_a_different_number_of_windows_is_not_quoted(tmp_path: Path) -> None:
    """Same episodes, same frames, same resize, same chunk — 16 windows per episode instead of 8.
    Every geometry field agrees and it is still a different window set, which is exactly the
    comparator-from-one-run-probe-from-another error T-37 recorded."""
    (tmp_path / "action_baselines.json").write_text(json.dumps(_baselines_doc()))
    args = _args(baselines_dir=str(tmp_path), data_dir=str(tmp_path / "nope"))
    assert cb.load_baselines(cb.window_config(_report()), args)["floor"]
    with pytest.raises(FileNotFoundError, match="'windows': 128"):
        cb.load_baselines(cb.window_config(_report(windows=128)), args)


def test_a_floor_fitted_with_the_other_window_selection_rule_is_not_quoted(tmp_path: Path) -> None:
    """``motion`` picks the highest-motion windows: same count, different subpopulation."""
    doc = _baselines_doc()
    doc["windows"]["window_select"] = "motion"
    (tmp_path / "action_baselines.json").write_text(json.dumps(doc))
    args = _args(baselines_dir=str(tmp_path), data_dir=str(tmp_path / "nope"))
    cfg = cb.window_config(_report(extra_data={"window_select": "linspace"}))
    with pytest.raises(FileNotFoundError, match="window_select"):
        cb.load_baselines(cfg, args)


def test_a_selection_rule_no_report_records_is_not_required_of_the_artifact(
    tmp_path: Path,
) -> None:
    """The deployed Cosmos Space predates ``window_select``. Requiring a field that is unknown on
    both sides would reject every artifact instead of checking anything."""
    (tmp_path / "action_baselines.json").write_text(json.dumps(_baselines_doc()))
    args = _args(baselines_dir=str(tmp_path), data_dir=str(tmp_path / "nope"))
    cfg = cb.window_config(_report())
    assert "window_select" not in cb.window_requirement(cfg)
    assert cb.load_baselines(cfg, args)["floor"]["joints"]["test_r2"] == pytest.approx(0.4563)


def test_the_floor_is_matched_to_the_reports_own_geometry_not_to_the_cli_flags(
    tmp_path: Path,
) -> None:
    """Offline, the flags describe nothing: a hand-fetched report carries the geometry it was run
    at. The decoy here has the right episodes and the wrong resize, and sorts first."""
    decoy = _baselines_doc()
    decoy["windows"]["resize"] = [999, 999]
    decoy["results"]["state_only"]["joints"]["test_r2"] = 0.1111
    (tmp_path / "action_baselines_a_decoy.json").write_text(json.dumps(decoy))
    (tmp_path / "action_baselines_match.json").write_text(json.dumps(_baselines_doc()))

    default_flags = cb.parse_args(["--baselines-dir", str(tmp_path), "--data-dir", "/nonexistent"])
    assert default_flags.frames != FRAMES, "the flags must disagree with the reports here"
    baselines = cb.load_baselines(cb.window_config(_report()), default_flags)
    assert Path(baselines["path"]).name == "action_baselines_match.json"
    assert baselines["floor"]["joints"]["test_r2"] == pytest.approx(0.4563)


def test_the_recompute_asks_for_the_reports_windows_and_not_for_the_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No artifact matches, so one is made. It has to be made on the reports' window set: 128
    windows over 8 episodes is 16 per episode, whatever ``--windows-per-episode`` says."""
    seen: dict[str, Any] = {}

    def fake_main(argv: list[str]) -> int:
        seen["argv"] = argv
        out = Path(argv[argv.index("--out") + 1])
        doc = _baselines_doc(windows=128)
        doc["windows"]["window_select"] = "motion"
        out.write_text(json.dumps(doc))
        return 0

    monkeypatch.setitem(sys.modules, "probe_action_baselines", SimpleNamespace(main=fake_main))
    args = _args(baselines_dir=str(tmp_path), data_dir=str(tmp_path), windows_per_episode=8)
    cfg = cb.window_config(_report(windows=128, extra_data={"window_select": "motion"}))
    baselines = cb.load_baselines(cfg, args)
    argv = seen["argv"]
    assert argv[argv.index("--windows-per-episode") + 1] == "16"
    assert argv[argv.index("--window-select") + 1] == "motion"
    assert argv[argv.index("--frames") + 1] == str(FRAMES)
    assert baselines["best_input_only"]["features"] == BEST_ON_VAL


def test_a_recompute_that_landed_on_other_windows_is_refused_rather_than_quoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asking for a window set and getting one are two statements. A local mirror that yields 64
    windows where the Spaces scored 128 would otherwise become the floor under the probes."""

    def fake_main(argv: list[str]) -> int:
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps(_baselines_doc(windows=64)))
        return 0

    monkeypatch.setitem(sys.modules, "probe_action_baselines", SimpleNamespace(main=fake_main))
    args = _args(baselines_dir=str(tmp_path), data_dir=str(tmp_path))
    with pytest.raises(cb.ProbeFailed, match="windows"):
        cb.load_baselines(cb.window_config(_report(windows=128)), args)


def test_a_baselines_run_that_exits_nonzero_is_not_read_back_as_an_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It exits before writing, so the file on disk is either absent or the previous run's."""
    stale = _baselines_doc()
    stale["results"]["state_only"]["joints"]["test_r2"] = 0.1111
    (tmp_path / "action_baselines_ep8.json").write_text(json.dumps(stale))
    monkeypatch.setitem(
        sys.modules, "probe_action_baselines", SimpleNamespace(main=lambda argv: 3)
    )
    args = _args(baselines_dir=str(tmp_path), data_dir=str(tmp_path))
    with pytest.raises(cb.ProbeFailed, match="exited 3"):
        cb.load_baselines(cb.window_config(_report(windows=128)), args)


# ---- the width control -----------------------------------------------------------------------


def test_a_projection_is_reproducible_from_its_seed_and_differs_between_seeds() -> None:
    x = np.random.default_rng(0).standard_normal((20, 300)).astype(np.float32)
    assert np.array_equal(cb.project(x, 32, seed=1), cb.project(x, 32, seed=1))
    assert not np.allclose(cb.project(x, 32, seed=1), cb.project(x, 32, seed=2))
    assert cb.project(x, 32, seed=1).shape == (20, 32)


def test_the_projection_matches_the_one_t37_measured_with() -> None:
    """Same construction as ``probe_action_baselines``: seed 100+s, scaled 1/sqrt(fan_in). If this
    drifts, a width-matched number here stops being comparable with the archived ones."""
    expected = np.random.default_rng(100 + 2).standard_normal((300, 32)).astype(np.float32)
    assert np.allclose(cb.random_projection(300, 32, seed=2), expected / np.sqrt(300))


def test_nuisance_padding_keeps_the_signal_and_adds_only_uninformative_columns() -> None:
    """The padded tensor has to be the same features plus junk, or the row measures two changes at
    once and neither can be attributed to width."""
    x = np.random.default_rng(0).standard_normal((20, 8)).astype(np.float32)
    padded = cb.nuisance_pad(x, 32, seed=1)
    assert padded.shape == (20, 32)
    assert np.array_equal(padded[:, :8], x), "the informative columns come first and unchanged"
    assert np.array_equal(padded, cb.nuisance_pad(x, 32, seed=1))
    assert not np.allclose(padded, cb.nuisance_pad(x, 32, seed=2))
    assert np.array_equal(cb.nuisance_pad(x, 4, seed=1), x), "no room to pad, nothing added"


def test_the_width_matched_arm_scores_every_seed_and_reports_the_spread() -> None:
    windows = _fake_windows()
    state, y_joint, y_grip, split = cb.window_tensors(windows, CHUNK_STEPS)
    arm = cb.score_at_width(state, y_joint, y_grip, split, (1.0, 10.0), 16, seeds=(0, 1, 2))
    assert [row["seed"] for row in arm["seeds"]] == [0, 1, 2]
    joints = [row["joints"] for row in arm["seeds"]]
    assert arm["joints_mean"] == pytest.approx(float(np.mean(joints)), abs=1e-4)
    assert arm["joints_spread"] == pytest.approx(max(joints) - min(joints), abs=1e-4)
    assert arm["width"] == 16


def test_the_windows_are_split_by_episode_so_no_episode_is_on_two_sides_of_it() -> None:
    """Leakage here inflates every number the width control produces, including the floor it
    checks itself against — and it would leak silently, because a leaky split scores *better*."""
    windows = _fake_windows()
    split = cb.window_tensors(windows, CHUNK_STEPS)[3]
    episode_of = np.asarray([w["episode"] for w in windows])
    train, val, test = (set(episode_of[split[k]].tolist()) for k in ("train", "val", "test"))
    assert train and val and test
    assert not (train & test) and not (train & val) and not (val & test)
    assert train | val | test == set(range(8))
    assert sum(len(split[k]) for k in ("train", "val", "test")) == len(windows)


def test_two_differently_wide_feature_sets_are_scored_at_one_common_width() -> None:
    """The whole point of the arm: Wan is 3072-dim per block and Cosmos 4096, so the raw pair of
    numbers is partly a comparison of tensor widths."""
    windows = _fake_windows()
    state = cb.window_tensors(windows, CHUNK_STEPS)[0]
    wide = cb.project(state, 400, seed=7)
    narrow = cb.project(state, 90, seed=7)
    args = _args(match_width=24, alphas="1,10")
    rows = {"wide": {"dim": 400, "blocks": [0]}, "narrow": {"dim": 90, "blocks": [0]}}
    control = cb.width_control(args, CFG, rows, {"wide": wide, "narrow": narrow}, windows)
    assert control["mode"] == "backbone_features"
    assert {arm["matched"]["width"] for arm in control["arms"].values()} == {24}
    assert control["arms"]["wide"]["native"]["width"] == 400
    assert control["arms"]["narrow"]["native"]["width"] == 90


def test_the_native_row_is_the_features_own_score_and_not_a_reprojection() -> None:
    """``native`` sits in the artifact next to the report's raw number and reads as identity. A
    square Gaussian is not identity: it rotates the features and ``probe_r2`` standardises per
    column afterwards, which moved 32-dim proprioception from 0.4563 to 0.5429 at 12 episodes."""
    windows = _fake_windows()
    x, y_joint, y_grip, split = cb.window_tensors(windows, CHUNK_STEPS)
    alphas = (1.0, 10.0)
    args = _args(match_width=24, alphas="1,10")
    rows = {"state": {"dim": STATE_DIM, "blocks": [0]}}
    control = cb.width_control(args, CFG, rows, {"state": x}, windows)
    native = control["arms"]["state"]["native"]
    assert native["width"] == STATE_DIM
    assert native["joints"] == pytest.approx(cb.wan.probe_r2(x, y_joint, split, alphas)["test_r2"])
    assert native["gripper"] == pytest.approx(cb.wan.probe_r2(x, y_grip, split, alphas)["test_r2"])
    square = cb.score_at_width(x, y_joint, y_grip, split, alphas, STATE_DIM, seeds=(0,))
    assert abs(native["joints"] - square["joints_mean"]) > 0.05, (
        "if a square projection were a no-op there would be nothing to pin here"
    )


def test_the_shuffled_control_destroys_the_window_to_label_pairing() -> None:
    windows = _fake_windows()
    state, y_joint, y_grip, split = cb.window_tensors(windows, CHUNK_STEPS)
    alphas = (1.0, 10.0, 100.0)
    honest = cb.score_at_width(state, y_joint, y_grip, split, alphas, 64, seeds=(0,))
    control = cb.shuffled_control(state, y_joint, y_grip, split, alphas)
    assert honest["joints_mean"] > 0.5, "the fixture must have signal for the control to remove"
    assert abs(control["joints_mean"]) < 0.1
    assert len(control["seeds"]) == 3, "one permutation is a draw from a null, not a null"


def test_the_control_reports_its_largest_permutation_and_not_its_mildest() -> None:
    """``joints_worst`` is what the table prints and what says the null has a tail: on the real
    12-episode windows one permutation reached +0.1344 on gripper while its siblings sat at
    −0.0038 and −0.0273. Reporting the mildest of the three would hide exactly that."""
    windows = _fake_windows()
    state, y_joint, y_grip, split = cb.window_tensors(windows, CHUNK_STEPS)
    control = cb.shuffled_control(state, y_joint, y_grip, split, (1.0, 10.0, 100.0))
    joints = [row["joints"] for row in control["seeds"]]
    grips = [row["gripper"] for row in control["seeds"]]
    assert min(abs(j) for j in joints) < max(abs(j) for j in joints), "seeds must differ in size"
    assert abs(control["joints_worst"]) == pytest.approx(max(abs(j) for j in joints), abs=5e-5)
    assert abs(control["gripper_worst"]) == pytest.approx(max(abs(g) for g in grips), abs=5e-5)
    assert min(abs(control["joints_worst"] - j) for j in joints) < 5e-5, "a seed, not a statistic"


def test_each_arms_shuffled_control_is_run_on_that_arms_own_features() -> None:
    """A control computed on the other backbone's tensor is the other backbone's null. This is the
    branch ``--features`` turns on, i.e. the one the width-matched claim would rest on."""
    windows = _fake_windows()
    state = cb.window_tensors(windows, CHUNK_STEPS)[0]
    args = _args(match_width=24, alphas="1,10")
    rows = {"wide": {"dim": 400, "blocks": [0]}, "narrow": {"dim": 90, "blocks": [0]}}
    features = {"wide": cb.project(state, 400, seed=7), "narrow": cb.project(state, 90, seed=7)}
    control = cb.width_control(args, CFG, rows, features, windows)
    assert control["arms"]["wide"]["shuffled"]["dim"] == 400
    assert control["arms"]["narrow"]["shuffled"]["dim"] == 90
    for arm in control["arms"].values():
        assert len(arm["shuffled"]["seeds"]) == 3, "one permutation is a draw from a null"
        assert len(arm["matched"]["seeds"]) == 3, "T-37's coin flip is what the seeds prevent"
        assert arm["shuffled"]["joints_mean"] < 0.1
        assert arm["matched"]["joints_mean"] > 0.4, "the honest arm must be far from its control"


def test_the_carried_state_arm_reproduces_the_floor_it_is_carried_from() -> None:
    """The unprojected score is the baselines artifact's ``state_only`` row. Equality is what says
    the locally rebuilt windows are the windows the floor was fitted on — so the number has to be
    checked against the state-only *joints* ridge, not against whatever the code produced."""
    windows = _fake_windows()
    state, y_joint, y_grip, split = cb.window_tensors(windows, CHUNK_STEPS)
    alphas = (1.0, 10.0)
    floor_joints = cb.wan.probe_r2(state, y_joint, split, alphas)["test_r2"]
    floor_gripper = cb.wan.probe_r2(state, y_grip, split, alphas)["test_r2"]
    assert floor_joints - floor_gripper > 0.4, "the fixture must tell the two columns apart"

    args = _args(match_width=16, alphas="1,10")
    rows = {"wan": {"dim": 96, "blocks": [0]}, "cosmos": {"dim": 128, "blocks": [0]}}
    floor = {"joints": {"test_r2": floor_joints}, "gripper": {"test_r2": floor_gripper}}
    control = cb.width_control(args, CFG, rows, {}, windows, floor)
    assert control["mode"] == "carried_state"
    assert control["unprojected"]["joints"] == pytest.approx(floor_joints)
    assert control["unprojected"]["gripper"] == pytest.approx(floor_gripper)
    assert control["reproduces_floor"]["agrees"]
    assert [arm["width"] for arm in control["carried"]] == [STATE_DIM, 96, 128, 16], (
        "the source width is carried too: a projection that does not change the width at all is "
        "what says how much of the rest is width and how much is the change of basis"
    )

    stale = {"joints": {"test_r2": floor_joints + 0.05}, "gripper": {"test_r2": floor_gripper}}
    stale_run = cb.width_control(args, CFG, rows, {}, windows, stale)
    assert not stale_run["reproduces_floor"]["agrees"]


def test_the_carried_rows_report_the_joint_channels_in_the_joints_column() -> None:
    """The fixture's gripper channels are noise, so a joints column that is really the gripper
    labels shows up as a row near zero — and every width penalty in the report is a difference
    between two numbers in this column."""
    windows = _fake_windows()
    args = _args(match_width=16, alphas="1,10")
    rows = {"wan": {"dim": 96, "blocks": [0]}}
    control = cb.width_control(args, CFG, rows, {}, windows)
    assert control["unprojected"]["joints"] > 0.5 > control["unprojected"]["gripper"]
    for arm in control["carried"] + control["carried_with_nuisance"]:
        assert arm["joints_mean"] > arm["gripper_mean"] + 0.2


def test_padding_a_signal_with_nuisance_columns_costs_what_projecting_it_does_not() -> None:
    """The projected rows cannot answer a width question: a rank-32 tensor keeps its row space
    under any projection, so 112 and 8192 dims hold the same information and scored 0.5586 vs
    0.5584 on the real windows. What a wide backbone tensor adds is directions the ridge has to
    regularise away, and only the padded rows have those."""
    windows = _fake_windows()
    args = _args(match_width=16, alphas="1,10")
    rows = {"wan": {"dim": 96, "blocks": [0]}, "cosmos": {"dim": 128, "blocks": [0]}}
    control = cb.width_control(args, CFG, rows, {}, windows)

    carried = {arm["width"]: arm["joints_mean"] for arm in control["carried"]}
    padded = {arm["width"]: arm["joints_mean"] for arm in control["carried_with_nuisance"]}
    assert sorted(padded) == [96, 128], "16 dims is narrower than the signal — nothing to pad"
    assert abs(carried[96] - carried[128]) < 0.05, "projection is flat in width, by construction"
    assert padded[96] < carried[96] - 0.1 and padded[128] < carried[128] - 0.1
    assert all(arm["informative_dims"] == 32 for arm in control["carried_with_nuisance"])


def test_the_label_split_follows_the_reports_chunk_length_and_not_the_flags() -> None:
    """``chunk_steps`` is what separates joint deltas from gripper synergies inside a label. Read
    off a stale flag it does not raise — it silently scores the gripper channels as joints."""
    windows = _fake_windows()
    args = _args(chunk_steps=99, match_width=16, alphas="1,10")
    rows = {"wan": {"dim": 96, "blocks": [0]}}
    control = cb.width_control(args, CFG, rows, {}, windows)
    assert control["unprojected"]["joints"] > 0.5, "the report's chunk length keeps 60 joint dims"
    assert control["unprojected"]["gripper"] is not None
    assert control["carried"][0]["gripper_mean"] is not None


def test_the_width_control_says_it_is_unavailable_instead_of_guessing() -> None:
    rows = {"wan": {"dim": 6144, "blocks": [0]}, "cosmos": {"dim": 8192, "blocks": [0]}}
    control = cb.width_control(_args(data_dir="/nonexistent"), CFG, rows, {}, None)
    assert control["mode"] == "unavailable"
    assert "/nonexistent" in control["reason"]
    assert control["widths"] == {"wan": 6144, "cosmos": 8192}


def test_the_local_rebuild_takes_its_window_count_and_rule_from_the_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--windows-per-episode`` is intent; 128 windows over 8 episodes is evidence. Rebuilding on
    the flag would fit the width rows and the floor on windows the probes never scored."""
    seen: dict[str, Any] = {}

    def fake_build(local: Any) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
        seen["per_episode"] = local.windows_per_episode
        seen["select"] = local.window_select
        seen["frames"] = local.frames
        return _fake_windows(episodes=8, per_episode=local.windows_per_episode), "instr", {}

    monkeypatch.setattr(cb.wan, "build_windows", fake_build)
    args = _args(data_dir=str(tmp_path), windows_per_episode=8)
    cfg = cb.window_config(_report(windows=128, extra_data={"window_select": "motion"}))
    windows = cb.build_windows_for(args, cfg)
    assert seen == {"per_episode": 16, "select": "motion", "frames": FRAMES}
    assert len(windows) == 128


def test_a_local_rebuild_that_lands_on_other_windows_than_the_reports_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same episodes and same flags can still yield a different number of windows — a truncated
    local mirror, a video one frame shorter. Silently, the width rows would then describe a
    different experiment than the probe rows above them in the same table."""
    monkeypatch.setattr(
        cb.wan, "build_windows", lambda local: (_fake_windows(episodes=8, per_episode=7), "i", {})
    )
    args = _args(data_dir=str(tmp_path))
    with pytest.raises(cb.WindowMismatch, match="rebuilt 56 windows"):
        cb.build_windows_for(args, cb.window_config(_report(windows=64)))


def test_no_local_corpus_means_no_local_windows_rather_than_an_error(tmp_path: Path) -> None:
    assert cb.build_windows_for(_args(data_dir=str(tmp_path / "absent")), CFG) is None


def test_features_given_as_a_block_stack_are_collapsed_onto_the_blocks_the_report_chose(
    tmp_path: Path,
) -> None:
    stack = np.arange(6 * 4 * 5, dtype=np.float32).reshape(6, 4, 5)
    path = tmp_path / "wan.npy"
    np.save(path, stack)
    rows = {"wan": {"blocks": [1, 3]}}
    features = cb.load_features([f"wan={path}"], rows)
    assert features["wan"].shape == (6, 10)
    assert np.array_equal(features["wan"][:, :5], stack[:, 1])
    assert np.array_equal(features["wan"][:, 5:], stack[:, 3])


# ---- the table -------------------------------------------------------------------------------


def test_the_table_carries_both_backbones_the_floor_and_the_comparator() -> None:
    rows = {
        "wan": {"blocks": [2, 10], "dim": 6144, "joints": 0.3652, "gripper": 0.6976},
        "cosmos": {"blocks": [11, 24], "dim": 8192, "joints": 0.3240, "gripper": 0.6126},
    }
    cfg = cb.window_config(_report())
    table = cb.render_table(
        8, cfg, rows, _baselines(), {"mode": "unavailable", "reason": "no corpus"}
    )
    assert "8 episodes" in table
    assert "64 windows" in table
    assert "split 4/2/2 episodes" in table
    for expected in ("wan · blocks 2,10", "0.3652", "cosmos · blocks 11,24", "0.3240"):
        assert expected in table
    assert "state-only floor" in table
    assert "0.4563" in table
    assert f"best input-only · {BEST_ON_VAL}" in table
    assert "0.5118" in table
    assert "width control unavailable" in table
    assert "assumed:" not in table, "nothing was assumed for this table"


def test_the_table_prints_the_test_selected_comparator_underneath_and_calls_it_optimistic() -> None:
    """Both numbers or neither: a reader with the artifact open can compute the test-argmax by eye
    and quote it as the bar, and on the real 12-episode rows that is 0.029 joints of optimism."""
    table = cb.render_table(
        8,
        cb.window_config(_report()),
        {},
        _baselines(),
        {"mode": "unavailable", "reason": "no corpus"},
    )
    assert f"best input-only · {BEST_ON_VAL}" in table
    assert "val-selected, as the backbone rows are" in table
    assert f"best on test · {BEST_ON_TEST}" in table
    assert "0.5407" in table
    assert "optimistic, not the bar" in table


def test_the_table_says_out_loud_which_field_was_assumed() -> None:
    reports = {"wan": _report(extra_data={"window_select": "linspace"}), "cosmos": _report()}
    cfg = cb.assert_same_windows(reports, None, ("window_select",))
    table = cb.render_table(
        8, cfg, {}, _baselines(), {"mode": "unavailable", "reason": "no corpus"}
    )
    assert "assumed: window_select='linspace' for cosmos" in table


def test_the_table_shows_the_seed_spread_of_every_projected_row() -> None:
    windows = _fake_windows()
    args = _args(match_width=16, alphas="1,10")
    rows = {"wan": {"dim": 96, "blocks": [0]}, "cosmos": {"dim": 128, "blocks": [0]}}
    width = cb.width_control(args, CFG, rows, {}, windows)
    table = cb.render_table(8, cb.window_config(_report()), {}, _baselines(), width)
    assert "carried state · width 96 · 3 seeds" in table
    assert "carried state · width 128 · 3 seeds" in table
    assert "carried state + nuisance · width 96 · 3 seeds" in table
    assert table.count("spread ") == 6, "four projected widths and the two that can be padded"
    assert "must sit near 0" in table


# ---- the offline path ---------------------------------------------------------------------


def test_from_reports_assembles_the_comparison_without_touching_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _forbidden(*_a: Any, **_k: Any) -> None:
        raise AssertionError("the offline path must not call a Space")

    monkeypatch.setattr(cb, "call_space", _forbidden)
    monkeypatch.setattr(cb, "resolve_token", _forbidden)

    wan_path = tmp_path / "wan_ep8.json"
    cosmos_path = tmp_path / "cosmos_ep8.json"
    wan_path.write_text(json.dumps(_report()))
    cosmos_path.write_text(
        json.dumps(_report(feature_dim=4096, num_layers=36, blocks=(11, 24), joints=0.3240))
    )
    (tmp_path / "action_baselines.json").write_text(json.dumps(_baselines_doc()))
    out = tmp_path / "compare.json"

    code = cb.main([
        "--from-reports", f"wan={wan_path}", f"cosmos={cosmos_path}",
        "--frames", str(FRAMES), "--height", str(RESIZE[0]), "--width", str(RESIZE[1]),
        "--chunk-steps", str(CHUNK_STEPS),
        "--baselines-dir", str(tmp_path), "--data-dir", str(tmp_path / "absent"),
        "--out", str(out),
    ])  # fmt: skip

    assert code == 0
    assert "wan · blocks 2,10" in capsys.readouterr().out
    doc = json.loads(out.read_text())
    assert doc["mode"] == "from-reports"
    assert doc["sizes"] == [8]
    result = doc["results"]["8"]
    assert result["backbones"]["wan"]["joints"] == pytest.approx(0.3652)
    assert result["backbones"]["cosmos"]["joints"] == pytest.approx(0.3240)
    assert result["backbones"]["cosmos"]["feature_shape"] == [64, 36, 4096]
    assert result["backbones"]["wan"]["timings"]["peak_vram_gb"] == 24.61
    assert result["backbones"]["wan"]["dataset_revision"] == "d89c126a71"
    assert result["baselines"]["best_input_only"]["features"] == BEST_ON_VAL
    assert result["width_control"]["mode"] == "unavailable"
    assert len(result["config_sha256"]) == 16
    assert result["table"].startswith("=== 8 episodes")


def test_from_reports_refuses_two_reports_covering_different_corpus_sizes(tmp_path: Path) -> None:
    small, large = tmp_path / "a.json", tmp_path / "b.json"
    small.write_text(json.dumps(_report(episodes=8)))
    large.write_text(json.dumps(_report(episodes=12)))
    with pytest.raises(cb.WindowMismatch, match="corpus sizes"):
        cb.collect_offline([[str(small), str(large)]])


def test_a_report_fetched_by_this_driver_carries_its_origin_into_an_offline_rerun(
    tmp_path: Path,
) -> None:
    """The Space id and the resolved model id are in the Gradio log, not in the report. Without
    the sidecar an offline reassembly would silently produce a less traceable artifact (AC-04)."""
    wan_path, cosmos_path = tmp_path / "wan_ep8.json", tmp_path / "cosmos_ep8.json"
    wan_path.write_text(json.dumps(_report()))
    cosmos_path.write_text(json.dumps(_report()))
    cb.meta_path(wan_path).write_text(
        json.dumps({"space": "huhn511/wam-wan-smoke", "model_id": "Wan-AI/x", "wall_s": 91.2})
    )
    runs = cb.collect_offline([[f"wan={wan_path}", f"cosmos={cosmos_path}"]])
    assert runs[8]["wan"]["meta"]["space"] == "huhn511/wam-wan-smoke"
    assert runs[8]["wan"]["meta"]["wall_s"] == 91.2
    assert runs[8]["cosmos"]["meta"]["space"] is None, "no sidecar means unknown, not inferred"


def test_a_bare_report_path_is_labelled_by_its_filename(tmp_path: Path) -> None:
    path = tmp_path / "cosmos3_ep48.json"
    path.write_text(json.dumps(_report()))
    label, resolved = cb._labelled_path(str(path))
    assert label == "cosmos3_ep48"
    assert resolved == path
    assert cb._labelled_path(f"wan={path}") == ("wan", path)


def test_hand_fetched_reports_are_not_held_to_flags_they_never_saw(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Offline there is nothing to hold a report to but the other report: a file downloaded from a
    Space UI carries whatever geometry it was run at, and the flags describe nothing. Every
    geometry flag is left at its default here, and every one of them disagrees with the fixture."""
    wan_path, cosmos_path = tmp_path / "wan_ep8.json", tmp_path / "cosmos_ep8.json"
    wan_path.write_text(json.dumps(_report()))
    cosmos_path.write_text(json.dumps(_report(feature_dim=4096, num_layers=36, blocks=(11, 24))))
    (tmp_path / "action_baselines.json").write_text(json.dumps(_baselines_doc()))
    defaults = cb.parse_args([])
    assert (defaults.frames, defaults.chunk_steps) != (FRAMES, CHUNK_STEPS)

    code = cb.main([
        "--from-reports", f"wan={wan_path}", f"cosmos={cosmos_path}",
        "--baselines-dir", str(tmp_path), "--data-dir", str(tmp_path / "absent"),
        "--out", str(tmp_path / "compare.json"),
    ])  # fmt: skip

    assert code == 0, capsys.readouterr().err
    assert "=== 8 episodes" in capsys.readouterr().out


def test_a_space_that_returns_something_other_than_what_was_asked_for_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Online the flags are evidence of intent and the Space is held to them. Both Spaces agreeing
    with each other proves nothing here: they would agree while both ignoring the request."""
    monkeypatch.setattr(cb, "resolve_token", lambda explicit: "token")
    monkeypatch.setattr(
        cb,
        "call_space",
        lambda spec, size, args, token: (_report(episodes=8), {"space": spec.space}),
    )
    (tmp_path / "action_baselines.json").write_text(json.dumps(_baselines_doc()))

    code = cb.main([
        "--sizes", "24",
        "--frames", str(FRAMES), "--height", str(RESIZE[0]), "--width", str(RESIZE[1]),
        "--chunk-steps", str(CHUNK_STEPS),
        "--baselines-dir", str(tmp_path), "--data-dir", str(tmp_path / "absent"),
        "--reports-dir", str(tmp_path / "reports"), "--out", str(tmp_path / "compare.json"),
    ])  # fmt: skip

    captured = capsys.readouterr()
    assert code == 2
    assert "requested" in captured.err
    assert "===" not in captured.out


# ---- calling a Space --------------------------------------------------------------------------


def _fake_gradio(log: str, report: Any) -> Any:
    """A ``gradio_client`` whose ``Client.predict`` yields one canned ``(log, report)`` pair."""

    class Client:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def predict(self, *_a: Any, **_k: Any) -> tuple[str, Any]:
            return log, report

    return SimpleNamespace(Client=Client)


def test_a_space_that_answers_with_a_failed_run_is_refused_rather_than_scored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash inside the handler is caught in the Space and comes back as ``{"ok": False}`` with
    an HTTP 200. Nothing raises, so the report has to be read before it is believed."""
    failed = {"ok": False, "error": "CUDA out of memory"}
    monkeypatch.setitem(sys.modules, "gradio_client", _fake_gradio("model: x\n", failed))
    with pytest.raises(cb.ProbeFailed, match="CUDA out of memory"):
        cb.call_space(cb.SPACES["wan"], 12, _args(), "token")


def test_a_successful_call_records_the_model_the_space_actually_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Space's ``MODEL_ID`` is a variable someone can change; the log says what ran."""
    log = 'model: nvidia/Cosmos3-Nano-v2\nhost: {"space_id": "huhn511/wam-cosmos3-probe"}'
    monkeypatch.setitem(sys.modules, "gradio_client", _fake_gradio(log, _report()))
    report, meta = cb.call_space(cb.SPACES["cosmos"], 12, _args(), "token")
    assert report["ok"] is True
    assert meta["declared_model_id"] == "nvidia/Cosmos3-Nano"
    assert meta["model_id"] == "nvidia/Cosmos3-Nano-v2"
    assert meta["space"] == "huhn511/wam-cosmos3-probe"


def test_a_nonzero_start_is_refused_because_the_spaces_always_begin_at_episode_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both probe tabs download ``range(episodes)``. Honouring ``--start 12`` remotely would
    compare episodes 12–23 against episodes 0–11, and the agreement check cannot see it: both
    Spaces make the same mistake, so their reports agree."""
    monkeypatch.setitem(sys.modules, "gradio_client", _fake_gradio("model: x\n", _report()))
    with pytest.raises(cb.SpaceUnavailable, match="--start 12"):
        cb.call_space(cb.SPACES["wan"], 12, _args(start=12), "token")


def test_without_a_token_the_driver_says_so_instead_of_calling_a_private_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(get_token=lambda: None))
    with pytest.raises(cb.SpaceUnavailable, match="--from-reports"):
        cb.resolve_token(None)
    assert cb.resolve_token("explicit") == "explicit", "an explicit token needs no login"


def test_a_mismatched_pair_of_reports_exits_nonzero_instead_of_printing_a_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wan_path, cosmos_path = tmp_path / "wan.json", tmp_path / "cosmos.json"
    wan_path.write_text(json.dumps(_report()))
    cosmos_path.write_text(json.dumps(_report(split={"train": [0, 1], "val": [2], "test": [3]})))
    (tmp_path / "action_baselines.json").write_text(json.dumps(_baselines_doc()))

    code = cb.main([
        "--from-reports", f"wan={wan_path}", f"cosmos={cosmos_path}",
        "--frames", str(FRAMES), "--height", str(RESIZE[0]), "--width", str(RESIZE[1]),
        "--chunk-steps", str(CHUNK_STEPS),
        "--baselines-dir", str(tmp_path), "--data-dir", str(tmp_path / "absent"),
        "--out", str(tmp_path / "compare.json"),
    ])  # fmt: skip

    captured = capsys.readouterr()
    assert code == 2
    assert "split_episodes" in captured.err
    assert "===" not in captured.out, "a void comparison must not print a table"
    assert not (tmp_path / "compare.json").exists()
