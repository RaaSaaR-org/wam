"""Tests for wam.interfaces.versioning (T-05, FR-10, AC-04)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from wam.interfaces.protocols import INTERFACES_VERSION
from wam.interfaces.schema import SCHEMA_VERSION, CanonicalSpaceSpec
from wam.interfaces.versioning import (
    WAM_CONFIG_VERSION,
    JsonlRunLogger,
    RunMetadata,
    config_hash,
    load_config,
    read_git_commit,
)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

FIXED_TIME = datetime(2026, 7, 26, 12, 0, 0, tzinfo=timezone.utc)


def fixed_clock() -> datetime:
    return FIXED_TIME


def make_metadata(config: dict[str, Any] | None = None, run_id: str = "run-001") -> RunMetadata:
    return RunMetadata.create(
        run_id,
        config if config is not None else {"lr": 1e-4, "seed": 7},
        checkpoint_ref="ckpt://test/0",
        dataset_snapshot_ref="ds://test/0",
        git_commit="deadbeef",
        clock=fixed_clock,
    )


class TestConfigHash:
    def test_stable_across_key_order(self) -> None:
        a = {"a": 1, "b": {"c": 2, "d": [1, 2, 3]}, "e": "x"}
        b = {"e": "x", "b": {"d": [1, 2, 3], "c": 2}, "a": 1}
        assert config_hash(a) == config_hash(b)

    def test_is_sha256_hex(self) -> None:
        assert re.fullmatch(r"[0-9a-f]{64}", config_hash({"a": 1}))

    def test_value_change_changes_hash(self) -> None:
        assert config_hash({"a": 1}) != config_hash({"a": 2})

    def test_nested_list_order_matters(self) -> None:
        assert config_hash({"a": [1, 2]}) != config_hash({"a": [2, 1]})

    def test_tuple_equals_list(self) -> None:
        assert config_hash({"a": (1, 2)}) == config_hash({"a": [1, 2]})

    def test_pydantic_model_equals_dump(self) -> None:
        spec = CanonicalSpaceSpec(joint_names=("j0", "j1"))
        assert config_hash(spec) == config_hash(spec.model_dump(mode="json"))

    def test_nested_pydantic_model(self) -> None:
        class Cfg(BaseModel):
            space: CanonicalSpaceSpec
            lr: float

        spec = CanonicalSpaceSpec(joint_names=("j0",))
        cfg = Cfg(space=spec, lr=1e-4)
        assert config_hash(cfg) == config_hash({"space": spec.model_dump(mode="json"), "lr": 1e-4})

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError):
            config_hash({"a": object()})


class TestReadGitCommit:
    def test_returns_string(self) -> None:
        commit = read_git_commit()
        assert isinstance(commit, str)
        assert commit == "unknown" or re.fullmatch(r"[0-9a-f]{40}", commit)

    def test_unknown_outside_repo(self, tmp_path: Path) -> None:
        assert read_git_commit(cwd=tmp_path) == "unknown"


class TestRunMetadata:
    def test_create_stamps_injected_clock(self) -> None:
        meta = make_metadata()
        assert meta.created_at == FIXED_TIME
        assert meta.git_commit == "deadbeef"
        assert meta.config_hash == config_hash({"lr": 1e-4, "seed": 7})
        assert meta.schema_version == SCHEMA_VERSION
        assert meta.interfaces_version == INTERFACES_VERSION

    def test_serialization_roundtrip(self) -> None:
        meta = make_metadata()
        restored = RunMetadata.model_validate_json(meta.model_dump_json())
        assert restored == meta

    def test_to_dict_is_json_safe(self) -> None:
        d = make_metadata().to_dict()
        parsed = json.loads(json.dumps(d))
        assert parsed["run_id"] == "run-001"
        assert parsed["checkpoint_ref"] == "ckpt://test/0"
        assert parsed["dataset_snapshot_ref"] == "ds://test/0"
        assert parsed["created_at"] == "2026-07-26T12:00:00Z"

    def test_frozen(self) -> None:
        meta = make_metadata()
        with pytest.raises(ValidationError):
            meta.run_id = "other"  # type: ignore[misc]

    def test_default_git_commit_read(self) -> None:
        meta = RunMetadata.create("r", {"a": 1}, clock=fixed_clock)
        assert isinstance(meta.git_commit, str) and meta.git_commit

    def test_run_metadata_round_trips_train_episode_ids_in_recorded_order(self) -> None:
        """The ids travel as safetensors metadata (a JSON string) and come back as a tuple.

        Order is load-bearing, not cosmetic: ``dataset_snapshot_ref`` is a sequential digest
        over the episodes in the order the trainer iterated them, and ``eval_t16.verify_split``
        replays exactly this list to reproduce it. A round trip that sorted, deduplicated or
        set-ified the field would break the proof rather than the formatting.
        """
        ids = ("gr00t-apple-000070", "gr00t-apple-000027", "gr00t-apple-000038")
        meta = RunMetadata.create(
            "rung-040",
            {"lr": 1e-4},
            dataset_snapshot_ref="ds://test/0",
            train_episode_ids=list(ids),
            git_commit="deadbeef",
            clock=fixed_clock,
        )
        assert meta.train_episode_ids == ids

        restored = RunMetadata.model_validate(json.loads(json.dumps(meta.to_dict())))
        assert restored.train_episode_ids == ids
        assert restored == meta

    def test_run_metadata_defaults_train_episode_ids_to_none_for_archived_records(self) -> None:
        """Every checkpoint written before I-8 lacks the key entirely.

        ``None`` is what selects the COMPLEMENT split proof in the evaluator, so this default is
        the reason ``runs/t16-lora-seed0`` re-scores under the rule it was recorded under
        instead of needing a migration.
        """
        archived = {
            "run_id": "t16-lora-seed0",
            "config_hash": "45ee9e6035eb6afe0721e33d807800a307b445cc73fca60b95111d913dae0d63",
            "git_commit": "78fc56d888a71088dac16b375bc9b54ebab33b0c",
            "schema_version": SCHEMA_VERSION,
            "interfaces_version": INTERFACES_VERSION,
            "checkpoint_ref": "/valhalla/.../step-020000.tmp/model.safetensors",
            "dataset_snapshot_ref": "sha256:598f193fcf1c160236688b3a7ade22ef6b33ad910a74ba4aa32c",
            "created_at": "2026-07-30T01:30:02.943441Z",
        }
        assert "train_episode_ids" not in archived
        assert RunMetadata.model_validate(archived).train_episode_ids is None

    def test_train_episode_ids_do_not_move_the_config_hash(self) -> None:
        """RunMetadata is not an input to config_hash, so adding the field cannot restate any
        archived experiment's identity. Pinned because a hash that moved would silently split
        one requeue chain into two experiments (AC-04)."""
        config = {"lr": 1e-4, "seed": 7}
        without = RunMetadata.create("r", config, git_commit="x", clock=fixed_clock)
        with_ids = RunMetadata.create(
            "r", config, train_episode_ids=["a", "b"], git_commit="x", clock=fixed_clock
        )
        assert without.config_hash == with_ids.config_hash == config_hash(config)


class TestJsonlRunLogger:
    def test_every_line_stamped(self, tmp_path: Path) -> None:
        meta = make_metadata()
        path = tmp_path / "run.jsonl"
        with JsonlRunLogger(path, meta) as logger:
            logger.log({"event": "start"})
            logger.log({"event": "step", "loss": 0.5})
            logger.log({"event": "end"})
        lines = path.read_text().splitlines()
        assert len(lines) == 3
        for line in lines:
            record = json.loads(line)
            assert record["run_id"] == meta.run_id
            assert record["config_hash"] == meta.config_hash

    def test_stamps_override_caller_keys(self, tmp_path: Path) -> None:
        meta = make_metadata()
        path = tmp_path / "run.jsonl"
        with JsonlRunLogger(path, meta) as logger:
            logger.log({"run_id": "spoofed", "config_hash": "spoofed"})
        record = json.loads(path.read_text().splitlines()[0])
        assert record["run_id"] == meta.run_id
        assert record["config_hash"] == meta.config_hash

    def test_append_only_across_sessions(self, tmp_path: Path) -> None:
        meta = make_metadata()
        path = tmp_path / "run.jsonl"
        with JsonlRunLogger(path, meta) as logger:
            logger.log({"event": "first"})
        with JsonlRunLogger(path, meta) as logger:
            logger.log({"event": "second"})
        events = [json.loads(line)["event"] for line in path.read_text().splitlines()]
        assert events == ["first", "second"]

    def test_log_metadata_record(self, tmp_path: Path) -> None:
        meta = make_metadata()
        path = tmp_path / "run.jsonl"
        with JsonlRunLogger(path, meta) as logger:
            logger.log_metadata()
        record = json.loads(path.read_text().splitlines()[0])
        assert record["kind"] == "run_metadata"
        assert record["checkpoint_ref"] == "ckpt://test/0"
        assert record["dataset_snapshot_ref"] == "ds://test/0"

    def test_log_when_closed_raises(self, tmp_path: Path) -> None:
        logger = JsonlRunLogger(tmp_path / "run.jsonl", make_metadata())
        with pytest.raises(RuntimeError):
            logger.log({"event": "x"})

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "run.jsonl"
        with JsonlRunLogger(path, make_metadata()) as logger:
            logger.log({"event": "x"})
        assert path.exists()


class TestLoadConfig:
    def test_missing_version_key(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("robot:\n  name: x\n")
        with pytest.raises(ValueError, match="wam_config_version"):
            load_config(path)

    def test_incompatible_major(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text('wam_config_version: "99.0.0"\n')
        with pytest.raises(ValueError, match="incompatible"):
            load_config(path)

    def test_non_mapping_document(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("- 1\n- 2\n")
        with pytest.raises(ValueError, match="mapping"):
            load_config(path)

    def test_same_major_different_minor_accepted(self, tmp_path: Path) -> None:
        major = WAM_CONFIG_VERSION.split(".")[0]
        path = tmp_path / "ok.yaml"
        path.write_text(f'wam_config_version: "{major}.999.0"\nx: 1\n')
        assert load_config(path)["x"] == 1


#: Every shipped robot config, DISCOVERED rather than listed. The hand-maintained
#: parametrize lists this replaced went stale the moment configs/robot/isaac_g1.yaml landed:
#: that file's own header claimed this module covered it while no entry here named it, so it
#: shipped with none of the checks below applied to it. A glob cannot silently omit the next
#: one; test_the_config_glob_found_the_configs_it_is_supposed_to_cover keeps it from silently
#: matching nothing instead.
ROBOT_CONFIGS = sorted(f"robot/{p.name}" for p in (CONFIGS_DIR / "robot").glob("*.yaml"))
#: The subset driving G1Adapter — every shipped robot config except MockRobot's.
G1_ROBOT_CONFIGS = [p for p in ROBOT_CONFIGS if p != "robot/mock.yaml"]


class TestShippedConfigs:
    def test_the_config_glob_found_the_configs_it_is_supposed_to_cover(self) -> None:
        """A glob that matched nothing would make every parametrized test in this class pass
        vacuously — the failure mode you trade for when you stop hand-listing. Naming the
        known configs catches a deletion or a rename without capping what may be added."""
        assert set(ROBOT_CONFIGS) >= {
            "robot/mock.yaml",
            "robot/g1.yaml",
            "robot/mujoco_g1.yaml",
            "robot/isaac_g1.yaml",
        }
        assert "robot/mock.yaml" not in G1_ROBOT_CONFIGS
        assert len(G1_ROBOT_CONFIGS) == len(ROBOT_CONFIGS) - 1

    @pytest.mark.parametrize("rel_path", [*ROBOT_CONFIGS, "safety/default.yaml"])
    def test_loads_with_valid_version(self, rel_path: str) -> None:
        cfg = load_config(CONFIGS_DIR / rel_path)
        assert cfg["wam_config_version"] == WAM_CONFIG_VERSION

    @pytest.mark.parametrize("rel_path", ROBOT_CONFIGS)
    def test_robot_config_consistency(self, rel_path: str) -> None:
        cfg = load_config(CONFIGS_DIR / rel_path)
        robot = cfg["robot"]
        spec = CanonicalSpaceSpec(
            joint_names=tuple(robot["canonical_space"]["joint_names"]),
            gripper_dims=robot["canonical_space"]["gripper_dims"],
        )
        limits = robot["limits"]
        n = spec.num_joints
        for key in ("q_min", "q_max", "dq_max", "ddq_max"):
            assert len(limits[key]) == n, f"{rel_path}: {key} length != {n}"
        assert all(lo < hi for lo, hi in zip(limits["q_min"], limits["q_max"]))
        assert all(v > 0 for v in limits["dq_max"])
        assert len(limits["gripper_min"]) == spec.gripper_dims
        assert len(limits["gripper_max"]) == spec.gripper_dims

    @pytest.mark.parametrize("rel_path", G1_ROBOT_CONFIGS)
    def test_g1_yaml_canonical_space_matches_adapter_spec(self, rel_path: str) -> None:
        """Coordination gate: every versioned G1 yaml and the hard-wired G1_SPEC/G1Config
        must describe the SAME canonical space — episodes, policies and calibration files
        built from the yaml would otherwise be unusable on the adapter. Every G1 config is
        covered: the hardware one and the two sim ones (MuJoCo and Isaac), which drive the
        same adapter through different transports."""
        from wam.robot.g1 import G1_SPEC, G1Config

        robot = load_config(CONFIGS_DIR / rel_path)["robot"]
        spec = CanonicalSpaceSpec(**robot["canonical_space"])
        assert spec == G1_SPEC  # names, order, gripper_dims, ee conventions
        limits = robot["limits"]
        cfg = G1Config(
            q_min=tuple(limits["q_min"]),
            q_max=tuple(limits["q_max"]),
            dq_max=tuple(limits["dq_max"]),
            control_dt_s=float(robot["control"]["dt_s"]),
        )
        assert cfg.q_min == tuple(limits["q_min"])
        assert len(limits["ddq_max"]) == G1_SPEC.num_joints
        assert len(limits["gripper_min"]) == G1_SPEC.gripper_dims

    def test_mujoco_g1_yaml_gains_are_sim_only_and_well_formed(self) -> None:
        """Sim gains must not leak into the hardware placeholders (OD-08), and kd must be the
        per-joint critical damping shape, never a flat number.

        The kd-shape assertion is MuJoCo-SPECIFIC and stays that way: those numbers are
        per-joint critical damping derived from the rig's measured inertias, which is why the
        spread is large. ``isaac_g1.yaml`` also ships gains and is checked separately, with a
        weaker rule, because Isaac Lab's published numbers are flat per body group."""
        from wam.robot.g1 import G1_SPEC, G1Config

        robot = load_config(CONFIGS_DIR / "robot/mujoco_g1.yaml")["robot"]
        gains = robot["gains"]
        assert len(gains["kp"]) == len(gains["kd"]) == G1_SPEC.num_joints
        assert all(v > 0 for v in gains["kp"] + gains["kd"])
        assert max(gains["kd"]) > 4.0 * min(gains["kd"]), "kd must be per-joint, not flat"
        hardware = load_config(CONFIGS_DIR / "robot/g1.yaml")["robot"]
        assert "gains" not in hardware, "sim gains must not appear in the hardware config"
        # It really does build the adapter config it claims to.
        limits = robot["limits"]
        G1Config(
            q_min=tuple(limits["q_min"]),
            q_max=tuple(limits["q_max"]),
            dq_max=tuple(limits["dq_max"]),
            kp=tuple(gains["kp"]),
            kd=tuple(gains["kd"]),
            control_dt_s=float(robot["control"]["dt_s"]),
        )

    def test_isaac_g1_yaml_gains_are_sim_only_well_formed_and_not_mistaken_for_measured(
        self,
    ) -> None:
        """The Isaac config's gains are the ones ``--robot isaac_g1`` actually loads, and
        NOBODY HAS MEASURED THEM on PhysX. They are Isaac Lab's published G1 magnitudes,
        roughly an order of magnitude above the MuJoCo rig's measured ones (waist 5000 vs
        500). This pins the shape and the provenance, not the values: asserting the numbers
        would only re-state the file, while asserting they differ from MuJoCo's is what stops
        someone "harmonising" the two and quietly implying one is evidence for the other.
        """
        from wam.robot.g1 import G1_SPEC, G1Config

        robot = load_config(CONFIGS_DIR / "robot/isaac_g1.yaml")["robot"]
        gains = robot["gains"]
        assert len(gains["kp"]) == len(gains["kd"]) == G1_SPEC.num_joints
        assert all(v > 0 for v in gains["kp"] + gains["kd"])
        hardware = load_config(CONFIGS_DIR / "robot/g1.yaml")["robot"]
        assert "gains" not in hardware, "sim gains must not appear in the hardware config"

        mujoco = load_config(CONFIGS_DIR / "robot/mujoco_g1.yaml")["robot"]["gains"]
        assert tuple(gains["kp"]) != tuple(mujoco["kp"]), (
            "the two sim backends' gains were made equal; if that was a measurement rather "
            "than a copy, replace this test with the measurement"
        )
        limits = robot["limits"]
        G1Config(
            q_min=tuple(limits["q_min"]),
            q_max=tuple(limits["q_max"]),
            dq_max=tuple(limits["dq_max"]),
            kp=tuple(gains["kp"]),
            kd=tuple(gains["kd"]),
            control_dt_s=float(robot["control"]["dt_s"]),
        )

    def test_mujoco_g1_yaml_track_window_is_sim_only_and_matches_the_module(self) -> None:
        """The bounded feed-forward window (T-25c) is sized from the SIM gains and must not
        leak into the hardware config, for the same reason the gains must not: at g1.yaml's
        kp=20/kd=0.5 placeholders the tracking error is ~0.17 rad, roughly 6x this window, so
        copying it across would clamp on every step and silently throttle the feed-forward.

        It must also exceed the measured 0.0299 rad steady-state tracking error at dq_max, or
        the clamp bites during normal fast motion — the failure mode that made 0.02 too small.
        """
        from wam.robot.g1 import G1_SPEC
        from wam.robot.mujoco_g1 import SIM_Q_TRACK_WINDOW

        robot = load_config(CONFIGS_DIR / "robot/mujoco_g1.yaml")["robot"]
        window = robot["control"]["q_track_window"]
        assert len(window) == G1_SPEC.num_joints
        assert all(w >= 0.0299 for w in window), "below the measured tracking error at dq_max"
        # The no-config get_robot("mujoco_g1") path must enforce the same window as this file.
        assert tuple(float(w) for w in window) == SIM_Q_TRACK_WINDOW
        hardware = load_config(CONFIGS_DIR / "robot/g1.yaml")["robot"]
        assert "q_track_window" not in hardware.get("control", {}), (
            "the sim window must not appear in the hardware config (OD-08)"
        )

    def test_rollouts_isaac_builder_reads_every_field_the_yaml_declares(self) -> None:
        """The same failure ``test_view_sim_honours_...`` was written for, one backend later.

        ``_build_isaac_g1`` maps the yaml onto ``IsaacG1Robot``'s constructor by hand, key by
        key, and a key it silently ignores is invisible: the robot still boots, with a
        different physics rate or on a different device than the file says. Nobody can
        construct ``IsaacG1Robot`` here (no Isaac Sim), so the builder is run for its kwargs —
        which is the part that can be wrong on a Mac — and the factory is left uncalled.

        Written generically: every ``sim:`` key in the yaml must reach a real parameter of
        ``IsaacG1Robot.__init__`` and appear in the kwargs the builder assembles.
        """
        import argparse
        import importlib.util
        import inspect
        import sys

        from wam.robot.isaac_g1 import IsaacG1Robot

        path = CONFIGS_DIR / "robot/isaac_g1.yaml"
        spec_r = importlib.util.spec_from_file_location(
            "rollout", CONFIGS_DIR.parent / "scripts" / "rollout.py"
        )
        rollout = importlib.util.module_from_spec(spec_r)
        sys.modules["rollout"] = rollout
        spec_r.loader.exec_module(rollout)

        captured: dict[str, object] = {}

        def spy(name: str, **kwargs: object):
            captured.update(kwargs)
            raise RuntimeError("stop before Isaac")

        rollout.get_robot = spy
        args = argparse.Namespace(robot_config=path, image_hw=None)
        canonical, dt_s, limits, factory = rollout._build_isaac_g1(args)
        with pytest.raises(RuntimeError, match="stop before Isaac"):
            factory(0, jitter=False)

        section = load_config(path)["robot"]
        assert dt_s == float(section["control"]["dt_s"])
        accepted = set(inspect.signature(IsaacG1Robot.__init__).parameters)
        sim_keys = dict(section.get("sim", {}))
        assert sim_keys, "the yaml no longer declares a sim: block"
        for key in sim_keys:
            # 'scene' is the repo-wide yaml spelling; IsaacG1Robot takes it as scene_path.
            param = "scene_path" if key == "scene" else key
            assert param in accepted, f"sim.{key} is not a parameter of IsaacG1Robot"
            assert param in captured, f"_build_isaac_g1 drops sim.{key} from the yaml"
        assert captured["config"].control_dt_s == dt_s
        assert len(limits["ddq_max"]) == canonical.num_joints

    def test_view_sim_honours_every_robot_config_field_the_adapter_accepts(self) -> None:
        """``scripts/view_sim.py`` claims to drive "the same chain as scripts/rollout.py", and
        ``docs/sim.md`` repeats it. That is only true if it reads the same fields.

        Found by adversarial review of T-25c: ``rollout.py`` was taught ``q_track_window`` and
        ``view_sim.py`` was not, so the two entry points silently ran DIFFERENT control laws off
        the same file — 0.44 vs 0.96 of a commanded travel, and 3.3x the accel_limit
        interventions the doc quotes. Written generically rather than for that one field, so the
        next field added to the yaml and forgotten here fails too.
        """
        import argparse
        import importlib.util
        import sys

        path = CONFIGS_DIR / "robot/mujoco_g1.yaml"
        spec = importlib.util.spec_from_file_location(
            "view_sim", CONFIGS_DIR.parent / "scripts" / "view_sim.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["view_sim"] = module
        spec.loader.exec_module(module)
        built = module._build_robot(path, realtime=False)[1]

        section = load_config(path)["robot"]
        declared: dict[str, object] = {}
        for block in ("limits", "gains", "control"):
            for key, value in dict(section.get(block, {})).items():
                name = "control_dt_s" if key == "dt_s" else key
                if name in type(built).model_fields:
                    declared[name] = value
        assert "q_track_window" in declared, "the yaml no longer declares the field under test"
        for name, value in declared.items():
            got = getattr(built, name)
            want = tuple(float(x) for x in value) if isinstance(value, list) else float(value)
            assert got == want, f"view_sim.py drops robot config field {name!r}"

        # And the same field really does reach rollout.py's builder for the same file.
        spec_r = importlib.util.spec_from_file_location(
            "rollout", CONFIGS_DIR.parent / "scripts" / "rollout.py"
        )
        rollout = importlib.util.module_from_spec(spec_r)
        sys.modules["rollout"] = rollout
        spec_r.loader.exec_module(rollout)
        args = argparse.Namespace(robot_config=path, image_hw=None)
        assert rollout._build_mujoco_g1(args)[1] == built.control_dt_s

    def test_safety_config_fields(self) -> None:
        cfg = load_config(CONFIGS_DIR / "safety/default.yaml")
        n = len(cfg["q_min"])
        assert n >= 1
        for key in ("q_max", "dq_max", "ddq_max"):
            assert len(cfg[key]) == n
        assert all(lo < hi for lo, hi in zip(cfg["q_min"], cfg["q_max"]))
        assert all(v > 0 for v in cfg["dq_max"] + cfg["ddq_max"])
        assert len(cfg["workspace_min"]) == 3 and len(cfg["workspace_max"]) == 3
        assert all(lo < hi for lo, hi in zip(cfg["workspace_min"], cfg["workspace_max"]))
        assert cfg["gripper_rate_max"] > 0
        assert cfg["chunk_timeout_s"] > 0
        assert cfg["timeout_policy"] in ("hold", "stop")

    def test_safety_default_validates_as_safety_config(self) -> None:
        """Field-name coordination gate with wam.safety.config.SafetyConfig (T-04)."""
        safety_config = pytest.importorskip("wam.safety.config")
        cfg = safety_config.SafetyConfig.from_yaml(CONFIGS_DIR / "safety/default.yaml")
        mock = load_config(CONFIGS_DIR / "robot/mock.yaml")
        assert cfg.num_joints == len(mock["robot"]["canonical_space"]["joint_names"])
        # Safety limits must be at least as tight as the mock robot limits.
        limits = mock["robot"]["limits"]
        assert all(s >= r for s, r in zip(cfg.q_min, limits["q_min"]))
        assert all(s <= r for s, r in zip(cfg.q_max, limits["q_max"]))
        assert all(s <= r for s, r in zip(cfg.dq_max, limits["dq_max"]))

    def test_config_hash_of_shipped_config_is_stable(self) -> None:
        cfg1 = load_config(CONFIGS_DIR / "robot/mock.yaml")
        cfg2 = yaml.safe_load((CONFIGS_DIR / "robot/mock.yaml").read_text())
        assert config_hash(cfg1) == config_hash(cfg2)
