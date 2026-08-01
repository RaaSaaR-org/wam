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


class TestShippedConfigs:
    @pytest.mark.parametrize(
        "rel_path",
        [
            "robot/mock.yaml",
            "robot/g1.yaml",
            "robot/mujoco_g1.yaml",
            "safety/default.yaml",
        ],
    )
    def test_loads_with_valid_version(self, rel_path: str) -> None:
        cfg = load_config(CONFIGS_DIR / rel_path)
        assert cfg["wam_config_version"] == WAM_CONFIG_VERSION

    @pytest.mark.parametrize(
        "rel_path", ["robot/mock.yaml", "robot/g1.yaml", "robot/mujoco_g1.yaml"]
    )
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

    @pytest.mark.parametrize("rel_path", ["robot/g1.yaml", "robot/mujoco_g1.yaml"])
    def test_g1_yaml_canonical_space_matches_adapter_spec(self, rel_path: str) -> None:
        """Coordination gate: every versioned G1 yaml and the hard-wired G1_SPEC/G1Config
        must describe the SAME canonical space — episodes, policies and calibration files
        built from the yaml would otherwise be unusable on the adapter. Both G1 configs are
        covered: the hardware one and the MuJoCo (E2) one, which drives the same adapter."""
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
        """The MuJoCo config is the only shipped one carrying a `gains:` block. It must not
        leak into the hardware placeholders (OD-08), and kd must be the per-joint critical
        damping shape, never a flat number."""
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
