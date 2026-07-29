"""Tests for scripts/hf_job_wan_smoke.py — the parts that run without the 5B weights.

The GPU checks in this script need real Wan weights and a cluster. Its *inputs* do not, and that
is where it actually broke: on Discoverer+ job 183565 the adapter passed six checks against the
real 5B model and then died building the state embedding, because ``StateMLPConfig`` was left at
``gripper_dims=1`` while a real G1 episode carries one gripper value per hand. The 4 GPU-hours were
spent proving the expensive path works and the cheap path does not. So the cheap path gets tests.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from wam.data.episode import EpisodeWriter
from wam.robot.mock import MockRobot

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load("hf_job_wan_smoke")


def _write_episode(root: Path, *, num_joints: int, gripper_dims: int) -> Path:
    """A one-state WAM episode whose canonical space matches a given robot geometry."""
    robot = MockRobot(num_joints=num_joints, gripper_dims=gripper_dims)
    path = root / f"ep-{num_joints}j{gripper_dims}g"
    with EpisodeWriter(path, path.name, robot.spec, 30.0, "pick the apple") as writer:
        writer.add_state(robot.read_state())
    return path


def _args(episode: Path | None) -> argparse.Namespace:
    return argparse.Namespace(episode=str(episode) if episode else None)


def test_state_embedding_accepts_a_two_dim_gripper(tmp_path: Path) -> None:
    """The 183565 regression: a real G1 episode has gripper_dims=2 and must not be rejected."""
    episode = _write_episode(tmp_path, num_joints=15, gripper_dims=2)
    report = smoke.Report()

    embedding = smoke.state_embedding(_args(episode), report)

    assert tuple(embedding.shape) == (32,)
    assert np.isfinite(embedding.detach().numpy()).all()
    assert report.info["state_dim"] == 15
    assert report.info["gripper_dims"] == 2


@pytest.mark.parametrize(("num_joints", "gripper_dims"), [(6, 1), (15, 2), (7, 3)])
def test_state_embedding_follows_the_episode_geometry(
    tmp_path: Path, num_joints: int, gripper_dims: int
) -> None:
    """Both dims come from the episode, so no default can silently disagree with the data."""
    episode = _write_episode(tmp_path, num_joints=num_joints, gripper_dims=gripper_dims)
    report = smoke.Report()

    smoke.state_embedding(_args(episode), report)

    assert (report.info["state_dim"], report.info["gripper_dims"]) == (num_joints, gripper_dims)


def test_state_embedding_synthetic_path_is_self_consistent() -> None:
    """Without --episode the encoder is built from synthetic_state(), not from hardcoded dims."""
    state = smoke.synthetic_state()
    report = smoke.Report()

    embedding = smoke.state_embedding(_args(None), report)

    assert tuple(embedding.shape) == (32,)
    assert report.info["state_dim"] == len(state.q)
    assert report.info["gripper_dims"] == len(state.gripper_state)


def test_state_embedding_is_deterministic(tmp_path: Path) -> None:
    """Seeded init: two calls on the same episode agree, so probe deltas mean something."""
    episode = _write_episode(tmp_path, num_joints=15, gripper_dims=2)

    first = smoke.state_embedding(_args(episode), smoke.Report())
    second = smoke.state_embedding(_args(episode), smoke.Report())

    np.testing.assert_array_equal(first.detach().numpy(), second.detach().numpy())


def test_synthetic_state_responds_to_fill_values() -> None:
    """The ablation path perturbs q/dq to move the embedding; it must actually change the state."""
    base = smoke.synthetic_state()
    perturbed = smoke.synthetic_state(fill_q=0.5, fill_dq=0.3)

    assert np.allclose(base.q, 0.0)
    assert np.allclose(perturbed.q, 0.5)
    assert np.allclose(perturbed.dq, 0.3)
    assert base.q.shape == perturbed.q.shape
