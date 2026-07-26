"""Unit tests for wam.interfaces.protocols (runtime-checkable conformance with tiny stubs)."""

from __future__ import annotations

from typing import Any

import numpy as np

from wam.interfaces.protocols import (
    INTERFACES_VERSION,
    ActionDecoder,
    ActionEncoder,
    BackboneAdapter,
    Observation,
    Policy,
    RobotAdapter,
    SafetyFilter,
    SafetyIntervention,
    StateEncoder,
)
from wam.interfaces.schema import ActionChunk, ActionMode, IMUState, RobotState

N = 3


def make_state() -> RobotState:
    return RobotState(
        timestamp_ns=0,
        q=np.zeros(N, dtype=np.float32),
        dq=np.zeros(N, dtype=np.float32),
        imu=IMUState(
            orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
            angular_velocity=np.zeros(3, dtype=np.float32),
            linear_acceleration=np.zeros(3, dtype=np.float32),
        ),
        gripper_state=np.zeros(1, dtype=np.float32),
    )


def make_chunk(t: int = 8) -> ActionChunk:
    return ActionChunk(
        mode=ActionMode.JOINT_DELTA,
        targets=np.zeros((t, N), dtype=np.float32),
        gripper_target=np.zeros(t, dtype=np.float32),
        dt_s=0.1,
    )


class StubStateEncoder:
    @property
    def embedding_dim(self) -> int:
        return 8

    def encode(self, state: RobotState) -> Any:
        return np.zeros(self.embedding_dim, dtype=np.float32)


class StubActionEncoder:
    @property
    def latent_dim(self) -> int:
        return 4

    def encode(self, chunk: ActionChunk) -> Any:
        return np.zeros((chunk.num_steps, self.latent_dim), dtype=np.float32)


class StubActionDecoder:
    def decode(self, features: Any) -> ActionChunk:
        return make_chunk()


class StubBackbone:
    @property
    def name(self) -> str:
        return "stub-i2v"

    @property
    def feature_dim(self) -> int:
        return 16

    def condition_video(self, video: Any) -> Any:
        return video

    def condition_text(self, text: str) -> Any:
        return text

    def condition_state(self, state_embedding: Any) -> Any:
        return state_embedding

    def features(self, video_ctx: Any, text_ctx: Any, state_ctx: Any) -> Any:
        return np.zeros((1, self.feature_dim), dtype=np.float32)


class StubSafetyFilter:
    def filter(
        self, state: RobotState, chunk: ActionChunk
    ) -> tuple[ActionChunk, list[SafetyIntervention]]:
        return chunk, []


class StubRobotAdapter:
    @property
    def limits(self) -> dict[str, np.ndarray]:
        return {
            "q_min": np.full(N, -1.0, dtype=np.float32),
            "q_max": np.full(N, 1.0, dtype=np.float32),
            "dq_max": np.full(N, 2.0, dtype=np.float32),
        }

    def read_state(self) -> RobotState:
        return make_state()

    def execute(self, chunk: ActionChunk, prefix_steps: int) -> None:
        pass

    def hold(self) -> None:
        pass

    def estop(self) -> None:
        pass


class StubPolicy:
    def predict(self, observation: Observation) -> ActionChunk:
        return make_chunk()


def test_interfaces_version_is_semver_string() -> None:
    assert isinstance(INTERFACES_VERSION, str)
    assert len(INTERFACES_VERSION.split(".")) == 3


def test_stubs_conform_to_protocols() -> None:
    assert isinstance(StubStateEncoder(), StateEncoder)
    assert isinstance(StubActionEncoder(), ActionEncoder)
    assert isinstance(StubActionDecoder(), ActionDecoder)
    assert isinstance(StubBackbone(), BackboneAdapter)
    assert isinstance(StubSafetyFilter(), SafetyFilter)
    assert isinstance(StubRobotAdapter(), RobotAdapter)
    assert isinstance(StubPolicy(), Policy)


def test_incomplete_stub_does_not_conform() -> None:
    class MissingMethods:
        @property
        def feature_dim(self) -> int:
            return 16

    assert not isinstance(MissingMethods(), BackboneAdapter)
    assert not isinstance(object(), Policy)
    assert not isinstance(object(), RobotAdapter)


def test_protocols_are_independent() -> None:
    # A StateEncoder is not an ActionEncoder (different member names).
    assert not isinstance(StubStateEncoder(), ActionEncoder)
    assert not isinstance(StubActionEncoder(), StateEncoder)


def test_safety_filter_contract() -> None:
    chunk = make_chunk()
    out, interventions = StubSafetyFilter().filter(make_state(), chunk)
    assert isinstance(out, ActionChunk)
    assert interventions == []


def test_safety_intervention_fields() -> None:
    iv = SafetyIntervention(kind="joint_limit", detail="q[2] above q_max", timestamp_ns=123)
    assert (iv.kind, iv.detail, iv.timestamp_ns) == ("joint_limit", "q[2] above q_max", 123)


def test_observation_carries_images_state_instruction() -> None:
    obs = Observation(
        images={"front": np.zeros((4, 4, 3), dtype=np.uint8)},
        state=make_state(),
        instruction="pick up the red cup",
    )
    chunk = StubPolicy().predict(obs)
    assert chunk.validate() == []
    assert obs.images["front"].shape == (4, 4, 3)


def test_end_to_end_stub_wiring() -> None:
    # StateEncoder -> Backbone -> ActionDecoder -> SafetyFilter -> RobotAdapter type-flow.
    robot = StubRobotAdapter()
    state = robot.read_state()
    backbone = StubBackbone()
    features = backbone.features(
        backbone.condition_video(None),
        backbone.condition_text("place the cup"),
        backbone.condition_state(StubStateEncoder().encode(state)),
    )
    chunk = StubActionDecoder().decode(features)
    safe_chunk, interventions = StubSafetyFilter().filter(state, chunk)
    robot.execute(safe_chunk, prefix_steps=4)
    assert interventions == []
    assert set(robot.limits) >= {"q_min", "q_max", "dq_max"}
