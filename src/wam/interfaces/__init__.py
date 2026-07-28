"""Versioned core contracts: canonical state/action schema and module protocols.

Change with care — everything in WAM depends on these (T-01/T-02, FR-06/FR-09).
"""

from wam.interfaces.protocols import (
    INTERFACES_VERSION,
    ActionDecoder,
    ActionEncoder,
    BackboneAdapter,
    FlowBackbone,
    Observation,
    Policy,
    RobotAdapter,
    SafetyFilter,
    SafetyIntervention,
    StateEncoder,
)
from wam.interfaces.schema import (
    SCHEMA_VERSION,
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    IMUState,
    NormalizationSpec,
    RobotState,
    ValidityMask,
)
from wam.interfaces.versioning import (
    WAM_CONFIG_VERSION,
    JsonlRunLogger,
    RunMetadata,
    config_hash,
    load_config,
    read_git_commit,
)

__all__ = [
    "INTERFACES_VERSION",
    "SCHEMA_VERSION",
    "WAM_CONFIG_VERSION",
    "ActionChunk",
    "ActionDecoder",
    "ActionEncoder",
    "ActionMode",
    "BackboneAdapter",
    "CanonicalSpaceSpec",
    "FlowBackbone",
    "IMUState",
    "JsonlRunLogger",
    "NormalizationSpec",
    "Observation",
    "Policy",
    "RobotAdapter",
    "RobotState",
    "RunMetadata",
    "SafetyFilter",
    "SafetyIntervention",
    "StateEncoder",
    "ValidityMask",
    "config_hash",
    "load_config",
    "read_git_commit",
]
