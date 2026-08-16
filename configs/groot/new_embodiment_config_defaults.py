# The NEW_EMBODIMENT modality config for nvidia/GR00T-N1.7-AppleToPlate.
#
#   python gr00t/experiment/launch_finetune.py --modality-config-path <this file> ...
#   python gr00t/data/stats.py                 --modality-config-path <this file> ...
#
# WHY THIS FILE EXISTS AT ALL, and why it is NOT a copy of upstream's G1 entry. Isaac-GR00T ships
# MODALITY_CONFIGS["unitree_g1_full_body_with_waist_height_nav_cmd"] in
# gr00t/configs/data/embodiment_configs.py, which is the obvious candidate and is the wrong one for
# two independent reasons.
#
# First, the tag. That entry is keyed to EmbodimentTag.UNITREE_G1, which upstream's own enum files
# under "pre-registered POSTTRAIN tags (require finetuned checkpoint)" — nvidia/GR00T-N1.7-3B is the
# BASE checkpoint and carries no trained weights in that slot. NEW_EMBODIMENT is the tag upstream's
# finetuning path is written around (getting_started/finetune_new_embodiment.md), and the registry
# is a plain dict keyed by tag value, so the config must be declared under the tag we actually pass.
#
# Second, the contents. The action block below is NOT upstream's G1 block — see the comment on it.
# NVIDIA's published finetune of this very corpus used a different horizon and five more keys.
#
# IT IS A COPY, AND THE COPY IS THE HAZARD. Divergence from upstream is silent: a wrong `rep` does
# not raise, it trains a policy against the wrong target and the loss curve looks normal. Verify
# before trusting, do not eyeball:
#
#   .venv/bin/python configs/groot/verify_new_embodiment_config.py
#
# THE KEY ORDER IS NOT ALPHABETICAL AND NOT THE STATE ORDER. State is laid out in the dataset's
# own meta/modality.json order (legs, waist, arms, hands = 43 dims). Action is a DIFFERENT order and
# a DIFFERENT set: arms, hands, waist, then the two command channels. Reordering either one silently
# mis-slices the vector — this is exactly how vla-training/groot/*.py (28-dim Dex3, arms-first)
# would corrupt a 43-dim source, which is why none of those six files may be used here.
#
# ARMS ARE RELATIVE, EVERYTHING ELSE IS ABSOLUTE. This drives use_relative_action in the trainer and
# is what makes meta/relative_stats.json necessary. `action_configs` must therefore stay populated:
# gr00t.data.stats.generate_rel_stats returns EARLY and writes nothing when action_configs is None,
# and the missing file is then only discovered at train time.

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


def _non_eef(rep: ActionRepresentation) -> ActionConfig:
    """Every channel on this embodiment is joint-space, so only `rep` ever varies."""
    return ActionConfig(rep=rep, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)


_RELATIVE = ActionRepresentation.RELATIVE
_ABSOLUTE = ActionRepresentation.ABSOLUTE

APPLE_TO_PLATE_CONFIG = {
    # One camera. The dataset has exactly one video feature — observation.images.ego_view, 402
    # videos for 402 episodes — and meta/modality.json maps the short key `ego_view` onto it. Any
    # two-camera config from the Dex3 line is inapplicable, not merely suboptimal.
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    # Full 43-dim state, in the dataset's own order: legs 0:12, waist 12:15, arms 15:29,
    # hands 29:43. The legs are observed but never commanded, which is why they appear here and
    # not below.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_leg",
            "right_leg",
            "waist",
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
    ),
    # HORIZON 16 AND TWELVE KEYS — NVIDIA'S VALUES FOR THIS CORPUS, NOT UPSTREAM'S G1 DEFAULT.
    # This block was first written as a copy of MODALITY_CONFIGS["unitree_g1_full_body_with_waist_
    # height_nav_cmd"]: delta_indices=range(50), seven keys, base_height before navigate. All three
    # are wrong here, and none of them would have raised.
    #
    # The correction comes from two primary sources that agree:
    #
    #   1. experiment_cfg/conf.yaml inside a COMPLETED finetune of this corpus
    #      (models/finetunes/groot_recipeA_ckpt5000) — a run's own record of the config it trained
    #      under, on dataset apple_pnp_h200/dataset, embodiment_tag new_embodiment.
    #   2. nvidia/GR00T-N1.7-ApplePnP-V1's ONNX export, NVIDIA's own published finetune of
    #      GR00T-N1.7-3B on GR00T-N1.7-AppleToPlate. decode_action emits [1, 16, D] for each of
    #      these twelve keys, and preprocess_video bakes embodiment_id = 10 = new_embodiment.
    #
    # Sixteen is also exactly the benchmark's chunk length (PR-07 §4: "our chunks are 16 steps at
    # 30 Hz"), so the comparison is clean — neither truncated nor padded, and the artifact does not
    # have to report it as either.
    #
    # THE FIVE effort_* KEYS ARE NOT OPTIONAL DECORATION. They are real columns in the parquet
    # (action.effort_left_arm and friends, verified in the schema) and NVIDIA trained against them.
    # Dropping them would train a 66-dim head as a 28-dim one and still converge, which is the
    # failure mode this whole comment exists to prevent. We do not score them; that is the
    # evaluator's business, not the recipe's.
    #
    # ORDER IS LOAD-BEARING TWICE OVER: navigate_command precedes base_height_command (the reverse
    # of the first draft), and action_configs is positional — entry i describes modality_keys[i].
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            "waist",
            "navigate_command",
            "base_height_command",
            "effort_left_arm",
            "effort_right_arm",
            "effort_left_hand",
            "effort_right_hand",
            "effort_waist",
        ],
        action_configs=[
            _non_eef(_RELATIVE),  # left_arm
            _non_eef(_RELATIVE),  # right_arm
            # The G1 hand is driven by near-binary open/close codes rather than a continuous
            # trajectory, so a delta against the previous frame carries no usable signal.
            _non_eef(_ABSOLUTE),  # left_hand
            _non_eef(_ABSOLUTE),  # right_hand
            _non_eef(_ABSOLUTE),  # waist
            _non_eef(_ABSOLUTE),  # navigate_command
            _non_eef(_ABSOLUTE),  # base_height_command
            # Efforts are torques, not positions: a relative torque is not a meaningful quantity.
            _non_eef(_ABSOLUTE),  # effort_left_arm
            _non_eef(_ABSOLUTE),  # effort_right_arm
            _non_eef(_ABSOLUTE),  # effort_left_hand
            _non_eef(_ABSOLUTE),  # effort_right_hand
            _non_eef(_ABSOLUTE),  # effort_waist
        ],
    ),
    # Not a literal task string. meta/modality.json redirects this key to the `task_index` column,
    # which meta/tasks.jsonl resolves to "move the apple to the plate" — no trailing period, which
    # docs/vla-benchmark.md §2 point 4 makes a protocol invariant.
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# Import-time side effect, on purpose: launch_finetune.load_modality_config() does nothing but
# importlib.import_module(path.stem), so registration has to happen here or not at all.
register_modality_config(APPLE_TO_PLATE_CONFIG, EmbodimentTag.NEW_EMBODIMENT)
