#!/usr/bin/env python
"""Prove our NEW_EMBODIMENT config still equals the recorded recipe, and still fits the dataset.

    .venv/bin/python configs/groot/verify_new_embodiment_config.py [--dataset <root>]

new_embodiment_config_defaults.py is a hand-written config. Drift in it is silent: swap one
RELATIVE for one ABSOLUTE and nothing raises, the loss curve looks ordinary, and the policy is
trained against the wrong target. Same for a dropped key or a shortened horizon.

WHAT IT IS CHECKED AGAINST, AND WHY NOT UPSTREAM. The obvious reference,
MODALITY_CONFIGS["unitree_g1_full_body_with_waist_height_nav_cmd"], is the wrong one — this file
used to use it and it hid three real defects at once (horizon 50 instead of 16, five missing
effort_* keys, and navigate/base_height transposed). The reference is instead
``reference_applepnp_new_embodiment.json``, extracted verbatim from
``experiment_cfg/conf.yaml`` inside a COMPLETED finetune of this corpus — a run's own record of
what it actually trained under, which upstream's generic G1 entry is not. It carries the source
sha256. NVIDIA's own nvidia/GR00T-N1.7-ApplePnP-V1 ONNX export agrees with it independently:
decode_action emits [1, 16, D] for exactly these twelve keys.

Two independent checks, because they fail differently:

  A. CONFIG vs RECORDED RECIPE — catches our config drifting from what actually trained.
  B. CONFIG vs DATASET         — catches the config naming a key meta/modality.json does not
                                 define, which is what happens when a 28-dim Dex3 config is
                                 pointed at a 43-dim corpus. That one *does* raise eventually,
                                 but at train time.

Exit 0 = both pass. Exit 1 = a real mismatch. Exit 2 = could not run the check at all, which is
NOT a pass — an unrunnable verifier must never read as a green light.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REFERENCE = Path(__file__).resolve().parent / "reference_applepnp_new_embodiment.json"
DEFAULT_DATASET = Path.home() / "wam-t041/raw/GR00T-N1.7-AppleToPlate"

# The four modality groups a config must define. Checked explicitly rather than by iterating
# whichever keys happen to be present: a config that simply omits "action" would otherwise compare
# equal on everything it does define.
GROUPS = ("video", "state", "action", "language")


def _fail(msg: str) -> None:
    print(f"  MISMATCH: {msg}")


def _action_config_fields(ac) -> tuple:
    """(rep, type, format) as plain UPPERCASE strings.

    Normalises three shapes onto one: our side hands over ``ActionConfig`` enums, the reference
    JSON hands over the strings the trainer serialised (``"RELATIVE"``, ``"NON_EEF"``, …), and an
    enum's ``.value`` may be either case. Comparing the objects directly would make the check
    depend on ``__eq__`` across two representations, which is how a verifier ends up reporting a
    difference that is only a difference in spelling.
    """
    if isinstance(ac, dict):
        raw = [ac.get(f) for f in ("rep", "type", "format")]
    else:
        raw = [getattr(getattr(ac, f, None), "value", getattr(ac, f, None)) for f in ("rep", "type", "format")]
    return tuple(None if v is None else str(v).upper() for v in raw)


def compare_group(name: str, ours, reference: dict) -> list[str]:
    """``ours`` is a live ``ModalityConfig``; ``reference`` is the recorded recipe as plain JSON."""
    problems: list[str] = []

    ref_keys = list(reference.get("modality_keys") or [])
    ref_deltas = list(reference.get("delta_indices") or [])

    if list(ours.modality_keys) != ref_keys:
        problems.append(
            f"{name}.modality_keys: ours={list(ours.modality_keys)} recorded={ref_keys}"
        )
    if list(ours.delta_indices) != ref_deltas:
        o = list(ours.delta_indices)
        problems.append(
            f"{name}.delta_indices: ours len={len(o)} {o[:4]}… recorded len={len(ref_deltas)} {ref_deltas[:4]}…"
        )

    ours_ac = getattr(ours, "action_configs", None)
    ref_ac = reference.get("action_configs")
    if (ours_ac is None) != (ref_ac is None):
        problems.append(f"{name}.action_configs: ours={'None' if ours_ac is None else 'set'} "
                        f"recorded={'None' if ref_ac is None else 'set'}")
    elif ours_ac is not None:
        if len(ours_ac) != len(ref_ac):
            problems.append(f"{name}.action_configs: {len(ours_ac)} entries vs recorded {len(ref_ac)}")
        else:
            # Zip against modality_keys so a mismatch names the channel, not an index.
            for key, a, b in zip(ours.modality_keys, ours_ac, ref_ac):
                fa, fb = _action_config_fields(a), _action_config_fields(b)
                if fa != fb:
                    problems.append(f"{name}.action_configs[{key}]: ours={fa} recorded={fb}")
    return problems


def check_against_dataset(ours: dict, root: Path) -> list[str]:
    problems: list[str] = []
    modality_path = root / "meta/modality.json"
    if not modality_path.exists():
        return [f"dataset: {modality_path} does not exist"]

    modality = json.loads(modality_path.read_text())

    # state/action keys must be defined by the dataset. video keys likewise, and the language key
    # is checked against the annotation block rather than assumed.
    for group, block in (("state", "state"), ("action", "action"), ("video", "video")):
        defined = set(modality.get(block, {}))
        for key in ours[group].modality_keys:
            if key not in defined:
                problems.append(
                    f"{group}: config names {key!r}, absent from meta/modality.json[{block!r}] "
                    f"(defined: {sorted(defined)})"
                )

    for key in ours["language"].modality_keys:
        # modality.json spells the annotation without the leading "annotation." namespace.
        bare = key.split("annotation.", 1)[-1]
        if bare not in modality.get("annotation", {}):
            problems.append(
                f"language: config names {key!r} -> {bare!r}, absent from "
                f"meta/modality.json['annotation'] (defined: {sorted(modality.get('annotation', {}))})"
            )

    # The state slices must tile 0..N with no gap and no overlap. A gap here means the config
    # trains on a vector that is not the one the parquet stores.
    spans = [(modality["state"][k]["start"], modality["state"][k]["end"], k)
             for k in ours["state"].modality_keys]
    spans.sort()
    cursor = 0
    for start, end, key in spans:
        if start != cursor:
            problems.append(f"state: {key} starts at {start}, expected {cursor} (gap or overlap)")
        cursor = end

    info_path = root / "meta/info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        declared = info.get("features", {}).get("observation.state", {}).get("shape", [None])[0]
        if declared is not None and cursor != declared:
            problems.append(f"state: slices cover {cursor} dims, info.json declares {declared}")
        else:
            print(f"  state slices tile 0..{cursor}, matching info.json observation.state")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help="LeRobot dataset root holding meta/modality.json")
    args = ap.parse_args()

    try:
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        from gr00t.data.embodiment_tags import EmbodimentTag
    except ImportError as exc:  # noqa: BLE001 - the message matters more than the type
        print(f"CANNOT VERIFY: Isaac-GR00T is not importable ({exc}).")
        print("This is exit 2, not a pass. Run inside the env that has gr00t on its path.")
        return 2

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import new_embodiment_config_defaults  # noqa: F401  (import registers the config)
    except Exception as exc:  # noqa: BLE001
        print(f"CANNOT VERIFY: importing new_embodiment_config_defaults failed ({exc}).")
        return 2

    if not REFERENCE.is_file():
        print(f"CANNOT VERIFY: {REFERENCE} is missing.")
        print("It is the recorded recipe this config is checked against; without it there is")
        print("nothing to compare to, and 'nothing to compare to' is not a pass.")
        return 2
    reference_doc = json.loads(REFERENCE.read_text())
    reference = reference_doc["modality_configs"]
    provenance = reference_doc.get("_provenance", {})

    ours = MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]

    problems: list[str] = []

    print(f"A. NEW_EMBODIMENT vs {REFERENCE.name}")
    print(f"   recorded from {provenance.get('source', '?')}")
    print(f"   sha256 {str(provenance.get('source_sha256', '?'))[:16]}…"
          f"  horizon {provenance.get('action_horizon', '?')}"
          f"  tag {provenance.get('embodiment_tag', '?')}")
    missing = [g for g in GROUPS if g not in ours or g not in reference]
    if missing:
        problems.append(f"missing modality groups: {missing}")
    for group in GROUPS:
        if group in ours and group in reference:
            problems.extend(compare_group(group, ours[group], reference[group]))
    extra = set(ours) - set(reference)
    if extra:
        problems.append(f"config defines groups the recorded recipe does not: {sorted(extra)}")
    for p in problems:
        _fail(p)
    if not problems:
        print("  identical on keys, delta_indices and action_configs")

    print(f"B. NEW_EMBODIMENT vs dataset at {args.dataset}")
    dataset_problems = check_against_dataset(ours, args.dataset)
    for p in dataset_problems:
        _fail(p)
    if not dataset_problems:
        print("  every config key is defined by meta/modality.json")
    problems.extend(dataset_problems)

    print()
    if problems:
        print(f"FAIL — {len(problems)} mismatch(es). Do not train on this config.")
        return 1
    print("PASS — config matches the recorded recipe and fits the dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
