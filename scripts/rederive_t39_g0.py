#!/usr/bin/env python3
"""T-47 / PR-13 — does T-39's `VOID` survive its instrument being repaired?

    scripts/rederive_t39_g0.py \
        --dataset datasets/gr00t-apple-full \
        --raw-dataset ~/wam-t041/raw/GR00T-N1.7-AppleToPlate \
        --holdout configs/splits/t18_holdout_episodes.txt \
        --out runs/t47-g0-rederivation

The rule is ``T47_RULE_V1`` in ``docs/preregistration/PR-13-t39-g0-rederivation.md``, committed in
``7d93ab6`` before this file existed.

T-39's ``VOID`` was decided by ONE CLAUSE, not by the policy arm — that never ran.
``PR-07-positive-control.md:137-141``: *"`oracle_action` must reach L1. Below that […] T-39 is VOID
and the finding is recorded against the label pipeline."* It came in at −359.41. PR-12 showed that
number was produced by an adapter whose step 0 carried ~90 % of the error. This asks only whether
the clause still fires once that is repaired.

NOTHING IS RE-IMPLEMENTED, AND HERE THAT IS STRICTER THAN USUAL. PR-13 §8 requires the repaired
anchoring to be the SAME FUNCTION OBJECT PR-12 scored, so ``chained_oracle_action_chunks`` is
imported from ``probe_step_zero_anchor`` rather than re-derived. A re-derivation through a
lookalike would replicate the copy, which is the failure this whole line of work exists to name.

THE CHUNK SET IS THE POINT OF THIS DRIVER AND IS FIXED IN THE PRE-REGISTRATION, NOT HERE. Three
cells at ``d = 0``:

    bridge     unmodified anchoring, the FULL 1 040 chunks — T-39's own set, so exactly one number
               in the artifact is directly comparable to PR-07-RESULT's table.
    control    unmodified anchoring, the ANCHORABLE set.
    repaired   V-chain, the ANCHORABLE set.

V-chain needs ``start >= 1``, so it cannot score each episode's first chunk. The anchorable set is
the full holdout minus those, and BOTH compared cells are scored on exactly it — a difference of
chunk set between the two cells being compared would make them not a comparison, and quoting the
repaired cell against the 1 040-chunk bridge would compare two sets. G0.3 checks both at runtime.

``oracle_state`` never calls ``commanded_to_chunk`` and is therefore untouched by V-chain. It is
scored unmodified, as G0's other clause.

NO WITNESS IS LOADED, for the reason the earlier drivers give: every arm is ground truth, no model
is trained or consulted, so there is nothing a train/holdout leak could contaminate. The holdout
file is the episode list.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from wam.evaluation import (  # noqa: E402
    bench_metrics,
    build_eval_pairs,
    evaluate_policy,
)

MATERIAL_FLOOR_PP = 10.0
"""Borrowed from ``I8_RULE_V3`` via PR-07 §5 for the fifth time, so the floor cannot be the finding."""

PR07_ORACLE_ACTION_L1 = -359.41
"""PR-07-RESULT's headline on the full 1 040 chunks. G0.2 must reproduce it."""

BRIDGE_TOLERANCE_PP = 0.5
"""PR-13 §5 G0.2. Wider than that and this is not comparable to the archive."""

ORACLE_STATE_FLOOR_PCT = 90.0
"""T-39's own G0 first clause, unchanged."""

BENCH_SPECS = ("0.1.0", "0.2.0")
"""0.1.0 is what the rule reads — it is what T39_RULE_V1 read and what the archive was scored under."""


def _load_script(name: str) -> Any:
    import importlib.util

    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def per_step_profile(predictions: list[Any]) -> list[float]:
    """``benchmark.py:525,563``'s ``per_step_mse``, recomputed because only its ratio is exposed."""
    steps = max(p.target.targets.shape[0] for p in predictions)
    total = np.zeros(steps, dtype=np.float64)
    count = np.zeros(steps, dtype=np.float64)
    for pred in predictions:
        p = pred.predicted.targets.astype(np.float64)
        t = pred.target.targets.astype(np.float64)
        n = t.shape[0]
        total[:n] += ((p - t) ** 2).mean(axis=1)
        count[:n] += 1.0
    return [float(v) for v in total / np.maximum(count, 1.0)]


def _score(predictions: list[Any], run_name: str) -> dict[str, Any]:
    bench = bench_metrics(predictions, run_name=run_name, spec_version=BENCH_SPECS[0])
    alt = bench_metrics(predictions, run_name=run_name, spec_version=BENCH_SPECS[1])
    profile = per_step_profile(predictions)
    total = float(sum(profile))
    return {
        "spec_0_2_0": {
            "skill_vs_repeat_pct": float(alt.skill_vs_repeat_pct),
            "smoothness_ratio": float(alt.smoothness_ratio),
            "level_name": str(alt.level_name),
            "score": float(alt.score),
        },
        "num_chunks": int(bench.num_predictions),
        "num_episodes": int(bench.num_episodes),
        "skill_vs_repeat_pct": float(bench.skill_vs_repeat_pct),
        "ci_skill_vs_repeat_pct": float(bench.ci_skill_vs_repeat_pct),
        "skill_vs_zero_pct": float(bench.skill_vs_zero_pct),
        "horizon_ratio": float(bench.horizon_ratio),
        "smoothness_ratio": float(bench.smoothness_ratio),
        "level_name": str(bench.level_name),
        "score": float(bench.score),
        "per_step_mse": profile,
        "step_zero_share_pct": 100.0 * profile[0] / total if total > 0.0 else 0.0,
    }


def _verdict(repaired: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    """``T47_RULE_V1``, PR-13 §5. Precedence fixed in the pre-registration: S, then W, then I."""
    l1 = repaired["skill_vs_repeat_pct"]
    l2 = repaired["ci_skill_vs_repeat_pct"]
    if l1 <= 0.0:
        return {
            "verdict": "S",
            "reading": (
                f"The VOID SURVIVES its instrument being repaired. The repaired cell scores "
                f"{l1:+.2f} % on the anchorable set, still below L1, against the unmodified "
                f"{control['skill_vs_repeat_pct']:+.2f} %. PR-12's gain was a property of the "
                "trimmed chunk set and does not generalise to the set T39_RULE_V1's G0 actually "
                "read. The label space really is the problem, and this is a far stronger negative "
                "than the project currently holds."
            ),
        }
    if l1 >= MATERIAL_FLOOR_PP and l2 > 0.0:
        return {
            "verdict": "W",
            "reading": (
                f"T39_RULE_V1's G0 blocking clause DOES NOT HOLD under a corrected instrument: the "
                f"repaired cell reaches L1 at {l1:+.2f} % and L2 at {l2:+.2f} % on T-39's own "
                f"holdout, against the unmodified {control['skill_vs_repeat_pct']:+.2f} %. The "
                "premise of VOID (labels) is withdrawn by measurement. THIS IS NOT A VERDICT ON "
                "T-39 and cannot be — P/N/M/I all require the policy arm, which never ran. It "
                "licenses CORRECTING every document asserting that no policy trained on this "
                "corpus's action column can clear our bar. Correcting a claim is not lifting a "
                "gate: the training gate is untouched and remains the project owner's call, no "
                "statement about GR00T or any policy is licensed, and the fourteen negatives are "
                "not retro-validated."
            ),
        }
    return {
        "verdict": "I",
        "reading": (
            f"Indeterminate. The repaired cell reaches L1 at {l1:+.2f} % with L2 {l2:+.2f} %, but W "
            f"needs L1 >= {MATERIAL_FLOOR_PP} AND L2 > 0. Nothing is licensed, and PR-13 §4's "
            "set-representativeness question becomes the live one."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--camera", default="ego")
    parser.add_argument("--chunk-steps", type=int)
    args = parser.parse_args(argv)

    eval_t39 = _load_script("eval_t39_baseline")
    probe = _load_script("probe_step_zero_anchor")
    convert = eval_t39._load_script("convert_lerobot_g1")

    from wam.data.episode import EpisodeReader, list_episodes

    holdout_order = [
        line.strip()
        for line in args.holdout.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    by_name = {p.name: p for p in list_episodes(args.dataset)}
    missing = [e for e in holdout_order if e not in by_name]
    if missing:
        raise SystemExit(f"{len(missing)} holdout episode(s) absent from {args.dataset}: {missing[:5]}")
    episode_dirs = [by_name[e] for e in holdout_order]

    first = EpisodeReader(episode_dirs[0])
    chunk_steps = args.chunk_steps or eval_t39.episode_chunk_steps(first)
    mapping = eval_t39.gripper_mapping_from_manifest(first.manifest, convert)

    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"T-47 G0 re-derivation | {len(episode_dirs)} holdout episodes | chunk_steps {chunk_steps} "
        f"| gripper {mapping.kind}"
    )

    started = time.perf_counter()
    cache: dict[str, dict[str, Any]] = {}
    for episode_dir in episode_dirs:
        reader = EpisodeReader(episode_dir)
        episode_id = reader.manifest.episode_id
        cache[episode_id] = {
            "reader": reader,
            "pairs": build_eval_pairs(episode_dir, args.camera, chunk_steps, num_frames=None),
            "raw": eval_t39.read_raw_episode(args.raw_dataset, episode_id),
        }
    print(f"  loaded {len(cache)} episodes in {time.perf_counter() - started:.1f}s")

    def chunks_for(entry: dict, cell: str) -> dict[int, Any]:
        reader = entry["reader"]
        if cell == "oracle_state":
            return eval_t39.oracle_state_chunks(reader, chunk_steps, mapping)
        if cell == "repaired":
            # PR-13 §8: the SAME function object PR-12 scored, not a re-derivation of it.
            return probe.chained_oracle_action_chunks(
                eval_t39, reader, entry["raw"], chunk_steps, mapping, convert, delay=0
            )
        return eval_t39.oracle_action_chunks(reader, entry["raw"], chunk_steps, mapping, convert)

    # The anchorable set: every scored timestamp the REPAIRED cell can reach, per episode. Computed
    # once and applied to the control too, so the two compared cells are scored over one set.
    anchorable: dict[str, set[int]] = {}
    for episode_id, entry in cache.items():
        keys = set(chunks_for(entry, "repaired"))
        anchorable[episode_id] = {
            int(obs.state.timestamp_ns)
            for obs, _t, _e in entry["pairs"]
            if int(obs.state.timestamp_ns) in keys
        }

    def run(cell: str, *, restrict: bool) -> dict[str, Any]:
        predictions: list[Any] = []
        for episode_id, entry in cache.items():
            pairs = entry["pairs"]
            if restrict:
                pairs = [p for p in pairs if int(p[0].state.timestamp_ns) in anchorable[episode_id]]
            if not pairs:
                continue
            policy = eval_t39.ChunkLookupPolicy(chunks_for(entry, cell), episode_id=episode_id)
            predictions.extend(evaluate_policy(policy, pairs))
        if not predictions:
            raise SystemExit(f"no chunks scored for cell={cell}")
        return _score(predictions, run_name=f"t47-{cell}{'-anchorable' if restrict else '-full'}")

    results: dict[str, Any] = {
        "rule": "T47_RULE_V1",
        "preregistration": "docs/preregistration/PR-13-t39-g0-rederivation.md",
        "dataset": str(args.dataset),
        "raw_dataset": str(args.raw_dataset),
        "holdout": str(args.holdout),
        "chunk_steps": chunk_steps,
        "gripper_mapping": mapping.kind,
        "num_episodes": len(episode_dirs),
    }
    gates: list[str] = []

    print("\n=== G0.1  oracle_state, full set (floor 90 %)")
    g0_state = run("oracle_state", restrict=False)
    results["g0_oracle_state"] = g0_state
    print(f"    {g0_state['skill_vs_repeat_pct']:+.2f} %   ({g0_state['num_chunks']} chunks)")
    if g0_state["skill_vs_repeat_pct"] < ORACLE_STATE_FLOOR_PCT:
        gates.append(
            f"G0.1 FAILED: oracle_state {g0_state['skill_vs_repeat_pct']:+.2f} % < "
            f"{ORACLE_STATE_FLOOR_PCT} %. The adapter is broken; this is a code fix."
        )

    print(f"\n=== G0.2  bridge: unmodified oracle_action, FULL set (PR-07 said "
          f"{PR07_ORACLE_ACTION_L1:+.2f})")
    bridge = run("unmodified", restrict=False)
    results["bridge_full"] = bridge
    drift = bridge["skill_vs_repeat_pct"] - PR07_ORACLE_ACTION_L1
    results["bridge_drift_pp"] = drift
    print(
        f"    {bridge['skill_vs_repeat_pct']:+.2f} %   ({bridge['num_chunks']} chunks)   "
        f"drift {drift:+.3f} pp"
    )
    if abs(drift) > BRIDGE_TOLERANCE_PP:
        gates.append(
            f"G0.2 FAILED: bridge drifted {drift:+.3f} pp from PR-07's "
            f"{PR07_ORACLE_ACTION_L1:+.2f} (tolerance ±{BRIDGE_TOLERANCE_PP}). Not comparable to "
            "the archive."
        )

    print("\n=== the anchorable set")
    control = run("unmodified", restrict=True)
    repaired = run("repaired", restrict=True)
    results["control_anchorable"] = control
    results["repaired_anchorable"] = repaired
    expected = bridge["num_chunks"] - len(episode_dirs)
    results["anchorable_expected"] = expected
    print(
        f"    expected {expected} = {bridge['num_chunks']} full − {len(episode_dirs)} episodes | "
        f"control {control['num_chunks']}   repaired {repaired['num_chunks']}"
    )
    if control["num_chunks"] != repaired["num_chunks"]:
        gates.append(
            f"G0.3 FAILED: control scored {control['num_chunks']} chunks and repaired "
            f"{repaired['num_chunks']}. They are not a comparison."
        )
    if repaired["num_chunks"] != expected:
        gates.append(
            f"G0.3 FAILED: anchorable set is {repaired['num_chunks']}, expected {expected} "
            f"(full − one per episode). The drop rule is not what PR-13 §3 registered."
        )

    # G0.4 — V-chain reached row 0 and only row 0. Inherited unchanged from PR-12 §6 G0.3.
    row0: list[float] = []
    rest_max = 0.0
    for episode_id, entry in cache.items():
        unmod = chunks_for(entry, "unmodified")
        chain = chunks_for(entry, "repaired")
        for t_ns in sorted(set(unmod) & set(chain) & anchorable[episode_id]):
            a = np.asarray(unmod[t_ns].targets, dtype=np.float64)
            b = np.asarray(chain[t_ns].targets, dtype=np.float64)
            row0.append(float(np.sqrt(((a[0] - b[0]) ** 2).mean())))
            rest_max = max(rest_max, float(np.abs(a[1:] - b[1:]).max()))
    mean_row0 = float(np.mean(row0)) if row0 else 0.0
    results["g0_vchain_rows"] = {"row0_rms_mean": mean_row0, "rows_1_plus_max_abs_diff": rest_max}
    print(f"    V-chain row0 RMS {mean_row0:.4e}   rows 1.. max|diff| {rest_max:.3e}")
    if mean_row0 <= 0.0:
        gates.append("G0.4 FAILED: V-chain is a no-op on row 0; a flat result would read as S.")
    if rest_max != 0.0:
        gates.append(
            f"G0.4 FAILED: V-chain changed rows 1.. by up to {rest_max:.3e}; it must touch row 0 "
            "and nothing else."
        )

    print(
        f"\n{'cell':>22}  {'chunks':>7}  {'L1':>10}  {'L2':>10}  {'vs0':>9}  {'horiz':>8}  "
        f"{'smooth':>7}  {'step0%':>7}  level"
    )
    for name, cell in (
        ("bridge (full)", bridge),
        ("control (anchorable)", control),
        ("repaired (anchorable)", repaired),
    ):
        print(
            f"{name:>22}  {cell['num_chunks']:>7}  {cell['skill_vs_repeat_pct']:>+10.2f}  "
            f"{cell['ci_skill_vs_repeat_pct']:>+10.2f}  {cell['skill_vs_zero_pct']:>+9.2f}  "
            f"{cell['horizon_ratio']:>8.4f}  {cell['smoothness_ratio']:>7.3f}  "
            f"{cell['step_zero_share_pct']:>7.2f}  {cell['level_name']}"
        )

    results["verdict"] = (
        {"verdict": "INVALID", "reading": " ".join(gates)} if gates else _verdict(repaired, control)
    )
    (args.out / "rederivation.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    verdict = results["verdict"]
    print(f"\n=== T47_RULE_V1 -> {verdict['verdict']}")
    print(f"    {verdict['reading']}")
    print(f"\nwrote {args.out / 'rederivation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
