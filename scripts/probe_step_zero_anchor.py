#!/usr/bin/env python3
"""T-46 / PR-12 — step 0 is the only heterogeneous element of the chunk. Does homogenising it help?

    scripts/probe_step_zero_anchor.py \
        --dataset datasets/gr00t-apple-full \
        --raw-dataset ~/wam-t041/raw/GR00T-N1.7-AppleToPlate \
        --holdout configs/splits/t18_holdout_episodes.txt \
        --out runs/t46-step-zero

The rule is ``T46_RULE_V1`` in ``docs/preregistration/PR-12-step-zero-anchor-heterogeneity.md``,
committed in ``6b4f836`` before this file existed. PR-12 registers exactly ONE prediction — that
V-chain clears L1 — because the diagnostic half was already measured (``docs/smoothness-ratio-audit.md``)
and registering a prediction whose answer exists on disk is not a pre-registration.

THE ADAPTER IS IMPORTED, NEVER RE-IMPLEMENTED, and so is the delay. ``commanded_to_chunk``,
``raw_anchor_indices``, ``read_raw_episode`` and the gripper mapping come from
``scripts/eval_t39_baseline.py``; ``delayed_oracle_action_chunks`` and ``trim_pairs`` come from
``scripts/sweep_label_anchoring.py``, so the anchor ``d`` this probe runs at is literally T-44's
code rather than a second copy of it. The only new things here are the two cells.

WHAT V-CHAIN ACTUALLY CHANGES — one argument.

    unmodified   commanded_to_chunk(action[s : s+T],  state[index])
    V-chain      commanded_to_chunk(action[s : s+T],  action[s-1])

``commanded_to_chunk`` builds ``q_from = [canonical_q(anchor)] + q_cmd[:-1]`` and returns
``q_cmd - q_from`` (``eval_t39_baseline.py:252-255``). The anchor therefore appears in row 0 and
NOWHERE ELSE, so passing the previous COMMAND instead of the measured STATE changes ``targets[0]``
and leaves rows 1..T-1 bit-identical **by construction rather than by care**. G0.3 checks it at
runtime anyway, because "by construction" is what the last three of these experiments each believed
about something.

A DESIGN DEFECT IN PR-12, FOUND BY THIS FILE'S TESTS BEFORE ANY CELL EXISTED, AND RECORDED HERE
RATHER THAN BY AMENDING THE PRE-REGISTRATION. PR-12 §3 makes ``d = -2`` the primary anchor and
describes V-chain as changing one thing. At ``d != 0`` it changes two:

    unmodified   anchor = state[index]            the eval timestamp's state
    V-chain      anchor = action[index + d - 1]   the command preceding the SLICE

The slice moved by ``d``, so the command preceding it sits ``d`` steps from the eval timestamp —
under perfect tracking, ``state[index + d]``. "Homogenise step 0" and "keep the anchor at the eval
timestamp" become contradictory requirements once the slice has moved, and no third definition
satisfies both. ``test_at_a_nonzero_delay_v_chain_also_moves_the_anchor_and_that_is_a_confound``
pins the confound at exactly ``d`` steps and no more, which is what makes it reportable.

**So the unconfounded test of PR-12's P2 is the ``d = 0`` cell**, and the ``d = -2`` cell that
``T46_RULE_V1`` reads for its verdict is a JOINT test of the delay and the homogenisation. PR-12 §6
already requires both anchors on both halves to be recorded, so the unconfounded reading is
available without touching the rule — and the result document must say which is which instead of
quoting the verdict alone. This is the same handling PR-11's tie-break defect got: fixed in the
driver, pinned by a test, reported in the result, **not** back-fitted into the registered file.

WHY V-MASK CAN BE A SLICE AND NOT A SECOND SCORER. ``bench_metrics``'s repeat baseline is
``_causal_previous_action`` (``benchmark.py:359-377``), which takes the previous chunk's step
``clip(stride-1, 0, T-1)``. Our eval chunks are non-overlapping (``convert_lerobot_g1.relabel_chunks``
steps by ``chunk_steps``), so ``stride == T``: unmasked that clips to ``T-1``, the previous chunk's
last step; masked to ``T-1`` steps it clips to ``T-2``, which — because masking dropped index 0 and
shifted everything down by one — IS the same original element. The masked run therefore scores
against the identical repeat baseline rather than a shifted one, and
``tests/test_probe_step_zero_anchor.py`` pins that rather than leaving it as this paragraph.
The runtime guard for the same thing is the ``stride == chunk_steps`` assertion in :func:`mask_step_zero`.

NO WITNESS IS LOADED, for the reason ``sweep_label_anchoring`` gives: every arm here is ground
truth, no model is trained or consulted, so there is nothing a train/holdout leak could contaminate.
The holdout file is the episode list and the A/B split key.
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
from wam.evaluation.offline import ChunkPrediction  # noqa: E402
from wam.interfaces.schema import ActionChunk  # noqa: E402

MATERIAL_FLOOR_PP = 10.0
"""Borrowed from ``I8_RULE_V3`` via PR-07 §5 for the fourth time, so the floor cannot be the finding."""

BRIDGE_TOLERANCE_PP = 0.5
"""PR-12 §6 G0.1. Wider than that and this is not the same measurement as T-44/T-45."""

ORACLE_STATE_FLOOR_PCT = 90.0
"""PR-12 §6 G0.2, inherited from PR-07 §5."""

T44_UNMODIFIED_L1 = {
    (-2, "A"): -224.89,
    (-2, "B"): -379.68,
    (0, "A"): -253.70,
    (0, "B"): -410.03,
}
"""T-44/T-45's unmodified cells, the four numbers G0.1 must reproduce. PR-12 §6."""

COHERENCE_FLOOR_PCT = 50.0
"""PR-12 §6 verdict X. Below this, the instrument is misunderstood and no verdict is worth having."""

BENCH_SPECS = ("0.1.0", "0.2.0")
"""0.1.0 is what the rule reads — every archived number was scored under it. 0.2.0 rides along so
PR-12 §5C's bland-side exposure is on record without a re-score, exactly as T-44 did it."""


def _load_script(name: str) -> Any:
    """Import a sibling script as a module, so the adapter is used and not copied."""
    import importlib.util

    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def chained_oracle_action_chunks(
    eval_t39: Any,
    reader: Any,
    raw: dict[str, np.ndarray],
    chunk_steps: int,
    mapping: Any,
    convert: Any,
    *,
    delay: int,
) -> dict[int, Any]:
    """V-chain: ``{t_ns: chunk}`` anchored on the PREVIOUS COMMAND instead of the measured state.

    This is ``sweep_label_anchoring.delayed_oracle_action_chunks`` with the anchor argument
    changed from ``state[index]`` to ``action[start - 1]``. Everything else — the command slice,
    the delay, the gripper path, the chaining for ``t > 0`` — is identical, and the eligibility
    test additionally requires ``start >= 1`` so the previous command exists.

    Under perfect tracking (``action[i] == q[i+1]``, the premise ``commanded_to_chunk``'s docstring
    states) ``state[index] == action[index - 1]`` and the two anchorings coincide EXACTLY. That is
    PR-12 §3's claim 1 and ``tests/test_probe_step_zero_anchor.py`` asserts it on a synthetic
    perfectly-tracking episode — if it ever fails, V-chain is a change of premise and this probe is
    asking a different question than it says it is.
    """
    anchors = eval_t39.raw_anchor_indices(reader, raw)
    action = np.asarray(raw["action"], dtype=np.float32)
    chunks: dict[int, Any] = {}
    for chunk, _prefix, t_ns in reader.read_actions():
        index = anchors[int(t_ns)]
        start = index + delay
        if start < 1 or start + chunk_steps > action.shape[0]:
            continue
        chunks[int(t_ns)] = eval_t39.commanded_to_chunk(
            action[start : start + chunk_steps],
            action[start - 1],
            dt_s=float(chunk.dt_s),
            mapping=mapping,
            convert=convert,
        )
    return chunks


def mask_step_zero(predictions: list[ChunkPrediction]) -> list[ChunkPrediction]:
    """V-mask: drop step 0 from BOTH arms of every prediction. Changes no label, only what is scored.

    Both arms, never the prediction alone. The target has no discontinuity at step 0 — it is a
    homogeneous first difference at every index — so dropping its step 0 costs it a legitimate term
    and biases the comparison AGAINST the finding. It is the conservative direction and it is the
    one the peer's independent audit also took.

    Masking the ``ChunkPrediction`` rather than re-implementing ``bench_metrics`` is what keeps
    every baseline masked identically: ``zero_mse`` reads the same sliced target, ``repeat_mse``
    reads ``_causal_previous_action`` over the sliced chunk list, and the critical-quantile
    threshold is recomputed from the sliced motion energies. PR-12 §5B is the trap this addresses —
    removing the largest element of a sum and reporting the sum got smaller is arithmetic, not a
    finding, and it is only a finding if the RATIO against an identically-masked baseline moves.
    """
    masked: list[ChunkPrediction] = []
    for pred in predictions:
        if pred.target.targets.shape[0] < 2 or pred.predicted.targets.shape[0] < 2:
            raise SystemExit("mask_step_zero needs chunks of at least 2 steps")
        masked.append(
            ChunkPrediction(
                predicted=_drop_first(pred.predicted),
                target=_drop_first(pred.target),
                episode_id=pred.episode_id,
                t_ns=pred.t_ns,
            )
        )
    return masked


def _drop_first(chunk: ActionChunk) -> ActionChunk:
    return ActionChunk(
        mode=chunk.mode,
        targets=chunk.targets[1:].copy(),
        gripper_target=chunk.gripper_target[1:].copy(),
        dt_s=chunk.dt_s,
        schema_version=chunk.schema_version,
    )


def per_step_profile(predictions: list[ChunkPrediction]) -> list[float]:
    """Mean squared error at each within-chunk index — ``benchmark.py:525,563``'s ``per_step_mse``.

    Recomputed here because ``BenchReport`` exposes only its ratio (``horizon_ratio``). The driver
    cross-checks ``profile[-1] / profile[0]`` against the reported ``horizon_ratio``, so this is
    pinned to the scorer's own arithmetic rather than merely resembling it.
    """
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


def step_zero_share(profile: list[float]) -> float:
    """Fraction of the summed per-step MSE carried by index 0, in percent. PR-12 §6 verdict X."""
    total = float(sum(profile))
    return 100.0 * profile[0] / total if total > 0.0 else 0.0


def _score(predictions: list[Any], run_name: str) -> dict[str, Any]:
    """The ladder rungs plus the diagnostics, under both bench specs, plus the per-step profile."""
    bench = bench_metrics(predictions, run_name=run_name, spec_version=BENCH_SPECS[0])
    alt = bench_metrics(predictions, run_name=run_name, spec_version=BENCH_SPECS[1])
    profile = per_step_profile(predictions)
    return {
        "spec_0_2_0": {
            "skill_vs_repeat_pct": float(alt.skill_vs_repeat_pct),
            "ci_skill_vs_repeat_pct": float(alt.ci_skill_vs_repeat_pct),
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
        "step_zero_share_pct": step_zero_share(profile),
    }


def _verdict(cells: dict[str, dict[str, dict]], primary: int) -> dict[str, Any]:
    """``T46_RULE_V1``, PR-12 §6. Precedence fixed IN THE PRE-REGISTRATION: X, then C, then D, then I.

    Unlike T-44 — where PR-10 left precedence open and the driver had to decide it in a docstring —
    PR-12 §6 fixes the order in the registered file, so this function only applies it.
    """
    key = str(primary)
    unmod_b = cells["unmodified"][key + "|B"]
    chain_b = cells["v_chain"][key + "|B"]

    # THE SHARE IS A PROPERTY OF THE UNMODIFIED PROFILE, AND CAN ONLY BE.
    #
    # PR-12 §6 words verdict X as "V-mask's step-0 share of MSE", and read literally that is a
    # quantity that cannot exist: V-mask's profile HAS no step 0 — dropping it is what V-mask is.
    # `step_zero_share_pct` on a masked cell is the share of the first SURVIVING step, which is
    # step 1, and it is ~6 % for the same reason every non-zero step is ~6 %. §4 names the
    # registered quantity unambiguously — "step 0 dominating the MSE sum ... recorded, not
    # predicted" — and it is a property of the unmodified per-step profile.
    #
    # The first run of this driver read the masked cell and returned X ("coherence failure") off a
    # number that was definitionally the wrong one. That is recorded in the result document rather
    # than quietly corrected: a verdict function that reads the wrong field produces a finite,
    # plausible verdict, which is this project's recurring failure mode and not a footnote.
    share = cells["unmodified"][key + "|A"]["step_zero_share_pct"]
    if share < COHERENCE_FLOOR_PCT:
        return {
            "verdict": "X",
            "step_zero_share_pct": share,
            "reading": (
                f"COHERENCE FAILURE. Step 0 carries {share:.1f} % of the per-step MSE sum, below "
                f"the {COHERENCE_FLOOR_PCT:.0f} % floor. The published jerk decomposition puts "
                "96.7 % of the predicted jerk sum in index 0 and horizon_ratio ~0.006 implies "
                "~92 % of the MSE sum, so a number this low means PR-12 has misunderstood its own "
                "instrument. Nothing is concluded and nothing is licensed, whatever V-chain did."
            ),
        }

    gain = chain_b["skill_vs_repeat_pct"] - unmod_b["skill_vs_repeat_pct"]
    if chain_b["skill_vs_repeat_pct"] > 0.0 and gain >= MATERIAL_FLOOR_PP:
        return {
            "verdict": "C",
            "step_zero_share_pct": share,
            "b_gain_pp": gain,
            "reading": (
                f"V-chain clears L1 on held-out half B: {chain_b['skill_vs_repeat_pct']:+.2f} % "
                f"against the unmodified {unmod_b['skill_vs_repeat_pct']:+.2f} %, a gain of "
                f"{gain:.2f} pp at or above the borrowed floor of {MATERIAL_FLOOR_PP} pp. The "
                "deficit was an anchoring heterogeneity and the repair is one line of the adapter. "
                "This licenses a defect report against commanded_to_chunk's step-0 anchoring and "
                "licenses PROPOSING a relabel — carrying PR-12 §3's stated cost, that the chunk "
                "loses its only tie to measured state. It relabels nothing, retro-validates none "
                "of the fourteen negatives, and is not a licence to train."
            ),
        }
    return {
        "verdict": "D",
        "step_zero_share_pct": share,
        "b_gain_pp": gain,
        "reading": (
            f"The diagnosis holds and the cheap fix does not reach it. Step 0 carries "
            f"{share:.1f} % of the per-step MSE sum, but V-chain gives "
            f"{chain_b['skill_vs_repeat_pct']:+.2f} % on held-out half B against the unmodified "
            f"{unmod_b['skill_vs_repeat_pct']:+.2f} % — a gain of {gain:.2f} pp, and C needs B "
            f"above L1 AND a gain of at least {MATERIAL_FLOOR_PP} pp. So the tracking offset is "
            "real but not constant: it does not cancel between two consecutive commands either. "
            "Licenses measuring the offset's structure — per-joint, velocity-dependent — and "
            "nothing else."
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
    sweep = _load_script("sweep_label_anchoring")
    convert = eval_t39._load_script("convert_lerobot_g1")

    from wam.data.episode import EpisodeReader, list_episodes

    # Order comes from the committed file, not the filesystem: the A/B split is "even index in
    # t18_holdout_episodes.txt" and a directory listing gives the same SET with a different SPLIT.
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
    halves = {
        "A": [d for i, d in enumerate(episode_dirs) if i % 2 == 0],
        "B": [d for i, d in enumerate(episode_dirs) if i % 2 == 1],
    }

    first = EpisodeReader(episode_dirs[0])
    chunk_steps = args.chunk_steps or eval_t39.episode_chunk_steps(first)
    mapping = eval_t39.gripper_mapping_from_manifest(first.manifest, convert)

    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"T-46 step-zero probe | {len(episode_dirs)} holdout episodes "
        f"(A={len(halves['A'])}, B={len(halves['B'])}) | chunk_steps {chunk_steps} | "
        f"gripper {mapping.kind}"
    )

    started = time.perf_counter()
    cache: dict[str, dict[str, Any]] = {}
    for episode_dir in episode_dirs:
        reader = EpisodeReader(episode_dir)
        episode_id = reader.manifest.episode_id
        pairs = build_eval_pairs(episode_dir, args.camera, chunk_steps, num_frames=None)
        cache[episode_id] = {
            "reader": reader,
            "pairs": sweep.trim_pairs(pairs),
            "raw": eval_t39.read_raw_episode(args.raw_dataset, episode_id),
        }
    print(f"  loaded {len(cache)} episodes in {time.perf_counter() - started:.1f}s")

    def chunks_for(entry: dict, cell: str, delay: int) -> dict[int, Any]:
        reader = entry["reader"]
        if cell == "oracle_state":
            return eval_t39.oracle_state_chunks(reader, chunk_steps, mapping)
        if cell == "v_chain":
            return chained_oracle_action_chunks(
                eval_t39, reader, entry["raw"], chunk_steps, mapping, convert, delay=delay
            )
        return sweep.delayed_oracle_action_chunks(
            eval_t39, reader, entry["raw"], chunk_steps, mapping, convert, delay=delay
        )

    def run(cell: str, half: str, delay: int) -> dict[str, Any]:
        predictions: list[Any] = []
        for episode_dir in halves[half]:
            entry = cache[EpisodeReader(episode_dir).manifest.episode_id]
            if not entry["pairs"]:
                continue
            source = "unmodified" if cell == "v_mask" else cell
            chunks = chunks_for(entry, source, delay)
            policy = eval_t39.ChunkLookupPolicy(
                chunks, episode_id=entry["reader"].manifest.episode_id
            )
            predictions.extend(evaluate_policy(policy, entry["pairs"]))
        if not predictions:
            raise SystemExit(f"no chunks scored for cell={cell} half={half} delay={delay}")
        if cell == "v_mask":
            predictions = mask_step_zero(predictions)
        return _score(predictions, run_name=f"t46-{cell}-{half}-d{delay:+d}")

    results: dict[str, Any] = {
        "rule": "T46_RULE_V1",
        "preregistration": "docs/preregistration/PR-12-step-zero-anchor-heterogeneity.md",
        "dataset": str(args.dataset),
        "raw_dataset": str(args.raw_dataset),
        "holdout": str(args.holdout),
        "chunk_steps": chunk_steps,
        "gripper_mapping": mapping.kind,
        "half_a": [d.name for d in halves["A"]],
        "half_b": [d.name for d in halves["B"]],
    }
    gates: list[str] = []

    # -- G0.2  oracle_state ------------------------------------------------------------------
    print("\n=== G0.2  oracle_state, unmodified, d=0 (floor 90 %)")
    g0_state = run("oracle_state", "A", 0)
    g0_state_b = run("oracle_state", "B", 0)
    results["g0_oracle_state"] = {"A": g0_state, "B": g0_state_b}
    print(f"    A {g0_state['skill_vs_repeat_pct']:+.2f} %   B {g0_state_b['skill_vs_repeat_pct']:+.2f} %")
    if min(g0_state["skill_vs_repeat_pct"], g0_state_b["skill_vs_repeat_pct"]) < ORACLE_STATE_FLOOR_PCT:
        gates.append(
            f"G0.2 FAILED: oracle_state below {ORACLE_STATE_FLOOR_PCT} %. The harness changed; "
            "this is a code fix, not a threshold."
        )

    # -- G0.3  both cells reach the array, and ONLY where they should -------------------------
    print("\n=== G0.3  V-chain touches row 0 and nothing else")
    g0_rows: dict[str, Any] = {}
    for delay in (-2, 0):
        row0_rms: list[float] = []
        rest_max = 0.0
        keys_match = True
        for entry in cache.values():
            unmod = chunks_for(entry, "unmodified", delay)
            chain = chunks_for(entry, "v_chain", delay)
            # RESTRICTED TO THE SCORED TIMESTAMPS, and that restriction is the registered one.
            # V-chain needs `start >= 1`, so at d = 0 it drops each episode's chunk at index 0 —
            # which `trim_pairs` drops from scoring anyway, for every cell, before any number
            # exists. Comparing the untrimmed dictionaries compares chunks that enter nothing.
            # PR-12 §3 puts the requirement on the RETAINED count ("must match the unmodified
            # cell's ... checked by G0.1"), and the retained set is what this now compares;
            # `g0_scored_chunk_counts` below asserts that count directly, which is stricter than
            # the key check was in the dimension the pre-registration actually names.
            scored = {int(obs.state.timestamp_ns) for obs, _target, _eid in entry["pairs"]}
            if (set(unmod) & scored) != (set(chain) & scored):
                keys_match = False
            for t_ns in sorted(set(unmod) & set(chain) & scored):
                a = np.asarray(unmod[t_ns].targets, dtype=np.float64)
                b = np.asarray(chain[t_ns].targets, dtype=np.float64)
                row0_rms.append(float(np.sqrt(((a[0] - b[0]) ** 2).mean())))
                rest_max = max(rest_max, float(np.abs(a[1:] - b[1:]).max()))
        mean_rms = float(np.mean(row0_rms)) if row0_rms else 0.0
        g0_rows[str(delay)] = {
            "row0_rms_mean": mean_rms,
            "rows_1_plus_max_abs_diff": rest_max,
            "chunk_keys_identical": keys_match,
        }
        print(
            f"  d={delay:+d}  row0 RMS {mean_rms:.4e}   rows 1.. max|diff| {rest_max:.3e}   "
            f"keys identical: {keys_match}"
        )
        if mean_rms <= 0.0:
            gates.append(
                f"G0.3 FAILED at d={delay:+d}: V-chain is a no-op on row 0. A manipulation that "
                "does not reach the array produces a flat grid that reads as a confident negative."
            )
        if rest_max != 0.0:
            gates.append(
                f"G0.3 FAILED at d={delay:+d}: V-chain changed rows 1.. by up to {rest_max:.3e}. "
                "It must touch row 0 and nothing else — a V-chain that re-chained every step "
                "would be a different experiment wearing this one's name."
            )
        if not keys_match:
            gates.append(
                f"G0.3 FAILED at d={delay:+d}: V-chain and the unmodified cell retained different "
                "chunk sets, so their scores are not a comparison."
            )
    results["g0_vchain_rows"] = g0_rows

    # -- the cells ---------------------------------------------------------------------------
    cells: dict[str, dict[str, dict]] = {"unmodified": {}, "v_mask": {}, "v_chain": {}}
    for delay in (-2, 0):
        tag = "PRIMARY" if delay == -2 else "secondary"
        print(
            f"\n=== anchor d = {delay:+d}  ({tag})\n"
            f"{'cell':>12}  {'A L1':>10}  {'B L1':>10}  {'A L2':>10}  {'A smooth':>9}  "
            f"{'A horiz':>9}  {'A step0%':>9}"
        )
        for cell in ("unmodified", "v_mask", "v_chain"):
            for half in ("A", "B"):
                cells[cell][f"{delay}|{half}"] = run(cell, half, delay)
            a = cells[cell][f"{delay}|A"]
            b = cells[cell][f"{delay}|B"]
            print(
                f"{cell:>12}  {a['skill_vs_repeat_pct']:>+10.2f}  {b['skill_vs_repeat_pct']:>+10.2f}  "
                f"{a['ci_skill_vs_repeat_pct']:>+10.2f}  {a['smoothness_ratio']:>9.3f}  "
                f"{a['horizon_ratio']:>9.5f}  {a['step_zero_share_pct']:>9.2f}"
            )

        # G0.3, second half: V-mask must score 15 steps against the unmodified 16.
        unmasked_steps = len(cells["unmodified"][f"{delay}|A"]["per_step_mse"])
        masked_steps = len(cells["v_mask"][f"{delay}|A"]["per_step_mse"])
        if masked_steps != unmasked_steps - 1:
            gates.append(
                f"G0.3 FAILED at d={delay:+d}: V-mask scored {masked_steps} steps against the "
                f"unmodified {unmasked_steps}; it must be exactly one fewer."
            )

        # PR-12 §3's actual retained-count requirement, asserted on the numbers themselves rather
        # than on the chunk dictionaries that feed them. Three cells scored over different chunk
        # sets are not a comparison, whatever their key sets looked like before trimming.
        counts = {
            cell: cells[cell][f"{delay}|{half}"]["num_chunks"]
            for cell in ("unmodified", "v_mask", "v_chain")
            for half in ("A", "B")
        }
        results.setdefault("g0_scored_chunk_counts", {})[str(delay)] = counts
        for half in ("A", "B"):
            scored = {
                cell: cells[cell][f"{delay}|{half}"]["num_chunks"]
                for cell in ("unmodified", "v_mask", "v_chain")
            }
            if len(set(scored.values())) != 1:
                gates.append(
                    f"G0.1 FAILED at d={delay:+d} half {half}: cells scored different chunk "
                    f"counts {scored}. They are not a comparison."
                )

        # G0.1, the bridge to T-44/T-45.
        for half in ("A", "B"):
            expected = T44_UNMODIFIED_L1[(delay, half)]
            got = cells["unmodified"][f"{delay}|{half}"]["skill_vs_repeat_pct"]
            drift = got - expected
            results.setdefault("g0_bridge_drift_pp", {})[f"{delay}|{half}"] = drift
            if abs(drift) > BRIDGE_TOLERANCE_PP:
                gates.append(
                    f"G0.1 FAILED at d={delay:+d} half {half}: unmodified L1 {got:+.2f} against "
                    f"T-44's {expected:+.2f}, drift {drift:+.3f} pp (tolerance "
                    f"±{BRIDGE_TOLERANCE_PP}). This is not the same measurement."
                )
    results["cells"] = cells

    # The per-step profile is pinned to the scorer's own arithmetic rather than merely resembling
    # it: horizon_ratio is last/first of exactly this vector (benchmark.py:563-565).
    checks: dict[str, float] = {}
    for cell in ("unmodified", "v_chain"):
        for delay in (-2, 0):
            entry = cells[cell][f"{delay}|A"]
            profile = entry["per_step_mse"]
            implied = profile[-1] / profile[0] if profile[0] > 0 else float("nan")
            checks[f"{cell}|{delay}"] = implied - entry["horizon_ratio"]
    results["profile_vs_horizon_ratio_residual"] = checks
    worst = max(abs(v) for v in checks.values())
    print(f"\n  per-step profile reproduces horizon_ratio to {worst:.2e}")
    if worst > 1e-9:
        gates.append(
            f"per_step_profile disagrees with the scorer's horizon_ratio by {worst:.2e}; the "
            "profile is not the vector the metric is computed from and step_zero_share is unsafe."
        )

    results["verdict"] = (
        {"verdict": "INVALID", "reading": " ".join(gates)} if gates else _verdict(cells, -2)
    )

    (args.out / "probe.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    verdict = results["verdict"]
    print(f"\n=== T46_RULE_V1 -> {verdict['verdict']}")
    print(f"    {verdict['reading']}")
    print(f"\nwrote {args.out / 'probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
