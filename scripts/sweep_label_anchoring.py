#!/usr/bin/env python3
"""T-44 / PR-10 — sweep the commanded chunk's source index and ask whether the mismatch is timing.

    scripts/sweep_label_anchoring.py \
        --dataset datasets/gr00t-apple-full \
        --raw-dataset ~/wam-t041/raw/GR00T-N1.7-AppleToPlate \
        --holdout configs/splits/t18_holdout_episodes.txt \
        --out runs/t44-anchoring-sweep

T-39 returned VOID (labels): the corpus's own commanded ``action`` column fails L1 by −359.41 pp
under our scorer while ``oracle_state`` scores a bit-exact ``mse 0.0``. PR-07-RESULT recorded the
constant-lag reading of that failure as an *interpretation* and explicitly refused to establish it.
This is the experiment that separates it from the alternative, and the rule is ``T44_RULE_V1`` in
``docs/preregistration/PR-10-label-anchoring-delay-sweep.md``, committed before this file existed.

THE ADAPTER IS IMPORTED, NEVER RE-IMPLEMENTED. ``commanded_to_chunk``, ``raw_anchor_indices``,
``read_raw_episode`` and the gripper mapping all come from ``scripts/eval_t39_baseline.py``, and
the bench specs from ``wam.evaluation``. A sweep run through a second copy of the adapter measures
the copy — which is the same argument PR-07 §3 makes about the trainer, applied one level down.
The only thing written here is the delay itself, and :func:`delayed_oracle_action_chunks` is a
seven-line variant of ``oracle_action_chunks`` whose ONLY difference is the ``+ delay`` on the
command slice.

WHAT MOVES AND WHAT DOES NOT (PR-10 §3). The command slice moves. The anchor state does not: it
stays ``state[index]``, the position the chunk actually starts from, because shifting it too would
change two things at once and make the curve uninterpretable. ``tests/test_sweep_label_anchoring.py``
kills that mutant.

THE CHUNK SET IS THE INTERSECTION OVER THE WHOLE SWEEP, NOT PER-DELAY. A shifted index runs off the
end of the episode for ``delay > 0`` and off the front for ``delay < 0``, so the eligible chunks
differ by delay — and nine scores over nine different chunk sets are not a comparison. The first
and last eval pair of every episode are dropped for every arm and every delay, uniformly, which
covers ``|delay| <= 4`` at chunk length 16 with room to spare. The retained count is therefore
BELOW PR-07's 1 040 and nothing here is directly comparable to that table; ``--bridge`` scores
``delay=0`` on the full set once, which is the only bridge between the two documents.

NO WITNESS IS LOADED, AND THAT IS NOT AN OVERSIGHT. ``eval_t39_baseline`` verifies the split
because it scores a *checkpoint* and the holdout has to be provably unseen. Both arms here are
ground truth: no model is trained, loaded or consulted, so there is nothing for a train/holdout
leak to contaminate. The holdout file is used only as the episode list and as the A/B split key.
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
"""Borrowed from ``I8_RULE_V3`` via PR-07 §5, not coined here, so the floor cannot be the finding."""

PR07_ORACLE_ACTION_L1 = -359.41
"""PR-07-RESULT's headline, on the full 1 040 chunks. The bridge in §5 G0.2 must reproduce it."""

BRIDGE_TOLERANCE_PP = 0.5
"""±0.5 pp, PR-10 §5 G0.2. Wider than that and this is not the same measurement."""

ORACLE_STATE_FLOOR_PCT = 90.0
"""PR-10 §5 G0.1, inherited from PR-07 §5: below this the harness changed and no verdict issues."""


def _load_eval_t39() -> Any:
    """Import ``scripts/eval_t39_baseline.py`` as a module — the adapter lives there."""
    import importlib.util

    path = _REPO_ROOT / "scripts" / "eval_t39_baseline.py"
    spec = importlib.util.spec_from_file_location("eval_t39_baseline", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_t39_baseline"] = module
    spec.loader.exec_module(module)
    return module


def delayed_oracle_action_chunks(
    eval_t39: Any,
    reader: Any,
    raw: dict[str, np.ndarray],
    chunk_steps: int,
    mapping: Any,
    convert: Any,
    *,
    delay: int,
) -> dict[int, Any]:
    """``{t_ns: chunk}`` from the source ``action`` column, read ``delay`` steps away.

    This is ``eval_t39_baseline.oracle_action_chunks`` with one term added. ``delay=0`` must
    produce a byte-identical result to it, and the tests assert exactly that rather than trusting
    the reading of this docstring.

    The anchor stays ``state[index]``. See the module docstring: moving it as well would change
    two things at once, and a curve produced that way answers no question anyone asked.
    """
    anchors = eval_t39.raw_anchor_indices(reader, raw)
    action = np.asarray(raw["action"], dtype=np.float32)
    state = np.asarray(raw["state"], dtype=np.float32)
    chunks: dict[int, Any] = {}
    for chunk, _prefix, t_ns in reader.read_actions():
        index = anchors[int(t_ns)]
        start = index + delay
        if start < 0 or start + chunk_steps > action.shape[0]:
            continue
        chunks[int(t_ns)] = eval_t39.commanded_to_chunk(
            action[start : start + chunk_steps],
            state[index],
            dt_s=float(chunk.dt_s),
            mapping=mapping,
            convert=convert,
        )
    return chunks


def trim_pairs(pairs: list[Any]) -> list[Any]:
    """Drop the first and last eval pair — the intersection rule, PR-10 §3.

    Applied to every arm and every delay identically. Applying it per-delay instead would score
    each delay on the chunks that delay happens to reach, which flatters whichever delay reaches
    fewest; ``tests/test_sweep_label_anchoring.py`` kills that mutant too.
    """
    if len(pairs) <= 2:
        return []
    return pairs[1:-1]


BENCH_SPECS = ("0.1.0", "0.2.0")
"""Both, because PR-10 §5 says "under both bench specs" and that is not optional.

**0.1.0 is what the rule reads.** It is the spec every archived number in this repo was scored
under and the one ``T39_RULE_V1`` read, so it is the one a comparison to PR-07-RESULT has to use;
every verdict-bearing number below is 0.1.0's. 0.2.0 is written beside it — one extra call over
predictions already in memory — so the two-sided L4 band is on record without a re-score, exactly
as ``eval_t39_baseline.BENCH_SPECS_WRITTEN`` does it. Deciding afterwards which spec to report
would not be free."""


def _score(predictions: list[Any], run_name: str) -> dict[str, Any]:
    """The two ladder rungs plus the two diagnostics, under both bench specs."""
    bench = bench_metrics(predictions, run_name=run_name, spec_version=BENCH_SPECS[0])
    alt = bench_metrics(predictions, run_name=run_name, spec_version=BENCH_SPECS[1])
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
    }


def _verdict(curve_a: dict[int, dict], curve_b: dict[int, dict], max_delay: int) -> dict[str, Any]:
    """``T44_RULE_V1``, PR-10 §5, applied to the two curves.

    PRECEDENCE, AND ONE HONEST GAP. PR-10's verdict table lists T, J, E, I but does not fix what
    happens when two conditions hold at once. The driver applies **E first**, then J, then T, then
    I, because E's own text is "Nothing is concluded" — if the best delay sits on the window edge,
    the window is the wrong window and no other reading of the curve is safe, including J's
    "it is not a delay". This precedence was chosen while writing the driver and BEFORE any curve
    was computed, and it is recorded here rather than in PR-10 because amending a pre-registration
    after the fact is exactly what the repo's rules forbid.
    """
    best_delay = max(curve_a, key=lambda d: curve_a[d]["skill_vs_repeat_pct"])
    best_a = curve_a[best_delay]["skill_vs_repeat_pct"]
    any_l1_on_a = any(v["skill_vs_repeat_pct"] > 0.0 for v in curve_a.values())

    if abs(best_delay) == max_delay:
        return {
            "verdict": "E",
            "d_star": best_delay,
            "reading": (
                f"d* = {best_delay:+d} sits on the sweep's endpoint, so the optimum may lie "
                f"outside the window. Nothing is concluded. PR-10 §5 permits ONE extension, to "
                f"±{2 * max_delay}, re-read under the same rule. There is no second extension."
            ),
        }
    if not any_l1_on_a:
        return {
            "verdict": "J",
            "d_star": best_delay,
            "reading": (
                "No delay in the swept window clears L1 on half A. The commanded and executed "
                "spaces are not a shifted copy of one another, so the anchor is not the defect. "
                f"The object of study is the jerk (smoothness_ratio "
                f"{curve_a[0]['smoothness_ratio']:.2f} at d=0), and PR-04's collection spec — "
                "what KIND of data — becomes the live question."
            ),
        }
    b_star = curve_b[best_delay]["skill_vs_repeat_pct"]
    b_zero = curve_b[0]["skill_vs_repeat_pct"]
    gain = b_star - b_zero
    if best_delay != 0 and b_star > 0.0 and gain >= MATERIAL_FLOOR_PP:
        return {
            "verdict": "T",
            "d_star": best_delay,
            "b_gain_pp": gain,
            "reading": (
                f"Confirmed on held-out half B: L1 {b_star:+.2f} % at d* = {best_delay:+d}, a gain "
                f"of {gain:.2f} pp over B's own d=0, at or above the borrowed floor of "
                f"{MATERIAL_FLOOR_PP} pp. Our labels are anchored {best_delay:+d} step(s) off this "
                "corpus's controller. This licenses a defect report naming d* and licenses "
                "PROPOSING a relabel; it does not relabel anything and it retro-validates none of "
                "the fourteen negatives."
            ),
        }
    return {
        "verdict": "I",
        "d_star": best_delay,
        "b_gain_pp": gain,
        "reading": (
            f"Indeterminate. Best on A is d* = {best_delay:+d} at {best_a:+.2f} %, but held-out "
            f"half B gives {b_star:+.2f} % there against {b_zero:+.2f} % at d=0 — a gain of "
            f"{gain:.2f} pp. T needs d* != 0, B above L1, AND a gain of at least "
            f"{MATERIAL_FLOOR_PP} pp. Nothing is licensed and nothing is relabelled."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="converted WAM episodes")
    parser.add_argument("--raw-dataset", type=Path, required=True, help="the LeRobot source")
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--camera", default="ego")
    parser.add_argument("--max-delay", type=int, default=4, help="sweep -N..+N; PR-10 fixes N=4")
    parser.add_argument("--chunk-steps", type=int)
    parser.add_argument(
        "--skip-bridge",
        action="store_true",
        help="skip the full-1040-chunk d=0 bridge. Only for development — G0.2 is a gate.",
    )
    args = parser.parse_args(argv)

    eval_t39 = _load_eval_t39()
    convert = eval_t39._load_script("convert_lerobot_g1")

    from wam.data.episode import EpisodeReader, list_episodes

    # ORDER COMES FROM THE COMMITTED FILE, NOT FROM THE FILESYSTEM. The A/B split is "even index
    # in t18_holdout_episodes.txt", which is only reproducible if the order is the file's. A
    # directory listing would give the same SET and a different SPLIT.
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
    half_a = [d for i, d in enumerate(episode_dirs) if i % 2 == 0]
    half_b = [d for i, d in enumerate(episode_dirs) if i % 2 == 1]

    first = EpisodeReader(episode_dirs[0])
    chunk_steps = args.chunk_steps or eval_t39.episode_chunk_steps(first)
    mapping = eval_t39.gripper_mapping_from_manifest(first.manifest, convert)
    if args.max_delay >= chunk_steps:
        raise SystemExit(
            f"--max-delay {args.max_delay} >= chunk_steps {chunk_steps}: dropping one pair at each "
            "end no longer guarantees every retained chunk exists at every delay, which is the "
            "intersection rule PR-10 §3 fixes."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"T-44 sweep | {len(episode_dirs)} holdout episodes (A={len(half_a)}, B={len(half_b)}) | "
        f"chunk_steps {chunk_steps} | gripper {mapping.kind} | delays "
        f"{-args.max_delay}..{args.max_delay}"
    )

    # Pairs and raw parquet are read ONCE per episode and reused across every delay. Rebuilding
    # them per delay would decode the same video nine times and change nothing.
    started = time.perf_counter()
    cache: dict[str, dict[str, Any]] = {}
    for episode_dir in episode_dirs:
        reader = EpisodeReader(episode_dir)
        episode_id = reader.manifest.episode_id
        pairs = build_eval_pairs(episode_dir, args.camera, chunk_steps, num_frames=None)
        cache[episode_id] = {
            "dir": episode_dir,
            "reader": reader,
            "pairs_full": pairs,
            "pairs": trim_pairs(pairs),
            "raw": eval_t39.read_raw_episode(args.raw_dataset, episode_id),
        }
    print(f"  loaded {len(cache)} episodes in {time.perf_counter() - started:.1f}s")

    def run_arm(dirs: list[Path], *, arm: str, delay: int, trimmed: bool) -> dict[str, float]:
        predictions: list[Any] = []
        for episode_dir in dirs:
            entry = cache[EpisodeReader(episode_dir).manifest.episode_id]
            pairs = entry["pairs"] if trimmed else entry["pairs_full"]
            if not pairs:
                continue
            reader = entry["reader"]
            if arm == "oracle_state":
                chunks = eval_t39.oracle_state_chunks(reader, chunk_steps, mapping)
            else:
                chunks = delayed_oracle_action_chunks(
                    eval_t39, reader, entry["raw"], chunk_steps, mapping, convert, delay=delay
                )
            policy = eval_t39.ChunkLookupPolicy(
                chunks, episode_id=reader.manifest.episode_id
            )
            predictions.extend(evaluate_policy(policy, pairs))
        if not predictions:
            raise SystemExit(f"no chunks scored for arm={arm} delay={delay}")
        return _score(predictions, run_name=f"t44-{arm}-d{delay:+d}")

    results: dict[str, Any] = {
        "rule": "T44_RULE_V1",
        "preregistration": "docs/preregistration/PR-10-label-anchoring-delay-sweep.md",
        "dataset": str(args.dataset),
        "raw_dataset": str(args.raw_dataset),
        "holdout": str(args.holdout),
        "chunk_steps": chunk_steps,
        "gripper_mapping": mapping.kind,
        "max_delay": args.max_delay,
        "half_a": [d.name for d in half_a],
        "half_b": [d.name for d in half_b],
    }

    # -- G0, which runs first and can stop everything ---------------------------------------
    print("\n=== G0.1  oracle_state at d=0, trimmed set (floor 90 %)")
    g0a = run_arm(episode_dirs, arm="oracle_state", delay=0, trimmed=True)
    results["g0_oracle_state"] = g0a
    print(f"    L1 {g0a['skill_vs_repeat_pct']:+.2f} %   ({g0a['num_chunks']} chunks)")

    if not args.skip_bridge:
        print(f"\n=== G0.2  bridge: oracle_action at d=0 on the FULL set (PR-07 said "
              f"{PR07_ORACLE_ACTION_L1:+.2f})")
        bridge = run_arm(episode_dirs, arm="oracle_action", delay=0, trimmed=False)
        results["g0_bridge"] = bridge
        drift = bridge["skill_vs_repeat_pct"] - PR07_ORACLE_ACTION_L1
        results["g0_bridge_drift_pp"] = drift
        print(
            f"    L1 {bridge['skill_vs_repeat_pct']:+.2f} %   ({bridge['num_chunks']} chunks)   "
            f"drift {drift:+.3f} pp"
        )

    # -- the sweep ---------------------------------------------------------------------------
    delays = list(range(-args.max_delay, args.max_delay + 1))
    curve_a: dict[int, dict] = {}
    curve_b: dict[int, dict] = {}
    print(f"\n=== sweep, oracle_action, trimmed set\n{'d':>4}  {'A L1':>10}  {'B L1':>10}  "
          f"{'A L2':>10}  {'B L2':>10}  {'A smooth':>9}")
    for delay in delays:
        curve_a[delay] = run_arm(half_a, arm="oracle_action", delay=delay, trimmed=True)
        curve_b[delay] = run_arm(half_b, arm="oracle_action", delay=delay, trimmed=True)
        print(
            f"{delay:>+4}  {curve_a[delay]['skill_vs_repeat_pct']:>+10.2f}  "
            f"{curve_b[delay]['skill_vs_repeat_pct']:>+10.2f}  "
            f"{curve_a[delay]['ci_skill_vs_repeat_pct']:>+10.2f}  "
            f"{curve_b[delay]['ci_skill_vs_repeat_pct']:>+10.2f}  "
            f"{curve_a[delay]['smoothness_ratio']:>9.2f}"
        )
    results["curve_a"] = {str(d): v for d, v in curve_a.items()}
    results["curve_b"] = {str(d): v for d, v in curve_b.items()}

    # -- the rule ----------------------------------------------------------------------------
    gates: list[str] = []
    if g0a["skill_vs_repeat_pct"] < ORACLE_STATE_FLOOR_PCT:
        gates.append(
            f"G0.1 FAILED: oracle_state {g0a['skill_vs_repeat_pct']:+.2f} % < "
            f"{ORACLE_STATE_FLOOR_PCT} %. The harness changed; this is a code fix, not a threshold."
        )
    if not args.skip_bridge and abs(results["g0_bridge_drift_pp"]) > BRIDGE_TOLERANCE_PP:
        gates.append(
            f"G0.2 FAILED: bridge drifted {results['g0_bridge_drift_pp']:+.3f} pp from PR-07's "
            f"{PR07_ORACLE_ACTION_L1:+.2f} (tolerance ±{BRIDGE_TOLERANCE_PP}). This is not the "
            "same measurement and nothing may be compared to PR-07-RESULT."
        )
    if gates:
        results["verdict"] = {"verdict": "INVALID", "reading": " ".join(gates)}
    else:
        results["verdict"] = _verdict(curve_a, curve_b, args.max_delay)

    (args.out / "sweep.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    verdict = results["verdict"]
    print(f"\n=== T44_RULE_V1 -> {verdict['verdict']}")
    print(f"    {verdict['reading']}")
    print(f"\nwrote {args.out / 'sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
