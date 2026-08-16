#!/usr/bin/env python3
"""T-45 / PR-11 — low-pass the commanded column and ask whether the residual jerk is removable.

    scripts/sweep_command_lowpass.py \
        --dataset datasets/gr00t-apple-full \
        --raw-dataset ~/wam-t041/raw/GR00T-N1.7-AppleToPlate \
        --holdout configs/splits/t18_holdout_episodes.txt \
        --out runs/t45-lowpass-sweep

Both PR-10 runs found the same thing and ended on the same sentence: the commanded column leads the
executed state by about two control steps, that offset is real, and what survives every anchoring
either run tried is the **jerk** — 3-5x the executed trajectory's, against an L4 gate of 2.0. A
shift re-indexes a signal; it cannot smooth one. A position-controlled arm, however, *is* a
low-pass filter. This asks whether reconstructing that filter in software closes the rest.

The rule is ``T45_RULE_V1`` in ``docs/preregistration/PR-11-command-lowpass-sweep.md``, committed in
``90a0570`` before this file existed.

EVERYTHING EXCEPT THE FILTER IS IMPORTED. ``delayed_oracle_action_chunks``, ``trim_pairs`` and the
scoring come from ``scripts/sweep_label_anchoring.py`` (T-44); ``commanded_to_chunk``,
``raw_anchor_indices``, ``read_raw_episode``, ``oracle_state_chunks`` and ``ChunkLookupPolicy`` come
from ``scripts/eval_t39_baseline.py`` through it. The filter is applied to the episode's ``action``
array *before* the chunk builder ever sees it, so the builder is not modified at all — which is
also why ``fc = None`` is bit-identical to a T-44 cell rather than merely close to one.

THE FILTER IS THE PREREGISTRATION'S, NOT A CHOICE MADE HERE. See :func:`lowpass_kernel`. Zero phase
because a causal filter's lag would be indistinguishable from the delay PR-10 just measured; numpy
because ``scipy`` is absent from the WAM venv and installing it would move the dependency set every
number in ``docs/benchmark.md`` was produced under; whole-episode because filtering inside a chunk
puts a discontinuity at every boundary, exactly where ``horizon_ratio`` looks.

THE GRID CAN LIE IN ONE PARTICULAR WAY, AND G0.3 IS THE ANSWER. A filter threaded through the call
chain but never actually applied produces a flat grid and a confident verdict that the jerk is
irreducible — the same shape as a real **R** and indistinguishable from it in the output. So every
filtered cell records the RMS change it made to the array, the gate requires that change to be
non-zero and to grow monotonically as the cutoff falls, and the numbers land in the artifact rather
than only in a test. That gate exists because the author of ``PR-10-anchor-delay-sweep.md`` reported
paying a mutation test to catch the analogous defect in the offset knob.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from wam.evaluation import build_eval_pairs, evaluate_policy  # noqa: E402

MATERIAL_FLOOR_PP = 10.0
"""Borrowed from ``I8_RULE_V3`` for the third time rather than coined (PR-11 §5)."""

CUTOFFS_HZ = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0)
"""PR-11 §3, fixed. ``None`` — the no-op control and the bridge to T-44 — is appended by the driver."""

ANCHORS = (-2, 0)
"""``-2`` is PRIMARY (the best anchor both PR-10 runs found); ``0`` is secondary. Taken from T-44
and held fixed: searching anchor and cutoff together is a 2-D garden of forking paths."""

T44_NO_OP = {
    (-2, "A"): -224.89,
    (-2, "B"): -379.68,
    (0, "A"): -253.70,
    (0, "B"): -410.03,
}
"""PR-11 §5 G0.1. The no-op cells must reproduce T-44's curve or this is not the same measurement."""

BRIDGE_TOLERANCE_PP = 0.5
ORACLE_STATE_FLOOR_PCT = 90.0


def _load_sibling(name: str) -> Any:
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def lowpass_kernel(fc_hz: float, fs_hz: float) -> np.ndarray:
    """The PR-11 §3 kernel, spelled exactly as the pre-registration spells it.

    A Hann-windowed sinc. Symmetric, so its phase response is identically zero — the property the
    whole design rests on, and one the tests check rather than take from this sentence. Normalised
    to unit sum so DC passes untouched: without that, every filtered cell would also be scaled, and
    a scale change is precisely the shrinkage confound PR-11 §4 exists to keep out.
    """
    if not fc_hz > 0 or not fc_hz < fs_hz / 2.0:
        raise SystemExit(f"cutoff {fc_hz} Hz must lie in (0, {fs_hz / 2}) — Nyquist is the no-op")
    half = int(np.ceil(2.0 * fs_hz / fc_hz))
    n = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.sinc(2.0 * fc_hz / fs_hz * n) * np.hanning(2 * half + 1)
    return (kernel / kernel.sum()).astype(np.float64)


def lowpass(array: np.ndarray, fc_hz: float, fs_hz: float) -> np.ndarray:
    """Zero-phase low-pass of ``[n, channels]``, per channel, edge-clamped at both ends.

    Edge-clamped rather than reflected: reflection invents a symmetry the recording does not have.
    The chunks at both ends of every episode are dropped anyway by the inherited trim rule, so the
    clamp never reaches a scored chunk — but it is stated because "the padding did not matter" is
    a claim, and an unstated claim is the kind this repo keeps paying for.
    """
    kernel = lowpass_kernel(fc_hz, fs_hz)
    half = (kernel.shape[0] - 1) // 2
    arr = np.asarray(array, dtype=np.float64)
    padded = np.pad(arr, ((half, half), (0, 0)), mode="edge")
    out = np.empty_like(arr)
    for channel in range(arr.shape[1]):
        out[:, channel] = np.convolve(padded[:, channel], kernel, mode="valid")
    return out.astype(np.float32)


def filtered_raw(raw: dict[str, np.ndarray], fc_hz: float | None, fs_hz: float) -> tuple[dict, float]:
    """``raw`` with its ``action`` column low-passed, plus the RMS change that made (G0.3).

    ``fc_hz=None`` is the no-op control and returns the input untouched with an RMS change of
    exactly 0.0 — which is what makes the no-op cell bit-identical to T-44 rather than merely close.
    """
    if fc_hz is None:
        return raw, 0.0
    action = np.asarray(raw["action"], dtype=np.float32)
    smoothed = lowpass(action, fc_hz, fs_hz)
    rms = float(np.sqrt(np.mean((smoothed - action) ** 2)))
    return {**raw, "action": smoothed}, rms


def _verdict(
    grid_a: dict[float | None, dict],
    grid_b: dict[float | None, dict],
    *,
    any_l1_anywhere: bool,
    lowest_cutoff: float,
) -> dict[str, Any]:
    """``T45_RULE_V1``, PR-11 §5, evaluated in the pre-registered order **E, R, S, F, I**.

    Precedence is fixed in the pre-registration this time. PR-10's table left it open and T-44's
    driver had to settle it in a docstring, which is a decision made in the wrong document.

    ONE THING PR-11 DID NOT FIX EITHER, FOUND BY A TEST BEFORE ANY REAL CELL EXISTED: what "the
    best cutoff" means when cells TIE. A naive ``max`` returns whichever key it met first, which on
    a flat grid is the lowest cutoff — and the lowest cutoff winning is exactly the condition for
    verdict **E**. So a grid where filtering changed nothing at all would have reported "over-
    smoothing wins monotonically", the one reading it most certainly does not support.

    Ties therefore break toward **less filtering** — larger ``fc``, and the no-op ahead of every
    cutoff. That direction is the conservative one in both places it matters: it cannot manufacture
    a filtering finding out of a flat grid, and it cannot manufacture an **E** out of one either.
    Chosen while writing the driver and before any cell was computed, recorded here rather than
    added to PR-11, because amending a pre-registration after the fact is what the rules forbid.
    """
    cells = {k: v for k, v in grid_a.items() if k is not None}
    # sort key: score first, then LARGER fc wins the tie.
    fc_star = max(cells, key=lambda fc: (cells[fc]["skill_vs_repeat_pct"], fc))
    if grid_a[None]["skill_vs_repeat_pct"] >= cells[fc_star]["skill_vs_repeat_pct"]:
        fc_star = None

    if fc_star == lowest_cutoff:
        return {
            "verdict": "E",
            "fc_star": fc_star,
            "reading": (
                f"fc* = {fc_star} Hz is the lowest cutoff in the grid, so over-smoothing wins "
                "monotonically — the shrinkage signature at the grid edge. Nothing is concluded. "
                "PR-11 §5 permits ONE extension, to 0.5 Hz, re-read under the same rule."
            ),
        }
    if not any_l1_anywhere:
        return {
            "verdict": "R",
            "fc_star": fc_star,
            "reading": (
                "No cutoff clears L1 on half A at either anchor. The residual is not "
                "high-frequency content, so post-processing these labels will not reconcile the "
                "commanded and executed streams. The question moves to PR-04's collection spec — "
                "what KIND of data — and away from processing this data better."
            ),
        }

    b_star = grid_b[fc_star]["skill_vs_repeat_pct"]
    b_noop = grid_b[None]["skill_vs_repeat_pct"]
    l1_gain = b_star - b_noop
    zero_gain = grid_b[fc_star]["skill_vs_zero_pct"] - grid_b[None]["skill_vs_zero_pct"]
    clears = fc_star is not None and b_star > 0.0 and l1_gain >= MATERIAL_FLOOR_PP

    if clears and zero_gain < MATERIAL_FLOOR_PP:
        return {
            "verdict": "S",
            "fc_star": fc_star,
            "l1_gain_pp": l1_gain,
            "zero_gain_pp": zero_gain,
            "reading": (
                f"fc* = {fc_star} Hz clears L1 on held-out half B ({b_star:+.2f} %, gain "
                f"{l1_gain:.2f} pp) but skill_vs_zero_pct gains only {zero_gain:.2f} pp, under the "
                f"{MATERIAL_FLOOR_PP} pp floor. Shrinkage toward zero cannot improve the comparison "
                "against predicting no motion, so the L1 gain is magnitude shrinkage rather than "
                "noise removal. Licenses nothing, and stands as a warning that any smoothing of "
                "labels anywhere in this project can climb an MSE ratio for free."
            ),
        }
    if clears:
        return {
            "verdict": "F",
            "fc_star": fc_star,
            "l1_gain_pp": l1_gain,
            "zero_gain_pp": zero_gain,
            "reading": (
                f"Confirmed on held-out half B: L1 {b_star:+.2f} % at fc* = {fc_star} Hz, a gain of "
                f"{l1_gain:.2f} pp over B's own no-op cell, with skill_vs_zero_pct also up "
                f"{zero_gain:.2f} pp — so it is not shrinkage. The residual is the arm's low-pass. "
                "Licenses a defect report naming (d*, fc*) and licenses PROPOSING a relabel; it "
                "relabels nothing and retro-validates none of the fourteen negatives."
            ),
        }
    return {
        "verdict": "I",
        "fc_star": fc_star,
        "l1_gain_pp": l1_gain,
        "zero_gain_pp": zero_gain,
        "reading": (
            f"Indeterminate. Best on A is fc* = {fc_star} Hz; held-out half B gives {b_star:+.2f} % "
            f"there against {b_noop:+.2f} % unfiltered, a gain of {l1_gain:.2f} pp. F needs fc* not "
            f"the no-op, B above L1, and a gain of at least {MATERIAL_FLOOR_PP} pp."
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

    sweep = _load_sibling("sweep_label_anchoring")
    eval_t39 = sweep._load_eval_t39()
    convert = eval_t39._load_script("convert_lerobot_g1")

    from wam.data.episode import EpisodeReader, list_episodes

    holdout_order = [
        line.strip()
        for line in args.holdout.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    by_name = {p.name: p for p in list_episodes(args.dataset)}
    episode_dirs = [by_name[e] for e in holdout_order]
    halves = {
        "A": [d for i, d in enumerate(episode_dirs) if i % 2 == 0],
        "B": [d for i, d in enumerate(episode_dirs) if i % 2 == 1],
    }

    first = EpisodeReader(episode_dirs[0])
    chunk_steps = args.chunk_steps or eval_t39.episode_chunk_steps(first)
    mapping = eval_t39.gripper_mapping_from_manifest(first.manifest, convert)
    fs_hz = 1.0 / eval_t39.episode_dt_s(first)
    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"T-45 low-pass sweep | {len(episode_dirs)} holdout episodes | chunk_steps {chunk_steps} | "
        f"gripper {mapping.kind} | fs {fs_hz:.3f} Hz | cutoffs {CUTOFFS_HZ} + no-op | "
        f"anchors {ANCHORS}"
    )

    started = time.perf_counter()
    cache: dict[str, dict[str, Any]] = {}
    for episode_dir in episode_dirs:
        reader = EpisodeReader(episode_dir)
        episode_id = reader.manifest.episode_id
        cache[episode_id] = {
            "reader": reader,
            "pairs": sweep.trim_pairs(build_eval_pairs(episode_dir, args.camera, chunk_steps, num_frames=None)),
            "raw": eval_t39.read_raw_episode(args.raw_dataset, episode_id),
        }
    print(f"  loaded {len(cache)} episodes in {time.perf_counter() - started:.1f}s")

    def run_cell(dirs: list[Path], *, delay: int, fc: float | None) -> dict[str, Any]:
        predictions: list[Any] = []
        rms_changes: list[float] = []
        for episode_dir in dirs:
            entry = cache[EpisodeReader(episode_dir).manifest.episode_id]
            if not entry["pairs"]:
                continue
            raw, rms = filtered_raw(entry["raw"], fc, fs_hz)
            rms_changes.append(rms)
            chunks = sweep.delayed_oracle_action_chunks(
                eval_t39, entry["reader"], raw, chunk_steps, mapping, convert, delay=delay
            )
            policy = eval_t39.ChunkLookupPolicy(
                chunks, episode_id=entry["reader"].manifest.episode_id
            )
            predictions.extend(evaluate_policy(policy, entry["pairs"]))
        if not predictions:
            raise SystemExit(f"no chunks scored for delay={delay} fc={fc}")
        out = sweep._score(predictions, run_name=f"t45-d{delay:+d}-fc{fc}")
        out["filter_rms_change"] = float(np.mean(rms_changes)) if rms_changes else 0.0
        return out

    def run_oracle_state(dirs: list[Path]) -> dict[str, Any]:
        predictions: list[Any] = []
        for episode_dir in dirs:
            entry = cache[EpisodeReader(episode_dir).manifest.episode_id]
            if not entry["pairs"]:
                continue
            chunks = eval_t39.oracle_state_chunks(entry["reader"], chunk_steps, mapping)
            policy = eval_t39.ChunkLookupPolicy(
                chunks, episode_id=entry["reader"].manifest.episode_id
            )
            predictions.extend(evaluate_policy(policy, entry["pairs"]))
        return sweep._score(predictions, run_name="t45-oracle-state")

    results: dict[str, Any] = {
        "rule": "T45_RULE_V1",
        "preregistration": "docs/preregistration/PR-11-command-lowpass-sweep.md",
        "dataset": str(args.dataset),
        "raw_dataset": str(args.raw_dataset),
        "fs_hz": fs_hz,
        "chunk_steps": chunk_steps,
        "gripper_mapping": mapping.kind,
        "cutoffs_hz": list(CUTOFFS_HZ),
        "anchors": list(ANCHORS),
        "half_a": [d.name for d in halves["A"]],
        "half_b": [d.name for d in halves["B"]],
    }

    print("\n=== G0.2  oracle_state at d=0, unfiltered (floor 90 %)")
    g0b = run_oracle_state(episode_dirs)
    results["g0_oracle_state"] = g0b
    print(f"    L1 {g0b['skill_vs_repeat_pct']:+.2f} %   ({g0b['num_chunks']} chunks)")

    cutoffs: list[float | None] = [*CUTOFFS_HZ, None]
    grids: dict[int, dict[str, dict[float | None, dict]]] = {}
    for delay in ANCHORS:
        tag = "PRIMARY" if delay == ANCHORS[0] else "secondary"
        print(f"\n=== anchor d = {delay:+d}  ({tag})")
        print(f"{'fc':>6}  {'A L1':>10}  {'B L1':>10}  {'A vs0':>10}  {'B vs0':>10}  "
              f"{'A smooth':>9}  {'rmsΔ':>9}")
        grids[delay] = {"A": {}, "B": {}}
        for fc in cutoffs:
            for half in ("A", "B"):
                grids[delay][half][fc] = run_cell(halves[half], delay=delay, fc=fc)
            a, b = grids[delay]["A"][fc], grids[delay]["B"][fc]
            label = "no-op" if fc is None else f"{fc:g} Hz"
            print(
                f"{label:>6}  {a['skill_vs_repeat_pct']:>+10.2f}  {b['skill_vs_repeat_pct']:>+10.2f}  "
                f"{a['skill_vs_zero_pct']:>+10.2f}  {b['skill_vs_zero_pct']:>+10.2f}  "
                f"{a['smoothness_ratio']:>9.2f}  {a['filter_rms_change']:>9.2e}"
            )
    results["grids"] = {
        str(d): {h: {("noop" if fc is None else f"{fc:g}"): cell for fc, cell in g.items()}
                 for h, g in halves_.items()}
        for d, halves_ in grids.items()
    }

    # -- G0 ------------------------------------------------------------------------------------
    gates: list[str] = []
    for (delay, half), expected in T44_NO_OP.items():
        got = grids[delay][half][None]["skill_vs_repeat_pct"]
        drift = got - expected
        results.setdefault("g0_bridges", {})[f"d{delay:+d}-{half}"] = {
            "expected": expected, "got": got, "drift_pp": drift
        }
        if abs(drift) > BRIDGE_TOLERANCE_PP:
            gates.append(
                f"G0.1 FAILED: no-op cell d={delay:+d} half {half} gave {got:+.2f}, T-44 recorded "
                f"{expected:+.2f} (drift {drift:+.3f} pp > ±{BRIDGE_TOLERANCE_PP})."
            )
    if g0b["skill_vs_repeat_pct"] < ORACLE_STATE_FLOOR_PCT:
        gates.append(f"G0.2 FAILED: oracle_state {g0b['skill_vs_repeat_pct']:+.2f} % < 90 %.")

    # G0.3 — the filter must provably reach the array, and bite harder as the cutoff falls.
    rms_by_fc = [
        (fc, grids[ANCHORS[0]]["A"][fc]["filter_rms_change"]) for fc in CUTOFFS_HZ
    ]
    results["g0_filter_rms_by_fc"] = {f"{fc:g}": r for fc, r in rms_by_fc}
    if any(r <= 0.0 for _fc, r in rms_by_fc):
        gates.append(
            "G0.3 FAILED: at least one filtered cell changed the array by exactly 0. The filter is "
            "not reaching the data, and a flat grid would read as 'the jerk is irreducible'."
        )
    elif any(rms_by_fc[i][1] <= rms_by_fc[i + 1][1] for i in range(len(rms_by_fc) - 1)):
        gates.append(
            f"G0.3 FAILED: RMS change is not monotonically larger as fc falls: {rms_by_fc}."
        )

    if gates:
        results["verdict"] = {"verdict": "INVALID", "reading": " ".join(gates)}
    else:
        primary = ANCHORS[0]
        any_l1 = any(
            cell["skill_vs_repeat_pct"] > 0.0
            for delay in ANCHORS
            for fc, cell in grids[delay]["A"].items()
            if fc is not None
        )
        results["verdict"] = _verdict(
            grids[primary]["A"], grids[primary]["B"],
            any_l1_anywhere=any_l1, lowest_cutoff=min(CUTOFFS_HZ),
        )

    (args.out / "sweep.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    verdict = results["verdict"]
    print(f"\n=== T45_RULE_V1 -> {verdict['verdict']}")
    print(f"    {verdict['reading']}")
    print(f"\nwrote {args.out / 'sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
