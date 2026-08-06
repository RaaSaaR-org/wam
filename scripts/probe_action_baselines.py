"""CPU baselines that decide whether T-37's GPU probe is worth running.

T-37 (`docs/backbone-eval.md`) proposes feeding Cosmos-Predict2's **pretrained action port** the
true past actions and asking whether the residual stream becomes more linearly predictive of the
next action chunk than the same port fed zeros. That design has a hole, and this script is the
measurement that sizes it.

**The hole.** T-34 measured lag-1 autocorrelation of **0.927** on this corpus, and T-27 already
showed a causal repeat-last-action baseline beats the trained model. So a port fed the true past
actions can "carry action signal" for a reason that has nothing to do with physical
understanding: the features become a linear image of *what we just put in*, and the ridge reads
the next chunk off the autocorrelation. Any gate that only compares port-fed-actions against
port-fed-zeros would pass on that alone.

**What this measures, all on CPU, no backbone at all:**

- ``state_only`` — proprioception ``[q, dq, gripper]``, the floor T-15/T-24 used. Reproducing
  it here is the check that this script's windows and split are the archived ones.
- ``past_joint`` — the previous action chunk in canonical joint-delta space. This is the
  autocorrelation, made explicit.
- ``past_ee`` — the same window as **Bridge-shaped end-effector actions** (`wam.robot.
  kinematics`), i.e. literally the tensor Cosmos's port consumes.
- ``past_ee_plus_state`` — both, because the port sees an image too.

**How to read the result.** The best of these is the value obtainable from the probe's *own
inputs* with no video model in the loop. A GPU probe can only be said to have learned something
if it beats that, not merely the state-only floor.

**What it found (2026-08-05, `runs/backbone_eval/action_baselines*.json`, and see
`docs/backbone-eval.md` §4a):** ``state_only`` reproduces the archived 0.456 / 0.881 to four
digits at 12 episodes, which is the check that these windows are T-24's. But the floor is not a
constant — it climbs to 0.4879 at 24 episodes and 0.5129 at 48 — so quoting 0.456 as an absolute
bar compares across sample sizes. ``past_ee`` looks like it clears the floor at 12 episodes
(0.4576 vs 0.4563) and does not at 48 (0.3954 vs 0.5129): it *degrades* with corpus size, the
signature of a small-n result. The robust finding is ``past_joint_proj + state`` at
**0.540 / 0.539 / 0.541** joints across three seeds at 48 episodes, ~+0.03 over proprioception
alone. ``past_joint``'s raw −0.0950 at 12 episodes was 256 dims against 56 training rows, not an
information statement — matching the width reverses it.

Run it at more than one ``--episodes`` before quoting anything from it.

Windows, labels, episode split and ridge code are imported from ``hf_job_wan_probe`` unchanged —
one implementation, so a difference here cannot be a difference in the harness.

Usage:

    .venv/bin/python scripts/probe_action_baselines.py --data-dir data/raw/gr00t_apple
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_SRC = _HERE.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import convert_lerobot_g1 as conv
import hf_job_wan_probe as wan

#: Which hand's synergy becomes the port's gripper channel. T-34 measured the right hand frozen
#: at 0.0007 rad across all 402 episodes and 171 625 samples; channel 0 (left) is the live one.
ACTIVE_HAND = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/raw/gr00t_apple")
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--windows-per-episode", type=int, default=8)
    p.add_argument("--window-select", choices=("linspace", "motion"), default="linspace")
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--height", type=int, default=192)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--chunk-steps", type=int, default=16)
    p.add_argument("--instruction", default=None)
    p.add_argument("--alphas", default="1,10,100,1000,10000")
    p.add_argument("--scene", default=None, help="MJCF for FK (default configs/sim/g1_scene.xml)")
    p.add_argument("--out", default="runs/backbone_eval/action_baselines.json")
    return p.parse_args(argv)


def past_action_features(
    args: argparse.Namespace, windows: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Per window, the action chunk *preceding* its label, in joint space and in EE space.

    Returns ``(past_joint [N, K*16], past_ee [N, K*7], info)``. A window whose start is closer
    to the episode beginning than one chunk has no predecessor; those rows are **zero-filled
    and counted** rather than dropped, because dropping them would change the window set and
    break the like-for-like comparison against the archived ``state_only`` number.
    """
    from wam.robot.kinematics import G1Kinematics, ee_action_sequence

    kin = G1Kinematics(args.scene)
    source = Path(args.data_dir)
    k = args.chunk_steps

    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    joint_rows, ee_rows, padded = [], [], 0
    for w in windows:
        ep, start = int(w["episode"]), int(w["start"])
        if ep not in cache:
            data = conv.read_source_episode(
                source / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
            )
            cache[ep] = (
                conv.canonical_q(data["state"]),
                conv.gripper_state(data["state"])[:, ACTIVE_HAND],
            )
        q, grip = cache[ep]
        if start < k:
            padded += 1
            joint_rows.append(np.zeros(k * (wan.NUM_JOINTS + 1), dtype=np.float32))
            ee_rows.append(np.zeros(k * 7, dtype=np.float32))
            continue
        # The predecessor window [start-k, start]: k deltas ending exactly where the label begins.
        seg_q, seg_grip = q[start - k : start + 1], grip[start - k : start + 1]
        dq = np.diff(seg_q.astype(np.float64), axis=0)  # [k, 15]
        joint_rows.append(
            np.concatenate([dq.reshape(-1), seg_grip[1:].astype(np.float64)]).astype(np.float32)
        )
        ee_rows.append(ee_action_sequence(seg_q, seg_grip, kinematics=kin).reshape(-1))

    info = {
        "ee_body": kin.ee_body,
        "active_hand": ACTIVE_HAND,
        "windows_without_predecessor": padded,
        "past_joint_dim": k * (wan.NUM_JOINTS + 1),
        "past_ee_dim": k * 7,
    }
    return (
        np.stack(joint_rows).astype(np.float32),
        np.stack(ee_rows).astype(np.float32),
        info,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    alphas = tuple(float(a) for a in args.alphas.split(",") if a.strip())

    windows, instruction, winfo = wan.build_windows(args)
    if not windows:
        print("no windows built — check --data-dir", file=sys.stderr)
        return 1

    y = np.stack([w["label"] for w in windows])
    joint_dim = min(args.chunk_steps * wan.NUM_JOINTS, y.shape[1])
    y_joint, y_grip = y[:, :joint_dim], y[:, joint_dim:]
    split = wan.episode_split(np.asarray([w["episode"] for w in windows]))

    state_x = np.stack(
        [np.concatenate([w["state"].q, w["state"].dq, w["state"].gripper_state]) for w in windows]
    ).astype(np.float32)
    past_joint, past_ee, kinfo = past_action_features(args, windows)

    # Controls. `past_ee` beating the proprioception floor is the load-bearing number here, so
    # it gets two ways to be wrong and one confound removed:
    #   shuffled   — same features, rows permuted. Destroys the window->label pairing while
    #                keeping width and marginal distribution. Anything much above 0 means the
    #                ridge is exploiting the split, not the actions.
    #   joint_proj — past_joint randomly projected to past_ee's exact width. past_joint is 256
    #                dims against 56 training windows, so its score confounds "joint space is
    #                worse" with "it is wider". Matching the width separates them.
    #   ee_pos     — translation only, 48 dims. If this holds most of past_ee, the orientation
    #                channels (the euler ones, the shakiest part of the representation) are not
    #                what is carrying the result.
    rng = np.random.default_rng(0)
    features = {
        "state_only": state_x,
        "past_joint": past_joint,
        "past_ee": past_ee,
        "past_ee_plus_state": np.concatenate([past_ee, state_x], axis=1),
        "past_ee_shuffled": past_ee[rng.permutation(past_ee.shape[0])],
        "past_ee_pos": past_ee.reshape(past_ee.shape[0], args.chunk_steps, 7)[:, :, :3].reshape(
            past_ee.shape[0], -1
        ),
    }
    # Three seeds, because a single random projection scoring well can be luck. If the spread
    # across seeds is wide, the projected number means nothing and only the spread gets quoted.
    for seed in (0, 1, 2):
        proj = np.random.default_rng(100 + seed).standard_normal(
            (past_joint.shape[1], past_ee.shape[1])
        ).astype(np.float32) / np.sqrt(past_joint.shape[1])
        features[f"past_joint_proj_s{seed}"] = past_joint @ proj
        features[f"past_joint_proj_s{seed}_plus_state"] = np.concatenate(
            [past_joint @ proj, state_x], axis=1
        )
    rows = {
        name: {
            "joints": wan.probe_r2(x, y_joint, split, alphas),
            "gripper": wan.probe_r2(x, y_grip, split, alphas),
            "dim": int(x.shape[1]),
        }
        for name, x in features.items()
    }

    print(f"\n{len(windows)} windows, {winfo['episodes'][0]}..{winfo['episodes'][-1]}, "
          f"split {len(split['train'])}/{len(split['val'])}/{len(split['test'])}, "
          f"{kinfo['windows_without_predecessor']} zero-filled\n")
    print(f"{'features':>20}  {'dim':>5}  {'joints test R2':>15}  {'gripper test R2':>16}  alpha")
    for name, row in rows.items():
        print(
            f"{name:>20}  {row['dim']:5d}  {row['joints']['test_r2']:15.4f}  "
            f"{row['gripper']['test_r2']:16.4f}  {row['joints']['alpha']:g}"
        )

    # The bar a GPU probe has to clear is not the archived constant 0.456 — that is a
    # 12-episode number and the floor moves with episode count (0.4563 / 0.4879 / 0.5129 at
    # 12 / 24 / 48). It is the best score reachable from the probe's OWN inputs with no video
    # model in the loop, recomputed on whatever windows this run built.
    floor = rows["state_only"]["joints"]
    best_name, best_row = max(
        ((n, r) for n, r in rows.items() if n != "state_only" and "shuffled" not in n),
        key=lambda t: t[1]["joints"]["test_r2"],
    )
    best_grip = max(
        r["gripper"]["test_r2"] for n, r in rows.items() if "shuffled" not in n
    )
    print(
        f"\nstate-only floor      joints {floor['test_r2']:.4f}  gripper "
        f"{rows['state_only']['gripper']['test_r2']:.4f}"
        f"\nbest input-only       joints {best_row['joints']['test_r2']:.4f}  gripper "
        f"{best_grip:.4f}   ({best_name})"
        f"\n-> a backbone probe on these windows must clear "
        f"joints {max(floor['test_r2'], best_row['joints']['test_r2']):.4f} and gripper "
        f"{best_grip:.4f} to have added anything."
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "windows": winfo,
                "instruction": instruction,
                "kinematics": kinfo,
                "split": {k: [int(e) for e in split[k]] for k in ("train_eps", "val_eps", "test_eps")},
                "results": rows,
            },
            indent=2,
        )
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
