#!/usr/bin/env python3
"""Re-score an ALREADY-TRAINED archived run so the AC-07 ladder stops being mixed-mode.

Nothing here trains. It loads a finished checkpoint, replays the holdout that run was scored
on, and writes a second set of E1/bench artifacts next to — never over — the archived ones.

    scripts/rescore_archived.py --run-dir runs/d1-full-gen-seed0 \
        --out runs/d1-full-gen-seed0/eval-t29-history --frame-history

WHY THIS EXISTS
---------------
T-29 (Slurm job 184648, 2026-08-01) found that every WAM number recorded before that date was
measured with a train/inference mismatch. ``EpisodeDataset`` fed training the real
``backbone.num_frames`` window ending at the chunk; ``predict()`` was handed ONE frame and tiled
it ``num_frames`` times, so a video backbone read a clip that stands still. Re-scoring
``t16-lora-seed0`` with the real window moved ``skill_vs_repeat_pct`` from −32.45 % (tiled) to
−21.80 % (real window): +10.65 pp, real, about a third of the gap to the repeat-last-action
baseline, and NOT enough to clear the L1 gate. The level stayed L0.

Only ``t16-lora-seed0`` was re-measured. The other two runs of the AC-07 comparison table are
still tiled-only:

  runs/t18-real-ablation-seed0   the world-action arm,   skill_vs_repeat_pct −129.0 %
  runs/d1-full-gen-seed0         the action-only baseline, skill_vs_repeat_pct  −20.9 %

So the AC-07 ladder is currently MIXED-MODE: one arm windowed, two arms tiled. That is not a
comparison, and no amount of re-reading the table fixes it. This script is what makes it one
again — it is the only reason the file exists.

WHY IT IS NOT ``eval_t16.py``
-----------------------------
``eval_t16.py`` is the model for this script and everything reusable is imported from it
(``readout_tag``, ``guard_out_dir``, ``resolve_checkpoint``, ``dataset_snapshot_hash``). Three
things differ, and each one is a property of the ARCHIVE rather than a preference:

  checkpoint layout  These runs hold a FLAT ``checkpoint.safetensors`` at the run root, not
                     ``checkpoints/step-N/model.safetensors``. Auto-detected, both shapes work.
  two model kinds    ``t18-real-ablation-seed0`` embeds a ``JointTrainingConfig``,
                     ``d1-full-gen-seed0`` an action-only one. Sniffed from the embedded config,
                     never declared by the caller — a flag there is a way to mispair a loader
                     with a checkpoint and get a plausible number out of it.
  split proof        Both carry ``train_episode_ids=None``, so ``eval_t16``'s disjointness path
                     does not apply, and its complement path would prove something these runs
                     never claimed. The holdout is recovered from the run's OWN
                     ``predictions.jsonl`` — the episodes it was actually scored on — and
                     cross-checked for set-equality against
                     ``configs/splits/t18_holdout_episodes.txt``. A re-score on a different
                     holdout is not a re-score, so a mismatch refuses. What IS still proven is
                     that the dataset on disk is byte-identical to the one the run trained on:
                     ``dataset_snapshot_ref`` is recomputed before anything is scored.

THE CONTROL (``--verify-tiled``)
--------------------------------
Both modes are runnable and tiled stays the default, because a re-score that cannot reproduce
the archived number is not trustworthy — it would be a fresh measurement wearing a re-score's
name. ``--verify-tiled`` runs the tiled mode and refuses unless the recomputed ``mse`` matches
the archived ``bench.json``'s within ``--verify-tolerance`` (relative). It also reports the
largest per-element disagreement against the archived ``predictions.jsonl``, so "reproduced" is
a measured claim and not an inference from one aggregate.

Exact agreement is a property of the DEVICE, not of the ladder, and that has been measured here
rather than assumed. Both archived runs were evaluated on ``mps`` — their embedded ``device``
field says so — while this script defaults to ``cpu``; T-29 got its bit-for-bit reproduction
because both of its arms ran on one CUDA device. Re-scoring each of them tiled, 1040 chunks
against their own archived ``predictions.jsonl``:

  d1-full-gen-seed0   --device mps   max |Δ| 0          BIT-IDENTICAL, mse rel Δ 1.4e-15
  d1-full-gen-seed0   --device cpu   max |Δ| 1.79e-07   mse rel Δ 1.7e-06
  t18-real-ablation   --device mps   max |Δ| 8.94e-08   mse rel Δ 1.05e-08
  t18-real-ablation   --device cpu   max |Δ| 1.27e-07   mse rel Δ 1.42e-06

Both archives ARE reproducible, and the residuals are float32, not model. The action-only arm
comes back bit-for-bit on its own device: ``overfit_d1`` scored it through
``ActionOnlyModel.predict``, the same method ``CheckpointPolicy`` calls. The world-action arm
does not quite, even on ``mps``, and the reason is worth recording rather than tolerating: the
inline policy that produced that archive pre-pooled the token axis (``feats.mean(dim=1)``) before
``ActionHead.decode``, where ``JointWorldActionModel.predict`` lets ``decode`` do the pooling.
On CPU those two expressions are bit-identical (measured directly, 0.0 over 10 chunks); on MPS
they reduce differently and land ~2e-08 apart. This script keeps the SHARED path (see
:func:`build_archived_policy`) rather than re-creating the historical wrapper to chase that bit:
the number that has to be comparable is the one today's runtime produces.

Run the control on the archive's own device when you can — that removes the residual that has
nothing to do with the model. On any other device read the max-|Δ| line, which is printed next to
the gate precisely because an aggregate ``mse`` cannot distinguish "same predictions" from
"errors that cancelled".

Artifacts, names and the one-readout-per-``--out`` rule are ``eval_t16``'s, so
``scripts/run_bench.py`` and ``run_bench.py --compare`` accept the output unchanged. ``--out``
is REQUIRED and may not be the run dir: the archived ``predictions.jsonl`` and ``bench.json``
are the thing being compared against, and a re-score that overwrites its own baseline has
destroyed the evidence it was run to produce.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
# eval_t16 is a sibling script, not a package module. Running this file directly already puts
# scripts/ on sys.path, but importlib-loading it (the tests) does not — so it is made explicit
# rather than left to depend on how the module was reached.
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import eval_t16 as ev

from wam.evaluation import (
    E1Report,
    bench_metrics,
    build_eval_pairs,
    e1_metrics,
    evaluate_policy,
    load_episode_ids,
    load_predictions_jsonl,
    save_predictions_jsonl,
)

FLAT_CHECKPOINT = "checkpoint.safetensors"
"""What the archived runs actually hold: one file at the run root, written by
``JointTrainer.save_checkpoint`` / ``ActionOnlyTrainer.save_checkpoint`` directly rather than by
``train_t16_lora``'s step-directory rotation."""

AUTO = "auto"

DEFAULT_VERIFY_TOLERANCE = 1e-4
"""Relative tolerance of ``--verify-tiled`` on ``mse``, set from a measurement.

Re-scoring ``d1-full-gen-seed0`` tiled reproduces its archive bit-for-bit on ``mps``, the device
that archive was produced on (mse relative Δ 1.4e-15, i.e. float noise in the metric itself), and
lands at 1.7e-06 on ``cpu`` — the float32 backend, not the model. The default sits ~60x above the
measured cross-device residual and still ~4 orders of magnitude below the effect it has to stay
sensitive to: the T-29 tiled/windowed swap moved ``skill_vs_repeat_pct`` by 10.65 pp.

It is deliberately NOT a knob for making a failing control pass. A re-score that needs a wider
gate to agree with its archive has not reproduced it, and the honest response is to report that
number, not to widen the gate — which is why the max per-element |Δ| against the archived
predictions is printed next to the verdict and recorded in ``rescore.json``.
"""


def resolve_archived_checkpoint(run_dir: Path, which: str) -> Path:
    """The weights file for ``--checkpoint``, covering both layouts this repo has produced.

    ``auto`` prefers the flat ``checkpoint.safetensors`` at the run root (every pre-T-16 run,
    including both arms this script exists for) and falls back to ``eval_t16``'s ``latest`` /
    ``checkpoints/step-*`` resolution. Anything else is handed to ``eval_t16.resolve_checkpoint``
    unchanged, except an explicit path to a file, which is taken as given — a re-score should
    never have to guess at a filename it was told.
    """
    if which != AUTO:
        explicit = Path(which)
        if explicit.is_file():
            return explicit
        return ev.resolve_checkpoint(run_dir, which)
    flat = run_dir / FLAT_CHECKPOINT
    if flat.is_file():
        return flat
    if (run_dir / ev.LATEST_LINK).exists() or (run_dir / ev.CHECKPOINT_DIRNAME).is_dir():
        return ev.resolve_checkpoint(run_dir, ev.LATEST_LINK)
    raise SystemExit(
        f"{run_dir}: no {FLAT_CHECKPOINT} at the run root and no {ev.LATEST_LINK}/"
        f"{ev.CHECKPOINT_DIRNAME}/step-* to fall back to — this is not a restorable run"
    )


def sniff_checkpoint(model_path: Path) -> tuple[str, Any, Any]:
    """``(kind, config, metadata)`` — which MODEL a checkpoint holds, from its embedded config.

    Deliberately not a CLI flag. The two loaders are not interchangeable (``policies.py``'s
    module contract says so), and a mispaired one either fails at load or, worse, restores a
    subset of the tensors and returns a plausible-looking chunk. The config dict travels inside
    the file, so it cannot go missing the way a sidecar or an operator's memory can.

    ``action_encoder`` is the discriminator because it is the world branch's own field: a joint
    run has to encode action chunks into latents to train the flow, an action-only run never
    does. ``loss`` (l1|l2 regression) is the action-only side of the same fact. A config with
    neither is refused rather than guessed at — that is a checkpoint shape this repo has not
    produced, and the failure mode of guessing is a number nobody can trace.
    """
    from wam.training._utils import load_checkpoint_raw

    _, config_dict, metadata = load_checkpoint_raw(model_path)
    if "action_encoder" in config_dict:
        from wam.training.joint import JointTrainingConfig

        return "joint", JointTrainingConfig.model_validate(config_dict), metadata
    if "loss" in config_dict:
        from wam.training.action_only import ActionOnlyConfig

        return "action_only", ActionOnlyConfig.model_validate(config_dict), metadata
    raise SystemExit(
        f"{model_path}: embedded config has neither 'action_encoder' (joint) nor 'loss' "
        f"(action-only); keys are {sorted(config_dict)}. Refusing to guess which model this "
        "is — the wrong loader can restore a subset of the tensors and still return a chunk."
    )


def build_archived_policy(
    kind: str,
    model_path: Path,
    device: str,
    camera: str | None,
    backbone_source: str | None,
):
    """The runtime policy for an archived checkpoint — via the SHARED loaders, both kinds.

    ``load_joint_policy`` and ``CheckpointPolicy`` both take an arbitrary checkpoint path, so
    neither needs a copy here. That matters more than saving lines: both loaders route
    ``predict()`` through ``resolve_frame_context``, which is the one place that decides tiled
    vs. real window. A second decode path in this script would be free to disagree with the
    deployed one about the very thing the re-score is measuring.
    """
    if kind == "joint":
        from wam.runtime.policies import load_joint_policy

        return load_joint_policy(
            model_path, device=device, camera=camera, backbone_source=backbone_source
        )
    from wam.runtime.policies import CheckpointPolicy

    if backbone_source is not None:
        raise SystemExit(
            "--backbone-source does not apply to an action-only checkpoint: its TinyVideoBackbone "
            "weights are in the file, so there is no external base to point at"
        )
    return CheckpointPolicy(model_path, device, camera=camera)


def verify_dataset_snapshot(dataset: Path, recorded_ref: str | None) -> str:
    """Refuse unless ``dataset`` still hashes to what the run trained on.

    The manifests embed sha256 checksums of their own parquet/mp4, so this binds the re-score to
    the exact bytes rather than to a directory name. It is the one provenance claim these
    archived checkpoints DO make (``train_episode_ids`` is ``None``, so nothing else about their
    split is recoverable from the file), and re-scoring a run against a dataset that has moved
    underneath it produces a number that cannot be compared with the archived one — which is the
    entire purpose of the exercise.
    """
    from wam.data.episode import list_episodes

    if recorded_ref is None:
        raise SystemExit(
            "the checkpoint records no dataset_snapshot_ref, so the episodes on disk cannot be "
            "tied to what it trained on. Pass --skip-dataset-check to score anyway (the output "
            "is branded)."
        )
    episodes = list_episodes(dataset)
    if not episodes:
        raise SystemExit(f"{dataset}: no episodes found")
    recomputed = ev.dataset_snapshot_hash(dataset, episodes)
    if recomputed != recorded_ref:
        raise SystemExit(
            "REFUSING TO RE-SCORE — the dataset is not the one this run trained on.\n"
            f"  checkpoint records: {recorded_ref}\n"
            f"  {dataset} ({len(episodes)} episodes) hashes to: {recomputed}\n"
            "A re-score is only a re-score against identical bytes; against different ones it "
            "is a new measurement that will be read as a comparison. Point --dataset at the "
            "snapshot the run used, or pass --skip-dataset-check to brand the output as "
            "unverified."
        )
    return recomputed


def recover_holdout(run_dir: Path, reference: Path) -> tuple[list[str], int]:
    """``(holdout episode ids, archived chunk count)`` from the run's OWN ``predictions.jsonl``.

    The archived predictions are the only record of what these runs were actually scored on —
    their checkpoints carry ``train_episode_ids=None``, so the split cannot be read out of the
    metadata, and reconstructing it from a file the run never referenced would be a guess. Taking
    it from the predictions makes the re-score score the same chunks by construction.

    Which is also why it is cross-checked rather than trusted: a run scored on a different 40
    episodes would silently produce a number that lines up in a table with numbers it cannot be
    compared to. ``configs/splits/t18_holdout_episodes.txt`` is the reviewed, committed artifact
    the AC-07 table is defined against, so set-equality with it is the condition for the output
    belonging in that table at all.
    """
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.is_file():
        raise SystemExit(
            f"{predictions_path} not found — the holdout of an archived run is recovered from "
            "the predictions it was scored on, and there are none here to recover it from"
        )
    scored = load_episode_ids(predictions_path)
    num_chunks = sum(1 for line in predictions_path.read_text().splitlines() if line.strip())
    expected = load_episode_ids(reference)
    if scored != expected:
        only_run = sorted(scored - expected)
        only_ref = sorted(expected - scored)
        raise SystemExit(
            "REFUSING TO RE-SCORE — this run's holdout is not the reviewed one.\n"
            f"  scored by {run_dir.name} but not in {reference.name}: "
            f"{only_run[:5]}{'...' if len(only_run) > 5 else ''} ({len(only_run)})\n"
            f"  in {reference.name} but not scored by {run_dir.name}: "
            f"{only_ref[:5]}{'...' if len(only_ref) > 5 else ''} ({len(only_ref)})\n"
            "A re-score on a different holdout is not a re-score, and its bench numbers would "
            "sit in the AC-07 table next to numbers they cannot be compared with. Point "
            "--holdout at the split this run was actually scored on if that is deliberate."
        )
    return sorted(scored), num_chunks


def archived_bench(run_dir: Path) -> dict[str, Any] | None:
    """The run's archived ``bench.json`` payload, or ``None`` when it has none.

    Absent is a real state, not an error: not every archived run was re-scored on the T-27
    ladder. The comparison line degrades to a note; ``--verify-tiled`` does not, because a
    control with nothing to check against is not a control.
    """
    path = run_dir / "bench.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def max_prediction_delta(archived_path: Path, recomputed: list) -> tuple[float, int]:
    """``(max |Δ| over predicted targets+gripper, number of chunks compared)``.

    The claim ``--verify-tiled`` is really making is "these are the same predictions", and an
    aggregate ``mse`` can only ever be evidence for it. Two chunks that disagree in opposite
    directions leave the mean untouched. Aligned on ``(episode_id, t_ns)`` rather than on list
    order so a re-score that visits episodes differently still compares like with like.
    """
    import numpy as np

    archived = load_predictions_jsonl(archived_path)
    by_key = {(p.episode_id, p.t_ns): p for p in archived}
    worst = 0.0
    compared = 0
    for pred in recomputed:
        other = by_key.get((pred.episode_id, pred.t_ns))
        if other is None:
            raise SystemExit(
                f"archived predictions have no chunk for ({pred.episode_id}, t_ns={pred.t_ns}) — "
                "the re-score is not over the same chunks as the archive"
            )
        compared += 1
        worst = max(
            worst,
            float(np.abs(pred.predicted.targets - other.predicted.targets).max()),
            float(
                np.abs(
                    np.asarray(pred.predicted.gripper_target)
                    - np.asarray(other.predicted.gripper_target)
                ).max()
            ),
        )
    return worst, compared


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="an archived run directory (checkpoint + predictions.jsonl + bench.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="where to write this readout's artifacts. REQUIRED and may not be --run-dir: the "
        "archived artifacts are what the re-score is compared against",
    )
    parser.add_argument(
        "--dataset", type=Path, default=_REPO_ROOT / "datasets" / "gr00t-apple-full"
    )
    parser.add_argument(
        "--holdout",
        type=Path,
        default=ev.DEFAULT_HOLDOUT,
        help="the reviewed split the recovered holdout is cross-checked against "
        "(default: the T-18 holdout)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=AUTO,
        help=f"'{AUTO}' (flat {FLAT_CHECKPOINT}, else latest), 'latest', a step dir, or a file "
        "(default: %(default)s)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="default: %(default)s")
    parser.add_argument("--camera", type=str, default=None, help="override the trained camera key")
    parser.add_argument(
        "--backbone-source",
        type=str,
        default=None,
        help="local dir holding the frozen base weights, for a joint checkpoint whose recorded "
        "path does not exist here (never applies to an action-only checkpoint)",
    )
    parser.add_argument(
        "--frame-history",
        action="store_true",
        help="show the policy the real num_frames window ending at each chunk — the window "
        "EpisodeDataset fed during training — instead of one frame tiled num_frames times. OFF "
        "by default, same as eval_t16.py, so the default reproduces the archived number instead "
        "of redefining it. This flag is the whole point of the script; the default is its "
        "control.",
    )
    parser.add_argument(
        "--verify-tiled",
        action="store_true",
        help="refuse unless the recomputed mse matches the archived bench.json's within "
        "--verify-tolerance. Tiled mode only: comparing a windowed mse against a tiled archive "
        "is the measurement, not a control",
    )
    parser.add_argument(
        "--verify-tolerance",
        type=float,
        default=DEFAULT_VERIFY_TOLERANCE,
        help=f"relative tolerance of --verify-tiled on mse (default: {DEFAULT_VERIFY_TOLERANCE:g})",
    )
    parser.add_argument(
        "--skip-dataset-check",
        action="store_true",
        help="score even if the dataset no longer hashes to the run's dataset_snapshot_ref "
        "(brands the artifacts)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verify_tiled and args.frame_history:
        raise SystemExit(
            "--verify-tiled with --frame-history: the archived numbers are TILED, so checking a "
            "windowed re-score against them would fail on exactly the effect being measured. Run "
            "the two as separate passes into separate --out dirs — the tiled one is the control."
        )
    if args.verify_tolerance <= 0.0:
        raise SystemExit(f"--verify-tolerance must be > 0, got {args.verify_tolerance}")

    tag = ev.readout_tag(frame_history=args.frame_history, flow_steps=None)
    out_dir = args.out
    if out_dir.resolve() == args.run_dir.resolve():
        raise SystemExit(
            f"--out may not be --run-dir ({args.run_dir}). Every artifact this script writes has "
            "a fixed name, so it would overwrite the archived predictions.jsonl and bench.json — "
            "the baseline the re-score exists to be compared against. Use a subdirectory "
            "(runs/<id>/eval-t29-history is the T-29 convention)."
        )
    # Before anything expensive, and for the same reason eval_t16 does it: two readouts sharing
    # one --out silently leave an A/B of one arm against itself.
    ev.guard_out_dir(out_dir, tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_archived_checkpoint(args.run_dir, args.checkpoint)
    kind, config, metadata = sniff_checkpoint(model_path)
    print(f"checkpoint {model_path}")
    print(
        f"run {metadata.run_id} | config {metadata.config_hash[:12]} | model {kind} | "
        f"backbone {config.backbone.kind} | num_frames {config.backbone.num_frames} | "
        f"chunk {config.head.num_steps} steps"
    )

    if args.skip_dataset_check:
        print("WARNING: --skip-dataset-check — the dataset is NOT proven to be the trained one")
    else:
        verify_dataset_snapshot(args.dataset, metadata.dataset_snapshot_ref)
        print(f"dataset snapshot matches: {metadata.dataset_snapshot_ref}")

    holdout_ids, num_archived_chunks = recover_holdout(args.run_dir, args.holdout)
    print(
        f"holdout recovered from {args.run_dir.name}/predictions.jsonl: {len(holdout_ids)} "
        f"episodes / {num_archived_chunks} chunks, set-equal to {args.holdout.name}"
    )

    policy = build_archived_policy(kind, model_path, args.device, args.camera, args.backbone_source)
    num_frames = config.backbone.num_frames if args.frame_history else None
    pairs = []
    for episode_id in holdout_ids:
        pairs.extend(
            build_eval_pairs(
                args.dataset / episode_id,
                policy.camera,
                config.head.num_steps,
                num_frames=num_frames,
            )
        )
    if not pairs:
        raise SystemExit(f"no eval chunks built from {len(holdout_ids)} episode(s)")
    if len(pairs) != num_archived_chunks:
        raise SystemExit(
            f"REFUSING TO RE-SCORE — chunk count changed: the archive holds "
            f"{num_archived_chunks}, this pass built {len(pairs)}. Same episodes, different "
            "chunks is not a re-score; the dataset or the chunking rule has moved."
        )
    frames_note = (
        f"real {num_frames}-frame window (T-29, in-distribution)"
        if args.frame_history
        else f"1 frame tiled to {config.backbone.num_frames} (what the archive was measured with)"
    )
    print(
        f"re-scoring {len(pairs)} chunks over {len(holdout_ids)} episodes | frames: {frames_note}"
    )

    started = time.perf_counter()
    predictions = evaluate_policy(policy, pairs)
    elapsed_s = time.perf_counter() - started
    ms_per_chunk = 1000.0 * elapsed_s / len(pairs)
    save_predictions_jsonl(predictions, out_dir / "predictions.jsonl")

    from wam.data.episode import EpisodeReader

    spec = EpisodeReader(args.dataset / holdout_ids[0]).manifest.spec
    e1: E1Report = e1_metrics(predictions, spec)
    (out_dir / "e1.json").write_text(e1.to_json() + "\n")
    (out_dir / "e1.md").write_text(e1.render_markdown())

    # eval_t16's names and call, so run_bench.py re-scores this directory into the identical
    # files and `run_bench.py --compare` accepts it against any other arm on this holdout.
    run_name = metadata.run_id + tag
    bench = bench_metrics(predictions, run_name=run_name)
    (out_dir / "bench.json").write_text(bench.to_json() + "\n")
    (out_dir / "bench.md").write_text(bench.render_markdown())
    (out_dir / "timing.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "readout_tag": tag,
                "num_chunks": len(pairs),
                "seconds": elapsed_s,
                "ms_per_chunk": ms_per_chunk,
                "device": policy.device,
                "frame_history": bool(args.frame_history),
                "flow_steps": None,
                "flow_seed": None,
                "flow_mean_k": None,
                "flow_t0": None,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\nre-scored {len(pairs)} chunks in {elapsed_s:.1f}s ({ms_per_chunk:.1f} ms/chunk)")
    print(f"E1 action mse {e1.mse:.6g}")
    print(f"WAM-Bench {bench.level_name} — score {bench.score:.1f}/100")
    print(f"  vs zero-delta   {bench.skill_vs_zero_pct:+.1f}%")
    print(f"  vs repeat-last  {bench.skill_vs_repeat_pct:+.1f}%   <- the L1 bar (must be > 0)")
    for warning in bench.warnings:
        print(f"  warning: {warning}")

    record: dict[str, Any] = {
        "run_id": metadata.run_id,
        "model_kind": kind,
        "checkpoint": str(model_path),
        "readout_tag": tag,
        "frame_history": bool(args.frame_history),
        "num_frames": num_frames,
        "device": policy.device,
        "num_chunks": len(pairs),
        "num_episodes": len(holdout_ids),
        "holdout_reference": str(args.holdout),
        "dataset_snapshot_ref": metadata.dataset_snapshot_ref,
        "dataset_checked": not args.skip_dataset_check,
        "recomputed": {"mse": e1.mse, "skill_vs_repeat_pct": bench.skill_vs_repeat_pct},
    }

    archived = archived_bench(args.run_dir)
    if archived is None:
        print(f"\nno {args.run_dir}/bench.json — nothing archived to compare against")
    else:
        a_mse = float(archived["mse"])
        a_repeat = float(archived["skill_vs_repeat_pct"])
        record["archived"] = {
            "run_name": archived.get("run_name"),
            "mse": a_mse,
            "skill_vs_repeat_pct": a_repeat,
        }
        record["delta_pp"] = bench.skill_vs_repeat_pct - a_repeat
        record["mse_rel_delta"] = (e1.mse - a_mse) / a_mse if a_mse else None
        # The one line this whole script exists to print. Archived is ALWAYS a tiled number for
        # these runs; the mode of the recomputed one is named so the two are never read as the
        # same measurement.
        print(
            f"\nskill_vs_repeat_pct  archived {a_repeat:+.2f}% (tiled)"
            f"  ->  recomputed {bench.skill_vs_repeat_pct:+.2f}% "
            f"({'real window' if args.frame_history else 'tiled'})"
            f"   Δ {record['delta_pp']:+.2f} pp"
        )
        print(f"mse                  archived {a_mse:.6g}  ->  recomputed {e1.mse:.6g}")

    if args.verify_tiled:
        if archived is None:
            raise SystemExit(
                f"--verify-tiled needs {args.run_dir}/bench.json to check against, and there is "
                "none. A control with nothing to compare to is not a control."
            )
        worst, compared = max_prediction_delta(args.run_dir / "predictions.jsonl", predictions)
        rel = abs(record["mse_rel_delta"]) if record["mse_rel_delta"] is not None else float("inf")
        recorded_device = str(getattr(config, "device", "") or "")
        record["verify"] = {
            "tolerance": args.verify_tolerance,
            "mse_rel_delta": record["mse_rel_delta"],
            "max_abs_prediction_delta": worst,
            "chunks_compared": compared,
            "bit_identical": worst == 0.0,
            "device": policy.device,
            "archive_device": recorded_device or None,
            "passed": rel <= args.verify_tolerance,
        }
        print(
            f"\ncontrol: {compared} chunks vs the archived predictions.jsonl | "
            f"max |Δ| {worst:.3g} ({'BIT-IDENTICAL' if worst == 0.0 else 'not bit-identical'}) | "
            f"mse relative Δ {rel:.3g} (tolerance {args.verify_tolerance:g})"
        )
        # A non-zero residual on a device the archive was not produced on is expected and
        # explainable; the same residual on the SAME device is not, and would mean the code path
        # has moved. Naming both devices here is what tells those two cases apart at a glance.
        if worst != 0.0 and recorded_device and recorded_device != policy.device:
            print(
                f"  (the archive records device {recorded_device!r}, this pass ran on "
                f"{policy.device!r} — re-run with --device {recorded_device} for a residual-free "
                "control)"
            )
        if not record["verify"]["passed"]:
            (out_dir / "rescore.json").write_text(json.dumps(record, indent=2) + "\n")
            raise SystemExit(
                "REFUSING — the tiled re-score does NOT reproduce the archived number "
                f"(mse relative Δ {rel:.3g} > tolerance {args.verify_tolerance:g}).\n"
                f"  archived mse:   {float(archived['mse']):.9g}\n"
                f"  recomputed mse: {e1.mse:.9g}\n"
                f"  max |Δ| over {compared} chunks: {worst:.6g}\n"
                "Until the control reproduces, nothing measured by the same code path — the "
                "--frame-history number included — is trustworthy. Do not widen the tolerance to "
                "make this pass; find out why the two disagree. Scoring on a different device "
                "than the archive was produced on is the first thing to rule out (this archive "
                f"records device {recorded_device or 'unknown'!r}, this pass ran on "
                f"{policy.device!r})."
            )
        print("control PASSED — the tiled re-score reproduces the archive")

    (out_dir / "rescore.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.skip_dataset_check:
        (out_dir / "UNVERIFIED_DATASET").write_text(
            "Scored with --skip-dataset-check: the dataset was not proven to be the one this "
            "run trained on.\n"
        )
    print(f"\nwrote predictions.jsonl, e1.*, bench.*, timing.json, rescore.json to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
