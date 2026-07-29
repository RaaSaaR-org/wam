#!/usr/bin/env python3
"""Score a T-16 LoRA checkpoint on the held-out episodes it was never trained on.

``scripts/train_t16_lora.py`` produces weights and a training log; it deliberately does no
evaluation, because the fine-tune runs in preemptible 4-hour chunks and an eval bolted onto the
end of one of them would run on whichever chunk happened to finish. This is that eval, as its own
step: one GPU pass over the holdout, then everything else offline.

    scripts/eval_t16.py --run-dir runs/t16-lora-seed0 --device cuda

Writes into ``--out`` (default: the run dir):

  predictions.jsonl   the only artifact the scorers need — re-scorable forever, no GPU
  e1.json / e1.md     E1 action metrics (T-14 format, comparable to the baseline's)
  bench.json/.md      the WAM-Bench ladder (T-27), including the L1 bar this run has to clear

The whole point is a verdict you can trust, so the split is *proven*, not assumed. The trainer
hashes the manifests of the episodes it actually trained on into ``dataset_snapshot_ref``. This
script recomputes that hash over ``dataset - holdout`` and refuses to score unless it matches the
value embedded in the checkpoint. A mismatch means the holdout is not the complement of the
training set — the episodes may have been trained on — and any number produced would be
meaningless in the one way that matters.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from wam.evaluation import (
    E1Report,
    bench_metrics,
    build_eval_pairs,
    e1_metrics,
    evaluate_policy,
    load_episode_ids,
    save_predictions_jsonl,
)

CHECKPOINT_DIRNAME = "checkpoints"
MODEL_FILENAME = "model.safetensors"
LATEST_LINK = "latest"
DEFAULT_HOLDOUT = _REPO_ROOT / "configs" / "splits" / "t18_holdout_episodes.txt"


def dataset_snapshot_hash(root: Path, episodes: list[Path]) -> str:
    """Content hash over ``episodes``' manifests — must match ``train_t16_lora`` byte for byte.

    Every manifest embeds sha256 checksums of its own data files, so hashing manifests pins the
    full content without decoding a frame. Narrowed to the given episodes because a run that
    excludes a holdout has a different training set than one that does not.
    """
    from wam.data.episode import MANIFEST_FILENAME

    digest = hashlib.sha256()
    for episode_dir in episodes:
        digest.update(str(episode_dir.relative_to(root)).encode("utf-8"))
        digest.update((episode_dir / MANIFEST_FILENAME).read_bytes())
    return f"sha256:{digest.hexdigest()}"


def resolve_checkpoint(run_dir: Path, which: str) -> Path:
    """The ``model.safetensors`` for ``latest`` or for an explicit step dir/file.

    Whether the frozen base has to be supplied separately is NOT decided here — it follows from
    the backbone config embedded in the file itself (``requires_external_weights``), which travels
    with the checkpoint and cannot go missing the way a sidecar can.
    """
    if which == LATEST_LINK:
        link = run_dir / LATEST_LINK
        if link.exists():
            step_dir = link.resolve()
        else:
            steps = sorted((run_dir / CHECKPOINT_DIRNAME).glob("step-*"))
            if not steps:
                raise SystemExit(f"no checkpoints under {run_dir / CHECKPOINT_DIRNAME}")
            step_dir = steps[-1]
    else:
        step_dir = Path(which)
        if step_dir.is_file():
            step_dir = step_dir.parent

    model_path = step_dir / MODEL_FILENAME
    if not model_path.is_file():
        raise SystemExit(f"{model_path} missing — not a restorable checkpoint")
    return model_path


def verify_split(dataset: Path, holdout_ids: set[str], trained_ref: str) -> list[Path]:
    """Return the holdout episode dirs, or refuse: the hash must prove they were excluded."""
    from wam.data.episode import list_episodes

    episodes = list_episodes(dataset)
    present = {p.name for p in episodes}
    missing = holdout_ids - present
    if missing:
        raise SystemExit(
            f"holdout lists {len(missing)} episode(s) absent from {dataset}: {sorted(missing)[:5]}"
        )
    trained = [p for p in episodes if p.name not in holdout_ids]
    if not trained:
        raise SystemExit("holdout covers every episode — nothing was trained on")

    recomputed = dataset_snapshot_hash(dataset, trained)
    if recomputed != trained_ref:
        raise SystemExit(
            "REFUSING TO SCORE — split not provable.\n"
            f"  checkpoint trained on: {trained_ref}\n"
            f"  dataset minus holdout: {recomputed}\n"
            f"  ({len(trained)} train / {len(holdout_ids)} holdout under {dataset})\n"
            "The holdout is not the complement of the training set, so these episodes may have "
            "been trained on. Point --dataset/--holdout at what the fine-tune actually used."
        )
    print(f"split proven: {len(trained)} train / {len(holdout_ids)} holdout, hash matches")
    return [p for p in episodes if p.name in holdout_ids]


def build_policy(model_path: Path, device: str, camera: str | None):
    """``(policy, config, metadata)`` for the checkpoint, via the shared runtime loader."""
    from wam.runtime.policies import load_joint_policy
    from wam.training._utils import load_checkpoint_raw
    from wam.training.joint import JointTrainingConfig

    _, config_dict, metadata = load_checkpoint_raw(model_path)
    config = JointTrainingConfig.model_validate(config_dict)
    if config.backbone.requires_external_weights:
        print(f"{config.backbone.kind}: checkpoint carries no base weights -> loading them")
    policy = load_joint_policy(model_path, device=device, camera=camera)
    return policy, config, metadata


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="a train_t16_lora --out-dir")
    parser.add_argument(
        "--dataset", type=Path, default=_REPO_ROOT / "datasets" / "gr00t-apple-full"
    )
    parser.add_argument(
        "--holdout",
        type=Path,
        default=DEFAULT_HOLDOUT,
        help="episode id list, or a baseline predictions.jsonl (default: the T-18 holdout)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=LATEST_LINK,
        help="'latest' or a step dir (default: %(default)s)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--camera", type=str, default=None, help="override the trained camera key")
    parser.add_argument("--out", type=Path, default=None, help="default: --run-dir")
    parser.add_argument(
        "--skip-split-check",
        action="store_true",
        help="score even if the hash cannot prove the holdout was excluded (marks the artifacts)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out or args.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_checkpoint(args.run_dir, args.checkpoint)
    print(f"checkpoint {model_path}")
    policy, config, metadata = build_policy(model_path, args.device, args.camera)
    print(
        f"run {metadata.run_id} | config {metadata.config_hash[:12]} | "
        f"backbone {config.backbone.kind} | camera {policy.camera} | device {policy.device}"
    )

    holdout_ids = load_episode_ids(args.holdout)
    if args.skip_split_check:
        from wam.data.episode import list_episodes

        print("WARNING: --skip-split-check — the holdout is NOT proven to be unseen")
        holdout = [p for p in list_episodes(args.dataset) if p.name in holdout_ids]
    else:
        holdout = verify_split(args.dataset, holdout_ids, metadata.dataset_snapshot_ref)

    pairs = []
    for episode_dir in holdout:
        pairs.extend(build_eval_pairs(episode_dir, policy.camera, config.head.num_steps))
    if not pairs:
        raise SystemExit(f"no eval chunks built from {len(holdout)} episode(s)")
    print(f"scoring {len(pairs)} chunks over {len(holdout)} episodes ...")

    predictions = evaluate_policy(policy, pairs)
    save_predictions_jsonl(predictions, out_dir / "predictions.jsonl")

    from wam.data.episode import EpisodeReader

    spec = EpisodeReader(holdout[0]).manifest.spec
    e1: E1Report = e1_metrics(predictions, spec)
    (out_dir / "e1.json").write_text(e1.to_json() + "\n")
    (out_dir / "e1.md").write_text(e1.render_markdown())

    # Same two calls scripts/run_bench.py makes, so re-scoring this run later from the archived
    # predictions.jsonl reproduces these files exactly rather than a second dialect of them.
    bench = bench_metrics(predictions, run_name=metadata.run_id)
    (out_dir / "bench.json").write_text(bench.to_json() + "\n")
    (out_dir / "bench.md").write_text(bench.render_markdown())

    print(f"\nE1 action mse {e1.mse:.6g}")
    print(f"WAM-Bench {bench.level_name} — score {bench.score:.1f}/100")
    print(f"  vs zero-delta   {bench.skill_vs_zero_pct:+.1f}%")
    print(f"  vs repeat-last  {bench.skill_vs_repeat_pct:+.1f}%   <- the L1 bar (must be > 0)")
    for warning in bench.warnings:
        print(f"  warning: {warning}")
    print(f"\nwrote predictions.jsonl, e1.*, bench.* to {out_dir}")
    if args.skip_split_check:
        (out_dir / "UNPROVEN_SPLIT").write_text(
            "Scored with --skip-split-check: the holdout was not proven unseen.\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
