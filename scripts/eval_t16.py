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
hashes the manifests of the episodes it actually trained on into ``dataset_snapshot_ref``, and
this script reproduces that hash before it will score anything. There are two proofs, picked by
what the checkpoint records — never by a flag, because a flag is a way to make the refusal go
away:

  complement  ``train_episode_ids`` absent (every checkpoint written before I-8). Recompute the
              hash over ``dataset - holdout``. Proves both that the holdout was not trained on
              AND that everything else was. ``--train-episodes`` is optional here and, when
              given, is checked against that complement — so the flag is safe to pass on every
              call and a caller never has to predict which proof it will get.
  disjoint    ``train_episode_ids`` present. Requires ``--train-episodes``: the recorded ids
              must equal that file's as a MULTISET, must not intersect the holdout, must all
              exist, and must hash — in the recorded order — to ``dataset_snapshot_ref``.
              Proves the holdout was not trained on. It deliberately does NOT prove the training
              set was exhaustive, because an I-8 rung breaks that by construction while staying
              a perfectly valid measurement.

The witness file is not ceremony. Without it the ids and the hash are two fields of the same
self-description, so the comparison is the checkpoint against itself and cannot fail — measured:
a checkpoint trained on all eight ``mock-d1`` episodes that declared ``train_episode_ids=
("d1-0000",)`` printed "split proven (disjoint)" and returned the two episodes it had trained on.
The complement proof never had that hole, because ``dataset - holdout`` is built from the disk and
the holdout file rather than from anything the checkpoint says.

Exhaustiveness was never what made a number valid; it was what made two runs *comparable*, and
that is enforced where it belongs — ``run_bench.py --compare`` refuses runs whose holdouts
differ. Both proofs additionally refuse a holdout episode whose data-file checksums match a
trained one: the same recording under a second id is a leak the complement rule never looked
for and could not have caught.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Collection, Sequence
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


def content_digest(episode_dir: Path) -> str:
    """sha256 over the manifest's data-file checksums ONLY — deliberately id-free.

    Hashing ``manifest["checksums"]`` and not the whole manifest is the load-bearing part: the
    manifest embeds ``episode_id``, so a whole-file digest can never match across a rename, and
    the one thing this check exists to catch is the same recording filed under a second id.
    """
    from wam.data.episode import MANIFEST_FILENAME

    checksums = json.loads((episode_dir / MANIFEST_FILENAME).read_text())["checksums"]
    return hashlib.sha256(json.dumps(checksums, sort_keys=True).encode("utf-8")).hexdigest()


def _multiset_diff(a: Collection[str], b: Collection[str]) -> list[str]:
    """Ids in ``a`` that ``b`` does not cover, counting repeats."""
    return list((Counter(a) - Counter(b)).elements())


def verify_split(
    dataset: Path,
    holdout_ids: set[str],
    trained_ref: str,
    trained_ids: Sequence[str] | None = None,
    witness_ids: Collection[str] | None = None,
) -> list[Path]:
    """Return the holdout episode dirs, or refuse: the hash must prove they were excluded.

    ``trained_ids`` is the checkpoint's ``RunMetadata.train_episode_ids``. Absent (archived
    checkpoints) selects the complement proof, present selects the disjointness proof; see the
    module docstring for what each one does and does not establish. The distinction is driven
    by the checkpoint, never by a CLI flag — the ability to refuse must not be something a
    tired operator can switch off to make a warning stop.

    ``witness_ids`` is the id list read from ``--train-episodes``, and on the disjointness path
    it is **mandatory**. The reason is the whole reason this function exists. Under the
    complement proof the trained set is derived from two things the checkpoint does not control
    — the episodes on disk and the holdout file — so recomputing the snapshot hash over
    ``dataset - holdout`` and comparing it to ``trained_ref`` genuinely tests the checkpoint's
    claim. Under the disjointness proof, ``trained_ids`` and ``trained_ref`` are two fields of
    the *same* self-description, so hashing the episodes the checkpoint names and comparing the
    result to the hash the checkpoint reports compares the checkpoint against itself and cannot
    fail. Measured, not argued: a checkpoint trained on all eight ``mock-d1`` episodes that
    declares ``train_episode_ids=("d1-0000",)`` with the matching one-episode hash printed
    "split proven (disjoint)" and handed back the two episodes it had trained on. Requiring an
    external witness restores the missing anchor: what is trusted is the reviewed, committed
    split file (``configs/splits/i8_train_*.txt``), and the checkpoint has to agree with it.
    """
    from wam.data.episode import list_episodes

    episodes = list_episodes(dataset)
    present = {p.name for p in episodes}
    missing = holdout_ids - present
    if missing:
        raise SystemExit(
            f"holdout lists {len(missing)} episode(s) absent from {dataset}: {sorted(missing)[:5]}"
        )

    if trained_ids is None:
        proof = "complement"
        trained = [p for p in episodes if p.name not in holdout_ids]
        if not trained:
            raise SystemExit("holdout covers every episode — nothing was trained on")
        if witness_ids is not None:
            # A witness is not NEEDED here (the complement is already derived from two things the
            # checkpoint does not control), but refusing it was a mistake: it forced every caller
            # to know which proof a checkpoint would take before it could build a command line,
            # and every eval sbatch got that wrong. Checking it instead is strictly stronger than
            # ignoring it and makes --train-episodes safe to pass unconditionally.
            extra = sorted({p.name for p in trained} ^ set(witness_ids))
            if extra:
                raise SystemExit(
                    "REFUSING TO SCORE — this checkpoint is scored under the COMPLEMENT proof "
                    "(it records no explicit training set), but --train-episodes does not list "
                    f"the complement of the holdout. {len(extra)} id(s) differ: {extra[:5]}"
                    f"{'...' if len(extra) > 5 else ''}. Either the split file belongs to a "
                    "different run, or --holdout/--dataset is not the pair this run was trained "
                    "on."
                )
    else:
        proof = "disjoint"
        if witness_ids is None:
            raise SystemExit(
                "REFUSING TO SCORE — this checkpoint records an explicit training set "
                f"({len(trained_ids)} episode(s)), so the holdout is not its complement and the "
                "complement proof does not apply. The disjointness proof needs an external "
                "witness: pass --train-episodes with the split file the run was trained from "
                "(configs/splits/i8_train_*.txt). Without it, the ids and the hash both come "
                "from the checkpoint's own metadata and the check would compare the checkpoint "
                "against itself."
            )
        witness = list(witness_ids)
        # Multiset, not set. The hash below is taken over the recorded SEQUENCE, so a checkpoint
        # declaring ("d1-0000", "d1-0000") against a one-line witness would otherwise pass the
        # comparison while hashing a sequence the witness never named. Order is deliberately NOT
        # required: the trainer's iteration order is its own, and the witness authorises WHICH
        # episodes were trained on, not in what order they were visited.
        if sorted(witness) != sorted(trained_ids):
            only_ck = sorted(_multiset_diff(trained_ids, witness))
            only_wit = sorted(_multiset_diff(witness, trained_ids))
            raise SystemExit(
                "REFUSING TO SCORE — --train-episodes does not describe this checkpoint.\n"
                f"  in the checkpoint, not in the file: {only_ck[:5]} ({len(only_ck)})\n"
                f"  in the file, not in the checkpoint: {only_wit[:5]} ({len(only_wit)})\n"
                "The split file is the reviewed artifact; a checkpoint that disagrees with it "
                "is not the run that file describes. (Counts matter: the hash is over the "
                "recorded sequence, so a repeated id is a different training set.)"
            )
        leaked = sorted(set(trained_ids) & holdout_ids)
        if leaked:
            raise SystemExit(
                "REFUSING TO SCORE — the checkpoint records training on holdout episode(s): "
                f"{leaked[:5]}{'...' if len(leaked) > 5 else ''} ({len(leaked)} of "
                f"{len(holdout_ids)}). The number would be a training score wearing a "
                "holdout's name."
            )
        absent = sorted(set(trained_ids) - present)
        if absent:
            raise SystemExit(
                f"checkpoint trained on {len(absent)} episode(s) absent from {dataset}: "
                f"{absent[:5]}{'...' if len(absent) > 5 else ''} — --dataset is not the "
                "dataset this run was trained on"
            )
        by_name = {p.name: p for p in episodes}
        # RECORDED order, never sorted: dataset_snapshot_hash is a sequential digest, so
        # replaying the trainer's iteration order is what makes the next check reproducible.
        trained = [by_name[episode_id] for episode_id in trained_ids]
        if not trained:
            raise SystemExit("checkpoint records an empty training set — nothing was trained on")

    # Both paths land here: this is what binds a list of ids to the bytes on disk, because every
    # manifest embeds sha256 checksums of its own parquet/mp4, so an episode edited after
    # training no longer hashes to what was trained on. Note what it does NOT do on its own —
    # under the disjointness proof the ids being hashed came from the checkpoint, so this
    # comparison is only a real test because `witness` above forced those ids to match a file
    # the checkpoint did not write.
    recomputed = dataset_snapshot_hash(dataset, trained)
    if recomputed != trained_ref:
        raise SystemExit(
            f"REFUSING TO SCORE — split not provable ({proof}).\n"
            f"  checkpoint trained on: {trained_ref}\n"
            f"  recomputed over {len(trained)} episode(s): {recomputed}\n"
            f"  ({len(trained)} train / {len(holdout_ids)} holdout under {dataset})\n"
            + (
                "The holdout is not the complement of the training set, so these episodes may "
                "have been trained on. Point --dataset/--holdout at what the fine-tune actually "
                "used."
                if proof == "complement"
                else "The recorded training episodes no longer hash to what the checkpoint "
                "trained on — the dataset changed on disk, or the id list was edited. Either "
                "way the recorded ids no longer describe the bytes being scored."
            )
        )

    holdout_dirs = [p for p in episodes if p.name in holdout_ids]
    trained_content = {content_digest(p) for p in trained}
    duplicated = [p.name for p in holdout_dirs if content_digest(p) in trained_content]
    if duplicated:
        raise SystemExit(
            "REFUSING TO SCORE — holdout episode(s) are byte-identical in content to trained "
            f"ones: {duplicated[:5]}{'...' if len(duplicated) > 5 else ''}. A duplicate under a "
            "second id defeats the split without changing a single hash the id-based checks "
            "look at."
        )
    print(
        f"split proven ({proof}): {len(trained)} train / {len(holdout_ids)} holdout, hash matches"
    )
    return holdout_dirs


def build_policy(model_path: Path, device: str, camera: str | None, backbone_source: str | None):
    """``(policy, config, metadata)`` for the checkpoint, via the shared runtime loader."""
    from wam.runtime.policies import load_joint_policy
    from wam.training._utils import load_checkpoint_raw
    from wam.training.joint import JointTrainingConfig

    _, config_dict, metadata = load_checkpoint_raw(model_path)
    config = JointTrainingConfig.model_validate(config_dict)
    if config.backbone.requires_external_weights:
        recorded = getattr(config.backbone, "checkpoint_path", None)
        print(f"{config.backbone.kind}: checkpoint carries no base weights -> loading them")
        # Worth printing both: the recorded path is the cluster's, and a checkpoint scored on a
        # different box loads the base from somewhere else entirely. Which weights the numbers
        # below came from should not be something you have to infer.
        print(f"  base weights: {backbone_source or recorded} (recorded: {recorded})")
    policy = load_joint_policy(
        model_path, device=device, camera=camera, backbone_source=backbone_source
    )
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
    parser.add_argument(
        "--backbone-source",
        type=str,
        default=None,
        help="local dir holding the frozen base weights; required when scoring a checkpoint "
        "trained on another machine, whose recorded path does not exist here",
    )
    parser.add_argument(
        "--frame-history",
        action="store_true",
        help="show the policy the real num_frames window ending at each chunk — the same window "
        "EpisodeDataset fed during training — instead of one frame tiled num_frames times. OFF "
        "by default so this reproduces the runs recorded before 2026-07-30; the A/B between the "
        "two modes is T-29 (docs/improvements.md I-7). Use a separate --out per mode.",
    )
    parser.add_argument(
        "--train-episodes",
        type=Path,
        default=None,
        help="the split file the run was trained from (configs/splits/i8_train_*.txt). REQUIRED "
        "for a checkpoint that records an explicit training set, because the disjointness proof "
        "needs a witness the checkpoint did not write — see verify_split. OPTIONAL, and CHECKED "
        "against the complement, for a checkpoint trained on the holdout's complement: there the "
        "dataset and the holdout file are already the external anchor, so the witness is a "
        "redundant cross-check rather than the anchor. Safe to pass unconditionally, which is "
        "the point — a caller should not have to know which proof a checkpoint will take before "
        "it can build a command line.",
    )
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
    policy, config, metadata = build_policy(
        model_path, args.device, args.camera, args.backbone_source
    )
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
        witness_ids = None
        if args.train_episodes is not None:
            witness_ids = load_episode_ids(args.train_episodes)
        holdout = verify_split(
            args.dataset,
            holdout_ids,
            metadata.dataset_snapshot_ref,
            metadata.train_episode_ids,
            witness_ids,
        )

    # T-29: which frames the policy sees. OFF reproduces every result recorded before
    # 2026-07-30 (one frame, tiled by predict()); ON feeds the window training actually used.
    num_frames = config.backbone.num_frames if args.frame_history else None
    pairs = []
    for episode_dir in holdout:
        pairs.extend(
            build_eval_pairs(
                episode_dir, policy.camera, config.head.num_steps, num_frames=num_frames
            )
        )
    if not pairs:
        raise SystemExit(f"no eval chunks built from {len(holdout)} episode(s)")
    frames_note = (
        f"real {num_frames}-frame window (T-29)"
        if args.frame_history
        else f"1 frame tiled to {config.backbone.num_frames} (historical default)"
    )
    print(f"scoring {len(pairs)} chunks over {len(holdout)} episodes | frames: {frames_note}")

    predictions = evaluate_policy(policy, pairs)
    save_predictions_jsonl(predictions, out_dir / "predictions.jsonl")

    from wam.data.episode import EpisodeReader

    spec = EpisodeReader(holdout[0]).manifest.spec
    e1: E1Report = e1_metrics(predictions, spec)
    (out_dir / "e1.json").write_text(e1.to_json() + "\n")
    (out_dir / "e1.md").write_text(e1.render_markdown())

    # Same two calls scripts/run_bench.py makes, so re-scoring this run later from the archived
    # predictions.jsonl reproduces these files exactly rather than a second dialect of them.
    # The frame mode goes into run_name, not just the log: two predictions.jsonl from the same
    # checkpoint differ only in what the policy was shown, and a report that does not say which
    # is a report that will eventually be compared against the wrong one.
    run_name = metadata.run_id + ("+frame_history" if args.frame_history else "")
    bench = bench_metrics(predictions, run_name=run_name)
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
