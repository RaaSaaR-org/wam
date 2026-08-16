#!/usr/bin/env python3
"""Post-train the vendored NVIDIA policy on OUR committed split, and write the split witness.

    scripts/train_t39_baseline.py --vendor-root third_party/isaac-gr00t \
        --trainer-entrypoint scripts/gr00t_finetune.py \
        --model-id <id> --model-dir <staged weights> \
        --dataset data/raw/gr00t_apple --wam-dataset datasets/gr00t-apple-full \
        --train-episodes configs/splits/i8_train_362.txt \
        --exclude-episodes configs/splits/t18_holdout_episodes.txt \
        --embodiment-tag new_embodiment \
        --modality-config configs/groot/new_embodiment_config_defaults.py \
        --out-dir runs/t39-baseline-seed0 \
        -- --base_model_path <staged weights> --embodiment_tag new_embodiment ...

T-39 / PR-07. This driver does exactly two things of its own, both of them things the vendored
trainer cannot do for us, and nothing else:

  1. RESTRICT THE EPISODE SET to the committed split. The vendored trainer takes a LeRobot root
     and trains on all of it, so "train on these 362" is materialised as a subset VIEW of the
     source dataset (:func:`build_lerobot_subset`) rather than requested with a flag that may or
     may not exist. Symlinks, filtered metadata, no copied parquet.
  2. WRITE THE WITNESS. ``run_metadata.json`` in the ``RunMetadata`` shape
     ``eval_t16.verify_split`` consumes: the ordered ``train_episode_ids`` and the
     ``dataset_snapshot_ref`` they hash to. Without it the holdout cannot be PROVEN unseen and
     the eval refuses to score — correctly (PR-07 §3).

THE TRAINER ITSELF IS NOT OURS AND IS NOT MODIFIED. It is invoked as a SUBPROCESS, as shipped, in
its own venv. That is not squeamishness about imports: a positive control run through our
reimplementation of someone else's recipe is not a positive control, because a failure would again
be ambiguous between the recipe and our copy of it — the exact ambiguity T-39 exists to remove.
Running it out-of-process additionally keeps its torch/flash-attn pins out of any process that
touches WAM numbers.

``--trainer-entrypoint`` has NO DEFAULT, for the same reason ``MODEL_ID`` has none in
``70_train_t39_baseline.sbatch``: the vendored entrypoint's path and flag names were never
verified from a primary source (PR-07 §8 item 5), and a plausible guess baked in here would either
fail late or, worse, run something adjacent and record it as the recipe.

TWO DATASET PATHS, deliberately:

  --dataset      the LeRobot SOURCE. What the vendored trainer eats.
  --wam-dataset  the CONVERTED WAM episodes. What the snapshot hash is taken over, because that
                 is the dataset the evaluator recomputes it against and the one every other
                 number in this repo was measured on.

They are two views of one recording, and the witness has to be expressed in the second or
``verify_split`` cannot check it. (Correction to the first version of
``70_train_t39_baseline.sbatch``, which passed only the raw root: recorded here rather than
silently patched, and it changes no threshold — the gate lives in ``71_eval_t39_control.sbatch``
and nothing has been submitted.)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from wam.evaluation import load_episode_ids
from wam.interfaces.versioning import RunMetadata

DONE_FILENAME = "DONE"
WITNESS_FILENAME = "run_metadata.json"
SUBSET_DIRNAME = "lerobot_subset"

REQUIRED_META = ("info.json", "episodes.jsonl")
"""Source metadata this driver knows how to filter. Anything else is refused rather than copied
through: a subset whose metadata still describes all 402 episodes is a subset in name only, and
the trainer would happily read past the end of it."""


def _load_script(name: str) -> Any:
    """Import a sibling script — see ``eval_t39_baseline._load_script`` for why this exists."""
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_t39_{name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - unreachable for a real file
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_training_episodes(
    wam_dataset: Path, train_ids: set[str], holdout_ids: set[str]
) -> list[Path]:
    """The converted episode dirs to train on, in ``list_episodes`` order, or refuse.

    Order is not cosmetic: ``dataset_snapshot_hash`` is a SEQUENTIAL digest and the evaluator
    replays the recorded order to reproduce it, so the order this returns is the order the
    witness must record.

    The holdout intersection is checked HERE, at the producing end, and not only in the
    evaluator. ``train_t16_lora._training_episodes`` makes the same argument: a split file naming
    a held-out episode is the exact leak the whole proof exists to stop, and if it is caught only
    at scoring time the leak has already reached a checkpoint.
    """
    from wam.data.episode import list_episodes

    episodes = list_episodes(wam_dataset)
    if not episodes:
        raise SystemExit(f"no episodes under {wam_dataset} — is --wam-dataset the converted root?")
    present = {p.name for p in episodes}

    missing = sorted(train_ids - present)
    if missing:
        raise SystemExit(
            f"--train-episodes lists {len(missing)} episode(s) absent from {wam_dataset}: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    leaked = sorted(train_ids & holdout_ids)
    if leaked:
        raise SystemExit(
            f"REFUSING TO TRAIN — the split file names {len(leaked)} holdout episode(s): "
            f"{leaked[:5]}{'...' if len(leaked) > 5 else ''}. Training on them would make every "
            "number the eval prints a training score wearing a holdout's name."
        )
    return [p for p in episodes if p.name in train_ids]


def build_lerobot_subset(source: Path, out: Path, indices: list[int]) -> dict[str, Any]:
    """A LeRobot root containing only ``indices``: symlinked data, filtered metadata.

    Symlinks rather than copies — 362 episodes of video is not worth duplicating on a shared
    filesystem, and a symlinked parquet has the same bytes and therefore the same checksums.

    ``meta/episodes.jsonl`` is filtered to the selected ``episode_index`` values and
    ``meta/info.json``'s totals are recomputed from what survived. That is the part that makes
    this a real restriction: a trainer that trusts ``info.json`` over the directory listing (and
    they generally do) would otherwise iterate 402 episodes over a directory holding 362 and
    either crash or, worse, skip silently and train on a set nobody chose.

    Anything the filter does not recognise is refused rather than passed through — see
    ``REQUIRED_META``.
    """
    meta_src = source / "meta"
    for name in REQUIRED_META:
        if not (meta_src / name).is_file():
            raise SystemExit(
                f"{meta_src / name} missing — this driver only knows how to subset a LeRobot root "
                "with the standard meta/ layout, and guessing at a variant would produce a "
                "subset whose metadata does not describe it."
            )

    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)
    (out / "data" / "chunk-000").mkdir(parents=True)

    wanted = set(indices)
    linked_parquet = 0
    for index in sorted(wanted):
        name = f"episode_{index:06d}.parquet"
        src = source / "data" / "chunk-000" / name
        if not src.is_file():
            raise SystemExit(f"{src} missing — --dataset is not the source of this split")
        (out / "data" / "chunk-000" / name).symlink_to(src.resolve())
        linked_parquet += 1

    # Videos live under videos/chunk-000/<camera>/episode_XXXXXX.mp4. The camera directory names
    # are the source's, never assumed: a dataset with a second camera must keep both, and a
    # dataset with none must not fail here.
    linked_video = 0
    video_root = source / "videos" / "chunk-000"
    if video_root.is_dir():
        for camera_dir in sorted(p for p in video_root.iterdir() if p.is_dir()):
            dest = out / "videos" / "chunk-000" / camera_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for index in sorted(wanted):
                name = f"episode_{index:06d}.mp4"
                src = camera_dir / name
                if src.is_file():
                    (dest / name).symlink_to(src.resolve())
                    linked_video += 1

    kept: list[dict[str, Any]] = []
    total_frames = 0
    for line in (meta_src / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        if int(record.get("episode_index", -1)) in wanted:
            kept.append(record)
            total_frames += int(record.get("length", 0))
    if len(kept) != len(wanted):
        found = {int(r["episode_index"]) for r in kept}
        raise SystemExit(
            f"meta/episodes.jsonl describes {len(kept)} of the {len(wanted)} selected episodes; "
            f"missing {sorted(wanted - found)[:5]}. A trainer reading this metadata would train "
            "on a different set than the split file names."
        )
    (out / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in kept), encoding="utf-8"
    )

    info = json.loads((meta_src / "info.json").read_text(encoding="utf-8"))
    info["total_episodes"] = len(kept)
    if total_frames:
        info["total_frames"] = total_frames
    if "splits" in info:
        # The source's split ranges name episode indices that are no longer all present. Leaving
        # them would hand the trainer a range it cannot resolve; a single train span over what is
        # actually here is the only honest rewrite.
        info["splits"] = {"train": f"0:{len(kept)}"}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    for extra in sorted(meta_src.glob("*.json*")):
        if extra.name in REQUIRED_META:
            continue
        shutil.copy2(extra, out / "meta" / extra.name)

    return {
        "episodes": len(kept),
        "parquet_links": linked_parquet,
        "video_links": linked_video,
        "total_frames": total_frames,
    }


def resolve_embodiment_tag(vendor_root: Path, tag: str) -> str:
    """``tag`` -> the ``EmbodimentTag`` NAME upstream's ``stats.py`` will accept.

    THE TWO ENTRYPOINTS DISAGREE ABOUT SPELLING, and neither is wrong. ``launch_finetune.py``
    declares ``embodiment_tag: str`` and hands it to ``EmbodimentTag.resolve()``, which matches a
    name OR a value, case-insensitively — so ``new_embodiment`` works there. ``gr00t/data/stats.py``
    declares ``embodiment_tag: EmbodimentTag``, so tyro builds a choice list of enum NAMES and
    rejects ``new_embodiment`` outright in favour of ``NEW_EMBODIMENT``.

    Uppercasing the string is NOT the fix, and looks like one: most tags have a name that is not
    the shout-case of their value (``XDOF`` is ``xdof_relative_eef_relative_joint``), so
    ``.upper()`` would invent ``XDOF_RELATIVE_EEF_RELATIVE_JOINT`` and fail on everything except
    the one tag we happen to use. Ask the vendored enum instead, in the tree that will actually
    run, so the answer is that tree's rather than this file's idea of it.
    """
    probe = (
        "import sys;"
        "from gr00t.data.embodiment_tags import EmbodimentTag;"
        "print(EmbodimentTag.resolve(sys.argv[1]).name)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, tag],
        cwd=vendor_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"{tag!r} is not an embodiment tag the vendored tree knows.\n"
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def generate_subset_stats(
    subset: Path,
    *,
    vendor_root: Path,
    embodiment_tag: str,
    modality_config: Path,
) -> dict[str, Any]:
    """Write ``meta/stats.json`` and ``meta/relative_stats.json`` INTO the subset.

    ``LeRobotEpisodeLoader.__init__`` hard-asserts ``meta/stats.json`` (upstream
    ``lerobot_episode_loader.py:180``), and ``relative_stats.json`` is not optional in practice
    either: ``left_arm``/``right_arm`` are ``RELATIVE`` in this embodiment's action config, so
    without it the relative branch normalises against nothing. Neither file ships with
    ``nvidia/GR00T-N1.7-AppleToPlate`` — upstream expects you to run ``gr00t/data/stats.py``.

    OVER THE SUBSET, AND DELIBERATELY NOT OVER THE SOURCE. These statistics are min/max, mean/std
    and q01/q99 percentiles, and ``launch_finetune.py`` normalises every input with them. Computed
    over all 402 episodes they would carry the 40 holdout episodes' extremes into the training
    signal — a quiet leak that the witness cannot catch, because the witness proves which episodes
    were *trained on*, not which ones a percentile saw. Computed over the 362 that survived
    :func:`build_lerobot_subset`, they describe exactly the data the trainer is allowed to see.
    Cost: a few CPU-minutes on 71 MB of parquet, inside a job that is otherwise GPU-bound.

    Run as a SUBPROCESS in the vendored tree, for the same reason the trainer is (PR-07 §3): these
    numbers are upstream's, produced by upstream's code, and importing ``gr00t`` into this process
    would put its torch pin in the same interpreter as ``wam.*``.
    """
    # EVERY PATH IS RESOLVED BEFORE THE COMMAND IS BUILT, because the subprocess runs with
    # cwd=vendor_root and a relative path means something different there. A relative
    # --vendor-root doubles into vendor/vendor/gr00t/data/stats.py and the run dies with a bare
    # "exited 2"; a relative --dataset-path or --modality-config silently resolves against the
    # vendored tree instead of the caller's directory, which is worse — stats.py would refuse a
    # missing file, but a same-named file inside the vendor tree would be read without complaint.
    # Found by running the dry run from a relative path on the workstation, 2026-08-16. It never
    # surfaced on the cluster because the sbatch builds ${PROJ}-rooted absolute paths throughout.
    vendor_root = vendor_root.resolve()
    subset = subset.resolve()
    modality_config = modality_config.resolve()

    stats_script = vendor_root / "gr00t" / "data" / "stats.py"
    if not stats_script.is_file():
        raise SystemExit(
            f"{stats_script} missing — the vendored tree does not carry upstream's statistics "
            "generator, so the trainer would assert on a dataset with no meta/stats.json."
        )
    if not modality_config.is_file():
        raise SystemExit(
            f"{modality_config} missing. A custom embodiment tag is absent from upstream's "
            "MODALITY_CONFIGS registry until a --modality-config-path registers it, and "
            "gr00t/data/stats.py refuses rather than writing a partial set."
        )
    tag_name = resolve_embodiment_tag(vendor_root, embodiment_tag)
    command = [
        sys.executable,
        str(stats_script),
        "--dataset-path",
        str(subset),
        "--embodiment-tag",
        tag_name,
        "--modality-config-path",
        str(modality_config),
    ]
    print(f"\n=== normalization stats over the subset:\n    {' '.join(command)}\n")
    completed = subprocess.run(command, cwd=vendor_root, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            f"gr00t/data/stats.py exited {completed.returncode} — refusing to train against a "
            "dataset whose normalization statistics were not written."
        )
    written = {}
    for name in ("stats.json", "relative_stats.json"):
        path = subset / "meta" / name
        if not path.is_file():
            raise SystemExit(
                f"stats.py exited 0 but {path} is absent. Upstream's generate_rel_stats returns "
                "early when the registered modality config carries no action_configs; check "
                f"{modality_config}."
            )
        written[name] = path.stat().st_size
    return {
        "embodiment_tag": embodiment_tag,
        "embodiment_tag_name": tag_name,
        "modality_config": str(modality_config),
        **written,
    }


def write_witness(
    out_dir: Path,
    *,
    run_id: str,
    model_id: str,
    trained_ids: list[str],
    snapshot_ref: str,
    config: dict[str, Any],
) -> RunMetadata:
    """``run_metadata.json`` — the artifact that makes the checkpoint scorable at all.

    Same class and same fields ``train_t16_lora.py`` embeds in a WAM checkpoint header. The
    vendored trainer writes its own format and knows nothing about ``RunMetadata``, so the witness
    lives beside the checkpoint instead of inside it; ``eval_t39_baseline.load_witness`` reads it
    and hands it to the same ``verify_split``.

    ``model_id`` goes into the recorded config verbatim, so the checkpoint identity the operator
    stated on the command line travels with the numbers rather than living in a log.
    """
    metadata = RunMetadata.create(
        run_id=run_id,
        config=config,
        checkpoint_ref=model_id,
        dataset_snapshot_ref=snapshot_ref,
        train_episode_ids=trained_ids,
    )
    (out_dir / WITNESS_FILENAME).write_text(json.dumps(metadata.to_dict(), indent=2) + "\n")
    return metadata


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument(
        "--trainer-entrypoint",
        required=True,
        help="path to the vendored trainer, relative to --vendor-root. No default: PR-07 §8.",
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="the LeRobot SOURCE root")
    parser.add_argument(
        "--wam-dataset",
        type=Path,
        required=True,
        help="the CONVERTED WAM dataset — what dataset_snapshot_ref is taken over",
    )
    parser.add_argument("--train-episodes", type=Path, required=True)
    parser.add_argument("--exclude-episodes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default="latest")
    parser.add_argument(
        "--embodiment-tag",
        required=True,
        help="the tag the SUBSET's statistics are generated under. Must be the same tag passed "
        "to the trainer after --, or the trainer normalises with statistics computed for a "
        "different modality layout. No default: PR-07 §8.",
    )
    parser.add_argument(
        "--modality-config",
        type=Path,
        required=True,
        help="the .py that registers --embodiment-tag in upstream's MODALITY_CONFIGS. Same file "
        "the trainer is given as --modality_config_path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build the subset and the witness, then stop before the trainer. Everything this "
        "driver is responsible for runs; nothing that needs a GPU does.",
    )
    parser.add_argument(
        "trainer_args",
        nargs=argparse.REMAINDER,
        help="everything after -- is forwarded to the vendored trainer verbatim",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    eval_t16 = _load_script("eval_t16")
    convert = _load_script("convert_lerobot_g1")

    train_ids = load_episode_ids(args.train_episodes)
    holdout_ids = load_episode_ids(args.exclude_episodes)
    trained = resolve_training_episodes(args.wam_dataset, train_ids, holdout_ids)

    # The evaluator's own function, not a second copy of the same arithmetic: this hash is the
    # one thing the two scripts MUST agree on byte for byte, and the way to guarantee that is for
    # there to be one implementation of it.
    snapshot_ref = eval_t16.dataset_snapshot_hash(args.wam_dataset, trained)
    trained_ids = [p.name for p in trained]
    indices = [_episode_index(name) for name in trained_ids]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subset = args.out_dir / SUBSET_DIRNAME
    stats = build_lerobot_subset(args.dataset, subset, indices)
    # Immediately after the subset exists and before anything reads it. build_lerobot_subset
    # rmtree's its output, so these files cannot be produced by an earlier pass and survive.
    stats["normalization"] = generate_subset_stats(
        subset,
        vendor_root=args.vendor_root,
        embodiment_tag=args.embodiment_tag,
        modality_config=args.modality_config,
    )

    run_id = args.out_dir.name
    config = {
        "task": "T-39",
        "preregistration": "docs/preregistration/PR-07-positive-control.md",
        "model_id": args.model_id,
        "model_dir": str(args.model_dir),
        "trainer_entrypoint": args.trainer_entrypoint,
        "vendor_root": str(args.vendor_root),
        "source_dataset": str(args.dataset),
        "wam_dataset": str(args.wam_dataset),
        "train_episodes_file": str(args.train_episodes),
        "exclude_episodes_file": str(args.exclude_episodes),
        "num_train_episodes": len(trained_ids),
        "num_holdout_episodes": len(holdout_ids),
        "seed": args.seed,
        "source_state_dim": convert.SOURCE_STATE_DIM,
        "embodiment_tag": args.embodiment_tag,
        "modality_config": str(args.modality_config),
        "subset": stats,
    }
    metadata = write_witness(
        args.out_dir,
        run_id=run_id,
        model_id=args.model_id,
        trained_ids=trained_ids,
        snapshot_ref=snapshot_ref,
        config=config,
    )

    print(f"=== T-39 positive control ({run_id})")
    print(f"    model      {args.model_id}  <- {args.model_dir}")
    print(f"    train      {len(trained_ids)} episodes from {args.train_episodes.name}")
    print(f"    holdout    {len(holdout_ids)} episodes, excluded and NOT in the subset")
    print(f"    subset     {subset} ({stats['parquet_links']} parquet, {stats['video_links']} mp4)")
    print(f"    stats      {args.embodiment_tag} over the subset, not the source")
    print(f"    snapshot   {snapshot_ref}")
    print(f"    witness    {args.out_dir / WITNESS_FILENAME} (config {metadata.config_hash[:12]})")

    if args.dry_run:
        print("\n--dry-run: subset and witness written, trainer NOT invoked")
        return 0

    entrypoint = args.vendor_root / args.trainer_entrypoint
    if not entrypoint.is_file():
        raise SystemExit(
            f"{entrypoint} missing. The trainer is vendored UNMODIFIED and is the whole point of "
            "this experiment (PR-07 §3); pointing --trainer-entrypoint at a reimplementation "
            "would make a failure unattributable again."
        )
    forwarded = [a for a in (args.trainer_args or []) if a != "--"]
    command = [
        sys.executable,
        str(entrypoint),
        "--dataset-path",
        str(subset),
        "--output-dir",
        str(args.out_dir / "checkpoints"),
        *forwarded,
    ]
    print(f"\n=== invoking the vendored trainer as shipped:\n    {' '.join(command)}\n")
    env = dict(os.environ)
    env["WAM_T39_MODEL_DIR"] = str(args.model_dir)
    env["WAM_T39_SEED"] = str(args.seed)
    completed = subprocess.run(command, cwd=args.vendor_root, env=env, check=False)
    if completed.returncode != 0:
        print(f"vendored trainer exited {completed.returncode} — no DONE written")
        return completed.returncode

    (args.out_dir / DONE_FILENAME).write_text(
        f"T-39 baseline complete at {datetime.now(timezone.utc).isoformat()}\n"
        f"model_id: {args.model_id}\n"
        f"train episodes: {len(trained_ids)}\n"
        f"dataset_snapshot_ref: {snapshot_ref}\n"
    )
    print(f"\nwrote {args.out_dir / DONE_FILENAME}")
    print("next: sbatch cluster/discoverer/71_eval_t39_control.sbatch")
    return 0


def _episode_index(episode_id: str) -> int:
    """``gr00t-apple-000020`` -> ``20``. Same rule as the evaluator's ``raw_episode_index``."""
    tail = episode_id.rsplit("-", 1)[-1]
    if not tail.isdigit():
        raise SystemExit(
            f"cannot map {episode_id!r} to a LeRobot episode index — the converted ids end in the "
            "zero-padded source index (scripts/convert_lerobot_g1.py)"
        )
    return int(tail)


if __name__ == "__main__":
    raise SystemExit(main())
