#!/usr/bin/env python3
"""T-16 LoRA fine-tune entry point — one link in a REQUEUE CHAIN (M3, PRD §10.3).

Driven by ``cluster/discoverer/50_train_t16.sbatch``. The cluster caps every job at 4 hours
and runs ``PreemptMode=REQUEUE``, so the deliverable run is not one process — it is a chain
of them sharing one ``--out-dir``. Checkpoint-and-resume is therefore the architecture here,
not an optimization, and three properties carry the whole chain:

1. **Stop at a step boundary.** SIGUSR1 (Slurm, 5 min before the wall clock), SIGTERM
   (preemption) and SIGINT (a human) only set a flag; the loop finishes the step it is in,
   checkpoints and exits 0. Raising out of a signal handler lands at an arbitrary bytecode —
   mid-backward, mid-safetensors-write — and turns the checkpoint that preemption is supposed
   to produce into the corrupt one.
2. **Resume the SAMPLER, not just the weights.** ``EpochFeeder`` checkpoints
   ``(epoch, batch_in_epoch)``. A resume that restarts the shuffle at step 0 replays the same
   opening batches in every 4-hour chunk and the run never sees the tail of the dataset — a
   failure that is invisible in the loss curve.
3. **One config hash for the chain (AC-04).** ``--resume`` uses the CHECKPOINT's config
   verbatim (only ``--device`` may be overridden). An edited YAML is reported, never applied:
   otherwise chunk 7 of a chain would silently be a different experiment in the same directory.
4. **One training set for the chain.** Every checkpoint records both ``dataset_snapshot_ref``
   and the ordered ``train_episode_ids`` it was fed, and a resume whose episode set hashes to
   something else is fatal. The I-8 rungs (``--train-episodes``) share one dataset root and
   differ only in that file, so the wrong ``--out-dir`` would otherwise stamp a small rung's
   provenance onto a large rung's weights — wrong, self-consistent, and undetectable later.

Contract with the batch script (its header documents the same three lines):
  - SIGUSR1 -> checkpoint at the next step boundary, then exit 0;
  - write ``${OUT}/DONE`` once the configured step budget is finished — that file is the ONLY
    signal separating "finished" from "ran out of wall clock", since both exit 0;
  - ``--resume latest`` picks the newest complete checkpoint in ``--out-dir`` (and starts
    fresh when there is none — the first job of a chain passes the same flag as the tenth).

Every non-crash path returns 0. A genuine failure (config mismatch, OOM, missing data) exits
non-zero on purpose: the batch script refuses to requeue an unsignalled failure, so a broken
config costs one job instead of an infinite crash loop against the GPU allocation.

Usage (CPU dry run of the flag surface, no Wan weights):
    .venv/bin/python scripts/train_t16_lora.py \
        --training-config configs/training/joint_gr00t.yaml \
        --dataset datasets/mock-d1 --out-dir runs/t16-dry --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from wam.backbones.registry import build_backbone, build_backbone_config
from wam.interfaces.versioning import (
    JsonlRunLogger,
    RunMetadata,
    config_hash,
    load_config,
    read_git_commit,
)
from wam.runtime.offload import (
    OFFLOAD_TEXT_HELP,
    advise_alloc_conf,
    distinct_instructions,
    offload_text_encoder,
)
from wam.training import EpisodeDataset, JointTrainer, JointTrainingConfig, collate_episode_batch
from wam.training._utils import (
    CHECKPOINT_CONFIG_KEY,
    CHECKPOINT_METADATA_KEY,
    prepare_tensor_batch,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

DONE_FILENAME = "DONE"
CHECKPOINT_DIRNAME = "checkpoints"
LATEST_LINK = "latest"
MODEL_FILENAME = "model.safetensors"
TRAINER_STATE_FILENAME = "trainer_state.pt"
STEP_DIR_PREFIX = "step-"
STEP_DIR_DIGITS = 6

#: Prime multiplier mixing the run seed into the per-epoch shuffle seed. Prime so that two
#: (seed, epoch) pairs cannot collide for any realistic epoch count — ``seed + epoch`` would
#: make run 0/epoch 1 see exactly the order run 1/epoch 0 saw.
_EPOCH_SEED_STRIDE = 100003


def _log(message: str) -> None:
    """Timestamped line on stdout, flushed.

    Slurm redirects stdout to a FILE, so Python block-buffers it. Without the flush a job that
    is killed at the wall clock loses its last buffer — precisely the lines saying where the
    final checkpoint went.
    """
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}", flush=True)


# -- cooperative stop ------------------------------------------------------------------------


class StopSignal:
    """Cooperative stop flag for SIGUSR1 / SIGTERM / SIGINT.

    The handler does one thing: set a bool (plus a counter and the raw signal number — all
    plain attribute writes). It never raises, never touches the model, the optimizer or a
    file. The loop polls ``requested`` at step boundaries, where the model, the optimizer
    moments and the sampler position are consistent by construction.

    All three signals mean the same thing here — finish this step, checkpoint, exit 0 — but
    they arrive from different places: SIGUSR1 from ``#SBATCH --signal=B:USR1@300`` forwarded
    by the batch script, SIGTERM from preemption or ``scancel``, SIGINT from a terminal.

    Consequence, on purpose: a run wedged INSIDE a step cannot be stopped this way at all.
    Ctrl-C twice will not help; SIGKILL is the escape hatch, and the last checkpoint is what
    survives it.
    """

    DEFAULT_SIGNALS: tuple[int, ...] = (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)

    def __init__(self) -> None:
        self.requested = False
        self.count = 0
        self.signum: int | None = None
        self._previous: dict[int, Any] = {}

    @property
    def name(self) -> str:
        """Name of the first signal received (``'-'`` if none) — resolved lazily, not in the
        handler, so the handler itself stays a handful of attribute writes."""
        return signal.Signals(self.signum).name if self.signum is not None else "-"

    def _handle(self, signum: int, _frame: Any) -> None:
        self.requested = True
        self.count += 1
        if self.signum is None:
            self.signum = signum

    def install(self, signals: Sequence[int] | None = None) -> StopSignal:
        """Install the handler, remembering the previous ones for :meth:`restore`."""
        for sig in signals if signals is not None else self.DEFAULT_SIGNALS:
            self._previous[sig] = signal.signal(sig, self._handle)
        return self

    def restore(self) -> None:
        """Put the previous handlers back.

        Load-bearing in-process: the tests call ``main()`` inside pytest, and leaving a
        SIGINT handler that only flips a bool behind would make Ctrl-C stop working for the
        rest of the session.
        """
        while self._previous:
            sig, handler = self._previous.popitem()
            signal.signal(sig, handler)


# -- resumable sampling ----------------------------------------------------------------------


class EpochFeeder:
    """Deterministic, RESUMABLE batch stream over an ``EpisodeDataset``.

    The requeue chain only sees the whole dataset if the sampler resumes where it stopped.
    Restarting the shuffle at step 0 in every 4-hour chunk replays the same opening batches
    forever: at ~14.5k samples, batch 8 and a few thousand steps per chunk the model would
    keep re-fitting the same slice and never reach the tail — and the loss curve keeps
    falling, so nothing about that failure is visible.

    Hence the position is ``(epoch, batch_in_epoch)`` and both go into the checkpoint. The
    shuffle of epoch ``e`` is a pure function of ``(seed, e)``: ``randperm`` under a generator
    seeded ``seed * _EPOCH_SEED_STRIDE + e``, independent of the global torch stream (which
    the model init and the trainer's own generator move around). That permutation is handed to
    the ``DataLoader`` as its SAMPLER, which is what makes :meth:`seek` exact AND cheap —
    skipped batches are indices dropped off the front, never decoded frames.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        batch_size: int,
        seed: int,
        device: torch.device,
        drop_last: bool = False,
    ) -> None:
        size = len(dataset)
        if size == 0:
            raise ValueError("EpochFeeder needs a non-empty dataset")
        self.dataset = dataset
        # Same clamp the in-repo trainers use: a batch larger than the set would yield an
        # empty epoch with drop_last, and the D1-sized fixtures are smaller than batch_size.
        self.batch_size = max(1, min(int(batch_size), size))
        self.seed = int(seed)
        self.device = device
        self.drop_last = bool(drop_last)
        self._epoch = 0
        self._batch_in_epoch = 0

    @property
    def batches_per_epoch(self) -> int:
        size = len(self.dataset)
        return size // self.batch_size if self.drop_last else -(-size // self.batch_size)

    @property
    def position(self) -> tuple[int, int]:
        """``(epoch, index of the NEXT batch in that epoch)`` — what gets checkpointed."""
        return self._epoch, self._batch_in_epoch

    def epoch_order(self, epoch: int) -> list[int]:
        """Sample indices of ``epoch``, reproducible from ``(seed, epoch)`` alone."""
        generator = torch.Generator().manual_seed(self.seed * _EPOCH_SEED_STRIDE + int(epoch))
        return torch.randperm(len(self.dataset), generator=generator).tolist()

    def seek(self, epoch: int, batch_in_epoch: int) -> None:
        """Jump to a checkpointed position. Call before iterating — a live iterator holds a
        DataLoader built for the old position and would ignore the jump."""
        if epoch < 0 or batch_in_epoch < 0:
            raise ValueError(f"seek({epoch}, {batch_in_epoch}): positions must be >= 0")
        if batch_in_epoch > self.batches_per_epoch:
            raise ValueError(
                f"seek({epoch}, {batch_in_epoch}): epoch has only "
                f"{self.batches_per_epoch} batches at batch_size {self.batch_size} — the "
                "checkpoint was written with a different batch size or dataset"
            )
        self._epoch, self._batch_in_epoch = int(epoch), int(batch_in_epoch)

    def _loader(self, epoch: int, start_batch: int) -> DataLoader:
        indices = self.epoch_order(epoch)[start_batch * self.batch_size :]
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=indices,  # the materialized permutation, already advanced past start_batch
            collate_fn=collate_episode_batch,
            num_workers=0,
            drop_last=self.drop_last,
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Endless stream of device-ready batches, advancing the epoch when one runs out."""
        while True:
            for batch in self._loader(self._epoch, self._batch_in_epoch):
                self._batch_in_epoch += 1  # before the yield: position == NEXT batch to consume
                yield prepare_tensor_batch(batch, self.device)
            self._epoch += 1
            self._batch_in_epoch = 0


# -- checkpointing ---------------------------------------------------------------------------


def _fsync_path(path: Path) -> None:
    """fsync a file or directory entry (durability of the rename, not just the bytes)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(path: Path) -> None:
    for child in sorted(path.iterdir()):
        if child.is_file():
            _fsync_path(child)
    _fsync_path(path)


class CheckpointManager:
    """Atomic, prunable checkpoint store under ``out_dir``.

    Layout::

        out_dir/checkpoints/step-000600/model.safetensors  weights + embedded config/metadata
        out_dir/checkpoints/step-000600/trainer_state.pt   optimizer + RNG + sampler position
        out_dir/latest -> checkpoints/step-000600          relative symlink

    Atomicity is the point. The job can be SIGKILLed at any instant, including in the middle
    of a 30-minute-interval write, and the next job in the chain must still find an intact
    ``latest``. So everything is written into ``step-NNNNNN.tmp/``, fsynced, then ``os.replace``d
    into place (a rename inside one filesystem is atomic), and only then does ``latest`` move —
    itself a rename of ``latest.tmp``. ``latest`` therefore points at the previous complete
    checkpoint or at the new complete one, never at a half-written directory. :meth:`latest`
    re-validates the target anyway and falls back to the newest intact directory.

    The symlink is RELATIVE so the whole run directory can be moved or copied off scratch.
    """

    def __init__(self, out_dir: str | Path, *, total_limit: int | None = None) -> None:
        self.out_dir = Path(out_dir)
        self.checkpoints_dir = self.out_dir / CHECKPOINT_DIRNAME
        self.latest_link = self.out_dir / LATEST_LINK
        self.total_limit = total_limit

    # -- inspection --------------------------------------------------------------------

    @staticmethod
    def is_complete(path: Path) -> bool:
        """True iff ``path`` holds BOTH checkpoint files — the definition of "restorable"."""
        return (
            path.is_dir()
            and (path / MODEL_FILENAME).is_file()
            and (path / TRAINER_STATE_FILENAME).is_file()
        )

    @staticmethod
    def step_of(path: Path) -> int | None:
        name = path.name
        if not name.startswith(STEP_DIR_PREFIX):
            return None
        suffix = name[len(STEP_DIR_PREFIX) :]
        return int(suffix) if suffix.isdigit() else None  # ".tmp"/".stale" fail isdigit()

    def existing(self) -> list[Path]:
        """Complete checkpoint directories, oldest first."""
        if not self.checkpoints_dir.is_dir():
            return []
        found = [
            child
            for child in self.checkpoints_dir.iterdir()
            if self.step_of(child) is not None and self.is_complete(child)
        ]
        return sorted(found, key=lambda p: self.step_of(p) or 0)

    def latest(self) -> Path | None:
        """Newest restorable checkpoint, or ``None``.

        Prefers the ``latest`` symlink but never trusts it: a truncated write, a manual
        ``rm -rf`` or a scratch purge would otherwise abort the whole chain when a perfectly
        good older checkpoint is sitting right next to it.
        """
        if self.latest_link.is_symlink():
            target = Path(os.path.realpath(self.latest_link))
            if self.is_complete(target):
                return target
            _log(f"WARN {self.latest_link} -> {target} is not a complete checkpoint; scanning")
        found = self.existing()
        return found[-1] if found else None

    # -- writing -----------------------------------------------------------------------

    def save(
        self,
        trainer: JointTrainer,
        *,
        step: int,
        epoch: int,
        batch_in_epoch: int,
        run_id: str,
        elapsed_s: float = 0.0,
        dataset_snapshot_ref: str | None = None,
        train_episode_ids: Sequence[str] | None = None,
        git_commit: str | None = None,
        adapter_only: bool = False,
    ) -> Path:
        """Write ``step`` atomically, repoint ``latest``, prune. Returns the final directory."""
        from wam.training._utils import save_checkpoint as write_checkpoint

        final = self.checkpoints_dir / f"{STEP_DIR_PREFIX}{step:0{STEP_DIR_DIGITS}d}"
        tmp = final.with_name(final.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)

        # Adapters only for a big backbone: trainable_state_dict() is LoRA + heads, so a
        # 30-minute checkpoint interval stays affordable next to a frozen multi-GB base. The
        # embedded config + RunMetadata ride along either way (FR-10, AC-04).
        payload_state = trainer.model.trainable_state_dict() if adapter_only else None
        # The RunMetadata is assembled HERE rather than inside JointTrainer.save_checkpoint,
        # which would otherwise need a fourth pure pass-through kwarg (and the same one added
        # to ActionOnlyTrainer to keep them from diverging) to serve one caller. The chain
        # driver is the component that knows which episodes were fed; the trainer does not.
        # Field-for-field identical to what trainer.save_checkpoint would have produced —
        # including checkpoint_ref pointing at the .tmp path, which is what every archived
        # checkpoint records — so no recorded provenance changes shape.
        metadata = RunMetadata.create(
            run_id,
            trainer.config,
            checkpoint_ref=str(tmp / MODEL_FILENAME),
            dataset_snapshot_ref=dataset_snapshot_ref,
            train_episode_ids=train_episode_ids,
            git_commit=git_commit,
        )
        write_checkpoint(
            trainer.model, trainer.config, tmp / MODEL_FILENAME, metadata, state_dict=payload_state
        )
        trainer.metadata = metadata  # keep the trainer's own view of its last write intact
        torch.save(
            {
                "trainer": trainer.state_dict(),  # optimizer moments + both RNG streams
                "step": int(step),
                "epoch": int(epoch),
                "batch_in_epoch": int(batch_in_epoch),
                "elapsed_s": float(elapsed_s),
                "adapter_only": bool(adapter_only),
                "run_metadata": metadata.to_dict(),
            },
            tmp / TRAINER_STATE_FILENAME,
        )
        _fsync_tree(tmp)

        if final.exists():
            # Re-saving the same step (a resume that stops before completing a step).
            # os.replace refuses a non-empty destination directory, so move it aside first.
            stale = final.with_name(final.name + ".stale")
            shutil.rmtree(stale, ignore_errors=True)
            os.replace(final, stale)
        os.replace(tmp, final)
        shutil.rmtree(final.with_name(final.name + ".stale"), ignore_errors=True)
        self._point_latest_at(final)
        self.prune()
        return final

    def _point_latest_at(self, path: Path) -> None:
        tmp_link = self.out_dir / (LATEST_LINK + ".tmp")
        if tmp_link.is_symlink() or tmp_link.exists():
            tmp_link.unlink()
        os.symlink(os.path.join(CHECKPOINT_DIRNAME, path.name), tmp_link, target_is_directory=True)
        os.replace(tmp_link, self.latest_link)  # atomic swap; never unlink-then-symlink
        _fsync_path(self.out_dir)

    def prune(self) -> list[Path]:
        """Keep the newest ``total_limit`` checkpoints; drop leftovers of killed writes.

        The newest is structurally safe: slicing ``[:-limit]`` with ``limit >= 1`` can never
        include the last element, so the checkpoint ``latest`` just started pointing at cannot
        be the one pruned.
        """
        removed: list[Path] = []
        if self.checkpoints_dir.is_dir():
            for child in self.checkpoints_dir.iterdir():
                if child.is_dir() and child.name.endswith((".tmp", ".stale")):
                    shutil.rmtree(child, ignore_errors=True)  # corpse of an interrupted write
        if self.total_limit is None or self.total_limit < 1:
            return removed
        for path in self.existing()[: -self.total_limit]:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
        return removed

    # -- reading -----------------------------------------------------------------------

    def restore(self, trainer: JointTrainer, path: Path) -> dict[str, Any]:
        """Load weights + optimizer/RNG/sampler state into ``trainer``; returns the payload.

        Two assertions turn a silent wrong-model resume into a hard stop:

        - ``unexpected == []`` — a tensor in the file that the model has no slot for means the
          architecture moved under the checkpoint; loading the rest would train a hybrid.
        - ``missing`` may only contain FROZEN parameters and buffers — that is exactly the
          complement of ``trainable_state_dict()``, i.e. what ``--save-adapter-only`` leaves
          out on purpose (it comes from the base weights instead). A missing TRAINABLE tensor
          means a head silently kept its fresh random init.
        """
        from safetensors.torch import load_file

        result = trainer.model.load_state_dict(load_file(str(path / MODEL_FILENAME)), strict=False)
        if result.unexpected_keys:
            raise RuntimeError(
                f"{path}: checkpoint has {len(result.unexpected_keys)} tensor(s) the model does "
                f"not accept, e.g. {sorted(result.unexpected_keys)[:5]} — config/architecture "
                "drift between this code and the checkpoint"
            )
        allowed = set(trainer.model.frozen_parameter_names())
        allowed |= {name for name, _ in trainer.model.named_buffers()}
        missing = sorted(set(result.missing_keys) - allowed)
        if missing:
            raise RuntimeError(
                f"{path}: {len(missing)} TRAINABLE tensor(s) absent from the checkpoint, e.g. "
                f"{missing[:5]} — those would silently resume from a fresh random init"
            )
        # weights_only=False: our own file, and it holds RunMetadata-shaped plain dicts next to
        # the tensors. map_location='cpu' is required, not cosmetic — torch's RNG setters take
        # CPU ByteTensors, and Optimizer.load_state_dict moves the moments to the params itself.
        payload = torch.load(path / TRAINER_STATE_FILENAME, map_location="cpu", weights_only=False)
        trainer.load_state_dict(payload["trainer"])
        return payload


# -- config resolution -----------------------------------------------------------------------


def _read_embedded_config(model_path: Path) -> tuple[dict[str, Any], RunMetadata]:
    """Config + ``RunMetadata`` from a checkpoint header, WITHOUT materializing the tensors.

    The config has to be known before the model exists (it is what builds it), and an
    adapter-only Wan checkpoint is still hundreds of MB — reading it twice per resume is pure
    startup cost on a job whose wall clock is the scarce resource.
    """
    from safetensors import safe_open

    with safe_open(str(model_path), framework="pt") as handle:
        meta = handle.metadata() or {}
    if CHECKPOINT_CONFIG_KEY not in meta or CHECKPOINT_METADATA_KEY not in meta:
        raise ValueError(f"{model_path}: not a WAM checkpoint (missing embedded config/metadata)")
    config_dict: dict[str, Any] = json.loads(meta[CHECKPOINT_CONFIG_KEY])
    metadata = RunMetadata.model_validate(json.loads(meta[CHECKPOINT_METADATA_KEY]))
    return config_dict, metadata


def _resolve_backbone_block(
    training: dict[str, Any],
    backbone_config: Path | None,
    *,
    source: str | None,
    allow_download: bool,
    device: str | None = None,
) -> dict[str, Any]:
    """The ``backbone:`` section to train with, spliced from ``--backbone-config``.

    The split is deliberate: WHAT the backbone is (architecture, readout depth, LoRA surface)
    is a versioned artifact under ``configs/model/``, while WHERE its weights sit is
    machine-local. Baking the cluster path into a committed YAML would make ``config_hash``
    differ between two machines training the identical model, and AC-04 traceability dies with
    it — hence ``--backbone-source`` on the command line, folded in here.

    ``checkpoint_path``/``allow_download``/``device`` are applied only if the resolved backbone
    kind actually declares them, so the tiny CPU twin can take the same flag set (FR-09: no
    branch on a concrete backbone anywhere).

    ``--device`` has to land HERE and not only on ``training.device``: ``--backbone-config``
    replaces this block wholesale, so the device the training YAML declared is gone and a
    backbone kind that carries its own device field falls back to its default (``cuda`` for
    Wan). The backbone is then built with ``load=True`` BEFORE ``JointTrainer`` moves the model,
    so the drift is not cosmetic — it materializes the whole frozen tower on the wrong device,
    which on a CUDA-less box is a hard failure and on a GPU box is a pointless round trip.
    """
    block = dict(training.get("backbone") or {})
    if backbone_config is not None:
        data = load_config(backbone_config)
        if "backbone" not in data:
            raise SystemExit(f"{backbone_config}: missing top-level 'backbone' section")
        block = dict(data["backbone"])
    # Validate here so a broken backbone file names ITSELF in the error, instead of surfacing
    # as an opaque discriminated-union failure inside the training config.
    parsed = build_backbone_config(block)
    fields = type(parsed).model_fields
    for name, value, warn in (
        ("checkpoint_path", str(source) if source else None, True),
        ("allow_download", True if allow_download else None, True),  # the flag only turns it ON
        # No warning: a backbone that holds no weights of its own (tiny) has nothing to place,
        # and JointTrainer.to(device) already covers it. Silence keeps the CPU twin's log clean.
        ("device", device, False),
    ):
        if value is None:
            continue
        if name not in fields:
            if warn:
                _log(f"WARN backbone kind {parsed.kind!r} has no {name!r} field — flag ignored")
            continue
        block[name] = value
    return block


def _build_yaml_config(args: argparse.Namespace) -> JointTrainingConfig:
    """``--training-config`` + spliced backbone + CLI overrides -> validated config.

    Dim mismatches are NOT patched here. ``JointTrainingConfig._consistent_dims`` raises when
    e.g. ``head.feature_dim`` does not match the backbone's, and that error is the deliverable:
    quietly rewriting the head width to fit a swapped backbone produces a model that trains
    fine and answers a different question than the one the config describes.
    """
    data = load_config(args.training_config)
    if "training" not in data:
        raise SystemExit(f"{args.training_config}: missing top-level 'training' section")
    training = dict(data["training"])
    training["backbone"] = _resolve_backbone_block(
        training,
        args.backbone_config,
        source=args.backbone_source,
        allow_download=args.allow_download,
        device=args.device,
    )
    for key, value in (
        ("device", args.device),
        ("camera", args.camera),
        ("seed", args.seed),
        ("lr", args.lr),
        ("backbone_lr", args.backbone_lr),
        ("batch_size", args.batch_size),
        ("steps", args.steps),
    ):
        if value is not None:
            training[key] = value
    return JointTrainingConfig.model_validate(training)


def _restored_config(
    checkpoint: Path, device: str | None
) -> tuple[JointTrainingConfig, RunMetadata]:
    """The checkpoint's own config, with ``--device`` as the single permitted override."""
    config_dict, metadata = _read_embedded_config(checkpoint / MODEL_FILENAME)
    if device is not None and device != config_dict.get("device"):
        _log(
            f"WARN --device {device!r} overrides the checkpoint's {config_dict.get('device')!r}. "
            "It is the one field a resume may change — and it IS part of config_hash, so the "
            "chain's hash changes with it"
        )
        config_dict = {**config_dict, "device": device}
    return JointTrainingConfig.model_validate(config_dict), metadata


def _report_config_drift(
    restored: JointTrainingConfig,
    yaml_config: JointTrainingConfig | None,
    yaml_error: Exception | None,
) -> None:
    """Say out loud when the YAML on disk has moved away from the resumed checkpoint.

    AC-04 wants every rollout traceable to ONE config hash, and a run is a chain of jobs. If
    chunk 7 silently picked up an edited YAML, the directory would hold two experiments
    sharing a checkpoint lineage and no way to tell which steps belonged to which. So the
    checkpoint always wins and the drift is only reported.
    """
    if yaml_error is not None:
        _log(f"WARN --training-config no longer validates ({yaml_error!r}); resuming anyway")
        return
    if yaml_config is None:
        return
    restored_hash, yaml_hash = config_hash(restored), config_hash(yaml_config)
    if restored_hash == yaml_hash:
        return
    on_disk = yaml_config.model_dump(mode="json")
    in_ckpt = restored.model_dump(mode="json")
    changed = sorted(key for key, value in on_disk.items() if in_ckpt.get(key) != value)
    _log(
        f"WARN config drift: on-disk {yaml_hash[:12]} != checkpoint {restored_hash[:12]} "
        f"(sections {changed}). Using the CHECKPOINT's config; point --out-dir somewhere new "
        "for a genuinely different experiment"
    )


def _dataset_snapshot_hash(root: Path, episodes: Sequence[Path] | None = None) -> str:
    """Content hash of a dataset directory (AC-04 ``dataset_snapshot_ref``).

    Mirrors ``scripts/overfit_d1.py``: every manifest embeds sha256 checksums of its data
    files, so hashing the manifests pins the full content without decoding a single frame.

    ``episodes`` narrows the hash to the episodes actually trained on. That matters: a run
    that excludes a holdout has a different training set than one that does not, and a
    snapshot ref covering the whole root would report the two as identical.
    """
    from wam.data.episode import MANIFEST_FILENAME, list_episodes

    digest = hashlib.sha256()
    for episode_dir in list_episodes(root) if episodes is None else episodes:
        digest.update(str(episode_dir.relative_to(root)).encode("utf-8"))
        digest.update((episode_dir / MANIFEST_FILENAME).read_bytes())
    return f"sha256:{digest.hexdigest()}"


def _load_excluded_ids(path: Path) -> set[str]:
    """Thin alias for ``wam.evaluation.load_episode_ids``.

    The evaluator (``scripts/eval_t16.py``) reads the same file with the same function, so the
    fine-tune's exclusion and the holdout it is scored on cannot drift apart.
    """
    from wam.evaluation import load_episode_ids

    return load_episode_ids(path)


def _training_episodes(
    root: Path, exclude: Path | None, include: Path | None = None
) -> tuple[list[Path], set[str]]:
    """Episode dirs under ``root`` minus ``exclude``, narrowed to ``include``, verified.

    ``include`` is an I-8 rung: a committed subset of the training set, so the only thing that
    varies between rungs is how many episodes there are. It is checked against the holdout
    first, because a rung file naming a held-out episode is the exact leak the whole split
    proof exists to stop — and it has to die at the PRODUCING end too, or the leak reaches a
    checkpoint and only the evaluator stands between it and a published number.

    The filter is applied to the already-sorted ``list_episodes`` order and never re-sorts:
    ``_dataset_snapshot_hash`` is an order-sensitive sequential digest, and the evaluator
    replays the recorded order to reproduce it.
    """
    from wam.data.episode import list_episodes

    episodes = list_episodes(root)
    excluded: set[str] = set()
    if exclude is not None:
        excluded = _load_excluded_ids(exclude)
        present = {p.name for p in episodes}
        missing = excluded - present
        if missing:
            # Silently training on an episode the evaluator believes is held out is the exact
            # failure this flag exists to prevent, so a split that does not line up is fatal.
            raise SystemExit(
                f"--exclude-episodes lists {len(missing)} episode(s) absent from {root}: "
                f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
            )
    kept = [p for p in episodes if p.name not in excluded]
    if include is not None:
        included = _load_excluded_ids(include)
        overlap = sorted(included & excluded)
        if overlap:
            raise SystemExit(
                f"--train-episodes and --exclude-episodes share {len(overlap)} episode(s): "
                f"{overlap[:5]}{'...' if len(overlap) > 5 else ''} — a rung that names a "
                "held-out episode would train on what it is later scored against"
            )
        absent = sorted(included - {p.name for p in episodes})
        if absent:
            # A rung file that does not line up with the dataset is a typo, not a smaller rung.
            raise SystemExit(
                f"--train-episodes lists {len(absent)} episode(s) absent from {root}: "
                f"{absent[:5]}{'...' if len(absent) > 5 else ''}"
            )
        kept = [p for p in kept if p.name in included]
    if not kept and (exclude is not None or include is not None):
        selectors = " / ".join(
            flag
            for flag, given in (
                ("--exclude-episodes", exclude is not None),
                ("--train-episodes", include is not None),
            )
            if given
        )
        raise SystemExit(f"{selectors} removed every episode under {root}")
    return kept, excluded


def _resolve_resume(manager: CheckpointManager, value: str | None) -> Path | None:
    """``--resume`` -> a checkpoint directory or ``None`` (fresh start).

    ``latest`` resolving to nothing is NOT an error: the first job of a chain passes exactly
    the same flags as the tenth, and refusing to start would deadlock the whole schedule. An
    explicit path that is not restorable IS an error — the operator asked for that one.
    """
    if value is None or value.strip().lower() in ("", "none", "off", "no"):
        return None
    if value.strip().lower() == "latest":
        found = manager.latest()
        _log(
            f"--resume latest -> {found}"
            if found is not None
            else "--resume latest: no checkpoint yet, starting fresh (first job of the chain)"
        )
        return found
    path = Path(value)
    if path.name == MODEL_FILENAME:
        path = path.parent  # tolerate a path to the weights file
    if not manager.is_complete(path):
        raise SystemExit(
            f"--resume {value}: not a complete checkpoint directory "
            f"(need {MODEL_FILENAME} + {TRAINER_STATE_FILENAME})"
        )
    return path


# -- training --------------------------------------------------------------------------------


def _training_step(
    trainer: JointTrainer, batches: Sequence[dict[str, Any]], *, step: int
) -> dict[str, float]:
    """One optimizer update over ``len(batches)`` micro-batches.

    A single micro-batch delegates to ``JointTrainer.step`` verbatim. That is not a
    micro-optimization: the resume-equivalence test compares this loop against that exact
    method, so the common path must not grow a second implementation that can drift from it.
    ``--grad-accum > 1`` takes the local loop only because ``JointTrainer.step`` is defined as
    one complete forward/backward/clip/update.
    """
    if len(batches) == 1:
        return trainer.step(batches[0], step=step)
    scale = 1.0 / len(batches)
    trainer.optimizer.zero_grad(set_to_none=True)
    totals: dict[str, float] = {}
    for batch in batches:
        losses = trainer.compute_losses(batch)
        (losses["total"] * scale).backward()  # scaled so the update matches one big batch
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * scale
    grad_norm = float(
        torch.nn.utils.clip_grad_norm_(
            [p for p in trainer.model.parameters() if p.requires_grad], trainer.config.grad_clip
        )
    )
    trainer.optimizer.step()
    return {**totals, "step": float(step), "grad_norm": grad_norm}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train_t16_lora.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # -- what the batch script passes ---------------------------------------------------
    parser.add_argument(
        "--backbone-config",
        type=Path,
        default=None,
        help="YAML with a top-level 'backbone:' section replacing the training config's "
        "(e.g. configs/model/wan22_ti2v_5b.yaml); default: use the training config's own",
    )
    parser.add_argument(
        "--backbone-source",
        type=str,
        default=None,
        help="machine-local weights directory -> backbone checkpoint_path (never in a YAML)",
    )
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="episode dataset root")
    parser.add_argument(
        "--exclude-episodes",
        type=Path,
        default=None,
        help="file of episode ids to hold out (plain list, or a baseline predictions.jsonl) — "
        "fine-tuning on the episodes the ablation scores on invalidates the AC-07 verdict",
    )
    parser.add_argument(
        "--train-episodes",
        type=Path,
        default=None,
        help="file of episode ids to train on (plain list, or a predictions.jsonl) — one rung of "
        "the I-8 data-scaling curve; default: every episode under --dataset minus "
        "--exclude-episodes",
    )
    parser.add_argument("--camera", type=str, default=None, help="override training.camera")
    parser.add_argument("--out-dir", type=Path, required=True, help="run dir, shared by the chain")
    parser.add_argument(
        "--resume",
        type=str,
        default="none",
        help="'latest' (newest checkpoint in --out-dir, fresh start if none), 'none', or a path",
    )
    parser.add_argument("--checkpoint-every-min", type=float, default=30.0)
    parser.add_argument("--checkpoints-total-limit", type=int, default=3)
    parser.add_argument(
        "--save-adapter-only",
        action="store_true",
        help="checkpoint trainable tensors only (LoRA + heads) instead of the frozen base too",
    )
    parser.add_argument("--device", type=str, default=None, help="override training.device")
    # -- defaulted extras ---------------------------------------------------------------
    parser.add_argument("--steps", type=int, default=None, help="override training.steps")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=1, help="micro-batches per update")
    parser.add_argument(
        "--max-hours",
        type=float,
        default=3.6,
        help="wall-clock budget; stop at the next step boundary (default leaves ~24 min of "
        "the 4 h cap for startup, the final checkpoint and the requeue)",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--allow-download", action="store_true", help="let the backbone fetch")
    parser.add_argument(
        "--offload-text",
        action="store_true",
        help=OFFLOAD_TEXT_HELP + ". READ THIS BEFORE USING IT ON A LONG RUN: the umT5 forward "
        "then runs on the CPU, and condition_text memoizes per instruction STRING, so the cost "
        "is one CPU encode per DISTINCT instruction in the training set, not per step. That is "
        "free for the GR00T corpus (all 402 episodes of gr00t-apple-full and -grip carry the "
        "one instruction 'move the apple to the plate'), and a per-batch stall on a "
        "multi-instruction set. This script counts the distinct instructions at startup and "
        "warns when there is more than one.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="default: --out-dir basename")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve configs, count episodes, print the plan, exit 0 — no weights, no GPU",
    )
    args = parser.parse_args(argv)
    if args.grad_accum < 1:
        parser.error("--grad-accum must be >= 1")
    if args.log_every < 1:
        parser.error("--log-every must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir: Path = args.out_dir
    done_marker = out_dir / DONE_FILENAME
    # FIRST, before anything expensive: a requeued job whose predecessor finished must not
    # train one more chunk. The batch script also checks DONE, but a manual re-submit or a
    # scheduler race would otherwise restart a completed run.
    if done_marker.is_file():
        _log(f"{done_marker} exists — run already finished, nothing to do")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    manager = CheckpointManager(out_dir, total_limit=args.checkpoints_total_limit)
    resume_from = _resolve_resume(manager, args.resume)

    yaml_config: JointTrainingConfig | None = None
    yaml_error: Exception | None = None
    try:
        yaml_config = _build_yaml_config(args)
    except Exception as exc:
        # Tolerated ONLY when a checkpoint can supply the config instead: a resume must not be
        # blocked by a YAML that was edited (or moved) after the chain started.
        if resume_from is None:
            raise  # fresh start: a config that does not validate is a hard, loud failure
        yaml_error = exc

    ckpt_metadata: RunMetadata | None = None
    if resume_from is None:
        config = yaml_config  # never None here — the except above re-raised on a fresh start
        _log(f"fresh start from {args.training_config}")
    else:
        config, ckpt_metadata = _restored_config(resume_from, args.device)
        _report_config_drift(config, yaml_config, yaml_error)

    run_id = args.run_id or out_dir.name
    hash_ = config_hash(config)
    _log(
        f"run_id={run_id} config_hash={hash_[:12]} backbone={config.backbone.kind} "
        f"device={config.device} steps={config.steps} batch={config.batch_size} "
        f"grad_accum={args.grad_accum} camera={config.camera}"
    )

    if args.dry_run:
        # Login-node pre-flight: proves the flag surface, the config splice and the dim
        # cross-checks before a job is ever queued. Deliberately touches neither the weights
        # nor the frames — both are the expensive part and neither can fail in a new way here.
        episodes, excluded = (
            _training_episodes(args.dataset, args.exclude_episodes, args.train_episodes)
            if args.dataset.is_dir()
            else ([], set())
        )
        _log(
            f"dry run OK | dataset {args.dataset} ({len(episodes)} train episodes, "
            f"{len(excluded)} held out"
            + (f", rung {args.train_episodes}" if args.train_episodes else "")
            + f") | out {out_dir} | "
            f"resume={resume_from} | checkpoint every {args.checkpoint_every_min} min, "
            f"keep {args.checkpoints_total_limit}, adapter_only={args.save_adapter_only} | "
            f"max_hours={args.max_hours}"
        )
        return 0

    episodes, excluded = _training_episodes(
        args.dataset, args.exclude_episodes, args.train_episodes
    )
    train_episode_ids = [p.name for p in episodes]
    _log(
        f"training on {len(episodes)} episodes, {len(excluded)} held out"
        + (f" via {args.exclude_episodes}" if args.exclude_episodes else " (no holdout)")
        + (f" | rung {args.train_episodes}" if args.train_episodes else "")
    )
    dataset = EpisodeDataset(
        episodes,
        camera=config.camera,
        num_frames=config.backbone.num_frames,
        chunk_steps=config.head.num_steps,
    )
    snapshot_ref = _dataset_snapshot_hash(args.dataset, episodes)
    if ckpt_metadata is not None and ckpt_metadata.dataset_snapshot_ref not in (None, snapshot_ref):
        # Three I-8 rungs share one dataset root and differ ONLY in --train-episodes, so
        # resuming rung-120's --out-dir with rung-40's rung file is one copy-paste away. That
        # would load rung-120's weights and stamp rung-40's snapshot ref onto the final
        # checkpoint: wrong, internally consistent, and the split proof would then PASS on a
        # lie. _report_config_drift does not cover it — the episode list is not part of the
        # config — so it has to be checked against the recorded snapshot instead.
        recorded = ckpt_metadata.train_episode_ids
        raise SystemExit(
            "REFUSING TO RESUME — this job's training set is not the one the checkpoint was "
            "trained on.\n"
            f"  checkpoint trained on: {ckpt_metadata.dataset_snapshot_ref} "
            f"({len(recorded) if recorded is not None else 'unrecorded'} episodes)\n"
            f"  this job would train on: {snapshot_ref} ({len(episodes)} episodes)\n"
            f"  (--dataset {args.dataset}, --exclude-episodes {args.exclude_episodes}, "
            f"--train-episodes {args.train_episodes})\n"
            "Point --out-dir at a new directory for a different training set."
        )
    # On the cluster the repo is rsynced without .git, so read_git_commit() finds no repository
    # and reports "unknown" — which would leave the deliverable run with no code provenance.
    # cluster/discoverer/sync.sh stamps the hash into WAM_GIT_COMMIT for exactly this case.
    git_commit = os.environ.get("WAM_GIT_COMMIT", "").strip() or read_git_commit(_REPO_ROOT)

    # Seed BEFORE the backbone so an injected backbone initializes from the same stream the
    # non-injected path would have used; JointTrainer re-seeds and builds the heads after.
    torch.manual_seed(config.seed)
    advise_alloc_conf(config.device)
    backbone = build_backbone(config.backbone, load=True)
    trainer = JointTrainer(config, backbone=backbone)
    feeder = EpochFeeder(
        dataset, batch_size=config.batch_size, seed=config.seed, device=trainer.device
    )

    step = 0
    if resume_from is not None:
        payload = manager.restore(trainer, resume_from)
        step = int(payload["step"])
        feeder.seek(int(payload["epoch"]), int(payload["batch_in_epoch"]))

    if args.offload_text:
        # LAST, on purpose: JointTrainer.__init__ ends in .to(self.device), which
        # WanFlowBackbone._apply forwards to the held towers, so anything offloaded before this
        # point comes straight back onto the GPU. Restore only fills tensors in place.
        instructions = distinct_instructions(episodes)
        if len(instructions) > 1:
            # Loud, because the cost model inverts here. condition_text memoizes per prompt
            # string, so one instruction means one CPU umT5 forward for the entire run; N
            # instructions mean the cache misses whenever a batch introduces an unseen one, and
            # on a set with many prompts that is a CPU forward inside the training step.
            _log(
                f"WARNING --offload-text: this training set has {len(instructions)} distinct "
                "instructions. The umT5 forward now runs on the CPU and is memoized per prompt, "
                "so the first appearance of each one stalls the step it lands in. That is fine "
                "for a handful and NOT fine for a corpus with per-episode phrasing — measure a "
                "few steps before committing to a long run."
            )
        offload_text_encoder(backbone, log=_log)
        _log(f"--offload-text: umT5 on CPU, {len(instructions)} distinct instruction(s) to encode")
    _log(
        f"{len(dataset)} samples, {feeder.batches_per_epoch} batches/epoch | start at step "
        f"{step}/{config.steps}, sampler at epoch {feeder.position[0]} batch {feeder.position[1]}"
    )

    metadata = RunMetadata.create(
        run_id,
        config,
        checkpoint_ref=str(out_dir),
        dataset_snapshot_ref=snapshot_ref,
        train_episode_ids=train_episode_ids,
        git_commit=git_commit,
    )
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n"
    )

    started = time.monotonic()
    deadline = started + args.max_hours * 3600.0
    interval_s = args.checkpoint_every_min * 60.0
    last_checkpoint_at = started
    saved_step: int | None = None
    reason = "step budget"
    stop = StopSignal().install()
    batches = iter(feeder)

    def _save(current: int) -> Path:
        epoch, batch_in_epoch = feeder.position
        path = manager.save(
            trainer,
            step=current,
            epoch=epoch,
            batch_in_epoch=batch_in_epoch,
            run_id=run_id,
            elapsed_s=time.monotonic() - started,
            dataset_snapshot_ref=snapshot_ref,
            train_episode_ids=train_episode_ids,
            git_commit=git_commit,
            adapter_only=args.save_adapter_only,
        )
        _log(f"checkpoint step {current} -> {path} (sampler epoch {epoch} batch {batch_in_epoch})")
        return path

    try:
        with JsonlRunLogger(out_dir / "training_log.jsonl", metadata) as logger:
            logger.log_metadata()
            trainer.model.train()
            while step < config.steps:
                # Both stop conditions are polled HERE, at the boundary between two updates:
                # weights, optimizer moments and sampler position all agree at this point, so
                # whatever we checkpoint next resumes exactly.
                if stop.requested:
                    reason = f"{stop.name} (stop requested)"
                    break
                if time.monotonic() >= deadline:
                    reason = f"--max-hours {args.max_hours} reached"
                    break
                micro = [next(batches) for _ in range(args.grad_accum)]
                entry = _training_step(trainer, micro, step=step)
                step += 1
                if step % args.log_every == 0 or step == config.steps:
                    logger.log({"kind": "step", **entry})
                    _log(
                        f"step {step}/{config.steps} total={entry['total']:.6g} "
                        f"video={entry['video']:.4g} action_flow={entry['action_flow']:.4g} "
                        f"grad_norm={entry['grad_norm']:.4g}"
                    )
                now = time.monotonic()
                if now - last_checkpoint_at >= interval_s and step < config.steps:
                    _save(step)
                    saved_step, last_checkpoint_at = step, now
            # ALWAYS on the way out — signalled, out of time or finished. Skipped only when the
            # interval already wrote this exact step, which is the same bytes.
            if saved_step != step:
                _save(step)
                saved_step = step
    finally:
        stop.restore()

    if step >= config.steps:
        done_marker.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "config_hash": hash_,
                    "steps": step,
                    "checkpoint": str(manager.latest()),
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    # Not in config_hash and not in RunMetadata (it describes where a frozen
                    # tower sat, not what was trained) — so without this line a 20k-step run
                    # that encoded its text on the CPU is indistinguishable from one that did
                    # not. It is only the LAST leg's value: a chain that resumed with a
                    # different setting is not recorded here, which is why
                    # scripts/run_i8_rung_local.sh pins it in the resume stamp instead.
                    "offload_text": bool(args.offload_text),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        _log(f"COMPLETE {step}/{config.steps} steps — wrote {done_marker}, no requeue needed")
    else:
        _log(
            f"stopped at step {step}/{config.steps} ({reason}); state is in "
            f"{manager.latest()} — requeue with the same flags to continue"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
