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
  timing.json         wall time of the GPU pass — the one number that is NOT recomputable later

One ``--out`` holds exactly one readout. The four names above are fixed and ``--out`` defaults to
``--run-dir``, so an A/B that forgets to vary it overwrites its own A arm with its B arm; the
script refuses that now instead of leaving two identical halves and no record (see
``guard_out_dir``). How the policy was driven is written into every report as a suffix on
``run_name`` — ``+frame_history`` (T-29), ``+flow32s0`` (T-30), plus ``k``/``t`` for T-30's two
control arms — because several prediction files from one checkpoint otherwise look like several
checkpoints.

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
import time
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
from wam.runtime.offload import OFFLOAD_TEXT_HELP, advise_alloc_conf, offload_text_encoder

CHECKPOINT_DIRNAME = "checkpoints"
MODEL_FILENAME = "model.safetensors"
LATEST_LINK = "latest"
DEFAULT_HOLDOUT = _REPO_ROOT / "configs" / "splits" / "t18_holdout_episodes.txt"

DEFAULT_FLOW_STEPS = 32
"""Euler steps for ``--flow-sampler``, fixed before the T-30 A/B was run.

The field the sampler integrates was never rectified — nothing in the joint objective straightens
it — so the "rectified flow needs one step" reading of the literature does not transfer, and a
1-step sampler measures the straightness of the field rather than what the action branch learned.
Steps are also nearly free next to the backbone: a step is one 3105->256->32 MLP at batch 1
against a measured 79 ms Wan pass (``runs/wan_probe/2026-07-29-zerogpu-5b-readouts.json``), so the
default is set high enough that the integrator is not the suspect. ``63_eval_t30_flow_head.sbatch``
sweeps {1, 4, 16, 32, 64} to show the verdict does not hinge on this number; changing the default
afterwards means a NEW constant, not an edit to this one, or old runs stop being re-derivable.
"""

DEFAULT_FLOW_SEED = 0

DEFAULT_FLOW_MEAN_K = 1
"""Draws averaged per chunk by ``--flow-mean-k``; 1 == one draw, the deployable readout.

Above 1 this is the T-30 rule's MSE-fair arm. ``skill_vs_repeat_pct`` is MSE-derived, and for any
calibrated conditional ``E‖a - draw‖² = E‖a - mean‖² + E‖draw - mean‖²``: a single unbiased draw
scores worse than the conditional mean by exactly the conditional variance, so comparing one draw
against a mean-seeking regression head charges the sampler for sampling. Averaging k draws leaves
1/k of that penalty. It is a MEASUREMENT arm, never a deployment candidate — averaging draws is
the mean-seeking the flow branch exists to avoid.
"""

DEFAULT_FLOW_T0 = 0.0
"""Start of the Euler integration for ``--flow-t0``; 0.0 == from pure noise, the plain readout.

Above 0 this is the T-30 rule's conditioning-mismatch control. Training paired (features from
video noised to t, action latent noised to the same t); the sampler computes one backbone pass at
t=1 on the clean observation and reuses it at every t_k, so near t=0 the velocity head is
evaluated on a combination it never saw. Starting at t0 from the regression chunk re-encoded and
noised to t0 restricts the integration to the region where the two timesteps roughly agree — see
``JointWorldActionModel.sample_action_chunk``. Also a measurement arm: it reads the regression
head first, so it is two readouts per cycle, not one.
"""


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


class ControlArmPolicy:
    """A loaded policy driven through the two T-30 control arms (``--flow-mean-k``/``--flow-t0``).

    ``load_joint_policy`` owns the **deployable** readout surface — the regression head and the
    plain flow sampler — because ``rollout.py`` and ``serve_policy.py`` load through it too, and a
    readout that could ship has to be reachable from all three without a second implementation.
    These two arms are deliberately kept off that surface, because neither is a thing to deploy:
    averaging k draws re-introduces the mean-seeking the flow branch exists to avoid, and the warm
    start runs the regression head first, so a robot on it would pay for both readouts per cycle.
    They exist to make the T-30 comparison readable, and they stop at this script.

    It **wraps**, it does not reimplement: :meth:`predict` calls the same
    ``JointWorldActionModel.predict`` the policy itself would call, with two more keywords. There
    is no second decode path for the arms to disagree with the deployed one about.
    """

    def __init__(self, policy, *, mean_k: int, t0: float) -> None:
        self._policy = policy
        self._mean_k = mean_k
        self._t0 = t0

    def __getattr__(self, name: str):
        """Everything that is not :meth:`predict` — ``camera``, ``device``, ``metadata``,
        ``model``, ``flow_steps`` — is the wrapped policy's, unchanged. Delegating instead of
        re-declaring keeps this from going stale when ``JointCheckpointPolicy`` grows a field."""
        return getattr(self._policy, name)

    def predict(self, observation):
        """Policy protocol. ``model.predict`` is already ``@torch.no_grad()`` (so is
        ``sample_action_chunk``), which is why this wrapper needs no torch of its own."""
        return self._policy.model.predict(
            observation,
            camera=self._policy.camera,
            flow_steps=self._policy.flow_steps,
            flow_seed=self._policy.flow_seed,
            flow_mean_k=self._mean_k,
            flow_t0=self._t0,
        )


def build_policy(
    model_path: Path,
    device: str,
    camera: str | None,
    backbone_source: str | None,
    *,
    offload_text: bool = False,
    flow_steps: int | None = None,
    flow_seed: int = DEFAULT_FLOW_SEED,
    flow_mean_k: int = DEFAULT_FLOW_MEAN_K,
    flow_t0: float = DEFAULT_FLOW_T0,
):
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
        model_path,
        device=device,
        camera=camera,
        backbone_source=backbone_source,
        flow_steps=flow_steps,
        flow_seed=flow_seed,
        # The LOAD-time half of the same flag. Without it the umT5 tower is resident for the
        # whole of load_joint_policy — ~24.18 GB of weights before a single activation — and the
        # offload below cannot run until that has already happened.
        cpu_pinned=("text_encoder",) if offload_text else (),
    )
    if offload_text:
        # After load_joint_policy, never before: JointCheckpointPolicy.__init__ does the
        # .to(device) that WanFlowBackbone._apply forwards to the held towers, so an earlier
        # offload would be undone. With the pin above this is now a no-op move on a tower that
        # is already parked; it stays because it is ALSO the loud refusal that catches
        # --offload-text aimed at a non-Wan checkpoint. Before the ControlArmPolicy wrap only
        # for readability — the wrapper delegates .model, so either side would resolve.
        offload_text_encoder(policy, log=print)
    if flow_mean_k != DEFAULT_FLOW_MEAN_K or flow_t0 != DEFAULT_FLOW_T0:
        policy = ControlArmPolicy(policy, mean_k=flow_mean_k, t0=flow_t0)
    return policy, config, metadata


def readout_tag(
    *,
    frame_history: bool,
    flow_steps: int | None,
    flow_seed: int = DEFAULT_FLOW_SEED,
    flow_mean_k: int = DEFAULT_FLOW_MEAN_K,
    flow_t0: float = DEFAULT_FLOW_T0,
) -> str:
    """The suffix naming HOW the policy was driven: ``+frame_history+flow32s0k8t0.6``.

    One function, three consumers — ``bench.json``'s ``run_name``, ``timing.json``, and
    :func:`guard_out_dir` below. Several prediction files from one checkpoint differ only in how
    the policy was driven, so this tag is the only thing that keeps a comparison between two of
    them from silently reading as a comparison between two checkpoints.

    An arm left at its default appends nothing, which is what keeps archived artifacts parsing
    and re-scoring identically: ``+frame_history`` and ``+flow32s0`` still mean exactly what they
    meant before the control arms existed.
    """
    tag = "+frame_history" if frame_history else ""
    if flow_steps is not None:
        tag += f"+flow{flow_steps}s{flow_seed}"
        if flow_mean_k != DEFAULT_FLOW_MEAN_K:
            tag += f"k{flow_mean_k}"
        if flow_t0 != DEFAULT_FLOW_T0:
            tag += f"t{flow_t0:g}"
    return tag


def archived_readout_tag(out_dir: Path) -> str | None:
    """The tag of the artifacts already in ``out_dir``, or ``None`` if there are none to read.

    ``timing.json`` records the tag verbatim since the control arms landed and the individual
    fields before that; ``bench.json``'s ``run_name`` carries it for every artifact ever written,
    including those from before ``timing.json`` existed. Both are read so the guard works on an
    archive as well as on a fresh run.
    """
    timing_path = out_dir / "timing.json"
    if timing_path.is_file():
        record = json.loads(timing_path.read_text())
        if "readout_tag" in record:
            return str(record["readout_tag"])
        return readout_tag(
            frame_history=bool(record.get("frame_history")),
            flow_steps=record.get("flow_steps"),
            flow_seed=record.get("flow_seed") or DEFAULT_FLOW_SEED,
            flow_mean_k=record.get("flow_mean_k") or DEFAULT_FLOW_MEAN_K,
            flow_t0=record.get("flow_t0") or DEFAULT_FLOW_T0,
        )
    bench_path = out_dir / "bench.json"
    if bench_path.is_file():
        run_name = str(json.loads(bench_path.read_text()).get("run_name", ""))
        _, plus, rest = run_name.partition("+")
        return f"+{rest}" if plus else ""
    return None


def guard_out_dir(out_dir: Path, tag: str) -> None:
    """Refuse to write one readout's artifacts over another's in the same ``--out``.

    ``predictions.jsonl``, ``e1.*``, ``bench.*`` and ``timing.json`` have fixed names and ``--out``
    defaults to ``--run-dir``, so scoring the regression head and then the flow sampler without
    passing ``--out`` twice replaced the A arm of the A/B with the B arm — leaving an A/B of B
    against itself, with nothing on disk to show it. Two help strings said "use a separate --out
    per mode"; that is the whole of what enforced it until now.

    Re-running the SAME arm is allowed and stays idempotent: a re-score after an interrupted pass
    has to work, and it reproduces the same four files. The one hole left is a pass that died
    between writing ``predictions.jsonl`` and writing the reports — that directory identifies as
    empty and the next arm may overwrite it, which is right: a pass with no report is not an arm.
    """
    archived = archived_readout_tag(out_dir)
    if archived is None or archived == tag:
        return
    raise SystemExit(
        f"REFUSING TO SCORE — {out_dir} already holds artifacts from a different readout.\n"
        f"  on disk:      run_name ...{archived or '(no suffix: tiled frames, regression head)'}\n"
        f"  this command: run_name ...{tag or '(no suffix: tiled frames, regression head)'}\n"
        "Every file this script writes has a fixed name, so continuing would overwrite one arm "
        "of the A/B with the other and leave no trace that it happened. Pass a separate --out "
        "per readout (that is what 63_eval_t30_flow_head.sbatch does), or delete the directory "
        "if the old arm is genuinely dead."
    )


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
        "--offload-text",
        action="store_true",
        help=OFFLOAD_TEXT_HELP + ". NOT bit-identical, and nobody has measured the difference: "
        "condition_text runs the umT5 forward on whichever device the tower is on, so this moves "
        "a bf16 encode from cuBLAS to the CPU, and PyTorch guarantees no cross-device bitwise "
        "equality. The result is memoized and becomes the frozen text context for the whole eval, "
        "so any drift applies uniformly to every episode rather than varying within a run. Do not "
        "mix offloaded and non-offloaded numbers in one comparison. Cost is one "
        "CPU umT5 encode per DISTINCT instruction, because condition_text memoizes per prompt — "
        "and every episode of the default dataset (gr00t-apple-full) carries the same one, so "
        "in practice that is a single encode per run. OFF by default, like every flag here.",
    )
    parser.add_argument(
        "--frame-history",
        action="store_true",
        help="show the policy the real num_frames window ending at each chunk — the same window "
        "EpisodeDataset fed during training — instead of one frame tiled num_frames times. OFF "
        "by default so this reproduces the runs recorded before 2026-08-01, all of which are "
        "'tiled' numbers; T-29 (job 184648) ran the A/B and measured the difference at +10.65 pp "
        "on skill_vs_repeat_pct, not enough to clear L1. This flag is the in-distribution mode — "
        "prefer it for new runs. Needs a separate --out per mode, which this script refuses to "
        "run without.",
    )
    parser.add_argument(
        "--flow-sampler",
        action="store_true",
        help="read the action chunk out of the trained rectified-flow branch (velocity_head + "
        "action_recon) instead of the single-shot regression head. OFF by default so this "
        "reproduces every run recorded before the flag existed; the A/B against the regression "
        "head is T-30 (docs/improvements.md I-3). Needs a separate --out per readout, which this "
        "script now refuses to run without.",
    )
    # default=None rather than the constant so "given but without --flow-sampler" is detectable:
    # a half-typed command that silently scored the regression head into a flow-named --out dir
    # would be indistinguishable from the real thing months later.
    parser.add_argument(
        "--flow-steps",
        type=int,
        default=None,
        help=f"Euler steps for --flow-sampler (default: {DEFAULT_FLOW_STEPS})",
    )
    parser.add_argument(
        "--flow-seed",
        type=int,
        default=None,
        help=f"noise seed for --flow-sampler (default: {DEFAULT_FLOW_SEED})",
    )
    parser.add_argument(
        "--flow-mean-k",
        type=int,
        default=None,
        help="average k independent draws (seeds --flow-seed .. +k-1) per chunk. The MSE-fair "
        "arm of the T-30 rule: one draw scores worse than the conditional mean by exactly the "
        "conditional variance, so a single-draw MSE comparison against a mean-seeking regression "
        f"head is rigged against the sampler (default: {DEFAULT_FLOW_MEAN_K}, one draw)",
    )
    parser.add_argument(
        "--flow-t0",
        type=float,
        default=None,
        help="start the Euler integration at t0 from the regression chunk re-encoded and noised "
        "to t0, instead of at t=0 from pure noise. The conditioning-mismatch control of the T-30 "
        "rule: the velocity head was trained on features noised to the same t, and the sampler "
        f"feeds it t=1 features at every step (default: {DEFAULT_FLOW_T0}, from noise)",
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
    given = [
        name
        for name, value in (
            ("--flow-steps", args.flow_steps),
            ("--flow-seed", args.flow_seed),
            ("--flow-mean-k", args.flow_mean_k),
            ("--flow-t0", args.flow_t0),
        )
        if value is not None
    ]
    if not args.flow_sampler and given:
        raise SystemExit(
            f"{', '.join(given)}: the --flow-steps family only means something with "
            "--flow-sampler. Without it the regression head is scored, and an output dir named "
            "after a flow run would be indistinguishable from one."
        )
    flow_steps = None
    flow_seed = args.flow_seed if args.flow_seed is not None else DEFAULT_FLOW_SEED
    flow_mean_k = DEFAULT_FLOW_MEAN_K
    flow_t0 = DEFAULT_FLOW_T0
    if args.flow_sampler:
        flow_steps = args.flow_steps if args.flow_steps is not None else DEFAULT_FLOW_STEPS
        flow_mean_k = args.flow_mean_k if args.flow_mean_k is not None else DEFAULT_FLOW_MEAN_K
        flow_t0 = args.flow_t0 if args.flow_t0 is not None else DEFAULT_FLOW_T0
        # Refused here rather than downstream: JointCheckpointPolicy does reject flow_steps < 1,
        # but on the Wan path load_joint_policy has already spent minutes building the frozen
        # multi-GB base by the time its constructor runs — and the control arms are not checked
        # until the first predict() at all.
        if flow_steps < 1:
            raise SystemExit(f"--flow-steps must be >= 1, got {flow_steps}")
        if flow_mean_k < 1:
            raise SystemExit(f"--flow-mean-k must be >= 1, got {flow_mean_k}")
        if not 0.0 <= flow_t0 < 1.0:
            raise SystemExit(
                f"--flow-t0 must be in [0, 1), got {flow_t0}: t0=1 integrates zero steps and "
                "returns the warm start unchanged, which is the regression chunk laundered "
                "through action_recon and reported as a flow arm."
            )
    tag = readout_tag(
        frame_history=args.frame_history,
        flow_steps=flow_steps,
        flow_seed=flow_seed,
        flow_mean_k=flow_mean_k,
        flow_t0=flow_t0,
    )
    out_dir = args.out or args.run_dir
    # Before anything expensive: --out defaults to --run-dir and every artifact name is fixed,
    # so a second readout into the same directory silently overwrites the first one's A/B half.
    guard_out_dir(out_dir, tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_checkpoint(args.run_dir, args.checkpoint)
    print(f"checkpoint {model_path}")
    advise_alloc_conf(args.device)
    policy, config, metadata = build_policy(
        model_path,
        args.device,
        args.camera,
        args.backbone_source,
        offload_text=args.offload_text,
        flow_steps=flow_steps,
        flow_seed=flow_seed,
        flow_mean_k=flow_mean_k,
        flow_t0=flow_t0,
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
    # 2026-08-01 (one frame, tiled by predict()); ON feeds the window training actually used,
    # which is the in-distribution mode. Only t16-lora-seed0 has been scored both ways so far
    # (+10.65 pp on skill_vs_repeat_pct, still failing L1), so a run's frame mode belongs next to
    # its numbers wherever tiled and windowed results meet.
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
    if flow_steps is None:
        head_note = "single-shot regression (historical default)"
    else:
        arms = "".join(
            (
                f", mean of {flow_mean_k} draws" if flow_mean_k != DEFAULT_FLOW_MEAN_K else "",
                f", warm start at t0={flow_t0:g}" if flow_t0 != DEFAULT_FLOW_T0 else "",
            )
        )
        head_note = f"rectified-flow sampler, {flow_steps} steps, seed {flow_seed}{arms} (T-30)"
    print(f"scoring {len(pairs)} chunks over {len(holdout)} episodes | frames: {frames_note}")
    print(f"action head: {head_note}")

    # Wall time is the one number in this script that cannot be recovered from the archived
    # artifacts, and the T-30 A/B runs both readouts on one GPU — so the ms/chunk delta IS the
    # measured sampler cost, and collecting it costs a perf_counter call.
    started = time.perf_counter()
    predictions = evaluate_policy(policy, pairs)
    elapsed_s = time.perf_counter() - started
    ms_per_chunk = 1000.0 * elapsed_s / len(pairs)
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
    run_name = metadata.run_id + tag
    bench = bench_metrics(predictions, run_name=run_name)
    (out_dir / "bench.json").write_text(bench.to_json() + "\n")
    (out_dir / "bench.md").write_text(bench.render_markdown())

    # Separate file rather than a bench.json field: bench.json is re-derived from
    # predictions.jsonl by run_bench.py, and a timing recorded there would be silently dropped
    # on the first re-score — which is exactly when someone would trust it.
    (out_dir / "timing.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                # The tag verbatim, so the --out guard reads it back instead of reconstructing
                # it from four fields that a later arm could add a fifth to.
                "readout_tag": tag,
                "num_chunks": len(pairs),
                "seconds": elapsed_s,
                "ms_per_chunk": ms_per_chunk,
                "device": policy.device,
                # Next to "device" because it qualifies it: with this on, the umT5 encode ran on
                # the CPU even though everything else ran on `device`, and nothing here has
                # measured that the two agree bitwise. A bench.json scored with the flag would
                # otherwise be indistinguishable from one scored without it.
                "offload_text": bool(args.offload_text),
                "frame_history": bool(args.frame_history),
                "flow_steps": flow_steps,
                "flow_seed": flow_seed if flow_steps is not None else None,
                "flow_mean_k": flow_mean_k if flow_steps is not None else None,
                "flow_t0": flow_t0 if flow_steps is not None else None,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\nscored {len(pairs)} chunks in {elapsed_s:.1f}s ({ms_per_chunk:.1f} ms/chunk)")
    print(f"E1 action mse {e1.mse:.6g}")
    print(f"WAM-Bench {bench.level_name} — score {bench.score:.1f}/100")
    print(f"  vs zero-delta   {bench.skill_vs_zero_pct:+.1f}%")
    print(f"  vs repeat-last  {bench.skill_vs_repeat_pct:+.1f}%   <- the L1 bar (must be > 0)")
    for warning in bench.warnings:
        print(f"  warning: {warning}")
    print(f"\nwrote predictions.jsonl, e1.*, bench.*, timing.json to {out_dir}")
    if args.skip_split_check:
        (out_dir / "UNPROVEN_SPLIT").write_text(
            "Scored with --skip-split-check: the holdout was not proven unseen.\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
