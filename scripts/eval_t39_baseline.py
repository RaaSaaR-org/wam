#!/usr/bin/env python3
"""Score the T-39 positive control (PR-07) with WAM's own scorer, through WAM's own harness.

    scripts/eval_t39_baseline.py --run-dir runs/t39-baseline-seed0 --arm policy \
        --dataset datasets/gr00t-apple-full --raw-dataset data/raw/gr00t_apple \
        --holdout configs/splits/t18_holdout_episodes.txt \
        --train-episodes configs/splits/i8_train_362.txt --out <dir>

Fourteen recorded experiments in this repo are negatives, and every one of them compares a WAM
variant against a *trivial* baseline. None compares anything against a method known to work, so
none can separate "our approach is wrong" from "nothing clears this bar on this corpus, under this
scorer". This script scores the method that is supposed to work. The verdict rule is NOT here — it
is ``cluster/discoverer/71_eval_t39_control.sbatch``, committed before anything was submitted.

WHAT IS HELD IDENTICAL, and how. Not by copying values into this file: by calling the same
functions and reading the same committed files that produced every other number in
``docs/benchmark.md`` — ``eval_t16.verify_split`` for the split proof, ``build_eval_pairs`` for the
chunks, ``bench_metrics``/``e1_metrics`` for the score, ``configs/splits/*.txt`` for the episodes.
``evaluate_policy`` calls ``policy.predict(observation)`` and nothing else, which is the seam that
lets a foreign policy be scored by the identical harness. What necessarily differs is listed in
PR-07 §3 and is the price of the experiment, not an oversight.

THE FOUR ARMS, in the order ``71_*.sbatch`` runs them:

  oracle_state   The holdout's own future EXECUTED STATES pushed back through the adapter. The
                 label pipeline fed its own output: ``targets[t] = q[s+t+1] - q[s+t]`` is exactly
                 how ``convert_lerobot_g1.relabel_chunks`` built the targets, so a correct adapter
                 reproduces them to float32 and scores ~+100 %. Anything else is OUR bug — most
                 likely the delta anchor, the joint ordering or the gripper reduction — and
                 T39_RULE_V1 VOIDs the experiment rather than blaming the policy for it.

  oracle_action  The dataset's native 43-dim ``action`` column through the SAME adapter. This is
                 the arm worth the most, and it is not a formality. Our labels are relabeled from
                 executed STATE; GR00T is trained to predict the COMMANDED action. If the ground
                 truth GR00T predicts cannot clear L1 under our scorer, then no policy trained on
                 that column can, T-39 is unrunnable as designed, and every number in
                 ``docs/benchmark.md`` is bounded by a label-space mismatch nobody had measured.
                 It is a first-class finding, which is why it runs before the arm it can void.

  policy         The post-trained checkpoint. Its commanded output goes through
                 :func:`commanded_to_chunk` — the SAME function ``oracle_action`` uses — which is
                 what makes ``oracle_action`` a genuine ceiling for it rather than a separate
                 measurement that happens to be nearby.

  train40        ``--score-episodes``: the same policy on 40 episodes it DID train on. A
                 diagnostic and an upper bound, never a headline; T39_RULE_V1 uses it only to
                 separate "cannot fit" from "cannot generalise". The split proof still runs in
                 full first, because the point of the arm is that these episodes were trained on
                 — which is a claim about the witness and has to be *proven*, not assumed.

THE VENDORED POLICY IS BEHIND A NAMED CONTRACT (``--policy-entrypoint``), for the same reason
``MODEL_ID`` has no default in ``70_train_t39_baseline.sbatch``: the vendored trainer's inference
API has not been verified from a primary source, and a guess baked in here would end up in an
artifact as a fact. The contract this script requires is small, stated in
:func:`load_commanded_policy`, and any real API is a few lines of shim away from it inside the
t39 venv. What the shim may NOT do is convert to canonical units — that is this file's job, done
once, shared with the oracle.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

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
from wam.interfaces import ActionChunk, ActionMode, Observation
from wam.interfaces.versioning import RunMetadata

ARMS = ("oracle_state", "oracle_action", "policy")

BENCH_SPECS_WRITTEN = ("0.1.0", "0.2.0")
"""Both WAM-Bench specs, always. ``bench.json`` is 0.1.0 — the spec every archived run in this
repo was scored under and the one T39_RULE_V1 reads — and ``bench_0.2.0.json`` is written beside
it so the two-sided L4 band is on record without a re-score. Writing both costs one function call
over predictions already in memory; deciding later which one to report would not."""

WITNESS_FILENAME = "run_metadata.json"


def _load_script(name: str) -> Any:
    """Import a sibling script as a module.

    ``scripts/`` is not a package and these files carry ``if __name__ == "__main__"`` mains, so
    this is how one script reuses another's functions. It is load-bearing rather than convenient:
    PR-07 §3 holds the split proof and the label mapping identical *by calling the same code*, and
    a copy of ``relabel_chunks``'s anchoring in this file would be a second definition free to
    drift from the one that actually built the dataset.
    """
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_t39_{name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - unreachable for a real file
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ the label mapping, shared


class GripperMapping:
    """How a 43-dim source vector's hand joints become the canonical gripper scalar.

    Read from the converted dataset's manifest, never re-fitted here. The ``active-hand`` affine
    is a property of the conversion SET (``convert_lerobot_g1``'s module docstring: 30 / 120 / 402
    episodes give three different affines for the same physical aperture), so re-deriving it from
    whatever episodes this eval happens to touch would put the oracle arms on a different scale
    than the targets they are scored against — and the discrepancy would look like an adapter bug.
    """

    def __init__(self, *, affine: Any, column: int | None) -> None:
        self.affine = affine
        self.column = column

    @property
    def kind(self) -> str:
        return "legacy" if self.affine is None else "active-hand"

    def reduce(self, gripper_pair: np.ndarray) -> np.ndarray:
        """[..., 2] canonical per-hand synergies -> [...] the single scored channel.

        Mirrors ``relabel_chunks``: the active hand's column when the conversion fitted an affine,
        the both-hand mean under the legacy mapping. The mean is the second half of the T-31 bug
        and is reproduced here only because archived legacy datasets were built with it.
        """
        arr = np.asarray(gripper_pair, dtype=np.float32)
        if self.column is None:
            return arr.mean(axis=-1).astype(np.float32)
        return arr[..., self.column].astype(np.float32)


def gripper_mapping_from_manifest(manifest: Any, convert: Any) -> GripperMapping:
    """The conversion's own gripper mapping, reconstructed from what it recorded.

    Refuses rather than guesses. A manifest that says ``active-hand`` but carries no
    normalization block cannot be reproduced, and silently falling back to the legacy mapping
    would rescale the whole gripper channel — an oracle arm scored under the wrong mapping fails
    ``ORACLE_STATE_FLOOR_PCT`` and would be read as a delta-anchor bug, which is the one diagnosis
    this arm exists to make unambiguous.
    """
    extra = dict(getattr(manifest, "extra", None) or {})
    mapping = dict(extra.get("mapping") or {})
    kind = str(mapping.get("gripper_mapping", "legacy"))
    if kind == "legacy":
        return GripperMapping(affine=None, column=None)
    if kind != "active-hand":
        raise SystemExit(
            f"unknown gripper_mapping {kind!r} in the manifest — this script reproduces the two "
            "mappings scripts/convert_lerobot_g1.py can write, and cannot reproduce a third."
        )

    # normalization_specs(), not manifest.normalization: the manifest stores plain dicts and the
    # parsed NormalizationSpec is the shape HandAffine.to_spec wrote, so round-tripping through
    # the parser is what guarantees this reads back the affine the conversion recorded.
    specs = manifest.normalization_specs() or {}
    spec = specs.get("gripper_target")
    if spec is None:
        raise SystemExit(
            "manifest declares the active-hand gripper mapping but records no "
            "normalization['gripper_target'] affine. The affine is dataset-level and cannot be "
            "re-derived from these episodes without putting the oracle arms on a different "
            "scale than the targets they are scored against — see PR-07 §4."
        )
    mean = tuple(getattr(spec, "mean", ()) or ())
    std = tuple(getattr(spec, "std", ()) or ())
    if not mean or not std:
        raise SystemExit("normalization['gripper_target'] carries no mean/std — cannot rebuild")

    target_note = str(mapping.get("gripper_target", ""))
    if target_note.startswith("left"):
        column = 0
    elif target_note.startswith("right"):
        column = 1
    else:
        raise SystemExit(
            f"manifest does not say which hand the target channel is ({target_note!r}). Guessing "
            "between the two would silently score a frozen hand against a live one."
        )
    affine = convert.HandAffine(
        active="left" if column == 0 else "right",
        offset=float(mean[0]),
        span=float(std[0]),
        p2p_left=0.0,
        p2p_right=0.0,
    )
    return GripperMapping(affine=affine, column=column)


def commanded_to_chunk(
    commanded: np.ndarray,
    anchor_state: np.ndarray,
    *,
    dt_s: float,
    mapping: GripperMapping,
    convert: Any,
) -> ActionChunk:
    """[T, 43] COMMANDED absolute source positions + the [43] state they start from -> a chunk.

    This is the adapter, and the anchoring is the whole of it::

        targets[t]        = canonical_q(commanded[t]) - canonical_q(state_at_step(t))
        gripper_target[t] = gripper channel of commanded[t]

    with ``state_at_step(0) = anchor_state`` and ``state_at_step(t>0)`` the position the previous
    command asked for. Why that and not something else:

    ``convert_lerobot_g1.relabel_chunks`` builds the target as ``q[s+t+1] - q[s+t]`` — the
    displacement the robot EXECUTED over step ``s+t``. The source ``action`` column holds the
    absolute position the controller was TOLD to reach at step ``s+t``. Under position control
    those are the same quantity measured at two points of one loop: the command issued at ``s+t``
    is what produces the state at ``s+t+1``. So the commanded displacement over step ``s+t`` is
    ``action[s+t] - q[s+t]``, and a dataset with perfect tracking (``action[i] == q[i+1]``) makes
    the two identical. ``tests/test_t39_baseline.py`` pins exactly that, and kills the three
    plausible mis-anchorings — ``action[t+1] - q[t]``, ``action[t] - q[t+1]`` and the
    first-difference ``action[t+1] - action[t]`` — which all produce finite, plausible, wrong
    numbers that no shape or range assertion catches. That is T-37's transposed-``xmat`` lesson,
    and it is the reason the mutation test exists rather than an argument.

    Steps after the first chain through the commands rather than through unavailable future
    states: a policy emits an open-loop chunk, so by construction nothing observed the state at
    ``s+t`` for ``t > 0``. Chaining is what an open-loop chunk MEANS, and it is also what makes
    the sum of the chunk's deltas equal the total commanded displacement.
    """
    commanded = np.asarray(commanded, dtype=np.float32)
    if commanded.ndim != 2 or commanded.shape[1] != convert.SOURCE_STATE_DIM:
        raise SystemExit(
            f"commanded actions must be [T, {convert.SOURCE_STATE_DIM}], got {commanded.shape}"
        )
    anchor = np.asarray(anchor_state, dtype=np.float32).reshape(-1)
    if anchor.shape[0] != convert.SOURCE_STATE_DIM:
        raise SystemExit(f"anchor state must be [{convert.SOURCE_STATE_DIM}], got {anchor.shape}")

    q_cmd = convert.canonical_q(commanded)  # [T, 15]
    # The position each step starts from: the anchor, then what the previous command asked for.
    q_from = np.concatenate([convert.canonical_q(anchor)[None, :], q_cmd[:-1]], axis=0)
    targets = (q_cmd - q_from).astype(np.float32)

    grip_pair = convert.gripper_state(commanded, mapping.affine)  # [T, 2]
    gripper_target = mapping.reduce(grip_pair)
    return ActionChunk(
        mode=ActionMode.JOINT_DELTA,
        targets=targets,
        gripper_target=np.asarray(gripper_target, dtype=np.float32),
        dt_s=float(dt_s),
    )


# ------------------------------------------------------------------------------------ arms


class ChunkLookupPolicy:
    """A ``Policy`` that answers from a precomputed ``{t_ns: chunk}`` table for ONE episode.

    Scoped to one episode deliberately. ``Observation`` carries no episode id and LeRobot
    timestamps restart at zero per episode, so a table keyed by ``t_ns`` across the whole holdout
    would collide on the first chunk of all 40 episodes and quietly answer with the wrong one.
    Building one lookup per episode and concatenating the predictions is identical to a single
    pass and cannot alias.
    """

    def __init__(self, chunks: dict[int, ActionChunk], *, episode_id: str) -> None:
        self._chunks = chunks
        self._episode_id = episode_id

    def predict(self, observation: Observation) -> ActionChunk:
        t_ns = int(observation.state.timestamp_ns)
        chunk = self._chunks.get(t_ns)
        if chunk is None:
            raise SystemExit(
                f"{self._episode_id}: no oracle chunk anchored at t_ns={t_ns}. The eval pairs are "
                "anchored on recorded chunk timestamps, so a miss means the states and the "
                "actions in this episode do not share a clock — the oracle would silently score "
                "a neighbouring chunk."
            )
        return chunk


class CommandedPolicy:
    """Wraps the vendored model so that only its COMMANDED output crosses into our units.

    The vendored callable returns source-space absolute positions and knows nothing about
    canonical joints, gripper synergies or delta anchoring; :func:`commanded_to_chunk` does all of
    that, once, and ``oracle_action`` calls the same function on the ground-truth column. That
    shared call is what makes the oracle a ceiling for this policy instead of a separate number
    that happens to sit nearby.
    """

    def __init__(
        self,
        infer: Any,
        *,
        camera: str,
        chunk_steps: int,
        dt_s: float,
        mapping: GripperMapping,
        convert: Any,
        raw_states: dict[str, np.ndarray],
        anchors: dict[tuple[str, int], int],
        episode_id: str,
    ) -> None:
        self._infer = infer
        self._camera = camera
        self._chunk_steps = chunk_steps
        self._dt_s = dt_s
        self._mapping = mapping
        self._convert = convert
        self._raw_states = raw_states
        self._anchors = anchors
        self._episode_id = episode_id

    def predict(self, observation: Observation) -> ActionChunk:
        images = observation.image_history or {}
        video = images.get(self._camera)
        if video is None:
            video = np.asarray(observation.images[self._camera])[None, ...]
        t_ns = int(observation.state.timestamp_ns)
        index = self._anchors.get((self._episode_id, t_ns))
        if index is None:
            raise SystemExit(
                f"{self._episode_id}: no raw state index for t_ns={t_ns} — the converted episode "
                "and the raw parquet do not share a clock, so the policy would be anchored on the "
                "wrong step."
            )
        anchor = self._raw_states[self._episode_id][index]
        commanded = np.asarray(
            self._infer(
                {
                    "video": np.asarray(video),
                    "state": np.asarray(anchor, dtype=np.float32),
                    "instruction": observation.instruction,
                }
            ),
            dtype=np.float32,
        )
        if commanded.ndim != 2:
            raise SystemExit(
                f"the policy returned {commanded.shape}; the contract is [horizon, "
                f"{self._convert.SOURCE_STATE_DIM}] absolute source-space positions"
            )
        if commanded.shape[0] < self._chunk_steps:
            raise SystemExit(
                f"the policy returned a {commanded.shape[0]}-step horizon and the bar is scored "
                f"over {self._chunk_steps} steps. Padding the tail would score invented steps as "
                "predictions; this is a mismatch to fix in the shim, not to absorb here."
            )
        return commanded_to_chunk(
            commanded[: self._chunk_steps],
            anchor,
            dt_s=self._dt_s,
            mapping=self._mapping,
            convert=self._convert,
        )


def load_commanded_policy(entrypoint: str, model_dir: Path, device: str, vendor_root: Path | None):
    """Import ``module:factory`` and build the vendored inference callable.

    THE CONTRACT, in full::

        factory(model_dir: Path, device: str) -> callable
        callable({"video": uint8 [T, H, W, 3], "state": float32 [43], "instruction": str})
            -> float32 [horizon, 43]   ABSOLUTE positions in the source 43-dim layout

    Absolute source-space positions, not deltas and not canonical joints. Every conversion into
    WAM's units happens in :func:`commanded_to_chunk`, so the shim cannot accidentally apply a
    second one and the oracle arm exercises the identical path. ``horizon`` may exceed the scored
    chunk length and is truncated; shorter is refused.

    No default entrypoint, for the same reason ``MODEL_ID`` has none: the vendored inference API
    has not been verified from a primary source (PR-07 §8), and a plausible guess here would run,
    produce numbers, and put an unverified assumption into an artifact as a fact.
    """
    if ":" not in entrypoint:
        raise SystemExit(f"--policy-entrypoint must be 'module.path:factory', got {entrypoint!r}")
    module_name, _, factory_name = entrypoint.partition(":")
    if vendor_root is not None:
        sys.path.insert(0, str(vendor_root))
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"cannot import {module_name!r} ({exc}). This runs in $PROJ/virt_envs/t39, not the "
            "WAM venv — see PR-07 §8 item 4."
        ) from exc
    factory = getattr(module, factory_name, None)
    if factory is None:
        raise SystemExit(f"{module_name!r} has no attribute {factory_name!r}")
    return factory(model_dir, device)


# ------------------------------------------------------------------------------- episode work


def episode_chunk_steps(reader: Any) -> int:
    """The scored chunk length, taken from the dataset instead of from a flag.

    ``build_eval_pairs`` needs a length and the dataset already fixes one: the conversion wrote
    non-overlapping chunks of ``--chunk-steps``. Reading it back means the oracle arms cannot be
    scored over a different horizon than the targets were built with, which a CLI default would
    eventually allow.
    """
    actions = reader.read_actions()
    if not actions:
        raise SystemExit(f"{reader.manifest.episode_id}: no recorded action chunks")
    lengths = {int(chunk.num_steps) for chunk, _, _ in actions}
    if len(lengths) != 1:
        raise SystemExit(
            f"{reader.manifest.episode_id}: mixed chunk lengths {sorted(lengths)} — the scored "
            "horizon would depend on which chunks happen to be in the holdout"
        )
    return lengths.pop()


def episode_dt_s(reader: Any) -> float:
    """The episode's step period, read off its own recorded chunks.

    ``EpisodeManifest`` carries no fps field — the writer turns ``fps`` into per-sample
    timestamps — so the authoritative period is the one stored on the chunks themselves. Taking
    it from there rather than from a CLI default keeps a policy chunk on the same clock as the
    target it is differenced against.
    """
    periods = {round(float(chunk.dt_s), 9) for chunk, _, _ in reader.read_actions()}
    if len(periods) != 1:
        raise SystemExit(
            f"{reader.manifest.episode_id}: mixed chunk dt_s {sorted(periods)} — this episode "
            "does not have one clock"
        )
    return periods.pop()


def oracle_state_chunks(reader: Any, chunk_steps: int, mapping: GripperMapping) -> dict:
    """``{t_ns: chunk}`` rebuilt from the episode's own recorded states.

    Identity by construction: ``relabel_chunks`` built the stored targets as
    ``q[s+t+1] - q[s+t]`` over exactly these states, so a correct rebuild returns them bit for
    bit. That is what makes this a plumbing check with a known answer rather than a measurement —
    ``ORACLE_STATE_FLOOR_PCT`` is set at 90 % against a true value of ~100 %.
    """
    states = reader.read_states()
    ts = np.asarray([s.timestamp_ns for s in states], dtype=np.int64)
    q = np.stack([np.asarray(s.q, dtype=np.float32) for s in states])
    grip = np.stack([np.asarray(s.gripper_state, dtype=np.float32) for s in states])
    reduced = mapping.reduce(grip)

    chunks: dict[int, ActionChunk] = {}
    for _chunk, _prefix, t_ns in reader.read_actions():
        index = int(np.searchsorted(ts, t_ns))
        if index >= ts.shape[0] or int(ts[index]) != int(t_ns):
            raise SystemExit(
                f"{reader.manifest.episode_id}: chunk at t_ns={t_ns} has no state with the same "
                "timestamp. The conversion anchors every chunk on a state timestamp, so this "
                "dataset was not written by scripts/convert_lerobot_g1.py."
            )
        if index + chunk_steps >= q.shape[0]:
            continue
        chunks[int(t_ns)] = ActionChunk(
            mode=ActionMode.JOINT_DELTA,
            targets=np.diff(q[index : index + chunk_steps + 1], axis=0).astype(np.float32),
            gripper_target=reduced[index + 1 : index + chunk_steps + 1].astype(np.float32),
            dt_s=float(_chunk.dt_s),
        )
    return chunks


def raw_episode_index(episode_id: str) -> int:
    """``gr00t-apple-000020`` -> ``20``, refusing anything that is not that shape."""
    tail = episode_id.rsplit("-", 1)[-1]
    if not tail.isdigit():
        raise SystemExit(
            f"cannot map {episode_id!r} to a LeRobot episode index. The converted ids end in the "
            "zero-padded source index (scripts/convert_lerobot_g1.py); an id that does not is "
            "from a different conversion and must not be paired with this raw dataset."
        )
    return int(tail)


def read_raw_episode(raw_root: Path, episode_id: str) -> dict[str, np.ndarray]:
    """``{state [n,43], action [n,43], ts_ns [n]}`` for one source episode.

    The ``action`` column is read HERE and nowhere else in this repo: ``convert_lerobot_g1``
    explicitly does not use it (its module docstring says so), which is precisely why
    ``oracle_action`` had to be built to find out whether it can clear our bar at all.
    """
    import pyarrow.parquet as pq

    index = raw_episode_index(episode_id)
    path = raw_root / "data" / "chunk-000" / f"episode_{index:06d}.parquet"
    if not path.is_file():
        raise SystemExit(f"{path} missing — --raw-dataset is not the source of {episode_id}")
    table = pq.read_table(path, columns=["observation.state", "action", "timestamp"])
    state = np.stack(table["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float32)
    action = np.stack(table["action"].to_numpy(zero_copy_only=False)).astype(np.float32)
    ts_s = table["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
    if state.shape != action.shape:
        raise SystemExit(f"{path}: state {state.shape} and action {action.shape} disagree")
    return {
        "state": state,
        "action": action,
        "ts_ns": np.round(ts_s * 1e9).astype(np.int64),
    }


def raw_anchor_indices(reader: Any, raw: dict[str, np.ndarray]) -> dict[int, int]:
    """``{chunk t_ns: index into the raw arrays}``, by exact timestamp match.

    Exact, never nearest. The conversion rounds source seconds to nanoseconds the same way this
    does, so an inexact match means the two files are not the same recording — and a nearest-match
    fallback would score the oracle one step out and report it as a label-space mismatch, which is
    the finding this arm is supposed to be able to make.
    """
    ts = np.asarray(raw["ts_ns"], dtype=np.int64)
    anchors: dict[int, int] = {}
    for _chunk, _prefix, t_ns in reader.read_actions():
        index = int(np.searchsorted(ts, t_ns))
        if index >= ts.shape[0] or int(ts[index]) != int(t_ns):
            raise SystemExit(
                f"{reader.manifest.episode_id}: chunk t_ns={t_ns} is not a timestamp of "
                "--raw-dataset. The converted episode and the parquet are not the same recording."
            )
        anchors[int(t_ns)] = index
    return anchors


def oracle_action_chunks(
    reader: Any,
    raw: dict[str, np.ndarray],
    chunk_steps: int,
    mapping: GripperMapping,
    convert: Any,
) -> dict:
    """``{t_ns: chunk}`` built from the source ``action`` column via :func:`commanded_to_chunk`."""
    anchors = raw_anchor_indices(reader, raw)
    action = np.asarray(raw["action"], dtype=np.float32)
    state = np.asarray(raw["state"], dtype=np.float32)
    chunks: dict[int, ActionChunk] = {}
    for chunk, _prefix, t_ns in reader.read_actions():
        index = anchors[int(t_ns)]
        if index + chunk_steps > action.shape[0]:
            continue
        chunks[int(t_ns)] = commanded_to_chunk(
            action[index : index + chunk_steps],
            state[index],
            dt_s=float(chunk.dt_s),
            mapping=mapping,
            convert=convert,
        )
    return chunks


# ------------------------------------------------------------------------------------- main


def load_witness(run_dir: Path) -> RunMetadata:
    """The training run's ``run_metadata.json``, or refuse.

    ``eval_t16.py`` reads this out of the checkpoint header because WAM writes it there. The
    vendored trainer writes its own checkpoint format and knows nothing about ``RunMetadata``, so
    ``train_t39_baseline.py`` writes the witness beside it — same fields, same meaning, checked
    the same way by the same ``verify_split``.
    """
    path = run_dir / WITNESS_FILENAME
    if not path.is_file():
        raise SystemExit(
            f"{path} missing. Without the witness the holdout cannot be PROVEN unseen, and an "
            "unproven holdout is not a cheaper measurement — it is not one (PR-07 §3)."
        )
    return RunMetadata.model_validate(json.loads(path.read_text()))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="the CONVERTED WAM dataset")
    parser.add_argument("--raw-dataset", type=Path, help="the LeRobot source (oracle_action only)")
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument(
        "--train-episodes",
        type=Path,
        required=True,
        help="the committed split file the run trained on — the external witness verify_split "
        "needs on the disjointness path",
    )
    parser.add_argument(
        "--score-episodes",
        type=Path,
        help="score these instead of the holdout (the train40 DIAGNOSTIC). Must be a subset of "
        "the recorded training set: an in-distribution arm that is not in distribution is just "
        "a second holdout wearing the wrong label.",
    )
    parser.add_argument("--camera", default="ego")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-steps", type=int, help="override the dataset's chunk length")
    parser.add_argument("--policy-entrypoint", help="'module.path:factory' — see the docstring")
    parser.add_argument("--model-dir", type=Path, help="staged weights for the policy arm")
    parser.add_argument("--vendor-root", type=Path, help="prepended to sys.path for the shim")
    parser.add_argument(
        "--frame-history",
        action="store_true",
        help="fill Observation.image_history with the real window (T-29). Off by default so the "
        "arm reproduces how every archived number in this repo was measured.",
    )
    parser.add_argument("--num-frames", type=int, default=9, help="window size for --frame-history")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    eval_t16 = _load_script("eval_t16")
    convert = _load_script("convert_lerobot_g1")

    if args.arm == "oracle_action" and args.raw_dataset is None:
        raise SystemExit("--arm oracle_action needs --raw-dataset: the action column lives there")
    if args.arm == "policy":
        if args.policy_entrypoint is None or args.model_dir is None:
            raise SystemExit("--arm policy needs --policy-entrypoint and --model-dir")
        if args.raw_dataset is None:
            raise SystemExit(
                "--arm policy needs --raw-dataset: the vendored model consumes the source 43-dim "
                "state, and reconstructing it from the canonical 15 would invent the 28 joints "
                "the conversion dropped."
            )

    witness = load_witness(args.run_dir)
    holdout_ids = load_episode_ids(args.holdout)
    witness_ids = load_episode_ids(args.train_episodes)

    # ALWAYS, including for the train40 diagnostic. The diagnostic's whole claim is that these
    # episodes WERE trained on, which is a statement about the witness — so the witness has to be
    # proven sound before the arm is allowed to lean on it.
    holdout_dirs = eval_t16.verify_split(
        args.dataset,
        holdout_ids,
        witness.dataset_snapshot_ref,
        witness.train_episode_ids,
        witness_ids,
    )

    in_distribution = args.score_episodes is not None
    if in_distribution:
        scored_ids = load_episode_ids(args.score_episodes)
        trained = set(witness.train_episode_ids or ())
        stray = sorted(scored_ids - trained)
        if stray:
            raise SystemExit(
                f"REFUSING TO SCORE — --score-episodes names {len(stray)} episode(s) the run did "
                f"not train on: {stray[:5]}{'...' if len(stray) > 5 else ''}. This arm is the "
                "IN-DISTRIBUTION diagnostic and T39_RULE_V1 reads it as an upper bound; episodes "
                "outside the training set would make it a second holdout under a name that says "
                "the opposite."
            )
        from wam.data.episode import list_episodes

        scored_dirs = [p for p in list_episodes(args.dataset) if p.name in scored_ids]
        if not scored_dirs:
            raise SystemExit(f"--score-episodes matched nothing under {args.dataset}")
    else:
        scored_dirs = holdout_dirs

    args.out.mkdir(parents=True, exist_ok=True)

    from wam.data.episode import EpisodeReader

    first = EpisodeReader(scored_dirs[0])
    chunk_steps = args.chunk_steps or episode_chunk_steps(first)
    mapping = gripper_mapping_from_manifest(first.manifest, convert)
    num_frames = args.num_frames if args.frame_history else None
    print(
        f"arm {args.arm} | {len(scored_dirs)} episode(s) | chunk_steps {chunk_steps} | "
        f"gripper {mapping.kind} | frames "
        f"{'real ' + str(num_frames) + '-window' if num_frames else 'single (tiled by the policy)'}"
    )
    if in_distribution:
        print("=== IN-DISTRIBUTION DIAGNOSTIC — these episodes were TRAINED ON, never a headline")

    infer = None
    if args.arm == "policy":
        infer = load_commanded_policy(
            args.policy_entrypoint, args.model_dir, args.device, args.vendor_root
        )

    predictions = []
    started = time.perf_counter()
    for episode_dir in scored_dirs:
        reader = EpisodeReader(episode_dir)
        episode_id = reader.manifest.episode_id
        pairs = build_eval_pairs(episode_dir, args.camera, chunk_steps, num_frames=num_frames)
        if not pairs:
            continue
        if args.arm == "oracle_state":
            policy: Any = ChunkLookupPolicy(
                oracle_state_chunks(reader, chunk_steps, mapping), episode_id=episode_id
            )
        elif args.arm == "oracle_action":
            raw = read_raw_episode(args.raw_dataset, episode_id)
            policy = ChunkLookupPolicy(
                oracle_action_chunks(reader, raw, chunk_steps, mapping, convert),
                episode_id=episode_id,
            )
        else:
            raw = read_raw_episode(args.raw_dataset, episode_id)
            policy = CommandedPolicy(
                infer,
                camera=args.camera,
                chunk_steps=chunk_steps,
                dt_s=episode_dt_s(reader),
                mapping=mapping,
                convert=convert,
                raw_states={episode_id: raw["state"]},
                anchors={
                    (episode_id, t_ns): index
                    for t_ns, index in raw_anchor_indices(reader, raw).items()
                },
                episode_id=episode_id,
            )
        predictions.extend(evaluate_policy(policy, pairs))
    elapsed_s = time.perf_counter() - started

    if not predictions:
        raise SystemExit(f"no eval chunks built from {len(scored_dirs)} episode(s)")
    save_predictions_jsonl(predictions, args.out / "predictions.jsonl")

    spec = first.manifest.spec
    e1: E1Report = e1_metrics(predictions, spec)
    (args.out / "e1.json").write_text(e1.to_json() + "\n")
    (args.out / "e1.md").write_text(e1.render_markdown())

    run_name = f"{witness.run_id}+t39-{args.arm}" + ("+train40" if in_distribution else "")
    bench = None
    for spec_version in BENCH_SPECS_WRITTEN:
        report = bench_metrics(predictions, run_name=run_name, spec_version=spec_version)
        name = "bench" if spec_version == BENCH_SPECS_WRITTEN[0] else f"bench_{spec_version}"
        (args.out / f"{name}.json").write_text(report.to_json() + "\n")
        (args.out / f"{name}.md").write_text(report.render_markdown())
        if bench is None:
            bench = report

    (args.out / "timing.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "arm": args.arm,
                "in_distribution": in_distribution,
                "num_chunks": len(predictions),
                "seconds": elapsed_s,
                "ms_per_chunk": 1000.0 * elapsed_s / len(predictions),
                "device": args.device,
                "frame_history": bool(args.frame_history),
                "num_frames": num_frames,
                "chunk_steps": chunk_steps,
                "gripper_mapping": mapping.kind,
                "policy_entrypoint": args.policy_entrypoint,
                "model_dir": str(args.model_dir) if args.model_dir else None,
            },
            indent=2,
        )
        + "\n"
    )
    if in_distribution:
        (args.out / "IN_DISTRIBUTION").write_text(
            "These episodes were TRAINED ON. T39_RULE_V1 reads this arm as an upper bound and a\n"
            "diagnostic only. It is never a headline (PR-07 §5).\n"
        )

    assert bench is not None
    print(f"\nscored {len(predictions)} chunks in {elapsed_s:.1f}s")
    print(f"E1 action mse {e1.mse:.6g}")
    print(f"WAM-Bench {bench.level_name} — score {bench.score:.1f}/100")
    print(f"  vs zero-delta   {bench.skill_vs_zero_pct:+.2f}%")
    print(f"  vs repeat-last  {bench.skill_vs_repeat_pct:+.2f}%   <- L1")
    print(f"  critical subset {bench.ci_skill_vs_repeat_pct:+.2f}%   <- L2")
    for warning in bench.warnings:
        print(f"  warning: {warning}")
    print(f"\nwrote predictions.jsonl, e1.*, bench.*, timing.json to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
