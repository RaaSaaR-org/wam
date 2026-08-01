"""Offline evaluation E1 (T-14): action-prediction metrics on holdout episodes.

Contracts:
- Operates on PREDICTIONS (``ChunkPrediction`` pairs), fully decoupled from model internals:
  any object implementing the ``Policy`` protocol can be evaluated via ``evaluate_policy``.
- Torch-free by design (numpy only); metric math runs in float64.
- Action-MSE/MAE are DIAGNOSTIC metrics (PRD 10.4): they gate offline iteration (M2/M3), the
  real acceptance metrics are closed-loop success/safety (E3/E4).
- Serialization: one prediction per JSONL line, see ``prediction_to_dict`` for the format.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from wam.interfaces import (
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    Observation,
    Policy,
)

EVAL_VERSION = "0.1.0"

GRIPPER_BINARIZE_THRESHOLD = 0.5
"""Gripper accuracy binarization: values >= threshold count as 'closed' (gripper is in [0,1])."""


@dataclass
class ChunkPrediction:
    """One evaluated sample: predicted vs. demonstrated (target) chunk at episode time t_ns."""

    predicted: ActionChunk
    target: ActionChunk
    episode_id: str
    t_ns: int


class EpisodeMetrics(BaseModel):
    """Per-episode aggregate over all chunk predictions of that episode."""

    model_config = ConfigDict(frozen=True)

    num_chunks: int = Field(ge=1)
    mse: float
    mae: float
    gripper_accuracy: float


class E1Report(BaseModel):
    """E1 offline-replay metrics report (T-14 dashboard payload).

    - ``mse``/``mae``: element-wise error over all target dims, steps and predictions.
    - ``per_joint_*``: keyed by canonical joint name when a spec is given and the action mode is
      JOINT_DELTA, else by generic ``dim_<i>`` labels.
    - ``per_step_*``: error per horizon step (index 0 = first step) — shows error growth over
      the chunk. With variable chunk lengths each step averages the predictions that reach it.
    - ``gripper_accuracy``: fraction of steps where predicted and target gripper agree after
      binarization at ``GRIPPER_BINARIZE_THRESHOLD``. Only readable against the two measurements
      below — on a channel that never opens it is the majority-class rate and nothing else.
    - ``gripper_dynamic_range``: peak-to-peak of the DEMONSTRATED gripper signal.
    - ``gripper_majority_pct``: accuracy a constant predictor of the demonstrated majority class
      would score. Both default to 0.0 so archived ``runs/*/e1*.json`` still parses; no threshold
      lives here, the admissibility gate has exactly one home (``wam.evaluation.gripper``).
    - ``smoothness_*``: mean squared second temporal difference of the targets (lower =
      smoother); predicted vs. target shows whether the policy is jerkier than the demos.
      0.0 when no chunk has >= 3 steps.
    """

    model_config = ConfigDict(frozen=True)

    report_version: str = EVAL_VERSION
    mode: str
    num_predictions: int = Field(ge=1)
    num_episodes: int = Field(ge=1)
    horizon_steps: int = Field(ge=1)
    target_dim: int = Field(ge=1)
    mse: float
    mae: float
    per_joint_mse: dict[str, float]
    per_joint_mae: dict[str, float]
    per_step_mse: tuple[float, ...]
    per_step_mae: tuple[float, ...]
    gripper_accuracy: float
    gripper_dynamic_range: float = 0.0
    gripper_majority_pct: float = 0.0
    smoothness_pred: float
    smoothness_target: float
    per_episode: dict[str, EpisodeMetrics]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> E1Report:
        return cls.model_validate_json(text)

    def render_markdown(self) -> str:
        """Human-readable metrics dashboard (one table per metric group)."""
        lines = [
            "# E1 offline evaluation",
            "",
            f"- predictions: {self.num_predictions} chunks over {self.num_episodes} episode(s)",
            (
                f"- action mode: {self.mode} | target dim: {self.target_dim} | "
                f"horizon: {self.horizon_steps} steps"
            ),
            "",
            "## Overall",
            "",
            "| metric | value |",
            "| --- | --- |",
            f"| MSE | {_fmt(self.mse)} |",
            f"| MAE | {_fmt(self.mae)} |",
            f"| gripper accuracy | {self._gripper_cell(self.gripper_accuracy)} |",
            f"| gripper dynamic range | {_fmt(self.gripper_dynamic_range)} |",
            f"| gripper majority-class baseline | {self.gripper_majority_pct:.1f}% |",
            f"| smoothness (predicted) | {_fmt(self.smoothness_pred)} |",
            f"| smoothness (target) | {_fmt(self.smoothness_target)} |",
            "",
            "## Per joint",
            "",
            "| joint | MSE | MAE |",
            "| --- | --- | --- |",
        ]
        for name in self.per_joint_mse:
            lines.append(
                f"| {name} | {_fmt(self.per_joint_mse[name])} | {_fmt(self.per_joint_mae[name])} |"
            )
        lines += [
            "",
            "## Per horizon step (error growth)",
            "",
            "| step | MSE | MAE |",
            "| --- | --- | --- |",
        ]
        for i, (s_mse, s_mae) in enumerate(zip(self.per_step_mse, self.per_step_mae)):
            lines.append(f"| {i} | {_fmt(s_mse)} | {_fmt(s_mae)} |")
        lines += [
            "",
            "## Per episode",
            "",
            "| episode | chunks | MSE | MAE | gripper accuracy |",
            "| --- | --- | --- | --- | --- |",
        ]
        for ep_id, ep in self.per_episode.items():
            lines.append(
                f"| {ep_id} | {ep.num_chunks} | {_fmt(ep.mse)} | {_fmt(ep.mae)} | "
                f"{self._gripper_cell(ep.gripper_accuracy)} |"
            )
        return "\n".join(lines) + "\n"

    @property
    def gripper_withheld_reason(self) -> str:
        """Why ``gripper_accuracy`` is not a number worth rendering, or "" if it is.

        The SAME admissibility rule ``bench.json`` applies, evaluated here so the two artifacts
        of one run cannot disagree. They did: ``bench.json`` returned ``gripper_accuracy=None``
        with a reason while ``e1.md`` rendered ``| gripper accuracy | 0.89351 |`` for the same
        predictions, because T-27 added the diagnostics NEXT to the number instead of in front
        of it — which is the exact failure the withholding change was made to fix.
        """
        from wam.evaluation.gripper import GRIPPER_MIN_DYNAMIC_RANGE

        if self.gripper_dynamic_range < GRIPPER_MIN_DYNAMIC_RANGE:
            return (
                f"gripper_dynamic_range {self.gripper_dynamic_range:.3f} < "
                f"{GRIPPER_MIN_DYNAMIC_RANGE}; majority-class baseline "
                f"{self.gripper_majority_pct:.1f}%"
            )
        return ""

    def _gripper_cell(self, value: float) -> str:
        """A withheld number must not be renderable as a number — including per episode.

        Per episode especially: a reader scanning that column for "which episodes did the
        gripper work on" is asking precisely the question a majority-class rate cannot answer.
        """
        reason = self.gripper_withheld_reason
        return f"n/a — withheld ({reason})" if reason else _fmt(value)


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _dim_labels(mode: str, dim: int, spec: CanonicalSpaceSpec | None) -> list[str]:
    if spec is not None and mode == ActionMode.JOINT_DELTA.value and dim == spec.num_joints:
        return list(spec.joint_names)
    return [f"dim_{i}" for i in range(dim)]


def _check_pair(index: int, pred: ChunkPrediction) -> None:
    """Reject structurally unusable pairs with a precise error (index into the input list)."""
    p, t = pred.predicted, pred.target
    for role, chunk in (("predicted", p), ("target", t)):
        if not isinstance(chunk.targets, np.ndarray) or chunk.targets.ndim != 2:
            raise ValueError(f"prediction[{index}].{role}: targets must be a [T, D] array")
        if chunk.targets.shape[0] < 1:
            raise ValueError(f"prediction[{index}].{role}: empty chunk (T == 0)")
        if not np.isfinite(chunk.targets).all():
            raise ValueError(f"prediction[{index}].{role}: targets contain NaN/Inf")
        g = np.asarray(chunk.gripper_target)
        if g.ndim != 1 or g.shape[0] != chunk.targets.shape[0]:
            raise ValueError(f"prediction[{index}].{role}: gripper_target must be [T]")
        if not np.isfinite(g).all():
            raise ValueError(f"prediction[{index}].{role}: gripper_target contains NaN/Inf")
    if p.targets.shape != t.targets.shape:
        raise ValueError(
            f"prediction[{index}]: predicted/target shape mismatch "
            f"{p.targets.shape} vs {t.targets.shape}"
        )
    if p.mode != t.mode:
        raise ValueError(f"prediction[{index}]: predicted/target mode mismatch")


def e1_metrics(
    predictions: Sequence[ChunkPrediction],
    spec: CanonicalSpaceSpec | None = None,
) -> E1Report:
    """Compute the E1 offline metrics report over a set of chunk predictions.

    All predictions must share the action mode and target dim D; chunk length T may vary
    (per-step metrics then average the predictions that reach each step index).
    Raises ValueError on empty input or structurally inconsistent pairs.
    """
    if len(predictions) == 0:
        raise ValueError("e1_metrics: no predictions given")

    first = predictions[0].predicted
    mode = first.mode.value if isinstance(first.mode, ActionMode) else str(first.mode)
    dim = int(first.targets.shape[1])
    max_t = 0
    for i, pred in enumerate(predictions):
        _check_pair(i, pred)
        p_mode = pred.predicted.mode
        p_mode_str = p_mode.value if isinstance(p_mode, ActionMode) else str(p_mode)
        if p_mode_str != mode:
            raise ValueError(f"prediction[{i}]: mode {p_mode_str!r} != {mode!r}")
        if int(pred.predicted.targets.shape[1]) != dim:
            raise ValueError(
                f"prediction[{i}]: target dim {pred.predicted.targets.shape[1]} != {dim}"
            )
        max_t = max(max_t, pred.predicted.num_steps)

    sq_sum = 0.0
    abs_sum = 0.0
    n_elems = 0
    dim_sq = np.zeros(dim, dtype=np.float64)
    dim_abs = np.zeros(dim, dtype=np.float64)
    dim_cnt = np.zeros(dim, dtype=np.float64)
    step_sq = np.zeros(max_t, dtype=np.float64)
    step_abs = np.zeros(max_t, dtype=np.float64)
    step_cnt = np.zeros(max_t, dtype=np.float64)
    grip_match = 0
    grip_total = 0
    grip_closed = 0
    grip_min = np.inf
    grip_max = -np.inf
    smooth_sq = {"pred": 0.0, "target": 0.0}
    smooth_cnt = {"pred": 0, "target": 0}
    per_ep: dict[str, dict[str, float]] = {}

    for pred in predictions:
        p = pred.predicted.targets.astype(np.float64)
        t = pred.target.targets.astype(np.float64)
        err = p - t
        sq = err**2
        ab = np.abs(err)
        steps = err.shape[0]

        sq_sum += float(sq.sum())
        abs_sum += float(ab.sum())
        n_elems += err.size
        dim_sq += sq.sum(axis=0)
        dim_abs += ab.sum(axis=0)
        dim_cnt += float(steps)
        step_sq[:steps] += sq.sum(axis=1)
        step_abs[:steps] += ab.sum(axis=1)
        step_cnt[:steps] += float(dim)

        gp = np.asarray(pred.predicted.gripper_target, dtype=np.float64)
        gt = np.asarray(pred.target.gripper_target, dtype=np.float64)
        matches = int(
            ((gp >= GRIPPER_BINARIZE_THRESHOLD) == (gt >= GRIPPER_BINARIZE_THRESHOLD)).sum()
        )
        grip_match += matches
        grip_total += int(gp.shape[0])
        grip_closed += int((gt >= GRIPPER_BINARIZE_THRESHOLD).sum())
        grip_min = min(grip_min, float(gt.min()))
        grip_max = max(grip_max, float(gt.max()))

        for key, x in (("pred", p), ("target", t)):
            if x.shape[0] >= 3:
                d2 = x[2:] - 2.0 * x[1:-1] + x[:-2]
                smooth_sq[key] += float((d2**2).sum())
                smooth_cnt[key] += d2.size

        ep = per_ep.setdefault(
            pred.episode_id,
            {"chunks": 0.0, "sq": 0.0, "abs": 0.0, "elems": 0.0, "gmatch": 0.0, "gtotal": 0.0},
        )
        ep["chunks"] += 1.0
        ep["sq"] += float(sq.sum())
        ep["abs"] += float(ab.sum())
        ep["elems"] += float(err.size)
        ep["gmatch"] += float(matches)
        ep["gtotal"] += float(gp.shape[0])

    closed_frac = grip_closed / grip_total if grip_total else 0.0
    labels = _dim_labels(mode, dim, spec)
    per_episode = {
        ep_id: EpisodeMetrics(
            num_chunks=int(ep["chunks"]),
            mse=ep["sq"] / ep["elems"],
            mae=ep["abs"] / ep["elems"],
            gripper_accuracy=(ep["gmatch"] / ep["gtotal"]) if ep["gtotal"] else 0.0,
        )
        for ep_id, ep in sorted(per_ep.items())
    }
    return E1Report(
        mode=mode,
        num_predictions=len(predictions),
        num_episodes=len(per_episode),
        horizon_steps=max_t,
        target_dim=dim,
        mse=sq_sum / n_elems,
        mae=abs_sum / n_elems,
        per_joint_mse={lab: float(dim_sq[i] / dim_cnt[i]) for i, lab in enumerate(labels)},
        per_joint_mae={lab: float(dim_abs[i] / dim_cnt[i]) for i, lab in enumerate(labels)},
        per_step_mse=tuple(float(v) for v in step_sq / step_cnt),
        per_step_mae=tuple(float(v) for v in step_abs / step_cnt),
        gripper_accuracy=(grip_match / grip_total) if grip_total else 0.0,
        gripper_dynamic_range=float(grip_max - grip_min) if grip_total else 0.0,
        gripper_majority_pct=max(closed_frac, 1.0 - closed_frac) * 100.0 if grip_total else 0.0,
        smoothness_pred=(smooth_sq["pred"] / smooth_cnt["pred"]) if smooth_cnt["pred"] else 0.0,
        smoothness_target=(
            (smooth_sq["target"] / smooth_cnt["target"]) if smooth_cnt["target"] else 0.0
        ),
        per_episode=per_episode,
    )


def evaluate_policy(
    policy: Policy,
    episodes: Iterable[tuple[Observation, ActionChunk] | tuple[Observation, ActionChunk, str]],
    *,
    episode_id: str = "holdout",
) -> list[ChunkPrediction]:
    """Run any ``Policy`` over (observation, target-chunk) pairs and collect predictions.

    Policy-agnostic: only ``policy.predict(observation)`` is called (no model internals).
    Items may optionally carry an episode id as a third element; otherwise ``episode_id``
    is used for all items. ``t_ns`` is taken from ``observation.state.timestamp_ns``.
    """
    predictions: list[ChunkPrediction] = []
    for item in episodes:
        if len(item) == 3:
            obs, target, ep_id = item
        else:
            obs, target = item
            ep_id = episode_id
        predictions.append(
            ChunkPrediction(
                predicted=policy.predict(obs),
                target=target,
                episode_id=ep_id,
                t_ns=int(obs.state.timestamp_ns),
            )
        )
    return predictions


def load_episode_ids(path: str | Path) -> set[str]:
    """Episode ids from a plain one-per-line list **or** from a ``predictions.jsonl``.

    Accepting the predictions file directly is the point: the fine-tune excludes a holdout with
    ``--exclude-episodes`` and the evaluator scores exactly that holdout, so pointing both at one
    file makes them share a single definition of the split instead of two that can drift apart.
    Blank lines and ``#`` comments are ignored.
    """
    ids: set[str] = set()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            episode_id = json.loads(line).get("episode_id")
            if episode_id:
                ids.add(str(episode_id))
        else:
            ids.add(line)
    return ids


def build_eval_pairs(
    episode_dir: str | Path,
    camera: str,
    chunk_steps: int,
    *,
    num_frames: int | None = None,
) -> list[tuple[Observation, ActionChunk, str]]:
    """``(Observation, target chunk, episode_id)`` triples for :func:`evaluate_policy`.

    One triple per recorded action chunk. The observation is the frame and state in effect *at*
    the chunk's timestamp — the last of each at or before ``ts``, never the next one, which would
    hand the policy an observation from after the decision it is being asked to make.

    Chunks shorter than ``chunk_steps`` are skipped and longer ones truncated, matching
    ``EpisodeDataset``'s contract, so the holdout samples line up with what training saw.

    ``num_frames`` fills ``Observation.image_history`` with the window ending at that frame — the
    **same** window ``EpisodeDataset`` selected during training, via the same
    :func:`~wam.data.episode.frame_window_indices`. Left ``None`` the observation carries a single
    frame and a video policy tiles it, which is what every result recorded before 2026-07-30 was
    measured with; the default therefore reproduces those runs rather than silently redefining
    them (T-29, ``docs/improvements.md`` I-7).

    Memory: the window is a *view* into the episode's frame array whenever it does not need
    clamping, so history costs nothing beyond the array already held by the single-frame path.
    Only the first few chunks of an episode, where the window runs off the front, are copies.

    ``EpisodeReader`` is imported inside the function on purpose: reading frames pulls in the
    video decoding stack, and ``wam.evaluation`` is imported by torch-free, video-free consumers
    (``scripts/run_bench.py`` scoring an archived ``predictions.jsonl`` on a laptop).
    """
    from wam.data.episode import EpisodeReader, frame_window_indices

    reader = EpisodeReader(episode_dir)
    frames = reader.read_frames(camera)
    frame_ts = reader.frame_timestamps(camera)
    states = reader.read_states()
    state_ts = np.asarray([s.timestamp_ns for s in states], dtype=np.int64)
    instruction = reader.manifest.instruction
    episode_id = reader.manifest.episode_id

    pairs: list[tuple[Observation, ActionChunk, str]] = []
    for chunk, _executed_prefix, ts in reader.read_actions():
        if chunk.num_steps < chunk_steps:
            continue
        target = (
            chunk
            if chunk.num_steps == chunk_steps
            else ActionChunk(
                mode=chunk.mode,
                targets=np.asarray(chunk.targets[:chunk_steps], dtype=np.float32),
                gripper_target=np.asarray(chunk.gripper_target[:chunk_steps], dtype=np.float32),
                dt_s=chunk.dt_s,
            )
        )
        frame_idx = max(int(np.searchsorted(frame_ts, ts, side="right")) - 1, 0)
        state = states[max(int(np.searchsorted(state_ts, ts, side="right")) - 1, 0)]
        history = None
        if num_frames is not None:
            indices = frame_window_indices(frame_idx, num_frames, frames.shape[0])
            lo = frame_idx - num_frames + 1
            # A contiguous unclamped window is a slice, i.e. a view — no copy. Fancy indexing
            # (the clamped case, only near the episode start) copies.
            window = frames[lo : frame_idx + 1] if lo >= 0 else frames[indices]
            history = {camera: window}
        obs = Observation(
            images={camera: frames[frame_idx]},
            state=state,
            instruction=instruction,
            image_history=history,
        )
        pairs.append((obs, target, episode_id))
    return pairs


def holdout_split(
    episode_ids: Iterable[str],
    ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Deterministic episode-level train/holdout split.

    ``ratio`` is the holdout fraction (0 < ratio < 1). Ids are deduplicated and sorted before
    shuffling with ``random.Random(seed)``, so the split depends only on the id SET and the
    seed — never on input order. With >= 2 ids both sides are guaranteed non-empty.
    Returns ``(train_ids, holdout_ids)``, each sorted.
    """
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"holdout_split: ratio must be in (0, 1), got {ratio}")
    ids = sorted(set(episode_ids))
    n = len(ids)
    if n == 0:
        return [], []
    if n == 1:
        return list(ids), []
    n_holdout = min(max(round(n * ratio), 1), n - 1)
    rng = random.Random(seed)
    rng.shuffle(ids)
    holdout = ids[n - n_holdout :]
    train = ids[: n - n_holdout]
    return sorted(train), sorted(holdout)


# --- JSONL serialization (used by scripts/eval_offline.py) -------------------------------------
#
# One prediction per line:
# {"episode_id": "ep001", "t_ns": 0,
#  "predicted": {"mode": "joint_delta", "dt_s": 0.05,
#                "targets": [[...], ...], "gripper_target": [...]},
#  "target":    {"mode": "joint_delta", "dt_s": 0.05,
#                "targets": [[...], ...], "gripper_target": [...]}}


def _chunk_to_dict(chunk: ActionChunk) -> dict[str, Any]:
    mode = chunk.mode.value if isinstance(chunk.mode, ActionMode) else str(chunk.mode)
    return {
        "mode": mode,
        "dt_s": float(chunk.dt_s),
        "targets": np.asarray(chunk.targets, dtype=np.float32).tolist(),
        "gripper_target": np.asarray(chunk.gripper_target, dtype=np.float32).tolist(),
    }


def _chunk_from_dict(data: Mapping[str, Any]) -> ActionChunk:
    return ActionChunk(
        mode=ActionMode(data["mode"]),
        targets=np.asarray(data["targets"], dtype=np.float32),
        gripper_target=np.asarray(data["gripper_target"], dtype=np.float32),
        dt_s=float(data["dt_s"]),
    )


def prediction_to_dict(pred: ChunkPrediction) -> dict[str, Any]:
    """Serialize one ChunkPrediction to a JSON-compatible dict (see module format doc)."""
    return {
        "episode_id": pred.episode_id,
        "t_ns": int(pred.t_ns),
        "predicted": _chunk_to_dict(pred.predicted),
        "target": _chunk_to_dict(pred.target),
    }


def prediction_from_dict(data: Mapping[str, Any]) -> ChunkPrediction:
    """Inverse of ``prediction_to_dict``."""
    return ChunkPrediction(
        predicted=_chunk_from_dict(data["predicted"]),
        target=_chunk_from_dict(data["target"]),
        episode_id=str(data["episode_id"]),
        t_ns=int(data["t_ns"]),
    )


def save_predictions_jsonl(predictions: Sequence[ChunkPrediction], path: str | Path) -> None:
    """Write predictions as JSONL (one ``prediction_to_dict`` object per line)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(prediction_to_dict(pred)) + "\n")


def load_predictions_jsonl(path: str | Path) -> list[ChunkPrediction]:
    """Load predictions from a JSONL file written in the documented format."""
    predictions: list[ChunkPrediction] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                predictions.append(prediction_from_dict(json.loads(line)))
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: bad prediction record: {exc}") from exc
    return predictions
