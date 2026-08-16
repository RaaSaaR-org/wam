"""T-46 / PR-12 §9 — the probe's load-bearing behaviours, each pinned by the mutant it fears.

PR-12 makes exactly one claim about the *code* rather than about the corpus, and everything else
rests on it: **V-chain and the current anchoring are identical under perfect tracking.** If that is
false, V-chain is a change of premise and the experiment is asking a different question than the
pre-registration says it is. That is `test_v_chain_equals_the_current_anchoring_under_perfect_tracking`
and it is the reason this file exists.

The rest guard the two traps PR-12 §5 names:

  A. **A manipulation that is silently a no-op.** A V-chain that reduced to the current anchoring
     produces a flat grid, and a flat grid reads as a confident negative — the same shape a real
     verdict D has. Pinned two-sided: row 0 must move, rows 1.. must not.
  B. **An instrument that flatters itself.** V-mask removes the largest element of a sum and then
     reports the sum got smaller, which is arithmetic. It is only a finding if the ratio against an
     IDENTICALLY MASKED baseline moves, so the baselines are pinned as masked too — including the
     non-obvious one, `_causal_previous_action`, whose clip index shifts when the chunk shortens.

The fixtures are synthetic and tiny on purpose: what is under test is index arithmetic, and a real
episode would test pyarrow.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from wam.evaluation import bench_metrics  # noqa: E402
from wam.evaluation.benchmark import _causal_previous_action  # noqa: E402
from wam.evaluation.offline import ChunkPrediction  # noqa: E402
from wam.interfaces.schema import ActionChunk, ActionMode  # noqa: E402


def _load(name: str):
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = _load("probe_step_zero_anchor")
sweep = _load("sweep_label_anchoring")
eval_t39 = sweep._load_eval_t39()
convert = eval_t39._load_script("convert_lerobot_g1")


CHUNK_STEPS = 4
N_STEPS = 40
DT_S = 1.0 / 30.0


def _raw(*, tracking_offset: float = 0.0, seed: int = 0) -> dict[str, np.ndarray]:
    """A source episode with a controllable steady-state tracking offset.

    ``tracking_offset == 0`` gives the adapter's premise exactly: ``action[i] == state[i+1]``, so
    the command the controller was given at ``i`` is the position the robot reached at ``i+1``.
    A non-zero value adds a CONSTANT to every command — the mechanism PR-12 §1 describes, the one
    that cancels in every command-to-command difference and survives only in step 0.
    """
    rng = np.random.default_rng(seed)
    dim = convert.SOURCE_STATE_DIM
    state = np.cumsum(rng.normal(scale=0.01, size=(N_STEPS + 1, dim)), axis=0).astype(np.float32)
    action = (state[1:] + tracking_offset).astype(np.float32)
    return {
        "state": state[:-1],
        "action": action,
        "ts_ns": (np.arange(N_STEPS) * int(DT_S * 1e9)).astype(np.int64),
    }


class _Reader:
    """The two things the chunk builders ask of an EpisodeReader, and nothing else."""

    def __init__(self, anchors: list[int], ts_ns: np.ndarray) -> None:
        self._anchors = anchors
        self._ts = ts_ns
        self.manifest = SimpleNamespace(episode_id="synthetic-000000")

    def read_actions(self):
        for index in self._anchors:
            yield SimpleNamespace(dt_s=DT_S), CHUNK_STEPS, int(self._ts[index])


def _fixture(*, tracking_offset: float = 0.0):
    raw = _raw(tracking_offset=tracking_offset)
    # Start at CHUNK_STEPS, not 0: the real driver trims the first eval pair of every episode, and
    # V-chain needs `start - 1 >= 0`. Anchoring at index 0 would test a chunk the probe drops.
    anchors = list(range(CHUNK_STEPS, N_STEPS - CHUNK_STEPS, CHUNK_STEPS))
    reader = _Reader(anchors, raw["ts_ns"])
    # affine=None is the LEGACY mapping — the one `datasets/gr00t-apple-full` was built with, and
    # therefore the one every number this probe produces is scored under.
    mapping = eval_t39.GripperMapping(affine=None, column=None)
    return raw, reader, mapping


def _unmodified(reader, raw, mapping, delay):
    return sweep.delayed_oracle_action_chunks(
        eval_t39, reader, raw, CHUNK_STEPS, mapping, convert, delay=delay
    )


def _v_chain(reader, raw, mapping, delay):
    return probe.chained_oracle_action_chunks(
        eval_t39, reader, raw, CHUNK_STEPS, mapping, convert, delay=delay
    )


# ---------------------------------------------------------------- the claim everything rests on


def test_v_chain_equals_the_current_anchoring_under_perfect_tracking_at_d_zero():
    """PR-12 §3 claim 1, and the most load-bearing assertion in this file.

    With ``action[i] == state[i+1]`` we have ``state[index] == action[index - 1]``, so anchoring on
    the previous COMMAND and anchoring on the measured STATE are the same vector and the two chunk
    builders must agree bit for bit. This is what makes V-chain "the same premise made robust to
    the premise failing" rather than a different premise — the sentence PR-12 §3 spends a paragraph
    on, asserted here instead of argued there.

    **At `d = 0`.** See the next test for why that qualifier is load-bearing and is not in PR-12.
    """
    raw, reader, mapping = _fixture(tracking_offset=0.0)
    unmod = _unmodified(reader, raw, mapping, 0)
    chain = _v_chain(reader, raw, mapping, 0)
    assert set(unmod) == set(chain)
    assert unmod, "fixture produced no chunks — the test would pass vacuously"
    for t_ns in unmod:
        np.testing.assert_array_equal(unmod[t_ns].targets, chain[t_ns].targets)
        np.testing.assert_array_equal(unmod[t_ns].gripper_target, chain[t_ns].gripper_target)


@pytest.mark.parametrize("delay", [-2, -1, 1])
def test_at_a_nonzero_delay_v_chain_also_moves_the_anchor_and_that_is_a_confound(delay):
    """**A DESIGN DEFECT IN PR-12, FOUND BY ITS OWN TESTS BEFORE ANY CELL EXISTED.**

    PR-12 §3 registers `d = −2` as the primary anchor and describes V-chain as changing one thing.
    At `d != 0` it changes two, and the two cannot be separated by construction:

        unmodified  anchor = state[index]           <- the eval timestamp's state
        V-chain     anchor = action[index + d - 1]  <- the command preceding the SLICE

    The slice moved by `d`, so the command preceding it is `d` steps away from the eval timestamp.
    Under perfect tracking that is `state[index + d]`, which this test pins. "Homogenise step 0"
    and "keep the anchor at the eval timestamp" are contradictory requirements once the slice has
    moved — there is no third definition that satisfies both.

    Consequence, recorded here and in the driver rather than by amending the pre-registration
    (which is what this repo forbids): **the unconfounded test of P2 is the `d = 0` cell.** The
    `d = −2` cell, which `T46_RULE_V1` reads for its verdict, is a JOINT test of the delay and the
    homogenisation. PR-12 §6 already requires both anchors to be recorded, so the unconfounded
    reading is available without changing the rule — and the result document must state which is
    which rather than quoting the verdict alone.
    """
    raw, reader, mapping = _fixture(tracking_offset=0.0)
    chain = _v_chain(reader, raw, mapping, delay)
    anchors = eval_t39.raw_anchor_indices(reader, raw)
    state = raw["state"]
    assert chain
    for t_ns, chunk in chain.items():
        index = anchors[int(t_ns)]
        # The anchor is the state `delay` steps from the eval timestamp — no more, no less. That
        # bounds the confound at exactly the delay, which is what makes it reportable.
        expected_anchor = convert.canonical_q(state[index + delay : index + delay + 1])[0]
        first_command = convert.canonical_q(raw["action"][index + delay : index + delay + 1])[0]
        np.testing.assert_allclose(
            chunk.targets[0], first_command - expected_anchor, rtol=0, atol=1e-6
        )


# ------------------------------------------------------------------------- trap A, both sides


@pytest.mark.parametrize("delay", [-2, 0])
def test_v_chain_moves_row_zero_when_tracking_is_imperfect(delay):
    """Trap A, first side: the manipulation must reach the array.

    A V-chain that silently reduced to the current anchoring would satisfy every OTHER test here,
    including the perfect-tracking one above — which is exactly why that test cannot be the only
    one. Under a real tracking offset the two anchorings must disagree.
    """
    raw, reader, mapping = _fixture(tracking_offset=0.05)
    unmod = _unmodified(reader, raw, mapping, delay)
    chain = _v_chain(reader, raw, mapping, delay)
    common = sorted(set(unmod) & set(chain))
    assert common
    assert any(
        not np.array_equal(unmod[t].targets[0], chain[t].targets[0]) for t in common
    ), "V-chain left row 0 unchanged under a tracking offset — it is not reaching the anchor"


@pytest.mark.parametrize("delay", [-2, -1, 0, 1])
def test_v_chain_touches_row_zero_and_nothing_else(delay):
    """Trap A, second side, and the one that matters more.

    ``commanded_to_chunk`` puts the anchor in row 0 and only row 0, so rows 1.. must be bit-identical
    whatever the anchor is. A V-chain that re-chained every step would also 'work' — it would
    produce finite, plausible, wrong numbers — and it would be a different experiment wearing this
    one's name. This is the offline half of the driver's G0.3.
    """
    raw, reader, mapping = _fixture(tracking_offset=0.05)
    unmod = _unmodified(reader, raw, mapping, delay)
    chain = _v_chain(reader, raw, mapping, delay)
    for t_ns in sorted(set(unmod) & set(chain)):
        np.testing.assert_array_equal(unmod[t_ns].targets[1:], chain[t_ns].targets[1:])


def test_v_chain_row_zero_is_exactly_the_previous_command_difference():
    """The definition itself, checked against arithmetic done outside the adapter.

    ``targets[0]`` must be ``canonical_q(action[s]) - canonical_q(action[s-1])`` — PR-12 §3's one
    line, computed here from the raw array rather than by calling the thing under test.
    """
    raw, reader, mapping = _fixture(tracking_offset=0.05)
    chain = _v_chain(reader, raw, mapping, 0)
    anchors = eval_t39.raw_anchor_indices(reader, raw)
    action = raw["action"]
    for t_ns, chunk in chain.items():
        s = anchors[int(t_ns)]
        expected = convert.canonical_q(action[s : s + 1])[0] - convert.canonical_q(
            action[s - 1 : s]
        )[0]
        np.testing.assert_allclose(chunk.targets[0], expected, rtol=0, atol=1e-6)


def test_v_chain_drops_the_chunk_that_has_no_previous_command():
    """`start - 1 < 0` has to be a drop, not a wrap to `action[-1]`.

    numpy would happily index `action[-1]`, the episode's LAST command, and produce a finite
    plausible chunk anchored on the future. The eligibility test exists to prevent exactly that.
    """
    raw = _raw()
    reader = _Reader([0], raw["ts_ns"])
    mapping = eval_t39.GripperMapping(affine=None, column=None)
    assert _v_chain(reader, raw, mapping, 0) == {}
    assert _unmodified(reader, raw, mapping, 0) != {}, "the unmodified cell should still keep it"


# ------------------------------------------------------------------------- trap B, the instrument


def _chunk(targets: np.ndarray) -> ActionChunk:
    return ActionChunk(
        mode=ActionMode.JOINT_DELTA,
        targets=targets.astype(np.float32),
        gripper_target=np.zeros(targets.shape[0], dtype=np.float32),
        dt_s=DT_S,
    )


def _predictions(steps: int = 8, chunks: int = 6, seed: int = 1) -> list[ChunkPrediction]:
    """Non-overlapping chunks, which is what `relabel_chunks` produces and what the clip relies on."""
    rng = np.random.default_rng(seed)
    out: list[ChunkPrediction] = []
    for i in range(chunks):
        target = rng.normal(scale=0.01, size=(steps, 15))
        predicted = target + rng.normal(scale=0.001, size=(steps, 15))
        # A big step-0 discontinuity, the thing PR-12 is about.
        predicted[0] += 0.5
        out.append(
            ChunkPrediction(
                predicted=_chunk(predicted),
                target=_chunk(target),
                episode_id="ep-0" if i < chunks // 2 else "ep-1",
                t_ns=int((i % (chunks // 2)) * steps * DT_S * 1e9),
            )
        )
    return out


def test_mask_drops_exactly_one_step_from_both_arms():
    """Both arms, never the prediction alone.

    The target has no discontinuity at step 0, so dropping its step 0 costs it a legitimate term
    and biases the comparison AGAINST the finding. Masking the prediction only would remove the
    contaminated element from one side and keep it on the other — an instrument built to find what
    it is looking for.
    """
    preds = _predictions(steps=8)
    masked = probe.mask_step_zero(preds)
    assert len(masked) == len(preds)
    for before, after in zip(preds, masked, strict=True):
        assert after.predicted.targets.shape[0] == before.predicted.targets.shape[0] - 1
        assert after.target.targets.shape[0] == before.target.targets.shape[0] - 1
        assert after.predicted.gripper_target.shape[0] == before.predicted.gripper_target.shape[0] - 1
        assert after.target.gripper_target.shape[0] == before.target.gripper_target.shape[0] - 1
        np.testing.assert_array_equal(after.target.targets, before.target.targets[1:])
        np.testing.assert_array_equal(after.predicted.targets, before.predicted.targets[1:])


def test_mask_does_not_mutate_its_input():
    """The driver scores the unmodified cell from the same prediction list; aliasing would corrupt it."""
    preds = _predictions(steps=8)
    before = [p.target.targets.copy() for p in preds]
    probe.mask_step_zero(preds)
    for original, pred in zip(before, preds, strict=True):
        np.testing.assert_array_equal(original, pred.target.targets)


def test_the_repeat_baseline_is_the_same_element_masked_or_not():
    """Trap B's non-obvious half, and the reason V-mask can be a slice rather than a second scorer.

    `_causal_previous_action` takes the previous chunk's step `clip(stride - 1, 0, T - 1)`. Masking
    shortens the chunk, so the clip lands one index lower — and because masking ALSO shifted every
    element down by one, that lower index is the SAME original element. If it were not, the masked
    run would score against a shifted baseline and every masked ratio would be uninterpretable.

    This holds because our eval chunks are non-overlapping (`stride == T`). The driver asserts that
    precondition at runtime; this test is why the assertion is allowed to be the only guard.
    """
    preds = _predictions(steps=8)
    masked = probe.mask_step_zero(preds)
    # Per episode, exactly as `bench_metrics` does it (benchmark.py:471-475). Calling this across
    # an episode boundary computes a stride from two unrelated timestamps, which is a property of
    # the test rather than of the scorer.
    for episode_id in {p.episode_id for p in preds}:
        group = [i for i, p in enumerate(preds) if p.episode_id == episode_id]
        plain = [preds[i] for i in group]
        cut = [masked[i] for i in group]
        for index in range(1, len(plain)):
            unmasked_prev = _causal_previous_action(
                plain, index, plain[index].target.targets.shape[0]
            )
            masked_prev = _causal_previous_action(cut, index, cut[index].target.targets.shape[0])
            np.testing.assert_array_equal(unmasked_prev, masked_prev)


def test_masking_does_not_by_itself_move_the_skill_ratio():
    """The arithmetic-versus-finding distinction, made a test.

    On chunks whose step 0 is NOT differentially worse than the baseline's — here, a uniform error
    at every step — masking must leave `skill_vs_repeat_pct` roughly where it was, even though the
    raw MSE changes. If masking moved the ratio on its own, PR-12 §5B's guard would be worthless
    and every V-mask number would be an artefact of removing a term.
    """
    rng = np.random.default_rng(7)
    preds: list[ChunkPrediction] = []
    for i in range(6):
        target = rng.normal(scale=0.01, size=(8, 15))
        predicted = target + rng.normal(scale=0.003, size=(8, 15))  # uniform error, no step-0 spike
        preds.append(
            ChunkPrediction(
                predicted=_chunk(predicted),
                target=_chunk(target),
                episode_id="ep-0" if i < 3 else "ep-1",
                t_ns=int((i % 3) * 8 * DT_S * 1e9),
            )
        )
    plain = bench_metrics(preds, run_name="plain", spec_version="0.1.0")
    masked = bench_metrics(probe.mask_step_zero(preds), run_name="masked", spec_version="0.1.0")
    assert abs(masked.skill_vs_repeat_pct - plain.skill_vs_repeat_pct) < 15.0


# ------------------------------------------------------------------------------- the profile


def test_profile_has_one_entry_per_step():
    preds = _predictions(steps=8)
    assert len(probe.per_step_profile(preds)) == 8
    assert len(probe.per_step_profile(probe.mask_step_zero(preds))) == 7


def test_profile_reproduces_the_scorers_horizon_ratio():
    """The profile is pinned to `benchmark.py`'s own arithmetic, not merely shaped like it.

    `horizon_ratio` is `per_step_mse[-1] / per_step_mse[0]` (benchmark.py:563-565). If this
    recomputation disagreed, `step_zero_share_pct` would be measuring a different vector than the
    one the verdict's coherence check assumes — and the driver would report verdict X or C off a
    number that is not the metric's.
    """
    preds = _predictions(steps=8)
    profile = probe.per_step_profile(preds)
    bench = bench_metrics(preds, run_name="profile", spec_version="0.1.0")
    assert profile[-1] / profile[0] == pytest.approx(bench.horizon_ratio, rel=1e-12)


def test_step_zero_share_is_a_percentage_of_the_summed_profile():
    assert probe.step_zero_share([3.0, 1.0]) == pytest.approx(75.0)
    assert probe.step_zero_share([0.0, 0.0]) == 0.0


def test_a_step_zero_spike_dominates_the_share_and_masking_removes_it():
    """The mechanism itself, on a fixture built to have it — the controlled version of the corpus."""
    preds = _predictions(steps=8)  # `predicted[0] += 0.5`
    share = probe.step_zero_share(probe.per_step_profile(preds))
    assert share > 90.0, f"the spiked fixture should be step-0 dominated, got {share:.1f} %"
    masked_share = probe.step_zero_share(probe.per_step_profile(probe.mask_step_zero(preds)))
    assert masked_share < 50.0
