"""T-44 / PR-10 §8 — the sweep's two load-bearing behaviours, each pinned by a mutant.

PR-10 §8 names two mutants specifically, and the reason is `tests/test_t39_baseline.py`'s: both
produce finite, plausible, wrong CURVES that no assertion on shape, range or monotonicity catches.
A sweep is a picture, and a wrong picture is more persuasive than a wrong number.

  1. **Shifting the anchor as well as the command index.** PR-10 §3 fixes that only the command
     slice moves. A driver that also moved `state[index]` to `state[index + delay]` would still
     produce nine finite scores and a curve with a maximum somewhere — and that maximum would be a
     property of the mutation, not of the controller.
  2. **Trimming per-delay instead of by the intersection.** Scoring each delay on whatever chunks
     that delay happens to reach makes the delays that reach fewest chunks look different for a
     reason that has nothing to do with anchoring.

The fixtures are synthetic and tiny on purpose. What is under test is index arithmetic, and a real
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


def _load(name: str):
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sweep = _load("sweep_label_anchoring")
eval_t39 = sweep._load_eval_t39()
convert = eval_t39._load_script("convert_lerobot_g1")


CHUNK_STEPS = 4
N_STEPS = 40
DT_S = 1.0 / 30.0


def _raw(seed: int = 0) -> dict[str, np.ndarray]:
    """A source episode whose `action` LEADS `state` by exactly one step.

    Constructed, not sampled: `action[i] == state[i + 1]`, which is the adapter docstring's
    perfect-tracking premise. Under it `delay = 0` is exactly right and any other delay is wrong,
    so the sweep's own arithmetic has a known answer to be checked against.
    """
    rng = np.random.default_rng(seed)
    dim = convert.SOURCE_STATE_DIM
    state = np.cumsum(rng.normal(scale=0.01, size=(N_STEPS + 1, dim)), axis=0).astype(np.float32)
    action = state[1:].copy()
    return {
        "state": state[:-1],
        "action": action,
        "ts_ns": (np.arange(N_STEPS) * int(DT_S * 1e9)).astype(np.int64),
    }


class _Reader:
    """The two things `delayed_oracle_action_chunks` asks of an EpisodeReader, and nothing else."""

    def __init__(self, anchors: list[int], ts_ns: np.ndarray) -> None:
        self._anchors = anchors
        self._ts = ts_ns
        self.manifest = SimpleNamespace(episode_id="synthetic-000000")

    def read_actions(self):
        for index in self._anchors:
            yield SimpleNamespace(dt_s=DT_S), CHUNK_STEPS, int(self._ts[index])


@pytest.fixture
def fixture():
    raw = _raw()
    anchors = list(range(0, N_STEPS - CHUNK_STEPS, CHUNK_STEPS))
    reader = _Reader(anchors, raw["ts_ns"])
    # affine=None is the LEGACY mapping — the one `datasets/gr00t-apple-full` was built with, and
    # therefore the one every number this sweep will produce is scored under.
    mapping = eval_t39.GripperMapping(affine=None, column=None)
    return raw, reader, mapping


def _chunks(reader, raw, mapping, delay):
    return sweep.delayed_oracle_action_chunks(
        eval_t39, reader, raw, CHUNK_STEPS, mapping, convert, delay=delay
    )


def test_delay_zero_is_byte_identical_to_the_unswept_adapter(fixture):
    """`delay=0` must BE `oracle_action_chunks`, not merely agree with it.

    This is the join between PR-10 and PR-07-RESULT: the bridge in §5 G0.2 compares a number from
    this file against one produced by `eval_t39_baseline`, and that comparison is only meaningful
    if the two code paths coincide at zero.
    """
    raw, reader, mapping = fixture
    swept = _chunks(reader, raw, mapping, 0)
    direct = eval_t39.oracle_action_chunks(reader, raw, CHUNK_STEPS, mapping, convert)
    assert set(swept) == set(direct)
    for t_ns in swept:
        np.testing.assert_array_equal(swept[t_ns].targets, direct[t_ns].targets)
        np.testing.assert_array_equal(swept[t_ns].gripper_target, direct[t_ns].gripper_target)


def test_a_nonzero_delay_actually_changes_the_targets(fixture):
    """Guards the degenerate pass: a driver that silently ignored `delay` would satisfy every
    other test in this file."""
    raw, reader, mapping = fixture
    base = _chunks(reader, raw, mapping, 0)
    for delay in (-1, 1, 2):
        shifted = _chunks(reader, raw, mapping, delay)
        common = set(base) & set(shifted)
        assert common, f"delay={delay} retained no chunk in common with delay=0"
        assert any(
            not np.array_equal(base[t].targets, shifted[t].targets) for t in common
        ), f"delay={delay} produced identical targets — the delay is not reaching the command slice"


def test_perfect_tracking_makes_delay_zero_exact(fixture):
    """MUTANT 1, stated positively first.

    On a corpus where `action[i] == state[i+1]`, the commanded displacement over step `i` is
    exactly the executed one, so `delay=0` reproduces the executed deltas to floating point. Any
    driver that shifted the ANCHOR along with the command would break this, because the chunk
    would then start from a position the robot was not at.
    """
    raw, reader, mapping = fixture
    chunks = _chunks(reader, raw, mapping, 0)
    q = convert.canonical_q(raw["state"])
    for t_ns, chunk in chunks.items():
        index = int(np.searchsorted(raw["ts_ns"], t_ns))
        expected = np.diff(q[index : index + CHUNK_STEPS + 1], axis=0)
        np.testing.assert_allclose(chunk.targets, expected, atol=1e-5)


def test_mutant_shifting_the_anchor_too_is_not_what_we_do(fixture):
    """MUTANT 1. Shift the anchor as well and the result must DIFFER from the driver's.

    The mutant is finite and plausible — that is the whole problem with it — so the assertion is
    that the two disagree, not that the mutant explodes.
    """
    raw, reader, mapping = fixture

    def mutant(delay: int):
        anchors = eval_t39.raw_anchor_indices(reader, raw)
        action = np.asarray(raw["action"], dtype=np.float32)
        state = np.asarray(raw["state"], dtype=np.float32)
        out = {}
        for chunk, _prefix, t_ns in reader.read_actions():
            index = anchors[int(t_ns)]
            start = index + delay
            if start < 0 or start + CHUNK_STEPS > action.shape[0]:
                continue
            out[int(t_ns)] = eval_t39.commanded_to_chunk(
                action[start : start + CHUNK_STEPS],
                state[start],  # <-- THE MUTATION: the anchor moves too
                dt_s=DT_S,
                mapping=mapping,
                convert=convert,
            )
        return out

    delay = 2
    ours = _chunks(reader, raw, mapping, delay)
    theirs = mutant(delay)
    common = set(ours) & set(theirs)
    assert common
    assert any(
        not np.allclose(ours[t].targets, theirs[t].targets) for t in common
    ), "the anchor-shifting mutant produced our numbers — the anchor is moving when it must not"
    # And at delay=0 the mutation is invisible, which is exactly why a delay-0-only test would
    # have passed it through.
    assert set(mutant(0)) == set(_chunks(reader, raw, mapping, 0))


def test_trim_pairs_is_the_intersection_not_a_per_delay_filter():
    """MUTANT 2. `trim_pairs` drops one pair at each end and does not consult the delay.

    A per-delay filter is the plausible alternative and it is wrong for a reason that never shows
    up in the output: it silently scores each delay on a different chunk set, so the curve compares
    nine populations rather than nine anchorings.
    """
    pairs = list(range(10))
    trimmed = sweep.trim_pairs(pairs)
    assert trimmed == list(range(1, 9))
    # Same answer regardless of what delay is about to be applied: the signature cannot even
    # express a dependence on it.
    assert sweep.trim_pairs(pairs) == trimmed
    assert sweep.trim_pairs([1, 2]) == []
    assert sweep.trim_pairs([]) == []


def test_trimming_leaves_every_retained_chunk_reachable_at_every_delay(fixture):
    """The claim `trim_pairs` rests on: one pair at each end covers the whole sweep window.

    If this ever stops holding, `ChunkLookupPolicy` raises on a missing chunk rather than scoring
    silently — but it would raise mid-sweep, after some delays had already been recorded.
    """
    raw, reader, mapping = fixture
    anchors = sorted(eval_t39.raw_anchor_indices(reader, raw).values())
    retained = anchors[1:-1]
    max_delay = CHUNK_STEPS - 1
    for delay in range(-max_delay, max_delay + 1):
        reachable = set(_chunks(reader, raw, mapping, delay))
        ts = raw["ts_ns"]
        for index in retained:
            assert int(ts[index]) in reachable, f"chunk at {index} unreachable at delay={delay}"


def test_verdict_d_star_zero_cannot_be_T():
    """`T44_RULE_V1`: if the best delay is the one we already use, the sweep found nothing.

    Pinned because it is the single most tempting way to read a flattering curve — a large positive
    B score at d=0 is not a timing finding, it is the status quo.
    """
    curve = {d: {"skill_vs_repeat_pct": 50.0 if d == 0 else 10.0, "smoothness_ratio": 8.5}
             for d in range(-4, 5)}
    out = sweep._verdict(curve, curve, 4)
    assert out["verdict"] != "T"
    assert out["d_star"] == 0


def test_verdict_endpoint_takes_precedence_over_everything():
    """E first. See `_verdict`'s docstring for why, and for the admission that PR-10's table did
    not fix this precedence."""
    curve = {d: {"skill_vs_repeat_pct": float(d * 10), "smoothness_ratio": 8.5}
             for d in range(-4, 5)}
    out = sweep._verdict(curve, curve, 4)
    assert out["verdict"] == "E"
    assert out["d_star"] == 4


def test_verdict_J_when_nothing_clears_l1_on_A():
    curve = {d: {"skill_vs_repeat_pct": -100.0 - abs(d), "smoothness_ratio": 8.52}
             for d in range(-4, 5)}
    out = sweep._verdict(curve, curve, 4)
    assert out["verdict"] == "J"
    assert "not a shifted copy" in out["reading"]


def test_verdict_T_needs_the_material_floor_on_the_held_out_half():
    """A gain below `MATERIAL_FLOOR_PP` on B is I, not T — the floor is borrowed, not coined."""
    curve_a = {d: {"skill_vs_repeat_pct": 40.0 if d == 1 else -50.0, "smoothness_ratio": 8.5}
               for d in range(-4, 5)}
    thin = {d: {"skill_vs_repeat_pct": 5.0 if d == 1 else 0.5, "smoothness_ratio": 8.5}
            for d in range(-4, 5)}
    assert sweep._verdict(curve_a, thin, 4)["verdict"] == "I"

    fat = {d: {"skill_vs_repeat_pct": 40.0 if d == 1 else -50.0, "smoothness_ratio": 8.5}
           for d in range(-4, 5)}
    out = sweep._verdict(curve_a, fat, 4)
    assert out["verdict"] == "T"
    assert out["d_star"] == 1
    assert out["b_gain_pp"] == pytest.approx(90.0)
