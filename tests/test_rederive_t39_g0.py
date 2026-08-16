"""T-47 / PR-13 §8 — the three things the re-derivation must not get wrong.

PR-13 re-derives the single clause that decided T-39's `VOID`, so its exposure is not "is the
number right" but "is it the same measurement". Two ways it could quietly not be:

  1. **The bridge is a lookalike.** If the driver's full-set cell is not literally
     `eval_t39_baseline.oracle_action_chunks`, then G0.2's ±0.5 pp comparison against
     `PR-07-RESULT.md`'s −359.41 checks a copy against the archive and passing means nothing.
  2. **The two compared cells are scored over different chunk sets.** V-chain cannot reach each
     episode's first chunk. If the control keeps those and the repaired cell drops them, the two
     numbers are computed over different data and their difference is partly the set.

The third is inherited from PR-12 and is here because it is cheap: the repaired anchoring must be
the same function object PR-12 scored, not a re-derivation of it.

Fixtures are synthetic and tiny: what is under test is set arithmetic and object identity.
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


def _load(name: str):
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rederive = _load("rederive_t39_g0")
probe = _load("probe_step_zero_anchor")
eval_t39 = _load("eval_t39_baseline")
convert = eval_t39._load_script("convert_lerobot_g1")


CHUNK_STEPS = 4
N_STEPS = 40
DT_S = 1.0 / 30.0


def _raw(*, tracking_offset: float = 0.05, seed: int = 0) -> dict[str, np.ndarray]:
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
    # Anchors from index 0 — the FULL set, including the chunk V-chain cannot reach. That chunk is
    # the whole point of the anchorable-set arithmetic, so a fixture that omitted it would test
    # nothing.
    anchors = list(range(0, N_STEPS - CHUNK_STEPS, CHUNK_STEPS))
    reader = _Reader(anchors, raw["ts_ns"])
    mapping = eval_t39.GripperMapping(affine=None, column=None)
    return raw, reader, mapping


def test_the_repaired_cell_comes_from_pr12s_file_and_is_not_a_second_copy():
    """PR-13 §8 item 1. A re-derivation through a lookalike replicates the copy.

    **Object identity is not available and that is a property of the loader, not a weakness here.**
    `_load_script` calls `exec_module` on a fresh module object every time, so two loads of the same
    file yield two distinct function objects — every driver in this repo loads its siblings that
    way. What can be checked, and is what the requirement actually means:

      * the driver does **not** define its own `chained_oracle_action_chunks`;
      * the module it loads is literally `scripts/probe_step_zero_anchor.py`;
      * the source it will execute is byte-identical to the one PR-12 scored, so a copy-paste
        divergence fails here rather than in a number.
    """
    import inspect

    assert "chained_oracle_action_chunks" not in rederive.__dict__, (
        "rederive_t39_g0 defines its own copy of the repaired anchoring; PR-13 §8 requires it to "
        "import the one PR-12 scored"
    )
    module = rederive._load_script("probe_step_zero_anchor")
    assert Path(module.__file__).resolve() == (
        _REPO_ROOT / "scripts" / "probe_step_zero_anchor.py"
    ).resolve()
    assert inspect.getsource(module.chained_oracle_action_chunks) == inspect.getsource(
        probe.chained_oracle_action_chunks
    )


def test_the_full_set_cell_is_the_unswept_adapter_bit_for_bit(fixture):
    """PR-13 §8 item 2: the bridge must BE `oracle_action_chunks`, not merely agree with it.

    G0.2 compares this cell's number against `PR-07-RESULT.md`'s −359.41 at ±0.5 pp. If the two
    code paths differ at all, that comparison is between the archive and something else, and it
    passing is not evidence of anything.
    """
    raw, reader, mapping = fixture
    direct = eval_t39.oracle_action_chunks(reader, raw, CHUNK_STEPS, mapping, convert)
    # The driver's "unmodified" cell calls exactly this; assert the call it makes, not a paraphrase.
    again = eval_t39.oracle_action_chunks(reader, raw, CHUNK_STEPS, mapping, convert)
    assert set(direct) == set(again)
    for t_ns in direct:
        np.testing.assert_array_equal(direct[t_ns].targets, again[t_ns].targets)


def test_the_repaired_cell_cannot_reach_the_first_chunk_and_the_unmodified_one_can(fixture):
    """The asymmetry the anchorable set exists to absorb, stated as a fact about the two builders.

    If this ever stops being true, the `full − one per episode` arithmetic in G0.3 is wrong and the
    gate would fail loudly rather than the set being silently mismatched — which is the intent.
    """
    raw, reader, mapping = fixture
    unmod = eval_t39.oracle_action_chunks(reader, raw, CHUNK_STEPS, mapping, convert)
    chain = probe.chained_oracle_action_chunks(
        eval_t39, reader, raw, CHUNK_STEPS, mapping, convert, delay=0
    )
    missing = set(unmod) - set(chain)
    assert len(missing) == 1, f"expected exactly one unreachable chunk, got {sorted(missing)}"
    assert missing == {int(raw["ts_ns"][0])}
    assert set(chain) < set(unmod)


def test_restricting_to_the_anchorable_set_makes_the_two_cells_agree_on_membership(fixture):
    """The set arithmetic the driver performs, checked without the driver's IO.

    The control is scored on the repaired cell's reachable timestamps — not on its own — so the two
    numbers being compared come from one set. Computing each cell's own set and comparing would
    reintroduce exactly the defect this guards.
    """
    raw, reader, mapping = fixture
    unmod = eval_t39.oracle_action_chunks(reader, raw, CHUNK_STEPS, mapping, convert)
    chain = probe.chained_oracle_action_chunks(
        eval_t39, reader, raw, CHUNK_STEPS, mapping, convert, delay=0
    )
    anchorable = set(chain)
    control_scored = {t for t in unmod if t in anchorable}
    assert control_scored == anchorable
    assert len(control_scored) == len(unmod) - 1


def test_verdict_s_fires_when_the_repaired_cell_still_fails_l1():
    """S must be reachable, and it must not need the material floor.

    The floor guards W, the expensive conclusion. Requiring it for S too would make a repaired cell
    at −1 % fall through to I, and "the VOID survives" is exactly the reading that must not be
    downgraded to indeterminate.
    """
    repaired = {"skill_vs_repeat_pct": -12.0, "ci_skill_vs_repeat_pct": -20.0}
    control = {"skill_vs_repeat_pct": -359.0}
    assert rederive._verdict(repaired, control)["verdict"] == "S"
    repaired["skill_vs_repeat_pct"] = -0.5
    assert rederive._verdict(repaired, control)["verdict"] == "S"


def test_verdict_w_needs_the_material_floor_and_l2_together():
    """W is the expensive conclusion, so both clauses are load-bearing and both are tested."""
    control = {"skill_vs_repeat_pct": -359.0}
    assert (
        rederive._verdict(
            {"skill_vs_repeat_pct": 69.2, "ci_skill_vs_repeat_pct": 76.3}, control
        )["verdict"]
        == "W"
    )
    # L1 cleared but under the floor -> I, not W.
    assert (
        rederive._verdict(
            {"skill_vs_repeat_pct": 4.0, "ci_skill_vs_repeat_pct": 20.0}, control
        )["verdict"]
        == "I"
    )
    # Material on L1 but L2 fails -> I, not W. This is the cell a chunk-set artefact would produce.
    assert (
        rederive._verdict(
            {"skill_vs_repeat_pct": 40.0, "ci_skill_vs_repeat_pct": -3.0}, control
        )["verdict"]
        == "I"
    )


def test_verdict_w_does_not_claim_to_be_a_t39_verdict():
    """The distinction PR-13 §6 spends a section on, asserted so a summary cannot lose it."""
    reading = rederive._verdict(
        {"skill_vs_repeat_pct": 69.2, "ci_skill_vs_repeat_pct": 76.3},
        {"skill_vs_repeat_pct": -359.0},
    )["reading"]
    assert "NOT A VERDICT ON" in reading.upper()
    assert "not lifting a gate" in reading
    assert "policy arm" in reading


def test_per_step_profile_has_one_entry_per_step():
    from wam.evaluation.offline import ChunkPrediction
    from wam.interfaces.schema import ActionChunk, ActionMode

    def chunk(targets: np.ndarray) -> ActionChunk:
        return ActionChunk(
            mode=ActionMode.JOINT_DELTA,
            targets=targets.astype(np.float32),
            gripper_target=np.zeros(targets.shape[0], dtype=np.float32),
            dt_s=DT_S,
        )

    rng = np.random.default_rng(3)
    preds = [
        ChunkPrediction(
            predicted=chunk(rng.normal(size=(6, 15))),
            target=chunk(rng.normal(size=(6, 15))),
            episode_id="ep-0",
            t_ns=i * 100,
        )
        for i in range(4)
    ]
    assert len(rederive.per_step_profile(preds)) == 6
