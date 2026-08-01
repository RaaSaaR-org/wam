"""Tests for `scripts/probe_velocity_head.py` — measuring the velocity head's gain in t.

The script's whole output is two statistics about a checkpoint nobody will re-derive by hand, and
its finding is a NEGATIVE one: "this head is time-blind". A negative finding is exactly the shape
of result a broken probe produces for free — a statistic that is always ~0, a Jacobian taken
through the wrong argument, a state dict that never loaded. So the tests are built around
calibration points where the right answer is known in advance, not around "does it print numbers":

  S_t == 0 exactly     A head with its t column ZEROED is time-blind BY CONSTRUCTION, and the
                       statistic has to bottom out at exactly zero on it — not "small".

  S_t on the correct    The analytically correct field v* = c/(1-t) is built explicitly and fed
  field                through the same code path. Without this, a probe that returns "time-blind"
                       for EVERY input passes the zeroed-column test and every real checkpoint,
                       and the finding is an artifact of the tool. The expected value is not a
                       magic number either: it is `ideal_t_flatness(grid)`, which the same field
                       must reproduce to float tolerance.

  a KNOWN gain         `ghat` is asserted against a head whose weights were built to realize
                       v = -g z + const for a chosen g, so "recovers the gain" is checked rather
                       than assumed. A Jacobian taken w.r.t. the wrong input, or with the sign
                       flipped, cannot survive it.

  strict=True          A checkpoint missing one velocity_head tensor must fail loudly. Loaded
                       non-strictly, the missing tensor stays at its random init — and a random
                       MLP measures as t-flat too, so the script would report its headline
                       finding from a head that never trained. That is the one failure mode with
                       no symptom in the output.

Everything runs on CPU against a synthetic checkpoint built in a tmp dir from the shipped
`configs/training/joint.yaml`, the fixture pattern `test_rescore_archived.py` established. In
particular nothing here reads `runs/t16-lora-seed0`: it is a 315 MB artifact that is not in the
repo, and a test suite that skips when it is absent is a test suite that does not run.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_JOINT_YAML = _REPO_ROOT / "configs" / "training" / "joint.yaml"

#: The gain the "known gain" head is built to realize. Well above 1 so a probe that returned the
#: identity, a zero, or 1/(1-t) at any probed t would fail it.
KNOWN_GAIN = 2.75


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pv = _load("probe_velocity_head")

from wam.training._utils import CHECKPOINT_CONFIG_KEY, CHECKPOINT_METADATA_KEY
from wam.training.joint import ActionVelocityHead, JointTrainingConfig


@pytest.fixture(scope="module")
def config() -> JointTrainingConfig:
    """The shipped joint config: latent 32, features 64, one hidden layer of 64.

    Small enough for a Jacobian per sample to be free, and REAL — the head under test is the one
    a `train_t16_lora` run would checkpoint, not a stand-in with convenient dimensions.
    """
    return JointTrainingConfig.from_yaml(_JOINT_YAML)


def _dims(config: JointTrainingConfig) -> tuple[int, int]:
    return config.action_encoder.latent_dim, config.backbone.feature_dim


def _head(config: JointTrainingConfig, seed: int = 0) -> ActionVelocityHead:
    torch.manual_seed(seed)
    latent_dim, feature_dim = _dims(config)
    head = ActionVelocityHead(latent_dim, feature_dim, tuple(config.velocity_hidden_dims))
    head.eval()
    return head


def _write_checkpoint(
    path: Path,
    head: ActionVelocityHead,
    config: JointTrainingConfig,
    *,
    drop: str | None = None,
    embed_config: bool = True,
) -> Path:
    """A checkpoint shaped like a real one: `velocity_head.*` tensors + the embedded config.

    Other branches (`action_recon.*` here) are written too, because "pick out only the head's
    tensors" is part of what the loader has to do — a checkpoint holding nothing else would let a
    load-everything implementation pass.
    """
    from safetensors.torch import save_file

    from wam.interfaces import RunMetadata

    state = {f"{pv.VELOCITY_PREFIX}{k}": v.contiguous() for k, v in head.state_dict().items()}
    if drop is not None:
        assert state.pop(f"{pv.VELOCITY_PREFIX}{drop}", None) is not None, drop
    state["action_recon.0.weight"] = torch.zeros(4, 4)

    metadata = {
        CHECKPOINT_METADATA_KEY: json.dumps(
            RunMetadata.create("probe-test", config, git_commit="0" * 40).to_dict(), sort_keys=True
        )
    }
    if embed_config:
        metadata[CHECKPOINT_CONFIG_KEY] = config.model_dump_json()
    save_file(state, str(path), metadata=metadata)
    return path


class IdealFlowField(torch.nn.Module):
    """The field a rectified-flow sampler actually needs, as an `ActionVelocityHead` stand-in.

    With `x1 = 0` (any fixed clean latent does; the statistic is scale-free) the correct velocity
    at `z_t` is `v* = (x1 - z_t)/(1 - t) = -z_t/(1 - t)`. Its gain is `1/(1-t)` exactly, so it
    pins BOTH statistics at once: `t_flatness` must return `ideal_t_flatness(grid)` and
    `latent_gain` must return `1/(1-t)` with no off-diagonal mass.

    Deliberately not an `ActionVelocityHead` with fitted weights — the point is a field whose
    answer is known analytically, so a disagreement is the probe's, never the fixture's.
    """

    def forward(self, z_t: torch.Tensor, pooled_features: torch.Tensor, t: torch.Tensor):
        return -z_t / (1.0 - t.reshape(-1, 1, 1))


def _known_gain_head(
    config: JointTrainingConfig, gain: float, *, epsilon: float = 1e-4, seed: int = 0
) -> ActionVelocityHead:
    """An `ActionVelocityHead` whose weights realize `v = -gain * z + const`, near-exactly.

    The head is `Linear -> LayerNorm -> GELU -> Linear`, and LayerNorm is scale-INVARIANT in its
    centered input, so no choice of weights makes the composition globally linear in z. It can be
    made linear to first order, which is what a Jacobian measures: the first layer sends z in as a
    small perturbation `epsilon * P z` riding on a large constant bias, so everything between the
    two `Linear`s operates at a fixed point and contributes its (constant) Jacobian `J` there. The
    output layer is then the exact left inverse of that composed map, scaled by `-gain`:

        dv/dz  =  W2 @ J @ (epsilon P)  =  -gain * pinv(M) @ M  =  -gain * I

    `J` is taken by autograd rather than derived by hand, so the fixture cannot disagree with
    torch's own LayerNorm/GELU about what their derivative is. Residual curvature is O(epsilon).
    """
    latent_dim, feature_dim = _dims(config)
    head = _head(config, seed=seed)
    hidden = config.velocity_hidden_dims[0]
    first, last = head.mlp[0], head.mlp[-1]
    middle = torch.nn.Sequential(*list(head.mlp)[1:-1])  # LayerNorm -> GELU

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        weight = torch.zeros(hidden, latent_dim + feature_dim + 1)
        # Features and t get NO first-layer weight: the map has to be a pure function of z for
        # "the recovered gain is `gain`" to be a statement about the probe and not about a draw.
        weight[:, :latent_dim] = epsilon * torch.randn(hidden, latent_dim, generator=generator)
        first.weight.copy_(weight)
        first.bias.copy_(torch.randn(hidden, generator=generator))

        jacobian = torch.autograd.functional.jacobian(middle, first.bias.clone())
        composed = jacobian @ weight[:, :latent_dim]  # [hidden, latent] d(middle)/dz
        last.weight.copy_(-gain * torch.linalg.pinv(composed))
        last.bias.zero_()
    head.eval()
    return head


# -- the calibration points --------------------------------------------------------------------


def test_a_zeroed_t_column_scores_exactly_zero_flatness(config: JointTrainingConfig) -> None:
    """The bottom of the scale, and it has to be exact rather than small.

    Zeroing `mlp.0.weight`'s last column removes the timestep from the computation entirely, so
    every t on the grid produces the bit-identical output and the spread around their mean is
    0.0, not 1e-9. An implementation that leaked t in through some other path — or that swept
    something else while claiming to sweep t — cannot produce exactly zero here.
    """
    latent_dim, feature_dim = _dims(config)
    head = _head(config)
    with torch.no_grad():
        head.mlp[0].weight[:, -1].zero_()

    score, norm = pv.t_flatness(
        head,
        latent_dim,
        feature_dim,
        pv.sampler_grid(pv.DEFAULT_STEPS),
        chunk_steps=config.head.num_steps,
        feature_scale=1.0,
        samples=4,
        seed=0,
    )

    assert score == 0.0, "a head that cannot see t must be measured as perfectly time-blind"
    assert norm > 0.0, "the head still has an output — the zero is flatness, not a dead head"


def test_the_correct_field_scores_the_analytic_flatness(config: JointTrainingConfig) -> None:
    """The top of the scale — the test that makes the finding falsifiable.

    Without it, a probe that reports "time-blind" for every input passes the zeroed-column test
    and every real checkpoint, and the headline result is an artifact of the tool rather than a
    property of the head. The expected value is `ideal_t_flatness(grid)`, derived independently
    of the sweep, so the two have to agree on a field neither of them fitted.
    """
    latent_dim, feature_dim = _dims(config)
    grid = pv.sampler_grid(pv.DEFAULT_STEPS)

    score, _ = pv.t_flatness(
        IdealFlowField(),
        latent_dim,
        feature_dim,
        grid,
        chunk_steps=config.head.num_steps,
        feature_scale=1.0,
        samples=4,
        seed=0,
    )

    assert score == pytest.approx(pv.ideal_t_flatness(grid), rel=1e-4)
    assert score > 5.0, "the correct field is strongly t-dependent on the deployed grid"


def test_the_correct_fields_gain_is_recovered_as_one_over_one_minus_t(
    config: JointTrainingConfig,
) -> None:
    """`ghat` must track `1/(1-t)` on the field that has it — including at the far end of the
    grid, where the gain is 32x its value at t=0. A probe that silently evaluated at a fixed t,
    or that averaged over the grid, would return one number for all three."""
    latent_dim, feature_dim = _dims(config)
    grid = pv.sampler_grid(pv.DEFAULT_STEPS)

    for t in (grid[0], grid[len(grid) // 2], grid[-1]):
        ghat, off_ratio = pv.latent_gain(
            IdealFlowField(),
            latent_dim,
            feature_dim,
            t=t,
            feature_scale=1.0,
            samples=2,
            seed=1,
        )
        assert ghat == pytest.approx(1.0 / (1.0 - t), rel=1e-5)
        assert off_ratio == pytest.approx(0.0, abs=1e-6), "the field is exactly diagonal"


def test_ghat_recovers_a_known_gain_from_a_heads_weights(config: JointTrainingConfig) -> None:
    """The Jacobian probe against a head built to have gain `KNOWN_GAIN`, weights and all.

    Sign, argument and normalization all have to be right at once: `-dv/dz` averaged over the
    diagonal is only `KNOWN_GAIN` if the derivative is taken w.r.t. the LATENT (not the features,
    not t), negated (the sampler steps `z <- z + v dt`, so a contracting field has a negative
    Jacobian) and averaged over the diagonal rather than summed.
    """
    latent_dim, feature_dim = _dims(config)
    head = _known_gain_head(config, KNOWN_GAIN)

    ghat, off_ratio = pv.latent_gain(
        head, latent_dim, feature_dim, t=0.5, feature_scale=1.0, samples=8, seed=1
    )

    assert ghat == pytest.approx(KNOWN_GAIN, rel=1e-3)
    assert off_ratio < 1e-2, (
        "the constructed map is diagonal — a large ratio means the left inverse did not land, "
        "so ghat would be a mean over something that is not a gain"
    )


def test_the_known_gain_head_is_flat_in_t_and_the_probe_says_so(
    config: JointTrainingConfig,
) -> None:
    """The constructed head takes no t at all, so it is the archived head's failure mode in pure
    form: a real, weight-realized, strictly contracting field with a gain that does not move."""
    latent_dim, feature_dim = _dims(config)
    head = _known_gain_head(config, KNOWN_GAIN)
    grid = pv.sampler_grid(pv.DEFAULT_STEPS)

    gains = [
        pv.latent_gain(head, latent_dim, feature_dim, t=t, feature_scale=1.0, samples=4, seed=1)[0]
        for t in (grid[0], grid[-1])
    ]

    assert gains[0] == pytest.approx(gains[1], rel=1e-6)
    assert 1.0 / (1.0 - grid[-1]) > 10.0 * gains[0], "the correct gain at grid end is far above it"


def test_ideal_t_flatness_is_a_property_of_the_grid_alone() -> None:
    """It is printed as the reference the measurement is read against, so it may not drift with
    anything but the grid: on the deployed 32-step grid the mean gain is the harmonic number
    H_32 and the spread is `(32 - H_32)/H_32`."""
    grid = pv.sampler_grid(32)
    harmonic = sum(1.0 / k for k in range(1, 33))

    assert pv.ideal_t_flatness(grid) == pytest.approx((32.0 - harmonic) / harmonic, rel=1e-9)
    assert pv.ideal_t_flatness(pv.sampler_grid(1)) == 0.0  # one t: nothing to be flat against


def test_the_grid_is_the_deployed_samplers(config: JointTrainingConfig) -> None:
    """`sample_action_chunk` integrates over `{t0, t0+dt, ..., 1-dt}` and never reaches t=1,
    because training draws t from `torch.rand` whose support is [0, 1). Probing a t the head was
    never evaluated at would measure extrapolation and report it as the deployed field."""
    grid = pv.sampler_grid(32)

    assert len(grid) == 32
    assert grid[0] == 0.0
    assert grid[-1] == pytest.approx(31.0 / 32.0)
    assert max(grid) < 1.0
    assert pv.sampler_grid(4, t0=0.5) == pytest.approx((0.5, 0.625, 0.75, 0.875))


# -- loading: what the script refuses to guess at ------------------------------------------------


def test_loads_only_the_velocity_head_from_a_checkpoint(
    config: JointTrainingConfig, tmp_path: Path
) -> None:
    """Round trip: the restored head must be the checkpointed one tensor for tensor, and the
    dimensions must come out of the embedded config rather than out of a guess at the tensors."""
    head = _head(config, seed=3)
    path = _write_checkpoint(tmp_path / "model.safetensors", head, config)

    restored, restored_config = pv.load_velocity_head(path)

    for name, tensor in head.state_dict().items():
        assert torch.equal(restored.state_dict()[name], tensor), name
    assert restored_config.action_encoder.latent_dim == config.action_encoder.latent_dim
    assert restored_config.backbone.feature_dim == config.backbone.feature_dim
    assert not restored.training, (
        "the head must come back in eval mode — probing it in train mode measures a "
        "configuration the sampler never runs"
    )


def test_a_missing_velocity_head_tensor_is_refused_not_partially_loaded(
    config: JointTrainingConfig, tmp_path: Path
) -> None:
    """The one failure mode with no symptom in the output.

    `strict=False` would leave `mlp.0.weight` at its random init and the script would go on to
    report a confident, plausible, entirely fictional t-flat gain — the same headline it prints
    for a real checkpoint. So the load must fail loudly, and the message must say why rather than
    surfacing torch's bare key list.
    """
    path = _write_checkpoint(
        tmp_path / "model.safetensors", _head(config), config, drop="mlp.0.weight"
    )

    with pytest.raises(SystemExit, match="Refusing to load the subset non-strictly"):
        pv.load_velocity_head(path)


def test_a_checkpoint_without_the_embedded_config_exits_with_a_message(
    config: JointTrainingConfig, tmp_path: Path
) -> None:
    """The head's latent/feature column split is not recoverable from the tensors — `mlp.0.weight`
    is one 3105-wide block — so without the config there is no head to build, only a guess."""
    path = _write_checkpoint(
        tmp_path / "model.safetensors", _head(config), config, embed_config=False
    )

    with pytest.raises(SystemExit, match=CHECKPOINT_CONFIG_KEY):
        pv.load_velocity_head(path)


def test_a_checkpoint_without_a_velocity_head_exits_with_a_message(
    config: JointTrainingConfig, tmp_path: Path
) -> None:
    """An action-only run (T-13) trains no flow branch at all. There is no timestep conditioning
    in it to probe, and saying so beats an empty table."""
    from safetensors.torch import save_file

    path = tmp_path / "action-only.safetensors"
    save_file(
        {"head.mlp.0.weight": torch.zeros(2, 2)},
        str(path),
        metadata={CHECKPOINT_CONFIG_KEY: config.model_dump_json()},
    )

    with pytest.raises(SystemExit, match="no velocity_head"):
        pv.load_velocity_head(path)


def test_a_missing_checkpoint_exits_and_a_step_dir_resolves(
    config: JointTrainingConfig, tmp_path: Path
) -> None:
    """`--checkpoint` takes a run/step dir or the file inside it, the same shape
    `check_action_latent.py` accepts, so the two can be pointed at one path."""
    step_dir = tmp_path / "step-000010"
    step_dir.mkdir()
    _write_checkpoint(step_dir / pv.MODEL_FILENAME, _head(config), config)

    assert pv.resolve_model_path(step_dir) == step_dir / pv.MODEL_FILENAME
    assert pv.resolve_model_path(step_dir / pv.MODEL_FILENAME) == step_dir / pv.MODEL_FILENAME
    with pytest.raises(SystemExit, match="not a restorable checkpoint"):
        pv.resolve_model_path(tmp_path / "nowhere")


# -- end to end ----------------------------------------------------------------------------------


def test_the_cli_probes_a_checkpoint_and_records_the_known_gain(
    config: JointTrainingConfig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole path, on a checkpoint whose answer is known: build the `KNOWN_GAIN` head, save
    it, and check that the number reaching the JSON record — through arg parsing, resolution,
    strict loading and both statistics — is still that gain at every feature scale and every t."""
    path = _write_checkpoint(
        tmp_path / "model.safetensors", _known_gain_head(config, KNOWN_GAIN), config
    )
    record_path = tmp_path / "probe.json"

    argv = [
        "--checkpoint", str(path),
        "--samples", "4",
        "--steps", "8",
        "--feature-scale", "0.5",
        "--feature-scale", "2.0",
        "--json", str(record_path),
    ]  # fmt: skip

    assert pv.main(argv) == 0

    record = json.loads(record_path.read_text())
    assert record["checkpoint"] == str(path)
    assert record["latent_dim"] == config.action_encoder.latent_dim
    assert record["feature_dim"] == config.backbone.feature_dim
    assert record["chunk_steps"] == config.head.num_steps
    assert [entry["feature_scale"] for entry in record["t_flatness"]] == [0.5, 2.0]
    assert record["grid"] == {"steps": 8, "t_first": 0.0, "t_last": pytest.approx(0.875)}

    for entry in record["gain"]:
        assert entry["ghat"] == pytest.approx(KNOWN_GAIN, rel=1e-3), entry
        assert entry["ideal_gain"] == pytest.approx(1.0 / (1.0 - entry["t"]))
    # The head takes no t, so its flatness is a hard zero — and the reference it is printed
    # against is not.
    assert all(entry["s_t"] == 0.0 for entry in record["t_flatness"])
    assert record["ideal_t_flatness"] > 1.0

    printed = capsys.readouterr().out
    assert "first-layer weight blocks" in printed
    assert "ideal 1/(1-t)" in printed, "the reference has to be printed beside the measurement"
    assert "reading:" in printed


def test_the_first_layer_blocks_split_the_columns_the_way_the_head_reads_them(
    config: JointTrainingConfig,
) -> None:
    """t is ONE column of `mlp.0.weight`; latent and features are blocks. Mis-slicing them is
    invisible in the total norm and would silently reassign the head's budget in the report."""
    latent_dim, feature_dim = _dims(config)
    head = _head(config)
    with torch.no_grad():
        head.mlp[0].weight.zero_()
        head.mlp[0].weight[:, latent_dim + feature_dim] = 3.0  # the t column only

    blocks = pv.first_layer_blocks(head, latent_dim, feature_dim)

    assert blocks["latent"] == (latent_dim, 0.0)
    assert blocks["feats"] == (feature_dim, 0.0)
    hidden = config.velocity_hidden_dims[0]
    assert blocks["t"][0] == 1
    assert blocks["t"][1] == pytest.approx(3.0 * hidden**0.5)


def test_first_layer_blocks_refuse_a_head_that_does_not_match_the_config(
    config: JointTrainingConfig,
) -> None:
    """The column split comes from the config, so a config that does not describe this head would
    slice the blocks at the wrong offsets and report a plausible, wrong budget."""
    latent_dim, feature_dim = _dims(config)

    with pytest.raises(SystemExit, match="columns"):
        pv.first_layer_blocks(_head(config), latent_dim + 1, feature_dim)
