"""Tests for ``--offload-text`` — the umT5 tower moved off the GPU from the scripts people run.

``WanI2VAdapter.offload`` was implemented and tested long before anything could reach it: until
now ``scripts/hf_job_wan_smoke.py`` was its only caller, so ``eval_t16``, ``dream``,
``train_t16_lora``, ``serve_policy`` and ``rollout`` all held ~11 GB of frozen text encoder
resident for the whole run after using it once. These tests cover the wiring that closes that.

Everything runs on CPU with no Wan weights. The adapter under test is the REAL
:class:`wam.backbones.wan_i2v.WanI2VAdapter`, built through ``test_wan_flow.make_backbone``'s
``attach()`` path with toy towers — so the chain being walked, the method being called and its
argument are the production ones, not a mock's idea of them.

TWO LAYERS, DELIBERATELY. :class:`TestTheChain` and :class:`TestTheOffloadCall` pin the shared
helper against that real adapter. :class:`TestTheScriptsAreWired` pins each script separately,
because the helper being correct says nothing about whether a script calls it — which was the
entire defect. Every wiring test has a paired assertion that the default run does NOT offload:
without it, a flag hard-wired to True would pass.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _extra in (_REPO_ROOT / "src", _REPO_ROOT / "scripts", _REPO_ROOT / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from test_wan_flow import joint_config, make_backbone

from wam.backbones.registry import build_backbone
from wam.backbones.tiny import TinyBackboneConfig
from wam.runtime.offload import (
    OFFLOAD_TEXT_HELP,
    advise_alloc_conf,
    distinct_instructions,
    offload_text_encoder,
    resolve_wan_adapter,
)
from wam.training.joint import JointTrainingConfig, JointWorldActionModel

_MOCK_D1 = _REPO_ROOT / "datasets" / "mock-d1"
_GR00T_APPLE = _REPO_ROOT / "datasets" / "gr00t-apple"
_JOINT_YAML = _REPO_ROOT / "configs" / "training" / "joint.yaml"
_GR00T_YAML = _REPO_ROOT / "configs" / "training" / "joint_gr00t.yaml"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = _load("eval_t16")
dr = _load("dream")
tw = _load("train_t16_lora")
sp = _load("serve_policy")
ro = _load("rollout")


class _Metadata:
    run_id = "fake-run"
    config_hash = "0" * 64
    checkpoint_ref = None
    dataset_snapshot_ref = None


class FakePolicy:
    """The shape ``load_joint_policy`` returns, as far as the offload wiring can see."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.metadata = _Metadata()
        self.camera = "front"
        self.device = "cpu"
        self.flow_steps = None


def _second_device() -> str | None:
    """A device that is NOT the CPU, or None. Needed to observe an offload as a real move.

    The flag exists for CUDA, but any non-CPU device makes the transition observable, and MPS is
    the one available on the machines this repo is developed on.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def spy_on_offload(adapter: Any, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    """Record every ``offload(...)`` call while still performing it. Returns the record."""
    calls: list[tuple[str, ...]] = []
    original = adapter.offload

    def recording(*components: str) -> None:
        calls.append(components)
        original(*components)

    monkeypatch.setattr(adapter, "offload", recording)
    return calls


def spy_on_helper(module: Any, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record what a script hands to ``offload_text_encoder``, without needing a Wan model."""
    seen: list[Any] = []
    monkeypatch.setattr(
        module, "offload_text_encoder", lambda obj, **kwargs: seen.append(obj) or obj
    )
    return seen


# -- the chain ---------------------------------------------------------------------------------


class TestTheChain:
    """``WanFlowBackbone`` keeps the towers out of the module tree, so the adapter is reachable
    only as a plain attribute. These pin the walk that finds it."""

    def test_a_bare_backbone_resolves_to_its_own_adapter(self) -> None:
        backbone = make_backbone()
        assert resolve_wan_adapter(backbone) is backbone.adapter

    def test_a_model_resolves_through_backbone(self) -> None:
        backbone = make_backbone()
        model = JointWorldActionModel(joint_config(), backbone=backbone)
        assert resolve_wan_adapter(model) is backbone.adapter

    def test_a_policy_resolves_through_model_and_backbone(self) -> None:
        """The full production chain: policy.model -> .backbone -> .adapter."""
        backbone = make_backbone()
        policy = FakePolicy(JointWorldActionModel(joint_config(), backbone=backbone))
        assert resolve_wan_adapter(policy) is backbone.adapter

    def test_the_adapter_is_not_in_the_module_tree_it_was_found_through(self) -> None:
        """Why the walk cannot just look for a submodule — and why offload() is reachable at all
        without dragging 10 GB into every state_dict()."""
        backbone = make_backbone()
        assert backbone.adapter not in list(backbone.modules())
        assert not any("adapter" in name for name in backbone.state_dict())

    def test_a_tiny_backbone_is_refused_loudly_not_silently_skipped(self) -> None:
        """The defect this guards against is a flag that appears to work and frees nothing."""
        tiny = build_backbone(TinyBackboneConfig())
        with pytest.raises(RuntimeError, match="needs a Wan-backed model"):
            resolve_wan_adapter(tiny)

    def test_the_refusal_names_the_chain_it_walked(self) -> None:
        config = JointTrainingConfig.from_yaml(_JOINT_YAML)
        tiny = build_backbone(config.backbone)
        policy = FakePolicy(JointWorldActionModel(config, backbone=tiny))
        with pytest.raises(RuntimeError) as err:
            resolve_wan_adapter(policy)
        message = str(err.value)
        assert "FakePolicy" in message and ".model ->" in message and ".backbone ->" in message

    def test_an_object_with_no_chain_at_all_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="needs a Wan-backed model"):
            resolve_wan_adapter(object())


# -- the call ----------------------------------------------------------------------------------


class TestTheOffloadCall:
    def test_it_offloads_exactly_the_text_encoder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backbone = make_backbone()
        calls = spy_on_offload(backbone.adapter, monkeypatch)
        offload_text_encoder(backbone)
        assert calls == [("text_encoder",)]

    def test_it_returns_the_adapter_it_acted_on(self) -> None:
        backbone = make_backbone()
        assert offload_text_encoder(backbone) is backbone.adapter

    def test_it_really_moves_the_tower(self) -> None:
        """Not just 'a method was called': the tower's parameters END UP on the CPU device.

        This asserts a TRANSITION, which is the whole point. The obvious version of this test —
        build the backbone, offload, assert the params are on CPU — passes against an ``offload``
        replaced by ``lambda *a: None``, because ``make_backbone`` already builds on CPU and
        ``.to("cpu")`` on a CPU module is a no-op. It proved nothing. The tower has to start
        somewhere else for the move to be observable at all.
        """
        accelerator = _second_device()
        if accelerator is None:
            pytest.skip("no non-CPU device here; the CPU-only twin below is what runs everywhere")
        backbone = make_backbone()
        tower = backbone.adapter._text_encoder
        # The transformer is the CONTROL, and it has to start on the accelerator too or the
        # control is as vacuous as the test this one replaced: asserting a CPU-built module is
        # on the CPU cannot fail. It starts resident, and it must STAY resident.
        control = backbone.adapter._transformer
        tower.to(accelerator)
        control.to(accelerator)
        assert {p.device.type for p in tower.parameters()} == {accelerator}, "setup failed"
        assert {p.device.type for p in control.parameters()} == {accelerator}, "setup failed"

        offload_text_encoder(backbone)

        assert {p.device.type for p in tower.parameters()} == {"cpu"}
        # An offload that swept every tower to the CPU would satisfy the line above while
        # costing the user the transformer they need resident — and would be slower, not faster.
        assert {p.device.type for p in control.parameters()} == {accelerator}

    def test_it_targets_the_text_tower_and_no_other(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The CPU-only twin of the test above: runs everywhere, and can still fail.

        Where that one needs a second device to observe a move, this one observes the CALL —
        which tower was asked to relocate, and where to. Between them the two cover 'it moved'
        and 'only that one moved' on any machine.
        """
        backbone = make_backbone()
        adapter = backbone.adapter
        moved: list[tuple[str, object]] = []
        for name in ("text_encoder", "image_encoder", "vae", "transformer"):
            module = getattr(adapter, f"_{name}")
            if module is None:
                continue
            monkeypatch.setattr(
                module, "to", lambda target, _n=name: moved.append((_n, target)), raising=False
            )

        offload_text_encoder(backbone)

        assert moved == [("text_encoder", "cpu")]

    def test_the_prompt_cache_survives_the_move(self) -> None:
        """(c) — the memo is keyed by the prompt STRING and holds a tensor on the adapter's own
        device, so moving the tower neither invalidates it nor strands it on the CPU."""
        backbone = make_backbone()
        offload_text_encoder(backbone)
        first = backbone.condition_text("move the apple to the plate")
        second = backbone.condition_text("move the apple to the plate")
        assert first is second, "a second encode of the same prompt must be a cache hit"
        assert list(backbone.adapter._text_cache) == ["move the apple to the plate"]
        assert first.device == torch.device(backbone.adapter.device)

    def test_a_second_distinct_prompt_costs_a_second_encode(self) -> None:
        """The cost model the training flag's help text is making a claim about."""
        backbone = make_backbone()
        offload_text_encoder(backbone)
        backbone.condition_text("one")
        backbone.condition_text("two")
        assert set(backbone.adapter._text_cache) == {"one", "two"}


# -- what the training flag's cost actually depends on -----------------------------------------


@pytest.mark.skipif(not _GR00T_APPLE.is_dir(), reason="datasets/gr00t-apple not present")
class TestDistinctInstructions:
    """The training help text claims the GR00T corpus is a single instruction. Measured here
    rather than asserted, because it is the whole basis for calling the CPU encode free."""

    def test_the_gr00t_corpus_is_one_instruction(self) -> None:
        episodes = sorted(p.parent for p in _GR00T_APPLE.glob("*/manifest.json"))
        assert distinct_instructions(episodes) == {"move the apple to the plate"}

    @pytest.mark.skipif(not _MOCK_D1.is_dir(), reason="datasets/mock-d1 not present")
    def test_mock_d1_has_several_which_is_what_makes_the_warning_fire(self) -> None:
        episodes = sorted(p.parent for p in _MOCK_D1.glob("*/manifest.json"))
        assert len(distinct_instructions(episodes)) > 1


# -- the allocator hint --------------------------------------------------------------------------


class TestAllocConfHint:
    def test_it_fires_on_cuda_when_nothing_is_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
        assert advise_alloc_conf("cuda") is True

    def test_it_stays_quiet_on_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
        assert advise_alloc_conf("cpu") is False

    def test_it_does_not_nag_a_user_who_already_set_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        assert advise_alloc_conf("cuda:0") is False

    def test_it_never_sets_the_variable_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A hint, not a setting: it must not change allocator behaviour behind the run record."""
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
        advise_alloc_conf("cuda")
        import os

        assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ


# -- one name for one thing ----------------------------------------------------------------------


class TestOneNameForOneThing:
    def test_every_script_spells_the_flag_the_same_way(self) -> None:
        parsed = {
            "eval_t16": ev._parse_args(["--run-dir", "x"]),
            "dream": dr.parse_args([]),
            "rollout": ro._parse_args([]),
            "serve_policy": sp._parse_args(["--dummy"]),
            "train_t16_lora": tw._parse_args(
                ["--training-config", "c", "--dataset", "d", "--out-dir", "o"]
            ),
        }
        for name, args in parsed.items():
            assert hasattr(args, "offload_text"), f"{name} has no --offload-text"

    def test_the_flag_is_off_by_default_everywhere(self) -> None:
        """The hard repo convention: archived runs must stay reproducible without edits."""
        assert ev._parse_args(["--run-dir", "x"]).offload_text is False
        assert dr.parse_args([]).offload_text is False
        assert ro._parse_args([]).offload_text is False
        assert sp._parse_args(["--dummy"]).offload_text is False
        assert (
            tw._parse_args(
                ["--training-config", "c", "--dataset", "d", "--out-dir", "o"]
            ).offload_text
            is False
        )

    def test_passing_it_turns_it_on_everywhere(self) -> None:
        assert ev._parse_args(["--run-dir", "x", "--offload-text"]).offload_text is True
        assert dr.parse_args(["--offload-text"]).offload_text is True
        assert (
            ro._parse_args(["--policy", "joint", "--offload-text"]).offload_text is True
        )
        assert (
            sp._parse_args(["--checkpoint", "c", "--joint", "--offload-text"]).offload_text is True
        )
        assert (
            tw._parse_args(
                ["--training-config", "c", "--dataset", "d", "--out-dir", "o", "--offload-text"]
            ).offload_text
            is True
        )

    def test_the_shared_help_text_is_the_smoke_tests_wording(self) -> None:
        """``hf_job_wan_smoke.py`` owns the original phrasing; drift would make them two flags."""
        source = (_REPO_ROOT / "scripts" / "hf_job_wan_smoke.py").read_text(encoding="utf-8")
        assert OFFLOAD_TEXT_HELP in source

    @pytest.mark.parametrize(
        "module, argv",
        [
            (sp, ["--checkpoint", "c", "--offload-text"]),
            (ro, ["--policy", "checkpoint", "--offload-text"]),
        ],
    )
    def test_it_is_refused_where_there_is_no_umt5_tower(self, module: Any, argv: list[str]) -> None:
        """Refused rather than ignored: an accepted no-op flag promises relief it cannot give."""
        parse = module._parse_args
        with pytest.raises(SystemExit):
            parse(argv)


# -- the scripts ---------------------------------------------------------------------------------


@pytest.fixture
def wan_policy() -> FakePolicy:
    """A policy whose chain ends in a REAL WanI2VAdapter, on CPU, with no Wan weights."""
    return FakePolicy(JointWorldActionModel(joint_config(), backbone=make_backbone()))


class TestTheScriptsAreWired:
    """One test per entry point. Each asserts BOTH that the flag reaches the offload and that
    the default run leaves the tower alone — the second half is what stops a hard-wired True
    from passing, and the first is what fails if the wiring is deleted."""

    # -- serve_policy and rollout: full strength, no helper patching --------------------------

    def test_serve_policy_offloads_the_real_adapter(
        self, monkeypatch: pytest.MonkeyPatch, wan_policy: FakePolicy
    ) -> None:
        from wam.runtime import policies

        monkeypatch.setattr(policies, "load_joint_policy", lambda *a, **k: wan_policy)
        calls = spy_on_offload(wan_policy.model.backbone.adapter, monkeypatch)
        sp._build_joint_policy(Path("ckpt.safetensors"), "cpu", None, None, True)
        assert calls == [("text_encoder",)]

    def test_serve_policy_leaves_the_tower_alone_by_default(
        self, monkeypatch: pytest.MonkeyPatch, wan_policy: FakePolicy
    ) -> None:
        from wam.runtime import policies

        monkeypatch.setattr(policies, "load_joint_policy", lambda *a, **k: wan_policy)
        calls = spy_on_offload(wan_policy.model.backbone.adapter, monkeypatch)
        sp._build_joint_policy(Path("ckpt.safetensors"), "cpu", None, None)
        assert calls == []

    def test_rollout_offloads_the_real_adapter(
        self, monkeypatch: pytest.MonkeyPatch, wan_policy: FakePolicy
    ) -> None:
        from wam.runtime import policies

        monkeypatch.setattr(policies, "load_joint_policy", lambda *a, **k: wan_policy)
        calls = spy_on_offload(wan_policy.model.backbone.adapter, monkeypatch)
        args = ro._parse_args(["--policy", "joint", "--offload-text"])
        ro._build_policy(args, None, 0.05)
        assert calls == [("text_encoder",)]

    def test_rollout_leaves_the_tower_alone_by_default(
        self, monkeypatch: pytest.MonkeyPatch, wan_policy: FakePolicy
    ) -> None:
        from wam.runtime import policies

        monkeypatch.setattr(policies, "load_joint_policy", lambda *a, **k: wan_policy)
        calls = spy_on_offload(wan_policy.model.backbone.adapter, monkeypatch)
        ro._build_policy(ro._parse_args(["--policy", "joint"]), None, 0.05)
        assert calls == []

    # -- dream: end to end over a real dataset -------------------------------------------------

    @pytest.mark.skipif(not _GR00T_APPLE.is_dir(), reason="datasets/gr00t-apple not present")
    @pytest.mark.parametrize("flag, expected", [(["--offload-text"], 1), ([], 0)])
    def test_dream_offloads_only_when_asked(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        flag: list[str],
        expected: int,
    ) -> None:
        from wam.runtime import policies

        config = JointTrainingConfig.from_yaml(_GR00T_YAML)
        model = JointWorldActionModel(config)
        model.eval()
        monkeypatch.setattr(policies, "load_joint_policy", lambda *a, **k: FakePolicy(model))
        seen = spy_on_helper(dr, monkeypatch)
        checkpoint = tmp_path / "model.safetensors"
        checkpoint.write_bytes(b"unused: the loader is patched")
        rc = dr.main(
            [
                "--dataset", str(_GR00T_APPLE),
                "--camera", config.camera,
                "--checkpoint", str(checkpoint),
                "--out", str(tmp_path / "out"),
                "--episodes", "1",
                "--windows-per-episode", "3",
                "--steps", "2",
                *flag,
            ]
        )  # fmt: skip
        assert rc == 0
        assert len(seen) == expected
        if expected:
            assert seen[0].model is model, "dream must offload through the loaded policy"

    @pytest.mark.skipif(not _GR00T_APPLE.is_dir(), reason="datasets/gr00t-apple not present")
    def test_dream_labels_the_mode_next_to_its_peak_vram(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A peak measured with umT5 parked on the CPU is not comparable with the archived
        32.47/32.54 GB runs, so the report has to say which mode produced it."""
        from wam.runtime import policies

        config = JointTrainingConfig.from_yaml(_GR00T_YAML)
        model = JointWorldActionModel(config)
        monkeypatch.setattr(policies, "load_joint_policy", lambda *a, **k: FakePolicy(model))
        spy_on_helper(dr, monkeypatch)
        checkpoint = tmp_path / "model.safetensors"
        checkpoint.write_bytes(b"unused")
        rc = dr.main(
            [
                "--dataset", str(_GR00T_APPLE),
                "--camera", config.camera,
                "--checkpoint", str(checkpoint),
                "--out", str(tmp_path / "out"),
                "--episodes", "1",
                "--windows-per-episode", "3",
                "--steps", "2",
                "--offload-text",
            ]
        )  # fmt: skip
        assert rc == 0
        report = json.loads((tmp_path / "out" / "dream.json").read_text(encoding="utf-8"))
        assert report["info"]["args"]["offload_text"] is True

    # -- eval_t16 and train_t16_lora: end to end over the CPU fixtures -------------------------

    @pytest.mark.skipif(not _MOCK_D1.is_dir(), reason="datasets/mock-d1 not present")
    @pytest.mark.parametrize("flag, expected", [(["--offload-text"], 1), ([], 0)])
    def test_train_offloads_only_when_asked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flag: list[str], expected: int
    ) -> None:
        seen = spy_on_helper(tw, monkeypatch)
        rc = tw.main(
            [
                "--training-config", str(_JOINT_YAML),
                "--dataset", str(_MOCK_D1),
                "--out-dir", str(tmp_path / "run"),
                "--steps", "2",
                "--batch-size", "4",
                "--device", "cpu",
                *flag,
            ]
        )  # fmt: skip
        assert rc == 0
        assert len(seen) == expected
        if expected:
            # It must hand over the BACKBONE, and do it after JointTrainer.__init__ has run its
            # .to(device) — offloading before that move would be undone by it
            # (WanFlowBackbone._apply forwards device moves to the held towers).
            handed_over = seen[0]
            assert isinstance(handed_over, torch.nn.Module)
            assert hasattr(handed_over, "encode_video"), "not a FlowBackbone"
            assert not hasattr(handed_over, "backbone"), "that is the model, not the backbone"

    @pytest.mark.skipif(not _MOCK_D1.is_dir(), reason="datasets/mock-d1 not present")
    def test_train_warns_when_the_instruction_varies(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """mock-d1 carries four distinct instructions, so the CPU umT5 forward is paid four
        times rather than once. Silence here would be the performance cliff this flag must not
        ship."""
        spy_on_helper(tw, monkeypatch)
        rc = tw.main(
            [
                "--training-config", str(_JOINT_YAML),
                "--dataset", str(_MOCK_D1),
                "--out-dir", str(tmp_path / "run"),
                "--steps", "2",
                "--batch-size", "4",
                "--device", "cpu",
                "--offload-text",
            ]
        )  # fmt: skip
        assert rc == 0
        out = capsys.readouterr().out
        assert "WARNING --offload-text" in out
        assert "distinct instructions" in out

    @pytest.mark.skipif(not _MOCK_D1.is_dir(), reason="datasets/mock-d1 not present")
    def test_train_on_a_tiny_backbone_refuses_instead_of_pretending(self, tmp_path: Path) -> None:
        """Unpatched: the real helper runs, finds no Wan adapter, and says so."""
        with pytest.raises(RuntimeError, match="needs a Wan-backed model"):
            tw.main(
                [
                    "--training-config", str(_JOINT_YAML),
                    "--dataset", str(_MOCK_D1),
                    "--out-dir", str(tmp_path / "run"),
                    "--steps", "2",
                    "--batch-size", "4",
                    "--device", "cpu",
                    "--offload-text",
                ]
            )  # fmt: skip

    @pytest.mark.skipif(not _MOCK_D1.is_dir(), reason="datasets/mock-d1 not present")
    @pytest.mark.parametrize("flag, expected", [(["--offload-text"], 1), ([], 0)])
    def test_eval_offloads_only_when_asked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flag: list[str], expected: int
    ) -> None:
        run_dir = tmp_path / "run"
        holdout = tmp_path / "holdout.txt"
        holdout.write_text("d1-0006\nd1-0007\n")
        assert (
            tw.main(
                [
                    "--training-config", str(_JOINT_YAML),
                    "--dataset", str(_MOCK_D1),
                    "--out-dir", str(run_dir),
                    "--exclude-episodes", str(holdout),
                    "--steps", "2",
                    "--batch-size", "4",
                    "--device", "cpu",
                ]
            ) == 0
        )  # fmt: skip
        # The I-8 disjointness proof needs an external witness naming the training split; it is
        # unrelated to this flag, but the evaluator refuses to score without it.
        witness = tmp_path / "train-episodes.txt"
        witness.write_text("\n".join(f"d1-{i:04d}" for i in range(6)) + "\n")
        seen = spy_on_helper(ev, monkeypatch)
        rc = ev.main(
            [
                "--run-dir", str(run_dir),
                "--dataset", str(_MOCK_D1),
                "--holdout", str(holdout),
                "--train-episodes", str(witness),
                "--device", "cpu",
                "--out", str(tmp_path / "eval"),
                *flag,
            ]
        )  # fmt: skip
        assert rc == 0
        assert len(seen) == expected
        if expected:
            assert hasattr(seen[0], "model"), "eval must offload through the loaded policy"
