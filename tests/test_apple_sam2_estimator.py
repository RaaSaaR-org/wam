"""PR-08 §4's estimator adapter: the contract, and every way it is asked to be wrong.

None of this loads a weight, opens a socket, or touches a GPU. ``sam2``, ``transformers`` and
``huggingface_hub`` are stubbed into ``sys.modules``, which is the only way this file could exist at
all: no SAM 2, GroundingDINO or Depth-Anything checkpoint is staged anywhere in this project, so the
number the adapter produces is not testable here — but every way of producing a WRONG one is, and
that is the half that sets a gate. Real ``torch``, ``numpy`` and ``PIL`` are used, so the shapes and
dtypes the tests assert on are the shapes and dtypes the real path would carry.

The hub stub is installed for EVERY test, not only the ones about caching: the adapter resolves the
SAM 2 checkpoint file through ``hf_hub_download`` itself now, and a stub that was optional would let
a green test suite make a real request.
"""

from __future__ import annotations

import importlib
import pathlib
import re
import sys
import types

import numpy as np
import pytest
import torch

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

MODULE = "estimators.apple_sam2"
_ADAPTER_SOURCE = _REPO / "scripts" / "estimators" / "apple_sam2.py"

#: The pins the orchestrator read off the HF API on 2026-08-22 and locked. Restated here rather than
#: imported from the module under test on purpose: a test that reads the constant it is checking
#: cannot notice the constant changing, and these three commits are what the staging job stages and
#: what the committed gate number will claim to have been measured with.
CONTRACT_PINS = {
    "facebook/sam2-hiera-large": "e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251",
    "IDEA-Research/grounding-dino-base": "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf": (
        "d2fc6a93601aabb1139a3bf0ebfcb4e89c67817f"
    ),
}


# -- stubs -------------------------------------------------------------------------------------------
#
# Shaped after the real call sites rather than after what the adapter happens to call, so that a
# refactor which stops calling `post_process_grounded_object_detection`, or starts loading per frame,
# fails here instead of passing.


class _StubPredictor:
    """SAM 2 image predictor. Records the box it was prompted with — that is a tested behaviour."""

    def __init__(self, sam_model=None) -> None:
        self.sam_model = sam_model
        self.image_hw: tuple[int, int] | None = None
        self.boxes_seen: list[np.ndarray] = []

    @classmethod
    def from_pretrained(cls, model_id, device=None):
        raise AssertionError(
            "SAM2ImagePredictor.from_pretrained cannot carry a revision (sam2's _hf_download "
            "calls hf_hub_download with none), so the adapter must not use it."
        )

    def set_image(self, frame):
        self.image_hw = frame.shape[:2]

    def predict(self, box=None, multimask_output=False):
        assert self.image_hw is not None, "predict before set_image"
        assert multimask_output is False, "the contract returns ONE mask"
        self.boxes_seen.append(np.asarray(box).reshape(-1))
        h, w = self.image_hw
        mask = np.zeros((1, h, w), dtype=np.float32)
        x0, y0, x1, y1 = np.asarray(box).reshape(-1).astype(int)
        mask[0, y0:y1, x0:x1] = 1.0
        return mask, np.asarray([0.9]), None


class _StubProcessor:
    def __init__(self, state: dict) -> None:
        self.state = state

    def __call__(self, images=None, text=None, return_tensors=None):
        self.state["prompts"].append(text)
        return _StubBatch()

    def post_process_grounded_object_detection(
        self, outputs, input_ids=None, threshold=0.25, text_threshold=0.25, target_sizes=None
    ):
        self.state["thresholds"].append((threshold, text_threshold))
        self.state["target_sizes"].append(target_sizes)
        detections = self.state["detections"].pop(0) if self.state["detections"] else []
        return [{
            "scores": torch.tensor([d[0] for d in detections], dtype=torch.float32),
            "boxes": torch.tensor([d[1] for d in detections], dtype=torch.float32).reshape(-1, 4),
        }]


class _StubBatch(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=torch.zeros((1, 4), dtype=torch.long))

    def to(self, device):
        return self


class _StubDetectorModel:
    def __init__(self) -> None:
        self.eval_called = False

    def to(self, device):
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, **kwargs):
        return types.SimpleNamespace()


class _StubDepthPipe:
    def __init__(self, state: dict) -> None:
        self.state = state
        self.model = types.SimpleNamespace(config=state["depth_config"])

    def __call__(self, image):
        w, h = image.size
        shape = self.state["depth_shape"] or (h, w)
        return {"predicted_depth": torch.full(shape, 0.75, dtype=torch.float32)}


def _record_load(state, loader: str, repo_id, revision, local_files_only) -> None:
    """Every hub-touching load, with the offline flag AS IT WAS at call time.

    The offline flag is read here rather than after the fact because that is the only way to tell
    "it was set for the duration of the load" from "it was set at some point".
    """
    hub = state["hub"]
    state["loads"].append({
        "loader": loader,
        "repo_id": repo_id,
        "revision": revision,
        "local_files_only": local_files_only,
        "offline": hub.constants.HF_HUB_OFFLINE if hub is not None else None,
    })


def _install_hub(monkeypatch, state: dict, cached: set[tuple[str, str]] | None = None):
    """A ``huggingface_hub`` whose cache is a set of ``(repo_id, revision)`` pairs.

    ``None`` means "everything staged at whatever pin is asked for". A pair that is not in the set
    raises, which is what a cache staged at a DIFFERENT commit looks like from the probe's side.
    """
    hub = types.ModuleType("huggingface_hub")
    constants = types.ModuleType("huggingface_hub.constants")
    constants.HF_HUB_OFFLINE = False
    hub.constants = constants

    def snapshot_download(repo_id=None, revision=None, local_files_only=False):
        assert local_files_only is True, "the cache probe must never be able to fetch"
        state["probes"].append((repo_id, revision))
        if cached is not None and (repo_id, revision) not in cached:
            raise FileNotFoundError(f"{repo_id}@{revision} is not in the cache")
        return f"/cache/{repo_id}/{revision}"

    def hf_hub_download(repo_id=None, filename=None, revision=None, local_files_only=False):
        _record_load(state, "hf_hub_download", repo_id, revision, local_files_only)
        state["files"].append((repo_id, filename, revision))
        if cached is not None and (repo_id, revision) not in cached:
            raise FileNotFoundError(f"{repo_id}@{revision} is not in the cache")
        return f"/cache/{repo_id}/{revision}/{filename}"

    hub.snapshot_download = snapshot_download
    hub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    state["hub"] = hub
    return hub


def _install(monkeypatch, *, detections=None, depth_type="metric", max_depth=20.0,
             depth_shape=None, sam2_present=True, transformers_present=True,
             sam2_ids=None, hub=True) -> dict:
    """Put the stubs in ``sys.modules`` and hand back the shared state the tests read."""
    counters = {"detector_loads": 0, "predictor_loads": 0, "depth_loads": 0}
    config = types.SimpleNamespace(depth_estimation_type=depth_type, max_depth=max_depth)
    state = {
        "detections": list(detections if detections is not None else []),
        "prompts": [],
        "thresholds": [],
        "target_sizes": [],
        "depth_config": config,
        "depth_shape": depth_shape,
        "counters": counters,
        "predictors": [],
        "loads": [],
        "probes": [],
        "files": [],
        "sam2_build": [],
        "hub": None,
    }

    if hub:
        _install_hub(monkeypatch, state)

    if not sam2_present:
        # ``None`` in sys.modules is Python's own "this import is blocked" sentinel, and it is what
        # lets the refusal be tested without uninstalling anything.
        monkeypatch.setitem(sys.modules, "sam2", None)
    else:
        sam2 = types.ModuleType("sam2")
        predictor_mod = types.ModuleType("sam2.sam2_image_predictor")
        build_mod = types.ModuleType("sam2.build_sam")

        # The real mapping, copied from sam2 1.1.0's build_sam.py.
        filenames = {
            "facebook/sam2-hiera-large": (
                "configs/sam2/sam2_hiera_l.yaml", "sam2_hiera_large.pt",
            ),
            "facebook/sam2-hiera-tiny": (
                "configs/sam2/sam2_hiera_t.yaml", "sam2_hiera_tiny.pt",
            ),
        }
        if sam2_ids is not None:
            filenames = {k: v for k, v in filenames.items() if k in sam2_ids}

        def build_sam2(config_file=None, ckpt_path=None, device=None):
            state["sam2_build"].append((config_file, ckpt_path, device))
            return types.SimpleNamespace(kind="sam2-model", ckpt_path=ckpt_path)

        class SAM2ImagePredictor(_StubPredictor):
            def __init__(self, sam_model=None) -> None:
                super().__init__(sam_model)
                counters["predictor_loads"] += 1
                state["predictors"].append(self)

        build_mod.HF_MODEL_ID_TO_FILENAMES = filenames
        build_mod.build_sam2 = build_sam2
        predictor_mod.SAM2ImagePredictor = SAM2ImagePredictor
        sam2.build_sam = build_mod
        sam2.sam2_image_predictor = predictor_mod
        monkeypatch.setitem(sys.modules, "sam2", sam2)
        monkeypatch.setitem(sys.modules, "sam2.build_sam", build_mod)
        monkeypatch.setitem(sys.modules, "sam2.sam2_image_predictor", predictor_mod)

    if not transformers_present:
        monkeypatch.setitem(sys.modules, "transformers", None)
    else:
        tf = types.ModuleType("transformers")

        class AutoProcessor:
            @classmethod
            def from_pretrained(cls, model_id, revision=None, local_files_only=False):
                counters["detector_loads"] += 1
                _record_load(state, "AutoProcessor", model_id, revision, local_files_only)
                return _StubProcessor(state)

        class AutoModelForZeroShotObjectDetection:
            @classmethod
            def from_pretrained(cls, model_id, revision=None, local_files_only=False):
                _record_load(state, "AutoModel", model_id, revision, local_files_only)
                return _StubDetectorModel()

        def pipeline(task, model=None, revision=None, local_files_only=False, device=None):
            assert task == "depth-estimation"
            counters["depth_loads"] += 1
            _record_load(state, "pipeline", model, revision, local_files_only)
            return _StubDepthPipe(state)

        tf.AutoProcessor = AutoProcessor
        tf.AutoModelForZeroShotObjectDetection = AutoModelForZeroShotObjectDetection
        tf.pipeline = pipeline
        monkeypatch.setitem(sys.modules, "transformers", tf)

    return state


def _fresh_import(monkeypatch):
    """Import the adapter with the current stubs and env, never a cached copy."""
    for name in [n for n in sys.modules if n == "estimators" or n.startswith("estimators.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(MODULE)


@pytest.fixture(autouse=True)
def _clean_estimator_imports():
    """A module that reads env at import and caches models at module level must not leak either."""
    yield
    for name in [n for n in sys.modules if n == "estimators" or n.startswith("estimators.")]:
        del sys.modules[name]


@pytest.fixture()
def loaded(monkeypatch):
    """The adapter with stubs in place, downloads permitted (nothing is fetched), one apple found."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    state = _install(monkeypatch, detections=[[(0.9, [10.0, 10.0, 30.0, 30.0])]] * 64)
    module = _fresh_import(monkeypatch)
    return module, state


@pytest.fixture()
def staged(monkeypatch):
    """The adapter on a machine staged at the pins, with downloads NOT permitted.

    This is the cluster's situation and the one the first pass got wrong: the cache holds each repo
    at a commit sha and nothing may be fetched.
    """
    monkeypatch.delenv("WAM_PR08_ALLOW_DOWNLOAD", raising=False)
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    state = _install(monkeypatch, detections=[[(0.9, [10.0, 10.0, 30.0, 30.0])]] * 64, hub=False)
    _install_hub(monkeypatch, state, cached=set(CONTRACT_PINS.items()))
    module = _fresh_import(monkeypatch)
    return module, state


def _frame(h=48, w=64) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# -- the contract -------------------------------------------------------------------------------------


def test_it_implements_exactly_the_contract_measure_est_drift_fixes(loaded):
    """Both halves are callable, and the three optional constants `Estimators` reads are present."""
    module, _ = loaded
    assert callable(module.segment)
    assert callable(module.estimate_depth)
    assert isinstance(module.ESTIMATOR_NAME, str) and module.ESTIMATOR_NAME
    assert isinstance(module.ESTIMATOR_VERSION, str) and module.ESTIMATOR_VERSION
    assert isinstance(module.GATE_QUALIFIED, bool)


def test_estimators_refuses_a_module_missing_either_half(loaded):
    """The refusal this adapter is written against, exercised rather than asserted about."""
    import measure_est_drift as ed

    module, _ = loaded
    no_depth = types.SimpleNamespace(segment=module.segment)
    with pytest.raises(ed.EstimatorUnavailable, match="estimate_depth"):
        ed.Estimators(no_depth, MODULE)
    no_segment = types.SimpleNamespace(estimate_depth=module.estimate_depth)
    with pytest.raises(ed.EstimatorUnavailable, match="segment"):
        ed.Estimators(no_segment, MODULE)
    assert ed.Estimators(module, MODULE).name == module.ESTIMATOR_NAME


def test_segment_returns_a_bool_mask_on_the_frames_grid(loaded):
    module, _ = loaded
    frame = _frame()
    mask = module.segment(frame)
    assert mask.shape == frame.shape[:2]
    assert mask.dtype == np.bool_
    assert mask.any(), "the stub detector found a box; the mask must not be empty"


def test_estimate_depth_returns_float32_metres_on_the_frames_grid(loaded):
    module, _ = loaded
    frame = _frame()
    depth = module.estimate_depth(frame)
    assert depth.shape == frame.shape[:2]
    assert depth.dtype == np.float32


def test_an_rgba_frame_is_accepted_with_the_alpha_dropped(loaded):
    """Replicator's rgb annotator hands back four channels; a capture from anything but
    isaac_binding.render_frame may still carry the alpha."""
    module, _ = loaded
    rgba = np.dstack([_frame(), np.full((48, 64), 255, dtype=np.uint8)])
    assert module.segment(rgba).shape == (48, 64)


def test_version_names_all_three_checkpoints_their_revisions_and_both_thresholds(loaded):
    """The artifact records only name and version, so anything absent from the version string is
    invisible to whoever reads the committed gate number later — and a repo id without a commit
    identifies no particular weights (AC-04)."""
    module, _ = loaded
    version = module.ESTIMATOR_VERSION
    for piece in (module.SAM2_MODEL_CHECKPOINT, module.GROUNDING_DINO_MODEL_CHECKPOINT,
                  module.DEPTH_MODEL_CHECKPOINT, module.SAM2_MODEL_REVISION,
                  module.GROUNDING_DINO_MODEL_REVISION, module.DEPTH_MODEL_REVISION,
                  str(module.BOX_THRESHOLD), str(module.TEXT_THRESHOLD)):
        assert piece in version


# -- the pins ------------------------------------------------------------------------------------------


def test_the_pins_are_the_ones_the_contract_locked(loaded):
    """Not "a revision is set" — THESE revisions, which are what the staging job stages."""
    module, _ = loaded
    assert CONTRACT_PINS[module.SAM2_MODEL_CHECKPOINT] == module.SAM2_MODEL_REVISION
    assert (
        CONTRACT_PINS[module.GROUNDING_DINO_MODEL_CHECKPOINT]
        == module.GROUNDING_DINO_MODEL_REVISION
    )
    assert CONTRACT_PINS[module.DEPTH_MODEL_CHECKPOINT] == module.DEPTH_MODEL_REVISION


def test_the_depth_default_is_the_metric_head_not_a_relative_one(loaded):
    """PR-08 §4 step 3 compares against Isaac's distance_to_camera in METRES. Video-Depth-Anything
    is affine-invariant inverse depth: there are no metres in it, so the error would not be one."""
    module, _ = loaded
    assert module.DEPTH_MODEL_ID_DEFAULT == (
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    )
    assert "Metric" in module.DEPTH_MODEL_CHECKPOINT
    assert "Video-Depth-Anything" not in module.DEPTH_MODEL_CHECKPOINT


def test_the_pins_are_extractable_from_the_source_the_way_the_staging_job_extracts_them():
    """Single source of truth: the sbatch reads these six lines instead of restating them, so the
    format they are written in is a contract and not a style choice."""
    text = _ADAPTER_SOURCE.read_text()
    extracted = {}
    for name in ("SAM2", "GROUNDING_DINO", "DEPTH"):
        repo = re.search(rf'^{name}_MODEL_ID_DEFAULT = "([^"]+)"$', text, re.M)
        rev = re.search(rf'^{name}_MODEL_REVISION_DEFAULT = "([0-9a-f]{{40}})"$', text, re.M)
        assert repo is not None, f"{name}_MODEL_ID_DEFAULT is not a one-line string literal"
        assert rev is not None, f"{name}_MODEL_REVISION_DEFAULT is not a one-line 40-hex literal"
        extracted[repo.group(1)] = rev.group(1)
    assert extracted == CONTRACT_PINS


def test_a_moving_pointer_is_refused_at_import(monkeypatch):
    """`main` names whatever upstream pushed last, so recording it records nothing."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    monkeypatch.setenv("WAM_PR08_SAM2_CHECKPOINT", "facebook/sam2-hiera-tiny")
    monkeypatch.setenv("WAM_PR08_SAM2_REVISION", "main")
    _install(monkeypatch)
    with pytest.raises(ImportError) as excinfo:
        _fresh_import(monkeypatch)
    assert "not a 40-hex commit sha" in str(excinfo.value)


def test_overriding_an_id_without_its_revision_is_refused_at_import(monkeypatch):
    """A new repo at the old commit resolves to nothing; the old repo at a new commit is a silently
    different checkpoint. They are one setting."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    monkeypatch.setenv("WAM_PR08_SAM2_CHECKPOINT", "facebook/sam2-hiera-tiny")
    _install(monkeypatch)
    with pytest.raises(ImportError) as excinfo:
        _fresh_import(monkeypatch)
    message = str(excinfo.value)
    assert "WAM_PR08_SAM2_REVISION is not" in message
    assert "Set both, or neither" in message


def test_checkpoints_are_overridable_by_env_var_in_id_and_revision_pairs(monkeypatch):
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    monkeypatch.setenv("WAM_PR08_SAM2_CHECKPOINT", "facebook/sam2-hiera-tiny")
    monkeypatch.setenv("WAM_PR08_SAM2_REVISION", "0" * 40)
    state = _install(monkeypatch, detections=[[(0.9, [1.0, 1.0, 5.0, 5.0])]])
    module = _fresh_import(monkeypatch)
    assert module.SAM2_MODEL_CHECKPOINT == "facebook/sam2-hiera-tiny"
    assert "facebook/sam2-hiera-tiny" in module.ESTIMATOR_VERSION
    assert "0" * 40 in module.ESTIMATOR_VERSION
    module.segment(_frame())
    assert state["files"] == [("facebook/sam2-hiera-tiny", "sam2_hiera_tiny.pt", "0" * 40)]


# -- a cache staged at a commit is the cache this has to see ---------------------------------------------


def test_the_probe_asks_for_the_pinned_revision_not_the_repos_default(staged):
    """huggingface_hub writes no refs/main for a cache staged at a sha, so an unpinned probe reports
    "not cached" on exactly the machine where the weights ARE staged — and the escape hatch would
    then fetch a different revision, with nothing recording that it differed."""
    module, state = staged
    assert module.available() is True
    assert set(state["probes"]) == set(CONTRACT_PINS.items())


def test_the_pair_runs_on_a_sha_staged_cache_with_downloads_forbidden(staged):
    """The end-to-end shape of the high finding: staged at the pins, WAM_PR08_ALLOW_DOWNLOAD unset,
    and it works — no refusal, no fetch, no escape hatch."""
    module, state = staged
    assert module.segment(_frame()).any()
    module.estimate_depth(_frame())
    assert len(state["loads"]) == 4, "detector processor + detector model + sam2 file + depth"
    for load in state["loads"]:
        assert load["revision"] == CONTRACT_PINS[load["repo_id"]]
        assert load["local_files_only"] is True


def test_a_cache_staged_at_a_different_commit_is_not_available(monkeypatch):
    """Same repo id, other weights. `auto` must not select this, and a load must not proceed."""
    monkeypatch.delenv("WAM_PR08_ALLOW_DOWNLOAD", raising=False)
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    state = _install(monkeypatch, hub=False)
    wrong = {(repo, "f" * 40) for repo in CONTRACT_PINS}
    _install_hub(monkeypatch, state, cached=wrong)
    module = _fresh_import(monkeypatch)

    assert module.available() is False
    with pytest.raises(module.EstimatorWeightsMissing) as excinfo:
        module.segment(_frame())
    message = str(excinfo.value)
    assert module.GROUNDING_DINO_MODEL_REVISION in message
    assert "pinned revision" in message


def test_available_is_true_only_when_every_checkpoint_is_cached(monkeypatch):
    """A segmenter running without its checkpoints does not crash — it returns empty masks, every
    step drops, and coverage: 0.0 reads as a fact about the corpus."""
    monkeypatch.delenv("WAM_PR08_ALLOW_DOWNLOAD", raising=False)
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    state = _install(monkeypatch, hub=False)
    _install_hub(monkeypatch, state, cached=set(CONTRACT_PINS.items()))
    module = _fresh_import(monkeypatch)

    every = {c.repo_id for c in module.CHECKPOINTS}
    assert len(every) == 3, "detector, segmenter and depth, named in one place"
    assert module.available() is True

    _install_hub(monkeypatch, state, cached={
        (repo, rev) for repo, rev in CONTRACT_PINS.items()
        if repo != module.SAM2_MODEL_CHECKPOINT
    })
    assert module.available() is False


def test_permission_to_download_does_not_make_the_weights_available(monkeypatch):
    """`auto` picking a method must never be the thing that starts a 3 GB fetch on a node that may
    not be allowed to make one."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    state = _install(monkeypatch, hub=False)
    _install_hub(monkeypatch, state, cached=set())
    module = _fresh_import(monkeypatch)
    assert module.available() is False


# -- offline is enforced, not merely intended ---------------------------------------------------------


def test_every_load_runs_with_hub_offline_set_and_local_files_only(staged):
    """The docstring promises 3 GB will not be fetched. A pre-check is a guard a load can walk past:
    a moved ref, a partial snapshot, an etag revalidation. So the flag is set for the duration of
    every load AND every loader is told local_files_only, because transformers keeps its own copy
    of the flag from its import."""
    module, state = staged
    module.segment(_frame())
    module.estimate_depth(_frame())

    loaders = {load["loader"] for load in state["loads"]}
    assert loaders == {"AutoProcessor", "AutoModel", "hf_hub_download", "pipeline"}
    for load in state["loads"]:
        assert load["offline"] is True, f"{load['loader']} loaded with the hub still online"
        assert load["local_files_only"] is True


def test_the_offline_flag_is_restored_afterwards(staged):
    """This module is a library; a process-wide switch it never turns back off is its caller's bug."""
    module, state = staged
    assert state["hub"].constants.HF_HUB_OFFLINE is False
    module.segment(_frame())
    assert state["hub"].constants.HF_HUB_OFFLINE is False


def test_permitting_the_download_is_the_only_thing_that_lifts_offline(loaded):
    """WAM_PR08_ALLOW_DOWNLOAD=1 says the decision has been taken. It is the whole escape hatch,
    and it is visible in stats()."""
    module, state = loaded
    module.segment(_frame())
    for load in state["loads"]:
        assert load["offline"] is False
        assert load["local_files_only"] is False
    assert module.stats()["downloads_permitted"] is True


def test_a_hub_that_cannot_be_switched_offline_is_refused_rather_than_trusted(monkeypatch):
    """If offline mode cannot be enforced, the promise in the docstring is not being kept, and a
    load that "probably" will not fetch is exactly the plausible-instead-of-refusing shape."""
    monkeypatch.delenv("WAM_PR08_ALLOW_DOWNLOAD", raising=False)
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    state = _install(monkeypatch, hub=False)
    _install_hub(monkeypatch, state, cached=set(CONTRACT_PINS.items()))
    module = _fresh_import(monkeypatch)
    monkeypatch.delitem(sys.modules, "huggingface_hub.constants")
    del state["hub"].constants

    with pytest.raises(module.EstimatorDependencyMissing, match="offline mode cannot be enforced"):
        module.segment(_frame())


# -- SAM 2 is loaded at the pin, which its own loader cannot do -------------------------------------


def test_sam2_is_built_from_the_pinned_checkpoint_file_not_from_pretrained(staged):
    """sam2's build_sam2_hf -> _hf_download calls hf_hub_download WITHOUT a revision and forwards
    no kwargs, so from_pretrained resolves refs/main and cannot be pinned. The stub's
    from_pretrained raises, so using it would fail this test rather than pass it quietly."""
    module, state = staged
    module.segment(_frame())

    assert state["files"] == [
        (module.SAM2_MODEL_CHECKPOINT, "sam2_hiera_large.pt", module.SAM2_MODEL_REVISION)
    ]
    config_file, ckpt_path, device = state["sam2_build"][0]
    assert config_file == "configs/sam2/sam2_hiera_l.yaml"
    assert module.SAM2_MODEL_REVISION in ckpt_path
    assert device == "cpu"


def test_a_sam2_without_the_two_names_this_needs_refuses_instead_of_falling_back(monkeypatch):
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    _install(monkeypatch, detections=[[(0.9, [1.0, 1.0, 5.0, 5.0])]])
    module = _fresh_import(monkeypatch)
    del sys.modules["sam2.build_sam"].build_sam2

    with pytest.raises(module.EstimatorCheckpointUnusable) as excinfo:
        module.segment(_frame())
    message = str(excinfo.value)
    assert "HF_MODEL_ID_TO_FILENAMES and build_sam2" in message
    assert f"{module.SAM2_MODEL_CHECKPOINT}@{module.SAM2_MODEL_REVISION}" in message


def test_a_checkpoint_sam2_does_not_know_is_refused_and_the_known_ids_are_listed(monkeypatch):
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    _install(monkeypatch, detections=[[(0.9, [1.0, 1.0, 5.0, 5.0])]],
             sam2_ids={"facebook/sam2-hiera-tiny"})
    module = _fresh_import(monkeypatch)

    with pytest.raises(module.EstimatorCheckpointUnusable) as excinfo:
        module.segment(_frame())
    message = str(excinfo.value)
    assert "does not know the checkpoint" in message
    assert "facebook/sam2-hiera-tiny" in message


# -- no detection is an event, not an error -----------------------------------------------------------


def test_no_detection_yields_an_all_false_mask_and_never_raises(monkeypatch):
    """The hand occludes the apple. The callers drop and COUNT that; raising would kill the run."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    _install(monkeypatch, detections=[[]])
    module = _fresh_import(monkeypatch)

    mask = module.segment(_frame())
    assert mask.shape == (48, 64)
    assert mask.dtype == np.bool_
    assert not mask.any()
    assert module.NO_DETECTION_FRAMES == 1


def test_a_frame_with_no_detection_does_not_get_the_previous_frames_mask(monkeypatch):
    """A carried-over mask invents a displacement that was never observed and pulls the p95 DOWN,
    which widens G0b — conservative-looking and backwards."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    _install(monkeypatch, detections=[[(0.9, [10.0, 10.0, 30.0, 30.0])], []])
    module = _fresh_import(monkeypatch)

    assert module.segment(_frame()).any()
    assert not module.segment(_frame()).any()
    assert module.NO_DETECTION_FRAMES == 1
    assert module.SEGMENT_CALLS == 2


def test_an_empty_mask_from_a_detected_box_is_counted_separately(monkeypatch):
    """Both drop the step, but they are different events: one is the apple not being there, the
    other is the segmenter failing on a frame where it was. A coverage shortfall with no recorded
    explanation is a run that has to be done again to find out why."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    _install(monkeypatch, detections=[[(0.9, [4.0, 4.0, 4.0, 4.0])]])
    module = _fresh_import(monkeypatch)

    assert not module.segment(_frame()).any()
    assert module.NO_DETECTION_FRAMES == 0
    assert module.EMPTY_MASK_FRAMES == 1
    assert module.stats()["n_frames_with_empty_mask"] == 1


def test_the_highest_scoring_box_prompts_sam2_not_the_biggest_one(monkeypatch):
    """A box that CONTAINS the apple (the plate, the table) is an ambiguous SAM 2 prompt, and the
    resulting centroid tracks whatever dominates it while still looking like a centroid."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    state = _install(monkeypatch, detections=[[
        (0.41, [0.0, 0.0, 60.0, 44.0]),    # the whole table: bigger, less confident
        (0.88, [12.0, 8.0, 26.0, 22.0]),   # the apple
    ]])
    module = _fresh_import(monkeypatch)
    module.segment(_frame())

    assert state["predictors"][0].boxes_seen[0].tolist() == [12.0, 8.0, 26.0, 22.0]


# -- loaded once, not once per frame -------------------------------------------------------------------


def test_the_models_are_loaded_once_per_process_and_cached(loaded):
    """SAM 2 hiera-large per frame turns a few-minute calibration into an overnight one and changes
    no number, which is the kind of cost that is discovered after the run."""
    module, state = loaded
    for _ in range(5):
        module.segment(_frame())
        module.estimate_depth(_frame())

    assert state["counters"]["detector_loads"] == 1
    assert state["counters"]["predictor_loads"] == 1
    assert state["counters"]["depth_loads"] == 1


def test_reset_models_actually_drops_the_cache(loaded):
    """Proves the cache is a cache rather than a one-shot that happened to be called once."""
    module, state = loaded
    module.segment(_frame())
    module.reset_models()
    module.segment(_frame())
    assert state["counters"]["predictor_loads"] == 2


def test_the_checkpoint_ids_reaching_the_loaders_are_the_module_constants(loaded):
    """Cosmos-Transfer2.5's own ids, per §4 step 2's 'the same segmenter' — not a local path."""
    module, state = loaded
    module.segment(_frame())
    module.estimate_depth(_frame())
    reached = {load["repo_id"] for load in state["loads"]}
    assert reached == {
        module.SAM2_MODEL_CHECKPOINT,
        module.GROUNDING_DINO_MODEL_CHECKPOINT,
        module.DEPTH_MODEL_CHECKPOINT,
    }
    assert module.SAM2_MODEL_CHECKPOINT == "facebook/sam2-hiera-large"
    assert "/" in module.SAM2_MODEL_CHECKPOINT and not module.SAM2_MODEL_CHECKPOINT.startswith("/")


# -- the prompt ------------------------------------------------------------------------------------------


def test_the_prompt_is_lowercased_and_period_terminated_for_grounding_dino(monkeypatch):
    """Its documented phrase format. 'Apple' without the period parses differently in the text
    encoder and simply detects less, which shows up as low coverage and a p95 over the survivors."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    monkeypatch.setenv("WAM_PR08_OBJECT_PROMPT", "Apple")
    state = _install(monkeypatch, detections=[[(0.9, [1.0, 1.0, 5.0, 5.0])]])
    module = _fresh_import(monkeypatch)

    assert module.OBJECT_TEXT_PROMPT == "apple."
    assert module.OBJECT_TEXT_PROMPT_RAW == "Apple"  # the normalisation is recorded, not hidden
    module.segment(_frame())
    assert state["prompts"] == ["apple."]


def test_the_prompt_the_harness_cannot_see_is_at_least_recorded_and_named(loaded):
    """--object-class is set separately and this module cannot see it, so the mismatch cannot be
    refused here. It is put in front of whoever reads the artifact instead, and named as a blocker
    rather than left as a thing someone would have to already know."""
    module, _ = loaded
    record = module.stats()
    assert record["object_text_prompt"] == "apple."
    assert "--object-class" in record["object_text_prompt_note"]
    assert any("--object-class" in b for b in module.GATE_QUALIFICATION_BLOCKERS)


def test_the_configured_thresholds_reach_the_post_processor(loaded):
    """Recorded in ESTIMATOR_VERSION and ignored by the call would be the worst of both."""
    module, state = loaded
    module.segment(_frame())
    assert state["thresholds"] == [(module.BOX_THRESHOLD, module.TEXT_THRESHOLD)]
    assert state["target_sizes"] == [[(48, 64)]]


# -- depth: metric or nothing -----------------------------------------------------------------------------


def test_a_metric_checkpoint_is_recorded_as_metric(loaded):
    module, _ = loaded
    module.estimate_depth(_frame())
    assert module.DEPTH_ESTIMATION_TYPE == "metric"
    assert module.DEPTH_IS_METRIC is True
    assert module.DEPTH_MAX_DEPTH_M == pytest.approx(20.0)


def test_a_relative_checkpoint_refuses_at_load_and_says_why(monkeypatch):
    """Relative Depth-Anything is affine-free INVERSE depth: no units, ordered backwards, and it
    lands in the artifact under a key called mean_m looking entirely reasonable."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    monkeypatch.setenv("WAM_PR08_DEPTH_CHECKPOINT", "depth-anything/Depth-Anything-V2-Large-hf")
    monkeypatch.setenv("WAM_PR08_DEPTH_REVISION", "a" * 40)
    _install(monkeypatch, depth_type="relative", max_depth=None)
    module = _fresh_import(monkeypatch)

    with pytest.raises(module.EstimatorCheckpointUnusable) as excinfo:
        module.estimate_depth(_frame())
    message = str(excinfo.value)
    assert "RELATIVE" in message
    assert "depth-anything/Depth-Anything-V2-Large-hf" in message
    assert "WAM_PR08_DEPTH_CHECKPOINT" in message
    assert "Depth-Anything-V2-Metric-Indoor-Large-hf" in message
    # It is recorded as well as refused: a caller that catches this still has to be able to say
    # what it caught.
    assert module.DEPTH_IS_METRIC is False
    assert module.DEPTH_ESTIMATION_TYPE == "relative"


def test_a_config_that_does_not_say_is_treated_as_relative(monkeypatch):
    """transformers' DepthAnythingConfig defaults to 'relative', and an unstated claim is not a
    claim — the same rule measure_geom_tol applies to gate_qualified."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    state = _install(monkeypatch)
    state["depth_config"] = types.SimpleNamespace()
    module = _fresh_import(monkeypatch)
    with pytest.raises(module.EstimatorDependencyMissing):
        module.estimate_depth(_frame())
    assert module.DEPTH_IS_METRIC is False


def test_the_relative_refusal_is_not_an_import_failure_even_though_it_is_catchable_as_one(loaded):
    """It subclasses ImportError so resolve_estimators prints it, but "this depth map has no metres
    in it" is not "the module is not importable", and a caller has to be able to tell them apart."""
    module, _ = loaded
    assert issubclass(module.EstimatorCheckpointUnusable, module.EstimatorDependencyMissing)
    assert issubclass(module.EstimatorWeightsMissing, module.EstimatorDependencyMissing)
    assert issubclass(module.EstimatorDependencyMissing, ImportError)
    assert not issubclass(module.EstimatorCheckpointUnusable, module.EstimatorWeightsMissing)


def test_a_depth_map_on_a_different_grid_is_refused_not_resized(monkeypatch):
    """The pipeline's own post-processing is what resizes to the source grid; a mismatch means it
    did not run, which means the VALUES are not what this module claims either."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    _install(monkeypatch, depth_shape=(37, 37))
    module = _fresh_import(monkeypatch)
    with pytest.raises(RuntimeError, match=r"\(37, 37\) map for a \(48, 64\) frame"):
        module.estimate_depth(_frame())


# -- input validation --------------------------------------------------------------------------------------


def test_a_float_frame_is_refused_rather_than_rescaled(loaded):
    """[0, 1] and [0, 255] floats are indistinguishable from the array alone and rescale to
    different pictures — a different detection, and therefore a different centroid."""
    module, _ = loaded
    with pytest.raises(ValueError, match="uint8"):
        module.segment(_frame().astype(np.float32) / 255.0)
    with pytest.raises(ValueError, match="uint8"):
        module.estimate_depth(_frame().astype(np.float32) / 255.0)


def test_a_non_image_array_is_refused(loaded):
    module, _ = loaded
    with pytest.raises(ValueError, match=r"\(H, W, 3\|4\)"):
        module.segment(np.zeros((48, 64), dtype=np.uint8))


# -- gate qualification is opt-in, and this module does not opt in --------------------------------------------


def test_it_does_not_claim_to_be_a_gate_estimator_and_says_why(loaded):
    module, _ = loaded
    assert module.GATE_QUALIFIED is False
    assert module.GATE_QUALIFICATION_BLOCKERS
    joined = " ".join(module.GATE_QUALIFICATION_BLOCKERS)
    assert "Never executed" in joined
    assert "coverage" in joined


def test_measure_est_drift_reads_the_flag_as_false(loaded):
    """The claim has to survive the harness that consumes it, not just the module that makes it."""
    import measure_est_drift as ed

    estimators = ed.Estimators(loaded[0], MODULE)
    assert estimators.gate_qualified is False
    assert estimators.name == loaded[0].ESTIMATOR_NAME
    assert estimators.version == loaded[0].ESTIMATOR_VERSION


def test_stats_carries_what_the_artifact_would_need(loaded):
    module, _ = loaded
    module.segment(_frame())
    record = module.stats()
    assert record["gate_qualified"] is False
    assert record["n_segment_calls"] == 1
    assert record["depth_checkpoint"] == module.DEPTH_MODEL_CHECKPOINT
    assert record["depth_revision"] == module.DEPTH_MODEL_REVISION
    assert record["segmenter_revision"] == module.SAM2_MODEL_REVISION
    assert record["detector_revision"] == module.GROUNDING_DINO_MODEL_REVISION
    assert record["object_text_prompt"] == "apple."


# -- the loud refusals -----------------------------------------------------------------------------------------


def test_a_missing_sam2_fails_at_import_naming_everything_it_looked_for(monkeypatch):
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    _install(monkeypatch, sam2_present=False)
    with pytest.raises(ImportError) as excinfo:
        _fresh_import(monkeypatch)
    message = str(excinfo.value)
    assert "sam2" in message
    assert "facebook/sam2-hiera-large" in message
    assert CONTRACT_PINS["facebook/sam2-hiera-large"] in message
    assert "IDEA-Research/grounding-dino-base" in message
    assert "98_build_transfer25_env.sbatch" in message
    assert "WAM_PR08_ALLOW_DOWNLOAD" in message
    assert "ASK" in message
    # The two measurements share the segmenter, so the refusal has to point at the other one.
    assert "no_segmenter_message" in message
    assert sys.executable in message


def test_a_missing_transformers_fails_at_import_too(monkeypatch):
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    _install(monkeypatch, transformers_present=False)
    with pytest.raises(ImportError) as excinfo:
        _fresh_import(monkeypatch)
    assert "transformers" in str(excinfo.value)


def test_the_dependency_failure_reaches_resolve_estimators_as_a_message_not_a_traceback(monkeypatch):
    """It subclasses ImportError precisely so `measure` prints the diagnosis and exits 2."""
    import measure_est_drift as ed

    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    _install(monkeypatch, sam2_present=False)
    for name in [n for n in sys.modules if n == "estimators" or n.startswith("estimators.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    with pytest.raises(ed.EstimatorUnavailable) as excinfo:
        ed.resolve_estimators(MODULE)
    assert "sam2" in str(excinfo.value)
    assert "no gate-qualified" not in str(excinfo.value), "this is the module's refusal, not auto's"


def test_uncached_weights_are_refused_rather_than_downloaded(monkeypatch):
    """~3 GB is a download at scale, which is the project owner's call — and the provider forbids
    agents on the login node outright."""
    monkeypatch.delenv("WAM_PR08_ALLOW_DOWNLOAD", raising=False)
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    state = _install(monkeypatch, detections=[[(0.9, [1.0, 1.0, 5.0, 5.0])]], hub=False)
    _install_hub(monkeypatch, state, cached=set())

    module = _fresh_import(monkeypatch)
    with pytest.raises(module.EstimatorWeightsMissing) as excinfo:
        module.segment(_frame())
    message = str(excinfo.value)
    assert "IDEA-Research/grounding-dino-base" in message
    assert "WAM_PR08_ALLOW_DOWNLOAD=1" in message
    assert "not in the local hub cache" in message


def test_a_missing_segmenter_refuses_even_on_a_frame_where_nothing_is_detected(monkeypatch):
    """The check used to sit behind the detection branch, so a capture in which nothing was found
    ran to completion with no segmenter at all and reported coverage: 0.0 as a fact about the
    corpus — the exact outcome available() exists to prevent."""
    monkeypatch.delenv("WAM_PR08_ALLOW_DOWNLOAD", raising=False)
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    state = _install(monkeypatch, detections=[[]], hub=False)
    _install_hub(monkeypatch, state, cached={
        (repo, rev) for repo, rev in CONTRACT_PINS.items() if "sam2" not in repo
    })
    module = _fresh_import(monkeypatch)

    with pytest.raises(module.EstimatorWeightsMissing) as excinfo:
        module.segment(_frame())
    assert module.SAM2_MODEL_CHECKPOINT in str(excinfo.value)


def test_allowing_the_download_skips_the_cache_check_entirely(monkeypatch):
    """Otherwise the guard would still be consulting a cache it has just been told not to require."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    state = _install(monkeypatch, detections=[[(0.9, [1.0, 1.0, 5.0, 5.0])]])

    def snapshot_download(**kwargs):
        raise AssertionError("the cache must not be consulted once the fetch is permitted")

    state["hub"].snapshot_download = snapshot_download

    module = _fresh_import(monkeypatch)
    assert module.segment(_frame()).any()
    assert state["probes"] == []


def test_a_broken_huggingface_hub_is_not_reported_as_missing_weights(monkeypatch):
    """`snapshot_download` raising ImportError is a package problem, and calling it a cache problem
    sends whoever reads the refusal to stage weights that are already there."""
    monkeypatch.delenv("WAM_PR08_ALLOW_DOWNLOAD", raising=False)
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    state = _install(monkeypatch, hub=False)
    _install_hub(monkeypatch, state, cached=set(CONTRACT_PINS.items()))
    module = _fresh_import(monkeypatch)
    del state["hub"].snapshot_download

    with pytest.raises(ImportError) as excinfo:
        module.available()
    assert "snapshot_download" in str(excinfo.value)
    assert "not in the local hub cache" not in str(excinfo.value)


def test_huggingface_hub_is_one_of_the_packages_the_import_check_names(monkeypatch):
    """It is a dependency of this module in its own right now — the pins are resolved through it
    and offline mode is enforced through it — so a missing one has to be nameable."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    _install(monkeypatch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError) as excinfo:
        _fresh_import(monkeypatch)
    assert "huggingface_hub" in str(excinfo.value)


def test_the_declared_checkpoints_measure_geom_tol_records_carry_the_commits(loaded):
    """measure_geom_tol records what the adapter DECLARES it loads, and re-runs "the same
    estimator" at gate time. A repo id alone cannot be joined to the staging manifest, which
    records resolved shas."""
    module, _ = loaded
    declared = module.ESTIMATOR_CHECKPOINTS
    assert set(declared) == {"detector", "segmenter", "depth"}
    assert declared["segmenter"] == (
        f"{module.SAM2_MODEL_CHECKPOINT}@{module.SAM2_MODEL_REVISION}"
    )
    for value in declared.values():
        repo, _, revision = value.partition("@")
        assert CONTRACT_PINS[repo] == revision
