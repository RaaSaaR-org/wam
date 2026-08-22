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


#: Where the fruit is in every stub frame, and it is the box ``_install``'s default detection
#: returns for a reason: since 2026-08-22 the adapter REFUSES a mask that contains essentially none
#: of the object (PR-08 V6), and ``_StubPredictor.predict`` returns the prompted box as the mask. A
#: frame of pure noise would therefore be refused on every test in this file — correctly, which is
#: the point. So the stub frames contain an apple where the stub detector says one is.
STUB_APPLE_BOX = (10, 10, 30, 30)


def _frame(h=48, w=64, apple: tuple[int, int, int, int] | None = STUB_APPLE_BOX) -> np.ndarray:
    """A noisy but COLD frame with a warm, saturated blob at ``apple``.

    Cold everywhere else on purpose: the reference predicate is ``r > 90 and r - b > 50 and
    saturation > 0.35``, so the background is drawn from a blue-dominant range that cannot satisfy
    it whatever the noise does. ``apple=None`` gives a frame with no fruit in it at all, which is
    the occlusion case and is refused.
    """
    rng = np.random.default_rng(0)
    frame = np.empty((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = rng.integers(0, 80, size=(h, w), dtype=np.uint8)     # R low: cold
    frame[:, :, 1] = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    frame[:, :, 2] = rng.integers(120, 256, size=(h, w), dtype=np.uint8)  # B high: cold
    if apple is not None:
        x0, y0, x1, y1 = apple
        frame[y0:y1, x0:x1] = (220, 30, 30)
    return frame


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


def test_the_prompt_the_harness_cannot_see_is_recorded_and_no_longer_only_recorded(loaded):
    """This module still cannot see ``--object-class``, but the coupling is no longer left to a
    reader: the flag defaults to this prompt and refuses an explicit value that disagrees, so the
    blocker that named the coupling is discharged — in the harness, with the discharge written
    down rather than the blocker quietly disappearing."""
    module, _ = loaded
    record = module.stats()
    assert record["object_text_prompt"] == "apple."
    assert "--object-class" in record["object_text_prompt_note"]
    assert not any("--object-class" in b for b in module.GATE_QUALIFICATION_BLOCKERS)
    assert any("--object-class" in d for d in module.GATE_QUALIFICATION_DISCHARGED)


def test_the_configured_thresholds_reach_the_post_processor(loaded):
    """Recorded in ESTIMATOR_VERSION and ignored by the call would be the worst of both."""
    module, state = loaded
    module.segment(_frame())
    assert state["thresholds"] == [(module.BOX_THRESHOLD, module.TEXT_THRESHOLD)]
    assert state["target_sizes"] == [[(48, 64)]]


# -- the generator's operating point, not ours ---------------------------------------------------------


def test_the_detection_thresholds_are_the_generators_own_numbers(loaded):
    """PR-08 §4 step 2 asks for THE SAME segmenter as Cosmos-Transfer2.5's, and a checkpoint id is
    not an operating point: the same GroundingDINO at 0.35 and at 0.15 detects on different frames,
    which is a different mask, a different centroid and a different no-detection rate.

    These four numbers are read off the generator's own
    ``cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_model.py`` (cluster copy, 2026-08-22),
    which post-processes at ``threshold=0.15, text_threshold=0.25`` and retries once at
    ``(0.1, 0.1)``. They are restated here rather than imported from the module under test, because
    a test that reads the constant it checks cannot notice the constant moving — and the one edit
    this file exists to catch is somebody "tuning" them on AppleToPlate, which would make this a
    different segmenter from the one that draws the conditioning masks.
    """
    module, _ = loaded
    assert module.BOX_THRESHOLD == 0.15
    assert module.TEXT_THRESHOLD == 0.25
    assert module.RETRY_BOX_THRESHOLD == 0.1
    assert module.RETRY_TEXT_THRESHOLD == 0.1
    assert module.BOX_SELECTION == "highest_score"


def test_the_retry_fires_only_when_the_first_pass_found_no_box_at_all(monkeypatch):
    """Upstream retries on ``len(boxes) == 0`` and on nothing else. Retrying on a low score, or
    looping the thresholds down until something appears, is a detector of our own design — and a
    budget measured with it is a budget for an error the generator does not commit."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    # Frame 1: a weak box, well under the 0.35 this module used to use but over 0.15. Frame 2:
    # nothing at either pass.
    state = _install(monkeypatch, detections=[[(0.17, [10.0, 10.0, 30.0, 30.0])], [], []])
    module = _fresh_import(monkeypatch)

    assert module.segment(_frame()).any()
    assert state["thresholds"] == [(0.15, 0.25)], "a box was found; there is nothing to retry"
    assert module.RETRY_FRAMES == 0

    assert not module.segment(_frame()).any()
    assert state["thresholds"][1:] == [(0.15, 0.25), (0.1, 0.1)], "exactly one retry, at (0.1, 0.1)"
    assert module.RETRY_FRAMES == 1
    assert module.RETRY_RECOVERED_FRAMES == 0
    assert module.NO_DETECTION_FRAMES == 1


def test_a_box_found_only_by_the_retry_is_used_and_counted(monkeypatch):
    """The retry buys coverage by accepting a weak detection, and on an occluded frame a weak
    detection lands on something that is not the apple — raising coverage while degrading the mask,
    i.e. hiding inside the one number the harness gates on. It is upstream's behaviour so it stays;
    counting it is what makes its contribution visible instead of assumed."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    state = _install(monkeypatch, detections=[[], [(0.12, [8.0, 8.0, 20.0, 20.0])]])
    module = _fresh_import(monkeypatch)

    assert module.segment(_frame()).any()
    assert state["thresholds"] == [(0.15, 0.25), (0.1, 0.1)]
    assert state["predictors"][0].boxes_seen[0].tolist() == [8.0, 8.0, 20.0, 20.0]
    assert module.RETRY_FRAMES == 1
    assert module.RETRY_RECOVERED_FRAMES == 1
    assert module.NO_DETECTION_FRAMES == 0
    record = module.stats()
    assert record["n_frames_retry_fired"] == 1
    assert record["n_frames_retry_recovered"] == 1


def test_the_retry_pass_also_takes_the_highest_scoring_box(monkeypatch):
    """The selection rule does not change between the two passes — upstream sorts once, after
    whichever post-processing produced boxes."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    state = _install(monkeypatch, detections=[[], [
        (0.11, [0.0, 0.0, 60.0, 44.0]),    # the whole table, found only at the retry threshold
        (0.13, [12.0, 8.0, 26.0, 22.0]),   # the apple, barely
    ]])
    module = _fresh_import(monkeypatch)
    module.segment(_frame())

    assert state["predictors"][0].boxes_seen[0].tolist() == [12.0, 8.0, 26.0, 22.0]


def test_the_detector_runs_once_even_when_the_post_processing_runs_twice(monkeypatch):
    """post_process_grounded_object_detection is a pure function of the outputs and the two
    thresholds, so a second forward pass would buy identical logits at the price of a second
    inference over every no-detection frame — of which this corpus has many."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    state = _install(monkeypatch, detections=[[], []])
    module = _fresh_import(monkeypatch)
    module.segment(_frame())

    assert len(state["thresholds"]) == 2, "two post-processings"
    assert state["prompts"] == ["apple."], "one processor call, i.e. one forward pass"


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
    assert "coverage" in joined
    # The two that survive 2026-08-22 and are the reason the flag is still False: no human has
    # looked at a mask this adapter produced, and it re-detects per frame where the generator
    # propagates one mask across the clip.
    assert "NOBODY HAS LOOKED AT A MASK" in joined
    assert "PER-FRAME SEGMENTATION IS NOT UPSTREAM'S PROPAGATION" in joined


def test_the_stale_never_executed_blocker_is_withdrawn_by_evidence_and_not_by_deletion(loaded):
    """Job 189583 staged all three checkpoints at the pinned revisions and 189588 drove this
    adapter over 720 corpus frames, so 'never executed, no checkpoint staged' is false and must not
    keep standing. It is ALSO not simply gone: a blocker that vanishes between two commits looks
    identical whether it was satisfied or deleted by whoever found it inconvenient, and only one of
    those is allowed to shorten the list."""
    module, _ = loaded
    assert not any("Never executed" in b for b in module.GATE_QUALIFICATION_BLOCKERS)
    discharged = " ".join(module.GATE_QUALIFICATION_DISCHARGED)
    assert "189583" in discharged and "189588" in discharged
    assert module.stats()["gate_qualification_discharged"] == list(
        module.GATE_QUALIFICATION_DISCHARGED
    )


def test_execution_is_not_confused_with_correctness(loaded):
    """The distinction the rewritten blocker 1 rests on, asserted so a later edit cannot blur it:
    coverage 1.0 says a box came back on every frame, not that it was the apple's box."""
    module, _ = loaded
    blocker = next(b for b in module.GATE_QUALIFICATION_BLOCKERS if "LOOKED AT A MASK" in b)
    assert "coverage 1.0" in blocker
    assert "not that it was the APPLE's box" in blocker


def test_the_pilot_job_the_withdrawal_rests_on_is_labelled_as_an_untracked_claim(loaded):
    """Blocker 1's central factual claim — that this adapter has run end to end over the corpus —
    rests on job 189588, and 189588 is recorded in no tracked file in this repository.
    ``.mc/tasks/todo/T-040-*.md`` records 189583, 189584, 189585 and 189586 and not that one.

    A blocker tuple is where this project writes down what is and is not established, so a number
    cited there that nobody can look up has to say so in the same sentence. It must also say that
    the run it cites predates the operating point this file now pins: the pilot ran at
    box_threshold 0.35 with no retry branch, which is a different segmenter from the one whose
    output the blocker is reasoning about."""
    module, _ = loaded
    blocker = next(b for b in module.GATE_QUALIFICATION_BLOCKERS if "LOOKED AT A MASK" in b)
    assert "189588" in blocker
    assert "NOT RECORDED ANYWHERE TRACKED" in blocker, (
        "the untracked citation must be labelled where it is used, not only in a report"
    )
    assert "0.35" in blocker, "and it must say the pilot ran at the superseded operating point"


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
    # The box is STUB_APPLE_BOX rather than an arbitrary corner of the frame because the mask this
    # produces has to survive the mask-validity filter for `.any()` below to be about the cache.
    state = _install(monkeypatch, detections=[[(0.9, [10.0, 10.0, 30.0, 30.0])]])

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


# -- the committed segmenter contract ------------------------------------------------------------------
#
# configs/transfer25/pr08_geom_tol.json is the file PR-08 §4 step 2 is checked against: it records
# WHICH segmenter both halves of §6's subtraction must have used, and it is committed BEFORE either
# number is measured so that the method cannot be chosen after seeing the result. These tests are
# the reason it cannot rot: the module and the file are compared field for field, and the sidecar is
# compared to the bytes.

GEOM_TOL_CONTRACT = _REPO / "configs" / "transfer25" / "pr08_geom_tol.json"


def _contract_doc() -> dict:
    import json

    return json.loads(GEOM_TOL_CONTRACT.read_text(encoding="utf-8"))


def test_the_committed_contract_is_the_modules_own_segmenter_block(loaded):
    """Field for field, not "looks similar". Every entry in this block changes which frames get a
    mask or where its centroid lands — the prompt, both thresholds, the retry pair, the box rule,
    the propagation mode, the two checkpoint pins — and the whole purpose of committing it early is
    that a later edit to the module has to be an edit to the committed file too, in the same
    reviewable change."""
    module, _ = loaded
    assert _contract_doc()["segmenter"] == module.SEGMENTER_CONTRACT


def test_the_committed_contract_names_the_join_key_and_the_pinned_weights(loaded):
    """A contract that named a segmenter loosely would be worse than none: it would read as a check
    that ran and agreed."""
    module, _ = loaded
    block = _contract_doc()["segmenter"]
    assert block["method_name"] == module.ESTIMATOR_NAME
    assert block["detector"]["revision"] == CONTRACT_PINS[block["detector"]["repo"]]
    assert block["segmenter"]["revision"] == CONTRACT_PINS[block["segmenter"]["repo"]]
    assert block["depth"]["revision"] == CONTRACT_PINS[block["depth"]["repo"]]
    assert block["box_threshold"] == 0.15
    assert block["text_threshold"] == 0.25
    assert block["retry_box_threshold"] == 0.1
    assert block["retry_text_threshold"] == 0.1
    assert block["box_selection"] == "highest_score"
    assert block["propagation"] == "per_frame"
    assert block["upstream_propagation"] == "sam2_video_predictor"
    assert block["pixel_grid_hw"] == [480, 640]


def test_the_committed_document_is_a_contract_plus_a_measurement_and_never_only_a_measurement():
    """The file is ONE document in TWO declared sections and this asserts the invariant that holds
    in both of its lifetimes, rather than the one that holds only today.

    Before GEOM_TOL is measured every measurement slot is null: a contract that arrived with numbers
    in it would be a measurement that chose its own method afterwards, which is the failure the
    committed style partition exists to prevent. Afterwards ``measure_geom_tol.py`` writes its
    artifact over this same path — that path is its default ``--out`` and its ``--merge`` target —
    and the contract section has to still be there, because the day it is not is the day
    ``cross_check_geom_tol`` refuses every run with ``geom_tol_does_not_record_segmenter_params``.
    A test that only asserted the nulls would go red on the commit that lands the number, and the
    obvious fix for a red test is to delete it."""
    doc = _contract_doc()
    # 1.1.0 (2026-08-22) added the `est_drift_estimator_name` measurement slot. The contract
    # section is otherwise untouched — same segmenter block, byte for byte — and the bump is here
    # so that a slot cannot be added to this document without the change being reviewed.
    assert doc["spec_version"] == "1.1.0"
    assert "§4 step 2" in doc["what_this_is"]

    # The contract section names itself, and names what may be filled in. Both lists are what
    # measure_geom_tol.merge_committed_contract() copies forward and fills.
    assert set(doc["contract_fields"]) == {
        "spec_version", "what_this_is", "contract_fields", "measurement_fields", "segmenter"
    }
    assert set(doc["measurement_fields"]) == {
        "geom_tol_px", "geom_tol_source", "est_drift_p95_px", "est_drift_source",
        # THE JOIN KEY, and a slot rather than a note in somebody's head. PR-08 §4 step 2 requires
        # both halves of GEOM_TOL - EST_DRIFT_P95 to come from one segmenter; the two numbers are
        # measured by two scripts into two files and merged here, and run_g0_gates can only check
        # the claim if the name arrives with the number.
        "est_drift_estimator_name",
        "gate_margin_px",
    }
    for key in doc["contract_fields"]:
        assert key in doc, f"the contract section is missing {key}"

    if doc.get("GEOM_TOL_px") is None:
        for key in doc["measurement_fields"]:
            assert doc[key] is None, f"{key} must be null until it is measured"
    else:
        # A measured artifact landed here. One quantity, one value, under both spellings —
        # run_g0_gates._first_present refuses a document that states it twice and differently.
        assert doc["geom_tol_px"] == doc["GEOM_TOL_px"]
        assert doc["geom_tol_source"], "a measured tolerance must say what produced it"


def test_the_measured_artifact_would_still_carry_the_contract_the_reader_looks_for(
    loaded, tmp_path, monkeypatch
):
    """The end-to-end property the whole two-section design exists for, exercised without a GPU:
    take the REAL committed document, hand it to the writer's carry-forward, and the reader must
    still find the segmenter block afterwards.

    This is the failure that made the gate unreachable: measure_geom_tol.py's default --out is this
    file, so the first real run wrote a document with no segmenter anywhere and every later
    measure_est_drift run refused, correctly and permanently."""
    import json
    import shutil

    import measure_est_drift as ed
    import measure_geom_tol as mgt

    module, _ = loaded
    target = tmp_path / "pr08_geom_tol.json"
    shutil.copy(GEOM_TOL_CONTRACT, target)

    # The shape measure_geom_tol builds for a sam2 run, minus everything irrelevant here.
    record = {
        "measured_by": "scripts/measure_geom_tol.py",
        "measured_date": "2026-08-22",
        "git_commit": "deadbeef",
        "resolution_hw": [480, 640],
        "gate_qualified": True,
        "GEOM_TOL_px": 3.4,
        "mask_method": {"name": module.ESTIMATOR_NAME,
                        "params": {"segmenter": dict(module.SEGMENTER_CONTRACT)}},
    }
    carried = mgt.merge_committed_contract(target, record)
    assert carried == module.SEGMENTER_CONTRACT
    target.write_text(json.dumps(record, indent=2))

    monkeypatch.setattr(ed, "GEOM_TOL_ARTIFACT", target)
    reasons, compare = ed.cross_check_geom_tol(
        [480, 640], module.ESTIMATOR_NAME, module.SEGMENTER_CONTRACT
    )
    assert "geom_tol_does_not_record_segmenter_params" not in reasons
    assert "segmenter_params_disagree_with_geom_tol" not in reasons
    assert compare["geom_tol_segmenter_contract_at"] == "segmenter"


def test_the_contracts_sha256_sidecar_matches_its_bytes():
    """The same discipline configs/transfer25/pr08_style_partition.json.sha256 uses: hash of the
    file's bytes, hexdigest plus a newline, so "the file the gate read is the file that was
    committed" is checkable with sha256sum instead of trusted."""
    import hashlib

    payload = GEOM_TOL_CONTRACT.read_bytes()
    sidecar = GEOM_TOL_CONTRACT.parent / (GEOM_TOL_CONTRACT.name + ".sha256")
    assert sidecar.read_text(encoding="utf-8") == hashlib.sha256(payload).hexdigest() + "\n"


def _contract_at(tmp_path, monkeypatch, **overrides):
    """Point measure_est_drift at a copy of the REAL committed contract, optionally perturbed."""
    import copy
    import json

    import measure_est_drift as ed

    doc = _contract_doc()
    block = copy.deepcopy(doc["segmenter"])
    block.update(overrides)
    doc["segmenter"] = block
    p = tmp_path / "pr08_geom_tol.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(ed, "GEOM_TOL_ARTIFACT", p)
    return ed


def test_the_cross_check_agrees_with_the_real_committed_contract(loaded, tmp_path, monkeypatch):
    """The baseline the two refusals below are refusals FROM. GEOM_TOL itself is not measured yet,
    so the run is still disqualified — on the gate flag, which is the honest reason — and not on
    the segmenter."""
    module, _ = loaded
    ed = _contract_at(tmp_path, monkeypatch)
    reasons, compare = ed.cross_check_geom_tol(
        [480, 640], module.ESTIMATOR_NAME, module.SEGMENTER_CONTRACT
    )
    assert "segmenter_params_disagree_with_geom_tol" not in reasons
    assert "mask_method_disagrees_with_estimator" not in reasons
    assert "resolution_disagrees_with_geom_tol" not in reasons
    assert compare["segmenter_param_disagreements"] == []
    assert compare["geom_tol_segmenter_contract_at"] == "segmenter"
    assert "geom_tol_is_not_gate_qualified" in reasons


def test_the_cross_check_refuses_a_method_name_mismatch(loaded, tmp_path, monkeypatch):
    """§4 step 2 says the SAME segmenter. Two names is two quantities, and §6 subtracts them."""
    module, _ = loaded
    ed = _contract_at(tmp_path, monkeypatch, method_name="hsv-red-diagnostic")
    reasons, compare = ed.cross_check_geom_tol(
        [480, 640], module.ESTIMATOR_NAME, module.SEGMENTER_CONTRACT
    )
    assert "mask_method_disagrees_with_estimator" in reasons
    assert "segmenter_params_disagree_with_geom_tol" in reasons
    assert [d["field"] for d in compare["segmenter_param_disagreements"]] == ["method_name"]


def test_the_cross_check_refuses_a_detection_parameter_mismatch(loaded, tmp_path, monkeypatch):
    """The failure a name-only check cannot see: the identical adapter, the identical name, one
    threshold moved. It detects on different frames, produces different centroids, and reports
    ESTIMATOR_NAME throughout."""
    module, _ = loaded
    ed = _contract_at(tmp_path, monkeypatch, box_threshold=0.35)
    reasons, compare = ed.cross_check_geom_tol(
        [480, 640], module.ESTIMATOR_NAME, module.SEGMENTER_CONTRACT
    )
    assert "mask_method_disagrees_with_estimator" not in reasons, "the NAME still agrees"
    assert "segmenter_params_disagree_with_geom_tol" in reasons
    (mismatch,) = compare["segmenter_param_disagreements"]
    assert mismatch == {"field": "box_threshold", "geom_tol": 0.35, "this_run": 0.15}


def test_the_cross_check_refuses_a_committed_document_that_records_no_segmenter(
    loaded, tmp_path, monkeypatch
):
    """The shape measure_geom_tol.py used to write over the contract, and the refusal that made the
    gate unreachable rather than wrong: a document at the committed path that names a mask_method
    but no segmenter cannot answer "the same segmenter?", so the run is disqualified. Failing closed
    is the intended direction, and it is still the behaviour — what changed on 2026-08-22 is that
    measure_geom_tol refuses to PRODUCE such a document at that path (blocker 3, now discharged),
    so this shape can only arrive by hand or from an older script."""
    import json

    import measure_est_drift as ed

    module, _ = loaded
    p = tmp_path / "pr08_geom_tol.json"
    p.write_text(json.dumps({
        "resolution_hw": [480, 640],
        "gate_qualified": True,
        "mask_method": {"name": module.ESTIMATOR_NAME, "params": {"checkpoints": {}}},
        "GEOM_TOL_px": 3.4,
    }), encoding="utf-8")
    monkeypatch.setattr(ed, "GEOM_TOL_ARTIFACT", p)
    reasons, _ = ed.cross_check_geom_tol(
        [480, 640], module.ESTIMATOR_NAME, module.SEGMENTER_CONTRACT
    )
    assert "geom_tol_does_not_record_segmenter_params" in reasons
    # The blocker was DISCHARGED, not deleted — this module's own rule is that a condition which
    # disappears looks the same whether it was satisfied or dropped, and only one of those is
    # allowed to shorten the list.
    assert not any("OVERWRITTEN by the measurement" in b
                   for b in module.GATE_QUALIFICATION_BLOCKERS)
    assert any("OVERWRITTEN by the measurement" in d and "merge_committed_contract" in d
               for d in module.GATE_QUALIFICATION_DISCHARGED), (
        "a blocker closed by another file's change must carry that evidence here"
    )


def test_a_measured_artifact_that_carries_the_contract_forward_is_accepted(
    loaded, tmp_path, monkeypatch
):
    """What the fix for that blocker has to look like from this side: the contract at
    mask_method.params.segmenter, which is where the reader already looks for it."""
    import json

    import measure_est_drift as ed

    module, _ = loaded
    p = tmp_path / "pr08_geom_tol.json"
    p.write_text(json.dumps({
        "resolution_hw": [480, 640],
        "gate_qualified": True,
        "mask_method": {
            "name": module.ESTIMATOR_NAME,
            "params": {"segmenter": module.SEGMENTER_CONTRACT},
        },
        "GEOM_TOL_px": 3.4,
    }), encoding="utf-8")
    monkeypatch.setattr(ed, "GEOM_TOL_ARTIFACT", p)
    reasons, compare = ed.cross_check_geom_tol(
        [480, 640], module.ESTIMATOR_NAME, module.SEGMENTER_CONTRACT
    )
    assert reasons == []
    assert compare["geom_tol_segmenter_contract_at"] == "mask_method.params.segmenter"


def test_the_harness_takes_the_object_from_this_module_instead_of_asking_twice(loaded):
    """The discharged blocker, exercised through the harness that discharges it: --object-class
    defaults to this module's prompt, and an explicit value naming another object is fatal rather
    than measured as a large plausible p95."""
    import measure_est_drift as ed

    module, _ = loaded
    est = ed.Estimators(module, MODULE)
    assert est.object_text_prompt == "apple."
    assert ed.resolve_object_class(None, est) == ("apple", "estimator_prompt", [])
    assert ed.resolve_object_class("apple", est)[0] == "apple"
    # AND THE SPELLING THE HELP TEXT INVITES. --object-class defaults to this module's
    # OBJECT_TEXT_PROMPT, which is "apple." with GroundingDINO's terminating period, so an operator
    # copying the documented default and typing it explicitly is the normal case. It used to return
    # the RAW string on the agreement path while the default path returned the normalised one:
    # object_ids() compares label.strip().lower() and does NOT strip the period, so "apple." matched
    # no Replicator label, every frame counted as frames_without_label, and the run ended at
    # coverage 0.0 reporting "the apple is not in this scene" about a notation difference.
    assert ed.resolve_object_class("apple.", est)[0] == "apple"
    assert ed.resolve_object_class("Apple", est)[0] == "apple"
    # Normalising strips the notation and NOTHING ELSE, so this cannot widen what matches.
    with pytest.raises(ed.EstimatorUnavailable):
        ed.resolve_object_class("red apple.", est)
    with pytest.raises(ed.EstimatorUnavailable) as excinfo:
        ed.resolve_object_class("plate", est)
    message = str(excinfo.value)
    assert "'plate'" in message and "'apple.'" in message
    assert "SUBTRACTED from GEOM_TOL" in message


def test_the_harness_reads_this_modules_contract_as_the_segmenter_it_cross_checks(loaded):
    """Estimators is what writes the artifact, so a contract it cannot see is a contract nothing
    downstream can check."""
    import measure_est_drift as ed

    module, _ = loaded
    est = ed.Estimators(module, MODULE)
    assert est.segmenter_contract == module.SEGMENTER_CONTRACT


# -- the detection scores, which are the evidence blocker 2 asks for -----------------------------
#
# The blocker wants "the recorded detection-score distribution and retry counts from a full pass, so
# the retry's contribution is visible rather than assumed". The counts were already here; the scores
# were computed inside ``_best_box`` and thrown away, so a 171 600-frame pass could say how OFTEN
# the retry fired and never how weak the detections it bought were. A 169-frame local audit found
# nine frames whose mask was a confident, well-formed mask of the PLATE — and every one of them
# scored between 0.155 and 0.264 while the correct masks sat at p25 0.758. That separation is only
# visible in the values.


def test_the_winning_score_is_recorded_for_every_frame_that_got_a_box(monkeypatch):
    """One entry per detected frame, in call order, and it is the score of the box SAM 2 was
    prompted with — not the highest score seen, which on the retry path is a different number."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    _install(monkeypatch, detections=[
        [(0.41, [0.0, 0.0, 60.0, 44.0]), (0.88, [12.0, 8.0, 26.0, 22.0])],
        [(0.62, [10.0, 10.0, 30.0, 30.0])],
    ])
    module = _fresh_import(monkeypatch)

    module.segment(_frame())
    module.segment(_frame())
    assert module.DETECTION_SCORES == [pytest.approx(0.88), pytest.approx(0.62)]
    assert module.stats()["n_detection_scores"] == 2
    assert module.stats()["detection_scores_attr"] == "DETECTION_SCORES"


def test_a_frame_with_no_box_records_no_score_rather_than_a_zero(monkeypatch):
    """A zero would be a detection that scored zero. There was no detection: the frame is in
    n_frames_without_detection and in nothing else, and the list is not index-aligned to frames."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    _install(monkeypatch, detections=[[], [], [(0.7, [10.0, 10.0, 30.0, 30.0])]])
    module = _fresh_import(monkeypatch)

    assert not module.segment(_frame()).any()      # first pass empty, retry empty
    module.segment(_frame())
    record = module.stats()
    assert module.DETECTION_SCORES == [pytest.approx(0.7)]
    assert record["n_frames_without_detection"] == 1
    assert record["n_detection_scores"] == 1
    assert record["n_segment_calls"] - record["n_frames_without_detection"] == 1


def test_a_score_the_retry_bought_is_recorded_and_is_below_the_primary_threshold(monkeypatch):
    """THE POINT OF THE LIST. The first pass discards everything under BOX_THRESHOLD, so a recorded
    score below it can only have come from the (0.10, 0.10) retry — which makes "how much of
    coverage did the retry buy, and how weak was it" readable off the values instead of assumed."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    _install(monkeypatch, detections=[[], [(0.11, [10.0, 10.0, 30.0, 30.0])]])
    module = _fresh_import(monkeypatch)

    module.segment(_frame())
    assert module.RETRY_FRAMES == 1 and module.RETRY_RECOVERED_FRAMES == 1
    assert module.DETECTION_SCORES == [pytest.approx(0.11)]
    assert module.DETECTION_SCORES[0] < module.BOX_THRESHOLD
    assert module.stats()["n_detection_scores"] == 1
    assert "below box_threshold" in module.stats()["detection_scores_meaning"]


def test_the_counters_are_cumulative_and_the_module_says_so(loaded):
    """The property every caller has to know about, asserted rather than left to be discovered.

    Nothing here resets them, so two measurements in one interpreter share them — which is why
    measure_geom_tol/measure_est_drift snapshot and difference instead of copying. A module that
    quietly reset them per run would break that arrangement in the other direction, so the
    behaviour is pinned from this side too.
    """
    module, _ = loaded
    for _ in range(3):
        module.segment(_frame())
    first = module.stats()
    assert first["n_segment_calls"] == 3
    module.segment(_frame())
    assert module.stats()["n_segment_calls"] == 4, "the counters are not cumulative any more"
    assert len(module.DETECTION_SCORES) == 4
    assert "snapshot" in first["counters_are_cumulative"]


def test_recording_the_scores_did_not_qualify_the_gate(loaded):
    """Producing evidence and accepting it are two different acts. The blocker that asks for these
    numbers stays in the tuple until a human reads them; nothing in this module may retire it."""
    module, _ = loaded
    assert module.GATE_QUALIFIED is False
    assert any("detection-score distribution and retry counts" in b
               for b in module.GATE_QUALIFICATION_BLOCKERS)
    assert not any("detection-score distribution" in d
                   for d in module.GATE_QUALIFICATION_DISCHARGED)


# -- PR-08 V6: the mask-validity filter --------------------------------------------------------------
#
# THE FINDING IT ANSWERS. Job 189637 drove this adapter over 382 frames of AppleToPlate (24
# episodes, artifact runs/pr08-mask-audit/MASK_AUDIT.json) and a local CPU audit over 169 more, and
# both found the same defect: twelve frames carried a confident, well-formed mask of THE PLATE. Those
# masks are ~30 900 px against a median apple of 6 185, they sit at 0.97-0.98 plate overlap, and they
# produce a centroid, a displacement and a p95 that all look exactly like measurements.
#
# THE ARGUMENT THESE TESTS HAVE TO CARRY. A threshold introduced into a gate path is a number
# somebody chose, and PR-08 §4 step 2 is about not choosing numbers. The defence is that this one
# does not matter: the two populations are separated by a gap so wide that every threshold inside it
# produces the IDENTICAL partition of the audited frames. That is a checkable claim and it is checked
# here rather than asserted in prose.


#: The audit's own per-frame numbers: ``(episode, frame_index, IoU of the returned mask against the
#: colour heuristic, 1 when the audit flagged it plate_overlap)``. EMBEDDED rather than read from the
#: artifact because ``runs/`` is not tracked — a test that skipped when the file was absent would be
#: a test that never ran in CI, which for the load-bearing claim of a pre-registration is no test at
#: all. ``test_the_embedded_audit_numbers_are_the_artifacts_own`` re-checks them against the artifact
#: on any machine that still has it, so the copy cannot drift where it can be compared.
AUDIT_FRAMES: tuple[tuple, ...] = (
    ("000000",0,0.9715), ("000000",159,0.9781), ("000000",160,0.9781), ("000000",167,0.9818),
    ("000000",168,0.9775), ("000000",175,0.9813), ("000000",176,0.9838), ("000000",183,0.9818),
    ("000000",184,0.9817), ("000000",196,0.9796), ("000000",393,0.9737), ("000000",470,0.9733),
    ("000000",471,0.9742), ("000000",478,0.9734), ("000000",479,0.9734), ("000000",589,0.9749),
    ("000018",0,0.9688), ("000018",112,0.9703), ("000018",113,0.9712), ("000018",120,0.9693),
    ("000018",121,0.9676), ("000018",128,0.9719), ("000018",129,0.9697), ("000018",132,0.9748),
    ("000018",136,0.9714), ("000018",137,0.9751), ("000018",140,0.9694), ("000018",141,0.9714),
    ("000018",265,0.9792), ("000018",397,0.9755), ("000036",0,0.9748), ("000036",92,0.9755),
    ("000036",93,0.9757), ("000036",100,0.97), ("000036",101,0.9674), ("000036",108,0.9554),
    ("000036",109,0.9546), ("000036",116,0.9677), ("000036",117,0.9637), ("000036",131,0.9661),
    ("000036",133,0.9713), ("000036",134,0.9705), ("000036",136,0.9671), ("000036",137,0.9725),
    ("000036",261,0.9714), ("000036",392,0.9764), ("000055",0,0.9659), ("000055",104,0.9782),
    ("000055",105,0.975), ("000055",112,0.9661), ("000055",113,0.9679), ("000055",118,0.9709),
    ("000055",119,0.9642), ("000055",120,0.9641), ("000055",121,0.9662), ("000055",128,0.9747),
    ("000055",129,0.9747), ("000055",133,0.9731), ("000055",267,0.9695), ("000055",400,0.9688),
    ("000073",0,0.9792), ("000073",64,0.9775), ("000073",65,0.9786), ("000073",72,0.9828),
    ("000073",73,0.9815), ("000073",80,0.9579), ("000073",81,0.9706), ("000073",88,0.9522),
    ("000073",89,0.9583), ("000073",103,0.894), ("000073",104,0.8989), ("000073",107,0.8932),
    ("000073",108,0.8897), ("000073",166,0.931), ("000073",331,0.9773), ("000073",497,0.978),
    ("000091",0,0.9829), ("000091",109,0.9692), ("000091",110,0.9716), ("000091",117,0.9641),
    ("000091",118,0.9632), ("000091",125,0.9449), ("000091",126,0.9414), ("000091",133,0.8984),
    ("000091",134,0.9008), ("000091",150,0.8962), ("000091",151,0.8904), ("000091",156,0.8638),
    ("000091",157,0.8871), ("000091",162,0.9519), ("000091",325,0.9675), ("000091",487,0.9649),
    ("000094",81,0.9827), ("000094",82,0.9814), ("000094",90,0.973), ("000094",91,0.9687),
    ("000094",106,0.8977), ("000094",107,0.8585), ("000094",108,0.0,1), ("000094",109,0.0,1),
    ("000094",129,0.0,1), ("000094",130,0.0,1), ("000094",133,0.0,1), ("000094",134,0.0,1),
    ("000094",136,0.0,1), ("000094",137,0.0,1), ("000094",143,0.0,1), ("000094",144,0.0,1),
    ("000094",149,0.0,1), ("000094",150,0.7834), ("000094",151,0.8493), ("000094",152,0.0,1),
    ("000094",153,0.8765), ("000094",154,0.9266), ("000094",208,0.9438), ("000094",209,0.9458),
    ("000110",0,0.9598), ("000110",62,0.9676), ("000110",63,0.9682), ("000110",70,0.9484),
    ("000110",71,0.9483), ("000110",78,0.9506), ("000110",79,0.9508), ("000110",86,0.8679),
    ("000110",87,0.9362), ("000110",89,0.9328), ("000110",90,0.9186), ("000110",138,0.96),
    ("000110",277,0.9608), ("000110",415,0.9775), ("000128",0,0.9647), ("000128",94,0.9699),
    ("000128",95,0.9697), ("000128",102,0.9582), ("000128",103,0.9594), ("000128",110,0.9589),
    ("000128",111,0.9651), ("000128",118,0.9616), ("000128",119,0.9689), ("000128",150,0.9509),
    ("000128",243,0.7492), ("000128",244,0.7712), ("000128",245,0.7918), ("000128",299,0.9558),
    ("000128",449,0.9566), ("000146",0,0.9718), ("000146",116,0.9767), ("000146",117,0.9733),
    ("000146",124,0.9789), ("000146",125,0.9798), ("000146",132,0.9759), ("000146",133,0.9781),
    ("000146",139,0.9761), ("000146",140,0.9747), ("000146",141,0.9765), ("000146",221,0.974),
    ("000146",222,0.9716), ("000146",241,0.9725), ("000146",242,0.969), ("000146",279,0.9751),
    ("000146",418,0.9708), ("000165",0,0.9603), ("000165",79,0.9394), ("000165",80,0.9356),
    ("000165",81,0.9359), ("000165",90,0.9648), ("000165",91,0.9634), ("000165",98,0.9438),
    ("000165",99,0.9407), ("000165",106,0.9422), ("000165",107,0.9439), ("000165",114,0.9533),
    ("000165",115,0.9526), ("000165",127,0.9495), ("000165",254,0.9553), ("000165",381,0.9666),
    ("000183",0,0.9705), ("000183",33,0.9694), ("000183",34,0.9726), ("000183",41,0.9718),
    ("000183",42,0.973), ("000183",49,0.9708), ("000183",50,0.9683), ("000183",57,0.9628),
    ("000183",58,0.9742), ("000183",123,0.9577), ("000183",124,0.9646), ("000183",126,0.9612),
    ("000183",129,0.9545), ("000183",130,0.9592), ("000183",252,0.968), ("000183",378,0.9746),
    ("000201",0,0.9675), ("000201",93,0.9714), ("000201",94,0.9732), ("000201",101,0.9711),
    ("000201",102,0.9704), ("000201",109,0.9733), ("000201",110,0.9731), ("000201",117,0.9764),
    ("000201",118,0.9765), ("000201",124,0.9771), ("000201",234,0.971), ("000201",235,0.9729),
    ("000201",249,0.9577), ("000201",270,0.9704), ("000201",271,0.9716), ("000201",373,0.9686),
    ("000219",0,0.9667), ("000219",95,0.9682), ("000219",96,0.9712), ("000219",103,0.9677),
    ("000219",104,0.9726), ("000219",111,0.9725), ("000219",112,0.9714), ("000219",119,0.9732),
    ("000219",120,0.9719), ("000219",128,0.9333), ("000219",166,0.946), ("000219",167,0.9449),
    ("000219",171,0.9495), ("000219",172,0.9453), ("000219",256,0.9702), ("000219",384,0.968),
    ("000237",0,0.9382), ("000237",117,0.9396), ("000237",118,0.9354), ("000237",125,0.9377),
    ("000237",126,0.9374), ("000237",133,0.9359), ("000237",134,0.9381), ("000237",141,0.937),
    ("000237",142,0.9396), ("000237",149,0.9348), ("000237",189,0.9514), ("000237",190,0.9506),
    ("000237",194,0.9537), ("000237",195,0.9555), ("000237",298,0.9586), ("000237",447,0.9626),
    ("000256",0,0.9627), ("000256",107,0.9238), ("000256",108,0.9227), ("000256",115,0.9212),
    ("000256",116,0.9141), ("000256",123,0.9441), ("000256",124,0.9475), ("000256",130,0.9354),
    ("000256",131,0.9385), ("000256",132,0.9469), ("000256",218,0.9245), ("000256",219,0.9182),
    ("000256",222,0.9248), ("000256",223,0.9372), ("000256",260,0.9659), ("000256",390,0.9771),
    ("000274",0,0.9723), ("000274",149,0.966), ("000274",150,0.9593), ("000274",157,0.9572),
    ("000274",158,0.952), ("000274",165,0.9386), ("000274",166,0.9443), ("000274",171,0.9306),
    ("000274",173,0.9329), ("000274",174,0.9386), ("000274",280,0.959), ("000274",281,0.9533),
    ("000274",286,0.9514), ("000274",287,0.9291), ("000274",343,0.957), ("000274",514,0.9608),
    ("000292",0,0.9719), ("000292",20,0.9783), ("000292",21,0.9777), ("000292",28,0.979),
    ("000292",29,0.9777), ("000292",36,0.9756), ("000292",37,0.9753), ("000292",44,0.9744),
    ("000292",45,0.9725), ("000292",68,0.9633), ("000292",69,0.9623), ("000292",77,0.9643),
    ("000292",78,0.9631), ("000292",126,0.9712), ("000292",252,0.9731), ("000292",378,0.9615),
    ("000310",0,0.9767), ("000310",138,0.959), ("000310",139,0.9615), ("000310",146,0.9704),
    ("000310",147,0.9675), ("000310",153,0.9692), ("000310",154,0.9658), ("000310",155,0.9648),
    ("000310",158,0.9548), ("000310",159,0.9777), ("000310",162,0.9776), ("000310",163,0.9784),
    ("000310",186,0.9667), ("000310",371,0.9747), ("000310",557,0.9751), ("000328",0,0.9748),
    ("000328",99,0.9632), ("000328",100,0.9645), ("000328",107,0.9631), ("000328",108,0.9543),
    ("000328",115,0.9476), ("000328",116,0.9589), ("000328",123,0.954), ("000328",124,0.9539),
    ("000328",138,0.9469), ("000328",139,0.9457), ("000328",149,0.9447), ("000328",168,0.9478),
    ("000328",169,0.945), ("000328",299,0.9663), ("000328",448,0.9533), ("000346",0,0.9725),
    ("000346",124,0.9586), ("000346",125,0.955), ("000346",132,0.9534), ("000346",133,0.9553),
    ("000346",140,0.9472), ("000346",141,0.9324), ("000346",148,0.9715), ("000346",149,0.9688),
    ("000346",154,0.9723), ("000346",179,0.9178), ("000346",180,0.9195), ("000346",195,0.9298),
    ("000346",196,0.9267), ("000346",307,0.973), ("000346",461,0.973), ("000365",0,0.9756),
    ("000365",80,0.9407), ("000365",81,0.9176), ("000365",83,0.9471), ("000365",84,0.9536),
    ("000365",101,0.9702), ("000365",102,0.9674), ("000365",109,0.9663), ("000365",110,0.9655),
    ("000365",117,0.9671), ("000365",118,0.9705), ("000365",121,0.9722), ("000365",125,0.9749),
    ("000365",126,0.9671), ("000365",242,0.969), ("000365",363,0.974), ("000383",0,0.9469),
    ("000383",71,0.9459), ("000383",72,0.9482), ("000383",79,0.9256), ("000383",80,0.9391),
    ("000383",87,0.8162), ("000383",88,0.8497), ("000383",89,0.906), ("000383",90,0.9008),
    ("000383",94,0.9131), ("000383",95,0.9281), ("000383",96,0.9628), ("000383",134,0.9392),
    ("000383",267,0.9694), ("000383",401,0.9682), ("000401",0,0.9747), ("000401",104,0.9111),
    ("000401",105,0.9095), ("000401",112,0.9356), ("000401",113,0.9286), ("000401",118,0.9348),
    ("000401",119,0.9448), ("000401",120,0.9314), ("000401",121,0.9357), ("000401",125,0.9383),
    ("000401",128,0.9414), ("000401",129,0.9414), ("000401",136,0.9373), ("000401",137,0.9361),
    ("000401",251,0.9719), ("000401",376,0.9755),
)

#: The 12 the audit flagged as the plate, by identity rather than by IoU — otherwise the sweep below
#: would be checking that a threshold partitions the frames the way the threshold partitions them.
AUDIT_PLATE_FRAMES = frozenset((ep, fi) for ep, fi, _iou, *flag in AUDIT_FRAMES if flag)

AUDIT_ARTIFACT = _REPO / "runs" / "pr08-mask-audit" / "MASK_AUDIT.json"


def _audit_partition(threshold: float) -> frozenset:
    """Which audited frames the filter refuses at ``threshold``."""
    return frozenset(
        (ep, fi) for ep, fi, iou, *_ in AUDIT_FRAMES if iou < threshold
    )


def test_the_two_populations_are_separated_by_a_gap_and_not_by_a_threshold():
    """The whole basis for admitting a number here: there is nothing in between to get wrong."""
    correct = [iou for ep, fi, iou, *_ in AUDIT_FRAMES if (ep, fi) not in AUDIT_PLATE_FRAMES]
    plate = [iou for ep, fi, iou, *_ in AUDIT_FRAMES if (ep, fi) in AUDIT_PLATE_FRAMES]

    assert len(AUDIT_FRAMES) == 382 and len(plate) == 12
    assert max(plate) == 0.0, "a mask of the plate contains NO warm apple pixels, exactly"
    assert min(correct) == pytest.approx(0.7492), "the lowest correct mask still agrees strongly"


def test_every_threshold_in_the_gap_partitions_the_audited_frames_identically():
    """THE INSENSITIVITY EVIDENCE, made checkable rather than assertable.

    PR-08 V6 rests on this: the value of ``MASK_VALIDITY_MIN_IOU`` cannot be tuned to flatter a
    number, because moving it anywhere inside the gap changes nothing at all. The sweep is run at 1
    pp steps across the whole gap and the partition is compared to the frames a PERSON flagged as
    the plate in the contact sheets, not to anything this threshold computed.
    """
    thresholds = [round(0.01 * k, 2) for k in range(1, 75)]  # 0.01 .. 0.74, i.e. inside (0, 0.7492)
    for t in thresholds:
        assert _audit_partition(t) == AUDIT_PLATE_FRAMES, (
            f"the partition moved at threshold {t}: the gap runs from 0.0 to 0.7492 and every cut "
            "inside it must refuse exactly the twelve plate frames and nothing else"
        )


def test_the_threshold_this_module_ships_is_inside_that_range(loaded):
    """A plateau nobody's value sits on would be an argument about a different number."""
    module, _ = loaded
    assert 0.0 < module.MASK_VALIDITY_MIN_IOU < 0.7492
    assert _audit_partition(module.MASK_VALIDITY_MIN_IOU) == AUDIT_PLATE_FRAMES


def test_the_embedded_audit_numbers_are_the_artifacts_own():
    """Where the artifact is still on disk, the embedded copy is compared to it frame for frame."""
    if not AUDIT_ARTIFACT.is_file():
        pytest.skip(f"{AUDIT_ARTIFACT} is not on this machine (runs/ is not tracked)")
    import json

    frames = json.loads(AUDIT_ARTIFACT.read_text(encoding="utf-8"))["frames"]
    theirs = {
        (f["episode"].replace("episode_", ""), int(f["frame_index"])):
            (round(float(f["warm_apple_iou"]), 4), "plate_overlap" in f["flags"])
        for f in frames
    }
    ours = {(ep, fi): (iou, bool(flag)) for ep, fi, iou, *flag in AUDIT_FRAMES}
    assert ours == theirs


# -- what the filter does to a frame -------------------------------------------------------------


def test_a_confident_mask_of_the_wrong_object_is_refused(monkeypatch):
    """The finding, reproduced in miniature: a well-formed mask containing none of the fruit.

    It comes back all-False — which both PR-08 §4 harnesses already drop and count — rather than as
    a centroid that would have looked like a measurement.
    """
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    state = _install(monkeypatch, detections=[[(0.26, [34.0, 4.0, 60.0, 44.0])]])
    module = _fresh_import(monkeypatch)

    mask = module.segment(_frame())
    assert not mask.any()
    assert module.MASK_REFUSED_FRAMES == 1
    assert module.MASK_VALIDITY_IOU == [pytest.approx(0.0)]
    # The box SAM 2 was prompted with is still the detector's own: nothing was re-detected.
    assert state["predictors"][0].boxes_seen[0].tolist() == [34.0, 4.0, 60.0, 44.0]


def test_a_correct_mask_is_returned_untouched(monkeypatch):
    """The filter is a gate on the output, not a post-processing of it: an accepted mask is bit for
    bit the mask SAM 2 drew."""
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    _install(monkeypatch, detections=[[(0.83, [10.0, 10.0, 30.0, 30.0])]])
    module = _fresh_import(monkeypatch)

    mask = module.segment(_frame())
    expected = np.zeros((48, 64), dtype=bool)
    expected[10:30, 10:30] = True
    assert np.array_equal(mask, expected)
    assert module.MASK_REFUSED_FRAMES == 0
    assert module.MASK_VALIDITY_IOU == [pytest.approx(1.0)]


def test_a_refusal_is_a_third_event_and_never_collapses_into_the_other_two(monkeypatch):
    """No detection, an empty mask and a refusal all drop the step and all mean different things.

    Collapsing them would make a coverage shortfall unreadable: "the apple was not there", "the
    segmenter failed on a frame where it was" and "the segmenter masked the plate" call for three
    different responses, and only the third one is a defect in what the mask says.
    """
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    _install(monkeypatch, detections=[
        [],                                         # no detection
        [(0.9, [4.0, 4.0, 4.0, 4.0])],              # a box, and SAM 2 fills nothing
        [(0.26, [34.0, 4.0, 60.0, 44.0])],          # a box, filled, and it is not the fruit
        [(0.83, [10.0, 10.0, 30.0, 30.0])],         # the fruit
    ])
    module = _fresh_import(monkeypatch)

    for _ in range(4):
        module.segment(_frame())

    record = module.stats()
    assert record["n_segment_calls"] == 4
    assert record["n_frames_without_detection"] == 1
    assert record["n_frames_with_empty_mask"] == 1
    assert record["n_frames_mask_refused"] == 1
    assert record["n_frames_mask_refused_no_reference"] == 0
    # The check ran on the two frames that produced a mask, and on no others.
    assert record["n_mask_validity_iou"] == 2
    assert len(module.MASK_VALIDITY_IOU) == (
        record["n_segment_calls"]
        - record["n_frames_without_detection"]
        - record["n_frames_with_empty_mask"]
    )


def test_a_frame_with_no_visible_fruit_is_refused_and_counted_as_the_hard_case(monkeypatch):
    """THE THREAT TO VALIDITY, pinned from the code's side.

    When the fruit is not visible at all, nothing here can confirm any mask, so the frame is
    refused — and that refusal removes a HARD frame from the measured population rather than a wrong
    one. For EST_DRIFT_P95, a p95 that is SUBTRACTED from GEOM_TOL, that plausibly makes the number
    smaller and the tolerance wider, i.e. it errs in the generator's favour. It is counted apart so
    the size of the effect is a number in every artifact rather than an argument.
    """
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    _install(monkeypatch, detections=[[(0.21, [10.0, 10.0, 30.0, 30.0])]])
    module = _fresh_import(monkeypatch)

    assert not module.segment(_frame(apple=None)).any()
    assert module.MASK_REFUSED_FRAMES == 1
    assert module.MASK_REFUSED_NO_REFERENCE_FRAMES == 1
    assert module.NO_DETECTION_FRAMES == 0
    assert "SUBTRACTED" in module.stats()["mask_validity_threat_to_validity"]


def test_a_refused_frame_still_records_its_detection_score(monkeypatch):
    """The evidence blocker 2 asks for must not be destroyed by the fix for blocker 1.

    The plate masks in the audit scored 0.167-0.309 against a median 0.829 for the correct ones, and
    that separation is only visible if a refused frame's score is still in the list.
    """
    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
    _install(monkeypatch, detections=[[(0.26, [34.0, 4.0, 60.0, 44.0])]])
    module = _fresh_import(monkeypatch)

    module.segment(_frame())
    assert module.DETECTION_SCORES == [pytest.approx(0.26)]
    assert module.stats()["n_detection_scores"] == 1


def test_the_filter_changed_no_detection_parameter(loaded):
    """§4 step 2: the operating point is the GENERATOR's, and a validity check on the output is not
    permission to go and improve the detector."""
    module, state = loaded
    module.segment(_frame())

    assert module.BOX_THRESHOLD == 0.15
    assert module.TEXT_THRESHOLD == 0.25
    assert module.RETRY_BOX_THRESHOLD == 0.1
    assert module.RETRY_TEXT_THRESHOLD == 0.1
    assert module.BOX_SELECTION == "highest_score"
    assert module.OBJECT_TEXT_PROMPT == "apple."
    assert state["thresholds"] == [(0.15, 0.25)]


def test_the_colour_reference_is_the_census_own_predicate(loaded):
    """Pinned to the copy in scripts/audit_apple_masks.py pixel for pixel.

    The predicate is restated in the adapter rather than imported, so the thing that keeps the two
    from drifting has to be a test. If they drift, "the fruit is not visible here" stops meaning the
    same thing in the census, in the audit and in this refusal.
    """
    module, _ = loaded
    import audit_apple_masks as audit

    rng = np.random.default_rng(7)
    for _ in range(5):
        frame = rng.integers(0, 256, size=(31, 37, 3), dtype=np.uint8)
        assert np.array_equal(module.object_color_reference(frame), audit.warm_apple_mask(frame))
    assert np.array_equal(
        module.object_color_reference(_frame()), audit.warm_apple_mask(_frame())
    )


def test_the_reference_finds_the_fruit_and_nothing_else_in_a_stub_frame(loaded):
    """A sanity check on the fixture itself: a test whose frames were all reference-empty would pass
    the refusal tests for the wrong reason."""
    module, _ = loaded
    reference = module.object_color_reference(_frame())
    expected = np.zeros((48, 64), dtype=bool)
    expected[10:30, 10:30] = True
    assert np.array_equal(reference, expected)
    assert not module.object_color_reference(_frame(apple=None)).any()


def test_the_contract_carries_the_filter_so_two_artifacts_can_be_compared(loaded):
    """§6 subtracts GEOM_TOL and EST_DRIFT_P95. A GEOM_TOL measured with the filter minus an
    EST_DRIFT_P95 measured without it is a subtraction across two different frame populations, and
    it would still look like arithmetic. Absence is a disagreement in contract_disagreements(), so
    recording it here is what makes the mismatch refusable."""
    module, _ = loaded
    assert module.SEGMENTER_CONTRACT["mask_validity_min_iou"] == module.MASK_VALIDITY_MIN_IOU
    assert module.SEGMENTER_CONTRACT["mask_validity_reference"] == module.MASK_VALIDITY_REFERENCE
    assert f"mask_val_min_iou={module.MASK_VALIDITY_MIN_IOU}" in module.ESTIMATOR_VERSION
    assert _contract_doc()["segmenter"]["mask_validity_min_iou"] == module.MASK_VALIDITY_MIN_IOU


def test_measure_geom_tol_differences_the_new_counters_as_this_runs_numbers(loaded):
    """Otherwise the refusal count in an artifact would be a lifetime total of the process."""
    import measure_geom_tol as mgt

    module, _ = loaded
    for key in ("n_frames_mask_refused", "n_frames_mask_refused_no_reference",
                "n_mask_validity_iou"):
        assert key in mgt.ADAPTER_RUN_COUNTERS
        assert key in module.stats()


def test_producing_the_fix_did_not_accept_it(loaded):
    """Blocker 1 is discharged by a person looking at the evidence and editing the tuple, not by an
    adapter that has stopped producing the masks the evidence was about. Nothing here may flip it,
    and blocker 3 — per-frame segmentation vs upstream's propagation — is untouched by any of it."""
    module, _ = loaded
    assert module.GATE_QUALIFIED is False
    assert any("NOBODY HAS LOOKED AT A MASK" in b for b in module.GATE_QUALIFICATION_BLOCKERS)
    assert any("PER-FRAME SEGMENTATION IS NOT UPSTREAM'S PROPAGATION" in b
               for b in module.GATE_QUALIFICATION_BLOCKERS)
    assert len(module.GATE_QUALIFICATION_BLOCKERS) == 3
    assert not any("mask-validity" in d.lower() for d in module.GATE_QUALIFICATION_DISCHARGED)
