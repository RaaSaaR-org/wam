"""Cosmos3-Edge backbone adapter (E-01/E-02/E-03; weights not staged).

``BackboneAdapter`` over NVIDIA's Cosmos 3 Edge omni model — the 4B checkpoint the edge
sub-project is built around. The module imports torch-free and diffusers-free: every heavy
import happens inside :meth:`Cosmos3EdgeAdapter.load`, and construction never touches the
filesystem, the network or a GPU (FR-09/AC-05).

Provenance is marked throughout. ``[OK]`` = read off code or a fetched primary source,
``[doc]`` = vendor prose, ``[?]`` = open. Nothing marked ``[?]`` is given a value here.

VERIFIED [OK]
- Model ids exist (HTTP 200): ``nvidia/Cosmos3-Edge-Policy-DROID`` (post-trained policy) and
  ``nvidia/Cosmos3-Edge`` (base). Both are ~9.2 GB on disk; resident weights are 9.14 GB
  (transformer 3.37B + vision encoder 0.49B + VAE 0.70B = 4.56B params, measured from the
  safetensors headers). E-03.
- Transformer geometry, from ``transformer/config.json`` of BOTH repos: ``hidden_size`` 2048,
  ``num_hidden_layers`` 28, ``num_attention_heads`` 16, ``num_key_value_heads`` 8,
  ``head_dim`` 128, ``action_dim`` 64, ``action_gen`` true, ``num_embodiment_domains`` 32.
- ``action_dim = 64`` is a global MAXIMUM action width shared by every embodiment (the
  training config calls it ``max_action_dim``). Narrower domains are zero-padded on the
  channel axis and predictions are sliced back down
  (``pipeline_cosmos3_omni.py:861-869``, ``:929-930``, ``:1769-1770``). E-02.
- An embodiment is a *string* -> *domain id* -> *row index* into two learned
  ``nn.Embedding`` tables inside ``DomainAwareLinear``
  (``transformer_cosmos3.py:154-177``, instantiated at ``:381-382``). Adding a G1/Dex3 row is
  therefore neither an architecture change nor a config entry: it is a new row that must be
  TRAINED. In the released policy checkpoint exactly ONE row (droid, d8, width 8) is trained;
  every other row — ``agibotworld`` included — sits at random init. The BASE checkpoint has 10
  trained rows, ``agibotworld`` at width 29 among them, so the base is the only warm-start
  source for a humanoid row. **Reproduce with**
  ``.venv/bin/python scripts/probe_cosmos3_domain_rows.py nvidia/Cosmos3-Edge[-Policy-DROID]``
  — header + one 8 MB tensor over HTTP Range, no checkpoint download, ~30 s. Run on both
  checkpoints 2026-08-16; the recovered widths reproduce NVIDIA's published embodiment table
  exactly (av 9, camera_pose 9, hand_pose 57, franka-dual 20, agibotworld 29, ...), which is
  what makes the policy-checkpoint result credible rather than merely asserted. E-02.
- Text is structurally required, in the plumbing and on the action tokens' attention path:
  ``prompt`` is the first required positional of ``Cosmos3OmniPipeline.__call__``
  (``:1250-1252``), ``text_tokenizer`` is a required pipeline component (``:366``),
  ``input_ids``/``text_indexes``/``und_len`` are required positionals of
  ``Cosmos3OmniTransformer.forward`` (``transformer_cosmos3.py:554-559``), and every
  generation token cross-attends to the text keys/values in every layer (``:82-94``,
  ``:683-692``). ``prompt=""`` is *accepted* (``check_inputs`` only type-checks, ``:963-966``)
  but the chat template plus two special tokens guarantee ``und_len >= 3`` anyway, so the
  model always sees a text stream. See :meth:`Cosmos3EdgeAdapter.condition_text`. E-01.
- The diffusers Cosmos 3 graph has NO proprioception input. Neither
  ``Cosmos3OmniPipeline.__call__`` nor ``Cosmos3OmniTransformer.forward`` takes a state /
  proprio argument (grepped: zero hits for ``use_state``/``proprio`` in either file), even
  though NVIDIA's DROID post-training recipe trains with ``use_state=true``. Hence
  :meth:`Cosmos3EdgeAdapter.condition_state` raises rather than inventing a slot. E-02.
- Only BF16 is tested by NVIDIA; FP4/FP8/FP16 are explicitly not supported (both model
  cards). The 5090's fp8/fp4 tensor cores therefore buy nothing here — there is no
  quantization lever. E-03.
- The 4B fits this workstation's RTX 5090 (32607 MiB): ~12-14 GB estimated inference peak.
  The 16B Cosmos3-Nano did not (T-24 peaked at 36.2 GB). E-03.

OPEN [?] — deliberately NOT given a default anywhere in this module
- The emitted action width for ``droid_lerobot`` is disputed. diffusers'
  ``_EMBODIMENT_TO_RAW_ACTION_DIM`` says 10 (``pipeline_cosmos3_omni.py:221``); the
  Policy-DROID model card says 8 ("8 action values per timestep", chunks ``[16, 8]`` /
  ``[32, 8]``), and the policy checkpoint measures 8 live channels. 10 != 8, so
  :attr:`Cosmos3EdgeConfig.raw_action_dim` has NO default and
  :meth:`Cosmos3EdgeAdapter.require_raw_action_dim` raises until a real forward pass or
  cosmos-framework source settles it.
- Which transformer depth to read intermediate features from. Cosmos 3 publishes no
  feature-readout recipe and this repo has measured none (the Wan readout depths came from a
  measurement on G1 action labels; nothing equivalent exists here). :meth:`features` refuses
  rather than guessing a layer index.
- Whether a 28-dim G1/Dex3 embodiment converges head-only / with LoRA, and how many
  iterations a 3152-episode corpus needs. NVIDIA's reference run is 256 GPUs x 10 000 iters;
  no scaled-down recipe is published. E-02.
- On-device feasibility. NVIDIA's 15 Hz figure holds on a Jetson AGX Thor T5000 only (the
  T4000 misses by 3.5%), and no Cosmos 3 action-path measurement exists for ANY Orin part.
  The G1's compute module is a configurable accessory whose variant is unconfirmed. E-03.

Swapping this adapter in must not change the data schema or the robot API (FR-09/AC-05).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: Stable backbone identifier (AC-04 traceability).
COSMOS3_EDGE_NAME = "cosmos3-edge"

#: Post-trained action policy — the checkpoint the edge sub-project targets. [OK, HTTP 200]
COSMOS3_EDGE_POLICY_MODEL_ID = "nvidia/Cosmos3-Edge-Policy-DROID"
#: Base omni checkpoint (10 trained embodiment rows, agibotworld@29 among them; the only
#: warm-start source for a humanoid). [OK, measured — scripts/probe_cosmos3_domain_rows.py]
COSMOS3_EDGE_BASE_MODEL_ID = "nvidia/Cosmos3-Edge"

#: Transformer hidden size = width of any readout. [OK, transformer/config.json]
COSMOS3_EDGE_HIDDEN_DIM = 2048
#: Transformer depth. [OK, transformer/config.json]
COSMOS3_EDGE_NUM_LAYERS = 28
#: Global MAXIMUM action width shared by all embodiments (training config: max_action_dim). [OK]
COSMOS3_EDGE_MAX_ACTION_DIM = 64
#: Fixed size of the per-embodiment weight table. [OK, transformer/config.json]
COSMOS3_EDGE_NUM_EMBODIMENT_DOMAINS = 32

#: The one embodiment row actually trained in the released policy checkpoint. [OK, measured]
POLICY_DOMAIN_NAME = "droid_lerobot"
#: Canonical G1 + Dex3 action width in this repo (14 arm + 14 hand). NOT a Cosmos embodiment:
#: no such domain row exists or is trained in either checkpoint — see E-02. Kept as a named
#: constant so nothing downstream retypes 28 as if Cosmos already supported it.
G1_DEX3_ACTION_DIM = 28

#: Policy defaults declared by the checkpoint itself. [OK, checkpoint.json "policy" block]
DEFAULT_ACTION_CHUNK_SIZE = 32
DEFAULT_CONDITIONING_FPS = 15.0
#: Sampling settings for the real-time PyTorch policy path. [doc, model card + tech report 6.3.1]
DEFAULT_NUM_INFERENCE_STEPS = 4
DEFAULT_GUIDANCE_SCALE = 3.0
#: Action conditioning resolution tier. [OK, CosmosActionCondition.resolution_tier default]
DEFAULT_RESOLUTION_TIER = 480
ACTION_RESOLUTION_TIERS: tuple[int, ...] = (256, 480, 704, 720)
#: Camera-perspective labels the action caption understands. [OK, pipeline_cosmos3_omni.py:170-176]
VIEW_POINTS: tuple[str, ...] = ("ego_view", "third_person_view", "wrist_view", "concat_view")
#: NVIDIA tests BF16 only; FP4/FP8/FP16 are explicitly unsupported. [OK, both model cards]
TESTED_DTYPE = "bfloat16"
#: What this CONFIG will accept — deliberately NOT a claim about what NVIDIA supports. Only
#: :data:`TESTED_DTYPE` is on-recipe; float16 and float32 are permitted for CPU-side
#: construction and debugging, and a number produced under either is off-recipe by definition.
ACCEPTED_DTYPES: tuple[str, ...] = ("bfloat16", "float16", "float32")
#: Deprecated alias — the old name read as a vendor-support claim, which it never was.
SUPPORTED_DTYPES: tuple[str, ...] = ACCEPTED_DTYPES

_WEIGHTS_MISSING_MSG = (
    "Cosmos3-Edge weights not available — stage a local snapshot and pass "
    "checkpoint_path=<snapshot dir> (or model_id='nvidia/Cosmos3-Edge-Policy-DROID' with "
    "allow_download=True, ~9.2 GB / 9.14 GB resident), and install diffusers>=0.39; nothing "
    "is downloaded implicitly."
)

_RAW_ACTION_DIM_UNSETTLED_MSG = (
    "Cosmos3-Edge raw action width is UNSETTLED for domain_name={domain!r} — the sources "
    "disagree: diffusers 0.39.0 _EMBODIMENT_TO_RAW_ACTION_DIM says 10 for 'droid_lerobot' "
    "(pipeline_cosmos3_omni.py:221) while the Cosmos3-Edge-Policy-DROID model card says 8 "
    "('8 action values per timestep', chunks [16, 8] / [32, 8]) and the checkpoint measures 8 "
    "live channels. Set Cosmos3EdgeConfig(raw_action_dim=...) explicitly once a real forward "
    "pass (or cosmos-framework source) settles it; this adapter will not guess a width."
)

_NO_STATE_INPUT_MSG = (
    "Cosmos3-Edge exposes no proprioception input — neither Cosmos3OmniPipeline.__call__ nor "
    "Cosmos3OmniTransformer.forward takes a state/proprio argument (diffusers 0.39.0), even "
    "though NVIDIA's DROID post-training recipe trains with use_state=true. There is no "
    "verified slot to project a StateEncoder embedding into, so this adapter refuses rather "
    "than inventing one. Settling it needs cosmos-framework's action policy server (see "
    "E-02); until then state conditioning is out of scope for this backbone."
)

_FEATURE_READOUT_UNVERIFIED_MSG = (
    "Cosmos3-Edge intermediate-feature readout depth is unverified — Cosmos 3 publishes no "
    "readout recipe and none has been measured here. Hooking a layer index picked by analogy "
    f"with Wan would be a guess. The width is known ({COSMOS3_EDGE_HIDDEN_DIM}, "
    "transformer/config.json hidden_size) over "
    f"{COSMOS3_EDGE_NUM_LAYERS} layers; which depths carry action-relevant signal is not. "
    "Settle it with a readout ablation on staged weights before enabling this path."
)


@dataclass(frozen=True)
class Cosmos3EdgeConfig:
    """Declarative config for :class:`Cosmos3EdgeAdapter`.

    A stdlib frozen dataclass, not a pydantic model, on purpose: it is deliberately NOT a
    member of the ``BackboneConfig`` discriminated union that ``build_backbone_config``
    validates. Post-training a Cosmos 3 embodiment row is an unpriced open question (E-02),
    so no training config may select this backbone yet — and a config that cannot be trained
    should not be able to masquerade as one that can.

    Torch-free by construction: every field is a JSON primitive, so the whole thing stays
    YAML-round-trippable and hashable by ``wam.interfaces.versioning.config_hash`` (AC-04).
    """

    #: Hub repo. Defaults to the post-trained POLICY checkpoint; ``COSMOS3_EDGE_BASE_MODEL_ID``
    #: is the base (its 10 trained embodiment rows are the only warm-start source, E-02).
    model_id: str = COSMOS3_EDGE_POLICY_MODEL_ID
    #: Local snapshot directory; takes precedence over ``model_id`` when set.
    checkpoint_path: str | None = None
    #: Stays False so nothing ever blocks on a 9.2 GB pull implicitly.
    allow_download: bool = False
    #: Only ``bfloat16`` is tested by NVIDIA; the others are accepted but unsupported upstream.
    dtype: str = TESTED_DTYPE
    device: str = "cuda"
    #: Width of any intermediate-feature readout = transformer hidden_size. [OK]
    feature_dim: int = COSMOS3_EDGE_HIDDEN_DIM
    #: Embodiment row. The authoritative name table lives in diffusers/cosmos-framework and is
    #: checked at load time, not here — the installed diffusers copy is known to be staler than
    #: upstream (E-02), so a local mirror would go wrong silently.
    domain_name: str = POLICY_DOMAIN_NAME
    action_chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE
    conditioning_fps: float = DEFAULT_CONDITIONING_FPS
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    resolution_tier: int = DEFAULT_RESOLUTION_TIER
    view_point: str = "ego_view"
    #: NO DEFAULT ON PURPOSE [?]: diffusers says 10 for droid_lerobot, the model card says 8.
    #: ``None`` means "unsettled" and every use raises — see ``require_raw_action_dim``.
    raw_action_dim: int | None = None
    #: An empty instruction is accepted by the pipeline but is off-distribution (E-01), so it
    #: has to be opted into deliberately rather than reached by passing "".
    allow_empty_instruction: bool = False

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {self.feature_dim}")
        if self.action_chunk_size < 1:
            raise ValueError(f"action_chunk_size must be >= 1, got {self.action_chunk_size}")
        if self.conditioning_fps <= 0:
            raise ValueError(f"conditioning_fps must be positive, got {self.conditioning_fps}")
        if self.num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {self.num_inference_steps}")
        if self.guidance_scale <= 0:
            raise ValueError(f"guidance_scale must be positive, got {self.guidance_scale}")
        if self.resolution_tier not in ACTION_RESOLUTION_TIERS:
            raise ValueError(
                f"resolution_tier must be one of {list(ACTION_RESOLUTION_TIERS)}, "
                f"got {self.resolution_tier}"
            )
        if self.view_point not in VIEW_POINTS:
            raise ValueError(
                f"view_point must be one of {list(VIEW_POINTS)}, got {self.view_point!r}"
            )
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"dtype must be one of {list(SUPPORTED_DTYPES)}, got {self.dtype!r}")
        if not self.domain_name:
            raise ValueError("domain_name must be a non-empty embodiment name")
        if self.raw_action_dim is not None and not (
            1 <= self.raw_action_dim <= COSMOS3_EDGE_MAX_ACTION_DIM
        ):
            raise ValueError(
                f"raw_action_dim must be in [1, {COSMOS3_EDGE_MAX_ACTION_DIM}] (the model's "
                f"global action_dim), got {self.raw_action_dim}"
            )

    @property
    def requires_external_weights(self) -> bool:
        """True: the transformer, VAE and vision encoder stay OUT of the module tree."""
        return True


class Cosmos3EdgeAdapter:
    """``BackboneAdapter`` over Cosmos3-Edge. Construction is free; weights are not.

    Construction never touches the filesystem, network or GPU, and never imports torch or
    diffusers — :meth:`load` does. Until then every weight-requiring method raises
    ``RuntimeError(_WEIGHTS_MISSING_MSG)``.

    Three methods deviate from "just call the model", each for a reason recorded above:
    :meth:`condition_state` refuses (no proprioception input exists in the graph),
    :meth:`features` refuses after loading (no verified readout depth), and
    :meth:`require_raw_action_dim` refuses (10 vs 8 disagreement). Each raises a message
    naming what would settle it.
    """

    def __init__(self, config: Cosmos3EdgeConfig | None = None, **overrides: Any) -> None:
        if config is not None and overrides:
            raise TypeError(f"pass either config= or field kwargs, not both: {sorted(overrides)}")
        self.config = config if config is not None else Cosmos3EdgeConfig(**overrides)
        self.checkpoint_path = (
            Path(self.config.checkpoint_path) if self.config.checkpoint_path else None
        )
        self._loaded = False
        self._pipeline: Any = None

    # ---- BackboneAdapter protocol (metadata is available without weights) -------------------

    @property
    def name(self) -> str:
        return COSMOS3_EDGE_NAME

    @property
    def feature_dim(self) -> int:
        """Transformer hidden size (2048) — the width any readout would have. [OK]"""
        return self.config.feature_dim

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def describe(self) -> dict[str, Any]:
        """Configured geometry — goes into run logs for AC-04 traceability."""
        return {
            "name": self.name,
            "source": str(self.checkpoint_path or self.config.model_id),
            "loaded": self._loaded,
            "feature_dim": self.feature_dim,
            "hidden_dim": COSMOS3_EDGE_HIDDEN_DIM,
            "num_layers": COSMOS3_EDGE_NUM_LAYERS,
            "max_action_dim": COSMOS3_EDGE_MAX_ACTION_DIM,
            "num_embodiment_domains": COSMOS3_EDGE_NUM_EMBODIMENT_DOMAINS,
            "domain_name": self.config.domain_name,
            "raw_action_dim": self.config.raw_action_dim,  # None = unsettled [?]
            "action_chunk_size": self.config.action_chunk_size,
            "conditioning_fps": self.config.conditioning_fps,
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "resolution_tier": self.config.resolution_tier,
            "view_point": self.config.view_point,
            "dtype": self.config.dtype,
            "device": self.config.device,
        }

    # ---- loading -----------------------------------------------------------------------------

    def _resolve_source(self) -> str:
        if self.checkpoint_path is not None:
            if not self.checkpoint_path.exists():
                raise RuntimeError(
                    f"{_WEIGHTS_MISSING_MSG} (checkpoint_path={self.checkpoint_path})"
                )
            return str(self.checkpoint_path)
        if self.config.model_id and self.config.allow_download:
            return self.config.model_id
        raise RuntimeError(
            f"{_WEIGHTS_MISSING_MSG} (checkpoint_path=None, model_id="
            f"{self.config.model_id!r}, allow_download={self.config.allow_download})"
        )

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(_WEIGHTS_MISSING_MSG)

    def load(self) -> None:
        """Build the ``Cosmos3OmniPipeline`` from the configured source.

        Raises ``RuntimeError(_WEIGHTS_MISSING_MSG)`` when the snapshot is absent or diffusers
        is not installed. Never downloads unless ``allow_download`` is set.
        """
        if self._loaded:
            return
        source = self._resolve_source()
        try:  # lazy: diffusers is not a hard dependency of wam
            import torch
            from diffusers import Cosmos3OmniPipeline
        except ImportError as err:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(f"{_WEIGHTS_MISSING_MSG} ({err})") from err

        self._pipeline = Cosmos3OmniPipeline.from_pretrained(
            source,
            torch_dtype=getattr(torch, self.config.dtype),
            local_files_only=not self.config.allow_download,
        )
        self._loaded = True

    # ---- action width ------------------------------------------------------------------------

    def require_raw_action_dim(self) -> int:
        """The embodiment's unpadded action width, or raise if it is still unsettled [?].

        There is no default: diffusers 0.39.0 and the Policy-DROID model card disagree (10 vs
        8) for ``droid_lerobot``, and a wrong width here is the kind of bug that trains and
        deploys cleanly while moving the wrong channels.
        """
        if self.config.raw_action_dim is None:
            raise NotImplementedError(
                _RAW_ACTION_DIM_UNSETTLED_MSG.format(domain=self.config.domain_name)
            )
        return self.config.raw_action_dim

    # ---- conditioning ------------------------------------------------------------------------

    def condition_video(self, video: Any) -> dict[str, Any]:
        """Past frames -> the single conditioning frame the action path takes.

        Cosmos 3 policy mode conditions on ONE image (``CosmosActionCondition.image``; for
        ``policy``/``forward_dynamics`` "only its first frame is used" of a supplied video —
        ``pipeline_cosmos3_omni.py:283-286``). WAM hands the backbone a clip whose LAST entry
        is the current observation (``Observation.image_history``), so the current frame is
        what is selected here.

        Accepts ``[H, W, 3]``, ``[F, H, W, 3]`` or ``[1, F, H, W, 3]``, uint8 0..255 or float
        in [0, 1]. Weights-free on purpose: this is pure framing, and the pipeline does its
        own resize/pad into the ``resolution_tier`` canvas.
        """
        frames = np.asarray(video)
        if frames.ndim == 5:
            if frames.shape[0] != 1:
                raise ValueError(
                    "Cosmos3-Edge action conditioning takes a single clip; got batch "
                    f"{frames.shape[0]}. Call once per observation."
                )
            frames = frames[0]
        if frames.ndim == 4:
            frames = frames[-1]  # current observation is the LAST frame of the history
        if frames.ndim != 3 or frames.shape[-1] != 3:
            raise ValueError(
                "video must be [H, W, 3], [F, H, W, 3] or [1, F, H, W, 3] with 3 channels, "
                f"got shape {tuple(np.asarray(video).shape)}"
            )
        if frames.dtype != np.uint8:
            values = frames.astype(np.float32)
            if values.min() < 0.0 or values.max() > 1.0:
                raise ValueError(
                    "float video must be in [0, 1]; pass uint8 for 0..255 data, got range "
                    f"[{values.min()}, {values.max()}]"
                )
            frames = np.clip(values * 255.0, 0, 255).astype(np.uint8)
        return {
            "image": frames,
            "resolution_tier": self.config.resolution_tier,
            "chunk_size": self.config.action_chunk_size,
            "domain_name": self.config.domain_name,
        }

    def condition_text(self, text: str) -> dict[str, Any]:
        """Instruction -> the fields of Cosmos 3's structured action caption.

        TEXT IS NOT OPTIONAL HERE, and this adapter does not pretend otherwise (E-01). At the
        pipeline level ``prompt`` is a required positional with no default
        (``pipeline_cosmos3_omni.py:1250-1252``); at the transformer level ``input_ids`` /
        ``text_indexes`` / ``und_len`` are required positionals with no None branch
        (``transformer_cosmos3.py:554-559``); ``text_tokenizer`` is a required pipeline
        component (``:366``); and every generation token — action tokens included —
        cross-attends to the text keys/values in every layer (``:82-94``, ``:683-692``).

        ``prompt=""`` is *accepted* (``check_inputs`` only type-checks, ``:963-966``) and lands
        verbatim as ``"description": ""`` in the caption, but it is not a null path: the chat
        template plus two special tokens keep ``und_len >= 3``, CFG then amplifies only
        viewpoint/duration/fps/resolution metadata, and the policy falls back to its
        unconditional prior. NVIDIA's own RoboLab numbers put merely *vague* instructions at
        15.4% success against 22.9% default and 28.8% specific [doc], and an empty instruction
        carries strictly less information than vague. So empty must be opted into via
        ``Cosmos3EdgeConfig(allow_empty_instruction=True)``, not reached by accident.

        For a single-task MVP the cheap, in-distribution move is a CONSTANT correct sentence,
        not an empty one — pass the same instruction every tick. This method returns caption
        *fields*, not token ids: tokenizing needs the staged ``text_tokenizer``.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        if not text.strip() and not self.config.allow_empty_instruction:
            raise ValueError(
                "empty instruction rejected: Cosmos3-Edge always consumes a text stream "
                "(prompt is a required positional; und_len >= 3 even for ''), so an empty "
                "prompt is not 'no language' — it drops the policy onto its unconditional "
                "prior, below NVIDIA's 15.4% vague-instruction RoboLab number. Pass a "
                "constant task sentence, or set allow_empty_instruction=True to measure the "
                "degraded case deliberately (E-01)."
            )
        return {
            "description": text,
            "view_point": self.config.view_point,
            "fps": self.config.conditioning_fps,
            "chunk_size": self.config.action_chunk_size,
            # NVIDIA's tuned choice for every action mode is the null string [doc, report 6.3.1].
            "negative_prompt": "",
            "guidance_scale": self.config.guidance_scale,
        }

    def condition_state(self, state_embedding: Any) -> Any:
        """Always raises: Cosmos 3's released graph has no proprioception input.

        See ``_NO_STATE_INPUT_MSG``. Refusing is the point — a fabricated state slot would
        train and run cleanly while conditioning on nothing.
        """
        raise NotImplementedError(_NO_STATE_INPUT_MSG)

    # ---- feature readout ----------------------------------------------------------------------

    def features(self, video_ctx: Any, text_ctx: Any, state_ctx: Any) -> Any:
        """Intermediate activations for the ActionDecoder — not yet available.

        Without staged weights this raises ``RuntimeError(_WEIGHTS_MISSING_MSG)``. With them it
        raises ``NotImplementedError``: the readout width is known (2048) but the readout
        DEPTH is not, and Cosmos 3 publishes no recipe for it.
        """
        self._require_loaded()
        raise NotImplementedError(_FEATURE_READOUT_UNVERIFIED_MSG)
