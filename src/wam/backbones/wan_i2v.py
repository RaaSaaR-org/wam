"""Wan I2V backbone adapter (T-15, OD-04; reference: DreamZero [R1]/[R2]).

diffusers-backed integration for the open fallback backbone. Works with every Wan variant
that ships a ``WanTransformer3DModel`` + ``AutoencoderKLWan`` + umT5 text encoder:

- ``Wan-AI/Wan2.2-TI2V-5B-Diffusers``  — 30 blocks, inner dim 3072, 48 latent ch, VAE 16x16x4
- ``Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`` — 40 blocks, inner dim 5120, 16 latent ch, VAE 8x8x4

ALL model dimensions are read from the loaded configs; the ``WAN_*`` constants below are only
14B defaults used for reporting before ``load()``. The module imports torch-free: torch,
diffusers and transformers are imported inside ``load()``, nothing is downloaded implicitly
(``local_files_only=True`` unless ``allow_download`` is set).

Pipeline (DreamZero recipe):

- ``condition_video()``: past frames -> Wan-VAE latents ``[B, z, F', H/s, W/s]``
  (F' = 1 + (F-1)//temporal_stride), normalized with ``latents_mean/std``; for I2V checkpoints
  with a CLIP tower (``transformer.config.image_dim``) also the image embedding of the LAST
  observed frame (``hidden_states[-2]``, as in the diffusers Wan I2V pipeline).
- ``condition_text()``: frozen umT5 -> ``[B, max_text_tokens, text_dim]``, padding zeroed.
- ``condition_state()``: StateEncoder output ``[B, E]`` -> ONE extra token in TEXT-context
  space ``[B, 1, text_dim]``, appended to the text context; the DiT's own text embedder maps
  it into the residual stream. The projection is the only trainable part of this adapter.
- ``features()``: forward hooks on the residual-stream OUTPUT of blocks ``feature_blocks``
  (default: mid/late depth = ``num_layers//2`` and ``3*num_layers//4``), averaged
  -> ``[B, S, inner_dim]`` with S = F' * (H/s/p_h) * (W/s/p_w).

Training (T-16, PRD §10.3) reuses that single DiT call through the ``FlowBackbone`` pathway:
``encode_video()``/``decode_video()`` for the VAE round-trip, ``forward_flow()`` for one
velocity + feature pass, ``add_lora()`` for the trainable part. ``forward_flow()`` is the ONLY
place Wan's conventions (timesteps counting DOWN from 1000, velocity pointing noise-ward) are
translated into WAM's (``x_t = (1-t)*x0 + t*x1``, ``t=1`` clean); nothing above the adapter —
neither the losses nor ``JointWorldActionModel.co_denoise`` — may learn about them (FR-09).

Swapping this adapter in must not change the data schema or the robot API (FR-09/AC-05).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

WAN_NAME = "wan2.1-i2v"
WAN_DIT_HIDDEN_DIM = 5120
WAN_DIT_NUM_BLOCKS = 40
WAN_DIT_NUM_HEADS = 40
WAN_VAE_LATENT_CHANNELS = 16
WAN_VAE_SPATIAL_STRIDE = 8
WAN_VAE_TEMPORAL_STRIDE = 4
WAN_PATCH_SIZE: tuple[int, int, int] = (1, 2, 2)
WAN_TEXT_DIM = 4096
DEFAULT_FEATURE_BLOCKS: tuple[int, ...] = (20, 30)
DEFAULT_MAX_TEXT_TOKENS = 512

# Wan's noise schedule is discretized into 1000 training steps counted DOWNWARDS (1000 = pure
# noise, 0 = clean) — the mirror image of WAM's t in [0, 1] with t=1 clean. Used by exactly one
# expression, in forward_flow().
WAN_NUM_TRAIN_TIMESTEPS = 1000
# Every Linear inside a Wan transformer block: self-attention (attn1), cross-attention (attn2)
# and the FFN. Verified by introspection on WanTransformer3DModel; matched by SUFFIX, so the
# same six names cover all 30/40 blocks. attn2.add_k_proj/add_v_proj exist only on checkpoints
# with a CLIP image tower and are appended by add_lora() when config.added_kv_proj_dim is set.
DEFAULT_LORA_TARGETS: tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0", "net.0.proj", "net.2")

_LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"  # diffusers' save_lora_adapter layout

_WEIGHTS_MISSING_MSG = (
    "Wan2.1 weights not available — pass checkpoint_path=<local snapshot dir> (or "
    "model_id=<hub repo> with allow_download=True) and install the runtime "
    "(diffusers>=0.35 + transformers); nothing is downloaded implicitly."
)


class WanBackboneConfig(BaseModel):
    """Declarative config for :class:`WanI2VAdapter` — the ``kind="wan_i2v"`` arm of the
    backbone union next to :class:`~wam.backbones.tiny.TinyBackboneConfig`.

    TORCH-FREE by construction: every field is a JSON primitive, a ``Literal`` or a tuple, so
    the whole training config stays YAML-round-trippable and hashable by
    ``wam.interfaces.versioning.config_hash`` (AC-04). Never put a ``torch.dtype`` in here —
    ``dtype``/``lora_param_dtype`` are the string names the adapter resolves at load time.

    Frozen + ``extra="forbid"``: with the backbone config a discriminated union, a typo'd or
    tiny-shaped field must fail loudly instead of silently training an all-defaults model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["wan_i2v"] = "wan_i2v"
    # Weight source: exactly one of the two, checkpoint_path (a local snapshot) taking
    # precedence. allow_download stays False so a training run never blocks on a 34 GB pull.
    model_id: str | None = None
    checkpoint_path: str | None = None
    allow_download: bool = False
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    device: str = "cuda"
    device_map: str | None = None
    feature_dim: int = Field(default=3072, gt=0, description="DiT inner dim (TI2V-5B: 3072).")
    num_frames: int = Field(default=9, gt=0, description="Pixel frames per clip (VAE input).")
    image_hw: tuple[int, int] = Field(
        default=(128, 160),
        description="Frames are resized to this before the VAE; must be DiT-legal (see "
        "encode_video), which raw dataset sizes such as GR00T's 120x160 are not.",
    )
    state_embedding_dim: int = Field(default=32, gt=0)
    max_text_tokens: int = Field(default=512, gt=0)
    feature_blocks: tuple[int, ...] = Field(
        default=(2, 10),
        description="Readout depths; measured on GR00T-G1 action labels, not a heuristic "
        "(configs/model/wan22_ti2v_5b.yaml).",
    )
    lora_rank: int = Field(default=32, gt=0)
    lora_alpha: int = Field(default=64, gt=0)
    lora_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    lora_targets: tuple[str, ...] = DEFAULT_LORA_TARGETS
    lora_blocks: tuple[int, ...] | None = Field(
        default=None, description="Restrict LoRA to these block depths; None = every block."
    )
    lora_param_dtype: Literal["float32", "bfloat16"] = "float32"
    gradient_checkpointing: bool = True

    @model_validator(mode="after")
    def _validate(self) -> WanBackboneConfig:
        height, width = self.image_hw
        if height <= 0 or width <= 0:
            raise ValueError(f"image_hw must be positive, got {self.image_hw}")
        if not self.feature_blocks or any(b < 0 for b in self.feature_blocks):
            raise ValueError(
                f"feature_blocks must be non-empty and >= 0, got {self.feature_blocks}"
            )
        return self

    @property
    def requires_external_weights(self) -> bool:
        """True: the DiT, VAE and text tower stay OUT of the module tree.

        So a checkpoint of a Wan-backed model holds only ``backbone.lora.*`` and
        ``backbone.state_proj.*`` — never the multi-GB base. Restoring one needs the base
        supplied as an already-loaded backbone and ``strict=False``; anything that loads a
        checkpoint has to branch on this rather than assume the file is self-contained.
        """
        return True


def default_feature_blocks(num_layers: int) -> tuple[int, ...]:
    """Mid/late-depth readout blocks for a DiT of ``num_layers`` (40 -> (20, 30))."""
    if num_layers < 2:
        raise ValueError(f"num_layers must be >= 2, got {num_layers}")
    return (num_layers // 2, (3 * num_layers) // 4)


class WanI2VAdapter:
    """``BackboneAdapter`` over a Wan DiT: conditioning + intermediate feature readout.

    Construction never touches the filesystem, network or GPU — ``load()`` does. Until then
    every weight-requiring method raises ``RuntimeError(_WEIGHTS_MISSING_MSG)``.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        feature_blocks: tuple[int, ...] | None = None,
        device: str = "cpu",
        *,
        model_id: str | None = None,
        dtype: str = "bfloat16",
        timestep: int = 0,
        max_text_tokens: int = DEFAULT_MAX_TEXT_TOKENS,
        allow_download: bool = False,
        device_map: str | None = None,
    ) -> None:
        blocks: tuple[int, ...] | None = None
        if feature_blocks is not None:
            blocks = tuple(int(b) for b in feature_blocks)
            if not blocks or any(b < 0 or b >= WAN_DIT_NUM_BLOCKS for b in blocks):
                raise ValueError(
                    f"feature_blocks must be non-empty indices in [0, {WAN_DIT_NUM_BLOCKS}) "
                    f"(deepest known Wan DiT), got {feature_blocks!r}"
                )
        if max_text_tokens < 1:
            raise ValueError(f"max_text_tokens must be >= 1, got {max_text_tokens}")
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.model_id = model_id
        self.feature_blocks = blocks if blocks is not None else DEFAULT_FEATURE_BLOCKS
        self._requested_blocks = blocks
        self.device = device
        self.dtype = dtype
        self.timestep = int(timestep)
        self.max_text_tokens = int(max_text_tokens)
        self.allow_download = bool(allow_download)
        # accelerate device_map: stream shards straight to the target device instead of
        # materializing the whole model in host RAM first. Required wherever host RAM is
        # smaller than the checkpoint (a stock 16 GB Space vs. a 34 GB Wan repo); worth it
        # anyway — it loads the 5B in 7.4 s where a CPU round-trip needs tens of seconds.
        self.device_map = device_map
        self._loaded = False
        self._transformer: Any = None
        self._vae: Any = None
        self._text_encoder: Any = None
        self._tokenizer: Any = None
        self._image_encoder: Any = None
        self._image_processor: Any = None
        self._state_proj: Any = None
        self._feature_dim = WAN_DIT_HIDDEN_DIM
        self._num_layers = WAN_DIT_NUM_BLOCKS
        self._in_channels = 2 * WAN_VAE_LATENT_CHANNELS + WAN_VAE_TEMPORAL_STRIDE
        self._latent_channels = WAN_VAE_LATENT_CHANNELS
        self._text_dim = WAN_TEXT_DIM
        self._patch_size = WAN_PATCH_SIZE
        self._image_dim: int | None = None
        self._vae_spatial = WAN_VAE_SPATIAL_STRIDE
        self._vae_temporal = WAN_VAE_TEMPORAL_STRIDE
        # The instruction is constant for a whole rollout (and for most of an epoch), while the
        # umT5 tower is the priciest frozen part to re-run — and it is deterministic.
        self._text_cache: dict[Any, Any] = {}
        self._lora_adapter: str | None = None
        # Adapters arrive enabled (peft's own default); set_lora_enabled() tracks the toggle.
        self._lora_enabled: bool = True

    # ---- BackboneAdapter protocol (metadata is available without weights) ---------------

    @property
    def name(self) -> str:
        return WAN_NAME

    @property
    def feature_dim(self) -> int:
        """Inner dim of the loaded DiT; the 14B default (5120) before ``load()``."""
        return self._feature_dim

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def state_projection(self) -> Any:
        """The lazily built state->text-context projection (trainable; None before use)."""
        return self._state_proj

    def describe(self) -> dict[str, Any]:
        """Derived model geometry — goes into run logs for AC-04 traceability."""
        return {
            "name": self.name,
            "source": str(self.checkpoint_path or self.model_id or ""),
            "loaded": self._loaded,
            "num_layers": self._num_layers,
            "feature_blocks": list(self.feature_blocks),
            "feature_dim": self._feature_dim,
            "in_channels": self._in_channels,
            "latent_channels": self._latent_channels,
            "text_dim": self._text_dim,
            "patch_size": list(self._patch_size),
            "image_dim": self._image_dim,
            "vae_spatial_stride": self._vae_spatial,
            "vae_temporal_stride": self._vae_temporal,
            "dtype": self.dtype,
            "device": self.device,
            "device_map": self.device_map,
            "timestep": self.timestep,
        }

    # ---- loading -------------------------------------------------------------------------

    def load(self) -> None:
        """Load VAE + umT5 + DiT (+ CLIP tower for I2V checkpoints) from the configured source.

        No sampling pipeline and no scheduler are built: this adapter only ever runs a single
        DiT forward for feature readout.
        """
        if self._loaded:
            return
        source = self._resolve_source()
        try:  # lazy: none of these is a hard dependency of wam
            import torch
            from diffusers import AutoencoderKLWan, WanTransformer3DModel
            from transformers import AutoTokenizer, UMT5EncoderModel
        except ImportError as err:  # pragma: no cover - exercised on machines without the extra
            raise RuntimeError(f"{_WEIGHTS_MISSING_MSG} ({err})") from err

        torch_dtype = self._model_dtype()
        common: dict[str, Any] = {"local_files_only": not self.allow_download}
        weights = dict(common)
        if self.device_map:  # shards go straight to the device; host RAM stays flat
            weights["device_map"] = self.device_map
        # VAE stays fp32 (diffusers guidance: better encode/decode quality).
        vae = AutoencoderKLWan.from_pretrained(
            source, subfolder="vae", torch_dtype=torch.float32, **weights
        )
        transformer = WanTransformer3DModel.from_pretrained(
            source, subfolder="transformer", torch_dtype=torch_dtype, **weights
        )
        text_encoder = UMT5EncoderModel.from_pretrained(
            source, subfolder="text_encoder", torch_dtype=torch_dtype, **weights
        )
        tokenizer = AutoTokenizer.from_pretrained(source, subfolder="tokenizer", **common)
        image_encoder = image_processor = None
        if getattr(transformer.config, "image_dim", None):
            from transformers import CLIPImageProcessor, CLIPVisionModel

            image_encoder = CLIPVisionModel.from_pretrained(
                source, subfolder="image_encoder", torch_dtype=torch.float32, **weights
            )
            image_processor = CLIPImageProcessor.from_pretrained(
                source, subfolder="image_processor", **common
            )
        self.attach(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            image_processor=image_processor,
            # device_map already placed every shard — a second .to() would defeat the point.
            move_to_device=not self.device_map,
        )

    def attach(
        self,
        *,
        transformer: Any,
        vae: Any,
        text_encoder: Any = None,
        tokenizer: Any = None,
        image_encoder: Any = None,
        image_processor: Any = None,
        move_to_device: bool = True,
    ) -> None:
        """Attach already-constructed modules (used by ``load()`` and by tests with stubs).

        Everything is frozen and put in eval mode: this adapter is a feature extractor;
        fine-tuning (LoRA on the DiT) unfreezes explicitly in the training code (§10.3).
        """
        cfg = transformer.config
        blocks = getattr(transformer, "blocks", None)
        if blocks is None:
            raise RuntimeError(
                f"{type(transformer).__name__} has no '.blocks' — cannot hook the residual "
                "stream; the diffusers Wan DiT exposes its transformer blocks as '.blocks'"
            )
        num_layers = int(getattr(cfg, "num_layers", len(blocks)))
        if num_layers != len(blocks):
            raise RuntimeError(f"config.num_layers={num_layers} but {len(blocks)} blocks found")
        resolved = self._requested_blocks or default_feature_blocks(num_layers)
        if any(b >= num_layers for b in resolved):
            raise ValueError(
                f"feature_blocks {resolved} out of range for a {num_layers}-block DiT "
                f"({self.model_id or self.checkpoint_path}); pass feature_blocks explicitly"
            )
        self._transformer = transformer
        self._vae = vae
        self._text_encoder = text_encoder
        self._tokenizer = tokenizer
        self._image_encoder = image_encoder
        self._image_processor = image_processor
        # Both caches describe the modules we are replacing here, not the adapter.
        self._text_cache.clear()
        self._lora_adapter = None
        self._lora_enabled = True

        self.feature_blocks = tuple(resolved)
        self._num_layers = num_layers
        self._feature_dim = int(cfg.num_attention_heads) * int(cfg.attention_head_dim)
        self._in_channels = int(cfg.in_channels)
        self._text_dim = int(cfg.text_dim)
        self._patch_size = tuple(int(p) for p in cfg.patch_size)  # type: ignore[assignment]
        self._image_dim = getattr(cfg, "image_dim", None)
        vae_cfg = vae.config
        latents_mean = list(getattr(vae_cfg, "latents_mean", []))
        self._latent_channels = int(getattr(vae_cfg, "z_dim", 0) or len(latents_mean))
        downsample = getattr(vae_cfg, "temperal_downsample", None)  # diffusers spelling
        if downsample:
            self._vae_temporal = 2 ** int(sum(bool(d) for d in downsample))
            self._vae_spatial = 2 ** len(downsample)
        # Wan 2.2 VAEs pixel-shuffle by `patch_size` on top of the downsample stack, so the
        # stride above (8) understates the real 16x compression. Those configs state the
        # factors outright — trust them; Wan 2.1 configs have neither field and derive fine.
        for attr, field in (
            ("_vae_spatial", "scale_factor_spatial"),
            ("_vae_temporal", "scale_factor_temporal"),
        ):
            value = getattr(vae_cfg, field, None)
            if value:
                setattr(self, attr, int(value))

        for module in (transformer, vae, text_encoder, image_encoder):
            if module is None:
                continue
            module.eval()
            module.requires_grad_(False)
            if move_to_device:
                module.to(self.device)
        self._loaded = True

    def offload(self, *components: str) -> None:
        """Move named components ('text_encoder', 'image_encoder', 'vae', 'transformer') to CPU.

        Peak-VRAM relief on 24 GB cards: the umT5 tower is only needed while encoding the
        instruction, which for a fixed task is done once per rollout.
        """
        self._require_loaded()
        known = {
            "text_encoder": self._text_encoder,
            "image_encoder": self._image_encoder,
            "vae": self._vae,
            "transformer": self._transformer,
        }
        for name in components:
            if name not in known:
                raise ValueError(f"unknown component {name!r}; known: {sorted(known)}")
            module = known[name]
            if module is not None:
                module.to("cpu")

    def _resolve_source(self) -> str:
        if self.checkpoint_path is not None:
            if not self.checkpoint_path.exists():
                raise RuntimeError(
                    f"{_WEIGHTS_MISSING_MSG} (checkpoint_path={self.checkpoint_path})"
                )
            return str(self.checkpoint_path)
        if self.model_id:
            return self.model_id
        raise RuntimeError(f"{_WEIGHTS_MISSING_MSG} (checkpoint_path=None, model_id=None)")

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(_WEIGHTS_MISSING_MSG)

    # ---- conditioning ----------------------------------------------------------------------

    def condition_video(self, video: Any) -> dict[str, Any]:
        """Past frames -> ``{"latents": [B, z, F', H/s, W/s], "image_embeds": ... | None}``.

        Accepts uint8 ``[B, F, H, W, 3]`` / ``[F, H, W, 3]`` (0..255) or a float tensor already
        in [-1, 1] with the same layout. H and W must be multiples of
        ``vae_spatial_stride * patch_size``.
        """
        self._require_loaded()
        import torch

        pixels = self._to_pixel_tensor(video)  # [B, 3, F, H, W] fp32 in [-1, 1]
        self._check_pixel_grid(pixels)
        with torch.no_grad():
            posterior = self._vae.encode(pixels.to(self._device_of(self._vae))).latent_dist
            latents = posterior.mode()  # deterministic: no sampling in a feature extractor
            latents = self._normalize_latents(latents)
            image_embeds = self._encode_last_frame(pixels)
        dtype = self._model_dtype()
        return {
            "latents": latents.to(device=self.device, dtype=dtype),
            "image_embeds": None
            if image_embeds is None
            else image_embeds.to(device=self.device, dtype=dtype),
        }

    def condition_text(self, text: str | list[str]) -> Any:
        """Instruction(s) -> frozen umT5 context ``[B, max_text_tokens, text_dim]``.

        Padding positions are zeroed (same convention as the diffusers Wan pipelines).
        Memoized per prompt: the tower is frozen and deterministic, so re-encoding the same
        instruction every training step (or every closed-loop tick) is pure waste. ``attach()``
        drops the cache, since it swaps the very tower that produced it.
        """
        self._require_loaded()
        import torch

        if self._tokenizer is None or self._text_encoder is None:
            raise RuntimeError("no text encoder attached — load() builds tokenizer + umT5")
        key = text if isinstance(text, str) else tuple(text)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached
        prompts = [text] if isinstance(text, str) else list(text)
        batch = self._tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_text_tokens,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        enc_device = self._device_of(self._text_encoder)
        ids = batch["input_ids"].to(enc_device)
        mask = batch["attention_mask"].to(enc_device)
        with torch.no_grad():
            embeds = self._text_encoder(ids, attention_mask=mask).last_hidden_state
        masked = embeds * mask.unsqueeze(-1).to(embeds.dtype)
        result = masked.to(device=self.device, dtype=self._model_dtype())
        self._text_cache[key] = result
        return result

    def condition_state(self, state_embedding: Any) -> Any:
        """StateEncoder output ``[B, E]`` -> one extra context token ``[B, 1, text_dim]``.

        The projection is built on first use if the caller did not build it eagerly (fp32,
        ``requires_grad=True``) — it is the adapter's only non-LoRA trainable block; register it
        with the optimizer via ``state_projection`` or :meth:`trainable_parameters`.
        """
        self._require_loaded()
        import torch

        x = self._as_tensor(state_embedding).to(torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError(f"state_embedding must be [B, E] or [E], got shape {tuple(x.shape)}")
        proj = self.build_state_projection(int(x.shape[-1]))
        token = proj(x.to(proj.weight.device)).unsqueeze(1)
        return token.to(self._model_dtype())

    def build_state_projection(self, embed_dim: int) -> Any:
        """Create (once) the trainable state -> text-context projection; returns it.

        EAGER on purpose: an optimizer is built before the first batch, so a projection that
        only materializes inside the first ``condition_state()`` call would never be registered
        and would train nothing. Lives on the TRANSFORMER's device, not ``self.device``: with
        ``device_map`` the model is sharded and ``self.device`` is only the nominal target.
        Stays fp32 while the DiT may be bf16 — it is one small matrix, and its gradients are
        the adapter's whole proprioception path.
        """
        self._require_loaded()
        import torch
        from torch import nn

        dim = int(embed_dim)
        if dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")
        if self._state_proj is not None:
            if self._state_proj.in_features != dim:
                raise ValueError(
                    f"state embedding dim changed: projection expects "
                    f"{self._state_proj.in_features}, got {dim}"
                )
            return self._state_proj
        device = self._device_of(self._transformer)
        self._state_proj = nn.Linear(dim, self._text_dim).to(device=device, dtype=torch.float32)
        return self._state_proj

    # ---- feature readout ---------------------------------------------------------------

    def features(self, video_ctx: Any, text_ctx: Any, state_ctx: Any) -> Any:
        """One DiT forward with hooks on ``feature_blocks`` -> ``[B, S, feature_dim]``.

        ``video_ctx`` is the dict from :meth:`condition_video` (or a bare latent tensor);
        ``state_ctx`` may be None. The denoising step is the adapter's ``timestep`` attribute
        (the protocol signature is fixed — FR-09). Gradients follow the caller's autograd
        context; wrap in ``torch.no_grad()`` for pure inference.
        """
        import torch

        per_block = self.features_by_block(video_ctx, text_ctx, state_ctx)
        return torch.stack([per_block[i] for i in self.feature_blocks], dim=0).mean(dim=0)

    def features_by_block(
        self, video_ctx: Any, text_ctx: Any, state_ctx: Any, blocks: tuple[int, ...] | None = None
    ) -> dict[int, Any]:
        """One DiT forward hooked on ``blocks`` (default ``feature_blocks``) -> {index: [B, S, D]}.

        The per-block variant exists for readout ablations — which depth carries the most
        conditioning signal; :meth:`features` is the protocol method and averages its blocks.
        """
        return self._forward_dit(video_ctx, text_ctx, state_ctx, blocks=blocks)[1]

    def _forward_dit(
        self,
        latents: Any,
        text_ctx: Any,
        state_ctx: Any,
        *,
        timestep: Any = None,
        blocks: tuple[int, ...] | None = None,
        condition_latents: Any = None,
    ) -> tuple[Any, dict[int, Any]]:
        """The single DiT call of this adapter -> ``(output, {block_index: activation})``.

        Readout (:meth:`features_by_block`) and the flow pathway (:meth:`forward_flow`) differ
        only in what they do with the two return values, so they cannot drift apart — and the
        denoised ``output``, which the readout path throws away, costs nothing extra here.

        ``latents`` may be the :meth:`condition_video` dict (CLIP image embeds included) or a
        bare latent tensor. ``timestep=None`` reproduces the historical constant-``self.timestep``
        long tensor byte-for-byte, so the recorded probe/ablation numbers (docs/hf_jobs.md,
        runs/wan_probe/) stay reproducible; the flow pathway passes its own float schedule.
        """
        self._require_loaded()
        import torch

        selected = tuple(int(b) for b in (self.feature_blocks if blocks is None else blocks))
        if not selected or any(b < 0 or b >= self._num_layers for b in selected):
            raise ValueError(
                f"blocks must be non-empty indices in [0, {self._num_layers}), got {selected!r}"
            )
        video_ctx = latents
        latent_tensor = video_ctx["latents"] if isinstance(video_ctx, dict) else video_ctx
        image_embeds = video_ctx.get("image_embeds") if isinstance(video_ctx, dict) else None
        hidden_states = self._build_hidden_states(latent_tensor, condition_latents)
        if text_ctx is None:
            raise ValueError("text_ctx is required (condition_text output)")
        batch = int(hidden_states.shape[0])
        encoder_hidden_states = text_ctx
        # One instruction, many clips: condition_text() encodes a single prompt once per rollout
        # while the latent batch is the training batch — broadcast instead of re-encoding.
        if int(text_ctx.shape[0]) == 1 and batch > 1:
            encoder_hidden_states = text_ctx.expand(batch, -1, -1)
        if state_ctx is not None:
            encoder_hidden_states = torch.cat([encoder_hidden_states, state_ctx], dim=1)
        if self._image_dim and image_embeds is None:
            raise ValueError(
                "this checkpoint expects CLIP image conditioning "
                f"(image_dim={self._image_dim}) — pass the condition_video() dict"
            )
        ts = timestep
        if ts is None:
            ts = torch.full((batch,), int(self.timestep), dtype=torch.long, device=self.device)

        captured: dict[int, Any] = {}
        handles = []
        transformer_blocks = self._transformer.blocks
        for index in set(selected):
            handles.append(
                transformer_blocks[index].register_forward_hook(self._make_hook(captured, index))
            )
        try:
            output = self._transformer(
                hidden_states=hidden_states,
                timestep=ts,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=image_embeds,
                return_dict=False,
            )[0]
        finally:
            for handle in handles:
                handle.remove()

        missing = [i for i in selected if i not in captured]
        if missing:
            raise RuntimeError(f"no activation captured for blocks {missing}")
        for index, activation in captured.items():
            if activation.shape[-1] != self._feature_dim:
                raise RuntimeError(
                    f"block {index} activation has last dim {activation.shape[-1]}, expected "
                    f"{self._feature_dim} — block output is not the residual stream"
                )
        return output, captured

    @staticmethod
    def _make_hook(sink: dict[int, Any], index: int) -> Any:
        def hook(_module: Any, _args: Any, output: Any) -> None:
            sink[index] = output[0] if isinstance(output, tuple) else output

        return hook

    def token_grid(self, num_frames: int, height: int, width: int) -> tuple[int, int, int]:
        """The ``(F', H', W')`` token layout for ``num_frames`` pixel frames at ``H`` x ``W``.

        Patchify is a strided ``Conv3d`` followed by ``flatten(2).transpose(1, 2)``
        (``WanTransformer3DModel.patch_embedding``), so the sequence axis of a ``[B, S, D]``
        activation is **row-major over (F', H', W')** and reshapes to ``[B, F', H', W', D]``.
        That reshape is what a geometry-preserving readout needs (docs/improvements.md I-1);
        the mean-pool readout does not care, which is precisely the confound I-1 tests.

        Only the token *count* is self-checkable here (:meth:`expected_token_count` is its
        product, and the probe asserts it against the real activation). The axis *order* is a
        property of diffusers' patch embedder, not of this adapter.
        """
        latent_frames = 1 + (num_frames - 1) // self._vae_temporal
        p_t, p_h, p_w = self._patch_size
        return (
            latent_frames // p_t,
            height // self._vae_spatial // p_h,
            width // self._vae_spatial // p_w,
        )

    def expected_token_count(self, num_frames: int, height: int, width: int) -> int:
        """S for ``num_frames`` pixel frames at ``height`` x ``width`` (shape self-check)."""
        frames, rows, cols = self.token_grid(num_frames, height, width)
        return frames * rows * cols

    # ---- FlowBackbone: latent round-trip + rectified-flow pass (T-16) ---------------------

    def encode_video(self, video: Any, *, image_hw: tuple[int, int] | None = None) -> Any:
        """Raw frames -> clean flow latents ``[B, z, F', H/s, W/s]``, fp32.

        Same input contract as :meth:`condition_video` (uint8 ``[B, F, H, W, 3]`` /
        ``[F, H, W, 3]``, or float already in [-1, 1]) plus an optional resize: recorded
        datasets do not come in DiT-legal sizes — GR00T-G1 frames are 120x160 and 120 is not a
        multiple of ``vae_spatial * patch`` (16), so the caller passes e.g.
        ``image_hw=(128, 160)`` and gets a clip the VAE and the patch embedder both accept.

        fp32 and ``no_grad`` on purpose: this is the flow TARGET. Rounding it to bf16 would put
        the quantization noise of the target into every velocity loss, and a gradient path back
        into the frozen VAE would train nothing while doubling the activation memory.
        """
        self._require_loaded()
        import torch

        pixels = self._to_pixel_tensor(video)  # [B, 3, F, H, W] fp32 in [-1, 1]
        if image_hw is not None:
            pixels = self._resize_pixels(pixels, (int(image_hw[0]), int(image_hw[1])))
        self._check_pixel_grid(pixels)
        with torch.no_grad():
            posterior = self._vae.encode(pixels.to(self._device_of(self._vae))).latent_dist
            latents = self._normalize_latents(posterior.mode())  # deterministic: no sampling
        return latents.to(device=self.device, dtype=torch.float32)

    def decode_video(self, video_latents: Any) -> Any:
        """Flow latents -> pixel frames ``[B, F, H, W, 3]`` in [0, 1] (inverse of encode_video).

        The [0, 1] range is WAM's pixel convention (see the tiny backbone's identity VAE), not
        Wan's [-1, 1] — everything above the adapter compares predicted to recorded frames in
        [0, 1]. Inference-only: rollout previews and E2/E3 artefacts, never a loss.
        """
        self._require_loaded()
        import torch

        latents = self._as_tensor(video_latents)
        if latents.ndim != 5:
            raise ValueError(
                f"video_latents must be [B, z, F', h, w], got shape {tuple(latents.shape)}"
            )
        vae_device = self._device_of(self._vae)
        vae_dtype = next(self._vae.parameters()).dtype
        with torch.no_grad():
            scaled = self.denormalize_latents(latents.to(device=vae_device, dtype=torch.float32))
            decoded = self._vae.decode(scaled.to(vae_dtype))
        frames = decoded.sample if hasattr(decoded, "sample") else decoded[0]  # [B, 3, F, H, W]
        frames = ((frames.float() + 1.0) / 2.0).clamp(0.0, 1.0)
        return frames.permute(0, 2, 3, 4, 1).contiguous()

    def num_video_tokens(self, video_latents: Any) -> int:
        """Leading video tokens of :meth:`forward_flow`'s features, from the latent geometry.

        Wan's DiT sequence IS the patchified video — text and state ride in through
        cross-attention, not as sequence positions — so this is the FULL feature length. It is
        derived per batch rather than read off a config because the clip size is a property of
        the data, and cross-checked against :meth:`expected_token_count`, which computes the
        same number from the PIXEL geometry. Disagreement means our stride bookkeeping is off
        and every downstream token slice would be silently misaligned.
        """
        latents = self._as_tensor(video_latents)
        if latents.ndim != 5:
            raise ValueError(
                f"video_latents must be [B, z, F', h, w], got shape {tuple(latents.shape)}"
            )
        _, _, frames, height, width = (int(v) for v in latents.shape)
        p_t, p_h, p_w = self._patch_size
        tokens = (frames // p_t) * (height // p_h) * (width // p_w)
        expected = self.expected_token_count(
            (frames - 1) * self._vae_temporal + 1,
            height * self._vae_spatial,
            width * self._vae_spatial,
        )
        if tokens != expected:
            raise RuntimeError(
                f"token count disagrees with the pixel-side derivation ({tokens} vs {expected}) "
                f"for latents {tuple(latents.shape)} — check vae strides / patch size"
            )
        return tokens

    def forward_flow(
        self, video_latents: Any, t: Any, text_ctx: Any, state_ctx: Any
    ) -> tuple[Any, Any]:
        """One rectified-flow pass -> ``(velocity, features)``; WAM conventions in, WAM out.

        Exactly two translations happen here, and nowhere else in the codebase — that is the
        whole point of this method, so ``losses.py`` and ``JointWorldActionModel.co_denoise``
        never learn anything Wan-specific (FR-09):

        1. TIMESTEP. WAM's ``t`` is a flow position in [0, 1] with ``t=1`` CLEAN; Wan's
           scheduler counts denoising steps DOWNWARDS from ``WAN_NUM_TRAIN_TIMESTEPS``
           (1000 = pure noise). So ``ts = (1 - t) * 1000``, float32 — Wan2.2 takes float
           timesteps and even per-token ones, so there is nothing to round here.
        2. SIGN. The DiT predicts the velocity of ITS trajectory, which runs noise-ward
           (clean -> noise); WAM's target is ``v = x1 - x0``, the same line traversed the other
           way. Hence the minus.

        Get either wrong and training still converges — to the time-reverse of the intended
        video, with an action branch conditioned on features from the wrong noise level.

        ``video_latents`` is the NOISED latent ``[B, z, F', h, w]`` (fp32 from
        :meth:`encode_video`, mixed with noise by the caller); ``t`` is a scalar or ``[B]``.
        Returns velocity with exactly that shape, and ``[B, S, feature_dim]`` features averaged
        over ``feature_blocks``, both fp32 so the losses never see the model dtype.
        """
        self._require_loaded()
        import torch

        latents = self._as_tensor(video_latents)
        if latents.ndim != 5:
            raise ValueError(
                f"video_latents must be [B, z, F', h, w], got shape {tuple(latents.shape)}"
            )
        batch = int(latents.shape[0])
        z = int(latents.shape[1])
        if self._in_channels != z:
            # Readout can fake the I2V packing (context frames are both the "noisy" and the
            # "clean" half — see _build_hidden_states); TRAINING cannot: which frames are
            # context, and what the temporal mask encodes per diffusion step, is unresolved.
            # Wan2.2-TI2V-5B has in_channels == z == 48 and never reaches this branch.
            raise NotImplementedError(
                f"flow training needs a DiT whose in_channels ({self._in_channels}) equals the "
                f"latent width ({z}); this checkpoint uses the Wan I2V [noisy | mask | clean] "
                "packing, whose training semantics are unresolved — use Wan2.2-TI2V-5B for the "
                "flow pathway, or features()/features_by_block() for frozen readout"
            )
        t_vec = torch.as_tensor(t, dtype=torch.float32, device=latents.device)
        if t_vec.ndim == 0:
            t_vec = t_vec.reshape(1)
        if t_vec.ndim != 1:
            raise ValueError(f"t must be a scalar or [B] vector, got shape {tuple(t_vec.shape)}")
        if t_vec.shape[0] == 1 and batch > 1:
            t_vec = t_vec.expand(batch)
        if t_vec.shape[0] != batch:
            raise ValueError(f"t batch {t_vec.shape[0]} does not match latent batch {batch}")
        ts = ((1.0 - t_vec) * WAN_NUM_TRAIN_TIMESTEPS).to(torch.float32)

        hidden = latents.to(device=self._device_of(self._transformer), dtype=self._model_dtype())
        output, captured = self._forward_dit(
            hidden, text_ctx, state_ctx, timestep=ts.to(hidden.device)
        )
        features = torch.stack([captured[i] for i in self.feature_blocks], dim=0).mean(dim=0)
        return -output.float(), features.float()

    def frozen_part_names(self) -> tuple[str, ...]:
        """Attribute names of the parts frozen for good: the VAE and the text/image towers.

        The DiT is NOT in here — it is frozen too, but ``add_lora()`` deliberately reopens a
        low-rank slice of it, and a frozen-parts audit must not flag that as a leak.
        """
        names = ("_vae", "_text_encoder", "_image_encoder")
        return tuple(name for name in names if getattr(self, name) is not None)

    # ---- LoRA fine-tuning (T-16, PRD §10.3) ----------------------------------------------

    def add_lora(
        self,
        *,
        rank: int = 32,
        alpha: int = 64,
        dropout: float = 0.0,
        target_modules: tuple[str, ...] | list[str] | None = None,
        blocks: tuple[int, ...] | None = None,
        adapter_name: str = "wam",
        param_dtype: str = "float32",
    ) -> int:
        """Inject LoRA into the frozen DiT; returns the number of trainable parameters.

        Uses ``transformer.add_adapter()``, NOT ``peft.get_peft_model()``: the latter wraps the
        model and renames the module tree (``base_model.model.blocks.i``), which would break
        every ``.blocks[i]`` hook this adapter registers for feature readout — the action branch
        would go dark the moment fine-tuning starts.

        ``blocks`` restricts the adaptation to a few depths (e.g. the readout blocks and below);
        by default every block is adapted.
        """
        self._require_loaded()
        try:
            from peft import LoraConfig
        except ImportError as err:  # pragma: no cover - exercised on machines without the extra
            raise RuntimeError(
                f"LoRA fine-tuning requires peft (pip install peft) ({err})"
            ) from err

        attached = getattr(self._transformer, "peft_config", None) or {}
        if self._lora_adapter is not None or adapter_name in attached:
            raise RuntimeError(
                f"a LoRA adapter is already attached ({self._lora_adapter or adapter_name!r}); "
                "a second injection would stack adapters on the same base weights and silently "
                "change what a checkpoint means — build a fresh adapter instead"
            )
        targets = [str(name) for name in (target_modules or DEFAULT_LORA_TARGETS)]
        if getattr(self._transformer.config, "added_kv_proj_dim", None):
            # I2V checkpoints route the CLIP image tokens through attn2's extra k/v projections;
            # leaving those frozen would adapt only half of the cross-attention.
            targets += ["attn2.add_k_proj", "attn2.add_v_proj"]
        spec: str | list[str] = targets
        if blocks is not None:
            depths = tuple(int(b) for b in blocks)
            if not depths or any(b < 0 or b >= self._num_layers for b in depths):
                raise ValueError(
                    f"blocks must be non-empty indices in [0, {self._num_layers}), got {blocks!r}"
                )
            # peft matches plain target names by SUFFIX, which cannot express "only at these
            # depths" — a fully anchored regex can, and anchoring is what keeps 'blocks.1' from
            # also matching 'blocks.10'.
            suffixes = "|".join(re.escape(name) for name in targets)
            indices = "|".join(str(b) for b in depths)
            spec = rf"^blocks\.(?:{indices})\.(?:.*\.)?(?:{suffixes})$"

        self._transformer.add_adapter(
            LoraConfig(
                r=int(rank), lora_alpha=int(alpha), lora_dropout=float(dropout), target_modules=spec
            ),
            adapter_name=adapter_name,
        )
        trainable = self._activate_lora_parameters(param_dtype)
        if not trainable:  # pragma: no cover - peft raises first, but never train a no-op
            raise RuntimeError(f"LoRA injection matched no module against {spec!r}")
        self._lora_adapter = adapter_name
        self._lora_enabled = True
        return trainable

    def _activate_lora_parameters(self, param_dtype: str) -> int:
        """Unfreeze + upcast the freshly injected LoRA weights; returns their parameter count.

        ``attach()`` calls ``requires_grad_(False)`` on the whole DiT, and peft injects into
        that frozen tree — without this pass the optimizer would get an empty (or, worse, a
        silently bf16) parameter set and the run would look like it trains. fp32 masters keep
        Adam's second moment meaningful under a bf16 base.
        """
        import torch

        supported = {"float32": torch.float32, "bfloat16": torch.bfloat16}
        dtype = supported.get(param_dtype)
        if dtype is None:
            raise ValueError(
                f"unknown lora param_dtype {param_dtype!r}; supported: {sorted(supported)}"
            )
        total = 0
        for name, param in self._transformer.named_parameters():
            if "lora_" not in name:
                continue
            if param.dtype != dtype:
                param.data = param.data.to(dtype)
            param.requires_grad_(True)
            total += int(param.numel())
        return total

    def lora_parameters(self) -> dict[str, Any]:
        """``{qualified name: nn.Parameter}`` of the injected LoRA weights (optimizer input)."""
        self._require_loaded()
        return {n: p for n, p in self._transformer.named_parameters() if "lora_" in n}

    def trainable_parameters(self) -> dict[str, Any]:
        """Everything this adapter contributes to an optimizer: LoRA weights + state projection.

        Assembled by name instead of filtering ``requires_grad`` over the DiT: a base weight
        that somehow stayed unfrozen must show up as an audit failure, not as free training.
        """
        params = {f"transformer.{n}": p for n, p in self.lora_parameters().items()}
        if self._state_proj is not None:
            params.update({f"state_proj.{n}": p for n, p in self._state_proj.named_parameters()})
        return params

    def lora_state_dict(self) -> dict[str, Any]:
        """Detached CPU copy of the LoRA weights — all a fine-tune checkpoint has to carry."""
        return {name: p.detach().to("cpu").clone() for name, p in self.lora_parameters().items()}

    def load_lora_state_dict(self, state_dict: dict[str, Any]) -> None:
        """In-place inverse of :meth:`lora_state_dict` (the adapter must already be injected)."""
        import torch

        params = self.lora_parameters()
        if not params:
            raise RuntimeError("no LoRA adapter attached — call add_lora() or load_lora() first")
        missing = sorted(set(params) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(params))
        if missing or unexpected:
            raise ValueError(
                f"LoRA state dict does not match the injected adapter: "
                f"missing={missing[:4]}, unexpected={unexpected[:4]}"
            )
        with torch.no_grad():
            for name, param in params.items():
                value = self._as_tensor(state_dict[name])
                if tuple(value.shape) != tuple(param.shape):
                    raise ValueError(
                        f"LoRA weight {name} has shape {tuple(value.shape)}, expected "
                        f"{tuple(param.shape)}"
                    )
                param.copy_(value.to(device=param.device, dtype=param.dtype))

    def set_lora_enabled(self, enabled: bool) -> bool:
        """Switch the attached adapter on/off in place; returns the previous state.

        The base-arm control for any comparison that asks what the fine-tune did (T-35): same
        process, same weights in memory, same sampling noise, adapter bypassed. The alternative
        — loading the base a second time — costs 19 GB and puts a second model build between the
        two numbers being compared, which is exactly the kind of difference that later turns out
        to explain the result.

        Toggling is peft's own ``disable_adapters``/``enable_adapters`` on the DiT, so the delta
        is the adapter and nothing else. Deliberately NOT a scale knob: a partial scale is a
        model that was never trained, and the archived generate-tab table already shows that
        reading such a clip requires knowing the effective scaling is alpha/r = 2.0 and not 1.0.
        """
        self._require_loaded()
        if self._lora_adapter is None:
            raise RuntimeError("no LoRA adapter attached — nothing to enable or disable")
        was_enabled = self._lora_enabled
        if enabled:
            self._transformer.enable_adapters()
        else:
            self._transformer.disable_adapters()
        self._lora_enabled = bool(enabled)
        return was_enabled

    @property
    def lora_enabled(self) -> bool:
        """Whether the attached adapter is currently in the forward pass."""
        return self._lora_adapter is not None and self._lora_enabled

    def save_lora(self, directory: str | Path) -> Path:
        """Write the adapter in the diffusers LoRA layout; returns the weight file path.

        Portable on purpose: the same file loads into a stock diffusers Wan pipeline, so a
        checkpoint can be inspected (does the predicted video look right?) without any of WAM.
        """
        self._require_loaded()
        if self._lora_adapter is None:
            raise RuntimeError("no LoRA adapter attached — call add_lora() or load_lora() first")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._transformer.save_lora_adapter(str(path), adapter_name=self._lora_adapter)
        return path / _LORA_WEIGHT_NAME

    def load_lora(
        self, directory: str | Path, *, adapter_name: str = "wam", param_dtype: str = "float32"
    ) -> int:
        """Inverse of :meth:`save_lora`: inject a saved adapter; returns trainable param count.

        Do NOT call ``add_lora()`` first — peft rebuilds the config from the saved ranks.
        """
        self._require_loaded()
        if self._lora_adapter is not None:
            raise RuntimeError(
                f"a LoRA adapter is already attached ({self._lora_adapter!r}); load_lora() "
                "builds its own from the saved weights — start from a fresh adapter"
            )
        self._transformer.load_lora_adapter(
            str(Path(directory)),
            adapter_name=adapter_name,
            prefix=None,
            weight_name=_LORA_WEIGHT_NAME,
        )
        self._lora_adapter = adapter_name
        self._lora_enabled = True
        return self._activate_lora_parameters(param_dtype)

    def enable_gradient_checkpointing(self, enable: bool = True) -> None:
        """Trade DiT activation memory for a recompute (the 5B will not fit 24 GB otherwise).

        diffusers checkpoints with ``use_reentrant=False``, which is REQUIRED here and not just
        preferable: the reentrant variant detaches block outputs from the autograd graph, and
        the readout hooks capture exactly those outputs — the action branch would silently
        train on constants.
        """
        self._require_loaded()
        if enable:
            self._transformer.enable_gradient_checkpointing()
        else:
            self._transformer.disable_gradient_checkpointing()

    # ---- internals -----------------------------------------------------------------------

    def _build_hidden_states(self, latents: Any, condition_latents: Any = None) -> Any:
        """Latents -> DiT input channels (I2V checkpoints expect [latents, mask, condition])."""
        import torch

        z = int(latents.shape[1])
        if self._in_channels == z:  # TI2V-style: latents go in unchanged
            return latents
        mask_channels = self._in_channels - 2 * z
        if mask_channels != self._vae_temporal:
            raise RuntimeError(
                f"cannot build DiT input: in_channels={self._in_channels} matches neither "
                f"{z} (plain) nor 2*{z}+{self._vae_temporal} (Wan I2V layout)"
            )
        # Every context frame is observed, so the temporal mask is all-ones.
        mask = torch.ones(
            (latents.shape[0], mask_channels, *latents.shape[2:]),
            dtype=latents.dtype,
            device=latents.device,
        )
        # Readout has a single tensor and passes it as both halves — unchanged behaviour. A
        # sampler/trainer that separates noisy latents from clean context passes the latter as
        # condition_latents; the flow pathway refuses this layout outright (see forward_flow).
        condition = latents if condition_latents is None else condition_latents
        return torch.cat([latents, mask, condition], dim=1)

    def _check_pixel_grid(self, pixels: Any) -> None:
        """Reject frame sizes the VAE + patch embedder cannot tile (H, W multiples of s*p)."""
        mod_h = self._vae_spatial * self._patch_size[1]
        mod_w = self._vae_spatial * self._patch_size[2]
        _, _, _, height, width = pixels.shape
        if height % mod_h or width % mod_w:
            raise ValueError(
                f"frame size {height}x{width} must be a multiple of {mod_h}x{mod_w} "
                f"(vae stride {self._vae_spatial} x patch {self._patch_size[1:]})"
            )

    def _resize_pixels(self, pixels: Any, image_hw: tuple[int, int]) -> Any:
        """Bilinear per-frame resize of [B, 3, F, H, W] (no temporal interpolation, ever)."""
        import torch

        batch, channels, frames, height, width = pixels.shape
        if (height, width) == image_hw:
            return pixels
        flat = pixels.transpose(1, 2).reshape(batch * frames, channels, height, width)
        resized = torch.nn.functional.interpolate(
            flat, size=image_hw, mode="bilinear", align_corners=False
        )
        return resized.reshape(batch, frames, channels, *image_hw).transpose(1, 2).contiguous()

    def _normalize_latents(self, latents: Any) -> Any:
        import torch

        cfg = self._vae.config
        mean = getattr(cfg, "latents_mean", None)
        std = getattr(cfg, "latents_std", None)
        if not mean or not std:
            return latents
        shape = (1, -1, 1, 1, 1)
        mean_t = torch.tensor(mean, dtype=latents.dtype, device=latents.device).view(*shape)
        std_t = torch.tensor(std, dtype=latents.dtype, device=latents.device).view(*shape)
        return (latents - mean_t) / std_t

    def denormalize_latents(self, latents: Any) -> Any:
        """Inverse of :meth:`_normalize_latents`: back from unit scale into the VAE's own scale.

        Public because anything that hands latents to the VAE decoder needs it — the flow model
        works entirely in the normalized space, the decoder only understands the raw one.
        """
        import torch

        cfg = self._vae.config
        mean = getattr(cfg, "latents_mean", None)
        std = getattr(cfg, "latents_std", None)
        if not mean or not std:
            return latents
        shape = (1, -1, 1, 1, 1)
        mean_t = torch.tensor(mean, dtype=latents.dtype, device=latents.device).view(*shape)
        std_t = torch.tensor(std, dtype=latents.dtype, device=latents.device).view(*shape)
        return latents * std_t + mean_t

    def _encode_last_frame(self, pixels: Any) -> Any:
        """CLIP embedding of the last observed frame (diffusers uses hidden_states[-2])."""
        if self._image_encoder is None:
            return None
        import torch

        enc_device = self._device_of(self._image_encoder)
        last = pixels[:, :, -1]  # [B, 3, H, W] in [-1, 1]
        images = ((last.float() + 1.0) / 2.0).clamp(0.0, 1.0)
        if self._image_processor is not None:
            processed = self._image_processor(images=list(images.cpu()), return_tensors="pt")
            values = processed["pixel_values"].to(enc_device, dtype=torch.float32)
        else:  # pragma: no cover - only for stubs without a processor
            values = images.to(enc_device, dtype=torch.float32)
        out = self._image_encoder(values, output_hidden_states=True)
        return out.hidden_states[-2]

    def _to_pixel_tensor(self, video: Any) -> Any:
        import torch

        x = self._as_tensor(video)
        if x.ndim == 4:
            x = x.unsqueeze(0)
        if x.ndim != 5:
            raise ValueError(f"video must be [B, F, H, W, 3] or [F, H, W, 3], got {tuple(x.shape)}")
        if x.shape[-1] != 3:
            raise ValueError(f"video must end in 3 RGB channels, got {tuple(x.shape)}")
        if x.dtype == torch.uint8 or (x.dtype.is_floating_point and float(x.max()) > 1.5):
            x = x.to(torch.float32) / 127.5 - 1.0
        x = x.to(torch.float32).clamp(-1.0, 1.0)
        return x.permute(0, 4, 1, 2, 3).contiguous().to(self.device)  # [B, 3, F, H, W]

    @staticmethod
    def _device_of(module: Any) -> Any:
        """Device a module's weights actually live on (may differ after ``offload()``)."""
        return next(module.parameters()).device

    @staticmethod
    def _as_tensor(value: Any) -> Any:
        import numpy as np
        import torch

        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(value))
        return torch.as_tensor(value)

    def _model_dtype(self) -> Any:
        import torch

        supported = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = supported.get(self.dtype)
        if dtype is None:
            raise ValueError(f"unknown dtype {self.dtype!r}; supported: {sorted(supported)}")
        return dtype
