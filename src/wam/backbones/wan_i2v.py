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

Swapping this adapter in must not change the data schema or the robot API (FR-09/AC-05).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

_WEIGHTS_MISSING_MSG = (
    "Wan2.1 weights not available — pass checkpoint_path=<local snapshot dir> (or "
    "model_id=<hub repo> with allow_download=True) and install the runtime "
    "(diffusers>=0.35 + transformers); nothing is downloaded implicitly."
)


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
        common = {"local_files_only": not self.allow_download}
        # VAE stays fp32 (diffusers guidance: better encode/decode quality).
        vae = AutoencoderKLWan.from_pretrained(
            source, subfolder="vae", torch_dtype=torch.float32, **common
        )
        transformer = WanTransformer3DModel.from_pretrained(
            source, subfolder="transformer", torch_dtype=torch_dtype, **common
        )
        text_encoder = UMT5EncoderModel.from_pretrained(
            source, subfolder="text_encoder", torch_dtype=torch_dtype, **common
        )
        tokenizer = AutoTokenizer.from_pretrained(source, subfolder="tokenizer", **common)
        image_encoder = image_processor = None
        if getattr(transformer.config, "image_dim", None):
            from transformers import CLIPImageProcessor, CLIPVisionModel

            image_encoder = CLIPVisionModel.from_pretrained(
                source, subfolder="image_encoder", torch_dtype=torch.float32, **common
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
        mod_h = self._vae_spatial * self._patch_size[1]
        mod_w = self._vae_spatial * self._patch_size[2]
        _, _, _, height, width = pixels.shape
        if height % mod_h or width % mod_w:
            raise ValueError(
                f"frame size {height}x{width} must be a multiple of {mod_h}x{mod_w} "
                f"(vae stride {self._vae_spatial} x patch {self._patch_size[1:]})"
            )
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
        """
        self._require_loaded()
        import torch

        if self._tokenizer is None or self._text_encoder is None:
            raise RuntimeError("no text encoder attached — load() builds tokenizer + umT5")
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
        return masked.to(device=self.device, dtype=self._model_dtype())

    def condition_state(self, state_embedding: Any) -> Any:
        """StateEncoder output ``[B, E]`` -> one extra context token ``[B, 1, text_dim]``.

        The projection is created lazily on first use (fp32, ``requires_grad=True``) — it is
        the adapter's only trainable parameter block; register it with the optimizer via
        ``state_projection``.
        """
        self._require_loaded()
        import torch
        from torch import nn

        x = self._as_tensor(state_embedding).to(torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError(f"state_embedding must be [B, E] or [E], got shape {tuple(x.shape)}")
        embed_dim = int(x.shape[-1])
        if self._state_proj is None:
            self._state_proj = nn.Linear(embed_dim, self._text_dim).to(self.device)
        elif self._state_proj.in_features != embed_dim:
            raise ValueError(
                f"state embedding dim changed: projection expects "
                f"{self._state_proj.in_features}, got {embed_dim}"
            )
        token = self._state_proj(x.to(self.device)).unsqueeze(1)
        return token.to(self._model_dtype())

    # ---- feature readout ---------------------------------------------------------------

    def features(self, video_ctx: Any, text_ctx: Any, state_ctx: Any) -> Any:
        """One DiT forward with hooks on ``feature_blocks`` -> ``[B, S, feature_dim]``.

        ``video_ctx`` is the dict from :meth:`condition_video` (or a bare latent tensor);
        ``state_ctx`` may be None. The denoising step is the adapter's ``timestep`` attribute
        (the protocol signature is fixed — FR-09). Gradients follow the caller's autograd
        context; wrap in ``torch.no_grad()`` for pure inference.
        """
        self._require_loaded()
        import torch

        latents = video_ctx["latents"] if isinstance(video_ctx, dict) else video_ctx
        image_embeds = video_ctx.get("image_embeds") if isinstance(video_ctx, dict) else None
        hidden_states = self._build_hidden_states(latents)
        if text_ctx is None:
            raise ValueError("text_ctx is required (condition_text output)")
        encoder_hidden_states = text_ctx
        if state_ctx is not None:
            encoder_hidden_states = torch.cat([text_ctx, state_ctx], dim=1)
        if self._image_dim and image_embeds is None:
            raise ValueError(
                "this checkpoint expects CLIP image conditioning "
                f"(image_dim={self._image_dim}) — pass the condition_video() dict"
            )
        ts = torch.full(
            (hidden_states.shape[0],), int(self.timestep), dtype=torch.long, device=self.device
        )

        captured: dict[int, Any] = {}
        handles = []
        blocks = self._transformer.blocks
        for index in self.feature_blocks:
            handles.append(blocks[index].register_forward_hook(self._make_hook(captured, index)))
        try:
            self._transformer(
                hidden_states=hidden_states,
                timestep=ts,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=image_embeds,
                return_dict=False,
            )
        finally:
            for handle in handles:
                handle.remove()

        missing = [i for i in self.feature_blocks if i not in captured]
        if missing:
            raise RuntimeError(f"no activation captured for blocks {missing}")
        stacked = torch.stack([captured[i] for i in self.feature_blocks], dim=0)
        if stacked.shape[-1] != self._feature_dim:
            raise RuntimeError(
                f"hooked activations have last dim {stacked.shape[-1]}, expected "
                f"{self._feature_dim} — block output is not the residual stream"
            )
        return stacked.mean(dim=0)

    @staticmethod
    def _make_hook(sink: dict[int, Any], index: int) -> Any:
        def hook(_module: Any, _args: Any, output: Any) -> None:
            sink[index] = output[0] if isinstance(output, tuple) else output

        return hook

    def expected_token_count(self, num_frames: int, height: int, width: int) -> int:
        """S for ``num_frames`` pixel frames at ``height`` x ``width`` (shape self-check)."""
        latent_frames = 1 + (num_frames - 1) // self._vae_temporal
        p_t, p_h, p_w = self._patch_size
        return (
            (latent_frames // p_t)
            * (height // self._vae_spatial // p_h)
            * (width // self._vae_spatial // p_w)
        )

    # ---- internals -----------------------------------------------------------------------

    def _build_hidden_states(self, latents: Any) -> Any:
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
        return torch.cat([latents, mask, latents], dim=1)

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
