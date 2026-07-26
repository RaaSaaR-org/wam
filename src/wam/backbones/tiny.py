"""Tiny fully-functional video backbone for tests and D1 overfit runs (T-15, FR-09).

Contracts:
- Implements ``wam.interfaces.protocols.BackboneAdapter`` exactly (structural conformance);
  additionally exposes a video head (``predict_video_latents``) for the video-branch loss and
  a flow-matching pathway (``forward_flow``) for joint video/action training (T-16, §10.3).
- Fully self-contained: NO external tokenizer, NO downloads. Text conditioning uses a
  deterministic crc32 hash embedding; the "VAE" is the identity (tiny operates directly in
  pixel space, so "video latents" here means float frames in [B, F, H, W, 3]).
- Deterministic: no dropout, no RNG in forward; construction under ``torch.manual_seed`` is
  reproducible bit-for-bit on CPU.
- Token layout contract: sequences built by ``features()``/``forward_flow`` keep the VIDEO
  tokens FIRST (``config.num_video_tokens`` of them), then text, then one state token.
  ``predict_video_latents`` relies on this ordering.
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn

BACKBONE_NAME = "tiny"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class TinyBackboneConfig(BaseModel):
    """Configuration for :class:`TinyVideoBackbone`. Frozen (hashable into config_hash)."""

    model_config = ConfigDict(frozen=True)

    feature_dim: int = Field(default=64, gt=0, description="Token width; last dim of features().")
    patch_size: int = Field(default=8, gt=0, description="Square spatial patch edge in pixels.")
    depth: int = Field(default=2, ge=1, le=4, description="Number of transformer blocks (1-2).")
    num_heads: int = Field(default=4, gt=0)
    num_frames: int = Field(default=4, ge=1, description="Context/prediction frames per clip.")
    image_hw: tuple[int, int] = Field(default=(32, 32), description="Expected frame (H, W).")
    text_vocab: int = Field(default=256, gt=0, description="Hash-embedding table size.")
    max_text_tokens: int = Field(default=16, gt=0)
    state_embedding_dim: int = Field(
        default=32, gt=0, description="Expected StateEncoder.embedding_dim of the input."
    )

    @model_validator(mode="after")
    def _validate(self) -> TinyBackboneConfig:
        h, w = self.image_hw
        if h <= 0 or w <= 0 or h % self.patch_size or w % self.patch_size:
            raise ValueError(
                f"image_hw {self.image_hw} must be positive and divisible by "
                f"patch_size {self.patch_size}"
            )
        if self.feature_dim % self.num_heads:
            raise ValueError(
                f"feature_dim {self.feature_dim} must be divisible by num_heads {self.num_heads}"
            )
        return self

    @property
    def patches_per_frame(self) -> int:
        h, w = self.image_hw
        return (h // self.patch_size) * (w // self.patch_size)

    @property
    def num_video_tokens(self) -> int:
        return self.num_frames * self.patches_per_frame


def _sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    """[B] float timesteps in [0, 1] -> [B, dim] sinusoidal embedding (deterministic)."""
    half = dim // 2
    exponent = -math.log(10000.0) / max(half - 1, 1)
    freqs = torch.exp(exponent * torch.arange(half, dtype=torch.float32, device=t.device))
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class TinyVideoBackbone(nn.Module):
    """Small patchify-transformer backbone. Functional stand-in for Wan2.1/FLUX 3 adapters:
    same interface, tiny dims, CPU-fast, deterministic."""

    def __init__(self, config: TinyBackboneConfig | None = None) -> None:
        super().__init__()
        self.config = config or TinyBackboneConfig()
        c = self.config
        d = c.feature_dim
        patch_dim = 3 * c.patch_size * c.patch_size

        self.video_proj = nn.Linear(patch_dim, d)
        self.text_embedding = nn.Embedding(c.text_vocab, d)
        self.state_proj = nn.Linear(c.state_embedding_dim, d)
        self.video_pos = nn.Parameter(torch.empty(1, c.num_video_tokens, d))
        self.text_pos = nn.Parameter(torch.empty(1, c.max_text_tokens, d))
        self.modality = nn.Parameter(torch.empty(3, d))  # 0=video, 1=text, 2=state
        nn.init.normal_(self.video_pos, std=0.02)
        nn.init.normal_(self.text_pos, std=0.02)
        nn.init.normal_(self.modality, std=0.02)

        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=c.num_heads,
            dim_feedforward=2 * d,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=c.depth, enable_nested_tensor=False)
        # Small deconv head: per-patch feature -> pixel patch (video branch loss / velocity).
        self.video_head = nn.ConvTranspose2d(d, 3, kernel_size=c.patch_size, stride=c.patch_size)

    # ---- BackboneAdapter protocol -------------------------------------------------------

    @property
    def name(self) -> str:
        return BACKBONE_NAME

    @property
    def feature_dim(self) -> int:
        return self.config.feature_dim

    def condition_video(self, video: Any) -> Any:
        """uint8/float video [B, F, H, W, 3] or [F, H, W, 3] (numpy or torch) ->
        video tokens [B, num_video_tokens, feature_dim]."""
        frames = self._to_video_tensor(video)
        tokens = self.video_proj(self._patchify(frames))
        return tokens + self.video_pos + self.modality[0]

    def condition_text(self, text: str) -> Any:
        """Instruction string -> [1, T, feature_dim] via deterministic crc32 hash embedding
        (T <= max_text_tokens; empty text maps to one padding token). No external tokenizer."""
        ids = self._hash_token_ids(text)
        idx = torch.tensor([ids], dtype=torch.long, device=self.modality.device)
        emb = self.text_embedding(idx) + self.text_pos[:, : idx.shape[1]]
        return emb + self.modality[1]

    def condition_state(self, state_embedding: Any) -> Any:
        """StateEncoder output [E], [B, E] or [B, 1, E] -> one state token [B, 1, feature_dim]."""
        emb = torch.as_tensor(state_embedding, dtype=torch.float32)
        if emb.ndim == 1:
            emb = emb[None, None, :]
        elif emb.ndim == 2:
            emb = emb[:, None, :]
        if emb.ndim != 3 or emb.shape[1] != 1 or emb.shape[-1] != self.config.state_embedding_dim:
            raise ValueError(
                f"state_embedding must be [E], [B, E] or [B, 1, E] with "
                f"E={self.config.state_embedding_dim}, got {tuple(emb.shape)}"
            )
        return self.state_proj(emb) + self.modality[2]

    def features(self, video_ctx: Any, text_ctx: Any, state_ctx: Any) -> Any:
        """Run the transformer on [video | text | state] tokens -> [B, S, feature_dim].
        Video tokens stay FIRST in the sequence (see module docstring)."""
        seq = self._build_sequence(video_ctx, text_ctx, state_ctx)
        return self.blocks(seq)

    def forward(self, video_ctx: Any, text_ctx: Any, state_ctx: Any) -> Any:
        return self.features(video_ctx, text_ctx, state_ctx)

    # ---- Video branch / flow-matching pathway -------------------------------------------

    def predict_video_latents(self, features: Any) -> Tensor:
        """Features [B, S, feature_dim] (video tokens first) -> predicted video "latents"
        [B, F, H, W, 3] via the deconv head. Used for the video-branch loss (FR-03)."""
        feats = torch.as_tensor(features)
        n = self.config.num_video_tokens
        if feats.ndim != 3 or feats.shape[-1] != self.config.feature_dim or feats.shape[1] < n:
            raise ValueError(
                f"features must be [B, S>={n}, {self.config.feature_dim}], got {tuple(feats.shape)}"
            )
        return self._unpatchify_head(feats[:, :n])

    def forward_flow(
        self, video_latents: Any, t: Any, text_ctx: Any, state_ctx: Any
    ) -> tuple[Tensor, Tensor]:
        """Flow-matching pathway: noisy video "latents" [B, F, H, W, 3] (or unbatched) plus
        timestep ``t`` (scalar or [B], in [0, 1]) conditioned on text/state contexts.

        Returns ``(velocity_pred, features)`` where ``velocity_pred`` has the video shape
        (flow-matching target) and ``features`` is [B, S, feature_dim] for the action branch.
        """
        frames = self._to_video_tensor(video_latents)
        b = frames.shape[0]
        t_vec = torch.as_tensor(t, dtype=torch.float32, device=frames.device)
        if t_vec.ndim == 0:
            t_vec = t_vec.reshape(1)
        if t_vec.ndim != 1:
            raise ValueError(f"t must be a scalar or [B] vector, got shape {tuple(t_vec.shape)}")
        if t_vec.shape[0] == 1 and b > 1:
            t_vec = t_vec.expand(b)
        if t_vec.shape[0] != b:
            raise ValueError(f"t batch {t_vec.shape[0]} does not match video batch {b}")

        tokens = self.video_proj(self._patchify(frames)) + self.video_pos + self.modality[0]
        temb = self.time_mlp(_sinusoidal_embedding(t_vec, self.config.feature_dim))
        tokens = tokens + temb[:, None, :]
        feats = self.blocks(self._build_sequence(tokens, text_ctx, state_ctx))
        velocity = self._unpatchify_head(feats[:, : self.config.num_video_tokens])
        return velocity, feats

    # ---- Internals -----------------------------------------------------------------------

    def _to_video_tensor(self, video: Any) -> Tensor:
        if isinstance(video, np.ndarray):
            frames = torch.from_numpy(np.ascontiguousarray(video))
        else:
            frames = torch.as_tensor(video)
        if frames.ndim == 4:
            frames = frames[None]
        c = self.config
        expected = (c.num_frames, *c.image_hw, 3)
        if frames.ndim != 5 or tuple(frames.shape[1:]) != expected:
            raise ValueError(
                f"video must be [B, F, H, W, 3] or [F, H, W, 3] with (F, H, W, C)={expected}, "
                f"got shape {tuple(frames.shape)}"
            )
        frames = frames.to(self.modality.device)
        if frames.dtype == torch.uint8:
            return frames.float() / 255.0
        return frames.float()

    def _patchify(self, frames: Tensor) -> Tensor:
        """[B, F, H, W, 3] -> [B, num_video_tokens, 3 * patch_size**2]."""
        b, f, h, w, _ = frames.shape
        p = self.config.patch_size
        x = frames.reshape(b, f, h // p, p, w // p, p, 3)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)
        return x.reshape(b, f * (h // p) * (w // p), 3 * p * p)

    def _unpatchify_head(self, video_tokens: Tensor) -> Tensor:
        """[B, num_video_tokens, D] -> deconv head -> [B, F, H, W, 3]."""
        c = self.config
        h, w = c.image_hw
        p = c.patch_size
        b = video_tokens.shape[0]
        x = video_tokens.reshape(b * c.num_frames, h // p, w // p, c.feature_dim)
        x = self.video_head(x.permute(0, 3, 1, 2))
        return x.permute(0, 2, 3, 1).reshape(b, c.num_frames, h, w, 3)

    def _hash_token_ids(self, text: str) -> list[int]:
        words = _TOKEN_RE.findall(text.lower())[: self.config.max_text_tokens]
        if not words:
            return [0]
        return [zlib.crc32(word.encode("utf-8")) % self.config.text_vocab for word in words]

    def _build_sequence(self, video_ctx: Any, text_ctx: Any, state_ctx: Any) -> Tensor:
        video = torch.as_tensor(video_ctx)
        d = self.config.feature_dim
        n = self.config.num_video_tokens
        if video.ndim != 3 or video.shape[1] != n or video.shape[-1] != d:
            raise ValueError(f"video_ctx must be [B, {n}, {d}], got {tuple(video.shape)}")
        b = video.shape[0]
        text = self._broadcast_batch(torch.as_tensor(text_ctx), b, "text_ctx")
        state = self._broadcast_batch(torch.as_tensor(state_ctx), b, "state_ctx")
        return torch.cat([video, text, state], dim=1)

    def _broadcast_batch(self, ctx: Tensor, batch: int, label: str) -> Tensor:
        if ctx.ndim != 3 or ctx.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"{label} must be [B, T, {self.config.feature_dim}], got {tuple(ctx.shape)}"
            )
        if ctx.shape[0] == batch:
            return ctx
        if ctx.shape[0] == 1:
            return ctx.expand(batch, -1, -1)
        raise ValueError(f"{label} batch {ctx.shape[0]} does not match video batch {batch}")
