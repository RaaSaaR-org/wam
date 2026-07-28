"""Trainable ``FlowBackbone`` module wrapping :class:`WanI2VAdapter` (T-16, PRD §10.3).

``JointWorldActionModel`` needs an ``nn.Module``: only registered parameters reach the
optimizer, ``.to(device)``, ``clip_grad_norm_`` and the checkpoint. ``WanI2VAdapter`` is
deliberately NOT one — it is a plain object holding three frozen third-party towers.

The whole design of this wrapper is about where that module boundary is drawn: around what
TRAINS, and nothing else. Registering the DiT (or the VAE, or umT5) as a submodule would make
``model.state_dict()`` ~10 GB, make ``TrainingMonitor.snapshot_params`` clone those 10 GB on
every monitored step, and degrade "save adapters only" into a flag someone has to remember on
every call site. So exactly two things are registered:

- ``state_proj`` — the state -> text-context projection, the adapter's only non-LoRA trainable;
- ``lora``      — an ``nn.ParameterDict`` ALIASING the LoRA parameters peft injected into the
  DiT blocks. Aliases, not copies: the very same ``nn.Parameter`` objects, so AdamW,
  ``clip_grad_norm_`` and ``TrainingMonitor.module_grad_norms`` all act on the tensors the DiT
  actually computes with. ``nn.Module.__setattr__`` only intercepts Parameter/Module/Tensor, so
  the adapter itself passes straight through into ``__dict__`` and stays invisible to
  ``state_dict()`` / ``parameters()`` / ``.to()``.

That last point is also why :meth:`_apply` is overridden: the frozen towers would otherwise be
stranded on CPU when the trainer moves the model to the GPU. It forwards DEVICE moves only —
never dtype, since the DiT is bf16, the VAE fp32 and the LoRA masters fp32 by design (§10.3) —
and re-points the aliases afterwards, because a cross-device ``_apply`` rebuilds parameters
instead of updating them in place and would otherwise leave the optimizer holding orphans.

The authoritative training artifact stays peft's own adapter directory
(:meth:`save_adapter`), not this module's ``state_dict``: it loads into a stock diffusers Wan
pipeline, so a checkpoint can be watched as video without any of WAM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from wam.backbones.wan_i2v import WanBackboneConfig, WanI2VAdapter

__all__ = ["WanFlowBackbone"]

#: ``nn.ParameterDict`` keys may not contain "." (``register_parameter`` rejects them) while
#: peft's parameter names are full dotted paths. "__" is the replacement: no diffusers or peft
#: name contains a double underscore, so the mangling stays collision-free and reversible.
_DOT_REPLACEMENT = "__"

_STATE_PROJ_FILE = "state_proj.safetensors"
_DESCRIBE_FILE = "backbone.json"

#: Modules the adapter holds but nobody registers. Named here (rather than reached for one by
#: one) because every device move has to cover all of them or the run splits across devices.
_HELD_MODULE_ATTRS = ("_transformer", "_vae", "_text_encoder", "_image_encoder")


class WanFlowBackbone(nn.Module):
    """Wan DiT + VAE + umT5 behind the ``FlowBackbone`` protocol, with only the adapters
    registered as parameters (see module docstring).

    Construction is free: it builds the :class:`WanI2VAdapter` skeleton and nothing else.
    :meth:`load` pulls the weights and runs the whole §10.3 setup (geometry check, state
    projection, LoRA injection, gradient checkpointing, aliasing); :meth:`attach` is the same
    setup over already-constructed modules, which is how the CPU tests exercise this class
    without any Wan weights.
    """

    def __init__(self, config: WanBackboneConfig, adapter: WanI2VAdapter | None = None) -> None:
        super().__init__()
        self.config = config
        # PLAIN attribute (module docstring): a WanI2VAdapter is not an nn.Module, so
        # nn.Module.__setattr__ passes it through to __dict__ and its 10 GB of held weights
        # never enter state_dict()/parameters()/.to(). Guarded because that is a load-bearing
        # property of this file, not a coincidence.
        adapter = adapter if adapter is not None else self._build_adapter(config)
        if isinstance(adapter, nn.Module):
            raise TypeError(
                f"{type(adapter).__name__} is an nn.Module — holding it here would register the "
                "DiT/VAE/text tower as submodules and put ~10 GB of frozen weights into every "
                "state_dict() and every snapshot_params() call"
            )
        self._adapter = adapter
        # The two trainable parts. Both are filled in by attach()/load(); an unloaded backbone
        # has no parameters at all, which is exactly right — there is nothing to optimize yet.
        self.state_proj: nn.Linear | None = None
        self.lora = nn.ParameterDict()

    @staticmethod
    def _build_adapter(config: WanBackboneConfig) -> WanI2VAdapter:
        """Config -> adapter skeleton. No filesystem, no network, no GPU (that is ``load()``)."""
        return WanI2VAdapter(
            checkpoint_path=config.checkpoint_path,
            feature_blocks=config.feature_blocks,
            device=config.device,
            model_id=config.model_id,
            dtype=config.dtype,
            max_text_tokens=config.max_text_tokens,
            allow_download=config.allow_download,
            device_map=config.device_map,
        )

    # ---- identity / held modules ---------------------------------------------------------

    @property
    def adapter(self) -> WanI2VAdapter:
        """The wrapped adapter (read-only: swapping it would orphan the LoRA aliases)."""
        return self._adapter

    @property
    def name(self) -> str:
        return self._adapter.name

    @property
    def feature_dim(self) -> int:
        """DiT inner dim, from the CONFIG: the action head is sized from it before any weight
        exists, and :meth:`_check_geometry` refuses a checkpoint that disagrees."""
        return self.config.feature_dim

    @property
    def is_loaded(self) -> bool:
        return self._adapter.is_loaded

    @property
    def vae(self) -> Any:
        """Frozen Wan-VAE. A read-only property, NOT a submodule: the frozen-parts registry
        resolves ``frozen_part_names()`` with ``getattr``, which a property satisfies without
        pulling 500 MB of weights into ``state_dict()``."""
        return self._adapter._vae

    @property
    def text_encoder(self) -> Any:
        """Frozen umT5 tower — same read-only-property reasoning as :attr:`vae`."""
        return self._adapter._text_encoder

    def frozen_part_names(self) -> tuple[str, ...]:
        """The parts frozen for good (PRD §10.3 step 4).

        The DiT is deliberately absent: it is frozen too, but ``add_lora()`` reopens a low-rank
        slice of it and a frozen-parts audit must not flag that as a leak.
        """
        return ("vae", "text_encoder")

    def _held_modules(self) -> tuple[nn.Module, ...]:
        """The attached, unregistered modules — the ones ``.to()`` would otherwise miss."""
        modules = (getattr(self._adapter, attr) for attr in _HELD_MODULE_ATTRS)
        return tuple(m for m in modules if m is not None)

    # ---- setup ---------------------------------------------------------------------------

    def load(self, *, lora_dir: str | Path | None = None) -> None:
        """Load the weights and run the full §10.3 setup. Heavyweight (disk/network/GPU).

        ``lora_dir`` resumes from a :meth:`save_adapter` directory instead of injecting fresh
        adapters — the requeue path of a Slurm run (``--resume latest``).
        """
        self._adapter.load()
        self._setup(lora_dir)

    def attach(self, *, lora_dir: str | Path | None = None, **modules: Any) -> None:
        """Attach already-constructed modules (``transformer=``, ``vae=``, ...) + the same setup.

        Same keyword arguments as :meth:`WanI2VAdapter.attach`. This is the ONLY way the whole
        training pathway is reachable on CPU without Wan weights, so it is a first-class entry
        point rather than a test hook.
        """
        self._adapter.attach(**modules)
        self._setup(lora_dir)

    def _setup(self, lora_dir: str | Path | None) -> None:
        cfg = self.config
        self._check_geometry()
        # Eagerly, before the optimizer exists: a projection built lazily inside the first
        # condition_state() call would never be registered and would silently train nothing.
        self.state_proj = self._adapter.build_state_projection(cfg.state_embedding_dim)
        if lora_dir is None:
            self._adapter.add_lora(
                rank=cfg.lora_rank,
                alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
                target_modules=cfg.lora_targets,
                blocks=cfg.lora_blocks,
                param_dtype=cfg.lora_param_dtype,
            )
        else:
            self._load_adapter_dir(Path(lora_dir))
        self._adapter.enable_gradient_checkpointing(cfg.gradient_checkpointing)
        self._alias_lora_parameters()

    def _check_geometry(self) -> None:
        """Refuse a loaded model that does not match the config — HERE, not 2000 steps in.

        Every one of these is otherwise a late, misleading failure: a feature_dim mismatch
        surfaces as a shape error deep in the action head, an illegal ``image_hw`` only on the
        first batch, and the I2V channel packing only inside ``forward_flow``. The
        feature_blocks RANGE is already checked by ``attach()`` against the real block count
        (the adapter's ``__init__`` can only check against the deepest known Wan, 40 blocks, so
        depths 30..39 construct fine and break on the 30-block TI2V-5B); what is checked here
        is that the adapter ended up reading out the depths this config asked for at all.
        """
        cfg = self.config
        geometry = self._adapter.describe()
        if self._adapter.feature_dim != cfg.feature_dim:
            raise RuntimeError(
                f"loaded DiT has inner dim {self._adapter.feature_dim} but the config declares "
                f"feature_dim={cfg.feature_dim} — the action head is built from the config, so "
                "this would only surface as a shape error inside the head"
            )
        if tuple(self._adapter.feature_blocks) != tuple(cfg.feature_blocks):
            raise RuntimeError(
                f"adapter reads out blocks {tuple(self._adapter.feature_blocks)} but the config "
                f"asks for {tuple(cfg.feature_blocks)} — an injected adapter must be built from "
                "this same config (readout depth is a measured quantity, not a default)"
            )
        latent_channels, in_channels = geometry["latent_channels"], geometry["in_channels"]
        if in_channels != latent_channels:
            raise NotImplementedError(
                f"flow training needs a DiT whose in_channels ({in_channels}) equals the VAE "
                f"latent width ({latent_channels}); this checkpoint uses the Wan I2V "
                "[noisy | mask | clean] packing — use Wan2.2-TI2V-5B for the flow pathway"
            )
        stride, patch = geometry["vae_spatial_stride"], geometry["patch_size"]
        mod_h, mod_w = stride * patch[1], stride * patch[2]
        height, width = cfg.image_hw
        if height % mod_h or width % mod_w:
            raise ValueError(
                f"image_hw {tuple(cfg.image_hw)} is not DiT-legal: H and W must be multiples of "
                f"{mod_h}x{mod_w} (vae stride {stride} x patch {tuple(patch[1:])}). Every batch "
                "is resized to image_hw, so this would fail on the first one"
            )

    def _alias_lora_parameters(self) -> None:
        """(Re-)point ``self.lora`` at the DiT's LoRA parameters — the same objects, never copies.

        Rebuilt rather than mutated because ``_apply`` may have replaced the parameter objects
        (see :meth:`_apply`); the adapter's tree is always the authority.
        """
        aliased = nn.ParameterDict()
        for name, param in self._adapter.lora_parameters().items():
            aliased[self._mangle(name)] = param
        self.lora = aliased

    @staticmethod
    def _mangle(name: str) -> str:
        """``blocks.0.attn1.to_q.lora_A.default.weight`` -> a legal ParameterDict key."""
        if _DOT_REPLACEMENT in name:
            raise RuntimeError(
                f"cannot alias LoRA parameter {name!r}: it already contains "
                f"{_DOT_REPLACEMENT!r}, so the key would not map back to a unique module path"
            )
        return name.replace(".", _DOT_REPLACEMENT)

    # ---- device placement ------------------------------------------------------------------

    def _apply(self, fn: Any, recurse: bool = True) -> WanFlowBackbone:
        """Forward DEVICE moves to the held modules; never dtype (§10.3 mixes them on purpose).

        Without this, ``trainer.model.to("cuda")`` would move the LoRA adapters and the state
        projection while leaving the DiT, VAE and text tower on CPU — every forward would then
        die on a device mismatch, or worse, run at CPU speed.
        """
        module = super()._apply(fn, recurse=recurse)
        device = self._moved_device(fn)
        if device is not None:
            for held in self._held_modules():
                held.to(device)
            # The adapter places its own intermediates (pixels, the readout timestep) on this
            # string; leaving it behind would split the batch across devices.
            self._adapter.device = str(device)
        # A cross-device _apply cannot update parameters in place, so it REPLACES them and the
        # aliases above now point at orphans the DiT no longer uses — and the optimizer is
        # built after .to(), so it would faithfully optimize those orphans forever.
        if len(self.lora):
            self._alias_lora_parameters()
        return module

    def _moved_device(self, fn: Any) -> torch.device | None:
        """Device ``fn`` moves tensors to, or ``None`` when it is not a device move.

        Probed on a tensor that already lives where the held modules live, which is what makes
        the three cases separable: a pure dtype cast comes back on the same device (skip),
        ``.to("cpu")`` on a CPU model is a genuine no-op (skip), and ``.to("cpu")`` on a
        GPU-resident model does move (forward it).
        """
        if self.config.device_map:
            # accelerate already placed every shard, possibly across several devices; a blanket
            # .to() would undo that and, on the hardware that needs device_map, OOM.
            return None
        modules = self._held_modules()
        if not modules:
            return None
        current = next(modules[0].parameters()).device
        with torch.no_grad():
            probe = fn(torch.zeros((), device=current))
        return None if probe.device == current else probe.device

    # ---- BackboneAdapter: conditioning + frozen readout ------------------------------------

    def condition_video(self, video: Any) -> Any:
        return self._adapter.condition_video(video)

    def condition_text(self, text: str | list[str]) -> Any:
        return self._adapter.condition_text(text)

    def condition_state(self, state_embedding: Any) -> Any:
        return self._adapter.condition_state(state_embedding)

    def features(self, video_ctx: Any, text_ctx: Any, state_ctx: Any) -> Any:
        return self._adapter.features(video_ctx, text_ctx, state_ctx)

    # ---- FlowBackbone ------------------------------------------------------------------------

    def encode_video(self, video: Any) -> Tensor:
        """Raw frames -> clean flow latents, resized to ``config.image_hw`` on the way in.

        The protocol signature has no size argument by design: which grid this backbone needs
        is the backbone's business (Wan: multiples of vae_stride x patch), and a dataset that
        records 120x160 frames must not have to know it (FR-09).
        """
        return self._adapter.encode_video(video, image_hw=self.config.image_hw)

    def decode_video(self, video_latents: Any) -> Tensor:
        return self._adapter.decode_video(video_latents)

    def num_video_tokens(self, video_latents: Any) -> int:
        return self._adapter.num_video_tokens(video_latents)

    def forward_flow(
        self, video_latents: Any, t: Any, text_ctx: Any, state_ctx: Any
    ) -> tuple[Tensor, Tensor]:
        """One rectified-flow pass -> ``(velocity, features)``, plus one cheap, loud assertion.

        ``features`` are captured by forward hooks on block OUTPUTS. diffusers currently
        gradient-checkpoints with ``use_reentrant=False``, which keeps those activations
        graph-connected — but the reentrant variant detaches them, and if that default ever
        flips upstream the action branch would train on constants with no shape error, no dtype
        error and a loss curve that merely looks disappointing. Checking one boolean per step
        is a trade worth making against silently wasting a GPU allocation.
        """
        velocity, features = self._adapter.forward_flow(video_latents, t, text_ctx, state_ctx)
        if self.training and not features.requires_grad and self._has_trainable_lora():
            raise RuntimeError(
                "backbone features came back detached from the autograd graph while LoRA "
                "parameters are trainable — the action branch would learn nothing. Likely "
                "cause: gradient checkpointing switched to use_reentrant=True, which detaches "
                "the block outputs the readout hooks capture"
            )
        return velocity, features

    def _has_trainable_lora(self) -> bool:
        return any(param.requires_grad for param in self.lora.values())

    # ---- checkpointing ----------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Derived geometry + config + adapter sizes; goes into run logs (AC-04). JSON-safe."""
        info: dict[str, Any] = dict(self._adapter.describe())
        info["config"] = self.config.model_dump(mode="json")
        info["lora_tensors"] = len(self.lora)
        info["lora_parameters"] = sum(int(p.numel()) for p in self.lora.values())
        info["trainable_parameters"] = sum(
            int(p.numel()) for p in self.parameters() if p.requires_grad
        )
        return info

    def save_adapter(self, directory: str | Path) -> Path:
        """Write everything a resume needs and nothing the base checkpoint already has.

        Three files: peft's own adapter directory (portable — it loads into a stock diffusers
        Wan pipeline, so a checkpoint can be watched as video without WAM), the state
        projection, and ``backbone.json`` so a checkpoint carries the geometry it was trained
        against instead of trusting whatever config is on disk months later (AC-04).
        """
        from safetensors.torch import save_file

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._adapter.save_lora(path)
        if self.state_proj is not None:
            save_file(
                {n: p.detach().cpu().contiguous() for n, p in self.state_proj.state_dict().items()},
                str(path / _STATE_PROJ_FILE),
            )
        (path / _DESCRIBE_FILE).write_text(json.dumps(self.describe(), indent=2, sort_keys=True))
        return path

    def _load_adapter_dir(self, path: Path) -> None:
        """Inverse of :meth:`save_adapter` (ranks come from the saved peft config, not ours)."""
        from safetensors.torch import load_file

        self._adapter.load_lora(path, param_dtype=self.config.lora_param_dtype)
        weights = path / _STATE_PROJ_FILE
        if not weights.exists():
            raise FileNotFoundError(
                f"{weights} is missing — the state projection is trained, so resuming without "
                "it would silently reset the whole proprioception path to a fresh init"
            )
        assert self.state_proj is not None  # built by _setup() before this is reached
        self.state_proj.load_state_dict(load_file(str(weights)))
