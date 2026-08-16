"""``--offload-text``: park the Wan umT5 tower on the CPU, from the entry points that run.

:meth:`wam.backbones.wan_i2v.WanI2VAdapter.offload` has existed and been tested since the
readout smoke test, but until now ``scripts/hf_job_wan_smoke.py`` was the only caller. Every
script anyone actually runs against a checkpoint — ``eval_t16``, ``dream``, ``train_t16_lora``,
``serve_policy``, ``rollout`` — held the umT5 weights resident on the GPU for the whole run
after using them once. This module is the shared wiring that closes that gap.

THE CHAIN. ``WanFlowBackbone`` keeps the DiT, VAE and umT5 deliberately OUT of the module tree
(``wan_flow.py`` docstring: a ``WanI2VAdapter`` is not an ``nn.Module``, so its weights never
enter ``state_dict()``). The adapter is therefore reachable only as a plain attribute, and
:func:`resolve_wan_adapter` walks the one path that exists::

    JointCheckpointPolicy .model -> JointWorldActionModel   (runtime/policies.py)
                          .backbone -> WanFlowBackbone      (training/joint.py:268)
                          .adapter -> WanI2VAdapter         (backbones/wan_flow.py:107)

Training reaches the same adapter one link in, from the ``WanFlowBackbone`` that
``build_backbone(..., load=True)`` returns.

ORDER MATTERS. ``WanFlowBackbone._apply`` (wan_flow.py:255) forwards every device move to the
held modules on purpose, so ``model.to("cuda")`` pulls the text encoder back onto the GPU. An
offload issued BEFORE the final ``.to(device)`` is silently undone. Every caller here must
offload after the model is resident — after ``load_joint_policy`` returns, or after
``JointTrainer.__init__`` (joint.py:711) has done its ``.to(self.device)``.

...AND THAT IS WHY AN OFFLOAD CANNOT LOWER THE *LOAD* PEAK. Having to run last means running
after the umT5 tower has already been on the accelerator once. On a card with room for all
three towers that is merely wasteful; on a card without it, the process is already dead — the
Wan weights are 24.18 GB (DiT 10.00 + umT5 11.36 + VAE 2.82) before a single activation, and no
batch size touches that. Measured 2026-08-17 on a 32 GB RTX 5090 sharing the card with a
12.70 GB co-tenant: ``torch.OutOfMemoryError`` inside ``WanI2VAdapter.attach``'s
``module.to(self.device)``, this process holding 18.49 GB, with ``--offload-text`` passed.
:func:`pin_text_encoder_to_cpu` is the fix and is the opposite discipline — it must run BEFORE
``load()``, and the pin then survives every later ``.to(device)`` rather than being undone by
it. Use the pin when the load has to fit; use the offload when only the steady state does.

COST. ``condition_text`` runs the umT5 forward on whichever device the tower is on
(``wan_i2v.py``: ``enc_device = self._device_of(self._text_encoder)``), so after an offload the
instruction is encoded on the CPU. That is paid once per DISTINCT instruction, not once per
call: the result is memoized under the prompt string and stored on ``self.device``, so the cache
survives the tower moving and every later hit is a GPU tensor. See
:func:`distinct_instructions` for the one caller that has to care.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

__all__ = [
    "OFFLOAD_TEXT_HELP",
    "advise_alloc_conf",
    "distinct_instructions",
    "offload_text_encoder",
    "pin_text_encoder_to_cpu",
    "resolve_wan_adapter",
]

#: Verbatim from ``scripts/hf_job_wan_smoke.py`` so there is one name and one description for
#: one thing. Callers that need to add a caveat append to it rather than rewording it.
OFFLOAD_TEXT_HELP = "move the umT5 tower to CPU after encoding (peak-VRAM relief on 24 GB cards)"

#: Walked in this order, skipping links the object does not have, so the same function serves a
#: policy (all three hops), a model (two) and a bare backbone (one).
_CHAIN_ATTRS = ("model", "backbone", "adapter")


def resolve_wan_adapter(obj: Any) -> Any:
    """Policy | model | backbone -> the ``WanI2VAdapter`` holding the frozen towers.

    Raises ``RuntimeError`` naming the chain it walked when the object does not end at
    something with an ``offload()``. Failing loudly is the point: the alternative — a
    ``--offload-text`` that quietly does nothing on a tiny-backbone checkpoint — hands the user
    back the same 32 GB peak with no clue why the flag did not help.
    """
    node = obj
    chain = [type(obj).__name__]
    for attr in _CHAIN_ATTRS:
        nxt = getattr(node, attr, None)
        if nxt is None:
            continue
        node = nxt
        chain.append(f".{attr} -> {type(node).__name__}")
    reached_an_adapter = callable(getattr(node, "offload", None))
    if not reached_an_adapter:
        raise RuntimeError(
            "--offload-text needs a Wan-backed model: no offload() at the end of "
            + "".join(chain)
            + ". Only the Wan backbone holds a separate umT5 tower to move; a tiny-backbone "
            "checkpoint has nothing to offload, so the flag would be a silent no-op."
        )
    return node


def pin_text_encoder_to_cpu(obj: Any, *, log: Callable[[str], None] | None = None) -> Any:
    """Pin the umT5 tower to the CPU so it never reaches the accelerator. Returns the adapter.

    Call BEFORE the weights load — with ``build_backbone(..., load=False)``, then this, then
    ``backbone.load()``. That ordering is the exact inverse of :func:`offload_text_encoder`'s,
    and deliberately so: an offload has to run last to survive, a pin has to run first to help.
    See the module docstring for why only the pin lowers the load peak.
    """
    adapter = resolve_wan_adapter(obj)
    adapter.pin_to_cpu("text_encoder")
    if log is not None:
        log(
            "[offload-text] umT5 pinned to CPU BEFORE load: it never reaches the accelerator, so "
            "the load peak drops by the tower's 11.36 GB rather than only the steady state."
        )
    return adapter


def offload_text_encoder(obj: Any, *, log: Callable[[str], None] | None = None) -> Any:
    """Move the umT5 tower to CPU and drop the freed blocks. Returns the adapter.

    Call only once the model is on its final device — see the module docstring on ``_apply``.
    """
    adapter = resolve_wan_adapter(obj)
    adapter.offload("text_encoder")
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if log is not None:
        log(
            "[offload-text] umT5 -> CPU. Instructions are encoded on CPU from here on, once "
            "per distinct instruction (condition_text memoizes per prompt)."
        )
    return adapter


def distinct_instructions(episode_dirs: Iterable[Path]) -> set[str]:
    """The instruction strings of these episodes, read from manifests only (no frames).

    Exists for the training entry point: with ``--offload-text`` the umT5 forward moves to the
    CPU, and it is the SIZE OF THIS SET, not the step count, that decides whether that is free
    or a per-batch stall.
    """
    from wam.data.episode import EpisodeReader

    found: set[str] = set()
    for path in episode_dirs:
        # verify_checksums=False: this is a metadata peek, not a data-integrity gate — the
        # dataset's own loader still verifies whatever it is configured to verify.
        found.add(EpisodeReader(path, verify_checksums=False).manifest.instruction)
    return found


#: Both spellings PyTorch 2.13 accepts (``PYTORCH_CUDA_ALLOC_CONF`` is the back-compat name).
_ALLOC_CONF_VARS = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")

_ALLOC_CONF_HINT = (
    "[hint] PYTORCH_CUDA_ALLOC_CONF is unset. On a card near its VRAM ceiling, run with\n"
    "         export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
    "       which is what every cluster job does (cluster/discoverer/50_train_t16.sbatch:41) "
    "and what nothing sets locally. It has to be exported before the process starts."
)


def advise_alloc_conf(device: Any, *, stream: Any = None) -> bool:
    """Print the allocator hint when running on CUDA without one set. Returns whether it did.

    DELIBERATELY A HINT, NOT A SETTING. Setting it from Python would work — PyTorch reads the
    variable lazily, in the construct-on-first-use static of
    ``CUDAAllocatorConfig::instance()`` (``torch/include/c10/cuda/CUDAAllocatorConfig.h:133``),
    so any assignment landing before the first CUDA allocation is honoured. It is not done here
    because ``expandable_segments`` changes allocator behaviour and therefore the peak-VRAM
    figures these scripts record, and a re-scored archived run would silently report numbers
    that are not comparable with the ones in ``runs/``. The cluster's mechanism — export it in
    the job script, where it is visible in the run record — is the one this repo already has.
    """
    if "cuda" not in str(device).lower():
        return False
    if any(os.environ.get(name, "").strip() for name in _ALLOC_CONF_VARS):
        return False
    print(_ALLOC_CONF_HINT, file=stream if stream is not None else sys.stderr)
    return True
