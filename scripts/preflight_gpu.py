#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Hard gate before the first GPU command on a fresh single-card box (docs/local_gpu.md).

Run this FIRST, with the box's own interpreter, before anything that costs GPU minutes.

The dependency list above is deliberately **empty**, and that is not an oversight: this script's
job is to report a broken environment, so it may only import the standard library. Launching it
through ``uv run`` would install torch/diffusers/... into a throwaway venv and the script would
then cheerfully certify an environment nobody is going to run::

    .venv/bin/python scripts/preflight_gpu.py            # correct
    uv run scripts/preflight_gpu.py                      # certifies uv's venv, not yours

What it replaces. ``docs/local_gpu.md`` §0 USED TO say (it now points here instead, and keeps the
one-liner only as the cautionary tale under "Why not the old one-liner")::

    python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
    # expect (12, 0)

That check cannot catch the failure it is aimed at. A wheel built for older architectures reports
``(12, 0)`` correctly — the capability comes from the *driver*, not from the compiled cubins — and
then dies at the first kernel launch with ``no kernel image is available for execution on the
device``. Section 3 below launches real kernels (bf16 matmul, fp32 matmul, SDPA, conv2d,
layernorm), synchronizes, and checks the outputs are finite. That is the only check here that can
fail for the right reason.

Sections
    1. environment      python, platform, venv, allocator env vars
    2. torch            version, CUDA toolkit, device, compute capability
    3. kernel launch    the section that matters — real kernels, really synchronized
    4. memory           VRAM total/free, host RAM (weights materialize in host RAM before load)
    5. dependencies     import-probe + the corrected `pip install` line for what is missing
    6. assets           optional --backbone-source / --dataset / --checkpoint
    7. VRAM budget      measured peaks per entry point, headroom on THIS card, verdict, lever

Exit code is 0 iff nothing FAILed. ``--json`` writes the same report machine-readably.

    .venv/bin/python scripts/preflight_gpu.py
    .venv/bin/python scripts/preflight_gpu.py --backbone-source /models/Wan2.2-TI2V-5B \\
        --dataset datasets/gr00t-apple-full --checkpoint runs/t16-lora-seed0/checkpoints/step-020000
    .venv/bin/python scripts/preflight_gpu.py --json preflight.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------------------------
# Measured VRAM peaks.
#
# UNITS, because this is where the arithmetic goes wrong: every number below is
# `torch.cuda.max_memory_allocated() / 1e9` as recorded by the script that produced it
# (hf_job_wan_smoke.py:424, dream.py:634, hf_job_wan_probe.py:964) — DECIMAL GB, and *allocated*,
# not reserved. A "32 GB" RTX 5090 carries 32 GiB, which is 34.36 decimal GB, so the card and the
# artifacts must be compared in decimal GB or the 5090 looks 2.4 GB smaller than it is. This file
# therefore works in decimal GB throughout and never converts.
#
# Two things an allocated peak does NOT include, both of which the card must still supply: the
# CUDA context (a few hundred MB the allocator never sees) and the caching allocator's
# reserved-but-unallocated slack. See ALLOCATOR_OVERHEAD_GB.

#: runs/smoke/183599/wan_smoke_report.json:129 — H200, 5 frames 256x448, `--offload-text` NOT set.
#: Taken AFTER reset_peak_memory_stats() (hf_job_wan_smoke.py:385), so it EXCLUDES the load
#: transient: this is the steady-state readout forward, not the worst moment of the run.
WAN_SMOKE_PEAK_GB = 24.28
#: runs/smoke/183599/wan_smoke_report.json:130 — the same forward, reserved instead of allocated.
WAN_SMOKE_RESERVED_GB = 25.18
#: runs/wan_probe/183600/wan_probe_report.json:155 — independent confirmation of the smoke figure.
WAN_PROBE_PEAK_GB = 24.65
#: runs/dream/t35-zerogpu-seed0/dream.json:172 — ZeroGPU (RTX PRO 6000, 96 GB), 16 clips/32 steps.
#: dream.py never calls reset_peak_memory_stats(), so this one INCLUDES the load transient.
DREAM_PEAK_GB = 32.47
#: runs/dream/t36-zerogpu-motion-seed0/dream.json:265 — same script, motion window. The worse of
#: the two, and therefore the one the budget table is computed against.
DREAM_MOTION_PEAK_GB = 32.54
#: runs/presentation/t16_lora_futures/scale0.report.json:66 — diffusers WanImageToVideoPipeline.
DIFFUSERS_GENERATE_PEAK_GB = 31.55
#: runs/presentation/wan_futures/faithful_hand_visible.json:18 — the same pipeline without LoRA.
DIFFUSERS_GENERATE_MIN_GB = 31.39
#: Resident floor for training. The weight terms are MEASURED — parameter counts read from the
#: safetensors headers of Wan-AI/Wan2.2-TI2V-5B-Diffusers and multiplied by the dtype each tower
#: is loaded at (wan_i2v.py:288-296): DiT bf16 10.00 + umT5 bf16 11.36 + VAE fp32 2.82 = 24.18,
#: which lands 0.42 % under the 24.28 GB measured on job 183599 without having used it. The
#: training state is read out of runs/t16-lora-seed0/checkpoints/step-020000: trainable fp32 0.33
#: + grads 0.33 + AdamW exp_avg/exp_avg_sq 0.66 = 1.32. Full derivation, with the byte counts:
#: configs/training/joint_wan_gr00t_5090.yaml's header.
#: THE VAE IS 2.82, NOT ~1.4. 1.41 GB is the bf16 figure; wan_i2v.py:289-291 hard-wires fp32 and
#: there is no flag. An earlier version of this file carried the bf16 number and therefore a
#: floor 1.8 GB too low. ACTIVATIONS ARE NOT IN THIS FLOOR — see TRAINING_BATCH2_PEAK_GB_ESTIMATE.
TRAINING_FLOOR_GB_ESTIMATE = 25.50
#: The floor plus what a real step adds at the batch size the local runbook trains at:
#: activations (CPU-profiled at the exact shapes, ~0.005 + 0.209 GB/sample), the allocator's
#: measured 1.0370 reserved/allocated ratio, and ~0.8 GB of CUDA context. ESTIMATE — the
#: extrapolation from CPU errs in both directions and the cuDNN conv3d workspace for the fp32 VAE
#: is unquantified. This is the number docs/local_gpu.md §0c prints for train_t16_lora.py, and
#: that table claims to be this one; they have to stay equal.
TRAINING_BATCH2_PEAK_GB_ESTIMATE = 27.70

#: 25.18 - 24.28 on job 183599 — the only run in this repo that recorded both numbers for one
#: forward. The allocator asked the driver for 0.90 GB more than the tensors needed.
ALLOCATOR_OVERHEAD_GB = round(WAN_SMOKE_RESERVED_GB - WAN_SMOKE_PEAK_GB, 2)

#: A run whose measured *allocated* peak leaves less than this is called TIGHT rather than FITS.
#: Basis: ALLOCATOR_OVERHEAD_GB (0.90, measured above) + the CUDA context, which allocated-GB never
#: counts and which is a few hundred MB, + room for fragmentation. 2.0 GB is the smallest margin
#: that covers those three without claiming a precision nobody measured.
TIGHT_MARGIN_GB = 2.0

#: 32 GiB expressed in the decimal GB this file works in. Used only when there is no CUDA device
#: to ask (a CPU-only Mac), and labelled as an assumption wherever it is printed.
NOMINAL_RTX5090_TOTAL_GB = 34.36

#: The Wan weights are materialized in HOST RAM during load wherever accelerate's device_map is
#: not reachable (see DEVICE_MAP_FLAG). Below this floor the load swaps or is OOM-killed, and the
#: user never sees a CUDA error at all — which is the confusing way for this to fail.
HOST_RAM_FLOOR_GB = 32.0

MIN_PYTHON = (3, 10)  # pyproject.toml:5, requires-python
BLACKWELL_CAPABILITY = (12, 0)  # sm_120
MIN_CUDA_FOR_BLACKWELL = (12, 8)

CU128_FIX = (
    "pip install --index-url https://download.pytorch.org/whl/cu128 "
    "--force-reinstall torch torchvision"
)

STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"


# ---------------------------------------------------------------------------------------------
# Reporting — the [PASS]/[FAIL] idiom of scripts/hf_job_wan_smoke.py:109, plus WARN and a fix line.


@dataclass
class Check:
    section: str
    name: str
    status: str
    detail: str = ""
    fix: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "fix": self.fix,
        }


class Report:
    """Collects check results. Any FAIL makes the preflight exit non-zero; WARN does not."""

    def __init__(self, echo: bool = True) -> None:
        self.checks: list[Check] = []
        self.info: dict[str, Any] = {}
        self.echo = echo
        self._section: str = ""

    def section(self, title: str) -> None:
        self._section = title
        if self.echo:
            print(f"\n=== {title} ===", flush=True)

    def record(self, name: str, status: str, detail: Any = "", fix: str = "") -> str:
        entry = Check(self._section, name, status, str(detail), fix)
        self.checks.append(entry)
        if self.echo:
            print(f"[{status}] {name}: {detail}", flush=True)
            if fix and status != STATUS_PASS:
                for i, line in enumerate(fix.splitlines()):
                    prefix = "        fix: " if i == 0 else "             "
                    print(prefix + line, flush=True)
        return status

    def check(self, name: str, ok: bool, detail: Any = "", fix: str = "") -> bool:
        """PASS or FAIL — a FAIL blocks the run."""
        self.record(name, STATUS_PASS if ok else STATUS_FAIL, detail, fix)
        return bool(ok)

    def soft(self, name: str, ok: bool, detail: Any = "", fix: str = "") -> bool:
        """PASS or WARN — worth saying, not worth blocking on."""
        self.record(name, STATUS_PASS if ok else STATUS_WARN, detail, fix)
        return bool(ok)

    @property
    def failed(self) -> list[str]:
        return [c.name for c in self.checks if c.status == STATUS_FAIL]

    @property
    def warned(self) -> list[str]:
        return [c.name for c in self.checks if c.status == STATUS_WARN]

    def statuses(self) -> list[str]:
        return [c.status for c in self.checks]


def exit_code(statuses: Iterable[str]) -> int:
    """0 iff nothing FAILed. WARN never blocks — a warning that blocks stops being read."""
    return 1 if any(s == STATUS_FAIL for s in statuses) else 0


# ---------------------------------------------------------------------------------------------
# 1-2. environment + torch


def parse_version_tuple(text: str | None) -> tuple[int, ...] | None:
    """'12.8' -> (12, 8); '2.9.0+cu128' -> (2, 9, 0, 128). None when there is nothing to parse."""
    if not text:
        return None
    digits = re.findall(r"\d+", str(text))
    return tuple(int(d) for d in digits) if digits else None


def cuda_toolkit_verdict(
    capability: tuple[int, int] | None, torch_cuda_version: str | None
) -> tuple[str, str, str]:
    """(status, detail, fix) for 'is this wheel built for this card?'.

    Deliberately a WARN and never a FAIL: this is version arithmetic, and section 3 launches
    actual kernels. When the two disagree, the kernels are right. Saying FAIL here would let a
    working box be blocked by a string comparison, which is the mistake this repo keeps writing
    postmortems about.
    """
    if capability is None:
        return STATUS_WARN, "no CUDA device — capability unknown", ""
    toolkit = parse_version_tuple(torch_cuda_version)
    cap_text = f"sm_{capability[0]}{capability[1]}"
    if toolkit is None:
        return (
            STATUS_WARN,
            f"{cap_text} card, but torch.version.cuda is None (CPU-only wheel)",
            CU128_FIX,
        )
    if capability >= BLACKWELL_CAPABILITY and toolkit[:2] < MIN_CUDA_FOR_BLACKWELL:
        return (
            STATUS_WARN,
            (
                f"{cap_text} (Blackwell) against a CUDA {torch_cuda_version} wheel — expect "
                "'no kernel image is available for execution on the device' at the first "
                "launch; section 3 is the verdict, this is only the prediction"
            ),
            CU128_FIX,
        )
    return STATUS_PASS, f"{cap_text} against a CUDA {torch_cuda_version} wheel", ""


def in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def section_environment(report: Report) -> None:
    report.section("1. environment")
    version = sys.version_info[:3]
    report.check(
        "python.version",
        version[:2] >= MIN_PYTHON,
        f"{platform.python_version()} (pyproject requires >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})",
        fix=f"install python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]} and rebuild the venv",
    )
    report.record("platform", STATUS_PASS, f"{platform.platform()} / {platform.machine()}")
    report.soft(
        "python.venv",
        in_virtualenv(),
        f"sys.prefix={sys.prefix}"
        + ("" if in_virtualenv() else " — this looks like a system interpreter"),
        fix="python3 -m venv .venv && . .venv/bin/activate",
    )
    report.record("python.executable", STATUS_PASS, sys.executable)

    alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    report.soft(
        "env.PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments" in alloc,
        alloc or "(unset)",
        fix="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # every cluster sbatch "
        "sets this (cluster/discoverer/50_train_t16.sbatch:41); nothing sets it locally, and "
        "on a card this close to its ceiling fragmentation is the difference between fitting "
        "and not",
    )
    report.info["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
        "venv": in_virtualenv(),
        "PYTORCH_CUDA_ALLOC_CONF": alloc or None,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }


def import_torch() -> tuple[Any | None, BaseException | None]:
    try:
        # Imported here rather than at module level: the whole point of this script is that it
        # has to run, and report, in an environment where this may fail.
        import torch

        return torch, None
    except BaseException as err:  # noqa: BLE001 — a broken CUDA install can raise anything
        return None, err


def section_torch(report: Report, torch_mod: Any | None, err: BaseException | None) -> dict:
    report.section("2. torch")
    if torch_mod is None:
        report.check("torch.import", False, f"{type(err).__name__}: {err}", fix=CU128_FIX)
        report.info["torch"] = {"importable": False, "error": f"{type(err).__name__}: {err}"}
        return report.info["torch"]

    available = bool(torch_mod.cuda.is_available())
    facts: dict[str, Any] = {
        "importable": True,
        "version": torch_mod.__version__,
        "cuda_build": torch_mod.version.cuda,
        "cuda_available": available,
        "device_name": None,
        "capability": None,
        "device_count": torch_mod.cuda.device_count() if available else 0,
    }
    report.check(
        "torch.import", True, f"{torch_mod.__version__} (built for CUDA {torch_mod.version.cuda})"
    )
    report.soft(
        "torch.cuda.is_available",
        available,
        available,
        fix="no CUDA device visible — on the 5090 box check the driver (nvidia-smi) and that "
        "torch is not a CPU-only wheel; on a Mac this WARN is expected and the GPU sections "
        "below are skipped",
    )
    capability: tuple[int, int] | None = None
    if available:
        facts["device_name"] = torch_mod.cuda.get_device_name(0)
        capability = tuple(torch_mod.cuda.get_device_capability(0))  # type: ignore[assignment]
        facts["capability"] = list(capability) if capability else None
        report.record(
            "torch.device",
            STATUS_PASS,
            f"{facts['device_name']} (sm_{capability[0]}{capability[1]}, "
            f"{facts['device_count']} visible)",
        )
    status, detail, fix = cuda_toolkit_verdict(capability, torch_mod.version.cuda)
    report.record("torch.wheel_matches_card", status, detail, fix)
    report.info["torch"] = facts
    return facts


# ---------------------------------------------------------------------------------------------
# 3. kernel launch proof


KERNEL_IMAGE_MARKERS = (
    "no kernel image is available",
    "cuda error: no kernel image",
    "no kernel image is available for execution on the device",
)


def is_kernel_image_error(message: str) -> bool:
    """True for the failure docs/local_gpu.md §0 is aimed at and cannot actually detect."""
    lowered = str(message).lower()
    return any(marker in lowered for marker in KERNEL_IMAGE_MARKERS)


def kernel_failure_fix(message: str, capability: tuple[int, int] | None) -> str:
    if is_kernel_image_error(message):
        cap = f"sm_{capability[0]}{capability[1]}" if capability else "this card"
        return (
            f"this wheel has no cubins for {cap} — the capability printed fine and the kernel "
            f"still had nowhere to run, which is exactly what the get_device_capability() "
            f"one-liner cannot see (docs/local_gpu.md §0, 'Why not the old one-liner').\n"
            f"{CU128_FIX}"
        )
    return (
        "not the kernel-image failure — read the message above before reinstalling anything; "
        "check `nvidia-smi` and the driver/toolkit pairing"
    )


def kernel_probes() -> tuple[tuple[str, Callable[[Any, str], Any]], ...]:
    """The kernels the WAM entry points actually launch, smallest possible instance of each.

    SDPA is here because it *is* the attention path: nothing in this repo sets
    ``attn_implementation``, so diffusers falls through to
    ``torch.nn.functional.scaled_dot_product_attention``. conv2d is the VAE's path and layernorm
    is in every DiT block. None of this needs the Wan weights and all of it needs real cubins.
    """

    def bf16_matmul(torch_mod: Any, device: str) -> Any:
        a = torch_mod.randn(256, 256, device=device, dtype=torch_mod.bfloat16)
        return a @ a

    def fp32_matmul(torch_mod: Any, device: str) -> Any:
        a = torch_mod.randn(256, 256, device=device, dtype=torch_mod.float32)
        return a @ a

    def sdpa(torch_mod: Any, device: str) -> Any:
        shape = (1, 4, 128, 64)
        q = torch_mod.randn(*shape, device=device, dtype=torch_mod.bfloat16)
        k = torch_mod.randn(*shape, device=device, dtype=torch_mod.bfloat16)
        v = torch_mod.randn(*shape, device=device, dtype=torch_mod.bfloat16)
        return torch_mod.nn.functional.scaled_dot_product_attention(q, k, v)

    def conv2d(torch_mod: Any, device: str) -> Any:
        x = torch_mod.randn(1, 4, 32, 32, device=device, dtype=torch_mod.float32)
        w = torch_mod.randn(8, 4, 3, 3, device=device, dtype=torch_mod.float32)
        return torch_mod.nn.functional.conv2d(x, w, padding=1)

    def layernorm(torch_mod: Any, device: str) -> Any:
        x = torch_mod.randn(2, 16, 64, device=device, dtype=torch_mod.bfloat16)
        return torch_mod.nn.functional.layer_norm(x, (64,))

    return (
        ("bf16_matmul", bf16_matmul),
        ("fp32_matmul", fp32_matmul),
        ("sdpa", sdpa),
        ("conv2d", conv2d),
        ("layernorm", layernorm),
    )


def section_kernels(report: Report, torch_mod: Any | None, facts: dict) -> None:
    report.section("3. kernel launch proof")
    if torch_mod is None or not facts.get("cuda_available"):
        report.soft(
            "kernels.skipped",
            False,
            "no CUDA device — nothing was launched, so nothing about this box's GPU is proven "
            "by this run",
            fix="run this again on the 5090 box; on a Mac there is nothing to prove here",
        )
        report.info["kernels"] = {"ran": False, "reason": "no cuda device"}
        return

    capability = tuple(facts["capability"]) if facts.get("capability") else None
    results: dict[str, Any] = {"ran": True, "ops": {}}
    for name, fn in kernel_probes():
        try:
            out = fn(torch_mod, "cuda")
            torch_mod.cuda.synchronize()  # the launch is async; without this we prove nothing
            finite = bool(torch_mod.isfinite(out).all())
            spread = float(out.float().std())
        except RuntimeError as err:
            message = str(err)
            results["ops"][name] = {"ok": False, "error": message}
            report.check(
                f"kernel.{name}",
                False,
                message.splitlines()[0][:200],
                fix=kernel_failure_fix(message, capability),  # type: ignore[arg-type]
            )
            continue
        except BaseException as err:  # noqa: BLE001 — a bad wheel is not limited to RuntimeError
            results["ops"][name] = {"ok": False, "error": f"{type(err).__name__}: {err}"}
            report.check(f"kernel.{name}", False, f"{type(err).__name__}: {err}", fix=CU128_FIX)
            continue
        ok = finite and spread > 0.0
        results["ops"][name] = {"ok": ok, "finite": finite, "std": spread}
        report.check(
            f"kernel.{name}",
            ok,
            f"launched, synchronized, finite={finite}, std={spread:.4f}",
            fix="the kernel ran and returned non-finite or constant output — that is a numerics "
            "problem, not an install problem; do not spend GPU hours on top of it",
        )
    report.info["kernels"] = results


# ---------------------------------------------------------------------------------------------
# 4. memory


def parse_meminfo_total_gb(text: str) -> float | None:
    """MemTotal out of /proc/meminfo, in decimal GB. None when the field is not there."""
    match = re.search(r"^MemTotal:\s+(\d+)\s*kB", text, re.MULTILINE)
    if not match:
        return None
    return int(match.group(1)) * 1024 / 1e9


def host_ram_gb() -> tuple[float | None, str]:
    """(total host RAM in decimal GB, where the number came from)."""
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        try:
            total = parse_meminfo_total_gb(meminfo.read_text())
        except OSError:
            total = None
        if total is not None:
            return total, "/proc/meminfo MemTotal"
    try:  # macOS and every other POSIX box without /proc
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return pages * page_size / 1e9, "os.sysconf(SC_PHYS_PAGES * SC_PAGE_SIZE)"
    except (ValueError, OSError, AttributeError):
        pass
    return None, "unavailable"


def section_memory(report: Report, torch_mod: Any | None, facts: dict) -> dict:
    report.section("4. memory")
    memory: dict[str, Any] = {"vram_total_gb": None, "vram_free_gb": None, "host_ram_gb": None}

    if torch_mod is not None and facts.get("cuda_available"):
        total = torch_mod.cuda.get_device_properties(0).total_memory / 1e9
        try:
            free_bytes, total_bytes = torch_mod.cuda.mem_get_info(0)
            free = free_bytes / 1e9
            total = total_bytes / 1e9
        except (RuntimeError, AttributeError):
            free = None
        memory["vram_total_gb"] = round(total, 2)
        memory["vram_free_gb"] = round(free, 2) if free is not None else None
        report.record(
            "vram.total",
            STATUS_PASS,
            f"{total:.2f} GB decimal (= {total / 1.073741824:.1f} GiB, which is how the card is "
            f"marketed) on {facts.get('device_name')}",
        )
        if free is not None:
            report.soft(
                "vram.free",
                free > total * 0.9,
                f"{free:.2f} GB of {total:.2f} GB free",
                fix="something else is holding VRAM — the budget table below assumes an empty "
                "card, so close it before believing any verdict there",
            )
    else:
        report.soft(
            "vram.total",
            False,
            "no CUDA device — the budget table below uses a nominal 32 GiB "
            f"({NOMINAL_RTX5090_TOTAL_GB} GB decimal) card, which is an ASSUMPTION",
        )

    ram, source = host_ram_gb()
    memory["host_ram_gb"] = round(ram, 2) if ram is not None else None
    memory["host_ram_source"] = source
    device_map = scripts_exposing_flag(REPO_ROOT / "scripts", DEVICE_MAP_FLAG)
    without = [n for n, ok in device_map.items() if not ok]
    memory["device_map_lever"] = device_map
    if ram is None:
        report.soft("host.ram", False, f"could not determine host RAM ({source})")
    else:
        report.check(
            "host.ram",
            ram >= HOST_RAM_FLOOR_GB,
            f"{ram:.1f} GB via {source} (floor {HOST_RAM_FLOOR_GB:.0f} GB)",
            fix="the ~22-24 GB of Wan weights are materialized in HOST RAM before they ever "
            "reach the card, because accelerate's device_map — which would stream the shards "
            "straight to the GPU — is not exposed by "
            + (", ".join(without) if without else "any entry point")
            + ". Under this floor the load swaps or gets OOM-killed and you never see a CUDA "
            "error at all. Either add RAM, or wire --device-map into the script you run.",
        )
    report.soft(
        "host.device_map_lever",
        not without,
        f"{DEVICE_MAP_FLAG} reachable from {len(device_map) - len(without)}/{len(device_map)} "
        "entry points" + (f" — missing on {', '.join(without)}" if without else ""),
        fix="--device-map is the lever that removes the host-RAM floor entirely; "
        "WanI2VAdapter already takes device_map (hf_job_wan_smoke.py:70 passes it through).",
    )
    report.info["memory"] = memory
    return memory


# ---------------------------------------------------------------------------------------------
# 5. dependency completeness
#
# The bug this section exists to detect was shipped and live: docs/local_gpu.md §0 said
#   pip install -e '.[dev]'
# and dev is ["pytest", "ruff==0.16.0", "peft>=0.14"] (pyproject.toml:62). No diffusers, no
# transformers, no accelerate, no safetensors, no pyarrow, no opencv, no av, no websockets. Since
# src/wam/data/episode.py:37 imports pyarrow at MODULE level, every GPU command in that runbook
# died on import before it ever touched the GPU. The runbook now installs the `local` extra and
# keeps the old line only as the cautionary note under §0. This section stays: it is what makes
# that claim checkable on the box rather than believed, and it is the only thing that catches a
# HALF-installed environment.
#
# The extra -> package mapping is READ FROM pyproject.toml at runtime, never hardcoded, so the
# fix line this prints cannot drift away from what the project actually declares.


@dataclass(frozen=True)
class Requirement:
    """One import the entry points need, and what pip calls it."""

    module: str
    distribution: str
    needed_by: str


REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement("numpy", "numpy", "every module"),
    Requirement("pydantic", "pydantic", "wam.interfaces.schema (module level)"),
    Requirement("yaml", "pyyaml", "config loading (module level)"),
    Requirement("typing_extensions", "typing_extensions", "wam.data.episode:40 (module level)"),
    Requirement("torch", "torch", "every GPU entry point"),
    Requirement("safetensors", "safetensors", "checkpoint + LoRA adapter I/O"),
    Requirement("pyarrow", "pyarrow", "wam.data.episode:37 (module level — nothing runs without)"),
    Requirement("cv2", "opencv-python", "wam.data.episode frame decode (lazy import)"),
    Requirement("av", "av", "wam.data.episode video read/write (lazy import)"),
    Requirement("diffusers", "diffusers", "wam.backbones.wan_i2v — the Wan DiT + VAE"),
    Requirement("transformers", "transformers", "the umT5 text tower"),
    Requirement("accelerate", "accelerate", "diffusers weight loading / device_map"),
    Requirement("peft", "peft", "the T-16 LoRA adapters"),
    Requirement("sentencepiece", "sentencepiece", "umT5 tokenizer"),
    Requirement("ftfy", "ftfy", "umT5 prompt cleanup"),
    Requirement("websockets", "websockets", "scripts/serve_policy.py + wam.runtime.server"),
    Requirement("pytest", "pytest", "docs/local_gpu.md §0 repo health check"),
    Requirement("ruff", "ruff", "docs/local_gpu.md §0 repo health check"),
)


def canonical_dist(name: str) -> str:
    """PEP 503 normalization: 'opencv_python' and 'opencv-python' are one package."""
    return re.sub(r"[-_.]+", "-", str(name)).strip().lower()


def requirement_dist(requirement: str) -> str:
    """'peft>=0.14' -> 'peft'; 'websockets>=12' -> 'websockets'; 'x[y]' -> 'x'."""
    head = re.split(r"[<>=!~;\[\s]", str(requirement).strip(), maxsplit=1)[0]
    return canonical_dist(head)


def _extras_via_regex(text: str) -> dict[str, tuple[str, ...]]:
    """tomllib-free fallback for python 3.10 (pyproject.toml:5 allows it; tomllib is 3.11+).

    A line-based state machine over the two tables that matter. It handles both `k = ["a", "b"]`
    and the multi-line form pyproject.toml actually uses, and it strips comments first so the
    prose in this project's `wan = [...]` block cannot smuggle a package name in.
    """
    out: dict[str, list[str]] = {}
    section: str | None = None
    collecting: str | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if collecting is not None:
            body, closed = (line.split("]", 1)[0], True) if "]" in line else (line, False)
            out[collecting].extend(re.findall(r"[\"']([^\"']+)[\"']", body))
            if closed:
                collecting = None
            continue
        header = re.match(r"^\[([^\]]+)\]", line)
        if header:
            section = header.group(1)
            continue
        if section not in ("project", "project.optional-dependencies"):
            continue
        match = re.match(r"^([A-Za-z0-9_.\-]+)\s*=\s*\[(.*)$", line)
        if not match:
            continue
        key, rest = match.group(1), match.group(2)
        if section == "project" and key != "dependencies":
            continue
        name = "(core)" if section == "project" else key
        out[name] = []
        body, closed = (rest.split("]", 1)[0], True) if "]" in rest else (rest, False)
        out[name].extend(re.findall(r"[\"']([^\"']+)[\"']", body))
        if not closed:
            collecting = name
    # An EMPTY extra is kept, because the tomllib branch keeps it and these two must agree on
    # python 3.10 and 3.11 alike. `isaac = []` is a real one: it exists to name a backend whose
    # dependency must NOT be resolved by this venv. Only an empty `(core)` is dropped, matching
    # the `if core:` in parse_pyproject_extras.
    return {
        k: tuple(requirement_dist(i) for i in v)
        for k, v in out.items()
        if v or k != "(core)"
    }


def parse_pyproject_extras(text: str) -> dict[str, tuple[str, ...]]:
    """extra name -> the distributions it installs. '(core)' is [project].dependencies.

    Order of the returned dict follows pyproject's declaration order, which is what makes the
    generated pip line stable across runs.
    """
    try:
        import tomllib  # 3.11+; _extras_via_regex covers the 3.10 that pyproject.toml:5 allows
    except ModuleNotFoundError:
        return _extras_via_regex(text)
    data = tomllib.loads(text)
    project = data.get("project", {})
    out: dict[str, tuple[str, ...]] = {}
    core = project.get("dependencies") or []
    if core:
        out["(core)"] = tuple(requirement_dist(r) for r in core)
    for extra, reqs in (project.get("optional-dependencies") or {}).items():
        out[extra] = tuple(requirement_dist(r) for r in reqs)
    return out


def extras_providing(distribution: str, extras: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Which extras install this distribution (declaration order, '(core)' included)."""
    target = canonical_dist(distribution)
    return tuple(name for name, dists in extras.items() if target in dists)


def _find_spec_probe(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except BaseException:  # noqa: BLE001 — a half-installed package can raise from its __init__
        return False


def _import_probe(module: str) -> bool:
    try:
        __import__(module)
        return True
    except BaseException:  # noqa: BLE001
        return False


def probe_requirements(
    requirements: Sequence[Requirement],
    extras: dict[str, tuple[str, ...]],
    probe: Callable[[str], bool] = _find_spec_probe,
) -> list[dict[str, Any]]:
    """One row per requirement: present?, and which extras would install it."""
    rows: list[dict[str, Any]] = []
    for req in requirements:
        rows.append(
            {
                "module": req.module,
                "distribution": req.distribution,
                "needed_by": req.needed_by,
                "present": bool(probe(req.module)),
                "extras": list(extras_providing(req.distribution, extras)),
            }
        )
    return rows


def cover_extras(
    missing: Sequence[str], extras: dict[str, tuple[str, ...]]
) -> tuple[list[str], list[str]]:
    """Greedy minimal set of extras covering the missing distributions.

    Returns (extras to install, distributions no extra provides). '(core)' is excluded from the
    cover: a bare `pip install -e .` already brings it, so naming it in a bracket list is wrong.
    Ties break on declaration order, so the printed command is deterministic.
    """
    order = {name: i for i, name in enumerate(extras)}
    wanted = {canonical_dist(d) for d in missing}
    candidates = {name: set(dists) for name, dists in extras.items() if name != "(core)"}
    chosen: list[str] = []
    while wanted:
        remaining = [n for n in candidates if n not in chosen]
        # Most-covering wins; ties break on pyproject declaration order, so the command is stable.
        best = max(remaining, key=lambda n: (len(wanted & candidates[n]), -order[n]), default=None)
        if best is None or not (wanted & candidates[best]):
            break
        chosen.append(best)
        wanted -= candidates[best]
    chosen.sort(key=lambda n: order[n])
    return chosen, sorted(wanted)


def pip_command(
    missing: Sequence[str], extras: dict[str, tuple[str, ...]], torch_missing: bool = False
) -> str:
    """The corrected install line for whatever is actually missing."""
    if not missing:
        return ""
    chosen, uncovered = cover_extras(missing, extras)
    lines: list[str] = []
    if torch_missing:
        lines.append(f"{CU128_FIX}   # FIRST — an extra would pull a default-index torch")
    if chosen:
        lines.append(f"pip install -e '.[{','.join(chosen)}]'")
    else:
        lines.append("pip install -e .")
    if uncovered:
        lines.append(f"pip install {' '.join(uncovered)}   # declared by no extra in pyproject.toml")
    return "\n".join(lines)


def section_dependencies(report: Report, deep: bool) -> dict:
    report.section("5. python dependencies")
    pyproject = REPO_ROOT / "pyproject.toml"
    try:
        extras = parse_pyproject_extras(pyproject.read_text())
    except OSError as err:
        report.check(
            "deps.pyproject",
            False,
            f"cannot read {pyproject}: {err}",
            fix="run this from a checkout of the repo",
        )
        report.info["dependencies"] = {"error": str(err)}
        return {}
    report.record(
        "deps.pyproject",
        STATUS_PASS,
        f"{len(extras)} groups read from {pyproject.relative_to(REPO_ROOT)}: {', '.join(extras)}",
    )

    probe = _import_probe if deep else _find_spec_probe
    rows = probe_requirements(REQUIREMENTS, extras, probe=probe)
    missing = [r for r in rows if not r["present"]]
    for row in rows:
        if row["present"]:
            continue
        provided = ", ".join(row["extras"]) or "no extra"
        report.record(
            f"deps.{row['module']}",
            STATUS_FAIL,
            f"MISSING ({row['distribution']}, provided by: {provided}) — needed by "
            f"{row['needed_by']}",
        )
    torch_missing = any(r["distribution"] == "torch" for r in missing)
    command = pip_command([r["distribution"] for r in missing], extras, torch_missing)
    report.check(
        "deps.complete",
        not missing,
        f"{len(rows) - len(missing)}/{len(rows)} importable"
        + (f" — missing {', '.join(r['module'] for r in missing)}" if missing else ""),
        fix=(
            "the minimal cover for what is actually missing, computed from pyproject.toml's own "
            "extras table (docs/local_gpu.md §0 installs the wider `local` bracket, which is a "
            "superset and also correct — `.[dev]` is only [pytest, ruff, peft] and is the "
            "install this check exists to catch):\n" + command
        )
        if missing
        else "",
    )
    if not deep:
        report.soft(
            "deps.probe_depth",
            True,
            "probed with importlib.util.find_spec — present but broken installs (the classic "
            "cv2 without libGL.so.1) still pass here",
            fix="re-run with --deep-import to actually import each one (slower, conclusive)",
        )
    report.info["dependencies"] = {
        "extras": {k: list(v) for k, v in extras.items()},
        "probe": "import" if deep else "find_spec",
        "packages": rows,
        "missing": [r["distribution"] for r in missing],
        "pip_command": command,
    }
    return report.info["dependencies"]


# ---------------------------------------------------------------------------------------------
# 6. assets


BACKBONE_SUBDIRS = ("transformer", "vae", "text_encoder")


def directory_size_bytes(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def inspect_backbone_source(path: Path) -> dict[str, Any]:
    """A Wan snapshot dir is only usable if the three towers WanI2VAdapter loads are all there."""
    out: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_dir(),
        "missing": [],
        "size_bytes": 0,
        "size_gb": 0.0,
    }
    if not out["exists"]:
        out["missing"] = list(BACKBONE_SUBDIRS)
        return out
    out["missing"] = [d for d in BACKBONE_SUBDIRS if not (path / d).is_dir()]
    out["size_bytes"] = directory_size_bytes(path)
    out["size_gb"] = round(out["size_bytes"] / 1e9, 2)
    return out


def inspect_dataset(path: Path) -> dict[str, Any]:
    """Episode dirs are the ones carrying a manifest.json (wam.data.episode's layout)."""
    out: dict[str, Any] = {"path": str(path), "exists": path.is_dir(), "episodes": 0}
    if not out["exists"]:
        return out
    out["episodes"] = sum(
        1 for child in sorted(path.iterdir()) if child.is_dir() and (child / "manifest.json").is_file()
    )
    return out


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    """A train_t16_lora checkpoint dir holds model.safetensors (+ trainer_state.pt to resume)."""
    out: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "weights": None,
        "size_gb": 0.0,
    }
    if not out["exists"]:
        return out
    if path.is_file():
        out["weights"] = path.name
        out["size_gb"] = round(path.stat().st_size / 1e9, 3)
        return out
    for candidate in ("model.safetensors", "adapter_model.safetensors", "model.pt"):
        if (path / candidate).is_file():
            out["weights"] = candidate
            out["size_gb"] = round((path / candidate).stat().st_size / 1e9, 3)
            break
    out["resumable"] = (path / "trainer_state.pt").is_file()
    return out


def section_assets(report: Report, args: argparse.Namespace) -> None:
    report.section("6. assets")
    assets: dict[str, Any] = {}
    if not any((args.backbone_source, args.dataset, args.checkpoint)):
        report.record(
            "assets.skipped",
            STATUS_PASS,
            "no --backbone-source / --dataset / --checkpoint given — nothing checked",
        )
        report.info["assets"] = assets
        return

    if args.backbone_source:
        found = inspect_backbone_source(Path(args.backbone_source))
        assets["backbone_source"] = found
        report.check(
            "assets.backbone_source",
            found["exists"] and not found["missing"],
            f"{found['path']}: "
            + (
                f"{found['size_gb']} GB, {'/'.join(BACKBONE_SUBDIRS)} present"
                if found["exists"] and not found["missing"]
                else f"missing {', '.join(found['missing']) or 'directory'}"
            ),
            fix="--backbone-source must point at a Wan2.2-TI2V-5B snapshot with transformer/, "
            "vae/ and text_encoder/ subdirs. It is NOT optional for a Discoverer+ checkpoint: "
            "the weight path is kept out of the config so config_hash matches across machines "
            "(docs/local_gpu.md §2).",
        )
    if args.dataset:
        found = inspect_dataset(Path(args.dataset))
        assets["dataset"] = found
        report.check(
            "assets.dataset",
            found["exists"] and found["episodes"] > 0,
            f"{found['path']}: {found['episodes']} episode dirs (manifest.json)",
            fix="--dataset must point at a WAM dataset root whose children are episode dirs",
        )
    if args.checkpoint:
        found = inspect_checkpoint(Path(args.checkpoint))
        assets["checkpoint"] = found
        report.check(
            "assets.checkpoint",
            found["exists"] and found["weights"] is not None,
            f"{found['path']}: "
            + (f"{found['weights']} ({found['size_gb']} GB)" if found["weights"] else "no weights"),
            fix="point --checkpoint at a step dir holding model.safetensors, e.g. "
            "runs/t16-lora-seed0/checkpoints/step-020000",
        )
    report.info["assets"] = assets


# ---------------------------------------------------------------------------------------------
# 7. VRAM budget table


TRAINING_PROVENANCE = (
    "ESTIMATE, NOT A MEASUREMENT — no training run in runs/ records peak VRAM. Weight terms are "
    f"measured (safetensors headers x loaded dtype: DiT 10.00 + umT5 11.36 + VAE fp32 2.82 = "
    f"24.18) and so is the training state read out of "
    f"runs/t16-lora-seed0/checkpoints/step-020000 (0.33 + 0.33 + 0.66 = 1.32), giving a resident "
    f"floor of {TRAINING_FLOOR_GB_ESTIMATE:.2f} GB. The {TRAINING_BATCH2_PEAK_GB_ESTIMATE:.2f} GB "
    "here adds the ESTIMATED terms on top: activations at batch_size 2 (CPU-profiled at the exact "
    "shapes, then extrapolated to CUDA), the allocator's 1.0370 reserved/allocated ratio, and "
    "~0.8 GB of CUDA context. The cuDNN conv3d workspace for the fp32 VAE is the largest "
    "unquantified term. Derivation with byte counts: "
    "configs/training/joint_wan_gr00t_5090.yaml's header. Throughput reference: "
    "runs/_slurm_logs/t16.183601.out:5 records batch=8 on an H200, and the step timestamps "
    "in that log give ~0.42 s/step over steps 10->70."
)

EVAL_PROVENANCE = (
    "INFERRED, NOT MEASURED — no eval/rollout/serve artifact in runs/ carries a peak_vram_gb "
    "field. Same three resident towers as the smoke job through the same adapter, at a smaller "
    "geometry (T-16 is 9 frames 128x160 = 60 tokens against the smoke's 5 frames 256x448), so "
    "the smoke's steady-state 24.28 GB is carried across. This is the STEADY STATE only — the "
    "load transient on this path has never been measured either. UNVERIFIED."
)

#: The biggest single lever in the repo is WanI2VAdapter.offload("text_encoder") (wan_i2v.py:397,
#: tested at tests/test_wan_i2v.py:375) — it drops the ~11 GB umT5 tower after condition_text.
#: Whether a given entry point can REACH it is a property of the working tree, not a fact to
#: hardcode: it was true of only hf_job_wan_smoke.py when this file was written, and it is being
#: wired into the others. So it is detected at runtime rather than asserted here.
OFFLOAD_FLAG = "--offload-text"
#: The other lever with the same property: --device-map streams shards to the GPU instead of
#: materializing the checkpoint in host RAM, which is what HOST_RAM_FLOOR_GB is a floor for.
DEVICE_MAP_FLAG = "--device-map"
OFFLOAD_ENTRY_POINTS = (
    "hf_job_wan_smoke.py",
    "eval_t16.py",
    "dream.py",
    "rollout.py",
    "serve_policy.py",
    "train_t16_lora.py",
)

#: Printed as its own check, because it applies across rows and no per-row number can express it.
LOAD_TRANSIENT_WARNING = (
    "Two of the peaks above were recorded after reset_peak_memory_stats() and therefore exclude "
    "the moment the weights land on the card. The one path that did capture a load transient "
    f"(dream.py, no reset) peaked at {DREAM_MOTION_PEAK_GB} GB where its steady state is much "
    "lower, so on a 32 GiB card the load itself — not the forward — may be the moment that "
    "OOMs, and nothing in runs/ measures it for eval_t16/rollout/serve_policy. Measure it first: "
    "run hf_job_wan_smoke.py on this box and watch nvidia-smi during load."
)


def scripts_exposing_flag(
    scripts_dir: Path, flag: str, names: Sequence[str] = OFFLOAD_ENTRY_POINTS
) -> dict[str, bool]:
    """Which entry points expose a given CLI flag today.

    A file that cannot be read counts as False. Reporting a lever as available when the script
    was not even readable would be the one wrong direction to fail in.
    """
    found: dict[str, bool] = {}
    for name in names:
        try:
            found[name] = flag in (scripts_dir / name).read_text()
        except OSError:
            found[name] = False
    return found


@dataclass(frozen=True)
class BudgetEntry:
    entry_point: str
    peak_gb: float
    measured: bool
    provenance: str
    lever: str
    caveat: str = ""


def budget_entries() -> tuple[BudgetEntry, ...]:
    """The five things the user will actually launch, worst measured peak first."""
    return (
        BudgetEntry(
            entry_point="scripts/dream.py",
            peak_gb=DREAM_MOTION_PEAK_GB,
            measured=True,
            provenance=(
                f"runs/dream/t36-zerogpu-motion-seed0/dream.json:265 = {DREAM_MOTION_PEAK_GB} GB; "
                f"runs/dream/t35-zerogpu-seed0/dream.json:172 = {DREAM_PEAK_GB} GB"
            ),
            caveat=(
                "measured on a 96 GB ZeroGPU card, and dream.py never calls "
                "reset_peak_memory_stats(), so this INCLUDES the load transient — treat it as a "
                "floor, not a ceiling"
            ),
            lever=(
                "fewer clips (--episodes x --windows-per-episode; the archived run recorded "
                "clips: 16), fewer Euler --steps, fewer --num-frames; or drop the ~11 GB umT5 "
                "tower after conditioning via WanI2VAdapter.offload() (wan_i2v.py:397, tested at "
                "tests/test_wan_i2v.py:375). Whether dream.py can reach it is reported by the "
                "budget.offload_lever check below, not asserted here"
            ),
        ),
        BudgetEntry(
            entry_point="diffusers WanImageToVideoPipeline (hf_job_wan_probe.py)",
            peak_gb=DIFFUSERS_GENERATE_PEAK_GB,
            measured=True,
            provenance=(
                f"runs/presentation/t16_lora_futures/scale0.report.json:66 = "
                f"{DIFFUSERS_GENERATE_PEAK_GB} GB (LoRA); "
                f"runs/presentation/wan_futures/faithful_hand_visible.json:18 = "
                f"{DIFFUSERS_GENERATE_MIN_GB} GB (base)"
            ),
            caveat="stock diffusers pipeline: it denoises and decodes video, unlike predict()",
            lever="enable_model_cpu_offload() on the pipeline, or fewer frames / lower resolution",
        ),
        BudgetEntry(
            entry_point="scripts/hf_job_wan_smoke.py (frozen readout)",
            peak_gb=WAN_SMOKE_PEAK_GB,
            measured=True,
            provenance=(
                f"runs/smoke/183599/wan_smoke_report.json:129 = {WAN_SMOKE_PEAK_GB} GB allocated "
                f"/ :130 = {WAN_SMOKE_RESERVED_GB} GB reserved (H200, 5 frames 256x448, "
                f"offload_text=false); independently {WAN_PROBE_PEAK_GB} GB at "
                f"runs/wan_probe/183600/wan_probe_report.json:155"
            ),
            caveat=(
                "recorded after reset_peak_memory_stats() (hf_job_wan_smoke.py:385), so it "
                "EXCLUDES the load transient"
            ),
            lever=(
                "--offload-text is already wired here (hf_job_wan_smoke.py:376) and drops the "
                "~11 GB umT5 tower after condition_text; unmeasured, docs/local_gpu.md estimates "
                "~13 GB with it"
            ),
        ),
        BudgetEntry(
            entry_point="scripts/eval_t16.py / rollout.py / serve_policy.py",
            peak_gb=WAN_SMOKE_PEAK_GB,
            measured=False,
            provenance=EVAL_PROVENANCE,
            caveat="no eval artifact in runs/ carries a peak_vram_gb field",
            lever=(
                "the biggest single lever in the repo, worth the ~11 GB umT5 tower: offload() "
                "is implemented and tested (wan_i2v.py:397) and the question is only whether "
                "these scripts expose --offload-text. See the budget.offload_lever check below "
                "— it reads the scripts rather than trusting this sentence."
            ),
        ),
        BudgetEntry(
            entry_point="scripts/train_t16_lora.py (batch 2 + accum 4)",
            peak_gb=TRAINING_BATCH2_PEAK_GB_ESTIMATE,
            measured=False,
            provenance=TRAINING_PROVENANCE,
            caveat=(
                f"the resident floor alone is {TRAINING_FLOOR_GB_ESTIMATE:.2f} GB and is measured; "
                "everything above it here is not, and the cuDNN conv3d workspace for the fp32 VAE "
                "is unbounded in this arithmetic. --offload-text drops 11.36 GB of it "
                "(docs/local_gpu.md §5 puts that arm at ~15.9 GB)"
            ),
            lever=(
                "--offload-text (nearly free on this corpus: one distinct instruction, so one CPU "
                "umT5 forward in 20 000 steps); gradient checkpointing is already on "
                "(wan_i2v.py:123); lower --batch-size and raise --grad-accum by the same factor "
                "to hold the effective batch at 8, and there is no 8-bit optimizer here"
            ),
        ),
    )


@dataclass(frozen=True)
class BudgetRow:
    entry: BudgetEntry
    total_gb: float
    headroom_gb: float
    verdict: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_point": self.entry.entry_point,
            "peak_gb": self.entry.peak_gb,
            "measured": self.entry.measured,
            "card_total_gb": round(self.total_gb, 2),
            "headroom_gb": round(self.headroom_gb, 2),
            "verdict": self.verdict,
            "provenance": self.entry.provenance,
            "caveat": self.entry.caveat,
            "lever": self.entry.lever,
        }


def verdict(peak_gb: float, total_gb: float, tight_margin_gb: float = TIGHT_MARGIN_GB) -> str:
    """FITS / TIGHT / WILL OOM, in decimal GB. See TIGHT_MARGIN_GB for where the margin is from.

    ``peak_gb`` is ``max_memory_allocated()`` -- ALLOCATED bytes. What has to fit in the card is
    OCCUPIED bytes, which is larger by the caching allocator's reserved-but-unallocated slack.
    So the OOM threshold is ALLOCATOR_OVERHEAD_GB, not zero: a row with +0.36 GB of nominal
    headroom is already 0.54 GB past what the driver will be asked for. That 0.90 GB is measured
    (job 183599:129-130), not assumed -- it is the one place in this file where a number is
    strong enough to move a verdict rather than merely widen a margin.
    """
    headroom = total_gb - peak_gb
    if headroom <= ALLOCATOR_OVERHEAD_GB:
        return "WILL OOM"
    if headroom < tight_margin_gb:
        return "TIGHT"
    return "FITS"


def budget_rows(
    total_gb: float,
    entries: Sequence[BudgetEntry] | None = None,
    tight_margin_gb: float = TIGHT_MARGIN_GB,
) -> list[BudgetRow]:
    return [
        BudgetRow(
            entry=e,
            total_gb=total_gb,
            headroom_gb=total_gb - e.peak_gb,
            verdict=verdict(e.peak_gb, total_gb, tight_margin_gb),
        )
        for e in (entries if entries is not None else budget_entries())
    ]


def render_budget_table(rows: Sequence[BudgetRow], card_label: str) -> str:
    """Fixed-width table + numbered footnotes. Provenance travels with the number, always."""
    if not rows:
        return "(no budget entries)"
    name_w = 55
    lines = [
        f"card: {card_label}",
        "all figures DECIMAL GB — torch.cuda.max_memory_allocated()/1e9, the unit every artifact",
        "in runs/ uses. A '32 GB' card is 32 GiB = 34.36 decimal GB; mixing the two unit systems",
        "makes the card look 2.4 GB smaller than it is, so nothing here is ever converted to GiB.",
        "",
        f"{'entry point':<{name_w}} {'peak':>7} {'headroom':>9}  {'verdict':<12} src",
        "-" * (name_w + 34),
    ]
    for i, row in enumerate(rows, start=1):
        peak = ("" if row.entry.measured else "~") + format(row.entry.peak_gb, ".2f")
        label = row.verdict + ("" if row.entry.measured else " (est)")
        lines.append(
            f"{row.entry.entry_point[:name_w]:<{name_w}} {peak:>7} "
            f"{row.headroom_gb:>+9.2f}  {label:<12} [{i}]"
        )
    lines.append("")
    for i, row in enumerate(rows, start=1):
        kind = "MEASURED" if row.entry.measured else "NOT MEASURED"
        lines.append(f"[{i}] {row.entry.entry_point}  ({kind})")
        lines.append(f"    source: {row.entry.provenance}")
        if row.entry.caveat:
            lines.append(f"    caveat: {row.entry.caveat}")
        lines.append(f"    lever:  {row.entry.lever}")
    lines += [
        "",
        f"TIGHT means headroom < {TIGHT_MARGIN_GB:.1f} GB, and that margin is not arbitrary: an",
        "allocated peak excludes the CUDA context and the caching allocator's slack, and the one",
        f"run that recorded both saw reserved - allocated = {ALLOCATOR_OVERHEAD_GB:.2f} GB, at",
        "runs/smoke/183599/wan_smoke_report.json:129-130. A '~' peak and an '(est)' verdict mean",
        "the number was never measured — read the footnote before planning around it.",
    ]
    return "\n".join(lines)


def section_budget(report: Report, memory: dict, override_gb: float | None) -> None:
    report.section("7. VRAM budget on this card")
    # The basis is FREE VRAM, not the board total, wherever free was measured.
    #
    # Budgeting against the total credits you with memory something else is already holding. A
    # desktop compositor takes 0.3-1.5 GB of a 5090 and section 4 only WARNs below 90% free, so
    # it passes there and then never reaches this table. Against the total, the dream row
    # (32.54 peak) reads TIGHT on every plausible card -- 34.36, 34.19, 33.40, 33.20, 32.60 all
    # give a positive headroom -- so it can only reach WILL OOM on a card that does not exist,
    # which is a check that cannot fire when it matters. Against free VRAM on a box with a
    # desktop up, it fires.
    measured_free = memory.get("vram_free_gb")
    measured_total = memory.get("vram_total_gb")
    if override_gb:
        total = float(override_gb)
        label = f"{total:.2f} GB (from --card-gb)"
    elif measured_free:
        total = float(measured_free)
        held = f", {measured_total - measured_free:.2f} GB already held" if measured_total else ""
        label = f"{total:.2f} GB FREE, measured{held}"
    elif measured_total:
        total = float(measured_total)
        label = f"{total:.2f} GB total measured (free was unreadable — optimistic)"
    else:
        total = NOMINAL_RTX5090_TOTAL_GB
        label = f"{total:.2f} GB ASSUMED (nominal 32 GiB RTX 5090; no CUDA device here to ask)"
    rows = budget_rows(float(total))
    table = render_budget_table(rows, label)
    if report.echo:  # --quiet must leave stdout as pure JSON
        print(table, flush=True)
        print(flush=True)
    for row in rows:
        status = {
            "FITS": STATUS_PASS,
            "TIGHT": STATUS_WARN,
            "WILL OOM": STATUS_WARN if not row.entry.measured else STATUS_FAIL,
        }[row.verdict]
        if not measured_total and override_gb is None:
            # Nothing was measured on this box; a FAIL here would be a verdict about an
            # assumed card. Say so instead of blocking.
            status = STATUS_WARN if status == STATUS_FAIL else status
        report.record(
            f"budget.{row.entry.entry_point.split()[0]}",
            status,
            f"{'~' if not row.entry.measured else ''}{row.entry.peak_gb:.2f} GB peak, "
            f"{row.headroom_gb:+.2f} GB headroom -> {row.verdict}"
            + ("" if row.entry.measured else " (estimate, not a measurement)"),
            fix=row.entry.lever if row.verdict != "FITS" else "",
        )
    # This used to be a hardcoded STATUS_WARN, which is a check that cannot report anything: it
    # said the same thing on a 12 GB laptop and an 80 GB H100. The transient is still unmeasured
    # -- that has not changed and the fix text still says so -- but whether it THREATENS you is
    # a function of the budget above. It only matters when a row is already close, so that is
    # what it now reports.
    closest = min((r.headroom_gb for r in rows), default=0.0)
    report.record(
        "budget.load_transient",
        STATUS_WARN if closest < TIGHT_MARGIN_GB else STATUS_PASS,
        f"unmeasured on this path; the tightest row leaves {closest:+.2f} GB, "
        + (
            "which is inside the margin a load spike could occupy"
            if closest < TIGHT_MARGIN_GB
            else "so a load spike has room even unmeasured"
        ),
        fix=LOAD_TRANSIENT_WARNING if closest < TIGHT_MARGIN_GB else "",
    )
    reach = scripts_exposing_flag(REPO_ROOT / "scripts", OFFLOAD_FLAG)
    wired = [n for n, ok in reach.items() if ok]
    unwired = [n for n, ok in reach.items() if not ok]
    report.soft(
        "budget.offload_lever",
        not unwired,
        f"{OFFLOAD_FLAG} reachable from {len(wired)}/{len(reach)} entry points"
        + (f" — missing on {', '.join(unwired)}" if unwired else ""),
        fix="WanI2VAdapter.offload('text_encoder') (wan_i2v.py:397) drops the ~11 GB umT5 tower "
        "and is already implemented and tested; a script that does not expose the flag cannot "
        "use it. Expected saving is UNMEASURED — docs/local_gpu.md's ~13 GB is arithmetic.",
    )
    report.info["offload_lever"] = reach
    report.info["budget"] = {
        "card_total_gb": round(float(total), 2),
        "card_source": label,
        "tight_margin_gb": TIGHT_MARGIN_GB,
        "allocator_overhead_gb": ALLOCATOR_OVERHEAD_GB,
        "load_transient_warning": LOAD_TRANSIENT_WARNING,
        "rows": [r.as_dict() for r in rows],
        "table": table,
    }


# ---------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preflight gate for a single-GPU WAM box (docs/local_gpu.md §0)."
    )
    p.add_argument("--backbone-source", default=None, help="Wan2.2-TI2V-5B snapshot dir to check")
    p.add_argument("--dataset", default=None, help="dataset root to check")
    p.add_argument("--checkpoint", default=None, help="checkpoint dir/file to check")
    p.add_argument(
        "--card-gb",
        type=float,
        default=None,
        help="override total VRAM (decimal GB) for the budget table — e.g. 34.36 for a 32 GiB "
        "5090, 141.0 for an H200. Default: measured, else a labelled assumption.",
    )
    p.add_argument(
        "--deep-import",
        action="store_true",
        help="actually import each dependency instead of find_spec (slower, catches installs "
        "that are present but broken)",
    )
    p.add_argument("--json", dest="json_out", default=None, help="write the report to this path")
    p.add_argument("--quiet", action="store_true", help="suppress the human-readable log")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = Report(echo=not args.quiet)
    if not args.quiet:
        print("WAM GPU preflight — run this before spending a GPU minute.", flush=True)

    section_environment(report)
    torch_mod, err = import_torch()
    facts = section_torch(report, torch_mod, err)
    section_kernels(report, torch_mod, facts)
    memory = section_memory(report, torch_mod, facts)
    section_dependencies(report, args.deep_import)
    section_assets(report, args)
    section_budget(report, memory, args.card_gb)

    code = exit_code(report.statuses())
    payload = {
        "ok": code == 0,
        "checks": [c.as_dict() for c in report.checks],
        "info": report.info,
    }
    if args.json_out:
        try:
            Path(args.json_out).write_text(json.dumps(payload, indent=2, default=str))
            if not args.quiet:
                print(f"\nreport -> {args.json_out}", flush=True)
        except OSError as write_err:
            print(f"could not write {args.json_out}: {write_err}", flush=True)
    elif args.quiet:
        print(json.dumps(payload, indent=2, default=str), flush=True)

    if not args.quiet:
        print("\n=== verdict ===", flush=True)
        if report.warned:
            print(f"[WARN] {len(report.warned)}: {', '.join(report.warned)}", flush=True)
        if report.failed:
            print(f"[FAIL] {len(report.failed)}: {', '.join(report.failed)}", flush=True)
            print("BLOCKED — fix the FAILs above before running anything on the GPU.", flush=True)
        else:
            print(
                f"ALL {len(report.checks)} CHECKS PASSED OR WARNED — clear to proceed.", flush=True
            )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
