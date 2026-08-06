"""Wan2.2 against Cosmos3-Nano on identical windows, at three corpus sizes (T-38).

Both frozen backbones have already lost to a proprioception ridge — Wan at T-15, Cosmos3-Nano at
T-24 (joints 0.359 / gripper 0.708 against 0.456 / 0.881). Those were two separate runs, months
apart, each compared against a constant quoted from the first of them, and T-37 measured that the
constant is not a constant: the state-only floor climbs 0.4563 -> 0.4879 -> 0.5129 as the corpus
grows 12 -> 24 -> 48 episodes, and a feature set that cleared the 12-episode floor (``past_ee``,
0.4576 vs 0.4563) sat 0.12 *below* it at 48. Every archived backbone verdict in this repo is a
12-episode verdict, and 12 episodes is 56 training windows.

Three things follow, and this driver does all three:

1. **One window set, both backbones.** It calls the two deployed ZeroGPU Spaces with identical
   parameters and then *verifies from the returned reports* that the episode list, window count,
   context frames, resize, chunk length, instruction and the resulting train/val/test episode
   split agree. Having passed the same flags is not evidence that the same windows came back — a
   silently mismatched window set is the one defect that would invalidate the whole comparison,
   so it is checked against the artifacts, not against intent.
2. **Three corpus sizes**, 12 / 24 / 48 by default, each with its own recomputed floor. A
   single-size answer is what T-37 showed can reverse.
3. **A width-matched arm**, below.

**The width confound.** Wan's residual stream is 3072-dim, Cosmos3's is 4096-dim (measured shapes
``(96, 30, 3072)`` and ``(96, 36, 4096)`` in ``runs/wan_probe/`` and ``runs/cosmos3_probe/``), so
the two-block candidate a report quotes is 6144 dims for Wan and 8192 for Cosmos — against 56
training rows. T-37 measured what that does to a ridge: one feature set scored **-0.0950** at 256
dims and **~0.45** at 112 dims, same information. So a raw Wan-vs-Cosmos delta is not only "which
prior is better", it is also "which tensor is wider", and the wider one is not the same backbone
in both directions.

**What can and cannot be controlled from here.** The Spaces' probe endpoints return a log and a
report — not the pooled features (their Gradio outputs are ``[log, JSON]``; changing that means
redeploying, which is out of scope). So:

- With ``--features``, the real arm runs: both backbones' pooled features random-projected to one
  common width, over three seeds, scored with the same ridge/split/labels, plus a row-shuffled
  control per backbone that must land near 0.
- Without them the driver carries one fixed, known-informative CPU feature set — proprioception,
  32 dims, the floor itself — to each backbone's exact width and scores it the same way. It is
  **not** a substitute for projecting the backbones' own features and is labelled
  ``carried_state``, not ``width_matched``, everywhere it appears. Two ways, because only one of
  them reads width at all: a random projection keeps the row space, so the projected floor holds
  the same information at 112 dims as at 8192 and scores the same (measured: 0.5586 vs 0.5584,
  inside one seed spread); padding those 32 dims out with uninformative columns is the half that
  costs a ridge on 56 rows something, and it does — 0.3154 at 112, 0.0064 at 6144.

The floor and the best input-only comparator are not quoted, ever. They come from
``scripts/probe_action_baselines.py`` run on the same window config, because 0.456 / 0.881 are
12-episode numbers and putting them next to a 48-episode probe compares across sample sizes. The
comparator is chosen on **validation**, like the backbone rows; its test-argmax sibling is printed
underneath and labelled optimistic, because on these artifacts the two differ by up to 0.029.

**What it found (2026-08-05, `runs/backbone_eval/compare_backbones.json`, docs §7).** The
12-episode row reproduces T-15 and T-24 to four digits, so nothing drifted. Above that size the
**ranking reverses**: Wan leads by 0.041 joints at 12 episodes and trails by 0.040 at 48
(0.3652 / 0.3011 / 0.3867 against Cosmos's 0.3240 / 0.2837 / 0.4267). A single-size head-to-head
between these two backbones therefore measures nothing about the backbones — and a single-size
head-to-head is what the archive consists of. What does not reverse: both lose to the
proprioception floor at all three sizes, by 0.086 to 0.187 joints, with no trend.

Whether width explains the gap between them is **open**. Carrying a fixed 32-dim signal to 6144 vs
8192 dims moves the projected rows by 0.004 and the padded rows by 0.013, and both are inside their
own seed spread — the 4:3 width ratio is below what this control can resolve, which is not the same
statement as "width is not the explanation". What the padded rows do show is that width is not
cheap: the same 32 dims are worth 0.46–0.51 alone and 0.01–0.07 inside 6144. And the +0.10 that
carrying up appears to buy is not width at all — +0.087 of it is already there at 32 -> 32, with no
width change, from the change of basis alone.

The agreement check earned itself on the first real run: the deployed Wan Space records
``window_select`` and the deployed Cosmos copy predates the field. That is what
``--assume-default`` exists for, and it refuses anything but the documented default.

Usage::

    .venv/bin/python scripts/compare_backbones.py --assume-default window_select
    .venv/bin/python scripts/compare_backbones.py --sizes 12
    .venv/bin/python scripts/compare_backbones.py \\
        --from-reports wan=wan_ep12.json cosmos=cosmos_ep12.json       # no network at all

Artifacts land in ``runs/backbone_eval/`` and carry both Space ids, the resolved model ids, the
dataset snapshot each Space read, feature shapes, peak VRAM, wall clock and the full window config
plus its hash (AC-04).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_SRC = _HERE.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import hf_job_wan_probe as wan

#: Projection seeds and the 1/sqrt(fan_in) scaling are T-37's, so a width-matched number here is
#: directly comparable with the ``past_joint_proj_s*`` rows in ``runs/backbone_eval/``.
PROJECTION_SEED_BASE = 100
DEFAULT_SEEDS = (0, 1, 2)

#: Seed base for the nuisance columns, kept apart from the projection's so that a padded row and a
#: projected row of the same width are two draws and not one.
NUISANCE_SEED_BASE = 200

#: T-37 measured this width to be safe and 256 to be catastrophic at 56 training rows (0.45 vs
#: -0.0950 on the same information), which is the only reason it is 112 and not a round number.
DEFAULT_MATCH_WIDTH = 112

#: Fields of ``info.data`` that define the window set. ``dataset`` is deliberately absent: it is a
#: host-specific cache path, identical in meaning and different in text between two Spaces. Its
#: snapshot revision is compared separately (see :func:`dataset_revision`) and recorded for AC-04.
WINDOW_KEYS = (
    "episodes",
    "windows",
    "frames",
    "resize",
    "chunk_steps",
    "label_dim",
    "instruction",
    "window_select",
)
_ABSENT = "<absent>"

#: What a report that predates a field necessarily ran, per ``hf_job_wan_probe.build_windows``:
#: ``getattr(args, "window_select", "linspace")``. Measured, not assumed in general — the two
#: Spaces carry different vintages of the probe (the Wan Space records ``window_select`` in
#: ``info.data``, the deployed Cosmos copy predates the field), and a field one side does not
#: record is unknown until someone says out loud which value it stands for. ``--assume-default``
#: is that statement, it lands in the artifact, and it is refused when the recorded value is
#: anything other than the default here — a run that says ``motion`` chose different windows.
WINDOW_DEFAULTS = {"window_select": "linspace"}


class WindowMismatch(RuntimeError):
    """The two runs did not score the same windows. Nothing downstream is comparable."""


class SpaceUnavailable(RuntimeError):
    """A deployed Space could not be reached or refused the call."""


class ProbeFailed(RuntimeError):
    """A Space answered, but the run inside it did not produce a usable report."""


@dataclass(frozen=True)
class SpaceSpec:
    """One deployed ZeroGPU Space's probe endpoint, as read off its ``app.py``."""

    space: str
    api_name: str
    #: The Space's ``MODEL_ID`` default. The id the run actually used is parsed out of the returned
    #: log (both apps print ``model: <id>`` first), so a changed Space variable cannot go unseen.
    model_id: str
    extra_inputs: tuple[Any, ...] = ()


#: ``deploy/wan-smoke-space/app.py:525`` and ``deploy/cosmos3-probe-space/app.py:265``. Both probe
#: handlers take (episodes, windows/ep, frames, height, width, chunk_steps) as plain gr.Number, so
#: the corpus-size sweep needs no Space change. Wan's takes a seventh input, the readout spec, and
#: it is pinned to ``mean`` here: the Cosmos tab has no readout control and always mean-pools, so
#: asking Wan for its default ``mean,grid2x2,rand4`` would put readouts in one report that the
#: other backbone never ran, next to numbers that invite being compared.
SPACES: dict[str, SpaceSpec] = {
    "wan": SpaceSpec(
        space="huhn511/wam-wan-smoke",
        api_name="/run_probe",
        model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        extra_inputs=("mean",),
    ),
    "cosmos": SpaceSpec(
        space="huhn511/wam-cosmos3-probe",
        api_name="/run_probe",
        model_id="nvidia/Cosmos3-Nano",
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", default="12,24,48", help="corpus sizes in episodes")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--windows-per-episode", type=int, default=8)
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--height", type=int, default=192)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--chunk-steps", type=int, default=16)
    p.add_argument("--alphas", default="1,10,100,1000,10000")
    p.add_argument(
        "--from-reports",
        nargs=2,
        action="append",
        metavar="REPORT",
        help="two probe reports to compare offline, as [label=]path.json. Repeat the flag once "
        "per corpus size; the size is read from the reports. No network is touched.",
    )
    p.add_argument(
        "--features",
        nargs="*",
        default=[],
        metavar="NAME=PATH",
        help="pooled features per backbone, [N, D] or [N, blocks, D] .npy/.npz. Enables the real "
        "width-matched arm; without it the width control falls back to carried_state.",
    )
    p.add_argument(
        "--assume-default",
        default="",
        metavar="KEY[,KEY]",
        help="window fields one report records and the other does not, to be read as the "
        f"documented default ({WINDOW_DEFAULTS}). The assumption is written into the artifact, "
        "and it is refused if the value actually recorded is not that default.",
    )
    p.add_argument("--match-width", type=int, default=DEFAULT_MATCH_WIDTH)
    p.add_argument("--data-dir", default="data/raw/gr00t_apple")
    p.add_argument("--baselines-dir", default="runs/backbone_eval")
    p.add_argument("--out", default="runs/backbone_eval/compare_backbones.json")
    p.add_argument("--reports-dir", default="runs/backbone_eval/reports")
    p.add_argument("--token", default=None, help="HF token (default: huggingface_hub.get_token())")
    return p.parse_args(argv)


# ---- window agreement --------------------------------------------------------------------


def window_config(report: dict[str, Any]) -> dict[str, Any]:
    """The window set a report describes, as the fields that define it.

    Missing keys become ``"<absent>"`` rather than being skipped: two reports that both predate a
    field agree about it, and one report that has it while the other does not is a disagreement,
    which is the opposite of what silently dropping unknown keys would conclude.
    """
    info = report.get("info", {})
    data = info.get("data", {})
    cfg: dict[str, Any] = {key: data.get(key, _ABSENT) for key in WINDOW_KEYS}
    cfg["split_episodes"] = info.get("probe", {}).get("split_episodes", _ABSENT)
    return cfg


def requested_config(args: argparse.Namespace, size: int) -> dict[str, Any]:
    """What was asked for, in the shape a report reports it."""
    return {
        "episodes": list(range(args.start, args.start + size)),
        "frames": args.frames,
        "resize": [args.height, args.width],
        "chunk_steps": args.chunk_steps,
    }


def config_disagreements(cfg: dict[str, Any], requested: dict[str, Any]) -> list[str]:
    """Fields where a returned config differs from what was requested."""
    return [
        f"{key}: requested {value!r}, report has {cfg.get(key, _ABSENT)!r}"
        for key, value in requested.items()
        if cfg.get(key, _ABSENT) != value
    ]


def dataset_revision(report: dict[str, Any]) -> str | None:
    """The Hub snapshot revision a Space read, or ``None`` for a non-Hub path.

    ``info.data.dataset`` is ``.../datasets--nvidia--GR00T-N1.7-AppleToPlate/snapshots/<sha>`` on a
    Space and a plain directory locally. Only the ``<sha>`` form is a claim about *which* corpus,
    so only that form is compared; a local mirror is recorded and not enforced.
    """
    dataset = str(report.get("info", {}).get("data", {}).get("dataset", ""))
    parts = Path(dataset).parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def apply_assumed_defaults(
    configs: dict[str, dict[str, Any]], keys: tuple[str, ...]
) -> dict[str, Any]:
    """Fill a field only some reports record with the default the others necessarily ran.

    Mutates ``configs`` and returns what was assumed, so the assumption travels into the artifact
    instead of living in whoever typed the flag. Refused, and therefore still a mismatch, when the
    recorded value is not :data:`WINDOW_DEFAULTS` — the point of the field is that ``motion``
    selects a different subpopulation of windows (`hf_job_wan_probe.build_windows`), so no flag
    may wave that through.
    """
    assumed: dict[str, Any] = {}
    for key in keys:
        if key not in WINDOW_DEFAULTS:
            raise ValueError(f"no documented default for {key!r}; known: {sorted(WINDOW_DEFAULTS)}")
        missing = [name for name, cfg in configs.items() if cfg[key] == _ABSENT]
        present = {cfg[key] for cfg in configs.values() if cfg[key] != _ABSENT}
        if not missing or len(present) != 1:
            continue  # nothing to fill, or the reports that do record it already disagree
        (value,) = present
        # Only the documented default may stand in for a field a report does not record. A run
        # that recorded `motion` scored a chosen subpopulation of windows, so propagating it onto
        # a report that never claimed it would be inventing agreement rather than stating it.
        if value != WINDOW_DEFAULTS[key]:
            continue
        for name in missing:
            configs[name][key] = value
        assumed[key] = {"value": value, "assumed_for": missing}
    return assumed


def assert_same_windows(
    reports: dict[str, dict[str, Any]],
    requested: dict[str, Any] | None = None,
    assume_default: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Verify every report describes the same windows; return that config.

    Raises :class:`WindowMismatch` listing every disagreeing field with both values. This is the
    gate the whole comparison rests on, so it reports *all* the differences rather than the first.
    """
    if len(reports) < 2:
        raise WindowMismatch(f"need two reports to compare, got {sorted(reports)}")
    configs = {name: window_config(report) for name, report in reports.items()}
    assumed = apply_assumed_defaults(configs, assume_default)
    names = list(configs)
    reference = names[0]
    problems = [
        f"{key}: " + ", ".join(f"{name}={configs[name][key]!r}" for name in names)
        for key in (*WINDOW_KEYS, "split_episodes")
        if any(configs[name][key] != configs[reference][key] for name in names[1:])
    ]
    revisions = {name: dataset_revision(report) for name, report in reports.items()}
    known = {name: rev for name, rev in revisions.items() if rev is not None}
    if len(set(known.values())) > 1:
        problems.append(
            "dataset revision: " + ", ".join(f"{name}={rev}" for name, rev in known.items())
        )
    if requested is not None:
        problems += [
            f"{name} {problem}"
            for name, cfg in configs.items()
            for problem in config_disagreements(cfg, requested)
        ]
    if problems:
        one_sided = sorted(
            key
            for key in WINDOW_KEYS
            if key in WINDOW_DEFAULTS
            and any(cfg[key] == _ABSENT for cfg in configs.values())
            and any(cfg[key] != _ABSENT for cfg in configs.values())
        )
        if one_sided:
            problems.append(
                f"hint: --assume-default {','.join(one_sided)} records in the artifact that the "
                "report omitting the field ran the documented default. It still refuses if the "
                "value the other report recorded is not that default."
            )
        raise WindowMismatch(
            "the runs did not score the same windows — the comparison is void:\n  "
            + "\n  ".join(problems)
        )
    cfg = dict(configs[reference])
    if assumed:
        cfg["assumed_defaults"] = assumed
    return cfg


# ---- reading a probe report --------------------------------------------------------------


def check_detail(report: dict[str, Any], name: str) -> str | None:
    """The ``detail`` string of one named check, or ``None`` if the check did not run."""
    for check in report.get("checks", []):
        if check.get("name") == name:
            return check.get("detail")
    return None


def feature_shape(report: dict[str, Any]) -> list[int] | None:
    """``[N, blocks, D]`` as the run measured it, off the ``probe.features_finite`` check.

    Taken from the check rather than from ``info.geometry`` because the check records the tensor
    that was actually scored; geometry records what the config said the model has.
    """
    detail = check_detail(report, "probe.features_finite") or ""
    match = re.search(r"\((\d+(?:\s*,\s*\d+)*)\)", detail)
    return [int(part) for part in match.group(1).split(",")] if match else None


def backbone_row(report: dict[str, Any]) -> dict[str, Any]:
    """The comparable numbers out of one probe report.

    The headline pair is ``suggested_*`` — the two blocks that report's own *validation* R^2
    picked, scored on test. That is the only non-circular number in the file: ``measured_*`` and
    ``heuristic_*`` are fixed block indices that differ between the two Spaces' defaults (20,29 and
    15,22 for Wan; 2,12 and 18,26 for Cosmos), so quoting them across backbones compares different
    depths, and the best single block is chosen *on test* and is therefore optimistic. Both are
    kept in the artifact so the choice can be re-litigated without re-running a GPU.
    """
    probe = report.get("info", {}).get("probe")
    if not probe or "candidates" not in probe:
        raise ProbeFailed("report has no info.probe.candidates — the ridge analysis never ran")
    candidates = probe["candidates"]
    key = next((name for name in candidates if name.startswith("suggested_")), None)
    if key is None:
        raise ProbeFailed(f"no suggested_* candidate in {sorted(candidates)}")
    per_block = probe.get("per_block", {})
    best_block = max(per_block, key=lambda b: per_block[b]["joints"]["test_r2"], default=None)
    shape = feature_shape(report)
    blocks = list(probe.get("suggested_blocks", []))
    return {
        "candidate": key,
        "blocks": blocks,
        "joints": candidates[key]["joints"]["test_r2"],
        "gripper": candidates[key].get("gripper", {}).get("test_r2"),
        "dim": (shape[-1] * len(blocks)) if shape and blocks else None,
        "feature_shape": shape,
        "geometry": report.get("info", {}).get("geometry", {}),
        "timings": report.get("info", {}).get("timings", {}),
        "dataset": report.get("info", {}).get("data", {}).get("dataset"),
        "dataset_revision": dataset_revision(report),
        "candidates": {
            name: {
                "joints": row["joints"]["test_r2"],
                "gripper": row.get("gripper", {}).get("test_r2"),
            }
            for name, row in candidates.items()
        },
        "best_single_block": (
            {
                "block": int(best_block),
                "joints": per_block[best_block]["joints"]["test_r2"],
                "gripper": per_block[best_block].get("gripper", {}).get("test_r2"),
                "selected_on": "test — optimistic, not the headline",
            }
            if best_block is not None
            else None
        ),
    }


def parse_log_header(log: str) -> dict[str, Any]:
    """``model:`` and the ``host:`` JSON both Spaces print before requesting a GPU."""
    header: dict[str, Any] = {}
    for line in log.splitlines():
        if line.startswith("model:"):
            header["model_id"] = line.split(":", 1)[1].strip()
        elif line.startswith("host:"):
            try:
                header["host"] = json.loads(line.split(":", 1)[1].strip())
            except json.JSONDecodeError:
                header["host"] = line.split(":", 1)[1].strip()
    return header


# ---- the input-only comparators, recomputed not quoted -------------------------------------


def input_only_rows(results: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Every row a backbone would have to beat: not the floor, not a control.

    ``state_only`` is the floor and is reported on its own line; any ``*shuffled*`` row is a
    control, and a control that can become the bar is not a control.
    """
    rows = [
        (name, row)
        for name, row in results.items()
        if name != "state_only" and "shuffled" not in name
    ]
    if not rows:
        raise ValueError("no input-only rows in the baselines artifact")
    return rows


def best_input_only(results: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The strongest feature set reachable without a video model, chosen on **validation** R^2.

    Selected on val because the backbone rows are: :func:`backbone_row` quotes each report's
    val-selected block pair and flags the test-selected one as optimistic. A comparator picked on
    test and put next to it is not the same protocol on the two sides of the table, and the
    difference is not small at these sample sizes. Measured on the shipped artifacts: at 12
    episodes the test-argmax is ``past_ee_plus_state`` at 0.5407 while its val is 0.4922, below
    ``past_joint_proj_s1_plus_state`` (val 0.5465, test 0.5118) — 0.029 of the published bar was
    selection optimism. At 48 episodes the test-argmax is ``past_joint_proj_s2``, i.e. the luckiest
    of three interchangeable projection seeds (test 0.5385 / 0.5223 / 0.5463), which is the very
    coin flip :func:`score_at_width`'s spread exists to expose.

    The row is returned whole, so the gripper number belongs to the same feature set as the joints
    number — a bar assembled from the best joints of one row and the best gripper of another is not
    a bar any single model has to clear.
    """
    return max(input_only_rows(results), key=lambda item: item[1]["joints"]["val_r2"])


def best_input_only_on_test(results: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The same rows, argmax on test. Kept beside the val-selected bar, never as the bar.

    It is the size of the selection optimism, printed rather than hidden: a reader who wants the
    most demanding number in the file can have it, labelled as what it is.
    """
    return max(input_only_rows(results), key=lambda item: item[1]["joints"]["test_r2"])


def _baselines_from(path: Path, doc: dict[str, Any]) -> dict[str, Any]:
    """Floor + comparators out of one ``probe_action_baselines`` artifact."""
    name, row = best_input_only(doc["results"])
    optimistic_name, optimistic = best_input_only_on_test(doc["results"])
    return {
        "path": str(path),
        "floor": doc["results"]["state_only"],
        "best_input_only": {"features": name, "selected_on": "val", **row},
        "best_input_only_on_test": {
            "features": optimistic_name,
            "joints": optimistic["joints"]["test_r2"],
            "gripper": optimistic.get("gripper", {}).get("test_r2"),
            "selected_on": "test — optimistic, not the bar",
        },
    }


def window_requirement(cfg: dict[str, Any]) -> dict[str, Any]:
    """The subset of a verified window config that a baselines artifact has to reproduce.

    Taken from the reports, not from the CLI flags. Offline the two can differ — hand-fetched
    reports carry whatever geometry they were run at — and if the floor were matched to the flags
    instead, the table would put a comparator from one window set next to a probe from another.
    That is the T-37 error with a different cause.

    ``windows`` and ``window_select`` are in here because the four geometry fields do not identify
    a window set: 8 and 16 windows per episode over the same 24 episodes at the same resize agree
    on every one of them and are different experiments, and ``motion`` picks a different
    subpopulation at identical counts. ``window_select`` is required only when the verified config
    records it — a field no report carries is unknown, and requiring an unknown would reject every
    artifact instead of checking anything.
    """
    keys = ["episodes", "windows", "frames", "resize", "chunk_steps"]
    if cfg.get("window_select", _ABSENT) != _ABSENT:
        keys.append("window_select")
    return {key: cfg[key] for key in keys}


def windows_per_episode(required: dict[str, Any], args: argparse.Namespace) -> int:
    """How many windows per episode reproduce the reports' own window count.

    A report records the total, not the per-episode number, so it is divided back out: the flag is
    intent and the report is evidence, exactly as for the geometry. The flag is the fallback for a
    total that does not divide evenly, which happens because ``linspace`` de-duplicates its picks
    on an episode with fewer eligible chunks than requested (`hf_job_wan_probe.build_windows`).
    Either way the rebuilt count is checked against the report afterwards.
    """
    total, episodes = required.get("windows"), required.get("episodes") or []
    if isinstance(total, int) and episodes and total % len(episodes) == 0:
        return total // len(episodes)
    return args.windows_per_episode


def load_baselines(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Floor + best input-only for these windows, from an artifact built on the same ones.

    Every candidate artifact in ``--baselines-dir`` is checked against the window requirement
    before it is used; a file built on other episodes is skipped rather than quoted. If none
    matches, ``probe_action_baselines`` is run to make one — the same implementation, so a number
    here cannot differ from a number there by being computed differently.
    """
    required = window_requirement(cfg)
    for path in sorted(Path(args.baselines_dir).glob("action_baselines*.json")):
        doc = json.loads(path.read_text())
        if not config_disagreements(doc.get("windows", {}), required):
            return _baselines_from(path, doc)
    return _recompute_baselines(cfg, args, required)


def _recompute_baselines(
    cfg: dict[str, Any], args: argparse.Namespace, required: dict[str, Any]
) -> dict[str, Any]:
    episodes = required["episodes"]
    size, start = len(episodes), episodes[0]
    if not Path(args.data_dir).is_dir():
        raise FileNotFoundError(
            f"no baselines artifact in {args.baselines_dir} matches {required}, and "
            f"--data-dir {args.data_dir} does not exist to make one. Run:\n"
            f"  .venv/bin/python scripts/probe_action_baselines.py --episodes {size} "
            f"--start {start} --out {args.baselines_dir}/action_baselines_ep{size}.json"
        )
    import probe_action_baselines as pab

    height, width = required["resize"]
    out = Path(args.baselines_dir) / f"action_baselines_ep{size}.json"
    argv = [
        "--data-dir", args.data_dir,
        "--episodes", str(size),
        "--start", str(start),
        "--windows-per-episode", str(windows_per_episode(required, args)),
        "--frames", str(required["frames"]),
        "--height", str(height),
        "--width", str(width),
        "--chunk-steps", str(required["chunk_steps"]),
        "--alphas", args.alphas,
        "--out", str(out),
    ]  # fmt: skip
    if required.get("window_select") not in (None, _ABSENT):
        argv += ["--window-select", str(required["window_select"])]
    code = pab.main(argv)
    if code != 0:
        raise ProbeFailed(f"probe_action_baselines exited {code} for {size} episodes")
    doc = json.loads(out.read_text())
    # The recompute is asked for the reports' window set; whether it landed on it is a different
    # statement, and a local mirror that yields a different count would otherwise be quoted as the
    # floor for windows it never saw.
    landed = config_disagreements(doc.get("windows", {}), required)
    if landed:
        raise ProbeFailed(
            "the recomputed baselines did not land on the reports' windows:\n  "
            + "\n  ".join(landed)
        )
    return _baselines_from(out, doc)


# ---- the width control ---------------------------------------------------------------------


def random_projection(fan_in: int, width: int, seed: int) -> np.ndarray:
    """A seeded Gaussian projection, scaled 1/sqrt(fan_in) as in T-37."""
    rng = np.random.default_rng(PROJECTION_SEED_BASE + seed)
    return rng.standard_normal((fan_in, width)).astype(np.float32) / np.sqrt(fan_in)


def project(x: np.ndarray, width: int, seed: int) -> np.ndarray:
    return x @ random_projection(x.shape[1], width, seed)


def nuisance_pad(x: np.ndarray, width: int, seed: int) -> np.ndarray:
    """``x`` widened to ``width`` with columns that carry nothing.

    Unit-variance Gaussian, and the amplitude is irrelevant: ``probe_r2`` standardises every column
    before the ridge sees it (`hf_job_wan_probe.py:505`), so a nuisance column is a nuisance column
    at any scale. Its own seed base so a padded row and a projected row of the same width are not
    two views of the same draw.
    """
    rng = np.random.default_rng(NUISANCE_SEED_BASE + seed)
    pad = rng.standard_normal((x.shape[0], max(width - x.shape[1], 0))).astype(np.float32)
    return np.concatenate([x, pad], axis=1)


def _seed_rows(
    make_x: Any,
    y_joint: np.ndarray,
    y_grip: np.ndarray,
    split: dict[str, np.ndarray],
    alphas: tuple[float, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    """One scored row per seed. ``make_x`` is the only thing that differs between the arms."""
    rows = []
    for seed in seeds:
        x = make_x(seed)
        row: dict[str, Any] = {
            "seed": seed,
            "joints": wan.probe_r2(x, y_joint, split, alphas)["test_r2"],
        }
        if y_grip.size:
            row["gripper"] = wan.probe_r2(x, y_grip, split, alphas)["test_r2"]
        rows.append(row)
    return rows


def _spread(rows: list[dict[str, Any]]) -> dict[str, Any]:
    joints = [row["joints"] for row in rows]
    grips = [row["gripper"] for row in rows if "gripper" in row]
    return {
        "seeds": rows,
        "joints_mean": round(float(np.mean(joints)), 4),
        "joints_spread": round(float(max(joints) - min(joints)), 4),
        "gripper_mean": round(float(np.mean(grips)), 4) if grips else None,
        "gripper_spread": round(float(max(grips) - min(grips)), 4) if grips else None,
    }


def unprojected_score(
    x: np.ndarray,
    y_joint: np.ndarray,
    y_grip: np.ndarray,
    split: dict[str, np.ndarray],
    alphas: tuple[float, ...],
) -> dict[str, Any]:
    """``x`` scored as it stands. Nothing is applied to it, which is what makes it *its* number.

    Not :func:`score_at_width` at ``x.shape[1]``: a square Gaussian is not a no-op. It rotates the
    features, and ``probe_r2`` standardises per column afterwards, so the ridge's spherical prior
    lands on a different basis and the score moves. Measured on the 12-episode windows, 32-dim
    proprioception reads 0.4563 unprojected and 0.5429 through a 32x32 projection.
    """
    return {
        "width": int(x.shape[1]),
        "joints": wan.probe_r2(x, y_joint, split, alphas)["test_r2"],
        "gripper": wan.probe_r2(x, y_grip, split, alphas)["test_r2"] if y_grip.size else None,
    }


def score_at_width(
    x: np.ndarray,
    y_joint: np.ndarray,
    y_grip: np.ndarray,
    split: dict[str, np.ndarray],
    alphas: tuple[float, ...],
    width: int,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """Score ``x`` projected to ``width``, once per seed, and report the spread.

    The spread is the point, not decoration: T-37 found one projection landing on a good number by
    luck, so a single seed here would be a coin flip presented as a measurement.

    **What this arm cannot answer.** A projection is a change of basis, not of information: for
    ``width >= rank(x)`` the projected tensor spans the same row space, so the ridge is offered the
    same signal at every target width and the score is flat. Measured on the 12-episode windows
    with 32-dim proprioception: 0.5429 / 0.5586 / 0.5595 / 0.5616 / 0.5542 / 0.5584 joints at
    32 / 112 / 256 / 1024 / 6144 / 8192 dims — a 73x width change (112 vs 8192) moves it 0.0002,
    inside the 112 row's own seed spread of 0.0116. Use it to compare two tensors *at one width*;
    :func:`score_with_nuisance` is the arm that reads a width difference.
    """
    rows = _seed_rows(lambda seed: project(x, width, seed), y_joint, y_grip, split, alphas, seeds)
    return {"width": width, "source_dim": int(x.shape[1]), **_spread(rows)}


def score_with_nuisance(
    x: np.ndarray,
    y_joint: np.ndarray,
    y_grip: np.ndarray,
    split: dict[str, np.ndarray],
    alphas: tuple[float, ...],
    width: int,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """Score ``x``'s information hidden among ``width - x.shape[1]`` uninformative directions.

    This is what a wide backbone tensor asks of a ridge on 56 training rows and a projected 32-dim
    floor does not: thousands of directions that have to be regularised away, of which the
    projection arm adds exactly none. Same 12-episode windows, same 32 informative dims, three
    seeds: joints 0.3154 at 112 dims, 0.1900 at 1024, 0.0064 at 6144, 0.0298 at 8192 — against
    0.5429 to 0.5616 for the projection at those widths. The penalty this measures is the one the
    head-to-head can actually be confounded by.
    """
    rows = _seed_rows(
        lambda seed: nuisance_pad(x, width, seed), y_joint, y_grip, split, alphas, seeds
    )
    return {
        "width": max(width, int(x.shape[1])),
        "source_dim": int(x.shape[1]),
        "informative_dims": int(x.shape[1]),
        **_spread(rows),
    }


def shuffled_control(
    x: np.ndarray,
    y_joint: np.ndarray,
    y_grip: np.ndarray,
    split: dict[str, np.ndarray],
    alphas: tuple[float, ...],
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """Same features, rows permuted against the labels. Must land near 0.

    Width and marginal distribution survive the permutation; only the window-to-label pairing
    dies. So a non-zero score here is the split being exploited, not a feature doing work.

    Run over the same seeds as the projection, and for the same reason: measured on these windows,
    a single permutation of the 32-dim proprioception vector reached **0.13** on gripper at 12
    episodes while its siblings sat at 0.00. One permutation is a draw from a null distribution
    with a fat tail, not a null.
    """
    rows = _seed_rows(
        lambda seed: x[np.random.default_rng(seed).permutation(x.shape[0])],
        y_joint,
        y_grip,
        split,
        alphas,
        seeds,
    )
    joints = [row["joints"] for row in rows]
    grips = [row["gripper"] for row in rows if "gripper" in row]
    return {
        "dim": int(x.shape[1]),
        "seeds": rows,
        "joints_mean": round(float(np.mean(joints)), 4),
        "joints_worst": round(float(max(joints, key=abs)), 4),
        "gripper_mean": round(float(np.mean(grips)), 4) if grips else None,
        "gripper_worst": round(float(max(grips, key=abs)), 4) if grips else None,
    }


def window_tensors(
    windows: list[dict[str, Any]], chunk_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """``(state, y_joint, y_grip, split)`` — built exactly as the probe and the baselines do."""
    y = np.stack([w["label"] for w in windows])
    joint_dim = min(chunk_steps * wan.NUM_JOINTS, y.shape[1])
    state = np.stack(
        [np.concatenate([w["state"].q, w["state"].dq, w["state"].gripper_state]) for w in windows]
    ).astype(np.float32)
    split = wan.episode_split(np.asarray([w["episode"] for w in windows]))
    return state, y[:, :joint_dim], y[:, joint_dim:], split


def load_features(spec: list[str], rows: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    """``name=path`` -> pooled ``[N, D]``.

    A ``[N, blocks, D]`` array is collapsed onto the blocks that backbone's report chose, so the
    width-matched arm projects the same tensor the raw row reports and not a different one.
    """
    features: dict[str, np.ndarray] = {}
    for item in spec:
        name, _, raw = item.partition("=")
        if not raw:
            raise ValueError(f"--features wants NAME=PATH, got {item!r}")
        loaded = np.load(raw)
        array = loaded[next(iter(loaded.files))] if hasattr(loaded, "files") else loaded
        if array.ndim == 3:
            blocks = rows.get(name, {}).get("blocks") or [0]
            array = np.concatenate([array[:, int(b)] for b in blocks], axis=1)
        features[name] = np.asarray(array, dtype=np.float32)
    return features


def width_control(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    features: dict[str, np.ndarray],
    windows: list[dict[str, Any]] | None,
    floor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The arm that separates "better prior" from "wider tensor".

    ``backbone_features`` mode is the real one and needs ``--features``; ``carried_state`` is what
    is measurable when only the reports came back, and it answers a strictly weaker question: what
    does a ridge on this split do to one fixed known-informative signal when the tensor around it
    changes size.

    That question has two halves and this returns both, because only one of them is a width
    measurement. ``carried`` projects the 32-dim floor to each width: same information at every
    width, and its flatness says nothing about width (see :func:`score_at_width`).
    ``carried_with_nuisance`` keeps those 32 dims and fills the rest with uninformative columns,
    which is the half a 6144-dim residual stream actually imposes.

    ``cfg`` is the verified report config, and the chunk length comes from there rather than from
    the flags: it is what separates the joint channels of a label from the gripper channels, and
    reading it off a default would score gripper synergies as joint deltas.
    """
    alphas = tuple(float(a) for a in args.alphas.split(",") if a.strip())
    if windows is None:
        return {
            "mode": "unavailable",
            "reason": (
                f"--data-dir {args.data_dir} is not readable, so the labels and the episode split "
                "cannot be rebuilt locally. The raw joints/gripper numbers below are "
                "width-confounded and must not be read as a prior comparison."
            ),
            "widths": {name: row.get("dim") for name, row in rows.items()},
        }
    state, y_joint, y_grip, split = window_tensors(windows, cfg["chunk_steps"])
    seeds = DEFAULT_SEEDS
    if features:
        arms = {
            # `native` is this backbone's own unprojected number — the one the raw row in the
            # table reports — so it must not go through a projection on the way. `shuffled` is
            # built from this arm's `x` and not from some shared tensor: a control computed on
            # another backbone's features is that backbone's null, not this one's.
            name: {
                "native": unprojected_score(x, y_joint, y_grip, split, alphas),
                "matched": score_at_width(
                    x, y_joint, y_grip, split, alphas, args.match_width, seeds
                ),
                "shuffled": shuffled_control(x, y_joint, y_grip, split, alphas, seeds),
            }
            for name, x in features.items()
        }
        return {"mode": "backbone_features", "match_width": args.match_width, "arms": arms}

    # The source width leads the list on purpose. It is the same tensor at the same size through
    # the same machinery, so the step from `unprojected` to it is what a projection costs or buys
    # when the width does not change at all — and on the real windows that step is +0.087 of the
    # +0.098 the 6144 row shows. Without this row all of it reads as a width effect.
    widths = list(
        dict.fromkeys(
            [
                int(state.shape[1]),
                *sorted({row["dim"] for row in rows.values() if row.get("dim")}),
                args.match_width,
            ]
        )
    )
    unprojected = unprojected_score(state, y_joint, y_grip, split, alphas)
    return {
        "mode": "carried_state",
        "match_width": args.match_width,
        "caveat": (
            "proprioception (32 dims, the floor) put at each backbone's candidate width two ways. "
            "`carried` random-projects it, which preserves its row space: every width holds the "
            "same information, so that family cannot read a width difference and its flatness is "
            "not evidence of one. `carried_with_nuisance` keeps the 32 informative dims and pads "
            "the rest with uninformative columns, which is the part of being wide that costs a "
            "ridge something. Neither is the backbones' own features at matched width — that "
            "needs --features."
        ),
        "widths": {name: row.get("dim") for name, row in rows.items()},
        "unprojected": unprojected,
        # The windows rebuilt here must be the windows the baselines artifact was fitted on, or
        # the floor in the table belongs to a different experiment than the width rows. The
        # unprojected state score IS that artifact's state_only row, so equality is the proof.
        "reproduces_floor": _floor_agreement(unprojected, floor),
        "carried": [
            score_at_width(state, y_joint, y_grip, split, alphas, width, seeds)
            for width in widths
        ],
        "carried_with_nuisance": [
            score_with_nuisance(state, y_joint, y_grip, split, alphas, width, seeds)
            for width in widths
            if width > state.shape[1]
        ],
        "shuffled": shuffled_control(state, y_joint, y_grip, split, alphas, seeds),
    }


def _floor_agreement(
    unprojected: dict[str, Any], floor: dict[str, Any] | None
) -> dict[str, Any] | None:
    if floor is None:
        return None
    artifact = floor["joints"]["test_r2"]
    return {
        "artifact": artifact,
        "recomputed": unprojected["joints"],
        "agrees": abs(artifact - unprojected["joints"]) < 5e-4,
    }


# ---- calling the Spaces ---------------------------------------------------------------------


def resolve_token(explicit: str | None) -> str:
    if explicit:
        return explicit
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        raise SpaceUnavailable(
            "no Hugging Face token: both Spaces are private, so the call cannot be made.\n"
            "  Fix: `huggingface-cli login`, or pass --token, or assemble the comparison offline "
            "with --from-reports a.json b.json."
        )
    return token


def call_space(
    spec: SpaceSpec, size: int, args: argparse.Namespace, token: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one Space's probe endpoint. Returns ``(report, meta)``.

    Both handlers are generators; ``gradio_client`` returns the last yield, which is the pair
    ``(log, report)``. A run that crashed inside the Space yields ``{"ok": False, ...}`` instead of
    raising, so the report is checked rather than assumed.
    """
    from gradio_client import Client

    if args.start != 0:
        raise SpaceUnavailable(
            f"--start {args.start} cannot be honoured remotely: both probe tabs download "
            "range(episodes) (deploy/wan-smoke-space/app.py:209), so the Space always starts at "
            "episode 0. Run the probe locally, or compare with --from-reports."
        )
    inputs = (
        size,
        args.windows_per_episode,
        args.frames,
        args.height,
        args.width,
        args.chunk_steps,
        *spec.extra_inputs,
    )
    started = time.perf_counter()
    try:
        client = Client(spec.space, token=token, verbose=False)
        log, report = client.predict(*inputs, api_name=spec.api_name)
    except Exception as exc:
        raise SpaceUnavailable(
            f"{spec.space} did not answer {spec.api_name}: {exc}\n"
            "  Likely: the Space is asleep (open it once in a browser), the ZeroGPU daily quota "
            "(40 min with PRO) is spent, or the token has no access to this private Space.\n"
            "  Either way the comparison can still be assembled by hand: download each report "
            "from the Space UI and pass --from-reports wan=<a>.json cosmos=<b>.json."
        ) from exc
    wall = round(time.perf_counter() - started, 1)
    if not isinstance(report, dict) or not report.get("ok"):
        detail = report.get("error") if isinstance(report, dict) else report
        raise ProbeFailed(
            f"{spec.space} ran but reported failure at {size} episodes: {detail}\n"
            f"  last 400 chars of its log:\n{str(log)[-400:]}"
        )
    meta = {"space": spec.space, "wall_s": wall, "declared_model_id": spec.model_id}
    meta.update(parse_log_header(str(log)))
    return report, meta


# ---- assembly and output ---------------------------------------------------------------------


def _labelled_path(token: str) -> tuple[str, Path]:
    """``label=path`` or a bare path, whose stem becomes the label."""
    label, sep, raw = token.partition("=")
    if not sep:
        return Path(token).stem, Path(token)
    return label, Path(raw)


def meta_path(report_path: Path) -> Path:
    """Sibling that carries which Space produced a report, and how long it took.

    A probe report says nothing about its own origin — the Space id, the resolved model id and the
    wall clock live in the Gradio log, not in the JSON. The online path writes them here so an
    offline reassembly of the same files is not a lossier artifact than the run that fetched them.
    """
    return report_path.with_name(f"{report_path.stem}.meta.json")


def collect_offline(pairs: list[list[str]]) -> dict[int, dict[str, Any]]:
    """``--from-reports`` -> ``{size: {label: (report, meta)}}``, size read from the reports."""
    runs: dict[int, dict[str, Any]] = {}
    for pair in pairs:
        loaded: dict[str, Any] = {}
        for token in pair:
            label, path = _labelled_path(token)
            report = json.loads(path.read_text())
            # A hand-fetched report has no sidecar, and then the origin is honestly unknown
            # rather than guessed from the filename.
            meta = {"source": str(path), "space": None, "model_id": None}
            sidecar = meta_path(path)
            if sidecar.is_file():
                meta.update(json.loads(sidecar.read_text()))
            loaded[label] = {"report": report, "meta": meta}
        sizes = {
            len(window_config(entry["report"])["episodes"]) for entry in loaded.values()
        }
        if len(sizes) != 1:
            raise WindowMismatch(f"paired reports cover different corpus sizes: {sorted(sizes)}")
        runs[sizes.pop()] = loaded
    return runs


def render_table(
    size: int,
    cfg: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    baselines: dict[str, Any],
    width: dict[str, Any],
) -> str:
    """One corpus size, one table. Returned as text so the artifact and stdout cannot drift."""
    split = cfg.get("split_episodes") or {}
    counts = "/".join(str(len(split.get(k, []))) for k in ("train", "val", "test"))
    header = (
        f"=== {size} episodes · {cfg.get('windows')} windows · {cfg.get('frames')} frames · "
        f"{'x'.join(str(v) for v in cfg.get('resize', []))} · chunk {cfg.get('chunk_steps')} · "
        f"split {counts} episodes ==="
    )
    lines = [header]
    for key, assumption in (cfg.get("assumed_defaults") or {}).items():
        lines.append(
            f"assumed: {key}={assumption['value']!r} for "
            f"{', '.join(assumption['assumed_for'])} — the report does not record it"
        )
    lines.append(f"{'arm':<42}{'dim':>6}{'joints':>10}{'gripper':>10}  note")

    def line(name: str, dim: Any, joints: Any, gripper: Any, note: str = "") -> str:
        dim_s = f"{dim:>6}" if dim is not None else f"{'?':>6}"
        joints_s = f"{joints:>10.4f}" if joints is not None else f"{'—':>10}"
        grip_s = f"{gripper:>10.4f}" if gripper is not None else f"{'—':>10}"
        return f"{name:<42}{dim_s}{joints_s}{grip_s}  {note}"

    for name, row in rows.items():
        blocks = ",".join(str(b) for b in row["blocks"])
        label = f"{name} · blocks {blocks}"
        lines.append(line(label, row["dim"], row["joints"], row["gripper"], "val-selected pair"))

    source = Path(baselines["path"]).name
    floor = baselines["floor"]
    lines.append(
        line(
            "state-only floor",
            floor.get("dim"),
            floor["joints"]["test_r2"],
            floor["gripper"]["test_r2"],
            source,
        )
    )
    best = baselines["best_input_only"]
    lines.append(
        line(
            f"best input-only · {best['features']}",
            best.get("dim"),
            best["joints"]["test_r2"],
            best["gripper"]["test_r2"],
            f"{source} · val-selected, as the backbone rows are",
        )
    )
    # The test-argmax of the same rows, printed because it is what a reader would otherwise
    # compute by eye off the artifact and quote as the bar. The gap between the two lines is the
    # selection optimism, and at 12 episodes it was 0.029 joints.
    optimistic = baselines.get("best_input_only_on_test")
    if optimistic and optimistic["features"] != best["features"]:
        lines.append(
            line(
                f"  (same rows, best on test · {optimistic['features']})",
                None,
                optimistic["joints"],
                optimistic["gripper"],
                "test-selected — optimistic, not the bar",
            )
        )

    def projected(label: str, arm: dict[str, Any]) -> str:
        return line(
            f"{label} · {len(arm['seeds'])} seeds",
            arm["width"],
            arm["joints_mean"],
            arm["gripper_mean"],
            f"spread {arm['joints_spread']:.4f}",
        )

    def control(label: str, arm: dict[str, Any]) -> str:
        return line(
            f"shuffled {label} · {len(arm['seeds'])} seeds",
            arm["dim"],
            arm["joints_mean"],
            arm["gripper_mean"],
            f"control, must sit near 0 (worst seed {arm['joints_worst']:+.4f})",
        )

    if width["mode"] == "backbone_features":
        for name, arm in width["arms"].items():
            lines.append(projected(f"width-matched {name}", arm["matched"]))
            lines.append(control(name, arm["shuffled"]))
    elif width["mode"] == "carried_state":
        lines += [projected(f"carried state · width {a['width']}", a) for a in width["carried"]]
        lines += [
            projected(f"carried state + nuisance · width {a['width']}", a)
            for a in width.get("carried_with_nuisance", [])
        ]
        lines.append(control("carried state", width["shuffled"]))
        agreement = width.get("reproduces_floor")
        if agreement is not None:
            verdict = "agrees" if agreement["agrees"] else "DISAGREES — the floor is not this one"
            lines.append(
                f"local rebuild of these windows reproduces the artifact floor: "
                f"{agreement['recomputed']:.4f} vs {agreement['artifact']:.4f} — {verdict}"
            )
    else:
        lines.append(f"width control unavailable — {width['reason']}")
    return "\n".join(lines)


def config_hash(cfg: dict[str, Any]) -> str:
    """AC-04: one short hash over the window config every row in a table was scored on."""
    return sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class Comparison:
    """Everything one corpus size produced, in the order it was produced."""

    size: int
    window_config: dict[str, Any]
    backbones: dict[str, dict[str, Any]]
    baselines: dict[str, Any]
    width: dict[str, Any]
    table: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def compare_one(args: argparse.Namespace, size: int, loaded: dict[str, Any]) -> Comparison:
    """Agreement check first, then everything else off the config it verified."""
    reports = {name: entry["report"] for name, entry in loaded.items()}
    # Online, the Space is also held to what it was asked for. Offline there is nothing to hold
    # the reports to but each other — hand-fetched files carry their own geometry, which is why
    # the floor and the local windows below are matched to the reports rather than to the flags.
    requested = requested_config(args, size) if args.from_reports is None else None
    assume = tuple(k.strip() for k in args.assume_default.split(",") if k.strip())
    cfg = assert_same_windows(reports, requested, assume)
    rows = {name: backbone_row(report) for name, report in reports.items()}
    for name, entry in loaded.items():
        rows[name]["run"] = entry["meta"]
    features = load_features(args.features, rows)
    baselines = load_baselines(cfg, args)
    windows = build_windows_for(args, cfg)
    width = width_control(args, cfg, rows, features, windows, baselines["floor"])
    comparison = Comparison(size, cfg, rows, baselines, width)
    comparison.table = render_table(size, cfg, rows, baselines, width)
    comparison.meta = {"config_sha256": config_hash(cfg)}
    return comparison


def build_windows_for(
    args: argparse.Namespace, cfg: dict[str, Any]
) -> list[dict[str, Any]] | None:
    """Local windows for the width control, or ``None`` when the corpus is not on this machine.

    Built from the *verified report config*, not the CLI flags — including how many windows per
    episode and which selection rule, both of which the flags can disagree with — so the width rows
    and the floor are fitted on the windows the probes were actually scored on. The rebuilt count
    is then checked against the reports': a local mirror that yields a different number of windows
    is a different experiment, and it would show up as a clean table with the wrong floor in it.
    """
    if not Path(args.data_dir).is_dir():
        return None
    episodes = cfg["episodes"]
    height, width = cfg["resize"]
    select = cfg.get("window_select", _ABSENT)
    local = argparse.Namespace(
        data_dir=args.data_dir,
        episodes=len(episodes),
        start=episodes[0],
        windows_per_episode=windows_per_episode(cfg, args),
        window_select=WINDOW_DEFAULTS["window_select"] if select == _ABSENT else select,
        frames=cfg["frames"],
        height=height,
        width=width,
        chunk_steps=cfg["chunk_steps"],
        instruction=None,
    )
    windows, _, _ = wan.build_windows(local)
    if not windows:
        return None
    expected = cfg.get("windows")
    if isinstance(expected, int) and len(windows) != expected:
        raise WindowMismatch(
            f"the local corpus rebuilt {len(windows)} windows for episodes {episodes[0]}.."
            f"{episodes[-1]}, but both reports were scored on {expected}. The width control and "
            "the floor would be fitted on a window set the probes never saw.\n"
            "  Fix: --windows-per-episode, or a local mirror of the dataset snapshot the reports "
            "name."
        )
    return windows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.from_reports:
            runs = collect_offline(args.from_reports)
        else:
            token = resolve_token(args.token)
            runs = {}
            reports_dir = Path(args.reports_dir)
            reports_dir.mkdir(parents=True, exist_ok=True)
            for size in [int(s) for s in args.sizes.split(",") if s.strip()]:
                loaded = {}
                for name, spec in SPACES.items():
                    print(f"[{size} ep] {spec.space} …", flush=True)
                    report, meta = call_space(spec, size, args, token)
                    path = reports_dir / f"{name}_ep{size}.json"
                    path.write_text(json.dumps(report, indent=2))
                    meta["source"] = str(path)
                    meta_path(path).write_text(json.dumps(meta, indent=2))
                    loaded[name] = {"report": report, "meta": meta}
                runs[size] = loaded

        comparisons = []
        for size in sorted(runs):
            comparison = compare_one(args, size, runs[size])
            comparisons.append(comparison)
            print("\n" + comparison.table)
    except WindowMismatch as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2
    except (SpaceUnavailable, ProbeFailed, FileNotFoundError) as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "task": "T-38",
                "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "mode": "from-reports" if args.from_reports else "spaces",
                "spaces": {name: spec.space for name, spec in SPACES.items()},
                "match_width": args.match_width,
                "projection_seeds": list(DEFAULT_SEEDS),
                "sizes": [c.size for c in comparisons],
                "results": {
                    str(c.size): {
                        "window_config": c.window_config,
                        "config_sha256": c.meta["config_sha256"],
                        "backbones": c.backbones,
                        "baselines": c.baselines,
                        "width_control": c.width,
                        "table": c.table,
                    }
                    for c in comparisons
                },
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
