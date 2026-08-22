#!/usr/bin/env python
"""T-040 / PR-08 §8 item 3 — the Cosmos-Transfer2.5 restyle driver `97` calls.

    python scripts/restyle_transfer25.py --checkpoint-path <ckpt> --no-guardrails \
           --manifest <SOURCE/manifest.json> --styles <STYLE_PARTITION> --style-set <set> \
           --work-list <work.jsonl> --out <raw-dir> --resolution 640x480 --nproc <n> \
           --control <spec>

WHAT THIS IS. Cosmos-Transfer2.5, FROZEN, restyling frames of episodes that already have actions.
The recorded teleop trajectory is carried over unchanged and is never regenerated; the pixels are
an input perturbation and the label stays correct. That is PR-08 §2's whole argument, and it is why
this file only ever reads the action side.

PR-08 §6's G0c RUNS HERE, AND THAT IS THE POINT OF IT. G0a and G0b are checkers over a finished
corpus. G0c is not: its sentence is "the defect cannot enter", because the only instrument that
could detect the generic-manipulator defect after the fact — ``video_fidelity`` — has been measured
against it and cannot see it, and §6 refuses an IoU threshold on the robot mask in the same breath
("would be a coined number"). A gate solved by construction has to live in the construction. So
every clip this driver calls a success has had the real robot's pixels composited back over it from
the SOURCE frame, on every frame, through :mod:`robot_composite`; a clip that has not been
composited never reaches that status and its ``vision.mp4`` is renamed to ``vision.uncomposited.mp4``
so the harvest cannot file it either. There is no flag, no environment variable and no backend that
skips it — ``--backend null`` composites too, because it is a placeholder GENERATOR, not a
placeholder pipeline.

NOTHING HERE IS LICENSED TO RUN. PR-08 §1 gates generation, and staging Transfer2.5's weights is a
download at scale — the project owner's call under the sub-project rule. This script exists so that
the decision is about generating, not about whether a driver could be written.

THE API IS READ, NOT GUESSED. `97` refused to write this file on the grounds that guessing the
module path, the payload schema and the conditioning flags would be worse than saying so. It was
right. Everything below is cited to `docs/transfer25-api.md`, which is cited to file and line in
nvidia-cosmos/cosmos-transfer2.5 @ main, read 2026-08-16. **Three of our spellings do not exist
upstream and are translated at the boundary here** (api §7): `--no-guardrails` is upstream's
`--disable-guardrails`; `--resolution 640x480` is upstream's bucket key `"480"` plus a 4:3 input;
and the per-unit output layout `97` requires is ours, because upstream writes flat.

FOUR THINGS THIS FILE REFUSES TO DEFAULT, each because a default would silently produce clips that
look fine and are not evidence:

1. **The seed.** `97` makes it mandatory in the work unit: arm C is ten samples of ONE prompt, so a
   driver that ignores `row["seed"]` yields either ten identical files or ten irreproducible ones,
   and the generator-fingerprint control has measured nothing either way.
2. **The control modality and weight.** `configs/transfer25/styles.toml` commits the prompts, the
   seeds and the partition — and says NOTHING about which Transfer2.5 control block conditions the
   restyle. That choice decides how much geometry survives, which is exactly what PR-08 §6's G0b
   gate measures, so picking it after looking at clips is the same failure the style partition
   exists to prevent. There is therefore no default: `--control` is required and is recorded.
3. **Per-unit fault isolation.** Upstream's `generate()` loops with no try/except and `keep_going`
   covers only guardrail blocks (api §6), so one unreadable video kills a whole batch. Every unit
   is run inside its own guard here.
4. **`--no-guardrails`.** Not a preference. The guardrail's RetinaFace postprocessor writes blurred
   pixels back into the frame and the blurred frames are what reach disk (api §3). Restyling to
   study geometry and then blurring the hand edits the evidence.

BACKENDS. `--backend transfer25` imports the real framework. `--backend null` writes a deterministic
placeholder instead of calling a model — it exercises the manifest reading, the seed channel, the
work-list contract, the per-unit isolation, the G0c composite and the output layout without a GPU, a
checkout or a weight. That is what the tests run, and it is the only backend that will work on this
workstation.

The placeholder is a REAL, decodable mp4 of flat colour fields whose bytes are a function of the
sample — it used to be an undecodable text blob, and that had to change when G0c landed: a composite
needs a clip it can decode, and exempting the null backend from compositing would have created the
one thing G0c must not have, a reachable code path that skips it. The seed determinism the arm-C
control depends on is unchanged: same sample, same bytes; different seed, different bytes.

**WHAT KEEPS A PLACEHOLDER OUT OF A CORPUS IS NOT ITS FILE FORMAT.** It used to be, in the weak
sense that the blob would not decode, and this docstring used to claim `screen_corpus.py` would
reject one — which was never true: `screen_corpus.py` scores ACTION columns, episode by episode, and
contains no reference to a backend, to this status file or to pixels at all. Now that the
placeholder is a valid mp4 the claim has to be enforced somewhere real, and it is enforced in two
places that do not depend on anyone reading prose:

1. `97_transfer25_restyle.sbatch` never passes `--backend`, so the cluster path always runs
   `transfer25`. A test walks the sbatch's actual command lines and asserts it.
2. Every unit stamps `"backend"` into its own `sample_outputs.json`, and 97's harvest REFUSES to
   file a clip whose record does not say `transfer25` — loudly, as a fatal, because a placeholder
   in a chunk directory means the generation path was edited and the chunk is not what its record
   says it is.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import traceback
from dataclasses import dataclass, field

import numpy as np

# A sibling in scripts/, imported by name. Anchoring sys.path on this file's own directory rather
# than trusting the caller's: the sbatch invokes this with an absolute path from ${FRAMEWORK} as
# the working directory, so sys.path[0] is not scripts/ there, and an ImportError at that point
# would be a chunk of the run lost to a path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import robot_composite  # noqa: E402  — PR-08 §6 G0c; see the module docstring above

#: Upstream's resolution field is a key into ``VIDEO_RES_SIZE_INFO`` keyed by (bucket, aspect), NOT
#: a WxH string (api §2). PR-08 §3 fixes 640x480, so this is the one entry that matters and the map
#: exists to make the translation visible rather than to support other sizes.
_RESOLUTION_BUCKETS = {"640x480": ("480", "4,3")}

#: The control blocks upstream accepts (api §8). Anything else is a typo, and a typo that reached
#: ``extra="forbid"`` would fail 10 050 clips in, not now.
_CONTROL_KEYS = ("depth", "seg", "edge", "vis")


class DriverError(RuntimeError):
    """A refusal, as opposed to a crash. Carries a message meant for the operator."""


@dataclass(frozen=True)
class WorkUnit:
    """One row of ``work.jsonl``. The schema is `97`'s, quoted in its header."""

    unit: str
    episode: str
    frames: int
    style: str
    repeat: int
    seed: int


@dataclass(frozen=True)
class Control:
    """One control block: a modality and its weight."""

    key: str
    weight: float


@dataclass
class Outcome:
    unit: str
    status: str
    detail: str = ""
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# reading the inputs
# --------------------------------------------------------------------------------------------


def load_manifest(path: pathlib.Path) -> dict[str, dict]:
    """``SOURCE/manifest.json`` → ``{episode_id: entry}``, with the resolution asserted.

    The resolution check is `97`'s, repeated here rather than trusted: this script can be run
    outside that job, and the 120x160 converted corpus has a perfectly valid manifest that would
    sail through every generic check while producing clips worthless to a VLA that trains at
    640x480 (PR-08 §3).
    """
    man = json.loads(path.read_text(encoding="utf-8"))
    res = tuple(man.get("resolution") or ())
    if res != (640, 480):
        raise DriverError(
            f"{path} declares resolution {res or 'nothing'}, not (640, 480). PR-08 §3 fixes "
            "640x480 as the GR00T N1.7 ego_view contract; a restyle at any other size is not the "
            "registered experiment."
        )
    episodes = {}
    for entry in man.get("episodes") or ():
        ep_id = entry.get("id")
        if not ep_id:
            raise DriverError(f"{path} has an episode entry with no 'id': {entry!r}")
        episodes[str(ep_id)] = entry
    if not episodes:
        raise DriverError(f"{path} lists no episodes.")
    return episodes


def load_styles(path: pathlib.Path, style_set: str) -> dict[str, dict]:
    """The committed partition → ``{style_id: style}`` for one set.

    Reads the set from the file and never synthesises one. An earlier draft of `97`'s own expansion
    invented the identity style on the assumption that arm C needed no committed prompt; that was
    wrong, because arm C is the control that decides whether a gain from arm B is diversity or
    generator fingerprint, so its prompt is exactly as load-bearing as B's.
    """
    styles = json.loads(path.read_text(encoding="utf-8"))
    if style_set not in styles:
        raise DriverError(
            f"{path} has no '{style_set}' set. It must come from the COMMITTED partition; a prompt "
            "invented here would make the arm it belongs to unattributable."
        )
    out = {}
    for style in styles[style_set]:
        sid = style.get("id")
        if not sid:
            raise DriverError(f"{path}: a style in '{style_set}' has no 'id'.")
        prompt = style.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise DriverError(
                f"{path}: style {sid!r} in '{style_set}' has no usable 'prompt'. Every set "
                "including identity carries a committed prompt — see the file's own [identity_style]."
            )
        out[str(sid)] = style
    return out


def load_work_list(path: pathlib.Path) -> list[WorkUnit]:
    """``work.jsonl`` → units, with ``seed`` required rather than defaulted.

    `97`: *"a driver that does not set its sampler from row['seed'] must fail loudly rather than
    default."* This is that refusal, and it fires per row so the message names the row.
    """
    units: list[WorkUnit] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        row = json.loads(raw)
        missing = [k for k in ("unit", "episode", "frames", "style", "repeat", "seed") if k not in row]
        if missing:
            raise DriverError(
                f"{path}:{lineno} is missing {missing}. The work list is the seed channel and its "
                "schema is fixed by 97_transfer25_restyle.sbatch; a row without a seed cannot be "
                "run, because arm C's ten identity clips would be ten identical or ten "
                "irreproducible files and the fingerprint control would have measured nothing."
            )
        seed = row["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise DriverError(
                f"{path}:{lineno} has seed {seed!r}, which is not an int. It travels into "
                "chunk_metadata.json and is what makes a single clip reproducible from the record."
            )
        unit = str(row["unit"])
        if unit in seen:
            # Upstream requires `name` unique across a batch (api §4), and a duplicate here would
            # also make two rows write the same output directory — the second silently winning.
            raise DriverError(f"{path}:{lineno} repeats unit {unit!r}.")
        seen.add(unit)
        units.append(
            WorkUnit(
                unit=unit,
                episode=str(row["episode"]),
                frames=int(row["frames"]),
                style=str(row["style"]),
                repeat=int(row["repeat"]),
                seed=seed,
            )
        )
    if not units:
        raise DriverError(f"{path} is empty — nothing to do.")
    return units


def parse_controls(spec: str) -> list[Control]:
    """``"depth:0.5,seg:0.5"`` → controls, validated against upstream's key set.

    NO DEFAULT, and this is the second refusal in the file. `styles.toml` commits the prompts, the
    seeds and the partition, and is silent on the conditioning — so the modality and its weight are
    an uncommitted degree of freedom that decides how much geometry survives, which is the very
    thing PR-08 §6's G0b gate measures. Defaulting it here would let this script pick the number
    the gate is scored against.
    """
    controls: list[Control] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, weight = part.partition(":")
        key = key.strip()
        if key not in _CONTROL_KEYS:
            raise DriverError(
                f"--control names {key!r}, which is not a Transfer2.5 control block. "
                f"Upstream accepts {list(_CONTROL_KEYS)} (docs/transfer25-api.md §8), and its "
                'config forbids extra keys, so a typo fails at validation rather than degrading.'
            )
        if not weight.strip():
            raise DriverError(
                f"--control entry {part!r} has no weight. Write it as '{key}:<0..1>'. The weight "
                "is not cosmetic: it trades appearance change against geometry preservation, and "
                "G0b scores the geometry."
            )
        try:
            w = float(weight)
        except ValueError as exc:
            raise DriverError(f"--control entry {part!r} has a non-numeric weight.") from exc
        if not 0.0 <= w <= 1.0:
            raise DriverError(f"--control entry {part!r}: weight must be in [0.0, 1.0] (api §8).")
        controls.append(Control(key=key, weight=w))
    if not controls:
        raise DriverError(
            "--control is required and must name at least one block. Upstream's model_post_init "
            'raises "No controls provided" for a spec with none (api §8), and this script will not '
            "choose one for you: the conditioning is uncommitted in styles.toml, so a default here "
            "would be this file picking a value the G0b geometry gate is then scored against."
        )
    return controls


# --------------------------------------------------------------------------------------------
# building the payload
# --------------------------------------------------------------------------------------------


def resolve_resolution(arg: str) -> tuple[str, str]:
    """``"640x480"`` → upstream's ``(bucket, aspect)``. Ours in, theirs out."""
    if arg not in _RESOLUTION_BUCKETS:
        raise DriverError(
            f"--resolution {arg!r} is not supported. PR-08 §3 fixes 640x480 and upstream's "
            "`resolution` is a bucket key rather than a WxH string, so only sizes with a known "
            "bucket can be requested honestly (docs/transfer25-api.md §2)."
        )
    return _RESOLUTION_BUCKETS[arg]


def build_sample(
    unit: WorkUnit,
    *,
    source_root: pathlib.Path,
    episode: dict,
    style: dict,
    controls: list[Control],
    bucket: str,
) -> dict:
    """One ``InferenceArguments``-shaped dict (api §8). Required keys first, so a miss is loud."""
    video = source_root / str(episode["video"])
    if not video.is_file():
        raise DriverError(f"unit {unit.unit}: source video {video} does not exist.")
    sample: dict = {
        "name": unit.unit,
        "prompt": style["prompt"],
        "video_path": str(video),
        "seed": unit.seed,
        "resolution": bucket,
        # Default True upstream; set explicitly so the record says which behaviour ran rather than
        # inheriting whatever the pinned revision defaults to (api §2).
        "keep_input_resolution": True,
    }
    for control in controls:
        # control_path omitted ON PURPOSE ONLY WHERE THE MANIFEST HAS NOTHING TO OFFER: omitting it
        # makes Transfer2.5 estimate the map itself, with its OWN depth/segmentation models — a
        # different estimator from the isaac_binding.py annotators GEOM_TOL was measured against
        # (api §8, "the item this dissolves"). Where the manifest carries a map, it is passed, so
        # the run uses the estimator the geometry budget characterises.
        block: dict = {"control_weight": control.weight}
        supplied = episode.get({"depth": "depth", "seg": "segmentation"}.get(control.key, ""))
        if supplied:
            path = source_root / str(supplied)
            if not path.is_file():
                raise DriverError(
                    f"unit {unit.unit}: manifest names {control.key} map {path}, which is missing. "
                    "Refusing to silently fall back to the framework's own estimator — that would "
                    "swap the estimator GEOM_TOL was measured against for a different one."
                )
            block["control_path"] = str(path)
        sample[control.key] = block
    return sample


# --------------------------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------------------------


def _null_backend(sample: dict, out_dir: pathlib.Path) -> dict:
    """Write a deterministic placeholder clip. No model, no GPU, no checkout.

    Exercises everything around the model call. The bytes are a function of the sample only, so a
    rerun with the same seed produces the same file and the seed channel is testable end to end —
    which is the property arm C actually depends on.

    IT IS A REAL MP4, at the source's exact frame count and geometry, and that is a requirement
    rather than a nicety. PR-08 §6's G0c composites the real robot back over every generated frame,
    unconditionally, and a composite needs a clip it can decode and a clip that pairs frame-for-frame
    with its source. Writing a text blob here and exempting it from the composite would have created
    exactly what G0c must not have: a reachable code path that skips compositing. So the placeholder
    decodes, and the composite runs on it like everything else.

    The content is flat colour fields taken from the digest — visibly not a restyle, so nothing
    downstream can mistake a placeholder for one even though it is now a valid video. libx264 is
    byte-deterministic for identical input and parameters, which is what keeps the seed assertion
    above true.
    """
    import hashlib

    payload = json.dumps(sample, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()

    source = robot_composite.decode_clip(pathlib.Path(sample["video_path"]))
    seed_bytes = np.frombuffer(bytes.fromhex(digest), dtype=np.uint8)
    frames = np.empty_like(source)
    for index in range(source.shape[0]):
        for channel in range(3):
            frames[index, :, :, channel] = seed_bytes[(index * 3 + channel) % seed_bytes.size]
    fps = robot_composite.container_fps(pathlib.Path(sample["video_path"])) or 30.0
    robot_composite.encode_clip(frames, out_dir / "vision.mp4", fps)
    return {"backend": "null", "digest": f"sha256:{digest}"}


def _transfer25_backend(sample: dict, out_dir: pathlib.Path, setup: dict) -> dict:
    """Call the real framework in-process and move its flat output into our per-unit layout.

    In-process rather than subprocess because upstream exposes the class `examples/inference.py`
    itself imports (api §9), so there is no reason to pay a process and a CLI-parse per clip.
    """
    from cosmos_transfer2.config import InferenceArguments, SetupArguments  # type: ignore
    from cosmos_transfer2.inference import Control2WorldInference  # type: ignore

    # batch_hint_keys is normally computed by from_files(); with one sample per call it is just
    # this sample's control keys (api §9).
    hint_keys = sorted(k for k in _CONTROL_KEYS if k in sample)

    # `model` LOOKS optional -- SetupArguments declares it with a default (config.py:305). It is
    # not. validate_model is a mode="before" validator, so it runs on the raw dict before pydantic
    # applies any default, and an absent key raises "model is required" (config.py:263-270). Job
    # 189142 died on exactly this after loading the checkpointer, and because the driver reports a
    # dead unit as an error rather than a non-zero exit, the sbatch above it timed the crash.
    #
    # WHICH name to pass depends on how many controls there are, and the two cases are not alike:
    #   one  control key  -- upstream keeps our --checkpoint-path (inference.py:52-62) and `model`
    #                        is what picks the variant, so it has to BE that key.
    #   many control keys -- upstream takes the multi-branch branch (inference.py:64-72). `model`
    #                        there only decides `.distilled`, which is False either way.
    # hint_keys[0] is exact in the first case and inert in the second.
    args = SetupArguments(**{**setup, "model": hint_keys[0]})
    parsed = InferenceArguments(**sample)
    inference = Control2WorldInference(args, hint_keys)
    produced = inference.generate([parsed], out_dir)

    # generate() SKIPS guardrail-blocked samples rather than raising (api §9), so an empty list is
    # a silent failure and must not be read as success.
    if not produced:
        raise DriverError(
            f"unit {sample['name']}: the framework returned no output path. With guardrails "
            "disabled this should not happen; it is reported as an error rather than harvested."
        )
    flat = out_dir / f"{sample['name']}.mp4"
    if not flat.is_file():
        raise DriverError(f"unit {sample['name']}: expected {flat}, which was not written.")
    flat.replace(out_dir / "vision.mp4")
    # The checkpoints upstream ACTUALLY loaded, read back off the object rather than restated from
    # what we passed in. With more than one control key `--checkpoint-path` is not consulted at all
    # (inference.py:64-72) and every entry here comes from upstream's own registry, at the revision
    # pinned in checkpoints_transfer2.py -- which is NOT ${TRANSFER_MODEL_REVISION}. PR-08 §6 wants
    # the generator recorded; this is the only place that can record the one that ran.
    return {
        "backend": "transfer25",
        "produced": [str(p) for p in produced],
        "checkpoints_loaded": [str(c) for c in inference.checkpoint_list],
        "checkpoint_path_honoured": len(hint_keys) == 1,
    }


# --------------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------------


#: Where an uncomposited model output is put when the G0c composite refuses it. Named rather than
#: deleted so the failure is inspectable, and renamed rather than left in place because the harvest
#: in 97_transfer25_restyle.sbatch keys on a file called ``vision.mp4`` existing.
UNCOMPOSITED_QUARANTINE = "vision.uncomposited.mp4"


def _quarantine_uncomposited(out_dir: pathlib.Path) -> str | None:
    """Make sure no file called ``vision.mp4`` survives a unit that did not finish compositing.

    The harvest files a clip when ``vision.mp4`` exists AND the status is ``success``. Those are two
    independent conditions and this breaks the first as well, on purpose: PR-08 §6 G0c's claim is
    that the generic-manipulator defect *cannot enter*, and a claim that rests on one status field
    being read correctly by every future consumer is weaker than one that rests on the file not
    being there. The bytes are kept under a name nothing looks for, because "the composite refused
    this clip" is a fact about the generator worth being able to look at.
    """
    video = out_dir / "vision.mp4"
    if not video.exists():
        return None
    quarantined = out_dir / UNCOMPOSITED_QUARANTINE
    video.replace(quarantined)
    return str(quarantined)


def run_unit(
    unit: WorkUnit,
    sample: dict,
    out_root: pathlib.Path,
    backend: str,
    setup: dict,
    composite: "robot_composite.CompositeContext",
) -> Outcome:
    """One unit, isolated. Composites, then writes ``sample_outputs.json`` LAST.

    The ordering is `97`'s requirement and it is load-bearing for the harvest: upstream writes its
    own args sidecar BEFORE generation and before the guardrail check (api §5), so file presence
    proves the job was attempted, not that a video exists. 96 reads a status this driver wrote
    after looking at the mp4, which is a different claim.

    G0c SITS BETWEEN THE BACKEND AND THE STATUS, AND ``composite`` IS A REQUIRED ARGUMENT. Not
    optional with a ``None`` default, not read from a flag, not skipped for one backend: a caller
    that has no context cannot call this function at all, and the only thing that builds a context is
    ``robot_composite.build_context``, which has no way to build one that does not composite. The
    three lines are therefore the whole of the guarantee — success implies composited, an exception
    implies quarantined, and there is no fourth outcome.
    """
    out_dir = out_root / unit.unit
    out_dir.mkdir(parents=True, exist_ok=True)
    record = out_dir / "sample_outputs.json"
    # A stale success from an earlier attempt would be harvested as finished work before this run
    # had a chance to overwrite it, so the claim is withdrawn before the work starts.
    record.unlink(missing_ok=True)
    # Likewise a quarantined clip from an earlier attempt: it says "this unit's composite refused"
    # and it is about to stop being true one way or the other.
    (out_dir / UNCOMPOSITED_QUARANTINE).unlink(missing_ok=True)

    try:
        extra = (
            _null_backend(sample, out_dir)
            if backend == "null"
            else _transfer25_backend(sample, out_dir, setup)
        )
        video = out_dir / "vision.mp4"
        if not video.is_file() or video.stat().st_size == 0:
            raise DriverError(f"unit {unit.unit}: vision.mp4 missing or empty after the backend ran.")
        extra["g0c"] = composite.composite(
            source_video=pathlib.Path(sample["video_path"]),
            generated_video=video,
            expected_frames=unit.frames,
        )
        outcome = Outcome(unit=unit.unit, status="success", extra=extra)
    except Exception as exc:  # noqa: BLE001 — per-unit isolation is the point; see the module docstring
        # Deliberately broad. Upstream's generate() has no try/except and `keep_going` covers only
        # guardrail blocks (api §6), so one unreadable video would otherwise take the chunk with it.
        quarantined = _quarantine_uncomposited(out_dir)
        outcome = Outcome(
            unit=unit.unit,
            status="error",
            detail=f"{type(exc).__name__}: {exc}",
            extra={
                "traceback": traceback.format_exc(limit=8),
                "g0c": {
                    "composited": False,
                    "uncomposited_output_quarantined_to": quarantined,
                    "note": (
                        "PR-08 §6 G0c: this unit produced no composited clip, so it produced no "
                        "clip. Anything the backend wrote has been renamed out of the harvest's "
                        "way."
                    ),
                },
            },
        )

    record.write_text(
        json.dumps(
            {
                "status": outcome.status,
                "unit": unit.unit,
                "episode": unit.episode,
                "style": unit.style,
                "repeat": unit.repeat,
                "seed": unit.seed,
                "frames": unit.frames,
                "detail": outcome.detail,
                **outcome.extra,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return outcome


def shard(units: list[WorkUnit], nproc: int) -> list[WorkUnit]:
    """Take this rank's share when launched under torchrun.

    Upstream loads the model once and runs a batch SEQUENTIALLY; its multi-GPU story is torchrun
    (api §4). So ``--nproc > 1`` means "I am one of N ranks", and the refusal below exists because
    an unsharded run under torchrun would generate every clip N times, each rank overwriting the
    others — N× the GPU-hours for one corpus, and PARTITION_CEILING_GPU_H would be blown by a
    factor nobody could see in the output.
    """
    if nproc <= 1:
        return units
    try:
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    except KeyError as exc:
        raise DriverError(
            f"--nproc {nproc} but {exc.args[0]} is not set. Upstream runs a batch sequentially in "
            "one process; parallelism is torchrun, so this driver must be launched under it and "
            "shards by rank. Running it unsharded would generate every clip once per rank."
        ) from exc
    if world != nproc:
        raise DriverError(f"--nproc {nproc} disagrees with WORLD_SIZE={world}.")
    return [u for i, u in enumerate(units) if i % world == rank]


def build_parser() -> argparse.ArgumentParser:
    """The driver's whole command-line surface, in one place a test can enumerate.

    Factored out of ``main`` for exactly one reason: PR-08 §6 G0c says the composite is
    *unconditional*, and the cheapest way for that to stop being true is for somebody to add an
    ``--skip-composite`` in six months for a debugging session and leave it in. A test walks these
    actions and asserts that no option controls whether compositing happens — the three G0c options
    below choose where its inputs come from and how often the DIAGNOSTIC is sampled, and none of
    them can turn it off.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-path", required=True)
    ap.add_argument("--manifest", required=True, type=pathlib.Path)
    ap.add_argument("--styles", required=True, type=pathlib.Path)
    ap.add_argument("--style-set", required=True)
    ap.add_argument("--work-list", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--resolution", default="640x480")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument(
        "--require-success",
        action="store_true",
        help=(
            "Exit non-zero if any unit failed. OFF by default and it must stay off for the chunked "
            "run, whose whole resume story is that a partial chunk exits 0 and is re-driven. It is "
            "for the TIMING path, where the opposite is true: a unit that died is not a slow unit, "
            "and a wall clock measured around it is not a measurement."
        ),
    )
    ap.add_argument(
        "--control",
        required=True,
        help="e.g. 'depth:0.5,seg:0.5'. Required and never defaulted — see parse_controls().",
    )
    ap.add_argument(
        "--no-guardrails",
        action="store_true",
        help="OUR spelling; upstream's is --disable-guardrails. Mandatory on Cosmos inference in "
        "this project: the RetinaFace postprocessor writes blurred pixels back into the frame and "
        "those are the frames that reach disk (docs/transfer25-api.md §3).",
    )
    ap.add_argument("--backend", choices=("transfer25", "null"), default="transfer25")
    ap.add_argument(
        "--robot-mask-area-bound",
        type=pathlib.Path,
        default=robot_composite.AREA_BOUND_ARTIFACT,
        help=(
            "The COMMITTED artifact carrying the largest fraction of a frame a robot mask may cover "
            "(PR-08 §6 G0c). Defaults to the tracked configs/transfer25/pr08_robot_mask_area.json. "
            "It is a path and never a number: this driver will not take a bound on the command "
            "line, because a bound typed at submit time is a threshold nobody committed. The file's "
            "sha256 lands in every unit's record."
        ),
    )
    ap.add_argument(
        "--mask-cache",
        type=pathlib.Path,
        default=None,
        help=(
            "Where the per-source-episode robot masks are cached. The mask is a property of the "
            "SOURCE frame, so it is identical across all 25 restyles of an episode and computing it "
            "once instead of 25 times is the difference between ~172k segmentations and ~4.3M. "
            "Defaults to a 'robot_masks' directory beside --out; point it above the chunk "
            "directories to share across style sets. Entries are keyed on the source bytes, the "
            "prompt and the estimator version, so nothing stale can be reused."
        ),
    )
    ap.add_argument(
        "--iou-stride",
        type=int,
        default=10,
        help=(
            "Sample the robot-mask IoU diagnostic every Nth frame. This is a sampling rate for a "
            "number PR-08 §6 says twice is a DIAGNOSTIC ON THE GENERATOR AND NEVER A GATE, so it "
            "cannot become a finding. THE COMPOSITE HAS NO STRIDE and this flag does not touch it."
        ),
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if not args.no_guardrails:
            raise DriverError(
                "--no-guardrails is mandatory. The guardrail's RetinaFace postprocessor rewrites "
                "frames and blurs the hand, and the rewritten frames are what land on disk "
                "(docs/transfer25-api.md §3) — restyling to study geometry and then blurring the "
                "manipulator edits the evidence. It also avoids the gated nvidia/Cosmos-Guardrail1 "
                "download, whose licence is the account holder's to accept and not an agent's."
            )
        bucket, aspect = resolve_resolution(args.resolution)
        controls = parse_controls(args.control)
        episodes = load_manifest(args.manifest)
        styles = load_styles(args.styles, args.style_set)
        units = shard(load_work_list(args.work_list), args.nproc)
        source_root = args.manifest.parent
        args.out.mkdir(parents=True, exist_ok=True)

        setup = {
            "output_dir": args.out,
            "disable_guardrails": True,   # our --no-guardrails, translated (api §3)
            "checkpoint_path": args.checkpoint_path,
        }

        # PR-08 §6 G0c, resolved BEFORE the first unit and not negotiable afterwards. Two things
        # are decided here and both are run-level facts rather than per-unit ones: whether the
        # committed area bound exists (it is a refusal if not, and the message says what to measure)
        # and whether the pinned GroundingDINO + SAM 2 checkpoints are staged. Discovering a missing
        # checkpoint inside the per-unit guard would turn one honest refusal into N identical
        # per-unit errors that read like a flaky generator, spend a pass of the chunk's rail, and
        # send the operator to look at the wrong thing.
        try:
            composite = robot_composite.build_context(
                # Not an input to the composite: the corpus the committed area bound is checked
                # AGAINST. A bound is a claim about a distribution over these episodes, and the
                # loader refuses one measured over any other manifest.
                source_manifest=args.manifest,
                area_bound_path=args.robot_mask_area_bound,
                iou_stride=args.iou_stride,
                cache_dir=args.mask_cache or (args.out.parent / "robot_masks"),
            )
        except robot_composite.CompositeError as exc:
            # Translated rather than propagated: `main`'s contract with 97 is "FATAL: <reason>",
            # exit 1, and a CompositeError escaping here would be a traceback the sbatch reads as a
            # crash of the driver rather than as a refusal it can act on.
            raise DriverError(str(exc)) from exc

        print(
            f"=== restyle: {len(units)} units | set={args.style_set} | "
            f"controls={','.join(f'{c.key}:{c.weight}' for c in controls)} | "
            f"resolution {args.resolution} -> bucket {bucket!r} aspect {aspect!r} | "
            f"backend={args.backend}",
            flush=True,
        )
        print(
            "=== G0c: robot pixels composited back on EVERY frame of EVERY clip | "
            f"mask={composite.masker.provenance()['name']} prompt={robot_composite.ROBOT_TEXT_PROMPT!r} | "
            f"area bound {composite.bound.max_frame_fraction} from {composite.bound.artifact} | "
            "IoU recorded as a DIAGNOSTIC, never a gate",
            flush=True,
        )

        failures = 0
        for i, unit in enumerate(units, start=1):
            episode = episodes.get(unit.episode)
            style = styles.get(unit.style)
            if episode is None or style is None:
                missing = "episode" if episode is None else "style"
                # A refusal, not a per-unit error: the work list disagreeing with the manifest or
                # the committed partition means the inputs do not describe one experiment, and
                # every remaining unit is suspect for the same reason.
                raise DriverError(
                    f"unit {unit.unit} names {missing} "
                    f"{unit.episode if episode is None else unit.style!r}, which is not in "
                    f"{args.manifest if episode is None else args.styles}."
                )
            sample = build_sample(
                unit,
                source_root=source_root,
                episode=episode,
                style=style,
                controls=controls,
                bucket=bucket,
            )
            outcome = run_unit(unit, sample, args.out, args.backend, setup, composite)
            failures += outcome.status != "success"
            print(f"[{i}/{len(units)}] {unit.unit} {outcome.status} {outcome.detail}", flush=True)

        print(f"=== done: {len(units) - failures} success, {failures} error", flush=True)
        if failures and args.require_success:
            print(
                f"FATAL: --require-success and {failures} of {len(units)} units failed. "
                "Read the per-unit sample_outputs.json for the reason.",
                file=sys.stderr,
            )
            return 1
        # A non-zero exit on any failure would make 96's resume impossible — the whole point of the
        # per-unit status file is that a partial chunk is resumable. The count is reported and the
        # harvest decides. --require-success is the one caller that needs the opposite.
        return 0
    except DriverError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
