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

AND G0c IS ASKED FIRST, NOT ONLY LAST. The robot mask is a property of the SOURCE frame, so every
refusal ``robot_composite.check_mask`` can produce is knowable before the generator is called;
until :func:`preflight_source_masks` landed, none of them were, and a unit G0c was always going to
refuse was refused only after its clip had been generated in full. That is not a hypothetical
ordering nit: 385 of the corpus's 402 episodes are refused by one half of that check or the other
(``runs/pr08-robot-mask-area/POOLED.json``), the episode the TIMING=1 path times is one of them,
and each attempt bought an H200 slot and no THROUGHPUT.json. **The check itself is unchanged and
the post-composite call is still what decides.** The preflight cannot make a clip pass — it runs
the same function, over the same masks, against the same bound, in the same frame order, so what it
refuses is a subset of what the composite refuses. It moves the discovery, never the decision.

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


class SourceMaskRefusal(robot_composite.CompositeError):
    """G0c refused this unit on its SOURCE masks, BEFORE the generator was called.

    A subclass rather than a new error type or a flag, so that nothing that already handles the
    refusal has to learn about it: ``run_unit``'s per-unit guard catches ``Exception``, ``main``
    translates ``robot_composite.CompositeError`` into a driver refusal, and every existing caller
    that catches ``CompositeError`` catches this too. What the subclass buys is a NAME in the
    per-unit record — ``detail`` reads ``SourceMaskRefusal: …`` — because "G0c refused this clip"
    and "G0c refused this clip after we paid to generate it" are the same verdict at prices that
    differ by half a GPU-hour, and only one of them is worth an operator's attention.
    """


@dataclass
class SourceMaskMemo:
    """Which source episodes THIS RUN has already watched G0c refuse, keyed so it cannot lie.

    Every style-instance of an episode revisits the same source, and the robot mask is a property
    of the SOURCE frame — identical across every restyle of it, which is the whole argument for
    :class:`robot_composite.MaskCache`. Without a memo the units that share one refused episode each
    re-decode the source, each re-read ~16 MB of packed masks and each re-walk every frame through
    ``check_mask``, to reach the identical refusal once per unit. The mask cache makes the second
    discovery cheap; it does not make it free, and it cannot make it zero.

    HOW MANY THAT IS, COUNTED FOR ONE INVOCATION RATHER THAN FOR THE STAGE. This memo's lifetime is
    one process, and one invocation of this driver generates exactly one ``--style-set``, so what it
    can save is the repeats WITHIN a set: four per episode in `T40_RULE_V11` §2's stage 1 (whose 8
    style-instances are 4 train plus 4 matched identity repeats, submitted as two jobs), and ten per
    episode per set in the full 25-instance rendering. Stated that way and not as "eight", because a
    per-run memo cannot save anything across two submissions and a docstring that implied it could
    would be describing a cache this class is not.

    THE KEY IS THE PREDICATE'S OWN INPUTS AND NOTHING ELSE. ``check_mask``'s verdict on an episode
    is a function of exactly three things: the source bytes, the segmenter that turns them into
    masks, and the bound the covered fraction is compared against. The first two are precisely what
    ``MaskCache.key`` hashes — reused here rather than restated, so a segmenter re-pin can never
    invalidate the mask cache while leaving this memo standing — and the third is added on top,
    because ``MaskCache`` has no reason to know about a bound and half of ``check_mask`` is that
    bound. A memo keyed on the episode id instead would be a memo that survives exactly the changes
    it must not survive.

    TWO PROPERTIES THIS MUST HAVE, AND HAS BY CONSTRUCTION:

    * **It never lets an episode pass.** Only refusals are stored. A unit whose source masks cleared
      the check is checked again on the next style-instance, at the cost of a cache read, because a
      memo of PASSES would be a way past the gate the moment any of its three inputs changed in a
      way the key failed to capture. Storing only the refusals means the worst a stale entry can do
      is refuse a unit that G0c also refuses.
    * **It never skips an episode silently.** A memo hit is not a ``continue``: the unit still goes
      through ``run_unit``, which withdraws any stale ``sample_outputs.json`` from an earlier pass
      and writes an ``error`` record carrying the remembered reason. A unit skipped without that
      record would leave an earlier attempt's ``success`` on disk for the harvest to file. The hit
      is also counted and printed, per unit and again in the run's closing line.
    """

    #: memo key → the refusal message ``check_mask`` produced the first time it was seen.
    refusals: dict[str, str] = field(default_factory=dict)
    #: Units refused straight out of the memo, i.e. without a mask pass of their own.
    hits: int = 0
    #: Units refused by an actual mask pass, each of which put one entry in ``refusals``.
    misses: int = 0

    @staticmethod
    def key(source_video: pathlib.Path, composite: "robot_composite.CompositeContext") -> str:
        """The three inputs ``check_mask``'s verdict is a function of, hashed the way G0c hashes."""
        return json.dumps(
            {
                # The source bytes AND the segmenter identity, from the one definition of that pair
                # in the codebase. Sharing it with the mask cache is deliberate: two keys over the
                # same facts drift, and the drift is silent.
                "masks": robot_composite.MaskCache.key(source_video, composite.masker.provenance()),
                # The other half of the predicate. Both fields, not just the number: a bound file
                # re-measured to the same fraction under a different distribution is a different
                # claim, and the sha256 is what says so.
                "bound": [composite.bound.max_frame_fraction, composite.bound.artifact_sha256],
            },
            sort_keys=True,
        )

    def recall(self, key: str) -> str | None:
        return self.refusals.get(key)

    def remember(self, key: str, message: str) -> None:
        self.refusals[key] = message

    def summary(self, units: int) -> str:
        refused = self.hits + self.misses
        return (
            f"=== G0c source-mask preflight: {refused} of {units} units refused BEFORE generation "
            f"({len(self.refusals)} distinct sources; {self.hits} of the refusals were served from "
            "this run's memo and cost no mask pass)"
        )


def preflight_source_masks(
    unit: WorkUnit,
    sample: dict,
    composite: "robot_composite.CompositeContext",
    memo: SourceMaskMemo | None = None,
) -> dict:
    """Run G0c's own per-frame check over the SOURCE masks before the backend is called.

    WHY THIS EXISTS. ``composite_clip`` takes its masks from the SOURCE video
    (``robot_composite.source_masks``), so every refusal ``check_mask`` can produce is knowable
    before a single GPU-second of generation is spent — and until this function existed, none of
    them were. The order was: generate the clip, decode it, decode the source, segment the source,
    and only then refuse on frame 0. Measured against the corpus that is not a rare path: 385 of 402
    episodes are refused by one half of ``check_mask`` or the other
    (``runs/pr08-robot-mask-area/POOLED.json``), and the episode the TIMING=1 path times is one of
    them — it burns an H200 slot generating 590 frames to reproduce a refusal two committed
    artifacts already predict, and writes no THROUGHPUT.json.

    WHAT THIS IS NOT. It does not weaken, bypass, waive or shortcut G0c. ``composite_clip``'s
    post-composite check is untouched and still runs over every frame of every clip; nothing here
    can make a clip pass that would otherwise fail. This moves the DISCOVERY earlier. The decision
    is exactly where it was.

    WHY IT CANNOT DISAGREE WITH THE CHECK IT ANTICIPATES. The predicate is not re-implemented: this
    calls ``robot_composite.check_mask`` itself, with ``composite.bound`` — the same frozen bound
    object the composite will use — over masks obtained from ``robot_composite.source_masks`` for
    the same source path, which is the same call ``composite_clip`` makes and which goes through the
    same :class:`robot_composite.MaskCache`. Frames are walked from 0 upward in the same order, so
    even the frame the refusal names is the same. The one difference is the ``source=`` string in
    the message, which no predicate reads: ``composite_clip`` names the generated clip, which does
    not exist yet here, so this names the source. The refusals this raises are therefore a SUBSET of
    the refusals ``composite_clip`` raises — a strict subset, because ``composite_clip`` also
    refuses on frame-count disagreements it checks before ever reaching ``check_mask``, and those
    are deliberately not duplicated here rather than re-stated in a second place where they could
    drift.

    AND IT DOES NOT MOVE THE THROUGHPUT MEASURAND, WITH ONE COST NAMED RATHER THAN HIDDEN. The
    masks are computed inside the driver, inside the window ``97_transfer25_restyle.sbatch`` times,
    and cached; the composite that follows a passing preflight reads them back instead of computing
    them. The number of source-mask passes per timed episode — the expensive half, a segmentation
    per frame — is one before this change and one after it. What is genuinely new is ONE EXTRA
    DECODE of the source clip per unit, because ``source_masks`` takes frames and there is no way to
    hand ``composite_clip`` the array this function already holds. That is seconds against a
    generation measured in minutes, it lands inside the timed window rather than outside it, and it
    moves the derived GPU-h ceiling in the conservative direction. It is written down here because a
    timing artifact whose measurand moved silently is the defect this whole file's timing path
    exists to avoid. (With ``cache=None`` — which the driver never constructs, since
    ``--mask-cache`` always resolves to a path — the mask pass would be paid twice as well, and the
    cost is accepted there in exchange for the refusal being cheap.)

    AND IT INHERITS PR-08 V9's OBJECT-FILTER COUNTERS, BECAUSE MOVING THE MASK PASS MOVED THEM.
    ``composite_clip`` differences ``masker.filter_counters`` around its own ``source_masks`` call
    and writes the delta into every clip's record, and ``robot_composite`` states the reason in one
    line: *"a filter whose firing is not recorded cannot be told apart from a corpus that never
    triggered it."* Once this function computes the masks first, that call is a CACHE HIT for every
    clip on the driver's path, the delta is zero for every clip, and the whole corpus's record says
    nothing about whether the apple-drop filter ever fired — including
    ``frames_emptied_by_the_filter``, which is the counter that distinguishes "the segmenter found
    no robot" from "the filter removed the only detection", the exact question `T40_RULE_V12`'s
    empty-mask semantics turn on. ``composite_clip``'s note already says all-zero-with-cache means
    "not measured here" rather than "never fired", so nothing downstream is made to lie; the number
    simply stopped existing. So it is differenced HERE, around the one pass that actually runs the
    masker, and recorded in this function's own block. It is the same arithmetic over the same
    counters — nothing is re-implemented and nothing can be double-counted, because AT MOST one of
    the two blocks is non-zero for a given clip and each says whether its own masks came from the
    cache. Both read zero when the masks were already cached before this unit started, which is the
    honest answer: this run did not run the filter on that episode, an earlier one did.
    """
    source_video = pathlib.Path(sample["video_path"])
    key = SourceMaskMemo.key(source_video, composite) if memo is not None else None

    if memo is not None and key is not None:
        remembered = memo.recall(key)
        if remembered is not None:
            memo.hits += 1
            raise SourceMaskRefusal(
                f"unit {unit.unit}: {remembered}\n"
                "       (Remembered from an earlier unit of this run over the same source, the same "
                "segmenter and the same bound — see SourceMaskMemo. The mask pass was not repeated; "
                "the refusal is the one that pass produced.)"
            )

    # Differenced around the mask pass and nothing else, exactly as ``composite_clip`` does it: the
    # counters are cumulative over the masker's life, so only the caller that brackets a call can
    # say what that call did. The IoU diagnostic is not in this bracket for the same reason it is
    # not in the composite's — it masks GENERATED frames, and the filter's behaviour there is a
    # different question.
    before_filter = dict(getattr(composite.masker, "filter_counters", {}) or {})
    frames = robot_composite.decode_clip(source_video)
    masks, from_cache = robot_composite.source_masks(source_video, frames, composite)
    after_filter = dict(getattr(composite.masker, "filter_counters", {}) or {})
    try:
        for index in range(masks.shape[0]):
            robot_composite.check_mask(
                masks[index],
                frame_index=index,
                bound=composite.bound,
                source=str(source_video),
            )
    except robot_composite.CompositeError as exc:
        message = (
            f"{exc}\n"
            "       REFUSED BEFORE GENERATION. The robot mask is a property of the SOURCE frame, so "
            "this verdict was reachable without the generator; PR-08 §6 G0c would have raised the "
            "same refusal after the clip was generated, and raising it here costs seconds instead "
            "of an H200 slot. Nothing has been waived: the post-composite check is unchanged and "
            "this unit produced no clip."
        )
        if memo is not None and key is not None:
            memo.remember(key, message)
            memo.misses += 1
        raise SourceMaskRefusal(f"unit {unit.unit}: {message}") from exc

    return {
        "checked": True,
        "frames_checked": int(masks.shape[0]),
        "masks_from_cache": bool(from_cache),
        "bound": composite.bound.max_frame_fraction,
        "robot_mask_object_filter": {
            "note": (
                "PR-08 V9, counted over the SOURCE frames of this clip, differenced around the "
                "mask pass this preflight runs. This is the block that carries the counts on the "
                "driver's path: the preflight computes the source masks, so the composite's own "
                "copy of these counters is differenced around a cache hit and reads all-zero with "
                "masks_from_cache true. At most one of the two blocks is non-zero for a clip, and "
                "each one's masks_from_cache says whether it was the pass that ran the filter; "
                "both read zero when the masks were cached before this unit started."
            ),
            "masks_from_cache": bool(from_cache),
            "max_iou": float(robot_composite.ROBOT_MASK_OBJECT_MAX_IOU),
            **{
                name: int(after_filter.get(name, 0) - before_filter.get(name, 0))
                for name in sorted(set(after_filter) | set(before_filter))
            },
        },
        "predicate": (
            "robot_composite.check_mask over robot_composite.source_masks — the same function over "
            "the same masks the post-composite check runs, called before the backend"
        ),
    }


def run_unit(
    unit: WorkUnit,
    sample: dict,
    out_root: pathlib.Path,
    backend: str,
    setup: dict,
    composite: "robot_composite.CompositeContext",
    memo: SourceMaskMemo | None = None,
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

    G0c ALSO SITS BEFORE THE BACKEND NOW, and that is an addition to the order rather than a change
    to it. :func:`preflight_source_masks` runs the same ``check_mask`` over the same source masks
    first, so a unit G0c will refuse is refused before the generator is called. It is inside this
    guard, not above the loop, because it is a per-unit fact: one episode's masks say nothing about
    the next episode's, and a refusal that killed the chunk would take 401 innocent episodes with
    it. The post-composite call below is unchanged and is still what decides.
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
        # BEFORE the backend, deliberately and as the first thing this unit does. See
        # preflight_source_masks: the masks come from the SOURCE, so every refusal check_mask can
        # produce is knowable here, and discovering it after generation is how the TIMING=1 path
        # came to spend an H200 slot per attempt to learn something two committed artifacts already
        # said.
        preflight = preflight_source_masks(unit, sample, composite, memo)
        extra = (
            _null_backend(sample, out_dir)
            if backend == "null"
            else _transfer25_backend(sample, out_dir, setup)
        )
        video = out_dir / "vision.mp4"
        if not video.is_file() or video.stat().st_size == 0:
            raise DriverError(f"unit {unit.unit}: vision.mp4 missing or empty after the backend ran.")
        # Recorded, not merely done: a reader of one clip's record can then see that its source
        # masks were checked before it was generated as well as after it was composited.
        extra["g0c_source_mask_preflight"] = preflight
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
                    # True when the source-mask preflight refused this unit, i.e. when no generation
                    # was paid for. It is a fact about the price of this refusal and it belongs in
                    # the record: without it, "0 success, N error" reads the same whether the chunk
                    # spent an hour of H200 time or four seconds.
                    "refused_before_generation": isinstance(exc, SourceMaskRefusal),
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

        # One memo per RUN, built here so its lifetime is exactly this invocation's. It is not
        # persisted to disk on purpose: a memo that outlived the process would have to be
        # invalidated by everything that invalidates its key, and the file that already solves that
        # problem correctly is the mask cache. Within one invocation the eight style-instances of
        # stage 1 that share an episode are the case that matters, and they are all in this loop.
        memo = SourceMaskMemo()
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
            outcome = run_unit(unit, sample, args.out, args.backend, setup, composite, memo)
            failures += outcome.status != "success"
            print(f"[{i}/{len(units)}] {unit.unit} {outcome.status} {outcome.detail}", flush=True)

        # Printed unconditionally, including as a line of zeros. The count of units G0c refused
        # before they were generated is the difference between a chunk that spent its GPU-hours and
        # one that did not, and a number that only appears when it is non-zero is a number nobody
        # can compare across runs.
        print(memo.summary(len(units)), flush=True)
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
