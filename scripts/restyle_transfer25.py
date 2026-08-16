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
work-list contract, the per-unit isolation and the output layout without a GPU, a checkout or a
weight. That is what the tests run, and it is the only backend that will work on this workstation.
A null-backend clip is stamped `"backend": "null"` in its own status file and `screen_corpus.py`
would reject it; nothing downstream can mistake one for a restyle.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import traceback
from dataclasses import dataclass, field

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
    """Write a deterministic placeholder. No model, no GPU, no checkout.

    Exercises everything around the model call. The bytes are a function of the sample only, so a
    rerun with the same seed produces the same file and the seed channel is testable end to end —
    which is the property arm C actually depends on.
    """
    import hashlib

    payload = json.dumps(sample, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    (out_dir / "vision.mp4").write_bytes(b"NULLBACKEND\n" + digest.encode("ascii") + b"\n")
    return {"backend": "null", "digest": f"sha256:{digest}"}


def _transfer25_backend(sample: dict, out_dir: pathlib.Path, setup: dict) -> dict:
    """Call the real framework in-process and move its flat output into our per-unit layout.

    In-process rather than subprocess because upstream exposes the class `examples/inference.py`
    itself imports (api §9), so there is no reason to pay a process and a CLI-parse per clip.
    """
    from cosmos_transfer2.config import InferenceArguments, SetupArguments  # type: ignore
    from cosmos_transfer2.inference import Control2WorldInference  # type: ignore

    args = SetupArguments(**setup)
    parsed = InferenceArguments(**sample)
    # batch_hint_keys is normally computed by from_files(); with one sample per call it is just
    # this sample's control keys (api §9).
    hint_keys = sorted(k for k in _CONTROL_KEYS if k in sample)
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
    return {"backend": "transfer25", "produced": [str(p) for p in produced]}


# --------------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------------


def run_unit(unit: WorkUnit, sample: dict, out_root: pathlib.Path, backend: str, setup: dict) -> Outcome:
    """One unit, isolated. Writes ``sample_outputs.json`` LAST and only after asserting the mp4.

    The ordering is `97`'s requirement and it is load-bearing for the harvest: upstream writes its
    own args sidecar BEFORE generation and before the guardrail check (api §5), so file presence
    proves the job was attempted, not that a video exists. 96 reads a status this driver wrote
    after looking at the mp4, which is a different claim.
    """
    out_dir = out_root / unit.unit
    out_dir.mkdir(parents=True, exist_ok=True)
    record = out_dir / "sample_outputs.json"
    # A stale success from an earlier attempt would be harvested as finished work before this run
    # had a chance to overwrite it, so the claim is withdrawn before the work starts.
    record.unlink(missing_ok=True)

    try:
        extra = (
            _null_backend(sample, out_dir)
            if backend == "null"
            else _transfer25_backend(sample, out_dir, setup)
        )
        video = out_dir / "vision.mp4"
        if not video.is_file() or video.stat().st_size == 0:
            raise DriverError(f"unit {unit.unit}: vision.mp4 missing or empty after the backend ran.")
        outcome = Outcome(unit=unit.unit, status="success", extra=extra)
    except Exception as exc:  # noqa: BLE001 — per-unit isolation is the point; see the module docstring
        # Deliberately broad. Upstream's generate() has no try/except and `keep_going` covers only
        # guardrail blocks (api §6), so one unreadable video would otherwise take the chunk with it.
        outcome = Outcome(
            unit=unit.unit,
            status="error",
            detail=f"{type(exc).__name__}: {exc}",
            extra={"traceback": traceback.format_exc(limit=8)},
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


def main(argv: list[str] | None = None) -> int:
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
    args = ap.parse_args(argv)

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

        print(
            f"=== restyle: {len(units)} units | set={args.style_set} | "
            f"controls={','.join(f'{c.key}:{c.weight}' for c in controls)} | "
            f"resolution {args.resolution} -> bucket {bucket!r} aspect {aspect!r} | "
            f"backend={args.backend}",
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
            outcome = run_unit(unit, sample, args.out, args.backend, setup)
            failures += outcome.status != "success"
            print(f"[{i}/{len(units)}] {unit.unit} {outcome.status} {outcome.detail}", flush=True)

        print(f"=== done: {len(units) - failures} success, {failures} error", flush=True)
        # A non-zero exit on any failure would make 96's resume impossible — the whole point of the
        # per-unit status file is that a partial chunk is resumable. The count is reported and the
        # harvest decides.
        return 0
    except DriverError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
