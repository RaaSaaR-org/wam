#!/usr/bin/env python3
"""Take the review page's saved verdicts and put them where the two open items read them.

WHY THIS EXISTS
---------------
``scripts/build_review_page.py`` produces the page a person actually reviews on, and the page saves
their verdicts into itself. That leaves the verdicts in an HTML file, which is where nothing looks
for them. This moves them into the two artifacts that are the evidence:

- the apple-mask verdicts, through ``scripts/record_mask_audit_verdicts.py`` -- **not** around it.
  That tool owns the frame-to-sheet mapping, the coverage arithmetic and the refusals, and a second
  implementation of any of them is how a verdict gets attached to a frame nobody saw;
- the area-tail verdicts, into ``TAIL_VERDICTS.json`` beside the sample they were given on, because
  ``T40_RULE_V13`` §3.2 asks what the frames above the bound *were* and this is the answer.

WHAT IT REFUSES
---------------
A key the page carries that neither source names. A verdict outside the vocabulary of its own
section -- the two sections ask different questions and swapping their answers would be silent.
An empty reviewer. It writes no bound and discharges nothing.

THE COMPRESSION IS NAMED RATHER THAN HIDDEN
-------------------------------------------
The page holds one verdict per tile. ``record_mask_audit_verdicts.py`` takes a per-sheet default
plus exceptions, which is the unit a reviewer works in. Converting the first into the second is
lossless -- the majority verdict of a sheet becomes its default and every other tile becomes an
exception -- but it is a conversion, so ``REVIEW_PAGE_INGEST.json`` records the page's own per-tile
verdicts alongside the defaults derived from them. A reader who distrusts the derivation can
recompute it.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import html
import importlib.util
import json
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

STATE_SCRIPT = re.compile(
    r'<script type="application/json" id="wam-state">(.*?)</script>', re.S
)

TAIL_VERDICTS = ("arm", "table", "mixed", "undecidable")


class IngestError(RuntimeError):
    """Refuse loudly. A verdict on the wrong frame is worse than no verdict."""


def load_recorder():
    path = REPO_ROOT / "scripts" / "record_mask_audit_verdicts.py"
    spec = importlib.util.spec_from_file_location("_rmv", path)
    if spec is None or spec.loader is None:
        raise IngestError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_state(path: pathlib.Path) -> dict:
    """The saved state, from the page itself or from a JSON copy of it."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        return json.loads(text)
    match = STATE_SCRIPT.search(text)
    if match is None:
        raise IngestError(
            f"{path} carries no <script id=\"wam-state\"> block. That is either not the review "
            "page or a copy saved by something that dropped it."
        )
    return json.loads(html.unescape(match.group(1)))


def split_keys(state: dict, audit: dict, tail: dict) -> tuple[dict, dict]:
    """``(mask_verdicts, tail_verdicts)``, refusing any key neither source names."""
    mask_keys = {f"{f['episode']}:{f['frame_index']}" for f in audit["frames"]}
    tail_keys = {f"{f['episode']}:{f['frame_index']}" for f in tail["frames"]}
    overlap = mask_keys & tail_keys
    if overlap:
        raise IngestError(
            "the same frame appears in both the apple sample and the area-tail sample "
            f"({sorted(overlap)[:3]}), so a verdict on it cannot be routed by key alone. "
            "The two sections ask different questions and the answer would be ambiguous."
        )

    mask: dict[str, str] = {}
    tail_out: dict[str, str] = {}
    unknown: list[str] = []
    for key, verdict in state.get("verdicts", {}).items():
        if key in mask_keys:
            mask[key] = verdict
        elif key in tail_keys:
            tail_out[key] = verdict
        else:
            unknown.append(key)
    if unknown:
        raise IngestError(
            f"{len(unknown)} verdict(s) name frames neither sample contains, first "
            f"{unknown[:3]}. Refusing rather than dropping them silently."
        )
    return mask, tail_out


def check_vocabulary(mask: dict, tail: dict, recorder) -> None:
    for key, verdict in mask.items():
        if verdict not in recorder.VERDICTS:
            raise IngestError(
                f"{key} carries {verdict!r}, which is not an apple-mask verdict "
                f"{recorder.VERDICTS}. The tail section's vocabulary is not this section's."
            )
    for key, verdict in tail.items():
        if verdict not in TAIL_VERDICTS:
            raise IngestError(
                f"{key} carries {verdict!r}, which is not an area-tail verdict {TAIL_VERDICTS}."
            )


def derive_sheets(mask: dict, audit: dict, recorder) -> tuple[dict[str, str], list[str], dict]:
    """Per-sheet defaults and per-tile exceptions, in the recorder's own argument shape."""
    mapping = recorder.sheet_index(audit["frames"])
    by_sheet: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for position, frame in enumerate(audit["frames"]):
        key = f"{frame['episode']}:{frame['frame_index']}"
        if key in mask:
            by_sheet[mapping[position]][key] = mask[key]

    defaults: dict[str, str] = {}
    exceptions: list[str] = []
    detail: dict[str, dict] = {}
    for sheet, verdicts in sorted(by_sheet.items()):
        tally = collections.Counter(verdicts.values())
        # Ties resolve to the first verdict in the recorder's own ordering, so the derivation is
        # deterministic rather than dependent on dict order.
        top = max(tally.items(), key=lambda kv: (kv[1], -recorder.VERDICTS.index(kv[0])))[0]
        defaults[sheet] = top
        odd = {k: v for k, v in verdicts.items() if v != top}
        for key, verdict in sorted(odd.items()):
            exceptions.append(f"{key}={verdict}:vom Prüfblatt abweichend, auf der Review-Seite gesetzt")
        detail[sheet] = {
            "default": top,
            "tally": dict(tally),
            "n_tiles": len(verdicts),
            "exceptions": odd,
        }
    return defaults, exceptions, detail


def tail_artifact(tail: dict, sample: dict, reviewer: str, state: dict, page: pathlib.Path) -> dict:
    by_key = {f"{f['episode']}:{f['frame_index']}": f for f in sample["frames"]}
    tally = collections.Counter(tail.values())
    frames = []
    for key in sorted(tail):
        frame = by_key[key]
        frames.append(
            {
                "episode": frame["episode"],
                "frame_index": frame["frame_index"],
                "verdict": tail[key],
                "recorded_fraction": frame["recorded_fraction"],
                "recomputed_fraction": frame["recomputed_fraction"],
                "mismatch": frame.get("mismatch", False),
                "sheet": frame["sheet"],
            }
        )
    return {
        "schema": "wam.pr08_area_tail_verdicts/1",
        "answers": (
            "T40_RULE_V13 §3.2's requirement that a bound_rationale state whether the frames above "
            "the bound were LOOKED AT and what they were."
        ),
        "writes_a_bound": False,
        "not_a_discharge": (
            "This records what a person saw. It commits no max_frame_fraction, signs no rule and "
            "licenses no clip. V13 stays an unsigned draft until a person signs §5."
        ),
        "vocabulary": {
            "arm": "a legitimately near-camera robot arm. A bound must NOT fire on this.",
            "table": "the mask has grounded on the tablecloth, background or plate.",
            "mixed": "arm AND scene in the same mask.",
            "undecidable": "not decidable from this frame.",
        },
        "established_by": reviewer,
        "correlated_observer": (
            "A model reading masks produced by a model-built pipeline is not this check. "
            "--reviewer records who looked; it cannot verify it."
        ),
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "source_page": str(page),
        "page_saved_at": state.get("saved_at"),
        "sample": {
            "artifact": "runs/pr08-area-tail-look/TAIL_SAMPLE.json",
            "threshold": sample["threshold"],
            "n_frames_in_sample": len(sample["frames"]),
            "n_frames_with_a_verdict": len(tail),
        },
        "tally": dict(tally),
        "frames": frames,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--page", type=pathlib.Path, required=True, help="the saved review page, or a .json copy of its state")
    ap.add_argument("--reviewer", required=True, help="who looked. Recorded, not verified.")
    ap.add_argument("--audit-dir", type=pathlib.Path, default=REPO_ROOT / "runs" / "pr08-mask-audit")
    ap.add_argument("--look-dir", type=pathlib.Path, default=REPO_ROOT / "runs" / "pr08-area-tail-look")
    ap.add_argument("--dry-run", action="store_true", help="derive and print, write nothing")
    args = ap.parse_args(argv)

    if not args.reviewer.strip():
        raise SystemExit("--reviewer is empty. Who looked is the whole point of the record.")

    recorder = load_recorder()
    try:
        state = read_state(args.page)
        audit = json.loads((args.audit_dir / "MASK_AUDIT.json").read_text())
        sample = json.loads((args.look_dir / "TAIL_SAMPLE.json").read_text())
        mask, tail = split_keys(state, audit, sample)
        check_vocabulary(mask, tail, recorder)
        defaults, exceptions, detail = derive_sheets(mask, audit, recorder)
    except IngestError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    print(f"page saved at      {state.get('saved_at')}")
    print(f"apple verdicts     {len(mask)}  over {len(defaults)} sheets, {len(exceptions)} exceptions")
    print(f"area-tail verdicts {len(tail)}  {dict(collections.Counter(tail.values()))}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    recorder_argv = ["--reviewer", args.reviewer, "--audit-dir", str(args.audit_dir), "--out",
                     str(args.audit_dir / "MASK_AUDIT_VERDICTS.json")]
    for sheet, verdict in sorted(defaults.items()):
        recorder_argv += ["--sheet", f"{sheet}={verdict}"]
    for item in exceptions:
        recorder_argv += ["--except", item]
    print()
    rc = recorder.main(recorder_argv)
    if rc != 0:
        return rc

    ingest = {
        "schema": "wam.pr08_review_page_ingest/1",
        "produced_by": "scripts/ingest_review_page.py",
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "established_by": args.reviewer,
        "source_page": str(args.page),
        "page_saved_at": state.get("saved_at"),
        "compression": (
            "The page holds ONE VERDICT PER TILE. The recorder takes a per-sheet default plus "
            "exceptions. per_tile below is the page's own record; per_sheet is what was derived "
            "from it and handed to the recorder. A reader who distrusts the derivation can "
            "recompute it from per_tile."
        ),
        "per_sheet": detail,
        "per_tile": dict(sorted(mask.items())),
        "area_tail_per_tile": dict(sorted(tail.items())),
    }
    ingest_path = args.audit_dir / "REVIEW_PAGE_INGEST.json"
    ingest_path.write_text(json.dumps(ingest, indent=2) + "\n")
    print(f"wrote {ingest_path}")

    tail_path = args.look_dir / "TAIL_VERDICTS.json"
    tail_path.write_text(
        json.dumps(tail_artifact(tail, sample, args.reviewer, state, args.page), indent=2) + "\n"
    )
    print(f"wrote {tail_path}")
    counts = collections.Counter(tail.values())
    print(f"  area tail         {dict(counts)} over {len(tail)}/{len(sample['frames'])} frames")
    print("\nBoth artifacts are evidence. Neither discharges a blocker or commits a bound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
