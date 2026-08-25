#!/usr/bin/env python3
"""Record a person's verdicts on the PR-08 §4 apple-mask contact sheets.

WHY THIS EXISTS
---------------
``apple_sam2.GATE_QUALIFICATION_BLOCKERS``' first entry discharges on *"a human looking at a
sample of overlaid masks spanning the corpus (occluded frames, apple-out-of-frame frames, and the
grasp)"*, and its second entry opens *"Discharged by the same evidence as blocker 1, plus ..."*
with that "plus" already supplied. So one look discharges two blockers and nothing else discharges
either -- see ``PR-08-NOTE-2026-08-25-the-mujoco-iou-cannot-discharge-blocker-1.md``.

``audit_apple_masks.py`` asks the reviewer to fill ``OBSERVATIONS.template.json`` by hand. That is
**382 JSON entries**, which is a real obstacle to the one act no session can perform, and an
obstacle that produces no evidence. This tool takes the same verdicts in the unit a person
actually reviews in.

THE CONTACT SHEET IS THE REVIEW UNIT, AND THAT IS NOT A SHORTCUT
---------------------------------------------------------------
A sheet is 12 captioned tiles. Somebody who looks at a sheet and says *"all of these are the
apple except this one, which is the plate"* has made a judgement about **each of the twelve** --
that is what looking at a contact sheet is. So this accepts a per-sheet default plus per-frame
exceptions, and expands it.

What it must not do, and does not:

- **It never marks a frame nobody saw.** ``looked_at`` is set only on frames belonging to a sheet
  the reviewer named on the command line. Every other frame keeps ``looked_at: false`` and
  ``verdict: null``, and the artifact records the coverage as a fraction.
- **It never hides that the verdicts were given per sheet.** ``review_method`` says so, and each
  frame carries ``verdict_source`` of ``sheet_default`` or ``explicit_exception``. A reader can
  tell the two apart, which they could not if this wrote 382 identical-looking entries.
- **It does not decide anything.** No blocker is discharged here; that is an edit to the blocker
  tuple, with this artifact as the evidence.

THE MAPPING FROM FRAME TO SHEET IS RECONSTRUCTED, AND THEN CHECKED
-----------------------------------------------------------------
``audit_apple_masks.py`` chunks tiles per stratum in record order, 12 to a sheet. This recomputes
that and then **verifies every sheet it derives actually exists on disk**, and that it derived
every stratum sheet that does. A silent off-by-one here would attach a person's verdict to
frames they never saw, which is worse than having no verdict.

The ``flagged-NN`` sheets are deliberately NOT usable as a review unit: they are a cross-cutting
triage view whose tiles also appear on their own stratum sheets, so a default over them would
write verdicts twice under two different justifications.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import pathlib
import sys

SHEET_TILES = (
    12  # audit_apple_masks.py's --sheet-tiles default; verified against the sheets on disk
)

VERDICTS = ("apple", "partial", "wrong_object", "no_mask", "undecidable")

#: The three things blocker 1 names by name. A sample that misses one is not the sample it asks
#: for, and the artifact says so rather than letting a reader assume coverage.
BLOCKER_1_STRATA = {
    "occluded": "occluded frames",
    "min_visibility": "apple-out-of-frame frames",
    "grasp": "the grasp",
}

CORRELATED_OBSERVER = (
    "A model reading masks produced by a model-built pipeline is not the human check this "
    "blocker asks for. --reviewer records who looked; it cannot verify it."
)


def sheet_index(frames: list[dict]) -> dict[int, str]:
    """``frames`` position -> the stratum sheet whose tiles include it."""
    by_stratum: dict[str, list[int]] = collections.OrderedDict()
    for i, f in enumerate(frames):
        by_stratum.setdefault(f["stratum"], []).append(i)
    out: dict[int, str] = {}
    for stratum, positions in by_stratum.items():
        for n, pos in enumerate(positions):
            out[pos] = f"{stratum}-{n // SHEET_TILES:02d}"
    return out


def verify_mapping(derived: dict[int, str], sheets_dir: pathlib.Path) -> None:
    """Refuse unless the reconstruction matches the PNGs that exist.

    The failure this guards is silent and severe: an off-by-one attaches a reviewer's verdict to
    tiles they never saw.
    """
    on_disk = {p.stem for p in sheets_dir.glob("*.png")}
    if not on_disk:
        raise SystemExit(f"FATAL: no sheets in {sheets_dir}")
    want = set(derived.values())
    stratum_on_disk = {s for s in on_disk if not s.startswith("flagged-")}
    if want != stratum_on_disk:
        raise SystemExit(
            "FATAL: the frame->sheet reconstruction disagrees with the sheets on disk.\n"
            f"  derived but absent: {sorted(want - stratum_on_disk)}\n"
            f"  present but not derived: {sorted(stratum_on_disk - want)}\n"
            "Do not record verdicts against a mapping that does not hold: it would attach a "
            "person's judgement to frames they never looked at."
        )


def parse_exception(raw: str) -> tuple[str, int, str, str]:
    """``episode_000094:129=wrong_object`` or ``...=wrong_object:the plate``."""
    try:
        where, _, what = raw.partition("=")
        episode, _, frame = where.partition(":")
        verdict, _, note = what.partition(":")
        idx = int(frame)
    except ValueError as exc:  # pragma: no cover - argparse surfaces the message
        raise SystemExit(f"--except {raw!r}: expected episode:frame=verdict[:note] ({exc})")
    if verdict not in VERDICTS:
        raise SystemExit(f"--except {raw!r}: verdict must be one of {VERDICTS}")
    return episode.strip(), idx, verdict, note.strip()


def build(
    template: dict,
    *,
    reviewer: str,
    reviewed: dict[str, str],
    exceptions: list[tuple[str, int, str, str]],
    sheets_dir: pathlib.Path,
) -> dict:
    frames = [dict(f) for f in template["frames"]]
    mapping = sheet_index(frames)
    verify_mapping(mapping, sheets_dir)

    unknown = sorted(set(reviewed) - set(mapping.values()))
    if unknown:
        raise SystemExit(
            f"FATAL: --sheet named {unknown}, which are not stratum sheets. "
            "flagged-NN sheets are a cross-cutting triage view whose tiles appear on their own "
            "stratum sheets; a default over them would write each verdict twice."
        )

    for i, frame in enumerate(frames):
        sheet = mapping[i]
        frame["sheet"] = sheet
        if sheet in reviewed:
            frame["looked_at"] = True
            frame["verdict"] = reviewed[sheet]
            frame["verdict_source"] = "sheet_default"
        else:
            frame["looked_at"] = False
            frame["verdict"] = None
            frame["verdict_source"] = None

    by_key = {(f["episode"], int(f["frame_index"])): f for f in frames}
    applied = []
    for episode, idx, verdict, note in exceptions:
        frame = by_key.get((episode, idx))
        if frame is None:
            raise SystemExit(f"FATAL: --except names {episode} f{idx:05d}, not an audited frame.")
        if frame["sheet"] not in reviewed:
            raise SystemExit(
                f"FATAL: --except names {episode} f{idx:05d} on sheet {frame['sheet']}, which was "
                "not passed with --sheet. An exception to a sheet nobody declared reviewing is a "
                "verdict on a frame nobody looked at."
            )
        frame["verdict"] = verdict
        frame["observed"] = note
        frame["verdict_source"] = "explicit_exception"
        applied.append({"episode": episode, "frame_index": idx, "verdict": verdict, "note": note})

    seen = [f for f in frames if f["looked_at"]]
    tally = collections.Counter(f["verdict"] for f in seen)
    missing = sorted(
        k for k, v in BLOCKER_1_STRATA.items() if not any(s.startswith(f"{k}-") for s in reviewed)
    )

    return {
        "schema": template["schema"],
        "step": template["step"],
        "established_by": reviewer,
        "established_by_note": CORRELATED_OBSERVER,
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "review_method": (
            "contact-sheet review: a per-sheet default verdict over its 12 captioned tiles, plus "
            "per-frame exceptions. Each frame records verdict_source so sheet_default and "
            "explicit_exception are distinguishable."
        ),
        "recorded_by_tool": "scripts/record_mask_audit_verdicts.py",
        "sheets_reviewed": dict(sorted(reviewed.items())),
        "exceptions_applied": applied,
        "coverage": {
            "sheets_reviewed": len(reviewed),
            "sheets_total": len(set(mapping.values())),
            "frames_reviewed": len(seen),
            "frames_total": len(frames),
            "frames_reviewed_fraction": len(seen) / float(len(frames)),
        },
        # Blocker 1 names three things. Recorded rather than assumed, in both directions.
        "blocker_1_named_strata": {
            "required": BLOCKER_1_STRATA,
            "covered": sorted(set(BLOCKER_1_STRATA) - set(missing)),
            "not_covered": missing,
            "sample_spans_what_blocker_1_names": not missing,
        },
        "verdict_tally": dict(tally),
        "discharges_no_blocker": (
            "This is evidence, not a discharge. Discharging blocker 1 is an edit moving its "
            "wording into GATE_QUALIFICATION_DISCHARGED with this artifact cited beside it."
        ),
        "contact_sheets": template.get("contact_sheets", []),
        "verdict_values": list(VERDICTS),
        "frames": frames,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--audit-dir", type=pathlib.Path, default=pathlib.Path("runs/pr08-mask-audit"))
    ap.add_argument(
        "--reviewer",
        required=True,
        help="who looked. Written to established_by; see the correlated-observer note.",
    )
    ap.add_argument(
        "--sheet",
        action="append",
        default=[],
        metavar="NAME=VERDICT",
        help=f"a sheet reviewed and its default verdict, e.g. grasp-00=apple. "
        f"Verdicts: {'/'.join(VERDICTS)}",
    )
    ap.add_argument(
        "--except",
        dest="exceptions",
        action="append",
        default=[],
        metavar="EP:FRAME=VERDICT[:NOTE]",
        help="a tile on a reviewed sheet that differs from that sheet's default",
    )
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)

    if not args.sheet:
        raise SystemExit(
            "--sheet is required: nothing was reviewed, so there is nothing to record."
        )

    reviewed: dict[str, str] = {}
    for item in args.sheet:
        name, _, verdict = item.partition("=")
        if verdict not in VERDICTS:
            raise SystemExit(f"--sheet {item!r}: verdict must be one of {VERDICTS}")
        reviewed[name.strip()] = verdict

    template_path = args.audit_dir / "OBSERVATIONS.template.json"
    if not template_path.exists():
        raise SystemExit(f"FATAL: {template_path} does not exist.")

    out = build(
        json.loads(template_path.read_text()),
        reviewer=args.reviewer,
        reviewed=reviewed,
        exceptions=[parse_exception(e) for e in args.exceptions],
        sheets_dir=args.audit_dir / "sheets",
    )

    dest = args.out or (args.audit_dir / "OBSERVATIONS.json")
    dest.write_text(json.dumps(out, indent=2) + "\n")

    cov = out["coverage"]
    print(f"wrote {dest}")
    print(f"  reviewer          {out['established_by']}")
    print(f"  sheets reviewed   {cov['sheets_reviewed']}/{cov['sheets_total']}")
    print(
        f"  frames with a verdict {cov['frames_reviewed']}/{cov['frames_total']} "
        f"({cov['frames_reviewed_fraction']:.1%})"
    )
    print(f"  tally             {out['verdict_tally']}")
    b1 = out["blocker_1_named_strata"]
    if b1["sample_spans_what_blocker_1_names"]:
        print("  blocker 1 strata  all three covered (occluded, apple-out-of-frame, grasp)")
    else:
        print(
            f"  blocker 1 strata  NOT COVERED: {b1['not_covered']} — the sample does not span "
            "what blocker 1 names",
            file=sys.stderr,
        )
    print("\nThis is evidence. It discharges nothing; that edit is a person's, with this cited.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
