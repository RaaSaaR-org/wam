#!/usr/bin/env python3
"""Compare a blind second opinion against the recorded human verdicts, tile by tile.

WHY THIS EXISTS
---------------
``runs/pr08-mask-audit/MASK_AUDIT_VERDICTS.json`` carries one person's judgement on 382 overlaid
masks, and blockers 1 and 2 are supposed to discharge on it. A record that nothing ever checks is
indistinguishable from a record that was mis-clicked, and on 2026-08-26 the project owner asked
for exactly that check.

**A MODEL IS NOT A SECOND HUMAN.** ``MASK_AUDIT.json``'s own correlated-observer warning applies in
full: a model reading masks produced by a model-built pipeline can reproduce the same misreading on
both sides. So this comparison is asymmetric on purpose, and the asymmetry is the whole design:

- where the two agree, **nothing is established**. Agreement between a person and a correlated
  observer is not corroboration, and this artifact never claims it is;
- where they disagree, **something is worth a second look by the person**. A disagreement cannot
  say which side is right; it can only say that one of them is wrong, which is a question a human
  can then answer cheaply because the tile is named.

That is why the output is a QUESTION LIST, not a score. There is no accuracy number in it, because
an accuracy number would imply one of the two sides is ground truth and neither is.

WHAT IT WILL NOT DO
-------------------
It does not edit the human record. It does not discharge a blocker. It does not write a bound. A
disagreement it surfaces is resolved by a person looking again, and by nothing else.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Verdicts that mean "the mask is on the apple", against those that mean it is not. A disagreement
#: WITHIN a class is a severity difference; a disagreement ACROSS it is a disagreement about what is
#: in the picture, and the two must not be counted together.
ON_THE_APPLE = {"apple", "partial"}
NOT_THE_APPLE = {"wrong_object", "no_mask"}
UNDECIDED = {"undecidable"}

#: The area-tail section's vocabulary, split the same way: does the mask contain scene or not.
TAIL_CLEAN = {"arm"}
TAIL_CONTAMINATED = {"table", "mixed"}


class CompareError(RuntimeError):
    """Refuse loudly rather than compare two things that are not comparable."""


def classify(verdict: str) -> str:
    if verdict in ON_THE_APPLE or verdict in TAIL_CLEAN:
        return "clean"
    if verdict in NOT_THE_APPLE or verdict in TAIL_CONTAMINATED:
        return "not_clean"
    if verdict in UNDECIDED:
        return "undecided"
    raise CompareError(f"{verdict!r} belongs to neither section's vocabulary")


def load_opinions(directory: pathlib.Path) -> dict[str, dict]:
    """Every ``group-*.json`` a blind reviewer wrote, keyed by tile."""
    tiles: dict[str, dict] = {}
    files = sorted(directory.glob("group-*.json"))
    if not files:
        raise CompareError(f"no group-*.json under {directory}")
    for path in files:
        payload = json.loads(path.read_text())
        for tile in payload["tiles"]:
            key = tile["key"]
            if key in tiles:
                raise CompareError(
                    f"{key} was judged by two groups ({tiles[key]['group']} and "
                    f"{payload['group']}). Overlapping assignments make the count wrong."
                )
            tiles[key] = {
                "verdict": tile["verdict"],
                "confidence": tile.get("confidence"),
                "saw": tile.get("saw", ""),
                "sheet": tile.get("sheet"),
                "group": payload["group"],
            }
    return tiles


def load_human(audit_dir: pathlib.Path, look_dir: pathlib.Path) -> dict[str, dict]:
    """The recorded verdicts, from both sections."""
    human: dict[str, dict] = {}
    observations = audit_dir / "MASK_AUDIT_VERDICTS.json"
    if observations.is_file():
        payload = json.loads(observations.read_text())
        for frame in payload["frames"]:
            if not frame.get("looked_at"):
                continue
            key = f"{frame['episode']}:{frame['frame_index']}"
            human[key] = {
                "verdict": frame["verdict"],
                "section": "mask",
                "source": frame.get("verdict_source"),
                "sheet": frame.get("sheet"),
            }
    tail = look_dir / "TAIL_VERDICTS.json"
    if tail.is_file():
        payload = json.loads(tail.read_text())
        for frame in payload["frames"]:
            key = f"{frame['episode']}:{frame['frame_index']}"
            human[key] = {
                "verdict": frame["verdict"],
                "section": "tail",
                "source": "review_page",
                "sheet": frame.get("sheet"),
            }
    if not human:
        raise CompareError(
            f"neither {observations} nor {tail} exists, so there is no recorded verdict to check"
        )
    return human


def compare(human: dict[str, dict], opinion: dict[str, dict]) -> dict:
    both = sorted(set(human) & set(opinion))
    disagreements = []
    severity_only = []
    for key in both:
        h, o = human[key]["verdict"], opinion[key]["verdict"]
        if h == o:
            continue
        entry = {
            "key": key,
            "sheet": human[key].get("sheet") or opinion[key].get("sheet"),
            "section": human[key]["section"],
            "recorded": h,
            "second_opinion": o,
            "second_opinion_confidence": opinion[key]["confidence"],
            "second_opinion_saw": opinion[key]["saw"],
        }
        if classify(h) == classify(o):
            severity_only.append(entry)
        else:
            disagreements.append(entry)

    by_sheet: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for entry in disagreements:
        by_sheet[str(entry["sheet"])][f"{entry['recorded']}->{entry['second_opinion']}"] += 1

    return {
        "n_tiles_compared": len(both),
        "n_recorded_not_reviewed_blind": len(set(human) - set(opinion)),
        "n_reviewed_blind_not_recorded": len(set(opinion) - set(human)),
        "n_identical": len(both) - len(disagreements) - len(severity_only),
        "n_severity_only": len(severity_only),
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
        "severity_only": severity_only,
        "disagreements_by_sheet": {k: dict(v) for k, v in sorted(by_sheet.items())},
    }


def build(human: dict, opinion: dict, groups: pathlib.Path) -> dict:
    result = compare(human, opinion)
    return {
        "schema": "wam.pr08_second_opinion/1",
        "produced_by": "scripts/compare_second_opinion.py",
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "what_this_is": (
            "A blind model re-read of the same contact-sheet tiles, compared against the recorded "
            "human verdicts, requested by the project owner on 2026-08-26."
        ),
        "what_agreement_establishes": (
            "NOTHING. The second reader is a correlated observer — a model reading masks produced "
            "by a model-built pipeline — and MASK_AUDIT.json warns that such a reader can "
            "reproduce the same misreading on both sides. Agreement is not corroboration and no "
            "accuracy figure is computed here, because neither side is ground truth."
        ),
        "what_disagreement_establishes": (
            "That one of the two is wrong on a NAMED tile, which a person can then settle by "
            "looking at that tile. That is the entire product of this comparison."
        ),
        "not_a_discharge": (
            "This discharges no blocker, edits no recorded verdict and writes no bound."
        ),
        "blind": (
            "The second readers were instructed not to open MASK_AUDIT_VERDICTS.json, "
            "REVIEW_PAGE_INGEST.json, TAIL_VERDICTS.json or the review page. That instruction is "
            "recorded, not verified."
        ),
        "opinion_source": str(groups),
        "groups": sorted({str(t["group"]) for t in opinion.values()}),
        **result,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--opinions", type=pathlib.Path, required=True, help="directory of group-*.json")
    ap.add_argument("--audit-dir", type=pathlib.Path, default=REPO_ROOT / "runs" / "pr08-mask-audit")
    ap.add_argument(
        "--look-dir", type=pathlib.Path, default=REPO_ROOT / "runs" / "pr08-area-tail-look"
    )
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)

    try:
        opinion = load_opinions(args.opinions)
        human = load_human(args.audit_dir, args.look_dir)
        artifact = build(human, opinion, args.opinions)
    except CompareError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    dest = args.out or (args.audit_dir / "SECOND_OPINION.json")
    dest.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {dest}")
    print(f"  tiles compared        {artifact['n_tiles_compared']}")
    print(f"  identical             {artifact['n_identical']}")
    print(f"  severity only         {artifact['n_severity_only']}")
    print(f"  REAL DISAGREEMENTS    {artifact['n_disagreements']}")
    for sheet, counts in artifact["disagreements_by_sheet"].items():
        print(f"    {sheet:22s} {counts}")
    if artifact["n_recorded_not_reviewed_blind"]:
        print(f"  not re-read blind     {artifact['n_recorded_not_reviewed_blind']}")
    print(
        "\nAgreement establishes nothing — the second reader is a correlated observer. "
        "Only the disagreements are a product, and only a person can settle them."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
