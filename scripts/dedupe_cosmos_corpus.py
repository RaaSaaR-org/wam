#!/usr/bin/env python3
"""Delete byte-identical duplicate clips from the TRAIN split of a prepared Cosmos corpus.

WHY THIS EXISTS. ``g1-dex3-graspsquare-dataset`` is not a second task. It is an exact byte-copy of
``g1-dex3-blockstacking-dataset``: all 299 clips share a sha256 with their same-numbered twin, and
the LeRobot task string the manifest recorded for it reads "camera packaging" — a third dataset's
label on a second dataset's name over the first dataset's pixels. Corpus-wide that is 295 duplicate
pairs inside train, 3137 unique clips among 3432.

THE PART THAT MATTERS IS NOT THE WEIGHTING. Double-weighting one source is a bias; it is visible in
the manifest and it is arguable. The eval is not. Four of the thirty pre-registered eval prompts —
13% of the val split — are byte-identical to clips in TRAIN::

    val g1-dex3-blockstacking-dataset_episode_000077_clip000 == train …graspsquare…_000077_clip000
    val g1-dex3-blockstacking-dataset_episode_000126_clip000 == train …graspsquare…_000126_clip000
    val g1-dex3-graspsquare-dataset_episode_000224_clip000   == train …blockstacking…_000224_clip000
    val g1-dex3-graspsquare-dataset_episode_000239_clip000   == train …blockstacking…_000239_clip000

``make_t041_eval_prompts.py`` and ``check_prompts_are_held_out`` both compare **uuids**, so a clip
the adapter memorised passes every holdout check we own while wearing a different filename. The
bias runs toward the LoRA arm, which is the pre-registered hypothesis: exactly the direction where
a contaminated holdout is least likely to be questioned and most expensive to have believed.

THE FIX, AND ITS SHAPE. Two rules, train only, in this order:

1. **Contamination.** Delete every train clip whose sha256 equals the sha256 of any val clip. This
   is what makes the holdout real.
2. **Redundancy.** Among train clips that duplicate each other, keep exactly one — the
   lexicographically smallest uuid — and delete the rest. The tie-break is deterministic on
   purpose: a rule that resolved by filesystem order would give a different corpus on a different
   machine, and the corpus is the thing AC-04 traces a rollout back to.

VAL IS NOT TOUCHED. Not one file, not one manifest entry. Val stays at 30 clips so the eval keeps
n=30 and G0a keeps its registered >=15/30 threshold; a re-split would silently retire a
pre-registered number. It also means the re-sync to the cluster is deletions plus a manifest rather
than a 14 GB re-upload.

WHAT IS REGENERATED. Everything keyed per clip, or the corpus fails its own gates: the manifest's
``clips.train`` and ``counts``, ``train/video_dataset_file.jsonl`` (filtered, never re-captioned),
its ``.summary.json`` counts, ``train/decode_report.json``, ``train/captions/<uuid>/``, and
``MANIFEST_SHA256``. A decode report describing clips that no longer exist is the same class of
defect as a manifest describing clips that were never written.

    python3 scripts/dedupe_cosmos_corpus.py ~/wam-t041/cosmos-g1-embodiment --check
    python3 scripts/dedupe_cosmos_corpus.py ~/wam-t041/cosmos-g1-embodiment

Idempotent: a second run finds nothing to delete and leaves manifest.json byte-identical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

SPLITS = ("train", "val")


def _sha256(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def inventory(root: pathlib.Path, manifest: dict, split: str) -> dict[str, pathlib.Path]:
    """uuid -> mp4, having proved the manifest and the directory describe the same set of clips.

    Both directions are fatal and for the same reason: this script decides what to delete from the
    manifest's view of the corpus, so a file the manifest does not know about would survive a
    deletion pass invisibly, and a manifest entry with no file would be "deduplicated" against a
    hash it does not have.
    """
    videos = root / split / "videos"
    if not videos.is_dir():
        raise SystemExit(f"FATAL: {videos} missing — that is not a prepared corpus.")
    on_disk = {p.stem: p for p in sorted(videos.glob("*.mp4"))}
    in_manifest = {c["uuid"] for c in manifest["clips"][split]}
    orphan_entries = sorted(in_manifest - set(on_disk))
    orphan_files = sorted(set(on_disk) - in_manifest)
    if orphan_entries or orphan_files:
        raise SystemExit(
            f"FATAL: {split} manifest and {videos} disagree about what exists — "
            f"{len(orphan_entries)} entries without a file ({orphan_entries[:3]}), "
            f"{len(orphan_files)} files without an entry ({orphan_files[:3]}).\n"
            "       Deduplication rewrites both; it will not run against a corpus that is already "
            "inconsistent."
        )
    return on_disk


def hash_split(files: dict[str, pathlib.Path], jobs: int) -> dict[str, str]:
    """Hash every clip. Threaded because this is 14 GB of read, and it is all I/O."""
    uuids = list(files)
    with ThreadPoolExecutor(max_workers=max(jobs, 1)) as pool:
        digests = list(pool.map(lambda u: _sha256(files[u]), uuids))
    return dict(zip(uuids, digests))


def cross_check_manifest_hashes(manifest: dict, split: str, measured: dict[str, str]) -> None:
    """The manifest's per-clip sha256 is provenance; a file that no longer matches it is not ours.

    Re-hashing rather than reading ``clip["sha256"]`` is the whole basis of this script's decision,
    so the recorded value gets used as the thing to disagree with. A mismatch means the bytes
    changed after preparation — deleting clips by content is not something to do to a corpus whose
    content is already unexplained.
    """
    bad = [c["uuid"] for c in manifest["clips"][split] if measured[c["uuid"]] != c["sha256"]]
    if bad:
        raise SystemExit(
            f"FATAL: {len(bad)} {split} clips no longer hash to the sha256 the manifest recorded "
            f"({bad[:3]}). The corpus was modified after preparation; fix that first."
        )


def plan(train: dict[str, str], val: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Which train uuids each rule deletes, and what each one collides with.

    Returns ``(contaminated, redundant)``: uuid -> the val uuid it leaks, and uuid -> the train
    uuid kept in its place. Rule 1 runs first so that a clip which is both a val duplicate and a
    train duplicate is reported as the contamination it is, rather than as bookkeeping.
    """
    val_by_sha: dict[str, list[str]] = {}
    for uuid, sha in val.items():
        val_by_sha.setdefault(sha, []).append(uuid)

    contaminated = {u: min(val_by_sha[sha]) for u, sha in train.items() if sha in val_by_sha}

    groups: dict[str, list[str]] = {}
    for uuid, sha in train.items():
        if uuid not in contaminated:
            groups.setdefault(sha, []).append(uuid)
    redundant: dict[str, str] = {}
    for members in groups.values():
        if len(members) > 1:
            keep = min(members)
            redundant.update({u: keep for u in members if u != keep})
    return contaminated, redundant


def filter_jsonl(path: pathlib.Path, keep: set[str]) -> int:
    """Drop the deleted clips' records, byte-preserving every line that survives.

    Not regenerated from the manifest: the captions in here are the captioner's output and the
    only copy of it that the trainer reads. Re-serialising them would change whitespace, which
    changes the token count, which is a real bound in the recipe.
    """
    lines = [ln for ln in path.read_text().splitlines(keepends=True) if ln.strip()]
    kept = [ln for ln in lines if json.loads(ln)["uuid"] in keep]
    path.write_text("".join(kept))
    return len(lines) - len(kept)


def filter_summary(path: pathlib.Path, kept: int, dropped_with_caption: int) -> None:
    """Bring the converter's own tally in line with the file it describes.

    ``30_caption_corpus.sh`` prints this next to the jsonl it just built, and a summary claiming
    3432 records beside a 3133-line jsonl is the kind of disagreement that gets read as a captioning
    failure. The drop counters are left alone: they record why the *converter* dropped clips, and it
    dropped none — these clips are gone from its input entirely, which is what a re-run would see.
    """
    if not path.is_file():
        return
    summary = json.loads(path.read_text())
    summary["records_kept"] = kept
    if "records_with_caption_json" in summary:
        summary["records_with_caption_json"] -= dropped_with_caption
    path.write_text(json.dumps(summary, indent=2) + "\n")


def filter_decode_report(path: pathlib.Path, keep: set[str]) -> None:
    report = json.loads(path.read_text())
    results = [r for r in report["results"] if pathlib.Path(r["path"]).stem in keep]
    missing = keep - {pathlib.Path(r["path"]).stem for r in results}
    if missing:
        raise SystemExit(
            f"FATAL: {len(missing)} surviving clips have no record in {path} "
            f"({sorted(missing)[:3]}). Re-run scripts/verify_clip_decode.py; a decode gate that "
            "does not cover the corpus is not a gate."
        )
    report["results"] = results
    report["checked"] = len(results)
    path.write_text(json.dumps(report, indent=2) + "\n")


def write_manifest(root: pathlib.Path, manifest: dict) -> str:
    """Serialise, stamp, and prove the stamp is the one ``sha256sum manifest.json`` produces.

    The digest covers the BYTES ON DISK including the trailing newline — see the same note in
    prepare_cosmos_corpus.py. 92b_register_corpus.sbatch compares ``sha256sum manifest.json``
    against MANIFEST_SHA256 and refuses the corpus on a mismatch, so a stamp taken over anything
    else does not fail here, it fails on the cluster after a 14 GB transfer.
    """
    path = root / "manifest.json"
    body = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    path.write_text(body)
    digest = hashlib.sha256(body.encode()).hexdigest()
    (root / "MANIFEST_SHA256").write_text(digest + "\n")
    reread = hashlib.sha256(path.read_bytes()).hexdigest()
    if reread != digest:
        raise SystemExit(
            f"FATAL: wrote MANIFEST_SHA256={digest} but manifest.json on disk hashes to {reread}. "
            "Something re-encoded the file between write and read; the corpus would be rejected at "
            "registration."
        )
    return digest


def verify(root: pathlib.Path, manifest: dict, sha: dict[str, dict[str, str]]) -> None:
    """The post-conditions that make the deletion worth having done."""
    train = {c["uuid"] for c in manifest["clips"]["train"]}
    val = {c["uuid"] for c in manifest["clips"]["val"]}

    val_shas = {sha["val"][u] for u in val}
    leaked = sorted(u for u in train if sha["train"][u] in val_shas)
    if leaked:
        raise SystemExit(
            f"FATAL: {len(leaked)} train clips still share bytes with a val clip ({leaked[:3]}). "
            "The holdout is still not a holdout."
        )
    seen: dict[str, str] = {}
    for uuid in sorted(train):
        twin = seen.setdefault(sha["train"][uuid], uuid)
        if twin != uuid:
            raise SystemExit(f"FATAL: train still contains duplicate bytes: {twin} == {uuid}.")

    for split, expected in (("train", train), ("val", val)):
        on_disk = {p.stem for p in (root / split / "videos").glob("*.mp4")}
        if on_disk != expected:
            raise SystemExit(
                f"FATAL: after deletion {split}/videos holds {len(on_disk)} files for "
                f"{len(expected)} manifest entries."
            )
        if manifest["counts"][split] != len(expected):
            raise SystemExit(
                f"FATAL: manifest counts.{split}={manifest['counts'][split]} but the clip list has "
                f"{len(expected)} entries."
            )
        jsonl = root / split / "video_dataset_file.jsonl"
        if jsonl.is_file():
            rows = sum(1 for ln in jsonl.read_text().splitlines() if ln.strip())
            if rows != len(expected):
                raise SystemExit(f"FATAL: {jsonl} has {rows} records for {len(expected)} clips.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", type=pathlib.Path, help="the prepared corpus directory")
    ap.add_argument("--check", action="store_true",
                    help="report what would be deleted and exit non-zero if anything would; "
                         "write nothing")
    ap.add_argument("--jobs", type=int, default=8, help="parallel hash readers")
    args = ap.parse_args(argv)

    root: pathlib.Path = args.corpus.expanduser()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"FATAL: {manifest_path} missing.")
    manifest = json.loads(manifest_path.read_text())

    files = {s: inventory(root, manifest, s) for s in SPLITS}
    sha = {}
    for split in SPLITS:
        sha[split] = hash_split(files[split], args.jobs)
        cross_check_manifest_hashes(manifest, split, sha[split])
        print(f"{split}: {len(sha[split])} clips, {len(set(sha[split].values()))} unique sha256",
              file=sys.stderr)

    contaminated, redundant = plan(sha["train"], sha["val"])
    for uuid, twin in sorted(contaminated.items()):
        print(f"  contamination: train/{uuid} == val/{twin}", file=sys.stderr)
    print(f"rule 1 (contamination): {len(contaminated)} train clips\n"
          f"rule 2 (redundancy):    {len(redundant)} train clips", file=sys.stderr)

    doomed = set(contaminated) | set(redundant)
    # Structurally impossible — ``plan`` only ever looks at train uuids — and asserted anyway,
    # because "val was not touched" is the promise the whole preregistration rests on and it should
    # not depend on a reader tracing which dict came from where.
    stray = doomed & set(files["val"])
    if stray:
        raise SystemExit(f"FATAL: the plan would delete val clips ({sorted(stray)[:3]}). Refusing.")

    if not doomed:
        print("nothing to delete; train is already free of duplicates and of val's bytes",
              file=sys.stderr)
        verify(root, manifest, sha)
        stamped = (root / "MANIFEST_SHA256").read_text().strip()
        actual = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if stamped != actual:
            raise SystemExit(
                f"FATAL: MANIFEST_SHA256 says {stamped}, sha256sum manifest.json says {actual}. "
                "No clips need deleting, but this corpus would be rejected at registration."
            )
        return 0

    keep = set(files["train"]) - doomed
    print(f"train {len(files['train'])} -> {len(keep)}  (val {len(files['val'])} unchanged)",
          file=sys.stderr)
    if args.check:
        print("--check: nothing written", file=sys.stderr)
        return 1

    captions = root / "train" / "captions"
    dropped_with_caption = 0
    for uuid in sorted(doomed):
        files["train"][uuid].unlink()
        cap = captions / uuid
        if cap.is_dir():
            shutil.rmtree(cap)
            dropped_with_caption += 1

    jsonl = root / "train" / "video_dataset_file.jsonl"
    if jsonl.is_file():
        removed = filter_jsonl(jsonl, keep)
        print(f"dropped {removed} records from {jsonl.name}", file=sys.stderr)
        filter_summary(jsonl.with_name(jsonl.name + ".summary.json"), len(keep),
                       dropped_with_caption)
    report = root / "train" / "decode_report.json"
    if report.is_file():
        filter_decode_report(report, keep)

    manifest["clips"]["train"] = [c for c in manifest["clips"]["train"] if c["uuid"] in keep]
    manifest["counts"]["train"] = len(manifest["clips"]["train"])
    # Recorded, not merely done. ``sources`` still reports what the scan kept per source — that is
    # a true statement about the scan and rewriting it would make the manifest claim GraspSquare
    # was never read — so without this block the difference between those counts and ``counts``
    # has no explanation in the artifact a rollout is traced back to. Unioned with any earlier
    # run's record so a second pass extends the history instead of erasing it.
    record = manifest.get("dedupe") or {}
    removed = record.get("removed") or {}
    manifest["dedupe"] = {
        "tool": "scripts/dedupe_cosmos_corpus.py",
        "scope": "train only; val is never modified",
        "rules": {
            "contamination": "train clip whose sha256 matches any val clip's sha256",
            "redundancy": "among train clips sharing a sha256, keep the lexicographically "
                          "smallest uuid",
        },
        "removed": {
            "contamination": sorted(set(removed.get("contamination", [])) | set(contaminated)),
            "redundancy": sorted(set(removed.get("redundancy", [])) | set(redundant)),
        },
    }

    digest = write_manifest(root, manifest)
    for split in SPLITS:
        sha[split] = {u: h for u, h in sha[split].items() if u not in doomed}
    verify(root, manifest, sha)
    print(f"MANIFEST_SHA256={digest}", file=sys.stderr)
    print("verify with:  sha256sum manifest.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
