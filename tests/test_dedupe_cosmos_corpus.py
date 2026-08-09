"""Tests for ``scripts/dedupe_cosmos_corpus.py``.

The shipped T-041 corpus contained ``g1-dex3-graspsquare-dataset`` as an exact byte-copy of
``g1-dex3-blockstacking-dataset``, and four of the thirty pre-registered eval prompts were
byte-identical to clips in TRAIN. Every holdout check we own compares uuids, so nothing caught it;
the eval would have scored an adapter on clips it had memorised, in the direction of the registered
hypothesis.

So the properties under test here are not "the script runs". They are the four things whose failure
is silent:

1. a train clip that duplicates a **val** clip is gone, because that is what makes the holdout real;
2. a train-train duplicate leaves **exactly one** survivor, and which one is decided by the uuid
   rather than by the filesystem — a corpus that differs between machines is not a corpus AC-04 can
   trace a rollout back to;
3. **val is never touched**, in any file, by any rule;
4. ``sha256sum manifest.json`` reproduces ``MANIFEST_SHA256`` — 92b_register_corpus.sbatch refuses
   the corpus otherwise, after the 14 GB transfer.

These build synthetic corpora: the mp4s are byte blobs whose *content* is the whole point, and
nothing here decodes video.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import dedupe_cosmos_corpus as dcc  # noqa: E402


def make_root(tmp: Path, train: dict[str, bytes], val: dict[str, bytes]) -> Path:
    """A prepared, captioned corpus: uuid -> the bytes of its clip, per split.

    Writes every artifact the real corpus has — manifest, videos, per-clip caption directories,
    the SFT jsonl and its summary, and the decode report — because keeping them mutually
    consistent is most of what the script does, and a fixture that omits them would let a
    regression through.
    """
    root = tmp / "corpus"
    manifest = {
        "schema": "wam.cosmos_corpus/1",
        "seed": 0,
        "sources": {"src": {"root": "/raw/src", "camera_key": "(sole)", "kept": 0,
                            "dropped": {"too_short": 0, "too_long": 0, "missing_video": 0}}},
        "counts": {},
        "clips": {},
    }
    for split, clips in (("train", train), ("val", val)):
        (root / split / "videos").mkdir(parents=True)
        (root / split / "captions").mkdir(parents=True)
        entries, rows, decoded = [], [], []
        for uuid, blob in sorted(clips.items()):
            video = root / split / "videos" / f"{uuid}.mp4"
            video.write_bytes(blob)
            caps = root / split / "captions" / uuid
            caps.mkdir()
            (caps / "caption.json").write_text(json.dumps({"caption": uuid}))
            entries.append({
                "uuid": uuid, "source_id": uuid.split("_episode_")[0], "episode_index": 0,
                "src_path": f"/raw/src/{uuid}.mp4", "frames": 100, "fps": 30.0,
                "duration_s": 3.333, "width": 640, "height": 480, "task": "do the thing",
                "from_s": None, "to_s": None, "src_codec": "av1",
                "src_sha256": hashlib.sha256(blob).hexdigest(),
                "sha256": hashlib.sha256(blob).hexdigest(),
            })
            rows.append({"uuid": uuid, "duration": 3.333, "width": 640, "height": 480,
                         "vision_path": f"videos/{uuid}.mp4", "t2w_windows": []})
            decoded.append({"path": str(video), "ok": True, "decoded": 2, "declared": 100,
                            "error": "", "width": 640, "height": 480, "fps": 30.0})
        manifest["clips"][split] = entries
        manifest["counts"][split] = len(entries)
        jsonl = root / split / "video_dataset_file.jsonl"
        jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))
        jsonl.with_name(jsonl.name + ".summary.json").write_text(json.dumps(
            {"records_kept": len(rows), "records_with_caption_json": len(rows),
             "records_dropped": 0, "drops_by_reason": {}}, indent=2) + "\n")
        (root / split / "decode_report.json").write_text(json.dumps(
            {"cv2": "4.13.0", "checked": len(decoded), "failed": 0, "results": decoded},
            indent=2) + "\n")
    body = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (root / "manifest.json").write_text(body)
    (root / "MANIFEST_SHA256").write_text(hashlib.sha256(body.encode()).hexdigest() + "\n")
    return root


def uuids(root: Path, split: str) -> set[str]:
    return {p.stem for p in (root / split / "videos").glob("*.mp4")}


def fingerprint(root: Path, split: str) -> list[tuple[str, str]]:
    """Every file under one split, by relative path and content — the "was val touched" oracle."""
    base = root / split
    return sorted((str(p.relative_to(base)), hashlib.sha256(p.read_bytes()).hexdigest())
                  for p in base.rglob("*") if p.is_file())


# --- 1. contamination: the reason this exists ------------------------------------------------


def test_a_train_clip_duplicating_a_val_clip_is_deleted(tmp_path):
    root = make_root(
        tmp_path,
        train={"src_episode_000001_clip000": b"leaked", "src_episode_000002_clip000": b"own"},
        val={"held_episode_000001_clip000": b"leaked"},
    )
    assert dcc.main([str(root)]) == 0
    assert uuids(root, "train") == {"src_episode_000002_clip000"}
    assert uuids(root, "val") == {"held_episode_000001_clip000"}


def test_contamination_survives_being_named_something_else(tmp_path):
    """The defect in the shipped corpus: same bytes, different source_id, different episode number.

    A uuid comparison — which is what make_t041_eval_prompts.py and check_prompts_are_held_out
    both do — reports this pair as disjoint.
    """
    root = make_root(
        tmp_path,
        train={"graspsquare_episode_000077_clip000": b"same pixels"},
        val={"blockstacking_episode_000077_clip000": b"same pixels"},
    )
    dcc.main([str(root)])
    assert uuids(root, "train") == set()
    assert json.loads((root / "manifest.json").read_text())["counts"]["train"] == 0


def test_check_mode_reports_and_writes_nothing(tmp_path):
    root = make_root(
        tmp_path,
        train={"a_episode_000001_clip000": b"dup", "b_episode_000001_clip000": b"dup"},
        val={"v_episode_000001_clip000": b"unique"},
    )
    before = (root / "manifest.json").read_bytes()
    assert dcc.main([str(root), "--check"]) == 1
    assert uuids(root, "train") == {"a_episode_000001_clip000", "b_episode_000001_clip000"}
    assert (root / "manifest.json").read_bytes() == before


# --- 2. redundancy: one survivor, chosen by uuid ---------------------------------------------


def test_train_duplicates_leave_exactly_the_smallest_uuid(tmp_path):
    root = make_root(
        tmp_path,
        train={"zulu_episode_000001_clip000": b"dup", "alpha_episode_000001_clip000": b"dup",
               "mike_episode_000001_clip000": b"dup", "solo_episode_000001_clip000": b"other"},
        val={"v_episode_000001_clip000": b"unique"},
    )
    dcc.main([str(root)])
    assert uuids(root, "train") == {"alpha_episode_000001_clip000", "solo_episode_000001_clip000"}


def test_the_survivor_does_not_depend_on_directory_order(tmp_path):
    """Same clips, opposite creation order: the corpus must come out the same either way."""
    pairs = [("zulu_episode_000001_clip000", b"dup"), ("alpha_episode_000001_clip000", b"dup")]
    survivors = []
    for order in (pairs, list(reversed(pairs))):
        root = make_root(tmp_path / str(len(survivors)), train=dict(order),
                         val={"v_episode_000001_clip000": b"unique"})
        dcc.main([str(root)])
        survivors.append(uuids(root, "train"))
    assert survivors[0] == survivors[1] == {"alpha_episode_000001_clip000"}


def test_a_clip_that_is_both_contaminated_and_duplicated_is_not_kept_as_a_survivor(tmp_path):
    """Rule 2 must not "keep one" out of a group whose bytes are in val — that keeps the leak.

    ``alpha`` is the lexicographically smallest of the three, so a redundancy pass running first
    would elect it and delete only its two twins, leaving the val clip's bytes in train.
    """
    root = make_root(
        tmp_path,
        train={"alpha_episode_000001_clip000": b"leaked", "mike_episode_000001_clip000": b"leaked",
               "zulu_episode_000001_clip000": b"leaked"},
        val={"v_episode_000001_clip000": b"leaked"},
    )
    dcc.main([str(root)])
    assert uuids(root, "train") == set()


# --- 3. val is never touched -----------------------------------------------------------------


def test_val_is_bit_identical_afterwards(tmp_path):
    root = make_root(
        tmp_path,
        train={"a_episode_000001_clip000": b"dup", "b_episode_000001_clip000": b"dup",
               "c_episode_000001_clip000": b"leaked", "d_episode_000001_clip000": b"own"},
        val={"v_episode_000001_clip000": b"leaked", "w_episode_000001_clip000": b"held"},
    )
    before = fingerprint(root, "val")
    dcc.main([str(root)])
    assert fingerprint(root, "val") == before
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["counts"]["val"] == 2
    assert {c["uuid"] for c in manifest["clips"]["val"]} == {"v_episode_000001_clip000",
                                                             "w_episode_000001_clip000"}


def test_val_duplicating_val_is_left_alone(tmp_path):
    """Whatever we think of two identical eval prompts, shrinking val retires a registered n."""
    root = make_root(
        tmp_path,
        train={"a_episode_000001_clip000": b"own"},
        val={"v_episode_000001_clip000": b"twin", "w_episode_000001_clip000": b"twin"},
    )
    assert dcc.main([str(root)]) == 0
    assert len(uuids(root, "val")) == 2


# --- 4. the corpus stays internally consistent ------------------------------------------------


def test_every_per_clip_artifact_follows_the_deleted_clips(tmp_path):
    root = make_root(
        tmp_path,
        train={"a_episode_000001_clip000": b"dup", "b_episode_000001_clip000": b"dup",
               "c_episode_000001_clip000": b"own"},
        val={"v_episode_000001_clip000": b"unique"},
    )
    dcc.main([str(root)])
    survivors = {"a_episode_000001_clip000", "c_episode_000001_clip000"}
    jsonl = root / "train" / "video_dataset_file.jsonl"
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    assert {r["uuid"] for r in rows} == survivors
    assert {p.name for p in (root / "train" / "captions").iterdir()} == survivors
    report = json.loads((root / "train" / "decode_report.json").read_text())
    assert report["checked"] == 2 and report["failed"] == 0
    assert {Path(r["path"]).stem for r in report["results"]} == survivors
    summary = json.loads(jsonl.with_name(jsonl.name + ".summary.json").read_text())
    assert summary["records_kept"] == summary["records_with_caption_json"] == 2


def test_surviving_jsonl_lines_are_byte_preserved(tmp_path):
    """The captions in here are the captioner's only output; re-serialising changes token counts."""
    root = make_root(
        tmp_path,
        train={"a_episode_000001_clip000": b"dup", "b_episode_000001_clip000": b"dup"},
        val={"v_episode_000001_clip000": b"unique"},
    )
    jsonl = root / "train" / "video_dataset_file.jsonl"
    kept_line = next(l for l in jsonl.read_text().splitlines(keepends=True)
                     if "a_episode_000001_clip000" in l)
    dcc.main([str(root)])
    assert jsonl.read_text() == kept_line


def test_manifest_sha256_is_the_hash_of_the_bytes_on_disk(tmp_path):
    """`sha256sum manifest.json` must reproduce MANIFEST_SHA256 — 92b_register_corpus.sbatch runs
    exactly that command, and a digest taken over anything else fails on the cluster."""
    root = make_root(
        tmp_path,
        train={"a_episode_000001_clip000": b"dup", "b_episode_000001_clip000": b"dup"},
        val={"v_episode_000001_clip000": b"unique"},
    )
    dcc.main([str(root)])
    stamped = (root / "MANIFEST_SHA256").read_text().strip()
    assert stamped == hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    out = subprocess.run(["sha256sum", "manifest.json"], cwd=root,
                         capture_output=True, text=True, check=True).stdout.split()[0]
    assert out == stamped


def test_the_deletion_is_recorded_in_the_manifest(tmp_path):
    """Without it, ``sources.kept`` and ``counts`` disagree with no explanation in the artifact."""
    root = make_root(
        tmp_path,
        train={"a_episode_000001_clip000": b"dup", "b_episode_000001_clip000": b"dup",
               "c_episode_000001_clip000": b"leaked"},
        val={"v_episode_000001_clip000": b"leaked"},
    )
    dcc.main([str(root)])
    record = json.loads((root / "manifest.json").read_text())["dedupe"]
    assert record["removed"]["contamination"] == ["c_episode_000001_clip000"]
    assert record["removed"]["redundancy"] == ["b_episode_000001_clip000"]


# --- 5. idempotence and refusals ---------------------------------------------------------------


def test_a_second_run_is_a_no_op(tmp_path):
    root = make_root(
        tmp_path,
        train={"a_episode_000001_clip000": b"dup", "b_episode_000001_clip000": b"dup",
               "c_episode_000001_clip000": b"leaked"},
        val={"v_episode_000001_clip000": b"leaked"},
    )
    dcc.main([str(root)])
    after = {s: fingerprint(root, s) for s in ("train", "val")}
    manifest = (root / "manifest.json").read_bytes()
    assert dcc.main([str(root), "--check"]) == 0
    assert dcc.main([str(root)]) == 0
    assert {s: fingerprint(root, s) for s in ("train", "val")} == after
    assert (root / "manifest.json").read_bytes() == manifest


def test_a_manifest_entry_without_a_file_is_fatal(tmp_path):
    root = make_root(tmp_path, train={"a_episode_000001_clip000": b"own"},
                     val={"v_episode_000001_clip000": b"unique"})
    (root / "train" / "videos" / "a_episode_000001_clip000.mp4").unlink()
    with pytest.raises(SystemExit) as e:
        dcc.main([str(root)])
    assert "entries without a file" in str(e.value)


def test_a_file_without_a_manifest_entry_is_fatal(tmp_path):
    root = make_root(tmp_path, train={"a_episode_000001_clip000": b"own"},
                     val={"v_episode_000001_clip000": b"unique"})
    (root / "train" / "videos" / "stowaway_episode_000001_clip000.mp4").write_bytes(b"own")
    with pytest.raises(SystemExit) as e:
        dcc.main([str(root)])
    assert "files without an entry" in str(e.value)


def test_bytes_that_no_longer_match_the_recorded_hash_are_fatal(tmp_path):
    """Deleting clips by content is not something to do to a corpus whose content is unexplained."""
    root = make_root(tmp_path, train={"a_episode_000001_clip000": b"own"},
                     val={"v_episode_000001_clip000": b"unique"})
    (root / "train" / "videos" / "a_episode_000001_clip000.mp4").write_bytes(b"tampered")
    with pytest.raises(SystemExit) as e:
        dcc.main([str(root)])
    assert "no longer hash" in str(e.value)


def test_a_surviving_clip_missing_from_the_decode_report_is_fatal(tmp_path):
    root = make_root(
        tmp_path,
        train={"a_episode_000001_clip000": b"dup", "b_episode_000001_clip000": b"dup",
               "c_episode_000001_clip000": b"own"},
        val={"v_episode_000001_clip000": b"unique"},
    )
    report = root / "train" / "decode_report.json"
    data = json.loads(report.read_text())
    data["results"] = [r for r in data["results"] if "c_episode" not in r["path"]]
    report.write_text(json.dumps(data, indent=2) + "\n")
    with pytest.raises(SystemExit) as e:
        dcc.main([str(root)])
    assert "no record in" in str(e.value)
