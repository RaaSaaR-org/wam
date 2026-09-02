"""The tolerance has to name a corpus, and the corpus it names is not the one the generator reads.

These tests exist because of a defect this repository shipped and caught with its own gate, and the
sequence is the point. ``--bind-source-manifest`` landed on 2026-09-02 to close
``docs/investigations/2026-08-28-two-generate-path-gaps-recorded-not-fixed.md`` §1, and it bound
``configs/transfer25/pr08_geom_tol.json`` to the AV1 apple tree — correctly, because every one of
the sixteen GEOM_TOL shards records that tree as its ``corpus``. But
``97_transfer25_restyle.sbatch`` restyles the **H.264 transcode**, since Cosmos-Transfer2.5 opens
clips with ``cv2.VideoCapture`` and the generation venv's cv2 cannot decode AV1. So the first
submission after the binding was added would have been refused by the binding.

The refusal was right. What makes the two trees interchangeable is not that one of them says
``lossless: true`` about itself — that is the assumption moved into a JSON field — but a
frame-by-frame comparison, and ``--accept-equivalent-manifest`` records a second digest only with
that proof re-verified beside it. Everything below is about what must NOT pass.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "measure_geom_tol.py"

EPISODES = [("episode_000000", 5), ("episode_000001", 7), ("episode_000002", 4)]


def _manifest(codec: str) -> dict:
    return {
        "video_key": "observation.images.ego_view",
        "resolution": [640, 480],
        "fps": 30,
        "source": {"repo_id": "x/y", "codecs": [codec], "materialized": "copy"},
        "episodes": [{"id": i, "frames": n, "video": f"videos/{i}.mp4"} for i, n in EPISODES],
    }


def _write_json(path: Path, obj) -> str:
    raw = (json.dumps(obj, indent=2) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _proof(source_manifest_sha: str) -> dict:
    return {
        "schema": "1.0.0",
        "complete": True,
        "generated_at": "2026-08-22T17:04:03+0200",
        "source_manifest_sha256": source_manifest_sha,
        "decoder": {"name": "pyav", "version": "18.0.0"},
        "encoder": {"codec": "libx264", "crf": "0"},
        "clips_total": len(EPISODES),
        "clips_proven_bit_exact": len(EPISODES),
        "clips": [
            {
                "id": i,
                "ok": True,
                "error": "",
                "compare_stride": 1,
                "frames_source": n,
                "frames_output": n,
                "frames_compared": n,
                "max_abs_delta": {"yuv420p": 0, "rgb24": 0},
            }
            for i, n in EPISODES
        ],
    }


@pytest.fixture()
def bound(tmp_path: Path):
    """A committed document already bound to the AV1 tree, plus the H.264 tree beside it."""
    av1 = tmp_path / "av1.manifest.json"
    av1_sha = _write_json(av1, _manifest("av1"))
    h264 = tmp_path / "h264.manifest.json"
    h264_sha = _write_json(h264, _manifest("h264"))

    out = tmp_path / "pr08_geom_tol.json"
    doc = {
        "spec_version": "1.3.0",
        "what_this_is": "PR-08 §4 step 2 — a fixture, not the committed artifact",
        "contract_fields": ["spec_version", "what_this_is", "contract_fields",
                            "measurement_fields", "segmenter"],
        "measurement_fields": ["geom_tol_px"],
        "segmenter": {"name": "fixture"},
        "geom_tol_px": 0.5,
        "gate_qualified": True,
        "frame_width": 640,
        "frame_height": 480,
        "fps": 30,
        "per_episode": [{"episode": i, "n_frames": n} for i, n in EPISODES],
        "source_manifest_sha256": av1_sha,
    }
    _write_json(out, doc)
    (out.parent / (out.name + ".sha256")).write_text(
        hashlib.sha256(out.read_bytes()).hexdigest() + "\n"
    )
    return {
        "out": out, "av1": av1, "av1_sha": av1_sha, "h264": h264, "h264_sha": h264_sha,
        "proof_body": _proof(av1_sha), "tmp": tmp_path,
    }


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


def _accept(bound, proof_body=None, **over) -> subprocess.CompletedProcess:
    proof = bound["tmp"] / "TRANSCODE_PROOF.json"
    _write_json(proof, bound["proof_body"] if proof_body is None else proof_body)
    argv = [
        "--out", str(bound["out"]),
        "--bind-source-manifest", str(over.get("bind", bound["av1"])),
        "--accept-equivalent-manifest", str(over.get("candidate", bound["h264"])),
    ]
    if over.get("with_proof", True):
        argv += ["--equivalence-proof", str(proof)]
    return _run(*argv)


def _unchanged(bound) -> bool:
    on_disk = hashlib.sha256(bound["out"].read_bytes()).hexdigest()
    sidecar = (bound["out"].parent / (bound["out"].name + ".sha256")).read_text().strip()
    return on_disk == sidecar


def test_a_proven_equivalent_tree_is_accepted_and_carries_its_proofs_digest(bound):
    """The whole point: a SECOND digest, admitted only with the evidence that earned it."""
    res = _accept(bound)
    assert res.returncode == 0, res.stderr
    doc = json.loads(bound["out"].read_text())
    (entry,) = doc["accepted_equivalent_manifests"]
    assert entry["sha256"] == bound["h264_sha"]
    assert entry["sha256"] != doc["source_manifest_sha256"]
    # The digest of the proof is recorded, because the sbatch re-checks the file beside SOURCE
    # against it. An equivalence whose evidence can be swapped out afterwards is not one.
    assert entry["proof"]["sha256"] == hashlib.sha256(
        (bound["tmp"] / "TRANSCODE_PROOF.json").read_bytes()
    ).hexdigest()
    assert entry["proof"]["verified_here"]["frames_compared"] == sum(n for _, n in EPISODES)
    assert entry["proof"]["verified_here"]["compare_stride"] == 1
    assert entry["proof"]["verified_here"]["max_abs_delta_over_every_clip_and_space"] == 0
    assert "accepted_equivalent_manifests" in doc["measurement_fields"]


def test_one_non_zero_pixel_delta_anywhere_refuses_the_whole_equivalence(bound):
    """A tolerance in PIXELS transfers between two encodings only if the pixels are the same.

    Not "close": the same. A max-abs delta of 1 in one channel of one clip is a different
    decode, and a tolerance of 0.48 px held over it is a number about other frames.
    """
    proof = copy.deepcopy(bound["proof_body"])
    proof["clips"][1]["max_abs_delta"]["rgb24"] = 1
    res = _accept(bound, proof_body=proof)
    assert res.returncode == 2
    assert "NON-ZERO pixel difference" in res.stderr
    assert _unchanged(bound) and "accepted_equivalent_manifests" not in bound["out"].read_text()


def test_a_proof_that_skips_a_clip_proves_nothing_about_the_corpus(bound):
    proof = copy.deepcopy(bound["proof_body"])
    proof["clips"].pop(0)
    res = _accept(bound, proof_body=proof)
    assert res.returncode == 2
    assert "compares 2 clips" in res.stderr
    assert _unchanged(bound)


def test_a_sampled_comparison_bounds_nothing_between_the_samples(bound):
    proof = copy.deepcopy(bound["proof_body"])
    proof["clips"][0]["compare_stride"] = 4
    res = _accept(bound, proof_body=proof)
    assert res.returncode == 2
    assert "stride" in res.stderr
    assert _unchanged(bound)


def test_a_partial_proof_is_refused_even_when_every_clip_it_reached_was_clean(bound):
    proof = copy.deepcopy(bound["proof_body"])
    proof["complete"] = False
    res = _accept(bound, proof_body=proof)
    assert res.returncode == 2
    assert "complete=False" in res.stderr
    assert _unchanged(bound)


def test_a_second_digest_without_its_proof_is_a_waiver_and_is_refused_as_one(bound):
    res = _accept(bound, with_proof=False)
    assert res.returncode == 2
    assert "needs --equivalence-proof" in res.stderr
    assert _unchanged(bound)


def test_accepting_the_bound_manifest_as_its_own_equivalent_is_refused(bound):
    res = _accept(bound, candidate=bound["av1"])
    assert res.returncode == 2
    assert "IS the bound manifest" in res.stderr
    assert _unchanged(bound)


def test_the_bound_manifest_is_re_verified_rather_than_trusted(bound):
    """Passing some other file as the thing the document is bound to must not quietly work."""
    other = bound["tmp"] / "other.manifest.json"
    _write_json(other, _manifest("vp9"))
    res = _accept(bound, bind=other)
    assert res.returncode == 2
    assert "is bound to" in res.stderr
    assert _unchanged(bound)


def test_a_candidate_that_enumerates_a_different_corpus_is_refused_before_the_proof_is_read(bound):
    """Ids are reused across trees; this is the collision the digest exists to survive."""
    man = _manifest("h264")
    man["episodes"].append({"id": "episode_000003", "frames": 9, "video": "v"})
    other = bound["tmp"] / "wider.manifest.json"
    _write_json(other, man)
    res = _accept(bound, candidate=other)
    assert res.returncode == 2
    assert "DIFFERENT corpora" in res.stderr
    assert _unchanged(bound)


def test_a_candidate_on_another_pixel_grid_is_refused(bound):
    man = _manifest("h264")
    man["resolution"] = [1280, 720]
    other = bound["tmp"] / "hd.manifest.json"
    _write_json(other, man)
    res = _accept(bound, candidate=other)
    assert res.returncode == 2
    assert "GEOM_TOL is in pixels" in res.stderr
    assert _unchanged(bound)


def test_an_unbound_document_has_nothing_for_an_equivalent_to_be_equivalent_to(bound):
    doc = json.loads(bound["out"].read_text())
    doc.pop("source_manifest_sha256")
    _write_json(bound["out"], doc)
    (bound["out"].parent / (bound["out"].name + ".sha256")).write_text(
        hashlib.sha256(bound["out"].read_bytes()).hexdigest() + "\n"
    )
    res = _accept(bound)
    assert res.returncode == 2
    assert "names no corpus" in res.stderr
    assert _unchanged(bound)


def test_a_disqualified_tolerance_is_not_given_a_second_corpus_either(bound):
    doc = json.loads(bound["out"].read_text())
    doc["gate_qualified"] = False
    _write_json(bound["out"], doc)
    (bound["out"].parent / (bound["out"].name + ".sha256")).write_text(
        hashlib.sha256(bound["out"].read_bytes()).hexdigest() + "\n"
    )
    res = _accept(bound)
    assert res.returncode == 2
    assert _unchanged(bound)


def test_a_proof_naming_a_source_that_is_not_the_bound_one_demands_that_file(bound):
    """The one link established by inspection is required to be inspectable.

    The real proof was produced on the workstation and names the workstation's copy of the AV1
    manifest, which is not byte-identical to the cluster's. That is allowed — but only if the file
    is handed over so the difference can be measured, and it is recorded field by field.
    """
    proof = copy.deepcopy(bound["proof_body"])
    proof["source_manifest_sha256"] = "0" * 64
    res = _accept(bound, proof_body=proof)
    assert res.returncode == 2
    assert "--equivalence-proof-source" in res.stderr
    assert _unchanged(bound)

    # Handed over, checked, and the difference recorded rather than asserted away.
    twin = _manifest("av1")
    twin["source"]["materialized"] = "symlink"
    twin_p = bound["tmp"] / "workstation.manifest.json"
    proof["source_manifest_sha256"] = _write_json(twin_p, twin)
    proof_p = bound["tmp"] / "TRANSCODE_PROOF.json"
    _write_json(proof_p, proof)
    res = _run(
        "--out", str(bound["out"]),
        "--bind-source-manifest", str(bound["av1"]),
        "--accept-equivalent-manifest", str(bound["h264"]),
        "--equivalence-proof", str(proof_p),
        "--equivalence-proof-source", str(twin_p),
    )
    assert res.returncode == 0, res.stderr
    entry = json.loads(bound["out"].read_text())["accepted_equivalent_manifests"][0]
    src = entry["proof_source_manifest"]
    assert src["equals_bound_manifest"] is False
    assert list(src["differs_from_bound_manifest_in"]) == ["source.materialized"]
    assert src["differs_from_bound_manifest_in"]["source.materialized"] == ["copy", "symlink"]


def test_the_proofs_source_is_refused_when_its_digest_is_not_the_one_the_proof_names(bound):
    proof = copy.deepcopy(bound["proof_body"])
    proof["source_manifest_sha256"] = "0" * 64
    decoy = bound["tmp"] / "decoy.manifest.json"
    _write_json(decoy, _manifest("av1"))
    proof_p = bound["tmp"] / "TRANSCODE_PROOF.json"
    _write_json(proof_p, proof)
    res = _run(
        "--out", str(bound["out"]),
        "--bind-source-manifest", str(bound["av1"]),
        "--accept-equivalent-manifest", str(bound["h264"]),
        "--equivalence-proof", str(proof_p),
        "--equivalence-proof-source", str(decoy),
    )
    assert res.returncode == 2
    assert "as its source" in res.stderr
    assert _unchanged(bound)


def test_the_mode_refuses_to_ride_along_with_a_measurement(bound):
    """It measures nothing. Combined with --merge it would look like the merge had read a proof."""
    res = _run(
        "--out", str(bound["out"]),
        "--merge", str(bound["tmp"]),
        "--bind-source-manifest", str(bound["av1"]),
        "--accept-equivalent-manifest", str(bound["h264"]),
        "--equivalence-proof", str(bound["av1"]),
    )
    assert res.returncode != 0
    assert "names a different job" in (res.stderr + res.stdout)
    assert _unchanged(bound)
